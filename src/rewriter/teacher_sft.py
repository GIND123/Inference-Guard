"""Rewriter SFT set from a local teacher LLM.

Replaces regex-heuristic SFT targets with rewrites from a stronger open model.
The heuristic targets taught the student the heuristic's bugs: its location
regex turns "I am moving from Denver, CO to Seattle, WA to start as a software
engineer at Amazon" into "I am moving locally."

What goes in
    Train-split comments that a SynthPAI human annotator marked inferable
    (``reviews.human.<attr>.estimate`` non-empty) for age, location,
    occupation or education. The input file must hash to the train entry of
    data/manifests/split_manifest.json; val/test are refused.

How a teacher rewrite is accepted -- none of these use our own risk model or
the held-out attacker, so selection is independent of how we evaluate:
    1. non-empty and changed (whitespace-normalised);
    2. not degenerate under src.evaluation.utility.check_degeneracy;
    3. no verbatim leak: no part of the author's true value (profile) or of
       the annotator's estimate for an inferable attribute appears as a whole
       token run in the rewrite. A floor, not a privacy guarantee;
    4. NLI (real cross-encoder): neither direction's contradiction
       probability exceeds MAX_CONTRADICTION, so the rewrite does not change
       what the comment says.
    5. The rewrite still says the same thing: the original entails it
       (forward entailment >= MIN_FORWARD_ENTAILMENT) or it keeps most of the
       wording (token cosine >= MIN_TOKEN_COSINE). This stops a teacher that
       answers the comment instead of rewriting it.

    Forward entailment alone was the first version of rule 4, and the pilot
    showed it is the wrong test for a privacy rewrite: generalising
    "More women geologists" to "More women in my job" is neutral, not
    entailed (forward 0.0021, contradiction 0.0001), because generalising
    removes the very information entailment looks for. Real meaning changes
    show up as contradiction instead ("sutures" -> "threads": 0.97).

The held-out adversary (Phi-4-mini) is refused as a teacher. It must never
generate, select or filter training data.

What comes out (in --out-dir)
    <split>.jsonl             SFT rows: {"messages": [...], "enable_thinking": false,
                              "metadata": {...}}. The messages are exactly what
                              inference serves (src.rewriter.prompting) plus the
                              target as the final assistant turn. About
                              IDENTITY_RATIO of rows are identity no-ops drawn
                              from comments with nothing inferable.
    candidates_<split>.jsonl  every teacher output with its verdict, for error analysis
    manifest_<split>.json     provenance and counts

Usage:
    python -m src.rewriter.teacher_sft --split train --device cuda --out-dir data/sft/teacher_v1
    python -m src.rewriter.teacher_sft --split train --backend mlx --out-dir data/sft/teacher_v1  # Apple silicon
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.evaluation.utility import check_degeneracy, compute_fallback_token_cosine
from src.rewriter.generate_training_data import SYSTEM_PROMPT
from src.rewriter.prompting import build_rewrite_messages

logger = logging.getLogger(__name__)

# Qwen3-4B was the first choice and returned 29 of 40 sampled comments
# unchanged in the pilot, even when told what a reader had inferred.
TEACHERS = {
    "hf": "Qwen/Qwen2.5-7B-Instruct",
    "mlx": "mlx-community/Qwen2.5-7B-Instruct-4bit",  # same model, 4-bit MLX conversion for Apple silicon
}
NLI_DEFAULT = "cross-encoder/nli-deberta-v3-large"
# Never a teacher, in any conversion: the held-out adversary family.
HELD_OUT_MARKERS = ("phi-4",)

ATTRIBUTES: Tuple[str, ...] = ("age", "location", "occupation", "education")
# SynthPAI field names, in both profile and reviews.human.
_FIELD = {"age": "age", "location": "city_country", "occupation": "occupation", "education": "education"}

# First-cut thresholds, not fitted. Reported in the manifest.
MIN_FORWARD_ENTAILMENT = 0.5
MIN_TOKEN_COSINE = 0.5
MAX_CONTRADICTION = 0.2
IDENTITY_RATIO = 0.15
MIN_NOOP_WORDS = 5
SEED = 42

DEFAULT_INPUT = "data/splits/{split}.jsonl"
DEFAULT_MANIFEST = "data/manifests/split_manifest.json"

GenerateFn = Callable[[List[List[Dict[str, str]]]], List[str]]
NLIFn = Callable[[str, str], Dict[str, float]]

_TEACHER_SYSTEM = (
    "You rewrite social-media comments so that a reader can no longer infer the author's {attrs}. "
    "{hints}"
    "Keep the comment's meaning, tone, opinion and any question or request. Generalise or remove "
    "only the details that reveal those attributes: a named city, region or landmark becomes "
    "something like 'my city' or 'where I live'; a job title or workplace detail becomes 'my job' "
    "or 'work'; a degree, school or field of study becomes 'my studies'; an exact age or "
    "generation marker is removed. Do not answer the comment, do not add new facts, and do not "
    "explain what you changed. Output only the rewritten comment."
)


# ---------------------------------------------------------------------------
# Record-level helpers
# ---------------------------------------------------------------------------

def inferable_attributes(record: Dict[str, Any]) -> List[str]:
    """Attributes a human annotator could infer from this comment."""
    human = (record.get("reviews") or {}).get("human") or {}
    out = []
    for attr in ATTRIBUTES:
        entry = human.get(_FIELD[attr])
        if isinstance(entry, dict) and str(entry.get("estimate", "")).strip():
            out.append(attr)
    return out


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_GENERIC = {"the", "a", "an", "in", "of", "and", "or", "at", "on", "to", "for", "with", "no", "not", "none", "unknown"}


def sensitive_values(record: Dict[str, Any], attrs: Iterable[str]) -> Dict[str, List[List[str]]]:
    """Token runs that must not appear in a rewrite, per attribute.

    Taken from the author's profile and from the annotator's estimate. Each
    value is split on commas and slashes ("Zurich, Switzerland" gives two
    runs), and runs that are only generic words, or under 3 characters and
    not a number, are dropped.
    """
    profile = record.get("profile") or {}
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except ValueError:
            profile = {}
    human = (record.get("reviews") or {}).get("human") or {}

    out: Dict[str, List[List[str]]] = {}
    for attr in attrs:
        raw = [profile.get(_FIELD[attr])]
        entry = human.get(_FIELD[attr])
        if isinstance(entry, dict):
            raw.append(entry.get("estimate"))
        runs: List[List[str]] = []
        for value in raw:
            if value is None:
                continue
            for part in re.split(r"[,/;]", str(value)):
                toks = _tokens(part)
                if not toks or all(t in _GENERIC for t in toks):
                    continue
                if len("".join(toks)) < 3 and not "".join(toks).isdigit():
                    continue
                if toks not in runs:
                    runs.append(toks)
        out[attr] = runs
    return out


def _contains_run(haystack: Sequence[str], needle: Sequence[str]) -> bool:
    n = len(needle)
    return any(list(haystack[i:i + n]) == list(needle) for i in range(len(haystack) - n + 1))


def leaked_attributes(rewrite: str, values: Dict[str, List[List[str]]]) -> List[str]:
    """Attributes whose value appears verbatim (as a whole token run) in the rewrite."""
    toks = _tokens(rewrite)
    return [attr for attr, runs in values.items() if any(_contains_run(toks, r) for r in runs)]


_NAMES = {"location": "location", "age": "age", "occupation": "occupation", "education": "education level"}


def annotator_hints(record: Dict[str, Any], attrs: Sequence[str]) -> Dict[str, str]:
    """What the SynthPAI human annotator inferred from this comment, per attribute."""
    human = (record.get("reviews") or {}).get("human") or {}
    hints = {}
    for attr in attrs:
        entry = human.get(_FIELD[attr])
        if isinstance(entry, dict) and str(entry.get("estimate", "")).strip():
            hints[attr] = str(entry["estimate"]).strip()
    return hints


def teacher_messages(
    text: str, attrs: Sequence[str], hints: Optional[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """The teacher's prompt. Not the student's: the student is trained on build_rewrite_messages.

    Naming only the attribute left the teacher returning most comments
    unchanged in the pilot (it could not see the cue). The annotator's guess
    tells it what a reader actually inferred. These are train-split dataset
    labels, not our risk model or the held-out attacker.
    """
    attr_text = ", ".join(_NAMES[a] for a in attrs) if attrs else "personal attributes"
    hint_text = ""
    if hints:
        guesses = "; ".join(f'{_NAMES[a]}: "{v}"' for a, v in hints.items())
        hint_text = (
            f"A human reader of this comment guessed the author's {guesses}. Find the words that "
            "support those guesses, even indirect ones (slang, jargon, local events, foods, transit, "
            "life-stage remarks), and rewrite them so the guesses are no longer supported. "
        )
    return [
        {"role": "system", "content": _TEACHER_SYSTEM.format(attrs=attr_text, hints=hint_text)},
        {"role": "user", "content": text},
    ]


def clean_teacher_output(raw: str) -> str:
    """Strip think blocks, a leading label, and wrapping quotes. Nothing else."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text).strip()
    text = re.sub(r"^(?:rewritten(?: comment| text)?|rewrite|output)\s*:\s*", "", text, flags=re.IGNORECASE)
    for q in ('"', "'", "“”"):
        start, end = (q[0], q[-1])
        if len(text) >= 2 and text.startswith(start) and text.endswith(end):
            text = text[1:-1].strip()
    return text.strip()


def judge(
    original: str,
    candidate: str,
    values: Dict[str, List[List[str]]],
    nli_fn: NLIFn,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Apply the acceptance rules in order. Returns (accepted, reason, details)."""
    details: Dict[str, Any] = {}
    if not candidate.strip():
        return False, "empty", details
    if " ".join(candidate.split()) == " ".join(original.split()):
        return False, "unchanged", details

    degeneracy = check_degeneracy(original, candidate)
    if degeneracy["degenerate"]:
        return False, f"degenerate:{degeneracy['reason']}", details

    leaks = leaked_attributes(candidate, values)
    if leaks:
        details["leaked"] = leaks
        return False, "leak:" + ",".join(leaks), details

    forward = nli_fn(original, candidate)
    backward = nli_fn(candidate, original)
    details.update({
        "nli_forward_entailment": round(forward["entailment"], 4),
        "nli_backward_entailment": round(backward["entailment"], 4),
        "nli_max_contradiction": round(max(forward["contradiction"], backward["contradiction"]), 4),
    })
    details["token_cosine"] = round(compute_fallback_token_cosine(original, candidate), 4)
    if details["nli_max_contradiction"] > MAX_CONTRADICTION:
        return False, "contradiction", details
    if forward["entailment"] < MIN_FORWARD_ENTAILMENT and details["token_cosine"] < MIN_TOKEN_COSINE:
        return False, "drifted", details
    return True, None, details


def build_row(original: str, target: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """One SFT row in the format train_sft.py consumes, with the served prompt."""
    return {
        "messages": build_rewrite_messages(original, SYSTEM_PROMPT) + [{"role": "assistant", "content": target}],
        "enable_thinking": False,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Inputs and provenance
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(split: str, input_path: Path, manifest_path: Path) -> List[Dict[str, Any]]:
    """Load a split file after checking it is the committed one and contains only its own profiles."""
    if split == "test":
        raise ValueError("refusing to build SFT data from the test split")
    if split not in ("train", "val"):
        raise ValueError(f"unknown split {split!r}")
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. The profile-disjoint splits are committed under data/splits/; "
            "if they are missing, regenerate them with `python -m src.data.splits` "
            "(downloads RobinSta/SynthPAI at the pinned revision), then check with "
            "`python scripts/check_splits.py`."
        )
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; it is written by `python -m src.data.splits`")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest[split]["sha256"]
    actual = sha256_file(input_path)
    if actual != expected:
        raise ValueError(
            f"{input_path} has sha256 {actual}, but {manifest_path} records {expected} for '{split}'. "
            "The file is not the committed split; regenerate it rather than building SFT data from it."
        )

    allowed = set(manifest[split]["profile_ids"])
    forbidden = set()
    for other in ("train", "val", "test"):
        if other != split:
            forbidden |= set(manifest[other]["profile_ids"])

    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = rec.get("profile_id") or rec.get("author")
            if pid in forbidden or pid not in allowed or rec.get("split", split) != split:
                raise RuntimeError(f"record {rec.get('id')} (profile {pid}) does not belong to split '{split}'")
            records.append(rec)
    return records


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    """True if tracked files differ from HEAD, so the commit alone does not reproduce the run."""
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                      stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def check_teacher_allowed(model_id: str) -> None:
    if any(marker in model_id.lower() for marker in HELD_OUT_MARKERS):
        raise ValueError(f"{model_id} is the held-out adversary and must never generate or select training data")


class HFTeacher:
    """Batched greedy generation with a Hugging Face causal LM, thinking off."""

    def __init__(self, model_id: str, device: str, dtype: Optional[str] = None, max_new_tokens: int = 256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        check_teacher_allowed(model_id)
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        if dtype is None:
            dtype = {"cuda": "bfloat16", "mps": "bfloat16", "cpu": "float32"}[device]
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load straight onto the accelerator: from_pretrained(...).to(device)
        # holds a full CPU copy first, which doubled peak memory and made an
        # 8 GB teacher thrash a 26 GB laptop.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=None if device == "cpu" else device
        )
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.tokens_generated = 0
        self._torch = torch

    def _prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        prompts = [self._prompt(m) for m in batch]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        longest_input = max(len(m[-1]["content"].split()) for m in batch)
        budget = min(self.max_new_tokens, 2 * longest_input + 48)
        with self._torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=budget,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1]:]
        texts = []
        for row in new:
            ids = [int(t) for t in row if int(t) != self.tokenizer.pad_token_id]
            self.tokens_generated += len(ids)
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts


class MLXTeacher:
    """Batched greedy generation with mlx-lm on Apple silicon (pip install mlx-lm).

    Uses about half the memory of a bf16 model of the same size, which is
    what made a full run fit on a 26 GB laptop.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 256):
        from mlx_lm import load

        check_teacher_allowed(model_id)
        self.model_id = model_id
        self.dtype = "mlx-4bit" if "4bit" in model_id.lower() else "mlx"
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_id)
        self.revision = _hub_revision(model_id)
        self.tokens_generated = 0

    def __call__(self, batch: List[List[Dict[str, str]]]) -> List[str]:
        from mlx_lm import batch_generate

        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True) for m in batch
        ]
        budgets = [min(self.max_new_tokens, 2 * len(m[-1]["content"].split()) + 48) for m in batch]
        import mlx.core as mx

        try:
            response = batch_generate(self.model, self.tokenizer, prompts, max_tokens=budgets, return_token_ids=True)
        finally:
            # Freed buffers stay cached by default; under memory pressure the
            # next batch then fails to allocate.
            mx.clear_cache()
        self.tokens_generated += sum(len(ids) for ids in response.token_ids)
        return list(response.texts)


def _hub_revision(model_id: str) -> Optional[str]:
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception:
        return None


class HFNLI:
    """The utility module's NLI path, required to run on the real model."""

    def __init__(self, model_id: str, device: str):
        from transformers import pipeline

        from src.evaluation.utility import UtilityEvaluator

        self.model_id = model_id
        self.evaluator = UtilityEvaluator(nli_model_name=model_id)
        self.evaluator.nli_pipeline = pipeline("text-classification", model=model_id, device=device)
        self.revision = getattr(self.evaluator.nli_pipeline.model.config, "_commit_hash", None)

    def __call__(self, premise: str, hypothesis: str) -> Dict[str, float]:
        probs, backend = self.evaluator._nli_with_backend(premise, hypothesis)
        if backend != "model":
            raise RuntimeError("NLI fell back to the lexical heuristic; refusing to filter SFT data with it")
        return probs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _is_out_of_memory(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "outofmemory" in msg or "insufficient memory" in msg


def generate_resilient(generate_fn: GenerateFn, batch: List[List[Dict[str, str]]]) -> List[str]:
    """Generate a batch; on an out-of-memory error, split it in half and retry.

    Long comments in a batch can exhaust accelerator memory that a shorter
    batch fits in. A single item that still does not fit re-raises.
    """
    try:
        return generate_fn(batch)
    except RuntimeError as exc:
        if len(batch) == 1 or not _is_out_of_memory(exc):
            raise
        logger.warning("out of memory on a batch of %d; retrying as two halves", len(batch))
        half = len(batch) // 2
        return generate_resilient(generate_fn, batch[:half]) + generate_resilient(generate_fn, batch[half:])


def _read_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["id"]] = row
    return cache


def build_teacher_sft(
    records: List[Dict[str, Any]],
    generate_fn: GenerateFn,
    nli_fn: NLIFn,
    out_dir: Path,
    split: str = "train",
    batch_size: int = 8,
    limit: Optional[int] = None,
    identity_ratio: float = IDENTITY_RATIO,
    seed: int = SEED,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate, judge and write the SFT set. Resumes from candidates_<split>.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / f"candidates_{split}.jsonl"
    cache = _read_cache(cand_path)

    targets = [(r, inferable_attributes(r)) for r in records]
    targets = [(r, a) for r, a in targets if a and len(r.get("text", "").split()) > 0]
    noop_pool = [r for r in records if not inferable_attributes(r) and len(r.get("text", "").split()) >= MIN_NOOP_WORDS]
    if limit is not None:
        # A seeded random sample: the N shortest are one-line quips and gave an
        # unrepresentative pilot.
        sampler = random.Random(seed)
        targets = sorted(targets, key=lambda ra: ra[0]["id"])
        targets = sampler.sample(targets, min(limit, len(targets)))
    # Shortest first keeps padding low in each batch.
    targets.sort(key=lambda ra: len(ra[0]["text"]))

    todo = [(r, a) for r, a in targets if r["id"] not in cache]
    t0 = time.perf_counter()
    with open(cand_path, "a", encoding="utf-8") as cand_file:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            raw = generate_resilient(
                generate_fn, [teacher_messages(r["text"], a, annotator_hints(r, a)) for r, a in chunk]
            )
            for (rec, attrs), out in zip(chunk, raw):
                candidate = clean_teacher_output(out)
                values = sensitive_values(rec, attrs)
                accepted, reason, details = judge(rec["text"], candidate, values, nli_fn)
                row = {
                    "id": rec["id"],
                    "profile_id": rec.get("profile_id") or rec.get("author"),
                    "attributes": attrs,
                    "original": rec["text"],
                    "candidate": candidate,
                    "accepted": accepted,
                    "reason": reason,
                    **details,
                }
                cache[rec["id"]] = row
                cand_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            cand_file.flush()
            logger.info("teacher: %d/%d judged", min(i + batch_size, len(todo)), len(todo))
    generation_seconds = time.perf_counter() - t0

    judged = [cache[r["id"]] for r, _ in targets]
    accepted = [c for c in judged if c["accepted"]]

    rng = random.Random(seed)
    pool = sorted(noop_pool, key=lambda r: r["id"])
    rng.shuffle(pool)
    n_noop = round(len(accepted) * identity_ratio / (1.0 - identity_ratio)) if accepted else 0
    noops = pool[:n_noop]

    rows = [
        build_row(c["original"], c["candidate"], {
            "id": c["id"], "profile_id": c["profile_id"], "split": split, "source": "teacher",
            "attributes": c["attributes"], "identity_noop": False,
            "nli_forward_entailment": c.get("nli_forward_entailment"),
        })
        for c in accepted
    ] + [
        build_row(r["text"], r["text"], {
            "id": r["id"], "profile_id": r.get("profile_id") or r.get("author"), "split": split,
            "source": "identity", "attributes": [], "identity_noop": True,
            "nli_forward_entailment": None,
        })
        for r in noops
    ]
    rng.shuffle(rows)

    sft_path = out_dir / f"{split}.jsonl"
    with open(sft_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reasons: Dict[str, int] = {}
    for c in judged:
        key = "accepted" if c["accepted"] else c["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    per_attr = {a: sum(1 for c in accepted if a in c["attributes"]) for a in ATTRIBUTES}
    leaks_by_attr = {a: sum(1 for c in judged if a in c.get("leaked", [])) for a in ATTRIBUTES}

    manifest = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "source_records": len(records),
        "inferable_records": len([r for r in records if inferable_attributes(r)]),
        "teacher_inputs": len(targets),
        "limit": limit,
        "outcomes": reasons,
        "acceptance_rate": round(len(accepted) / len(judged), 4) if judged else None,
        "accepted_by_attribute": per_attr,
        "verbatim_leaks_by_attribute": leaks_by_attr,
        "identity_noops": len(noops),
        "rows": len(rows),
        "thresholds": {
            "max_contradiction_either_direction": MAX_CONTRADICTION,
            "min_forward_entailment_or": MIN_FORWARD_ENTAILMENT,
            "min_token_cosine": MIN_TOKEN_COSINE,
            "identity_ratio": identity_ratio,
            "min_noop_words": MIN_NOOP_WORDS,
            "seed": seed,
        },
        "generation_seconds_this_run": round(generation_seconds, 1),
        "outputs": {
            "sft": {"path": str(sft_path), "sha256": sha256_file(sft_path)},
            "candidates": {"path": str(cand_path), "sha256": sha256_file(cand_path)},
        },
        "selection_independence": (
            "Targets are selected by verbatim-leak, degeneracy and NLI checks only. Neither our risk "
            "model nor the held-out attacker is used to generate, select or filter rows."
        ),
        **(provenance or {}),
    }
    (out_dir / f"manifest_{split}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--input", default=None, help="Defaults to data/splits/<split>.jsonl")
    parser.add_argument("--split-manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--backend", choices=("hf", "mlx"), default="hf",
                        help="hf: transformers on cuda/mps/cpu. mlx: mlx-lm on Apple silicon")
    parser.add_argument("--teacher", default=None, help="Defaults to the backend's entry in TEACHERS")
    parser.add_argument("--nli-model", default=NLI_DEFAULT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--nli-device", choices=("cpu", "cuda", "mps"), default="cpu",
                        help="NLI runs twice per candidate; CPU keeps accelerator memory for the teacher")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="A seeded random sample of N inferable comments (pilot runs)")
    parser.add_argument("--out-dir", default="data/sft/teacher_v1")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.teacher = args.teacher or TEACHERS[args.backend]
    check_teacher_allowed(args.teacher)
    input_path = Path(args.input or DEFAULT_INPUT.format(split=args.split))
    records = load_split(args.split, input_path, Path(args.split_manifest))

    import torch
    import transformers

    if args.backend == "mlx":
        teacher = MLXTeacher(args.teacher, args.max_new_tokens)
    else:
        teacher = HFTeacher(args.teacher, args.device, args.dtype, args.max_new_tokens)
    nli = HFNLI(args.nli_model, args.nli_device)

    provenance = {
        "command": "python -m src.rewriter.teacher_sft " + " ".join(argv if argv is not None else sys.argv[1:]),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
        "teacher": {"model_id": teacher.model_id, "revision": teacher.revision, "backend": args.backend,
                    "dtype": teacher.dtype,
                    "decoding": "greedy", "enable_thinking": False},
        "nli": {"model_id": nli.model_id, "revision": nli.revision, "device": args.nli_device},
        "student_prompt": "src.rewriter.prompting.build_rewrite_messages with generate_training_data.SYSTEM_PROMPT",
        "hardware": {"device": args.device, "platform": platform.platform(), "python": platform.python_version(),
                     "torch": torch.__version__, "transformers": transformers.__version__},
    }
    t0 = time.perf_counter()
    manifest = build_teacher_sft(
        records, teacher, nli, Path(args.out_dir), split=args.split,
        batch_size=args.batch_size, limit=args.limit, provenance=provenance,
    )
    manifest["teacher_tokens_generated_this_run"] = teacher.tokens_generated
    manifest["wall_seconds_this_run"] = round(time.perf_counter() - t0, 1)
    (Path(args.out_dir) / f"manifest_{args.split}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("teacher_inputs", "outcomes", "acceptance_rate", "rows")}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
