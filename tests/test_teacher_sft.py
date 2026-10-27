"""Teacher-LLM SFT builder: selection rules, split hygiene, train/serve prompt parity.

No model is loaded here: generation and NLI are injected as fakes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.rewriter import teacher_sft as ts
from src.rewriter.generate_training_data import SYSTEM_PROMPT
from src.rewriter.prompting import build_rewrite_messages


def _human(**estimates):
    fields = {"age": "age", "location": "city_country", "occupation": "occupation", "education": "education"}
    human = {f: {"estimate": "", "certainty": 0, "hardness": 0} for f in fields.values()}
    for attr, est in estimates.items():
        human[fields[attr]] = {"estimate": est, "certainty": 4, "hardness": 2}
    return {"human": human}


def _record(rid, pid, text, split="train", **estimates):
    return {
        "id": rid,
        "profile_id": pid,
        "author": pid,
        "split": split,
        "text": text,
        "profile": {"age": 31, "city_country": "Zurich, Switzerland", "occupation": "software engineer",
                    "education": "Masters in Computer Science"},
        "reviews": _human(**estimates),
    }


def entailing_nli(premise, hypothesis):
    return {"entailment": 0.9, "contradiction": 0.01, "neutral": 0.09}


# --- record helpers ---------------------------------------------------------

def test_inferable_attributes_reads_human_estimates():
    rec = _record("c1", "pers1", "text", occupation="software engineer", location="Zurich")
    assert ts.inferable_attributes(rec) == ["location", "occupation"]
    assert ts.inferable_attributes(_record("c2", "pers1", "text")) == []


def test_sensitive_values_split_and_filter():
    rec = _record("c1", "pers1", "text", location="Switzerland", age="30")
    values = ts.sensitive_values(rec, ["location", "age"])
    assert ["zurich"] in values["location"] and ["switzerland"] in values["location"]
    assert ["31"] in values["age"] and ["30"] in values["age"]


@pytest.mark.parametrize(
    "rewrite,expected",
    [
        ("I moved to Zurich last year.", ["location"]),
        ("Life in my city is expensive.", []),
        ("As a software engineer I review code.", ["occupation"]),
        ("As an engineer I review code.", []),  # partial run is not a verbatim leak
        ("I turned 31 this spring.", ["age"]),
        ("I have 310 cats.", []),  # whole tokens only
    ],
)
def test_leak_check_is_whole_token_runs(rewrite, expected):
    rec = _record("c1", "pers1", "x", location="Zurich", occupation="software engineer", age="31")
    values = ts.sensitive_values(rec, ["location", "occupation", "age"])
    assert ts.leaked_attributes(rewrite, values) == expected


def test_clean_teacher_output():
    assert ts.clean_teacher_output("<think>\n\n</think>\n\nI live in my city.") == "I live in my city."
    assert ts.clean_teacher_output('Rewritten comment: "I like my job."') == "I like my job."


# --- acceptance rules ---------------------------------------------------------

ORIGINAL = "Working as a software engineer in Zurich means long hours and great coffee."
VALUES = {"location": [["zurich"], ["switzerland"]], "occupation": [["software", "engineer"]]}


@pytest.mark.parametrize(
    "candidate,nli,reason",
    [
        ("", entailing_nli, "empty"),
        (ORIGINAL + "  ", entailing_nli, "unchanged"),
        ("an area an area an area an area", entailing_nli, "degenerate:repetition"),
        ("Working as a software engineer in my city means long hours and great coffee.", entailing_nli,
         "leak:occupation"),
        # A generalisation is neutral, not entailed; it keeps the wording, so it passes.
        ("Working in my field in my city means long hours and great coffee.",
         lambda p, h: {"entailment": 0.002, "contradiction": 0.0, "neutral": 0.998}, None),
        # The teacher answered instead of rewriting: neither entailed nor the same wording.
        ("You should ask your manager for flexible hours and buy a better espresso machine.",
         lambda p, h: {"entailment": 0.01, "contradiction": 0.05, "neutral": 0.94}, "drifted"),
        ("Working in my field in my city means long hours and great coffee.",
         lambda p, h: {"entailment": 0.9, "contradiction": 0.5, "neutral": 0.0}, "contradiction"),
        ("Working in my field in my city means long hours and great coffee.", entailing_nli, None),
    ],
)
def test_judge_reasons_in_order(candidate, nli, reason):
    accepted, got, _ = ts.judge(ORIGINAL, candidate, VALUES, nli)
    assert got == reason
    assert accepted is (reason is None)


def test_held_out_attacker_is_refused_as_teacher():
    for model_id in ("microsoft/Phi-4-mini-instruct", "mlx-community/Phi-4-mini-instruct-4bit"):
        with pytest.raises(ValueError, match="held-out"):
            ts.check_teacher_allowed(model_id)
    for model_id in ts.TEACHERS.values():
        ts.check_teacher_allowed(model_id)


# --- train / serve parity -------------------------------------------------------

def test_sft_row_prompt_is_the_served_prompt():
    row = ts.build_row("I live in Zurich.", "I live in my city.", {})
    assert row["messages"][:-1] == build_rewrite_messages("I live in Zurich.", SYSTEM_PROMPT)
    assert row["messages"][-1] == {"role": "assistant", "content": "I live in my city."}
    assert row["enable_thinking"] is False


def test_inference_uses_the_shared_builder():
    pytest.importorskip("torch")
    from src.rewriter.inference import QwenRewriterInference

    rw = QwenRewriterInference(adapter_path=None)
    prompt = rw.format_chat_prompt("I live in Zurich.")
    for m in build_rewrite_messages("I live in Zurich.", rw.system_prompt):
        assert m["content"] in prompt


# --- split hygiene ------------------------------------------------------------

def _write_split(tmp_path, records, split="train", other_profiles=("pers9",)):
    path = tmp_path / f"{split}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    profiles = sorted({r["profile_id"] for r in records})
    manifest = {
        "train": {"sha256": digest if split == "train" else "x", "profile_ids": profiles if split == "train" else []},
        "val": {"sha256": digest if split == "val" else "x", "profile_ids": profiles if split == "val" else []},
        "test": {"sha256": "x", "profile_ids": list(other_profiles)},
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return path, mpath


def test_load_split_refuses_test():
    with pytest.raises(ValueError, match="test split"):
        ts.load_split("test", Path("x"), Path("y"))


def test_load_split_missing_input_says_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.data.splits"):
        ts.load_split("train", tmp_path / "nope.jsonl", tmp_path / "manifest.json")


def test_load_split_rejects_a_modified_file(tmp_path):
    path, mpath = _write_split(tmp_path, [_record("c1", "pers1", "hi")])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the committed split"):
        ts.load_split("train", path, mpath)


def test_load_split_rejects_a_foreign_profile(tmp_path):
    recs = [_record("c1", "pers1", "hi"), _record("c2", "pers9", "leaked from test")]
    path, mpath = _write_split(tmp_path, recs)
    manifest = json.loads(mpath.read_text())
    manifest["train"]["profile_ids"] = ["pers1", "pers9"]
    mpath.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="does not belong"):
        ts.load_split("train", path, mpath)


def test_committed_train_split_passes_the_check():
    records = ts.load_split("train", Path("data/splits/train.jsonl"), Path("data/manifests/split_manifest.json"))
    assert len(records) == 5232


# --- end to end with fakes ------------------------------------------------------

def test_build_writes_rows_candidates_and_manifest(tmp_path):
    records = [
        _record("a1", "pers1", "Working as a software engineer in Zurich is fun.", occupation="software engineer",
                location="Zurich"),
        _record("a2", "pers1", "My software engineer team ships on Fridays.", occupation="software engineer"),
    ] + [_record(f"n{i}", "pers2", f"Nothing personal here at all, just thought number {i}.") for i in range(10)]

    outputs = {
        "Working as a software engineer in Zurich is fun.": "Working in my field in my city is fun.",
        "My software engineer team ships on Fridays.": "My software engineer team ships on Fridays!",
    }

    calls = []

    def fake_generate(batch):
        calls.append(len(batch))
        return [outputs[m[-1]["content"]] for m in batch]

    manifest = ts.build_teacher_sft(records, fake_generate, entailing_nli, tmp_path, batch_size=4)

    assert manifest["teacher_inputs"] == 2
    assert manifest["outcomes"] == {"accepted": 1, "leak": 1}
    assert manifest["identity_noops"] == 0  # round(1 * 0.15 / 0.85) == 0
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["messages"][-1]["content"] == "Working in my field in my city is fun."
    cands = [json.loads(line) for line in (tmp_path / "candidates_train.jsonl").read_text().splitlines()]
    assert {c["reason"] for c in cands} == {None, "leak:occupation"}
    assert json.loads((tmp_path / "manifest_train.json").read_text())["rows"] == 1

    # Resume: a second run generates nothing new.
    calls.clear()
    ts.build_teacher_sft(records, fake_generate, entailing_nli, tmp_path, batch_size=4)
    assert calls == []


def test_identity_noops_are_about_fifteen_percent(tmp_path):
    texts = [f"I work as a software engineer and this is post {i}." for i in range(17)]
    records = [_record(f"a{i}", "pers1", t, occupation="software engineer") for i, t in enumerate(texts)]
    records += [_record(f"n{i}", "pers2", f"Nothing personal here at all, just thought number {i}.") for i in range(10)]

    def fake_generate(batch):
        return [m[-1]["content"].replace("software engineer", "professional") for m in batch]

    manifest = ts.build_teacher_sft(records, fake_generate, entailing_nli, tmp_path)
    assert manifest["outcomes"] == {"accepted": 17}
    assert manifest["identity_noops"] == 3  # round(17 * 0.15 / 0.85)
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    noops = [r for r in rows if r["metadata"]["identity_noop"]]
    assert len(noops) == 3
    assert all(r["messages"][-1]["content"] in r["messages"][-2]["content"] for r in noops)


def test_teacher_prompt_carries_the_annotator_guess():
    rec = _record("c1", "pers1", "yeah devs can lift... weights & bugs!", occupation="software developer")
    attrs = ts.inferable_attributes(rec)
    msgs = ts.teacher_messages(rec["text"], attrs, ts.annotator_hints(rec, attrs))
    assert 'occupation: "software developer"' in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": rec["text"]}


def test_out_of_memory_splits_the_batch_and_keeps_order():
    seen = []

    def flaky(batch):
        seen.append(len(batch))
        if len(batch) > 2:
            raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
        return [m[-1]["content"].upper() for m in batch]

    batch = [[{"role": "user", "content": c}] for c in "abcde"]
    assert ts.generate_resilient(flaky, batch) == ["A", "B", "C", "D", "E"]
    assert seen == [5, 2, 3, 1, 2]  # 5 fails -> 2 + 3; 3 fails -> 1 + 2


def test_other_errors_are_not_swallowed():
    def broken(batch):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        ts.generate_resilient(broken, [[{"role": "user", "content": "a"}]] * 4)
