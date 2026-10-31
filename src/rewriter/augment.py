"""Position-, surface- and register-randomized augmentation for the rewriter SFT set.

Why this module exists
----------------------
The v1 SFT set (``data/sft/teacher_v1``) taught the student *where* PII sits
rather than *how to find* it:

  * 69% of supervised edits started in the first third of the input (the cue
    was at the beginning), 21% middle, 10% end;
  * inputs were 12-word casual comments (p25=9, p75=28) while real user inputs
    are long, formal, multi-clause prompts;
  * location appeared in only 10.9% of rows, occupation in 57%.

A student trained on that distribution learns the positional prior "the cue is
at the start" and surface-memorizes the handful of values it has seen. This
module composes training examples **by construction** with known cue spans at
*controlled, uniformly sampled positions* (preamble, postamble, mid-sentence,
parenthetical, split across both ends), drawn from *surface-diverse* persona
banks (worldwide cities, 120+ occupations, global institutions), across
*registers* (several frame variants per position) - so the only way to solve
the task is content-based detection.

Because the composer knows exactly which facts it injected and where, it also
knows the ground-truth target and can verify removal of every injected surface
form (see ``src.rewriter.teacher_sft.judge_augmented``), closing the "relocated
instead of removed" failure mode that the old verbatim-value check missed.

Deterministic and torch-free: rows can be built on a laptop with no GPU and no
teacher model, then mixed into the teacher SFT build with
``python -m src.rewriter.teacher_sft --augment-dir data/sft/augmented_v1``.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.rewriter.persona_banks import BANKS_DIR, _load_bank

POSITIONS: Tuple[str, ...] = ("preamble", "postamble", "mid_sentence", "parenthetical", "split")

# Attribute sampling weights. Location was the thinnest class in v1 (10.9% of
# rows), so it is boosted; occupation was dominant and is damped.
ATTRIBUTE_WEIGHTS: Dict[str, float] = {
    "age": 1.0,
    "location": 1.6,
    "occupation": 0.9,
    "education": 1.0,
    "income": 0.8,
    "local_reference": 1.2,
}

# Rendering frames per position. Several variants per position give register
# diversity. Every {persona} slot accepts both noun fragments ("a 34-year-old")
# and predicate fragments ("based in Krakow"), so frames never assume a
# phrase type. _u / _l suffixes uppercase / lowercase the first character.
_FRAME_VARIANTS: Dict[str, List[str]] = {
    "preamble": [
        "I am {persona}. {carrier}",
        "Quick context: I'm {persona}. {carrier}",
        "{persona_u}, for the record. {carrier}",
        "So: I'm {persona}. {carrier}",
    ],
    "postamble": [
        "{carrier} FYI, I am {persona}.",
        "{carrier} By the way, I'm {persona}.",
        "{carrier} For context: {persona_u}.",
        "{carrier} Relevant background - I am {persona}.",
    ],
    "mid_sentence": [
        "{first_half} as {persona_l}, {second_half}",
        "{first_half}, being {persona_l}, {second_half}",
        "{first_half} - {persona_u}, for context - {second_half}",
        "{first_half} - and, {persona_l}, this keeps coming up - {second_half}",
        "{first_half}; {persona_l}, for what it's worth; {second_half}",
    ],
    "parenthetical": [
        "{carrier} (for context: I'm {persona})",
        "{carrier} [context: {persona_u}]",
        "Before answering - I'm {persona} - {carrier_l}",
    ],
    "split": [
        "I am {fact_a}. {carrier} Also, I'm {fact_b}.",
        "{fact_a_u} here. {carrier} Oh, and I'm {fact_b}.",
        "For starters: I'm {fact_a}. {carrier} Related: I'm {fact_b}.",
    ],
}

# Persona fragments per attribute. "{v}" is the drawn surface value. Fragments
# carry their own articles/prepositions, so any {persona} slot accepts them.
_PERSONA_FRAGMENTS: Dict[str, List[str]] = {
    "age": ["a {v}", "a {v}", "a {v}"],
    "location": ["based in {v}", "living in {v}", "from {v}", "in {v}"],
    "occupation": ["a {v}", "working as a {v}", "a {v} by trade"],
    "education_inst": ["a {v} graduate", "a {v} alum", "studied at {v}"],
    "education_degree": ["holder of {v}", "with {v}"],
    "income": ["earning {v}", "on {v}", "making {v}"],
    "local_transit": ["a regular on {v}", "someone who takes {v} every day"],
    "local_landmark": ["living near {v}", "a {v} regular"],
    "local_custom": ["a big fan of {v}", "someone who swears by {v}"],
}

# Parameter generalizations for carriers whose answer depends on a personal
# attribute: the persona must be GENERALIZED, not deleted, to keep utility.
# Every phrase fits the target template "For someone {param}: {carrier}".
_PARAM_PHRASES: Dict[str, List[str]] = {
    "age": ["at my current life stage", "at my stage of life"],
    "location": ["in my local climate", "living in my region"],
    "occupation": ["working in my field", "in my line of work"],
    "education": ["with my academic background", "with my level of studies"],
    "income": ["with my income level", "on my current income"],
    "local_reference": ["relying on my local transit options"],
}


def _phrase_in(haystack: str, needle: str) -> bool:
    """Case-insensitive whole-phrase containment (word boundaries)."""
    if not needle:
        return False
    return re.search(rf"(?<![\w]){re.escape(needle)}(?![\w])", haystack, re.IGNORECASE) is not None


@dataclass
class InjectedFact:
    """One persona fact injected into an augmented example."""

    attribute: str
    value: str            # canonical bank value, e.g. "Krakow"
    surface: str          # exact string appearing in the input, e.g. "Krakow"
    variants: List[str] = field(default_factory=list)  # other surface forms that must not appear
    fragment: str = ""    # rendered persona fragment, e.g. "living in Krakow"


@dataclass
class AugmentedExample:
    """One composed training example with known cue spans."""

    id: str
    input: str
    target: str
    attributes: List[str]
    position: str
    register: str
    injected: List[InjectedFact]
    needs_persona: bool
    carrier_id: str

    def to_record(self) -> Dict:
        return {
            "id": self.id,
            "input": self.input,
            "target": self.target,
            "attributes": self.attributes,
            "position": self.position,
            "register": self.register,
            "injected": [
                {"attribute": f.attribute, "value": f.value, "surface": f.surface,
                 "variants": f.variants, "fragment": f.fragment}
                for f in self.injected
            ],
            "needs_persona": self.needs_persona,
            "carrier_id": self.carrier_id,
            "source": "augment",
        }

    @staticmethod
    def from_record(rec: Dict) -> "AugmentedExample":
        return AugmentedExample(
            id=rec["id"],
            input=rec["input"],
            target=rec["target"],
            attributes=rec.get("attributes", []),
            position=rec.get("position", "unknown"),
            register=rec.get("register", "unknown"),
            injected=[InjectedFact(**f) for f in rec.get("injected", [])],
            needs_persona=rec.get("needs_persona", False),
            carrier_id=rec.get("carrier_id", ""),
        )


def _lower_first(text: str) -> str:
    """Lowercase the first character unless it starts a multi-cap proper noun."""
    if text and text[0].isupper() and (len(text) < 2 or not text[1].isupper()):
        return text[0].lower() + text[1:]
    return text


def _upper_first(text: str) -> str:
    if text:
        return text[0].upper() + text[1:]
    return text


class PersonaComposer:
    """Composes position-randomized examples from carriers + persona banks."""

    def __init__(self, banks_dir: Optional[Path] = None, carriers_path: Optional[Path] = None) -> None:
        self.banks_dir = Path(banks_dir) if banks_dir else BANKS_DIR
        carriers_path = carriers_path or (self.banks_dir / "carriers.json")
        with open(carriers_path, encoding="utf-8") as f:
            self.carriers: List[Dict] = json.load(f)["carriers"]
        self.cities = _load_bank("cities", self.banks_dir)["entries"]
        self.occupations = _load_bank("occupations", self.banks_dir)["entries"]
        edu = _load_bank("education", self.banks_dir)
        self.institutions = edu["institutions"]
        self.degrees = edu.get("degrees", [])
        self.local_entries = _load_bank("local_references", self.banks_dir)["entries"]

    # -- persona drawing ----------------------------------------------------

    def _draw_fact(self, attribute: str, rng: random.Random) -> InjectedFact:
        """Draw one fact and render its persona fragment.

        The fragment carries its own article/preposition so any frame slot
        accepts it; `surface` is the value string checked by the removal gate.
        """
        if attribute == "age":
            style = rng.choice(["exact", "exact_yo", "band", "gen"])
            if style == "exact":
                age = rng.randint(19, 78)
                return InjectedFact("age", str(age), f"{age}-year-old", [], f"a {age}-year-old")
            if style == "exact_yo":
                age = rng.randint(19, 78)
                return InjectedFact("age", str(age), f"{age}yo", [], f"a {age}yo")
            if style == "band":
                band = rng.choice(["twenty-something", "thirty-something", "forty-something",
                                   "fifty-something", "sixty-something"])
                return InjectedFact("age", band, band, [], f"a {band}")
            gen = rng.choice(["millennial", "Gen Z-er", "Gen X-er", "boomer"])
            return InjectedFact("age", gen, gen, [], f"a {gen}")
        if attribute == "income":
            amount = rng.choice([28, 35, 45, 52, 60, 75, 85, 95, 110, 140, 180, 240, 320, 450])
            style = rng.choice(["k", "full", "band"])
            if style == "k":
                surface = f"${amount}k a year"
            elif style == "full":
                surface = f"${amount},000 a year"
            else:
                surface = rng.choice(["a six-figure salary", "minimum wage"])
            verb = rng.choice(["earning", "on", "making"])
            return InjectedFact("income", f"{amount}k", surface, [], f"{verb} {surface}")
        if attribute == "location":
            entry = rng.choice(self.cities)
            variants = [entry["value"]] + entry.get("variants", [])
            frag = rng.choice(_PERSONA_FRAGMENTS["location"]).format(v=entry["value"])
            return InjectedFact("location", entry["value"], entry["value"], variants, frag)
        if attribute == "occupation":
            entry = rng.choice(self.occupations)
            variants = [entry["value"]] + entry.get("variants", [])
            frag = rng.choice(_PERSONA_FRAGMENTS["occupation"]).format(v=entry["value"].strip())
            return InjectedFact("occupation", entry["value"], entry["value"], variants, frag)
        if attribute == "education":
            if rng.random() < 0.6:
                entry = rng.choice(self.institutions)
                variants = [entry["value"]] + entry.get("variants", [])
                frag = rng.choice(_PERSONA_FRAGMENTS["education_inst"]).format(v=entry["value"])
                return InjectedFact("education", entry["value"], entry["value"], variants, frag)
            degree = rng.choice(self.degrees)
            frag = rng.choice(_PERSONA_FRAGMENTS["education_degree"]).format(v=degree)
            return InjectedFact("education", degree, degree, [], frag)
        if attribute == "local_reference":
            entry = rng.choice(self.local_entries)
            variants = [entry["value"]] + entry.get("variants", [])
            kind = entry.get("kind", "custom")
            key = f"local_{kind}" if f"local_{kind}" in _PERSONA_FRAGMENTS else "local_custom"
            frag = rng.choice(_PERSONA_FRAGMENTS[key]).format(v=entry["value"])
            return InjectedFact("local_reference", entry["value"], entry["value"], variants, frag)
        raise ValueError(f"unknown attribute {attribute!r}")

    def _persona_clause(self, facts: List[InjectedFact]) -> str:
        """Join persona fragments into one readable clause."""
        parts = [f.fragment for f in facts if f.fragment]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    # -- composition ---------------------------------------------------------

    def _split_carrier(self, carrier: Dict) -> Optional[Tuple[str, str]]:
        """Split a carrier at a natural mid-sentence point, if one exists.

        Prefers the carrier's explicit ``split_hint``, then a clause boundary
        (comma or semicolon), then the start of the final short prepositional
        segment ("... consistent hashing | in Go"). Splitting at conjunction
        words like "and" can break fixed phrases ("ser and estar"), so they
        are avoided.
        """
        text = carrier["text"]
        hint = carrier.get("split_hint")
        if hint and hint in text:
            first, second = text.split(hint, 1)
            return first.rstrip(), second.lstrip()
        boundaries = [m.start() for m in re.finditer(r"[,;]", text)]
        if boundaries:
            cut = boundaries[-1]
            return text[:cut].rstrip(), text[cut + 1:].lstrip()
        # Final prepositional segment: split before the last "in/for/with/
        # under/on/about/during" whose tail is short and whose head is long
        # enough to stand alone.
        candidates = [
            m for m in re.finditer(r"\s\b(?:in|for|with|under|on|about|during)\b\s", text)
            if len(text) - m.end() <= 60 and m.start() >= 20
        ]
        if candidates:
            cut = candidates[-1].start()
            return text[:cut].rstrip(), text[cut:].lstrip()
        return None

    def compose(
        self,
        carrier: Dict,
        position: str,
        attributes: Sequence[str],
        rng: random.Random,
        example_id: str,
    ) -> AugmentedExample:
        text = carrier["text"]
        # Redraw facts whose surface already appears in the carrier: injecting
        # "minimum wage" into a question about minimum-wage elasticities would
        # make the correct target leak the injected span by construction.
        facts: List[InjectedFact] = []
        for attribute in attributes:
            for _ in range(6):
                fact = self._draw_fact(attribute, rng)
                if not _phrase_in(text, fact.surface) and not any(
                    _phrase_in(text, v) for v in fact.variants if v
                ):
                    facts.append(fact)
                    break
        if not facts:
            # Every draw collided with carrier content; fall back to a no-op.
            return self.compose_noop(carrier, example_id)
        clause = self._persona_clause(facts)
        needs_persona = bool(carrier.get("needs_persona"))
        param_attr = carrier.get("persona_param") or (attributes[0] if attributes else None)

        if position == "split" and len(facts) >= 2:
            frame = rng.choice(_FRAME_VARIANTS["split"])
            composed = frame.format(
                fact_a=self._persona_clause([facts[0]]),
                fact_a_u=_upper_first(self._persona_clause([facts[0]])),
                carrier=text,
                fact_b=self._persona_clause(facts[1:]),
            )
        elif position == "mid_sentence":
            halves = self._split_carrier(carrier)
            if halves is None:
                # No natural split point in this carrier; use parenthetical.
                return self.compose(carrier, "parenthetical", attributes, rng, example_id)
            first, second = halves
            frame = rng.choice(_FRAME_VARIANTS["mid_sentence"])
            composed = frame.format(
                first_half=first,
                persona_u=_upper_first(clause),
                persona_l=_lower_first(clause),
                second_half=second,
            )
        else:
            frame = rng.choice(_FRAME_VARIANTS[position])
            composed = frame.format(
                persona=clause,
                persona_u=_upper_first(clause),
                persona_l=_lower_first(clause),
                carrier=text,
                carrier_l=_lower_first(text),
            )

        composed = re.sub(r"\s*([,;:])\s*([,;:])", r"", composed)
        composed = re.sub(r"\s{2,}", " ", composed).strip()
        composed = _upper_first(composed)

        # Target: prune (persona irrelevant) or generalize (persona is a
        # parameter the answer needs). The carrier is kept verbatim - proper
        # nouns and technical content must survive untouched.
        if not needs_persona or not attributes:
            target = text.strip()
        else:
            param = rng.choice(_PARAM_PHRASES.get(param_attr, _PARAM_PHRASES["occupation"]))
            target = f"For someone {param}: {text.strip()}"

        register = f"{position}:{len(frame)}"
        return AugmentedExample(
            id=example_id,
            input=composed,
            target=target,
            attributes=list(attributes),
            position=position,
            register=register,
            injected=facts,
            needs_persona=needs_persona,
            carrier_id=carrier["id"],
        )

    def compose_noop(self, carrier: Dict, example_id: str) -> AugmentedExample:
        """A carrier with no injected persona: the correct rewrite is identity."""
        text = carrier["text"].strip()
        return AugmentedExample(
            id=example_id, input=text, target=text, attributes=[], position="none",
            register="noop", injected=[], needs_persona=False, carrier_id=carrier["id"],
        )


def _weighted_attributes(rng: random.Random, k: int) -> List[str]:
    pool = list(ATTRIBUTE_WEIGHTS)
    weights = [ATTRIBUTE_WEIGHTS[a] for a in pool]
    picked: List[str] = []
    while len(picked) < min(k, len(pool)):
        choice = rng.choices(pool, weights=weights, k=1)[0]
        if choice not in picked:
            picked.append(choice)
    return picked


def compose_corpus(
    n: int,
    seed: int = 42,
    noop_ratio: float = 0.10,
    banks_dir: Optional[Path] = None,
    carriers_path: Optional[Path] = None,
) -> List[AugmentedExample]:
    """Compose `n` unique augmented examples with balanced positions.

    Position distribution is uniform over POSITIONS (20% each for the
    persona-bearing share); attribute counts follow ATTRIBUTE_WEIGHTS;
    `noop_ratio` of examples carry no persona at all (identity targets).
    """
    rng = random.Random(seed)
    composer = PersonaComposer(banks_dir=banks_dir, carriers_path=carriers_path)
    examples: List[AugmentedExample] = []
    seen_inputs: Set[str] = set()
    attempts = 0
    max_attempts = n * 40
    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        idx = len(examples)
        carrier = rng.choice(composer.carriers)
        if rng.random() < noop_ratio:
            example = composer.compose_noop(carrier, f"aug_{idx:05d}")
        else:
            position = POSITIONS[idx % len(POSITIONS)]
            n_attrs = rng.choices([1, 2, 3], weights=[0.55, 0.33, 0.12], k=1)[0]
            if position == "split":
                n_attrs = max(2, n_attrs)
            attributes = _weighted_attributes(rng, n_attrs)
            example = composer.compose(carrier, position, attributes, rng, f"aug_{idx:05d}")
        if example.input in seen_inputs:
            continue
        seen_inputs.add(example.input)
        examples.append(example)
    return examples


def verify_targets(examples: Sequence[AugmentedExample]) -> List[str]:
    """Return ids of examples whose target still contains an injected surface."""
    bad = []
    for ex in examples:
        for f in ex.injected:
            surfaces = [f.surface] + [v for v in f.variants if v]
            if any(_phrase_in(ex.target, s) for s in surfaces if s):
                bad.append(ex.id)
                break
    return bad


def write_corpus(examples: Sequence[AugmentedExample], out_path: Path) -> Dict:
    """Write examples as JSONL plus a distribution manifest next to them."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_record(), ensure_ascii=False) + "\n")
    positions: Dict[str, int] = {}
    attrs: Dict[str, int] = {}
    for ex in examples:
        positions[ex.position] = positions.get(ex.position, 0) + 1
        for a in ex.attributes:
            attrs[a] = attrs.get(a, 0) + 1
    leaked = verify_targets(examples)
    stats = {
        "rows": len(examples),
        "position_distribution": positions,
        "attribute_distribution": attrs,
        "noop_rows": sum(1 for ex in examples if not ex.attributes),
        "target_leaks": len(leaked),
        "leaked_ids": leaked[:20],
        "note": "targets verified free of every injected surface form",
    }
    (out_path.parent / "augment_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000, help="Number of examples to compose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noop-ratio", type=float, default=0.10)
    parser.add_argument("--out", default="data/sft/augmented_v1/augmented.jsonl")
    args = parser.parse_args(argv)
    examples = compose_corpus(args.n, seed=args.seed, noop_ratio=args.noop_ratio)
    stats = write_corpus(examples, Path(args.out))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
