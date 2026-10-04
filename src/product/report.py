"""The privacy report a user actually sees, and the rules it must obey.

`risk_bands.py` turns per-attribute scores into bands. This composes a band
summary with the explicit-PII findings into the one object the UI renders and
the one the evaluation harness compares before and after. It exists so the
product's honesty rules live in code with tests around them, rather than in a
design document that a template can quietly contradict.

FOUR RULES, EACH FROM SOMETHING THAT WENT WRONG
-----------------------------------------------
1. EXPLICIT AND INFERENTIAL RISK NEVER COMBINE INTO ONE NUMBER.
   Session 04 reported `confidence = max(risk_scores)` with `name` among the
   dimensions, so the headline was dominated by name detection -- work
   Presidio already does and the baseline we are supposed to be beating.
   Merging them credits our inferential system for the baseline's results.
   They are two fields here and there is no method that averages them.

2. REMAINING EXPLICIT PII OVERRIDES ANY INFERENTIAL IMPROVEMENT.
   The resume case in `data/logs/session04_usage_log.csv`: location fell
   0.859 -> 0.006 while the protected text still read
   `John Doe | ... | jdoe@example.com | (303) 555-0192`. Every aggregate we
   had called that a success. A rewrite that leaves a name, an email and a
   phone number has failed, whatever happened to the inferential scores, and
   `Comparison.verdict` says so.

3. UTILITY IS A GUARDRAIL, NOT A SEPARATE SCORE TO ADMIRE.
   Deleting the message is perfect privacy. A privacy gain below the utility
   floor is reported as a cost, not a win.

4. WE NEVER SAY SAFE, CLEAN, OR ANONYMOUS.
   The project cannot guarantee anonymity, and a user who over-trusts the
   tool and therefore shares more is worse off than one with no tool at all.
   That is the highest-cost failure in the lean canvas and it is a product
   failure, not a model failure. `FORBIDDEN_CLAIMS` is asserted in tests.

Nothing here stores message text. Reports travel into logs and evidence
files, and a public repo is the wrong place for what a participant typed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Literal, Mapping

from .risk_bands import ATTRIBUTES, Band, Cue, RiskSummary, summarize

# Presidio entity types that are unambiguously identifying on their own. A
# rewrite that leaves one of these behind has failed rule 2, regardless of
# what the inferential scores did.
HARD_PII_TYPES: frozenset[str] = frozenset({
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE",
    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "MEDICAL_LICENSE",
    "IP_ADDRESS", "CRYPTO",
})

# Words the product must never use about a protected message. Checked in
# tests against every string this module can emit.
FORBIDDEN_CLAIMS: tuple[str, ...] = ("anonymous", "anonymised", "anonymized", "safe to send", "fully protected")

# Below this, the rewrite has changed the meaning enough that a privacy gain
# is not worth having. Provisional, like every other threshold here -- see
# thresholds.py for the argument about deriving these from data.
DEFAULT_UTILITY_FLOOR = 0.70

Verdict = Literal["improved", "improved_with_cost", "explicit_pii_remains", "no_change", "worse"]


@dataclass(frozen=True)
class PiiSpan:
    """One explicit identifier located by Presidio. Offsets, never the text."""

    entity_type: str
    start: int
    end: int
    score: float = 1.0

    @property
    def is_hard(self) -> bool:
        return self.entity_type.upper() in HARD_PII_TYPES


@dataclass(frozen=True)
class ExplicitFindings:
    """The traditional-PII half of the report. Deliberately its own object."""

    spans: tuple[PiiSpan, ...] = ()

    @property
    def count(self) -> int:
        return len(self.spans)

    @property
    def hard_count(self) -> int:
        return sum(1 for s in self.spans if s.is_hard)

    @property
    def by_type(self) -> dict[str, int]:
        return dict(Counter(s.entity_type for s in self.spans))


@dataclass(frozen=True)
class PrivacyReport:
    """One message, assessed. Two halves that never merge."""

    explicit: ExplicitFindings
    inferential: RiskSummary

    @property
    def inferential_overall(self) -> float:
        """Worst attribute. Never includes explicit PII -- see rule 1."""
        return self.inferential.overall

    @property
    def inferential_band(self) -> Band:
        return self.inferential.overall_band

    def to_dict(self) -> dict:
        return {
            "explicit": {
                "count": self.explicit.count,
                "hard_count": self.explicit.hard_count,
                "by_type": self.explicit.by_type,
                "spans": [
                    {"type": s.entity_type, "start": s.start, "end": s.end}
                    for s in self.explicit.spans
                ],
            },
            "inferential": self.inferential.to_dict(),
        }


def compose(
    pii_spans: Iterable[PiiSpan | Mapping[str, object]] = (),
    risk_scores: Mapping[str, float] | None = None,
    cues: Iterable[Cue | Mapping[str, object]] = (),
    leakage_delta: float | None = None,
) -> PrivacyReport:
    """Build a report from a Presidio result and a risk-model result.

    `name` is dropped from `risk_scores` if present. It is explicit PII and
    belongs on the other side of the report; leaving it in is how the Session
    04 headline came to be a name detector. `ATTRIBUTES` does not contain it,
    so this is belt and braces against a caller passing the raw dict.
    """
    spans = tuple(s if isinstance(s, PiiSpan) else PiiSpan(**s) for s in pii_spans)  # type: ignore[arg-type]
    scores = {a: v for a, v in (risk_scores or {}).items() if a in ATTRIBUTES}
    if not scores:
        raise ValueError(
            f"no inferential scores for any of {ATTRIBUTES}; "
            f"got {sorted((risk_scores or {}))}"
        )
    return PrivacyReport(
        explicit=ExplicitFindings(spans=spans),
        inferential=summarize(scores, cues=cues, leakage_delta=leakage_delta),
    )


@dataclass(frozen=True)
class Comparison:
    """Before and after, with a verdict that cannot be gamed by one attribute."""

    verdict: Verdict
    inferential_before: float
    inferential_after: float
    inferential_delta: float
    per_attribute_delta: Mapping[str, float]
    explicit_before: int
    explicit_after: int
    hard_pii_remaining: int
    utility: float | None = None
    utility_floor: float = DEFAULT_UTILITY_FLOOR
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_win(self) -> bool:
        return self.verdict == "improved"


def compare(
    before: PrivacyReport,
    after: PrivacyReport,
    utility: float | None = None,
    utility_floor: float = DEFAULT_UTILITY_FLOOR,
) -> Comparison:
    """Grade a rewrite. Order of checks is the product's priority order.

    Explicit PII is checked first and overrides everything, because no
    inferential improvement compensates for leaving a phone number in.
    """
    shared = sorted(set(before.inferential.scores) & set(after.inferential.scores))
    if not shared:
        raise ValueError("before and after share no attributes")

    deltas = {
        a: round(before.inferential.scores[a] - after.inferential.scores[a], 4)
        for a in shared
    }
    overall_delta = round(before.inferential_overall - after.inferential_overall, 4)
    hard_remaining = after.explicit.hard_count
    notes: list[str] = []

    if hard_remaining:
        best = max(deltas, key=lambda a: deltas[a])
        if deltas[best] > overall_delta + 0.05:
            notes.append(
                f"{best} fell {deltas[best]:.3f} but overall moved {overall_delta:.3f}; "
                "reporting the per-attribute figure alone would misdescribe this"
            )
        verdict: Verdict = "explicit_pii_remains"
        notes.insert(0, f"{hard_remaining} identifying span(s) still present after rewrite")
    elif overall_delta <= 0.01:
        verdict = "no_change" if overall_delta >= -0.01 else "worse"
    elif utility is not None and utility < utility_floor:
        verdict = "improved_with_cost"
        notes.append(
            f"utility {utility:.2f} is below the {utility_floor:.2f} floor; "
            "the rewrite changed the meaning enough to matter"
        )
    else:
        verdict = "improved"

    return Comparison(
        verdict=verdict,
        inferential_before=before.inferential_overall,
        inferential_after=after.inferential_overall,
        inferential_delta=overall_delta,
        per_attribute_delta=deltas,
        explicit_before=before.explicit.count,
        explicit_after=after.explicit.count,
        hard_pii_remaining=hard_remaining,
        utility=utility,
        utility_floor=utility_floor,
        notes=tuple(notes),
    )


_VERDICT_TEXT = {
    "improved": "Inference risk reduced.",
    "improved_with_cost": "Inference risk reduced, but the meaning changed.",
    "explicit_pii_remains": "Not protected — identifying details are still present.",
    "no_change": "No meaningful change in inference risk.",
    "worse": "This rewrite increased inference risk.",
}


def render(report: PrivacyReport) -> str:
    """The risk panel, as text. Phrased as attacker inference throughout."""
    lines: list[str] = []

    e = report.explicit
    if e.count:
        types = ", ".join(f"{t} x{n}" if n > 1 else t for t, n in sorted(e.by_type.items()))
        lines.append(f"Explicit identifiers found: {e.count} ({types})")
    else:
        lines.append("Explicit identifiers found: none")

    lines.append(
        f"Inference risk: {report.inferential_band.value} "
        f"(worst attribute: {report.inferential.primary})"
    )
    for attr in sorted(report.inferential.scores, key=lambda a: -report.inferential.scores[a]):
        band = report.inferential.bands[attr].value
        lines.append(f"  {attr:<12}{report.inferential.scores[attr]:.2f}  {band}")

    if report.inferential.cues:
        spans = ", ".join(f'"{c.span}"' for c in report.inferential.cues)
        lines.append(f"Cues an attacker could use: {spans}")

    if report.inferential.leakage_delta is not None:
        lines.append(
            f"This message changes conversation leakage by "
            f"{report.inferential.leakage_delta:+.2f}"
        )
    return "\n".join(lines)


def render_comparison(comparison: Comparison) -> str:
    """Before/after, led by the verdict rather than by the best number."""
    lines = [_VERDICT_TEXT[comparison.verdict]]
    lines.append(
        f"Inference risk {comparison.inferential_before:.2f} -> "
        f"{comparison.inferential_after:.2f} "
        f"({comparison.inferential_delta:+.2f} overall, worst attribute)"
    )
    for attr, d in sorted(comparison.per_attribute_delta.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {attr:<12}{d:+.3f}")
    lines.append(
        f"Explicit identifiers {comparison.explicit_before} -> {comparison.explicit_after}"
    )
    if comparison.utility is not None:
        lines.append(f"Meaning preserved: {comparison.utility:.2f}")
    lines.extend(f"Note: {n}" for n in comparison.notes)
    # Rule 4. Said on every comparison, including the good ones.
    lines.append(
        "This reduces what a model can infer. It does not make you unidentifiable."
    )
    return "\n".join(lines)
