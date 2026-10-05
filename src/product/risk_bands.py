"""User-facing interpretation of attribute risk scores.

The risk estimator emits a float in [0, 1] per attribute. Those floats are not
the product. What a user sees is a band, a primary risk, and a short list of
cues, and this module is the one place where that translation happens.

It exists because of a specific failure from Session 04. On the resume row of
``data/logs/session04_usage_log.csv`` the location score fell 0.859 -> 0.006
while overall confidence moved only 0.997 -> 0.987, because the rewrite scrubbed
"Denver CO" and left the name, email and phone in place. Quoting the location
delta on its own turned a failed rewrite into our best-looking result. So
``overall`` here is always the worst attribute -- never a mean, never one the
caller picked.

THRESHOLDS ARE PROVISIONAL. The 0.30 / 0.60 cuts are carried over unchanged from
the Session 04 notebook and have never been calibrated: no temperature scaling,
no reliability diagram, no ECE. They are placeholders so the UI has something
stable to render against, and they must be re-derived from validation data
before any of these bands appear in a report as a finding. See issue #17.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

# The four attribute families in scope for v1. Stretch attributes (income,
# relationship status, health) are deliberately absent -- adding one here
# without a trained head would silently report 0.0 as "LOW" rather than
# "not measured".
ATTRIBUTES: tuple[str, ...] = ("age", "location", "occupation", "education")


class Band(str, Enum):
    """Coarse risk label shown to the user instead of a raw probability."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Lower edge of each band. A score is MEDIUM at 0.30 and HIGH at 0.60.
MEDIUM_EDGE = 0.30
HIGH_EDGE = 0.60

# Only cues at or above this importance are worth showing; below it the
# highlight is noise and teaches users to ignore the highlighting.
CUE_IMPORTANCE_FLOOR = 0.20

# Showing more than this many highlighted spans at once makes the message
# unreadable and every span look equally guilty.
MAX_CUES_SHOWN = 5


def band_for(score: float) -> Band:
    """Map a single calibrated-ish risk score to its display band."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"risk score must be in [0, 1], got {score!r}")
    if score >= HIGH_EDGE:
        return Band.HIGH
    if score >= MEDIUM_EDGE:
        return Band.MEDIUM
    return Band.LOW


@dataclass(frozen=True)
class Cue:
    """A span the estimator leaned on, and which attribute it fed."""

    span: str
    attribute: str
    importance: float


@dataclass(frozen=True)
class RiskSummary:
    """Everything the UI needs for the input risk panel, and nothing else."""

    overall: float
    overall_band: Band
    primary: str | None
    secondary: tuple[str, ...]
    bands: Mapping[str, Band]
    scores: Mapping[str, float]
    cues: tuple[Cue, ...] = ()
    leakage_delta: float | None = None

    def to_dict(self) -> dict:
        """Structured form, matching the output contract in the design doc."""
        return {
            "overall": self.overall,
            "overall_band": self.overall_band.value,
            "primary": self.primary,
            "secondary": list(self.secondary),
            "bands": {a: b.value for a, b in self.bands.items()},
            "scores": dict(self.scores),
            "cues": [
                {"span": c.span, "attribute": c.attribute, "importance": c.importance}
                for c in self.cues
            ],
            "leakage_delta": self.leakage_delta,
        }


def summarize(
    scores: Mapping[str, float],
    cues: Iterable[Cue | Mapping[str, object]] = (),
    leakage_delta: float | None = None,
) -> RiskSummary:
    """Turn per-attribute scores into the one summary the UI and report share.

    ``overall`` is the maximum across attributes by design. An average would let
    three quiet attributes hide one loud one, which is exactly how the resume
    row in the Session 04 log came to look like a success.

    Attributes missing from ``scores`` are omitted rather than defaulted to 0.0,
    so "we did not measure this" never renders as "this is safe".
    """
    known = {a: float(s) for a, s in scores.items() if a in ATTRIBUTES}
    if not known:
        raise ValueError(f"no scores for any of {ATTRIBUTES}; got {sorted(scores)}")

    bands = {a: band_for(s) for a, s in known.items()}

    # Sort by descending score, then name, so ties render the same way twice.
    ranked = sorted(known.items(), key=lambda kv: (-kv[1], kv[0]))
    primary = ranked[0][0]
    overall = ranked[0][1]

    # Secondary risks are the remaining attributes that still clear MEDIUM.
    # Listing LOW attributes here would pad the warning and dilute the primary.
    secondary = tuple(a for a, s in ranked[1:] if band_for(s) is not Band.LOW)

    return RiskSummary(
        overall=overall,
        overall_band=band_for(overall),
        primary=primary,
        secondary=secondary,
        bands=bands,
        scores=known,
        cues=_top_cues(cues),
        leakage_delta=leakage_delta,
    )


def _top_cues(cues: Iterable[Cue | Mapping[str, object]]) -> tuple[Cue, ...]:
    """Drop weak cues and cap the rest, highest importance first."""
    normalized = [c if isinstance(c, Cue) else Cue(**c) for c in cues]  # type: ignore[arg-type]
    strong = [c for c in normalized if c.importance >= CUE_IMPORTANCE_FLOOR]
    strong.sort(key=lambda c: (-c.importance, c.span))
    return tuple(strong[:MAX_CUES_SHOWN])


def describe(summary: RiskSummary) -> str:
    """One-line plain-English warning.

    Phrasing is deliberate. The design doc's ethics section requires we say what
    an attacker could infer, not what is true about the user -- the model has no
    access to the latter and saying otherwise launders a guess into a fact.
    """
    if summary.overall_band is Band.LOW:
        return "No strong attribute inference detected in this message."

    attrs = [summary.primary, *summary.secondary]
    listed = attrs[0] if len(attrs) == 1 else ", ".join(attrs[:-1]) + f" and {attrs[-1]}"
    return (
        f"{summary.overall_band.value} inference risk: an attacker reading this "
        f"message could plausibly infer your {listed}."
    )


def format_delta(before: RiskSummary, after: RiskSummary) -> dict[str, float]:
    """Per-attribute before/after change, for the comparison panel.

    Returned per attribute rather than as a single number on purpose: a single
    averaged delta is what let one collapsing attribute mask three that did not
    move.
    """
    shared = set(before.scores) & set(after.scores)
    if not shared:
        raise ValueError("before and after summaries share no attributes")
    deltas = {a: round(before.scores[a] - after.scores[a], 4) for a in sorted(shared)}
    # The honest headline is the smallest improvement, not the largest.
    deltas["overall"] = round(before.overall - after.overall, 4)
    return deltas
