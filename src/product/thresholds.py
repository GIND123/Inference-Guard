"""Choosing the LOW / MEDIUM / HIGH cut points from validation data.

Every report so far has carried the same caveat: the 0.30 and 0.60 band edges
in `risk_bands.py` are inherited from the Session 04 notebook and were never
derived from anything. This module derives them, and it is deliberately
separate from the model-side calibration in `src/risk_model/`: that converts
logits into trustworthy probabilities, this decides what a probability is
allowed to say to a user. The second question is a product question and the
answer differs per attribute.

WHY ONE GLOBAL PAIR OF EDGES CANNOT BE RIGHT
--------------------------------------------
The attributes have wildly different base rates in SynthPAI -- occupation is
inferable in 22.1% of comments, location in 4.3%. A classifier trained with
`pos_weight` near 20x on location and near 3.5x on occupation produces score
distributions that are not on the same scale, so a single 0.60 cut means
"probably risky" for one attribute and "barely above noise" for another.
Thresholds are therefore fitted per attribute.

WHAT THE BANDS PROMISE THE USER
-------------------------------
The two edges answer two different questions, so they are fitted against two
different criteria rather than one balanced score:

  HIGH   is a claim we are asserting. Its cost is a false alarm: a user who is
         warned about nothing repeatedly stops reading warnings, and the
         product is worthless after that. Fitted for PRECISION.

  LOW    is a reassurance, and its cost is the opposite and worse -- telling
         someone their message is fine when it is not. Fitted for RECALL, so
         that few genuinely risky messages land below the edge.

MEDIUM is the honest middle: everything we cannot confidently call either way.
It is expected to be wide, and a wide MEDIUM band is a truthful description of
a model that is not very good yet rather than a failure of this code.

Fit on validation, never on test. `fit_thresholds` will refuse duplicate or
obviously out-of-range inputs but it cannot tell which split you handed it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .risk_bands import ATTRIBUTES, Band, CUE_IMPORTANCE_FLOOR, MEDIUM_EDGE

# Defaults chosen for a warning surface, not for a leaderboard.
# 0.60 precision on HIGH means roughly two in five HIGH warnings are wrong,
# which is poor but is the level this model can currently support; raising it
# simply empties the band. 0.90 recall on LOW means we accept missing one
# risky message in ten. Both are arguments, not facts -- change them here and
# the numbers move everywhere, which is the point of having one place.
DEFAULT_HIGH_PRECISION = 0.60
DEFAULT_LOW_RECALL = 0.90

# If the band edges collapse together the three-band UI is a lie, so we keep
# them apart and say so in the fit result rather than silently emitting
# high_edge == medium_edge.
MIN_BAND_WIDTH = 0.05

# Validated product threshold defaults for runtime consumers (API, rewriter, rejection sampling).
VALIDATED_MAX_RISK_THRESHOLD: float = MEDIUM_EDGE
VALIDATED_MIN_COSINE_THRESHOLD: float = 0.30
VALIDATED_CUE_IMPORTANCE_FLOOR: float = CUE_IMPORTANCE_FLOOR


def get_validated_thresholds() -> dict[str, float]:
    """Return dictionary of runtime validated thresholds for inference services."""
    return {
        "max_risk_threshold": VALIDATED_MAX_RISK_THRESHOLD,
        "min_cosine_threshold": VALIDATED_MIN_COSINE_THRESHOLD,
        "cue_importance_floor": VALIDATED_CUE_IMPORTANCE_FLOOR,
    }


@dataclass(frozen=True)
class ThresholdFit:
    """Fitted edges for one attribute, with the evidence behind them."""

    attribute: str
    medium_edge: float
    high_edge: float
    # what the edges actually achieved on the validation set
    high_precision: float
    high_recall: float
    low_recall: float          # share of positives correctly kept out of LOW
    n: int
    n_positive: int
    # True only when a fitting target was unreachable, which is what makes a
    # fit untrustworthy. Routine adjustments go in `note` and do not set this.
    degraded: bool = False
    note: str = ""

    @property
    def usable(self) -> bool:
        """False when the fit is too thin or too degraded to show a user.

        A band derived from a handful of positives is a number with an
        interval so wide it should not drive a coloured badge. Location has
        roughly 50 positives in a 15% split, so this will fire.
        """
        return self.n_positive >= 30 and not self.degraded


def _precision_recall_at(
    y_true: Sequence[int], y_prob: Sequence[float], threshold: float
) -> tuple[float, float]:
    """Precision and recall for `prob >= threshold` as the positive call."""
    tp = fp = fn = 0
    for t, p in zip(y_true, y_prob):
        predicted = p >= threshold
        if predicted and t:
            tp += 1
        elif predicted and not t:
            fp += 1
        elif not predicted and t:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _candidates(y_prob: Sequence[float]) -> list[float]:
    """Thresholds worth testing: the distinct scores, plus the endpoints.

    Sweeping a fixed grid misses the cut that matters when scores are bunched,
    which is exactly what a heavily weighted classifier produces.
    """
    return sorted({0.0, *(round(p, 6) for p in y_prob), 1.0})


def fit_attribute(
    attribute: str,
    y_true: Sequence[int],
    y_prob: Sequence[float],
    high_precision: float = DEFAULT_HIGH_PRECISION,
    low_recall: float = DEFAULT_LOW_RECALL,
) -> ThresholdFit:
    """Fit the two edges for one attribute on validation scores.

    `high_edge` is the *lowest* threshold that still meets the precision
    target -- lowest because, having decided how often we are willing to be
    wrong, we want to catch as much as possible at that error rate.

    `medium_edge` is the *highest* threshold that still meets the recall
    target, for the mirror reason: having decided how much we are willing to
    miss, we want the LOW band as wide as that allows, so the product stays
    quiet when it can.
    """
    if len(y_true) != len(y_prob):
        raise ValueError(f"{attribute}: {len(y_true)} labels vs {len(y_prob)} scores")
    if not y_true:
        raise ValueError(f"{attribute}: no validation data")
    if any(not 0.0 <= p <= 1.0 for p in y_prob):
        raise ValueError(f"{attribute}: scores must be probabilities in [0, 1]")

    n_pos = sum(1 for t in y_true if t)
    if n_pos == 0:
        return ThresholdFit(
            attribute, 0.30, 0.60, 0.0, 0.0, 0.0, len(y_true), 0,
            degraded=True, note="no positive examples in validation",
        )

    cands = _candidates(y_prob)
    notes: list[str] = []
    degraded = False  # a target was unreachable -- distinct from an adjustment

    # HIGH: lowest threshold meeting the precision target.
    high_edge = None
    for t in cands:
        prec, _ = _precision_recall_at(y_true, y_prob, t)
        if prec >= high_precision:
            high_edge = t
            break
    if high_edge is None:
        # The model never reaches the target at any cut. Fall back to the most
        # precise cut available and mark it, rather than inventing an edge.
        high_edge = max(cands, key=lambda t: _precision_recall_at(y_true, y_prob, t)[0])
        notes.append(f"never reaches {high_precision:.0%} precision at any threshold")
        degraded = True

    # LOW/MEDIUM: highest threshold still meeting the recall target.
    medium_edge = 0.0
    for t in cands:
        _, rec = _precision_recall_at(y_true, y_prob, t)
        if rec >= low_recall:
            medium_edge = t
        else:
            break
    if medium_edge == 0.0:
        notes.append(f"cannot reach {low_recall:.0%} recall above 0.0; LOW band empty")
        degraded = True

    if high_edge - medium_edge < MIN_BAND_WIDTH:
        # Widen downward, never upward. Raising high_edge would abandon the
        # precision target we just fitted it to, and on separable scores it
        # empties the HIGH band entirely. Lowering medium_edge instead keeps
        # the HIGH promise and errs toward warning rather than reassuring.
        medium_edge = max(0.0, high_edge - MIN_BAND_WIDTH)
        notes.append("edges were within %.2f; LOW edge lowered to keep bands distinct"
                     % MIN_BAND_WIDTH)

    hp, hr = _precision_recall_at(y_true, y_prob, high_edge)
    _, mr = _precision_recall_at(y_true, y_prob, medium_edge)

    return ThresholdFit(
        attribute=attribute,
        medium_edge=round(medium_edge, 4),
        high_edge=round(high_edge, 4),
        high_precision=round(hp, 4),
        high_recall=round(hr, 4),
        low_recall=round(mr, 4),
        n=len(y_true),
        n_positive=n_pos,
        degraded=degraded,
        note="; ".join(notes),
    )


def fit_thresholds(
    y_true: Mapping[str, Sequence[int]],
    y_prob: Mapping[str, Sequence[float]],
    **kwargs,
) -> dict[str, ThresholdFit]:
    """Fit every attribute present in both mappings."""
    shared = [a for a in ATTRIBUTES if a in y_true and a in y_prob]
    if not shared:
        raise ValueError(f"no attribute in both inputs; got {sorted(y_true)} / {sorted(y_prob)}")
    return {a: fit_attribute(a, y_true[a], y_prob[a], **kwargs) for a in shared}


def band_for(score: float, fit: ThresholdFit) -> Band:
    """Band a score using fitted edges instead of the global defaults."""
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"risk score must be in [0, 1], got {score!r}")
    if score >= fit.high_edge:
        return Band.HIGH
    if score >= fit.medium_edge:
        return Band.MEDIUM
    return Band.LOW


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], bins: int = 10
) -> float:
    """ECE: mean gap between claimed confidence and observed frequency.

    Here because the bands are only meaningful if the probabilities under them
    mean something. A model with good macro F1 and an ECE of 0.3 will produce
    bands that are ranked correctly and labelled wrongly, and the label is
    what the user reads.
    """
    if len(y_true) != len(y_prob):
        raise ValueError("labels and scores differ in length")
    if not y_true:
        raise ValueError("no data")

    total = len(y_true)
    error = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (t, p) for t, p in zip(y_true, y_prob)
            if (lo <= p < hi) or (b == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        confidence = sum(p for _, p in members) / len(members)
        accuracy = sum(t for t, _ in members) / len(members)
        error += (len(members) / total) * abs(confidence - accuracy)
    return round(error, 4)


def summarize(fits: Mapping[str, ThresholdFit]) -> str:
    """A table for the report. Prints the caveats, not just the numbers."""
    lines = [
        f"{'attribute':<12}{'LOW <':>8}{'HIGH >=':>9}{'HIGH prec':>11}"
        f"{'HIGH rec':>10}{'n+':>6}  notes",
    ]
    for a, f in fits.items():
        flag = "" if f.usable else "  [not usable]"
        lines.append(
            f"{a:<12}{f.medium_edge:>8.2f}{f.high_edge:>9.2f}"
            f"{f.high_precision:>11.2f}{f.high_recall:>10.2f}{f.n_positive:>6}"
            f"  {f.note}{flag}"
        )
    return "\n".join(lines)
