"""Multi-turn dynamic privacy leakage state tracking for conversational sessions.

Tracks cumulative attribute disclosure across 3 tiers:
1. Independent evidence accumulation across turns.
2. Joint quasi-identifier entropy reduction.
3. Contextual concatenated inference and leakage_delta calculation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.product.risk_bands import ATTRIBUTES, Band, band_for

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnRiskRecord:
    """Detailed record of privacy risk for a single dialogue turn."""

    turn_number: int
    session_id: str
    raw_text: str
    turn_scores: Dict[str, float]
    cumulative_scores: Dict[str, float]
    overall_turn_risk: float
    overall_cumulative_risk: float
    joint_entropy: float
    leakage_delta: float
    is_breached: bool
    risk_band: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


def estimate_turn_scores(text: str) -> Dict[str, float]:
    """Estimate attribute disclosure probability for a single turn."""
    from src.rewriter.generate_training_data import estimate_privacy_risk
    res = estimate_privacy_risk(text)
    return {a: res.get(a, 0.05) for a in ATTRIBUTES}


class ConversationTracker:
    """Tracks multi-turn dynamic privacy leakage across conversation turns."""

    def __init__(
        self,
        risk_estimator_fn: Optional[Callable[[str], Dict[str, float]]] = None,
        entropy_threshold: float = 1.50,
        high_risk_threshold: float = 0.60,
    ) -> None:
        self.risk_estimator_fn = risk_estimator_fn or estimate_turn_scores
        self.entropy_threshold = entropy_threshold
        self.high_risk_threshold = high_risk_threshold

        # session_id -> list of TurnRiskRecord
        self.sessions: Dict[str, List[TurnRiskRecord]] = {}

    def _compute_cumulative_scores(
        self,
        history: Sequence[TurnRiskRecord],
        current_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute cumulative probabilities: P_cum(A) = 1 - prod_{t=1}^T (1 - P_t(A))."""
        cumulative: Dict[str, float] = {}

        for attr in ATTRIBUTES:
            prod_complement = 1.0 - current_scores.get(attr, 0.0)
            for prev_turn in history:
                prev_score = prev_turn.turn_scores.get(attr, 0.0)
                prod_complement *= (1.0 - prev_score)

            cum_val = 1.0 - prod_complement
            cumulative[attr] = round(max(0.0, min(1.0, cum_val)), 4)

        return cumulative

    def _compute_joint_entropy(self, cumulative_scores: Dict[str, float]) -> float:
        """Compute remaining anonymity entropy across the 4 quasi-identifiers.

        Each undisclosed attribute contributes 1.0 bit of uncertainty.
        As cumulative exposure probability P_cum(A) rises towards 1.0,
        the attacker's uncertainty regarding the author's identity collapses towards 0.
        """
        entropy = 0.0
        for attr in ATTRIBUTES:
            p_disclosed = cumulative_scores.get(attr, 0.0)
            uncertainty = max(0.0, 1.0 - p_disclosed)
            entropy += uncertainty

        return round(float(entropy), 4)

    def add_turn(self, raw_text: str, session_id: str = "default") -> TurnRiskRecord:
        """Process a new conversational turn and compute updated dynamic risk metrics.

        Args:
            raw_text: User message text in the conversation.
            session_id: Session identifier.

        Returns:
            TurnRiskRecord containing turn scores, cumulative scores, joint entropy, and leakage_delta.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        history = self.sessions[session_id]
        turn_num = len(history) + 1

        # Turn level scores
        turn_scores = self.risk_estimator_fn(raw_text)
        overall_turn_risk = max(turn_scores.values()) if turn_scores else 0.0

        # Cumulative scores
        cumulative_scores = self._compute_cumulative_scores(history, turn_scores)
        overall_cumulative_risk = max(cumulative_scores.values()) if cumulative_scores else 0.0

        # Joint entropy
        joint_entropy = self._compute_joint_entropy(cumulative_scores)

        # Leakage delta relative to previous turn
        if history:
            prev_overall = history[-1].overall_cumulative_risk
            leakage_delta = round(overall_cumulative_risk - prev_overall, 4)
        else:
            leakage_delta = round(overall_cumulative_risk, 4)

        # Breach alert condition: cumulative risk >= high_risk_threshold or entropy < threshold
        is_breached = bool(
            overall_cumulative_risk >= self.high_risk_threshold
            or joint_entropy < self.entropy_threshold
        )

        risk_band = band_for(overall_cumulative_risk).value

        record = TurnRiskRecord(
            turn_number=turn_num,
            session_id=session_id,
            raw_text=raw_text,
            turn_scores=turn_scores,
            cumulative_scores=cumulative_scores,
            overall_turn_risk=round(overall_turn_risk, 4),
            overall_cumulative_risk=round(overall_cumulative_risk, 4),
            joint_entropy=joint_entropy,
            leakage_delta=leakage_delta,
            is_breached=is_breached,
            risk_band=risk_band,
        )

        self.sessions[session_id].append(record)
        return record

    def get_history(self, session_id: str = "default") -> List[TurnRiskRecord]:
        """Return the sequence of turn records for a session."""
        return list(self.sessions.get(session_id, []))

    def reset_session(self, session_id: str = "default") -> None:
        """Reset history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]


# Backward-compatible lightweight SessionState and SessionManager
@dataclass
class Message:
    role: str
    content: str
    timestamp: float


@dataclass
class SessionState:
    session_id: str
    user_id: str
    status: str = "active"
    messages: List[Message] = field(default_factory=list)


class SessionManager:
    """Manages local sessions and persistent pseudonyms (backward compatibility)."""

    def __init__(self, storage_dir: str = ".state"):
        self.storage_dir = storage_dir
        self.sessions_file = os.path.join(storage_dir, "sessions.json")
        self.sessions: Dict[str, SessionState] = {}
        os.makedirs(self.storage_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.sessions_file):
            try:
                with open(self.sessions_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        messages = [Message(**m) for m in v.get("messages", [])]
                        v["messages"] = messages
                        self.sessions[k] = SessionState(**v)
            except Exception:
                pass

    def _save(self) -> None:
        sessions_dict = {k: asdict(v) for k, v in self.sessions.items()}
        with open(self.sessions_file, "w", encoding="utf-8") as f:
            json.dump(sessions_dict, f, indent=2)

    def start_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionState(session_id=session_id, user_id=user_id)
        self._save()
        return session_id

    def end_session(self, session_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id].status = "ended"
            self._save()

    def add_message(self, session_id: str, role: str, content: str, timestamp: float) -> None:
        if session_id in self.sessions:
            msg = Message(role=role, content=content, timestamp=timestamp)
            self.sessions[session_id].messages.append(msg)
            self._save()
