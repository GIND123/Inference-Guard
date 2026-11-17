"""Multi-turn dynamic privacy leakage state tracking for conversational sessions.

Tracks cumulative attribute disclosure across 3 tiers:
1. Independent evidence accumulation across turns.
2. Joint quasi-identifier entropy reduction.
3. Contextual concatenated inference and leakage_delta calculation.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.product.risk_bands import ATTRIBUTES, band_for

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
    residual_anonymity_capacity: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        d = asdict(self)
        d["residual_anonymity_capacity"] = self.residual_anonymity_capacity
        return d



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
        # session_id -> wall-clock timestamp of the last add_turn, for TTL / LRU
        # eviction so the in-memory tracker cannot grow unbounded. Mirrors the
        # session_timestamps pattern in src/pipeline/orchestrator.py.
        self.session_last_seen: Dict[str, float] = {}

    def _compute_cumulative_scores(
        self,
        history: Sequence[TurnRiskRecord],
        current_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute cumulative probabilities with Bayesian evidence gating.

        Bayesian Evidence Gating:
        Background noise at or below NOISE_FLOOR (P_t <= 0.05) represents model baseline
        uncertainty rather than genuine disclosure evidence. Gating this baseline out
        prevents 20 innocent conversational turns from compounding into an unearned breach
        alarm (1 - (1 - 0.05)^20 = 0.642).
        """
        NOISE_FLOOR = 0.05
        cumulative: Dict[str, float] = {}

        for attr in ATTRIBUTES:
            curr_score = current_scores.get(attr, 0.0)
            eff_curr = curr_score if curr_score > NOISE_FLOOR else 0.0
            prod_complement = 1.0 - eff_curr

            for prev_turn in history:
                prev_score = prev_turn.turn_scores.get(attr, 0.0)
                eff_prev = prev_score if prev_score > NOISE_FLOOR else 0.0
                prod_complement *= (1.0 - eff_prev)

            cum_val = 1.0 - prod_complement
            # If no active evidence above noise floor was ever disclosed, retain max baseline noise level
            if cum_val == 0.0:
                cum_val = max(
                    curr_score,
                    max((prev.turn_scores.get(attr, 0.0) for prev in history), default=0.0),
                )
            cumulative[attr] = round(max(0.0, min(1.0, cum_val)), 4)

        return cumulative

    def _compute_residual_anonymity_capacity(self, cumulative_scores: Dict[str, float]) -> float:
        """Compute Residual Anonymity Capacity (RAC) across the 4 quasi-identifiers.

        Reframed from 'joint entropy' to prevent mathematical conflation with Shannon entropy.
        Each undisclosed quasi-identifier attribute contributes 1.0 unit of anonymity capacity.
        As cumulative exposure probability P_cum(A) rises towards 1.0, the residual capacity
        collapses from 4.0 towards 0.0.
        """
        capacity = 0.0
        for attr in ATTRIBUTES:
            p_disclosed = cumulative_scores.get(attr, 0.0)
            uncertainty = max(0.0, 1.0 - p_disclosed)
            capacity += uncertainty

        return round(float(capacity), 4)

    def _compute_joint_entropy(self, cumulative_scores: Dict[str, float]) -> float:
        """Backward-compatible alias for _compute_residual_anonymity_capacity."""
        return self._compute_residual_anonymity_capacity(cumulative_scores)

    def add_turn(
        self,
        raw_text: str,
        session_id: str = "default",
        now: Optional[float] = None,
    ) -> TurnRiskRecord:
        """Process a new conversational turn and compute updated dynamic risk metrics.

        Args:
            raw_text: User message text in the conversation.
            session_id: Session identifier.
            now: Optional wall-clock timestamp (seconds) recorded as the session's
                last-activity time. Defaults to ``time.time()``; injectable so tests
                can drive TTL / LRU eviction without real sleeps.

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

        # Cumulative scores with Bayesian evidence gating
        cumulative_scores = self._compute_cumulative_scores(history, turn_scores)
        overall_cumulative_risk = max(cumulative_scores.values()) if cumulative_scores else 0.0

        # Residual Anonymity Capacity (formerly joint entropy)
        residual_capacity = self._compute_residual_anonymity_capacity(cumulative_scores)
        joint_entropy = residual_capacity

        # Leakage delta relative to previous turn
        if history:
            prev_overall = history[-1].overall_cumulative_risk
            leakage_delta = round(overall_cumulative_risk - prev_overall, 4)
        else:
            leakage_delta = round(overall_cumulative_risk, 4)

        # Breach alert condition: cumulative risk >= high_risk_threshold or capacity < threshold
        is_breached = bool(
            overall_cumulative_risk >= self.high_risk_threshold
            or residual_capacity < self.entropy_threshold
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
            residual_anonymity_capacity=residual_capacity,
        )

        self.sessions[session_id].append(record)
        self.session_last_seen[session_id] = time.time() if now is None else now
        return record

    def get_history(self, session_id: str = "default") -> List[TurnRiskRecord]:
        """Return the sequence of turn records for a session."""
        return list(self.sessions.get(session_id, []))

    def reset_session(self, session_id: str = "default") -> None:
        """Reset history for a session (backward-compatible)."""
        self.sessions.pop(session_id, None)
        self.session_last_seen.pop(session_id, None)

    def delete_session(self, session_id: str = "default") -> bool:
        """Delete all tracked state for a session.

        Clears the session's conversation/turn/risk history and its last-seen
        timestamp. Idempotent: returns ``True`` when state was present and
        removed, ``False`` when the session was unknown. Standardizes on the
        same removal as :meth:`reset_session`, which remains for backward
        compatibility.
        """
        existed = session_id in self.sessions or session_id in self.session_last_seen
        self.reset_session(session_id)
        return existed

    def evict_idle(
        self,
        ttl_seconds: Optional[float] = None,
        max_sessions: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[str]:
        """Reap sessions so the in-memory tracker cannot grow unbounded.

        Two cheap dict-scan passes (no background thread), mirroring the TTL
        cleanup pattern in src/pipeline/orchestrator.py:

        1. TTL: remove any session whose last activity is older than
           ``ttl_seconds`` (when provided and positive).
        2. Cap: if more than ``max_sessions`` remain, LRU-evict the
           least-recently-seen sessions until the count is within the cap.

        Args:
            ttl_seconds: Idle time-to-live in seconds; ``None``/non-positive
                disables the TTL pass.
            max_sessions: Maximum number of sessions to retain; ``None``/
                non-positive disables the cap.
            now: Optional current timestamp (seconds); defaults to
                ``time.time()``. Injectable so tests avoid real sleeps.

        Returns:
            The list of evicted session ids (TTL-expired first, then LRU).
        """
        current = time.time() if now is None else now
        evicted: List[str] = []

        # 1. TTL pass.
        if ttl_seconds is not None and ttl_seconds > 0:
            expired = [
                sid
                for sid, last_seen in self.session_last_seen.items()
                if current - last_seen > ttl_seconds
            ]
            for sid in expired:
                self.reset_session(sid)
                evicted.append(sid)

        # 2. Max-session cap via LRU eviction.
        if max_sessions is not None and max_sessions > 0:
            overflow = len(self.session_last_seen) - max_sessions
            if overflow > 0:
                # Oldest last-seen first (least recently used).
                lru_order = sorted(
                    self.session_last_seen.items(), key=lambda kv: kv[1]
                )
                for sid, _ in lru_order[:overflow]:
                    self.reset_session(sid)
                    evicted.append(sid)

        return evicted


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
