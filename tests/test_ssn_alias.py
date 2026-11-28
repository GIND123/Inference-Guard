"""Dashed US SSNs are aliased like phone numbers; look-alike numbers are left alone."""

from src.contracts.schemas import ConversationState
from src.conversation.entity_map import DeterministicPseudonymManager


def _run(text, referents_only):
    mgr = DeterministicPseudonymManager(referents_only=referents_only)
    state = ConversationState(conversation_id="ssn_unit")
    out, spans, counts = mgr.pseudonymize_turn(text, state, session_secret="s" * 64, blind_index=True)
    return out, spans, counts


def test_dashed_ssn_is_aliased_in_both_modes():
    text = "My email is john.doe@example.com and SSN 123-45-6789."
    for referents_only in (True, False):
        out, spans, counts = _run(text, referents_only)
        assert "123-45-6789" not in out
        assert "SSN_1" in out
        assert counts["US_SSN"] == 1
        assert any(s.original_surface == "123-45-6789" for s in spans)


def test_ssn_lookalikes_are_not_aliased():
    for text in ("Order 123456789 shipped.", "Case 2024-12-3456-7 is open.", "Ref 1123-45-67890."):
        out, _, counts = _run(text, referents_only=False)
        assert "SSN_" not in out, text
        assert counts["US_SSN"] == 0
