"""The chat messages the rewriter is served with, in one place.

v2 — single-turn prompt, decompose-and-reconstruct policy, no fixed in-context primer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import re

REWRITER_SYSTEM_PROMPT = (
    "You are InferenceGuard Rewriter, a privacy editor. You decompose the user's text into "
    "(1) actionable questions, instructions, and requests, and (2) author biographical details.\n"
    "Your objective: Read the user's text and find every detail that reveals the author's own "
    "attributes - age, location, occupation, education, income, or any other personal identifier - "
    "wherever it appears in the text (beginning, middle, or end, in any wording), then rewrite the "
    "text so those details can no longer be inferred, while strictly preserving ALL core questions, "
    "tasks, and requests in natural, fluent English.\n"
    "RULES:\n"
    "1. Decompose & Reconstruct: Identify all user requests across all clauses. If an identifying "
    "detail is embedded mid-sentence (e.g. 'as a 42 year old software professional, help me find...'), "
    "excise or generalize the attribute and smoothly reconnect the clauses. NEVER truncate or drop "
    "downstream questions or requests.\n"
    "2. Remove or generalize author details: Specific cities, landmarks, or transit lines become 'my area'; "
    "job titles and workplaces become 'my work' or 'my role'; schools and degrees become 'my studies'; "
    "exact ages and income figures are removed or generalized to life stages ('as a working professional').\n"
    "3. Preserve non-author content: Keep all third-party subjects (patients, clients), code, algorithms, "
    "technical specs, and statutory jurisdictions that are the subject of inquiry.\n"
    "4. Zero-PII inputs: If the text reveals no personal attributes about the author, return it verbatim.\n"
    "5. Format: Output only the final rewritten text - no explanations, meta-commentary, or filler."
)

REWRITE_INSTRUCTION = (
    "Rewrite the text below so that nothing in it reveals the author's personal attributes "
    "(age, location, occupation, education, income or other identifiers). Look for them "
    "anywhere in the text. Preserve the text's intent, questions and technical content. If "
    "nothing is revealing, return the text unchanged. Output only the rewritten text.\n\n"
    "TEXT TO REWRITE:\n"
)


def build_rewrite_messages(
    text: str,
    system_prompt: Optional[str] = None,
    few_shot: Optional[Sequence[Tuple[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build single-turn chat messages [system, user] for rewriting."""
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt if system_prompt is not None else REWRITER_SYSTEM_PROMPT}
    ]
    if few_shot:
        for ex_input, ex_output in few_shot:
            messages.append({"role": "user", "content": REWRITE_INSTRUCTION + ex_input})
            messages.append({"role": "assistant", "content": ex_output})
    messages.append({"role": "user", "content": REWRITE_INSTRUCTION + text})
    return messages


def is_personal_disclosure(sentence: str) -> bool:
    """Return True if the sentence functions primarily as an autobiographical persona disclosure."""
    s = sentence.strip()
    if "?" in s:
        return False
    # Third-party case descriptions (patients, clients, legal subjects, algorithms)
    # are problem entities, not autobiographical.
    if re.search(
        r"\b(?:patient|client|plaintiff|defendant|subject|infant|child|algorithm|code|server|database"
        r"|male|female|woman|man|presents? with|presenting with|case of)\b",
        s,
        re.IGNORECASE,
    ):
        return False
    patterns = [
        r"^(?:I am|I'm|As|Being|Currently|My name is)\b",
        r"^(?:I live in|I work as|I work at|I make|I earn|I have worked|I work)\b",
        r"^(?:FYI,?|By the way,?|Note that|P\.?S\.?)\s+(?:I am|I'm|I live|I work|I earn|I'm a)",
        r"\b(?:retired|freshman|senior partner|cardiologist|software engineer|resident at|doctor|nurse)\b",
        r"\b\d{1,2}[-\s]*(?:years?[-\s]*old|yo|y/o)\b",
    ]
    return any(bool(re.search(p, s, re.IGNORECASE)) for p in patterns)


def prune_persona_bidirectional(text: str) -> str:
    """Prune personal autobiographical framing clauses appearing before, after, or embedded in the query."""
    text = text.strip()
    if not text:
        return ""
    original = text

    # 1. Trailing parentheticals containing personal cues: e.g. '(Freshman at MIT ...)'
    paren_match = re.search(r"\s*\(([^)]+)\)[.!?]?$", text)
    if paren_match:
        inside = paren_match.group(1).strip()
        if (
            is_personal_disclosure(inside)
            or any(
                k in inside.lower()
                for k in [
                    "earning",
                    "salary",
                    "living in",
                    "student",
                    "freshman",
                    "engineer",
                    "doctor",
                    "cardiologist",
                ]
            )
        ):
            text = text[:paren_match.start()].strip()

    # 2. Check multi-sentence structures: split by sentence boundary preserving decimal numbers
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s.strip()]
    has_anaphora = any(
        re.search(
            r"\b(?:these\s+credentials|this\s+background|my\s+background|my\s+profile|"
            r"my\s+stats|my\s+qualifications|these\s+stats|am\s+I\s+qualified|eligible|chances?\s+of)\b",
            s,
            re.IGNORECASE,
        )
        for s in sentences[1:]
    )
    while len(sentences) > 1 and is_personal_disclosure(sentences[0]) and not has_anaphora:
        sentences.pop(0)
    while len(sentences) > 1 and is_personal_disclosure(sentences[-1]):
        sentences.pop()
    text = " ".join(sentences).strip()

    # 3. Check introductory clause in a single sentence: 'As an 18yo ..., can you ...'
    clause_match = re.search(
        r"^(?:As|Being|Since I am|Given that I am|I am|I'm)\s+[^,;\n]+?,\s*(can\b|could\b|please\b|how\b|what\b|why\b|where\b|when\b|who\b|is\b|are\b|do\b|does\b|would\b)(.*)$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if clause_match:
        text = (clause_match.group(1) + clause_match.group(2)).strip()

    # 3b. Embedded personal clause followed by another clause/request
    embedded_match = None
    if not has_anaphora:
        embedded_match = re.search(
            r"\s*,?\s*\bas\s+(?:an?|\d{1,2})\b[^.,;\n]*?(?:\b(?:years?\s+old|yo|y/o|professional|engineer|developer|doctor|lawyer|consultant|worker|nurse|manager)\b)[^.,;\n]*",
            text,
            flags=re.IGNORECASE,
        )
    if embedded_match:
        tail = text[embedded_match.end():].lstrip(",; ")
        if tail:
            head = text[:embedded_match.start()].rstrip(",; ")
            text = f"{head}, {tail}" if head else tail

    # 4. Trailing postamble clause without period (protecting against truncating downstream requests)
    text = re.sub(
        r"\s*(?:(?:By the way|FYI|Note|P\.?S\.?)[:,\s]+)?(?:I am|I'm|I live in|Living in|Working as|I work as|I earn|Earning|My age is|As an?)\s+(?!.*?\b(?:can\b|could\b|please\b|help\b|how\b|what\b|why\b|where\b|when\b|who\b|recommend\b))[^\n]*?[.!?]?$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Capitalize the first letter only when pruning actually removed leading material
    if text and text != original and text[0].islower():
        text = text[0].upper() + text[1:]
    return text
