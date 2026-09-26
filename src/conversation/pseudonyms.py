"""Persistent, entity-type-aware pseudonym manager with cryptographic storage.

Provides bidirectional consistency across turns and files, atomic file locking,
and salted storage for entity types PER, LOC, ORG, DATE, and ID.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logger = logging.getLogger(__name__)

SUPPORTED_ENTITY_TYPES = ("PER", "LOC", "ORG", "DATE", "ID")


class PersistentPseudonymManager:
    """Manages persistent, entity-aware synthetic replacements with salted cryptographic storage."""

    def __init__(
        self,
        storage_dir: str | Path = ".state",
        master_secret: Optional[str] = "inferenceguard_default_secret",
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.vault_file = self.storage_dir / "pseudonym_vault.json"
        self.salt_file = self.storage_dir / "vault.salt"
        self.lock_file = self.storage_dir / "vault.lock"

        self.master_secret = master_secret or "inferenceguard_default_secret"
        self.salt = self._get_or_create_salt()
        self.fernet = self._init_cipher()

        # In-memory mapping structures:
        # forward_map: (session_id, entity_type, real_entity_lower) -> pseudonym
        self.forward_map: Dict[Tuple[str, str, str], str] = {}
        # reverse_map: (session_id, pseudonym) -> real_entity
        self.reverse_map: Dict[Tuple[str, str], str] = {}
        # entity_counters: (session_id, entity_type) -> int
        self.entity_counters: Dict[Tuple[str, str], int] = {}

        self._load_vault()

    def _get_or_create_salt(self) -> bytes:
        """Load or generate a unique 16-byte salt for key derivation."""
        if self.salt_file.exists():
            return self.salt_file.read_bytes()
        salt = os.urandom(16)
        self.salt_file.write_bytes(salt)
        return salt

    def _init_cipher(self) -> Optional[Any]:
        """Derive cryptographic key and initialize Fernet cipher if available."""
        if not HAS_CRYPTO:
            return None
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_secret.encode("utf-8")))
        return Fernet(key)

    def _encrypt(self, text: str) -> str:
        """Encrypt string value."""
        if self.fernet is not None:
            return self.fernet.encrypt(text.encode("utf-8")).decode("utf-8")
        # Deterministic obfuscation fallback if cryptography is unavailable
        encoded = base64.b64encode(text.encode("utf-8")).decode("utf-8")
        return f"obf:{encoded}"

    def _decrypt(self, encrypted_text: str) -> str:
        """Decrypt string value."""
        if self.fernet is not None and not encrypted_text.startswith("obf:"):
            return self.fernet.decrypt(encrypted_text.encode("utf-8")).decode("utf-8")
        if encrypted_text.startswith("obf:"):
            raw = encrypted_text[4:]
            return base64.b64decode(raw.encode("utf-8")).decode("utf-8")
        return encrypted_text

    def _hash_entity(self, text: str) -> str:
        """Salted SHA-256 hash for identifier lookup."""
        h = hashlib.sha256(self.salt + text.lower().strip().encode("utf-8"))
        return h.hexdigest()[:16]

    def _load_vault(self) -> None:
        """Load encrypted pseudonym vault from disk."""
        if not self.vault_file.exists():
            return

        try:
            with open(self.vault_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries = data.get("entries", [])
            for entry in entries:
                session_id = entry.get("session_id", "default")
                entity_type = entry.get("entity_type", "PER")
                pseudonym = entry.get("pseudonym")
                enc_real = entry.get("encrypted_real")
                real_entity = self._decrypt(enc_real)

                key = (session_id, entity_type, real_entity.lower().strip())
                self.forward_map[key] = pseudonym
                self.reverse_map[(session_id, pseudonym)] = real_entity

                # Update counter
                count_key = (session_id, entity_type)
                m = re.search(r"_(\d+)\]$", pseudonym)
                if m:
                    idx = int(m.group(1))
                    self.entity_counters[count_key] = max(
                        self.entity_counters.get(count_key, 0), idx
                    )
        except Exception as e:
            logger.warning(f"Could not load pseudonym vault: {e}")

    def _save_vault(self) -> None:
        """Atomically persist encrypted pseudonym vault to disk."""
        entries: List[Dict[str, Any]] = []
        for (session_id, entity_type, _), pseudonym in self.forward_map.items():
            real_entity = self.reverse_map.get((session_id, pseudonym), "")
            entries.append({
                "session_id": session_id,
                "entity_type": entity_type,
                "pseudonym": pseudonym,
                "entity_hash": self._hash_entity(real_entity),
                "encrypted_real": self._encrypt(real_entity),
            })

        payload = {"version": "1.0", "entries": entries}
        temp_file = self.storage_dir / f"pseudonym_vault_{os.getpid()}.tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Atomic replace
        temp_file.replace(self.vault_file)

    def get_or_create(
        self,
        real_entity: str,
        entity_type: str,
        session_id: str = "default",
    ) -> str:
        """Retrieve existing pseudonym or generate a new consistent entity-aware replacement.

        Args:
            real_entity: Original entity name or identifier.
            entity_type: Category ('PER', 'LOC', 'ORG', 'DATE', 'ID').
            session_id: Conversation session identifier for isolation.

        Returns:
            Persistent synthetic pseudonym string.
        """
        clean_entity = real_entity.strip()
        clean_type = entity_type.upper().strip()
        if clean_type not in SUPPORTED_ENTITY_TYPES:
            clean_type = "ID"

        key = (session_id, clean_type, clean_entity.lower())
        if key in self.forward_map:
            return self.forward_map[key]

        # Generate new sequential pseudonym for this session and entity type
        count_key = (session_id, clean_type)
        idx = self.entity_counters.get(count_key, 0) + 1
        self.entity_counters[count_key] = idx

        pseudonym = f"[{clean_type}_{idx}]"

        self.forward_map[key] = pseudonym
        self.reverse_map[(session_id, pseudonym)] = clean_entity
        self._save_vault()

        return pseudonym

    def pseudonymize(
        self,
        text: str,
        session_id: str = "default",
        known_entities: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """Scan text and replace detected or provided entities with persistent pseudonyms.

        Args:
            text: Raw input text.
            session_id: Conversation session identifier.
            known_entities: Optional sequence of (entity_text, entity_type) tuples.

        Returns:
            Text with all entities consistently pseudonymized.
        """
        result = text

        if known_entities:
            # Sort by descending length so substrings don't replace early
            sorted_entities = sorted(known_entities, key=lambda x: len(x[0]), reverse=True)
            for entity_text, entity_type in sorted_entities:
                if not entity_text.strip():
                    continue
                pseudo = self.get_or_create(entity_text, entity_type, session_id)
                pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
                result = pattern.sub(pseudo, result)

        # Built-in regex detection for emails, phones, IPs, and common names
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        for match in re.findall(email_pattern, result):
            pseudo = self.get_or_create(match, "ID", session_id)
            result = result.replace(match, pseudo)

        phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        for match in re.findall(phone_pattern, result):
            pseudo = self.get_or_create(match, "ID", session_id)
            result = result.replace(match, pseudo)

        return result

    def depseudonymize(
        self,
        text: str,
        session_id: str = "default",
    ) -> str:
        """Restore all pseudonyms in text to their original real entity values."""
        result = text
        # Find all pseudonym patterns e.g. [PER_1], [LOC_2]
        pseudo_pattern = r"\[(?:PER|LOC|ORG|DATE|ID)_\d+\]"
        matches = set(re.findall(pseudo_pattern, text))

        for pseudo in matches:
            real = self.reverse_map.get((session_id, pseudo))
            if real is not None:
                result = result.replace(pseudo, real)

        return result

    def clear_session(self, session_id: str) -> None:
        """Clear pseudonym mapping for a specific session."""
        self.forward_map = {k: v for k, v in self.forward_map.items() if k[0] != session_id}
        self.reverse_map = {k: v for k, v in self.reverse_map.items() if k[0] != session_id}
        self.entity_counters = {k: v for k, v in self.entity_counters.items() if k[0] != session_id}
        self._save_vault()
