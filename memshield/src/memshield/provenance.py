"""
provenance.py — Cryptographic content integrity + full provenance for RAG chunks.

Ingest-time integrity layer:
  1. Canonicalize text (deterministic normalization before hashing)
  2. SHA-256 hash of canonical form
  3. Full provenance metadata: source, timestamp, authority, chunk_id
  4. Read-time tamper detection: recompute hash, compare

This detects post-ingestion tampering. Poison detection is handled
by the retrieval-time signals (ProGRank, RAGMask, influence, etc.).
"""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from typing import Any


def canonicalize(text: str) -> str:
    """Deterministic text canonicalization before hashing.

    Normalizes whitespace, unicode form, case-folds, strips zero-width
    characters, and collapses runs — so semantically-identical content
    always produces the same hash regardless of encoding variation.
    """
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Strip zero-width characters (common in injection attempts)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff]", "", text)

    # Strip ANSI escape sequences
    text = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)

    # Normalize whitespace: collapse all runs to single space, strip edges
    text = re.sub(r"\s+", " ", text).strip()

    # Case-fold for hash consistency (mixed-case variants → same hash)
    text = text.casefold()

    return text


class ContentHasher:
    """SHA-256 content hashing with canonicalization for RAG chunk provenance."""

    HASH_KEY = "content_hash"
    CANON_HASH_KEY = "canon_hash"
    SOURCE_KEY = "provenance_source"
    TIMESTAMP_KEY = "provenance_ts"
    AUTHORITY_KEY = "provenance_authority"
    CHUNK_ID_KEY = "provenance_chunk_id"

    @staticmethod
    def hash_raw(text: str) -> str:
        """SHA-256 of raw UTF-8 text (backward-compatible)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_canonical(text: str) -> str:
        """SHA-256 of canonicalized text (preferred for integrity checks)."""
        return hashlib.sha256(canonicalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def hash_and_attach(
        cls,
        text: str,
        metadata: dict | None = None,
        source: str = "unknown",
        authority: float = 0.5,
        chunk_id: str = "",
    ) -> dict:
        """Compute hashes and full provenance, attach to metadata.

        Args:
            text: Raw chunk text
            metadata: Existing metadata dict (copied, not mutated)
            source: Provenance source identifier (URL, file path, API name)
            authority: Authority prior score [0.0, 1.0]
            chunk_id: Unique chunk identifier
        """
        meta = dict(metadata) if metadata else {}

        # Dual hashes: raw (backward compat) + canonical (primary integrity)
        meta[cls.HASH_KEY] = cls.hash_raw(text)
        meta[cls.CANON_HASH_KEY] = cls.hash_canonical(text)

        # Full provenance metadata
        meta[cls.SOURCE_KEY] = source
        meta[cls.TIMESTAMP_KEY] = time.time()
        meta[cls.AUTHORITY_KEY] = authority
        if chunk_id:
            meta[cls.CHUNK_ID_KEY] = chunk_id

        return meta

    @classmethod
    def verify(cls, text: str, metadata: dict | None) -> bool:
        """Verify content integrity at retrieval time.

        Recomputes hash from current text and compares to stored hash.
        Uses canonical hash if available, falls back to raw hash.

        Returns True if integrity verified or no hash stored (legacy data).
        """
        if not metadata:
            return True

        # Prefer canonical hash (stronger — ignores encoding variation)
        if cls.CANON_HASH_KEY in metadata:
            return metadata[cls.CANON_HASH_KEY] == cls.hash_canonical(text)

        # Fall back to raw hash (backward compatibility)
        if cls.HASH_KEY in metadata:
            return metadata[cls.HASH_KEY] == cls.hash_raw(text)

        # No provenance data — allow legacy chunks
        return True

    @classmethod
    def is_tampered(cls, text: str, metadata: dict | None) -> bool:
        """Convenience inverse of verify(). True if tampered."""
        return not cls.verify(text, metadata)

    @classmethod
    def get_provenance(cls, metadata: dict | None) -> dict[str, Any]:
        """Extract provenance fields from metadata."""
        if not metadata:
            return {}
        return {
            "source": metadata.get(cls.SOURCE_KEY, "unknown"),
            "timestamp": metadata.get(cls.TIMESTAMP_KEY, 0.0),
            "authority": metadata.get(cls.AUTHORITY_KEY, 0.5),
            "chunk_id": metadata.get(cls.CHUNK_ID_KEY, ""),
            "has_canon_hash": cls.CANON_HASH_KEY in metadata,
            "has_raw_hash": cls.HASH_KEY in metadata,
        }
