"""
provenance.py — Cryptographic content integrity + provenance for RAG chunks.

Ingest-time integrity layer (extended for PROVE architecture):
  1. Canonicalize text (deterministic normalization before hashing)
  2. SHA-256 hash of canonical form
  3. Full provenance metadata: source_class, source_id, timestamp, chunk_id
  4. HMAC-SHA256 seal binding (canon_hash, source_class, source_id, ts)
     under a per-install key (Android Keystore on device, env-var or file
     fallback in dev / on Linux)
  5. Read-time verification: recompute hash AND verify HMAC; either failing
     is a hard tamper signal.

Two failure modes are now distinguished:
  - content_tamper: canonical hash mismatches (someone rewrote the text)
  - provenance_tamper: HMAC mismatches (source_class / source_id / ts was
    rewritten, OR the key changed)
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import time
import unicodedata
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            "source_class": metadata.get(ProvenanceSeal.SOURCE_CLASS_KEY),
            "source_id": metadata.get(ProvenanceSeal.SOURCE_ID_KEY),
            "has_canon_hash": cls.CANON_HASH_KEY in metadata,
            "has_raw_hash": cls.HASH_KEY in metadata,
            "has_seal": ProvenanceSeal.SEAL_KEY in metadata,
        }


# ── HMAC sealing layer ────────────────────────────────────────────────────────
#
# Binds (canon_hash, source_class, source_id, ingest_ts) under a per-install
# key. The key MUST be opaque to the agent (Android Keystore on device;
# env var or 0600 file in dev). Verification is constant-time.
#
# An attacker who can rewrite memory rows but cannot extract the key cannot
# produce a valid seal — so any class/id/ts rewrite is caught at retrieval.


_DEFAULT_KEY_PATH = Path(os.environ.get(
    "MEMSHIELD_HMAC_KEY_PATH",
    str(Path.home() / ".memshield" / "hmac.key"),
))
_KEY_ENV_VAR = "MEMSHIELD_HMAC_KEY"


def _load_or_create_dev_key(path: Path = _DEFAULT_KEY_PATH) -> bytes:
    """Dev / Linux fallback key loader.

    Priority:
      1. MEMSHIELD_HMAC_KEY env var (hex-encoded)  — for tests / CI
      2. File at MEMSHIELD_HMAC_KEY_PATH           — for local dev
      3. Generate a new 32-byte key, write 0600    — first-run

    On Android, the key is supplied by the sidecar (Keystore-backed) and
    set via the env var at agent startup; we never touch the filesystem
    fallback in production.
    """
    env = os.environ.get(_KEY_ENV_VAR)
    if env:
        try:
            return bytes.fromhex(env)
        except ValueError:
            logger.warning("MEMSHIELD_HMAC_KEY env var is not valid hex; ignoring.")

    if path.exists():
        try:
            return path.read_bytes()
        except OSError as e:
            logger.warning("Cannot read HMAC key file %s: %s — regenerating.", path, e)

    # First run: mint a new key, persist with restrictive perms.
    key = secrets.token_bytes(32)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        os.chmod(path, 0o600)
        logger.info("Minted new MemShield HMAC key at %s (0600).", path)
    except OSError as e:
        logger.warning("Could not persist HMAC key to %s (%s); using ephemeral key.", path, e)
    return key


# Process-wide cached key (loaded lazily). The HMAC verification is
# constant-time so the key never needs to be exposed.
_cached_key: bytes | None = None


def _key() -> bytes:
    global _cached_key
    if _cached_key is None:
        _cached_key = _load_or_create_dev_key()
    return _cached_key


def reset_key_cache() -> None:
    """For tests — force re-read of key on next operation."""
    global _cached_key
    _cached_key = None


class ProvenanceSeal:
    """HMAC-SHA256 seal binding content hash + source class + source id + ts.

    The seal is the only mechanism that binds *who claims authorship* of a
    chunk to its *content*. Without it, an attacker who can write to the
    memory store could relabel a T3_UNTRUSTED chunk as T0_USER_TYPED and
    promote its capabilities to the maximum set.

    Implementation note: the seal does NOT cover the chunk_id (it is a
    storage key, not security-relevant) or the authority float (legacy
    field; replaced by source_class).
    """

    # Metadata keys (kept short, namespaced)
    SEAL_KEY         = "prov_seal"          # hex HMAC
    SEAL_VERSION_KEY = "prov_seal_v"        # int — bump on format change
    SOURCE_CLASS_KEY = "source_class"       # str name, e.g. "T3_UNTRUSTED"
    SOURCE_ID_KEY    = "source_id"          # canonical id

    SEAL_VERSION = 1

    @classmethod
    def _seal_input(
        cls,
        canon_hash: str,
        source_class: str,
        source_id: str,
        ts: float,
    ) -> bytes:
        """Deterministic serialization of the sealed fields.

        Order and separators are part of the protocol — DO NOT change
        without bumping SEAL_VERSION.
        """
        # NUL-separated; canon_hash is hex (ASCII) so collision-safe.
        # ts is rounded to int seconds to make seals reproducible across
        # restart (we don't care about sub-second granularity for trust).
        ts_int = int(ts)
        return f"{cls.SEAL_VERSION}\0{canon_hash}\0{source_class}\0{source_id}\0{ts_int}".encode("utf-8")

    @classmethod
    def compute(
        cls,
        canon_hash: str,
        source_class: str,
        source_id: str,
        ts: float,
        key: bytes | None = None,
    ) -> str:
        """Compute the seal HMAC. Returns hex digest."""
        k = key if key is not None else _key()
        msg = cls._seal_input(canon_hash, source_class, source_id, ts)
        return hmac.new(k, msg, hashlib.sha256).hexdigest()

    @classmethod
    def seal_metadata(
        cls,
        text: str,
        source_class: str,
        source_id: str,
        metadata: dict | None = None,
        ts: float | None = None,
        key: bytes | None = None,
    ) -> dict:
        """Compute canonical hash + HMAC seal and attach to metadata.

        Returns a NEW metadata dict (does not mutate input). Existing
        provenance fields are preserved / overwritten consistently.
        """
        meta = dict(metadata) if metadata else {}
        ts = ts if ts is not None else time.time()
        canon = ContentHasher.hash_canonical(text)

        meta[ContentHasher.HASH_KEY] = ContentHasher.hash_raw(text)
        meta[ContentHasher.CANON_HASH_KEY] = canon
        meta[ContentHasher.TIMESTAMP_KEY] = ts
        meta[cls.SOURCE_CLASS_KEY] = source_class
        meta[cls.SOURCE_ID_KEY] = source_id
        meta[cls.SEAL_VERSION_KEY] = cls.SEAL_VERSION
        meta[cls.SEAL_KEY] = cls.compute(canon, source_class, source_id, ts, key=key)
        return meta

    @classmethod
    def verify(
        cls,
        text: str,
        metadata: dict | None,
        key: bytes | None = None,
    ) -> tuple[bool, str]:
        """Verify both content hash AND HMAC seal.

        Returns (is_valid, reason). reason ∈
            "ok" | "no_seal" | "content_tamper" | "provenance_tamper" |
            "missing_field" | "version_mismatch".

        Legacy chunks (no seal) → (True, "no_seal"). Callers can choose
        whether to admit unsealed legacy entries; the policy gate refuses
        unsealed chunks for any side-effect action.
        """
        if not metadata:
            return True, "no_seal"
        if cls.SEAL_KEY not in metadata:
            return True, "no_seal"

        # Content integrity first — cheaper, catches text rewrites.
        if not ContentHasher.verify(text, metadata):
            return False, "content_tamper"

        # Required fields
        try:
            canon       = metadata[ContentHasher.CANON_HASH_KEY]
            src_cls     = metadata[cls.SOURCE_CLASS_KEY]
            src_id      = metadata[cls.SOURCE_ID_KEY]
            ts          = metadata[ContentHasher.TIMESTAMP_KEY]
            stored_seal = metadata[cls.SEAL_KEY]
            version     = metadata.get(cls.SEAL_VERSION_KEY, 0)
        except KeyError:
            return False, "missing_field"

        if version != cls.SEAL_VERSION:
            return False, "version_mismatch"

        expected = cls.compute(canon, src_cls, src_id, ts, key=key)
        if not hmac.compare_digest(expected, stored_seal):
            return False, "provenance_tamper"
        return True, "ok"

    @classmethod
    def is_sealed(cls, metadata: dict | None) -> bool:
        """True iff metadata has a seal (regardless of validity)."""
        return bool(metadata) and cls.SEAL_KEY in metadata


__all__ = [
    "canonicalize",
    "ContentHasher",
    "ProvenanceSeal",
    "reset_key_cache",
]
