"""
embedding_fn.py — single source of truth for the ChromaDB embedding model.

The agent_kb collection (skills + memories + KB) is embedded and queried with
ONE model. It MUST be identical at every collection-open site or cosine
similarity is meaningless. Import get_embedding_fn() everywhere instead of
constructing an embedding function inline.

Default model: BAAI/bge-small-en-v1.5 — local, ~130MB, MTEB ~62 (vs the old
ChromaDB default all-MiniLM-L6-v2 ~56). Fully offline after first download;
no data leaves the device. Chosen over OpenAI embeddings deliberately: this is
an anti-poisoning security system; shipping every memory/query to a third
party would contradict its own threat model.

Override with env PRISM_EMBED_MODEL. If sentence-transformers or the model is
unavailable this module raises: the store is embedded with this model, and any
substitute embedder returns wrong similarities. Failing hard beats corrupting
retrieval. Fix: pip install -r requirements.txt (and allow one HF download).
"""
from __future__ import annotations

import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("PRISM_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Process-wide singleton keys — stored in sys.modules so they survive the
# double-import problem where this file is loaded as both 'embedding_fn' and
# 'scripts.embedding_fn' (separate module objects, separate module-level globals).
_CACHE_KEY = "_prism_ef_singleton"
_LOCK_KEY  = "_prism_ef_lock"

if _LOCK_KEY not in sys.modules:
    sys.modules[_LOCK_KEY] = threading.Lock()  # type: ignore[assignment]

_load_lock: threading.Lock = sys.modules[_LOCK_KEY]  # type: ignore[assignment]


class _SentenceTransformerEF:
    """Thin ChromaDB-compatible wrapper around sentence_transformers.SentenceTransformer.

    Bypasses chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    which does its own import-time version checks that break with sentence-transformers
    >= 3.x even when the package is correctly installed.

    ChromaDB 1.5+ requires name() as a method (called by is_legacy()) and
    build_from_config()/get_config() stubs; returns NotImplemented to opt into
    legacy mode, which skips config serialization checks.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer
        # Try local cache first — avoids all HF network requests when model is cached.
        # Falls back to network download on first run (model not yet cached).
        try:
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self._model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        vecs = self._model.encode(input, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def name(self) -> str:
        # Must match the name chromadb stored when the collection was first created
        # with SentenceTransformerEmbeddingFunction ("sentence_transformer").
        return "sentence_transformer"

    def get_config(self):
        return NotImplemented

    def build_from_config(self, config):
        return NotImplemented


def embedding_tag() -> str:
    """Stable string identifying the active model."""
    entry = sys.modules.get(_CACHE_KEY)
    tag = entry[1] if entry else None
    return tag or f"pending:{EMBED_MODEL}"


def get_embedding_fn():
    """Return the process-wide ChromaDB embedding function (singleton).

    Uses sys.modules as the cache so the same object is returned regardless of
    whether this module was imported as 'embedding_fn' or 'scripts.embedding_fn'.
    """
    entry = sys.modules.get(_CACHE_KEY)
    if entry is not None:
        return entry[0]

    with _load_lock:
        entry = sys.modules.get(_CACHE_KEY)
        if entry is not None:
            return entry[0]

        try:
            # httpx is the HTTP client huggingface_hub uses internally.
            import logging as _logging
            _logging.getLogger("httpx").setLevel(_logging.WARNING)
            _logging.getLogger("huggingface_hub").setLevel(_logging.WARNING)

            fn = _SentenceTransformerEF(EMBED_MODEL)
            _ = fn(["warmup probe"])
            tag = f"st:{EMBED_MODEL}"
            sys.modules[_CACHE_KEY] = (fn, tag)  # type: ignore[assignment]
            logger.info(f"[Embed] active model: {EMBED_MODEL} (sentence-transformers)")
            return fn

        except Exception as exc:
            raise RuntimeError(
                f"[Embed] FAILED to load {EMBED_MODEL} ({exc}). "
                f"The agent_kb store is embedded with this model; substituting "
                f"another embedder returns wrong similarities. "
                f"Fix: pip install -r requirements.txt and ensure HuggingFace "
                f"is reachable once for the initial download."
            ) from exc
