"""
influence.py — Leave-one-out influence scoring for RAG poisoning detection.

For a query q and retrieved document set C = {d_1, ..., d_k}:
  1. Generate full answer:        a_0 = G(q, C)
  2. For each doc d_i in C:       a_{-i} = G(q, C without d_i)
  3. Semantic drift:              SemDrift_i = 1 - cos(E(a_0), E(a_{-i}))
  4. Citation drift:              CiteDrift_i = 1 - Jaccard(Cites(a_0), Cites(a_{-i}))
  5. Influence score:             I_i = gamma * SemDrift_i + (1-gamma) * CiteDrift_i

High influence ≈ the answer changes drastically when d_i is removed, which
is expected for a legitimate authoritative source but also characteristic of
a poisoned document that hijacks generation. Downstream layers (authority
prior, ProGRank) disambiguate the two cases.

The module is generator/embedder agnostic — callers supply G and E as callables.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────────

@dataclass
class InfluenceResult:
    """Influence score for a single retrieved document."""
    doc_index: int
    doc_id: str
    semantic_drift: float
    citation_drift: float
    influence_score: float


@dataclass
class InfluenceReport:
    """Full leave-one-out report for a retrieval set."""
    query: str
    scores: list[InfluenceResult] = field(default_factory=list)
    baseline_answer: str = ""

    def ranked(self, descending: bool = True) -> list[InfluenceResult]:
        """Return scores sorted by influence (highest first by default)."""
        return sorted(self.scores, key=lambda r: r.influence_score, reverse=descending)

    def flagged(self, threshold: float = 0.6) -> list[InfluenceResult]:
        """Return docs whose influence exceeds the threshold."""
        return [r for r in self.scores if r.influence_score >= threshold]


# ── Citation extraction ──────────────────────────────────────────────────────

# Matches quoted spans ≥4 words that likely came from a source doc.
_QUOTE_RE = re.compile(r'"([^"]{20,})"')
# Matches "according to <source>" or "[source]" style references.
_REF_RE = re.compile(r'(?:according to|per|from|source[:\s])\s*([^\.,;]+)', re.IGNORECASE)
# Matches bracketed citations like [1], [Smith 2024], [doc-abc]
_BRACKET_RE = re.compile(r'\[([^\]]{1,60})\]')


def extract_citations(text: str) -> set[str]:
    """Extract citation-like spans from generated text.

    Uses multiple heuristics:
      - Quoted passages (≥20 chars)
      - "according to X" / "per X" attributions
      - Bracketed references [X]
      - Sentences containing source-indicator words

    Returns a set of normalized citation strings for Jaccard comparison.
    """
    cites: set[str] = set()

    for m in _QUOTE_RE.finditer(text):
        cites.add(_normalize_cite(m.group(1)))

    for m in _REF_RE.finditer(text):
        cites.add(_normalize_cite(m.group(1)))

    for m in _BRACKET_RE.finditer(text):
        cites.add(_normalize_cite(m.group(1)))

    return cites


def _normalize_cite(s: str) -> str:
    """Lowercase, strip, collapse whitespace for consistent Jaccard."""
    return re.sub(r'\s+', ' ', s.strip().lower())


# ── Core math ────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on degenerate input."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets. Returns 1.0 if both empty."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Leave-one-out engine ────────────────────────────────────────────────────

GeneratorFn = Callable[[str, list[str]], str]
EmbedderFn = Callable[[str], np.ndarray]


def compute_influence(
    query: str,
    documents: list[str],
    doc_ids: list[str],
    generator: GeneratorFn,
    embedder: EmbedderFn,
    gamma: float = 0.5,
    cite_extractor: Callable[[str], set[str]] | None = None,
) -> InfluenceReport:
    """Compute leave-one-out influence scores for each document in the retrieval set.

    Args:
        query: The user query.
        documents: Retrieved document texts (the set C).
        doc_ids: Corresponding document IDs.
        generator: G(query, context_docs) -> answer string.
                   Called k+1 times (once for full set, once per leave-one-out).
        embedder: E(text) -> embedding vector (numpy array).
                  Called k+1 times on the generated answers.
        gamma: Weight balancing semantic vs citation drift.
               γ=1.0 → pure semantic, γ=0.0 → pure citation.
        cite_extractor: Optional custom citation extractor. Defaults to
                        extract_citations().

    Returns:
        InfluenceReport with per-document influence scores.
    """
    if len(documents) != len(doc_ids):
        raise ValueError(f"documents ({len(documents)}) and doc_ids ({len(doc_ids)}) must match")

    extract = cite_extractor or extract_citations
    k = len(documents)
    report = InfluenceReport(query=query)

    # ── Step 1: baseline answer from full context ────────────────────────
    try:
        a0 = generator(query, documents)
    except Exception as exc:
        logger.error(f"Generator failed on full context: {exc}")
        raise
    report.baseline_answer = a0

    e0 = embedder(a0)
    cites0 = extract(a0)

    # ── Step 2: leave-one-out for each document ──────────────────────────
    for i in range(k):
        docs_without_i = documents[:i] + documents[i + 1:]

        try:
            a_minus_i = generator(query, docs_without_i) if docs_without_i else ""
        except Exception as exc:
            logger.warning(f"Generator failed for leave-one-out (doc {i}): {exc}")
            # Fail-closed: treat generator failure as maximum influence
            report.scores.append(InfluenceResult(
                doc_index=i,
                doc_id=doc_ids[i],
                semantic_drift=1.0,
                citation_drift=1.0,
                influence_score=1.0,
            ))
            continue

        # Semantic drift: 1 - cos(E(a0), E(a_{-i}))
        if a_minus_i:
            e_minus_i = embedder(a_minus_i)
            sem_drift = 1.0 - cosine_similarity(e0, e_minus_i)
        else:
            # No docs left → empty answer → maximum semantic drift
            sem_drift = 1.0

        # Citation drift: 1 - Jaccard(Cites(a0), Cites(a_{-i}))
        cites_minus_i = extract(a_minus_i) if a_minus_i else set()
        cite_drift = 1.0 - jaccard_similarity(cites0, cites_minus_i)

        # Combined influence score
        influence = gamma * sem_drift + (1.0 - gamma) * cite_drift

        report.scores.append(InfluenceResult(
            doc_index=i,
            doc_id=doc_ids[i],
            semantic_drift=sem_drift,
            citation_drift=cite_drift,
            influence_score=influence,
        ))

    return report
