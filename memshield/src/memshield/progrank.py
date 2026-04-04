"""
progrrank.py -- ProGRank-style perturbation instability for RAG poisoning defense.

Probe-gradient instability reranking:
  1. For the original query q, retrieve top-k documents with scores.
  2. Generate N perturbed queries {q'_1, ..., q'_N} (paraphrase, typo, synonym).
  3. For each q'_j, retrieve top-k and record each document's rank.
  4. For each doc d_i, compute rank instability:
     PGR_i = std(ranks_i) / mean(ranks_i)   (coefficient of variation)

Clean, genuinely relevant documents maintain stable rankings across query
perturbations. Poisoned documents that exploit specific trigger phrases
show volatile rankings -- high PGR.

The module is retriever-agnostic: callers supply a retrieval function.
Query perturbation can use a provided perturber or built-in strategies.
"""
from __future__ import annotations

import logging
import random
import re
import string
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InstabilityResult:
    """ProGRank instability score for a single document."""
    doc_id: str
    pgr_score: float          # coefficient of variation of ranks
    mean_rank: float
    std_rank: float
    ranks: list[int]          # rank in each perturbed retrieval (0-indexed, -1 = not retrieved)
    original_rank: int


@dataclass
class InstabilityReport:
    """ProGRank instability report for a retrieval set."""
    query: str
    n_perturbations: int
    results: list[InstabilityResult] = field(default_factory=list)

    def flagged(self, threshold: float = 1.0) -> list[InstabilityResult]:
        """Return docs whose PGR instability exceeds threshold."""
        return [r for r in self.results if r.pgr_score >= threshold]

    def ranked(self, descending: bool = True) -> list[InstabilityResult]:
        """Return results sorted by PGR instability."""
        return sorted(self.results, key=lambda r: r.pgr_score, reverse=descending)


# ── Query perturbation strategies ────────────────────────────────────────────

def _typo_perturb(query: str, rng: random.Random) -> str:
    """Insert a random typo (swap adjacent chars, drop a char, or add a char)."""
    words = query.split()
    if not words:
        return query
    idx = rng.randint(0, len(words) - 1)
    w = words[idx]
    if len(w) < 2:
        return query
    op = rng.choice(["swap", "drop", "add"])
    if op == "swap" and len(w) >= 2:
        pos = rng.randint(0, len(w) - 2)
        w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
    elif op == "drop":
        pos = rng.randint(0, len(w) - 1)
        w = w[:pos] + w[pos + 1:]
    elif op == "add":
        pos = rng.randint(0, len(w))
        w = w[:pos] + rng.choice(string.ascii_lowercase) + w[pos:]
    words[idx] = w
    return " ".join(words)


def _synonym_perturb(query: str, rng: random.Random) -> str:
    """Lightweight synonym substitution (common words only, no external deps)."""
    _SYNONYMS = {
        "what": ["which", "what"],
        "how": ["in what way", "how"],
        "show": ["display", "list", "present"],
        "find": ["locate", "search for", "look up"],
        "get": ["retrieve", "fetch", "obtain"],
        "make": ["create", "build", "construct"],
        "big": ["large", "huge", "sizable"],
        "small": ["tiny", "little", "compact"],
        "good": ["great", "excellent", "fine"],
        "bad": ["poor", "terrible", "awful"],
        "important": ["significant", "crucial", "key"],
        "use": ["utilize", "employ", "apply"],
        "help": ["assist", "aid", "support"],
    }
    words = query.split()
    if not words:
        return query
    # Try to find a substitutable word
    candidates = [(i, w) for i, w in enumerate(words) if w.lower() in _SYNONYMS]
    if not candidates:
        return _typo_perturb(query, rng)  # fallback
    idx, word = rng.choice(candidates)
    replacement = rng.choice(_SYNONYMS[word.lower()])
    # Preserve original casing of first char
    if word[0].isupper():
        replacement = replacement.capitalize()
    words[idx] = replacement
    return " ".join(words)


def _drop_word_perturb(query: str, rng: random.Random) -> str:
    """Drop a random non-essential word."""
    words = query.split()
    if len(words) <= 2:
        return _typo_perturb(query, rng)
    idx = rng.randint(0, len(words) - 1)
    return " ".join(words[:idx] + words[idx + 1:])


def default_perturber(query: str, n: int, seed: int = 42) -> list[str]:
    """Generate n perturbed versions of query using mixed strategies."""
    rng = random.Random(seed)
    strategies = [_typo_perturb, _synonym_perturb, _drop_word_perturb]
    perturbations = []
    for _ in range(n):
        strategy = rng.choice(strategies)
        p = strategy(query, rng)
        # Ensure perturbation is actually different
        if p == query:
            p = _typo_perturb(query, rng)
        perturbations.append(p)
    return perturbations


# ── Core engine ──────────────────────────────────────────────────────────────

# retriever(query) -> list of (doc_id, score) in ranked order
RetrieverFn = Callable[[str], list[tuple[str, float]]]
PerturberFn = Callable[[str, int], list[str]]


def compute_instability(
    query: str,
    retriever: RetrieverFn,
    n_perturbations: int = 10,
    top_k: int = 20,
    perturber: PerturberFn | None = None,
    not_retrieved_rank: int | None = None,
) -> InstabilityReport:
    """Compute ProGRank perturbation instability for retrieved documents.

    Args:
        query: Original user query.
        retriever: Function(query) -> [(doc_id, score), ...] in ranked order.
        n_perturbations: Number of perturbed queries to generate.
        top_k: Consider only top-k results from each retrieval.
        perturber: Custom query perturbation function. If None, uses
                   default_perturber (typo + synonym + word-drop mix).
        not_retrieved_rank: Rank assigned when a doc is not in top-k for a
                           perturbed query. Defaults to top_k + 1.

    Returns:
        InstabilityReport with per-document instability scores.
    """
    perturb = perturber or (lambda q, n: default_perturber(q, n))
    fallback_rank = not_retrieved_rank if not_retrieved_rank is not None else top_k + 1

    # ── Step 1: original retrieval ───────────────────────────────────────
    original_results = retriever(query)[:top_k]
    original_ranking = {doc_id: rank for rank, (doc_id, _) in enumerate(original_results)}

    # Collect all doc_ids seen across all retrievals
    all_doc_ids = set(original_ranking.keys())

    # ── Step 2: perturbed retrievals ─────────────────────────────────────
    perturbations = perturb(query, n_perturbations)
    perturbed_rankings: list[dict[str, int]] = []

    for pq in perturbations:
        results = retriever(pq)[:top_k]
        ranking = {doc_id: rank for rank, (doc_id, _) in enumerate(results)}
        perturbed_rankings.append(ranking)
        all_doc_ids.update(ranking.keys())

    # ── Step 3: compute instability per document ─────────────────────────
    report = InstabilityReport(query=query, n_perturbations=n_perturbations)

    for doc_id in all_doc_ids:
        orig_rank = original_ranking.get(doc_id, fallback_rank)
        ranks = [pr.get(doc_id, fallback_rank) for pr in perturbed_rankings]

        all_ranks = np.array(ranks, dtype=float)
        mean_r = float(all_ranks.mean())
        std_r = float(all_ranks.std())

        # Coefficient of variation (PGR score)
        if mean_r > 1e-9:
            pgr = std_r / mean_r
        else:
            # Consistently rank 0 → perfectly stable
            pgr = 0.0

        report.results.append(InstabilityResult(
            doc_id=doc_id,
            pgr_score=pgr,
            mean_rank=mean_r,
            std_rank=std_r,
            ranks=ranks,
            original_rank=orig_rank,
        ))

    return report
