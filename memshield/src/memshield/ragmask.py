"""
ragmask.py -- RAGMask token-masking fragility for RAG poisoning detection.

Idea (from RAGMask / token-fragility analysis):
  For a retrieved document d with tokens [t_1, ..., t_n]:
    1. Compute baseline relevance: sim_0 = cos(E(q), E(d))
    2. For each token t_j, mask it:  d_j = d with t_j replaced by [MASK]
    3. Compute masked relevance:     sim_j = cos(E(q), E(d_j))
    4. Token fragility:              f_j = max(0, sim_0 - sim_j)
    5. Document fragility:           M_d = max(f_j) / mean(f_j)  [peakiness]

High M_d means the document's retrieval relevance depends on a small number
of trigger tokens -- characteristic of poisoned documents that embed specific
adversarial phrases to hijack retrieval ranking.

Clean documents have diffuse relevance across many tokens (low M_d).
Poisoned documents concentrate relevance in trigger tokens (high M_d).

Optimization: for long documents, sample tokens rather than exhaustive masking.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenFragility:
    """Fragility score for a single token."""
    token: str
    index: int
    fragility: float  # max(0, sim_0 - sim_j)


@dataclass
class FragilityResult:
    """RAGMask fragility analysis for a single document."""
    doc_id: str
    doc_text: str
    baseline_sim: float
    fragility_score: float       # M_d = max(f_j) / mean(f_j)  (peakiness ratio)
    max_token_fragility: float   # max(f_j)
    mean_token_fragility: float  # mean(f_j)
    top_tokens: list[TokenFragility] = field(default_factory=list)
    n_tokens_evaluated: int = 0


@dataclass
class FragilityReport:
    """RAGMask fragility report for a set of retrieved documents."""
    query: str
    results: list[FragilityResult] = field(default_factory=list)

    def flagged(self, threshold: float = 5.0) -> list[FragilityResult]:
        """Return docs whose fragility score exceeds threshold."""
        return [r for r in self.results if r.fragility_score >= threshold]

    def ranked(self, descending: bool = True) -> list[FragilityResult]:
        """Return results sorted by fragility score."""
        return sorted(self.results, key=lambda r: r.fragility_score, reverse=descending)


# ── Tokenization ─────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r'\S+')

def _tokenize(text: str) -> list[tuple[int, int, str]]:
    """Split text into whitespace-delimited tokens with character spans."""
    return [(m.start(), m.end(), m.group()) for m in _TOKEN_RE.finditer(text)]


def _mask_token(text: str, start: int, end: int, mask: str = "[MASK]") -> str:
    """Replace a single token span with the mask token."""
    return text[:start] + mask + text[end:]


# ── Core ─────────────────────────────────────────────────────────────────────

EmbedderFn = Callable[[str], np.ndarray]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_fragility(
    query: str,
    documents: list[str],
    doc_ids: list[str],
    embedder: EmbedderFn,
    max_tokens_per_doc: int = 200,
    top_k_tokens: int = 10,
    mask_token: str = "[MASK]",
) -> FragilityReport:
    """Compute RAGMask token-masking fragility for each document.

    Args:
        query: The user query.
        documents: Retrieved document texts.
        doc_ids: Corresponding document IDs.
        embedder: E(text) -> embedding vector. Called O(n_tokens) times per doc.
        max_tokens_per_doc: If a doc has more tokens, uniformly sample this many.
        top_k_tokens: Number of top-fragility tokens to keep per doc.
        mask_token: Replacement string for masked tokens.

    Returns:
        FragilityReport with per-document fragility scores.
    """
    if len(documents) != len(doc_ids):
        raise ValueError("documents and doc_ids must have same length")

    report = FragilityReport(query=query)
    q_emb = embedder(query)

    for doc, doc_id in zip(documents, doc_ids):
        tokens = _tokenize(doc)
        if not tokens:
            report.results.append(FragilityResult(
                doc_id=doc_id, doc_text=doc, baseline_sim=0.0,
                fragility_score=0.0, max_token_fragility=0.0,
                mean_token_fragility=0.0, n_tokens_evaluated=0,
            ))
            continue

        # Baseline similarity
        d_emb = embedder(doc)
        sim_0 = _cosine(q_emb, d_emb)

        # Sample tokens if document is long
        eval_indices = list(range(len(tokens)))
        if len(tokens) > max_tokens_per_doc:
            rng = np.random.default_rng(hash(doc_id) & 0xFFFFFFFF)
            eval_indices = sorted(rng.choice(len(tokens), size=max_tokens_per_doc, replace=False))

        fragilities: list[TokenFragility] = []
        for idx in eval_indices:
            start, end, tok_text = tokens[idx]
            masked_doc = _mask_token(doc, start, end, mask_token)
            masked_emb = embedder(masked_doc)
            sim_j = _cosine(q_emb, masked_emb)
            f_j = max(0.0, sim_0 - sim_j)
            fragilities.append(TokenFragility(token=tok_text, index=idx, fragility=f_j))

        # Compute peakiness: max / mean
        f_values = np.array([tf.fragility for tf in fragilities])
        max_f = float(f_values.max()) if len(f_values) > 0 else 0.0
        mean_f = float(f_values.mean()) if len(f_values) > 0 else 0.0

        if mean_f > 1e-9:
            peakiness = max_f / mean_f
        else:
            # All tokens have ~0 fragility → flat → not suspicious
            peakiness = 1.0

        # Keep top-k most fragile tokens for interpretability
        top = sorted(fragilities, key=lambda t: t.fragility, reverse=True)[:top_k_tokens]

        report.results.append(FragilityResult(
            doc_id=doc_id,
            doc_text=doc,
            baseline_sim=sim_0,
            fragility_score=peakiness,
            max_token_fragility=max_f,
            mean_token_fragility=mean_f,
            top_tokens=top,
            n_tokens_evaluated=len(eval_indices),
        ))

    return report
