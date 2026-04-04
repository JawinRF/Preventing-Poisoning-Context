"""
scorer.py -- Composite poison score and reranking for RAG poisoning defense.

Combines all defense signals into a single poison probability:

  PoisonScore_i = sigma(w1*PGR_i + w2*M_i + w3*I_i + w4*Copy_i - w5*A_i + w6*Tamper_i + bias)

Where:
  PGR_i    = ProGRank perturbation instability (progrank.py)
  M_i      = RAGMask token fragility (ragmask.py)
  I_i      = Leave-one-out influence score (influence.py)
  Copy_i   = Copy/verbatim overlap ratio (inline, simple)
  A_i      = Authority prior (authority.py) — SUBTRACTIVE (trusted = less suspicious)
  Tamper_i = Provenance tamper flag (provenance.py) — binary 0/1

sigma() is the logistic sigmoid, mapping the linear combination to [0, 1].

Reranking: documents are sorted by (1 - PoisonScore) * original_relevance,
so poisoned documents sink to the bottom of results.

Weight tuning: weights can be set manually or learned from labeled data
using logistic regression on the signal vector.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ez = math.exp(x)
    return ez / (1.0 + ez)


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ScorerWeights:
    """Weights for composite poison score.

    Default values are a reasonable starting point.
    Tune with labeled data using PoisonScorer.fit().
    """
    w_pgr: float = 1.5       # ProGRank instability
    w_mask: float = 0.8      # RAGMask token fragility
    w_influence: float = 1.2  # Leave-one-out influence
    w_copy: float = 0.6      # Verbatim copy ratio
    w_authority: float = 2.0  # Authority prior (subtractive — higher = safer)
    w_tamper: float = 3.0     # Provenance tamper flag (binary, high weight)
    bias: float = -1.0        # Negative bias: default to "not poisoned"

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.w_pgr, self.w_mask, self.w_influence,
            self.w_copy, -self.w_authority, self.w_tamper, self.bias,
        ])


# ── Signal vector ────────────────────────────────────────────────────────────

@dataclass
class SignalVector:
    """Raw signal values for a single document."""
    doc_id: str
    pgr: float = 0.0          # ProGRank instability
    mask_fragility: float = 0.0  # RAGMask peakiness
    influence: float = 0.0    # Leave-one-out influence
    copy_ratio: float = 0.0   # Verbatim overlap with query/other docs
    authority: float = 0.5    # Authority prior
    tamper: float = 0.0       # 1.0 if provenance check failed, 0.0 otherwise
    original_score: float = 1.0  # Original retrieval relevance score

    def to_feature_vector(self) -> np.ndarray:
        """Feature vector for scoring (includes bias term)."""
        return np.array([
            self.pgr, self.mask_fragility, self.influence,
            self.copy_ratio, self.authority, self.tamper, 1.0,  # 1.0 = bias term
        ])


@dataclass
class ScoredDocument:
    """A document with its composite poison score and reranked position."""
    doc_id: str
    poison_score: float        # sigma(w . x) in [0, 1]
    reranked_score: float      # (1 - poison_score) * original_relevance
    verdict: str               # ALLOW | QUARANTINE | BLOCK
    signals: SignalVector
    rank: int = 0


@dataclass
class ScoringReport:
    """Full scoring report for a retrieval set."""
    documents: list[ScoredDocument] = field(default_factory=list)
    weights: ScorerWeights = field(default_factory=ScorerWeights)

    def reranked(self) -> list[ScoredDocument]:
        """Return documents sorted by reranked_score (best first)."""
        docs = sorted(self.documents, key=lambda d: d.reranked_score, reverse=True)
        for i, d in enumerate(docs):
            d.rank = i
        return docs

    def blocked(self) -> list[ScoredDocument]:
        return [d for d in self.documents if d.verdict == "BLOCK"]

    def quarantined(self) -> list[ScoredDocument]:
        return [d for d in self.documents if d.verdict == "QUARANTINE"]

    def allowed(self) -> list[ScoredDocument]:
        return [d for d in self.documents if d.verdict == "ALLOW"]


# ── Scorer ───────────────────────────────────────────────────────────────────

class PoisonScorer:
    """Composite poison scoring and reranking engine."""

    def __init__(
        self,
        weights: ScorerWeights | None = None,
        block_threshold: float = 0.75,
        quarantine_threshold: float = 0.50,
    ):
        self.weights = weights or ScorerWeights()
        self.block_threshold = block_threshold
        self.quarantine_threshold = quarantine_threshold

    def score(self, signals: list[SignalVector]) -> ScoringReport:
        """Score a batch of documents and produce verdicts + reranking.

        Args:
            signals: Per-document signal vectors.

        Returns:
            ScoringReport with scored documents and reranking.
        """
        report = ScoringReport(weights=self.weights)
        w = self.weights.to_vector()

        for sv in signals:
            x = sv.to_feature_vector()
            logit = float(np.dot(w, x))
            poison_prob = _sigmoid(logit)

            # Reranked score: down-weight suspicious docs
            reranked = (1.0 - poison_prob) * sv.original_score

            # Verdict
            if poison_prob >= self.block_threshold:
                verdict = "BLOCK"
            elif poison_prob >= self.quarantine_threshold:
                verdict = "QUARANTINE"
            else:
                verdict = "ALLOW"

            report.documents.append(ScoredDocument(
                doc_id=sv.doc_id,
                poison_score=poison_prob,
                reranked_score=reranked,
                verdict=verdict,
                signals=sv,
            ))

        return report

    def fit(
        self,
        labeled_signals: list[SignalVector],
        labels: list[int],
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        l2_lambda: float = 0.01,
    ) -> ScorerWeights:
        """Fit weights from labeled data using logistic regression with L2.

        Args:
            labeled_signals: Signal vectors for labeled examples.
            labels: 1 = poisoned, 0 = clean.
            learning_rate: Gradient descent step size.
            n_iterations: Number of gradient descent iterations.
            l2_lambda: L2 regularization strength.

        Returns:
            Updated ScorerWeights (also updates self.weights in place).
        """
        n = len(labeled_signals)
        if n != len(labels):
            raise ValueError("signals and labels must match")
        if n == 0:
            return self.weights

        X = np.array([sv.to_feature_vector() for sv in labeled_signals])
        y = np.array(labels, dtype=float)
        w = self.weights.to_vector().copy()

        for it in range(n_iterations):
            logits = X @ w
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))

            # Gradient: X^T (preds - y) / n + L2
            grad = X.T @ (preds - y) / n + l2_lambda * w
            w -= learning_rate * grad

            if it % 200 == 0:
                loss = -np.mean(y * np.log(preds + 1e-12) + (1 - y) * np.log(1 - preds + 1e-12))
                logger.debug(f"fit iter={it} loss={loss:.4f}")

        # Update weights from learned vector
        self.weights = ScorerWeights(
            w_pgr=float(w[0]),
            w_mask=float(w[1]),
            w_influence=float(w[2]),
            w_copy=float(w[3]),
            w_authority=float(-w[4]),  # stored as negative in vector
            w_tamper=float(w[5]),
            bias=float(w[6]),
        )
        return self.weights


# ── Verbatim copy detection ──────────────────────────────────────────────────

def compute_copy_ratio(doc: str, query: str, other_docs: list[str] | None = None,
                       ngram_size: int = 4) -> float:
    """Compute verbatim overlap ratio between doc and query/other docs.

    Uses character n-gram overlap (Jaccard of n-grams).
    High copy ratio suggests the doc was crafted to mirror the query
    (a common poisoning technique to boost retrieval rank).
    """
    doc_ngrams = _char_ngrams(doc.lower(), ngram_size)
    if not doc_ngrams:
        return 0.0

    # Overlap with query
    query_ngrams = _char_ngrams(query.lower(), ngram_size)
    query_overlap = len(doc_ngrams & query_ngrams) / len(doc_ngrams) if doc_ngrams else 0.0

    # Overlap with other docs (detect near-duplicates / coordinated injection)
    max_doc_overlap = 0.0
    if other_docs:
        for other in other_docs:
            other_ngrams = _char_ngrams(other.lower(), ngram_size)
            if doc_ngrams and other_ngrams:
                overlap = len(doc_ngrams & other_ngrams) / len(doc_ngrams | other_ngrams)
                max_doc_overlap = max(max_doc_overlap, overlap)

    return max(query_overlap, max_doc_overlap)


def _char_ngrams(text: str, n: int) -> set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}
