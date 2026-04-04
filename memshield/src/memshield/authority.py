"""
authority.py -- Authority prior scoring for RAG poisoning defense.

Each retrieved document gets a trust score A_i in [0, 1] based on provenance
signals that are hard for an attacker to forge:

  A_i = w_s * SourceTrust + w_d * DomainRep + w_p * Popularity + w_f * Freshness

Components:
  - SourceTrust: categorical trust based on ingestion source type
    (official docs > curated KB > web scrape > user upload > unknown)
  - DomainRep: domain-level reputation from a configurable allowlist/blocklist
    with optional PageRank-style external scoring
  - Popularity: entity co-occurrence popularity — do other trusted docs
    corroborate the same claims? (Jaccard overlap of entity sets)
  - Freshness: time-decay factor — older docs from trusted sources get
    a small bonus (established), while very new docs from unknown sources
    get penalized (potential injection window)

The authority prior acts as a Bayesian prior that down-weights suspicious
sources before the composite poison score is computed. A high-influence doc
from a trusted source is likely authoritative; a high-influence doc from an
unknown source is likely poisoned.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Source trust tiers ───────────────────────────────────────────────────────

# Default trust scores by source category. Configurable via AuthorityConfig.
DEFAULT_SOURCE_TRUST: dict[str, float] = {
    "official_docs": 0.95,
    "curated_kb": 0.85,
    "verified_api": 0.80,
    "web_scrape": 0.40,
    "user_upload": 0.30,
    "synthetic": 0.20,   # model-generated content (shadow memory)
    "unknown": 0.10,
}


@dataclass
class AuthorityConfig:
    """Configuration for authority prior computation."""
    source_trust: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SOURCE_TRUST))

    # Domain reputation: domain -> trust score
    domain_allowlist: dict[str, float] = field(default_factory=dict)
    domain_blocklist: set[str] = field(default_factory=set)

    # Component weights (must sum to ~1.0)
    w_source: float = 0.40
    w_domain: float = 0.25
    w_popularity: float = 0.20
    w_freshness: float = 0.15

    # Freshness parameters
    freshness_half_life_days: float = 90.0  # trust decays for unknown sources
    freshness_trusted_bonus: float = 0.1    # established docs from trusted sources


@dataclass
class AuthorityResult:
    """Authority prior score for a single document."""
    doc_id: str
    authority_score: float       # A_i in [0, 1]
    source_trust: float
    domain_reputation: float
    popularity: float
    freshness: float
    source_category: str
    domain: str


@dataclass
class AuthorityReport:
    """Authority report for a set of documents."""
    results: list[AuthorityResult] = field(default_factory=list)

    def get(self, doc_id: str) -> AuthorityResult | None:
        for r in self.results:
            if r.doc_id == doc_id:
                return r
        return None

    def scores_dict(self) -> dict[str, float]:
        """Return {doc_id: authority_score} mapping."""
        return {r.doc_id: r.authority_score for r in self.results}


# ── Authority engine ─────────────────────────────────────────────────────────

class AuthorityScorer:
    """Compute authority prior scores for retrieved documents."""

    def __init__(self, config: AuthorityConfig | None = None):
        self.config = config or AuthorityConfig()

    def score_documents(
        self,
        doc_ids: list[str],
        metadatas: list[dict[str, Any]],
        corpus_entities: dict[str, set[str]] | None = None,
    ) -> AuthorityReport:
        """Compute authority prior for each document.

        Args:
            doc_ids: Document identifiers.
            metadatas: Per-document metadata dicts. Expected keys:
                - provenance_source: source identifier (URL, path, category)
                - provenance_ts: ingestion timestamp (epoch seconds)
                - provenance_authority: manually assigned authority (override)
                - source_category: one of DEFAULT_SOURCE_TRUST keys
                - domain: source domain (e.g. "docs.python.org")
                - entities: set or list of named entities in the doc
            corpus_entities: Optional {doc_id: entity_set} for the full corpus,
                used for popularity (entity corroboration). If None, popularity
                defaults to 0.5.

        Returns:
            AuthorityReport with per-document scores.
        """
        if len(doc_ids) != len(metadatas):
            raise ValueError("doc_ids and metadatas must match in length")

        report = AuthorityReport()
        cfg = self.config

        for doc_id, meta in zip(doc_ids, metadatas):
            meta = meta or {}
            # ── Source trust ─────────────────────────────────────────────
            category = meta.get("source_category", "unknown")
            src_trust = cfg.source_trust.get(category, cfg.source_trust.get("unknown", 0.10))

            # Allow manual override from provenance
            if "provenance_authority" in meta:
                manual = float(meta["provenance_authority"])
                # Blend manual with categorical: take the higher if manual > 0.5
                src_trust = max(src_trust, manual) if manual > 0.5 else src_trust

            # ── Domain reputation ────────────────────────────────────────
            domain = meta.get("domain", _extract_domain(meta.get("provenance_source", "")))
            if domain in cfg.domain_blocklist:
                domain_rep = 0.0
            elif domain in cfg.domain_allowlist:
                domain_rep = cfg.domain_allowlist[domain]
            else:
                # Unknown domain — neutral
                domain_rep = 0.5

            # ── Popularity (entity corroboration) ────────────────────────
            doc_entities = set(meta.get("entities", []))
            if corpus_entities and doc_entities:
                popularity = _entity_corroboration(doc_id, doc_entities, corpus_entities)
            else:
                popularity = 0.5  # neutral when no entity data

            # ── Freshness ────────────────────────────────────────────────
            ts = meta.get("provenance_ts", 0.0)
            freshness = _compute_freshness(ts, src_trust, cfg)

            # ── Composite authority ──────────────────────────────────────
            authority = (
                cfg.w_source * src_trust
                + cfg.w_domain * domain_rep
                + cfg.w_popularity * popularity
                + cfg.w_freshness * freshness
            )
            authority = max(0.0, min(1.0, authority))

            report.results.append(AuthorityResult(
                doc_id=doc_id,
                authority_score=authority,
                source_trust=src_trust,
                domain_reputation=domain_rep,
                popularity=popularity,
                freshness=freshness,
                source_category=category,
                domain=domain,
            ))

        return report


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_domain(source: str) -> str:
    """Best-effort domain extraction from a source string."""
    if "://" in source:
        # URL-like: extract domain
        after_scheme = source.split("://", 1)[1]
        domain = after_scheme.split("/", 1)[0].split(":", 1)[0]
        return domain.lower()
    return ""


def _entity_corroboration(
    doc_id: str,
    doc_entities: set[str],
    corpus_entities: dict[str, set[str]],
) -> float:
    """Fraction of this doc's entities that appear in other docs (corroboration).

    High corroboration → entities are well-established in the corpus → trustworthy.
    Low corroboration → unique claims → potentially injected.
    """
    if not doc_entities:
        return 0.5

    corroborated = 0
    for entity in doc_entities:
        # Count how many OTHER docs contain this entity
        other_count = sum(
            1 for other_id, other_ents in corpus_entities.items()
            if other_id != doc_id and entity in other_ents
        )
        if other_count > 0:
            corroborated += 1

    return corroborated / len(doc_entities)


def _compute_freshness(
    ingestion_ts: float,
    source_trust: float,
    cfg: AuthorityConfig,
) -> float:
    """Compute freshness score.

    - Trusted sources (high src_trust) get a small bonus for being established.
    - Unknown sources get penalized if very new (potential injection window).
    - Very old docs from unknown sources also decay (stale injection attempt
      that was never cleaned up).
    """
    if ingestion_ts <= 0:
        # No timestamp — neutral
        return 0.5

    age_days = (time.time() - ingestion_ts) / 86400.0

    if source_trust >= 0.7:
        # Trusted source: established docs are good, slight freshness bonus
        return min(1.0, 0.7 + cfg.freshness_trusted_bonus * min(age_days / 30.0, 1.0))

    # Untrusted source: exponential decay
    # Very new (< 1 day): suspicious injection window → lower score
    # Moderate age: slightly better (survived without cleanup)
    # Very old: decay (stale, uncleaned)
    half_life = cfg.freshness_half_life_days
    if age_days < 1.0:
        return 0.3  # fresh unknown source — suspicious
    decay = 0.5 ** (age_days / half_life)
    # Peak around 7-30 days, then decay
    return min(0.7, 0.5 + 0.2 * (1.0 - decay))
