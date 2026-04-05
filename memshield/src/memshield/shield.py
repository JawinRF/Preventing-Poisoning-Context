"""
shield.py — MemShield core: defense-in-depth RAG poisoning defense.

Ingest-time scanning pipeline (scan_chunk):
  0. Text normalization / deobfuscation (base64, unicode, zero-width)
  1. Injection pattern matching (high confidence → BLOCK)
  2. Suspicious pattern matching (medium confidence → QUARANTINE)
  3. Statistical anomaly detection (long chunks, high symbol density)
  4. TinyBERT ML classifier (fine-tuned for prompt injection)
  5. DeBERTa ML classifier (ProtectAI prompt injection detector)

Retrieval-time defense pipeline (query → _filter_results → _score_retrieval_set):
  - Cryptographic provenance verification (tamper detection)
  - Leave-one-out influence scoring (semantic + citation drift)
  - RAGMask token-masking fragility (trigger token concentration)
  - Authority prior (source trust, domain reputation, entity corroboration)
  - Verbatim copy detection (query-mirroring attacks)
  - ProGRank perturbation instability (optional, expensive)
  - Composite poison scorer: σ(w·x) → ALLOW / QUARANTINE / BLOCK
  - Reranking: (1 - poison_score) × relevance
"""
from __future__ import annotations

import re, uuid, sys, os, logging
from dataclasses import dataclass
from typing import Any, Callable
from pathlib import Path
from .audit import AuditLogger
from .provenance import ContentHasher
from .influence import compute_influence
from .ragmask import compute_fragility
from .authority import AuthorityScorer, AuthorityConfig
from .progrank import compute_instability
from .scorer import PoisonScorer, ScorerWeights, SignalVector, compute_copy_ratio
from .config import (
    _INJECTION_PATTERNS, _SUSPICIOUS_PATTERNS,
    ShieldConfig, FailurePolicy,
)

logger = logging.getLogger(__name__)

# ── Lazy imports for PRISM modules ───────────────────────────────────────────
# These live in scripts/ and require sys.path setup. Graceful degradation
# if not available — shield still works with regex-only.

_NORMALIZER_AVAILABLE = False
_ML_AVAILABLE = False

def _ensure_scripts_path():
    """Add scripts/ to sys.path so prism_shield package is importable."""
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))),
        "scripts",
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

try:
    _ensure_scripts_path()
    from prism_shield.normalizer import Normalizer as _Normalizer
    from prism_shield.base import MemoryEntry as _MemoryEntry
    _NORMALIZER_AVAILABLE = True
except ImportError:
    _Normalizer = None  # type: ignore[assignment,misc]
    _MemoryEntry = None  # type: ignore[assignment,misc]

try:
    _ensure_scripts_path()
    from prism_shield.layer2_local_llm import LocalLLMValidator as _LocalLLMValidator
    from prism_shield.layer3_deberta import DeBERTaValidator as _DeBERTaValidator
    _ML_AVAILABLE = True
except ImportError:
    _LocalLLMValidator = None  # type: ignore[assignment,misc]
    _DeBERTaValidator = None  # type: ignore[assignment,misc]


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ShieldResult:
    verdict: str        # ALLOW | BLOCK | QUARANTINE
    confidence: float
    reason: str
    chunk_id: str
    chunk_text: str
    pattern_matched: str | None = None
    layer_triggered: str | None = None


# ── Core shield ───────────────────────────────────────────────────────────────

class MemShield:
    """
    Defense-in-depth RAG poisoning scanner.

    Wraps a ChromaDB collection (optional) and provides multi-layer
    scanning at both ingest and retrieval time.

    Ingest-time: normalization → regex → statistical → ML
    Retrieval-time: provenance → influence → ragmask → authority → scorer → rerank

    Failure policy: FAIL_CLOSED — on any error, block the chunk.
    """

    def __init__(
        self,
        collection=None,
        audit_log: str | Path = "data/memshield_audit.jsonl",
        fail_policy: str = "FAIL_CLOSED",
        quarantine_path: str | Path = "data/memshield_quarantine.jsonl",
        config: ShieldConfig | None = None,
        generator: Callable[[str, list[str]], str] | None = None,
        embedder: Callable[[str], Any] | None = None,
        authority_config: AuthorityConfig | None = None,
        scorer_weights: ScorerWeights | None = None,
        **kwargs,  # absorb legacy 'strategy' kwarg for backward compat
    ):
        self.collection    = collection
        self.config        = config or ShieldConfig()
        self.fail_policy   = fail_policy
        self.auditor       = AuditLogger(audit_log)
        self.quarantine    = Path(quarantine_path)
        self.quarantine.parent.mkdir(parents=True, exist_ok=True)

        # Normalization layer
        self._normalizer = None
        if self.config.enable_normalization:
            if _NORMALIZER_AVAILABLE:
                self._normalizer = _Normalizer()
                logger.info("[MemShield] Normalization: ON")
            else:
                raise ImportError(
                    "MemShield: enable_normalization=True but prism_shield.normalizer "
                    "not importable. Set enable_normalization=False to proceed without."
                )

        # ML layers
        self._tinybert = None
        self._deberta = None
        if self.config.enable_ml_layers:
            if not _ML_AVAILABLE:
                raise ImportError(
                    "MemShield: enable_ml_layers=True but prism_shield ML modules "
                    "not importable. Install torch/transformers or set enable_ml_layers=False."
                )
            self._tinybert = _LocalLLMValidator(self.config.ml_model_path)
            self._deberta = _DeBERTaValidator()
            logger.info("[MemShield] ML Layers: ON (TinyBERT + DeBERTa)")

        # Retrieval defense pipeline
        self._generator = generator
        self._embedder = embedder
        self._authority_scorer = AuthorityScorer(authority_config)
        self._poison_scorer = PoisonScorer(
            weights=scorer_weights,
            block_threshold=self.config.retrieval_block_threshold,
            quarantine_threshold=self.config.retrieval_quarantine_threshold,
        )

        if self.config.enable_retrieval_defense:
            if not embedder:
                raise ValueError(
                    "MemShield: enable_retrieval_defense=True requires an embedder callable. "
                    "Pass embedder=func where func(text) -> numpy array."
                )
            logger.info("[MemShield] Retrieval defense: ON (influence + ragmask + authority + scorer)")
            if not generator:
                logger.warning(
                    "[MemShield] No generator provided — influence scoring disabled "
                    "(ragmask + authority + copy + provenance still active)"
                )

        if self.config.enable_provenance:
            logger.info("[MemShield] Provenance: ON (SHA-256 content hash verification)")

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        session_id: str = "default",
        **kwargs,
    ) -> dict:
        """
        Drop-in replacement for collection.query().
        Poisoned chunks are removed from results and audit-logged.

        When enable_retrieval_defense is True, surviving chunks go through
        the full pipeline (influence, ragmask, authority, scorer) and are
        reranked by (1 - poison_score).
        """
        try:
            raw = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                **kwargs,
            )
        except Exception as exc:
            logger.error(f"ChromaDB query failed: {exc}")
            if self.fail_policy == "FAIL_CLOSED":
                return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
            raise

        # Pass the first query text for retrieval defense scoring
        query_text = query_texts[0] if query_texts else ""
        return self._filter_results(raw, session_id, query_text=query_text)

    def scan_chunk(self, text: str, chunk_id: str = "") -> ShieldResult:
        """Scan a single chunk through all enabled layers. Returns ShieldResult."""
        chunk_id = chunk_id or str(uuid.uuid4())[:8]

        # ── Layer 0: Normalization / deobfuscation ───────────────────────
        scan_text = text
        if self._normalizer and _MemoryEntry:
            try:
                entry = _MemoryEntry(id="", text=text, ingestion_path="rag_store")
                scan_text = self._normalizer.normalize(entry)
            except Exception as exc:
                logger.warning(f"Normalization failed: {exc}")
                return ShieldResult(
                    verdict="BLOCK",
                    confidence=0.90,
                    reason=f"Normalization failed (fail-closed): {exc}",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    layer_triggered="Layer0-Normalization",
                )

        # ── Layer 1: Injection patterns (high confidence → BLOCK) ────────
        for pat in _INJECTION_PATTERNS:
            if pat.search(scan_text):
                return ShieldResult(
                    verdict="BLOCK",
                    confidence=0.97,
                    reason=f"Injection pattern matched: {pat.pattern[:60]}",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    pattern_matched=pat.pattern,
                    layer_triggered="Layer1-Regex",
                )

        # ── Layer 2: Suspicious patterns (medium confidence → QUARANTINE)
        for pat in _SUSPICIOUS_PATTERNS:
            if pat.search(scan_text):
                return ShieldResult(
                    verdict="QUARANTINE",
                    confidence=0.72,
                    reason=f"Suspicious pattern matched: {pat.pattern[:60]}",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    pattern_matched=pat.pattern,
                    layer_triggered="Layer2-Regex",
                )

        # ── Layer 3: Statistical anomaly ─────────────────────────────────
        if len(scan_text) > 2000:
            symbol_ratio = sum(
                1 for c in scan_text if not c.isalnum() and not c.isspace()
            ) / len(scan_text)
            if symbol_ratio > 0.35:
                return ShieldResult(
                    verdict="QUARANTINE",
                    confidence=0.65,
                    reason=f"Statistical anomaly: high symbol density ({symbol_ratio:.2f})",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    layer_triggered="Layer3-Stats",
                )

        # ── Layer 4: TinyBERT ML classifier ──────────────────────────────
        if self._tinybert:
            try:
                ml_result = self._tinybert.evaluate(scan_text, ingestion_path="rag_store")
                if ml_result.verdict != "ALLOW":
                    return ShieldResult(
                        verdict=ml_result.verdict,
                        confidence=ml_result.confidence,
                        reason=ml_result.reason,
                        chunk_id=chunk_id,
                        chunk_text=text,
                        layer_triggered="Layer4-TinyBERT",
                    )
            except Exception as exc:
                logger.warning(f"TinyBERT evaluation failed: {exc}")
                return ShieldResult(
                    verdict="BLOCK",
                    confidence=0.85,
                    reason=f"ML evaluation failed (fail-closed): {exc}",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    layer_triggered="Layer4-TinyBERT",
                )

        # ── Layer 5: DeBERTa ML classifier ───────────────────────────────
        if self._deberta:
            try:
                ml_result = self._deberta.evaluate(scan_text, ingestion_path="rag_store")
                if ml_result.verdict != "ALLOW":
                    return ShieldResult(
                        verdict=ml_result.verdict,
                        confidence=ml_result.confidence,
                        reason=ml_result.reason,
                        chunk_id=chunk_id,
                        chunk_text=text,
                        layer_triggered="Layer5-DeBERTa",
                    )
            except Exception as exc:
                logger.warning(f"DeBERTa evaluation failed: {exc}")
                return ShieldResult(
                    verdict="BLOCK",
                    confidence=0.85,
                    reason=f"ML evaluation failed (fail-closed): {exc}",
                    chunk_id=chunk_id,
                    chunk_text=text,
                    layer_triggered="Layer5-DeBERTa",
                )

        # ── All layers passed ────────────────────────────────────────────
        return ShieldResult(
            verdict="ALLOW",
            confidence=0.95,
            reason="No injection patterns detected",
            chunk_id=chunk_id,
            chunk_text=text,
            layer_triggered="none",
        )

    def scan(self, chunks: list[str]) -> list[tuple[str, bool, str]]:
        """
        Scan a list of text chunks for poisoning.
        Returns: [(chunk_text, is_poisoned, reason), ...]
        """
        results = []
        for chunk in chunks:
            sr = self.scan_chunk(chunk)
            is_poisoned = sr.verdict != "ALLOW"
            results.append((chunk, is_poisoned, sr.reason))
        return results

    def validate_reads(self, documents: list[dict]) -> list[dict]:
        """
        Filter a list of document dicts, returning only those that pass scanning.
        Each document should have a 'content' key with the text to scan.
        """
        allowed = []
        for doc in documents:
            text = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            sr = self.scan_chunk(text)
            if sr.verdict == "ALLOW":
                allowed.append(doc)
            else:
                logger.warning(f"validate_reads BLOCKED: {sr.reason}")
        return allowed

    def ingest_with_scan(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        source: str = "unknown",
        authority: float = 0.5,
        session_id: str = "ingest",
    ) -> dict:
        """
        Scan documents through all enabled layers BEFORE adding to ChromaDB.
        Only clean documents are stored with full provenance. Returns stats dict.
        """
        if self.collection is None:
            raise ValueError("No ChromaDB collection configured")
        if metadatas is None:
            metadatas = [{} for _ in documents]

        accepted_docs, accepted_ids, accepted_meta = [], [], []
        stats: dict[str, Any] = {"accepted": 0, "blocked": 0, "quarantined": 0, "details": []}

        for doc, doc_id, meta in zip(documents, ids, metadatas):
            result = self.scan_chunk(doc, chunk_id=doc_id)

            self.auditor.log_retrieval(
                verdict=result.verdict,
                confidence=result.confidence,
                reason=result.reason,
                chunk_id=doc_id,
                chunk_text=doc,
                collection=getattr(self.collection, "name", "unknown"),
                session_id=session_id,
                metadata={"event": "ingest_scan", "source": source},
            )

            if result.verdict == "ALLOW":
                accepted_docs.append(doc)
                accepted_ids.append(doc_id)
                accepted_meta.append({**meta, "source": source})
                stats["accepted"] += 1
            elif result.verdict == "QUARANTINE":
                self._quarantine_chunk(doc, doc_id, result)
                stats["quarantined"] += 1
            else:
                stats["blocked"] += 1

            stats["details"].append({
                "id": doc_id, "verdict": result.verdict, "reason": result.reason,
            })

        if accepted_docs:
            self.add_with_provenance(
                documents=accepted_docs,
                ids=accepted_ids,
                metadatas=accepted_meta,
                source=source,
                authority=authority,
            )

        return stats

    def add_with_provenance(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict] | None = None,
        source: str = "unknown",
        authority: float = 0.5,
        **kwargs,
    ) -> None:
        """Add documents to ChromaDB with canonical hashes + full provenance."""
        if self.collection is None:
            raise ValueError("No ChromaDB collection configured")
        if metadatas is None:
            metadatas = [{} for _ in documents]
        hashed_meta = [
            ContentHasher.hash_and_attach(
                doc, meta, source=source, authority=authority, chunk_id=doc_id,
            )
            for doc, meta, doc_id in zip(documents, metadatas, ids)
        ]
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=hashed_meta,
            **kwargs,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _filter_results(self, raw: dict, session_id: str, query_text: str = "") -> dict:
        """Remove blocked/quarantined chunks from ChromaDB results.

        Two-phase filtering:
          Phase 1: Per-chunk scan (regex/ML/provenance) — fast, independent
          Phase 2: Cross-document retrieval defense (influence/ragmask/authority/scorer)
                   — requires the full surviving set, runs only if enabled
        """
        if not raw.get("documents"):
            return raw

        filtered_docs, filtered_meta, filtered_ids = [], [], []

        raw_distances = raw.get("distances") or [[] for _ in raw["documents"]]

        for batch_idx, (batch_docs, batch_meta, batch_ids) in enumerate(zip(
            raw["documents"], raw["metadatas"], raw["ids"]
        )):
            batch_dists = raw_distances[batch_idx] if batch_idx < len(raw_distances) else []

            # ── Phase 1: per-chunk scan + provenance ─────────────────────
            phase1_docs, phase1_meta, phase1_ids = [], [], []
            phase1_dists: list[float] = []
            tamper_flags: dict[str, float] = {}

            for i, (doc, meta, cid) in enumerate(zip(batch_docs, batch_meta or [], batch_ids)):

                # Provenance verification
                if self.config.enable_provenance and not ContentHasher.verify(doc, meta):
                    result = ShieldResult(
                        verdict="BLOCK",
                        confidence=0.99,
                        reason="Provenance check failed: content hash mismatch "
                               "(possible post-ingestion tampering)",
                        chunk_id=cid,
                        chunk_text=doc,
                        layer_triggered="Provenance",
                    )
                    tamper_flags[cid] = 1.0
                else:
                    result = self.scan_chunk(doc, chunk_id=cid)
                    tamper_flags[cid] = 0.0

                self.auditor.log_retrieval(
                    verdict=result.verdict,
                    confidence=result.confidence,
                    reason=result.reason,
                    chunk_id=cid,
                    chunk_text=doc,
                    collection=getattr(self.collection, "name", "unknown"),
                    session_id=session_id,
                    metadata=meta or {},
                )

                if result.verdict == "ALLOW":
                    phase1_docs.append(doc)
                    phase1_meta.append(meta)
                    phase1_ids.append(cid)
                    phase1_dists.append(batch_dists[i] if i < len(batch_dists) else 0.0)
                elif result.verdict == "QUARANTINE":
                    self._quarantine_chunk(doc, cid, result)
                    logger.warning(f"QUARANTINED chunk {cid}: {result.reason}")
                else:
                    logger.warning(f"BLOCKED chunk {cid}: {result.reason}")

            # ── Phase 2: retrieval defense (cross-document scoring) ──────
            if self.config.enable_retrieval_defense and phase1_docs and self._embedder:
                scored = self._score_retrieval_set(
                    query_text, phase1_docs, phase1_ids, phase1_meta,
                    phase1_dists, tamper_flags, session_id,
                )
                clean_docs, clean_meta, clean_ids = [], [], []
                for sd in scored:
                    idx = phase1_ids.index(sd.doc_id)
                    if sd.verdict == "ALLOW":
                        clean_docs.append(phase1_docs[idx])
                        clean_meta.append(phase1_meta[idx])
                        clean_ids.append(sd.doc_id)
                    elif sd.verdict == "QUARANTINE":
                        self._quarantine_chunk(
                            phase1_docs[idx], sd.doc_id,
                            ShieldResult(
                                verdict="QUARANTINE",
                                confidence=sd.poison_score,
                                reason=f"Retrieval defense: poison_score={sd.poison_score:.3f}",
                                chunk_id=sd.doc_id,
                                chunk_text=phase1_docs[idx],
                                layer_triggered="RetrievalDefense-Scorer",
                            ),
                        )
                        logger.warning(f"QUARANTINED chunk {sd.doc_id}: poison_score={sd.poison_score:.3f}")
                    else:
                        logger.warning(f"BLOCKED chunk {sd.doc_id}: poison_score={sd.poison_score:.3f}")
            else:
                clean_docs, clean_meta, clean_ids = phase1_docs, phase1_meta, phase1_ids

            filtered_docs.append(clean_docs)
            filtered_meta.append(clean_meta)
            filtered_ids.append(clean_ids)

        raw["documents"] = filtered_docs
        raw["metadatas"] = filtered_meta
        raw["ids"]       = filtered_ids
        return raw

    def _score_retrieval_set(
        self,
        query: str,
        docs: list[str],
        doc_ids: list[str],
        metadatas: list[dict],
        distances: list[float],
        tamper_flags: dict[str, float],
        session_id: str,
    ) -> list:
        """Run the full retrieval defense pipeline on Phase 1 survivors.

        Computes per-document signal vectors, then scores with the composite
        poison scorer. Returns ScoredDocuments in reranked order.
        """
        import numpy as np
        from .scorer import ScoredDocument

        k = len(docs)
        signals: list[SignalVector] = []

        # Convert ChromaDB distances to relevance scores in [0, 1].
        # ChromaDB returns L2 distances (lower = more relevant).
        relevance_scores: dict[str, float] = {}
        for did, dist in zip(doc_ids, distances):
            relevance_scores[did] = 1.0 / (1.0 + dist) if dist else 1.0

        # ── Authority prior ──────────────────────────────────────────────
        # Bridge provenance metadata to authority scorer fields.
        # add_with_provenance stores provenance_source/provenance_authority;
        # authority scorer expects source_category/provenance_authority.
        enriched_meta = []
        for m in metadatas:
            em = dict(m) if m else {}
            # If source_category not set, derive from provenance_source
            if "source_category" not in em and "provenance_source" in em:
                em["source_category"] = em["provenance_source"]
            # If source was passed as "source" key (from ingest_with_scan)
            if "source_category" not in em and "source" in em:
                em["source_category"] = em["source"]
            enriched_meta.append(em)
        authority_report = self._authority_scorer.score_documents(doc_ids, enriched_meta)
        auth_map = authority_report.scores_dict()

        # ── RAGMask token fragility ──────────────────────────────────────
        try:
            frag_report = compute_fragility(query, docs, doc_ids, self._embedder)
            frag_map = {r.doc_id: r.fragility_score for r in frag_report.results}
        except Exception as exc:
            logger.warning(f"RAGMask fragility failed: {exc}")
            frag_map = {did: 0.0 for did in doc_ids}

        # ── Leave-one-out influence ──────────────────────────────────────
        influence_map: dict[str, float] = {did: 0.0 for did in doc_ids}
        if self._generator:
            try:
                inf_report = compute_influence(
                    query, docs, doc_ids,
                    self._generator, self._embedder,
                    gamma=self.config.influence_gamma,
                )
                influence_map = {r.doc_id: r.influence_score for r in inf_report.scores}
            except Exception as exc:
                logger.warning(f"Influence scoring failed: {exc}")

        # ── ProGRank instability (optional, expensive) ───────────────────
        pgr_map: dict[str, float] = {did: 0.0 for did in doc_ids}
        if self.config.enable_progrank and self.collection:
            try:
                def _retriever(q: str) -> list[tuple[str, float]]:
                    r = self.collection.query(query_texts=[q], n_results=k * 2)
                    results = []
                    for rid, rdist in zip(r["ids"][0], r.get("distances", [[]])[0] or []):
                        results.append((rid, 1.0 / (1.0 + rdist) if rdist else 1.0))
                    return results

                pgr_report = compute_instability(
                    query, _retriever,
                    n_perturbations=self.config.progrank_perturbations,
                    top_k=k * 2,
                )
                pgr_map = {r.doc_id: r.pgr_score for r in pgr_report.results}
            except Exception as exc:
                logger.warning(f"ProGRank instability failed: {exc}")

        # ── Copy ratio ───────────────────────────────────────────────────
        copy_map: dict[str, float] = {}
        for i, (doc, did) in enumerate(zip(docs, doc_ids)):
            other_docs = docs[:i] + docs[i+1:]
            copy_map[did] = compute_copy_ratio(doc, query, other_docs)

        # ── Build signal vectors ─────────────────────────────────────────
        # Normalize fragility (raw range ~1-15) to [0,1] via sigmoid centered at 5.
        # Peakiness < 5 is normal; > 8 is suspicious.
        def _norm_frag(raw: float) -> float:
            return 1.0 / (1.0 + np.exp(-(raw - 5.0) / 2.0))

        for did in doc_ids:
            signals.append(SignalVector(
                doc_id=did,
                pgr=pgr_map.get(did, 0.0),
                mask_fragility=_norm_frag(frag_map.get(did, 0.0)),
                influence=influence_map.get(did, 0.0),
                copy_ratio=copy_map.get(did, 0.0),
                authority=auth_map.get(did, 0.5),
                tamper=tamper_flags.get(did, 0.0),
                original_score=relevance_scores.get(did, 1.0),
            ))

        # ── Composite scoring + reranking ────────────────────────────────
        report = self._poison_scorer.score(signals)
        reranked = report.reranked()

        # Audit the retrieval defense results
        for sd in reranked:
            sv = sd.signals
            self.auditor.log_retrieval(
                verdict=sd.verdict,
                confidence=sd.poison_score,
                reason=(
                    f"RetrievalDefense: poison={sd.poison_score:.3f} "
                    f"pgr={sv.pgr:.2f} mask={sv.mask_fragility:.2f} "
                    f"infl={sv.influence:.2f} copy={sv.copy_ratio:.2f} "
                    f"auth={sv.authority:.2f} tamper={sv.tamper:.0f}"
                ),
                chunk_id=sd.doc_id,
                chunk_text="",
                collection=getattr(self.collection, "name", "unknown"),
                session_id=session_id,
                metadata={"layer": "RetrievalDefense"},
            )

        return reranked

    def _quarantine_chunk(self, text: str, chunk_id: str, result: ShieldResult) -> None:
        import json
        from datetime import datetime, timezone
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chunk_id": chunk_id,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "reason": result.reason,
            "layer_triggered": result.layer_triggered,
            "text_preview": text[:200],
        }
        with self.quarantine.open("a") as f:
            f.write(json.dumps(record) + "\n")
