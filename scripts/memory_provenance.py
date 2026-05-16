"""
memory_provenance.py — Stage-1 birth-prior signals for autonomous memory learning.

Called from _record_experience (agent_prism.py) after every task completion.
Determines the birth trust of auto-saved memories WITHOUT any LLM call.

Stage-1 (always-on, cheap):
    1. T3 presence  — any T3 source in context → PRIOR_T3
    2. Causal overlap — memory's novel span semantically matches a T3 source text
                        → PRIOR_FLAGGED (Stage-2 trigger)
    3. Consensus contradiction — memory contradicts the trusted-memory centroid
                                  for its intent bucket → flag (future extension)

Stage-2 (gated, rare): leave-one-T3-out counterfactual re-derivation.
    Fires only when Stage-1 trips. Stub here — extend when planner call overhead
    is acceptable on-device.

Trust scores never enter the agent prompt; they live in ChromaDB metadata only.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Stage-1 thresholds (tunable)
CAUSAL_OVERLAP_THRESHOLD = 0.25   # novel-word Jaccard vs T3 text → flag
_MIN_NOVEL_WORDS         = 3      # skip overlap check if too few novel words


def compute_birth_prior(
    memory_text: str,
    task: str,
    t3_texts: list[str],
) -> bool:
    """Return True if Stage-1 causal-overlap signal trips (→ PRIOR_FLAGGED).

    Algorithm:
      1. Tokenise memory_text and task into word sets.
      2. Novel span = words in memory but NOT in task (these weren't in the
         original goal — they had to come from somewhere else in context).
      3. For each T3 source text, compute Jaccard(novel_span, t3_words).
      4. If any T3 source overlaps the novel span above threshold → True.

    Returns False (no flag) when there are no T3 sources in context or when
    novel span overlap is below threshold (i.e. auto-memory was derived from
    the task + UI, not from injected content).
    """
    if not t3_texts:
        return False

    mem_words  = _tokenise(memory_text)
    task_words = _tokenise(task)
    novel      = mem_words - task_words

    if len(novel) < _MIN_NOVEL_WORDS:
        return False

    for t3_text in t3_texts:
        t3_words = _tokenise(t3_text)
        if not t3_words:
            continue
        overlap = len(novel & t3_words) / max(len(novel), len(t3_words))
        if overlap >= CAUSAL_OVERLAP_THRESHOLD:
            logger.warning(
                f"[Provenance] Stage-1 causal overlap {overlap:.2f} >= "
                f"{CAUSAL_OVERLAP_THRESHOLD} — birth prior → PRIOR_FLAGGED. "
                f"T3 text: {t3_text[:60]!r}"
            )
            return True

    return False


def get_causal_t3_fps(
    memory_text: str,
    task: str,
    t3_fingerprints: list[str],
    t3_texts: list[str],
) -> list[str]:
    """Return T3 fingerprints whose text has high causal overlap with memory_text.

    Used by Stage-2 auto-tombstone to identify WHICH T3 sources authored the
    drift before flagging them via flag_t3_source().
    """
    if not t3_fingerprints:
        return []

    mem_words  = _tokenise(memory_text)
    task_words = _tokenise(task)
    novel      = mem_words - task_words

    if len(novel) < _MIN_NOVEL_WORDS:
        return []

    flagging = []
    for fp, text in zip(t3_fingerprints, t3_texts):
        t3_words = _tokenise(text)
        if not t3_words:
            continue
        overlap = len(novel & t3_words) / max(len(novel), len(t3_words))
        if overlap >= CAUSAL_OVERLAP_THRESHOLD:
            logger.warning(
                f"[Provenance] T3 source {fp} causal overlap {overlap:.2f} — "
                f"flagging as drift author"
            )
            flagging.append(fp)
    return flagging


def run_stage2_counterfactual(
    memory_text: str,
    task: str,
    t3_fingerprints: list[str],
    t3_texts: list[str],
    planner_fn,          # callable: (task, context_without_t3) -> decision_str
    flag_fn,             # callable: (fingerprint) -> None
) -> bool:
    """Stage-2: leave-one-T3-out counterfactual re-derivation.

    For each in-context T3 source, calls planner_fn with that source ablated.
    If ablating the source flips the behavior the memory encodes, that source
    authored the drift: flag_fn is called on its fingerprint.

    Returns True if any source was identified and flagged.

    NOTE: This is the only mechanism that catches on-manifold MINJA-class drift
    (subtle behavioral drift that passes TinyBERT/DeBERTa). Cost = one planner
    call per T3 source. Gated by Stage-1 — only fires when Stage-1 trips.
    """
    if not t3_fingerprints:
        return False

    flagged_any = False
    for fp, t3_text in zip(t3_fingerprints, t3_texts):
        try:
            ablated_decision = planner_fn(task, context_without=t3_text)
            if _behavior_diverges(memory_text, ablated_decision):
                logger.warning(
                    f"[Provenance] Stage-2 flip on ablating {fp} — "
                    f"source authored drift. Flagging."
                )
                flag_fn(fp)
                flagged_any = True
        except Exception as exc:
            logger.warning(f"[Provenance] Stage-2 counterfactual failed for {fp}: {exc}")

    return flagged_any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lower-case word tokens, stripping punctuation. Excludes short stop-words."""
    _STOPWORDS = {"the", "a", "an", "to", "in", "on", "of", "for",
                  "and", "or", "is", "it", "i", "my", "me", "at", "by"}
    words = set(re.findall(r"[a-z]+", text.lower()))
    return words - _STOPWORDS


def _behavior_diverges(memory_text: str, ablated_decision: str) -> bool:
    """Rough semantic divergence check: significant non-overlap between
    what the memory encodes and the ablated decision.

    A high-quality implementation would use the embedding cosine distance.
    This version uses word-level Jaccard as a cheap proxy.
    """
    mem_words = _tokenise(memory_text)
    dec_words = _tokenise(ablated_decision)
    if not mem_words or not dec_words:
        return False
    jaccard = len(mem_words & dec_words) / len(mem_words | dec_words)
    # Low overlap between memory and the ablated decision → memory wasn't
    # derived from the task alone; the ablated T3 source caused the drift.
    return jaccard < 0.15
