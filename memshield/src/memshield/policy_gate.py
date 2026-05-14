"""
policy_gate.py — Deterministic policy interpreter for the PROVE architecture.

The single enforcement primitive. Called between the planner LLM's action
decision and the action's execution. Every blocking decision is a rule
check on inputs whose authenticity has been cryptographically verified.

NO scoring. NO machine learning. NO thresholds beyond integer counts.

Inputs the gate sees (all sealed at ingestion via ProvenanceSeal):
  - action verb + concrete fact (recipient, URL, payload)
  - retrieved memory chunks supporting that fact
    (each carrying source_class, source_id, prov_seal, chunk text)

Checks, in order:
  1. HMAC seal verification on every chunk          (provenance integrity)
  2. Action → policy lookup                         (well-known action)
  3. T0_USER_TYPED short-circuit                    (user is trust root)
  4. Quorum: ≥ k supporting chunks                  (RobustRAG-style)
  5. Source-class diversity ≥ k                     (cross-channel agreement)
  6. Source-id diversity ≥ k                        (cross-instance agreement)
  7. Capability containment per supporting chunk    (Biba write-up rule)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

from .source_class import (
    Capability,
    SourceClass,
    cap_set_for,
)
from .policy_tables import policy_for, QUORUM_K
from .provenance import ProvenanceSeal

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

class Decision(Enum):
    ALLOW    = "ALLOW"     # gate authorizes execution
    BLOCK    = "BLOCK"     # gate rejects; do not execute
    ESCALATE = "ESCALATE"  # gate cannot decide alone; ask the user


@dataclass
class SupportingChunk:
    """A memory chunk + its supporting evidence for a specific fact.

    The caller (typically the agent step loop) is responsible for asserting
    that this chunk supports the fact in question — usually by way of the
    Q-LLM extractor having returned the fact's value from this chunk's text.
    """
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    extracted_value: Any | None = None  # what the Q-LLM extracted, if any


@dataclass
class GateInput:
    """Everything the gate needs to make one authorisation decision."""
    action: str                          # raw verb the planner emitted, e.g. "send_sms"
    fact_key: str                        # which field is contested, e.g. "recipient"
    fact_value: Any                      # the value (e.g. "+919876543210")
    supporting: list[SupportingChunk]    # chunks that allegedly support the fact

    # Optional: hint that the user explicitly typed the fact in their query.
    # If true and the fact_value matches typed_value, the gate treats this
    # as if a T0_USER_TYPED chunk were among the supporters. The agent
    # step loop is responsible for setting these honestly.
    user_typed_value: Any | None = None


@dataclass
class GateDecision:
    """Result of one gate call. ALL fields go to the audit log."""
    decision: Decision
    reason: str
    action: str
    action_kind: str
    risk: str
    required_caps: list[str]
    required_quorum: int
    user_waiver: bool                    # T0_USER_TYPED short-circuit fired
    supporting_count: int
    distinct_classes: list[str]
    distinct_ids: list[str]
    failed_seals: list[str] = field(default_factory=list)
    cap_failures: list[str] = field(default_factory=list)  # chunks present but ineligible (not a block; logged for audit)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_source_class(meta: dict[str, Any]) -> SourceClass | None:
    """Read source_class from chunk metadata.

    Returns None if absent or unknown. The gate treats None as T3_UNTRUSTED
    (conservative): unsealed legacy chunks get the lowest caps.
    """
    raw = meta.get(ProvenanceSeal.SOURCE_CLASS_KEY)
    if raw is None:
        return None
    try:
        return SourceClass[raw] if isinstance(raw, str) else SourceClass(raw)
    except (KeyError, ValueError):
        return None


def _chunk_source_id(meta: dict[str, Any]) -> str:
    """Read source_id from chunk metadata; "unknown" if absent."""
    sid = meta.get(ProvenanceSeal.SOURCE_ID_KEY)
    return sid if isinstance(sid, str) and sid else "unknown"


# ── The gate ──────────────────────────────────────────────────────────────────

def authorize(gi: GateInput, *, hmac_key: bytes | None = None) -> GateDecision:
    """Authorise (or refuse) one concrete action under the PROVE policy.

    The decision is fully determined by the inputs — no randomness, no
    learned weights, no state. The same inputs always produce the same
    decision.

    Failure modes:

      - BLOCK("provenance_tamper:<chunk_id>"):
          A supporting chunk's HMAC didn't verify. Treated as if the chunk
          doesn't exist; the supporting set is reduced by one.
          If the surviving set still satisfies the quorum, we ALLOW with
          a warning. If not, we BLOCK.

      - BLOCK("insufficient_quorum"):
          Fewer than k supporting chunks (after dropping failed seals).

      - BLOCK("class_diversity"):
          Supporting chunks span < k distinct source classes.

      - BLOCK("id_diversity"):
          Supporting chunks span < k distinct source identifiers.

      - BLOCK("capability:<chunk_id>"):
          A supporting chunk's class does not grant the required caps.

      - ALLOW("user_typed_waiver"):
          User typed the contested value in the query; T0_USER_TYPED root
          authorises directly.

      - ALLOW("quorum_met"):
          All checks passed.

      - ESCALATE("user_confirm_required"):
          Gate would block, but the action's policy flags it for user
          confirmation. Caller surfaces a dialogue; user becomes the new
          T0_USER_TYPED root if they confirm.
    """
    policy = policy_for(gi.action)
    k = QUORUM_K[policy.risk]
    required = policy.required_caps

    # ── Step 1: HMAC seal verification ───────────────────────────────────────
    valid: list[SupportingChunk] = []
    failed_seals: list[str] = []
    for c in gi.supporting:
        if not ProvenanceSeal.is_sealed(c.metadata):
            # Unsealed legacy chunk — participates with conservative class T3.
            valid.append(c)
            continue
        ok, reason = ProvenanceSeal.verify(c.text, c.metadata, key=hmac_key)
        if not ok:
            failed_seals.append(f"{c.chunk_id}:{reason}")
            logger.warning(
                "Policy gate: dropped chunk %s due to %s", c.chunk_id, reason,
            )
            continue
        valid.append(c)

    # ── Step 1b: Capability filter ───────────────────────────────────────────
    #
    # Cap-insufficient chunks are evidence of a fact but cannot *authorise*
    # the action (Biba write-up rule). Filter them out BEFORE quorum counting
    # rather than blocking on them — an unrelated T3 notification in context
    # should not prevent a well-sourced tap/send_sms from proceeding.
    all_classes = [_chunk_source_class(c.metadata) or SourceClass.T3_UNTRUSTED for c in valid]
    cap_eligible = [
        (c, cls)
        for c, cls in zip(valid, all_classes)
        if (cap_set_for(cls) & required) == required
    ]
    cap_failures = [
        f"{c.chunk_id}:{cls.name}"
        for c, cls in zip(valid, all_classes)
        if (cap_set_for(cls) & required) != required
    ]

    classes = [cls for _, cls in cap_eligible]
    ids     = [_chunk_source_id(c.metadata) for c, _ in cap_eligible]
    distinct_class_names = sorted({cls.name for cls in classes})
    distinct_ids_list    = sorted(set(ids))

    base_decision = GateDecision(
        decision=Decision.BLOCK,
        reason="",
        action=gi.action,
        action_kind=policy.kind,
        risk=policy.risk.name,
        required_caps=[c.name for c in Capability if c & required],
        required_quorum=k,
        user_waiver=False,
        supporting_count=len(cap_eligible),
        distinct_classes=distinct_class_names,
        distinct_ids=distinct_ids_list,
        failed_seals=failed_seals,
        cap_failures=cap_failures,
    )

    # ── Step 2: User-typed waiver ────────────────────────────────────────────
    if gi.user_typed_value is not None and gi.user_typed_value == gi.fact_value:
        contradictions = [
            c.chunk_id for c in valid
            if c.extracted_value is not None and c.extracted_value != gi.fact_value
        ]
        if contradictions:
            return _replace_decision(
                base_decision, Decision.BLOCK,
                f"user_value_contradicted_by:{','.join(contradictions[:3])}",
            )
        return _replace_decision(
            base_decision, Decision.ALLOW, "user_typed_waiver",
            user_waiver=True,
        )

    # ── Step 3: Quorum count (cap-eligible chunks only) ──────────────────────
    if len(cap_eligible) < k:
        return _escalate_or_block(
            base_decision, policy, "insufficient_quorum",
        )

    # ── Step 4: Class diversity ──────────────────────────────────────────────
    if len(set(classes)) < k:
        return _escalate_or_block(
            base_decision, policy, "class_diversity",
        )

    # ── Step 5: Source-id diversity ──────────────────────────────────────────
    if len(set(ids)) < k:
        return _escalate_or_block(
            base_decision, policy, "id_diversity",
        )

    # ── ALLOW ────────────────────────────────────────────────────────────────
    return _replace_decision(base_decision, Decision.ALLOW, "quorum_met")


# ── Decision constructors ─────────────────────────────────────────────────────

def _replace_decision(
    base: GateDecision,
    decision: Decision,
    reason: str,
    user_waiver: bool = False,
) -> GateDecision:
    """Return a copy of base with decision/reason/user_waiver overridden."""
    return GateDecision(
        decision=decision,
        reason=reason,
        action=base.action,
        action_kind=base.action_kind,
        risk=base.risk,
        required_caps=base.required_caps,
        required_quorum=base.required_quorum,
        user_waiver=user_waiver,
        supporting_count=base.supporting_count,
        distinct_classes=base.distinct_classes,
        distinct_ids=base.distinct_ids,
        failed_seals=base.failed_seals,
        cap_failures=base.cap_failures,
    )


def _escalate_or_block(
    base: GateDecision,
    policy,
    reason: str,
) -> GateDecision:
    """Choose ESCALATE if policy.user_confirm_on_block, else BLOCK."""
    out = Decision.ESCALATE if policy.user_confirm_on_block else Decision.BLOCK
    return _replace_decision(base, out, reason)


__all__ = [
    "Decision",
    "SupportingChunk",
    "GateInput",
    "GateDecision",
    "authorize",
]
