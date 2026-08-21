"""
policy_tables.py — Static policy tables for the PROVE policy gate.

These tables ARE the security policy. They are deliberately:
  - small (≤ 50 lines of data total)
  - explicit (no derivation, no inheritance, no scoring)
  - reviewable by hand (every cell is a security decision)

Two tables:

  ACTION_CAPS[action_kind] -> Capability    Which caps the action requires.
  QUORUM_K [risk_level]    -> int           How many distinct sources required.

Two helpers map agent actions to these:

  action_kind(action_name)  -> str          Canonical action class.
  risk_level(action_kind)   -> RiskLevel    Effect class.

The intent is: when a new agent capability is added, BOTH the action_kind
and its (risk, caps) row must be added here. grep'ing for an action name
finds its full policy in one file.
"""
from __future__ import annotations

from enum import IntEnum

from .source_class import Capability


# ── Risk levels ───────────────────────────────────────────────────────────────

class RiskLevel(IntEnum):
    """Severity of an action's external effect.

    Higher value = more severe. Quorum requirement grows with risk.
    """
    R0_READ_ONLY  = 0   # answer / summarize — no state change
    R1_LOCAL      = 1   # local UI / device-internal change, no network
    R2_EXTERNAL   = 2   # network or messaging effect
    R3_PRIVILEGED = 3   # install / system / payment / credentials


# ── Quorum table ──────────────────────────────────────────────────────────────
#
# k(R) = number of supporting chunks required, drawn from k distinct source
# classes AND k distinct source identifiers. Larger k = stronger formal bound.
# RobustRAG (Thm 1) certifies k'=1 of n=10; our gate is strictly stronger
# because of the diversity requirements over class AND id.

QUORUM_K: dict[RiskLevel, int] = {
    RiskLevel.R0_READ_ONLY:  1,
    RiskLevel.R1_LOCAL:      2,
    RiskLevel.R2_EXTERNAL:   3,
    RiskLevel.R3_PRIVILEGED: 4,
}


# ── Action → kind mapping ─────────────────────────────────────────────────────
#
# `action` is the raw verb the agent emitted (e.g. "tap", "type", "open_app").
# `action_kind` is the policy bucket. Many concrete actions map to the same
# kind; the kind determines both required caps and risk.
#
# Unknown actions default to "unknown" (highest risk).

_ACTION_TO_KIND: dict[str, str] = {
    # Read-only / observation
    "done":           "read_only",
    "fail":           "read_only",
    "noop":           "read_only",
    "wait":           "read_only",

    # Local-UI navigation (no external effect)
    "tap":            "local_ui",
    "press":          "local_ui",
    "swipe":          "local_ui",
    "scroll":         "local_ui",
    "open_app":       "local_ui",

    # Local input (becomes external if used in a network field — handled
    # by inspecting the target package + field at the call site).
    "type":           "local_ui_type",
    "clear":          "local_ui",        # clear a text field
    "web_tap":        "local_ui",        # tap inside a webview
    "web_type":       "local_ui_type",   # type inside a webview

    # External / network
    "send_sms":       "send_message",
    "send_email":     "send_message",
    "share":          "send_message",
    "post":           "network_request",
    "external_consent": "network_request",
    "open_url":       "open_url",

    # File system
    "write_file":     "file_write_external",
    "delete_file":    "file_write_external",

    # Privileged
    "install_app":    "install_app",
    "uninstall_app":  "install_app",
    "system_setting": "system_setting",
    "grant_permission": "system_setting",
    "payment":        "payment",
}


def action_kind(action: str) -> str:
    """Map a raw agent action verb to its policy kind."""
    return _ACTION_TO_KIND.get(action, "unknown")


# ── Action-kind policy table ──────────────────────────────────────────────────
#
# Each kind has: required Capability mask, RiskLevel, optional flag for
# "needs user confirmation" (escalation rather than hard block).
#
# CRITICAL invariant: required_caps for a kind must be a SUBSET of the
# cap_set of the SOURCE CLASS that's allowed to authorize it. If you grant
# SEND_MESSAGE here, ensure at least one source class actually has it.

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionPolicy:
    kind: str
    required_caps: Capability
    risk: RiskLevel
    user_confirm_on_block: bool = False   # if True, BLOCK escalates to user


ACTION_POLICY: dict[str, ActionPolicy] = {
    "read_only": ActionPolicy(
        kind="read_only",
        required_caps=Capability.ANSWER,
        risk=RiskLevel.R0_READ_ONLY,
    ),

    "local_ui": ActionPolicy(
        kind="local_ui",
        required_caps=Capability.LOCAL_UI,
        risk=RiskLevel.R1_LOCAL,
    ),

    "local_ui_type": ActionPolicy(
        kind="local_ui_type",
        required_caps=Capability.LOCAL_UI,
        risk=RiskLevel.R1_LOCAL,
    ),

    "send_message": ActionPolicy(
        kind="send_message",
        required_caps=Capability.SEND_MESSAGE,
        risk=RiskLevel.R2_EXTERNAL,
        user_confirm_on_block=True,
    ),

    "network_request": ActionPolicy(
        kind="network_request",
        required_caps=Capability.NETWORK_REQ,
        risk=RiskLevel.R2_EXTERNAL,
        user_confirm_on_block=True,
    ),

    "open_url": ActionPolicy(
        kind="open_url",
        # OPEN_URL_AUTH is the conservative requirement — even non-auth URLs
        # can leak referrer / track. If you need a less strict variant, add
        # a separate "open_url_public" kind.
        required_caps=Capability.OPEN_URL_AUTH,
        risk=RiskLevel.R2_EXTERNAL,
        user_confirm_on_block=True,
    ),

    "file_write_external": ActionPolicy(
        kind="file_write_external",
        required_caps=Capability.FILE_WRITE_EXT,
        risk=RiskLevel.R2_EXTERNAL,
        user_confirm_on_block=True,
    ),

    "install_app": ActionPolicy(
        kind="install_app",
        required_caps=Capability.INSTALL_APP,
        risk=RiskLevel.R3_PRIVILEGED,
        user_confirm_on_block=True,
    ),

    "system_setting": ActionPolicy(
        kind="system_setting",
        required_caps=Capability.SYSTEM_SETTING,
        risk=RiskLevel.R3_PRIVILEGED,
        user_confirm_on_block=True,
    ),

    "payment": ActionPolicy(
        kind="payment",
        required_caps=Capability.PAYMENT,
        risk=RiskLevel.R3_PRIVILEGED,
        user_confirm_on_block=True,
    ),

    # Conservative default: treat as privileged. Forces explicit policy
    # addition for any new action kind.
    "unknown": ActionPolicy(
        kind="unknown",
        required_caps=Capability.PAYMENT | Capability.INSTALL_APP,  # very broad → likely blocked
        risk=RiskLevel.R3_PRIVILEGED,
        user_confirm_on_block=True,
    ),
}


def policy_for(action: str) -> ActionPolicy:
    """Resolve an action verb to its full policy entry."""
    return ACTION_POLICY[action_kind(action)]


def quorum_for_action(action: str) -> int:
    """Number of distinct (class, id) supporting sources required."""
    return QUORUM_K[policy_for(action).risk]


__all__ = [
    "RiskLevel",
    "QUORUM_K",
    "ActionPolicy",
    "ACTION_POLICY",
    "action_kind",
    "policy_for",
    "quorum_for_action",
]
