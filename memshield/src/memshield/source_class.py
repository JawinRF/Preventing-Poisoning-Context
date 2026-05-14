"""
source_class.py — Provenance taxonomy (Biba integrity lattice + capabilities).

Every byte that reaches the agent carries a deterministic source class drawn
from a fixed enum. Classes form an integrity lattice under the ordering

    T0_USER_TYPED  ⊐  T0_DEVICE_OWNED  ⊐  T1_SIGNED_TRUSTED
                   ⊐  T2_UNSIGNED_KNOWN ⊐  T3_UNTRUSTED

with T_SYNTHETIC absorbing the GLB (greatest lower bound) of its parents —
synthetic memories cannot rise above the least-trusted source that produced
them. This is exactly the Biba (1977) integrity model.

Each class is paired with a capability bitmask: which side-effect action
categories the class is permitted to *contribute to authorizing*. The
policy gate (policy_gate.py) checks that every chunk supporting a decision
carries a cap_set that covers the action's required capabilities.

NO scoring. NO learned thresholds. Classification is rules-based on
ingestion metadata; misclassification defaults conservative (T3_UNTRUSTED).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from typing import Any


# ── Capability bitmask ────────────────────────────────────────────────────────

class Capability(IntFlag):
    """Side-effect categories a memory chunk can authorize.

    A chunk's cap_set is a subset of these flags. An action requires a
    specific capability mask; the policy gate ANDs the action's mask with
    each supporting chunk's cap_set and rejects if any required bit is
    missing on any supporting chunk.

    NEW capabilities go HERE — and into the cap_set table below — never
    silently inferred elsewhere.
    """
    # Read-only / no external effect
    ANSWER         = 1 << 0   # respond to user's question
    SUMMARIZE      = 1 << 1   # produce summary into transient context
    STORE_INFO     = 1 << 2   # add to long-term memory (own-side only)

    # Low-stakes local effect
    LOCAL_UI       = 1 << 3   # toggle setting, set alarm, change wallpaper
    READ_SENSOR    = 1 << 4   # read additional sensor / system value

    # External effect (network / messaging / file)
    SEND_MESSAGE   = 1 << 5   # SMS, email, push, share
    OPEN_URL_AUTH  = 1 << 6   # navigate to URL with credentials/cookies
    FILE_WRITE_EXT = 1 << 7   # write to /sdcard/ or shared storage
    NETWORK_REQ    = 1 << 8   # HTTP POST/PUT/DELETE to non-allowlisted endpoint

    # Privileged
    INSTALL_APP    = 1 << 9   # install / update an app
    SYSTEM_SETTING = 1 << 10  # change system-wide setting
    PAYMENT        = 1 << 11  # initiate a payment / billing intent


# Useful aggregates
_ALL_CAPS = Capability(0)
for _c in Capability:
    _ALL_CAPS |= _c

_READ_ONLY = (
    Capability.ANSWER
    | Capability.SUMMARIZE
)

_LOW_RISK = (
    _READ_ONLY
    | Capability.STORE_INFO
    | Capability.LOCAL_UI
    | Capability.READ_SENSOR
)


# ── Source class lattice ──────────────────────────────────────────────────────

class SourceClass(IntEnum):
    """Provenance class. Higher integer value = higher trust.

    The IntEnum ordering IS the lattice ordering: class1 ⊐ class2 iff
    class1.value > class2.value. Comparisons are direct: cls >= cls.

    T_SYNTHETIC is a special bottom-merge state for model-generated content;
    its effective trust is the GLB of its parent chunks (see glb()).
    """
    T3_UNTRUSTED      = 10   # default — random web, unknown SMS, scraped page, clipboard
    T2_UNSIGNED_KNOWN = 20   # source the agent has interacted with before, not flagged
    T1_SIGNED_TRUSTED = 30   # allowlisted HTTPS domain / user-launched signed app
    T0_DEVICE_OWNED   = 40   # system services not driven by external input (clock, battery, GPS)
    T0_USER_TYPED     = 50   # direct keyboard input into the agent's own UI

    # Synthetic gets its own marker so audit logs can distinguish it from
    # the T3 sources it may have inherited from. For lattice operations it
    # is treated as the GLB of its parents.
    T_SYNTHETIC       = 5

    @property
    def label(self) -> str:
        return self.name


# ── Capability table ──────────────────────────────────────────────────────────

# CRITICAL: this table is the *security policy*. Every cell must be reviewed
# whenever a new Capability is added. We keep it explicit (no derivation)
# so that grep'ing for a capability name finds every class that grants it.
SOURCE_CAPS: dict[SourceClass, Capability] = {
    # Direct user input gets everything.
    SourceClass.T0_USER_TYPED: _ALL_CAPS,

    # Device-owned values can drive low-risk actions but never side-effects;
    # they're scalar facts, not commands. (Bug guard: a battery reading
    # cannot authorize sending an SMS.)
    SourceClass.T0_DEVICE_OWNED: _LOW_RISK,

    # Allowlisted trusted sources get everything *except* installs / system
    # settings / payment — those require user_typed regardless.
    SourceClass.T1_SIGNED_TRUSTED: (
        _LOW_RISK
        | Capability.SEND_MESSAGE
        | Capability.OPEN_URL_AUTH
        | Capability.FILE_WRITE_EXT
        | Capability.NETWORK_REQ
    ),

    # Known unsigned sources: store + summarize + answer. No external effect.
    SourceClass.T2_UNSIGNED_KNOWN: _READ_ONLY | Capability.STORE_INFO,

    # Untrusted: read-only only. Cannot authorize anything that touches the
    # outside world. This is the central security property.
    SourceClass.T3_UNTRUSTED: _READ_ONLY,

    # Synthetic content gets the answer/summarize floor by default; the
    # *actual* cap_set is the GLB of its parents (computed in glb()).
    SourceClass.T_SYNTHETIC: _READ_ONLY,
}


def cap_set_for(cls: SourceClass) -> Capability:
    """Return the capability bitmask for a source class."""
    return SOURCE_CAPS[cls]


def lattice_glb(classes: list[SourceClass]) -> SourceClass:
    """Greatest lower bound under the lattice ordering.

    Used for synthetic memory lineage: a reflection drawing on chunks
    {c1, c2, c3} carries class = min(class(c1), class(c2), class(c3)),
    so cannot rise above its least-trusted parent.

    Empty input → T3_UNTRUSTED (conservative).
    """
    if not classes:
        return SourceClass.T3_UNTRUSTED
    # IntEnum ordering is integrity-trust order; min() picks lowest trust.
    # Exclude the special T_SYNTHETIC marker — it should never appear as a
    # *parent* class in a healthy graph, but if it does, treat as T3.
    effective = [c if c != SourceClass.T_SYNTHETIC else SourceClass.T3_UNTRUSTED for c in classes]
    return min(effective)


def cap_intersection(cap_sets: list[Capability]) -> Capability:
    """Intersection of multiple cap_sets (lineage propagation).

    Empty input → 0 (no capabilities; conservative).
    """
    if not cap_sets:
        return Capability(0)
    out = cap_sets[0]
    for c in cap_sets[1:]:
        out &= c
    return out


# ── Classifier ────────────────────────────────────────────────────────────────

# Sources whose identifier is recognised even if no metadata flag set.
# These are conservative — we'd rather under-trust than over-trust.
_DEVICE_OWNED_IDS = frozenset({
    "system_clock", "battery_level", "gps_coords", "wifi_state",
    "device_locale", "screen_brightness",
})

# Known internal ingestion paths from the existing PRISM sidecar.
_PATH_TO_CLASS: dict[str, SourceClass] = {
    "user_typed":            SourceClass.T0_USER_TYPED,
    "device_state":          SourceClass.T0_DEVICE_OWNED,
    "system_service":        SourceClass.T0_DEVICE_OWNED,

    # Per Android sidecar /v1/context paths
    "ui_accessibility":      SourceClass.T3_UNTRUSTED,  # arbitrary foreground app
    "notification_context":  SourceClass.T3_UNTRUSTED,
    "sms_context":           SourceClass.T3_UNTRUSTED,
    "contacts_context":      SourceClass.T3_UNTRUSTED,
    "calendar_context":      SourceClass.T3_UNTRUSTED,
    "clipboard_context":     SourceClass.T3_UNTRUSTED,
    "rag_retrieval":         SourceClass.T2_UNSIGNED_KNOWN,
    "experience":            SourceClass.T_SYNTHETIC,
    "partial_experience":    SourceClass.T_SYNTHETIC,
}


@dataclass
class IngestionContext:
    """Metadata available at ingestion time.

    Pass the richest info you have. Missing fields → conservative default.
    """
    ingestion_path: str           # e.g. "sms_context", "ui_accessibility", "user_typed"
    source_id: str                # canonical identifier (phone number, URL, package, "self")
    user_typed: bool = False      # True iff the user typed this into PRISM's own UI
    is_device_owned: bool = False # True iff value comes from a non-injectable system service
    trusted_domain: bool = False  # True iff URL/app is in user's allowlist (HTTPS + valid cert)
    app_signed_trusted: bool = False  # signed by an OEM/vendor in trust-anchor set
    user_launched_app: bool = False   # True iff user opened this app, not just received a push
    seen_before: bool = False     # already in MemShield's recall index
    is_synthetic: bool = False    # produced by an LLM call
    synthetic_parents: list[SourceClass] | None = None  # parent classes if synthetic


def classify(ctx: IngestionContext) -> SourceClass:
    """Deterministic source-class assignment from ingestion metadata.

    Order matters: rules are tried top-down; first match wins. Conservative
    by default — when in doubt, T3_UNTRUSTED.
    """
    # Synthetic content: GLB of parents, regardless of other flags.
    if ctx.is_synthetic:
        if ctx.synthetic_parents:
            return lattice_glb(ctx.synthetic_parents)
        return SourceClass.T3_UNTRUSTED

    # User-typed beats everything.
    if ctx.user_typed or ctx.ingestion_path == "user_typed":
        return SourceClass.T0_USER_TYPED

    # Device-owned values from un-injectable system services.
    if ctx.is_device_owned or ctx.source_id in _DEVICE_OWNED_IDS:
        return SourceClass.T0_DEVICE_OWNED

    # Allowlisted + user-launched + signed → T1.
    if ctx.trusted_domain or (ctx.app_signed_trusted and ctx.user_launched_app):
        return SourceClass.T1_SIGNED_TRUSTED

    # Path-based classification takes priority over seen_before for channels
    # that are always untrusted regardless of history (SMS, notifications,
    # contacts, calendar, clipboard, UI). An attacker who plants the same
    # content repeatedly would otherwise raise their own trust level.
    path_class = _PATH_TO_CLASS.get(ctx.ingestion_path)
    if path_class == SourceClass.T3_UNTRUSTED:
        return SourceClass.T3_UNTRUSTED

    # Known source we've interacted with before — only reachable for paths
    # not locked to T3 above (e.g. rag_retrieval, or unknown paths).
    if ctx.seen_before:
        return SourceClass.T2_UNSIGNED_KNOWN

    # Remaining path-based class (T2, T_SYNTHETIC, etc.).
    if path_class is not None:
        return path_class

    # Default: untrusted.
    return SourceClass.T3_UNTRUSTED


# ── Convenience: human-readable summary ───────────────────────────────────────

def describe_caps(caps: Capability) -> list[str]:
    """Return the list of capability names set in the bitmask."""
    return [c.name for c in Capability if c & caps]


def describe_class(cls: SourceClass) -> dict[str, Any]:
    """JSON-friendly description of a source class for audit logs."""
    return {
        "name": cls.name,
        "lattice_rank": int(cls),
        "cap_set": describe_caps(cap_set_for(cls)),
    }


__all__ = [
    "Capability",
    "SourceClass",
    "SOURCE_CAPS",
    "cap_set_for",
    "lattice_glb",
    "cap_intersection",
    "IngestionContext",
    "classify",
    "describe_caps",
    "describe_class",
]
