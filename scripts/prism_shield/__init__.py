"""
PRISM Shield – ingestion-layer defense pipeline for mobile agents.

This package is intentionally lightweight; most users should interact via
the `PrismShield` class, which evaluates incoming context events from
Android-style ingestion paths (clipboard, intents, notifications, UI
accessibility dumps, etc.).
"""

from .base import MemoryEntry, ValidationResult

__all__ = [
    "MemoryEntry",
    "ValidationResult",
    "PrismShield",
]


def __getattr__(name: str):
    if name == "PrismShield":
        from .pipeline import PrismShield

        return PrismShield
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
