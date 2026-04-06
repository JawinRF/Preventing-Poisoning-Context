"""
screen_context.py
-----------------
Defines the ScreenContext dataclass — the ground-truth window state
produced by the Android WindowManager + AccessibilityService layer
and consumed by A-MemGuard for deterministic screen type detection.

This module is pure Python — no Android dependencies.
The Android side serializes to JSON and writes to a Unix domain socket.
The Python pipeline reads and deserializes it here.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class ScreenType(Enum):
    """
    Deterministic screen type classification.
    Derived from WindowManager window type + AccessibilityService node analysis.
    NOT keyword-based — derived from structural signals only.
    """
    UNKNOWN          = auto()
    HOME_LAUNCHER    = auto()
    SYSTEM_DIALOG    = auto()   # AlertDialog, PermissionDialog
    SYSTEM_OVERLAY   = auto()   # TYPE_APPLICATION_OVERLAY windows
    KEYBOARD         = auto()   # IME window
    NOTIFICATION     = auto()   # Notification shade
    BROWSER          = auto()   # WebView-rooted hierarchy
    MESSAGING        = auto()   # Scrollable conversation list + EditText
    ECOMMERCE        = auto()   # Price patterns + RecyclerView
    SETTINGS         = auto()   # com.android.settings or AOSP settings package
    MEDIA             = auto()  # VideoView or SurfaceView dominant
    FORM_INPUT       = auto()   # Dominant EditText nodes, labels
    DOCUMENT_VIEWER  = auto()   # ScrollView + dense TextView hierarchy
    GENERIC_APP      = auto()   # Foreground app, uncategorized


@dataclass
class VisibleNode:
    """A single accessibility node that is confirmed visible-to-user."""
    resource_id: str
    class_name: str        # e.g. "android.widget.TextView"
    text: str
    content_desc: str
    bounds_px: tuple[int, int, int, int]   # left, top, right, bottom


@dataclass
class ScreenContext:
    """
    Ground-truth window state at the moment of agent inference.
    Produced by Android service, consumed by A-MemGuard.
    """
    # Identity
    foreground_package: str          # e.g. "com.android.chrome"
    foreground_activity: str         # e.g. "org.chromium.chrome.browser.ChromeTabbedActivity"
    window_type: int                 # Android WindowManager.LayoutParams.type value

    # Geometry
    screen_width_px: int
    screen_height_px: int

    # Semantic tree (only visible-to-user=true nodes)
    visible_nodes: list[VisibleNode] = field(default_factory=list)

    # Derived — filled in by ScreenTypeClassifier, not Android layer
    screen_type: ScreenType = ScreenType.UNKNOWN

    # Visible text corpus — concatenation of all visible node texts
    # Used for consistency checks against payload content
    visible_text_corpus: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ScreenContext":
        nodes = [VisibleNode(**n) for n in d.get("visible_nodes", [])]
        ctx = cls(
            foreground_package=d["foreground_package"],
            foreground_activity=d["foreground_activity"],
            window_type=d.get("window_type", 1),
            screen_width_px=d.get("screen_width_px", 1080),
            screen_height_px=d.get("screen_height_px", 2400),
            visible_nodes=nodes,
        )
        ctx.visible_text_corpus = " ".join(
            n.text for n in nodes if n.text.strip()
        )
        return ctx

    def to_dict(self) -> dict:
        return {
            "foreground_package": self.foreground_package,
            "foreground_activity": self.foreground_activity,
            "window_type": self.window_type,
            "screen_width_px": self.screen_width_px,
            "screen_height_px": self.screen_height_px,
            "visible_nodes": [
                {
                    "resource_id": n.resource_id,
                    "class_name": n.class_name,
                    "text": n.text,
                    "content_desc": n.content_desc,
                    "bounds_px": n.bounds_px,
                }
                for n in self.visible_nodes
            ],
        }


# Sentinel: used when no WindowManager data is available (e.g. unit tests)
NULL_CONTEXT = ScreenContext(
    foreground_package="unknown",
    foreground_activity="unknown",
    window_type=1,
    screen_width_px=1080,
    screen_height_px=2400,
    screen_type=ScreenType.UNKNOWN,
)
