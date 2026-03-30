"""
defended_device.py — Wrapper around uiautomator2 device that enforces PRISM checks.

Agents use DefendedDevice instead of raw device + manual PRISM checks.
This prevents defense logic duplication and ensures no action can bypass PRISM.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_client import PrismClient

logger = logging.getLogger(__name__)

# Allowed packages — anything not on this list gets PRISM-checked
ALLOWED_PACKAGES = {
    "todolist.scheduleplanner.dailyplanner.todo.reminders",
    "com.google.android.deskclock",
    "com.android.chrome",
    "com.google.android.calendar",
    "com.termux",
    "com.android.launcher3",
    "com.android.settings",
}

# Dangerous patterns in outgoing typed text (compiled once at module load)
DANGEROUS_TYPE_PATTERNS = re.compile(
    r"(?i)("
    r"https?://|"
    r"adb\s+shell|"
    r"su\s+-c|"
    r"pm\s+grant|pm\s+install|"
    r"am\s+start.*-d\s+|"
    r"curl\s+|wget\s+|"
    r"rm\s+-rf|"
    r"chmod\s+[0-7]{3}"
    r")"
)


class DefendedDevice:
    """
    Wraps a uiautomator2 device and a PrismClient.
    All actions go through PRISM defense before touching the device.

    Usage:
        dd = DefendedDevice(d, prism, serial)
        result = dd.execute("tap", {"text": "Confirm"})
    """

    def __init__(self, device, prism: PrismClient | None, serial: str,
                 action_settle_time: float = 1.5):
        self._d = device
        self._prism = prism
        self._serial = serial
        self._action_settle_time = action_settle_time

    @property
    def device(self):
        """Access the raw device for non-action calls (window_size, screen_on, etc.)."""
        return self._d

    # ── PRISM defense layer ──────────────────────────────────────────────────

    def _check_prism(self, action: str, params: dict) -> str | None:
        """
        Run PRISM checks on outgoing actions.
        Returns "blocked_by_prism" if blocked, None if allowed.
        """
        if not self._prism:
            return None

        if action == "tap":
            tap_text = params.get("text", "") + params.get("desc", "")
            if tap_text.strip():
                r = self._prism.inspect(tap_text, "ui_accessibility", "tap_action")
                if not r.allowed:
                    return "blocked_by_prism"

        elif action == "type":
            text_data = params.get("text", "")
            if text_data:
                if DANGEROUS_TYPE_PATTERNS.search(text_data):
                    logger.warning(f"BLOCKED typed text (dangerous pattern): {text_data[:60]}")
                    return "blocked_by_prism"
                r = self._prism.inspect(text_data, "clipboard", "text_input")
                if not r.allowed:
                    return "blocked_by_prism"

        elif action == "open_app":
            pkg = params.get("package", "")
            if pkg and pkg not in ALLOWED_PACKAGES:
                r = self._prism.inspect(f"open:{pkg}", "android_intents", "app_launch")
                if not r.allowed:
                    return "blocked_by_prism"

        return None

    # ── Action execution ─────────────────────────────────────────────────────

    def _clear_focused_field(self):
        """Select all text in focused field and delete it."""
        subprocess.run(
            ["adb", "-s", self._serial, "shell", "input", "keyevent", "KEYCODE_MOVE_HOME"],
            timeout=3, capture_output=True,
        )
        subprocess.run(
            ["adb", "-s", self._serial, "shell", "input", "keyevent", "--longpress", "KEYCODE_DEL"],
            timeout=3, capture_output=True,
        )
        time.sleep(0.1)
        subprocess.run(
            ["adb", "-s", self._serial, "shell", "input", "keyevent", "KEYCODE_CTRL_LEFT", "KEYCODE_A"],
            timeout=3, capture_output=True,
        )
        subprocess.run(
            ["adb", "-s", self._serial, "shell", "input", "keyevent", "KEYCODE_DEL"],
            timeout=3, capture_output=True,
        )
        time.sleep(0.1)

    def execute(self, action: str, params: dict) -> str:
        """
        Execute an action on the device with PRISM defense.
        Returns: "ok", "blocked_by_prism", "not found: ...", "error: ...", etc.
        """
        # Defense layer — check before executing
        blocked = self._check_prism(action, params)
        if blocked:
            return blocked

        try:
            if action == "tap":
                if "text" in params:
                    el = self._d(text=params["text"])
                    if el.exists(timeout=3):
                        el.click()
                        return "ok"
                    return f"not found: text={params['text']}"
                if "desc" in params:
                    el = self._d(description=params["desc"])
                    if el.exists(timeout=3):
                        el.click()
                        return "ok"
                    return f"not found: desc={params['desc']}"
                if "class" in params:
                    cls = params["class"]
                    el = self._d(className=f"android.widget.{cls}")
                    if el.exists(timeout=3):
                        el.click()
                        return "ok"
                    return f"not found: class={cls}"

            elif action == "type":
                text = params.get("text", "")
                if text:
                    self._clear_focused_field()
                    escaped = text.replace(" ", "%s")
                    cmd = ["adb", "-s", self._serial, "shell", "input", "text", escaped]
                    subprocess.run(cmd, timeout=5, capture_output=True)
                    time.sleep(0.3)
                return "ok"

            elif action == "clear":
                self._clear_focused_field()
                return "ok"

            elif action == "swipe":
                w, h = self._d.window_size()
                cx, cy = w // 2, h // 2
                dirs = {
                    "up":    (cx, int(h * .7), cx, int(h * .3)),
                    "down":  (cx, int(h * .3), cx, int(h * .7)),
                    "left":  (int(w * .8), cy, int(w * .2), cy),
                    "right": (int(w * .2), cy, int(w * .8), cy),
                }
                self._d.swipe(*dirs.get(params.get("direction", "up"), dirs["up"]), duration=0.4)
                return "ok"

            elif action == "press":
                self._d.press(params.get("key", "back"))
                return "ok"

            elif action == "open_app":
                self._d.app_start(params.get("package", ""))
                time.sleep(2.5)
                return "ok"

            elif action in ("done", "fail"):
                return action

        except Exception as e:
            return f"error: {e}"

        return "unknown"
