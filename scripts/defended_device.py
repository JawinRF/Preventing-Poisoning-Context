"""
defended_device.py — Wrapper around uiautomator2 device that enforces PRISM checks.

Agents use DefendedDevice instead of raw device + manual PRISM checks.
This prevents defense logic duplication and ensures no action can bypass PRISM.

Visual grounding (VLM-based XML-spoofing defense) runs synchronously before taps.
Slow (~12s on CPU) but security-correct: blocks XML-spoofing before the tap lands.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism_client import PrismClient

logger = logging.getLogger(__name__)

# Try to import visual grounding module
try:
    from prism_shield.visual_grounding import visual_grounder, VisualGroundingResult
    _VISUAL_GROUNDING_AVAILABLE = True
except ImportError:
    _VISUAL_GROUNDING_AVAILABLE = False
    VisualGroundingResult = None


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
        self._last_screenshot_path: str | None = None
        
        # Initialize visual grounding VLM
        if _VISUAL_GROUNDING_AVAILABLE:
            visual_grounder.initialize()
            logger.info("Visual grounding VLM initialized in DefendedDevice")
        else:
            logger.warning("Visual grounding not available - XML spoofing attacks possible")

    @property
    def device(self):
        """Access the raw device for non-action calls (window_size, screen_on, etc.)."""
        return self._d

    # ── Visual Grounding (XML-spoofing defense — synchronous blocking) ──────────

    def _capture_screenshot(self) -> str | None:
        """Capture current screen for visual grounding verification."""
        try:
            scripts_dir = Path(__file__).resolve().parent
            temp_dir = scripts_dir.parent / "data" / "screenshots"
            temp_dir.mkdir(parents=True, exist_ok=True)

            screenshot_path = temp_dir / f"vg_screen_{uuid.uuid4().hex[:8]}.png"
            self._d.screenshot(str(screenshot_path))
            self._last_screenshot_path = str(screenshot_path)
            return str(screenshot_path)
        except Exception as e:
            logger.warning(f"Screenshot capture for visual grounding failed: {e}")
            return None

    def _verify_visual_grounding(self, target_text: str | None = None,
                                  target_desc: str | None = None) -> bool:
        """Verify the target element exists in actual screen pixels.

        Synchronous and blocking — slow (~12s on CPU) but security-correct.
        Blocks XML-spoofing attacks where malicious apps fake accessibility tree.

        Returns True if verified or VLM unavailable, False if spoofing detected.
        """
        if not _VISUAL_GROUNDING_AVAILABLE:
            logger.debug("Visual grounding unavailable (module not installed) — allowing tap")
            return True

        if not visual_grounder._initialized or not visual_grounder.llm:
            # Model failed to load or hasn't finished init — don't brick the agent
            logger.warning("Visual grounding VLM not initialized — allowing tap")
            return True

        screenshot_path = self._capture_screenshot()
        if not screenshot_path:
            logger.warning("Cannot capture screenshot for visual grounding — allowing tap")
            return True

        result = visual_grounder.verify_element(
            screenshot_path=screenshot_path,
            target_text=target_text,
            target_desc=target_desc,
        )

        MIN_CONFIDENCE = 0.7
        if result.confidence < MIN_CONFIDENCE:
            logger.warning(
                f"VISUAL GROUNDING LOW CONFIDENCE: {result.confidence:.2f} < {MIN_CONFIDENCE} "
                f"for '{target_text or target_desc}' — blocking tap"
            )
            return False

        if not result.verified:
            logger.warning(
                f"VISUAL GROUNDING BLOCKED: '{target_text or target_desc}' "
                f"not found in screen pixels — possible XML spoofing! "
                f"Reason: {result.reason}"
            )
            return False

        logger.info(f"Visual grounding verified: {result.reason} (confidence: {result.confidence:.2f})")
        return True
    
    # ── PRISM defense layer ──────────────────────────────────────────────────

    def _resolve_verdict(self, r) -> str | None:
        """
        Handle ALLOW / BLOCK / QUARANTINE verdicts.
        For QUARANTINE, polls the sidecar for VLM resolution.
        Returns "blocked_by_prism" if blocked, None if allowed.
        """
        if r.allowed:
            return None
        if r.verdict == "QUARANTINE" and r.ticket_id:
            logger.info(f"QUARANTINE verdict (ticket={r.ticket_id}) — polling for VLM resolution...")
            resolved = self._prism.poll_quarantine(r.ticket_id)
            if resolved.allowed:
                logger.info(f"Quarantine lifted: {resolved.reason}")
                return None
            logger.warning(f"Quarantine confirmed: {resolved.reason}")
        return "blocked_by_prism"

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
                result = self._resolve_verdict(r)
                if result:
                    return result

        elif action == "type":
            text_data = params.get("text", "")
            if text_data:
                if DANGEROUS_TYPE_PATTERNS.search(text_data):
                    logger.warning(f"BLOCKED typed text (dangerous pattern): {text_data[:60]}")
                    return "blocked_by_prism"
                r = self._prism.inspect(text_data, "clipboard", "text_input")
                result = self._resolve_verdict(r)
                if result:
                    return result

        elif action == "open_app":
            pkg = params.get("package", "")
            if pkg and pkg not in ALLOWED_PACKAGES:
                r = self._prism.inspect(f"open:{pkg}", "android_intents", "app_launch")
                result = self._resolve_verdict(r)
                if result:
                    return result

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
                target_text = params.get("text")
                target_desc = params.get("desc")

                # Synchronous VLM verification — blocks XML spoofing attacks
                if target_text or target_desc:
                    if not self._verify_visual_grounding(target_text, target_desc):
                        return "blocked_by_visual_grounding"

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
