"""
defended_device.py — Wrapper around uiautomator2 device that enforces PRISM checks.

Agents use DefendedDevice instead of raw device + manual PRISM checks.
This prevents defense logic duplication and ensures no action can bypass PRISM.

Tap integrity uses OS-level checks via the Android sidecar (/v1/ui-integrity):
  - Foreground package verification
  - Overlay / obscuration window detection
  - Target node existence + bounds validity + interactability
  - Dual-snapshot stability (node consistent across two rapid tree captures)

Design rationale (research-backed, replaces prior VLM visual grounding):
  - ANDROIDWORLD: accessibility tree outperforms screenshot-VLM for Android agents
  - TapTrap (USENIX Security 2025): OS-level flags, not vision, stop tapjacking
  - Android guidance: filterTouchesWhenObscured, FLAG_WINDOW_IS_PARTIALLY_OBSCURED
  - SeeClick/ScreenAI: even specialized UI-vision models get ~53% grounding accuracy;
    a tiny general VLM is not a defensible security boundary
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from prism_client import PrismClient

logger = logging.getLogger(__name__)

# Android sidecar endpoint for UI integrity checks
_SIDECAR_UI_INTEGRITY_URL = "http://127.0.0.1:8766/v1/ui-integrity"
_SIDECAR_TIMEOUT_S = 3

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

    # ── UI Integrity (OS-level tap safety — replaces VLM visual grounding) ───

    def _verify_ui_integrity(
        self,
        target_text: str | None = None,
        target_desc: str | None = None,
        expected_package: str | None = None,
    ) -> bool:
        """Verify tap target via deterministic OS-level checks on the Android sidecar.

        Checks (all fast, <100ms total):
          1. Foreground package matches expected target
          2. No suspicious overlay / obscuration windows
          3. Target node exists in accessibility tree with valid bounds
          4. Node is enabled and visible
          5. Node is stable across two rapid accessibility snapshots

        Returns True if all checks pass or sidecar unavailable, False if blocked.
        """
        payload = {}
        if target_text:
            payload["target_text"] = target_text
        if target_desc:
            payload["target_desc"] = target_desc
        if expected_package:
            payload["expected_package"] = expected_package

        try:
            req = Request(
                _SIDECAR_UI_INTEGRITY_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=_SIDECAR_TIMEOUT_S) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError, json.JSONDecodeError) as e:
            # Sidecar unavailable — allow tap (fail-open for availability,
            # Layer 1-3 text pipeline remains the primary defense)
            logger.warning(f"UI integrity sidecar unavailable: {e} — allowing tap")
            return True

        verdict = result.get("verdict", "ALLOW")
        checks = result.get("checks", [])

        if verdict == "BLOCK":
            failed = [c for c in checks if not c.get("pass", True)]
            reasons = ", ".join(c.get("check", "?") for c in failed)
            logger.warning(
                f"UI INTEGRITY BLOCKED tap on '{target_text or target_desc}': "
                f"failed checks: [{reasons}]"
            )
            for c in failed:
                logger.debug(f"  check={c.get('check')}: {json.dumps(c)}")
            return False

        logger.debug(
            f"UI integrity passed for '{target_text or target_desc}' "
            f"({len(checks)} checks, pkg={result.get('foreground_package', '?')})"
        )
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

                # OS-level UI integrity check (deterministic, <100ms)
                if target_text or target_desc:
                    # Snapshot foreground package so sidecar can verify it hasn't changed
                    try:
                        expected_pkg = self._d.app_current().get("package")
                    except Exception:
                        expected_pkg = None
                    if not self._verify_ui_integrity(target_text, target_desc, expected_pkg):
                        return "blocked_by_ui_integrity"

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
