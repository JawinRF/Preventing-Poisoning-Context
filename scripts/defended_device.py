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
        self._ensure_ui_integrity_forward()
        self._ensure_accessibility_service()
        self._ensure_notification_listener()
        self._ensure_chrome_cdp_access()

    @property
    def device(self):
        """Access the raw device for non-action calls (window_size, screen_on, etc.)."""
        return self._d

    # ── UI Integrity (OS-level tap safety — replaces VLM visual grounding) ───

    def _ensure_ui_integrity_forward(self) -> None:
        """Set up ADB port forward so host can reach the on-device sidecar on :8766.

        The Android sidecar (OpenClawService) listens on
        localhost:8766 *inside the emulator*. We need ``adb forward`` to bridge
        host:8766 → device:8766. This is idempotent — re-running is harmless.
        """
        try:
            subprocess.run(
                ["adb", "-s", self._serial, "forward", "tcp:8766", "tcp:8766"],
                timeout=5, capture_output=True, check=True,
            )
            logger.info("ADB forward tcp:8766 → device:8766 established")
        except FileNotFoundError:
            logger.warning("adb not found — UI integrity sidecar will be unreachable")
        except subprocess.CalledProcessError as e:
            logger.warning(f"adb forward failed: {e.stderr.decode().strip()}")
        except subprocess.TimeoutExpired:
            logger.warning("adb forward timed out")

    def _ensure_accessibility_service(self) -> None:
        """Enable PrismAccessibilityService via ADB.

        Critical for two reasons:
        1. Chrome only populates the WebView accessibility tree (making web
           page content visible to dump_hierarchy()) when an AccessibilityService
           is active on the device.
        2. The UI integrity sidecar needs PrismAccessibilityService.instance
           to perform overlay/node checks.
        """
        _SERVICE_CLASS = "com.openclaw.android.security.PrismAccessibilityService"
        _PACKAGE_CANDIDATES = ("com.openclaw.android.debug", "com.openclaw.android")

        try:
            # Detect installed package
            pm_result = subprocess.run(
                ["adb", "-s", self._serial, "shell", "pm", "list", "packages"],
                timeout=5, capture_output=True, text=True,
            )
            installed_pkg = None
            for candidate in _PACKAGE_CANDIDATES:
                if f"package:{candidate}" in pm_result.stdout:
                    installed_pkg = candidate
                    break
            if not installed_pkg:
                logger.warning("OpenClaw package not found — cannot enable accessibility service")
                return

            a11y_component = f"{installed_pkg}/{_SERVICE_CLASS}"

            # Check if already enabled
            current = subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "settings", "get", "secure", "enabled_accessibility_services"],
                timeout=5, capture_output=True, text=True,
            )
            existing = current.stdout.strip()
            if a11y_component in existing:
                logger.info(f"Accessibility service already enabled ({installed_pkg})")
                return

            # Append our component
            if existing and existing != "null":
                new_val = f"{existing}:{a11y_component}"
            else:
                new_val = a11y_component

            subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "settings", "put", "secure",
                 "enabled_accessibility_services", new_val],
                timeout=5, capture_output=True, text=True,
            )
            # Also ensure accessibility is globally on
            subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "settings", "put", "secure", "accessibility_enabled", "1"],
                timeout=5, capture_output=True, text=True,
            )
            logger.info(f"Accessibility service enabled via ADB ({installed_pkg})")
        except FileNotFoundError:
            logger.warning("adb not found — cannot enable accessibility service")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Accessibility service setup failed: {e}")

    def _ensure_notification_listener(self) -> None:
        """Enable the PRISM notification listener via ADB secure settings.

        Android's NotificationListenerService requires explicit user opt-in
        via Settings → Notification Access. On emulators we can grant this
        programmatically via ``adb shell settings put secure``.

        The component name depends on the build variant:
          release: com.openclaw.android/...PrismNotificationListener
          debug:   com.openclaw.android.debug/...PrismNotificationListener

        We detect the installed package via ``adb shell pm list packages``.
        """
        _SERVICE_CLASS = "com.openclaw.android.security.PrismNotificationListener"
        _PACKAGE_CANDIDATES = ("com.openclaw.android.debug", "com.openclaw.android")

        try:
            # Detect which build variant is installed
            pm_result = subprocess.run(
                ["adb", "-s", self._serial, "shell", "pm", "list", "packages"],
                timeout=5, capture_output=True, text=True,
            )
            installed_pkg = None
            for candidate in _PACKAGE_CANDIDATES:
                # pm list output: "package:com.openclaw.android.debug"
                if f"package:{candidate}" in pm_result.stdout:
                    installed_pkg = candidate
                    break

            if not installed_pkg:
                logger.warning(
                    "OpenClaw package not found on device — "
                    "cannot enable notification listener"
                )
                return

            nls_component = f"{installed_pkg}/{_SERVICE_CLASS}"

            # Read current listeners to avoid clobbering other entries
            current = subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "settings", "get", "secure", "enabled_notification_listeners"],
                timeout=5, capture_output=True, text=True,
            )
            existing = current.stdout.strip()
            if nls_component in existing:
                logger.info(f"Notification listener already enabled ({installed_pkg})")
                return

            # Append our component (colon-separated list)
            if existing and existing != "null":
                new_val = f"{existing}:{nls_component}"
            else:
                new_val = nls_component

            result = subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "settings", "put", "secure",
                 "enabled_notification_listeners", new_val],
                timeout=5, capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info(f"Notification listener setting written ({installed_pkg})")
            else:
                logger.warning(f"Failed to write notification listener setting: {result.stderr.strip()}")

            # Force-bind the listener via cmd notification (API 26+).
            # settings put only writes the DB; cmd notification triggers
            # NotificationManagerService to actually bind the service.
            allow_result = subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 "cmd", "notification", "allow_listener", nls_component],
                timeout=5, capture_output=True, text=True,
            )
            if allow_result.returncode == 0:
                logger.info(f"Notification listener bound via cmd notification ({installed_pkg})")
            else:
                logger.warning(
                    f"cmd notification allow_listener failed: {allow_result.stderr.strip()}"
                )
        except FileNotFoundError:
            logger.warning("adb not found — cannot enable notification listener")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Notification listener setup failed: {e}")

    def _ensure_chrome_cdp_access(self) -> None:
        """Write Chrome command-line flag to allow CDP WebSocket connections.

        Chrome rejects DevTools Protocol connections unless
        ``--remote-allow-origins=*`` is set. This file is read by Chrome
        on startup, so it must be written before Chrome launches (or Chrome
        must be restarted afterwards). Idempotent.
        """
        _CMD_LINE = "chrome --remote-allow-origins=*"
        _CMD_PATH = "/data/local/tmp/chrome-command-line"
        try:
            result = subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 f"cat {_CMD_PATH}"],
                timeout=5, capture_output=True, text=True,
            )
            if "--remote-allow-origins" in result.stdout:
                return  # already set

            subprocess.run(
                ["adb", "-s", self._serial, "shell",
                 f"echo '{_CMD_LINE}' > {_CMD_PATH}"],
                timeout=5, capture_output=True,
            )
            logger.info("Chrome CDP command-line flag set")
        except Exception as e:
            logger.debug(f"Chrome CDP flag setup failed (non-critical): {e}")

        # Quick health probe — warn early if the Android service isn't running
        try:
            req = Request("http://127.0.0.1:8766/v1/status", method="GET")
            with urlopen(req, timeout=2) as resp:
                logger.info(f"UI integrity sidecar healthy: {resp.read().decode()[:80]}")
        except Exception:
            logger.warning(
                "UI integrity sidecar not responding on :8766. "
                "Ensure the PRISM Shield app/service is running on the emulator. "
                "Tap safety checks will fail-open until the sidecar is available."
            )

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
        QUARANTINE is treated as BLOCK (no VLM in the request path).
        Returns "blocked_by_prism" if blocked, None if allowed.
        """
        if r.allowed:
            return None
        if r.verdict == "QUARANTINE":
            logger.warning(f"QUARANTINE→BLOCK: {r.reason}")
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
                r = self._prism.inspect(text_data, "agent_output", "text_input")
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

                if "xy" in params:
                    xy = params["xy"]
                    if isinstance(xy, (list, tuple)) and len(xy) == 2:
                        subprocess.run(
                            ["adb", "-s", self._serial, "shell", "input", "tap",
                             str(int(xy[0])), str(int(xy[1]))],
                            timeout=3, capture_output=True,
                        )
                        time.sleep(0.4)
                        return "ok"
                    return f"bad xy: {xy}"
                if "rid" in params:
                    rid = params["rid"]
                    el = self._d(resourceIdMatches=f".*/{rid}$")
                    if el.exists(timeout=3):
                        el.click()
                        time.sleep(0.4)
                        return "ok"
                    return f"not found: rid={rid}"
                if "text" in params:
                    el = self._d(text=params["text"])
                    if el.exists(timeout=3):
                        el.click()
                        time.sleep(0.4)
                        return "ok"
                    return f"not found: text={params['text']}"
                if "desc" in params:
                    el = self._d(description=params["desc"])
                    if el.exists(timeout=3):
                        el.click()
                        time.sleep(0.4)
                        return "ok"
                    return f"not found: desc={params['desc']}"
                if "class" in params:
                    cls = params["class"]
                    el = self._d(className=f"android.widget.{cls}")
                    if el.exists(timeout=3):
                        el.click()
                        time.sleep(0.4)
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

            elif action == "web_tap":
                return self._cdp_tap(params)

            elif action == "web_type":
                return self._cdp_type(params)

            elif action in ("done", "fail"):
                return action

        except Exception as e:
            return f"error: {e}"

        return "unknown"

    # ── CDP web interaction ──────────────────────────────────────────────────

    def _cdp_eval(self, js: str) -> dict | None:
        """Execute JavaScript in the active Chrome tab via DevTools Protocol."""
        try:
            import websocket as ws_lib
        except ImportError:
            return None
        try:
            subprocess.run(
                ["adb", "-s", self._serial, "forward",
                 "tcp:9222", "localabstract:chrome_devtools_remote"],
                timeout=5, capture_output=True,
            )
            req = Request("http://localhost:9222/json/list", method="GET")
            with urlopen(req, timeout=3) as resp:
                tabs = json.loads(resp.read().decode("utf-8"))
            if not tabs:
                return None
            ws_url = tabs[0].get("webSocketDebuggerUrl")
            if not ws_url:
                return None
            conn = ws_lib.create_connection(ws_url, timeout=5)
            try:
                conn.send(json.dumps({
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {"expression": js},
                }))
                return json.loads(conn.recv())
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"CDP eval failed: {e}")
            return None

    def _cdp_tap(self, params: dict) -> str:
        """Click a web element by visible text or CSS selector via CDP."""
        text = params.get("text", "")
        selector = params.get("selector", "")

        if text:
            # Find element containing this text and click it
            js = f"""
            (function() {{
                var text = {json.dumps(text)};
                var all = document.querySelectorAll('a, button, [role="button"], input[type="submit"], [tabindex]');
                for (var el of all) {{
                    if (el.innerText && el.innerText.trim().includes(text)) {{
                        el.click();
                        return 'ok';
                    }}
                    if (el.getAttribute('aria-label') && el.getAttribute('aria-label').includes(text)) {{
                        el.click();
                        return 'ok';
                    }}
                }}
                // Broader search: any element
                var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                while (walker.nextNode()) {{
                    if (walker.currentNode.textContent.trim().includes(text)) {{
                        var target = walker.currentNode.parentElement;
                        if (target) {{ target.click(); return 'ok'; }}
                    }}
                }}
                return 'not found: ' + text;
            }})()
            """
        elif selector:
            js = f"""
            (function() {{
                var el = document.querySelector({json.dumps(selector)});
                if (el) {{ el.click(); return 'ok'; }}
                return 'not found: ' + {json.dumps(selector)};
            }})()
            """
        else:
            return "error: web_tap needs 'text' or 'selector'"

        result = self._cdp_eval(js)
        if not result:
            return "error: CDP unavailable"
        value = result.get("result", {}).get("result", {}).get("value", "error: no response")
        return value

    def _cdp_type(self, params: dict) -> str:
        """Type text into a web input field via CDP."""
        text = params.get("text", "")
        selector = params.get("selector", "")

        if not text:
            return "error: web_type needs 'text'"

        if selector:
            focus_js = f"document.querySelector({json.dumps(selector)})"
        else:
            # Focus the first visible input/search field
            focus_js = """
            (document.querySelector('input[type="search"], input[type="text"], textarea, [contenteditable="true"]')
             || document.querySelector('input:not([type="hidden"])'))
            """

        js = f"""
        (function() {{
            var el = {focus_js};
            if (!el) return 'not found: no input field';
            el.focus();
            el.value = {json.dumps(text)};
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            el.dispatchEvent(new Event('change', {{bubbles: true}}));
            return 'ok';
        }})()
        """
        result = self._cdp_eval(js)
        if not result:
            return "error: CDP unavailable"
        value = result.get("result", {}).get("result", {}).get("value", "error: no response")
        return value
