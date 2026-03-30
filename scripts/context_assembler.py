"""
context_assembler.py — Gathers context from 6 Android ingestion paths,
filters each through the PRISM Shield sidecar, and returns a clean
AssembledContext that the agent LLM can safely consume.
(Network response monitoring is planned but not yet implemented.)

This is the core defense: PRISM sits BETWEEN the Android sources and the LLM.
"""
from __future__ import annotations
import json, logging, re, socket, subprocess, uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from prism_client import PrismClient

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AssembledContext:
    task: str
    step: int = 0
    screen_changed: bool = True
    ui_elements: list[dict] = field(default_factory=list)
    notifications: list[dict] = field(default_factory=list)
    clipboard: str | None = None
    intent_data: list[dict] = field(default_factory=list)
    storage_data: list[dict] = field(default_factory=list)
    rag_context: list[str] = field(default_factory=list)
    blocked_counts: dict[str, int] = field(default_factory=dict)
    audit_trail: list[dict] = field(default_factory=list)
    screenshot_path: str | None = None  # For VLM verification on QUARANTINE

    def to_prompt_dict(self) -> dict:
        """Build the dict that gets sent to the LLM."""
        d = {
            "task": self.task,
            "step": self.step,
            "screen_changed": self.screen_changed,
            "screen": self.ui_elements,
        }
        if self.notifications:
            d["notifications"] = [
                f"[{n['package']}] {n['title']}: {n['text']}"
                for n in self.notifications
            ]
        if self.clipboard:
            d["clipboard"] = self.clipboard
        if self.rag_context:
            d["context"] = self.rag_context

        total_blocked = sum(self.blocked_counts.values())
        if total_blocked > 0:
            d["security_note"] = (
                f"PRISM Shield filtered {total_blocked} potentially malicious "
                f"item(s) from your context. Proceed with the legitimate task."
            )
        return d


@dataclass
class Notification:
    package: str
    title: str
    text: str


# ── Context Assembler ─────────────────────────────────────────────────────────

class ContextAssembler:
    """
    Gathers context from 6 Android ingestion paths, filters each
    through the PRISM Shield sidecar, and returns only clean data.
    """

    def __init__(
        self,
        device,                        # uiautomator2 device object
        prism: PrismClient,
        serial: str = "emulator-5554",
        memshield=None,                # optional MemShield instance for RAG
        watched_paths: list[str] | None = None,
    ):
        self.device = device
        self.prism = prism
        self.serial = serial
        self.memshield = memshield
        self.watched_paths = watched_paths or []

    def assemble(
        self,
        task: str,
        step: int = 0,
        last_sig: str | None = None,
        rag_query: str | None = None,
        agent_typed_texts: set[str] | None = None,
        recent_actions: list[dict] | None = None,
    ) -> AssembledContext:
        """
        Main entry point. Gathers all sources, filters through PRISM,
        returns clean AssembledContext.
        """
        ctx = AssembledContext(task=task, step=step)
        self._agent_typed_texts = agent_typed_texts or set()

        # 1. UI Accessibility (most critical path)
        ctx.ui_elements, ui_blocked, screenshot_path = self._gather_ui()
        ctx.blocked_counts["ui_accessibility"] = ui_blocked
        ctx.screenshot_path = screenshot_path  # Store for VLM to use

        # Compute screen signature for change detection
        current_sig = self._sig(ctx.ui_elements)
        ctx.screen_changed = current_sig != last_sig

        # 2. Notifications
        ctx.notifications, notif_blocked = self._gather_notifications()
        ctx.blocked_counts["notifications"] = notif_blocked

        # 3. Clipboard
        ctx.clipboard, clip_blocked = self._gather_clipboard()
        ctx.blocked_counts["clipboard"] = clip_blocked

        # 4. Android Intents
        ctx.intent_data, intent_blocked = self._gather_intents()
        ctx.blocked_counts["android_intents"] = intent_blocked

        # 5. Shared Storage
        ctx.storage_data, stor_blocked = self._gather_storage()
        ctx.blocked_counts["shared_storage"] = stor_blocked

        # 6. RAG Store
        ctx.rag_context, rag_blocked = self._gather_rag(rag_query or task, recent_actions)
        ctx.blocked_counts["rag_store"] = rag_blocked

        return ctx

    # ── 1. UI Accessibility ──────────────────────────────────────────────────

    def _capture_screenshot(self) -> str | None:
        """Capture screenshot and save to temp file for VLM processing."""
        try:
            # Create temp directory for screenshots
            scripts_dir = Path(__file__).resolve().parent
            temp_dir = scripts_dir.parent / "data" / "screenshots"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            screenshot_path = temp_dir / f"screen_{uuid.uuid4().hex[:8]}.png"
            self.device.screenshot(str(screenshot_path))
            logger.debug(f"Screenshot captured: {screenshot_path}")
            return str(screenshot_path)
        except Exception as e:
            logger.warning(f"Screenshot capture failed: {e}")
            return None

    def _gather_ui(self) -> tuple[list[dict], int, str | None]:
        """Read screen dump, filter through PRISM, return clean elements + screenshot path."""
        # Capture screenshot first (for VLM on QUARANTINE)
        screenshot_path = self._capture_screenshot()
        
        try:
            raw_xml = self.device.dump_hierarchy()
            root = ET.fromstring(raw_xml)
        except Exception as exc:
            logger.warning(f"UI hierarchy dump failed (fail-closed, returning empty): {exc}")
            return [], 0, screenshot_path

        elements = self._parse_ui_tree(root)
        if not elements:
            return [], 0, screenshot_path

        # Fast path: concatenate all text, single PRISM check
        all_text = " ".join(
            f"{e.get('text', '')} {e.get('desc', '')}".strip()
            for e in elements if e.get("text") or e.get("desc")
        )

        if not all_text.strip():
            return elements[:30], 0, screenshot_path

        batch_result = self.prism.inspect(
            text=all_text,
            ingestion_path="ui_accessibility",
            source_type="accessibility",
            source_name="screen_dump",
            metadata={"screenshot_path": screenshot_path} if screenshot_path else {},
        )

        if batch_result.allowed:
            # Entire screen is clean — pass everything through
            return elements[:30], 0, screenshot_path

        # Slow path: screen flagged, filter per-element to find the poison
        allowed = []
        blocked_count = 0

        for elem in elements:
            elem_text = f"{elem.get('text', '')} {elem.get('desc', '')}".strip()
            if not elem_text:
                # Structural elements (no text) are safe
                allowed.append(elem)
                continue

            # Skip PRISM for:
            # 1. Short UI labels (buttons, dates) — false positives in Layer 2/3
            # 2. Input fields — contain agent's own typed text
            # 3. Text that matches what the agent recently typed — not external
            if len(elem_text) <= 20 or elem.get("input_field"):
                allowed.append(elem)
                continue

            if self._is_agent_text(elem_text):
                allowed.append(elem)
                continue

            result = self.prism.inspect(
                text=elem_text,
                ingestion_path="ui_accessibility",
                source_type="accessibility",
                source_name=elem.get("package", "unknown"),
                metadata={"screenshot_path": screenshot_path} if screenshot_path else {},
            )

            if result.allowed:
                allowed.append(elem)
            else:
                blocked_count += 1
                # Replace with safe placeholder so LLM knows something was here
                allowed.append({
                    "class": elem.get("class", "View"),
                    "text": "[PRISM_FILTERED]",
                })
                logger.warning(
                    f"UI element BLOCKED: '{elem_text[:60]}' — {result.reason}"
                )

        return allowed[:30], blocked_count, screenshot_path

    def _parse_ui_tree(self, root: ET.Element) -> list[dict]:
        """Parse XML hierarchy into element dicts (reused from agent.py)."""
        elems = []
        for node in root.iter():
            text = node.attrib.get("text", "").strip()
            desc = node.attrib.get("content-desc", "").strip()
            cls = node.attrib.get("class", "").split(".")[-1]
            click = node.attrib.get("clickable") == "true"
            enabled = node.attrib.get("enabled", "true") == "true"
            selected = node.attrib.get("selected", "false") == "true"
            focused = node.attrib.get("focused", "false") == "true"
            hint = node.attrib.get("hint", "").strip()
            pkg = node.attrib.get("package", "")

            if "EditText" in cls or "TextInputEditText" in cls:
                e = {"class": cls, "input_field": True}
                if text: e["text"] = text
                if desc: e["desc"] = desc
                if hint: e["hint"] = hint
                if not enabled: e["disabled"] = True
                if focused: e["focused"] = True
                if pkg: e["package"] = pkg
                elems.append(e)
                continue

            if not text and not desc:
                continue

            e = {"class": cls}
            if text: e["text"] = text
            if desc: e["desc"] = desc
            if click: e["clickable"] = True
            if not enabled: e["disabled"] = True
            if selected: e["selected"] = True
            if focused: e["focused"] = True
            if pkg: e["package"] = pkg
            elems.append(e)

        # Clickable elements first
        sorted_elems = []
        for e in elems:
            c = e.pop("clickable", False)
            if c:
                sorted_elems.insert(0, e)
            else:
                sorted_elems.append(e)

        return sorted_elems

    # ── 2. Notifications ─────────────────────────────────────────────────────

    def _gather_notifications(self) -> tuple[list[dict], int]:
        """Read active notifications via native Android socket.
        
        Raises exception if native socket fails (no silent fallback to ADB).
        This ensures the native pipeline is always used and any issues are visible.
        """
        # Native socket only (PrismNotificationListener)
        notifications = self._gather_notifications_native()
        if notifications is None:
            raise RuntimeError(
                "Native notification socket unavailable. "
                "Ensure PrismNotificationListener service is running on the device."
            )

        if not notifications:
            return [], 0

        allowed = []
        blocked = 0

        for notif in notifications:
            text = f"{notif.title} {notif.text}".strip()
            if not text:
                continue

            r = self.prism.inspect(
                text=text,
                ingestion_path="notifications",
                source_type="notification",
                source_name=notif.package,
            )

            if r.allowed:
                allowed.append({
                    "package": notif.package,
                    "title": notif.title,
                    "text": notif.text,
                })
            else:
                blocked += 1
                logger.warning(
                    f"Notification BLOCKED: [{notif.package}] '{text[:60]}'"
                )

        return allowed, blocked

    def _gather_notifications_native(self) -> list[Notification] | None:
        """Read notifications via native Android socket from PrismNotificationListener."""
        LOCAL_PORT = 8767  # Fixed local port for ADB forward

        try:
            # Forward local:LOCAL_PORT to device's localabstract:prism_notif socket
            subprocess.run(
                ["adb", "-s", self.serial, "forward", f"tcp:{LOCAL_PORT}", "localabstract:prism_notif"],
                capture_output=True, timeout=3,
            )

            # Connect via forwarded port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect(("127.0.0.1", LOCAL_PORT))

            sock.send(b'{"action":"list"}\n')
            response = sock.recv(4096).decode("utf-8")
            sock.close()

            data = json.loads(response)
            notifications = []
            for n in data.get("notifications", []):
                notifications.append(
                    Notification(
                        package=n.get("package", "unknown"),
                        title=n.get("title", ""),
                        text=n.get("text", "")
                    )
                )

            logger.debug(f"Native notifications: {len(notifications)} received")
            return notifications

        except Exception as e:
            logger.debug(f"Native notification socket unavailable: {e}")
            return None

    # ── 3. Clipboard ─────────────────────────────────────────────────────────

    def _gather_clipboard(self) -> tuple[str | None, int]:
        """Read clipboard content via ADB, filter through PRISM."""
        try:
            result = subprocess.run(
                ["adb", "-s", self.serial, "shell",
                 "service", "call", "clipboard", "2", "s16", "com.android.shell"],
                capture_output=True, text=True, timeout=3,
            )
            clip_text = self._parse_service_call(result.stdout)
        except Exception as exc:
            logger.warning(f"Clipboard read failed (fail-closed, returning empty): {exc}")
            return None, 0

        if not clip_text:
            return None, 0

        r = self.prism.inspect(
            text=clip_text,
            ingestion_path="clipboard",
            source_type="clipboard",
            source_name="system_clipboard",
        )

        if r.allowed:
            return clip_text, 0

        logger.warning(f"Clipboard BLOCKED: '{clip_text[:60]}' — {r.reason}")
        return None, 1

    @staticmethod
    def _parse_service_call(output: str) -> str | None:
        """Parse text from `service call clipboard` output."""
        parts = re.findall(r"'([^']*)'", output)
        text = "".join(parts).replace(".", "").strip()
        return text if text and len(text) > 1 else None

    # ── 4. Android Intents ───────────────────────────────────────────────────

    def _gather_intents(self) -> tuple[list[dict], int]:
        """Read recent intent broadcasts from logcat, filter through PRISM."""
        try:
            result = subprocess.run(
                ["adb", "-s", self.serial, "shell",
                 "logcat", "-d", "-s", "ActivityManager:I", "-t", "20"],
                capture_output=True, text=True, timeout=3,
            )
        except Exception as exc:
            logger.warning(f"Intent gathering failed (fail-closed, returning empty): {exc}")
            return [], 0

        intents = []
        for line in result.stdout.split("\n"):
            if "START" in line and "dat=" in line:
                m = re.search(r"dat=(\S+)", line)
                if m:
                    intents.append({"type": "deep_link", "data": m.group(1)})

        if not intents:
            return [], 0

        allowed = []
        blocked = 0

        for intent in intents:
            r = self.prism.inspect(
                text=intent["data"],
                ingestion_path="android_intents",
                source_type="intent",
                source_name="activity_manager",
            )
            if r.allowed:
                allowed.append(intent)
            else:
                blocked += 1

        return allowed, blocked

    # ── Network Responses (NOT IMPLEMENTED) ────────────────────────────────
    # Would require a proxy or VPN-based traffic interceptor on the device.
    # Not called from assemble() — kept as interface placeholder.

    def _gather_network(self) -> tuple[list[dict], int]:
        """Not implemented — no proxy/VPN interceptor available."""
        return [], 0

    # ── 5. Shared Storage ────────────────────────────────────────────────────

    def _gather_storage(self) -> tuple[list[dict], int]:
        """Read watched files from device storage, filter through PRISM."""
        if not self.watched_paths:
            return [], 0

        allowed = []
        blocked = 0

        for path in self.watched_paths:
            try:
                result = subprocess.run(
                    ["adb", "-s", self.serial, "shell", "cat", path],
                    capture_output=True, text=True, timeout=3,
                )
                content = result.stdout.strip()
            except Exception as exc:
                logger.warning(f"Storage file read failed for {path} (fail-closed, skipping): {exc}")
                continue

            if not content:
                continue

            r = self.prism.inspect(
                text=content[:2000],
                ingestion_path="shared_storage",
                source_type="file",
                source_name=path,
            )

            if r.allowed:
                allowed.append({"path": path, "content": content[:500]})
            else:
                blocked += 1
                logger.warning(f"Storage file BLOCKED: {path} — {r.reason}")

        return allowed, blocked

    # ── 7. RAG Store ─────────────────────────────────────────────────────────

    def _gather_rag(
        self, query: str, recent_actions: list[dict] | None = None,
    ) -> tuple[list[str], int]:
        """Query MemShield-wrapped ChromaDB with task + conversational context."""
        if self.memshield is None:
            return [], 0

        # Enrich query with recent successful actions for better retrieval
        enriched = query
        if recent_actions:
            action_context = " ".join(
                f"{a['action']} {a.get('params', {}).get('text', '')}"
                for a in recent_actions[-2:]
                if a.get("result") == "ok"
            ).strip()
            if action_context:
                enriched = f"{query} | recent: {action_context}"

        try:
            results = self.memshield.query(
                query_texts=[enriched],
                n_results=5,
                session_id=self.prism.session_id,
            )
            docs = results.get("documents", [[]])[0]
            return docs, 0
        except Exception as e:
            logger.warning(f"RAG query failed: {e}")
            return [], 0

    # ── Utilities ────────────────────────────────────────────────────────────

    def _is_agent_text(self, text: str) -> bool:
        """Check if text contains something the agent itself typed."""
        for typed in self._agent_typed_texts:
            # The screen may show the typed text verbatim, truncated,
            # or repeated (from previous failed attempts)
            if typed in text or text in typed:
                return True
        return False

    @staticmethod
    def _sig(elems: list[dict]) -> str:
        """Screen signature for change detection."""
        parts = []
        for e in elems:
            part = f"{e.get('text', '')}{e.get('desc', '')}{e.get('class', '')}"
            if e.get("disabled"): part += "_D"
            if e.get("selected"): part += "_S"
            if e.get("focused"): part += "_F"
            if part.strip():
                parts.append(part)
        return str(sorted(parts))

    def get_screen_sig(self, ctx: AssembledContext) -> str:
        """Public accessor for screen signature."""
        return self._sig(ctx.ui_elements)
