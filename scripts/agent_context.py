"""
agent_context.py
Simulates how a real mobile agent assembles context from the device.
Notifications are read via ADB, then routed through PRISM before
being added to the agent's context window.
"""
import subprocess, requests, uuid, logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PRISM_URL = "http://localhost:8765/v1/inspect"

@dataclass
class Notification:
    pkg: str
    title: str
    text: str

def read_active_notifications(serial: str = "emulator-5554") -> list[Notification]:
    """
    Read current notifications from the device via ADB dumpsys.
    This is what a real agent would do to get notification context.
    """
    result = subprocess.run(
        f"adb -s {serial} shell dumpsys notification --noredact",
        shell=True, capture_output=True, text=True
    )
    notifications = []
    current_pkg = "unknown"
    current_title = ""
    current_text = ""

    for line in result.stdout.split("\n"):
        line = line.strip()
        if "NotificationRecord" in line and "pkg=" in line:
            # Save previous
            if current_text:
                notifications.append(Notification(current_pkg, current_title, current_text))
            # Parse new
            import re
            m = re.search(r"pkg=(\S+)", line)
            current_pkg = m.group(1) if m else "unknown"
            current_title = ""
            current_text = ""
        elif line.startswith("android.title"):
            current_title = line.split("=", 1)[-1].strip()
        elif line.startswith("android.text"):
            current_text = line.split("=", 1)[-1].strip()

    if current_text:
        notifications.append(Notification(current_pkg, current_title, current_text))

    return notifications

def build_agent_context(
    task: str,
    serial: str = "emulator-5554",
    session_id: str = "demo",
) -> dict:
    """
    Assembles agent context from:
    - The user's task
    - Active notifications (filtered through PRISM)

    Returns:
    {
      "task": str,
      "safe_notifications": [...],   # passed PRISM
      "blocked_notifications": [...], # blocked by PRISM
      "context_text": str,            # what the agent actually sees
    }
    """
    print(f"\n📱 Reading notifications from device...")
    notifications = read_active_notifications(serial)
    print(f"   Found {len(notifications)} active notification(s)")

    safe = []
    blocked = []

    for notif in notifications:
        text = f"{notif.title} {notif.text}".strip()
        if not text:
            continue

        # Route through PRISM
        payload = {
            "entry_id":       str(uuid.uuid4()),
            "text":           text,
            "ingestion_path": "notifications",
            "source_type":    "notification",
            "source_name":    notif.pkg,
            "session_id":     session_id,
            "run_id":         "agent-context-build",
        }
        try:
            resp = requests.post(PRISM_URL, json=payload, timeout=5)
            decision = resp.json()
            verdict = decision.get("verdict", "BLOCK")
        except Exception as exc:
            decision = {"reason": str(exc)}
            verdict = "BLOCK"  # fail closed

        if verdict == "ALLOW":
            safe.append(notif)
            print(f"   ✅ ALLOWED: [{notif.pkg}] '{text[:60]}'")
        else:
            blocked.append(notif)
            print(f"   🚫 BLOCKED: [{notif.pkg}] '{text[:60]}'")
            print(f"      Reason: {decision.get('reason','')[:80]}")

    # Build context string — only safe notifications
    context_parts = [f"User task: {task}"]
    if safe:
        context_parts.append("\nDevice notifications:")
        for n in safe:
            context_parts.append(f"  - [{n.pkg}] {n.title}: {n.text}")

    return {
        "task": task,
        "safe_notifications": safe,
        "blocked_notifications": blocked,
        "context_text": "\n".join(context_parts),
    }
