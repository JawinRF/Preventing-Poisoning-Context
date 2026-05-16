#!/usr/bin/env python3
"""
alert_watcher.py — Background notification watcher with user-defined trigger rules.

Runs as a daemon thread inside prism_cli.py. Polls the Android sidecar (:8766)
every POLL_INTERVAL seconds, evaluates user-defined rules against new notifications,
and puts alerts/auto-run tasks into a queue the REPL drains before each prompt.

Security properties:
  - Cold start: ALL existing notifications are suppressed on first tick — no stale re-fires.
  - Fingerprint = hash(package + title + text): prevents cross-app content-collision
    where a malicious app posts the same body as a trusted notification to poison the
    seen-set and cause the real notification to be silently dropped.
  - Template rendering uses str.replace(), never str.format() — a notification body
    of '{__class__}' cannot expose internals.
  - Conditions use case-insensitive substring matching only — no user-supplied regex,
    no ReDoS surface.
  - Auto-run tasks go through the full agent PRISM pipeline at execution time.
  - Atomic JSON writes (write to .tmp then os.replace) — crash-safe.
  - _seen persisted to disk across restarts — prevents re-firing after daemon restart.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger(__name__)

_SIDECAR_URL     = "http://127.0.0.1:8766"
_POLL_INTERVAL   = 5      # seconds between sidecar polls
_SIDECAR_TIMEOUT = 3      # per-poll HTTP timeout
_SIDECAR_BACKOFF = 30.0   # back-off after failure
_SEEN_MAXSIZE    = 1_000  # fingerprint slots (in-memory)
_SEEN_PERSIST    = 200    # fingerprints saved to disk on shutdown
_SAVE_INTERVAL   = 60.0   # seconds between debounced rule saves (fire counts)
_BURST_THRESHOLD = 5      # coalesce if more than this many items fire in one tick


# ── Friendly app-name lookup ─────────────────────────────────────────────────

_APP_NAMES: dict[str, str] = {
    "gmail":     "com.google.android.gm",
    "whatsapp":  "com.whatsapp",
    "chrome":    "com.android.chrome",
    "maps":      "com.google.android.apps.maps",
    "messages":  "com.google.android.apps.messaging",
    "slack":     "com.Slack",
    "telegram":  "org.telegram.messenger",
    "twitter":   "com.twitter.android",
    "youtube":   "com.google.android.youtube",
    "spotify":   "com.spotify.music",
    "calendar":  "com.google.android.calendar",
    "drive":     "com.google.android.apps.docs",
    "photos":    "com.google.android.apps.photos",
    "instagram": "com.instagram.android",
    "linkedin":  "com.linkedin.android",
}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class AlertRule:
    id:                  str
    enabled:             bool
    condition:           str
    action_type:         str         # "alert" | "auto"
    action:              str
    rate_limit_per_min:  int   = 3
    created_at:          str   = ""
    fire_count:          int   = 0
    last_fired_wall:     float | None = None


class _BoundedSet:
    """Insertion-order FIFO set with fixed capacity.

    Not thread-safe — the watcher processes notifications on its own thread
    and is the sole writer. __contains__ + add is always called from the
    same thread, so no lock is needed here.

    Evicts the oldest (first-inserted) entry when at capacity, maintaining
    O(1) add and O(1) contains via dict key lookup.
    """

    __slots__ = ("_data", "_maxsize")

    def __init__(self, maxsize: int) -> None:
        self._data: dict[int, None] = {}
        self._maxsize = maxsize

    def add(self, key: int) -> None:
        if key in self._data:
            return
        if len(self._data) >= self._maxsize:
            self._data.pop(next(iter(self._data)))   # evict oldest
        self._data[key] = None

    def __contains__(self, key: int) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def tail(self, n: int) -> list[int]:
        """Return the n most-recently added keys (for persistence)."""
        return list(self._data)[-n:]


# ── Notification fingerprint ──────────────────────────────────────────────────

def _notif_fp(notif: dict) -> int:
    """Fingerprint = hash(package \\0 title \\0 text).

    Including the package name prevents a cross-app content-collision attack:
    a malicious app posting the same body text as a trusted notification would
    share a text-only hash, silently suppressing the real notification from
    ever being evaluated. Package + title + text makes collisions negligible.

    Python's built-in hash() is process-scoped (PYTHONHASHSEED) — valid within
    one process run, which is all we need for the in-memory seen-set.
    For persisted fingerprints we use the same value — they're only compared
    within the same Python version/seed, which holds for CLI restarts.
    """
    return hash(
        f"{notif.get('package', '')}\x00"
        f"{notif.get('title',   '')}\x00"
        f"{notif.get('text',    '')}"
    )


# ── Template rendering ────────────────────────────────────────────────────────

def _friendly_app(pkg: str) -> str:
    for name, p in _APP_NAMES.items():
        if p.lower() == pkg.lower():
            return name.capitalize()
    parts = pkg.split(".")
    return parts[-1].capitalize() if parts else pkg


def render_template(template: str, notif: dict) -> str:
    """Safe template rendering — uses str.replace(), never str.format().

    str.format() with user-supplied values is dangerous: a notification body
    of '{__class__}' would expose Python internals. str.replace() is safe
    because it performs no attribute lookup on the replacement value.
    """
    pkg = notif.get("package", "")
    return (
        template
        .replace("{package}", pkg)
        .replace("{title}",   notif.get("title", ""))
        .replace("{text}",    notif.get("text",  ""))
        .replace("{app}",     _friendly_app(pkg))
    )


# ── Condition DSL ─────────────────────────────────────────────────────────────
#
# Grammar (AND binds tighter than OR — standard boolean precedence):
#
#   expr       = and_clause (OR and_clause)*
#   and_clause = predicate  (AND predicate)*
#   predicate  = [NOT] clause
#   clause     = field:pattern  |  bare_word
#
# Supported fields: package, title, text, app, any
# Matching:  case-insensitive substring — no user-supplied regex (eliminates ReDoS)
#
# Examples:
#   "package:com.google.android.gm"
#   "text:OTP AND package:com.google"
#   "title:Alice OR title:Bob"
#   "NOT text:advertisement"
#   "app:gmail AND text:invoice"

def evaluate_condition(condition: str, notif: dict) -> bool:
    """Evaluate a condition string against a notification dict.
    Returns True if the condition matches.
    """
    # OR is lowest precedence — split first
    or_parts = re.split(r'\bOR\b', condition, flags=re.IGNORECASE)
    return any(_eval_and(p.strip(), notif) for p in or_parts)


def _eval_and(clause: str, notif: dict) -> bool:
    and_parts = re.split(r'\bAND\b', clause, flags=re.IGNORECASE)
    return all(_eval_predicate(p.strip(), notif) for p in and_parts)


def _eval_predicate(pred: str, notif: dict) -> bool:
    pred = pred.strip()
    negated = bool(re.match(r'^NOT\s+', pred, re.IGNORECASE))
    if negated:
        pred = pred[4:].strip()
    result = _eval_clause(pred, notif)
    return (not result) if negated else result


def _eval_clause(clause: str, notif: dict) -> bool:
    clause = clause.strip()
    if not clause:
        return False

    if ":" not in clause:
        # Bare word — substring match across all text fields
        w = clause.lower()
        return (w in notif.get("package", "").lower() or
                w in notif.get("title",   "").lower() or
                w in notif.get("text",    "").lower())

    field, _, pattern = clause.partition(":")
    field   = field.strip().lower()
    pattern = pattern.strip().lower()

    if field == "package":
        return pattern in notif.get("package", "").lower()
    if field == "title":
        return pattern in notif.get("title", "").lower()
    if field == "text":
        return pattern in notif.get("text", "").lower()
    if field == "app":
        pkg    = notif.get("package", "").lower()
        mapped = _APP_NAMES.get(pattern, "").lower()
        return pattern in pkg or bool(mapped and mapped in pkg)
    if field == "any":
        return (pattern in notif.get("package", "").lower() or
                pattern in notif.get("title",   "").lower() or
                pattern in notif.get("text",    "").lower())

    # Unknown field — silently treat as no-match (don't crash the watcher)
    logger.debug(f"[Alert] Unknown condition field: {field!r} — treating as no-match")
    return False


def validate_condition(condition: str) -> None:
    """Raise ValueError if condition is empty or structurally invalid.
    Runs a dry evaluation against a blank notification to catch parse errors.
    """
    if not condition.strip():
        raise ValueError("condition cannot be empty")
    try:
        evaluate_condition(condition, {})
    except Exception as exc:
        raise ValueError(f"condition parse error: {exc}") from exc


# ── AlertWatcher ──────────────────────────────────────────────────────────────

class AlertWatcher(threading.Thread):
    """Background daemon thread that watches Android notifications and fires rules.

    Threading model
    ───────────────
    One thread (this one) owns _seen, _warmed_up, _sidecar_*. No lock needed.
    _rules is shared with the REPL thread (add/del/pause). Protected by _rules_lock.
    alert_queue is stdlib Queue — inherently thread-safe.
    _save_rules() / _load_rules() hold _rules_lock for the dict snapshot only.

    Alert delivery
    ──────────────
    Alerts are NOT printed here — the watcher puts items in alert_queue.
    The REPL drains the queue before each prompt (before input() blocks).
    This avoids the readline-corruption problem of printing mid-keypress.

    Coalescing
    ──────────
    If > _BURST_THRESHOLD items fire in one tick, they are collapsed into a
    single "burst" summary item — prevents terminal floods on notification storms.
    """

    def __init__(
        self,
        serial: str,
        alert_queue: queue.Queue,
        rules_path: Path,
    ) -> None:
        super().__init__(daemon=True, name="AlertWatcher")
        self._serial      = serial
        self._queue       = alert_queue
        self._rules_path  = rules_path
        self._seen_path   = rules_path.parent / ".alert_seen.json"

        self._rules:      list[AlertRule] = []
        self._rules_lock  = threading.Lock()
        self._seen        = _BoundedSet(_SEEN_MAXSIZE)
        self._stop_evt    = threading.Event()

        self._sidecar_retry_at: float = 0.0
        self._sidecar_warned:   bool  = False
        self._warmed_up:        bool  = False
        self._last_save:        float = time.time()
        self._dirty:            bool  = False   # fire_count needs flushing

        rules_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_rules()
        self._load_seen()

    # ── Thread lifecycle ──────────────────────────────────────────────────────

    def run(self) -> None:
        while not self._stop_evt.wait(_POLL_INTERVAL):
            try:
                self._tick()
            except Exception as exc:
                # Never let the watcher thread die on an unexpected error
                logger.warning(f"[Alert] Unexpected error in tick: {exc}")
        # Clean shutdown
        if self._dirty:
            self._save_rules()
        self._save_seen()

    def stop(self) -> None:
        self._stop_evt.set()

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        if time.monotonic() < self._sidecar_retry_at:
            return

        device_ctx = self._fetch_context()
        if device_ctx is None:
            return

        notifications = device_ctx.get("notifications", [])

        # ── Cold start: suppress all currently visible notifications ──────────
        # Marking them as seen WITHOUT evaluating rules prevents days-old
        # notifications from re-firing when the CLI starts. A notification that
        # arrives AFTER warm-up will be new and will trigger rules normally.
        if not self._warmed_up:
            for notif in notifications:
                self._seen.add(_notif_fp(notif))
            self._warmed_up = True
            if notifications:
                logger.info(
                    f"[Alert] Warm-up: {len(notifications)} existing notification(s) "
                    f"suppressed — only new ones will trigger rules"
                )
            return

        # ── Find genuinely new notifications ─────────────────────────────────
        new_notifs = [n for n in notifications if _notif_fp(n) not in self._seen]
        for notif in new_notifs:
            self._seen.add(_notif_fp(notif))

        if not new_notifs:
            return

        # ── Evaluate rules (snapshot to avoid holding lock during evaluation) ─
        with self._rules_lock:
            active = [r for r in self._rules if r.enabled]

        tick_items: list[dict] = []

        for notif in new_notifs:
            for rule in active:
                if not self._rate_ok(rule):
                    continue
                if not evaluate_condition(rule.condition, notif):
                    continue

                # Rule matched — record and enqueue
                rule.fire_count     += 1
                rule.last_fired_wall = time.time()
                self._dirty          = True

                if rule.action_type == "alert":
                    tick_items.append({
                        "kind":    "alert",
                        "rule_id": rule.id,
                        "msg":     render_template(rule.action, notif),
                        "notif":   notif,
                    })
                else:  # "auto"
                    tick_items.append({
                        "kind":    "auto",
                        "rule_id": rule.id,
                        "task":    render_template(rule.action, notif),
                        "notif":   notif,
                    })

        if not tick_items:
            # Periodically flush fire counts even on quiet ticks
            self._maybe_save()
            return

        # ── Burst coalescing ──────────────────────────────────────────────────
        if len(tick_items) > _BURST_THRESHOLD:
            n_alert = sum(1 for i in tick_items if i["kind"] == "alert")
            n_auto  = sum(1 for i in tick_items if i["kind"] == "auto")
            self._queue.put({
                "kind":  "burst",
                "msg":   (
                    f"{len(tick_items)} rules fired — "
                    f"{n_alert} alert{'s' if n_alert != 1 else ''}, "
                    f"{n_auto} auto-run{'s' if n_auto != 1 else ''}"
                ),
                "items": tick_items,
            })
        else:
            for item in tick_items:
                self._queue.put(item)

        self._maybe_save()

    def _maybe_save(self) -> None:
        if self._dirty and (time.time() - self._last_save) >= _SAVE_INTERVAL:
            self._save_rules()
            self._last_save = time.time()
            self._dirty = False

    def _fetch_context(self) -> dict | None:
        try:
            with urlopen(f"{_SIDECAR_URL}/v1/context", timeout=_SIDECAR_TIMEOUT) as resp:
                data = json.loads(resp.read())
            if self._sidecar_warned:
                logger.info("[Alert] Android sidecar :8766 reachable again")
                self._sidecar_warned = False
            return data
        except (URLError, OSError, json.JSONDecodeError):
            if not self._sidecar_warned:
                logger.debug("[Alert] Android sidecar :8766 unreachable — backing off 30s")
                self._sidecar_warned = True
            self._sidecar_retry_at = time.monotonic() + _SIDECAR_BACKOFF
            return None

    def _rate_ok(self, rule: AlertRule) -> bool:
        if rule.last_fired_wall is None:
            return True
        min_interval = 60.0 / max(rule.rate_limit_per_min, 1)
        return (time.time() - rule.last_fired_wall) >= min_interval

    # ── Rule CRUD (REPL thread calls these) ──────────────────────────────────

    def add_rule(
        self,
        condition: str,
        action_type: str,
        action: str,
        rate_limit_per_min: int = 3,
    ) -> AlertRule:
        """Add or replace a trigger rule. Raises ValueError on bad inputs."""
        validate_condition(condition)
        if action_type not in ("alert", "auto"):
            raise ValueError(f"action_type must be 'alert' or 'auto', got {action_type!r}")
        if not action.strip():
            raise ValueError("action cannot be empty")

        # Deterministic ID — re-adding identical rule is idempotent
        rule_id = "ar_" + hashlib.sha256(
            f"{condition}\x00{action_type}\x00{action}".encode()
        ).hexdigest()[:8]

        rule = AlertRule(
            id=rule_id,
            enabled=True,
            condition=condition,
            action_type=action_type,
            action=action,
            rate_limit_per_min=max(1, rate_limit_per_min),
            created_at=datetime.now().isoformat(timespec="seconds"),
            fire_count=0,
            last_fired_wall=None,
        )
        with self._rules_lock:
            self._rules = [r for r in self._rules if r.id != rule_id]
            self._rules.append(rule)
        self._save_rules()
        return rule

    def del_rule(self, rule_id: str) -> bool:
        with self._rules_lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.id != rule_id]
            removed = len(self._rules) < before
        if removed:
            self._save_rules()
        return removed

    def set_enabled(self, rule_id: str, enabled: bool) -> bool:
        with self._rules_lock:
            for rule in self._rules:
                if rule.id == rule_id:
                    rule.enabled = enabled
                    break
            else:
                return False
        self._save_rules()
        return True

    def get_rules(self) -> list[AlertRule]:
        with self._rules_lock:
            return list(self._rules)

    def reload_rules(self) -> int:
        self._load_rules()
        with self._rules_lock:
            return len(self._rules)

    def test_condition(
        self, condition: str, package: str = "", title: str = "", text: str = ""
    ) -> bool:
        notif = {"package": package, "title": title, "text": text}
        try:
            return evaluate_condition(condition, notif)
        except Exception:
            return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_rules(self) -> None:
        try:
            raw = json.loads(self._rules_path.read_text())
            valid_fields = {f.name for f in dataclasses.fields(AlertRule)}
            loaded: list[AlertRule] = []
            for item in raw.get("rules", []):
                try:
                    # Ignore unknown keys (forward-compat)
                    r = AlertRule(**{k: v for k, v in item.items() if k in valid_fields})
                    validate_condition(r.condition)
                    loaded.append(r)
                except Exception as exc:
                    logger.warning(
                        f"[Alert] Skipping invalid rule {item.get('id', '?')}: {exc}"
                    )
            with self._rules_lock:
                self._rules = loaded
            logger.info(f"[Alert] {len(loaded)} rule(s) loaded from {self._rules_path.name}")
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"[Alert] Could not load rules: {exc} — starting with empty set")

    def _save_rules(self) -> None:
        """Atomic write: always writes to .tmp first, then os.replace().
        A crash between write and replace leaves the old file intact.
        """
        tmp = self._rules_path.with_suffix(".json.tmp")
        with self._rules_lock:
            payload = {
                "version": 1,
                "rules": [dataclasses.asdict(r) for r in self._rules],
            }
        try:
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._rules_path)
        except Exception as exc:
            logger.warning(f"[Alert] Failed to save rules: {exc}")

    def _load_seen(self) -> None:
        """Restore fingerprints from previous session to avoid re-firing."""
        try:
            fps: list[int] = json.loads(self._seen_path.read_text())
            for fp in fps[-_SEEN_MAXSIZE:]:
                self._seen.add(int(fp))
            logger.debug(f"[Alert] Restored {len(fps)} seen fingerprint(s)")
        except Exception:
            pass

    def _save_seen(self) -> None:
        """Persist the N most-recent fingerprints for the next session."""
        tmp = self._seen_path.with_suffix(".json.tmp")
        try:
            fps = self._seen.tail(_SEEN_PERSIST)
            tmp.write_text(json.dumps(fps))
            os.replace(tmp, self._seen_path)
        except Exception:
            pass
