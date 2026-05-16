#!/usr/bin/env python3
"""Tests for alert_watcher.py — condition DSL, templates, BoundedSet, AlertWatcher CRUD."""
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from alert_watcher import (
    AlertRule,
    AlertWatcher,
    _BoundedSet,
    _notif_fp,
    evaluate_condition,
    render_template,
    validate_condition,
    _BURST_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def notif(pkg="com.example.app", title="Hello", text="World") -> dict:
    return {"package": pkg, "title": title, "text": text}


def make_watcher(tmp_path) -> AlertWatcher:
    import queue
    q = queue.Queue()
    rules_path = tmp_path / "alert_rules.json"
    w = AlertWatcher(serial="emulator-5554", alert_queue=q, rules_path=rules_path)
    return w


# ── 1. Condition DSL ──────────────────────────────────────────────────────────

class TestConditionDSL:
    def test_text_match(self):
        n = notif(text="Your OTP is 123456")
        assert evaluate_condition("text:OTP", n) is True

    def test_text_no_match(self):
        n = notif(text="Hello World")
        assert evaluate_condition("text:OTP", n) is False

    def test_and_both_true(self):
        n = notif(pkg="com.google.android.gm", text="invoice")
        assert evaluate_condition("package:com.google AND text:invoice", n) is True

    def test_and_one_false(self):
        n = notif(pkg="com.google.android.gm", text="hello")
        assert evaluate_condition("package:com.google AND text:invoice", n) is False

    def test_or_first_true(self):
        n = notif(title="Alice")
        assert evaluate_condition("title:Alice OR title:Bob", n) is True

    def test_or_second_true(self):
        n = notif(title="Bob message")
        assert evaluate_condition("title:Alice OR title:Bob", n) is True

    def test_or_both_false(self):
        n = notif(title="Charlie")
        assert evaluate_condition("title:Alice OR title:Bob", n) is False

    def test_not_inverts(self):
        n = notif(text="Buy now! Special offer!")
        assert evaluate_condition("NOT text:advertisement", n) is True
        assert evaluate_condition("NOT text:offer", n) is False

    def test_case_insensitive(self):
        n = notif(text="Your OTP code is 999")
        assert evaluate_condition("text:otp", n) is True
        assert evaluate_condition("TEXT:OTP", n) is True

    def test_bare_word_any_field(self):
        n = notif(title="Gmail", text="New message")
        assert evaluate_condition("Gmail", n) is True
        assert evaluate_condition("missing_keyword", n) is False

    def test_any_field(self):
        n = notif(pkg="com.test", title="Hi", text="code 9999")
        assert evaluate_condition("any:code", n) is True
        assert evaluate_condition("any:absent", n) is False

    def test_operator_precedence_and_over_or(self):
        # "A AND B OR C" should parse as "(A AND B) OR C"
        n_both = notif(pkg="com.x", text="promo")
        n_c    = notif(title="urgent")
        n_none = notif(title="quiet")
        cond = "package:com.x AND text:promo OR title:urgent"
        assert evaluate_condition(cond, n_both) is True
        assert evaluate_condition(cond, n_c) is True
        assert evaluate_condition(cond, n_none) is False

    def test_validate_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_condition("")

    def test_validate_whitespace_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_condition("   ")

    def test_unknown_field_no_crash(self):
        # Unknown fields are treated as no-match, never crash
        n = notif(text="hello")
        assert evaluate_condition("unknownfield:hello", n) is False


# ── 2. Template injection safety ──────────────────────────────────────────────

class TestTemplateInjection:
    def test_format_spec_not_evaluated(self):
        # If str.format() were used, {__class__} would expand to "<class 'str'>"
        n = notif(text="{__class__}", title="{__class__}")
        out = render_template("msg: {text} / {title}", n)
        assert "<class" not in out
        assert "{__class__}" in out   # literal pass-through

    def test_substitution_works(self):
        n = notif(pkg="com.google.android.gm", title="Meeting", text="5pm tomorrow")
        out = render_template("From {app}: [{title}] {text}", n)
        assert "Gm" in out or "gmail" in out.lower() or "com.google" in out
        assert "Meeting" in out
        assert "5pm tomorrow" in out

    def test_empty_template(self):
        n = notif()
        assert render_template("", n) == ""

    def test_no_placeholders(self):
        n = notif()
        assert render_template("static string", n) == "static string"


# ── 3. BoundedSet ─────────────────────────────────────────────────────────────

class TestBoundedSet:
    def test_add_and_contains(self):
        s = _BoundedSet(5)
        s.add(1)
        assert 1 in s
        assert 2 not in s

    def test_fifo_eviction(self):
        s = _BoundedSet(3)
        for i in range(4):
            s.add(i)
        # 0 should have been evicted (FIFO)
        assert 0 not in s
        assert 1 in s
        assert 2 in s
        assert 3 in s

    def test_duplicate_no_eviction(self):
        s = _BoundedSet(3)
        s.add(1); s.add(2); s.add(1)   # re-adding 1 must not evict
        assert len(s) == 2

    def test_tail(self):
        s = _BoundedSet(10)
        for i in range(5):
            s.add(i)
        tail = s.tail(3)
        assert tail == [2, 3, 4]

    def test_len(self):
        s = _BoundedSet(10)
        s.add(1); s.add(2)
        assert len(s) == 2


# ── 4. Fingerprint uniqueness ─────────────────────────────────────────────────

class TestFingerprint:
    def test_same_text_diff_package_different_fp(self):
        n1 = notif(pkg="com.a", text="OTP 1234")
        n2 = notif(pkg="com.b", text="OTP 1234")
        assert _notif_fp(n1) != _notif_fp(n2)

    def test_same_notif_stable(self):
        n = notif(pkg="com.x", title="T", text="U")
        assert _notif_fp(n) == _notif_fp(n)

    def test_different_text_different_fp(self):
        n1 = notif(text="hello")
        n2 = notif(text="world")
        assert _notif_fp(n1) != _notif_fp(n2)


# ── 5. Rule CRUD + idempotency ────────────────────────────────────────────────

class TestRuleCRUD:
    def test_add_rule(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:OTP", "alert", "OTP detected: {text}")
        assert r.enabled is True
        assert r.action_type == "alert"
        assert len(w.get_rules()) == 1

    def test_add_idempotent(self, tmp_path):
        w = make_watcher(tmp_path)
        r1 = w.add_rule("text:OTP", "alert", "OTP: {text}")
        r2 = w.add_rule("text:OTP", "alert", "OTP: {text}")
        assert r1.id == r2.id
        assert len(w.get_rules()) == 1

    def test_del_rule(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:hello", "alert", "hi")
        assert w.del_rule(r.id) is True
        assert len(w.get_rules()) == 0

    def test_del_nonexistent(self, tmp_path):
        w = make_watcher(tmp_path)
        assert w.del_rule("ar_notexist") is False

    def test_pause_resume(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:test", "alert", "testing")
        assert w.set_enabled(r.id, False) is True
        assert w.get_rules()[0].enabled is False
        assert w.set_enabled(r.id, True) is True
        assert w.get_rules()[0].enabled is True

    def test_invalid_condition_raises(self, tmp_path):
        w = make_watcher(tmp_path)
        with pytest.raises(ValueError):
            w.add_rule("", "alert", "something")

    def test_invalid_action_type_raises(self, tmp_path):
        w = make_watcher(tmp_path)
        with pytest.raises(ValueError, match="action_type"):
            w.add_rule("text:hi", "badtype", "something")


# ── 6. test_condition helper ──────────────────────────────────────────────────

class TestTestCondition:
    def test_match(self, tmp_path):
        w = make_watcher(tmp_path)
        assert w.test_condition("text:OTP", text="Your OTP is 123456") is True

    def test_no_match(self, tmp_path):
        w = make_watcher(tmp_path)
        assert w.test_condition("text:OTP", text="Normal message") is False

    def test_bad_condition_returns_false(self, tmp_path):
        w = make_watcher(tmp_path)
        # Empty condition would raise in validate but test_condition catches it
        assert w.test_condition("", text="anything") is False


# ── 7. Rate limiting ──────────────────────────────────────────────────────────

class TestRateLimiting:
    def test_rate_exceeded(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:x", "alert", "x", rate_limit_per_min=60)
        r.last_fired_wall = time.time()   # just fired
        assert w._rate_ok(r) is False

    def test_rate_ok_after_interval(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:x", "alert", "x", rate_limit_per_min=60)
        r.last_fired_wall = time.time() - 2.0   # 2s ago; interval = 1s
        assert w._rate_ok(r) is True

    def test_rate_ok_never_fired(self, tmp_path):
        w = make_watcher(tmp_path)
        r = w.add_rule("text:x", "alert", "x", rate_limit_per_min=3)
        assert w._rate_ok(r) is True   # last_fired_wall is None


# ── 8. Atomic JSON write ──────────────────────────────────────────────────────

class TestAtomicWrite:
    def test_rules_file_written_atomically(self, tmp_path):
        w = make_watcher(tmp_path)
        w.add_rule("text:test", "alert", "test alert")
        rules_path = tmp_path / "alert_rules.json"
        assert rules_path.exists()
        tmp_path2 = rules_path.with_suffix(".json.tmp")
        # The .tmp file should NOT remain after a successful write
        assert not tmp_path2.exists()

    def test_rules_file_valid_json(self, tmp_path):
        w = make_watcher(tmp_path)
        w.add_rule("text:json", "alert", "valid json")
        data = json.loads((tmp_path / "alert_rules.json").read_text())
        assert "rules" in data
        assert data["version"] == 1


# ── 9. Seen persistence ───────────────────────────────────────────────────────

class TestSeenPersistence:
    def test_save_and_load_seen(self, tmp_path):
        import queue as _q
        q = _q.Queue()
        rules_path = tmp_path / "alert_rules.json"

        # Create watcher, add some fingerprints
        w1 = AlertWatcher(serial="", alert_queue=q, rules_path=rules_path)
        w1._seen.add(111)
        w1._seen.add(222)
        w1._save_seen()

        # New watcher should restore them
        w2 = AlertWatcher(serial="", alert_queue=q, rules_path=rules_path)
        assert 111 in w2._seen
        assert 222 in w2._seen


# ── 10. Burst threshold ───────────────────────────────────────────────────────

class TestBurstCoalescing:
    def test_burst_fires_single_summary(self, tmp_path):
        import queue as _q
        q = _q.Queue()
        rules_path = tmp_path / "alert_rules.json"
        w = AlertWatcher(serial="", alert_queue=q, rules_path=rules_path)

        # Add BURST_THRESHOLD+2 distinct rules (different actions → different IDs)
        # all matching the same notification.  Each rule has its own last_fired_wall
        # (starts None), so all fire in the same tick without rate-limiting each other.
        for i in range(_BURST_THRESHOLD + 2):
            w.add_rule("text:trigger", "alert", f"rule {i} fired: {{text}}")

        w._warmed_up = True
        single = notif(text="trigger message here")

        w._fetch_context = lambda: {"notifications": [single]}
        w._tick()

        items = []
        while not q.empty():
            items.append(q.get_nowait())

        assert len(items) == 1, f"expected 1 burst item, got {len(items)}: {items}"
        assert items[0]["kind"] == "burst"
        assert "rules fired" in items[0]["msg"]

    def test_no_burst_below_threshold(self, tmp_path):
        import queue as _q
        q = _q.Queue()
        rules_path = tmp_path / "alert_rules.json"
        w = AlertWatcher(serial="", alert_queue=q, rules_path=rules_path)

        # Add exactly BURST_THRESHOLD distinct rules — none exceed threshold
        for i in range(_BURST_THRESHOLD):
            w.add_rule("text:trigger", "alert", f"rule {i}: {{text}}")

        w._warmed_up = True
        single = notif(text="trigger message here")

        w._fetch_context = lambda: {"notifications": [single]}
        w._tick()

        items = []
        while not q.empty():
            items.append(q.get_nowait())

        # Exactly at threshold → individual items (coalescing is strictly >threshold)
        assert len(items) == _BURST_THRESHOLD
        assert all(i["kind"] == "alert" for i in items)
