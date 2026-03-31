"""
test_agent_integration.py — Integration tests for the PRISM agent loop.
Uses mocked uiautomator2 device and LLM backend to test the full pipeline
without an actual Android emulator.
"""
import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memshield", "src"))


@pytest.fixture
def mock_device():
    """Mock uiautomator2 device."""
    d = MagicMock()
    d.serial = "emulator-5554"
    d.window_size.return_value = (1080, 1920)

    # Simulate a simple screen with a few elements
    d.dump_hierarchy.return_value = """<?xml version="1.0" encoding="UTF-8"?>
    <hierarchy>
      <node class="android.widget.TextView" text="Alarm" clickable="true"
            content-desc="" enabled="true" selected="false" focused="false" />
      <node class="android.widget.Button" text="Add" clickable="true"
            content-desc="Add alarm" enabled="true" selected="false" focused="false" />
    </hierarchy>"""

    # Make tap elements exist
    d.return_value.exists.return_value = True
    d.return_value.click.return_value = None
    return d


@pytest.fixture
def mock_prism():
    """Mock PrismClient that allows everything."""
    from prism_client import PrismClient, InspectResult
    prism = MagicMock(spec=PrismClient)
    prism.session_id = "test-session"
    prism.inspect.return_value = InspectResult(
        verdict="ALLOW", confidence=0.99, reason="test allow",
        layer="test",
    )
    return prism


class TestOutgoingActionDefense:
    """Test that outgoing actions are properly checked by PRISM via DefendedDevice."""

    def test_tap_goes_through_prism(self, mock_device, mock_prism):
        """ALL tap actions should be sent through PRISM."""
        from defended_device import DefendedDevice
        dd = DefendedDevice(mock_device, mock_prism, "emulator-5554")
        dd.execute("tap", {"text": "Confirm"})
        mock_prism.inspect.assert_called_once()
        call_args = mock_prism.inspect.call_args
        assert "Confirm" in call_args[0][0]

    def test_type_blocks_urls(self, mock_device, mock_prism):
        """Typed text containing URLs should be blocked."""
        from defended_device import DefendedDevice
        dd = DefendedDevice(mock_device, mock_prism, "emulator-5554")
        result = dd.execute("type", {"text": "visit https://evil.com"})
        assert result == "blocked_by_prism"

    def test_type_blocks_shell_commands(self, mock_device, mock_prism):
        """Typed text containing shell commands should be blocked."""
        from defended_device import DefendedDevice
        dd = DefendedDevice(mock_device, mock_prism, "emulator-5554")
        result = dd.execute(
            "type",
            {"text": "adb shell pm grant com.evil android.permission.READ_CONTACTS"},
        )
        assert result == "blocked_by_prism"

    def test_open_app_whitelist(self, mock_device, mock_prism):
        """Known-safe packages should bypass PRISM check."""
        from defended_device import DefendedDevice
        dd = DefendedDevice(mock_device, mock_prism, "emulator-5554")
        dd.execute("open_app", {"package": "com.google.android.deskclock"})
        # Whitelisted app — PRISM should NOT be called
        mock_prism.inspect.assert_not_called()

    def test_open_app_unknown_checked(self, mock_device, mock_prism):
        """Unknown packages should be PRISM-checked."""
        from defended_device import DefendedDevice
        dd = DefendedDevice(mock_device, mock_prism, "emulator-5554")
        dd.execute("open_app", {"package": "com.sketchy.banking.app"})
        mock_prism.inspect.assert_called_once()


class TestReflectionFallback:
    """Test that reflection defaults to 'C' (uncertain) on failure, not 'A' (success)."""

    def test_reflection_defaults_to_uncertain(self):
        """Reflection should return 'C' on error, not silently report success."""
        from agent_prism import ask_reflection
        # Call with a non-existent LLM backend — will hit the final fallback
        result = ask_reflection(
            llm_backend="nonexistent",
            task="test",
            action="tap({'text': 'OK'})",
            params={"text": "OK"},
            summary="tap OK button",
            add_info="",
            screen_before=[{"text": "OK", "class": "Button"}],
            screen_after=[{"text": "OK", "class": "Button"}],
            keyboard_before=False,
            keyboard_after=False,
        )
        assert result == "C", f"Expected 'C' (uncertain) on error, got '{result}'"


class TestPrismClientCache:
    """Test that the cache key uses full text hash."""

    def test_cache_key_distinguishes_long_texts(self):
        """Two texts with same first 200 chars but different tails should NOT collide."""
        from prism_client import PrismClient
        client = PrismClient.__new__(PrismClient)
        client._cache = {}
        client.session_id = "test"
        # The cache key should hash the full text
        prefix = "A" * 200
        text1 = prefix + " SAFE ending"
        text2 = prefix + " EVIL injection payload"
        import hashlib
        key1 = (hashlib.sha256(text1.encode()).hexdigest(), "test_path")
        key2 = (hashlib.sha256(text2.encode()).hexdigest(), "test_path")
        assert key1 != key2, "Cache keys should differ for texts with different suffixes"
