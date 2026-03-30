"""
window_context_reader.py
------------------------
Reads the current ScreenContext from the Android WindowManager bridge service.
The Android service writes JSON-serialized ScreenContext objects to a TCP socket.

Architecture: The Android-side service listens on a TCP port on the device.
The host connects via ADB port forwarding:
    adb forward tcp:PRISM_WCR_PORT tcp:PRISM_WCR_PORT

Set PRISM_WCR_HOST / PRISM_WCR_PORT env vars to override defaults.
Defaults: 127.0.0.1:18765 (localhost via ADB forward).

The Python pipeline reads the LATEST cached context synchronously.
This is a non-blocking read — if no context is available, NULL_CONTEXT is returned.

In unit tests and CI (no Android service running), always returns NULL_CONTEXT.
"""

import json
import os
import socket
import threading
from .screen_context import ScreenContext, NULL_CONTEXT
from .screen_type_classifier import enrich

# TCP connection to Android service via ADB port forwarding
WCR_HOST = os.getenv("PRISM_WCR_HOST", "127.0.0.1")
WCR_PORT = int(os.getenv("PRISM_WCR_PORT", "18765"))
READ_TIMEOUT_MS = 2.0   # Hard cap: must not stall the <2ms pipeline


class WindowContextReader:
    """
    Maintains a background thread that listens on a TCP socket (forwarded
    from Android via `adb forward`) and caches the most recent ScreenContext.
    The pipeline calls get_context() which returns the cached value instantly
    — zero blocking in the hot path.
    """

    def __init__(self, host: str = WCR_HOST, port: int = WCR_PORT):
        self._host = host
        self._port = port
        self._lock         = threading.Lock()
        self._latest       = NULL_CONTEXT
        self._running      = False
        self._thread: threading.Thread | None = None

    def start(self):
        """Start background listener thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="WindowContextReader"
        )
        self._thread.start()

    def stop(self):
        self._running = False

    def get_context(self) -> ScreenContext:
        """Non-blocking. Returns the most recently received ScreenContext."""
        with self._lock:
            return self._latest

    def _listen_loop(self):
        """
        Background thread: connects to the Android service via TCP
        (forwarded through `adb forward tcp:PORT tcp:PORT`) and reads
        new ScreenContext JSON objects as they arrive.
        Each message is a newline-terminated JSON object.
        """
        while self._running:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(READ_TIMEOUT_MS / 1000.0)
                sock.connect((self._host, self._port))
                buf = ""
                while self._running:
                    try:
                        chunk = sock.recv(4096).decode("utf-8", errors="replace")
                        if not chunk:
                            break
                        buf += chunk
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            line = line.strip()
                            if line:
                                self._process_message(line)
                    except socket.timeout:
                        continue
                sock.close()
            except (ConnectionRefusedError, OSError):
                # Android service not running (CI, unit test, or ADB forward not set up)
                # Stay in loop, retry every second
                import time
                time.sleep(1.0)
            except Exception as e:
                import logging
                logging.getLogger("PrismShield.WindowCtx").warning(
                    f"Socket listener error (retrying in 0.5s): {e}"
                )
                import time
                time.sleep(0.5)

    def _process_message(self, json_str: str):
        try:
            d = json.loads(json_str)
            ctx = ScreenContext.from_dict(d)
            enrich(ctx)   # fills in screen_type deterministically
            with self._lock:
                self._latest = ctx
        except Exception as e:
            import logging
            logging.getLogger("PrismShield.WindowCtx").warning(
                f"Malformed JSON from window service (keeping previous context): {e}"
            )


# Module-level singleton — one reader per process
_reader = WindowContextReader()


def start_reader():
    """Call once at service startup."""
    _reader.start()


def get_current_context() -> ScreenContext:
    """Call from A-MemGuard at inference time. Always returns instantly."""
    return _reader.get_context()
