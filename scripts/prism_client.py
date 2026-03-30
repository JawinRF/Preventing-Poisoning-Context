"""
prism_client.py — Thin HTTP client for the PRISM Shield sidecar.
Single point of contact for all PRISM Shield interactions.
"""
from __future__ import annotations
import hashlib, logging, os, time, uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)

SIDECAR_URL = os.getenv("PRISM_SIDECAR_URL", "http://localhost:8765")


@dataclass
class InspectResult:
    verdict: str          # ALLOW | BLOCK | QUARANTINE
    confidence: float
    reason: str
    layer: str            # Layer1-Heuristics | Layer2-LocalLLM | Layer3-DeBERTa | ...
    placeholder: str | None = None
    ticket_id: str | None = None

    @property
    def allowed(self) -> bool:
        return self.verdict == "ALLOW"


class PrismClient:
    """HTTP client for the PRISM Shield sidecar (/v1/inspect)."""

    def __init__(
        self,
        sidecar_url: str = SIDECAR_URL,
        timeout: float = 15.0,
        fail_closed: bool = True,
        session_id: str = "default",
    ):
        self.url = sidecar_url.rstrip("/")
        self.timeout = timeout
        self.fail_closed = fail_closed
        self.session_id = session_id
        self._cache: OrderedDict[tuple, InspectResult] = OrderedDict()
        self._cache_maxsize = 500

    def inspect(
        self,
        text: str,
        ingestion_path: str,
        source_type: str = "agent_input",
        source_name: str = "prism_agent",
        entry_id: str | None = None,
        metadata: dict | None = None,
    ) -> InspectResult:
        """Send text through the PRISM pipeline. Returns InspectResult."""
        cache_key = (hashlib.sha256(text.encode()).hexdigest(), ingestion_path)
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = {
            "entry_id": entry_id or str(uuid.uuid4())[:12],
            "text": text,
            "ingestion_path": ingestion_path,
            "source_type": source_type,
            "source_name": source_name,
            "session_id": self.session_id,
            "run_id": os.getenv("RUN_ID", "default"),
            "metadata": metadata or {},
        }

        try:
            resp = requests.post(
                f"{self.url}/v1/inspect",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            result = InspectResult(
                verdict=data.get("verdict", "BLOCK"),
                confidence=data.get("confidence", 0.0),
                reason=data.get("reason", "unknown"),
                layer=data.get("layer_triggered", "unknown"),
                placeholder=data.get("placeholder"),
                ticket_id=data.get("ticket_id"),
            )
        except Exception as e:
            logger.warning(f"PRISM sidecar error: {e}")
            if self.fail_closed:
                result = InspectResult(
                    verdict="BLOCK", confidence=0.0,
                    reason=f"sidecar_error: {e}", layer="error",
                )
            else:
                result = InspectResult(
                    verdict="ALLOW", confidence=0.0,
                    reason=f"sidecar_error (fail-open): {e}", layer="error",
                )

        # Cache (LRU bounded)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
        else:
            if len(self._cache) >= self._cache_maxsize:
                self._cache.popitem(last=False)  # Evict oldest
        self._cache[cache_key] = result

        if not result.allowed:
            logger.warning(f"PRISM {result.verdict}: [{ingestion_path}] {result.reason}")

        return result

    def is_allowed(self, text: str, ingestion_path: str, **kwargs) -> bool:
        """Convenience: True only if verdict == ALLOW."""
        return self.inspect(text, ingestion_path, **kwargs).allowed

    def filter_batch(
        self,
        items: list[dict],
        ingestion_path: str,
        text_key: str = "text",
        **kwargs,
    ) -> tuple[list[dict], list[dict]]:
        """Filter a list of dicts. Returns (allowed, blocked)."""
        allowed, blocked = [], []
        for item in items:
            text = item.get(text_key, "")
            if not text:
                allowed.append(item)
                continue
            result = self.inspect(text, ingestion_path, **kwargs)
            if result.allowed:
                allowed.append(item)
            else:
                blocked.append(item)
        return allowed, blocked

    def poll_quarantine(self, ticket_id: str, timeout_s: float = 5.0,
                        poll_interval_s: float = 0.5) -> InspectResult:
        """
        Poll the sidecar for a quarantine ticket's resolution.
        The VLM runs asynchronously — this waits briefly for its verdict.
        Returns BLOCK if the ticket isn't resolved within timeout.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                resp = requests.get(
                    f"{self.url}/v1/ticket/{ticket_id}",
                    timeout=2,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "PENDING")
                    if status != "PENDING":
                        return InspectResult(
                            verdict=status,
                            confidence=data.get("confidence", 0.0),
                            reason=data.get("reason", "quarantine_resolved"),
                            layer=data.get("layer_triggered", "VLM"),
                        )
            except Exception:
                pass
            time.sleep(poll_interval_s)

        logger.warning(f"Quarantine ticket {ticket_id} not resolved within {timeout_s}s — blocking")
        return InspectResult(
            verdict="BLOCK", confidence=1.0,
            reason="quarantine_timeout", layer="VLM",
        )

    def health(self) -> bool:
        """Check if the sidecar is alive."""
        try:
            resp = requests.get(f"{self.url}/health", timeout=2)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"PRISM sidecar health check failed: {exc}")
            return False


class NullPrismClient(PrismClient):
    """No-op PRISM client that allows everything without contacting the sidecar.
    Used for undefended A/B demo runs where we want zero filtering."""

    def __init__(self):
        self.session_id = "null"
        self._cache = {}

    def inspect(self, text, ingestion_path, **kwargs):
        return InspectResult(
            verdict="ALLOW", confidence=0.0,
            reason="prism_disabled", layer="none",
        )

    def health(self):
        return True
