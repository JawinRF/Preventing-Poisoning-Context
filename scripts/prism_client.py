"""
prism_client.py — Thin HTTP client for the PRISM Shield sidecar.
Single point of contact for all PRISM Shield interactions.
"""
from __future__ import annotations
import hashlib, logging, os, threading, uuid
from collections import OrderedDict
from dataclasses import dataclass

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
        self._cache_lock = threading.Lock()

    # ── Cache primitives ──────────────────────────────────────────────────────

    def _cache_key(self, text: str, ingestion_path: str) -> tuple:
        return (hashlib.sha256(text.encode()).hexdigest(), ingestion_path)

    def _cache_get(self, key: tuple) -> InspectResult | None:
        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                self._cache.move_to_end(key)  # promote to MRU end
            return result

    def _cache_put(self, key: tuple, result: InspectResult) -> None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._cache_maxsize:
                    self._cache.popitem(last=False)  # evict LRU
            self._cache[key] = result

    # ── Result helpers ────────────────────────────────────────────────────────

    def _parse_result(self, data: dict) -> InspectResult:
        return InspectResult(
            verdict=data.get("verdict", "BLOCK"),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", "unknown"),
            layer=data.get("layer_triggered", "unknown"),
            placeholder=data.get("placeholder"),
            ticket_id=data.get("ticket_id"),
        )

    def _error_result(self, exc: Exception) -> InspectResult:
        return InspectResult(
            verdict="BLOCK" if self.fail_closed else "ALLOW",
            confidence=0.0,
            reason=f"sidecar_error: {exc}",
            layer="error",
        )

    def _build_payload(
        self,
        text: str,
        ingestion_path: str,
        source_type: str,
        source_name: str,
        entry_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        return {
            "entry_id":       entry_id or str(uuid.uuid4())[:12],
            "text":           text,
            "ingestion_path": ingestion_path,
            "source_type":    source_type,
            "source_name":    source_name,
            "session_id":     self.session_id,
            "run_id":         os.getenv("RUN_ID", "default"),
            "metadata":       metadata or {},
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def inspect(
        self,
        text: str,
        ingestion_path: str,
        source_type: str = "agent_input",
        source_name: str = "prism_agent",
        entry_id: str | None = None,
        metadata: dict | None = None,
    ) -> InspectResult:
        """Send one text through the PRISM pipeline."""
        key = self._cache_key(text, ingestion_path)
        hit = self._cache_get(key)
        if hit is not None:
            return hit

        try:
            resp = requests.post(
                f"{self.url}/v1/inspect",
                json=self._build_payload(text, ingestion_path, source_type,
                                         source_name, entry_id, metadata),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            result = self._parse_result(resp.json())
        except Exception as e:
            logger.warning(f"PRISM sidecar error: {e}")
            result = self._error_result(e)

        self._cache_put(key, result)
        if not result.allowed:
            logger.warning(f"PRISM {result.verdict}: [{ingestion_path}] {result.reason}")
        return result

    def inspect_batch(
        self,
        texts: list[str],
        ingestion_path: str,
        source_type: str = "agent_input",
        source_name: str = "prism_agent",
    ) -> list[InspectResult]:
        """
        Scan N texts in a single HTTP round-trip.

        Algorithm:
          1. Cache-check every text                         O(N)
          2. Group cache-misses by content hash             O(M), M ≤ N
             identical texts share one sidecar call
          3. Single POST /v1/inspect/batch for M unique texts
          4. Fan results back to all N positions            O(N)
          5. Cache-put all M results                        O(M)

        Net cost: O(N) time, 1 HTTP call (was N).
        """
        if not texts:
            return []

        # Phase 1 — cache check; compute keys once for reuse in phase 2
        keys    = [self._cache_key(t, ingestion_path) for t in texts]
        results: list[InspectResult | None] = [self._cache_get(k) for k in keys]

        # Phase 2 — group misses by content key (dedup identical texts)
        # miss_groups: key → [positions in results[] that share this content]
        miss_groups: dict[tuple, list[int]] = {}
        for i, (key, hit) in enumerate(zip(keys, results)):
            if hit is None:
                miss_groups.setdefault(key, []).append(i)

        if not miss_groups:
            return results  # type: ignore[return-value]  # all cache hits

        # Phase 3 — single HTTP call for all unique uncached texts
        unique_keys  = list(miss_groups)
        unique_texts = [texts[miss_groups[k][0]] for k in unique_keys]

        items = [
            self._build_payload(text, ingestion_path, source_type, source_name)
            for text in unique_texts
        ]

        try:
            resp = requests.post(
                f"{self.url}/v1/inspect/batch",
                json={"items": items},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            scan_results = [self._parse_result(d) for d in resp.json().get("results", [])]
            # Pad with error results if sidecar returned fewer than expected
            if len(scan_results) < len(unique_keys):
                pad = self._error_result(RuntimeError("short batch response"))
                scan_results += [pad] * (len(unique_keys) - len(scan_results))
        except Exception as e:
            logger.warning(f"PRISM batch sidecar error: {e}")
            err = self._error_result(e)
            scan_results = [err] * len(unique_keys)

        # Phase 4+5 — fan out + cache
        for key, scan_result in zip(unique_keys, scan_results):
            self._cache_put(key, scan_result)
            if not scan_result.allowed:
                logger.warning(
                    f"PRISM {scan_result.verdict}: [{ingestion_path}] {scan_result.reason}"
                )
            for pos in miss_groups[key]:
                results[pos] = scan_result

        return results  # type: ignore[return-value]

    def filter_batch(
        self,
        items: list[dict],
        ingestion_path: str,
        text_key: str = "text",
        **kwargs,
    ) -> tuple[list[dict], list[dict]]:
        """Filter a list of dicts via a single inspect_batch call."""
        texts = [str(item.get(text_key) or "") for item in items]
        scan  = self.inspect_batch(texts, ingestion_path)
        allowed, blocked = [], []
        for item, result in zip(items, scan):
            (allowed if result.allowed else blocked).append(item)
        return allowed, blocked

    def is_allowed(self, text: str, ingestion_path: str, **kwargs) -> bool:
        return self.inspect(text, ingestion_path, **kwargs).allowed

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/health", timeout=2)
            return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"PRISM sidecar health check failed: {exc}")
            return False


class NullPrismClient(PrismClient):
    """No-op PRISM client — allows everything, never contacts the sidecar.
    Used for undefended A/B demo runs."""

    def __init__(self):
        self.session_id = "null"
        self._cache = {}
        self._cache_lock = threading.Lock()

    def inspect(self, text, ingestion_path, **kwargs) -> InspectResult:
        return InspectResult(verdict="ALLOW", confidence=0.0,
                             reason="prism_disabled", layer="none")

    def inspect_batch(self, texts, ingestion_path, **kwargs) -> list[InspectResult]:
        allow = InspectResult(verdict="ALLOW", confidence=0.0,
                              reason="prism_disabled", layer="none")
        return [allow] * len(texts)

    def health(self) -> bool:
        return True
