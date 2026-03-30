from __future__ import annotations

import collections
import concurrent.futures
import json
import logging
import os
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import asdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prism.sidecar")

try:
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
MEMSHIELD_SRC_DIR = SCRIPTS_DIR.parent / "memshield" / "src"
if str(MEMSHIELD_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MEMSHIELD_SRC_DIR))

from openclaw_adapter.audit import log_audit
from openclaw_adapter.models import InspectRequest, InspectResponse
from openclaw_adapter.quarantine_store import load_ticket
from openclaw_adapter.source_mapper import map_ingestion_path
from memshield import FailurePolicy, MemShield, ShieldConfig
from prism_shield import MemoryEntry, PrismShield
from prism_shield.ui_extractor import UIExtractor
from prism_shield.window_context_reader import get_current_context


BLOCK_PLACEHOLDER = "[PRISM_BLOCKED untrusted context removed before model assembly]"
QUARANTINE_PLACEHOLDER = "[PRISM_QUARANTINED suspicious context pending verification]"
DEFAULT_HOST = os.getenv("PRISM_SIDECAR_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PRISM_SIDECAR_PORT", "8765"))
ENABLE_MEMSHIELD_RAG = os.getenv("PRISM_ENABLE_MEMSHIELD_RAG", "1").lower() not in {"0", "false", "no"}
UI_EXTRACTOR = UIExtractor()
_EXECUTOR = ThreadPoolExecutor(max_workers=8)

# ── Rate Limiter ─────────────────────────────────────────────────────────────

_RATE_LIMIT_MAX = int(os.getenv("PRISM_RATE_LIMIT", "100"))  # requests per second per session
_rate_windows: dict[str, collections.deque] = {}


def _check_rate_limit(session_id: str) -> bool:
    """Return True if request is allowed, False if rate limited."""
    now = _time.monotonic()
    window = _rate_windows.setdefault(session_id, collections.deque())
    # Purge entries older than 1 second
    while window and window[0] < now - 1.0:
        window.popleft()
    if len(window) >= _RATE_LIMIT_MAX:
        return False
    window.append(now)
    return True


def _model_dump(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[no-any-return]
    return model.dict()  # type: ignore[no-any-return]


def _validate_model(model_cls, payload: dict):
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(payload)
    return model_cls.parse_obj(payload)


def _require_secret_value(provided: str | None) -> None:
    expected = os.getenv("PRISM_SIDECAR_SECRET")
    if expected and provided != expected:
        raise HTTPException(status_code=403, detail="forbidden")


@lru_cache(maxsize=1)
def get_pipeline() -> PrismShield:
    return PrismShield()


@lru_cache(maxsize=1)
def get_memshield() -> MemShield:
    return MemShield(
        config=ShieldConfig(
            confidence_threshold=0.5,
            failure_policy=FailurePolicy.BLOCK,
            enable_provenance=True,
            enable_normalization=True,
            enable_ml_layers=True,
        ),
    )


def _build_audit(request: InspectRequest, ingestion_path: str, ticket_id: str | None = None) -> dict:
    return {
        "path": ingestion_path,
        "source_type": request.source_type,
        "source_name": request.source_name,
        "session_id": request.session_id,
        "run_id": request.run_id,
        "ticket_id": ticket_id,
    }


def _inspect_rag_store(text: str) -> tuple[str, float, str, str]:
    shield = get_memshield()
    result = shield.scan_chunk(text)
    layer = f"MemShield/{result.layer_triggered}" if result.layer_triggered else "MemShield"
    return (result.verdict, result.confidence, result.reason, layer)


def handle_inspect(request: InspectRequest) -> InspectResponse:
    # Rate limit per session
    sid = request.session_id or "anonymous"
    if not _check_rate_limit(sid):
        logger.warning(f"Rate limited session {sid}")
        return InspectResponse(
            verdict="BLOCK",
            confidence=1.0,
            reason="Rate limited — too many requests per second",
            ingestion_path=request.ingestion_path or "unknown",
        )

    ingestion_path = request.ingestion_path or map_ingestion_path(
        request.source_type,
        request.source_name,
        request.metadata,
    )

    try:
        entry_text = request.text
        if ingestion_path == "ui_accessibility":
            entry_text = UI_EXTRACTOR.extract(request.text)

        if ingestion_path == "rag_store" and ENABLE_MEMSHIELD_RAG:
            verdict, confidence, reason, layer_triggered = _inspect_rag_store(entry_text)
            ticket_id = None
            if verdict == "BLOCK":
                placeholder = BLOCK_PLACEHOLDER
            elif verdict == "QUARANTINE":
                placeholder = QUARANTINE_PLACEHOLDER
            else:
                placeholder = None
            audit = _build_audit(request, ingestion_path, ticket_id)
            log_audit(
                request.entry_id,
                verdict,
                ingestion_path,
                request.source_type,
                request.session_id,
                request.run_id,
                reason=reason,
                source_name=request.source_name,
                ticket_id=ticket_id,
            )
            return InspectResponse(
                verdict=verdict,
                confidence=confidence,
                reason=reason,
                layer_triggered=layer_triggered,
                normalized_text=entry_text,
                ticket_id=ticket_id,
                placeholder=placeholder,
                audit=audit,
            )

        entry = MemoryEntry(
            id=request.entry_id,
            text=entry_text,
            ingestion_path=ingestion_path,
            metadata=dict(request.metadata),
        )
        pipeline = get_pipeline()
        
        # Debug logging and timeout wrapper
        logger.info(f"evaluate_sync start: path={request.ingestion_path} text_len={len(request.text)}")
        try:
            future = _EXECUTOR.submit(pipeline.evaluate_sync, entry)
            result = future.result(timeout=15)
            logger.info(f"evaluate_sync complete: verdict={result.verdict}")
        except FuturesTimeoutError:
            logger.warning("evaluate_sync timeout (15s exceeded)")
            audit = _build_audit(request, ingestion_path)
            return InspectResponse(
                verdict="BLOCK",
                confidence=0.99,
                reason="pipeline_timeout",
                layer_triggered="timeout",
                normalized_text=request.text[:100],
                ticket_id=None,
                placeholder="[PRISM_BLOCKED pipeline timeout]",
                audit=audit,
            )

        placeholder = None
        ticket_id = result.ticket_id
        if result.verdict == "QUARANTINE":
            placeholder = QUARANTINE_PLACEHOLDER
            screenshot_path = request.metadata.get("screenshot_path") or request.metadata.get("screenshot")
            screen_context = request.metadata.get("screen_context")
            if not screen_context:
                screen_context = get_current_context().to_dict()
            pipeline.submit_quarantine(ticket_id, screenshot_path, screen_context)
        elif result.verdict == "BLOCK":
            placeholder = BLOCK_PLACEHOLDER

        audit = _build_audit(request, ingestion_path, ticket_id)
        log_audit(
            request.entry_id,
            result.verdict,
            ingestion_path,
            request.source_type,
            request.session_id,
            request.run_id,
            reason=result.reason,
            source_name=request.source_name,
            ticket_id=ticket_id,
        )

        return InspectResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            reason=result.reason,
            layer_triggered=result.layer_triggered,
            normalized_text=result.normalized_text or "",
            ticket_id=ticket_id,
            placeholder=placeholder,
            audit=audit,
        )
    except HTTPException:
        raise
    except Exception:
        audit = _build_audit(request, ingestion_path)
        log_audit(
            request.entry_id,
            "BLOCK",
            ingestion_path,
            request.source_type,
            request.session_id,
            request.run_id,
            reason="sidecar_error",
            source_name=request.source_name,
        )
        return InspectResponse(
            verdict="BLOCK",
            confidence=1.0,
            reason="sidecar_error",
            layer_triggered="Sidecar",
            normalized_text="",
            ticket_id=None,
            placeholder=BLOCK_PLACEHOLDER,
            audit=audit,
        )


def handle_get_ticket(ticket_id: str) -> dict:
    ticket = load_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="ticket_not_found")
    return asdict(ticket)


def handle_inspect_batch(items: list[dict]) -> list[dict]:
    """Process multiple inspect requests sequentially.

    NOTE: handle_inspect already submits work to _EXECUTOR internally,
    so we must NOT submit handle_inspect itself to _EXECUTOR — that would
    cause nested executor submissions and deadlock under load.
    """
    results = []
    for item in items:
        try:
            req = _validate_model(InspectRequest, item)
            resp = handle_inspect(req)
            results.append(_model_dump(resp))
        except Exception:
            results.append({
                "verdict": "BLOCK", "confidence": 1.0,
                "reason": "batch_item_error", "layer_triggered": "error",
            })
    return results


def health() -> dict[str, str]:
    return {"status": "ok"}


if FASTAPI_AVAILABLE:
    app = FastAPI(title="PRISM OpenClaw Sidecar", version="0.1.0")

    def _require_secret(request: Request) -> None:
        _require_secret_value(request.headers.get("X-PRISM-Secret"))

    @app.post("/v1/inspect", response_model=InspectResponse, dependencies=[Depends(_require_secret)])
    def inspect_route(request: InspectRequest) -> InspectResponse:
        return handle_inspect(request)

    @app.post("/v1/inspect/batch", dependencies=[Depends(_require_secret)])
    def inspect_batch_route(request: Request) -> JSONResponse:
        import asyncio
        body = asyncio.get_event_loop().run_until_complete(request.json())
        results = handle_inspect_batch(body.get("items", []))
        return JSONResponse({"results": results})

    @app.get("/v1/ticket/{ticket_id}", dependencies=[Depends(_require_secret)])
    def get_ticket_route(ticket_id: str) -> JSONResponse:
        return JSONResponse(handle_get_ticket(ticket_id))

    @app.get("/health")
    def health_route() -> dict[str, str]:
        return health()
else:
    app = None


class PrismRequestHandler(BaseHTTPRequestHandler):
    server_version = "PrismSidecar/0.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(200, health())
                return

            if parsed.path.startswith("/v1/ticket/"):
                _require_secret_value(self.headers.get("X-PRISM-Secret"))
                ticket_id = parsed.path.rsplit("/", 1)[-1]
                self._send_json(200, handle_get_ticket(ticket_id))
                return

            self._send_json(404, {"detail": "not_found"})
        except HTTPException as exc:
            self._send_json(exc.status_code, {"detail": exc.detail})
        except Exception:
            self._send_json(500, {"detail": "internal_error"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            _require_secret_value(self.headers.get("X-PRISM-Secret"))
            payload = self._read_json()

            if parsed.path == "/v1/inspect":
                request = _validate_model(InspectRequest, payload)
                response = handle_inspect(request)
                self._send_json(200, _model_dump(response))
            elif parsed.path == "/v1/inspect/batch":
                results = handle_inspect_batch(payload.get("items", []))
                self._send_json(200, {"results": results})
            else:
                self._send_json(404, {"detail": "not_found"})
                return
        except HTTPException as exc:
            self._send_json(exc.status_code, {"detail": exc.detail})
        except Exception as exc:
            self._send_json(500, {"detail": "internal_error", "error": str(exc)})


def run_server() -> None:
    if FASTAPI_AVAILABLE:
        import uvicorn

        uvicorn.run(app, host=DEFAULT_HOST, port=DEFAULT_PORT)
        return

    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), PrismRequestHandler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
