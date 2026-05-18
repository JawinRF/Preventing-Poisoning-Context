"""
content_extractor.py — Format-aware semantic payload extractor.

Before ML models score text, detect the container format and extract only the
semantic payload. This prevents distribution shift where injection classifiers
trained on natural language receive structured system data (Android XML
hierarchies, file wrappers, intent JSON, HTML) — formats the models have
never been trained on.

Handled formats (in detection order):
  - Android accessibility XML  : <?xml … <hierarchy><node text="…" …/>
  - File block wrapper          : --- START FILE: name --- … --- END FILE ---
  - HTML                        : <html> … </html>
  - JSON (intent / config)      : {"action": …, "data": …} or any JSON object
  - Plain text                  : returned unchanged
"""
from __future__ import annotations

import json
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

_FILE_START = "--- START FILE:"
_FILE_END   = "--- END FILE ---"

# Only human-visible text attributes; class / resource-id are structural noise
_XML_TEXT_ATTRS = ("text", "content-desc", "hint", "label")


class ContentExtractor:
    """Extract semantic payload from structured formats before ML scoring."""

    def extract(self, text: str, ingestion_path: str = "") -> str:
        """Return the semantic payload, or text unchanged if format is unrecognised."""
        if not isinstance(text, str) or not text.strip():
            return text or ""

        s = text.strip()

        if s.startswith("<?xml") or s.startswith("<hierarchy"):
            return self._from_xml(s) or text

        if _FILE_START in s:
            return self._from_file_block(s) or text

        lower = s.lower()
        if lower.startswith("<html") or ("<body" in lower and "</body>" in lower):
            return self._from_html(s) or text

        if s.startswith("{") or s.startswith("["):
            return self._from_json(s) or text

        return text

    # ── Format handlers ───────────────────────────────────────────────────────

    def _from_xml(self, xml_text: str) -> str:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return ""
        tokens: list[str] = []
        seen: set[str] = set()
        for elem in root.iter():
            for attr in _XML_TEXT_ATTRS:
                val = (elem.get(attr) or "").strip()
                if val and val not in seen:
                    seen.add(val)
                    tokens.append(val)
        return " ".join(tokens)

    def _from_file_block(self, text: str) -> str:
        parts: list[str] = []
        remaining = text
        while _FILE_START in remaining:
            start   = remaining.find(_FILE_START)
            newline = remaining.find("\n", start)
            if newline == -1:
                break
            inner_start = newline + 1
            end = remaining.find(_FILE_END, inner_start)
            if end == -1:
                inner     = remaining[inner_start:].strip()
                remaining = ""
            else:
                inner     = remaining[inner_start:end].strip()
                remaining = remaining[end + len(_FILE_END):]
            if inner:
                extracted = self._extract_inner(inner)
                parts.append(extracted if extracted else inner)
        return "\n".join(parts)

    def _extract_inner(self, text: str) -> str:
        s = text.strip()
        if s.startswith("{") or s.startswith("["):
            return self._from_json(s)
        if s.startswith("<?xml") or s.startswith("<"):
            return self._from_xml(s)
        return s

    def _from_json(self, json_text: str) -> str:
        try:
            obj = json.loads(json_text)
        except (json.JSONDecodeError, ValueError):
            # Retry: Android intent data sometimes embeds literal control
            # characters (newlines, tabs) inside JSON string values, making
            # the first parse fail. Replace with spaces to recover structure.
            try:
                sanitized = (json_text
                             .replace('\n', ' ')
                             .replace('\r', ' ')
                             .replace('\t', ' '))
                obj = json.loads(sanitized)
            except (json.JSONDecodeError, ValueError):
                return ""
        if isinstance(obj, dict) and "action" in obj and "data" in obj:
            # Android intent shape — score data field first to surface injections
            # before the structural action package name dilutes the signal.
            return self._from_android_intent(obj)
        tokens: list[str] = []
        _collect_strings(obj, tokens)
        return " ".join(tokens)

    def _from_html(self, html_text: str) -> str:
        collector = _HtmlTextCollector()
        try:
            collector.feed(html_text)
        except Exception:
            return ""
        return " ".join(collector.parts)

    def _from_android_intent(self, obj: dict) -> str:
        """For Android intent JSON, score the 'data' field first (most likely attack surface),
        then append other values. The 'action' package name is structural noise."""
        tokens: list[str] = []
        # data field first — that's where injections hide
        data_val = obj.get("data", "")
        if isinstance(data_val, str) and data_val.strip():
            tokens.append(data_val.strip())
        # remaining values except 'action' (package name is structural noise)
        for k, v in obj.items():
            if k in ("data", "action"):  # action is a package name — structural noise
                continue
            if isinstance(v, str):
                s = v.strip()
                if s:
                    tokens.append(s)
            else:
                _collect_strings(v, tokens)
        return " ".join(tokens)


def _collect_strings(obj: object, out: list[str]) -> None:
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_strings(item, out)


class _HtmlTextCollector(HTMLParser):
    # Skip layout/metadata tags — keep script content for injection scanning
    _SKIP = frozenset(("style", "head", "meta", "link", "noscript"))

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP and self._depth > 0:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth:
            return
        s = data.strip()
        if s:
            self.parts.append(s)

    def handle_comment(self, data: str) -> None:
        # HTML comments are a common injection vector — include them for scoring
        s = data.strip()
        if s:
            self.parts.append(s)
