"""
doc_chunker.py — Simple document chunking for RAG ingestion.
Splits text files into semantically meaningful chunks for ChromaDB storage.
Uses sentence boundaries for splitting large paragraphs (not raw character counts).
"""
import re
from pathlib import Path

# Sentence boundary: period/question/exclamation followed by whitespace or end
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Keeps each sentence intact."""
    parts = _SENTENCE_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunk_text(text: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Split text into chunks by paragraphs, merging small ones.
    Large paragraphs are split on sentence boundaries with overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # Flush current buffer first
            if current:
                chunks.append(current.strip())
                current = ""
            # Split on sentence boundaries with overlap
            sentences = _split_sentences(para)
            buf = ""
            for i, sent in enumerate(sentences):
                if buf and len(buf) + len(sent) + 1 > max_chars:
                    chunks.append(buf.strip())
                    # Overlap: start next chunk with last N sentences from previous
                    overlap_start = max(0, i - overlap_sentences)
                    buf = " ".join(sentences[overlap_start:i]) + " " + sent
                else:
                    buf = f"{buf} {sent}" if buf else sent
            if buf.strip():
                chunks.append(buf.strip())
        elif len(current) + len(para) + 2 > max_chars:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks


def load_and_chunk(filepath: str, max_chars: int = 500, overlap_sentences: int = 1) -> list[str]:
    """
    Load a file and return chunks. Supports .txt, .md.
    PDF support requires PyPDF2 (optional).
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PDF ingestion requires PyPDF2: pip install PyPDF2")
        reader = PdfReader(str(path))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    return chunk_text(text, max_chars=max_chars, overlap_sentences=overlap_sentences)
