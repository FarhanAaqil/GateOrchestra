"""
assistant/documents.py
======================
Uploaded documents: text extraction, chunking and retrieval.

Text, Markdown, CSV, JSON and code files are read as UTF-8; PDFs through
pypdf. Text is split into overlapping chunks of about 900 characters on
paragraph boundaries. Retrieval ranks chunks with TF-IDF cosine similarity
(scikit-learn, already a dependency), so it needs no embedding API and no
extra tokens.
"""

from __future__ import annotations

import base64
import io
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
CHUNK_CHARS = 900
OVERLAP_CHARS = 150
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".sql",
    ".yaml",
    ".yml",
    ".xml",
    ".java",
    ".c",
    ".cpp",
    ".go",
    ".rs",
    ".rb",
}


class DocumentError(ValueError):
    pass


def extract_text(name: str, data: bytes) -> str:
    suffix = ("." + name.rsplit(".", 1)[-1].lower()) if "." in name else ""
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError("Files are limited to 10 MB.")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as err:  # pragma: no cover - dependency is declared
            raise DocumentError("PDF support needs the pypdf package.") from err
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as err:
            raise DocumentError(f"Could not read the PDF ({type(err).__name__}).") from err
    elif suffix in TEXT_SUFFIXES or not suffix:
        text = data.decode("utf-8", errors="replace")
    else:
        raise DocumentError(
            f"Unsupported file type {suffix}. Use text, Markdown, code, CSV, JSON or PDF."
        )
    text = re.sub(r"[ \t]+\n", "\n", text).strip()
    if not text:
        raise DocumentError("No text could be extracted from this file.")
    return text


def decode_upload(content_base64: str) -> bytes:
    try:
        return base64.b64decode(content_base64, validate=True)
    except (ValueError, TypeError) as err:
        raise DocumentError("The upload was not valid base64.") from err


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Paragraph-aware chunks of about ``size`` characters with some overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > size:  # a single huge paragraph is cut on sentence-ish edges
            cut = max(para.rfind(". ", 0, size), para.rfind("\n", 0, size), size // 2)
            piece, para = para[: cut + 1].strip(), para[cut + 1 :].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece)
        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
    if current:
        chunks.append(current)
    return chunks


_WORD = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "es", "ed", "ly", "s")


def _stem(word: str) -> str:
    """A light stemmer so 'returns', 'returned' and 'returning' all match 'return'."""
    for suffix in _SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + ("y" if suffix in ("ies", "ied") else "")
    return word


def _analyze(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


def top_chunks(
    question: str, chunks: list[dict], k: int = 4, min_score: float = 0.05
) -> list[dict]:
    """The ``k`` chunks most similar to ``question`` (each dict needs a 'text' key).

    A document set that fits in ``k`` chunks is returned whole: ranking a
    handful of passages can only lose context.
    """
    if not chunks or not question.strip():
        return []
    if len(chunks) <= k:
        return [{**c, "score": 1.0} for c in chunks]
    texts = [c["text"] for c in chunks]
    try:
        vectorizer = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
        matrix = vectorizer.fit_transform([*texts, question])
    except ValueError:  # nothing but stop words
        return []
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    return [{**chunks[i], "score": float(scores[i])} for i in ranked[:k] if scores[i] >= min_score]


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{h['name']} · part {h['idx'] + 1}]\n{h['text']}" for h in hits)
