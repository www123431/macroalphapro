"""engine.research_store.red_lessons.papers_chroma — ChromaDB ingest for paper full-text.

Separate collection from the existing history_rag store. Chunked paragraphs
with section heuristics. Reuses the sentence-transformers stack already
deployed (paraphrase-multilingual-mpnet-base-v2 — see RAG eval session
2026-06-03).

Doctrine:
  - Each chunk has structured metadata: doi, year, authors, title, venue,
    section, paragraph_idx, source_kind, candidate_names (lessons that
    cite this paper)
  - Collection name: `papers_fulltext` (distinct from history_rag's
    collection)
  - Idempotent ingest: re-running for the same doi REPLACES (not duplicates).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHROMA_DIR = _REPO_ROOT / "data" / "research_store" / "papers_chroma"
COLLECTION_NAME = "papers_fulltext"

_EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"


# ─────────────────────── chunking ─────────────────────────────────────


# Heuristic section markers — academic finance papers tend to use
# "1. Introduction", "2.  Data", "Section 3.", "References", etc.
_SECTION_PATTERNS = [
    re.compile(r"^\s*(\d+[.\s]+(?:[A-Z][A-Za-z &'-]+))\s*$", re.MULTILINE),
    re.compile(r"^\s*(Section\s+\d+[\.\:]?\s*[A-Z][A-Za-z &'-]*)\s*$", re.MULTILINE),
    re.compile(r"^\s*(Abstract|Introduction|Conclusion|References)\s*$", re.MULTILINE),
]


_REFERENCES_HEADING_RE = re.compile(
    r"^\s*(References?|Bibliography|Works\s+Cited|REFERENCES)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_PAGE_NUMBER_ONLY_RE = re.compile(r"^[\s\d\-—–.,(c)©Page]+$")


def _strip_references_and_after(text: str) -> str:
    """Truncate at the first 'References' / 'Bibliography' heading.

    Paper References sections are full of citation noise (long author lists,
    journal names) that pollute embeddings with non-content tokens. We
    discard everything from the References heading onward.
    """
    m = _REFERENCES_HEADING_RE.search(text)
    if m is None:
        return text
    # Only treat as References if it appears in the latter half of the paper
    if m.start() < len(text) * 0.4:
        # Likely a within-body reference (e.g. "See References in §2"); ignore.
        return text
    return text[: m.start()]


def _is_page_number_chunk(text: str) -> bool:
    """Filter out chunks that are essentially page-number / running-header
    noise (common from PDF extraction)."""
    if len(text) > 100:
        return False
    return bool(_PAGE_NUMBER_ONLY_RE.match(text))


def _split_paragraphs(text: str) -> list[str]:
    """Split by blank lines, normalize whitespace, drop short and page-num chunks."""
    text = _strip_references_and_after(text)
    paragraphs = re.split(r"\n\s*\n", text)
    out = []
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) < 50:
            continue
        if _is_page_number_chunk(p):
            continue
        out.append(p)
    return out


def _detect_section(text: str, char_idx: int) -> str:
    """Walk backwards from char_idx to find the nearest section heading."""
    head = text[:char_idx]
    candidates = []
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(head):
            candidates.append((m.start(), m.group(1).strip()))
    if not candidates:
        return "body"
    candidates.sort()
    return candidates[-1][1][:60]


@dataclass(frozen=True)
class PaperChunk:
    doi:             str
    chunk_id:        str
    text:            str
    section:         str
    paragraph_idx:   int
    metadata:        dict[str, Any]


def chunk_paper(full_text: str, *,
                doi: str,
                title: str = "",
                year: int | None = None,
                authors: tuple[str, ...] = (),
                venue: str = "",
                source_kind: str = "",
                candidate_names: tuple[str, ...] = (),
                min_chars: int = 200,
                max_chars: int = 1800) -> list[PaperChunk]:
    """Chunk a paper into paragraph-anchored, section-tagged pieces.

    Strategy:
      - split on blank lines into paragraphs
      - merge consecutive short paragraphs into chunks of [min, max] chars
      - assign each chunk to its containing section (nearest preceding heading)
    """
    if not doi:
        raise ValueError("doi is required for chunk_paper (used as id prefix)")

    paragraphs = _split_paragraphs(full_text)
    if not paragraphs:
        return []

    chunks: list[PaperChunk] = []
    buf = ""
    buf_start_idx = 0
    paragraph_idx = 0
    cursor = 0
    for i, p in enumerate(paragraphs):
        # find p in full_text starting from cursor
        loc = full_text.find(p[:80], cursor) if len(p) >= 80 else full_text.find(p, cursor)
        if loc < 0:
            loc = cursor
        cursor = loc + len(p)

        if not buf:
            buf = p
            buf_start_idx = loc
        else:
            cand = buf + "\n\n" + p
            if len(cand) <= max_chars:
                buf = cand
            else:
                # flush buf
                if len(buf) >= min_chars:
                    section = _detect_section(full_text, buf_start_idx)
                    cid = f"{doi}::p{paragraph_idx:04d}"
                    chunks.append(PaperChunk(
                        doi=doi, chunk_id=cid, text=buf, section=section,
                        paragraph_idx=paragraph_idx,
                        metadata={
                            "doi": doi, "title": title, "year": year or 0,
                            "authors": ", ".join(authors), "venue": venue,
                            "section": section, "paragraph_idx": paragraph_idx,
                            "source_kind": source_kind,
                            "candidate_names": ", ".join(candidate_names),
                        },
                    ))
                    paragraph_idx += 1
                buf = p
                buf_start_idx = loc

    # flush final buffer
    if buf and len(buf) >= min_chars:
        section = _detect_section(full_text, buf_start_idx)
        cid = f"{doi}::p{paragraph_idx:04d}"
        chunks.append(PaperChunk(
            doi=doi, chunk_id=cid, text=buf, section=section,
            paragraph_idx=paragraph_idx,
            metadata={
                "doi": doi, "title": title, "year": year or 0,
                "authors": ", ".join(authors), "venue": venue,
                "section": section, "paragraph_idx": paragraph_idx,
                "source_kind": source_kind,
                "candidate_names": ", ".join(candidate_names),
            },
        ))

    return chunks


# ─────────────────────── chromadb singleton + ingest ──────────────────


_chroma_client = None
_chroma_collection = None
_embed_fn = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _get_embed_fn():
    """Return a ChromaDB SentenceTransformerEmbeddingFunction reusing the
    same model as the history_rag store."""
    global _embed_fn
    if _embed_fn is None:
        from chromadb.utils import embedding_functions
        _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_EMBED_MODEL,
        )
    return _embed_fn


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = _get_client()
        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embed_fn(),
        )
    return _chroma_collection


def delete_paper(doi: str) -> int:
    """Delete all chunks of a paper. Returns number deleted (best-effort)."""
    if not doi:
        return 0
    coll = get_collection()
    # ChromaDB doesn't support prefix-delete; we query by metadata equality
    res = coll.get(where={"doi": doi})
    ids = res.get("ids") or []
    if ids:
        coll.delete(ids=ids)
    return len(ids)


def ingest_chunks(chunks: list[PaperChunk]) -> int:
    """Idempotent ingest: if any chunk's doi already exists, delete first,
    then add fresh chunks."""
    if not chunks:
        return 0
    coll = get_collection()
    dois = {c.doi for c in chunks}
    for d in dois:
        delete_paper(d)
    coll.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)


def collection_stats() -> dict[str, int]:
    coll = get_collection()
    return {"n_chunks": coll.count()}
