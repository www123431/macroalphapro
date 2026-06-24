"""engine.agents.papers_curator.doctrine_index — Tier-2 query_doctrine.

Builds + queries a ChromaDB collection over the principal's memory
files (`~/.claude/projects/.../memory/*.md`). When A's synthesis or
B's review fires, query_doctrine(topic_hint, top_k=5) returns the
top-K most relevant memory entries so the LLM reasons against the
principal's accumulated doctrine instead of from scratch.

Pre-tier-2 (everything before this module): synthesis_context's
_load_doctrine_snippets and strengthener_runner's
_load_doctrine_snippets were both stubs returning (). A and B
reasoned without ever seeing the 200+ memory entries the principal
has locked. THIS WAS THE LARGEST QUALITY BOTTLENECK.

Storage:
  data/research_store/doctrine_chroma/  (separate from papers_chroma —
  different content semantic, different schema, different reindex
  cadence).
  Collection name: doctrine_v1.

Embedding: SentenceTransformer (same model as papers_chroma) so the
embedding cost is amortized.

Re-ingest policy: incremental by mtime. Each entry tracks
last_indexed_mtime; query_doctrine() lazy-ingests changed files
before query. Full rebuild via ingest_doctrine(force=True).

Frontmatter contract (the principal's auto-memory format):
  ---
  name:        <slug>
  description: <one-line>
  metadata:
    type: feedback | project | reference | user
  ---
  <markdown body>

Failure mode handling:
  - chromadb / SentenceTransformer not installed → return () empty
    tuple (A/B fall back to no-doctrine reasoning, same as pre-tier-2)
  - memory dir doesn't exist → return () empty
  - file parse failure → log + skip that file, continue

Cost: $0 (local SentenceTransformer embedding; no LLM call). One-time
~30s indexing cost on 316 files; incremental updates milliseconds.
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent.parent
_DOCTRINE_DIR    = _REPO_ROOT / "data" / "research_store" / "doctrine_chroma"
_COLLECTION_NAME = "doctrine_v1"
_EMBED_MODEL     = "all-MiniLM-L6-v2"

# Default memory directory — overrideable via env for cross-machine
# portability + testing.
_DEFAULT_MEMORY_DIR = Path(
    os.environ.get(
        "MACROALPHA_MEMORY_DIR",
        str(Path.home() / ".claude" / "projects" /
            "c--Users-${USER}-Desktop-intern" / "memory")
    )
)


# ────────────────────────────────────────────────────────────────────
# Parsed entry shape
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class DoctrineEntry:
    file_id:       str   # slug from frontmatter `name:` field
    file_path:     str
    name:          str
    description:   str
    entry_type:    str   # feedback / project / reference / user / other
    body:          str
    mtime:         float


_FRONTMATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL
)


def _parse_memory_file(path: Path) -> Optional[DoctrineEntry]:
    """Parse a single memory file. Returns None if the file doesn't
    have parseable frontmatter — caller logs + skips. The
    MEMORY.md index file is skipped (it's the index, not an entry)."""
    if path.name == "MEMORY.md":
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("doctrine_index: failed to read %s: %s", path, exc)
        return None

    m = _FRONTMATTER_RE.match(text)
    if not m:
        # Files without frontmatter — skip silently, they're not
        # part of the doctrine corpus
        return None
    fm_raw, body = m.group(1), m.group(2)

    # Light YAML parse — we only need name / description /
    # metadata.type. Avoid pulling in pyyaml for this since the format
    # is well-controlled by the auto-memory system.
    name        = _grab(fm_raw, "name") or path.stem
    description = _grab(fm_raw, "description") or ""
    entry_type  = _grab(fm_raw, "type") or "other"

    try:
        mtime = path.stat().st_mtime
    except Exception:
        mtime = 0.0

    return DoctrineEntry(
        file_id     = name,
        file_path   = str(path),
        name        = name,
        description = description,
        entry_type  = entry_type,
        body        = body.strip(),
        mtime       = mtime,
    )


def _grab(fm: str, key: str) -> Optional[str]:
    """Lightweight 'key: value' grab on YAML-ish frontmatter. Handles
    both top-level fields (name, description) and one level of nesting
    (metadata.type). Trims surrounding whitespace + quotes."""
    pattern = re.compile(
        rf"^\s*{re.escape(key)}\s*:\s*(.*?)$",
        re.MULTILINE,
    )
    m = pattern.search(fm)
    if not m:
        return None
    v = m.group(1).strip()
    # Strip surrounding quotes if present
    if (v.startswith('"') and v.endswith('"')) or \
       (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return v


# ────────────────────────────────────────────────────────────────────
# Iteration over memory dir
# ────────────────────────────────────────────────────────────────────
def iter_memory_entries(memory_dir: Optional[Path] = None
                          ) -> Iterable[DoctrineEntry]:
    """Walk the memory directory, yield parsed entries. Empty + silent
    on missing dir (caller treats as 'no doctrine corpus yet')."""
    md = memory_dir or _DEFAULT_MEMORY_DIR
    if not md.is_dir():
        return
    for p in sorted(md.glob("*.md")):
        e = _parse_memory_file(p)
        if e is not None:
            yield e


# ────────────────────────────────────────────────────────────────────
# Chroma plumbing
# ────────────────────────────────────────────────────────────────────
_chroma_client = None
_chroma_collection = None
_embed_fn = None


def _get_client():
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            _DOCTRINE_DIR.mkdir(parents=True, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=str(_DOCTRINE_DIR))
        except Exception as exc:
            logger.warning("doctrine_index: chroma client failed: %s", exc)
            return None
    return _chroma_client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is None:
        try:
            from chromadb.utils import embedding_functions
            _embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=_EMBED_MODEL,
            )
        except Exception as exc:
            logger.warning("doctrine_index: embed fn failed: %s", exc)
            return None
    return _embed_fn


def get_doctrine_collection():
    """Return the doctrine_v1 ChromaDB collection (lazy init). Returns
    None on infra failure — caller treats as 'no doctrine retrieval
    available, fall back to empty snippets'."""
    global _chroma_collection
    if _chroma_collection is None:
        client = _get_client()
        embed_fn = _get_embed_fn()
        if client is None or embed_fn is None:
            return None
        try:
            _chroma_collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                embedding_function=embed_fn,
            )
        except Exception as exc:
            logger.warning("doctrine_index: collection create failed: %s", exc)
            return None
    return _chroma_collection


# ────────────────────────────────────────────────────────────────────
# Ingest — incremental by mtime, full rebuild on force=True
# ────────────────────────────────────────────────────────────────────
def ingest_doctrine(
    memory_dir:  Optional[Path] = None,
    *,
    force:       bool = False,
) -> dict:
    """Add new + changed entries to the doctrine collection.

    force=False (default): only files whose mtime > last_indexed_mtime
                            on the existing chroma row are re-embedded
                            (incremental).
    force=True:             delete the whole collection first; rebuild
                            from scratch.

    Returns:
      {n_scanned: int, n_added: int, n_updated: int, n_skipped: int,
       n_unparseable: int}
    """
    coll = get_doctrine_collection()
    if coll is None:
        return {"n_scanned": 0, "n_added": 0, "n_updated": 0,
                "n_skipped": 0, "n_unparseable": 0,
                "error": "chroma collection unavailable"}

    if force:
        try:
            existing = coll.get()
            if existing.get("ids"):
                coll.delete(ids=existing["ids"])
        except Exception as exc:
            logger.warning("doctrine_index: force-delete failed: %s", exc)

    # Read existing mtimes so we can decide "skip vs re-embed"
    existing_mtimes: dict[str, float] = {}
    if not force:
        try:
            existing = coll.get()
            ids   = existing.get("ids") or []
            metas = existing.get("metadatas") or []
            for i, m in zip(ids, metas):
                if m and "mtime" in m:
                    try:
                        existing_mtimes[i] = float(m["mtime"])
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    md = memory_dir or _DEFAULT_MEMORY_DIR
    n_scanned = n_added = n_updated = n_skipped = n_unparseable = 0
    if not md.is_dir():
        return {"n_scanned": 0, "n_added": 0, "n_updated": 0,
                "n_skipped": 0, "n_unparseable": 0,
                "error": f"memory dir not found: {md}"}

    new_ids:  list[str] = []
    new_docs: list[str] = []
    new_meta: list[dict] = []
    update_ids: list[str] = []
    update_docs: list[str] = []
    update_meta: list[dict] = []

    for p in sorted(md.glob("*.md")):
        n_scanned += 1
        e = _parse_memory_file(p)
        if e is None:
            n_unparseable += 1
            continue
        prev_mtime = existing_mtimes.get(e.file_id)
        if prev_mtime is not None and prev_mtime >= e.mtime:
            n_skipped += 1
            continue
        # Embed text = description + body (description is high-density
        # summary; body provides context for retrieval relevance)
        doc = f"{e.description}\n\n{e.body}"
        meta = {
            "name":        e.name,
            "description": e.description,
            "entry_type":  e.entry_type,
            "file_path":   e.file_path,
            "mtime":       e.mtime,
        }
        if prev_mtime is None:
            new_ids.append(e.file_id)
            new_docs.append(doc)
            new_meta.append(meta)
            n_added += 1
        else:
            update_ids.append(e.file_id)
            update_docs.append(doc)
            update_meta.append(meta)
            n_updated += 1

    try:
        if new_ids:
            coll.add(ids=new_ids, documents=new_docs, metadatas=new_meta)
        if update_ids:
            # ChromaDB's upsert handles both update + add cleanly
            coll.upsert(ids=update_ids, documents=update_docs,
                          metadatas=update_meta)
    except Exception as exc:
        logger.warning("doctrine_index: ingest write failed: %s", exc)
        return {"n_scanned": n_scanned, "n_added": 0, "n_updated": 0,
                "n_skipped": n_skipped, "n_unparseable": n_unparseable,
                "error": str(exc)}

    return {
        "n_scanned":     n_scanned,
        "n_added":       n_added,
        "n_updated":     n_updated,
        "n_skipped":     n_skipped,
        "n_unparseable": n_unparseable,
    }


# ────────────────────────────────────────────────────────────────────
# Public retrieval API
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class DoctrineHit:
    """Search result. Shape matches the existing DoctrineHit dataclass
    in synthesis.py + strengthener/review.py — synthesis_context and
    strengthener_runner adapt this into their callers' types."""
    name:        str
    description: str
    entry_type:  str
    snippet:     str         # first ~400 chars of body — fits LLM prompt
    distance:    float       # chroma cosine distance (0 = identical)
    file_path:   str


def query_doctrine(
    topic_hint:    str,
    *,
    top_k:         int = 5,
    memory_dir:    Optional[Path] = None,
    auto_ingest:   bool = True,
) -> tuple[DoctrineHit, ...]:
    """Retrieve the top-K doctrine entries most relevant to topic_hint.

    Empty topic_hint → returns () (no anchor, no point firing chroma).
    chroma infra failure → returns () (caller falls back to empty
    doctrine_snippets — same as pre-tier-2 stub behavior).

    auto_ingest=True (default) runs ingest_doctrine() lazily on first
    use of the day so the collection stays warm without an explicit
    cron step.
    """
    if not topic_hint or not topic_hint.strip():
        return ()

    if auto_ingest:
        try:
            ingest_doctrine(memory_dir=memory_dir, force=False)
        except Exception as exc:
            logger.warning("doctrine_index: auto-ingest failed: %s", exc)

    coll = get_doctrine_collection()
    if coll is None:
        return ()
    try:
        res = coll.query(query_texts=[topic_hint], n_results=top_k)
    except Exception as exc:
        logger.warning("doctrine_index: query failed: %s", exc)
        return ()

    ids       = (res.get("ids") or [[]])[0]
    docs      = (res.get("documents") or [[]])[0]
    metas     = (res.get("metadatas") or [[]])[0]
    distances = (res.get("distances") or [[]])[0]

    out: list[DoctrineHit] = []
    for i, doc, m, d in zip(ids, docs, metas, distances):
        m = m or {}
        out.append(DoctrineHit(
            name        = str(m.get("name") or i),
            description = str(m.get("description") or "")[:400],
            entry_type  = str(m.get("entry_type") or "other"),
            snippet     = (doc or "")[:400],
            distance    = float(d) if d is not None else 1.0,
            file_path   = str(m.get("file_path") or ""),
        ))
    return tuple(out)
