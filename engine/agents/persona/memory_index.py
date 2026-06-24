"""engine/agents/persona/memory_index.py — semantic index over /memory/*.md.

Phase A.7 Wave 3.3. Embeds every human-curated memory file once with
sentence-transformers all-MiniLM-L6-v2 (cheap 384-dim, ~85MB model);
persists the matrix to disk and rebuilds only when a memory file is
newer than the index. Queries are encoded on the fly and ranked by
cosine similarity.

Why semantic and not just keyword: keyword search misses paraphrases.
"emoji" matches `feedback_no_emojis_2026-05-19.md` but "should agents
use icons?" finds nothing. Semantic catches both.

Why MiniLM and not a heavier model: solo-PM scale (~80 memory files,
short paragraphs). MiniLM gives 90%+ of the recall at 1/4 the latency
of mpnet-base and runs entirely on CPU.

Index layout (one NPZ file at data/cache/memory_index.npz):
  filenames: array of relative .md filenames
  embeddings: (n_files, 384) float32
  built_at: ISO timestamp
  source_mtime_max: float (max mtime of source files at build time)

Index is invalidated automatically when any source file has mtime
greater than source_mtime_max. Lazy build — the first call to
search_memory() in a process rebuilds if needed, otherwise loads.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_MODEL_NAME       = "all-MiniLM-L6-v2"
_CACHE_DIR        = Path("data/cache")
_INDEX_FILE       = _CACHE_DIR / "memory_index.npz"
_TOP_K_DEFAULT    = 5


# Module-level singletons — model loaded once per process.
_MODEL = None
_INDEX_CACHE: dict = {}   # keys: filenames / embeddings / source_mtime_max


def _memory_dir() -> Path | None:
    """Resolve the Claude Code memory directory for the current project.

    Mirrors the path-sanitization heuristic in read_project_memory.
    Returns None if not found — callers must handle gracefully.
    """
    cwd = Path.cwd().resolve()
    sanitized = (
        str(cwd).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    candidate = Path.home() / ".claude" / "projects" / sanitized / "memory"
    if candidate.exists():
        return candidate
    # Lowercase-drive variant on Windows
    alt = Path.home() / ".claude" / "projects" / sanitized.lower() / "memory"
    if alt.exists():
        return alt
    return None


def _load_model():
    """Lazy-import + cache the sentence-transformers model. First call
    in a process takes ~1-3s; subsequent calls are O(1)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _list_source_files(mem_dir: Path) -> list[Path]:
    """Sorted list of all .md files in the memory directory, skipping
    MEMORY.md (the index, not a memory entry itself)."""
    return sorted(
        f for f in mem_dir.glob("*.md")
        if f.name != "MEMORY.md"
    )


def _max_mtime(files: list[Path]) -> float:
    if not files:
        return 0.0
    return max(f.stat().st_mtime for f in files)


def _build_index(mem_dir: Path) -> dict:
    """Rebuild the embedding index from scratch. Cached to disk via NPZ."""
    import numpy as np
    files = _list_source_files(mem_dir)
    if not files:
        return {"filenames": [], "embeddings": np.empty((0, 384), dtype="float32"),
                "source_mtime_max": 0.0}

    model = _load_model()
    texts: list[str] = []
    names: list[str] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            logger.warning("memory_index: skipping %s: %s", f, exc)
            continue
        # Embed the WHOLE file (MiniLM truncates at 256 tokens internally;
        # memory files are short enough that this is fine and we want
        # broad-document semantics not chunk-level).
        texts.append(text)
        names.append(f.name)

    logger.info("memory_index: building over %d files (model=%s)",
                len(texts), _MODEL_NAME)
    embeddings = model.encode(
        texts,
        normalize_embeddings = True,   # cosine becomes dot product
        show_progress_bar    = False,
        convert_to_numpy     = True,
    ).astype("float32")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _INDEX_FILE,
        filenames        = np.array(names, dtype=object),
        embeddings       = embeddings,
        source_mtime_max = np.float64(_max_mtime(files)),
    )
    return {
        "filenames":        names,
        "embeddings":       embeddings,
        "source_mtime_max": _max_mtime(files),
    }


def _ensure_index(mem_dir: Path) -> dict:
    """Load cached index from disk; rebuild if any source file is newer."""
    global _INDEX_CACHE
    import numpy as np

    source_mtime_max = _max_mtime(_list_source_files(mem_dir))

    # Check in-process cache first
    if _INDEX_CACHE and _INDEX_CACHE.get("source_mtime_max", -1) >= source_mtime_max:
        return _INDEX_CACHE

    # Check disk cache
    if _INDEX_FILE.exists():
        try:
            data = np.load(_INDEX_FILE, allow_pickle=True)
            cached_mtime = float(data["source_mtime_max"])
            if cached_mtime >= source_mtime_max:
                _INDEX_CACHE = {
                    "filenames":        list(data["filenames"]),
                    "embeddings":       data["embeddings"],
                    "source_mtime_max": cached_mtime,
                }
                logger.debug("memory_index: loaded cache (mtime=%s, n=%d)",
                             cached_mtime, len(_INDEX_CACHE["filenames"]))
                return _INDEX_CACHE
        except Exception as exc:
            logger.warning("memory_index: cache load failed, rebuilding: %s", exc)

    _INDEX_CACHE = _build_index(mem_dir)
    return _INDEX_CACHE


def search_memory(query: str, top_k: int = _TOP_K_DEFAULT) -> list[dict]:
    """Return the top-K most semantically similar memory files for ``query``.

    Each result dict has: file (str), score (float 0-1), description (str).
    Returns [] on empty memory dir / model load failure / encoding error
    so callers can fall back to keyword search.
    """
    mem_dir = _memory_dir()
    if mem_dir is None:
        return []
    try:
        index = _ensure_index(mem_dir)
        if not index["filenames"]:
            return []

        model = _load_model()
        import numpy as np
        q_emb = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")[0]

        scores = index["embeddings"] @ q_emb   # cosine since both normalized
        order  = np.argsort(-scores)[:top_k]

        results: list[dict] = []
        for idx in order:
            name = index["filenames"][int(idx)]
            score = float(scores[int(idx)])
            # Read the file again for the description line — cheap (~80 files)
            try:
                text = (mem_dir / name).read_text(encoding="utf-8", errors="ignore")
                desc = ""
                for line in text.split("\n")[:15]:
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip()
                        break
            except Exception:
                desc = ""
            results.append({
                "file":        name,
                "score":       round(score, 4),
                "description": desc[:300],
            })
        return results
    except Exception as exc:
        logger.warning("memory_index.search_memory(%r) failed: %s", query, exc)
        return []


def invalidate_cache() -> None:
    """Clear the in-process and on-disk cache. Useful for tests + after
    bulk memory file edits."""
    global _INDEX_CACHE
    _INDEX_CACHE = {}
    try:
        if _INDEX_FILE.exists():
            _INDEX_FILE.unlink()
    except Exception as exc:
        logger.debug("memory_index.invalidate_cache: %s", exc)
