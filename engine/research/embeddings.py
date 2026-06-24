"""Embedding service + per-ledger semantic index for /ask retrieval.

Doctrine: Vector RAG was deferred while keyword + recency was good enough
for N=35 mechanisms. By 2026-06 the ledgers had grown enough that synonym
misses became measurable (e.g. "trailing performance" missing
"trailing_sharpe" hits). Replacing that lookup with a 384-d MiniLM
encoder and per-ledger npz indices.

Model: sentence-transformers/all-MiniLM-L6-v2 — 80 MB, 384-d, mean-pooled,
already L2-normalized on output. CPU torch is fine; embedding the whole
ledger takes seconds, not minutes.

Storage: one .npz per ledger at data/research/_embedding_index/. Each
archive holds row_hashes (uint64), embeddings (N, 384) float32, and
payloads (object array, JSON snippets). Builder dedupes by row_hash so
rebuilds are incremental.

CN net: HF_ENDPOINT defaults to hf-mirror.com if unset, matching the
SJTU/aliyun mirror discipline already used elsewhere in the repo
(per `feedback_check_latest_memory_before_new_line_2026-05-21`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_INDEX_DIR = _REPO_ROOT / "data" / "research" / "_embedding_index"

_model = None


def _get_model():
    """Lazy singleton model load. CN-friendly: sets HF_ENDPOINT to the
    hf-mirror only if user hasn't already configured one."""
    global _model
    if _model is not None:
        return _model
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode(texts: str | list[str]) -> np.ndarray:
    """Encode one or more strings. Returns (N, 384) float32 array,
    L2-normalized so dot product = cosine similarity."""
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)
    vecs = _get_model().encode(
        batch,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32", copy=False)
    return vecs[0] if single else vecs


def _row_hash(snippet: str) -> np.uint64:
    h = hashlib.blake2b(snippet.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(h, "big", signed=False))


def _load_index(ledger_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = _INDEX_DIR / f"{ledger_key}.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return z["row_hashes"], z["embeddings"], z["payloads"]
    except Exception:
        return None


def _save_index(ledger_key: str, row_hashes: np.ndarray,
                embeddings: np.ndarray, payloads: np.ndarray) -> None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # np.savez appends .npz to the basename automatically — write to a
    # tmp basename, then rename the resulting .npz.
    tmp_base = _INDEX_DIR / f"_{ledger_key}.partial"
    np.savez(str(tmp_base), row_hashes=row_hashes,
             embeddings=embeddings, payloads=payloads)
    tmp_actual = _INDEX_DIR / f"_{ledger_key}.partial.npz"
    target = _INDEX_DIR / f"{ledger_key}.npz"
    if target.exists():
        target.unlink()
    tmp_actual.rename(target)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


# ── Per-ledger snippet builders ─────────────────────────────────────
#
# Snippet text is what gets embedded. Keep them short, content-focused;
# do not include hashes / timestamps that are mechanically uninformative
# (would only collide with literal date queries).


def _snippet_l4(row: dict) -> tuple[str, dict]:
    p = row.get("proposal") or {}
    council = row.get("council") or {}
    pipeline = row.get("pipeline") or {}
    text = " ".join([
        p.get("title") or "",
        p.get("family") or "",
        p.get("proposed_role") or "",
        (p.get("motivation") or "")[:400],
        f"council consensus {council.get('consensus','')}",
        f"pipeline {pipeline.get('final_decision','')}",
        f"alignment {row.get('verdict_alignment','')}",
    ]).strip()
    payload = {
        "iteration_id":     row.get("iteration_id"),
        "ts":               row.get("ts"),
        "proposal_title":   p.get("title"),
        "proposal_family":  p.get("family"),
        "proposed_role":    p.get("proposed_role"),
        "consensus":        council.get("consensus"),
        "council_run_id":   council.get("run_id"),
        "pipeline_decision": pipeline.get("final_decision"),
        "verdict_alignment": row.get("verdict_alignment"),
    }
    return text, payload


def _snippet_pfh(row: dict) -> tuple[str, dict]:
    rationale = row.get("rationale") or ""
    top = row.get("top") or []
    fam_blob = " ".join(t.get("family", "") for t in top[:6])
    text = " ".join([
        "pfh suggestions",
        rationale[:600],
        fam_blob,
    ]).strip()
    payload = {
        "ts":           row.get("ts"),
        "k":            row.get("k"),
        "rationale":    rationale[:300],
        "top_families": [t.get("family") for t in top[:6]],
    }
    return text, payload


def _snippet_council(row: dict) -> tuple[str, dict]:
    p = row.get("proposal") or {}
    text = " ".join([
        p.get("title") or "",
        p.get("family") or "",
        f"consensus {row.get('consensus','')}",
        (row.get("rationale") or "")[:600],
    ]).strip()
    payload = {
        "run_id":          row.get("run_id"),
        "ts":              row.get("ts"),
        "stage":           row.get("stage"),
        "consensus":       row.get("consensus"),
        "proposal_title":  p.get("title"),
        "proposal_family": p.get("family"),
        "rationale":       (row.get("rationale") or "")[:300],
    }
    return text, payload


def _snippet_decay(row: dict) -> tuple[str, dict]:
    text = " ".join([
        f"sleeve {row.get('sleeve','')}",
        f"library {row.get('library_id','')}",
        f"alert {row.get('alert_level','')}",
        f"trailing sharpe {row.get('trailing_sharpe','')}",
        (row.get("recommendation") or "")[:200],
    ]).strip()
    payload = {
        "sleeve":          row.get("sleeve"),
        "library_id":      row.get("library_id"),
        "audit_date":      row.get("audit_date"),
        "trailing_sharpe": row.get("trailing_sharpe"),
        "alert_level":     row.get("alert_level"),
        "recommendation":  row.get("recommendation"),
    }
    return text, payload


def _snippet_materialization(meta: dict) -> tuple[str, dict]:
    val = meta.get("validation") or {}
    axes = meta.get("compose_axes") or {}
    text = " ".join([
        f"spec {meta.get('spec_id','')}",
        f"universe {axes.get('universe','')}",
        f"signal {axes.get('signal','')}",
        f"weighting {axes.get('weighting','')}",
        f"ann sharpe {val.get('observed_ann_sharpe','')}",
        f"ann vol {val.get('observed_ann_vol','')}",
    ]).strip()
    payload = {
        "spec_id":         meta.get("spec_id"),
        "materialized_at": meta.get("materialized_at"),
        "validation":      val,
        "compose_axes":    axes,
    }
    return text, payload


_LEDGER_SOURCES = {
    "l4_iterations":    ("data/research/l4_iterations.jsonl",       _snippet_l4),
    "pfh_suggestions":  ("data/research/pfh_suggestions.jsonl",     _snippet_pfh),
    "council_runs":     ("data/research/council_runs.jsonl",        _snippet_council),
    "decay_audits":     ("data/research/decay_sentinel_history.jsonl", _snippet_decay),
}


def _materialization_rows() -> Iterable[dict]:
    """Materializations live as one .meta.json per spec, not a jsonl."""
    computed_dir = _REPO_ROOT / "data" / "feature_store" / "_computed"
    if not computed_dir.is_dir():
        return
    for p in sorted(computed_dir.glob("*.meta.json")):
        try:
            yield json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue


# ── Build / refresh index ───────────────────────────────────────────


def build_index(ledger_key: str, *, max_rows: int = 5000) -> dict:
    """Build (or incrementally extend) one ledger's embedding index.

    Returns build summary: {ledger, n_rows_total, n_new_embedded,
    n_existing_kept}. Incremental: hashes existing payloads; rows whose
    hash is already in the index are skipped."""
    if ledger_key not in _LEDGER_SOURCES and ledger_key != "materializations":
        raise ValueError(f"unknown ledger_key: {ledger_key}")

    if ledger_key == "materializations":
        builder = _snippet_materialization
        rows = list(_materialization_rows())
    else:
        rel_path, builder = _LEDGER_SOURCES[ledger_key]
        path = _REPO_ROOT / rel_path
        rows = list(_iter_jsonl(path))

    # Cap to most-recent N to bound storage
    rows = rows[-max_rows:]

    existing = _load_index(ledger_key)
    known: dict[int, int] = {}
    if existing is not None:
        for i, h in enumerate(existing[0]):
            known[int(h)] = i

    snippets: list[str] = []
    payloads: list[dict] = []
    hashes: list[np.uint64] = []
    reused: list[int] = []   # indices into existing arrays

    for row in rows:
        try:
            text, payload = builder(row)
        except Exception:
            continue
        if not text.strip():
            continue
        h = _row_hash(text)
        hi = int(h)
        if hi in known:
            reused.append(known[hi])
            continue
        snippets.append(text)
        payloads.append(payload)
        hashes.append(h)

    if snippets:
        new_embeds = encode(snippets)
    else:
        new_embeds = np.zeros((0, _EMBED_DIM), dtype=np.float32)

    if existing is not None and reused:
        re_emb = existing[1][np.array(reused, dtype=np.int64)]
        re_pay = existing[2][np.array(reused, dtype=np.int64)]
        re_hash = existing[0][np.array(reused, dtype=np.int64)]
        all_emb = np.vstack([re_emb, new_embeds]) if snippets else re_emb
        all_pay = np.concatenate([re_pay, np.array(payloads, dtype=object)]) \
                   if snippets else re_pay
        all_hash = np.concatenate([re_hash, np.array(hashes, dtype=np.uint64)]) \
                    if snippets else re_hash
    else:
        all_emb = new_embeds
        all_pay = np.array(payloads, dtype=object)
        all_hash = np.array(hashes, dtype=np.uint64)

    _save_index(ledger_key, all_hash, all_emb, all_pay)

    return {
        "ledger":           ledger_key,
        "n_rows_total":     int(all_emb.shape[0]),
        "n_new_embedded":   len(snippets),
        "n_existing_kept":  len(reused),
    }


def build_all() -> list[dict]:
    """Build/refresh every known ledger. Returns one summary per."""
    out = []
    for k in list(_LEDGER_SOURCES.keys()) + ["materializations"]:
        out.append(build_index(k))
    return out


# ── T3.3 (2026-06-05 audit R1): auto-rebuild stale indices ──────────
#
# Pre-T3.3, the semantic index was rebuilt only on explicit `build_all`
# call (cron / manual). Audit found indices last rebuilt 2026-06-02
# while underlying jsonls had appended through 2026-05-31+: 3 days of
# new research events were invisible to the semantic retriever, falling
# back to keyword-only and missing synonym matches.
#
# T3.3 strategy: on `search_all` (the hot path), check each ledger's
# source-jsonl mtime vs its index .npz mtime. If source is newer, fire
# an incremental rebuild (the build_index path dedupes by hash, so only
# truly-new rows are encoded — typically sub-second on CPU).
#
# Throttle: each ledger refreshes at most once per _MIN_REFRESH_INTERVAL.
# Prevents a tight retrieval loop from re-checking mtimes 100x/sec.
# Disable globally via env EMBEDDINGS_AUTO_REBUILD=0 (e.g. for tests).

_MIN_REFRESH_INTERVAL_SEC = 60.0     # don't rebuild more often than this per ledger
_LAST_REFRESH_TS: "dict[str, float]" = {}


def _source_mtime(ledger_key: str) -> float:
    """Wall-clock mtime of the ledger's underlying source. Returns
    +inf if source missing (we won't trigger a rebuild for nothing)."""
    if ledger_key == "materializations":
        d = _REPO_ROOT / "data" / "feature_store" / "_computed"
        if not d.is_dir():
            return float("inf")
        # max mtime across all .meta.json files
        try:
            return max((p.stat().st_mtime for p in d.glob("*.meta.json")),
                       default=0.0)
        except Exception:
            return float("inf")
    rel = _LEDGER_SOURCES.get(ledger_key, (None, None))[0]
    if not rel:
        return float("inf")
    p = _REPO_ROOT / rel
    return p.stat().st_mtime if p.is_file() else float("inf")


def _index_mtime(ledger_key: str) -> float:
    """Wall-clock mtime of the saved index npz. 0.0 means "missing/older
    than anything", so a stale check will trigger a rebuild."""
    p = _INDEX_DIR / f"{ledger_key}.npz"
    return p.stat().st_mtime if p.is_file() else 0.0


def is_index_stale(ledger_key: str) -> bool:
    """True iff the index is missing OR its source jsonl is newer."""
    return _source_mtime(ledger_key) > _index_mtime(ledger_key)


def auto_refresh_if_stale(ledger_key: str) -> "dict | None":
    """Throttled stale-check + incremental rebuild. Returns the
    build_index summary if a rebuild ran, None otherwise.

    Respects EMBEDDINGS_AUTO_REBUILD=0 env (returns None unconditionally
    when set, useful for tests).
    """
    if os.environ.get("EMBEDDINGS_AUTO_REBUILD", "1") == "0":
        return None
    import time as _t
    now = _t.time()
    last = _LAST_REFRESH_TS.get(ledger_key, 0.0)
    if now - last < _MIN_REFRESH_INTERVAL_SEC:
        return None
    if not is_index_stale(ledger_key):
        # Still mark the throttle so we don't re-mtime-check on every call
        _LAST_REFRESH_TS[ledger_key] = now
        return None
    try:
        summary = build_index(ledger_key)
        _LAST_REFRESH_TS[ledger_key] = now
        return summary
    except Exception:
        # Don't let an embed failure break /ask retrieval
        _LAST_REFRESH_TS[ledger_key] = now
        return None


# ── Semantic retrieval ──────────────────────────────────────────────


def search(ledger_key: str, query: str, top_k: int = 6) -> list[dict]:
    """Cosine-similarity top-K payloads for a single ledger. Returns
    list of payload dicts annotated with `_semantic_score`. Empty list
    if index is missing — caller should fall back to keyword retrieval."""
    idx = _load_index(ledger_key)
    if idx is None:
        return []
    _hashes, embeds, payloads = idx
    if embeds.shape[0] == 0:
        return []
    qv = encode(query)
    sims = embeds @ qv
    order = np.argsort(-sims)[:top_k]
    out: list[dict] = []
    for i in order:
        score = float(sims[int(i)])
        payload = dict(payloads[int(i)])
        payload["_semantic_score"] = round(score, 4)
        out.append(payload)
    return out


def search_all(query: str, top_k: int = 6) -> dict:
    """Top-K per ledger. Convenience for the /ask retrieval call site.

    T3.3: before searching, throttled auto-refresh runs per ledger so
    indices stay fresh as the jsonls are appended. Disable globally
    via EMBEDDINGS_AUTO_REBUILD=0 (tests). Refresh interval throttled
    to once per _MIN_REFRESH_INTERVAL_SEC per ledger.
    """
    keys = list(_LEDGER_SOURCES.keys()) + ["materializations"]
    for k in keys:
        auto_refresh_if_stale(k)
    return {k: search(k, query, top_k=top_k) for k in keys}


def index_status() -> dict:
    """Diagnostic — show what's indexed. For UI / CLI."""
    out = {}
    for k in list(_LEDGER_SOURCES.keys()) + ["materializations"]:
        idx = _load_index(k)
        out[k] = {
            "indexed":    idx is not None,
            "n_rows":     int(idx[1].shape[0]) if idx is not None else 0,
            "embed_dim":  int(idx[1].shape[1]) if idx is not None and idx[1].size else None,
        }
    return out
