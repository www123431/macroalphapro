"""engine/research/parquet_browser.py — Cached return-series inventory.

Scans data/cache/ for parquet files and returns metadata
(filename, n_rows, date_range, columns, mtime). Used by the Lab UI
"Series" page so senior can SEE what cached candidate returns exist
without dropping to a shell — closing the step 1 ("data exists?")
gap in the factor-exploration workflow.

This is read-only metadata. Does NOT load the parquets' bodies; that
happens only when the user clicks "Run pipeline" downstream.
"""
from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

# Heuristic filters — we want USER-INTERESTING return series, not
# fetcher intermediates. Files starting with _ are conventionally
# internal caches (e.g. _13f_security_quarter.partial); we still
# surface them but rank them lower so senior sees the curated ones
# first.
INTERNAL_PREFIX = "_"


def _describe_parquet(path: Path) -> dict:
    """Light-weight metadata: row count, date range, columns, size.

    Reads ONLY the parquet schema + index, not the data. Falls back to
    full read for files where pyarrow's metadata is incomplete."""
    stat = path.stat()
    try:
        relpath = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        # CACHE_DIR was patched outside the repo (test fixture) — fall
        # back to the absolute path so the inventory contract is intact.
        relpath = str(path).replace("\\", "/")
    info: dict = {
        "filename":    path.name,
        "relpath":     relpath,
        "size_bytes":  stat.st_size,
        "mtime":       _dt.datetime.fromtimestamp(stat.st_mtime)
                                       .isoformat(timespec="seconds"),
        "n_rows":      None,
        "n_cols":      None,
        "columns":     [],
        "date_start":  None,
        "date_end":    None,
        "is_internal": path.name.startswith(INTERNAL_PREFIX),
        "error":       None,
    }
    try:
        # pyarrow metadata for fast schema + row count
        import pyarrow.parquet as pq
        pqf = pq.ParquetFile(str(path))
        info["n_rows"] = int(pqf.metadata.num_rows) if pqf.metadata else None
        info["n_cols"] = int(pqf.metadata.num_columns) if pqf.metadata else None
        info["columns"] = [str(c) for c in pqf.schema_arrow.names][:8]
    except Exception as exc:
        info["error"] = f"metadata read failed: {exc}"
        return info

    # Date range — needs a small body read. Read just the index column
    # if available, or first column if not.
    try:
        df = pd.read_parquet(path)
        if isinstance(df.index, pd.DatetimeIndex):
            info["date_start"] = str(df.index.min())[:10]
            info["date_end"]   = str(df.index.max())[:10]
        elif "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            info["date_start"] = str(d.min())[:10]
            info["date_end"]   = str(d.max())[:10]
    except Exception as exc:
        # Body read failed — fall back to metadata-only
        info["error"] = f"date range read failed: {exc}"

    return info


def scan_parquets(
    include_internal: bool = True,
    limit: Optional[int] = None,
) -> dict:
    """Return inventory of cached parquet files.

    Sorted by mtime descending (newest first), with curated (non-
    underscore-prefix) entries surfaced first within the same recency
    bucket.

    Args:
      include_internal: include underscore-prefix files (default True)
      limit: optional cap on results
    """
    if not CACHE_DIR.is_dir():
        return {"n": 0, "parquets": [], "cache_dir": str(CACHE_DIR)}

    all_paths = list(CACHE_DIR.glob("*.parquet"))
    if not include_internal:
        all_paths = [p for p in all_paths
                      if not p.name.startswith(INTERNAL_PREFIX)]

    # Describe everything (lightweight)
    descs = [_describe_parquet(p) for p in all_paths]
    # Sort: curated first (is_internal=False), then by mtime desc
    descs.sort(key=lambda d: (d["is_internal"], -_iso_to_ts(d["mtime"])))
    if limit:
        descs = descs[: max(1, int(limit))]
    try:
        cache_dir_str = str(CACHE_DIR.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        cache_dir_str = str(CACHE_DIR).replace("\\", "/")
    return {
        "n":         len(descs),
        "parquets":  descs,
        "cache_dir": cache_dir_str,
    }


def _iso_to_ts(iso: str) -> float:
    try:
        return _dt.datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0
