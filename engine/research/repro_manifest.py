"""engine/research/repro_manifest.py — Phase 1 P0b of loop robustness.

Reproducibility Manifest: every pipeline audit captures the input state
sufficient to reproduce the output. Without this, audits drift silently
as data caches age + code mutates.

What it captures (senior-quant minimum institutional standard):
  git_commit_sha           code state at audit time (or 'uncommitted-...')
  working_dir_dirty        bool: are there uncommitted changes?
  data_file_manifest       {path: {mtime, size_bytes, sha256_first_2mb}}
                              for key cached data files used
  python_version           sys.version_info
  library_versions         pandas / numpy / scipy / openai / etc.
  run_timestamp            UTC ISO format
  pipeline_config          {phase, n_splits, ewma_lambda, etc.}
  output_hash              SHA-256 of stringified output dict
                              (used to verify deterministic reruns)

Per [[feedback-loop-is-robustness-doctrine-2026-05-31]] Phase 1 P0b.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard cached files we care about for audit reproducibility.
# Extended as we audit more sleeves.
STANDARD_DATA_FILES = [
    "data/cache/crsp_hist_daily_ret.parquet",
    "data/cache/crsp_vwretd_daily.parquet",
    "data/cache/_pead_ts_panel_2014_2023.parquet",
    "data/cache/_compustat_funda_for_barra.parquet",
    "data/cache/_compustat_company_gics.parquet",
    "data/cache/_compustat_funda_sich_pit.parquet",
    "data/cache/_barra_lite_factors_phase3.parquet",
    "data/cache/_dpead_recon_base.parquet",
    "data/cache/_dpead_sector_neutral_pit.parquet",
    "data/cache/_vix_spx_daily.parquet",
]


@dataclasses.dataclass
class ReproManifest:
    git_commit_sha:      str
    working_dir_dirty:   bool
    data_file_manifest:  dict
    python_version:      str
    library_versions:    dict
    run_timestamp:       str
    pipeline_config:     dict
    output_hash:         str | None = None    # set AFTER pipeline produces output

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── git state ────────────────────────────────────────────────────────────

def _git_commit_sha() -> tuple[str, bool]:
    """Return (sha, working_dir_dirty)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).strip())
        return sha, dirty
    except Exception:
        return "uncommitted-or-not-a-repo", True


# ── data file checksums ─────────────────────────────────────────────────

def _data_file_manifest(paths: list[str] | None = None) -> dict:
    """For each path, record mtime + size + sha256 of first 2MB."""
    paths = paths or STANDARD_DATA_FILES
    out: dict = {}
    for rel_path in paths:
        full = REPO_ROOT / rel_path
        if not full.exists():
            out[rel_path] = {"exists": False}
            continue
        stat = full.stat()
        # SHA-256 of first 2MB (cheap; full file may be very large)
        hasher = hashlib.sha256()
        try:
            with open(full, "rb") as f:
                hasher.update(f.read(2 * 1024 * 1024))
            checksum = hasher.hexdigest()[:16]
        except Exception:
            checksum = None
        out[rel_path] = {
            "exists": True,
            "mtime": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "size_bytes": stat.st_size,
            "sha256_first_2mb": checksum,
        }
    return out


# ── library versions ─────────────────────────────────────────────────────

def _library_versions() -> dict:
    libs = ["pandas", "numpy", "scipy", "openai", "tomli", "yaml",
            "yfinance", "pandas_datareader"]
    out: dict = {}
    for name in libs:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "unknown")
            out[name] = str(version)
        except Exception:
            out[name] = "not_installed"
    return out


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


# ── output hash ──────────────────────────────────────────────────────────

def hash_output(output: Any) -> str:
    """Stable SHA-256 of the output dict. Used to verify deterministic
    reruns: 2 audits with same input + same manifest should produce
    same output_hash."""
    try:
        if hasattr(output, "to_dict"):
            d = output.to_dict()
        elif isinstance(output, dict):
            d = output
        else:
            d = {"output": str(output)}
        canonical = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except Exception as exc:
        logger.warning("output hash failed: %s", exc)
        return "hash_failed"


# ── public entry ─────────────────────────────────────────────────────────

def build_manifest(pipeline_config: dict | None = None,
                       data_paths: list[str] | None = None) -> ReproManifest:
    """Build a fresh manifest at audit start. Caller fills in
    output_hash after pipeline produces output."""
    sha, dirty = _git_commit_sha()
    return ReproManifest(
        git_commit_sha=sha,
        working_dir_dirty=dirty,
        data_file_manifest=_data_file_manifest(data_paths),
        python_version=_python_version(),
        library_versions=_library_versions(),
        run_timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        pipeline_config=pipeline_config or {},
    )


def verify_reproducible(manifest_a: ReproManifest,
                              manifest_b: ReproManifest) -> dict:
    """Compare two manifests + check whether output_hash matches.
    Returns diagnostic dict explaining any differences."""
    diffs: dict = {}
    if manifest_a.git_commit_sha != manifest_b.git_commit_sha:
        diffs["git_commit_sha"] = (manifest_a.git_commit_sha,
                                       manifest_b.git_commit_sha)
    if manifest_a.python_version != manifest_b.python_version:
        diffs["python_version"] = (manifest_a.python_version,
                                        manifest_b.python_version)
    # Library version diffs
    lib_diffs = {}
    for lib in set(manifest_a.library_versions) | set(manifest_b.library_versions):
        v_a = manifest_a.library_versions.get(lib)
        v_b = manifest_b.library_versions.get(lib)
        if v_a != v_b:
            lib_diffs[lib] = (v_a, v_b)
    if lib_diffs:
        diffs["library_versions"] = lib_diffs
    # Data file diffs (mtime / checksum)
    file_diffs = {}
    files = set(manifest_a.data_file_manifest) | set(manifest_b.data_file_manifest)
    for f in files:
        a = manifest_a.data_file_manifest.get(f, {})
        b = manifest_b.data_file_manifest.get(f, {})
        if a.get("sha256_first_2mb") != b.get("sha256_first_2mb"):
            file_diffs[f] = {
                "a_checksum": a.get("sha256_first_2mb"),
                "b_checksum": b.get("sha256_first_2mb"),
            }
    if file_diffs:
        diffs["data_files_changed"] = file_diffs
    output_match = (manifest_a.output_hash == manifest_b.output_hash) \
                       if (manifest_a.output_hash and manifest_b.output_hash) else None
    return {
        "fully_reproducible": (not diffs and output_match),
        "output_hash_match": output_match,
        "differences": diffs,
    }


if __name__ == "__main__":
    # Quick demo / smoke
    m = build_manifest(pipeline_config={"phase": 3, "n_splits": 2})
    print(json.dumps(m.to_dict(), indent=2, default=str))
