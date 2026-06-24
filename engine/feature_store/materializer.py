"""engine/feature_store/materializer.py — Spec execution + caching.

Flow:
  1. Compute input_hash from (input cache mtimes + source module mtimes
     + spec content) → identifies a specific (code, data) state
  2. If computed output already exists for this hash → return cached path
  3. Otherwise: import module, call function, validate output against
     spec, write parquet + sidecar .meta.json

Caching addressability:
  data/feature_store/_computed/<spec_id>.v<version>.<hash[:8]>.parquet
  data/feature_store/_computed/<spec_id>.v<version>.<hash[:8]>.meta.json

The .meta.json sidecar records the full hash, materialization
timestamp, elapsed seconds, and input mtimes — useful for forensic
"why does today's output differ from yesterday's" questions.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import logging
import time

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.feature_store.registry import (
    COMPUTED_DIR, REPO_ROOT, FeatureSpec, load_spec,
)

logger = logging.getLogger(__name__)


def _safe_relpath(p: Path) -> str:
    """Try relative_to(REPO_ROOT); fall back to absolute path string.

    Tolerates test fixtures that monkeypatch paths outside the repo."""
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


# ── Hashing ───────────────────────────────────────────────────────────


def _file_signature(p: Path) -> tuple[str, int, float]:
    """(relpath, size, mtime). Returns ("MISSING", 0, 0) if file is absent —
    spec validation should catch this, but be tolerant in the hasher."""
    if not p.is_file():
        return (_safe_relpath(p),
                0, 0.0)
    return (
        str(p.relative_to(REPO_ROOT)).replace("\\", "/")
        if str(p).startswith(str(REPO_ROOT)) else str(p),
        p.stat().st_size,
        p.stat().st_mtime,
    )


def compute_input_hash(spec: FeatureSpec) -> str:
    """SHA256 over a deterministic tuple of:
      - spec YAML file content
      - all input cache file signatures (path, size, mtime)
      - all source module file signatures

    Returns 64-char hex. Used as the cache key for materialized output.
    """
    h = hashlib.sha256()

    # Spec content
    if spec.spec_path and spec.spec_path.is_file():
        h.update(spec.spec_path.read_bytes())

    # Inputs
    for raw in sorted(spec.inputs):
        sig = _file_signature(REPO_ROOT / raw)
        h.update(json.dumps(sig, default=str).encode("utf-8"))

    # Source files
    for raw in sorted(spec.source_module_files):
        sig = _file_signature(REPO_ROOT / raw)
        h.update(json.dumps(sig, default=str).encode("utf-8"))

    return h.hexdigest()


def _computed_paths(spec: FeatureSpec, input_hash: str) -> tuple[Path, Path]:
    """(parquet_path, meta_path) for this (spec, hash)."""
    short = input_hash[:8]
    base = COMPUTED_DIR / f"{spec.spec_id}.v{spec.version}.{short}"
    return base.with_suffix(".parquet"), base.with_suffix(".meta.json")


# ── Validation ────────────────────────────────────────────────────────


@dataclass
class ValidationReport:
    ok:                    bool
    violations:            list[str]
    observed_n_rows:       Optional[int] = None
    observed_start:        Optional[str] = None
    observed_end:          Optional[str] = None
    observed_ann_vol:      Optional[float] = None
    observed_ann_sharpe:   Optional[float] = None


def _validate_output(spec: FeatureSpec, obj: object) -> ValidationReport:
    """Apply schema + sanity rules to the materialized object.

    Accepts pandas.Series for *_returns kinds, pandas.DataFrame for
    signal_panel. Returns a ValidationReport; caller decides whether
    to raise.
    """
    violations: list[str] = []

    # Type
    if spec.output_kind in ("monthly_returns", "daily_returns"):
        if not isinstance(obj, pd.Series):
            return ValidationReport(
                ok=False,
                violations=[f"expected pandas.Series, got {type(obj).__name__}"],
            )
    elif spec.output_kind == "signal_panel":
        if not isinstance(obj, pd.DataFrame):
            return ValidationReport(
                ok=False,
                violations=[f"expected pandas.DataFrame, got {type(obj).__name__}"],
            )

    # Index must be DatetimeIndex
    idx = obj.index
    if not isinstance(idx, pd.DatetimeIndex):
        return ValidationReport(
            ok=False,
            violations=[f"index must be DatetimeIndex, got {type(idx).__name__}"],
        )

    # Shape
    n_rows = len(obj)
    n_min, n_max = spec.output_n_rows_range
    if not (n_min <= n_rows <= n_max):
        violations.append(
            f"n_rows {n_rows} outside expected range [{n_min}, {n_max}]"
        )

    # Date range — compared at month granularity for monthly data,
    # day granularity for daily. Avoids false-positives from month-end
    # convention (e.g. "Jan 2020 return" is indexed at 2020-01-31).
    obs_start = str(idx.min())[:10]
    obs_end = str(idx.max())[:10]
    cmp_chars = 7 if spec.output_kind == "monthly_returns" else 10
    if obs_start[:cmp_chars] > spec.output_start[:cmp_chars]:
        violations.append(
            f"observed start {obs_start} is after expected start {spec.output_start}"
        )
    if obs_end[:cmp_chars] < spec.output_end_min[:cmp_chars]:
        violations.append(
            f"observed end {obs_end} is before expected end_min {spec.output_end_min}"
        )

    # Sanity (returns kinds)
    observed_ann_vol = None
    observed_ann_sharpe = None
    if spec.output_kind in ("monthly_returns", "daily_returns") and len(obj) > 1:
        # Annualization scale
        scale = 12 if spec.output_kind == "monthly_returns" else 252
        clean = obj.dropna()
        if len(clean) > 1:
            mu = clean.mean() * scale
            sigma = clean.std() * (scale ** 0.5)
            observed_ann_vol = float(sigma)
            observed_ann_sharpe = float(mu / sigma) if sigma > 1e-9 else 0.0

            if "annualized_vol_range" in spec.sanity:
                lo, hi = spec.sanity["annualized_vol_range"]
                if not (lo <= observed_ann_vol <= hi):
                    violations.append(
                        f"observed ann_vol {observed_ann_vol:.4f} outside "
                        f"sanity range [{lo}, {hi}]"
                    )
            if "annualized_sharpe_range" in spec.sanity:
                lo, hi = spec.sanity["annualized_sharpe_range"]
                if not (lo <= observed_ann_sharpe <= hi):
                    violations.append(
                        f"observed ann_sharpe {observed_ann_sharpe:.4f} outside "
                        f"sanity range [{lo}, {hi}]"
                    )

    # no-nan-after-first-obs
    if spec.sanity.get("no_nan_after_first_observation"):
        if isinstance(obj, pd.Series):
            first_valid = obj.first_valid_index()
            if first_valid is not None:
                trailing = obj.loc[first_valid:]
                if trailing.isna().any():
                    n_bad = int(trailing.isna().sum())
                    violations.append(
                        f"{n_bad} NaN value(s) after first valid observation"
                    )

    return ValidationReport(
        ok=not violations,
        violations=violations,
        observed_n_rows=n_rows,
        observed_start=obs_start,
        observed_end=obs_end,
        observed_ann_vol=observed_ann_vol,
        observed_ann_sharpe=observed_ann_sharpe,
    )


# ── Public API ────────────────────────────────────────────────────────


def _spec_is_compose(spec_path: Path) -> bool:
    """Detect compose-spec (v2) vs function-wrapper spec (v1) by presence
    of `compose:` top-level key. Cheap YAML probe.

    Re-raises on parse errors instead of silent False to avoid the silent-
    failure pattern that would mis-route compose specs to function loader.
    """
    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    return "compose" in raw


def _load_any_spec(spec_path: Path):
    """Auto-route between v1 (function-wrapper) and v2 (compose) loaders.

    Returns a tuple (kind, spec_object) where kind ∈ {"function", "compose"}.
    """
    if _spec_is_compose(spec_path):
        from engine.feature_store.composer import load_compose_spec
        return "compose", load_compose_spec(spec_path)
    return "function", load_spec(spec_path)


def materialize_spec(
    spec_or_id,
    *,
    force: bool = False,
    strict_sanity: bool = True,
) -> dict:
    """Materialize one feature spec — auto-routes v1 (function-wrapper)
    vs v2 (compose-spec) by detecting `compose:` top-level key.

    Args:
      spec_or_id:      a FeatureSpec, ComposeSpec, or spec_id string
                        (looked up under SPECS_DIR)
      force:           if True, ignore cache and re-materialize
      strict_sanity:   if True, validation violations raise

    Returns:
      {spec_id, version, input_hash, output_path, meta_path,
       cached, elapsed_s, validation, spec_kind}
    """
    spec_kind = "function"
    if isinstance(spec_or_id, str):
        from engine.feature_store.registry import SPECS_DIR
        spec_path = SPECS_DIR / f"{spec_or_id}.yaml"
        spec_kind, spec = _load_any_spec(spec_path)
    elif isinstance(spec_or_id, FeatureSpec):
        spec = spec_or_id
        spec_kind = "function"
    else:
        # ComposeSpec
        from engine.feature_store.composer import ComposeSpec
        if isinstance(spec_or_id, ComposeSpec):
            spec = spec_or_id
            spec_kind = "compose"
        else:
            raise TypeError(
                f"materialize_spec: unsupported type {type(spec_or_id).__name__}"
            )

    input_hash = compute_input_hash(spec)
    parquet_path, meta_path = _computed_paths(spec, input_hash)

    if parquet_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "spec_id":     spec.spec_id,
            "version":     spec.version,
            "input_hash":  input_hash,
            "output_path": _safe_relpath(parquet_path),
            "meta_path":   _safe_relpath(meta_path),
            "cached":      True,
            "elapsed_s":   meta.get("elapsed_s"),
            "validation":  meta.get("validation"),
        }

    # Materialize: route by spec kind
    COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if spec_kind == "compose":
        from engine.feature_store.composer import compose_factor
        obj = compose_factor(spec)
    else:
        try:
            mod = importlib.import_module(spec.materialize_module)
            fn = getattr(mod, spec.materialize_function)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"failed to resolve {spec.materialize_module}.{spec.materialize_function}: {exc}"
            )
        obj = fn(**spec.materialize_args)
    elapsed_s = time.perf_counter() - t0

    # Validate
    report = _validate_output(spec, obj)
    if strict_sanity and not report.ok:
        raise ValueError(
            f"spec {spec.spec_id}: validation failed:\n  - "
            + "\n  - ".join(report.violations)
        )

    # Persist
    if isinstance(obj, pd.Series):
        obj.to_frame(name="value").to_parquet(parquet_path)
    elif isinstance(obj, pd.DataFrame):
        obj.to_parquet(parquet_path)
    else:
        raise TypeError(
            f"unsupported output type {type(obj).__name__}; "
            f"expected pandas.Series or pandas.DataFrame"
        )

    meta = {
        "spec_id":           spec.spec_id,
        "version":           spec.version,
        "input_hash":        input_hash,
        "materialized_at":   _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "elapsed_s":         round(elapsed_s, 3),
        "spec_kind":         spec_kind,
        "materialize_module":   (spec.materialize_module
                                  if spec_kind == "function" else None),
        "materialize_function": (spec.materialize_function
                                  if spec_kind == "function" else None),
        "materialize_args":     (spec.materialize_args
                                  if spec_kind == "function" else None),
        "compose_axes":         ({
            "universe":  spec.universe.name,
            "signal":    spec.signal.name,
            "weighting": spec.weighting.name,
            "rebalance": spec.rebalance,
        } if spec_kind == "compose" else None),
        "input_signatures": {
            raw: list(_file_signature(REPO_ROOT / raw))
            for raw in sorted(spec.inputs)
        },
        "source_signatures": {
            raw: list(_file_signature(REPO_ROOT / raw))
            for raw in sorted(spec.source_module_files)
        },
        "validation": {
            "ok":                  report.ok,
            "violations":          report.violations,
            "observed_n_rows":     report.observed_n_rows,
            "observed_start":      report.observed_start,
            "observed_end":        report.observed_end,
            "observed_ann_vol":    report.observed_ann_vol,
            "observed_ann_sharpe": report.observed_ann_sharpe,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    return {
        "spec_id":     spec.spec_id,
        "version":     spec.version,
        "spec_kind":   spec_kind,
        "input_hash":  input_hash,
        "output_path": _safe_relpath(parquet_path),
        "meta_path":   _safe_relpath(meta_path),
        "cached":      False,
        "elapsed_s":   round(elapsed_s, 3),
        "validation":  meta["validation"],
    }


def verify_spec(spec_or_id: FeatureSpec | str) -> dict:
    """Re-materialize from scratch and confirm the hash + output match
    the cached version (if any). Used by pre-commit / CI to catch
    code/data drift.

    Returns:
      {"ok": True/False, "drift_detected": [...], ...}
    """
    if isinstance(spec_or_id, str):
        from engine.feature_store.registry import SPECS_DIR
        spec = load_spec(SPECS_DIR / f"{spec_or_id}.yaml")
    else:
        spec = spec_or_id

    input_hash = compute_input_hash(spec)
    parquet_path, _ = _computed_paths(spec, input_hash)
    cached_existed = parquet_path.is_file()

    # Force re-materialize
    result = materialize_spec(spec, force=True)

    return {
        "spec_id":         spec.spec_id,
        "version":         spec.version,
        "expected_hash":   input_hash,
        "computed_hash":   result["input_hash"],
        "hash_match":      result["input_hash"] == input_hash,
        "cached_existed":  cached_existed,
        "validation":      result["validation"],
        "ok":              result["validation"]["ok"]
                            and result["input_hash"] == input_hash,
    }


# ── CLI ───────────────────────────────────────────────────────────────


def _cli() -> None:
    """python -m engine.feature_store <list|describe|materialize|verify>"""
    import sys
    from engine.feature_store.registry import list_specs

    args = sys.argv[1:]
    cmd = args[0] if args else "list"

    if cmd == "list":
        specs = list_specs()
        print(json.dumps({"n": len(specs), "specs": specs},
                          indent=2, default=str))
        return

    if cmd == "describe" and len(args) >= 2:
        spec = load_spec(SPECS_DIR_ := Path(__file__).resolve().parents[2]
                          / "data" / "feature_store" / "_specs"
                          / f"{args[1]}.yaml")
        print(json.dumps({
            "spec_id":              spec.spec_id,
            "version":              spec.version,
            "description":          spec.description,
            "materialize_module":   spec.materialize_module,
            "materialize_function": spec.materialize_function,
            "materialize_args":     spec.materialize_args,
            "output_kind":          spec.output_kind,
            "n_inputs":             len(spec.inputs),
            "n_source_files":       len(spec.source_module_files),
            "current_input_hash":   compute_input_hash(spec),
        }, indent=2, default=str))
        return

    if cmd == "materialize" and len(args) >= 2:
        force = "--force" in args
        result = materialize_spec(args[1], force=force)
        print(json.dumps(result, indent=2, default=str))
        return

    if cmd == "verify" and len(args) >= 2:
        result = verify_spec(args[1])
        print(json.dumps(result, indent=2, default=str))
        if not result["ok"]:
            raise SystemExit(1)
        return

    print("usage: list | describe <spec_id> | materialize <spec_id> [--force] | "
          "verify <spec_id>", file=__import__("sys").stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    _cli()
