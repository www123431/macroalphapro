"""
scripts/run_v4_d3_lock_znorm_params.py — W3 D3 (2026-05-08).

Pre-registration: docs/spec_multivariate_msm_v4_narrative.md §2.7.3
Spec id: 47

One-time procedure: read in-sample 1994-2018 raw_score from D2c cache, compute
(μ_in_sample, σ_in_sample), patch engine/narrative_classifier.py to set the
module constants `_RAW_SCORE_INSAMPLE_MEAN` / `_STD`, then call amend_spec
with kind='clarification' to record the locked values in the amendment log.

Forbidden per spec §6 / §2.7.3:
  • Re-running this on OOS data (the script enforces in_sample_end=2018-12-31)
  • Recomputing on a smaller in-sample subset
  • Manually overriding the locked values

After this runs:
  • narrative_classifier.compute_narrative_score(text) becomes callable
  • engine/narrative_classifier.py amendment log records the lock
  • Spec amend_log records (μ, σ) values + cache size + run timestamp
"""
from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_PARQUET = ROOT / "data" / "fomc_statements" / "cache.parquet"
NC_PATH = ROOT / "engine" / "narrative_classifier.py"
SPEC_PATH = ROOT / "docs" / "spec_multivariate_msm_v4_narrative.md"

IN_SAMPLE_END = pd.Timestamp("2018-12-31")
IN_SAMPLE_START = pd.Timestamp("1994-01-01")


def compute_in_sample_params(df: pd.DataFrame) -> tuple[float, float, int]:
    """Return (mean, std, n) on raw_score in [1994-01-01, 2018-12-31]."""
    in_sample = df[
        (df["date"] >= IN_SAMPLE_START) & (df["date"] <= IN_SAMPLE_END)
    ]
    if len(in_sample) < 50:
        raise RuntimeError(
            f"D3: only {len(in_sample)} in-sample obs — < 50 minimum for "
            f"stable z-norm. Run D2c first or investigate cache."
        )
    mu = float(in_sample["raw_score"].mean())
    sigma = float(in_sample["raw_score"].std(ddof=1))
    if sigma <= 0:
        raise RuntimeError(f"D3: in-sample σ = {sigma:.6e} ≤ 0 — degenerate.")
    return mu, sigma, len(in_sample)


def patch_narrative_classifier(mu: float, sigma: float, n: int) -> None:
    """Replace the two None placeholders in narrative_classifier.py with locked values."""
    src = NC_PATH.read_text(encoding="utf-8")

    # Replace _RAW_SCORE_INSAMPLE_MEAN
    pat_mean = re.compile(
        r"_RAW_SCORE_INSAMPLE_MEAN:\s*Optional\[float\]\s*=\s*None"
    )
    new_mean = (
        f"_RAW_SCORE_INSAMPLE_MEAN: Optional[float] = "
        f"{mu:.10e}  # locked W3 D3 2026-05-08, n={n}"
    )
    if not pat_mean.search(src):
        raise RuntimeError("D3: _RAW_SCORE_INSAMPLE_MEAN placeholder not found in module")
    src = pat_mean.sub(new_mean, src, count=1)

    # Replace _RAW_SCORE_INSAMPLE_STD
    pat_std = re.compile(
        r"_RAW_SCORE_INSAMPLE_STD:\s*Optional\[float\]\s*=\s*None"
    )
    new_std = (
        f"_RAW_SCORE_INSAMPLE_STD: Optional[float] = "
        f"{sigma:.10e}   # locked W3 D3 2026-05-08, n={n}"
    )
    if not pat_std.search(src):
        raise RuntimeError("D3: _RAW_SCORE_INSAMPLE_STD placeholder not found in module")
    src = pat_std.sub(new_std, src, count=1)

    NC_PATH.write_text(src, encoding="utf-8")
    print(f"  patched {NC_PATH}")


def amend_spec_with_lock(mu: float, sigma: float, n: int) -> int:
    from engine.preregistration import amend_spec
    reason = (
        f"D3 z-norm parameter lock: in-sample 1994-01..2018-12 raw_score "
        f"computed from {n} verified FOMC statements (D2c cache). "
        f"μ_in_sample = {mu:.10e}, σ_in_sample = {sigma:.10e}. "
        f"engine/narrative_classifier.py module constants patched in-place; "
        f"compute_narrative_score() now callable. Per spec §6 + §2.7.3, "
        f"these values are FROZEN — re-computing on OOS data forbidden."
    )
    return amend_spec(
        path=str(SPEC_PATH),
        kind="clarification",
        reason=reason,
    )


def main():
    if not CACHE_PARQUET.exists():
        print(f"[error] cache.parquet not found at {CACHE_PARQUET}")
        print("Run scripts/run_v4_universe_feasibility.py (D2c) first.")
        sys.exit(1)

    df = pd.read_parquet(CACHE_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df)} statements from D2c cache")

    mu, sigma, n = compute_in_sample_params(df)
    print(f"\nIn-sample 1994-2018: n={n}")
    print(f"  μ_in_sample = {mu:.10e}")
    print(f"  σ_in_sample = {sigma:.10e}")

    print("\nPatching engine/narrative_classifier.py ...")
    patch_narrative_classifier(mu, sigma, n)

    print("\nVerifying compute_narrative_score is callable ...")
    # Force-reload to pick up the patched constants
    import importlib
    import engine.narrative_classifier as nc
    importlib.reload(nc)
    sample = nc.compute_narrative_score("The Committee will tighten policy")
    print(f"  test sample z-score: {sample:+.4f}  (callable ✓)")

    print("\nRecording amend_spec ...")
    sid = amend_spec_with_lock(mu, sigma, n)
    print(f"  amend_spec id={sid}")

    print("\n✓ D3 lock complete. Next: D4 v4 regime path.")


if __name__ == "__main__":
    main()
