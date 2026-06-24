"""
scripts/run_v4_d2c_diagnostic.py — W3 D2c follow-up diagnostic (2026-05-08).

Pre-test audit augmenting docs/spec_multivariate_msm_v4_narrative.md §3.8.

Runs AFTER scripts/run_v4_universe_feasibility.py finishes; reads the cached
data/fomc_statements/cache.parquet (no network).

Adds two diagnostic blocks to the §3.8 verdict report:

  • Cross-era z-norm stability (concern #3 from pre-test audit):
      - mean(raw_score) and std(raw_score) per era
      - era-wise z-shift = (mean_era - pooled_mean) / pooled_std
      - flag if any |z-shift| > 1.0 → potential structural break in pooled z-norm
      - HMM EM might mis-classify era boundaries as regime transitions

  • OOS forward-fill chain length (concern #4):
      - per OOS month: chain_length = 0 if meeting in [t-30, t] else 1 + prior
      - report max / mean / fraction(chain ≥ 2) over 2019-01..2024-12
      - long fill chains = stale narrative, weakening the "forward-looking" claim

Both are descriptive — they do NOT FAIL the §3.8 gate (per spec). They are
documented in verdict report so D5 hypothesis test interpretation has the
right context (e.g. "if v4 verdict is INSUFFICIENT, fill-chain stats may
explain the small effect").
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.narrative_classifier import aggregate_monthly  # noqa: E402

CACHE_PARQUET = ROOT / "data" / "fomc_statements" / "cache.parquet"
VERDICT_PATH = ROOT / "data" / "multivariate_msm_v4" / "d2c_universe_feasibility.txt"

IN_SAMPLE_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")
OOS_END = pd.Timestamp("2024-12-31")


def cross_era_stability(df: pd.DataFrame) -> dict:
    """Compute per-era raw_score mean/std + z-shift relative to pooled stats."""
    in_sample = df[df["date"] < IN_SAMPLE_END]
    if in_sample.empty:
        return {"status": "NO_DATA"}

    pooled_mean = float(in_sample["raw_score"].mean())
    pooled_std = float(in_sample["raw_score"].std(ddof=1))

    era_stats: list[dict] = []
    max_abs_zshift = 0.0
    for era, era_df in in_sample.groupby("era", sort=False):
        n = len(era_df)
        if n < 5:
            continue
        m = float(era_df["raw_score"].mean())
        s = float(era_df["raw_score"].std(ddof=1))
        z_shift = (m - pooled_mean) / pooled_std if pooled_std > 0 else 0.0
        era_stats.append({
            "era":              era,
            "n":                n,
            "mean":             m,
            "std":              s,
            "mean_words":       int(era_df["word_count"].mean()),
            "z_shift_to_pool":  z_shift,
        })
        max_abs_zshift = max(max_abs_zshift, abs(z_shift))

    return {
        "status":            "FLAG" if max_abs_zshift > 1.0 else "OK",
        "pooled_mean":       pooled_mean,
        "pooled_std":        pooled_std,
        "max_abs_zshift":    max_abs_zshift,
        "interpretation":    (
            "If max|z-shift| > 1, era-mixing could be mis-read by HMM as "
            "regime transition — disclose in D5 verdict interpretation."
        ),
        "per_era": era_stats,
    }


def fill_chain_stats(df: pd.DataFrame) -> dict:
    """Compute OOS forward-fill chain lengths per month."""
    oos_df = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].copy()
    if oos_df.empty:
        return {"status": "NO_DATA"}
    oos_meeting_scores = pd.Series(
        oos_df["raw_score"].values,
        index=pd.to_datetime(oos_df["date"].values),
    )

    month_ends = pd.date_range(OOS_START, OOS_END, freq="ME")
    chain_lens: list[int] = []
    prior_chain = 0
    prior_value = None
    for ts in month_ends:
        # Aggregate this month
        val = aggregate_monthly(oos_meeting_scores, ts.date(), prior_value=prior_value)
        # Determine if this month's value came from a meeting in window or a fill
        in_window = (
            (oos_meeting_scores.index >= ts - pd.Timedelta(days=30))
            & (oos_meeting_scores.index <= ts)
        )
        n_in = int(in_window.sum())
        if n_in > 0:
            chain_lens.append(0)
            prior_chain = 0
        else:
            prior_chain += 1
            chain_lens.append(prior_chain)
        if not (isinstance(val, float) and np.isnan(val)):
            prior_value = val

    chain = np.array(chain_lens, dtype=int)
    return {
        "status":              "OK" if chain.max() <= 3 else "FLAG",
        "n_oos_months":        int(len(chain)),
        "max_fill_length":     int(chain.max()),
        "mean_fill_length":    float(chain.mean()),
        "frac_chain_geq_1":    float((chain >= 1).mean()),
        "frac_chain_geq_2":    float((chain >= 2).mean()),
        "interpretation":      (
            "Long fill chains (>2 months) weaken the 'forward-looking' claim — "
            "narrative_score is stale by then. If v4 verdict is INSUFFICIENT, "
            "high frac_chain_geq_2 may partially explain low signal density."
        ),
    }


def append_diagnostic_to_verdict(cross_era: dict, fill_chain: dict) -> None:
    if not VERDICT_PATH.exists():
        print(f"[error] verdict not found: {VERDICT_PATH}")
        return

    existing = VERDICT_PATH.read_text(encoding="utf-8")

    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("D2c FOLLOW-UP DIAGNOSTIC (post-test rigor pre-audit, 2026-05-08)")
    lines.append("=" * 78)
    lines.append("")
    lines.append("These two blocks DO NOT FAIL §3.8 (descriptive, not gates) but")
    lines.append("are documented BEFORE D5 walk-forward run, so D5 verdict")
    lines.append("interpretation has the right context (no post-hoc revision).")
    lines.append("")

    # Cross-era stability
    lines.append("-- Cross-era z-norm stability --")
    lines.append(f"  Status: {cross_era.get('status')}")
    if cross_era.get("status") not in ("NO_DATA",):
        lines.append(f"  pooled_mean: {cross_era['pooled_mean']:.6f}")
        lines.append(f"  pooled_std:  {cross_era['pooled_std']:.6f}")
        lines.append(f"  max |z-shift to pooled|: {cross_era['max_abs_zshift']:.3f}")
        lines.append("  Per era:")
        for s in cross_era["per_era"]:
            lines.append(
                f"    {s['era']:<25} n={s['n']:>3} "
                f"mean={s['mean']:+.5f} std={s['std']:.5f} "
                f"avg_words={s['mean_words']} "
                f"z-shift={s['z_shift_to_pool']:+.3f}"
            )
        lines.append(f"  Interpretation: {cross_era['interpretation']}")
    lines.append("")

    # Fill-chain
    lines.append("-- OOS forward-fill chain length (2019-2024) --")
    lines.append(f"  Status: {fill_chain.get('status')}")
    if fill_chain.get("status") not in ("NO_DATA",):
        lines.append(f"  n_oos_months:      {fill_chain['n_oos_months']}")
        lines.append(f"  max_fill_length:   {fill_chain['max_fill_length']}")
        lines.append(f"  mean_fill_length:  {fill_chain['mean_fill_length']:.2f}")
        lines.append(f"  frac_chain ≥ 1:    {fill_chain['frac_chain_geq_1']:.1%}")
        lines.append(f"  frac_chain ≥ 2:    {fill_chain['frac_chain_geq_2']:.1%}")
        lines.append(f"  Interpretation: {fill_chain['interpretation']}")
    lines.append("")

    lines.append("=" * 78)
    lines.append("End of D2c follow-up diagnostic.")
    lines.append("=" * 78)

    VERDICT_PATH.write_text(existing.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")
    print(f"appended diagnostic to {VERDICT_PATH}")


def main():
    if not CACHE_PARQUET.exists():
        print(f"[error] cache.parquet not found at {CACHE_PARQUET}")
        print("Run scripts/run_v4_universe_feasibility.py first.")
        sys.exit(1)

    df = pd.read_parquet(CACHE_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df)} verified statements from cache")

    print("\n-- cross-era stability --")
    cross_era = cross_era_stability(df)
    for k, v in cross_era.items():
        if k == "per_era":
            for s in v:
                print(f"   {s}")
        else:
            print(f"   {k}: {v}")

    print("\n-- fill-chain stats --")
    fill_chain = fill_chain_stats(df)
    for k, v in fill_chain.items():
        print(f"   {k}: {v}")

    print()
    append_diagnostic_to_verdict(cross_era, fill_chain)


if __name__ == "__main__":
    main()
