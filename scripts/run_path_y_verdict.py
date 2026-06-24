"""
scripts/run_path_y_verdict.py — Path Y Vol-Target Overlay verdict.

5 overlay-specific gates per spec §2.6 (different from V/W/X sleeve gates):
  G1 PRIMARY  Sharpe(overlay net) − Sharpe(baseline net) ≥ +0.05
  G2          Newey-West HAC t-stat on Sharpe DIFFERENCE > 1.96
  G3          Max DD reduction ≥ 1.0 percentage point
  G4          Bootstrap 95% CI on Sharpe difference excludes 0
  G5          Overlay ≥ baseline in ≥ 2 of 3 crisis windows (RELATIVE)

Decision rule: 5/5 PASS · 4/5 MARGINAL · ≤3/5 FAIL.
"""
from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.run_path_v_verdict import (
    WEEKLY_RFR, NW_LAG, BOOTSTRAP_N, BOOTSTRAP_BLOCK, CRISIS_WINDOWS,
    annualized_sharpe, newey_west_t,
)


G1_SHARPE_DIFF_THRESHOLD  = 0.05
G2_T_THRESHOLD            = 1.96
G3_DD_REDUCTION_THRESHOLD = 0.01   # 1.0 pp = 0.01 in decimal
G5_RELATIVE_WINS_REQUIRED = 2


def max_drawdown(weekly_returns: pd.Series) -> float:
    """Return max drawdown as a negative decimal (e.g., -0.15 for -15%)."""
    nav = (1 + weekly_returns.fillna(0)).cumprod()
    running_peak = nav.cummax()
    dd = (nav / running_peak) - 1.0
    return float(dd.min())


def crisis_returns_overlay(
    series: pd.Series,
) -> dict:
    """Cumulative return per crisis window for the given series."""
    out = {}
    idx = pd.to_datetime(series.index)
    s = series.copy()
    s.index = idx
    for label, (start, end) in CRISIS_WINDOWS.items():
        sub = s.loc[start:end].dropna()
        if len(sub) == 0:
            out[label] = float("nan")
        else:
            out[label] = float((1.0 + sub).prod() - 1.0)
    return out


def paired_sharpe_diff_bootstrap(
    overlay_returns:  np.ndarray,
    baseline_returns: np.ndarray,
    n_resample:       int = BOOTSTRAP_N,
    block_mean:       int = BOOTSTRAP_BLOCK,
    seed:             int = 42,
) -> tuple[float, float]:
    """Stationary paired block bootstrap CI for Sharpe difference."""
    rng = np.random.default_rng(seed)
    T = len(overlay_returns)
    if T < 50:
        return (float("nan"), float("nan"))
    p = 1.0 / block_mean
    diffs = np.empty(n_resample, dtype=float)
    for b in range(n_resample):
        ov_s = np.empty(T)
        bl_s = np.empty(T)
        i = rng.integers(0, T)
        for t in range(T):
            ov_s[t] = overlay_returns[i]
            bl_s[t] = baseline_returns[i]
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T
        ov_excess = ov_s - WEEKLY_RFR
        bl_excess = bl_s - WEEKLY_RFR
        if ov_excess.std() <= 0 or bl_excess.std() <= 0:
            diffs[b] = 0.0
            continue
        sh_ov = ov_excess.mean() / ov_excess.std() * math.sqrt(52)
        sh_bl = bl_excess.mean() / bl_excess.std() * math.sqrt(52)
        diffs[b] = sh_ov - sh_bl
    return (float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)))


def main() -> None:
    print("=== Path Y Vol-Target Overlay verdict run ===")
    print()

    pv_path = REPO_ROOT / "data" / "portfolio_replay" / "v1_path_y_voltarget_weekly.parquet"
    df = pd.read_parquet(pv_path)
    df.index = pd.to_datetime(df.index)

    # Use only weeks where overlay was applied (post-warmup)
    overlay = df["overlay_net"].dropna()
    baseline = df["baseline"].loc[overlay.index]

    print(f"Overlay-eval period: n={len(overlay)} weeks, "
          f"{overlay.index.min().date()} → {overlay.index.max().date()}")
    print()

    # ── G1 PRIMARY ──
    sh_ov = annualized_sharpe(overlay)
    sh_bl = annualized_sharpe(baseline)
    sh_diff = sh_ov - sh_bl
    g1_pass = sh_diff >= G1_SHARPE_DIFF_THRESHOLD
    print(f"G1 PRIMARY   Sharpe overlay = {sh_ov:+.4f}, baseline = {sh_bl:+.4f}, "
          f"diff = {sh_diff:+.4f}   threshold >= {G1_SHARPE_DIFF_THRESHOLD}   "
          f"-> {'PASS' if g1_pass else 'FAIL'}")

    # ── G2 (Newey-West on per-week Sharpe-diff INNOVATIONS) ──
    # Approximation: per-week excess-return difference (overlay-baseline) tested
    # for non-zero mean via NW HAC. This is the standard "paired return" test
    # for Sharpe difference at fixed denominator approximation.
    excess_ov = (overlay - WEEKLY_RFR).to_numpy()
    excess_bl = (baseline - WEEKLY_RFR).to_numpy()
    diff_excess = excess_ov - excess_bl
    nw_t = newey_west_t(diff_excess, lag=NW_LAG)
    g2_pass = (not math.isnan(nw_t)) and (nw_t > G2_T_THRESHOLD)
    print(f"G2           Newey-West t on excess-diff (lag-8) = {nw_t:+.4f}   "
          f"threshold > {G2_T_THRESHOLD}   -> {'PASS' if g2_pass else 'FAIL'}")

    # ── G3 ──
    dd_ov = max_drawdown(overlay)
    dd_bl = max_drawdown(baseline)
    dd_reduction = abs(dd_bl) - abs(dd_ov)
    g3_pass = dd_reduction >= G3_DD_REDUCTION_THRESHOLD
    print(f"G3           Max DD baseline = {dd_bl*100:+.2f}%, overlay = {dd_ov*100:+.2f}%, "
          f"reduction = {dd_reduction*100:+.2f}pp   threshold >= 1.0pp   "
          f"-> {'PASS' if g3_pass else 'FAIL'}")

    # ── G4 ──
    ci_lo, ci_hi = paired_sharpe_diff_bootstrap(
        overlay.to_numpy(), baseline.to_numpy(),
    )
    g4_pass = (not math.isnan(ci_lo)) and (ci_lo > 0)
    print(f"G4           Bootstrap 95% CI on Sharpe diff = [{ci_lo:+.4f}, {ci_hi:+.4f}]   "
          f"excludes 0? -> {'PASS' if g4_pass else 'FAIL'}")

    # ── G5 ──
    crisis_ov = crisis_returns_overlay(overlay)
    crisis_bl = crisis_returns_overlay(baseline)
    wins = sum(1 for k in CRISIS_WINDOWS
                if (not math.isnan(crisis_ov.get(k, float('nan')))
                     and not math.isnan(crisis_bl.get(k, float('nan')))
                     and crisis_ov[k] >= crisis_bl[k]))
    g5_pass = wins >= G5_RELATIVE_WINS_REQUIRED
    print(f"G5           Relative wins (overlay >= baseline) = {wins} of 3   "
          f"threshold >= {G5_RELATIVE_WINS_REQUIRED}   "
          f"-> {'PASS' if g5_pass else 'FAIL'}")
    for label in CRISIS_WINDOWS:
        ov = crisis_ov[label]
        bl = crisis_bl[label]
        win = ov >= bl
        marker = "WIN " if win else "LOSE"
        print(f"               [{marker}] {label}: overlay={ov*100:+.2f}%, "
              f"baseline={bl*100:+.2f}%, delta={(ov-bl)*100:+.2f}pp")
    print()

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
    if n_pass == 5:
        verdict = "PASS"
    elif n_pass == 4:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    print(f"=================================================================")
    print(f"  Path Y: {n_pass}/5 gates PASS  ->  VERDICT: {verdict}")
    print(f"=================================================================")

    today = datetime.date.today()
    payload = {
        "spec_id":  72,
        "spec_hash": "4fc22eb8",
        "spec_name": "Portfolio Vol-Target Overlay (Moreira-Muir 2017)",
        "run_date": today.isoformat(),
        "window": {"start": str(overlay.index.min().date()),
                   "end":   str(overlay.index.max().date()),
                   "n_weeks_eval": int(len(overlay))},
        "verdict":  verdict,
        "n_pass":   int(n_pass),
        "gates": {
            "G1_sharpe_diff":          {"value": sh_diff, "threshold": G1_SHARPE_DIFF_THRESHOLD,
                                         "pass": g1_pass, "sh_overlay": sh_ov, "sh_baseline": sh_bl},
            "G2_newey_west_t_on_diff": {"value": nw_t, "threshold": G2_T_THRESHOLD,
                                         "pass": g2_pass},
            "G3_dd_reduction_pp":      {"value": dd_reduction * 100,
                                         "threshold_pp": 1.0,
                                         "pass": g3_pass,
                                         "dd_overlay": dd_ov, "dd_baseline": dd_bl},
            "G4_bootstrap_ci_diff":    {"lo": ci_lo, "hi": ci_hi, "pass": g4_pass},
            "G5_relative_crisis_wins": {"count": wins, "threshold": G5_RELATIVE_WINS_REQUIRED,
                                         "pass": g5_pass,
                                         "overlay_returns": crisis_ov,
                                         "baseline_returns": crisis_bl},
        },
        "scale_stats": {
            "mean":        float(df["gross_scale"].mean()),
            "std":         float(df["gross_scale"].std()),
            "min":         float(df["gross_scale"].min()),
            "max":         float(df["gross_scale"].max()),
            "pct_at_cap":  float((df["gross_scale"] >= 1.999).mean()),
            "pct_at_floor": float((df["gross_scale"] <= 0.001).mean()),
        },
        "annual_tc_drag": float(df["overlay_tc"].sum() / (len(df) / 52.0)),
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"path_y_verdict_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Verdict saved: {out_path}")


if __name__ == "__main__":
    main()
