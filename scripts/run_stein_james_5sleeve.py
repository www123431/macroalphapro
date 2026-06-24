"""
scripts/run_stein_james_5sleeve.py — Bayes-Stein SAA on 5-sleeve composition.

Extends Tier-1 audit class A #2 (2026-05-14, 4-sleeve) with Path AC included.

5-sleeve set: K1_BAB, D_PEAD, PATH_N, CTA_PQTIX, AC (TLT/GLD insurance).
AC weekly returns: AB 2014-23 same-window proxy (since current 4-sleeve has
no pre-2014 history). Path AC's extended-window verdict (2005-23, Sharpe
+0.24) stands separately as v3 capability evidence; this 2014-23 same-window
analysis is the apples-to-apples comparison for SAA weight allocation.

Per Stein-James doctrine (allocation_shrinkage.py):
  - Bayes-Stein on means (Jorion 1986)
  - Ledoit-Wolf on covariance (Ledoit-Wolf 2004)
  - Constrained Markowitz solve under multiple constraint sets

Constraint sets evaluated:
  1. Pure-optimal: unconstrained Markowitz on shrunk inputs
  2. Capped 60%: no single sleeve > 60%
  3. Proposed deployment: K1=36% locked, CTA=10% locked, AC capped 15%,
     D_PEAD + Path N share remainder
"""
from __future__ import annotations

import dataclasses
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

from engine.portfolio.allocation_shrinkage import (
    bayes_stein_shrink_mean, ledoit_wolf_shrink_cov, solve_optimal_weights,
    WEEKLY_RFR_USD,
)


# Proposed deployment configuration (institutional defaults; Tier 3 review-able)
PROPOSED_K1_FIXED:        float = 0.36
PROPOSED_CTA_FIXED:       float = 0.10
PROPOSED_AC_MAX:          float = 0.15  # cap AC at 15% per Asness-Israelov 2017 RMS practice
RISK_AVERSION:            float = 2.0


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.4f}%"


def _fmt_wt(x: float) -> str:
    return f"{x*100:.1f}%"


def main() -> None:
    print("=" * 78)
    print("Stein-James SAA on 5-sleeve composition {K1, D-PEAD, Path N, CTA, AC}")
    print("=" * 78)
    print()

    df = pd.read_parquet(
        REPO_ROOT / "data" / "portfolio_replay" / "v2_per_strategy_returns_5sleeve_weekly.parquet"
    ).astype("float64").fillna(0.0)
    strategies = list(df.columns)
    n_weeks = len(df)
    print(f"Sample: {df.index.min()} -> {df.index.max()}, n={n_weeks} weeks")
    print(f"Strategies: {strategies}")
    print()

    returns_arr = df.values.astype(np.float64)
    excess = returns_arr - WEEKLY_RFR_USD

    sample_means_w = excess.mean(axis=0)
    sample_cov = np.cov(excess.T, ddof=1)
    sample_vols_w = np.sqrt(np.diag(sample_cov))
    sample_sharpe_ann = sample_means_w / sample_vols_w * math.sqrt(52)

    print("Sample statistics:")
    print(f"{'Strategy':<22} {'Mean (weekly)':>15} {'Vol':>10} {'Sharpe (ann)':>15}")
    for i, s in enumerate(strategies):
        print(f"{s:<22} {_fmt_pct(sample_means_w[i]):>15} "
              f"{sample_vols_w[i]*100:>9.3f}% {sample_sharpe_ann[i]:>15.4f}")
    print()

    print("Sample correlation matrix:")
    print(" " * 22, " ".join(f"{s:>10}" for s in strategies))
    for i, s_i in enumerate(strategies):
        diag = [sample_cov[k][k] for k in range(len(strategies))]
        row = f"{s_i:<22}"
        for j in range(len(strategies)):
            if diag[i] > 0 and diag[j] > 0:
                rho = sample_cov[i][j] / math.sqrt(diag[i] * diag[j])
            else:
                rho = float("nan")
            row += f" {rho:>10.4f}"
        print(row)
    print()

    # Bayes-Stein shrink means
    shrunk_means_w, mean_w = bayes_stein_shrink_mean(sample_means_w, sample_cov, n_obs=n_weeks)
    # Ledoit-Wolf shrink cov
    shrunk_cov, cov_alpha = ledoit_wolf_shrink_cov(excess)
    shrunk_vols_w = np.sqrt(np.diag(shrunk_cov))
    shrunk_sharpe_ann = shrunk_means_w / shrunk_vols_w * math.sqrt(52)

    ones = np.ones(len(strategies))
    cov_inv = np.linalg.pinv(sample_cov)
    grand_mean = float((ones @ cov_inv @ sample_means_w) / float(ones @ cov_inv @ ones))

    print(f"Bayes-Stein mean shrinkage intensity w     = {mean_w:.4f}")
    print(f"Ledoit-Wolf cov shrinkage intensity alpha  = {cov_alpha:.4f}")
    print(f"Grand mean (precision-weighted) weekly     = {grand_mean*100:+.4f}%")
    print()
    print("Per-strategy Sharpe (sample vs shrunk):")
    print(f"{'Strategy':<22} {'Sample':>10} {'Shrunk':>10} {'Delta':>10}")
    for i, s in enumerate(strategies):
        d = shrunk_sharpe_ann[i] - sample_sharpe_ann[i]
        print(f"{s:<22} {sample_sharpe_ann[i]:>10.4f} {shrunk_sharpe_ann[i]:>10.4f} {d:>+10.4f}")
    print()

    # Constraint set 1: unconstrained
    w_unc = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=RISK_AVERSION, max_weight=1.0,
    )

    # Constraint set 2: capped 60%
    w_caps = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=RISK_AVERSION, max_weight=0.60,
    )

    # Constraint set 3: proposed deployment
    proposed_locks = {
        "K1_BAB":              (PROPOSED_K1_FIXED, PROPOSED_K1_FIXED),
        "CTA_PQTIX":           (PROPOSED_CTA_FIXED, PROPOSED_CTA_FIXED),
        "AC_proxy_AB_2014_23": (0.0, PROPOSED_AC_MAX),
    }
    w_proposed = solve_optimal_weights(
        shrunk_means_w, shrunk_cov, strategies,
        risk_aversion=RISK_AVERSION, max_weight=1.0,
        sleeve_locks=proposed_locks,
    )

    # Current 4-sleeve SAA (reference, padded with AC=0)
    current_4 = {"K1_BAB": 0.36, "D_PEAD": 0.27, "PATH_N": 0.27, "CTA_PQTIX": 0.10,
                  "AC_proxy_AB_2014_23": 0.0}

    print("Weights under constraint sets:")
    print(f"{'Strategy':<22} {'Current 4sleeve':>16} {'Proposed (locks)':>18} {'Capped 60%':>13} {'Unconstrained':>16}")
    for s in strategies:
        print(f"{s:<22} {_fmt_wt(current_4.get(s, 0)):>16} "
              f"{_fmt_wt(w_proposed[s]):>18} "
              f"{_fmt_wt(w_caps[s]):>13} "
              f"{_fmt_wt(w_unc[s]):>16}")
    print()

    # Forward Sharpe estimates
    def _portfolio_sharpe(w_dict: dict, means: np.ndarray, cov: np.ndarray) -> float:
        w = np.array([w_dict.get(s, 0.0) for s in strategies], dtype=np.float64)
        mu = float(w @ means)
        vol = float(math.sqrt(w @ cov @ w))
        if vol < 1e-9:
            return 0.0
        return mu / vol * math.sqrt(52)

    forward = {
        "current_4sleeve":     _portfolio_sharpe(current_4, shrunk_means_w, shrunk_cov),
        "proposed_5sleeve":    _portfolio_sharpe(w_proposed, shrunk_means_w, shrunk_cov),
        "capped_60":           _portfolio_sharpe(w_caps,    shrunk_means_w, shrunk_cov),
        "unconstrained":       _portfolio_sharpe(w_unc,     shrunk_means_w, shrunk_cov),
    }
    print("Forward Sharpe estimates (shrunk inputs):")
    for k, v in forward.items():
        print(f"  {k:<22} {v:>+10.4f}")
    print()

    # Sharpe gain from adding AC
    sharpe_gain = forward["proposed_5sleeve"] - forward["current_4sleeve"]
    print(f"Sharpe lift from adding AC (proposed deployment vs current 4-sleeve): {sharpe_gain:+.4f}")
    print()

    # Decision logic
    if sharpe_gain >= 0.01:
        recommendation = "ADD AC AT PROPOSED WEIGHTS"
        rationale = (
            f"Adding Path AC at proposed weights (K1 {_fmt_wt(w_proposed['K1_BAB'])} / "
            f"D-PEAD {_fmt_wt(w_proposed['D_PEAD'])} / Path N {_fmt_wt(w_proposed['PATH_N'])} / "
            f"CTA {_fmt_wt(w_proposed['CTA_PQTIX'])} / AC {_fmt_wt(w_proposed['AC_proxy_AB_2014_23'])}) "
            f"lifts shrunk portfolio Sharpe by {sharpe_gain:+.4f}. Combined with AC's "
            f"PASS verdict on v3 insurance class extended-window evaluation, route "
            f"through Tier 3 supervisor approval for SAA amendment."
        )
    elif sharpe_gain >= -0.005:
        recommendation = "ADD AC AT MINIMUM ALLOCATION (Phase 1 paper-trade)"
        rationale = (
            f"Shrunk Sharpe gain {sharpe_gain:+.4f} is statistically indistinguishable "
            f"from zero on same-window analysis, but Path AC has PASS verdict on v3 "
            f"insurance gates with strong extended-window evidence (2008 GFC +18pp, "
            f"G7 +7.42pp DD reduction). Phase 1 paper-trade deployment at modest "
            f"weight begins forward evidence accumulation; full allocation pending 6-12mo "
            f"forward IC."
        )
    else:
        recommendation = "HOLD AC — same-window evidence weaker than extended"
        rationale = (
            f"Same-window shrunk analysis shows Sharpe drag {sharpe_gain:+.4f}. AC's "
            f"value lives in 2008 GFC + 2011 / 2020 crisis windows captured in "
            f"extended evaluation but absent in 2014-23 same-window data. Consider "
            f"holding deployment until Sprint E E-1 audit (2026-07-15) reveals "
            f"whether existing sleeves show decay."
        )

    print("=" * 78)
    print(f"RECOMMENDATION: {recommendation}")
    print("=" * 78)
    print(rationale)
    print()

    # Save JSON
    today = datetime.date.today()
    payload = {
        "audit_date":        today.isoformat(),
        "n_weeks":           int(n_weeks),
        "window_start":      str(df.index.min().date()),
        "window_end":        str(df.index.max().date()),
        "strategies":        strategies,
        "sample_means_weekly":   {s: float(sample_means_w[i]) for i, s in enumerate(strategies)},
        "sample_vols_weekly":    {s: float(sample_vols_w[i])  for i, s in enumerate(strategies)},
        "sample_sharpe_ann":     {s: float(sample_sharpe_ann[i]) for i, s in enumerate(strategies)},
        "sample_covariance":     sample_cov.tolist(),
        "mean_shrinkage_w":      float(mean_w),
        "cov_shrinkage_alpha":   float(cov_alpha),
        "grand_mean_weekly":     float(grand_mean),
        "shrunk_means_weekly":   {s: float(shrunk_means_w[i]) for i, s in enumerate(strategies)},
        "shrunk_sharpe_ann":     {s: float(shrunk_sharpe_ann[i]) for i, s in enumerate(strategies)},
        "weights_current_4sleeve":    current_4,
        "weights_proposed_5sleeve":   w_proposed,
        "weights_capped_60":          w_caps,
        "weights_unconstrained":      w_unc,
        "forward_sharpe_estimates":   forward,
        "sharpe_gain_from_ac":        float(sharpe_gain),
        "recommendation":             recommendation,
        "rationale":                  rationale,
        "ac_v3_verdict_reference":    {"spec_id": 77, "hash": "4db40176",
                                        "verdict": "PASS 4/4", "window": "2005-2023"},
        "honest_caveats": [
            "AC same-window proxy uses AB 2014-23 returns (TLT/GLD strategy identical). "
            "Path AC extended-window verdict (2005-23 + 60/40 baseline) is the v3 PASS evidence.",
            "K1/D-PEAD/Path N have no pre-2014 history; cannot do extended same-window 5-sleeve analysis.",
            "Sharpe gain estimate is on shrunk Bayes-Stein inputs, in-sample 2014-23. "
            "Forward Sharpe per deployment_design.md expected 0.85-1.15.",
            "Adding AC reduces D_PEAD + Path N total allocation from 54% to (54 - AC%); "
            "this is a Tier 3 governance decision, not auto-implemented.",
        ],
    }
    out_path = REPO_ROOT / "data" / "portfolio_replay" / f"saa_stein_james_5sleeve_{today.isoformat()}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Stein-James 5-sleeve audit saved: {out_path}")


if __name__ == "__main__":
    main()
