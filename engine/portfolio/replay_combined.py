"""
engine/portfolio/replay_combined.py — Sprint B historical replay.

Reads pre-computed daily/weekly return series for 4 strategies, resamples to
common weekly frequency, combines via paper-trade allocation (36/27/27/10),
and outputs combined portfolio statistics + verdict JSON.

This is HISTORICAL replay (in-sample validation), NOT forward paper trade.
Forward paper trade (daily auto-run with real-time data feed) is Sprint D.

Sprint B verdict question:
  - Does the 4-component combined portfolio Sharpe land in deployment_design.md
    expected forward band 0.85-1.15?
  - Are pairwise correlations stable over time (or do they drift / spike in crises)?
  - Do crisis windows 2018-Q4 / 2020-COVID / 2022 show CTA crisis-on benefit?

Output: data/portfolio_replay/v1_combined_replay_verdict.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Locked allocation (paper-trade per deployment_design.md, NOT real-capital)
# ─────────────────────────────────────────────────────────────────────────────
ALLOCATION_LOCKED: dict[str, float] = {
    "K1_BAB":    0.36,
    "D_PEAD":    0.27,
    "PATH_N":    0.27,
    "CTA_PQTIX": 0.10,
}
assert abs(sum(ALLOCATION_LOCKED.values()) - 1.0) < 1e-9

# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed return series paths
# ─────────────────────────────────────────────────────────────────────────────
PATHS = {
    "K1_BAB":    "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet",
    "D_PEAD":    "data/path_c_dhs/walk_forward_pead.parquet",
    "PATH_N":    "data/path_n/v1_reconstitution_10y_amend1_10bp_daily.parquet",
    "CTA_PQTIX": "data/path_o_cta/v1_cta_saa_daily.parquet",
}

# Crisis windows per Path O capability evidence
CRISIS_WINDOWS: dict[str, tuple[datetime.date, datetime.date]] = {
    "2018_VolMageddon": (datetime.date(2018, 10, 1), datetime.date(2018, 12, 31)),
    "2020_COVID":       (datetime.date(2020, 2, 15), datetime.date(2020, 4, 30)),
    "2022_Inflation":   (datetime.date(2022, 1, 1),  datetime.date(2022, 12, 31)),
}


@dataclass
class ReplayResult:
    """Combined portfolio replay output."""
    window:              tuple[datetime.date, datetime.date]
    strategy_returns_weekly: pd.DataFrame   # cols = strategies, index = week-Friday
    combined_returns_weekly: pd.Series
    combined_metrics:    dict
    per_strategy_metrics: dict
    pairwise_correlation: dict
    crisis_period_returns: dict
    sleeve_attribution:  dict
    notes:               list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_k1_weekly() -> pd.Series:
    """K1 BAB native weekly returns. Index = Friday weekly close 2014-2023."""
    df = pd.read_parquet(PATHS["K1_BAB"])
    # Map integer index 0..520 → Fri-weekly dates starting 2014-01-03
    start = pd.Timestamp("2014-01-03")
    dates = pd.bdate_range(start, periods=len(df), freq="W-FRI")
    df.index = dates
    return df["k1_weekly_returns"].rename("K1_BAB")


def _load_dpead_weekly() -> pd.Series:
    """D-PEAD daily net returns → resample to weekly Fri-close geometric link."""
    df = pd.read_parquet(PATHS["D_PEAD"])
    daily = df["r_long_short_net"].copy()
    daily.index = pd.to_datetime(daily.index)
    # Geometric link of daily returns to weekly
    weekly = (1 + daily).resample("W-FRI").prod() - 1
    return weekly.rename("D_PEAD")


def _load_path_n_weekly() -> pd.Series:
    """Path N daily net returns → weekly geometric link."""
    df = pd.read_parquet(PATHS["PATH_N"])
    daily = df["strategy_return"].copy()
    daily.index = pd.to_datetime(daily.index)
    weekly = (1 + daily).resample("W-FRI").prod() - 1
    return weekly.rename("PATH_N")


def _load_cta_weekly() -> pd.Series:
    """CTA combined net returns (90% SPY + 10% PQTIX standalone, per Path O saa).

    For replay we want the CTA STANDALONE component (PQTIX alone), not the
    combined-with-SPY series. Use pqtix_return column. The SPY portion is
    NOT applied here because the alpha portfolio (K1/D-PEAD/Path N) already
    has equity exposure.
    """
    df = pd.read_parquet(PATHS["CTA_PQTIX"])
    daily = df["pqtix_return"].copy()
    daily.index = pd.to_datetime(daily.index)
    weekly = (1 + daily).resample("W-FRI").prod() - 1
    return weekly.rename("CTA_PQTIX")


def load_all_strategy_returns_weekly() -> pd.DataFrame:
    """Load + align all 4 strategy weekly returns on Friday weekly close."""
    series_list = [
        _load_k1_weekly(),
        _load_dpead_weekly(),
        _load_path_n_weekly(),
        _load_cta_weekly(),
    ]
    df = pd.concat(series_list, axis=1).dropna()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Combined portfolio math
# ─────────────────────────────────────────────────────────────────────────────
def compute_combined_returns(
    strategy_returns_weekly: pd.DataFrame,
    allocation:              dict[str, float] = ALLOCATION_LOCKED,
) -> pd.Series:
    """Weighted sum of strategy returns per allocation."""
    weights = pd.Series(allocation)
    aligned = strategy_returns_weekly[list(allocation.keys())]
    return (aligned * weights).sum(axis=1).rename("combined_return")


def _annualized_stats(weekly_returns: pd.Series) -> dict:
    """Annualize weekly returns: × 52 mean, × √52 vol."""
    r = weekly_returns.dropna()
    if len(r) < 13:
        return {"ann_ret": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "max_dd": float("nan"),
                "n_weeks": int(len(r))}
    ann_ret = float(r.mean() * 52)
    ann_vol = float(r.std() * math.sqrt(52))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
    wealth = (1 + r.fillna(0)).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "ann_ret":  round(ann_ret, 4),
        "ann_vol":  round(ann_vol, 4),
        "sharpe":   round(sharpe, 4),
        "max_dd":   round(max_dd, 4),
        "n_weeks":  int(len(r)),
    }


def _pairwise_correlation_matrix(returns: pd.DataFrame) -> dict:
    """Pearson pairwise correlation across strategies."""
    corr = returns.corr()
    out = {}
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            rho = float(corr.loc[a, b])
            out[f"{a}__{b}"] = round(rho, 4)
    return out


def _crisis_period_combined_returns(
    combined_weekly: pd.Series,
) -> dict:
    """Total cumulative combined return over each crisis window."""
    idx = pd.to_datetime(combined_weekly.index)
    s = combined_weekly.copy()
    s.index = idx
    out = {}
    for name, (start, end) in CRISIS_WINDOWS.items():
        sub = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
        if sub.empty:
            out[name] = None
        else:
            out[name] = round(float((1 + sub.fillna(0)).prod() - 1.0), 4)
    return out


def _sleeve_attribution(
    strategy_returns_weekly: pd.DataFrame,
    allocation:              dict[str, float] = ALLOCATION_LOCKED,
) -> dict:
    """Compute per-sleeve cumulative contribution to combined return.

    Sleeve mapping:
      K1_BAB     → etf_l1
      D_PEAD     → ss_sp500
      PATH_N     → ss_sp500
      CTA_PQTIX  → cta_defensive
    """
    sleeve_map = {
        "K1_BAB":    "etf_l1",
        "D_PEAD":    "ss_sp500",
        "PATH_N":    "ss_sp500",
        "CTA_PQTIX": "cta_defensive",
    }
    weighted = strategy_returns_weekly.multiply(pd.Series(allocation), axis=1)
    cum = (1 + weighted).prod() - 1.0  # cumulative return per strategy contribution
    sleeve_cum: dict[str, float] = {}
    sleeve_alloc: dict[str, float] = {}
    for strat, sleeve in sleeve_map.items():
        sleeve_cum[sleeve] = sleeve_cum.get(sleeve, 0.0) + float(cum[strat])
        sleeve_alloc[sleeve] = sleeve_alloc.get(sleeve, 0.0) + allocation[strat]
    return {
        "sleeve_cumulative_contribution": {k: round(v, 4) for k, v in sleeve_cum.items()},
        "sleeve_total_allocation":         {k: round(v, 4) for k, v in sleeve_alloc.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level replay
# ─────────────────────────────────────────────────────────────────────────────
def run_replay(
    allocation: dict[str, float] = ALLOCATION_LOCKED,
) -> ReplayResult:
    """Run full 4-component weekly replay over common backtest period."""
    returns_weekly = load_all_strategy_returns_weekly()
    if returns_weekly.empty:
        raise RuntimeError("No common dates across 4 strategies; check parquet alignment")

    combined = compute_combined_returns(returns_weekly, allocation)

    window = (
        returns_weekly.index.min().date(),
        returns_weekly.index.max().date(),
    )

    per_strat = {
        col: _annualized_stats(returns_weekly[col])
        for col in returns_weekly.columns
    }
    combined_metrics = _annualized_stats(combined)
    pairwise = _pairwise_correlation_matrix(returns_weekly)
    crisis = _crisis_period_combined_returns(combined)
    sleeve = _sleeve_attribution(returns_weekly, allocation)

    notes = [
        "Replay uses WEEKLY frequency (K1 native weekly; D-PEAD/Path N/CTA resampled).",
        "Pre-computed strategy daily/weekly returns loaded from existing backtest parquets.",
        "TC already net in source returns: K1 native, D-PEAD r_long_short_net, Path N strategy_return, CTA pqtix_return.",
        "This is HISTORICAL in-sample replay; forward paper trade with daily Sprint D orchestrator is separate.",
        "CTA leg uses PQTIX standalone (not 90% SPY + 10% PQTIX combined); alpha sleeves provide equity exposure.",
        "ρ here is in-sample on SAME period that strategies were SELECTED; forward ρ likely higher (Stein shrinkage applied conceptually in Sprint E).",
    ]

    return ReplayResult(
        window                = window,
        strategy_returns_weekly = returns_weekly,
        combined_returns_weekly = combined,
        combined_metrics      = combined_metrics,
        per_strategy_metrics  = per_strat,
        pairwise_correlation  = pairwise,
        crisis_period_returns = crisis,
        sleeve_attribution    = sleeve,
        notes                 = notes,
    )


def save_combined_returns_parquet(result: ReplayResult, save_path: Path) -> Path:
    """Persist combined weekly returns series for downstream consumers (VaR/CVaR etc)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.combined_returns_weekly.to_frame(name="combined_return")
    df.index.name = "week_end"
    df.to_parquet(save_path)
    return save_path


def save_per_strategy_returns_parquet(result: ReplayResult, save_path: Path) -> Path:
    """Persist per-strategy weekly returns DataFrame for Stein-James shrinkage etc."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = result.strategy_returns_weekly.copy()
    df.index.name = "week_end"
    df.to_parquet(save_path)
    return save_path


def save_replay_verdict(result: ReplayResult, save_path: Path) -> dict:
    """Save replay verdict to JSON."""
    verdict = {
        "run_at":         datetime.datetime.utcnow().isoformat() + "Z",
        "window":         f"{result.window[0].isoformat()} to {result.window[1].isoformat()}",
        "n_weeks":        len(result.combined_returns_weekly),
        "allocation":     ALLOCATION_LOCKED,
        "combined_metrics":    result.combined_metrics,
        "per_strategy_metrics": result.per_strategy_metrics,
        "pairwise_correlation": result.pairwise_correlation,
        "crisis_period_returns": result.crisis_period_returns,
        "sleeve_attribution":    result.sleeve_attribution,
        "expected_forward_band": {
            "sharpe_low":  0.85,
            "sharpe_high": 1.15,
            "max_dd_target_pp": -0.06,
            "note": "Per docs/portfolio_deployment_design_2026-05-13.md §6 forward expectations",
        },
        "honest_disclose":      result.notes,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Sprint B 4-component historical replay")
    parser.add_argument("--save", action="store_true",
                        help="Save verdict JSON to data/portfolio_replay/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("[replay_combined] loading 4 strategy weekly return series...")
    result = run_replay()
    n = len(result.combined_returns_weekly)
    print(f"[replay_combined] common weeks: {n} ({result.window[0]} to {result.window[1]})")

    print()
    print("=== Per-strategy weekly stats (annualized) ===")
    for strat, m in result.per_strategy_metrics.items():
        print(f"  {strat:<10} Sharpe={m['sharpe']:+.3f} "
              f"ann_ret={m['ann_ret']:+.2%} ann_vol={m['ann_vol']:.2%} "
              f"maxDD={m['max_dd']:+.2%} n={m['n_weeks']}")

    print()
    print("=== Combined portfolio (36% K1 + 27% D-PEAD + 27% Path N + 10% CTA) ===")
    m = result.combined_metrics
    print(f"  Sharpe={m['sharpe']:+.3f} "
          f"ann_ret={m['ann_ret']:+.2%} ann_vol={m['ann_vol']:.2%} "
          f"maxDD={m['max_dd']:+.2%}")

    print()
    print("=== Pairwise correlations ===")
    for pair, rho in result.pairwise_correlation.items():
        print(f"  {pair}: ρ={rho:+.3f}")

    print()
    print("=== Crisis-period combined returns ===")
    for window, ret in result.crisis_period_returns.items():
        if ret is None:
            print(f"  {window}: N/A")
        else:
            print(f"  {window}: {ret*100:+.2f}%")

    print()
    print("=== Sleeve attribution (cumulative) ===")
    for sleeve, cum in result.sleeve_attribution["sleeve_cumulative_contribution"].items():
        alloc = result.sleeve_attribution["sleeve_total_allocation"][sleeve]
        print(f"  {sleeve:<14} alloc={alloc:.0%} cum_contrib={cum*100:+.2f}%")

    if args.save:
        save_path = Path("data/portfolio_replay/v1_combined_replay_verdict.json")
        verdict = save_replay_verdict(result, save_path)
        print(f"\nVerdict saved to {save_path}")
        returns_path = Path("data/portfolio_replay/v1_combined_returns_weekly.parquet")
        save_combined_returns_parquet(result, returns_path)
        print(f"Combined weekly returns saved to {returns_path}")
        per_strat_path = Path("data/portfolio_replay/v1_per_strategy_returns_weekly.parquet")
        save_per_strategy_returns_parquet(result, per_strat_path)
        print(f"Per-strategy weekly returns saved to {per_strat_path}")


if __name__ == "__main__":
    main()
