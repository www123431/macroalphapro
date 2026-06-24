"""
engine/portfolio/correlation_sentinel.py — Sprint G daily rho drift monitor.

PROBLEM
-------
deployment_design.md §1 anchors the entire 4-alpha portfolio thesis on
"all pairwise |rho| < 0.10" empirically. Sprint B replay confirmed this
IN-SAMPLE (2014-2023). But correlations are not stable in finance —
they drift, especially in stress regimes (Christoffersen-Errunza 2000).

Without DAILY drift monitoring, rho could silently drift to 0.35 over 6mo,
breaking the diversification assumption while combined Sharpe quietly
falls from 1.3 to 0.6. Manual quarterly re-runs of Sprint B replay
are too slow.

SOLUTION (Sprint G)
-------------------
Daily-running sentinel that:
  1. Computes trailing 12-week pairwise correlation (≈60 trading days)
     across all 4 strategy daily/weekly return series
  2. Compares to deployment_design baseline (Sprint B in-sample rho matrix)
  3. Flags any pair with |rho| > 0.20 (WARN) or > 0.30 (CRITICAL)
  4. Integrates with Watchdog notification stack via rule_pairwise_correlation_drift

INTEGRATION
-----------
- Reuses engine.portfolio.replay_combined.load_all_strategy_returns_weekly()
- Window 12 weeks ≈ 60 trading days (honors user spec; matches K1 weekly cadence)
- For forward paper trade: when PaperTradeStrategyLog daily_net_return populates
  (post Sprint H), correlation_sentinel can ALSO read forward returns and
  detect drift in real time

Output: data/portfolio_replay/correlation_sentinel_<date>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants
# ─────────────────────────────────────────────────────────────────────────────
ROLLING_WINDOW_WEEKS_LOCKED: int = 12        # ≈ 60 trading days
WARN_THRESHOLD:              float = 0.20    # |rho| > 0.20 → WARN
CRITICAL_THRESHOLD:          float = 0.30    # |rho| > 0.30 → CRITICAL

# Sprint B in-sample baseline rho matrix (deployment_design.md §1, 2014-2023 replay).
# Source: data/portfolio_replay/v1_combined_replay_verdict.json
# Contains the 4 strategies present in the 2014-2023 backtest window. AC_TLT_GLD
# was Tier-3 approved 2026-05-15; its baseline pairs are not in this in-sample
# dict and will appear with rho_baseline=None in the sentinel output (drift
# severity still computed from |rho_trailing| against thresholds). Computing
# AC pair baselines requires Path AC's 2005-2023 extended replay; deferred
# to a follow-up commit with explicit governance entry.
BASELINE_RHO_IN_SAMPLE: dict[tuple[str, str], float] = {
    ("K1_BAB",    "D_PEAD"):    -0.107,
    ("K1_BAB",    "PATH_N"):     0.027,
    ("K1_BAB",    "CTA_PQTIX"): -0.061,
    ("D_PEAD",    "PATH_N"):     0.008,
    ("D_PEAD",    "CTA_PQTIX"):  0.220,   # highest baseline pair
    ("PATH_N",    "CTA_PQTIX"): -0.032,
}


def _get_strategy_order() -> list[str]:
    """Strategies to monitor for trailing rho drift — sourced from registry
    so adding a new strategy in engine/strategies/adapters.py auto-propagates
    here (no more hand-maintaining a parallel list)."""
    from engine.strategies import get_registry
    return list(get_registry().names())


@dataclass
class PairwiseCorrelation:
    """One pairwise correlation measurement."""
    pair_a:               str
    pair_b:               str
    rho_trailing:         float
    rho_baseline:         float
    abs_drift:            float           # |rho_trailing| - |rho_baseline|
    severity:             str             # "CLEAN" / "WARN" / "CRITICAL"


@dataclass
class CorrelationSentinelReport:
    """Sprint G report output."""
    as_of:               datetime.date
    window_weeks:        int
    sample_n_weeks:      int
    correlations:        list[PairwiseCorrelation]
    max_abs_rho:         float
    max_drift:           float
    alerts:              list[str]              # human-readable alert lines
    severity:            str                    # overall worst severity
    notes:               list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core math
# ─────────────────────────────────────────────────────────────────────────────
def compute_trailing_correlation_matrix(
    returns_weekly:        pd.DataFrame,
    as_of:                 datetime.date,
    window_weeks:          int = ROLLING_WINDOW_WEEKS_LOCKED,
) -> tuple[pd.DataFrame, int]:
    """Slice returns to trailing window ending at as_of; compute pairwise rho.

    Args:
        returns_weekly: DataFrame indexed by week-Friday, columns = strategies
        as_of:           report date; window = (as_of - window_weeks weeks, as_of]
        window_weeks:    trailing window length

    Returns:
        (correlation_matrix DataFrame, sample_n_weeks_actually_used)
    """
    cutoff_start = pd.Timestamp(as_of) - pd.Timedelta(weeks=window_weeks)
    cutoff_end   = pd.Timestamp(as_of)
    idx = pd.to_datetime(returns_weekly.index)
    mask = (idx > cutoff_start) & (idx <= cutoff_end)
    sub = returns_weekly[mask]
    if sub.empty or len(sub) < 4:
        # too few samples for meaningful correlation
        return pd.DataFrame(), int(len(sub))
    return sub.corr(method="pearson"), int(len(sub))


def classify_pair(
    pair_a:        str,
    pair_b:        str,
    rho_trailing:  float,
    rho_baseline:  float,
) -> PairwiseCorrelation:
    """Classify drift severity for one pair."""
    abs_rho = abs(rho_trailing)
    abs_baseline = abs(rho_baseline)
    abs_drift = abs_rho - abs_baseline

    if abs_rho > CRITICAL_THRESHOLD:
        severity = "CRITICAL"
    elif abs_rho > WARN_THRESHOLD:
        severity = "WARN"
    else:
        severity = "CLEAN"

    return PairwiseCorrelation(
        pair_a       = pair_a,
        pair_b       = pair_b,
        rho_trailing = round(float(rho_trailing), 4),
        rho_baseline = round(float(rho_baseline), 4),
        abs_drift    = round(float(abs_drift), 4),
        severity     = severity,
    )


def run_correlation_sentinel(
    as_of:               Optional[datetime.date] = None,
    window_weeks:        int                      = ROLLING_WINDOW_WEEKS_LOCKED,
) -> CorrelationSentinelReport:
    """Sprint G entry point — daily rho drift check across all 4 strategy pairs."""
    if as_of is None:
        as_of = datetime.datetime.utcnow().date()

    # Reuse Sprint B parquet loaders
    from engine.portfolio.replay_combined import load_all_strategy_returns_weekly

    returns_weekly = load_all_strategy_returns_weekly()
    corr_matrix, n_weeks = compute_trailing_correlation_matrix(
        returns_weekly, as_of, window_weeks=window_weeks,
    )

    correlations: list[PairwiseCorrelation] = []
    alerts: list[str] = []

    if corr_matrix.empty:
        notes = [
            f"Insufficient data: {n_weeks} weeks in window ending {as_of}; "
            f"need ≥ 4 weeks for meaningful correlation.",
            "Sprint G v1 reads existing backtest parquets (end 2023-12-22 for K1). "
            "For dates past parquet end, sentinel reports INSUFFICIENT_DATA — "
            "this is data layer issue, not portfolio thesis broken.",
        ]
        return CorrelationSentinelReport(
            as_of           = as_of,
            window_weeks    = window_weeks,
            sample_n_weeks  = n_weeks,
            correlations    = [],
            max_abs_rho     = float("nan"),
            max_drift       = float("nan"),
            alerts          = ["INSUFFICIENT_DATA: cannot compute trailing correlation"],
            severity        = "INSUFFICIENT_DATA",
            notes           = notes,
        )

    # Compute pair-wise drift classification.
    # Strategy list comes from the registry (engine.strategies) so newly added
    # strategies appear here automatically. Pairs without an in-sample baseline
    # (e.g. AC_TLT_GLD against 2014-2023 alphas) get rho_baseline=NaN; severity
    # is still classified from |rho_trailing| against the absolute thresholds.
    strategy_order = _get_strategy_order()
    for i, a in enumerate(strategy_order):
        for b in strategy_order[i+1:]:
            if a not in corr_matrix.columns or b not in corr_matrix.columns:
                continue
            rho_trailing = float(corr_matrix.loc[a, b])
            # Try both (a, b) and (b, a) orderings of the baseline dict key.
            baseline_raw = BASELINE_RHO_IN_SAMPLE.get((a, b))
            if baseline_raw is None:
                baseline_raw = BASELINE_RHO_IN_SAMPLE.get((b, a))
            baseline = float(baseline_raw) if baseline_raw is not None else float("nan")
            pair_result = classify_pair(a, b, rho_trailing, baseline)
            correlations.append(pair_result)

            # Alert message: format baseline as "n/a" when not available rather than "nan".
            baseline_str = f"{baseline:+.3f}" if not math.isnan(baseline) else "n/a"
            if pair_result.severity == "CRITICAL":
                alerts.append(
                    f"CRITICAL: {a} vs {b} trailing rho={rho_trailing:+.3f} "
                    f"(baseline {baseline_str}); exceeds {CRITICAL_THRESHOLD} threshold "
                    f"— portfolio diversification assumption BROKEN"
                )
            elif pair_result.severity == "WARN":
                alerts.append(
                    f"WARN: {a} vs {b} trailing rho={rho_trailing:+.3f} "
                    f"(baseline {baseline_str}); exceeds {WARN_THRESHOLD} threshold "
                    f"— diversification drift watch"
                )

    # Overall severity = worst
    severities = [c.severity for c in correlations]
    if "CRITICAL" in severities:
        overall = "CRITICAL"
    elif "WARN" in severities:
        overall = "WARN"
    else:
        overall = "CLEAN"

    max_abs_rho = max((abs(c.rho_trailing) for c in correlations), default=float("nan"))
    max_drift   = max((c.abs_drift for c in correlations),         default=float("nan"))

    notes = [
        f"Window: trailing {window_weeks} weeks (~{window_weeks*5} trading days, "
        f"~{window_weeks*7} calendar days).",
        f"Sample: {n_weeks} weekly observations (some may overlap window start).",
        "Baseline rho from Sprint B in-sample replay 2014-2023 "
        "(data/portfolio_replay/v1_combined_replay_verdict.json).",
        "Forward paper trade (Sprint D-2) will accumulate into PaperTradeStrategyLog; "
        "Sprint G v2 will read both backtest parquets AND forward returns for full coverage.",
        f"Locked thresholds: WARN |rho|>{WARN_THRESHOLD}, CRITICAL |rho|>{CRITICAL_THRESHOLD} "
        "(per Sprint G spec; corresponds to portfolio Sharpe degradation 1.3→0.6).",
        "rho rises in stress regimes (Christoffersen-Errunza 2000); WARN tier expected to "
        "fire during 2018-Q4/2020-COVID/2022 historical windows.",
    ]

    return CorrelationSentinelReport(
        as_of           = as_of,
        window_weeks    = window_weeks,
        sample_n_weeks  = n_weeks,
        correlations    = correlations,
        max_abs_rho     = round(max_abs_rho, 4) if not math.isnan(max_abs_rho) else float("nan"),
        max_drift       = round(max_drift, 4)   if not math.isnan(max_drift)   else float("nan"),
        alerts          = alerts,
        severity        = overall,
        notes           = notes,
    )


def save_sentinel_report(report: CorrelationSentinelReport, save_path: Path) -> dict:
    """Save sentinel report to JSON."""
    payload = {
        "as_of":           report.as_of.isoformat(),
        "window_weeks":    report.window_weeks,
        "sample_n_weeks":  report.sample_n_weeks,
        "severity":        report.severity,
        "max_abs_rho":     report.max_abs_rho if not (isinstance(report.max_abs_rho, float) and math.isnan(report.max_abs_rho)) else None,
        "max_drift":       report.max_drift   if not (isinstance(report.max_drift,   float) and math.isnan(report.max_drift))   else None,
        "thresholds": {
            "warn":     WARN_THRESHOLD,
            "critical": CRITICAL_THRESHOLD,
        },
        "correlations": [
            {
                "pair_a":       c.pair_a,
                "pair_b":       c.pair_b,
                "rho_trailing": c.rho_trailing,
                # NaN baseline (e.g. AC_TLT_GLD pairs without in-sample data)
                # serialises to JSON null rather than the string "nan".
                "rho_baseline": None if math.isnan(c.rho_baseline) else c.rho_baseline,
                "abs_drift":    None if math.isnan(c.abs_drift)    else c.abs_drift,
                "severity":     c.severity,
            }
            for c in report.correlations
        ],
        "alerts":  report.alerts,
        "notes":   report.notes,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint G correlation drift sentinel")
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Report date YYYY-MM-DD (default: today UTC date)",
    )
    parser.add_argument(
        "--window-weeks", type=int, default=ROLLING_WINDOW_WEEKS_LOCKED,
        help=f"Trailing window in weeks (default {ROLLING_WINDOW_WEEKS_LOCKED})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report JSON to data/portfolio_replay/correlation_sentinel_<date>.json",
    )
    args = parser.parse_args()

    if args.as_of:
        as_of = datetime.date.fromisoformat(args.as_of)
    else:
        as_of = datetime.datetime.utcnow().date()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_correlation_sentinel(as_of, window_weeks=args.window_weeks)

    print(f"=== Correlation Sentinel — {report.as_of} ===")
    print(f"Window: trailing {report.window_weeks} weeks ({report.sample_n_weeks} samples)")
    print(f"Overall severity: {report.severity}")
    print(f"Max |rho|: {report.max_abs_rho}  |  Max drift vs baseline: {report.max_drift}")
    print()
    print("Pairwise correlations:")
    for c in report.correlations:
        marker = " ← " + c.severity if c.severity != "CLEAN" else ""
        drift_str = f"{c.abs_drift:+.3f}"
        print(f"  {c.pair_a:<10} vs {c.pair_b:<10}  rho={c.rho_trailing:+.3f}  "
              f"baseline={c.rho_baseline:+.3f}  drift={drift_str}{marker}")

    print()
    if report.alerts:
        print(f"Alerts ({len(report.alerts)}):")
        for a in report.alerts:
            print(f"  ! {a}")
    else:
        print("Alerts: 0 (all CLEAN)")

    print()
    print("Notes:")
    for n in report.notes:
        print(f"  - {n}")

    if args.save:
        save_path = Path(f"data/portfolio_replay/correlation_sentinel_{as_of.isoformat()}.json")
        save_sentinel_report(report, save_path)
        print(f"\nSaved to {save_path}")

    return 1 if report.severity == "CRITICAL" else 0


if __name__ == "__main__":
    sys.exit(main())
