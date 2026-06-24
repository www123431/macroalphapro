"""scripts/run_factor_library_d4a.py — In-sample (1996-2009) factor returns +
Stage 1 BHY-FDR + correlation filter → SELECTED_FACTORS_V1.

Spec: docs/spec_factor_library_v1.md §2.2 + §3.2 + §3.4
Pre-registration: in-sample = 1996-01-01 → 2009-12-31 (per spec §3.4); OOS reserved
for Stage 2.

Usage (must run locally; yfinance + universe DB required):
    python scripts/run_factor_library_d4a.py [--cache-dir data/factor_library_in_sample]

Output:
    1. data/factor_library_in_sample/closes.parquet        — daily prices cache
    2. data/factor_library_in_sample/factor_returns.parquet — monthly factor returns (5 cols)
    3. data/factor_library_in_sample/d4a_report.txt        — Stage 1 + selection report
    4. Console: SELECTED_FACTORS_V1 list (copy-paste back to engine/factor_library.py)

Deterministic: same input prices → same retained list. Re-running with cached
closes is fast (skips yfinance). Delete cache to force re-fetch.

Pre-test rigor disclosure (per feedback_pretest_experimental_rigor.md rule 8):
The script discloses ACTUAL coverage start date (factors may have NaN returns for
months before sufficient universe membership), not just nominal 1996-01.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

# Add project root to path so this can be invoked from anywhere
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.factor_library import (  # noqa: E402
    FACTOR_REGISTRY,
    compute_factor_returns_series,
    bhy_fdr_filter,
    select_independent_factors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Spec-locked constants
IN_SAMPLE_START = datetime.date(1996, 1,  1)
IN_SAMPLE_END   = datetime.date(2009, 12, 31)
CORR_THRESHOLD  = 0.7         # spec §2.2
FDR_ALPHA       = 0.05        # spec §3.2
BENCHMARK       = "SPY"


def _fetch_or_load_closes(cache_dir: pathlib.Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch all-time daily closes for universe + SPY benchmark, cached locally.

    Returns:
        (closes_df, asset_classes_dict)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "closes.parquet"

    if cache_path.exists():
        logger.info("Loading cached closes from %s", cache_path)
        closes = pd.read_parquet(cache_path)
        # Re-derive asset_classes from current universe registry
        from engine.universe_manager import get_universe_by_class
        universe_by_class = get_universe_by_class()
        asset_classes: dict[str, str] = {}
        for cls, sector_to_t in universe_by_class.items():
            for sector, ticker in sector_to_t.items():
                if ticker in closes.columns:
                    asset_classes[ticker] = cls
        return closes, asset_classes

    logger.info("No cache; fetching from yfinance (this needs ~2 min + network)")
    from engine.signal import _fetch_closes
    from engine.universe_manager import get_universe_by_class

    universe_by_class = get_universe_by_class()
    all_tickers = set()
    asset_classes: dict[str, str] = {}
    for cls, sector_to_t in universe_by_class.items():
        for sector, ticker in sector_to_t.items():
            all_tickers.add(ticker)
            asset_classes[ticker] = cls
    all_tickers.add(BENCHMARK)
    tickers = sorted(all_tickers)

    # Fetch from 1995-01 to 2010-12 (in-sample window + 1y lookback buffer)
    start = datetime.date(1995, 1, 1)
    end   = datetime.date(2010, 12, 31)
    logger.info("Fetching %d tickers from %s to %s", len(tickers), start, end)
    closes = _fetch_closes(tickers, start, end)
    if closes.empty:
        raise RuntimeError("yfinance returned empty DataFrame; check network connectivity")

    closes.to_parquet(cache_path)
    logger.info("Cached %d rows × %d tickers to %s", len(closes), closes.shape[1], cache_path)
    return closes, asset_classes


def _build_rebalance_dates(start: datetime.date, end: datetime.date) -> pd.DatetimeIndex:
    """Monthly rebalance: last business day of each month in [start, end]."""
    return pd.date_range(start=start, end=end, freq="BME")  # Business Month End


def main(cache_dir: str = "data/factor_library_in_sample") -> int:
    cache_path = pathlib.Path(cache_dir)
    closes, asset_classes = _fetch_or_load_closes(cache_path)
    if BENCHMARK not in closes.columns:
        raise RuntimeError(f"Benchmark {BENCHMARK} missing from closes")
    benchmark_close = closes[BENCHMARK]
    universe_closes = closes.drop(columns=[BENCHMARK])

    rebalance_dates = _build_rebalance_dates(IN_SAMPLE_START, IN_SAMPLE_END)
    logger.info("Built %d rebalance dates from %s to %s",
                len(rebalance_dates), rebalance_dates[0].date(), rebalance_dates[-1].date())

    # Compute monthly factor returns for each of the 5 v1 candidates
    factor_returns_dict: dict[str, pd.Series] = {}
    for factor_id in FACTOR_REGISTRY:
        logger.info("Computing %s factor returns...", factor_id)
        s = compute_factor_returns_series(
            factor_id,
            universe_closes,
            rebalance_dates,
            asset_classes=asset_classes,
            benchmark_close=benchmark_close,
        )
        factor_returns_dict[factor_id] = s
        n_valid = int(s.notna().sum())
        first_valid = s.first_valid_index()
        logger.info("  %s: %d/%d valid months; first valid = %s",
                    factor_id, n_valid, len(s),
                    first_valid.date() if first_valid is not None else "NEVER")

    factor_returns = pd.DataFrame(factor_returns_dict)
    factor_returns.to_parquet(cache_path / "factor_returns.parquet")

    # Coverage disclosure (rule 8 — show actual vs nominal coverage)
    coverage = {}
    for f in factor_returns.columns:
        s = factor_returns[f].dropna()
        coverage[f] = {
            "n_valid":     int(len(s)),
            "first_valid": s.index.min().date().isoformat() if len(s) else "NEVER",
            "last_valid":  s.index.max().date().isoformat() if len(s) else "NEVER",
        }

    # Stage 1 BHY-FDR per-factor inclusion test
    # NW HAC t-stat for mean ≠ 0 (one-sided positive)
    p_values: dict[str, float] = {}
    sharpe_annualized: dict[str, float] = {}
    for f in factor_returns.columns:
        s = factor_returns[f].dropna()
        if len(s) < 24:  # need ≥ 2 years for stable t-stat
            p_values[f] = float("nan")
            sharpe_annualized[f] = float("nan")
            continue
        try:
            import statsmodels.api as sm
            model = sm.OLS(s.values, np.ones(len(s))).fit(
                cov_type="HAC", cov_kwds={"maxlags": int(len(s) ** (1/3))}
            )
            t_stat = float(model.tvalues[0])
            # One-sided p-value (test mean > 0)
            from scipy.stats import norm
            p_one_sided = 1.0 - norm.cdf(t_stat)
            p_values[f] = float(p_one_sided)
        except Exception as exc:
            logger.warning("NW t-stat failed for %s: %s", f, exc)
            p_values[f] = float("nan")
        sharpe_annualized[f] = float((s.mean() / s.std(ddof=1)) * np.sqrt(12)) if s.std(ddof=1) > 0 else float("nan")

    bhy_pass = bhy_fdr_filter(p_values, alpha=FDR_ALPHA)

    # Among BHY-pass factors, run greedy correlation filter (spec §2.2)
    bhy_pass_factors = [f for f, ok in bhy_pass.items() if ok]
    if len(bhy_pass_factors) < 1:
        retained = []
    else:
        in_sample_for_corr = factor_returns[bhy_pass_factors].dropna(how="all")
        # Drop rows with all-NaN; select_independent_factors handles within-column NaN
        if in_sample_for_corr.empty:
            retained = []
        else:
            retained = select_independent_factors(
                in_sample_for_corr,
                candidates=bhy_pass_factors,
                corr_threshold=CORR_THRESHOLD,
            )

    # Build report
    report_lines = [
        "=" * 70,
        "Factor Library v1 — Stage 1 In-Sample Selection (W1 D4a)",
        f"Spec: docs/spec_factor_library_v1.md §2.2 + §3.2",
        f"Run date: {datetime.date.today().isoformat()}",
        f"In-sample window (nominal per spec §3.4): {IN_SAMPLE_START} to {IN_SAMPLE_END}",
        "=" * 70,
        "",
        "Coverage (actual valid months per factor):",
    ]
    for f, cov in coverage.items():
        report_lines.append(f"  {f:18s} n={cov['n_valid']:3d}  first={cov['first_valid']}  last={cov['last_valid']}")

    report_lines.extend([
        "",
        "Stage 1 — Per-factor NW HAC t-stat (one-sided H1: mean > 0):",
        f"{'factor':<18s} {'in_sample_Sharpe':>16s} {'p_one_sided':>14s} {'BHY_pass':>10s}",
    ])
    for f in factor_returns.columns:
        s = sharpe_annualized.get(f, float("nan"))
        p = p_values.get(f, float("nan"))
        ok = "PASS" if bhy_pass.get(f, False) else "fail"
        report_lines.append(f"{f:<18s} {s:>16.3f} {p:>14.4f} {ok:>10s}")

    report_lines.extend([
        "",
        f"BHY-FDR α={FDR_ALPHA}, N=5 candidates → c(5)=Σ(1/k)≈2.283",
        f"BHY-pass factors (subset entering corr filter): {bhy_pass_factors}",
        "",
        f"Stage 2 — Greedy corr filter (Spearman, threshold={CORR_THRESHOLD}):",
        f"  Retained factors (in-sample Sharpe descending): {retained}",
        "",
        f"SELECTED_FACTORS_V1 = {tuple(retained)!r}",
        "",
        "Spec §3.3 PASS gate Stage 1 condition:",
        f"  retained ≥ 3 factors → {'OK' if len(retained) >= 3 else 'STAGE_1_FAIL'}",
        "=" * 70,
    ])

    report = "\n".join(report_lines)
    print(report)

    report_path = cache_path / "d4a_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report saved to %s", report_path)

    print()
    print(">>> NEXT STEPS:")
    print(f"  1. Review {report_path}")
    print(f"  2. Update engine/factor_library.py: SELECTED_FACTORS_V1 = {tuple(retained)!r}")
    print(f"  3. amend_spec(path='engine/factor_library.py', kind='clarification',")
    print(f"               reason='W1 D4a: locked SELECTED_FACTORS_V1 from in-sample analysis ...')")
    print(f"  4. Run pytest + Tier R audit to verify clean")
    return 0 if len(retained) >= 1 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/factor_library_in_sample",
                        help="Directory for closes.parquet + factor_returns.parquet + report")
    args = parser.parse_args()
    sys.exit(main(args.cache_dir))
