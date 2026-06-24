"""scripts/run_multivariate_msm_d6.py — Multivariate MSM v1 OOS verdict run.

Spec: docs/spec_multivariate_msm_v1.md §3.1 / §3.2 / §3.3 / §3.4 / §3.5
Pre-registration: spec_id=41 (registered 2026-05-09)

Walk-forward backtest:
  In-sample (HARKing protection only; no hyperparam tuning per spec §3.5): 2010-01 to 2018-12
  OOS test window:                                                         2019-01 to 2024-12 (72 months)

For each month-end in 2010-01 to 2024-12:
  - Run multivariate MSM (3 features) → (p_risk_on, fallback_flag)
    On ConvergenceError / InsufficientData / MissingFeatureData:
      - Increment fallback counter
      - Use univariate p_risk_on as fallback (per spec §4.2 fallback chain)
  - Run univariate MSM (existing engine.regime path) → p_risk_on baseline

Then on the OOS slice (2019-01 onward):
  - Construct overlay returns: position = 2*p − 1; overlay = position × SPY_monthly_return
  - Compute ΔŜ multivariate − univariate annualized
  - Politis-Romano stationary bootstrap with Politis-White auto-block (1000 resamples)
  - Memmel Z descriptive only (per spec §3.1 underpowered honesty)
  - Apply spec §3.2 decision rule

Local-run requirement:
  - FRED CSV (network)
  - yfinance daily SPY + ^VIX (network)

Usage (must run on local machine; sandbox blocks DNS):
    python scripts/run_multivariate_msm_d6.py [--cache-dir data/multivariate_msm/]

Output:
    data/multivariate_msm/walk_forward_probs.parquet  — per-month p_risk_on (multi + uni) + fallback_flag
    data/multivariate_msm/spy_monthly.parquet         — SPY monthly returns 2010-2024
    data/multivariate_msm/d6_verdict.txt              — full verdict report

Pre-test rigor disclosure (rule-8): expected achieved_power at observed ρ̂ ≈ 0.05
(severely underpowered per spec §3.3); verdict treated as descriptive
effect-size + bootstrap CI, not formal hypothesis test.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.multivariate_msm_verdict import (  # noqa: E402
    compute_overlay_returns,
    compute_verdict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Spec-locked windows
IN_SAMPLE_START = datetime.date(2010, 1, 1)
IN_SAMPLE_END   = datetime.date(2018, 12, 31)
OOS_START       = datetime.date(2019, 1, 1)
OOS_END         = datetime.date(2024, 12, 31)
WALK_FORWARD_START = IN_SAMPLE_START   # we still run in-sample months for full-window robustness
WALK_FORWARD_END   = OOS_END


def _walk_forward_probs(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """For each month-end, fit multivariate + univariate MSM and capture filtered
    p_risk_on at the as_of date. Records fallback flag when multivariate raises.

    Returns DataFrame with columns:
        p_multivariate, p_univariate, multivariate_failed (bool)
    indexed by rebalance_dates.
    """
    from engine.regime import (
        _get_regime_multivariate, get_regime_on,
        ConvergenceError, InsufficientData, MissingFeatureData,
    )
    rows = []
    for ts in rebalance_dates:
        d = ts.date()
        # Multivariate path with explicit error handling
        p_multi = float("nan")
        multi_failed = False
        try:
            r_multi = _get_regime_multivariate(as_of=d, train_end=d)
            p_multi = float(r_multi.p_risk_on)
        except (ConvergenceError, InsufficientData, MissingFeatureData) as exc:
            multi_failed = True
            logger.info("multivariate fallback at %s: %s: %s", d, type(exc).__name__, exc)
        except Exception as exc:
            multi_failed = True
            logger.warning("multivariate UNEXPECTED failure at %s: %s", d, exc)

        # Univariate via existing get_regime_on (use_multivariate=False forces univariate path only)
        try:
            r_uni = get_regime_on(as_of=d, train_end=d, use_multivariate=False)
            p_uni = float(r_uni.p_risk_on)
        except Exception as exc:
            logger.warning("univariate failure at %s: %s", d, exc)
            p_uni = float("nan")

        # On multivariate failure, use univariate as fallback per spec §4.2
        if multi_failed and not np.isnan(p_uni):
            p_multi = p_uni

        rows.append({
            "date":               ts,
            "p_multivariate":     p_multi,
            "p_univariate":       p_uni,
            "multivariate_failed": multi_failed,
        })
        logger.info("walk-forward %s: p_multi=%.3f p_uni=%.3f failed=%s",
                    d, p_multi, p_uni, multi_failed)

    df = pd.DataFrame(rows).set_index("date")
    return df


def _fetch_spy_monthly_returns(start: datetime.date, end: datetime.date) -> pd.Series:
    """Monthly SPY returns from yfinance (last close of each month)."""
    from engine.signal import _fetch_closes
    closes = _fetch_closes(["SPY"], start - datetime.timedelta(days=10), end)
    if closes.empty or "SPY" not in closes.columns:
        return pd.Series(dtype=float)
    spy = closes["SPY"].resample("ME").last().dropna()
    return spy.pct_change().dropna()


def main(cache_dir: str = "data/multivariate_msm") -> int:
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    rebalance = pd.date_range(WALK_FORWARD_START, WALK_FORWARD_END, freq="BME")
    logger.info("Walk-forward over %d month-ends from %s to %s",
                len(rebalance), rebalance[0].date(), rebalance[-1].date())

    probs_cache = cache_path / "walk_forward_probs.parquet"
    if probs_cache.exists():
        logger.info("Using cached walk-forward probs from %s", probs_cache)
        probs = pd.read_parquet(probs_cache)
        probs.index = pd.to_datetime(probs.index)
    else:
        probs = _walk_forward_probs(rebalance)
        probs.to_parquet(probs_cache)
        logger.info("Cached walk-forward probs to %s", probs_cache)

    spy_cache = cache_path / "spy_monthly.parquet"
    if spy_cache.exists():
        spy = pd.read_parquet(spy_cache).iloc[:, 0]
        spy.index = pd.to_datetime(spy.index)
    else:
        spy = _fetch_spy_monthly_returns(WALK_FORWARD_START, WALK_FORWARD_END)
        if not spy.empty:
            spy.to_frame("spy_ret").to_parquet(spy_cache)

    if spy.empty:
        logger.error("SPY monthly returns unavailable — cannot compute overlay")
        return 2

    # Construct overlay returns (full window, then slice OOS)
    overlay_multi = compute_overlay_returns(probs["p_multivariate"], spy)
    overlay_uni   = compute_overlay_returns(probs["p_univariate"],  spy)

    oos_start_ts = pd.Timestamp(OOS_START)
    oos_end_ts   = pd.Timestamp(OOS_END)
    overlay_multi_oos = overlay_multi.loc[(overlay_multi.index >= oos_start_ts) & (overlay_multi.index <= oos_end_ts)]
    overlay_uni_oos   = overlay_uni.loc[(overlay_uni.index >= oos_start_ts) & (overlay_uni.index <= oos_end_ts)]

    probs_oos = probs.loc[(probs.index >= oos_start_ts) & (probs.index <= oos_end_ts)]
    fallback_rate = float(probs_oos["multivariate_failed"].mean()) if len(probs_oos) > 0 else 0.0

    verdict = compute_verdict(
        overlay_multi_oos, overlay_uni_oos,
        fallback_rate=fallback_rate,
        n_resamples=1000,
    )

    report_lines = [
        "=" * 70,
        "Multivariate MSM v1 — OOS Verdict (W1 D6)",
        f"Spec: docs/spec_multivariate_msm_v1.md §3.1 + §3.2 + §3.4",
        f"Run date: {datetime.date.today().isoformat()}",
        f"In-sample (HARKing protection): {IN_SAMPLE_START} to {IN_SAMPLE_END}",
        f"OOS window:                     {OOS_START} to {OOS_END}",
        f"Walk-forward total months:      {len(rebalance)}",
        f"OOS months captured:            {verdict.n_oos_months}",
        "=" * 70,
        "",
        "Effect-size estimates (DESCRIPTIVE; spec §3.1 underpowered honest):",
        f"  Sharpe(multivariate overlay)  = {verdict.sharpe_multivariate:+.3f}",
        f"  Sharpe(univariate overlay)    = {verdict.sharpe_univariate:+.3f}",
        f"  ΔŜ                            = {verdict.delta_sharpe:+.3f}",
        "",
        "Bootstrap (Politis-Romano stationary + Politis-White auto-block):",
        f"  95% CI for ΔŜ                 = [{verdict.bootstrap_ci_lower:+.3f}, {verdict.bootstrap_ci_upper:+.3f}]",
        f"  Block size                    = {verdict.bootstrap_block_size}",
        f"  N resamples                   = {verdict.bootstrap_n_resamples}",
        "",
        "Memmel Z (descriptive secondary metric only, spec §3.1):",
        f"  Z                             = {verdict.memmel_z:+.3f}",
        f"  Paired ρ̂                      = {verdict.paired_correlation:+.3f}",
        f"  Achieved power at observed ρ̂  = {verdict.achieved_power_descriptive:.1%}",
        "",
        "Fallback diagnostic (spec §3.4):",
        f"  Multivariate fallback rate    = {verdict.fallback_rate:.1%}",
        f"  Tier:                         {'UNINTERPRETABLE' if verdict.fallback_rate >= 0.50 else 'STRONG_CAVEAT' if verdict.fallback_rate >= 0.25 else 'SOFT_CAVEAT' if verdict.fallback_rate >= 0.10 else 'NORMAL'}",
        "",
        f"DECISION: {verdict.decision}",
        "",
        "Spec §3.2 PASS gate (locked): ΔŜ ≥ +0.10 AND bootstrap CI 下界 > 0 AND fallback < 50%",
        "=" * 70,
    ]
    report = "\n".join(report_lines)
    print(report)
    (cache_path / "d6_verdict.txt").write_text(report, encoding="utf-8")

    print()
    print(">>> NEXT STEPS:")
    print(f"  1. Review {cache_path / 'd6_verdict.txt'}")
    print(f"  2. If PASS: stage PendingApproval(production_signal_swap)")
    print(f"     supervisor selects c ∈ [0.3, 0.7] per spec §3.6 + amends regime.py")
    print(f"     to flip _USE_MULTIVARIATE_REGIME = True")
    print(f"  3. If FAIL/MARGINAL: write verdict to docs/decisions/multivariate_msm_v1_<DATE>.md")
    print(f"     amend_spec(kind='superseded') on docs/spec_multivariate_msm_v1.md")
    print(f"  4. Either way: run pytest + Tier R audit")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/multivariate_msm",
                        help="Directory for walk-forward probs + SPY + verdict cache")
    args = parser.parse_args()
    sys.exit(main(args.cache_dir))
