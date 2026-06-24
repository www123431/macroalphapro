"""scripts/run_multivariate_msm_v3_d6.py — Multivariate MSM v3 verdict run.

Spec: docs/spec_multivariate_msm_v3.md (registered post-2026-05-08; supersedes v2
which was withdrawn after D4 gate empirically invalidated HYG-LQD ETF return-spread
proxy: Pearson r = -0.03 vs FRED OAS-diff).

v3 = 2-feature multivariate MSM (yield_spread + VIX), drops ig_hy_credit_spread
entirely. Retains v2 D1 (VIX anchor) + D3 (ternary overlay) + D5 (descriptive verdict).

Usage (must run locally; FRED API + yfinance network needed):
    python scripts/run_multivariate_msm_v3_d6.py [--cache-dir data/multivariate_msm_v3/]

Output:
    data/multivariate_msm_v3/walk_forward_probs_v3.parquet
    data/multivariate_msm_v3/spy_monthly.parquet
    data/multivariate_msm_v3/d6_v3_verdict.txt

Decision labels:
    DESCRIPTIVE_POSITIVE       ΔŜ ≥ +0.05 AND CI lower > 0      → supervisor PendingApproval
    DESCRIPTIVE_INSUFFICIENT   non-negative below threshold     → no ship
    DESCRIPTIVE_NEGATIVE       ΔŜ < 0                           → 9th falsification (sound design)
    UNINTERPRETABLE            fallback rate ≥ 50%              → spec §3.4
"""
from __future__ import annotations

import argparse
import datetime
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.multivariate_msm_verdict import (  # noqa: E402
    compute_overlay_returns_ternary,
    compute_verdict_v2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IN_SAMPLE_START    = datetime.date(2010, 1, 1)
IN_SAMPLE_END      = datetime.date(2018, 12, 31)
OOS_START          = datetime.date(2019, 1, 1)
OOS_END            = datetime.date(2024, 12, 31)
WALK_FORWARD_START = IN_SAMPLE_START
WALK_FORWARD_END   = OOS_END


def _walk_forward_probs_v3(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    from engine.regime import (
        _get_regime_multivariate_v3, get_regime_on,
        ConvergenceError, InsufficientData, MissingFeatureData,
    )
    rows = []
    for ts in rebalance_dates:
        d = ts.date()
        p_multi, multi_failed = float("nan"), False
        try:
            r_multi = _get_regime_multivariate_v3(as_of=d, train_end=d)
            p_multi = float(r_multi.p_risk_on)
        except (ConvergenceError, InsufficientData, MissingFeatureData) as exc:
            multi_failed = True
            logger.info("multivariate v3 fallback at %s: %s: %s", d, type(exc).__name__, exc)
        except Exception as exc:
            multi_failed = True
            logger.warning("multivariate v3 UNEXPECTED at %s: %s", d, exc)

        try:
            r_uni = get_regime_on(as_of=d, train_end=d, use_multivariate=False)
            p_uni = float(r_uni.p_risk_on)
        except Exception as exc:
            logger.warning("univariate failure at %s: %s", d, exc)
            p_uni = float("nan")

        if multi_failed and not np.isnan(p_uni):
            p_multi = p_uni  # spec §4.2 fallback chain

        rows.append({
            "date":                ts,
            "p_multivariate_v3":   p_multi,
            "p_univariate":        p_uni,
            "multivariate_failed": multi_failed,
        })
        logger.info("walk-forward v3 %s: p_multi=%.3f p_uni=%.3f failed=%s",
                    d, p_multi, p_uni, multi_failed)
    return pd.DataFrame(rows).set_index("date")


def _fetch_spy_monthly_returns(start: datetime.date, end: datetime.date) -> pd.Series:
    from engine.signal import _fetch_closes
    closes = _fetch_closes(["SPY"], start - datetime.timedelta(days=10), end)
    if closes.empty or "SPY" not in closes.columns:
        return pd.Series(dtype=float)
    spy = closes["SPY"].resample("ME").last().dropna()
    return spy.pct_change().dropna()


def main(cache_dir: str = "data/multivariate_msm_v3") -> int:
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use freq="ME" (calendar month-end) to align with SPY .resample("ME") index.
    # Original v3 run (2026-05-08) used freq="BME" which produced business-month-end
    # labels (e.g. 2024-06-28 Friday) while SPY index was 2024-06-30 Sunday — caused
    # 21 of 72 OOS months to drop on intersection. Calendar-end labels carry the same
    # last-trading-day data; only label changes. Bug fix, not hypothesis change.
    rebalance = pd.date_range(WALK_FORWARD_START, WALK_FORWARD_END, freq="ME")
    logger.info("Walk-forward over %d month-ends (ME-aligned) from %s to %s",
                len(rebalance), rebalance[0].date(), rebalance[-1].date())

    probs_cache = cache_path / "walk_forward_probs_v3.parquet"
    if probs_cache.exists():
        logger.info("Using cached walk-forward probs from %s", probs_cache)
        probs = pd.read_parquet(probs_cache)
        probs.index = pd.to_datetime(probs.index)
    else:
        probs = _walk_forward_probs_v3(rebalance)
        probs.to_parquet(probs_cache)

    spy_cache = cache_path / "spy_monthly.parquet"
    if spy_cache.exists():
        spy = pd.read_parquet(spy_cache).iloc[:, 0]
        spy.index = pd.to_datetime(spy.index)
    else:
        spy = _fetch_spy_monthly_returns(WALK_FORWARD_START, WALK_FORWARD_END)
        if not spy.empty:
            spy.to_frame("spy_ret").to_parquet(spy_cache)

    if spy.empty:
        logger.error("SPY monthly returns unavailable")
        return 3

    overlay_multi = compute_overlay_returns_ternary(probs["p_multivariate_v3"], spy)
    overlay_uni   = compute_overlay_returns_ternary(probs["p_univariate"],      spy)

    oos_start_ts = pd.Timestamp(OOS_START)
    oos_end_ts   = pd.Timestamp(OOS_END)
    overlay_multi_oos = overlay_multi.loc[(overlay_multi.index >= oos_start_ts) & (overlay_multi.index <= oos_end_ts)]
    overlay_uni_oos   = overlay_uni.loc[(overlay_uni.index   >= oos_start_ts) & (overlay_uni.index   <= oos_end_ts)]

    probs_oos     = probs.loc[(probs.index >= oos_start_ts) & (probs.index <= oos_end_ts)]
    fallback_rate = float(probs_oos["multivariate_failed"].mean()) if len(probs_oos) > 0 else 0.0

    verdict = compute_verdict_v2(
        overlay_multi_oos, overlay_uni_oos,
        fallback_rate=fallback_rate, n_resamples=1000,
    )

    report_lines = [
        "=" * 70,
        "Multivariate MSM v3 — OOS Verdict (W1 D6 v3, 2-feature post-D4-validation)",
        f"Spec: docs/spec_multivariate_msm_v3.md",
        f"Run date: {datetime.date.today().isoformat()}",
        f"Features (locked): yield_spread + VIX (ig_hy dropped per v2 D4 invalidation)",
        f"In-sample (HARKing protection): {IN_SAMPLE_START} to {IN_SAMPLE_END}",
        f"OOS window:                     {OOS_START} to {OOS_END}",
        f"OOS months captured:            {verdict.n_oos_months}",
        "=" * 70,
        "",
        "Effect-size estimates (DESCRIPTIVE only per spec §3.2 D5 framework):",
        f"  Sharpe(multivariate v3 overlay) = {verdict.sharpe_multivariate:+.3f}",
        f"  Sharpe(univariate overlay)      = {verdict.sharpe_univariate:+.3f}",
        f"  ΔŜ                              = {verdict.delta_sharpe:+.3f}",
        "",
        "Bootstrap (Politis-Romano stationary + Politis-White auto-block):",
        f"  95% CI for ΔŜ                   = [{verdict.bootstrap_ci_lower:+.3f}, {verdict.bootstrap_ci_upper:+.3f}]",
        f"  CI lower > 0?                   = {verdict.ci_lower_above_zero}",
        f"  CI lower ≥ +0.05?               = {verdict.ci_lower_above_threshold}  (ship-suggesting heuristic)",
        f"  Block size                      = {verdict.bootstrap_block_size}",
        "",
        "Memmel Z (descriptive secondary):",
        f"  Z                               = {verdict.memmel_z:+.3f}",
        f"  Paired ρ̂                        = {verdict.paired_correlation:+.3f}",
        f"  Achieved power (descriptive)    = {verdict.achieved_power_descriptive:.1%}",
        "",
        "Fallback diagnostic:",
        f"  Multivariate v3 fallback rate   = {verdict.fallback_rate:.1%}",
        f"  Tier:                           {'UNINTERPRETABLE' if verdict.fallback_rate >= 0.50 else 'STRONG_CAVEAT' if verdict.fallback_rate >= 0.25 else 'SOFT_CAVEAT' if verdict.fallback_rate >= 0.10 else 'NORMAL'}",
        "",
        f"DECISION LABEL (descriptive): {verdict.decision_label}",
        "",
        "Spec_v3 §3.2 supervisor framework:",
        "  DESCRIPTIVE_POSITIVE      → supervisor may PendingApproval(production_signal_swap)",
        "  DESCRIPTIVE_INSUFFICIENT  → no ship; descriptive disclosure only",
        "  DESCRIPTIVE_NEGATIVE      → 9th falsification chain entry (architecture sound)",
        "  UNINTERPRETABLE           → spec §3.4 fallback ≥ 50%",
        "=" * 70,
    ]
    report = "\n".join(report_lines)
    print(report)
    (cache_path / "d6_v3_verdict.txt").write_text(report, encoding="utf-8")

    print()
    print(">>> NEXT STEPS:")
    print(f"  1. Review {cache_path / 'd6_v3_verdict.txt'}")
    print(f"  2. Paste verdict report back to assistant for housekeeping")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/multivariate_msm_v3")
    args = parser.parse_args()
    sys.exit(main(args.cache_dir))
