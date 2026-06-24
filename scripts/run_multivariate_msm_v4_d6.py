"""scripts/run_multivariate_msm_v4_d6.py — Multivariate MSM v4 OOS verdict.

Spec: docs/spec_multivariate_msm_v4_narrative.md (id=47, single-source FOMC
amendment 2026-05-08; D3 z-norm μ/σ locked; D2c §3.8 5/5 PASS).

v4 = 3-feature multivariate MSM (yield_spread + VIX + narrative_score).
Paired test: v4 overlay vs **v3 overlay** (production baseline) on OOS 2019-2024.

Per spec §3.1: ΔŜ = Sharpe(v4 overlay) − Sharpe(v3 overlay).
Per spec §3.2 decision rule:
    DESCRIPTIVE_POSITIVE      ΔŜ ≥ +0.05 AND CI lower > 0  → supervisor MAY PendingApproval(v3→v4)
    DESCRIPTIVE_INSUFFICIENT  non-negative below threshold → no ship
    DESCRIPTIVE_NEGATIVE      ΔŜ < 0                       → multivariate hypothesis path PERMANENTLY CLOSED
    UNINTERPRETABLE           v4 fallback ≥ 50%            → spec §3.4

Output:
    data/multivariate_msm_v4/walk_forward_probs_v4.parquet
    data/multivariate_msm_v4/walk_forward_probs_v3_paired.parquet  (paired snapshot)
    data/multivariate_msm_v4/spy_monthly.parquet
    data/multivariate_msm_v4/d6_v4_verdict.txt
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

# Spec §3.5 OOS window (matches v3 for paired comparability)
IN_SAMPLE_START    = datetime.date(1994, 1, 1)
IN_SAMPLE_END      = datetime.date(2018, 12, 31)
OOS_START          = datetime.date(2019, 1, 1)
OOS_END            = datetime.date(2024, 12, 31)
# Walk-forward starts at v3 reference start (2010) so HMM has same warmup as v3
WALK_FORWARD_START = datetime.date(2010, 1, 1)
WALK_FORWARD_END   = OOS_END


def _walk_forward_probs_v4_and_v3(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Run BOTH v4 and v3 at each rebalance date for clean paired test.

    Per spec §3.1: ΔŜ comparison requires BOTH overlays computed at SAME
    rebalance dates. Running v3 alongside v4 (rather than reusing v3 cache)
    guarantees no date-mismatch confound.
    """
    from engine.regime import (
        _get_regime_multivariate_v4,
        _get_regime_multivariate_v3,
        get_regime_on,
        ConvergenceError, InsufficientData, MissingFeatureData,
    )
    rows = []
    for ts in rebalance_dates:
        d = ts.date()

        # v4 (3-feature)
        p_v4, v4_failed = float("nan"), False
        try:
            r_v4 = _get_regime_multivariate_v4(as_of=d, train_end=d)
            p_v4 = float(r_v4.p_risk_on)
        except (ConvergenceError, InsufficientData, MissingFeatureData) as exc:
            v4_failed = True
            logger.info("v4 fallback at %s: %s: %s", d, type(exc).__name__, exc)
        except Exception as exc:
            v4_failed = True
            logger.warning("v4 UNEXPECTED at %s: %s", d, exc)

        # v3 (2-feature, production baseline)
        p_v3, v3_failed = float("nan"), False
        try:
            r_v3 = _get_regime_multivariate_v3(as_of=d, train_end=d)
            p_v3 = float(r_v3.p_risk_on)
        except (ConvergenceError, InsufficientData, MissingFeatureData) as exc:
            v3_failed = True
            logger.info("v3 fallback at %s: %s: %s", d, type(exc).__name__, exc)
        except Exception as exc:
            v3_failed = True
            logger.warning("v3 UNEXPECTED at %s: %s", d, exc)

        # Univariate (last-resort fallback per spec §4.2 chain)
        try:
            r_uni = get_regime_on(as_of=d, train_end=d, use_multivariate=False)
            p_uni = float(r_uni.p_risk_on)
        except Exception as exc:
            logger.warning("univariate failure at %s: %s", d, exc)
            p_uni = float("nan")

        # Apply spec §4.2 fallback chain to BOTH v4 and v3 independently:
        # v4 → v3 → univariate, but for paired test we only need v4 and v3.
        # If v4 fails, fall back to v3's value (spec-compliant chain).
        # If v3 fails, fall back to univariate.
        if v4_failed and not np.isnan(p_v3):
            p_v4 = p_v3
        elif v4_failed and not np.isnan(p_uni):
            p_v4 = p_uni
        if v3_failed and not np.isnan(p_uni):
            p_v3 = p_uni

        rows.append({
            "date":         ts,
            "p_v4":         p_v4,
            "p_v3":         p_v3,
            "p_univariate": p_uni,
            "v4_failed":    v4_failed,
            "v3_failed":    v3_failed,
        })
        logger.info("walk-fwd %s: p_v4=%.3f p_v3=%.3f p_uni=%.3f v4_failed=%s",
                    d, p_v4, p_v3, p_uni, v4_failed)
    return pd.DataFrame(rows).set_index("date")


def _fetch_spy_monthly_returns(start: datetime.date, end: datetime.date) -> pd.Series:
    from engine.signal import _fetch_closes
    closes = _fetch_closes(["SPY"], start - datetime.timedelta(days=10), end)
    if closes.empty or "SPY" not in closes.columns:
        return pd.Series(dtype=float)
    spy = closes["SPY"].resample("ME").last().dropna()
    return spy.pct_change().dropna()


def main(cache_dir: str = "data/multivariate_msm_v4") -> int:
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    rebalance = pd.date_range(WALK_FORWARD_START, WALK_FORWARD_END, freq="ME")
    logger.info("Walk-forward over %d month-ends from %s to %s",
                len(rebalance), rebalance[0].date(), rebalance[-1].date())

    probs_cache = cache_path / "walk_forward_probs_v4.parquet"
    if probs_cache.exists():
        logger.info("Using cached walk-forward probs from %s", probs_cache)
        probs = pd.read_parquet(probs_cache)
        probs.index = pd.to_datetime(probs.index)
    else:
        probs = _walk_forward_probs_v4_and_v3(rebalance)
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

    overlay_v4 = compute_overlay_returns_ternary(probs["p_v4"], spy)
    overlay_v3 = compute_overlay_returns_ternary(probs["p_v3"], spy)

    oos_start_ts = pd.Timestamp(OOS_START)
    oos_end_ts   = pd.Timestamp(OOS_END)
    overlay_v4_oos = overlay_v4.loc[(overlay_v4.index >= oos_start_ts) & (overlay_v4.index <= oos_end_ts)]
    overlay_v3_oos = overlay_v3.loc[(overlay_v3.index >= oos_start_ts) & (overlay_v3.index <= oos_end_ts)]

    probs_oos     = probs.loc[(probs.index >= oos_start_ts) & (probs.index <= oos_end_ts)]
    fallback_rate = float(probs_oos["v4_failed"].mean()) if len(probs_oos) > 0 else 0.0

    # compute_verdict_v2 takes (multi_overlay, baseline_overlay).
    # For v4 paired test: multi=v4 overlay, baseline=v3 overlay.
    verdict = compute_verdict_v2(
        overlay_v4_oos, overlay_v3_oos,
        fallback_rate=fallback_rate, n_resamples=1000,
    )

    report_lines = [
        "=" * 72,
        "Multivariate MSM v4 — OOS Verdict (W3 D5, paired vs v3 production baseline)",
        f"Spec: docs/spec_multivariate_msm_v4_narrative.md (id=47)",
        f"Run date: {datetime.date.today().isoformat()}",
        "Features (locked v4): yield_spread + VIX + narrative_score (single-source FOMC)",
        "Baseline (locked v3): yield_spread + VIX",
        f"In-sample (z-norm lock 2026-05-08): {IN_SAMPLE_START} to {IN_SAMPLE_END}",
        f"OOS window:                          {OOS_START} to {OOS_END}",
        f"OOS months captured:                 {verdict.n_oos_months}",
        f"Walk-forward window:                 {WALK_FORWARD_START} to {WALK_FORWARD_END}",
        "=" * 72,
        "",
        "Effect-size estimates (DESCRIPTIVE only per spec §3.2 / §3.3 underpowered):",
        f"  Sharpe(v4 overlay)              = {verdict.sharpe_multivariate:+.3f}",
        f"  Sharpe(v3 overlay)              = {verdict.sharpe_univariate:+.3f}",
        f"  ΔŜ (v4 - v3)                    = {verdict.delta_sharpe:+.3f}",
        "",
        "Bootstrap (Politis-Romano stationary + Politis-White auto-block, 1000 resamples):",
        f"  95% CI for ΔŜ                   = [{verdict.bootstrap_ci_lower:+.3f}, {verdict.bootstrap_ci_upper:+.3f}]",
        f"  CI lower > 0?                   = {verdict.ci_lower_above_zero}",
        f"  CI lower ≥ +0.05?               = {verdict.ci_lower_above_threshold}  (ship-suggesting heuristic)",
        f"  Block size                      = {verdict.bootstrap_block_size}",
        "",
        "Memmel (2003) Z (descriptive secondary):",
        f"  Z                               = {verdict.memmel_z:+.3f}",
        f"  Paired ρ̂ (v4 vs v3)             = {verdict.paired_correlation:+.3f}",
        f"  Achieved power (descriptive)    = {verdict.achieved_power_descriptive:.1%}",
        "",
        "Fallback diagnostic:",
        f"  v4 fallback rate                = {verdict.fallback_rate:.1%}",
        f"  Tier:                           {'UNINTERPRETABLE' if verdict.fallback_rate >= 0.50 else 'STRONG_CAVEAT' if verdict.fallback_rate >= 0.25 else 'SOFT_CAVEAT' if verdict.fallback_rate >= 0.10 else 'NORMAL'}",
        "",
        f"DECISION LABEL (descriptive): {verdict.decision_label}",
        "",
        "Spec_v4 §3.2 framework:",
        "  DESCRIPTIVE_POSITIVE      → supervisor MAY PendingApproval(v3→v4 production swap)",
        "  DESCRIPTIVE_INSUFFICIENT  → no ship; descriptive disclosure only",
        "  DESCRIPTIVE_NEGATIVE      → 8th falsification (multivariate path PERMANENTLY CLOSED)",
        "  UNINTERPRETABLE           → spec §3.4 fallback ≥ 50%",
        "=" * 72,
    ]
    report = "\n".join(report_lines)
    print(report)
    (cache_path / "d6_v4_verdict.txt").write_text(report, encoding="utf-8")

    print()
    print(">>> NEXT STEPS:")
    print(f"  1. Review {cache_path / 'd6_v4_verdict.txt'}")
    print(f"  2. Verdict housekeeping per spec §10 amendment trigger")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/multivariate_msm_v4")
    args = parser.parse_args()
    sys.exit(main(args.cache_dir))
