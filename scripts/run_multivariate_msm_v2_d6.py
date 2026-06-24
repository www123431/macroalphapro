"""scripts/run_multivariate_msm_v2_d6.py — Multivariate MSM v2 verdict run.

Spec: docs/spec_multivariate_msm_v2.md (registered post-2026-05-08; supersedes v1
which was withdrawn for D1-D5 architectural defects, NOT falsification).

v2 fixes:
  D1 — VIX-anchored regime ID (replaces argmax-yield_spread; stable across walk-forward)
  D2 — K=2 lock + explicit misspec disclosure (BIC-k=3 acknowledged but K=2 kept for param-count)
  D3 — Ternary overlay [0.45, 0.55] hysteresis (replaces binary 2p-1)
  D4 — PRE-VERDICT proxy validation (Pearson r between FRED OAS-diff and ETF return-spread on
       2023-08+ overlap; r < 0.5 → spec INVALID, abort)
  D5 — Descriptive-only verdict labels (no PASS/FAIL gate; supervisor judgment)

Usage (must run locally; FRED API + yfinance network needed):
    python scripts/run_multivariate_msm_v2_d6.py [--cache-dir data/multivariate_msm_v2/]

Output:
    data/multivariate_msm_v2/proxy_validation.txt        — D4 §3.6 result (gate)
    data/multivariate_msm_v2/walk_forward_probs_v2.parquet
    data/multivariate_msm_v2/spy_monthly.parquet
    data/multivariate_msm_v2/d6_v2_verdict.txt           — full verdict report

Decision labels (replacing v1 PASS/FAIL):
    DESCRIPTIVE_POSITIVE       ΔŜ ≥ +0.05 AND CI lower > 0      → supervisor may PendingApproval
    DESCRIPTIVE_INSUFFICIENT   non-negative but below threshold  → no ship
    DESCRIPTIVE_NEGATIVE       ΔŜ < 0                            → 9th falsification (sound design)
    UNINTERPRETABLE            fallback rate ≥ 50%               → spec §3.4
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

from engine.regime import _validate_proxy_against_fred  # noqa: E402
from engine.multivariate_msm_verdict import (  # noqa: E402
    compute_overlay_returns_ternary,
    compute_verdict_v2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Spec_v2 windows (same as v1 §3.5; HARKing protection only)
IN_SAMPLE_START    = datetime.date(2010, 1, 1)
IN_SAMPLE_END      = datetime.date(2018, 12, 31)
OOS_START          = datetime.date(2019, 1, 1)
OOS_END            = datetime.date(2024, 12, 31)
WALK_FORWARD_START = IN_SAMPLE_START
WALK_FORWARD_END   = OOS_END


def _step_1_proxy_validation(cache_path: pathlib.Path) -> tuple[float, str, dict]:
    """Spec_v2 §3.6 D4 fix: validate ETF proxy against FRED OAS on overlap window."""
    logger.info("=" * 70)
    logger.info("STEP 1: Pre-verdict proxy validation (spec_v2 §3.6 / D4 fix)")
    logger.info("=" * 70)
    r, status, diag = _validate_proxy_against_fred()
    report = [
        f"Pearson r (FRED Δ(OAS_diff) vs ETF (LQD-HYG) return-spread): {r:.4f}",
        f"Status: {status}",
        f"n overlapping months: {diag.get('n_obs', 'NA')}",
        f"FRED first valid: {diag.get('fred_first', 'NA')}",
        f"FRED last valid:  {diag.get('fred_last', 'NA')}",
        f"Method: {diag.get('method', 'NA')}",
        "Thresholds:",
    ]
    thr = diag.get("thresholds", {})
    for k, v in thr.items():
        report.append(f"  {k}: {v}")
    txt = "\n".join(report)
    logger.info("\n%s", txt)

    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "proxy_validation.txt").write_text(txt, encoding="utf-8")
    return r, status, diag


def _walk_forward_probs_v2(rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """v2 walk-forward: at each as_of, run multivariate v2 + univariate baseline."""
    from engine.regime import (
        _get_regime_multivariate_v2, get_regime_on,
        ConvergenceError, InsufficientData, MissingFeatureData,
    )
    rows = []
    for ts in rebalance_dates:
        d = ts.date()
        p_multi, multi_failed = float("nan"), False
        try:
            r_multi = _get_regime_multivariate_v2(as_of=d, train_end=d)
            p_multi = float(r_multi.p_risk_on)
        except (ConvergenceError, InsufficientData, MissingFeatureData) as exc:
            multi_failed = True
            logger.info("multivariate v2 fallback at %s: %s: %s", d, type(exc).__name__, exc)
        except Exception as exc:
            multi_failed = True
            logger.warning("multivariate v2 UNEXPECTED at %s: %s", d, exc)

        try:
            r_uni = get_regime_on(as_of=d, train_end=d, use_multivariate=False)
            p_uni = float(r_uni.p_risk_on)
        except Exception as exc:
            logger.warning("univariate failure at %s: %s", d, exc)
            p_uni = float("nan")

        if multi_failed and not np.isnan(p_uni):
            p_multi = p_uni  # fallback per spec §4.2

        rows.append({
            "date":                ts,
            "p_multivariate_v2":   p_multi,
            "p_univariate":        p_uni,
            "multivariate_failed": multi_failed,
        })
        logger.info("walk-forward v2 %s: p_multi=%.3f p_uni=%.3f failed=%s",
                    d, p_multi, p_uni, multi_failed)
    return pd.DataFrame(rows).set_index("date")


def _fetch_spy_monthly_returns(start: datetime.date, end: datetime.date) -> pd.Series:
    from engine.signal import _fetch_closes
    closes = _fetch_closes(["SPY"], start - datetime.timedelta(days=10), end)
    if closes.empty or "SPY" not in closes.columns:
        return pd.Series(dtype=float)
    spy = closes["SPY"].resample("ME").last().dropna()
    return spy.pct_change().dropna()


def main(cache_dir: str = "data/multivariate_msm_v2") -> int:
    cache_path = pathlib.Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Proxy validation (spec_v2 §3.6 / D4 fix) ────────────────────
    r, status, _diag = _step_1_proxy_validation(cache_path)
    if status.startswith("INVALID"):
        logger.error("=" * 70)
        logger.error("PROXY VALIDATION FAILED: status=%s (r=%s)", status, r)
        logger.error("Per spec_v2 §3.6, v2 verdict run aborted.")
        logger.error("=" * 70)
        return 2

    logger.info("Proxy validation: status=%s r=%.4f → proceeding to walk-forward", status, r)

    # ── STEP 2: Walk-forward ────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 2: Walk-forward probabilities (v2 VIX-anchored regime ID)")
    logger.info("=" * 70)
    rebalance = pd.date_range(WALK_FORWARD_START, WALK_FORWARD_END, freq="BME")
    probs_cache = cache_path / "walk_forward_probs_v2.parquet"
    if probs_cache.exists():
        logger.info("Using cached walk-forward probs from %s", probs_cache)
        probs = pd.read_parquet(probs_cache)
        probs.index = pd.to_datetime(probs.index)
    else:
        probs = _walk_forward_probs_v2(rebalance)
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

    # ── STEP 3: v2 ternary overlay returns + verdict ─────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 3: Ternary overlay returns + descriptive verdict (v2)")
    logger.info("=" * 70)
    overlay_multi = compute_overlay_returns_ternary(probs["p_multivariate_v2"], spy)
    overlay_uni   = compute_overlay_returns_ternary(probs["p_univariate"],      spy)

    oos_start_ts = pd.Timestamp(OOS_START)
    oos_end_ts   = pd.Timestamp(OOS_END)
    overlay_multi_oos = overlay_multi.loc[(overlay_multi.index >= oos_start_ts) & (overlay_multi.index <= oos_end_ts)]
    overlay_uni_oos   = overlay_uni.loc[  (overlay_uni.index   >= oos_start_ts) & (overlay_uni.index   <= oos_end_ts)]

    probs_oos     = probs.loc[(probs.index >= oos_start_ts) & (probs.index <= oos_end_ts)]
    fallback_rate = float(probs_oos["multivariate_failed"].mean()) if len(probs_oos) > 0 else 0.0

    verdict = compute_verdict_v2(
        overlay_multi_oos, overlay_uni_oos,
        fallback_rate=fallback_rate,
        n_resamples=1000,
    )

    report_lines = [
        "=" * 70,
        "Multivariate MSM v2 — OOS Verdict (W1 D6 v2)",
        f"Spec: docs/spec_multivariate_msm_v2.md §3.1 + §3.2 + §3.6",
        f"Run date: {datetime.date.today().isoformat()}",
        f"In-sample (HARKing protection): {IN_SAMPLE_START} to {IN_SAMPLE_END}",
        f"OOS window:                     {OOS_START} to {OOS_END}",
        f"OOS months captured:            {verdict.n_oos_months}",
        "=" * 70,
        "",
        f"§3.6 Proxy validation: status={status}, Pearson r={r:.4f}",
        "",
        "Effect-size estimates (DESCRIPTIVE only per spec_v2 §3.2 D5 framework):",
        f"  Sharpe(multivariate v2 overlay) = {verdict.sharpe_multivariate:+.3f}",
        f"  Sharpe(univariate overlay)      = {verdict.sharpe_univariate:+.3f}",
        f"  ΔŜ                              = {verdict.delta_sharpe:+.3f}",
        "",
        "Bootstrap (Politis-Romano stationary + Politis-White auto-block):",
        f"  95% CI for ΔŜ                   = [{verdict.bootstrap_ci_lower:+.3f}, {verdict.bootstrap_ci_upper:+.3f}]",
        f"  CI lower > 0?                   = {verdict.ci_lower_above_zero}",
        f"  CI lower ≥ +0.05?               = {verdict.ci_lower_above_threshold}  (ship-suggesting heuristic)",
        f"  Block size                      = {verdict.bootstrap_block_size}",
        "",
        "Memmel Z (descriptive secondary, spec §3.1):",
        f"  Z                               = {verdict.memmel_z:+.3f}",
        f"  Paired ρ̂                        = {verdict.paired_correlation:+.3f}",
        f"  Achieved power (descriptive)    = {verdict.achieved_power_descriptive:.1%}",
        "",
        "Fallback diagnostic (spec §3.4):",
        f"  Multivariate v2 fallback rate   = {verdict.fallback_rate:.1%}",
        f"  Tier:                           {'UNINTERPRETABLE' if verdict.fallback_rate >= 0.50 else 'STRONG_CAVEAT' if verdict.fallback_rate >= 0.25 else 'SOFT_CAVEAT' if verdict.fallback_rate >= 0.10 else 'NORMAL'}",
        "",
        f"DECISION LABEL (descriptive): {verdict.decision_label}",
        "",
        "Spec_v2 §3.2 supervisor framework:",
        "  DESCRIPTIVE_POSITIVE        → supervisor may PendingApproval(production_signal_swap)",
        "  DESCRIPTIVE_INSUFFICIENT    → no ship; descriptive disclosure only",
        "  DESCRIPTIVE_NEGATIVE        → 9th falsification chain entry; v2 superseded",
        "  UNINTERPRETABLE             → spec §3.4 fallback ≥ 50%",
        "=" * 70,
    ]
    report = "\n".join(report_lines)
    print(report)
    (cache_path / "d6_v2_verdict.txt").write_text(report, encoding="utf-8")

    print()
    print(">>> NEXT STEPS:")
    print(f"  1. Review {cache_path / 'd6_v2_verdict.txt'}")
    print(f"  2. Per decision label, paste verdict report back to assistant for housekeeping")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="data/multivariate_msm_v2",
                        help="Directory for caches + verdict report")
    args = parser.parse_args()
    sys.exit(main(args.cache_dir))
