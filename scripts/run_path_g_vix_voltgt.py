"""
scripts/run_path_g_vix_voltgt.py — Path G CLI runner.

Pre-registration: docs/spec_path_g_vix_voltgt_v1.md (id=66) §八
Reuses Path F infrastructure heavily.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np
from dataclasses import asdict

logger = logging.getLogger("path_g.run_voltgt")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_g_vix_voltgt.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_vix_voltgt_10y")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import validate_reference, _compute_git_blob_hash, _resolve_to_abs
    from engine.path_f.vix_data import fetch_vix_panel
    from engine.path_f.vix_signal import apply_risk_management
    from engine.path_g.vol_targeted_signal import (
        compute_vol_targeted_signal, compute_strategy_returns_voltgt,
        TARGET_VOL_ANNUAL_LOCKED,
    )
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
        GATE_1_SHARPE_THRESHOLD, GATE_1_NW_T_THRESHOLD,
        PathFVerdict,
    )

    SPEC_PATH = "docs/spec_path_g_vix_voltgt_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)

    # 1. Data fetch (reuse Path F)
    logger.info("Fetching VIX + VIX3M + SVXY...")
    panel = fetch_vix_panel(start, end)
    svxy_returns = panel["SVXY"].pct_change().dropna()
    panel = panel.loc[svxy_returns.index]
    logger.info("Panel: %d daily obs", len(panel))

    # 2. Vol-targeted signal (NEW vs Path F)
    target_pos = compute_vol_targeted_signal(panel, svxy_returns)
    target_pos = target_pos.loc[svxy_returns.index]
    n_zero = int((target_pos == 0).sum())
    n_active = int((target_pos > 0).sum())
    mean_pos = float(target_pos[target_pos > 0].mean())
    logger.info("Vol-targeted signal: %d days active (mean position %.3f), %d days cash",
                n_active, mean_pos, n_zero)

    # 3. Risk management (reuse Path F)
    effective_pos, stop_loss_events = apply_risk_management(target_pos, svxy_returns)
    logger.info("Risk mgmt: %d stop-loss triggers", len(stop_loss_events))

    # 4. Strategy returns (vol-targeted)
    strategy_returns = compute_strategy_returns_voltgt(effective_pos, svxy_returns)
    logger.info("Strategy daily returns: mean %.6f, std %.4f (vs target vol %.4f/√252)",
                strategy_returns.mean(), strategy_returns.std(),
                TARGET_VOL_ANNUAL_LOCKED / np.sqrt(252))

    # 5. Stats Method A
    sh_A = annualized_sharpe(strategy_returns, 252)
    nw_A = newey_west_t(strategy_returns, lag=60)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, 252)
    ann_ret_A = float(strategy_returns.mean() * 252)
    ann_vol_A = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    # 6. Sub-period
    sp_regime = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)

    # 7. OOS
    oos = oos_hold_out(strategy_returns)

    # 8. Incremental α vs K1
    k1_path = REPO_ROOT / "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet"
    if k1_path.exists():
        k1_paired = pd.read_parquet(k1_path)
        weekly_returns = k1_paired['k1_weekly_returns'].values
        n_weeks = len(weekly_returns)
        weekly_dates = pd.date_range(start='2014-01-06', periods=n_weeks, freq='W-MON')
        k1_daily = pd.Series(0.0, index=strategy_returns.index)
        for i, wd in enumerate(weekly_dates):
            if i >= n_weeks: break
            week_r = weekly_returns[i]
            daily_eq = (1 + week_r) ** (1/5) - 1
            for offset in range(5):
                target_d = wd + pd.Timedelta(days=offset)
                future = strategy_returns.index[strategy_returns.index >= target_d]
                if len(future) > 0:
                    k1_daily.loc[future[0]] = daily_eq
        ia = incremental_alpha_vs_baseline(strategy_returns, k1_daily)
    else:
        ia = {"gate_5_pass": False, "warning": "K1 baseline not found"}

    # 9. Gates (ETF sleeve thresholds via Path F module locks)
    gate_1 = bool(sh_A is not None and not np.isnan(sh_A) and sh_A >= GATE_1_SHARPE_THRESHOLD and
                  nw_A is not None and not np.isnan(nw_A) and nw_A >= GATE_1_NW_T_THRESHOLD)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)
    gate_5 = bool(ia.get("gate_5_pass"))

    if not gate_1:
        decision = "FAIL"
    elif not (gate_3 and gate_4 and gate_5):
        decision = "INDIVIDUAL_PASS_BUT_NON_INDEPENDENT"
    else:
        decision = "PASS_INDEPENDENT"

    cum = (1 + strategy_returns).cumprod()
    cum_ret = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else 0.0
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    verdict = PathFVerdict(
        decision=decision,
        spec_hash=spec_hash,
        wave="G-vix-voltgt",
        universe_source="yfinance_svxy_voltgt_12pct",
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_trades=0,  # vol-targeted: continuous position, not discrete trades
        n_stop_loss_triggers=len(stop_loss_events),
        method_A_sharpe_net=float(sh_A) if not np.isnan(sh_A) else None,
        method_A_nw_t=float(nw_A) if not np.isnan(nw_A) else None,
        method_A_ci_lower=ci_lo if not np.isnan(ci_lo) else None,
        method_A_ci_upper=ci_hi if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret_A,
        method_A_ann_vol=ann_vol_A,
        method_B_n_trades=0,
        method_B_mean_trade_return=None,
        subperiod_regime=sp_regime,
        subperiod_random_rolling=sp_rolling,
        oos_hold_out=oos,
        incremental_alpha_vs_K1=ia,
        gate_1_individual_pass=gate_1,
        gate_2_selective_bhy="DEMOTED_SINGLE_TEST",
        gate_3_oos_pass=gate_3,
        gate_4_subperiod_pass=gate_4,
        gate_5_incremental_pass=gate_5,
        cumulative_return=cum_ret,
        max_drawdown=max_dd,
        stop_loss_events=[{k: str(v) if hasattr(v, 'isoformat') else v for k, v in e.items()} for e in stop_loss_events],
        honest_disclose=[
            "Vol-target 12% ex-ante locked from Moskowitz 2012 canonical; NOT data-fitted from Path F vol 39.7%",
            "21d realized vol lookback; not optimized",
            "Vol-scale capped at 1.0 (no leverage)",
            "Same Path F caveats: SVXY 2018 restructure, post-publication Cheng 2019 overlap, daily approximation",
            "Distinct hypothesis from Path F (binary position): structurally different position-sizing rule",
        ],
    )

    out_dir = REPO_ROOT / "data/path_g"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
    strategy_returns.to_frame("strategy_return").to_parquet(out_dir / f"{args.run_id}_daily.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS:")
    logger.info("  Sharpe:     %.4f", sh_A if not np.isnan(sh_A) else 0)
    logger.info("  NW t:       %.4f", nw_A if not np.isnan(nw_A) else 0)
    logger.info("  CI 95%%:     [%.4f, %.4f]", ci_lo or 0, ci_hi or 0)
    logger.info("  Ann return: %.4f (= %.2f%%/yr)", ann_ret_A, ann_ret_A * 100)
    logger.info("  Ann vol:    %.4f (target %.4f)", ann_vol_A, TARGET_VOL_ANNUAL_LOCKED)
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual):       %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 2 (Selective BHY):    %s", verdict.gate_2_selective_bhy)
    logger.info("  Gate 3 (OOS hold-out):     %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):  %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):    %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative 10y: %.4f", cum_ret)
    logger.info("Max DD:         %.4f", max_dd)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
