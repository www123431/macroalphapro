"""
scripts/run_path_f_vix_term_structure.py — Path F CLI runner.

Pre-registration: docs/spec_path_f_vix_term_structure_v1.md (id=65) §八
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

logger = logging.getLogger("path_f.run_vix")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_f_vix_term_structure.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_vix_termstr_10y")
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
    from engine.path_f.vix_signal import (
        compute_signal, apply_risk_management,
        compute_strategy_returns, derive_trade_log,
    )
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
        GATE_1_SHARPE_THRESHOLD, GATE_1_NW_T_THRESHOLD,
        PathFVerdict,
    )

    SPEC_PATH = "docs/spec_path_f_vix_term_structure_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason)
        return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)

    # 1. Data fetch
    logger.info("Fetching VIX + VIX3M + SVXY...")
    panel = fetch_vix_panel(start, end)
    logger.info("Panel: %d daily obs", len(panel))
    if len(panel) < 252:
        logger.error("Insufficient data")
        return 3

    # 2. Compute SVXY daily returns
    svxy_returns = panel["SVXY"].pct_change().dropna()
    panel = panel.loc[svxy_returns.index]

    # 3. Signal
    target_pos = compute_signal(panel)
    target_pos = target_pos.loc[svxy_returns.index]
    logger.info("Signal: %d days target=1.0, %d days target=0.0",
                int((target_pos == 1.0).sum()), int((target_pos == 0.0).sum()))

    # 4. Risk management
    effective_pos, stop_loss_events = apply_risk_management(target_pos, svxy_returns)
    logger.info("Risk mgmt: %d stop-loss triggers, %d days effective=0",
                len(stop_loss_events), int((effective_pos == 0).sum()))

    # 5. Strategy returns
    strategy_returns = compute_strategy_returns(effective_pos, svxy_returns)
    logger.info("Strategy daily returns: mean %.6f, std %.4f",
                strategy_returns.mean(), strategy_returns.std())

    # 6. Trade log
    trades = derive_trade_log(effective_pos)
    n_trades = len(trades)
    logger.info("Trade log: %d trades", n_trades)

    # 7. Method A daily TS stats
    sh_A = annualized_sharpe(strategy_returns, 252)
    nw_A = newey_west_t(strategy_returns, lag=60)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, 252)
    ann_ret_A = float(strategy_returns.mean() * 252)
    ann_vol_A = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    # 8. Method B trade-time stats
    if n_trades > 0:
        # Compute per-trade return as cumulative return during holding
        trade_returns = []
        for _, t in trades.iterrows():
            entry = pd.Timestamp(t['entry_date'])
            exit  = pd.Timestamp(t['exit_date'])
            mask = (strategy_returns.index >= entry) & (strategy_returns.index < exit)
            seg = strategy_returns[mask]
            if len(seg) > 0:
                cum_ret = (1 + seg).prod() - 1
                trade_returns.append(cum_ret)
        mean_trade = float(np.mean(trade_returns)) if trade_returns else None
    else:
        mean_trade = None

    # 9. Sub-period
    sp_regime = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)

    # 10. OOS
    oos = oos_hold_out(strategy_returns)

    # 11. Incremental α vs K1
    k1_path = REPO_ROOT / "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet"
    if k1_path.exists():
        k1_paired = pd.read_parquet(k1_path)
        weekly_returns = k1_paired['k1_weekly_returns'].values
        n_weeks = len(weekly_returns)
        weekly_dates = pd.date_range(start='2014-01-06', periods=n_weeks, freq='W-MON')
        k1_daily = pd.Series(0.0, index=strategy_returns.index)
        for i, wd in enumerate(weekly_dates):
            if i >= n_weeks:
                break
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

    # 12. Gates
    gate_1 = bool(sh_A is not None and not np.isnan(sh_A) and sh_A >= GATE_1_SHARPE_THRESHOLD and
                  nw_A is not None and not np.isnan(nw_A) and nw_A >= GATE_1_NW_T_THRESHOLD)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)
    gate_5 = bool(ia.get("gate_5_pass"))

    # 13. Decision
    if not gate_1:
        decision = "FAIL"
    elif not (gate_3 and gate_4 and gate_5):
        decision = "INDIVIDUAL_PASS_BUT_NON_INDEPENDENT"
    else:
        decision = "PASS_INDEPENDENT"

    # 14. Cumulative + DD
    cum = (1 + strategy_returns).cumprod()
    cum_ret = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else 0.0
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    verdict = PathFVerdict(
        decision=decision,
        spec_hash=spec_hash,
        wave="F-vix-term-structure",
        universe_source="yfinance_svxy_single_ticker",
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_trades=n_trades,
        n_stop_loss_triggers=len(stop_loss_events),
        method_A_sharpe_net=float(sh_A) if not np.isnan(sh_A) else None,
        method_A_nw_t=float(nw_A) if not np.isnan(nw_A) else None,
        method_A_ci_lower=ci_lo if not np.isnan(ci_lo) else None,
        method_A_ci_upper=ci_hi if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret_A,
        method_A_ann_vol=ann_vol_A,
        method_B_n_trades=n_trades,
        method_B_mean_trade_return=mean_trade,
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
            "Cheng 2019 window 1990-2017; we test 2014-2023 = 5y overlap + 5y post-publication OOS",
            "SVXY restructure 2018-02-28 (-1x to -0.5x daily leverage); strategy adapts via daily signal",
            "2018-02-05 vol-mageddon SVXY -90% — stop-loss + 30-day cooling-off rule locked ex-ante per spec §2.3",
            "Single-ETF universe (SVXY) — simpler than basket; cleaner mechanism",
            "Daily rebalance signal at close d -> position at open d+1 (1-day execution lag)",
            "Threshold 0.95 contango chosen ex-ante from Cheng 2019 paper convention; not optimized",
        ],
    )

    # Persist
    out_dir = REPO_ROOT / "data/path_f"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")

    strategy_returns.to_frame("strategy_return").to_parquet(out_dir / f"{args.run_id}_daily.parquet")
    trades.to_parquet(out_dir / f"{args.run_id}_trades.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS:")
    logger.info("  Sharpe:     %.4f", sh_A if not np.isnan(sh_A) else 0)
    logger.info("  NW t:       %.4f", nw_A if not np.isnan(nw_A) else 0)
    logger.info("  CI 95%%:     [%.4f, %.4f]", ci_lo or 0, ci_hi or 0)
    logger.info("  Ann return: %.4f (= %.2f%%/yr)", ann_ret_A, ann_ret_A * 100)
    logger.info("  Ann vol:    %.4f", ann_vol_A)
    logger.info("Method B trade-time:")
    logger.info("  N trades:   %d", n_trades)
    logger.info("  Mean trade: %s", f"{mean_trade:.4f}" if mean_trade is not None else "n/a")
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual):       %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 2 (Selective BHY):    %s", verdict.gate_2_selective_bhy)
    logger.info("  Gate 3 (OOS hold-out):     %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):  %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):    %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative 10y:     %.4f", cum_ret)
    logger.info("Max DD:             %.4f", max_dd)
    logger.info("Stop-loss triggers: %d", len(stop_loss_events))
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
