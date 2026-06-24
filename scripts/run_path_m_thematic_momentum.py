"""
scripts/run_path_m_thematic_momentum.py — Path M Thematic ETF Momentum CLI runner.

Pre-registration: docs/spec_path_m_thematic_momentum_v1.md (id=69 hash a3f50c9f) §八.

Pipeline:
  1. Spec hash validation
  2. Fetch 34 locked thematic ETFs from yfinance (cached)
  3. Compute 12-1 monthly momentum panel
  4. Form top-3/bot-3 cohorts at each month-end
  5. Compute daily L-S returns + TC drag (5bp roundtrip)
  6. Stats + 5-gate ETF sleeve verdict
  7. Dual-baseline incremental α (K1 + D-PEAD)
  8. Long-only top-3 secondary disclosure
  9. Persist verdict + daily parquet
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("path_m.run_thematic_mom")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_m_thematic_momentum.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_thematic_momentum_10y")
    p.add_argument("--smoke",  action="store_true")
    p.add_argument("--verbose","-v", action="store_true")
    return p.parse_args()


@dataclass
class PathMVerdict:
    decision:                    str
    spec_hash:                   str
    wave:                        str = "M-thematic-mom"
    universe_source:             str = "thematic_etf_34_locked"
    window_start:                str = ""
    window_end:                  str = ""
    n_daily_obs:                 int = 0
    n_rebalances:                int = 0
    n_tickers_locked:            int = 34
    universe_coverage_mean_pct:  float = 0.0
    mean_long_leg_size:          float = 0.0
    mean_short_leg_size:         float = 0.0
    annual_turnover_one_way_pct: float = 0.0
    tc_drag_annual_pct:          float = 0.0
    method_A_sharpe_gross:       Optional[float] = None
    method_A_sharpe_net:         Optional[float] = None
    method_A_nw_t:               Optional[float] = None
    method_A_ci_lower:           Optional[float] = None
    method_A_ci_upper:           Optional[float] = None
    method_A_ann_return:         float = 0.0
    method_A_ann_vol:            float = 0.0
    subperiod_regime:            dict = field(default_factory=dict)
    subperiod_random_rolling:    dict = field(default_factory=dict)
    oos_hold_out:                dict = field(default_factory=dict)
    incremental_alpha_vs_K1:     dict = field(default_factory=dict)
    incremental_alpha_vs_DPEAD:  dict = field(default_factory=dict)
    gate_1_individual_pass:      bool = False
    gate_2_selective_bhy:        str = "DEMOTED_SINGLE_TEST"
    gate_3_oos_pass:             bool = False
    gate_4_subperiod_pass:       bool = False
    gate_5_incremental_pass:     bool = False
    cumulative_return:           float = 0.0
    max_drawdown:                float = 0.0
    long_only_top3_sharpe:       Optional[float] = None
    long_only_top3_nw_t:         Optional[float] = None
    honest_disclose:             list = field(default_factory=list)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import validate_reference, _compute_git_blob_hash, _resolve_to_abs
    from engine.path_m.thematic_momentum_strategy import (
        LOCKED_UNIVERSE_LIST, N_UNIVERSE_LOCKED, NW_LAG_LOCKED,
        compute_monthly_momentum, form_long_short_cohorts, compute_strategy_returns,
    )
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
        GATE_1_SHARPE_THRESHOLD, GATE_1_NW_T_THRESHOLD,
    )

    SPEC_PATH = "docs/spec_path_m_thematic_momentum_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    if args.smoke:
        end_date = datetime.date(start_date.year + 1, 12, 31)
        logger.info("SMOKE MODE — %s → %s", start_date, end_date)

    # ── Step 1: Fetch 34 locked thematic ETFs ─────────────────────────────────
    logger.info("Fetching %d locked thematic ETFs from yfinance...", N_UNIVERSE_LOCKED)
    fetch_start = (start_date - datetime.timedelta(days=400)).isoformat()  # 1y warm-up
    data = yf.download(LOCKED_UNIVERSE_LIST, start=fetch_start, end=end_date.isoformat(),
                       progress=False, auto_adjust=True)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.dropna(how='all').ffill()
    logger.info("Prices panel: %d dates × %d tickers", *close.shape)

    # Validate universe coverage
    coverage_per_ticker = {}
    for t in LOCKED_UNIVERSE_LIST:
        if t in close.columns:
            n = close[t].loc[start_date.isoformat():end_date.isoformat()].dropna().shape[0]
            coverage_per_ticker[t] = n
    logger.info("Universe coverage: %d/%d tickers with data",
                len(coverage_per_ticker), N_UNIVERSE_LOCKED)

    # Trim to spec window
    close_in = close.loc[start_date.isoformat():end_date.isoformat()]

    # ── Step 2: Compute 12-1 monthly momentum ─────────────────────────────────
    logger.info("Computing monthly momentum (12-1 canonical)...")
    momentum_panel = compute_monthly_momentum(close_in)
    n_valid_months = momentum_panel.dropna(how='all').shape[0]
    logger.info("Momentum panel: %d month-ends, %d with at least one valid signal",
                momentum_panel.shape[0], n_valid_months)

    # ── Step 3: Form L-S cohorts ──────────────────────────────────────────────
    long_cohorts, short_cohorts = form_long_short_cohorts(momentum_panel)
    logger.info("Cohorts formed: %d rebalance months", len(long_cohorts))

    # Universe coverage stats: mean fraction of locked universe with valid signal at each rebalance
    coverage_pcts = []
    for rd in long_cohorts:
        n_valid = momentum_panel.loc[rd].notna().sum()
        coverage_pcts.append(n_valid / N_UNIVERSE_LOCKED * 100)
    universe_cov_mean = float(np.mean(coverage_pcts)) if coverage_pcts else 0.0
    logger.info("Mean universe coverage: %.1f%% (%d/%d)",
                universe_cov_mean, int(universe_cov_mean * N_UNIVERSE_LOCKED / 100), N_UNIVERSE_LOCKED)

    # ── Step 4: Compute strategy returns ──────────────────────────────────────
    strat = compute_strategy_returns(close_in, long_cohorts, short_cohorts)
    logger.info("Strategy: %d rebalances, mean L=%.1f / S=%.1f, annual turnover %.1f%%/one-way, TC drag %.2f%%/yr",
                strat.n_rebalances, strat.mean_long_size, strat.mean_short_size,
                strat.annual_turnover_one_way_pct, strat.tc_drag_annual_pct)

    strategy_returns = strat.daily_returns
    gross_returns    = strat.daily_gross

    # ── Step 5: Stats ─────────────────────────────────────────────────────────
    sh_gross = annualized_sharpe(gross_returns)
    sh_net   = annualized_sharpe(strategy_returns)
    nw_t     = newey_west_t(strategy_returns, lag=NW_LAG_LOCKED)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, n_resamples=1000)
    ann_ret  = float(strategy_returns.mean() * 252)
    ann_vol  = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    sp_regime  = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)
    oos = oos_hold_out(strategy_returns)

    # ── Step 6: Incremental α vs K1 + D-PEAD ──────────────────────────────────
    incremental_K1 = {"gate_5_pass": False, "warning": "K1 baseline not loaded"}
    incremental_DPEAD = {"gate_5_pass": False, "warning": "D-PEAD baseline not loaded"}

    k1_path = REPO_ROOT / "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet"
    if k1_path.exists():
        k1_paired = pd.read_parquet(k1_path)
        weekly = k1_paired['k1_weekly_returns'].values
        weekly_dates = pd.date_range(start='2014-01-06', periods=len(weekly), freq='W-MON')
        k1_daily = pd.Series(0.0, index=strategy_returns.index)
        for i, wd in enumerate(weekly_dates):
            if i >= len(weekly): break
            daily_eq = (1 + weekly[i]) ** (1/5) - 1
            for offset in range(5):
                target_d = wd + pd.Timedelta(days=offset)
                future = strategy_returns.index[strategy_returns.index >= target_d]
                if len(future) > 0:
                    k1_daily.loc[future[0]] = daily_eq
        incremental_K1 = incremental_alpha_vs_baseline(strategy_returns, k1_daily)

    dpead_path = REPO_ROOT / "data/path_c_dhs/walk_forward_pead.parquet"
    if dpead_path.exists():
        dpead_df = pd.read_parquet(dpead_path)
        dpead_daily = dpead_df['r_long_short_net'].reindex(strategy_returns.index).fillna(0.0)
        incremental_DPEAD = incremental_alpha_vs_baseline(strategy_returns, dpead_daily)

    ir_K1    = abs(incremental_K1.get('information_ratio', 0.0) or 0.0)
    ir_DPEAD = abs(incremental_DPEAD.get('information_ratio', 0.0) or 0.0)
    stronger = "DPEAD" if ir_DPEAD >= ir_K1 else "K1"
    gate_5 = bool(
        incremental_DPEAD.get('gate_5_pass') if stronger == "DPEAD"
        else incremental_K1.get('gate_5_pass')
    )

    # ── Step 7: Gates ─────────────────────────────────────────────────────────
    gate_1 = bool(sh_net is not None and not np.isnan(sh_net) and sh_net >= GATE_1_SHARPE_THRESHOLD
                  and nw_t is not None and not np.isnan(nw_t) and nw_t >= GATE_1_NW_T_THRESHOLD)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)

    if not gate_1:
        if sh_net is not None and not np.isnan(sh_net) and sh_net >= 0.25 and nw_t >= 1.3:
            decision = "MARGINAL"
        else:
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

    # ── Step 8: Secondary — long-only top-3 disclosure ───────────────────────
    long_only = pd.Series(0.0, index=strategy_returns.index)
    daily_ret_panel = close_in.pct_change()
    rebal_dates = sorted(long_cohorts.keys())
    for i, rd in enumerate(rebal_dates):
        if i + 1 >= len(rebal_dates): break
        next_rd = rebal_dates[i + 1]
        longs = long_cohorts[rd]
        hold_mask = (daily_ret_panel.index > rd) & (daily_ret_panel.index <= next_rd)
        for d in daily_ret_panel.index[hold_mask]:
            r = daily_ret_panel.loc[d, longs].dropna().mean()
            if not np.isnan(r):
                long_only.loc[d] = r
    lo_sh = annualized_sharpe(long_only)
    lo_nw = newey_west_t(long_only, lag=NW_LAG_LOCKED)

    verdict = PathMVerdict(
        decision=decision, spec_hash=spec_hash,
        window_start=start_date.isoformat(), window_end=end_date.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_rebalances=int(strat.n_rebalances),
        universe_coverage_mean_pct=universe_cov_mean,
        mean_long_leg_size=float(strat.mean_long_size),
        mean_short_leg_size=float(strat.mean_short_size),
        annual_turnover_one_way_pct=float(strat.annual_turnover_one_way_pct),
        tc_drag_annual_pct=float(strat.tc_drag_annual_pct),
        method_A_sharpe_gross=float(sh_gross) if not np.isnan(sh_gross) else None,
        method_A_sharpe_net=float(sh_net) if not np.isnan(sh_net) else None,
        method_A_nw_t=float(nw_t) if not np.isnan(nw_t) else None,
        method_A_ci_lower=float(ci_lo) if not np.isnan(ci_lo) else None,
        method_A_ci_upper=float(ci_hi) if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret, method_A_ann_vol=ann_vol,
        subperiod_regime=sp_regime, subperiod_random_rolling=sp_rolling, oos_hold_out=oos,
        incremental_alpha_vs_K1=incremental_K1,
        incremental_alpha_vs_DPEAD=incremental_DPEAD,
        gate_1_individual_pass=gate_1,
        gate_3_oos_pass=gate_3, gate_4_subperiod_pass=gate_4,
        gate_5_incremental_pass=gate_5,
        cumulative_return=cum_ret, max_drawdown=max_dd,
        long_only_top3_sharpe=float(lo_sh) if not np.isnan(lo_sh) else None,
        long_only_top3_nw_t=float(lo_nw) if not np.isnan(lo_nw) else None,
        honest_disclose=[
            "34 thematic ETF universe is JUDGMENT-DEFINED despite rule-based criteria (≥1500 day history)",
            "Robustness scout 2026-05-13: ARK-out Sharpe 0.67 ≈ baseline 0.68; signal NOT ARK-driven",
            "Robustness scout: leg-size top-2..7 all PASS gate; 12-1 lookback robust 9-12mo",
            "TC drag 5bp roundtrip locked (Tier 3 thematic standing rule); real TC may be 3-8bp",
            "Capacity ~$30-100M sleeve; NOT institutional-scale $1B+",
            "Jegadeesh-Titman 1993 = 33y post-pub; capacity protection rationale for survival",
            "Long-only top-3 variant disclosed as secondary (Sharpe in verdict)",
            "Spec hash a3f50c9f locked BEFORE backtest; pre-registration verified",
        ],
    )

    out_dir = REPO_ROOT / "data/path_m"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
    strategy_returns.to_frame("strategy_return").to_parquet(out_dir / f"{args.run_id}_daily.parquet")
    gross_returns.to_frame("gross_return").to_parquet(out_dir / f"{args.run_id}_gross_daily.parquet")
    long_only.to_frame("long_only_return").to_parquet(out_dir / f"{args.run_id}_longonly_daily.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS (ETF sleeve gates 0.4 / 1.8):")
    logger.info("  Sharpe gross: %.4f", sh_gross if not np.isnan(sh_gross) else 0)
    logger.info("  Sharpe net:   %.4f", sh_net if not np.isnan(sh_net) else 0)
    logger.info("  NW t (60):    %.4f", nw_t if not np.isnan(nw_t) else 0)
    logger.info("  CI 95%%:       [%.4f, %.4f]", ci_lo or 0, ci_hi or 0)
    logger.info("  Ann return:   %.4f", ann_ret)
    logger.info("  Ann vol:      %.4f", ann_vol)
    logger.info("Sub-period:")
    for k in ['pre_covid', 'covid', 'post_covid']:
        if k in sp_regime:
            logger.info("  %-12s n=%d  Sharpe=%.4f", k,
                        sp_regime[k].get('n_obs', 0), sp_regime[k].get('sharpe', 0) or 0)
    logger.info("Incremental α vs K1:    IR=%.4f, gate=%s",
                incremental_K1.get('information_ratio', 0) or 0,
                incremental_K1.get('gate_5_pass', False))
    logger.info("Incremental α vs DPEAD: IR=%.4f, gate=%s",
                incremental_DPEAD.get('information_ratio', 0) or 0,
                incremental_DPEAD.get('gate_5_pass', False))
    logger.info("Long-only top-3 (secondary): Sharpe %.4f NW t %.4f",
                lo_sh if not np.isnan(lo_sh) else 0, lo_nw if not np.isnan(lo_nw) else 0)
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual 0.4/1.8): %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 3 (OOS hold-out):       %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):    %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):      %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative: %.4f, Max DD: %.4f", cum_ret, max_dd)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
