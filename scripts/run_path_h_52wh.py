"""
scripts/run_path_h_52wh.py — Path H 52-Week-High Momentum CLI runner.

Pre-registration: docs/spec_path_h_52wh_v1.md (id=67 hash 7ecbaa3e) §八.

Pipeline:
  1. Spec hash validation
  2. Build per-month-end top-1500 universe via CRSP market-cap rank
  3. Bulk-fetch CRSP daily prices (start - 1y warm-up through end)
  4. Compute nearness panel + daily returns
  5. Form monthly long/short decile cohorts
  6. Compute strategy daily L-S returns with TC drag
  7. Stats: Sharpe / NW t / bootstrap CI / sub-periods / OOS / dual-baseline IR
  8. 5-gate single-stock decision
  9. Persist verdict.json + daily.parquet
"""
from __future__ import annotations

import argparse
import calendar
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

logger = logging.getLogger("path_h.run_52wh")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_h_52wh.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_52wh_10y")
    p.add_argument("--smoke",  action="store_true", help="Run 2014 single year smoke test")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _last_trading_day_of_month(year: int, month: int) -> datetime.date:
    """Return last calendar day of month (CRSP will snap to actual trading day inside)."""
    last_day = calendar.monthrange(year, month)[1]
    return datetime.date(year, month, last_day)


def _month_end_dates(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Return list of last-calendar-day-of-month within [start, end]."""
    dates = []
    y, m = start.year, start.month
    while True:
        d = _last_trading_day_of_month(y, m)
        if d > end:
            break
        if d >= start:
            dates.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates


def _build_universe_panel(
    month_ends: list[datetime.date],
    rank_max: int = 1500,
) -> tuple[dict, set]:
    """Query top-rank_max universe per month-end.

    Returns:
        (universe_at_month_end: dict[pd.Timestamp -> set[str]],
         all_unique_tickers: set[str])
    """
    from engine.universe_singlename.constituents_loader import load_russell2000_proxy_at_date

    universe = {}
    all_tickers: set = set()
    for i, m_end in enumerate(month_ends):
        try:
            result = load_russell2000_proxy_at_date(m_end, rank_min=1, rank_max=rank_max)
            tickers = set(result.tickers)
            if len(tickers) < 100:
                logger.warning("Month %s: only %d tickers — skip", m_end, len(tickers))
                continue
            universe[pd.Timestamp(m_end)] = tickers
            all_tickers.update(tickers)
            if (i + 1) % 12 == 0:
                logger.info("Universe built: %d/%d months (cumulative %d unique tickers)",
                            i + 1, len(month_ends), len(all_tickers))
        except Exception as exc:
            logger.error("Universe fetch failed for %s: %s", m_end, exc)
            raise
    return universe, all_tickers


@dataclass
class PathHVerdict:
    decision:                 str
    spec_hash:                str
    wave:                     str = "H-52wh"
    universe_source:          str = "crsp_top1500_mktcap_rank"
    window_start:             str = ""
    window_end:               str = ""
    n_daily_obs:              int = 0
    n_cohorts:                int = 0
    mean_long_leg_size:       float = 0.0
    mean_short_leg_size:      float = 0.0
    annual_turnover_one_way_pct: float = 0.0
    tc_drag_annual_pct:       float = 0.0
    method_A_sharpe_gross:    Optional[float] = None
    method_A_sharpe_net:      Optional[float] = None
    method_A_nw_t:            Optional[float] = None
    method_A_ci_lower:        Optional[float] = None
    method_A_ci_upper:        Optional[float] = None
    method_A_ann_return:      float = 0.0
    method_A_ann_vol:         float = 0.0
    subperiod_regime:         dict = field(default_factory=dict)
    subperiod_random_rolling: dict = field(default_factory=dict)
    oos_hold_out:             dict = field(default_factory=dict)
    incremental_alpha_vs_K1:    dict = field(default_factory=dict)
    incremental_alpha_vs_DPEAD: dict = field(default_factory=dict)
    gate_1_individual_pass:   bool = False
    gate_2_selective_bhy:     str = "DEMOTED_SINGLE_TEST"
    gate_3_oos_pass:          bool = False
    gate_4_subperiod_pass:    bool = False
    gate_5_incremental_pass:  bool = False
    cumulative_return:        float = 0.0
    max_drawdown:             float = 0.0
    honest_disclose:          list = field(default_factory=list)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import validate_reference, _compute_git_blob_hash, _resolve_to_abs
    from engine.path_h.nearness_strategy import (
        compute_nearness_panel, form_monthly_cohorts, compute_strategy_returns,
        LOOKBACK_DAYS_LOCKED, NW_LAG_LOCKED, UNIVERSE_RANK_MAX_LOCKED,
    )
    from engine.universe_singlename.crsp_loader import bulk_fetch_crsp_daily_panel
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
    )

    # Single-stock sleeve gates per standing rule
    GATE_1_SHARPE_THRESHOLD = 0.50
    GATE_1_NW_T_THRESHOLD   = 2.00

    SPEC_PATH = "docs/spec_path_h_52wh_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)

    if args.smoke:
        end = datetime.date(start.year, 12, 31)
        logger.info("SMOKE MODE — running %s → %s only", start, end)

    # 1. Build per-month-end universe
    month_ends = _month_end_dates(start, end)
    logger.info("Building universe across %d month-ends...", len(month_ends))
    universe, all_tickers = _build_universe_panel(month_ends, rank_max=UNIVERSE_RANK_MAX_LOCKED)
    logger.info("Universe: %d months, %d unique tickers", len(universe), len(all_tickers))

    if len(universe) < (len(month_ends) * 0.8):
        logger.error("Universe coverage too low (<80%%): %d / %d months", len(universe), len(month_ends))
        return 3

    # 2. Bulk-fetch CRSP daily prices (warm-up 1 year before start for 52WH)
    warmup_start = start - datetime.timedelta(days=400)  # extra buffer for non-trading days
    logger.info("Fetching CRSP daily panel for %d tickers, [%s, %s]...",
                len(all_tickers), warmup_start, end)
    prices_panel = bulk_fetch_crsp_daily_panel(
        sorted(all_tickers), warmup_start, end, use_cache=True,
    )
    logger.info("Prices panel: %d dates × %d tickers", prices_panel.shape[0], prices_panel.shape[1])

    # 3. Daily returns
    daily_returns_panel = prices_panel.pct_change()
    # Trim to in-window for backtest (returns indexed by date)
    in_window_mask = (daily_returns_panel.index >= pd.Timestamp(start)) & \
                     (daily_returns_panel.index <= pd.Timestamp(end))
    daily_returns_in = daily_returns_panel.loc[in_window_mask]

    # 4. Nearness panel (uses pre-window data via rolling 252d)
    nearness = compute_nearness_panel(prices_panel, lookback=LOOKBACK_DAYS_LOCKED)
    logger.info("Nearness panel: %d dates × %d tickers (with values)",
                nearness.shape[0], (~nearness.isna()).any(axis=0).sum())

    # 5. Form cohorts
    cohorts = form_monthly_cohorts(nearness, universe)
    logger.info("Cohorts formed: %d / %d months", len(cohorts), len(universe))

    if len(cohorts) < 10:
        logger.error("Too few cohorts to backtest: %d", len(cohorts))
        return 4

    # 6. Strategy returns
    logger.info("Computing strategy returns over %d trading days...", len(daily_returns_in))
    strat = compute_strategy_returns(cohorts, daily_returns_in)
    logger.info("Strategy: %d cohorts; mean long %d / short %d; annual turnover %.1f%%/one-way; TC drag %.2f%%/yr",
                strat.n_cohorts, int(strat.mean_long_size), int(strat.mean_short_size),
                strat.annual_turnover_one_way_pct, strat.tc_drag_annual_pct)

    strategy_returns = strat.daily_returns
    gross_returns    = strat.daily_gross

    # 7. Stats Method A
    sh_gross = annualized_sharpe(gross_returns)
    sh_net   = annualized_sharpe(strategy_returns)
    nw_t     = newey_west_t(strategy_returns, lag=NW_LAG_LOCKED)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, n_resamples=1000)
    ann_ret  = float(strategy_returns.mean() * 252)
    ann_vol  = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    # 8. Sub-periods
    sp_regime  = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)

    # 9. OOS hold-out
    oos = oos_hold_out(strategy_returns)

    # 10. Incremental α: dual-baseline (K1 + D-PEAD)
    incremental_K1 = {"gate_5_pass": False, "warning": "K1 baseline not loaded"}
    incremental_DPEAD = {"gate_5_pass": False, "warning": "D-PEAD baseline not loaded"}

    # K1 weekly → daily
    k1_path = REPO_ROOT / "data/path_c_k1/v1_k1_size_expanded_paired_returns.parquet"
    if k1_path.exists():
        k1_paired = pd.read_parquet(k1_path)
        weekly_returns = k1_paired['k1_weekly_returns'].values
        n_weeks = len(weekly_returns)
        weekly_dates = pd.date_range(start='2014-01-06', periods=n_weeks, freq='W-MON')
        k1_daily = pd.Series(0.0, index=strategy_returns.index)
        for i, wd in enumerate(weekly_dates):
            if i >= n_weeks: break
            daily_eq = (1 + weekly_returns[i]) ** (1/5) - 1
            for offset in range(5):
                target_d = wd + pd.Timedelta(days=offset)
                future = strategy_returns.index[strategy_returns.index >= target_d]
                if len(future) > 0:
                    k1_daily.loc[future[0]] = daily_eq
        incremental_K1 = incremental_alpha_vs_baseline(strategy_returns, k1_daily)

    # D-PEAD daily L-S net
    dpead_path = REPO_ROOT / "data/path_c_dhs/walk_forward_pead.parquet"
    if dpead_path.exists():
        dpead_df = pd.read_parquet(dpead_path)
        dpead_daily = dpead_df['r_long_short_net'].reindex(strategy_returns.index).fillna(0.0)
        incremental_DPEAD = incremental_alpha_vs_baseline(strategy_returns, dpead_daily)

    # Choose STRONGER baseline for Gate 5 (per spec §3.2): higher |IR| → stronger
    ir_K1    = abs(incremental_K1.get('information_ratio', 0.0) or 0.0)
    ir_DPEAD = abs(incremental_DPEAD.get('information_ratio', 0.0) or 0.0)
    stronger_baseline = "DPEAD" if ir_DPEAD >= ir_K1 else "K1"
    gate_5 = bool(
        incremental_DPEAD.get('gate_5_pass') if stronger_baseline == "DPEAD"
        else incremental_K1.get('gate_5_pass')
    )

    # Gates 1 / 3 / 4
    gate_1 = bool(sh_net is not None and not np.isnan(sh_net) and sh_net >= GATE_1_SHARPE_THRESHOLD
                  and nw_t is not None and not np.isnan(nw_t) and nw_t >= GATE_1_NW_T_THRESHOLD)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)

    # Decision
    if not gate_1:
        # Check MARGINAL
        if (sh_net is not None and not np.isnan(sh_net) and sh_net >= 0.3
            and nw_t is not None and not np.isnan(nw_t) and nw_t >= 1.5):
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

    verdict = PathHVerdict(
        decision=decision,
        spec_hash=spec_hash,
        window_start=start.isoformat(),
        window_end=end.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_cohorts=int(strat.n_cohorts),
        mean_long_leg_size=float(strat.mean_long_size),
        mean_short_leg_size=float(strat.mean_short_size),
        annual_turnover_one_way_pct=float(strat.annual_turnover_one_way_pct),
        tc_drag_annual_pct=float(strat.tc_drag_annual_pct),
        method_A_sharpe_gross=float(sh_gross) if not np.isnan(sh_gross) else None,
        method_A_sharpe_net=float(sh_net) if not np.isnan(sh_net) else None,
        method_A_nw_t=float(nw_t) if not np.isnan(nw_t) else None,
        method_A_ci_lower=float(ci_lo) if not np.isnan(ci_lo) else None,
        method_A_ci_upper=float(ci_hi) if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret,
        method_A_ann_vol=ann_vol,
        subperiod_regime=sp_regime,
        subperiod_random_rolling=sp_rolling,
        oos_hold_out=oos,
        incremental_alpha_vs_K1=incremental_K1,
        incremental_alpha_vs_DPEAD=incremental_DPEAD,
        gate_1_individual_pass=gate_1,
        gate_3_oos_pass=gate_3,
        gate_4_subperiod_pass=gate_4,
        gate_5_incremental_pass=gate_5,
        cumulative_return=cum_ret,
        max_drawdown=max_dd,
        honest_disclose=[
            f"Gate 5 used STRONGER baseline ({stronger_baseline}); both reported",
            "George-Hwang 2004 + Birru 2015 → 12-20y of arbitrage opportunity since publication",
            "252-day history requirement excludes IPOs <1y old (systematic bias against newest names)",
            "Monthly rebalance, daily return: turnover annualized ~20-30% one-way; TC drag estimate may be conservative or aggressive",
            "Possible NEGATIVE ρ with K1 BAB (BAB shorts high-beta = often near-high names) — dual-baseline forces honest reckoning",
            "Same-day spec lock + impl + run; spec hash 7ecbaa3e locked BEFORE backtest",
            "6-month overlapping cohorts create return autocorrelation; NW lag=126 may underestimate true SE if regime shifts mid-hold",
        ],
    )

    out_dir = REPO_ROOT / "data/path_h"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
    strategy_returns.to_frame("strategy_return").to_parquet(out_dir / f"{args.run_id}_daily.parquet")
    gross_returns.to_frame("gross_return").to_parquet(out_dir / f"{args.run_id}_gross_daily.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS (single-stock sleeve gates 0.5 / 2.0):")
    logger.info("  Sharpe gross: %.4f", sh_gross if not np.isnan(sh_gross) else 0)
    logger.info("  Sharpe net:   %.4f", sh_net if not np.isnan(sh_net) else 0)
    logger.info("  NW t (lag %d): %.4f", NW_LAG_LOCKED, nw_t if not np.isnan(nw_t) else 0)
    logger.info("  CI 95%%:       [%.4f, %.4f]", ci_lo or 0, ci_hi or 0)
    logger.info("  Ann return:   %.4f (= %.2f%%/yr)", ann_ret, ann_ret * 100)
    logger.info("  Ann vol:      %.4f", ann_vol)
    logger.info("Sub-period (regime split):")
    logger.info("  Pre-COVID:    Sharpe %.3f", sp_regime.get('pre_covid', {}).get('sharpe', 0))
    logger.info("  COVID:        Sharpe %.3f", sp_regime.get('covid', {}).get('sharpe', 0))
    logger.info("  Post-COVID:   Sharpe %.3f", sp_regime.get('post_covid', {}).get('sharpe', 0))
    logger.info("Incremental α vs K1: IR=%.3f, gate_5_pass=%s",
                incremental_K1.get('information_ratio', 0.0) or 0.0,
                incremental_K1.get('gate_5_pass', False))
    logger.info("Incremental α vs D-PEAD: IR=%.3f, gate_5_pass=%s",
                incremental_DPEAD.get('information_ratio', 0.0) or 0.0,
                incremental_DPEAD.get('gate_5_pass', False))
    logger.info("Gate 5 stronger baseline: %s", stronger_baseline)
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual 0.5/2.0): %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 2 (Selective BHY):      %s", verdict.gate_2_selective_bhy)
    logger.info("  Gate 3 (OOS hold-out):       %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):    %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):      %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative %dy: %.4f", (end.year - start.year), cum_ret)
    logger.info("Max DD:         %.4f", max_dd)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
