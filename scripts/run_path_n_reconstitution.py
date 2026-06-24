"""
scripts/run_path_n_reconstitution.py — Path N Index Reconstitution Drift CLI runner.

Pre-registration: docs/spec_path_n_index_reconstitution_drift_v1.md (id=70 hash c92d2c36) §八.

Pipeline:
  1. Spec hash validation
  2. Query CRSP msp500list for ADD events 2014-2023
  3. Fetch CRSP daily returns for affected permnos
  4. Build add-event strategy (T-5 to T-1 long-only, equal weight, 30bp TC)
  5. Stats + 5-gate single-stock verdict
  6. Dual-baseline incremental α (K1 + D-PEAD)
  7. Persist verdict + daily parquet
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

logger = logging.getLogger("path_n.run_reconstitution")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_n_reconstitution.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_reconstitution_10y")
    p.add_argument("--tc-bps", type=float, default=30.0,
                   help="Roundtrip TC bps; Amendment 1 SS-Tier-1 S&P 500 = 10bp")
    p.add_argument("--verbose","-v", action="store_true")
    return p.parse_args()


@dataclass
class PathNVerdict:
    decision:                    str
    spec_hash:                   str
    wave:                        str = "N-reconstitution"
    universe_source:             str = "crsp_msp500list_adds"
    window_start:                str = ""
    window_end:                  str = ""
    n_daily_obs:                 int = 0
    n_events:                    int = 0
    n_active_days:               int = 0
    mean_concurrent_events:      float = 0.0
    annual_turnover_events:      float = 0.0
    tc_drag_annual_pct:          float = 0.0
    method_A_sharpe_gross:       Optional[float] = None
    method_A_sharpe_net:         Optional[float] = None
    method_A_nw_t:               Optional[float] = None
    method_A_ci_lower:           Optional[float] = None
    method_A_ci_upper:           Optional[float] = None
    method_A_ann_return_gross:   float = 0.0
    method_A_ann_return_net:     float = 0.0
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
    cumulative_return_net:       float = 0.0
    max_drawdown_net:            float = 0.0
    honest_disclose:             list = field(default_factory=list)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import validate_reference, _compute_git_blob_hash, _resolve_to_abs
    from engine.path_n.reconstitution_strategy import (
        build_add_event_strategy, NW_LAG_LOCKED, PRE_EVENT_DAYS_LOCKED,
    )
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
    )
    from engine.universe_singlename.crsp_loader import _open_wrds_connection

    # Single-stock sleeve gates per standing rule
    GATE_1_SHARPE = 0.50
    GATE_1_NW_T   = 2.00
    MARG_SHARPE   = 0.30
    MARG_NW_T     = 1.50

    SPEC_PATH = "docs/spec_path_n_index_reconstitution_drift_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start = datetime.date.fromisoformat(args.start)
    end   = datetime.date.fromisoformat(args.end)

    # ── Step 1: Query S&P 500 add events ──────────────────────────────────────
    logger.info("Querying CRSP msp500list for ADD events %s to %s...", start, end)
    conn = _open_wrds_connection()
    try:
        sql = f"""
        SELECT permno, start, ending
        FROM crsp.msp500list
        WHERE start BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
        ORDER BY start
        """
        events_raw = conn.raw_sql(sql, date_cols=['start', 'ending'])
        events = events_raw.rename(columns={'start': 'effective_date'}).copy()
        events['event_type'] = 'ADD'
        events = events[['permno', 'effective_date', 'event_type']]
        events['permno'] = events['permno'].astype(int)
        logger.info("Add events found: %d", len(events))

        # Fetch CRSP daily returns
        permno_list = ",".join(str(p) for p in sorted(events['permno'].unique()))
        # Need T-5 buffer pre-window
        fetch_start = (start - datetime.timedelta(days=30)).isoformat()
        fetch_end = end.isoformat()
        sql = f"""
        SELECT permno, date, ret
        FROM crsp.dsf
        WHERE permno IN ({permno_list})
          AND date BETWEEN '{fetch_start}' AND '{fetch_end}'
        """
        daily = conn.raw_sql(sql, date_cols=['date'])
    finally:
        conn.close()

    daily['ret'] = pd.to_numeric(daily['ret'], errors='coerce')
    daily['permno'] = daily['permno'].astype(int)
    panel = daily.pivot_table(index='date', columns='permno', values='ret', aggfunc='first')
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    logger.info("Daily returns panel: %d × %d", *panel.shape)

    # ── Step 2: Build strategy ────────────────────────────────────────────────
    logger.info("Building add-event strategy (T-5 to T-1, long-only, %.1fbp TC)...", args.tc_bps)
    result = build_add_event_strategy(events, panel, tc_bps_roundtrip=args.tc_bps)
    logger.info("Strategy: %d events, %d active days, mean concurrent %.1f, "
                "annual turnover %.1f events/yr, TC drag %.2f%%/yr",
                result.n_events, result.n_active_days, result.mean_concurrent_events,
                result.annual_turnover, result.tc_drag_annual_pct)

    strategy_returns = result.daily_returns.loc[str(start):str(end)]
    gross_returns    = result.daily_gross.loc[str(start):str(end)]

    # ── Step 3: Stats ─────────────────────────────────────────────────────────
    sh_gross = annualized_sharpe(gross_returns)
    sh_net   = annualized_sharpe(strategy_returns)
    nw_t     = newey_west_t(strategy_returns, lag=NW_LAG_LOCKED)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, n_resamples=1000)
    ann_ret_gross = float(gross_returns.mean() * 252)
    ann_ret_net   = float(strategy_returns.mean() * 252)
    ann_vol       = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    sp_regime  = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)
    oos = oos_hold_out(strategy_returns)

    # ── Step 4: Incremental α dual-baseline ───────────────────────────────────
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

    # ── Step 5: Gates ─────────────────────────────────────────────────────────
    gate_1 = bool(sh_net is not None and not np.isnan(sh_net) and sh_net >= GATE_1_SHARPE
                  and nw_t is not None and not np.isnan(nw_t) and nw_t >= GATE_1_NW_T)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)

    if not gate_1:
        if sh_net >= MARG_SHARPE and nw_t >= MARG_NW_T:
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

    verdict = PathNVerdict(
        decision=decision, spec_hash=spec_hash,
        window_start=start.isoformat(), window_end=end.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_events=int(result.n_events),
        n_active_days=int(result.n_active_days),
        mean_concurrent_events=float(result.mean_concurrent_events),
        annual_turnover_events=float(result.annual_turnover),
        tc_drag_annual_pct=float(result.tc_drag_annual_pct),
        method_A_sharpe_gross=float(sh_gross) if not np.isnan(sh_gross) else None,
        method_A_sharpe_net=float(sh_net) if not np.isnan(sh_net) else None,
        method_A_nw_t=float(nw_t) if not np.isnan(nw_t) else None,
        method_A_ci_lower=float(ci_lo) if not np.isnan(ci_lo) else None,
        method_A_ci_upper=float(ci_hi) if not np.isnan(ci_hi) else None,
        method_A_ann_return_gross=ann_ret_gross,
        method_A_ann_return_net=ann_ret_net,
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
        cumulative_return_net=cum_ret,
        max_drawdown_net=max_dd,
        honest_disclose=[
            "Chen-Noronha-Singal 2004 = 22y post-publication; Patel-Welch 2017 showed effect halved 1995-2014",
            "Pre-event window assumed 5 trading days (canonical CNS 2004); actual S&P lead varies",
            "Scout gross Sharpe 0.81; TC modeling applied -10 to -20bp Sharpe expected",
            "Pre-COVID Sharpe ~0.42 below strict single-stock gate 0.5 (above 0.4 ETF gate)",
            "Sub-period concentration: COVID + Post-COVID Sharpe 1.59 / 0.99 drive aggregate",
            "Survivorship-clean by construction (CRSP msp500list is point-in-time historical)",
            "Universe rule-based (S&P 500 official adds, no judgment selection)",
            "Capacity $100-500M (large-cap names but 5-day hold limits scale)",
            "Spec hash c92d2c36 locked BEFORE formal backtest",
        ],
    )

    out_dir = REPO_ROOT / "data/path_n"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
    strategy_returns.to_frame("strategy_return").to_parquet(out_dir / f"{args.run_id}_daily.parquet")
    gross_returns.to_frame("gross_return").to_parquet(out_dir / f"{args.run_id}_gross_daily.parquet")
    result.event_returns.to_parquet(out_dir / f"{args.run_id}_event_returns.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS (single-stock sleeve gates 0.5 / 2.0):")
    logger.info("  Sharpe gross: %.4f", sh_gross if not np.isnan(sh_gross) else 0)
    logger.info("  Sharpe net:   %.4f", sh_net if not np.isnan(sh_net) else 0)
    logger.info("  NW t (lag %d): %.4f", NW_LAG_LOCKED, nw_t if not np.isnan(nw_t) else 0)
    logger.info("  CI 95%%:       [%.4f, %.4f]", ci_lo or 0, ci_hi or 0)
    logger.info("  Ann ret gross: %.4f  net: %.4f", ann_ret_gross, ann_ret_net)
    logger.info("  Ann vol:       %.4f", ann_vol)
    logger.info("  TC drag/yr:    %.2f%%", result.tc_drag_annual_pct)
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
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual 0.5/2.0): %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 3 (OOS hold-out):       %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):    %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):      %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative net: %.4f, Max DD net: %.4f", cum_ret, max_dd)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
