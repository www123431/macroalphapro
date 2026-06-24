"""
scripts/run_path_l_profitability.py — Path L Novy-Marx Profitability CLI runner.

Pre-registration: docs/spec_path_l_profitability_v1.md (id=68 hash 5a2ab1cc) §八.

Pipeline:
  1. Spec hash validation
  2. Build top-1500 CRSP universe (3 sampling dates)
  3. Fetch GPA signal panel via comp.fundq
  4. Top-N market cap filter per quarter (consistent with Path D)
  5. Cross-section rank + decile leg assignment
  6. Daily returns panel (CRSP dsf)
  7. Walk-forward L/S backtest (reuse Path C pead_backtest)
  8. Stats + 5-gate verdict
  9. Ensemble test: 50/50 Path L + D-PEAD blend
 10. Persist verdict + daily parquet
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

logger = logging.getLogger("path_l.run_profitability")


def _parse_args():
    p = argparse.ArgumentParser(prog="run_path_l_profitability.py")
    p.add_argument("--start",  default="2014-01-01")
    p.add_argument("--end",    default="2023-12-31")
    p.add_argument("--run-id", default="v1_profitability_10y")
    p.add_argument("--top-n",  type=int, default=1500)
    p.add_argument("--mock",   action="store_true")
    p.add_argument("--smoke",  action="store_true", help="2014 single year smoke")
    p.add_argument("--verbose","-v", action="store_true")
    return p.parse_args()


@dataclass
class PathLVerdict:
    decision:                  str
    spec_hash:                  str
    wave:                       str = "L-profitability"
    universe_source:            str = "crsp_top1500_compustat"
    window_start:               str = ""
    window_end:                 str = ""
    n_daily_obs:                int = 0
    n_quarters:                 int = 0
    n_firm_quarters_used:       int = 0
    n_firm_quarters_excluded:   int = 0
    exclusion_breakdown:        dict = field(default_factory=dict)
    method_A_sharpe_gross:      Optional[float] = None
    method_A_sharpe_net:        Optional[float] = None
    method_A_nw_t:              Optional[float] = None
    method_A_ci_lower:          Optional[float] = None
    method_A_ci_upper:          Optional[float] = None
    method_A_ann_return:        float = 0.0
    method_A_ann_vol:           float = 0.0
    subperiod_regime:           dict = field(default_factory=dict)
    subperiod_random_rolling:   dict = field(default_factory=dict)
    oos_hold_out:               dict = field(default_factory=dict)
    incremental_alpha_vs_K1:    dict = field(default_factory=dict)
    incremental_alpha_vs_DPEAD: dict = field(default_factory=dict)
    gate_1_individual_pass:     bool = False
    gate_2_selective_bhy:       str = "DEMOTED_SINGLE_TEST"
    gate_3_oos_pass:            bool = False
    gate_4_subperiod_pass:      bool = False
    gate_5_incremental_pass:    bool = False
    cumulative_return:          float = 0.0
    max_drawdown:               float = 0.0
    # Ensemble test fields (NEW for Path L)
    ensemble_dpead_sharpe:      Optional[float] = None
    ensemble_dpead_nw_t:        Optional[float] = None
    ensemble_dpead_correlation: Optional[float] = None
    ensemble_lift_pct:          Optional[float] = None
    ensemble_verdict:           str = "NOT_COMPUTED"
    post_covid_ensemble_sharpe: Optional[float] = None
    post_covid_lift_verdict:    str = "NOT_COMPUTED"
    honest_disclose:            list = field(default_factory=list)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from engine.preregistration import validate_reference, _compute_git_blob_hash, _resolve_to_abs
    from engine.path_l.profitability_signal_panel import (
        bulk_fetch_profitability_signal_panel, is_wrds_available,
    )
    from engine.path_c.sue_signal import rank_within_quarter, assign_decile_legs
    from engine.path_c.pead_backtest import (
        run_walk_forward_pead, persist_walk_forward_result, HOLD_TRADING_DAYS_LOCKED,
    )
    from engine.path_f.vix_backtest import (
        annualized_sharpe, newey_west_t, bootstrap_ci_sharpe,
        regime_sub_period, random_rolling_sub_period, oos_hold_out,
        incremental_alpha_vs_baseline,
    )

    GATE_1_SHARPE = 0.50  # single-stock sleeve
    GATE_1_NW_T   = 2.00
    MARG_SHARPE   = 0.30
    MARG_NW_T     = 1.50

    SPEC_PATH = "docs/spec_path_l_profitability_v1.md"
    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason); return 2
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    if args.smoke:
        end_date = datetime.date(start_date.year, 12, 31)
        logger.info("SMOKE MODE — %s → %s", start_date, end_date)

    # ── Step 1: Universe ──────────────────────────────────────────────────────
    if args.mock:
        tickers = [f"MOCK{i:03d}" for i in range(args.top_n)]
        logger.info("MOCK MODE: %d synthetic tickers", len(tickers))
    else:
        if not is_wrds_available():
            logger.error("WRDS not available"); return 3
        from engine.universe_singlename.constituents_loader import load_russell2000_proxy_at_date
        sampling_dates = [
            datetime.date(start_date.year, 1, 31),
            datetime.date((start_date.year + end_date.year) // 2, 6, 30),
            datetime.date(end_date.year, 6, 30),
        ]
        tickers_set: set = set()
        for s_date in sampling_dates:
            if s_date > end_date: continue
            r = load_russell2000_proxy_at_date(s_date, rank_min=1, rank_max=args.top_n)
            tickers_set.update(r.tickers)
            logger.info("  %s: %d tickers", s_date, len(r.tickers))
        tickers = sorted(tickers_set)
        logger.info("Universe union: %d unique tickers", len(tickers))

    # ── Step 2: Signal panel ──────────────────────────────────────────────────
    logger.info("Fetching GPA signal panel for %d tickers", len(tickers))
    result = bulk_fetch_profitability_signal_panel(
        tickers=tickers, start_date=start_date, end_date=end_date,
        mock_mode=args.mock,
    )
    logger.info("GPA panel: %d firm-quarters (mode=%s)", result.n_firm_quarters, result.mode)
    if result.panel.empty:
        logger.error("Empty panel — aborting"); return 5

    panel = result.panel.copy()

    # Top-N filter per quarter
    if "market_cap_at_q" in panel.columns and panel["market_cap_at_q"].notna().any():
        keep_indices = []
        for quarter, group in panel.groupby("fiscal_yearq"):
            top_n_idx = group.nlargest(args.top_n, "market_cap_at_q").index
            keep_indices.extend(top_n_idx)
        panel = panel.loc[keep_indices].reset_index(drop=True)
        logger.info("Panel after top-%d filter: %d firm-quarters", args.top_n, len(panel))

    # Drop rows with NaN gpa (insufficient history)
    n_before = len(panel)
    panel = panel[panel["gpa"].notna()].copy()
    n_excluded = n_before - len(panel)
    logger.info("Panel after gpa NaN drop: %d (excluded %d)", len(panel), n_excluded)

    # ── Step 3: Rank + decile leg ─────────────────────────────────────────────
    ranked = rank_within_quarter(panel.copy(), sue_col="gpa", tie_break_col="ticker")
    # rank_within_quarter outputs `sue_rank_pct` column name regardless of input col
    legged = assign_decile_legs(
        ranked,
        long_threshold=0.9, short_threshold=0.1,
        rank_col="sue_rank_pct",
    )
    leg_counts = legged["leg"].value_counts().to_dict()
    logger.info("Decile legs: %s", leg_counts)

    # ── Step 4: Daily returns panel ────────────────────────────────────────────
    last_rdq = legged["rdq"].max()
    if hasattr(last_rdq, "date"):
        last_rdq = last_rdq.date()
    returns_end = max(end_date, last_rdq + datetime.timedelta(days=90)) if last_rdq else end_date
    logger.info("Fetching daily returns [%s, %s]", start_date, returns_end)

    if args.mock:
        bdates = pd.bdate_range(start=start_date, end=returns_end)
        rng = np.random.default_rng(42)
        cols = {t: rng.normal(0.0001, 0.01, size=len(bdates)) for t in tickers}
        returns_panel = pd.DataFrame(cols, index=bdates)
        returns_panel.index.name = "date"
    else:
        from engine.universe_singlename.crsp_loader import bulk_fetch_crsp_daily_panel
        price_panel = bulk_fetch_crsp_daily_panel(
            tickers=tickers, start_date=start_date, end_date=returns_end,
            mock_mode=False,
        )
        returns_panel = price_panel.pct_change(fill_method=None).dropna(how="all")
    logger.info("Returns panel: %d × %d", *returns_panel.shape)

    # ── Step 5: Walk-forward ──────────────────────────────────────────────────
    sig_for_bt = legged.rename(columns={"ticker": "ticker_ibes"})
    logger.info("Walk-forward profitability, hold=%d days", HOLD_TRADING_DAYS_LOCKED)
    wf = run_walk_forward_pead(
        signal_panel=sig_for_bt,
        returns_panel=returns_panel,
        window_start=start_date, window_end=end_date,
        checkpoint_run_id=args.run_id,
        spec_hash_at_run=spec_hash,
    )
    if wf.daily_returns.empty:
        logger.error("Walk-forward empty"); return 6

    daily = wf.daily_returns
    strategy_returns = daily["r_long_short_net"]
    gross_returns    = daily["r_long_short"]

    # ── Step 6: Stats ─────────────────────────────────────────────────────────
    sh_gross = annualized_sharpe(gross_returns)
    sh_net   = annualized_sharpe(strategy_returns)
    nw_t     = newey_west_t(strategy_returns, lag=60)
    ci_lo, ci_hi = bootstrap_ci_sharpe(strategy_returns, n_resamples=1000)
    ann_ret  = float(strategy_returns.mean() * 252)
    ann_vol  = float(strategy_returns.std(ddof=1) * np.sqrt(252))

    sp_regime  = regime_sub_period(strategy_returns)
    sp_rolling = random_rolling_sub_period(strategy_returns)
    oos = oos_hold_out(strategy_returns)

    # ── Step 7: Incremental α vs K1 + D-PEAD ──────────────────────────────────
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
    else:
        dpead_daily = None

    ir_K1    = abs(incremental_K1.get('information_ratio', 0.0) or 0.0)
    ir_DPEAD = abs(incremental_DPEAD.get('information_ratio', 0.0) or 0.0)
    stronger = "DPEAD" if ir_DPEAD >= ir_K1 else "K1"
    gate_5 = bool(
        incremental_DPEAD.get('gate_5_pass') if stronger == "DPEAD"
        else incremental_K1.get('gate_5_pass')
    )

    # ── Step 8: Gates ─────────────────────────────────────────────────────────
    gate_1 = bool(sh_net is not None and not np.isnan(sh_net) and sh_net >= GATE_1_SHARPE
                  and nw_t is not None and not np.isnan(nw_t) and nw_t >= GATE_1_NW_T)
    gate_3 = bool(oos["gate_3_pass"])
    gate_4 = bool(sp_regime.get("regime_all_positive") and sp_rolling.get("all_positive") is True)

    if gate_1:
        decision = "PASS" if (gate_3 and gate_4 and gate_5) else "INDIVIDUAL_PASS_BUT_NON_INDEPENDENT"
    elif sh_net >= MARG_SHARPE and nw_t >= MARG_NW_T:
        decision = "MARGINAL"
    else:
        decision = "FAIL"

    cum = (1 + strategy_returns).cumprod()
    cum_ret = float(cum.iloc[-1] - 1.0) if len(cum) > 0 else 0.0
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # ── Step 9: Ensemble test 50/50 with D-PEAD ───────────────────────────────
    ensemble_verdict_str = "NOT_COMPUTED"
    ensemble_sharpe = None
    ensemble_nw = None
    rho_LD = None
    ensemble_lift_pct = None
    post_covid_ensemble_sh = None
    post_covid_lift_verdict_str = "NOT_COMPUTED"

    if dpead_daily is not None:
        ensemble = 0.5 * strategy_returns + 0.5 * dpead_daily
        ensemble_sharpe = annualized_sharpe(ensemble)
        ensemble_nw = newey_west_t(ensemble, lag=60)
        sh_dpead_full = annualized_sharpe(dpead_daily)
        rho_LD = float(strategy_returns.corr(dpead_daily))

        best_individual = max(
            sh_net if not np.isnan(sh_net) else 0,
            sh_dpead_full if not np.isnan(sh_dpead_full) else 0,
        )
        if best_individual > 0:
            ensemble_lift_pct = (ensemble_sharpe / best_individual - 1.0) * 100
        else:
            ensemble_lift_pct = None

        # Diversification verdict
        if (ensemble_sharpe >= 1.1 * best_individual) and abs(rho_LD) < 0.4:
            ensemble_verdict_str = "ENSEMBLE_LIFT"
        else:
            ensemble_verdict_str = "NO_ENSEMBLE_LIFT"

        # Post-COVID specific ensemble test
        post_covid = ensemble.loc["2022-01-01":]
        if len(post_covid) > 30:
            post_covid_ensemble_sh = annualized_sharpe(post_covid)
            sh_dpead_post = annualized_sharpe(dpead_daily.loc["2022-01-01":])
            if post_covid_ensemble_sh > max(0.8, 1.1 * sh_dpead_post):
                post_covid_lift_verdict_str = "POST_COVID_LIFT_CONFIRMED"
            else:
                post_covid_lift_verdict_str = "NO_POST_COVID_LIFT"

    verdict = PathLVerdict(
        decision=decision, spec_hash=spec_hash,
        window_start=start_date.isoformat(), window_end=end_date.isoformat(),
        n_daily_obs=int(len(strategy_returns)),
        n_quarters=int(wf.n_quarters_processed),
        n_firm_quarters_used=int(wf.n_firm_quarters_active),
        n_firm_quarters_excluded=int(n_excluded),
        exclusion_breakdown=result.exclusion_stats,
        method_A_sharpe_gross=float(sh_gross) if not np.isnan(sh_gross) else None,
        method_A_sharpe_net=float(sh_net) if not np.isnan(sh_net) else None,
        method_A_nw_t=float(nw_t) if not np.isnan(nw_t) else None,
        method_A_ci_lower=float(ci_lo) if not np.isnan(ci_lo) else None,
        method_A_ci_upper=float(ci_hi) if not np.isnan(ci_hi) else None,
        method_A_ann_return=ann_ret, method_A_ann_vol=ann_vol,
        subperiod_regime=sp_regime, subperiod_random_rolling=sp_rolling,
        oos_hold_out=oos,
        incremental_alpha_vs_K1=incremental_K1,
        incremental_alpha_vs_DPEAD=incremental_DPEAD,
        gate_1_individual_pass=gate_1,
        gate_3_oos_pass=gate_3, gate_4_subperiod_pass=gate_4,
        gate_5_incremental_pass=gate_5,
        cumulative_return=cum_ret, max_drawdown=max_dd,
        ensemble_dpead_sharpe=float(ensemble_sharpe) if ensemble_sharpe is not None and not np.isnan(ensemble_sharpe) else None,
        ensemble_dpead_nw_t=float(ensemble_nw) if ensemble_nw is not None and not np.isnan(ensemble_nw) else None,
        ensemble_dpead_correlation=rho_LD,
        ensemble_lift_pct=ensemble_lift_pct,
        ensemble_verdict=ensemble_verdict_str,
        post_covid_ensemble_sharpe=post_covid_ensemble_sh,
        post_covid_lift_verdict=post_covid_lift_verdict_str,
        honest_disclose=[
            "Novy-Marx 2013 13 years post-pub; AFP 2019 QMJ 7 years; substantial arbitrage time",
            "FF5 RMW (operating profitability) 11 years deployed since 2015",
            "Quality ETFs (QUAL/USMV/SPHQ/JKQ) explicitly include profitability scoring",
            "TTM 4Q sum GP / lagged 4Q-avg TA per Novy-Marx canonical (not optimized)",
            "Ensemble test exploratory; doesn't change individual Path L PASS/FAIL",
            "Same-day spec lock + impl + run; spec hash 5a2ab1cc locked BEFORE backtest",
            "Path L tests if fundamental quality (Novy-Marx) complements behavioral underreaction (D-PEAD)",
        ],
    )

    out_dir = REPO_ROOT / "data/path_l"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_dict = asdict(verdict)
    verdict_dict["run_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / f"{args.run_id}_verdict.json").write_text(
        json.dumps(verdict_dict, indent=2, default=str), encoding="utf-8")
    persist_walk_forward_result(wf, parquet_path=out_dir / f"walk_forward_profitability.parquet")

    logger.info("=" * 70)
    logger.info("FINAL DECISION: %s", decision)
    logger.info("Method A daily TS (single-stock sleeve gates 0.5 / 2.0):")
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
    logger.info("Incremental α vs K1:    IR=%.4f, gate_5=%s",
                incremental_K1.get('information_ratio', 0) or 0,
                incremental_K1.get('gate_5_pass', False))
    logger.info("Incremental α vs DPEAD: IR=%.4f, gate_5=%s",
                incremental_DPEAD.get('information_ratio', 0) or 0,
                incremental_DPEAD.get('gate_5_pass', False))
    if ensemble_sharpe is not None:
        logger.info("ENSEMBLE (50/50 L + DPEAD):")
        logger.info("  Ensemble Sharpe:  %.4f", ensemble_sharpe)
        logger.info("  Ensemble NW t:    %.4f", ensemble_nw)
        logger.info("  ρ(L, DPEAD):      %.4f", rho_LD)
        logger.info("  Lift vs best individual: %s",
                    f"{ensemble_lift_pct:+.2f}%" if ensemble_lift_pct is not None else "n/a")
        logger.info("  Verdict:          %s", ensemble_verdict_str)
        if post_covid_ensemble_sh is not None:
            logger.info("  Post-COVID ensemble Sharpe: %.4f  → %s",
                        post_covid_ensemble_sh, post_covid_lift_verdict_str)
    logger.info("5-Gate:")
    logger.info("  Gate 1 (Individual 0.5/2.0): %s", "PASS" if gate_1 else "FAIL")
    logger.info("  Gate 3 (OOS):                %s", "PASS" if gate_3 else "FAIL")
    logger.info("  Gate 4 (Sub-period dual):    %s", "PASS" if gate_4 else "FAIL")
    logger.info("  Gate 5 (Incremental α):      %s", "PASS" if gate_5 else "FAIL")
    logger.info("Cumulative %dy: %.4f", (end_date.year - start_date.year + 1), cum_ret)
    logger.info("Max DD:         %.4f", max_dd)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
