"""
scripts/run_dhs_behavioral_walk_forward.py — Path D DHS 2020 behavioral 2-factor CLI.

Pre-registration: docs/spec_path_d_dhs_behavioral_2factor_v1.md (id=62) §4.1 + §八.

Pipeline (3 portfolios in parallel):
  1. Load top-N CRSP universe via load_russell2000_proxy_at_date(rank_min=1, rank_max=N)
  2. Fetch PEAD-TS signal panel (Compustat fundq epspxq + ajexq)
  3. Fetch FIN signal panel (Compustat fundq cshoq + balance sheet items)
  4. Top-N filter per quarter via market_cap_at_q (PEAD panel as canonical universe)
  5. Build 3 signal panels:
     a. PEAD-only — assign decile legs on sue
     b. FIN-only  — assign decile legs on fin composite (via assign_fin_decile_legs)
     c. COMBINED  — inner-join PEAD + FIN, average ranks, decile leg
  6. Fetch CRSP daily returns panel (shared across 3 walk-forwards)
  7. Run 3 walk-forwards using pead_backtest.run_walk_forward_pead
  8. Build per-portfolio verdicts + sub-period split (Pre/COVID/Post)
  9. Compute aggregate decision code per spec §3.3
 10. Persist 3 walk-forward parquets + 1 aggregate verdict.json

Usage:
  py -3.11 scripts/run_dhs_behavioral_walk_forward.py
  py -3.11 scripts/run_dhs_behavioral_walk_forward.py --start 2014-01-01 --end 2023-12-31 \
                                                    --top-n 1500 --run-id v1_dhs_10y
  py -3.11 scripts/run_dhs_behavioral_walk_forward.py --mock --top-n 30 --start 2014-01-01 --end 2015-12-31
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

logger = logging.getLogger("path_d.run_dhs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_dhs_behavioral_walk_forward.py",
        description="Path D DHS 2020 PEAD time-series + FIN composite walk-forward orchestrator",
    )
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end",   default="2023-12-31")
    p.add_argument("--top-n", type=int, default=1500)
    p.add_argument("--run-id", default="v1_dhs_10y")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _classify_per_portfolio(sharpe_net: float, nw_t: float) -> str:
    """Spec §3.2 industry-grade gates (BHY-demoted, single-test)."""
    if sharpe_net >= 0.5 and nw_t >= 2.0:
        return "PASS"
    if sharpe_net >= 0.3 and nw_t >= 1.5:
        return "MARGINAL"
    return "FAIL"


def _sub_period_stats(
    daily_series: pd.Series,
    *,
    nw_lag: int = 60,
) -> dict:
    """Compute Sharpe + NW t for Pre-COVID (≤2019), COVID (2020-2021), Post-COVID (2022-2023)."""
    from engine.path_c.verdict import (
        compute_annualized_sharpe, compute_nw_t_stat,
    )
    daily_series = daily_series.copy()
    daily_series.index = pd.to_datetime(daily_series.index)

    periods = {
        "pre_covid":  (datetime.date(2014, 1, 1), datetime.date(2019, 12, 31)),
        "covid":      (datetime.date(2020, 1, 1), datetime.date(2021, 12, 31)),
        "post_covid": (datetime.date(2022, 1, 1), datetime.date(2023, 12, 31)),
    }
    out = {}
    for label, (lo, hi) in periods.items():
        mask = (daily_series.index.date >= lo) & (daily_series.index.date <= hi)
        seg = daily_series[mask].dropna()
        if len(seg) < 20:
            out[label] = {"n_obs": int(len(seg)), "sharpe": None, "nw_t": None}
            continue
        sharpe = float(compute_annualized_sharpe(seg, periods_per_year=252))
        nw_t   = float(compute_nw_t_stat(seg, lag=nw_lag))
        out[label] = {
            "n_obs":  int(len(seg)),
            "sharpe": sharpe if np.isfinite(sharpe) else None,
            "nw_t":   nw_t   if np.isfinite(nw_t)   else None,
        }
    return out


def _classify_aggregate(per_portfolio: dict, subperiods: dict) -> str:
    """Spec §3.3 project-level aggregate code."""
    decisions = [per_portfolio[p]["decision"] for p in ("PEAD", "FIN", "COMBINED")]
    sharpes   = [per_portfolio[p]["sharpe_net"] for p in ("PEAD", "FIN", "COMBINED")]

    n_pass = sum(1 for d in decisions if d == "PASS")
    n_marginal = sum(1 for d in decisions if d == "MARGINAL")
    n_positive = sum(1 for s in sharpes if (s is not None and s > 0))

    if n_pass >= 1:
        return "DHS_FULL_PASS"
    if n_marginal >= 1 or n_positive >= 2:
        return "DHS_MARGINAL_DIRECTIONAL"

    # Check regime-stable: ≥ 2 portfolios with all 3 sub-periods Sharpe positive
    n_regime_stable = 0
    for p in ("PEAD", "FIN", "COMBINED"):
        sp = subperiods.get(p, {})
        if (sp.get("pre_covid",  {}).get("sharpe") or 0) > 0 and \
           (sp.get("covid",      {}).get("sharpe") or 0) > 0 and \
           (sp.get("post_covid", {}).get("sharpe") or 0) > 0:
            n_regime_stable += 1
    if n_regime_stable >= 2:
        return "DHS_REGIME_STABLE_FAIL"
    return "DHS_FAIL_AGGREGATE"


def _build_signal_panel_pead_only(pead_panel: pd.DataFrame) -> pd.DataFrame:
    """PEAD-only: rank within quarter on sue + assign decile legs."""
    from engine.path_c.sue_signal import rank_within_quarter, assign_decile_legs
    if pead_panel.empty:
        return pead_panel.copy()
    ranked = rank_within_quarter(
        pead_panel.copy(),
        sue_col="sue",
        tie_break_col="ticker",
    )
    legged = assign_decile_legs(
        ranked,
        long_threshold=0.9,
        short_threshold=0.1,
        rank_col="sue_rank_pct",
    )
    return legged


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start_date = datetime.date.fromisoformat(args.start)
    end_date   = datetime.date.fromisoformat(args.end)
    mock_mode  = args.mock

    from engine.path_c.pead_ts_signal_panel import (
        bulk_fetch_pead_ts_signal_panel,
        is_wrds_available,
    )
    from engine.path_c.fin_signal_panel import bulk_fetch_fin_signal_panel
    from engine.path_c.fin_signal import assign_fin_decile_legs
    from engine.path_c.dhs_combined_signal import assign_combined_decile_legs
    from engine.path_c.pead_backtest import (
        run_walk_forward_pead,
        persist_walk_forward_result,
        HOLD_TRADING_DAYS_LOCKED,
    )
    from engine.path_c.verdict import build_pead_verdict, persist_verdict
    from engine.preregistration import (
        validate_reference, compute_pre_registration_n_trials,
        _compute_git_blob_hash, _resolve_to_abs,
    )

    SPEC_PATH = "docs/spec_path_d_dhs_behavioral_2factor_v1.md"

    ok, reason = validate_reference(SPEC_PATH)
    if not ok:
        logger.error("Spec validation failed: %s", reason)
        return 2
    logger.info("Spec validated.")
    n_trials = compute_pre_registration_n_trials()
    logger.info("Project cumulative n_trials = %d", n_trials)
    spec_hash = _compute_git_blob_hash(_resolve_to_abs(SPEC_PATH))
    logger.info("Spec hash: %s", spec_hash)

    # ── Step 1: Universe ─────────────────────────────────────────────────────
    if mock_mode:
        tickers = [f"MOCK{i:03d}" for i in range(args.top_n)]
        logger.info("MOCK MODE: %d synthetic tickers", len(tickers))
    else:
        if not is_wrds_available():
            logger.error("WRDS not configured. Pass --mock for synthetic smoke.")
            return 3
        from engine.universe_singlename.constituents_loader import (
            load_russell2000_proxy_at_date,
        )
        try:
            # Sample at 3 dates across window to capture entering/exiting firms
            sampling_dates = [
                datetime.date(2014, 1, 31),
                datetime.date(2018, 6, 30),
                datetime.date(2023, 6, 30),
            ]
            tickers_set: set = set()
            for s_date in sampling_dates:
                if s_date > end_date:
                    continue
                r = load_russell2000_proxy_at_date(s_date, rank_min=1, rank_max=args.top_n)
                tickers_set.update(r.tickers)
                logger.info("  %s: %d tickers (top-%d)", s_date, len(r.tickers), args.top_n)
            tickers = sorted(tickers_set)
            logger.info("Universe union: %d unique tickers", len(tickers))
        except Exception as exc:
            logger.error("Universe loader failed: %s", exc)
            return 4

    # ── Step 2: PEAD-TS panel ────────────────────────────────────────────────
    logger.info("Fetching PEAD-TS signal panel for %d tickers", len(tickers))
    pead_result = bulk_fetch_pead_ts_signal_panel(
        tickers=tickers, start_date=start_date, end_date=end_date,
        mock_mode=mock_mode,
    )
    logger.info("PEAD-TS panel: %d firm-quarters (mode=%s)",
                pead_result.n_firm_quarters, pead_result.mode)
    if pead_result.panel.empty:
        logger.error("Empty PEAD-TS panel — aborting.")
        return 5

    # ── Step 3: FIN panel ────────────────────────────────────────────────────
    logger.info("Fetching FIN signal panel for %d tickers", len(tickers))
    fin_result = bulk_fetch_fin_signal_panel(
        tickers=tickers, start_date=start_date, end_date=end_date,
        mock_mode=mock_mode,
    )
    logger.info("FIN panel: %d firm-quarters", fin_result.n_firm_quarters)

    # ── Step 4: Top-N filter per quarter (PEAD panel canonical) ──────────────
    pead_panel = pead_result.panel.copy()
    fin_panel  = fin_result.panel.copy()

    if "market_cap_at_q" in pead_panel.columns and pead_panel["market_cap_at_q"].notna().any():
        keep_indices = []
        for quarter, group in pead_panel.groupby("fiscal_yearq"):
            top_n_idx = group.nlargest(args.top_n, "market_cap_at_q").index
            keep_indices.extend(top_n_idx)
        pead_panel = pead_panel.loc[keep_indices].reset_index(drop=True)
        logger.info("PEAD panel after top-%d filter: %d firm-quarters",
                    args.top_n, len(pead_panel))

    # Drop PEAD rows with NaN sue (thin sigma / no eps_lag4)
    pead_panel = pead_panel[pead_panel["sue"].notna()].copy()
    logger.info("PEAD panel after NaN sue drop: %d", len(pead_panel))

    # Apply same top-N gvkeys to FIN panel (universe consistency)
    pead_gvkeys = set(pead_panel["gvkey"].astype(str))
    fin_panel["gvkey"] = fin_panel["gvkey"].astype(str)
    fin_panel = fin_panel[fin_panel["gvkey"].isin(pead_gvkeys)].copy()
    logger.info("FIN panel after universe-match filter: %d", len(fin_panel))

    # ── Step 5: Build 3 signal panels ────────────────────────────────────────
    logger.info("Building 3 signal panels: PEAD / FIN / COMBINED")
    sig_pead = _build_signal_panel_pead_only(pead_panel)
    sig_fin  = assign_fin_decile_legs(fin_panel)
    sig_comb = assign_combined_decile_legs(pead_panel, fin_panel)

    for label, sig in [("PEAD", sig_pead), ("FIN", sig_fin), ("COMBINED", sig_comb)]:
        if sig.empty:
            logger.warning("%s signal panel empty — will produce empty walk-forward", label)
            continue
        leg_counts = sig["leg"].value_counts().to_dict() if "leg" in sig.columns else {}
        logger.info("  %s: %d firm-quarters, legs=%s", label, len(sig), leg_counts)

    # ── Step 6: Daily returns panel (shared) ─────────────────────────────────
    all_signal_rdqs = pd.concat([sig_pead.get("rdq"), sig_fin.get("rdq"), sig_comb.get("rdq")],
                                 axis=0)
    last_rdq = all_signal_rdqs.dropna().max() if not all_signal_rdqs.empty else None
    if hasattr(last_rdq, "date"):
        last_rdq = last_rdq.date()
    returns_end = max(end_date, last_rdq + datetime.timedelta(days=90)) if last_rdq else end_date
    logger.info("Fetching daily returns [%s, %s]", start_date, returns_end)

    if mock_mode:
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
    logger.info("Returns panel: %d daily obs × %d tickers", *returns_panel.shape)

    # ── Step 7: Run 3 walk-forwards ──────────────────────────────────────────
    dhs_data_dir = REPO_ROOT / "data" / "path_c_dhs"
    dhs_data_dir.mkdir(parents=True, exist_ok=True)

    portfolios_data = {}
    for label, sig in [("PEAD", sig_pead), ("FIN", sig_fin), ("COMBINED", sig_comb)]:
        if sig.empty or "leg" not in sig.columns:
            logger.warning("%s skipped — empty signal panel", label)
            portfolios_data[label] = None
            continue
        sig_for_bt = sig.rename(columns={"ticker": "ticker_ibes"})
        logger.info("Walk-forward [%s]: hold=%d days", label, HOLD_TRADING_DAYS_LOCKED)
        wf = run_walk_forward_pead(
            signal_panel=sig_for_bt,
            returns_panel=returns_panel,
            window_start=start_date, window_end=end_date,
            checkpoint_run_id=f"{args.run_id}_{label.lower()}",
            spec_hash_at_run=spec_hash,
        )
        if wf.daily_returns.empty:
            logger.warning("%s walk-forward produced empty daily_returns", label)
            portfolios_data[label] = None
            continue
        parquet_path = dhs_data_dir / f"walk_forward_{label.lower()}.parquet"
        persist_walk_forward_result(wf, parquet_path=parquet_path)
        portfolios_data[label] = (wf, sig_for_bt)
        logger.info("  %s: %d quarters, %d firm-q active, %d daily obs",
                    label, wf.n_quarters_processed, wf.n_firm_quarters_active,
                    len(wf.daily_returns))

    # ── Step 8: Per-portfolio verdicts + sub-period split ───────────────────
    logger.info("Building per-portfolio verdicts + sub-period split")
    per_portfolio = {}
    subperiods    = {}

    universe_label = f"crsp_top{args.top_n}_compustat"
    for label, payload in portfolios_data.items():
        if payload is None:
            per_portfolio[label] = {
                "decision":    "FAIL",
                "sharpe_net":  None,
                "nw_t":        None,
                "reason":      "empty_walk_forward",
            }
            subperiods[label] = {}
            continue
        wf, sig_for_bt = payload
        wave_label = f"D-dhs-{label.lower()}"
        verdict = build_pead_verdict(
            wf, sig_for_bt,
            spec_hash=spec_hash, spec_path=SPEC_PATH,
            effective_n_trials=n_trials,
            wave=wave_label,
            universe_source=f"{universe_label}_{label.lower()}",
        )
        # Re-classify decision under industry-grade gates (verdict.decision may use older rule)
        decision = _classify_per_portfolio(verdict.sharpe_net, verdict.nw_t_stat)
        per_portfolio[label] = {
            "decision":              decision,
            "sharpe_gross":          float(verdict.sharpe_gross),
            "sharpe_net":            float(verdict.sharpe_net),
            "nw_t":                  float(verdict.nw_t_stat),
            "nw_lag":                int(verdict.nw_lag),
            "bootstrap_ci_lower":    float(verdict.bootstrap_ci_lower),
            "bootstrap_ci_upper":    float(verdict.bootstrap_ci_upper),
            "n_daily_obs":           int(verdict.n_daily_observations),
            "n_firm_quarters_used":  int(verdict.n_firm_quarters_used),
            "cumulative_return":     float(verdict.cumulative_return),
            "max_drawdown":          float(verdict.max_drawdown),
            "long_only_sharpe":      float(verdict.long_only_sharpe),
        }
        # Sub-period split on daily net L/S returns
        sp = _sub_period_stats(wf.daily_returns["r_long_short_net"], nw_lag=verdict.nw_lag)
        subperiods[label] = sp
        logger.info("  %s verdict: %s | Sharpe net %.3f | NW t %.3f",
                    label, decision, verdict.sharpe_net, verdict.nw_t_stat)

    # ── Step 9: Aggregate decision per spec §3.3 ─────────────────────────────
    aggregate = _classify_aggregate(per_portfolio, subperiods)
    logger.info("AGGREGATE decision: %s", aggregate)

    # ── Step 10: Persist aggregate verdict.json ──────────────────────────────
    aggregate_verdict = {
        "decision_aggregate":         aggregate,
        "decision_per_portfolio":     {k: v.get("decision", "FAIL") for k, v in per_portfolio.items()},
        "spec_hash":                  spec_hash,
        "spec_path":                  SPEC_PATH,
        "run_at":                     datetime.datetime.utcnow().isoformat() + "Z",
        "wave":                       "D-dhs-behavioral",
        "universe_source":            universe_label,
        "window_start":               args.start,
        "window_end":                 args.end,
        "top_n":                      int(args.top_n),
        "n_trials_at_verdict":        int(n_trials),
        "per_portfolio_stats":        per_portfolio,
        "subperiod_split":            subperiods,
        "mode":                       "mock" if mock_mode else "wrds",
        "honest_disclose":            [
            "FIN horizon shortened from DHS 12mo to 60d for path_c comparability",
            "Standalone signal test, NOT DHS-4 factor regression replication",
            "Post-publication decay risk: DHS 2020 published; 2020-2023 is TRUE OOS",
            "Sloan accruals default-zero handling for cheq/dlcq/txpq if NULL",
            "Top-1500 broader universe excludes microcap where behavioral biases may be stronger",
            "FIN signal effective universe excludes financial firms/REITs (Sloan limitation, not bug)",
            "ajexq retrospective split adjustment may slightly bias absolute SUE values; cross-section ranking unaffected",
        ],
    }

    verdict_path = dhs_data_dir / f"{args.run_id}_verdict.json"
    verdict_path.write_text(json.dumps(aggregate_verdict, indent=2, default=str),
                            encoding="utf-8")

    logger.info("=" * 70)
    logger.info("FINAL VERDICT: %s", aggregate)
    for label in ("PEAD", "FIN", "COMBINED"):
        pp = per_portfolio.get(label, {})
        logger.info("  %-8s: %s | Sharpe net %s | NW t %s",
                    label, pp.get("decision", "?"),
                    f"{pp.get('sharpe_net'):.3f}" if pp.get('sharpe_net') is not None else "n/a",
                    f"{pp.get('nw_t'):.3f}" if pp.get('nw_t') is not None else "n/a")
    logger.info("  Artifacts: %s", verdict_path)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
