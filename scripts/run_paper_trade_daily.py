"""
scripts/run_paper_trade_daily.py — Sprint D-2 daily forward paper-trade auto-run.

Production scheduled task (Windows Task Scheduler MacroAlphaPro_PaperTrade,
daily 06:00 SGT). Six sequential steps:

  0. Circuit-breaker pre-flight (engine.circuit_breaker.evaluate). SEVERE →
     abort with exit code 4 (signal generation halted, manual reset required).
     MEDIUM → log warning + proceed (paper-trade has no LLM calls in critical
     path, MEDIUM only constrains LLM-heavy work).
  1. Refresh SP500AnnouncementEvent feed from Wikipedia + EDGAR
     (Sprint D-1 module engine.data_sources.sp500_announcements)
  2. Run 4-component paper-trade orchestrator for today
     (engine.portfolio.paper_trade_combined.run_paper_trade_day)
  3. Persist result to PaperTradeStrategyLog table
     (engine.portfolio.paper_trade_combined.persist_run_to_db)
  4. Sprint H non-blocking — per-trade forensic attribution
  5. 2026-05-14 non-blocking — backfill daily_gross_return on today's row
     so forward window can populate VaR/CVaR / ρ Sentinel / weekly_recon
  6. Tier-1 #3 Phase B non-blocking — backfill tc_drag_today + daily_net_return
     via ADV-aware cost_model (engine.execution.cost_model)

Logs to data/paper_trade/daily_run_<date>.log and stdout. Exit codes:
  0 = full success
  1 = orchestrator ran but persistence partial failure
  2 = orchestrator failed (no DB write)
  3 = feed refresh failed (orchestrator still runs with stale feed)
  4 = circuit-breaker SEVERE — signal generation halted
  5 = Risk Manager pre-trade HARD HALT — book not persisted (spec id=69)
  6 = DQ Inspector pre-batch HARD HALT — data quality critical (spec id=70,
      only effective when --dq-enforcement-mode=enforce; default 'off' = shadow)

Watchdog can monitor data/paper_trade/daily_run_*.log files via existing
file-system rule (engine.auto_audit_rules).

2026-05-18: Risk Manager Agent v1.0 (spec id=69; current hash in
SpecRegistry — call engine.preregistration.list_specs() or
engine.agents.persona.tools.lookup_spec(69)) added two new steps to
the cycle:
  Step 2.5 PRE-TRADE GATE  — between orchestrator + persist; HARD HALT
                             → skip persist, write _HALT.json, exit 5
  Step 3.5 POST-TRADE GATE — between persist + Sprint H; soft-warn only
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOG_DIR = REPO_ROOT / "data" / "paper_trade"


def _setup_logging(as_of: datetime.date) -> Path:
    """File + stdout logging. Returns log file path for caller verification."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"daily_run_{as_of.isoformat()}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicate output on re-runs
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_file


def step_circuit_breaker_preflight(as_of: datetime.date) -> dict:
    """Step 0 (2026-05-14): evaluate engine.circuit_breaker before any work.

    SEVERE  → halt — caller returns exit 4.
    MEDIUM  → log warning, proceed (paper-trade has no critical-path LLM).
    NONE    → proceed normally.

    Returns dict {level, reason, halt}. Wired from main(); caller decides exit.
    """
    logger = logging.getLogger("step_circuit_breaker_preflight")
    try:
        from engine.circuit_breaker import (
            evaluate as _cb_evaluate, LEVEL_SEVERE, LEVEL_MEDIUM, LEVEL_NONE,
        )
    except Exception as exc:
        logger.warning("Circuit-breaker import failed (proceeding): %s", exc)
        return {"level": "import_failed", "reason": str(exc), "halt": False}

    state = _cb_evaluate(as_of)
    if state.level == LEVEL_SEVERE:
        logger.error(
            "Circuit breaker SEVERE — aborting paper-trade run. Reason: %s",
            state.reason,
        )
        logger.error(
            "Manual reset required: engine.circuit_breaker.manual_reset() "
            "or via Streamlit pages/circuit_breaker.py / pages/ops_watchdog.py."
        )
        return {"level": state.level, "reason": state.reason, "halt": True}
    if state.level == LEVEL_MEDIUM:
        logger.warning(
            "Circuit breaker MEDIUM (LLM quota): %s — paper-trade proceeds "
            "(no LLM in critical path).",
            state.reason,
        )
        return {"level": state.level, "reason": state.reason, "halt": False}
    logger.info("Circuit breaker pre-flight: NONE (clean).")
    return {"level": state.level, "reason": "", "halt": False}


def step_refresh_sp500_feed(
    lookback_days_edgar: int = 60,
) -> dict:
    """Step 1: refresh SP500AnnouncementEvent from Wikipedia + EDGAR."""
    logger = logging.getLogger("step_refresh_sp500_feed")
    from engine.data_sources.sp500_announcements import (
        fetch_wikipedia_sp500_changes,
        fetch_edgar_8k_sp500_filings,
        reconcile_announcements,
        persist_announcements,
    )

    today = datetime.date.today()
    edgar_start = today - datetime.timedelta(days=lookback_days_edgar)

    wiki = fetch_wikipedia_sp500_changes()
    logger.info("Wikipedia: %d events", len(wiki))

    edgar = fetch_edgar_8k_sp500_filings(edgar_start, today)
    logger.info("EDGAR %s to %s: %d filings", edgar_start, today, len(edgar))

    # Only reconcile recent events (last 90 days) to avoid massive churn
    cutoff = today - datetime.timedelta(days=90)
    recent_wiki = [e for e in wiki if e.effective_date >= cutoff]
    reconciled = reconcile_announcements(recent_wiki, edgar)
    logger.info("Reconciled: %d recent events", len(reconciled))

    result = persist_announcements(reconciled)
    logger.info("Persistence: %s", result)
    return result


def step_run_orchestrator(as_of: datetime.date) -> "PaperTradeRunResult":
    """Step 2: invoke 4-component orchestrator for as_of date."""
    logger = logging.getLogger("step_run_orchestrator")
    from engine.portfolio.paper_trade_combined import run_paper_trade_day

    result = run_paper_trade_day(as_of)
    logger.info(
        "Orchestrator ran: %d strategies, %d combined positions, gross=%.4f",
        len(result.signals),
        len(result.combined_portfolio),
        float(result.combined_portfolio.abs().sum()) if len(result.combined_portfolio) else 0.0,
    )
    for sig in result.signals:
        logger.info(
            "  %s [%s] sleeve=%s n_pos=%d (intra_w=%.2f)",
            sig.strategy_name, sig.status, sig.sleeve_id,
            sig.n_positions, sig.intra_sleeve_weight,
        )
    return result


def step_dq_pre_batch(
    as_of:              datetime.date,
    enforcement_mode:   str = "off",
) -> dict:
    """Step 0.5 (DQ Inspector v1.0 spec id=70, 2026-05-19): pre-batch gate.

    Runs cheap freshness checks (modes 1/2/3/4) BEFORE feed refresh.
    Behaviour governed by --dq-enforcement-mode CLI flag:
      off       — log only; alerts NOT persisted; no HALT (pure observation)
      advisory  — alerts persisted + _HALT.json marker; daily cycle proceeds
      enforce   — alerts persisted + _HALT.json + exit 6 on HARD HALT

    Phase 6b initial shadow deployment defaults to 'off' so first 1-2 weeks
    are observation only. User flips to 'advisory' then 'enforce' after
    shadow-period review.
    """
    logger = logging.getLogger("step_dq_pre_batch")
    try:
        from engine.agents.dq_inspector.orchestrator_hook import pre_batch_gate
        # dry_run=True for 'off' mode skips DB writes; 'advisory'/'enforce' persist alerts
        dry_run = (enforcement_mode == "off")
        rm = pre_batch_gate(as_of, dry_run=dry_run)
        logger.info(
            "DQ pre-batch: severity=%s halt=%s n_breaches=%d enforcement=%s",
            rm.severity, rm.halt, len(rm.breaches), enforcement_mode,
        )
        for b in rm.breaches:
            logger.warning(
                "  [DQ mode %s] %s: %s",
                b.mode_id, b.severity, b.rule_description[:120],
            )
        return {
            "halt":             rm.halt,
            "severity":         rm.severity,
            "n_breaches":       len(rm.breaches),
            "n_hard_halt":      sum(1 for b in rm.breaches if b.severity == "HARD_HALT"),
            "enforcement_mode": enforcement_mode,
        }
    except Exception as exc:
        logger.exception("Step 0.5 (DQ pre-batch) failed (non-blocking): %s", exc)
        return {"error": str(exc), "halt": False}


def step_dq_post_feed(
    as_of:              datetime.date,
    enforcement_mode:   str = "off",
) -> dict:
    """Step 1.5 (DQ Inspector v1.0 spec id=70, Phase 6c 2026-05-19):
    post-feed gate after feed refresh but BEFORE orchestrator runs.

    Wires modes 5/6/7/9: K1 ETF universe coverage + D-PEAD panel coverage
    + class-aware price anomaly + NaN burst. Inputs are gathered by
    engine.agents.dq_inspector.post_feed_inputs.gather_post_feed_inputs
    (yfinance batch on K1+AC ~47 tickers + PEAD panel parquet row count).

    Mode 7 anomaly check scoped to K1 + AC active universe; D-PEAD per-
    stock anomaly is downstream of D-PEAD's own validator and not
    re-evaluated here.

    Enforcement semantics mirror step_dq_pre_batch:
      off       — log only; alerts NOT persisted; no HALT
      advisory  — alerts persisted; daily cycle proceeds
      enforce   — alerts persisted; exit 6 on HARD HALT
    """
    logger = logging.getLogger("step_dq_post_feed")
    try:
        from engine.agents.dq_inspector.orchestrator_hook import post_feed_gate
        from engine.agents.dq_inspector.post_feed_inputs import (
            gather_post_feed_inputs,
        )

        inputs = gather_post_feed_inputs(as_of)
        dry_run = (enforcement_mode == "off")
        rm = post_feed_gate(
            as_of              = inputs["as_of"],
            k1_n_with_price    = inputs["k1_n_with_price"],
            pead_n_with_rdq    = inputs["pead_n_with_rdq"],
            daily_returns      = inputs["daily_returns"],
            ticker_to_sleeves  = inputs["ticker_to_sleeves"],
            n_nan_close        = inputs["n_nan_close"],
            n_universe         = inputs["n_universe"],
            dry_run            = dry_run,
        )
        logger.info(
            "DQ post-feed: severity=%s halt=%s n_breaches=%d "
            "(k1_priced=%d/43 pead_rdq=%d nan_close=%d) enforcement=%s",
            rm.severity, rm.halt, len(rm.breaches),
            inputs["k1_n_with_price"], inputs["pead_n_with_rdq"],
            inputs["n_nan_close"], enforcement_mode,
        )
        for b in rm.breaches:
            logger.warning(
                "  [DQ mode %s] %s: %s",
                b.mode_id, b.severity, b.rule_description[:120],
            )
        return {
            "halt":             rm.halt,
            "severity":         rm.severity,
            "n_breaches":       len(rm.breaches),
            "n_hard_halt":      sum(1 for b in rm.breaches if b.severity == "HARD_HALT"),
            "enforcement_mode": enforcement_mode,
            "k1_n_with_price":  inputs["k1_n_with_price"],
            "pead_n_with_rdq":  inputs["pead_n_with_rdq"],
        }
    except Exception as exc:
        logger.exception("Step 1.5 (DQ post-feed) failed (non-blocking): %s", exc)
        return {"error": str(exc), "halt": False}


def step_dq_post_batch(
    as_of:              datetime.date,
    enforcement_mode:   str = "off",
) -> dict:
    """Step 3.6 (DQ Inspector v1.0): post-batch gate.

    Row-count regression check (modes 10a/10b) against yesterday's
    PaperTradeStrategyLog row count. Non-blocking — book already persisted
    by this point. HARD HALT here escalates legacy CB SEVERE so tomorrow's
    run requires manual_reset (Mode 10b only).

    Mode 8 volume dropoff WIRING DEFERRED to Phase 6c (needs active-universe
    volume fetch).
    """
    logger = logging.getLogger("step_dq_post_batch")
    try:
        from engine.agents.dq_inspector.orchestrator_hook import post_batch_gate
        from engine.db_models import PaperTradeStrategyLog
        from engine.memory import SessionFactory

        # Query yesterday + today row counts
        yesterday = as_of - datetime.timedelta(days=1)
        sess = SessionFactory()
        try:
            today_rows = (
                sess.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.date == as_of)
                    .count()
            )
            yesterday_rows = (
                sess.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.date == yesterday)
                    .count()
            )
        finally:
            sess.close()

        # If yesterday had no rows (e.g. weekend, first run), skip regression check
        if yesterday_rows == 0:
            logger.info("DQ post-batch: no yesterday rows; row-count check skipped")
            return {"skipped": True, "reason": "no_yesterday_rows"}

        dry_run = (enforcement_mode == "off")
        rm = post_batch_gate(
            as_of, today_rows=today_rows, yesterday_rows=yesterday_rows,
            dry_run=dry_run,
        )
        logger.info(
            "DQ post-batch: severity=%s halt=%s n_breaches=%d (today=%d / yesterday=%d) enforcement=%s",
            rm.severity, rm.halt, len(rm.breaches), today_rows, yesterday_rows,
            enforcement_mode,
        )
        for b in rm.breaches:
            logger.warning(
                "  [DQ mode %s] %s: %s",
                b.mode_id, b.severity, b.rule_description[:120],
            )
        return {
            "halt":             rm.halt,
            "severity":         rm.severity,
            "n_breaches":       len(rm.breaches),
            "today_rows":       today_rows,
            "yesterday_rows":   yesterday_rows,
            "enforcement_mode": enforcement_mode,
        }
    except Exception as exc:
        logger.exception("Step 3.6 (DQ post-batch) failed (non-blocking): %s", exc)
        return {"error": str(exc), "halt": False}


def step_risk_manager_pre_trade(result) -> dict:
    """Step 2.5 (Risk Manager v1.0 spec id=69, 2026-05-18): pre-trade gate.

    Runs 12 deterministic risk gates against today's combined book BEFORE
    persistence. HARD HALT (modes 1/3/4/5/6b/7b/9) blocks subsequent
    persist_run_to_db; caller exits 5 + writes _HALT.json.

    Returns dict with {halt, severity, n_breaches, n_hard_halt}. Caller
    decides exit behavior.
    """
    logger = logging.getLogger("step_risk_manager_pre_trade")
    from engine.agents.risk_manager.orchestrator_hook import pre_trade_gate
    rm_result = pre_trade_gate(result, compute_var=True, dry_run=False)
    logger.info(
        "Risk Manager pre-trade: severity=%s halt=%s n_breaches=%d (%d HARD_HALT)",
        rm_result.severity, rm_result.halt, len(rm_result.breaches),
        sum(1 for b in rm_result.breaches if b.severity == "HARD_HALT"),
    )
    for b in rm_result.breaches:
        logger.warning(
            "  [%s] %s: %s",
            b.mode_id, b.severity, b.rule_description[:120],
        )
    return {
        "halt":         rm_result.halt,
        "severity":     rm_result.severity,
        "n_breaches":   len(rm_result.breaches),
        "n_hard_halt":  sum(1 for b in rm_result.breaches if b.severity == "HARD_HALT"),
    }


def step_persist_to_db(result) -> dict:
    """Step 3: write PaperTradeRunResult to PaperTradeStrategyLog table."""
    logger = logging.getLogger("step_persist_to_db")
    from engine.portfolio.paper_trade_combined import persist_run_to_db
    counts = persist_run_to_db(result)
    logger.info("DB persistence: %s", counts)
    return counts


def step_risk_manager_post_trade(result) -> dict:
    """Step 3.5 (Risk Manager v1.0 spec id=69, 2026-05-18): post-trade gate.

    Re-runs the 12 gates against the now-persisted state. NEVER halts
    (book already saved); HARD HALT breaches are logged for investigation
    but do not roll back persistence.

    Non-blocking: any exception here is logged + swallowed so it doesn't
    abort the daily run.
    """
    logger = logging.getLogger("step_risk_manager_post_trade")
    try:
        from engine.agents.risk_manager.orchestrator_hook import post_trade_gate
        rm_result = post_trade_gate(result, compute_var=True, dry_run=False)
        logger.info(
            "Risk Manager post-trade: severity=%s n_breaches=%d (%d HARD_HALT to investigate)",
            rm_result.severity, len(rm_result.breaches),
            sum(1 for b in rm_result.breaches if b.severity == "HARD_HALT"),
        )
        return {
            "severity":     rm_result.severity,
            "n_breaches":   len(rm_result.breaches),
            "n_hard_halt":  sum(1 for b in rm_result.breaches if b.severity == "HARD_HALT"),
        }
    except Exception as exc:
        logger.exception("Step 3.5 (Risk Manager post-trade) failed (non-blocking): %s", exc)
        return {"error": str(exc)}


def step_fill_daily_tc(as_of: datetime.date) -> dict:
    """Step 6 (Tier-1 #3 Phase B, 2026-05-14): backfill tc_drag_today +
    daily_net_return on today's rows.

    Reads today vs prior positions per strategy, computes turnover,
    batch-fetches ADV + vol via yfinance, calls
    engine.execution.cost_model.compute_portfolio_tc, persists. Capacity
    warnings (size/ADV > 20%) tallied for log.

    Non-blocking — failure here does NOT fail the daily run.
    """
    logger = logging.getLogger("step_fill_daily_tc")
    from engine.portfolio.paper_trade_combined import fill_daily_tc
    counts = fill_daily_tc(as_of)
    logger.info("Daily TC backfill: %s", counts)
    return counts


def step_fill_daily_returns(as_of: datetime.date) -> dict:
    """Step 5 (2026-05-14): backfill `daily_gross_return` on today's rows.

    Reads each strategy's prior row, prices its positions close-to-close via
    yfinance, writes weighted return to today's row. Required so forward
    paper-trade window can populate VaR/CVaR overlay, ρ Sentinel, weekly
    reconciliation, Sharpe deviation alerts.

    Non-blocking — failure here does NOT fail the daily run.
    """
    logger = logging.getLogger("step_fill_daily_returns")
    from engine.portfolio.paper_trade_combined import fill_daily_returns
    counts = fill_daily_returns(as_of)
    logger.info("Daily return backfill: %s", counts)
    return counts


def step_roll_daily_nav(as_of: datetime.date) -> dict:
    """Step 5.5 (2026-06-02): persist PortfolioNavSnapshot for as_of.

    Restores the daily NAV rollup that used to be triggered manually via
    the Streamlit pages/performance_report.py page (archived 2026-05-14).
    Without this step the NAV chart / liveness data_freshness probe
    drift days behind reality — the silent failure the user surfaced on
    2026-06-02 (NAV stuck at 5/12 while heartbeat read OK).

    roll_daily_nav() is idempotent: re-runs on the same date return the
    existing snapshot. Non-blocking — failure here does NOT fail the
    daily run; the data_freshness probe will surface persistent staleness
    on the next liveness check.
    """
    logger = logging.getLogger("step_roll_daily_nav")
    try:
        from engine.portfolio_returns import roll_daily_nav
        snap = roll_daily_nav(as_of)
        logger.info(
            "NAV rollup ok: nav_open=%s nav_close=%s daily_md=%s",
            snap.get("nav_open"), snap.get("nav_close"),
            snap.get("daily_modified_dietz"),
        )
        return {"ok": True, "nav_close": snap.get("nav_close")}
    except Exception as exc:
        logger.exception("step_roll_daily_nav failed (non-fatal): %s", exc)
        return {"ok": False, "error": str(exc)}


def step_persist_attribution(result, as_of: datetime.date) -> dict:
    """Step 4 (Sprint H): write per-trade attribution log to PaperTradeTradeLog + JSONL.

    Non-blocking: failures here do NOT fail the daily run. Sprint H is forensic
    infrastructure — Step 3 (PaperTradeStrategyLog daily aggregate) is the
    authoritative critical-path persist.
    """
    logger = logging.getLogger("step_persist_attribution")
    from engine.portfolio.paper_trade_combined import (
        is_rebalance_day_k1, is_rebalance_day_d_pead,
        is_rebalance_day_path_n, is_rebalance_day_cta,
    )
    from engine.portfolio.attribution_logger import (
        attributions_from_result, persist_attribution_to_db,
        persist_attribution_to_jsonl,
    )

    is_rebal = {
        "K1_BAB":    is_rebalance_day_k1(as_of),
        "D_PEAD":    is_rebalance_day_d_pead(as_of),
        "PATH_N":    is_rebalance_day_path_n(as_of),
        "CTA_PQTIX": is_rebalance_day_cta(as_of),
    }
    rows = attributions_from_result(result, is_rebal)
    n_db    = persist_attribution_to_db(rows)
    n_jsonl = persist_attribution_to_jsonl(rows)
    logger.info("Sprint H attribution persisted: %d rows to DB, %d to JSONL", n_db, n_jsonl)
    return {"db_rows": n_db, "jsonl_rows": n_jsonl, "is_rebalance_per_strategy": is_rebal}


def _emit_heartbeat(
    *,
    as_of:           datetime.date,
    exit_code:       int,
    log_file:        Path,
    halted_at_step:  Optional[str] = None,
    run_result:      "Optional[object]" = None,
    errors:          Optional[list[str]] = None,
) -> None:
    """P0 liveness layer (2026-06-02): persist a heartbeat row at every
    daily-run exit path. Best-effort — any failure here is logged and
    swallowed so it CANNOT mask the real exit code or crash the cron.

    P1 augmentation (2026-06-02): on SUCCESS exits, also pull
    broker_reconciliation (intended vs broker vs live positions) and
    nav_anomaly (z-score of today's NAV move). These are non-blocking
    BEFORE the heartbeat write, so a stalled Alpaca API never delays
    the heartbeat row itself.

    Called from main()'s finally block so it fires on:
      * normal SUCCESS path
      * every early-return halt (CB, DQ, Risk, orchestrator-fail)
      * unhandled exceptions (re-raised after recording)
    """
    try:
        from engine.research.liveness_heartbeat import record_run
        n_strategies: Optional[int] = None
        gross_weight: Optional[float] = None
        if run_result is not None:
            try:
                n_strategies = len(run_result.signals)
            except Exception:
                pass
            try:
                cp = run_result.combined_portfolio
                gross_weight = float(cp.abs().sum()) if len(cp) else 0.0
            except Exception:
                pass

        # P1a broker reconciliation — only meaningful on SUCCESS-ish exits
        broker_echo: Optional[dict] = None
        if exit_code == 0:
            try:
                from engine.research.broker_reconciliation import reconcile
                broker_echo = reconcile(as_of)
            except Exception:
                logging.getLogger("broker_reconciliation").exception(
                    "broker reconciliation failed (non-fatal)"
                )

        # P0c data freshness (2026-06-02): probe critical data sources
        # for staleness. Catches "cron ran but data didn't update" silent
        # failures. ALWAYS runs (even on halt exits) so a downstream
        # writer breakage shows up as soon as the next heartbeat fires.
        data_freshness_summary: Optional[dict] = None
        data_sources_list: Optional[list] = None
        try:
            from engine.research import data_freshness as _DF
            sources = _DF.check_sources()
            data_sources_list = sources
            data_freshness_summary = _DF.summarize(sources)
        except Exception:
            logging.getLogger("data_freshness").exception(
                "data freshness probe failed (non-fatal)"
            )

        # P1b NAV anomaly — record today's NAV (prefer post-trade equity
        # from broker_echo.live; fall back to pre-trade equity_before)
        nav_anomaly_verdict: Optional[dict] = None
        nav_equity: Optional[float] = None
        if isinstance(broker_echo, dict):
            live = broker_echo.get("live") if isinstance(broker_echo, dict) else None
            if isinstance(live, dict) and live.get("equity") is not None:
                nav_equity = float(live["equity"])
            elif broker_echo.get("equity_before") is not None:
                nav_equity = float(broker_echo["equity_before"])
        if nav_equity is not None:
            try:
                from engine.research.nav_anomaly import record_nav
                nav_anomaly_verdict = record_nav(as_of=as_of, equity=nav_equity)
            except Exception:
                logging.getLogger("nav_anomaly").exception(
                    "nav anomaly check failed (non-fatal)"
                )

        # Extract n_orders / n_fills / equity / broker from broker_echo
        # if present — these were stubbed None when only run_result was
        # available; broker_echo gives them authoritatively.
        n_orders     = (broker_echo or {}).get("n_orders_submitted")
        n_fills      = (broker_echo or {}).get("n_fills")
        equity_before= (broker_echo or {}).get("equity_before")
        broker_ack   = (broker_echo or {}).get("broker_ack")

        record_run(
            as_of=as_of,
            exit_code=exit_code,
            n_orders=n_orders,
            n_fills=n_fills,
            equity_before=equity_before,
            n_strategies=n_strategies,
            gross_weight=gross_weight,
            halted_at_step=halted_at_step,
            broker_ack=broker_ack,
            log_file=log_file,
            errors=list(errors or []),
            broker_echo=broker_echo,
            nav_anomaly=nav_anomaly_verdict,
            data_freshness=data_freshness_summary,
            data_sources=data_sources_list,
        )
    except Exception:
        logging.getLogger("liveness_heartbeat").exception(
            "heartbeat emit failed (non-fatal)"
        )


def _main_inner(*, hb_state: dict) -> int:
    """Original main body. Mutates hb_state['halted_at_step'],
    ['run_result'], ['errors'] so the outer main() finally block can
    record an accurate heartbeat row regardless of which exit path
    triggered."""
    parser = argparse.ArgumentParser(
        description="Sprint D-2 daily forward paper-trade auto-run"
    )
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Run date YYYY-MM-DD (default: today UTC date)",
    )
    parser.add_argument(
        "--skip-feed-refresh", action="store_true",
        help="Skip Step 1 (Wikipedia/EDGAR refresh); use existing DB state only",
    )
    parser.add_argument(
        "--ignore-circuit-breaker", action="store_true",
        help="Emergency override: proceed even if Step 0 reports SEVERE. "
             "Use only when CB state is known stale and manual reset is in progress.",
    )
    parser.add_argument(
        "--dq-enforcement-mode",
        choices=["off", "advisory", "enforce"], default="off",
        help="DQ Inspector v1.0 (spec id=70) enforcement level. "
             "off (default 2026-05-19 initial deployment) = shadow mode, "
             "log alerts but no HALT + no DB writes. "
             "advisory = persist alerts + _HALT.json marker, daily cycle proceeds. "
             "enforce = exit 6 on pre-batch HARD HALT. "
             "Promote off→advisory→enforce after 1-2 weeks shadow observation per "
             "project_dq_inspector_shadow_phase_2026-05-19 protocol.",
    )
    args = parser.parse_args()

    if args.as_of:
        as_of = datetime.date.fromisoformat(args.as_of)
    else:
        as_of = datetime.datetime.utcnow().date()

    log_file = _setup_logging(as_of)
    logger = logging.getLogger("run_paper_trade_daily")
    logger.info("=== Daily paper-trade run start: as_of=%s ===", as_of)
    logger.info("Log file: %s", log_file)

    # Heartbeat state — populated as we progress, read by outer main()
    # finally block. Must be populated BEFORE any return.
    hb_state["as_of"]    = as_of
    hb_state["log_file"] = log_file

    # Step 0: circuit-breaker pre-flight
    cb_result = step_circuit_breaker_preflight(as_of)
    if cb_result.get("halt"):
        if args.ignore_circuit_breaker:
            logger.warning(
                "Circuit breaker SEVERE override active (--ignore-circuit-breaker); "
                "proceeding despite halt flag."
            )
        else:
            logger.error("=== Daily paper-trade run HALTED (circuit breaker SEVERE) ===")
            hb_state["halted_at_step"] = "step_circuit_breaker_preflight"
            return 4

    # Step 0.5 (DQ Inspector v1.0 spec id=70, 2026-05-19): pre-batch gate
    # Default --dq-enforcement-mode=off → shadow observation only (no HALT).
    # Per project_dq_inspector_shadow_phase_2026-05-19 protocol.
    dq_pre = step_dq_pre_batch(as_of, enforcement_mode=args.dq_enforcement_mode)
    if dq_pre.get("halt") and args.dq_enforcement_mode == "enforce":
        logger.error(
            "=== Daily paper-trade run HALTED (DQ Inspector pre-batch: %d HARD_HALT modes) ===",
            dq_pre.get("n_hard_halt", 0),
        )
        hb_state["halted_at_step"] = "step_dq_pre_batch"
        return 6

    feed_result = None
    if not args.skip_feed_refresh:
        try:
            feed_result = step_refresh_sp500_feed()
        except Exception as exc:
            logger.exception("Step 1 (feed refresh) failed: %s", exc)
            # Continue anyway — orchestrator can run with stale feed
            feed_result = {"error": str(exc)}
    else:
        logger.info("Step 1 (feed refresh) skipped per --skip-feed-refresh")

    # Step 1.5 (DQ Inspector v1.0 spec id=70, Phase 6c 2026-05-19): post-feed gate
    # Modes 5/6/7/9 — universe coverage + price anomaly + NaN burst.
    # Default --dq-enforcement-mode=off → shadow observation only.
    dq_post_feed = step_dq_post_feed(as_of, enforcement_mode=args.dq_enforcement_mode)
    if dq_post_feed.get("halt") and args.dq_enforcement_mode == "enforce":
        logger.error(
            "=== Daily paper-trade run HALTED (DQ Inspector post-feed: %d HARD_HALT modes) ===",
            dq_post_feed.get("n_hard_halt", 0),
        )
        hb_state["halted_at_step"] = "step_dq_post_feed"
        return 6

    try:
        run_result = step_run_orchestrator(as_of)
        hb_state["run_result"] = run_result
    except Exception as exc:
        logger.exception("Step 2 (orchestrator) failed: %s", exc)
        logger.error("=== Daily paper-trade run FAILED (orchestrator) ===")
        hb_state["halted_at_step"] = "step_run_orchestrator"
        hb_state["errors"].append(f"orchestrator: {exc}")
        return 2

    # Step 2.5 (Risk Manager v1.0 spec id=69, 2026-05-18): pre-trade gate
    # HARD HALT blocks persistence. If the gate itself errors (e.g. registry
    # not loaded), log + proceed — pre-trade gate is not the source of truth
    # for halt; legacy CB at Step 0 handles VIX-side halt independently.
    try:
        rm_pre = step_risk_manager_pre_trade(run_result)
        if rm_pre.get("halt"):
            logger.error(
                "=== Daily paper-trade run HALTED (Risk Manager pre-trade: %d HARD_HALT mode breaches) ===",
                rm_pre.get("n_hard_halt", 0),
            )
            hb_state["halted_at_step"] = "step_risk_manager_pre_trade"
            return 5
    except Exception as exc:
        logger.exception("Step 2.5 (Risk Manager pre-trade) failed (non-blocking, proceeding): %s", exc)

    try:
        persist_counts = step_persist_to_db(run_result)
    except Exception as exc:
        logger.exception("Step 3 (persistence) failed: %s", exc)
        logger.error("=== Daily paper-trade run PARTIAL (DB write failed) ===")
        hb_state["halted_at_step"] = "step_persist_to_db"
        hb_state["errors"].append(f"persist: {exc}")
        return 1

    if persist_counts.get("errors", 0) > 0:
        logger.error("=== Daily paper-trade run PARTIAL (DB errors) ===")
        hb_state["halted_at_step"] = "step_persist_to_db"
        hb_state["errors"].append(
            f"persist_counts.errors={persist_counts.get('errors')}"
        )
        return 1

    # Step 3.5 (Risk Manager v1.0 spec id=69, 2026-05-18): post-trade gate
    # Non-blocking soft-warn check on the persisted state.
    step_risk_manager_post_trade(run_result)

    # Step 3.6 (DQ Inspector v1.0 spec id=70, 2026-05-19): post-batch gate
    # Row-count regression check (mode 10a/10b). Non-blocking by design —
    # book already persisted; 10b escalates legacy CB SEVERE for tomorrow.
    step_dq_post_batch(as_of, enforcement_mode=args.dq_enforcement_mode)

    # Step 4 (Sprint H, non-blocking): per-trade forensic attribution
    try:
        step_persist_attribution(run_result, as_of)
    except Exception as exc:
        logger.exception("Step 4 (Sprint H attribution) failed (non-blocking): %s", exc)
        # Continue — Sprint H failure does not invalidate Step 3 critical path

    # Step 5 (2026-05-14, non-blocking): backfill daily_gross_return on today's row
    try:
        step_fill_daily_returns(as_of)
    except Exception as exc:
        logger.exception("Step 5 (daily return backfill) failed (non-blocking): %s", exc)
        # Continue — backfill is recoverable next run; not critical for signal generation

    # Step 5.5 (2026-06-02, non-blocking): persist PortfolioNavSnapshot.
    # Restores the daily NAV rollup the archived Streamlit page used to
    # trigger (see scripts/run_paper_trade_daily.step_roll_daily_nav docs).
    # Idempotent — safe to re-run for the same as_of.
    try:
        step_roll_daily_nav(as_of)
    except Exception as exc:
        logger.exception("Step 5.5 (NAV rollup) failed (non-blocking): %s", exc)

    # Step 6 (Tier-1 #3 Phase B, 2026-05-14, non-blocking): backfill tc_drag_today
    # + daily_net_return via ADV-aware cost model. Must run AFTER Step 5
    # (depends on daily_gross_return to compute net).
    try:
        step_fill_daily_tc(as_of)
    except Exception as exc:
        logger.exception("Step 6 (daily TC backfill) failed (non-blocking): %s", exc)
        # Continue — TC backfill is recoverable next run

    # Step 7 (Sub-phase A1+A3, 2026-05-15 evening, non-blocking): build UI
    # artifact JSON for fast page-render pattern. Per "lambda architecture":
    # daily batch pre-computes all UI state → pages read static artifact
    # instead of live DB on every Streamlit rerun. Must run AFTER Steps 1-6
    # so all DB writes are committed before artifact dump.
    try:
        from engine.portfolio.build_ui_artifact import build_ui_artifact
        artifact_path = build_ui_artifact(as_of_date=as_of)
        logger.info(
            "Step 7 (UI artifact build) ok: %s (%d KB)",
            artifact_path.name, artifact_path.stat().st_size // 1024,
        )
    except Exception as exc:
        logger.exception("Step 7 (UI artifact build) failed (non-blocking): %s", exc)
        # Continue — pages will fall back to live DB if artifact missing

    # Feed-refresh failure is non-blocking; flag but continue success exit
    if feed_result and feed_result.get("error"):
        logger.warning("Feed refresh had errors (non-blocking)")
        return 3

    logger.info("=== Daily paper-trade run SUCCESS ===")
    return 0


def main() -> int:
    """Outer wrapper: drives _main_inner, then writes a liveness
    heartbeat in a finally block so the heartbeat fires on every
    exit path — including unhandled exceptions and signal kills."""
    hb_state: dict = {
        "as_of":          None,
        "log_file":       None,
        "run_result":     None,
        "halted_at_step": None,
        "errors":         [],
    }
    exit_code = 99
    try:
        exit_code = _main_inner(hb_state=hb_state)
        return exit_code
    except BaseException as exc:
        # Including SystemExit / KeyboardInterrupt — record then re-raise
        hb_state["errors"].append(f"unhandled: {type(exc).__name__}: {exc}")
        hb_state["halted_at_step"] = hb_state.get("halted_at_step") or "unhandled_exception"
        raise
    finally:
        _emit_heartbeat(
            as_of=hb_state.get("as_of") or datetime.datetime.utcnow().date(),
            exit_code=exit_code,
            log_file=hb_state.get("log_file"),
            halted_at_step=hb_state.get("halted_at_step"),
            run_result=hb_state.get("run_result"),
            errors=hb_state.get("errors"),
        )


if __name__ == "__main__":
    sys.exit(main())
