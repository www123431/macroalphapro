"""
engine/portfolio/weekly_recon.py — Sprint D-3 weekly reconciliation report.

Reads recent PaperTradeStrategyLog rows (populated by Sprint D-2 daily auto-run),
computes rolling forward statistics, and compares to backtest expectation.

Alert conditions (signal-level, NOT alpha-level — strategy hasn't decayed,
infrastructure has degraded):
  - Any strategy has been NO_SIGNAL for >3 consecutive days (data dependency broken)
  - Any strategy has been ERROR for >1 day (code-level failure)
  - Daily rebalance flag pattern diverges from expected cadence
  - PaperTradeStrategyLog has gap > 2 days (Task Scheduler missed runs)

Forward-vs-backtest comparison (when we have enough data):
  - Realized rolling 21-day Sharpe vs backtest forward expectation
  - Pairwise correlation drift
  - Crisis-window behavior (if any crisis windows fall within report period)

Output: data/portfolio_replay/weekly_recon_<YYYY-MM-DD>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Forward expectation per deployment_design.md §6 (Sprint B replay verdict).
# Contains the 4 strategies present in Sprint B 2014-2023 in-sample replay.
# AC_TLT_GLD (Tier-3 approved 2026-05-15) is INSURANCE class — Sharpe ~0 by
# design (G6 alpha gate FAIL by spec, G7 DD-reduction PASS). Listed here as
# None so the recon summary records "no expectation set" rather than 0/0,
# preserving the alpha vs insurance class distinction.
FORWARD_EXPECTATIONS = {
    "K1_BAB":     {"sharpe": 0.55, "vol": 0.08},
    "D_PEAD":     {"sharpe": 0.50, "vol": 0.11},
    "PATH_N":     {"sharpe": 0.60, "vol": 0.18},
    "CTA_PQTIX":  {"sharpe": 0.43, "vol": 0.10},
    "AC_TLT_GLD": {"sharpe": None, "vol": None},   # insurance class — no alpha expectation
}

ALERT_NO_SIGNAL_THRESHOLD_DAYS = 3   # NO_SIGNAL for >3 consecutive days = data dependency broken
ALERT_ERROR_THRESHOLD_DAYS     = 1   # ERROR for >1 day = code-level failure
ALERT_DATA_GAP_THRESHOLD_DAYS  = 2   # > 2 calendar days with no rows = Task Scheduler missed runs


def _get_expected_strategies() -> list[str]:
    """Strategies to monitor for streak / data-gap alerts — sourced from
    the registry (engine.strategies) so a new strategy added in adapters.py
    is automatically alerted on without hand-editing this file."""
    from engine.strategies import get_registry
    return list(get_registry().names())


@dataclass
class ReconAlert:
    """One reconciliation alert."""
    severity:   str        # 'INFO' | 'WARN' | 'CRITICAL'
    category:   str        # 'data_gap' / 'no_signal_streak' / 'error_streak' / 'sharpe_deviation'
    strategy:   Optional[str]
    message:    str
    details:    dict = field(default_factory=dict)


@dataclass
class WeeklyReconReport:
    """Output of weekly reconciliation run."""
    report_date:        datetime.date
    window_start:       datetime.date
    window_end:         datetime.date
    n_days_with_data:   int
    n_days_in_window:   int
    per_strategy_summary: dict          # {strategy: {status_count: {OK:n, NO_SIGNAL:n, ERROR:n}, current_streak: {...}}}
    alerts:             list[ReconAlert]
    notes:              list[str]


def _load_recent_paper_trade_log(
    report_date: datetime.date,
    lookback_days: int,
):
    """Load PaperTradeStrategyLog rows for [report_date - lookback, report_date]."""
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    init_db()
    start = report_date - datetime.timedelta(days=lookback_days)
    sess = SessionFactory()
    try:
        rows = (
            sess.query(PaperTradeStrategyLog)
                .filter(PaperTradeStrategyLog.date >= start)
                .filter(PaperTradeStrategyLog.date <= report_date)
                .order_by(PaperTradeStrategyLog.date.asc(), PaperTradeStrategyLog.strategy_name.asc())
                .all()
        )
        return [
            {
                "date": r.date,
                "strategy_name": r.strategy_name,
                "sleeve_id": r.sleeve_id,
                "status": r.status,
                "is_rebalance_day": r.is_rebalance_day,
                "n_positions": r.n_positions,
                "daily_gross_return": r.daily_gross_return,
                "daily_net_return": r.daily_net_return,
                "tc_drag_today": r.tc_drag_today,
            }
            for r in rows
        ]
    finally:
        sess.close()


def _detect_alerts(
    rows: list[dict],
    window_start: datetime.date,
    window_end: datetime.date,
) -> list[ReconAlert]:
    """Run alert detection rules on the loaded data."""
    alerts: list[ReconAlert] = []

    if not rows:
        alerts.append(ReconAlert(
            severity="CRITICAL", category="data_gap", strategy=None,
            message=f"NO ROWS in PaperTradeStrategyLog for window {window_start} to {window_end} "
                    f"({(window_end - window_start).days + 1} days)",
            details={"window_start": str(window_start), "window_end": str(window_end)},
        ))
        return alerts

    # Group by date to find calendar gaps
    df = pd.DataFrame(rows)
    days_with_data = sorted(df["date"].unique())

    # Gap detection: find missing days between window_start and window_end (NYSE-approximate)
    expected_days = pd.bdate_range(window_start, window_end).date.tolist()
    missing_days = [d for d in expected_days if d not in set(days_with_data)]
    if missing_days:
        alerts.append(ReconAlert(
            severity="WARN" if len(missing_days) <= ALERT_DATA_GAP_THRESHOLD_DAYS else "CRITICAL",
            category="data_gap",
            strategy=None,
            message=f"{len(missing_days)} business days missing data in window: "
                    f"{missing_days[:5]}{'...' if len(missing_days) > 5 else ''}",
            details={"missing_days": [str(d) for d in missing_days]},
        ))

    # Per-strategy streak detection
    for strategy in _get_expected_strategies():
        strat_rows = df[df["strategy_name"] == strategy].copy().sort_values("date")
        if strat_rows.empty:
            alerts.append(ReconAlert(
                severity="CRITICAL", category="missing_strategy", strategy=strategy,
                message=f"Strategy {strategy} has NO rows in window {window_start} to {window_end}",
                details={},
            ))
            continue

        # Trailing-end streaks
        statuses = strat_rows["status"].tolist()
        trailing_status = statuses[-1]
        streak_len = 1
        for s in reversed(statuses[:-1]):
            if s == trailing_status:
                streak_len += 1
            else:
                break

        if trailing_status == "NO_SIGNAL" and streak_len > ALERT_NO_SIGNAL_THRESHOLD_DAYS:
            alerts.append(ReconAlert(
                severity="WARN", category="no_signal_streak", strategy=strategy,
                message=f"{strategy}: NO_SIGNAL for last {streak_len} days "
                        f"(threshold {ALERT_NO_SIGNAL_THRESHOLD_DAYS}). "
                        f"Data dependency may be broken (cache stale, source down).",
                details={"streak_len": streak_len},
            ))
        elif trailing_status == "ERROR" and streak_len > ALERT_ERROR_THRESHOLD_DAYS:
            alerts.append(ReconAlert(
                severity="CRITICAL", category="error_streak", strategy=strategy,
                message=f"{strategy}: ERROR for last {streak_len} days "
                        f"(threshold {ALERT_ERROR_THRESHOLD_DAYS}). "
                        f"Code-level failure; investigate immediately.",
                details={"streak_len": streak_len},
            ))

    return alerts


def _per_strategy_summary(rows: list[dict]) -> dict:
    """Aggregate status counts + most-recent state per strategy."""
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["strategy_name", "status", "date", "is_rebalance_day", "n_positions"]
    )
    out: dict = {}
    for strategy in _get_expected_strategies():
        strat = df[df["strategy_name"] == strategy] if not df.empty else df
        # Safe .get() — strategy may have no FORWARD_EXPECTATIONS entry
        # (e.g. insurance class) → forward_sharpe = None.
        _expect = FORWARD_EXPECTATIONS.get(strategy, {"sharpe": None, "vol": None})
        if strat.empty:
            out[strategy] = {
                "n_rows": 0,
                "status_counts": {},
                "latest_status": None,
                "latest_date": None,
                "n_rebalance_events": 0,
                "expected_forward_sharpe": _expect["sharpe"],
            }
            continue
        status_counts = strat["status"].value_counts().to_dict()
        latest = strat.sort_values("date").iloc[-1]
        out[strategy] = {
            "n_rows": int(len(strat)),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "latest_status": str(latest["status"]),
            "latest_date": str(latest["date"]),
            "n_rebalance_events": int(strat["is_rebalance_day"].sum()),
            "expected_forward_sharpe": _expect["sharpe"],
        }
    return out


def run_weekly_recon(
    report_date:    datetime.date,
    lookback_days:  int = 7,
) -> WeeklyReconReport:
    """Run weekly reconciliation for [report_date - lookback, report_date]."""
    window_start = report_date - datetime.timedelta(days=lookback_days)
    window_end   = report_date

    rows = _load_recent_paper_trade_log(report_date, lookback_days)
    per_strat = _per_strategy_summary(rows)
    alerts = _detect_alerts(rows, window_start, window_end)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    n_days_with_data = int(df["date"].nunique()) if not df.empty else 0
    n_days_in_window = (window_end - window_start).days + 1

    notes = [
        "Window is calendar days; NYSE business days expected = ~5/week.",
        "Forward expectation Sharpe values are from deployment_design.md §6 (Sprint B replay verdict).",
        "Sprint D-3 v1: detection alerts only (data_gap / streak detection).",
        "Sprint E v2 will add Sharpe-vs-expectation deviation alerts once forward data accumulated (>60 days).",
    ]

    return WeeklyReconReport(
        report_date         = report_date,
        window_start        = window_start,
        window_end          = window_end,
        n_days_with_data    = n_days_with_data,
        n_days_in_window    = n_days_in_window,
        per_strategy_summary = per_strat,
        alerts              = alerts,
        notes               = notes,
    )


def save_recon_report(report: WeeklyReconReport, save_path: Path) -> dict:
    """Save weekly recon report to JSON."""
    payload = {
        "report_date":      report.report_date.isoformat(),
        "window":           f"{report.window_start} to {report.window_end}",
        "n_days_with_data": report.n_days_with_data,
        "n_days_in_window": report.n_days_in_window,
        "per_strategy_summary": report.per_strategy_summary,
        "alerts": [
            {"severity": a.severity, "category": a.category, "strategy": a.strategy,
             "message": a.message, "details": a.details}
            for a in report.alerts
        ],
        "notes": report.notes,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Sprint D-3 weekly reconciliation report")
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Report date YYYY-MM-DD (default: today UTC date)",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=7,
        help="Lookback window in calendar days (default 7)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report JSON to data/portfolio_replay/",
    )
    args = parser.parse_args()

    if args.as_of:
        as_of = datetime.date.fromisoformat(args.as_of)
    else:
        as_of = datetime.datetime.utcnow().date()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = run_weekly_recon(as_of, args.lookback_days)

    print(f"=== Weekly Recon Report: {report.report_date} ===")
    print(f"Window: {report.window_start} to {report.window_end} "
          f"({report.n_days_with_data}/{report.n_days_in_window} cal days with data)")
    print()
    print("Per-strategy:")
    for strat, summary in report.per_strategy_summary.items():
        print(f"  {strat:<10} n_rows={summary['n_rows']:>3}  "
              f"latest={summary['latest_status']}  "
              f"counts={summary['status_counts']}  "
              f"rebal_events={summary['n_rebalance_events']}")

    print()
    if report.alerts:
        print(f"Alerts ({len(report.alerts)}):")
        for a in report.alerts:
            print(f"  [{a.severity:<8}] {a.category:<20} "
                  f"{a.strategy or '-':<12} {a.message}")
    else:
        print("Alerts: 0 (all clean)")

    print()
    print("Notes:")
    for n in report.notes:
        print(f"  - {n}")

    if args.save:
        save_path = Path(f"data/portfolio_replay/weekly_recon_{as_of.isoformat()}.json")
        save_recon_report(report, save_path)
        print(f"\nSaved to {save_path}")

    # Non-zero exit if CRITICAL alerts present (for Watchdog integration)
    n_critical = sum(1 for a in report.alerts if a.severity == "CRITICAL")
    return 1 if n_critical > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
