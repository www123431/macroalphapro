"""
engine/agents/dq_inspector/orchestrator_hook.py — Phase 6 daily integration.

3 hooks per spec §2.3 (Q1 resolution):
  pre_batch_gate(as_of)    — modes 1/2/3/4 (freshness)
  post_feed_gate(as_of, ...) — modes 5/6/7/9 (coverage + anomaly)
  post_batch_gate(today_rows, ...) — modes 8/10a/10b (row-count + volume)

HARD HALT in pre_batch_gate → scripts/run_paper_trade_daily.py exits 6
BEFORE feed refresh (avoids expensive Wikipedia + EDGAR fetch on doomed
run). HARD HALT in post_feed_gate → exit 6 BEFORE orchestrator runs.
HARD HALT in post_batch_gate → soft-warn only at this stage (book is
already persisted; HALT here escalates legacy CB SEVERE for tomorrow's
run).

Each hook persists alerts (unless dry_run) and writes _HALT.json marker
on HARD HALT. Cross-agent escalation: HARD HALT propagates to legacy
circuit_breaker.set_external_halt_flag(source='dq_inspector_*').
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd

from engine.agents.dq_inspector.agent import DQInspectorRunResult
from engine.agents.dq_inspector.gates import (
    Breach,
    any_hard_halt,
    classify_severity,
    evaluate_post_batch,
    evaluate_post_feed,
    evaluate_pre_batch,
)
from engine.agents.dq_inspector.persist import persist_breaches_to_db

logger = logging.getLogger(__name__)


HALT_FLAG_DIR = Path("data/dq_inspector/halts")


def _write_halt_marker(
    as_of:     datetime.date,
    phase:     str,
    breaches:  list[Breach],
    severity:  str,
) -> Path:
    """Write _HALT.json marker for downstream visibility."""
    HALT_FLAG_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of":            as_of.isoformat(),
        "phase":            phase,
        "halt_decision":    True,
        "severity":         severity,
        "n_breaches":       len(breaches),
        "hard_halt_modes":  sorted({b.mode_id for b in breaches if b.severity == "HARD_HALT"}),
        "first_breach":     {
            "mode_id":          breaches[0].mode_id,
            "rule_description": breaches[0].rule_description,
            "source_id":        breaches[0].extra.get("source_id", ""),
        } if breaches else None,
        "spec_id":          70,
        "written_at_utc":   datetime.datetime.utcnow().isoformat() + "Z",
    }
    path = HALT_FLAG_DIR / f"{as_of.isoformat()}_{phase}_HALT.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _persist_severe_to_legacy_cb(reason: str, source: str) -> None:
    """Cross-bridge to engine.circuit_breaker.set_external_halt_flag."""
    try:
        from engine.circuit_breaker import set_external_halt_flag
        set_external_halt_flag(reason=reason, source=source)
        logger.error("DQ Inspector escalated SEVERE to legacy CB (source=%s)", source)
    except Exception:
        logger.exception("DQ escalation to legacy CB failed (non-fatal)")


def _wrap_run_result(
    breaches:    list[Breach],
    as_of:       datetime.date,
    phase:       str,
    halt:        bool,
    dry_run:     bool,
    audit_ids:   tuple[str, ...],
    started:     datetime.datetime,
) -> DQInspectorRunResult:
    finished = datetime.datetime.utcnow()
    return DQInspectorRunResult(
        started_at_iso    = started.isoformat(),
        finished_at_iso   = finished.isoformat(),
        today_iso         = as_of.isoformat(),
        phase             = phase,
        dry_run           = dry_run,
        n_modes_evaluated = {"pre_batch": 4, "post_feed": 4, "post_batch": 3}.get(phase, 0),
        breaches          = tuple(breaches),
        halt              = halt,
        severity          = classify_severity(breaches),
        narratives        = (),
        llm_cost_usd      = 0.0,
        audit_alert_ids   = audit_ids,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PRE-BATCH GATE — 06:01 SGT before feed refresh
# ──────────────────────────────────────────────────────────────────────────────
def pre_batch_gate(
    as_of:    datetime.date,
    dry_run:  bool = False,
) -> DQInspectorRunResult:
    """Cheap freshness checks (modes 1/2/3/4). HARD HALT → exit 6
    BEFORE Step 1 feed refresh. Persists alerts unless dry_run."""
    started = datetime.datetime.utcnow()
    breaches = evaluate_pre_batch(as_of)
    halt = any_hard_halt(breaches)
    sev = classify_severity(breaches)

    audit_ids: tuple[str, ...] = ()
    if not dry_run and breaches:
        ids = persist_breaches_to_db(
            breaches, as_of, phase="pre_batch", halt_decision=halt,
        )
        audit_ids = tuple(ids)
        if halt:
            first_hard = next(b for b in breaches if b.severity == "HARD_HALT")
            _persist_severe_to_legacy_cb(
                reason = f"dq pre-batch mode {first_hard.mode_id}: {first_hard.rule_description[:280]}",
                source = "dq_inspector_pre_batch",
            )
            _write_halt_marker(as_of, "pre_batch", breaches, sev)
            logger.error(
                "DQ PRE-BATCH HALT — %d breaches (%s); HARD_HALT modes: %s",
                len(breaches), sev,
                sorted({b.mode_id for b in breaches if b.severity == "HARD_HALT"}),
            )

    return _wrap_run_result(breaches, as_of, "pre_batch", halt, dry_run,
                            audit_ids, started)


# ──────────────────────────────────────────────────────────────────────────────
# POST-FEED GATE — 06:04 SGT after Step 1 feed refresh, before Step 2 orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def post_feed_gate(
    as_of:               datetime.date,
    k1_n_with_price:     int,
    pead_n_with_rdq:     int,
    daily_returns:       Optional["pd.Series"] = None,
    ticker_to_sleeves:   Optional[dict[str, set[str]]] = None,
    n_nan_close:         int = 0,
    n_universe:          int = 0,
    dry_run:             bool = False,
) -> DQInspectorRunResult:
    """Coverage + anomaly + NaN burst (modes 5/6/7/9). HARD HALT → exit 6
    BEFORE Step 2 run_paper_trade_day."""
    started = datetime.datetime.utcnow()
    breaches = evaluate_post_feed(
        as_of, k1_n_with_price, pead_n_with_rdq,
        daily_returns=daily_returns,
        ticker_to_sleeves=ticker_to_sleeves,
        n_nan_close=n_nan_close,
        n_universe=n_universe,
    )
    halt = any_hard_halt(breaches)
    sev = classify_severity(breaches)

    audit_ids: tuple[str, ...] = ()
    if not dry_run and breaches:
        ids = persist_breaches_to_db(
            breaches, as_of, phase="post_feed", halt_decision=halt,
        )
        audit_ids = tuple(ids)
        if halt:
            first_hard = next(b for b in breaches if b.severity == "HARD_HALT")
            _persist_severe_to_legacy_cb(
                reason = f"dq post-feed mode {first_hard.mode_id}: {first_hard.rule_description[:280]}",
                source = "dq_inspector_post_feed",
            )
            _write_halt_marker(as_of, "post_feed", breaches, sev)
            logger.error(
                "DQ POST-FEED HALT — %d breaches (%s); HARD_HALT modes: %s",
                len(breaches), sev,
                sorted({b.mode_id for b in breaches if b.severity == "HARD_HALT"}),
            )

    return _wrap_run_result(breaches, as_of, "post_feed", halt, dry_run,
                            audit_ids, started)


# ──────────────────────────────────────────────────────────────────────────────
# POST-BATCH GATE — 06:09 SGT after paper_trade persist
# ──────────────────────────────────────────────────────────────────────────────
def post_batch_gate(
    as_of:              datetime.date,
    today_rows:         int,
    yesterday_rows:     int,
    volume_today:       Optional[dict[str, float]] = None,
    volume_60d_median:  Optional[dict[str, float]] = None,
    dry_run:            bool = False,
) -> DQInspectorRunResult:
    """Row-count + volume dropoff (modes 8/10a/10b). HARD HALT here is
    informational — book already persisted. Mode 10b STILL escalates to
    legacy CB so next day's cycle requires manual_reset."""
    started = datetime.datetime.utcnow()
    breaches = evaluate_post_batch(
        today_rows, yesterday_rows,
        volume_today=volume_today,
        volume_60d_median=volume_60d_median,
    )
    halt = any_hard_halt(breaches)
    sev = classify_severity(breaches)

    audit_ids: tuple[str, ...] = ()
    if not dry_run and breaches:
        ids = persist_breaches_to_db(
            breaches, as_of, phase="post_batch", halt_decision=halt,
        )
        audit_ids = tuple(ids)
        if halt:
            first_hard = next(b for b in breaches if b.severity == "HARD_HALT")
            _persist_severe_to_legacy_cb(
                reason = f"dq post-batch mode {first_hard.mode_id}: {first_hard.rule_description[:280]}",
                source = "dq_inspector_post_batch",
            )
            _write_halt_marker(as_of, "post_batch", breaches, sev)
            logger.error(
                "DQ POST-BATCH HALT — %d breaches (%s) — book already "
                "persisted; legacy CB escalated for tomorrow's run",
                len(breaches), sev,
            )

    return _wrap_run_result(breaches, as_of, "post_batch", halt, dry_run,
                            audit_ids, started)
