"""Tests for engine.research.discovery.pipeline_health."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from engine.research.discovery import pipeline_health as ph
from engine.research.discovery.pipeline_health import (
    HealthLevel, CheckResult, _aggregate,
)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Redirect all log paths to tmp."""
    paths = {
        "DISCOVERY_RUNS":         tmp_path / "discovery_runs.jsonl",
        "DISCOVERY_QUEUE":        tmp_path / "discovery_queue.jsonl",
        "DISCOVERY_BORDERLINE":    tmp_path / "discovery_borderline.jsonl",
        "GATE_RUNS":              tmp_path / "gate_runs.jsonl",
        "LLM_COST":               tmp_path / "llm_cost.jsonl",
        "PAPER_TRADE_DIR":         tmp_path / "paper_trade",
        "DECAY_REPORT":           tmp_path / "decay_report.json",
        "OPS_WIDGET_STATE":       tmp_path / "ops_widget.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(ph, name, path)
    return {"tmp": tmp_path, "paths": paths}


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ── _aggregate ────────────────────────────────────────────────────────

def test_aggregate_all_ok():
    checks = [
        CheckResult("a", HealthLevel.OK, "ok"),
        CheckResult("b", HealthLevel.OK, "ok"),
    ]
    assert _aggregate(checks) == HealthLevel.OK


def test_aggregate_any_alert_is_alert():
    checks = [
        CheckResult("a", HealthLevel.OK, "ok"),
        CheckResult("b", HealthLevel.ALERT, "alert"),
    ]
    assert _aggregate(checks) == HealthLevel.ALERT


def test_aggregate_warn_without_alert():
    checks = [
        CheckResult("a", HealthLevel.OK, "ok"),
        CheckResult("b", HealthLevel.WARN, "warn"),
    ]
    assert _aggregate(checks) == HealthLevel.WARN


def test_aggregate_unknown_returned_when_mixed():
    """When checks include UNKNOWN with no ALERT/WARN/all-OK → UNKNOWN."""
    checks = [
        CheckResult("a", HealthLevel.OK, "ok"),
        CheckResult("b", HealthLevel.UNKNOWN, "?"),
    ]
    assert _aggregate(checks) == HealthLevel.UNKNOWN


# ── check_discovery_freshness ─────────────────────────────────────────

def test_discovery_freshness_no_log_alerts(isolated):
    result = ph.check_discovery_freshness()
    assert result.status == HealthLevel.ALERT
    assert result.remedy is not None
    # Borrowed from PIT audit style — remedy must be actionable
    assert "discovery" in result.remedy.lower()


def test_discovery_freshness_recent_ok(isolated):
    runs = isolated["paths"]["DISCOVERY_RUNS"]
    now = datetime.datetime.utcnow().isoformat()
    _write_jsonl(runs, [{"timestamp_utc": now, "papers_fetched": 10}])
    result = ph.check_discovery_freshness()
    assert result.status == HealthLevel.OK


def test_discovery_freshness_old_warns(isolated):
    runs = isolated["paths"]["DISCOVERY_RUNS"]
    old = (datetime.datetime.utcnow()
            - datetime.timedelta(hours=30)).isoformat()
    _write_jsonl(runs, [{"timestamp_utc": old}])
    result = ph.check_discovery_freshness()
    assert result.status == HealthLevel.WARN


def test_discovery_freshness_very_old_alerts_with_remedy(isolated):
    runs = isolated["paths"]["DISCOVERY_RUNS"]
    very_old = (datetime.datetime.utcnow()
                 - datetime.timedelta(days=10)).isoformat()
    _write_jsonl(runs, [{"timestamp_utc": very_old}])
    result = ph.check_discovery_freshness()
    assert result.status == HealthLevel.ALERT
    assert result.remedy
    assert "schtasks" in result.remedy or "scripts/run_paper_discovery" in result.remedy


# ── check_queue_drain ─────────────────────────────────────────────────

def test_queue_drain_empty_ok(isolated):
    result = ph.check_queue_drain()
    assert result.status == HealthLevel.OK


def test_queue_drain_bloated_alerts(isolated):
    queue = isolated["paths"]["DISCOVERY_QUEUE"]
    border = isolated["paths"]["DISCOVERY_BORDERLINE"]
    # Write > 2x threshold so ALERT (not WARN) fires
    _write_jsonl(queue, [{"id": i} for i in range(150)])
    _write_jsonl(border, [{"id": i} for i in range(100)])
    result = ph.check_queue_drain()
    assert result.status == HealthLevel.ALERT
    assert result.remedy is not None


# ── check_llm_budget ──────────────────────────────────────────────────

def test_llm_budget_no_spend_ok(isolated):
    result = ph.check_llm_budget()
    assert result.status == HealthLevel.OK


def test_llm_budget_under_budget_ok(isolated):
    cost = isolated["paths"]["LLM_COST"]
    now = datetime.datetime.utcnow().isoformat()
    _write_jsonl(cost, [
        {"ts": now, "cost_usd": 5.0},
        {"ts": now, "cost_usd": 10.0},
    ])
    result = ph.check_llm_budget()
    assert result.status == HealthLevel.OK


def test_llm_budget_runaway_alerts(isolated):
    cost = isolated["paths"]["LLM_COST"]
    now = datetime.datetime.utcnow().isoformat()
    _write_jsonl(cost, [
        {"ts": now, "cost_usd": 200.0},   # way above $50 budget
    ])
    result = ph.check_llm_budget()
    assert result.status == HealthLevel.ALERT
    assert result.remedy is not None


# ── Book-side checks ──────────────────────────────────────────────────

def test_paper_trade_freshness_missing_dir_unknown(isolated):
    result = ph.check_paper_trade_freshness()
    assert result.status == HealthLevel.UNKNOWN


def test_paper_trade_freshness_recent_log_ok(isolated):
    pt_dir = isolated["paths"]["PAPER_TRADE_DIR"]
    pt_dir.mkdir()
    log_file = pt_dir / "daily_run_2024-01-01.log"
    log_file.write_text("ok", encoding="utf-8")
    # mtime is now → fresh
    result = ph.check_paper_trade_freshness()
    assert result.status == HealthLevel.OK


def test_decay_sentinel_missing_unknown(isolated):
    result = ph.check_decay_sentinel()
    assert result.status == HealthLevel.UNKNOWN


def test_ops_watchdog_missing_unknown(isolated):
    result = ph.check_ops_watchdog()
    assert result.status == HealthLevel.UNKNOWN


def test_ops_watchdog_ok_status(isolated):
    state = isolated["paths"]["OPS_WIDGET_STATE"]
    state.write_text(json.dumps({"severity": "OK", "as_of": "2024-01-01"}),
                       encoding="utf-8")
    result = ph.check_ops_watchdog()
    assert result.status == HealthLevel.OK


def test_ops_watchdog_alert_status(isolated):
    state = isolated["paths"]["OPS_WIDGET_STATE"]
    state.write_text(json.dumps({"severity": "CRITICAL", "as_of": "2024-01-01"}),
                       encoding="utf-8")
    result = ph.check_ops_watchdog()
    assert result.status == HealthLevel.ALERT


# ── report aggregator ────────────────────────────────────────────────

def test_report_returns_complete_shape(isolated):
    r = ph.report()
    for k in ("status", "as_of", "checks", "tunables"):
        assert k in r
    # All check names present (book + discovery)
    names = {c["name"] for c in r["checks"]}
    for required in ("discovery_freshness", "gate_freshness",
                       "queue_drain", "llm_budget",
                       "paper_trade_freshness", "decay_sentinel",
                       "ops_watchdog"):
        assert required in names


def test_report_status_aggregates_from_checks(isolated):
    cost = isolated["paths"]["LLM_COST"]
    now = datetime.datetime.utcnow().isoformat()
    # Force LLM budget alert
    _write_jsonl(cost, [{"ts": now, "cost_usd": 1000}])
    r = ph.report()
    assert r["status"] in ("WARN", "ALERT")


# ── Remedy field (borrowed from PIT audit display) ──────────────────

def test_all_alerts_have_remedy(isolated):
    """Senior borrow: every ALERT must include a remedy string."""
    # Force a few alerts
    pt_dir = isolated["paths"]["PAPER_TRADE_DIR"]
    pt_dir.mkdir()
    very_old_log = pt_dir / "daily_run_x.log"
    very_old_log.write_text("x", encoding="utf-8")
    import os
    old_ts = datetime.datetime.utcnow().timestamp() - 86400 * 5    # 5 days ago
    os.utime(very_old_log, (old_ts, old_ts))

    r = ph.report()
    for c in r["checks"]:
        if c["status"] == "ALERT":
            assert c["remedy"], f"{c['name']} ALERT but no remedy: {c}"
