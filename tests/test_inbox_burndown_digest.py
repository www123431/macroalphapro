"""burn-2 tests — source_burndown_digest pulls plan + outcome JSONs."""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from engine.inbox import composer


def _write_plan(plan_dir: Path, *, date_str: str, plan_id: str, actual_k: int,
                dry_run: bool, candidates=None, usage_summary="0/15") -> Path:
    plan_dir.mkdir(parents=True, exist_ok=True)
    fp = plan_dir / f"{date_str}_{plan_id[:8]}.json"
    fp.write_text(json.dumps({
        "plan_id":       plan_id,
        "ts":            f"{date_str}T09:00:00Z",
        "target_k":      3,
        "actual_k":      actual_k,
        "candidates":    candidates or [],
        "usage_before":  {},
        "usage_summary": usage_summary,
        "queue_size":    100,
        "cap_status":    "",
        "skipped_counts": {},
        "dry_run":       dry_run,
    }), encoding="utf-8")
    return fp


def _write_outcomes(outcome_dir: Path, *, date_str: str, plan_id: str,
                     outcomes: list[dict]) -> Path:
    outcome_dir.mkdir(parents=True, exist_ok=True)
    fp = outcome_dir / f"{date_str}_{plan_id[:8]}.json"
    fp.write_text(json.dumps({
        "plan_id": plan_id,
        "ran_at":  f"{date_str}T09:10:00Z",
        "outcomes": outcomes,
    }), encoding="utf-8")
    return fp


def _today_str() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


# ── Plan-only (dry-run) digest ─────────────────────────────────────


def test_digest_dry_run_plan_only(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    today = _today_str()
    _write_plan(pd, date_str=today, plan_id="plan-uuid-1234",
                actual_k=3, dry_run=True,
                candidates=[
                    {"family": "VALUE",         "hypothesis_id": "h1"},
                    {"family": "PROFITABILITY", "hypothesis_id": "h2"},
                    {"family": "SIZE",          "hypothesis_id": "h3"},
                ])
    items = composer.source_burndown_digest(plan_dir=pd, outcome_dir=od)
    assert len(items) == 1
    item = items[0]
    assert item["source"] == "burndown_cron"
    assert "DRY-RUN" in item["title"]
    assert "VALUE" in item["title"] or "PROFITABILITY" in item["title"]
    assert item["tone"] == "info"
    assert item["metadata"]["dry_run"] is True
    assert item["lane"] == composer._LANE_ENGINE


# ── Execution outcomes digest ──────────────────────────────────────


def test_digest_with_executed_outcomes(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    today = _today_str()
    _write_plan(pd, date_str=today, plan_id="planhash00aaaa",
                actual_k=3, dry_run=False)
    _write_outcomes(od, date_str=today, plan_id="planhash00aaaa",
                     outcomes=[
                         {"verdict": "GREEN",    "decay_severity": "none",   "extraction_ok": True,  "refusal_reason": None},
                         {"verdict": "MARGINAL", "decay_severity": "mild",   "extraction_ok": True,  "refusal_reason": None},
                         {"verdict": None,       "decay_severity": None,     "extraction_ok": True,  "refusal_reason": "TEMPLATE_NOT_CERTIFIED"},
                     ])
    items = composer.source_burndown_digest(plan_dir=pd, outcome_dir=od)
    assert len(items) == 1
    item = items[0]
    assert "G1/M1/R0" in item["title"]
    assert "refused 1" in item["title"]
    assert item["tone"] == "info"   # No RED, no severe decay
    md = item["metadata"]
    assert md["verdicts"] == {"GREEN": 1, "MARGINAL": 1, "RED": 0}
    assert md["refused"] == 1


def test_digest_red_or_severe_decay_warn_tone(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    today = _today_str()
    _write_plan(pd, date_str=today, plan_id="planuuid-warn",
                actual_k=2, dry_run=False)
    _write_outcomes(od, date_str=today, plan_id="planuuid-warn",
                     outcomes=[
                         {"verdict": "RED",      "decay_severity": None,    "extraction_ok": True, "refusal_reason": None},
                         {"verdict": "GREEN",    "decay_severity": "broken","extraction_ok": True, "refusal_reason": None},
                     ])
    items = composer.source_burndown_digest(plan_dir=pd, outcome_dir=od)
    assert items[0]["tone"] == "warn"
    assert "decay_severe" in items[0]["metadata"]
    assert items[0]["metadata"]["decay_severe"] == 1


def test_digest_extract_failure_summary(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    today = _today_str()
    _write_plan(pd, date_str=today, plan_id="planuuid-extract",
                actual_k=2, dry_run=False)
    _write_outcomes(od, date_str=today, plan_id="planuuid-extract",
                     outcomes=[
                         {"verdict": None, "decay_severity": None,
                          "extraction_ok": False, "extraction_error": "EXTRACT_RETURNED_NONE",
                          "refusal_reason": None},
                         {"verdict": "GREEN", "decay_severity": "none",
                          "extraction_ok": True, "refusal_reason": None},
                     ])
    items = composer.source_burndown_digest(plan_dir=pd, outcome_dir=od)
    assert "extract_fail 1" in items[0]["title"]
    assert items[0]["metadata"]["extract_fail"] == 1
    assert "extraction failure" in items[0]["summary"]


def test_digest_empty_when_no_plans(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    pd.mkdir()
    assert composer.source_burndown_digest(plan_dir=pd, outcome_dir=od) == []


def test_digest_respects_days_back_window(tmp_path):
    pd = tmp_path / "plans"
    od = tmp_path / "outcomes"
    # Old plan from 5 days ago — should be excluded with days_back=2
    old_date = (_dt.datetime.utcnow() - _dt.timedelta(days=5)).strftime("%Y-%m-%d")
    today = _today_str()
    _write_plan(pd, date_str=old_date, plan_id="plan-uuid-OLD",
                actual_k=1, dry_run=True)
    _write_plan(pd, date_str=today, plan_id="plan-uuid-NEW",
                actual_k=1, dry_run=True)
    items = composer.source_burndown_digest(plan_dir=pd, outcome_dir=od, days_back=2)
    assert len(items) == 1
    assert "plan-uui" not in items[0]["title"] or today in items[0]["title"]
