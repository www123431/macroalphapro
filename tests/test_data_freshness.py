"""Tests for engine.research.data_freshness + the heartbeat verdict
downgrade path that consumes its output (P0c).

The user surfaced the gap on 2026-06-02: a heartbeat row read "OK"
while NAV PATH on the dashboard was 21 days stale. Each test below
locks in one piece of the contract that closes that gap.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    from engine.research import data_freshness as DF
    from engine.research import liveness_heartbeat as L
    monkeypatch.setattr(DF, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(L, "LIVENESS_LEDGER", tmp_path / "data" / "research" / "liveness_heartbeat.jsonl")
    (tmp_path / "data" / "research").mkdir(parents=True)
    (tmp_path / "data" / "ui_artifact").mkdir(parents=True)
    yield tmp_path


def test_classify_buckets():
    from engine.research import data_freshness as DF
    assert DF._classify(0)    == DF.STATUS_FRESH
    assert DF._classify(1)    == DF.STATUS_FRESH
    assert DF._classify(2)    == DF.STATUS_AGING
    assert DF._classify(5)    == DF.STATUS_STALE
    assert DF._classify(30)   == DF.STATUS_DEAD


def test_age_days_handles_bad_input():
    from engine.research import data_freshness as DF
    assert DF._age_days_from("") is None
    assert DF._age_days_from("not-a-date") is None
    assert DF._age_days_from("2026-06-02", now=_dt.date(2026, 6, 5)) == 3.0


def test_decay_sentinel_probe_classifies_age(tmp_repo):
    from engine.research import data_freshness as DF
    (tmp_repo / "data" / "research" / "decay_sentinel_history.jsonl").write_text(
        json.dumps({"audit_date": "2026-05-12", "sleeve": "x"}) + "\n"
        + json.dumps({"audit_date": "2026-05-12", "sleeve": "y"}) + "\n",
        encoding="utf-8",
    )
    r = DF.check_decay_sentinel(now=_dt.date(2026, 6, 2))
    assert r["latest_date"] == "2026-05-12"
    assert r["age_days"]   == 21.0
    assert r["status"]     == DF.STATUS_DEAD
    assert r["n_rows"]     == 2


def test_decay_sentinel_missing_file(tmp_repo):
    from engine.research import data_freshness as DF
    r = DF.check_decay_sentinel(now=_dt.date(2026, 6, 2))
    assert r["status"] == DF.STATUS_MISSING


def test_ui_artifact_probe(tmp_repo):
    from engine.research import data_freshness as DF
    (tmp_repo / "data" / "ui_artifact" / "2026-05-10.json").write_text("{}", encoding="utf-8")
    (tmp_repo / "data" / "ui_artifact" / "2026-06-01.json").write_text("{}", encoding="utf-8")
    r = DF.check_ui_artifact(now=_dt.date(2026, 6, 2))
    assert r["latest_date"] == "2026-06-01"
    assert r["age_days"]   == 1.0
    assert r["status"]     == DF.STATUS_FRESH
    assert r["n_rows"]     == 2


def test_check_sources_runs_all_probes_without_raising(tmp_repo):
    """Even if every backing source is missing, check_sources returns
    a dict per probe — best-effort is non-negotiable."""
    from engine.research import data_freshness as DF
    rows = DF.check_sources(now=_dt.date(2026, 6, 2))
    assert len(rows) == 4
    sources = {r["source"] for r in rows}
    assert sources == {"nav_history", "decay_sentinel", "paper_trade_log", "ui_artifact"}
    # Each row has the stable contract fields
    for r in rows:
        assert "status" in r
        assert "source" in r


def test_summarize_worst_status_is_dead_when_any_source_dead(tmp_repo):
    from engine.research import data_freshness as DF
    fake = [
        {"source": "a", "status": DF.STATUS_FRESH},
        {"source": "b", "status": DF.STATUS_DEAD},
        {"source": "c", "status": DF.STATUS_AGING},
    ]
    s = DF.summarize(fake)
    assert s["worst_status"] == DF.STATUS_DEAD
    assert s["worst_source"] == "b"
    assert s["n_dead"] == 1
    assert "DEAD" in s["headline"]


def test_summarize_clean_book_is_fresh(tmp_repo):
    from engine.research import data_freshness as DF
    fake = [
        {"source": "a", "status": DF.STATUS_FRESH},
        {"source": "b", "status": DF.STATUS_FRESH},
    ]
    s = DF.summarize(fake)
    assert s["worst_status"] == DF.STATUS_FRESH
    assert "fresh" in s["headline"].lower()


# ── End-to-end: heartbeat verdict downgrade ─────────────────────────


def test_assess_liveness_downgrades_to_warn_when_data_dead(tmp_repo):
    """The bug the user surfaced: heartbeat status=success, but data is
    21 days stale. Previously this returned verdict=OK. After P0c, the
    verdict should be WARN_STATUS with an explanation pointing at the
    dead source."""
    from engine.research import liveness_heartbeat as L
    from engine.research import data_freshness as DF

    L.record_run(
        as_of=_dt.date(2026, 6, 1),
        exit_code=0,
        n_orders=114, n_fills=114,
        data_freshness={
            "worst_status":  DF.STATUS_DEAD,
            "worst_source":  "nav_history",
            "n_dead": 1, "n_stale": 0, "n_aging": 0,
            "n_fresh": 3, "n_missing": 0, "n_unknown": 0, "n_total": 4,
            "headline": "1 data source DEAD — oldest: nav_history. Cron may be running but writing to a dead pipe.",
        },
        data_sources=[
            {"source": "nav_history",     "status": DF.STATUS_DEAD,  "latest_date": "2026-05-12", "age_days": 20.0},
            {"source": "decay_sentinel",  "status": DF.STATUS_FRESH, "latest_date": "2026-06-01", "age_days": 0.0},
            {"source": "paper_trade_log", "status": DF.STATUS_FRESH, "latest_date": "2026-06-01", "age_days": 0.0},
            {"source": "ui_artifact",     "status": DF.STATUS_FRESH, "latest_date": "2026-06-01", "age_days": 0.0},
        ],
    )
    verdict = L.assess_liveness(now_utc=_dt.datetime(2026, 6, 1, 23, 30))
    assert verdict["verdict"] == "WARN_STATUS"
    assert "nav_history" in verdict["explanation"]
    assert "DEAD" in verdict["explanation"] or "dead" in verdict["explanation"].lower()


def test_assess_liveness_stays_ok_when_data_only_aging(tmp_repo):
    """Aging (1-3d) is informational, not a verdict downgrade."""
    from engine.research import liveness_heartbeat as L
    from engine.research import data_freshness as DF

    L.record_run(
        as_of=_dt.date(2026, 6, 1),
        exit_code=0,
        data_freshness={
            "worst_status":  DF.STATUS_AGING,
            "worst_source":  "decay_sentinel",
            "n_dead": 0, "n_stale": 0, "n_aging": 1,
            "n_fresh": 3, "n_missing": 0, "n_unknown": 0, "n_total": 4,
            "headline": "1 source aging",
        },
    )
    verdict = L.assess_liveness(now_utc=_dt.datetime(2026, 6, 1, 23, 30))
    assert verdict["verdict"] == "OK"


def test_record_run_persists_data_freshness_fields(tmp_repo):
    """The data_freshness + data_sources fields must round-trip through
    the JSONL ledger so the UI can read them."""
    from engine.research import liveness_heartbeat as L
    summary = {
        "worst_status": "fresh", "worst_source": None,
        "n_dead": 0, "n_stale": 0, "n_aging": 0, "n_fresh": 2,
        "n_missing": 0, "n_unknown": 0, "n_total": 2,
        "headline": "All 2 data sources fresh",
    }
    sources = [{"source": "a", "status": "fresh"}, {"source": "b", "status": "fresh"}]
    L.record_run(as_of=_dt.date(2026, 6, 1), exit_code=0,
                  data_freshness=summary, data_sources=sources)
    rows = L.read_recent(1)
    assert rows[0]["data_freshness"] == summary
    assert rows[0]["data_sources"] == sources
