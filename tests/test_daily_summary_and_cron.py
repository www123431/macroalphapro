"""Tests for research_daily_summary + install_research_cron."""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Redirect all data paths to tmp."""
    from scripts import research_daily_summary as ds
    monkeypatch.setattr(ds, "DISCOVERY_QUEUE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(ds, "DISCOVERY_BORDERLINE", tmp_path / "borderline.jsonl")
    monkeypatch.setattr(ds, "DISCOVERY_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(ds, "DISCOVERY_RUNS", tmp_path / "runs.jsonl")
    monkeypatch.setattr(ds, "DISCOVERY_REJECTED", tmp_path / "rejected.jsonl")
    monkeypatch.setattr(ds, "GATE_RUNS", tmp_path / "gate_runs.jsonl")
    monkeypatch.setattr(ds, "LLM_COST", tmp_path / "cost.jsonl")
    monkeypatch.setattr(ds, "REPO_ROOT", tmp_path)
    return {"tmp": tmp_path, "ds": ds}


def _write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ── build_summary ────────────────────────────────────────────────────────

def test_build_summary_empty_state(isolated_data):
    """Summary renders with no data — all sections show 0."""
    s = isolated_data["ds"].build_summary()
    assert "Research Daily Summary" in s
    assert "Queue State" in s
    assert "Discovery Runs" in s
    assert "Review Activity" in s
    assert "Strict Gate Activity" in s
    assert "LLM Cost" in s


def test_build_summary_counts_queues(isolated_data):
    ds = isolated_data["ds"]
    _write_jsonl(ds.DISCOVERY_QUEUE, [
        {"title": "A"}, {"title": "B"}, {"title": "C"},
    ])
    _write_jsonl(ds.DISCOVERY_BORDERLINE, [
        {"title": "X"},
    ])
    s = ds.build_summary()
    assert "**3** entries" in s
    assert "**1** entries" in s


def test_build_summary_counts_recent_runs(isolated_data):
    ds = isolated_data["ds"]
    now = datetime.datetime.utcnow().isoformat()
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=3)).isoformat()
    _write_jsonl(ds.DISCOVERY_RUNS, [
        {"timestamp_utc": now, "papers_fetched": 10,
          "summary": {"stage_counts": {"skip": 8, "queued": 2}}},
        {"timestamp_utc": old, "papers_fetched": 5,
          "summary": {"stage_counts": {"skip": 5}}},
    ])
    s = ds.build_summary()
    # Last 24h: 1 run, 10 papers; older one excluded
    assert "Runs:    **1**" in s
    assert "Papers fetched: **10**" in s


def test_build_summary_skip_reasons(isolated_data):
    ds = isolated_data["ds"]
    now = datetime.datetime.utcnow().isoformat()
    _write_jsonl(ds.DISCOVERY_REJECTED, [
        {"title": "A", "skipped_at": now, "skip_reason": "off_topic"},
        {"title": "B", "skipped_at": now, "skip_reason": "off_topic"},
        {"title": "C", "skipped_at": now, "skip_reason": "weak_signal"},
    ])
    s = ds.build_summary()
    assert "Skipped: **3**" in s
    assert "off_topic" in s


def test_build_summary_gate_verdicts(isolated_data):
    ds = isolated_data["ds"]
    now = datetime.datetime.utcnow().isoformat()
    _write_jsonl(ds.GATE_RUNS, [
        {"ts": now, "verdict": "RED"},
        {"ts": now, "verdict": "GREEN"},
        {"ts": now, "verdict": "RED", "provisional_synthetic": True},
    ])
    s = ds.build_summary()
    assert "RED" in s
    assert "GREEN" in s
    assert "Provisional-synthetic" in s
    assert "**1**" in s


def test_build_summary_llm_cost(isolated_data):
    ds = isolated_data["ds"]
    now = datetime.datetime.utcnow().isoformat()
    week_ago = (datetime.datetime.utcnow() - datetime.timedelta(days=3)).isoformat()
    _write_jsonl(ds.LLM_COST, [
        {"ts": now, "cost_usd": 0.05},
        {"ts": now, "cost_usd": 0.03},
        {"ts": week_ago, "cost_usd": 0.10},
    ])
    s = ds.build_summary()
    assert "Today" in s
    assert "$0.08" in s   # today's total
    assert "$0.18" in s   # week total (all three)


def test_no_emoji_in_summary(isolated_data):
    """Per [[feedback-no-emoji-2026-05-30]] — daily summary text
    must not contain emoji."""
    ds = isolated_data["ds"]
    _write_jsonl(ds.DISCOVERY_QUEUE, [{"title": "X"}])
    s = ds.build_summary()
    for emoji in ("📚", "📌", "🟢", "🟠", "⚠️", "✓", "✗"):
        assert emoji not in s


def test_write_summary_creates_file(isolated_data, tmp_path):
    ds = isolated_data["ds"]
    target = tmp_path / "sub" / "summary.md"
    p = ds.write_summary("# hello", target)
    assert p.exists()
    assert "hello" in p.read_text(encoding="utf-8")


# ── _within helper ──────────────────────────────────────────────────────

def test_within_recent_is_true(isolated_data):
    now = datetime.datetime.utcnow().isoformat()
    assert isolated_data["ds"]._within(now, 1) is True


def test_within_old_is_false(isolated_data):
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=10)).isoformat()
    assert isolated_data["ds"]._within(old, 1) is False


def test_within_invalid_returns_false(isolated_data):
    assert isolated_data["ds"]._within("not a date", 1) is False
    assert isolated_data["ds"]._within("", 1) is False


# ── install_research_cron ────────────────────────────────────────────────

def test_unix_crontab_lines_contain_required_entries():
    from scripts.install_research_cron import unix_crontab_lines
    lines = unix_crontab_lines()
    joined = "\n".join(lines)
    assert "run_paper_discovery.py" in joined
    assert "research_daily_summary.py" in joined
    assert "--new-flow" in joined
    assert "--backfill" in joined


def test_unix_install_dry_run_does_not_raise(capsys):
    from scripts.install_research_cron import unix_install
    unix_install(dry_run=True)
    out = capsys.readouterr().out
    assert "Unix cron entries:" in out
    assert "dry-run" in out
