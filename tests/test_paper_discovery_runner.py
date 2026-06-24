"""Tests for scripts.run_paper_discovery — Phase 7 cadence runner.

We test the discover_new_flow / discover_backfill orchestrators by
mocking the fetcher + run_discovery_batch so no network is required.
The full e2e is covered by the upstream Phase 8a/8b tests.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest


@pytest.fixture
def sample_papers_df():
    return pd.DataFrame([
        {"source": "arxiv", "source_id": "2401.001", "title": "T1",
         "authors": "A", "abstract": "abs", "categories": "q-fin.PM",
         "submitted_date": "2024-01-15", "updated_date": None,
         "pdf_url": "x", "abs_url": "y"},
        {"source": "nber",  "source_id": "w32100", "title": "T2",
         "authors": "B", "abstract": "abs", "categories": "G12",
         "submitted_date": "2024-02-01", "updated_date": None,
         "pdf_url": "x", "abs_url": "y"},
    ])


def _import_runner():
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from scripts import run_paper_discovery
    return run_paper_discovery


# ── discover_new_flow ──────────────────────────────────────────────────────

def test_discover_new_flow_happy_path(sample_papers_df):
    runner = _import_runner()
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        return_value=sample_papers_df,
    ), mock.patch(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        return_value={"total": 2, "queued": 1, "review_with_caveat": 0,
                       "stage_counts": {"queue_for_review": 1, "skip": 1}},
    ):
        result = runner.discover_new_flow(use_llm=False)
    assert result["mode"] == "new_flow"
    assert result["papers_fetched"] == 2
    assert result["by_source"] == {"arxiv": 1, "nber": 1}
    assert result["summary"]["queued"] == 1


def test_discover_new_flow_empty():
    runner = _import_runner()
    empty = pd.DataFrame(columns=["source", "title"])
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        return_value=empty,
    ):
        result = runner.discover_new_flow(use_llm=False)
    assert result["papers_fetched"] == 0
    assert result["summary"] is None


# ── discover_backfill ──────────────────────────────────────────────────────

def test_discover_backfill_happy_path(sample_papers_df):
    runner = _import_runner()
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_historical_backfill",
        return_value=sample_papers_df,
    ), mock.patch(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        return_value={"total": 2, "queued": 0, "review_with_caveat": 0,
                       "stage_counts": {"skip": 2}},
    ):
        result = runner.discover_backfill(
            "2018-01-01", "2024-12-31",
            max_per_year=100, sources=["arxiv", "nber"],
            use_llm=False,
        )
    assert result["mode"] == "backfill"
    assert result["start_date"] == "2018-01-01"
    assert result["end_date"] == "2024-12-31"
    assert result["papers_fetched"] == 2
    assert result["summary"]["total"] == 2


def test_discover_backfill_only_arxiv_subset(sample_papers_df):
    """sources=['arxiv'] passed through to fetcher."""
    runner = _import_runner()
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_historical_backfill"
    ) as fetch_mock, mock.patch(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        return_value={"total": 0, "queued": 0, "review_with_caveat": 0,
                       "stage_counts": {}},
    ):
        fetch_mock.return_value = sample_papers_df
        runner.discover_backfill(
            "2020-01-01", "2020-12-31",
            sources=["arxiv"], use_llm=False,
        )
    _, kwargs = fetch_mock.call_args
    assert kwargs.get("sources") == ["arxiv"]


# ── _write_run_log ─────────────────────────────────────────────────────────

def test_write_run_log_appends(tmp_path, monkeypatch):
    """Verify log line shape and that subsequent calls append."""
    runner = _import_runner()
    monkeypatch.chdir(tmp_path)
    runner._write_run_log({"mode": "new_flow", "papers_fetched": 5,
                              "summary": {"total": 5}})
    runner._write_run_log({"mode": "backfill", "papers_fetched": 100,
                              "summary": {"total": 100}})
    log_path = tmp_path / "data" / "research" / "discovery_runs.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["mode"] == "new_flow"
    assert parsed[1]["mode"] == "backfill"
    for p in parsed:
        assert "timestamp_utc" in p
        datetime.datetime.fromisoformat(p["timestamp_utc"])    # well-formed


# ── main CLI ───────────────────────────────────────────────────────────────

def test_main_new_flow_invocation(sample_papers_df, tmp_path, monkeypatch):
    runner = _import_runner()
    monkeypatch.chdir(tmp_path)
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        return_value=sample_papers_df,
    ), mock.patch(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        return_value={"total": 2, "queued": 0, "review_with_caveat": 0,
                       "stage_counts": {"skip": 2}},
    ):
        rc = runner.main(["--new-flow", "--no-llm", "--max-per-source", "10"])
    assert rc == 0
    # run log emitted
    assert (tmp_path / "data" / "research" / "discovery_runs.jsonl").exists()


def test_main_backfill_invocation(sample_papers_df, tmp_path, monkeypatch):
    runner = _import_runner()
    monkeypatch.chdir(tmp_path)
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_historical_backfill",
        return_value=sample_papers_df,
    ), mock.patch(
        "engine.research.discovery.discovery_pipeline.run_discovery_batch",
        return_value={"total": 2, "queued": 0, "review_with_caveat": 0,
                       "stage_counts": {"skip": 2}},
    ):
        rc = runner.main([
            "--backfill", "--start", "2020-01-01", "--end", "2020-12-31",
            "--max-per-year", "50", "--sources", "arxiv,nber", "--no-llm",
        ])
    assert rc == 0


def test_main_empty_fetch_returns_nonzero(monkeypatch, tmp_path):
    """No papers fetched = run treated as 'soft failure' (exit 1) so cron
    can detect it. This is intentional: 0 papers from new-flow over 14 days
    is suspicious (likely source_health all-red)."""
    runner = _import_runner()
    monkeypatch.chdir(tmp_path)
    empty = pd.DataFrame(columns=["source", "title"])
    with mock.patch(
        "engine.research.discovery.multi_source_dispatch.fetch_new_flow",
        return_value=empty,
    ):
        rc = runner.main(["--new-flow", "--no-llm"])
    assert rc == 1
