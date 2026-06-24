"""Tests for Frontier 2 (2026-06-01) — L4 cron continuous background.

Tests the pure-Python paths (seed picker + ledger + cooling filter)
deterministically without spinning up Temporal. Schedule API helpers
(enable / disable / cron_status) are covered by mocking the Temporal
Client — the workflow body itself is exercised by the existing l4
worker integration when a live Temporal server is available.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
from pathlib import Path
from unittest import mock

import pytest

from engine.research.l4_cron import (
    SEED_COOLING_DAYS,
    _append_cron_run_log,
    _read_recent_seeds,
    pick_seed_for_today,
    read_recent_cron_runs,
)


@pytest.fixture
def isolated_ledger(tmp_path, monkeypatch):
    """Redirect cron ledger to a tmp file so tests don't touch
    data/research/l4_cron_runs.jsonl."""
    fake = tmp_path / "l4_cron_runs.jsonl"
    monkeypatch.setattr(
        "engine.research.l4_cron.L4_CRON_RUNS_LEDGER", fake,
    )
    return fake


# ── ledger ──────────────────────────────────────────────────────────────


def test_append_and_read_round_trip(isolated_ledger):
    rid = _append_cron_run_log({
        "seed": "test seed", "title": "T", "family": "fam",
        "source": "library", "child_workflow_id": "wf-1",
    })
    assert rid
    rows = read_recent_cron_runs(limit=10)
    assert len(rows) == 1
    assert rows[0]["id"] == rid
    assert rows[0]["seed"] == "test seed"


def test_read_recent_returns_newest_first(isolated_ledger):
    for i in range(3):
        _append_cron_run_log({"seed": f"s{i}", "title": f"t{i}",
                                "family": "f", "source": "library"})
    rows = read_recent_cron_runs(limit=10)
    # Most recently appended first
    assert [r["seed"] for r in rows] == ["s2", "s1", "s0"]


# ── cooling filter ──────────────────────────────────────────────────────


def test_recent_seeds_within_window(isolated_ledger):
    now = _dt.datetime.utcnow()
    fresh = now - _dt.timedelta(days=1)
    stale = now - _dt.timedelta(days=SEED_COOLING_DAYS + 2)
    # Write directly so we can control ts
    with isolated_ledger.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"seed": "fresh seed",
                              "ts": fresh.isoformat(timespec="seconds") + "Z"}) + "\n")
        f.write(json.dumps({"seed": "stale seed",
                              "ts": stale.isoformat(timespec="seconds") + "Z"}) + "\n")
    seeds = _read_recent_seeds(within_days=SEED_COOLING_DAYS)
    assert "fresh seed" in seeds
    assert "stale seed" not in seeds


def test_pick_seed_skips_recently_run(isolated_ledger):
    """Picker must skip seeds that were run within the cooling window
    and return the next-ranked candidate."""
    # Mock the suggestion engine to return 2 known seeds
    fake_suggestions = {
        "suggestions": [
            {"seed": "top seed",   "title": "T1", "family": "f1",
             "source": "library", "score": 0.9,  "proposed_role": "alpha_seeker"},
            {"seed": "next seed",  "title": "T2", "family": "f2",
             "source": "library", "score": 0.8,  "proposed_role": "alpha_seeker"},
        ],
    }
    # Pre-populate ledger with the top seed → should pick #2
    _append_cron_run_log({
        "seed": "top seed", "title": "T1", "family": "f1",
        "source": "library",
    })
    with mock.patch(
        "engine.research.suggestion_engine.get_candidate_suggestions",
        return_value=fake_suggestions,
    ):
        out = pick_seed_for_today(limit=10)
    assert out is not None
    assert out["seed"] == "next seed"


def test_pick_seed_returns_none_when_all_exhausted(isolated_ledger):
    fake_suggestions = {
        "suggestions": [
            {"seed": "only seed", "title": "T", "family": "f",
             "source": "library", "score": 0.5,  "proposed_role": "alpha_seeker"},
        ],
    }
    _append_cron_run_log({
        "seed": "only seed", "title": "T", "family": "f",
        "source": "library",
    })
    with mock.patch(
        "engine.research.suggestion_engine.get_candidate_suggestions",
        return_value=fake_suggestions,
    ):
        out = pick_seed_for_today(limit=10)
    assert out is None


def test_pick_seed_smoke_against_real_suggestion_engine(isolated_ledger):
    """Picker works against the real suggestion_engine + empty ledger.

    Smoke test — confirms the integration doesn't blow up and returns
    a structured suggestion dict matching the contract."""
    out = pick_seed_for_today(limit=10)
    if out is None:
        # Real engine might return empty in some env states — that's a
        # valid outcome, just skip rather than fail
        pytest.skip("real suggestion_engine returned no suggestions")
    for key in ("seed", "title", "family", "proposed_role",
                "source", "score"):
        assert key in out, f"missing {key}"


# ── Schedule API helpers (Temporal client mocked) ───────────────────────


def _make_fake_client(handle):
    """Build a fake Temporal Client where:
      - Client.connect() is async (returns the fake client)
      - client.get_schedule_handle(id) is SYNC (returns `handle`)
      - client.create_schedule(...) is async
    """
    fake_client = mock.MagicMock()           # sync-by-default container
    fake_client.get_schedule_handle.return_value = handle
    fake_client.create_schedule = mock.AsyncMock()
    return fake_client


def test_enable_l4_cron_creates_when_missing():
    from engine.research import l4_cron

    fake_handle = mock.AsyncMock()
    fake_handle.describe.side_effect = RuntimeError("not found")
    fake_client = _make_fake_client(fake_handle)

    with mock.patch("temporalio.client.Client.connect",
                     new=mock.AsyncMock(return_value=fake_client)):
        out = asyncio.run(l4_cron.enable_l4_cron(cron_spec="0 12 * * *"))
    assert out["ok"] is True
    assert out["action"] == "created"
    assert out["cron"] == "0 12 * * *"
    fake_client.create_schedule.assert_awaited_once()


def test_enable_l4_cron_updates_when_exists():
    from engine.research import l4_cron

    fake_handle = mock.AsyncMock()
    fake_handle.describe.return_value = mock.MagicMock()
    fake_client = _make_fake_client(fake_handle)

    with mock.patch("temporalio.client.Client.connect",
                     new=mock.AsyncMock(return_value=fake_client)):
        out = asyncio.run(l4_cron.enable_l4_cron())
    assert out["action"] == "updated"
    fake_handle.update.assert_awaited_once()
    fake_client.create_schedule.assert_not_called()


def test_disable_l4_cron_pauses_by_default():
    from engine.research import l4_cron

    fake_handle = mock.AsyncMock()
    fake_client = _make_fake_client(fake_handle)

    with mock.patch("temporalio.client.Client.connect",
                     new=mock.AsyncMock(return_value=fake_client)):
        out = asyncio.run(l4_cron.disable_l4_cron())
    assert out["ok"] is True
    assert out["action"] == "paused"
    fake_handle.pause.assert_awaited_once()
    fake_handle.delete.assert_not_called()


def test_disable_l4_cron_deletes_when_requested():
    from engine.research import l4_cron

    fake_handle = mock.AsyncMock()
    fake_client = _make_fake_client(fake_handle)

    with mock.patch("temporalio.client.Client.connect",
                     new=mock.AsyncMock(return_value=fake_client)):
        out = asyncio.run(l4_cron.disable_l4_cron(delete=True))
    assert out["action"] == "deleted"
    fake_handle.delete.assert_awaited_once()
    fake_handle.pause.assert_not_called()


def test_cron_status_when_temporal_down(isolated_ledger):
    """Schedule absent / Temporal connect fails → exists: False,
    NOT a raise. UI uses this to render 'cron offline' state."""
    from engine.research import l4_cron

    with mock.patch("temporalio.client.Client.connect",
                     new=mock.AsyncMock(side_effect=ConnectionRefusedError())):
        out = asyncio.run(l4_cron.cron_status())
    assert out["schedule"]["exists"] is False
    assert "error" in out["schedule"]
    assert isinstance(out["recent_runs"], list)
