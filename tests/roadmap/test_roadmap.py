"""Tests for engine.roadmap — YAML-persisted research-axis registry."""
from __future__ import annotations

import pytest


@pytest.fixture
def isolated_roadmap(tmp_path, monkeypatch):
    """Redirect axes.yaml to tmp."""
    rdir = tmp_path / "roadmap"
    rdir.mkdir(parents=True, exist_ok=True)
    from engine.roadmap import store
    monkeypatch.setattr(store, "_AXES_PATH", rdir / "axes.yaml")
    yield {"rdir": rdir}


def test_upsert_then_get(isolated_roadmap):
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier

    a = store.upsert_axis(
        axis_id="test_axis_a",
        name="Test axis A",
        state=AxisState.active,
        tier=AxisTier.committed,
        rationale="Test rationale",
        family="test_family",
        next_actions=("step 1", "step 2"),
    )
    assert a.axis_id == "test_axis_a"
    assert a.state == AxisState.active
    assert a.tier == AxisTier.committed
    assert a.created_ts
    assert a.updated_ts == a.created_ts   # first insert

    got = store.get_axis("test_axis_a")
    assert got is not None
    assert got.name == "Test axis A"
    assert got.next_actions == ("step 1", "step 2")


def test_upsert_preserves_created_ts(isolated_roadmap):
    """Re-upserting an existing axis updates updated_ts but preserves
    created_ts + created_by."""
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier

    first = store.upsert_axis(
        axis_id="ax", name="Test", state=AxisState.active,
        tier=AxisTier.candidate, rationale="initial",
        actor="user-A",
    )
    import time; time.sleep(1.1)    # ensure ts differs
    second = store.upsert_axis(
        axis_id="ax", name="Test (updated)", state=AxisState.paused,
        tier=AxisTier.candidate, rationale="paused for review",
        actor="user-B",
    )
    assert second.created_ts == first.created_ts
    assert second.created_by == "user-A"
    assert second.updated_ts != first.updated_ts
    assert second.updated_by == "user-B"
    assert second.name == "Test (updated)"
    assert second.state.value == "paused"


def test_list_filtered_by_state(isolated_roadmap):
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier

    store.upsert_axis("a1", "A1", AxisState.active, AxisTier.committed, "r")
    store.upsert_axis("a2", "A2", AxisState.queued, AxisTier.committed, "r")
    store.upsert_axis("a3", "A3", AxisState.closed, AxisTier.committed, "r")

    actives = store.list_axes(state=AxisState.active)
    assert len(actives) == 1
    assert actives[0].axis_id == "a1"

    queued = store.list_axes(state="queued")
    assert len(queued) == 1


def test_list_sorts_active_first(isolated_roadmap):
    """list_axes() sort order: active > queued > paused > closed."""
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier

    store.upsert_axis("a_closed", "A", AxisState.closed, AxisTier.committed, "r")
    store.upsert_axis("a_active", "A", AxisState.active, AxisTier.committed, "r")
    store.upsert_axis("a_queued", "A", AxisState.queued, AxisTier.committed, "r")
    store.upsert_axis("a_paused", "A", AxisState.paused, AxisTier.committed, "r")

    listed = store.list_axes()
    states = [a.state.value for a in listed]
    assert states == ["active", "queued", "paused", "closed"]


def test_delete(isolated_roadmap):
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier

    store.upsert_axis("doomed", "x", AxisState.closed, AxisTier.scratchpad, "r")
    assert store.get_axis("doomed") is not None
    assert store.delete_axis("doomed") is True
    assert store.get_axis("doomed") is None
    assert store.delete_axis("doomed") is False   # idempotent


def test_decay_estimate_attachment(isolated_roadmap):
    """Gap B integration: axis can carry a cached decay estimate dict."""
    from engine.roadmap import store
    from engine.roadmap.schema import AxisState, AxisTier
    from engine.decay_forecast import estimate_for_family

    estimate = estimate_for_family("earnings_underreaction")
    a = store.upsert_axis(
        "with_decay", "Test", AxisState.active, AxisTier.committed, "r",
        family="earnings_underreaction",
        decay_estimate=estimate.to_dict(),
    )
    got = store.get_axis("with_decay")
    assert got.decay_estimate is not None
    assert got.decay_estimate["risk"] == "HIGH"


def test_serde_roundtrip(isolated_roadmap):
    from engine.roadmap import store
    from engine.roadmap.schema import (
        AxisState, AxisTier, AxisOutcome,
    )

    a = store.upsert_axis(
        "rt", "Round-trip", AxisState.closed, AxisTier.committed,
        "test rationale",
        outcome=AxisOutcome.GREEN,
        parent_axis_id="parent_x",
        family="carry",
        related_subject_ids=("subj_1", "subj_2"),
        related_memory_files=("mem_x", "mem_y"),
        next_actions=("step A", "step B"),
        blocking_notes="none",
    )
    got = store.get_axis("rt")
    assert got == a
