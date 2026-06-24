"""tests/test_emit_doctrine_signal_detected.py — Phase 2.0 step 9a.

Direct tests for emit.doctrine_signal_detected. Verifies:
  - Event type + verdict mapping
  - pattern_name + severity wired into metrics
  - Tags include 'doctrine_signal' + pattern_name
  - Severity = CRITICAL bumps to verdict=RED
  - Unknown severity raises InvalidEventError
  - parent_event_ids propagate (lineage to source events)
  - Subject must already be registered (factor/sleeve etc — real
    consumer registers the subject before emit)
"""
from __future__ import annotations

import pytest

from engine.research_store import emit, registry
from engine.research_store.exceptions import InvalidEventError
from engine.research_store.schema import EventType, SubjectType, Verdict


def _patch_append(monkeypatch):
    """Intercept store.append so tests don't write real events."""
    captured: list = []
    from engine.research_store import store
    monkeypatch.setattr(store, "append", lambda ev: captured.append(ev))
    from engine.research_store import emit as emit_mod
    monkeypatch.setattr(emit_mod, "_publish_to_bus", lambda ev: None)
    return captured


def _ensure_subject(subject_id: str, subject_type=SubjectType.factor):
    try:
        registry.require(subject_id)
    except Exception:
        registry.register_subject(
            subject_id=subject_id, subject_type=subject_type,
            description="(test guard)", created_by="test_emit_doctrine_signal",
        )


# ─────────────────────────────────────────────────────────────────────
# Event shape
# ─────────────────────────────────────────────────────────────────────
def test_emit_writes_doctrine_signal_event_type(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    eid = emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"family": "EARNINGS_DRIFT", "red_count": 3},
        summary="EARNINGS_DRIFT family: 3 RED verdicts in last 30 days",
    )
    assert eid
    assert len(captured) == 1
    ev = captured[0]
    assert ev.event_type == EventType.doctrine_signal_detected


def test_pattern_name_and_severity_in_metrics(monkeypatch):
    """The metrics dict must carry pattern_name + severity so consumers
    can filter by rule without re-deriving from tags."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="sleeve_sharpe_decay",
        metrics={"sleeve_id": "cross_asset_carry", "rolling_sharpe": 0.21},
        summary="cross_asset_carry rolling 6M Sharpe 0.21 < 0.30 floor",
        severity="CRITICAL",
    )
    m = captured[0].metrics
    assert m["pattern_name"] == "sleeve_sharpe_decay"
    assert m["severity"]     == "CRITICAL"
    # Caller's metrics preserved alongside
    assert m["sleeve_id"]     == "cross_asset_carry"
    assert m["rolling_sharpe"] == 0.21


def test_tags_include_doctrine_signal_and_pattern_name(monkeypatch):
    """Downstream queries filter by tags='doctrine_signal' or
    tags=<pattern_name> — both must be present."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="gate_rejection_spike",
        metrics={"reject_rate": 0.85, "n_runs": 20},
        summary="Strict gate reject rate 85% across last 20 runs",
        tags=("ops_attention",),
    )
    ev = captured[0]
    assert "doctrine_signal" in ev.tags
    assert "gate_rejection_spike" in ev.tags
    # Caller's extra tags preserved
    assert "ops_attention" in ev.tags


# ─────────────────────────────────────────────────────────────────────
# Severity → verdict mapping
# ─────────────────────────────────────────────────────────────────────
def test_severity_info_maps_to_neutral(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"red_count": 2},
        summary="2 REDs — not yet a cluster, just noting",
        severity="INFO",
    )
    assert captured[0].verdict == Verdict.NEUTRAL


def test_severity_warn_maps_to_marginal(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"red_count": 3},
        summary="3 REDs — cluster forming",
        severity="WARN",
    )
    assert captured[0].verdict == Verdict.MARGINAL


def test_severity_critical_maps_to_red(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="sleeve_sharpe_decay",
        metrics={"rolling_sharpe": 0.05},
        summary="Sleeve essentially flat — drop everything",
        severity="CRITICAL",
    )
    assert captured[0].verdict == Verdict.RED


def test_default_severity_is_warn(monkeypatch):
    """Most pattern signals are WARN by default — explicit INFO or
    CRITICAL is a deliberate choice."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"red_count": 3},
        summary="cluster",
    )
    assert captured[0].verdict == Verdict.MARGINAL
    assert captured[0].metrics["severity"] == "WARN"


def test_severity_case_insensitive(monkeypatch):
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={},
        summary="x",
        severity="critical",
    )
    assert captured[0].verdict == Verdict.RED
    assert captured[0].metrics["severity"] == "CRITICAL"   # normalized


def test_unknown_severity_raises(monkeypatch):
    """Typos in severity should hard-fail at emit time, NOT silently
    map to NEUTRAL — caller's intent is lost otherwise."""
    _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    with pytest.raises(InvalidEventError):
        emit.doctrine_signal_detected(
            subject_id="auto_test_subject",
            pattern_name="family_red_cluster",
            metrics={},
            summary="x",
            severity="VERY_BAD",   # not a known value
        )


# ─────────────────────────────────────────────────────────────────────
# Lineage
# ─────────────────────────────────────────────────────────────────────
def test_parent_event_ids_propagate(monkeypatch):
    """A pattern signal SHOULD point back to the source events that
    triggered it (e.g. the 3 RED verdict events forming the cluster).
    Lineage queries walk this DAG to find root causes."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"red_count": 3},
        summary="cluster",
        parent_event_ids=("ev_red_1", "ev_red_2", "ev_red_3"),
    )
    assert captured[0].parent_event_ids == ("ev_red_1", "ev_red_2", "ev_red_3")


def test_family_field_set_for_family_pattern(monkeypatch):
    """When the pattern is family-level, family field should be set —
    enables 'all signals in family X' queries."""
    captured = _patch_append(monkeypatch)
    _ensure_subject("auto_test_subject")
    emit.doctrine_signal_detected(
        subject_id="auto_test_subject",
        pattern_name="family_red_cluster",
        metrics={"red_count": 3},
        summary="cluster",
        family="EARNINGS_DRIFT",
    )
    assert captured[0].family == "EARNINGS_DRIFT"


# ─────────────────────────────────────────────────────────────────────
# Subject registration enforcement
# ─────────────────────────────────────────────────────────────────────
def test_unregistered_subject_raises(monkeypatch):
    """The shared _emit() runs registry.require() — subjects MUST be
    registered before emit. Catches typos and missing-registration bugs
    rather than silently inventing a subject."""
    _patch_append(monkeypatch)
    # Force a subject_id that's definitely not registered
    from engine.research_store.exceptions import SubjectNotRegisteredError
    with pytest.raises(SubjectNotRegisteredError):
        emit.doctrine_signal_detected(
            subject_id="definitely_unregistered_subject_xyz_404",
            pattern_name="family_red_cluster",
            metrics={},
            summary="x",
        )
