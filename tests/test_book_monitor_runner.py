"""tests/test_book_monitor_runner.py — Phase 2.0 step 9c.

Runner tests. Mocks filter_events + emit so no real store dependency.
Verifies:
  - Happy path: rules find hits, runner emits, returns event_ids
  - Dedup: matching prior signal suppresses fresh emit
  - force_emit bypasses dedup
  - dry_run skips emit but still returns hits + is_fresh annotation
  - Empty events → no hits, no emit
  - Rule exceptions recorded in errors[], don't kill batch
  - Emit exceptions recorded, partial success on remaining hits
"""
from __future__ import annotations

import datetime as _dt
from types import SimpleNamespace

from engine.research_store.schema import (
    EventType, ResearchEvent, SubjectType, Verdict,
)


_NOW = "2026-06-06T12:00:00Z"


def _ts(days_ago: int) -> str:
    base = _dt.datetime.fromisoformat(_NOW.replace("Z", ""))
    return (base - _dt.timedelta(days=days_ago)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")


def _red(subject_id: str, family: str, days_ago: int, event_id: str | None = None) -> ResearchEvent:
    return ResearchEvent(
        event_id     = event_id or f"ev_{subject_id}",
        event_type   = EventType.factor_verdict_filed,
        ts           = _ts(days_ago),
        session_id   = "test",
        actor        = "test",
        subject_type = SubjectType.factor,
        subject_id   = subject_id,
        verdict      = Verdict.RED,
        metrics      = {},
        artifacts    = {},
        parent_event_ids = (),
        family       = family,
        tags         = (),
        summary      = "test",
        git_sha      = "test",
    )


def _signal(family: str, pattern: str = "family_red_cluster",
              days_ago: int = 1) -> ResearchEvent:
    """Mock prior doctrine_signal_detected event."""
    return ResearchEvent(
        event_id     = f"sig_{family}",
        event_type   = EventType.doctrine_signal_detected,
        ts           = _ts(days_ago),
        session_id   = "test",
        actor        = "engine.agents.book_monitor",
        subject_type = SubjectType.factor,
        subject_id   = f"some_subject_{family}",
        verdict      = Verdict.MARGINAL,
        metrics      = {"pattern_name": pattern, "family": family},
        artifacts    = {},
        parent_event_ids = (),
        family       = family,
        tags         = ("doctrine_signal", pattern),
        summary      = "prior",
        git_sha      = "test",
    )


def _patch_store(monkeypatch, events: list[ResearchEvent]):
    """Patch filter_events to return synthetic events filtered by the
    same predicates the real store uses."""
    from engine.research_store import store

    def _fake_filter_events(event_type=None, subject_type=None, subject_id=None,
                              verdict=None, family=None, since=None, limit=None):
        out = list(events)
        if event_type is not None:
            out = [e for e in out if e.event_type.value == event_type]
        if since is not None:
            out = [e for e in out if e.ts >= since]
        if limit is not None:
            out = out[:limit]
        out.sort(key=lambda e: e.ts, reverse=True)
        return out
    monkeypatch.setattr(store, "filter_events", _fake_filter_events)


def _patch_emit(monkeypatch, fail: bool = False, fail_for_family: str | None = None):
    """Capture emit calls. fail=True raises; fail_for_family raises only
    for matching family (tests partial-success path)."""
    captured: list = []
    from engine.research_store import emit
    def _fake(**kw):
        if fail:
            raise RuntimeError("emit broken")
        if fail_for_family is not None and kw.get("family") == fail_for_family:
            raise RuntimeError(f"emit broken for {fail_for_family}")
        captured.append(kw)
        return f"ev_emit_{len(captured)}"
    monkeypatch.setattr(emit, "doctrine_signal_detected", _fake)
    return captured


# ─────────────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────────────
def test_no_events_returns_clean_zero(monkeypatch):
    from engine.agents.book_monitor.runner import run_book_monitor
    _patch_store(monkeypatch, [])
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_events_scanned"] == 0
    assert result["n_hits_total"] == 0
    assert result["n_hits_fresh"] == 0
    assert result["n_emitted"] == 0
    assert captured == []
    assert result["errors"] == []


def test_fresh_cluster_emits_one_signal(monkeypatch):
    """3 REDs in same family + no prior signal → 1 fresh hit, 1 emit."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "EARNINGS_DRIFT", i) for i in range(3)]
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_total"] == 1
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 1
    assert len(result["event_ids"]) == 1
    assert len(captured) == 1
    assert captured[0]["family"] == "EARNINGS_DRIFT"
    assert captured[0]["pattern_name"] == "family_red_cluster"
    assert captured[0]["severity"] == "WARN"


def test_two_families_clustering_emits_two_signals(monkeypatch):
    from engine.agents.book_monitor.runner import run_book_monitor
    events = (
        [_red(f"a{i}", "EARNINGS_DRIFT", i, event_id=f"ev_a_{i}") for i in range(3)]
        +
        [_red(f"b{i}", "MOMENTUM", i+5, event_id=f"ev_b_{i}") for i in range(3)]
    )
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_total"] == 2
    assert result["n_emitted"] == 2
    families_emitted = {c["family"] for c in captured}
    assert families_emitted == {"EARNINGS_DRIFT", "MOMENTUM"}


# ─────────────────────────────────────────────────────────────────────
# Dedup
# ─────────────────────────────────────────────────────────────────────
def test_dedup_suppresses_matching_prior_signal(monkeypatch):
    """3 REDs in MOMENTUM + a prior MOMENTUM family_red_cluster signal
    within dedup window → hit found but NOT emitted (dedup-suppressed)."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "MOMENTUM", i) for i in range(3)]
    events.append(_signal("MOMENTUM", days_ago=3))
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_total"] == 1
    assert result["n_hits_fresh"] == 0
    assert result["n_emitted"] == 0
    assert captured == []
    # The hit IS reported in result["hits"] but marked is_fresh=False
    assert result["hits"][0]["is_fresh"] is False


def test_dedup_does_not_suppress_different_family(monkeypatch):
    """Prior MOMENTUM signal must NOT dedup a CARRY cluster."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "CARRY", i) for i in range(3)]
    events.append(_signal("MOMENTUM", days_ago=3))
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 1
    assert captured[0]["family"] == "CARRY"


def test_dedup_does_not_suppress_different_rule(monkeypatch):
    """A prior decay-rule signal must NOT dedup a family-cluster hit
    (different rule_name even if same subject)."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "MOMENTUM", i) for i in range(3)]
    events.append(_signal("MOMENTUM", pattern="sleeve_sharpe_decay",
                            days_ago=3))
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 1


def test_old_prior_signal_does_not_suppress(monkeypatch):
    """Prior signal OUTSIDE dedup window → cluster fires again."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "MOMENTUM", i) for i in range(3)]
    events.append(_signal("MOMENTUM", days_ago=30))  # outside default 7d
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor()
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 1


def test_force_emit_bypasses_dedup(monkeypatch):
    """Operator override — re-fire even with matching prior signal."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "MOMENTUM", i) for i in range(3)]
    events.append(_signal("MOMENTUM", days_ago=1))
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor(force_emit=True)
    assert result["n_emitted"] == 1
    assert captured[0]["family"] == "MOMENTUM"
    # All hits marked is_fresh=True under force_emit
    assert all(h["is_fresh"] for h in result["hits"])


# ─────────────────────────────────────────────────────────────────────
# dry_run
# ─────────────────────────────────────────────────────────────────────
def test_dry_run_skips_emit(monkeypatch):
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "EARNINGS_DRIFT", i) for i in range(3)]
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor(dry_run=True)
    assert result["n_hits_total"] == 1
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 0
    assert captured == []


# ─────────────────────────────────────────────────────────────────────
# Failure paths — fail-safe contract
# ─────────────────────────────────────────────────────────────────────
def test_event_load_failure_recorded_not_raised(monkeypatch):
    from engine.agents.book_monitor.runner import run_book_monitor
    from engine.research_store import store
    def _boom(**kw):
        raise RuntimeError("disk error")
    monkeypatch.setattr(store, "filter_events", _boom)
    result = run_book_monitor()
    assert any("load" in e for e in result["errors"])
    assert result["n_hits_total"] == 0


def test_emit_failure_recorded_not_raised(monkeypatch):
    """A broken emit should be caught — caller still gets the hits
    list so they know what would have been emitted."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "EARNINGS_DRIFT", i) for i in range(3)]
    _patch_store(monkeypatch, events)
    _patch_emit(monkeypatch, fail=True)
    result = run_book_monitor()
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 0   # the emit raised
    assert any("emit:" in e for e in result["errors"])


def test_partial_emit_success(monkeypatch):
    """One family's emit raises, the other succeeds — the runner
    persists what it can and records the failure."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = (
        [_red(f"a{i}", "EARNINGS_DRIFT", i, event_id=f"ev_a_{i}") for i in range(3)]
        +
        [_red(f"b{i}", "MOMENTUM", i+5, event_id=f"ev_b_{i}") for i in range(3)]
    )
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch, fail_for_family="MOMENTUM")
    result = run_book_monitor()
    assert result["n_hits_fresh"] == 2
    assert result["n_emitted"] == 1   # EARNINGS_DRIFT only
    assert captured[0]["family"] == "EARNINGS_DRIFT"
    assert any("MOMENTUM" in e for e in result["errors"])


# ─────────────────────────────────────────────────────────────────────
# rule_kwargs pass-through
# ─────────────────────────────────────────────────────────────────────
def test_rule_kwargs_passed_to_rule(monkeypatch):
    """Caller can override per-rule kwargs — e.g. threshold=2 for an
    audit-mode stricter scan."""
    from engine.agents.book_monitor.runner import run_book_monitor
    events = [_red(f"f{i}", "MOMENTUM", i) for i in range(2)]
    _patch_store(monkeypatch, events)
    captured = _patch_emit(monkeypatch)
    result = run_book_monitor(
        rule_kwargs={"family_red_cluster": {"threshold": 2}})
    # With threshold=2, 2 REDs clear the bar
    assert result["n_hits_fresh"] == 1
    assert result["n_emitted"] == 1
