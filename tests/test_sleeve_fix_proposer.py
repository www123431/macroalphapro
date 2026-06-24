"""tests/test_sleeve_fix_proposer.py — Stage B P2 piece 1.

Tests deterministic template-based D→B coupling. Events stubbed via
SimpleNamespace + filter_events / save_hypothesis monkeypatched so
tests are offline + fast.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _sig(*, event_id="ev_sig_1", subject_id="cross_asset_carry",
          family="CARRY", pattern_name="sleeve_sharpe_decay",
          summary="Sharpe below floor", severity="WARN",
          ts="2026-06-07T10:00:00Z", extra_metrics=None):
    metrics = {"pattern_name": pattern_name, "severity": severity}
    if extra_metrics:
        metrics.update(extra_metrics)
    return SimpleNamespace(
        event_id   = event_id,
        event_type = "doctrine_signal_detected",
        subject_id = subject_id,
        family     = family,
        ts         = ts,
        summary    = summary,
        metrics    = metrics,
    )


@pytest.fixture
def stub_filter_events(monkeypatch):
    """Yield a list to be returned by filter_events; tests mutate it."""
    signals: list = []
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: list(signals)
                            if kw.get("event_type") ==
                                 "doctrine_signal_detected"
                            else [])
    return signals


@pytest.fixture
def stub_save(monkeypatch):
    """Capture save_hypothesis calls; never touch disk."""
    captured: list = []
    from engine.research_store.hypothesis import store as hyp_st
    def _fake_save(h, path=None, *, validate_strict=True,
                    skip_cross_checks=False):
        captured.append(h)
    monkeypatch.setattr(hyp_st, "save_hypothesis", _fake_save)
    return captured


@pytest.fixture
def empty_already(monkeypatch):
    """Stub _already_proposed_event_ids → empty set."""
    from engine.agents.strengthener import sleeve_fix_proposer as sfp
    monkeypatch.setattr(sfp, "_already_proposed_event_ids",
                          lambda **kw: set())


# ────────────────────────────────────────────────────────────────────
# Pattern templates
# ────────────────────────────────────────────────────────────────────
def test_sleeve_sharpe_decay_template():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        build_fix_hypothesis_from_signal,
    )
    h = build_fix_hypothesis_from_signal(_sig(
        pattern_name="sleeve_sharpe_decay",
        subject_id="cross_asset_carry",
    ))
    assert h.addresses_decay_in == "cross_asset_carry"
    assert "Sharpe" in h.claim or "decay" in h.claim.lower()
    assert h.mechanism_subtype == "sleeve_decay_response"
    assert h.synthesizes_event_ids == ("ev_sig_1",)
    assert h.extraction_method.value == "llm_synthesis"


def test_family_red_cluster_template():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        build_fix_hypothesis_from_signal,
    )
    h = build_fix_hypothesis_from_signal(_sig(
        pattern_name="family_red_cluster",
        subject_id="auto_aaa123",      # spec-hash, NOT a sleeve
        family="MOMENTUM",
        extra_metrics={"n_red": 5},
    ))
    # auto_<hash> is NOT a sleeve id → addresses_decay_in stays None
    assert h.addresses_decay_in is None
    assert "MOMENTUM" in h.claim or "5" in h.claim
    assert h.mechanism_subtype == "family_pause_proposal"


def test_gate_rejection_spike_template():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        build_fix_hypothesis_from_signal,
    )
    h = build_fix_hypothesis_from_signal(_sig(
        pattern_name="gate_rejection_spike",
        extra_metrics={"n_red": 12},
    ))
    assert h.mechanism_subtype == "filter_tighten_proposal"
    assert "reject" in h.claim.lower() or "12" in h.claim


def test_unknown_pattern_falls_through_to_generic():
    """An unrecognized pattern_name still produces a Hypothesis (better
    to surface than to silently drop)."""
    from engine.agents.strengthener.sleeve_fix_proposer import (
        build_fix_hypothesis_from_signal,
    )
    h = build_fix_hypothesis_from_signal(_sig(
        pattern_name="weird_new_pattern",
    ))
    assert h.mechanism_subtype == "unknown_signal_response"
    assert "weird_new_pattern" in h.claim


# ────────────────────────────────────────────────────────────────────
# subject_id → sleeve_id heuristic
# ────────────────────────────────────────────────────────────────────
def test_sleeve_heuristic_accepts_lowercase_underscores():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _looks_like_sleeve_id,
    )
    assert _looks_like_sleeve_id("cross_asset_carry") is True
    assert _looks_like_sleeve_id("time_series_momentum") is True


def test_sleeve_heuristic_rejects_auto_hash():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _looks_like_sleeve_id,
    )
    assert _looks_like_sleeve_id("auto_abc123def") is False


def test_sleeve_heuristic_rejects_uppercase():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _looks_like_sleeve_id,
    )
    assert _looks_like_sleeve_id("CARRY") is False


def test_sleeve_heuristic_rejects_empty():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _looks_like_sleeve_id,
    )
    assert _looks_like_sleeve_id("") is False
    assert _looks_like_sleeve_id(None) is False


# ────────────────────────────────────────────────────────────────────
# Family mapping
# ────────────────────────────────────────────────────────────────────
def test_family_resolves_known():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _resolve_family,
    )
    from engine.research_store.red_lessons.mechanism_families import (
        MechanismFamily,
    )
    assert _resolve_family("CARRY") == MechanismFamily.CARRY
    assert _resolve_family("carry") == MechanismFamily.CARRY


def test_family_resolves_unknown_to_other():
    from engine.agents.strengthener.sleeve_fix_proposer import (
        _resolve_family,
    )
    from engine.research_store.red_lessons.mechanism_families import (
        MechanismFamily,
    )
    assert _resolve_family("NONSENSE") == MechanismFamily.OTHER
    assert _resolve_family(None) == MechanismFamily.OTHER
    assert _resolve_family("") == MechanismFamily.OTHER


# ────────────────────────────────────────────────────────────────────
# propose_sleeve_fixes orchestration
# ────────────────────────────────────────────────────────────────────
def test_propose_basic(stub_filter_events, stub_save, empty_already):
    """3 signals, none seen before → 3 fix hypotheses persisted."""
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    stub_filter_events.extend([
        _sig(event_id="ev_1"),
        _sig(event_id="ev_2", subject_id="time_series_momentum"),
        _sig(event_id="ev_3", pattern_name="family_red_cluster"),
    ])
    r = propose_sleeve_fixes()
    assert r["n_signals_seen"] == 3
    assert r["n_proposed"] == 3
    assert r["n_persisted"] == 3
    assert r["n_already_done"] == 0
    assert len(stub_save) == 3


def test_propose_idempotent(stub_filter_events, stub_save, monkeypatch):
    """Signals already linked to a hypothesis → skipped."""
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    from engine.agents.strengthener import sleeve_fix_proposer as sfp
    monkeypatch.setattr(sfp, "_already_proposed_event_ids",
                          lambda **kw: {"ev_1", "ev_3"})
    stub_filter_events.extend([
        _sig(event_id="ev_1"),
        _sig(event_id="ev_2"),
        _sig(event_id="ev_3"),
    ])
    r = propose_sleeve_fixes()
    assert r["n_signals_seen"] == 3
    assert r["n_already_done"] == 2
    assert r["n_proposed"] == 1
    assert len(stub_save) == 1


def test_propose_dry_run_skips_persist(stub_filter_events, stub_save,
                                          empty_already):
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    stub_filter_events.append(_sig(event_id="ev_1"))
    r = propose_sleeve_fixes(dry_run=True)
    assert r["n_proposed"] == 1
    assert r["n_persisted"] == 0
    assert stub_save == []


def test_propose_respects_max_signals(stub_filter_events, stub_save,
                                          empty_already):
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    for i in range(10):
        stub_filter_events.append(_sig(event_id=f"ev_{i}"))
    r = propose_sleeve_fixes(max_signals=3)
    assert r["n_proposed"] == 3
    assert r["n_persisted"] == 3


def test_propose_empty_signals_returns_clean(stub_filter_events,
                                                stub_save, empty_already):
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    r = propose_sleeve_fixes()
    assert r["n_signals_seen"] == 0
    assert r["n_proposed"] == 0
    assert r["errors"] == []


def test_propose_persist_failure_recorded(stub_filter_events,
                                              empty_already,
                                              monkeypatch):
    """save_hypothesis raising → error captured, run continues."""
    from engine.agents.strengthener.sleeve_fix_proposer import (
        propose_sleeve_fixes,
    )
    from engine.research_store.hypothesis import store as hyp_st
    def _broken_save(*a, **kw):
        raise RuntimeError("disk full")
    monkeypatch.setattr(hyp_st, "save_hypothesis", _broken_save)

    stub_filter_events.extend([
        _sig(event_id="ev_a"),
        _sig(event_id="ev_b"),
    ])
    r = propose_sleeve_fixes()
    assert r["n_proposed"] == 2
    assert r["n_persisted"] == 0
    assert len(r["errors"]) == 2
    assert all("persist:" in e for e in r["errors"])


# ────────────────────────────────────────────────────────────────────
# Hypothesis schema compliance (validates against the store's rules)
# ────────────────────────────────────────────────────────────────────
def test_proposed_hypothesis_validates():
    """The built Hypothesis must pass its own validate() — otherwise
    save_hypothesis(strict=True) would refuse it."""
    from engine.agents.strengthener.sleeve_fix_proposer import (
        build_fix_hypothesis_from_signal,
    )
    h = build_fix_hypothesis_from_signal(_sig())
    errs = h.validate()
    assert errs == [], f"Hypothesis failed validate(): {errs}"
