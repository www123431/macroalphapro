"""tests/test_sleeve_strengthen_scan.py — Stage B P3b.

Tests the per-sleeve scan orchestrator. LLM proposer + hypothesis
store + event store are all mocked so tests are offline + fast +
free.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def _make_sleeve_yaml(tmp_path: Path, name: str, **fields) -> Path:
    """Write a sleeve YAML and return its path."""
    base = {
        "_schema_version": 2,
        "id": name,
        "family": "carry",
        "purpose": "deployed_sleeve",
        "canonical_paper_id": "test_paper_2020_jf",
        "canonical_universe": "G10",
        "typical_sample": "2000-present",
        "mechanism_economics": "Test mechanism.",
        "status_in_our_book": "DEPLOYED",
    }
    base.update(fields)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(base), encoding="utf-8")
    return p


def _valid_proposal(**override):
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        StrengthenProposal,
    )
    base = dict(
        claim="Test proposal claim describing concrete improvement.",
        improvement_kind="regime_filter",
        mechanism_subtype="vix_overlay",
        predicted_magnitude="moderate",
        required_data=("VIX daily",),
        test_methodology="engine.validation.decay_sentinel",
        references_paper_ids=(),
        expected_outcome_prior="likely_REJECT_per_HXZ_65pct",
        rationale="VIX surge correlates with carry crashes empirically.",
        generation_ts="2026-06-07T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(override)
    return StrengthenProposal(**base)


@pytest.fixture
def empty_events(monkeypatch):
    """Stub filter_events to return empty for all queries."""
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [])


@pytest.fixture
def stub_save(monkeypatch):
    """Stub save_hypothesis to capture calls without disk I/O."""
    captured = []
    from engine.research_store.hypothesis import store as hyp_st
    def _fake_save(h, path=None, *, validate_strict=True,
                    skip_cross_checks=False):
        captured.append(h)
    monkeypatch.setattr(hyp_st, "save_hypothesis", _fake_save)
    return captured


# ────────────────────────────────────────────────────────────────────
# YAML enumeration + filter
# ────────────────────────────────────────────────────────────────────
def test_load_sleeve_yamls_skips_underscore_prefix(tmp_path):
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _load_sleeve_yamls,
    )
    _make_sleeve_yaml(tmp_path, "good_one")
    # underscore-prefixed file should be skipped
    bad = tmp_path / "_meta.yaml"
    bad.write_text(yaml.safe_dump({"_schema_version": 1}), encoding="utf-8")
    rows = _load_sleeve_yamls(library_dir=tmp_path)
    ids = [r.get("id") for r in rows]
    assert "good_one" in ids
    assert all(not (r.get("_yaml_path") or "").endswith("_meta.yaml")
                 for r in rows)


def test_is_scan_worthy_filters_research_only(tmp_path):
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _is_scan_worthy,
    )
    assert _is_scan_worthy({"status_in_our_book": "DEPLOYED"}) is True
    assert _is_scan_worthy({"status_in_our_book": "RESEARCH"}) is False
    assert _is_scan_worthy({"purpose": "deployed_sleeve"}) is True
    assert _is_scan_worthy({"purpose": "cousin_anchor"}) is True
    assert _is_scan_worthy({}) is False
    assert _is_scan_worthy({"status_in_our_book": "DECOMMISSIONED"}) is False


# ────────────────────────────────────────────────────────────────────
# ISO week helper (idempotency key)
# ────────────────────────────────────────────────────────────────────
def test_iso_week_id_format():
    import datetime as _dt
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _iso_week_id,
    )
    # A known ISO week — 2026-06-07 is a Sunday, ISO week 23
    s = _iso_week_id(_dt.datetime(2026, 6, 7))
    assert s == "2026-W23"
    # Format check on default (no arg)
    s2 = _iso_week_id()
    assert s2.startswith("2026-W")


# ────────────────────────────────────────────────────────────────────
# Context builder
# ────────────────────────────────────────────────────────────────────
def test_build_context_pulls_recent_events(tmp_path, monkeypatch):
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    from engine.research_store import store as st

    red_events = [
        SimpleNamespace(event_id="ev_red_1"),
        SimpleNamespace(event_id="ev_red_2"),
    ]
    decay_events = [SimpleNamespace(event_id="ev_decay_1")]
    def _fake_filter(**kw):
        if kw.get("event_type") == "factor_verdict_filed":
            return list(red_events)
        if kw.get("event_type") == "doctrine_signal_detected":
            return list(decay_events)
        return []
    monkeypatch.setattr(st, "filter_events", _fake_filter)

    ctx = scan._build_context({
        "id": "test_sleeve",
        "family": "carry",
        "canonical_paper_id": "test_paper",
        "mechanism_economics": "x",
        "canonical_universe": "G10",
        "typical_sample": "2000-present",
        "purpose": "deployed_sleeve",
        "status_in_our_book": "DEPLOYED",
    })
    assert ctx.sleeve_id == "test_sleeve"
    assert ctx.recent_family_red_ids == ("ev_red_1", "ev_red_2")
    assert ctx.recent_decay_alert_ids == ("ev_decay_1",)
    assert "deployed_sleeve" in ctx.deployed_summary
    assert "DEPLOYED" in ctx.deployed_summary


def test_build_context_handles_event_store_failure(monkeypatch):
    """filter_events raising → context built with empty event ids."""
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    from engine.research_store import store as st
    def _broken(**kw):
        raise RuntimeError("store unreachable")
    monkeypatch.setattr(st, "filter_events", _broken)
    ctx = scan._build_context({
        "id": "test", "family": "carry",
        "canonical_paper_id": "p",
        "mechanism_economics": "x",
        "canonical_universe": "u", "typical_sample": "s",
        "purpose": "deployed_sleeve",
    })
    assert ctx.recent_family_red_ids == ()
    assert ctx.recent_decay_alert_ids == ()


# ────────────────────────────────────────────────────────────────────
# Proposal → Hypothesis adapter
# ────────────────────────────────────────────────────────────────────
def test_proposal_to_hypothesis_validates():
    """Adapted Hypothesis must pass schema validation."""
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        SleeveContext,
    )
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _proposal_to_hypothesis,
    )
    ctx = SleeveContext(
        sleeve_id="cross_asset_carry", family="CARRY",
        canonical_paper_id="kmpv_2018",
        mechanism_economics="x", canonical_universe="G10",
        typical_sample="2000-present", deployed_summary="DEPLOYED",
        snapshot_ts="2026-06-07T00:00:00Z",
    )
    h = _proposal_to_hypothesis(_valid_proposal(), ctx=ctx)
    errs = h.validate()
    assert errs == [], f"Hypothesis failed validate: {errs}"


def test_proposal_to_hypothesis_carries_provenance():
    """Provenance fields must be set so future attribution can JOIN."""
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        SleeveContext,
    )
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _proposal_to_hypothesis,
    )
    ctx = SleeveContext(
        sleeve_id="cross_asset_carry", family="CARRY",
        canonical_paper_id="kmpv_2018",
        mechanism_economics="x", canonical_universe="G10",
        typical_sample="2000-present", deployed_summary="DEPLOYED",
        recent_family_red_ids=("ev_red_1",),
        recent_decay_alert_ids=("ev_alert_1",),
        snapshot_ts="2026-06-07T00:00:00Z",
    )
    prop = _valid_proposal(
        improvement_kind="cost_aware_exec",
        references_paper_ids=("extra_paper_2024",),
    )
    h = _proposal_to_hypothesis(prop, ctx=ctx)
    assert h.addresses_decay_in == "cross_asset_carry"
    assert "source:active_b_sleeve_scan" in h.tags
    assert "sleeve:cross_asset_carry" in h.tags
    assert "improvement_kind:cost_aware_exec" in h.tags
    # canonical + extra both in synthesizes_paper_ids
    assert "kmpv_2018" in h.synthesizes_paper_ids
    assert "extra_paper_2024" in h.synthesizes_paper_ids
    # Recent event ids linked
    assert "ev_red_1" in h.synthesizes_event_ids
    assert "ev_alert_1" in h.synthesizes_event_ids


# ────────────────────────────────────────────────────────────────────
# Idempotency state
# ────────────────────────────────────────────────────────────────────
def test_scan_state_roundtrip(tmp_path):
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _load_scan_state, _save_scan_state,
    )
    state = {"sleeve_a": "2026-W23", "sleeve_b": "2026-W22"}
    _save_scan_state(state, state_dir=tmp_path)
    assert _load_scan_state(state_dir=tmp_path) == state


def test_scan_state_missing_returns_empty(tmp_path):
    from engine.agents.strengthener.sleeve_strengthen_scan import (
        _load_scan_state,
    )
    assert _load_scan_state(state_dir=tmp_path) == {}


# ────────────────────────────────────────────────────────────────────
# Full orchestration — happy path
# ────────────────────────────────────────────────────────────────────
def test_scan_happy_path(tmp_path, monkeypatch, empty_events, stub_save):
    """3 deployed sleeves, proposer returns 2 proposals each → 6
    hypotheses persisted, state file updated."""
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "sleeve_a")
    _make_sleeve_yaml(tmp_path, "sleeve_b")
    _make_sleeve_yaml(tmp_path, "sleeve_c")
    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [_valid_proposal(), _valid_proposal()])

    state_dir = tmp_path / "state"
    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = state_dir,
        max_sleeves = 10,
    )
    assert r["n_sleeves_eligible"] == 3
    assert r["n_sleeves_scanned"] == 3
    assert r["n_proposals_total"] == 6
    assert r["n_proposals_persisted"] == 6
    assert len(stub_save) == 6

    # State file has all 3 sleeves stamped with this week
    persisted_state = json.loads(
        (state_dir / "scanned_weeks.json").read_text(encoding="utf-8"))
    assert set(persisted_state.keys()) == {"sleeve_a", "sleeve_b",
                                              "sleeve_c"}


def test_scan_respects_max_sleeves_cap(tmp_path, monkeypatch,
                                          empty_events, stub_save):
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    for i in range(5):
        _make_sleeve_yaml(tmp_path, f"sleeve_{i}")
    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [_valid_proposal()])

    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = tmp_path / "state",
        max_sleeves = 2,
    )
    assert r["n_sleeves_scanned"] == 2


def test_scan_skips_already_scanned_this_week(tmp_path, monkeypatch,
                                                 empty_events, stub_save):
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "sleeve_a")
    _make_sleeve_yaml(tmp_path, "sleeve_b")

    # Pre-seed state: sleeve_a already scanned THIS week
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    iso_week = scan._iso_week_id()
    (state_dir / "scanned_weeks.json").write_text(
        json.dumps({"sleeve_a": iso_week}), encoding="utf-8")

    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [_valid_proposal()])

    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = state_dir,
        max_sleeves = 10,
    )
    # sleeve_a skipped (already this week), sleeve_b scanned
    assert r["n_sleeves_scanned"] == 1
    assert r["n_sleeves_skipped"] == 1


def test_scan_force_overrides_dedup(tmp_path, monkeypatch,
                                       empty_events, stub_save):
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "sleeve_a")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    iso_week = scan._iso_week_id()
    (state_dir / "scanned_weeks.json").write_text(
        json.dumps({"sleeve_a": iso_week}), encoding="utf-8")

    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [_valid_proposal()])

    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = state_dir,
        force       = True,
    )
    assert r["n_sleeves_scanned"] == 1
    assert r["n_sleeves_skipped"] == 0


def test_scan_dry_run_skips_persist(tmp_path, monkeypatch,
                                       empty_events, stub_save):
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "sleeve_a")
    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [_valid_proposal()])

    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = tmp_path / "state",
        dry_run     = True,
    )
    assert r["n_proposals_total"] == 1
    assert r["n_proposals_persisted"] == 0
    assert stub_save == []


def test_scan_proposer_exception_isolates_per_sleeve(
    tmp_path, monkeypatch, empty_events, stub_save,
):
    """Proposer raising for one sleeve → others still process; error
    captured per-sleeve."""
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "good")
    _make_sleeve_yaml(tmp_path, "broken")

    def _fake_proposer(ctx):
        if ctx.sleeve_id == "broken":
            raise RuntimeError("anthropic 500")
        return [_valid_proposal()]
    monkeypatch.setattr(scan, "run_strengthen_proposer", _fake_proposer)

    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = tmp_path / "state",
    )
    # 'broken' raised → counted as eligible but proposer failure does
    # NOT increment n_sleeves_scanned (we only count successful
    # context+proposer pairs). Good sleeve still went through.
    assert r["n_sleeves_scanned"] == 1
    assert any("broken: proposer" in e for e in r["errors"])
    assert len(stub_save) == 1


def test_scan_empty_proposer_returns_zero_marks_state(
    tmp_path, monkeypatch, empty_events, stub_save,
):
    """If proposer returns [] (no proposals), sleeve is still marked
    as scanned this week so we don't re-burn LLM budget on it."""
    from engine.agents.strengthener import (
        sleeve_strengthen_scan as scan,
    )
    _make_sleeve_yaml(tmp_path, "healthy_sleeve")
    monkeypatch.setattr(scan, "run_strengthen_proposer",
        lambda ctx: [])

    state_dir = tmp_path / "state"
    r = scan.run_sleeve_strengthen_scan(
        library_dir = tmp_path,
        state_dir   = state_dir,
    )
    assert r["n_proposals_total"] == 0
    assert r["n_sleeves_scanned"] == 1
    state = json.loads(
        (state_dir / "scanned_weeks.json").read_text(encoding="utf-8"))
    assert "healthy_sleeve" in state
