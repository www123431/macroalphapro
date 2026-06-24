"""tests/test_factor_verdict_emit.py — Tier C-2c.

Tests for engine.agents.strengthener.factor_verdict_emit: ensures
GREEN/MARGINAL/RED template results emit factor_verdict_filed
events with the right shape, dispatcher-internal verdicts don't
emit, subject auto-registration is idempotent, and the capability
evidence file is written before emit (contract).

emit + registry calls are NOT mocked — they hit the real research
store. Each test runs in a tmp evidence dir + uses unique
hypothesis_ids so events don't collide. Events ARE persisted to
the real events.jsonl, which is acceptable since tier_c_auto tags
make them easily filterable + skip-able by other consumers.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest


def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_" + uuid.uuid4().hex[:8],
        signal_kind="time_series_momentum",
        universe="us_equities_sector_etf",
        date_range="2020-01:2024-12",
        signal_inputs=("etf.adj_close.spy",),
        rebal="weekly",
        weighting="signed_signal_volatility_targeted",
        expected_holding_period="weekly",
        min_obs_months=24,
        pit_audits=("lookahead",),
        cost_model="engine.execution.cost_model.basic",
        rationale="test",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


def _tpl_result(verdict="GREEN", t_stat=2.5, sharpe=1.20):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    return TemplateResult(
        verdict          = verdict,
        summary          = (f"TSMOM test result: Sharpe={sharpe:.2f}, "
                              f"t={t_stat:.2f} → {verdict}"),
        metrics          = {
            "sharpe": sharpe, "nw_t_stat": t_stat, "n_obs_months": 60,
            "n_tickers": 35, "ann_return": 0.08, "ann_vol": 0.10,
            "n_trials": 1,
        },
        artifacts        = {},
        template_version = "v1.0_test",
    )


@pytest.fixture
def tmp_evidence(tmp_path, monkeypatch):
    """Redirect EVIDENCE_DIR + registry + event store + PNL_DIR to
    tmp so tests don't pollute docs/capability_evidence/tier_c_auto/,
    data/research_store/subjects.yaml, events.jsonl, or
    data/research_store/tier_c_pnl/.

    Found 2026-06-08: original fixture only redirected EVIDENCE_DIR
    and tests were silently writing 5 throwaway subjects + 8
    events to the real store. Cleaned up + this fixture now
    isolates ALL persistence including L2-4 parquet artifacts."""
    from engine.agents.strengthener import factor_verdict_emit as fve
    from engine.research_store import registry as reg
    from engine.research_store import store as st

    tmp_dir = tmp_path / "evidence"
    monkeypatch.setattr(fve, "EVIDENCE_DIR", tmp_dir)
    monkeypatch.setattr(fve, "PNL_DIR", tmp_path / "tier_c_pnl")
    # Registry: subjects.yaml + aliases.yaml
    monkeypatch.setattr(reg, "_SUBJECTS_PATH",
                          tmp_path / "subjects.yaml")
    monkeypatch.setattr(reg, "_ALIASES_PATH",
                          tmp_path / "aliases.yaml")
    # Event store: events.jsonl
    monkeypatch.setattr(st, "_EVENTS_PATH",
                          tmp_path / "events.jsonl")
    return tmp_dir


# ────────────────────────────────────────────────────────────────────
# L2-4 prep: PnL persistence
# ────────────────────────────────────────────────────────────────────
def _tpl_with_pnl(verdict="GREEN"):
    """Template result with a small PnL DataFrame in artifacts."""
    import pandas as pd
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    df = pd.DataFrame({
        "pnl_gross":    [0.01] * 60,
        "pnl_net_13bp": [0.009] * 60,
        "pnl_net_80bp": [0.005] * 60,
        "turnover":     [0.5] * 60,
    }, index=idx)
    return TemplateResult(
        verdict          = verdict,
        summary          = "pnl-bearing fixture",
        metrics          = {"sharpe": 1.0, "nw_t_stat": 2.5, "n_obs_months": 60},
        artifacts        = {"pnl_series_df": df},
        template_version = "fixture_v1",
    )


def test_write_pnl_parquet_persists_dataframe(tmp_evidence, tmp_path):
    from engine.agents.strengthener.factor_verdict_emit import (
        write_pnl_parquet, PNL_DIR,
    )
    spec = _spec(hypothesis_id="hid_pnl_happy")
    path = write_pnl_parquet(spec, _tpl_with_pnl(verdict="GREEN"))
    assert path is not None
    assert path.exists()
    assert path.suffix == ".parquet"
    assert path.name.endswith("_GREEN.parquet")
    # Round-trip read
    import pandas as pd
    df = pd.read_parquet(path)
    assert set(df.columns) >= {"date", "pnl_gross", "pnl_net_13bp",
                                  "pnl_net_80bp", "turnover"}
    assert len(df) == 60


def test_write_pnl_parquet_returns_none_when_no_artifact(tmp_evidence):
    from engine.agents.strengthener.factor_verdict_emit import (
        write_pnl_parquet,
    )
    spec = _spec(hypothesis_id="hid_pnl_missing")
    # Default template fixture has artifacts={} — no pnl_series_df
    result = write_pnl_parquet(spec, _tpl_result(verdict="GREEN"))
    assert result is None


def test_write_pnl_parquet_returns_none_on_empty_df(tmp_evidence):
    import pandas as pd
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    from engine.agents.strengthener.factor_verdict_emit import (
        write_pnl_parquet,
    )
    spec = _spec(hypothesis_id="hid_pnl_empty")
    tr = TemplateResult(
        verdict="GREEN", summary="", metrics={},
        artifacts={"pnl_series_df": pd.DataFrame()},
        template_version="v1",
    )
    assert write_pnl_parquet(spec, tr) is None


def test_emit_records_parquet_path_in_metrics_and_artifacts(tmp_evidence):
    """End-to-end: emit_tier_c_verdict with a pnl-bearing template
    must record the parquet path in BOTH event.metrics
    (pnl_series_parquet) and event.artifacts (pnl_series_parquet)."""
    from engine.agents.strengthener.factor_verdict_emit import (
        emit_tier_c_verdict,
    )
    from engine.research_store.store import filter_events

    spec = _spec(hypothesis_id="hid_pnl_e2e",
                   signal_kind="cross_sectional_rank",
                   signal_inputs=("compustat.funda.gp_at",))
    eid = emit_tier_c_verdict(spec, "PROFITABILITY",
                                  _tpl_with_pnl(verdict="GREEN"))
    assert eid
    # Retrieve the event
    evs = [e for e in filter_events(event_type="factor_verdict_filed")
            if e.event_id == eid]
    assert len(evs) == 1
    ev = evs[0]
    assert ev.metrics["pnl_series_parquet"]
    assert ev.metrics["pnl_series_parquet"].endswith(".parquet")
    # artifacts is a tuple of (key, value) pairs in the schema —
    # check it's there as a key
    art_dict = dict(ev.artifacts or ())
    assert "pnl_series_parquet" in art_dict
    assert art_dict["pnl_series_parquet"] == ev.metrics["pnl_series_parquet"]


def test_emit_succeeds_when_pnl_write_fails(tmp_evidence, monkeypatch):
    """PnL persistence MUST be best-effort. If write_pnl_parquet
    raises (e.g. pyarrow missing), the verdict event STILL emits."""
    from engine.agents.strengthener import factor_verdict_emit as fve
    from engine.research_store.store import filter_events

    def _crash(spec, tr):
        raise RuntimeError("pyarrow boom")
    monkeypatch.setattr(fve, "write_pnl_parquet", _crash)

    spec = _spec(hypothesis_id="hid_pnl_resilience",
                   signal_kind="cross_sectional_rank",
                   signal_inputs=("compustat.funda.gp_at",))
    eid = fve.emit_tier_c_verdict(spec, "PROFITABILITY",
                                       _tpl_with_pnl(verdict="GREEN"))
    assert eid, "verdict event MUST land even when PnL persist crashes"
    evs = [e for e in filter_events(event_type="factor_verdict_filed")
            if e.event_id == eid]
    ev = evs[0]
    # pnl_series_parquet absent from metrics + artifacts on crash
    assert "pnl_series_parquet" not in (ev.metrics or {})
    assert "pnl_series_parquet" not in dict(ev.artifacts or ())


# ────────────────────────────────────────────────────────────────────
# Subject identity
# ────────────────────────────────────────────────────────────────────
def test_auto_subject_id_deterministic():
    from engine.agents.strengthener.factor_verdict_emit import (
        auto_subject_id,
    )
    s = _spec(hypothesis_id="hid_aaaaaaaa-bbbb-cccc",
                signal_kind="time_series_momentum")
    assert auto_subject_id(s) == "tier_c_auto_hid_aaaa_time_series_momentum"
    # Same hypothesis_id + same signal_kind → same subject
    s2 = _spec(hypothesis_id="hid_aaaaaaaa-bbbb-cccc",
                 signal_kind="time_series_momentum",
                 date_range="1999-01:2024-12")   # different DR
    assert auto_subject_id(s) == auto_subject_id(s2)


def test_auto_subject_id_changes_with_signal_kind():
    from engine.agents.strengthener.factor_verdict_emit import (
        auto_subject_id,
    )
    s_a = _spec(hypothesis_id="x", signal_kind="time_series_momentum")
    s_b = _spec(hypothesis_id="x", signal_kind="carry")
    assert auto_subject_id(s_a) != auto_subject_id(s_b)


def test_ensure_subject_registered_is_idempotent():
    from engine.agents.strengthener.factor_verdict_emit import (
        ensure_subject_registered, auto_subject_id,
    )
    from engine.research_store import registry
    s = _spec(hypothesis_id="hid_idem_" + uuid.uuid4().hex[:6])
    sid_first  = ensure_subject_registered(s, family_hint="MOMENTUM")
    sid_second = ensure_subject_registered(s, family_hint="MOMENTUM")
    assert sid_first == sid_second
    # Confirm it's actually in the registry
    subj = registry.resolve(sid_first)
    assert subj is not None
    assert subj.subject_type == "factor"
    assert subj.family == "MOMENTUM"


# ────────────────────────────────────────────────────────────────────
# Capability evidence stub
# ────────────────────────────────────────────────────────────────────
def test_write_capability_evidence_creates_file(tmp_evidence):
    from engine.agents.strengthener.factor_verdict_emit import (
        write_capability_evidence,
    )
    s = _spec()
    tr = _tpl_result()
    p = write_capability_evidence(s, tr, dispatch_event_id="ev_x",
                                       family_hint="MOMENTUM")
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "GREEN" in body
    assert "tier_c_auto" in body
    assert "MOMENTUM" in body
    assert s.hypothesis_id in body
    assert "Sharpe=" in body


def test_write_capability_evidence_overwrites_same_spec(tmp_evidence):
    """Same spec_hash → same file. Re-dispatching overwrites (most
    recent run is the relevant evidence)."""
    from engine.agents.strengthener.factor_verdict_emit import (
        write_capability_evidence,
    )
    s = _spec()
    p1 = write_capability_evidence(s, _tpl_result(verdict="GREEN"),
                                        dispatch_event_id="ev_1",
                                        family_hint="X")
    p2 = write_capability_evidence(s, _tpl_result(verdict="GREEN",
                                                       t_stat=3.5,
                                                       sharpe=1.80),
                                        dispatch_event_id="ev_2",
                                        family_hint="X")
    assert p1 == p2   # same spec_hash + same verdict → same path
    body = p2.read_text(encoding="utf-8")
    assert "1.80" in body and "ev_2" in body


def test_write_capability_evidence_differs_per_verdict(tmp_evidence):
    """Same spec, different verdict (re-run on extended data) →
    separate file, no collision."""
    from engine.agents.strengthener.factor_verdict_emit import (
        write_capability_evidence,
    )
    s = _spec()
    p_g = write_capability_evidence(s, _tpl_result(verdict="GREEN"),
                                         dispatch_event_id="ev_g",
                                         family_hint="X")
    p_r = write_capability_evidence(s, _tpl_result(verdict="RED",
                                                        t_stat=0.5,
                                                        sharpe=0.10),
                                         dispatch_event_id="ev_r",
                                         family_hint="X")
    assert p_g != p_r


# ────────────────────────────────────────────────────────────────────
# emit_tier_c_verdict — short-circuit paths
# ────────────────────────────────────────────────────────────────────
def test_emit_short_circuits_on_non_emittable_verdict(tmp_evidence):
    """PENDING_TEMPLATE_BUILD / DATA_ERROR / etc. are dispatcher-
    internal states, NOT research findings. Must NOT emit + must
    NOT register a subject + must NOT write evidence."""
    from engine.agents.strengthener.factor_verdict_emit import (
        emit_tier_c_verdict,
    )
    s = _spec(hypothesis_id="hid_nonemit_" + uuid.uuid4().hex[:6])
    for v in ("PENDING_TEMPLATE_BUILD", "DATA_ERROR",
                "EXECUTION_ERROR", "INSUFFICIENT_HISTORY",
                "UNSUPPORTED_UNIVERSE", "CUSTOM_CODE_REQUIRED"):
        eid = emit_tier_c_verdict(
            s, family_hint="X", template_result=_tpl_result(verdict=v))
        assert eid is None, f"verdict {v} should NOT emit"
    # No evidence files written for these
    assert not any(tmp_evidence.glob("*.md")) if tmp_evidence.exists() \
        else True


# ────────────────────────────────────────────────────────────────────
# emit_tier_c_verdict — happy path emits to real event store
# ────────────────────────────────────────────────────────────────────
def test_emit_green_writes_event_and_evidence(tmp_evidence):
    from engine.agents.strengthener.factor_verdict_emit import (
        emit_tier_c_verdict,
    )
    from engine.research_store.store import by_event_id
    s = _spec(hypothesis_id="hid_green_" + uuid.uuid4().hex[:6])
    tr = _tpl_result(verdict="GREEN", t_stat=2.50, sharpe=1.20)
    eid = emit_tier_c_verdict(s, family_hint="MOMENTUM",
                                  template_result=tr,
                                  dispatch_event_id="ev_disp_green")
    assert eid is not None
    # Evidence file written
    md_files = list(tmp_evidence.glob("*_GREEN.md"))
    assert len(md_files) == 1
    # Event persisted with right shape
    ev = by_event_id(eid)
    assert ev is not None
    assert ev.event_type.value == "factor_verdict_filed"
    assert ev.verdict.value == "GREEN"
    assert ev.family == "MOMENTUM"
    assert "tier_c_auto" in ev.tags
    assert s.signal_kind in ev.tags
    # Provenance metrics present
    m = ev.metrics or {}
    assert m["tier_c_auto"] is True
    assert m["source_hypothesis_id"] == s.hypothesis_id
    assert m["auto_test_llm_model"] == "claude-sonnet-4-6"
    assert m["auto_test_spec_hash"]   # 16-char hex
    assert m["extractor_workload"] == "strengthener_factor_spec"
    assert m["dispatcher_version"]
    assert m["n_trials"] == 1
    assert m["sharpe"] == 1.20
    # Evidence path in artifacts (relative, posix)
    assert ev.artifacts.get("evidence_doc"), "evidence_doc artifact missing"


def test_emit_red_also_emits(tmp_evidence):
    """RED verdicts also count as research findings (they tell the
    n_trials counter + decay sentinel that a test happened)."""
    from engine.agents.strengthener.factor_verdict_emit import (
        emit_tier_c_verdict,
    )
    from engine.research_store.store import by_event_id
    s = _spec(hypothesis_id="hid_red_" + uuid.uuid4().hex[:6])
    eid = emit_tier_c_verdict(
        s, family_hint="MOMENTUM",
        template_result=_tpl_result(verdict="RED", t_stat=-0.5,
                                       sharpe=-0.20),
    )
    assert eid is not None
    ev = by_event_id(eid)
    assert ev.verdict.value == "RED"


# ────────────────────────────────────────────────────────────────────
# Dispatcher end-to-end with emission
# ────────────────────────────────────────────────────────────────────
def test_dispatch_with_emittable_verdict_returns_event_id(
    tmp_path, tmp_evidence, monkeypatch,
):
    """End-to-end through dispatch_factor_spec: a template that
    returns GREEN should result in a verdict_event_id field on the
    out dict + a real factor_verdict_filed in the event store."""
    from engine.agents.strengthener import factor_dispatcher as fd
    # Force the template to return GREEN regardless of universe
    monkeypatch.setitem(
        fd.TEMPLATE_REGISTRY, "time_series_momentum",
        lambda spec: _tpl_result(verdict="GREEN", t_stat=2.10),
    )
    monkeypatch.setattr(fd, "_family_n_trials_now", lambda fam: 0)
    log_path = tmp_path / "log.jsonl"
    s = _spec(hypothesis_id="hid_e2e_" + uuid.uuid4().hex[:6])
    out = fd.dispatch_factor_spec(s, family_hint="MOMENTUM",
                                       spec_approved=True,
                                       log_path=log_path)
    assert out["refusal"] is None
    assert out["template_result"]["verdict"] == "GREEN"
    assert out["verdict_event_id"]   # NEW field from C-2c
    assert out["dispatch_event_id"]  # audit log id (different from
                                       # verdict_event_id)


def test_dispatch_with_pending_verdict_does_not_emit(
    tmp_path, tmp_evidence, monkeypatch,
):
    """Stub-template paths return PENDING_TEMPLATE_BUILD. Dispatcher
    should NOT call emit + out dict should NOT have verdict_event_id.
    """
    from engine.agents.strengthener import factor_dispatcher as fd
    monkeypatch.setattr(fd, "_family_n_trials_now", lambda fam: 0)
    log_path = tmp_path / "log.jsonl"
    # vrp still stubbed in C-2e.1 (cross_sectional_rank shipped)
    s = _spec(signal_kind="vrp",
                universe="us_equities_sp500",
                signal_inputs=("optionmetrics.standardized_options.vrp",))
    out = fd.dispatch_factor_spec(s, family_hint="VOL_RISK_PREMIUM",
                                       spec_approved=True,
                                       log_path=log_path)
    assert out["template_result"]["verdict"] == "PENDING_TEMPLATE_BUILD"
    assert "verdict_event_id" not in out or \
        out.get("verdict_event_id") is None
