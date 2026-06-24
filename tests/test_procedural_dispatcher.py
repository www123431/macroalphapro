"""tests/test_procedural_dispatcher.py — Stage B close-loop.

Tests the procedural auto-spec dispatcher. llm_call + filter_events
+ dispatch log path all mocked/redirected so tests are offline +
fast + free.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _proc_hyp(*, hypothesis_id="hid_test", mechanism_subtype="family_pause_proposal",
               predicted_direction_value="zero",
               addresses_decay_in=None,
               synthesizes_event_ids=("ev_1",),
               mechanism_family_value="PROFITABILITY",
               test_methodology="run F8 classifier",
               claim="test claim",
               required_data=("RED events",)):
    """Build a minimal Hypothesis-shaped namespace with the fields
    the dispatcher reads."""
    return SimpleNamespace(
        hypothesis_id        = hypothesis_id,
        mechanism_subtype    = mechanism_subtype,
        predicted_direction  = SimpleNamespace(value=predicted_direction_value),
        mechanism_family     = SimpleNamespace(value=mechanism_family_value),
        addresses_decay_in   = addresses_decay_in,
        synthesizes_event_ids = synthesizes_event_ids,
        test_methodology     = test_methodology,
        claim                = claim,
        required_data        = required_data,
        predicted_magnitude  = "marginal",
    )


def _mock_llm_spec(*, kind, args=None, rationale="test"):
    """Build an llm_call result with a valid emit_dispatch_spec tool
    call payload."""
    return SimpleNamespace(
        text       = "",
        tool_calls = (SimpleNamespace(
            name="emit_dispatch_spec",
            input={"dispatch_kind": kind,
                    "args": args or {},
                    "rationale": rationale},
        ),),
        model      = "claude-sonnet-4-6",
    )


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    """Redirect DISPATCH_LOG_PATH to tmp file."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    log_path = tmp_path / "procedural_dispatch_log.jsonl"
    monkeypatch.setattr(pd, "DISPATCH_LOG_PATH", log_path)
    return log_path


# ────────────────────────────────────────────────────────────────────
# Eligibility check
# ────────────────────────────────────────────────────────────────────
def test_is_procedural_accepts_zero_direction_proposal_with_provenance():
    from engine.agents.strengthener.procedural_dispatcher import (
        is_procedural_hypothesis,
    )
    h = _proc_hyp()
    assert is_procedural_hypothesis(h) is True


def test_is_procedural_rejects_nonzero_direction():
    """Factor-return hypotheses (predicted_direction != zero) stay
    human-test per look-ahead-risk doctrine."""
    from engine.agents.strengthener.procedural_dispatcher import (
        is_procedural_hypothesis,
    )
    h = _proc_hyp(predicted_direction_value="positive")
    assert is_procedural_hypothesis(h) is False


def test_is_procedural_rejects_non_procedural_subtype():
    from engine.agents.strengthener.procedural_dispatcher import (
        is_procedural_hypothesis,
    )
    h = _proc_hyp(mechanism_subtype="cross_sectional_rank_signal")
    assert is_procedural_hypothesis(h) is False


def test_is_procedural_rejects_no_provenance():
    """Without addresses_decay_in OR synthesizes_event_ids, dispatcher
    has no context to operate on."""
    from engine.agents.strengthener.procedural_dispatcher import (
        is_procedural_hypothesis,
    )
    h = _proc_hyp(addresses_decay_in=None, synthesizes_event_ids=())
    assert is_procedural_hypothesis(h) is False


def test_is_procedural_accepts_subtype_variants():
    """Several procedural subtypes should match the regex."""
    from engine.agents.strengthener.procedural_dispatcher import (
        is_procedural_hypothesis,
    )
    for st in ["family_pause_proposal", "sleeve_decay_response",
                "filter_tighten_proposal", "data_quality_audit",
                "red_cluster_recount", "failure_mode_classify"]:
        h = _proc_hyp(mechanism_subtype=st)
        assert is_procedural_hypothesis(h), f"subtype {st} rejected"


# ────────────────────────────────────────────────────────────────────
# Spec extraction (LLM mocked)
# ────────────────────────────────────────────────────────────────────
def test_extract_spec_happy_path(monkeypatch):
    from engine.agents.strengthener import procedural_dispatcher as pd
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: _mock_llm_spec(
            kind="failure_mode_classify",
            args={"red_event_ids": ["ev_1"], "threshold_pct": 0.5}))
    spec = pd.extract_dispatch_spec(_proc_hyp())
    assert spec is not None
    assert spec.dispatch_kind == "failure_mode_classify"
    assert spec.args["red_event_ids"] == ["ev_1"]


def test_extract_spec_unknown_kind_returns_none(monkeypatch):
    """If LLM somehow emits a dispatch_kind outside the controlled
    enum, return None so the orchestrator falls through to 'unrecognized'."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: _mock_llm_spec(kind="invented_kind",
                                       args={}))
    assert pd.extract_dispatch_spec(_proc_hyp()) is None


def test_extract_spec_llm_failure_returns_none(monkeypatch):
    from engine.agents.strengthener import procedural_dispatcher as pd
    def _broken(**kw): raise RuntimeError("anthropic 500")
    monkeypatch.setattr(pd, "llm_call", _broken)
    assert pd.extract_dispatch_spec(_proc_hyp()) is None


def test_extract_spec_no_tool_call_returns_none(monkeypatch):
    from engine.agents.strengthener import procedural_dispatcher as pd
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: SimpleNamespace(text="prose", tool_calls=(),
                                        model="x"))
    assert pd.extract_dispatch_spec(_proc_hyp()) is None


# ────────────────────────────────────────────────────────────────────
# Dispatchers — failure_mode_classify (the 47893a71 case)
# ────────────────────────────────────────────────────────────────────
def test_failure_mode_classify_above_threshold_recommends_pause(
    monkeypatch,
):
    """3/3 events classify as F8 → above 0.5 threshold → MARGINAL
    'recommend pause'."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st

    fake_events = [
        SimpleNamespace(event_id="ev_1",
                         metrics={"deflated_sr": 0.1}),
        SimpleNamespace(event_id="ev_2",
                         metrics={"deflated_sr": 0.2}),
        SimpleNamespace(event_id="ev_3",
                         metrics={"deflated_sr": 0.3}),
    ]
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: list(fake_events))
    dr = pd._dispatch_failure_mode_classify({
        "red_event_ids": ["ev_1", "ev_2", "ev_3"],
        "threshold_pct": 0.5,
    })
    assert dr["verdict"] == "MARGINAL"
    assert dr["metrics"]["n_events"] == 3
    assert dr["metrics"]["n_f8"] == 3
    assert "pause" in dr["summary"].lower()


def test_failure_mode_classify_below_threshold_green(monkeypatch):
    """Below F8-fraction threshold → GREEN 'no pause'. To trigger
    non-F8 classification we use corr_with_book > 0.5 (F3 subsumed)
    which fires a SPECIFIC rule and skips the F8 backstop."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st

    # Mix: 1 F8 (deflated_sr<0.9), 2 F3-subsumed (high corr_with_book
    # → specific rule fires, F8 backstop doesn't)
    fake_events = [
        SimpleNamespace(event_id="ev_1",
                         metrics={"deflated_sr": 0.1}),
        SimpleNamespace(event_id="ev_2",
                         metrics={"deflated_sr": 1.2,
                                   "corr_with_book": 0.85}),
        SimpleNamespace(event_id="ev_3",
                         metrics={"deflated_sr": 1.5,
                                   "corr_with_book": 0.9}),
    ]
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: list(fake_events))
    dr = pd._dispatch_failure_mode_classify({
        "red_event_ids": ["ev_1", "ev_2", "ev_3"],
        "threshold_pct": 0.5,
    })
    assert dr["verdict"] == "GREEN"
    assert dr["metrics"]["n_f8"] == 1
    assert dr["metrics"]["f8_pct"] < 0.5


def test_failure_mode_classify_empty_input_marginal(monkeypatch):
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st
    monkeypatch.setattr(st, "filter_events", lambda **kw: [])
    dr = pd._dispatch_failure_mode_classify({"red_event_ids": []})
    assert dr["verdict"] == "MARGINAL"
    assert dr["metrics"]["n_events"] == 0


# ────────────────────────────────────────────────────────────────────
# Dispatchers — red_cluster_recount (verifies dedup integration)
# ────────────────────────────────────────────────────────────────────
def test_red_cluster_recount_post_dedup_green(monkeypatch):
    """Recount of family with only duplicates → 0 unique → GREEN."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st

    # 3 raw events but all same subject → dedup to 1 → no cluster
    dup_event = SimpleNamespace(
        event_id="ev_dup", subject_id="auto_X",
        family="PROFITABILITY", verdict="RED",
        ts="2026-06-05T00:00:00Z",
        event_type=SimpleNamespace(value="factor_verdict_filed"),
        metrics={"source_hypothesis_id": "hyp_X"},
    )
    # check_family_red_cluster reads ev.event_type vs EventType enum,
    # so we need to patch differently — just provide the right enum
    from engine.research_store.schema import EventType, Verdict
    real_events = [
        SimpleNamespace(
            event_id=f"ev_{i}", subject_id="auto_X",
            event_type=EventType.factor_verdict_filed,
            verdict=Verdict.RED, ts=f"2026-06-0{i+1}T00:00:00Z",
            family="PROFITABILITY",
            metrics={"source_hypothesis_id": "hyp_X"},
        )
        for i in range(3)
    ]
    monkeypatch.setattr(st, "filter_events",
                          lambda **kw: real_events)
    dr = pd._dispatch_red_cluster_recount({
        "family": "PROFITABILITY", "window_days": 30,
    })
    assert dr["verdict"] == "GREEN"
    assert dr["metrics"]["raw_red_count"] == 3
    assert dr["metrics"]["deduped_cluster"] == 0


def test_red_cluster_recount_missing_family():
    from engine.agents.strengthener import procedural_dispatcher as pd
    dr = pd._dispatch_red_cluster_recount({})
    assert dr["verdict"] == "MARGINAL"
    assert "no family" in dr["summary"]


# ────────────────────────────────────────────────────────────────────
# Dispatchers — decay_resentinel_rerun (stub for now)
# ────────────────────────────────────────────────────────────────────
def test_decay_resentinel_returns_human_handoff():
    from engine.agents.strengthener import procedural_dispatcher as pd
    dr = pd._dispatch_decay_resentinel_rerun({"sleeve_id": "cross_asset_carry"})
    assert dr["verdict"] == "MARGINAL"
    assert "cross_asset_carry" in dr["summary"]
    assert dr["metrics"]["dispatched"] is False   # stub


# ────────────────────────────────────────────────────────────────────
# Dispatchers — unrecognized
# ────────────────────────────────────────────────────────────────────
def test_unrecognized_dispatcher_returns_marginal_human_needed():
    from engine.agents.strengthener import procedural_dispatcher as pd
    dr = pd._dispatch_unrecognized({})
    assert dr["verdict"] == "MARGINAL"
    assert "unrecognized" in dr["summary"].lower() or \
            "needs human" in dr["summary"].lower()


# ────────────────────────────────────────────────────────────────────
# Full orchestration — auto_dispatch_procedural
# ────────────────────────────────────────────────────────────────────
def test_auto_dispatch_full_flow_persists_log(monkeypatch, tmp_log):
    """Eligible hypothesis → LLM extracts spec → dispatcher runs →
    log appended → returns event id."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st

    # LLM returns failure_mode_classify spec
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: _mock_llm_spec(
            kind="failure_mode_classify",
            args={"red_event_ids": ["ev_1"], "threshold_pct": 0.5}))
    # Dispatcher's event fetch
    monkeypatch.setattr(st, "filter_events",
        lambda **kw: [SimpleNamespace(
            event_id="ev_1", metrics={"deflated_sr": 0.1})])

    h = _proc_hyp(synthesizes_event_ids=("ev_1",))
    r = pd.auto_dispatch_procedural(h)
    assert r["eligible"] is True
    assert r["spec"]["dispatch_kind"] == "failure_mode_classify"
    assert r["dispatch_result"]["verdict"] == "MARGINAL"
    assert r["emitted_event_id"] is not None
    assert r["errors"] == []

    # Log file has 1 row
    lines = [l for l in tmp_log.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["hypothesis_id"] == "hid_test"
    assert rec["spec"]["dispatch_kind"] == "failure_mode_classify"
    assert rec["dispatch_result"]["verdict"] == "MARGINAL"


def test_auto_dispatch_ineligible_short_circuits(monkeypatch, tmp_log):
    """Non-procedural hypothesis → no LLM call, no log write,
    eligible=False."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: pytest.fail("LLM must NOT be called for ineligible"))
    h = _proc_hyp(predicted_direction_value="positive")
    r = pd.auto_dispatch_procedural(h)
    assert r["eligible"] is False
    assert r["spec"] is None
    assert r["emitted_event_id"] is None
    assert not tmp_log.exists()


def test_auto_dispatch_llm_failure_falls_through_to_unrecognized(
    monkeypatch, tmp_log,
):
    """LLM raises → spec=None → falls through to 'unrecognized'
    dispatcher → still logged (so we can see what was skipped)."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    def _broken(**kw): raise RuntimeError("anthropic down")
    monkeypatch.setattr(pd, "llm_call", _broken)
    h = _proc_hyp()
    r = pd.auto_dispatch_procedural(h)
    assert r["eligible"] is True
    assert r["spec"]["dispatch_kind"] == "unrecognized"
    assert r["dispatch_result"]["verdict"] == "MARGINAL"
    assert r["emitted_event_id"] is not None


def test_auto_dispatch_dispatcher_exception_recorded(monkeypatch, tmp_log):
    """If a dispatcher raises, error captured + degraded MARGINAL
    result + still logged (audit trail preserved)."""
    from engine.agents.strengthener import procedural_dispatcher as pd
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: _mock_llm_spec(kind="failure_mode_classify",
                                       args={}))
    # Force dispatcher to raise
    def _broken_disp(args):
        raise RuntimeError("dispatcher boom")
    monkeypatch.setattr(pd, "DISPATCHERS",
        {**pd.DISPATCHERS,
          "failure_mode_classify": _broken_disp})
    h = _proc_hyp()
    r = pd.auto_dispatch_procedural(h)
    assert any("dispatcher:failure_mode_classify" in e
                 for e in r["errors"])
    assert r["dispatch_result"]["verdict"] == "MARGINAL"
    assert r["emitted_event_id"] is not None   # still logged


def test_auto_dispatch_dry_run_skips_log(monkeypatch, tmp_log):
    from engine.agents.strengthener import procedural_dispatcher as pd
    from engine.research_store import store as st
    monkeypatch.setattr(pd, "llm_call",
        lambda **kw: _mock_llm_spec(kind="failure_mode_classify",
                                       args={"red_event_ids": []}))
    monkeypatch.setattr(st, "filter_events", lambda **kw: [])
    h = _proc_hyp()
    r = pd.auto_dispatch_procedural(h, dry_run=True)
    assert r["eligible"] is True
    assert r["emitted_event_id"] is None
    assert not tmp_log.exists()


# ────────────────────────────────────────────────────────────────────
# Workload registration sanity
# ────────────────────────────────────────────────────────────────────
def test_workload_registered():
    from engine.llm.call import _resolve_workload
    provider, model = _resolve_workload("strengthener_spec_extract")
    assert provider == "anthropic"
    assert "claude" in model
