"""tests/test_sleeve_strengthen_proposer.py — Stage B P3a.

Tests the LLM-driven sleeve strengthen proposer. llm_call is mocked
so tests are offline + free + deterministic.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def _ctx(**kw):
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        SleeveContext,
    )
    base = dict(
        sleeve_id           = "cross_asset_carry",
        family              = "CARRY",
        canonical_paper_id  = "koijen_moskowitz_pedersen_vrugt_2018_jfe",
        mechanism_economics = "Carry captures persistent risk premium.",
        canonical_universe  = "G10 govt + commodity + DM FX + DM equity",
        typical_sample      = "1990-present",
        deployed_summary    = "Sharpe ~0.85, last refresh 2026-05-30",
        snapshot_ts         = "2026-06-07T00:00:00Z",
    )
    base.update(kw)
    return SleeveContext(**base)


def _mock_llm_result(*, tool_payload=None, raw_text=""):
    """Build the llm_call result shape (text + tool_calls + model)."""
    tool_calls: tuple = ()
    if tool_payload is not None:
        tool_calls = (SimpleNamespace(
            name="emit_strengthen_proposals", input=tool_payload),)
    return SimpleNamespace(
        text       = raw_text,
        tool_calls = tool_calls,
        model      = "claude-sonnet-4-6",
    )


def _valid_candidate(**override):
    """Schema-valid candidate dict for the tool payload."""
    base = {
        "claim": "Add VIX overlay to dampen drawdown in carry crashes.",
        "improvement_kind": "regime_filter",
        "mechanism_subtype": "vix_regime_overlay",
        "predicted_magnitude": "moderate",
        "required_data": ["VIX index daily from CBOE"],
        "test_methodology": ("engine.validation.decay_sentinel + "
                              "engine.regime backtest on the existing "
                              "carry sleeve PnL"),
        "expected_outcome_prior": "likely_REJECT_per_HXZ_65pct",
    }
    base.update(override)
    return base


# ────────────────────────────────────────────────────────────────────
# Happy path
# ────────────────────────────────────────────────────────────────────
def test_returns_proposals_from_valid_payload(monkeypatch):
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    payload = {
        "candidates": [
            _valid_candidate(),
            _valid_candidate(
                claim="Replace carry with cost-aware variant",
                improvement_kind="cost_aware_exec",
                predicted_magnitude="marginal",
            ),
        ],
    }
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload=payload))
    proposals = ssp.run_strengthen_proposer(_ctx())
    assert len(proposals) == 2
    assert proposals[0].improvement_kind == "regime_filter"
    assert proposals[1].improvement_kind == "cost_aware_exec"
    assert proposals[0].model == "claude-sonnet-4-6"
    assert proposals[0].generation_ts != ""


def test_returns_empty_when_llm_returns_empty(monkeypatch):
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload={"candidates": []}))
    assert ssp.run_strengthen_proposer(_ctx()) == []


# ────────────────────────────────────────────────────────────────────
# Failure modes — all degrade to []
# ────────────────────────────────────────────────────────────────────
def test_llm_exception_returns_empty(monkeypatch):
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    def _broken(**kw):
        raise RuntimeError("anthropic 500")
    monkeypatch.setattr(ssp, "llm_call", _broken)
    assert ssp.run_strengthen_proposer(_ctx()) == []


def test_tool_not_called_returns_empty(monkeypatch):
    """LLM produced text but didn't invoke the tool → []."""
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload=None,
                            raw_text="I have no proposals."))
    assert ssp.run_strengthen_proposer(_ctx()) == []


def test_candidates_not_a_list_returns_empty(monkeypatch):
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload={"candidates": "oops"}))
    assert ssp.run_strengthen_proposer(_ctx()) == []


# ────────────────────────────────────────────────────────────────────
# Per-candidate validation
# ────────────────────────────────────────────────────────────────────
def test_unknown_improvement_kind_is_dropped(monkeypatch):
    """If LLM emits an improvement_kind outside the controlled enum,
    that candidate is dropped but the run still completes."""
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    payload = {"candidates": [
        _valid_candidate(),
        _valid_candidate(improvement_kind="invented_new_category"),
        _valid_candidate(improvement_kind="risk_overlay"),
    ]}
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload=payload))
    proposals = ssp.run_strengthen_proposer(_ctx())
    assert len(proposals) == 2
    assert {p.improvement_kind for p in proposals} == {
        "regime_filter", "risk_overlay"}


def test_missing_required_field_drops_candidate(monkeypatch):
    """A candidate missing 'claim' drops; siblings still pass."""
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    payload = {"candidates": [
        _valid_candidate(),
        {  # missing claim
            "improvement_kind": "regime_filter",
            "mechanism_subtype": "x",
            "predicted_magnitude": "marginal",
            "required_data": ["X"],
            "test_methodology": "engine.foo",
            "expected_outcome_prior": "y",
        },
    ]}
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload=payload))
    proposals = ssp.run_strengthen_proposer(_ctx())
    assert len(proposals) == 1


def test_hard_cap_3_even_if_more_emitted(monkeypatch):
    """Schema says maxItems=3 but defensively cap at 3 anyway."""
    from engine.agents.strengthener import (
        sleeve_strengthen_proposer as ssp,
    )
    payload = {"candidates": [_valid_candidate() for _ in range(5)]}
    monkeypatch.setattr(ssp, "llm_call",
                          lambda **kw: _mock_llm_result(
                            tool_payload=payload))
    assert len(ssp.run_strengthen_proposer(_ctx())) == 3


# ────────────────────────────────────────────────────────────────────
# Prompt-building
# ────────────────────────────────────────────────────────────────────
def test_format_input_includes_sleeve_identity():
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        _format_input,
    )
    out = _format_input(_ctx())
    assert "cross_asset_carry" in out
    assert "CARRY" in out
    assert "koijen_moskowitz_pedersen_vrugt_2018_jfe" in out


def test_format_input_shows_empty_recent_state_explicitly():
    """Sleeve with no recent REDs / decay alerts → explicit "none"
    so the LLM knows the absence is REAL (vs we forgot to include)."""
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        _format_input,
    )
    out = _format_input(_ctx())
    assert "RECENT_FAMILY_RED_VERDICTS: none in window" in out
    assert "RECENT_DECAY_ALERTS FOR THIS SLEEVE: none" in out


def test_format_input_lists_red_ids_when_present():
    from engine.agents.strengthener.sleeve_strengthen_proposer import (
        _format_input,
    )
    out = _format_input(_ctx(
        recent_family_red_ids=("ev_red1", "ev_red2"),
        recent_decay_alert_ids=("ev_alert1",),
    ))
    assert "ev_red1" in out
    assert "ev_red2" in out
    assert "ev_alert1" in out


# ────────────────────────────────────────────────────────────────────
# Workload registration sanity (catches typo bugs)
# ────────────────────────────────────────────────────────────────────
def test_workload_is_registered():
    """strengthener_propose must resolve in the LLM workload routing
    table; otherwise llm_call would raise at runtime."""
    from engine.llm.call import _resolve_workload
    provider, model = _resolve_workload("strengthener_propose")
    assert provider == "anthropic"
    assert "claude" in model
