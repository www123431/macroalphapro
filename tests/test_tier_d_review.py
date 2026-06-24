"""tests/test_tier_d_review.py — Phase 1 Commit 4.

Tests the Tier D routing entry point + diagnostic metrics +
dispatcher integration. Per docs/spec_role_aware_test_routing.md
§15.A3: non-alpha sleeves get diagnostics, NOT verdict.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# should_route_to_tier_d — routing decision
# ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("role,expected", [
    ("alpha",       False),
    ("overlay",     False),
    ("insurance",   True),
    ("diversifier", True),
    ("hedge",       True),
    (None,          False),  # legacy spec → default Tier C
])
def test_should_route_by_investment_role(role, expected):
    from engine.agents.strengthener.tier_d_review import (
        should_route_to_tier_d,
    )
    spec = SimpleNamespace(investment_role=role)
    assert should_route_to_tier_d(spec) is expected


# ────────────────────────────────────────────────────────────────────
# _compute_role_minimal_diagnostics
# ────────────────────────────────────────────────────────────────────
def test_diagnostics_compute_on_real_series():
    from engine.agents.strengthener.tier_d_review import (
        _compute_role_minimal_diagnostics,
    )
    rng = np.random.default_rng(7)
    idx = pd.date_range("2014-01-31", periods=60, freq="ME")
    s = pd.Series(rng.normal(0.005, 0.04, 60), index=idx)
    d = _compute_role_minimal_diagnostics(s)
    assert d["n_months"] == 60
    assert "ann_return_pct" in d
    assert "max_drawdown_pct" in d
    assert d["max_drawdown_pct"] < 0  # always negative
    assert 0 <= d["hit_rate_pct"] <= 100
    assert d["phase_3_pending"] is True
    assert "methodology research" in d["phase_3_pending_reason"]


def test_diagnostics_insufficient_history():
    from engine.agents.strengthener.tier_d_review import (
        _compute_role_minimal_diagnostics,
    )
    s = pd.Series([0.01, 0.02, -0.01],
                    index=pd.date_range("2024-01-31", periods=3, freq="ME"))
    d = _compute_role_minimal_diagnostics(s)
    assert d.get("insufficient_history") is True
    assert d["n_months"] == 3


# ────────────────────────────────────────────────────────────────────
# dispatch_tier_d — full path
# ────────────────────────────────────────────────────────────────────
def _build_spec(**kw):
    base = dict(
        hypothesis_id="hid_test",
        investment_role="insurance",
        statistical_role="directional",
        asset_class="equity",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _build_template_result(verdict="GREEN", with_pnl=True):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    artifacts = {}
    if with_pnl:
        rng = np.random.default_rng(11)
        idx = pd.date_range("2014-01-31", periods=60, freq="ME")
        df = pd.DataFrame({
            "pnl_gross":    rng.normal(0.005, 0.04, 60),
            "pnl_net_13bp": rng.normal(0.005, 0.04, 60),
            "pnl_net_80bp": rng.normal(0.003, 0.04, 60),
            "turnover":     rng.uniform(0.1, 0.3, 60),
        }, index=idx)
        artifacts["pnl_series_df"] = df
    return TemplateResult(
        verdict=verdict, summary="test",
        metrics={}, artifacts=artifacts,
        template_version="v1",
    )


def test_dispatch_tier_d_produces_diagnostic_payload(tmp_path,
                                                          monkeypatch):
    """Tier D output dict has expected structure + writes to queue."""
    from engine.agents.strengthener import tier_d_review as td
    monkeypatch.setattr(td, "TIER_D_LOG_PATH",
                          tmp_path / "tier_d_review_queue.jsonl")
    spec = _build_spec()
    tr = _build_template_result()
    result = td.dispatch_tier_d(spec, family_hint="OTHER",
                                     template_result=tr,
                                     dispatch_event_id="disp_abc")
    assert result["tier"] == "D"
    assert result["investment_role"] == "insurance"
    assert result["human_review_required"] is True
    assert result["phase_3_pending"] is True
    assert "diagnostic_metrics" in result
    assert result["diagnostic_metrics"]["n_months"] == 60
    # Queue row written
    log_path = tmp_path / "tier_d_review_queue.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["tier"] == "D"
    assert row["review_status"] == "PENDING_HUMAN_REVIEW"


def test_dispatch_tier_d_handles_missing_pnl(tmp_path, monkeypatch):
    from engine.agents.strengthener import tier_d_review as td
    monkeypatch.setattr(td, "TIER_D_LOG_PATH",
                          tmp_path / "queue.jsonl")
    result = td.dispatch_tier_d(_build_spec(), family_hint="X",
                                     template_result=_build_template_result(
                                         with_pnl=False))
    assert result["diagnostic_metrics"]["pnl_series_missing"] is True


# ────────────────────────────────────────────────────────────────────
# Dispatcher integration: Tier D bypasses Tier C entirely
# ────────────────────────────────────────────────────────────────────
def test_dispatcher_routes_insurance_to_tier_d(tmp_path, monkeypatch):
    """An insurance-role FactorSpec should NOT trigger any Tier C
    lens / self_doubt / emit. Output gets tier_d_result instead."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener import tier_d_review as td
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    monkeypatch.setattr(td, "TIER_D_LOG_PATH",
                          tmp_path / "queue.jsonl")
    monkeypatch.setattr(fd, "_family_n_trials_now", lambda fam: 0)

    spec = FactorSpec(
        hypothesis_id           = "hid_insurance_test",
        signal_kind             = "cross_sectional_rank",
        universe                = "us_equities_top_3000",
        date_range              = "2014-01:2024-12",
        signal_inputs           = ("crsp.msf.ret",),
        rebal                   = "monthly",
        weighting               = "decile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months          = 60,
        pit_audits              = ("lookahead",),
        cost_model              = "engine.execution.cost_model.basic",
        rationale               = "test",
        extracted_ts            = "2026-06-09T00:00:00Z",
        model                   = "claude-sonnet-4-6",
        investment_role         = "insurance",
    )
    # Stub the template registry to return a clean GREEN with a
    # pnl_series_df so Tier C lenses WOULD run if not bypassed
    def _fake_template(s):
        return _build_template_result(verdict="GREEN", with_pnl=True)
    monkeypatch.setitem(fd.TEMPLATE_REGISTRY,
                          "cross_sectional_rank", _fake_template)

    # Stub self_doubt + emit — should NOT be called for Tier D
    sd_called = {"n": 0}
    def _spy_sd(*a, **kw):
        sd_called["n"] += 1
        return None
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "assess_self_doubt", _spy_sd)

    emit_called = {"n": 0}
    from engine.agents.strengthener import factor_verdict_emit as emit_mod
    def _spy_emit(*a, **kw):
        emit_called["n"] += 1
        return "eid"
    monkeypatch.setattr(emit_mod, "emit_tier_c_verdict", _spy_emit)

    out = fd.dispatch_factor_spec(
        spec, family_hint="OTHER", spec_approved=True,
        log_path=tmp_path / "log.jsonl",
    )

    # Tier D output present
    assert "tier_d_result" in out
    assert out["tier_d_result"]["tier"] == "D"
    assert out["tier_d_result"]["human_review_required"] is True
    # Tier C did NOT run
    assert "anchor_orthogonality" not in out
    assert "industry_extension" not in out
    assert "cross_asset_extension" not in out
    assert sd_called["n"] == 0
    assert emit_called["n"] == 0


def test_dispatcher_alpha_role_still_goes_through_tier_c(
    tmp_path, monkeypatch,
):
    """Explicit alpha role still triggers normal Tier C flow."""
    from engine.agents.strengthener import factor_dispatcher as fd
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec

    monkeypatch.setattr(fd, "_family_n_trials_now", lambda fam: 0)

    spec = FactorSpec(
        hypothesis_id           = "hid_alpha_test",
        signal_kind             = "cross_sectional_rank",
        universe                = "us_equities_top_3000",
        date_range              = "2014-01:2024-12",
        signal_inputs           = ("crsp.msf.ret",),
        rebal                   = "monthly",
        weighting               = "decile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months          = 60,
        pit_audits              = ("lookahead",),
        cost_model              = "engine.execution.cost_model.basic",
        rationale               = "test",
        extracted_ts            = "2026-06-09T00:00:00Z",
        model                   = "claude-sonnet-4-6",
        investment_role         = "alpha",
    )
    def _fake_template(s):
        return _build_template_result(verdict="GREEN", with_pnl=False)
    monkeypatch.setitem(fd.TEMPLATE_REGISTRY,
                          "cross_sectional_rank", _fake_template)

    out = fd.dispatch_factor_spec(
        spec, family_hint="PROFITABILITY", spec_approved=True,
        log_path=tmp_path / "log.jsonl",
    )
    assert "tier_d_result" not in out
    # template_result populated normally
    assert out["template_result"]["verdict"] == "GREEN"
