"""tests/test_self_doubt.py — Tier C L3-2 Self-Doubt module.

Offline tests for the assess_self_doubt function. LLM call mocked.
Covers happy path, all defensive failure paths (no tool called,
bad confidence range, etc.), non-emittable verdicts skip, and the
graceful-degradation contract.
"""
from __future__ import annotations

import dataclasses as _dc
from types import SimpleNamespace

import pytest


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────
def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_sd",
        signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000",
        date_range="1992-01:2024-12",
        signal_inputs=("compustat.funda.gp_at",),
        rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly",
        min_obs_months=120,
        pit_audits=("lookahead",),
        cost_model="basic", rationale="test",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


def _tpl(verdict="GREEN", **extras):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    base_metrics = {
        "sharpe": 0.67, "nw_t_stat": 3.57, "ann_return": 0.069,
        "n_months": 395, "avg_turnover": 0.17,
        "naive_verdict": "GREEN", "cost_robust_verdict": "GREEN",
        "cost_stress": {
            "0bp":  {"sharpe": 0.70, "nw_t_stat": 3.69, "verdict": "GREEN"},
            "30bp": {"sharpe": 0.63, "nw_t_stat": 3.41, "verdict": "GREEN"},
            "60bp": {"sharpe": 0.57, "nw_t_stat": 3.14, "verdict": "GREEN"},
            "80bp": {"sharpe": 0.53, "nw_t_stat": 2.95, "verdict": "GREEN"},
        },
        "drawdown_naive": {
            "max_drawdown_pct": -0.205,
            "max_underwater_months": 42,
            "calmar_ratio": 0.32,
        },
        "replication": {
            "status": "REPLICATED",
            "our_t": 3.04, "paper_reported_t": 3.0, "t_gap": 0.044,
        },
    }
    base_metrics.update(extras)
    return TemplateResult(
        verdict=verdict, summary=f"GP/A test verdict {verdict}",
        metrics=base_metrics, artifacts={},
        template_version="v1.1_2026-06-08",
    )


def _mock_llm_self_doubt(
    *,
    confidence=0.55,
    confidence_reason="GP/A REPLICATED (gap 0.044) and cost-robust at 80bp; restatement bias on 22.8% fallback rows is the main residual concern.",
    caveats=("GP/A 1992-2024 t=3.57 is somewhat higher than overlap-window 3.04, mild post-pub divergence to investigate",
              "Universe top-3000 may include micro-cap drivers — consider universe_size=500 variant",
              "n_trials in PROFITABILITY family currently below CAUTION threshold but accumulating"),
    methodological_concerns=("B4 EW-only L/S: VW would likely halve alpha per Asness",
                                "B6 no anchor orthogonality check yet"),
    suspicious_metrics=("post-2010 segment t implied >3.57 — counter to McLean-Pontiff decay prediction",),
    model="claude-sonnet-4-6",
):
    return SimpleNamespace(
        text="",
        tool_calls=(SimpleNamespace(
            name="emit_self_doubt",
            input={
                "confidence": confidence,
                "confidence_reason": confidence_reason,
                "caveats": list(caveats),
                "methodological_concerns": list(methodological_concerns),
                "suspicious_metrics": list(suspicious_metrics),
            },
        ),),
        model=model,
    )


# ────────────────────────────────────────────────────────────────────
# Dataclass integrity
# ────────────────────────────────────────────────────────────────────
def test_self_doubt_dataclass_is_frozen():
    from engine.agents.strengthener.self_doubt import SelfDoubtAssessment
    sd = SelfDoubtAssessment(
        confidence=0.5, confidence_reason="r",
        caveats=(), methodological_concerns=(),
        suspicious_metrics=(),
        assessment_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        sd.confidence = 0.9   # type: ignore[misc]


# ────────────────────────────────────────────────────────────────────
# Happy path
# ────────────────────────────────────────────────────────────────────
def test_assess_self_doubt_happy_path(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
                          lambda **kw: _mock_llm_self_doubt())
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="PROFITABILITY",
        n_trials_family=3,
    )
    assert sd is not None
    assert 0.0 <= sd.confidence <= 0.99
    assert sd.confidence == 0.55
    assert len(sd.caveats) >= 2
    assert sd.model == "claude-sonnet-4-6"
    assert sd.assessment_ts


def test_assess_self_doubt_passes_workload_correctly(monkeypatch):
    """Guards against silently downgrading to a wrong workload/model."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured.update(kw)
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(_spec(), _tpl(),
                                family_hint="X", n_trials_family=0)
    assert captured["workload"] == "strengthener_self_doubt"
    assert captured["agent_id"] == "strengthener_self_doubt"
    assert captured["scope"] == "tier_c_l3_2_self_doubt"


# ────────────────────────────────────────────────────────────────────
# Non-emittable verdicts: short-circuit, no LLM call
# ────────────────────────────────────────────────────────────────────
def test_skip_pending_template_build(monkeypatch):
    """PENDING_TEMPLATE_BUILD = system state, not research finding.
    Should NOT spend Sonnet $0.04 on it."""
    from engine.agents.strengthener import self_doubt as sd_mod
    called = {"n": 0}
    def _counter(**kw):
        called["n"] += 1
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _counter)
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(verdict="PENDING_TEMPLATE_BUILD"),
        family_hint="X", n_trials_family=0,
    )
    assert sd is None
    assert called["n"] == 0


def test_skip_data_error(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    called = {"n": 0}
    monkeypatch.setattr(sd_mod, "llm_call",
                          lambda **kw: (called.update(n=called["n"]+1)
                                          or _mock_llm_self_doubt()))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(verdict="DATA_ERROR"),
        family_hint="X", n_trials_family=0,
    )
    assert sd is None
    assert called["n"] == 0


def test_skip_execution_error(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    called = {"n": 0}
    monkeypatch.setattr(sd_mod, "llm_call",
                          lambda **kw: (called.update(n=called["n"]+1)
                                          or _mock_llm_self_doubt()))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(verdict="EXECUTION_ERROR"),
        family_hint="X", n_trials_family=0,
    )
    assert sd is None
    assert called["n"] == 0


# ────────────────────────────────────────────────────────────────────
# All 3 emittable verdicts trigger assessment
# ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("v", ["GREEN", "MARGINAL", "RED"])
def test_emittable_verdicts_trigger_assessment(v, monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
                          lambda **kw: _mock_llm_self_doubt())
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(verdict=v),
        family_hint="X", n_trials_family=0,
    )
    assert sd is not None


# ────────────────────────────────────────────────────────────────────
# Defensive: LLM failure paths
# ────────────────────────────────────────────────────────────────────
def test_returns_none_on_llm_exception(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    def _boom(**kw):
        raise RuntimeError("api timeout")
    monkeypatch.setattr(sd_mod, "llm_call", _boom)
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is None


def test_returns_none_when_tool_not_called(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
        lambda **kw: SimpleNamespace(
            text="declined", tool_calls=(),
            model="claude-sonnet-4-6"))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is None


def test_rejects_confidence_above_99(monkeypatch):
    """Confidence MUST be 0-0.99 (never 1.0 — anti-overconfidence)."""
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
        lambda **kw: _mock_llm_self_doubt(confidence=1.0))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is None


def test_rejects_confidence_negative(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
        lambda **kw: _mock_llm_self_doubt(confidence=-0.1))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is None


def test_rejects_non_numeric_confidence(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
        lambda **kw: _mock_llm_self_doubt(confidence="high"))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is None


# ────────────────────────────────────────────────────────────────────
# Output truncation safety (string-length contract)
# ────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────
# L2-4 Commit 3: anchor_orthogonality injection
# ────────────────────────────────────────────────────────────────────
def _sample_anchor_ortho(headline_gap_t: float = 1.7):
    """Build a representative anchor_orthogonality dict shaped like
    compute_for_tier_c_pnl_series output. headline_gap_t controls how
    much t-stat the anchors absorb (3.57 headline - gap = residual)."""
    residual_t = 3.57 - headline_gap_t
    return {
        "alpha_monthly":   0.0026,
        "alpha_annual":    0.0316,
        "alpha_nw_t":      residual_t,
        "alpha_nw_se":     0.0014,
        "betas":           {"MKT_RF": 0.13, "SMB": 0.12, "HML": -0.35,
                              "RMW": 0.67, "CMA": 0.16, "MOM": -0.01},
        "beta_nw_t":       {"MKT_RF": 3.5, "SMB": 1.99, "HML": -4.95,
                              "RMW": 10.0, "CMA": 1.58, "MOM": -0.14},
        "r2":              0.252,
        "r2_adj":          0.241,
        "n_overlap":       395,
        "anchor_names":    ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
        "nw_lag_used":     5,
        "window":          "1992-02:2024-12",
        "anchor_library":  "ken_french_ff5_mom",
    }


def test_assess_self_doubt_accepts_anchor_orthogonality(monkeypatch):
    """Pass anchor_orthogonality kwarg through; assert it lands in
    the user message sent to Sonnet (spy on llm_call)."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(),
        family_hint="PROFITABILITY", n_trials_family=2,
        anchor_orthogonality=_sample_anchor_ortho(),
    )
    user = captured["user"]
    assert "ANCHOR-ORTHOGONALITY" in user
    assert "ken_french_ff5_mom" in user
    assert "RMW" in user
    # Residual α t-stat (1.87) should appear
    assert "+1.870" in user or "1.870" in user
    # Loadings table rendered with significance markers
    assert "***" in user   # RMW |t|=10 is *** (>2.58)


def test_assess_self_doubt_warns_when_anchor_orthogonality_absent(
    monkeypatch,
):
    """If anchor_orthogonality is None, prompt must EXPLICITLY say
    'not computed' so the model doesn't silently assume orthogonality."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0,
        anchor_orthogonality=None,
    )
    assert "ANCHOR-ORTHOGONALITY: not computed" in captured["user"]


def _sample_subsample_stability(unstable: bool = True):
    """Build a representative subsample_stability dict. unstable=True
    matches GP/A pattern (one-window dominance + decay)."""
    if unstable:
        windows = [
            {"start": "1992-02", "end": "2000-03", "n_months": 98,
             "sharpe_ann": 0.251, "nw_t_stat": 0.72,
             "ann_return": 0.0267, "ann_vol": 0.1062},
            {"start": "2000-04", "end": "2008-06", "n_months": 99,
             "sharpe_ann": 1.447, "nw_t_stat": 4.04,
             "ann_return": 0.1274, "ann_vol": 0.0880},
            {"start": "2008-07", "end": "2016-09", "n_months": 99,
             "sharpe_ann": 0.423, "nw_t_stat": 1.23,
             "ann_return": 0.0392, "ann_vol": 0.0928},
            {"start": "2016-10", "end": "2024-12", "n_months": 99,
             "sharpe_ann": 0.683, "nw_t_stat": 1.86,
             "ann_return": 0.0830, "ann_vol": 0.1215},
        ]
        return {
            "n_splits": 4, "n_total_months": 395, "windows": windows,
            "worst_best_sharpe_ratio": 0.174,
            "institutional_stable": False,
            "monotone_decay": False, "monotone_growth": False,
            "decay_slope_per_year": 0.00007, "decay_slope_t": 0.40,
        }
    return {
        "n_splits": 4, "n_total_months": 240,
        "windows": [
            {"start": "2005-01", "end": "2010-12", "n_months": 60,
             "sharpe_ann": 0.85, "nw_t_stat": 2.1,
             "ann_return": 0.085, "ann_vol": 0.10},
        ] * 4,
        "worst_best_sharpe_ratio": 0.95,
        "institutional_stable": True,
        "monotone_decay": False, "monotone_growth": False,
        "decay_slope_per_year": 0.0001, "decay_slope_t": 0.1,
    }


def test_assess_self_doubt_accepts_subsample_stability(monkeypatch):
    """L2-5 Commit 2: subsample_stability kwarg must land in user
    message sent to Sonnet."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(),
        family_hint="PROFITABILITY", n_trials_family=2,
        subsample_stability=_sample_subsample_stability(unstable=True),
    )
    user = captured["user"]
    assert "SUBSAMPLE STABILITY" in user
    assert "n_splits" in user
    assert "worst/best Sharpe ratio" in user
    # GP/A-like unstable case: ratio 0.174 should appear
    assert "0.174" in user
    # Window breakdown rendered
    assert "1992-02" in user
    assert "2000-04" in user
    # McLean-Pontiff comparison rendered (decay pct between halves)
    assert "Pre-pub" in user or "Post-pub" in user


def test_assess_self_doubt_warns_when_subsample_absent(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0,
        subsample_stability=None,
    )
    assert "SUBSAMPLE STABILITY: not computed" in captured["user"]


def _sample_industry_extension(genuine_alpha: bool = False):
    """GP/A pattern (genuine_alpha=False) or PIT SN pattern (=True)."""
    if genuine_alpha:
        return {
            "alpha_full_monthly": 0.005, "alpha_full_annual": 0.06,
            "alpha_full_nw_t": 8.68,  # joint > stage1 (genuine)
            "alpha_full_nw_se": 0.0006,
            "alpha_ff5mom_only_nw_t": 8.07,
            "delta_alpha_monthly": -0.0003,
            "delta_alpha_nw_t_approx": -0.61,  # negative = genuine
            "ff5mom_betas": {"MOM": 0.356, "CMA": -0.337},
            "ff5mom_beta_nw_t": {"MOM": 6.11, "CMA": -2.62},
            "industry_betas": {"NoDur": -0.12, "Utils": 0.05},
            "industry_beta_nw_t": {"NoDur": -1.24, "Utils": 0.88},
            "r2_full": 0.46, "r2_adj_full": 0.44, "n_overlap": 123,
            "industry_names": ["NoDur", "Utils"],
            "nw_lag_used": 3, "window": "2014-01:2024-03",
            "industry_joint_f_test": {
                "f_stat": 0.85, "f_pvalue": 0.62,
                "df_num": 12, "df_denom": 104,
            },
            "industry_snapshot_sha": "abc12345",
            "model_form": "joint_ff5mom_plus_12_industry",
        }
    # GP/A pattern: alpha collapses, industries explain
    return {
        "alpha_full_monthly": -0.0019,
        "alpha_full_annual": -0.0234,
        "alpha_full_nw_t": -1.38,  # joint α negative
        "alpha_full_nw_se": 0.0014,
        "alpha_ff5mom_only_nw_t": 1.88,
        "delta_alpha_monthly": 0.0045,
        "delta_alpha_nw_t_approx": 3.26,  # positive = industries ate alpha
        "ff5mom_betas": {"RMW": 0.65, "HML": -0.32},
        "ff5mom_beta_nw_t": {"RMW": 9.8, "HML": -4.7},
        "industry_betas": {"BusEq": 0.54, "Shops": 0.34, "Manuf": 0.19},
        "industry_beta_nw_t": {"BusEq": 8.46, "Shops": 6.07, "Manuf": 2.60},
        "r2_full": 0.495, "r2_adj_full": 0.473, "n_overlap": 395,
        "industry_names": ["BusEq", "Shops", "Manuf"],
        "nw_lag_used": 5, "window": "1992-02:2024-12",
        "industry_joint_f_test": {
            "f_stat": 8.39, "f_pvalue": 2.7e-31,
            "df_num": 12, "df_denom": 376,
        },
        "industry_snapshot_sha": "def67890",
        "model_form": "joint_ff5mom_plus_12_industry",
    }


def test_assess_self_doubt_accepts_industry_extension_gpa_pattern(monkeypatch):
    """L2-6 Commit 3: GP/A-pattern industry_extension lands in user
    message — Δα positive, joint α negative, industry F p tiny."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(),
        family_hint="PROFITABILITY", n_trials_family=3,
        industry_extension=_sample_industry_extension(genuine_alpha=False),
    )
    user = captured["user"]
    assert "INDUSTRY EXTENSION" in user
    assert "Δα" in user or "delta" in user.lower()
    # GP/A pattern: positive Δα ≈ +3.26
    assert "+3.26" in user
    # α_full = -1.38
    assert "-1.380" in user or "-1.38" in user
    # Top industry tilts rendered: BusEq with *** (t=8.46)
    assert "BusEq" in user
    assert "***" in user
    # Industry F p-value very small
    assert "2.7e-31" in user or "2.700e-31" in user


def test_assess_self_doubt_accepts_industry_extension_pitsn_pattern(monkeypatch):
    """PIT-SN-pattern: industry_extension lands; Δα negative,
    α_full > α_FF5MOM."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(),
        family_hint="OTHER", n_trials_family=0,
        industry_extension=_sample_industry_extension(genuine_alpha=True),
    )
    user = captured["user"]
    assert "INDUSTRY EXTENSION" in user
    # PIT SN: α_full = 8.68 > α_FF5MOM = 8.07
    assert "+8.680" in user or "+8.68" in user
    assert "+8.070" in user or "+8.07" in user
    # Δα negative (-0.61)
    assert "-0.61" in user or "-0.610" in user


def test_assess_self_doubt_accepts_routing_decisions(monkeypatch):
    """Phase 1 Commit 5: routing_decisions audit trail rendered in
    user message + spec 7-axis context rendered."""
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(),
        family_hint="PROFITABILITY", n_trials_family=2,
        routing_decisions=[
            {"lens": "anchor_regression", "action": "executed"},
            {"lens": "industry_extension", "action": "skipped_conditional",
             "reason": "anchor α t-stat below 1.0"},
            {"lens": "cross_asset_extension", "action": "executed"},
            {"lens": "subsample_stability",
             "action": "skipped_inapplicable",
             "reason": "applicable_to does not match spec metadata"},
        ],
    )
    user = captured["user"]
    assert "ROUTING DECISIONS" in user
    assert "anchor_regression" in user
    assert "skipped_conditional" in user
    assert "skipped_inapplicable" in user
    assert "anchor α t-stat below 1.0" in user
    # 7-axis spec context rendered
    assert "ROLE-AWARE ROUTING AXES" in user
    assert "investment_role" in user
    assert "asset_class" in user


def test_assess_self_doubt_warns_when_industry_extension_absent(monkeypatch):
    from engine.agents.strengthener import self_doubt as sd_mod
    captured = {}
    def _spy(**kw):
        captured["user"] = kw.get("user")
        return _mock_llm_self_doubt()
    monkeypatch.setattr(sd_mod, "llm_call", _spy)
    sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0,
        industry_extension=None,
    )
    assert "INDUSTRY EXTENSION: not computed" in captured["user"]


def test_assess_self_doubt_backward_compat_without_anchor_kw(monkeypatch):
    """assess_self_doubt MUST accept the legacy 5-arg call (no
    anchor_orthogonality) — older callers don't know about L2-4."""
    from engine.agents.strengthener import self_doubt as sd_mod
    monkeypatch.setattr(sd_mod, "llm_call",
                          lambda **kw: _mock_llm_self_doubt())
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0,
    )
    assert sd is not None


def test_oversize_strings_truncated_not_dropped(monkeypatch):
    """LLM might emit a caveat slightly over 300 chars — truncate
    not crash."""
    from engine.agents.strengthener import self_doubt as sd_mod
    long_caveat = "x" * 500
    monkeypatch.setattr(sd_mod, "llm_call",
        lambda **kw: _mock_llm_self_doubt(
            caveats=(long_caveat, long_caveat)))
    sd = sd_mod.assess_self_doubt(
        _spec(), _tpl(), family_hint="X", n_trials_family=0)
    assert sd is not None
    assert all(len(c) <= 300 for c in sd.caveats)
