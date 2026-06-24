"""tests/test_factor_spec_extractor.py — Tier C-1.

Tests the factor backtest spec extractor. llm_call mocked so tests
are offline + fast + free. The extractor is pure LLM-spec-translation
+ enum validation — NO I/O in scope of this module.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


# ────────────────────────────────────────────────────────────────────
# Test fixtures
# ────────────────────────────────────────────────────────────────────
def _factor_hyp(
    *,
    hypothesis_id="hid_test",
    mechanism_family="PROFITABILITY",
    mechanism_subtype="gross_profitability_cross_section",
    predicted_direction="positive",
    predicted_magnitude="annualized Sharpe 0.4-0.7",
    required_data=("WRDS CRSP monthly", "Compustat funda annual"),
    test_methodology=("Sort top 3000 US equities by gross "
                       "profitability (GP/AT), long top decile short "
                       "bottom decile, dollar-neutral, monthly rebal."),
    claim="Gross profitability predicts cross-section of returns.",
    source_chunk_ids=("chunk_1",),
    synthesizes_paper_ids=(),
    synthesizes_event_ids=(),
    extraction_method="llm_extract",
):
    """Build a Hypothesis-shaped namespace with the fields the
    extractor reads."""
    return SimpleNamespace(
        hypothesis_id        = hypothesis_id,
        mechanism_family     = SimpleNamespace(value=mechanism_family),
        mechanism_subtype    = mechanism_subtype,
        predicted_direction  = SimpleNamespace(value=predicted_direction),
        predicted_magnitude  = predicted_magnitude,
        required_data        = required_data,
        test_methodology     = test_methodology,
        claim                = claim,
        source_chunk_ids     = source_chunk_ids,
        synthesizes_paper_ids = synthesizes_paper_ids,
        synthesizes_event_ids = synthesizes_event_ids,
        extraction_method    = SimpleNamespace(value=extraction_method),
    )


def _mock_llm_spec(
    *,
    signal_kind="cross_sectional_rank",
    universe="us_equities_top_3000",
    date_range="2000-01:2024-12",
    signal_inputs=("compustat.funda.gp_at",),
    rebal="monthly",
    weighting="decile_long_short_dollar_neutral",
    expected_holding_period="monthly",
    min_obs_months=120,
    pit_audits=("restatement", "lookahead", "survivorship"),
    cost_model="engine.execution.cost_model.basic",
    rationale="GP/AT is a textbook cross-sectional rank factor.",
    model="claude-sonnet-4-6",
    # Phase 6 (2026-06-08): Optional B-class FactorSpec v2 payload fields
    universe_size=None,
    n_buckets=None,
    signal_lookback_m=None,
    signal_skip_m=None,
    vol_target_annual=None,
    weighting_scheme_alt=None,
    # Phase 1 (2026-06-09): role-aware routing axes
    investment_role=None,
    statistical_role=None,
    asset_class=None,
    mechanism=None,
    horizon=None,
    capacity_tier=None,
    data_dependency_type=None,
    regime_sensitivity=None,
):
    """Build an llm_call result with a valid emit_factor_spec tool
    call payload. B-class fields included only when non-None
    (mimics LLM behavior: omit / null / explicit value)."""
    payload = {
        "signal_kind": signal_kind,
        "universe": universe,
        "date_range": date_range,
        "signal_inputs": list(signal_inputs),
        "rebal": rebal,
        "weighting": weighting,
        "expected_holding_period": expected_holding_period,
        "min_obs_months": min_obs_months,
        "pit_audits": list(pit_audits),
        "cost_model": cost_model,
        "rationale": rationale,
    }
    if universe_size is not None:
        payload["universe_size"] = universe_size
    if n_buckets is not None:
        payload["n_buckets"] = n_buckets
    if signal_lookback_m is not None:
        payload["signal_lookback_m"] = signal_lookback_m
    if signal_skip_m is not None:
        payload["signal_skip_m"] = signal_skip_m
    if vol_target_annual is not None:
        payload["vol_target_annual"] = vol_target_annual
    if weighting_scheme_alt is not None:
        payload["weighting_scheme_alt"] = weighting_scheme_alt
    # Role-aware axes (only emit when test populates)
    for fname, val in (
        ("investment_role",       investment_role),
        ("statistical_role",      statistical_role),
        ("asset_class",           asset_class),
        ("mechanism",             mechanism),
        ("horizon",               horizon),
        ("capacity_tier",         capacity_tier),
        ("data_dependency_type",  data_dependency_type),
        ("regime_sensitivity",    regime_sensitivity),
    ):
        if val is not None:
            payload[fname] = val
    return SimpleNamespace(
        text       = "",
        tool_calls = (SimpleNamespace(
            name="emit_factor_spec",
            input=payload,
        ),),
        model      = model,
    )


# ────────────────────────────────────────────────────────────────────
# Eligibility check
# ────────────────────────────────────────────────────────────────────
def test_is_factor_accepts_factor_hypothesis_with_provenance():
    from engine.agents.strengthener.factor_spec_extractor import (
        is_factor_hypothesis,
    )
    assert is_factor_hypothesis(_factor_hyp()) is True


def test_is_factor_rejects_zero_direction():
    """Procedural hypotheses (predicted_direction=zero) go through
    procedural_dispatcher, not factor_spec_extractor."""
    from engine.agents.strengthener.factor_spec_extractor import (
        is_factor_hypothesis,
    )
    assert is_factor_hypothesis(
        _factor_hyp(predicted_direction="zero")) is False


def test_is_factor_rejects_methodology_subtype():
    """Methodology / multi-testing / overfit / microstructure
    research routes to escape hatch directly without spending Sonnet
    $0.03 (audit 2026-06-08 found ~40 OTHER family rows like this)."""
    from engine.agents.strengthener.factor_spec_extractor import (
        is_factor_hypothesis,
    )
    for st in ["multiple_testing_correction_for_factor_discovery",
                "backtest_overfitting_with_compensation_effects",
                "false_discovery_rate_in_published_factors",
                "anomaly_data_snooping_decay_pre_and_post_discovery",
                "optimal_stopping_strategy_selection",
                "transaction_cost_estimation_price_impact",
                "permanent_vs_temporary_price_impact"]:
        h = _factor_hyp(mechanism_subtype=st)
        assert is_factor_hypothesis(h) is False, f"{st} should be filtered"


def test_is_factor_rejects_no_provenance():
    """Without source_chunk_ids, synthesizes_paper_ids, or
    synthesizes_event_ids, extractor has no grounding."""
    from engine.agents.strengthener.factor_spec_extractor import (
        is_factor_hypothesis,
    )
    h = _factor_hyp(source_chunk_ids=(), synthesizes_paper_ids=(),
                      synthesizes_event_ids=())
    assert is_factor_hypothesis(h) is False


def test_is_factor_accepts_synthesis_with_paper_provenance():
    """LLM_SYNTHESIS hypotheses have synthesizes_paper_ids instead
    of source_chunk_ids — should still pass eligibility."""
    from engine.agents.strengthener.factor_spec_extractor import (
        is_factor_hypothesis,
    )
    h = _factor_hyp(source_chunk_ids=(),
                      synthesizes_paper_ids=("paper_42",),
                      extraction_method="llm_synthesis")
    assert is_factor_hypothesis(h) is True


# ────────────────────────────────────────────────────────────────────
# Spec extraction — happy path
# ────────────────────────────────────────────────────────────────────
def test_extract_factor_spec_happy_path(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
                          lambda **kw: _mock_llm_spec())
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec is not None
    assert spec.signal_kind == "cross_sectional_rank"
    assert spec.universe == "us_equities_top_3000"
    assert spec.date_range == "2000-01:2024-12"
    assert spec.rebal == "monthly"
    assert spec.weighting == "decile_long_short_dollar_neutral"
    assert spec.min_obs_months == 120
    assert "restatement" in spec.pit_audits
    assert spec.cost_model.startswith("engine.execution.cost_model")
    assert spec.model == "claude-sonnet-4-6"
    assert spec.hypothesis_id == "hid_test"
    assert spec.is_escape_hatch() is False


def test_extract_factor_spec_passes_workload_correctly(monkeypatch):
    """Verify workload routing — guards against silently using the
    wrong (cheaper Haiku / wrong-provider) model."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    captured = {}
    def _spy(**kw):
        captured.update(kw)
        return _mock_llm_spec()
    monkeypatch.setattr(fse, "llm_call", _spy)
    fse.extract_factor_spec(_factor_hyp())
    assert captured.get("workload") == "strengthener_factor_spec"
    assert captured.get("agent_id") == "strengthener_factor_spec"
    assert captured.get("scope") == "tier_c_factor_spec_extractor"


# ────────────────────────────────────────────────────────────────────
# Spec extraction — escape hatch
# ────────────────────────────────────────────────────────────────────
def test_extract_factor_spec_escape_hatch(monkeypatch):
    """When LLM signals requires_custom_code, extractor returns a
    FactorSpec with is_escape_hatch()=True so caller can surface
    to /approvals as a custom-code reminder."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(
            signal_kind="requires_custom_code",
            universe="unknown_universe",
            rationale=("This hypothesis requires intraday price-impact "
                       "regression with live institutional trade data; "
                       "no template covers it."),
        ))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec is not None
    assert spec.is_escape_hatch() is True
    assert spec.universe == "unknown_universe"


# ────────────────────────────────────────────────────────────────────
# Spec extraction — defensive failures
# ────────────────────────────────────────────────────────────────────
def test_extract_returns_none_on_llm_exception(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    def _boom(**kw):
        raise RuntimeError("api timeout")
    monkeypatch.setattr(fse, "llm_call", _boom)
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_returns_none_when_tool_not_called(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: SimpleNamespace(text="sorry no spec",
                                       tool_calls=(),
                                       model="claude-sonnet-4-6"))
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_rejects_unknown_signal_kind(monkeypatch):
    """LLM hallucinates a new signal_kind not in the controlled
    enum — extractor must reject, not silently accept."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(signal_kind="machine_learning_oof"))
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_rejects_unknown_universe(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(universe="china_a_shares"))
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_rejects_unknown_rebal(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(rebal="continuous"))
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_rejects_unknown_weighting(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(weighting="markowitz_min_variance"))
    assert fse.extract_factor_spec(_factor_hyp()) is None


def test_extract_rejects_bad_date_range_format(monkeypatch):
    from engine.agents.strengthener import factor_spec_extractor as fse
    for bad in ["2000-01-01:2024-12-31", "2000:2024",
                  "2000-01 to 2024-12", "since 2000", ""]:
        def _stub(_b=bad, **_kw):
            return _mock_llm_spec(date_range=_b)
        monkeypatch.setattr(fse, "llm_call", _stub)
        assert fse.extract_factor_spec(_factor_hyp()) is None, bad


def test_extract_returns_none_when_ineligible(monkeypatch):
    """Eligibility gate runs BEFORE LLM call — saves $0.03 on
    candidates that obviously don't fit."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    called = {"n": 0}
    def _counter(**kw):
        called["n"] += 1
        return _mock_llm_spec()
    monkeypatch.setattr(fse, "llm_call", _counter)
    # zero-direction hypothesis: should skip LLM entirely
    assert fse.extract_factor_spec(
        _factor_hyp(predicted_direction="zero")) is None
    assert called["n"] == 0


# ────────────────────────────────────────────────────────────────────
# Phase 6 (2026-06-08): Optional B-class FactorSpec v2 propagation
# ────────────────────────────────────────────────────────────────────
def test_extract_b_class_all_none_by_default(monkeypatch):
    """When LLM omits B-class fields (most common case), FactorSpec
    fields are None → template uses default."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call", lambda **kw: _mock_llm_spec())
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec is not None
    assert spec.universe_size is None
    assert spec.n_buckets is None
    assert spec.signal_lookback_m is None
    assert spec.signal_skip_m is None
    assert spec.vol_target_annual is None
    assert spec.weighting_scheme_alt is None


def test_extract_b_class_universe_size_500_for_small_cap(monkeypatch):
    """When LLM populates universe_size (e.g. small-cap hypothesis),
    FactorSpec carries through to dispatcher."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(universe_size=500))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.universe_size == 500


def test_extract_b_class_decile(monkeypatch):
    """LLM populates n_buckets=10 for decile sort hypothesis."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(n_buckets=10))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.n_buckets == 10


def test_extract_b_class_tsmom_6_1(monkeypatch):
    """TSMOM 6-1 variant: signal_lookback_m=6, signal_skip_m=1."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(signal_lookback_m=6,
                                       signal_skip_m=1))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.signal_lookback_m == 6
    assert spec.signal_skip_m == 1


def test_extract_b_class_vol_target_15pct(monkeypatch):
    """Hypothesis specifying 15% vol target."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(vol_target_annual=0.15))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.vol_target_annual == 0.15


def test_extract_b_class_value_weighted(monkeypatch):
    """LLM populates weighting_scheme_alt='vw'."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(weighting_scheme_alt="vw"))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.weighting_scheme_alt == "vw"


def test_extract_b_class_bad_enum_ignored(monkeypatch):
    """Defensive: LLM hallucinates weighting_scheme_alt='markowitz' →
    parser drops it (None) with warning, doesn't crash."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(weighting_scheme_alt="markowitz"))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.weighting_scheme_alt is None


def test_extract_b_class_bad_int_ignored(monkeypatch):
    """Defensive: LLM hallucinates universe_size='lots' → parser
    drops it (None) with warning, doesn't crash."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(universe_size="lots"))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.universe_size is None


def test_extract_b_class_combo_full_variant(monkeypatch):
    """End-to-end LLM proposing GP/A on top-500 + decile + VW (the
    Layer 3 critic-style variant)."""
    from engine.agents.strengthener import factor_spec_extractor as fse
    monkeypatch.setattr(fse, "llm_call",
        lambda **kw: _mock_llm_spec(
            universe_size=500, n_buckets=10,
            weighting_scheme_alt="vw"))
    spec = fse.extract_factor_spec(_factor_hyp())
    assert spec.universe_size == 500
    assert spec.n_buckets == 10
    assert spec.weighting_scheme_alt == "vw"


# ────────────────────────────────────────────────────────────────────
# Output dataclass integrity
# ────────────────────────────────────────────────────────────────────
def test_factor_spec_is_frozen():
    """FactorSpec is frozen so callers can't mutate signal_kind
    post-extraction (which would break dispatcher routing trust)."""
    from engine.agents.strengthener.factor_spec_extractor import (
        FactorSpec,
    )
    spec = FactorSpec(
        hypothesis_id="h1", signal_kind="carry",
        universe="fx_g10", date_range="2000-01:2024-12",
        signal_inputs=("fx.spot",), rebal="monthly",
        weighting="rank_weighted",
        expected_holding_period="monthly", min_obs_months=180,
        pit_audits=(), cost_model="engine.execution.cost_model.basic",
        rationale="FX carry classical setup.",
        extracted_ts="2026-06-08T00:00:00Z",
        model="claude-sonnet-4-6",
    )
    with pytest.raises(Exception):  # dataclass FrozenInstanceError
        spec.signal_kind = "vrp"   # type: ignore[misc]


def test_signal_kinds_and_universe_enums_have_escape_hatches():
    """Enums must include their respective 'no fit' values, or the
    extractor can't legitimately route to escape hatch."""
    from engine.agents.strengthener.factor_spec_extractor import (
        SIGNAL_KINDS, UNIVERSES,
    )
    assert "requires_custom_code" in SIGNAL_KINDS
    assert "unknown_universe" in UNIVERSES


# ────────────────────────────────────────────────────────────────────
# Phase 1 (2026-06-09): Role-aware routing axes
# Per docs/spec_role_aware_test_routing.md v2 (commit 2ca50bf2)
# ────────────────────────────────────────────────────────────────────
def test_role_axes_all_none_by_default(monkeypatch):
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.factor_spec_extractor.llm_call",
        lambda **kw: _mock_llm_spec(),
    )
    spec = extract_factor_spec(_factor_hyp())
    assert spec is not None
    assert spec.investment_role      is None
    assert spec.statistical_role     is None
    assert spec.asset_class          is None
    assert spec.mechanism            is None
    assert spec.horizon              is None
    assert spec.capacity_tier        is None
    assert spec.data_dependency_type is None
    assert spec.regime_sensitivity   is None


def test_role_axes_investment_role_extracted(monkeypatch):
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.factor_spec_extractor.llm_call",
        lambda **kw: _mock_llm_spec(investment_role="insurance"),
    )
    spec = extract_factor_spec(_factor_hyp())
    assert spec.investment_role == "insurance"


def test_role_axes_split_independent(monkeypatch):
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.factor_spec_extractor.llm_call",
        lambda **kw: _mock_llm_spec(
            investment_role="alpha",
            statistical_role="arbitrage",
        ),
    )
    spec = extract_factor_spec(_factor_hyp())
    assert spec.investment_role  == "alpha"
    assert spec.statistical_role == "arbitrage"


def test_role_axes_full_population(monkeypatch):
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.factor_spec_extractor.llm_call",
        lambda **kw: _mock_llm_spec(
            investment_role="alpha",
            statistical_role="directional",
            asset_class="equity",
            mechanism="behavioral",
            horizon="monthly",
            capacity_tier="100m_to_1b",
            data_dependency_type="fundamental",
            regime_sensitivity="known_regime_break",
        ),
    )
    spec = extract_factor_spec(_factor_hyp())
    assert spec.investment_role      == "alpha"
    assert spec.statistical_role     == "directional"
    assert spec.asset_class          == "equity"
    assert spec.mechanism            == "behavioral"
    assert spec.horizon              == "monthly"
    assert spec.capacity_tier        == "100m_to_1b"
    assert spec.data_dependency_type == "fundamental"
    assert spec.regime_sensitivity   == "known_regime_break"


def test_role_axes_bad_enum_ignored(monkeypatch):
    from engine.agents.strengthener.factor_spec_extractor import (
        extract_factor_spec,
    )
    monkeypatch.setattr(
        "engine.agents.strengthener.factor_spec_extractor.llm_call",
        lambda **kw: _mock_llm_spec(
            investment_role="ALPHA-INVENTED-VALUE",
            asset_class="crypto",
        ),
    )
    spec = extract_factor_spec(_factor_hyp())
    assert spec.investment_role is None
    assert spec.asset_class     is None


def _spec_no_role_axes(**overrides):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id           = "hid_test",
        signal_kind             = "cross_sectional_rank",
        universe                = "us_equities_top_3000",
        date_range              = "2000-01:2024-12",
        signal_inputs           = ("compustat.funda.gp_at",),
        rebal                   = "monthly",
        weighting               = "decile_long_short_dollar_neutral",
        expected_holding_period = "monthly",
        min_obs_months          = 120,
        pit_audits              = ("lookahead",),
        cost_model              = "engine.execution.cost_model.basic",
        rationale               = "test",
        extracted_ts            = "2026-06-09T00:00:00Z",
        model                   = "claude-sonnet-4-6",
    )
    base.update(overrides)
    return FactorSpec(**base)


def test_infer_legacy_axes_equity_directional():
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    inferred = infer_legacy_axes(_spec_no_role_axes())
    assert inferred["asset_class"]          == "equity"
    assert inferred["statistical_role"]     == "directional"
    assert inferred["data_dependency_type"] == "fundamental"
    assert inferred["investment_role"]      == "alpha"
    assert inferred["horizon"]              == "monthly"
    assert inferred["capacity_tier"]        == "unknown"
    assert inferred["regime_sensitivity"]   == "unknown"


def test_infer_legacy_axes_cross_asset_carry():
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    inferred = infer_legacy_axes(_spec_no_role_axes(
        signal_kind="carry",
        signal_inputs=("crsp.msf.price", "fred.macro.dxy"),
    ))
    assert inferred["asset_class"]      == "cross_asset"
    assert inferred["statistical_role"] == "directional"
    assert inferred["data_dependency_type"] == "price"


def test_infer_legacy_axes_unknown_signal_kind_no_inference():
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    inferred = infer_legacy_axes(_spec_no_role_axes(
        signal_kind="weird_new_signal_kind",
    ))
    assert "asset_class"      not in inferred
    assert "statistical_role" not in inferred
    assert inferred["investment_role"]    == "alpha"
    assert inferred["capacity_tier"]      == "unknown"
    assert inferred["regime_sensitivity"] == "unknown"


def test_infer_legacy_axes_respects_explicit_spec_values():
    from engine.agents.strengthener.factor_spec_extractor import (
        infer_legacy_axes,
    )
    inferred = infer_legacy_axes(_spec_no_role_axes(
        asset_class="fixed_income",
        investment_role="hedge",
    ))
    assert "asset_class"      not in inferred
    assert "investment_role"  not in inferred


def test_role_axes_listed_in_tool_schema():
    from engine.agents.strengthener.factor_spec_extractor import (
        _SPEC_TOOL,
    )
    props = _SPEC_TOOL["input_schema"]["properties"]
    expected = {
        "investment_role", "statistical_role", "asset_class",
        "mechanism", "horizon", "capacity_tier",
        "data_dependency_type", "regime_sensitivity",
    }
    assert expected <= set(props.keys())
    for k in expected:
        assert "null" in props[k]["type"]
