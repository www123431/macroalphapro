"""tests/test_template_contract.py — Tier C L2-1 Phase 5.

Unit tests for engine.agents.strengthener.templates._template_contract:
TemplateContract dataclass, registry, freshness check, scope lookup.
Plus dispatcher gate #10 integration tests.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────────────
# TemplateContract dataclass
# ────────────────────────────────────────────────────────────────────
def test_contract_is_frozen():
    """Contracts are immutable post-construction — modifications
    require new commit + audit cert."""
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract,
    )
    c = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="user", pit_audit_date="2026-06-08",
        pit_audit_notes="t",
        supported_signal_kinds=("a",), supported_universes=("b",),
        supported_signals=("c",),
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        c.template_version = "v1"   # type: ignore[misc]


def test_contract_is_fresh_default_today():
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract,
    )
    today = _dt.date.today().isoformat()
    c = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="user", pit_audit_date=today,
        pit_audit_notes="", supported_signal_kinds=("a",),
        supported_universes=("b",), supported_signals=("c",),
    )
    assert c.is_fresh() is True


def test_contract_is_stale_after_365d():
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract,
    )
    stale = (_dt.date.today() - _dt.timedelta(days=400)).isoformat()
    c = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="user", pit_audit_date=stale,
        pit_audit_notes="", supported_signal_kinds=("a",),
        supported_universes=("b",), supported_signals=("c",),
    )
    assert c.is_fresh() is False


def test_contract_is_fresh_with_explicit_as_of():
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract,
    )
    c = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="user", pit_audit_date="2026-06-08",
        pit_audit_notes="", supported_signal_kinds=("a",),
        supported_universes=("b",), supported_signals=("c",),
    )
    # 100d later still fresh
    assert c.is_fresh(_dt.date(2026, 9, 16)) is True
    # 400d later stale
    assert c.is_fresh(_dt.date(2027, 7, 13)) is False


def test_contract_handles_malformed_audit_date():
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract,
    )
    c = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="user", pit_audit_date="not-a-date",
        pit_audit_notes="", supported_signal_kinds=("a",),
        supported_universes=("b",), supported_signals=("c",),
    )
    assert c.is_fresh() is False  # malformed = not fresh


# ────────────────────────────────────────────────────────────────────
# Registry + lookup
# ────────────────────────────────────────────────────────────────────
def test_registry_contains_shipped_templates():
    """Two templates shipped as of L2-1 Phase 5:
       cross_sec_us_equities + tsmom_sector_etf."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    assert "cross_sec_us_equities" in CONTRACT_REGISTRY
    assert "tsmom_sector_etf" in CONTRACT_REGISTRY


def test_contract_for_scope_finds_cross_sec():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("cross_sectional_rank", "us_equities_top_3000")
    assert c is not None
    assert c.template_name == "cross_sec_us_equities"


def test_contract_for_scope_finds_tsmom():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    c = contract_for_scope("time_series_momentum", "us_equities_sector_etf")
    assert c is not None
    assert c.template_name == "tsmom_sector_etf"


def test_contract_for_scope_returns_none_for_unsupported_combo():
    from engine.agents.strengthener.templates._template_contract import (
        contract_for_scope,
    )
    # cross_sec doesn't support fx_g10
    assert contract_for_scope("cross_sectional_rank", "fx_g10") is None
    # tsmom doesn't support us_equities_top_3000
    assert contract_for_scope("time_series_momentum",
                                  "us_equities_top_3000") is None


def test_data_shape_requirement_is_frozen():
    """Phase 6: DataShapeRequirement is immutable per architectural
    doctrine — modifying = new commit + audit cert update."""
    from engine.agents.strengthener.templates._template_contract import (
        DataShapeRequirement,
    )
    d = DataShapeRequirement(source="x", frequency="annual",
                                aggregation="fy_total")
    with pytest.raises(Exception):  # FrozenInstanceError
        d.frequency = "quarterly"   # type: ignore[misc]


def test_data_shape_frequency_vocabulary():
    """Phase 6: FREQUENCIES = controlled vocabulary."""
    from engine.agents.strengthener.templates._template_contract import (
        FREQUENCIES,
    )
    assert FREQUENCIES == frozenset(
        {"annual", "quarterly", "monthly", "daily"})


def test_data_shape_aggregation_vocabulary():
    """Phase 6: AGGREGATIONS = controlled vocabulary (None = no aggregation)."""
    from engine.agents.strengthener.templates._template_contract import (
        AGGREGATIONS,
    )
    assert AGGREGATIONS == frozenset({"fy_total", "trailing_ttm",
                                          "latest_quarter", "fy_end",
                                          None})


def test_cross_sec_contract_declares_annual_compustat():
    """Phase 6: cross_sec_us_equities declares annual frequency for
    comp_pit.funda source — surfaces the B-fix cohort intent
    architecturally."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    c = CONTRACT_REGISTRY["cross_sec_us_equities"]
    assert len(c.required_data_shape) >= 1
    funda_shape = next(
        (s for s in c.required_data_shape
         if s.source == "comp_pit.funda"), None,
    )
    assert funda_shape is not None
    assert funda_shape.frequency == "annual"
    assert funda_shape.aggregation == "fy_total"


def test_cross_sec_contract_declares_monthly_crsp():
    """Phase 6: cross_sec also uses CRSP monthly for prices."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    c = CONTRACT_REGISTRY["cross_sec_us_equities"]
    crsp_shape = next(
        (s for s in c.required_data_shape
         if s.source == "crsp.msf"), None,
    )
    assert crsp_shape is not None
    assert crsp_shape.frequency == "monthly"


def test_tsmom_contract_declares_daily_etf():
    """Phase 6: tsmom_sector_etf declares daily ETF closes."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    c = CONTRACT_REGISTRY["tsmom_sector_etf"]
    assert len(c.required_data_shape) >= 1
    etf_shape = next(
        (s for s in c.required_data_shape
         if "etf" in s.source), None,
    )
    assert etf_shape is not None
    assert etf_shape.frequency == "daily"


def test_all_required_data_shapes_use_valid_vocabulary():
    """Phase 6: every declared shape's frequency + aggregation
    must be in the controlled vocab. Catches typos."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY, FREQUENCIES, AGGREGATIONS,
    )
    for name, c in CONTRACT_REGISTRY.items():
        for shape in c.required_data_shape:
            assert shape.frequency in FREQUENCIES, (
                f"contract {name}: invalid frequency "
                f"{shape.frequency!r}")
            assert shape.aggregation in AGGREGATIONS, (
                f"contract {name}: invalid aggregation "
                f"{shape.aggregation!r}")


def test_all_shipped_contracts_are_fresh():
    """All shipped contracts must be currently fresh (within 365d).
    Catches the case where someone updates a template but forgets to
    bump audit_date."""
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    for name, c in CONTRACT_REGISTRY.items():
        assert c.is_fresh(), f"contract {name} stale: {c.pit_audit_date}"


# ────────────────────────────────────────────────────────────────────
# Dispatcher gate #10 integration
# ────────────────────────────────────────────────────────────────────
def _spec(**kw):
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    base = dict(
        hypothesis_id="hid_t", signal_kind="cross_sectional_rank",
        universe="us_equities_top_3000", date_range="2010-01:2020-12",
        signal_inputs=("crsp.msf.derived.vol_12m",), rebal="monthly",
        weighting="quintile_long_short_dollar_neutral",
        expected_holding_period="monthly", min_obs_months=60,
        pit_audits=(), cost_model="basic", rationale="t",
        extracted_ts="2026-06-08T00:00:00Z", model="claude-sonnet-4-6",
    )
    base.update(kw)
    return FactorSpec(**base)


@pytest.fixture
def tmp_log(tmp_path, monkeypatch):
    from engine.agents.strengthener import factor_dispatcher as fd
    log = tmp_path / "log.jsonl"
    monkeypatch.setattr(fd, "_family_n_trials_now", lambda fam: 0)
    return log


def test_gate_10_passes_certified_cross_sec_combo(tmp_log):
    """cross_sectional_rank + us_equities_top_3000 has a registered
    fresh contract → dispatcher gate passes."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(signal_kind="cross_sectional_rank",
                universe="us_equities_top_3000"),
        spec_approved=True, family_hint="X", log_path=tmp_log,
    )
    assert r is None


def test_gate_10_passes_certified_tsmom_combo(tmp_log):
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(signal_kind="time_series_momentum",
                universe="us_equities_sector_etf",
                signal_inputs=("etf.adj_close.spy",)),
        spec_approved=True, family_hint="X", log_path=tmp_log,
    )
    assert r is None


def test_gate_10_passes_stub_kinds_unshipped(tmp_log):
    """vrp / event_drift are KNOWN-not-yet-shipped stub kinds. They
    bypass gate #10 to let the user see PENDING_TEMPLATE_BUILD
    verdict from the template registry stub — distinguishing 'not
    yet built' from 'TEMPLATE_NOT_CERTIFIED'.

    (carry removed from this list when C-2f shipped its template +
    contract, commit d9785c1d — stale test caught in the 2026-06-09
    full-suite audit, fixed 2026-06-10.)"""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    for sk in ("vrp", "event_drift"):
        r = pre_dispatch_check(
            _spec(signal_kind=sk, universe="us_equities_sp500",
                    signal_inputs=("optionmetrics.standardized_options.x",)),
            spec_approved=True, family_hint="X", log_path=tmp_log,
        )
        assert r is None or r.reason_code != "TEMPLATE_NOT_CERTIFIED", \
            f"{sk} should bypass gate #10"


def test_gate_10_refuses_uncertified_combo(tmp_log):
    """cross_sectional_rank + us_equities_sp500 has NO registered
    contract → dispatcher refuses with TEMPLATE_NOT_CERTIFIED."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(signal_kind="cross_sectional_rank",
                universe="us_equities_sp500"),
        spec_approved=True, family_hint="X", log_path=tmp_log,
    )
    assert r is not None
    assert r.reason_code == "TEMPLATE_NOT_CERTIFIED"
    assert r.metrics["signal_kind"] == "cross_sectional_rank"
    assert r.metrics["universe"] == "us_equities_sp500"


def test_gate_10_skipped_for_escape_hatch(tmp_log):
    """requires_custom_code bypasses gate #10 — escape hatch
    semantics."""
    from engine.agents.strengthener.factor_dispatcher import (
        pre_dispatch_check,
    )
    r = pre_dispatch_check(
        _spec(signal_kind="requires_custom_code",
                universe="unknown_universe",
                signal_inputs=("custom.weird.data",)),
        spec_approved=True, family_hint="X", log_path=tmp_log,
    )
    # Either passes (None) or refused for OTHER reason (not cert)
    if r is not None:
        assert r.reason_code != "TEMPLATE_NOT_CERTIFIED"
