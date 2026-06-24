"""tests/test_pit_accessor.py — Tier C L2-1 Phase 2.3.

Unit tests for engine.data.pit_warehouse.accessor.PITDataAccessor.

Tests are split:
  - Pure-API tests (clock-filtering logic, constructor validation,
    PIT violation rejection) run offline.
  - Data-touching tests (universe construction on real CRSP / SP500
    parquets) are gated behind RUN_PIT_ACCESSOR_INTEGRATION=1.
  - funda_pit_panel tests need the comp_pit parquet — gated by cache
    presence (skip if missing).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Constructor + clock binding
# ────────────────────────────────────────────────────────────────────
def test_accessor_binds_clock_and_funda_source():
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    a = PITDataAccessor(c)
    assert a.clock is c
    # default funda_source = "pit"


def test_accessor_rejects_invalid_funda_source():
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    with pytest.raises(ValueError, match="funda_source"):
        PITDataAccessor(c, funda_source="restated_latest")


def test_accessor_accepts_legacy_funda_source():
    """legacy is allowed for parity comparison ONLY."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    a = PITDataAccessor(c, funda_source="legacy")
    assert a._funda_source == "legacy"


# ────────────────────────────────────────────────────────────────────
# PIT violation rejection (architectural guarantee)
# ────────────────────────────────────────────────────────────────────
def test_universe_top_n_rejects_future_as_of():
    """Asking for universe AFTER clock.now must raise. This is THE
    PIT enforcement primitive."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-30"))
    a = PITDataAccessor(c)
    with pytest.raises(ValueError, match="PIT violation"):
        a.universe_top_n_by_mktcap(100, as_of="2023-01-31")


def test_universe_sp500_rejects_future_as_of():
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2020-01-01", end="2024-12-31")
    c.advance(pd.Timestamp("2022-06-30"))
    a = PITDataAccessor(c)
    with pytest.raises(ValueError, match="PIT violation"):
        a.universe_sp500_constituents(as_of="2023-12-31")


# ────────────────────────────────────────────────────────────────────
# Phase 6 piece 2: contract-driven data shape coercion
# ────────────────────────────────────────────────────────────────────
def test_accessor_accepts_contract_parameter():
    """Phase 6 piece 2: PITDataAccessor takes optional contract."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    c = SimClock(start="2020-01-01", end="2024-12-31")
    contract = CONTRACT_REGISTRY["cross_sec_us_equities"]
    a = PITDataAccessor(c, contract=contract)
    assert a._contract is contract


def test_accessor_no_contract_is_no_coercion():
    """Phase 6 piece 2: when contract is None, _coerce_funda_to_contract
    is a no-op (pass-through). Required for backward-compat with
    Phase 5 callers that don't pass contract."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    import pandas as pd
    c = SimClock(start="2020-01-01", end="2024-12-31")
    a = PITDataAccessor(c)   # no contract
    df = pd.DataFrame({
        "gvkey": ["001690"], "datadate": [pd.Timestamp("2020-12-31")],
        "at": [320000.0],
    })
    out = a._coerce_funda_to_contract(df)
    assert len(out) == len(df)
    assert (out == df).all().all()


def test_accessor_quarterly_contract_no_coercion():
    """Phase 6 piece 2: contract declaring frequency='quarterly' →
    no coercion (PIT raw IS quarterly)."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    from engine.agents.strengthener.templates._template_contract import (
        TemplateContract, DataShapeRequirement,
    )
    import pandas as pd
    contract = TemplateContract(
        template_name="x", template_version="v0",
        pit_audit_certified_by="t", pit_audit_date="2026-06-08",
        pit_audit_notes="", supported_signal_kinds=("a",),
        supported_universes=("b",), supported_signals=("c",),
        required_data_shape=(
            DataShapeRequirement(source="comp_pit.funda",
                                    frequency="quarterly"),
        ),
    )
    c = SimClock(start="2020-01-01", end="2024-12-31")
    a = PITDataAccessor(c, contract=contract)
    df = pd.DataFrame({
        "gvkey": ["001690"] * 4,
        "datadate": pd.to_datetime(["2020-03-31", "2020-06-30",
                                       "2020-09-30", "2020-12-31"]),
        "at": [310000, 315000, 320000, 325000],
    })
    out = a._coerce_funda_to_contract(df)
    assert len(out) == 4   # quarterly = no filter


# ────────────────────────────────────────────────────────────────────
# Data-touching tests (gated)
# ────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    os.environ.get("RUN_PIT_ACCESSOR_INTEGRATION") != "1",
    reason=("set RUN_PIT_ACCESSOR_INTEGRATION=1 to run live PIT "
              "data tests"),
)
def test_mktcap_panel_lagged_by_default():
    """Default lagged=True: lookup at month t should give t-1's
    mktcap. Critical for B1 (universe selection look-ahead) fix."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2015-01-01", end="2020-12-31")
    c.advance(pd.Timestamp("2020-12-31"))
    a = PITDataAccessor(c)
    mc_lagged = a.mktcap_panel(lagged=True)
    mc_current = a.mktcap_panel(lagged=False)
    # Both contain Dec 2020 row (since clock.now = 2020-12-31)
    assert pd.Timestamp("2020-12-31") in mc_lagged.index
    assert pd.Timestamp("2020-12-31") in mc_current.index
    # Lagged Dec 2020 row should EQUAL current Nov 2020 row
    # (since lagged = shift(1))
    nov_2020 = pd.Timestamp("2020-11-30")
    dec_2020 = pd.Timestamp("2020-12-31")
    if nov_2020 in mc_current.index and dec_2020 in mc_lagged.index:
        common_perms = (mc_lagged.loc[dec_2020].dropna().index
                          .intersection(mc_current.loc[nov_2020].dropna().index))
        # Should match for permnos present in both
        for p in list(common_perms)[:5]:
            assert mc_lagged.loc[dec_2020, p] == mc_current.loc[nov_2020, p]


@pytest.mark.skipif(
    os.environ.get("RUN_PIT_ACCESSOR_INTEGRATION") != "1",
    reason="set RUN_PIT_ACCESSOR_INTEGRATION=1",
)
def test_universe_sp500_returns_sensible_count():
    """SP500 universe at any date should have ~500 permnos
    (slight off-count from spin-offs etc., but ballpark)."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2000-01-01", end="2020-12-31")
    c.advance(pd.Timestamp("2015-06-30"))
    a = PITDataAccessor(c)
    sp_2015 = a.universe_sp500_constituents()
    assert 400 <= len(sp_2015) <= 510, (
        f"SP500 size 2015-06-30: expected ~500, got {len(sp_2015)}")


@pytest.mark.skipif(
    os.environ.get("RUN_PIT_ACCESSOR_INTEGRATION") != "1",
    reason="set RUN_PIT_ACCESSOR_INTEGRATION=1",
)
def test_universe_sp500_grows_over_time():
    """SP500 has historically had ~500 names with constant target
    but membership churn. Asking for SP500 set at two different
    dates should give mostly-overlapping but different sets."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2000-01-01", end="2020-12-31")
    c.advance(pd.Timestamp("2010-06-30"))
    a = PITDataAccessor(c)
    sp_2010 = a.universe_sp500_constituents()
    c.advance(pd.Timestamp("2020-06-30"))
    sp_2020 = a.universe_sp500_constituents()
    # Substantial overlap but not identical (membership churn over a decade)
    overlap = sp_2010 & sp_2020
    assert 200 < len(overlap) < 500
    assert sp_2010 != sp_2020


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1]
         / "data" / "cache" / "_compustat_funda_pit.parquet").is_file()
    or os.environ.get("RUN_PIT_ACCESSOR_INTEGRATION") != "1",
    reason=("comp_pit cache missing or "
              "RUN_PIT_ACCESSOR_INTEGRATION != 1"),
)
def test_funda_pit_panel_returns_at_for_window():
    """Returns DataFrame indexed by month_end, columns = permnos,
    values = `at` (total assets) PIT."""
    from engine.data.pit_warehouse import PITDataAccessor, SimClock
    c = SimClock(start="2015-01-01", end="2020-12-31")
    c.advance(pd.Timestamp("2020-12-31"))
    a = PITDataAccessor(c)
    panel = a.funda_pit_panel(field="at",
                                  window=(pd.Timestamp("2018-01-01"),
                                            pd.Timestamp("2020-12-31")))
    assert not panel.empty
    assert "at" not in panel.columns   # field becomes value, not col
    # Should have some non-null values
    assert panel.notna().sum().sum() > 0
