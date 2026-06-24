"""tests/test_signal_registry.py — Commit 1 of the flexibility chain.

Locks the S-class signal registry contract:
  - registry validation (fields exist, kinds/directions/status valid,
    alias canonical round-trip)
  - alias matching parity with the pre-registry regex table
  - direction declared ≠ baked into formulas (formulas return RAW)
  - contract supported_signals derives from registry
  - proposed-status entries excluded from dispatchable set

The heavyweight parity proof is NOT here — it's the existing golden
snapshot suite (test_tier_c_golden_snapshot.py): GP/A through the
migrated template must reproduce α t=1.8797 / RMW β=0.667 to the
pinned tolerances. That suite ran against the pre-registry template;
passing it post-migration IS the byte-level migration proof.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Registry validation
# ────────────────────────────────────────────────────────────────────
def test_registry_validates_clean():
    from engine.research.signal_registry import validate_registry
    errors = validate_registry()
    assert errors == [], f"registry validation errors: {errors}"


def test_nine_grandfathered_signals_dispatchable():
    from engine.research.signal_registry import dispatchable_signals
    assert set(dispatchable_signals()) == {
        "mktcap", "vol_12m", "ret_12_1", "ret_6_1", "reversal_1m",
        "gp_at", "book_to_market", "at_growth", "roe",
    }


def test_funda_signals_derived():
    from engine.research.signal_registry import funda_signals
    assert funda_signals() == frozenset(
        {"gp_at", "book_to_market", "at_growth", "roe"})


def test_required_columns_resolved_through_catalog():
    from engine.research.signal_registry import required_columns
    assert required_columns("gp_at") == ["sale", "cogs", "at"]
    assert required_columns("roe") == ["ni", "ceq"]
    # b/m: ceq from compustat; mktcap is crsp → excluded from funda cols
    assert required_columns("book_to_market") == ["ceq"]


def test_every_field_reference_in_catalog():
    from engine.research.signal_registry import (
        SIGNAL_REGISTRY, FIELD_CATALOG,
    )
    for key, sdef in SIGNAL_REGISTRY.items():
        for f in sdef.required_fields:
            assert f in FIELD_CATALOG, f"{key} references unknown field {f}"


# ────────────────────────────────────────────────────────────────────
# Alias matching — parity with the pre-registry table
# (mirrors the template's signal-picker tests; both must agree)
# ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("hint,expected", [
    (("gross_profitability",), "gp_at"),
    (("log_book_to_market",), "book_to_market"),
    (("asset_growth_yoy",), "at_growth"),
    (("return_on_equity",), "roe"),
    (("momentum_12_1",), "ret_12_1"),
    (("mom_6_1",), "ret_6_1"),
    (("short_term_reversal",), "reversal_1m"),
    (("low_volatility",), "vol_12m"),
    (("market_equity",), "mktcap"),
    (("intraday_overnight_drift",), None),
    ((), None),
])
def test_alias_matching(hint, expected):
    from engine.research.signal_registry import match_signal_key
    assert match_signal_key(hint) == expected


def test_canonical_keys_round_trip():
    """Every signal's own key resolves back to itself."""
    from engine.research.signal_registry import (
        SIGNAL_REGISTRY, match_signal_key,
    )
    for key in SIGNAL_REGISTRY:
        assert match_signal_key((key,)) == key, (
            f"canonical {key!r} resolves to {match_signal_key((key,))!r}")


# ────────────────────────────────────────────────────────────────────
# Direction declared, formulas RAW
# ────────────────────────────────────────────────────────────────────
def test_long_low_signals_declared():
    """The 4 inverted anomalies carry direction in the MANIFEST, not
    hidden minus signs in formula bodies."""
    from engine.research.signal_registry import SIGNAL_REGISTRY
    expected_low = {"mktcap", "vol_12m", "reversal_1m", "at_growth"}
    actual_low = {k for k, s in SIGNAL_REGISTRY.items()
                    if s.direction == "long_low"}
    assert actual_low == expected_low


def test_crsp_formula_returns_raw_value():
    """mktcap formula returns mktcap AS-IS (positive); orientation is
    the template's job. If someone re-bakes a minus sign into the
    formula, double-negation would silently flip the factor."""
    from engine.research.signal_registry import SIGNAL_REGISTRY
    from types import SimpleNamespace
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    mc = pd.DataFrame({1: [10.0]*6, 2: [20.0]*6}, index=idx)
    rets = pd.DataFrame({1: [0.01]*6, 2: [0.02]*6}, index=idx)
    raw = SIGNAL_REGISTRY["mktcap"].formula(
        SimpleNamespace(rets=rets, mktcap=mc))
    assert (raw.values > 0).all(), "mktcap formula must return RAW (+)"


def test_template_applies_direction_centrally():
    """_build_signal output for mktcap = NEGATED raw (long_low)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _build_signal,
    )
    rows = []
    for me in pd.date_range("2020-01-31", periods=6, freq="ME"):
        for permno, mc in ((1, 10.0), (2, 20.0)):
            rows.append({"month_end": me, "permno": permno,
                           "ret": 0.01, "mktcap": mc})
    panel = pd.DataFrame(rows)
    sig = _build_signal(panel, "mktcap")
    # Smaller cap (permno 1) must have HIGHER score (long side)
    assert (sig[1] > sig[2]).all()


def test_funda_long_formula_at_growth_raw():
    """at_growth formula returns RAW growth (high growth = high raw);
    direction long_low handles the inversion."""
    from engine.research.signal_registry import SIGNAL_REGISTRY
    idx = list(range(26))
    merged = pd.DataFrame({
        "permno":    [1] * 13 + [2] * 13,
        "month_end": list(pd.date_range("2020-01-31", periods=13, freq="ME")) * 2,
        # permno 1 doubles assets (high growth); permno 2 flat
        "at": [100 + i * 10 for i in range(13)] + [100.0] * 13,
    }, index=idx)
    raw = SIGNAL_REGISTRY["at_growth"].formula(merged)
    # After 12m lag becomes available (last row of each permno):
    g1 = raw.iloc[12]    # permno 1: (220/100)-1 = 1.2
    g2 = raw.iloc[25]    # permno 2: 0.0
    assert g1 > 1.0
    assert abs(g2) < 1e-9


# ────────────────────────────────────────────────────────────────────
# Contract derivation + status gate
# ────────────────────────────────────────────────────────────────────
def test_contract_supported_signals_derives_from_registry():
    from engine.agents.strengthener.templates._template_contract import (
        CONTRACT_REGISTRY,
    )
    from engine.research.signal_registry import dispatchable_signals
    contract = CONTRACT_REGISTRY["cross_sec_us_equities"]
    assert set(contract.supported_signals) == set(dispatchable_signals())


def test_proposed_status_excluded_from_dispatchable(monkeypatch):
    """A proposed entry is visible in the registry but NOT in
    dispatchable_signals() — it can't burn dispatch quota until the
    human approves its verification card."""
    import engine.research.signal_registry as sr
    probe = sr.SignalDefinition(
        key="probe_xyz", kind="crsp_panel", direction="long_high",
        family="TEST", required_fields=("crsp.msf.ret",),
        formula=lambda ctx: ctx.rets,
        aliases=(r"probe_xyz_nomatch",),
        paper_citation="n/a", pit_notes="n/a",
        status="proposed",
    )
    monkeypatch.setitem(sr.SIGNAL_REGISTRY, "probe_xyz", probe)
    assert "probe_xyz" in sr.SIGNAL_REGISTRY
    assert "probe_xyz" not in sr.dispatchable_signals()


def test_template_picker_reexport_parity():
    """The template's _pick_signal_key must agree with the registry
    matcher on every dispatchable key (no drift between the two
    entry points)."""
    from engine.agents.strengthener.templates.cross_sec_us_equities import (
        _pick_signal_key,
    )
    from engine.research.signal_registry import (
        match_signal_key, dispatchable_signals,
    )
    for key in dispatchable_signals():
        assert _pick_signal_key((key,)) == match_signal_key((key,)) == key
