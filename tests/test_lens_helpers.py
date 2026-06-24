"""tests/test_lens_helpers.py — B.2 artifacts contract helper tests.

Per engine.research.lens_helpers: templates DECLARE which column is
the default-cost net PnL; lenses READ that declaration. This test
suite locks the contract:

  1. Explicit pnl_default_col is preferred when present.
  2. Legacy fallback (lowest-bp pnl_net_<N>bp) still works for
     un-migrated templates.
  3. Bad declarations (column missing from df) trigger fallback,
     don't crash.
  4. Gross column resolution mirrors net.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────
@pytest.fixture
def equity_artifacts():
    """Equity template shape — declares pnl_net_13bp."""
    df = pd.DataFrame({
        "pnl_gross":    np.linspace(0.01, 0.02, 60),
        "pnl_net_13bp": np.linspace(0.008, 0.018, 60),
        "pnl_net_80bp": np.linspace(0.004, 0.014, 60),
        "turnover":    np.linspace(0.1, 0.2, 60),
    }, index=pd.date_range("2020-01-31", periods=60, freq="ME"))
    return {
        "pnl_series_df":   df,
        "pnl_default_col": "pnl_net_13bp",
        "pnl_gross_col":   "pnl_gross",
    }


@pytest.fixture
def fx_carry_artifacts():
    """FX carry template shape — declares pnl_net_8bp."""
    df = pd.DataFrame({
        "pnl_gross":    np.linspace(0.01, 0.02, 60),
        "pnl_net_8bp":  np.linspace(0.009, 0.019, 60),
        "pnl_net_24bp": np.linspace(0.007, 0.017, 60),
        "turnover":    np.full(60, 0.07),
    }, index=pd.date_range("2020-01-31", periods=60, freq="ME"))
    return {
        "pnl_series_df":   df,
        "pnl_default_col": "pnl_net_8bp",
        "pnl_gross_col":   "pnl_gross",
    }


@pytest.fixture
def legacy_artifacts():
    """Un-migrated template — no pnl_default_col declaration."""
    df = pd.DataFrame({
        "pnl_gross":    np.linspace(0.01, 0.02, 60),
        "pnl_net_13bp": np.linspace(0.008, 0.018, 60),
        "turnover":     np.linspace(0.1, 0.2, 60),
    }, index=pd.date_range("2020-01-31", periods=60, freq="ME"))
    return {"pnl_series_df": df}


# ────────────────────────────────────────────────────────────────────
# resolve_default_net_col — explicit declaration honored
# ────────────────────────────────────────────────────────────────────
def test_resolve_net_col_prefers_declared(equity_artifacts):
    from engine.research.lens_helpers import resolve_default_net_col
    assert resolve_default_net_col(equity_artifacts) == "pnl_net_13bp"


def test_resolve_net_col_prefers_declared_for_fx(fx_carry_artifacts):
    """FX template declares pnl_net_8bp — must return THAT, not
    accidentally pick pnl_net_24bp via some heuristic."""
    from engine.research.lens_helpers import resolve_default_net_col
    assert resolve_default_net_col(fx_carry_artifacts) == "pnl_net_8bp"


def test_resolve_net_col_legacy_fallback(legacy_artifacts):
    """Un-migrated template (no declaration) → legacy heuristic
    picks lowest-bp net column."""
    from engine.research.lens_helpers import resolve_default_net_col
    assert resolve_default_net_col(legacy_artifacts) == "pnl_net_13bp"


def test_resolve_net_col_legacy_picks_lowest_bp():
    from engine.research.lens_helpers import resolve_default_net_col
    df = pd.DataFrame({
        "pnl_gross":     [0.01] * 5,
        "pnl_net_24bp":  [0.005] * 5,
        "pnl_net_8bp":   [0.009] * 5,
        "pnl_net_16bp":  [0.007] * 5,
    })
    assert resolve_default_net_col({"pnl_series_df": df}) == "pnl_net_8bp"


def test_resolve_net_col_returns_none_when_no_net_cols():
    from engine.research.lens_helpers import resolve_default_net_col
    df = pd.DataFrame({"pnl_gross": [0.01] * 5})
    assert resolve_default_net_col({"pnl_series_df": df}) is None


def test_resolve_net_col_returns_none_on_empty_input():
    from engine.research.lens_helpers import resolve_default_net_col
    assert resolve_default_net_col(None) is None
    assert resolve_default_net_col({}) is None


def test_resolve_net_col_bad_declaration_falls_back_to_legacy():
    """Template declared a column that doesn't exist on the df →
    helper logs a warning and falls back to the legacy heuristic
    instead of crashing or returning the bad name."""
    from engine.research.lens_helpers import resolve_default_net_col
    df = pd.DataFrame({
        "pnl_gross":    [0.01] * 5,
        "pnl_net_13bp": [0.008] * 5,
    })
    bad = {
        "pnl_series_df":   df,
        "pnl_default_col": "pnl_net_TYPO_99bp",   # doesn't exist
    }
    # Should NOT return the bad name; should fall back to legacy
    assert resolve_default_net_col(bad) == "pnl_net_13bp"


# ────────────────────────────────────────────────────────────────────
# resolve_gross_col mirrors net resolution
# ────────────────────────────────────────────────────────────────────
def test_resolve_gross_col_default(equity_artifacts):
    from engine.research.lens_helpers import resolve_gross_col
    assert resolve_gross_col(equity_artifacts) == "pnl_gross"


def test_resolve_gross_col_handles_legacy(legacy_artifacts):
    from engine.research.lens_helpers import resolve_gross_col
    # Un-migrated template → universal convention "pnl_gross"
    assert resolve_gross_col(legacy_artifacts) == "pnl_gross"


def test_resolve_gross_col_returns_none_when_absent():
    from engine.research.lens_helpers import resolve_gross_col
    df = pd.DataFrame({"pnl_net_8bp": [0.01] * 5})
    assert resolve_gross_col({"pnl_series_df": df}) is None


# ────────────────────────────────────────────────────────────────────
# slice_pnl_net_and_gross — convenience for lens runners
# ────────────────────────────────────────────────────────────────────
def test_slice_returns_both_series(equity_artifacts):
    from engine.research.lens_helpers import slice_pnl_net_and_gross
    net, gross = slice_pnl_net_and_gross(equity_artifacts)
    assert net is not None
    assert gross is not None
    assert len(net) == 60
    assert len(gross) == 60
    # Net == declared pnl_net_13bp column
    pd.testing.assert_series_equal(
        net.reset_index(drop=True),
        equity_artifacts["pnl_series_df"]["pnl_net_13bp"].reset_index(drop=True),
        check_names=False,
    )


def test_slice_handles_fx_artifacts(fx_carry_artifacts):
    """FX template's pnl_net_8bp resolved correctly — locks the
    very bug B.2 was created to fix."""
    from engine.research.lens_helpers import slice_pnl_net_and_gross
    net, gross = slice_pnl_net_and_gross(fx_carry_artifacts)
    pd.testing.assert_series_equal(
        net.reset_index(drop=True),
        fx_carry_artifacts["pnl_series_df"]["pnl_net_8bp"].reset_index(drop=True),
        check_names=False,
    )


def test_slice_returns_none_pair_for_empty():
    from engine.research.lens_helpers import slice_pnl_net_and_gross
    net, gross = slice_pnl_net_and_gross(None)
    assert net is None
    assert gross is None
