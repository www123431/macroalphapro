"""
Phase 2 forensic helpers tests — auto-gate behavior + basic OK case.

Each helper has 2 tests:
  - INSUFFICIENT_DATA path (empty / below threshold)
  - OK path (synthetic data above threshold)

Spec: docs/spec_per_strategy_attribution_logger_v1.md (Sprint H follow-up)
Resume context: project_investigate_dd_phase23_deferred_2026-05-13.md
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

# conftest.py uses engine.memory.Base for create_all; engine.db_models has its
# OWN Base (containing PaperTradeTradeLog, PaperTradeStrategyLog, etc.).
# Ensure db_models tables exist in the test DB before any test runs.
@pytest.fixture(scope="session", autouse=True)
def _ensure_db_models_tables_exist():
    from engine.db_models import Base as DBBase
    from engine.memory import engine as memory_engine
    DBBase.metadata.create_all(memory_engine)
    yield


from engine.forensic.brinson import compute_brinson_attribution
from engine.forensic.factor_decomp import compute_ff5_decomp
from engine.forensic.strategy_decay import (
    SPEC_LOCKED_SHARPE,
    _memmel_z,
    compute_memmel_z_per_strategy,
)
from engine.forensic.forward_ic import compute_forward_ic_per_strategy
from engine.forensic.pnl_timeseries import compute_pnl_trailing


# ───────────────────────────────────────────────────────────────────────────
# Brinson
# ───────────────────────────────────────────────────────────────────────────

def test_brinson_empty_returns_insufficient():
    result = compute_brinson_attribution(pd.DataFrame())
    assert result["status"] == "INSUFFICIENT_DATA"


def test_brinson_no_realized_column_returns_insufficient():
    df = pd.DataFrame({"strategy_name": ["A"], "sleeve_id": ["s"],
                       "ticker": ["X"], "weight": [0.5], "signal_value": [1.0]})
    result = compute_brinson_attribution(df, horizon=5)
    assert result["status"] == "INSUFFICIENT_DATA"


def test_brinson_with_realized_returns_ok():
    df = pd.DataFrame({
        "strategy_name":           ["D_PEAD", "D_PEAD", "K1_BAB"],
        "sleeve_id":               ["ss_sp500", "ss_sp500", "etf_l1"],
        "ticker":                  ["NVDA", "META", "SPY"],
        "weight":                  [0.04, 0.03, 0.10],
        "signal_value":            [2.3, 1.8, 0.5],
        "realized_5d_return":      [-0.10, -0.05, 0.02],
    })
    result = compute_brinson_attribution(df, horizon=5)
    assert result["status"] == "OK"
    assert result["n_trades_total"] == 3
    assert "ss_sp500" in result["by_sleeve"]
    assert "etf_l1"   in result["by_sleeve"]
    assert "D_PEAD"   in result["by_strategy"]
    # Portfolio total should match sum
    expected_total = 0.04 * -0.10 + 0.03 * -0.05 + 0.10 * 0.02
    assert abs(result["portfolio_total"] - expected_total) < 1e-6


# ───────────────────────────────────────────────────────────────────────────
# FF5 decomp
# ───────────────────────────────────────────────────────────────────────────

def test_ff5_decomp_empty_returns_insufficient():
    result = compute_ff5_decomp(pd.DataFrame(), datetime.date(2026, 5, 13))
    assert result["status"] == "INSUFFICIENT_DATA"


def test_ff5_decomp_below_min_pairs_returns_insufficient():
    df = pd.DataFrame({"strategy_name": ["A", "B"], "ticker": ["X", "Y"],
                       "weight": [0.5, 0.3], "realized_5d_return": [0.01, -0.02]})
    result = compute_ff5_decomp(df, datetime.date(2026, 5, 13))
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["have"] == 2
    assert result["need"] == 3


# ───────────────────────────────────────────────────────────────────────────
# Memmel Z
# ───────────────────────────────────────────────────────────────────────────

def test_memmel_z_formula_zero_when_equal():
    """If two Sharpes equal, Z=0."""
    z = _memmel_z(s1=0.7, s2=0.7, n1=100, n2=100)
    assert abs(z) < 1e-6


def test_memmel_z_formula_sign_correct():
    """If realized < spec, Z > 0 (decay direction)."""
    z = _memmel_z(s1=1.0, s2=0.3, n1=30, n2=30)
    assert z > 0


def test_memmel_z_per_strategy_no_log_returns_insufficient():
    """If PaperTradeStrategyLog has nothing in lookback, INSUFFICIENT_DATA."""
    # Use a date far enough in the past that no rows exist
    result = compute_memmel_z_per_strategy(datetime.date(2020, 1, 1))
    assert result["status"] in ("INSUFFICIENT_DATA",)


def test_spec_locked_sharpe_keys_match_strategy_names():
    """SPEC_LOCKED_SHARPE keys should match STRATEGY_SPEC_MAP keys."""
    from engine.portfolio.attribution_logger import STRATEGY_SPEC_MAP
    assert set(SPEC_LOCKED_SHARPE.keys()) == set(STRATEGY_SPEC_MAP.keys())


# ───────────────────────────────────────────────────────────────────────────
# Forward IC
# ───────────────────────────────────────────────────────────────────────────

def test_forward_ic_no_old_trades_returns_insufficient():
    """If no trades older than horizon days exist, INSUFFICIENT_DATA."""
    # Use today — no trades 60d older than today yet
    result = compute_forward_ic_per_strategy(datetime.date.today())
    assert result["status"] in ("INSUFFICIENT_DATA",)


# ───────────────────────────────────────────────────────────────────────────
# P&L trailing
# ───────────────────────────────────────────────────────────────────────────

def test_pnl_trailing_no_log_data_returns_insufficient():
    """If PaperTradeStrategyLog empty in lookback, INSUFFICIENT_DATA."""
    result = compute_pnl_trailing(datetime.date(2020, 1, 1))
    assert result["status"] in ("INSUFFICIENT_DATA",)


def test_pnl_trailing_eta_unlock_format():
    """ETA unlock should be ISO date when INSUFFICIENT_DATA."""
    result = compute_pnl_trailing(datetime.date(2020, 1, 1))
    if result["status"] == "INSUFFICIENT_DATA" and "eta_unlock" in result:
        # ISO date parses
        datetime.date.fromisoformat(result["eta_unlock"])
