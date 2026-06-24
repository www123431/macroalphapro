"""
tests/test_portfolio_ensemble_dispatch.py — Sprint Week 4 (spec id=50 §2.4).

Pre-registration: docs/spec_factor_ensemble_v1.md §2.4 amendment 2026-05-09 (Issue #1).

Verifies:
  • PRODUCTION_SIGNAL=ensemble_v1: ensemble_v1 column injected from
    compute_ensemble_signal call; downstream df["tsmom"] aliases to ensemble_v1.
  • Empty / all-zero ensemble → fallback to ql01_bab (NOT tsmom).
  • ensemble compute exception → fallback to ql01_bab with WARN entry.
  • If both ensemble and ql01_bab unavailable → empty portfolio with explicit
    refusal to fall back to tsmom (REJECTED_PRODUCTION_SIGNALS guard).
  • PRODUCTION_SIGNAL=ql01_bab (default): existing behavior unchanged
    (regression check — does NOT touch ensemble path).
"""
from __future__ import annotations

import datetime
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from engine.portfolio import construct_portfolio


def _signal_df(ql01_value=1.0, tsmom_value=1.0, vol=0.10, n=4):
    tickers = [f"TICK{i}" for i in range(n)]
    return pd.DataFrame({
        "ticker":   tickers,
        "tsmom":    [tsmom_value, -tsmom_value, tsmom_value, 0.0],
        "ql01_bab": [ql01_value, -ql01_value, ql01_value, 0.0],
        "ann_vol":  [vol] * n,
    }, index=tickers)


def _patch_universe_asset_classes(monkeypatch):
    monkeypatch.setattr(
        "engine.universe_manager.get_asset_class_map",
        lambda: {f"TICK{i}": "equity_sector" for i in range(10)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ensemble_v1 PRIMARY path — ensemble produces non-zero signal
# ─────────────────────────────────────────────────────────────────────────────

def test_ensemble_v1_primary_path_uses_ensemble_signal(monkeypatch):
    """When PRODUCTION_SIGNAL=ensemble_v1 + compute_ensemble_signal returns
    non-zero values, dispatch should use them (not ql01_bab)."""
    _patch_universe_asset_classes(monkeypatch)
    monkeypatch.setattr("engine.portfolio.PRODUCTION_SIGNAL", "ensemble_v1")

    fake_ensemble = pd.Series(
        [0.5, -0.5, 0.5, 0.0],
        index=["TICK0", "TICK1", "TICK2", "TICK3"],
        dtype=float,
    )
    with mock.patch("engine.factor_ensemble.compute_ensemble_signal", return_value=fake_ensemble) as m:
        result = construct_portfolio(
            signal_df=_signal_df(),
            as_of=datetime.date(2026, 6, 1),
        )
        # Must have called compute_ensemble_signal once
        assert m.called, "compute_ensemble_signal must be invoked when PRODUCTION_SIGNAL=ensemble_v1"

    # Portfolio should be non-empty (ensemble produced signal)
    assert not result.weights.empty, "ensemble dispatch produced empty weights"
    # No fallback warning expected
    assert not any("falling back" in w for w in result.warnings), (
        f"unexpected fallback warning in primary path: {result.warnings}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. ensemble all-zero / all-NaN → fallback to ql01_bab
# ─────────────────────────────────────────────────────────────────────────────

def test_ensemble_v1_all_zero_falls_back_to_ql01_bab(monkeypatch):
    """Spec §2.4 lock: ensemble all-zero → fall back to ql01_bab, NEVER tsmom."""
    _patch_universe_asset_classes(monkeypatch)
    monkeypatch.setattr("engine.portfolio.PRODUCTION_SIGNAL", "ensemble_v1")

    zero_ensemble = pd.Series(
        [0.0, 0.0, 0.0, 0.0],
        index=["TICK0", "TICK1", "TICK2", "TICK3"],
        dtype=float,
    )
    with mock.patch("engine.factor_ensemble.compute_ensemble_signal", return_value=zero_ensemble):
        result = construct_portfolio(
            signal_df=_signal_df(ql01_value=1.0),
            as_of=datetime.date(2026, 6, 1),
        )

    # Fallback warning explicitly mentions ql01_bab degraded mode
    fallback_warnings = [w for w in result.warnings if "ensemble_v1 unavailable" in w or "falling back to ql01_bab" in w]
    assert fallback_warnings, f"expected ensemble→ql01_bab fallback warning, got {result.warnings}"
    # CRITICAL — must NOT mention tsmom fallback
    assert not any("falling back to tsmom" in w for w in result.warnings), (
        f"FORBIDDEN: fallback to tsmom (in REJECTED_PRODUCTION_SIGNALS); warnings={result.warnings}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_ensemble_signal raises → degraded fallback to ql01_bab
# ─────────────────────────────────────────────────────────────────────────────

def test_ensemble_v1_compute_exception_falls_back_to_ql01_bab(monkeypatch):
    """If compute_ensemble_signal raises (yfinance failure / etc.), dispatch
    should warn and fall back to ql01_bab (NEVER tsmom)."""
    _patch_universe_asset_classes(monkeypatch)
    monkeypatch.setattr("engine.portfolio.PRODUCTION_SIGNAL", "ensemble_v1")

    with mock.patch(
        "engine.factor_ensemble.compute_ensemble_signal",
        side_effect=RuntimeError("simulated yfinance outage"),
    ):
        result = construct_portfolio(
            signal_df=_signal_df(ql01_value=1.0),
            as_of=datetime.date(2026, 6, 1),
        )

    failure_warnings = [w for w in result.warnings if "ensemble_v1 compute failed" in w]
    assert failure_warnings, f"expected ensemble compute-failed warning, got {result.warnings}"
    # CRITICAL — no tsmom fallback
    assert not any("falling back to tsmom" in w for w in result.warnings), (
        f"FORBIDDEN: fallback to tsmom; warnings={result.warnings}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. ensemble + ql01_bab BOTH unavailable → refuse tsmom fallback, empty portfolio
# ─────────────────────────────────────────────────────────────────────────────

def test_ensemble_v1_no_ql01_bab_refuses_tsmom_fallback(monkeypatch):
    """Final guard: if ensemble fails AND ql01_bab is also empty/missing,
    do NOT fall back to tsmom — return empty portfolio with explicit refusal."""
    _patch_universe_asset_classes(monkeypatch)
    monkeypatch.setattr("engine.portfolio.PRODUCTION_SIGNAL", "ensemble_v1")

    # ql01_bab all-zero — fallback tier 1 fails
    df_bad = _signal_df(ql01_value=0.0)

    with mock.patch(
        "engine.factor_ensemble.compute_ensemble_signal",
        side_effect=RuntimeError("simulated ensemble failure"),
    ):
        result = construct_portfolio(signal_df=df_bad, as_of=datetime.date(2026, 6, 1))

    # Must be empty + explicit refusal warning
    assert result.weights.empty, "empty portfolio expected when both ensemble and ql01_bab unavailable"
    refusal = [w for w in result.warnings if "refusing to fall back to tsmom" in w]
    assert refusal, f"expected explicit tsmom-refusal warning, got {result.warnings}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRODUCTION_SIGNAL=ql01_bab (default) — regression: ensemble path NOT invoked
# ─────────────────────────────────────────────────────────────────────────────

def test_ql01_bab_default_does_not_invoke_ensemble(monkeypatch):
    """Regression: default PRODUCTION_SIGNAL=ql01_bab path must NOT call
    compute_ensemble_signal (existing production unchanged)."""
    monkeypatch.setattr("engine.portfolio.PRODUCTION_SIGNAL", "ql01_bab")

    with mock.patch("engine.factor_ensemble.compute_ensemble_signal") as m:
        construct_portfolio(
            signal_df=_signal_df(ql01_value=1.0),
            as_of=datetime.date(2026, 6, 1),
        )
    assert not m.called, "compute_ensemble_signal must NOT be called when PRODUCTION_SIGNAL=ql01_bab"
