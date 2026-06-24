"""
tests/test_paper_trade_combined.py — Sprint A orchestrator smoke tests.

Sprint A bar: verify the 4-component orchestrator produces well-formed output
with deterministic interface contracts. Network-dependent K1 BAB signal call
is skipped in CI (manual integration via main() CLI).
"""
from __future__ import annotations

import datetime

import pandas as pd
import pytest


def test_orchestrator_module_imports():
    from engine.portfolio.paper_trade_combined import (
        PAPER_TRADE_SLEEVE_ALLOCATION,
        INTRA_SS_SP500_WEIGHTS,
        LEVERAGE_FACTOR,
        STRATEGY_DISPLAY_META,
        STRATEGY_ORDER,
        StrategySignal,
        PaperTradeRunResult,
        get_k1_bab_signal,
        get_d_pead_signal,
        get_path_n_signal,
        get_cta_pqtix_signal,
        run_paper_trade_day,
    )
    # 4-sleeve composition post-Path-AC Tier-3-amendment 2026-05-15.
    assert PAPER_TRADE_SLEEVE_ALLOCATION == {
        "etf_l1":           0.324,
        "ss_sp500":         0.486,
        "cta_defensive":    0.090,
        "rms_crisis_hedge": 0.100,
    }
    assert abs(sum(PAPER_TRADE_SLEEVE_ALLOCATION.values()) - 1.0) < 1e-9
    assert INTRA_SS_SP500_WEIGHTS == {"d_pead": 0.50, "path_n": 0.50}
    assert LEVERAGE_FACTOR == 1.5    # Path B Tier-3-amendment 2026-05-15
    # Registry contract: STRATEGY_DISPLAY_META covers 5 strategies in canonical order.
    assert STRATEGY_ORDER == [
        "K1_BAB", "D_PEAD", "PATH_N", "CTA_PQTIX", "AC_TLT_GLD",
    ]
    assert set(STRATEGY_DISPLAY_META) == set(STRATEGY_ORDER)
    for s in STRATEGY_ORDER:
        meta = STRATEGY_DISPLAY_META[s]
        assert meta["sleeve_id"] in PAPER_TRADE_SLEEVE_ALLOCATION
        assert "spec_id" in meta and "spec_hash_short" in meta
        assert "color" in meta and meta["color"].startswith("#")


def test_cta_pqtix_signal_always_returns_pqtix_100pct():
    """CTA SAA is passive long-only at 100% sleeve weight."""
    from engine.portfolio.paper_trade_combined import get_cta_pqtix_signal
    sig = get_cta_pqtix_signal(datetime.date(2025, 6, 30))
    assert sig.strategy_name == "CTA_PQTIX"
    assert sig.sleeve_id == "cta_defensive"
    assert sig.status == "OK"
    assert sig.n_positions == 1
    assert list(sig.weights.index) == ["PQTIX"]
    assert abs(sig.weights["PQTIX"] - 1.0) < 1e-9


def test_d_pead_and_path_n_real_signals_sleeve_contract():
    """Sprint B replaced STUBs with real signals. Verify sleeve/intra-weight
    contract holds (status can be OK / NO_SIGNAL / ERROR depending on as_of
    date and cache availability)."""
    from engine.portfolio.paper_trade_combined import (
        get_d_pead_signal, get_path_n_signal,
    )
    d_pead = get_d_pead_signal(datetime.date(2025, 6, 30))
    path_n = get_path_n_signal(datetime.date(2025, 6, 30))

    # Sleeve/intra-weight contract invariant
    assert d_pead.sleeve_id == "ss_sp500"
    assert d_pead.intra_sleeve_weight == 0.50
    assert d_pead.status in {"OK", "NO_SIGNAL", "ERROR"}

    assert path_n.sleeve_id == "ss_sp500"
    assert path_n.intra_sleeve_weight == 0.50
    assert path_n.status in {"OK", "NO_SIGNAL", "ERROR"}


def test_orchestrator_with_synthetic_signals():
    """Replace all 4 adapters with synthetic StrategySignal; verify the
    combine + attribution math is correct without network dependencies."""
    from engine.portfolio.paper_trade_combined import (
        StrategySignal, _combine_intra_sleeve, PAPER_TRADE_SLEEVE_ALLOCATION,
    )
    from engine.portfolio_sleeves import SleeveCapitalConfig, combine_sleeve_weights

    # Synthetic 4 strategies
    k1 = StrategySignal(
        strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
        weights=pd.Series({"SPY": 0.5, "QQQ": -0.3, "TLT": 0.2}),
        n_positions=3, status="OK",
    )
    d_pead = StrategySignal(
        strategy_name="D_PEAD", sleeve_id="ss_sp500", intra_sleeve_weight=0.50,
        weights=pd.Series({"AAPL": 0.6, "MSFT": 0.4}),
        n_positions=2, status="OK",
    )
    path_n = StrategySignal(
        strategy_name="PATH_N", sleeve_id="ss_sp500", intra_sleeve_weight=0.50,
        weights=pd.Series({"NVDA": 1.0}),
        n_positions=1, status="OK",
    )
    cta = StrategySignal(
        strategy_name="CTA_PQTIX", sleeve_id="cta_defensive", intra_sleeve_weight=1.0,
        weights=pd.Series({"PQTIX": 1.0}),
        n_positions=1, status="OK",
    )

    # Intra-sleeve combination for ss_sp500
    ss_combined = _combine_intra_sleeve([d_pead, path_n])
    # D-PEAD: AAPL 0.6×0.5=0.3, MSFT 0.4×0.5=0.2
    # Path N: NVDA 1.0×0.5=0.5
    assert abs(ss_combined["AAPL"] - 0.30) < 1e-9
    assert abs(ss_combined["MSFT"] - 0.20) < 1e-9
    assert abs(ss_combined["NVDA"] - 0.50) < 1e-9

    # Full portfolio combine
    sleeve_weights = {
        "etf_l1":        k1.weights * k1.intra_sleeve_weight,
        "ss_sp500":      ss_combined,
        "cta_defensive": cta.weights * cta.intra_sleeve_weight,
    }
    cfg = SleeveCapitalConfig(allocations=dict(PAPER_TRADE_SLEEVE_ALLOCATION))
    combined = combine_sleeve_weights(sleeve_weights, config=cfg)

    # Expected contributions sourced from the live PAPER_TRADE_SLEEVE_ALLOCATION
    # (registry-backed) — updated 2026-05-18 from hardcoded 0.36/0.54/0.10 that
    # pre-dated the 2026-05-15 AC TLT/GLD addition. Numbers will now track
    # future sleeve-weight changes without test edits.
    w_etf = PAPER_TRADE_SLEEVE_ALLOCATION["etf_l1"]
    w_ss  = PAPER_TRADE_SLEEVE_ALLOCATION["ss_sp500"]
    w_cta = PAPER_TRADE_SLEEVE_ALLOCATION["cta_defensive"]

    assert abs(combined["SPY"]   - (0.5  * w_etf)) < 1e-9
    assert abs(combined["AAPL"]  - (0.30 * w_ss))  < 1e-9
    assert abs(combined["PQTIX"] - (1.0  * w_cta)) < 1e-9
    assert abs(combined["NVDA"]  - (0.50 * w_ss))  < 1e-9


def test_run_paper_trade_day_returns_result_shape():
    """End-to-end orchestrator call returns proper PaperTradeRunResult shape.
    K1 BAB signal may be OK or NO_SIGNAL/ERROR depending on yfinance access;
    we only verify shape, not signal content."""
    from engine.portfolio.paper_trade_combined import (
        run_paper_trade_day, PaperTradeRunResult,
    )
    result = run_paper_trade_day(datetime.date(2025, 6, 30))

    # Source of truth: STRATEGY_ORDER from registry. Updated 2026-05-18 from
    # the hardcoded 4-strat assertion that pre-dated AC_TLT_GLD (added 2026-05-15).
    from engine.portfolio.paper_trade_combined import STRATEGY_ORDER
    assert isinstance(result, PaperTradeRunResult)
    assert result.as_of == datetime.date(2025, 6, 30)
    assert len(result.signals) == len(STRATEGY_ORDER)
    strategy_names = {sig.strategy_name for sig in result.signals}
    assert strategy_names == set(STRATEGY_ORDER)

    # CTA always works (no network deps)
    cta_sig = next(s for s in result.signals if s.strategy_name == "CTA_PQTIX")
    assert cta_sig.status == "OK"
    assert "PQTIX" in cta_sig.weights.index

    # AC TLT/GLD is also always OK (passive 50/50, no network deps)
    ac_sig = next(s for s in result.signals if s.strategy_name == "AC_TLT_GLD")
    assert ac_sig.status == "OK"
    assert set(ac_sig.weights.index) == {"TLT", "GLD"}

    # Sprint B: D-PEAD and Path N now have real signals; status depends on
    # cache availability + events in window (OK / NO_SIGNAL / ERROR all valid)
    d_pead_sig = next(s for s in result.signals if s.strategy_name == "D_PEAD")
    assert d_pead_sig.status in {"OK", "NO_SIGNAL", "ERROR"}

    path_n_sig = next(s for s in result.signals if s.strategy_name == "PATH_N")
    assert path_n_sig.status in {"OK", "NO_SIGNAL", "ERROR"}

    # Sleeve attribution must cover every sleeve registered in PAPER_TRADE_SLEEVE_ALLOCATION.
    from engine.portfolio.paper_trade_combined import (
        PAPER_TRADE_SLEEVE_ALLOCATION, LEVERAGE_FACTOR,
    )
    assert set(result.sleeve_attribution.keys()) == set(PAPER_TRADE_SLEEVE_ALLOCATION.keys())

    # Combined portfolio must include PQTIX at (cta_sleeve_weight × intra_w × leverage).
    # CTA always works (passive overlay, no network deps), so this is deterministic.
    expected_pqtix = PAPER_TRADE_SLEEVE_ALLOCATION["cta_defensive"] * 1.0 * LEVERAGE_FACTOR
    assert "PQTIX" in result.combined_portfolio.index
    assert abs(result.combined_portfolio["PQTIX"] - expected_pqtix) < 1e-9


def test_orchestrator_idempotent_within_same_date():
    """Two runs for same as_of_date should produce identical strategy_name set
    and identical CTA position (stateless guarantee). Other strategy outputs
    may have minor numerical variation if yfinance returns slightly different
    cached data, but the structural contract is preserved."""
    from engine.portfolio.paper_trade_combined import run_paper_trade_day

    as_of = datetime.date(2025, 6, 30)
    r1 = run_paper_trade_day(as_of)
    r2 = run_paper_trade_day(as_of)

    # Same strategy set
    assert {s.strategy_name for s in r1.signals} == {s.strategy_name for s in r2.signals}
    # Same sleeve attribution structure
    assert set(r1.sleeve_attribution.keys()) == set(r2.sleeve_attribution.keys())
    # CTA position deterministic
    assert abs(r1.combined_portfolio.get("PQTIX", 0.0)
               - r2.combined_portfolio.get("PQTIX", 0.0)) < 1e-9
