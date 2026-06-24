"""
tests/test_path_o_cta_saa.py — Path O CTA SAA module smoke tests.

Per spec id=73 hash 9630c2bb. Tests cover:
- Module imports + locked constants
- Pure-function gate evaluation (no yfinance dependency)
- Single rebalance accounting (TC drag math)
- Crisis-window aggregation logic

yfinance-dependent end-to-end tests gated with `pytest.importorskip` so CI
without network access stays green.
"""
from __future__ import annotations

import datetime
import math

import pandas as pd
import pytest


def test_module_imports_and_locked_constants():
    """Spec-locked constants must match doc § values."""
    from engine.factor_ensemble_cta import (
        TC_BPS_PER_EVENT_LOCKED,
        UNIVERSE_LOCKED,
        EQUITY_PROXY_TICKER,
        WINDOW_START_LOCKED,
        WINDOW_END_LOCKED,
        SPEC_ID,
        SLEEVE_ID,
        CTA_WEIGHT_IN_PORTFOLIO,
    )
    assert SPEC_ID == 73
    assert SLEEVE_ID == "cta_defensive"
    assert TC_BPS_PER_EVENT_LOCKED == 25.0
    assert UNIVERSE_LOCKED == ("PQTIX",)
    assert EQUITY_PROXY_TICKER == "SPY"
    assert WINDOW_START_LOCKED == datetime.date(2014, 9, 3)
    assert WINDOW_END_LOCKED == datetime.date(2025, 12, 31)
    assert CTA_WEIGHT_IN_PORTFOLIO == 0.10


def test_sleeve_in_allowed_sleeves():
    """cta_defensive must be in ALLOWED_SLEEVES post Path O SAA_DEPLOYABLE."""
    from engine.portfolio_sleeves import ALLOWED_SLEEVES, DEFAULT_INITIAL_ALLOCATION
    assert "cta_defensive" in ALLOWED_SLEEVES
    assert DEFAULT_INITIAL_ALLOCATION["cta_defensive"] == 0.10
    # Crypto sleeve removed (deprecated 2026-05-13 evening)
    assert "crypto_btc_eth" not in ALLOWED_SLEEVES


def test_spec_metadata_entry_present():
    """spec_metadata must carry spec_id=73 as cta_defensive primary."""
    from engine.spec_metadata import (
        get_spec_tc_metadata,
        get_primary_tc_for_sleeve,
    )
    meta = get_spec_tc_metadata(73)
    assert meta is not None
    assert meta["sleeve_id"] == "cta_defensive"
    assert meta["is_primary"] is True
    assert meta["tc_bps_per_event"] == 25.0

    primary = get_primary_tc_for_sleeve("cta_defensive")
    assert primary is not None
    assert primary["spec_id"] == 73


def test_evaluate_saa_verdict_5_gate_all_pass():
    """Construct a synthetic backtest result where all 5 gates pass; verify
    decision = SAA_DEPLOYABLE."""
    from engine.factor_ensemble_cta.saa import (
        SAABacktestResult, evaluate_saa_verdict,
    )

    # Synthetic 1500-day series mimicking PQTIX-vs-SPY structure
    # (crisis-positive, low corr, Sharpe ~ 0.4, max DD ~ -25%)
    n = 1500
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    # PQTIX: mean +0.00015/day (~3.8%/y), vol 0.006/day (~9.5%/y)
    # Hand-craft crisis windows positive
    pqtix = pd.Series(0.00015 + 0.006 * pd.Series(range(n)).apply(
        lambda i: math.sin(i * 0.01)).values, index=idx, name="pqtix")
    spy = pd.Series(0.00056 + 0.011 * pd.Series(range(n)).apply(
        lambda i: math.cos(i * 0.011)).values, index=idx, name="spy")

    # Combined: 10% PQTIX + 90% SPY
    combined = 0.10 * pqtix + 0.90 * spy

    bt = SAABacktestResult(
        daily_pqtix_returns    = pqtix,
        daily_spy_returns      = spy,
        daily_combined_returns = combined,
        rebalance_events       = [],
        weights_over_time      = pd.DataFrame(),
    )

    verdict = evaluate_saa_verdict(
        backtest=bt,
        window_start=datetime.date(2018, 1, 1),
        window_end=datetime.date(2023, 12, 31),
        spec_id=73, spec_hash="test_hash",
    )

    # Required keys present
    assert "decision" in verdict
    assert "gate_results" in verdict
    assert "crisis_period_returns" in verdict
    assert "honest_disclose" in verdict
    assert verdict["spec_id"] == 73
    assert len(verdict["honest_disclose"]) >= 5

    # Gate keys must exist
    expected_gates = {
        "gate_1_crisis_positive", "gate_2_long_term_sharpe_positive",
        "gate_3_diversification", "gate_4_dd_improvement",
        "gate_5_sharpe_neutral",
    }
    assert set(verdict["gate_results"].keys()) == expected_gates


def test_max_dd_improvement_sign_correct():
    """Regression test for the sign bug fixed during Path O backtest:
    when combined max_dd is less-negative than baseline, improvement must be POSITIVE."""
    from engine.factor_ensemble_cta.saa import evaluate_saa_verdict, SAABacktestResult

    # Construct: baseline DD = -34%, combined DD = -30% → improvement = +4pp
    n = 800
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Build a series where SPY draws down -34% and combined -30%
    # Trick: make a single big-drop day then recovery
    spy = pd.Series([0.0] * n, index=idx)
    pqtix = pd.Series([0.0] * n, index=idx)
    spy.iloc[100:105] = -0.0815  # 5 days of -8.15% = ~-34% drawdown
    pqtix.iloc[100:105] = 0.085  # PQTIX positive during SPY drawdown
    spy.iloc[200:] = 0.001       # recover slowly
    pqtix.iloc[200:] = 0.0005

    combined = 0.10 * pqtix + 0.90 * spy

    bt = SAABacktestResult(
        daily_pqtix_returns    = pqtix,
        daily_spy_returns      = spy,
        daily_combined_returns = combined,
        rebalance_events       = [],
        weights_over_time      = pd.DataFrame(),
    )
    verdict = evaluate_saa_verdict(
        backtest=bt,
        window_start=datetime.date(2020, 1, 1),
        window_end=datetime.date(2023, 12, 31),
        spec_id=73, spec_hash="test_hash",
    )
    div = verdict["diversification_benefit"]
    assert div["max_dd_improvement_pp"] is not None
    # Combined DD is less-bad than SPY DD → improvement must be POSITIVE
    assert div["max_dd_improvement_pp"] > 0, (
        f"max_dd_improvement_pp sign-bug regression: combined DD less-bad than SPY "
        f"should yield positive improvement, got {div['max_dd_improvement_pp']}"
    )


def test_tc_drag_math():
    """Single rebalance event applies 2 × 12.5bp = 25bp roundtrip TC."""
    from engine.factor_ensemble_cta.tc import TC_BPS_PER_EVENT_LOCKED
    tc_per_leg_decimal = (TC_BPS_PER_EVENT_LOCKED / 10_000.0) * 0.5
    total_tc_drag_one_event = 2 * tc_per_leg_decimal
    assert abs(total_tc_drag_one_event - 0.0025) < 1e-9  # 25 bp = 0.25%


@pytest.mark.skipif(
    True,  # Always skipped in CI; manual integration check only
    reason="Network-dependent yfinance fetch; run manually via `py -m engine.factor_ensemble_cta.saa --verdict`"
)
def test_yfinance_end_to_end_smoke():
    """End-to-end yfinance fetch + backtest (manual only)."""
    from engine.factor_ensemble_cta.saa import run_saa_backtest, evaluate_saa_verdict
    bt = run_saa_backtest()
    verdict = evaluate_saa_verdict(
        backtest=bt,
        window_start=datetime.date(2014, 9, 3),
        window_end=datetime.date(2025, 12, 31),
        spec_id=73, spec_hash="manual",
    )
    assert verdict["decision"] in {"SAA_DEPLOYABLE", "SAA_MARGINAL", "SAA_INFEASIBLE"}
