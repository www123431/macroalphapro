"""tests/test_risk_manager_gates.py — Phase 9 G1 detection recall + unit tests.

Covers G1 verdict gate: every detector function in gates.py must correctly
identify its trigger condition. Each of 12 modes gets:
  - Positive test (breach injected → exactly one Breach returned)
  - Negative test (clean state → no Breach)
  - Edge case test where applicable (boundary values, multi-sleeve etc.)

Also exercises:
  - Breach dataclass schema invariance
  - evaluate_all_modes top-level integration
  - classify_severity + any_hard_halt semantics
"""
from __future__ import annotations

import pytest
import pandas as pd

from engine.agents.risk_manager.gates import (
    Breach,
    evaluate_all_modes,
    classify_severity,
    any_hard_halt,
    gate_mode_1a_book_abs_cap,
    gate_mode_1b_intra_sleeve_cap,
    gate_mode_2_sleeve_drift,
    gate_mode_3_gross_leverage,
    gate_mode_4_net_exposure,
    gate_mode_5_hhi,
    gate_mode_6_var_95,
    gate_mode_6b_var_95_model_integrity,
    gate_mode_7_es_95,
    gate_mode_7b_es_95_model_integrity,
    gate_mode_8_short_side_ratio,
    gate_mode_9_min_ok_strategies,
    gate_mode_10_cross_cancel,
    _compute_hhi_book_level,
    _build_ticker_to_sleeves,
)
from engine.strategies import get_registry
from engine.portfolio.paper_trade_combined import StrategySignal


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def registry():
    return get_registry()


@pytest.fixture
def healthy_signals():
    """5 strategies, all OK, with REALISTIC intra-strategy weights that
    pass Mode 1b's sleeve_class intra caps (15% equity, 5% single-stock,
    50% insurance, 100% CTA).

    Real K1 BAB has ~30 holdings; D-PEAD ~150; PATH_N ~5-20 event-driven.
    Synthesize realistic distributions here rather than 2-3 huge bets.
    """
    k1_tickers = [f"E{i:02d}" for i in range(20)]   # 20 K1 ETFs
    k1_weights = pd.Series({t: 0.05 for t in k1_tickers})  # each = 5%, gross = 1.0

    dpead_tickers = [f"S{i:03d}" for i in range(40)]  # 40 stocks
    dpead_weights = pd.Series({t: 0.025 for t in dpead_tickers})  # each = 2.5%

    pathn_tickers = [f"N{i:02d}" for i in range(25)]  # 25 reconstitution names
    pathn_weights = pd.Series({t: 0.04 for t in pathn_tickers})   # each = 4%

    return [
        StrategySignal(
            strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
            weights=k1_weights,
            n_positions=len(k1_tickers), status="OK",
        ),
        StrategySignal(
            strategy_name="D_PEAD", sleeve_id="ss_sp500", intra_sleeve_weight=0.5,
            weights=dpead_weights,
            n_positions=len(dpead_tickers), status="OK",
        ),
        StrategySignal(
            strategy_name="PATH_N", sleeve_id="ss_sp500", intra_sleeve_weight=0.5,
            weights=pathn_weights,
            n_positions=len(pathn_tickers), status="OK",
        ),
        StrategySignal(
            strategy_name="CTA_PQTIX", sleeve_id="cta_defensive", intra_sleeve_weight=1.0,
            weights=pd.Series({"PQTIX": 1.0}),
            n_positions=1, status="OK",
        ),
        StrategySignal(
            strategy_name="AC_TLT_GLD", sleeve_id="rms_crisis_hedge", intra_sleeve_weight=1.0,
            weights=pd.Series({"TLT": 0.5, "GLD": 0.5}),
            n_positions=2, status="OK",
        ),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1a — book-level absolute single-ticker cap (operational risk)
# ──────────────────────────────────────────────────────────────────────────────
class TestMode1aBookAbsCap:
    """Mode 1a checks uniform 25% absolute book cap — protects against
    issuer/ETF blowups regardless of which sleeve(s) the ticker belongs to."""

    def test_within_cap_no_breach(self):
        # Mixed weights all below 25% absolute
        combined = pd.Series({"AAPL": 0.20, "GLD": 0.10, "SPY": -0.15})
        assert gate_mode_1a_book_abs_cap(combined) == []

    def test_breach_above_cap(self):
        combined = pd.Series({"AAPL": 0.30})  # 30% > 25% cap
        breaches = gate_mode_1a_book_abs_cap(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "1a"
        assert breaches[0].severity == "HARD_HALT"
        assert breaches[0].affected == ("AAPL",)
        assert abs(breaches[0].observed_value - 0.30) < 1e-9
        assert abs(breaches[0].threshold - 0.25) < 1e-9
        assert breaches[0].extra["risk_layer"] == "operational"

    def test_short_position_subject_to_cap(self):
        combined = pd.Series({"SPY": -0.27})  # |-0.27| > 0.25 cap
        breaches = gate_mode_1a_book_abs_cap(combined)
        assert len(breaches) == 1
        assert abs(breaches[0].observed_value - 0.27) < 1e-9

    def test_cap_is_sleeve_agnostic(self):
        """1a must NOT need a registry / signals lookup — issuer risk is
        uniform across all sleeves (designed property)."""
        # Function signature accepts ONLY combined; would fail to call with
        # the old (combined, signals, registry) shape.
        combined = pd.Series({"X": 0.10})
        assert gate_mode_1a_book_abs_cap(combined) == []

    def test_at_cap_boundary_no_breach(self):
        combined = pd.Series({"X": 0.25})  # exactly at cap — not over
        assert gate_mode_1a_book_abs_cap(combined) == []


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1b — per-strategy intra-strategy ticker cap (concentration risk)
# ──────────────────────────────────────────────────────────────────────────────
class TestMode1bIntraSleeveCap:
    """Mode 1b checks the strategy's INTRA-strategy ticker weight against
    the sleeve_class cap. Cross-strategy aggregation is Mode 1a's job."""

    def test_healthy_signals_no_breach(self, healthy_signals, registry):
        # AC TLT/GLD has 50% intra on TLT/GLD — within INSURANCE 50% cap.
        # K1 has ~0.3-0.4 intra — within EQUITY 15%? No, 0.4 > 0.15 → fires.
        # Use a less-extreme fixture for "no breach":
        signals = [
            StrategySignal(
                strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
                weights=pd.Series({"SPY": 0.10, "QQQ": -0.08, "IWM": 0.05}),
                n_positions=3, status="OK",
            ),
        ]
        assert gate_mode_1b_intra_sleeve_cap(signals, registry) == []

    def test_equity_intra_breach(self, registry):
        # K1 holding SPY at 20% intra — exceeds ALPHA_EQUITY_LS 15% cap
        signals = [
            StrategySignal(
                strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
                weights=pd.Series({"SPY": 0.20}),
                n_positions=1, status="OK",
            ),
        ]
        breaches = gate_mode_1b_intra_sleeve_cap(signals, registry)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "1b"
        assert breaches[0].severity == "HARD_HALT"
        assert breaches[0].affected == ("SPY",)
        assert breaches[0].extra["strategy"] == "K1_BAB"
        assert breaches[0].extra["sleeve_class"] == "alpha_equity_ls"
        assert breaches[0].extra["risk_layer"] == "strategy_concentration"

    def test_insurance_50_50_allowed(self, registry):
        # AC TLT/GLD at 50/50 — exactly INSURANCE cap = no breach
        signals = [
            StrategySignal(
                strategy_name="AC_TLT_GLD", sleeve_id="rms_crisis_hedge",
                intra_sleeve_weight=1.0,
                weights=pd.Series({"TLT": 0.50, "GLD": 0.50}),
                n_positions=2, status="OK",
            ),
        ]
        assert gate_mode_1b_intra_sleeve_cap(signals, registry) == []

    def test_cta_overlay_full_concentration_allowed(self, registry):
        # CTA single-instrument fund at 100% intra — within CTA_OVERLAY cap
        signals = [
            StrategySignal(
                strategy_name="CTA_PQTIX", sleeve_id="cta_defensive",
                intra_sleeve_weight=1.0,
                weights=pd.Series({"PQTIX": 1.00}),
                n_positions=1, status="OK",
            ),
        ]
        assert gate_mode_1b_intra_sleeve_cap(signals, registry) == []

    def test_non_ok_signal_skipped(self, registry):
        # NO_SIGNAL strategy → no breach even if weights nominally violate
        signals = [
            StrategySignal(
                strategy_name="PATH_N", sleeve_id="ss_sp500",
                intra_sleeve_weight=0.5,
                weights=pd.Series({"NVDA": 0.99}),
                n_positions=1, status="NO_SIGNAL",
            ),
        ]
        assert gate_mode_1b_intra_sleeve_cap(signals, registry) == []

    def test_short_position_subject_to_cap(self, registry):
        signals = [
            StrategySignal(
                strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
                weights=pd.Series({"SPY": -0.25}),
                n_positions=1, status="OK",
            ),
        ]
        breaches = gate_mode_1b_intra_sleeve_cap(signals, registry)
        assert len(breaches) == 1
        assert abs(breaches[0].observed_value - 0.25) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Cross-sleeve overlap regression — the GLD/TLT case that motivated 1a/1b split
# ──────────────────────────────────────────────────────────────────────────────
class TestCrossSleeveOverlapRegression:
    """When the same ticker appears in TWO sleeves (e.g. GLD in K1 BAB
    equity AND AC TLT/GLD insurance), the two-tier design must NOT
    produce the pre-amend false-positive HALT. Mode 1a checks book
    aggregate (25% cap, GLD ~7.5% passes), Mode 1b checks each strategy
    separately (K1's intra GLD ~7.7% passes 15% equity cap; AC's intra
    GLD 50% passes 50% insurance cap)."""

    def test_gld_overlap_passes_both_gates(self, registry):
        # Reproduces 2026-05-19 shadow-phase finding: book GLD 7.5%,
        # K1 intra GLD 7.7%, AC intra GLD 50% — pre-amend HARD_HALT,
        # post-amend clean.
        signals = [
            StrategySignal(
                strategy_name="K1_BAB", sleeve_id="etf_l1", intra_sleeve_weight=1.0,
                weights=pd.Series({"GLD": 0.077, "TLT": 0.077, "SPY": 0.10}),
                n_positions=3, status="OK",
            ),
            StrategySignal(
                strategy_name="AC_TLT_GLD", sleeve_id="rms_crisis_hedge",
                intra_sleeve_weight=1.0,
                weights=pd.Series({"TLT": 0.50, "GLD": 0.50}),
                n_positions=2, status="OK",
            ),
        ]
        # Book-aggregate GLD ≈ 0.077*0.324 + 0.50*0.10 = 0.075
        combined = pd.Series({"GLD": 0.075, "TLT": 0.075, "SPY": 0.0324})
        assert gate_mode_1a_book_abs_cap(combined) == []
        assert gate_mode_1b_intra_sleeve_cap(signals, registry) == []

    def test_gld_overlap_book_stacking_caught_by_1a(self, registry):
        """If a hypothetical re-allocation pushed GLD book weight > 25%
        (each strategy individually compliant), Mode 1a still catches it
        — operational issuer risk is independent of strategy intent."""
        combined = pd.Series({"GLD": 0.30})  # impossible from current
                                              # allocation but tests gate
        breaches = gate_mode_1a_book_abs_cap(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "1a"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2 — relative sleeve drift > 10% of target
# ──────────────────────────────────────────────────────────────────────────────
class TestMode2SleeveDrift:
    def test_at_target_no_breach(self):
        attrib = {"etf_l1": 0.324, "ss_sp500": 0.486}
        target = {"etf_l1": 0.324, "ss_sp500": 0.486}
        assert gate_mode_2_sleeve_drift(attrib, target) == []

    def test_5pct_relative_drift_no_breach(self):
        # 5% relative drift — under 10% threshold
        attrib = {"etf_l1": 0.324 * 1.05}
        target = {"etf_l1": 0.324}
        assert gate_mode_2_sleeve_drift(attrib, target) == []

    def test_15pct_relative_drift_breaches(self):
        # 15% relative drift — over 10% threshold
        attrib = {"etf_l1": 0.324 * 1.15}
        target = {"etf_l1": 0.324}
        breaches = gate_mode_2_sleeve_drift(attrib, target)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "2"
        assert breaches[0].severity == "SOFT_WARN"
        assert breaches[0].affected == ("etf_l1",)

    def test_zero_target_sleeve_skipped(self):
        # target=0 (e.g. real-capital ss_sp500=0 case) — skip drift check
        attrib = {"ss_sp500": 0.10}
        target = {"ss_sp500": 0.0}
        assert gate_mode_2_sleeve_drift(attrib, target) == []

    def test_under_deployed_also_breaches(self):
        # When sleeve effective < target by >10% relative — STILL breaches
        attrib = {"etf_l1": 0.0}                     # 100% under
        target = {"etf_l1": 0.324}
        breaches = gate_mode_2_sleeve_drift(attrib, target)
        assert len(breaches) == 1
        # rel = |0 - 0.324| / 0.324 = 1.0
        assert abs(breaches[0].observed_value - 1.0) < 1e-9

    def test_status_aware_all_no_signal_sleeve_skipped(self):
        """Event-driven day: both strategies in ss_sp500 sleeve report
        NO_SIGNAL; effective collapses to 0 but drift is structural, not
        anomalous. With signals provided, Mode 2 must skip this sleeve."""
        signals = [
            StrategySignal(strategy_name="D_PEAD",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series(dtype=float),
                           n_positions=0, status="NO_SIGNAL"),
            StrategySignal(strategy_name="PATH_N",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series(dtype=float),
                           n_positions=0, status="NO_SIGNAL"),
        ]
        attrib = {"ss_sp500": 0.0}                    # collapsed
        target = {"ss_sp500": 0.486}
        assert gate_mode_2_sleeve_drift(attrib, target, signals) == []

    def test_status_aware_partial_ok_uses_scaled_baseline(self):
        """Only D-PEAD OK today (intra=0.5); expected = 0.486*0.5 = 0.243.
        Effective 0.243 should be at-baseline, no breach."""
        signals = [
            StrategySignal(strategy_name="D_PEAD",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series({"AAPL": 0.243}),
                           n_positions=1, status="OK"),
            StrategySignal(strategy_name="PATH_N",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series(dtype=float),
                           n_positions=0, status="NO_SIGNAL"),
        ]
        attrib = {"ss_sp500": 0.243}
        target = {"ss_sp500": 0.486}
        assert gate_mode_2_sleeve_drift(attrib, target, signals) == []

    def test_status_aware_partial_ok_with_drift_still_fires(self):
        """Only D-PEAD OK (intra=0.5), expected = 0.243, but effective is
        only 0.10 (drift |0.10-0.243|/0.243 = 58%) — must STILL fire."""
        signals = [
            StrategySignal(strategy_name="D_PEAD",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series({"AAPL": 0.10}),
                           n_positions=1, status="OK"),
            StrategySignal(strategy_name="PATH_N",   sleeve_id="ss_sp500",
                           intra_sleeve_weight=0.5,  weights=pd.Series(dtype=float),
                           n_positions=0, status="NO_SIGNAL"),
        ]
        attrib = {"ss_sp500": 0.10}
        target = {"ss_sp500": 0.486}
        breaches = gate_mode_2_sleeve_drift(attrib, target, signals)
        assert len(breaches) == 1
        assert breaches[0].affected == ("ss_sp500",)
        # rel = |0.10 - 0.243| / 0.243 ≈ 0.588
        assert breaches[0].observed_value > 0.5

    def test_status_aware_none_signals_preserves_legacy_behavior(self):
        """When signals=None (legacy caller), behavior must match pre-refinement
        path — full target as baseline, all sleeves checked."""
        attrib = {"etf_l1": 0.0}
        target = {"etf_l1": 0.324}
        # No signals → falls back to legacy behavior → breach fires.
        breaches = gate_mode_2_sleeve_drift(attrib, target, signals=None)
        assert len(breaches) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3 — gross leverage
# ──────────────────────────────────────────────────────────────────────────────
class TestMode3GrossLeverage:
    def test_under_cap_no_breach(self):
        combined = pd.Series({"A": 0.5, "B": -0.5, "C": 0.4})  # gross = 1.4
        assert gate_mode_3_gross_leverage(combined) == []

    def test_at_cap_no_breach(self):
        # gross = 1.60 exactly — at cap, not over
        combined = pd.Series({"A": 1.0, "B": -0.6})
        assert gate_mode_3_gross_leverage(combined) == []

    def test_over_cap_hard_halt(self):
        combined = pd.Series({"A": 0.9, "B": -0.8, "C": 0.7})  # gross = 2.4
        breaches = gate_mode_3_gross_leverage(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "3"
        assert breaches[0].severity == "HARD_HALT"
        assert abs(breaches[0].observed_value - 2.4) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4 — net exposure outside [-0.5, +1.5]
# ──────────────────────────────────────────────────────────────────────────────
class TestMode4NetExposure:
    def test_within_band_no_breach(self):
        combined = pd.Series({"A": 0.8, "B": -0.3})  # net = +0.5
        assert gate_mode_4_net_exposure(combined) == []

    def test_above_max_hard_halt(self):
        combined = pd.Series({"A": 1.6})  # net = +1.6 > 1.5
        breaches = gate_mode_4_net_exposure(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "4"
        assert breaches[0].severity == "HARD_HALT"

    def test_below_min_hard_halt(self):
        combined = pd.Series({"A": -0.6})  # net = -0.6 < -0.5
        breaches = gate_mode_4_net_exposure(combined)
        assert len(breaches) == 1
        assert breaches[0].extra["net_below_min"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Mode 5 — HHI (abs-normalized, Markowitz convention)
# ──────────────────────────────────────────────────────────────────────────────
class TestMode5HHI:
    def test_diversified_no_breach(self):
        # 10 equal-weight tickers → HHI ~ 0.10
        combined = pd.Series({f"T{i}": 0.1 for i in range(10)})
        assert gate_mode_5_hhi(combined) == []

    def test_concentrated_hard_halt(self):
        # One ticker dominates → HHI close to 1
        combined = pd.Series({"BIG": 0.95, "SMALL": 0.01})
        breaches = gate_mode_5_hhi(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "5"
        assert breaches[0].severity == "HARD_HALT"

    def test_abs_normalized_not_naive_squared(self):
        # The senior upgrade: signed (combined**2).sum() vs abs-norm
        ls = pd.Series({"A": 0.9, "B": -0.01})
        hhi = _compute_hhi_book_level(ls)
        # Abs-normalised: gross 0.91, w_norm A=0.989 / B=0.011, HHI = 0.978
        assert abs(hhi - 0.978) < 0.01

    def test_zero_book(self):
        # Empty / all-zero combined → HHI 0
        assert _compute_hhi_book_level(pd.Series({"A": 0.0, "B": 0.0})) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Mode 6 / 6b — VaR-95 soft + model-integrity hard
# ──────────────────────────────────────────────────────────────────────────────
class TestMode6Var95:
    def test_none_input_no_breach(self):
        # VaR unavailable → modes 6/6b no-op
        assert gate_mode_6_var_95(None) == []
        assert gate_mode_6b_var_95_model_integrity(None) == []

    def test_var_above_warn_no_breach(self):
        # VaR-95 = -2% (better than -3% warn floor)
        assert gate_mode_6_var_95(-0.02) == []

    def test_var_below_warn_soft(self):
        # VaR-95 = -4% (worse than -3% warn floor)
        breaches = gate_mode_6_var_95(-0.04)
        assert len(breaches) == 1
        assert breaches[0].severity == "SOFT_WARN"

    def test_var_below_hard_halt(self):
        # VaR-95 = -10% (worse than -9% hard halt)
        breaches = gate_mode_6b_var_95_model_integrity(-0.10)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "6b"
        assert breaches[0].severity == "HARD_HALT"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 7 / 7b — ES-95 soft + hard
# ──────────────────────────────────────────────────────────────────────────────
class TestMode7Es95:
    def test_es_above_warn_no_breach(self):
        assert gate_mode_7_es_95(-0.04) == []

    def test_es_below_warn_soft(self):
        breaches = gate_mode_7_es_95(-0.06)
        assert len(breaches) == 1
        assert breaches[0].severity == "SOFT_WARN"

    def test_es_below_hard_halt(self):
        breaches = gate_mode_7b_es_95_model_integrity(-0.20)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "7b"
        assert breaches[0].severity == "HARD_HALT"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 8 — short-side aggregate > 50% of gross
# ──────────────────────────────────────────────────────────────────────────────
class TestMode8ShortSide:
    def test_balanced_no_breach(self):
        # 40% short / 60% long → 40% short ratio
        combined = pd.Series({"A": 0.6, "B": -0.4})
        assert gate_mode_8_short_side_ratio(combined) == []

    def test_short_heavy_breach(self):
        # 60% short / 40% long → 60% short ratio
        combined = pd.Series({"A": 0.4, "B": -0.6})
        breaches = gate_mode_8_short_side_ratio(combined)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "8"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 9 — minimum OK strategies (cb-cascade)
# ──────────────────────────────────────────────────────────────────────────────
class TestMode9MinOk:
    def _make_sigs(self, statuses):
        return [
            StrategySignal(
                strategy_name=f"S{i}", sleeve_id="etf_l1",
                intra_sleeve_weight=1.0, weights=pd.Series(),
                n_positions=0, status=st,
            )
            for i, st in enumerate(statuses)
        ]

    def test_all_ok_no_breach(self, registry):
        sigs = self._make_sigs(["OK"] * 5)
        assert gate_mode_9_min_ok_strategies(sigs, registry) == []

    def test_three_ok_no_breach(self, registry):
        sigs = self._make_sigs(["OK", "OK", "OK", "NO_SIGNAL", "ERROR"])
        assert gate_mode_9_min_ok_strategies(sigs, registry) == []

    def test_two_ok_hard_halt(self, registry):
        sigs = self._make_sigs(["OK", "OK", "NO_SIGNAL", "NO_SIGNAL", "ERROR"])
        breaches = gate_mode_9_min_ok_strategies(sigs, registry)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "9"
        assert breaches[0].severity == "HARD_HALT"


# ──────────────────────────────────────────────────────────────────────────────
# Mode 10 — cross-cancel ticker count
# ──────────────────────────────────────────────────────────────────────────────
class TestMode10CrossCancel:
    def test_no_overlap_no_breach(self):
        sigs = [
            StrategySignal("A", "ss_sp500", 0.5, pd.Series({"X": 0.5, "Y": 0.5}), 2, "OK"),
            StrategySignal("B", "ss_sp500", 0.5, pd.Series({"Z": 1.0}), 1, "OK"),
        ]
        assert gate_mode_10_cross_cancel(sigs) == []

    def test_few_overlap_no_breach(self):
        # 3 tickers long+short → under 5 cap
        sigs = [
            StrategySignal("A", "ss_sp500", 0.5, pd.Series({f"T{i}": 0.2 for i in range(3)}), 3, "OK"),
            StrategySignal("B", "ss_sp500", 0.5, pd.Series({f"T{i}": -0.2 for i in range(3)}), 3, "OK"),
        ]
        assert gate_mode_10_cross_cancel(sigs) == []

    def test_many_overlap_soft_warn(self):
        # 6 tickers long+short → exceeds 5 cap
        sigs = [
            StrategySignal("A", "ss_sp500", 0.5, pd.Series({f"T{i}": 0.1 for i in range(8)}), 8, "OK"),
            StrategySignal("B", "ss_sp500", 0.5, pd.Series({f"T{i}": -0.1 for i in range(6)}), 6, "OK"),
        ]
        breaches = gate_mode_10_cross_cancel(sigs)
        assert len(breaches) == 1
        assert breaches[0].mode_id == "10"
        assert breaches[0].severity == "SOFT_WARN"


# ──────────────────────────────────────────────────────────────────────────────
# evaluate_all_modes + classify_severity + any_hard_halt
# ──────────────────────────────────────────────────────────────────────────────
class TestTopLevelIntegration:
    def test_evaluate_all_modes_clean(self, healthy_signals, registry):
        """Diversified book mimicking real production (~150 tickers) — no HARD_HALT.

        Real paper_trade run on 2026-05-18 had 153 tickers with HHI ~0.07.
        Synthetic 5-ticker is unrealistically concentrated (HHI 0.27); must
        synthesise enough small positions to mirror production diversification.
        """
        # 100 small equity positions + concentrated AC/CTA holdings (mirror real book)
        small = {f"E{i:03d}": 0.002 for i in range(100)}     # 100 × 0.2% = 20% gross
        big = {"TLT": 0.075, "GLD": 0.075, "PQTIX": 0.135}    # 28.5% gross
        combined = pd.Series({**small, **big})
        target = registry.sleeve_allocation_dict()
        attrib = {sid: target[sid] for sid in target}        # at-target sleeves
        breaches = evaluate_all_modes(
            combined           = combined,
            signals            = healthy_signals,
            sleeve_attribution = attrib,
            sleeve_target      = target,
            registry           = registry,
        )
        # Diversified production-like book — no HARD_HALT expected
        assert not any_hard_halt(breaches), (
            f"unexpected HARD_HALT on diversified book: "
            f"{[(b.mode_id, b.rule_description[:60]) for b in breaches if b.severity == 'HARD_HALT']}"
        )

    def test_classify_severity_priority(self):
        b_hh = Breach("1", "HARD_HALT", "", 0, 0, (), {}, "s")
        b_sw = Breach("2", "SOFT_WARN", "", 0, 0, (), {}, "s")
        assert classify_severity([b_hh]) == "SEVERE"
        assert classify_severity([b_sw, b_sw]) == "MEDIUM"
        assert classify_severity([b_sw]) == "LIGHT"
        assert classify_severity([]) == "NONE"
        # HARD_HALT dominates regardless of warn count
        assert classify_severity([b_hh, b_sw, b_sw]) == "SEVERE"

    def test_any_hard_halt_helper(self):
        assert any_hard_halt([]) is False
        assert any_hard_halt([Breach("2", "SOFT_WARN", "", 0, 0, (), {}, "s")]) is False
        assert any_hard_halt([Breach("1", "HARD_HALT", "", 0, 0, (), {}, "s")]) is True


# ──────────────────────────────────────────────────────────────────────────────
# Breach dataclass schema invariance — locked 8 fields
# ──────────────────────────────────────────────────────────────────────────────
class TestBreachSchema:
    def test_field_count_locked(self):
        """8-field locked schema must not gain/lose fields silently."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(Breach)}
        expected = {
            "mode_id", "severity", "rule_description",
            "observed_value", "threshold", "affected",
            "extra", "spec_anchor",
        }
        assert fields == expected, f"Breach schema drifted: {fields} != {expected}"

    def test_frozen(self):
        b = Breach("1", "HARD_HALT", "", 0, 0, (), {}, "s")
        with pytest.raises(Exception):
            b.mode_id = "2"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# _build_ticker_to_sleeves multi-sleeve safety (Upgrade #1)
# ──────────────────────────────────────────────────────────────────────────────
class TestTickerSleeveMapping:
    def test_returns_set_not_str(self):
        sigs = [
            StrategySignal("X", "ss_sp500", 0.5, pd.Series({"AAPL": 1.0}), 1, "OK"),
        ]
        m = _build_ticker_to_sleeves(sigs)
        assert m == {"AAPL": {"ss_sp500"}}

    def test_multi_sleeve_ticker_aggregates(self):
        sigs = [
            StrategySignal("X", "ss_sp500", 0.5, pd.Series({"TLT": 0.5}), 1, "OK"),
            StrategySignal("Y", "rms_crisis_hedge", 1.0, pd.Series({"TLT": 0.5}), 1, "OK"),
        ]
        m = _build_ticker_to_sleeves(sigs)
        assert m["TLT"] == {"ss_sp500", "rms_crisis_hedge"}

    def test_non_ok_signal_excluded(self):
        sigs = [
            StrategySignal("X", "ss_sp500", 0.5, pd.Series({"AAPL": 1.0}), 1, "ERROR"),
        ]
        m = _build_ticker_to_sleeves(sigs)
        assert m == {}
