"""tests/test_options_bs_pricer.py — verify Black-Scholes pricer
against Hull "Options, Futures, and Other Derivatives" reference values.

Hull Ch 15 Example 15.6 (9th ed): S=42, K=40, T=0.5, r=0.10, sigma=0.20
  call price = 4.76
  put price  = 0.81

Plus put-call parity sanity checks and IV inversion round-trip.
"""
from __future__ import annotations

import math

import pytest

from engine.research.options.bs_pricer import (
    bs_delta, bs_full, bs_gamma, bs_implied_vol, bs_price, bs_vega,
    strike_from_delta,
)


# ── Hull reference values ──────────────────────────────────────────────


class TestHullReference:
    def test_hull_ch15_example_15_6_call(self):
        # Hull 9th ed, Example 15.6: S=42, K=40, T=0.5, r=0.10, sigma=0.20
        # Expected call price 4.76 (Hull rounds to 2 decimals)
        p = bs_price(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20, cp="C")
        assert abs(p - 4.76) < 0.01

    def test_hull_ch15_example_15_6_put(self):
        p = bs_price(S=42, K=40, T=0.5, r=0.10, q=0.0, sigma=0.20, cp="P")
        assert abs(p - 0.81) < 0.01

    def test_put_call_parity(self):
        # Parity: C - P = S*e^(-qT) - K*e^(-rT)
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.25
        c = bs_price(S, K, T, r, q, sigma, "C")
        p = bs_price(S, K, T, r, q, sigma, "P")
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs(lhs - rhs) < 1e-8


# ── Greeks sanity checks ───────────────────────────────────────────────


class TestGreeks:
    def test_call_delta_in_zero_one(self):
        d = bs_delta(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, cp="C")
        assert 0 < d < 1

    def test_put_delta_in_minus_one_zero(self):
        d = bs_delta(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20, cp="P")
        assert -1 < d < 0

    def test_atm_call_delta_close_to_half(self):
        # ATM call with no carry adjustments → delta ≈ 0.5
        d = bs_delta(S=100, K=100, T=1.0, r=0.0, q=0.0, sigma=0.20, cp="C")
        assert abs(d - 0.5) < 0.05

    def test_deep_otm_put_delta_close_to_zero(self):
        # Strike 50% below spot → near-zero put delta
        d = bs_delta(S=100, K=50, T=0.25, r=0.05, q=0.0, sigma=0.20, cp="P")
        assert abs(d) < 0.01

    def test_gamma_positive_and_peaks_atm(self):
        atm = bs_gamma(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
        otm = bs_gamma(S=100, K=80, T=1.0, r=0.05, q=0.0, sigma=0.20)
        itm = bs_gamma(S=100, K=120, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert atm > 0
        assert atm > otm
        assert atm > itm

    def test_vega_positive(self):
        v = bs_vega(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert v > 0

    def test_bs_full_bundle(self):
        g = bs_full(S=100, K=95, T=0.25, r=0.05, q=0.02, sigma=0.30, cp="P")
        assert g.price > 0
        assert -1 < g.delta < 0
        assert g.gamma >= 0
        assert g.vega > 0


# ── IV inversion round-trip ────────────────────────────────────────────


class TestImpliedVolInversion:
    def test_round_trip_atm_call(self):
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.25
        p = bs_price(S, K, T, r, q, sigma, "C")
        iv = bs_implied_vol(S, K, T, r, q, p, "C")
        assert abs(iv - sigma) < 1e-4

    def test_round_trip_otm_put(self):
        S, K, T, r, q, sigma = 100, 90, 0.25, 0.05, 0.02, 0.35
        p = bs_price(S, K, T, r, q, sigma, "P")
        iv = bs_implied_vol(S, K, T, r, q, p, "P")
        assert abs(iv - sigma) < 1e-4

    def test_round_trip_low_vol(self):
        S, K, T, r, q, sigma = 100, 100, 0.5, 0.03, 0.0, 0.05
        p = bs_price(S, K, T, r, q, sigma, "C")
        iv = bs_implied_vol(S, K, T, r, q, p, "C")
        assert abs(iv - sigma) < 1e-4

    def test_price_below_intrinsic_returns_nan(self):
        # Price too low (below intrinsic) → no feasible IV
        iv = bs_implied_vol(S=100, K=90, T=1.0, r=0.05, q=0.0,
                            price=0.001, cp="C")
        assert math.isnan(iv)


# ── strike_from_delta ──────────────────────────────────────────────────


class TestStrikeFromDelta:
    def test_round_trip_put(self):
        S, T, r, q, sigma = 100, 30 / 365, 0.05, 0.02, 0.25
        target_delta = -0.20
        K = strike_from_delta(S, T, r, q, sigma, target_delta, "P")
        recovered = bs_delta(S, K, T, r, q, sigma, "P")
        assert abs(recovered - target_delta) < 1e-6

    def test_round_trip_call(self):
        S, T, r, q, sigma = 100, 60 / 365, 0.05, 0.0, 0.30
        target_delta = 0.30
        K = strike_from_delta(S, T, r, q, sigma, target_delta, "C")
        recovered = bs_delta(S, K, T, r, q, sigma, "C")
        assert abs(recovered - target_delta) < 1e-6

    def test_otm_put_strike_below_spot(self):
        S = 100
        K = strike_from_delta(S, 30 / 365, 0.05, 0.0, 0.20, -0.10, "P")
        assert K < S

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            strike_from_delta(100, 0.25, 0.05, 0.0, 0.25, -0.20, "C")  # call w/ neg delta
        with pytest.raises(ValueError):
            strike_from_delta(100, 0.25, 0.05, 0.0, 0.25, 1.5, "C")    # > 1
