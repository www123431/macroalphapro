"""engine/research/options/bs_pricer.py — Black-Scholes-Merton
European option pricing + greeks + implied vol inversion.

Reference:
  Black & Scholes (1973) "The Pricing of Options and Corporate
    Liabilities", Journal of Political Economy
  Merton (1973) "Theory of Rational Option Pricing", Bell Journal of
    Economics and Management Science
  Hull, "Options, Futures, and Other Derivatives" 9th ed, Ch 15+18
  Manaster & Koehler (1982) "The Calculation of Implied Variances
    from the Black-Scholes Model: A Note", Journal of Finance — used
    for IV seed value (Brenner-Subrahmanyam 1988 approximation).

Conventions:
  S      spot price
  K      strike
  T      time to expiry in YEARS (e.g. 30/365 = 0.0822 for 30-day)
  r      risk-free rate (continuous compounding, annualized)
  q      dividend yield (continuous, annualized; 0 for non-div equity)
  sigma  annualized vol (decimal, NOT pct — 0.20 not 20)
  cp     "C" or "P"

All inputs in float; tests verify against Hull Table 13.2 +
OptionMetrics reference cases.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import stats as _stats

CP = Literal["C", "P"]


def _check_inputs(S: float, K: float, T: float, sigma: float) -> None:
    if S <= 0:
        raise ValueError(f"spot S must be > 0, got {S}")
    if K <= 0:
        raise ValueError(f"strike K must be > 0, got {K}")
    if T <= 0:
        raise ValueError(f"time T must be > 0, got {T}")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> tuple[float, float]:
    if sigma == 0:
        # Degenerate but well-defined: deterministic terminal value
        return float("inf"), float("inf")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, cp: CP) -> float:
    """Black-Scholes-Merton price for European option with continuous
    dividend yield q."""
    _check_inputs(S, K, T, sigma)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if sigma == 0:
        intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T)) if cp == "C" \
                    else max(0.0, K * math.exp(-r * T) - S * math.exp(-q * T))
        return intrinsic
    if cp == "C":
        return (S * math.exp(-q * T) * _stats.norm.cdf(d1)
                - K * math.exp(-r * T) * _stats.norm.cdf(d2))
    if cp == "P":
        return (K * math.exp(-r * T) * _stats.norm.cdf(-d2)
                - S * math.exp(-q * T) * _stats.norm.cdf(-d1))
    raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")


def bs_delta(S: float, K: float, T: float, r: float, q: float,
             sigma: float, cp: CP) -> float:
    """Delta = dPrice/dS. For calls: e^(-qT) Phi(d1); for puts: -e^(-qT) Phi(-d1)."""
    _check_inputs(S, K, T, sigma)
    if sigma == 0:
        if cp == "C":
            return 1.0 * math.exp(-q * T) if S > K else 0.0
        return -1.0 * math.exp(-q * T) if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if cp == "C":
        return math.exp(-q * T) * _stats.norm.cdf(d1)
    if cp == "P":
        return -math.exp(-q * T) * _stats.norm.cdf(-d1)
    raise ValueError(f"cp must be 'C' or 'P', got {cp!r}")


def bs_gamma(S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
    """Gamma is same for calls and puts."""
    _check_inputs(S, K, T, sigma)
    if sigma == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return (math.exp(-q * T) * _stats.norm.pdf(d1)
            / (S * sigma * math.sqrt(T)))


def bs_vega(S: float, K: float, T: float, r: float, q: float,
            sigma: float) -> float:
    """Vega = dPrice/dSigma per 1.0 vol change (NOT per 1 vol point —
    multiply by 0.01 to get per-pct vega)."""
    _check_inputs(S, K, T, sigma)
    if sigma == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _stats.norm.pdf(d1) * math.sqrt(T)


@dataclass(frozen=True)
class OptionGreeks:
    """Bundle of price + greeks for one (S, K, T, r, q, sigma, cp) call."""
    price: float
    delta: float
    gamma: float
    vega: float


def bs_full(S: float, K: float, T: float, r: float, q: float,
            sigma: float, cp: CP) -> OptionGreeks:
    """Compute price + delta + gamma + vega in one call."""
    return OptionGreeks(
        price=bs_price(S, K, T, r, q, sigma, cp),
        delta=bs_delta(S, K, T, r, q, sigma, cp),
        gamma=bs_gamma(S, K, T, r, q, sigma),
        vega=bs_vega(S, K, T, r, q, sigma),
    )


# ── Implied volatility inversion ───────────────────────────────────────


def _bs_iv_seed(S: float, K: float, T: float, r: float, q: float,
                price: float, cp: CP) -> float:
    """Brenner-Subrahmanyam (1988) at-the-money approximation seed for
    Newton iterations. Works for ATM and near-ATM; otherwise we fall
    back to bisection."""
    forward = S * math.exp((r - q) * T)
    # ATM volatility approximation
    if abs(forward - K) / K < 0.05:
        # Brenner-Subrahmanyam: sigma ≈ price * sqrt(2π/T) / S
        return max(0.01, price * math.sqrt(2 * math.pi / T) / S)
    return 0.30  # generic fallback


def bs_implied_vol(S: float, K: float, T: float, r: float, q: float,
                   price: float, cp: CP,
                   tol: float = 1e-6,
                   max_iter: int = 100) -> float:
    """Invert Black-Scholes for implied vol via Newton-Raphson with
    vega gradient + bisection fallback.

    Returns NaN if the target price is outside the feasible no-arb range.
    """
    _check_inputs(S, K, T, 0.01)
    # No-arb feasibility check
    forward_disc = S * math.exp(-q * T)
    strike_disc = K * math.exp(-r * T)
    if cp == "C":
        intrinsic = max(0.0, forward_disc - strike_disc)
        upper = forward_disc
    else:
        intrinsic = max(0.0, strike_disc - forward_disc)
        upper = strike_disc
    if price < intrinsic - tol or price > upper + tol:
        return float("nan")

    # Newton with vega gradient
    sigma = _bs_iv_seed(S, K, T, r, q, price, cp)
    for _ in range(max_iter):
        p = bs_price(S, K, T, r, q, sigma, cp)
        v = bs_vega(S, K, T, r, q, sigma)
        if v < 1e-10:
            break
        diff = p - price
        if abs(diff) < tol:
            return sigma
        new_sigma = sigma - diff / v
        if new_sigma <= 0:
            new_sigma = sigma / 2
        sigma = new_sigma

    # Bisection fallback if Newton failed to converge
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bs_price(S, K, T, r, q, mid, cp)
        if abs(p - price) < tol:
            return mid
        if p < price:
            lo = mid
        else:
            hi = mid
    return mid


# ── Strike-from-delta helper ───────────────────────────────────────────


def strike_from_delta(S: float, T: float, r: float, q: float,
                      sigma: float, target_delta: float, cp: CP,
                      tol: float = 1e-6, max_iter: int = 100) -> float:
    """Given target delta (signed: positive for calls, negative for puts),
    find the strike K such that bs_delta(S, K, T, r, q, sigma, cp) ==
    target_delta.

    Uses Newton iteration on delta as a function of K. Falls back to
    bisection if Newton diverges (rare for monotone delta surface).
    """
    if cp == "C" and target_delta <= 0 or cp == "C" and target_delta >= 1:
        raise ValueError(f"call delta must be in (0, 1), got {target_delta}")
    if cp == "P" and target_delta >= 0 or cp == "P" and target_delta <= -1:
        raise ValueError(f"put delta must be in (-1, 0), got {target_delta}")

    # Closed-form solve via inverse of delta formula:
    #   call_delta = e^(-qT) * Phi(d1)  →  d1 = Phi^-1(call_delta * e^(qT))
    #   put_delta  = -e^(-qT) * Phi(-d1) →  d1 = -Phi^-1(-put_delta * e^(qT))
    # Then K = S * exp((r-q+0.5*sigma²)*T - d1*sigma*sqrt(T))
    e_qT = math.exp(q * T)
    if cp == "C":
        u = target_delta * e_qT
        if u >= 1 or u <= 0:
            raise ValueError(f"call delta {target_delta} infeasible at q={q}")
        d1 = float(_stats.norm.ppf(u))
    else:
        u = -target_delta * e_qT
        if u >= 1 or u <= 0:
            raise ValueError(f"put delta {target_delta} infeasible at q={q}")
        d1 = -float(_stats.norm.ppf(u))
    K = S * math.exp((r - q + 0.5 * sigma ** 2) * T - d1 * sigma * math.sqrt(T))
    return K
