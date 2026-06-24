"""
engine/factors_singlename — Single-stock factor modules for Stage 2.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2

Wave A (yfinance daily prices, no vintage fundamentals):
  - tsmom.py            — Moskowitz-Ooi-Pedersen 2012 single-stock TSMOM
  - bab.py              — Frazzini-Pedersen 2014 single-stock BAB
  - dividend_yield.py   — KMPV 2018 simplified (placeholder for Wave B Value)

Wave B (WRDS Compustat vintage fundamentals, post-approval):
  - value_pe.py         — Fama-French 1993 E/P value factor
  - quality_4comp.py    — AFP 2019 4-component Quality
  - fundamentals_cache.py — vintage fundamentals cache

All factor signal functions follow universal contract:
  compute_<X>_signal(as_of, universe, asset_classes=None, panel=None) -> pd.Series
  - returns Series indexed by ticker
  - cross-section signed signal (e.g. ±1 for tertile-based, or continuous z-score)
  - NaN for tickers without sufficient data (caller MUST handle)
"""
from engine.factors_singlename.tsmom import (
    LOOKBACK_MONTHS_LOCKED,
    SKIP_MONTHS_LOCKED,
    compute_tsmom_singlestock_signal,
)
from engine.factors_singlename.bab import (
    BETA_WINDOW_DAYS_LOCKED,
    BETA_BENCHMARK_LOCKED,
    compute_bab_singlestock_signal,
)
from engine.factors_singlename.dividend_yield import (
    DIVIDEND_LOOKBACK_DAYS_LOCKED,
    compute_dividend_yield_singlestock_signal,
)

__all__ = [
    "LOOKBACK_MONTHS_LOCKED",
    "SKIP_MONTHS_LOCKED",
    "compute_tsmom_singlestock_signal",
    "BETA_WINDOW_DAYS_LOCKED",
    "BETA_BENCHMARK_LOCKED",
    "compute_bab_singlestock_signal",
    "DIVIDEND_LOOKBACK_DAYS_LOCKED",
    "compute_dividend_yield_singlestock_signal",
]
