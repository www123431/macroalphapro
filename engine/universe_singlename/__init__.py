"""
engine/universe_singlename — Single-stock universe management.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52)
Project axis: per project_reframe_quant_alpha_agentic_ops_2026-05-09.md (量化主线)

Wave A (yfinance + best-effort historical):
  - sp500_constituents_wikipedia.py — Wikipedia archive scraper
  - sp500_constituents_github.py    — GitHub fja05680/sp500 dataset
  - sp500_constituents_proxy.py     — current-mktcap-top-500 proxy

Wave B (WRDS CRSP — post-approval):
  - sp500_constituents_crsp.py      — CRSP vintage authoritative

Per Issue #2 (pre-Wave-A audit): Wave A runs all 3 sources for sensitivity check.
If pairwise Sharpe range > ±0.10, results flagged universe-source-sensitive.
"""
from engine.universe_singlename.constituents_loader import (
    UNIVERSE_SOURCES_LOCKED,
    load_sp500_constituents_at_date,
    load_sp500_constituents_panel,
    SP500ConstituentsResult,
)

__all__ = [
    "UNIVERSE_SOURCES_LOCKED",
    "load_sp500_constituents_at_date",
    "load_sp500_constituents_panel",
    "SP500ConstituentsResult",
]
