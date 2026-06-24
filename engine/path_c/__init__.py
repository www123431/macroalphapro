"""
engine/path_c/ — Phase 2 Path C event-driven anomaly research.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57)
Strategy memo: project_post_wave_b_strategy_2026-05-12.md

Path C #1: Earnings PEAD (Post-Earnings Announcement Drift)
- Sprint 2: earnings_panel.py (THIS package) — firm-quarter I/B/E/S + Compustat + CRSP join
- Sprint 3: sue_signal.py — SUE = (actual - consensus_median) / dispersion
- Sprint 4: pead_backtest.py — walk-forward decile portfolio formation
- Sprint 5: verdict.py — Sharpe / NW t (lag=60) / BHY-FDR aggregation

Project axis: quant alpha + agentic ops; 0 LLM in alpha loop (deterministic).
"""
from __future__ import annotations

# Locked constants from spec §2.3 (LOCKED at register time, hash ac203e0dc3de)
CONSENSUS_LOCK_WINDOW_DAYS: int = 90   # 90d pre-rdq window for analyst forecasts
MIN_ANALYSTS_REQUIRED:      int = 2    # exclude firm-quarter if fewer than 2 analysts

# Locked from spec §1 + §2.1
WINDOW_START_LOCKED: str = "2014-01-01"
WINDOW_END_LOCKED:   str = "2023-12-31"
UNIVERSE_TOP_N_LOCKED: int = 200       # top-N by market cap (kickoff brief §12)

# Locked from spec §2.4
HOLD_TRADING_DAYS_LOCKED:   int = 60   # post-rdq+1 to rdq+60 trading days
DECILE_LONG_THRESHOLD:      float = 0.90  # top decile (≥ 90th percentile)
DECILE_SHORT_THRESHOLD:     float = 0.10  # bottom decile (≤ 10th percentile)

# Locked from spec §2.5
NW_LAG_TRADING_DAYS_LOCKED: int = 60   # Newey-West lag for daily returns (matches 60-day overlap)

# Locked from spec §2.4 step 9 honest-disclose / §3.5
TC_BPS_ROUNDTRIP_LOCKED:    float = 30.0   # 30bp roundtrip per kickoff brief §8

from engine.path_c.earnings_panel import (
    bulk_fetch_earnings_panel,
    is_wrds_available,
    EarningsPanelResult,
)

__all__ = [
    "CONSENSUS_LOCK_WINDOW_DAYS",
    "MIN_ANALYSTS_REQUIRED",
    "WINDOW_START_LOCKED",
    "WINDOW_END_LOCKED",
    "UNIVERSE_TOP_N_LOCKED",
    "HOLD_TRADING_DAYS_LOCKED",
    "DECILE_LONG_THRESHOLD",
    "DECILE_SHORT_THRESHOLD",
    "NW_LAG_TRADING_DAYS_LOCKED",
    "TC_BPS_ROUNDTRIP_LOCKED",
    "bulk_fetch_earnings_panel",
    "is_wrds_available",
    "EarningsPanelResult",
]
