"""
engine/agents/dq_inspector/thresholds.py — Tier-3-locked DQ thresholds.

Phase 3 of DQ Inspector Agent v1.0 (spec id=70, hash 31b5ad97).

All numeric thresholds for the 10 detectors live in this single
frozen dataclass + 2 dicts. Mutating any value requires a spec
amendment row in spec_metadata.amendment_log + governance log entry.

Companion lockdown test in tests/test_dq_thresholds_locked.py
(Phase 8) diffs runtime values against hardcoded LOCKED_* tables;
silent drift fails CI.

Q resolutions baked in:
  Q2 (§2.1b) — per-series FRED staleness via FRED_MAX_STALENESS_BDAYS
  Q3 (§2.1a) — class-aware Mode 7 anomaly caps via MODE_7_CAP_BY_TICKER_CLASS
  Q4         — row-count regression two-tier (10a moderate / 10b catastrophic)
"""
from __future__ import annotations

import dataclasses


# ──────────────────────────────────────────────────────────────────────────────
# Q2 RESOLUTION — per-series FRED staleness thresholds (business days)
# FRED series have heterogeneous cadence; uniform threshold is wrong.
# Threshold = release-cadence + standard grace window (5 business days).
# Unknown series fall back to global default with SOFT WARN severity.
#
# 2026-05-19 recalibration — catalog now strictly mirrors series actually
# consumed by engine modules. Adding a new FRED series to macro_fetcher /
# regime / history MUST be accompanied by a new entry here (tight coupling
# enforced by tests/test_dq_thresholds_locked.py).
#
# Consumer audit — 2026-05-19 PM senior re-audit found 9 additional
# production consumers missed in the first revision. Test
# `tests/test_dq_fred_catalog_completeness.py` now statically scans
# engine source files for FRED series_id literals and fails CI if
# any aren't in this dict (prevents future regression).
#
# Production-consumer matrix (one row per source module):
#   engine.macro_fetcher._SERIES         : CPIAUCSL CPILFESL PCEPI
#                                          PCEPILFE UNRATE PAYEMS GS10
#                                          T10Y2Y UMCSENT
#   engine.macro_fetcher._YC_TENORS      : DGS1 DGS2 DGS5 GS10 DGS30
#   engine.macro_fetcher._POLICY_SERIES  : FEDFUNDS SOFR
#   engine.macro_fetcher._BREAKEVEN      : T5YIE T10YIE
#   engine.regime._fetch_fred            : DGS10 DGS2 BAMLC0A4CBBB
#   engine.history._fetch_fred_series    : CPIAUCSL FEDFUNDS DGS10 DGS2
#                                          UNRATE
#   engine.fomc_surprise_override        : FEDFUNDS DFEDTARU DFEDTARL
#
# Union (21 unique series). Removed earlier in this revision:
#   TEDRATE (FRED discontinued Jan 2022 — guaranteed false-positive HALT),
#   VIXCLS / DCOILWTICO / DTWEXBGS / DEXUSEU / ICSA / CCSA / INDPRO /
#   HOUST / RSAFS / GDP / GDPC1 / CORPPROFIT (catalog-only; no consumer
#   depends on these). Skipped (FRED returns no data via free API key):
#   NAPM — engine.macro_fetcher:419 calls it but FRED returns empty;
#   adding to catalog would generate chronic false-positive alerts.
#   Future maintainer touching that call should replace or remove the
#   dead reference.
# ──────────────────────────────────────────────────────────────────────────────
#
# Cadence note (FRED semantics):
#   obs_date in FRED responses is the OBSERVATION period (the month/day
#   the value measures), NOT the release date. For a monthly series like
#   CPIAUCSL, the April 2026 value carries obs_date=2026-04-01 even though
#   it is released around 2026-05-13. Staleness measure here is
#   bdays(today − obs_date), so the threshold must cover one full release
#   cycle (~21 bdays/month) plus a grace window (10-15 bdays for the
#   release lag itself).
#
# Series GS10 / FEDFUNDS are MONTHLY in FRED (not daily, despite naming
# in macro_fetcher.py labels). Their daily-cadence counterparts are
# DGS10 / DFF — only the daily IDs get daily thresholds.
FRED_MAX_STALENESS_BDAYS: dict[str, int] = {
    # ── Daily Treasury yields (daily, regime + macro + history) ───────
    "DGS1":       2,    # 1y Treasury (macro_fetcher._YC_TENORS)
    "DGS2":       2,    # 2y Treasury (regime + history + YC)
    "DGS5":       2,    # 5y Treasury (macro_fetcher._YC_TENORS)
    "DGS10":      2,    # 10y Treasury (regime + history + YC)
    "DGS30":      2,    # 30y Treasury (macro_fetcher._YC_TENORS)
    "T10Y2Y":     2,    # 10y-2y spread (macro_fetcher)
    # ── Daily breakeven inflation (macro_fetcher._BREAKEVEN_SERIES) ───
    "T5YIE":      3,    # 5y breakeven inflation — release gaps up to 3d
    "T10YIE":     3,    # 10y breakeven inflation — release gaps up to 3d
    # ── Daily policy rates ────────────────────────────────────────────
    "SOFR":       2,    # secured overnight rate (macro_fetcher._POLICY)
    "DFEDTARU":   3,    # Fed funds target upper (fomc_surprise_override)
    "DFEDTARL":   3,    # Fed funds target lower (fomc_surprise_override)
    # ── Daily credit spread (regime.py crisis detection) ──────────────
    "BAMLC0A4CBBB": 2,  # BBB corporate OAS spread (regime._fetch_fred)
    # ── Monthly Treasury / policy (FRED publishes end-of-month) ───────
    "GS10":      45,    # 10y constant-maturity, monthly avg (macro_fetcher)
    "FEDFUNDS":  45,    # Fed funds effective, monthly (history + fomc)
    # ── Monthly real-economy (mid-month publish for prior month;
    #    ~22 bdays/month + ~15 bdays grace) ──────────────────────────
    "CPIAUCSL":  40,    # CPI all items (macro_fetcher + history)
    "CPILFESL":  40,    # core CPI ex food/energy (macro_fetcher)
    "UNRATE":    40,    # unemployment rate (macro_fetcher + history)
    "PAYEMS":    40,    # nonfarm payrolls (macro_fetcher)
    # ── Monthly with longer publication lag (~end-of-month) ───────────
    "PCEPI":     65,    # PCE price index (macro_fetcher)
    "PCEPILFE":  65,    # core PCE (macro_fetcher)
    # Michigan sentiment — UMCSENT real-world publish lag was 57bd on
    # 2026-05-19 (April final not yet on FRED while March final is
    # latest). Cross-checked via FRED CSV endpoint + API — both agree.
    # Root cause = FRED publication variability, NOT API key / consumer
    # bug. 65bd absorbs one missed release cycle.
    "UMCSENT":   65,
}

FRED_DEFAULT_FALLBACK_BDAYS: int = 7    # unknown series → SOFT WARN at 7bd


# ──────────────────────────────────────────────────────────────────────────────
# Q3 RESOLUTION — class-aware Mode 7 single-day return anomaly caps
# Uniform 30% threshold was wrong because universes have heterogeneous
# empirical volatility distributions (institutional fact, not heuristic).
# ──────────────────────────────────────────────────────────────────────────────
MODE_7_CAP_BY_TICKER_CLASS: dict[str, float] = {
    "etf":           0.30,    # K1 BAB 43 ETFs + AC TLT/GLD + SPY; max ~20% even in 2020-03 COVID
    "single_stock":  0.50,    # D-PEAD top-1500 + Path N S&P 500; post-earnings 25-40% legitimate
    "fund_of_funds": 0.25,    # CTA PQTIX; NAV-based smoothness
    "unknown":       0.30,    # fallback to ETF-conservative
}


# ──────────────────────────────────────────────────────────────────────────────
# Other thresholds — single frozen dataclass per RM pattern
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class DQThresholds:
    """All non-dict numeric thresholds for the 10 detectors.

    Field semantics map 1:1 to spec §2.1 modes 2-10.
    (Mode 1 FRED uses FRED_MAX_STALENESS_BDAYS dict above;
     Mode 7 anomaly uses MODE_7_CAP_BY_TICKER_CLASS dict above.)
    """
    # Mode 2 — yfinance bab_compat cache freshness
    yfinance_bab_cache_max_trading_days:  int = 1     # > 1 trading day stale = HARD HALT

    # Mode 3 — D-PEAD panel cache freshness (calendar days, not business)
    pead_panel_max_calendar_days:          int = 60    # > 60d stale = SOFT WARN

    # Mode 4 — S&P 500 reconstitution feed freshness
    sp500_feed_max_calendar_days:          int = 30    # > 30d no new event = SOFT WARN

    # Mode 5 — K1 ETF universe coverage minimum
    k1_universe_coverage_min:              float = 0.90  # < 90% of 43 ETFs priced = HARD HALT
    k1_universe_expected_n:                int = 43

    # Mode 6 — D-PEAD stock universe coverage minimum
    pead_universe_coverage_min:            float = 0.80  # < 80% of 1500 stocks rdq-cached = HARD HALT
    pead_universe_expected_n:              int = 1500

    # Mode 8 — volume dropoff signal (delisting / corporate action)
    volume_dropoff_ratio:                  float = 0.10  # current < 10% of 60d median = SOFT WARN
    volume_dropoff_lookback_days:          int = 60

    # Mode 9 — NaN burst within active universe
    nan_burst_fraction_max:                float = 0.05  # > 5% NaN = HARD HALT

    # Mode 10a — row-count regression moderate
    row_count_regression_moderate:         float = 0.20  # drop > 20% rel = SOFT WARN

    # Mode 10b — row-count regression catastrophic
    row_count_regression_catastrophic:     float = 0.50  # drop > 50% rel = HARD HALT


# Singleton import path: `from engine.agents.dq_inspector.thresholds
#                         import DQ_THRESHOLDS, FRED_MAX_STALENESS_BDAYS,
#                                MODE_7_CAP_BY_TICKER_CLASS`
DQ_THRESHOLDS: DQThresholds = DQThresholds()


THRESHOLDS_GOVERNANCE_LOG: list[str] = [
    "2026-05-19 initial lockdown — Phase 3 of DQ Inspector spec id=70 hash 31b5ad97; "
    "values aligned with spec §2.1 + §2.1a + §2.1b (Q2/Q3/Q4 senior-reviewed)",
    "2026-05-19 shadow-phase recalibration — FRED_MAX_STALENESS_BDAYS catalog "
    "reduced from 22→12 series; removed TEDRATE (FRED discontinued Jan 2022) + "
    "11 series with no engine consumer (VIXCLS/DCOILWTICO/DTWEXBGS/DEXUSEU/ICSA/"
    "CCSA/INDPRO/HOUST/RSAFS/GDP/GDPC1/CORPPROFIT); added 3 series consumed by "
    "engine.macro_fetcher but previously missing from DQ (GS10/PCEPILFE/UMCSENT). "
    "Net effect: zero behavior change on series we actually depend on; eliminates "
    "shadow-phase false-positive HALTs on TEDRATE + unused series.",
]
