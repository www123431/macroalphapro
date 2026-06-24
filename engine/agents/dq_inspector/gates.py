"""
engine/agents/dq_inspector/gates.py — 10 deterministic DQ detectors.

Phase 2 of DQ Inspector v1.0 (spec id=70, hash 31b5ad97). Each gate
returns a list of Breach dataclasses; orchestrator aggregates and
decides halt via classify_severity.

3-hook split per spec §2.3:
  PRE-BATCH gate (cheap freshness, no fetched data needed):
    modes 1 / 2 / 3 / 4
  POST-FEED gate (uses refreshed data: coverage + anomaly):
    modes 5 / 6 / 7 / 9
  POST-BATCH gate (uses persisted state: row-count):
    modes 8 / 10a / 10b

DOCTRINE invariants (inherited from RM, per [[feedback-spec-lock-is
-decision-contract]]):
  - Thresholds from thresholds.DQ_THRESHOLDS + dicts
  - No LLM / no network / no DB writes
  - Pure functional — same inputs → same outputs
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import math
from typing import TYPE_CHECKING, Literal, Optional


def _fmt_int(v: float) -> str:
    """Safe int formatter for rule_description f-strings.

    Handles inf / nan from inspector edge cases (missing file → inf,
    API failure → nan). Returns string suitable for human prose.
    """
    if v is None or math.isnan(v):
        return "n/a"
    if math.isinf(v):
        return "∞ (missing)"
    return str(int(v))

if TYPE_CHECKING:
    import pandas as pd

from engine.agents.dq_inspector.source_inspectors import (
    SourceCheckResult,
    check_fred_freshness,
    check_pead_panel_cache,
    check_price_anomaly,
    check_sp500_feed_freshness,
    check_universe_coverage,
    check_yfinance_bab_cache,
)
from engine.agents.dq_inspector.thresholds import (
    DQ_THRESHOLDS,
    FRED_MAX_STALENESS_BDAYS,
)

logger = logging.getLogger(__name__)


SeverityLiteral = Literal["HARD_HALT", "SOFT_WARN"]


# ──────────────────────────────────────────────────────────────────────────────
# Breach schema — mirrors engine.agents.risk_manager.gates.Breach (8 fields)
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class Breach:
    """One DQ detector breach. Schema mirrors RM Breach for downstream parity."""
    mode_id:           str
    severity:          str                      # "HARD_HALT" / "SOFT_WARN"
    rule_description:  str
    observed_value:    float
    threshold:         float
    affected:          tuple[str, ...]
    extra:             dict
    spec_anchor:       str


# ──────────────────────────────────────────────────────────────────────────────
# Helper — convert SourceCheckResult → Breach (preserves all fields)
# ──────────────────────────────────────────────────────────────────────────────
def _result_to_breach(
    r:                 SourceCheckResult,
    mode_id:           str,
    severity:          SeverityLiteral,
    rule_description:  str,
    affected:          tuple[str, ...],
    spec_anchor:       str,
) -> Breach:
    return Breach(
        mode_id          = mode_id,
        severity         = severity,
        rule_description = rule_description,
        observed_value   = float(r.observed_value),
        threshold        = float(r.threshold),
        affected         = affected,
        extra            = {**r.extra, "source_id": r.source_id},
        spec_anchor      = spec_anchor,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1 — FRED series staleness (PRE-BATCH)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_1_fred_staleness(as_of: datetime.date) -> list[Breach]:
    """Iterate FRED_MAX_STALENESS_BDAYS dict; collect breaches.

    Returns at most len(dict) breaches; typical case = 0 breaches.
    Unknown series (those in inspector calls but absent from dict) get
    skipped here — they're handled by the inspector's fallback to
    FRED_DEFAULT_FALLBACK_BDAYS at WARN severity.
    """
    breaches: list[Breach] = []
    for series_id in FRED_MAX_STALENESS_BDAYS:
        r = check_fred_freshness(as_of, series_id)
        if r.is_breach:
            breaches.append(_result_to_breach(
                r,
                mode_id          = "1",
                severity         = "HARD_HALT",
                rule_description = (
                    f"FRED series {series_id!r} last update {r.extra.get('last_obs_date','?')} "
                    f"is {_fmt_int(r.observed_value)} business days stale; max {_fmt_int(r.threshold)} allowed"
                ),
                affected         = (f"fred:{series_id}",),
                spec_anchor      = "spec id=70 §2.1 Mode 1",
            ))
    return breaches


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2 — yfinance bab_compat cache staleness (PRE-BATCH)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_2_bab_cache(as_of: datetime.date) -> list[Breach]:
    r = check_yfinance_bab_cache(as_of)
    if not r.is_breach:
        return []
    return [_result_to_breach(
        r,
        mode_id          = "2",
        severity         = "HARD_HALT",
        rule_description = (
            f"yfinance bab_compat cache stale: mtime {r.extra.get('mtime','?')} "
            f"vs today {as_of} — {_fmt_int(r.observed_value)} trading days "
            f"(max {_fmt_int(r.threshold)}). K1 BAB signal generation depends on this cache."
        ),
        affected         = ("engine.factors.bab_compat",),
        spec_anchor      = "spec id=70 §2.1 Mode 2",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3 — D-PEAD panel cache staleness (PRE-BATCH)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_3_pead_panel(as_of: datetime.date) -> list[Breach]:
    r = check_pead_panel_cache(as_of)
    if not r.is_breach:
        return []
    return [_result_to_breach(
        r,
        mode_id          = "3",
        severity         = "SOFT_WARN",
        rule_description = (
            f"D-PEAD signal panel parquet stale: mtime {r.extra.get('mtime','?')} "
            f"({_fmt_int(r.observed_value)} calendar days; max {_fmt_int(r.threshold)})"
        ),
        affected         = ("engine.path_c.dhs panel parquet",),
        spec_anchor      = "spec id=70 §2.1 Mode 3",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 4 — S&P 500 reconstitution feed staleness (PRE-BATCH)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_4_sp500_feed(as_of: datetime.date) -> list[Breach]:
    r = check_sp500_feed_freshness(as_of)
    if not r.is_breach:
        return []
    return [_result_to_breach(
        r,
        mode_id          = "4",
        severity         = "SOFT_WARN",
        rule_description = (
            f"S&P 500 reconstitution feed stale: last detected_at "
            f"{r.extra.get('last_detected_at','?')} "
            f"({_fmt_int(r.observed_value)} calendar days; max {_fmt_int(r.threshold)})"
        ),
        affected         = ("engine.data_sources.sp500_announcements",),
        spec_anchor      = "spec id=70 §2.1 Mode 4",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 5 — K1 universe coverage (POST-FEED)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_5_k1_coverage(
    n_with_price:  int,
) -> list[Breach]:
    """Caller supplies count of K1 ETFs with today's price. Expected=43."""
    r = check_universe_coverage(
        "k1_universe",
        n_with_data = n_with_price,
        expected_n  = DQ_THRESHOLDS.k1_universe_expected_n,
        min_frac    = DQ_THRESHOLDS.k1_universe_coverage_min,
    )
    if not r.is_breach:
        return []
    return [_result_to_breach(
        r,
        mode_id          = "5",
        severity         = "HARD_HALT",
        rule_description = (
            f"K1 ETF universe coverage {r.observed_value:.1%} below "
            f"{r.threshold:.0%} minimum ({n_with_price}/{DQ_THRESHOLDS.k1_universe_expected_n} ETFs priced)"
        ),
        affected         = ("k1_universe",),
        spec_anchor      = "spec id=70 §2.1 Mode 5",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 6 — D-PEAD universe coverage (POST-FEED)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_6_pead_coverage(
    n_with_rdq:  int,
) -> list[Breach]:
    """Caller supplies count of top-1500 stocks with rdq cached."""
    r = check_universe_coverage(
        "pead_universe",
        n_with_data = n_with_rdq,
        expected_n  = DQ_THRESHOLDS.pead_universe_expected_n,
        min_frac    = DQ_THRESHOLDS.pead_universe_coverage_min,
    )
    if not r.is_breach:
        return []
    return [_result_to_breach(
        r,
        mode_id          = "6",
        severity         = "HARD_HALT",
        rule_description = (
            f"D-PEAD stock universe coverage {r.observed_value:.1%} below "
            f"{r.threshold:.0%} minimum ({n_with_rdq}/{DQ_THRESHOLDS.pead_universe_expected_n} stocks rdq-cached)"
        ),
        affected         = ("pead_universe",),
        spec_anchor      = "spec id=70 §2.1 Mode 6",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 7 — class-aware price tick anomaly (POST-FEED)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_7_price_anomaly(
    daily_returns:      "pd.Series",
    ticker_to_sleeves:  Optional[dict[str, set[str]]] = None,
) -> list[Breach]:
    """Iterate active universe daily returns; class-aware caps per Q3."""
    breaches: list[Breach] = []
    for ticker, ret in daily_returns.items():
        if ret is None or ret != ret:    # NaN guard
            continue
        r = check_price_anomaly(
            str(ticker),
            daily_return       = float(ret),
            ticker_to_sleeves  = ticker_to_sleeves,
        )
        if not r.is_breach:
            continue
        breaches.append(_result_to_breach(
            r,
            mode_id          = "7",
            severity         = "HARD_HALT",
            rule_description = (
                f"Price tick anomaly: {ticker} 1-day return "
                f"{r.extra['signed_return']:+.2%} exceeds "
                f"{r.threshold:.0%} cap for {r.extra['ticker_class']} class"
            ),
            affected         = (str(ticker),),
            spec_anchor      = "spec id=70 §2.1a Mode 7",
        ))
    return breaches


# ──────────────────────────────────────────────────────────────────────────────
# Mode 8 — volume dropoff (POST-BATCH)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_8_volume_dropoff(
    volume_today:           dict[str, float],
    volume_60d_median:      dict[str, float],
) -> list[Breach]:
    """For each ticker, today's volume < ratio × 60d median = SOFT WARN.

    Caller supplies dicts; this function is pure functional.
    """
    ratio_threshold = DQ_THRESHOLDS.volume_dropoff_ratio
    breaches: list[Breach] = []
    for ticker, vol_today in volume_today.items():
        median = volume_60d_median.get(ticker)
        if median is None or median <= 0:
            continue
        if vol_today / median >= ratio_threshold:
            continue
        breaches.append(Breach(
            mode_id          = "8",
            severity         = "SOFT_WARN",
            rule_description = (
                f"Volume dropoff: {ticker} today {vol_today:.0f} is "
                f"{vol_today/median:.1%} of 60d median {median:.0f} "
                f"(threshold {ratio_threshold:.0%}); delisting / corporate-action risk"
            ),
            observed_value   = float(vol_today / median),
            threshold        = float(ratio_threshold),
            affected         = (str(ticker),),
            extra            = {
                "volume_today":   float(vol_today),
                "volume_median":  float(median),
            },
            spec_anchor      = "spec id=70 §2.1 Mode 8",
        ))
    return breaches


# ──────────────────────────────────────────────────────────────────────────────
# Mode 9 — NaN burst within active universe (POST-FEED)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_9_nan_burst(
    n_nan_close:   int,
    n_universe:    int,
) -> list[Breach]:
    """If > N% of active universe has NaN close → HARD HALT."""
    if n_universe <= 0:
        return []
    fraction = n_nan_close / n_universe
    threshold = DQ_THRESHOLDS.nan_burst_fraction_max
    if fraction <= threshold:
        return []
    return [Breach(
        mode_id          = "9",
        severity         = "HARD_HALT",
        rule_description = (
            f"NaN burst: {n_nan_close}/{n_universe} ({fraction:.1%}) "
            f"of active universe has NaN close (max {threshold:.0%})"
        ),
        observed_value   = float(fraction),
        threshold        = float(threshold),
        affected         = (),
        extra            = {
            "n_nan":      n_nan_close,
            "n_universe": n_universe,
        },
        spec_anchor      = "spec id=70 §2.1 Mode 9",
    )]


# ──────────────────────────────────────────────────────────────────────────────
# Mode 10a / 10b — row-count regression (POST-BATCH, two-tier per Q4)
# ──────────────────────────────────────────────────────────────────────────────
def gate_mode_10_row_count_regression(
    today_rows:      int,
    yesterday_rows:  int,
) -> list[Breach]:
    """Two-tier per Q4 resolution:
      Mode 10a: drop > 20% rel → SOFT WARN
      Mode 10b: drop > 50% rel → HARD HALT (escalate to legacy CB)
    """
    if yesterday_rows <= 0:
        return []
    drop_ratio = max(0.0, (yesterday_rows - today_rows) / yesterday_rows)
    th_moderate = DQ_THRESHOLDS.row_count_regression_moderate
    th_catastrophic = DQ_THRESHOLDS.row_count_regression_catastrophic
    if drop_ratio > th_catastrophic:
        return [Breach(
            mode_id          = "10b",
            severity         = "HARD_HALT",
            rule_description = (
                f"Catastrophic row-count drop: {today_rows} today vs {yesterday_rows} "
                f"yesterday ({drop_ratio:.0%} drop, max {th_catastrophic:.0%}); "
                f"daily_batch likely lost data — escalating legacy CB SEVERE"
            ),
            observed_value   = drop_ratio,
            threshold        = th_catastrophic,
            affected         = ("PaperTradeStrategyLog",),
            extra            = {
                "today_rows":      today_rows,
                "yesterday_rows":  yesterday_rows,
                "tier":            "10b_catastrophic",
            },
            spec_anchor      = "spec id=70 §2.1 Mode 10b (Q4)",
        )]
    if drop_ratio > th_moderate:
        return [Breach(
            mode_id          = "10a",
            severity         = "SOFT_WARN",
            rule_description = (
                f"Moderate row-count drop: {today_rows} today vs {yesterday_rows} "
                f"yesterday ({drop_ratio:.0%} drop, max {th_moderate:.0%})"
            ),
            observed_value   = drop_ratio,
            threshold        = th_moderate,
            affected         = ("PaperTradeStrategyLog",),
            extra            = {
                "today_rows":      today_rows,
                "yesterday_rows":  yesterday_rows,
                "tier":            "10a_moderate",
            },
            spec_anchor      = "spec id=70 §2.1 Mode 10a (Q4)",
        )]
    return []


# ──────────────────────────────────────────────────────────────────────────────
# Top-level — 3 phase-specific evaluators + classify_severity (mirror RM)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_pre_batch(as_of: datetime.date) -> list[Breach]:
    """Pre-batch gate: cheap freshness checks (modes 1/2/3/4) ONLY.

    Runs at 06:01 SGT before feed refresh. Uses file mtime + DB MAX
    queries + FRED API (one call per series). No yfinance fetches.
    """
    return (
        gate_mode_1_fred_staleness(as_of)
        + gate_mode_2_bab_cache(as_of)
        + gate_mode_3_pead_panel(as_of)
        + gate_mode_4_sp500_feed(as_of)
    )


def evaluate_post_feed(
    as_of:              datetime.date,
    k1_n_with_price:    int,
    pead_n_with_rdq:    int,
    daily_returns:      Optional["pd.Series"]                = None,
    ticker_to_sleeves:  Optional[dict[str, set[str]]]        = None,
    n_nan_close:        int                                   = 0,
    n_universe:         int                                   = 0,
) -> list[Breach]:
    """Post-feed gate: coverage + anomaly + NaN burst (modes 5/6/7/9).

    Runs at 06:04 SGT after feed refresh. Caller supplies counts +
    daily returns Series; this function is pure on the inputs.
    """
    out: list[Breach] = []
    out += gate_mode_5_k1_coverage(k1_n_with_price)
    out += gate_mode_6_pead_coverage(pead_n_with_rdq)
    if daily_returns is not None and len(daily_returns) > 0:
        out += gate_mode_7_price_anomaly(daily_returns, ticker_to_sleeves)
    out += gate_mode_9_nan_burst(n_nan_close, n_universe)
    return out


def evaluate_post_batch(
    today_rows:           int,
    yesterday_rows:       int,
    volume_today:         Optional[dict[str, float]] = None,
    volume_60d_median:    Optional[dict[str, float]] = None,
) -> list[Breach]:
    """Post-batch gate: row-count + volume dropoff (modes 8/10a/10b).

    Runs at 06:09 SGT after persist. Caller supplies persisted state
    statistics.
    """
    out: list[Breach] = []
    out += gate_mode_10_row_count_regression(today_rows, yesterday_rows)
    if volume_today and volume_60d_median:
        out += gate_mode_8_volume_dropoff(volume_today, volume_60d_median)
    return out


def classify_severity(breaches: list[Breach]) -> str:
    """Same scheme as RM: NONE / LIGHT / MEDIUM / SEVERE."""
    if any(b.severity == "HARD_HALT" for b in breaches):
        return "SEVERE"
    n_warn = sum(1 for b in breaches if b.severity == "SOFT_WARN")
    if n_warn >= 2:
        return "MEDIUM"
    if n_warn == 1:
        return "LIGHT"
    return "NONE"


def any_hard_halt(breaches: list[Breach]) -> bool:
    return any(b.severity == "HARD_HALT" for b in breaches)
