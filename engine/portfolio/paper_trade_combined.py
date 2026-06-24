"""
engine/portfolio/paper_trade_combined.py — Sprint A 4-component paper-trade
orchestrator backbone.

STATELESS DESIGN per senior施工建议 (2026-05-13):
  - No in-memory position state between runs
  - Each invocation reads market data + recomputes signals → outputs positions
  - Idempotent: same as_of_date → same output
  - Restart-safe: any crash leaves no half-state

SCOPE (Sprint A):
  - Demonstrate 4 components can be invoked from one orchestrator
  - Combine via engine.portfolio_sleeves.combine_sleeve_weights
  - Output: daily portfolio weights + per-strategy attribution + run metadata
  - NOT in scope: persistence DB, P&L attribution table, daily backfill loop,
    Task Scheduler integration (Sprint B/C/D)

LAYERED ALLOCATION (paper-trade override, NOT real-capital allocation):
  Updated 2026-05-15 (Tier 3 APPROVED): added Path AC TLT/GLD as insurance sleeve at
  10%; existing 4 sleeves reduced proportionally by 10% per Asness-Israelov 2017
  RMS insurance-budget framework. See
  docs/decisions/saa_path_ac_addition_review_2026-05-15.md.

  - etf_l1            sleeve: 32.4% (K1 BAB = 100% of sleeve)
  - ss_sp500          sleeve: 48.6% (D-PEAD 50% + Path N 50% of sleeve)
  - cta_defensive     sleeve:  9.0% (PQTIX = 100% of sleeve)
  - rms_crisis_hedge  sleeve: 10.0% (Path AC TLT/GLD 50/50 monthly rebalance)
  ----------------------------------------------------------------------
  Total                       100.0%

NOTE: this is PAPER-TRADE intended allocation per deployment_design.md.
DEFAULT_INITIAL_ALLOCATION (real capital, Tier 3 governed) remains
{etf_l1: 0.90, ss_sp500: 0.00, cta_defensive: 0.10}. The paper trade
orchestrator does NOT change real-capital state.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import Optional

import pandas as pd

from engine.portfolio_sleeves import (
    SleeveCapitalConfig,
    combine_sleeve_weights,
)
from engine.portfolio.attribution_logger import (
    TradeAttribution,
    STRATEGY_SPEC_MAP,
)
from engine.strategies import get_registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paper-trade allocation (intended deployment per docs/portfolio_deployment_design_2026-05-13.md)
# Truth source moved to engine/strategies/adapters.py:_populate_registry()
# (Tier 3 governed). This name remains as a back-compat shim that 11+ consumer
# modules import; values are derived from the registry on module import.
# ─────────────────────────────────────────────────────────────────────────────
PAPER_TRADE_SLEEVE_ALLOCATION: dict[str, float] = get_registry().sleeve_allocation_dict()

# Path B leverage Tier 3 amendment (2026-05-15 evening).
# Per docs/decisions/saa_path_b_leverage_2026-05-15.md.
# Modigliani-Miller 1958: leverage preserves Sharpe under RFR borrow assumption.
# 1.5x raises portfolio vol 5.71% -> 8.56% (institutional norm 8-10%).
# Paper-trade simulation only; real-capital deployment requires separate
# broker integration + Tier 3 governance event (DEFAULT_INITIAL_ALLOCATION
# in portfolio_sleeves.py UNCHANGED at 90/0/10).
LEVERAGE_FACTOR: float = 1.5

# Intra-ss_sp500 strategy split (D-PEAD 50% / Path N 50%)
INTRA_SS_SP500_WEIGHTS: dict[str, float] = {
    "d_pead":   0.50,
    "path_n":   0.50,
}


# ─────────────────────────────────────────────────────────────────────────────
# Strategy display registry — back-compat shim sourced from
# engine.strategies.get_registry(). UI pages (positions / portfolio_history /
# executive_brief / system_hub) still import these names; the data now flows
# from each StrategyModule's META class attribute in
# engine/strategies/adapters.py instead of a literal dict here.
#
# Prior bug pattern (2026-05-14 → 2026-05-15): 4 pages hardcoded 4-strategy /
# 3-sleeve weights and missed AC TLT/GLD addition. The registry abstraction
# enforces single-source-of-truth at the class level — adding a new strategy
# requires editing only adapters.py + creating its META.
#
# Per-strategy weight on combined book = PAPER_TRADE_SLEEVE_ALLOCATION[sleeve_id]
# × intra_sleeve_w × LEVERAGE_FACTOR.
# ─────────────────────────────────────────────────────────────────────────────
STRATEGY_DISPLAY_META: dict[str, dict] = get_registry().display_meta_dict()

# Canonical display order (registry preserves insertion order from adapters.py).
STRATEGY_ORDER: list[str] = list(get_registry().names())


def get_strategy_book_weight(strategy_name: str) -> float:
    """Effective weight of a strategy in the combined book (after leverage).

    book_weight = sleeve_allocation × intra_sleeve_w × LEVERAGE_FACTOR
    """
    meta = STRATEGY_DISPLAY_META[strategy_name]
    sleeve_w = PAPER_TRADE_SLEEVE_ALLOCATION[meta["sleeve_id"]]
    return sleeve_w * meta["intra_sleeve_w"] * LEVERAGE_FACTOR


@dataclasses.dataclass(frozen=True)
class StrategySignal:
    """Per-strategy daily output."""
    strategy_name:        str           # 'K1_BAB' / 'D_PEAD' / 'PATH_N' / 'CTA_PQTIX'
    sleeve_id:            str
    intra_sleeve_weight:  float         # share of this strategy within its sleeve
    weights:              pd.Series     # {ticker: weight} normalized to intra-sleeve scale
    n_positions:          int
    status:               str           # 'OK' / 'STUB' / 'ERROR' / 'NO_SIGNAL'
    notes:                str = ""
    # Sprint H v1.0: per-trade forensic context. Default empty tuple for back-compat.
    # Populated by get_*_signal() functions when status == 'OK'.
    trade_attributions:   tuple[TradeAttribution, ...] = ()


@dataclasses.dataclass(frozen=True)
class PaperTradeRunResult:
    """Daily orchestrator output."""
    as_of:                  datetime.date
    signals:                list[StrategySignal]
    combined_portfolio:     pd.Series          # {ticker: portfolio_weight after sleeve allocation}
    sleeve_attribution:     dict[str, float]   # {sleeve_id: total absolute weight}
    run_timestamp_utc:      datetime.datetime
    errors:                 list[str]
    intended_allocation:    dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# Sprint B: rebalance cadence detection (per strategy)
# ─────────────────────────────────────────────────────────────────────────────
# Each strategy has different rebalance cadence. Daily orchestrator must
# distinguish "rebalance day" (compute new weights + apply TC drag) from
# "hold day" (carry yesterday's positions; mark-to-market only). Without this
# distinction, daily orchestrator would over-count TC by 5-20x.

def is_rebalance_day_k1(as_of: datetime.date) -> bool:
    """K1 BAB: last NYSE trading day of month.

    Approximation using calendar (no NYSE holiday adjustment in Sprint B):
    True if next calendar day is in a different month. For exact NYSE
    rebalance day, Sprint D should integrate pandas_market_calendars.
    """
    next_day = as_of + datetime.timedelta(days=1)
    return as_of.month != next_day.month


def is_rebalance_day_d_pead(
    as_of:      datetime.date,
    cache_path: str = "data/path_c_dhs/_pead_ts_signal_panel.parquet",
) -> bool:
    """D-PEAD: any firm in panel cache has rdq == as_of.

    Returns False if cache missing (graceful degradation).
    """
    try:
        from pathlib import Path as _Path
        if not _Path(cache_path).exists():
            return False
        panel = pd.read_parquet(cache_path, columns=["rdq"])
        rdq_dates = pd.to_datetime(panel["rdq"]).dt.date
        return bool((rdq_dates == as_of).any())
    except Exception:
        logger.exception("is_rebalance_day_d_pead failed for %s", as_of)
        return False


def is_rebalance_day_path_n(
    as_of:          datetime.date,
    msp500_events:  Optional[pd.DataFrame] = None,
    lookahead_days: int = 5,
) -> bool:
    """Path N: any pending S&P 500 add with effective_date in (as_of, as_of + 5 NYSE days].

    Sprint B-3 will populate msp500_events from CRSP query. For this is_*
    check we use calendar days as approximation (NYSE holiday adjustment
    deferred to Sprint D).

    Returns False if events DataFrame None / empty.
    """
    if msp500_events is None or msp500_events.empty:
        return False
    cutoff_start = as_of + datetime.timedelta(days=1)
    cutoff_end   = as_of + datetime.timedelta(days=lookahead_days)
    eff_dates = pd.to_datetime(msp500_events["effective_date"]).dt.date
    in_window = (eff_dates >= cutoff_start) & (eff_dates <= cutoff_end)
    return bool(in_window.any())


def is_rebalance_day_cta(
    as_of:                  datetime.date,
    current_pqtix_weight:   Optional[float] = None,
    target_pqtix_weight:    float = 0.10,
    drift_threshold:        float = 0.02,
) -> bool:
    """CTA PQTIX: Dec 31 OR |actual_weight - target| > 2pp drift.

    current_pqtix_weight=None → only check Dec 31 (drift unknown).
    """
    is_year_end = (as_of.month == 12 and as_of.day == 31)
    if is_year_end:
        return True
    if current_pqtix_weight is None:
        return False
    return abs(current_pqtix_weight - target_pqtix_weight) > drift_threshold


def is_rebalance_day(strategy_name: str, as_of: datetime.date, **kwargs) -> bool:
    """Dispatcher for per-strategy rebalance day detection.

    Routes through the StrategyRegistry — each StrategyModule subclass's
    ``is_rebalance_day`` method handles its own kwargs (see adapters.py).

    Preserves the legacy ValueError on unknown strategy_name for back-compat
    with tests/test_paper_trade_combined_sprint_b.py.

    Behavior note: ``"AC_TLT_GLD"`` previously raised ValueError here; in the
    registry-based dispatcher it now returns False (preserving the
    silently-swallowed-to-False behavior at the caller in run_paper_trade_day
    step 5, which wraps the call in try/except). See AcTltGldStrategy
    docstring in adapters.py.
    """
    reg = get_registry()
    if strategy_name not in reg:
        raise ValueError(f"Unknown strategy_name {strategy_name!r}")
    return reg.get(strategy_name).is_rebalance_day(as_of, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy daily signal adapters
# ─────────────────────────────────────────────────────────────────────────────

def get_k1_bab_signal(as_of: datetime.date) -> StrategySignal:
    """K1 BAB ETF signal — Frazzini-Pedersen 2014 on Path K1 45-ETF universe.

    Universe = Tier-1 (33 sector/asset/region) + 10 size/style ETFs per spec id=61.
    Loaded via engine.path_c.k1_universe.get_k1_universe() — single source of truth.

    Returns normalized weights summing to 1.0 in absolute value (long+short).
    """
    try:
        from engine.factors.bab_compat import compute_bab_signal
        from engine.path_c.k1_universe import get_k1_universe

        universe_dict = get_k1_universe()
        universe_tickers = sorted(set(universe_dict.values()))

        sig = compute_bab_signal(
            as_of=as_of, universe=universe_tickers, use_cache=True,
        )
        sig = sig.dropna()
        if sig.empty or sig.abs().sum() == 0:
            return StrategySignal(
                strategy_name       = "K1_BAB",
                sleeve_id           = "etf_l1",
                intra_sleeve_weight = 1.0,
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = "compute_bab_signal returned empty; insufficient history for as_of",
            )
        # Normalize to gross weight 1.0 (long+short summing to 1.0 absolute)
        gross_abs = sig.abs().sum()
        weights = sig / gross_abs

        # Sprint H: build per-ticker attribution (signal_value = -β rank z proxy via BAB sig)
        horizon_k1 = STRATEGY_SPEC_MAP["K1_BAB"][2]   # 30
        event_k1 = as_of.isoformat()                  # monthly rebalance day
        attributions: list[TradeAttribution] = []
        for ticker, w in weights.items():
            if w == 0:
                continue
            attributions.append(TradeAttribution(
                ticker                = str(ticker),
                side                  = "long" if w > 0 else "short",
                weight                = float(w),
                signal_value          = float(sig.get(ticker, 0.0)),   # BAB tertile signal value
                event_trigger         = event_k1,
                expected_horizon_days = horizon_k1,
                notes_json            = f'{{"raw_bab_signal": {float(sig.get(ticker, 0.0)):.6f}}}',
            ))

        return StrategySignal(
            strategy_name       = "K1_BAB",
            sleeve_id           = "etf_l1",
            intra_sleeve_weight = 1.0,
            weights             = weights,
            n_positions         = int((weights != 0).sum()),
            status              = "OK",
            notes               = f"BAB tertile signal over {len(universe_tickers)} K1 ETFs "
                                  f"(Tier-1 + size/style per spec id=61)",
            trade_attributions  = tuple(attributions),
        )
    except Exception as exc:
        logger.exception("get_k1_bab_signal failed for %s: %s", as_of, exc)
        return StrategySignal(
            strategy_name       = "K1_BAB",
            sleeve_id           = "etf_l1",
            intra_sleeve_weight = 1.0,
            weights             = pd.Series(dtype=float),
            n_positions         = 0,
            status              = "ERROR",
            notes               = f"exception: {exc}",
        )


def get_d_pead_signal(
    as_of:           datetime.date,
    cache_path:      Optional[str] = "data/path_c_dhs/_pead_ts_signal_panel.parquet",
    pead_window_days: int          = 90,
    universe_top_n:   int          = 1500,
    long_decile:      float        = 0.10,
    short_leg_weight: float        = 0.0,
) -> StrategySignal:
    """D-PEAD signal — Path D DHS Behavioral 2-factor PEAD-TS leg (spec id=62 c5d9cd09).

    Bernard-Thomas 1989 time-series SUE: post-earnings drift in top SUE decile.
    Reads from cached PEAD-TS panel (data/path_c_dhs/_pead_ts_signal_panel.parquet).

    Algorithm:
      1. Read SUE panel cache (firm-quarter SUE per Compustat fundq B-T 1989 method)
      2. Filter to firms with rdq in (as_of - pead_window_days, as_of]
         (PEAD drift window is ~60 trading days = ~90 calendar days per Path D spec)
      3. Take latest SUE per firm (handle multiple announcements in window)
      4. Filter universe to top-N by market_cap (top-1500 per Path D)
      5. Cross-section rank by SUE; LONG = top decile equal-weight.
      6. If ``short_leg_weight`` > 0: add SHORT = bottom decile, combined = long − w·short
         per spec id=62 Amendment A.1 (w=0.7 is the spec-blessed tilt). Net weight then
         = 1 − w, gross = 1 + w. DEFAULT 0.0 → long-only = the CURRENTLY DEPLOYED behaviour
         (byte-identical; the live call passes no flag). The L/S path is gated off until the
         book gross/net + RM gates are re-calibrated for the added shorts — see audit
         docs/live_delivers_backtest_audit_2026-05-25.md (validated≠deployed reconciliation).

    Honest disclosure:
      - Uses HISTORICAL cache; for live paper-trade beyond 2023-12-28 needs
        cache refresh via engine.path_c.pead_ts_signal_panel.bulk_fetch_*
        with mock_mode=False (real WRDS Compustat pull)
      - Returns NO_SIGNAL if no rdq events in window or cache stale
    """
    try:
        from pathlib import Path as _Path
        cache_full_path = _Path(cache_path) if cache_path else None
        if cache_full_path is None or not cache_full_path.exists():
            return StrategySignal(
                strategy_name       = "D_PEAD",
                sleeve_id           = "ss_sp500",
                intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["d_pead"],
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = f"PEAD-TS cache missing at {cache_path}; "
                                      f"run engine.path_c.pead_ts_signal_panel to refresh",
            )

        panel = pd.read_parquet(cache_full_path)
        # Normalize rdq to date for comparison
        rdq_series = pd.to_datetime(panel["rdq"]).dt.date
        cutoff = as_of - datetime.timedelta(days=pead_window_days)
        active = panel[(rdq_series > cutoff) & (rdq_series <= as_of)].copy()

        if active.empty:
            return StrategySignal(
                strategy_name       = "D_PEAD",
                sleeve_id           = "ss_sp500",
                intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["d_pead"],
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = f"No rdq events in PEAD window ({cutoff} to {as_of}); "
                                      f"cache spans {pd.to_datetime(panel['rdq']).min().date()} to "
                                      f"{pd.to_datetime(panel['rdq']).max().date()}",
            )

        # Latest SUE per firm (deduplicate concurrent rdqs within window)
        active = active.sort_values("rdq")
        latest = active.groupby("ticker", as_index=False).tail(1)

        # Universe filter: top-N by market cap (Path D top-1500 convention)
        latest = latest.dropna(subset=["sue", "market_cap_at_q"])
        if len(latest) > universe_top_n:
            latest = latest.nlargest(universe_top_n, "market_cap_at_q")

        if latest.empty:
            return StrategySignal(
                strategy_name       = "D_PEAD",
                sleeve_id           = "ss_sp500",
                intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["d_pead"],
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = "Post-universe-filter empty",
            )

        # Rank by SUE. LONG = top decile equal-weight. SHORT = bottom decile (tilted by
        # short_leg_weight, spec A.1). short_leg_weight=0.0 → long-only (deployed default).
        latest = latest.sort_values("sue", ascending=False)
        n_top = max(1, int(round(len(latest) * long_decile)))
        top_decile = latest.head(n_top)
        long_w = pd.Series(1.0 / n_top, index=top_decile["ticker"].astype(str).values)

        bot_decile = latest.iloc[0:0]   # empty unless short leg engaged
        if short_leg_weight > 0 and len(latest) >= 2 * n_top:
            bot_decile = latest.tail(n_top)
            short_w = pd.Series(short_leg_weight / n_top,
                                index=bot_decile["ticker"].astype(str).values)
            weights = long_w.subtract(short_w, fill_value=0.0).rename("d_pead_weight")
        else:
            weights = long_w.rename("d_pead_weight")

        # Sprint H: per-trade forensic attribution (both legs)
        horizon_d = STRATEGY_SPEC_MAP["D_PEAD"][2]   # 60
        attributions: list[TradeAttribution] = []
        for _, row in top_decile.iterrows():
            ticker_str = str(row["ticker"]); sue_val = float(row["sue"])
            attributions.append(TradeAttribution(
                ticker                = ticker_str,
                side                  = "long",
                weight                = float(weights.get(ticker_str, 0.0)),
                signal_value          = sue_val,
                event_trigger         = pd.to_datetime(row["rdq"]).date().isoformat(),
                expected_horizon_days = horizon_d,
                notes_json            = f'{{"sue": {sue_val:.6f}, "decile": "top_{int(long_decile*100)}pct"}}',
            ))
        for _, row in bot_decile.iterrows():
            ticker_str = str(row["ticker"]); sue_val = float(row["sue"])
            attributions.append(TradeAttribution(
                ticker                = ticker_str,
                side                  = "short",
                weight                = float(weights.get(ticker_str, 0.0)),
                signal_value          = sue_val,
                event_trigger         = pd.to_datetime(row["rdq"]).date().isoformat(),
                expected_horizon_days = horizon_d,
                notes_json            = f'{{"sue": {sue_val:.6f}, "decile": "bottom_{int(long_decile*100)}pct", "tilt": {short_leg_weight:.2f}}}',
            ))

        _construction = (f"long−{short_leg_weight:g}×short (spec A.1)"
                         if short_leg_weight > 0 else "long-only")
        return StrategySignal(
            strategy_name       = "D_PEAD",
            sleeve_id           = "ss_sp500",
            intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["d_pead"],
            weights             = weights,
            n_positions         = int(len(weights)),
            status              = "OK",
            notes               = f"PEAD-TS {_construction}, top/bottom {long_decile:.0%} of "
                                  f"{len(latest)} firms with rdq in last {pead_window_days}d "
                                  f"(B-T 1989 SUE; top-{universe_top_n} mcap universe)",
            trade_attributions  = tuple(attributions),
        )
    except Exception as exc:
        logger.exception("get_d_pead_signal failed for %s: %s", as_of, exc)
        return StrategySignal(
            strategy_name       = "D_PEAD",
            sleeve_id           = "ss_sp500",
            intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["d_pead"],
            weights             = pd.Series(dtype=float),
            n_positions         = 0,
            status              = "ERROR",
            notes               = f"exception: {exc}",
        )


def get_path_n_signal(
    as_of:           datetime.date,
    events_path:     Optional[str] = "data/path_n/v1_reconstitution_10y_amend1_10bp_event_returns.parquet",
    name_concentration_cap: float = 0.25,
    mode:            str          = "auto",
) -> StrategySignal:
    """Path N reconstitution drift signal — spec id=70 amend 1 hash 60887180.

    Chen-Noronha-Singal 2004 long-only T-5 to T-1 entry on S&P 500 adds.

    Modes:
      'auto'     — try live SP500AnnouncementEvent DB first; if no recent
                   events found, fall back to backtest parquet
      'backtest' — historical replay path; uses CRSP msp500list event parquet
                   (entry_date / exit_date per permno; permno identifiers)
      'live'     — Sprint D-1 forward path; uses SP500AnnouncementEvent DB
                   table populated by data_sources.sp500_announcements
                   (ticker identifiers; T-5 to T-1 window from effective_date)

    Algorithm (live mode):
      1. Query SP500AnnouncementEvent for ADD events with effective_date
         in (as_of, as_of + 5 calendar days]
      2. Equal-weight long-only; cap per-name at 25% concentration
      3. Tickers (not permno) — production-ready

    Algorithm (backtest mode):
      1. Read event parquet (entry_date / exit_date per permno)
      2. Filter to events where as_of is in [entry_date, exit_date]
      3. permno identifiers (ticker mapping deferred to Sprint D-2)
    """
    # Live mode: Sprint D-1 SP500AnnouncementEvent DB
    if mode in ("auto", "live"):
        try:
            from engine.data_sources.sp500_announcements.reconciler import (
                load_pending_path_n_events,
            )
            pending = load_pending_path_n_events(as_of, lookahead_days=5)
            if pending:
                n = len(pending)
                per_name = min(1.0 / n, name_concentration_cap)
                tickers = [str(e["ticker"]).upper() for e in pending]
                weights = pd.Series(
                    [per_name] * n, index=tickers, name="path_n_weight",
                )
                # Sprint H: per-event forensic attribution
                horizon_n = STRATEGY_SPEC_MAP["PATH_N"][2]   # 5
                attributions: list[TradeAttribution] = []
                for ev in pending:
                    eff_d = pd.to_datetime(ev["effective_date"]).date().isoformat()
                    tk = str(ev["ticker"]).upper()
                    attributions.append(TradeAttribution(
                        ticker                = tk,
                        side                  = "long",
                        weight                = float(per_name),
                        signal_value          = 1.0,   # binary event signal
                        event_trigger         = eff_d,
                        expected_horizon_days = horizon_n,
                        notes_json            = f'{{"effective_date": "{eff_d}", "mode": "live"}}',
                    ))
                return StrategySignal(
                    strategy_name       = "PATH_N",
                    sleeve_id           = "ss_sp500",
                    intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
                    weights             = weights,
                    n_positions         = n,
                    status              = "OK",
                    notes               = f"LIVE: S&P 500 add events from Wikipedia+EDGAR feed "
                                          f"(spec D-1 hash TBD); {n} concurrent ADDs "
                                          f"with effective_date in (as_of, +5d]; tickers used",
                    trade_attributions  = tuple(attributions),
                )
            elif mode == "live":
                return StrategySignal(
                    strategy_name       = "PATH_N",
                    sleeve_id           = "ss_sp500",
                    intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
                    weights             = pd.Series(dtype=float),
                    n_positions         = 0,
                    status              = "NO_SIGNAL",
                    notes               = f"LIVE mode: no pending S&P 500 ADD events in 5-day window from {as_of}",
                )
            # else 'auto' mode: fall through to backtest
        except Exception as exc:
            logger.exception("get_path_n_signal live-mode failed: %s", exc)
            if mode == "live":
                return StrategySignal(
                    strategy_name       = "PATH_N",
                    sleeve_id           = "ss_sp500",
                    intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
                    weights             = pd.Series(dtype=float),
                    n_positions         = 0,
                    status              = "ERROR",
                    notes               = f"LIVE mode exception: {exc}",
                )
            # else 'auto': fall through to backtest

    # Backtest mode (legacy Sprint B): CRSP msp500list event parquet
    try:
        from pathlib import Path as _Path
        ev_path = _Path(events_path) if events_path else None
        if ev_path is None or not ev_path.exists():
            return StrategySignal(
                strategy_name       = "PATH_N",
                sleeve_id           = "ss_sp500",
                intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = f"Path N event parquet missing at {events_path}",
            )

        events = pd.read_parquet(ev_path)
        events["entry_date"] = pd.to_datetime(events["entry_date"]).dt.date
        events["exit_date"]  = pd.to_datetime(events["exit_date"]).dt.date

        active = events[(events["entry_date"] <= as_of) & (events["exit_date"] >= as_of)]

        if active.empty:
            return StrategySignal(
                strategy_name       = "PATH_N",
                sleeve_id           = "ss_sp500",
                intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
                weights             = pd.Series(dtype=float),
                n_positions         = 0,
                status              = "NO_SIGNAL",
                notes               = f"No active S&P 500 add events on {as_of}",
            )

        # Equal-weight long-only; cap at name_concentration_cap (25%)
        n = len(active)
        per_name_weight = min(1.0 / n, name_concentration_cap)
        # Use permno as identifier; ticker mapping deferred to Sprint D
        idx = [f"permno_{int(p)}" for p in active["permno"].values]
        weights = pd.Series([per_name_weight] * n, index=idx, name="path_n_weight")

        # Sprint H: per-event forensic attribution (backtest mode)
        horizon_n = STRATEGY_SPEC_MAP["PATH_N"][2]   # 5
        attributions: list[TradeAttribution] = []
        for _, ev in active.iterrows():
            entry_d = pd.to_datetime(ev["entry_date"]).date().isoformat()
            exit_d  = pd.to_datetime(ev["exit_date"]).date().isoformat()
            permno_id = f"permno_{int(ev['permno'])}"
            attributions.append(TradeAttribution(
                ticker                = permno_id,
                side                  = "long",
                weight                = float(per_name_weight),
                signal_value          = 1.0,
                event_trigger         = entry_d,
                expected_horizon_days = horizon_n,
                notes_json            = f'{{"entry_date": "{entry_d}", "exit_date": "{exit_d}", "mode": "backtest"}}',
            ))

        return StrategySignal(
            strategy_name       = "PATH_N",
            sleeve_id           = "ss_sp500",
            intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
            weights             = weights,
            n_positions         = n,
            status              = "OK",
            notes               = f"S&P 500 reconstitution adds active T-5 to T-1 ({n} concurrent events; "
                                  f"permno identifiers; ticker mapping deferred to Sprint D)",
            trade_attributions  = tuple(attributions),
        )
    except Exception as exc:
        logger.exception("get_path_n_signal failed for %s: %s", as_of, exc)
        return StrategySignal(
            strategy_name       = "PATH_N",
            sleeve_id           = "ss_sp500",
            intra_sleeve_weight = INTRA_SS_SP500_WEIGHTS["path_n"],
            weights             = pd.Series(dtype=float),
            n_positions         = 0,
            status              = "ERROR",
            notes               = f"exception: {exc}",
        )


def get_cta_pqtix_signal(as_of: datetime.date) -> StrategySignal:
    """CTA PQTIX SAA signal — Path O spec id=73 hash 9630c2bb.

    PQTIX is a passive 100% sleeve holding. Daily signal = always long PQTIX.
    Rebalance triggers (annual + ±2% drift) live in
    engine.factor_ensemble_cta.saa.run_saa_backtest; for daily paper-trade
    orchestrator the signal is simply "hold PQTIX at full sleeve weight".
    """
    from engine.factor_ensemble_cta import CTA_WEIGHT_IN_PORTFOLIO, SPEC_ID, SLEEVE_ID

    # Sprint H: CTA single-holding attribution
    horizon_cta = STRATEGY_SPEC_MAP["CTA_PQTIX"][2]    # 0 = continuous
    attribution = TradeAttribution(
        ticker                = "PQTIX",
        side                  = "long",
        weight                = 1.0,
        signal_value          = None,                  # no signal — passive overlay
        event_trigger         = "annual_or_2pct_drift_rebal",
        expected_horizon_days = horizon_cta,
        notes_json            = '{"saa_passive": true, "rebalance_rule": "annual+2pct_drift"}',
    )

    return StrategySignal(
        strategy_name       = "CTA_PQTIX",
        sleeve_id           = SLEEVE_ID,                # 'cta_defensive'
        intra_sleeve_weight = 1.0,
        weights             = pd.Series({"PQTIX": 1.0}),
        n_positions         = 1,
        status              = "OK",
        notes               = f"Path O spec_id={SPEC_ID} SAA passive hold (annual+drift rebalance, "
                              f"managed by engine.factor_ensemble_cta.saa)",
        trade_attributions  = (attribution,),
    )


def get_ac_tlt_gld_signal(as_of: datetime.date) -> StrategySignal:
    """Path AC TLT/GLD insurance sleeve signal — spec id=77 hash 4db40176.

    Passive 50/50 TLT + GLD long-only with monthly rebalance to exact target.
    Daily signal = always hold 50/50 (drift between rebalances allowed by spec).
    Per Path AC v3 insurance class verdict 2026-05-15 (PASS 4/4 on extended
    2005-2023 window + 60/40 SPY/AGG baseline).
    """
    horizon_ac = STRATEGY_SPEC_MAP["AC_TLT_GLD"][2]   # 30
    attributions = (
        TradeAttribution(
            ticker                = "TLT",
            side                  = "long",
            weight                = 0.5,
            signal_value          = None,                    # passive sleeve, no signal
            event_trigger         = "monthly_rebalance_to_50_50",
            expected_horizon_days = horizon_ac,
            notes_json            = '{"insurance_sleeve": true, "mechanism": "flight_to_quality"}',
        ),
        TradeAttribution(
            ticker                = "GLD",
            side                  = "long",
            weight                = 0.5,
            signal_value          = None,
            event_trigger         = "monthly_rebalance_to_50_50",
            expected_horizon_days = horizon_ac,
            notes_json            = '{"insurance_sleeve": true, "mechanism": "gold_safe_haven"}',
        ),
    )

    return StrategySignal(
        strategy_name       = "AC_TLT_GLD",
        sleeve_id           = "rms_crisis_hedge",
        intra_sleeve_weight = 1.0,
        weights             = pd.Series({"TLT": 0.5, "GLD": 0.5}),
        n_positions         = 2,
        status              = "OK",
        notes               = "Path AC spec_id=77 v3 insurance class passive hold "
                              "(monthly rebalance to 50/50, drift allowed between)",
        trade_attributions  = attributions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator entry point
# ─────────────────────────────────────────────────────────────────────────────

def _combine_intra_sleeve(
    signals_in_sleeve: list[StrategySignal],
) -> pd.Series:
    """Combine multiple strategies' weights within a single sleeve, weighted by
    each strategy's intra_sleeve_weight."""
    if not signals_in_sleeve:
        return pd.Series(dtype=float)
    contributions: list[pd.Series] = []
    for sig in signals_in_sleeve:
        if sig.weights is None or sig.weights.empty:
            continue
        contributions.append(sig.weights * float(sig.intra_sleeve_weight))
    if not contributions:
        return pd.Series(dtype=float)
    combined = pd.concat(contributions, axis=1).fillna(0.0).sum(axis=1)
    return combined[combined.abs() > 1e-12].astype(float)


def run_paper_trade_day(as_of: datetime.date) -> PaperTradeRunResult:
    """
    Sprint A entry point — execute one stateless daily paper-trade run.

    Algorithm:
      1. Call each strategy's daily signal adapter
      2. Group by sleeve_id; combine intra-sleeve via weighted sum
      3. Apply paper-trade sleeve allocation (36/54/10) via combine_sleeve_weights
      4. Output: per-strategy signals + combined portfolio + sleeve attribution
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")

    run_ts = datetime.datetime.utcnow()
    errors: list[str] = []

    # Step 1: gather all strategy signals via registry iteration.
    # Order matches STRATEGY_ORDER (registry insertion order from adapters.py):
    # K1_BAB → D_PEAD → PATH_N → CTA_PQTIX → AC_TLT_GLD. Each StrategyModule's
    # generate_signal() delegates to the existing get_*_signal() function below.
    signals: list[StrategySignal] = [
        strat.generate_signal(as_of) for strat in get_registry()
    ]
    for sig in signals:
        if sig.status == "ERROR":
            errors.append(f"{sig.strategy_name}: {sig.notes}")
        logger.info(
            "paper_trade_day %s: %s [%s] sleeve=%s intra_w=%.2f n_pos=%d",
            as_of, sig.strategy_name, sig.status, sig.sleeve_id,
            sig.intra_sleeve_weight, sig.n_positions,
        )

    # Step 2: group by sleeve_id, combine intra-sleeve
    sleeve_buckets: dict[str, list[StrategySignal]] = {}
    for sig in signals:
        sleeve_buckets.setdefault(sig.sleeve_id, []).append(sig)

    sleeve_weights: dict[str, pd.Series] = {}
    for sleeve_id, sigs in sleeve_buckets.items():
        sleeve_weights[sleeve_id] = _combine_intra_sleeve(sigs)

    # Step 3: build paper-trade SleeveCapitalConfig, apply allocation × LEVERAGE_FACTOR
    # Path B 1.5x leverage Tier 3 amendment 2026-05-15.
    paper_config = SleeveCapitalConfig(allocations=dict(PAPER_TRADE_SLEEVE_ALLOCATION))
    combined = combine_sleeve_weights(
        sleeve_weights,
        config=paper_config,
        leverage_factor=LEVERAGE_FACTOR,
    )

    # Step 3b: ETF Holdings LLM Risk Monitor cap overlay (spec id=49 v3 §2.10)
    # PAPER_TRADE_ONLY mode by default (ETF_HOLDINGS_DEPLOYMENT_MODE='paper_only').
    # ETFs with active cap state get per-ticker MAX_WEIGHT (typically 25%→15% × 5d).
    # Non-equity ETFs + 单股 + CTA holdings unaffected (only equity ETFs are screened).
    try:
        from engine.etf_holdings_risk_monitor import (
            get_per_ticker_max_weight_dict,
            cleanup_expired_cap_state,
        )
        # Phase A3 defense-in-depth: purge expired cap entries on every paper trade run
        # (don't rely on monthly run alone — Task Scheduler might miss a month)
        n_purged = cleanup_expired_cap_state(as_of=as_of)
        if n_purged > 0:
            logger.info("ETF Holdings: purged %d expired cap_state entries", n_purged)

        per_ticker_cap = get_per_ticker_max_weight_dict(
            base_max_weight  = 0.25,            # project MAX_WEIGHT default
            as_of            = as_of,
            paper_trade_mode = True,            # paper trade orchestrator, safe to apply
        )
        if per_ticker_cap:
            n_capped = 0
            for tkr, cap in per_ticker_cap.items():
                if tkr in combined.index:
                    orig = combined.loc[tkr]
                    if abs(orig) > cap:
                        # Scale this ticker's weight to fit per-ticker cap
                        sign = 1 if orig > 0 else -1
                        combined.loc[tkr] = sign * cap
                        n_capped += 1
            if n_capped > 0:
                logger.info("ETF Holdings cap overlay applied: %d ETFs capped (spec id=49 v3)",
                            n_capped)
    except Exception as exc:
        logger.warning("ETF Holdings cap overlay skipped (non-fatal): %s", exc)

    # Step 4: sleeve attribution = total absolute weight contributed per sleeve
    sleeve_attribution: dict[str, float] = {}
    for sleeve_id, w in sleeve_weights.items():
        if w is None or w.empty:
            sleeve_attribution[sleeve_id] = 0.0
        else:
            sleeve_attribution[sleeve_id] = float(
                w.abs().sum() * paper_config.allocations.get(sleeve_id, 0.0)
            )

    return PaperTradeRunResult(
        as_of                = as_of,
        signals              = signals,
        combined_portfolio   = combined,
        sleeve_attribution   = sleeve_attribution,
        run_timestamp_utc    = run_ts,
        errors               = errors,
        intended_allocation  = dict(PAPER_TRADE_SLEEVE_ALLOCATION),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sprint D-2: DB persistence for daily auto-run
# ─────────────────────────────────────────────────────────────────────────────
def persist_run_to_db(
    result:  PaperTradeRunResult,
    session: Optional[object] = None,
) -> dict:
    """Persist PaperTradeRunResult to PaperTradeStrategyLog table.

    One row per strategy per as_of date. Idempotent via composite PK
    (date, strategy_name) — re-running orchestrator for same day updates
    existing rows in place.

    Sprint D-2: this is the bridge between in-memory orchestrator output
    and persistent state consumed by Watchdog + future UI + analytics.

    Returns: dict with counts {inserted, updated, errors}.
    """
    import json as _json
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    inserted = 0
    updated = 0
    errors_count = 0
    try:
        for sig in result.signals:
            # Compute is_rebalance_day for this strategy
            try:
                rebal = is_rebalance_day(sig.strategy_name, result.as_of)
            except Exception:
                rebal = False

            # Build positions_json — persist FULL position set (Step 15 fix,
            # 2026-05-14). Previously truncated to top-20 with comment "keep
            # payload manageable, reconstructable from cache" — but UI consumers
            # (positions page combined aggregation, sector concentration, etc.)
            # do NOT reconstruct; they read positions_json directly. Truncation
            # was creating accounting gaps surfaced as "persist-gap" warnings
            # ($234K idle on day-1 D-PEAD: 130 of 150 names missing × $1.8K each).
            #
            # Payload sizing: D-PEAD ~150 names × ~50 bytes JSON each = ~7.5KB
            # per row. SQLite Text column trivial; per-day total across 4
            # strategies ≤ ~30KB / 11MB per year. Acceptable cost for accurate
            # downstream aggregation.
            positions_dict: dict[str, float] = {}
            if sig.weights is not None and not sig.weights.empty:
                positions_dict = {
                    str(idx): float(sig.weights[idx])
                    for idx in sig.weights.index
                    if abs(float(sig.weights[idx])) > 1e-9
                }
            signal_meta = {
                "n_positions_full": int(sig.n_positions),
                "n_positions_persisted": len(positions_dict),
                "intra_sleeve_weight": float(sig.intra_sleeve_weight),
            }

            existing = (
                sess.query(PaperTradeStrategyLog)
                    .filter_by(date=result.as_of, strategy_name=sig.strategy_name)
                    .first()
            )
            if existing:
                existing.sleeve_id           = sig.sleeve_id
                existing.status              = sig.status
                existing.is_rebalance_day    = bool(rebal)
                existing.n_positions         = int(sig.n_positions)
                existing.intra_sleeve_weight = float(sig.intra_sleeve_weight)
                existing.positions_json      = _json.dumps(positions_dict, ensure_ascii=False)
                existing.signal_metadata_json = _json.dumps(signal_meta, ensure_ascii=False)
                existing.notes               = sig.notes
                updated += 1
            else:
                row = PaperTradeStrategyLog(
                    date                = result.as_of,
                    strategy_name       = sig.strategy_name,
                    sleeve_id           = sig.sleeve_id,
                    status              = sig.status,
                    is_rebalance_day    = bool(rebal),
                    n_positions         = int(sig.n_positions),
                    intra_sleeve_weight = float(sig.intra_sleeve_weight),
                    positions_json      = _json.dumps(positions_dict, ensure_ascii=False),
                    signal_metadata_json = _json.dumps(signal_meta, ensure_ascii=False),
                    notes               = sig.notes,
                )
                sess.add(row)
                inserted += 1
        sess.commit()
    except Exception as exc:
        logger.exception("persist_run_to_db failed: %s", exc)
        sess.rollback()
        errors_count = 1
    finally:
        if own_session:
            sess.close()

    return {"inserted": inserted, "updated": updated, "errors": errors_count}


# ─────────────────────────────────────────────────────────────────────────────
# Daily return backfill (Sprint D-2 follow-up, 2026-05-14)
# ─────────────────────────────────────────────────────────────────────────────
def fill_daily_returns(
    as_of:   datetime.date,
    session: Optional[object] = None,
) -> dict:
    """Compute and persist `daily_gross_return` for each strategy row dated `as_of`.

    Convention: row[as_of].daily_gross_return = return realized between the
    prior row's date and `as_of`, evaluating the PRIOR row's positions at
    close-to-close yfinance prices. This is the standard paper-trade
    backfill flow (signal decided at t, return earned over [t, t+holding]).

    On the very first row for a strategy (no prior position) daily_gross_return
    stays None — that's accurate. `tc_drag_today` and `daily_net_return` are
    left None until ADV-based TC ships (Tier-1 #3).

    Returns: dict with counts {filled, skipped_no_prior, skipped_no_price, errors}.
    """
    import json as _json
    import yfinance as _yf
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    filled = 0
    skipped_no_prior = 0
    skipped_no_price = 0
    errors = 0
    try:
        today_rows = (
            sess.query(PaperTradeStrategyLog)
                .filter(PaperTradeStrategyLog.date == as_of)
                .all()
        )
        for row in today_rows:
            if row.daily_gross_return is not None:
                continue  # idempotent re-run
            prior = (
                sess.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.strategy_name == row.strategy_name,
                            PaperTradeStrategyLog.date < as_of)
                    .order_by(PaperTradeStrategyLog.date.desc())
                    .first()
            )
            if prior is None or not prior.positions_json:
                skipped_no_prior += 1
                continue
            try:
                prior_positions = _json.loads(prior.positions_json)
            except Exception:
                errors += 1
                continue
            if not prior_positions:
                row.daily_gross_return = 0.0
                filled += 1
                continue

            tickers = sorted(prior_positions.keys())
            try:
                # Fetch close-to-close from prior.date to as_of (auto_adjust to
                # capture dividends/splits — total return).
                start = prior.date - datetime.timedelta(days=2)
                end   = as_of + datetime.timedelta(days=2)
                data = _yf.download(
                    tickers, start=start.isoformat(), end=end.isoformat(),
                    auto_adjust=True, progress=False, multi_level_index=False,
                )
                if "Close" in data.columns:
                    close = data["Close"]
                else:
                    close = data
                if isinstance(close, pd.Series):
                    close = close.to_frame(name=tickers[0])
            except Exception as exc:
                logger.warning(
                    "yfinance fetch failed for %s on %s: %s",
                    row.strategy_name, as_of, exc,
                )
                errors += 1
                continue

            # Find close on or before prior.date, and on or before as_of
            close.index = pd.to_datetime(close.index).date
            avail_dates = sorted(d for d in close.index if d <= as_of)
            prior_close_date = max((d for d in avail_dates if d <= prior.date),
                                   default=None)
            today_close_date = max((d for d in avail_dates if d <= as_of),
                                   default=None)
            if (prior_close_date is None or today_close_date is None
                    or prior_close_date >= today_close_date):
                skipped_no_price += 1
                continue

            weighted_ret = 0.0
            usable = 0
            for tkr, w in prior_positions.items():
                if tkr not in close.columns:
                    continue
                p_prev = close.at[prior_close_date, tkr]
                p_now  = close.at[today_close_date, tkr]
                if pd.isna(p_prev) or pd.isna(p_now) or p_prev <= 0:
                    continue
                weighted_ret += float(w) * float(p_now / p_prev - 1.0)
                usable += 1
            if usable == 0:
                skipped_no_price += 1
                continue
            row.daily_gross_return = float(weighted_ret)
            filled += 1
        sess.commit()
    except Exception as exc:
        logger.exception("fill_daily_returns failed: %s", exc)
        sess.rollback()
        errors += 1
    finally:
        if own_session:
            sess.close()

    return {
        "filled":           filled,
        "skipped_no_prior": skipped_no_prior,
        "skipped_no_price": skipped_no_price,
        "errors":           errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TC drag + net-return backfill (Tier-1 audit #3 Phase B, 2026-05-14)
# ─────────────────────────────────────────────────────────────────────────────
# Simulated paper-trade NAV — used to convert turnover weights to USD for
# ADV-based TC computation. Stable across runs (paper-only deploy mode).
_PAPER_TRADE_NAV_USD = 1_000_000.0
_DEFAULT_VOL_ANN_ETF = 0.15
_DEFAULT_VOL_ANN_SS  = 0.25


def _fetch_adv_vol_batch(tickers: tuple[str, ...]) -> dict:
    """Batch-fetch 60-day ADV ($-volume) and annualized vol via yfinance.

    Returns {ticker: {"adv_usd": float, "vol_ann": float}}. Failed tickers
    get default vol + 0 ADV (which trips capacity_warning downstream).
    Used by fill_daily_tc; cached at caller scope (one fetch per run).
    """
    import yfinance as _yf
    import math as _math
    out: dict[str, dict] = {}
    if not tickers:
        return out
    try:
        data = _yf.download(
            list(tickers), period="3mo", auto_adjust=True,
            progress=False, multi_level_index=False,
        )
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            close = data["Close"]
            volume = data.get("Volume")
        else:
            close = data
            volume = None
    except Exception as exc:
        logger.warning("Batch yfinance fetch for TC failed: %s", exc)
        return {tk: {"adv_usd": 0.0, "vol_ann": _DEFAULT_VOL_ANN_SS} for tk in tickers}

    # Single-ticker → Series, multi-ticker → DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
        if volume is not None and isinstance(volume, pd.Series):
            volume = volume.to_frame(name=tickers[0])

    for tk in tickers:
        adv_usd = 0.0
        vol_ann = _DEFAULT_VOL_ANN_SS
        try:
            if tk in close.columns:
                px = close[tk].dropna().tail(60)
                if len(px) >= 20:
                    rets = px.pct_change().dropna()
                    if len(rets) > 0:
                        vol_ann = float(rets.std() * _math.sqrt(252))
                if volume is not None and tk in volume.columns:
                    vol_ser = volume[tk].dropna().tail(60)
                    if len(vol_ser) >= 20 and len(px) >= 20:
                        adv_usd = float((vol_ser * px.loc[vol_ser.index]).mean())
        except Exception:
            pass
        out[tk] = {"adv_usd": adv_usd, "vol_ann": vol_ann if vol_ann > 0 else _DEFAULT_VOL_ANN_SS}
    return out


def fill_daily_tc(
    as_of:   datetime.date,
    session: Optional[object] = None,
    nav_usd: float = _PAPER_TRADE_NAV_USD,
) -> dict:
    """Compute and persist `tc_drag_today` + `daily_net_return` for `as_of`.

    Logic per strategy row on `as_of`:
      1. Find prior row (most recent date < as_of for same strategy).
      2. Compute turnover per ticker = |w_today - w_prior| × strategy_notional
         where strategy_notional = sleeve_alloc × intra_sleeve_weight × NAV.
         (Tickers in only-one of the two sets count as full-weight turnover.)
      3. Batch-fetch ADV + vol_ann via yfinance for all turnover tickers.
      4. Call engine.execution.cost_model.compute_portfolio_tc.
      5. Write tc_drag_today = total_tc_usd / strategy_notional (decimal),
         and daily_net_return = daily_gross_return - tc_drag_today
         (NULL daily_gross_return → daily_net_return stays NULL).

    Non-rebal days where positions are identical → tc_drag_today = 0,
    daily_net_return = daily_gross_return.

    Returns: dict with {filled, skipped_no_prior, no_turnover, capacity_warnings, errors}.
    """
    import json as _json
    from engine.memory import init_db, SessionFactory
    from engine.db_models import PaperTradeStrategyLog
    from engine.execution.cost_model import compute_portfolio_tc

    init_db()
    own_session = session is None
    sess = session if session is not None else SessionFactory()

    filled = 0
    skipped_no_prior = 0
    no_turnover = 0
    capacity_warnings = 0
    errors = 0
    try:
        today_rows = (
            sess.query(PaperTradeStrategyLog)
                .filter(PaperTradeStrategyLog.date == as_of)
                .all()
        )

        # Phase 1: collect all turnover tickers across strategies for batch fetch
        per_strategy_turnover: dict[str, dict[str, float]] = {}
        all_tickers: set[str] = set()
        for row in today_rows:
            if not row.positions_json:
                continue
            prior = (
                sess.query(PaperTradeStrategyLog)
                    .filter(PaperTradeStrategyLog.strategy_name == row.strategy_name,
                            PaperTradeStrategyLog.date < as_of)
                    .order_by(PaperTradeStrategyLog.date.desc())
                    .first()
            )
            if prior is None:
                skipped_no_prior += 1
                continue
            try:
                today_pos = _json.loads(row.positions_json)
                prior_pos = _json.loads(prior.positions_json or "{}")
            except Exception:
                errors += 1
                continue
            sleeve_w = PAPER_TRADE_SLEEVE_ALLOCATION.get(row.sleeve_id, 0.0)
            intra_w  = float(row.intra_sleeve_weight or 0.0)
            strategy_notional = sleeve_w * intra_w * nav_usd
            if strategy_notional <= 0:
                continue
            # Turnover per ticker in USD
            all_tks = set(today_pos.keys()) | set(prior_pos.keys())
            turnover_usd: dict[str, float] = {}
            for tk in all_tks:
                delta = abs(float(today_pos.get(tk, 0.0)) - float(prior_pos.get(tk, 0.0)))
                if delta > 1e-9:
                    turnover_usd[tk] = delta * strategy_notional
            if not turnover_usd:
                no_turnover += 1
                # Still need to update net_return = gross (no TC)
                if row.daily_gross_return is not None:
                    row.tc_drag_today = 0.0
                    row.daily_net_return = float(row.daily_gross_return)
                    filled += 1
                continue
            per_strategy_turnover[row.strategy_name] = turnover_usd
            all_tickers.update(turnover_usd.keys())

        # Phase 2: batch-fetch ADV + vol for all unique tickers
        adv_vol_map = _fetch_adv_vol_batch(tuple(sorted(all_tickers))) if all_tickers else {}

        # Phase 3: compute TC + write per-row
        for row in today_rows:
            if row.strategy_name not in per_strategy_turnover:
                continue
            turnover_usd = per_strategy_turnover[row.strategy_name]
            sleeve_w = PAPER_TRADE_SLEEVE_ALLOCATION.get(row.sleeve_id, 0.0)
            intra_w  = float(row.intra_sleeve_weight or 0.0)
            strategy_notional = sleeve_w * intra_w * nav_usd
            adv_map = {tk: adv_vol_map.get(tk, {}).get("adv_usd", 0.0)
                       for tk in turnover_usd}
            vol_map = {tk: adv_vol_map.get(tk, {}).get("vol_ann", _DEFAULT_VOL_ANN_SS)
                       for tk in turnover_usd}
            tc_result = compute_portfolio_tc(
                turnover_usd_by_ticker=turnover_usd,
                adv_by_ticker=adv_map,
                vol_ann_by_ticker=vol_map,
                nav_usd=strategy_notional,
            )
            tc_drag = tc_result["total_tc_drag_decimal"]
            row.tc_drag_today = float(tc_drag)
            if row.daily_gross_return is not None:
                row.daily_net_return = float(row.daily_gross_return) - float(tc_drag)
            capacity_warnings += tc_result["n_capacity_warnings"]
            filled += 1

        sess.commit()
    except Exception as exc:
        logger.exception("fill_daily_tc failed: %s", exc)
        sess.rollback()
        errors += 1
    finally:
        if own_session:
            sess.close()

    return {
        "filled":             filled,
        "skipped_no_prior":   skipped_no_prior,
        "no_turnover":        no_turnover,
        "capacity_warnings":  capacity_warnings,
        "errors":             errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _format_run_result(result: PaperTradeRunResult) -> str:
    lines = []
    lines.append(f"=== Paper Trade Day: {result.as_of} ===")
    lines.append(f"Run UTC:  {result.run_timestamp_utc.isoformat()}Z")
    lines.append(f"Errors:   {len(result.errors)}")
    if result.errors:
        for e in result.errors:
            lines.append(f"  ! {e}")
    lines.append("")
    lines.append("Per-strategy signals:")
    for sig in result.signals:
        lines.append(
            f"  {sig.strategy_name:<10} [{sig.status:<9}] sleeve={sig.sleeve_id:<14} "
            f"intra_w={sig.intra_sleeve_weight:.2f}  n_pos={sig.n_positions}"
        )
        if sig.notes:
            lines.append(f"             notes: {sig.notes}")
    lines.append("")
    lines.append("Sleeve attribution (absolute gross weight contributed):")
    for sleeve_id, attrib in result.sleeve_attribution.items():
        lines.append(f"  {sleeve_id:<14}  {attrib:+.4f}  (intended alloc {result.intended_allocation.get(sleeve_id, 0.0):.2%})")
    lines.append("")
    lines.append(f"Combined portfolio: n_tickers={len(result.combined_portfolio)}, "
                 f"gross={result.combined_portfolio.abs().sum():.4f}, "
                 f"net={result.combined_portfolio.sum():+.4f}")
    if not result.combined_portfolio.empty:
        top = result.combined_portfolio.abs().sort_values(ascending=False).head(10)
        lines.append("Top 10 by |weight|:")
        for tk, w in top.items():
            signed = result.combined_portfolio[tk]
            lines.append(f"  {tk:<8}  {signed:+.4f}")
    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Paper-trade orchestrator — 4-component daily signal integration"
    )
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Run date YYYY-MM-DD (default: today UTC date)",
    )
    args = parser.parse_args()

    if args.as_of:
        as_of = datetime.date.fromisoformat(args.as_of)
    else:
        as_of = datetime.datetime.utcnow().date()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = run_paper_trade_day(as_of)
    print(_format_run_result(result))


if __name__ == "__main__":
    main()
