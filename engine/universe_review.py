"""
engine/universe_review.py — Quarterly Universe Review
=======================================================
Quarterly process that evaluates whether active universe ETFs should be
exited or replaced based on liquidity, correlation, and momentum decay.

Design constraints (P6 blueprint):
  - EXIT_AUM     : $500M (ETF AUM threshold for forced review)
  - UNIVERSE_MAX : 40 active members at any time
  - Gate         : GATE_UNIVERSE_CHANGE must be "true" in SystemConfig
  - All exit/add proposals → PendingApproval(universe_change) for human review
  - No automatic removals — purely advisory

Called from: daily_batch.py quarterly ERA thread (Phase 5)
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_GATE_KEY      = "GATE_UNIVERSE_CHANGE"
_EXIT_AUM_M    = 500.0      # $500M AUM threshold below which ETF is flagged
_UNIVERSE_MAX  = 40         # Maximum active universe members
_CORR_CAP      = 0.90       # Flag pair if rolling 12M correlation > 0.90
_CORR_WINDOW   = 252        # Trading days for correlation window


# ── Gate ──────────────────────────────────────────────────────────────────────

def _gate_enabled() -> bool:
    try:
        from engine.memory import get_system_config
        return str(get_system_config(_GATE_KEY, "false")).lower() == "true"
    except Exception:
        return False


# ── Data helpers ──────────────────────────────────────────────────────────────

@dataclass
class UniverseReviewResult:
    as_of:            datetime.date
    n_active:         int              = 0
    liquidity_flags:  list[str]        = field(default_factory=list)
    correlation_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    momentum_decay:   list[str]        = field(default_factory=list)
    over_capacity:    bool             = False
    proposals_written: int             = 0
    skipped_reason:   str             = ""


def _get_active_universe_etfs() -> dict[str, str]:
    """Return {sector: ticker} for all active universe members."""
    try:
        from engine.universe_manager import get_active_universe
        return get_active_universe()
    except Exception:
        from engine.history import get_active_sector_etf
        return get_active_sector_etf()


def _check_aum(ticker: str) -> float | None:
    """Return ETF AUM in $M from yfinance, or None on failure."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        aum = info.get("totalAssets") or info.get("fundInceptionDate")
        if isinstance(aum, (int, float)) and aum > 0:
            return float(aum) / 1_000_000   # convert to $M
    except Exception:
        pass
    return None


def _check_liquidity_flags(active_etf: dict[str, str]) -> list[str]:
    """Return list of sector names where ETF AUM < EXIT_AUM_M."""
    flags: list[str] = []
    for sector, ticker in active_etf.items():
        aum = _check_aum(ticker)
        if aum is not None and aum < _EXIT_AUM_M:
            flags.append(sector)
            logger.debug("universe_review: %s AUM $%.0fM < $%.0fM threshold", ticker, aum, _EXIT_AUM_M)
    return flags


def _check_high_correlations(
    active_etf: dict[str, str],
    as_of: datetime.date,
) -> list[tuple[str, str, float]]:
    """
    Return sector pairs with rolling 12M daily-return correlation > CORR_CAP.
    High correlation (>0.90) means the pair is nearly redundant.
    """
    try:
        import yfinance as yf
        import pandas as pd
        tickers = list(active_etf.values())
        sector_map = {v: k for k, v in active_etf.items()}
        start = as_of - datetime.timedelta(days=_CORR_WINDOW + 30)
        px = yf.download(
            tickers, start=str(start), end=str(as_of),
            auto_adjust=True, progress=False,
        )
        if isinstance(px.columns, pd.MultiIndex):
            closes = px["Close"]
        else:
            closes = px[["Close"]].rename(columns={"Close": tickers[0]})

        rets   = closes.pct_change().dropna()
        corr_m = rets.corr()

        flagged: list[tuple[str, str, float]] = []
        tickers_present = [t for t in tickers if t in corr_m.columns]
        for i, t1 in enumerate(tickers_present):
            for t2 in tickers_present[i + 1:]:
                c = float(corr_m.loc[t1, t2])
                if c > _CORR_CAP:
                    s1 = sector_map.get(t1, t1)
                    s2 = sector_map.get(t2, t2)
                    flagged.append((s1, s2, round(c, 3)))
        return flagged
    except Exception as exc:
        logger.debug("universe_review: correlation check failed: %s", exc)
        return []


def _check_momentum_decay(
    active_etf: dict[str, str],
    as_of: datetime.date,
) -> list[str]:
    """
    Flag sectors where 12-1M TSMOM has been -1 (short) for ≥3 consecutive months.
    Persistent short signal on a universe member suggests thesis drift.
    """
    try:
        from engine.memory import SignalRecord, SessionFactory
        decayed: list[str] = []
        cutoff = as_of - datetime.timedelta(days=100)   # ~3 months
        with SessionFactory() as s:
            for sector, ticker in active_etf.items():
                recent = (
                    s.query(SignalRecord.tsmom_signal)
                     .filter(
                         SignalRecord.ticker == ticker,
                         SignalRecord.date >= cutoff,
                         SignalRecord.date <= as_of,
                     )
                     .order_by(SignalRecord.date.desc())
                     .limit(3)
                     .all()
                )
                signals = [r[0] for r in recent if r[0] is not None]
                if len(signals) >= 3 and all(s == -1 for s in signals):
                    decayed.append(sector)
        return decayed
    except Exception as exc:
        logger.debug("universe_review: momentum decay check failed: %s", exc)
        return []


def _write_proposals(
    as_of: datetime.date,
    flags: list[str],
    reason_prefix: str,
) -> int:
    """Write PendingApproval(universe_change) rows for flagged sectors."""
    from engine.memory import PendingApproval, SessionFactory
    written = 0
    active_etf = _get_active_universe_etfs()
    with SessionFactory() as s:
        for sector in flags:
            ticker = active_etf.get(sector, "UNKNOWN")
            existing = (
                s.query(PendingApproval)
                 .filter(
                     PendingApproval.approval_type == "universe_change",
                     PendingApproval.status == "pending",
                     PendingApproval.sector == sector,
                 )
                 .first()
            )
            if not existing:
                s.add(PendingApproval(
                    approval_type="universe_change",
                    priority="normal",
                    sector=sector,
                    ticker=ticker,
                    triggered_condition=f"{reason_prefix}: {sector} ({ticker})",
                    triggered_date=as_of,
                    suggested_weight=None,
                ))
                written += 1
        s.commit()
    return written


# ── Main entry point ───────────────────────────────────────────────────────────

def run_universe_review(as_of: datetime.date) -> UniverseReviewResult:
    """
    Run quarterly universe review. Returns a UniverseReviewResult summary.
    All exit proposals require human approval (GATE_UNIVERSE_CHANGE guards entry).
    """
    result = UniverseReviewResult(as_of=as_of)

    if not _gate_enabled():
        result.skipped_reason = "GATE_UNIVERSE_CHANGE disabled"
        logger.debug("universe_review: gate disabled — skipping")
        return result

    active_etf = _get_active_universe_etfs()
    result.n_active = len(active_etf)

    if result.n_active > _UNIVERSE_MAX:
        result.over_capacity = True
        logger.warning(
            "universe_review: %d active members exceeds MAX=%d",
            result.n_active, _UNIVERSE_MAX,
        )

    # ── 1. Liquidity check (AUM < $500M) ─────────────────────────────────────
    result.liquidity_flags = _check_liquidity_flags(active_etf)

    # ── 2. High-correlation pairs (>0.90) ─────────────────────────────────────
    result.correlation_pairs = _check_high_correlations(active_etf, as_of)

    # ── 3. Persistent short momentum (TSMOM = -1 for ≥3M) ────────────────────
    result.momentum_decay = _check_momentum_decay(active_etf, as_of)

    # ── 4. Write proposals ────────────────────────────────────────────────────
    n = 0
    if result.liquidity_flags:
        n += _write_proposals(as_of, result.liquidity_flags,
                              f"Universe review: AUM < ${_EXIT_AUM_M:.0f}M")
    if result.momentum_decay:
        n += _write_proposals(as_of, result.momentum_decay,
                              "Universe review: TSMOM -1 持续≥3个月")
    # Correlation pairs: flag the lower-liquidity member of each pair
    corr_flag_sectors: list[str] = []
    for s1, s2, corr in result.correlation_pairs:
        aum1 = _check_aum(active_etf.get(s1, "")) or 0.0
        aum2 = _check_aum(active_etf.get(s2, "")) or 0.0
        to_flag = s1 if aum1 <= aum2 else s2
        corr_flag_sectors.append(to_flag)
    if corr_flag_sectors:
        n += _write_proposals(
            as_of, list(set(corr_flag_sectors)),
            f"Universe review: 相关性>{_CORR_CAP:.0%}高度重叠",
        )
    result.proposals_written = n

    logger.info(
        "universe_review %s: n_active=%d liquidity=%d corr_pairs=%d decay=%d proposals=%d",
        as_of,
        result.n_active,
        len(result.liquidity_flags),
        len(result.correlation_pairs),
        len(result.momentum_decay),
        result.proposals_written,
    )
    return result
