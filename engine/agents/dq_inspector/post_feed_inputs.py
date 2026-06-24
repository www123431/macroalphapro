"""
engine/agents/dq_inspector/post_feed_inputs.py — Phase 6c input gatherer.

Builds the kwargs dict required by orchestrator_hook.post_feed_gate by
talking to yfinance + the D-PEAD panel cache + the strategy registry.

Senior scoping decision (2026-05-19): Mode 7 / 9 active universe is
intentionally limited to K1 + AC tickers (~47), NOT the full D-PEAD
1500-stock universe. Three reasons:

  1. yfinance batch fetch on 1500 tickers in the 06:01-06:09 SGT
     critical path adds 30-60s every day for marginal anomaly-detection
     value (D-PEAD's own pipeline already validates close-prices per
     event; redundant here).

  2. D-PEAD's universe is event-driven — most stocks have NO position
     today. Anomaly check on inactive names produces noise without
     signal value.

  3. Mode 6 already gates the D-PEAD universe at the cache-coverage
     layer (rdq panel row count). Per-stock anomaly defense is
     downstream of D-PEAD's own signal validator.

So in practice:
  Mode 5 — K1 universe coverage via yfinance batch  (45 tickers, ~3s)
  Mode 6 — D-PEAD coverage via panel parquet count  (no fetch, instant)
  Mode 7 — Price anomaly on K1 + AC active universe (~47 tickers)
  Mode 9 — NaN burst on the same K1 + AC universe   (same fetch)

If a future spec amend wants D-PEAD per-stock anomaly coverage, this
file is the single place to expand `_build_active_universe`.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_PEAD_PANEL_PATH = Path("data/path_c_dhs/_pead_ts_signal_panel.parquet")


def _build_active_universe() -> tuple[list[str], dict[str, set[str]]]:
    """Union of K1 (45 ETFs) + AC TLT/GLD (2 ETFs) ≈ 47 tickers.

    Returns (sorted unique tickers, ticker → {sleeve_id, ...} map). The
    map is what Mode 7's class-aware classify_ticker consumes to pick
    the right anomaly cap.
    """
    ticker_to_sleeves: dict[str, set[str]] = {}

    try:
        from engine.path_c.k1_universe import get_k1_universe
        for ticker in set(get_k1_universe().values()):
            ticker_to_sleeves.setdefault(ticker.upper(), set()).add("etf_l1")
    except Exception as exc:
        logger.warning("post_feed_inputs: K1 universe load failed: %s", exc)

    # AC TLT/GLD insurance sleeve
    for ticker in ("TLT", "GLD"):
        ticker_to_sleeves.setdefault(ticker, set()).add("rms_crisis_hedge")

    return sorted(ticker_to_sleeves), ticker_to_sleeves


def _fetch_two_day_closes(
    tickers:  list[str],
    as_of:    datetime.date,
) -> pd.DataFrame:
    """Fetch yesterday + today closes for the universe via yfinance.

    Returns DataFrame[date, ticker] with up to 2 rows. May contain NaN
    for delisted / non-trading tickers — downstream gates count those.
    """
    if not tickers:
        return pd.DataFrame()
    try:
        from engine.signal import _fetch_closes
        # 7-day window catches Friday→Monday weekend gap + holidays
        start = as_of - datetime.timedelta(days=7)
        closes = _fetch_closes(tickers, start, as_of)
        return closes
    except Exception as exc:
        logger.warning("post_feed_inputs: _fetch_closes failed: %s", exc)
        return pd.DataFrame()


def _count_pead_universe_coverage() -> int:
    """Return number of unique tickers in the D-PEAD panel parquet.

    Empty-or-missing panel → 0 (Mode 6 will then HARD HALT, which is
    correct — D-PEAD cannot generate signals without a panel).
    """
    if not _PEAD_PANEL_PATH.exists():
        return 0
    try:
        panel = pd.read_parquet(_PEAD_PANEL_PATH, columns=["ticker"])
        return int(panel["ticker"].nunique())
    except Exception as exc:
        logger.warning("post_feed_inputs: PEAD panel read failed: %s", exc)
        return 0


def gather_post_feed_inputs(as_of: datetime.date) -> dict:
    """Build the kwargs dict consumed by post_feed_gate.

    All numeric metrics computed defensively — failures degrade to
    "no data" values that produce SOFT (not spurious HARD HALT) signals
    downstream, matching the pre_batch pattern.
    """
    tickers, ticker_to_sleeves = _build_active_universe()

    closes = _fetch_two_day_closes(tickers, as_of)

    # Mode 5 — K1 universe coverage. n_with_price = K1 tickers with a
    # non-NaN row anywhere in the 2-day window (yfinance occasionally
    # has same-day gaps; conservative on the OK side).
    try:
        from engine.path_c.k1_universe import get_k1_universe
        k1_tickers = {t.upper() for t in get_k1_universe().values()}
    except Exception:
        k1_tickers = set()
    if closes.empty:
        k1_n_with_price = 0
    else:
        k1_priced = {t for t in closes.columns if t.upper() in k1_tickers
                     and closes[t].notna().any()}
        k1_n_with_price = len(k1_priced)

    # Mode 6 — D-PEAD universe coverage from panel parquet
    pead_n_with_rdq = _count_pead_universe_coverage()

    # Mode 7 — daily returns. Compute pct_change on 2-day closes; pick
    # the most recent valid return per ticker.
    if closes.empty:
        daily_returns: pd.Series = pd.Series(dtype=float)
    else:
        rets = closes.pct_change(fill_method=None)
        # Last available return per ticker
        daily_returns = rets.ffill().iloc[-1] if len(rets) > 0 else pd.Series(dtype=float)
        daily_returns = daily_returns.dropna()

    # Mode 9 — NaN burst on TODAY's close across active universe
    n_universe = len(tickers)
    if closes.empty:
        n_nan_close = n_universe          # 100% NaN if fetch failed
    else:
        today_ts = pd.Timestamp(as_of)
        # yfinance may return last trading day if today is weekend; use last available row
        if len(closes.index) == 0:
            today_row = pd.Series(dtype=float)
        else:
            today_row = closes.iloc[-1]
        # Count NaN cells in today's row across active universe
        coverage_set = set(closes.columns)
        n_nan_close = sum(
            1 for t in tickers
            if t not in coverage_set or pd.isna(today_row.get(t, np.nan))
        )

    logger.info(
        "post_feed_inputs: as_of=%s n_universe=%d k1_priced=%d pead_panel_rows=%d "
        "daily_returns_n=%d nan_close=%d",
        as_of, n_universe, k1_n_with_price, pead_n_with_rdq,
        len(daily_returns), n_nan_close,
    )

    return {
        "as_of":              as_of,
        "k1_n_with_price":    k1_n_with_price,
        "pead_n_with_rdq":    pead_n_with_rdq,
        "daily_returns":      daily_returns,
        "ticker_to_sleeves":  ticker_to_sleeves,
        "n_nan_close":        n_nan_close,
        "n_universe":         n_universe,
    }
