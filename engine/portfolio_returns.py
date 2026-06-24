"""
engine/portfolio_returns.py — Daily NAV rollup with cash-flow normalization.

Spec: docs/spec_performance_reporting_v1.md  (sha256[:16] f1c9b693f7a6a6df).

Per spec §3.2 each PortfolioNavSnapshot row records three NAV states:
  nav_open       = previous day's nav_close (or initial NAV on cold start)
  nav_after_flow = nav_open + external_flow (deposit/withdraw)
  nav_close      = nav_after_flow × (1 + portfolio_gross_return)

Why no internal flow in nav_close:
  yfinance auto_adjust=True returns total-return adjusted prices, i.e. dividend
  reinvested into asset close. Adding dividend cash flow on top would double-
  count. CashFlow.is_external=False rows are decorative audit (project tracks
  dividends conceptually) but do NOT enter the rollup. If a future spec
  amendment switches to auto_adjust=False (rare), this contract changes.

Public API:
  roll_daily_nav(date, *, force=False, return_provider=None) -> dict
  get_nav_series(start, end) -> pd.DataFrame
  get_nav_with_flows(start, end) -> pd.DataFrame
  initial_nav() -> float
"""
from __future__ import annotations

import datetime
import logging
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


def initial_nav() -> float:
    """
    Default starting NAV for the portfolio. Reads SystemConfig
    `paper_trading_nav` (defaults to 1_000_000.0 USD).
    """
    try:
        from engine.memory import get_system_config
        return float(get_system_config("paper_trading_nav", "1000000"))
    except Exception:
        return 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Asset-return provider (pluggable for tests)
# ─────────────────────────────────────────────────────────────────────────────

def _default_return_provider(
    tickers: list[str],
    date:    datetime.date,
) -> dict[str, float]:
    """
    Default: pull yfinance daily return for each ticker on `date`.
    Returns {ticker: daily_return_decimal}; missing tickers map to 0.0.

    auto_adjust=True so dividends are already in the close price.
    """
    if not tickers:
        return {}
    try:
        import yfinance as yf
        # Pull a small window to ensure the target date has both close-1 and close
        prev = date - datetime.timedelta(days=10)
        df = yf.download(
            tickers, start=prev.isoformat(),
            end=(date + datetime.timedelta(days=2)).isoformat(),
            auto_adjust=True, progress=False, multi_level_index=False,
            threads=False,
        )
        if df is None or df.empty or "Close" not in df.columns:
            return {t: 0.0 for t in tickers}
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0])
        # date returns = pct change; pick the row at <= target date
        target = pd.Timestamp(date).normalize()
        # Filter index to <= target, take last available for each ticker
        close = close.sort_index()
        close = close[close.index <= target]
        if len(close) < 2:
            return {t: 0.0 for t in tickers}
        last_two = close.tail(2)
        ret_row = last_two.iloc[-1] / last_two.iloc[-2] - 1.0
        out = {}
        for t in tickers:
            v = ret_row.get(t)
            out[t] = float(v) if v is not None and pd.notna(v) else 0.0
        return out
    except Exception as exc:
        logger.warning("default_return_provider yfinance failed: %s", exc)
        return {t: 0.0 for t in tickers}


# ─────────────────────────────────────────────────────────────────────────────
# Rollup
# ─────────────────────────────────────────────────────────────────────────────

def roll_daily_nav(
    date:           datetime.date,
    *,
    force:          bool = False,
    return_provider: Callable | None = None,
    session:        Any | None = None,
) -> dict:
    """
    Compute and persist a PortfolioNavSnapshot for `date`. Idempotent: if a
    snapshot already exists for `date`, returns it unchanged unless force=True.

    Returns dict view of the snapshot.
    """
    from engine.memory import (
        PortfolioNavSnapshot, CashFlow, SessionFactory,
    )
    from engine.portfolio_tracker import get_current_positions
    from sqlalchemy import func

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        existing = sess.query(PortfolioNavSnapshot).filter(
            PortfolioNavSnapshot.snapshot_date == date
        ).one_or_none()
        if existing and not force:
            return _row_to_dict(existing)
        if existing:
            sess.delete(existing)
            sess.flush()

        # ── 1. nav_open: previous snapshot's nav_close, else initial ────────
        prev = (
            sess.query(PortfolioNavSnapshot)
                .filter(PortfolioNavSnapshot.snapshot_date < date)
                .order_by(PortfolioNavSnapshot.snapshot_date.desc())
                .first()
        )
        if prev is not None:
            nav_open = float(prev.nav_close)
        else:
            # Cold start: initial NAV (system config) + any *applied* CashFlow
            # rows BEFORE `date` count as funded capital.
            base = initial_nav()
            prior_external = sess.query(
                func.coalesce(func.sum(CashFlow.amount_usd), 0.0)
            ).filter(
                CashFlow.flow_date < date,
                CashFlow.status == "applied",
                CashFlow.is_external.is_(True),
            ).scalar() or 0.0
            nav_open = float(base + prior_external)

        # ── 2. external flow today ─────────────────────────────────────────
        ext_flow_today = sess.query(
            func.coalesce(func.sum(CashFlow.amount_usd), 0.0)
        ).filter(
            CashFlow.flow_date == date,
            CashFlow.status == "applied",
            CashFlow.is_external.is_(True),
        ).scalar() or 0.0
        ext_flow_today = float(ext_flow_today)

        nav_after_flow = nav_open + ext_flow_today

        # ── 3. portfolio gross return ──────────────────────────────────────
        try:
            positions_df = get_current_positions(as_of=date)
        except Exception as exc:
            logger.warning("roll_daily_nav: get_current_positions failed: %s", exc)
            positions_df = None

        portfolio_ret = 0.0
        if positions_df is not None and not positions_df.empty:
            tickers = list(positions_df["ticker"].dropna().unique())
            provider = return_provider or _default_return_provider
            try:
                ret_map = provider(tickers, date)
            except Exception as exc:
                logger.warning("roll_daily_nav: return_provider failed: %s", exc)
                ret_map = {t: 0.0 for t in tickers}
            # Sum weight × return; use actual_weight if available, else target
            for _, row in positions_df.iterrows():
                w = row.get("actual_weight")
                if w is None or pd.isna(w):
                    w = row.get("target_weight") or 0.0
                tkr = row.get("ticker")
                r_t = float(ret_map.get(tkr, 0.0) or 0.0)
                portfolio_ret += float(w) * r_t

        nav_close = nav_after_flow * (1.0 + portfolio_ret)
        gross_pnl = nav_close - nav_after_flow

        # ── 4. daily Modified Dietz with start-of-day flow assumption ──────
        # Per spec §2.1: when external flow at start of day, w=1, denom=nav_open+ext.
        if abs(nav_after_flow) < 1e-9:
            daily_md = 0.0
        else:
            daily_md = (nav_close - nav_open - ext_flow_today) / nav_after_flow

        # ── 5. benchmark close (SPY total-return) ──────────────────────────
        bench = None
        try:
            import yfinance as yf
            spy = yf.download(
                "SPY",
                start=(date - datetime.timedelta(days=5)).isoformat(),
                end=(date + datetime.timedelta(days=2)).isoformat(),
                auto_adjust=True, progress=False, multi_level_index=False,
                threads=False,
            )
            if spy is not None and not spy.empty:
                spy = spy.sort_index()
                spy = spy[spy.index <= pd.Timestamp(date).normalize()]
                if len(spy):
                    bench = float(spy["Close"].iloc[-1])
        except Exception:
            pass

        row = PortfolioNavSnapshot(
            snapshot_date         = date,
            nav_open              = nav_open,
            external_flow         = ext_flow_today,
            nav_after_flow        = nav_after_flow,
            nav_close             = nav_close,
            gross_pnl             = gross_pnl,
            benchmark_close       = bench,
            daily_modified_dietz  = daily_md,
            created_at            = datetime.datetime.utcnow(),
        )
        sess.add(row)
        sess.commit()
        logger.info(
            "roll_daily_nav %s: nav_open=%.2f ext=%+.2f nav_close=%.2f "
            "gross_pnl=%+.2f md=%+.4f%%",
            date, nav_open, ext_flow_today, nav_close, gross_pnl,
            daily_md * 100,
        )
        return _row_to_dict(row)
    finally:
        if own:
            sess.close()


def _row_to_dict(r) -> dict:
    return {
        "snapshot_date":        r.snapshot_date,
        "nav_open":             r.nav_open,
        "external_flow":        r.external_flow,
        "nav_after_flow":       r.nav_after_flow,
        "nav_close":            r.nav_close,
        "gross_pnl":            r.gross_pnl,
        "benchmark_close":      r.benchmark_close,
        "daily_modified_dietz": r.daily_modified_dietz,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Read API
# ─────────────────────────────────────────────────────────────────────────────

def get_nav_series(
    start: datetime.date | None = None,
    end:   datetime.date | None = None,
    *,
    session: Any | None = None,
) -> pd.DataFrame:
    """Return DataFrame indexed by snapshot_date with NAV cols."""
    from engine.memory import PortfolioNavSnapshot, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        q = sess.query(PortfolioNavSnapshot)
        if start is not None:
            q = q.filter(PortfolioNavSnapshot.snapshot_date >= start)
        if end is not None:
            q = q.filter(PortfolioNavSnapshot.snapshot_date <= end)
        rows = q.order_by(PortfolioNavSnapshot.snapshot_date.asc()).all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([_row_to_dict(r) for r in rows])
        df = df.set_index("snapshot_date")
        return df
    finally:
        if own:
            sess.close()


def get_nav_with_flows(
    start: datetime.date | None = None,
    end:   datetime.date | None = None,
    *,
    session: Any | None = None,
) -> pd.DataFrame:
    """
    NAV series with external cash-flow markers as a 'flow' column. Useful for
    UI overlays (NAV chart with deposit/withdraw markers).
    """
    from engine.cash_management import get_cash_flow_history

    nav_df = get_nav_series(start=start, end=end, session=session)
    if nav_df.empty:
        return nav_df

    flows = get_cash_flow_history(
        start=start, end=end, external_only=True, applied_only=True,
        session=session,
    )
    flow_by_date: dict[datetime.date, float] = {}
    for f in flows:
        d = f["flow_date"]
        flow_by_date[d] = flow_by_date.get(d, 0.0) + f["amount_usd"]
    nav_df["flow"] = [flow_by_date.get(d, 0.0) for d in nav_df.index]
    return nav_df
