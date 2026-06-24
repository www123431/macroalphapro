"""
Portfolio Tracker — Simulated Execution Layer
=============================================
Stateful forward-testing engine that bridges the statistical signal pipeline
and live trading. Unlike engine/backtest.py (stateless historical replay),
this module maintains position state across rebalancing cycles.

Key distinction vs backtest
---------------------------
- backtest.py  : uses historical data, stateless, each month independent
- portfolio_tracker.py : runs in real time, position state persists,
                         last month's actual_weight is the starting point
                         for this month's trade generation.

Core functions
--------------
  get_current_positions(as_of)         → latest SimulatedPosition snapshot
  generate_rebalance_trades(...)       → trade list from current → target
  execute_rebalance(date, dry_run)     → full monthly rebalance cycle
  record_monthly_return(month)         → post-month return attribution

Integration
-----------
  Consumes: engine/signal.get_signal_dataframe()
            engine/regime.get_regime_on()
            engine/portfolio.construct_portfolio()
            engine/memory.SimulatedPosition / SimulatedTrade / SimulatedMonthlyReturn
"""

from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from engine.memory import (
    SessionFactory,
    SimulatedPosition,
    SimulatedTrade,
    SimulatedMonthlyReturn,
    get_system_config,
)
from engine.signal import get_signal_dataframe
from engine.regime import get_regime_on
from engine.portfolio import construct_portfolio
from engine.history import get_active_sector_etf

logger = logging.getLogger(__name__)

# ── Cost assumptions (matching backtest.py) ────────────────────────────────────
_COST_BPS = 10          # one-way transaction cost estimate
_MIN_TRADE_SIZE = 0.005 # 0.5% — trades smaller than this are skipped


def _get_nav() -> float:
    """Read paper-trading NAV from SystemConfig. Default 1,000,000."""
    try:
        return float(get_system_config("paper_trading_nav", "1000000"))
    except Exception:
        return 1_000_000.0


def _compute_shares(weight: float, nav: float, price: float) -> float | None:
    """Convert weight fraction to share count. Returns None if price unavailable."""
    if price and price > 0 and nav > 0:
        return abs(weight) * nav / price
    return None


# ── Read layer ─────────────────────────────────────────────────────────────────

def get_current_positions(as_of: Optional[datetime.date] = None) -> pd.DataFrame:
    """
    Return the most recent SimulatedPosition snapshot as a DataFrame.

    Args:
        as_of: if provided, return the latest snapshot on or before this date.
               if None, return the most recent snapshot in the DB.

    Returns:
        DataFrame with index=sector and columns:
            ticker, target_weight, actual_weight, entry_price,
            regime_label, signal_tsmom, snapshot_date, notes
        Empty DataFrame if no positions found.
    """
    with SessionFactory() as session:
        q = session.query(SimulatedPosition)
        if as_of is not None:
            q = q.filter(SimulatedPosition.snapshot_date <= as_of)

        # Find the latest snapshot_date
        latest = (
            session.query(SimulatedPosition.snapshot_date)
            .order_by(SimulatedPosition.snapshot_date.desc())
            .first()
        )
        if latest is None:
            return pd.DataFrame()

        latest_date = latest[0]
        if as_of is not None:
            rows = (
                session.query(SimulatedPosition)
                .filter(SimulatedPosition.snapshot_date == latest_date)
                .all()
            )
        else:
            rows = (
                session.query(SimulatedPosition)
                .filter(SimulatedPosition.snapshot_date == latest_date)
                .all()
            )

        if not rows:
            return pd.DataFrame()

        records = [
            {
                "sector":         r.sector,
                "ticker":         r.ticker,
                "snapshot_date":  r.snapshot_date,
                "target_weight":  r.target_weight,
                "actual_weight":  r.actual_weight if r.actual_weight is not None else r.target_weight,
                "entry_price":    r.entry_price,
                "regime_label":   r.regime_label,
                "signal_tsmom":   r.signal_tsmom,
                "notes":          r.notes,
                "shares_held":    r.shares_held,
                "cost_basis":     r.cost_basis,
                "position_value": r.position_value,
                "direction":      r.direction,
                "trailing_high":  r.trailing_high,
            }
            for r in rows
        ]
        df = pd.DataFrame(records).set_index("sector")
        return df


# ── Trade generation ───────────────────────────────────────────────────────────

def generate_rebalance_trades(
    current_positions: pd.DataFrame,
    new_weights:       pd.Series,
    signal_df:         pd.DataFrame,
    prev_signal_df:    Optional[pd.DataFrame],
    regime:            Optional[object],
    prev_regime_label: Optional[str],
    rebalance_date:    datetime.date,
    min_trade_size:    float = _MIN_TRADE_SIZE,
) -> list[dict]:
    """
    Compare current positions against target weights and generate trade list.

    Trigger reason priority:
      1. signal_flip   — TSMOM direction reversed since last snapshot
      2. regime_change — MSM regime label changed
      3. rebalance     — weight drift exceeds min_trade_size
      4. new_position  — sector not in current positions (first entry)

    Args:
        current_positions : output of get_current_positions()
        new_weights       : pd.Series (index=sector, values=target_weight)
        signal_df         : current month's signal DataFrame
        prev_signal_df    : last month's signal DataFrame (may be None)
        regime            : current RegimeResult
        prev_regime_label : regime label from last snapshot (may be None)
        rebalance_date    : date of this rebalancing cycle
        min_trade_size    : minimum |delta| to generate a trade

    Returns:
        List of trade dicts, each with keys matching SimulatedTrade fields.
    """
    trades = []

    # All sectors to consider: union of current holdings and new targets
    current_sectors = set(current_positions.index) if not current_positions.empty else set()
    target_sectors  = set(new_weights.index[new_weights != 0])
    all_sectors     = current_sectors | target_sectors

    current_regime = regime.regime if regime is not None else None

    for sector in all_sectors:
        ticker = get_active_sector_etf().get(sector, "")
        w_before = float(current_positions.loc[sector, "actual_weight"]) \
            if sector in current_sectors else 0.0
        w_after  = float(new_weights.get(sector, 0.0))
        delta    = w_after - w_before

        # Skip negligible moves
        if abs(delta) < min_trade_size:
            continue

        # Determine action
        action = "BUY" if delta > 0 else "SELL"

        # Determine trigger reason
        _dir = {1: "多头", -1: "空头", 0: "平仓"}
        trigger = "月度权重漂移再平衡"

        # TSMOM signal changed (any direction shift, including to/from flat)
        if prev_signal_df is not None and not prev_signal_df.empty and \
                signal_df is not None and not signal_df.empty and \
                sector in signal_df.index and sector in prev_signal_df.index:
            cur_sig  = float(signal_df.loc[sector, "tsmom"])
            prev_sig = float(prev_signal_df.loc[sector, "tsmom"])
            if cur_sig != prev_sig and (cur_sig != 0 or prev_sig != 0):
                _from = _dir.get(int(prev_sig), f"{int(prev_sig):+d}")
                _to   = _dir.get(int(cur_sig),  f"{int(cur_sig):+d}")
                trigger = f"TSMOM动量信号翻转（{_from}→{_to}）"

        # Macro regime switch — lower priority than signal flip
        if trigger == "月度权重漂移再平衡" and prev_regime_label is not None and \
                current_regime is not None and current_regime != prev_regime_label:
            trigger = f"宏观制度切换（{prev_regime_label.upper()}→{current_regime.upper()}）"

        # First entry into this sector
        if sector not in current_sectors:
            trigger = "新建仓位（首次入场）"

        # Full exit from a sector
        if w_after == 0.0 and sector in current_sectors:
            action = "SELL"
            if trigger == "月度权重漂移再平衡":
                trigger = "权重归零（平仓退出）"

        trades.append({
            "trade_date":     rebalance_date,
            "sector":         sector,
            "ticker":         ticker,
            "action":         action,
            "weight_before":  round(w_before, 6),
            "weight_after":   round(w_after,  6),
            "weight_delta":   round(delta,    6),
            "cost_bps":       round(abs(delta) * _COST_BPS, 2),
            "trigger_reason": trigger,
        })

    return trades


# ── Full rebalance cycle ───────────────────────────────────────────────────────

def execute_rebalance(
    rebalance_date: datetime.date,
    dry_run:        bool = True,
    lookback_months: int = 12,
    skip_months:     int = 1,
    nav:            float | None = None,
) -> dict:
    """
    Run the full monthly rebalancing cycle.

    Steps:
      1. get_current_positions()          — current holdings
      2. get_signal_dataframe()           — compute this month's signals
      3. get_regime_on()                  — current macro regime
      4. construct_portfolio()            — build target weights
      5. generate_rebalance_trades()      — compute delta vs current
      6. Calculate turnover and cost
      7. dry_run=False → write trades + new positions to DB

    Args:
        rebalance_date   : month-end date for this rebalancing
        dry_run          : if True, compute and return but do not persist
        lookback_months  : TSMOM formation period (default 12)
        skip_months      : skip most recent months (default 1)

    Returns:
        {
          "trades":          list[dict],
          "new_positions":   dict[sector → {weight, ticker, signal}],
          "total_cost_bps":  float,
          "turnover":        float,   # sum of |weight_delta|
          "regime":          str,
          "n_long":          int,
          "n_short":         int,
          "warnings":        list[str],
        }
    """
    warnings_out: list[str] = []
    _nav = nav if nav is not None else _get_nav()

    # ── Step 1: current positions ──────────────────────────────────────────────
    current_pos = get_current_positions(as_of=rebalance_date)
    prev_regime_label = None
    if not current_pos.empty and "regime_label" in current_pos.columns:
        labels = current_pos["regime_label"].dropna().unique()
        if len(labels) > 0:
            prev_regime_label = str(labels[0])

    # ── Step 2: signal DataFrame ───────────────────────────────────────────────
    signal_df = get_signal_dataframe(
        as_of=rebalance_date,
        lookback_months=lookback_months,
        skip_months=skip_months,
    )
    if signal_df.empty:
        warnings_out.append("信号计算失败，无法执行再平衡")
        return {"trades": [], "new_positions": {}, "total_cost_bps": 0.0,
                "turnover": 0.0, "regime": "unknown", "n_long": 0,
                "n_short": 0, "warnings": warnings_out}

    # Previous signal (from last snapshot date) for signal_flip detection
    prev_signal_df = None
    if not current_pos.empty:
        prev_snap_date = current_pos["snapshot_date"].iloc[0] \
            if "snapshot_date" in current_pos.columns else None
        if prev_snap_date is not None:
            try:
                prev_signal_df = get_signal_dataframe(
                    as_of=prev_snap_date,
                    lookback_months=lookback_months,
                    skip_months=skip_months,
                )
            except Exception as e:
                logger.warning("prev_signal_df fetch failed: %s", e)

    # ── Step 3: regime ─────────────────────────────────────────────────────────
    try:
        regime = get_regime_on(as_of=rebalance_date, train_end=rebalance_date)
    except Exception as e:
        logger.warning("get_regime_on failed: %s", e)
        regime = None
        warnings_out.append(f"Regime 检测失败，使用无 overlay 模式: {e}")

    # ── Step 4: target portfolio ───────────────────────────────────────────────
    portfolio = construct_portfolio(signal_df, regime=regime)
    if portfolio.weights.empty:
        warnings_out.extend(portfolio.warnings)
        warnings_out.append("组合构建失败，目标权重为空")
        return {"trades": [], "new_positions": {}, "total_cost_bps": 0.0,
                "turnover": 0.0, "regime": regime.regime if regime else "unknown",
                "n_long": 0, "n_short": 0, "warnings": warnings_out}
    warnings_out.extend(portfolio.warnings)

    # ── Step 5: generate trades ────────────────────────────────────────────────
    trades = generate_rebalance_trades(
        current_positions=current_pos,
        new_weights=portfolio.weights,
        signal_df=signal_df,
        prev_signal_df=prev_signal_df,
        regime=regime,
        prev_regime_label=prev_regime_label,
        rebalance_date=rebalance_date,
    )

    # ── Step 5b: build new_positions_list (fetches prices) ───────────────────
    # Must come before trade enrichment so _price_map is available.
    regime_label = regime.regime if regime is not None else "unknown"
    new_positions_list = []
    for sector, weight in portfolio.weights.items():
        ticker = get_active_sector_etf().get(sector, "")
        tsmom_val = int(signal_df.loc[sector, "tsmom"]) \
            if sector in signal_df.index else 0

        # Fetch closing price (best-effort)
        entry_price = None
        try:
            px = yf.download(ticker, start=str(rebalance_date - datetime.timedelta(days=5)),
                             end=str(rebalance_date + datetime.timedelta(days=1)),
                             progress=False, auto_adjust=True)
            if not px.empty:
                entry_price = float(px["Close"].iloc[-1])
        except Exception:
            pass

        _shares    = _compute_shares(float(weight), _nav, entry_price)
        _cost      = round(_shares * entry_price, 2) if _shares and entry_price else None
        _pos_value = _cost

        new_positions_list.append({
            "sector":         sector,
            "ticker":         ticker,
            "target_weight":  round(float(weight), 6),
            "actual_weight":  round(float(weight), 6),
            "entry_price":    entry_price,
            "regime_label":   regime_label,
            "signal_tsmom":   tsmom_val,
            "shares_held":    _shares,
            "cost_basis":     _cost,
            "position_value": _pos_value,
        })

    # ── Step 5c: enrich trades with share-level execution fields ─────────────
    _price_map = {p["sector"]: p.get("entry_price") for p in new_positions_list}
    for t in trades:
        _px = _price_map.get(t["sector"])
        if _px and _px > 0:
            _trade_shares = _compute_shares(abs(t["weight_delta"]), _nav, _px)
            t["fill_price"] = round(_px, 4)
            t["shares"]     = round(_trade_shares, 4) if _trade_shares else None
            t["notional"]   = round(_trade_shares * _px, 2) if _trade_shares else None
        else:
            t["fill_price"] = None
            t["shares"]     = None
            t["notional"]   = None

    # ── Step 6: summary metrics ────────────────────────────────────────────────
    turnover      = sum(abs(t["weight_delta"]) for t in trades)
    total_cost_bps = sum(t["cost_bps"] for t in trades)

    # ── Step 7: persist (only if dry_run=False) ────────────────────────────────
    if not dry_run:
        _write_rebalance_to_db(
            rebalance_date=rebalance_date,
            trades=trades,
            new_positions=new_positions_list,
        )
        logger.info(
            "execute_rebalance: persisted %d trades, %d positions for %s",
            len(trades), len(new_positions_list), rebalance_date,
        )

    return {
        "trades":          trades,
        "new_positions":   new_positions_list,
        "total_cost_bps":  round(total_cost_bps, 2),
        "turnover":        round(turnover, 4),
        "regime":          regime_label,
        "n_long":          portfolio.n_long,
        "n_short":         portfolio.n_short,
        "warnings":        warnings_out,
    }


def _write_rebalance_to_db(
    rebalance_date: datetime.date,
    trades:         list[dict],
    new_positions:  list[dict],
) -> None:
    with SessionFactory() as session:
        # Upsert positions: delete old snapshot for this date, re-insert
        session.query(SimulatedPosition).filter(
            SimulatedPosition.snapshot_date == rebalance_date
        ).delete()

        for pos in new_positions:
            session.add(SimulatedPosition(
                snapshot_date  = rebalance_date,
                sector         = pos["sector"],
                ticker         = pos["ticker"],
                target_weight  = pos["target_weight"],
                actual_weight  = pos["actual_weight"],
                entry_price    = pos.get("entry_price"),
                regime_label   = pos.get("regime_label"),
                signal_tsmom   = pos.get("signal_tsmom"),
                shares_held    = pos.get("shares_held"),
                cost_basis     = pos.get("cost_basis"),
                position_value = pos.get("position_value"),
            ))

        for t in trades:
            session.add(SimulatedTrade(
                trade_date     = t["trade_date"],
                sector         = t["sector"],
                ticker         = t["ticker"],
                action         = t["action"],
                weight_before  = t["weight_before"],
                weight_after   = t["weight_after"],
                weight_delta   = t["weight_delta"],
                cost_bps       = t.get("cost_bps"),
                trigger_reason = t.get("trigger_reason"),
                shares         = t.get("shares"),
                fill_price     = t.get("fill_price"),
                notional       = t.get("notional"),
            ))

        session.commit()


# ── Tactical intra-month position update ──────────────────────────────────────

def apply_tactical_weight_update(
    update_date:         datetime.date,
    sector_adjustments:  dict[str, float] | None = None,
    new_entries:         list[dict]        | None = None,
    nav:                 float | None = None,
) -> None:
    """
    Write an intra-month SimulatedPosition snapshot reflecting tactical changes.

    Called by _patrol_daily_tactical() for Layer-2 auto-executions so that
    get_current_positions() immediately returns the updated state.

    Args:
        update_date        : today's date (becomes the new snapshot_date)
        sector_adjustments : {sector: new_actual_weight} — overwrites weight for
                             existing positions (e.g. regime compress at 0.5×)
        new_entries        : list of dicts with keys matching SimulatedPosition
                             fields (sector, ticker, target_weight, actual_weight,
                             entry_price, signal_tsmom, shares_held, …)
        nav                : portfolio NAV for shares calculation; reads DB if None
    """
    _nav = nav if nav is not None else _get_nav()

    with SessionFactory() as session:
        # Load latest positions as mutable dicts
        latest = (
            session.query(SimulatedPosition.snapshot_date)
            .order_by(SimulatedPosition.snapshot_date.desc())
            .first()
        )
        existing: list[SimulatedPosition] = []
        if latest:
            existing = (
                session.query(SimulatedPosition)
                .filter(SimulatedPosition.snapshot_date == latest[0])
                .all()
            )

        # Build sector → dict map from existing positions
        pos_map: dict[str, dict] = {}
        for p in existing:
            pos_map[p.sector] = {
                "snapshot_date": update_date,
                "sector":        p.sector,
                "ticker":        p.ticker,
                "target_weight": p.target_weight,
                "actual_weight": p.actual_weight,
                "entry_price":   p.entry_price,
                "regime_label":  p.regime_label,
                "signal_tsmom":  p.signal_tsmom,
                "shares_held":   p.shares_held,
                "cost_basis":    p.cost_basis,
                "position_value": p.position_value,
                "track":         getattr(p, "track", "main"),
            }

        # Apply weight adjustments (e.g. regime compress)
        if sector_adjustments:
            for sec, new_w in sector_adjustments.items():
                if sec in pos_map:
                    pos_map[sec]["actual_weight"] = new_w
                    pos_map[sec]["target_weight"] = new_w
                    if _nav and new_w > 0:
                        pos_map[sec]["position_value"] = new_w * _nav

        # Add new tactical entries
        if new_entries:
            for entry in new_entries:
                sec = entry["sector"]
                pos_map[sec] = {
                    "snapshot_date": update_date,
                    **entry,
                }

        # Upsert snapshot for update_date
        session.query(SimulatedPosition).filter(
            SimulatedPosition.snapshot_date == update_date
        ).delete()

        for p in pos_map.values():
            session.add(SimulatedPosition(
                snapshot_date  = update_date,
                sector         = p["sector"],
                ticker         = p.get("ticker", ""),
                target_weight  = p.get("target_weight"),
                actual_weight  = p.get("actual_weight"),
                entry_price    = p.get("entry_price"),
                regime_label   = p.get("regime_label"),
                signal_tsmom   = p.get("signal_tsmom"),
                shares_held    = p.get("shares_held"),
                cost_basis     = p.get("cost_basis"),
                position_value = p.get("position_value"),
                track          = p.get("track", "main"),
            ))
        session.commit()

    logger.info(
        "apply_tactical_weight_update: %d positions written for %s "
        "(adjustments=%s, new_entries=%d)",
        len(pos_map),
        update_date,
        list(sector_adjustments.keys()) if sector_adjustments else [],
        len(new_entries) if new_entries else 0,
    )


# ── Monthly return attribution ─────────────────────────────────────────────────

def record_monthly_return(return_month: datetime.date) -> dict:
    """
    Record return attribution for a completed month.
    Called at the start of the following month.

    Steps:
      1. Load SimulatedPosition snapshot for return_month
      2. Fetch each sector ETF's price return for that calendar month
      3. Compute contribution = actual_weight × sector_return
      4. Write SimulatedMonthlyReturn rows

    Args:
        return_month: first day of the month whose returns to record
                      (e.g. datetime.date(2026, 3, 1) for March 2026)

    Returns:
        {
          "return_month":    date,
          "total_return":    float,
          "contributions":   dict[sector → float],
          "regime_label":    str,
          "n_profitable":    int,
          "n_losing":        int,
        }
    """
    # ── Find the position snapshot that was active during return_month ─────────
    # The snapshot_date should be at/before the start of return_month
    month_start = return_month.replace(day=1)
    month_end   = (month_start.replace(month=month_start.month % 12 + 1, day=1)
                   if month_start.month < 12
                   else month_start.replace(year=month_start.year + 1, month=1, day=1))

    with SessionFactory() as session:
        snap_row = (
            session.query(SimulatedPosition.snapshot_date)
            .filter(SimulatedPosition.snapshot_date < month_end)
            .order_by(SimulatedPosition.snapshot_date.desc())
            .first()
        )
        if snap_row is None:
            return {
                "return_month": return_month,
                "total_return": 0.0,
                "contributions": {},
                "regime_label": "unknown",
                "n_profitable": 0,
                "n_losing": 0,
                "error": "No position snapshot found before this month",
            }

        snapshot_date = snap_row[0]
        positions = (
            session.query(SimulatedPosition)
            .filter(SimulatedPosition.snapshot_date == snapshot_date)
            .all()
        )

    if not positions:
        return {
            "return_month": return_month,
            "total_return": 0.0,
            "contributions": {},
            "regime_label": "unknown",
            "n_profitable": 0,
            "n_losing": 0,
            "error": "Empty position snapshot",
        }

    regime_label = positions[0].regime_label or "unknown"

    # ── Fetch ETF returns for the month ────────────────────────────────────────
    tickers = list({p.ticker for p in positions if p.ticker})
    try:
        px = yf.download(
            tickers,
            start=str(month_start),
            end=str(month_end),
            progress=False,
            auto_adjust=True,
        )
        if not px.empty:
            if isinstance(px.columns, pd.MultiIndex):
                closes = px["Close"]
            else:
                closes = px[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            closes = pd.DataFrame()
    except Exception as e:
        logger.warning("record_monthly_return: yfinance fetch failed: %s", e)
        closes = pd.DataFrame()

    # Compute monthly returns: (last close - first close) / first close
    sector_returns: dict[str, float] = {}
    for ticker in tickers:
        if closes.empty or ticker not in closes.columns:
            sector_returns[ticker] = None
            continue
        series = closes[ticker].dropna()
        if len(series) < 2:
            sector_returns[ticker] = None
        else:
            sector_returns[ticker] = float(series.iloc[-1] / series.iloc[0] - 1)

    # ── Compute contributions and write to DB ──────────────────────────────────
    contributions: dict[str, float] = {}
    total_return = 0.0
    n_profitable = 0
    n_losing     = 0

    rows_to_write = []
    for pos in positions:
        weight  = pos.actual_weight if pos.actual_weight is not None else pos.target_weight
        ret     = sector_returns.get(pos.ticker)
        contrib = (weight * ret) if ret is not None else None
        is_prof = (contrib > 0) if contrib is not None else None

        contributions[pos.sector] = contrib
        if contrib is not None:
            total_return += contrib
            if contrib > 0:
                n_profitable += 1
            else:
                n_losing += 1

        rows_to_write.append(SimulatedMonthlyReturn(
            return_month=month_start,
            sector=pos.sector,
            weight_held=weight,
            sector_return=ret,
            contribution=contrib,
            regime_label=regime_label,
            is_profitable=is_prof,
        ))

    # Upsert: remove existing rows for this month, re-insert
    with SessionFactory() as session:
        session.query(SimulatedMonthlyReturn).filter(
            SimulatedMonthlyReturn.return_month == month_start
        ).delete()
        for row in rows_to_write:
            session.add(row)
        session.commit()

    return {
        "return_month":  return_month,
        "total_return":  round(total_return, 6),
        "contributions": contributions,
        "regime_label":  regime_label,
        "n_profitable":  n_profitable,
        "n_losing":      n_losing,
    }


# ── History readers for UI ─────────────────────────────────────────────────────

def load_all_monthly_returns() -> pd.DataFrame:
    """Load all SimulatedMonthlyReturn rows as a DataFrame."""
    with SessionFactory() as session:
        rows = session.query(SimulatedMonthlyReturn).order_by(
            SimulatedMonthlyReturn.return_month
        ).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "return_month": r.return_month,
            "sector":       r.sector,
            "weight_held":  r.weight_held,
            "sector_return": r.sector_return,
            "contribution": r.contribution,
            "regime_label": r.regime_label,
            "is_profitable": r.is_profitable,
        }
        for r in rows
    ])


def load_trade_history(limit: int = 200) -> pd.DataFrame:
    """Load recent SimulatedTrade rows as a DataFrame."""
    with SessionFactory() as session:
        rows = (
            session.query(SimulatedTrade)
            .order_by(SimulatedTrade.trade_date.desc())
            .limit(limit)
            .all()
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "trade_date":     r.trade_date,
            "sector":         r.sector,
            "ticker":         r.ticker,
            "action":         r.action,
            "weight_before":  r.weight_before,
            "weight_after":   r.weight_after,
            "weight_delta":   r.weight_delta,
            "cost_bps":       r.cost_bps,
            "trigger_reason": r.trigger_reason,
            "shares":         r.shares,
            "fill_price":     r.fill_price,
            "notional":       r.notional,
        }
        for r in rows
    ])
