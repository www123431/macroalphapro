"""
engine/path_c/pead_backtest.py — Path C #1 PEAD walk-forward backtest.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §2.4 + §2.6

Pipeline (deterministic, 0 LLM):
  1. Sprint 2 earnings_panel → firm-quarter inputs
  2. Sprint 3 sue_signal → per-firm-quarter `leg` ∈ {long, short, flat, excluded}
  3. THIS sprint: for each long/short firm-quarter, hold from rdq+1 to rdq+60
     trading days (skipping day 0 per spec §2.4 step 7). Aggregate daily
     long-short returns across all active firms. Apply TC drag.
  4. Sprint 5 verdict → Sharpe / NW t (lag=60) / BHY-FDR / decision

Key functions:
  - trading_day_after(date, n) : NYSE bday arithmetic
  - compute_position_windows(panel) : per-firm-quarter [start, end]
  - compute_daily_long_short_returns(signal_panel, returns_panel) : main aggregation
  - compute_annual_turnover(daily_aggregates) : roundtrip count per year
  - apply_tc_drag(gross_returns, turnover, bps_roundtrip) : net P&L
  - run_walk_forward_pead(...) : orchestrator with per-quarter checkpoint

Reused from existing modules:
  - engine.factor_ensemble_singlename._checkpoint : JSONL resume capability
  - engine.universe_singlename.crsp_loader.bulk_fetch_crsp_daily_panel : daily prices
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.path_c import (
    HOLD_TRADING_DAYS_LOCKED,
    TC_BPS_ROUNDTRIP_LOCKED,
)

logger = logging.getLogger(__name__)


# ── Storage ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_DIR  = _REPO_ROOT / "data" / "path_c"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_WALK_FORWARD_PARQUET    = _DATA_DIR / "walk_forward_pead.parquet"
_CHECKPOINT_DIR_DEFAULT  = _DATA_DIR / "pead_checkpoints"


# ── Public result type ─────────────────────────────────────────────────────
@dataclasses.dataclass
class WalkForwardPeadResult:
    """Output of run_walk_forward_pead.

    `daily_returns` columns:
      - r_long          : daily mean return across active long firms
      - r_short         : daily mean return across active short firms
      - r_long_short    : r_long - r_short (gross daily P&L)
      - r_long_short_net: gross minus TC drag (annualized turnover × bps/10000 / 252)
      - n_long          : count of active long firms that day
      - n_short         : count of active short firms that day
    """
    daily_returns:           pd.DataFrame
    n_quarters_processed:    int
    n_firm_quarters_active:  int     # long + short legs only
    annual_turnover_estimate: float
    tc_bps_roundtrip:        float
    window_start:            datetime.date
    window_end:              datetime.date
    spec_hash_at_run:        Optional[str] = None


# ── Trading day arithmetic ──────────────────────────────────────────────────
def trading_day_after(date: datetime.date, n_days: int) -> datetime.date:
    """Return the date `n_days` trading days after `date` (NYSE bday proxy).

    Uses pandas.bdate_range (Mon-Fri excluding US holidays approximation).
    Exact NYSE-holiday handling differs by ≤5 days/year; impact on Sharpe
    computed over 2400 daily obs is negligible (<0.5%).

    Per spec §2.4 step 7: skip rdq day 0; trading_day_after(rdq, 1) returns
    the FIRST eligible holding day.
    """
    if n_days < 1:
        raise ValueError(f"trading_day_after: n_days must be ≥ 1, got {n_days}")
    days = pd.bdate_range(start=date + datetime.timedelta(days=1), periods=n_days)
    return days[-1].date()


def compute_position_windows(
    signal_panel: pd.DataFrame,
    *,
    hold_trading_days: int = HOLD_TRADING_DAYS_LOCKED,
) -> pd.DataFrame:
    """Per firm-quarter row, add `window_start` and `window_end` (datetime.date).

    window_start = trading_day_after(rdq, 1)   (skips rdq day 0)
    window_end   = trading_day_after(rdq, hold_trading_days)

    Only computed for `leg` in {long, short}; flat/excluded rows get NaT.
    """
    if signal_panel.empty:
        out = signal_panel.copy()
        out["window_start"] = pd.NaT
        out["window_end"]   = pd.NaT
        return out
    required = {"rdq", "leg"}
    missing = required - set(signal_panel.columns)
    if missing:
        raise ValueError(f"compute_position_windows: panel missing columns {missing}")

    out = signal_panel.copy()
    # Object dtype to hold a mix of None and datetime.date without pandas
        # downcast-to-datetime64 warnings.
    window_starts = [None] * len(out)
    window_ends   = [None] * len(out)
    active_mask = out["leg"].isin(["long", "short"]).reset_index(drop=True)
    rdq_series = out["rdq"].reset_index(drop=True)
    for pos in range(len(out)):
        if not active_mask.iloc[pos]:
            continue
        rdq = rdq_series.iloc[pos]
        if hasattr(rdq, "date"):
            rdq = rdq.date()
        if not isinstance(rdq, datetime.date):
            continue
        window_starts[pos] = trading_day_after(rdq, 1)
        window_ends[pos]   = trading_day_after(rdq, hold_trading_days)
    out = out.reset_index(drop=True)
    out["window_start"] = pd.Series(window_starts, dtype="object")
    out["window_end"]   = pd.Series(window_ends, dtype="object")
    return out


# ── Daily long-short aggregation (vectorized) ──────────────────────────────
def compute_daily_long_short_returns(
    signal_panel:  pd.DataFrame,
    returns_panel: pd.DataFrame,
    *,
    hold_trading_days: int = HOLD_TRADING_DAYS_LOCKED,
    ticker_col:        str = "ticker_ibes",
) -> pd.DataFrame:
    """Aggregate per-firm position windows into daily L-S portfolio returns.

    Args:
        signal_panel:   Sprint 3 output (must have ticker_ibes, rdq, leg cols)
        returns_panel:  DataFrame indexed by trading day, columns = tickers,
                        values = **daily returns** (caller MUST apply
                        pct_change() before passing; this function does NOT
                        convert prices to returns — passing prices will
                        silently produce nonsense Sharpe). Must extend
                        ≥60 trading days past the last expected rdq to avoid
                        end-of-window drift truncation (function emits a
                        warning if not).
        hold_trading_days: position lifetime (default = HOLD_TRADING_DAYS_LOCKED = 60)
        ticker_col:     column in signal_panel matching returns_panel column names

    Returns:
        DataFrame indexed by date, columns = [r_long, r_short, r_long_short,
        n_long, n_short].

    Algorithm (vectorized):
      1. Filter signal_panel to leg ∈ {long, short}
      2. Per row, generate list of trading days in [window_start, window_end]
      3. Explode into long-form position book: [date, ticker, leg]
      4. Inner-join position book with returns (already long-form via melt)
      5. Group by (date, leg), compute mean return + count
      6. Pivot to wide form for output
    """
    # Filter to active firm-quarters
    active = signal_panel[signal_panel["leg"].isin(["long", "short"])].copy()
    if active.empty or returns_panel.empty:
        return pd.DataFrame(
            columns=["r_long", "r_short", "r_long_short", "n_long", "n_short"],
            index=pd.DatetimeIndex([], name="date"),
        )

    required_cols = {ticker_col, "rdq", "leg"}
    missing = required_cols - set(active.columns)
    if missing:
        raise ValueError(f"compute_daily_long_short_returns: signal_panel missing {missing}")

    # Compute position window dates per firm-quarter
    if "window_start" not in active.columns or "window_end" not in active.columns:
        active = compute_position_windows(active, hold_trading_days=hold_trading_days)

    # Build position book: for each firm-quarter, explode into 60 daily rows
    position_book_rows = []
    for _, row in active.iterrows():
        ws = row["window_start"]
        we = row["window_end"]
        if pd.isna(ws) or pd.isna(we):
            continue
        window_days = pd.bdate_range(start=ws, end=we)
        ticker = row[ticker_col]
        leg = row["leg"]
        for d in window_days:
            position_book_rows.append({"date": d, "ticker": ticker, "leg": leg})
    if not position_book_rows:
        return pd.DataFrame(
            columns=["r_long", "r_short", "r_long_short", "n_long", "n_short"],
            index=pd.DatetimeIndex([], name="date"),
        )
    book = pd.DataFrame(position_book_rows)
    book["date"] = pd.to_datetime(book["date"]).dt.normalize()
    # Dedupe (date, ticker, leg): a single firm may have two overlapping
    # holds (Q1 + Q2 both top-decile within 60-day overlap window — ~10-15%
    # of firms in any quarter). Without dedupe, mean() in groupby double-
    # counts that firm. L-M 2006 standard is equal-weight per FIRM, not per
    # position-row. Rigor audit fix 2026-05-12 (finding D).
    book = book.drop_duplicates(subset=["date", "ticker", "leg"]).reset_index(drop=True)

    # Coverage check: warn if returns_panel ends before any position window
    # extends. End-of-window drift truncation systematically underweights
    # late-window firms (e.g., 2023Q3 announcements in a 2014-2023 returns
    # panel). Rigor audit warning 2026-05-12 (finding A).
    latest_position_end = (
        pd.to_datetime(active["window_end"].dropna()).max()
        if not active["window_end"].dropna().empty else None
    )
    returns_max = pd.to_datetime(returns_panel.index.max())
    if latest_position_end is not None and returns_max < latest_position_end:
        truncated_days = (latest_position_end - returns_max).days
        logger.warning(
            "returns_panel ends %s but positions extend to %s — %d calendar days "
            "truncated. End-of-window firms have incomplete drift windows. Pass "
            "returns_panel extending ≥60 trading days past last rdq for full hold.",
            returns_max.date(), latest_position_end.date(), truncated_days,
        )

    # Melt returns to long form
    returns_long = (
        returns_panel.reset_index()
        .rename(columns={returns_panel.index.name or "index": "date"})
        .melt(id_vars="date", var_name="ticker", value_name="ret")
        .dropna(subset=["ret"])
    )
    returns_long["date"] = pd.to_datetime(returns_long["date"]).dt.normalize()

    # Inner-join book × returns
    merged = book.merge(returns_long, on=["date", "ticker"], how="inner")
    if merged.empty:
        return pd.DataFrame(
            columns=["r_long", "r_short", "r_long_short", "n_long", "n_short"],
            index=pd.DatetimeIndex([], name="date"),
        )

    # Aggregate per (date, leg)
    grouped = (
        merged.groupby(["date", "leg"])
        .agg(ret=("ret", "mean"), n=("ret", "count"))
        .unstack(level="leg")
    )

    # Build output frame (handle missing leg columns gracefully)
    out = pd.DataFrame(index=grouped.index)
    r_long_col  = grouped.get(("ret", "long"))
    r_short_col = grouped.get(("ret", "short"))
    n_long_col  = grouped.get(("n",   "long"))
    n_short_col = grouped.get(("n",   "short"))
    out["r_long"]  = (r_long_col  if r_long_col  is not None else pd.Series(0.0, index=grouped.index)).fillna(0.0)
    out["r_short"] = (r_short_col if r_short_col is not None else pd.Series(0.0, index=grouped.index)).fillna(0.0)
    out["r_long_short"] = out["r_long"] - out["r_short"]
    out["n_long"]  = (n_long_col  if n_long_col  is not None else pd.Series(0,   index=grouped.index)).fillna(0).astype(int)
    out["n_short"] = (n_short_col if n_short_col is not None else pd.Series(0,   index=grouped.index)).fillna(0).astype(int)

    out.index.name = "date"
    return out.sort_index()


# ── Turnover + TC drag ──────────────────────────────────────────────────────
def compute_annual_turnover(
    daily_aggregates: pd.DataFrame,
    *,
    hold_trading_days: int = HOLD_TRADING_DAYS_LOCKED,
) -> float:
    """Estimate annualized roundtrip turnover.

    Each firm-quarter position enters at rdq+1, exits at rdq+60. So:
      annual_roundtrip_turnover ≈ 252 / hold_trading_days

    For hold=60 → ~4.2 roundtrips/year. This is a structural estimate (not
    derived from `daily_aggregates`; included as argument for API symmetry
    and future per-quarter dynamic turnover).
    """
    if hold_trading_days <= 0:
        return 0.0
    return 252.0 / float(hold_trading_days)


def apply_tc_drag(
    gross_daily_returns: pd.Series,
    *,
    annual_turnover:      float,
    tc_bps_roundtrip:     float = TC_BPS_ROUNDTRIP_LOCKED,
    trading_days_per_year: int  = 252,
) -> pd.Series:
    """Convert gross daily returns to net via uniform TC drag.

    Daily drag = (tc_bps / 10_000) × annual_turnover / trading_days_per_year.
    Subtracted uniformly from each gross daily return.

    Example: hold=60, turnover ≈ 4.2/yr, bps=30 → 30/10000 × 4.2 / 252 ≈
    5.0 × 10^-5 per day ≈ 1.26% per year drag.
    """
    daily_drag = (tc_bps_roundtrip / 10_000.0) * annual_turnover / float(trading_days_per_year)
    return gross_daily_returns - daily_drag


# ── Per-quarter checkpoint helpers (mirrors _checkpoint pattern) ───────────
def _checkpoint_path(run_id: str, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or _CHECKPOINT_DIR_DEFAULT
    base.mkdir(parents=True, exist_ok=True)
    safe_id = run_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    return base / f"{safe_id}.jsonl"


def write_quarter_checkpoint(
    run_id:           str,
    quarter:          str,
    n_long_quarter:   int,
    n_short_quarter:  int,
    n_excluded:       int,
    *,
    base_dir:         Optional[Path] = None,
) -> None:
    """Append one completed quarter to the JSONL checkpoint."""
    path = _checkpoint_path(run_id, base_dir)
    record = {
        "quarter":         quarter,
        "n_long":          int(n_long_quarter),
        "n_short":         int(n_short_quarter),
        "n_excluded":      int(n_excluded),
        "completed_at":    datetime.datetime.utcnow().isoformat() + "Z",
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_quarter_checkpoints(
    run_id:    str,
    *,
    base_dir:  Optional[Path] = None,
) -> list[dict]:
    """Read all completed quarter records from checkpoint JSONL."""
    path = _checkpoint_path(run_id, base_dir)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("checkpoint: skipping malformed line in %s", path)
    return out


# ── Walk-forward orchestrator ──────────────────────────────────────────────
def run_walk_forward_pead(
    *,
    signal_panel:        pd.DataFrame,
    returns_panel:       pd.DataFrame,
    window_start:        datetime.date,
    window_end:          datetime.date,
    hold_trading_days:   int = HOLD_TRADING_DAYS_LOCKED,
    tc_bps_roundtrip:    float = TC_BPS_ROUNDTRIP_LOCKED,
    checkpoint_run_id:   Optional[str] = None,
    checkpoint_base_dir: Optional[Path] = None,
    spec_hash_at_run:    Optional[str] = None,
) -> WalkForwardPeadResult:
    """End-to-end PEAD walk-forward driver.

    Args:
        signal_panel:  Sprint 3 output (build_sue_signal_panel result)
        returns_panel: daily returns (NOT prices); index=date, cols=tickers
        window_start / window_end: backtest window per spec §2.1
        hold_trading_days: locked at 60 per spec §六
        tc_bps_roundtrip:  locked at 30.0 per spec §六
        checkpoint_run_id: if set, write per-quarter JSONL checkpoints

    Returns:
        WalkForwardPeadResult with daily_returns DataFrame + diagnostics.
    """
    # Validate
    if signal_panel.empty:
        logger.warning("run_walk_forward_pead: empty signal_panel — nothing to backtest")
        return WalkForwardPeadResult(
            daily_returns=pd.DataFrame(),
            n_quarters_processed=0,
            n_firm_quarters_active=0,
            annual_turnover_estimate=0.0,
            tc_bps_roundtrip=tc_bps_roundtrip,
            window_start=window_start,
            window_end=window_end,
            spec_hash_at_run=spec_hash_at_run,
        )

    # Compute position windows + daily aggregation
    signal_with_windows = compute_position_windows(
        signal_panel, hold_trading_days=hold_trading_days,
    )
    daily = compute_daily_long_short_returns(
        signal_with_windows, returns_panel,
        hold_trading_days=hold_trading_days,
    )

    # TC drag
    turnover = compute_annual_turnover(daily, hold_trading_days=hold_trading_days)
    daily["r_long_short_net"] = apply_tc_drag(
        daily["r_long_short"],
        annual_turnover=turnover,
        tc_bps_roundtrip=tc_bps_roundtrip,
    )

    # Per-quarter checkpoint (if run_id provided)
    if checkpoint_run_id:
        for quarter, group in signal_panel.groupby("fiscal_yearq"):
            n_long  = int((group["leg"] == "long").sum())
            n_short = int((group["leg"] == "short").sum())
            n_excl  = int((group["leg"] == "excluded").sum())
            write_quarter_checkpoint(
                checkpoint_run_id, quarter,
                n_long_quarter=n_long, n_short_quarter=n_short, n_excluded=n_excl,
                base_dir=checkpoint_base_dir,
            )

    n_active = int(signal_panel["leg"].isin(["long", "short"]).sum())
    n_quarters = int(signal_panel["fiscal_yearq"].nunique())

    return WalkForwardPeadResult(
        daily_returns=daily,
        n_quarters_processed=n_quarters,
        n_firm_quarters_active=n_active,
        annual_turnover_estimate=turnover,
        tc_bps_roundtrip=tc_bps_roundtrip,
        window_start=window_start,
        window_end=window_end,
        spec_hash_at_run=spec_hash_at_run,
    )


# ── Persistence ────────────────────────────────────────────────────────────
def persist_walk_forward_result(
    result:       WalkForwardPeadResult,
    *,
    parquet_path: Optional[Path] = None,
) -> Path:
    """Save daily_returns DataFrame to parquet."""
    path = parquet_path or _WALK_FORWARD_PARQUET
    if result.daily_returns.empty:
        logger.warning("persist_walk_forward_result: empty daily_returns — skipping write")
        return path
    result.daily_returns.to_parquet(path)
    logger.info(
        "walk_forward_pead persisted: %d daily obs → %s",
        len(result.daily_returns), path,
    )
    return path
