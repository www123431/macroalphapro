"""
engine/factor_ensemble_crypto/walk_forward.py — Path N walk-forward engine.

Per spec id=71 hash 48db143d §2.5:
  - 2018-01-01 to 2026-05-13 window
  - Month-end UTC rebalance
  - 50% capital each asset, dollar-neutral within-asset (signal × 0.5)
  - Hold 1 month, no intra-month adjustments
  - 25 bp roundtrip TC on leg flips

Output:
  - monthly_returns: pd.Series indexed by month-end, values = net portfolio return
  - signal_panel: pd.DataFrame of signals per asset per rebalance
  - position_changes: pd.Series count of leg flips per month
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import Optional

import pandas as pd

from engine.factor_ensemble_crypto.data_loader import (
    UNIVERSE_LOCKED,
    WINDOW_START_LOCKED,
    WINDOW_END_LOCKED,
    get_month_end_utc_dates,
    load_crypto_panel,
)
from engine.factor_ensemble_crypto.signal import (
    LOOKBACK_MONTHS_LOCKED,
    SKIP_MONTHS_LOCKED,
    compute_tsmom_signal_panel,
)
from engine.factor_ensemble_crypto.tc import (
    TC_BPS_PER_EVENT_LOCKED,
    apply_tc_to_returns,
)

logger = logging.getLogger(__name__)

# Spec metadata (constants)
SPEC_ID:   int = 71
SLEEVE_ID: str = "crypto_btc_eth"


@dataclasses.dataclass(frozen=True)
class WalkForwardResult:
    monthly_gross_returns: pd.Series
    monthly_net_returns:   pd.Series
    signal_panel:          pd.DataFrame
    position_panel:        pd.DataFrame
    position_changes:      pd.Series
    rebalance_dates:       list


def run_walk_forward(
    universe:    tuple[str, ...] = UNIVERSE_LOCKED,
    window_start: datetime.date  = WINDOW_START_LOCKED,
    window_end:   datetime.date  = WINDOW_END_LOCKED,
    daily_prices: Optional[pd.DataFrame] = None,
    use_cache:   bool = True,
) -> WalkForwardResult:
    """
    Run the full walk-forward backtest. Returns aggregated time series + diagnostic panels.

    Args:
        universe: tuple of yfinance tickers (default LOCKED BTC + ETH)
        window_start / window_end: backtest window (LOCKED at spec)
        daily_prices: optional pre-loaded panel (for tests; default = fetch via data_loader)
        use_cache: if True (default), use parquet cache where available

    Returns:
        WalkForwardResult with all backtest output.
    """
    # ── 1. Load daily prices ──────────────────────────────────────────
    if daily_prices is None:
        daily_prices = load_crypto_panel(
            universe=universe, window_start=window_start, window_end=window_end,
            use_cache=use_cache,
        )
    if daily_prices.empty:
        raise RuntimeError("Path N walk-forward: empty price panel")

    # ── 2. Build rebalance schedule ──────────────────────────────────
    rebalance_dates = get_month_end_utc_dates(window_start, window_end)
    if len(rebalance_dates) < 90:
        logger.warning(
            "Path N walk_forward: only %d month-ends in window (spec §4.3 kill-gate "
            "requires ≥90); proceeding but verdict may be underpowered",
            len(rebalance_dates),
        )

    # ── 3. Compute signal at each rebalance date ─────────────────────
    signal_panel = compute_tsmom_signal_panel(
        daily_prices, rebalance_dates,
        lookback_months=LOOKBACK_MONTHS_LOCKED,
        skip_months=SKIP_MONTHS_LOCKED,
    )

    # ── 4. Translate signals to positions (50% capital weight each, signed) ─
    # When signal is None (insufficient lookback), position = 0
    position_panel = signal_panel.copy()
    for col in position_panel.columns:
        position_panel[col] = position_panel[col].apply(
            lambda s: 0.5 * s if s is not None and s in (-1, 0, 1) else 0.0
        )
    position_panel = position_panel.astype(float)

    # ── 5. Compute monthly gross portfolio return ────────────────────
    # For each rebalance date t, position taken at t held until next rebalance t+1.
    # Monthly gross return = sum over assets of (position[s, t] × asset_return[s, t→t+1])
    px_date_indexed = daily_prices.copy()
    px_date_indexed.index = pd.to_datetime(px_date_indexed.index).date

    monthly_gross: list[float] = []
    monthly_index: list = []
    for i in range(len(rebalance_dates) - 1):
        t_curr = rebalance_dates[i]
        t_next = rebalance_dates[i + 1]
        # Use the price AT each rebalance date (last available trading day ≤ t)
        from engine.factor_ensemble_crypto.signal import _last_date_at_or_before
        p_curr_date = _last_date_at_or_before(px_date_indexed.index, t_curr)
        p_next_date = _last_date_at_or_before(px_date_indexed.index, t_next)
        if p_curr_date is None or p_next_date is None:
            continue
        ret = 0.0
        for asset in daily_prices.columns:
            pos = position_panel.loc[pd.Timestamp(t_curr), asset]
            try:
                p_curr = float(px_date_indexed.loc[p_curr_date, asset])
                p_next = float(px_date_indexed.loc[p_next_date, asset])
                if p_curr > 0:
                    asset_ret = (p_next / p_curr) - 1.0
                else:
                    asset_ret = 0.0
            except Exception:
                asset_ret = 0.0
            ret += pos * asset_ret
        monthly_gross.append(ret)
        monthly_index.append(pd.Timestamp(t_next))

    monthly_gross_returns = pd.Series(monthly_gross, index=pd.DatetimeIndex(monthly_index),
                                       name="monthly_gross_return")

    # ── 6. Compute position changes for TC application ───────────────
    # A "flip" = sign change between consecutive rebalance positions on an asset.
    # TC charged on month t = number of legs flipped at the start of month t.
    position_changes: list[int] = []
    for i in range(1, len(rebalance_dates)):
        n_flips = 0
        for asset in daily_prices.columns:
            prev = position_panel.loc[pd.Timestamp(rebalance_dates[i - 1]), asset]
            curr = position_panel.loc[pd.Timestamp(rebalance_dates[i]), asset]
            # flip occurs when sign changes (including 0 ↔ ±0.5)
            if (prev > 0 and curr <= 0) or (prev < 0 and curr >= 0) or (prev == 0 and curr != 0):
                n_flips += 1
        position_changes.append(n_flips)
    # Align to monthly_gross_returns index (positions taken at start of period i+1
    # incur cost recorded on the period's return)
    position_changes_series = pd.Series(
        position_changes,
        index=monthly_gross_returns.index[:len(position_changes)],
        name="n_legs_flipped",
    )
    # Reindex to match if lengths drift
    position_changes_series = position_changes_series.reindex(monthly_gross_returns.index).fillna(0)

    # ── 7. Apply TC to get net returns ───────────────────────────────
    monthly_net_returns = apply_tc_to_returns(
        monthly_gross_returns, position_changes_series, TC_BPS_PER_EVENT_LOCKED,
    )

    return WalkForwardResult(
        monthly_gross_returns = monthly_gross_returns,
        monthly_net_returns   = monthly_net_returns,
        signal_panel          = signal_panel,
        position_panel        = position_panel,
        position_changes      = position_changes_series,
        rebalance_dates       = rebalance_dates,
    )
