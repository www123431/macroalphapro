"""
engine/factor_ensemble_v2/beta_neutral.py — Beta-neutralization for TSMOM only.

Pre-registration: docs/spec_factor_ensemble_v2_robust.md §2.3

Locked rationale: per AFP 2014 QMJ implementation, factors that are NOT
intentional beta-bets get neutralized. BAB IS a beta tilt by design → no
neutralize. Carry-eq via dividend yield correlates with value (slight neg
beta) → preserve original signal. Quality is NaN walk-forward → moot.
Therefore ONLY TSMOM is beta-neutralized.

Mechanism: at each rebalance date, compute per-ticker β against SPY using
60-day window. For TSMOM long set L and short set S:
  long_beta_total  = Σ β_i for i in L
  short_beta_total = Σ β_i for i in S
Scale shorts by (long_beta_total / short_beta_total) → net portfolio β ≈ 0.

If insufficient β data (<60 trading days history) → TSMOM signal → 0 for
that period (logged).
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.3 — only TSMOM β-neutralized
BETA_NEUTRAL_FACTORS_LOCKED: tuple[str, ...] = ("tsmom",)

# Window matches portfolio.py + walk-forward harness (60 trading days)
BETA_WINDOW_DAYS: int = 60
BETA_BENCHMARK_TICKER: str = "SPY"


def compute_beta_panel(
    panel:    pd.DataFrame,
    as_of:    datetime.date,
    tickers:  list[str],
    benchmark: str = BETA_BENCHMARK_TICKER,
) -> pd.Series:
    """Compute per-ticker β against benchmark over 60-day window ending at as_of-1.

    Returns:
        pd.Series indexed by ticker, β value (NaN for tickers with insufficient history).
    """
    if panel is None or panel.empty:
        return pd.Series(dtype=float, index=tickers)
    if benchmark not in panel.columns:
        logger.warning("compute_beta_panel: benchmark %s not in panel — returning NaN", benchmark)
        return pd.Series(np.nan, index=tickers, dtype=float)

    end = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=120)  # buffer for non-trading days

    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub = panel.loc[mask].dropna(how="all")
    if sub.empty or len(sub) < BETA_WINDOW_DAYS:
        return pd.Series(np.nan, index=tickers, dtype=float)

    bench_returns = sub[benchmark].pct_change().dropna().tail(BETA_WINDOW_DAYS)
    if len(bench_returns) < BETA_WINDOW_DAYS // 2:
        return pd.Series(np.nan, index=tickers, dtype=float)

    bench_var = float(bench_returns.var(ddof=0))
    if bench_var <= 1e-12:
        return pd.Series(np.nan, index=tickers, dtype=float)

    out: dict[str, float] = {}
    for ticker in tickers:
        if ticker not in sub.columns:
            out[ticker] = float("nan")
            continue
        ticker_returns = sub[ticker].pct_change().dropna().tail(BETA_WINDOW_DAYS)
        if len(ticker_returns) < BETA_WINDOW_DAYS // 2:
            out[ticker] = float("nan")
            continue
        # Align on common dates
        common = ticker_returns.index.intersection(bench_returns.index)
        if len(common) < BETA_WINDOW_DAYS // 2:
            out[ticker] = float("nan")
            continue
        cov = float(np.cov(
            ticker_returns.loc[common].values,
            bench_returns.loc[common].values,
            ddof=0,
        )[0, 1])
        out[ticker] = cov / bench_var
    return pd.Series(out, dtype=float)


def beta_neutralize_tsmom(
    tsmom_signal: pd.Series,
    beta_panel:   pd.Series,
) -> pd.Series:
    """Apply beta-neutralization to TSMOM signal per spec §2.3.

    Input: tsmom_signal indexed by ticker, values are signed signal magnitudes
    (e.g. ±1 from get_signal_dataframe.tsmom column).

    Mechanism:
      - long_idx  = tickers with tsmom > 0
      - short_idx = tickers with tsmom < 0
      - long_beta_total  = Σ β_i × signal_i for long set
      - short_beta_total = Σ β_i × |signal_i| for short set
      - If both > 0: scale shorts by (long_beta_total / short_beta_total)
      - If insufficient β data (< 50% of nonzero tickers have valid β) → return all-zero

    Returns:
        pd.Series same index, β-neutralized signed signal.
        Empty/all-zero on insufficient β data → caller MUST check and log.
    """
    if tsmom_signal is None or tsmom_signal.empty:
        return pd.Series(dtype=float)

    sig = tsmom_signal.copy().astype(float)
    if beta_panel is None or beta_panel.empty:
        # No β data → conservatively return all-zero (signal lost this period)
        logger.warning("beta_neutralize_tsmom: empty beta_panel — TSMOM signal zeroed this period")
        return pd.Series(0.0, index=sig.index, dtype=float)

    # Align β with signal index
    beta_aligned = beta_panel.reindex(sig.index)

    # Identify nonzero signal tickers + valid β
    nonzero_mask = sig.abs() > 1e-9
    valid_beta = beta_aligned.notna() & nonzero_mask
    n_nonzero = int(nonzero_mask.sum())
    n_valid = int(valid_beta.sum())
    if n_nonzero == 0:
        return sig
    if n_valid < n_nonzero // 2:
        logger.warning(
            "beta_neutralize_tsmom: only %d/%d nonzero tickers have valid β — TSMOM zeroed",
            n_valid, n_nonzero,
        )
        return pd.Series(0.0, index=sig.index, dtype=float)

    # Compute long/short β totals (using only tickers with valid β)
    long_mask  = (sig > 0) & valid_beta
    short_mask = (sig < 0) & valid_beta
    long_beta_total  = float((sig[long_mask]  * beta_aligned[long_mask]).sum())
    short_beta_total = float((sig[short_mask].abs() * beta_aligned[short_mask]).sum())

    if long_beta_total <= 1e-9 or short_beta_total <= 1e-9:
        # Single-direction (all long or all short) — can't neutralize, return as-is
        # but log: per spec §2.3 fallback is "return as-is" not zero (preserves directional info)
        return sig

    # Scale shorts to match long β total
    short_scale = long_beta_total / short_beta_total
    out = sig.copy()
    out.loc[short_mask] = sig.loc[short_mask] * short_scale  # short signals are negative; scale magnitudes
    # Tickers with NaN β: keep original signal (don't zero) — they don't contribute to net β estimate
    return out
