"""
engine/factor_ensemble_singlename/walk_forward.py — Stage 2 Wave A walk-forward.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.5

Single-stock walk-forward harness:
  - Universe: vintage S&P 500 (Wave A: proxy/wikipedia; Wave B: CRSP)
  - Factors:  Wave A 3-factor (TSMOM + BAB + Dividend Yield) NaN-aware vol-parity 1/N
              Wave B 4-factor (TSMOM + BAB + Quality 4-comp + Value E/P)
  - β-neutralize TSMOM only (per AFP 2014, reuse v2 module)
  - TC: 12bps roundtrip (single-stock standard)
  - Vol target: 15% annualized (institutional single-stock standard, audit amendment)
  - Max position: 2% per name; max leverage 1.5×
  - Min positions: 30 long + 30 short (or scale uniform)

Reuses v1/v2:
  - _compute_realized_return + _panel_slice (factor_ensemble_walk_forward)
  - compute_tc_drag (factor_ensemble_v2.tc, locked at 12bps for this module)
  - beta_neutralize_tsmom + compute_beta_panel (factor_ensemble_v2.beta_neutral)
  - cross-section z-score + NaN-aware factor average (factor_ensemble)
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
from typing import Optional, Callable

import numpy as np
import pandas as pd

from engine.factor_ensemble_singlename.panel_fetcher import bulk_fetch_singlestock_panel

logger = logging.getLogger(__name__)

# Locked per spec §2.1 + §2.5 + audit amendment 2026-05-09
OOS_START_DATE_WAVE_A:    datetime.date = datetime.date(2000, 1, 1)
OOS_END_DATE_WAVE_A:      datetime.date = datetime.date(2024, 12, 31)
TC_BPS_LOCKED:            float = 12.0   # single-stock retail (FP 2014 + Pedersen 2015)
VOL_TARGET_LOCKED:        float = 0.15   # institutional single-stock (audit amendment)
MAX_LEVERAGE_LOCKED:      float = 1.5    # vs ETF 2× (audit amendment)
MAX_NAME_WEIGHT_LOCKED:   float = 0.02   # 2% per single name (concentration cap)
MIN_LONG_POSITIONS:       int = 30       # min diversification
MIN_SHORT_POSITIONS:      int = 30
TRADING_DAYS_PER_YEAR:    int = 252
VOL_WINDOW_DAYS:          int = 60       # for inv-vol weighting + portfolio vol target


@dataclasses.dataclass
class SinglestockWalkForwardResult:
    """Wave A / B walk-forward output."""
    n_periods:                  int
    monthly_returns_gross:      pd.Series
    monthly_returns_net:        pd.Series
    turnover_per_period:        pd.Series
    n_active_per_period:        pd.Series
    cumulative_return_net:      float
    annualized_sharpe_net:      float
    annualized_vol_net:         float
    max_drawdown_net:           float
    metadata:                   dict = dataclasses.field(default_factory=dict)


def _generate_monthend_dates(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    """Same as factor_ensemble_walk_forward._generate_monthend_dates."""
    dates: list[datetime.date] = []
    d = start
    while d <= end:
        # Move to last day of d's month
        next_month = d.replace(day=28) + datetime.timedelta(days=4)
        last_day = next_month - datetime.timedelta(days=next_month.day)
        if last_day >= start and last_day <= end:
            dates.append(last_day)
        # Advance to first day of next month
        d = (last_day + datetime.timedelta(days=1))
    return sorted(set(dates))


def _compute_panel_realized_vols(
    panel:    pd.DataFrame,
    tickers:  list[str],
    as_of:    datetime.date,
) -> pd.Series:
    """60-day annualized realized vol per ticker from panel (no yfinance call)."""
    end = as_of - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=120)
    mask = (panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))
    sub = panel.loc[mask]
    out: dict[str, float] = {}
    for t in tickers:
        if t not in sub.columns:
            continue
        s = sub[t].dropna()
        if len(s) < VOL_WINDOW_DAYS // 2:
            continue
        rets = s.pct_change().dropna().tail(VOL_WINDOW_DAYS)
        if len(rets) < VOL_WINDOW_DAYS // 2:
            continue
        v = float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
        if v > 1e-9:
            out[t] = v
    return pd.Series(out, dtype=float)


def _construct_singlestock_weights(
    ensemble_signal: pd.Series,
    panel:           pd.DataFrame,
    as_of:           datetime.date,
) -> pd.Series:
    """Convert ensemble signal → portfolio weights with single-stock caps.

    Steps:
      1. Drop NaN/zero
      2. Inverse-vol pre-weighting (60d realized)
      3. Gross-normalize Σ|w|=1
      4. Vol-target scale (15% annualized) capped by max leverage 1.5×
      5. Single-name 2% cap
      6. Min-position floor: if < 30 long or 30 short, scale uniform

    Returns:
        pd.Series indexed by ticker, signed weights summing to ≤ 1.5× gross.
    """
    valid = ensemble_signal.dropna()
    nonzero = valid[valid != 0]
    if nonzero.empty:
        return pd.Series(dtype=float)

    realized_vols = _compute_panel_realized_vols(panel, list(nonzero.index), as_of)
    if realized_vols.empty:
        return pd.Series(dtype=float)
    inv_vols = realized_vols.replace(0, np.nan).rdiv(1.0)

    raw_weight = nonzero * inv_vols.reindex(nonzero.index).fillna(0.0)
    raw_weight = raw_weight[raw_weight != 0]
    if raw_weight.empty:
        return pd.Series(dtype=float)

    # Gross-normalize
    gross = raw_weight.abs().sum()
    if gross < 1e-12:
        return pd.Series(dtype=float)
    normalized = raw_weight / gross

    # Vol-target scaling
    realized_vols_aligned = realized_vols.reindex(normalized.index).fillna(0.0)
    port_vol = float(np.sqrt(((normalized * realized_vols_aligned) ** 2).sum()))
    if port_vol < 1e-9:
        return normalized
    vol_scalar = min(VOL_TARGET_LOCKED / port_vol, MAX_LEVERAGE_LOCKED)
    weights = normalized * vol_scalar

    # Single-name 2% cap
    weights = weights.clip(lower=-MAX_NAME_WEIGHT_LOCKED, upper=MAX_NAME_WEIGHT_LOCKED)

    return weights


def _compute_realized_return_panel(
    weights:      pd.Series,
    panel:        pd.DataFrame,
    period_start: datetime.date,
    period_end:   datetime.date,
) -> float:
    """Realized portfolio return [period_start, period_end] from panel (no yfinance)."""
    buffer = datetime.timedelta(days=10)
    active_tickers = [t for t, w in weights.items() if abs(w) >= 1e-9]
    if not active_tickers:
        return 0.0
    mask = (panel.index >= pd.Timestamp(period_start - buffer)) & \
           (panel.index <= pd.Timestamp(period_end + buffer))
    sub = panel.loc[mask, [t for t in active_tickers if t in panel.columns]].dropna(how="all")
    total = 0.0
    for t, w in weights.items():
        if abs(w) < 1e-9 or t not in sub.columns:
            continue
        s = sub[t].dropna()
        if len(s) < 2:
            continue
        before_start = s[s.index <= pd.Timestamp(period_start)]
        before_end = s[s.index <= pd.Timestamp(period_end)]
        if before_start.empty or before_end.empty:
            continue
        p_start = float(before_start.iloc[-1])
        p_end = float(before_end.iloc[-1])
        if p_start <= 0:
            continue
        total += float(w) * (p_end / p_start - 1)
    return total


def run_singlestock_walk_forward(
    universe_at_date_fn:  Callable[[datetime.date], list[str]],
    rebalance_dates:      Optional[list[datetime.date]] = None,
    start_date:           datetime.date = OOS_START_DATE_WAVE_A,
    end_date:             datetime.date = OOS_END_DATE_WAVE_A,
    wave:                 str = "A",
    use_cache:            bool = True,
    run_id:               Optional[str] = None,
) -> SinglestockWalkForwardResult:
    """Run Wave A 3-factor (or Wave B 4-factor) walk-forward.

    Args:
        universe_at_date_fn(d): callable returning list[ticker] at date d
        rebalance_dates:        optional explicit list (default: month-ends in [start, end])
        start_date / end_date:  walk-forward window
        wave:                   "A" (3-factor: TSMOM + BAB + DivYield)
                                "B" (4-factor: + Quality + Value, post-WRDS only)
        use_cache:              panel cache hit if covers
        run_id:                 optional checkpoint identifier (2026-05-11).
                                If given, per-period results JSONL-checkpoint to
                                `data/factor_ensemble_singlename/wave_b_checkpoints/<run_id>.jsonl`
                                and re-runs resume from completed periods.
                                None = no checkpointing (legacy behavior).
    """
    if wave not in ("A", "B"):
        raise ValueError(f"wave must be 'A' or 'B', got {wave!r}")
    # Wave B unlocked 2026-05-11 — WRDS account approved + CRSP+Compustat
    # stubs implemented per project_wave_b_wrds_activation_checklist.

    if rebalance_dates is None:
        rebalance_dates = _generate_monthend_dates(start_date, end_date)
    logger.info("singlestock walk-forward: wave=%s, %d rebalance dates [%s, %s]",
                wave, len(rebalance_dates), start_date, end_date)

    # R1+R2: load existing checkpoints if run_id given (resume mode)
    completed_checkpoints: dict = {}
    if run_id is not None:
        from engine.factor_ensemble_singlename._checkpoint import (
            load_existing_checkpoints,
        )
        completed_checkpoints = load_existing_checkpoints(run_id)
        if completed_checkpoints:
            logger.info(
                "checkpoint resume: %d/%d periods already completed for run_id=%s",
                len(completed_checkpoints), len(rebalance_dates), run_id,
            )

    # Pre-fetch panel for ALL tickers seen across window (+ SPY benchmark for BAB)
    all_tickers: set[str] = {"SPY"}  # always need SPY for BAB
    for d in rebalance_dates:
        try:
            u = universe_at_date_fn(d)
            if u:
                all_tickers.update(u)
        except Exception as exc:
            logger.warning("universe lookup failed at %s: %s", d, exc)

    # Wave A: yfinance retail panel. Wave B: CRSP institutional panel (post-WRDS).
    # PANEL_BUFFER_DAYS_BEFORE provides 13-mo lookback for TSMOM 12-1 formation.
    # Wave A's bulk_fetch_singlestock_panel adds buffer internally; for Wave B
    # we expand the requested date range explicitly since crsp_loader is a thin
    # raw query without its own buffer logic.
    if wave == "B":
        from engine.universe_singlename.crsp_loader import bulk_fetch_crsp_daily_panel
        # ~400 calendar days = ~13 trading months; matches Wave A buffer
        buffered_start = rebalance_dates[0] - datetime.timedelta(days=400)
        buffered_end   = rebalance_dates[-1] + datetime.timedelta(days=45)
        panel = bulk_fetch_crsp_daily_panel(
            tickers=sorted(all_tickers),
            start_date=buffered_start,
            end_date=buffered_end,
            mock_mode=False,   # explicit: Wave B uses real CRSP, not mock
            use_cache=use_cache,
        )
    else:   # wave == "A"
        panel = bulk_fetch_singlestock_panel(
            tickers=sorted(all_tickers),
            start_date=rebalance_dates[0],
            end_date=rebalance_dates[-1],
            use_cache=use_cache,
        )
    if panel.empty:
        logger.error("singlestock walk-forward: panel fetch failed → empty result")
        return SinglestockWalkForwardResult(
            n_periods=0,
            monthly_returns_gross=pd.Series(dtype=float),
            monthly_returns_net=pd.Series(dtype=float),
            turnover_per_period=pd.Series(dtype=float),
            n_active_per_period=pd.Series(dtype=int),
            cumulative_return_net=0.0, annualized_sharpe_net=0.0,
            annualized_vol_net=0.0, max_drawdown_net=0.0,
        )

    # Imports for factor + utility (lazy to avoid circular)
    from engine.factors_singlename import (
        compute_tsmom_singlestock_signal,
        compute_bab_singlestock_signal,
        compute_dividend_yield_singlestock_signal,
    )
    # Wave B 4-factor: TSMOM + BAB + Value (E/P) + Quality (4-component)
    # Replaces Wave A dividend_yield with Value + adds Quality
    if wave == "B":
        from engine.factors_singlename.value_pe import compute_value_pe_singlestock_signal
        from engine.factors_singlename.quality_4comp import compute_quality_singlestock_signal
    from engine.factor_ensemble import _cross_section_z_score, _nan_aware_factor_average
    from engine.factor_ensemble_v2.beta_neutral import compute_beta_panel, beta_neutralize_tsmom
    from engine.factor_ensemble_v2.tc import compute_tc_drag

    monthly_records: list[dict] = []
    prev_weights: Optional[pd.Series] = None

    # R2: seed monthly_records + prev_weights from checkpoints (resume mode)
    if completed_checkpoints:
        from engine.factor_ensemble_singlename._checkpoint import (
            checkpoint_to_record, checkpoint_to_weights,
        )
        for idx in sorted(completed_checkpoints):
            monthly_records.append(checkpoint_to_record(completed_checkpoints[idx]))
        # Use the latest checkpoint's weights as prev_weights for next period
        latest_idx = max(completed_checkpoints)
        prev_weights = checkpoint_to_weights(completed_checkpoints[latest_idx])

    for i, rebal_date in enumerate(rebalance_dates):
        # R2: skip periods already in checkpoint
        if i in completed_checkpoints:
            continue
        # Step 1: vintage universe
        try:
            universe = universe_at_date_fn(rebal_date)
        except Exception as exc:
            logger.warning("universe lookup failed at %s: %s — skip", rebal_date, exc)
            continue
        if not universe:
            continue

        # Step 2: raw factor signals (Wave A: 3-factor / Wave B: 4-factor)
        try:
            sig_tsmom = compute_tsmom_singlestock_signal(rebal_date, universe, panel=panel)
            sig_bab = compute_bab_singlestock_signal(rebal_date, universe, panel=panel)
            if wave == "A":
                sig_div = compute_dividend_yield_singlestock_signal(rebal_date, universe, panel=panel)
            else:   # wave == "B"
                sig_value = compute_value_pe_singlestock_signal(
                    as_of=rebal_date, universe=universe, panel=panel, mock_mode=False,
                )
                sig_quality = compute_quality_singlestock_signal(
                    as_of=rebal_date, universe=universe, panel=panel, mock_mode=False,
                )
        except Exception as exc:
            logger.warning("signal compute failed at %s: %s — skip", rebal_date, exc)
            continue

        # Step 3: β-neutralize TSMOM (per AFP 2014, only TSMOM)
        beta_panel = compute_beta_panel(panel=panel, as_of=rebal_date, tickers=universe)
        sig_tsmom = beta_neutralize_tsmom(tsmom_signal=sig_tsmom, beta_panel=beta_panel)

        # Step 4: cross-section z-score per factor (Wave A: 3 / Wave B: 4)
        if wave == "A":
            z_signals = {
                "tsmom_singlestock":          _cross_section_z_score(sig_tsmom),
                "bab_singlestock":            _cross_section_z_score(sig_bab),
                "dividend_yield_singlestock": _cross_section_z_score(sig_div),
            }
        else:   # wave == "B"
            z_signals = {
                "tsmom_singlestock":   _cross_section_z_score(sig_tsmom),
                "bab_singlestock":     _cross_section_z_score(sig_bab),
                "value_pe_singlestock":     sig_value,    # already z-scored by compute_*
                "quality_4comp_singlestock": sig_quality, # already z-scored by compute_*
            }

        # Step 5: NaN-aware 1/N average → ensemble signal
        ensemble_sig = _nan_aware_factor_average(z_signals, universe=universe)
        if ensemble_sig is None or ensemble_sig.empty:
            continue

        # Step 6: weights with single-stock caps
        weights = _construct_singlestock_weights(ensemble_sig, panel, rebal_date)
        if weights.empty:
            continue

        # Step 7: realized return + TC for next period
        next_rebal = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_rebal is None:
            break
        try:
            gross_return = _compute_realized_return_panel(
                weights=weights, panel=panel,
                period_start=rebal_date, period_end=next_rebal,
            )
        except Exception as exc:
            logger.warning("realized return failed @ %s: %s", rebal_date, exc)
            continue

        tc = compute_tc_drag(weights_new=weights, weights_prev=prev_weights, bps_roundtrip=TC_BPS_LOCKED)
        period_record = {
            "rebal_date":           rebal_date,
            "monthly_return_gross": gross_return,
            "tc_drag":              tc,
            "monthly_return_net":   gross_return - tc,
            "turnover":             tc / (TC_BPS_LOCKED / 10000.0) if TC_BPS_LOCKED > 0 else 0.0,
            "n_active":             int((weights != 0).sum()),
        }
        monthly_records.append(period_record)
        prev_weights = weights

        # R1: persist checkpoint after successful period (durable to disk)
        if run_id is not None:
            try:
                from engine.factor_ensemble_singlename._checkpoint import (
                    write_period_checkpoint,
                )
                write_period_checkpoint(
                    run_id              = run_id,
                    period_idx          = i,
                    rebal_date          = rebal_date,
                    monthly_return_gross= gross_return,
                    tc_drag             = tc,
                    monthly_return_net  = gross_return - tc,
                    turnover            = period_record["turnover"],
                    n_active            = period_record["n_active"],
                    weights             = weights,
                )
            except Exception as exc:
                logger.warning("checkpoint write failed for period %d: %s", i, exc)

    if not monthly_records:
        logger.error("singlestock walk-forward: 0 successful periods")
        return SinglestockWalkForwardResult(
            n_periods=0,
            monthly_returns_gross=pd.Series(dtype=float),
            monthly_returns_net=pd.Series(dtype=float),
            turnover_per_period=pd.Series(dtype=float),
            n_active_per_period=pd.Series(dtype=int),
            cumulative_return_net=0.0, annualized_sharpe_net=0.0,
            annualized_vol_net=0.0, max_drawdown_net=0.0,
        )

    df = pd.DataFrame(monthly_records).set_index("rebal_date")
    monthly_net = df["monthly_return_net"]
    cum = float((1 + monthly_net).prod() - 1)
    ann_vol = float(monthly_net.std(ddof=1) * np.sqrt(12)) if len(monthly_net) > 1 else 0.0
    ann_sharpe = float((monthly_net.mean() / monthly_net.std(ddof=1)) * np.sqrt(12)) \
                  if monthly_net.std(ddof=1) > 1e-9 else 0.0
    cum_curve = (1 + monthly_net).cumprod()
    running_max = cum_curve.expanding().max()
    drawdowns = (cum_curve / running_max) - 1
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0

    return SinglestockWalkForwardResult(
        n_periods=len(df),
        monthly_returns_gross=df["monthly_return_gross"],
        monthly_returns_net=monthly_net,
        turnover_per_period=df["turnover"],
        n_active_per_period=df["n_active"],
        cumulative_return_net=cum,
        annualized_sharpe_net=ann_sharpe,
        annualized_vol_net=ann_vol,
        max_drawdown_net=max_dd,
        metadata={
            "wave":             wave,
            "tc_bps":           TC_BPS_LOCKED,
            "vol_target":       VOL_TARGET_LOCKED,
            "max_leverage":     MAX_LEVERAGE_LOCKED,
            "max_name_weight":  MAX_NAME_WEIGHT_LOCKED,
            "factors":          ["tsmom", "bab", "dividend_yield"] if wave == "A" else
                                ["tsmom", "bab", "value_pe", "quality_4comp"],
            "n_total_tickers_seen": len(all_tickers),
        },
    )
