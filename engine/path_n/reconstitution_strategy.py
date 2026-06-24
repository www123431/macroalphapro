"""
engine/path_n/reconstitution_strategy.py — S&P 500 add-event Pre-effective Drift strategy.

Pre-registration: docs/spec_path_n_index_reconstitution_drift_v1.md (id=70 hash c92d2c36) §2.3
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Spec §六 LOCKED
PRE_EVENT_DAYS_LOCKED         = 5      # T-5 to T-1 (5 trading days)
TC_BPS_ROUNDTRIP_LOCKED       = 30.0   # single-stock standing rule
NW_LAG_LOCKED                 = 21     # cover 5-day hold cycle + autocorr


@dataclass
class ReconstitutionResult:
    daily_returns:      pd.Series  # net daily strategy returns
    daily_gross:        pd.Series  # gross (pre-TC)
    n_events:           int
    n_active_days:      int
    mean_concurrent_events: float
    annual_turnover:    float
    tc_drag_annual_pct: float
    event_returns:      pd.DataFrame  # per-event T-5 to T-1 returns


def build_add_event_strategy(
    events: pd.DataFrame,
    daily_returns_panel: pd.DataFrame,
    pre_event_days: int = PRE_EVENT_DAYS_LOCKED,
    tc_bps_roundtrip: float = TC_BPS_ROUNDTRIP_LOCKED,
) -> ReconstitutionResult:
    """Build daily strategy series from S&P 500 add events.

    Args:
        events: DataFrame with cols (permno, effective_date, event_type='ADD')
        daily_returns_panel: DataFrame indexed by date, cols = permno, values = daily return
        pre_event_days: T-N to T-1 window (5 per CNS 2004 canonical)
        tc_bps_roundtrip: 30bp single-stock standing rule

    Returns:
        ReconstitutionResult with strategy_returns + diagnostics
    """
    all_dates = daily_returns_panel.index
    add_events = events[events['event_type'] == 'ADD'].copy()
    add_events['effective_date'] = pd.to_datetime(add_events['effective_date'])
    add_events['permno'] = add_events['permno'].astype(int)

    gross = pd.Series(0.0, index=all_dates)
    n_active = pd.Series(0, index=all_dates)
    tc_drag = pd.Series(0.0, index=all_dates)

    event_record = []  # per-event return diagnostics

    for _, evt in add_events.iterrows():
        eff = evt['effective_date']
        permno = evt['permno']
        if permno not in daily_returns_panel.columns:
            continue
        idx_eff = all_dates.searchsorted(eff)
        if idx_eff < pre_event_days + 1 or idx_eff >= len(all_dates):
            continue
        # T-5 to T-1: positions held; daily returns observed T-4 through T-1
        # Entry at close T-5; first return = T-4; last return = T-1 (5 days = T-4..T-0 but T-0 is after exit)
        # Per spec: enter T-5 close, exit T-1 close → daily returns over T-4..T-1 (4 daily returns) for that event

        entry_idx = idx_eff - pre_event_days        # T-5
        exit_idx = idx_eff - 1                       # T-1
        active_idxs = list(range(entry_idx + 1, exit_idx + 1))  # T-4..T-1 (returns observed)

        if len(active_idxs) == 0:
            continue

        # Apply TC at entry (T-5) and exit (T-1) — half-roundtrip each side
        entry_date = all_dates[entry_idx]
        exit_date = all_dates[exit_idx]
        tc_per_side = (tc_bps_roundtrip / 2) / 10000  # 15bp
        tc_drag.loc[entry_date] += tc_per_side
        tc_drag.loc[exit_date] += tc_per_side

        evt_total_ret = 0.0
        for ti in active_idxs:
            d = all_dates[ti]
            r = daily_returns_panel.iloc[ti][permno]
            if pd.notna(r):
                gross.loc[d] += float(r)
                n_active.loc[d] += 1
                evt_total_ret += float(r)

        event_record.append({
            'permno': permno,
            'effective_date': eff,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'event_return': evt_total_ret,
        })

    # Equal-weight aggregation: divide by # active events that day
    gross_ew = gross / n_active.replace(0, np.nan)
    gross_ew = gross_ew.fillna(0)

    # TC normalization: when N events active, each contributes 1/N to portfolio
    # but TC is in units of "% per event entry/exit"; for equal-weight portfolio,
    # TC drag at entry/exit days = tc_per_side × (1/n_active_at_event)
    # Simpler: aggregate tc drag is approximate at 30bp × 24 events / annual ≈ 7.2bp/yr nominal,
    # but capital-weighted is bigger. Use straightforward approximation.
    tc_drag_norm = tc_drag / n_active.replace(0, np.nan).bfill().fillna(1)
    tc_drag_norm = tc_drag_norm.fillna(0)
    net_returns = gross_ew - tc_drag_norm

    n_active_days = int((n_active > 0).sum())
    mean_concurrent = float(n_active[n_active > 0].mean()) if (n_active > 0).any() else 0.0

    # Annual turnover: each event 2 trades (entry + exit)
    n_events = len(event_record)
    years = (all_dates[-1] - all_dates[0]).days / 365.25
    annual_turn_events = n_events / years if years > 0 else 0
    # TC drag annualized
    tc_drag_ann = float(tc_drag_norm.sum() / years * 100) if years > 0 else 0.0

    return ReconstitutionResult(
        daily_returns=net_returns,
        daily_gross=gross_ew,
        n_events=n_events,
        n_active_days=n_active_days,
        mean_concurrent_events=mean_concurrent,
        annual_turnover=annual_turn_events,
        tc_drag_annual_pct=tc_drag_ann,
        event_returns=pd.DataFrame(event_record),
    )
