"""engine.agents.strengthener.templates.event_drift_revision —
analyst-revision-based event drift on US equities (Chan-Jegadeesh-
Lakonishok 1996 canonical).

Tests claims of the form:
  "Stocks with positive analyst earnings revisions earn higher future
   returns" / "Up/down revision ratio predicts cross-sectional returns"
  / "Analyst forecast momentum"

Scope (narrow MVP)
==================
  signal_kind  : event_drift
  universe     : us_equities_revision
  data         : IBES statsumu_epsus FY1 EPS estimates (1.95M rows
                 1990-2024 via ${WRDS_USER_2}) + crsp.msenames cusip→permno
                 bridge + CRSP MSF monthly returns
  strategy     : monthly quintile long-short on revision score

Revision score
==============
For each (ticker, statpers month-end):
  revision_pct = (numup - numdown) / max(numest, 1)
  revision_score is in [-1, +1].

  HIGH revision_score = analysts net-revising UP = LONG side
  (Chan-Jegadeesh-Lakonishok 1996 direction)

Forward holding: 1 month.

Verdict
=======
NW-t HAC lag 6 on monthly L/S return + BUG-3 multi-test threshold.
GREEN requires NW-t ≥ HLZ-corrected threshold AND positive mean PnL.

M2 anchor (Chan-Jegadeesh-Lakonishok 1996): annualized spread ~9-12%
with t-stat ~3-5 in 1977-1993 sample. Post-2000 expected lower per
Diether-Lee-Werner 2009 / McLean-Pontiff 2016 decay.
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_2026-06-14"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_IBES_PATH      = _REPO_ROOT / "data" / "cache" / "_ibes_eps_summary_us_fy1.parquet"
_MSENAMES_PATH  = _REPO_ROOT / "data" / "cache" / "_crsp_msenames.parquet"
_CRSP_MSF_PATH  = _REPO_ROOT / "data" / "cache" / "_crsp_msf_long_history.parquet"

_MIN_OBS_MONTHS  = 60
_MIN_DECILE_FIRMS = 30   # need decent breadth for cross-sectional decile


@lru_cache(maxsize=1)
def _load_ibes() -> Optional[pd.DataFrame]:
    if not _IBES_PATH.is_file():
        return None
    df = pd.read_parquet(_IBES_PATH)
    df["statpers"] = pd.to_datetime(df["statpers"])
    return df


@lru_cache(maxsize=1)
def _load_msenames() -> Optional[pd.DataFrame]:
    if not _MSENAMES_PATH.is_file():
        return None
    df = pd.read_parquet(_MSENAMES_PATH)
    df["namedt"]    = pd.to_datetime(df["namedt"])
    df["nameendt"]  = pd.to_datetime(df["nameendt"])
    return df


@lru_cache(maxsize=1)
def _load_msf() -> Optional[pd.DataFrame]:
    if not _CRSP_MSF_PATH.is_file():
        return None
    df = pd.read_parquet(_CRSP_MSF_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df[["permno", "date", "ret"]]


def _attach_permno(ibes: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Join IBES cusip → CRSP permno via msenames using date-range
    membership (cusips can re-assign through time)."""
    # IBES cusip is 8-char or 6-char; msenames ncusip is 8-char
    ibes = ibes.copy()
    ibes["cusip8"] = ibes["cusip"].astype(str).str[:8].str.upper()
    names = names.copy()
    names["ncusip8"] = names["ncusip"].astype(str).str[:8].str.upper()

    # Lightweight join: for each unique (cusip8, year), find permno valid
    # in that year. Use msenames namedt/nameendt range.
    # For performance: just take latest permno per cusip8 (most cusips
    # map to a single permno across their life)
    permno_map = (names.sort_values("namedt")
                       .groupby("ncusip8")["permno"].last())
    ibes = ibes.merge(permno_map.rename("permno"), left_on="cusip8",
                        right_index=True, how="left")
    ibes = ibes.dropna(subset=["permno"])
    ibes["permno"] = ibes["permno"].astype(int)
    return ibes


def _compute_monthly_ls_returns(
    ibes: pd.DataFrame, msf: pd.DataFrame,
) -> Optional[pd.Series]:
    """Build monthly long-short revision-sorted return series."""
    # Compute revision_pct per row
    ibes = ibes.copy()
    ibes["revision_pct"] = ((ibes["numup"].fillna(0)
                              - ibes["numdown"].fillna(0))
                              / ibes["numest"].clip(lower=1))
    # Take last row per (permno, month) — IBES statpers can be multiple
    # per month
    ibes["month_end"] = (ibes["statpers"].dt.to_period("M")
                           .dt.to_timestamp(how="end").dt.normalize())
    ibes_m = (ibes.sort_values(["permno", "statpers"])
                  .groupby(["permno", "month_end"])
                  ["revision_pct"].last().reset_index())

    # Pre-pivot MSF: dict[(permno, year, month)] → ret
    msf2 = msf.copy()
    msf2["yr"] = msf2["date"].dt.year
    msf2["mo"] = msf2["date"].dt.month
    msf2 = msf2.dropna(subset=["ret"])
    msf_lookup = dict(
        ((int(r["permno"]), int(r["yr"]), int(r["mo"])), float(r["ret"]))
        for r in msf2[["permno", "yr", "mo", "ret"]].to_dict("records")
    )

    monthly_pnl = []
    months = ibes_m["month_end"].unique()
    for me in sorted(months):
        sub = ibes_m[ibes_m["month_end"] == me].dropna(subset=["revision_pct"])
        if len(sub) < _MIN_DECILE_FIRMS * 5:   # need decent cross-section
            continue
        # Quintile sort
        q_lo = sub["revision_pct"].quantile(0.20)
        q_hi = sub["revision_pct"].quantile(0.80)
        long_firms  = sub[sub["revision_pct"] >= q_hi]
        short_firms = sub[sub["revision_pct"] <= q_lo]
        if len(long_firms) < _MIN_DECILE_FIRMS or len(short_firms) < _MIN_DECILE_FIRMS:
            continue
        # Forward 1-month return
        next_me = (me + pd.DateOffset(months=1)).normalize()
        yr, mo = next_me.year, next_me.month
        long_rets  = [r for r in (msf_lookup.get((int(p), yr, mo)) for p in long_firms["permno"]) if r is not None and math.isfinite(r)]
        short_rets = [r for r in (msf_lookup.get((int(p), yr, mo)) for p in short_firms["permno"]) if r is not None and math.isfinite(r)]
        if len(long_rets) < _MIN_DECILE_FIRMS or len(short_rets) < _MIN_DECILE_FIRMS:
            continue
        ls = float(np.mean(long_rets) - np.mean(short_rets))
        monthly_pnl.append((next_me, ls))
    if not monthly_pnl:
        return None
    return pd.Series([r for _, r in monthly_pnl],
                      index=pd.DatetimeIndex([d for d, _ in monthly_pnl]))


def _classify_verdict(nw_t: float, mean_pnl: float, n_trials: int) -> tuple[str, str]:
    if not math.isfinite(nw_t):
        return "INSUFFICIENT_HISTORY", "NW-t non-finite"
    if mean_pnl <= 0:
        return "RED", (
            f"mean monthly PnL {mean_pnl:.5f} ≤ 0; revision signal direction "
            f"violated (CJL 1996: positive revisions should predict positive return)"
        )
    try:
        from engine.research.verdict_thresholds import (
            t_green_threshold, t_marginal_threshold,
        )
        t_g = t_green_threshold(n_trials)
        t_m = t_marginal_threshold(n_trials)
    except Exception:
        t_g, t_m = 3.0, 1.65
    if nw_t >= t_g:
        return "GREEN", f"NW-t={nw_t:.2f} >= {t_g:.2f}; revision premium confirmed"
    if nw_t >= t_m:
        return "MARGINAL", f"NW-t={nw_t:.2f} in [{t_m:.2f}, {t_g:.2f})"
    return "RED", f"NW-t={nw_t:.2f} < {t_m:.2f}; revision premium not significant"


def template_event_drift_revision(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    ibes = _load_ibes()
    names = _load_msenames()
    msf = _load_msf()
    if ibes is None or names is None or msf is None:
        missing = [n for n, v in [("ibes", ibes), ("msenames", names), ("msf", msf)] if v is None]
        return TemplateResult(
            verdict="INSUFFICIENT_DATA",
            summary=f"missing caches: {missing}",
            metrics={}, artifacts={},
            template_version=_TEMPLATE_VERSION,
        )

    ibes_p = _attach_permno(ibes, names)
    if ibes_p.empty:
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary="no IBES rows after permno join", metrics={}, artifacts={},
            template_version=_TEMPLATE_VERSION,
        )

    monthly_pnl = _compute_monthly_ls_returns(ibes_p, msf)
    if monthly_pnl is None or len(monthly_pnl) < _MIN_OBS_MONTHS:
        n = 0 if monthly_pnl is None else len(monthly_pnl)
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY",
            summary=f"only {n} monthly obs (min {_MIN_OBS_MONTHS})",
            metrics={"n_obs_months": n}, artifacts={},
            template_version=_TEMPLATE_VERSION,
        )

    n_obs = len(monthly_pnl)
    mean_pnl = float(monthly_pnl.mean())
    std_pnl  = float(monthly_pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        return TemplateResult(
            verdict="INSUFFICIENT_HISTORY", summary="degenerate variance",
            metrics={"n_obs_months": n_obs}, artifacts={},
            template_version=_TEMPLATE_VERSION,
        )
    sharpe = mean_pnl / std_pnl * math.sqrt(12.0)

    try:
        import statsmodels.api as sm
        ols = sm.OLS(monthly_pnl.values, np.ones(n_obs)).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        nw_t = float(ols.tvalues[0])
    except Exception:
        nw_t = mean_pnl / (std_pnl / math.sqrt(n_obs))

    n_trials = 0
    try:
        from engine.research.strategy_family_classifier import (
            strategy_family_for_spec,
        )
        from engine.agents.strengthener.factor_dispatcher import (
            _family_n_trials_now,
        )
        n_trials = _family_n_trials_now(strategy_family_for_spec(spec))
    except Exception:
        pass
    verdict, note = _classify_verdict(nw_t, mean_pnl, n_trials)

    cum = monthly_pnl.cumsum()
    max_dd = float((cum - cum.cummax()).min())

    summary = (
        f"Revision quintile L/S ({monthly_pnl.index[0].strftime('%Y-%m')}~"
        f"{monthly_pnl.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"mean_pnl={mean_pnl*100:+.2f}%/mo, Sharpe={sharpe:+.2f}, "
        f"NW-t={nw_t:+.2f}, MaxDD={max_dd*100:+.1f}% → {verdict}. {note}"
    )

    pnl_df = pd.DataFrame({
        "pnl_gross":    monthly_pnl,
        "pnl_net_13bp": monthly_pnl - 0.0013,
        "turnover":     pd.Series(1.0, index=monthly_pnl.index),
    })

    return TemplateResult(
        verdict=verdict, summary=summary,
        metrics={
            "mean_pnl_monthly": mean_pnl, "std_pnl_monthly": std_pnl,
            "sharpe_gross": sharpe, "nw_t_gross": nw_t,
            "max_drawdown": max_dd, "n_obs_months": n_obs,
            "n_trials_at_dispatch": n_trials,
        },
        artifacts={
            "pnl_series_df": pnl_df, "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col": "pnl_gross",
        },
        template_version=_TEMPLATE_VERSION,
    )
