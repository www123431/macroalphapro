"""engine.agents.strengthener.templates.event_drift_pead — Post-Earnings
Announcement Drift (Bernard-Thomas 1989 canonical).

Tests claims of the form:
  "After earnings announcement event E, returns drift for K days/weeks"
  "Post-earnings-announcement drift exists and is exploitable"
  "Standardized unexpected earnings (SUE) predicts post-announcement
   returns"

Scope (narrow MVP)
==================
  signal_kind  : event_drift
  universe     : us_equities_pead
  data         : Compustat fundq (smallcap subset, 2011-2025) +
                 CRSP MSF monthly returns 1990-2024 + CCM gvkey↔permno link
  strategy     : monthly decile long-short on SUE-ranked recent
                 announcers

SUE definition (seasonal random walk per Bernard-Thomas 1989)
=============================================================
For each firm-quarter (gvkey, fyearq, fqtr):
  earnings_surprise_t = epspxq_t - epspxq_{t-4}    (year-over-year EPS change)
  sigma_t             = std(surprise) over trailing 8 quarters
  SUE_t               = earnings_surprise_t / sigma_t

Firms with <8 quarters of history get NULL SUE → excluded.

Portfolio formation (monthly)
=============================
For each month-end M:
  candidates = firms whose most-recent rdq ∈ [M-60d, M-2d]
               AND have a finite SUE
  Sort candidates by SUE.
  Long top decile (highest positive surprise).
  Short bottom decile (most negative surprise).
  Equal-weight within decile.
  Hold for 1 month → portfolio return for month M+1.

Why this window: 2-day buffer after announcement avoids event-day
return contamination (price reaction is announcement itself; PEAD
is the post-event drift). 60-day lookback captures announcements
that happened recently enough to still be in the drift window.

Verdict
=======
Sharpe + NW-t (HAC lag 6) on monthly long-short PnL.
Multi-testing thresholds via BUG-3 verdict_thresholds.
GREEN requires NW-t ≥ HLZ-floor / Bonferroni-scaled threshold AND
positive mean PnL (otherwise short-PEAD which has no theoretical basis).

M2 anchor (Chordia-Goyal-Sadka 2009): post-2000 PEAD is WEAKER than
Bernard-Thomas 1989's 1974-1986 sample. Anchor: mean monthly PnL >
0 AND |NW-t| > 1.0 in the 2011-2024 sub-sample. Bernard-Thomas
reported quarterly spread ~4-5% in their sample; modern data gives
~0.5-1.5% monthly spreads, Sharpe ~0.3-0.7.

Known limitations (defer to v2)
================================
- Data is smallcap subset (3356 gvkeys, 2011-2025), not full
  Russell 3000. May overstate or understate PEAD depending on
  small-vs-large differential decay (Chordia-Goyal-Sadka argues
  smaller-cap stocks STILL show PEAD; larger don't).
- No SUE z-score scaling across firms (just raw sort within month)
- No transaction cost adjustment beyond simple cost_model='13bp_per_rt'
- Monthly granularity loses precision vs daily drift window
- No IBES analyst-consensus SUE (uses naive seasonal random walk
  per Bernard-Thomas 1989, NOT Foster-Olsen-Shevlin 1984 which
  uses analyst-revision SUE)
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_2026-06-13"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FUNDQ_PATH = _REPO_ROOT / "data" / "cache" / "_smallcap_fundq.parquet"
_CRSP_MSF_PATH = _REPO_ROOT / "data" / "cache" / "_crsp_msf_long_history.parquet"
_CCM_LINK_PATH = _REPO_ROOT / "data" / "cache" / "_crsp_ccm_link.parquet"

_MIN_OBS_MONTHS    = 36       # need ≥ 3 years of valid drift portfolio months
_MIN_DECILE_FIRMS  = 5        # require ≥ 5 firms per decile per month
_DRIFT_LOOKBACK_D  = 60       # announcements eligible in this many days back
_DRIFT_BUFFER_D    = 2        # exclude announcements in past N days (event-day buffer)
_MIN_SUE_HIST_Q    = 8        # need ≥ 8 quarters to compute sigma(surprise)
_MAX_HOLD_MONTHS_FORWARD = 1  # hold for 1 month forward (the simplest)


# ── Data loading ──────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_fundq() -> Optional[pd.DataFrame]:
    if not _FUNDQ_PATH.is_file():
        return None
    df = pd.read_parquet(_FUNDQ_PATH)
    # Schema: gvkey / datadate / rdq / fyearq / fqtr / epspxq / cshoq
    df = df.dropna(subset=["gvkey", "rdq", "epspxq", "fyearq", "fqtr"])
    df["rdq"]      = pd.to_datetime(df["rdq"])
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


@lru_cache(maxsize=1)
def _load_msf() -> Optional[pd.DataFrame]:
    if not _CRSP_MSF_PATH.is_file():
        return None
    df = pd.read_parquet(_CRSP_MSF_PATH)
    if not {"permno", "date", "ret"}.issubset(df.columns):
        return None
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["permno", "date"])
    return df[["permno", "date", "ret"]]


@lru_cache(maxsize=1)
def _load_ccm_link() -> Optional[pd.DataFrame]:
    if not _CCM_LINK_PATH.is_file():
        return None
    df = pd.read_parquet(_CCM_LINK_PATH)
    df["linkdt"]     = pd.to_datetime(df["linkdt"],     errors="coerce")
    df["linkenddt"]  = pd.to_datetime(df["linkenddt"],  errors="coerce")
    # Use primary links only to dedup
    if "linkprim" in df.columns:
        df = df[df["linkprim"].isin(["P", "C"])]
    return df


# ── SUE computation (Bernard-Thomas 1989 seasonal random walk) ────


def _compute_sue_panel(fundq: pd.DataFrame) -> pd.DataFrame:
    """Compute SUE for every firm-quarter that has ≥8 quarters of history.

    Returns DataFrame with columns: gvkey, rdq, sue.
    """
    df = fundq.copy().sort_values(["gvkey", "fyearq", "fqtr"])
    # 4-quarter lagged EPS (same firm, same fqtr, previous fyearq)
    df["epsq_lag4"] = (
        df.groupby(["gvkey", "fqtr"])["epspxq"].shift(1)
    )
    df["surprise"] = df["epspxq"] - df["epsq_lag4"]

    # Trailing-8-quarter std of surprise
    df["sigma_8q"] = (
        df.groupby("gvkey")["surprise"]
          .transform(lambda s: s.shift(1).rolling(8, min_periods=_MIN_SUE_HIST_Q).std())
    )
    # SUE: handle zero-or-NaN sigma → NaN
    df["sue"] = np.where(
        (df["sigma_8q"] > 0) & np.isfinite(df["sigma_8q"]) & np.isfinite(df["surprise"]),
        df["surprise"] / df["sigma_8q"],
        np.nan,
    )
    out = df.dropna(subset=["sue", "rdq"])[["gvkey", "rdq", "sue"]].copy()
    return out


# ── Portfolio formation ──────────────────────────────────────────


def _attach_permno_via_ccm(
    sue_panel: pd.DataFrame, ccm: pd.DataFrame,
) -> pd.DataFrame:
    """Vectorized gvkey + rdq → permno mapping via merge_asof + window filter.

    CCM link windows: a (gvkey, permno) pair is valid for rdq if
    linkdt <= rdq <= linkenddt (linkenddt NaT means open-ended).

    Approach: sort sue_panel by rdq, sort ccm by linkdt, merge_asof
    with by=gvkey + direction='backward' to find the most recent
    linkdt that started before rdq, then filter rows where linkenddt
    is NaT or >= rdq. Output one row per (gvkey, rdq) at most.
    """
    cc = ccm[["gvkey", "permno", "linkdt", "linkenddt"]].copy()
    cc["linkdt"] = cc["linkdt"].fillna(pd.Timestamp("1900-01-01"))
    cc["gvkey"] = cc["gvkey"].astype(str)
    # merge_asof needs the left's on-key globally sorted AND each by-group
    # right side sorted by on-key. Easiest = sort by linkdt globally.
    cc = cc.sort_values("linkdt").reset_index(drop=True)

    sp = sue_panel.copy()
    sp["gvkey"] = sp["gvkey"].astype(str)
    sp = sp.sort_values("rdq").reset_index(drop=True)

    merged = pd.merge_asof(
        sp, cc,
        left_on   = "rdq",
        right_on  = "linkdt",
        by        = "gvkey",
        direction = "backward",
    )
    # Filter: linkenddt must be NaT or >= rdq
    open_ended = merged["linkenddt"].isna()
    in_window  = merged["linkenddt"] >= merged["rdq"]
    merged = merged[open_ended | in_window]
    merged = merged.dropna(subset=["permno"])
    merged["permno"] = merged["permno"].astype(int)
    # Drop duplicates: keep one permno per (gvkey, rdq) — first match
    merged = merged.drop_duplicates(subset=["gvkey", "rdq"], keep="first")
    return merged[["gvkey", "rdq", "sue", "permno"]]


def _build_monthly_long_short_returns(
    sue_panel: pd.DataFrame, msf: pd.DataFrame, ccm: pd.DataFrame,
) -> Optional[pd.Series]:
    """Construct monthly long-short PEAD return series.

    For each month-end M:
      eligible = firms with rdq ∈ [M-DRIFT_LOOKBACK_D, M-DRIFT_BUFFER_D]
                 (most recent announcement per firm)
      decile sort by SUE, long top decile / short bottom decile, EW
      portfolio return = next month's mean(long) - mean(short)
    """
    # Vectorized permno attach (replaces per-row apply loop)
    sp = _attach_permno_via_ccm(sue_panel, ccm)
    if sp.empty:
        return None

    # Pre-pivot MSF to dict[(permno, (year, month))] → ret for O(1) lookup
    msf2 = msf.copy()
    msf2["yr"] = msf2["date"].dt.year
    msf2["mo"] = msf2["date"].dt.month
    msf2 = msf2.dropna(subset=["ret"])
    # Build the lookup dict
    msf_lookup: dict[tuple[int, int, int], float] = dict(
        ((int(r["permno"]), int(r["yr"]), int(r["mo"])), float(r["ret"]))
        for r in msf2[["permno", "yr", "mo", "ret"]].to_dict("records")
    )

    sp = sp.sort_values("rdq")
    start = sp["rdq"].min() + pd.DateOffset(months=2)
    end = sp["rdq"].max() + pd.DateOffset(months=2)
    month_ends = pd.date_range(start, end, freq="ME")

    monthly_pnl = []
    for me in month_ends:
        lookback_start = me - pd.Timedelta(days=_DRIFT_LOOKBACK_D)
        lookback_end   = me - pd.Timedelta(days=_DRIFT_BUFFER_D)
        eligible = sp[
            (sp["rdq"] >= lookback_start) &
            (sp["rdq"] <= lookback_end)
        ]
        if eligible.empty:
            continue
        latest = eligible.sort_values("rdq").drop_duplicates(
            subset=["gvkey"], keep="last",
        )
        if len(latest) < _MIN_DECILE_FIRMS * 10:
            continue

        q_lo = latest["sue"].quantile(0.10)
        q_hi = latest["sue"].quantile(0.90)
        long_firms  = latest[latest["sue"] >= q_hi]
        short_firms = latest[latest["sue"] <= q_lo]
        if len(long_firms) < _MIN_DECILE_FIRMS or len(short_firms) < _MIN_DECILE_FIRMS:
            continue

        next_me = (me + pd.DateOffset(months=1)).normalize()
        yr, mo = next_me.year, next_me.month

        long_rets  = [
            r for r in (msf_lookup.get((int(p), yr, mo)) for p in long_firms["permno"])
            if r is not None and math.isfinite(r)
        ]
        short_rets = [
            r for r in (msf_lookup.get((int(p), yr, mo)) for p in short_firms["permno"])
            if r is not None and math.isfinite(r)
        ]
        if len(long_rets) < _MIN_DECILE_FIRMS or len(short_rets) < _MIN_DECILE_FIRMS:
            continue

        ls_return = float(np.mean(long_rets) - np.mean(short_rets))
        monthly_pnl.append((next_me, ls_return))

    if not monthly_pnl:
        return None
    s = pd.Series(
        [r for _, r in monthly_pnl],
        index=pd.DatetimeIndex([d for d, _ in monthly_pnl]),
    )
    return s.dropna()


# ── Verdict classification ────────────────────────────────────────


def _classify_verdict(
    nw_t: float, mean_pnl: float, n_trials: int,
) -> tuple[str, str]:
    if not math.isfinite(nw_t):
        return "INSUFFICIENT_HISTORY", "NW-t non-finite"
    if mean_pnl <= 0:
        return "RED", (
            f"mean monthly PnL {mean_pnl:.5f} ≤ 0; PEAD has no theoretical "
            f"basis on the short side (short = LOW SUE → underperform)"
        )

    try:
        from engine.research.verdict_thresholds import (
            t_green_threshold, t_marginal_threshold,
        )
        t_g = t_green_threshold(n_trials)
        t_m = t_marginal_threshold(n_trials)
    except Exception:
        t_g, t_m = 1.96, 1.65

    if nw_t >= t_g:
        return "GREEN", f"NW-t={nw_t:.2f} >= {t_g:.2f}; positive PEAD confirmed"
    if nw_t >= t_m:
        return "MARGINAL", f"NW-t={nw_t:.2f} in [{t_m:.2f}, {t_g:.2f})"
    return "RED", f"NW-t={nw_t:.2f} < {t_m:.2f}; PEAD not statistically significant"


# ── Entry point ──────────────────────────────────────────────────


def template_event_drift_pead(spec: FactorSpec):
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    fundq = _load_fundq()
    msf   = _load_msf()
    ccm   = _load_ccm_link()
    if fundq is None or msf is None or ccm is None:
        missing = [
            n for n, v in [("fundq", fundq), ("msf", msf), ("ccm", ccm)]
            if v is None
        ]
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"missing required caches: {missing}",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    sue_panel = _compute_sue_panel(fundq)
    if sue_panel.empty:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = "SUE panel empty after 8-quarter history filter",
            metrics          = {"n_obs_months": 0},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    monthly_pnl = _build_monthly_long_short_returns(sue_panel, msf, ccm)
    if monthly_pnl is None or len(monthly_pnl) < _MIN_OBS_MONTHS:
        n = 0 if monthly_pnl is None else len(monthly_pnl)
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"only {n} valid monthly portfolio obs "
                                  f"(min {_MIN_OBS_MONTHS})"),
            metrics          = {"n_obs_months": n},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    n_obs    = len(monthly_pnl)
    mean_pnl = float(monthly_pnl.mean())
    std_pnl  = float(monthly_pnl.std(ddof=1))
    if std_pnl <= 0 or not math.isfinite(std_pnl):
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = "PnL series degenerate variance",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    sharpe = mean_pnl / std_pnl * math.sqrt(12.0)

    try:
        import statsmodels.api as sm
        x = np.ones(n_obs)
        ols = sm.OLS(monthly_pnl.values, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        nw_t = float(ols.tvalues[0])
    except Exception:
        nw_t = mean_pnl / (std_pnl / math.sqrt(n_obs))

    # n_trials for multi-testing
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

    # MaxDD
    cum = monthly_pnl.cumsum()
    running_max = cum.cummax()
    max_dd = float((cum - running_max).min())

    summary = (
        f"PEAD long-short ({monthly_pnl.index[0].strftime('%Y-%m')}~"
        f"{monthly_pnl.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"mean_pnl={mean_pnl*100:+.2f}%/mo, Sharpe={sharpe:+.2f}, "
        f"NW-t={nw_t:+.2f}, MaxDD={max_dd*100:+.1f}% → {verdict}. {note}"
    )

    pnl_df = pd.DataFrame({
        "pnl_gross":    monthly_pnl,
        "pnl_net_13bp": monthly_pnl - 0.0013,   # rough 1-side 13bp turnover cost
        "turnover":     pd.Series(1.0, index=monthly_pnl.index),  # ~100%/mo
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "mean_pnl_monthly":     mean_pnl,
            "std_pnl_monthly":      std_pnl,
            "sharpe_gross":         sharpe,
            "nw_t_gross":           nw_t,
            "max_drawdown":         max_dd,
            "n_obs_months":         n_obs,
            "n_trials_at_dispatch": n_trials,
        },
        artifacts        = {
            "pnl_series_df":   pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
