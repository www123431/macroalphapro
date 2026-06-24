"""engine.agents.strengthener.templates.tsmom_sector_etf — Tier C-2b.

Time-series momentum on the 35-ETF sector universe. First real
template after the C-2a dispatcher skeleton. Chosen for end-to-end
smallness: data layer is intact (DB universe table + _fetch_closes),
sample size is tractable, and the Moskowitz et al. 2012 TSMOM(12,1)
specification is canonical.

Scope (intentionally narrow per piece-by-piece doctrine):
  - signal_kind  : time_series_momentum
  - universe     : us_equities_sector_etf (only — C-2 follow-ups
                    add cross_asset_etf / commodity_futures_27 etc.)
  - lookback     : 52 weeks (hardcoded — adding per-spec parameters
                    requires extending FactorSpec.template_params,
                    a separate spec change)
  - skip         : 4 weeks (canonical skip-the-most-recent-month)
  - rebal        : weekly Friday
  - weighting    : signed signal × vol-target (10% annual per asset)
  - cost         : 13bp per round-trip * weekly turnover

Why NOT wrap engine.b_plus_search.run_single_strategy_weekly:
  fitness check 2026-06-08 found the existing wrapper has 2
  pre-existing bugs (broken import + signature mismatch). Per
  [[feedback-no-fear-of-rework-only-unusable-2026-06-01]] +
  "用不了 = 等于没有", reviving dormant infra is more expensive
  than a clean 250-line self-contained template.

Verdict thresholds — mirror engine.factor_lab.runner._classify_verdict:
  GREEN     : |nw_t_stat| >= 1.96   (5% two-sided)
  MARGINAL  : 1.65 <= |nw_t_stat| < 1.96
  RED       : |nw_t_stat| < 1.65
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants — locked per scope above
# ────────────────────────────────────────────────────────────────────
_LOOKBACK_WEEKS = 52
_SKIP_WEEKS     = 4
_VOL_TARGET     = 0.10        # 10% annual per asset (Moskowitz et al.)
_REBAL_DOW      = "W-FRI"     # weekly Friday close
_TC_BP_PER_RT   = 13.0        # 13 bp per round-trip turnover (sector ETF)
_VOL_LOOKBACK_W = 13          # weeks for realized vol estimation
_MIN_TICKERS    = 10          # if universe shrinks below, refuse

_TEMPLATE_VERSION = "v1.0_2026-06-08"


# ────────────────────────────────────────────────────────────────────
# Verdict thresholds (mirror engine.factor_lab.runner)
# ────────────────────────────────────────────────────────────────────
# L2-1 Phase 2.5: A-class safety constants from _safety_constants
from engine.agents.strengthener._safety_constants import (
    T_GREEN as _T_GREEN,
    T_MARGINAL as _T_MARGINAL,
)


# ────────────────────────────────────────────────────────────────────
# Universe resolver — direct DB read, no b_plus_search dependency
# ────────────────────────────────────────────────────────────────────
def _resolve_sector_etf_universe() -> dict[str, str]:
    """Return {sector_label: ticker} for the 35-ETF sector universe.
    Reads UniverseETF table directly to avoid b_plus_search rot."""
    from engine.memory import SessionFactory
    from engine.universe_manager import UniverseETF
    with SessionFactory() as s:
        rows = (s.query(UniverseETF)
                  .filter(UniverseETF.active == True,    # noqa: E712
                          UniverseETF.batch <= 4)
                  .all())
        return {r.sector: r.ticker for r in rows}


# ────────────────────────────────────────────────────────────────────
# Date range parsing (FactorSpec.date_range = "YYYY-MM:YYYY-MM")
# ────────────────────────────────────────────────────────────────────
def _parse_date_range(s: str) -> tuple[_dt.date, _dt.date]:
    """Inclusive month bounds → (first-of-month, end-of-month). Raises
    ValueError on bad format (defense in depth — extractor already
    pattern-validates, but tests still cover this path)."""
    if ":" not in s:
        raise ValueError(f"date_range must contain ':': {s!r}")
    a, b = s.split(":", 1)
    start = _dt.date.fromisoformat(f"{a.strip()}-01")
    end_ts = pd.Timestamp(f"{b.strip()}-01") + pd.offsets.MonthEnd(0)
    return start, end_ts.date()


# ────────────────────────────────────────────────────────────────────
# Weekly TSMOM signal + backtest
# ────────────────────────────────────────────────────────────────────
def _weekly_friday_resample(closes: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then resample to weekly Friday closes (last
    obs per week). Drops any tickers with insufficient history."""
    closes = closes.ffill(limit=5)
    weekly = closes.resample(_REBAL_DOW).last()
    return weekly.dropna(how="all")


def _tsmom_signal_weekly(
    weekly_closes:    pd.DataFrame,
    lookback_weeks:   int,
    skip_weeks:       int,
) -> pd.DataFrame:
    """For each Friday t, signal[t,i] = sign(close[t-skip] / close[t-skip-lookback] - 1).

    Returns DataFrame aligned to weekly_closes index, values in
    {-1, 0, +1}. First (lookback+skip) rows are NaN by construction.
    """
    horizon = lookback_weeks + skip_weeks
    # Past return: close[t-skip] / close[t-horizon] - 1
    past = weekly_closes.shift(skip_weeks)
    past_baseline = weekly_closes.shift(horizon)
    past_ret = past / past_baseline - 1.0
    return np.sign(past_ret)


def _realized_vol_weekly(
    weekly_closes:  pd.DataFrame,
    vol_lookback_w: int,
) -> pd.DataFrame:
    """Realized weekly return std over a trailing window, annualized.
    Returns DataFrame of annualized vol per ticker per Friday."""
    weekly_ret = weekly_closes.pct_change()
    return (weekly_ret.rolling(vol_lookback_w).std()
              * math.sqrt(52))


def _run_tsmom_backtest(
    weekly_closes:    pd.DataFrame,
    lookback_weeks:   int,
    skip_weeks:       int,
    vol_target:       float,
    tc_bp_per_rt:     float,
) -> tuple[pd.Series, dict]:
    """Run the weekly TSMOM long-short backtest. Returns (pnl_series,
    diagnostics) where pnl_series is gross-of-cost net-of-tc weekly
    returns indexed by Friday closes.

    Equal-risk-contribution sizing per asset: weight_i = sign_i * (
    vol_target / realized_vol_i), capped at ±1 per asset.
    """
    sig = _tsmom_signal_weekly(weekly_closes, lookback_weeks,
                                 skip_weeks)
    vol = _realized_vol_weekly(weekly_closes, _VOL_LOOKBACK_W)
    weekly_ret = weekly_closes.pct_change()

    # Per-asset vol-targeted weight (sized so each asset contributes
    # ~vol_target / sqrt(N) ann vol; we average across active assets)
    raw_weight = sig * (vol_target / vol.replace(0, np.nan))
    raw_weight = raw_weight.clip(lower=-1.0, upper=1.0)
    # Average across active (non-NaN) assets each week
    n_active = raw_weight.notna().sum(axis=1).replace(0, np.nan)
    weight = raw_weight.div(n_active, axis=0).fillna(0.0)

    # Hold weight[t] over week (t, t+1] — earn weight[t] * ret[t+1]
    pnl_gross = (weight.shift(1) * weekly_ret).sum(axis=1)

    # Transaction cost: turnover * tc_bp / 10000
    turnover = (weight - weight.shift(1)).abs().sum(axis=1)
    tc_drag = turnover * (tc_bp_per_rt / 10_000.0)
    pnl_net = (pnl_gross - tc_drag).dropna()

    avg_turnover = float(turnover.dropna().mean())
    avg_n_active = float(n_active.dropna().mean()) if len(
        n_active.dropna()) else 0.0

    return pnl_net, {
        "avg_weekly_turnover": avg_turnover,
        "avg_n_active":         avg_n_active,
        "n_weeks_signaled":     int(weight.abs().sum(axis=1)
                                      .gt(0).sum()),
    }


# ────────────────────────────────────────────────────────────────────
# Verdict + summary
# ────────────────────────────────────────────────────────────────────
def _verdict_from_t(t_stat: float) -> str:
    if not math.isfinite(t_stat):
        return "RED"
    a = abs(t_stat)
    if a >= _T_GREEN:
        return "GREEN"
    if a >= _T_MARGINAL:
        return "MARGINAL"
    return "RED"


# ────────────────────────────────────────────────────────────────────
# Template entry point
# ────────────────────────────────────────────────────────────────────
def template_tsmom_sector_etf(spec: FactorSpec):
    """Tier C-2b template: TSMOM(52, 4) on the 35-ETF sector
    universe. Returns a TemplateResult."""
    # Local import keeps dispatcher / extractor / template cycle free
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # ── 1. Scope guards ────────────────────────────────────────────
    if spec.signal_kind != "time_series_momentum":
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = (f"signal_kind={spec.signal_kind!r} "
                                  "routed to tsmom template by mistake"),
            metrics          = {"misroute": True},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if spec.universe != "us_equities_sector_etf":
        return TemplateResult(
            verdict          = "UNSUPPORTED_UNIVERSE",
            summary          = (f"universe={spec.universe!r} not "
                                  "supported by tsmom_sector_etf "
                                  "(only us_equities_sector_etf)"),
            metrics          = {"unsupported_universe": spec.universe},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 2. Parse date range ────────────────────────────────────────
    try:
        start_date, end_date = _parse_date_range(spec.date_range)
    except ValueError as exc:
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = f"date_range parse failed: {exc}",
            metrics          = {"error": str(exc)},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 3. Resolve universe ────────────────────────────────────────
    universe = _resolve_sector_etf_universe()
    if len(universe) < _MIN_TICKERS:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = (f"universe has {len(universe)} tickers "
                                  f"< _MIN_TICKERS={_MIN_TICKERS}"),
            metrics          = {"n_tickers": len(universe)},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 4. Fetch closes (wider fetch window for lookback warmup) ──
    from engine.signal import _fetch_closes
    fetch_start = (pd.Timestamp(start_date)
                     - pd.Timedelta(weeks=_LOOKBACK_WEEKS + _SKIP_WEEKS
                                      + _VOL_LOOKBACK_W + 4)).date()
    tickers = list(universe.values())
    try:
        closes = _fetch_closes(tickers, fetch_start, end_date)
    except Exception as exc:
        logger.exception("tsmom_sector_etf: _fetch_closes raised")
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = (f"_fetch_closes failed: "
                                  f"{type(exc).__name__}: {exc}"),
            metrics          = {"error": str(exc)[:200]},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if closes is None or closes.empty:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = "_fetch_closes returned empty dataframe",
            metrics          = {"n_tickers_requested": len(tickers)},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 5. Weekly resample + backtest ──────────────────────────────
    weekly_closes = _weekly_friday_resample(closes)
    weekly_closes = weekly_closes.loc[
        weekly_closes.index.to_series().between(
            pd.Timestamp(fetch_start),
            pd.Timestamp(end_date))]

    # L2-1 Phase 4: B-class params from FactorSpec v2 with parity-
    # preserving fallback. Months → weeks conversion 1mo ≈ 4.33w
    # (round to int). When LLM doesn't populate spec fields, falls
    # back to Moskowitz et al. 2012 default TSMOM(12,1)=lookback52/skip4.
    eff_lookback_w = (round(spec.signal_lookback_m * 52 / 12)
                       if spec.signal_lookback_m is not None
                       else _LOOKBACK_WEEKS)
    eff_skip_w     = (round(spec.signal_skip_m * 52 / 12)
                       if spec.signal_skip_m is not None
                       else _SKIP_WEEKS)
    eff_vol_target = (spec.vol_target_annual
                       if spec.vol_target_annual is not None
                       else _VOL_TARGET)
    pnl_net, diag = _run_tsmom_backtest(
        weekly_closes,
        lookback_weeks=eff_lookback_w,
        skip_weeks=eff_skip_w,
        vol_target=eff_vol_target,
        tc_bp_per_rt=_TC_BP_PER_RT,
    )
    # Clip PnL to the user-requested date_range (drop warmup obs)
    pnl_net = pnl_net.loc[pnl_net.index >= pd.Timestamp(start_date)]

    # ── 6. Sample-size gate ────────────────────────────────────────
    n_obs_weeks  = len(pnl_net)
    n_obs_months = int(n_obs_weeks / 4.33)
    if n_obs_months < spec.min_obs_months:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"{n_obs_months} months of PnL < "
                                  f"min_obs_months={spec.min_obs_months}"),
            metrics          = {"n_obs_months":   n_obs_months,
                                 "n_obs_weeks":    n_obs_weeks,
                                 "min_required":   spec.min_obs_months},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 7. Stats (resample to monthly so research.ablation.metrics
    #     PERIODS_PER_YEAR=12 contract is satisfied) ───────────────
    from engine.research.ablation.metrics import (
        annualized_sharpe, newey_west_sharpe_se,
    )
    pnl_monthly = pnl_net.resample("ME").sum()
    if len(pnl_monthly) < 12:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"only {len(pnl_monthly)} monthly obs "
                                  "after resample — Sharpe SE unreliable"),
            metrics          = {"n_obs_monthly": len(pnl_monthly)},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    sharpe   = annualized_sharpe(pnl_monthly)
    se_sharpe = newey_west_sharpe_se(pnl_monthly)
    if (not math.isfinite(sharpe) or not math.isfinite(se_sharpe)
            or se_sharpe <= 0):
        t_stat = float("nan")
    else:
        t_stat = sharpe / se_sharpe

    ann_ret = float(pnl_monthly.mean()) * 12.0
    ann_vol = float(pnl_monthly.std(ddof=1)) * math.sqrt(12.0)

    verdict = _verdict_from_t(t_stat)
    summary = (f"TSMOM({eff_lookback_w},{eff_skip_w}) on "
                 f"{len(universe)} sector ETFs "
                 f"{spec.date_range}: Sharpe={sharpe:.2f}, "
                 f"t={t_stat:.2f}, n={n_obs_months}mo → {verdict}")

    # Phase 4.1 (2026-06-13): expose pnl_series_df + pnl_gross_col so
    # post_green_rigor FF5+MOM spanning check can run.
    import pandas as pd
    pnl_df = pd.DataFrame({
        "pnl_gross":    pnl_monthly,
        "pnl_net_13bp": pnl_monthly,   # tc already baked into pnl_monthly
        "turnover":     pd.Series(diag["avg_weekly_turnover"] * 4.0,
                                    index=pnl_monthly.index),
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "sharpe":              float(sharpe) if math.isfinite(sharpe) else None,
            "nw_t_stat":           float(t_stat) if math.isfinite(t_stat) else None,
            "nw_se_sharpe":        float(se_sharpe) if math.isfinite(se_sharpe) else None,
            "ann_return":          ann_ret if math.isfinite(ann_ret) else None,
            "ann_vol":             ann_vol if math.isfinite(ann_vol) else None,
            "n_obs_months":        n_obs_months,
            "n_obs_weeks":         n_obs_weeks,
            "n_tickers":           len(tickers),
            "n_tickers_in_data":   int(closes.shape[1]),
            "lookback_weeks":      eff_lookback_w,
            "skip_weeks":          eff_skip_w,
            "vol_target":          eff_vol_target,
            "tc_bp_per_rt":        _TC_BP_PER_RT,
            "avg_weekly_turnover": diag["avg_weekly_turnover"],
            "avg_n_active":        diag["avg_n_active"],
            # n_trials family contribution — used by downstream
            # n_trials_family_counter when (C-2c) emits
            # factor_verdict_filed
            "n_trials":            1,
        },
        artifacts        = {
            "pnl_series_df":   pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
