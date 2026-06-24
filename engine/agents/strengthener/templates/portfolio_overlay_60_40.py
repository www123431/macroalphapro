"""engine.agents.strengthener.templates.portfolio_overlay_60_40 — bt-flex-4.1.

Tests "X% allocation of strategy Y to a 60/40 portfolio" claims —
canonical Hurst-Ooi-Pedersen 2017 shape: "Adding a 20% allocation to
TSMOM to a traditional 60/40 portfolio reduces max drawdown, lowers
volatility, and increases returns over 1880-2016."

Scope (narrow per piece-by-piece doctrine)
=========================================
  signal_kind  : portfolio_overlay
  universe     : us_balanced_60_40 (60% SPY + 40% IEF)
  overlay      : single-asset TSMOM on SPY (12-month past-return sign,
                  vol-targeted to 10% annual)
  allocation   : spec.weighting_scheme_alt parsed for "0.10"-"0.30"
                  if not present, defaults to 0.20 (canonical HOP-2017)
  rebal        : monthly
  data window  : limited by SPY monthly cache (currently 2010-01..2024-12)
                  + IEF daily cache → resampled to monthly close

Verdict logic — DIFFERENT from single-factor templates
=====================================================
A portfolio-overlay test is NOT testing alpha relative to FF5 anchors.
It's testing whether the OVERLAID portfolio improves on the BASELINE.
Two relevant questions:

  1. Did Sharpe increase?  ΔSharpe = Sh(overlay) - Sh(baseline)
  2. Did max drawdown decrease?  ΔMaxDD = MDD(overlay) - MDD(baseline)
  3. Is ΔSharpe statistically distinguishable from zero?
     SE_diff = sqrt(SE_overlay^2 + SE_baseline^2 - 2*cov)  via Jobson-Korkie

GREEN     : ΔSharpe > +0.20 AND ΔMaxDD < -0.02 (≥2pp reduction)
            AND |ΔSharpe| / SE_diff >= 1.96
MARGINAL  : ΔSharpe > +0.10 AND ΔMaxDD < 0
            AND |ΔSharpe| / SE_diff >= 1.65
RED       : everything else

This is NOT comparable to factor strict-gate verdicts — it's a
portfolio engineering verdict. Senior lens stack (FF5 spanning,
spec_robustness) does NOT apply meaningfully. Dispatcher hooks
should skip spanning lenses for this signal_kind; for now they may
emit NA which downstream consumers treat as "not applicable".
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)

_TEMPLATE_VERSION = "v1.0_2026-06-11"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SPY_MONTHLY_PATH  = _REPO_ROOT / "data" / "multivariate_msm_v4" / "spy_monthly.parquet"
_BOND_ETF_PATH     = _REPO_ROOT / "data" / "cache" / "_bond_etf_px.parquet"

# Defaults — HOP-2017 canonical values
_DEFAULT_OVERLAY_PCT  = 0.20
_TSMOM_LOOKBACK_M     = 12
_TSMOM_VOL_TARGET     = 0.10        # 10% annual per-asset vol target
_TSMOM_VOL_LOOKBACK_M = 12
_MIN_OBS_MONTHS       = 36

_BASE_EQUITY_W = 0.60
_BASE_BOND_W   = 0.40
_BOND_PROXY    = "IEF"               # 7-10y treasury

# Verdict thresholds
# MaxDD sign convention: drawdowns are NEGATIVE numbers
# (e.g. -0.20 = 20% drawdown). Improvement = LESS NEGATIVE = delta is
# POSITIVE. So "≥2pp drawdown reduction" means maxdd_delta > +0.02.
_GREEN_SHARPE_DELTA    = 0.20
_GREEN_MAXDD_DELTA     = +0.02       # ≥ 2pp drawdown reduction (new MDD less painful)
_GREEN_T_THRESHOLD     = 1.96
_MARGINAL_SHARPE_DELTA = 0.10
_MARGINAL_T_THRESHOLD  = 1.65


# ────────────────────────────────────────────────────────────────────
# Data load
# ────────────────────────────────────────────────────────────────────


def _load_spy_monthly() -> Optional[pd.Series]:
    if not _SPY_MONTHLY_PATH.is_file():
        return None
    df = pd.read_parquet(_SPY_MONTHLY_PATH)
    col = "spy_ret" if "spy_ret" in df.columns else df.columns[0]
    return df[col].dropna()


def _load_ief_monthly() -> Optional[pd.Series]:
    """Resample daily IEF to monthly returns (last-day-of-month closes)."""
    if not _BOND_ETF_PATH.is_file():
        return None
    df = pd.read_parquet(_BOND_ETF_PATH)
    if _BOND_PROXY not in df.columns:
        return None
    px = df[_BOND_PROXY].dropna()
    # Resample to month-end last close
    monthly_close = px.resample("ME").last()
    return monthly_close.pct_change().dropna()


# ────────────────────────────────────────────────────────────────────
# Strategy construction
# ────────────────────────────────────────────────────────────────────


def _build_tsmom_overlay_returns(
    spy_ret: pd.Series,
    *,
    lookback_m: int = _TSMOM_LOOKBACK_M,
    vol_target: float = _TSMOM_VOL_TARGET,
    vol_lookback_m: int = _TSMOM_VOL_LOOKBACK_M,
) -> pd.Series:
    """Simple single-asset TSMOM on SPY: sign(past 12mo total return) × SPY,
    vol-targeted using trailing realized vol.

    Implementation:
      1. Past-lookback-month total return → sign +1 / -1
      2. Trailing realized monthly std → annualized vol estimate
      3. Position size = vol_target / realized_vol (clipped to [-3, +3])
      4. Returns: position(t) × spy_ret(t+1)
    Returns are aligned to spy_ret.index (loss of first `lookback_m` obs).
    """
    # Past 12-month total return (compound)
    log_ret = np.log1p(spy_ret)
    past_log = log_ret.rolling(lookback_m).sum()
    signal_sign = np.sign(np.expm1(past_log))

    # Realized vol (monthly std → annualize × sqrt(12))
    realized_vol = spy_ret.rolling(vol_lookback_m).std(ddof=1) * math.sqrt(12.0)

    # Position size = vol_target / realized_vol, clipped
    position = (vol_target / realized_vol).clip(-3.0, 3.0)
    signed_position = signal_sign * position

    # Returns: position(t) → realized at spy_ret(t+1)
    overlay_returns = signed_position.shift(1) * spy_ret
    return overlay_returns.dropna()


def _build_60_40_baseline(spy_ret: pd.Series, ief_ret: pd.Series) -> pd.Series:
    """Constant-weight 60/40 on monthly rebal. No drift; reset each month."""
    df = pd.concat({"spy": spy_ret, "ief": ief_ret}, axis=1).dropna()
    return _BASE_EQUITY_W * df["spy"] + _BASE_BOND_W * df["ief"]


def _build_overlaid_portfolio(
    baseline: pd.Series,
    overlay: pd.Series,
    overlay_pct: float,
) -> pd.Series:
    df = pd.concat({"base": baseline, "olay": overlay}, axis=1).dropna()
    return (1.0 - overlay_pct) * df["base"] + overlay_pct * df["olay"]


# ────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────


def _annualized_sharpe(monthly_returns: pd.Series) -> float:
    s = monthly_returns.dropna()
    if len(s) < 6 or s.std(ddof=1) <= 0:
        return float("nan")
    return float(s.mean() / s.std(ddof=1) * math.sqrt(12.0))


def _annualized_return(monthly_returns: pd.Series) -> float:
    s = monthly_returns.dropna()
    if len(s) < 1:
        return float("nan")
    return float(s.mean() * 12.0)


def _annualized_vol(monthly_returns: pd.Series) -> float:
    s = monthly_returns.dropna()
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1) * math.sqrt(12.0))


def _max_drawdown(monthly_returns: pd.Series) -> float:
    """Returns negative MaxDD (e.g. -0.30 = 30% drawdown)."""
    s = monthly_returns.dropna()
    if len(s) < 2:
        return float("nan")
    cum = (1.0 + s).cumprod()
    running_max = cum.cummax()
    dd = cum / running_max - 1.0
    return float(dd.min())


def _jobson_korkie_t_stat(
    s1: pd.Series, s2: pd.Series,
) -> tuple[float, float]:
    """Jobson-Korkie 1981 / Memmel 2003 — Sharpe ratio difference t-stat
    for two return series. Returns (t_stat, se_diff)."""
    df = pd.concat({"a": s1, "b": s2}, axis=1).dropna()
    if len(df) < 24:
        return float("nan"), float("nan")
    n = len(df)
    sh_a = _annualized_sharpe(df["a"])
    sh_b = _annualized_sharpe(df["b"])
    sigma_a = df["a"].std(ddof=1)
    sigma_b = df["b"].std(ddof=1)
    if sigma_a <= 0 or sigma_b <= 0:
        return float("nan"), float("nan")
    rho = df["a"].corr(df["b"])
    # SR diff variance per Memmel correction (omit higher-order terms)
    sr_a_m = sh_a / math.sqrt(12.0)
    sr_b_m = sh_b / math.sqrt(12.0)
    var_term = (
        2.0 * (1.0 - rho)
        + 0.5 * (sr_a_m**2 + sr_b_m**2 - 2.0 * sr_a_m * sr_b_m * rho**2)
    )
    if var_term <= 0:
        return float("nan"), float("nan")
    se_diff_m = math.sqrt(var_term / n)
    se_diff_annualized = se_diff_m * math.sqrt(12.0)
    sharpe_diff = sh_a - sh_b
    t_stat = sharpe_diff / se_diff_annualized if se_diff_annualized > 0 else float("nan")
    return float(t_stat), float(se_diff_annualized)


def _classify_verdict(sharpe_delta: float, maxdd_delta: float,
                      sharpe_diff_t: float) -> str:
    """Returns GREEN / MARGINAL / RED per docstring rules.

    Sign convention: MaxDD is NEGATIVE (e.g. -0.20). Improvement =
    LESS NEGATIVE = maxdd_delta POSITIVE. So GREEN requires
    maxdd_delta > +0.02 (≥2pp reduction)."""
    if (sharpe_delta > _GREEN_SHARPE_DELTA
            and maxdd_delta > _GREEN_MAXDD_DELTA
            and abs(sharpe_diff_t) >= _GREEN_T_THRESHOLD):
        return "GREEN"
    if (sharpe_delta > _MARGINAL_SHARPE_DELTA
            and maxdd_delta > 0.0
            and abs(sharpe_diff_t) >= _MARGINAL_T_THRESHOLD):
        return "MARGINAL"
    return "RED"


# ────────────────────────────────────────────────────────────────────
# Spec helpers
# ────────────────────────────────────────────────────────────────────


def _parse_overlay_pct(spec: FactorSpec) -> float:
    """Read overlay percentage from spec. The LLM extractor may set
    weighting_scheme_alt to a string like '0.20' or '20pct'; fall back
    to canonical 20% if unparseable."""
    raw = getattr(spec, "weighting_scheme_alt", None)
    if raw is None:
        return _DEFAULT_OVERLAY_PCT
    s = str(raw).strip().lower().replace("%", "").replace("pct", "")
    try:
        val = float(s)
        if val > 1.0:                    # given as percentage e.g. "20"
            val /= 100.0
        if 0.05 <= val <= 0.50:
            return val
    except ValueError:
        pass
    return _DEFAULT_OVERLAY_PCT


# ────────────────────────────────────────────────────────────────────
# Public entry — invoked by factor_dispatcher
# ────────────────────────────────────────────────────────────────────


def template_portfolio_overlay_60_40(spec: FactorSpec):
    """Strict-gate-compatible entry. Returns TemplateResult."""
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # ── 1. Load data ───────────────────────────────────────────────
    spy = _load_spy_monthly()
    ief = _load_ief_monthly()
    if spy is None or len(spy) < _MIN_OBS_MONTHS:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"SPY monthly cache missing or too short "
                                  f"(min {_MIN_OBS_MONTHS}mo required)",
            metrics          = {"spy_len": len(spy) if spy is not None else 0},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if ief is None or len(ief) < _MIN_OBS_MONTHS:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"IEF monthly returns missing or too short",
            metrics          = {"ief_len": len(ief) if ief is not None else 0},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 2. Build baseline 60/40 + TSMOM overlay ───────────────────
    baseline = _build_60_40_baseline(spy, ief)
    overlay_ret = _build_tsmom_overlay_returns(spy)
    overlay_pct = _parse_overlay_pct(spec)
    portfolio = _build_overlaid_portfolio(baseline, overlay_ret, overlay_pct)

    aligned = pd.concat({
        "base":      baseline,
        "overlay":   overlay_ret,
        "portfolio": portfolio,
    }, axis=1).dropna()

    n_obs = len(aligned)
    if n_obs < _MIN_OBS_MONTHS:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = f"only {n_obs} monthly obs after alignment "
                                  f"(min {_MIN_OBS_MONTHS})",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 3. Compute metrics ────────────────────────────────────────
    base_sharpe = _annualized_sharpe(aligned["base"])
    port_sharpe = _annualized_sharpe(aligned["portfolio"])
    base_ret    = _annualized_return(aligned["base"])
    port_ret    = _annualized_return(aligned["portfolio"])
    base_vol    = _annualized_vol(aligned["base"])
    port_vol    = _annualized_vol(aligned["portfolio"])
    base_mdd    = _max_drawdown(aligned["base"])
    port_mdd    = _max_drawdown(aligned["portfolio"])

    sharpe_delta = port_sharpe - base_sharpe
    maxdd_delta  = port_mdd - base_mdd
    ret_delta    = port_ret - base_ret
    vol_delta    = port_vol - base_vol

    sharpe_diff_t, sharpe_diff_se = _jobson_korkie_t_stat(
        aligned["portfolio"], aligned["base"],
    )

    verdict = _classify_verdict(sharpe_delta, maxdd_delta, sharpe_diff_t)

    summary = (
        f"60/40 + {overlay_pct*100:.0f}% TSMOM overlay on SPY "
        f"({aligned.index[0].strftime('%Y-%m')}~"
        f"{aligned.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"ΔSharpe={sharpe_delta:+.2f} (t={sharpe_diff_t:.2f}), "
        f"ΔMaxDD={maxdd_delta*100:+.1f}pp, "
        f"ΔRet={ret_delta*100:+.2f}pp → {verdict}"
    )

    # Cost adjustment — TSMOM overlay rebalances monthly at vol-target
    # changes. Approximate 100bp/year drag is conservative for SPY-only
    # TSMOM (real HOP cost ~50bp). Apply at portfolio level proportional
    # to overlay weight.
    cost_drag_annual = 0.0100 * overlay_pct
    port_ret_net = port_ret - cost_drag_annual

    _pnl_df = pd.DataFrame({
        "pnl_gross":     aligned["portfolio"],
        "pnl_net_13bp":  aligned["portfolio"] - (cost_drag_annual / 12.0),
        "pnl_baseline":  aligned["base"],
        "pnl_overlay":   aligned["overlay"],
        "turnover":      pd.Series(0.0, index=aligned.index),  # placeholder
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "overlay_pct":          overlay_pct,
            "n_obs_months":         n_obs,
            "base_sharpe":          base_sharpe,
            "overlay_portfolio_sharpe": port_sharpe,
            "sharpe_delta":         sharpe_delta,
            "sharpe_diff_t":        sharpe_diff_t,
            "sharpe_diff_se":       sharpe_diff_se,
            "base_ann_return":      base_ret,
            "overlay_portfolio_ann_return": port_ret,
            "overlay_portfolio_ann_return_net": port_ret_net,
            "ann_return_delta":     ret_delta,
            "base_ann_vol":         base_vol,
            "overlay_portfolio_ann_vol": port_vol,
            "ann_vol_delta":        vol_delta,
            "base_max_drawdown":    base_mdd,
            "overlay_portfolio_max_drawdown": port_mdd,
            "max_drawdown_delta":   maxdd_delta,
            "cost_drag_annual":     cost_drag_annual,
        },
        artifacts        = {
            "pnl_series_df":   _pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
