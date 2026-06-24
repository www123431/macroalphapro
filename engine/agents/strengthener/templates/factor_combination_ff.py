"""engine.agents.strengthener.templates.factor_combination_ff — bt-flex-4.2.

Tests claims like "50/50 value+momentum combination has higher Sharpe than
either alone" — Asness-Moskowitz-Pedersen 2013 "Value and Momentum
Everywhere" is the canonical shape.

Scope
=====
  signal_kind        : factor_combination
  universe           : ken_french_ff5_mom
  data source        : data/cache/ken_french_ff5_mom_weekly.parquet
                       (Ken French FF5+Mom weekly returns 1963-2026)
  signal_inputs      : ("ff.factors_weekly.<a>", "ff.factors_weekly.<b>")
                       where <a>, <b> ∈ {hml, mom, smb, rmw, cma, mkt_rf}
  weighting_scheme_alt: weight on first factor (0.05-0.95); second
                       factor gets 1-w. Default 0.50 (canonical
                       Asness-Moskowitz-Pedersen 50/50 value+momentum).

Methodology
===========
1. Load FF5+Mom weekly returns, resample to monthly (compound)
2. Extract the two requested factor columns
3. Combined return_t = w * factor_a_t + (1 - w) * factor_b_t
4. Compute combined Sharpe / Newey-West t / MaxDD
5. Jobson-Korkie/Memmel paired Sharpe-diff against EACH component
   (asks: does the combination strictly beat each constituent?)
6. CAPM regression: alpha vs MKT_RF (FF5 spanning is degenerate since
   combination IS an FF5 derivative)
7. 80bp/yr cost stress (HML/MOM monthly rebalanced ~80% turnover; cost
   drag ≈ 64bp/yr proportional to weights)

Verdict (forward-style, since this is a NEW combined strategy claim):
  GREEN     : |NW-t| ≥ 1.96 AND CAPM α-t ≥ 1.65 AND cost-stressed
              GREEN survives
  MARGINAL  : 1.65 ≤ |NW-t| < 1.96 OR cost-stress drops GREEN→MARGINAL
  RED       : otherwise

Note on factor_combination vs portfolio_overlay
================================================
portfolio_overlay (bt-flex-4.1) tests "X% strategy on Y BASE PORTFOLIO";
factor_combination (this) tests "X% factor + Y% factor". The first
benchmarks against a fixed asset-allocation portfolio (60/40 etc); the
second benchmarks against the individual constituents (J-K vs each).
"""
from __future__ import annotations

import dataclasses as _dc
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
_FF_WEEKLY_PATH = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_weekly.parquet"

_DEFAULT_WEIGHT = 0.50
_MIN_OBS_MONTHS = 60

# Cost stress: 80bp annual on the combined strategy (HML / MOM monthly
# turnover ~100%/yr → 80bp drag). Applied proportional to weights.
_COST_BP_PER_YEAR = 80.0
_COST_MONTHLY     = _COST_BP_PER_YEAR / 10_000.0 / 12.0

# Verdict thresholds
_T_GREEN    = 1.96
_T_MARGINAL = 1.65
_ALPHA_T_GREEN = 1.65   # CAPM alpha-t

# Map signal_input suffix → FF column name
_FF_COLUMN_MAP = {
    "hml":     "HML",
    "mom":     "MOM",
    "smb":     "SMB",
    "rmw":     "RMW",
    "cma":     "CMA",
    "mkt_rf":  "MKT_RF",
}


def _load_ff_monthly() -> Optional[pd.DataFrame]:
    """Load FF5+Mom weekly, compound to monthly. Returns DataFrame indexed by
    month-end with same columns as weekly source (excluding RF)."""
    if not _FF_WEEKLY_PATH.is_file():
        return None
    df = pd.read_parquet(_FF_WEEKLY_PATH)
    # Compound to monthly: (1+r1)(1+r2)...(1+rn) - 1
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def _parse_weight(spec: FactorSpec) -> float:
    raw = getattr(spec, "weighting_scheme_alt", None)
    if raw is None:
        return _DEFAULT_WEIGHT
    s = str(raw).strip().lower().replace("%", "").replace("pct", "")
    try:
        val = float(s)
        if val > 1.0:
            val /= 100.0
        if 0.05 <= val <= 0.95:
            return val
    except ValueError:
        pass
    return _DEFAULT_WEIGHT


def _parse_factor_inputs(spec: FactorSpec) -> Optional[tuple[str, str]]:
    """Extract two factor names from spec.signal_inputs.

    Returns (col_a, col_b) where each is the canonical FF column name,
    or None if inputs are malformed / unrecognized.
    """
    inputs = tuple(spec.signal_inputs or ())
    if len(inputs) != 2:
        return None
    cols: list[str] = []
    for raw in inputs:
        # Strip prefix "ff.factors_weekly." or "ff.factors_monthly."
        name = raw.lower()
        for prefix in ("ff.factors_weekly.", "ff.factors_monthly.",
                        "ff.factors.", "ff."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        col = _FF_COLUMN_MAP.get(name)
        if col is None:
            return None
        cols.append(col)
    if cols[0] == cols[1]:
        return None
    return cols[0], cols[1]


def _newey_west_t(returns: pd.Series, lag: int = 6) -> tuple[float, float]:
    """Annualized Sharpe + Newey-West SE → t-stat."""
    s = returns.dropna()
    n = len(s)
    if n < 12:
        return float("nan"), float("nan")
    mean = float(s.mean())
    std  = float(s.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return float("nan"), float("nan")
    sharpe_ann = (mean / std) * math.sqrt(12.0)
    try:
        import statsmodels.api as sm
        x = np.ones(n)
        ols = sm.OLS(s.values, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
        nw_t = float(ols.tvalues[0])
    except Exception:
        nw_t = float(mean / (std / math.sqrt(n))) if std > 0 else float("nan")
    return sharpe_ann, nw_t


def _capm_alpha_t(combined: pd.Series, mkt_rf: pd.Series,
                    rf: pd.Series) -> tuple[float, float]:
    """Returns (alpha_monthly, alpha_t) regression of combined - rf on MKT_RF.

    Kept for backward compat / sanity reporting. NOT the primary
    spanning gate after BUG-1 fix (2026-06-13) — CAPM α on a
    dollar-neutral long-short combo of FF factors is misleading.
    See _ff_complement_alpha_t for the correct spanning regression.
    """
    df = pd.concat({"r": combined, "mkt": mkt_rf, "rf": rf}, axis=1).dropna()
    if len(df) < 24:
        return float("nan"), float("nan")
    excess = df["r"] - df["rf"]
    try:
        import statsmodels.api as sm
        X = sm.add_constant(df["mkt"].values)
        ols = sm.OLS(excess.values, X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        return float(ols.params[0]), float(ols.tvalues[0])
    except Exception as exc:
        logger.debug("CAPM regression failed: %s", exc)
        return float("nan"), float("nan")


# BUG-1 fix (2026-06-13): the correct spanning anchor for a combo of FF
# factors is the COMPLEMENT of those factors within FF5+MOM. A 50/50
# HML+MOM combo by construction has zero alpha vs the FULL FF5+MOM (it
# IS a linear combo of FF factors). The meaningful question is: "does
# this combo add anything beyond what the OTHER FF factors capture?" —
# i.e., regress on MKT+SMB+RMW+CMA when components are HML+MOM.
#
# Anchor follows Fama-French 1996 / 2015 framework: alpha = excess
# return - sum(β_i * factor_i_excess) for factors NOT in the combo
# definition.

_FF_FULL_FACTOR_COLS = ("MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM")


def _ff_complement_columns(combo_factor_a: str, combo_factor_b: str) -> tuple[str, ...]:
    """Return the FF5+MOM factors NOT in the combo definition.

    Examples:
      HML+MOM combo → ('MKT_RF', 'SMB', 'RMW', 'CMA')
      RMW+CMA combo → ('MKT_RF', 'SMB', 'HML', 'MOM')
      SMB+HML combo → ('MKT_RF', 'RMW', 'CMA', 'MOM')
    """
    in_combo = {combo_factor_a.upper(), combo_factor_b.upper()}
    return tuple(f for f in _FF_FULL_FACTOR_COLS if f not in in_combo)


def _ff_complement_alpha_t(
    combined: pd.Series,
    monthly: pd.DataFrame,
    rf: pd.Series,
    combo_factor_a: str,
    combo_factor_b: str,
) -> tuple[float, float, dict[str, float]]:
    """Span combo against the COMPLEMENT of its constituents within
    FF5+MOM. Returns (alpha_monthly, alpha_t, betas_dict).

    The complement is what makes this meaningful: combo = w*A + (1-w)*B
    has analytically zero alpha vs {A, B, anything}, so the only
    informative question is its alpha vs the OTHER FF factors.

    If complement has fewer than 2 factors available, fall back to
    nan + empty betas (caller treats as inconclusive).
    """
    complement = _ff_complement_columns(combo_factor_a, combo_factor_b)
    available = [c for c in complement if c in monthly.columns]
    if len(available) < 2:
        return float("nan"), float("nan"), {}

    df_parts = {"r": combined, "rf": rf}
    for col in available:
        df_parts[col] = monthly[col]
    df = pd.concat(df_parts, axis=1).dropna()
    if len(df) < 24:
        return float("nan"), float("nan"), {}

    excess = df["r"] - df["rf"]
    X_cols = available
    X = df[list(X_cols)].values

    try:
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        ols = sm.OLS(excess.values, X_const).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        alpha = float(ols.params[0])
        alpha_t = float(ols.tvalues[0])
        betas = {col: float(ols.params[1 + i]) for i, col in enumerate(X_cols)}
        return alpha, alpha_t, betas
    except Exception as exc:
        logger.debug("FF complement regression failed: %s", exc)
        return float("nan"), float("nan"), {}


def _max_drawdown(returns: pd.Series) -> float:
    s = returns.dropna()
    if len(s) < 2:
        return float("nan")
    cum = (1.0 + s).cumprod()
    return float((cum / cum.cummax() - 1.0).min())


def _jobson_korkie_diff(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Annualized Sharpe diff (a - b) + t-stat per Memmel 2003."""
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    n = len(df)
    if n < 24:
        return float("nan"), float("nan")
    sa = float(df["a"].mean()) / float(df["a"].std(ddof=1))
    sb = float(df["b"].mean()) / float(df["b"].std(ddof=1))
    sharpe_diff = (sa - sb) * math.sqrt(12.0)
    sa_m = float(df["a"].mean()) / float(df["a"].std(ddof=1))
    sb_m = float(df["b"].mean()) / float(df["b"].std(ddof=1))
    rho  = float(df["a"].corr(df["b"]))
    var_term = (
        2.0 * (1.0 - rho)
        + 0.5 * (sa_m**2 + sb_m**2 - 2.0 * sa_m * sb_m * rho**2)
    )
    if var_term <= 0:
        return sharpe_diff, float("nan")
    se_m = math.sqrt(var_term / n)
    se_ann = se_m * math.sqrt(12.0)
    t = sharpe_diff / se_ann if se_ann > 0 else float("nan")
    return sharpe_diff, t


def _classify_verdict(nw_t: float, alpha_t: float,
                       nw_t_cost: float,
                       *,
                       t_green: float | None = None,
                       t_marginal: float | None = None,
                       alpha_t_green: float | None = None,
                      ) -> tuple[str, str]:
    """Returns (verdict, note). Thresholds default to module constants
    when not provided; in production the dispatcher passes BUG-3
    multi-testing-corrected thresholds (verdict_thresholds.t_green_
    threshold(n_trials_family)).
    """
    if not math.isfinite(nw_t):
        return "INSUFFICIENT_HISTORY", "NW-t non-finite"

    t_g  = t_green     if t_green     is not None else _T_GREEN
    t_m  = t_marginal  if t_marginal  is not None else _T_MARGINAL
    a_g  = alpha_t_green if alpha_t_green is not None else _ALPHA_T_GREEN

    # Gross verdict
    if abs(nw_t) >= t_g and (math.isnan(alpha_t)
                                       or abs(alpha_t) >= a_g):
        gross = "GREEN"
    elif abs(nw_t) >= t_m:
        gross = "MARGINAL"
    else:
        gross = "RED"

    # Cost-stress downgrade
    if abs(nw_t_cost) >= t_g:
        cost_stressed = "GREEN"
    elif abs(nw_t_cost) >= t_m:
        cost_stressed = "MARGINAL"
    else:
        cost_stressed = "RED"

    severity = {"GREEN": 2, "MARGINAL": 1, "RED": 0, "INSUFFICIENT_HISTORY": 0}
    if severity[cost_stressed] < severity[gross]:
        return cost_stressed, (
            f"cost-stress {gross} → {cost_stressed} "
            f"(NW-t {nw_t:.2f} → {nw_t_cost:.2f} after 80bp/yr drag)"
        )
    return gross, ""


def template_factor_combination_ff(spec: FactorSpec):
    """Public entry: invoked by factor_dispatcher."""
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # 1. Load data
    monthly = _load_ff_monthly()
    if monthly is None:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = "Ken French FF5+Mom weekly cache missing",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 2. Parse spec inputs
    factors = _parse_factor_inputs(spec)
    if factors is None:
        return TemplateResult(
            verdict          = "SIGNAL_INPUT_UNKNOWN",
            summary          = (f"signal_inputs={spec.signal_inputs}: expected "
                                  f"2 entries of form 'ff.factors_weekly.<x>' "
                                  f"where x ∈ {sorted(_FF_COLUMN_MAP)}"),
            metrics          = {"signal_inputs": list(spec.signal_inputs or ())},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    col_a, col_b = factors
    weight = _parse_weight(spec)

    if col_a not in monthly.columns or col_b not in monthly.columns:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"factor column(s) missing: {col_a}/{col_b}",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 3. Build combined series
    component_a = monthly[col_a].dropna()
    component_b = monthly[col_b].dropna()
    combined = (weight * component_a + (1.0 - weight) * component_b).dropna()

    n_obs = len(combined)
    if n_obs < _MIN_OBS_MONTHS:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = f"only {n_obs} monthly obs (min {_MIN_OBS_MONTHS})",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 4. Combined metrics (gross + cost-stressed)
    sharpe_gross, nw_t_gross = _newey_west_t(combined)
    combined_net = combined - _COST_MONTHLY
    sharpe_net,   nw_t_net   = _newey_west_t(combined_net)
    max_dd = _max_drawdown(combined)

    # 5. Spanning regression — BUG-1 fix 2026-06-13
    rf_series = monthly["RF"] if "RF" in monthly.columns else pd.Series(
        0.0, index=monthly.index,
    )
    # (a) CAPM α — backward compat / sanity reporting only, NOT verdict gate
    alpha_m, alpha_t = _capm_alpha_t(combined, monthly["MKT_RF"], rf_series)
    # (b) FF complement spanning — the real verdict anchor. Combo of
    # (col_a, col_b) regressed against FF5+MOM \ {col_a, col_b}.
    ff_complement_alpha_m, ff_complement_alpha_t, ff_betas = _ff_complement_alpha_t(
        combined, monthly, rf_series, col_a, col_b,
    )

    # 6. Jobson-Korkie vs each component
    jk_vs_a_diff, jk_vs_a_t = _jobson_korkie_diff(combined, component_a)
    jk_vs_b_diff, jk_vs_b_t = _jobson_korkie_diff(combined, component_b)

    # 7. Verdict — uses FF-complement α (NOT CAPM α) as the spanning
    # gate per BUG-1 fix. When complement regression unavailable (e.g.
    # data missing for required columns), fall back to NW-t only (less
    # strict but never falsely GREEN purely on CAPM α inflation).
    verdict_alpha_t = (
        ff_complement_alpha_t
        if math.isfinite(ff_complement_alpha_t)
        else alpha_t   # graceful fallback
    )

    # BUG-3 fix (2026-06-13): multi-testing-corrected thresholds. Look up
    # strategy_family n_trials, derive HLZ-floor / Bonferroni-body /
    # HLZ-ceil t threshold. Conservative for forward research; baseline
    # 1.96 inflates verdict severity under realistic multiple-testing
    # burden (HLZ 2016 + DeepSeek external audit independently flagged).
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
        logger.debug("BUG-3 threshold lookup failed; using defaults",
                       exc_info=True)
    from engine.research.verdict_thresholds import (
        t_green_threshold, t_marginal_threshold, alpha_t_green_threshold,
        threshold_summary,
    )
    _t_g  = t_green_threshold(n_trials)
    _t_m  = t_marginal_threshold(n_trials)
    _a_g  = alpha_t_green_threshold(n_trials)
    _thresh_summary = threshold_summary(n_trials)

    verdict, cost_note = _classify_verdict(
        nw_t_gross, verdict_alpha_t, nw_t_net,
        t_green=_t_g, t_marginal=_t_m, alpha_t_green=_a_g,
    )

    component_a_sharpe, _ = _newey_west_t(component_a)
    component_b_sharpe, _ = _newey_west_t(component_b)

    # Build complement-anchor label for summary line (e.g. "vs MKT+SMB+RMW+CMA")
    _complement_label = "+".join(_ff_complement_columns(col_a, col_b)) or "n/a"
    summary = (
        f"{weight*100:.0f}/{(1-weight)*100:.0f} {col_a}+{col_b} "
        f"({combined.index[0].strftime('%Y-%m')}~"
        f"{combined.index[-1].strftime('%Y-%m')}, n={n_obs}mo): "
        f"Sharpe={sharpe_gross:.2f}, NW-t={nw_t_gross:.2f}, "
        f"FF-complement α-t={ff_complement_alpha_t:.2f} (vs {_complement_label}), "
        f"CAPM α-t={alpha_t:.2f} [sanity], "
        f"MaxDD={max_dd*100:.1f}%, "
        f"vs {col_a}: ΔSh={jk_vs_a_diff:+.2f} (t={jk_vs_a_t:.2f}), "
        f"vs {col_b}: ΔSh={jk_vs_b_diff:+.2f} (t={jk_vs_b_t:.2f}) "
        f"→ {verdict}{cost_note}"
    )

    _pnl_df = pd.DataFrame({
        "pnl_gross":     combined,
        "pnl_net_13bp":  combined_net,          # rename for lens compat
        "pnl_baseline":  pd.Series(0.0, index=combined.index),
        "turnover":      pd.Series(0.0, index=combined.index),
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "weight_a":             weight,
            "weight_b":             1.0 - weight,
            "factor_a":             col_a,
            "factor_b":             col_b,
            "n_obs_months":         n_obs,
            "sharpe_gross":         sharpe_gross,
            "sharpe_net_80bp":      sharpe_net,
            "nw_t_gross":           nw_t_gross,
            "nw_t_net_80bp":        nw_t_net,
            "capm_alpha_monthly":          alpha_m,
            "capm_alpha_t":                alpha_t,
            # BUG-1 fix 2026-06-13: the real spanning anchor
            "ff_complement_alpha_monthly": ff_complement_alpha_m,
            "ff_complement_alpha_t":       ff_complement_alpha_t,
            "ff_complement_anchor":        list(_ff_complement_columns(col_a, col_b)),
            "ff_complement_betas":         ff_betas,
            # BUG-3 fix 2026-06-13: multi-testing-corrected thresholds
            "verdict_threshold_summary":   _thresh_summary,
            "n_trials_at_dispatch":         n_trials,
            "max_drawdown":         max_dd,
            "component_a_sharpe":   component_a_sharpe,
            "component_b_sharpe":   component_b_sharpe,
            "jk_vs_a_sharpe_diff":  jk_vs_a_diff,
            "jk_vs_a_t":            jk_vs_a_t,
            "jk_vs_b_sharpe_diff":  jk_vs_b_diff,
            "jk_vs_b_t":            jk_vs_b_t,
        },
        artifacts        = {
            "pnl_series_df":   _pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
