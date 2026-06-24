"""engine.agents.strengthener.templates.spanning_test_ff — BUG-2 fix.

Tests claims of the form:
  "Is anomaly X spanned by model M?" / "Does adding factor F to model M
   improve spanning?" / "Anomaly X is subsumed by FF5"

Sonnet drift case the architecture review flagged: when an FF5-vs-FF3
spanning claim arrives, Sonnet was stretching it into a
factor_combination spec (50/50 RMW+CMA) because no spanning template
existed. The drift produced verdicts that didn't match the original
claim. This template gives Sonnet a correct destination.

Scope (narrow MVP)
==================
  signal_kind     : spanning_test
  universe        : ken_french_ff5_mom
  data            : Ken French FF5+Mom weekly (1963-2026)
  test_asset      : one factor from {hml, mom, smb, rmw, cma}
  model           : remaining factors specified via signal_inputs
                    (default: FF5 minus the test_asset)
  spec.signal_inputs: tuple where the FIRST entry is the test asset,
                    the REST are the model factors:
                      ('ff.factors_weekly.mom',  # test asset
                       'ff.factors_weekly.mkt_rf',
                       'ff.factors_weekly.smb',
                       'ff.factors_weekly.hml',
                       'ff.factors_weekly.rmw',
                       'ff.factors_weekly.cma')
  rebal           : monthly (compounded from weekly)
  weighting_scheme_alt: unused

Methodology
===========
1. Load FF5+Mom weekly, compound to monthly
2. Regress: test_asset_excess = alpha + Σ β_i × model_factor_i + ε
   (HAC SE lag 6 via statsmodels)
3. Compute alpha-t with multi-testing-corrected threshold (BUG-3)
4. Verdict:
     NOT_SUBSUMED  |alpha-t| >= corrected threshold (orthogonal alpha)
                     → GREEN
     INDETERMINATE 1.65 <= |alpha-t| < corrected (boundary)
                     → MARGINAL
     SUBSUMED      |alpha-t| < 1.65 (model fully explains test asset)
                     → RED

Replication anchor (M2 hook)
============================
Well-established result: MOM is NOT spanned by FF5 (Asness-Frazzini-
Pedersen 2014 + Hou-Xue-Zhang 2015 confirm). Our 1963-2026 sample
should reproduce: alpha-t of MOM on FF5 > 2.5.

Test test_replicates_mom_not_spanned_by_ff5 in test_spanning_test_ff.py
guards this regression.
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

_TEMPLATE_VERSION = "v1.0_2026-06-13"

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FF_WEEKLY_PATH = _REPO_ROOT / "data" / "cache" / "ken_french_ff5_mom_weekly.parquet"

_MIN_OBS_MONTHS = 60

_FF_COLUMN_MAP = {
    "hml":      "HML",
    "mom":      "MOM",
    "smb":      "SMB",
    "rmw":      "RMW",
    "cma":      "CMA",
    "mkt_rf":   "MKT_RF",
}


def _canonical_ff_factor(raw: str) -> Optional[str]:
    name = (raw or "").lower()
    for prefix in ("ff.factors_weekly.", "ff.factors_monthly.",
                    "ff.factors.", "ff."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return _FF_COLUMN_MAP.get(name)


def _parse_spec_inputs(spec: FactorSpec) -> Optional[tuple[str, list[str]]]:
    """Returns (test_asset_col, [model_factor_cols]) or None on malformed."""
    inputs = tuple(spec.signal_inputs or ())
    if len(inputs) < 3:
        # Need at least 1 test asset + 2 model factors for a meaningful regression
        return None
    test_asset = _canonical_ff_factor(inputs[0])
    model_factors = []
    for raw in inputs[1:]:
        col = _canonical_ff_factor(raw)
        if col is None:
            return None
        if col == test_asset:
            return None   # model can't include its own dependent variable
        model_factors.append(col)
    if test_asset is None or not model_factors:
        return None
    return test_asset, model_factors


def _load_ff_monthly() -> Optional[pd.DataFrame]:
    if not _FF_WEEKLY_PATH.is_file():
        return None
    df = pd.read_parquet(_FF_WEEKLY_PATH)
    monthly = (1.0 + df).resample("ME").prod() - 1.0
    return monthly.dropna(how="all")


def _spanning_regression(
    test_asset: pd.Series, model_factors: pd.DataFrame, rf: pd.Series,
) -> tuple[float, float, dict[str, float], int]:
    """Returns (alpha_monthly, alpha_t, betas_dict, n_obs)."""
    df_parts = {"y": test_asset, "rf": rf}
    for col in model_factors.columns:
        df_parts[col] = model_factors[col]
    df = pd.concat(df_parts, axis=1).dropna()
    n = len(df)
    if n < 24:
        return float("nan"), float("nan"), {}, n

    excess = df["y"] - df["rf"]
    X_cols = list(model_factors.columns)
    X = df[X_cols].values

    try:
        import statsmodels.api as sm
        X_const = sm.add_constant(X)
        ols = sm.OLS(excess.values, X_const).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6},
        )
        alpha = float(ols.params[0])
        alpha_t = float(ols.tvalues[0])
        betas = {col: float(ols.params[1 + i]) for i, col in enumerate(X_cols)}
        return alpha, alpha_t, betas, n
    except Exception as exc:
        logger.debug("spanning regression failed: %s", exc)
        return float("nan"), float("nan"), {}, n


def _classify_verdict(
    alpha_t: float, n_trials: int,
) -> tuple[str, str, str]:
    """Returns (verdict, spanning_label, note).

    spanning_label is human-readable: SUBSUMED / INDETERMINATE / NOT_SUBSUMED.
    verdict is the canonical GREEN/MARGINAL/RED for the system.
    """
    if not math.isfinite(alpha_t):
        return "INSUFFICIENT_HISTORY", "INDETERMINATE", "alpha-t non-finite"

    # BUG-3 multi-testing-corrected thresholds
    try:
        from engine.research.verdict_thresholds import (
            alpha_t_green_threshold,
        )
        a_g = alpha_t_green_threshold(n_trials)
    except Exception:
        a_g = 2.0

    abs_t = abs(alpha_t)
    # BUG-7 fix (2026-06-13): NEGATIVE significant alpha is NOT a tradable
    # GREEN — it means the test asset has SIGNIFICANT UNDERPERFORMANCE vs
    # the model, which is "interesting research finding" (the asset is
    # NOT spanned) but NOT a tradable long-side strategy. Distinguish:
    #   alpha > 0, |t| >= threshold: GREEN (long-tradable orthogonal alpha)
    #   alpha < 0, |t| >= threshold: MARGINAL_NEGATIVE (research finding,
    #     not GREEN) — would be short-tradable but spanning verdicts are
    #     typically read as "this asset has independent alpha worth deploying"
    #   |t| in [1.65, threshold): MARGINAL (boundary)
    #   |t| < 1.65: RED (SUBSUMED)
    # Caught in production cron 2026-06-13 when HML-on-FF5-minus-HML gave
    # alpha=-27bp/mo alpha-t=-2.84 → previously falsely GREEN.
    if abs_t >= a_g:
        if alpha_t > 0:
            return "GREEN", "NOT_SUBSUMED", (
                f"alpha-t={alpha_t:.2f} clears threshold {a_g:.2f}; test "
                f"asset has POSITIVE orthogonal alpha to model (tradable long)"
            )
        else:
            return "MARGINAL", "NOT_SUBSUMED_NEGATIVE", (
                f"alpha-t={alpha_t:.2f} clears threshold {a_g:.2f} in "
                f"magnitude but is NEGATIVE; test asset under-performs the "
                f"model. Research finding (not subsumed) but NOT a tradable "
                f"GREEN — long-side investing would lose money."
            )
    if abs_t >= 1.65:
        return "MARGINAL", "INDETERMINATE", (
            f"alpha-t={alpha_t:.2f} in (1.65, {a_g:.2f}); boundary case"
        )
    return "RED", "SUBSUMED", (
        f"alpha-t={alpha_t:.2f} < 1.65; test asset spanned by model"
    )


def template_spanning_test_ff(spec: FactorSpec):
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

    # 2. Parse inputs
    parsed = _parse_spec_inputs(spec)
    if parsed is None:
        return TemplateResult(
            verdict          = "SIGNAL_INPUT_UNKNOWN",
            summary          = (f"signal_inputs={spec.signal_inputs}: expected "
                                  f"first entry = test asset, remaining = model "
                                  f"factors. All entries must be 'ff.<factor>' "
                                  f"with factor in {sorted(_FF_COLUMN_MAP)}; "
                                  f"need ≥3 entries; model can't include test asset"),
            metrics          = {"signal_inputs": list(spec.signal_inputs or ())},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    test_asset, model_factors = parsed

    if test_asset not in monthly.columns:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"test_asset column missing: {test_asset}",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    rf_series = monthly["RF"] if "RF" in monthly.columns else pd.Series(
        0.0, index=monthly.index,
    )

    # 3. Run spanning regression
    test_series = monthly[test_asset]
    model_df = monthly[[c for c in model_factors if c in monthly.columns]]
    if model_df.shape[1] < 2:
        return TemplateResult(
            verdict          = "INSUFFICIENT_DATA",
            summary          = f"only {model_df.shape[1]} model factor(s) available",
            metrics          = {},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    alpha_m, alpha_t, betas, n_obs = _spanning_regression(
        test_series, model_df, rf_series,
    )

    if n_obs < _MIN_OBS_MONTHS:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = f"only {n_obs} monthly obs (min {_MIN_OBS_MONTHS})",
            metrics          = {"n_obs_months": n_obs},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # 4. BUG-3 corrected thresholds via strategy_family n_trials
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

    verdict, spanning_label, note = _classify_verdict(alpha_t, n_trials)

    model_label = "+".join(model_factors)
    summary = (
        f"spanning_test: {test_asset} on {model_label} "
        f"({test_series.dropna().index[0].strftime('%Y-%m')}~"
        f"{test_series.dropna().index[-1].strftime('%Y-%m')}, "
        f"n={n_obs}mo): alpha={alpha_m*10000:+.0f}bp/mo, "
        f"alpha-t={alpha_t:+.2f} → {spanning_label} ({verdict}). {note}"
    )

    # Build a single-column "pnl" of the test asset for downstream
    # consumers expecting pnl_series_df shape
    _pnl_df = pd.DataFrame({
        "pnl_gross":     test_series,
        "pnl_net_13bp":  test_series,
        "turnover":      pd.Series(0.0, index=test_series.index),
    })

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "test_asset":           test_asset,
            "model_factors":        list(model_factors),
            "spanning_label":       spanning_label,
            "alpha_monthly":        alpha_m,
            "alpha_t":              alpha_t,
            "betas":                betas,
            "n_obs_months":         n_obs,
            "n_trials_at_dispatch": n_trials,
        },
        artifacts        = {
            "pnl_series_df":   _pnl_df,
            "pnl_default_col": "pnl_net_13bp",
            "pnl_gross_col":   "pnl_gross",
        },
        template_version = _TEMPLATE_VERSION,
    )
