"""engine.research.anchor_regression — Tier C L2-4 Commit 2.

Gibbons-Ross-Shanken (1989) style residual-alpha regression of a
candidate factor's monthly PnL against a panel of risk-factor anchor
series (e.g. Ken French FF-5 + Carhart MOM).

Mathematical contract:
  r_new(t) = α + Σ_i β_i · r_anchor_i(t) + ε(t)

  α          — monthly residual alpha (decimal)
  β_i        — anchor loading (dimensionless)
  ε(t)       — residual time-series
  t-stat α   — HAC (Newey-West) standard error on α / α

Why this matters (Hou-Xue-Zhang 2020 65%-replication-failure):
  A new factor's headline t-stat is INFLATED if it loads on known
  risk premia (RMW for profitability factors, MOM for momentum
  variants, HML for value tilts). Residual α t-stat strips the
  loadings and asks: is there a genuine OOS expected return ON TOP
  OF the canonical anchors?

  If headline t = 3.57 but residual α t-stat = 0.8 → factor is a
  known-risk-premium restatement. Demote, do NOT promote.

IO-FREE — pure function over pd.Series + pd.DataFrame inputs.
Wiring into emit / template lives in Commit 3.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Min overlap below which we refuse to regress — t-stat on alpha is
# garbage with too-few observations and the regression itself can
# be rank-deficient. 24 months = 2 years of monthly data, matches
# the minimum used by Bailey-Lopez de Prado for DSR power.
MIN_OVERLAP_MONTHS_DEFAULT = 24

# A.1 (2026-06-09): NW HAC lag formula moved to shared
# engine.research.lens_helpers (was duplicated across 3 modules).
from engine.research.lens_helpers import (
    nw_lag_rule_of_thumb as _nw_lag_rule_of_thumb,
)


def compute_residual_alpha(
    factor_pnl:    pd.Series,
    anchor_pnls:   pd.DataFrame,
    *,
    nw_lag:        Optional[int] = None,
    min_overlap:   int           = MIN_OVERLAP_MONTHS_DEFAULT,
    periods_per_year: int        = 12,
) -> Optional[dict]:
    """Run residual-α OLS with HAC SE.

    Args:
      factor_pnl: candidate factor's monthly PnL (DECIMAL, not
                  percent). Index = month-end DatetimeIndex.
      anchor_pnls: anchor panel; columns = anchor names. Index =
                  month-end DatetimeIndex. e.g. Ken French
                  MKT_RF / SMB / HML / RMW / CMA / MOM.
      nw_lag: HAC lag length. None → Newey-West rule-of-thumb
              floor(4·(N/100)^(2/9)) where N is overlap months.
      min_overlap: refuse regression if overlap < this many months.
                   Returns None instead of garbage stats.
      periods_per_year: 12 for monthly; would be 252 for daily.

    Returns:
      None when overlap < min_overlap or regression fails. Else:
        {
          "alpha_monthly":   float    — intercept (decimal/month)
          "alpha_annual":    float    — alpha_monthly * periods_per_year
          "alpha_nw_t":      float    — HAC t-stat on intercept
          "alpha_nw_se":     float    — HAC SE on intercept (monthly)
          "betas":           dict[str, float]  — loadings by anchor
          "beta_nw_t":       dict[str, float]  — HAC t-stat per anchor
          "r2":              float    — uncentered (textbook OLS R²)
          "r2_adj":          float    — degrees-of-freedom adjusted
          "residual_series": pd.Series — residual ε(t), month-end
          "n_overlap":       int      — months used
          "anchor_names":    tuple[str, ...]
          "nw_lag_used":     int
          "window":          "YYYY-MM:YYYY-MM"
        }

    Notes:
      - Anchors are NOT excess-of-risk-free. Standard practice in
        Fama-French regressions is to subtract RF from factor_pnl
        when RF is in anchor panel; here we run as-is and leave that
        choice to the caller (cleaner: factor PnL is already excess
        from a dollar-neutral L/S structure, so no RF subtraction).
      - We do NOT demote verdicts here. Caller decides what to do
        with the residual α t-stat.
      - Rank-deficient anchor panel raises via statsmodels — return
        None and log the failure.
    """
    # ── 1. Validate + align ─────────────────────────────────────────
    if factor_pnl is None or len(factor_pnl) == 0:
        logger.debug("anchor_reg: empty factor_pnl")
        return None
    if anchor_pnls is None or anchor_pnls.empty:
        logger.debug("anchor_reg: empty anchor_pnls")
        return None
    if not isinstance(factor_pnl.index, pd.DatetimeIndex):
        logger.warning("anchor_reg: factor_pnl index must be DatetimeIndex; "
                          "got %s", type(factor_pnl.index).__name__)
        return None
    if not isinstance(anchor_pnls.index, pd.DatetimeIndex):
        logger.warning("anchor_reg: anchor_pnls index must be DatetimeIndex; "
                          "got %s", type(anchor_pnls.index).__name__)
        return None

    # Align on date intersection; drop rows where ANY column is NaN.
    aligned = anchor_pnls.copy()
    aligned["__factor__"] = factor_pnl
    aligned = aligned.dropna(how="any")
    n_overlap = len(aligned)
    if n_overlap < min_overlap:
        logger.info("anchor_reg: insufficient overlap %d < %d, returning None",
                      n_overlap, min_overlap)
        return None

    y = aligned["__factor__"]
    X_no_const = aligned.drop(columns="__factor__")
    anchor_names = tuple(X_no_const.columns)

    # ── 2. Regress with HAC ─────────────────────────────────────────
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("anchor_reg: statsmodels not installed")
        return None

    lag = nw_lag if nw_lag is not None else _nw_lag_rule_of_thumb(n_overlap)
    X = sm.add_constant(X_no_const, has_constant="add")
    try:
        model   = sm.OLS(y.values, X.values, missing="raise")
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("anchor_reg: OLS+HAC failed: %s", exc)
        return None

    # statsmodels returns parameters in [const, anchor1, anchor2, ...] order
    params = results.params       # ndarray
    tvalues = results.tvalues
    bse = results.bse

    if not (np.isfinite(params).all() and np.isfinite(tvalues).all()):
        logger.warning("anchor_reg: non-finite parameters; returning None")
        return None

    alpha_monthly = float(params[0])
    alpha_nw_se   = float(bse[0])
    alpha_nw_t    = float(tvalues[0])

    betas      = {name: float(params[i + 1])
                    for i, name in enumerate(anchor_names)}
    beta_nw_t  = {name: float(tvalues[i + 1])
                    for i, name in enumerate(anchor_names)}

    # Residual series
    residual_array = y.values - X.values @ params
    residual_series = pd.Series(residual_array, index=y.index, name="residual")

    # ── 3. Build response ───────────────────────────────────────────
    window = (f"{aligned.index.min().strftime('%Y-%m')}:"
                f"{aligned.index.max().strftime('%Y-%m')}")

    return {
        "alpha_monthly":   alpha_monthly,
        "alpha_annual":    alpha_monthly * periods_per_year,
        "alpha_nw_t":      alpha_nw_t,
        "alpha_nw_se":     alpha_nw_se,
        "betas":           betas,
        "beta_nw_t":       beta_nw_t,
        "r2":              float(results.rsquared),
        "r2_adj":          float(results.rsquared_adj),
        "residual_series": residual_series,
        "n_overlap":       n_overlap,
        "anchor_names":    anchor_names,
        "nw_lag_used":     int(lag),
        "window":          window,
    }


# ────────────────────────────────────────────────────────────────────
# Convenience: load the cached Ken French anchor library
# ────────────────────────────────────────────────────────────────────
def load_famafrench_anchors(
    path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """A.1 (2026-06-09) shim: routes through anchor_library_registry.
    Kept for backwards compat with sibling lens modules + tests that
    import this name directly.

    Returns Ken French FF5+MOM monthly anchors in decimal units.
    None if the parquet is missing.
    """
    from engine.research.anchor_library_registry import load_library
    return load_library("ken_french_ff5_mom", explicit_path=path)


# ────────────────────────────────────────────────────────────────────
# L2-4 Stage 1 completion helpers (2026-06-09)
# ────────────────────────────────────────────────────────────────────
def _anchor_parquet_sha256(path_str: str) -> str:
    """A.1 shim: routes through anchor_library_registry SHA helper.
    Signature preserved for backwards compat with callers that pass
    an explicit path string."""
    from engine.research.anchor_library_registry import library_sha
    return library_sha("ken_french_ff5_mom", explicit_path=path_str)


def _compute_joint_loading_f_test(
    residual_alpha_result: dict,
    y_values:              np.ndarray,
    X_values:              np.ndarray,
    nw_lag:                int,
) -> Optional[dict]:
    """Re-fit the regression with statsmodels (so we have a results
    object) and run an F-test H0: all β_i = 0. Returns:
        {f_stat, f_pvalue, df_num, df_denom}
    or None on failure.

    Rationale: individual β t-stats tell you "is THIS anchor
    significant"; the joint F asks "does the factor load on ANY
    anchor at all". Rejecting H0 = factor is meaningfully correlated
    with FF5+MOM. Failing to reject = factor is orthogonal panel-wise
    even if some individual β scratched 1.96 by chance.
    """
    try:
        import statsmodels.api as sm
        model   = sm.OLS(y_values, X_values, missing="raise")
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})
        # Joint test: all non-intercept coefficients = 0. X has
        # const at position 0; anchors at positions 1..K.
        # Use restriction matrix R (n_anchors × n_params) — each row
        # tests one β = 0. Cleaner than string syntax which varies
        # across statsmodels versions.
        n_params  = X_values.shape[1]
        n_anchors = n_params - 1
        if n_anchors <= 0:
            return None
        R = np.zeros((n_anchors, n_params))
        for i in range(n_anchors):
            R[i, i + 1] = 1.0  # row i tests coefficient at position i+1
        f_test = results.f_test(R)
        return {
            "f_stat":   float(f_test.fvalue),
            "f_pvalue": float(f_test.pvalue),
            "df_num":   int(f_test.df_num),
            "df_denom": int(f_test.df_denom),
        }
    except Exception as exc:
        logger.warning("anchor_reg: joint F-test failed: %s", exc)
        return None


# ────────────────────────────────────────────────────────────────────
# High-level helper used by Tier C dispatcher (Commit 3 wiring;
# L2-4 Stage 1 completion 2026-06-09)
# ────────────────────────────────────────────────────────────────────
def compute_for_tier_c_pnl_series(
    pnl_series: pd.Series | pd.DataFrame,
    *,
    anchors:        Optional[pd.DataFrame] = None,
    anchor_library: str                    = "ken_french_ff5_mom",
    artifacts:      Optional[dict]         = None,
) -> Optional[dict]:
    """A.2 (2026-06-09): delegates to the residual_alpha_lens factory
    so the FF5+MOM lens shares one implementation with the LRV FX lens
    and any future single-stage anchor lens. The `anchor_library`
    parameter (default "ken_french_ff5_mom") selects which
    AnchorLibrary to use; this preserves the old signature for direct
    callers (tests, scripts) that import this name.
    """
    from engine.research.residual_alpha_lens import (
        compute_for_tier_c_pnl_series as _factory_compute,
    )
    return _factory_compute(
        anchor_library,
        pnl_series,
        anchors=anchors,
        artifacts=artifacts,
    )


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration (Phase 1 Commit 2, 2026-06-09;
# refactored A.2 2026-06-09 — generated by residual_alpha_lens factory)
# Per docs/spec_role_aware_test_routing.md §5 + §15
# ────────────────────────────────────────────────────────────────────
# A.2 (2026-06-09): LENS_DECLARATION generated by the residual_alpha_lens
# factory. Identical to the pre-A.2 hand-rolled declaration; both this
# lens and the FX-carry lens share one factory codepath.
#   - name preserved as "anchor_regression" so dispatcher routing +
#     event metric keys + endpoint consumers stay backward-compat
#   - consumed_by carries the equity-only downstream lens names
#     (industry_extension + cross_asset_extension) which are wired
#     to read this lens's output via the DAG resolver
from engine.research.residual_alpha_lens import make_residual_alpha_lens
import sys as _sys

LENS_DECLARATION = make_residual_alpha_lens(
    "ken_french_ff5_mom",
    lens_name     = "anchor_regression",
    consumed_by   = ("industry_extension", "cross_asset_extension"),
    # `wiring_module` enables late-binding lookup of
    # `anchor_regression.compute_for_tier_c_pnl_series` so external
    # monkey-patches on that public surface (used by dispatcher
    # integration tests) still take effect after A.2.
    wiring_module = _sys.modules[__name__],
)
