"""engine.research.industry_attribution — Tier C L2-6 lite Commit 2.

JOINT-MODEL alpha decomposition: factor PnL on (FF5+MOM ∪ 12-Industry)
in a single OLS+HAC regression. Per
[[feedback-fwl-sequential-residual-trap-2026-06-09]] (2026-06-09
amendment to [[feedback-anchor-panel-sequential-residual-doctrine]]):

  - α is computed from JOINT model (one true α after ALL panels
    explained), NOT from sequential residual regression
  - Δα = α_FF5MOM_only − α_full is the meaningful "alpha absorbed
    by adding industry" metric
  - Joint F-test on the industry SUBSET (H0: all γ_Industry = 0)
    answers "does industry panel add explanation independent of
    α magnitude"
  - Individual industry loadings come from the JOINT model
    (multicollinearity-aware) AND optionally from FWL-partialed
    industries (cleaner per-panel attribution)

The earlier "Stage 2 α₂" framing was a Frisch-Waugh-Lovell
violation — regressing ε₁ on raw X₂ produces α₂ ≈ 0 by OLS
construction regardless of true alpha. The empirical discovery
of this bug (PIT SN audit producing α₂ = -0.08 while joint F p =
0.62 says industries DON'T explain) is the smoking gun.

IO-FREE pure function — wiring lives in Commit 3.
"""
from __future__ import annotations

# hashlib removed: SHA delegated to anchor_library_registry (A.1)
import logging
import math
# lru_cache removed: SHA delegated to anchor_library_registry (A.1)
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MIN_OVERLAP_MONTHS_DEFAULT = 24

# A.1 (2026-06-09): canonical column list + NW lag formula moved
# to anchor_library_registry / lens_helpers respectively.
from engine.research.anchor_library_registry import (
    get_library as _get_library,
)
from engine.research.lens_helpers import (
    nw_lag_rule_of_thumb as _nw_lag_rule_of_thumb,
)
INDUSTRY_COLUMNS = _get_library("ff12_us_industry").anchor_columns


def compute_industry_extended_alpha(
    factor_pnl:      pd.Series,
    anchors_ff5mom:  pd.DataFrame,
    industries:      pd.DataFrame,
    *,
    nw_lag:        Optional[int] = None,
    min_overlap:   int           = MIN_OVERLAP_MONTHS_DEFAULT,
    periods_per_year: int        = 12,
) -> Optional[dict]:
    """Run JOINT OLS+HAC of factor PnL on [FF5+MOM ∪ Industries] and
    report α_full + per-anchor β + industry-subset joint F-test.

    Returns the JOINT-MODEL α — interpreted as "alpha after ALL
    18 anchors explained". Compare to FF5+MOM-only α (from
    anchor_regression.compute_for_tier_c_pnl_series) for the
    Δα ("alpha absorbed by industry") metric.

    Args:
      factor_pnl: candidate factor's monthly PnL (DECIMAL),
                  month-end DatetimeIndex
      anchors_ff5mom: Ken French FF5+MOM, columns
                     MKT_RF/SMB/HML/RMW/CMA/MOM
      industries: 12-Industry panel, columns from INDUSTRY_COLUMNS
      nw_lag: HAC lag, default Newey-West rule-of-thumb
      min_overlap: refuse if overlap < this many months
      periods_per_year: 12 for monthly

    Returns:
      None when:
        - any input empty / wrong index type
        - overlap < min_overlap
        - joint regression fails (rank-deficient etc.)
      Else dict with:
        alpha_monthly         joint α (decimal/month)
        alpha_annual          joint α × 12
        alpha_nw_t            HAC t-stat on joint α
        alpha_nw_se           HAC SE on joint α (monthly)
        ff5mom_betas          {anchor_name: β}        from joint model
        ff5mom_beta_nw_t      {anchor_name: NW-t}     from joint model
        industry_betas        {industry_name: β}       from joint model
        industry_beta_nw_t    {industry_name: NW-t}    from joint model
        r2                    joint model R²
        r2_adj                joint adjusted R²
        residual_series       pd.Series — joint residual ε
        n_overlap             months used
        nw_lag_used           int
        window                "YYYY-MM:YYYY-MM"
        industry_joint_f_test {f_stat, f_pvalue, df_num, df_denom}
                              H0: all 12 industry γ = 0
                              (valid under multicollinearity)
    """
    if factor_pnl is None or len(factor_pnl) == 0:
        return None
    if anchors_ff5mom is None or anchors_ff5mom.empty:
        return None
    if industries is None or industries.empty:
        return None
    for name, obj in (("factor_pnl", factor_pnl),
                         ("anchors_ff5mom", anchors_ff5mom),
                         ("industries", industries)):
        if not isinstance(obj.index, pd.DatetimeIndex):
            logger.warning("ind_reg: %s index must be DatetimeIndex", name)
            return None

    # Align all three on date intersection
    combined = anchors_ff5mom.copy()
    for c in industries.columns:
        if c in combined.columns:
            logger.warning("ind_reg: column collision %s between anchors "
                              "and industries; dropping from industry panel",
                              c)
            continue
        combined[c] = industries[c]
    combined["__factor__"] = factor_pnl
    combined = combined.dropna(how="any")
    n_overlap = len(combined)
    if n_overlap < min_overlap:
        logger.info("ind_reg: insufficient overlap %d < %d",
                      n_overlap, min_overlap)
        return None

    y = combined["__factor__"]
    X_no_const = combined.drop(columns="__factor__")
    ff5mom_names = tuple(c for c in anchors_ff5mom.columns
                            if c in X_no_const.columns)
    industry_names = tuple(c for c in industries.columns
                              if c in X_no_const.columns)

    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("ind_reg: statsmodels not installed")
        return None

    lag = nw_lag if nw_lag is not None else _nw_lag_rule_of_thumb(n_overlap)
    X = sm.add_constant(X_no_const, has_constant="add")
    try:
        model   = sm.OLS(y.values, X.values, missing="raise")
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("ind_reg: joint OLS+HAC failed: %s", exc)
        return None

    params  = results.params
    tvalues = results.tvalues
    bse     = results.bse
    if not (np.isfinite(params).all() and np.isfinite(tvalues).all()):
        logger.warning("ind_reg: non-finite parameters")
        return None

    # Position 0 = intercept; positions 1..K = FF5+MOM; positions
    # K+1..K+M = industries. Column order in X matches DataFrame.
    col_names = list(X_no_const.columns)  # excluding the const
    ff5mom_betas      = {}
    ff5mom_beta_nw_t  = {}
    industry_betas    = {}
    industry_beta_nw_t = {}
    for i, c in enumerate(col_names):
        b = float(params[i + 1])
        t = float(tvalues[i + 1])
        if c in ff5mom_names:
            ff5mom_betas[c]     = b
            ff5mom_beta_nw_t[c] = t
        elif c in industry_names:
            industry_betas[c]     = b
            industry_beta_nw_t[c] = t

    alpha_monthly = float(params[0])
    alpha_se      = float(bse[0])
    alpha_t       = float(tvalues[0])

    # Joint F-test: H0 — all industry coefficients = 0
    industry_joint_f: Optional[dict] = None
    try:
        n_params = X.shape[1]
        # Build R matrix: one row per industry, 1.0 at industry's
        # column position, zeros elsewhere
        ind_positions = [col_names.index(c) + 1 for c in industry_names]
        n_ind = len(ind_positions)
        if n_ind > 0:
            R = np.zeros((n_ind, n_params))
            for row_i, col_i in enumerate(ind_positions):
                R[row_i, col_i] = 1.0
            f_test = results.f_test(R)
            industry_joint_f = {
                "f_stat":   float(f_test.fvalue),
                "f_pvalue": float(f_test.pvalue),
                "df_num":   int(f_test.df_num),
                "df_denom": int(f_test.df_denom),
            }
    except Exception as exc:
        logger.warning("ind_reg: industry-subset F-test failed: %s", exc)

    residual_array = y.values - X.values @ params
    residual_series = pd.Series(residual_array, index=y.index,
                                   name="joint_residual")

    window = (f"{combined.index.min().strftime('%Y-%m')}:"
                f"{combined.index.max().strftime('%Y-%m')}")

    return {
        "alpha_monthly":           alpha_monthly,
        "alpha_annual":            alpha_monthly * periods_per_year,
        "alpha_nw_t":              alpha_t,
        "alpha_nw_se":             alpha_se,
        "ff5mom_betas":            ff5mom_betas,
        "ff5mom_beta_nw_t":        ff5mom_beta_nw_t,
        "industry_betas":          industry_betas,
        "industry_beta_nw_t":      industry_beta_nw_t,
        "r2":                      float(results.rsquared),
        "r2_adj":                  float(results.rsquared_adj),
        "residual_series":         residual_series,
        "n_overlap":               n_overlap,
        "nw_lag_used":             int(lag),
        "window":                  window,
        "industry_joint_f_test":   industry_joint_f,
    }


# ────────────────────────────────────────────────────────────────────
# Industry parquet loading + SHA pinning
# ────────────────────────────────────────────────────────────────────
def load_industry_anchors(
    path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """A.1 shim: routes through anchor_library_registry."""
    from engine.research.anchor_library_registry import load_library
    return load_library("ff12_us_industry", explicit_path=path)


def _industry_parquet_sha256(path_str: str) -> str:
    """A.1 shim: routes through anchor_library_registry SHA helper."""
    from engine.research.anchor_library_registry import library_sha
    return library_sha("ff12_us_industry", explicit_path=path_str)


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — runs joint extended model + computes Δα
# ────────────────────────────────────────────────────────────────────
def compute_for_tier_c_with_stage1_residual(
    stage1_result:   dict,
    pnl_series:      pd.Series | pd.DataFrame,
    *,
    industries:      Optional[pd.DataFrame] = None,
    artifacts:       Optional[dict]         = None,
) -> Optional[dict]:
    """Tier C wiring entry: re-run JOINT [FF5+MOM ∪ Industry] OLS on
    factor PnL, compare joint α to Stage 1 α, return JSON-safe dict.

    Naming note: "with_stage1_residual" name retained for backwards
    compatibility, but mechanics changed 2026-06-09 from sequential
    residual regression (FWL-violating bug) to JOINT model.

    Args:
      stage1_result: dict from anchor_regression Stage 1 (provides
                      α_FF5MOM_only for the Δα computation)
      pnl_series: Series (net-only) or DataFrame. When DataFrame is
                  passed and `artifacts` is supplied, column choice
                  flows through the pnl_default_col contract; else
                  falls back to legacy pnl_net_13bp lookup.
      industries: optional industry DataFrame; loaded from parquet
                  cache when None
      artifacts:  template_result.artifacts dict (B.2 contract).

    Returns None when industries missing or joint regression fails.
    """
    if stage1_result is None:
        return None
    if industries is None:
        industries = load_industry_anchors()
    if industries is None:
        return None

    from engine.research.lens_helpers import resolve_default_net_col
    if isinstance(pnl_series, pd.Series):
        pnl_net = pnl_series
    elif isinstance(pnl_series, pd.DataFrame):
        if artifacts is not None:
            col = resolve_default_net_col(artifacts)
        else:
            col = ("pnl_net_13bp"
                    if "pnl_net_13bp" in pnl_series.columns else None)
        if col is None or col not in pnl_series.columns:
            return None
        pnl_net = pnl_series[col].dropna()
    else:
        return None

    try:
        from engine.research.anchor_regression import load_famafrench_anchors
    except ImportError:
        return None
    anchors = load_famafrench_anchors()
    if anchors is None:
        return None

    result = compute_industry_extended_alpha(pnl_net, anchors, industries)
    if result is None:
        return None

    # Δα: α_FF5MOM_only - α_full (positive Δα = industry ATE alpha)
    alpha_ff5mom_only_monthly = stage1_result.get("alpha_monthly")
    alpha_ff5mom_only_t       = stage1_result.get("alpha_nw_t")
    delta_alpha_monthly = None
    delta_alpha_t = None  # NOTE: t-stat of DIFFERENCE is NOT a simple
                          # diff of t-stats due to different SE; report
                          # raw α diff for narrative use only
    if alpha_ff5mom_only_monthly is not None:
        delta_alpha_monthly = (alpha_ff5mom_only_monthly
                                  - result["alpha_monthly"])
    if alpha_ff5mom_only_t is not None:
        delta_alpha_t = alpha_ff5mom_only_t - result["alpha_nw_t"]

    # Industry parquet SHA for pinning
    from pathlib import Path as _P
    default_industry_parquet = (
        _P(__file__).resolve().parents[2]
        / "data" / "anchor_library" / "industries_12_monthly.parquet"
    )
    industry_sha = _industry_parquet_sha256(str(default_industry_parquet))

    return {
        # JOINT-model alpha (the real α_full)
        "alpha_full_monthly":      result["alpha_monthly"],
        "alpha_full_annual":       result["alpha_annual"],
        "alpha_full_nw_t":         result["alpha_nw_t"],
        "alpha_full_nw_se":        result["alpha_nw_se"],
        # Δα vs FF5+MOM-only (Stage 1)
        "alpha_ff5mom_only_nw_t":  alpha_ff5mom_only_t,
        "delta_alpha_monthly":     delta_alpha_monthly,
        "delta_alpha_nw_t_approx": delta_alpha_t,
        # Loadings from joint model
        "ff5mom_betas":            dict(result["ff5mom_betas"]),
        "ff5mom_beta_nw_t":        dict(result["ff5mom_beta_nw_t"]),
        "industry_betas":          dict(result["industry_betas"]),
        "industry_beta_nw_t":      dict(result["industry_beta_nw_t"]),
        # Joint R² + industry-subset F
        "r2_full":                 result["r2"],
        "r2_adj_full":             result["r2_adj"],
        "industry_joint_f_test":   result["industry_joint_f_test"],
        # Provenance
        "n_overlap":               result["n_overlap"],
        "industry_names":          list(INDUSTRY_COLUMNS),
        "nw_lag_used":             result["nw_lag_used"],
        "window":                  result["window"],
        "industry_snapshot_sha":   industry_sha or None,
        "model_form":              "joint_ff5mom_plus_12_industry",
    }


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration (Phase 1 Commit 2, 2026-06-09)
# Per §15.A4: conditional_on so we don't waste compute on industries
# when Stage 1 already shows no alpha.
# ────────────────────────────────────────────────────────────────────
def _runner_industry(spec, template_result, prior_outputs):
    """B.2: pass artifacts so pnl_default_col contract is honored."""
    artifacts = template_result.artifacts or {}
    pnl_df = artifacts.get("pnl_series_df")
    if pnl_df is None or len(pnl_df) == 0:
        return None
    stage1 = prior_outputs.get("anchor_regression")
    if stage1 is None:
        return None
    return compute_for_tier_c_with_stage1_residual(
        stage1, pnl_df, artifacts=artifacts,
    )


def _build_lens_declaration():
    from engine.research.lens_registry import LensDeclaration
    return LensDeclaration(
        name             = "industry_extension",
        version          = "v2_post_fwl_fix_2026-06-09",
        applicable_to    = {
            # US-equity 12-industry panel mis-specified for cross-asset
            "investment_role": ("alpha",),
            "asset_class":     ("equity", "multi_asset"),
        },
        input_protocols  = ("AnchorRegressionOutput",
                            "PnlSeriesDataFrameContract"),
        output_protocol  = "IndustryExtensionOutput",
        # Spec §15.A4: skip when Stage 1 already shows no alpha.
        # GP/A under Stage 1 α t = 1.88 is the borderline case
        # where industry extension produced meaningful spanning
        # diagnostic — keep threshold at |1.0| to catch this band.
        conditional_on   = {
            "lens":      "anchor_regression",
            "condition": lambda anchor_out: (
                abs(anchor_out.get("alpha_nw_t", 0)) >= 1.0
            ),
            "skip_reason_if_unmet":
                "anchor_regression α t-stat below 1.0 — industry "
                "extension uninformative on near-zero α factor",
        },
        fallback_chain   = (),
        output_schema    = {
            "primary":   "alpha_full_nw_t",
            "secondary": ("delta_alpha_nw_t_approx", "industry_betas",
                          "industry_beta_nw_t", "industry_joint_f_test",
                          "ff5mom_betas", "r2_full",
                          "industry_snapshot_sha"),
        },
        consumed_by      = ("cross_asset_extension",),
        runner           = _runner_industry,
    )


LENS_DECLARATION = _build_lens_declaration()
