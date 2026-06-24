"""engine.research.cross_asset_attribution — Tier C cross-asset lite.

JOINT-model alpha decomposition extended with FRED cross-asset macro
regime regressors (VIX_change, DXY_return, BAA_spread_change,
T10Y3M_change, T10YIE_change). Per
[[feedback-fwl-sequential-residual-trap-2026-06-09]]:
α is from the JOINT model, "sequential" framing is reporting only.

Mathematical contract:
  factor_PnL = α_full +
               Σβ_FF5MOM · r_FF5MOM +
               Σγ_Industry · r_Industry +
               Σδ_Macro · macro_change +
               ε

Reports α_full (alpha after ALL panels explained) + Δα vs each
nested model (FF5+MOM only, FF5+MOM+Industry, FF5+MOM+Industry+Macro
= full) + per-panel subset F-tests.

Designed primarily for cross-asset sleeves (carry, TSMOM) where US
equity industries are mis-specified and macro regime variables are
the correct anchors. Also valid for equity sleeves as an additional
regime control layer.

IO-FREE pure function — wiring lives in Commit 4.
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

# A.1 (2026-06-09): canonical column lists + NW lag formula moved
# to anchor_library_registry / lens_helpers respectively.
from engine.research.anchor_library_registry import (
    get_library as _get_library,
)
from engine.research.lens_helpers import (
    nw_lag_rule_of_thumb as _nw_lag_rule_of_thumb,
)
MACRO_COLUMNS = _get_library("macro_us").anchor_columns
LRV_FX_COLUMNS = _get_library("lrv_fx_carry").anchor_columns


def _model_form_label(include_industry: bool, include_fx: bool) -> str:
    """Compose human-readable model form for audit reports."""
    parts = ["ff5mom"]
    if include_industry:
        parts.append("industry")
    parts.append("macro")
    if include_fx:
        parts.append("lrv_fx")
    return "joint_" + "_plus_".join(parts)


def compute_macro_extended_alpha(
    factor_pnl:      pd.Series,
    anchors_ff5mom:  pd.DataFrame,
    industries:      Optional[pd.DataFrame],
    macro:           pd.DataFrame,
    *,
    fx_carry_panel:   Optional[pd.DataFrame] = None,
    nw_lag:        Optional[int] = None,
    min_overlap:   int           = MIN_OVERLAP_MONTHS_DEFAULT,
    periods_per_year: int        = 12,
) -> Optional[dict]:
    """Run JOINT OLS+HAC of factor PnL on [FF5+MOM ∪ Industry ∪ Macro].

    Args:
      factor_pnl: monthly PnL series, month-end DatetimeIndex
      anchors_ff5mom: Ken French FF5+MOM
      industries: 12-industry panel; pass None to skip industry panel
                  (e.g., for pure cross-asset sleeves like carry where
                  US equity industries are mis-specified)
      macro: cross-asset macro regime panel (VIX_change, DXY_return,
             BAA_spread_change, T10Y3M_change, T10YIE_change)
      nw_lag: HAC lag default Newey-West rule-of-thumb
      min_overlap: refuse if overlap < this many months
      periods_per_year: 12 for monthly

    Returns None on failure (input empty, rank-deficient, etc.),
    else dict with α_full + per-panel β + per-panel joint F-tests.
    """
    if factor_pnl is None or len(factor_pnl) == 0:
        return None
    if anchors_ff5mom is None or anchors_ff5mom.empty:
        return None
    if macro is None or macro.empty:
        return None
    for name, obj in (("factor_pnl", factor_pnl),
                         ("anchors_ff5mom", anchors_ff5mom),
                         ("macro", macro)):
        if not isinstance(obj.index, pd.DatetimeIndex):
            logger.warning("xa_reg: %s index must be DatetimeIndex", name)
            return None
    if industries is not None and not isinstance(industries.index,
                                                      pd.DatetimeIndex):
        logger.warning("xa_reg: industries index must be DatetimeIndex")
        return None

    # Build combined DataFrame
    combined = anchors_ff5mom.copy()
    ff5mom_names = tuple(combined.columns)
    industry_names: tuple[str, ...] = ()
    if industries is not None:
        for c in industries.columns:
            if c in combined.columns:
                logger.warning("xa_reg: column collision %s industry vs anchor; skip", c)
                continue
            combined[c] = industries[c]
        industry_names = tuple(c for c in industries.columns
                                  if c in combined.columns)
    macro_names_in: list[str] = []
    for c in macro.columns:
        if c in combined.columns:
            logger.warning("xa_reg: column collision %s macro vs prior; skip", c)
            continue
        combined[c] = macro[c]
        macro_names_in.append(c)
    macro_names = tuple(macro_names_in)

    # Phase 2 Commit 4 (2026-06-09): optional LRV FX carry panel
    fx_carry_names_in: list[str] = []
    if fx_carry_panel is not None and not fx_carry_panel.empty:
        for c in fx_carry_panel.columns:
            if c in combined.columns:
                logger.warning(
                    "xa_reg: column collision %s fx_carry vs prior; skip",
                    c)
                continue
            combined[c] = fx_carry_panel[c]
            fx_carry_names_in.append(c)
    fx_carry_names = tuple(fx_carry_names_in)

    combined["__factor__"] = factor_pnl
    combined = combined.dropna(how="any")
    n_overlap = len(combined)
    if n_overlap < min_overlap:
        logger.info("xa_reg: insufficient overlap %d < %d",
                      n_overlap, min_overlap)
        return None

    y = combined["__factor__"]
    X_no_const = combined.drop(columns="__factor__")

    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("xa_reg: statsmodels not installed")
        return None

    lag = nw_lag if nw_lag is not None else _nw_lag_rule_of_thumb(n_overlap)
    X = sm.add_constant(X_no_const, has_constant="add")
    try:
        model   = sm.OLS(y.values, X.values, missing="raise")
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    except (ValueError, np.linalg.LinAlgError) as exc:
        logger.warning("xa_reg: joint OLS+HAC failed: %s", exc)
        return None

    params  = results.params
    tvalues = results.tvalues
    bse     = results.bse
    if not (np.isfinite(params).all() and np.isfinite(tvalues).all()):
        logger.warning("xa_reg: non-finite parameters")
        return None

    col_names = list(X_no_const.columns)
    ff5mom_betas, ff5mom_beta_t = {}, {}
    industry_betas, industry_beta_t = {}, {}
    macro_betas, macro_beta_t = {}, {}
    fx_carry_betas, fx_carry_beta_t = {}, {}
    for i, c in enumerate(col_names):
        b = float(params[i + 1])
        t = float(tvalues[i + 1])
        if c in ff5mom_names:
            ff5mom_betas[c]   = b
            ff5mom_beta_t[c]  = t
        elif c in industry_names:
            industry_betas[c]  = b
            industry_beta_t[c] = t
        elif c in macro_names:
            macro_betas[c]  = b
            macro_beta_t[c] = t
        elif c in fx_carry_names:
            fx_carry_betas[c]  = b
            fx_carry_beta_t[c] = t

    alpha_monthly = float(params[0])
    alpha_se      = float(bse[0])
    alpha_t       = float(tvalues[0])

    def _subset_f(panel_names: tuple[str, ...]) -> Optional[dict]:
        try:
            n_params = X.shape[1]
            positions = [col_names.index(c) + 1 for c in panel_names
                          if c in col_names]
            n = len(positions)
            if n == 0:
                return None
            R = np.zeros((n, n_params))
            for r_i, col_i in enumerate(positions):
                R[r_i, col_i] = 1.0
            f_test = results.f_test(R)
            return {
                "f_stat":   float(f_test.fvalue),
                "f_pvalue": float(f_test.pvalue),
                "df_num":   int(f_test.df_num),
                "df_denom": int(f_test.df_denom),
            }
        except Exception as exc:
            logger.warning("xa_reg: subset F %s failed: %s",
                              panel_names, exc)
            return None

    industry_joint_f = _subset_f(industry_names) if industry_names else None
    macro_joint_f    = _subset_f(macro_names)
    fx_carry_joint_f = _subset_f(fx_carry_names) if fx_carry_names else None

    residual_array = y.values - X.values @ params
    residual_series = pd.Series(residual_array, index=y.index,
                                   name="joint_residual_xa")

    window = (f"{combined.index.min().strftime('%Y-%m')}:"
                f"{combined.index.max().strftime('%Y-%m')}")

    return {
        "alpha_monthly":           alpha_monthly,
        "alpha_annual":            alpha_monthly * periods_per_year,
        "alpha_nw_t":              alpha_t,
        "alpha_nw_se":             alpha_se,
        "ff5mom_betas":            ff5mom_betas,
        "ff5mom_beta_nw_t":        ff5mom_beta_t,
        "industry_betas":          industry_betas,
        "industry_beta_nw_t":      industry_beta_t,
        "macro_betas":             macro_betas,
        "macro_beta_nw_t":         macro_beta_t,
        # Phase 2 Commit 4: LRV FX carry panel (HML_FX, DOL)
        "fx_carry_betas":          fx_carry_betas,
        "fx_carry_beta_nw_t":      fx_carry_beta_t,
        "r2":                      float(results.rsquared),
        "r2_adj":                  float(results.rsquared_adj),
        "residual_series":         residual_series,
        "n_overlap":               n_overlap,
        "nw_lag_used":             int(lag),
        "window":                  window,
        "industry_joint_f_test":   industry_joint_f,
        "macro_joint_f_test":      macro_joint_f,
        "fx_carry_joint_f_test":   fx_carry_joint_f,
        "panels_included":         {
            "ff5mom":   list(ff5mom_names),
            "industry": list(industry_names),
            "macro":    list(macro_names),
            "fx_carry": list(fx_carry_names),
        },
    }


# ────────────────────────────────────────────────────────────────────
# Macro parquet loading + SHA pinning
# ────────────────────────────────────────────────────────────────────
def load_lrv_fx_anchors(
    path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """A.1 shim: routes through anchor_library_registry. Registered
    units="percent" applies /100 unit conversion (was previously
    NOT done — silently mixed % LRV with decimal factor PnL in joint
    regressions; A.1 fixes this)."""
    from engine.research.anchor_library_registry import load_library
    return load_library("lrv_fx_carry", explicit_path=path)


def load_macro_anchors(path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """A.1 shim: routes through anchor_library_registry."""
    from engine.research.anchor_library_registry import load_library
    return load_library("macro_us", explicit_path=path)


def _macro_parquet_sha256(path_str: str) -> str:
    """A.1 shim: routes through anchor_library_registry SHA helper."""
    from engine.research.anchor_library_registry import library_sha
    return library_sha("macro_us", explicit_path=path_str)


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — extends industry-extension with macro
# ────────────────────────────────────────────────────────────────────
def compute_for_tier_c_with_macro(
    stage1_result:        dict,
    industry_extension:   Optional[dict],
    pnl_series:           pd.Series | pd.DataFrame,
    *,
    include_industry:     bool = True,
    include_lrv_fx:       bool = True,
    include_gfx_vol:      bool = False,
    industries:           Optional[pd.DataFrame] = None,
    macro:                Optional[pd.DataFrame] = None,
    fx_carry:             Optional[pd.DataFrame] = None,
    artifacts:            Optional[dict]         = None,
) -> Optional[dict]:
    """Tier C wiring entry. Runs JOINT [FF5+MOM ∪ Industry? ∪ Macro] OLS
    on factor PnL. Reports α_full + Δα vs prior nested models +
    per-panel subset F-tests. JSON-safe.

    Args:
      stage1_result: dict from anchor_regression Stage 1 (provides
                      α_FF5MOM_only baseline for the deepest Δα)
      industry_extension: dict from industry_attribution (provides
                          α_FF5MOM+Industry baseline for the macro Δα)
      pnl_series: Series (net-only) or DataFrame (uses pnl_net_13bp)
      include_industry: True for equity sleeves; False for pure cross-
                         asset sleeves where US equity industries are
                         mis-specified (carry, TSMOM)
      industries: optional DataFrame; loaded from parquet if None
      macro: optional DataFrame; loaded from parquet if None

    Returns None when:
      - macro parquet missing
      - Stage 1 missing
      - joint regression fails
    """
    if stage1_result is None:
        return None
    if macro is None:
        macro = load_macro_anchors()
    if macro is None:
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

    industry_panel = None
    if include_industry:
        if industries is None:
            try:
                from engine.research.industry_attribution import (
                    load_industry_anchors,
                )
                industry_panel = load_industry_anchors()
            except ImportError:
                industry_panel = None
        else:
            industry_panel = industries

    # Restrict macro to derived columns only (don't regress on level)
    macro_regressors = macro[[c for c in MACRO_COLUMNS if c in macro.columns]]

    # B.3 (2026-06-10): join MSSS global FX volatility innovation
    # into the macro panel. GFX_VOL_change is conceptually a macro
    # regime regressor (same class as VIX_change) so it rides the
    # existing macro grouping — no new panel kwarg, no new beta
    # group, no core-function change. MSSS 2012: carry's loading on
    # Δσ_FX is THE textbook crisis-exposure attribution for FX
    # strategies; VIX_change correlates but is US-equity vol.
    #
    # Gated by include_gfx_vol because GFX_VOL starts 1999 and the
    # joint regression drops any row with a NaN — joining it into an
    # equity sleeve (PnL back to 1992) would silently truncate 84
    # months of sample for a regressor that's mis-specified for
    # equities anyway. The runner derives the gate from the
    # registry's applicable_asset_classes (single source of truth).
    if include_gfx_vol:
        try:
            from engine.research.anchor_library_registry import load_library
            gfx = load_library("msss_gfx_vol")
            if gfx is not None and not gfx.empty:
                gfx_cols = [c for c in gfx.columns
                              if c not in macro_regressors.columns]
                if gfx_cols:
                    macro_regressors = macro_regressors.join(
                        gfx[gfx_cols], how="left",
                    )
        except Exception:
            logger.exception(
                "xa_reg: GFX_VOL join failed; proceeding without")

    # Phase 2 Commit 4 (2026-06-09): optionally include LRV FX
    # carry anchors. For FX carry / cross-asset carry sleeves this is
    # the institutional-standard panel (replaces lite macro DXY proxy).
    fx_panel = None
    if include_lrv_fx:
        if fx_carry is None:
            fx_carry = load_lrv_fx_anchors()
        if fx_carry is not None and not fx_carry.empty:
            fx_panel = fx_carry[[c for c in LRV_FX_COLUMNS
                                  if c in fx_carry.columns]]

    result = compute_macro_extended_alpha(
        pnl_net, anchors, industry_panel, macro_regressors,
        fx_carry_panel=fx_panel,
    )
    if result is None:
        return None

    # Δα chain: FF5MOM-only → +Industry → +Macro = full
    a_ff5mom_t   = stage1_result.get("alpha_nw_t")
    a_industry_t = (industry_extension or {}).get("alpha_full_nw_t")
    a_full_t     = result["alpha_nw_t"]

    delta_vs_ff5mom = (a_ff5mom_t - a_full_t) if a_ff5mom_t is not None else None
    delta_vs_industry = (a_industry_t - a_full_t) if a_industry_t is not None else None

    # SHA for provenance
    from pathlib import Path as _P
    default_macro = (
        _P(__file__).resolve().parents[2]
        / "data" / "anchor_library" / "cross_asset_macro_monthly.parquet"
    )
    macro_sha = _macro_parquet_sha256(str(default_macro))

    return {
        "alpha_full_monthly":      result["alpha_monthly"],
        "alpha_full_annual":       result["alpha_annual"],
        "alpha_full_nw_t":         result["alpha_nw_t"],
        "alpha_full_nw_se":        result["alpha_nw_se"],
        # Baselines for narrative
        "alpha_ff5mom_only_nw_t":  a_ff5mom_t,
        "alpha_with_industry_nw_t": a_industry_t,
        # Δα vs nested baselines
        "delta_vs_ff5mom_nw_t":    delta_vs_ff5mom,
        "delta_vs_industry_nw_t":  delta_vs_industry,
        # Loadings + subset F-tests
        "ff5mom_betas":            dict(result["ff5mom_betas"]),
        "ff5mom_beta_nw_t":        dict(result["ff5mom_beta_nw_t"]),
        "industry_betas":          dict(result["industry_betas"]),
        "industry_beta_nw_t":      dict(result["industry_beta_nw_t"]),
        "macro_betas":             dict(result["macro_betas"]),
        "macro_beta_nw_t":         dict(result["macro_beta_nw_t"]),
        # Phase 2 Commit 4: LRV HML_FX + DOL panel
        "fx_carry_betas":          dict(result.get("fx_carry_betas", {})),
        "fx_carry_beta_nw_t":      dict(result.get("fx_carry_beta_nw_t", {})),
        "industry_joint_f_test":   result["industry_joint_f_test"],
        "macro_joint_f_test":      result["macro_joint_f_test"],
        "fx_carry_joint_f_test":   result.get("fx_carry_joint_f_test"),
        "r2_full":                 result["r2"],
        "r2_adj_full":             result["r2_adj"],
        "n_overlap":               result["n_overlap"],
        "nw_lag_used":             result["nw_lag_used"],
        "window":                  result["window"],
        "panels_included":         result["panels_included"],
        "macro_snapshot_sha":      macro_sha or None,
        "model_form":              _model_form_label(
            include_industry,
            bool(result.get("fx_carry_betas")),
        ),
    }


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration (Phase 1 Commit 2, 2026-06-09)
# Per §15.A4: declares fallback_chain placeholder (KMPV / Lustig
# anchor libraries not yet built — placeholders for future work)
# ────────────────────────────────────────────────────────────────────
def _runner_cross_asset(spec, template_result, prior_outputs):
    artifacts = template_result.artifacts or {}
    pnl_df = artifacts.get("pnl_series_df")
    if pnl_df is None or len(pnl_df) == 0:
        return None
    # B.3 (2026-06-10): stage1 comes from WHICHEVER anchor lens ran.
    # equity/multi/cross_asset → anchor_regression; fx →
    # fx_carry_anchor_regression. Mutually exclusive by the O.2
    # applicability matrix, same union logic as the dispatcher's
    # anchor_orthogonality slot. Pre-B.3 this read only
    # anchor_regression, so the cross-asset extension NEVER ran for
    # FX sleeves — carry had no macro attribution at all.
    stage1 = (prior_outputs.get("anchor_regression")
                or prior_outputs.get("fx_carry_anchor_regression"))
    if stage1 is None:
        return None
    industry_ext = prior_outputs.get("industry_extension")
    asset_class = getattr(spec, "asset_class", None)
    # Cross-asset sleeve: skip industry panel (mis-specified)
    include_industry = asset_class in (None, "equity", "multi_asset")
    # B.3: GFX_VOL gate from the registry's declared asset classes
    # (single source of truth — see anchor_library_registry).
    include_gfx_vol = False
    try:
        from engine.research.anchor_library_registry import get_library
        _gfx_lib = get_library("msss_gfx_vol")
        include_gfx_vol = bool(
            _gfx_lib is not None
            and asset_class in _gfx_lib.applicable_asset_classes
        )
    except Exception:
        pass
    return compute_for_tier_c_with_macro(
        stage1, industry_ext, pnl_df,
        include_industry=include_industry,
        include_gfx_vol=include_gfx_vol,
        artifacts=artifacts,
    )


def _build_lens_declaration():
    from engine.research.lens_registry import LensDeclaration
    return LensDeclaration(
        name             = "cross_asset_extension",
        version          = "v1_2026-06-09",
        applicable_to    = {
            # All alpha sleeves benefit from macro regime overlay.
            # For equity sleeves it's a kitchen-sink 23-factor stack
            # (current default); for cross-asset it's 11-factor.
            "investment_role": ("alpha",),
            # All asset classes
        },
        input_protocols  = ("AnchorRegressionOutput",
                            "PnlSeriesDataFrameContract"),
        output_protocol  = "CrossAssetExtensionOutput",
        # Same low-α skip as industry_extension
        conditional_on   = {
            # B.3 (2026-06-10): tuple — accept EITHER anchor lens
            # (equity FF5+MOM or FX LRV; mutually exclusive per the
            # O.2 applicability matrix). Pre-B.3 this bound only to
            # anchor_regression, so the macro extension NEVER ran
            # for FX sleeves.
            "lens":      ("anchor_regression",
                            "fx_carry_anchor_regression"),
            "condition": lambda anchor_out: (
                abs(anchor_out.get("alpha_nw_t", 0)) >= 1.0
            ),
            "skip_reason_if_unmet":
                "anchor-stage α t-stat below 1.0 — macro "
                "extension uninformative on near-zero α factor",
        },
        # Per §15.A4 future work: fallback chain placeholders
        # (KMPV registration, Lustig HML_FX self-built). Today the
        # macro lite IS the primary; no fallbacks yet.
        fallback_chain   = (),
        output_schema    = {
            "primary":   "alpha_full_nw_t",
            "secondary": ("delta_vs_ff5mom_nw_t", "macro_betas",
                          "macro_beta_nw_t", "macro_joint_f_test",
                          "panels_included", "macro_snapshot_sha"),
        },
        consumed_by      = (),    # leaf
        runner           = _runner_cross_asset,
    )


LENS_DECLARATION = _build_lens_declaration()
