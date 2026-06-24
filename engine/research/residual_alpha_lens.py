"""engine.research.residual_alpha_lens — factory for single-stage
residual α regression lenses, parameterized by AnchorLibrary name.

A.2 of the senior施工建议 (see [[project-tier-c-senior-construction-
plan-2026-06-09]]). Pre-A.2, the residual α regression machinery was
duplicated across `anchor_regression.py` (FF5+MOM) and
`fx_carry_anchor_regression.py` (LRV HML_FX+DOL). Both modules' Tier C
wiring helper + LENS_DECLARATION had near-identical structure
differing only in:
  - which AnchorLibrary they pulled from
  - the `applicable_to` asset_class filter
  - the `anchor_library` tag they wrote into the output dict

A.2 collapses both into a single factory:

  from engine.research.residual_alpha_lens import (
      compute_for_tier_c_pnl_series, make_residual_alpha_lens,
  )

  # Wiring helper takes the library NAME (string)
  out = compute_for_tier_c_pnl_series("ken_french_ff5_mom", pnl_df)

  # Lens factory generates a LensDeclaration per library
  LENS_DECLARATION = make_residual_alpha_lens("ken_french_ff5_mom")

Adding a new single-stage anchor lens after A.2 is now:
  1. Add the AnchorLibrary registration in anchor_library_registry.
  2. Create a 4-line module exposing
     `LENS_DECLARATION = make_residual_alpha_lens("<name>")`.

NOT collapsed into this factory:
  - `industry_attribution` (nested model, conditional_on stage1,
    Δα reporting — structurally different)
  - `cross_asset_attribution` (composite extension lens)

DESIGN INVARIANTS
=================
- The math engine (OLS+HAC) lives ONCE in
  `anchor_regression.compute_residual_alpha`. Both the FF5+MOM
  and LRV FX lens delegate to it. A.2 doesn't move that.
- Joint loading F-test (H0: all β = 0) also stays single-sourced
  via `anchor_regression._compute_joint_loading_f_test`.
- Output dict shape MATCHES AnchorRegressionOutput TypedDict — the
  same shape both pre-A.2 modules produced, so downstream consumers
  (self_doubt prompt rendering, factor_verdict_emit) need ZERO
  changes.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Tier C wiring helper — generic, parameterized by library name
# ────────────────────────────────────────────────────────────────────
def compute_for_tier_c_pnl_series(
    library_name: str,
    pnl_series:   pd.Series | pd.DataFrame,
    *,
    anchors:      Optional[pd.DataFrame] = None,
    artifacts:    Optional[dict]         = None,
) -> Optional[dict]:
    """Generic Tier C wiring entry for any single-stage residual α
    lens. Reads anchors from `library_name` via the central
    AnchorLibraryRegistry, runs `compute_residual_alpha`, attaches
    joint F-test + gross-dual block + anchor SHA + library tag.

    Args:
      library_name: registry key (e.g. "ken_french_ff5_mom",
                    "lrv_fx_carry").
      pnl_series:   Series (net only) OR DataFrame (uses
                    pnl_default_col / pnl_gross_col from artifacts
                    per [[feedback-explicit-artifacts-contract-no-
                    string-pattern-guessing-2026-06-09]]).
      anchors:      override for tests (skip registry load).
      artifacts:    template_result.artifacts dict (B.2 contract).

    Returns AnchorRegressionOutput-shaped dict, or None when:
      - library is unknown
      - anchor parquet missing
      - factor pnl empty / too short
      - regression rank-deficient
    """
    from engine.research.anchor_regression import (
        compute_residual_alpha, _compute_joint_loading_f_test,
    )
    from engine.research.anchor_library_registry import (
        get_library, library_sha, load_library,
    )
    from engine.research.lens_helpers import (
        resolve_default_net_col, resolve_gross_col,
    )

    lib = get_library(library_name)
    if lib is None:
        logger.warning("residual_alpha_lens: unknown library %r",
                          library_name)
        return None

    if anchors is None:
        anchors = load_library(library_name)
    if anchors is None:
        return None

    # Resolve net + gross series. Explicit artifacts contract preferred
    # per B.2 doctrine; legacy fallback for direct callers (tests).
    pnl_net:   Optional[pd.Series] = None
    pnl_gross: Optional[pd.Series] = None
    if isinstance(pnl_series, pd.Series):
        pnl_net = pnl_series
    elif isinstance(pnl_series, pd.DataFrame):
        if artifacts is not None:
            net_col   = resolve_default_net_col(artifacts)
            gross_col = resolve_gross_col(artifacts)
        else:
            # Legacy: prefer common templated column names
            net_col = next((c for c in ("pnl_net_13bp", "pnl_net_8bp")
                              if c in pnl_series.columns), None)
            gross_col = ("pnl_gross"
                          if "pnl_gross" in pnl_series.columns else None)
        if net_col is not None and net_col in pnl_series.columns:
            pnl_net = pnl_series[net_col].dropna()
        if gross_col is not None and gross_col in pnl_series.columns:
            pnl_gross = pnl_series[gross_col].dropna()
    else:
        return None

    if pnl_net is None or len(pnl_net) == 0:
        # Some FX templates may not ship a net column; use gross as
        # the regression target (carry has minuscule TC anyway).
        pnl_net = pnl_gross
    if pnl_net is None or len(pnl_net) == 0:
        return None

    result_net = compute_residual_alpha(pnl_net, anchors)
    if result_net is None:
        return None

    # Joint F-test (H0: all β = 0). Re-fit for the F-statistic since
    # compute_residual_alpha discards the fitted-results object.
    aligned = anchors.copy()
    aligned["__factor__"] = pnl_net
    aligned = aligned.dropna(how="any")
    joint_f_net = None
    if len(aligned) >= 24:
        y = aligned["__factor__"].values
        X_no_const = aligned.drop(columns="__factor__")
        try:
            import statsmodels.api as sm
            X = sm.add_constant(X_no_const, has_constant="add").values
            joint_f_net = _compute_joint_loading_f_test(
                result_net, y, X, result_net["nw_lag_used"],
            )
        except ImportError:
            joint_f_net = None

    # Gross dual regression — apples-to-apples vs published anchors
    # (which are typically pre-cost). When the template ships a gross
    # series, run the SAME residual α regression on it.
    gross_block: Optional[dict] = None
    if pnl_gross is not None and len(pnl_gross) > 0:
        result_gross = compute_residual_alpha(pnl_gross, anchors)
        if result_gross is not None:
            gross_block = {
                "alpha_monthly":   result_gross["alpha_monthly"],
                "alpha_annual":    result_gross["alpha_annual"],
                "alpha_nw_t":      result_gross["alpha_nw_t"],
                "alpha_nw_se":     result_gross["alpha_nw_se"],
                "betas":           dict(result_gross["betas"]),
                "beta_nw_t":       dict(result_gross["beta_nw_t"]),
                "r2":              result_gross["r2"],
                "n_overlap":       result_gross["n_overlap"],
            }

    anchor_sha = library_sha(library_name) or None

    return {
        "alpha_monthly":         result_net["alpha_monthly"],
        "alpha_annual":          result_net["alpha_annual"],
        "alpha_nw_t":            result_net["alpha_nw_t"],
        "alpha_nw_se":           result_net["alpha_nw_se"],
        "betas":                 dict(result_net["betas"]),
        "beta_nw_t":             dict(result_net["beta_nw_t"]),
        "r2":                    result_net["r2"],
        "r2_adj":                result_net["r2_adj"],
        "n_overlap":             result_net["n_overlap"],
        "anchor_names":          list(result_net["anchor_names"]),
        "nw_lag_used":           result_net["nw_lag_used"],
        "window":                result_net["window"],
        "anchor_library":        library_name,
        "gross":                 gross_block,
        "joint_loading_f_test":  joint_f_net,
        "anchor_snapshot_sha":   anchor_sha,
    }


# ────────────────────────────────────────────────────────────────────
# Lens factory — generates a LensDeclaration per library
# ────────────────────────────────────────────────────────────────────
def make_residual_alpha_lens(
    library_name:  str,
    *,
    lens_name:     Optional[str] = None,
    version:       Optional[str] = None,
    consumed_by:   tuple = (),
    wiring_module: Optional[object] = None,
):
    """Build a `LensDeclaration` for the residual α lens that uses
    the named AnchorLibrary.

    Args:
      library_name: AnchorLibraryRegistry key.
      lens_name:    LensDeclaration.name. Defaults to:
                      "anchor_regression"           for ken_french_ff5_mom
                      "fx_carry_anchor_regression"  for lrv_fx_carry
                      "<library>_residual_alpha"    otherwise
                    (these defaults preserve existing lens names so
                    dispatcher routing + event metric keys + endpoint
                    consumers stay backward-compatible.)
      version:      LensDeclaration.version. Defaults to
                    "factory_v1_<library>_2026-06-09".
      consumed_by:  downstream lens names that read this lens's
                    output (e.g. anchor_regression is consumed by
                    industry_extension + cross_asset_extension).

    Returns a frozen LensDeclaration ready to assign to module-level
    LENS_DECLARATION.
    """
    from engine.research.lens_registry import LensDeclaration
    from engine.research.anchor_library_registry import get_library

    lib = get_library(library_name)
    if lib is None:
        raise ValueError(
            f"make_residual_alpha_lens: unknown library {library_name!r}; "
            f"register it in anchor_library_registry first"
        )

    # Default lens name → preserve original module's lens name so
    # dispatcher routing + emitted metric keys stay backward-compat.
    if lens_name is None:
        _DEFAULTS = {
            "ken_french_ff5_mom": "anchor_regression",
            "lrv_fx_carry":       "fx_carry_anchor_regression",
        }
        lens_name = _DEFAULTS.get(library_name,
                                      f"{library_name}_residual_alpha")

    if version is None:
        version = f"factory_v1_{library_name}_2026-06-09"

    def _runner(spec, template_result, prior_outputs):
        """Generic runner. When `wiring_module` was provided, calls
        that module's `compute_for_tier_c_pnl_series` via getattr so
        monkey-patches on the module attribute still take effect (the
        runner doesn't capture a stale function reference). Otherwise
        uses this factory's generic helper directly.
        """
        artifacts = template_result.artifacts or {}
        pnl_df = artifacts.get("pnl_series_df")
        if pnl_df is None or len(pnl_df) == 0:
            return None
        if wiring_module is not None:
            # Late-binding lookup — monkey-patches on
            # wiring_module.compute_for_tier_c_pnl_series work.
            helper = getattr(wiring_module,
                                "compute_for_tier_c_pnl_series", None)
            if helper is not None:
                return helper(pnl_df, artifacts=artifacts)
        return compute_for_tier_c_pnl_series(
            library_name, pnl_df, artifacts=artifacts,
        )

    return LensDeclaration(
        name             = lens_name,
        version          = version,
        applicable_to    = {
            "investment_role": ("alpha", "overlay"),
            "asset_class":     tuple(lib.applicable_asset_classes),
        },
        input_protocols  = ("PnlSeriesDataFrameContract",),
        output_protocol  = "AnchorRegressionOutput",
        conditional_on   = None,    # always run when applicable
        fallback_chain   = (),
        output_schema    = {
            "primary":   "alpha_nw_t",
            "secondary": ("betas", "beta_nw_t", "r2", "r2_adj",
                          "joint_loading_f_test", "gross",
                          "anchor_snapshot_sha"),
        },
        consumed_by      = tuple(consumed_by),
        runner           = _runner,
    )
