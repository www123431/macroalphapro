"""engine.research._lens_protocols — TypedDict contracts for lens outputs.

Per docs/spec_role_aware_test_routing.md §15.A6: each lens module
declares its output as a TypedDict so downstream consumers can be
typechecked. Catches the "rename a field, silently break all
downstream" refactor bug that string-based dependencies miss.

Convention: each TypedDict declares ONLY the fields downstream
consumers actually access — not the full output dict. The TypedDict
is a refactor-safety contract, not a complete schema.

Why TypedDict not Protocol:
  - Our lens outputs are dict[str, Any], not class instances
  - TypedDict supports structural typing for dicts natively
  - mypy / pyright check `obj["field"]` access against TypedDict
  - Cleaner ergonomics for the dict-based output convention we
    already use (compute_for_tier_c_pnl_series → dict)
"""
from __future__ import annotations

from typing import Any, TypedDict


class AnchorRegressionOutput(TypedDict, total=False):
    """Stage 1: FF5+MOM anchor regression.

    Downstream consumers (industry_extension, cross_asset_extension,
    self_doubt prompt) access these fields.
    """
    alpha_monthly:        float
    alpha_annual:         float
    alpha_nw_t:           float
    alpha_nw_se:          float
    betas:                dict[str, float]
    beta_nw_t:            dict[str, float]
    r2:                   float
    r2_adj:               float
    n_overlap:            int
    anchor_names:         list[str]
    nw_lag_used:          int
    window:               str
    anchor_library:       str
    # L2-4 Stage 1 additions (gross / GRS F / SHA)
    gross:                dict[str, Any]
    joint_loading_f_test: dict[str, Any]
    anchor_snapshot_sha:  str | None


class SubsampleStabilityOutput(TypedDict, total=False):
    """L2-5: N-split decomposition."""
    n_splits:                int
    n_total_months:          int
    windows:                 list[dict[str, Any]]
    worst_best_sharpe_ratio: float | None
    institutional_stable:    bool
    monotone_decay:          bool
    monotone_growth:         bool
    decay_slope_per_year:    float | None
    decay_slope_t:           float | None


class IndustryExtensionOutput(TypedDict, total=False):
    """L2-6: joint FF5+MOM + 12-Industry regression.

    Downstream consumers (cross_asset_extension, self_doubt prompt)
    use alpha_full_nw_t + industry_joint_f_test + delta_alpha.
    """
    alpha_full_monthly:        float
    alpha_full_annual:         float
    alpha_full_nw_t:           float
    alpha_full_nw_se:          float
    alpha_ff5mom_only_nw_t:    float | None
    delta_alpha_monthly:       float | None
    delta_alpha_nw_t_approx:   float | None
    ff5mom_betas:              dict[str, float]
    industry_betas:            dict[str, float]
    industry_beta_nw_t:        dict[str, float]
    r2_full:                   float
    r2_adj_full:               float
    industry_joint_f_test:     dict[str, Any] | None
    industry_snapshot_sha:     str | None
    model_form:                str


class CrossAssetExtensionOutput(TypedDict, total=False):
    """Cross-asset macro extension: joint FF5+MOM + (Industry?) + Macro."""
    alpha_full_monthly:         float
    alpha_full_annual:          float
    alpha_full_nw_t:            float
    alpha_full_nw_se:           float
    alpha_ff5mom_only_nw_t:     float | None
    alpha_with_industry_nw_t:   float | None
    delta_vs_ff5mom_nw_t:       float | None
    delta_vs_industry_nw_t:     float | None
    ff5mom_betas:               dict[str, float]
    industry_betas:             dict[str, float]
    macro_betas:                dict[str, float]
    macro_beta_nw_t:            dict[str, float]
    r2_full:                    float
    industry_joint_f_test:      dict[str, Any] | None
    macro_joint_f_test:         dict[str, Any] | None
    panels_included:            dict[str, list[str]]
    macro_snapshot_sha:         str | None
    model_form:                 str


class SpecificationRobustnessOutput(TypedDict, total=False):
    """B (senior施工建议): neighborhood ablation of B-class params.
    Reports Sharpe stability and ROBUST/MARGINAL/LIKELY_OVERFIT verdict.

    n_trials_increment is ALWAYS 0 — these are robustness cells of one
    hypothesis, NOT N hypotheses, so they don't inflate Bailey-LdP DSR.
    """
    status:              str
    verdict:             str    # ROBUST|MARGINAL_OVERFIT|LIKELY_OVERFIT|UNTESTABLE
    stability_score:     float | None
    robust_bar:          float
    marginal_bar:        float
    base_sharpe:         float
    base_t:              float | None
    sharpe_median:       float
    sharpe_min:          float
    sharpe_max:          float
    neighborhood_size:   int
    successful_cells:    int
    errors:              int
    cells_tested:        list[dict[str, Any]]
    n_trials_increment:  int    # ALWAYS 0


class PnlSeriesDataFrameContract(TypedDict, total=False):
    """The template-produced pnl_series_df shape that lenses expect.
    Not a literal TypedDict for a DataFrame (DataFrames aren't dicts),
    but a documented contract for which columns must be present."""
    pnl_gross:    Any  # pd.Series, monthly
    pnl_net_13bp: Any
    pnl_net_80bp: Any
    turnover:     Any


REQUIRED_PNL_DF_COLUMNS = ("pnl_gross", "pnl_net_13bp",
                              "pnl_net_80bp", "turnover")


# Registry of all lens output TypedDicts by name. Used by
# lens_registry.py for declaration validation.
OUTPUT_PROTOCOL_REGISTRY: dict[str, type] = {
    "AnchorRegressionOutput":         AnchorRegressionOutput,
    "SubsampleStabilityOutput":       SubsampleStabilityOutput,
    "IndustryExtensionOutput":        IndustryExtensionOutput,
    "CrossAssetExtensionOutput":      CrossAssetExtensionOutput,
    "SpecificationRobustnessOutput":  SpecificationRobustnessOutput,
    "PnlSeriesDataFrameContract":     PnlSeriesDataFrameContract,
}
