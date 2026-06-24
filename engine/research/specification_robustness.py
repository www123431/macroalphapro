"""engine.research.specification_robustness — B of A→B→C senior施工建议.

Tests parameter-sensitivity of a candidate factor by re-running the
template across a small neighborhood of B-class parameter values, then
reporting Sharpe stability across the neighborhood.

WHY THIS EXISTS
===============
Senior referee always asks: "did you cherry-pick the parameters?" A
factor that passes Sharpe gate at 12-1 momentum but dies at 11-1 is
overfit (Lo 2002 lesson; Asness 2017 QMJ + HXZ 2020 q-factor +
Bali-Engle-Murray 2016 all REQUIRE this report).

Pre-B: zero coverage. Tier C ran factors at ONE parameter setting,
no idea whether nearby settings give garbage.

MATHEMATICAL CONTRACT
====================
For each set B-class parameter on the FactorSpec, generate
neighborhood variants (derived from B_CLASS_RANGES steps — chosen to
mirror the ±1 / ±2 step convention in the literature). Re-run the
template at each variant, extract Sharpe. Report:

  stability_score = median(sharpes) / max(sharpes)

Bar (per senior施工建议 lock 2026-06-09):
  >= 0.60     → ROBUST
  0.40-0.60   → MARGINAL_OVERFIT (warning, no auto-demote)
  <  0.40     → LIKELY_OVERFIT   (recommend manual review)

DSR DISCLAIMER
==============
The neighborhood ablation does NOT count toward Bailey-LdP DSR
n_trials. The N variant runs are ROBUSTNESS CHECKS of one hypothesis,
NOT N independent hypotheses. self_doubt prompt must surface this
explicitly so the LLM doesn't double-penalize. lens output carries
`n_trials_increment: 0` to make this machine-readable.

SCOPE
=====
- Only fires when template_result.verdict ∈ {GREEN, MARGINAL}
  (RED is already rejected; ablation would waste compute)
- Variants bypass dispatcher gates (no WEEKLY_CAP / N_TRIALS / cost
  gate) — called directly via TEMPLATE_REGISTRY[signal_kind]
- Returns None when:
  * spec sets no B-class params (nothing to vary, default-template
    can't be tested)
  * template raises on a majority of variants
  * fewer than 3 successful variants (statistical floor for median)
"""
from __future__ import annotations

import dataclasses as _dc
import logging
import math
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants — locked 2026-06-09 per senior施工建议
# ────────────────────────────────────────────────────────────────────

# flex-4 (2026-06-10, F-gap-2 fix): the neighborhood is DERIVED from
# B_CLASS_RANGES — every B-class param the spec sets gets ±1·step
# and ±2·step variants (step is range metadata; bounds clip). The
# old hardcoded NEIGHBORHOOD_DELTAS param-name list is gone: it
# silently excluded universe_size, so the CGS-2008 run (n_buckets at
# range max + universe_size=2000) had no variation room and reported
# INSUFFICIENT_VARIATION. Confirmed F-gap-2 in session 7326492e.
NEIGHBORHOOD_STEP_MULTIPLES: tuple[int, ...] = (-2, -1, +1, +2)


def _derive_neighborhood_deltas() -> dict[str, tuple]:
    """±1·step, ±2·step per B-class numeric param. Enum params
    (B_CLASS_ENUMS) have no ordering → excluded."""
    from engine.agents.strengthener.factor_dispatcher import B_CLASS_RANGES
    out: dict[str, tuple] = {}
    for field, rule in B_CLASS_RANGES.items():
        step = rule.get("step")
        if step is None:
            continue
        deltas = tuple(m * step for m in NEIGHBORHOOD_STEP_MULTIPLES)
        if rule["type"] is float:
            deltas = tuple(round(d, 10) for d in deltas)
        out[field] = deltas
    return out

# Stability bars — locked
ROBUST_BAR     = 0.60
MARGINAL_BAR   = 0.40

# Minimum variants required to call median statistically meaningful.
# 3 = bare minimum for a tail-resistant median (one outlier doesn't
# dominate); below this we report INSUFFICIENT_VARIATION.
MIN_VARIANTS   = 3

# Verdicts gating the lens. ROBUSTNESS only meaningful on factors
# that already passed naive Sharpe gate.
GATED_VERDICTS = frozenset({"GREEN", "MARGINAL"})


# ────────────────────────────────────────────────────────────────────
# Neighborhood generator
# ────────────────────────────────────────────────────────────────────
def _b_class_clip(field: str, value: Any) -> Optional[Any]:
    """Clip a neighborhood variant value to the B_CLASS_RANGES bound,
    or None when it'd fall outside the safe range entirely (caller
    should skip that variant)."""
    from engine.agents.strengthener.factor_dispatcher import B_CLASS_RANGES
    if field not in B_CLASS_RANGES:
        return value
    rule = B_CLASS_RANGES[field]
    if value < rule["min"] or value > rule["max"]:
        return None
    return value


def build_neighborhood_specs(base_spec) -> list:
    """Generate FactorSpec variants for each set B-class param.

    Returns a list of FactorSpec instances (one per neighborhood
    variant). The BASE spec is NOT included — caller already has
    its template_result. Skips variants that fall outside
    B_CLASS_RANGES.
    """
    variants: list = []
    for field, deltas in _derive_neighborhood_deltas().items():
        base_value = getattr(base_spec, field, None)
        if base_value is None:
            continue   # param not set; template default would apply
        for delta in deltas:
            new_value = base_value + delta
            if isinstance(delta, float):
                new_value = round(new_value, 10)
            clipped = _b_class_clip(field, new_value)
            if clipped is None:
                continue
            try:
                variant = _dc.replace(base_spec, **{field: clipped})
            except (TypeError, ValueError) as exc:
                logger.debug("spec_robust: variant build failed for "
                                 "%s=%s: %s", field, new_value, exc)
                continue
            variants.append((field, base_value, clipped, variant))
    return variants


# ────────────────────────────────────────────────────────────────────
# Stats extraction from a TemplateResult
# ────────────────────────────────────────────────────────────────────
def _extract_sharpe(template_result) -> Optional[float]:
    """Best-effort Sharpe extraction. Templates that fail / error
    return None and the variant is dropped from stability stats."""
    if template_result is None:
        return None
    if template_result.verdict not in ("GREEN", "MARGINAL", "RED"):
        # INSUFFICIENT_HISTORY / DATA_ERROR / EXECUTION_ERROR etc.
        return None
    sharpe = (template_result.metrics or {}).get("sharpe")
    if sharpe is None or not math.isfinite(sharpe):
        return None
    return float(sharpe)


def _extract_t(template_result) -> Optional[float]:
    if template_result is None:
        return None
    if template_result.verdict not in ("GREEN", "MARGINAL", "RED"):
        return None
    t = (template_result.metrics or {}).get("nw_t_stat")
    if t is None or not math.isfinite(t):
        return None
    return float(t)


# ────────────────────────────────────────────────────────────────────
# Stability scoring + verdict mapping
# ────────────────────────────────────────────────────────────────────
def _verdict_from_score(score: Optional[float]) -> str:
    if score is None or not math.isfinite(score):
        return "UNTESTABLE"
    if score >= ROBUST_BAR:
        return "ROBUST"
    if score >= MARGINAL_BAR:
        return "MARGINAL_OVERFIT"
    return "LIKELY_OVERFIT"


# ────────────────────────────────────────────────────────────────────
# Main lens runner
# ────────────────────────────────────────────────────────────────────
def compute_specification_robustness(
    spec,
    template_fn: Callable,
    base_template_result,
) -> Optional[dict]:
    """Run neighborhood ablation on the spec's B-class params and
    report stability stats.

    Args:
      spec: the original FactorSpec
      template_fn: TEMPLATE_REGISTRY[signal_kind] — accepts a spec,
                   returns a TemplateResult
      base_template_result: the original template_result (used to seed
                            the base Sharpe / t-stat row)

    Returns dict with cells_tested + stability_score + verdict, or
    None when neighborhood is too sparse (< MIN_VARIANTS cells).
    """
    # ── 1. Gate on verdict ────────────────────────────────────────
    if base_template_result.verdict not in GATED_VERDICTS:
        return None

    # ── 2. Base stats ─────────────────────────────────────────────
    base_sharpe = _extract_sharpe(base_template_result)
    base_t      = _extract_t(base_template_result)
    if base_sharpe is None:
        return None

    # ── 3. Build neighborhood specs ───────────────────────────────
    neighborhood = build_neighborhood_specs(spec)
    if len(neighborhood) < MIN_VARIANTS:
        return {
            "status":             "INSUFFICIENT_VARIATION",
            "verdict":            "UNTESTABLE",
            "reason":             (f"only {len(neighborhood)} variants "
                                     f"buildable from B-class params; "
                                     f"need >= {MIN_VARIANTS}"),
            "base_sharpe":        base_sharpe,
            "base_t":             base_t,
            "neighborhood_size":  len(neighborhood),
            "n_trials_increment": 0,
        }

    # ── 4. Run variants ───────────────────────────────────────────
    cells: list = [{
        "label":          "base",
        "param_changed":  None,
        "param_value":    None,
        "sharpe":         base_sharpe,
        "nw_t_stat":      base_t,
        "verdict":        base_template_result.verdict,
    }]
    errors = 0
    for field, base_value, variant_value, variant_spec in neighborhood:
        try:
            variant_result = template_fn(variant_spec)
        except Exception as exc:
            logger.warning("spec_robust: template raised for %s=%s: %s",
                              field, variant_value, exc)
            cells.append({
                "label":         f"{field}={variant_value}",
                "param_changed": field,
                "param_value":   variant_value,
                "sharpe":        None,
                "nw_t_stat":     None,
                "verdict":       "EXECUTION_ERROR",
            })
            errors += 1
            continue
        v_sharpe = _extract_sharpe(variant_result)
        v_t      = _extract_t(variant_result)
        cells.append({
            "label":         f"{field}={variant_value}",
            "param_changed": field,
            "param_value":   variant_value,
            "sharpe":        v_sharpe,
            "nw_t_stat":     v_t,
            "verdict":       (variant_result.verdict if variant_result
                                else "EXECUTION_ERROR"),
        })

    # ── 5. Stability score ────────────────────────────────────────
    successful_sharpes = [c["sharpe"] for c in cells
                              if c["sharpe"] is not None]
    if len(successful_sharpes) < MIN_VARIANTS:
        return {
            "status":             "INSUFFICIENT_SUCCESSFUL_VARIANTS",
            "verdict":            "UNTESTABLE",
            "reason":             (f"only {len(successful_sharpes)} "
                                     "variants produced finite Sharpe"),
            "base_sharpe":        base_sharpe,
            "base_t":             base_t,
            "neighborhood_size":  len(neighborhood),
            "errors":             errors,
            "cells_tested":       cells,
            "n_trials_increment": 0,
        }

    sharpes_sorted = sorted(successful_sharpes)
    median = sharpes_sorted[len(sharpes_sorted) // 2]
    s_min  = sharpes_sorted[0]
    s_max  = sharpes_sorted[-1]
    score: Optional[float]
    if s_max > 0:
        score = median / s_max
    else:
        score = None   # all-negative neighborhood — UNTESTABLE
    verdict = _verdict_from_score(score)

    return {
        "status":             "COMPLETE",
        "verdict":            verdict,
        "stability_score":    float(score) if score is not None else None,
        "robust_bar":         ROBUST_BAR,
        "marginal_bar":       MARGINAL_BAR,
        "base_sharpe":        base_sharpe,
        "base_t":             base_t,
        "sharpe_median":      float(median),
        "sharpe_min":         float(s_min),
        "sharpe_max":         float(s_max),
        "neighborhood_size":  len(neighborhood),
        "successful_cells":   len(successful_sharpes),
        "errors":             errors,
        "cells_tested":       cells,
        # SENIOR DOCTRINE: ablation cells are ROBUSTNESS CHECKS of
        # one hypothesis — they do NOT inflate Bailey-LdP DSR n_trials.
        # self_doubt prompt renders this disclaimer per senior施工建议.
        "n_trials_increment": 0,
    }


# ────────────────────────────────────────────────────────────────────
# Lens registry declaration
# ────────────────────────────────────────────────────────────────────
def _runner_spec_robust(spec, template_result, prior_outputs):
    """Lens runner.

    1. Looks up the template function for the spec's signal_kind.
    2. Builds neighborhood spec variants.
    3. Re-runs the template on each (bypassing dispatcher gates since
       this is a robustness check, not a new dispatch).
    4. Reports stability stats.

    Returns None when the verdict is RED (gated) or the lookup fails.
    """
    # Gate on verdict — skip when factor already rejected
    if template_result.verdict not in GATED_VERDICTS:
        return None

    # Resolve template function. Lazy import to avoid dispatcher
    # ↔ research import cycle.
    try:
        from engine.agents.strengthener.factor_dispatcher import (
            TEMPLATE_REGISTRY,
        )
    except ImportError:
        return None
    template_fn = TEMPLATE_REGISTRY.get(spec.signal_kind)
    if template_fn is None:
        return None

    return compute_specification_robustness(
        spec, template_fn, template_result,
    )


def _build_lens_declaration():
    from engine.research.lens_registry import LensDeclaration
    return LensDeclaration(
        name             = "specification_robustness",
        version          = "v1.0_2026-06-09",
        applicable_to    = {
            # alpha + overlay only. Insurance/diversifier route to
            # Tier D and never reach this lens; overlay benefits from
            # the same robustness check (overlay sleeves are still
            # parameter-sensitive).
            "investment_role": ("alpha", "overlay"),
            # All asset classes — the math is asset-class-agnostic.
        },
        input_protocols  = (),   # reads template_result directly
        output_protocol  = "SpecificationRobustnessOutput",
        conditional_on   = None,    # gated inside runner (template
                                       # verdict, not lens output)
        fallback_chain   = (),
        output_schema    = {
            "primary":   "stability_score",
            "secondary": ("verdict", "sharpe_median",
                          "neighborhood_size", "cells_tested",
                          "n_trials_increment"),
        },
        consumed_by      = (),   # leaf lens
        runner           = _runner_spec_robust,
    )


LENS_DECLARATION = _build_lens_declaration()
