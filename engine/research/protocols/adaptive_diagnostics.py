"""engine/research/protocols/adaptive_diagnostics.py — engine "应变" intelligence.

Senior-quant-AI engineering design for post-protocol failure analysis +
actionable recommendation. Inspects MultiLegVerdict + per-leg gate summaries
to detect 10+ canonical failure patterns and propose concrete fixes ranked
by impact/cost ratio.

Doctrine (per [[feedback-no-brittle-hardcoding-2026-05-30]] + [[feedback-
flexibility-rigor-balance-criterion-2026-05-30]]):
- Engine NEVER auto-modifies verdict — RED stays RED if the gate says so
- Engine SURFACES structured guidance instead of silent failure
- Recommendations are LAYERED:
    Layer 1 (this module): rule-based deterministic detectors
    Layer 2 (deferred to caller): empirical comparison runs
    Layer 3 (deferred to LLM): narrative synthesis
- Multi-evidence required: most detectors check ≥2 signals before firing
  (single-metric crossing threshold → high false-positive rate)
- Each Recommendation carries:
    - structured evidence list (auditable)
    - expected benefit range (Sharpe-t delta)
    - cost estimate (compute / dollars / wallclock)
    - confidence (Meta-Learner can calibrate from history)
    - falsification criterion (how to know the action didn't help)
    - alternative actions (ranked fallback)

Adding a new detector: write a function decorated with @register_detector(name).
No changes to existing code required.
"""
from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Taxonomy ────────────────────────────────────────────────────────────

class FailureCategory(enum.Enum):
    DATA_ISSUE       = "data_issue"        # insufficient / biased / missing
    SAMPLE_ISSUE     = "sample_issue"      # too short / regime-restricted
    UNIVERSE_ISSUE   = "universe_issue"    # cross-section too small / survivorship
    IMPLEMENTATION   = "implementation"    # cost / turnover / microstructure
    MECHANISM        = "mechanism"         # publication crowding / regime hostile
    STATISTICAL      = "statistical"       # DSR / multiple testing
    DESIGN           = "design"            # protocol family choice / decomposition


class Severity(enum.Enum):
    INFO  = "info"   # FYI; no action urgent
    WARN  = "warn"   # consider acting
    BLOCK = "block"  # dominant issue; address first


@dataclasses.dataclass
class CostEstimate:
    """Structured cost of taking the recommended action."""
    compute_minutes:  float     # additional run time
    dollars:          float     # API / WRDS cost
    wallclock_days:   float     # for data acquisition delays etc.

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class BenefitEstimate:
    """Predicted impact on the gate verdict."""
    sharpe_t_delta_lo: float    # low end of expected Sharpe-t change
    sharpe_t_delta_hi: float    # high end
    qualitative:       str      # "likely material" / "modest" / "marginal"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Recommendation:
    """A concrete, actionable suggestion with full audit trail."""
    pattern:                  str                 # detector name (registry key)
    category:                 str                 # FailureCategory.value
    severity:                 str                 # Severity.value
    action:                   str                 # specific instructions
    rationale:                str                 # why this would help
    evidence:                 list[str]           # specific data triggering this
    benefit:                  BenefitEstimate
    cost:                     CostEstimate
    confidence:               float               # 0-1
    falsification_criterion:  str                 # how to know it didn't help
    alternative_actions:      list[str]           # ranked fallbacks

    @property
    def impact_cost_ratio(self) -> float:
        """Used for ranking — bigger = higher priority."""
        midpoint_benefit = (self.benefit.sharpe_t_delta_lo
                              + self.benefit.sharpe_t_delta_hi) / 2.0
        # Approximate cost in compute-minute equivalents
        total_cost = max(
            self.cost.compute_minutes + self.cost.dollars * 60.0
                + self.cost.wallclock_days * 1440.0,
            0.01,
        )
        return midpoint_benefit / total_cost * self.confidence

    def to_dict(self) -> dict:
        return {
            "pattern":   self.pattern,
            "category":  self.category,
            "severity":  self.severity,
            "action":    self.action,
            "rationale": self.rationale,
            "evidence":  self.evidence,
            "benefit":   self.benefit.to_dict(),
            "cost":      self.cost.to_dict(),
            "confidence": self.confidence,
            "falsification_criterion": self.falsification_criterion,
            "alternative_actions":     self.alternative_actions,
            "impact_cost_ratio":       round(self.impact_cost_ratio, 4),
        }


# ── Detector registry (extensible — no enum edits) ──────────────────────

DetectorFn = Callable[[Any, dict], list[Recommendation]]
_DETECTORS: list[DetectorFn] = []


def register_detector(name: str):
    """Decorator: add a detector to the registry without code changes elsewhere."""
    def deco(fn: DetectorFn) -> DetectorFn:
        fn.detector_name = name
        _DETECTORS.append(fn)
        return fn
    return deco


# ── Helpers for evidence extraction ─────────────────────────────────────

def _leg_metric(verdict, leg_id: str, key: str):
    """Extract a metric from a specific leg's gate_summary."""
    for r in verdict.leg_results:
        if r.leg_id == leg_id and r.gate_summary:
            return r.gate_summary.get(key)
    return None


def _leg_sharpes(verdict) -> list[float]:
    return [r.gate_summary.get("standalone_sharpe")
              for r in verdict.leg_results
              if r.gate_summary and r.gate_summary.get("standalone_sharpe") is not None]


def _leg_alphas(verdict) -> list[float]:
    return [r.gate_summary.get("alpha_t_ff5umd")
              for r in verdict.leg_results
              if r.gate_summary and r.gate_summary.get("alpha_t_ff5umd") is not None]


# ── Detectors — each is a separate concern, no shared state ─────────────

@register_detector("sample_too_short")
def _detect_sample_too_short(verdict, context: dict) -> list[Recommendation]:
    """Multi-evidence: legs with 'sample too short' errors >= 1/3 of total."""
    short_legs = [r for r in verdict.leg_results
                    if r.error and "sample too short" in r.error.lower()]
    # Multi-evidence: need ≥2 short legs to call it systemic (1 = leg-specific)
    if len(short_legs) < 2:
        return []
    n_short = len(short_legs)
    template_id = context.get("template_id", "unknown")
    sample_months = context.get("sample_total_months", 0)
    warmup = context.get("template_warmup_months", 0)
    return [Recommendation(
        pattern="sample_too_short",
        category=FailureCategory.SAMPLE_ISSUE.value,
        severity=Severity.WARN.value,
        action=(
            f"Extend sample by ≥48 months. Current {sample_months}mo - {warmup}mo "
            f"warmup = {sample_months - warmup}mo effective; "
            f"sub-period split halves to ≤{(sample_months - warmup) // 2}mo, "
            f"below 24mo gate minimum."
        ),
        rationale=(
            f"Template {template_id} consumes {warmup}mo warmup. Sub-period "
            f"legs need ≥48mo each post-warmup for run_gate's 24mo floor + "
            f"meaningful Sharpe-t estimation."
        ),
        evidence=[
            f"{n_short}/{len(verdict.leg_results)} legs errored 'sample too short'",
            f"sample_total_months={sample_months}",
            f"template_warmup_months={warmup}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.3, sharpe_t_delta_hi=1.5,
            qualitative="likely material — would unlock 2+ legs currently failing"),
        cost=CostEstimate(compute_minutes=2.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.90,
        falsification_criterion=(
            "After extension, sub-period legs run without errors but still "
            "fail pass criteria → mechanism, not sample, is the issue."
        ),
        alternative_actions=[
            "Use generic_v1 protocol family (only 1 OOS leg, lower warmup penalty)",
            "Drop sub-period legs from family verdict_rule",
            "Use a template with smaller warmup (factor_quartile: 1mo + vol_lookback)",
        ],
    )]


@register_detector("universe_too_small")
def _detect_universe_too_small(verdict, context: dict) -> list[Recommendation]:
    """Cross-sectional template + universe < min_recommended_per_template."""
    template_id = context.get("template_id", "")
    universe_size = context.get("universe_size", 0)
    # min recommended — driven by typical decile L/S construction needs
    min_recommended_by_template = {
        "equity_xsmom":     200,
        "factor_quartile":  200,
        "cross_asset_tsmom":  6,   # TSMOM trades per-instrument; small OK
    }
    min_rec = min_recommended_by_template.get(template_id)
    if not min_rec or universe_size == 0 or universe_size >= min_rec:
        return []
    decile_size = universe_size // 10
    return [Recommendation(
        pattern="universe_too_small",
        category=FailureCategory.UNIVERSE_ISSUE.value,
        severity=Severity.WARN.value if universe_size > 30 else Severity.BLOCK.value,
        action=(
            f"Expand from {universe_size} to ≥{min_rec} tickers. "
            f"Specific options ranked by cost: "
            f"(1) Wikipedia SP500 scraper (free, ~500 tickers, <1min), "
            f"(2) yfinance with explicit liquid universe list (free, custom), "
            f"(3) WRDS CRSP DSF (paid, ~3000 tickers, requires auth)."
        ),
        rationale=(
            f"{template_id} ranks across universe + takes top/bottom deciles. "
            f"At {universe_size} tickers, decile = {decile_size} positions per "
            f"side. Noise dominates signal below {min_rec}."
        ),
        evidence=[
            f"template_id={template_id}",
            f"universe_size={universe_size}",
            f"min_recommended={min_rec}",
            f"decile_size={decile_size}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.5, sharpe_t_delta_hi=2.0,
            qualitative="material — noise reduction proportional to sqrt(n)"),
        cost=CostEstimate(compute_minutes=5.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.95,
        falsification_criterion=(
            "If after expansion Sharpe-t improves by <0.3, mechanism itself "
            "is weak and universe was not the binding constraint."
        ),
        alternative_actions=[
            "Test on a different template that doesn't need large cross-section "
            "(equity_tsmom is per-instrument, not cross-sectional)",
            "Restrict universe to a peer group + apply factor_quartile",
        ],
    )]


@register_detector("warmup_consumes_most_of_sample")
def _detect_high_warmup_ratio(verdict, context: dict) -> list[Recommendation]:
    """warmup-to-sample ratio > 40% — sub-period analyses degenerate."""
    warmup = context.get("template_warmup_months", 0)
    sample_total = context.get("sample_total_months", 0)
    if not sample_total or not warmup:
        return []
    ratio = warmup / sample_total
    if ratio < 0.4:
        return []
    template_id = context.get("template_id", "template")
    binding = context.get("binding", {}) or {}
    vol_lookback = binding.get("vol_target_lookback")
    return [Recommendation(
        pattern="warmup_consumes_most_of_sample",
        category=FailureCategory.DESIGN.value,
        severity=Severity.WARN.value,
        action=(
            f"Reduce warmup ratio (currently {ratio:.0%}). Ranked options: "
            f"(1) extend raw sample (cheap if data available), "
            f"(2) reduce vol_target_lookback in binding (current {vol_lookback or 'default'} → 12), "
            f"(3) disable vol_target entirely if not needed (binding vol_target: null), "
            f"(4) switch to factor_quartile (lower warmup) if signal can be precomputed."
        ),
        rationale=(
            f"Effective testable months = {sample_total - warmup} of {sample_total} raw. "
            f"For 5-leg protocol with sub-period halves, each leg works on "
            f"≤{(sample_total - warmup) // 2}mo. Run_gate needs ≥24mo."
        ),
        evidence=[
            f"warmup={warmup}mo",
            f"sample={sample_total}mo",
            f"ratio={ratio:.0%}",
            f"template_id={template_id}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.4, sharpe_t_delta_hi=1.2,
            qualitative="material — unlocks sub-period analysis"),
        cost=CostEstimate(compute_minutes=1.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.85,
        falsification_criterion=(
            "If after reducing vol_target_lookback the Sharpe goes negative, "
            "vol-targeting was actually necessary; revert."
        ),
        alternative_actions=[
            "Switch to generic_v1 protocol family with shorter leg requirements",
        ],
    )]


@register_detector("primary_pass_robustness_fail")
def _detect_overfit_fragile(verdict, context: dict) -> list[Recommendation]:
    """Primary leg passed but ALL robustness legs failed — overfit signature."""
    primary = next((r for r in verdict.leg_results if r.is_primary), None)
    if not primary or not primary.leg_passed:
        return []
    robustness = [r for r in verdict.leg_results if not r.is_primary]
    if not robustness:
        return []
    failed_robustness = [r for r in robustness if not r.leg_passed]
    if len(failed_robustness) < len(robustness):
        return []
    return [Recommendation(
        pattern="primary_pass_robustness_fail",
        category=FailureCategory.MECHANISM.value,
        severity=Severity.WARN.value,
        action=(
            "Mechanism overfits canonical spec. Do NOT deploy. "
            "Investigate which perturbation breaks it: "
            "(1) cost-stress: if 2x cost kills it, real net is fragile, "
            "(2) sub-period: if regime-specific, add regime overlay, "
            "(3) universe restriction: if microcap-driven, mechanism is a "
            "size/illiquidity premium not the claimed factor."
        ),
        rationale=(
            "All robustness legs failing with primary passing = canonical spec "
            "lives in a knife-edge. Standard overfitting signature; HXZ 2020 "
            "explicitly flags this as the dominant failure mode for published "
            "anomalies."
        ),
        evidence=[
            f"primary_passed=True",
            f"robustness_passed=0/{len(robustness)}",
            f"failed_legs={[r.leg_id for r in failed_robustness]}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=-0.5, sharpe_t_delta_hi=0.5,
            qualitative="diagnostic — investigation reveals true alpha vs overfit"),
        cost=CostEstimate(compute_minutes=10.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.85,
        falsification_criterion=(
            "If ONE specific perturbation accounts for ALL the failures, the "
            "mechanism may be valid OUTSIDE that constraint (still concerning)."
        ),
        alternative_actions=[
            "Mark mechanism RED in library with reason='overfit fragile'",
            "Generate diagnostic via #1 Research Diagnostician (Phase B)",
        ],
    )]


@register_detector("inverted_alpha_all_legs")
def _detect_inverted_alpha(verdict, context: dict) -> list[Recommendation]:
    """All legs show consistently NEGATIVE alpha-t — likely mechanism inverted."""
    alphas = _leg_alphas(verdict)
    if len(alphas) < 3:
        return []
    if not all(a < -0.5 for a in alphas):
        return []
    mean_alpha = sum(alphas) / len(alphas)
    return [Recommendation(
        pattern="inverted_alpha_all_legs",
        category=FailureCategory.MECHANISM.value,
        severity=Severity.INFO.value,
        action=(
            "Consistent NEGATIVE alpha — investigate regime inversion. DO NOT "
            "reverse the sign post-hoc (= overfitting). Instead: "
            "(1) check if mechanism inverted post-publication (regime change), "
            "(2) verify factor construction sign (e.g. low_vol_bab uses "
            "factor_sign=-1, low beta first), "
            "(3) compute inverse strategy with INDEPENDENT pre-registration."
        ),
        rationale=(
            f"All {len(alphas)} legs show alpha-t < -0.5 (mean {mean_alpha:.2f}). "
            f"Not random noise. Either (a) factor sign wrong in our binding, "
            f"(b) mechanism inverted in current regime, (c) sign-flip vs "
            f"canonical paper."
        ),
        evidence=[
            f"alpha-t per leg: {[round(a, 2) for a in alphas]}",
            f"mean_alpha_t={mean_alpha:.2f}",
            f"all_negative=True",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.0, sharpe_t_delta_hi=0.0,
            qualitative="diagnostic only — informs root cause"),
        cost=CostEstimate(compute_minutes=20.0, dollars=0.13, wallclock_days=0.0),
        confidence=0.80,
        falsification_criterion=(
            "If factor_sign verification shows construction matches paper "
            "AND inverse strategy fails too, mechanism is broken, not inverted."
        ),
        alternative_actions=[
            "Verify factor_sign in binding matches canonical paper",
            "Run #1 LLM Diagnostician for regime-change hypothesis",
            "Check mechanism_economics in library — does sign make sense?",
        ],
    )]


@register_detector("cost_stress_sensitivity")
def _detect_cost_binding(verdict, context: dict) -> list[Recommendation]:
    """cost_stress_2x leg has dramatically different Sharpe vs primary."""
    primary_sharpe = _leg_metric(verdict, "primary_test", "standalone_sharpe")
    stress_sharpe = _leg_metric(verdict, "cost_stress_2x", "standalone_sharpe")
    if primary_sharpe is None or stress_sharpe is None:
        return []
    if abs(primary_sharpe) < 0.05:    # too small to compare ratios
        return []
    drop = (primary_sharpe - stress_sharpe) / abs(primary_sharpe)
    if drop < 0.5:    # <50% drop is normal
        return []
    return [Recommendation(
        pattern="cost_stress_sensitivity",
        category=FailureCategory.IMPLEMENTATION.value,
        severity=Severity.WARN.value,
        action=(
            f"Cost is the binding constraint. Sharpe drops {drop:.0%} when "
            f"cost doubles. Real-world net likely much weaker than paper. "
            f"Try: (1) lower-turnover signal variant (12-1 → 12-3 skip), "
            f"(2) tilt-rebalance instead of full reconstitution monthly, "
            f"(3) cost-aware portfolio construction (Garleanu-Pedersen 2013)."
        ),
        rationale=(
            f"primary_sharpe={primary_sharpe:.3f}, "
            f"cost_stress_2x_sharpe={stress_sharpe:.3f}. Linear cost increase "
            f"causing super-linear Sharpe drop = non-linear cost surface; "
            f"likely turnover or position sizing exposed to round-trips."
        ),
        evidence=[
            f"primary_sharpe={primary_sharpe:.3f}",
            f"cost_stress_2x_sharpe={stress_sharpe:.3f}",
            f"drop_pct={drop:.0%}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=-0.2, sharpe_t_delta_hi=1.0,
            qualitative="modest — real-net stabilization, may not improve gross"),
        cost=CostEstimate(compute_minutes=5.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.75,
        falsification_criterion=(
            "If low-turnover variant doesn't improve net Sharpe, turnover "
            "wasn't the binding constraint; revisit signal definition."
        ),
        alternative_actions=[
            "Reduce rebal_freq from monthly to quarterly",
            "Apply tilt-rebalance with 20% turnover cap",
        ],
    )]


@register_detector("regime_concentration")
def _detect_regime_dependence(verdict, context: dict) -> list[Recommendation]:
    """First and second half Sharpes diverge sharply — regime-dependent."""
    fh_sharpe = _leg_metric(verdict, "subperiod_first_half", "standalone_sharpe")
    sh_sharpe = _leg_metric(verdict, "subperiod_second_half", "standalone_sharpe")
    if fh_sharpe is None or sh_sharpe is None:
        return []
    # Diverged if one is >2x the other AND they have different signs
    abs_lo, abs_hi = min(abs(fh_sharpe), abs(sh_sharpe)), max(abs(fh_sharpe), abs(sh_sharpe))
    if abs_hi < 2 * abs_lo and (fh_sharpe * sh_sharpe) >= 0:
        return []
    return [Recommendation(
        pattern="regime_concentration",
        category=FailureCategory.MECHANISM.value,
        severity=Severity.WARN.value,
        action=(
            f"Mechanism is regime-dependent. first_half_sharpe={fh_sharpe:.2f}, "
            f"second_half_sharpe={sh_sharpe:.2f}. Investigate: "
            f"(1) what changed between halves (regime / publication / crowding), "
            f"(2) consider regime-aware overlay (caution: see [[feedback-no-"
            f"regime-detection-in-book-2026-05-29]] — most regime overlays HURT), "
            f"(3) treat as 2 separate mechanisms with different valid windows."
        ),
        rationale=(
            f"Sub-period Sharpe divergence indicates non-stationary edge. "
            f"Likely causes: (a) post-publication decay (mechanism worked in "
            f"first half, arbitraged in second), (b) regime shift mid-sample "
            f"(e.g. 2008 GFC or 2020 COVID divides quality vs growth regimes)."
        ),
        evidence=[
            f"first_half_sharpe={fh_sharpe:.3f}",
            f"second_half_sharpe={sh_sharpe:.3f}",
            f"sign_flip={(fh_sharpe * sh_sharpe) < 0}",
            f"abs_ratio={abs_hi / max(abs_lo, 1e-6):.1f}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.0, sharpe_t_delta_hi=0.0,
            qualitative="diagnostic only — informs forward expectation"),
        cost=CostEstimate(compute_minutes=15.0, dollars=0.13, wallclock_days=0.0),
        confidence=0.80,
        falsification_criterion=(
            "If the divergent regime is structurally similar to current "
            "(e.g. both low-vol environments), divergence is noise; expect "
            "future similar to full-sample blend."
        ),
        alternative_actions=[
            "Mark mechanism with regime_dependent flag in library",
            "Deploy with vol-target reduction during opposite-regime periods",
            "Use #1 LLM Diagnostician for regime classification",
        ],
    )]


@register_detector("decomposition_contamination")
def _detect_pead_or_ff5_contamination(verdict, context: dict) -> list[Recommendation]:
    """Decomposition fails — alpha is just known factor exposure, not new."""
    failed_decomp = [d for d in verdict.decomp_results if not d.passed]
    if len(failed_decomp) < 2:    # need both to fail
        return []
    failed_ids = [d.check_id for d in failed_decomp]
    return [Recommendation(
        pattern="decomposition_contamination",
        category=FailureCategory.MECHANISM.value,
        severity=Severity.WARN.value,
        action=(
            f"Mechanism's alpha is absorbed by {', '.join(failed_ids)}. "
            f"Not a NEW edge — it's a re-skin of known factor exposure. "
            f"Don't deploy as a separate sleeve (would be doubling down on "
            f"existing book exposure). Possible value: (1) as a confirming "
            f"signal in book overlay, (2) as a low-conviction tilt within "
            f"another sleeve, (3) as a research dead-end (no deploy)."
        ),
        rationale=(
            f"Failed decomposition checks: {failed_ids}. After controlling "
            f"for FF5+UMD and/or PEAD, residual alpha is below 2.0 t-stat. "
            f"Per Hou-Xue-Zhang 2020, ~30% of published anomalies fail "
            f"orthogonality and are factor re-skins."
        ),
        evidence=[
            f"failed_decomposition_checks={failed_ids}",
            f"n_failed={len(failed_decomp)}",
        ],
        benefit=BenefitEstimate(
            sharpe_t_delta_lo=0.0, sharpe_t_delta_hi=0.0,
            qualitative="diagnostic only — saves a sleeve slot from redundancy"),
        cost=CostEstimate(compute_minutes=0.0, dollars=0.0, wallclock_days=0.0),
        confidence=0.90,
        falsification_criterion=(
            "If the mechanism shows independent post-pub OOS gains in samples "
            "WHERE the absorbing factor underperformed, the absorption may be "
            "in-sample collinearity, not true equivalence."
        ),
        alternative_actions=[
            "Mark mechanism RED with reason='decomposition contamination'",
            "Test as overlay on existing factor sleeve (not standalone)",
        ],
    )]


# ── Public API ──────────────────────────────────────────────────────────

def analyze_multi_leg_failure(
    verdict, *, context: dict | None = None
) -> list[Recommendation]:
    """Run all registered detectors against a MultiLegVerdict.

    Returns recommendations RANKED by impact_cost_ratio × severity weight.

    Args:
      verdict:  MultiLegVerdict from execute_protocol
      context:  dict of contextual info — detectors use what's relevant:
                  template_id, universe_size, sample_total_months,
                  template_warmup_months, binding, can_extend_sample, ...

    Returns: list[Recommendation], ranked.
    """
    context = context or {}
    out: list[Recommendation] = []
    for detector in _DETECTORS:
        try:
            recs = detector(verdict, context)
            out.extend(recs)
        except Exception as exc:
            logger.warning("detector %s raised: %s",
                            getattr(detector, "detector_name", "?"), exc)

    # Rank by impact_cost_ratio + severity tiebreaker
    severity_weight = {
        Severity.BLOCK.value: 3.0,
        Severity.WARN.value:  2.0,
        Severity.INFO.value:  1.0,
    }
    out.sort(
        key=lambda r: -(r.impact_cost_ratio * severity_weight.get(r.severity, 1.0))
    )
    return out


def format_recommendations(recommendations: list[Recommendation]) -> str:
    """Pretty-print for CLI output."""
    if not recommendations:
        return ("(no adaptive recommendations — no failure patterns detected; "
                "the verdict speaks for itself)")
    lines = []
    for i, r in enumerate(recommendations, 1):
        lines.append(f"#{i} [{r.severity.upper()}] {r.pattern} "
                       f"(category={r.category}, confidence={r.confidence:.0%})")
        lines.append(f"   ACTION:       {r.action}")
        lines.append(f"   RATIONALE:    {r.rationale}")
        lines.append(f"   EVIDENCE:     " + "; ".join(r.evidence))
        lines.append(f"   BENEFIT:      Sharpe-t ∈ [{r.benefit.sharpe_t_delta_lo:+.1f},"
                       f" {r.benefit.sharpe_t_delta_hi:+.1f}] ({r.benefit.qualitative})")
        lines.append(f"   COST:         {r.cost.compute_minutes:.0f}min + "
                       f"${r.cost.dollars:.2f} + {r.cost.wallclock_days:.1f}d wallclock")
        lines.append(f"   FALSIFY:      {r.falsification_criterion}")
        if r.alternative_actions:
            lines.append("   ALTERNATIVES:")
            for alt in r.alternative_actions:
                lines.append(f"     - {alt}")
        lines.append("")
    return "\n".join(lines)


def list_detectors() -> list[str]:
    return [getattr(d, "detector_name", d.__name__) for d in _DETECTORS]


def to_json_for_log(recommendations: list[Recommendation]) -> list[dict]:
    """Structured serialization for proposal_queue.jsonl / Meta-Learner."""
    return [r.to_dict() for r in recommendations]
