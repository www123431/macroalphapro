"""engine.agents.strengthener.self_doubt — Tier C L3-2 Self-Doubt module.

After dispatcher runs a template and produces verdict, this module
asks Sonnet to score the system's confidence in the result + list
specific caveats. Surfaces in event metrics + (Tier E) /approvals
UI as the "don't trust me too much" signal.

PHILOSOPHY (per docs/spec_tier_c_layer_2_3_roadmap.md §A.5 meta-design)
======================================================================
> "If your system is producing too many confident answers, the
>  system has a bug. Reality is noisier than that." — DE Shaw

Tier C produces GREEN/RED verdicts that LOOK professional. Without
explicit self-doubt, the principal will trust them too much. Each
verdict carries known silent-bug risk that VARIES per dispatch:
  - Did we use the right cohort? (B-fix issues)
  - Is this verdict in a fresh post-pub window? (McLean-Pontiff)
  - Has this mechanism class hit n_trials caution?
  - Are the metrics suspiciously CLEAN (e.g. t > paper t after
    post-pub period)?
  - Did replication MATCH or MISMATCH?

The L3-2 LLM call SCORES these per-verdict + reports caveats. UI
displays confidence + caveats prominently to FORCE the principal
to engage with them before accepting the verdict.

PATTERN-5-COMPLIANT
===================
Single Sonnet call per dispatch, strict JSON tool_use schema,
NO multi-agent debate, NO chain-of-thought iteration. Same shape
as factor_spec_extractor + strengthener_review (both proven safe).

COST
====
~$0.04 per verdict-having dispatch. Adds ~2x to Tier C per-dispatch
cost ($0.03 extract + $0.04 self-doubt = $0.07 total). Acceptable
given the trust-builder value.

GRACEFUL DEGRADATION
====================
LLM call failure → returns None → dispatcher just doesn't include
self_doubt in event metrics. NOT a hard blocker on verdict emit.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import logging
import math
from typing import Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Output dataclass — what self_doubt produces
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class SelfDoubtAssessment:
    """LLM self-assessment of system confidence in a verdict.

    Frozen — once recorded, doesn't change. Stored in
    factor_verdict_filed event metrics under key 'self_doubt'.
    """
    confidence:               float        # 0.0 - 1.0
    confidence_reason:        str          # 1-2 sentence summary
    caveats:                  tuple[str, ...]   # 2-5 specific concerns
    methodological_concerns:  tuple[str, ...]   # known silent bugs cited
    suspicious_metrics:       tuple[str, ...]   # specific numbers questioned
    assessment_ts:            str
    model:                    str


# ────────────────────────────────────────────────────────────────────
# Known silent-bug surface — injected into Sonnet's context
# ────────────────────────────────────────────────────────────────────
_KNOWN_SILENT_BUGS = """\
TIER C ARCHITECTURAL SILENT BUGS (as of 2026-06-08)
====================================================
Use this list to ground your caveats — cite SPECIFIC bugs when
they may affect this verdict.

B0 (BIG one, MOSTLY FIXED via Phase 1.5/1.6 bitemporal):
  comp_pit.funda PIT cache uses real comp.fundq.rdq for 77.2% of
  rows. Remaining 22.8% use fallback datadate+120d approximation.
  For fundamental signals (gp_at, book_to_market, at_growth, roe),
  ~3% of (gvkey, datadate) rows have >5% restatement diff vs
  latest-restated legacy data. AAPL alone shows 5.95% on `at`.

B1 (FIXED, commit 328154dc):
  Universe selection now uses lagged mktcap (shift 1 month) —
  same-month look-ahead removed.

B2 (PARTIAL): No survivorship-aware universe construction for
  us_equities_top_3000. Top-3000 by current mktcap may include
  firms that became investable later. PIT SP500 constituents
  (us_equities_sp500) IS survivor-bias-free.

B3 (FIXED via L2-3 multi-cost stress):
  Multi-cost verdict reported at 0/30/60/80bp. Headline verdict
  uses STRICTER of naive 13bp and 80bp.

B4 (NOT FIXED): EW-only L/S reporting. Academic standard reports
  EW + VW. Cross-sec template only does EW. Asness flagged "EW
  quintile L/S is a paper-tiger" — alpha typically halves on VW.

B5 (PARTIAL): n_trials tracked per family. Mechanism-class level
  aggregation not yet enforced (testing GP/A + ROE = 2 PROFITABILITY
  trials, but actually 2 mechanism-class "profitability ratio"
  trials sharing same hypothesis space).

B6 (NOW COMPUTED, RECORD-ONLY 2026-06-09): Anchor library
  orthogonality IS now checked against Ken French FF5 + MOM via
  OLS residual-α regression with Newey-West HAC SE. When the
  input includes an `anchor_orthogonality` block, USE IT to
  identify spanning. Verdict is NOT yet auto-demoted on residual
  α failure; the principal needs visibility BEFORE enforcement.

B7 (KNOWN scope limit): ROE uses annual ni/ceq; HXZ q-factor
  uses quarterly ROE-q. Our annual proxy is structurally weaker.

REPLICATION CAVEATS (per L2-2):
  Verdict status MISMATCH = our implementation differs from
  paper benchmark by > 0.5 t-stat. Always investigate WHY before
  trusting the verdict.

POST-PUBLICATION DECAY (McLean-Pontiff 2016):
  Published factors typically lose 32-58% of in-sample Sharpe
  post-publication. If our full-window t-stat is HIGHER than the
  paper-reported t in overlap window, something is suspicious —
  decay should make t LOWER not higher.
"""


# ────────────────────────────────────────────────────────────────────
# Sonnet prompt + tool schema
# ────────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = f"""\
You are the SELF-DOUBT module for Tier C, an automated factor-
backtest dispatcher. Your job is to score the system's confidence
in a single verdict AND list specific caveats that should make
the principal hesitate before acting on it.

You are NOT a cheerleader. Your job is to find reasons to DOUBT.
GREEN verdicts that look clean deserve EXTRA scrutiny — the
system's tendency is toward false confidence.

INPUT
=====
You receive:
  1. The FactorSpec the dispatcher ran (signal_kind / universe /
     dates / B-class params / paper_original_window etc.)
  2. The TemplateResult verdict + metrics (Sharpe / NW t-stat /
     cost_stress at 4 levels / drawdown / replication status)
  3. The headline verdict (GREEN / MARGINAL / RED)
  4. The list of KNOWN architectural silent bugs (see below)

KNOWN SILENT BUGS (cite these specifically when relevant):
{_KNOWN_SILENT_BUGS}

OUTPUT
======
Invoke the emit_self_doubt tool EXACTLY ONCE with:
  - confidence: 0.0-1.0 float (NEVER 1.0 — system always has SOME
    uncertainty; max realistic is ~0.85 for replicated + stress-
    robust + post-pub-active GREEN; typical 0.4-0.7)
  - confidence_reason: 1-2 sentences summarizing the score
  - caveats: 2-5 SPECIFIC concerns for THIS verdict (cite numbers
    from the metrics, not generic boilerplate)
  - methodological_concerns: subset of known silent bugs that
    affect THIS verdict (e.g. "B4 EW-only: alpha likely halves
    if reported VW"); empty list if none clearly apply
  - suspicious_metrics: specific metric values that look wrong
    or too clean (e.g. "Sharpe=0.84 is HIGHER than paper-reported
    0.6; McLean-Pontiff post-pub decay should make it LOWER")

CALIBRATION RUBRIC FOR CONFIDENCE
=================================
0.80-0.85: GREEN + REPLICATED gap < 0.1 + cost-robust at 80bp +
            drawdown < 30% + no methodological concerns apply +
            n_trials in family < CAUTION threshold (7) +
            residual α t-stat >= 1.96 (NOT spanned by anchors) +
            subsample worst/best Sharpe ratio >= 0.40
            (stable across regimes)
0.55-0.75: GREEN + at least 1 of: replication gap 0.1-0.5,
            cost-marginal at 80bp, drawdown 30-50%,
            residual α t-stat 1.65-1.96,
            subsample stable but not exceptional
0.35-0.55: GREEN but >=2 concerns OR MARGINAL verdict OR
            residual α t-stat 1.0-1.65 (partial spanning) OR
            subsample worst/best ratio < 0.40 (regime-dependent) OR
            empirical post-pub decay 32-58% (McLean-Pontiff zone)
0.15-0.35: RED verdict OR
            residual α t-stat < 1.0 with headline t > 1.96
            (factor is mostly RMW/HML/etc restatement) OR
            one-window concentration > 60% (single regime drove
            the headline) OR
            empirical post-pub decay > 50%
0.05-0.15: data-error / insufficient-history / replication MISMATCH

ANCHOR-ORTHOGONALITY RULE (L2-4, MANDATORY when block present)
==============================================================
When the input includes an `anchor_orthogonality` block (Tier C
L2-4 output: residual α regression vs Ken French FF5 + MOM):

  1. If |headline_t − residual_t| >= 1.5 → factor is PARTIALLY
     SPANNED by canonical risk premia. Caveat MUST name the
     dominant beta-loadings (|β NW t| > 1.96) and explain in
     plain English: "factor borrows ~X% of its t-stat from
     [anchor list]; the unique residual α is Y annual".

  2. If residual α t-stat < 1.96 while headline t > 1.96 → the
     GREEN verdict is HXZ-2020-class (one of the 65% that fails
     against q-factor model). knock confidence DOWN by ≥ 0.20.
     Caveat must say something like "GREEN verdict survives
     headline test but not anchor-residual test; this is
     restatement of [dominant beta] not novel alpha".

  3. If R² > 0.50 → anchors explain over half the factor's
     variance; even when residual α is significant, demand
     the principal answer "why allocate to this rather than
     just buying [dominant beta loadings]?". Add as
     methodological_concern.

  4. If no anchor_orthogonality block (data missing / regression
     failed) → caveat about INVISIBILITY: "anchor-orthogonality
     check did not run; factor's loading on FF5+MOM unknown;
     headline t-stat may be inflated."

  5. (L2-4 Stage 1) If `gross` sub-block present AND gross-vs-net
     residual α t-stat delta > 1.0 → COST is doing more of the
     spanning work than anchors. Caveat: "factor's pre-cost
     residual α (gross t-stat=X) clears 1.96, but post-13bp net
     residual α (t-stat=Y) does not — implementation is gated by
     execution cost more than by mechanism redundancy."

  6. (L2-4 Stage 1) If `joint_loading_f_test` p-value < 0.01 →
     orthogonality JOINTLY REJECTED. Even when individual β t-stats
     look modest, the panel is collectively significant. Caveat:
     "anchor panel jointly explains the factor (F-test p=X);
     orthogonality is a non-starter."

  7. (L2-4 Stage 1) If anchor_snapshot_sha is missing → caveat
     about PROVENANCE: "anchor library was not pinned; this
     verdict is not reproducible if Ken French data is revised."

SUBSAMPLE-STABILITY RULE (L2-5, MANDATORY when block present)
=============================================================
When the input includes a `subsample_stability` block (Tier C
L2-5 output: N-split decomposition of factor PnL):

  1. If worst/best Sharpe ratio < 0.40 OR institutional_stable
     is False → factor is REGIME-DEPENDENT, not a stable alpha.
     Caveat MUST identify WHICH window dominates and quantify
     "X% of the t-stat comes from [single window]; remaining
     windows contribute little." Knock confidence DOWN by
     ≥ 0.15.

  2. If monotone_decay is True OR empirical post-pub decay
     >= 32% → McLean-Pontiff 2016 empirical signature is
     present. Caveat MUST say the verdict's headline t-stat
     reflects PAST regime; OOS expectation should be discounted
     30-50% per McLean-Pontiff range. Knock confidence DOWN
     by ≥ 0.15 ADDITIONALLY (compounds with #1 if both apply).

  3. If monotone_growth is True → factor's Sharpe getting
     STRONGER over time is structurally suspicious — could be
     non-stationary trend (sector boom, regime shift, look-
     ahead leakage). Caveat MUST flag this as "non-causal
     trend candidate, requires investigation."

  4. If any sub-window has NEGATIVE Sharpe while overall is
     GREEN → crisis-survivability concern. Caveat MUST name
     the bad window AND assess whether the bad period maps
     to a known macro event (2008, 2020, 2022).

  5. If no subsample_stability block (insufficient months or
     PnL missing) → caveat about INVISIBILITY: "subsample
     decomposition not run; one-window-dominance and
     post-pub decay unknown."

INDUSTRY-EXTENSION RULE (L2-6 Commit 3, MANDATORY when block present)
======================================================================
When the input includes an `industry_extension` block (Tier C
L2-6 JOINT model: factor PnL ~ FF5+MOM ∪ 12-Industry):

Per [[feedback-fwl-sequential-residual-trap-2026-06-09]]: α is from
the JOINT model. Compare to FF5+MOM-only α via Δα. Two distinct
patterns surface here:

  1. **GP/A pattern — INDUSTRIES ABSORB ALPHA**:
     Δα t-stat > 1.5  AND  industry-subset F p < 0.01
     → adding the industry panel materially eroded the alpha.
     The factor's "unique" alpha was partly an industry tilt.
     Caveat MUST name the top 2 industry loadings (|t|>1.96) and
     state in plain English: "after adding 12-Industry to the
     anchor panel, residual α dropped from X to Y; β_<Industry>
     = Z (t=W) indicates the factor is partly a <industry>
     tilt rather than novel alpha." Knock confidence DOWN by ≥0.10.

  2. **PIT-SN pattern — INDUSTRIES DON'T ABSORB**:
     Δα t-stat ≤ 0  OR  α_full ≥ α_FF5MOM
     → industries provide explanation independent of alpha;
     joint α stayed strong or strengthened. This is GENUINE
     alpha that survives 17-factor joint orthogonalization.
     Caveat should REINFORCE this finding: "α_full = X (t=Y)
     after joint 18-factor model; industries do NOT absorb
     the alpha (Δα = Z). Industry F p = W indicates industries
     have explanatory power but it is orthogonal to the alpha."
     Confidence should NOT be further knocked down on this lens.

  3. If joint R² > 0.65 → very large fraction of factor variance
     is explained by 18-factor model. EVEN IF α_full is
     significant, demand: "is the residual α stable enough to
     allocate to, or is the factor mostly a derived combination
     of canonical exposures?"

  4. If no industry_extension block → caveat about INVISIBILITY:
     "industry-extended joint model did not run; Δα unknown;
     factor's spanning by industry panel is unmeasured."

CROSS-ASSET MACRO RULE (Cross-asset lite, MANDATORY when block present)
========================================================================
When the input includes a `cross_asset_extension` block, the verdict
has been tested against an additional macro regime panel (5 FRED
variables: VIX_change, DXY_return, BAA_spread_change, T10Y3M_change,
T10YIE_change). This is the STRICTEST joint model in the rigor
stack (up to 23 regressors).

  1. If α_full NW-t (from this block) drops > 1.5 t-units below
     the Stage 1 FF5+MOM-only NW-t → factor has KNOWN MACRO
     REGIME exposure that absorbed some alpha. Caveat MUST name
     the top 2 macro loadings (|β NW-t| > 1.96) and explain in
     plain English: "factor's apparent alpha is partly compensation
     for [macro regime] exposure (e.g., funding stress, carry
     crash, vol regime)". Knock confidence DOWN by ≥ 0.05.

  2. If α_full NW-t < 1.65 while Stage 1 NW-t > 1.96 → the alpha
     is FULLY EXPLAINED by combined FF5+MOM + Industry + Macro
     panel. This is the CROSS-ASSET version of the GP/A pattern.
     Caveat MUST say "alpha fails the 23-regressor kitchen-sink
     test; no novel return after canonical risk + industry +
     macro regime spanning". Knock confidence DOWN by ≥ 0.15.

  3. If α_full NW-t ≥ 1.96 AND macro F p < 0.01 → factor has
     SIGNIFICANT MACRO EXPOSURE but ALPHA SURVIVES. This is the
     PIT-SN / CARRY pattern — genuine alpha with acknowledged
     macro tilts. Caveat should REINFORCE: "factor survives the
     full 23-regressor model; macro exposure is documented but
     does not invalidate the alpha." Do NOT knock confidence.

  4. If macro F p > 0.10 → factor is orthogonal to macro regime.
     No further action; this is fine.

  5. If no cross_asset_extension block → caveat about INVISIBILITY:
     "macro regime extension did not run; factor's exposure to
     VIX/DXY/credit/term/breakeven regimes unmeasured."

NEVER cheerlead. NEVER copy-paste boilerplate. ALWAYS cite specific
numbers + specific known-bug IDs. The anchor-orthogonality block
exists precisely so you can ground every caveat in measured loadings.
"""


_TOOL_SCHEMA = {
    "name": "emit_self_doubt",
    "description": ("Emit a structured self-doubt assessment for a "
                    "single Tier C verdict. Score system confidence "
                    "0-1 + list specific caveats."),
    "input_schema": {
        "type": "object",
        "properties": {
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.99,   # NEVER 1.0 — system always has some uncertainty
            },
            "confidence_reason": {
                "type": "string",
                "minLength": 30,
                "maxLength": 400,
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string", "minLength": 20,
                           "maxLength": 300},
                "minItems": 2,
                "maxItems": 5,
            },
            "methodological_concerns": {
                "type": "array",
                "items": {"type": "string", "minLength": 10,
                           "maxLength": 300},
                "minItems": 0,
                "maxItems": 5,
            },
            "suspicious_metrics": {
                "type": "array",
                "items": {"type": "string", "minLength": 15,
                           "maxLength": 300},
                "minItems": 0,
                "maxItems": 5,
            },
        },
        "required": ["confidence", "confidence_reason", "caveats",
                      "methodological_concerns", "suspicious_metrics"],
        "additionalProperties": False,
    },
}


# ────────────────────────────────────────────────────────────────────
# User message assembly
# ────────────────────────────────────────────────────────────────────
def _format_user_message(spec, template_result, family_hint: str,
                            n_trials_family: int,
                            anchor_orthogonality: Optional[dict] = None,
                            subsample_stability:  Optional[dict] = None,
                            industry_extension:   Optional[dict] = None,
                            cross_asset_extension: Optional[dict] = None,
                            specification_robustness: Optional[dict] = None,
                            pnl_diagnostics:      Optional[dict] = None,
                            routing_decisions:    Optional[list] = None) -> str:
    """Render the spec + template_result for Sonnet's input.
    Keep it tight — only the fields that bear on doubt assessment."""
    m = template_result.metrics or {}
    parts = [
        "VERDICT TO ASSESS",
        "==================",
        f"  hypothesis_id:    {spec.hypothesis_id}",
        f"  signal_kind:      {spec.signal_kind}",
        f"  universe:         {spec.universe}",
        f"  date_range:       {spec.date_range}",
        f"  family_hint:      {family_hint}",
        f"  n_trials_family:  {n_trials_family}",
        "",
        "B-CLASS PARAMETERS (None = template default used):",
        f"  universe_size:        {spec.universe_size}",
        f"  n_buckets:            {spec.n_buckets}",
        f"  signal_lookback_m:    {spec.signal_lookback_m}",
        f"  signal_skip_m:        {spec.signal_skip_m}",
        f"  vol_target_annual:    {spec.vol_target_annual}",
        f"  weighting_scheme_alt: {spec.weighting_scheme_alt}",
        "",
        "ROLE-AWARE ROUTING AXES (Phase 1, None = legacy fallback inferred)",
        f"  investment_role:      {getattr(spec, 'investment_role', None)}",
        f"  statistical_role:     {getattr(spec, 'statistical_role', None)}",
        f"  asset_class:          {getattr(spec, 'asset_class', None)}",
        f"  mechanism:            {getattr(spec, 'mechanism', None)}",
        f"  horizon:              {getattr(spec, 'horizon', None)}",
        f"  capacity_tier:        {getattr(spec, 'capacity_tier', None)}",
        f"  data_dependency_type: {getattr(spec, 'data_dependency_type', None)}",
        f"  regime_sensitivity:   {getattr(spec, 'regime_sensitivity', None)}",
        "",
        "TEMPLATE RESULT",
        "===============",
        f"  HEADLINE VERDICT:  {template_result.verdict}",
        f"  summary:           {template_result.summary[:200]}",
    ]
    # Key metrics
    for key in ("sharpe", "nw_t_stat", "ann_return", "ann_vol",
                  "n_months", "avg_turnover", "naive_verdict",
                  "cost_robust_verdict"):
        if key in m and m[key] is not None:
            v = m[key]
            if isinstance(v, float):
                parts.append(f"  {key}: {v:.4f}")
            else:
                parts.append(f"  {key}: {v}")
    # Cost stress
    cs = m.get("cost_stress") or {}
    if cs:
        parts.append("")
        parts.append("COST STRESS (per-level Sharpe + t):")
        for level, s in cs.items():
            if s and s.get("sharpe") is not None:
                parts.append(
                    f"  {level}: Sharpe={s['sharpe']:.3f}, "
                    f"t={s['nw_t_stat']:.3f}, verdict={s['verdict']}")
    # Drawdown
    dd = m.get("drawdown_naive") or {}
    if dd:
        parts.append("")
        parts.append("DRAWDOWN (naive 13bp cost):")
        for key in ("max_drawdown_pct", "max_underwater_months",
                      "calmar_ratio"):
            v = dd.get(key)
            if v is not None:
                if isinstance(v, float):
                    parts.append(f"  {key}: {v:.4f}")
                else:
                    parts.append(f"  {key}: {v}")
    # Replication
    rep = m.get("replication") or {}
    if rep and rep.get("status"):
        parts.append("")
        parts.append("REPLICATION vs paper baseline:")
        parts.append(f"  status:    {rep.get('status')}")
        parts.append(f"  our_t:     {rep.get('our_t')}")
        parts.append(f"  paper_t:   {rep.get('paper_reported_t')}")
        parts.append(f"  t_gap:     {rep.get('t_gap')}")

    # L2-4 ANCHOR-ORTHOGONALITY — anchor library renders dynamically.
    # B.1 (2026-06-09): FX-carry sleeves run fx_carry_anchor_regression
    # (LRV HML_FX + DOL) instead of FF5+MOM; the output shape is
    # identical but the panel of anchor names + library tag differ.
    # Rendering picks the right header from `anchor_library` so the
    # LLM sees "LRV HML_FX + DOL" for FX carry and "Ken French FF5+MOM"
    # for equity, with no other code path changes.
    if anchor_orthogonality:
        ao = anchor_orthogonality
        _lib = (ao.get("anchor_library") or "").lower()
        if _lib.startswith("lrv_fx_carry"):
            _hdr = "FX-CARRY ANCHOR ORTHOGONALITY (LRV HML_FX + DOL residual α)"
        else:
            _hdr = "ANCHOR-ORTHOGONALITY (Ken French FF5+MOM residual α)"
        parts.append("")
        parts.append(_hdr)
        parts.append("=" * 55)
        parts.append(f"  anchor library:        {ao.get('anchor_library', '?')}")
        parts.append(f"  overlap window:        {ao.get('window', '?')}")
        parts.append(f"  n_overlap (months):    {ao.get('n_overlap', '?')}")
        parts.append(f"  R²:                    {ao.get('r2', float('nan')):.4f}")
        parts.append(f"  R² adj:                {ao.get('r2_adj', float('nan')):.4f}")
        parts.append(f"  residual α monthly:    {ao.get('alpha_monthly', float('nan'))*100:+.4f}%")
        parts.append(f"  residual α annual:     {ao.get('alpha_annual', float('nan'))*100:+.3f}%")
        parts.append(f"  residual α NW t-stat:  {ao.get('alpha_nw_t', float('nan')):+.3f}")
        parts.append(f"  residual α NW SE:      {ao.get('alpha_nw_se', float('nan'))*100:.4f}%/mo")
        parts.append("")
        parts.append("  anchor loadings (β, NW t-stat):")
        betas = ao.get("betas", {}) or {}
        beta_t = ao.get("beta_nw_t", {}) or {}
        for k in ao.get("anchor_names", []) or []:
            b = betas.get(k, float("nan"))
            t = beta_t.get(k, float("nan"))
            sig = ""
            if math.isfinite(t):
                sig = ("***" if abs(t) > 2.58 else
                       ("**"  if abs(t) > 1.96 else
                        ("*"   if abs(t) > 1.65 else "")))
            parts.append(f"    {k:8s}  β={b:+.4f}  t={t:+.3f}  {sig}")
        # Compute headline-vs-residual gap for fast LLM grok
        headline_t = m.get("nw_t_stat")
        if headline_t is not None and math.isfinite(ao.get("alpha_nw_t", float("nan"))):
            gap = abs(float(headline_t) - ao["alpha_nw_t"])
            parts.append("")
            parts.append(f"  HEADLINE vs RESIDUAL t-stat gap: {gap:.2f} units")
            parts.append("  (gap > 1.5 → factor partially spanned by anchors)")
        # L2-4 Stage 1: GROSS regression (apples-to-apples vs Ken French)
        gross = ao.get("gross")
        if gross:
            parts.append("")
            if _lib.startswith("lrv_fx_carry"):
                parts.append("  GROSS PnL regression (vs gross LRV FX-carry anchors):")
            else:
                parts.append("  GROSS PnL regression (vs gross Ken French anchors):")
            parts.append(f"    residual α NW t-stat:  "
                           f"{gross.get('alpha_nw_t', float('nan')):+.3f}")
            parts.append(f"    residual α annual:     "
                           f"{gross.get('alpha_annual', float('nan'))*100:+.3f}%")
            parts.append(f"    R²:                    "
                           f"{gross.get('r2', float('nan')):.4f}")
            # Gross vs net delta — pure cost erosion vs anchor effect
            net_t   = ao.get("alpha_nw_t", float("nan"))
            gross_t = gross.get("alpha_nw_t", float("nan"))
            if math.isfinite(net_t) and math.isfinite(gross_t):
                delta = gross_t - net_t
                parts.append(f"    GROSS vs NET delta:    "
                               f"{delta:+.3f} t-stat units")
                parts.append("    (large positive delta → cost erodes alpha "
                               "MORE than anchors do)")
        # L2-4 Stage 1: GRS-style joint F-test on loadings
        jf = ao.get("joint_loading_f_test")
        if jf:
            parts.append("")
            parts.append(f"  JOINT F-test H0: all β = 0")
            parts.append(f"    F-stat:    {jf.get('f_stat', float('nan')):.3f}")
            parts.append(f"    p-value:   {jf.get('f_pvalue', float('nan')):.4g}")
            parts.append(f"    df:        ({jf.get('df_num','?')}, "
                           f"{jf.get('df_denom','?')})")
            parts.append("    (p < 0.01 → factor STRONGLY loads on at least "
                           "one anchor; orthogonality REJECTED)")
        # L2-4 Stage 1: anchor snapshot pinning for reproducibility
        sha = ao.get("anchor_snapshot_sha")
        if sha:
            parts.append(f"  anchor snapshot SHA-256:  {sha[:12]}... "
                           f"(pinned for reproducibility)")
    else:
        parts.append("")
        parts.append("ANCHOR-ORTHOGONALITY: not computed for this verdict")
        parts.append("(check raised, anchor library missing, or insufficient overlap)")

    # L2-5 SUBSAMPLE STABILITY (n-split decomposition of factor PnL)
    if subsample_stability:
        ss = subsample_stability
        parts.append("")
        parts.append("SUBSAMPLE STABILITY (N-split decomposition)")
        parts.append("=" * 55)
        parts.append(f"  n_splits:                {ss.get('n_splits', '?')}")
        parts.append(f"  n_total_months:          {ss.get('n_total_months', '?')}")
        wb = ss.get("worst_best_sharpe_ratio")
        parts.append(f"  worst/best Sharpe ratio: "
                       f"{wb:.3f}" if wb is not None else
                       "  worst/best Sharpe ratio: N/A (best <= 0)")
        parts.append(f"  institutional_stable:    {ss.get('institutional_stable', '?')} "
                       f"(bar: worst/best >= 0.40 AND worst > 0)")
        parts.append(f"  monotone_decay:          {ss.get('monotone_decay', '?')} "
                       "(each split's Sharpe < prior)")
        parts.append(f"  monotone_growth:         {ss.get('monotone_growth', '?')} "
                       "(suspicious — possible non-stationary trend)")
        ds = ss.get("decay_slope_per_year")
        dt = ss.get("decay_slope_t")
        if ds is not None and dt is not None:
            parts.append(f"  decay slope:             {ds*100:+.4f}% / year")
            parts.append(f"  decay slope NW t-stat:   {dt:+.3f}")
        parts.append("")
        parts.append("  Per-window breakdown:")
        for w in ss.get("windows", []) or []:
            sharpe = w.get("sharpe_ann")
            tstat  = w.get("nw_t_stat")
            sharpe_str = (f"{sharpe:+.3f}" if sharpe is not None
                            else "    N/A")
            t_str = (f"{tstat:+.2f}" if tstat is not None
                       else "  N/A")
            parts.append(
                f"    {w.get('start','?')}->{w.get('end','?')}: "
                f"n={w.get('n_months','?')}mo  "
                f"Sharpe={sharpe_str}  NW-t={t_str}"
            )
        # Pre-pub vs post-pub crude split (first half vs second half
        # of Sharpe windows) — McLean-Pontiff 2016 quick check.
        sharpes = [w.get("sharpe_ann") for w in ss.get("windows", [])
                     if w.get("sharpe_ann") is not None]
        if len(sharpes) >= 4:
            half = len(sharpes) // 2
            first = sum(sharpes[:half]) / half
            second = sum(sharpes[half:]) / (len(sharpes) - half)
            if first > 0:
                decay = (1 - second / first) * 100
                parts.append("")
                parts.append(f"  Pre-pub avg Sharpe (windows 1..{half}): {first:+.3f}")
                parts.append(f"  Post-pub avg Sharpe (windows {half+1}..N): {second:+.3f}")
                parts.append(f"  Empirical decay: {decay:+.1f}%")
                parts.append("  (McLean-Pontiff 2016 predicts 32-58% for published factors)")
    else:
        parts.append("")
        parts.append("SUBSAMPLE STABILITY: not computed for this verdict")
        parts.append("(insufficient months for 4-split, or PnL series missing)")

    # L2-6 INDUSTRY EXTENSION (JOINT FF5+MOM ∪ 12-Industry model)
    # Per [[feedback-fwl-sequential-residual-trap-2026-06-09]]: α is
    # from JOINT model. Compare to FF5+MOM-only α via Δα.
    if industry_extension:
        ix = industry_extension
        parts.append("")
        parts.append("INDUSTRY EXTENSION (JOINT FF5+MOM ∪ 12-Industry model)")
        parts.append("=" * 55)
        parts.append(f"  α_FF5MOM_only NW-t (Stage 1):  "
                       f"{ix.get('alpha_ff5mom_only_nw_t', float('nan')):+.3f}")
        parts.append(f"  α_full NW-t (joint 18-factor):  "
                       f"{ix.get('alpha_full_nw_t', float('nan')):+.3f}")
        da_t = ix.get("delta_alpha_nw_t_approx")
        if da_t is not None:
            parts.append(f"  Δα NW-t (Stage 1 − full):       {da_t:+.3f}")
            parts.append("    [Δα > 1.5 → industries ABSORB alpha (GP/A pattern)]")
            parts.append("    [Δα ≈ 0 or < 0 → industries DON'T absorb (genuine alpha)]")
        parts.append(f"  α_full annual:                 "
                       f"{ix.get('alpha_full_annual', float('nan'))*100:+.3f}%")
        parts.append(f"  joint R² (18 factors):          "
                       f"{ix.get('r2_full', float('nan')):.4f}")
        # Industry-subset F-test
        ifj = ix.get("industry_joint_f_test") or {}
        parts.append(f"  industry-subset F (H0: all γ_Industry = 0):")
        parts.append(f"    F-stat: {ifj.get('f_stat', float('nan')):.3f}  "
                       f"p-value: {ifj.get('f_pvalue', float('nan')):.4g}  "
                       f"df: ({ifj.get('df_num','?')}, "
                       f"{ifj.get('df_denom','?')})")
        parts.append("    [p < 0.01 → industry panel adds explanation "
                       "(orthogonality rejected)]")
        # Top 3 industry loadings by |t|
        ibetas = ix.get("industry_betas") or {}
        ibet_t = ix.get("industry_beta_nw_t") or {}
        if ibetas:
            sorted_inds = sorted(ibet_t.items(),
                                    key=lambda kv: -abs(kv[1]))[:3]
            parts.append("  Top 3 industry tilts (joint model):")
            for k, t in sorted_inds:
                b = ibetas.get(k, float("nan"))
                sig = ("***" if abs(t) > 2.58 else
                       "**"  if abs(t) > 1.96 else
                       "*"   if abs(t) > 1.65 else "")
                parts.append(f"    {k:6s}  β={b:+.4f}  t={t:+.3f}  {sig}")
        sha = ix.get("industry_snapshot_sha")
        if sha:
            parts.append(f"  industry snapshot SHA-256: {sha[:12]}...")
    else:
        parts.append("")
        parts.append("INDUSTRY EXTENSION: not computed for this verdict")
        parts.append("(Stage 1 anchor regression missing or insufficient overlap)")

    # Cross-asset macro extension (JOINT model with macro regime panel)
    if cross_asset_extension:
        xa = cross_asset_extension
        parts.append("")
        parts.append("CROSS-ASSET MACRO EXTENSION (JOINT with macro regime)")
        parts.append("=" * 55)
        parts.append(f"  model form: {xa.get('model_form','?')}")
        parts.append(f"  α_FF5MOM-only NW-t (Stage 1):    "
                       f"{xa.get('alpha_ff5mom_only_nw_t', float('nan')):+.3f}")
        a_ind = xa.get("alpha_with_industry_nw_t")
        if a_ind is not None:
            parts.append(f"  α_+Industry NW-t (Stage 2):       {a_ind:+.3f}")
        parts.append(f"  α_full NW-t (Stage 3, w/ Macro):  "
                       f"{xa.get('alpha_full_nw_t', float('nan')):+.3f}")
        d_ff = xa.get("delta_vs_ff5mom_nw_t")
        if d_ff is not None:
            parts.append(f"  Δα FF5+MOM → full (full peel):   {d_ff:+.3f}")
            parts.append("    [positive Δα → joint model ate alpha]")
        d_ind = xa.get("delta_vs_industry_nw_t")
        if d_ind is not None:
            parts.append(f"  Δα +Industry → full (macro peel): {d_ind:+.3f}")
            parts.append("    [positive Δα → macro panel ate alpha "
                           "beyond Industry]")
        parts.append(f"  α_full annual:                    "
                       f"{xa.get('alpha_full_annual', float('nan'))*100:+.3f}%")
        parts.append(f"  joint R² (all panels):            "
                       f"{xa.get('r2_full', float('nan')):.4f}")
        mf = xa.get("macro_joint_f_test") or {}
        parts.append(f"  macro-subset F (H0: all macro β = 0):")
        parts.append(f"    F-stat: {mf.get('f_stat', float('nan')):.3f}  "
                       f"p-value: {mf.get('f_pvalue', float('nan')):.4g}")
        parts.append("    [p < 0.01 → macro regime adds significant "
                       "explanation]")
        mbetas = xa.get("macro_betas") or {}
        mbeta_t = xa.get("macro_beta_nw_t") or {}
        if mbetas:
            top3 = sorted(mbeta_t.items(),
                              key=lambda kv: -abs(kv[1]))[:3]
            parts.append("  Top 3 macro loadings (joint model):")
            for k, t in top3:
                b = mbetas.get(k, float("nan"))
                sig = ("***" if abs(t) > 2.58 else
                       "**"  if abs(t) > 1.96 else
                       "*"   if abs(t) > 1.65 else "")
                parts.append(f"    {k:20s} β={b:+.5f}  t={t:+.3f}  {sig}")
        sha = xa.get("macro_snapshot_sha")
        if sha:
            parts.append(f"  macro snapshot SHA-256: {sha[:12]}...")
    else:
        parts.append("")
        parts.append("CROSS-ASSET MACRO EXTENSION: not computed")
        parts.append("(macro library missing or stage 1 absent)")

    # B (2026-06-09): SPECIFICATION ROBUSTNESS — neighborhood ablation
    # of B-class params. CRITICAL: cells do NOT inflate DSR n_trials —
    # they are robustness checks of one hypothesis, not N hypotheses
    # (Asness 2017 / HXZ 2020 convention).
    if specification_robustness:
        sr = specification_robustness
        parts.append("")
        parts.append("SPECIFICATION ROBUSTNESS (B-class param neighborhood)")
        parts.append("=" * 55)
        parts.append(f"  status:             {sr.get('status', '?')}")
        parts.append(f"  verdict:            {sr.get('verdict', '?')}")
        if sr.get("stability_score") is not None:
            parts.append(f"  stability_score:    "
                           f"{sr['stability_score']:.3f}")
            parts.append(f"  bar (ROBUST):       "
                           f">= {sr.get('robust_bar', 0.60):.2f}")
            parts.append(f"  bar (MARGINAL):     "
                           f">= {sr.get('marginal_bar', 0.40):.2f}")
        if sr.get("sharpe_median") is not None:
            parts.append(f"  base Sharpe:        "
                           f"{sr.get('base_sharpe', float('nan')):+.3f}")
            parts.append(f"  neighborhood min:   "
                           f"{sr.get('sharpe_min', float('nan')):+.3f}")
            parts.append(f"  neighborhood med:   "
                           f"{sr['sharpe_median']:+.3f}")
            parts.append(f"  neighborhood max:   "
                           f"{sr.get('sharpe_max', float('nan')):+.3f}")
        parts.append(f"  cells tested:       "
                       f"{sr.get('successful_cells', '?')} / "
                       f"{sr.get('neighborhood_size', 0) + 1}")
        if sr.get("errors", 0) > 0:
            parts.append(f"  template errors:    {sr['errors']}")
        cells = sr.get("cells_tested") or []
        if cells:
            parts.append("")
            parts.append("  Per-cell breakdown:")
            for c in cells[:14]:    # cap at 14 cells in prompt
                sharpe = c.get("sharpe")
                tstat  = c.get("nw_t_stat")
                sharpe_str = (f"{sharpe:+.3f}" if sharpe is not None
                                else "  N/A")
                t_str = (f"{tstat:+.2f}" if tstat is not None
                           else "  N/A")
                parts.append(
                    f"    {c.get('label', '?'):28s}  "
                    f"Sharpe={sharpe_str}  NW-t={t_str}  "
                    f"verdict={c.get('verdict', '?')}"
                )
        parts.append("")
        parts.append("  DSR NOTE: these neighborhood cells are "
                       "ROBUSTNESS CHECKS")
        parts.append("  of one hypothesis, NOT N new hypotheses. They")
        parts.append("  DO NOT inflate Bailey-LdP DSR n_trials "
                       "(n_trials_increment=0).")
        parts.append("  Do not penalize the headline verdict for these "
                       "cells.")
    else:
        parts.append("")
        parts.append("SPECIFICATION ROBUSTNESS: not computed")
        parts.append("(template verdict was RED, or spec set no B-class "
                       "params to vary)")

    # N (2026-06-10): the 4 senior gates — DSR / ρ₁ / paper-OOS / power.
    # Each line carries its own interpretation so the LLM doesn't have
    # to recall the literature thresholds from training.
    if pnl_diagnostics:
        pdg = pnl_diagnostics
        parts.append("")
        parts.append("PNL DIAGNOSTICS (senior gates: DSR / rho1 / "
                       "paper-OOS / power)")
        parts.append("=" * 55)
        dsr = pdg.get("dsr") or {}
        if dsr.get("deflated_sr_prob") is not None:
            parts.append(
                f"  DSR P(Sharpe real | {dsr.get('n_trials_family')} "
                f"family trials): {dsr['deflated_sr_prob']:.3f}")
            parts.append(
                "    (< 0.90 → multiple-testing risk; the headline "
                "t-stat overstates significance)")
        r1 = pdg.get("rho1") or {}
        if r1.get("rho1") is not None:
            flag = "  <-- SMELL" if r1.get("smell") else ""
            parts.append(
                f"  rho1 lag-1 autocorr: {r1['rho1']:+.3f} "
                f"(t={r1.get('rho1_t', float('nan')):+.2f}){flag}")
            parts.append(
                f"    Sharpe-SE inflation if AR(1): "
                f"x{r1.get('sharpe_se_inflation', float('nan')):.2f} "
                f"(bar {r1.get('smell_bar')}; >bar also suggests "
                "smoothed/stale pricing)")
        po = pdg.get("paper_oos")
        if po:
            dead = "  <-- EFFECTIVELY DEAD" if po.get("effectively_dead") else ""
            parts.append(
                f"  paper-OOS Sharpe ratio: {po['oos_ratio']:+.2f} "
                f"(in-window {po['sharpe_in_window']:+.2f} over "
                f"{po['n_months_in']}mo -> post {po['sharpe_post_window']:+.2f} "
                f"over {po['n_months_post']}mo){dead}")
            parts.append(
                f"    (McLean-Pontiff normal decay keeps ratio "
                f"0.42-0.68; < {po.get('dead_bar')} = died post-pub)")
        pw = pdg.get("power") or {}
        table = pw.get("table") or {}
        if table:
            cells = ", ".join(
                f"SR={k.split('_')[1]}: {v:.0%}" for k, v in table.items())
            parts.append(
                f"  power of t>={pw.get('t_green')} gate at "
                f"n={pw.get('n_months')}mo: {cells}")
            parts.append(
                "    (low power → a RED verdict is WEAK evidence of "
                "absence; do not over-read short-sample REDs)")
    else:
        parts.append("")
        parts.append("PNL DIAGNOSTICS: not computed "
                       "(PnL series missing or < 24 months)")

    # Phase 1 Commit 5 (2026-06-09): routing decisions audit trail
    # per spec §15.A5. Lets you vocalize "lens X was skipped because
    # Y" in caveats. Shows which axes drove which decisions.
    if routing_decisions:
        parts.append("")
        parts.append("ROUTING DECISIONS (which lenses ran / skipped / failed)")
        parts.append("=" * 55)
        for rd in routing_decisions:
            action = rd.get("action", "?")
            lens = rd.get("lens", "?")
            line = f"  [{action:25s}] {lens}"
            reason = rd.get("reason")
            if reason:
                line += f"  — {reason[:150]}"
            parts.append(line)
        parts.append("")
        parts.append(
            "  USE THIS TRAIL: when a lens was SKIPPED_INAPPLICABLE, "
            "you cannot vocalize concerns dependent on its output "
            "(e.g., do not invent industry-spanning concerns when "
            "industry_extension was skipped — instead caveat that "
            "the spanning check was not performed)."
        )

    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Main entry — assess this verdict
# ────────────────────────────────────────────────────────────────────
def assess_self_doubt(
    spec,
    template_result,
    *,
    family_hint: str,
    n_trials_family: int = 0,
    anchor_orthogonality: Optional[dict] = None,
    subsample_stability:  Optional[dict] = None,
    industry_extension:   Optional[dict] = None,
    cross_asset_extension: Optional[dict] = None,
    specification_robustness: Optional[dict] = None,
    pnl_diagnostics:      Optional[dict] = None,
    routing_decisions:    Optional[list] = None,
) -> Optional[SelfDoubtAssessment]:
    """Single Sonnet call. Returns SelfDoubtAssessment or None on
    failure (graceful degradation — caller proceeds without).

    Args:
      spec: the FactorSpec the dispatcher ran
      template_result: the TemplateResult from dispatcher
      family_hint: mechanism_family
      n_trials_family: current family n_trials count
      anchor_orthogonality: optional L2-4 residual-α regression
            result (dict). When provided, the LLM vocalizes
            spanning concerns.
      subsample_stability: optional L2-5 N-split decomposition
            (dict). When provided, the LLM vocalizes one-window
            domination, post-pub decay, and crisis-killed patterns.

    Returns None when:
      - LLM call fails / raises
      - LLM doesn't call the emit_self_doubt tool
      - Validation fails (confidence out of range, etc.)
    """
    # Only assess verdicts that emit (GREEN / MARGINAL / RED).
    # Dispatcher-internal states (PENDING_TEMPLATE_BUILD, DATA_ERROR,
    # etc.) don't warrant self-doubt — they're not research findings.
    emittable = {"GREEN", "MARGINAL", "RED"}
    if template_result.verdict not in emittable:
        return None

    user_msg = _format_user_message(
        spec, template_result, family_hint, n_trials_family,
        anchor_orthogonality=anchor_orthogonality,
        subsample_stability=subsample_stability,
        industry_extension=industry_extension,
        cross_asset_extension=cross_asset_extension,
        specification_robustness=specification_robustness,
        pnl_diagnostics=pnl_diagnostics,
        routing_decisions=routing_decisions,
    )
    try:
        result = llm_call(
            workload   = "strengthener_self_doubt",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "strengthener_self_doubt",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 2048,
            scope      = "tier_c_l3_2_self_doubt",
        )
    except Exception as exc:
        logger.warning("self_doubt: llm_call failed for %s: %s",
                        spec.hypothesis_id, exc)
        return None

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "emit_self_doubt":
            payload = tc.input
            break
    if payload is None:
        logger.warning("self_doubt: %s did not call emit_self_doubt",
                        spec.hypothesis_id)
        return None

    try:
        conf = float(payload.get("confidence"))
    except (TypeError, ValueError):
        logger.warning("self_doubt: %s emitted bad confidence",
                        spec.hypothesis_id)
        return None
    if not (0.0 <= conf <= 0.99):
        logger.warning("self_doubt: %s confidence %s out of range",
                        spec.hypothesis_id, conf)
        return None

    return SelfDoubtAssessment(
        confidence              = conf,
        confidence_reason       = str(payload.get("confidence_reason")
                                        or "")[:400],
        caveats                 = tuple(
            str(x)[:300] for x in (payload.get("caveats") or ())
        ),
        methodological_concerns = tuple(
            str(x)[:300] for x in
            (payload.get("methodological_concerns") or ())
        ),
        suspicious_metrics      = tuple(
            str(x)[:300] for x in (payload.get("suspicious_metrics") or ())
        ),
        assessment_ts           = _dt.datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        model                   = result.model,
    )
