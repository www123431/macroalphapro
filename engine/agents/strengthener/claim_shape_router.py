"""engine.agents.strengthener.claim_shape_router — Phase 2.1 (2026-06-13).

Stage 0 classifier: before the existing extract_factor_spec runs (Stage 1),
tightly classify the hypothesis into ONE canonical claim SHAPE. The shape
maps 1-1 to a template's signal_kind (when shape is TESTABLE_DIRECT) and
acts as a guardrail in Stage 1's user prompt — preventing the documented
Sonnet drift where SPANNING / DECAY_STUDY / METHODOLOGY claims get
stretched into factor_combination or cross_sec specs.

Why this exists (BUG-2 generalized)
====================================
BUG-2 (spanning claims → factor_combination drift) was the first
detected case. Memory `project_sonnet_spec_interpretation_drift_2026-06-11`
documents that DECAY_STUDY / METHODOLOGY / FACTOR_STRUCTURE also drift
silently. The verdict math is right but the spec answers the wrong
question. Self-review catches ~50% (per Saunders-Sutskever 2023); the
fix is structural: force the SHAPE choice before extractor sees the
8-way signal_kind enum.

Two-stage flow
==============
  Stage 0 (this module):  ClaimShapeVerdict with confidence + rationale
  Stage 1 (existing):     extract_factor_spec receives shape_hint and
                          conditions its prompt narrowing

ClaimShape taxonomy (11 + UNCLEAR)
===================================
  TESTABLE_DIRECT (8) — route to existing template:
    CROSS_SECTIONAL_ALPHA → cross_sectional_rank
    TIME_SERIES_MOMENTUM  → time_series_momentum
    CARRY                 → carry
    SPANNING              → spanning_test (BUG-2 fix)
    FACTOR_COMBINATION    → factor_combination
    PORTFOLIO_OVERLAY     → portfolio_overlay
    VRP                   → vrp
    EVENT_DRIFT           → event_drift (template pending Phase 3.1)

  TESTABLE_FUTURE (2) — refuse with NEEDS_NEW_TEMPLATE:
    DECAY_STUDY           → no template yet (multi-period spanning)
    CAPACITY              → no template yet (live-AUM scaling)

  NOT_TESTABLE_AS_FACTOR (1) — refuse with WRONG_HYPOTHESIS_TYPE:
    FACTOR_STRUCTURE      → methodology claim, not factor proposal

  UNCLEAR — refuse with LOW_CONFIDENCE_CLASSIFY:
    classifier returns this when no shape exceeds confidence threshold

Cost
====
Stage 0: ~$0.001 per call (150in + 150out @ Sonnet pricing). At 15-20
extractions/wk, adds ~$0.015-0.02/wk on top of Stage 1 ~$0.075-0.10/wk.
ROI: prevents the ~10-15% wrong-shape verdicts that pollute belief layer.

Academic anchors
================
- Harvey-Liu-Zhu 2016 §3: specification mining is 80% of publication bias
- Cochrane 2011 AFA: "factor zoo" requires distinguishing claim shapes
- López de Prado AFML Ch.2: garbage classification → garbage verdicts
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from enum import Enum
from typing import Any, Optional

from engine.llm.call import call as llm_call

logger = logging.getLogger(__name__)


# ── Taxonomy ──────────────────────────────────────────────────────


class ClaimShape(str, Enum):
    """Canonical shape of a factor research claim. Maps to template
    signal_kind when TESTABLE_DIRECT."""
    # TESTABLE_DIRECT — route to existing template
    CROSS_SECTIONAL_ALPHA = "CROSS_SECTIONAL_ALPHA"
    TIME_SERIES_MOMENTUM  = "TIME_SERIES_MOMENTUM"
    CARRY                 = "CARRY"
    SPANNING              = "SPANNING"
    FACTOR_COMBINATION    = "FACTOR_COMBINATION"
    PORTFOLIO_OVERLAY     = "PORTFOLIO_OVERLAY"
    VRP                   = "VRP"
    EVENT_DRIFT           = "EVENT_DRIFT"
    # TESTABLE_FUTURE — no template yet
    DECAY_STUDY           = "DECAY_STUDY"
    CAPACITY              = "CAPACITY"
    # NOT_TESTABLE — wrong hypothesis type
    FACTOR_STRUCTURE      = "FACTOR_STRUCTURE"
    # Escape hatch
    UNCLEAR               = "UNCLEAR"


TESTABLE_DIRECT: frozenset[ClaimShape] = frozenset({
    ClaimShape.CROSS_SECTIONAL_ALPHA,
    ClaimShape.TIME_SERIES_MOMENTUM,
    ClaimShape.CARRY,
    ClaimShape.SPANNING,
    ClaimShape.FACTOR_COMBINATION,
    ClaimShape.PORTFOLIO_OVERLAY,
    ClaimShape.VRP,
    ClaimShape.EVENT_DRIFT,
})

TESTABLE_FUTURE: frozenset[ClaimShape] = frozenset({
    ClaimShape.DECAY_STUDY,
    ClaimShape.CAPACITY,
})

NOT_TESTABLE: frozenset[ClaimShape] = frozenset({
    ClaimShape.FACTOR_STRUCTURE,
})


# Maps ClaimShape → factor_dispatcher signal_kind (only TESTABLE_DIRECT)
SHAPE_TO_SIGNAL_KIND: dict[ClaimShape, str] = {
    ClaimShape.CROSS_SECTIONAL_ALPHA: "cross_sectional_rank",
    ClaimShape.TIME_SERIES_MOMENTUM:  "time_series_momentum",
    ClaimShape.CARRY:                 "carry",
    ClaimShape.SPANNING:              "spanning_test",
    ClaimShape.FACTOR_COMBINATION:    "factor_combination",
    ClaimShape.PORTFOLIO_OVERLAY:     "portfolio_overlay",
    ClaimShape.VRP:                   "vrp",
    ClaimShape.EVENT_DRIFT:           "event_drift",
}


_MIN_CONFIDENCE = 0.55   # below this → UNCLEAR refusal


# ── Verdict + refusal types ───────────────────────────────────────


@_dc.dataclass(frozen=True)
class ClaimShapeVerdict:
    """Stage 0 output."""
    shape:        ClaimShape
    confidence:   float           # 0.0-1.0
    rationale:    str             # 1-line explanation
    refusal:      Optional[str] = None   # set when shape ∈ NOT_TESTABLE
                                          # or TESTABLE_FUTURE or UNCLEAR

    @property
    def is_actionable(self) -> bool:
        return self.shape in TESTABLE_DIRECT and self.refusal is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape":      self.shape.value,
            "confidence": float(self.confidence),
            "rationale":  self.rationale,
            "refusal":    self.refusal,
        }


# ── Prompt + tool schema ──────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a quantitative research claim CLASSIFIER. Given a hypothesis
extracted from a finance paper, classify it into ONE canonical SHAPE so
a downstream specification extractor knows which testing template to
route to.

Do not propose a backtest. Do not invent data sources. ONLY classify.

If the claim does not cleanly fit one shape, choose UNCLEAR — DO NOT
pick the closest-fitting wrong shape. Forcing wrong shapes is the
documented BUG-2 / specification-drift failure mode this stage exists
to prevent.

Shape definitions (12 total):

  CROSS_SECTIONAL_ALPHA
    "Stocks with property X earn higher returns than stocks without X."
    Sortable signal, long-short or long-top-decile. Examples: value
    (B/M ratio), profitability (GP/A), low-volatility, size.

  TIME_SERIES_MOMENTUM
    "Asset's own recent return predicts future return." Single-asset
    or basket trend-following. Examples: 12-1 month momentum on
    sector ETF, TSMOM on commodity futures.

  CARRY
    "High-carry assets outperform low-carry within an asset class."
    Examples: FX carry (interest rate differential), commodity carry
    (front-back roll yield), equity yield carry.

  SPANNING
    "Is anomaly X spanned/subsumed by model M?" / "Does adding factor
    F to model M improve spanning?" The claim is ABOUT whether X has
    orthogonal alpha to M, NOT about whether X earns positive returns
    standalone. Examples: "MOM is not spanned by FF5" (Asness-Frazzini-
    Pedersen 2014), "QMJ subsumes profitability factor".

  FACTOR_COMBINATION
    "w% factor_a + (1-w)% factor_b combined beats either alone." The
    claim is about the BLEND. Examples: "50/50 value+momentum beats
    each" (Asness-Moskowitz-Pedersen 2013).

  PORTFOLIO_OVERLAY
    "Adding K% of strategy X to base portfolio P improves Sharpe."
    Examples: "20% TSMOM in 60/40" (HOP 2017), "10% tail hedge in
    risk parity".

  VRP
    "Implied volatility systematically exceeds realized volatility;
    short-vol earns positive risk premium." Examples: "SPX variance
    risk premium" (Carr-Wu 2009), "short straddle alpha".

  EVENT_DRIFT
    "After event E, returns drift for K days/weeks." Examples: PEAD
    (post-earnings-announcement drift), analyst revision drift,
    attention spike drift.

  DECAY_STUDY
    "Anomaly worked in period T1 but does NOT work in period T2"
    (publication-bias style). Examples: "PEAD weakens after 1990",
    "value premium declines post-2000". The claim is ABOUT the decay
    itself, not the original anomaly. No template exists yet.

  CAPACITY
    "Anomaly returns break above $N AUM." Examples: "small-cap value
    capacity ~$2B". The claim is about scalability limits, not the
    anomaly itself. No template exists yet.

  FACTOR_STRUCTURE
    "The right factor model is X" (e.g. FF5 vs HXZ4 vs M4). The
    claim is methodological — about which model to use, not whether
    a specific factor earns positive returns. Belongs in methodology
    queue, not factor dispatch.

  UNCLEAR
    Escape hatch. Use when the claim mixes multiple shapes OR cannot
    be classified into any single one with reasonable confidence.
    PREFER unclear OVER picking the closest-wrong shape.

Output via the classify_shape tool. confidence ∈ [0.0, 1.0]; below
0.55 the downstream router refuses to extract a spec.
"""


_TOOL_SCHEMA = {
    "name":        "classify_shape",
    "description": ("Emit the canonical claim shape + confidence + "
                      "1-line rationale."),
    "input_schema": {
        "type":       "object",
        "properties": {
            "shape": {
                "type":        "string",
                "enum":        [s.value for s in ClaimShape],
                "description": "The canonical claim shape.",
            },
            "confidence": {
                "type":        "number",
                "minimum":     0.0,
                "maximum":     1.0,
                "description": ("Probability this is the correct shape. "
                                  "< 0.55 → router refuses."),
            },
            "rationale": {
                "type":        "string",
                "description": ("One sentence (≤ 200 chars) citing the "
                                  "feature of the claim that drove the choice."),
            },
        },
        "required":             ["shape", "confidence", "rationale"],
        "additionalProperties": False,
    },
}


def _format_user(h) -> str:
    """Tight user prompt — only fields relevant to shape classification.
    Detailed methodology / data fields go to Stage 1, not Stage 0."""
    return "\n".join([
        f"HYPOTHESIS_ID:    {h.hypothesis_id}",
        f"MECHANISM_FAMILY: {h.mechanism_family.value}",
        f"MECHANISM_SUBTYPE: {h.mechanism_subtype}",
        "",
        "CLAIM:",
        (h.claim or "").strip(),
        "",
        "TEST_METHODOLOGY (if provided):",
        (h.test_methodology or "").strip(),
    ])


# ── Public API ────────────────────────────────────────────────────


def classify_claim_shape(h, *, llm_call_fn=None) -> ClaimShapeVerdict:
    """Stage 0: tight classification. Returns ClaimShapeVerdict.

    On LLM failure / tool not called / invalid enum: returns UNCLEAR
    with refusal="CLASSIFIER_UNAVAILABLE". Caller treats this same as
    low-confidence — refuses to extract a spec.

    `llm_call_fn` injection point for tests.
    """
    fn = llm_call_fn or llm_call
    try:
        result = fn(
            workload   = "strengthener_claim_shape",
            system     = _SYSTEM_PROMPT,
            user       = _format_user(h),
            agent_id   = "strengthener_claim_shape_router",
            tools      = [_TOOL_SCHEMA],
            max_tokens = 512,
            scope      = "tier_c_claim_shape_router",
        )
    except Exception as exc:
        logger.warning("claim_shape_router: llm_call failed for %s: %s",
                        h.hypothesis_id, exc)
        return ClaimShapeVerdict(
            shape       = ClaimShape.UNCLEAR,
            confidence  = 0.0,
            rationale   = f"classifier llm error: {type(exc).__name__}",
            refusal     = "CLASSIFIER_UNAVAILABLE",
        )

    payload = None
    for tc in (result.tool_calls or ()):
        if tc.name == "classify_shape":
            payload = tc.input
            break
    if payload is None:
        logger.warning("claim_shape_router: %s did not call classify_shape tool",
                        h.hypothesis_id)
        return ClaimShapeVerdict(
            shape       = ClaimShape.UNCLEAR,
            confidence  = 0.0,
            rationale   = "tool_not_called",
            refusal     = "CLASSIFIER_NO_TOOL_CALL",
        )

    raw_shape = str(payload.get("shape") or "")
    try:
        shape = ClaimShape(raw_shape)
    except ValueError:
        logger.warning("claim_shape_router: %s emitted unknown shape=%r",
                        h.hypothesis_id, raw_shape)
        return ClaimShapeVerdict(
            shape       = ClaimShape.UNCLEAR,
            confidence  = 0.0,
            rationale   = f"unknown shape: {raw_shape}",
            refusal     = "CLASSIFIER_INVALID_ENUM",
        )

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    rationale = str(payload.get("rationale") or "")[:300]

    # Compute refusal
    refusal = None
    if shape == ClaimShape.UNCLEAR or confidence < _MIN_CONFIDENCE:
        refusal = "LOW_CONFIDENCE_CLASSIFY"
    elif shape in NOT_TESTABLE:
        refusal = "WRONG_HYPOTHESIS_TYPE"
    elif shape in TESTABLE_FUTURE:
        refusal = "NEEDS_NEW_TEMPLATE"

    return ClaimShapeVerdict(
        shape       = shape,
        confidence  = confidence,
        rationale   = rationale,
        refusal     = refusal,
    )


def shape_hint_for_extractor(shape: ClaimShape) -> str:
    """Stage 1 helper: render a guardrail line that gets prepended to
    the existing extract_factor_spec user prompt. Constrains the
    signal_kind choice to the canonical mapping."""
    sk = SHAPE_TO_SIGNAL_KIND.get(shape)
    if sk is None:
        return ""
    return (
        f"[ROUTER_HINT] Stage 0 classified this claim as {shape.value} "
        f"with confidence ≥ {_MIN_CONFIDENCE}. The ONLY correct signal_kind "
        f"for this shape is '{sk}'. Do NOT pick another signal_kind. If "
        f"you cannot map the claim to '{sk}' with sensible inputs, return "
        f"signal_kind='requires_custom_code' with a rationale citing the "
        f"specific mismatch."
    )
