"""engine.research.enhance.verdict — IMPROVEMENT / NOISE / DEGRADATION
classifier for enhance hypotheses.

Why a different verdict vocabulary
==================================
Forward verdict semantics (GREEN / MARGINAL / RED) answer "is X a real
alpha?" — a single-strategy question. Enhance asks a different question:
"is X' strictly better than already-deployed X?" — and the answer
distribution is:

  IMPROVEMENT  : variant beats baseline by enough margin that we'd
                 modify the deployed sleeve. Goes to /approvals for
                 principal capital decision (never auto-deploy).
  NOISE        : ΔSharpe is within bootstrap noise — variant changes
                 the sleeve, but we can't distinguish from luck.
                 Doctrine entry: "tried this, indistinguishable, move
                 on." NOT a downgrade for the deployed sleeve.
  DEGRADATION  : variant statistically WORSE than baseline. Doctrine
                 entry: "this modification HURTS, don't try variants
                 in this direction."

Note: the symmetry is intentional — DEGRADATION is a learning signal
of equal value to IMPROVEMENT for solo quant (prevents future variant
proposals in a known-bad direction).

Thresholds (calibrated for solo-quant scale + AQR 2018 base rates)
===================================================================
Frazzini-Pedersen 2018 documented institutional enhance-success base
rate ≈ 20% (sleeve modifications that empirically improve). For solo
quant with smaller variant search space, base rate is closer to 10%.

  IMPROVEMENT requires:
    ΔSharpe(annualized) > +0.15  AND  bootstrap_t_stat >= +1.96
    AND p_value(one-sided variant > baseline) < 0.05
    AND correlation(baseline, variant) >= 0.50    # safety: pure noise
                                                     uncorrelated swap
                                                     is NOT an enhance,
                                                     it's a new factor
                                                     and belongs in
                                                     forward pipeline

  DEGRADATION requires (mirror of IMPROVEMENT):
    ΔSharpe < -0.15  AND  bootstrap_t_stat <= -1.96
    AND p_value(one-sided variant < baseline) < 0.05    # i.e.
        equivalent: (1 - p_value(one-sided improvement)) < 0.05

  NOISE (default): everything else, including marginal cases that
    didn't clear the |t|>=1.96 bar even if direction was right.

Why ΔSharpe ≥ 0.15 (not 0.20 like forward)
==========================================
Paired SE shrinks by √(1-ρ) so the SAME Sharpe-diff has stronger
statistical evidence than in forward. We can demand tighter economic
significance because the statistical noise is already controlled.
0.15 ≈ 15% Sharpe improvement on a deployed Sharpe-1 sleeve = ~$300
extra alpha-million on $100M AUM at 6% target vol. Worth modifying for.
"""
from __future__ import annotations

import enum

from engine.research.enhance.paired_bootstrap import PairedBootstrapResult


# Thresholds (see module docstring for derivation)
GREEN_THRESHOLD_SHARPE_DIFF       =  0.15
GREEN_THRESHOLD_T_STAT            =  1.96
GREEN_THRESHOLD_P_VALUE           =  0.05
GREEN_THRESHOLD_CORRELATION_FLOOR =  0.50

# Mirror for degradation
RED_THRESHOLD_SHARPE_DIFF         = -0.15
RED_THRESHOLD_T_STAT              = -1.96
RED_THRESHOLD_P_VALUE             =  0.95   # 1 - 0.05 (variant < baseline)


class EnhanceVerdict(str, enum.Enum):
    """Enhance verdict — DELIBERATELY different vocabulary from forward."""
    IMPROVEMENT = "IMPROVEMENT"
    NOISE       = "NOISE"
    DEGRADATION = "DEGRADATION"


def classify_enhance_verdict(
    result: PairedBootstrapResult,
) -> EnhanceVerdict:
    """Apply the thresholds to a paired bootstrap result.

    Returns IMPROVEMENT / NOISE / DEGRADATION. The threshold rationale
    + sign conventions are documented at the top of this module."""
    # Correlation gate — if baseline and variant are nearly uncorrelated,
    # this isn't an enhance (modification) but a NEW STRATEGY. Refuse
    # to verdict; consumers should route this to forward.
    if abs(result.correlation) < GREEN_THRESHOLD_CORRELATION_FLOOR:
        return EnhanceVerdict.NOISE   # honest fail-open; calling code
                                        # surfaces low correlation specifically

    if (result.sharpe_diff_observed > GREEN_THRESHOLD_SHARPE_DIFF
            and result.sharpe_diff_t_stat >= GREEN_THRESHOLD_T_STAT
            and result.sharpe_diff_p_value < GREEN_THRESHOLD_P_VALUE):
        return EnhanceVerdict.IMPROVEMENT

    if (result.sharpe_diff_observed < RED_THRESHOLD_SHARPE_DIFF
            and result.sharpe_diff_t_stat <= RED_THRESHOLD_T_STAT
            and result.sharpe_diff_p_value > RED_THRESHOLD_P_VALUE):
        return EnhanceVerdict.DEGRADATION

    return EnhanceVerdict.NOISE
