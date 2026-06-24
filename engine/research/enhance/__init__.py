"""engine.research.enhance — Phase 2 enhance verdict framework.

Doctrine
========
Forward research = "is X a real alpha?"
Enhance research = "does variant X' strictly improve deployed sleeve X?"

These are STATISTICALLY DIFFERENT problems and require different machinery.
Mixing them — running enhance through the forward strict-gate path — kills
real improvements because unpaired SE inflates by √(2(1-ρ)) ≈ 3x for the
typical ρ≈0.95 between deployed-sleeve and variant returns.

This subpackage provides the enhance-side machinery:

  - paired_bootstrap.py : Politis-Romano 1994 circular block bootstrap
                          for paired Sharpe-difference distribution
  - verdict.py          : IMPROVEMENT / NOISE / DEGRADATION classifier
                          (NOT GREEN/MARGINAL/RED — semantically different)
  - dispatcher.py       : dispatch_enhance_hypothesis(sleeve_id, variant)
                          one-call entry; reads deployed sleeve PnL via
                          SleeveProtocol.returns() + pairs against variant

Academic anchors:
  - López de Prado AFML Ch.2 (paired vs unpaired statistics)
  - Politis-Romano 1994 "The Stationary Bootstrap"
  - Jobson-Korkie 1981 / Memmel 2003 Sharpe-diff t-stat
  - Frazzini-Pedersen 2018 "Trading Costs" (institutional alpha = 70% enhance)
  - Bailey-LdP 2014 §3 (DSR n_trials applies to FORWARD only)

NOT in Phase 2 substrate (deferred to Phase 2.2):
  - LLM-driven variant generation from hypothesis (currently caller-supplied)
  - emit event type 'enhancement_evaluated' (substrate writes parallel jsonl)
  - automatic deploy_changed event on principal approval
"""
from engine.research.enhance.paired_bootstrap import (
    paired_block_bootstrap_sharpe_diff,
    paired_block_bootstrap_summary,
    PairedBootstrapResult,
)
from engine.research.enhance.verdict import (
    EnhanceVerdict,
    classify_enhance_verdict,
    GREEN_THRESHOLD_SHARPE_DIFF,
    GREEN_THRESHOLD_T_STAT,
)
from engine.research.enhance.dispatcher import (
    dispatch_enhance_hypothesis,
    EnhanceDispatchResult,
)

__all__ = [
    "paired_block_bootstrap_sharpe_diff",
    "paired_block_bootstrap_summary",
    "PairedBootstrapResult",
    "EnhanceVerdict",
    "classify_enhance_verdict",
    "GREEN_THRESHOLD_SHARPE_DIFF",
    "GREEN_THRESHOLD_T_STAT",
    "dispatch_enhance_hypothesis",
    "EnhanceDispatchResult",
]
