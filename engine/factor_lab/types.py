"""
engine/factor_lab/types.py — State machine enum + result dataclasses.

Spec: docs/spec_factor_lab.md §2.1 (state machine) + §3.3 (decision logic)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class FactorState(str, enum.Enum):
    """Factor lab lifecycle states (spec §2.1).

    Stored as string in SpecRegistry.lab_state column.
    Legacy spec rows (pre-LAB v1-v8 hypothesis tests) have lab_state=NULL —
    rendered as "pre-LAB legacy" by the UI, not part of the state machine.
    """

    DRAFT                 = "DRAFT"                  # spec not written yet
    PROPOSED              = "PROPOSED"               # spec drafted, not registered
    BLOCKED_UNDERPOWERED  = "BLOCKED_UNDERPOWERED"   # power_check rejected
    REGISTERED            = "REGISTERED"             # spec_hash locked, ready to test
    TESTING               = "TESTING"                # b_plus_search running
    PASS                  = "PASS"                   # raw 5% sig + BHY pass
    MARGINAL              = "MARGINAL"               # raw sig but BHY fail
    FAIL                  = "FAIL"                   # raw insig
    FAIL_UNDERPOWERED     = "FAIL_UNDERPOWERED"      # raw insig + n < n_required


# Legal state transitions (per spec §2.2 — strict, no backtracking).
# Any transition NOT listed here raises IllegalTransition.
_LEGAL_TRANSITIONS: dict[FactorState, set[FactorState]] = {
    FactorState.DRAFT:                {FactorState.PROPOSED},
    FactorState.PROPOSED:             {
        FactorState.REGISTERED,
        FactorState.BLOCKED_UNDERPOWERED,
    },
    FactorState.BLOCKED_UNDERPOWERED: set(),  # terminal — must rewrite spec
    FactorState.REGISTERED:           {FactorState.TESTING},
    FactorState.TESTING:              {
        FactorState.PASS,
        FactorState.MARGINAL,
        FactorState.FAIL,
        FactorState.FAIL_UNDERPOWERED,
    },
    FactorState.PASS:                 set(),  # terminal — verdict written
    FactorState.MARGINAL:             set(),
    FactorState.FAIL:                 set(),
    FactorState.FAIL_UNDERPOWERED:    set(),
}


class IllegalTransition(ValueError):
    """Raised when a state transition violates the spec §2.2 rules."""


def assert_legal_transition(src: FactorState, dst: FactorState) -> None:
    """Raise IllegalTransition unless src → dst is permitted by spec §2.2."""
    if dst not in _LEGAL_TRANSITIONS.get(src, set()):
        raise IllegalTransition(
            f"Illegal factor lab state transition: {src.value} → {dst.value}. "
            f"Per spec §2.2, allowed from {src.value}: "
            f"{sorted(s.value for s in _LEGAL_TRANSITIONS.get(src, set())) or '(terminal)'}."
        )


@dataclass(frozen=True)
class PowerCheckResult:
    """Result of a pre-test power analysis (spec §3.3).

    Returned by power.power_check() so the caller can decide whether to
    transition PROPOSED → REGISTERED or PROPOSED → BLOCKED_UNDERPOWERED.
    """
    decision:                       FactorState   # REGISTERED or BLOCKED_UNDERPOWERED
    n_required:                     int
    n_available:                    int
    achieved_power_at_n_available:  float          # in [0, 1]
    expected_sharpe_lift:           float
    baseline_sharpe:                float
    target_power:                   float
    target_alpha:                   float
    method:                         str = "Lo-Memmel-Cohen-2002-2003-1988"
    reason:                         str = ""

    def to_dict(self) -> dict:
        return {
            "decision":                      self.decision.value,
            "n_required":                    self.n_required,
            "n_available":                   self.n_available,
            "achieved_power_at_n_available": self.achieved_power_at_n_available,
            "expected_sharpe_lift":          self.expected_sharpe_lift,
            "baseline_sharpe":               self.baseline_sharpe,
            "target_power":                  self.target_power,
            "target_alpha":                  self.target_alpha,
            "method":                        self.method,
            "reason":                        self.reason,
        }
