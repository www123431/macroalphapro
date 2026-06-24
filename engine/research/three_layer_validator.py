"""engine/research/three_layer_validator.py — SLM Phase 2.5: 3-layer
voting framework for paper-trade / shadow-trade validation.

DOCTRINE (post-2026-05-31 redesign per user critique of OBF-only):
  Single statistical lens is fragile. Use three independent layers
  with different epistemics; require majority agreement to act.

  Layer 1: Bayesian posterior P(Sharpe > threshold | data)
    - Continuous, no fixed sample-size requirement
    - Naturally sequential, no alpha-spending math needed
    - Calibrated against the P-D8 honest deploy target as prior mean
    - PRIMARY layer for systematic strategies

  Layer 2: Deflated Sharpe Ratio (Bailey-LdP 2014)
    - Accounts for selection bias from ~N candidate strategies
    - Uses skew + kurtosis to handle non-normal returns
    - DeflSR ≥ 0.9 means observed Sharpe is unlikely to be the
      "luckiest of N" — survives cross-strategy multiple testing
    - SECONDARY layer providing cross-strategy honesty check

  Layer 3: O'Brien-Fleming boundary
    - Frequentist alpha-spending boundary (FDA Phase III standard)
    - Pre-registered critical values protect against post-hoc peeking
    - Conservative on early-stop ACCEPT (high t required); reasonable
      at terminal final-look (t = z_alpha)
    - SANITY-CHECK layer for skeptical reviewers
    - When window is 24+ months, terminal critical Sharpe ≈ 1.0-1.4
      which is realistic for institutional deploy

Voting logic (configurable; defaults per institutional norm):
  Final decision = majority(Layer 1, Layer 2, Layer 3)

  ACCEPT  if ≥ 2/3 layers ACCEPT
  REJECT  if ≥ 2/3 layers REJECT  OR  any layer hard-REJECT in
                                       "reject_is_blocking" mode
  CONTINUE otherwise

Asymmetry rationale:
  REJECT is "blocking" because cost of false-positive deploy
  (allocate capital to bad strategy) > cost of false-negative
  (miss a good strategy — can re-test later with longer data).
  Per Stein (1955) decision-theoretic asymmetry for irreversible
  actions with capital cost.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

from engine.research.bayesian_sharpe_updater import (
    BayesianDecision, BayesianSharpeResult, bayesian_sharpe_update,
)
from engine.research.sequential_testing import (
    BoundaryResult, OBrienFlemingBoundary, SequentialDecision,
    sharpe_t_stat,
)
from engine.validation.deflated_sharpe import (
    DSRResult, deflated_sharpe_ratio,
)


class ThreeLayerDecision(str, Enum):
    """Composite decision across all 3 layers."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CONTINUE = "CONTINUE"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class LayerVote(str, Enum):
    """Single-layer vote translated into the composite vocabulary."""

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    CONTINUE = "CONTINUE"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True)
class ThreeLayerResult:
    """Composite output across all 3 layers."""

    n_months: int
    layer1_bayesian: BayesianSharpeResult
    layer2_deflated_sr: DSRResult
    layer3_obf: Optional[BoundaryResult]
    layer1_vote: str
    layer2_vote: str
    layer3_vote: str
    final_decision: ThreeLayerDecision
    rationale: str
    evidence_passed: bool        # True iff final_decision == ACCEPT

    # T1.3 (2026-06-05 audit C2 fix): forward-OOS observation feeds
    # the verdict. When forward_oos_returns is supplied with >= min_oos_n
    # observations, oos Sharpe is computed AND a negative value forces
    # REJECT — preventing the "full-window passes, OOS collapses"
    # failure mode (mom_hedge_overlay 2026-05-31 precedent).
    forward_oos_sharpe:  Optional[float] = None
    forward_oos_n:       int             = 0
    forward_oos_blocked: bool            = False


def _bayesian_to_vote(b: BayesianSharpeResult) -> str:
    return {
        BayesianDecision.ACCEPT: "ACCEPT",
        BayesianDecision.REJECT: "REJECT",
        BayesianDecision.CONTINUE: "CONTINUE",
        BayesianDecision.INSUFFICIENT: "INSUFFICIENT",
    }[b.decision]


def _deflated_to_vote(d: DSRResult, accept_dsr: float = 0.90,
                      reject_dsr: float = 0.50) -> str:
    """Map DeflSR value to vote.

    Default thresholds:
      accept if deflated_sr ≥ 0.90 (Bailey-LdP institutional bar)
      reject if deflated_sr ≤ 0.50 (random-chance ceiling)
      continue otherwise
    """
    import math
    if d is None or math.isnan(d.deflated_sr):
        return "INSUFFICIENT"
    if d.deflated_sr >= accept_dsr:
        return "ACCEPT"
    if d.deflated_sr <= reject_dsr:
        return "REJECT"
    return "CONTINUE"


def _obf_to_vote(o: Optional[BoundaryResult]) -> str:
    if o is None:
        return "INSUFFICIENT"
    return {
        SequentialDecision.ACCEPT: "ACCEPT",
        SequentialDecision.REJECT: "REJECT",
        SequentialDecision.CONTINUE: "CONTINUE",
        SequentialDecision.INSUFFICIENT: "INSUFFICIENT",
    }[o.decision]


def _aggregate(votes: list[str], reject_is_blocking: bool) -> ThreeLayerDecision:
    """Majority aggregation with optional reject-blocking asymmetry."""
    n_accept = votes.count("ACCEPT")
    n_reject = votes.count("REJECT")
    n_continue = votes.count("CONTINUE")
    n_insufficient = votes.count("INSUFFICIENT")

    # Hard-block: any REJECT in blocking mode.
    # Per audit A2 F2 (2026-06-03): drop the `n_insufficient < 2`
    # qualifier. When 2 layers are INSUFFICIENT and 1 REJECTs, the
    # REJECT is the only signal we have — trust it MORE not less.
    if reject_is_blocking and n_reject >= 1:
        return ThreeLayerDecision.REJECT
    if n_accept >= 2:
        return ThreeLayerDecision.ACCEPT
    if n_reject >= 2:
        return ThreeLayerDecision.REJECT
    if n_insufficient >= 2:
        return ThreeLayerDecision.INSUFFICIENT
    return ThreeLayerDecision.CONTINUE


def evaluate_three_layer(
    *,
    sleeve_returns: pd.Series,
    prior_mean_sharpe: float,
    n_trials_across_research: Optional[int] = None,
    family: Optional[str] = None,
    obf_boundary: Optional[OBrienFlemingBoundary] = None,
    obf_month: Optional[int] = None,
    prior_sd: float = 0.5,
    threshold_sharpe: float = 0.50,
    bayesian_accept_p: float = 0.80,
    bayesian_reject_p: float = 0.20,
    dsr_accept: float = 0.90,
    dsr_reject: float = 0.50,
    reject_is_blocking: bool = True,
    var_sr_across_trials: Optional[float] = None,
    frequency: str = "monthly",
    forward_oos_returns: Optional[pd.Series] = None,
    min_oos_n_for_block: int = 6,
) -> ThreeLayerResult:
    """Run the 3-layer validation framework on observed sleeve returns.

    Parameters:
      sleeve_returns:        monthly returns observed in window
      prior_mean_sharpe:     P-D8 honest deploy target (Layer 1 prior mean)
      n_trials_across_research: N candidate strategies tested (Layer 2 DeflSR
                              selection-bias correction)
      obf_boundary, obf_month: if provided, Layer 3 runs; else SKIPPED
      threshold_sharpe:      Sharpe value tested for exceedance (Layer 1)
      bayesian_accept_p:     P(Sharpe > threshold) ≥ this → Layer 1 ACCEPT
      dsr_accept:            DeflSR ≥ this → Layer 2 ACCEPT
      reject_is_blocking:    True → any single REJECT vote blocks composite
                              ACCEPT (asymmetric cost rationale)
      var_sr_across_trials:  optional Var(Sharpe) across trials for DeflSR
                              (honest cross-trial dispersion)
    """
    r = sleeve_returns.dropna()
    n = len(r)

    # T1.2 (2026-06-05 audit C3 fix): periods_per_year was hardcoded to 12.
    # Weekly returns got mis-annualized by √(52/12)=2.08×, daily by √(252/12)=4.58×.
    # Any non-monthly candidate's DSR + Bayesian posterior were silently wrong.
    # Now: declared via `frequency` kwarg; sanity-checked against actual data
    # cadence when index is a DatetimeIndex.
    _FREQ_MAP = {"daily": 252, "weekly": 52, "monthly": 12}
    freq_key = (frequency or "monthly").lower()
    if freq_key not in _FREQ_MAP:
        raise ValueError(
            f"frequency must be one of {list(_FREQ_MAP)}, got {frequency!r}"
        )
    periods_per_year = _FREQ_MAP[freq_key]

    # Sanity-check: detect actual cadence from the index if possible.
    if isinstance(r.index, pd.DatetimeIndex) and len(r) >= 3:
        try:
            deltas_days = (r.index.to_series().diff().dt.days.dropna()
                           .astype(float))
            if len(deltas_days):
                median_delta = float(deltas_days.median())
                # Expected ≈ {monthly:30, weekly:7, daily:1}
                expected = {"daily": 1.0, "weekly": 7.0, "monthly": 30.0}[freq_key]
                # Allow ±50% tolerance for irregular calendars
                if abs(median_delta - expected) > 0.5 * expected:
                    logger.warning(
                        "T1.2 frequency_mismatch declared=%s expected_median_days=%.1f "
                        "observed_median_days=%.1f — declared cadence may be wrong, "
                        "verdict math will inflate proportionally",
                        freq_key, expected, median_delta,
                    )
        except Exception:
            pass    # cadence check is best-effort, never blocks

    # Auto-resolve n_trials from family if not explicitly provided.
    # Per Bailey-LdP §3, the correct N is WITHIN-family configurations
    # tried, NOT the codebase total. Family-aware default per
    # family_trial_counter.FAMILY_BUFFER_OVERRIDES.
    #
    # T1.1 (2026-06-05 audit C1 fix): when BOTH explicit AND family are
    # supplied, cross-check against ledger. Within ±2 slack, accept
    # explicit (small mid-stream variants are common). Beyond slack,
    # use max(explicit, ledger) — caller may legitimately TIGHTEN by
    # over-claiming (extra unlogged variants → more trials → lower DSR),
    # but cannot LOOSEN by under-claiming (which would inflate DSR by
    # 3-4× in extreme cases — the attack vector C1 closes).
    if n_trials_across_research is None:
        if family is None:
            raise ValueError(
                "must pass either n_trials_across_research OR family "
                "(family lets the system auto-resolve the within-family "
                "trial count via family_trial_counter)"
            )
        from engine.research.family_trial_counter import count_trials_in_family
        n_trials_across_research = count_trials_in_family(family)
    elif family is not None:
        # T1.1 cross-check path
        from engine.research.family_trial_counter import count_trials_in_family
        ledger_n = count_trials_in_family(family)
        delta = abs(int(n_trials_across_research) - int(ledger_n))
        if delta > 2:
            chosen = max(int(n_trials_across_research), int(ledger_n))
            logger.warning(
                "T1.1 n_trials_mismatch family=%s explicit=%d ledger=%d "
                "(delta=%d > 2 slack) -> using max=%d (anti-under-claim)",
                family, int(n_trials_across_research), int(ledger_n),
                delta, chosen,
            )
            n_trials_across_research = chosen
        elif delta > 0:
            logger.info(
                "T1.1 n_trials family=%s explicit=%d within slack of ledger=%d",
                family, int(n_trials_across_research), int(ledger_n),
            )

    # ── Layer 1: Bayesian ────────────────────────────────────────────
    layer1 = bayesian_sharpe_update(
        sleeve_returns=r,
        prior_mean=prior_mean_sharpe,
        prior_sd=prior_sd,
        threshold=threshold_sharpe,
        accept_posterior_prob=bayesian_accept_p,
        reject_posterior_prob=bayesian_reject_p,
        periods_per_year=periods_per_year,
    )
    layer1_vote = _bayesian_to_vote(layer1)

    # ── Layer 2: Deflated Sharpe Ratio ───────────────────────────────
    layer2 = deflated_sharpe_ratio(
        returns=r.values, n_trials=n_trials_across_research,
        var_sr_across_trials=var_sr_across_trials,
        periods_per_year=periods_per_year,
    )
    layer2_vote = _deflated_to_vote(layer2, accept_dsr=dsr_accept,
                                    reject_dsr=dsr_reject)

    # ── Layer 3: OBF sanity check ────────────────────────────────────
    layer3: Optional[BoundaryResult] = None
    layer3_vote = "INSUFFICIENT"
    if obf_boundary is not None and obf_month is not None:
        t_stat = sharpe_t_stat(r.values)
        try:
            layer3 = obf_boundary.decide(observed_t=t_stat, m=obf_month)
            layer3_vote = _obf_to_vote(layer3)
        except ValueError:
            layer3 = None
            layer3_vote = "INSUFFICIENT"

    # ── Aggregate ────────────────────────────────────────────────────
    final = _aggregate([layer1_vote, layer2_vote, layer3_vote],
                       reject_is_blocking=reject_is_blocking)

    # T1.3 (2026-06-05 audit C2 fix): forward-OOS hard gate.
    # mom_hedge_overlay 2026-05-31 precedent: full-window Sharpe 0.8
    # passed all 3 layers; forward-OOS was -1.2 → strategy lost money
    # in production. Now we plumb forward-OOS into the gate so the
    # full-window cannot bless a forward-collapse failure.
    #
    # Rule (asymmetric): negative OOS Sharpe with sufficient obs
    # forces REJECT regardless of in-sample layers. Positive OOS
    # confirms but does not strengthen the decision (we trust the
    # 3-layer aggregate when OOS is non-negative).
    oos_sharpe: Optional[float] = None
    oos_n: int = 0
    oos_blocked = False
    if forward_oos_returns is not None and not forward_oos_returns.empty:
        oos = forward_oos_returns.dropna()
        oos_n = int(len(oos))
        if oos_n >= 1:
            mean_oos = float(oos.mean())
            sd_oos = float(oos.std(ddof=1)) if oos_n > 1 else 0.0
            if sd_oos > 0:
                oos_sharpe = (mean_oos / sd_oos) * (periods_per_year ** 0.5)
                if oos_n >= min_oos_n_for_block and oos_sharpe < 0.0:
                    if final == ThreeLayerDecision.ACCEPT:
                        logger.warning(
                            "T1.3 OOS_block in-sample ACCEPT overridden to REJECT "
                            "by forward_oos_sharpe=%.3f (n_oos=%d, min=%d) — "
                            "mom_hedge_overlay-style forward-collapse prevention",
                            oos_sharpe, oos_n, min_oos_n_for_block,
                        )
                        oos_blocked = True
                        final = ThreeLayerDecision.REJECT
                    elif final == ThreeLayerDecision.CONTINUE:
                        # Even CONTINUE flips to REJECT on negative OOS —
                        # the in-sample evidence cannot be informative
                        # when forward window contradicts it
                        oos_blocked = True
                        final = ThreeLayerDecision.REJECT

    oos_str = (f"OOS Sharpe={oos_sharpe:.3f} (n={oos_n})"
               if oos_sharpe is not None else "OOS not provided")
    rationale = (
        f"Layer 1 (Bayesian P(Sharpe>{threshold_sharpe:.2f}): "
        f"{layer1.posterior_prob_above_threshold:.3f}) → {layer1_vote} | "
        f"Layer 2 (DeflSR: {layer2.deflated_sr:.3f}) → {layer2_vote} | "
        f"Layer 3 (OBF: {'t=' + f'{layer3.observed_t:+.2f}' if layer3 else 'N/A'}) "
        f"→ {layer3_vote} || OOS: {oos_str}"
        + (" → forced REJECT" if oos_blocked else "")
        + f" || composite={final.value}"
    )

    return ThreeLayerResult(
        n_months=n,
        layer1_bayesian=layer1,
        layer2_deflated_sr=layer2,
        layer3_obf=layer3,
        layer1_vote=layer1_vote,
        layer2_vote=layer2_vote,
        layer3_vote=layer3_vote,
        final_decision=final,
        rationale=rationale,
        evidence_passed=(final == ThreeLayerDecision.ACCEPT),
        forward_oos_sharpe=oos_sharpe,
        forward_oos_n=oos_n,
        forward_oos_blocked=oos_blocked,
    )
