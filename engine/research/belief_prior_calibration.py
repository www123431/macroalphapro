"""engine.research.belief_prior_calibration — Belief Layer Phase 4.

Closes the epistemic loop: autopsies (from belief-2) feed back into
belief-1's family prior. The more autopsies accumulate in a strategy
family, the better belief-1's predictions get.

Doctrine
========
- Air-gap preserved: this module reads autopsies (verdict OUTCOMES,
  already produced) + emits a UPDATED family prior. Never reads or
  writes predictions.jsonl. The cycle is:

      belief-1 predicts → verdict emits → belief-2 writes autopsy →
      belief-4 (this) updates prior → belief-1 next prediction uses
      updated prior

- Update rule (conservative): Beta-Dirichlet posterior over observed
  actual_verdict counts within the strategy_family, with the
  prior's pseudo-count weighted to remain anchored until enough
  evidence accumulates.

- Minimum samples to override (calibration cutoff): 5 autopsies in
  family. Below that, belief-1 falls through to its existing
  observation/override/default logic.

- Bias correction: when over_predicted_X count > threshold ratio,
  apply a Bayesian shrink toward the actual distribution observed.
  This is a one-step EM-style correction; belief-4 doesn't loop more.

Inputs
======
- data/research/autopsies.jsonl (belief-2 output)

Output
======
- Pure function: calibrated_family_prior(strategy_family) → dict[verdict, float]
- belief-1 calls this; non-None result overrides FAMILY_PRIOR_OVERRIDES
  and observed posterior; None means "fall through to existing logic"

Why Beta-Dirichlet, not simple bin counts
==========================================
- Small N (5-15 autopsies) is the realistic operating regime for solo
  quant. Simple frequencies are jumpy at low N.
- Dirichlet posterior smooths via pseudo-counts: belief-1's existing
  FAMILY_PRIOR_OVERRIDES become the prior pseudo-counts, and
  autopsies become observations.
- Alpha (smoothing strength) = 3.0 — same as belief-1's existing
  _SMOOTHING_ALPHA for consistency.

Caveat: bootstrap problem
=========================
The first belief-4 update for a new strategy_family uses DEFAULT_PRIOR
as the pseudo-count base. If DEFAULT_PRIOR is significantly off for
the family, the first 5-10 autopsies will be slow to correct.
Mitigation: belief-3 dashboard surfaces the calibration so principal
can manually tweak FAMILY_PRIOR_OVERRIDES if needed. Long-term
belief-4 converges anyway with enough N.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
AUTOPSIES_PATH = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"

# Calibration parameters
MIN_AUTOPSIES_FOR_OVERRIDE = 5    # below this, fall through to belief-1's existing logic
PRIOR_PSEUDO_COUNT_ALPHA   = 3.0   # match belief-1's _SMOOTHING_ALPHA


def _iter_autopsies(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("belief_prior_calibration: %s line %d malformed",
                                 path.name, ln_no)


def _autopsies_for_family(
    strategy_family: str, path: Optional[Path] = None,
) -> list[dict]:
    """Returns NON-SUPERSEDED autopsies for the family. Rows tagged with
    `superseded_by` (set by a later BUG-1 / correction row) are excluded
    so the calibrated prior reflects the corrected verdicts only."""
    fam_upper = (strategy_family or "").upper()
    out: list[dict] = []
    for row in _iter_autopsies(path or AUTOPSIES_PATH):
        if (row.get("strategy_family") or "").upper() != fam_upper:
            continue
        if row.get("superseded_by"):
            continue
        out.append(row)
    return out


def _verdict_counts(autopsies: list[dict]) -> dict[str, int]:
    counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
    for a in autopsies:
        v = a.get("actual_verdict", "")
        if v in counts:
            counts[v] += 1
    return counts


def _precision_weighted_counts(autopsies: list[dict]) -> dict[str, float]:
    """BUG-4 (2026-06-13): weight each verdict observation by precision
    (1/SE^2 ∝ N) so short-sample autopsies don't contribute equally to
    long-sample ones. Strict Bayesian treatment of unequal-N likelihood.

    Implementation: each autopsy contributes weight = n_obs_months /
    REFERENCE_SAMPLE (= 360mo / 30y). N=60 → weight 0.17 (downweight);
    N=360 → weight 1.0 (full). N=755 → weight 2.1 (long-sample upweight).
    Returns floats not ints — Dirichlet posterior handles fractional
    pseudo-counts cleanly.

    When n_obs_months is missing/zero, default to weight 1.0 (treat as
    average) so historical autopsies that don't carry the field still
    contribute normally.
    """
    REFERENCE_SAMPLE = 360.0  # 30 years monthly
    counts = {"GREEN": 0.0, "MARGINAL": 0.0, "RED": 0.0}
    for a in autopsies:
        v = a.get("actual_verdict", "")
        if v not in counts:
            continue
        n = a.get("n_obs_months", 0) or 0
        w = float(n) / REFERENCE_SAMPLE if n > 0 else 1.0
        counts[v] += w
    return counts


def _belief1_base_prior(strategy_family: str) -> dict[str, float]:
    """Lookup belief-1's existing FAMILY_PRIOR_OVERRIDES for this family,
    or DEFAULT_PRIOR if no override."""
    try:
        from engine.research.belief import (
            DEFAULT_PRIOR, FAMILY_PRIOR_OVERRIDES,
        )
    except Exception:
        # Fallback if belief module unavailable
        return {"GREEN": 0.20, "MARGINAL": 0.40, "RED": 0.40}
    fam = (strategy_family or "").upper()
    return dict(FAMILY_PRIOR_OVERRIDES.get(fam, DEFAULT_PRIOR))


def calibrated_family_prior(
    strategy_family: str,
    *,
    autopsies_path: Optional[Path] = None,
) -> Optional[dict[str, float]]:
    """Compute the closed-loop calibrated prior for a strategy family.

    Returns None when:
      - autopsies file missing
      - fewer than MIN_AUTOPSIES_FOR_OVERRIDE autopsies in family

    Returns dict{GREEN, MARGINAL, RED} summing to 1.0 otherwise.

    Math (Dirichlet posterior mean):
      posterior_p_i = (count_i + alpha * base_p_i) / (N + alpha)
    where base_p comes from belief-1's existing prior (override or default)
    and N is observed autopsy count.

    The interpretation: alpha is a "pseudo-count" representing the prior's
    confidence. With alpha=3.0 and base_p_green=0.20, the prior contributes
    "0.6 prior GREEN observations" — small enough that 5+ real autopsies
    move the posterior meaningfully, large enough that 1-2 noisy
    observations don't whipsaw it.
    """
    autopsies = _autopsies_for_family(
        strategy_family, autopsies_path or AUTOPSIES_PATH,
    )
    n = len(autopsies)
    if n < MIN_AUTOPSIES_FOR_OVERRIDE:
        return None

    # BUG-4 (2026-06-13): use precision-weighted counts (1/SE^2 ∝ N)
    # instead of raw integer counts. Strict Bayesian: a 60mo autopsy
    # carries less information than a 360mo autopsy. Reference: 360mo.
    counts = _precision_weighted_counts(autopsies)
    total_w = sum(counts.values())
    base = _belief1_base_prior(strategy_family)
    alpha = PRIOR_PSEUDO_COUNT_ALPHA

    denom = total_w + alpha
    return {
        "GREEN":    (counts["GREEN"]    + alpha * base["GREEN"])    / denom,
        "MARGINAL": (counts["MARGINAL"] + alpha * base["MARGINAL"]) / denom,
        "RED":      (counts["RED"]      + alpha * base["RED"])      / denom,
    }


def calibration_summary(
    strategy_family: str,
    *,
    autopsies_path: Optional[Path] = None,
) -> dict:
    """Diagnostic output for the calibration dashboard / inbox digest.
    Returns dict with autopsy count + base prior + calibrated prior +
    delta vs base, even when N is below override threshold."""
    autopsies = _autopsies_for_family(
        strategy_family, autopsies_path or AUTOPSIES_PATH,
    )
    n = len(autopsies)
    counts = _verdict_counts(autopsies)
    base = _belief1_base_prior(strategy_family)
    calibrated = calibrated_family_prior(
        strategy_family, autopsies_path=autopsies_path,
    )
    return {
        "strategy_family":          strategy_family,
        "n_autopsies":              n,
        "observed_counts":          counts,
        "base_prior":               base,
        "calibrated_prior":         calibrated,
        "override_active":          calibrated is not None,
        "delta_green":              ((calibrated["GREEN"] - base["GREEN"])
                                          if calibrated else 0.0),
    }
