"""engine.research.belief — Belief Layer Phase 1: predict-then-observe.

Doctrine
========
Every dispatch emits a predicted verdict distribution BEFORE the strict
gate runs. Predictions are **AIR-GAPPED** from verdict logic:

  * predictions live in `data/research/predictions.jsonl`, NOT in
    events.jsonl
  * lens / strict_gate / template / dispatcher logic MUST NOT import
    this module beyond the single emit hook in `dispatch_factor_spec`
  * structural invariant enforced by `tests/test_belief.py`

Why
===
Lopez de Prado 2014 False Strategy Theorem + Tetlock superforecasting
calibration + Arnott-Harvey-Markowitz 2019 Backtesting Protocols
step 6 ("track all decisions and their consequences").

A system that doesn't commit to a belief before observing cannot be
held accountable; without accountability, no calibration; without
calibration, no trustworthy verdict-trust signal. This is the
foundation of every later Belief Layer phase.

Phases
======
Phase 1 (THIS COMMIT): deterministic prior; predict + log per dispatch.
Phase 2: autopsy consumer reads predictions + verdicts → diagnoses surprises.
Phase 3: calibration dashboard surfaces Brier score by family.
Phase 4: closed-loop prior update from outcome ledger.
Phase 5: track-record-aware ask in SLM promotion prompts.

Schema (predictions.jsonl row, append-only)
===========================================
  prediction_id            uuid4
  ts                       ISO-8601 UTC
  session_id               active session pointer or 'unknown'
  subject_id               factor / hypothesis id
  family                   mechanism family (or null)
  predicted_verdict_dist   {GREEN: float, MARGINAL: float, RED: float} sums to 1
  anchor_evidence          tuple of short strings explaining the prior basis
  predicted_load_bearing   which assumptions could flip the prediction
  prediction_basis         audit string — how the dist was computed
  inputs                   {paper_year, signal_kind, ...} echo of inputs
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = _REPO_ROOT / "data" / "research" / "predictions.jsonl"

_LOCK = threading.Lock()


# ── Default prior (HXZ 2020 65% replication failure anchor) ─────────
# GREEN here means "would survive strict gate net-of-cost". HXZ found
# ~35% of published anomalies replicate; among those, fewer survive
# 80bp RT cost. So GREEN base rate ~0.20 is a calibrated starting point.
DEFAULT_PRIOR: dict[str, float] = {
    "GREEN":    0.20,
    "MARGINAL": 0.40,
    "RED":      0.40,
}

# Family-specific priors — overrides DEFAULT_PRIOR when family is known.
# Numbers calibrated from public replication audits + this project's own
# lessons (FF5 RMW absorbs most PROFITABILITY; MOMENTUM crashes post-2000s;
# VALUE compressed; QUALITY still has room).
FAMILY_PRIOR_OVERRIDES: dict[str, dict[str, float]] = {
    "PROFITABILITY":     {"GREEN": 0.12, "MARGINAL": 0.50, "RED": 0.38},
    "MOMENTUM":          {"GREEN": 0.18, "MARGINAL": 0.45, "RED": 0.37},
    "VALUE":             {"GREEN": 0.15, "MARGINAL": 0.45, "RED": 0.40},
    "QUALITY":           {"GREEN": 0.20, "MARGINAL": 0.45, "RED": 0.35},
    "VOL":               {"GREEN": 0.20, "MARGINAL": 0.45, "RED": 0.35},
    "DEEP_LEARNING":     {"GREEN": 0.10, "MARGINAL": 0.30, "RED": 0.60},
    "ANALYST_REVISION":  {"GREEN": 0.22, "MARGINAL": 0.43, "RED": 0.35},
    "INVESTMENT":        {"GREEN": 0.13, "MARGINAL": 0.50, "RED": 0.37},
    "ACCRUAL":           {"GREEN": 0.16, "MARGINAL": 0.44, "RED": 0.40},
    "CARRY":             {"GREEN": 0.22, "MARGINAL": 0.45, "RED": 0.33},
    "TSMOM":             {"GREEN": 0.20, "MARGINAL": 0.45, "RED": 0.35},
}

# Smoothing strength (Dirichlet pseudo-count) for observed family verdicts.
# Lower = trust observations more; higher = stick to prior.
#
# 2026-06-11 initial: 3.0 ("matches 5 observations").
# 2026-06-22 W6-rigor-A revision: 1.0. The threshold/alpha sweep on
# 85 historical autopsies (engine.research.belief_track_record_rigor.
# sweep_threshold_alpha) showed alpha=3 over-smooths family signal —
# step Brier 0.390 at (N=5, alpha=3) vs 0.332 at (N=3, alpha=1) on
# the same time-aware family-prior data. Family signal was being
# pulled too aggressively toward uniform. Lower alpha + lower
# threshold yields ~15% Brier improvement. Conservative middle point
# chosen (not the corner N=1/alpha=0.5 which could over-fit the
# small 85-pair sample).
_SMOOTHING_ALPHA = 1.0

# Threshold (in observed family verdicts) above which the observed
# posterior dominates the FAMILY_PRIOR_OVERRIDES fallback.
# 2026-06-11 initial: 5.
# 2026-06-22 W6-rigor-A revision: 3. Per the sweep above; 5 was
# filtering 13/21 families' data into the hand-calibrated overrides
# even when 3-4 observations were available.
_OBSERVED_POSTERIOR_THRESHOLD = 3


# ──────────────────────────────────────────────────────────────────────
# W7-arxiv-v05 (2026-06-22): per-family ensemble blend, FEATURE-FLAGGED
# OFF by default. Wires the v0.5 sweep finding (per-family w_fam mixing
# raw time-aware family-empirical with the existing pipeline output)
# into predict_verdict. Enabling this restarts the calibration measure-
# ment clock (paper Section 4.6 caveat) — capital-decision-class change.
#
# To enable:
#   set env var BELIEF_ENSEMBLE_BLEND_ENABLED=1
#   OR set the module attribute belief.BELIEF_ENSEMBLE_BLEND_ENABLED = True
#
# When enabled, predict_verdict's final dist is:
#   dist = w_fam × raw_family_empirical_dist + (1 − w_fam) × existing_dist
# where w_fam is FAMILY_OPTIMAL_W[family] if present, else
# GLOBAL_W_FALLBACK. raw_family_empirical_dist is the time-aware
# leave-one-out distribution of prior factor_verdict_filed events for
# the family (no smoothing, no penalty — the pipeline's
# smoothing/penalty/decay already applies in the (1−w_fam) leg).
#
# w_fam values from data/research/belief_ensemble_sweep.json (n=92):
import os as _os

# ACTIVATED 2026-06-22 (W7-arxiv-v07) per evidence-based principal
# decision: the W7-arxiv-v05 sweep on 92 autopsies showed 34% in-sample
# Brier reduction (0.375 → 0.246) with consistent per-family signal
# across 8 eligible families. Daily belief refresh cron auto-measures
# the realized outcome. Revertible by setting env var
# BELIEF_ENSEMBLE_BLEND_ENABLED=0 (overrides the True default) OR by
# editing this default back to False.
BELIEF_ENSEMBLE_BLEND_ENABLED = (
    _os.environ.get("BELIEF_ENSEMBLE_BLEND_ENABLED", "1") == "1"
)

# 2026-06-22 W7-arxiv-v09 honest amend (LOOCV-driven correction):
# The v0.5 in-sample sweep produced per-family w_fam values 0.6-1.0,
# yielding in-sample Brier 0.254. The v0.8 LOOCV honesty pass revealed
# that the per-family w optimization OVERFITS small-N families:
#
#   Pure family-empirical (w=1.0 forced, no per-family tuning): 0.260
#   LOOCV W7 per-family ensemble (sweep's per-family w):         0.278
#
# Per-family tuning added +0.018 noise vs simply trusting family-
# empirical entirely. The honest architectural move is to switch
# w_fam = 1.0 globally — pure family-empirical when N >= 3, the
# existing pipeline (w_fam = 0 fallback) when N < 3. This is
# directionally what the data says: when family history is
# sufficient, trust it and ignore the LLM-driven prior.
#
# Older per-family values preserved in the v0.5 sweep output at
# data/research/belief_ensemble_sweep.json for audit.
FAMILY_OPTIMAL_W: dict[str, float] = {
    # All 1.0 per LOOCV: pure family-empirical beats per-family w tuning.
    "CROSS_SEC_UNKNOWN":  1.0,
    "VRP":                1.0,
    "EVENT_DRIFT":        1.0,
    "SPANNING_HML":       1.0,
    "PROFITABILITY":      1.0,
    "SPANNING_SMB":       1.0,
    "SPANNING_MOM":       1.0,
    "SPANNING_CMA":       1.0,
}
GLOBAL_W_FALLBACK: float = 1.0
_ENSEMBLE_MIN_ELIGIBLE_N: int = 3


def _raw_family_empirical_at(family: str) -> tuple[dict[str, float], int]:
    """Time-aware raw family-empirical distribution (no smoothing).

    Reads factor_verdict_filed events for the family from the event store.
    Natural time-awareness: events that haven't been emitted yet are not
    in the store, so calling this at prediction time only sees prior
    verdicts. Returns (dist, n_eligible). n_eligible = 0 → caller skips
    ensemble blend.
    """
    try:
        from engine.research_store import store
        events = store.filter_events(
            event_type="factor_verdict_filed", family=family,
        )
    except Exception:
        return {"GREEN": 0.0, "MARGINAL": 0.0, "RED": 0.0}, 0
    counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
    for ev in events:
        v = ev.verdict.value if hasattr(ev.verdict, "value") else str(ev.verdict)
        if v in counts:
            counts[v] += 1
    n = sum(counts.values())
    if n == 0:
        return {"GREEN": 0.0, "MARGINAL": 0.0, "RED": 0.0}, 0
    return {k: counts[k] / n for k in counts}, n


def _apply_ensemble_blend(
    dist: dict[str, float],
    family_upper: str,
) -> tuple[dict[str, float], str | None]:
    """If the feature flag is enabled AND we have ≥ threshold eligible
    family verdicts, blend at w_fam. Returns (new_dist, audit_note).

    audit_note is None when no blend applied; string when applied (for
    inclusion in Prediction.prediction_basis).
    """
    if not BELIEF_ENSEMBLE_BLEND_ENABLED:
        return dist, None
    if not family_upper:
        return dist, None
    fam_emp, n = _raw_family_empirical_at(family_upper)
    if n < _ENSEMBLE_MIN_ELIGIBLE_N:
        return dist, None
    w_fam = FAMILY_OPTIMAL_W.get(family_upper, GLOBAL_W_FALLBACK)
    blended = {
        k: w_fam * fam_emp[k] + (1.0 - w_fam) * dist[k]
        for k in ("GREEN", "MARGINAL", "RED")
    }
    note = (f"W7-ensemble blend applied (w_fam={w_fam}, n_eligible={n}, "
              f"family-empirical raw mixed with pipeline output)")
    return blended, note

# Threshold above which n_trials family pressure is "load-bearing"
_N_TRIALS_PRESSURE_THRESHOLD = 10

# Threshold (in years) above which paper publication age triggers decay risk
_POST_PUBLICATION_DECAY_YEARS = 10


@_dc.dataclass(frozen=True)
class Prediction:
    """The forecast committed BEFORE dispatch runs.

    Immutable — to correct, append a new prediction and reference the
    prior via tags (not implemented in Phase 1; reserve for Phase 2).
    """
    prediction_id:           str
    ts:                      str
    session_id:              str
    subject_id:              str
    family:                  Optional[str]
    predicted_verdict_dist:  dict[str, float]
    anchor_evidence:         tuple[str, ...]
    predicted_load_bearing:  tuple[str, ...]
    prediction_basis:        str
    inputs:                  dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = _dc.asdict(self)
        d["anchor_evidence"]        = list(self.anchor_evidence)
        d["predicted_load_bearing"] = list(self.predicted_load_bearing)
        return d


# ── Helpers ─────────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize(dist: dict[str, float]) -> dict[str, float]:
    """Clip negatives → 0, renormalize to sum 1. Returns new dict."""
    clipped = {k: max(0.0, float(v)) for k, v in dist.items()}
    total = sum(clipped.values())
    if total <= 0:
        return dict(DEFAULT_PRIOR)
    return {k: v / total for k, v in clipped.items()}


def _family_observed_dist(family: str) -> tuple[dict[str, float], int]:
    """Read past factor_verdict_filed events for `family`, Dirichlet-smoothed
    posterior over (GREEN, MARGINAL, RED). NEUTRAL verdicts skipped (they
    aren't strict-gate outcomes).

    Returns (dist, N_observed). N=0 means no informative posterior; caller
    should fall back to FAMILY_PRIOR_OVERRIDES or DEFAULT_PRIOR.

    NOTE: this is the ONE place belief reads from events.jsonl. It reads
    PAST verdicts only — a verdict-time prediction does not see its own
    eventual outcome. Air-gap holds.
    """
    try:
        from engine.research_store import store
        events = store.filter_events(
            event_type = "factor_verdict_filed",
            family     = family,
        )
    except Exception as exc:
        logger.warning("belief: family_observed_dist read failed: %s", exc)
        return dict(DEFAULT_PRIOR), 0

    counts = {"GREEN": 0, "MARGINAL": 0, "RED": 0}
    for ev in events:
        v = ev.verdict.value if hasattr(ev.verdict, "value") else str(ev.verdict)
        if v in counts:
            counts[v] += 1

    n = sum(counts.values())
    if n == 0:
        return dict(DEFAULT_PRIOR), 0

    # Dirichlet posterior mean: (count_i + alpha) / (N + 3*alpha)
    denom = n + 3 * _SMOOTHING_ALPHA
    dist = {k: (counts[k] + _SMOOTHING_ALPHA) / denom for k in counts}
    return _normalize(dist), n


def _family_trial_pressure(family: str) -> tuple[int, bool]:
    """Returns (n_trials, is_load_bearing). is_load_bearing iff > threshold."""
    try:
        from engine.research.family_trial_counter import count_trials_in_family
        n = count_trials_in_family(family)
        return n, n > _N_TRIALS_PRESSURE_THRESHOLD
    except Exception as exc:
        logger.warning("belief: family_trial_pressure read failed: %s", exc)
        return 0, False


def _apply_n_trials_penalty(
    dist: dict[str, float], n_trials: int,
) -> dict[str, float]:
    """Bailey-LdP §3: each marginal trial above threshold compounds DSR
    penalty. We model this as multiplicative GREEN shrinkage redistributed
    to MARGINAL (not RED — penalty doesn't make a real factor fake, it
    makes the verdict harder to justify)."""
    if n_trials <= _N_TRIALS_PRESSURE_THRESHOLD:
        return dist
    # Linear: every 5 trials above threshold → 10% multiplicative GREEN shrink,
    # capped at 50% shrink total. Conservative.
    excess = n_trials - _N_TRIALS_PRESSURE_THRESHOLD
    shrink = min(0.50, 0.10 * (excess / 5.0))
    green_shed = dist["GREEN"] * shrink
    return _normalize({
        "GREEN":    dist["GREEN"] - green_shed,
        "MARGINAL": dist["MARGINAL"] + green_shed,
        "RED":      dist["RED"],
    })


def _apply_publication_age_penalty(
    dist: dict[str, float], paper_year: int, current_year: int,
) -> tuple[dict[str, float], bool]:
    """McLean-Pontiff 2016: 32-58% Sharpe drop post-publication. Old papers
    have higher RED prior. Returns (new_dist, was_applied)."""
    age = current_year - paper_year
    if age <= _POST_PUBLICATION_DECAY_YEARS:
        return dist, False
    # Shift 0.10 of GREEN to RED for each decade past threshold, capped 0.20.
    shift = min(0.20, 0.10 * (age - _POST_PUBLICATION_DECAY_YEARS) / 10.0)
    green_shed = dist["GREEN"] * shift
    return _normalize({
        "GREEN":    dist["GREEN"] - green_shed,
        "MARGINAL": dist["MARGINAL"],
        "RED":      dist["RED"] + green_shed,
    }), True


# ── Public API ──────────────────────────────────────────────────────


def predict_verdict(
    *,
    subject_id:     str,
    family:         Optional[str],
    paper_year:     Optional[int]  = None,
    signal_kind:    Optional[str]  = None,
    extra_inputs:   Optional[dict] = None,
    current_year:   Optional[int]  = None,
) -> Prediction:
    """Compute the predicted verdict distribution for an about-to-run dispatch.

    DETERMINISTIC — no LLM. The basis string records every adjustment so
    the prediction is fully auditable + reproducible from inputs alone.

    `family` should be the mechanism_family ('PROFITABILITY' / 'MOMENTUM' /
    etc). Case-insensitive lookup against FAMILY_PRIOR_OVERRIDES.
    """
    if current_year is None:
        # 2026-06-11 frozen — see Phase 4 for outcome-driven prior refresh.
        current_year = 2026

    inputs: dict[str, Any] = {
        "paper_year":  paper_year,
        "signal_kind": signal_kind,
    }
    if extra_inputs:
        inputs.update({k: v for k, v in extra_inputs.items()
                        if isinstance(v, (str, int, float, bool, type(None)))})

    anchors: list[str] = []
    load_bearing: list[str] = []
    basis_parts: list[str] = []

    # Step 1: family prior — priority order is
    #   (a) belief-4 closed-loop calibrated prior (autopsies-driven)
    #   (b) observed posterior from events.jsonl (raw verdict counts)
    #   (c) FAMILY_PRIOR_OVERRIDES (hand-calibrated by family)
    #   (d) DEFAULT_PRIOR (HXZ 65% replication anchor)
    fam_upper = (family or "").upper()
    # (a) belief-4 — closed loop from autopsies
    calibrated = None
    try:
        from engine.research.belief_prior_calibration import (
            calibrated_family_prior,
        )
        calibrated = calibrated_family_prior(fam_upper) if fam_upper else None
    except Exception:
        logger.debug("belief-4 calibration lookup failed; falling through", exc_info=True)
        calibrated = None

    if calibrated is not None:
        dist = calibrated
        basis_parts.append(
            f"belief-4 closed-loop calibrated prior (autopsies-driven, "
            f"strategy_family={fam_upper})"
        )
        anchors.append(f"calibrated prior from autopsies for {fam_upper}")
        load_bearing.append("calibration_source:belief_4")
    else:
        obs_dist, n_obs = _family_observed_dist(fam_upper) if fam_upper else (
            dict(DEFAULT_PRIOR), 0
        )
        if n_obs >= _OBSERVED_POSTERIOR_THRESHOLD:
            dist = obs_dist
            basis_parts.append(f"observed family posterior N={n_obs} (Dirichlet α={_SMOOTHING_ALPHA})")
            anchors.append(f"family observed N={n_obs} verdicts")
        elif fam_upper in FAMILY_PRIOR_OVERRIDES:
            dist = dict(FAMILY_PRIOR_OVERRIDES[fam_upper])
            basis_parts.append(f"family prior override (N_observed={n_obs} below threshold {_OBSERVED_POSTERIOR_THRESHOLD})")
            anchors.append(f"calibrated prior for family={fam_upper}")
        else:
            dist = dict(DEFAULT_PRIOR)
            basis_parts.append("default prior (HXZ 2020 65% replication failure anchor)")
            anchors.append("default prior (no family-specific calibration)")

    # Step 2: n_trials penalty (Bailey-LdP)
    if fam_upper:
        n_trials, pressure = _family_trial_pressure(fam_upper.lower())
        if pressure:
            dist = _apply_n_trials_penalty(dist, n_trials)
            basis_parts.append(
                f"Bailey-LdP n_trials penalty (family n={n_trials} > "
                f"{_N_TRIALS_PRESSURE_THRESHOLD})"
            )
            anchors.append(f"family trials n={n_trials} → DSR inflation")
            load_bearing.append("family_trials")
        else:
            basis_parts.append(f"n_trials={n_trials} below pressure threshold (no penalty)")

    # Step 3: post-publication decay (McLean-Pontiff)
    if paper_year is not None:
        dist, applied = _apply_publication_age_penalty(dist, paper_year, current_year)
        if applied:
            basis_parts.append(
                f"McLean-Pontiff post-pub decay (paper {paper_year}, "
                f"age {current_year - paper_year}y > {_POST_PUBLICATION_DECAY_YEARS}y)"
            )
            anchors.append(f"paper age {current_year - paper_year}y triggers decay risk")
            load_bearing.append("post_publication_decay")

    # Step 3.5: W7-arxiv-v05 ensemble blend (feature-flagged OFF default).
    # When enabled, mixes raw time-aware family-empirical at w_fam with
    # the existing pipeline output. Capital-decision-class — enabling
    # restarts calibration measurement (paper Section 4.6 caveat).
    dist, _ensemble_note = _apply_ensemble_blend(dist, fam_upper)
    if _ensemble_note is not None:
        basis_parts.append(_ensemble_note)
        anchors.append("W7-ensemble per-family blend (v0.5 sweep evidence)")
        load_bearing.append("ensemble_blend")

    # Step 4: mark spanning risk for heavily-mature families
    if fam_upper in {"PROFITABILITY", "VALUE", "MOMENTUM"}:
        load_bearing.append("spanning_risk")
        anchors.append(f"family {fam_upper} → FF5 spanning probable")

    # Resolve session id
    session_id = _resolve_session_id()

    pred = Prediction(
        prediction_id          = str(uuid.uuid4()),
        ts                     = _utc_iso(),
        session_id             = session_id,
        subject_id             = subject_id,
        family                 = fam_upper or None,
        predicted_verdict_dist = _normalize(dist),
        anchor_evidence        = tuple(anchors),
        predicted_load_bearing = tuple(load_bearing),
        prediction_basis       = "; ".join(basis_parts),
        inputs                 = inputs,
    )
    return pred


def _resolve_session_id() -> str:
    """Best-effort session id, mirrors emit._read_active_session pattern."""
    try:
        from engine.sessions import store as session_store
        active = session_store.get_active()
        if active:
            sid = active.get("session_id")
            if sid:
                return sid
    except Exception:
        pass
    import os
    return os.environ.get("CLAUDE_SESSION_ID", "unknown")


def log_prediction(pred: Prediction) -> str:
    """Append the prediction to data/research/predictions.jsonl. Returns
    prediction_id.

    The file is the ONLY belief-layer persistence surface. No other module
    should write to it; no lens / strict_gate / template should read from it
    (would corrupt the air-gap).
    """
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with PREDICTIONS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
    logger.info(
        "belief: logged prediction subject=%s family=%s GREEN=%.2f basis=%s",
        pred.subject_id, pred.family,
        pred.predicted_verdict_dist.get("GREEN", 0.0),
        pred.prediction_basis[:120],
    )
    return pred.prediction_id


def predict_and_log(
    *,
    subject_id:    str,
    family:        Optional[str],
    paper_year:    Optional[int]  = None,
    signal_kind:   Optional[str]  = None,
    extra_inputs:  Optional[dict] = None,
) -> Prediction:
    """Convenience: compute + log in one call. Use this from dispatcher."""
    pred = predict_verdict(
        subject_id   = subject_id,
        family       = family,
        paper_year   = paper_year,
        signal_kind  = signal_kind,
        extra_inputs = extra_inputs,
    )
    try:
        log_prediction(pred)
    except OSError as exc:
        # Log write failure must NOT block dispatch — prediction is
        # quality control, not a gate. Surface loudly but continue.
        logger.error("belief: log_prediction failed (continuing): %s", exc)
    return pred
