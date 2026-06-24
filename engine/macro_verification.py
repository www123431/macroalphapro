"""
Macro forecast verification + recent brief retrieval.

This module covers:
  - verify_macro_forecasts        : Brier scoring (deterministic)
  - generate_reflections_for_macro: reflection write loop
  - get_recent_macro_briefs       : MACRO-P P-AUDIT panel feed

MACRO-V (2026-05-04). Deterministic / 0 LLM. Implements proper-scoring-rule
verification (Brier 1950) on regime_assessment forecasts. For each
AlphaMemory row with source="macro_research" + macro_data_snapshot present
+ horizon expired + era_verdict NULL, fetch the actual regime at horizon
expiry from RegimeSnapshot and compute multi-class Brier:

    Brier = (1/K) × Σ_k (p_k − o_k)²

where K = 3 classes (risk-on / neutral / risk-off), p_k is forecast prob
on class k, o_k is the indicator on actual class.

Forecast probability distribution (rule-based, no LLM):
- p[stated_regime]    = confidence_raw                    # primary
- p[other_2]          = (1 - confidence_raw) / 2 each     # uncertainty mass

Verdict mapping (deterministic):
- Brier ≤ 0.20 → "logic_correct"  (well-calibrated correct prediction)
- 0.20–0.45    → "lucky_guess"     (around uniform-random benchmark of 0.222)
- Brier > 0.45 → "logic_wrong"

Academic anchor:
- Brier 1950 "Verification of Forecasts Expressed in Terms of Probability"
- Selten 1998 axiomatic characterization (multi-class generalization)
- Diebold-Lee-Weinbach 1994 ex-ante caveat — we use FILTERED probabilities
  from the live RegimeSnapshot at horizon date, not ex-post smoothed.

Cross-references:
- engine/agents/macro_research/agent.py:_run_internal (writes macro_data_snapshot)
- engine/memory.py:RegimeSnapshot (provides actual regime label)
- docs/decisions/macro_research_capability_pipeline.md (spec)
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REGIME_CLASSES = ("risk-on", "neutral", "risk-off")

# Horizon labels → days
HORIZON_DAYS: dict[str, int] = {
    "1M":  21,    # ≈ 1 trading month
    "3M":  63,
    "6M":  126,
    "12M": 252,
}

# Verdict thresholds (multi-class Brier with K=3 → uniform-random baseline = 0.222)
BRIER_LOGIC_CORRECT_MAX = 0.20
BRIER_LUCKY_GUESS_MAX   = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# Brier scorer
# ─────────────────────────────────────────────────────────────────────────────

def _build_forecast_distribution(
    regime_label: str | None,
    confidence_raw: float | None,
) -> dict[str, float]:
    """
    Build a 3-class probability distribution from a hard-label forecast +
    confidence. Conservative split: confidence on stated, remaining mass
    uniform on others.
    """
    p = {c: 0.0 for c in REGIME_CLASSES}
    if regime_label is None or regime_label not in REGIME_CLASSES:
        # Uniform fallback if forecast was malformed
        return {c: 1.0 / len(REGIME_CLASSES) for c in REGIME_CLASSES}
    conf = float(confidence_raw) if confidence_raw is not None else 0.5
    conf = max(0.34, min(0.99, conf))   # clamp: at least uniform / at most near-1
    p[regime_label] = conf
    others = [c for c in REGIME_CLASSES if c != regime_label]
    leftover = (1.0 - conf) / max(1, len(others))
    for c in others:
        p[c] = leftover
    return p


def _brier_score(forecast: dict[str, float], actual_regime: str) -> float:
    """Multi-class Brier on K=3."""
    score = 0.0
    for c in REGIME_CLASSES:
        p = forecast.get(c, 0.0)
        o = 1.0 if c == actual_regime else 0.0
        score += (p - o) ** 2
    return score / len(REGIME_CLASSES)


def _brier_to_verdict(brier: float) -> str:
    if brier <= BRIER_LOGIC_CORRECT_MAX:
        return "logic_correct"
    if brier <= BRIER_LUCKY_GUESS_MAX:
        return "lucky_guess"
    return "logic_wrong"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry: verify_macro_forecasts
# ─────────────────────────────────────────────────────────────────────────────

def verify_macro_forecasts(
    today: datetime.date | None = None,
    *,
    session: Any | None = None,
) -> dict:
    """
    Scan AlphaMemory[source=macro_research, era_verdict IS NULL] and verify
    any forecast whose horizon has expired. Returns a summary dict:

        {
          "as_of":        ISO date,
          "n_scanned":    total NULL-verdict rows considered,
          "n_skipped":    rows still within horizon (no actual yet),
          "n_verified":   rows newly verified (era_verdict written),
          "n_failed":     rows that errored during verification,
          "details":      list[{id, decision_date, regime_forecast,
                                actual_regime, brier, verdict}]
        }
    """
    from engine.memory import (
        AlphaMemory, RegimeSnapshot, SessionFactory,
    )
    from sqlalchemy import func

    if today is None:
        today = datetime.date.today()

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = (
            sess.query(AlphaMemory)
                .filter(AlphaMemory.source == "macro_research")
                .filter(AlphaMemory.era_verdict.is_(None))
                .filter(AlphaMemory.macro_data_snapshot.isnot(None))
                .order_by(AlphaMemory.decision_date.asc())
                .all()
        )

        n_scanned  = len(rows)
        n_skipped  = 0
        n_verified = 0
        n_failed   = 0
        details: list[dict] = []

        for am in rows:
            try:
                snap = json.loads(am.macro_data_snapshot or "{}")
            except Exception as e:
                n_failed += 1
                details.append({
                    "id":             int(am.id),
                    "decision_date":  str(am.decision_date),
                    "error":          f"snapshot parse failed: {e}",
                })
                continue

            horizon = snap.get("horizon", "1M")
            h_days = HORIZON_DAYS.get(horizon, HORIZON_DAYS["1M"])
            expected_verify_date = am.decision_date + datetime.timedelta(days=h_days)

            if expected_verify_date > today:
                n_skipped += 1
                continue

            forecast_regime = snap.get("regime_assessment")
            confidence_raw = snap.get("confidence_raw")

            actual_snap = (
                sess.query(RegimeSnapshot)
                    .filter(RegimeSnapshot.as_of_date <= expected_verify_date)
                    .order_by(RegimeSnapshot.as_of_date.desc())
                    .first()
            )
            if actual_snap is None:
                n_skipped += 1
                continue
            actual_regime = actual_snap.regime

            forecast_dist = _build_forecast_distribution(forecast_regime, confidence_raw)
            brier = _brier_score(forecast_dist, actual_regime)
            verdict = _brier_to_verdict(brier)

            am.era_verdict = verdict
            am.era_score = round(brier, 4)
            am.verified_at = datetime.datetime.utcnow()
            am.era_reasoning = (
                f"Brier={brier:.4f} on K={len(REGIME_CLASSES)}-class. "
                f"forecast={forecast_regime} (conf={confidence_raw}), "
                f"actual at {expected_verify_date}={actual_regime}. "
                f"Method: deterministic multi-class Brier (Brier 1950)."
            )
            n_verified += 1
            details.append({
                "id":             int(am.id),
                "decision_date":  str(am.decision_date),
                "horizon":        horizon,
                "verify_date":    str(expected_verify_date),
                "regime_forecast": forecast_regime,
                "actual_regime":  actual_regime,
                "brier":          round(brier, 4),
                "verdict":        verdict,
            })

        sess.commit()

        return {
            "as_of":      today.isoformat(),
            "n_scanned":  n_scanned,
            "n_skipped":  n_skipped,
            "n_verified": n_verified,
            "n_failed":   n_failed,
            "details":    details,
        }
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# MACRO-R: Reflection write loop
# ─────────────────────────────────────────────────────────────────────────────

def generate_reflections_for_macro(
    model: Any | None = None,
    today: datetime.date | None = None,
) -> dict:
    """
    For each AlphaMemory row with source="macro_research" + era_verdict NOT NULL
    (Brier already scored) but no corresponding AgentReflection row yet,
    generate a 4-section reflection memo via the existing
    engine.agents.reflection.build_and_persist_reflection helper.

    Persisted to AgentReflection (agent_id="macro_research"), feeds back
    into the retrieval loop already wired in MacroResearchAgent._run_internal.

    Returns dict:
        {
          "as_of":         ISO date,
          "n_eligible":    rows ready for reflection (verdict present + no refl)
          "n_written":     reflections persisted,
          "n_skipped":     already had a reflection,
          "n_failed":      LLM / schema errors,
          "details":       list[{alpha_memory_id, reflection_id?, error?}]
        }

    Layer note: generate_reflection_text uses LLM (Layer 1 capability text
    generation, allowed). Brier outcome scoring (era_verdict) is upstream
    and already deterministic — Layer 2 stays 0 LLM.
    """
    from engine.memory import (
        AlphaMemory, AgentReflection, SessionFactory,
    )
    from engine.agents.reflection import (
        build_and_persist_reflection, ReflectionInput,
    )

    if today is None:
        today = datetime.date.today()

    out_details: list[dict] = []
    n_eligible = 0
    n_written  = 0
    n_skipped  = 0
    n_failed   = 0

    with SessionFactory() as sess:
        rows = (
            sess.query(AlphaMemory)
                .filter(AlphaMemory.source == "macro_research")
                .filter(AlphaMemory.era_verdict.isnot(None))
                .order_by(AlphaMemory.decision_date.asc())
                .all()
        )

        # Existing reflection coverage by decision_ref_id
        existing_refl_ids = {
            r.decision_ref_id for r in
            sess.query(AgentReflection)
                .filter(AgentReflection.agent_id == "macro_research")
                .filter(AgentReflection.decision_ref_id.isnot(None))
                .all()
        }

        for am in rows:
            n_eligible += 1
            if am.id in existing_refl_ids:
                n_skipped += 1
                continue

            # Build snapshot for reflection input
            try:
                snap = json.loads(am.macro_data_snapshot or "{}")
            except Exception:
                snap = {}

            decision_summary = {
                "sector":            "macro",
                "direction":         snap.get("regime_assessment") or "neutral",
                "regime_assessment": snap.get("regime_assessment"),
                "horizon":           snap.get("horizon"),
                "confidence":        am.confidence,
                "rationale_excerpt": (am.logic_chain or "")[:300],
                "tail_risk":         (snap.get("tail_risk_narrative") or "")[:200],
                "era_verdict":       am.era_verdict,
                "era_score":         am.era_score,
            }

            inp = ReflectionInput(
                agent_id        = "macro_research",
                decision_date   = am.decision_date,
                decision_summary= decision_summary,
                realized_outcome= float(am.era_score) if am.era_score is not None else None,
                factor_context  = None,
                decision_ref_id = int(am.id),
                prior_reflections= [],
            )

            try:
                rid = build_and_persist_reflection(inp, model=model)
                n_written += 1
                out_details.append({
                    "alpha_memory_id": int(am.id),
                    "reflection_id":   int(rid),
                    "decision_date":   str(am.decision_date),
                    "verdict":         am.era_verdict,
                })
            except Exception as e:
                n_failed += 1
                out_details.append({
                    "alpha_memory_id": int(am.id),
                    "error":           str(e)[:200],
                })

    return {
        "as_of":      today.isoformat(),
        "n_eligible": n_eligible,
        "n_written":  n_written,
        "n_skipped":  n_skipped,
        "n_failed":   n_failed,
        "details":    out_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MACRO-P: panel feed — recent macro briefs for supervisor decision context
# ─────────────────────────────────────────────────────────────────────────────

def get_recent_macro_briefs(
    lookback_days: int = 30,
    *,
    today: datetime.date | None = None,
) -> list[dict]:
    """
    Returns recent AlphaMemory[source=macro_research] rows (newest first)
    flattened into UI-friendly dicts. Used by the P-AUDIT decision panel
    AUDIT tab to surface macro context at supervisor approval time.

    Each item includes:
        decision_date / regime_assessment / confidence / horizon /
        key_macro_driver / era_verdict / era_score / verified
    """
    from engine.memory import AlphaMemory, SessionFactory

    if today is None:
        today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=lookback_days)

    out: list[dict] = []
    with SessionFactory() as sess:
        rows = (
            sess.query(AlphaMemory)
                .filter(AlphaMemory.source == "macro_research")
                .filter(AlphaMemory.decision_date >= cutoff)
                .order_by(AlphaMemory.decision_date.desc())
                .all()
        )

    for am in rows:
        snap: dict = {}
        if am.macro_data_snapshot:
            try:
                snap = json.loads(am.macro_data_snapshot)
            except Exception:
                snap = {}
        out.append({
            "alpha_memory_id":   int(am.id),
            "decision_date":     str(am.decision_date),
            "regime_assessment": snap.get("regime_assessment"),
            "horizon":           snap.get("horizon"),
            "confidence":        am.confidence,
            "key_macro_driver":  (am.logic_chain or "")[:200],
            "tail_risk_excerpt": (snap.get("tail_risk_narrative") or "")[:200],
            "contradicts_current_regime":
                                 snap.get("contradicts_current_regime"),
            "era_verdict":       am.era_verdict,
            "era_score":         am.era_score,
            "verified_at":       (am.verified_at.isoformat()
                                  if am.verified_at else None),
        })
    return out
