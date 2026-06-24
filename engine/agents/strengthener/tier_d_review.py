"""engine.agents.strengthener.tier_d_review — Tier D role-specific review.

Per docs/spec_role_aware_test_routing.md §15.A3: non-alpha sleeves
(insurance / diversifier / hedge) are routed to Tier D instead of
Tier C. Rationale: Tier C rigor stack assumes alpha-seeking
investment role + alpha t-stat as load-bearing metric. Forcing
insurance / diversifier sleeves through Tier C produces meaningless
verdict semantics ("mom_hedge α t=-2.09" is the EXPECTED state of
a well-functioning insurance overlay, not a problem).

Tier D produces:
  - role-appropriate diagnostic metrics (drawdown, conditional
    return in stress periods, correlation structure stability)
  - human review trigger emitted to /approvals as
    TIER_D_HUMAN_REVIEW_REQUIRED
  - NO automated verdict
  - NO confidence score

Phase 3 will add:
  - Insurance: convexity coefficient (Bondarenko 2014), conditional
    return in 95th percentile worst equity months (Kelly-Pruitt
    2014), max drawdown protection ratio
  - Diversifier: time-varying correlation (Engle DCC 2002), tail
    dependence (Frahm 2005)

Phase 1 (this commit) implements only the ROUTING + a minimal
diagnostic placeholder. Phase 3 fills in the role-specific metrics
after Bondarenko / Kelly / Asness / Ilmanen methodology research.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
TIER_D_LOG_PATH = (REPO_ROOT / "data" / "strengthener"
                      / "tier_d_review_queue.jsonl")


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# Investment roles that go to Tier D instead of Tier C
TIER_D_INVESTMENT_ROLES = frozenset({
    "insurance", "diversifier", "hedge",
})


def should_route_to_tier_d(spec) -> bool:
    """True iff this spec's investment_role belongs to Tier D.

    Per spec §15.A3: alpha + overlay → Tier C; everything else →
    Tier D. NULL investment_role defaults to alpha → Tier C
    (legacy compat — existing dispatch unchanged).
    """
    role = getattr(spec, "investment_role", None)
    if role is None:
        # Legacy spec / extractor didn't fill → default Tier C
        return False
    return role in TIER_D_INVESTMENT_ROLES


# ────────────────────────────────────────────────────────────────────
# Minimal diagnostic metrics (Phase 1 placeholder)
# ────────────────────────────────────────────────────────────────────
def _compute_role_minimal_diagnostics(
    pnl_series: pd.Series,
) -> dict:
    """Compute the small set of role-agnostic diagnostics we can
    confidently report without Phase 3 methodology. These don't
    pretend to be a verdict — they're observability."""
    s = pnl_series.dropna()
    if len(s) < 12:
        return {"insufficient_history": True, "n_months": len(s)}
    cum = (1 + s).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = float(drawdown.min())
    underwater_count = int((drawdown < 0).sum())
    # Per-month return distribution
    n_neg = int((s < 0).sum())
    n_pos = int((s > 0).sum())
    return {
        "n_months":               len(s),
        "ann_return_pct":         float(s.mean() * 12 * 100),
        "ann_vol_pct":            float(s.std() * math.sqrt(12) * 100)
                                     if s.std() > 0 else 0.0,
        "max_drawdown_pct":       max_dd * 100,
        "n_underwater_months":    underwater_count,
        "n_positive_months":      n_pos,
        "n_negative_months":      n_neg,
        "hit_rate_pct":           (n_pos / len(s)) * 100,
        "phase_3_pending":        True,
        "phase_3_pending_reason":
            "role-specific rigor (insurance convexity / "
            "diversifier conditional correlation) requires "
            "methodology research per spec §9.1/§9.2 backlog. "
            "Current diagnostics are observability only — NOT "
            "a verdict.",
    }


# ────────────────────────────────────────────────────────────────────
# Tier D dispatch entry point
# ────────────────────────────────────────────────────────────────────
def dispatch_tier_d(
    spec,
    family_hint:        str,
    template_result,
    dispatch_event_id:  Optional[str] = None,
) -> dict:
    """Tier D dispatch: produce diagnostic metrics + queue human
    review. NO verdict, NO confidence score.

    Returns a dict shaped to slot into the dispatcher's `out` payload:
      {
        tier:                  "D",
        investment_role:       <spec.investment_role>,
        diagnostic_metrics:    {...},
        human_review_required: True,
        human_review_event_id: <uuid>,
        phase_3_pending:       True,
      }

    Side effect: writes a row to data/strengthener/tier_d_review_queue.jsonl
    + the dispatcher's caller is responsible for surfacing the
    queue entry to /approvals UI (Commit 5 / 6 will wire this).
    """
    import json
    import uuid

    artifacts = template_result.artifacts or {}
    pnl_df = artifacts.get("pnl_series_df")
    diagnostics: dict
    if pnl_df is not None and "pnl_net_13bp" in pnl_df.columns:
        diagnostics = _compute_role_minimal_diagnostics(
            pnl_df["pnl_net_13bp"]
        )
    else:
        diagnostics = {"pnl_series_missing": True}

    review_event_id = str(uuid.uuid4())
    row = {
        "review_event_id":     review_event_id,
        "ts":                  _utc_iso(),
        "tier":                "D",
        "hypothesis_id":       spec.hypothesis_id,
        "investment_role":     getattr(spec, "investment_role", None),
        "statistical_role":    getattr(spec, "statistical_role", None),
        "asset_class":         getattr(spec, "asset_class", None),
        "family_hint":         family_hint,
        "dispatch_event_id":   dispatch_event_id,
        "template_verdict":    template_result.verdict,
        "template_summary":    template_result.summary,
        "diagnostic_metrics":  diagnostics,
        "phase_3_pending":     True,
        "review_status":       "PENDING_HUMAN_REVIEW",
    }
    try:
        TIER_D_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TIER_D_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.exception("tier_d: log append failed for %s",
                            spec.hypothesis_id)

    return {
        "tier":                  "D",
        "investment_role":       row["investment_role"],
        "diagnostic_metrics":    diagnostics,
        "human_review_required": True,
        "human_review_event_id": review_event_id,
        "phase_3_pending":       True,
    }
