"""engine/research/red_attribution.py — RED verdict failure attribution.

Part of the measurement substrate ([[project-brainstorm-architecture-
2026-06-14]] audit). When a verdict comes back RED, the principal
records WHY in a structured category. Without attribution, every RED
is just "RED" — with attribution, we can:

  - Identify which gate failed to catch what (e.g. "α/γ missed
    spanning 6 times → improve R12 lesson injection")
  - Spot recurring failure modes per pack (e.g. "physics_analogies
    ideas always die on data_regime → tighten pack guidance")
  - Build the measurement loop that feeds calibration tracker
    (deferred, 6+ months out)

Tetlock 2015 (Superforecasting) calls this the FORECASTING JOURNAL
discipline — predicting then reflecting on WHY you were wrong is
the single highest-impact training intervention.

Storage: data/research/red_attributions.jsonl (append-only)
Schema:
  attribution_id, verdict_event_id, hypothesis_id, red_category,
  rationale, attributed_by, attributed_ts, schema_version=1
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
RED_ATTRIBUTIONS_PATH = _REPO_ROOT / "data" / "research" / "red_attributions.jsonl"

SCHEMA_VERSION = 1

# Controlled enum — extend ONLY when a category recurs without fit.
# Adding categories is fine; deleting them breaks historical analysis.
RED_CATEGORIES = {
    "data_regime_mismatch",   # data window doesn't cover the regime where idea matters
    "mechanism_wrong",        # economic intuition is just wrong
    "already_spanned",        # FF5+MOM explains it (R12 lesson should've caught)
    "cost_killed",            # alpha disappears under realistic costs (cost-aware fields too optimistic)
    "sub_period_unstable",    # GREEN in some sub-period, RED in another
    "novelty_overclaim",      # γ catalog already had it (γ missed the match)
    "power_too_low",          # n insufficient to detect realistic effect
    "implementation_bug",     # our code, not the idea
    "graveyard_dup",          # close to existing RED autopsy that graveyard collision didn't catch
    "data_unavailable",       # required data doesn't exist / is paywalled
    "other",                  # rationale should be specific
}


@_dc.dataclass(frozen=True)
class RedAttribution:
    attribution_id:    str
    verdict_event_id:  str
    hypothesis_id:     Optional[str]
    red_category:      str
    rationale:         str
    attributed_by:     str
    attributed_ts:     str
    schema_version:    int


class AttributionError(ValueError):
    pass


def attribute(
    verdict_event_id: str,
    red_category:     str,
    rationale:        str,
    *,
    hypothesis_id:    Optional[str] = None,
    attributed_by:    str = "principal",
) -> RedAttribution:
    """Record a RED attribution. Mandatory non-empty rationale +
    category from controlled enum. Raises AttributionError otherwise."""
    if red_category not in RED_CATEGORIES:
        raise AttributionError(
            f"red_category must be one of {sorted(RED_CATEGORIES)}, "
            f"got {red_category!r}")
    rationale = (rationale or "").strip()
    if len(rationale) < 10:
        raise AttributionError(
            "rationale required (min 10 chars) — RED attribution exists "
            "to learn from failures, vague rationales defeat the point")
    row = RedAttribution(
        attribution_id   = str(uuid.uuid4()),
        verdict_event_id = verdict_event_id,
        hypothesis_id    = hypothesis_id,
        red_category     = red_category,
        rationale        = rationale,
        attributed_by    = attributed_by,
        attributed_ts    = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        schema_version   = SCHEMA_VERSION,
    )
    RED_ATTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RED_ATTRIBUTIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_dc.asdict(row)) + "\n")
    return row


def for_verdict(verdict_event_id: str) -> list[dict]:
    """All attributions for one verdict, newest first."""
    if not RED_ATTRIBUTIONS_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in RED_ATTRIBUTIONS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("verdict_event_id") == verdict_event_id:
            out.append(r)
    out.sort(key=lambda r: r.get("attributed_ts", ""), reverse=True)
    return out


def list_all(limit: int = 500) -> list[dict]:
    if not RED_ATTRIBUTIONS_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in RED_ATTRIBUTIONS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    out.sort(key=lambda r: r.get("attributed_ts", ""), reverse=True)
    return out[:limit]


def category_counts(since_iso: Optional[str] = None) -> dict[str, int]:
    """Histogram of categories. For metrics dashboard."""
    out: dict[str, int] = {c: 0 for c in RED_CATEGORIES}
    for r in list_all(limit=10000):
        if since_iso and (r.get("attributed_ts") or "") < since_iso:
            continue
        cat = r.get("red_category")
        if cat in out:
            out[cat] += 1
    return out
