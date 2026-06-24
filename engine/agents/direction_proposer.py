"""engine.agents.direction_proposer — paper-corpus direction ranker.

Third autonomous agent. Unlike audit_verifier (gate verifier) and
graveyard_collision (collision check), this one is PROACTIVE: it
mines the paper corpus to tell the user what's worth testing NEXT.

Addresses the user's stated gap: "项目还能根据积攒的经验或者论文什么的
指引未来因子研究的方向". The on-page Forward queue shows hypotheses
ordered by static priority (HIGH / MEDIUM / LOW); this agent layers a
multi-dimensional ROI score on top:

  Sₚ  paper-priority (HIGH=3, MEDIUM=2, LOW=1)         + recency boost
  Sₐ  data availability (have=3, partial=2, missing=1, unknown=0)
  Sₒ  orthogonality to deployed book
      (1.0 if family not in {deployed families}; else 0.4)
  Sg  graveyard penalty
      (1.0 baseline; CLEAN keeps it; WARN -> 0.7; RISK -> 0.3)
  Sf  family-saturation discount
      (penalize families that already have N approved-but-untested
       vectors stacked up — diminishing marginal value)

Total score = Sₚ * Sₐ * Sₒ * Sg * Sf  (multiplicative — any 0 kills)

Top-N output, each direction carrying:
  - source paper_id + title + DOI
  - source_hypothesis_id (the chunk_id-traced claim)
  - family + subtype + predicted_direction
  - 5 component scores + total
  - graveyard verdict (CLEAN / WARN / RISK) + worst match if any
  - one-line rationale explaining the rank

Pure deterministic v1, no LLM. The structured output is the contract;
an LLM narrator can write 1-2 sentence prose summaries later.
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIBRARY_DIR = _REPO_ROOT / "data" / "research" / "mechanism_library"

# Family-saturation: penalize the Nth approved-untested vector in the
# same family by this factor. Discourages all chips on one family.
_SATURATION_FACTOR = 0.8


# ── Deployed book inventory ────────────────────────────────────────


def _deployed_families() -> set[str]:
    """Scan library YAML for status_in_our_book: DEPLOYED.
    Returns lowercase family names."""
    out: set[str] = set()
    if not _LIBRARY_DIR.exists():
        return out
    try:
        import yaml
    except ImportError:
        return out
    for p in _LIBRARY_DIR.glob("*.yaml"):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if str(data.get("status_in_our_book", "")).upper() != "DEPLOYED":
            continue
        fam = (data.get("family") or data.get("parent_family") or "").lower()
        if fam:
            out.add(fam)
    return out


# ── Data status ───────────────────────────────────────────────────


def _data_status_for(required_data: list[str]) -> str:
    """Mirrors the data_status logic used by the forward-vectors API
    so directions and the Forward page agree on classification.
    _data_coverage_for returns (status, have, missing) — only the
    first element is the label."""
    try:
        from api.routes_paper_chain import (
            _build_data_inventory, _data_coverage_for,
        )
        _build_data_inventory()
        result = _data_coverage_for(list(required_data or []))
        if isinstance(result, tuple) and result:
            return str(result[0])
        return str(result)
    except Exception:
        return "unknown"


# ── Component scores ──────────────────────────────────────────────


def _score_priority(priority: str) -> float:
    return {"high": 3.0, "medium": 2.0, "low": 1.0}.get(
        str(priority).lower(), 1.0
    )


def _score_data(status: str) -> float:
    """T1.5 (2026-06-05 audit D1 fix): unknown was 1.5 in code while the
    module docstring (line 13) declared 0. A multiplicative 0 kills the
    total score, which is the correct conservatism — without verified
    data presence, the hypothesis is not testable today and shouldn't
    show up in DIRECTIONS ranked above hypotheses with KNOWN data.
    Pre-fix, unknown=1.5 silently boosted unverified candidates above
    missing-data candidates, the opposite of what the design specified.
    """
    return {"have": 3.0, "partial": 2.0, "missing": 1.0,
            "unknown": 0.0}.get(str(status).lower(), 0.0)


def _score_orthogonality(family: str, deployed: set[str]) -> float:
    if not family:
        return 0.5
    return 1.0 if family.lower() not in deployed else 0.4


def _score_graveyard(verdict: str) -> float:
    return {"CLEAN": 1.0, "WARN": 0.7, "RISK": 0.3}.get(
        str(verdict).upper(), 1.0
    )


def _score_saturation(family: str, family_count: dict[str, int]) -> float:
    n = family_count.get(family.lower(), 0)
    # 1.0 for first, 0.8 for second, 0.64 for third, etc.
    return _SATURATION_FACTOR ** max(0, n)


# T3.5 (2026-06-05 audit D2 + D3 fix): additive composition with floors.
# Pre-T3.5 the total was sp * sa * so * sg * sf (multiplicative). Two bugs:
#   D2 — Sp (priority) and Sf (saturation) both reflect family hotness, so
#        multiplying them double-counts the family signal. A high-priority
#        idea in a saturated family was penalized twice.
#   D3 — multiplicative 0-kill, no floor. A graveyard WARN (sg=0.7) +
#        deployed family (so=0.4) + 4th-in-family (sf=0.51) collapsed the
#        total below 0.1 even for high-priority + have-data candidates,
#        wiping ideas that should have been rank-2 or rank-3.
#
# Additive composition: each dimension is normalized to roughly [0, 1],
# floored at _SCORE_FLOOR so no single low dimension can wipe the total,
# and Sf carries a smaller weight (0.5) since Sp already encodes some
# family/priority signal. Total = weighted mean, range roughly [0, 1].
_SCORE_FLOOR = 0.10   # below this, the dimension is "low but not killing"
_SF_WEIGHT   = 0.50   # saturation half-weighted vs other dims (D2 fix)


def _norm_floor(x: float, denom: float = 1.0) -> float:
    """Normalize x/denom to [0,1] and floor at _SCORE_FLOOR."""
    return max(_SCORE_FLOOR, min(1.0, float(x) / float(denom) if denom else 0.0))


def _compose_total_additive(sp: float, sa: float, so: float,
                            sg: float, sf: float) -> float:
    """T3.5 additive composition. Returns total in roughly [0.1, 1.0].

    Per-dimension normalizations (raw_max known from _score_* funcs):
      sp_n = sp / 3.0        (priority 1..3 -> 0.33..1.0)
      sa_n = sa / 3.0        (data 0..3 -> 0.0..1.0)
      so_n = so              (orthogonality already 0.4..1.0)
      sg_n = sg              (graveyard already 0.3..1.0)
      sf_n = sf              (saturation 0.0..1.0)

    Weighted mean: total = (sp_n + sa_n + so_n + sg_n + 0.5*sf_n) / 4.5.

    Data-availability hard-kill exception
    -------------------------------------
    sa == 0.0 (UNKNOWN data per T1.5) short-circuits to total=0.
    T1.5 made unverified-data candidates sink via multiplicative
    0-kill; T3.5 keeps that one specific kill alive because data
    availability is the ONE hard prerequisite — without verified
    data the candidate isn't testable today, regardless of how
    high priority or orthogonal it is. All other dimensions use
    the additive (no-kill) path so a single soft factor (graveyard
    WARN, saturation 4th in family) can't wipe a good idea.
    """
    if sa <= 0.0:
        return 0.0
    sp_n = _norm_floor(sp, 3.0)
    sa_n = _norm_floor(sa, 3.0)
    so_n = _norm_floor(so, 1.0)
    sg_n = _norm_floor(sg, 1.0)
    sf_n = _norm_floor(sf, 1.0)
    weighted_sum = sp_n + sa_n + so_n + sg_n + _SF_WEIGHT * sf_n
    weight_total = 4.0 + _SF_WEIGHT
    return weighted_sum / weight_total


# ── Build a direction ─────────────────────────────────────────────


def _build_direction(
    fv,
    pm_status_overlay: Optional[str],
    deployed: set[str],
    family_count: dict[str, int],
) -> dict:
    from engine.agents.graveyard_collision import check_collision

    family            = fv.mechanism_family.value if hasattr(fv.mechanism_family, "value") else str(fv.mechanism_family)
    mechanism_subtype = fv.mechanism_subtype
    priority          = fv.priority.value if hasattr(fv.priority, "value") else str(fv.priority)

    data_status = _data_status_for(list(fv.required_data or []))

    gc_result = check_collision(
        candidate_name    = fv.source_hypothesis_id,
        family            = family,
        mechanism_subtype = mechanism_subtype,
        claim_text        = fv.claim,
    )
    gc_verdict = gc_result.get("verdict", "CLEAN")

    sp = _score_priority(priority)
    sa = _score_data(data_status)
    so = _score_orthogonality(family, deployed)
    sg = _score_graveyard(gc_verdict)
    sf = _score_saturation(family, family_count)

    # T3.5: additive composition replaces multiplicative product.
    total = _compose_total_additive(sp, sa, so, sg, sf)

    # Rationale: surface the strongest signal driving the score
    bits = []
    if priority.lower() == "high":
        bits.append("high-priority paper")
    if data_status == "have":
        bits.append("data on disk")
    if family.lower() not in deployed:
        bits.append(f"family '{family}' not yet deployed — orthogonal")
    else:
        bits.append(f"family '{family}' already in book — diversification limited")
    if gc_verdict in ("WARN", "RISK"):
        bits.append(f"graveyard {gc_verdict} — re-read past RED first")
    if family_count.get(family.lower(), 0) >= 3:
        bits.append("family saturation — many similar candidates queued")
    rationale = "; ".join(bits)

    return {
        "rank":                None,    # filled after sort
        "source_paper_id":     fv.source_paper_id,
        "paper_title":         fv.paper_title,
        "source_hypothesis_id": fv.source_hypothesis_id,
        "claim":               fv.claim,
        "family":              family,
        "mechanism_subtype":   mechanism_subtype,
        "predicted_direction": fv.predicted_direction,
        "data_status":         data_status,
        "priority":            priority,
        "pm_status":           pm_status_overlay or "extracted",
        "scores": {
            "priority":      round(sp, 3),
            "data":          round(sa, 3),
            "orthogonality": round(so, 3),
            "graveyard":     round(sg, 3),
            "saturation":    round(sf, 3),
            "total":         round(total, 3),
            "composition":   "additive_v2_T3.5",   # was multiplicative_v1
        },
        "graveyard_verdict":   gc_verdict,
        "graveyard_n_scanned": gc_result.get("n_scanned", 0),
        "rationale":           rationale,
    }


# ── Public API ────────────────────────────────────────────────────


def _claim_type_map() -> dict[str, str]:
    """B.2-A3 (2026-06-05): build {source_hypothesis_id -> claim_type}
    map from hypothesis_specs store. Used to filter out non-factor
    claims (METHODOLOGY / MICROSTRUCTURE / etc.) before ranking.

    Latest-wins by extracted_ts (B.2-A4 hotfix 2026-06-05): re-extract
    runs append a NEW spec_id with version=1 (not a bump_version), so
    the pre-fix "s.version > 1" guard never matched and v1 UNKNOWN
    specs from the original backfill kept winning over v2 typed specs.
    Use extraction.extracted_ts to pick the latest extraction per
    hypothesis_id, which correctly reflects "the most recent
    classification we have".
    """
    try:
        from engine.hypothesis_spec.store import all_specs
    except Exception:
        return {}
    out: dict[str, str] = {}
    latest_ts: dict[str, str] = {}
    try:
        for s in all_specs():
            hid = s.source_hypothesis_id
            ts = (s.extraction.extracted_ts or s.created_ts or "")
            if hid not in latest_ts or ts > latest_ts[hid]:
                latest_ts[hid] = ts
                out[hid] = s.claim_type.value
    except Exception:
        logger.exception("direction_proposer: claim_type_map build failed")
    return out


_RANKABLE_CLAIM_TYPES = {"FACTOR_HYPOTHESIS", "UNKNOWN"}


def propose_directions(
    top: int = 5,
    family: Optional[str] = None,
) -> dict:
    """Build the ranked direction list.

    Args:
        top    — number of directions to return (1..20)
        family — optional family filter (e.g. 'CARRY'); case-insensitive

    Returns:
        {
          "generated_ts": iso8601,
          "deployed_families": [...],
          "n_candidates_scanned": int,
          "n_non_factor_excluded": int,
          "directions": [direction, ...],
        }

    The directions are sorted by total score descending. The same
    forward_vector can NOT appear twice in a single call (idempotent).

    B.2-A3: hypothesis_specs claim_type != FACTOR_HYPOTHESIS (and !=
    UNKNOWN for v1 backward-compat) are excluded — they're research
    evidence (methodology / microstructure / capacity / decay / etc.)
    but not testable strategy candidates. n_non_factor_excluded reports
    how many were filtered for transparency.
    """
    top = max(1, min(20, top))

    try:
        from engine.research_store.forward_vectors import generate_forward_vectors
        from engine.research_store.forward_vectors.review import load_latest_reviews
        vecs = generate_forward_vectors()
    except Exception as exc:
        logger.exception("direction_proposer: forward_vectors load failed")
        return {
            "generated_ts": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "error": f"forward_vectors_load:{exc}",
            "directions": [],
        }

    reviews = load_latest_reviews()
    ct_map = _claim_type_map()

    # Filter: drop rejected; keep extracted + reviewed + approved
    keep = []
    n_non_factor_excluded = 0
    for v in vecs:
        # B.2-A3 claim_type gate
        ct = ct_map.get(v.source_hypothesis_id, "UNKNOWN")
        if ct not in _RANKABLE_CLAIM_TYPES:
            n_non_factor_excluded += 1
            continue

        r = reviews.get(v.source_hypothesis_id)
        st = (r.status.value if r else "extracted").lower()
        if st == "rejected":
            continue
        if family:
            fv_fam = v.mechanism_family.value if hasattr(v.mechanism_family, "value") else str(v.mechanism_family)
            if fv_fam.lower() != family.lower():
                continue
        keep.append((v, st))

    # Family counts for saturation score
    family_count: dict[str, int] = {}
    for v, _st in keep:
        fv_fam = v.mechanism_family.value if hasattr(v.mechanism_family, "value") else str(v.mechanism_family)
        family_count[fv_fam.lower()] = family_count.get(fv_fam.lower(), 0) + 1

    deployed = _deployed_families()

    directions = []
    for v, st in keep:
        try:
            d = _build_direction(v, st, deployed, family_count)
            directions.append(d)
        except Exception:
            logger.exception("direction_proposer: failed to score %s", v.source_hypothesis_id)
            continue

    directions.sort(key=lambda d: d["scores"]["total"], reverse=True)
    for i, d in enumerate(directions[:top], start=1):
        d["rank"] = i

    return {
        "generated_ts":         _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "deployed_families":    sorted(deployed),
        "n_candidates_scanned": len(keep),
        "n_non_factor_excluded": n_non_factor_excluded,
        "directions":           directions[:top],
    }
