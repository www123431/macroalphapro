"""engine/research/suggestion_engine.py — Phase 4d.5: L1 candidate
seed recommender.

Static / heuristic — NO LLM call. Reads library + graveyard +
outcome_ledger + deployed role distribution, ranks candidates by:

  underexplored × no_cousin × role_gap × novelty_anchor

The output is a ranked list of seed ideas with rationale and risk
tags, fed to the Cockpit "Suggestions" panel so senior doesn't face
a blank textarea every time.

WHY heuristic-not-LLM (per senior 2026-06-01 discussion):
  - L1 is gap-finder, not idea-generator
  - LLM ranking introduces self-bias loop (model prefers what it has
    seen) without ANY calibration data to push back on it
  - L2 (Scout agent) becomes safe to add only after L4 outer loop has
    ~20 iterations of verdict_alignment data to validate Council
    itself works

Two sources blended:
  A) LIBRARY-derived: any entry with status UNTESTED / LIVE_UNTESTED
     / PENDING_DEPLOY (deployment-ready but not yet validated)
  B) SEED POOL: hardcoded list of senior-quant-known opportunities
     NOT yet in library — keeps the surface fresh + reflects tacit
     senior knowledge that isn't captured in JSONLs

Each scored on 0-1 dimensions; final score is weighted sum.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Senior knowledge — hardcoded seed pool (B source) ─────────────────
# Each entry: a candidate worth investigating that is NOT in the
# library yet. Surfaces senior tacit knowledge in a queryable form.
# Update this list as new high-leverage mechanisms are identified
# (research discoveries, paper reads, etc.) — the rest of the engine
# auto-picks up the additions.

SEED_POOL: list[dict] = [
    {
        "title": "Equity short-interest deciles (DLW 2009 mechanism)",
        "family": "short_interest_alpha",
        "parent_family": "equity_factor",
        "proposed_role": "alpha_seeker",
        "seed": (
            "Equity short-interest deciles as alpha signal — "
            "Diether-Lee-Werner 2009 mechanism in Russell 2000 "
            "microcap universe with weekly rebalance."
        ),
        "rationale": (
            "Short-sale positioning carries informed-trader signal; "
            "microcap focus aligns with limits-to-arbitrage where "
            "effect is strongest. Diversifies away from earnings-event "
            "alpha (D_PEAD family) by construction."
        ),
        "anchor_paper": "diether_lee_werner_2009_rfs",
    },
    {
        "title": "Bond TSMOM via 7-country panel (MOP 2012)",
        "family": "bond_tsmom",
        "parent_family": "macro_factor",
        "proposed_role": "risk_premium_harvester",
        "seed": (
            "Time-series momentum on G7 government bond futures, "
            "Moskowitz-Ooi-Pedersen 2012 12-1 mechanism, monthly "
            "rebalance with vol targeting."
        ),
        "rationale": (
            "TSMOM in bonds is the lowest-correlation classic factor "
            "to current book (no bond TSMOM deployed). MOP showed the "
            "effect is strong cross-asset, not just equity."
        ),
        "anchor_paper": "moskowitz_ooi_pedersen_2012_jfe",
    },
    {
        "title": "Equity quality-junk (QMJ, AFP 2019)",
        "family": "quality_junk",
        "parent_family": "equity_factor",
        "proposed_role": "diversifier",
        "seed": (
            "Quality-minus-junk anomaly via Asness-Frazzini-Pedersen "
            "2019 QMJ on Russell 1000 universe with monthly rebalance."
        ),
        "rationale": (
            "Quality is the canonical low-vol-period diversifier; book "
            "currently has zero pure-quality exposure. AFP 2019 QMJ "
            "shows international robustness."
        ),
        "anchor_paper": "asness_frazzini_pedersen_2019_jfqa",
    },
    {
        "title": "Commodity term structure (Schwartz 1997)",
        "family": "commodity_basis",
        "parent_family": "commodity_factor",
        "proposed_role": "risk_premium_harvester",
        "seed": (
            "Long-backwardation / short-contango commodity carry via "
            "Schwartz 1997 term-structure mechanism — sorting "
            "commodity futures by (front - second-month) basis."
        ),
        "rationale": (
            "Same KMPV carry framework already deployed for cross-"
            "asset, applied to within-commodity term structure. Should "
            "be orthogonal to existing cross-asset carry sleeve."
        ),
        "anchor_paper": "schwartz_1997_jfin",
    },
    {
        "title": "Cross-asset value (HML across asset classes)",
        "family": "cross_asset_value",
        "parent_family": "cross_asset_factor",
        "proposed_role": "diversifier",
        "seed": (
            "HML-style value applied across equity indices, bond "
            "yields, currency PPP, and commodity spot-vs-trend; equal "
            "risk-weight blend with monthly rebalance."
        ),
        "rationale": (
            "Cross-asset value is a canonical AQR / Asness 2013 finding "
            "with low correlation to single-asset-class value. Fills "
            "the value-style gap in our book."
        ),
        "anchor_paper": "asness_moskowitz_pedersen_2013_jf",
    },
    {
        "title": "Equity BAB (low-vol leverage)",
        "family": "betting_against_beta",
        "parent_family": "equity_factor",
        "proposed_role": "risk_premium_harvester",
        "seed": (
            "Betting-against-beta strategy — long low-β / short high-β "
            "equity, leveraged to market-neutral. Frazzini-Pedersen "
            "2014 mechanism on Russell 1000 universe."
        ),
        "rationale": (
            "BAB is among the most-replicated cross-section anomalies; "
            "book has only 1 equity sleeve so adding BAB increases "
            "within-equity diversification."
        ),
        "anchor_paper": "frazzini_pedersen_2014_jfe",
    },
    {
        "title": "Earnings-revisions extension to JP",
        "family": "earnings_revision",
        "parent_family": "equity_factor",
        "proposed_role": "alpha_seeker",
        "seed": (
            "Apply Ang-Hodrick-Xing-Zhang 2006 earnings revision "
            "mechanism to TOPIX with same FF12-sector-neutral "
            "construction as US D_PEAD pilot."
        ),
        "rationale": (
            "Earnings-revision is sibling to PEAD; combining the two "
            "in JP would form a complete forward-earnings information "
            "stack. WARN: graveyard has JP PEAD as cross-market RED."
        ),
        "anchor_paper": "ang_hodrick_xing_zhang_2006_jf",
    },
]


@dataclass
class Suggestion:
    """One ranked candidate seed."""
    title: str
    family: str
    seed: str
    rationale: str
    proposed_role: str
    parent_family: str
    risk_tag: str  # "low" | "medium" | "high"
    source: str    # "library" | "seed_pool"
    anchor_paper: Optional[str] = None
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Scoring helpers ───────────────────────────────────────────────────


def _role_gap_bonus(role: Optional[str], deployed_roles: dict[str, int]) -> float:
    """Roles that are scarce in deployed book get a bonus."""
    if not role:
        return 0.0
    total = sum(deployed_roles.values()) or 1
    share = deployed_roles.get(role, 0) / total
    # Below-30% share is a gap worth filling
    if share < 0.10:
        return 0.30
    if share < 0.25:
        return 0.20
    if share < 0.35:
        return 0.10
    return 0.0


def _graveyard_collision(
    family: Optional[str],
    graveyard_families: dict[str, int],
) -> tuple[float, str]:
    """Returns (penalty, risk_tag). Higher penalty = bigger collision.

    Uses graveyard._normalize_family() to alias-match (so
    'earnings_revision' collides with 'forward-earnings information'
    via the canonical alias group — same fix that fed the 12th catch
    upstream)."""
    if not family:
        return 0.0, "low"
    from engine.research.graveyard import _normalize_family
    target = _normalize_family(family)
    # Build alias-normalized view of graveyard counts (sum across aliases)
    cousins = 0
    for fam_name, count in graveyard_families.items():
        if _normalize_family(fam_name) == target:
            cousins += count
    if cousins >= 4:
        return 0.40, "high"
    if cousins >= 1:
        return 0.15, "medium"
    return 0.0, "low"


def _untested_bonus(status: Optional[str]) -> float:
    if status == "UNTESTED" or status == "LIVE_UNTESTED":
        return 0.35
    if status == "PENDING_DEPLOY":
        return 0.20
    return 0.0


# ── Public API ────────────────────────────────────────────────────────


def get_candidate_suggestions(limit: int = 10) -> dict:
    """Build a ranked list of candidate seeds.

    Returns:
      {
        "n_total": int,
        "by_source": {library: int, seed_pool: int},
        "suggestions": [Suggestion.to_dict(), ...]
      }
    Sorted by score descending.
    """
    # Lazy imports — these read disk + may be expensive
    from engine.research.graveyard import build_graveyard
    from engine.research.llm_tools import query_library

    # ── Gather signal sources ──
    library_entries = (query_library() or {}).get("entries", []) or []

    graveyard = build_graveyard()
    grave_fam_counts: dict[str, int] = {}
    for e in graveyard:
        if e.family:
            grave_fam_counts[e.family] = grave_fam_counts.get(e.family, 0) + 1

    # Count deployed roles for the role-gap bonus
    deployed_roles: dict[str, int] = {}
    for e in library_entries:
        if e.get("status") == "DEPLOYED" and e.get("proposed_role"):
            role = e["proposed_role"]
            deployed_roles[role] = deployed_roles.get(role, 0) + 1

    suggestions: list[Suggestion] = []

    # ── (A) Library-derived suggestions ──
    for e in library_entries:
        status = e.get("status")
        if status not in ("UNTESTED", "LIVE_UNTESTED", "PENDING_DEPLOY"):
            continue
        family = e.get("family")
        role = e.get("proposed_role")
        coll_penalty, risk_tag = _graveyard_collision(family, grave_fam_counts)
        untested = _untested_bonus(status)
        role_bonus = _role_gap_bonus(role, deployed_roles)
        score = max(0.0, min(1.0, untested + role_bonus - coll_penalty))
        rationale_bits = []
        if status:
            rationale_bits.append(f"library status={status}")
        if family:
            cc = grave_fam_counts.get(family, 0)
            if cc > 0:
                rationale_bits.append(f"family has {cc} graveyard cousin(s)")
            else:
                rationale_bits.append("no graveyard cousins in family")
        if role and role_bonus > 0:
            rationale_bits.append(f"role gap (book under-allocated to {role})")
        seed = (
            f"Test the {family or 'unknown-family'} mechanism "
            f"(library id={e.get('id')}) — currently {status}. "
            f"{e.get('purpose') or ''}"
        ).strip()
        suggestions.append(Suggestion(
            title=str(e.get("id") or "library_entry"),
            family=family or "unknown",
            seed=seed,
            rationale=" · ".join(rationale_bits),
            proposed_role=role or "alpha_seeker",
            parent_family="library_known",
            risk_tag=risk_tag,
            source="library",
            score=round(score, 3),
            score_components={
                "untested":   round(untested, 3),
                "role_gap":   round(role_bonus, 3),
                "collision_penalty": round(coll_penalty, 3),
            },
        ))

    # ── (B) Seed pool suggestions ──
    for entry in SEED_POOL:
        family = entry.get("family")
        role = entry.get("proposed_role")
        coll_penalty, risk_tag = _graveyard_collision(family, grave_fam_counts)
        role_bonus = _role_gap_bonus(role, deployed_roles)
        novelty_anchor = 0.25  # baseline for seed-pool (curated by senior)
        score = max(0.0, min(1.0, novelty_anchor + role_bonus - coll_penalty))
        bits = []
        cc = grave_fam_counts.get(family or "", 0)
        if cc > 0:
            bits.append(f"family has {cc} graveyard cousin(s)")
        else:
            bits.append("no graveyard cousins in family")
        if role_bonus > 0:
            bits.append(f"role gap (book under-allocated to {role})")
        bits.append(entry.get("rationale", ""))
        suggestions.append(Suggestion(
            title=entry["title"],
            family=family or "unknown",
            seed=entry["seed"],
            rationale=" · ".join([b for b in bits if b]),
            proposed_role=role or "alpha_seeker",
            parent_family=entry.get("parent_family") or "external",
            risk_tag=risk_tag,
            source="seed_pool",
            anchor_paper=entry.get("anchor_paper"),
            score=round(score, 3),
            score_components={
                "novelty_anchor":    round(novelty_anchor, 3),
                "role_gap":          round(role_bonus, 3),
                "collision_penalty": round(coll_penalty, 3),
            },
        ))

    # Sort by score desc, then by source (library first for tie-break)
    suggestions.sort(
        key=lambda s: (-s.score, 0 if s.source == "library" else 1),
    )

    out = suggestions[: max(1, limit)]
    return {
        "n_total":   len(out),
        "by_source": {
            "library":   sum(1 for s in out if s.source == "library"),
            "seed_pool": sum(1 for s in out if s.source == "seed_pool"),
        },
        "suggestions": [s.to_dict() for s in out],
    }
