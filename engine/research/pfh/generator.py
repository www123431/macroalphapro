"""engine/research/pfh/generator.py — Candidate enumeration.

Generates CandidateProposal objects from:
  1. EXTENSIONS: existing GREEN family + a new sub-variation
                 (e.g. carry GREEN → curve_slope_carry candidate)
  2. CROSS-MARKET: existing GREEN family + a different market
                 (e.g. D_PEAD US → D_PEAD JP — note JP is in graveyard so
                  this candidate gets penalized; user sees the warning)
  3. UNTESTED FAMILIES: families that appear in literature but haven't
                 been GREEN-tested, EXCLUDING those in graveyard

DESIGN PRINCIPLE: generator does NOT score — it only ENUMERATES. Scoring
happens downstream in proposer.py using engine/research/pfh/bayesian.py.
This separation lets the scorer be swapped without touching the
generator (and the generator be exhaustive without overwhelming the
scoring layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.research.pfh.catalog import (
    LabeledMechanism, per_family_counts,
)


@dataclass
class CandidateProposal:
    """A proposed factor to score + potentially turn into a compose-spec."""
    # Identity
    candidate_id:      str
    proposal_kind:     str        # "extension" / "cross_market" / "untested_family"
    family_normalized: str

    # The 4 axes (may be partial — generator only fills what it knows)
    universe:          Optional[str] = None  # name OR "<NEW: description>"
    signal_recipe:     Optional[str] = None
    weighting:         Optional[str] = None
    rebalance:         str = "monthly"

    # Evidence trail
    derived_from:      list[str] = field(default_factory=list)   # mechanism names
    cousin_warnings:   list[str] = field(default_factory=list)
    needs_new_axes:    list[str] = field(default_factory=list)   # axis YAMLs missing
    rationale_seeds:   list[str] = field(default_factory=list)   # short hints, NOT prose

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


# ── Generator rules ──────────────────────────────────────────────────


# A small catalog of "well-known but not deployed by us" families
# the generator can propose as untested-family candidates. Each entry
# pre-annotates which axes would need to exist.
_UNTESTED_FAMILY_SEEDS: list[dict] = [
    {
        "family_normalized": "rates_carry_curve_slope",
        "rationale": "Litterman-Scheinkman 1991 curve-slope premium; "
                     "extends our deployed level-carry sleeve",
        "needs_new_axes": ["universe: futures_g10_rates_2y10y",
                            "signal_recipe: curve_slope_carry"],
        "anchor_family": "carry",   # cousin-link for scoring
    },
    {
        "family_normalized": "equity_quality_minus_junk",
        "rationale": "AQR Quality-Minus-Junk (Asness-Frazzini-Pedersen 2019); "
                     "well-replicated but currently NOT in our library",
        "needs_new_axes": ["universe: equity_top1500_with_fundamentals",
                            "signal_recipe: composite_quality_z"],
        "anchor_family": "quality",
    },
    {
        "family_normalized": "fx_carry_dollar_factor",
        "rationale": "Lustig-Roussanov-Verdelhan 2011 dollar/carry factor; "
                     "currency-carry covariance with G10 dollar basket",
        "needs_new_axes": ["universe: fx_g10_baskets",
                            "signal_recipe: dollar_factor_decomposition"],
        "anchor_family": "carry",
    },
    {
        "family_normalized": "credit_carry_spread_ranking",
        "rationale": "Investment-grade vs high-yield carry — Asness-Pedersen "
                     "Carry Everywhere extension into credit",
        "needs_new_axes": ["universe: corp_bonds_ig_hy_etfs",
                            "signal_recipe: carry_yield_spread"],
        "anchor_family": "carry",
    },
    {
        "family_normalized": "commodities_term_structure_slope",
        "rationale": "Gorton-Hayashi-Rouwenhorst 2013 commodities term-structure "
                     "slope — extension of front-month carry to full curve",
        "needs_new_axes": ["universe: futures_commodities_full_curve",
                            "signal_recipe: curve_slope_normalized"],
        "anchor_family": "carry",
    },
    {
        "family_normalized": "options_skew_premium",
        "rationale": "Driessen-Maenhout 2007 / Bollerslev-Todorov 2011 "
                     "implied-skew premium — variance carry orthogonal to "
                     "vol_carry which we already tested RED",
        "needs_new_axes": ["universe: optionm_spx_skew_surface",
                            "signal_recipe: skew_level_minus_realized"],
        "anchor_family": "vol_carry",   # cousin warning will trigger
    },
]


def _generate_extensions(labels: list[LabeledMechanism]) -> list[CandidateProposal]:
    """For each GREEN family, propose 0-1 extension candidates citing
    the existing GREEN as anchor. Only fires for families with at least
    one GREEN."""
    out: list[CandidateProposal] = []
    by_fam = per_family_counts(labels)
    for fam_norm, info in by_fam.items():
        if info["n_green"] == 0:
            continue
        green_members = [m.name for m in labels
                          if m.family_normalized == fam_norm
                          and m.verdict == "GREEN"]
        out.append(CandidateProposal(
            candidate_id=f"pfh_ext_{fam_norm}",
            proposal_kind="extension",
            family_normalized=fam_norm,
            derived_from=green_members,
            needs_new_axes=[
                f"signal_recipe: {fam_norm}_extension_v1 (needs spec)"
            ],
            rationale_seeds=[
                f"existing GREEN: {', '.join(green_members[:3])}",
                f"extension targets unexplored sub-mechanism within family",
            ],
        ))
    return out


def _generate_cross_market(labels: list[LabeledMechanism]) -> list[CandidateProposal]:
    """For each GREEN family, propose cross-market variants. Target market
    list is the UNION of: (a) markets observed anywhere in labels (so
    cross-market warnings can fire on graveyard cousins) and (b) a small
    floor list of well-known markets we might want to translate into.
    """
    out: list[CandidateProposal] = []
    by_fam = per_family_counts(labels)
    _floor_markets = ["us_equity", "jp_equity", "eu_equity", "em_equity",
                      "cn_equity", "futures", "fx", "rates", "credit"]
    observed = {m.market for m in labels if m.market}
    _alt_markets = sorted(set(_floor_markets) | observed)

    for fam_norm, info in by_fam.items():
        if info["n_green"] == 0:
            continue
        greens = [m for m in labels
                   if m.family_normalized == fam_norm and m.verdict == "GREEN"]
        existing_markets = {m.market for m in greens if m.market}
        # Targets = markets the family hasn't been tested in
        all_in_fam_markets = {
            m.market for m in labels
            if m.family_normalized == fam_norm and m.market
        }
        for target in _alt_markets:
            if target in existing_markets:
                continue   # already deployed in this market
            # Check if this exact cross-market is in graveyard
            reds_in_target = [m for m in labels
                               if m.family_normalized == fam_norm
                               and m.market == target and m.verdict == "RED"]
            cid = f"pfh_xmkt_{fam_norm}_to_{target}"
            warnings = []
            if reds_in_target:
                warnings.append(
                    f"GRAVEYARD WARNING: {len(reds_in_target)} RED "
                    f"entries in family {fam_norm} × market {target} "
                    f"({', '.join(m.name for m in reds_in_target[:2])})"
                )
            out.append(CandidateProposal(
                candidate_id=cid,
                proposal_kind="cross_market",
                family_normalized=fam_norm,
                derived_from=[m.name for m in greens],
                cousin_warnings=warnings,
                needs_new_axes=[
                    f"universe: {target}_universe (needs spec)"
                ],
                rationale_seeds=[
                    f"GREEN in original market(s): {existing_markets or '?'}",
                    f"target market: {target}",
                    f"unexplored: {target not in all_in_fam_markets}",
                ],
            ))
    return out


def _generate_untested_families(labels: list[LabeledMechanism]) -> list[CandidateProposal]:
    """Seed-pool driven: families from known literature that aren't
    GREEN in our library AND aren't RED in graveyard."""
    out: list[CandidateProposal] = []
    by_fam = per_family_counts(labels)
    for seed in _UNTESTED_FAMILY_SEEDS:
        f = seed["family_normalized"]
        info = by_fam.get(f, {"n_green": 0, "n_yellow": 0, "n_red": 0})
        if info["n_green"] > 0:
            continue  # already GREEN, skip
        # Warn if RED count > 0 in this family
        warnings = []
        if info["n_red"] > 0:
            warnings.append(
                f"GRAVEYARD WARNING: {info['n_red']} RED entries in "
                f"family {f}"
            )
        # Warn if anchor family (cousin) has RED entries
        anchor = seed.get("anchor_family")
        if anchor:
            anchor_norm = anchor.lower().replace(" ", "_")
            anchor_info = by_fam.get(anchor_norm, {})
            if anchor_info.get("n_red", 0) > 0:
                warnings.append(
                    f"COUSIN WARNING: anchor family {anchor_norm} has "
                    f"{anchor_info['n_red']} RED entries"
                )
        out.append(CandidateProposal(
            candidate_id=f"pfh_untested_{f}",
            proposal_kind="untested_family",
            family_normalized=f,
            derived_from=[],
            cousin_warnings=warnings,
            needs_new_axes=seed["needs_new_axes"],
            rationale_seeds=[seed["rationale"]],
        ))
    return out


def generate_candidates(
    labels: list[LabeledMechanism],
    *,
    include_extensions:       bool = True,
    include_cross_market:     bool = True,
    include_untested_families: bool = True,
) -> list[CandidateProposal]:
    """Enumerate all candidate proposals from the labeled history.

    Order is deterministic: extensions first, cross-market second,
    untested-family third (matches PFH's "preserve→extend→explore"
    risk gradient — extensions safest, untested-family most speculative).
    """
    out: list[CandidateProposal] = []
    if include_extensions:
        out.extend(_generate_extensions(labels))
    if include_cross_market:
        out.extend(_generate_cross_market(labels))
    if include_untested_families:
        out.extend(_generate_untested_families(labels))
    return out
