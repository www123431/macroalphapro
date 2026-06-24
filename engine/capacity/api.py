"""engine.capacity.api — family-keyed capacity estimate query.

Sub-MVP per Gap C audit. Provides a fast lookup table for CANDIDATE
research axes (no full simulation required) — enables a capacity badge
on /lab/roadmap parallel to the decay badge.

For DETAILED capacity analysis at specific AUM levels under historical
fill data, use engine.portfolio.capacity_simulator (Pastor-Stambaugh
framework, full TC simulation, per-AUM Sharpe + DD).
"""
from __future__ import annotations

import math

from engine.capacity.schema import CapacityClass, CapacityEstimate


# Family-typical capacity heuristics.
#
# Source: industry capacity disclosures + academic capacity literature:
#   - AQR factor capacity notes (2018, 2021)
#   - Two Sigma research notes on factor capacity
#   - Pastor-Stambaugh 2002 / Berk-Green 2004 capacity-decay framework
#   - Korajczyk-Sadka 2004 momentum capacity testing
#   - Frazzini-Israel-Moskowitz 2018 trading costs calibration
#
# Numbers are ORDER OF MAGNITUDE, not point estimates. For specific-strategy
# decisions, run engine.portfolio.capacity_simulator.
#
# Three thresholds per family:
#   estimated_capacity_usd  — AUM where expected Sharpe falls to 50% of paper
#                              level (i.e., capacity halves)
#   comfortable_aum_usd     — AUM where ~80% Sharpe retained; institutional
#                              "sweet spot"
#   minimum_aum_usd         — below this, fixed infrastructure / data costs
#                              eat α. Set at $250k for single-quant ops.
FAMILY_CAPACITY_PARAMS: dict[str, dict] = {
    # Equity single-name factors
    "earnings_underreaction": {
        "capacity_class":           CapacityClass.MEDIUM,
        "estimated_capacity_usd":   500_000_000,    # $500M
        "comfortable_aum_usd":      150_000_000,
        "notes":                    "ADV-limited at $500M; turnover 4-6x annually amplifies impact.",
        "methodology":              "Korajczyk-Sadka 2004 capacity test on similar event-driven equity.",
    },
    "momentum": {
        "capacity_class":           CapacityClass.HIGH,
        "estimated_capacity_usd":   2_000_000_000,  # $2B
        "comfortable_aum_usd":      600_000_000,
        "notes":                    "Liquid universe but momentum has well-documented capacity-decay (Korajczyk-Sadka 2004).",
        "methodology":              "AQR / DFA disclosed momentum-strategy capacity in $1-5B range.",
    },
    "quality": {
        "capacity_class":           CapacityClass.HIGH,
        "estimated_capacity_usd":   3_000_000_000,
        "comfortable_aum_usd":      1_000_000_000,
        "notes":                    "Low turnover (~1x/yr) → high capacity. Stable fundamentals → less crowding sensitivity.",
        "methodology":              "AQR Quality-Minus-Junk fund capacity disclosures.",
    },
    "low_vol": {
        "capacity_class":           CapacityClass.HIGH,
        "estimated_capacity_usd":   2_000_000_000,
        "comfortable_aum_usd":      750_000_000,
        "notes":                    "Big-cap-biased → liquid. Frazzini-Pedersen BAB had institutional capacity $1B+.",
        "methodology":              "BAB fund (AQR) historical AUM range.",
    },
    "residual_momentum": {
        "capacity_class":           CapacityClass.MEDIUM,
        "estimated_capacity_usd":   800_000_000,
        "comfortable_aum_usd":      250_000_000,
        "notes":                    "Higher turnover than pure momentum; capacity shaved by ~50%.",
        "methodology":              "Inferred from momentum baseline × residualization overhead.",
    },

    # Cross-asset (futures markets are deep)
    "carry": {
        "capacity_class":           CapacityClass.VERY_HIGH,
        "estimated_capacity_usd":   5_000_000_000,  # $5B
        "comfortable_aum_usd":      1_500_000_000,
        "notes":                    "FX + commodity + bond futures markets aggregate >$200B daily; cross-asset diversification keeps single-leg exposures small.",
        "methodology":              "Koijen-Moskowitz-Pedersen-Vrugt 2018 capacity estimates for carry-everywhere.",
    },
    "tsmom": {
        "capacity_class":           CapacityClass.VERY_HIGH,
        "estimated_capacity_usd":   8_000_000_000,
        "comfortable_aum_usd":      2_500_000_000,
        "notes":                    "CTA industry runs $300B+ in trend strategies; capacity is enormous but Sharpe decays from crowding.",
        "methodology":              "AQR Managed Futures Strategy + Winton historical AUM.",
    },
    "cross_asset_hedge": {
        "capacity_class":           CapacityClass.HIGH,
        "estimated_capacity_usd":   2_000_000_000,
        "comfortable_aum_usd":      600_000_000,
        "notes":                    "Structural hedge sleeves are typically liquid; capacity bound by counterparty (vol-target overlays) not market.",
        "methodology":              "AQR Style Premia / Bridgewater All Weather hedge components.",
    },
    "vol_carry": {
        "capacity_class":           CapacityClass.MEDIUM,
        "estimated_capacity_usd":   500_000_000,
        "comfortable_aum_usd":      150_000_000,
        "notes":                    "Options markets have tighter capacity than futures; OptionMetrics universe is the bound.",
        "methodology":              "Volatility-strategy fund disclosures ($200M-$1B typical).",
    },

    # Insurance / structural
    "factor_hedge": {
        "capacity_class":           CapacityClass.HIGH,
        "estimated_capacity_usd":   2_000_000_000,
        "comfortable_aum_usd":      750_000_000,
        "notes":                    "Big-cap-hedge instruments (TLT/GLD/SPX put-spreads) are deep; capacity is high.",
        "methodology":              "Direct readout of SPX option / TLT ADV.",
    },
    "hedge_overlay": {
        "capacity_class":           CapacityClass.VERY_HIGH,
        "estimated_capacity_usd":   10_000_000_000,
        "comfortable_aum_usd":      3_000_000_000,
        "notes":                    "Vol-target overlays use futures + deep ETFs; effectively unbounded for solo-quant scale.",
        "methodology":              "Risk-parity / vol-target industry AUM ($500B+ aggregate).",
    },

    # Default fallback
    "_default": {
        "capacity_class":           CapacityClass.MEDIUM,
        "estimated_capacity_usd":   500_000_000,
        "comfortable_aum_usd":      150_000_000,
        "notes":                    "Default capacity assumption for unregistered family.",
        "methodology":              "MP 2016 average / industry median.",
    },
}


_MINIMUM_AUM_USD = 250_000   # Below this single-quant ops doesn't make sense


def estimate_for_family(family: str) -> CapacityEstimate:
    """Family-keyed capacity estimate for a candidate research axis.

    Args:
        family: Family label. If unknown, falls back to '_default'.

    Returns:
        CapacityEstimate with capacity_class + 3 AUM thresholds + notes.
    """
    family = (family or "_default").strip()
    params = FAMILY_CAPACITY_PARAMS.get(family)
    using_default = False
    if params is None:
        params = FAMILY_CAPACITY_PARAMS["_default"]
        using_default = True

    return CapacityEstimate(
        family=family,
        using_default=using_default,
        capacity_class=params["capacity_class"],
        estimated_capacity_usd=float(params["estimated_capacity_usd"]),
        comfortable_aum_usd=float(params["comfortable_aum_usd"]),
        minimum_aum_usd=_MINIMUM_AUM_USD,
        notes=params["notes"],
        methodology=params["methodology"],
        parent_family=None,
    )


def list_supported_families() -> list[dict]:
    """Return registered families with capacity parameters."""
    out = []
    for fam, params in FAMILY_CAPACITY_PARAMS.items():
        if fam == "_default":
            continue
        out.append({
            "family":                  fam,
            "capacity_class":          params["capacity_class"].value,
            "estimated_capacity_usd":  float(params["estimated_capacity_usd"]),
            "comfortable_aum_usd":     float(params["comfortable_aum_usd"]),
            "notes":                   params["notes"],
        })
    out.sort(key=lambda d: d["family"])
    return out
