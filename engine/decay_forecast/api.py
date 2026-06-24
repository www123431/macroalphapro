"""engine.decay_forecast.api — query interface for candidate decay estimates.

Re-uses FAMILY_DECAY_PARAMS table from engine.research.forward_decay_prediction
(single source of truth for MP 2016 / LR 2018 λ values), wrapping a
candidate-keyed query suitable for pre-flight wizard consumption.
"""
from __future__ import annotations

import datetime
import math

from engine.decay_forecast.schema import DecayEstimate, DecayRisk
from engine.research.forward_decay_prediction import FAMILY_DECAY_PARAMS


_DEFAULT_BASELINE_ALPHA = 0.05   # 5%/yr — average earnings underreaction baseline
_DEFAULT_PUB_YEAR_LOOKBACK = 0   # candidate has not been published yet


def _decay_at_year(baseline: float, lam: float, t_years: float) -> float:
    """Exponential decay model. Identical to forward_decay_prediction._decay_at_year."""
    return baseline * math.exp(-lam * t_years)


def _classify_risk(retention_5y: float) -> tuple[DecayRisk, str]:
    """Map 5-year retention ratio → coarse risk band + actionable note."""
    if retention_5y >= 0.70:
        return DecayRisk.LOW, (
            "Low decay family — most α expected to survive 5y. Family-typical "
            "for structural mechanisms (insurance, vol-target overlays)."
        )
    if retention_5y >= 0.40:
        return DecayRisk.MEDIUM, (
            "Moderate decay — 30-60% α loss over 5y typical for this family. "
            "Plan for retest at half-life."
        )
    if retention_5y >= 0.15:
        return DecayRisk.HIGH, (
            "Heavy decay family (well-arbitraged literature). Strongly consider "
            "whether your edge is genuinely novel vs. re-attempting a known mechanism."
        )
    return DecayRisk.SEVERE, (
        "SEVERE decay expected — published α essentially dead at 5y forward. "
        "Do not deploy without a clearly novel mechanism beyond the family baseline."
    )


def estimate_for_family(
    family: str,
    baseline_alpha: float | None = None,
    publication_year: int | None = None,
) -> DecayEstimate:
    """Forward decay estimate for a candidate factor by family.

    Args:
        family: Family label (e.g. 'earnings_underreaction', 'carry', 'momentum').
                If unknown, falls back to '_default' (MP 2016 average).
        baseline_alpha: Assumed α at audit time (default 5%/yr). UI may pass
                        the candidate's PFH posterior_mean × annualization.
        publication_year: Year the mechanism was published in academic
                          literature. None → treat as unpublished (years_since_pub=0).

    Returns:
        DecayEstimate with MP 2016 central path + LR 2018 lower bound +
        5y/10y forecasts + risk band + actionable note.
    """
    family = (family or "_default").strip()
    parent_family = None  # caller can pass parent_family via separate API later

    params = FAMILY_DECAY_PARAMS.get(family)
    using_default = False
    if params is None:
        params = FAMILY_DECAY_PARAMS["_default"]
        using_default = True

    if baseline_alpha is None:
        baseline_alpha = _DEFAULT_BASELINE_ALPHA

    current_year = datetime.date.today().year
    years_since_pub = (
        max(0.0, float(current_year - publication_year))
        if publication_year is not None
        else 0.0
    )

    lam = float(params["lambda"])
    lr_lam = float(params["lr_lambda"])
    half_life = math.log(2) / lam if lam > 0 else float("inf")

    # Central path (MP 2016)
    expected_now = _decay_at_year(baseline_alpha, lam, years_since_pub)
    expected_5y = _decay_at_year(baseline_alpha, lam, years_since_pub + 5)
    expected_10y = _decay_at_year(baseline_alpha, lam, years_since_pub + 10)

    # Pessimistic lower (LR 2018)
    expected_5y_lower = _decay_at_year(baseline_alpha, lr_lam, years_since_pub + 5)
    expected_10y_lower = _decay_at_year(baseline_alpha, lr_lam, years_since_pub + 10)

    retention_5y = expected_5y / baseline_alpha if baseline_alpha > 0 else 0.0
    risk, note = _classify_risk(retention_5y)

    return DecayEstimate(
        family=family,
        parent_family=parent_family,
        using_default=using_default,
        baseline_alpha=baseline_alpha,
        publication_year=publication_year,
        years_since_pub=years_since_pub,
        mp_2016_lambda=lam,
        lr_2018_lambda=lr_lam,
        half_life_years=half_life,
        expected_alpha_now=expected_now,
        expected_alpha_5y=expected_5y,
        expected_alpha_10y=expected_10y,
        expected_alpha_5y_lower=expected_5y_lower,
        expected_alpha_10y_lower=expected_10y_lower,
        risk=risk,
        note=note,
    )


def list_supported_families() -> list[dict]:
    """Return registered families + their decay parameters.
    UI dropdown can populate from this so user picks the right family
    for their candidate."""
    out = []
    for fam_name, params in FAMILY_DECAY_PARAMS.items():
        if fam_name == "_default":
            continue
        lam = float(params["lambda"])
        out.append({
            "family":           fam_name,
            "mp_2016_lambda":   lam,
            "lr_2018_lambda":   float(params["lr_lambda"]),
            "half_life_years":  math.log(2) / lam if lam > 0 else None,
            "notes":            params.get("notes", ""),
        })
    out.sort(key=lambda d: d["family"])
    return out
