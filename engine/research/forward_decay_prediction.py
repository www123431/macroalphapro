"""engine/research/forward_decay_prediction.py — Phase 1 P1a of loop
robustness.

Predicts EXPECTED REMAINING ALPHA for a deployed mechanism given:
  - publication date (if any)
  - factor family
  - years since publication
  - prior gate Sharpe

Per McLean-Pontiff 2016 (JF) "Does Academic Research Destroy Stock Return
Predictability" — average ~58% post-publication decay across 97
anomalies. Linnainmaa-Roberts 2018 (RFS) finds decay even larger,
~70-80% for sample tested.

Combined with our domain knowledge:
  - Pure equity factors: stronger decay (more arb)
  - Multi-asset / cross-section: moderate decay
  - Insurance / regime overlays: little expected decay (not arb-able same way)

OUTPUT: ForwardDecayPrediction with:
  expected_alpha_now
  half_life_years
  expected_alpha_at_year(t)
  confidence_band (mclean_pontiff_2016 lower / linnainmaa_roberts_2018 upper)
  recommended_review_date

Per [[feedback-loop-is-robustness-doctrine-2026-05-31]] Phase 1 P1a.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import math
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"

# Decay parameters per family (annualized exponential decay rate λ)
# Source: McLean-Pontiff 2016 + Linnainmaa-Roberts 2018 + our judgment.
# half_life_years = ln(2) / λ
# Empirical MP 2016 average decay over 4-yr post-pub: 58% → λ ≈ 0.217
# LR 2018 upper bound: 70% over 4 years → λ ≈ 0.301
FAMILY_DECAY_PARAMS = {
    # Pure equity factors — heavy arb-driven decay
    "earnings_underreaction":  {"lambda": 0.20, "lr_lambda": 0.30, "notes": "MP 2016 SUE row close to average"},
    "momentum":                {"lambda": 0.25, "lr_lambda": 0.35, "notes": "well-arb-able"},
    "quality":                 {"lambda": 0.15, "lr_lambda": 0.25, "notes": "harder to arb"},
    "low_vol":                 {"lambda": 0.18, "lr_lambda": 0.28, "notes": "BAB decayed substantially"},
    "residual_momentum":       {"lambda": 0.20, "lr_lambda": 0.30, "notes": "follows momentum decay"},

    # Cross-asset — moderate decay
    "carry":                   {"lambda": 0.12, "lr_lambda": 0.20, "notes": "risk premium harder to arb"},
    "tsmom":                   {"lambda": 0.15, "lr_lambda": 0.25, "notes": "CTA crowding"},
    "cross_asset_hedge":       {"lambda": 0.08, "lr_lambda": 0.15, "notes": "structural hedge"},
    "vol_carry":               {"lambda": 0.20, "lr_lambda": 0.30, "notes": "options-based, well-arb"},

    # Insurance / overlays — minimal decay (different mechanism)
    "factor_hedge":            {"lambda": 0.05, "lr_lambda": 0.10, "notes": "structural, not arb-able"},
    "hedge_overlay":           {"lambda": 0.05, "lr_lambda": 0.10, "notes": "structural"},

    # Default fallback
    "_default":                {"lambda": 0.20, "lr_lambda": 0.30, "notes": "average MP 2016"},
}


@dataclasses.dataclass
class ForwardDecayPrediction:
    mechanism_id:                  str
    family:                        str
    publication_year:              int | None
    current_year:                  int
    years_since_publication:       float
    baseline_alpha:                float       # alpha at audit time
    expected_alpha_now:            float       # baseline × decay-to-now
    expected_alpha_5yr_ahead:      float
    expected_alpha_10yr_ahead:     float
    half_life_years:               float
    mp_2016_lambda:                float
    lr_2018_lambda:                float
    recommended_review_date:       str         # date when alpha < 0.7 × current
    confidence_band: dict          # mp_lower / lr_upper / decay_curve

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _decay_at_year(baseline: float, lambda_: float, t_years: float) -> float:
    """Exponential decay model."""
    return baseline * math.exp(-lambda_ * t_years)


def _decay_curve(baseline: float, lambda_: float, max_years: int = 10) -> dict:
    """Return year-by-year forecast for first max_years years."""
    return {
        t: round(_decay_at_year(baseline, lambda_, t), 4)
        for t in range(max_years + 1)
    }


def predict_decay(mechanism_id: str,
                       baseline_alpha: float | None = None) -> ForwardDecayPrediction:
    """Build forward decay prediction for a library mechanism.

    Reads library YAML for:
      family (informs decay rate)
      was_known_before_our_data_cutoff.publication_date
      factor_exposure.alpha_annualized (default baseline if not supplied)
    """
    fp = LIBRARY_DIR / f"{mechanism_id}.yaml"
    if not fp.exists():
        raise FileNotFoundError(f"library entry {mechanism_id!r} not found at {fp}")
    entry = yaml.safe_load(fp.read_text(encoding="utf-8"))

    family = entry.get("family") or "_default"
    parent = entry.get("parent_family")
    # Try family first, then parent, then default
    decay_params = (FAMILY_DECAY_PARAMS.get(family)
                    or FAMILY_DECAY_PARAMS.get(parent)
                    or FAMILY_DECAY_PARAMS["_default"])

    # Pub year
    pub_block = entry.get("was_known_before_our_data_cutoff") or {}
    pub_str = pub_block.get("publication_date")
    if pub_str:
        try:
            pub_year = int(str(pub_str)[:4])
        except Exception:
            pub_year = None
    else:
        pub_year = None

    current_year = datetime.date.today().year
    years_since_pub = max(0.0, (current_year - pub_year)) if pub_year else 0.0

    # Baseline alpha
    if baseline_alpha is None:
        fe = entry.get("factor_exposure") or {}
        baseline_alpha = fe.get("alpha_annualized")
        if baseline_alpha is None:
            baseline_alpha = 0.05    # fallback assumption 5%/yr

    lam = decay_params["lambda"]
    lr_lam = decay_params["lr_lambda"]
    half_life = math.log(2) / lam if lam > 0 else float("inf")

    # Current expectation
    expected_now = _decay_at_year(baseline_alpha, lam, years_since_pub)

    # Year-N forecasts
    e_5y = _decay_at_year(baseline_alpha, lam, years_since_pub + 5)
    e_10y = _decay_at_year(baseline_alpha, lam, years_since_pub + 10)

    # Confidence band
    mp_lower = {t: _decay_at_year(baseline_alpha, lr_lam, years_since_pub + t)
                for t in [0, 1, 2, 3, 5, 10]}
    mp_main = {t: _decay_at_year(baseline_alpha, lam, years_since_pub + t)
                for t in [0, 1, 2, 3, 5, 10]}

    # Recommended review date: when alpha is expected to fall below 70% of
    # current expected
    target_alpha = expected_now * 0.7
    if expected_now > 0:
        review_years_ahead = math.log(expected_now / target_alpha) / lam
        review_date = datetime.date.today() + datetime.timedelta(
            days=int(review_years_ahead * 365.25),
        )
    else:
        review_date = datetime.date.today() + datetime.timedelta(days=365)

    return ForwardDecayPrediction(
        mechanism_id=mechanism_id,
        family=family,
        publication_year=pub_year,
        current_year=current_year,
        years_since_publication=years_since_pub,
        baseline_alpha=float(baseline_alpha),
        expected_alpha_now=float(expected_now),
        expected_alpha_5yr_ahead=float(e_5y),
        expected_alpha_10yr_ahead=float(e_10y),
        half_life_years=float(half_life),
        mp_2016_lambda=float(lam),
        lr_2018_lambda=float(lr_lam),
        recommended_review_date=review_date.isoformat(),
        confidence_band={
            "mp_2016_main": mp_main,
            "lr_2018_lower": mp_lower,
        },
    )


def predict_all_audited() -> list[ForwardDecayPrediction]:
    """Run prediction for every library entry with audit_status=audited."""
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8"))
            fe = entry.get("factor_exposure") or {}
            if fe.get("audit_status") != "audited":
                continue
            pred = predict_decay(fp.stem)
            out.append(pred)
        except Exception as exc:
            logger.warning("predict_decay failed for %s: %s", fp.stem, exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mechanism-id", default=None,
                          help="single mechanism to predict (default: all audited)")
    args = parser.parse_args()

    if args.mechanism_id:
        predictions = [predict_decay(args.mechanism_id)]
    else:
        predictions = predict_all_audited()

    print(f"[forward_decay] predicting for {len(predictions)} mechanism(s)")
    print()
    print(f"  {'mechanism':<28}  {'family':<14}  "
          f"{'pub':>5}  {'now':>7}  {'+5y':>7}  {'+10y':>7}  {'half-life':>9}")
    print(f"  {'-'*28}  {'-'*14}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
    for p in predictions:
        pub = p.publication_year or "n/a"
        print(f"  {p.mechanism_id:<28}  {p.family:<14}  {str(pub):>5}  "
              f"{p.expected_alpha_now:>+7.2%}  "
              f"{p.expected_alpha_5yr_ahead:>+7.2%}  "
              f"{p.expected_alpha_10yr_ahead:>+7.2%}  "
              f"{p.half_life_years:>8.1f}y")
        print(f"    review by: {p.recommended_review_date}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
