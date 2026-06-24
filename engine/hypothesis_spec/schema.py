"""hypothesis_spec.schema — typed dataclasses for the structured spec.

Design principles
-----------------
1. Immutable (frozen=True) — once a spec is hashed it must never change.
   Changes happen via spec.bump_version() returning a new spec.
2. JSON-roundtrippable — to_dict / from_dict are exact inverses.
3. Hash-deterministic — same content always produces the same spec_hash
   regardless of field-insertion order or Python version.
4. Multi-leg by design — many hypotheses combine 2+ signals (carry × momentum
   filter, value × quality, etc), so legs is a list not a single object.
5. Backward-compatible — schema_version field bumps when fields are
   added/removed; loaders must accept versions they understand and warn
   (not crash) on newer versions per LdP §2.

Spec example (in JSON form)
---------------------------
{
  "spec_version": 1,
  "spec_id": "uuid-...",
  "source_hypothesis_id": "h_abc...",
  "family": "CARRY",
  "claim_text": "FX carry on G10 with monthly rebalance",
  "legs": [{
    "signal_type": "CARRY_FORWARD_DISCOUNT",
    "lookback_periods": [1],
    "sign": "LONG_SHORT",
    "quantile": 0.30
  }],
  "universe": {
    "asset_class": "FX",
    "subset": "G10",
    "custom_tickers": null,
    "min_history_months": 36
  },
  "construction": {
    "weighting": "INV_VOL",
    "rebalance": "MONTHLY",
    "skip_first_day": true
  },
  "risk": {
    "vol_target_annual": 0.10,
    "max_leverage": 3.0,
    "turnover_cap_annual": null
  },
  "outcome": {
    "predicted_direction": "POSITIVE",
    "predicted_sharpe_lo": 0.3,
    "predicted_sharpe_hi": 0.8,
    "rationale": "well-documented forward-discount predictability"
  },
  "extraction": {
    "method": "claude_sonnet_4_6_v1",
    "confidence": 0.85,
    "extracted_ts": "2026-06-05T..."
  }
}
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import uuid as _uuid
from typing import Optional

from engine.hypothesis_spec.enums import (
    ClaimType, FamilyV2, AssetClass, SignalType, Sign,
    UniverseSubset, Weighting, Rebalance, PredictedDirection,
)


SPEC_VERSION = 2   # B.2-A1 (2026-06-05): added claim_type. v1 specs auto-load with claim_type=UNKNOWN.


# ── Sub-dataclasses ───────────────────────────────────────────


@_dc.dataclass(frozen=True)
class SignalLeg:
    """One signal leg of the hypothesis. A hypothesis can have multiple
    legs combined multiplicatively (filter) or additively (overlay)."""
    signal_type:      SignalType
    sign:             Sign                  = Sign.LONG_SHORT
    lookback_periods: tuple[int, ...]       = (12,)       # months
    quantile:         float                 = 0.30        # top/bottom 30% default
    role:             str                   = "primary"   # "primary" | "filter" | "overlay"
    note:             str                   = ""


@_dc.dataclass(frozen=True)
class Universe:
    asset_class:        AssetClass
    subset:             UniverseSubset      = UniverseSubset.ALL
    custom_tickers:     Optional[tuple[str, ...]] = None
    min_history_months: int                 = 36
    sub_classes:        tuple[AssetClass, ...] = ()  # for COMBINED


@_dc.dataclass(frozen=True)
class PortfolioConstruction:
    weighting:        Weighting             = Weighting.EQUAL
    rebalance:        Rebalance             = Rebalance.MONTHLY
    skip_first_day:   bool                  = True    # avoid PIT contamination
    holding_period_n: int                   = 1       # in rebalance units


@_dc.dataclass(frozen=True)
class RiskManagement:
    vol_target_annual:   Optional[float] = 0.10
    max_leverage:        Optional[float] = 3.0
    turnover_cap_annual: Optional[float] = None
    max_position:        Optional[float] = None
    drawdown_stop:       Optional[float] = None       # absolute -dd %


@_dc.dataclass(frozen=True)
class PredictedOutcome:
    predicted_direction:    PredictedDirection = PredictedDirection.POSITIVE
    predicted_sharpe_lo:    Optional[float]    = None
    predicted_sharpe_hi:    Optional[float]    = None
    rationale:              str                = ""


@_dc.dataclass(frozen=True)
class Extraction:
    """Provenance of HOW this spec was derived (LLM extraction etc)."""
    method:        str       = "manual"    # e.g. "claude_sonnet_4_6_v1"
    confidence:    float     = 1.0
    extracted_ts:  str       = ""
    extractor_v:  str        = "v1"


# ── Top-level spec ────────────────────────────────────────────


@_dc.dataclass(frozen=True)
class HypothesisSpec:
    """The structured form of a research hypothesis.

    Identity:
      spec_id              UUID4 per spec instance
      spec_version         schema version (current = 1)
      source_hypothesis_id link back to the free-text hypothesis row
      version              monotonic per-spec revision (bumps via .bump_version())

    Content:
      family               top-level mechanism family
      claim_text           the original natural-language claim (immutable archive)
      legs                 list of signal legs (carry × momentum = 2 legs)
      universe             which asset / subset
      construction         portfolio construction (weighting / rebalance)
      risk                 risk overlay
      outcome              what's predicted

    Provenance:
      extraction           how the spec was derived (LLM / manual)
      git_sha              repo HEAD at spec creation
      created_ts           when this spec instance was constructed
    """
    spec_id:              str
    spec_version:         int
    source_hypothesis_id: str
    version:              int

    claim_type:           ClaimType   # B.2-A1: gate for Composer / direction_proposer
    family:               FamilyV2    # only meaningful when claim_type == FACTOR_HYPOTHESIS
    claim_text:           str

    legs:                 tuple[SignalLeg, ...]
    universe:             Universe
    construction:         PortfolioConstruction
    risk:                 RiskManagement
    outcome:              PredictedOutcome
    extraction:           Extraction

    git_sha:              str
    created_ts:           str

    # ── Constructors ─────────────────────────────────────────

    @classmethod
    def new(
        cls,
        *,
        source_hypothesis_id: str,
        family:               FamilyV2,
        claim_text:           str,
        legs:                 tuple[SignalLeg, ...],
        universe:             Universe,
        claim_type:           ClaimType             = ClaimType.UNKNOWN,
        construction:         PortfolioConstruction = PortfolioConstruction(),
        risk:                 RiskManagement        = RiskManagement(),
        outcome:              PredictedOutcome      = PredictedOutcome(),
        extraction:           Extraction            = Extraction(),
        git_sha:              str = "",
    ) -> "HypothesisSpec":
        return cls(
            spec_id              = str(_uuid.uuid4()),
            spec_version         = SPEC_VERSION,
            source_hypothesis_id = source_hypothesis_id,
            version              = 1,
            claim_type           = claim_type,
            family               = family,
            claim_text           = claim_text,
            legs                 = tuple(legs),
            universe             = universe,
            construction         = construction,
            risk                 = risk,
            outcome              = outcome,
            extraction           = extraction,
            git_sha              = git_sha,
            created_ts           = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def bump_version(self, **changes) -> "HypothesisSpec":
        """Return a new spec with version+1 and any field changes
        (keyword-overrides). The original is immutable."""
        return _dc.replace(self,
                          spec_id    = str(_uuid.uuid4()),
                          version    = self.version + 1,
                          created_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                          **changes)

    # ── Serialization ────────────────────────────────────────

    def to_dict(self) -> dict:
        """Plain-dict for jsonl persistence + LLM context."""
        d = _dc.asdict(self)
        # Enums → str values for JSON cleanliness
        d["claim_type"]                      = self.claim_type.value
        d["family"]                          = self.family.value
        d["universe"]["asset_class"]         = self.universe.asset_class.value
        d["universe"]["subset"]              = self.universe.subset.value
        d["universe"]["sub_classes"]         = [a.value for a in self.universe.sub_classes]
        d["construction"]["weighting"]       = self.construction.weighting.value
        d["construction"]["rebalance"]       = self.construction.rebalance.value
        d["outcome"]["predicted_direction"]  = self.outcome.predicted_direction.value
        d["legs"] = [{
            "signal_type":      L.signal_type.value,
            "sign":             L.sign.value,
            "lookback_periods": list(L.lookback_periods),
            "quantile":         L.quantile,
            "role":             L.role,
            "note":             L.note,
        } for L in self.legs]
        if self.universe.custom_tickers is not None:
            d["universe"]["custom_tickers"] = list(self.universe.custom_tickers)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "HypothesisSpec":
        """Round-trip from a dict. Warns (does not crash) on unknown
        future spec_version per LdP §2."""
        import logging
        v = int(d.get("spec_version", 1))
        if v > SPEC_VERSION:
            logging.getLogger(__name__).warning(
                "HypothesisSpec.from_dict reading spec_version=%s > current=%s; "
                "best-effort load", v, SPEC_VERSION)
        u = d["universe"]
        legs = tuple(SignalLeg(
            signal_type      = SignalType(L["signal_type"]),
            sign             = Sign(L.get("sign", "LONG_SHORT")),
            lookback_periods = tuple(L.get("lookback_periods", (12,))),
            quantile         = float(L.get("quantile", 0.30)),
            role             = L.get("role", "primary"),
            note             = L.get("note", ""),
        ) for L in d.get("legs", []))
        universe = Universe(
            asset_class        = AssetClass(u["asset_class"]),
            subset             = UniverseSubset(u.get("subset", "ALL")),
            custom_tickers     = tuple(u["custom_tickers"]) if u.get("custom_tickers") else None,
            min_history_months = int(u.get("min_history_months", 36)),
            sub_classes        = tuple(AssetClass(x) for x in u.get("sub_classes", [])),
        )
        c = d.get("construction") or {}
        construction = PortfolioConstruction(
            weighting        = Weighting(c.get("weighting", "EQUAL")),
            rebalance        = Rebalance(c.get("rebalance", "MONTHLY")),
            skip_first_day   = bool(c.get("skip_first_day", True)),
            holding_period_n = int(c.get("holding_period_n", 1)),
        )
        r = d.get("risk") or {}
        risk = RiskManagement(
            vol_target_annual   = r.get("vol_target_annual"),
            max_leverage        = r.get("max_leverage"),
            turnover_cap_annual = r.get("turnover_cap_annual"),
            max_position        = r.get("max_position"),
            drawdown_stop       = r.get("drawdown_stop"),
        )
        o = d.get("outcome") or {}
        outcome = PredictedOutcome(
            predicted_direction = PredictedDirection(o.get("predicted_direction", "UNKNOWN")),
            predicted_sharpe_lo = o.get("predicted_sharpe_lo"),
            predicted_sharpe_hi = o.get("predicted_sharpe_hi"),
            rationale           = o.get("rationale", ""),
        )
        e = d.get("extraction") or {}
        extraction = Extraction(
            method       = e.get("method", "manual"),
            confidence   = float(e.get("confidence", 1.0)),
            extracted_ts = e.get("extracted_ts", ""),
            extractor_v  = e.get("extractor_v", "v1"),
        )
        # B.2-A1: claim_type added in v2; v1 specs default to UNKNOWN so
        # they're caught by Composer/direction_proposer FACTOR_HYPOTHESIS filter
        # and surfaced for re-extraction rather than silently misclassified.
        try:
            claim_type = ClaimType(d.get("claim_type", "UNKNOWN"))
        except ValueError:
            claim_type = ClaimType.UNKNOWN
        return cls(
            spec_id              = d["spec_id"],
            spec_version         = v,
            source_hypothesis_id = d["source_hypothesis_id"],
            version              = int(d.get("version", 1)),
            claim_type           = claim_type,
            family               = FamilyV2(d["family"]),
            claim_text           = d.get("claim_text", ""),
            legs                 = legs,
            universe             = universe,
            construction         = construction,
            risk                 = risk,
            outcome              = outcome,
            extraction           = extraction,
            git_sha              = d.get("git_sha", ""),
            created_ts           = d.get("created_ts", ""),
        )
