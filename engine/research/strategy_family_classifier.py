"""engine.research.strategy_family_classifier — canonical strategy_family
derived from FactorSpec content (not from hypothesis.mechanism_family).

Why this exists
===============
Pre-2026-06-12, factor_dispatcher used `hypothesis.mechanism_family` (claim-
origin classification) as the n_trials denominator for Bailey-Lopez de
Prado §3 Deflated Sharpe. The 2026-06-12 design-flaw audit caught:

  1c258025 (paper claim tagged VALUE)    → spec=HML+MOM 50/50 → event.family=VALUE
  6f1fbaf3 (paper claim tagged MOMENTUM) → spec=HML+MOM 50/50 → event.family=MOMENTUM

Same spec_hash, two different family counters incremented. Bailey-LdP
defines "family of strategies" as the **spec space being searched over**,
NOT the paper-origin claim taxonomy. Two papers about V+M correlation,
both tested with the same canonical 50/50 spec, should count as ONE
trial in a COMBINATION_HML_MOM strategy family.

Without this fix, the same effective trial accumulates penalty across
multiple unrelated claim-tagged families, falsely inflating multi-test
correction, and the real alpha gets misclassified RED.

Doctrine
========
- **claim_family** (= hypothesis.mechanism_family) stays for UI / human
  classification / paper-origin queries. Always inherited from the
  hypothesis at extraction time.
- **strategy_family** (this module) is the canonical Bailey-LdP n_trials
  denominator. Derived from spec content alone, after extraction.
- Two specs with the same canonical structure (same signal_kind, same
  universe, same sorted signal_inputs, same canonical weighting) MUST
  produce the same strategy_family — and therefore share the n_trials
  counter.

Backward compat
===============
- Historical events.jsonl rows have event.family = mechanism_family.
- New events emitted after this commit have event.family = strategy_family
  AND a tag `claim_family:<mechanism_family>` for paper-origin lookup.
- `_family_n_trials_now` reader updated to query by strategy_family
  semantics; old events with mechanism_family-named event.family will
  still match for legacy strategy families that share the name (VALUE /
  MOMENTUM / etc.), so the transition is gradual.

Public API
==========
strategy_family_for_spec(spec) -> str        — canonical family identifier
canonical_strategy_family_tag(spec) -> str   — tag form for events
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Canonical strategy_family identifiers — fixed vocabulary.
# Adding a new family = bump the version + update belief-1 prior
# overrides if the family deserves a non-default base rate.
_STRATEGY_FAMILY_VERSION = "v1_2026-06-12"


# Map for cross_sectional_rank signal_inputs → canonical family
_CROSS_SEC_SIGNAL_FAMILY: dict[str, str] = {
    "mktcap":         "SIZE",
    "vol_12m":        "LOW_VOL",
    "ret_12_1":       "MOMENTUM",
    "ret_6_1":        "MOMENTUM",
    "reversal_1m":    "REVERSAL",
    "gp_at":          "PROFITABILITY",
    "roe":            "PROFITABILITY",
    "book_to_market": "VALUE",
    "at_growth":      "INVESTMENT",
    "op_profit":      "PROFITABILITY",
}


# Ken French factor → canonical short name for factor_combination
_FF_FACTOR_CANONICAL: dict[str, str] = {
    "hml":      "HML",
    "mom":      "MOM",
    "smb":      "SMB",
    "rmw":      "RMW",
    "cma":      "CMA",
    "mkt_rf":   "MKT",
}


def _canonical_ff_factor_name(raw: str) -> Optional[str]:
    """Extract canonical FF factor name from signal_input prefix-stripped form."""
    name = (raw or "").lower()
    for prefix in ("ff.factors_weekly.", "ff.factors_monthly.",
                    "ff.factors_daily.", "ff.factors.", "ff."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return _FF_FACTOR_CANONICAL.get(name)


def _extract_cross_sec_signal_key(signal_inputs: tuple) -> Optional[str]:
    """Identify the dominant cross-sectional signal from signal_inputs."""
    for raw in signal_inputs or ():
        if not isinstance(raw, str):
            continue
        for key in _CROSS_SEC_SIGNAL_FAMILY:
            if key in raw.lower():
                return key
    return None


def strategy_family_for_spec(spec) -> str:
    """Compute the canonical strategy_family for a FactorSpec.

    Returns an uppercase identifier suitable as event.family + tag
    suffix. The classification is DETERMINISTIC and CONTENT-BASED —
    same spec_hash always produces the same strategy_family.

    Falls back to claim_family-style label when content doesn't match
    a known canonical structure; this keeps the function total without
    blowing up on partially-formed specs (e.g., requires_custom_code
    cases).
    """
    sk = getattr(spec, "signal_kind", "") or ""
    universe = getattr(spec, "universe", "") or ""
    inputs = tuple(getattr(spec, "signal_inputs", ()) or ())

    if sk == "cross_sectional_rank":
        key = _extract_cross_sec_signal_key(inputs)
        if key is not None:
            return _CROSS_SEC_SIGNAL_FAMILY[key]
        return "CROSS_SEC_UNKNOWN"

    if sk == "time_series_momentum":
        return "TSMOM"

    if sk == "carry":
        if "fx" in universe:
            return "CARRY_FX"
        if "commodity" in universe:
            return "CARRY_COMMODITY"
        if "treasury" in universe or "rates" in universe:
            return "CARRY_RATES"
        return "CARRY"

    if sk == "factor_combination":
        # Canonicalize: extract FF factor names + sort alphabetically so
        # 50/50 HML+MOM and 50/50 MOM+HML hash to the same family.
        factors = []
        for raw in inputs:
            name = _canonical_ff_factor_name(raw)
            if name is not None:
                factors.append(name)
        if len(factors) >= 2:
            factors_sorted = sorted(set(factors))
            return "COMBINATION_" + "_".join(factors_sorted)
        return "COMBINATION_UNKNOWN"

    if sk == "spanning_test":
        # Canonical family is "SPANNING_<test_asset>" because spanning
        # tests of the SAME test_asset against any model variant belong
        # to the same Bailey-LdP family ("does FF5 span MOM" / "does
        # FF5+RMW span MOM" / etc. all test MOM's orthogonal alpha).
        if inputs:
            test_asset = _canonical_ff_factor_name(inputs[0])
            if test_asset is not None:
                return f"SPANNING_{test_asset}"
        return "SPANNING_UNKNOWN"

    if sk == "portfolio_overlay":
        if "60_40" in universe:
            return "OVERLAY_60_40"
        return "OVERLAY_UNKNOWN"

    if sk == "vrp":
        return "VRP"

    if sk == "event_drift":
        return "EVENT_DRIFT"

    if sk == "requires_custom_code":
        return "CUSTOM_CODE"

    # Unknown / future signal_kinds — be honest, don't silently default
    # to a real strategy family
    return f"UNKNOWN_{sk.upper()}" if sk else "UNKNOWN"


def canonical_strategy_family_tag(spec) -> str:
    """Tag form: `strategy_family:<X>` for inclusion in event.tags."""
    return f"strategy_family:{strategy_family_for_spec(spec)}"


def claim_family_tag(family_hint: str) -> str:
    """Tag form: `claim_family:<X>` so the original mechanism_family
    is preserved on the event for UI / paper-origin queries even after
    event.family migrates to strategy_family semantics."""
    return f"claim_family:{(family_hint or 'UNKNOWN').upper()}"
