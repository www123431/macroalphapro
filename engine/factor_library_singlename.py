"""
engine/factor_library_singlename.py — Tier 1 mining lab content layer (single-stock).

Status: Tier 1 mining lab infrastructure (created 2026-05-10).

This module is the **content layer** for single-stock Tier 1 mining — provides
FACTOR_REGISTRY_SINGLENAME (named factor signal_fn closures + metadata) that
Tier 1 mining_runner consumes to walk-forward each candidate independently.

Mirrors `engine/factor_library.py` boundary pattern (ETF Tier 2 content layer):

    factor_library_singlename  →  factor_lab.power.power_check
    factor_lab                 →/  factor_library_singlename  (FORBIDDEN)
                                   ↑ enforced by Tier R rule
                                     `rule_factor_lab_no_factor_library_import`

Boundary invariant (per spec_factor_lab.md §6 + memory
`feedback_factor_research_3_tier_framework`): zero LLM imports — pure
deterministic signals. Tier 1 mining never invokes LLM in the alpha decision
loop.

Tier 1 vs Tier 2 distinction
----------------------------
Tier 1 mining (this module + `factor_lab.mining_runner`):
  - Spec registration: `factor_kind="infrastructure_spec"` → +0 trials
    (P-LAB exempt per spec_factor_lab §6.1)
  - Verdict tiers: directional_positive / dropped / promotable_to_tier_2
  - OOS holdout discipline (last 24mo reserved per
    feedback_factor_research_3_tier_framework Tier 1 design)
  - 0 BHY-FDR, 0 EFFECTIVE_N_TRIALS contribution
  - Output: `data/factor_mining_lab/<factor_id>_<date>.json` + verdict markdown
            in `docs/decisions/factor_mining_<factor_id>_<date>.md`

Tier 2 ETF candidate gate (`engine/factor_library.py` + `factor_lab.runner`):
  - Spec registration: `factor_kind=None` (research_hypothesis) → +1 trial
  - Verdict tiers: PASS / MARGINAL / FAIL / FAIL_UNDERPOWERED (BHY 5% / 10% sig)
  - Strict BHY-FDR + multi-baseline + power analysis
  - Output: per-factor verdict markdown + production candidate proposal

Promotion gate (Tier 1 → Tier 2)
--------------------------------
NOT automatic. Tier 1 mining producing `directional_positive` does NOT
auto-register a Tier 2 candidate. Promotion requires:
  1. Manual review of Tier 1 mining verdict
  2. Re-register the factor as a research_hypothesis spec (factor_kind=None,
     +1 trial via amend_spec/register_spec)
  3. Run through `factor_lab.runner` Tier 2 gate (BHY 5% sig)
  4. Capital allocation only after Tier 2 PASS + supervisor PendingApproval

This breaks HARKing R3 (silent Tier 1 → production promotion path).
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Type alias: Tier 1 mining factor signal_fn ─────────────────────────────
#
# Differs from `factor_library.SignalFn` (ETF Tier 2 takes only as_of and
# returns dict[ticker → weight]) because single-stock walk-forward owns
# the panel + universe — factor signal_fn just emits the cross-section
# z-score given those inputs. This matches existing Wave A pattern in
# engine/factors_singlename/ (tsmom / bab / dividend_yield / value_pe /
# quality_4comp).
#
# Args:
#   as_of:    point-in-time decision date (no look-ahead — uses data ≤ as_of)
#   universe: list of ticker symbols at as_of
#   panel:    pd.DataFrame indexed by date, columns = tickers, values = prices
#
# Returns:
#   pd.Series indexed by ticker, continuous z-score values within universe;
#   NaN for tickers with insufficient data.
SignalFnSinglename = Callable[
    [datetime.date, list[str], pd.DataFrame],
    pd.Series,
]


@dataclass(frozen=True)
class FactorSpecSinglename:
    """Metadata + signal_fn for a Tier 1 mining single-stock candidate.

    Instances are registered into FACTOR_REGISTRY_SINGLENAME below by
    individual factor modules (engine/factors_singlename/<factor_id>.py)
    via `register_factor()`.

    Per memory `feedback_factor_research_3_tier_framework`: Tier 1 mining
    is P-LAB exempt (factor_kind="infrastructure_spec"), so candidates here
    do NOT contribute to EFFECTIVE_N_TRIALS until promoted to Tier 2.
    """
    factor_id:        str         # e.g. "ivol_singlestock", "strev_singlestock"
    citation:         str         # full bibliographic reference
    asset_class:      str         # always "equity_singlename" for this registry
    formula_summary:  str         # one-line algorithmic summary
    signal_fn:        SignalFnSinglename
    expected_sign:    int         # +1 (long high-z = high return per academic prior)
                                  # -1 (long low-z = high return per academic prior, e.g. IVOL)


# ── Registry — populated lazily by individual factor modules ────────────────
FACTOR_REGISTRY_SINGLENAME: dict[str, FactorSpecSinglename] = {}


def register_factor(spec: FactorSpecSinglename) -> None:
    """Register a Tier 1 mining candidate.

    Called by individual factor modules (e.g. engine/factors_singlename/ivol.py)
    at module-load time so importing the registry triggers full population.

    Idempotent: re-registering the same factor_id with an identical spec is
    a no-op (silent). Re-registering with a different spec is an error
    (mining lab integrity — same factor_id should not have two definitions).
    """
    if not isinstance(spec, FactorSpecSinglename):
        raise TypeError(
            f"register_factor: expected FactorSpecSinglename, got {type(spec).__name__}"
        )
    if spec.asset_class != "equity_singlename":
        raise ValueError(
            f"register_factor: asset_class must be 'equity_singlename', "
            f"got {spec.asset_class!r}"
        )
    if spec.expected_sign not in (-1, +1):
        raise ValueError(
            f"register_factor: expected_sign must be -1 or +1, got {spec.expected_sign}"
        )

    existing = FACTOR_REGISTRY_SINGLENAME.get(spec.factor_id)
    if existing is not None:
        if existing == spec:
            return   # idempotent re-register
        raise ValueError(
            f"register_factor: {spec.factor_id!r} already registered with "
            f"different spec. Mining lab integrity requires unique definitions."
        )

    FACTOR_REGISTRY_SINGLENAME[spec.factor_id] = spec
    logger.info("Tier 1 mining lab: registered %s (%s)", spec.factor_id, spec.citation)


def get_factor(factor_id: str) -> FactorSpecSinglename:
    """Look up a registered Tier 1 mining candidate by id.

    Triggers lazy import of all known factor modules so the registry is
    populated before lookup. Failing to find the factor after lazy import
    means it is genuinely not registered (typo or unimplemented).
    """
    _ensure_registry_populated()
    if factor_id not in FACTOR_REGISTRY_SINGLENAME:
        raise KeyError(
            f"factor_id={factor_id!r} not in FACTOR_REGISTRY_SINGLENAME. "
            f"Known: {sorted(FACTOR_REGISTRY_SINGLENAME)}. "
            f"Implement engine/factors_singlename/{factor_id.split('_')[0]}.py "
            f"+ call register_factor() at module load."
        )
    return FACTOR_REGISTRY_SINGLENAME[factor_id]


def list_factors() -> list[str]:
    """Return all registered factor_ids (for CLI / capability evidence)."""
    _ensure_registry_populated()
    return sorted(FACTOR_REGISTRY_SINGLENAME)


# ── Lazy registry population ────────────────────────────────────────────────
# Individual factor modules (engine/factors_singlename/<factor_id>.py) call
# register_factor() at module load. We import the known set lazily here so
# `get_factor` and `list_factors` always see a populated registry.
#
# As of 2026-05-10 the known Tier 1 mining candidates are:
#   - ivol_singlestock    (engine/factors_singlename/ivol.py,   F-LAB-E2)
#   - strev_singlestock   (engine/factors_singlename/strev.py,  F-LAB-E3)
#
# When new candidates are added (e.g. profitability_novy_marx, low_vol_singlestock),
# extend `_KNOWN_FACTOR_MODULES` below.
_KNOWN_FACTOR_MODULES: tuple[str, ...] = (
    "engine.factors_singlename.ivol",
    "engine.factors_singlename.strev",
)

_registry_populated: bool = False


def _ensure_registry_populated() -> None:
    """Lazy import all known factor modules. Idempotent + fail-soft.

    A module that fails to import (e.g. file missing during F-LAB-E2/E3
    incremental development) logs a warning but does not raise — partial
    registry is better than total failure for early-stage scaffolding.
    """
    global _registry_populated
    if _registry_populated:
        return

    import importlib
    for module_path in _KNOWN_FACTOR_MODULES:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(
                "factor_library_singlename: lazy import %s failed: %s "
                "(factor not yet implemented?)",
                module_path, exc,
            )

    _registry_populated = True
