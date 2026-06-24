"""tests/test_strategy_meta_locked.py — spec-hash lockdown.

Code-level enforcement of [[feedback-spec-lock-is-decision-contract-2026-05-15]]:
each production strategy's (spec_id, spec_hash_short, sleeve_id, rebalance_days,
expected_horizon_days) is HARDCODED in this file. Any drift in adapters.py
META will fail the test.

Purpose
-------
The Engineer agent (Level 2.5) is permitted to edit engine/strategies/adapters.py
directly. But a silent change to ``META.spec_hash_short`` would let a "tweaked"
strategy claim the same identity as a previously locked spec — pure HARKing.
This test fails BEFORE such a change can be committed: the agent's pytest run
will see the contract violated and either self-correct or surface the diff
to the user for explicit Tier-3 governance approval.

If a strategy's spec is LEGITIMATELY amended (spec registry shows a new
hash via an explicit amendment row), this test MUST be edited in the same
commit as the META change. That edit IS the audit-chain link: a reviewer
sees both sides of the contract update in one diff.

DO NOT update these literals without:
  1. A corresponding spec_metadata amendment row
  2. A docs/decisions/*.md governance entry citing the trigger
  3. A new STRATEGY_HASH_GOVERNANCE_LOG entry below
"""
from __future__ import annotations

import pytest

from engine.strategies import get_registry


# ──────────────────────────────────────────────────────────────────────────────
# LOCKED IDENTITY TABLE — each row is (spec_id, hash, sleeve, rebal_d, horizon_d)
# Sourced from spec_registry table at the time of Week-1 refactor commit
# 8742a10 (2026-05-18). Verified against legacy STRATEGY_DISPLAY_META +
# STRATEGY_SPEC_MAP literals at refactor time.
# ──────────────────────────────────────────────────────────────────────────────
LOCKED_META: dict[str, dict] = {
    "K1_BAB": {
        "spec_id":               61,
        "spec_hash_short":       "a0bbcbda",
        "sleeve_id":             "etf_l1",
        "intra_sleeve_weight":   1.00,
        "rebalance_days":        30,
        "expected_horizon_days": 30,
    },
    "D_PEAD": {
        "spec_id":               62,
        "spec_hash_short":       "c5d9cd09",
        "sleeve_id":             "ss_sp500",
        "intra_sleeve_weight":   0.50,
        "rebalance_days":        60,
        "expected_horizon_days": 60,
    },
    "PATH_N": {
        "spec_id":               71,        # re-registered 2026-05-18 (was 70 doc ref)
        "spec_hash_short":       "60887180",
        "sleeve_id":             "ss_sp500",
        "intra_sleeve_weight":   0.50,
        "rebalance_days":        5,
        "expected_horizon_days": 5,
    },
    "CTA_PQTIX": {
        "spec_id":               72,        # re-registered 2026-05-18 (was 73 doc ref)
        "spec_hash_short":       "9630c2bb",
        "sleeve_id":             "cta_defensive",
        "intra_sleeve_weight":   1.00,
        "rebalance_days":        0,
        "expected_horizon_days": 0,
    },
    "AC_TLT_GLD": {
        "spec_id":               73,        # re-registered 2026-05-18 (was 77 doc ref)
        "spec_hash_short":       "4db40176",
        "sleeve_id":             "rms_crisis_hedge",
        "intra_sleeve_weight":   1.00,
        "rebalance_days":        30,
        "expected_horizon_days": 30,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Governance log — append a new line every time LOCKED_META is intentionally
# changed. Reviewers cross-reference this with the spec_metadata amendment row.
# ──────────────────────────────────────────────────────────────────────────────
STRATEGY_HASH_GOVERNANCE_LOG = [
    "2026-05-18 initial lockdown at commit 8742a10 (Week-1 refactor); "
    "values verified byte-identical to legacy paper_trade_combined.py "
    "STRATEGY_DISPLAY_META + attribution_logger.py STRATEGY_SPEC_MAP",
    "2026-05-18 evening (DQ Inspector spec_id collision discovery): "
    "register_spec(PATH_N + CTA_PQTIX + AC_TLT_GLD) ran for the first "
    "time. DB ids assigned were 71 / 72 / 73 respectively (PATH_N was "
    "previously hardcoded 70 — collided with newly-registered DQ "
    "Inspector spec id=70; CTA/AC hardcoded 73/77 were pre-DB-registry "
    "documentary refs). Hashes verified byte-identical to git blob "
    "(60887180 / 9630c2bb / 4db40176 unchanged). spec_ids updated in "
    "adapters.py + LOCKED_META; no spec content modified.",
]


# ──────────────────────────────────────────────────────────────────────────────
# LOCKED SLEEVE TABLE — Phase 0 of Risk Manager spec id=69 (current hash in SpecRegistry).
# sleeve_class drives sleeve-class-aware single-ticker caps in Mode 1 detector.
# Any drift here implicitly changes risk-management semantics and must be
# accompanied by a SLEEVE_CLASS_GOVERNANCE_LOG entry citing the spec amendment.
# ──────────────────────────────────────────────────────────────────────────────
LOCKED_SLEEVES: dict[str, dict] = {
    "etf_l1": {
        "target_weight":  0.324,
        "sleeve_class":   "alpha_equity_ls",
        "strategy_names": ("K1_BAB",),
    },
    "ss_sp500": {
        "target_weight":  0.486,
        "sleeve_class":   "alpha_single_stock",
        "strategy_names": ("D_PEAD", "PATH_N"),
    },
    "cta_defensive": {
        "target_weight":  0.090,
        "sleeve_class":   "cta_overlay",
        "strategy_names": ("CTA_PQTIX",),
    },
    "rms_crisis_hedge": {
        "target_weight":  0.100,
        "sleeve_class":   "insurance",
        "strategy_names": ("AC_TLT_GLD",),
    },
}

SLEEVE_CLASS_GOVERNANCE_LOG = [
    "2026-05-18 initial lockdown — Phase 0 of Risk Manager spec id=69 hash f763a717; "
    "sleeve_class assigned per spec §2.1a (alpha_equity_ls / alpha_single_stock / "
    "cta_overlay / insurance) at commit (this commit)",
]


@pytest.mark.parametrize("sleeve_id", list(LOCKED_SLEEVES))
def test_sleeve_identity_locked(sleeve_id: str):
    """Each production sleeve's (target_weight, sleeve_class, strategy_names)
    tuple is hash-locked at commit time.

    Failure mode: an Engineer agent / human edit changed engine/strategies/
    adapters.py Sleeve construction without updating this test. Resolve by
    either reverting the edit (if accidental) or by amending LOCKED_SLEEVES
    + appending to SLEEVE_CLASS_GOVERNANCE_LOG with the Tier-3 amendment
    citation.
    """
    sleeve = get_registry().get_sleeve(sleeve_id)
    locked = LOCKED_SLEEVES[sleeve_id]

    failures: list[str] = []
    if abs(sleeve.target_weight - locked["target_weight"]) > 1e-9:
        failures.append(
            f"  target_weight: locked={locked['target_weight']!r}, "
            f"actual={sleeve.target_weight!r}"
        )
    if sleeve.sleeve_class.value != locked["sleeve_class"]:
        failures.append(
            f"  sleeve_class: locked={locked['sleeve_class']!r}, "
            f"actual={sleeve.sleeve_class.value!r}"
        )
    if tuple(sleeve.strategy_names) != tuple(locked["strategy_names"]):
        failures.append(
            f"  strategy_names: locked={locked['strategy_names']!r}, "
            f"actual={tuple(sleeve.strategy_names)!r}"
        )

    assert not failures, (
        f"Sleeve {sleeve_id!r} drifted from locked values:\n"
        + "\n".join(failures)
        + "\n\nTo resolve: update LOCKED_SLEEVES + append a "
          "SLEEVE_CLASS_GOVERNANCE_LOG entry citing the spec amendment row."
    )


def test_no_extra_sleeves_in_registry():
    """Registry must not contain sleeves absent from LOCKED_SLEEVES."""
    registered = {s.sleeve_id for s in get_registry().sleeves()}
    locked     = set(LOCKED_SLEEVES.keys())
    new_sleeves = registered - locked
    assert not new_sleeves, (
        f"Registry contains sleeves not in LOCKED_SLEEVES: {sorted(new_sleeves)}.\n"
        f"Add a LOCKED_SLEEVES entry for each + a SLEEVE_CLASS_GOVERNANCE_LOG line."
    )


def test_sleeve_class_governance_log_non_empty():
    assert len(SLEEVE_CLASS_GOVERNANCE_LOG) >= 1
    assert all(isinstance(line, str) and len(line) > 20
               for line in SLEEVE_CLASS_GOVERNANCE_LOG)


@pytest.mark.parametrize("strategy_name", list(LOCKED_META))
def test_strategy_meta_identity_locked(strategy_name: str):
    """Each production strategy's identity tuple is hash-locked at commit time.

    Failure mode: an Engineer agent / human edit changed engine/strategies/
    adapters.py META without updating this test. Resolve by either reverting
    the META edit (if accidental) or by amending LOCKED_META in this file +
    appending to STRATEGY_HASH_GOVERNANCE_LOG (if intentional Tier-3-approved
    amendment, citing the spec_metadata row).
    """
    strat = get_registry().get(strategy_name)
    locked = LOCKED_META[strategy_name]

    failures: list[str] = []
    for field, expected in locked.items():
        actual = getattr(strat.META, field)
        if actual != expected:
            failures.append(f"  {field}: locked={expected!r}, actual={actual!r}")

    assert not failures, (
        f"Strategy {strategy_name!r} META drifted from locked values:\n"
        + "\n".join(failures)
        + "\n\nTo resolve: update LOCKED_META in this file + append a "
          "STRATEGY_HASH_GOVERNANCE_LOG entry citing the spec amendment row."
    )


def test_no_extra_strategies_in_registry():
    """The registry must not contain strategies absent from LOCKED_META.

    A new strategy being added to adapters.py without being added here is a
    HARKing escape hatch — someone could ship a strategy whose identity has
    never been reviewed by the lockdown test. Force the issue.
    """
    registered = set(get_registry().names())
    locked     = set(LOCKED_META.keys())
    new_strategies = registered - locked
    assert not new_strategies, (
        f"Registry contains strategies not in LOCKED_META: {sorted(new_strategies)}.\n"
        f"Add a LOCKED_META entry for each + a STRATEGY_HASH_GOVERNANCE_LOG line."
    )


def test_no_orphan_locked_strategies():
    """LOCKED_META must not contain strategies absent from the registry.

    Failure mode: a strategy was removed from adapters.py without removing
    its lockdown entry — stale lock that can never fail. Catch it.
    """
    registered = set(get_registry().names())
    locked     = set(LOCKED_META.keys())
    orphans = locked - registered
    assert not orphans, (
        f"LOCKED_META has entries for strategies no longer in registry: "
        f"{sorted(orphans)}. Remove the LOCKED_META entries + log the removal."
    )


def test_governance_log_non_empty():
    """At least one governance entry must exist (sanity guard against an
    empty log being committed)."""
    assert len(STRATEGY_HASH_GOVERNANCE_LOG) >= 1
    assert all(isinstance(line, str) and len(line) > 20
               for line in STRATEGY_HASH_GOVERNANCE_LOG)
