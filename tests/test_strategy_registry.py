"""tests/test_strategy_registry.py — contract tests for engine.strategies.

Slice 6 of Week 1 agent constellation refactor. Verifies:
  1. StrategyModule ABC enforces NAME + META class attrs at subclass time
  2. StrategyMeta dataclass invariants raise at construction
  3. Sleeve dataclass invariants raise at construction
  4. StrategyRegistry rejects duplicate registrations, unknown sleeves,
     and unbalanced sleeve / intra-sleeve weights
  5. The populated production registry matches the legacy literals
     byte-for-byte (display_meta_dict / spec_map_dict / sleeve_allocation_dict)

NO LLM CALLS, NO NETWORK — these are pure structural assertions runnable
in any environment.
"""
from __future__ import annotations

import datetime

import pytest

from engine.strategies import (
    ALLOWED_SLEEVES,
    Sleeve,
    SleeveClass,
    StrategyMeta,
    StrategyModule,
    StrategyRegistry,
    get_registry,
)


# Throwaway sleeve_class for tests that don't care about the class semantics —
# Phase 0 of Risk Manager (spec id=69) made the field required.
_TEST_CLASS = SleeveClass.ALPHA_EQUITY_LS


# ─── helper: build a valid throwaway META + minimal subclass ────────────────
_GOOD_META = StrategyMeta(
    spec_id               = 999,
    spec_hash_short       = "deadbeef",
    sleeve_id             = "etf_l1",
    intra_sleeve_weight   = 1.0,
    rebalance_days        = 30,
    expected_horizon_days = 30,
    label                 = "test strat",
    doctrine              = "test doctrine",
    universe              = "test universe",
    color                 = "#000000",
    display_short         = "TEST",
)


def _make_test_strategy(name: str = "TEST_STRAT", meta: StrategyMeta = _GOOD_META):
    class _T(StrategyModule):
        NAME = name
        META = meta
        def generate_signal(self, as_of):     return None
        def is_rebalance_day(self, as_of, **k): return False
    return _T()


# ─── 1. StrategyMeta invariants ─────────────────────────────────────────────
class TestStrategyMeta:
    def test_intra_weight_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="intra_sleeve_weight"):
            StrategyMeta(
                spec_id=1, spec_hash_short="abcdef12", sleeve_id="etf_l1",
                intra_sleeve_weight=1.5,  # > 1.0 invalid
                rebalance_days=30, expected_horizon_days=30,
                label="", doctrine="", universe="", color="", display_short="",
            )

    def test_negative_rebalance_days_rejected(self):
        with pytest.raises(ValueError, match="rebalance_days"):
            StrategyMeta(
                spec_id=1, spec_hash_short="abcdef12", sleeve_id="etf_l1",
                intra_sleeve_weight=1.0,
                rebalance_days=-1,  # negative invalid
                expected_horizon_days=30,
                label="", doctrine="", universe="", color="", display_short="",
            )

    def test_short_hash_rejected(self):
        with pytest.raises(ValueError, match="spec_hash_short"):
            StrategyMeta(
                spec_id=1, spec_hash_short="abc",  # too short
                sleeve_id="etf_l1", intra_sleeve_weight=1.0,
                rebalance_days=30, expected_horizon_days=30,
                label="", doctrine="", universe="", color="", display_short="",
            )

    def test_meta_is_frozen(self):
        m = _GOOD_META
        with pytest.raises(Exception):  # FrozenInstanceError / AttributeError
            m.spec_id = 0  # type: ignore[misc]


# ─── 2. StrategyModule subclass contract ────────────────────────────────────
class TestStrategyModuleSubclassContract:
    def test_abstract_class_cannot_instantiate(self):
        with pytest.raises(TypeError):
            StrategyModule()  # type: ignore[abstract]

    def test_subclass_missing_NAME_rejected(self):
        with pytest.raises(TypeError, match="NAME"):
            class _Bad(StrategyModule):
                META = _GOOD_META
                def generate_signal(self, as_of):     return None
                def is_rebalance_day(self, as_of, **k): return False

    def test_subclass_missing_META_rejected(self):
        with pytest.raises(TypeError, match="META"):
            class _Bad(StrategyModule):
                NAME = "BAD"
                def generate_signal(self, as_of):     return None
                def is_rebalance_day(self, as_of, **k): return False

    def test_subclass_META_wrong_type_rejected(self):
        with pytest.raises(TypeError, match="StrategyMeta instance"):
            class _Bad(StrategyModule):
                NAME = "BAD"
                META = "not a meta"  # type: ignore[assignment]
                def generate_signal(self, as_of):     return None
                def is_rebalance_day(self, as_of, **k): return False

    def test_subclass_lowercase_NAME_rejected(self):
        with pytest.raises(ValueError, match="UPPER_SNAKE_CASE"):
            class _Bad(StrategyModule):
                NAME = "lowercase_bad"
                META = _GOOD_META
                def generate_signal(self, as_of):     return None
                def is_rebalance_day(self, as_of, **k): return False

    def test_book_weight_uses_sleeve_and_leverage(self):
        s = _make_test_strategy()  # sleeve='etf_l1' intra=1.0
        w = s.book_weight({"etf_l1": 0.324}, leverage=1.5)
        assert abs(w - 0.486) < 1e-9

    def test_book_weight_unknown_sleeve_raises(self):
        s = _make_test_strategy()
        with pytest.raises(KeyError, match="etf_l1"):
            s.book_weight({"other_sleeve": 0.5})


# ─── 3. Sleeve invariants ───────────────────────────────────────────────────
class TestSleeve:
    def test_unknown_sleeve_id_rejected(self):
        with pytest.raises(ValueError, match="ALLOWED_SLEEVES"):
            Sleeve(sleeve_id="not_a_real_sleeve", display_name="X",
                   target_weight=0.5, strategy_names=(), sleeve_class=_TEST_CLASS)

    def test_negative_target_weight_rejected(self):
        with pytest.raises(ValueError, match="target_weight"):
            Sleeve(sleeve_id="etf_l1", display_name="X",
                   target_weight=-0.1, strategy_names=(), sleeve_class=_TEST_CLASS)

    def test_target_weight_above_one_rejected(self):
        with pytest.raises(ValueError, match="target_weight"):
            Sleeve(sleeve_id="etf_l1", display_name="X",
                   target_weight=1.01, strategy_names=(), sleeve_class=_TEST_CLASS)

    def test_non_sleeveclass_sleeve_class_rejected(self):
        """Phase 0 invariant: sleeve_class must be a SleeveClass instance, not str."""
        with pytest.raises(TypeError, match="SleeveClass instance"):
            Sleeve(sleeve_id="etf_l1", display_name="X",
                   target_weight=0.5, strategy_names=(),
                   sleeve_class="alpha_equity_ls")  # type: ignore[arg-type]


# ─── 4. StrategyRegistry mechanics ──────────────────────────────────────────
class TestStrategyRegistry:
    def test_duplicate_strategy_rejected(self):
        reg = StrategyRegistry()
        reg.register(_make_test_strategy("DUP_STRAT"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_make_test_strategy("DUP_STRAT"))

    def test_get_unknown_raises_keyerror(self):
        reg = StrategyRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("NONEXISTENT")

    def test_validate_catches_sleeve_weight_mismatch(self):
        reg = StrategyRegistry()
        reg.register(_make_test_strategy("S1"))
        reg.register_sleeve(Sleeve(
            sleeve_id="etf_l1", display_name="X",
            target_weight=0.5, strategy_names=("S1",),
            sleeve_class=_TEST_CLASS,
        ))
        # Only one sleeve at 0.5; total != 1.0
        with pytest.raises(ValueError, match="target_weights sum to"):
            reg.validate()

    def test_validate_catches_intra_sleeve_mismatch(self):
        # Two strategies in one sleeve with intra weights summing to 0.7
        bad_meta_a = StrategyMeta(
            spec_id=1, spec_hash_short="aaaaaaaa", sleeve_id="etf_l1",
            intra_sleeve_weight=0.4, rebalance_days=0, expected_horizon_days=0,
            label="", doctrine="", universe="", color="", display_short="",
        )
        bad_meta_b = StrategyMeta(
            spec_id=2, spec_hash_short="bbbbbbbb", sleeve_id="etf_l1",
            intra_sleeve_weight=0.3, rebalance_days=0, expected_horizon_days=0,
            label="", doctrine="", universe="", color="", display_short="",
        )
        reg = StrategyRegistry()
        reg.register(_make_test_strategy("A", bad_meta_a))
        reg.register(_make_test_strategy("B", bad_meta_b))
        reg.register_sleeve(Sleeve(
            sleeve_id="etf_l1", display_name="X",
            target_weight=1.0, strategy_names=("A", "B"),
            sleeve_class=_TEST_CLASS,
        ))
        with pytest.raises(ValueError, match="intra_sleeve_weights"):
            reg.validate()

    def test_validate_catches_unregistered_sleeve(self):
        reg = StrategyRegistry()
        reg.register(_make_test_strategy("X"))   # claims sleeve_id='etf_l1'
        # No sleeve registered — strategy points at non-existent sleeve
        with pytest.raises(ValueError, match="unregistered sleeve"):
            reg.validate()


# ─── 5. Populated production registry parity ────────────────────────────────
class TestProductionRegistry:
    """Verify the singleton registry (populated by adapters import) matches
    every legacy literal byte-for-byte. If a future change to adapters drifts
    from any of these, the test fails — registry IS the source of truth."""

    def test_six_strategies_in_canonical_order(self):
        # SAA amendment 2026-05-28: CARRY_FUTURES added as 6th strategy (NAV-only placeholder).
        reg = get_registry()
        assert reg.names() == (
            "K1_BAB", "D_PEAD", "PATH_N", "CTA_PQTIX", "AC_TLT_GLD", "CARRY_FUTURES",
        )

    def test_five_sleeves_registered(self):
        # SAA amendment 2026-05-28: carry_futures added as 5th sleeve.
        reg = get_registry()
        sleeve_ids = {sl.sleeve_id for sl in reg.sleeves()}
        assert sleeve_ids == {
            "etf_l1", "ss_sp500", "cta_defensive", "rms_crisis_hedge", "carry_futures",
        }

    def test_sleeve_allocation_sums_to_one(self):
        reg = get_registry()
        total = sum(sl.target_weight for sl in reg.sleeves())
        assert abs(total - 1.0) < 1e-9

    def test_intra_sleeve_weights_sum_to_one_per_sleeve(self):
        reg = get_registry()
        for sl in reg.sleeves():
            members = reg.strategies_for_sleeve(sl.sleeve_id)
            intra_sum = sum(s.META.intra_sleeve_weight for s in members)
            assert abs(intra_sum - 1.0) < 1e-9, (
                f"sleeve {sl.sleeve_id!r} intra weights sum to {intra_sum}"
            )

    def test_display_meta_matches_legacy_shim(self):
        from engine.portfolio.paper_trade_combined import STRATEGY_DISPLAY_META
        assert get_registry().display_meta_dict() == STRATEGY_DISPLAY_META

    def test_spec_map_matches_legacy_shim(self):
        from engine.portfolio.attribution_logger import STRATEGY_SPEC_MAP
        assert get_registry().spec_map_dict() == STRATEGY_SPEC_MAP

    def test_sleeve_allocation_matches_legacy_shim(self):
        from engine.portfolio.paper_trade_combined import PAPER_TRADE_SLEEVE_ALLOCATION
        assert get_registry().sleeve_allocation_dict() == PAPER_TRADE_SLEEVE_ALLOCATION

    def test_allowed_sleeves_consistent_with_portfolio_sleeves(self):
        """ALLOWED_SLEEVES still duplicated in engine.portfolio_sleeves until
        Slice 7 flips the dependency. This test enforces they stay in sync."""
        from engine.portfolio_sleeves import ALLOWED_SLEEVES as PS_ALLOWED
        assert ALLOWED_SLEEVES == PS_ALLOWED
