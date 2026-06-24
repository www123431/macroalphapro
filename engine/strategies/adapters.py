"""
engine/strategies/adapters.py — 5 concrete StrategyModule subclasses.

Slice 3 of Week 1 refactor. PURE DELEGATION — no business logic rewritten.
Each adapter:
  - Declares NAME + META at class level (sourced from the legacy
    STRATEGY_DISPLAY_META + STRATEGY_SPEC_MAP literals; values reproduced
    verbatim to guarantee zero behavior change).
  - Implements generate_signal() by calling the existing get_*_signal()
    function in paper_trade_combined.py.
  - Implements is_rebalance_day() by calling the existing
    is_rebalance_day_*() function (or, in the case of AC_TLT_GLD, by
    preserving the pre-existing silent-False behavior — see class docstring).

CIRCULAR IMPORT NOTE:
  paper_trade_combined.py will (in Slice 4) start importing from this module
  via get_registry(). To avoid an import cycle we use METHOD-LEVEL lazy
  imports for the legacy functions. Module-level imports here are limited to
  engine.strategies (base + registry) which have no upstream dependencies.

MODULE-LEVEL SIDE EFFECT:
  At the bottom of this file each strategy class + its Sleeve membership is
  registered into the singleton _REGISTRY exposed by engine.strategies.registry.
  Importing this module is therefore equivalent to "populate the registry".
"""
from __future__ import annotations

import datetime
from typing import Any, TYPE_CHECKING

from engine.strategies.base import StrategyMeta, StrategyModule
from engine.strategies.registry import Sleeve, SleeveClass, get_registry

if TYPE_CHECKING:
    from engine.portfolio.paper_trade_combined import StrategySignal


# ──────────────────────────────────────────────────────────────────────────────
# Strategy metadata — values mirror engine.portfolio.paper_trade_combined
# STRATEGY_DISPLAY_META + engine.portfolio.attribution_logger STRATEGY_SPEC_MAP.
# Two legacy dicts → one source of truth (the META class attribute).
# ──────────────────────────────────────────────────────────────────────────────
_META_K1_BAB = StrategyMeta(
    spec_id               = 61,
    spec_hash_short       = "a0bbcbda",
    sleeve_id             = "etf_l1",
    intra_sleeve_weight   = 1.00,
    rebalance_days        = 30,
    expected_horizon_days = 30,
    label                 = "K1 BAB size-expanded ETF betting-against-beta",
    doctrine              = "ETF L1 cross-sectional rank",
    universe              = "43 ETFs (33 Tier-1 + 10 size/style)",
    color                 = "#4C8EF7",
    display_short         = "K1 BAB",
)

_META_D_PEAD = StrategyMeta(
    spec_id               = 62,
    spec_hash_short       = "c5d9cd09",
    sleeve_id             = "ss_sp500",
    intra_sleeve_weight   = 0.50,
    rebalance_days        = 60,
    expected_horizon_days = 60,
    label                 = "DHS 2020 behavioral 2-factor (PEAD-TS + COMBINED)",
    doctrine              = "single-stock post-earnings drift",
    universe              = "top-1500 NYSE/NASDAQ CRSP point-in-time",
    color                 = "#23D333",
    display_short         = "D-PEAD",
)

_META_PATH_N = StrategyMeta(
    spec_id               = 71,                # was hardcoded 70 (pre-registry doc ref);
                                                # re-registered 2026-05-18 → DB id=71
    spec_hash_short       = "60887180",         # git blob hash unchanged (spec content unchanged)
    sleeve_id             = "ss_sp500",
    intra_sleeve_weight   = 0.50,
    rebalance_days        = 5,
    expected_horizon_days = 5,
    label                 = "Path N S&P 500 index reconstitution pre-effective drift",
    doctrine              = "single-name event-driven (Chen-Noronha-Singal 2004)",
    universe              = "S&P 500 reconstitution events (Wikipedia + EDGAR 8-K)",
    color                 = "#F59E0B",
    display_short         = "Path N",
)

_META_CTA_PQTIX = StrategyMeta(
    spec_id               = 72,                # was hardcoded 73 (pre-registry doc ref);
                                                # re-registered 2026-05-18 → DB id=72
    spec_hash_short       = "9630c2bb",
    sleeve_id             = "cta_defensive",
    intra_sleeve_weight   = 1.00,
    rebalance_days        = 0,    # continuous
    expected_horizon_days = 0,
    label                 = "CTA Defensive Overlay continuous 10% allocation",
    doctrine              = "tail-risk hedge / crisis-positive defensive sleeve",
    universe              = "PQTIX (CTA Defensive Overlay v1)",
    color                 = "#A855F7",
    display_short         = "CTA",
)

_META_AC_TLT_GLD = StrategyMeta(
    spec_id               = 73,                # was hardcoded 77 (pre-registry doc ref);
                                                # re-registered 2026-05-18 → DB id=73
    spec_hash_short       = "4db40176",
    sleeve_id             = "rms_crisis_hedge",
    intra_sleeve_weight   = 1.00,
    rebalance_days        = 30,
    expected_horizon_days = 30,
    label                 = "Path AC TLT/GLD 50/50 RMS insurance (Asness-Israelov 2017)",
    doctrine              = "flight-to-quality + gold safe-haven tail hedge",
    universe              = "TLT (long Treasury) + GLD (gold)",
    color                 = "#EC4899",
    display_short         = "AC TLT/GLD",
)

_META_CARRY_FUTURES = StrategyMeta(
    spec_id               = 77,                # cross-asset carry sleeve (§9 + §10 + §11)
    spec_hash_short       = "1726cf18",
    sleeve_id             = "carry_futures",
    intra_sleeve_weight   = 1.00,
    rebalance_days        = 30,
    expected_horizon_days = 30,
    label                 = "4-leg cross-asset carry (cmdty + FX + US-rates + G10-rates-XC)",
    doctrine              = "KMPV roll-yield premium / Koijen-Moskowitz-Pedersen-Vrugt 2018",
    universe              = "20 commodities + 9 G10+EM FX + 4 UST tenors + 7 G10 sovereigns",
    color                 = "#F59E0B",
    display_short         = "Carry 4-leg",
)


# ──────────────────────────────────────────────────────────────────────────────
# Concrete strategies
# ──────────────────────────────────────────────────────────────────────────────
class K1BabStrategy(StrategyModule):
    """K1 BAB ETF betting-against-beta — Frazzini-Pedersen 2014.

    Delegates to ``engine.portfolio.paper_trade_combined.get_k1_bab_signal``
    and ``is_rebalance_day_k1`` (last-NYSE-trading-day-of-month approximation).
    """
    NAME = "K1_BAB"
    META = _META_K1_BAB

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import get_k1_bab_signal
        return get_k1_bab_signal(as_of)

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        from engine.portfolio.paper_trade_combined import is_rebalance_day_k1
        return is_rebalance_day_k1(as_of)


class DPeadStrategy(StrategyModule):
    """D-PEAD DHS 2020 single-stock behavioral 2-factor.

    Delegates to ``engine.portfolio.paper_trade_combined.get_d_pead_signal``
    and ``is_rebalance_day_d_pead`` (rdq match in panel cache).
    """
    NAME = "D_PEAD"
    META = _META_D_PEAD

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import get_d_pead_signal
        return get_d_pead_signal(as_of)

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        from engine.portfolio.paper_trade_combined import is_rebalance_day_d_pead
        # Forward only the kwargs is_rebalance_day_d_pead actually accepts.
        if "cache_path" in kwargs:
            return is_rebalance_day_d_pead(as_of, cache_path=kwargs["cache_path"])
        return is_rebalance_day_d_pead(as_of)


class PathNStrategy(StrategyModule):
    """Path N S&P 500 reconstitution drift — Chen-Noronha-Singal 2004.

    Delegates to ``engine.portfolio.paper_trade_combined.get_path_n_signal``
    and ``is_rebalance_day_path_n`` (effective_date in 5-day forward window).
    """
    NAME = "PATH_N"
    META = _META_PATH_N

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import get_path_n_signal
        return get_path_n_signal(as_of)

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        from engine.portfolio.paper_trade_combined import is_rebalance_day_path_n
        return is_rebalance_day_path_n(
            as_of,
            msp500_events  = kwargs.get("msp500_events"),
            lookahead_days = kwargs.get("lookahead_days", 5),
        )


class CtaPqtixStrategy(StrategyModule):
    """CTA Defensive PQTIX SAA passive overlay (Path O spec id=73).

    Delegates to ``engine.portfolio.paper_trade_combined.get_cta_pqtix_signal``
    and ``is_rebalance_day_cta`` (Dec 31 OR ±2pp drift from 10% target).
    """
    NAME = "CTA_PQTIX"
    META = _META_CTA_PQTIX

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import get_cta_pqtix_signal
        return get_cta_pqtix_signal(as_of)

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        from engine.portfolio.paper_trade_combined import is_rebalance_day_cta
        return is_rebalance_day_cta(
            as_of,
            current_pqtix_weight = kwargs.get("current_pqtix_weight"),
            target_pqtix_weight  = kwargs.get("target_pqtix_weight", 0.10),
            drift_threshold      = kwargs.get("drift_threshold", 0.02),
        )


class CarryFuturesStrategy(StrategyModule):
    """Cross-asset 4-leg carry sleeve placeholder (spec 77 §10).

    L2 deployment placeholder. Generates NO ticker positions — the carry sleeve
    is currently MARKED at the book NAV level via
    `engine.portfolio.combined_book.build_carry_book()`. Real futures orders
    wait on G1 (IB paper integration, separate spec).

    `generate_signal()` returns a STUB StrategySignal with empty positions.
    `is_rebalance_day()` always False — no order-generation triggers.

    This pattern preserves the registry/UI/sleeve_allocation surface
    (carry visible as 5th sleeve at 30% weight) without writing real paper
    orders. When G1 lands, the placeholder is replaced by a real futures-order
    generator without changing the sleeve registration.

    SAA amendment 2026-05-28 (saa_carry_futures_addition_review_2026-05-28.md).
    """
    NAME = "CARRY_FUTURES"
    META = _META_CARRY_FUTURES

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import StrategySignal
        import pandas as pd
        return StrategySignal(
            strategy_name       = self.NAME,
            sleeve_id           = self.META.sleeve_id,
            intra_sleeve_weight = self.META.intra_sleeve_weight,
            weights             = pd.Series(dtype=float),  # empty -> no ticker orders
            n_positions         = 0,
            status              = "STUB",
            notes               = "L2 placeholder: NAV-marked via combined_book.build_carry_book(); real fills await G1 IB integration",
        )

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        # No order generation ever — sleeve is NAV-marked, not paper-traded.
        return False


class AcTltGldStrategy(StrategyModule):
    """Path AC TLT/GLD 50/50 insurance sleeve (spec id=77).

    Delegates to ``engine.portfolio.paper_trade_combined.get_ac_tlt_gld_signal``.

    Rebalance cadence: end-of-month, matching K1's last-NYSE-trading-day
    approximation. AC TLT/GLD spec declares rebalance_days=30, so reusing
    K1's EOM helper is the correct delegation.

    2026-05-18 fix: previously returned False unconditionally to preserve a
    pre-existing dispatcher bug (the legacy ``is_rebalance_day`` dispatcher
    raised ValueError on "AC_TLT_GLD" and the caller's try/except swallowed
    it to False). The Sprint H attribution log has been recording
    ``is_rebalance_day=False`` for every AC_TLT_GLD row since 2026-05-15;
    forensic replays built on that log will misread AC's rebalance history.
    See [[project-week1-refactor-status-2026-05-18]] followup #1.
    """
    NAME = "AC_TLT_GLD"
    META = _META_AC_TLT_GLD

    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        from engine.portfolio.paper_trade_combined import get_ac_tlt_gld_signal
        return get_ac_tlt_gld_signal(as_of)

    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        # K1's EOM helper matches AC's spec_30d cadence (last NYSE day of month).
        from engine.portfolio.paper_trade_combined import is_rebalance_day_k1
        return is_rebalance_day_k1(as_of)


# ──────────────────────────────────────────────────────────────────────────────
# Module-import side-effect: populate the registry singleton.
# Sleeve target_weights mirror the legacy PAPER_TRADE_SLEEVE_ALLOCATION dict.
# ──────────────────────────────────────────────────────────────────────────────
def _populate_registry() -> None:
    reg = get_registry()
    # Skip if already populated (idempotent on re-import; protects test reuse).
    if len(reg) > 0:
        return

    # Register all 6 strategies (5 LIVE + 1 NAV-only carry placeholder).
    reg.register(K1BabStrategy())
    reg.register(DPeadStrategy())
    reg.register(PathNStrategy())
    reg.register(CtaPqtixStrategy())
    reg.register(AcTltGldStrategy())
    reg.register(CarryFuturesStrategy())  # SAA amendment 2026-05-28; NAV-only placeholder

    # Register all 5 sleeves. SAA amendment 2026-05-28 (saa_carry_futures_addition_
    # review_2026-05-28.md): 4 existing sleeves × 0.7 + carry_futures @ 0.3 = 1.0.
    # Tier-3 governed at paper_trade_combined.py:60-65 (do not change without
    # corresponding spec amendment + SAA review doc).
    # sleeve_class enables Risk Manager sleeve-class-aware caps (spec id=69 §2.1a).
    reg.register_sleeve(Sleeve(
        sleeve_id      = "etf_l1",
        display_name   = "ETF L1 cross-sectional",
        target_weight  = 0.2268,   # was 0.324 (× 0.7 SAA 2026-05-28)
        strategy_names = ("K1_BAB",),
        sleeve_class   = SleeveClass.ALPHA_EQUITY_LS,
    ))
    reg.register_sleeve(Sleeve(
        sleeve_id      = "ss_sp500",
        display_name   = "Single-Stock S&P 500",
        target_weight  = 0.2952,   # was 0.3402 (Spec 79 L2.4 2026-05-28: -4.50pp shifted to rms_crisis_hedge)
        strategy_names = ("D_PEAD", "PATH_N"),
        sleeve_class   = SleeveClass.ALPHA_SINGLE_STOCK,
    ))
    reg.register_sleeve(Sleeve(
        sleeve_id      = "cta_defensive",
        display_name   = "CTA Defensive",
        target_weight  = 0.0743,   # Spec 80 2026-05-28: 0.0630 + 1.13pp from rms_crisis_hedge (mechanism diversification per H-O-P Crisis Alpha)
        strategy_names = ("CTA_PQTIX",),
        sleeve_class   = SleeveClass.CTA_OVERLAY,
    ))
    reg.register_sleeve(Sleeve(
        sleeve_id      = "rms_crisis_hedge",
        display_name   = "RMS Crisis Hedge",
        target_weight  = 0.1037,   # Spec 80 2026-05-28: 0.1150 - 1.13pp to cta_defensive (75/25 hedge mechanism split)
        strategy_names = ("AC_TLT_GLD",),
        sleeve_class   = SleeveClass.INSURANCE,
    ))
    reg.register_sleeve(Sleeve(
        sleeve_id      = "carry_futures",
        display_name   = "Cross-asset Carry (4-leg)",
        target_weight  = 0.300,    # NEW SAA 2026-05-28; spec 77 §10 4-leg combined GREEN
        strategy_names = ("CARRY_FUTURES",),
        sleeve_class   = SleeveClass.CARRY_FUTURES,
    ))

    # Cross-cutting invariants. Raises if any of:
    #   - strategy declares unknown sleeve_id
    #   - sleeve has zero strategies
    #   - sleeve weights do not sum to 1.0
    #   - intra-sleeve weights within any sleeve do not sum to 1.0
    reg.validate()


# Side effect on import — but bypassable in tests via the
# engine.strategies.registry._reset_registry_for_tests() helper.
_populate_registry()
