"""
engine/strategies/registry.py — Sleeve dataclass + StrategyRegistry singleton.

Single source of truth that orchestrator + UI pages iterate over instead of
hardcoding strategy names or importing the legacy ``STRATEGY_DISPLAY_META`` /
``STRATEGY_SPEC_MAP`` dicts.

The registry is populated lazily on first access by importing
``engine.strategies.adapters``, which registers each concrete StrategyModule
subclass.

DOCTRINE compliance:
  - Spec-lock: ``Sleeve.target_weight`` is frozen at class definition. Mutating
    a sleeve's weight requires editing the registry source file (caught by
    Devil's Advocate diff review in agent constellation).
  - Allowlist: ``ALLOWED_SLEEVES`` is the single source of truth.
    engine.portfolio_sleeves re-exports this symbol (Slice 7 flip 2026-05-18).
"""
from __future__ import annotations

import dataclasses
import enum
import logging
from typing import Iterator

from engine.strategies.base import StrategyModule

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sleeve allowlist — single source of truth (engine.portfolio_sleeves re-exports).
# ──────────────────────────────────────────────────────────────────────────────
ALLOWED_SLEEVES: frozenset[str] = frozenset({
    "etf_l1", "ss_sp500", "cta_defensive", "rms_crisis_hedge",
    "carry_futures",  # spec 77 §10 4-leg cross-asset carry (SAA amend 2026-05-28)
})


class SleeveClass(str, enum.Enum):
    """Capital-allocation sleeve categories — used by Risk Manager for
    sleeve-class-aware single-ticker caps (spec id=69 §2.1a).

    Markowitz 5% diversification heuristic only applies to multi-name equity
    sleeves. Single-instrument funds (CTA) and passive insurance (TLT/GLD)
    require different cap calibration; this Enum makes the class explicit.

    Class-to-cap mapping lives in engine.agents.risk_manager.thresholds
    (loaded when Risk Manager scaffold lands in Phase 1).
    """
    ALPHA_EQUITY_LS    = "alpha_equity_ls"      # diversified long/short ETF / single-stock book
    ALPHA_SINGLE_STOCK = "alpha_single_stock"   # 100+ name single-stock universe
    INSURANCE          = "insurance"             # passive insurance hedges (TLT / GLD)
    CTA_OVERLAY        = "cta_overlay"           # single-instrument managed-futures fund
    CARRY_FUTURES      = "carry_futures"         # cross-asset carry sleeve (spec 77; futures cmdty + FX + rates)


@dataclasses.dataclass(frozen=True)
class Sleeve:
    """Capital-allocation sleeve grouping one or more strategies.

    ``target_weight`` is the paper-trade allocation (sum across sleeves must
    equal 1.0). ``strategy_names`` references registered StrategyModule.NAME
    values — resolution to actual instances happens via the registry.

    ``sleeve_class`` (Phase 0 of Risk Manager spec id=69, 2026-05-18) categorises
    the sleeve for sleeve-class-aware risk gates. Required field — there is no
    sensible default because the choice has real risk-management consequences.

    Held separately from the real-capital DEFAULT_INITIAL_ALLOCATION in
    engine.portfolio_sleeves (which remains Tier-3-governed).
    """
    sleeve_id:        str
    display_name:     str
    target_weight:    float
    strategy_names:   tuple[str, ...]
    sleeve_class:     SleeveClass

    def __post_init__(self) -> None:
        if self.sleeve_id not in ALLOWED_SLEEVES:
            raise ValueError(
                f"sleeve_id {self.sleeve_id!r} not in ALLOWED_SLEEVES "
                f"({sorted(ALLOWED_SLEEVES)})"
            )
        if not (0.0 <= self.target_weight <= 1.0):
            raise ValueError(
                f"target_weight must be in [0,1], got {self.target_weight}"
            )
        if not isinstance(self.sleeve_class, SleeveClass):
            raise TypeError(
                f"sleeve_class must be a SleeveClass instance, "
                f"got {type(self.sleeve_class).__name__}"
            )


class StrategyRegistry:
    """Ordered registry of StrategyModule instances.

    Insertion order is preserved (matches the canonical display order in the
    legacy ``STRATEGY_ORDER`` list).
    """

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyModule] = {}
        self._sleeves:    dict[str, Sleeve]         = {}

    # ── strategy registration ────────────────────────────────────────────────
    def register(self, strategy: StrategyModule) -> None:
        name = strategy.NAME
        if name in self._strategies:
            raise ValueError(
                f"strategy {name!r} already registered "
                f"(existing: {type(self._strategies[name]).__name__}, "
                f"incoming: {type(strategy).__name__})"
            )
        if strategy.META.sleeve_id not in ALLOWED_SLEEVES:
            raise ValueError(
                f"strategy {name!r} declares sleeve_id "
                f"{strategy.META.sleeve_id!r} not in ALLOWED_SLEEVES"
            )
        self._strategies[name] = strategy
        logger.debug("registered strategy %s -> sleeve %s", name, strategy.META.sleeve_id)

    def get(self, name: str) -> StrategyModule:
        try:
            return self._strategies[name]
        except KeyError as e:
            raise KeyError(
                f"strategy {name!r} not registered (known: {sorted(self._strategies)})"
            ) from e

    def names(self) -> tuple[str, ...]:
        return tuple(self._strategies)

    def __iter__(self) -> Iterator[StrategyModule]:
        return iter(self._strategies.values())

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._strategies

    # ── sleeve registration ──────────────────────────────────────────────────
    def register_sleeve(self, sleeve: Sleeve) -> None:
        if sleeve.sleeve_id in self._sleeves:
            raise ValueError(f"sleeve {sleeve.sleeve_id!r} already registered")
        self._sleeves[sleeve.sleeve_id] = sleeve

    def get_sleeve(self, sleeve_id: str) -> Sleeve:
        try:
            return self._sleeves[sleeve_id]
        except KeyError as e:
            raise KeyError(
                f"sleeve {sleeve_id!r} not registered "
                f"(known: {sorted(self._sleeves)})"
            ) from e

    def sleeves(self) -> tuple[Sleeve, ...]:
        return tuple(self._sleeves.values())

    def strategies_for_sleeve(self, sleeve_id: str) -> tuple[StrategyModule, ...]:
        return tuple(
            s for s in self._strategies.values() if s.META.sleeve_id == sleeve_id
        )

    # ── consistency checks ───────────────────────────────────────────────────
    def validate(self) -> None:
        """Run cross-cutting invariants. Called after population.

        Invariants:
          1. Every registered sleeve has at least one strategy.
          2. Every strategy's sleeve_id has a registered Sleeve.
          3. Sleeve target_weights sum to ~1.0 (1e-6 tolerance).
          4. Intra-sleeve weights of strategies belonging to one sleeve sum
             to ~1.0 (1e-6 tolerance).
        """
        # 2 — strategy sleeve_id must be registered
        for s in self._strategies.values():
            if s.META.sleeve_id not in self._sleeves:
                raise ValueError(
                    f"strategy {s.NAME!r} references unregistered sleeve "
                    f"{s.META.sleeve_id!r}"
                )

        # 1 — every sleeve has at least one strategy
        for sleeve_id in self._sleeves:
            members = self.strategies_for_sleeve(sleeve_id)
            if not members:
                raise ValueError(f"sleeve {sleeve_id!r} has zero registered strategies")

        # 3 — sleeve weights sum to 1
        total_w = sum(sl.target_weight for sl in self._sleeves.values())
        if abs(total_w - 1.0) > 1e-6:
            raise ValueError(
                f"sleeve target_weights sum to {total_w}, expected 1.0"
            )

        # 4 — intra-sleeve weights sum to 1 per sleeve
        for sleeve_id in self._sleeves:
            members = self.strategies_for_sleeve(sleeve_id)
            intra_sum = sum(s.META.intra_sleeve_weight for s in members)
            if abs(intra_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"intra_sleeve_weights for sleeve {sleeve_id!r} sum to "
                    f"{intra_sum}, expected 1.0 (members: "
                    f"{[s.NAME for s in members]})"
                )

    # ── derived views consumed by UI / orchestrator ─────────────────────────
    def sleeve_allocation_dict(self) -> dict[str, float]:
        """Equivalent of legacy ``PAPER_TRADE_SLEEVE_ALLOCATION``."""
        return {sl.sleeve_id: sl.target_weight for sl in self._sleeves.values()}

    def display_meta_dict(self) -> dict[str, dict]:
        """Equivalent of legacy ``STRATEGY_DISPLAY_META``.

        Provided as a compatibility shim so UI pages can migrate incrementally
        — they read the dict the same way they do today, but the data flows
        from the registry instead of a hardcoded literal.
        """
        return {
            s.NAME: {
                "spec_id":         s.META.spec_id,
                "spec_hash_short": s.META.spec_hash_short,
                "rebalance_days":  s.META.rebalance_days,
                "sleeve_id":       s.META.sleeve_id,
                "intra_sleeve_w":  s.META.intra_sleeve_weight,
                "label":           s.META.label,
                "doctrine":        s.META.doctrine,
                "universe":        s.META.universe,
                "color":           s.META.color,
                "display_short":   s.META.display_short,
            }
            for s in self._strategies.values()
        }

    def spec_map_dict(self) -> dict[str, tuple[int, str, int]]:
        """Equivalent of legacy ``STRATEGY_SPEC_MAP`` in attribution_logger.py."""
        return {
            s.NAME: (s.META.spec_id, s.META.spec_hash_short, s.META.expected_horizon_days)
            for s in self._strategies.values()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton accessor.
# Adapters is a hard dependency — its import populates the registry via
# module-level side effect.
# ──────────────────────────────────────────────────────────────────────────────
_REGISTRY: StrategyRegistry | None = None


def get_registry() -> StrategyRegistry:
    """Return the process-wide StrategyRegistry singleton (lazy populated)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = StrategyRegistry()
        # Side-effect import: adapters module registers all 5 concrete strategies.
        # An ImportError here is a genuine failure (syntax error in adapters,
        # missing dependency, etc.) and MUST propagate so consumers crash early
        # rather than silently operating on an empty registry.
        import engine.strategies.adapters  # noqa: F401
    return _REGISTRY


def _reset_registry_for_tests() -> None:
    """Internal test helper. Do NOT call from production code."""
    global _REGISTRY
    _REGISTRY = None
