"""
engine/strategies/base.py — StrategyModule ABC + StrategyMeta dataclass.

Each concrete strategy declares its locked metadata as a class-level ``META``
attribute and implements two methods:

  - ``generate_signal(as_of)``       — returns a StrategySignal
  - ``is_rebalance_day(as_of, **kw)`` — returns True if today is a rebalance day

The ABC is intentionally THIN. It does NOT own:
  - sleeve-level allocation (lives in StrategyRegistry / Sleeve)
  - leverage factor (orchestrator concern)
  - signal cache / data fetching (each concrete strategy's adapter delegates
    to existing engine.factors.* / engine.path_c.* / engine.portfolio.* modules)

DOCTRINE compliance:
  - 0-LLM-in-DECISION: ABC has no LLM hooks. Concrete strategies remain pure.
  - Spec-lock: META is a frozen dataclass on a class attribute; mutating it
    requires editing the source file, which Engineer agent (Level 2.5) can do
    only via diff that user manually commits.
"""
from __future__ import annotations

import abc
import dataclasses
import datetime
from typing import Any, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    # Forward reference only — no runtime import to keep this module free of
    # heavy paper_trade_combined dependencies. Concrete strategy adapters
    # import StrategySignal at runtime from the original module.
    from engine.portfolio.paper_trade_combined import StrategySignal


@dataclasses.dataclass(frozen=True)
class StrategyMeta:
    """Per-strategy locked metadata.

    Mirrors the keys of the legacy ``STRATEGY_DISPLAY_META`` dict in
    paper_trade_combined.py plus ``expected_horizon_days`` from the legacy
    ``STRATEGY_SPEC_MAP`` dict in attribution_logger.py. Unifying these two
    dicts is the explicit goal of this refactor.

    Field semantics:
      spec_id              : integer primary key in spec_registry table
      spec_hash_short      : 8-char prefix of spec content hash (HARKing seal)
      sleeve_id            : must be a member of ALLOWED_SLEEVES
      intra_sleeve_weight  : share of this strategy within its sleeve [0, 1]
      rebalance_days       : approximate rebalance cadence; 0 means continuous
      expected_horizon_days: typical holding period (used by attribution_logger)
      label                : human-readable description for UI
      doctrine             : one-line academic / structural anchor
      universe             : universe descriptor for UI display
      color                : hex color for charts
      display_short        : compact label for table cells / sparklines
    """
    spec_id:                int
    spec_hash_short:        str
    sleeve_id:              str
    intra_sleeve_weight:    float
    rebalance_days:         int
    expected_horizon_days:  int
    label:                  str
    doctrine:               str
    universe:               str
    color:                  str
    display_short:          str

    def __post_init__(self) -> None:
        # Frozen-dataclass-friendly invariant checks. Raises at class-definition
        # time, not at runtime — the user sees the failure before commit.
        if not (0.0 <= self.intra_sleeve_weight <= 1.0):
            raise ValueError(
                f"intra_sleeve_weight must be in [0,1], got {self.intra_sleeve_weight}"
            )
        if self.rebalance_days < 0:
            raise ValueError(f"rebalance_days must be >= 0, got {self.rebalance_days}")
        if len(self.spec_hash_short) != 8:
            raise ValueError(
                f"spec_hash_short must be 8 chars, got {self.spec_hash_short!r}"
            )


class StrategyModule(abc.ABC):
    """Abstract base class for a paper-trade strategy.

    Subclasses must set two class-level attributes:
      NAME : str  — canonical uppercase identifier matching legacy keys
                    (e.g. ``"K1_BAB"``, ``"D_PEAD"``, ``"PATH_N"``,
                    ``"CTA_PQTIX"``, ``"AC_TLT_GLD"``)
      META : StrategyMeta

    And implement two methods:
      generate_signal(as_of)        -> StrategySignal
      is_rebalance_day(as_of, **kw) -> bool

    Convenience helper ``book_weight`` is provided in non-abstract form so the
    orchestrator can compute ``sleeve_w * intra_w * leverage`` uniformly across
    all strategies.
    """

    NAME: ClassVar[str]
    META: ClassVar[StrategyMeta]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Skip the check for intermediate ABCs (those that don't define NAME).
        if abc.ABC in cls.__bases__:
            return
        for required in ("NAME", "META"):
            if not hasattr(cls, required):
                raise TypeError(
                    f"{cls.__name__} must declare class attribute {required!r}"
                )
        if not isinstance(cls.META, StrategyMeta):
            raise TypeError(
                f"{cls.__name__}.META must be a StrategyMeta instance, "
                f"got {type(cls.META).__name__}"
            )
        if not (cls.NAME.isupper() and cls.NAME.replace("_", "").isalnum()):
            raise ValueError(
                f"{cls.__name__}.NAME must be UPPER_SNAKE_CASE, got {cls.NAME!r}"
            )

    @abc.abstractmethod
    def generate_signal(self, as_of: datetime.date) -> "StrategySignal":
        """Produce today's StrategySignal for this strategy.

        The signal's ``trade_attributions`` tuple MUST be populated when
        ``status == 'OK'`` so the attribution_logger can persist forensic
        context (Sprint H invariant).
        """

    @abc.abstractmethod
    def is_rebalance_day(self, as_of: datetime.date, **kwargs: Any) -> bool:
        """Return True if today is a rebalance day for this strategy.

        Kwargs are strategy-specific (e.g. D-PEAD takes ``cache_path``,
        Path N takes ``msp500_events``, CTA takes ``current_pqtix_weight``).
        Strategies that ignore kwargs simply do not consume them.
        """

    def book_weight(
        self,
        sleeve_allocation:  dict[str, float],
        leverage:           float = 1.0,
    ) -> float:
        """Effective weight of this strategy in the combined book.

        book_weight = sleeve_allocation[META.sleeve_id] * META.intra_sleeve_weight * leverage
        """
        try:
            sleeve_w = sleeve_allocation[self.META.sleeve_id]
        except KeyError as e:
            raise KeyError(
                f"sleeve_id {self.META.sleeve_id!r} not present in sleeve_allocation "
                f"(known keys: {sorted(sleeve_allocation)})"
            ) from e
        return sleeve_w * self.META.intra_sleeve_weight * leverage

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} NAME={self.NAME} "
            f"sleeve={self.META.sleeve_id} spec_id={self.META.spec_id} "
            f"hash={self.META.spec_hash_short}>"
        )
