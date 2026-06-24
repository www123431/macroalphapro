"""engine/research/sleeve_registry.py — typed Sleeve interface +
decorator-based registry for Strategy Lifecycle Manager Phase 0.

Replaces the if/else dispatch pattern in combined_book.py with a typed
registry. Every deployed sleeve registers via `@register_sleeve(...)`
and downstream code (combined_book, audit scripts, paper trade) resolves
sleeves by id at runtime.

Why a Protocol-based interface (not abstract base class):
  - Structural typing — existing sleeve modules can adopt the interface
    by just implementing the methods, no inheritance refactor required
  - Type checkers (mypy / pyright) verify conformance without runtime cost
  - Decouples lifecycle from sleeve implementation

Registry semantics:
  - Decorator REGISTERS; it does NOT auto-create state-store rows.
    Registration is a class-level fact (this code knows how to build
    this sleeve). State-store creation is a separate concern (this
    strategy is in lifecycle).
  - get_sleeve() returns the registered class instance OR raises KeyError
  - list_sleeves(state=...) filters by lifecycle state (requires DB read)

Design split rationale (separation of concerns):
  - sleeve_registry  → "what sleeves CAN we build?"     (code-level)
  - strategy_state_store → "what state are they IN?"    (lifecycle)
  - combined_book    → "how do we blend the LIVE ones?"  (allocation)

Three orthogonal axes, three modules. Anti-pattern: combining them
into one fat "deployment manager" class.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Protocol, runtime_checkable

import pandas as pd

from engine.research.strategy_lifecycle import (
    AuditBlocks,
    StrategyState,
)


# ── Sleeve interface ────────────────────────────────────────────────────


@runtime_checkable
class SleeveProtocol(Protocol):
    """Interface every registered sleeve must implement.

    @runtime_checkable enables `isinstance(obj, SleeveProtocol)` for
    defensive checks at registration time.
    """

    @property
    def strategy_id(self) -> str: ...

    @property
    def library_yaml_path(self) -> Path: ...

    def returns(self) -> pd.Series:
        """Monthly returns series (datetime-indexed). Caller must NOT
        mutate the result; sleeves cache aggressively."""
        ...

    def audit_blocks(self) -> AuditBlocks:
        """Typed audit blocks (cost_model + factor_exposure) loaded from
        the library YAML at construction time."""
        ...


# ── Registry ────────────────────────────────────────────────────────────


_REGISTRY: dict[str, type[SleeveProtocol]] = {}
_INSTANCES: dict[str, SleeveProtocol] = {}


class SleeveAlreadyRegisteredError(ValueError):
    """Raised on duplicate registration with strict=True."""


def register_sleeve(
    strategy_id: str,
    *,
    strict: bool = True,
) -> Callable[[type[SleeveProtocol]], type[SleeveProtocol]]:
    """Decorator: register a sleeve class under `strategy_id`.

    Usage:
        @register_sleeve("post_earnings_drift_pit_sn")
        class PitSnDpeadSleeve:
            strategy_id = "post_earnings_drift_pit_sn"
            library_yaml_path = Path("data/research/mechanism_library/...")
            def returns(self) -> pd.Series: ...
            def audit_blocks(self) -> AuditBlocks: ...

    strict=True (default) raises SleeveAlreadyRegisteredError on
    duplicate id. Set strict=False for hot-reload / test scenarios.
    """
    def decorator(cls: type[SleeveProtocol]) -> type[SleeveProtocol]:
        if strict and strategy_id in _REGISTRY:
            raise SleeveAlreadyRegisteredError(
                f"sleeve {strategy_id!r} already registered as "
                f"{_REGISTRY[strategy_id].__name__}"
            )
        # Verify the class conforms to SleeveProtocol BEFORE registering.
        # We can't instantiate here (no zero-arg guarantee) so we check
        # method presence + signatures.
        for attr in ("strategy_id", "library_yaml_path", "returns", "audit_blocks"):
            if not hasattr(cls, attr):
                raise TypeError(
                    f"sleeve {cls.__name__} missing required attribute {attr!r}"
                )
        _REGISTRY[strategy_id] = cls
        # Drop any cached instance so next get_sleeve() rebuilds.
        _INSTANCES.pop(strategy_id, None)
        return cls

    return decorator


def get_sleeve_class(strategy_id: str) -> type[SleeveProtocol]:
    """Resolve a registered sleeve class by id."""
    if strategy_id not in _REGISTRY:
        raise KeyError(
            f"sleeve {strategy_id!r} not registered; "
            f"known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[strategy_id]


def get_sleeve(strategy_id: str) -> SleeveProtocol:
    """Get a cached SleeveProtocol instance (constructed lazily).

    Sleeve classes must support zero-arg construction. Pass-through
    parameters belong in module-level constants or YAML fields, not
    constructor args — keeps the registry contract simple.
    """
    if strategy_id in _INSTANCES:
        return _INSTANCES[strategy_id]
    cls = get_sleeve_class(strategy_id)
    instance = cls()  # zero-arg construction
    if not isinstance(instance, SleeveProtocol):
        raise TypeError(
            f"sleeve {cls.__name__} does not satisfy SleeveProtocol"
        )
    _INSTANCES[strategy_id] = instance
    return instance


def list_registered_sleeves() -> list[str]:
    """All registered strategy_ids, sorted for deterministic output."""
    return sorted(_REGISTRY.keys())


def list_sleeves_in_state(state: StrategyState) -> list[SleeveProtocol]:
    """Filter registered sleeves by their lifecycle state (DB read).

    Requires the strategy_state_store DB; will trigger init_db() on
    first call. Sleeves registered in code but never created in the
    state store are excluded.
    """
    # Imported here to avoid circular import (state_store imports lifecycle,
    # registry imports lifecycle, so deferring this is safest).
    from engine.research.strategy_state_store import get_strategy

    out: list[SleeveProtocol] = []
    for sid in list_registered_sleeves():
        try:
            record = get_strategy(sid)
        except KeyError:
            continue
        if record.current_state == state:
            out.append(get_sleeve(sid))
    return out


def clear_registry_for_test() -> None:
    """Test-only: wipe the registry. Production code MUST NOT call this."""
    _REGISTRY.clear()
    _INSTANCES.clear()


# ── Helper: load audit blocks from YAML ─────────────────────────────────


def load_audit_blocks_from_yaml(yaml_path: Path) -> AuditBlocks:
    """Parse cost_model + factor_exposure blocks from a library YAML into
    typed AuditBlocks. Pydantic validation enforced.

    Use this in sleeve `audit_blocks()` implementations to avoid manual
    field-by-field parsing.
    """
    try:
        from ruamel.yaml import YAML
        yaml_loader = YAML(typ="safe")
        raw = yaml_loader.load(yaml_path.read_text(encoding="utf-8"))
    except ImportError:
        import yaml as _pyyaml
        raw = _pyyaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if "cost_model" not in raw or "factor_exposure" not in raw:
        raise ValueError(
            f"{yaml_path.name} missing cost_model or factor_exposure block"
        )

    # Pydantic v2 model_validate gracefully accepts extra keys but enforces
    # required fields per the @model_validator in lifecycle.py.
    from engine.research.strategy_lifecycle import (
        CostModelAudit,
        FactorExposureAudit,
    )

    cost = CostModelAudit.model_validate(raw["cost_model"])
    factor = FactorExposureAudit.model_validate(raw["factor_exposure"])
    return AuditBlocks(cost_model=cost, factor_exposure=factor)
