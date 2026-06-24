"""composer.contract — abstract base + registry for atomic components.

5 component roles map to 5 sections of HypothesisSpec:

  ComponentRole.SIGNAL       reads spec.legs[*].signal_type, lookback,
                              quantile; returns a signal panel
                              (DataFrame indexed by date, cols by asset)
  ComponentRole.UNIVERSE     reads spec.universe.asset_class, subset;
                              returns asset-set membership per date
  ComponentRole.WEIGHTING    reads spec.construction.weighting;
                              transforms signals → position weights
  ComponentRole.REBALANCE    reads spec.construction.rebalance;
                              filters dates to rebalance dates
  ComponentRole.RISK_FILTER  reads spec.risk.vol_target_annual etc;
                              produces an overlay multiplier on returns

Each Component is registered with a typed key (the spec value it
accepts). Composer looks up the right component per role given the
spec values.

Coverage contract
-----------------
Calling is_spec_covered(spec) walks every role and verifies a
component is registered for the corresponding spec value. Missing
component → covered=False + the gap is surfaced in the UI ("no
WEIGHTING component for INV_VOL yet; ask Claude to implement").

Failure mode
------------
NEVER silently substitute. If a component is missing, the composer
raises ComponentNotFound. The UI must offer Claude handoff (NOT
fake-test with a substitute).
"""
from __future__ import annotations

import abc
import enum
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional


# ── Roles ──────────────────────────────────────────────────


class ComponentRole(str, enum.Enum):
    """The 5 atomic component categories."""
    SIGNAL      = "SIGNAL"
    UNIVERSE    = "UNIVERSE"
    WEIGHTING   = "WEIGHTING"
    REBALANCE   = "REBALANCE"
    RISK_FILTER = "RISK_FILTER"


# ── Result envelope ──────────────────────────────────────


@dataclass(frozen=True)
class ComponentResult:
    """What a component returns to the composer.

    `data` is intentionally typed as Any because each role returns a
    different shape:
      SIGNAL      pd.DataFrame  (date × asset)
      UNIVERSE    pd.DataFrame  (date × asset, boolean membership)
      WEIGHTING   pd.DataFrame  (date × asset, weights summing to 0 or 1)
      REBALANCE   pd.DatetimeIndex
      RISK_FILTER pd.Series     (date → scalar multiplier)

    `metadata` captures provenance for the audit trail:
      - which spec fields the component consumed
      - what data vintages it pulled (panel start/end dates)
      - any internal parameters used

    Composer aggregates metadata across components into the final
    provenance row.
    """
    data:     Any
    metadata: dict


# ── Component base ──────────────────────────────────────


class Component(abc.ABC):
    """Subclass + decorate with @register_component(role, key)."""

    role:        ComponentRole = None        # type: ignore
    key:         str           = ""           # the spec value this component handles
    description: str           = ""

    @abc.abstractmethod
    def build(self, spec, context: dict) -> ComponentResult:
        """Read the spec, produce the component output.

        Args:
          spec     HypothesisSpec
          context  dict carrying outputs of prior-role components.
                    For example, a WEIGHTING component receives
                    {"signals": <SIGNAL ComponentResult>,
                     "universe": <UNIVERSE ComponentResult>} so it can
                    apply weighting only on in-universe assets.

        Returns a ComponentResult.

        Must raise (do not silently substitute) when:
          - required data missing on disk
          - spec field value contradicts this component's assumptions
        """
        raise NotImplementedError

    def covers(self, spec) -> bool:
        """Override if the component handles only a sub-range of spec
        values (e.g. an INV_VOL weighting needs > 24 obs of returns
        history; covers=False on very short hypothesis_specs)."""
        return True


# ── Registry ──────────────────────────────────────────


# (role, key.upper()) → Component subclass
COMPONENT_REGISTRY: dict[tuple[ComponentRole, str], type[Component]] = {}


def register_component(role: ComponentRole, key: str):
    """Decorator. Maps (role, key) → Component subclass."""
    rkey = key.upper().strip()
    def deco(cls: type[Component]) -> type[Component]:
        if not issubclass(cls, Component):
            raise TypeError(f"{cls.__name__} must subclass Component")
        cls.role = role
        cls.key  = rkey
        full_key = (role, rkey)
        if full_key in COMPONENT_REGISTRY:
            raise ValueError(
                f"duplicate component registration: {role.value}/{key}")
        COMPONENT_REGISTRY[full_key] = cls
        return cls
    return deco


def get_component(role: ComponentRole, key: str) -> Optional[type[Component]]:
    return COMPONENT_REGISTRY.get((role, key.upper().strip()))


def list_components() -> list[type[Component]]:
    return [cls for (_, _), cls in sorted(COMPONENT_REGISTRY.items(),
                                            key=lambda kv: (kv[0][0].value, kv[0][1]))]


def iter_role_components(role: ComponentRole) -> Iterator[type[Component]]:
    for (r, _k), cls in COMPONENT_REGISTRY.items():
        if r == role:
            yield cls


# ── Coverage query ─────────────────────────────────────


def _ensure_components_imported() -> None:
    """Side-effect import of all registered atomic components."""
    try:
        from engine.composer import components  # noqa: F401
    except Exception:
        pass


@dataclass(frozen=True)
class CoverageGap:
    role:           ComponentRole
    expected_key:   str
    reason:         str          # "missing" | "uncovered_range"


def is_spec_covered(spec) -> tuple[bool, list[CoverageGap]]:
    """Walk every role, verify a registered component exists for each
    spec value. Returns (all_covered, gaps_if_any).

    Note: a single hypothesis may have multiple SIGNAL legs; we require
    ALL of them to have a registered SIGNAL component."""
    _ensure_components_imported()
    gaps: list[CoverageGap] = []

    # SIGNAL — one component per leg
    for leg in spec.legs:
        cls = get_component(ComponentRole.SIGNAL, leg.signal_type.value)
        if cls is None:
            gaps.append(CoverageGap(
                role=ComponentRole.SIGNAL,
                expected_key=leg.signal_type.value,
                reason="missing",
            ))

    # UNIVERSE
    uni_key = f"{spec.universe.asset_class.value}__{spec.universe.subset.value}"
    cls = get_component(ComponentRole.UNIVERSE, uni_key)
    if cls is None:
        # try fallback to asset_class only (subset=ALL)
        cls = get_component(ComponentRole.UNIVERSE,
                            f"{spec.universe.asset_class.value}__ALL")
    if cls is None:
        gaps.append(CoverageGap(
            role=ComponentRole.UNIVERSE,
            expected_key=uni_key,
            reason="missing",
        ))

    # WEIGHTING
    cls = get_component(ComponentRole.WEIGHTING, spec.construction.weighting.value)
    if cls is None:
        gaps.append(CoverageGap(
            role=ComponentRole.WEIGHTING,
            expected_key=spec.construction.weighting.value,
            reason="missing",
        ))

    # REBALANCE
    cls = get_component(ComponentRole.REBALANCE, spec.construction.rebalance.value)
    if cls is None:
        gaps.append(CoverageGap(
            role=ComponentRole.REBALANCE,
            expected_key=spec.construction.rebalance.value,
            reason="missing",
        ))

    # RISK_FILTER — optional; only needed if spec.risk.vol_target_annual set
    if spec.risk.vol_target_annual is not None:
        cls = get_component(ComponentRole.RISK_FILTER, "VOL_TARGET")
        if cls is None:
            gaps.append(CoverageGap(
                role=ComponentRole.RISK_FILTER,
                expected_key="VOL_TARGET",
                reason="missing",
            ))

    return (len(gaps) == 0, gaps)


def coverage_summary() -> dict:
    """High-level summary for UI: per-role coverage count."""
    _ensure_components_imported()
    counts: dict[str, int] = {}
    keys_by_role: dict[str, list[str]] = {}
    for (role, key), _ in COMPONENT_REGISTRY.items():
        counts[role.value] = counts.get(role.value, 0) + 1
        keys_by_role.setdefault(role.value, []).append(key)
    for r in keys_by_role:
        keys_by_role[r].sort()
    return {
        "counts":      counts,
        "keys_by_role": keys_by_role,
        "total":       len(COMPONENT_REGISTRY),
    }


# ── Composer exception ──────────────────────────────────


class ComponentNotFound(Exception):
    """Raised by the composer when a required component is unregistered.
    Never silently substituted (LdP §2 / SR-11-7)."""
    def __init__(self, role: ComponentRole, key: str):
        self.role = role
        self.key  = key
        super().__init__(f"no component registered for {role.value}/{key}")
