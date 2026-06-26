"""engine.research.lens_registry — declarative lens registry + DAG resolver.

Per docs/spec_role_aware_test_routing.md v2 (commits 62621a22 +
2ca50bf2): replaces hardcoded `if pnl_df is not None: ...` chains in
factor_dispatcher.py with declarative lens registration. Each lens
module exposes a LENS_DECLARATION constant; this module auto-discovers
them, filters by FactorSpec metadata, topologically sorts by input
dependencies, and exposes a clean iterate API to the dispatcher.

Key design (per spec §15.A4 amendments):
  - Lenses declare APPLICABLE_TO filter (7-axis spec metadata)
  - Lenses declare INPUT_PROTOCOLS (TypedDict consumers; refactor-safe)
  - Lenses declare OUTPUT_PROTOCOL (TypedDict producers)
  - Lenses declare CONDITIONAL_ON (skip if prior lens output fails
    a predicate — prevents noise like running L2-6 when L2-4 already
    failed)
  - Lenses declare FALLBACK_CHAIN (alternate lens names if primary
    fails — e.g., KMPV → Lustig HML_FX → macro lite)
  - Lenses declare OUTPUT_SCHEMA primary/secondary split (partial
    output acceptable per spec §15.A4)
  - DAG resolution via topological sort; circular deps raise at
    discover() time, not at dispatch time

Commit 2 scope (this file): registry + DAG + applicability filter +
tests. Lenses declare LENS_DECLARATION constant in their own
module (separately commit in Commit 2 same PR).

Commit 3 scope: dispatcher refactor to USE this registry.
"""
from __future__ import annotations

import dataclasses as _dc
import importlib
import inspect
import logging
import pkgutil
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# LensDeclaration dataclass — what each lens module exposes
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class LensDeclaration:
    """One lens's declarative metadata. Modules expose
    `LENS_DECLARATION = LensDeclaration(...)` at module level.

    Fields below are documented in
    docs/spec_role_aware_test_routing.md §5 + §15.A4 + §15.A6.
    """
    name: str
    version: str

    # Applicability filter — values are tuples (OR semantics);
    # missing key OR empty tuple = wildcard (matches anything).
    # Example: {"investment_role": ("alpha",), "asset_class": ("equity",)}
    applicable_to: dict

    # TypedDict classes from _lens_protocols. Strings used to avoid
    # circular imports; resolved at discover() time.
    # Example: ("AnchorRegressionOutput", "PnlSeriesDataFrameContract")
    input_protocols: tuple

    # TypedDict class name from _lens_protocols.
    # Example: "IndustryExtensionOutput"
    output_protocol: str

    # Conditional skip predicate. None = always run.
    # Shape: {"lens": "<prior_lens_name>",
    #         "condition": <callable taking prior output dict>,
    #         "skip_reason_if_unmet": "<human-readable reason>"}
    conditional_on: Optional[dict]

    # Fallback chain — try alternative lens names if primary fails.
    # First entry = primary; subsequent = fallbacks in order.
    # Empty tuple = no fallback (primary or nothing).
    fallback_chain: tuple

    # Primary / secondary output fields. Primary MUST be present in
    # successful output; secondary nice-to-have (partial output OK).
    # Shape: {"primary": "<field_name>",
    #         "secondary": ("<field>", "<field>", ...)}
    output_schema: dict

    # Downstream lenses that consume this lens's output. Used by
    # DAG resolution. Empty tuple = leaf lens.
    consumed_by: tuple

    # The runner function. Signature:
    #   runner(spec: FactorSpec,
    #          template_result: TemplateResult,
    #          prior_outputs: dict[str, dict]) -> Optional[dict]
    # `prior_outputs` is keyed by lens name; lens reads its own
    # inputs out of it (per input_protocols declaration).
    runner: Callable

    def matches_spec(self, spec, fallback_axes: Optional[dict] = None) -> bool:
        """True iff this lens applies to the given FactorSpec.

        fallback_axes is the dict from infer_legacy_axes(spec) —
        used when spec field is None. Per spec §15.A5, the choice
        between explicit spec value and inferred fallback gets
        logged in routing_decisions.
        """
        for axis_key, allowed in self.applicable_to.items():
            if not allowed:
                continue  # wildcard
            spec_val = getattr(spec, axis_key, None)
            if spec_val is None and fallback_axes:
                spec_val = fallback_axes.get(axis_key)
            if spec_val is None:
                # Still None — can't determine applicability; default
                # to MATCH (conservative for the routing layer; lens
                # itself can refuse on its own input check)
                continue
            if spec_val not in allowed:
                return False
        return True


# ────────────────────────────────────────────────────────────────────
# Discovery — walk engine.research.* for LENS_DECLARATION constants
# ────────────────────────────────────────────────────────────────────
def discover_lenses(
    package_name: str = "engine.research",
) -> dict[str, LensDeclaration]:
    """Walk engine.research subpackages, find LENS_DECLARATION
    constants, return name → declaration dict.

    Skips modules that fail to import (logged), modules without
    LENS_DECLARATION (silently), and modules where the constant
    is the wrong type (logged at WARNING).
    """
    found: dict[str, LensDeclaration] = {}
    try:
        pkg = importlib.import_module(package_name)
    except ImportError as exc:
        logger.warning("lens_registry: cannot import %s: %s",
                          package_name, exc)
        return found

    # walk_packages of the package's __path__
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return found

    for module_info in pkgutil.walk_packages(pkg_path,
                                                  prefix=package_name + "."):
        # Skip private modules (_lens_protocols, _lens_registry) +
        # the registry itself
        if module_info.name.endswith("lens_registry"):
            continue
        try:
            mod = importlib.import_module(module_info.name)
        except ImportError as exc:
            logger.debug("lens_registry: skip %s (import fail): %s",
                            module_info.name, exc)
            continue
        decl = getattr(mod, "LENS_DECLARATION", None)
        if decl is None:
            continue
        if not isinstance(decl, LensDeclaration):
            logger.warning("lens_registry: %s.LENS_DECLARATION is %s "
                              "not LensDeclaration; skipping",
                              module_info.name, type(decl).__name__)
            continue
        if decl.name in found:
            logger.warning("lens_registry: duplicate lens name %r in "
                              "%s; ignoring (first wins)",
                              decl.name, module_info.name)
            continue
        found[decl.name] = decl
        logger.debug("lens_registry: discovered %s (%s)",
                        decl.name, module_info.name)
    return found


# ────────────────────────────────────────────────────────────────────
# Filter — applicable lenses for a given FactorSpec
# ────────────────────────────────────────────────────────────────────
def applicable_lenses(
    all_lenses: dict[str, LensDeclaration],
    spec,
    fallback_axes: Optional[dict] = None,
) -> list[LensDeclaration]:
    """Filter lenses by spec metadata. Returns list (order
    arbitrary; caller applies DAG sort separately)."""
    return [l for l in all_lenses.values()
            if l.matches_spec(spec, fallback_axes)]


# ────────────────────────────────────────────────────────────────────
# DAG resolution — topological sort by input dependencies
# ────────────────────────────────────────────────────────────────────
class CircularLensDependency(ValueError):
    """Raised at discover/resolve time if lenses form a cycle."""


def resolve_lens_dag(
    lenses: list[LensDeclaration],
) -> list[LensDeclaration]:
    """Topological sort by input dependencies.

    A lens A depends on lens B iff A.input_protocols contains
    B.output_protocol. Output order: dependencies before dependents.

    Raises CircularLensDependency on cycle.
    """
    # Build output_protocol → lens_name lookup
    producer_of: dict[str, str] = {}
    for lens in lenses:
        if lens.output_protocol in producer_of:
            # Two lenses produce the same output type — DAG can't
            # disambiguate; conservative skip (newest wins)
            logger.warning("lens_registry: duplicate output_protocol %r "
                              "between %s and %s",
                              lens.output_protocol,
                              producer_of[lens.output_protocol],
                              lens.name)
        producer_of[lens.output_protocol] = lens.name

    # Build edge list: lens_name → [dep_lens_name, ...]
    by_name = {l.name: l for l in lenses}
    deps: dict[str, list[str]] = {}
    for lens in lenses:
        deps[lens.name] = []
        for proto in lens.input_protocols:
            producer = producer_of.get(proto)
            if producer is not None and producer != lens.name:
                deps[lens.name].append(producer)

    # Kahn's algorithm: iteratively pop nodes with no unresolved deps
    result: list[LensDeclaration] = []
    remaining = dict(deps)
    while remaining:
        ready = [n for n, ds in remaining.items() if not ds]
        if not ready:
            cycle_names = sorted(remaining.keys())
            raise CircularLensDependency(
                f"circular lens dependency among {cycle_names}; "
                f"remaining deps: { {k: v for k, v in remaining.items()} }"
            )
        # Stable order within ready set: alphabetic by name
        ready.sort()
        for n in ready:
            result.append(by_name[n])
            del remaining[n]
            # Remove n from all other lenses' dep lists
            for other_deps in remaining.values():
                while n in other_deps:
                    other_deps.remove(n)
    return result


# ────────────────────────────────────────────────────────────────────
# Conditional execution check
# ────────────────────────────────────────────────────────────────────
def should_execute(
    lens: LensDeclaration,
    prior_outputs: dict[str, dict],
) -> tuple[bool, Optional[str]]:
    """Check whether a lens should be executed given prior outputs.

    Returns:
      (True, None) — proceed
      (False, reason_str) — skip with explanation for routing audit

    Honors:
      - conditional_on: if prior lens output fails the predicate, skip
      - input deps missing: if a non-fallback input protocol's
        producer didn't run, skip
    """
    if lens.conditional_on:
        prior_lens_name = lens.conditional_on.get("lens")
        condition       = lens.conditional_on.get("condition")
        # B.3 (2026-06-10): `lens` may be a tuple of alternative
        # producers — the first one with output wins. Needed because
        # the anchor stage is one of TWO mutually-exclusive lenses
        # (anchor_regression for equity, fx_carry_anchor_regression
        # for fx) and downstream conditionals must accept either.
        if isinstance(prior_lens_name, (tuple, list)):
            prior_output = next(
                (prior_outputs[n] for n in prior_lens_name
                  if prior_outputs.get(n) is not None),
                None,
            )
        else:
            prior_output = prior_outputs.get(prior_lens_name)
        if prior_output is None:
            return False, (f"conditional_on lens {prior_lens_name!r} "
                            "produced no output")
        try:
            if not condition(prior_output):
                return False, lens.conditional_on.get(
                    "skip_reason_if_unmet",
                    f"conditional_on predicate unmet against "
                    f"{prior_lens_name}",
                )
        except Exception as exc:
            logger.warning("lens_registry: %s conditional_on predicate "
                              "raised: %s; proceeding cautiously",
                              lens.name, exc)
    return True, None


# ────────────────────────────────────────────────────────────────────
# Validation — sanity-check a registry at discover time
# ────────────────────────────────────────────────────────────────────
def validate_registry(
    registry: dict[str, LensDeclaration],
) -> list[str]:
    """Return list of validation errors (empty list = clean).
    Caller can choose to log warnings vs raise on errors."""
    errors: list[str] = []
    from engine.research._lens_protocols import OUTPUT_PROTOCOL_REGISTRY

    # 1. Every output_protocol must be a known TypedDict name
    for name, lens in registry.items():
        if lens.output_protocol not in OUTPUT_PROTOCOL_REGISTRY:
            errors.append(
                f"{name}: output_protocol {lens.output_protocol!r} not "
                f"in OUTPUT_PROTOCOL_REGISTRY"
            )
        for proto in lens.input_protocols:
            if proto not in OUTPUT_PROTOCOL_REGISTRY:
                errors.append(
                    f"{name}: input_protocol {proto!r} not registered"
                )

    # 2. No circular dependencies
    try:
        resolve_lens_dag(list(registry.values()))
    except CircularLensDependency as exc:
        errors.append(str(exc))

    # 3. conditional_on.lens must reference an existing lens
    # B.3 (2026-06-10): conditional_on.lens may be a single lens name
    # OR a tuple of alternative producers (e.g. anchor_regression for
    # equity OR fx_carry_anchor_regression for FX). Validate each name
    # in the tuple individually; lens passes if ALL named producers exist.
    for name, lens in registry.items():
        if lens.conditional_on:
            target = lens.conditional_on.get("lens")
            if target:
                targets = target if isinstance(target, (tuple, list)) else (target,)
                missing = [t for t in targets if t not in registry]
                if missing:
                    errors.append(
                        f"{name}: conditional_on references unknown "
                        f"lens(es) {tuple(missing)!r}"
                    )

    return errors
