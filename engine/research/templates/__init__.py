"""Layer 1 strategy templates — AUTO-DISCOVERABLE.

Per [[feedback-no-brittle-hardcoding-2026-05-30]]:
Drop a new template_*.py file into this directory + declare entry function,
and it auto-registers. NO MANUAL TEMPLATES dict edits required.

Discovery protocol — each template module must declare ONE of:
  - `run_<template_id>` function (e.g. run_equity_xsmom)
  - or `TEMPLATE_ENTRY` constant pointing at the entry callable
  - and `TEMPLATE_ID` constant matching its registry key (optional;
    derived from module name by stripping leading underscore)

Optional declarations:
  - `warmup_months(binding: dict) -> int` (used by protocol designer)
  - `validate_composition(binding: dict)` (used by primitive_composition)

Existing templates:
  - equity_xsmom:          JT 1993 cross-sectional momentum
  - factor_quartile:       generic single-factor decile L/S
  - cross_asset_tsmom:     MOP 2012 per-instrument trend
  - primitive_composition: Tier 2 DAG of allowlisted primitives
"""
from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


def _discover_templates() -> dict[str, Callable]:
    """Scan this package directory for template modules and build registry.

    Convention:
      module name <X> → looks for run_<X> function in the module.
      e.g. equity_xsmom.py → run_equity_xsmom

    Modules can override via TEMPLATE_ENTRY + TEMPLATE_ID constants.
    Modules starting with underscore are skipped (private).
    """
    registry: dict[str, Callable] = {}
    pkg_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(pkg_path)]):
        name = module_info.name
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"engine.research.templates.{name}")
        except ImportError as exc:
            logger.warning("template %s import failed: %s", name, exc)
            continue

        # Explicit override path
        if hasattr(mod, "TEMPLATE_ENTRY") and hasattr(mod, "TEMPLATE_ID"):
            entry = mod.TEMPLATE_ENTRY
            template_id = mod.TEMPLATE_ID
            if callable(entry):
                registry[template_id] = entry
                continue

        # Convention path: run_<module_name>
        fn_name = f"run_{name}"
        entry = getattr(mod, fn_name, None)
        if callable(entry):
            registry[name] = entry
            continue

        # No discoverable entry — skip but log
        logger.debug("template module %s has no run_%s or TEMPLATE_ENTRY; skipped",
                      name, name)
    return registry


TEMPLATES: dict[str, Callable] = _discover_templates()


def reload_templates() -> dict[str, Callable]:
    """Re-scan for templates. Useful for dev/test when adding modules."""
    global TEMPLATES
    TEMPLATES = _discover_templates()
    return TEMPLATES


# Re-export the canonical entry functions for backward compat
from engine.research.templates.cross_asset_tsmom import run_cross_asset_tsmom
from engine.research.templates.equity_xsmom import run_equity_xsmom
from engine.research.templates.factor_quartile import run_factor_quartile
from engine.research.templates.primitive_composition import (
    run_primitive_composition,
    validate_composition,
)


__all__ = [
    "TEMPLATES",
    "reload_templates",
    "run_equity_xsmom",
    "run_factor_quartile",
    "run_cross_asset_tsmom",
    "run_primitive_composition",
    "validate_composition",
]
