"""engine/research/strategy_dsl_runner.py — Layer 3 DSL runner.

Bridges Hypothesis Generator proposals to executable strategy returns. Given
a proposal dict with an `execution_template` field, dispatches to the right
Layer 1 template and returns a `pd.Series` ready for `run_gate(returns, ...)`.

Doctrine:
- Templates dispatched via name → must be in the TEMPLATES registry
- No LLM coding here. Bindings are static dicts passed to template functions.
- The runner is a thin dispatcher. Bug surface is the templates + primitives.

Data inputs (price_panel, return_panel) MUST be provided by the caller.
This decouples the runner from data-loading details; the strategy module
that hosts the data loader passes them in.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from engine.research.templates import TEMPLATES

logger = logging.getLogger(__name__)


def run_proposal(proposal: dict, **data_kwargs: Any) -> pd.Series:
    """Translate a generator proposal to monthly L/S returns.

    Args:
      proposal: dict from hypothesis_generator with `execution_template:
                  {template_id, binding}` field
      **data_kwargs: data inputs forwarded to template (price_panel,
                      return_panel, etc.)

    Returns:
      pd.Series — ready for run_gate(returns_series, name=..., ...)

    Raises:
      KeyError: if template_id not registered
      ValueError: if proposal lacks execution_template field
    """
    et = proposal.get("execution_template")
    if not et or not et.get("template_id"):
        raise ValueError(
            f"proposal missing execution_template.template_id field: {proposal!r}"
        )
    template_id = et["template_id"]
    binding = et.get("binding") or {}

    if template_id not in TEMPLATES:
        raise KeyError(
            f"template_id {template_id!r} not in TEMPLATES registry; "
            f"known: {sorted(TEMPLATES)}"
        )

    template_fn = TEMPLATES[template_id]
    return template_fn(**binding, **data_kwargs)


def list_templates() -> list[str]:
    """List registered template IDs."""
    return sorted(TEMPLATES.keys())
