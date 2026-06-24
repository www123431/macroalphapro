"""engine/feature_store — Reproducibility layer for deployed factor sleeves.

L0 of the factor discovery engine: every deployed sleeve gets a YAML
spec pointing at its materialize function + reproducibility hash +
sanity checks. The spec is the contract; the function is the
implementation.

Goal: 5 years from now, `git checkout <commit> && python -m
engine.feature_store materialize <spec_id>` produces byte-identical
output to today's commit. This is Chen-Zimmermann 2021 OSAP pattern
adapted to a single-researcher workflow.

Design constraints:
  - DO NOT rewrite existing sleeve code. Each spec WRAPS an existing
    build_*_returns() function. Specs add reproducibility metadata,
    not new logic.
  - Specs live in data/feature_store/_specs/*.yaml (separate from
    data/research/mechanism_library/ to avoid disrupting existing
    library audit hooks)
  - Materialized output lives in data/feature_store/_computed/ with
    a (spec_id, version, input_hash) addressed path so cache
    invalidation is automatic
  - Sanity checks are MANDATORY — every materialize call validates
    output shape + range + no-nan-after-first-obs before returning.
    Bad materialization fails LOUD, not silent.
"""
from engine.feature_store.materializer import (
    materialize_spec,
    verify_spec,
)
from engine.feature_store.registry import (
    list_specs,
    load_spec,
)

__all__ = [
    "materialize_spec",
    "verify_spec",
    "list_specs",
    "load_spec",
]
