"""engine/research/templates/primitive_composition.py — Tier 2 flexible composition.

Closes the rigidity gap of named templates. For mechanisms that don't fit any
existing Layer 1 template, this template accepts a DAG-shaped composition of
allowlisted Layer 0 primitives.

CRITICAL: This is NOT LLM code generation. The LLM produces a JSON/YAML DAG
description; the runner executes ONLY primitives from PRIMITIVE_REGISTRY.
No eval(), no exec(), no arbitrary callables — every step is
`getattr(primitives_module, name)(**resolved_args)`.

Schema:
  binding:
    inputs:    list[str]      # names of data_kwargs to expose as initial state
    steps:     list[dict]     # each step: {id, primitive, args, outputs?}
    output:    str            # final state key to return

Each step:
  id:        unique name within the composition (used by downstream ref:)
  primitive: name from primitives.PRIMITIVE_REGISTRY (allowlist)
  args:      dict of arg_name → value | "ref:state_key" reference
  outputs:   optional list[str] for primitives with n_outputs > 1
              (e.g. top_bottom_membership → [long_mask, short_mask])

Validation (static, before any execution):
  V1 every primitive in registry
  V2 every arg in primitive's introspected signature
  V3 all required args present
  V4 every "ref:" target resolves to a prior step or input
  V5 no duplicate step ids
  V6 outputs count matches primitive's n_outputs
  V7 final output ref resolves
  V8 no cyclic refs (DAG must be acyclic — enforced by static ordering)
"""
from __future__ import annotations

import dataclasses
import inspect
import logging
from typing import Any

import pandas as pd

from engine.research import primitives as P
from engine.research.primitives import PRIMITIVE_REGISTRY

logger = logging.getLogger(__name__)


REF_PREFIX = "ref:"


@dataclasses.dataclass
class CompositionValidation:
    ok:      bool
    reasons: list[str]
    n_steps: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _is_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(REF_PREFIX)


def _ref_target(value: str) -> str:
    return value[len(REF_PREFIX):]


def validate_composition(binding: dict) -> CompositionValidation:
    """Static validation of a composition binding. Run BEFORE any execution.

    Returns CompositionValidation with all detected issues (does NOT raise)."""
    reasons: list[str] = []
    inputs = binding.get("inputs") or []
    steps = binding.get("steps") or []
    output = binding.get("output")

    if not isinstance(inputs, list):
        reasons.append("binding.inputs must be a list")
        inputs = []
    if not isinstance(steps, list) or not steps:
        reasons.append("binding.steps must be a non-empty list")
        steps = []
    if not output:
        reasons.append("binding.output is required")

    known_state = set(inputs)
    seen_ids: set[str] = set()

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            reasons.append(f"step[{i}] is not a dict")
            continue
        sid = step.get("id")
        if not sid:
            reasons.append(f"step[{i}] missing 'id'")
            continue
        if sid in seen_ids:
            reasons.append(f"step[{i}] duplicate id {sid!r}")
            continue
        seen_ids.add(sid)

        pname = step.get("primitive")
        if pname not in PRIMITIVE_REGISTRY:
            reasons.append(
                f"step[{sid}] primitive {pname!r} not in registry "
                f"(allowlist size={len(PRIMITIVE_REGISTRY)})"
            )
            known_state.add(sid)    # placeholder
            continue

        entry = PRIMITIVE_REGISTRY[pname]
        fn = entry["fn"]
        n_outputs = entry["n_outputs"]

        # V2 / V3: args must match signature
        sig = inspect.signature(fn)
        valid_params = set(sig.parameters.keys())
        required_params = {
            name for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                    inspect.Parameter.VAR_KEYWORD)
        }
        args = step.get("args") or {}
        unknown = set(args) - valid_params
        if unknown:
            reasons.append(f"step[{sid}] unknown args {sorted(unknown)}")
        missing = required_params - set(args)
        if missing:
            reasons.append(
                f"step[{sid}] missing required args {sorted(missing)}"
            )

        # V4: refs must resolve to known_state at this point
        for arg_name, arg_value in args.items():
            if _is_ref(arg_value):
                ref = _ref_target(arg_value)
                if ref not in known_state:
                    reasons.append(
                        f"step[{sid}] ref {arg_value!r} (arg {arg_name!r}) "
                        f"does not resolve to a prior step or input"
                    )

        # V6: outputs count
        declared_outputs = step.get("outputs")
        if declared_outputs is None:
            known_state.add(sid)
        elif isinstance(declared_outputs, list):
            if len(declared_outputs) != n_outputs:
                reasons.append(
                    f"step[{sid}] declared {len(declared_outputs)} outputs but "
                    f"primitive {pname!r} returns {n_outputs}"
                )
            for o in declared_outputs:
                if not isinstance(o, str):
                    reasons.append(f"step[{sid}] output entry must be string")
                else:
                    known_state.add(o)
        else:
            reasons.append(f"step[{sid}] outputs must be a list")

    # V7: final output ref
    if output and output not in known_state:
        reasons.append(
            f"binding.output {output!r} does not resolve to any step or input"
        )

    return CompositionValidation(
        ok=not reasons, reasons=reasons, n_steps=len(steps)
    )


def _resolve(value, state: dict):
    """Resolve a ref or pass through literal value."""
    if _is_ref(value):
        ref = _ref_target(value)
        if ref not in state:
            raise KeyError(f"unresolved reference {value!r} at runtime")
        return state[ref]
    return value


def run_primitive_composition(*, inputs: list[str] | None = None,
                                  steps: list[dict] | None = None,
                                  output: str | None = None,
                                  **data_kwargs: Any) -> pd.Series:
    """Execute a validated composition.

    Runs validate_composition first; raises ValueError on failure
    (this template never silently mis-executes).
    """
    binding = {"inputs": inputs or [], "steps": steps or [], "output": output}
    v = validate_composition(binding)
    if not v.ok:
        raise ValueError(f"composition validation failed: {v.reasons}")

    # Initialize state with declared inputs
    state: dict[str, Any] = {}
    for inp_name in binding["inputs"]:
        if inp_name not in data_kwargs:
            raise KeyError(
                f"declared input {inp_name!r} not in data_kwargs "
                f"(provided: {sorted(data_kwargs)})"
            )
        state[inp_name] = data_kwargs[inp_name]

    # Execute steps in order
    for step in binding["steps"]:
        sid = step["id"]
        pname = step["primitive"]
        entry = PRIMITIVE_REGISTRY[pname]
        fn = entry["fn"]
        n_outputs = entry["n_outputs"]

        # Resolve args
        args = step.get("args") or {}
        resolved = {k: _resolve(v, state) for k, v in args.items()}

        # Execute
        result = fn(**resolved)

        # Store outputs
        declared_outputs = step.get("outputs")
        if declared_outputs is None:
            state[sid] = result
        else:
            # Multi-output: unpack tuple
            if not isinstance(result, tuple) or len(result) != n_outputs:
                raise RuntimeError(
                    f"step {sid!r}: primitive {pname!r} expected to return "
                    f"tuple of {n_outputs}, got {type(result).__name__}"
                )
            for out_name, out_val in zip(declared_outputs, result):
                state[out_name] = out_val

    return state[binding["output"]]
