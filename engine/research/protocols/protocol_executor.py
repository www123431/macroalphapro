"""engine/research/protocols/protocol_executor.py — multi-leg execution.

Given an InstantiatedProtocol + a proposal (with execution_template) + data
kwargs, executes ALL legs (independent, parallel-eligible in spirit) via
the DSL runner, aggregates results into a MultiLegVerdict.

Doctrine:
- ALL legs run regardless of intermediate failures (no early-exit).
  Failure-mode analysis needs complete coverage.
- Each leg independently calls DSL runner + run_gate.
- protocol_hash is stamped into each gate_runs entry for audit.
- Decomposition checks operate on the PRIMARY leg only (since they require
  factor returns / book correlation from a specific run).
- Verdict aggregator applies the family's verdict_rule deterministically.
- ALL leg-level pass/fail decisions are pre-committed in the protocol;
  the executor does NOT make threshold judgments — it just evaluates against
  the protocol's pass_criteria.

Output: MultiLegVerdict — full result + overall GREEN/YELLOW/RED.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import Any

import pandas as pd

from engine.research.protocols.protocol_designer import (
    DecompositionCheck,
    InstantiatedProtocol,
    ResolvedLeg,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LegResult:
    leg_id:             str
    is_primary:         bool
    gate_summary:       dict | None
    pass_criteria_eval: dict[str, bool]         # criterion → passed?
    leg_passed:         bool
    error:              str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class DecompositionResult:
    check_id:    str
    requirement: dict
    eval:        dict[str, bool]
    passed:      bool

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MultiLegVerdict:
    protocol:         InstantiatedProtocol
    leg_results:      list[LegResult]
    decomp_results:   list[DecompositionResult]
    overall_verdict:  str            # GREEN | YELLOW | RED
    verdict_reasons:  list[str]      # which rule branch fired
    elapsed_seconds:  float
    recommendations:  list = dataclasses.field(default_factory=list)
    """Adaptive recommendations from adaptive_diagnostics.analyze_multi_leg_failure.
    NEVER modifies overall_verdict (rigor floor preserved); surfaces actionable
    guidance for the operator."""

    def to_dict(self) -> dict:
        return {
            "protocol":         self.protocol.to_dict(),
            "leg_results":      [r.to_dict() for r in self.leg_results],
            "decomp_results":   [d.to_dict() for d in self.decomp_results],
            "overall_verdict":  self.overall_verdict,
            "verdict_reasons":  self.verdict_reasons,
            "elapsed_seconds":  self.elapsed_seconds,
            "recommendations":  [r.to_dict() for r in self.recommendations],
        }


# ── Pass-criteria evaluation (deterministic) ────────────────────────────

_KNOWN_PASS_CRITERIA = frozenset([
    "sharpe_t_min", "deflated_sr_min", "oos_sharpe_min", "book_corr_max_abs",
    "sign_required", "sign_consistent_with_primary", "retain_ratio_vs_primary",
])


def _eval_pass_criteria(criteria: dict, gate_summary: dict,
                          primary_gate: dict | None = None) -> dict[str, bool]:
    """Evaluate each pass criterion against a gate run result.

    Args:
      criteria:      e.g. {sharpe_t_min: 3.0, deflated_sr_min: 0.9, ...}
      gate_summary:  output of run_gate (this leg)
      primary_gate:  primary leg's gate_summary (for retain_ratio checks)

    Returns: dict {criterion_name: passed_bool}
    """
    out: dict[str, bool] = {}
    sharpe = gate_summary.get("standalone_sharpe")
    sharpe_t = (sharpe * (gate_summary.get("n_months", 0) / 12.0) ** 0.5
                 if sharpe is not None and gate_summary.get("n_months") else None)
    # run_gate doesn't expose Sharpe-t directly; we compute it the same way it
    # does (sharpe * sqrt(n_years))
    # However if alpha_t_ff5umd is present, that's the more useful t-stat.

    alpha_t = gate_summary.get("alpha_t_ff5umd")
    dsr = gate_summary.get("deflated_sr")
    oos = gate_summary.get("oos_sharpe")
    book_corr = gate_summary.get("corr_with_book")

    if "sharpe_t_min" in criteria:
        # Prefer alpha-t (more robust to multi-factor exposure)
        t_used = alpha_t if alpha_t is not None else sharpe_t
        out["sharpe_t_min"] = (t_used is not None
                                  and t_used >= float(criteria["sharpe_t_min"]))

    if "deflated_sr_min" in criteria:
        out["deflated_sr_min"] = (dsr is not None
                                    and dsr >= float(criteria["deflated_sr_min"]))

    if "oos_sharpe_min" in criteria:
        out["oos_sharpe_min"] = (oos is not None
                                    and oos >= float(criteria["oos_sharpe_min"]))

    if "book_corr_max_abs" in criteria:
        if book_corr is None:
            out["book_corr_max_abs"] = True    # n/a is pass (no data to fail)
        else:
            out["book_corr_max_abs"] = (abs(book_corr)
                                          <= float(criteria["book_corr_max_abs"]))

    if "sign_required" in criteria:
        wanted = criteria["sign_required"]
        s = sharpe
        if wanted == "positive":
            out["sign_required"] = s is not None and s > 0
        elif wanted == "negative":
            out["sign_required"] = s is not None and s < 0
        else:    # either
            out["sign_required"] = s is not None

    if "sign_consistent_with_primary" in criteria and primary_gate:
        prim_s = primary_gate.get("standalone_sharpe")
        s = sharpe
        if prim_s is None or s is None:
            out["sign_consistent_with_primary"] = False
        else:
            out["sign_consistent_with_primary"] = (
                (prim_s >= 0 and s >= 0) or (prim_s < 0 and s < 0)
            )

    if "retain_ratio_vs_primary" in criteria and primary_gate:
        prim_s = primary_gate.get("standalone_sharpe")
        s = sharpe
        if prim_s is None or s is None or abs(prim_s) < 1e-9:
            out["retain_ratio_vs_primary"] = False
        else:
            ratio = s / prim_s
            out["retain_ratio_vs_primary"] = ratio >= float(
                criteria["retain_ratio_vs_primary"]
            )

    return out


def _eval_decomposition(check: DecompositionCheck,
                          primary_gate: dict | None) -> DecompositionResult:
    """Evaluate one decomposition check against the primary leg's gate result."""
    eval_dict: dict[str, bool] = {}
    if primary_gate is None:
        return DecompositionResult(
            check_id=check.id, requirement=dict(check.requirement),
            eval={}, passed=False,
        )

    req = check.requirement
    if "ff5_umd_alpha_t_min_abs" in req:
        alpha_t = primary_gate.get("alpha_t_ff5umd")
        passed = (alpha_t is not None
                   and abs(alpha_t) >= float(req["ff5_umd_alpha_t_min_abs"]))
        eval_dict["ff5_umd_alpha_t_min_abs"] = passed
        if "sign_must_match_primary" in req and req["sign_must_match_primary"]:
            prim_s = primary_gate.get("standalone_sharpe")
            sign_ok = (prim_s is not None and alpha_t is not None
                         and ((prim_s >= 0) == (alpha_t >= 0)))
            eval_dict["sign_must_match_primary"] = sign_ok

    if "pead_residual_alpha_t_min_abs" in req:
        alpha_t_pead = primary_gate.get("alpha_t_ff5umd_pead")
        passed = (alpha_t_pead is not None
                   and abs(alpha_t_pead)
                       >= float(req["pead_residual_alpha_t_min_abs"]))
        eval_dict["pead_residual_alpha_t_min_abs"] = passed

    all_passed = bool(eval_dict) and all(eval_dict.values())
    return DecompositionResult(
        check_id=check.id, requirement=dict(check.requirement),
        eval=eval_dict, passed=all_passed,
    )


# ── Verdict aggregation ─────────────────────────────────────────────────

def _aggregate_verdict(leg_results: list[LegResult],
                         decomp_results: list[DecompositionResult],
                         verdict_rule: dict) -> tuple[str, list[str]]:
    """Apply the family's verdict_rule to leg_results + decomp_results.

    Returns (verdict_str, reasons_list).
    """
    primary_pass = any(r.leg_passed for r in leg_results if r.is_primary)
    n_robustness = sum(1 for r in leg_results if not r.is_primary)
    n_robustness_pass = sum(
        1 for r in leg_results if not r.is_primary and r.leg_passed
    )
    all_decomp_pass = all(d.passed for d in decomp_results) if decomp_results else True

    state = {
        "primary_test_pass":   primary_pass,
        "n_robustness_total":  n_robustness,
        "n_robustness_pass":   n_robustness_pass,
        "all_decomposition_pass": all_decomp_pass,
    }

    def _check_rule_set(rules: list[dict]) -> tuple[bool, list[str]]:
        """All rules must pass; returns (all_pass, reasons)."""
        reasons = []
        ok = True
        for rule in (rules or []):
            for key, val in rule.items():
                if key == "primary_test_pass":
                    rule_ok = state["primary_test_pass"] == val
                elif key == "n_robustness_pass_geq":
                    rule_ok = state["n_robustness_pass"] >= int(val)
                elif key == "all_decomposition_pass":
                    rule_ok = state["all_decomposition_pass"] == val
                else:
                    reasons.append(f"unknown rule key {key!r}")
                    rule_ok = False
                if not rule_ok:
                    reasons.append(
                        f"{key}: required={val}, got={state.get(key) if key != 'n_robustness_pass_geq' else state['n_robustness_pass']}"
                    )
                    ok = False
        return ok, reasons

    green_rules = verdict_rule.get("GREEN_requires_all_of") or []
    green_ok, green_reasons = _check_rule_set(green_rules)
    if green_ok:
        return "GREEN", [f"primary_pass={primary_pass}",
                          f"robustness_pass={n_robustness_pass}/{n_robustness}",
                          f"decomp_pass={all_decomp_pass}"]

    yellow_rules = verdict_rule.get("YELLOW_requires_all_of") or []
    yellow_ok, yellow_reasons = _check_rule_set(yellow_rules)
    if yellow_ok:
        return "YELLOW", [f"primary_pass={primary_pass}",
                            f"robustness_pass={n_robustness_pass}/{n_robustness}",
                            f"GREEN_failed: {green_reasons}"]

    return "RED", [f"primary_pass={primary_pass}",
                     f"robustness_pass={n_robustness_pass}/{n_robustness}",
                     f"GREEN_failed: {green_reasons}",
                     f"YELLOW_failed: {yellow_reasons}"]


# ── Per-leg execution ───────────────────────────────────────────────────

def _execute_leg(
    leg: ResolvedLeg,
    proposal: dict,
    data_kwargs: dict[str, Any],
    primary_gate: dict | None,
    protocol_hash: str,
    pead_control: bool,
) -> LegResult:
    """Run ONE leg: build DSL proposal with leg's binding, run DSL, run gate."""
    from engine.research.strategy_dsl_runner import run_proposal as dsl_run
    from engine.research.pipeline import run_gate

    leg_proposal = copy.deepcopy(proposal)
    et = leg_proposal.get("execution_template") or {}
    if et:
        et["binding"] = leg.binding
        leg_proposal["execution_template"] = et

    try:
        returns = dsl_run(leg_proposal, **data_kwargs)
    except Exception as exc:
        logger.warning("leg %s DSL failed: %s", leg.id, exc)
        return LegResult(
            leg_id=leg.id, is_primary=leg.is_primary,
            gate_summary=None, pass_criteria_eval={}, leg_passed=False,
            error=f"dsl: {exc}",
        )

    if returns is None or len(returns.dropna()) < 24:
        return LegResult(
            leg_id=leg.id, is_primary=leg.is_primary,
            gate_summary=None, pass_criteria_eval={}, leg_passed=False,
            error=f"insufficient months: {len(returns.dropna()) if returns is not None else 0}",
        )

    # Sample-window mask (v1: simple date slice)
    sample_mask = ((returns.index >= leg.sample_start)
                    & (returns.index <= leg.sample_end))
    sliced = returns[sample_mask].dropna()
    if len(sliced) < 24:
        return LegResult(
            leg_id=leg.id, is_primary=leg.is_primary,
            gate_summary=None, pass_criteria_eval={}, leg_passed=False,
            error=f"sliced sample too short: {len(sliced)} months",
        )

    leg_gate_name = f"{proposal.get('mechanism_id', 'x')}_leg_{leg.id}_{protocol_hash}"
    try:
        gate_summary = run_gate(
            sliced, name=leg_gate_name,
            mechanism=f"{proposal.get('mechanism_id', 'x')} / leg {leg.id}",
            log=False, pead_control=pead_control,
        )
    except Exception as exc:
        logger.warning("leg %s run_gate failed: %s", leg.id, exc)
        return LegResult(
            leg_id=leg.id, is_primary=leg.is_primary,
            gate_summary=None, pass_criteria_eval={}, leg_passed=False,
            error=f"run_gate: {exc}",
        )

    criteria_eval = _eval_pass_criteria(leg.pass_criteria, gate_summary,
                                          primary_gate=primary_gate)
    leg_passed = bool(criteria_eval) and all(criteria_eval.values())
    return LegResult(
        leg_id=leg.id, is_primary=leg.is_primary,
        gate_summary=gate_summary, pass_criteria_eval=criteria_eval,
        leg_passed=leg_passed,
    )


# ── Phase 6c: auto-acquire helper ───────────────────────────────────────

def _auto_acquire_data(
    protocol: InstantiatedProtocol,
    proposal: dict,
    *,
    universe: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Fetch all required_data tokens for this protocol via orchestrator,
    adapt via dsl_adapter, return data_kwargs ready for DSL.

    Computes the FULL sample window across all legs (min start, max end)
    so the fetch happens ONCE; each leg slices its own subset downstream.

    Returns:
      (data_kwargs, failures) — failures is empty list on success
    """
    from engine.research.protocols.protocol_designer import load_mechanism
    from engine.data.orchestrator import assemble_data_kwargs
    from engine.data.dsl_adapter import assemble_dsl_kwargs

    mechanism_id = proposal.get("mechanism_id")
    if not mechanism_id:
        return {}, ["proposal lacks mechanism_id"]
    try:
        mechanism = load_mechanism(mechanism_id)
    except FileNotFoundError as exc:
        return {}, [f"library mechanism not found: {exc}"]
    required_data = mechanism.get("required_data") or []
    if not required_data:
        return {}, ["mechanism has no required_data declared"]

    # Full sample window across all legs
    full_start = min(leg.sample_start for leg in protocol.legs)
    full_end = max(leg.sample_end for leg in protocol.legs)

    # Fetch all tokens
    token_dfs, results = assemble_data_kwargs(
        required_data, start=full_start, end=full_end, universe=universe,
    )
    failures = []
    for r in results:
        if not r.success:
            failures.append(
                f"token {r.token!r}: {'; '.join(r.quality_caveats or ['unknown error'])}"
            )
    if failures:
        return {}, failures

    # Adapt to DSL shape
    template_id = (proposal.get("execution_template") or {}).get("template_id")
    data_kwargs = assemble_dsl_kwargs(token_dfs, template_id=template_id)
    return data_kwargs, []


# ── Main public API ─────────────────────────────────────────────────────

def execute_protocol(
    protocol: InstantiatedProtocol,
    proposal: dict,
    *,
    data_kwargs: dict[str, Any] | None = None,
    pead_control: bool = True,
    auto_acquire: bool = False,
    universe: str | None = None,
) -> MultiLegVerdict:
    """Run all legs + decomposition checks + aggregate verdict.

    Args:
      protocol:    InstantiatedProtocol from designer (immutable)
      proposal:    must contain execution_template
      data_kwargs: forwarded to DSL runner (price_panel, return_panel, etc.)
                    If None AND auto_acquire=True, fetched via orchestrator.
      pead_control: pass-through to run_gate
      auto_acquire: Phase 6c wire-up. When True AND data_kwargs is None,
                     the executor looks up mechanism's required_data, fetches
                     all tokens via engine.data.orchestrator (cache + fallback),
                     adapts via dsl_adapter (long→wide), then runs legs.
                     ON FAILURE of any required token, the protocol returns
                     RED with structured failure detail (NEVER silent synth).
      universe:    forwarded to fetch_token for universe filtering

    Returns: MultiLegVerdict with overall_verdict GREEN | YELLOW | RED.
    """
    import time
    t0 = time.time()

    # Phase 6c: auto-acquire data via orchestrator
    acquisition_failures: list[str] = []
    if data_kwargs is None and auto_acquire:
        data_kwargs, acquisition_failures = _auto_acquire_data(
            protocol, proposal, universe=universe,
        )
        if acquisition_failures:
            return MultiLegVerdict(
                protocol=protocol, leg_results=[], decomp_results=[],
                overall_verdict="RED",
                verdict_reasons=[
                    "data acquisition failed for required tokens",
                    *acquisition_failures,
                ],
                elapsed_seconds=time.time() - t0,
            )
    data_kwargs = data_kwargs or {}

    leg_results: list[LegResult] = []
    primary_gate: dict | None = None

    # Run primary first (so robustness legs can reference it)
    primary_leg = next((leg for leg in protocol.legs if leg.is_primary), None)
    if primary_leg is None:
        return MultiLegVerdict(
            protocol=protocol, leg_results=[], decomp_results=[],
            overall_verdict="RED",
            verdict_reasons=["no primary_test leg in protocol"],
            elapsed_seconds=time.time() - t0,
        )

    primary_result = _execute_leg(primary_leg, proposal, data_kwargs,
                                    primary_gate=None,
                                    protocol_hash=protocol.protocol_hash,
                                    pead_control=pead_control)
    leg_results.append(primary_result)
    if primary_result.gate_summary:
        primary_gate = primary_result.gate_summary

    # Run remaining legs
    for leg in protocol.legs:
        if leg.is_primary:
            continue
        leg_results.append(_execute_leg(leg, proposal, data_kwargs,
                                         primary_gate=primary_gate,
                                         protocol_hash=protocol.protocol_hash,
                                         pead_control=pead_control))

    # Decomposition checks (on primary leg)
    decomp_results = [
        _eval_decomposition(d, primary_gate) for d in protocol.decomposition_checks
    ]

    overall, reasons = _aggregate_verdict(
        leg_results, decomp_results, protocol.verdict_rule
    )

    verdict_obj = MultiLegVerdict(
        protocol=protocol, leg_results=leg_results,
        decomp_results=decomp_results, overall_verdict=overall,
        verdict_reasons=reasons, elapsed_seconds=time.time() - t0,
    )

    # Phase 6c adaptive: surface actionable recommendations when failure
    # patterns are detectable. NEVER modifies overall_verdict.
    try:
        from engine.research.protocols.adaptive_diagnostics import (
            analyze_multi_leg_failure,
        )
        # Build context dict for detectors
        binding = (proposal.get("execution_template") or {}).get("binding", {})
        template_id = (proposal.get("execution_template") or {}).get("template_id")
        # Sample total months: from primary leg's bounds
        primary_leg = next((l for l in protocol.legs if l.is_primary), None)
        sample_total_months = None
        if primary_leg:
            import datetime as _dt
            try:
                s = _dt.date.fromisoformat(primary_leg.sample_start)
                e = _dt.date.fromisoformat(primary_leg.sample_end)
                sample_total_months = int((e - s).days / 30.44)
            except Exception:
                pass
        # Universe size: introspect data_kwargs if equity-style panel
        universe_size = None
        for v in data_kwargs.values():
            if hasattr(v, "shape") and len(getattr(v, "shape", ())) >= 2:
                universe_size = v.shape[1]
                break
        # template warmup
        from engine.research.protocols.protocol_designer import compute_template_warmup
        try:
            warmup_months = compute_template_warmup(template_id, binding)
        except Exception:
            warmup_months = None

        ctx = {
            "template_id":             template_id,
            "binding":                 binding,
            "sample_total_months":     sample_total_months,
            "template_warmup_months":  warmup_months,
            "universe_size":           universe_size,
        }
        verdict_obj.recommendations = analyze_multi_leg_failure(verdict_obj,
                                                                  context=ctx)
    except Exception as exc:
        logger.warning("adaptive diagnostics failed: %s", exc)

    return verdict_obj
