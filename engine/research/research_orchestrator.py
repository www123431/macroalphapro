"""engine/research/research_orchestrator.py — Phase B chain integration.

Orchestrates the full per-proposal lifecycle:

  proposal (from hypothesis_generator or manual)
      ↓
  Layer 3 DSL runner → returns Series
      ↓
  run_gate → verdict (RED/YELLOW/GREEN)
      ↓
  library_writer updates our_observed (wired in pipeline.py, NEW1)
      ↓
  IF verdict ∈ {RED, YELLOW}:
      diagnose() → causal narrative (LLM tool-use)
      propose_mutation() → optional single-parameter variant (whitelist)

Doctrine:
- OPT-IN orchestration. run_gate alone does NOT auto-diagnose to avoid
  surprising the user with $0.13/call costs. Call run_full_chain explicitly.
- Each step is INDEPENDENT — failure in one step does NOT abort the rest.
- Logging is centralized: every step writes its own ledger;
  research_orchestrator_log.jsonl summarizes the chain.

The chain produces a structured ChainResult so the user can see:
- proposal info
- gate verdict
- diagnostic summary (if triggered)
- mutation proposal (if triggered)
- total cost / latency

Doctrine for Phase C cron compatibility:
- run_full_chain is callable by both manual user workflow AND scheduled cron
- All side effects ledgered (so cron re-runs are auditable)
- Idempotency check on gate_run already done? skip diagnose? — not done in v1;
  v1 always re-runs diagnose if verdict ≠ GREEN
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAIN_LOG = REPO_ROOT / "data" / "research" / "research_orchestrator_log.jsonl"


@dataclasses.dataclass
class ChainResult:
    proposal:        dict | None
    gate_verdict:    str | None        # RED | YELLOW | GREEN | None
    gate_summary:    dict | None       # full gate run entry
    diagnosis:       dict | None       # from diagnose()
    mutation:        dict | None       # from propose_mutation()
    steps_executed: list[str]          # ordered list of executed step names
    steps_failed:   list[dict]         # {step, error}
    cost_usd_total: float
    elapsed_seconds: float
    ts:              str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def run_full_chain(
    proposal: dict,
    *,
    data_kwargs: dict[str, Any] | None = None,
    use_llm_diagnose: bool = True,
    use_llm_mutation: bool = True,
    log: bool = True,
    gate_name: str | None = None,
    gate_mechanism: str | None = None,
    gate_pead_control: bool = True,
) -> ChainResult:
    """Run the full proposal → gate → diagnose → mutation chain.

    Args:
      proposal:           hypothesis_generator output (must have
                            execution_template field)
      data_kwargs:        forwarded to DSL runner (price_panel, return_panel, etc.)
      use_llm_diagnose:   if True, diagnose uses Anthropic; else deterministic
      use_llm_mutation:   if True, mutation proposer uses Anthropic
      log:                if True, append to ledger
      gate_name:          name= passed to run_gate; defaults to mechanism_id
      gate_mechanism:     mechanism= passed to run_gate; defaults to YAML field
      gate_pead_control:  pass-through to run_gate

    Returns: ChainResult — never raises (failures collected in steps_failed)
    """
    t0 = time.time()
    start_ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    data_kwargs = data_kwargs or {}
    steps_executed: list[str] = []
    steps_failed: list[dict] = []
    cost_usd_total = 0.0

    gate_summary = None
    gate_verdict = None
    diagnosis = None
    mutation = None

    name = gate_name or proposal.get("mechanism_id") or "unknown_candidate"
    mechanism = gate_mechanism or proposal.get("justification") or ""

    # Step 1: DSL → returns Series
    try:
        from engine.research.strategy_dsl_runner import run_proposal as dsl_run
        returns = dsl_run(proposal, **data_kwargs)
        steps_executed.append("dsl_runner")
    except Exception as exc:
        logger.warning("DSL runner failed: %s", exc)
        steps_failed.append({"step": "dsl_runner", "error": str(exc)})
        return _finalize(proposal, None, None, None, None,
                          steps_executed, steps_failed, cost_usd_total,
                          time.time() - t0, start_ts, log)

    if returns is None or len(returns.dropna()) < 24:
        steps_failed.append({"step": "dsl_runner",
                              "error": f"insufficient non-NaN months: "
                                        f"{len(returns.dropna()) if returns is not None else 0}"})
        return _finalize(proposal, None, None, None, None,
                          steps_executed, steps_failed, cost_usd_total,
                          time.time() - t0, start_ts, log)

    # Step 2: run_gate (also auto-triggers library_writer via NEW1 wire)
    try:
        from engine.research.pipeline import run_gate
        gate_summary = run_gate(returns.dropna(), name=name,
                                  mechanism=mechanism, log=log,
                                  pead_control=gate_pead_control)
        gate_verdict = gate_summary.get("verdict")
        steps_executed.append("run_gate")
    except Exception as exc:
        logger.warning("run_gate failed: %s", exc)
        steps_failed.append({"step": "run_gate", "error": str(exc)})
        return _finalize(proposal, None, None, None, None,
                          steps_executed, steps_failed, cost_usd_total,
                          time.time() - t0, start_ts, log)

    # Step 3: diagnose (only if RED/YELLOW)
    if gate_verdict in ("RED", "YELLOW"):
        try:
            from engine.agents.research_diagnostician.diagnostician import diagnose
            diagnosis = diagnose(name, use_llm=use_llm_diagnose, log=log)
            steps_executed.append("diagnose")
            cost_usd_total += float(diagnosis.get("cost_usd") or 0.0)
        except Exception as exc:
            logger.warning("diagnose failed: %s", exc)
            steps_failed.append({"step": "diagnose", "error": str(exc)})

        # Step 4: propose mutation (only after diagnose lands)
        if diagnosis is not None:
            try:
                from engine.research.mutation_proposer import propose_mutation
                mutation = propose_mutation(name, use_llm=use_llm_mutation, log=log)
                steps_executed.append("mutation_proposer")
                cost_usd_total += float(mutation.get("cost_usd") or 0.0)
            except Exception as exc:
                logger.warning("mutation_proposer failed: %s", exc)
                steps_failed.append({"step": "mutation_proposer", "error": str(exc)})

    return _finalize(proposal, gate_verdict, gate_summary, diagnosis, mutation,
                      steps_executed, steps_failed, cost_usd_total,
                      time.time() - t0, start_ts, log)


def _finalize(proposal: dict | None,
                gate_verdict: str | None,
                gate_summary: dict | None,
                diagnosis: dict | None,
                mutation: dict | None,
                steps_executed: list[str],
                steps_failed: list[dict],
                cost_usd_total: float,
                elapsed: float,
                start_ts: str,
                log: bool) -> ChainResult:
    """Build the ChainResult + append to ledger."""
    result = ChainResult(
        proposal=        proposal,
        gate_verdict=    gate_verdict,
        gate_summary=    gate_summary,
        diagnosis=       diagnosis,
        mutation=        mutation,
        steps_executed=  steps_executed,
        steps_failed=    steps_failed,
        cost_usd_total=  round(cost_usd_total, 4),
        elapsed_seconds= round(elapsed, 2),
        ts=              start_ts,
    )
    if log:
        CHAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CHAIN_LOG.open("a", encoding="utf-8") as f:
            # Trim heavy fields for the chain log; full info still in
            # gate_runs.jsonl / diagnostic_reports.jsonl / mutation_proposals.jsonl
            log_entry = {
                "ts":             start_ts,
                "candidate":      (proposal or {}).get("mechanism_id"),
                "gate_verdict":   gate_verdict,
                "steps_executed": steps_executed,
                "n_steps_failed": len(steps_failed),
                "cost_usd_total": result.cost_usd_total,
                "elapsed_seconds": result.elapsed_seconds,
            }
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    return result


def read_chain_log(limit: int = 50) -> list[dict]:
    if not CHAIN_LOG.exists():
        return []
    rows = [json.loads(l) for l in CHAIN_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]
