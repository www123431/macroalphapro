"""engine/research/outcome_ledger.py — Phase 4d: persistent ledger of
L4 discovery loop iterations.

Each row is one full loop: architect proposed → critics critiqued →
(optionally) candidate_pipeline_v2 ran → outcome recorded. This is
the AUTHORITATIVE record for L4 calibration + zero-LLM-in-decision
audit — without it, the loop is opaque even with verbose logs.

Distinct from data/research/override_ledger.jsonl (which records
HUMAN-initiated graveyard overrides + their outcomes). Both live
under data/research/ and are read by the Cockpit Outcomes tab.

Schema (one JSON object per line):

  {
    "ts":          "2026-06-01T12:34:56Z",
    "iteration_id": "iter-<12hex>",
    "workflow_id": "l4-<12hex>",
    "stage":       "completed",
    "proposal":    {title, family, parent_family, proposed_role, ...},
    "council":     {consensus, rationale, n_critics, run_id},
    "pipeline":    {
      "ran": bool,
      "final_decision": "PROMOTE_TO_GATE | HARD_REJECT | ...",
      "rationale": "...",
      "n_steps": int,
      "candidate_returns_path": "..."
    } | null,
    "verdict_alignment": "agree | council_wrong | pipeline_wrong | not_runnable",
    "elapsed_s":   float
  }

The verdict_alignment field is the CALIBRATION SIGNAL — over many
iterations, "agree" rate is the council's empirical track record.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
L4_LEDGER_PATH = REPO_ROOT / "data" / "research" / "l4_iterations.jsonl"


def _classify_alignment(
    council_consensus: Optional[str],
    pipeline_decision: Optional[str],
) -> str:
    """Compare council verdict vs pipeline empirical verdict.

    The calibration signal — over many iterations we expect the
    council to align with the pipeline (the deterministic empirical
    layer) most of the time. Persistent disagreement = the council
    is mis-calibrated (system prompts / tool surfaces need work).
    """
    if council_consensus is None or pipeline_decision is None:
        return "not_runnable"
    promotive = {"PROMOTE_TO_GATE", "PROMOTE_AS_REPLACEMENT"}
    rejective = {"HARD_REJECT", "SOFT_REJECT"}
    borderline = {"BORDERLINE_REVIEW"}

    if council_consensus == "APPROVE":
        if pipeline_decision in promotive:
            return "agree"
        if pipeline_decision in rejective:
            return "council_wrong"
        return "agree"  # borderline ≈ approve with caveats
    if council_consensus == "REJECT":
        if pipeline_decision in rejective:
            return "agree"
        if pipeline_decision in promotive:
            return "council_wrong"
        return "agree"
    if council_consensus == "NEEDS_REVISION":
        # NEEDS_REVISION means "council uncertain" — pipeline either
        # confirms (BORDERLINE) or pushes one way
        if pipeline_decision in borderline:
            return "agree"
        return "pipeline_resolved"  # neutral — pipeline answered the question
    return "not_runnable"


def append_l4_iteration(
    *,
    workflow_id: str,
    proposal: dict,
    council: dict,
    pipeline_report: Optional[dict] = None,
    elapsed_s: float = 0.0,
    human_override_verdict: Optional[str] = None,
) -> str:
    """Append one L4 iteration row. Returns the iteration_id.

    Best-effort persistence: disk failure logged but never raised,
    so a ledger outage cannot break the L4 loop itself."""
    iteration_id = f"iter-{uuid.uuid4().hex[:12]}"
    try:
        L4_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        pipeline_block: Optional[dict]
        if pipeline_report is None:
            pipeline_block = None
        else:
            pipeline_block = {
                "ran":            bool(pipeline_report.get("ran", True)),
                "final_decision": pipeline_report.get("final_decision"),
                "rationale":      (pipeline_report.get("rationale") or "")[:1000],
                "n_steps":        len(pipeline_report.get("step_results") or []),
                "candidate_returns_path": pipeline_report.get("candidate_returns_path"),
            }
        # Effective consensus = override if present, else council's
        effective_consensus = human_override_verdict or council.get("consensus")
        row = {
            "ts":            _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "iteration_id":  iteration_id,
            "workflow_id":   workflow_id,
            "stage":         "completed",
            "proposal":      proposal,
            "council":       {
                "consensus":   council.get("consensus"),
                "rationale":   (council.get("rationale") or "")[:1000],
                "n_critics":   len(council.get("verdicts") or []),
                "run_id":      council.get("run_id"),
            },
            "human_override": (
                {"verdict": human_override_verdict}
                if human_override_verdict else None
            ),
            "effective_consensus": effective_consensus,
            "pipeline":      pipeline_block,
            "verdict_alignment": _classify_alignment(
                effective_consensus,
                (pipeline_report or {}).get("final_decision"),
            ),
            "elapsed_s":     round(elapsed_s, 2),
        }
        with L4_LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # Frontier A (2026-06-01): also emit per-critic calibration rows
        # so we can measure individual critic accuracy + pairwise
        # redundancy + marginal information gain. Best-effort: failure
        # to write critic rows must NOT crash the L4 loop.
        try:
            from engine.research.critic_calibration import (
                append_critic_calibration_rows,
            )
            append_critic_calibration_rows(
                iteration_id=iteration_id,
                council=council,
                proposal=proposal,
                pipeline_report=pipeline_report,
            )
        except Exception:
            logger.exception("critic_calibration emission failed (non-fatal)")
        # Vector RAG (2026-06-02): refresh the relevant embedding indices
        # incrementally so /ask retrieval stays current. build_index() is
        # hash-deduped — only new rows get embedded. Best-effort: import
        # failures (sentence-transformers not installed in some envs) or
        # mirror-blocked HF downloads must NOT crash the L4 loop.
        try:
            from engine.research import embeddings as _E
            for ledger in ("l4_iterations", "council_runs"):
                try:
                    _E.build_index(ledger)
                except Exception:
                    logger.warning("embedding refresh %s failed", ledger,
                                    exc_info=True)
        except ImportError:
            pass   # sentence-transformers not installed; semantic retrieval disabled
    except Exception:
        logger.exception("L4 ledger append failed (non-fatal)")
    return iteration_id


def read_l4_iterations(
    limit: int = 50,
    consensus: Optional[str] = None,
    alignment: Optional[str] = None,
) -> list[dict]:
    """Read recent L4 iterations newest-first. Filters by council
    consensus or verdict_alignment for calibration analysis."""
    if not L4_LEDGER_PATH.is_file():
        return []
    out: list[dict] = []
    with L4_LEDGER_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if consensus and (r.get("council") or {}).get("consensus") != consensus:
                continue
            if alignment and r.get("verdict_alignment") != alignment:
                continue
            out.append(r)
    out.reverse()
    return out[: max(1, limit)]


def read_iteration_by_id(iteration_id: str) -> Optional[dict]:
    """Drill-down by id."""
    if not L4_LEDGER_PATH.is_file():
        return None
    with L4_LEDGER_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("iteration_id") == iteration_id:
                return r
    return None


def calibration_summary(limit: int = 200) -> dict:
    """Compute the calibration KPI: council vs pipeline agreement rate.

    Returns counts + percentage. Used by the Cockpit KPI strip as
    "council calibration" — over time this is the ONE number that
    tells you if your council is improving."""
    rows = read_l4_iterations(limit=limit)
    counts: dict[str, int] = {}
    for r in rows:
        a = r.get("verdict_alignment") or "unknown"
        counts[a] = counts.get(a, 0) + 1
    runnable = sum(c for a, c in counts.items()
                   if a not in ("not_runnable",))
    agree = counts.get("agree", 0)
    return {
        "n_total":      len(rows),
        "n_runnable":   runnable,
        "n_agree":      agree,
        "agree_pct":    round(agree / runnable * 100, 1) if runnable else None,
        "by_alignment": counts,
    }
