"""engine/agents/research_diagnostician/tools.py — deterministic tools the
LLM diagnostician calls during multi-turn tool-use.

Each tool is:
  - DETERMINISTIC (same input → same output)
  - Read-only (no state mutation)
  - Bounded latency (< 2s)
  - Has JSON schema (Anthropic tool format)
  - Has unit tests

Per the pre-impl checklist B (Tool quality contract) in
[[project-agentic-ai-real-architecture-2026-05-29]]:
> Bad tools poison the loop — LLM uses wrong data → diagnoses wrong → user
> trusts wrong answer

So tools are kept TIGHT. Each tool returns structured JSON the LLM can read.
No prose, no opinions in tool output — those are the LLM's job.

Tools exposed (6):
  T1 fetch_gate_evidence(candidate_name)         — pulls gate_runs entry
  T2 find_similar_candidates(candidate_name)     — uses Knowledge Graph
  T3 check_deployed_overlap(candidate_name)      — uses Knowledge Graph
  T4 subperiod_analysis(candidate_name, n)       — N-bucket Sharpe split
  T5 sample_stress_coverage(candidate_name)      — canonical stress periods
  T6 fetch_sleeve_health_history(sleeve_name, n) — decay sentinel artifacts
"""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.research.knowledge_graph import (
    KnowledgeGraph,
    build_graph,
    deployed_overlap_check,
    similar_candidates,
    _sample_coverage,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
GATE_LEDGER_PATH = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
DECAY_ARTIFACTS_DIR = REPO_ROOT / "data" / "decay_sentinel"


# ── Anthropic tool schemas (re-usable for LLM calls) ─────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "fetch_gate_evidence",
        "description": "Look up the strict-gate evidence for a candidate by exact name. "
                       "Returns the full gate_runs.jsonl entry (Sharpe, alpha-t, deflated SR, "
                       "OOS Sharpe, book correlation, verdict, sample window, etc).",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {
                    "type": "string",
                    "description": "Exact name of the candidate (e.g. 'quality_novymarx_2013_v1')"
                }
            },
            "required": ["candidate_name"]
        }
    },
    {
        "name": "find_similar_candidates",
        "description": "Find PRIOR candidates that share mechanism family (direct or parent rollup) "
                       "with this one. Use to detect 'this is a cousin of a previously-tested "
                       "mechanism' — informs whether the failure mode is consistent with prior REDs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"}
            },
            "required": ["candidate_name"]
        }
    },
    {
        "name": "check_deployed_overlap",
        "description": "Check whether this candidate shares mechanism family with any DEPLOYED "
                       "sleeve (direct or parent level). If parent_only overlap exists with a "
                       "high-weight sleeve, the candidate may be a 'cousin in disguise' of "
                       "existing book exposure — informs whether RED is structural or coincidental.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"}
            },
            "required": ["candidate_name"]
        }
    },
    {
        "name": "sample_stress_coverage",
        "description": "Return which canonical stress periods (2008 GFC / 2018 Vol-mageddon / "
                       "2020 COVID / 2022 rate-crash / etc) are inside vs outside this candidate's "
                       "test sample. Sample missing major stress periods is a red flag for "
                       "extrapolating Sharpe robustness.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"}
            },
            "required": ["candidate_name"]
        }
    },
    {
        "name": "fetch_sleeve_health_history",
        "description": "Read the last N daily decay_sentinel artifacts for a DEPLOYED SLEEVE "
                       "(e.g. equity_book, carry_book, tsmom_book) — returns time-ordered "
                       "rolling Sharpe / decay ratio / signal IC / structural_decay flag. Use "
                       "when diagnosing why a deployed sleeve is degrading (separate from new "
                       "candidate evaluation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "sleeve_name": {"type": "string"},
                "n_days":      {"type": "integer", "default": 7}
            },
            "required": ["sleeve_name"]
        }
    },
    {
        "name": "subperiod_analysis",
        "description": "(Optional) Split a candidate's gate evidence into per-subperiod Sharpe "
                       "to detect regime-dependent alpha. Returns subperiod count from the "
                       "gate_runs metadata; cannot recompute from raw returns without re-running.",
        "input_schema": {
            "type": "object",
            "properties": {
                "candidate_name": {"type": "string"}
            },
            "required": ["candidate_name"]
        }
    },
]


# ── Implementations ──────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class ToolResult:
    name:    str
    success: bool
    payload: dict
    error:   str | None = None

    def to_dict(self) -> dict:
        if self.error:
            return {"name": self.name, "success": False, "error": self.error}
        return {"name": self.name, "success": True, "payload": self.payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


def _load_gate_ledger() -> list[dict]:
    if not GATE_LEDGER_PATH.exists():
        return []
    return [json.loads(l) for l in GATE_LEDGER_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def _find_entry_by_name(name: str, ledger: list[dict] | None = None) -> dict | None:
    """Return the LATEST gate_runs entry whose name matches (case-insensitive
    exact match preferred; falls back to substring match)."""
    rows = ledger if ledger is not None else _load_gate_ledger()
    name_l = name.lower()
    matches = [r for r in rows if (r.get("name") or "").lower() == name_l]
    if not matches:
        matches = [r for r in rows if name_l in (r.get("name") or "").lower()]
    if not matches:
        return None
    # Return latest (last entry wins if multiple)
    return matches[-1]


# Tool 1
def fetch_gate_evidence(candidate_name: str) -> ToolResult:
    entry = _find_entry_by_name(candidate_name)
    if entry is None:
        return ToolResult("fetch_gate_evidence", False, {},
                           error=f"no gate_runs entry matches {candidate_name!r}")
    # Strip the bars block (LLM doesn't need to re-read thresholds)
    out = {k: v for k, v in entry.items() if k != "bars"}
    return ToolResult("fetch_gate_evidence", True, out)


# Tool 2
def find_similar_candidates_t(candidate_name: str,
                                graph: KnowledgeGraph | None = None) -> ToolResult:
    g = graph or build_graph()
    sims = similar_candidates(g, candidate_name)
    if not sims:
        return ToolResult("find_similar_candidates", True,
                           {"candidate_name": candidate_name, "n_similar": 0,
                            "similar": [],
                            "note": "no shared mechanism family or parent overlap with prior candidates"})
    similar_with_context = []
    for s in sims:
        # Pull the similar candidate's verdict
        verdict_node = g.neighbors(s, "received")
        verdict = verdict_node[0].id if verdict_node else "UNKNOWN"
        # Pull shared families from edge attrs
        out_edges = g.out_edges(
            g.nodes.get(("Candidate", candidate_name)), "similar_to")
        shared = next((dict(e.attrs).get("shared") for e in out_edges if e.target is s), "")
        similar_with_context.append({
            "name": s.id,
            "verdict": verdict,
            "shared_families": shared,
        })
    return ToolResult("find_similar_candidates", True,
                       {"candidate_name": candidate_name,
                        "n_similar": len(similar_with_context),
                        "similar": similar_with_context})


# Tool 3
def check_deployed_overlap_t(candidate_name: str,
                              graph: KnowledgeGraph | None = None) -> ToolResult:
    g = graph or build_graph()
    overlap = deployed_overlap_check(g, candidate_name)
    if "error" in overlap:
        return ToolResult("check_deployed_overlap", False, {},
                           error=overlap["error"])
    if not overlap:
        return ToolResult("check_deployed_overlap", True,
                           {"candidate_name": candidate_name,
                            "n_overlapping_sleeves": 0,
                            "overlap_by_sleeve": {},
                            "note": "no overlap with deployed sleeves — mechanism is structurally new"})
    return ToolResult("check_deployed_overlap", True,
                       {"candidate_name": candidate_name,
                        "n_overlapping_sleeves": len(overlap),
                        "overlap_by_sleeve": overlap})


# Tool 4
def sample_stress_coverage_t(candidate_name: str) -> ToolResult:
    entry = _find_entry_by_name(candidate_name)
    if entry is None:
        return ToolResult("sample_stress_coverage", False, {},
                           error=f"no gate_runs entry for {candidate_name!r}")
    start = entry.get("sample_start")
    end = entry.get("sample_end")
    n_months = entry.get("n_months")
    cov = _sample_coverage(start, end, n_months)
    return ToolResult("sample_stress_coverage", True,
                       {"candidate_name": candidate_name,
                        "sample_start": cov.start,
                        "sample_end": cov.end,
                        "n_months": cov.n_months,
                        "stress_covered": list(cov.stress_covered),
                        "stress_missed": list(cov.stress_missed),
                        "coverage_ratio": round(cov.coverage_ratio, 3)})


# Tool 5
def subperiod_analysis_t(candidate_name: str) -> ToolResult:
    """v1: returns IS vs OOS Sharpe split from gate_runs metadata (cannot
    recompute without re-running the strategy)."""
    entry = _find_entry_by_name(candidate_name)
    if entry is None:
        return ToolResult("subperiod_analysis", False, {},
                           error=f"no gate_runs entry for {candidate_name!r}")
    sharpe = entry.get("standalone_sharpe")
    oos = entry.get("oos_sharpe")
    if sharpe is None or oos is None:
        return ToolResult("subperiod_analysis", True,
                           {"candidate_name": candidate_name,
                            "note": "subperiod data unavailable (gate_runs entry missing Sharpe metrics)"})
    # We have full-sample and 2nd-half (oos). Roughly compute 1st-half.
    n_months = entry.get("n_months") or 0
    if n_months > 24:
        # If OOS is the 2nd half, and full is the avg, 1st half ≈ 2 * full - oos (rough)
        first_half_approx = round(2 * sharpe - oos, 3)
    else:
        first_half_approx = None
    return ToolResult("subperiod_analysis", True,
                       {"candidate_name": candidate_name,
                        "n_months": n_months,
                        "full_sample_sharpe": sharpe,
                        "second_half_sharpe": oos,
                        "first_half_sharpe_approx": first_half_approx,
                        "regime_diff_approx": round(oos - first_half_approx, 3)
                                              if first_half_approx is not None else None,
                        "note": "first_half_sharpe is approximate (computed from full vs 2nd-half); "
                                 "for exact value, re-run the strategy"})


# ── Dispatcher (the LLM tool-use loop calls this) ────────────────────────────

def fetch_sleeve_health_history_t(sleeve_name: str, n_days: int = 7) -> ToolResult:
    """Read the last N daily decay_sentinel artifacts for one sleeve. Returns
    time-ordered list of (date, status, rolling_sharpe, decay_ratio, signal_ic)."""
    if not DECAY_ARTIFACTS_DIR.exists():
        return ToolResult("fetch_sleeve_health_history", False, {},
                          error="no decay_sentinel artifacts directory")
    files = sorted(DECAY_ARTIFACTS_DIR.glob("decay_sentinel_*.json"))[-n_days:]
    if not files:
        return ToolResult("fetch_sleeve_health_history", False, {},
                          error="no decay_sentinel artifacts found")
    rows = []
    for fp in files:
        try:
            artifact = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        as_of = artifact.get("as_of") or fp.stem.replace("decay_sentinel_", "")
        mechs = artifact.get("mechanisms", {})
        if sleeve_name not in mechs:
            continue
        m = mechs[sleeve_name]
        rows.append({
            "as_of":          as_of,
            "rolling_sharpe": m.get("rolling_sharpe"),
            "rolling_t":      m.get("rolling_t"),
            "decay_ratio":    m.get("decay_ratio"),
            "signal_ic":      m.get("signal_ic"),
            "structural_decay": m.get("structural_decay"),
        })
    if not rows:
        return ToolResult("fetch_sleeve_health_history", True,
                          {"sleeve_name": sleeve_name,
                            "n_days_found": 0,
                            "history": [],
                            "note": f"sleeve {sleeve_name!r} not in recent decay artifacts"})
    return ToolResult("fetch_sleeve_health_history", True,
                       {"sleeve_name": sleeve_name,
                        "n_days_found": len(rows),
                        "history": rows})


_TOOL_DISPATCH = {
    "fetch_gate_evidence":         lambda **kw: fetch_gate_evidence(kw["candidate_name"]),
    "find_similar_candidates":     lambda **kw: find_similar_candidates_t(kw["candidate_name"]),
    "check_deployed_overlap":      lambda **kw: check_deployed_overlap_t(kw["candidate_name"]),
    "sample_stress_coverage":      lambda **kw: sample_stress_coverage_t(kw["candidate_name"]),
    "subperiod_analysis":          lambda **kw: subperiod_analysis_t(kw["candidate_name"]),
    "fetch_sleeve_health_history": lambda **kw: fetch_sleeve_health_history_t(
        kw["sleeve_name"], kw.get("n_days", 7)),
}


def execute_tool(name: str, **kwargs) -> ToolResult:
    """Dispatch by tool name. Unknown tool → error ToolResult (no exception
    bubbled up — protects the LLM loop)."""
    if name not in _TOOL_DISPATCH:
        return ToolResult(name, False, {}, error=f"unknown tool {name!r}")
    try:
        return _TOOL_DISPATCH[name](**kwargs)
    except Exception as exc:
        logger.warning("tool %s failed: %s", name, exc)
        return ToolResult(name, False, {}, error=f"{type(exc).__name__}: {exc}")
