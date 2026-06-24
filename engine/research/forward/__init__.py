"""engine.research.forward — FORWARD pipeline ("is this a real alpha?").

Status: CONTRACT MODULE (Week 2 of six-week-critical-path).

Currently a re-export shim — the actual FORWARD pipeline code lives in:
  - engine.agents.strengthener.factor_dispatcher    (orchestrator)
  - engine.agents.strengthener.templates.*           (per-template backtests)
  - engine.research.burndown_ranker                  (priority queue)
  - engine.research.burndown_caps                    (n_trials caps)
  - engine.research.burndown_planner                 (cron plan)
  - engine.research.burndown_executor                (cron exec)

Week 2-3 will physically move these here with import updates. Until
then, this module exposes the canonical entry points so external
callers can import `engine.research.forward` cleanly + the physical
move later is invisible.

See [[engine/research/__pipelines__.md]] for the full 3-pipeline doc.
"""
from __future__ import annotations

# Canonical FORWARD pipeline entry points (re-exported)
# Add more re-exports as code moves into this module
from engine.research.burndown_planner import plan as plan_forward_burndown
from engine.research.burndown_executor import BurndownExecutor as ForwardBurndownExecutor
from engine.research.burndown_ranker import (
    rank_candidates as rank_forward_candidates,
    RankedCandidate,
)

__all__ = [
    "plan_forward_burndown",
    "ForwardBurndownExecutor",
    "rank_forward_candidates",
    "RankedCandidate",
]
