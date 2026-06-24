"""engine/agents/decay_sentinel — Decay Sentinel agent wrapper.

The DETERMINISTIC monitoring substance lives in engine.validation.decay_sentinel
(per-mechanism rolling health, role-aware structural-decay, pairwise downside/stress
correlation, disciplined re-allocation). This package is the THIN agent layer over
that math:

  - narrator.py : turns the deterministic sentinel_report() dict into a terse
                  BlackRock-Slack daily briefing. The narrator only STATES the
                  math's verdict — it never decides (0-LLM-in-DECISION). Zero-cost
                  deterministic backend by default; LLM backend deferred.
  - agent.py    : daily cron entry point — build_mechanisms() -> sentinel_report()
                  -> narrate -> persist JSON artifact -> exit code (mirrors
                  engine.portfolio.correlation_sentinel + the ops_watchdog cron).

Right-sized for the two-mechanism book per
project-agent-rightsizing-single-mechanism-2026-05-21.
"""
