"""
engine/agents/ops_watchdog/prompt.py — ReAct prompt scaffolding for Watchdog.

This module owns:
  - WATCHDOG_ROLE_INTRO: the role description injected as `role_intro` into
    Tool 1's `run_react_agent` (overrides default Decision-Lineage framing).
  - build_watchdog_query(...): assembles today's findings + context into the
    natural-language `query` that the LLM reasons over.

Per spec §6 "Implementation iterative (unlocked)" row, prompt wording is
adjustable without spec amendment. What IS locked:
  - LLM never decides severity → captured by role intro telling LLM
    explicitly its summary is *context*, not classification.
  - LLM never decides auto-repair → role intro says repair recipes are
    hardcoded; LLM only describes which modes co-occurred.
  - 0-LLM-in-alpha-decision-loop → role intro frames Watchdog as
    operations-layer only.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


WATCHDOG_ROLE_INTRO = """You are the **Ops Watchdog Agent** for the MacroAlphaPro quant system (spec id=63, daily 06:10 SGT run, operations layer only, NEVER in the alpha decision loop).

Your job: reason over today's Auto-Audit findings and provide a CONCISE OPERATIONAL CONTEXT REPORT for the orchestrator. You do NOT decide severity (hardcoded in triage.py per MODE_SEVERITY_MAP_LOCKED). You do NOT decide whether to auto-repair (hardcoded in auto_repair.py per AUTO_REPAIR_RECIPES_LOCKED). You provide CONTEXT — historical baselines, co-occurrence patterns, and a short narrative summary.

CRITICAL DISCIPLINE — DO NOT VIOLATE:
1. **You MUST call at least one tool before giving final_answer.** No speculation; gather evidence first.
2. **Read-only operation**: you query state via the 10 tools listed below. You NEVER decide actions — the orchestrator dispatches notifications + recipes per hardcoded maps.
3. **Cite sources inline** in final_answer:
   - AuditFinding IDs: e.g. "finding id=42"
   - rule names: e.g. "rule_signal_panel_nan_scan"
   - mode keys: e.g. "mode_8_signal_nan"
   - dates: ISO format e.g. "2026-05-12"
   - memory filenames: e.g. "project_X.md" (only if you actually read one)
4. If a tool returns an error or empty data, try a different tool. Do NOT invent findings.
5. Keep final_answer ≤ 400 chars. The orchestrator only needs your narrative gist; structured triage already runs separately."""


def build_watchdog_query(
    today_iso:           str,
    findings_preview:    Iterable[dict],
    triage_pre_summary:  dict,
) -> str:
    """
    Compose the natural-language query that goes into `run_react_agent`.

    Args:
        today_iso: ISO date string of the Watchdog run (e.g. "2026-05-12")
        findings_preview: iterable of {rule_name, severity, snapshot_summary}
          dicts representing today's AuditFinding rows. Snapshot may be
          truncated to keep prompt size manageable.
        triage_pre_summary: output of triage.aggregate_severity() — the
          hardcoded severity decision (so LLM knows the dispatch is already
          determined and its job is purely context).

    Returns:
        A multi-line natural-language query string.
    """
    findings_list = list(findings_preview)
    n = len(findings_list)
    findings_block = (
        json.dumps(findings_list[:10], indent=2, default=str)
        if findings_list else "(no findings — all 12 modes green)"
    )
    triage_block = json.dumps(triage_pre_summary, indent=2, default=str)

    return f"""Today is {today_iso}. The Auto-Audit Loop just executed all WATCHDOG_RULES (11 Watchdog rules + 2 reused) and produced {n} AuditFinding rows on this date.

The hardcoded triage step has ALREADY classified severity per the locked map (this is for your awareness — your role is NOT to re-classify):
{triage_block}

Today's findings (first 10 shown):
{findings_block}

Your task:
1. Call read_historical_baseline / read_nav_change / read_trade_log as needed to give the orchestrator a SHORT narrative on what's happening today (e.g. "mode 11 fires every Monday post-holiday weekend — not surprising").
2. Note any cross-mode co-occurrence pattern (e.g. modes 2 + 8 together → upstream data feed problem).
3. If any finding's snapshot looks suspicious vs baseline (e.g. NAV move is just barely above the 3σ threshold), say so.
4. End with a single-paragraph final_answer ≤ 400 chars summarizing context. Cite rule_name + mode_key inline. Do NOT propose actions."""
