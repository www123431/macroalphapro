"""engine/agents/governance/authority.py — least-privilege enforcement at the executor.

THREAT-MODEL control (blueprint spec id=78 §8, Phase 3): today a persona's tool palette is
declared to the model (persona.tools), but the EXECUTOR doesn't re-check it. Defense-in-
depth: enforce least-privilege at the RESOURCE (the tool executor), not just at the prompt/
API declaration — so a persona induced (jailbreak / bug / mis-wired delegation) to emit a
tool_use outside its palette is BLOCKED at runtime + audited, never executed.

A persona's capability set = the names in its OWN declared palette (persona.tools) — the
single source of truth, already curated via select_tools(). Enforcement has ZERO false-
positive risk (a legit call is always in-palette), so the default mode is ENFORCE.

0-LLM, deterministic. Audit -> data/governance/authority_audit.jsonl.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
AUDIT_PATH = Path("data/governance/authority_audit.jsonl")


def capability_set(persona) -> frozenset:
    """The tool names this persona is permitted to call (its declared palette)."""
    return frozenset(t.get("name") for t in getattr(persona, "tools", ()) if t.get("name"))


def check_authority(persona, tool_name: str) -> tuple[bool, str]:
    caps = capability_set(persona)
    if tool_name in caps:
        return True, "in-palette"
    return False, (f"agent '{getattr(persona, 'agent_id', '?')}' not permitted to call "
                   f"'{tool_name}' (palette: {sorted(caps)})")


def _audit(agent_id: str, tool_name: str, allowed: bool, reason: str) -> None:
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.datetime.utcnow().isoformat() + "Z",
                                "agent_id": agent_id, "tool": tool_name,
                                "allowed": allowed, "reason": reason},
                               ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("authority audit write failed (non-fatal)")


def enforce_tool_call(persona, tool_name: str, tool_input: dict,
                      *, mode: "str | None" = None) -> tuple[str, bool]:
    """Authority-checked tool dispatch. mode from AGENT_AUTHORITY_MODE env, default
    'enforce' (zero false-positive: a legit call is always in-palette).
      enforce — out-of-palette call is BLOCKED (returns an error result, tool NOT run).
      warn    — log + audit but still run (transition only).
      off     — no check.
    In-palette (or off) -> delegate to persona.tool_executor. Out-of-palette calls are
    audited regardless of mode (the security signal is the attempt)."""
    mode = (mode or os.environ.get("AGENT_AUTHORITY_MODE") or "enforce").lower()
    agent_id = getattr(persona, "agent_id", "?")
    if mode == "off":
        return persona.tool_executor(tool_name, tool_input)

    allowed, reason = check_authority(persona, tool_name)
    if not allowed:
        _audit(agent_id, tool_name, allowed=False, reason=reason)
        logger.warning("AUTHORITY %s: %s", "BLOCK" if mode == "enforce" else "WARN", reason)
        if mode == "enforce":
            return (json.dumps({"error": f"authority denied: {reason}"}), True)
        # warn -> fall through and execute anyway
    return persona.tool_executor(tool_name, tool_input)
