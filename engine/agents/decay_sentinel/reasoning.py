"""engine/agents/decay_sentinel/reasoning.py — narrative-reasoning layer for the
Decay Sentinel daily report (Phase 1 Task II.B of research_agenda_2026-05-29).

Takes the DETERMINISTIC report produced by engine.validation.decay_sentinel
(per-mechanism rolling health, structural-decay flags, etc.) and enriches it
with EVIDENCE-CITED HUMAN-READABLE narratives + RECOMMENDED ACTIONS.

Two modes, BOTH preserve the 0-LLM-in-DECISION doctrine — the deterministic
report's verdicts (HEALTHY / WATCH / ACTION) are NEVER changed:

  - `narrate_deterministic(report)` — template-based, always works, no API
    dependency. The narrative is generated from the exact numbers in the
    report; every claim cites a specific metric. This is the daily-cron-
    safe default. Evidence-cited by construction.

  - `narrate_with_llm(report)` — optional richer narrative via Anthropic
    Claude (uses the Attribution Analyst persona system prompt). Falls back
    to deterministic mode on API failure or missing key. Adds nothing
    beyond the deterministic verdicts; only the prose.

Output schema (both modes):
    {
      "mode": "deterministic" | "llm" | "deterministic_fallback_<reason>",
      "overall": {
        "book_health": str,           # mirrors report["overall"]
        "narrative":   str,           # 2-4 sentences
        "recommended_action": str|None,
        "counts": {"action": int, "watch": int, "healthy": int}
      },
      "per_mechanism": {
        "<name>": {
          "name": str,
          "status": "HEALTHY"|"WATCH"|"ACTION",
          "narrative": str,            # 2-3 sentences
          "evidence":  list[dict],     # each: metric + value + threshold/baseline
          "recommended_action": str|None
        }, ...
      }
    }

Designed so the cron wrapper agent.py appends `reasoning` to the JSON
artifact at data/decay_sentinel/decay_sentinel_<date>.json, alongside the
existing deterministic report.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ── Role-specific judging context (matches decay_sentinel.py doctrine) ───────

_ROLE_CONTEXT = {
    "alpha":          "calm-period Sharpe is the primary metric — alpha must pay >0.4 unconditional",
    "trend":          "convex hedge — judged on crisis payoff, calm-Sharpe is by design",
    "insurance":      "convex hedge — judged on crisis payoff, calm-Sharpe is by design",
    "regime_premium": "regime-dependent premium (roll-yield) — judged on calm-period Sharpe AND risk-off behavior",
}


# ── Deterministic template-based narrator ────────────────────────────────────

def _fmt(x, n: int = 2, default: str = "n/a") -> str:
    """Format a number with N decimals, or 'n/a' if missing."""
    if x is None:
        return default
    try:
        xf = float(x)
        if xf != xf:   # NaN
            return default
        return f"{xf:.{n}f}"
    except (TypeError, ValueError):
        return default


def narrate_mechanism(name: str, m: dict) -> dict:
    """Build evidence-cited narrative for one mechanism. Pure function over the
    report's mechanism record — no I/O.

    Args:
        name: mechanism id (e.g. "D_PEAD", "cross_asset_carry")
        m:    report["mechanisms"][name] dict
    Returns:
        {name, status, narrative, evidence[], recommended_action}
    """
    role = m.get("role") or "unclassified"
    role_ctx = _ROLE_CONTEXT.get(role, "unclassified role — doctrine unclear, route to human")

    full_sh    = m.get("full_sharpe")
    roll_sh    = m.get("rolling_sharpe")
    roll_t     = m.get("rolling_t")
    decay_r    = m.get("decay_ratio")
    sig_ic     = m.get("signal_ic")
    struct     = bool(m.get("structural_decay", False))
    crisis     = m.get("crisis_payoff")
    stress_b   = m.get("stress_beta")
    mkt_b      = m.get("mkt_beta")
    weight     = m.get("weight")

    # ── Build evidence list (one entry per metric with context) ──
    evidence: list[dict] = []
    if full_sh is not None:
        evidence.append({"metric": "full_sample_sharpe", "value": round(float(full_sh), 3)})
    if roll_sh is not None:
        item = {"metric": "rolling_36m_sharpe", "value": round(float(roll_sh), 3)}
        if full_sh is not None:
            item["baseline_full"] = round(float(full_sh), 3)
            item["delta"] = round(float(roll_sh) - float(full_sh), 3)
        if decay_r is not None:
            item["decay_ratio"] = round(float(decay_r), 3)
        evidence.append(item)
    if roll_t is not None:
        evidence.append({"metric": "rolling_36m_sharpe_tstat",
                         "value": round(float(roll_t), 2),
                         "note": "t<2 = sharpe not significant on 36m window"})
    if sig_ic is not None:
        evidence.append({"metric": "rolling_36m_signal_ic",
                         "value": round(float(sig_ic), 4),
                         "fade_threshold": 0.005,
                         "verdict": "FADED" if float(sig_ic) < 0.005 else "OK"})
    if crisis is not None:
        evidence.append({"metric": "crisis_payoff",
                         "value": round(float(crisis), 4),
                         "role_relevance": "primary" if role in ("trend", "insurance") else "secondary"})
    if stress_b is not None:
        evidence.append({"metric": "stress_beta", "value": round(float(stress_b), 3)})

    # ── Synthesize narrative based on status + role doctrine ──
    if struct:
        # Structural decay flagged
        narrative_parts = [
            f"{name}: structural decay flagged."
        ]
        if roll_sh is not None and full_sh is not None:
            narrative_parts.append(
                f"Rolling 36m Sharpe {_fmt(roll_sh, 2)} vs full {_fmt(full_sh, 2)} "
                f"(decay ratio {_fmt(decay_r, 2)})."
            )
        if sig_ic is not None and float(sig_ic) < 0.005:
            narrative_parts.append(
                f"Signal IC {_fmt(sig_ic, 4)} below 0.005 fade threshold."
            )
        narrative_parts.append(f"Role: {role} — {role_ctx}.")
        narrative_parts.append(
            "Action: re-allocate per deterministic recommend_allocation() output before next rebal."
        )
        narrative = " ".join(narrative_parts)
        status = "ACTION"
        action = f"Run engine.validation.decay_sentinel.recommend_allocation() and apply to {name} weight"

    elif role in ("trend", "insurance"):
        # Convex hedge: don't judge on calm-Sharpe
        narrative = (
            f"{name}: convex hedge ({role}). "
            f"Calm-period Sharpe {_fmt(roll_sh, 2)} is by design "
            f"(insurance premium paid in calm). "
            f"Crisis payoff {_fmt(crisis, 4)}, stress beta {_fmt(stress_b, 2)}. "
            f"Doctrine: {role_ctx}. Status: HEALTHY."
        )
        status = "HEALTHY"
        action = None

    elif decay_r is not None and float(decay_r) < 0.5:
        # WATCH: Sharpe dropped >50% but no structural flag yet
        narrative = (
            f"{name}: rolling Sharpe {_fmt(roll_sh, 2)} vs full {_fmt(full_sh, 2)} — "
            f"decay ratio {_fmt(decay_r, 2)}, down >50%. "
            f"Signal IC {_fmt(sig_ic, 4)}." if sig_ic is not None else f"{name}: rolling Sharpe {_fmt(roll_sh, 2)} vs full {_fmt(full_sh, 2)} — decay ratio {_fmt(decay_r, 2)}, down >50%."
        )
        narrative += " Not structural yet — WATCH for one more quarter."
        status = "WATCH"
        action = f"Monitor {name} at next quarterly review; one more quarter of <50% decay ratio triggers structural flag"

    else:
        # Healthy alpha or regime_premium
        roll_text = (f"Rolling 36m Sharpe {_fmt(roll_sh, 2)} vs full {_fmt(full_sh, 2)} "
                     f"(decay ratio {_fmt(decay_r, 2)})." if roll_sh is not None else "")
        ic_text = (f" Signal IC {_fmt(sig_ic, 4)}." if sig_ic is not None else "")
        narrative = f"{name} ({role}): {roll_text}{ic_text} Within calm-period band. Status: HEALTHY."
        status = "HEALTHY"
        action = None

    return {
        "name": name,
        "status": status,
        "narrative": narrative,
        "evidence": evidence,
        "recommended_action": action,
        "role": role,
        "weight": weight,
    }


def narrate_overall(report: dict, per_mech: dict) -> dict:
    """Build book-level summary narrative aggregating per-mechanism statuses."""
    n_action  = sum(1 for x in per_mech.values() if x["status"] == "ACTION")
    n_watch   = sum(1 for x in per_mech.values() if x["status"] == "WATCH")
    n_healthy = sum(1 for x in per_mech.values() if x["status"] == "HEALTHY")
    book_health = report.get("overall", "UNKNOWN")
    realloc = bool(report.get("realloc_action", False))

    if realloc or n_action > 0:
        action_mechs = [n for n, x in per_mech.items() if x["status"] == "ACTION"]
        narrative = (
            f"Book health: {book_health}. "
            f"{n_action} mechanism(s) flagged for re-allocation: {', '.join(action_mechs)}. "
            f"{n_watch} on WATCH, {n_healthy} HEALTHY. "
            f"Action: run engine.validation.decay_sentinel.recommend_allocation() and "
            f"apply at next rebalance."
        )
        action = ("engine.validation.decay_sentinel.recommend_allocation() at next rebal; "
                  f"primary targets: {', '.join(action_mechs)}")

    elif n_watch > 0:
        watch_mechs = [n for n, x in per_mech.items() if x["status"] == "WATCH"]
        narrative = (
            f"Book health: {book_health}. "
            f"{n_watch} mechanism(s) on WATCH: {', '.join(watch_mechs)}. "
            f"{n_healthy} HEALTHY. No structural decay yet — monitor at quarterly review."
        )
        action = f"Monitor watch-list at next quarterly review ({n_watch} mechanism(s)): {', '.join(watch_mechs)}"

    else:
        narrative = (
            f"Book health: {book_health}. "
            f"All {n_healthy} mechanisms within calm-period bands. No action."
        )
        action = None

    return {
        "book_health": book_health,
        "narrative": narrative,
        "recommended_action": action,
        "counts": {"action": n_action, "watch": n_watch, "healthy": n_healthy},
    }


def narrate_deterministic(report: dict) -> dict:
    """Top-level deterministic narrator. ALWAYS works (no API, no I/O)."""
    per_mech: dict[str, dict] = {}
    for name, m in report.get("mechanisms", {}).items():
        per_mech[name] = narrate_mechanism(name, m)
    overall = narrate_overall(report, per_mech)
    return {
        "mode": "deterministic",
        "overall": overall,
        "per_mechanism": per_mech,
    }


# ── Optional LLM narrator (Anthropic Claude via Attribution Analyst persona) ─

def _read_anthropic_key() -> str | None:
    """Resolve ANTHROPIC_API_KEY from env or streamlit secrets."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


_LLM_PROMPT_TEMPLATE = """You are the Attribution Analyst — a forensic P&L decomposition agent. Your role-id is `attribution_analyst_forensic`. You operate downstream of the daily Decay Sentinel.

# Tone
- Terse. BlackRock-Slack grade. Active voice. No hedging.
- NO EMOJIS in any response, ever.
- BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.
- State the number, the date range, the sleeve. No vague "performed well" language.

# Doctrine constraints
- The verdicts (HEALTHY/WATCH/ACTION) are DETERMINISTIC and are already set in the report you are given. You do not change them.
- You explain WHY using the report numbers. Every claim must cite a specific metric.
- For trend/insurance roles, calm-period Sharpe is "by design" — do not flag.
- For alpha/regime_premium, calm-period Sharpe is primary. Signal IC < 0.005 is the documented fade threshold.

# Task
You receive the deterministic decay sentinel report below. For EACH mechanism, produce:
  - status: copy verbatim from report (HEALTHY/WATCH/ACTION)
  - narrative: 2-3 sentences citing specific numbers from the report
  - recommended_action: one concrete imperative or None

Then for the OVERALL book, produce the same fields.

Return STRICT JSON (no markdown fence) with this schema:
  {{"overall": {{"book_health": str, "narrative": str, "recommended_action": str|None,
                 "counts": {{"action": int, "watch": int, "healthy": int}}}},
   "per_mechanism": {{"<name>": {{"name": str, "status": str, "narrative": str,
                                   "evidence": [], "recommended_action": str|None,
                                   "role": str, "weight": number}} }} }}

# Report

{report_json}
"""


def narrate_with_llm(report: dict, timeout_s: float = 60.0) -> dict:
    """Anthropic Claude version. Falls back to deterministic on any failure."""
    key = _read_anthropic_key()
    if not key:
        result = narrate_deterministic(report)
        result["mode"] = "deterministic_fallback_no_api_key"
        return result

    try:
        import json as _json
        from anthropic import Anthropic
    except ImportError:
        result = narrate_deterministic(report)
        result["mode"] = "deterministic_fallback_no_sdk"
        return result

    try:
        client = Anthropic(api_key=key, timeout=timeout_s)
        prompt = _LLM_PROMPT_TEMPLATE.format(
            report_json=_json.dumps(report, indent=2, default=str))
        # Use a small/fast model for narrative generation — Haiku is enough
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else "{}"
        # Strip any accidental code fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("`").strip()

        parsed = _json.loads(text)
        # Sanity: must have overall + per_mechanism
        if "overall" not in parsed or "per_mechanism" not in parsed:
            raise ValueError(f"LLM response missing required keys: {list(parsed)}")

        # Doctrine guard: LLM cannot have changed deterministic statuses
        det = narrate_deterministic(report)
        for name, det_m in det["per_mechanism"].items():
            llm_m = parsed["per_mechanism"].get(name, {})
            if llm_m.get("status") != det_m["status"]:
                logger.warning("LLM tried to change %s status %s→%s; restoring deterministic",
                               name, det_m["status"], llm_m.get("status"))
                parsed["per_mechanism"][name]["status"] = det_m["status"]
        return {"mode": "llm", **parsed}

    except Exception as e:
        logger.warning("LLM narrator failed (%s); falling back to deterministic", e)
        result = narrate_deterministic(report)
        result["mode"] = f"deterministic_fallback_{type(e).__name__.lower()}"
        return result


def narrate_report(report: dict, llm: bool = False) -> dict:
    """Public entry. llm=False is the doctrine-safe default (always works)."""
    if llm:
        return narrate_with_llm(report)
    return narrate_deterministic(report)
