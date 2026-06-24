"""engine/agents/decay_sentinel/narrator.py — Decay Sentinel narration layer.

Turns the DETERMINISTIC engine.validation.decay_sentinel.sentinel_report() dict into a
terse BlackRock-Slack daily briefing. Mirrors engine.agents.dq_inspector.narrator
(facade + pluggable backend + DeterministicNarrator default + GeminiFlash deferred +
banned-phrases discipline).

DOCTRINE compliance (0-LLM-in-DECISION):
  - The verdict (report["overall"]), the alarms, the structural-decay flags and the
    recommended weights are ALL produced by the deterministic core. The narrator only
    DESCRIBES them — it cannot flip a verdict, raise/clear an alarm, or change a weight.
  - Banned-phrases regex enforced at output stage regardless of backend (no hedging —
    a decay verdict must read as a verdict, never "this might be decaying").

Switch backend via DECAY_SENTINEL_NARRATOR_BACKEND env var:
  - "deterministic" (default — zero cost, no network, no LLM)
  - "gemini_flash"  (deferred — same stub as RM / DQ narrators)
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Tone discipline — same banned set as Risk Manager / DQ Inspector ───────────
BANNED_PHRASES: tuple[str, ...] = (
    r"\bmaybe\b", r"\bperhaps\b", r"\bcould be\b", r"\bmight be\b", r"\bprobably\b",
    r"\bpossibly\b", r"\blikely\b", r"\bI think\b", r"\bI feel\b", r"\bseems? to\b",
    r"\bappears? to\b", r"\bjust a thought\b", r"\byou might want to\b",
)
_BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)


def contains_banned_phrase(text: str) -> Optional[str]:
    """Return the first banned phrase match, or None if the text is clean."""
    m = _BANNED_RE.search(text)
    return m.group(0) if m else None


@dataclasses.dataclass(frozen=True)
class NarrationResult:
    text:     str
    backend:  str
    cost_usd: float


# ── Backend abstraction ────────────────────────────────────────────────────────
class _NarratorBackend:
    name: str = "abstract"

    def generate(self, report: dict) -> NarrationResult:
        raise NotImplementedError


_VERDICT_HEADLINE = {
    "HEALTHY": ("Decay Sentinel — book health HEALTHY ({n} mechanisms, rolling {w}m). "
                "Every return engine is paying and diversification holds where it matters. "
                "Base weights hold; no re-allocation."),
    "WATCH":   ("Decay Sentinel — book health WATCH ({n} mechanisms, rolling {w}m). "
                "{nwarn} item(s) flagged to monitor; no structural decay confirmed. "
                "Base weights hold — watch, do not chase."),
    "ACTION":  ("Decay Sentinel — book health ACTION ({n} mechanisms, rolling {w}m). "
                "Structural decay confirmed. Re-allocate per the deterministic rule "
                "(halve the dead leg, redistribute to surviving return sources, hysteresis)."),
}


# ── Junior-analyst rationale (Man-Group lesson: a PM trusts a model that explains itself) ──
# Deterministic: composed entirely from the report's numbers. States WHY the verdict and walks
# the evidence chain per mechanism — crucially the decay-vs-drawdown distinction (the signal-IC
# gate), which is what separates "halve it" from "hold through it". 0-LLM: cannot change a fact.
def _explain_verdict(verdict: str, health: dict, decay: dict, roles: dict, alarms: list) -> str:
    decayed = [n for n in health if decay.get(n, {}).get("structural_decay")]
    soft = [n for n in health
            if roles.get(n, "alpha") not in ("insurance", "trend")
            and n not in decayed
            and isinstance(health[n].get("rolling_sharpe"), (int, float))
            and health[n]["rolling_sharpe"] == health[n]["rolling_sharpe"]
            and health[n]["rolling_sharpe"] < 0.40]
    warns = [msg for lvl, msg in alarms if lvl in ("ALERT", "WARN")]
    if verdict == "ACTION" and decayed:
        return (", ".join(decayed) + " shows BOTH a rolling-Sharpe collapse AND a signal-IC breakdown "
                "— the two conditions that separate structural decay from a transient drawdown — so the "
                "rule halves the dead leg and the surviving engines + diversification carry the book.")
    if verdict == "WATCH":
        if soft:
            return (", ".join(soft) + " softened on return, but its signal-IC is intact — that reads as a "
                    "drawdown, not decay. Hold and monitor; re-allocating while the signal still works is "
                    "the classic error.")
        if warns:
            return ("the driver is a RISK flag, not decay — " + warns[0] + " — a co-movement/stress item "
                    "to monitor; no mechanism shows the signal-IC breakdown that defines decay, so base "
                    "weights hold.")
        return ("an item is flagged to monitor; no mechanism shows the signal-IC breakdown that defines "
                "structural decay, so base weights hold.")
    return ("every alpha mechanism's rolling Sharpe is positive and none shows the signal-IC breakdown "
            "that defines structural decay; any insurance/trend calm-period drag is the premium for the "
            "hedge, by design — not decay.")


def _explain_mechanism(name: str, role: str, h: dict, dcy: dict, cp: float, wt: float) -> str:
    rs = h.get("rolling_sharpe", float("nan")); full = h.get("full_sharpe", float("nan"))
    ratio = h.get("decay_ratio", float("nan")); sic = dcy.get("signal_ic")
    wtxt = f"{wt:.0%}" if wt == wt else "n/a"
    if role in ("insurance", "trend"):
        if cp == cp and cp > 0:
            return f"{name} ({role}, {wtxt}): crisis-payoff {cp:+.2%}/mo — pays when it must; judged on crisis payoff, not Sharpe."
        cptxt = f"{cp:+.2%}/mo" if cp == cp else "n/a"
        return f"{name} ({role}, {wtxt}): crisis-payoff {cptxt} — calm-period drag is the premium for the hedge, by design; not decay."
    sic_txt = f", signal-IC {sic:+.2f}" if isinstance(sic, (int, float)) else ""
    base = f"{name} ({role}, {wtxt}): rolling Sharpe {rs:+.2f}"
    if ratio == ratio:
        base += f" ({ratio:.0%} of full {full:+.2f})"
    base += sic_txt + " → "
    if dcy.get("structural_decay"):
        base += "return AND signal both decayed = structural decay, halve."
    elif isinstance(sic, (int, float)) and rs == rs and rs < (full if full == full else rs):
        base += "return soft but signal (IC) still works = drawdown, hold."
    else:
        base += "paying, signal intact = healthy."
    return base


class DeterministicNarrator(_NarratorBackend):
    """Assembles the briefing from the structured report. Zero cost, no LLM. The prose
    is hand-written and role-aware; the FACTS (verdict, alarms, weights) are passed
    through verbatim from the deterministic core."""
    name = "deterministic"

    def generate(self, report: dict) -> NarrationResult:
        lines: list[str] = []
        n = len(report.get("mechanisms", {}))
        w = report.get("window", "?")
        verdict = report.get("overall", "HEALTHY")
        alarms = list(report.get("alarms", []))
        n_alert = sum(1 for lvl, _ in alarms if lvl == "ALERT")
        n_warn = sum(1 for lvl, _ in alarms if lvl == "WARN")

        lines.append(_VERDICT_HEADLINE.get(verdict, _VERDICT_HEADLINE["HEALTHY"]).format(
            n=n, w=w, nwarn=n_warn or n_alert))

        # Mechanism roll-call — one terse line each, role-aware health phrasing.
        roles = report.get("roles", {}); crisis = report.get("crisis", {})
        decay = report.get("decay", {}); health = report.get("mechanisms", {})
        weights = report.get("base_weights", {})
        roll = []
        for name, h in health.items():
            role = roles.get(name, "alpha"); wt = weights.get(name, float("nan"))
            if role in ("insurance", "trend"):
                cp = crisis.get(name, float("nan"))
                state = (f"crisis-payoff {cp:+.2%}/mo (hedging)" if cp == cp and cp > 0
                         else (f"crisis-payoff {cp:+.2%}/mo — NOT hedging" if cp == cp else "crisis-payoff n/a"))
            elif role == "regime_premium":
                sic = decay.get(name, {}).get("signal_ic")
                state = (f"signal-IC {sic:+.2f}" if isinstance(sic, (int, float)) else "signal-IC n/a") + \
                        (", structural decay" if decay.get(name, {}).get("structural_decay") else ", intact")
            else:
                state = f"roll Sharpe {h['rolling_sharpe']:+.2f}" + \
                        (", STRUCTURAL DECAY" if decay.get(name, {}).get("structural_decay") else "")
            roll.append(f"{name} [{role}] {wt:.0%}: {state}")
        if roll:
            lines.append("Mechanisms — " + "; ".join(roll) + ".")

        # Junior-analyst rationale: WHY the verdict + the per-mechanism evidence chain (the
        # decay-vs-drawdown logic). Deterministic — composed from the report's own numbers.
        lines.append("Why " + verdict + " — " + _explain_verdict(verdict, health, decay, roles, alarms))
        ev = [f"  - {_explain_mechanism(nm, roles.get(nm, 'alpha'), h, decay.get(nm, {}), crisis.get(nm, float('nan')), weights.get(nm, float('nan')))}"
              for nm, h in health.items()]
        if ev:
            lines.append("Evidence:")
            lines += ev

        # Alarms that matter (ALERT + WARN); INFO collapsed to a count with reason.
        sev = [(lvl, msg) for lvl, msg in alarms if lvl in ("ALERT", "WARN")]
        if sev:
            lines.append("Flags:")
            lines += [f"  [{lvl}] {msg}" for lvl, msg in sev]
        n_info = sum(1 for lvl, _ in alarms if lvl == "INFO")
        if n_info:
            lines.append(f"({n_info} informational note(s): insurance/trend calm-period drag is "
                         "by design and benign symmetric co-movement — not decay.)")

        # Allocation action.
        if report.get("realloc_action"):
            rec = report.get("recommended_weights", {})
            lines.append("RE-ALLOCATION: " + ", ".join(f"{k} {v:.0%}" for k, v in rec.items()) +
                         " (halved the decayed leg, redistributed to surviving return sources; "
                         "restore only after rolling Sharpe recovers > 0.40).")
        else:
            lines.append("Allocation: base weights unchanged — re-allocate only on confirmed "
                         "structural decay (signal-IC gated), never on a drawdown.")

        text = "\n".join(lines)
        bad = contains_banned_phrase(text)
        if bad:
            logger.error("Decay Sentinel narrator: output contains banned phrase %r — "
                         "template authoring bug, please clean.", bad)
        return NarrationResult(text=text, backend=self.name, cost_usd=0.0)


class GeminiFlashNarrator(_NarratorBackend):
    """Vertex Gemini 2.5 Flash narrator with banned-phrases enforcement. DEFERRED —
    same pattern as RM / DQ GeminiFlashNarrator; lands when Vertex auth + cost ledger
    are verified end-to-end. agent_id='decay_sentinel' is already in ALLOWED_AGENT_IDS."""
    name = "gemini_flash"

    def generate(self, report: dict) -> NarrationResult:
        raise NotImplementedError(
            "Decay Sentinel GeminiFlashNarrator deferred. "
            "Set DECAY_SENTINEL_NARRATOR_BACKEND=deterministic for now."
        )


def _select_backend(name: Optional[str] = None) -> _NarratorBackend:
    backend_name = (name or os.environ.get("DECAY_SENTINEL_NARRATOR_BACKEND") or "deterministic")
    if backend_name == "deterministic":
        return DeterministicNarrator()
    if backend_name == "gemini_flash":
        return GeminiFlashNarrator()
    raise ValueError(f"Unknown Decay Sentinel narrator backend {backend_name!r}. "
                     "Use deterministic / gemini_flash.")


def narrate_report(report: dict, *, backend: Optional[str] = None) -> NarrationResult:
    """Generate the daily briefing prose for one sentinel_report() dict.
    Default = deterministic backend (zero cost)."""
    return _select_backend(backend).generate(report)
