"""
engine/agents/risk_manager/narrator.py — Phase 7 narrative layer.

Generates one-paragraph BlackRock-Slack-tone prose per breach. Updates
RiskManagerAlert.narrative_text via persist.update_narrative.

DESIGN — facade + pluggable LLM backend
---------------------------------------
This module ships a FACADE that the rest of the system (UI, Watchdog,
DD investigation workflow) consumes. The actual LLM call is gated
behind `NarratorBackend.generate(...)` which has two implementations:

  - `DeterministicNarrator` (default, this commit) — produces fact-only
    template prose from the Breach fields without any LLM call. Zero
    cost, no network dependency, deterministic across reruns. Tone is
    BlackRock-Slack-grade (terse / active voice / no hedging) by
    construction since it's written by hand.

  - `GeminiFlashNarrator` (DEFERRED to next focused commit) — calls
    Vertex Gemini 2.5 Flash with temperature=0.1, banned-phrases regex,
    cost ledger integration. Will land when LLM API key + Vertex auth
    are verified in a clean session.

Switch via `RISK_MANAGER_NARRATOR_BACKEND` env var:
  - "deterministic" (default — this commit)
  - "gemini_flash"  (next commit)
  - "mock"          (tests)

DOCTRINE compliance:
  - LLM (when wired) NEVER runs before deterministic gates produced the
    breach list. Narrator only DESCRIBES; it cannot flip halt verdict.
  - Banned-phrases regex enforced at output stage regardless of backend
    (so even DeterministicNarrator's templates pass the same check).
  - Cost ledger write is mandatory for any LLM backend (deterministic
    backend skips ledger since cost = 0).

PersonaContext (forward-compatible stub)
----------------------------------------
The Persona Voice Layer sprint (per [[project-agent-team-persona-locked-
2026-05-18]]) will add character sheet + voice phrase library + cross-
agent reference. This module ACCEPTS a `PersonaContext` parameter today
but currently ignores its richer fields; only the role_id is used to
select prompt template. When the persona sprint lands, the template
selector + voice library lookups will read PersonaContext without
requiring narrator API changes.
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engine.agents.risk_manager.gates import Breach
    from engine.agents.risk_manager.agent import RiskManagerRunResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tone discipline — banned phrases (enforced at output, regardless of backend)
# ──────────────────────────────────────────────────────────────────────────────
# Hedging vocabulary that softens a halt verdict. Banned per spec §2.5
# (narrator never softens halt). If the LLM produces any of these, the
# response is rejected and the deterministic-template fallback fires.
BANNED_PHRASES: tuple[str, ...] = (
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bcould be\b",
    r"\bmight be\b",
    r"\bprobably\b",
    r"\bpossibly\b",
    r"\blikely\b",
    r"\bI think\b",
    r"\bI feel\b",
    r"\bseems? to\b",
    r"\bappears? to\b",
    r"\bjust a thought\b",
    r"\byou might want to\b",
    r"\bconsider\b",            # banned in advisory voice; use direct imperative
)

_BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)


def contains_banned_phrase(text: str) -> Optional[str]:
    """Return the first banned phrase match, or None if text is clean."""
    m = _BANNED_RE.search(text)
    return m.group(0) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# PersonaContext — forward-compatible stub for Phase Voice Layer sprint
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class PersonaContext:
    """Per-call persona context. Phase 7 uses role_id only.

    Future Persona Voice Layer sprint (37-52h) will populate the other
    fields from `engine.agents.capabilities.AGENT_CAPABILITIES` and the
    character sheet registry.
    """
    role_id:                str   = "head_of_risk_blackrock_slack"
    voice_phrase_library:   Optional[dict] = None   # filled by Persona sprint
    cross_agent_references: tuple[str, ...] = ()    # populated when δ layer lands
    episodic_memory_hits:   tuple[str, ...] = ()    # populated when ε layer lands


# ──────────────────────────────────────────────────────────────────────────────
# Backend abstraction
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class NarrationResult:
    """One narration call's output."""
    text:           str         # the prose
    backend:        str         # which backend was used
    cost_usd:       float       # 0.0 for deterministic
    rejected_drafts: tuple[str, ...] = ()  # banned-phrase failures rerolled


class _NarratorBackend:
    """Protocol — generate(breach, context) → NarrationResult."""
    name: str = "abstract"

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic template narrator (this commit)
# ──────────────────────────────────────────────────────────────────────────────
class DeterministicNarrator(_NarratorBackend):
    """Hand-written templates per mode. Zero cost, no network, no LLM.

    Produces BlackRock-Slack-grade prose by construction (no hedging,
    active voice, terse).
    """
    name = "deterministic"

    # Mode → template. Format spec uses Breach field names as placeholders.
    _TEMPLATES: dict[str, str] = {
        "1a": ("Book-level single-ticker concentration: {affected_first} at "
               "{observed_value:.2%} of book exceeds {threshold:.2%} "
               "operational-risk cap (issuer/ETF blowup defence; uniform "
               "across sleeves). Halt issued. Reduce cross-strategy "
               "stacking on this ticker before re-submission."),
        "1b": ("Intra-strategy concentration: strategy {extra_strategy!r} "
               "holds {affected_first} at {observed_value:.2%} within its "
               "own gross — exceeds {threshold:.2%} cap for sleeve_class "
               "{extra_sleeve_class!r}. Halt issued. Re-evaluate position "
               "sizing inside that strategy before re-submission."),
        "2":  ("Sleeve drift: {affected_first} effective weight diverges "
               "{observed_value:.0%} relative to target. Strategy NO_SIGNAL "
               "or cache staleness is the typical cause. Investigate the "
               "originating strategy's signal path."),
        "3":  ("Gross leverage breach: book at {observed_value:.2f}x exceeds "
               "{threshold:.2f}x Tier-3 cap. Halt issued. Reduce position "
               "magnitudes or reduce leverage factor before re-submission."),
        "4":  ("Net exposure outside band: book net at {observed_value:+.2f} "
               "outside the [-0.50, +1.50] mandate. Halt issued. Rebalance "
               "long/short composition before re-submission."),
        "5":  ("Concentration risk (HHI): book HHI {observed_value:.3f} "
               "exceeds {threshold:.2f} cap. Halt issued. Top positions "
               "dominate; either trim the largest weights or expand the "
               "active universe before re-submission."),
        "6":  ("Tail risk flag: 1-day VaR-95 at {observed_value:.2%} crosses "
               "the {threshold:.2%} soft-warn floor. Book persisted. "
               "Monitor for further deterioration; no halt issued."),
        "6b": ("Model-integrity breach: 1-day VaR-95 at {observed_value:.2%} "
               "is past 3x the warn threshold. Halt issued — the risk "
               "model itself is signaling distress. Investigate data inputs "
               "before re-submission."),
        "7":  ("Tail loss flag: 1-day ES-95 at {observed_value:.2%} crosses "
               "the {threshold:.2%} soft-warn floor. Book persisted. "
               "Monitor expected-shortfall trend."),
        "7b": ("Model-integrity breach: 1-day ES-95 at {observed_value:.2%} "
               "is past 3x the warn threshold. Halt issued — extreme-loss "
               "estimator in distress. Investigate before re-submission."),
        "8":  ("Short-side concentration: short positions represent "
               "{observed_value:.0%} of gross, exceeding the "
               "{threshold:.0%} ceiling. Book persisted; review balance "
               "between long and short exposures."),
        "9":  ("Strategy availability degraded: only {observed_value:.0f} "
               "of {extra_n_total} strategies are OK today, below the "
               "{threshold:.0f}-minimum threshold. Halt issued. Investigate "
               "data pipeline before tomorrow's run."),
        "10": ("Cross-cancel inefficiency: {observed_value:.0f} tickers held "
               "both long and short across strategies (cap {threshold:.0f}). "
               "Capital efficiency reduced. Review sleeve correlation."),
    }

    _FALLBACK_TEMPLATE = (
        "Breach in mode {mode_id} ({severity}): {rule_description}. "
        "Spec anchor: {spec_anchor}."
    )

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        template = self._TEMPLATES.get(breach.mode_id, self._FALLBACK_TEMPLATE)
        try:
            text = template.format(
                mode_id            = breach.mode_id,
                severity           = breach.severity,
                rule_description   = breach.rule_description,
                observed_value     = breach.observed_value if breach.observed_value is not None else 0.0,
                threshold          = breach.threshold if breach.threshold is not None else 0.0,
                affected_first     = breach.affected[0] if breach.affected else "the book",
                spec_anchor        = breach.spec_anchor,
                extra_n_total      = breach.extra.get("n_strategies_total", 0),
                extra_strategy     = breach.extra.get("strategy", "?"),
                extra_sleeve_class = breach.extra.get("sleeve_class", "?"),
            )
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("DeterministicNarrator: template format error: %s; fallback used", exc)
            text = self._FALLBACK_TEMPLATE.format(
                mode_id          = breach.mode_id,
                severity         = breach.severity,
                rule_description = breach.rule_description,
                spec_anchor      = breach.spec_anchor,
            )
        # Banned-phrase check — should never fail with handwritten templates,
        # but guard against template edits that introduce hedging.
        bad = contains_banned_phrase(text)
        if bad:
            logger.error(
                "DeterministicNarrator: template for mode %s contains banned phrase %r; "
                "this is a template authoring bug — please clean the template",
                breach.mode_id, bad,
            )
        return NarrationResult(
            text     = text,
            backend  = self.name,
            cost_usd = 0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Run-level rationale (junior-analyst layer) — explains the OVERALL verdict, incl. a
# CLEAR run (the per-breach templates only fire on a breach). Deterministic: composed from
# the per-mode utilizations the caller supplies (observed / limit). 0-LLM; cannot flip a verdict.
# `utilizations`: list of {"mode": str, "observed_txt": str, "limit_txt": str, "util": float}
# where util = observed/limit (fraction of the cap consumed; ≥1.0 = at/over the limit).
# ──────────────────────────────────────────────────────────────────────────────
def narrate_risk_summary(utilizations: list[dict], severity: str, halt: bool) -> str:
    live = [u for u in utilizations if isinstance(u.get("util"), (int, float)) and u["util"] == u["util"]]
    live.sort(key=lambda u: u["util"], reverse=True)
    if halt:
        breached = [u for u in live if u["util"] >= 1.0] or live[:1]
        head = ("Risk Manager — HALT. " + "; ".join(
            f"{u['mode']} {u['observed_txt']} vs {u['limit_txt']} ({u['util']:.0%} of cap)" for u in breached)
            + ". Book NOT persisted — fix the binding breach(es) before re-submission.")
        return head
    if not live:
        return "Risk Manager — CLEAR. No live numeric mode to bind on this book."
    b = live[0]
    nxt = live[1] if len(live) > 1 else None
    out = (f"Risk Manager — CLEAR ({severity}). All live risk modes pass. Binding constraint: "
           f"{b['mode']} at {b['observed_txt']} vs {b['limit_txt']} — {b['util']:.0%} utilized, "
           f"{max(0.0, 1 - b['util']):.0%} headroom.")
    if nxt:
        out += f" Next-closest: {nxt['mode']} ({nxt['util']:.0%} of its limit)."
    out += " Re-allocate/halt only on a HARD breach — headroom is monitored, not chased."
    bad = contains_banned_phrase(out)
    if bad:
        logger.error("narrate_risk_summary: banned phrase %r — template bug.", bad)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Gemini Flash backend — DEFERRED (TODO marker for next focused commit)
# ──────────────────────────────────────────────────────────────────────────────
class GeminiFlashNarrator(_NarratorBackend):
    """Vertex Gemini 2.5 Flash narrator with banned-phrases regex enforcement.

    NOT YET IMPLEMENTED. Will land in a focused next commit when LLM API
    auth + cost ledger integration are verified end-to-end. Pattern follows
    engine.forensic.news_context (Vertex REST httpx + thinkingConfig
    budget=0 + temperature=0.1 + structured JSON return).

    Calling .generate() before implementation lands raises explicitly
    rather than silently falling back to deterministic (would mask config
    error).
    """
    name = "gemini_flash"

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        raise NotImplementedError(
            "GeminiFlashNarrator implementation deferred to next commit. "
            "Set RISK_MANAGER_NARRATOR_BACKEND=deterministic for now."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────────
def _select_backend(name: Optional[str] = None) -> _NarratorBackend:
    backend_name = (name or os.environ.get("RISK_MANAGER_NARRATOR_BACKEND")
                    or "deterministic")
    if backend_name == "deterministic":
        return DeterministicNarrator()
    if backend_name == "gemini_flash":
        return GeminiFlashNarrator()
    raise ValueError(
        f"Unknown narrator backend {backend_name!r}. "
        f"Set RISK_MANAGER_NARRATOR_BACKEND to one of: deterministic / gemini_flash"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API — narrate(...) + narrate_run_result(...)
# ──────────────────────────────────────────────────────────────────────────────
def narrate_breach(
    breach:    "Breach",
    *,
    context:   Optional[PersonaContext] = None,
    backend:   Optional[str] = None,
) -> NarrationResult:
    """Generate prose for a single breach. Default backend = deterministic."""
    backend_impl = _select_backend(backend)
    ctx = context or PersonaContext()
    return backend_impl.generate(breach, ctx)


def narrate_run_result(
    rm_result:       "RiskManagerRunResult",
    *,
    context:         Optional[PersonaContext] = None,
    backend:         Optional[str] = None,
    update_db:       bool = True,
) -> list[NarrationResult]:
    """Generate prose for every breach in a RiskManagerRunResult.

    If update_db is True (default), persists each narrative to the
    RiskManagerAlert table via persist.update_narrative.

    Returns list[NarrationResult] in breach order — caller can render
    them in UI without re-loading DB.
    """
    if not rm_result.breaches:
        return []

    results: list[NarrationResult] = []
    for breach, alert_id in zip(rm_result.breaches, rm_result.audit_alert_ids):
        narration = narrate_breach(breach, context=context, backend=backend)
        results.append(narration)
        if update_db and alert_id:
            from engine.agents.risk_manager.persist import update_narrative
            try:
                update_narrative(
                    date           = datetime.date.fromisoformat(rm_result.today_iso),
                    alert_id       = alert_id,
                    narrative_text = narration.text,
                    cost_usd       = narration.cost_usd,
                )
            except Exception:
                logger.exception(
                    "narrator: update_narrative failed for alert %s (non-fatal)",
                    alert_id,
                )
    return results
