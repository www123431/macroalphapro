"""
engine/agents/dq_inspector/narrator.py — Phase 7 narrative layer.

Generates one-paragraph BlackRock-Slack-tone prose per DQ breach.
Updates DataQualityAlert.narrative_text via persist.update_narrative.

Mirrors engine.agents.risk_manager.narrator 1:1 in structure (facade +
pluggable backend + DeterministicNarrator default + GeminiFlash
deferred + banned-phrases discipline). Per spec id=70 §2.5.

DOCTRINE compliance:
  - LLM (when wired) NEVER runs before deterministic gates produced the
    breach list. Narrator only DESCRIBES; it cannot flip halt verdict.
  - Banned-phrases regex enforced at output stage regardless of backend.
  - DataQualityAlert.narrative_text + narrative_cost_usd populated via
    persist.update_narrative (cost = 0 for deterministic).

Switch via `DQ_INSPECTOR_NARRATOR_BACKEND` env var:
  - "deterministic" (default — this commit)
  - "gemini_flash"  (deferred)
"""
from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from engine.agents.dq_inspector.gates import Breach
    from engine.agents.dq_inspector.agent import DQInspectorRunResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Tone discipline — banned phrases (same set as Risk Manager)
# ──────────────────────────────────────────────────────────────────────────────
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
)

_BANNED_RE = re.compile("|".join(BANNED_PHRASES), re.IGNORECASE)


def contains_banned_phrase(text: str) -> Optional[str]:
    """Return the first banned phrase match, or None if text is clean."""
    m = _BANNED_RE.search(text)
    return m.group(0) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# PersonaContext — forward-compatible stub for Persona Voice Layer sprint
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class PersonaContext:
    """Per-call persona context. Phase 7 uses role_id only.

    Future Persona Voice Layer sprint will populate the other fields
    from engine.agents.capabilities.AGENT_CAPABILITIES.
    """
    role_id:                str   = "data_quality_inspector_blackrock_slack"
    voice_phrase_library:   Optional[dict] = None
    cross_agent_references: tuple[str, ...] = ()
    episodic_memory_hits:   tuple[str, ...] = ()


# ──────────────────────────────────────────────────────────────────────────────
# Backend abstraction
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class NarrationResult:
    """One narration call's output."""
    text:            str
    backend:         str
    cost_usd:        float
    rejected_drafts: tuple[str, ...] = ()


class _NarratorBackend:
    """Protocol — generate(breach, context) → NarrationResult."""
    name: str = "abstract"

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic template narrator
# ──────────────────────────────────────────────────────────────────────────────
class DeterministicNarrator(_NarratorBackend):
    """Hand-written templates per mode. Zero cost, no network, no LLM.

    Active voice, no hedging, terse — matches BlackRock-Slack tone by
    construction. Each template ends with an action verb directing the
    operator (re-fetch / investigate / wait-for-release / etc.).
    """
    name = "deterministic"

    _TEMPLATES: dict[str, str] = {
        "1":   ("FRED series freshness breach: {affected_first} last updated "
                "{extra_last_obs} is {observed_value:.0f} business days stale "
                "(max {threshold:.0f}). Daily macro pipeline depends on this "
                "series — refresh API + re-fetch before next orchestrator run."),
        "2":   ("yfinance bab_compat cache stale: {affected_first} mtime "
                "{extra_mtime} is {observed_value:.0f} trading days old "
                "(max {threshold:.0f}). K1 BAB signal generation depends on "
                "this cache. Halt issued — rebuild cache via "
                "engine.factors.bab_compat.compute_bab_signal before re-submit."),
        "3":   ("D-PEAD signal panel parquet stale: {affected_first} mtime "
                "{extra_mtime} is {observed_value:.0f} calendar days old "
                "(max {threshold:.0f}). Soft warn — refresh panel cache via "
                "the D-PEAD daily script in the next maintenance window."),
        "4":   ("S&P 500 reconstitution feed stale: last detected_at "
                "{extra_last_detected} is {observed_value:.0f} calendar days "
                "old (max {threshold:.0f}). Soft warn — confirm Wikipedia "
                "and EDGAR fetchers are running on their scheduled cron."),
        "5":   ("K1 ETF universe coverage degraded: {observed_value:.1%} of "
                "the {extra_expected:.0f}-ETF spec coverage (min "
                "{threshold:.0%}). Halt issued — investigate yfinance batch "
                "failures or delisted-ticker substitution before re-submit."),
        "6":   ("D-PEAD stock universe coverage degraded: {observed_value:.1%} "
                "of the {extra_expected:.0f}-stock spec coverage (min "
                "{threshold:.0%}). Halt issued — rebuild rdq cache from the "
                "earnings panel before re-submit."),
        "7":   ("Price tick anomaly: {affected_first} 1-day return "
                "{extra_signed_return:+.2%} exceeds {threshold:.0%} cap for "
                "{extra_ticker_class} class. Halt issued — confirm with "
                "Bloomberg or alternate source whether the move is real "
                "(split / merger event) or a data pipeline error."),
        "8":   ("Volume dropoff: {affected_first} today vs 60d median "
                "{observed_value:.1%} of normal (min {threshold:.0%}). "
                "Soft warn — possible delisting or corporate action; check "
                "before next rebalance includes the name."),
        "9":   ("NaN burst across active universe: {observed_value:.1%} of "
                "tickers have NaN close (max {threshold:.0%}). Halt issued "
                "— investigate price feed before orchestrator runs; book "
                "generated on this state would be unreliable."),
        "10a": ("Row-count drop: {extra_today_rows:.0f} rows today vs "
                "{extra_yesterday_rows:.0f} yesterday ({observed_value:.0%} "
                "drop, max {threshold:.0%}). Soft warn — daily_batch may "
                "have skipped strategies; review the persistence log."),
        "10b": ("Catastrophic row-count drop: {extra_today_rows:.0f} rows "
                "today vs {extra_yesterday_rows:.0f} yesterday "
                "({observed_value:.0%} drop, max {threshold:.0%}). Halt "
                "issued; legacy CB escalated to SEVERE — tomorrow's run "
                "requires manual_reset after root-cause investigation."),
    }

    _FALLBACK_TEMPLATE = (
        "DQ breach in mode {mode_id} ({severity}): {rule_description}. "
        "Spec anchor: {spec_anchor}."
    )

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        template = self._TEMPLATES.get(breach.mode_id, self._FALLBACK_TEMPLATE)
        try:
            text = template.format(
                mode_id             = breach.mode_id,
                severity            = breach.severity,
                rule_description    = breach.rule_description,
                observed_value      = breach.observed_value if breach.observed_value is not None else 0.0,
                threshold           = breach.threshold if breach.threshold is not None else 0.0,
                affected_first      = breach.affected[0] if breach.affected else "the source",
                spec_anchor         = breach.spec_anchor,
                # Mode-specific extras (None-safe via .get with default)
                extra_last_obs      = breach.extra.get("last_obs_date", "?"),
                extra_mtime         = breach.extra.get("mtime", "?"),
                extra_last_detected = breach.extra.get("last_detected_at", "?"),
                extra_expected      = breach.extra.get("expected_n", 0),
                extra_signed_return = breach.extra.get("signed_return", 0.0),
                extra_ticker_class  = breach.extra.get("ticker_class", "?"),
                extra_today_rows    = breach.extra.get("today_rows", 0),
                extra_yesterday_rows= breach.extra.get("yesterday_rows", 0),
            )
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("DQ DeterministicNarrator: template format error: %s; fallback used", exc)
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
                "DQ DeterministicNarrator: template for mode %s contains banned phrase %r; "
                "this is a template authoring bug — please clean the template",
                breach.mode_id, bad,
            )
        return NarrationResult(
            text     = text,
            backend  = self.name,
            cost_usd = 0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Gemini Flash backend — DEFERRED
# ──────────────────────────────────────────────────────────────────────────────
class GeminiFlashNarrator(_NarratorBackend):
    """Vertex Gemini 2.5 Flash narrator with banned-phrases regex enforcement.

    NOT YET IMPLEMENTED. Same deferred pattern as
    engine.agents.risk_manager.narrator.GeminiFlashNarrator; both will
    land together in a focused LLM-narrator commit when Vertex auth +
    cost ledger are verified end-to-end.
    """
    name = "gemini_flash"

    def generate(self, breach: "Breach", context: PersonaContext) -> NarrationResult:
        raise NotImplementedError(
            "DQ GeminiFlashNarrator deferred to next focused commit. "
            "Set DQ_INSPECTOR_NARRATOR_BACKEND=deterministic for now."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Backend selection
# ──────────────────────────────────────────────────────────────────────────────
def _select_backend(name: Optional[str] = None) -> _NarratorBackend:
    backend_name = (name or os.environ.get("DQ_INSPECTOR_NARRATOR_BACKEND")
                    or "deterministic")
    if backend_name == "deterministic":
        return DeterministicNarrator()
    if backend_name == "gemini_flash":
        return GeminiFlashNarrator()
    raise ValueError(
        f"Unknown DQ narrator backend {backend_name!r}. "
        f"Set DQ_INSPECTOR_NARRATOR_BACKEND to one of: deterministic / gemini_flash"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def narrate_breach(
    breach:    "Breach",
    *,
    context:   Optional[PersonaContext] = None,
    backend:   Optional[str] = None,
) -> NarrationResult:
    """Generate prose for a single DQ breach. Default = deterministic backend."""
    backend_impl = _select_backend(backend)
    ctx = context or PersonaContext()
    return backend_impl.generate(breach, ctx)


def narrate_run_result(
    dq_result:    "DQInspectorRunResult",
    *,
    context:      Optional[PersonaContext] = None,
    backend:      Optional[str] = None,
    update_db:    bool = True,
) -> list[NarrationResult]:
    """Generate prose for every breach in a DQInspectorRunResult.

    If update_db is True (default), persists each narrative to the
    DataQualityAlert table via persist.update_narrative.

    Returns list[NarrationResult] in breach order — caller can render
    them in UI without re-loading DB.
    """
    if not dq_result.breaches:
        return []

    results: list[NarrationResult] = []
    for breach, alert_id in zip(dq_result.breaches, dq_result.audit_alert_ids):
        narration = narrate_breach(breach, context=context, backend=backend)
        results.append(narration)
        if update_db and alert_id:
            from engine.agents.dq_inspector.persist import update_narrative
            try:
                update_narrative(
                    date           = datetime.date.fromisoformat(dq_result.today_iso),
                    alert_id       = alert_id,
                    narrative_text = narration.text,
                    cost_usd       = narration.cost_usd,
                )
            except Exception:
                logger.exception(
                    "DQ narrator: update_narrative failed for alert %s (non-fatal)",
                    alert_id,
                )
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Run-level data-quality verdict (junior-analyst layer) — explains CLEAN too (the
# per-breach templates only fire on a breach). Deterministic; 0-LLM; cannot flip a verdict.
# ──────────────────────────────────────────────────────────────────────────────
def narrate_dq_summary(breaches: list) -> str:
    """One-line run-level data-quality verdict: Why CLEAN/WARN/HALT + the driving check(s)."""
    def _sev(b):
        return getattr(b, "severity", "")
    hard = [b for b in breaches if _sev(b) == "HARD_HALT"]
    warn = [b for b in breaches if _sev(b) == "SOFT_WARN"]

    def _desc(b):
        aff = getattr(b, "affected", None) or []
        tag = f" ({aff[0]})" if aff else ""
        return f"mode {b.mode_id}{tag}: {b.rule_description}"

    if hard:
        text = ("DQ Inspector — HARD HALT. " + "; ".join(_desc(b) for b in hard) +
                ". Data is not fit to trade on — the batch is blocked. Refresh the flagged source(s) "
                "before re-run.")
    elif warn:
        text = ("DQ Inspector — WARN. " + "; ".join(_desc(b) for b in warn) +
                ". Soft staleness only — the batch proceeds; refresh the flagged source(s) to clear it.")
    else:
        text = ("DQ Inspector — CLEAN. All pre-batch freshness checks pass (FRED macro series, BAB "
                "cache, PEAD panel, S&P 500 feed are current). Data cleared the batch to run.")
    bad = contains_banned_phrase(text)
    if bad:
        logger.error("narrate_dq_summary: banned phrase %r — template bug.", bad)
    return text
