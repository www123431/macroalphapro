"""engine/agents/cross_review.py — Pattern 6 cross-agent DD orchestrator v1.

Phase 1 Task II.C of docs/decisions/research_agenda_2026-05-29.md. Closes the
"6 personas BUILT, 0 orchestrated" gap noted in commits 14abd75 + ba7de6c.

When a new strategy candidate appears (after engine.research.pipeline.run_gate
emits a verdict), this orchestrator routes it to ≥3 personas IN INDEPENDENT
SINGLE-TURN REVIEWS (Pattern 5 chat-history-sharing ban respected — each
persona receives only the candidate packet, never another persona's response).

The three v1 personas:
  - Devil's Advocate    — tries to kill the candidate (overfitting / sample
                          bias / mechanism story flaws); workload devils_advocate
  - Attribution Analyst — forensic decomposition (factor exposure / concentration
                          / cost reality); workload attribution_analyst
  - Risk Manager        — book-level implications (correlation / capacity /
                          weight implications); workload rm_agent

Output: a ReviewPacket with per-persona text + a consensus aggregation
(count_concerned / count_supportive / key_themes), persisted to
data/research/cross_review_ledger.jsonl. The packet is ADVISORY only — the
strict-gate verdict from run_gate is the source of truth and is never
overridden by persona consensus.

Two modes:
  - DETERMINISTIC (default + always-available): rule-based template reviews
    derived from gate_result. Used in unit tests and when no API key is
    configured. Every claim is evidence-cited by construction.
  - LLM: real single-turn Anthropic / DeepSeek calls per persona. Falls back
    to deterministic on per-persona basis if a call fails.

Doctrine constraints
--------------------
- 0-LLM-in-DECISION preserved: the gate verdict is unchanged regardless of
  persona consensus. Personas RECOMMEND; the human (or future strict
  governance) decides.
- No agent-to-agent autonomous interaction (Pattern 5 ban): each persona
  call is independent, no shared message history.
- Each call is cost-logged via engine.llm_cost_ledger.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "data" / "research" / "cross_review_ledger.jsonl"


# ─── Data shapes ─────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class CandidateContext:
    """The advisory packet shown to each persona. Small, structured, no PII."""
    name:            str
    mechanism:       str
    gate_result:     dict[str, Any]          # from engine.research.pipeline.run_gate
    returns_summary: dict[str, Any]          # n_months, ann_ret, ann_vol, sharpe, maxdd, hit_rate

    def to_prompt_block(self) -> str:
        """Render as a compact prompt-friendly summary."""
        gr = self.gate_result
        rs = self.returns_summary
        return (
            f"Candidate: {self.name}\n"
            f"Mechanism: {self.mechanism}\n"
            f"Sample: n_months={rs.get('n_months')} ({rs.get('range', 'n/a')})\n"
            f"Strict-gate result (n_trials={gr.get('n_trials')}):\n"
            f"  verdict             = {gr.get('verdict')}\n"
            f"  standalone_sharpe   = {gr.get('standalone_sharpe')}\n"
            f"  alpha_t_ff5umd      = {gr.get('alpha_t_ff5umd')}\n"
            f"  alpha_ann_ff5umd    = {gr.get('alpha_ann_ff5umd')}\n"
            f"  deflated_sr         = {gr.get('deflated_sr')}\n"
            f"  oos_sharpe          = {gr.get('oos_sharpe')}\n"
            f"  corr_with_book      = {gr.get('corr_with_book')}\n"
            f"Returns summary: ann_ret={rs.get('ann_ret')}  ann_vol={rs.get('ann_vol')}  "
            f"sharpe={rs.get('sharpe')}  maxdd={rs.get('maxdd')}  hit_rate={rs.get('hit_rate')}"
        )


@dataclasses.dataclass(frozen=True)
class PersonaReview:
    persona:  str        # display name
    workload: str        # one of engine.llm.call workloads
    agent_id: str        # one of ALLOWED_AGENT_IDS
    mode:     str        # "llm" | "deterministic" | "deterministic_fallback_<reason>"
    text:     str        # the review prose
    stance:   str        # "concerned" | "supportive" | "neutral" (parsed)
    themes:   list[str]  # extracted issue tags
    cost_usd: float = 0.0


@dataclasses.dataclass(frozen=True)
class ReviewPacket:
    candidate:     str
    timestamp:     str
    gate_verdict:  str
    n_reviews:     int
    reviews:       list[PersonaReview]
    consensus:     dict[str, Any]

    def to_jsonable(self) -> dict:
        return {
            "candidate":    self.candidate,
            "timestamp":    self.timestamp,
            "gate_verdict": self.gate_verdict,
            "n_reviews":    self.n_reviews,
            "consensus":    self.consensus,
            "reviews":      [dataclasses.asdict(r) for r in self.reviews],
        }


# ─── Persona role specs ─────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class PersonaSpec:
    display_name: str
    workload:     str
    agent_id:     str
    role_brief:   str             # one-liner mission for the prompt
    system_prompt: str            # full system prompt (tone + scope)


# Banned hedge vocabulary across all personas (tone discipline)
_BANNED_PHRASES = (
    "maybe", "perhaps", "could be", "might be", "probably", "possibly",
    "likely", "I think", "I feel", "seems to", "appears to",
)


_DA_SYSTEM = """You are the Devil's Advocate — a forensic critic whose job is to KILL weak strategy candidates before they reach the book. Your role-id is `devils_advocate_constrained_evidence`.

# Mission
Treat every candidate as guilty-until-proven-innocent. Cite specific numbers from the gate_result. Identify the SINGLE biggest reason this candidate should be rejected, or — if no kill-shot exists — say so explicitly.

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging. NO EMOJIS.
BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.

# Scope
Single-turn review of the candidate packet shown below. No tools. Return ONLY your review text — no JSON, no markdown headers.

# Output format
2-4 sentences. Cite at least 2 specific numbers from the packet. End with a one-line verdict line beginning with `STANCE:` followed by exactly one of {concerned, supportive, neutral}.

# Examples of valid stances
- STANCE: concerned   (you found a kill-shot)
- STANCE: supportive  (no kill-shot found; numbers are clean)
- STANCE: neutral     (ambiguous; defer to other reviewers)
"""

_AA_SYSTEM = """You are the Attribution Analyst — a forensic P&L decomposer. Your role-id is `attribution_analyst_forensic`.

# Mission
Decompose what's actually driving the returns of this candidate. Focus on: residual alpha vs FF5+UMD factors, concentration risk, cost reality (is the gross-vs-net gap suspicious), correlation with the existing book.

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging. NO EMOJIS.
BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.

# Scope
Single-turn review of the candidate packet shown below. No tools. Return ONLY your review text — no JSON, no markdown headers.

# Output format
2-4 sentences. Cite at least 2 specific numbers from the packet. End with a one-line verdict line beginning with `STANCE:` followed by exactly one of {concerned, supportive, neutral}.
"""

_RM_SYSTEM = """You are the Risk Manager — book-level risk gate. Your role-id is `head_of_risk_blackrock_slack`.

# Mission
Assess this candidate's BOOK-LEVEL implications: would adding it at 5-10% weight degrade the book's overall risk profile? Focus on correlation with the deployed book (PEAD leg), capacity at our scale, MaxDD contribution, and OOS Sharpe behavior.

# Tone
Terse. BlackRock-Slack grade. Active voice. No hedging. NO EMOJIS.
BANNED vocabulary: maybe, perhaps, could be, might be, probably, possibly, likely, I think, I feel, seems to, appears to.

# Scope
Single-turn review of the candidate packet shown below. No tools. Return ONLY your review text — no JSON, no markdown headers.

# Output format
2-4 sentences. Cite at least 2 specific numbers from the packet. End with a one-line verdict line beginning with `STANCE:` followed by exactly one of {concerned, supportive, neutral}.
"""


PERSONAS_V1: list[PersonaSpec] = [
    PersonaSpec(
        display_name = "Devil's Advocate",
        workload     = "devils_advocate",
        agent_id     = "devils_advocate",
        role_brief   = "kill-shot critique",
        system_prompt = _DA_SYSTEM,
    ),
    PersonaSpec(
        display_name = "Attribution Analyst",
        workload     = "attribution_analyst",
        agent_id     = "attribution_analyst",
        role_brief   = "P&L decomposition",
        system_prompt = _AA_SYSTEM,
    ),
    PersonaSpec(
        display_name = "Risk Manager",
        workload     = "rm_agent",
        agent_id     = "risk_manager",
        role_brief   = "book-level risk gate",
        system_prompt = _RM_SYSTEM,
    ),
]


# ─── Deterministic reviews (rule-based, always-available) ────────────────────

def _parse_stance(text: str) -> str:
    """Extract the STANCE: line value, default to neutral."""
    for line in reversed(text.splitlines()):
        s = line.strip().lower()
        if s.startswith("stance:"):
            v = s.split(":", 1)[1].strip()
            if v in ("concerned", "supportive", "neutral"):
                return v
    return "neutral"


def _extract_themes(text: str) -> list[str]:
    """Cheap theme extraction by keyword. Used for consensus aggregation."""
    text_l = text.lower()
    themes = []
    if any(k in text_l for k in ("overfit", "publication bias", "p-hack", "p hack")):
        themes.append("overfitting_risk")
    if any(k in text_l for k in ("decay", "stale", "faded")):
        themes.append("decay_risk")
    if any(k in text_l for k in ("correlation", "redundant", "duplicate")):
        themes.append("correlation_risk")
    if any(k in text_l for k in ("cost", "turnover", "slippage", "execution")):
        themes.append("cost_risk")
    if any(k in text_l for k in ("capacity", "liquidity", "size")):
        themes.append("capacity_risk")
    if any(k in text_l for k in ("sample", "window", "regime")):
        themes.append("sample_risk")
    if any(k in text_l for k in ("orthogonal", "diversification", "uncorrelated")):
        themes.append("diversification_benefit")
    return themes


def _strip_banned(text: str) -> str:
    """Replace banned hedge phrases with assertive equivalents (best-effort)."""
    out = text
    for phrase in _BANNED_PHRASES:
        out = out.replace(phrase, "")
        out = out.replace(phrase.capitalize(), "")
    return out


def _deterministic_devils_advocate(c: CandidateContext) -> str:
    gr = c.gate_result
    sh   = gr.get("standalone_sharpe")
    at   = gr.get("alpha_t_ff5umd")
    dsr  = gr.get("deflated_sr")
    oos  = gr.get("oos_sharpe")
    verdict = gr.get("verdict")

    kill_shots = []
    if sh is not None and sh < 0.4:
        kill_shots.append(f"standalone Sharpe {sh:.3f} below the 0.4 institutional floor")
    if at is not None and at < 2.0:
        kill_shots.append(f"alpha-t {at:.2f} not significant vs FF5+UMD")
    if dsr is not None and dsr < 0.5:
        kill_shots.append(f"deflated SR {dsr:.3f} far below the 0.90 multiple-testing bar")
    if oos is not None and oos < 0:
        kill_shots.append(f"OOS Sharpe {oos:.3f} negative — strategy fails post-IS")

    if kill_shots:
        body = f"Kill-shot: {kill_shots[0]}."
        if len(kill_shots) > 1:
            body += f" Secondary: {kill_shots[1]}."
        body += f" Gate verdict {verdict} is correct — reject."
        stance = "concerned"
    elif verdict == "GREEN":
        body = (f"No kill-shot found. Standalone Sharpe {sh:.3f}, alpha-t {at:.2f}, "
                f"DSR {dsr:.3f}, OOS {oos:.3f} all clear the strict bars. Deploy.")
        stance = "supportive"
    else:
        body = (f"Marginal case. Sharpe {sh}, alpha-t {at}, DSR {dsr}, OOS {oos}. "
                f"Verdict {verdict} stands but no decisive critique.")
        stance = "neutral"

    return body + f"\nSTANCE: {stance}"


def _deterministic_attribution(c: CandidateContext) -> str:
    gr = c.gate_result
    rs = c.returns_summary
    sh   = gr.get("standalone_sharpe")
    at   = gr.get("alpha_t_ff5umd")
    a_ann = gr.get("alpha_ann_ff5umd")
    cb   = gr.get("corr_with_book")
    annret = rs.get("ann_ret")

    parts = []
    if at is not None and abs(at) > 2.0:
        sign = "positive" if at > 0 else "negative"
        parts.append(f"Residual alpha vs FF5+UMD is {sign} and significant (t={at:.2f}, "
                     f"alpha_ann={a_ann}).")
    elif at is not None:
        parts.append(f"No residual alpha vs FF5+UMD (t={at:.2f}); returns "
                     f"{annret} fully explained by factor exposures.")
    if cb is not None:
        if abs(cb) < 0.2:
            parts.append(f"Book correlation {cb:.3f} — genuinely orthogonal addition.")
        elif abs(cb) < 0.5:
            parts.append(f"Book correlation {cb:.3f} — partial overlap with existing book.")
        else:
            parts.append(f"Book correlation {cb:.3f} — high overlap; diversification benefit limited.")

    if at is not None and at > 2.0 and cb is not None and abs(cb) < 0.5:
        stance = "supportive"
    elif at is not None and at < -2.0:
        stance = "concerned"
    else:
        stance = "neutral"

    body = " ".join(parts) if parts else f"Insufficient detail; sharpe {sh}."
    return body + f"\nSTANCE: {stance}"


def _deterministic_risk_manager(c: CandidateContext) -> str:
    gr = c.gate_result
    rs = c.returns_summary
    sh   = gr.get("standalone_sharpe")
    cb   = gr.get("corr_with_book")
    oos  = gr.get("oos_sharpe")
    maxdd = rs.get("maxdd")

    parts = []
    # Capacity at our scale: 100k paper, 5-10% candidate weight = 5-10k. Tiny.
    parts.append(f"At 5-10% candidate weight on a Sharpe {sh:.3f} sleeve, expected book "
                 f"Sharpe contribution is ~{0.05 * (sh or 0):+.3f} (5%) to ~{0.10 * (sh or 0):+.3f} (10%).")
    if oos is not None and (sh or 0) > 0 and oos < (sh or 0) * 0.5:
        parts.append(f"OOS Sharpe {oos:.3f} less than half the IS Sharpe {sh:.3f} — "
                     f"decay risk material.")
    if maxdd is not None and maxdd < -0.20:
        parts.append(f"MaxDD {maxdd*100:+.1f}% breaches the -20% book-level guard at "
                     f"any non-trivial weight.")
    if cb is not None and abs(cb) > 0.5:
        parts.append(f"Book correlation {cb:.3f} exceeds the 0.5 diversification bar — "
                     f"degrades the book's mechanism count.")

    has_kill = (maxdd is not None and maxdd < -0.20) or \
               (cb is not None and abs(cb) > 0.5) or \
               (sh is not None and sh < 0)
    stance = "concerned" if has_kill else "supportive" if (sh or 0) > 0.5 else "neutral"
    return " ".join(parts) + f"\nSTANCE: {stance}"


_DETERMINISTIC_HANDLERS = {
    "devils_advocate":     _deterministic_devils_advocate,
    "attribution_analyst": _deterministic_attribution,
    "rm_agent":            _deterministic_risk_manager,
}


# ─── LLM mode (optional) ─────────────────────────────────────────────────────

def _invoke_persona_llm(spec: PersonaSpec, ctx: CandidateContext) -> PersonaReview:
    try:
        from engine.llm.call import call
    except Exception as exc:
        return _invoke_persona_deterministic(
            spec, ctx, fallback_reason=f"sdk_load_failed_{type(exc).__name__}")
    try:
        res = call(
            workload     = spec.workload,
            system       = spec.system_prompt,
            user         = ctx.to_prompt_block(),
            agent_id     = spec.agent_id,
            max_tokens   = 512,
            cache_system = True,
            scope        = "cross_review_v1",
        )
        text   = _strip_banned((res.text or "").strip())
        stance = _parse_stance(text)
        themes = _extract_themes(text)
        return PersonaReview(
            persona  = spec.display_name,
            workload = spec.workload,
            agent_id = spec.agent_id,
            mode     = "llm",
            text     = text,
            stance   = stance,
            themes   = themes,
            cost_usd = float(getattr(res, "cost_usd", 0.0) or 0.0),
        )
    except Exception as exc:
        logger.warning("LLM persona %s failed (%s); falling back deterministic",
                       spec.display_name, exc)
        return _invoke_persona_deterministic(
            spec, ctx, fallback_reason=f"llm_failed_{type(exc).__name__}")


def _invoke_persona_deterministic(spec: PersonaSpec, ctx: CandidateContext,
                                   fallback_reason: str = "") -> PersonaReview:
    handler = _DETERMINISTIC_HANDLERS.get(spec.workload)
    if handler is None:
        text = f"{spec.display_name}: deterministic handler not implemented for workload {spec.workload}.\nSTANCE: neutral"
    else:
        text = handler(ctx)
    mode = "deterministic" if not fallback_reason else f"deterministic_fallback_{fallback_reason}"
    return PersonaReview(
        persona  = spec.display_name,
        workload = spec.workload,
        agent_id = spec.agent_id,
        mode     = mode,
        text     = text,
        stance   = _parse_stance(text),
        themes   = _extract_themes(text),
        cost_usd = 0.0,
    )


# ─── Consensus aggregation ───────────────────────────────────────────────────

def _aggregate_consensus(reviews: list[PersonaReview]) -> dict:
    n_concerned  = sum(1 for r in reviews if r.stance == "concerned")
    n_supportive = sum(1 for r in reviews if r.stance == "supportive")
    n_neutral    = sum(1 for r in reviews if r.stance == "neutral")
    all_themes = sorted({t for r in reviews for t in r.themes})
    if n_concerned > n_supportive:
        summary = "majority_concerned"
    elif n_supportive > n_concerned:
        summary = "majority_supportive"
    elif n_concerned == n_supportive and n_concerned > 0:
        summary = "split"
    else:
        summary = "no_consensus"
    return {
        "n_concerned":  n_concerned,
        "n_supportive": n_supportive,
        "n_neutral":    n_neutral,
        "summary":      summary,
        "themes":       all_themes,
    }


# ─── Public entry ───────────────────────────────────────────────────────────

def run_cross_review(candidate: CandidateContext,
                     use_llm: bool = False,
                     log: bool = True,
                     personas: list[PersonaSpec] | None = None) -> ReviewPacket:
    """Run N personas in independent single-turn reviews, return + log the packet.

    use_llm=False is the doctrine-safe default — always works without API.
    use_llm=True attempts real LLM calls per persona; falls back deterministic
    on per-persona basis on failure.
    """
    plist = personas if personas is not None else PERSONAS_V1
    reviews: list[PersonaReview] = []
    for spec in plist:
        if use_llm:
            reviews.append(_invoke_persona_llm(spec, candidate))
        else:
            reviews.append(_invoke_persona_deterministic(spec, candidate))

    consensus = _aggregate_consensus(reviews)
    packet = ReviewPacket(
        candidate    = candidate.name,
        timestamp    = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        gate_verdict = str(candidate.gate_result.get("verdict", "UNKNOWN")),
        n_reviews    = len(reviews),
        reviews      = reviews,
        consensus    = consensus,
    )

    if log:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(packet.to_jsonable(), ensure_ascii=False) + "\n")

    return packet


def read_ledger(limit: int = 50) -> list[dict]:
    """Read the cross-review ledger most-recent-first."""
    if not LEDGER_PATH.exists():
        return []
    rows = [json.loads(x) for x in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[-limit:][::-1]
