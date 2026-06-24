"""
engine/llm_cost_ledger.py — Unified LLM cost recording (Sprint 2B, 2026-05-10).

Single source of truth for LLM cost across all agents. Append-only JSON Lines
ledger at `data/llm_cost_ledger.jsonl`. Multi-dimensional query API.

Per Sprint 2A mini-spec (audit-validated 2026-05-10):
  - Implements spec id=53 §4.1 cost-persistence requirement (Tool 1 originally
    referenced fictional `LLMCallLog` ORM; this module supersedes that).
  - Preserves dual budget governance:
    * Operations agents (R-audit / S6 / RAG / DeepSeek) — runtime-tunable via
      engine.llm_budget; this module only records, does not enforce.
    * Spec-locked agents (ETF / FOMC / Tool 1) — hash-locked caps remain
      enforced inside their own modules; this module supplies the trailing-365d
      sum for their enforcement check.
  - Zero double-count with engine.key_pool (verified 2026-05-10: key_pool tracks
    RPM/RPD/quota/health, never cost_usd).

JSONL choice: append-only, crash-safe, stream-friendly, industry-standard
(LangSmith / LangFuse / OpenTelemetry use the same pattern).

Concurrency: portalocker for cross-process safe append (Streamlit + scheduler
cron may write concurrently).
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import portalocker

logger = logging.getLogger(__name__)


# ── Storage ─────────────────────────────────────────────────────────────────
_REPO_ROOT          = Path(__file__).resolve().parent.parent
_DATA_DIR           = _REPO_ROOT / "data"
_LEDGER_PATH        = _DATA_DIR / "llm_cost_ledger.jsonl"
_LOCK_PATH          = _DATA_DIR / "llm_cost_ledger.jsonl.lock"

# Size monitoring (warn-only; rotation deferred per Sprint 2A §8 risk register)
_LEDGER_SIZE_WARN_BYTES = 50 * 1024 * 1024   # 50 MB

# Allowed agent_ids (closed enumeration so a typo can't silently shard data)
ALLOWED_AGENT_IDS: frozenset[str] = frozenset({
    "r_audit",
    "s6_anomaly",
    "rag_synthesis",
    "deepseek",                # cross-agent DeepSeek calls (probe / Tool 2/4 future)
    "etf_holdings",            # spec id=49
    "fomc_override",           # spec id=48 (parked but ledger preserved)
    "tool1_decision_lineage",  # spec id=53 (cost persistence per §4.1)
    "spec_drafter",            # P3.5
    "macro_research",          # MACRO-V/R/P
    "ops_watchdog",            # spec id=63 (Ops Watchdog v1.0, daily 06:10 SGT)
    "d_pead_plus_llm_extractor",  # spec id=74 Sprint I (LLM-as-INPUT-FEATURE Pattern 1)
    "forensic_news_context",       # Sprint H follow-up: DD investigation LLM news summarizer (FORENSIC layer only, 0-LLM-in-DECISION preserved)
    # 2026-05-19 Persona MVP additions (multi-provider LLM router):
    "risk_manager",            # spec id=69 — RM Persona MVP tool-using agent
    "dq_inspector",            # spec id=70 — DQ narrator (when LLM backend wired)
    "devils_advocate",         # future DA agent (DeepSeek V4 Pro primary)
    "anomaly_sentinel",        # per-ticker forensic z-score + cluster persona
    "attribution_analyst",     # P&L attribution + sleeve / strategy decomposition persona
    "audit_recorder",          # auto_audit findings + spec amendment trail persona
    "chief_of_staff",          # Supervisor pattern — routes to 6 specialists (spec id=74)
    "decay_sentinel",          # two-mechanism book decay monitor — deterministic core, LLM narrator deferred
    "chat_ask",                # 2026-06-02 — /api/research/chat/ask scoped-RAG Q&A (ChatFloater + Cmd-J + /chat full page share this id)
    "research_ops_paper_scorer",   # 2026-06-02 — L2.2 Haiku-4.5 per-paper relevance scorer
    "research_ops_weekly_digest",  # 2026-06-02 — L2.3 Sonnet-4.6 weekly cross-paper digest
    "papers_curator_filter",       # 2026-06-05 — Phase 1.5 Employee A: 1-line tradable-factor judgment per crawled arxiv candidate (Deepseek V4 Pro)
    "papers_curator_summary",      # 2026-06-05 — Phase 1.5b Employee A: richer 5-field summary on YES-filtered candidates (eager) + lazy on NO when user requests
    "papers_curator_synthesis",    # 2026-06-06 — Phase 2.0 step 2 Employee A: cross-source synthesis emitting hypothesis candidates with extraction_method=LLM_SYNTHESIS (Sonnet 4.6 for quality)
    "strengthener_review",         # 2026-06-06 — Phase 2.0 step 11a Employee B: per-hypothesis second-pass review (APPROVE_FOR_PIPELINE / REJECT / DOCTRINE_AMENDMENT_NEEDED). Sonnet 4.6.
    "chief_of_staff_memo",         # 2026-06-06 — Phase 2.0 step 14b chief_of_staff: weekly 5-bullet memo summarizing D/A/B activity for principal's 30-second scan. Sonnet 4.6, ~$0.05/week.
    "strengthener_propose",        # 2026-06-07 — Stage B P3a Employee B: active per-sleeve strengthen proposer (single Sonnet 4.6 call producing 0-3 improvement-candidate hypotheses per deployed sleeve). Cost ceiling ~$0.05/sleeve × 13 deployed × weekly = ~$0.65/wk.
    "strengthener_spec_extract",   # 2026-06-07 — Procedural auto-spec dispatcher: Sonnet maps a procedural Hypothesis's test_methodology to a CONTROLLED dispatch_kind + structured args. LLM does NOT write code; deterministic Python dispatcher takes over from the JSON spec. ~$0.02/hypothesis.
    "papers_tier_classifier",      # 2026-06-07 — Stage C Phase A: batch tier classification of papers_registry entries into T1/T2/T3/UNCLASSIFIED per the three-libraries doctrine. ~$0.10-0.30 per batch.
    "papers_anchor_summary",       # 2026-06-07 — Stage C Phase B: 1-line meta-summary per T2_ANCHOR paper explaining what makes it a citation anchor for its mechanism class. ~$0.001/paper, 23 anchors × = $0.025 total.
    "strengthener_factor_spec",    # 2026-06-08 — Tier C-1 factor backtest spec extractor: Sonnet maps a B-approved factor Hypothesis (predicted_direction != zero) to a STRUCTURED backtest SPEC JSON (signal_kind enum + universe + dates + inputs + weighting + pit_audits). LLM does NOT write code; dispatcher (C-2) reads SPEC + invokes engine.factor_lab template. Pattern-5-compliant single call + strict schema. ~$0.03/hypothesis.
    "strengthener_self_doubt",     # 2026-06-08 — Tier C L3-2 Self-Doubt: post-dispatch Sonnet call scoring system confidence in verdict (0-0.99 never 1.0) + listing per-verdict caveats grounded in known silent bugs (B0-B7). Anti-trust UX per DE Shaw "if system produces too many confident answers, the system has a bug". Strict JSON tool schema. ~$0.04/dispatch.
    "strengthener_claim_shape_router",  # 2026-06-13 — Tier C Phase 2.1 claim-shape Stage 0 classifier. Tight pre-routing of hypothesis claim into one of 11 canonical shapes (CROSS_SECTIONAL_ALPHA / SPANNING / VRP / FACTOR_COMBINATION / PORTFOLIO_OVERLAY / EVENT_DRIFT / TIME_SERIES_MOMENTUM / CARRY / DECAY_STUDY / CAPACITY / FACTOR_STRUCTURE) + confidence + 1-line rationale. Prevents BUG-2-style Sonnet drift where SPANNING claims got stretched into factor_combination specs. ~$0.001/call.
    "pre_mortem",                  # 2026-06-14 — α Pre-Mortem Generator (Stigler 1973 adversarial review; Kahneman pre-mortem technique). Single skeptic-persona Sonnet call per hypothesis BEFORE strict-gate dispatch, producing a structured list of 3-7 concrete failure modes the gate MIGHT MISS. Inputs: hypothesis + family belief + graveyard top-3 collisions + n_trials counter + known silent bugs. Output: list[FailureMode] with severity + check_suggestion. ~$0.05/hypothesis. Reactive subscriber, not a debate.
    "cross_domain_transfer",       # 2026-06-14 — β Cross-Domain Transfer Generator (Frazzini-Pedersen 2018 "70% institutional alpha = enhance, not new factor"). Single cross-asset-thinker-persona Sonnet call per deployed GREEN sleeve, producing 1-2 testable mechanism transfers to OTHER asset classes (e.g. equity-VRP → bond vol surface). Output: list[TransferProposal] each routing to enhance pipeline. ~$0.30/sleeve, monthly cron.
    "replication_checker",         # 2026-06-14 — γ Replication Checker (Hou-Xue-Zhang 2020 q-factor catalog ~50% lit-replication failure + McLean-Pontiff 2016 post-pub decay catalog). Single lit-aware specialist persona Sonnet call per hypothesis, producing 1-3 ReplicationFlag items matching the hyp to known anomaly papers + their replication status. ~$0.05/hyp. Complements α (general adversarial) with literature-replication evidence.
    "brainstorm_divergent",        # 2026-06-14 — Brainstorm Layer 3 experience-conditioned divergent generator. Single Sonnet call per (seed pack, lesson context) combo, producing 3-5 structured BrainstormIdea items each with MANDATORY Popper-falsifier. Pattern-5-compliant single agent + strict schema. ~$0.20-0.25/session. Multi-provider augmentation (DeepSeek) in Phase 3.
    "papers_curator_claim_type_router",  # 2026-06-21 — W4-piece-2 ClaimType Stage 0 router LLM fallback (Haiku). Deterministic keyword router catches ~17%; the remaining 83% UNKNOWN are routed here. Single Haiku call per paper, controlled-enum output, ~$0.001/paper. Backfill cost ~$0.40 for 527 UNKNOWN papers; ongoing ~$0.001/paper for new ingest.
})

# 2026-05-19 — added "anthropic" for Persona MVP (Sonnet 4.6 / Haiku 4.5).
# See [[feedback-llm-provider-role-specialization-2026-05-19]] for the
# multi-provider role-specialization decision.
ALLOWED_PROVIDERS: frozenset[str] = frozenset({"gemini", "deepseek", "anthropic"})


# ── Public dataclass ────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class CostEntry:
    """One LLM call recorded to the unified ledger."""
    ts:                str        # ISO-8601 UTC, "2026-05-10T12:34:56Z"
    agent_id:          str        # one of ALLOWED_AGENT_IDS
    provider:          str        # one of ALLOWED_PROVIDERS
    model:             str        # e.g. "gemini-2.5-flash" / "deepseek-v4-flash"
    prompt_tokens:     int
    completion_tokens: int
    cost_usd:          float
    latency_ms:        int
    scope:             str = ""           # optional sub-scope, e.g. "react_step"
    extra:             dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_jsonl(self) -> str:
        """Serialize as a single JSONL line (no trailing newline)."""
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CostEntry":
        return cls(
            ts                = str(d.get("ts", "")),
            agent_id          = str(d.get("agent_id", "")),
            provider          = str(d.get("provider", "")),
            model             = str(d.get("model", "")),
            prompt_tokens     = int(d.get("prompt_tokens", 0) or 0),
            completion_tokens = int(d.get("completion_tokens", 0) or 0),
            cost_usd          = float(d.get("cost_usd", 0.0) or 0.0),
            latency_ms        = int(d.get("latency_ms", 0) or 0),
            scope             = str(d.get("scope", "") or ""),
            extra             = dict(d.get("extra") or {}),
        )


# ── Path override hook (for tests) ──────────────────────────────────────────
def _ledger_path() -> Path:
    """Indirected so tests can monkeypatch tmp_path without disk pollution
    (per feedback_test_isolation_no_disk_pollution rule)."""
    return _LEDGER_PATH


def _lock_path() -> Path:
    return _LOCK_PATH


# ── Recording API ───────────────────────────────────────────────────────────
def record_call(
    *,
    agent_id:          str,
    provider:          str,
    model:             str,
    prompt_tokens:     int,
    completion_tokens: int,
    cost_usd:          float,
    latency_ms:        int,
    scope:             str = "",
    extra:             Optional[dict[str, Any]] = None,
    ts:                Optional[str] = None,
) -> CostEntry:
    """Append-only thread/process-safe cost recording.

    Validates agent_id + provider against closed enumeration to prevent
    silent typo-induced data sharding. Caller responsible for cost_usd math
    (this module records, does not compute).

    Args:
        agent_id:  one of ALLOWED_AGENT_IDS (raises if not)
        provider:  one of ALLOWED_PROVIDERS (raises if not)
        model:     vendor-specific model id string
        prompt_tokens / completion_tokens / cost_usd / latency_ms: usage
        scope:     optional intra-agent sub-scope (free-form string)
        extra:     optional metadata dict (must be JSON-serializable)
        ts:        override timestamp (test-only); default = now (UTC)

    Returns:
        The CostEntry that was recorded (caller may inspect for assertions).
    """
    if agent_id not in ALLOWED_AGENT_IDS:
        raise ValueError(
            f"agent_id {agent_id!r} not in ALLOWED_AGENT_IDS "
            f"{sorted(ALLOWED_AGENT_IDS)} — typo? Add to allowlist if new agent."
        )
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(
            f"provider {provider!r} not in ALLOWED_PROVIDERS "
            f"{sorted(ALLOWED_PROVIDERS)} — typo? Add to allowlist if new provider."
        )
    if cost_usd < 0:
        raise ValueError(f"cost_usd must be non-negative; got {cost_usd}")

    entry = CostEntry(
        ts                = ts or _utc_iso_now(),
        agent_id          = agent_id,
        provider          = provider,
        model             = model,
        prompt_tokens     = int(prompt_tokens),
        completion_tokens = int(completion_tokens),
        cost_usd          = round(float(cost_usd), 8),
        latency_ms        = int(latency_ms),
        scope             = scope,
        extra             = dict(extra or {}),
    )

    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = entry.to_jsonl() + "\n"

    # Cross-process safe append. portalocker.Lock blocks competing writers.
    with portalocker.Lock(str(_lock_path()), timeout=10):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    # Size warning (warn-only; rotation deferred per Sprint 2A §8)
    try:
        size = path.stat().st_size
        if size > _LEDGER_SIZE_WARN_BYTES:
            logger.warning(
                "llm_cost_ledger.jsonl size %d bytes exceeds %d byte warn threshold "
                "— consider rotation",
                size, _LEDGER_SIZE_WARN_BYTES,
            )
    except OSError:
        pass

    return entry


def _utc_iso_now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ── Read API ────────────────────────────────────────────────────────────────
def _iter_entries() -> Iterable[CostEntry]:
    """Yield every CostEntry from the ledger, skipping malformed lines.

    Resilient to partial writes / corruption: bad lines are logged and
    skipped, not raised. (Append-only + JSONL = corruption only ever
    affects the last line in practice.)
    """
    path = _ledger_path()
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
                yield CostEntry.from_dict(d)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning(
                    "llm_cost_ledger.jsonl line %d malformed (%s); skipping",
                    line_no, exc,
                )


def get_calls(
    *,
    agent_id: Optional[str] = None,
    provider: Optional[str] = None,
    since:    Optional[datetime.date] = None,
    until:    Optional[datetime.date] = None,
    limit:    Optional[int] = None,
) -> list[CostEntry]:
    """Query interface; all filters AND-combined.

    Args:
        agent_id: filter by agent_id (exact match)
        provider: filter by provider (exact match)
        since:    inclusive lower-bound date (UTC) for entry.ts
        until:    inclusive upper-bound date (UTC) for entry.ts
        limit:    return at most N most-recent entries (None = all)

    Returns:
        List of CostEntry, ordered by ts ascending. (limit truncates from tail.)
    """
    out: list[CostEntry] = []
    for e in _iter_entries():
        if agent_id is not None and e.agent_id != agent_id:
            continue
        if provider is not None and e.provider != provider:
            continue
        e_date = _entry_date(e)
        if since is not None and e_date is not None and e_date < since:
            continue
        if until is not None and e_date is not None and e_date > until:
            continue
        out.append(e)
    if limit is not None and limit > 0:
        out = out[-limit:]
    return out


def _entry_date(e: CostEntry) -> Optional[datetime.date]:
    """Parse entry.ts to a date for since/until filtering."""
    try:
        # Accept "2026-05-10T12:34:56Z" or "2026-05-10T12:34:56+00:00"
        ts = e.ts.rstrip("Z")
        dt = datetime.datetime.fromisoformat(ts)
        return dt.date()
    except (ValueError, AttributeError):
        return None


def get_total_by_agent(
    *,
    since: Optional[datetime.date] = None,
    until: Optional[datetime.date] = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate by agent_id within optional date window.

    Returns:
        {agent_id: {total_usd, calls, last_ts, providers: {provider: usd}}}
    """
    agg: dict[str, dict[str, Any]] = {}
    for e in get_calls(since=since, until=until):
        a = agg.setdefault(e.agent_id, {
            "total_usd": 0.0,
            "calls":     0,
            "last_ts":   None,
            "providers": {},
        })
        a["total_usd"] = round(a["total_usd"] + e.cost_usd, 8)
        a["calls"]    += 1
        if a["last_ts"] is None or e.ts > a["last_ts"]:
            a["last_ts"] = e.ts
        a["providers"][e.provider] = round(
            a["providers"].get(e.provider, 0.0) + e.cost_usd, 8,
        )
    return agg


def get_total_by_provider(
    *,
    since: Optional[datetime.date] = None,
    until: Optional[datetime.date] = None,
) -> dict[str, float]:
    """Aggregate cost by provider within optional date window."""
    agg: dict[str, float] = {}
    for e in get_calls(since=since, until=until):
        agg[e.provider] = round(agg.get(e.provider, 0.0) + e.cost_usd, 8)
    return agg


def get_trailing_365d_total(
    agent_id: str,
    as_of: Optional[datetime.date] = None,
) -> float:
    """Trailing 365-day cost for one agent.

    Used by spec-locked agents (etf_holdings / fomc_override) for their
    annual-cap enforcement. They keep their hash-locked ANNUAL_BUDGET_USD
    constant; this function only supplies the trailing sum.

    Args:
        agent_id: which agent to sum
        as_of:    reference date (default = today UTC)

    Returns:
        Sum of cost_usd over the last 365 days (inclusive of as_of).
    """
    if as_of is None:
        as_of = datetime.datetime.utcnow().date()
    cutoff = as_of - datetime.timedelta(days=365)
    return round(
        sum(e.cost_usd for e in get_calls(agent_id=agent_id, since=cutoff, until=as_of)),
        8,
    )


def get_total_today(agent_id: Optional[str] = None) -> float:
    """Cost spent today (UTC).

    Optional agent_id filter; without it, returns sum across all agents.
    Used by RAG synthesis daily-budget check.
    """
    today = datetime.datetime.utcnow().date()
    return round(
        sum(e.cost_usd for e in get_calls(agent_id=agent_id, since=today, until=today)),
        8,
    )


def get_lifetime_total(agent_id: Optional[str] = None) -> float:
    """All-time cost, optionally filtered to one agent."""
    return round(
        sum(e.cost_usd for e in get_calls(agent_id=agent_id)),
        8,
    )


def get_call_count(agent_id: Optional[str] = None) -> int:
    """All-time call count, optionally filtered to one agent."""
    return sum(1 for _ in get_calls(agent_id=agent_id))


# ── Diagnostics ─────────────────────────────────────────────────────────────
def integrity_check() -> dict[str, Any]:
    """Parse-all integrity check; returns counts and any malformed lines.

    Useful for startup verification / Tier R audit hook.
    """
    path = _ledger_path()
    if not path.exists():
        return {
            "exists":         False,
            "total_lines":    0,
            "valid_entries":  0,
            "malformed_lines": [],
            "size_bytes":     0,
        }
    total = 0
    valid = 0
    malformed: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                json.loads(raw)
                valid += 1
            except json.JSONDecodeError:
                malformed.append(line_no)
    return {
        "exists":         True,
        "total_lines":    total,
        "valid_entries":  valid,
        "malformed_lines": malformed,
        "size_bytes":     path.stat().st_size,
        "size_warn":      path.stat().st_size > _LEDGER_SIZE_WARN_BYTES,
    }
