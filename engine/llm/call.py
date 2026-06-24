"""
engine/llm/call.py — single entrypoint for all project LLM calls.

`call(workload, system, user, tools, agent_id, ...)` selects the
provider + model based on workload, runs the call, records cost to
engine.llm_cost_ledger, and returns a provider-agnostic LLMCallResult.

Workload routing table (LOCKED 2026-05-19 per
[[feedback-llm-provider-role-specialization-2026-05-19]]):

  Workload          Provider     Model               Rationale
  ───────────────── ──────────── ─────────────────── ──────────────────
  narrator          anthropic    claude-haiku-4-5    short, persona, cheap
  rm_agent          anthropic    claude-sonnet-4-6   tool use reliability
  devils_advocate   deepseek     deepseek-v4-pro     1M ctx + cheap (stub)
  massive_context   deepseek     deepseek-v4-pro     1M ctx (stub)

Future workloads add a row here, optionally a new model to pricing.py,
done. No router class needed at current single-MSBA scale (YAGNI).

Caller contract:
  - Always supply `agent_id` (one of engine.llm_cost_ledger
    ALLOWED_AGENT_IDS — typo-fails fast). Cost ledger entry written
    on every successful call.
  - tools (if any) must be in Anthropic schema for anthropic workloads
    (deepseek adapter will translate when wired).
  - The return text + tool_calls is the primary surface; raw_usage
    exposed for forensics.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Workload routing table — the single source of truth
# ──────────────────────────────────────────────────────────────────────────────
_WORKLOAD_ROUTING: dict[str, tuple[str, str]] = {
    # workload                 (provider,     model)
    "narrator":                ("anthropic",  "claude-haiku-4-5"),
    "rm_agent":                ("anthropic",  "claude-sonnet-4-6"),
    "anomaly_sentinel":        ("anthropic",  "claude-sonnet-4-6"),
    "attribution_analyst":     ("anthropic",  "claude-sonnet-4-6"),
    "audit_recorder":          ("anthropic",  "claude-sonnet-4-6"),
    "decay_sentinel":          ("anthropic",  "claude-sonnet-4-6"),
    "chief_of_staff":          ("anthropic",  "claude-sonnet-4-6"),
    "spec_drafter":            ("anthropic",  "claude-sonnet-4-6"),
    # R1 cost-route audit (2026-06-05): A/B variant of spec_drafter that
    # routes to deepseek. Used by scripts/audits/audit_cost_route_spec_drafter.py
    # to compare classification agreement vs anthropic before promoting.
    # NOT for production use until A/B passes — leave spec_drafter on
    # anthropic until the audit decides.
    "spec_drafter_deepseek":   ("deepseek",   "deepseek-v4-pro"),
    "devils_advocate":         ("deepseek",   "deepseek-v4-pro"),
    "massive_context":         ("deepseek",   "deepseek-v4-pro"),
    # Phase 1.5 (2026-06-05): per-paper triage for Employee A.
    # ~$0.001/paper × 30-50 papers/day = ~$0.05/day. Deepseek wins
    # this workload: simple binary judgment + 1-line reason, no complex
    # tool schema (where R1 audit showed Deepseek's tool_use compliance
    # fails 50%+ of the time). Plain JSON parse with deterministic
    # format is the safer fit at this scale.
    "papers_curator_filter":   ("deepseek",   "deepseek-v4-pro"),
    # Phase 1.5b (2026-06-05): richer 5-field summary per YES-filtered
    # paper. ~$0.01 per paper, eager-run on YES candidates only;
    # lazy-run for NO candidates when user requests via UI.
    "papers_curator_summary":  ("deepseek",   "deepseek-v4-pro"),
    # W4-piece-2 (2026-06-21): ClaimType Stage 0 router LLM fallback.
    # Deterministic keyword router catches ~17% of papers; the remaining
    # 83% UNKNOWN need LLM judgment. Routed to Deepseek-v4-pro: at the
    # time of build the Anthropic credit balance was depleted, and the
    # task fits the deepseek pattern (single-call, plain JSON output,
    # no tool_use schema — same shape as papers_curator_filter which
    # uses deepseek successfully). Reasoning model — needs max_tokens
    # ~500 for reasoning + 1-line JSON output. ~$0.0003/paper at scale.
    "papers_curator_claim_type_router": ("deepseek",   "deepseek-v4-pro"),
    # Phase 2.0 step 2 (2026-06-06): Employee A cross-source synthesis.
    # Reads multi-source state snapshot (recent papers + sleeves +
    # decay + verdicts + memory) and emits 0-3 synthesized hypothesis
    # candidates to hypotheses.jsonl with extraction_method=
    # LLM_SYNTHESIS. Sonnet 4.6 NOT Deepseek: synthesis is the highest-
    # quality-bar call in the system, and R1 audit (2026-06-05) showed
    # Deepseek tool_use compliance fails 50%+ on complex schemas.
    # Weekly cron + on-demand button. Cost ceiling $0.10/call.
    "papers_curator_synthesis": ("anthropic",  "claude-sonnet-4-6"),
    # Phase 2.0 step 11a (2026-06-06): Employee B strengthener review.
    # Reads ONE Hypothesis row (typically A-synthesized) plus context
    # (active doctrine snippets, recent verdicts in family) and emits
    # one of {APPROVE_FOR_PIPELINE / REJECT / DOCTRINE_AMENDMENT_NEEDED}.
    # APPROVE → surfaces in /approvals as candidate-pipeline approval.
    # AMENDMENT → surfaces in /approvals as memory_amendment approval.
    # Sonnet 4.6 — same quality bar as A's synthesis (B is the second
    # gate before any human review attention; want it sharp + tool-
    # compliant). Cost ceiling $0.05/hypothesis review.
    "strengthener_review":      ("anthropic",  "claude-sonnet-4-6"),
    # Phase 2.0 step 14b (2026-06-06): chief_of_staff weekly memo writer.
    # ONE call per weekly session that summarizes D/A/B activity into a
    # 5-bullet memo for the principal's 30-second scan. Sonnet 4.6 —
    # the principal will read these every week + use them as the
    # anchor for next week's substrate decisions; tone + concreteness
    # matter more than throughput. Cost ceiling $0.05/week → ~$2.60/yr.
    "chief_of_staff_memo":      ("anthropic",  "claude-sonnet-4-6"),
    # Stage B P3a (2026-06-07): Employee B active sleeve strengthen
    # proposer. Reads ONE deployed sleeve's full context (canonical
    # paper + mechanism + deployed config KPI + recent family RED
    # verdicts + decay alerts + doctrine) and emits 0-3 concrete
    # improvement-candidate Hypotheses targeted at that sleeve. Each
    # candidate names an improvement_kind from a controlled enum
    # (regime_filter / cost_aware_exec / position_weighting /
    # replacement_seek / risk_overlay / data_quality_patch) so the
    # output is testable not vague. Sonnet 4.6 — same quality bar as
    # A's synthesis; this is a generator call requiring multi-source
    # reasoning + Pattern-5-compliant strict JSON tool_use.
    # Cost ceiling $0.05/sleeve. 13 deployed sleeves × weekly ≈
    # $0.65/week, $34/yr.
    "strengthener_propose":     ("anthropic",  "claude-sonnet-4-6"),
    # Stage B procedural auto-spec dispatcher (2026-06-07): given a
    # procedural Hypothesis (predicted_direction=zero, mechanism_subtype
    # matches /proposal|pause|audit|fix/) + its test_methodology, ask
    # Sonnet for STRUCTURED dispatch JSON (one of a CONTROLLED enum of
    # dispatch_kinds + args). The LLM does NOT write code or pick
    # functions — only maps test_methodology to a registered kind +
    # extracts args. Dispatcher is deterministic Python. Pattern-5-
    # compliant: single call + strict schema. ~$0.02/hypothesis.
    "strengthener_spec_extract":("anthropic",  "claude-sonnet-4-6"),
    # Stage C Phase A (2026-06-07): tier classification for the
    # three-libraries doctrine. Single Sonnet call per BATCH of papers
    # (cost-efficient + consistency across papers — same model sees
    # all 57 at once and picks tiers from a controlled enum). Strict
    # JSON output. ~$0.10-0.30 per batch.
    "papers_tier_classifier":   ("anthropic",  "claude-sonnet-4-6"),
    # Stage C Phase B (2026-06-07): T2 anchor 1-line summary. Per
    # paper, ~$0.001. Single call per anchor (NOT batched — each
    # summary is paper-specific and short). Sonnet over Haiku for
    # tighter prose + better mechanism-class fit.
    "papers_anchor_summary":    ("anthropic",  "claude-sonnet-4-6"),
    # Tier C-1 (2026-06-08): factor backtest spec extraction. Given ONE
    # B-approved factor Hypothesis (predicted_direction != zero) + its
    # claim/methodology/required_data, asks Sonnet for STRUCTURED
    # backtest SPEC JSON: signal_kind (one of a CONTROLLED enum) +
    # universe + date_range + signal_inputs + rebal + weighting +
    # min_obs_months + pit_audits. LLM does NOT write Python — only
    # fills slots in a typed schema; dispatcher (C-2) maps SPEC to
    # pre-built engine.factor_lab template. If hypothesis doesn't fit
    # any signal_kind cleanly, LLM picks 'requires_custom_code' escape
    # hatch + human takes over. ~$0.03/hypothesis. Per
    # docs/spec_tier_c_factor_backtest_auto_dispatcher.md.
    "strengthener_factor_spec": ("anthropic",  "claude-sonnet-4-6"),
    # Tier C Phase 2.1 claim-shape router (Stage 0, 2026-06-13). Tight
    # classification of hypothesis claim into one of 11 canonical shapes
    # (CROSS_SECTIONAL_ALPHA / SPANNING / VRP / FACTOR_COMBINATION /
    # PORTFOLIO_OVERLAY / EVENT_DRIFT / TIME_SERIES_MOMENTUM / CARRY /
    # DECAY_STUDY / CAPACITY / FACTOR_STRUCTURE) + confidence + 1-line
    # rationale. ~$0.001/call. Cheap pre-filter that prevents Sonnet
    # drift the 8-way signal_kind enum was vulnerable to. Per
    # engine.agents.strengthener.claim_shape_router.
    "strengthener_claim_shape": ("anthropic",  "claude-sonnet-4-6"),
    # Tier C L3-2 Self-Doubt module (2026-06-08): post-dispatch
    # Sonnet call that scores system confidence in the verdict +
    # lists per-verdict caveats grounded in known silent bugs. Each
    # verdict gets a confidence (0-0.99 NEVER 1.0) + caveats list.
    # ~$0.04/dispatch. Anti-trust UX — DE Shaw philosophy "if your
    # system produces too many confident answers, the system has a
    # bug". Per docs/spec_tier_c_layer_2_3_roadmap.md L3-2.
    "strengthener_self_doubt":  ("anthropic",  "claude-sonnet-4-6"),
    # α Pre-Mortem Generator (2026-06-14). Skeptic persona enumerates
    # failure modes the strict gate MIGHT MISS, BEFORE dispatch. Strict
    # JSON tool_use, single call, Pattern-5-compliant. Inputs include
    # family belief + graveyard collisions + n_trials counter so the
    # skeptic doesn't repeat known concerns. ~$0.05/hyp.
    "pre_mortem":              ("anthropic",  "claude-sonnet-4-6"),
    # β Cross-Domain Transfer Generator (2026-06-14). Cross-asset
    # thinker persona produces 1-2 mechanism transfers per deployed
    # GREEN sleeve. Output routes to enhance pipeline (Frazzini-Pedersen
    # 70% rule). ~$0.30/sleeve, monthly cron.
    "cross_domain_transfer":   ("anthropic",  "claude-sonnet-4-6"),
    # γ Replication Checker (2026-06-14). Lit-aware specialist that
    # scans a hypothesis claim against known replication-failure
    # catalogs (Hou-Xue-Zhang 2020 q-factor failures, McLean-Pontiff
    # 2016 catalog of post-pub decay). Pattern-5-compliant single call
    # + strict schema. ~$0.05/hyp. Complements α (general adversarial)
    # by adding literature-replication evidence.
    "replication_checker":     ("anthropic",  "claude-sonnet-4-6"),
    # Brainstorm Layer 3 — experience-conditioned divergent generator
    # (2026-06-14). Single Sonnet call per seed pack, conditioned on
    # lesson_distiller output. Output: 3-5 BrainstormIdea (each with
    # mandatory falsifier per Popper, target asset class, mechanism,
    # data requirements). ~$0.20-0.25/session. See
    # [[project-brainstorm-architecture-2026-06-14]].
    "brainstorm_divergent":    ("anthropic",  "claude-sonnet-4-6"),
}


def _resolve_workload(workload: str) -> tuple[str, str]:
    if workload not in _WORKLOAD_ROUTING:
        raise ValueError(
            f"unknown workload {workload!r}; choose from "
            f"{sorted(_WORKLOAD_ROUTING)}"
        )
    return _WORKLOAD_ROUTING[workload]


# ──────────────────────────────────────────────────────────────────────────────
# Public result shape
# ──────────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class ToolCall:
    """A single tool invocation the model requested."""
    id:    str
    name:  str
    input: dict


@dataclasses.dataclass(frozen=True)
class LLMCallResult:
    """Provider-agnostic result of one call().

    Fields:
      text:        primary text output (may be "" if only tool_calls)
      tool_calls:  list of ToolCall objects (empty if not a tool-using turn)
      stop_reason: provider-specific stop reason string
      model:       actual model id used (may differ from requested if alias)
      provider:    "anthropic" / "deepseek"
      cost_usd:    cost of this single call (recorded to ledger as well)
      latency_ms:  wall-clock time
      cache_read_tokens: input tokens served from cache (anthropic only)
      raw_usage:   full usage dict from provider (forensics)
    """
    text:              str
    tool_calls:        tuple[ToolCall, ...]
    stop_reason:       str
    model:             str
    provider:          str
    cost_usd:          float
    latency_ms:        int
    cache_read_tokens: int
    raw_usage:         dict


# ──────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ──────────────────────────────────────────────────────────────────────────────
def call(
    *,
    workload:     str,
    system:       str,
    agent_id:     str,
    user:         Optional[str] = None,
    messages:     Optional[list[dict]] = None,
    tools:        Optional[list[dict]] = None,
    max_tokens:   int = 1024,
    cache_system: bool = True,
    thinking:     bool = False,
    effort:       str = "low",
    scope:        str = "",
    record_cost:  bool = True,
) -> LLMCallResult:
    """Call an LLM for `workload`, routed to the right provider+model.

    Args:
      workload:     one of _WORKLOAD_ROUTING keys (raises ValueError if unknown)
      system:       system prompt (cached if cache_system=True and large enough)
      user:         single user message (single-turn convenience; ignored if
                    `messages` supplied). Exactly one of user/messages required.
      messages:     full conversation history (multi-turn / tool-use agent loop)
      agent_id:     one of engine.llm_cost_ledger.ALLOWED_AGENT_IDS (typo-fails)
      tools:        tool definitions in Anthropic schema (auto-translated for
                    deepseek when that provider lands)
      max_tokens:   completion cap; stream when > ~16K (not used at MVP scale)
      cache_system: wrap system in cache_control for ~90% input cost saving
      thinking:     enable adaptive thinking (default OFF for cost)
      effort:       low | medium | high (Sonnet/Opus only; Haiku ignores)
      scope:        optional sub-scope tag for cost ledger (e.g. "phase=narrator")
      record_cost:  record to llm_cost_ledger (default True; set False for tests)

    Returns:
      LLMCallResult with text + tool_calls + cost + usage diagnostics.
    """
    if messages is None and user is None:
        raise ValueError("must supply either `user` (single-turn) or `messages`")

    provider, model = _resolve_workload(workload)

    # Data-egress / residency guard (institutional governance, blueprint spec id=78 §2/§6).
    # Classifies the outbound payload sensitivity and checks it against the provider's
    # residency policy (e.g. position data must not reach the CN provider). Default mode
    # 'warn' = log+audit only (non-breaking); AGENT_EGRESS_MODE=enforce raises. Never
    # blocks an ALLOWED call; a guard bug is swallowed so it cannot break a live call.
    try:
        from engine.agents.governance.data_egress import guard_egress
        _payload = (system or "")
        if user:
            _payload += " " + user
        if messages:
            _payload += " " + " ".join(str(m.get("content", "")) for m in messages)
        guard_egress(provider, _payload, workload=workload, scope=scope)
    except ImportError:
        pass
    except Exception:
        from engine.agents.governance.data_egress import EgressViolation
        # Re-raise only an explicit enforce-mode violation; swallow everything else.
        import sys as _sys
        if isinstance(_sys.exc_info()[1], EgressViolation):
            raise

    # Dispatch to provider adapter
    if provider == "anthropic":
        from engine.llm.providers.anthropic_provider import call_anthropic
        raw = call_anthropic(
            model        = model,
            system       = system,
            user         = user,
            messages     = messages,
            tools        = tools,
            max_tokens   = max_tokens,
            cache_system = cache_system,
            thinking     = thinking,
            effort       = effort,
        )
    elif provider == "deepseek":
        from engine.llm.providers.deepseek_provider import call_deepseek
        raw = call_deepseek(
            model        = model,
            system       = system,
            user         = user,
            messages     = messages,
            tools        = tools,
            max_tokens   = max_tokens,
            cache_system = cache_system,
            thinking     = thinking,
            effort       = effort,
        )
    else:
        raise ValueError(f"unknown provider {provider!r}")

    # Compute cost
    from engine.llm.pricing import compute_cost
    cost_usd = compute_cost(
        model              = model,           # use routing model, not raw.model
                                              # (raw.model may carry date suffix)
        input_tokens       = raw.input_tokens,
        output_tokens      = raw.output_tokens,
        cache_read_tokens  = raw.cache_read_tokens,
        cache_write_tokens = raw.cache_write_tokens,
    )

    # Record to unified cost ledger
    if record_cost:
        try:
            from engine.llm_cost_ledger import record_call
            record_call(
                agent_id          = agent_id,
                provider          = provider,
                model             = model,
                prompt_tokens     = raw.input_tokens + raw.cache_read_tokens
                                    + raw.cache_write_tokens,
                completion_tokens = raw.output_tokens,
                cost_usd          = cost_usd,
                latency_ms        = raw.latency_ms,
                scope             = scope or workload,
                extra             = {
                    "workload":             workload,
                    "cache_read_tokens":    raw.cache_read_tokens,
                    "cache_write_tokens":   raw.cache_write_tokens,
                    "stop_reason":          raw.stop_reason,
                    "model_actual":         raw.model,
                },
            )
        except Exception as exc:
            # Ledger failure must not break the call — log + continue
            logger.exception(
                "llm.call: cost_ledger.record_call failed (non-fatal): %s", exc,
            )

    return LLMCallResult(
        text              = raw.text,
        tool_calls        = tuple(
            ToolCall(id=tc["id"], name=tc["name"], input=tc["input"])
            for tc in raw.tool_calls
        ),
        stop_reason       = raw.stop_reason,
        model             = raw.model,
        provider          = provider,
        cost_usd          = cost_usd,
        latency_ms        = raw.latency_ms,
        cache_read_tokens = raw.cache_read_tokens,
        raw_usage         = raw.raw_usage,
    )
