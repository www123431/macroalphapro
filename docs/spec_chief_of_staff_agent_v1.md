# Spec — Chief of Staff Agent v1.0 LOCKED

**起草日期**: 2026-05-19
**spec_id**: TBD (call `register_spec("docs/spec_chief_of_staff_agent_v1.md")` after first lock)
**Project axis**: Persona Voice Layer β.2 unified-chat-UI mini-sprint (per [[project-persona-agent-architecture-2026-05-19]] deferred backlog "single-pane orchestration")
**Pre-registration**: retro=False, n_trials_contributed=0, factor_kind="ui_infrastructure"
**Status**: **v1.0 LOCKED — Phase 1 build begins next commit**

---

## 一、Purpose

The user is a solo PM. After six tool-using personas shipped (RM / DQ /
Anomaly Sentinel / Attribution Analyst / Audit Recorder / Devil's
Advocate), the workflow gap is "I have to open six separate chat tabs
and decide which one to ask each question". This violates the "boss
with employees" mental model — a real PM has ONE chief of staff who
routes to specialists in the background.

The Chief of Staff (CoS) is a single conversational interface that
delegates to the six existing specialist personas via a tool call,
collects structured results, and synthesizes a single human-facing
answer.

**This is NOT a group chat.** Specialists never see each other's
output, never talk to each other, never see the user directly. CoS is
the only agent the user converses with; specialists are tools.

This is the **Supervisor / Orchestrator-Worker pattern** (Anthropic
Agent SDK / LangGraph standard). BlackRock Aladdin Copilot,
Bridgewater PARC, Two Sigma Veneer all use this same pattern.

## 二、Scope

### 2.1 What CoS does

- Receive the user's question
- Decide whether to answer directly (cheap facts: spec count, agent
  list, today's date) or delegate to one or more specialists
- For delegation: call `delegate_to_specialist(agent_id, query)` — at
  most three delegations per user turn (hard cap)
- Synthesize a single user-facing answer that cites each specialist by
  name ("Risk Manager reports: ...")
- Persist its own chat history via the standard ChatSession table

### 2.2 What CoS does NOT do

- Does NOT make verdicts on its own. Substantive claims must come from
  a specialist tool result.
- Does NOT call specialists in parallel (sequential only — each
  specialist's output may inform the next routing decision).
- Does NOT let specialists delegate to other specialists (Pattern 5
  ban — flat one-hop only).
- Does NOT mutate state. Read-only authority.
- Does NOT write to project memory or amend specs.
- Does NOT see specialists' internal tool-call logs in chat history
  (only the specialists' final_text comes back; tool logs persist for
  audit but not for CoS's reasoning context).

## 三、Architecture

```
            ┌──────────────────────────┐
   User ◀───▶│   Chief of Staff (CoS)   │
            │   workload=rm_agent       │   ← Sonnet 4.6, tool-using
            │   tools = [delegate, ...] │
            └──────────┬───────────────┘
                       │ delegate_to_specialist(agent_id, query)
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Risk Manager   DQ Inspector   Anomaly Sentinel  ... (6 specialists)
   (isolated subprocess of chat_turn; empty history; max 4 iter)
```

### 3.1 Tool palette

CoS gets a small tool set: ONE delegation tool plus three lightweight
direct-answer tools for trivia.

| Tool | Purpose |
|---|---|
| `delegate_to_specialist(agent_id, query, max_iterations=4)` | Route to one specialist. Returns structured JSON with final_text + cost + iteration count. |
| `lookup_spec(spec_id)` | Direct fact lookup for "what is spec id X" trivia — avoids paying for a Risk Manager round-trip. |
| `read_project_memory(query)` | Direct doctrine lookup for "what did we decide about emojis" trivia. |
| `list_personas()` | Returns the six agent_ids + one-line scope for each. CoS uses this to confirm routing decisions. |

No other tools. CoS does NOT have direct access to alert tables, NAV
history, or audit findings — those flow through the appropriate
specialist.

### 3.2 Routing rules (the locked decision contract)

CoS's system prompt encodes these routing decisions. Compliance
verified by Phase 2 lockdown test.

| User question contains | Route to |
|---|---|
| "VaR" / "leverage" / "concentration" / "HHI" / "sleeve drift" / "HARD HALT" / "halt" / "circuit breaker" / "Mode N" | risk_manager |
| "FRED" / "yfinance" / "cache" / "stale" / "NaN burst" / "row count" / "data freshness" / "universe coverage" | dq_inspector |
| "z-score" / "spike" / "anomaly" / "drawdown" / "volume" / specific ticker forensic query | anomaly_sentinel |
| "NAV" / "return" / "attribution" / "sleeve weight" / "P&L" / "Modified Dietz" / "external flow" | attribution_analyst |
| "audit finding" / "amendment" / "spec history" / "amendment log" / "lineage" / "AuditRun" | audit_recorder |
| "critique" / "counterfactual" / "p-hacking" / "what's wrong with my reasoning" / "challenge this claim" | devils_advocate |
| Cheap trivia (spec id, agent list, current date) | Answer directly with `lookup_spec` / `read_project_memory` / `list_personas` |

**Conflict resolution:** Question matches multiple specialists — CoS
calls each sequentially in the order they appear in the table above,
caps total delegations at 3, and aggregates the results in its final
synthesis. Example: "Is the book safe to trade after the GLD spike?" →
risk_manager (book safety) + anomaly_sentinel (GLD spike forensic).

**No-match fallback:** If no specialist clearly matches, CoS asks a
clarifying question rather than guessing. The locked rule is "delegate
or clarify, never confabulate."

### 3.3 Delegation isolation contract

Every `delegate_to_specialist(agent_id, query)` call runs in a fully
isolated sub-context:

- `history=[]` (specialist never sees prior CoS conversation, never
  sees other specialists' answers, never sees user's other questions)
- `max_iterations=4` (hard cap to prevent runaway tool loops; CoS
  default is 6 but specialists summoned by CoS get less so the user
  turn stays bounded)
- Specialist's own internal tool calls are LOGGED to its
  `chat_tool_log` session_state but NOT exposed to CoS — CoS only sees
  the specialist's `final_text` + cost + iteration count

This is the structural enforcement of Pattern 5 ban: even if a
specialist's system prompt referenced another specialist by name
("ask the Risk Manager"), it has no mechanism to actually do so.

### 3.4 Cost + budget

- Each `delegate_to_specialist` call inherits the specialist's normal
  cost (Sonnet 4.6 for tool-using, Haiku for narrators, etc.)
- CoS itself uses Sonnet 4.6 (needs reliable routing judgment)
- Per-user-turn hard cap: $0.10. If accumulated cost across CoS turn
  + delegations exceeds, the next `delegate_to_specialist` call
  returns `{"error": "budget_exceeded_for_turn"}` and CoS must answer
  with what it has.
- Cost ledger writes one entry per call (CoS turn + each specialist
  delegation) with separate `agent_id` so the dashboard shows per-agent
  consumption cleanly.

### 3.5 Audit trail

Every CoS turn produces an audit record (written to existing
`engine.observability.record_tool_call` infrastructure):

```json
{
  "as_of":              "2026-05-19T14:32:15Z",
  "user_question_hash": "<sha256 first 16 chars>",
  "cos_decision":       "delegate" | "answer_direct" | "clarify",
  "delegations":        [
    {"agent_id": "risk_manager", "iterations": 2, "cost_usd": 0.0042},
    {"agent_id": "anomaly_sentinel", "iterations": 3, "cost_usd": 0.0067}
  ],
  "total_cost_usd":     0.0109,
  "final_text_hash":    "<sha256 first 16 chars>",
  "budget_hit":         false
}
```

The user-facing text and full conversation persist in ChatSession
under `agent_id="chief_of_staff"`.

## 四、Build phases

### Phase 1 — persona + delegate tool (this commit)
- Add `delegate_to_specialist` + `list_personas` to engine.agents.persona.tools
- Add `select_tools(...)` palette for CoS: 4 tools as listed in §3.1
- Create engine/agents/persona/chief_of_staff.py with locked system prompt
- Register `chief_of_staff` agent_id in ALLOWED_AGENT_IDS
- Register `chief_of_staff` workload route → anthropic + claude-sonnet-4-6

### Phase 2 — Streamlit + persistence
- pages/chat_with_chief_of_staff.py (default landing for AGENTS nav)
- App nav: AGENTS section reorders to put CoS first
- Existing session_store works as-is (CoS is just another agent_id)

### Phase 3 — tests
- All existing parametrized persona invariants run against CoS automatically
- Add `TestChiefOfStaffSpecific` with routing-rule lockdown tests:
  - Owns delegation tool + 3 trivia tools
  - Does NOT own alert / NAV / audit-finding tools (those are
    specialist-only — CoS must delegate, not access directly)
  - System prompt lists all six specialist agent_ids by name
  - System prompt encodes the §3.2 routing keywords
- Unit-test `delegate_to_specialist` with a mocked specialist

### Phase 4 — DEFERRED to Wave 3 (P2)
- Structured specialist response: each specialist returns trailing JSON
  block so CoS can reason over it programmatically (currently CoS just
  re-reads the prose). Improves routing accuracy + enables dedup.
- Daily morning digest using CoS as the orchestrator.

## 五、Open questions resolved at lock

**Q1: Should CoS see specialists' tool-call logs?**
RESOLVED — No. Specialist tool logs persist in their own session_state
for audit. CoS only sees `final_text + cost + iterations`. Rationale:
exposing tool logs to CoS doubles token cost per turn and CoS doesn't
need them to synthesize; if user wants the tool log they open the
specialist's own chat page.

**Q2: Parallel delegation?**
RESOLVED — No, sequential only. Parallel calls would prevent CoS from
adapting (e.g. RM says "GLD anomaly suspected" → CoS routes Anomaly
Sentinel to check GLD specifically). Sequential cost overhead is
bounded by the 3-delegation cap.

**Q3: Can CoS write to project memory based on its synthesis?**
RESOLVED — No. Tier 3 memory is human-curated only (per
[[feedback-spec-amendment-workflow-and-hash-discipline-2026-05-19]]
doctrine extension: no agent writes to source-of-truth files). This
applies to CoS the same as every specialist.

**Q4: Group chat UI later — do we need a different architecture?**
RESOLVED — No, CoS architecture extends naturally. "Group chat" would
just be "render each specialist's individual chat alongside the CoS
chat in a multi-column layout". Pattern 5 ban still holds.

**Q5: What happens if a specialist call raises mid-turn?**
RESOLVED — Caught by `delegate_to_specialist`, returned as `{"error":
"specialist failed: <msg>"}` with `is_error=True`. CoS recovers in
its next iteration. User sees a degraded answer like "Risk Manager
unreachable; answering with what I know from Audit Recorder + memory."

## 六、Governance

- **NEVER soften halt verdicts**: if a specialist returns a HARD_HALT
  finding, CoS quotes it verbatim. No "the Risk Manager mentions
  there's a potential concern" softening.
- **NEVER fabricate specialist output**: every claim that names a
  specialist must trace back to an actual `delegate_to_specialist`
  call in this turn. Banned phrases: "the Risk Manager would say X
  but I haven't checked", "based on what I know about the DQ
  Inspector".
- **READ-ONLY enforced at multiple layers**: CoS prompt + tool palette
  (no write tools exposed) + delegation tool can only call read-only
  specialists.
- **Single-pane Pattern 5 enforcement**: there is no `agent_id`
  parameter the user can pass that lets one specialist invoke another.
  The delegation tool is exposed only to CoS.

## 七、Versioning + amendment

This spec is `v1.0`. Future amendments via the standard
`engine.preregistration.amend_spec(path, kind, reason)` workflow.
Per [[feedback-spec-amendment-workflow-and-hash-discipline-2026-05-19]]:
- never pin a literal hash inside this file (use `lookup_spec` at
  runtime to resolve current hash)
- always call `amend_spec` after any edit
- pre-commit hook (.githooks/pre-commit) blocks drift

Anchor for source code: `engine/agents/persona/chief_of_staff.py`
system_prompt MUST reference §3.2 routing rules and §3.3 isolation
contract verbatim.

### Amendment 2026-05-22 — 7th specialist: decay_sentinel
The constellation grew from six to SEVEN specialists: `decay_sentinel` (the book-health /
mechanism-decay agent — engine.agents.persona.decay_sentinel) is now a delegation target.
Wherever this spec says "six specialists", read SEVEN. Added to the §3.2 routing table:

| User question contains | Route to |
|---|---|
| "decaying" / "decay" / "still paying" / "is X still working" / "diversification" / "re-allocation" / "signal-IC" / "mechanism health" | decay_sentinel |

decay_sentinel owns mechanism/strategy decay + book diversification integrity + disciplined
re-allocation (rolling Sharpe / signal-IC, role-aware alpha/insurance/trend/regime_premium,
downside/stress correlation); it reads the deterministic book-health report and explains it
(0-LLM-in-DECISION). Wired into delegate_to_specialist + list_personas + the CoS prompt
(team table + routing table). Pattern 5 ban + isolation contract unchanged.
