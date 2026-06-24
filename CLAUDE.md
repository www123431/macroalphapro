# CLAUDE.md — project-root operating doctrine

Doctrine that applies across the whole repository. Subdirectory CLAUDE.md
files (e.g. `frontend/CLAUDE.md`) may add ADDITIONAL constraints; nothing
here is overridden by them.

---

## Research Event Emission Doctrine (2026-06-02, STANDING)

Every git commit that completes a research work block MUST emit at least
one event into the research store via `engine.research_store.emit.*`.
The store is the canonical record of what changed; UI surfaces, audit
tools, and downstream consumers read from `data/research_store/events.jsonl`.

**Why this exists.** Pre-doctrine, research outputs scattered across 5+
locations (`docs/capability_evidence/*.md`, `memory/*.md`, factory_ledger.jsonl,
gate_runs.jsonl, git commits). Every UI surface implemented its own scraping
logic → O(N×M) producer-consumer coupling. The event store collapses this
to O(N+M): producers emit, consumers query a typed surface. See
`engine/research_store/__init__.py` for the API.

### When to emit (the 8 canonical event types)

| Trigger | Helper | subject_type |
|---|---|---|
| Factor study completes with strict-gate verdict | `emit.factor_verdict(...)` | factor |
| Memory file written / amended | `emit.memory_locked(...)` | memory_doctrine |
| Spec document amended | `emit.spec_amended(...)` | spec |
| Active deploy config changed | `emit.deploy_changed(...)` | sleeve |
| Decay sentinel fires | `emit.decay_alert(...)` | sleeve |
| Data-quality breach surfaced | `emit.dq_breach(...)` | data_quality |
| Council critique completes | `emit.council_critique(...)` | factor |
| Capability evidence doc filed | `emit.capability_evidence_filed(...)` | factor |

### Pre-conditions (validated by `emit.*` — failures are loud, not silent)

1. **Artifact paths MUST exist on disk** before the emit call. The event
   references files as evidence; it does not create them. Write the
   capability evidence markdown / data parquet FIRST, then emit.

2. **`subject_id` MUST be in the registry**
   (`engine.research_store.registry`). If a new subject:
   ```python
   from engine.research_store import registry
   from engine.research_store.schema import SubjectType
   registry.register_subject(
       "my_new_subject",
       subject_type=SubjectType.factor,
       family="my_family",
       description="What this subject is",
   )
   ```
   No fuzzy auto-mapping. Typos are flagged with "did you mean" suggestions.

3. **`summary` MUST be 1-2 sentences** (≤ 400 chars). Detail belongs in
   the evidence doc, not the event.

4. **`parent_event_ids`** should be filled when this event depends on a
   prior one (e.g. a `capability_evidence_filed` event should reference
   the corresponding `factor_verdict_filed` event_id).

### Forbidden

- **Direct writes to `data/research_store/events.jsonl`**. Use `emit.*`.
- **Mutation of past events**. Events are immutable. To correct, emit a
  new event with `parent_event_ids` pointing to the prior; downstream
  consumers know to prefer the newer.
- **Bypassing the registry**. No "I'll register it later". If the subject
  isn't registered, register it FIRST, then emit.
- **Multi-event batches that share an event_id**. Each emit call is one
  event; UUIDs are auto-generated per call.

### One emit per commit (minimum)

If a commit changes research state without emitting, that state is
**invisible** to UI surfaces, audit tools, and your future self.
Pre-commit hook validation is planned (Phase 6); for now, this is on
the honor system, enforced by reviewer (Claude / user) judgment.

If a commit is pure infrastructure / docs / refactor (no research state
change), no emit is needed — but if in doubt, emit.

### Reading

Consumers use `engine.research_store.store.filter_events(...)` or the
forthcoming REST endpoints. Never scrape the jsonl directly; the typed
query API is the contract.

---

## Session Protocol Doctrine (2026-06-02, STANDING)

Every user-initiated Claude session must run under a **typed session
protocol**. Sessions are first-class workflow entities (same tier as a
git commit) with state machine, pre-flight checks, and exit conditions.

**Why this exists.** UI surfaces (Cockpit / Library / Graveyard / Decay)
answer "what is the system's state?" Claude answers "what should
change?" Without protocol, sessions drift — Claude duplicates UI work
in conversation; outputs leak into ad-hoc markdown nobody finds. With
protocol, every session has typed inputs (pre-flight digest), tracked
in-flight work (auto-tagged events), and verified outputs (exit check).

### The 5 session types

| type | when to use | duration |
|---|---|---|
| `research_new`   | test a new factor with strict gate | 2-6h |
| `audit`          | investigate bug / suspicious number / Layer 2 concern | 30min-3h |
| `ops`            | monitor / respond to alert (usually read-only) | 15min-1h |
| `doctrine`       | lock a lesson / amend memory file | 15-45min |
| `exploration`    | open-ended thinking; NO exit enforcement (escape hatch) | open-ended |

If user opens a session without specifying type, **ask** which one
applies — do not guess. Hybrid OK: infer + reflect back for
confirmation. Never assume.

### Session boundary — UI vs Claude division

Before Claude conversation starts, the user should have done these via UI:

- **research_new pre-flight**: Cockpit checked; `/research` graveyard
  searched for related verdicts; `/lab/library` checked for sleeve overlap.
- **audit pre-flight**: `/lab/library/detail` audit trail reviewed; Inbox
  alert reviewed.
- **ops pre-flight**: Cockpit / Risk / Book / Decay / Inbox scanned.
- **doctrine pre-flight**: memory index checked for similar prior
  doctrine.

Claude must NOT duplicate these in conversation. If user opens a session
without pre-flight context, ask them to do it on the UI first.

### Forbidden Claude behaviors

- Running factor pipelines outside an active research_new session
  (bypasses strict gate trust boundary)
- Emitting `factor_verdict_filed` without a parent `capability_evidence_filed`
  event (lineage must be intact)
- Closing a session without satisfying its exit conditions (use abandon
  if legitimately no artifact produced)
- Treating exploration session outputs as production verdicts (they
  are tagged `exploration` and excluded from strict-gate consumers)

### Active session integration

`engine.research_store.emit.*` auto-reads `data/sessions/_active.json`
and tags emitted events with `session:<id>` + `session_type:<type>`. No
explicit `session_id` parameter needed in normal usage.

### Exit conditions per type

| type | exit requirement |
|---|---|
| `research_new` | ≥1 `factor_verdict_filed` + ≥1 `capability_evidence_filed` (parent → verdict) |
| `audit` | ≥1 git commit OR ≥1 state-changing event |
| `ops` | none — always closes cleanly |
| `doctrine` | ≥1 `memory_doctrine_locked` |
| `exploration` | none — escape hatch |

Failed exit raises `ExitConditionsUnmetError`. Caller must either emit
the required artifacts or call `abandon_session()` with a stated reason.

### UI execution-button doctrine

Per 2026-06-02 audit: UI must NOT have free-form "execute" buttons that
bypass session protocol. All free-form execution flows through Claude
within a session. Allowed UI execute buttons:

- Data ops (fetch / refresh — deterministic data pulls)
- Cron control (pause / resume)
- Governance actions (approve / reject in /approvals)
- Session-gated pipeline runs (e.g. /research/candidate within an
  active research_new session)

Anything else is the wrong shape. /lab/factor-lab "Materialize" was
identified as anti-pattern and split into ideation-only browsing.


## Forward vs Enhance Statistical Separation Doctrine (2026-06-11, STANDING)

Two research-decision frameworks coexist in this codebase. They MUST
NEVER share verdict pipelines.

**FORWARD** — "Is X a real alpha?"
- Pipeline: `engine.agents.strengthener.factor_dispatcher`
- Statistics: FF5/HXZ spanning + Bailey-Lopez de Prado DSR family
  n_trials
- Verdict: GREEN / MARGINAL / RED
- Cron: daily 09:00 `scripts/burndown_run.py`

**ENHANCE** — "Does variant X' strictly improve deployed sleeve X?"
- Pipeline: `engine.research.enhance.dispatch_enhance_hypothesis`
- Statistics: **Politis-Romano 1994 paired block bootstrap** +
  Jobson-Korkie / Memmel Sharpe-diff
- Verdict: IMPROVEMENT / NOISE / DEGRADATION
- Cron: weekly Sunday 04:00 (Phase 3, deferred until LLM variant
  builder ships)

### Why separation is mandatory

Paired SE ≈ sqrt(2(1-ρ)/n); unpaired ≈ sqrt(1/n). At ρ≈0.95 (typical
sleeve↔variant), paired SE is **3.2x tighter**. Routing enhance through
forward thresholds kills 90%+ of real improvements. Routing forward
through enhance thresholds gives false IMPROVEMENT on uncorrelated new
strategies.

### Routing markers (any one → enhance class)

- `addresses_decay_in` field non-null
- `source:active_b_sleeve_scan` tag
- `source:doctrine_signal` tag
- `created_by` contains `sleeve_strengthen_scan` OR `sleeve_fix_proposer`

These markers MUST be checked in BOTH
`engine.research_store.hypothesis.classifier` AND
`engine.research.burndown_ranker`. Adding a new enhance producer =
updating both.

### Correlation gate (load-bearing)

|corr(baseline, variant)| < 0.50 → refuse with
`LOW_CORRELATION_NEW_FACTOR_ROUTE`. The hypothesis is NOT an enhance;
it's a new strategy and belongs in forward.

### Forbidden actions

- Routing an `addresses_decay_in` hypothesis through
  `factor_dispatcher` (Phase 1 filter prevents; new bypasses must add
  tests)
- Counting IMPROVEMENT verdicts in family n_trials for Bailey-LdP DSR
- Using belief-1 forward family prior to predict enhance outcomes
- Auto-deploying IMPROVEMENT — capital decisions stay HUMAN; routes to
  `/approvals` only

### Academic anchors

- López de Prado AFML Ch.2 — paired vs pooled statistics
- Politis-Romano 1994 — circular block bootstrap
- Jobson-Korkie 1981 / Memmel 2003 — Sharpe-diff t-stat
- Frazzini-Pedersen 2018 — institutional alpha = 70% enhance
- Bailey-LdP 2014 §3 — DSR applies to forward only

Full rationale + industrial precedent in
`memory/feedback_forward_vs_enhance_statistical_separation_2026-06-11.md`.

---

## (Add other project-wide doctrines below as they are locked.)
