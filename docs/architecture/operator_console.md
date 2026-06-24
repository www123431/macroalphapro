# Operator Console — Design Doc

**Status**: design lock (2026-06-23). Build follows after sign-off.
**Author**: Zhang Xizhe + Claude Opus 4.7 (session pair-design).
**Audience**: implementation reference + arxiv supplement Section 7
("System UX design").

---

## 0. Executive Summary

The project has matured to the stage where the UI must transition
from a monitoring surface (with the principal triggering pipelines
via Claude conversation or scripts) to an operator console where
external users — open-source contributors, second-quant operators,
future BYO-data users — drive the full research pipeline end-to-end
through clicks alone.

**Target audience**: Group B users (fork-and-use), not Group A
(read-and-evaluate). Group A is served by today's Demo-mode pages
(`/research/workflow`, `/research/calibration`) plus the arxiv
preprint. PPT works for Group A. UI does not — UI is for Group B.

**Construction philosophy**:

1. **Foundation-first**: before any of the 8 Pipeline Stations is
   built, the foundation infrastructure (async job queue, SSE
   progress stream, cost ledger, data-dependency tagging, multi-user
   schema) must be locked. Skipping foundation means 8 bespoke
   stations and unmaintainable code.

2. **Session-gated execution**: every pipeline trigger flows through
   an active typed session per CLAUDE.md Session Protocol Doctrine.
   The UI does not get free-form execute buttons. SessionLauncher
   (currently 0 visits in 11 days of telemetry — broken UX) gets
   rebuilt first.

3. **Doctrine consistency**: each station's design must hold the
   load-bearing project doctrines (FORWARD vs ENHANCE statistical
   separation, capital decisions stay human, Pattern 5 ban, etc.).
   UI cannot be where doctrine is silently violated.

**Cost honesty**: this is a 2-3 week full-time build (or 5-6 week
part-time). Estimated 85-105 hours total. Minimum viable E2E demo
(S1 + S4 + S6) is ~38-50 hours after Foundation + SessionLauncher.

---

## 1. Goals + Non-goals

### Goals (must hold)

- A Group B user can fork the repo, install dependencies, launch the
  app, start a typed session, and trigger an end-to-end research
  pipeline (paper → hypothesis → dispatch → verdict → autopsy →
  belief update) with no Claude conversation required.

- Every station emits typed events to the existing research store
  AND new operator console event store (audit + lineage).

- BYO data path: users without a WRDS subscription can run the
  pipeline on bundled demo fixtures + see the architecture work,
  without lying that they're running on real institutional data.

- Per-session cost cap: an LLM-heavy session cannot accidentally
  burn through API budget. Cost preview before each trigger.

- Multi-user-ready schema: even though deploy is single-user, event
  schema carries `actor_id` so future multi-user retrofit is cheap
  (data migration is exponentially more expensive than schema
  preparation).

### Non-goals (defer or out-of-scope)

- **Mobile-responsive**: desktop-first build. Static export complicates
  responsive design; defer to Phase 4 polish.
- **Multi-tenant SaaS**: each user runs their own fork. No central
  hosting infrastructure.
- **Real-money capital deployment trigger**: PROMOTE GREEN to
  paper-trade is OK from UI; real-money flag stays disabled by
  doctrine.
- **Demo Mode "Take the Tour"**: deferred per user decision —
  Group A audience uses PPT, not UI tour.
- **Automated rollback on halt**: rollback is a Human-gated S8
  station, never auto.

---

## 2. The 5 Architectural Locks

These 5 decisions must be made up-front. Each wrong choice produces
multi-day re-work later.

### D1 — Async execution model

**Decision**: extend the existing `engine/agents/workflow_executor/`
agent into a UI-facing job system. Do NOT build a parallel system.

**Rationale**:
- Workflow executor already has autorun pattern, failure_streak
  tracking, pause/resume.
- Dispatch runs take 1-3 minutes (Bootstrap CI B=10000, spanning
  regression, DSR multi-test). UI cannot block on these.
- FastAPI `BackgroundTasks` cannot persist across server restart —
  unsuitable for jobs queued by users who close the tab.

**Implementation**:
- Wrap each station's backend call as a `WorkflowJob` row in
  `data/operator_console/jobs.jsonl` (new typed store).
- Job states: `queued / running / completed / failed / cancelled`.
- API: `POST /api/console/trigger` enqueues; worker picks up via
  existing executor poll loop.
- UI polls `/api/console/status/{job_id}` and subscribes to SSE.

### D2 — Progress streaming

**Decision**: Server-Sent Events (SSE) via `sse-starlette`. Not
WebSocket. Not long-polling.

**Rationale**:
- Frontend is `output: 'export'` static (next.config.ts).
- Backend FastAPI serves both static frontend and API.
- SSE is one-direction (server → client), exactly what progress
  streams need.
- `sse-starlette` is the FastAPI-idiomatic library.
- WebSocket adds bidirectional state machine complexity not needed
  here.
- Long-polling is chatty and creates worse UX on reload.

**Stream protocol** (per station):
```
event: stage_started
data: {"stage": "bootstrap_ci", "started_ts": "...", "expected_seconds": 30}

event: stage_progress
data: {"stage": "bootstrap_ci", "pct": 47, "current": "resample 4700/10000"}

event: stage_completed
data: {"stage": "bootstrap_ci", "result": {"ci_lo": 0.31, "ci_hi": 0.42}}

event: job_completed
data: {"job_id": "...", "verdict": "GREEN", "result_url": "/research/lessons/..."}

event: job_failed
data: {"job_id": "...", "stage_failed": "spanning_regression", "error": "..."}
```

### D3 — BYO data boundary

**Decision**: 3-tier data dependency model. Every station declares
its tier; UI surfaces the tier label clearly so users know what
will work without their own WRDS subscription.

**Tiers**:
- **`user_data`** — user-supplied input (paper PDF, hypothesis text,
  factor spec). Works on any install.
- **`demo_fixture`** — bundled sample data (5 pre-computed mechanism
  spanning regression inputs, 3 sample papers, 2 sample autopsies).
  Works on any install. Marked in UI with "Demo Fixture" badge.
- **`snapshot_data`** — read-only view of existing research store
  (299 verdicts, 101 autopsies, 12 belief families). Works on any
  install but represents principal's run, not the user's.
- **`wrds_required`** — needs full WRDS subscription. Marked
  prominently. Pre-flight check refuses to trigger without WRDS env.

**Fixture authoring**:
- `data/_samples/papers/` — 3 hand-curated arxiv papers
- `data/_samples/spanning_inputs/` — 5 pre-computed regression panels
  for canonical mechanisms (BAB, PEAD, MOM, HML, GP/A)
- `data/_samples/autopsies/` — 2 sample autopsy rows (1 GREEN, 1 RED)

### D4 — Cost gating

**Decision**: per-session cost ledger with hard cap, preview-before-
trigger, anonymous-mode restrictions.

**Mechanism**:
- Session creation: user sets cost cap (default $1.00, max $5.00).
- Each station declares estimated LLM cost in its spec.
- Pre-flight rejects trigger if `current_session_spend + estimated >
  cap`.
- Anonymous user (no API key in `.streamlit/secrets.toml`): can
  trigger deterministic stations (statistical rigor, view-only) but
  not LLM-heavy stations (Synthesize, Extract).

**Storage**:
- `data/operator_console/session_cost_ledger.jsonl` — append-only,
  keyed by session_id.
- Existing `data/llm_cost_ledger.jsonl` continues to track absolute
  spend; new ledger tracks per-session.

**UI surface**:
- `CostCapBanner` component: sticky above page content when in
  active session. Shows `$0.34 spent / $1.00 cap`.
- Pre-flight panel: "This action will spend approximately $0.08.
  Proceed?"

### D5 — Multi-user-ready schema

**Decision**: every event, every session, every job carries
`actor_id`. Deploy is single-user (one principal); schema is
multi-user-ready.

**Rationale**:
- Adding `actor_id` to schema now: free.
- Adding `actor_id` to schema later: requires migrating thousands
  of events, breaks all downstream consumers, ~10-20h work.
- López de Prado AFML Ch.2: "schemas are forever; values are
  transient. Pay schema cost now."

**Implementation**:
- All new emit functions take `actor_id` parameter (default "principal"
  when not set, so existing callers continue to work).
- All API endpoints accept `X-Actor-Id` header (default to
  "principal").
- Display in UI: minimal (a small avatar circle), but schema is
  ready when needed.

---

## 3. Foundation Infrastructure

Module layout:

```
engine/operator_console/
  __init__.py
  pipeline_station.py     # PipelineStation abstract base class
  job_queue.py            # WorkflowJob model + queue + worker hooks
  job_store.py            # data/operator_console/jobs.jsonl
  cost_ledger.py          # per-session cost tracking (D4)
  data_dependency.py      # DataTier enum + fixture loaders (D3)
  sse_emitter.py          # standard SSE event protocol (D2)
  emit.py                 # typed events (mirror research_store.emit)
  schema.py               # JobState, DataTier, StationResult, etc.

api/routes_operator_console.py    # router; mounted in main.py
  POST   /api/console/trigger
  GET    /api/console/status/{job_id}
  GET    /api/console/stream/{job_id}    (SSE)
  POST   /api/console/cancel/{job_id}
  GET    /api/console/cost_status?session_id=X
  GET    /api/console/stations           (registry / discoverability)

frontend/components/operator_console/
  PipelineStation.tsx           # universal 5-element wrapper
  StationPreflight.tsx          # red/green light system
  StationConfigPanel.tsx        # generic JSON-schema-driven form
  StationProgressStream.tsx     # SSE consumer + stage progress UI
  StationResultPanel.tsx        # verdict + lineage links
  CostCapBanner.tsx             # session cost display
  ActiveSessionGuard.tsx        # wraps station, redirects if no session
  DataTierBadge.tsx             # user_data / demo / snapshot / wrds badges

frontend/app/(terminal)/research/console/
  page.tsx                      # station registry / launchpad
  [station]/page.tsx            # dynamic per-station page
```

Foundation effort: **10-15h** (build + test).

---

## 4. PipelineStation Abstract Pattern

Every station is a 5-element pattern. Backend and frontend each
implement the corresponding side of the contract.

### Backend interface

```python
class PipelineStation(ABC):
    """Base class for all stations. Each station implements the 5
    abstract methods + a static `STATION_SPEC` declaration."""

    STATION_SPEC: ClassVar[StationSpec]   # static metadata

    @abstractmethod
    def preflight(self, session: Session, config: dict) -> PreflightResult:
        """Run pre-conditions: data deps, cost cap, session validity.
        Returns red/yellow/green per check + overall blocker status."""

    @abstractmethod
    def estimate_cost(self, config: dict) -> CostEstimate:
        """Predict LLM/compute cost before user clicks trigger."""

    @abstractmethod
    async def execute(
        self, session: Session, config: dict,
        emitter: SSEEmitter, cancellation: CancellationToken,
    ) -> StationResult:
        """The real work. Stream progress via emitter. Respect cancel."""

    @abstractmethod
    def result_lineage(self, result: StationResult) -> list[str]:
        """Return list of next-station IDs this result enables.
        E.g. S1 → enables S2 (Synthesize) with the new paper_id."""

    @classmethod
    def render_config_form(cls) -> dict:
        """JSON Schema for the Config form. Frontend renders generic
        form from this; no per-station React code needed."""
```

### Standard StationSpec

```python
@dataclass(frozen=True)
class StationSpec:
    station_id:        str    # "S1_paper_ingest"
    title:             str    # "Paper Ingest"
    description:       str    # 1-line summary
    data_tier:         DataTier   # user_data / demo / snapshot / wrds
    requires_session_types: set[SessionType]   # which sessions can run this
    estimated_minutes: int        # expected wall-clock
    estimated_cost_usd: float     # expected LLM cost (worst case)
    icon:              str    # lucide-react icon name
```

### Standard StationResult

```python
@dataclass(frozen=True)
class StationResult:
    job_id:           str
    station_id:       str
    session_id:       str
    actor_id:         str           # D5
    started_ts:       str
    completed_ts:     str
    success:          bool
    artifacts:        dict[str, str]  # named artifact paths (markdown, json, parquet)
    events_emitted:   list[str]      # event_ids written to event store
    next_stations:    list[str]      # what user can do next
    cost_actual_usd:  float
```

### Frontend pattern

A single `<PipelineStation stationId="S1_paper_ingest" />` component
fetches the StationSpec, renders all 5 elements, handles SSE, and
chains to next-station UI. **One implementation; eight stations.**

---

## 5. The 8 Stations (Specs)

### S1 — Paper Ingest

| Field | Value |
|---|---|
| station_id | `S1_paper_ingest` |
| data_tier | `user_data` |
| requires_session_types | `{research_new, exploration}` |
| estimated_minutes | 1 |
| estimated_cost_usd | 0.001 (Haiku claim_type) |

**Pre-flight**:
- Session active and type valid
- Cost cap allows trigger
- ClaimType router model available (deterministic check)

**Config**:
- Upload PDF (max 10MB) OR paste arxiv URL OR paste DOI

**Trigger semantics**:
- Backend extracts text via PyMuPDF
- Routes through Stage-0 ClaimType (`engine/agents/papers_curator/
  claim_type_router.py`)
- Generates 1-paragraph summary (Haiku)
- Writes `papers_registry.jsonl` + emits `paper_registered`

**Progress stream stages**:
1. `text_extraction` (~5s)
2. `claim_type_classification` (~3s)
3. `summary_generation` (~10s, Haiku call)
4. `registry_write` (~1s)

**Result**:
- paper_id, claim_type, summary, source URL
- Lineage out: `S2_hypothesis_synthesize` (papers cluster), or
  direct view in `/research/papers/<id>`

---

### S2 — Hypothesis Synthesize

| Field | Value |
|---|---|
| station_id | `S2_hypothesis_synthesize` |
| data_tier | `user_data` (with snapshot_data for belief context) |
| requires_session_types | `{research_new}` |
| estimated_minutes | 3-5 |
| estimated_cost_usd | 0.10-0.20 (Sonnet cross-source synthesis, ~1500 token output) |

**Pre-flight**:
- ≥ 1 paper of `claim_type=FACTOR_HYPOTHESIS` in session scope
- Cost cap allows $0.20 (worst-case)

**Config**:
- Pick papers to include (multi-select from registry)
- Mechanism family hint (optional)
- Synthesis depth: `quick / standard / thorough`

**Trigger semantics**:
- Backend pulls papers + per-family belief context (from
  `belief_synthesis_context.py`)
- Calls Sonnet with cross-source synthesis prompt
- Citation-verifier sub-agent (already in `engine/agents/papers_curator/
  ssrn_crossref_crawler.py`) flags hallucinated refs
- Writes new hypothesis row to `hypotheses.jsonl`
- Emits `hypothesis_drafted`

**Progress stream stages**:
1. `belief_context_load` (~2s)
2. `synthesis_llm_call` (~120s, Sonnet, streamed token-by-token to UI)
3. `citation_verification` (~10s)
4. `hypothesis_persist` (~1s)

**Result**:
- hypothesis_id, claim, mechanism_family, predicted_direction
- Lineage out: `S3_factorspec_extract`

---

### S3 — FactorSpec Extract

| Field | Value |
|---|---|
| station_id | `S3_factorspec_extract` |
| data_tier | `user_data` |
| requires_session_types | `{research_new}` |
| estimated_minutes | 2 |
| estimated_cost_usd | 0.03 |

**Pre-flight**:
- Hypothesis exists with claim_type=FACTOR_HYPOTHESIS
- Mechanism family resolvable (alias map check)

**Config**:
- Hypothesis to extract from
- Override default template (optional)

**Trigger semantics**:
- Calls `engine/agents/strengthener/factor_spec_extractor.py`
- Output: hash-locked FactorSpec
- Emits `spec_amended` (or `spec_created`)

**Result**:
- spec_id, hash, FASTEXPR equivalent, template binding
- Lineage out: `S4_forward_dispatch` OR `S5_enhance_dispatch`
  (the burndown_ranker classifies which)

---

### S4 — FORWARD Dispatch

| Field | Value |
|---|---|
| station_id | `S4_forward_dispatch` |
| data_tier | `wrds_required` (or `demo_fixture` for demo path) |
| requires_session_types | `{research_new, audit}` |
| estimated_minutes | 5-10 |
| estimated_cost_usd | 0.10 |

**Pre-flight**:
- Spec exists and is hash-locked
- WRDS env available (skip if demo_fixture flag set)
- Cost cap allows $0.10
- DSR n_trials counter for the strategy_family available

**Config**:
- spec_id
- Data window (default: 2010-2024 for fixture, 2000-2024 for WRDS)
- Override anchor papers (optional)

**Trigger semantics**:
- Calls `engine/agents/strengthener/factor_dispatcher.py`
- 8 statistical rigor stages run in sequence
- Belief Layer pre-records prediction (air-gap discipline)
- Emits `factor_verdict_filed` + `capability_evidence_filed`

**Progress stream stages**:
1. `belief_predict_commit` (~1s)
2. `data_load` (~10s)
3. `spanning_regression` (~10s)
4. `bootstrap_ci` (~30s)
5. `nw_t_hac` (~5s)
6. `dsr_correction` (~3s)
7. `hosmer_lemeshow_calibration` (~5s)
8. `verdict_synthesis` (~5s)
9. `autopsy_join` (~2s)

**Result**:
- verdict (GREEN/MARGINAL/RED), CIs, all 8 test results, autopsy_id
- Lineage out: `S6_verdict_view` or `S7_promote` (if GREEN)

---

### S5 — ENHANCE Dispatch

| Field | Value |
|---|---|
| station_id | `S5_enhance_dispatch` |
| data_tier | `wrds_required` (or `demo_fixture`) |
| requires_session_types | `{research_new, audit}` |
| estimated_minutes | 5-10 |
| estimated_cost_usd | 0.08 |

**Pre-flight**:
- Spec exists
- Existing deployed sleeve specified
- Correlation > 0.50 (else reject with `LOW_CORRELATION_NEW_FACTOR_ROUTE`)
- Pipeline ROUTE_CLASS = `ENHANCE` (from classifier)

**Config**:
- New variant spec_id
- Existing sleeve to compare against
- Block bootstrap parameters (B=2000 default)

**Trigger semantics**:
- Calls `engine/research/enhance/dispatch_enhance_hypothesis()`
- Politis-Romano paired block bootstrap
- Jobson-Korkie / Memmel Sharpe-diff
- Emits `enhance_verdict_filed`

**Result**:
- verdict (IMPROVEMENT/NOISE/DEGRADATION), paired CI, magnitude
- Lineage out: `/approvals` (if IMPROVEMENT — human gate for swap)

---

### S6 — Verdict View + Autopsy + Belief Update

| Field | Value |
|---|---|
| station_id | `S6_verdict_view` |
| data_tier | `snapshot_data` (read-only) |
| requires_session_types | any |
| estimated_minutes | 1 |
| estimated_cost_usd | 0.0 |

**Pre-flight**:
- verdict_event_id exists

**Config**:
- verdict_event_id

**Trigger semantics**:
- Read-only render
- Show: prediction → verdict → autopsy → belief delta
- Live animation when called immediately after S4/S5 completion

**Result**:
- HTML render of full lineage
- Links to `/research/calibration` (updated Brier)
- Lineage out: `S7_promote` if GREEN, else terminate

---

### S7 — PROMOTE 9-gate

| Field | Value |
|---|---|
| station_id | `S7_promote` |
| data_tier | `snapshot_data` + `wrds_required` for cost/capacity checks |
| requires_session_types | `{research_new}` |
| estimated_minutes | 15-30 |
| estimated_cost_usd | 0.20 |

**Pre-flight**:
- verdict = GREEN
- 9 prior checks listed below

**Config**:
- verdict_event_id
- Target weight in book
- Role classification (alpha / insurance / regime_premium / trend)

**Trigger semantics — 9 sequential gates** (any FAIL halts):

1. FORWARD verdict GREEN (already verified)
2. Cost-robust (Almgren-Chriss optimal execution gap acceptable)
3. PIT clean (look-ahead audit passes)
4. Replication (γ persona confirms paper replication)
5. Multi-period (Mann-Kendall stability across 5 sub-periods)
6. Anchor-residual (post-FF5+MOM residual sharpe > threshold)
7. Cross-sleeve correlation (with each existing deployed sleeve)
8. Capacity (Pastor-Stambaugh / Berk-Green capacity ceiling)
9. **HUMAN approval** — fires UI modal "Approve deployment?",
   routes to `/approvals`

**Result**:
- promote_decision (APPROVED / REJECTED with reason)
- If APPROVED: emits `deploy_changed` and updates active book
- Lineage out: `/book` (book updated)

---

### S8b — Doctrine Lock (micro-station for `doctrine` session type)

| Field | Value |
|---|---|
| station_id | `S8b_doctrine_lock` |
| data_tier | `user_data` |
| requires_session_types | `{doctrine}` |
| estimated_minutes | 5 |
| estimated_cost_usd | 0.0 (deterministic) |

**Why this exists as its own station**: the 8 main stations all map
to `research_new` / `audit` / `ops`. The `doctrine` session type
exists per CLAUDE.md but had no UI surface in the initial draft —
users opening a doctrine session would have no station to use. S8b
fills this gap. It is intentionally simple (form-only, no async job)
because doctrine locking is a deterministic write, not a pipeline run.

**Pre-flight**:
- Session active and type = `doctrine`
- Memory directory writable

**Config**:
- Doctrine title (kebab-case slug)
- Doctrine body (markdown text area)
- Type: feedback / project / user / reference
- Related memories (multi-select from existing index, becomes
  `[[link]]` references)
- "Why" line (mandatory: the reason / past incident)
- "How to apply" line (mandatory: when this kicks in)

**Trigger semantics**:
- Writes new file to memory directory (with frontmatter)
- Appends index line to MEMORY.md
- Emits `memory_doctrine_locked` event with parent_event_ids if
  amending an existing doctrine

**Result**:
- doctrine_id, memory file path
- Lineage out: session can be closed (exit condition met)

---

### S8 — Roll back

| Field | Value |
|---|---|
| station_id | `S8_rollback` |
| data_tier | `snapshot_data` |
| requires_session_types | `{audit, ops}` |
| estimated_minutes | 5 |
| estimated_cost_usd | 0.0 |

**Pre-flight**:
- Deployed sleeve exists with deploy_event_id
- Last 7 days of paper-trade NAV available for impact preview

**Config**:
- sleeve to roll back
- Rollback target (revert N days / revert to deploy_event_id / full remove)
- Rationale (required free-text field)

**Trigger semantics**:
- Calls deployment rollback in `engine/portfolio/deployment.py`
- Emits `deploy_changed` with rollback semantics
- Generates rollback evidence markdown automatically

**Result**:
- Rollback complete, new active book state
- Lineage out: `/book` (book restored)

---

## 6. SessionLauncher Redesign

### Current state audit

- Route: `/research/sessions`
- Telemetry: **0 visits in 11 days** (principal never uses UI to
  launch sessions — uses Claude conversation instead)
- This is the load-bearing UX failure; before Operator Console can
  launch stations, the session entry must work.

### Failure mode hypotheses

1. **Entry is invisible**: principal hits `/lab/today` 48 times
   via muscle memory; sessions hidden in sidebar item 2.
2. **5-type friction**: choosing among `research_new / audit / ops /
   doctrine / exploration` blocks first action; needs examples.
3. **Exit conditions opaque**: principal afraid to start a session
   they can't close cleanly.
4. **Abandon is shameful**: no friendly UX for "this session didn't
   produce anything; close with reason".

### New design

**Entry**:
- Dashboard top: prominent "Start Session" CTA, badge with type count.
- Cmd-K palette: typing "start" autocompletes session types.
- Onboarding flow: first-time user sees session-type explanation
  with 1-paragraph example each.

**Type selection** (modal):
```
┌─────────────────────────────────────────────────────────┐
│  Start a session                                        │
├─────────────────────────────────────────────────────────┤
│  What are you doing today?                              │
│                                                         │
│  [Research a new factor]  research_new   2-6 hours      │
│   Like: "test BAB enhanced with quality filter"         │
│                                                         │
│  [Investigate something]  audit          30 min - 3h    │
│   Like: "look into yesterday's circuit breaker halt"    │
│                                                         │
│  [Monitor / respond]      ops            15 min - 1h    │
│   Like: "morning book check + decay alert review"       │
│                                                         │
│  [Lock a lesson]          doctrine       15-45 min      │
│   Like: "add the new fake-diversity rule to memory"     │
│                                                         │
│  [Just thinking]          exploration    open           │
│   Like: "what if we tried cross-asset momentum?"        │
└─────────────────────────────────────────────────────────┘
```

**Active session indicator**:
- Sticky banner on every page when active session: session type,
  elapsed time, exit progress (e.g. "research_new: 1/2 exit
  conditions met")
- Click to expand session detail, see what's left
- "Abandon (with reason)" prominent — no shame

**Exit flow**:
- When all exit conditions met → "Close session" CTA
- Session close generates auto-narration (LLM summary of session
  events, written to `data/sessions/<id>.md`)
- Cost summary surfaced

---

## 7. UX Flows

### First-time Group B user (idealized)

```
1. git clone, pip install, npm install, npm run build, python -m uvicorn api.main:app
2. Open http://localhost:8000/
3. Landing → "Enter terminal"
4. Dashboard sees "No active session — start one to use the console"
5. Clicks "Start Session"
6. Picks research_new with example "test BAB enhanced with quality"
7. Sets cost cap $1.00 (default)
8. Lands on console launchpad: 8 station cards
9. Cards show data_tier badges: 3 are "Demo Fixture", 1 is
   "WRDS required" (S4 demo path available)
10. Clicks S1 Paper Ingest → uploads sample PDF
11. Sees ClaimType tagged in 18s, paper registered
12. Clicks "Synthesize hypothesis" (S2) from result panel
13. ~3 min Sonnet streams synthesis to UI; cost $0.05
14. Clicks "Extract FactorSpec" (S3); ~2 min
15. Clicks "Dispatch FORWARD (demo)" (S4 demo fixture path)
16. Watches 8 statistical stages live; ~5 min
17. Gets RED verdict; routes to S6 view
18. Sees Brier update in real time
19. Closes session; auto-narration written
```

### Returning user (principal)

```
1. Opens dashboard, sees ongoing session if any
2. If new task: clicks "Start Session" → audit
3. Routed to relevant station based on context (e.g. clicking on
   a decay alert → suggests audit session targeting that sleeve)
4. Performs investigation via stations
5. Locks lesson via doctrine session if needed
6. Closes
```

---

## 8. Academic Anchors

| Concept | Anchor | Use here |
|---|---|---|
| Session protocol typed-state-machine | López de Prado AFML Ch.2 | Sessions are first-class workflow entities |
| Air-gap predictions from verdict logic | Tetlock 2017 | Preserved in S4 — Belief Layer fires before strict gate |
| Pre-mortem / cross-domain / replication review | Klein 2007 pre-mortem | α/β/γ personas invoked from S4 |
| Cost discipline | Almgren-Chriss 2001 | Per-station cost preview + cap |
| Multi-test correction | Bailey-López de Prado 2014 DSR | n_trials denominator surfaced in S4 pre-flight |
| Paired vs unpaired SE | Politis-Romano 1994 | Routing FORWARD vs ENHANCE in S4 vs S5 |
| Implementation shortfall | Perold 1988 | Station progress observability (decision_ts vs execute_ts) |
| Forecaster track-record discipline | Tetlock 2017 Superforecasters | Belief layer integration in S4 + S6 |
| Audit trail | SEC 17a-4 | Every emit immutable; supersedence chain via parent_event_ids |

---

## 9. Phasing + Effort

| Phase | Scope | Effort |
|---|---|---|
| **Phase 0a** | Foundation infrastructure (engine/operator_console/ + api/routes + frontend/components) — 4 sub-systems (job queue / SSE / cost ledger / data dependency tagging) | 15-20h |
| **Phase 0b** | SessionLauncher redesign + rebuild | 4-5h |
| **Phase 1** | S1 (Paper Ingest) + S4 (FORWARD Dispatch, 8-stage statistical rigor pipeline) + S6 (Verdict View) — minimum E2E | 27-32h |
| **Phase 2** | S5 (ENHANCE Dispatch) + S7 (PROMOTE 9-gate orchestration) — institutional story | 22-28h |
| **Phase 3** | S2 (Synthesize) + S3 (FactorSpec Extract) + S8 (Rollback) + S9 (Doctrine Lock, micro-station) — complete toolset | 22-26h |
| **Phase 4** | Polish: error states, observability dashboard, audit replay, RED-lesson warnings, decay-sentinel integration, mobile-responsive (defer if time), demo fixture authoring, i18n coverage | 15-20h |
| **Total** | | **105-131h** |

Calendar estimate: 3-4 weeks full-time, 6-8 weeks part-time.

**Note on estimate revision**: initial draft was 85-105h; revised up
~20% during self-audit after identifying (a) Foundation is 4 sub-
systems not 1, (b) S4 FORWARD dispatch alone is 12-15h including
8-stage statistical rigor pipeline + Belief Layer pre-record +
autopsy join + capability evidence emission, (c) Phase 4 must cover
i18n bilingual coverage per memory STANDING rule.

### Recommended commit cadence

- 1 commit per phase sub-task (~5h work each)
- Each commit must leave the system in a working state
- Each station's first commit must include its demo fixture
- No "WIP merge later" — every commit is a usable checkpoint

---

## 10. Risks + Mitigations

| Risk | Mitigation |
|---|---|
| WRDS vendor lock-out / IP allowlist drop | Demo fixtures bundled with each station; clearly-labeled `wrds_required` tier; documentation for users with own WRDS to wire credentials |
| LLM cost overrun | Per-session hard cap (D4); preview before trigger; anonymous-mode locks LLM-heavy stations |
| Async job orphaning (server restart kills running job) | Job state persisted to `data/operator_console/jobs.jsonl` before execution starts; restart picks up `running` rows and marks them `recovered_unknown` |
| Schema migration if multi-user added later | Already prepared: `actor_id` in every event/session/job from day 1 (D5) |
| User cancels mid-dispatch | CancellationToken honored only at stage boundaries — user must wait for current stage to complete (e.g. Bootstrap CI takes ~30s to reach next checkpoint). UI shows "cancelling…" state with current-stage timer; emit `job_cancelled` event when stage boundary hit; LLM cost charged for tokens already spent. **Cannot interrupt mid-stage; this is an accepted limitation.** |
| User session abandoned (closed tab) | Session-state persistence via URL params; reload reattaches to live SSE stream by job_id |
| Bad LLM citation in synthesis | Citation verifier sub-agent (existing in `engine/agents/papers_curator/ssrn_crossref_crawler.py`) runs as automatic post-step in S2 |
| Server restart mid-job | **Job state IS lossy across restart** (FastAPI background workers don't persist state mid-run). On startup, scan `jobs.jsonl` for rows with `status=running`, mark them `recovered_unknown`, and present user with "Resume / Abort" choice on next visit. Acknowledged limitation; real fix requires external queue (Celery/RQ), deferred to Phase 5. |
| Hit cost cap mid-execution (actual cost exceeds estimate) | Pre-flight uses worst-case estimate. If mid-execution cost exceeds cap by >20%, halt at next stage boundary, emit `job_halted_cost_cap`, return partial result. Default policy: HALT (safer than auto-extend). User can raise cap and resume. |

---

## 10.5. Cross-cutting Integration Requirements

Five project-wide requirements that every station + Foundation must
honor. Self-audit caught these missing from the initial draft.

### IR1 — i18n (bilingual coverage, STANDING memory rule)

Per `feedback_research_area_must_be_i18n_2026-06-06`: all new
user-facing strings in `/research/*` must use `t()` wrapper. Operator
Console pages live under `/research/console/*`, so this rule applies
without exception.

**Implementation**:
- Every label in `StationSpec` (title, description) has matching
  `labelKey` / `descriptionKey` for `t()` lookup
- `frontend/lib/i18n/zh.json` + `en.json` get a `console.*` namespace
- New station spec checklist requires both translations before merge

### IR2 — Error observability surface

When a station fails (any stage emits `stage_failed`), the UI must
provide:
- Inline error banner with stage name + first error line
- "Open forensic" link → opens a new doc `docs/operator_console/
  failures/<job_id>.md` with full context (config, stage events,
  error trace, environment)
- Emit `console_job_failed` event with structured failure category

Forensic generation uses the existing capability-evidence pattern;
no LLM cost incurred at error time (deterministic capture).

### IR3 — Audit Replay capability

Per SEC 17a-4 / SEC Rule 204-2 spirit (immutable, time-stamped,
queryable decision record):

`/research/sessions/<session_id>/replay` page renders every event
emitted within the session in temporal order with playback controls
(step / fast-forward / pause). Each event card shows full payload +
links to artifacts referenced. This is the audit deliverable an
external risk reviewer would inspect.

### IR4 — Decay sentinel hookup

When a deployed sleeve raises a `decay_alert` event, dashboard banner
suggests starting an `audit` session targeting that sleeve. Inside
the audit session, S8 Rollback station pre-flight should pre-populate
the sleeve_id from the decay_alert payload, making the rollback path
1-click.

### IR5 — RED lesson / graveyard warnings

When a user configures S2 Synthesize or S4 FORWARD Dispatch in a
mechanism family where the graveyard hit rate exceeds 60% RED in
last 12 months, pre-flight surfaces a yellow warning:
```
⚠ Family `<family>` has 8 RED / 12 total verdicts in last 12 months.
   Historical hit rate suggests low probability of survival.
   Top 3 RED reasons: <reason_a>, <reason_b>, <reason_c>.
   Continue anyway?
```

Pulls from `red_lessons.jsonl` via `query_graveyard` tool already
exposed through MCP. Implementation: pre-flight call to graveyard
summary endpoint.

---

## 11. Out-of-scope but worth flagging

- **Authentication**: not in scope; single-user fork model. If multi-
  user requested, this becomes Phase 5 (`actor_id` schema already
  in place, so retrofit is auth + RBAC layer only).
- **Production deployment trigger** (real money): permanently
  doctrine-blocked. UI displays "PAPER TRADE ONLY" badge.
- **Real-time intra-day rebalance**: not the project's design point;
  daily 23:00 rebalance via existing cron stays the only path.
- **Demo Mode "Take the Tour"**: deferred per user decision —
  Group A targeted via PPT.

---

## 12. Acknowledgments + Related Work

- The 3-layer SKILL framing from `docs/architecture/SKILL_three_layers.md`
  applies parallelly: this Operator Console is the **operationalization**
  of the L1 (Substrate) and L3 (Self-correction) layers, while L2
  (Experience) is the read-only data feeding the console.
- Concurrent work by QuantML on Kimi+WorldQuant BRAIN SKILL
  (2026-06-23) targets a different operating model (autonomous
  3-day unattended runs); this console is opposite (human-in-loop,
  session-gated, fully observable). Both can coexist as different
  modes of agentic-quant operation.

---

## 13. Sign-off

This doc is the lock-point. Once signed off (a git commit referencing
this doc by SHA), the 5 architectural locks (D1-D5) cannot be revisited
without a new design doc.

**Build commences with Phase 0a (Foundation) after sign-off.**
