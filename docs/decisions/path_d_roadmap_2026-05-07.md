# Path D Roadmap — Production-Grade Applied AI-Quant System MVP

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — supervisor approved 2026-05-07 |
| Total duration | 10-14 weeks (8-9 phases) |
| Identity | Production-grade applied AI-quant system MVP under free-data constraints |
| Supersedes | project_reframe_2026-05-03 (Path B) / project_applied_focus_reframe_2026-05-07 (Path C intermediate) |

## 0. Identity (locked)

**"在数据 / 资源限制下用 AI-era 架构做的 production-grade 应用量化系统 MVP，1 个 shipped 策略，rigorous 筛选方法论，self-improving agent 模式。"**

不是 fund / 不是 lab / 不是 startup — 是 **applied AI-quant engineering**。
target audience: senior tech / quant fund recruiter + MSBA committee.

## 1. Three-axis architecture

```
                    Production Engineering
                    (always-on, monitored, audited, recoverable)
                              │
              ┌───────────────┼───────────────┐
              │                               │
    Methodological Rigor              AI-era Architecture
    (pre-reg + amendment              (Tier R 3-layer +
     ledger + HARKing                  Project History RAG +
     R1-R4 + BHY FDR)                  Reflexion memory +
              │                        Auto-Spec Drafting)
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    QL01 BAB (FP 2014)
                    + 7+1 hypothesis chain
                    + forward NAV from 2026-05-07
```

## 2. Phases

### P0 — Narrative consolidation (3-5 days)
**Goal**: Eliminate "四不像" anxiety; project gets a coherent story.

Deliverables:
- `README.md` rewrite with 3-axis framing
- `docs/architecture_diagram.md` (Mermaid)
- `docs/elevator_pitch_30s.md`
- `docs/interview_cheatsheet.md` (8 distinctiveness × paper anchor × 30s talk)
- `docs/bab_risk_profile.md` (forward drawdown expectations + recovery plan)

Cascade audit tier: **T3 (light)** — pure docs.

Integration test: human review (supervisor read-through).

Rollback: docs only, trivial revert.

### P1 — Surface existing distinctiveness (1-1.5 weeks)
**Goal**: Make the 8 architectural distinctiveness points visible in UI + demo flow.

Deliverables:
- 10-min demo flow (positions → Brief → audit panel → spec inspection → trade ledger)
- UI consistency polish (mono fonts / terminal style / dark mode tokens unified)
- BAB transaction cost real model (replace 13bp constant with: spread + impact + fee model)
- Brief page polish (current is functional, make it exec-summary-grade)

Cascade audit tier: **T2 (medium)** — multi-page UI + cost model change.

Integration test: 22-page smoke + 1 demo dry-run.

Rollback: revert per page if needed.

### P2 — Project History RAG  ✅ **SHIPPED 2026-05-07** (1 day actual vs 1-2 wk planned)
**Goal**: First AI-era visible feature; supervisor query interface over project history.

**Status**: All 7 sub-sprints (P2.1-P2.7) shipped. 700 docs indexed across 5 sources. Live LLM synthesis verified ($0.0019/call, evidence_quality=strong). UI page registered. Daily incremental hook live. 14 pytest pass + recall@5 = 1.0. See `docs/agentic_rag_capability.md`.

Deliverables:
- New module `engine/agents/history_rag/`
  - `indexer.py` — incremental indexing of decision_log / spec_registry / macro_brief / audit_findings / reflections
  - `retriever.py` — semantic search over chromadb local store
  - `synthesizer.py` — optional LLM call to compose answer from retrieved context
- New table `rag_index_metadata` (track what's indexed + when)
- Daily indexer hook in `daily_batch.py` (incremental)
- New page `pages/research_console.py` OR Brief page tab
  - Search box + filters (source / date range)
  - Top-K retrieval display with row id / spec_hash references
  - Optional LLM-synthesized answer with citations
- pytest mock tests (4-6 cases: empty store / single hit / multi-hit / no-match)

Tech stack:
- `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- `chromadb` (local SQLite-backed, free)
- `gemini-2.5-flash` (existing pool, $0.01/query if synth enabled)

Cascade audit tier: **T1 (heavy)** — new agent + new dependencies + UI.

Integration test: pytest + 22-page smoke + manual query: "为什么 ship QL01 BAB" → expect retrieval of B++ migration spec.

Rollback: page deletion + module deletion + chromadb file removal.

Cost: ~$1-5/year LLM if synth enabled; $0 if retrieve-only.

### P3 — CFTC COT integration + factor research (4-6 weeks)
**Goal**: Distinctive free data + 8th hypothesis test.

Sub-phases:

#### P3a — COT data fetcher (1.5 weeks)
- New `engine/data_sources/cftc_cot.py`
- New table `cftc_cot_weekly` (commodity_code + report_type + week + commercial_long/short + non_commercial_long/short + retail_long/short + open_interest)
- ETF→commodity mapping (`engine/universe_manager.py` extension)
  - GLD → 088691 (Gold)
  - USO → 067651 (Crude Oil WTI)
  - UUP → 098662 (US Dollar Index)
  - DBA → multi-commodity (corn / wheat / soy)
  - etc. for 8-9 ETFs
- Weekly cron in `daily_batch.py` (Friday only, T+3 lag)
- Backfill 5y history one-time script
- pytest fetcher robustness (network fail / format change)

#### P3b — VVIX/SKEW expansion (1h trivial)
- Add `^VVIX` + `^SKEW` to regime classifier inputs
- Update `engine/regime.py` MSM features

#### P3c — COT-conditional BAB pre-reg test (2-3 weeks)
- Pre-reg spec doc using P3.5 auto-draft (once available)
- Hypothesis: speculator extreme positioning (top/bottom decile) → reverse signal for BAB on commodity ETFs
- Decision rule: Sharpe > X with NW t > 1.96 + BHY-aware in EFFECTIVE_N_TRIALS context
- Frame as **explore-only**, NOT ship candidate (no literature precedent for COT × BAB combination)
- Verdict added to falsification chain regardless of pass/marginal/fail
- Backtest with `engine/backtest.py` + freeze data via `data_snapshot.py`

Cascade audit tier: **T1 (heavy)** — new data source + new pre-reg.

Integration test: data validation (sample COT row matches CFTC website manual lookup) + Tier R critical sweep + pytest.

Rollback: drop tables + delete module + remove regime feature.

Cost: $0 (CFTC free).

### P3.5 — Auto-Spec Drafting (1-2 weeks)
**Goal**: LLM-as-scientific-collaborator capability; integrates with P3 spec creation.

Deliverables:
- New module `engine/agents/spec_drafter.py`
- Reuses `engine.key_pool` `get_model(response_schema=...)` pattern
- Input: natural language hypothesis + universe + regime context
- Output: pre-reg spec markdown draft (hypothesis / decision rule / N_TRIALS impact / risk profile / spec_hash placeholder)
- UI: orchestrator page new section "Draft Pre-reg Spec" or new page
- Output goes through Tier R-style validation gate before persist
- pytest mock LLM tests

Cascade audit tier: **T2 (medium)** — new agent + LLM call + spec creation flow.

Integration test: draft P3 COT spec via this tool, manual review, freeze.

Rollback: module + UI deletion.

Cost: $0.05/draft × ~10 drafts/year = $0.5/year.

### P4 — Reflection accumulation polish (2-3 weeks, in background)
**Goal**: Let reflection memory naturally accumulate; polish the write loop.

Deliverables:
- Verify `generate_reflections_for_pending` runs cleanly with real LLM (not None)
- Add cron trigger if missing (currently triggered by orchestrator)
- Reflection writer hash-chain (chain `prev_reflection_hash`) for tamper-evidence
- Reflection retrieval interface for sector_pipeline (deferred consumer; build now)
- pytest reflection writer + retrieval

Cascade audit tier: **T2** — modify existing agent.

Integration test: 1 reflection generated end-to-end with real LLM call.

Rollback: revert hash chain addition; existing infra preserved.

### P5 — Cull dead weight (3-5 days)
**Goal**: Reduce maintenance surface, keep only Path-D-aligned code.

Deliverables:
- Delete `engine/agents/macro_research/` (dormant, killed by meta-audit)
- Clean stale `track_b` / `narrative_overlay` references in code + docs
- Drop confirmed-dead empty tables: `track_b_*`, `narrative_*`, post-cleanup residual
- KEEP: reflection / NewsPerceiver / macro_brief_llm / sector_pipeline LLM debate (frozen, not extended)
- Update `audit_agent_liveness.py` KNOWN_AGENTS
- Update `auto_audit_rules.py` PRODUCTION_CODE_FILES

Cascade audit tier: **T2** — multi-file deletion.

Integration test: Tier R + pytest + 22-page smoke.

Rollback: git revert.

### P6 — Deployment + ops documentation (1 week)
**Goal**: Recruiter-facing engineering polish.

Deliverables:
- `docs/deployment_guide.md` (env setup / API keys / dependencies / first-run)
- `docs/api_reference.md` (CLI + key Python entry points)
- `docs/runbook.md` (failure modes + recovery procedures + escalation)
- `docs/operations_manual.md` (daily / weekly / monthly cycles)
- Architecture diagram with deployment topology

Cascade audit tier: **T3 (light)** — docs only.

Integration test: human review.

Rollback: trivial.

## 3. Dependencies graph

```
P0  ──→ P1
        │
        ├──→ P2 (RAG)  ─────────┐
        │                       │
        └──→ P3a (COT data) ─→ P3c (factor research) ──→ P3.5 (auto-spec, parallel with P3c)
                                                          │
                                                          ▼
                                                          P4 (reflection polish, in background)
                                                          │
                                                          ▼
                                                          P5 (cull) → P6 (docs)
```

P0 is gating. P1+P2 can run parallel. P3+P3.5 can interleave. P4 background. P5+P6 final.

## 4. Risk register

| Risk | Mitigation |
|---|---|
| CFTC COT data format changes break fetcher | Defensive parsing + Tier R rule for "n_rows weekly" SLA |
| RAG retrieval quality thin on 280 rows | Honest framing in demo: "architecture in place, accumulating naturally" |
| COT-conditional BAB shows messy result | Pre-spec FRAMED as explore-only, not ship candidate |
| BAB live regression during phases | P0 includes BAB risk profile doc; supervisor pre-warned |
| Timeline slips beyond 14 weeks | P5 + P6 can defer if needed; P0+P1+P2 gives demo-ready state in 4 weeks |
| Scope creep adding more capability layer | NEW spec must answer "Path D 3-axis fit" or rejected |

## 5. Out-of-scope (explicit)

- ❌ Real money / live trading
- ❌ Alt data (no satellite / sentiment / fine-grained)
- ❌ Multi-strategy ensemble
- ❌ HFT / tick / minute frequency
- ❌ LLM fine-tuning
- ❌ Multi-modal data
- ❌ News curator (Wave 8 rolled back; do not revisit)
- ❌ Narrative overlay revival (rejected 2026-05-03)
- ❌ Reflexion-style influence on portfolio decisions (0-LLM-in-eval red line)

## 6. Today's first action

**P0.1 — `README.md` rewrite (~2 hours)**

Sections:
1. WHAT IT IS (Path D identity statement)
2. NOT (3 explicit non-claims)
3. YES (3-axis with 8 distinctiveness points)
4. Live state (NAV / cycles / decisions counts)
5. Quickstart (run / view)
6. Architecture pointer (link to architecture_diagram.md)

Then commit + push for supervisor review tomorrow.

## 7. Memory entry

Saved at `memory/project_path_d_roadmap_locked_2026-05-07.md`.
