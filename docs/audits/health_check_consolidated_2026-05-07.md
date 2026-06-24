# Deep Health Check — Consolidated Findings (2026-05-07)

| Field | Value |
|---|---|
| Sprint | Applied-focus reframe deep audit |
| Days run | 3 (anti-pattern grep / DB introspection / runtime / write-path / UI) |
| Time spent | ~4 hours |
| Total findings | **3 active HIGH · 4 MEDIUM · 2 LOW · 4 legacy disclosures · 4 dead-capability triage items** |
| Smoke status | 20/21 pages PASS · signal_board 60s timeout (cold cache, not bug) |
| Tier R | 11 rules / 0 findings · hash chain INTACT |

> **Bottom line**: most "high-severity" findings from Day 1 turned out to be
> either already-fixed (Wave 0 Option A done) or **legacy data state** rather
> than active code bugs. Active bug list shrinks to 3 HIGH + 4 MEDIUM. The
> bigger applied issue is **capability-vs-data congruence** — many features
> claim to exist but their tables are 0-row.

---

## Wave 0 — Already done (this session)

| # | Item | Status |
|---|---|---|
| W0.1 | populator: `created_at` → `decision_date` semantic | ✅ memory.py:2280-2371 |
| W0.2 | populator: direction-signed `active_return / MAE / MFE / payoff_quality` | ✅ memory.py:2515-2557 |
| W0.3 | get_stats pending count: same fix | ✅ memory.py:5184-5223 |
| W0.4 | Migration: 138 rows un-verified + re-verified with new convention | ✅ scripts/_migrate_active_return_signed.py |
| W0.5 | Sanity: short P&L sign correct (SMH 2015-07 短仓 +10.7% confirmed); long unchanged; neutral NULL'd | ✅ |

---

## Active bugs (need code fix)

### 🔴 HIGH

#### H1 · DSR kurtosis convention bug
- **Location**: `engine/backtest.py:341-350`
- **Issue**: pandas `kurtosis()` returns excess (γ₄−3); BLP DSR formula expects raw γ₄
- **Code**: `(kurt - 1)/4` should be `(kurt + 2)/4` if keeping pandas excess kurt, OR use scipy with `fisher=False`
- **Cascade**: every reported DSR low by ~3/8·SR² in variance term → consistently overconfident
- **Fix cost**: 30 min code + 1h cascading recompute + amendment ledger

#### H2 · sector_pipeline doesn't pass weight_before/weight_after
- **Location**: `engine/agents/sector_pipeline/agent.py:298-332`
- **Issue**: save_decision call omits these kwargs → DecisionLog stores NULL
- **Note**: save_decision does write them when passed (memory.py:1218-1219); kwarg infrastructure correct
- **Fix**: compute pre-debate `weight_before` (quant baseline) + post-debate `weight_after` (after `weight_adjustment_pct`) in agent, pass to save_decision
- **Fix cost**: 1h code + 1 e2e test

#### H5 · portfolio_tracker omits direction on SimulatedPosition write
- **Location**: `engine/portfolio_tracker.py:611-623` (primary write — 46 NULL positions)
- **Reference**: `engine/memory.py:1374-1389` shows correct pattern (passes `direction=note_dir`) — 9 positions correctly populated
- **Fix**: derive direction from `signal_tsmom` sign or `target_weight` sign, pass to SimulatedPosition()
- **Fix cost**: 30 min code + smoke

### 🟡 MEDIUM

#### M2 · datetime.now / utcnow inconsistency
- **Location**: `engine/key_pool.py` (3 local + 1 UTC), `engine/macro_fetcher.py` (1), `engine/quant.py` (2)
- **Total**: 6 outliers vs 95 utc — system mostly UTC
- **Risk**: key rotation timing tz-shifted; cross-day comparisons inconsistent
- **Fix cost**: 30 min global replace

#### M3 · `engine/quant.py.tmp.7804.1774967677731` orphan backup file
- **Fix**: 1 line delete

#### W3.1 · `use_container_width=True` deprecation (will break after 2025-12-31)
- **Found in**: 7+ pages flagged in AppTest output (live_dashboard, signal_board, etc.)
- **Migration**: `use_container_width=True` → `width='stretch'`; `=False` → `width='content'`
- **Fix cost**: 30 min batch sed-style migration

#### M6 · agent_runs anomalously low
- sector_pipeline: 10 runs total / expected daily ≈ 180/6mo
- macro_research: 5 runs / weekly ≈ 26/6mo
- universe_review: 0 / quarterly ≈ 2-3/6mo
- memory_curator: 0 / monthly ≈ 6/6mo
- **Investigation needed**: scheduler trigger wired? agents are running but not logging? schedules mis-defined?

### 🟢 LOW

#### L1 · signal_board cold-cache 60s timeout in AppTest
- Not a bug — yfinance + MSM regime fit on cold cache takes ~40-80s; in production cached <1s
- **Action**: extend AppTest budget for this page, or warm cache fixture

#### L2 · approval_context.py:775 created_at-2h time-window
- **Verified**: this IS the right semantic (system-time approval-to-log pairing, not decision-date)
- No action needed

---

## Legacy disclosures (Wave 3 — RECLASSIFIED to NO-OP after deeper audit)

**Update 2026-05-07 (Wave 3 self-audit)**: original finding was overstated.
Full page sweep (4 ref sites: decision_journal / approval_analytics /
orchestrator / agent_observability) found existing UI already handles NULL
legacy fields gracefully via `if X is not None else "—"` / `or "—"`
fallbacks. Supervisor sees clean "—" for missing fields rather than broken
state. **No UI code change required.** New post-fix rows will populate
naturally; legacy "—" rows remain by design (pre-instrumentation).

These are **NULL fields on pre-2026-05-04 data** because the feature was added 2026-05-04.
Going forward, new rows DO get populated. UI is already defensive — no hint needed.

#### Legacy-1 · spec_hash NULL on 44 sector decisions (latest 2026-04-17)
- All 44 sector decisions pre-date 2026-05-04 spec_hash addition
- sector_pipeline DOES correctly compute and pass spec_hash now
- `_compute_git_blob_hash` verified working: returns `95004f3909e0b8ead2de2445d2c0e6ac6b9aa5c4`
- **Action**: UI should display "spec_hash linkage retroactive: pre-2026-05-04" badge, OR backfill these rows with current spec_hash (legitimate disclosure of when feature was added)

#### Legacy-2 · 8 PendingApproval rows have NULL review_rationale
- All 8 resolved on **2026-04-29 10:02** — exactly 5 days before P-AUDIT v1 launch (2026-05-04)
- UI approve path now correctly passes review_rationale (orchestrator.py:2475-2485)
- **Action**: UI should distinguish "approved without rationale (legacy)" vs "approved with rationale (post-launch)"

#### Legacy-3 · 7 auto_approved PA rows have NULL resolved_at
- Auto-approval path doesn't set timestamp — pre-launch behavior
- **Action**: 1-line fix in auto-approval path; backfill from `created_at` for legacy rows

#### Legacy-4 · 180 decisions have NULL `weight_before/weight_after`
- Once H2 is fixed going forward, these legacy rows will remain NULL
- **Action**: same UI disclosure pattern

---

## Dead-capability triage (M1 — 17 empty tables)

Per-table classification + recommended action:

| Table | Status | Action |
|---|---|---|
| `learning_log` | dead-by-design (LearningLog model unused) | ⚠️ Decide: drop column / drop model |
| `skill_library` | dead-by-design (cleanup 2026-05-07 documented) | ✅ no action |
| `agent_reflections` | unblocked Day 1 but no live LLM call yet | 🟢 next session run with model + populate |
| `harking_flags` | HARKing detector never fired any flag | ⚠️ adversarial test to confirm detector works on synthetic positive |
| `cash_flows` | P-FUND cash flow tracking unused / never invoked | ⚠️ wire entry point or document as "ready for use" |
| `auto_audit_findings` | sweep clean (which is healthy) | ✅ healthy zero state |
| `auto_audit_proposals` | Layer 1 LLM never invoked production | ⚠️ not a bug if no findings; but capability claim resting on this |
| `alpha_memory` | post-cleanup expected | ✅ no action |
| `risk_narrative_logs` | post-cleanup expected | ✅ no action |
| `circuit_breaker_log` | breaker never triggered | ✅ no action (good news) |
| `discovered_factors` | unwired / threshold never hit | ⚠️ document or remove |
| `watchlist_entries` | UI exists, never used? | ⚠️ wire or remove |
| `spillover_weights` | unwired | ⚠️ document or remove |
| `stress_test_log` | unwired | ⚠️ document or remove |
| `quant_pattern_log` | unwired | ⚠️ document or remove |
| `simulated_monthly_returns` | should populate from paper trading | ⚠️ verify population path |
| `anomaly_universe_events` | universe events never recorded | ⚠️ document |

**Summary**: 7 dead-by-design / 5 healthy zero / 5 needs investigation.

---

## Pipeline reactivation (M4 / M5 / M6 — Wave 4 RECLASSIFIED to NO-OP)

**Update 2026-05-07 (Wave 4 self-audit)**: Day 1 findings were based on
incorrect "6-month active" premise. cycle_states audit shows project's
active history starts 2026-05-04 — only 3 days active when Day 1 audit ran.

| Agent | True status | Bug? |
|---|---|---|
| sector_pipeline | last run 2026-05-04 03:53; daily_batch trigger wired | NO |
| macro_research | last run 2026-05-04 03:53; weekly via base.Agent | NO |
| universe_review | code OK; `GATE_UNIVERSE_CHANGE` disabled by design (conservative default); next quarterly trigger = 2026-07-01 (Q3) | NO |
| memory_curator | code OK; ran 2026-05-01 (1 report in DB); month-end trigger wired at daily_batch.py:2393; next = 2026-05-29/30 | NO |

quarterly chain wiring: confirmed at daily_batch.py:2483-2492; fires on
`_is_first_trading_day_of_quarter(t_day)`. Q2 trigger (2026-04-01) missed
because project not yet active. Next: Q3 2026-07-01.

agent_runs 15 / 3-day active period = 5/day healthy. `audit_agent_liveness`
already correctly handles function-based agents via downstream_table check
(script lines 138-142).

---

## Wave-fix plan with integration test gates

```
Wave 1 — Foundational corrections (low risk, isolated)            ~2 hours
  · H1 DSR kurt convention fix + cascading recompute
  · M2 datetime tz consistency global replace
  · M3 orphan tmp file delete
  · W3.1 use_container_width deprecation migration (batch)
  Integration test:  Tier R critical sweep · 21-page smoke ·
                     pytest unit · DSR sanity on 1 strategy

Wave 2 — Write-path completions (the NULL-100% cluster)            ~3 hours
  · H2 sector_pipeline pass weight_before / weight_after
  · H5 portfolio_tracker pass direction
  · Auto_approval set resolved_at (Legacy-3 fix)
  Integration test:  21-page smoke · 1 sector_pipeline e2e (need
                     real or mocked LLM call) · DB column NULL %
                     check (weight_before drops below 100% on new
                     rows) · 1 portfolio rebalance e2e
                     (direction populates)

Wave 3 — Legacy disclosures (UI / docs)                            ~1.5 hours
  · Legacy-1 / Legacy-2 / Legacy-4 UI badge / "pre-X-date" disclosure
  · Backfill spec_hash on 44 historical sector decisions
    (technically retroactive but documented)
  Integration test:  21-page smoke · supervisor view check on
                     decision_journal + system_hub approval list

Wave 4 — Pipeline reactivation                                     ~2.5 hours
  · M4 universe_review scheduler wire
  · M5 memory_curator scheduler wire (debug existing partial)
  · M6 audit which agents don't log to agent_runs
  Integration test:  agent_liveness audit improves;
                     trigger universe_review manually + verify row

Wave 5 — Capability-vs-data alignment                              ~3 hours
  · M1 per-table triage execution (drop dead / wire usable)
  · README + project_report.md update to honest capability state
  · NEW Tier R rule: capability_vs_data_congruence
  Integration test:  Tier R passes new rule on healthy zero-states
                     and flags genuine dead branches; doc smoke

Wave 6 — Defensive (test coverage)                                 ~2 hours
  · pytest: active_return populator long / short / neutral cases
  · pytest: portfolio_tracker direction-sign correctness
  · pytest: DSR convention check (after H1 fix)
  Integration test:  pytest 89 → ~95 passing

Total estimate: 14 hours (~2 working days)
```

## Recommended execution order

Wave 1 → Wave 2 → Wave 3 → Wave 4 → Wave 5 → Wave 6.

Reasoning:
- Wave 1 isolated; no dependencies
- Wave 2 depends on no other; some may need agents to actually run for end-to-end validation
- Wave 3 is UI / docs only; safe after data state is correct
- Wave 4 reactivates dormant capabilities; some may surface NEW data quality issues
- Wave 5 honest accounting based on now-clean state
- Wave 6 locks in the gains with tests

## What this audit changes about the 8-week plan

Original Week 1-8 (applied-focus 8-week plan):
```
W1 Item 5  W2 Item 1  W3 Gap A  W4-5 Item 4  W6 Gap B  W7 Item 6+C  W8 Gap D
```

New Week 1-9 (with deep audit absorbed):
```
W1 (done)  W1.5 Wave1+2+3+4+5+6  W2-W9 = original W2-W8 shifted by 1 wk
```

Net cost: **+1 week** to absorb the deep-audit findings. Acceptable per
supervisor 2026-05-07 ("没关系我接受这个 cost").

## Meta-observation (carries over to future audit infra)

Tier R + Tier 1 audit + agent_liveness all green on this codebase, but **the
deep audit found 3 active HIGH bugs they didn't catch**. The audit gap:
**field write-completeness** ("this column should be filled in the steady
state") + **capability-vs-data congruence** ("we claim capability X; does
its table have non-zero rows?"). Wave 5 proposes a new Tier R rule class
covering this — and that itself is a meaningful capability extension beyond
just bug fixing.
