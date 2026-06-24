# Day 1 Deep Health Check — Findings (2026-05-07)

| Field | Value |
|---|---|
| Sprint | Applied-focus reframe deep audit |
| Sweep types | Anti-pattern grep · DB introspection · Tier R critical · Agent liveness |
| Time spent | ~2 hours |
| Findings | **5 HIGH · 6 MEDIUM · 4 LOW · 7 healthy checks** |

> **Bottom line**: the system claims more capabilities than it actually generates
> data for. Many pages render UI from tables that have 0 rows, presenting
> empty-state to supervisor without flagging that the upstream agent/feature
> is dead. Distinct from previously-found Pattern 1 (created_at semantic) and
> Pattern 2 (direction sign) — both already fixed in Option A.

---

## 🔴 HIGH severity (user sees wrong number / dead claim presented as live)

### H1 · DSR kurtosis convention mismatch
- **Location**: `engine/backtest.py:341-350`
- **Issue**: `kurt = returns.kurtosis()` returns Fisher excess kurt (γ₄ − 3); formula uses `(kurt - 1)/4`. Bailey-López de Prado 2014 §3 expects raw γ₄ → `(γ₄ − 1)/4`. Discrepancy: variance term off by 3/4·SR² (low).
- **Impact**: every reported DSR systematically overconfident. Affects S1 multi-window, B++ Mass FDR, P-FUND.
- **Status**: known, scheduled Week 2 Item 1. **DO NOT defer past Week 2.**

### H2 · DecisionLog.weight_before / weight_after 100% NULL across 180 rows
- **Location**: write path unknown — needs grep
- **Issue**: DL-P0 attribution fields claimed in `project_decision_log_dl_p0` memory entry as "added 2026-05-03". 180 decisions later: 0 populated.
- **Impact**: attribution panels claim weight delta visibility; reality = always blank.
- **Severity**: HIGH — fake capability presented to supervisor.
- **Day 2 task**: trace where save_decision should set these.

### H3 · DecisionLog.spec_hash 100% NULL across 180 rows
- **Location**: write path unknown
- **Issue**: spec_hash is the field linking a decision to the pre-registration spec it was made under. Never written.
- **Impact**: pre-registration discipline claim is **structurally** sound (specs registered) but **decisions are not linked back to specs** → can't audit "which spec generated this decision". Big credibility gap if examiner asks.
- **Severity**: HIGH — methodological claim partially false.
- **Day 2 task**: verify whether this was meant to be auto-set on save or manually attached, and where the hook should be.

### H4 · PendingApproval review fields 100% NULL across 20 PAs
- **Location**: `engine/memory.py::resolve_pending_approval` write path
- **Issue**: `review_rationale / review_narrative_snapshot / review_narrative_hash / review_category / rejection_reason / rejection_category` all 100% NULL across 20 PAs.
- **Possible explanations**:
  - (a) all 20 PAs are still status='pending' — no supervisor ever resolved any (verify via status query)
  - (b) resolve path doesn't actually write these fields — bug
  - (c) PAs were synthetic from R-1.E test runs, not real supervisor actions
- **Impact**: P-AUDIT v1 capability claim ("3-layer expander + historical replay + narrative hash chain") rests on these fields. If always NULL, the claim is hollow.
- **Day 2 task**: query PA status distribution + grep resolve_pending_approval to find write side.

### H5 · simulated_positions.direction 83.6% NULL on 55 rows
- **Location**: position-write path in paper_trading or simulation modules
- **Issue**: direction (long/short) field NULL on 46/55 positions. A position without direction has no defined P&L sign.
- **Impact**: any P&L on these positions is direction-undefined → likely skewed in aggregations.
- **Day 2 task**: trace position-write side; either backfill direction or NULL means "unused row".

---

## 🟡 MEDIUM severity (silently dead / silently skewed)

### M1 · 17 tables completely empty
| Table | Likely cause |
|---|---|
| `learning_log` | LearningLog model unused / dead code |
| `skill_library` | documented dead branch (cleanup 2026-05-07) |
| `agent_reflections` | reflection trigger now unblocked but no live LLM call yet (we ran with model=None) |
| `harking_flags` | HARKing detector never fired any flag (low N spec base?) |
| `cash_flows` | P-FUND cash flow tracking unused / never invoked |
| `auto_audit_findings` | Tier R sweep clean since rule library finalized — could be genuine OR could be sweep results never persisted |
| `auto_audit_proposals` | Layer 1 LLM proposer not invoked in production |
| `alpha_memory` | post-cleanup expected (Track B killed) |
| `risk_narrative_logs` | post-cleanup expected (risk_narrative_agent killed) |
| `circuit_breaker_log` | breaker never triggered |
| `discovered_factors`, `watchlist_entries`, `spillover_weights`, `stress_test_log`, `quant_pattern_log`, `simulated_monthly_returns`, `anomaly_universe_events` | never wired or never reached threshold |

**Action**: per-table triage Day 2 — distinguish "dead by design" vs "dead by silent bug".

### M2 · datetime.now / utcnow inconsistency
- **Location**: `engine/key_pool.py` (3 local + 1 UTC in same file), `engine/macro_fetcher.py` (1 local), `engine/quant.py` (2 local)
- **Total**: 95 utcnow vs 6 now() — mostly UTC, 6 local outliers
- **Impact**: key rotation timing in key_pool could be off by tz offset (8 hours for CN); cross-day comparisons could classify "today" differently.
- **Fix cost**: 30 min — replace local `now()` with `utcnow()` in 6 spots.

### M3 · engine/quant.py.tmp.7804.1774967677731 orphan backup file
- **Location**: repo root engine/
- **Impact**: code hygiene; future readers might edit wrong file
- **Fix cost**: 1 line `rm`

### M4 · universe_review agent never run
- 0 total runs / expected cadence: every 90 days
- liveness audit flag: `NO_DOWNSTREAM_DATA_30D`
- claimed quarterly job; scheduler not wired or trigger never fired

### M5 · memory_curator agent never run
- 0 total runs / expected: every 30 days
- 1 downstream call recorded (probably from R-1 sprint test)
- claimed monthly job; never triggered

### M6 · agent_runs = 15 / agent_events = 15 abnormally low
- Project running 6+ months; 15 total agent runs ≈ 2.5/month
- Either agents aren't logging via this path, or agents really do run that rarely
- **Day 2 investigation**: which agents currently log to agent_runs vs not?

---

## 🟢 LOW severity (observation, not active bug)

### L1 · Pattern 1 instances outside populator/get_stats
9 other `created_at` references in code, all using correct `decision_date or created_at` fallback pattern. ✓ healthy.

### L2 · approval_context.py:775 uses `created_at - timedelta(hours=2)`
For approval-to-log time-window matching. System-time relationship is intentional here (not semantic-date). Not a bug.

### L3 · skill_registry.py:188 uses `log.created_at.date()`
For display-time string. May or may not be the right semantic depending on intent of LearningLog. Since learning_log is empty, no current impact.

### L4 · 6 local `datetime.now()` calls in scheduling-adjacent code (M2 above)
Captured under M2.

---

## ✅ HEALTHY checks (passed)

1. **Pattern 2 (direction sign)** — Option A applied; 138 rows re-verified with correct sign convention.
2. **Tier R critical sweep**: 11 rules / 0 findings / hash chain INTACT.
3. **quant.py Cornish-Fisher VaR** uses excess kurtosis correctly (different formula from Bailey-LdP DSR; both implementations correct under their respective formulas).
4. **spec_registry**: 36 rows — pre-registration baseline registered.
5. **regime_snapshots**: 962 rows — MSM fitting active.
6. **signal_records / signal_snapshots**: 317 / 205 — signal gen active.
7. **compute_hit_flag (reflection.py:42-52)** — uses predicted_sign × realized correctly.

---

## Day 2 sweep priorities

Based on Day 1 findings:

1. **H2 + H3 + H5**: write-path forensics. For each NULL-100% field, find the call-site that should set it. If found and dead → wire it. If never existed → document as legacy unused field, drop column or mark deprecated.
2. **H4**: PendingApproval status distribution + resolve_pending_approval write-side audit.
3. **M1**: per-table triage — dead-by-design (cleanup-validated) vs dead-by-silent-bug.
4. **M2**: tz consistency fix (30 min).
5. **M3**: orphan tmp file cleanup (1 min).
6. **M4 + M5 + M6**: scheduler wiring audit — what's supposed to fire and isn't.

**Time estimate**: Day 2 = ~3-4 hours.

---

## Meta-observation

Tier R + Tier 1 audit + agent liveness are all structurally green, BUT:

- Tier R checks **consistency** (schema vs ORM, hash chain, spec drift) — passed.
- Liveness checks **runtime activity** (last run timestamp) — caught 2 dead agents.
- Neither catches **field write-completeness** ("this column should be filled but isn't") or **capability-vs-data congruence** ("project claims feature X exists; table for X has 0 rows").

This is a real audit gap. **Day 3 deliverable should propose a new Tier R rule class for capability-vs-data congruence**: for each capability claim in README/project_report, identify the table(s) that should accumulate evidence, and flag if 0 rows persist beyond expected cadence.
