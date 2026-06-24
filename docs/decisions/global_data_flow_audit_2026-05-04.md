# Global Data-Flow Audit — 2026-05-04

**Auditor**: Claude (assistant) at user's direction
**Scope**: full-project sweep for data-flow integrity, schema drift, hash registry consistency, sign-convention violations, recently-added-code edge cases, performance hotspots
**Method**: [scripts/audits/audit_global_data_flow.py](../../scripts/audits/audit_global_data_flow.py) — 5 categories × 14 probes
**Result**: ✅ **No critical bugs, no real high-severity findings**

---

## Headline

| Severity | Count | Real / False-positive |
|---|---|---|
| **CRITICAL** | 0 | — |
| **HIGH**     | 1 | 1 false-positive (audit-script bug) |
| **MEDIUM**   | 1 | informational (by-design) |
| **LOW**      | 4 | 1 real + 2 false-positive (UNIQUE 隐式索引) + 1 cosmetic |

---

## A. Schema integrity (ORM ↔ SQLite)

**10 ORM tables checked**: SimulatedPosition / PortfolioNavSnapshot / CashFlow / PendingApproval / AgentReflection / DecisionLog / SpecRegistry / HARKingFlag / AgentRun / PaperTradingRun.

- ✅ All 10 tables present in DB
- ✅ No ORM-declared columns missing from DB
- ⚪ 1 LOW (cosmetic): `pending_approvals` has 1 column in SQLite that's absent from ORM (legacy field; reading still works)

**Verdict**: schema三方对齐 (ORM / migration / live DB) intact.

---

## B. Cross-module data conventions

### B.1 CashFlow sign convention
- ✅ Clean: 0 rows violate `deposit > 0`, `withdraw < 0`, `dividend > 0`, `fee < 0`
- 100% rows respect the spec §3.1 convention "amount_usd > 0 = INTO portfolio"

### B.2 DecisionLog.spec_hash auto-injection
- ⚠️ MEDIUM (informational): 0 of 44 historical sector DecisionLog rows have spec_hash populated
- **By design**: P-FUND-4b auto-injection only runs on NEW decisions after the S3.2 wiring (2026-05-04)
- Historical rows are NULL — acceptable per spec §1.3 ("不回填历史 spec 的'原版' hash")
- HARKing R3 won't false-fire on these (they have NULL, not unknown hash)

### B.3 Date type integrity
- ✅ `PortfolioNavSnapshot.snapshot_date` deserializes as Python `date`, not string

---

## C. Recently-added edge cases (P-FUND / S2 / S3)

### C.1 P-FUND-2 cold-start NAV with prior cash flow ✅
**Probe**: insert deposit on day 1, no snapshots exist; call `roll_daily_nav(day 2)` with constant-zero return provider.

Expected `nav_open = initial_nav + prior_external = $1M + $100k = $1.1M`. Got **$1.1M** exactly. Cold-start handling correct — prior applied external CashFlow rows accumulate into nav_open before today.

### C.2 XIRR mono-sign guard ✅
`compute_xirr` correctly raises `ValueError` on cash flows with one sign only.

### C.3 EFFECTIVE_N_TRIALS staleness ✅
Module-level `bt.EFFECTIVE_N_TRIALS = 44` matches `bt.refresh_effective_n_trials()` live result. P-FUND forward registration's +1 contribution propagated.

### C.4 SpecRegistry hash drift ✅
- 19 active specs registered
- **0 drifted** (current_hash matches live file hash for all)
- **0 missing files**
- HARKing R1 / R2 quiet — no silent edits.

### C.5 apply_tactical_weight_update empty-call ✅
With `sector_adjustments=None, new_entries=None`, function returns silently without crash.

---

## D. Indexes / performance hotspots

| Table | Indexed cols | Real gap? |
|---|---|---|
| `decision_logs` | only PK (id) | **YES** (44 rows now, low impact; will matter at 10k+) |
| `agent_reflections` | id, agent_id, decision_date, composite (agent+date) | ✓ adequate |
| `cash_flows` | id, flow_date | ✓ adequate |
| `portfolio_nav_snapshots` | snapshot_date (PK) | ✓ adequate |
| `simulated_positions` | snapshot_date+sector unique | ✓ adequate |
| `spec_registry` | UNIQUE(spec_path) → 隐式索引 | ✓ false-positive in audit |
| `harking_flags` | id | ✓ adequate (small table) |
| `paper_trading_runs` | UNIQUE(as_of_date, arm) → 隐式索引 | ✓ false-positive |
| `agent_runs` | id | ✓ adequate |

**Real recommendation**: add Index on `decision_logs.(tab_type, sector_name, decision_date)` composite for the most common UI query pattern. **Not blocking**; defer until row count grows.

---

## E. Cross-module integration probes

### E.1 BacktestMetrics construction
**Probe failed due to my audit-script bug**: I used field names like `sortino`, `win_rate`, `periods_per_year` that don't exist on this dataclass. Real fields are `win_rate_vs_bm`, `ir_vs_bm` (see [engine/backtest.py:145](../../engine/backtest.py#L145)). **Not a real finding** — audit script artifact.

### E.2 reflection retrieve unknown agent ✅
Returns `[]` for nonexistent agent_id, doesn't crash.

### E.3 _get_nav fallback chain ✅
`_get_nav() = $1,000,000` (snapshot table empty → falls back to SystemConfig → falls back to 1M default). All three tiers tested.

---

## Supplementary findings (second-pass deep scan)

### Cache TTL vs supervisor-action responsiveness (FIXED 2026-05-04)

`command_center._get_portfolio_stats` had `ttl=60s`. After supervisor deposit
on Performance Report, NAV on Command Center took up to 60s to update.
**Reduced to ttl=5s**. Other pages already use uncached `_resolve_live_nav()`.

### Track filter on `SimulatedPosition.snapshot_date.desc()` queries

15 occurrences across `engine/`. Only 1 (`daily_batch.py:786` —
`_drift_update_positions`) explicitly filters `track == "main"`.

**Current production**: only `track="main"` rows exist (per
`project_cleanup_2026-05-03.md` — track_b deleted). So queries return main
correctly. **Latent risk**: if track_b/c is re-enabled (future work), 14
queries will silently mix tracks. Mitigation deferred — adding `track="main"`
filter to all 14 sites would be defensive but invasive churn under current
single-track regime.

---

## Real action items (post-audit)

| ID | Severity | Recommendation | When |
|---|---|---|---|
| 1 | LOW | Add composite index on `decision_logs(tab_type, sector_name, decision_date)` | Defer until n_decisions > 5,000 rows |
| 2 | COSMETIC | Cleanup `pending_approvals` legacy column (DB has 1 extra) | Optional |
| 3 | NA | Audit script's BacktestMetrics probe needs corrected field names | Audit script only |
| 4 | LATENT | Add `track == "main"` filter to 14 `SimulatedPosition.snapshot_date.desc()` query sites | Defer until track_b/c re-introduced |
| 5 | FIXED | Reduced `command_center` NAV cache from 60s → 5s | Done 2026-05-04 |

**No production-grade bugs detected.**

---

## What this audit tells us

The project's **3-way schema alignment** (ORM declaration / migration ALTER list / live SQLite columns) is intact across 10 tables, after multiple sprints of additions (S2 reflection / S3 pre-registration / P-FUND performance reporting). The **2 newest writers** (CashFlow + PortfolioNavSnapshot) honor the sign and three-NAV-state conventions exactly as specified.

The **S3 forward-registration → EFFECTIVE_N_TRIALS automatic update → DSR threshold tightening** chain is end-to-end consistent (43 grid + 1 P-FUND forward = 44 effective). HARKing R1-R4 detection has 0 active flags because no spec drift exists.

The **cold-start edge case** (most fragile P-FUND-2 path) handles prior external cash flows correctly — a deposit before the first NAV snapshot accumulates into nav_open at first roll-up, not lost.

The single legitimate medium finding (historical NULL spec_hash) is by-design per spec §1.3 ("不回填历史 spec 的'原版' hash") — historical rows aren't retro-tagged because the registry can't reconstruct what hash they "would have had" had pre-registration existed at the time.

**Net**: project data layer is in shape consistent with the capability claims. No urgent fixes required.
