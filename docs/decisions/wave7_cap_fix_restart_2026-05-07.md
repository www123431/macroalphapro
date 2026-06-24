# Wave 7 — Cap-Fix Restart (2026-05-07)

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — Cap-fix restart applied 2026-05-07 |
| Trigger | Supervisor 2026-05-07 noticed 32-position display vs spec MAX_LONG=10 |
| Outcome | Pre-2026-05-04 simulated_positions / simulated_trades flagged `era='pre_cap_fix_legacy'`; clean rebalance written for 2026-05-07; live tracking V2 inception set |

## Diagnosis

**Symptom**: `pages/live_dashboard.py` HOLDINGS tab showed 32 positions, exceeding spec
`engine/portfolio.py:193-215` MAX_LONG=8 (default) / 10 (risk-on regime), MAX_SHORT_EQUITY=6.

**Root cause** (reproduced via `scripts/_diag_wave7_cap_reproduce.py`):

- Current `construct_portfolio` enforces cap correctly. Today's run for
  2026-05-07 produced 14 positions (7L + 7S in transition regime; later
  10L + 6S in risk-on regime), both within spec.
- Pre-2026-05-04 snapshots (notably 2026-04-25 with 36 positions: 34L + 2S)
  were written under a buggy code revision before the **EFA Three-Piece
  Strategy Uplift was REJECTED** (memory entry
  `project_efa_uplift_reject_2026-05-03`). During the uplift period the
  asset-class-conditional short cap path apparently bypassed `MAX_LONG` for
  the long side; revert restored single-pass `MAX_SHORT=6` but legacy
  snapshots were never cleaned.

**Code verdict**: NO patch required. Cap is enforced today.
**Data verdict**: 55 legacy positions + 75 legacy trades flagged + isolated
via `era` column.

## Fix sequence (8 steps applied 2026-05-07)

| # | Step | Outcome |
|---|---|---|
| 1 | `ALTER TABLE simulated_positions / simulated_trades ADD era VARCHAR(32) DEFAULT 'live'` | schema migration |
| 2 | UPDATE legacy rows SET era='pre_cap_fix_legacy' WHERE date < 2026-05-04 | 55 + 75 rows flagged |
| 2.5 | `get_current_positions(era='live')` default filter + ORM era column on SimulatedPosition / SimulatedTrade | callers see only live data by default |
| 3 | `execute_rebalance(2026-05-07, dry_run=False)` | clean 16-position snapshot (10L + 6S risk-on, cap warnings self-reported) |
| 4 | SystemConfig `inception_date_v2 = 2026-05-07` + `wave7_cap_fix_restart_at` + rationale | inception V2 timestamp |
| 5 | `pages/portfolio_journey.py` adds inception V2 caption + `include_legacy` checkbox (default off) | UI surfaces clean by default; audit toggle for transparency |
| 6 | This decision doc + project_report v11 entry | documentation |
| 7 | `amend_spec` ledger entry: `kind=clarification, reason=Wave 7 cap-fix restart...` | pre-registration discipline |
| 8 | Integration gate (Tier R critical / pytest 109 / 22-page smoke) | gate |

## What survives (audit-trail integrity)

- DecisionLog hash chain — UNAFFECTED (positions / trades not in chain payload)
- spec_registry rows — preserved with new amendment
- B++ Mass FDR / S1 multi-window backtest results — preserved (frozen by spec_hash)
- Falsification chain (7 reject + 1 marginal) — preserved
- Legacy 55 positions + 75 trades — preserved with era flag (NOT deleted)

## What changes for the supervisor

- Live UI (live_dashboard / portfolio_journey / risk_console) defaults to era='live' → only sees post-cap-fix data
- Backtest pages unchanged (backtest doesn't touch simulated_positions)
- Performance numbers from post-2026-05-07 onward are clean & spec-compliant
- Pre-V2 audit trail still accessible via `include_legacy=True` checkbox on portfolio_journey

## Reproducibility

Audit script: `scripts/_diag_wave7_cap_reproduce.py` reruns construct_portfolio
on today's signals; outputs ≤ 14 confirms cap enforcement.

DB backup: `macro_alpha_memory.db.before_wave7` (pre-fix snapshot, 3.9MB).

## Reference

- `engine/portfolio.py:191-215` — MAX_LONG / MAX_SHORT / regime-conditional caps
- `engine/portfolio_tracker.py:94-180` — get_current_positions with era filter
- `engine/db_models.py:516-520` — SimulatedPosition.era column
- `pages/portfolio_journey.py` — V2 inception caption + legacy toggle
- Memory: `project_efa_uplift_reject_2026-05-03` (root cause of buggy code revision)
