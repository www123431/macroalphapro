# Spec — Operations Watchdog Agent v1.0

**起草日期**: 2026-05-12 (v0.1 DRAFT → v1.0 after project-scan audit same day)
**Project axis**: per `project_reframe_quant_alpha_agentic_ops_2026-05-09.md` (agentic AI 辅线 — operations layer, never alpha)
**Pre-registration**: retro=False, n_trials_contributed=0, factor_kind="infrastructure_spec" (P-LAB exempt → +0 trials)
**Status**: **v1.0 LOCK CANDIDATE — pending user review before register_spec + implementation**

**Companions**:
- `spec_quant_co_pilot_decision_lineage_v1.md` (id=53, Tool 1 ReAct lineage — primitives reused here)
- `feedback_agent_addition_rule.md` (new agent must justify specific error modes eliminated — 12 modes enumerated below)
- `feedback_llm_component_removal_test_governance.md` (6mo removal test discipline)
- `feedback_dont_over_lock_in_spec.md` (lock-discipline 4-step test)

---

## v0.1 → v1.0 重大变更 (post-scan)

| Axis | v0.1 estimate | v1.0 (post-audit) |
|---|---|---|
| Scope | ~1,800 LoC, 12-13 days | **~900 LoC, 6.5-7 days** |
| Approach | Parallel detector + new persistence | **Thin LLM reasoner on top of existing Auto-Audit; new persistence reused** |
| New tables | `watchdog_log` | **None** — reuse `AuditFinding` + `AuditProposal` |
| Severity scheme | NONE/LOW/MED/HIGH (new) | **Reuse `engine.circuit_breaker` 4-level: NONE/LIGHT/MEDIUM/SEVERE** |
| Detection rules | 13 new tools + custom logic | **7 new Auto-Audit rules + read AuditFinding** |
| ReAct primitives | New | **Reuse `engine.quant_co_pilot.base.run_react_agent`** |

**Driver of changes**: Project scan revealed `engine.auto_audit_rules` already has 29 deterministic rules covering 5/12 error modes; `engine.circuit_breaker` already has 4-level severity + persistence; `AuditFinding/AuditProposal/PendingApproval` schemas cover persistence needs; `Tool 1` provides ReAct primitives. Watchdog v1.0 is repositioned as a thin **aggregator + reasoner + notification escalator + auto-repair executor**, not a from-scratch detection layer.

---

## 一、Purpose + Hypothesis

### Purpose

After 2026-05-12 production state (B++ retroactive PASS + K1 production swap + D-COMBINED single-stock PASS + Windows Task Scheduler daily 06:00 SGT registered), the project entered a **24mo forward paper-trade validation phase** with:
- 55 active ETFs in production universe (post-K1 swap)
- Daily automated batch (Task Scheduler, no Streamlit dependency)
- Multi-sleeve infrastructure (etf_l1 + ss_sp500 sleeves, 2026-05-10 new)
- 24mo OOS verdict gate for K1 + D-COMBINED (2028-05)

**Critical risk**: any silent failure during 24mo forward = lost OOS validation data = project deliverable delayed (must restart 24mo from failure point).

Current notification gap (verified pre-spec):
- `cycle_states` table: status persisted but no notification
- `AuditFinding` table: 29 rules write findings but no user-facing escalation
- `engine.circuit_breaker`: exists but only triggered by VIX/quota events, not pipeline-health events
- Task Scheduler MacroAlphaPro_DailyBatch (registered today): zero post-completion notification

**Watchdog v1.0 fills the notification + reasoning gap** by:
1. Running 7 NEW Auto-Audit rules for previously-uncovered error modes
2. Aggregating ALL AuditFinding rows (old + new) into cross-domain ReAct reasoning
3. Escalating to existing `circuit_breaker` severity tiers with user-facing notifications (Windows toast, email, dashboard widget)
4. Executing 5 deterministic auto-repair recipes (hardcoded mapping, NOT LLM-decided)

### Hypothesis (capability claim, NOT statistical)

Given the 12 production error modes (§2.1), Operations Watchdog v1.0:
1. Detects ≥ 80% of fail-mode occurrences within 1 daily cycle (≤ 1-day latency)
2. Auto-repairs **3** deterministic infrastructure failure types without human input (modes 1/2/6; modes 4/10/12 reclassified as detect-only per amendment 3 root-cause doctrine — see §2.5)
3. Escalates 7 decision-requiring failure types via Windows toast + email + dashboard widget
4. Achieves ≥ 0 false-positive SEVERE alerts in 30-day rolling window
5. Total LLM cost ≤ $5/month (1 daily ReAct call × $0.05 × 30 days)
6. Removal test pathway: 6mo grace, then "拆掉 LLM 用户体验更糟吗" honest review

### Distinction from existing components

| | Auto-Audit Loop (rules) | Tool 1 Lineage | ETF Holdings Monitor | circuit_breaker | **Watchdog v1.0** |
|---|---|---|---|---|---|
| Role | Rule-based hygiene | Lineage Q&A | Risk overlay on holdings | VIX/quota gate | **Cross-domain ops aggregator + LLM reasoner + notification + auto-repair** |
| Trigger | Weekly cron | User query | Monthly | VIX spike / LLM quota | **Daily 06:10 SGT (10min post Task Scheduler batch)** |
| Output | AuditFinding rows (silent) | Direct answer | Counterfactual log | Halt flag (silent) | **Severity tier → silent/dashboard/toast/email + auto-repair execution** |
| LLM use | Reviewer (failing removal test) | ReAct Q&A | Holdings screen | None | **Read-only ReAct over AuditFinding + cycle state** |
| Auto-repair | None | None | None | None | **5 deterministic recipes (hardcoded)** |

---

## 二、Architecture

### 2.1 12 Error Modes Watched (LOCKED at v1.0)

#### Operations layer (7 modes)

| # | Mode | Existing Auto-Audit rule | Watchdog v1.0 action | Auto-repair? |
|---|---|---|---|---|
| 1 | Cycle silently failed mid-batch | None | **NEW rule** `rule_cycle_state_completion` | ✅ Recipe `repair_retry_idempotent_batch` |
| 2 | yfinance stale data | Partial: `rule_universe_drift_vs_registered` | **NEW rule** `rule_universe_data_freshness_per_ticker` | ✅ Recipe `repair_force_fresh_fetch` |
| 3 | Delisted/restructured ETF | Partial: `rule_universe_drift_vs_registered` | Augment existing rule (PR-style) | ❌ Decision (deactivate/replace) |
| 4 | Sleeve drift (wrong sleeve_id) | ✅ `rule_sleeve_id_integrity` (full) | Read AuditFinding | ❌ **Detect-only (amendment 3 2026-05-13)**: sleeve_id NULL signals write-path bug; auto-backfill would mask root cause |
| 5 | Massive weight_delta unexplained | Partial: `rule_anomaly_screener_m1_drift` | **NEW rule** `rule_weight_delta_p99_unexplained` | ❌ Decision (legit flip vs spike) |
| 6 | Trade execution missing | None | **NEW rule** `rule_signal_trade_referential_integrity` | ✅ Recipe `repair_retry_execution_if_signal_active` |
| 7 | NAV anomaly unexplained | None | **NEW rule** `rule_nav_move_vs_rebalance_audit` | ❌ Decision (real move vs error) |

#### Trading layer (5 modes)

| # | Mode | Existing Auto-Audit rule | Watchdog v1.0 action | Auto-repair? |
|---|---|---|---|---|
| 8 | Signal computation NaN | None | **NEW rule** `rule_signal_panel_nan_scan` | ❌ Decision (signal bug) |
| 9 | TC drag computed wrong | Partial: `rule_backtest_vs_production_param_alignment` | **NEW rule** `rule_realized_tc_vs_spec_rate` — **MUST read each strategy's own spec-locked tc_bps; NOT hardcoded threshold** (amendment 2 2026-05-12) | ✅ Halt next batch (do not auto-correct) |
| 10 | Weight cap not enforced | Partial: `rule_etf_holdings_cap_clamp_bounds` | **NEW rule** `rule_max_position_weight_vs_cap` (covers all sleeves) | ❌ **Detect-only (amendment 3 2026-05-13)**: cap violation signals `construct_portfolio` cap-check bug; auto-truncate would mask root cause |
| 11 | Rebalance cadence drift | None | **NEW rule** `rule_rebalance_frequency_audit` | ❌ Decision (config bug) |
| 12 | REGIME_SCALE not applied | None | **NEW rule** `rule_regime_scale_vs_exposure_audit` | ❌ **Detect-only (amendment 3 2026-05-13)**: scale bypass signals portfolio construction bug; auto-reapply would mask root cause |
| **13** | **Watchdog daily LLM cost runaway** (amendment 1, 2026-05-12) | None | **NEW rule** `rule_watchdog_daily_cost_budget` | ✅ Halt next Watchdog run (not auto-correct cost) |

**Net new rules added to `engine.auto_audit_rules`**: **11 rules** (10 modes 1-12 minus mode 3 augment minus mode 4 reuse = 10 production-error rules, PLUS 1 meta-monitoring rule for mode 13 self-cost).
**Existing rules reused**: 2 (sleeve_id_integrity full for mode 4, others partial / augmented for mode 3).

**Amendment 2 (2026-05-12 evening, post Path E TC fix lesson)**: Mode 9 `rule_realized_tc_vs_spec_rate` implementation MUST read **each strategy's own `tc_bps_per_event` from its locked spec**, NOT hardcode global threshold. Different strategies have different correct TC:

- Path E v1+amendment 1: 4 bp/event (ETF tier-1, per `feedback_etf_tc_tier_model.md`)
- B++ T1 / K1: 8 bp historical (B++ Mass FDR spec id=44, K1 spec id=61)
- Path D (single-stock): 30 bp roundtrip (60-day post-rdq)
- Future ETF Path F+: tier-specific per standing rule (Tier 1 0.5-1.5bp / Tier 2 1.5-3bp / Tier 3 2-5bp)

Implementation contract:
```python
def rule_realized_tc_vs_spec_rate():
    for strategy in active_strategies:
        spec = read_spec(strategy.spec_id)
        locked_tc = spec.metadata['tc_bps_per_event']  # PER-SPEC, NOT hardcoded
        realized_tc = compute_realized_tc(strategy.recent_trades)
        if abs(realized_tc - locked_tc) / locked_tc > 0.5:
            yield AuditFinding(severity='HIGH', mode='mode_9_tc_drag_wrong', ...)
```

If implemented with hardcoded 8bp threshold → would false-positive on Path E (4bp), miss B++ if it drifts to 12bp etc. **Per-spec read is the correct implementation**.

Amendment kind: clarification (+0 trials). Cross-ref: `feedback_etf_tc_tier_model.md`, Path E spec id=64 amendment 1.

---

**Mode 13 detail (amendment 1, 2026-05-12 clarification, +0 trials per `feedback_amendment_trial_cost_retired`)**:

Gap identified post-spec-v1.0 lock: existing infrastructure has `rule_llm_cumulative_cost_budget` in WEEKLY_RULES (weekly cumulative check) and `engine.llm_budget` per-call enforcement, but **no daily-cadence check**. Worst case: Watchdog prompt template bug causes infinite ReAct loop → burns $50/day → weekly rule detects 7 days later after $350 spent. Mode 13 closes this gap.

Rule logic:
```python
def rule_watchdog_daily_cost_budget() -> RuleResult:
    """Read engine.llm_cost_ledger for component='ops_watchdog' rows for today.
    If sum(cost_usd) > $0.50 (2.5x expected $0.20 daily budget): SEVERE.
    Triggers halt_next_watchdog_run flag (separate from halt_next_batch flag).
    """
```

Threshold rationale: expected $0.20/day = 8 ReAct steps × $0.02. Cap $0.50 = 2.5x buffer covers occasional reasoning depth excursion but catches runaway (10-100x). $0.50/day × 30 = $15/month vs $5/month expected — generous but not unlimited.

### 2.2 Severity Tiers (REUSE `engine.circuit_breaker` 4-level)

```python
# Reuse engine.circuit_breaker constants:
LEVEL_NONE   = "none"     # All green; silent log
LEVEL_LIGHT  = "light"    # Single non-critical anomaly; dashboard widget yellow
LEVEL_MEDIUM = "medium"   # Multiple findings OR 1 ops failure; Windows toast + dashboard
LEVEL_SEVERE = "severe"   # Critical failure OR repair failed; toast persist + email + halt_next_batch flag
```

Mapping mode → severity is **hardcoded in `engine.agents.ops_watchdog.triage`** (NOT LLM-decided):

```python
MODE_SEVERITY_MAP_LOCKED = {
    "mode_1_cycle_failed":                   "medium",  # auto-repair attempt; severe only after 3 retries fail
    "mode_2_yfinance_stale":                 "medium",  # auto-repair attempt
    "mode_3_etf_delisted":                   "severe",  # decision required + halt
    "mode_4_sleeve_drift":                   "light",   # often cosmetic; backfillable
    "mode_5_weight_delta_unexplained":       "severe",  # potential data error
    "mode_6_trade_execution_missing":        "medium",  # auto-repair attempt
    "mode_7_nav_anomaly":                    "severe",  # always escalate
    "mode_8_signal_nan":                     "severe",  # signal logic bug suspicion
    "mode_9_tc_drag_wrong":                  "severe",  # halt next batch
    "mode_10_weight_cap_violation":          "medium",  # auto-truncate
    "mode_11_cadence_drift":                 "severe",  # config bug
    "mode_12_regime_scale_misapplied":       "medium",  # auto-reapply
    "mode_13_watchdog_cost_runaway":         "severe",  # amendment 1: halt next watchdog run
}
```

LLM contributes context (which modes co-occur, historical baseline comparison) but NEVER changes mapping. Adding a mode requires spec_amend.

### 2.3 ReAct flow (5-8 steps, REUSE `Tool 1` primitives)

```
┌──────────────────────────────────────────────────────────────┐
│ TRIGGER: 06:10 SGT (10min after Task Scheduler daily batch)  │
│   New independent Task Scheduler entry (separate from        │
│   MacroAlphaPro_DailyBatch). Idempotent — safe to retry.     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 0 — Pre-flight: trigger ALL 7 new + existing Auto-Audit │
│   rules. Writes to AuditFinding table (existing path).       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 1 — Watchdog ReAct agent invoked                        │
│   Reuse engine.quant_co_pilot.base.run_react_agent(          │
│     query=watchdog_prompt,                                   │
│     tool_dispatcher=watchdog_tool_dispatcher,                │
│     tool_descriptions=10 tools listed below,                 │
│     max_steps=8, cost_budget_usd=0.20)                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Steps 2-7 — LLM ReAct reasoning over:                        │
│   - read_audit_findings(today, severity_filter=*)            │
│   - read_cycle_state(today)                                  │
│   - read_trade_log(today)                                    │
│   - read_nav_change(today)                                   │
│   - read_signal_quality(today)                               │
│   - read_historical_baseline(metric, lookback=60d)           │
│   - read_pending_approvals(today)                            │
│   - get_circuit_breaker_state()                              │
│   - search_amendments_for_recent_spec(spec_id)               │
│   - read_capability_evidence(spec_id) [from Tool 1]          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 8 — Triage + dispatch                                   │
│   1. For each AuditFinding: lookup severity per LOCKED map  │
│   2. Aggregate to highest severity                            │
│   3. If mode in AUTO_REPAIR_RECIPES: execute recipe          │
│      - Recipe is hardcoded Python function                   │
│      - LLM does NOT decide whether to execute                │
│      - Max 3 retries per recipe, then escalate               │
│   4. Emit notification per severity:                         │
│      LEVEL_LIGHT: dashboard widget update                    │
│      LEVEL_MEDIUM: + Windows toast (win10toast)              │
│      LEVEL_SEVERE: + email (if SMTP configured) + set       │
│                    circuit_breaker.json halt_next_batch=True │
└──────────────────────────────────────────────────────────────┘
```

LLM cost cap: 8 ReAct steps × $0.02 = $0.16 per run. Daily budget $0.20. Monthly $5 hard cap (1 trigger/day × 30 days = $4.8 expected).

### 2.4 Read-only tools (10 tools — LOCKED at v1.0)

5 NEW + 5 REUSED from Tool 1's `engine.quant_co_pilot.tools`:

```python
WATCHDOG_TOOLS_LOCKED = {
    # NEW (5)
    "read_audit_findings":     watchdog.tools.read_audit_findings,     # AuditFinding table
    "read_cycle_state":        watchdog.tools.read_cycle_state,        # CycleState table
    "read_trade_log":          watchdog.tools.read_trade_log,          # SimulatedTrade table
    "read_nav_change":         watchdog.tools.read_nav_change,         # PortfolioNavSnapshot
    "read_historical_baseline": watchdog.tools.read_historical_baseline, # Percentile lookup

    # REUSED from Tool 1 (5)
    "read_spec_registry":           qcp_tools.read_spec_registry,
    "search_amendments":            qcp_tools.search_amendments,
    "read_capability_evidence":     qcp_tools.read_capability_evidence,
    "read_memory_file":             qcp_tools.read_memory_file,
    "read_verdict_json":            qcp_tools.read_verdict_json,
}
```

All tools are **read-only**. The only writes Watchdog performs are:
- AuditFinding rows (via existing auto_audit run path; standard write)
- Auto-repair recipe execution (deterministic, separate from LLM reasoning)
- Notification side-effects (toast, email, dashboard widget render)

No writes to: `simulated_positions`, `simulated_trades`, `portfolio_nav_snapshots`, `universe_etfs`. **Production state is read-only from Watchdog's perspective**.

### 2.5 Auto-repair recipes (LOCKED hardcoded mapping — NOT LLM-decided)

```python
# engine/agents/ops_watchdog/auto_repair.py — Phase 3 Option A scope (amendment 3 2026-05-13)
AUTO_REPAIR_RECIPES_LOCKED = {
    # ── Active (3 recipes) — transient/idempotent failures fixable via re-run
    "mode_1_cycle_failed":                  _repair_retry_idempotent_batch,
    "mode_2_yfinance_stale":                _repair_force_fresh_fetch,
    "mode_6_trade_execution_missing":       _repair_retry_execution_if_signal_active,
    # ── Deferred stubs (3 modes) — root-cause WRITE-PATH bugs; auto-fix would mask
    #     bug. Stubs return (False, deferred=True) → orchestrator escalates to human.
    "mode_4_sleeve_drift":                  _stub_deferred,
    "mode_10_weight_cap_violation":         _stub_deferred,
    "mode_12_regime_scale_misapplied":      _stub_deferred,
}

# Modes NOT in this table never get auto-repaired; always escalate.
# Adding a mode (or promoting a deferred mode to active) requires spec amendment
# with HARKing R1-R4 review.
```

**Amendment 3 (2026-05-13)**: Scope narrowed from 6 active recipes → 3. Modes
4/10/12 reclassified as detect-only stubs because the underlying violation
patterns (sleeve_id NULL / weight > cap / regime scale bypassed) signal
production write-path bugs. Auto-fixing the symptom would mask the bug and
let it recur. Escalation to human review (via PendingApproval / dashboard
banner) preserves the root-cause-fix discipline. Kind: `clarification`, +0
trials. Cross-ref: `feedback_amendment_trial_cost_retired.md`.

Each recipe:
1. Max 3 retry attempts
2. Each attempt logged to AuditProposal table (existing infrastructure)
3. Success → write AuditFinding.status='RESOLVED' + Watchdog log
4. After 3 retries fail → escalate to SEVERE + email + halt flag

LLM cannot bypass mapping. LLM detection of "this looks like mode X" triggers recipe X only if X is in table. Anything else escalates.

### 2.6 Notification dispatcher

```python
# engine/agents/ops_watchdog/notifications.py
def emit_notification(severity: str, summary: str, findings: list[AuditFinding]):
    if severity in ("light", "medium", "severe"):
        write_dashboard_widget_state(severity, summary, findings)
    if severity in ("medium", "severe"):
        send_windows_toast(summary, duration_seconds=10 if severity == "medium" else 30)
    if severity == "severe":
        send_email_if_configured(summary, findings_detail)
        set_circuit_breaker_halt_flag(reason=summary)
```

`win10toast` library to be added to `requirements.txt`. Email via stdlib `smtplib` (SMTP config in `.streamlit/secrets.toml`, optional).

Halt flag clears only via dashboard button (`pages/circuit_breaker.py` extends with "Acknowledge Watchdog Halt" button). Auto-clear forbidden (defeats safety purpose).

---

## 三、Pre-Registered Eval (descriptive, NOT statistical)

### 3.1 Wave 0 retroactive baseline

Cannot retroactively eval — Watchdog reacts to ongoing state. Wave 0 = first 30 days post-launch.

### 3.2 Eval metrics (locked, descriptive)

**After 30 days**:
- `n_watchdog_runs` (target ≥ 25 of 30; allow 5 PC-off days)
- `n_findings_per_run` (descriptive distribution; histogram in capability evidence)
- `n_auto_repairs_executed` (descriptive)
- `auto_repair_success_rate` (target ≥ 80%)
- `false_positive_severe_rate` (target 0)
- `mean_cost_per_run` (target ≤ $0.20)
- `notification_user_action_rate` (target descriptive — how often user acts on toast)

**After 6 months**:
- removal test: "拆掉 Watchdog LLM 用户体验变差吗?"
  - Criterion 1: Watchdog caught ≥ 3 incidents that would otherwise be silent → PASS
  - Criterion 2: auto-repair saved ≥ 5 human interventions → PASS
  - Either criterion → keep; both fail → KILL per `feedback_llm_component_removal_test_governance.md`

### 3.3 PASS criteria (locked v1.0)

After 30 days:
- All targets met → PASS_PRELIMINARY
- `false_positive_severe_rate` > 0.1 → PARTIAL (signal-to-noise tuning needed)
- `auto_repair_success_rate` < 0.6 → PARTIAL (recipes need refinement)

After 6 months:
- removal test PASS → PASS (capability validated)
- removal test FAIL → KILL Watchdog; revert to AuditFinding-only state

---

## 四、Implementation Contract

### 4.1 New module structure (~900 LoC estimate, post-audit; major reuse)

```
engine/agents/ops_watchdog/
  __init__.py                 # ~30 LoC
  agent.py                    # ~150 LoC, ReAct orchestrator (wraps Tool 1 base)
  tools.py                    # ~250 LoC, 5 NEW tools (other 5 from Tool 1)
  triage.py                   # ~100 LoC, severity classification (uses LOCKED map)
  auto_repair.py              # ~200 LoC, 6 deterministic recipes
  notifications.py            # ~120 LoC, dashboard + toast + email + halt flag
  prompt.py                   # ~50 LoC, ReAct prompt template

engine/auto_audit_rules.py    # patch: add 7 NEW rules (~250 LoC inline)

engine/circuit_breaker.py     # patch: add "watchdog_halt" trigger source (~30 LoC)

pages/live_dashboard.py       # patch: add "Operations Health" section (~80 LoC)
pages/circuit_breaker.py      # patch: add "Acknowledge Watchdog Halt" button (~40 LoC)

docs/
  spec_ops_watchdog_agent_v1.md  # THIS spec

tests/
  test_ops_watchdog_agent.py      # ~250 LoC, unit + integration + mock LLM
  test_auto_audit_rules_new.py    # ~150 LoC, 7 new rules unit tests

scripts/
  run_ops_watchdog.py             # ~50 LoC, CLI entry

requirements.txt                  # patch: add `win10toast`
```

**Total NEW LoC**: ~1,000 (~900 production + ~400 test). Down from v0.1 estimate ~1,800.

### 4.2 Reuse Summary

| Component | Source | Reuse type |
|---|---|---|
| ReAct skeleton + LLM caller | `engine.quant_co_pilot.base.run_react_agent` | Function call |
| 5 read tools (spec / amendments / capability / memory / verdict) | `engine.quant_co_pilot.tools` | Function call |
| Severity 4-level | `engine.circuit_breaker.LEVEL_*` | Constants |
| Halt flag persistence | `engine.circuit_breaker` + `.streamlit/circuit_breaker.json` | API extension |
| Finding persistence | `engine.auto_audit_models.AuditFinding` | Schema + write |
| Auto-repair plan persistence | `engine.auto_audit_models.AuditProposal` | Schema + write |
| Human gate | `engine.memory.PendingApproval` | Schema + write |
| Auto-Audit run framework | `engine.auto_audit_rules` | Add 7 rules to existing framework |
| LLM budget enforcement | `engine.llm_budget` | API call |

### 4.3 Tier R Audit Hooks (NEW rules for Watchdog's own integrity)

4 meta-monitoring rules added to `engine.auto_audit_rules` (3 original + 1 from amendment 1):

- `rule_watchdog_runs_daily`: assert Watchdog ran in last 24h (uses AuditFinding query for Watchdog's own findings)
- `rule_watchdog_halt_flag_not_stuck`: assert halt_next_batch flag not stuck True > 7 days
- `rule_watchdog_auto_repair_audit_trail`: assert every auto-repair logged with timestamp + recipe_name + outcome to AuditProposal
- `rule_watchdog_daily_cost_budget` (amendment 1, mode 13): assert Watchdog's own daily LLM cost ≤ $0.50; SEVERE + halt_next_watchdog_run if exceeded

---

## 五、Validation Gates (pre-launch acceptance)

- **Gate 1**: 7 new Auto-Audit rules unit tests pass (≥ 35 cases; 5 per rule typical)
- **Gate 2**: Watchdog agent module unit tests pass (≥ 25 cases including 5 auto-repair recipes)
- **Gate 3**: Integration test: simulate cycle failure → 7 rules fire → Watchdog ReAct reasons → triggers auto-repair → success → emits MEDIUM toast
- **Gate 4**: Integration test: simulate weight cap violation → rule fires → Watchdog auto-truncates → audit log written → emits MEDIUM
- **Gate 5**: Real LLM dogfood (mirror Tool 1 ship process): 3 representative scenarios, agent reasoning verified correct, cost ≤ $0.20
- **Gate 6**: Dashboard widget renders correctly with 4 mock states (NONE/LIGHT/MEDIUM/SEVERE)
- **Gate 7**: All existing 234+ path_c + Tool 1 + auto_audit tests still green (no regression)
- **Gate 8**: Manual end-to-end: register Task Scheduler entry "MacroAlphaPro_Watchdog" at 06:10 SGT → wait 24h → verify Watchdog ran + AuditFinding rows persisted + dashboard widget populated
- **Gate 9**: Cost actually ≤ $0.20/run measured over first 7 daily runs
- **Gate 10**: `win10toast` library installation verified; toast renders on Windows 10 Home China

---

## 六、Forbidden Modifications (HARKing R1-R4 + project-axis invariants)

Hash-locked at v1.0 register:

- **13 error modes**: exact list above (12 original + mode 13 cost runaway from amendment 1); adding modes requires spec_amend (HARKing R2)
- **Auto-repair recipe table**: 3 active modes (1/2/6) post-amendment-3 2026-05-13; modes 4/10/12 are deferred detect-only stubs (root-cause doctrine); adding new modes OR promoting deferred → active requires spec_amend
- **Severity mapping (MODE_SEVERITY_MAP_LOCKED)**: hardcoded per mode; not LLM-decided
- **Notification channels**: 4 channels (dashboard / toast / email / halt flag); new channels require spec_amend
- **Trigger cadence**: 06:10 SGT daily; not parametrized by LLM
- **LLM budget cap**: $0.20/run × 30 days = $6 monthly hard ceiling
- **Zero production-write capability**: Watchdog NEVER writes to portfolio / simulated_positions / simulated_trades / portfolio_nav_snapshots / universe_etfs tables; only to AuditFinding / AuditProposal / Watchdog's own log
- **Halt flag**: only Watchdog SEVERE severity can SET; only human can CLEAR via dashboard button
- **LLM does NOT decide auto-repair execution**: recipe mapping is hardcoded; LLM detection only triggers recipe lookup, not bypass
- **0 LLM in alpha/risk decision loop preserved**: Watchdog is operations layer

---

## 七、Out of Scope

- LLM-driven ETF deactivation decisions (mode 3 always escalates)
- LLM auto-modifying portfolio weights (modes 5/7 always escalate)
- LLM tuning REGIME_SCALE value (mode 12 only re-applies existing scale, never overrides)
- Watchdog-on-Watchdog meta-monitoring (avoid infinite loop; covered by `rule_watchdog_runs_daily` instead)
- Per-user customization (single-user project; no multi-tenant)
- Real-time streaming (daily cadence sufficient; intraday monitoring out of scope v1.0)
- Auto-rotation of WRDS / yfinance credentials (separate secrets-rotation workflow)

---

## 八、Reproducibility

```bash
# Manual run (debug + dry-run)
py -3.11 -m engine.agents.ops_watchdog --verbose --dry-run

# Production cron (Windows Task Scheduler entry, new task separate from MacroAlphaPro_DailyBatch)
$action = New-ScheduledTaskAction -Execute "C:\Windows\py.exe" `
    -Argument "-3.11 -m engine.agents.ops_watchdog --check" `
    -WorkingDirectory "${REPO_ROOT}\Desktop\intern"
$trigger = New-ScheduledTaskTrigger -Daily -At "06:10"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "MacroAlphaPro_Watchdog" -Action $action -Trigger $trigger -Settings $settings -Description "Operations Watchdog Agent daily check (10min after daily_batch). Spec id=63."

# Outputs
#   AuditFinding rows (existing table)
#   AuditProposal rows (auto-repair recipes)
#   data/ops_watchdog/{YYYY-MM-DD}_run.json  (Watchdog ReAct trace, optional)
#   engine/state/circuit_breaker.json (halt flag — actual path; spec wording
#     ".streamlit/circuit_breaker.json" was stale and corrected in amendment 3
#     2026-05-13 to match engine.circuit_breaker._STATE_FILE)
```

---

## 九、Decision Tree (capability output)

Per Watchdog run, output one of:

| Severity | Trigger | Action |
|---|---|---|
| NONE | All 12 checks green, no auto-repair triggered | Watchdog ReAct trace log, exit silently |
| LIGHT | 1 non-critical anomaly (LOW-severity AuditFinding) | Dashboard widget yellow + audit_log |
| MEDIUM | 2+ findings OR 1 ops failure (modes 1/2/4/6/10/12) OR auto-repair succeeded after retry | Windows toast + dashboard orange + audit_log |
| SEVERE | Critical failure (modes 3/5/7/8/9/11) OR auto-repair failed after 3 retries OR multiple MEDIUM concurrent | Windows toast persist + email (if SMTP) + halt_next_batch flag + dashboard red + audit_log |

---

## 十、文档元信息

| Field | Value |
|---|---|
| **Current version** | **v1.0 + amendment 1** (locked 2026-05-12; amendment 1 same day post-lock added mode 13 daily-cost meta-monitoring per gap surfaced by user review) |
| 起草日期 | 2026-05-12 |
| Spec id | (assigned at register_spec, expected id=63) |
| Project axis | quant alpha + agentic ops; 0 LLM in alpha decision loop |
| Project state at draft | 60+ specs / 4 PASS / 1 MARGINAL / 11 FAIL / Tool 1 SHIPPED / Tool 2/3/4 spec-locked / Watchdog v1.0 candidate |
| Hard locks (HARKing R1-R4) | 12 error modes / 6 auto-repair recipes / 4 notification channels / 06:10 trigger / $6/mo budget / 0 production-write |
| Implementation iterative (unlocked) | LLM prompt wording / dashboard widget styling / auto-repair retry counts (start 3) / win10toast vs plyer choice |
| Calendar gates | 30-day descriptive eval (2026-06-12 cut) / 6-month removal test (2026-11-12 cut, before 2026-11-09 LLM-component removal test 大限) |
| Estimated LoC (post-audit) | ~900 production + ~400 test = ~1,300 total |
| Estimated sprint (post-audit) | ~6.5-7 days (down from v0.1 12-13 days estimate after reuse mapping) |
| Reuse % | ~50% (Tool 1 ReAct base + circuit_breaker + AuditFinding/Proposal + auto_audit framework) |
| Academic anchors | None — pure operations infrastructure; not a hypothesis test |
| Pre-launch audit complete | ✅ project scan 2026-05-12 identified 29 existing auto_audit rules, 4-level severity in circuit_breaker, Tool 1 ReAct base, AuditFinding/Proposal schemas — all integrated as reuse mapping in v1.0 |
