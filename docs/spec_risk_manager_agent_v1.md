# Spec — Risk Manager Agent v1.0 LOCKED + BUILD COMPLETE Phases 0-11

**起草日期**: 2026-05-18 (v0.1 DRAFT → v1.0 LOCK CANDIDATE → LOCKED → Phases 0-11 same day);
**§2.1a amend** 2026-05-19 (Q1 → Q1a + Q1b two-tier; shadow-phase trigger).
**spec_id**: 69
**spec_hash**: stored in `SpecRegistry` table — call `engine.agents.persona.tools.lookup_spec(69)` (or `engine.preregistration.list_specs()`) for the current git-blob hash + amendment log. Literal hash strings are not pinned in this header anymore: they create a fixed-point bug (writing a hash into a file changes the file's hash). Historical amendments are visible in the SQL `amendment_log` column.
**Project axis**: per `project_agent_constellation_2026-05-17.md` Week 2 component A.1 (risk-side LLM, NOT alpha-side per [[feedback-llm-risk-side-not-alpha-side]])
**Pre-registration**: retro=False, n_trials_contributed=0, factor_kind="risk_management_infrastructure" (P-LAB exempt → +0 trials)
**Status**: **v1.0 BUILD COMPLETE 2026-05-18 — Phases 0-11 of §6 sequence all delivered same session. Production-ready pending one deferred item:** GeminiFlashNarrator LLM integration (DeterministicNarrator default works without LLM at zero cost).

## Build delivery summary (Phases 0-11 commits)

| Phase | Commit | Deliverable |
|---|---|---|
| 0  | c43307a | Sleeve.sleeve_class field + SleeveClass enum + LOCKED_SLEEVES test |
| 1  | 1b9e44f | engine.agents.risk_manager scaffold + RiskManagerRunResult |
| 2  | eef1d78 | gates.py — 12 deterministic detectors (4 senior upgrades) |
| 3  | 4d97a90 | thresholds.py — RISK_THRESHOLDS singleton + SLEEVE_CLASS_CAPS (retired 2026-05-19) |
| 4  | e1f9341 | RiskManagerAlert SQLAlchemy + persist.py idempotent UPSERT |
| 5  | e714782 | cb_absorption.py — strangler-fig + read/write separation |
| 6  | 9542fe7 | orchestrator_hook.py + scripts/run_paper_trade_daily.py wiring |
| 7  | b849208 | narrator.py — DeterministicNarrator + GeminiFlash stub + banned-phrases |
| 8  | 352716f | advisory.py — Engineer PR sign-off API (GREEN/YELLOW/RED) |
| 9  | d413a13 | tests/ — 213 RM tests across 8 files (G1-G5 verdict gates) |
| 10 | 112afda | pages/risk_console.py — Risk Manager alerts panel UI |
| 11 | (this)  | spec amendment marking BUILD COMPLETE + memory updates |

Total: ~3,300 lines of production code + 1,900 lines of tests, 301/301
passing (24 thresholds + 44 gates + 46 narrator + 26 advisory + 14
persist + 11 cb_absorption + 10 orchestrator + 38 verdict + 88 Week 1).

**Companions**:
- `feedback_agent_addition_rule.md` (10 observable error modes enumerated below)
- `feedback_llm_risk_side_not_alpha_side.md` (Sprint I empirical: risk-side LLM P(deployable) >> alpha-side; this spec aligns)
- `feedback_spec_lock_is_decision_contract_2026-05-15.md` (hash-lock after user review)
- `feedback_llm_component_removal_test_governance.md` (6mo removal test discipline)
- `project_week1_refactor_status_2026-05-18.md` Week 2 Option B decision
- `spec_ops_watchdog_agent_v1.md` (architectural sibling — same persona/Audit Recorder/LLM reasoner pattern)

---

## 一、Purpose + Hypothesis

### Purpose

After Week 1 Strategy/Sleeve refactor (commits 8742a10 → d1a243b, 2026-05-18), the project has:
- 5 production strategies registered through `engine.strategies` ABC
- 4 sleeves (etf_l1 32.4% / ss_sp500 48.6% / cta_defensive 9.0% / rms_crisis_hedge 10.0%) with 1.5× leverage
- Daily paper-trade orchestrator writing PaperTradeStrategyLog + PaperTradeTradeLog (Sprint H)
- Existing `engine.circuit_breaker` (VIX/quota gate, 4-level severity NONE/LIGHT/MEDIUM/SEVERE)
- Existing `engine.risk_metrics` (VaR/CVaR three-method, factor exposures)

**Critical risk-side gap**: there is NO pre-trade or post-trade guardrail enforcing position-size / sleeve-weight / aggregate-leverage limits. Specifically:

1. Engineer agent (Level 2.5, Week 2.5) will author new strategy adapters. **No mechanical gate** prevents a buggy strategy from emitting `weights={"X": 5.0}` (500% concentration) and propagating through the orchestrator's `combine_sleeve_weights` to corrupt the combined book.
2. **No post-aggregation alert** when book HHI / single-ticker weight / sleeve-weight drifts outside Tier-3-locked envelopes.
3. **VaR/CVaR monitoring is read-only** in Risk Console — no decisioning surface that can halt the daily batch on breach.
4. **circuit_breaker is fragmented**: VIX/quota live in `engine.circuit_breaker`, position-limit lives nowhere, sleeve-drift lives nowhere. Engineer / Anomaly Sentinel / Devil's Advocate have no single counterparty to consult on "is this trade safe to merge."

**Risk Manager Agent v1.0 fills this gap** by:
1. Running **deterministic pre-trade gates** before `run_paper_trade_day` commits the combined portfolio
2. Running **deterministic post-trade gates** after persistence (VaR/CVaR breach / sleeve drift / position concentration)
3. Absorbing the existing `engine.circuit_breaker` into a unified Risk Manager state machine (no duplicate severity classes)
4. Providing an **LLM-narration layer** (read-only) over breaches — generates one-paragraph plain-English summary attached to alerts. **DOES NOT decide whether to halt** — that remains the deterministic gate's job (0-LLM-in-DECISION preserved).
5. Exposing a `risk_manager.advisory(diff)` API consumed by Engineer agent PR flow

### Hypothesis (capability claim, NOT statistical)

Given the 10 production error modes (§2.1), Risk Manager v1.0:
1. Detects ≥ 95% of breach occurrences within the daily orchestrator cycle (≤ 1 day latency)
2. Blocks (returns `halt=True`) on all deterministic-rule breaches; **never** allows daily batch to commit a breach state
3. Generates LLM narrative for breach explanation at ≤ $0.05 per breach event; bounded by $5/month operational cap
4. Achieves ≥ 0 false-positive HALT alerts in 30-day rolling window (HALT must always be true breach)
5. Removal test pathway: 6mo grace, then "拆掉 LLM 用户体验更糟吗" honest review per `feedback_llm_component_removal_test_governance.md`. **Deterministic gates remain even if LLM layer is removed.**

### Distinction from existing components

| | circuit_breaker | risk_metrics | Auto-Audit | Watchdog v1.0 | **Risk Manager v1.0** |
|---|---|---|---|---|---|
| Trigger | VIX / quota | On-demand (Risk Console) | Weekly cron | Daily post-batch | **Pre-trade + Post-trade in same daily cycle** |
| Severity | 4-level | None | Findings only | 4-level (reuses CB) | **4-level (absorbs CB)** |
| Decision authority | Halt VIX trade only | None (read-only) | None | None (alerts only) | **Hard halt on deterministic gate breach** |
| LLM use | None | None | Reviewer (deprecated) | ReAct over findings | **Read-only narration over breaches** |
| Pre-trade hook | No | No | No | No | **Yes — `risk_manager.check_proposed_book(combined)` before persist** |
| Engineer integration | No | No | No | No | **Yes — `risk_manager.advisory(diff)` returns 🟢/🟡/🔴 sign-off** |

---

## 二、Architecture

### 2.1 10 Production Error Modes Watched (LOCKED at v1.0)

Each mode has: deterministic detector + halt threshold + LLM-narration trigger.

| # | Mode | Detector (deterministic) | Halt | Narrate |
|---|---|---|---|---|
| 1a | Book-level single-ticker absolute cap (operational/issuer risk) | `\|combined[t]\| > BOOK_SINGLE_TICKER_ABS_CAP` (§2.1a Q1a) | YES | YES |
| 1b | Per-strategy intra-strategy ticker cap (concentration risk) | `\|sig.weights[t]\| > SLEEVE_CLASS_INTRA_CAPS[cls]` (§2.1a Q1b) | YES | YES |
| 2 | Sleeve drift > 10% RELATIVE to target | `\|eff - target\| / target > 0.10` | NO (warn) | YES |
| 3 | Gross leverage > 1.6× (Tier-3 cap 1.5+10pp band) | `combined.abs().sum() > 1.6` | YES | YES |
| 4 | Net exposure outside [-50%, +150%] | `combined.sum() not in [-0.5, +1.5]` | YES | YES |
| 5 | HHI > 0.25 (Tier-3 concentration cap) | `(combined**2).sum() > 0.25` | YES | YES |
| 6 | 1-day VaR-95 < -3% NAV | `var_95_historical < -0.03` | NO (warn) | YES |
| 6b | 1-day VaR-95 < -9% NAV (model-integrity breach) | `var_95_historical < -0.09` | **YES** | YES |
| 7 | 1-day ES-95 < -5% NAV | `es_95_historical < -0.05` | NO (warn) | YES |
| 7b | 1-day ES-95 < -15% NAV (model-integrity breach) | `es_95_historical < -0.15` | **YES** | YES |
| 8 | Short-side aggregate > 50% of gross | `\|neg(combined)\| / gross > 0.50` | NO (warn) | YES |
| 9 | Number of OK strategies < `n_required` (default 3 of 5) | `sum(s.status=='OK') < n_required` | YES (cb-cascade) | YES |
| 10 | Cross-cancel ticker count > 5 (multiple strats long & short same name) | `count_cross_short > 5` | NO (warn) | YES |

Modes 1a / 1b / 3 / 4 / 5 / 6b / 7b / 9 → HARD HALT (rejected book, no persistence). Modes 2 / 6 / 7 / 8 / 10 → SOFT WARN (book persisted, flagged in dashboard).

**Note on mode 6/7**: VaR/ES alerts are SOFT at the project threshold because the metric is counterfactual (yfinance 2y × current weights) — model dispersion (G3) routinely 25-50% in stress. Mode 6b / 7b add a HARD HALT layer at 3× threshold: when VaR/ES blow through 3× the warn level, the model itself is in distress (Q4 senior review).

### 2.1a Two-tier single-ticker caps (Q1a + Q1b resolution)

**History.** Initial Q1 resolution (2026-05-18) collapsed two distinct risk concepts (operational/issuer risk vs. strategy concentration risk) into one cap (`_SLEEVE_CLASS_CAPS` lookup with conservative-min on cross-sleeve overlap). Shadow-phase 2026-05-19 surfaced the composition flaw: K1 BAB's 45-ETF universe includes GLD/TLT (legitimate low-β candidates), and AC TLT/GLD's insurance sleeve also holds them at 50/50 by design. Combined book GLD ≈ 7.5% (K1's β-neutralized 2.5% + AC's designed 5%) hit the etf_l1 5% conservative cap → false-positive HARD_HALT every day.

The fix isn't a special case — the abstraction was wrong. **Q1 is amended to Q1a + Q1b**: two independent gates defending two independent risks, matching the BlackRock Aladdin / AQR / Bridgewater PARC two-tier pattern.

#### Q1a — Mode 1a: book-level absolute single-ticker cap (operational risk)

| Constant | Value |
|---|---|
| `BOOK_SINGLE_TICKER_ABS_CAP` | **25%** |

Defends against issuer-specific blowups (single ETF delisting, tracking error, counterparty failure). Threshold applies UNIFORMLY across all sleeves and ticker types — Aladdin "single-name exposure limit" standard. Operates on `combined: pd.Series` (book weights post-sleeve-weighting); requires no signal or registry lookup.

```python
def gate_mode_1a_book_abs_cap(combined: pd.Series) -> list[Breach]:
    cap = BOOK_SINGLE_TICKER_ABS_CAP
    return [Breach(mode_id="1a", severity="HARD_HALT", ...)
            for ticker, w in combined.items() if abs(w) > cap]
```

#### Q1b — Mode 1b: per-strategy intra-strategy ticker cap (concentration risk)

Defends against any single strategy over-leaning on one ticker within its own gross. Evaluates `signal.weights` (strategy's intra-strategy weights, summing to ~1.0 absolute), NOT post-orchestration book weights. Per-sleeve-class because strategies have different universe sizes.

| `SleeveClass` | Intra Cap | Rationale |
|---|---|---|
| `ALPHA_EQUITY_LS` | **15%** | BAB tertile single-ETF typical 7-12%; 15% covers β-neutralized edge cases |
| `ALPHA_SINGLE_STOCK` | **5%** | 1500-name universe; no single name should exceed 5% intra |
| `INSURANCE` | **50%** | AC TLT/GLD 50/50 by design |
| `CTA_OVERLAY` | **100%** | single-fund overlay (PQTIX = 100% intra-sleeve) |

```python
def gate_mode_1b_intra_sleeve_cap(signals: list[StrategySignal], registry) -> list[Breach]:
    breaches: list[Breach] = []
    for sig in signals:
        if sig.status != "OK":
            continue
        cap = SLEEVE_CLASS_INTRA_CAPS[registry.get_sleeve(sig.sleeve_id).sleeve_class]
        for ticker, w in sig.weights.items():
            if abs(w) > cap:
                breaches.append(Breach(mode_id="1b", severity="HARD_HALT", ...))
    return breaches
```

#### Why two gates beat one

Cross-strategy aggregation = Mode 1a's job. Within-strategy concentration = Mode 1b's job. The two gates have no coordination — each does ONE thing and does it independently. Cross-sleeve ticker overlap (the original GLD/TLT case) stops being a special case:

- GLD book aggregate 7.5% < 25% ⇒ Mode 1a passes.
- K1 intra GLD ≈ 7.7% < 15% (equity_ls cap) ⇒ Mode 1b passes K1.
- AC intra GLD = 50% ≤ 50% (insurance cap) ⇒ Mode 1b passes AC.

No conservative-min, no exemption rule, no special-case branch. The combined book exposure is naturally bounded by `sum(sleeve_target × intra_weight)`, and the two caps defend the two independent risks.

#### Sleeve registrations (unchanged from Q1)

```python
Sleeve(sleeve_id="etf_l1",           sleeve_class="alpha_equity_ls",    ...)
Sleeve(sleeve_id="ss_sp500",         sleeve_class="alpha_single_stock", ...)
Sleeve(sleeve_id="cta_defensive",    sleeve_class="cta_overlay",        ...)
Sleeve(sleeve_id="rms_crisis_hedge", sleeve_class="insurance",          ...)
```

**Implementation cost** (vs. Q1 baseline): ~2h amend — `gates.py` split into two functions, `thresholds.py` swap, 12 new tests, narrator template split. `_SLEEVE_CLASS_CAPS` retired; `_conservative_cap_for_ticker` deleted (no longer needed).

### 2.2 Module layout

```
engine/agents/risk_manager/
├── __init__.py
├── base.py                  # RiskManagerAgent class (persona-shared with Watchdog)
├── gates.py                 # 10 deterministic detectors (pure functions)
├── thresholds.py            # Tier-3-locked thresholds (frozen dict; spec-hash protected)
├── narrator.py              # LLM narration layer (Gemini 2.5 Flash, ≤$0.05/breach)
├── advisory.py              # Engineer-PR sign-off API
└── persist.py               # Writes to RiskManagerAlert table (new)

engine/db_models.py — add:
class RiskManagerAlert(Base):
    id, date, mode_id, severity, halt_decision,
    deterministic_payload (JSON), narrative_text (str|None),
    affected_tickers (JSON), affected_sleeves (JSON),
    cost_usd (float), generated_at_utc, ...
```

**Absorbed from circuit_breaker**: `engine.circuit_breaker.evaluate` → `engine.agents.risk_manager.gates.evaluate_circuit_state`. Existing `cb_level` / `cb_reason` API preserved as compatibility shim.

**Not changed**: `engine.risk_metrics.*` (VaR/CVaR computation library) — Risk Manager IS a consumer of it.

### 2.3 Daily cycle integration

```
06:00 SGT Task Scheduler triggers MacroAlphaPro_DailyBatch
06:01 paper_trade_combined.run_paper_trade_day(today) builds combined book
06:02 ── PRE-TRADE GATE ──────────────────
       risk_manager.check_proposed_book(combined) → (halt: bool, alerts: list[RiskAlert])
       if halt: orchestrator skips persist_run_to_db, writes _HALT.json instead
06:03 if not halt: persist + attribution_logger writes 150+ trade rows
06:04 ── POST-TRADE GATE ─────────────────
       risk_manager.check_persisted_state(today) → soft warns (VaR/ES/HHI drift)
06:05 Watchdog reads RiskManagerAlert + AuditFinding rows
06:06 narrator.narrate(alerts) generates plain-English summaries (LLM, bounded $0.05/alert)
06:10 Notifications: dashboard widget + toast (severity ≥ MEDIUM) + email (SEVERE)
```

### 2.5 Persona layer (amendment 2026-05-18 evening)

Per [[project-agent-team-persona-locked-2026-05-18]] (user framing: "各个
agent 更像一个团队"), Risk Manager joins the persona-scope agent club.
**Persona attaches to narrative-output surfaces ONLY** — gates / persist /
advisory stay non-persona to preserve 0-LLM-in-DECISION.

Phase 7 narrator.py invariants:
- BlackRock Slack tone: terse, factual, active voice, no emoji
- Banned phrases regex enforced: `maybe / perhaps / could / might / probably`
- Temperature = 0.1 (low creativity, deterministic enough for risk comm)
- Post-generation check: any banned phrase → re-roll once → fallback to plain-factual template
- `PersonaContext` injection point: narrator.py accepts an optional
  PersonaContext parameter so the Persona Voice Layer sprint (37-52h
  future budget) can plug in character sheets without rewriting narrator

All 5 conversational layers (α/β/γ/δ/ε) approved:
- α plain narrative — Phase 7 of THIS build
- β multi-turn drilling — future Chat UI sprint (+10-15h, β.2 unified router)
- γ initiative — future Persona Voice Layer (Initiative trigger registry)
- δ cross-agent — future (Cross-agent event bus protocol)
- ε session memory — future (ResearchSessionMemory table)

### 2.4 Engineer-agent integration (Week 2.5 build dependency)

**Q2 resolution: ADVISORY ONLY** (no hard block).

Hard-block design rejected for two reasons:
1. Violates [[0-LLM-in-DECISION]]: hard-block would let LLM-derived risk model decide whether commits are allowed.
2. Contract violations (spec_hash drift, weight sum != 1, missing META) are ALREADY caught earlier by `tests/test_strategy_meta_locked.py` + `tests/test_strategy_registry.py` — those tests fail at Engineer's pytest step BEFORE Risk Manager ever sees the diff. Risk Manager's role is risk-surface advisory, not contract enforcement.

API:
```python
from engine.agents.risk_manager.advisory import sign_off

sign_off_result = sign_off(
    diff_text         = engineer_diff,
    affected_strategies = ["NEW_STRAT_X"],
    proposed_meta     = new_strategy_meta_dataclass,
)
# sign_off_result.verdict:
#   "GREEN"  if no breaches (no HARD HALT modes, no SOFT WARN modes)
#   "YELLOW" if only SOFT WARN modes triggered (2 / 6 / 7 / 8 / 10)
#   "RED"    if any HARD HALT mode triggered  (1 / 3 / 4 / 5 / 6b / 7b / 9)
# sign_off_result.reasons: [
#     "Mode 1: NEW_STRAT_X would push QQQ to 6.2% — cap is 5% for alpha_equity_ls sleeve",
#     "Mode 5: HHI of synthetic combined book = 0.31, exceeds 0.25 cap",
# ]
# sign_off_result.passing_modes: [...]   # modes the diff passes cleanly
# sign_off_result.cost_usd: 0.012        # LLM advisory cost (logged to ledger)
```

Verdict is advisory: user still does manual `git commit && git push`. The verdict + reasons appear in the Engineer agent's PR comment thread (Streamlit Engineer page) — user reads BOTH the diff and the advisory before committing.

GitHub branch-protection analogy: this is like a *reviewer comment*, not a *required status check*. Required status checks (the pytest suite from Slice 6) handle immutable rules. Risk Manager handles risk judgment.

---

## 三、Verdict Gate Matrix (5 gates)

| Gate | Test | Threshold | Method |
|---|---|---|---|
| **G1** Detection accuracy | Synthetic breach injection: 100 random portfolios with ≥1 mode breach each, detect all | ≥ 95% catch rate | unit test `test_risk_manager_synthetic_breach_recall.py` |
| **G2** False-positive HALT rate | Replay 30-day rolling window of actual paper trade history; count HALT alerts on healthy days | 0 false-positive HALT (true breaches OK) | integration test `test_risk_manager_replay_history.py` |
| **G3** VaR/CVaR cross-method agreement | For ≥ 95% of historical days, `abs(VaR_parametric - VaR_historical) / abs(VaR_historical) < 0.30` | ≤ 30% method dispersion (Q3 resolution: two-tier) | uses engine.risk_metrics |
| **G4** Circuit-breaker absorption parity | After absorption, `evaluate_circuit_state(today)` returns IDENTICAL severity to legacy `engine.circuit_breaker.evaluate(today)` for ≥ 90 historical days | 100% byte-identical | regression test against frozen baseline JSON |
| **G5** Cost ceiling | LLM ops cost over 30 rolling days | ≤ $5/month | `data/llm_cost_ledger.jsonl` aggregation |

**Q3 resolution — two-tier VaR/ES alerting** (deployment gate vs ops alerting separated):
- G3 deployment gate: 30% method-dispersion is `SAA_DEPLOYABLE`
- Runtime ops alert: 20% method-dispersion triggers `MODE_VARES_DISPERSION_WARN` (warn-narrate, no halt). User sees the gradient before it crosses the deployment threshold.

**Verdict matrix**:
- 5/5 PASS → SAA_DEPLOYABLE (production active)
- 4/5 PASS (G2 may fail in stress regime) → MARGINAL_DEPLOY (paper-trade only, monitor 30 days)
- ≤ 3/5 PASS → REJECT (do not deploy)

### 3.1 Q5 resolution — relative drift threshold for Mode 2

Original 2pp absolute drift was asymmetric across sleeve sizes:
- ss_sp500 0.486 target → 2pp = **4.1% relative drift** (lax)
- cta_defensive 0.09 target → 2pp = **22% relative drift** (way too lax for a small sleeve)

Per [[feedback-centralized-registry-pattern-2026-05-15]] and institutional convention (any multi-asset risk model uses relative drift), Mode 2 detector becomes:

```python
def gate_mode_2_sleeve_drift(sleeve_eff: dict, sleeve_target: dict) -> list[Breach]:
    breaches = []
    for sleeve_id, target in sleeve_target.items():
        if target == 0:
            continue                                    # skip zero-target sleeves
        eff = sleeve_eff.get(sleeve_id, 0.0)
        rel_drift = abs(eff - target) / target
        if rel_drift > 0.10:                            # 10% relative threshold (Q5)
            breaches.append(Breach(mode=2, sleeve=sleeve_id,
                                   rel_drift=rel_drift, abs_diff=abs(eff - target)))
    return breaches
```

Effective per-sleeve trigger thresholds at current Tier-3 allocation:

| Sleeve | Target | 10% rel-drift trigger (abs pp) |
|---|---|---|
| etf_l1 | 0.324 | 3.24pp |
| ss_sp500 | 0.486 | 4.86pp |
| cta_defensive | 0.09 | 0.9pp |
| rms_crisis_hedge | 0.10 | 1.0pp |

Small sleeves more sensitive (correct — they fluctuate more in absolute terms). Large sleeves more tolerant (correct — they have more natural variance). Anchor: `deployment_design.md` §3 "drift triggers shall be self-scaling with target size" (citation to be added when §3 is amendment-locked; for now spec § cite is forward-reference).

---

## 四、Cost ledger anchor

| Component | Per-event cost | Daily expected events | Monthly cap |
|---|---|---|---|
| Pre-trade gate (10 detectors, pure Python) | $0.00 | N/A | $0 |
| Post-trade gate (10 detectors) | $0.00 | N/A | $0 |
| Narrator (Gemini 2.5 Flash, ~500 tokens/breach) | $0.0001-0.0005 | 1-3 (typical), 10 (stress) | $0.30-1.50 |
| Engineer-PR advisory | $0.005-0.015 | 1-3/week | $0.30 |
| **Total budgeted** | | | **≤ $5/month** |

`data/llm_cost_ledger.jsonl` ALLOWED_AGENT_IDS gains: `risk_manager_narrator`, `risk_manager_advisory`. Bound at module init.

Hard cap enforcement: `engine.llm_cost_ledger.assert_under_cap("risk_manager", monthly_usd=5.0)` called before each LLM invocation. Above cap → skip LLM, persist alert with `narrative_text=None`, log `severity_escalated_by_cost_cap`.

---

## 五、Doctrine compliance

- ✅ **0-LLM-in-DECISION**: Gates 1-10 are pure deterministic functions in `gates.py`. LLM narrator runs AFTER halt/warn decision is made. LLM cannot flip a halt → no-halt.
- ✅ **Spec-lock**: `thresholds.py` is a frozen dataclass; mutating any threshold requires spec amendment + governance log entry.
- ✅ **HARKing prevention**: No post-hoc threshold tuning. Threshold change requires Tier-3 amendment row.
- ✅ **Audit chain**: Each RiskManagerAlert row gets `spec_hash_short` + `decision_lineage_id` linking back to gate + version.
- ✅ **Risk-side**: All 10 modes are RISK metrics, not alpha signals. P(PASS) prior 60-70% per [[feedback-llm-risk-side-not-alpha-side]].

---

## 六、Build effort estimate (~42-62h)

| Phase | Task | Hours |
|---|---|---|
| **0** | **Sleeve dataclass extension** — add `sleeve_class` field; update 4 Sleeve constructors in adapters.py; spec_class lockdown test similar to LOCKED_META | **2** |
| 1 | engine/agents/risk_manager/ scaffold + RiskManagerAgent class | 4 |
| 2 | gates.py — 12 deterministic detectors (10 modes + 6b + 7b model-integrity tiers) | 9 |
| 3 | thresholds.py — Tier-3-locked frozen dataclass + `BOOK_SINGLE_TICKER_ABS_CAP` + `SLEEVE_CLASS_INTRA_CAPS` (post §2.1a Q1a/Q1b amend) | 3 |
| 4 | RiskManagerAlert SQLAlchemy model + persist.py | 4 |
| 5 | circuit_breaker absorption (preserve API compatibility shim) | 6 |
| 6 | Daily cycle integration in paper_trade_combined.run_paper_trade_day | 4 |
| 7 | narrator.py — Gemini Flash LLM layer + cost ledger + cap enforcement | 6 |
| 8 | advisory.py — Engineer-agent PR sign-off API (advisory verdict + reasons) | 4 |
| 9 | Tests: G1 synthetic recall + G2 30d replay + G3 VaR agreement + G4 CB parity + G5 cost | 10 |
| 10 | Risk Console UI integration (alert feed + advisory verdict tile) | 6 |
| 11 | Documentation + memory updates | 2 |

**Range**: 42-62h. Phase 0 must commit BEFORE Phase 1-11 so Sleeve.sleeve_class is available when gates.py imports registry. Phase 0 is independent enough to be a separate commit + PR pair before the Risk Manager scaffold lands.

**Sequencing critical path**: 0 → 1 → 2 → 3 → 4 → 5 → 6 (orchestrator hook) | 7 / 8 / 10 parallel after 4 | 9 (tests) continuous | 11 last.

---

## 七、Q1-Q5 RESOLVED 2026-05-18 (per senior review)

| Q | Resolution | Section |
|---|---|---|
| Q1 single-ticker cap | **AMENDED 2026-05-19 → Q1a + Q1b two-tier** (see §2.1a). Initial Q1 sleeve-class-min approach produced false-positive on K1 × AC cross-sleeve GLD/TLT overlap. | §2.1a |
| Q1a issuer/operational risk | **Mode 1a — `BOOK_SINGLE_TICKER_ABS_CAP = 25%`** uniform across all sleeves (Aladdin single-name limit). | §2.1a |
| Q1b strategy concentration risk | **Mode 1b — `SLEEVE_CLASS_INTRA_CAPS`**: 15% / 5% / 50% / 100% for alpha_equity_ls / alpha_single_stock / insurance / cta_overlay; evaluated on intra-strategy `signal.weights`. | §2.1a |
| Q2 Engineer PR sign-off | **Advisory only** (no hard block) — contract violations caught earlier by Slice 6 lockdown tests | §2.4 |
| Q3 VaR cross-method threshold | **Two-tier**: G3 deployment gate at 30%, runtime warn-narrate at 20% | §3 |
| Q4 VaR/ES SOFT vs HARD | **SOFT at project threshold + HARD at 3× threshold** (modes 6b / 7b added for model-integrity breach) | §2.1 |
| Q5 Sleeve drift threshold | **Relative 10% drift** (not absolute 2pp); self-scaling across sleeve sizes | §3.1 |

---

## 八、Resume trigger

This spec is v1.0 LOCK CANDIDATE. To finalize:
1. Compute spec content hash (sha256 of §1-§6 locked sections, prefix-8)
2. Insert into spec_metadata table via `register_spec(...)` to obtain spec_id
3. Update MEMORY.md with locked spec_id + hash
4. Begin Phase 0 (Sleeve dataclass extension) then Phase 1 implementation per §6

If user signals "lock and start" → execute steps 1-4 atomically. If user signals "spec needs more work" → list remaining concerns and re-draft.
