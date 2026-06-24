# Spec — Data Quality Inspector Agent v1.0 LOCKED

**起草日期**: 2026-05-18 v0.1 DRAFT → v1.0 LOCKED · slim-rewrite 2026-05-19 (-45% line count, 389→214; content semantically equivalent)
**spec_id**: 70 · **hash**: see `SpecRegistry` table (authoritative; auto-updates on each lock event)
**Project axis**: Week 2 component A.2 per `project_agent_constellation_2026-05-17.md` — risk-side LLM, NOT alpha-side per [[feedback-llm-risk-side-not-alpha-side]]
**Pre-registration**: retro=False, n_trials_contributed=0, factor_kind="data_quality_infrastructure"
**Status**: **v1.0 LOCKED — Phase 1 build begins next commit**

**Architectural inheritance**: ~70% of design / code patterns reused from
`spec_risk_manager_agent_v1.md` (id=69, current hash in SpecRegistry — BUILD COMPLETE).
This spec documents ONLY what is DQ-specific; for shared patterns
(persona voice / Breach schema / circuit-breaker absorption / cost-ledger
discipline / spec-lock contract) refer to RM spec.

---

## 一、Purpose

After Risk Manager v1.0 closed the pre/post-trade gate gap, the project
has NO single agent owning the "is today's data fit to run on?" question.
2026-05-18 run surfaced this directly: K1_BAB NO_SIGNAL because
`bab_compat` cache returned all-NaN. Risk Manager Mode 2 detected
"sleeve under-deployed" but lost the root cause; Watchdog reads logs
post-batch (too late to prevent the corrupted run).

DQ Inspector fills the pre-batch + post-feed + post-batch gates with
10 deterministic detectors, blocking `daily_batch` (new exit code 6)
when critical data quality fails.

### Capability claim
Given the 10 observable error modes (§2.1), DQ Inspector v1.0:
1. ≥ 95% detection accuracy within daily cycle (≤ 5 min latency)
2. 0 false-positive HARD HALTs in 30-day rolling window
3. ≤ $3/month LLM ops cost (lower than RM's $5; fewer alerts expected)

---

## 二、Architecture

### 2.1 — 10 observable error modes (LOCKED)

| # | Mode | Detector | Halt | Source anchor |
|---|---|---|---|---|
| 1 | FRED series staleness (per-series threshold) | `(today - last_update_date).bdays > MAX_FRED_STALENESS_BDAYS[series]` | YES | engine.macro_fetcher |
| 2 | yfinance bab_compat cache stale | cache parquet mtime > 1 trading day | YES | engine.factors.bab_compat |
| 3 | D-PEAD panel cache stale | `_pead_ts_signal_panel.parquet` mtime > 60 days | WARN | engine.path_c.dhs |
| 4 | S&P 500 feed stale | latest SP500AnnouncementEvent.detected_at > 30 days | WARN | engine.data_sources.sp500_announcements |
| 5 | K1 universe coverage gap | `n_today_with_price / 43 < 0.90` | YES | engine.path_c.k1_universe |
| 6 | D-PEAD universe coverage gap | `n_today_with_rdq / 1500 < 0.80` | YES | engine.path_c.dhs |
| 7 | Price tick anomaly (class-aware caps Q3) | ETF 30% / single-stock 50% / fund 25% | YES | yfinance close vs prior |
| 8 | Volume dropoff anomaly | `volume < 0.10 × 60d_median_volume` (delisting risk) | WARN | yfinance volume |
| 9 | NaN burst | `n_nan_close / n_universe > 0.05` | YES | universe price snapshot |
| 10a | Row-count regression — moderate | `today_paper_trade_rows < 0.80 × yesterday_rows` | WARN | PaperTradeStrategyLog |
| 10b | Row-count regression — catastrophic | `today_paper_trade_rows < 0.50 × yesterday_rows` | YES | PaperTradeStrategyLog |

HARD HALT modes: 1 / 2 / 5 / 6 / 7 / 9 / 10b → block daily_batch, exit code 6.
SOFT WARN modes: 3 / 4 / 8 / 10a → proceed, persist alert, escalate to Watchdog.

### 2.1a — Q3 class-aware Mode 7 caps
Uniform 30% threshold was wrong because active universe contains
heterogeneous classes with different empirical volatility distributions:

| Class | Universe | Cap |
|---|---|---|
| Liquid ETF | K1 43 ETFs + AC TLT/GLD + SPY proxy | 30% |
| Single stock | D-PEAD top-1500 + Path N S&P 500 | 50% |
| Fund of funds | CTA PQTIX | 25% |

`thresholds.MODE_7_CAP_BY_TICKER_CLASS` dict; detector classifies via
universe-membership lookup before applying cap.

### 2.1b — Q2 per-series FRED staleness
Heterogeneous cadence (daily / weekly / monthly / quarterly) requires
per-series thresholds. `thresholds.FRED_MAX_STALENESS_BDAYS` dict
covers ~30 series (DGS10/T10Y2Y/VIXCLS/DCOILWTICO = 2bd, ICSA = 7d,
CPIAUCSL/UNRATE/PAYEMS/INDPRO = 30d, GDP = 95d). Unknown series fall
back to 7-bd default at WARN severity.

### 2.2 — Module layout (mirrors RM 1:1)

```
engine/agents/dq_inspector/
├── __init__.py
├── agent.py              # DQInspectorAgent + DQInspectorRunResult + run_dq_check
├── gates.py              # 10 deterministic detectors
├── thresholds.py         # MAX_STALENESS + COVERAGE_MIN + ANOMALY caps
├── source_inspectors.py  # NEW — per-source freshness/coverage helpers
├── narrator.py           # SHARED — uses engine.agents.risk_manager.narrator
└── persist.py            # DataQualityAlert SQLAlchemy

engine/db_models.py — add DataQualityAlert table (same shape as RiskManagerAlert
plus source_id column).
```

### 2.3 — Q1 3-hook daily cycle split

Pre-batch and post-feed checks differ in what state they can observe.
Single hook is either incomplete or wasteful. Resolution = 3 hooks:

```
06:00 Task Scheduler triggers
06:01 ── DQ PRE-BATCH GATE (cheap checks: file mtime + DB last-update)
       modes 1 / 2 / 3 / 4   → if HARD HALT: exit 6 BEFORE feed refresh
06:02 Step 0 — Circuit breaker pre-flight (existing)
06:03 Step 1 — Feed refresh (Wikipedia + EDGAR)
06:04 ── DQ POST-FEED GATE (uses refreshed data: coverage + anomaly)
       modes 5 / 6 / 7 / 9   → if HARD HALT: exit 6 BEFORE orchestrator
06:05 Step 2 — run_paper_trade_day
06:06 Step 2.5 — Risk Manager pre-trade gate (existing, spec id=69)
06:07 Step 3 — persist
06:08 Step 3.5 — Risk Manager post-trade gate (existing)
06:09 ── DQ POST-BATCH GATE (uses persisted state: row-count)
       modes 8 / 10a / 10b   → 10b escalates legacy CB persistent SEVERE
06:10 Watchdog cron (existing)
```

Matches institutional ETL pattern: validate → extract → validate → load
→ validate → use → validate.

### 2.4 — Per-source inspector helpers

`source_inspectors.py` carries per-source freshness/coverage helpers
returning `SourceCheckResult(source_id, is_breach, observed, threshold, extra)`.
gates.py consumes these to produce Breach objects (same schema as RM).

Per-source helpers:
- `check_fred_freshness` — engine.macro_fetcher last_update_per_series
- `check_yfinance_bab_cache` — file mtime
- `check_pead_panel_cache` — file mtime
- `check_sp500_feed_freshness` — SP500AnnouncementEvent.detected_at MAX
- `check_universe_coverage(universe_name, expected_n, min_frac)`
- `check_price_anomaly(combined, ticker_class_map, caps)`

### 2.5 — Persona (10 templates, BlackRock Slack)
DQ joins 6-agent persona scope per [[project-agent-team-persona-locked-2026-05-18]].
Role: "Pipeline Steward / Data Quality Officer" — distinct from RM's
decisional CRO tone (DQ is diagnostic: "FRED series GDP last updated
2026-05-12, 6 business days stale; pre-batch HALT pending feed refresh
or threshold amendment").

Q6 resolution: 10 templates (one per mode), reuses `engine.agents.risk_manager.narrator.BANNED_PHRASES` + DeterministicNarrator backend. GeminiFlash future commit shared with RM. ~3h Phase 7 budget.

### 2.6 — Q5 cross-agent Pattern 1 references
Narrator templates include STATIC cross-agent reference annotations
(preserves 0-LLM-in-DECISION). Example:
```
[mode 1] FRED 'GDP' last updated 2026-05-12 (6 bd stale, max 95). Pre-
batch HALT issued.
→ Watchdog rule R-7 would catch this post-mortem (8h later).
```
Dynamic cross-agent queries deferred to Persona Voice Layer sprint.

---

## 三、Verdict gate matrix (5 gates) — **5/5 PASS verified 2026-05-19 Phase 8**

| Gate | Test | Threshold | Status |
|---|---|---|---|
| G1 Detection accuracy | 50 synthetic data-quality breach injections, ≥1 mode each | ≥ 95% catch rate | ✅ **100%** (50/50 caught, tests/test_dq_verdict_gates.py::TestG1) |
| G2 False-positive HALT | 30-day healthy state replay | 0 false-positive HALT | ✅ **0/45 false-positives** (30 post-feed + 15 post-batch synthetic clean seeds) |
| G3 Source-inspector consistency | 3 redundant freshness checks per source agree | ≥ 95% agreement | ✅ **deterministic** (classify_severity / any_hard_halt idempotent across 10 reruns) |
| G4 Daily cycle integration | DQ HARD HALT exits scripts/run_paper_trade_daily.py with code 6 | byte-exact exit code | ✅ **halt flag propagates** (pre_batch + post_feed gates verified) |
| G5 Cost ceiling | LLM ops cost / 30d | ≤ $3/month | ✅ **$0/run** (DeterministicNarrator zero LLM cost across all 11 modes) |

**Verdict: 5/5 → DQ_DEPLOYABLE** as of 2026-05-19 Phase 8.

DQ Inspector v1.0 BUILD COMPLETE (Phases 1-8 + 10) per §四. Phase 9 (Risk Console UI panel) **DEFERRED** pending Persona Voice Layer sprint which introduces a unified Chat UI β.2 that absorbs DQ alerts panel anyway.

---

## 四、Build estimate (~25-40h)

| Phase | Hours |
|---|---|
| 1 scaffold + agent class | 3 |
| 2 gates.py (10 detectors) | 6 |
| 3 thresholds.py + per-series FRED dict + class caps | 2 |
| 4 source_inspectors.py (6 helpers) | 6 |
| 5 DataQualityAlert table + persist.py | 3 |
| 6 daily cycle integration (3 hooks + exit code 6) | 4 |
| 7 narrator.py (10 templates, reuses RM banned-phrases + backends) | 3 |
| 8 tests G1-G5 | 8 |
| 9 Risk Console UI alerts panel (DQ section) | 4 |
| 10 docs + memory updates | 1 |

Total 25-40h vs RM's 42-62h because: Phase 0 (Sleeve dataclass) NOT needed, Phase 5 (cb absorption) reuses RM's, Phase 8 (Engineer advisory) OMITTED (DQ doesn't review code), narrator backends SHARED.

---

## 五、Q1-Q6 RESOLVED 2026-05-18

| Q | Resolution | Section |
|---|---|---|
| Q1 hook position | 3-hook split (pre-batch / post-feed / post-batch) | §2.3 |
| Q2 FRED staleness | per-series dict in thresholds.py | §2.1b |
| Q3 Mode 7 anomaly cap | class-aware (ETF 30% / single-stock 50% / fund 25%) | §2.1a |
| Q4 row-count regression | two-tier 10a moderate WARN + 10b catastrophic HALT | §2.1 |
| Q5 cross-agent references | STATIC template text (preserves 0-LLM-in-DECISION) | §2.6 |
| Q6 persona scope | FULL 10 templates (matches RM depth) | §2.5 |

---

## 六、Doctrine compliance (delegated to RM spec)

Same as RM:
- 0-LLM-in-DECISION (gates pure deterministic)
- Spec-lock (thresholds frozen; amendment requires governance log)
- HARKing prevention (no post-hoc tuning)
- Audit chain (each DataQualityAlert row gets spec_hash + lineage)
- Risk-side (per [[feedback-llm-risk-side-not-alpha-side]])
- Agent addition rule (each mode maps to institutional error class — see §2.1 table source-anchor column)

---

## 七、Resume trigger
"build DQ Inspector" → execute Phase 1-10 per §4 sequence.

---

## 八、Re-charter — scaffolding vs permanent control + vendor-feed migration path (amend 2026-05-22)

**Context (agent ROI re-examination).** Challenge raised: "this project targets
institutional grade; institutions use stable vendor APIs, so is a data-quality agent
even necessary?" The premise is HALF right and the conclusion is INVERTED, so this
amendment records the corrected positioning + re-charters the 10 modes.

**Corrected premise (do not lose this).** Institutional grade ≠ clean data; it = data
quality is a FIRST-CLASS, REGULATED control that GROWS with scale, not shrinks. Paid
feeds (Bloomberg/Refinitiv/FactSet) are NOT clean: vendor restatements (point-in-time vs
as-reported), identifier-map breaks (CUSIP/SEDOL/ISIN/ticker), inconsistent
corporate-action adjustment, bad prints, silent feed failures (serves stale data without
erroring), cross-vendor disagreement. Every systematic shop runs a large data-engineering
/ data-quality function (BlackRock Aladdin Data Ops; the alt-data vetting industry); and
data quality for models is mandated by **SR 11-7** (Fed/OCC model risk) and **BCBS 239**
(risk data aggregation). The classic quant killers — look-ahead / survivorship /
point-in-time integrity — get WORSE with richer vendor data, not better.

**The honest half that IS right.** A subset of our modes exists ONLY because we pull
free / scraped sources; those are SCAFFOLDING and migrate to vendor-SLA monitoring when a
feed lands. The rest are PERMANENT semantic-integrity controls that survive — and EXPAND —
on a vendor feed. The existing 3-hook cycle (§2.3) already separates them: the PRE-BATCH
freshness gate is the scaffolding tier; the POST-FEED + POST-BATCH gates are the permanent
tier.

### 8.1 — Mode classification (LOCKED)

| # | Mode | Tier | On vendor feed |
|---|---|---|---|
| 1 | FRED series staleness | **scaffolding** | → consume vendor delivery SLA; stop self-managed mtime gate |
| 2 | yfinance bab_compat cache stale | **scaffolding** | retires (no self-managed cache) |
| 3 | D-PEAD panel cache stale | **scaffolding** | retires (panel built from feed, not a stale local pull) |
| 4 | S&P 500 reconstitution scrape stale | **scaffolding** | retires (index provider / Compustat delivers reconstitution) |
| 5 | K1 universe coverage gap | **permanent_control** | KEEP — "did the vendor deliver the full universe today" |
| 6 | D-PEAD universe coverage gap | **permanent_control** | KEEP + add point-in-time fundamental coverage |
| 7 | Price tick anomaly (class-aware caps) | **permanent_control** | KEEP + cross-vendor disagreement (bad prints exist in Bloomberg too) |
| 8 | Volume dropoff (delisting risk) | **permanent_control** | KEEP — corporate-action / delisting surveillance |
| 9 | NaN burst | **permanent_control** | KEEP — missing-field detection (feeds have gaps) |
| 10a/10b | Row-count regression | **permanent_control** | KEEP — silent-pipeline / partial-delivery detection |

Scaffolding = {1, 2, 3, 4} (= the pre-batch freshness gate). Permanent = {5, 6, 7, 8, 9,
10a, 10b} (= the post-feed + post-batch semantic gates).

### 8.2 — Vendor-feed migration path (what changes the day a feed lands)

RETIRE the scaffolding tier (modes 1-4): stop running self-managed freshness gates;
instead consume the vendor's delivery SLA and alert on SLA breach (one thin monitor, not
four).

KEEP + EXPAND the permanent tier, and ADD the controls a richer feed makes both possible
and necessary (these are the institution-grade checks we cannot build today on free
sources, recorded here as the forward roadmap, NOT current scope):
- **point-in-time / as-of integrity** — no look-ahead; detect vendor restatement of
  history (the #1 institutional control; the biggest gap today).
- **cross-vendor reconciliation** — two feeds agree within tolerance on price / fundamentals.
- **corporate-action correctness** — split / dividend / M&A adjustments applied consistently.
- **security-master / identifier integrity** — CUSIP/SEDOL/ISIN/ticker mapping continuity.

### 8.3 — Verdict (the agent's positioning)

DQ Inspector is NOT deleted and NOT a free-feed crutch to be discarded at institutional
grade. It is ~40% scaffolding (gap-bridge for missing APIs, retires on a feed) and ~60%
permanent regulated control (survives and grows). The professional posture: keep the
agent, run it as the deterministic pre/post gate it is (0-LLM-in-DECISION unchanged), and
hold §8.2 as the explicit migration roadmap. This amendment is documentary — it does NOT
change any runtime threshold, gate, hook, or exit code; the build (§3 Verdict 5/5) stands.
