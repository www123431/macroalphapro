# S6 — anomaly_screener Forward Test Pre-Registration (2026-05-05)

| Field | Value |
|---|---|
| Status | 🟢 PRE-REGISTERED — D4.0 of B-pragmatic-v2 sprint |
| Date | 2026-05-05 |
| Sibling docs | [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) (D1) · [hitl_architecture_audit_2026-05-05.md](hitl_architecture_audit_2026-05-05.md) (D2) |
| SpecRegistry entry | `s6.anomaly_screener.v1` (registered on D4.8) |
| Sprint | B-pragmatic-v2 D4 implementation |
| Forward window | 90 days from go-live; 60d hard mid-checkpoint + 90d final verdict |
| Committed budget | ~48 implementation hours + ≤ $250 LLM cost (Gemini 2.5 Flash, $380 ceiling) |
| Scope | **Slimmed-corrected** (2026-05-05): no sensitivity sweep, no calibration/Brier/ROC AUC, no 30d soft check, no live HARKing 4-rule detect (amendment ledger preserved). Recall + F1 retained (anomaly detection industry standard). Justification: master's project does not require oral defense; ETF-only universe + retail-tier data preclude top-journal target regardless; rigor traded for ~13h freed for thesis/portfolio/job package. Realistic publication target: SSRN preprint / arXiv q-fin / practitioner journals (JPM, FAJ, Quantitative Finance) — none require the cut metrics. |

---

## 0. Lakatos Notice

This spec is locked at the moment of SpecRegistry insertion. Any amendment requires
amend_spec() call with kind ∈ {clarification, threshold_tweak, hypothesis_amend,
endpoint_swap}. EFFECTIVE_N_TRIALS counter increments accordingly. Amendment
ledger records every change (append-only).

**Slimmed-corrected scope (2026-05-05)**: Live HARKing 4-rule auto-detect is
NOT enforced for this spec (single-supervisor project, low collusion risk).
Amendment ledger + spec_hash are sufficient for post-hoc audit. The 4-rule
detector remains available in `engine.preregistration.detect_harking()` for
manual / on-demand audit.

---

## 1. Research Hypotheses

### H_main
**LLM-augmented anomaly screener has marginal value over rule-based baseline
on portfolio anomaly detection, when measured by objective forward-event hit
rate (M1) and combined with supervisor judgement (M2).**

### H0 (null)
LLM detector hit rate ≤ rule-based baseline hit rate + 5pp on M1.
LLM does not contribute marginal value beyond simple price/concentration rules.

### H1 (alternative)
LLM detector hit rate > rule-based baseline hit rate + 5pp on M1, AND
supervisor acceptance rate (M2) ≥ 30%.

### Falsification Conditions

| Verdict | Decision rule | Action |
|---|---|---|
| **CLEAR_WIN** | M1: LLM > rule-based + 5pp **AND** M1 LLM > rule-based on each baseline (A=rule-only and B=rule+macro). M2: ≥ 30%. | Ship LLM; record first LLM-positive evidence in project. |
| **CLEAR_LOSS** | M1: LLM < rule-based - 5pp **OR** M1 LLM < rule-based against either baseline. | Reject LLM (8th falsification); ship rule-based. |
| **TIE / INCONCLUSIVE** | M1: \|LLM - rule\| ≤ 5pp on both baselines. | Don't ship LLM; ship rule-based; record as "inconclusive at n=N". |
| **CATASTROPHIC** | M1 precision both detectors < 30% AND supervisor M2 < 30% at 60d hard checkpoint. | Pull plug on entire S6; HITL retreats to 4 governance categories. |

This satisfies **Invariant 3** of D1: M1 (forward event verification) is supervisor-independent.

---

## 2. Pre-Registered Constants (ALL LOCKED)

### M1 Event Definition (price-driven, fallback default)

| Constant | Value | Reference |
|---|---|---|
| Event threshold (primary) | abs daily return > 2σ on 60-day rolling | Asness-Moskowitz-Pedersen 2013 conditional event study; standard quant practice |
| Forward window K | 5 trading days | Pre-FOMC drift literature (Lucca-Moench 2015) typical horizon |
| Universe scope | Current portfolio holdings + flagged tickers union | Restrict to relevant tickers for tractability |
| Cutoff filter | Events occurring ≥ 2026-01-15 (post LLM training cutoff) | D1 Invariant 2: avoid LLM lookahead leakage |

### M1 Reported Metrics (Slimmed-corrected, 2026-05-05)

The M1 evaluation reports three metrics, all deterministic:

| Metric | Definition | Why kept |
|---|---|---|
| **Precision** | (Flagged events that materialized) / (Total flagged events) | Verdict primary axis |
| **Recall** | (Flagged events that materialized) / (All universe events that occurred) | F1 requires it; anomaly detection industry standard (Aggarwal 2017 §1.4) |
| **F1** | 2 · precision · recall / (precision + recall) | Single summary stat for thesis tables; standard in anomaly detection literature |

**Removed** (Slimmed-corrected): Calibration plot, Brier score, ROC AUC. Rationale:
master's project without oral defense + LLM-self-reported confidence is known
poorly calibrated (Tian-Ye-Bowman 2023) so post-hoc calibration audit adds
limited evidence. ~5h freed.

The composite verdict (§1.4) uses **precision** as the comparison axis. Recall
is reported but not part of the binary win/loss criterion.

### M1 Sensitivity Sweep — REMOVED in Slimmed-corrected scope

Originally specified to run conditionally on ambiguous verdicts; cut as part of
the 2026-05-05 slim refactor. Pre-registered single point (2σ + K=5d) is the
sole basis for verdict. If post-hoc curiosity arises, sensitivity may be run
manually but **CANNOT** retroactively change the verdict (anti-HARKing).

### M2 Supervisor Acceptance

| Metric | Definition |
|---|---|
| Accept rate | (Cases supervisor marks "useful") / (Total cases shown) |
| Threshold for "useful" | Supervisor checkbox on case detail panel; binary |
| Minimum sample for verdict | 30 supervisor labels per arm at 60d checkpoint; 50 per arm at 90d |

### M3 Decision-Quality Case Study (DESCRIPTIVE ONLY, no inference)

| Field | Definition |
|---|---|
| Sample | All cases supervisor marks "useful" AND acts on (changes portfolio within 7d of accept) |
| Outcome window | T+5 to T+10 days post-action |
| Output | Markdown narrative per case: "What was flagged → action taken → realized outcome → lesson" |
| Inference | NONE. M3 is illustrative only. Sample n=20-30 too small for statistical claim. |

### Composite Verdict (SINGLE TEST, anti-multiple-testing)

```
verdict = composite_evaluator(M1_LLM_vs_baseline_A, M1_LLM_vs_baseline_B, M2)

CLEAR_WIN if (M1_diff_A ≥ +5pp) AND (M1_diff_B ≥ +5pp) AND (M2 ≥ 30%)
CLEAR_LOSS if (M1_diff_A ≤ -5pp) OR (M1_diff_B ≤ -5pp)
INCONCLUSIVE otherwise
```

No independent α=5% on each axis. Single composite test; no Bonferroni needed.

### Statistical Power

Pre-flight power analysis (binomial proportion test):
- Detecting +5pp lift at α=0.05 / power=0.80 needs **n ≥ 80 per arm**
- Estimated daily flag rate: 1-3 per arm × 90d = **90-270 per arm**
- Power: ≥ 80% expected at 90d. If actual rate < 1/day, extend to 120d.

---

## 3. Two-Detector Architecture

### Layer Assignment (per D1 Invariant 1)

| Component | Layer | LLM allowed? |
|---|---|---|
| Rule-based detector | 1 (generation) | No (deterministic if-then) |
| LLM detector | 1 (generation) | Yes — Gemini 2.5 Flash with thinking |
| Forward event verification (M1) | 2 (evaluation) | NO — deterministic SQL + price math |
| Composite verdict computation | 2 (evaluation) | NO |
| Hash chain on verdict snapshot | 3 (audit) | NO |

### Rule-Based Detector — Baseline A (rule-only) and Baseline B (rule + macro)

**Hard rules (D1 Invariant: bounded complexity)**:
- ≤ 6 if-then rules
- Features ONLY: price, rolling volatility, concentration, sector weights
- Baseline B may also use macro_research_agent regime_assessment output
- NO NLP, NO ML, NO LLM-derived features

```
Rule 1 (price spike):    abs(daily return) > 2σ_60d  → flag
Rule 2 (volume spike):   volume > 3 × 30d-median     → flag
Rule 3 (concentration):  any sector weight > 30%     → flag
Rule 4 (drawdown):       30d max DD > 10% on holding → flag
Rule 5 (cross-asset):    SPY return × bond return change of sign in same day → flag
Rule 6 (Baseline B only): macro_research regime_assessment shifts category in last 7d → flag
```

Confidence is Likert 1-5 anchored:
- 1 = passing single rule weak match
- 3 = passing single rule strong match OR two rules
- 5 = passing three or more rules

### LLM Detector

**Model**: `gemini-2.5-flash` (locked for entire 90d window)
**Mode**: Thinking enabled, temperature=0
**Thinking budget cap**: 5000 tokens per call
**Prompt cache**: System prompt + taxonomy cached; daily portfolio + news varies

**Inputs (Layer 1 generation only)**:
- Today's portfolio (sector + ticker + weights)
- Recent price summary (1d / 5d / 30d returns)
- Filtered news flow (≤ 50 items, deduplicated, portfolio-relevant)
- Concentration metrics

**Inputs explicitly EXCLUDED (anti-leakage)**:
- macro_research_agent output (B-6 isolation)
- Other LLM agent reflections
- Future-dated data (post-cutoff news only allowed if scan_date > news_publish_date)

**Output schema (JSON, enforced)**:
```json
{
  "flags": [
    {
      "ticker": "XLE",
      "sector": "Energy",
      "event_class": "news_driven|price_spike|concentration|cross_asset",
      "evidence_summary": "max 200 chars, factual citation only",
      "confidence_likert": 3,
      "horizon_days": 5
    }
  ]
}
```

LLM is forbidden from writing free-form "why this is concerning" narratives;
narrative composition uses deterministic templating (D1 Invariant 1, A-4
detection-vs-narrative separation).

### News Sources (Tier 1 required, Tier 2 optional)

Adjusted 2026-05-05 to match actual API keys available in `.streamlit/secrets.toml`.
Original spec named Finnhub/FRED/NewsAPI; substituted with Alpha Vantage +
GNews where keys are configured. Substitutions are equivalent in function
(company news + general headlines + sentiment) so this is a clarification
amendment, not a hypothesis change.

| Source | Tier | Type | Auth | Rate limit |
|---|---|---|---|---|
| SEC EDGAR 8-K filings | 1 (required) | Material event filings | None | 10 req/sec |
| yfinance news | 1 (required) | Headlines per ticker | None | n/a |
| Alpha Vantage NEWS_SENTIMENT | 1 (required, key in secrets.toml as `AV_KEY`) | Company news + pre-computed sentiment | API key | 5 req/min free |
| GNews | 2 (optional, key in secrets.toml as `GNEWS_KEY`) | General headlines fallback | API key | 100 req/day free |
| Finnhub / FRED / NewsAPI | DEFERRED future work | Not configured in this project | n/a | n/a |

**Filter pipeline**:
1. Per-source pull (rate-limited)
2. Keyword filter to portfolio tickers/sectors
3. Dedup by URL hash
4. Truncate to 50 most-recent
5. Send to LLM

This is anti-bloat (A-2 and B-8 audit findings).

---

## 4. Reproducibility Requirements (D1 Invariant 1)

| Requirement | Implementation |
|---|---|
| Model version lock | `gemini-2.5-flash` literal string; no auto-upgrade |
| Temperature | `0` (no stochasticity) |
| Thinking budget cap | 5000 tokens |
| Prompt cache | Full prompt + response stored with SHA-256, keyed on (date, ticker, hash) |
| Seed | 42 (where supported by API) |
| Reproducibility validation | Re-run any T-day's scan must produce byte-identical output (modulo Gemini API non-determinism, < 1% allowed) |

If the API non-determinism exceeds 1% on identical inputs, the run is flagged
and the affected days are excluded from final verdict (anti-spurious-result).

---

## 5. Cost Budget + Monitoring

| Resource | Budget | Mitigation |
|---|---|---|
| LLM cost | ≤ $250 over 90 days (50% of $380 credit) | Daily spend snapshot via `engine.cost_monitor`; alerts at 50% / 75% / 90% |
| API rate limits | Gemini paid tier RPM 60 (per key_pool RPM_HARD_LIMIT) | Existing key_pool throttle |
| Compute | Single daily scan ~5 min wall clock | Run during off-market hours |
| Supervisor time | ≤ 15 min/day labelling cases | bulk-label UI from D3 + skip allowed |

If cost exceeds 50% of budget at 30d soft check, drop to `gemini-2.5-flash-lite`
or scan every 2 days.

---

## 6. Mid-Checkpoint Logic

### 30-Day Soft Check — REMOVED in Slimmed-corrected scope

Originally specified for early warning; cut on 2026-05-05. Live dashboard
(D4.7) shows real-time M1/M2 metrics + cost burn rate at any time, so
explicit 30d alert is redundant — supervisor weekly review reads dashboard.

### 60-Day Hard Mid-Checkpoint (kill criterion)

Hard kill of S6 if **CATASTROPHIC**:
- M1 precision both detectors < 30% **AND**
- M2 supervisor acceptance < 30% across both arms

If catastrophic: archive S6 outputs, retreat HITL to 4 governance categories,
register negative verdict in Lakatos chain (8th falsification entry).

If only one detector catastrophic: ship the other, archive the failing one.

### 90-Day Final Verdict

Composite test (Section 1.4) executes; produces CLEAR_WIN / CLEAR_LOSS /
INCONCLUSIVE. Verdict is hashed and entered in pre-registration ledger.

---

## 7. Isolation Enforcement (B-6 / Paper-E)

| Constraint | Implementation |
|---|---|
| S6 must NOT read macro_research output | LLM prompt excludes any field from `macro_brief_snapshots`; verified by spec-test |
| S6 must NOT trigger paper_trading E arm changes | `strategy_arm_toggle` writes only via existing pages/orchestrator path; S6 anomaly cases cannot call that path; enforced by code-level check |
| S6 anomaly cases enter Governance Queue ONLY | `approval_class="llm_output"`, `approval_type="anomaly_screener"` |
| LLM detector never writes positions or trades | Layer 1 only; no execute_rebalance / apply_tactical_weight_update calls |

---

## 8. Backup Plan — Negative Verdict Thesis Chapter (B-7 pre-reg)

If verdict ∈ {CLEAR_LOSS, INCONCLUSIVE, CATASTROPHIC}, the following thesis
chapter outline is pre-committed:

```
Chapter 8 — S6 LLM Anomaly Screener: A Forward-Only Comparison

8.1 Hypothesis
    H_main: LLM-augmented anomaly screener has marginal value over rule-based.

8.2 Methodology
    90-day forward-only paired comparison (LLM detector vs Baseline A and
    Baseline B). Pre-registered M1 / M2 / M3. Composite verdict at +5pp.
    Documented limitations: supervisor-single, news-source-limited, n bounded.

8.3 Results (auto-filled at D90)
    [verdict / M1 / M2 / M3 / sensitivity / cost / mid-checkpoint outcome]

8.4 Discussion
    [verdict-conditional]
    If CLEAR_WIN: First LLM-positive evidence in project; LLM context-reasoning
                  marginal over rule on news-driven event class.
    If CLEAR_LOSS: 8th falsification; LLM rejected even on its strength
                   (pattern matching + news reasoning) at this scope.
    If INCONCLUSIVE: Underpowered evidence; future work in larger window.
    If CATASTROPHIC: Anomaly-detection abstraction itself rejected; HITL
                     retreats to 4 governance categories.

8.5 Implications for HITL Design
    [verdict-conditional discussion of D2 governance scope]

8.6 Lessons for Project Falsification Chain
    [link to all 7 prior falsifications and Lakatos discussion]
```

This pre-commitment ensures negative verdicts produce publishable chapters
without panic re-write at the deadline.

---

## 9. Implementation Plan (D4.1 - D4.8 sub-sprints)

| Sub-sprint | Scope | Estimate |
|---|---|---|
| D4.1 | Schema additions (anomaly_flags, anomaly_event_verifications tables) | 3h |
| D4.2 | Rule-based detector (6 rules, both baselines, Likert confidence) | 8h |
| D4.3 | News fetchers + filter pipeline (5 sources, dedupe, keyword filter) | 10h |
| D4.4 | LLM detector (Gemini 2.5 Flash, JSON schema, repro lock, cost monitor) | 12h |
| D4.5 | Forward verification engine (M1 precision + recall + F1 only; no calibration/ROC/Brier — Slimmed-corrected) | 4h |
| D4.6 | Daily cron + queue integration + isolation enforcement | 5h |
| D4.7 | UI panel (case detail, deterministic narrative, mid-checkpoint dashboards) | 10h |
| D4.8 | Verification + smoke + spec_hash registration + amendment ledger | 5h |
| **Total** | | **~48h (Slimmed-corrected, 2026-05-05)** |

Plus: ~10-15h supervisor labelling time over 90 days, ~$60-200 LLM cost.

---

## 10. References

**Academic**:
- Asness, Moskowitz & Pedersen 2013, *Value and Momentum Everywhere* (event study horizon)
- Lucca & Moench 2015, *The Pre-FOMC Announcement Drift* (forward window K=5 horizon evidence)
- Brier 1950, *Verification of Forecasts Expressed in Terms of Probability* (calibration)
- Tetlock 2015, *Superforecasting* (Likert anchoring on probability elicitation)
- Tian, Ye & Bowman 2023, *Just Ask for Calibration* (LLM calibration weakness)
- Zheng et al. 2023, *Judging LLM-as-a-Judge* (Layer 2 ban basis)
- Forbes & Rigobon 2002, *No Contagion, Only Interdependence* (cross-asset rule)
- Hansen 2005, *A Test for Superior Predictive Ability* (pre-registration discipline)
- Lakatos 1970, *The Methodology of Scientific Research Programmes* (negative verdict productivity)

**Compliance / Operational**:
- Knight Capital 2012 case (kill switch governance — falsification thesis chapter context)
- López de Prado 2018, *Advances in Financial Machine Learning* §10 (hash chain)

**Project-internal**:
- D1 — [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md)
- D2 — [hitl_architecture_audit_2026-05-05.md](hitl_architecture_audit_2026-05-05.md)
- All 7 prior falsifications in this archive

---

## 11. Amendment Ledger

| Date | Change | Author | spec_hash before | spec_hash after |
|---|---|---|---|---|
| 2026-05-05 | Initial pre-registration; H_main locked, all thresholds locked, baseline A/B defined, news Tier 1/2 declared, isolation enforced | zhangxizhe (supervisor) | (n/a — first commit) | TBD on D4.8 SpecRegistry insert |
| 2026-05-05 | Slimmed-corrected: removed sensitivity sweep / calibration / Brier / ROC AUC / 30d soft check / live HARKing 4-rule. Kept recall + F1 (industry standard anomaly detection). Total estimate 61h → 48h. Justification: master's no-defense; rigor traded for ~13h freed for thesis/portfolio/job. Pre-registration core (precision verdict + dual baseline + 60d hard checkpoint + post-cutoff isolation) preserved. | zhangxizhe | (still draft, not yet registered) | (still draft) |
| 2026-05-05 | News source substitution (clarification, not hypothesis amend): Finnhub → Alpha Vantage NEWS_SENTIMENT (key in secrets.toml). NewsAPI → GNews (key in secrets.toml). FRED deferred to future work. SEC EDGAR + yfinance unchanged. Equivalent function, so EFFECTIVE_N_TRIALS unaffected. | zhangxizhe | (draft) | (draft) |
| 2026-05-05 | Budget amendment (clarification, operational): LLM cost ceiling raised $200 → $250 (66% of $380 Gemini credit). Alerts still fire at 50/75/90% of new budget. Statistical thresholds unchanged → EFFECTIVE_N_TRIALS unaffected. | zhangxizhe | (draft) | (draft) |

Future amendments require:
- amend_spec() call with explicit `kind` (clarification / threshold_tweak / hypothesis_amend / endpoint_swap)
- EFFECTIVE_N_TRIALS counter increment per kind multiplier
- HARKing 4-rule check passed
- Rationale logged here

**Lakatos closure**: this spec, once registered, contributes 1 trial to
EFFECTIVE_N_TRIALS. Negative verdict produces 1 falsification entry. Positive
verdict produces project's first LLM-positive evidence.
