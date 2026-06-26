# MacroAlphaPro — Project Overview (External Show Version)

**Last updated**: 2026-06-23
**Repository**: solo-quant research workbench with LLM-augmented pipeline
**Status**: deployed book live (paper-trade), self-tuning calibration loop in production, public GitHub snapshot pipeline READY
**Lead**: [Zhang Xizhe](https://www.linkedin.com/in/zhangxizhe) · NUS MSBA 2026

---

## 1. One-Sentence Summary

A one-person quant research workbench that combines a 4-sleeve systematic strategy book (in-sample retrospective replay Sharpe 1.32, Lo-2002 SE 0.33, 95% CI [+0.67, +1.97], t=4.0; 486 weeks 2014-09 → 2023-12) with an LLM-augmented research pipeline that crawls papers, proposes hypotheses, runs rigorous statistical tests, and — load-bearing — publicly tracks the calibration of its own predictions before each test runs. **The single most differentiating finding: the predictor LOSES to a fair time-aware family-prior baseline by +0.114 Brier, and the loss was published instead of hidden.**

---

## 2. What Is Actually Shipped (External-Show-Ready)

### 2.1 Deployed Strategy Book

- **4 sleeves**: K1_BAB, D_PEAD, PATH_N, CTA_PQTIX
- **Backtest = in-sample retrospective replay** (2014-09-05 → 2023-12-22, 486 weekly obs ~9.4 years); NOT walk-forward / NOT CPCV — the deployment-design doc explicitly labels this in-sample
- **Combined metrics**: Sharpe **1.32**, MaxDD −5.8%, ann return 8.07%
- **Lo (2002) Sharpe SE**: 0.33, 95% CI [+0.67, +1.97], t-stat 4.0 vs SR=0 (significant at p<0.0001) — `scripts/reports/report_sharpe_se.py` reproduces
- **Per-sleeve SE caveat**: CTA_PQTIX standalone Sharpe 0.43 is NOT significant at 5% (p=0.094); its contribution to the combined book comes from diversification (near-zero correlation with the other 3), not standalone alpha. Honest disclosure
- **Forward expectation per design**: Sharpe 0.85–1.15 (in-sample → OOS degrades; banding reflects this)
- **Live paper trade**: started 2026-05-13 (cron daily; latest attribution log 2026-06-24; too short for Sharpe inference)
- **2 real halt events recorded since go-live**: 2026-05-19 (GLD +7.50% > 5% sleeve cap mode 1) and 2026-06-17 (PATH_N MRVL +25.00% > 5% intra-strategy cap mode 1b) — both pre-trade Risk Manager (spec id=69) stopped the order before it shipped
- Source: `data/portfolio_replay/v1_combined_replay_verdict.json` (private; values reproduced inline in arxiv §A.1)
- Detailed attribution: `data/research/deployed_book_attribution.md`

### 2.2 Public-Facing Paper (arxiv draft)

- `docs/arxiv_preprint_draft_2026-06-22.md` — 6141 words, markdown canonical source
- `docs/arxiv_preprint_2026-06-22.tex` — 830 lines, LaTeX submittable
- `docs/arxiv_compile_instructions.md` — Overleaf + local pdflatex compile paths
- `docs/figs/belief_fig*.png` — 3 figures (reliability diagram, per-family Brier CI, baseline comparison)

**Three load-bearing empirical findings** (all backed by data in repo):

1. **Predictor beats random** (Brier 0.374 vs 0.444, 95% CI [0.33, 0.41] strictly below baseline, p < 0.001)
2. **Predictor LOSES to family-prior baseline** (delta +0.114 Brier, CI [+0.06, +0.16] excludes zero) — **honest negative finding most labs would not publish**
3. **Hosmer-Lemeshow REJECTS calibration uniformity** (p < 0.001) — mid-confidence bins systematically over-confident

### 2.3 Belief Layer (Calibration Tracking) — LIVE

5-phase architecture, all phases shipped:

| Phase | Status | Artifact |
|---|---|---|
| 1. Predict-commit air-gap | LIVE since 2026-06-11 | `data/research/predictions.jsonl` (498 records) |
| 2. Autopsy join | LIVE | `data/research/autopsies.jsonl` (**94 records**) |
| 3. Calibration surface | LIVE (daily refresh) | `data/research/belief_track_record.md` |
| 4. Closed-loop prior calibration | LIVE | `engine/research/belief_prior_calibration.py` |
| 5. Track-record-aware synthesis | LIVE | `engine/research/belief_synthesis_context.py` |

**Self-tuning loop** (the central differentiator):

```
W6-rigor measure (0.374 LLM-only)
  → W6-rigor-A param tune (N=5→3, α=3→1) → 0.353 (-6%)
  → W7-arxiv-v05 sweep (per-family ensemble) → 0.254 in-sample
  → W7-v06 wire flag-gated OFF
  → W7-v07 ACTIVATE (per-family w_fam)
  → W7-v08 LOOCV honesty pass → 0.278 (overfit gap +0.018)
  → W7-v09 HONEST CORRECTION (w=1.0 global pure family-empirical) → 0.260
  → First OOS evidence: 2/2 ensemble-active pairs Brier 0.000 (perfect, n=2)
```

The system **measured → tuned → re-measured → revised** itself. Public audit trail in 26 git commits this session.

### 2.4 Paper Ingestion Pipeline (Stage 0 ClaimType Router) — LIVE

- **661 papers cached + tagged** (`data/papers_curator/cache_with_claim_type.jsonl`)
- 8 ClaimType labels: FACTOR_HYPOTHESIS, METHODOLOGY, DECAY_STUDY, CAPACITY, MICROSTRUCTURE, FACTOR_STRUCTURE, DOMAIN_FACT, OTHER
- Hybrid deterministic + LLM fallback (~$0.001/paper)
- Live wire-up: `engine/agents/papers_curator/summarizer.py` (every new paper auto-tags)
- Daily cron: `papers-curator-daily-ingest` 08:30
- Router v2 fix (commit `9e94ae12`): false-positive rate **30% → ~0%** by removing ambiguous single-word triggers ("anomaly", "premia")

### 2.5 End-to-End Autonomous Demonstration: Bond-VRP

A complete autonomous-research case study documented in paper Section 5 + repository:

```
papers_curator crawl
  → claim_type tag (FACTOR_HYPOTHESIS, conf 1.0)
  → cross-source synthesis (Sonnet, $0.05)
  → 2 candidates auto-emitted (Bond-VRP + Treasury duration-convexity)
  → mechanism_family alias resolution (synthesis_writer)
  → 3-layer governance chain (PIT whitelist + universe probe + template contract)
  → vrp_treasury template (MOVE + TLT, MVP)
  → verdict: RED (Sharpe +0.56, NW-t +1.69 < multi-test threshold 2.09)
  → autopsy pair: Brier 0.36 (well-calibrated against family prior)
```

The system autonomously **read papers → proposed novel cross-asset hypothesis → ran rigorous statistical test → emitted verdict consistent with literature (Carr-Wu 2009 equity-only)**. The human triggered Phase 2, the alias-map fix, and the template build; the human did not author the prediction, dispatch plan, verdict, or autopsy.

### 2.7 Frontend Story Dashboard — LIVE (2026-06-23)

Telemetry-driven UI integration response to a senior 3-perspective
audit (systems-arch / quant-AI-eng / academic). 4 phases shipped in
one day; see `INTERNAL_DESIGN_INDEX §14` for the full narrative.

- **Brier KPI tile** sticky on every page hero strip
  (`Brier 0.374 (+0.11 vs fam)`) — the honest negative finding is no
  longer markdown-only
- **`/research/calibration`** — first-class HONEST FINDING page with
  all 5 headline stats + bootstrap CIs + Hosmer-Lemeshow verdict +
  per-family belief depth
- **`/research/workflow`** — single-picture system map: 8-stage flow
  from 661 papers → 12 synth runs → 303 hypotheses → 206 specs →
  517 predictions → 299 verdicts → 94 autopsies → Brier 0.374
- **RESEARCH rail cull** 10 → 4 prominent + 6 in "More" disclosure,
  evidence-based from 11 days / 307 telemetry events
- **8 `/lab/*` redirects** recover 98+ muscle-memory 404 hits

Audience served: principal daily monitoring, recruiter 90-second
scan, arxiv reviewer "is this real?" check, future-Claude continuity.

### 2.7b Operator Console — 9 of 9 Pipeline Stations LIVE (2026-06-23 → 2026-06-25)

UI-triggerable end-to-end research pipeline. Lets an external operator
drive the full paper-to-verdict chain through clicks alone — no Claude
conversation required. Each station implements the 5-element contract
(preflight / cost estimate / JSON-schema config form / async execute
with SSE progress / lineage hints to next station).

| ID | Station | Cost | Capital? |
|---|---|---|---|
| S1 | Paper Ingest (arxiv URL → ClaimType → registry) | $0.00 | read-only |
| S2 | Hypothesize (LLM synthesis from recent corpus) | $0.10 | read-only |
| S3 | FactorSpec Extract (hypothesis → spec_hash) | $0.05 | read-only |
| S4 | FORWARD Dispatch (spec → strict-gate verdict, 8 stat tests) | $0.10 | read-only |
| S5 | ENHANCE Dispatch (variant CSV → paired bootstrap verdict) | $0.00 | read-only |
| S6 | Verdict View (drill prediction + autopsy + lineage) | $0.00 | read-only |
| S7 | PROMOTE 9-gate MVP (GREEN verdict → /approvals) | $0.00 | **mutates_capital → /approvals** |
| S8 | Rollback (deployed sleeve → /approvals) | $0.00 | **mutates_capital → /approvals** |
| S8b | Doctrine Lock (form → memory/*.md + event) | $0.00 | read-only |

**Capital-decision doctrine, code-enforced**: `StationSpec.mutates_capital`
is checked at import time. Any station declaring `mutates_capital=True`
whose source doesn't reference `_proposals.jsonl` raises
`CapitalDoctrineViolation` on register. Doctrine is the type system,
not a comment.

**Idempotency**: `create_job(..., idempotency_key=...)` short-circuits
duplicate jobs within a 60-second window — double-click on the trigger
button no longer burns cost twice.

Source: `engine/operator_console/` (9 stations + registry + worker +
cost ledger + event store), `frontend/components/operator_console/`,
`api/routes_operator_console.py` (15 endpoints), `docs/architecture/operator_console.md`.

### 2.8 Public GitHub Snapshot Pipeline — LIVE (2026-06-23)

B1 dual-repo architecture: continuous private dev, weekly sanitized
snapshot to a public mirror at `${REPO_ROOT}/Desktop/macroalphapro-public`.

- `.publishrc.yaml` — whitelist + sanitize regex + post-check forbidden
- `scripts/publish/build_public_snapshot.py` — 4-stage deterministic
  pipeline (collect → copy+sanitize → post-check → report)
- `scripts/publish/weekly_snapshot_wrapper.bat` — schtasks-ready
- `docs/PUBLISH_PIPELINE.md` — operating guide
- `LICENSE` (MIT) + portfolio-tailored `README.md`

Last build: **2,283 files / 32.2 MB / 0 forbidden patterns** (vs
8.4 GB raw private repo). USER ACTION pending: rotate AV/GNEWS keys,
create public repo on GitHub, run first push.

### 2.9 Statistical Rigor Pass (6 standard tests)

`engine/research/belief_track_record_rigor.py` runs:

- T1: Bootstrap CI on overall Brier (B=10000, percentile CI)
- T2: Baseline comparison (paired bootstrap on per-autopsy deltas)
- T2-time-aware: Fair family-prior baseline (no future-info leakage)
- T3: Sign test on optimism bias (binomial)
- T4: Per-family CI + Benjamini-Hochberg FDR (q=0.10)
- T5: Mann-Kendall trend test
- T6: Hosmer-Lemeshow calibration goodness-of-fit
- (added W7-v05): Threshold × alpha 2D sweep
- (added W7-v08): LOOCV ensemble robustness

All tests reproducible: `python scripts/reports/report_belief_track_record_rigor.py` ($0 LLM).

---

## 3. Architecture (3-Layer Diagram)

> **Note**: a separate doc `docs/architecture/SKILL_three_layers.md`
> re-frames the same architecture in the substrate / experience /
> self-correction layering used by concurrent agentic-quant work
> (QuantML / Kimi + WorldQuant BRAIN, 2026-06-23) and documents 6
> load-bearing differentiators (calibration tracking + honest negative
> finding + 8 statistical rigor tests + substrate ownership +
> 6-11 multi-layer epistemic review + role classification).



```
┌─────────────────────────────────────────────────────────────┐
│  4 AUTONOMOUS AGENTS (papers_curator / strengthener /       │
│                       persona α β γ / decay_sentinel)        │
│                                                              │
│  papers_curator → 50+ papers/day → tag → summarize →        │
│                   synthesize hypothesis candidates           │
│                                                              │
│  strengthener   → review → FactorSpec extract → dispatch     │
│  persona        → α pre-mortem / β cross-domain / γ replic. │
│  decay_sentinel → monitor deployed sleeves → emit alerts    │
└─────────────────────────────────────────────────────────────┘
                          │ hypothesis flow
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3-PIPELINE SPLIT (FORWARD / ENHANCE / PROMOTE)              │
│                                                              │
│  FORWARD   = "is X a real alpha?"                            │
│              tools: FF5+MOM spanning, NW-t HAC,              │
│                     Bailey-LdP DSR, BUG-3 multi-test         │
│              verdict: GREEN / MARGINAL / RED                 │
│                                                              │
│  ENHANCE   = "does X' improve deployed X?"                   │
│              tools: Politis-Romano paired bootstrap,         │
│                     Jobson-Korkie Sharpe-diff                │
│              verdict: IMPROVEMENT / NOISE / DEGRADATION      │
│                                                              │
│  PROMOTE   = "deploy as new sleeve?"                         │
│              9 checks (FORWARD GREEN, cost-robust, PIT,      │
│              replication, multi-period, anchor-residual,     │
│              cross-sleeve corr, capacity, human decision)    │
│              verdict: PROMOTE-READY → human approves         │
└─────────────────────────────────────────────────────────────┘
                          │ verdict events
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  BELIEF LAYER (5 phases, all LIVE)                           │
│                                                              │
│  Phase 1: predict-commit before each verdict (air-gapped)    │
│  Phase 2: autopsy joins prediction ↔ verdict                 │
│  Phase 3: track record markdown (daily refresh 06:35)        │
│  Phase 4: closed-loop prior calibration from autopsies       │
│  Phase 5: track-record-aware synthesizer context             │
│                                                              │
│  Brier tracking: 94 autopsies, 0.374 LLM-only,              │
│                  0.260 pure family-empirical (current prod)  │
└─────────────────────────────────────────────────────────────┘
                          │ feedback
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  RISK + DEPLOYMENT (12-gate Risk Manager + HUMAN gate)       │
│                                                              │
│  12 deterministic gates: position caps (2-tier Basel-III),  │
│  HHI, gross/net exposure, VaR-95, ES-95, kill switch         │
│  Capital decisions: HUMAN-in-loop (always)                   │
└─────────────────────────────────────────────────────────────┘
```

**Constraint doctrine** (load-bearing):
- Pattern 5 ban: NO multi-agent debate (per Tetlock fake-diversity)
- Capital decisions: HUMAN, never auto
- Event store: every action emits `factor_verdict_filed`/`memory_locked`/etc.
- Spec amendment: pre-commit hook blocks unregistered spec edits

---

## 4. Crons Auto-Running (Schtasks Registered)

| Cron | Schedule | Cost/run | Purpose |
|---|---|---|---|
| `papers-curator-daily-ingest` | Daily 08:30 | ~$0.10 | Crawl + tag + summarize new papers |
| `research-discover` | Daily 06:00 | $0 | Discovery cron |
| `research-forward-oos` | Daily 06:15 | $0 | FORWARD OOS check |
| `research-daily-summary` | Daily 06:30 | $0 | Daily memo |
| `daily-belief-refresh` | Daily 06:35 | $0 | Belief autopsy + track record + rigor regen |
| `burndown-daily` (actually weekly Mon+Thu 09:00) | Mon+Thu 09:00 | ~$0.30 | Dispatch top-3 hypothesis candidates |
| `research-backfill-weekly` | Sun 04:00 | $0 | Weekly backfill |
| `MacroAlphaPaperExecution` | Daily 23:00 | $0 | Paper trade NAV update |
| (multiple monitoring crons) | Daily | $0 | Watchdog / paper trade / direction proposer / etc. |

**Total LLM monthly cost** (verified from `data/llm_cost_ledger.jsonl`,
audit 2026-06-23): **~$19/month run-rate** ($26.75 over 41 days
2026-05-13 → 2026-06-22). Steady-state excluding initial backfill
spike (week of 2026-05-12 had 15,094 calls from one-time PDF backfill):
**~$16/month**. By model: Gemini Flash 15,094 calls ($11.70), Sonnet
523 ($12.27), DeepSeek-Pro 1,281 ($2.78), DeepSeek-Flash 11 ($0.003),
Haiku 1 ($0.0002). Sonnet 3% of calls / 46% of spend = reserved for
tool-use-heavy work as designed.

---

## 5. Key Empirical Findings (Reproducible)

Every number below is computed from data files in repo:

### 5.1 Predictor calibration (n=94 autopsy pairs)

| Variant | Brier | Reproducible from |
|---|---|---|
| Random uniform baseline | 0.444 | (definitional) |
| LLM-only (pre-W7-v07 production) | 0.374 | `python scripts/reports/report_belief_track_record_rigor.py` |
| In-sample per-family ensemble | 0.254 | `python scripts/reports/report_belief_ensemble_sweep.py` (in-sample) |
| LOOCV per-family ensemble | 0.278 | same script, LOOCV section |
| **Pure family-empirical (current prod post-v0.9)** | **0.260** | same data, w=1.0 forced |

### 5.2 Per-family LOOCV winners (n ≥ 3, post-W7-v09)

| Family | LOOCV Brier | reading |
|---|---|---|
| CROSS_SEC_UNKNOWN (n=16) | 0.022 | near-perfect |
| SPANNING_SMB (n=6) | 0.060 | near-perfect |
| SPANNING_CMA (n=5) | 0.072 | near-perfect |
| SPANNING_RMW (n=3) | 0.120 | small n but holding |

### 5.3 First OOS evidence under v0.9 corrected logic (2026-06-22)

| Pair | Ensemble fired | Predicted | Actual | Brier |
|---|---|---|---|---|
| SPANNING_SMB | YES (w=1.0) | 100% MARG | MARG | **0.000** |
| SPANNING_RMW | YES (w=1.0) | 100% RED | RED | **0.000** |
| COMBINATION_HML_MOM | no (n<3) | default prior | MARG | 0.360 |

**Ensemble-active mean: 0.000 (n=2 — too small to validate; directional confirmation only)**

### 5.4 Paper ingestion pipeline (661 papers)

| Class | n | % | Reading |
|---|---|---|---|
| OTHER | 300 | 47.4% | LLM says "fits no specific class" |
| METHODOLOGY | 91 | 14.4% | bootstrap / robust SE / multi-test |
| FACTOR_HYPOTHESIS | 82 | 13.0% | tradable strategy claims |
| DOMAIN_FACT | 81 | 12.8% | real-economy facts |
| MICROSTRUCTURE | 29 | 4.6% | bid-ask / dealer / HFT |
| UNKNOWN | 22 | 3.5% | router failure floor |
| DECAY_STUDY | 12 | 1.9% | post-pub decay |
| CAPACITY | 9 | 1.4% | AUM ceiling |
| FACTOR_STRUCTURE | 7 | 1.1% | spanning / model comp |

---

## 6. Reproducibility (How an External Reviewer Verifies)

All key claims reproduce from:

```bash
# 1. Calibration aggregates + rigor pass (free)
python scripts/reports/report_belief_track_record.py
python scripts/reports/report_belief_track_record_rigor.py

# 2. Per-family ensemble sweep + LOOCV (free)
python scripts/reports/report_belief_ensemble_sweep.py

# 3. Figures (free)
python scripts/reports/report_belief_track_record_figures.py

# 4. Deployed-book attribution (free)
python scripts/reports/report_deployed_book_attribution.py

# 5. ClaimType backfill measurement (~$0.15 if --use-llm)
python scripts/reports/report_papers_curator_claim_type_backfill.py

# 6. Convert markdown paper to LaTeX (free)
python scripts/reports/convert_arxiv_md_to_tex.py
```

Outputs land in `data/research/*.md` and `docs/figs/*.png`. JSON variants for machine consumption.

---

## 7. Honest Caveats (Front-Page; Not Buried)

1. **Sharpe 1.32 is BACKTEST REPLAY**, not live. Live paper trade is ~1 month, n=21 NAV records, cum return −0.18% (insufficient for Sharpe inference).
2. **LLM contribution to verdict prediction was MEASURED at ~10% globally**. The architectural fix (v0.9) routes around the LLM for verdict prediction when family history is sufficient. This is uncomfortable for "AI-driven quant" narrative but is what the data showed.
3. **n=94 is small for inferential claims**. The W7-v08 LOOCV result (0.278) is the CV-honest number; the first 2 OOS pairs at 0.000 are directional confirmation only.
4. **24 ENHANCE tests on deployed sleeves produced 0 IMPROVEMENT verdicts**. Per `memory/feedback_sizing_before_signal_2026-06-17`, deployed book is near-optimal for current substrate; signal-side enhance has limited room. Sizing-side levers (HRP / drop-one / vol target) remain unexplored.
5. **System produces ~1 deployable new sleeve per 6 months** (GP/A 2026-06-08; Bond-VRP RED today). HXZ 65% replication failure means most autonomous candidates fail; this is the expected rate.
6. **Architecture pattern is reproducible** (event store + spec governance + doctrine-as-code can be re-built in ~1 week by a senior engineer). The differentiator is not novel architecture; it is **the longitudinal calibration data + audit trail of honest iteration**.

---

## 8. What's NOT Yet Shipped (Honest)

- HRP-based sizing audit / drop-one Sharpe recommender (designed but deferred 5+ sessions; Sizing-before-Signal doctrine standing)
- Cross-asset extension scanner (β agent shipped 2026-06-14 but not auto-cron'd)
- chief_of_staff weekly digest (spec'd 2026-06-06, not deployed)
- PROMOTE pipeline 9-check auto-runner (engine.research.promote stub)
- Workflow Trace SVG advanced features (currently shows counts; future: pulse animation on in-flight nodes, click-to-drill expand with last-N events)

### 8.1 Recently shipped (was on this list, no longer)

- ~~Substrate gap detector~~ → `engine/research/factor_exposure_gap_detector.py` SHIPPED 671 LOC (see INTERNAL_DESIGN_INDEX §13)
- ~~Public open-source release blockers~~ → `.publishrc.yaml` + `scripts/publish/build_public_snapshot.py` + LICENSE + `docs/PUBLISH_PIPELINE.md` ALL SHIPPED 2026-06-23. Snapshot builds clean (2,283 files / 32 MB / 0 forbidden hits). Pending only user-action key rotation + first push.
- ~~UI for honest negative finding~~ → `/research/calibration` page + Brier KPI sticky on every page SHIPPED 2026-06-23.
- ~~Single-picture system map~~ → `/research/workflow` SHIPPED 2026-06-23.

---

## 9. Where to Find What (File Inventory)

| Purpose | Location |
|---|---|
| **The paper** (markdown source) | `docs/arxiv_preprint_draft_2026-06-22.md` |
| **The paper** (LaTeX) | `docs/arxiv_preprint_2026-06-22.tex` |
| Paper compile instructions | `docs/arxiv_compile_instructions.md` |
| Paper figures | `docs/figs/belief_fig*.png` |
| Project overview (this file) | `PROJECT_OVERVIEW.md` |
| Daily belief track record | `data/research/belief_track_record.md` |
| Belief rigor pass | `data/research/belief_track_record_rigor.md` |
| Ensemble sweep + LOOCV | `data/research/belief_ensemble_sweep.md`, `belief_ensemble_loocv.md` |
| Deployed book attribution | `data/research/deployed_book_attribution.md` |
| ClaimType coverage report | `data/papers_curator/claim_type_coverage_report.md` |
| Raw autopsies | `data/research/autopsies.jsonl` (94 records) |
| Raw predictions | `data/research/predictions.jsonl` (498 records) |
| Event store | `data/research_store/events.jsonl` |
| Tagged paper cache | `data/papers_curator/cache_with_claim_type.jsonl` (661 papers) |
| Project doctrine (memory) | `~/.claude/projects/.../memory/MEMORY.md` |
| Portfolio README (public-facing) | `README.md` |
| MIT license | `LICENSE` |
| Three-layer architecture (vs concurrent Kimi+WQ work) | `docs/architecture/SKILL_three_layers.md` |
| Public snapshot pipeline guide | `docs/PUBLISH_PIPELINE.md` |
| Snapshot config | `.publishrc.yaml` |
| Snapshot builder | `scripts/publish/build_public_snapshot.py` |
| Weekly auto-publish wrapper | `scripts/publish/weekly_snapshot_wrapper.bat` |
| UI workflow trace page | `frontend/app/(terminal)/research/workflow/page.tsx` |
| UI calibration headline page | `frontend/app/(terminal)/research/calibration/page.tsx` |
| Belief calibration API | `api/main.py` → `GET /api/research/belief/calibration` |
| Workflow counts API | `api/main.py` → `GET /api/research/workflow/counts` |
| UI usage telemetry | `data/telemetry/events.jsonl` (auto-collected, R4.2) |

---

## 10. Session Audit Trail (2026-06-22, this session)

**26 commits** producing the W6-rigor + W7-arxiv arc:

```
W6-rigor (6 stat tests on 85 prediction-verdict pairs)
W6-rigor-B (fair time-aware family-prior baseline)
W6-rigor-A (predict_verdict params N=5→3, α=3→1)
W6-rigor-A-validate (force-burndown 2 new pairs)
W4-E2E (autonomous pipeline runs paper → summary → hypothesis → dispatch)
W6-rigor-A-validate-loop-closed (Bond-VRP gets verdict)
W6-rigor-A (Bond-VRP full governance chain registered)
W6-rigor-A-router-v2 (kill 30% router false-positive rate)
W6-rigor-A-cron-peel (daily $0 belief refresh + schtasks register)
W7-arxiv-v01 (preprint draft, 3940 words)
W7-arxiv-v02 (figures + Appendix B reproducible code)
W7-arxiv-v03 (Appendix A from canonical replay verdict)
W7-arxiv-v04 (markdown → LaTeX converter + .tex source)
W7-arxiv-v05 (per-family ensemble sweep, 34% Brier reduction)
W7-arxiv-v06 (wire ensemble into predict_verdict, flag OFF default)
W7-arxiv-v07 (ACTIVATE ensemble blend)
W7-arxiv-v07b (first 2 OOS pairs)
W7-arxiv-v08 (LOOCV robustness: activation survives CV)
W7-arxiv-v09 (HONEST CORRECTION: w=1.0 global beats per-family)
W7-arxiv-v09-OOS (3 force burndowns, ensemble-active Brier 0.000)
```

Total LLM spend: ~$1.85.

### Session 2026-06-23 continuation (9 additional commits)

```
ccf73d9c  Publish pipeline + portfolio README (B1 dual-repo)
2a01dfe9  Phase B: Brier KPI strip + /research/calibration + 8 /lab redirects
64d698a7  Phase C: /research/workflow single-picture system map
3bb53889  Phase D: RESEARCH rail cull 10→4 + More disclosure (telemetry-driven)
(latest)  LICENSE + PUBLISH_PIPELINE.md + weekly cron wrapper + INTERNAL v3
```

Senior 3-perspective audit responded to in full (4-phase A→D plan
executed in one session; details in `INTERNAL_DESIGN_INDEX §14`).
Cumulative session total: 35 commits, ~$1.85 LLM.

---

## 11. Elevator Pitch (3 Audiences)

**To a quant academic**:
> "We operate a 4-sleeve systematic strategy book with rigorous calibration tracking. We measured the LLM-augmented predictor and found it loses to a deterministic family-empirical baseline by 0.114 Brier. We shipped the architectural fix, validated under LOOCV, and are now publicly tracking realized Brier vs predicted on new data. The paper, code, and audit trail are open."

**To an AI safety / honest-AI researcher**:
> "Most LLM-finance projects don't measure if their LLM is calibrated. We do, and we publish the negatives. The system has a 26-commit audit trail of measure → tune → re-measure → revise iterations. Bounded autonomy: capital decisions are human-gated, multi-agent debate banned, every prediction air-gapped from verdict pipeline."

**To a senior hiring manager**:
> "Solo quant + AI workbench, 6 months of operations, full reproducibility, honest negatives published. The codebase doctrine + calibration tracking pattern is the senior signal — not the Sharpe number."

---

*Generated 2026-06-22 as the single-entry-point project overview. Regenerable manually; not auto-cron'd (this file changes only when the project's external-facing narrative changes).*
