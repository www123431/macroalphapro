# MacroAlphaPro SKILL — Three Layers

**Why this document exists.** A concurrent work (QuantML, 2026-06-23,
*"WorldQuant 因子挖掘 SKILL 开源了：三天自进化，20+ Spectacular Alpha"*)
described a Kimi + WorldQuant BRAIN agentic factor-mining system in
three layers: API manual, alpha experience library, self-evolution
mechanism. The architecture is similar enough on the surface that a
reader of both projects asks: **how is MacroAlphaPro different?**

This doc answers that. Same three-layer framing for parallel reading,
but each layer is built differently. **The differences are not "we did
more work"; they are load-bearing design decisions that map to
institutional-quant baselines (Markowitz / Frazzini-Pedersen / López
de Prado / Asness-Pedersen) that the Kimi+WQ project does not
address.**

Last updated: 2026-06-23.

---

## At-a-glance comparison

| Layer | Kimi + WorldQuant BRAIN | MacroAlphaPro |
|---|---|---|
| **L1 Substrate** | Rented WorldQuant BRAIN platform (closed-box scoring) | Own WRDS data + FactorSpec schema + 6 dispatch templates + 3-pipeline split + 12-gate Risk Manager + role classifier |
| **L2 Experience** | Single SKILL.md + log files | Typed event store (894+ events, immutable) + intuition_rules + 633 ClaimType-tagged papers + per-family belief context + 106 RED lessons |
| **L3 Self-correction** | `evolve_skill.py` writes failures back to SKILL.md | **Belief Layer 5 phases** (predict-commit air-gap → autopsy → calibration → closed-loop prior → track-record-aware synthesizer) + **6-11 multi-layer epistemic review** (α/β/γ persona + Devil's Advocate × 2 + 8 statistical tests + post-GREEN rigor + external LLM audit + PIT audit + Council critique + capability evidence) |

---

## L1 — Substrate

> **Their answer to "how do I know if a factor works?"**: ask
> WorldQuant BRAIN. It tells you `Spectacular / OK / Rejected`.
>
> **Our answer**: run 8 academic statistical tests in our own code,
> against WRDS data we own, on a FactorSpec spec we control, with
> outcomes classified into 4 strategic roles.

### What's in L1

- **Data ownership**: WRDS subscription (Compustat fundamentals,
  CRSP returns, IBES analyst estimates, OptionMetrics surface).
  Source-of-truth lives on our disk. PIT audit module (`engine/research/
  pit_audit.py`) actively detects look-ahead leakage; verdicts surface
  PIT status in the dashboard.

- **FactorSpec schema** (`engine/research_store/schema.py`): hash-locked,
  audit-trail. Every dispatch is keyed to a spec_id. Spec amendments
  are tracked as `spec_amended` events; you can git-log any sleeve back
  to the spec that authorized it.

- **6 dispatch templates** under `engine/agents/strengthener/templates/`:
  - `cross_section_zscore` (default L/S quintile)
  - `vrp_treasury` (variance risk premium, MOVE + TLT)
  - `spanning_mom` / `spanning_hml` / `spanning_smb` / `spanning_cma` /
    `spanning_rmw` (FF5+MOM spanning regressions)
  - `event_drift_revision` (PEAD-family, analyst revision)
  - `profitability` (Novy-Marx GP/A and variants)
  - `spx_skew_premium` (option-implied tail premium)

- **3-pipeline split** (CLAUDE.md doctrine, 2026-06-11):
  - **FORWARD**: "Is X a real alpha?" → FF5+HXZ spanning,
    Newey-West HAC, Bailey-López de Prado DSR multi-test correction.
    Verdicts: GREEN / MARGINAL / RED.
  - **ENHANCE**: "Does X' strictly improve deployed X?" → Politis-
    Romano 1994 paired block bootstrap, Jobson-Korkie / Memmel
    Sharpe-diff t-stat. Verdicts: IMPROVEMENT / NOISE / DEGRADATION.
  - **PROMOTE**: "Deploy as new sleeve?" → 9 gates (FORWARD GREEN +
    cost-robust + PIT clean + replication + multi-period + anchor-
    residual + cross-sleeve correlation + capacity + human approval).

  Paired SE ≈ √(2(1-ρ)/n) for ENHANCE vs unpaired √(1/n) for FORWARD;
  at ρ≈0.95 paired is 3.2x tighter. Routing the wrong pipeline kills
  90%+ of real improvements OR gives false IMPROVEMENT on uncorrelated
  new strategies. Pipeline routing is enforced in both
  `engine.research_store.hypothesis.classifier` and
  `engine.research.burndown_ranker`.

- **12-gate Risk Manager** (`engine/risk_manager/`): position caps
  (2-tier Basel-III), HHI, gross/net exposure, VaR-95, ES-95, kill
  switch. Capital decisions stay HUMAN-gated (CLAUDE.md doctrine).

- **Role classifier** (institutional-quant baseline most agentic
  systems skip): every deployed sleeve gets a role label —
  `alpha` / `insurance` / `regime_premium` / `trend`. Different role
  ⇒ different evaluation metric:
  - `alpha` → rolling_sharpe, full_sharpe
  - `insurance` → **crisis_payoff** (Sharpe is often negative for
    insurance assets — TLT/GLD lose in calm periods but pay out in
    crisis; using Sharpe to evaluate insurance is a category error)
  - `regime_premium` → signal_ic (information coefficient)
  - `trend` → crisis_payoff + rolling_sharpe (dual-use)

  Without role classification a system can only collect alpha-shaped
  factors, which produces a book without diversification — a
  well-known institutional-quant failure mode (Markowitz mean-variance
  assumes you've already separated risk-on from hedge).

### Why this matters vs Kimi+WQ

WorldQuant BRAIN is a "high-concurrency factor validator" (their
phrasing). It runs the IS check, correlation filter, and PnL
computation in their cloud. You don't see how `Spectacular` is
defined. You can't add a PIT integrity check. You can't classify
your insurance asset as insurance (BRAIN will reject it for low
Sharpe). You can't split FORWARD from ENHANCE statistics (BRAIN
treats every submission as forward).

Owning the substrate costs more (we maintain the data pipeline; they
just submit FASTEXPR). It buys: full statistical visibility, the
freedom to test mechanisms BRAIN's scoring rejects, and resilience
to vendor risk (they themselves quote "look at Gemini CLI shutting
down" as a precedent).

---

## L2 — Experience

> **Their answer to "how does the system avoid repeating mistakes?"**:
> `SKILL.md` accumulates field-name traps and Fitness patterns.
>
> **Our answer**: typed event store + per-family belief context +
> 106 RED lessons + intuition rules + capability evidence files —
> every research state change is one of 8 typed events, immutable,
> auditable, queryable through a typed surface.

### What's in L2

- **Typed event store** (`engine/research_store/`, CLAUDE.md
  Research Event Emission Doctrine). 894+ events as of 2026-06-23,
  append-only `events.jsonl`. 8 canonical event types:
  - `factor_verdict_filed` (299) — strict-gate verdict landed
  - `memory_doctrine_locked` (299) — lesson learned, written to disk
  - `decay_alert` (188) — sleeve degradation detected
  - `capability_evidence_filed` (61) — markdown evidence for each
    GREEN verdict, audit-ready
  - `post_green_rigor_run` (21) — DEAD_POST_PUB / SUBSUMED /
    SHORT_FEE_KILLS rigor pass
  - `papers_curator_synthesis_run` (12)
  - `forward_vector_created` (11)
  - `council_critique` / `spec_amended` / `dq_breach` (low count,
    high signal)

  Pre-conditions are validated at the emit boundary: subject_id must
  be registered, artifact paths must exist on disk, summary ≤ 400
  chars, parent_event_ids honored for lineage. Direct writes to
  `events.jsonl` are forbidden — use `emit.*` helpers. **One emit
  per research-state-changing commit is doctrine** (honor-system for
  now; pre-commit hook validation planned).

- **Per-family belief context** (`engine/research/belief_synthesis_context.py`):
  the synthesizer (Sonnet) sees, before generating a new candidate,
  the belief layer's per-family verdict distribution. 12 families,
  82 observations, GREEN/MARGINAL/RED ratio per family. Directional
  hints (EXPLORE / AVOID / MARGINAL-ONLY / MIXED) modulate prompt
  framing. The LLM never proposes a new candidate without knowing
  the historical batting average of its family.

- **intuition_rules.yaml** (`data/research/intuition_rules.yaml`):
  deterministic Python heuristics distilled from verdict history.
  Examples: "spanning_mom verdicts under cost-aware execution survive
  at higher Sharpe thresholds than no-cost", "PEAD-family decays
  ~30%/year post-2010", "carry sleeves require regime-conditional
  sizing". These rules run BEFORE the LLM is called — cheap filter,
  prevents repeating known mistakes.

- **633 ClaimType-tagged papers** (`data/papers_curator/cache_with_claim_type.jsonl`):
  Stage-0 ClaimType router classifies every arxiv/SSRN paper into 8
  classes: FACTOR_HYPOTHESIS / METHODOLOGY / DECAY_STUDY / CAPACITY /
  MICROSTRUCTURE / FACTOR_STRUCTURE / DOMAIN_FACT / OTHER. False-
  positive rate dropped 30% → ~0% after router v2 (2026-06-22 commit
  `9e94ae12`). Downstream synthesis only sees FACTOR_HYPOTHESIS
  papers when looking for new alpha — METHODOLOGY papers go to a
  separate rigor-improvement queue.

- **106 RED lessons** (`data/research_store/red_lessons.jsonl`): every
  RED verdict produces a structured lesson row (mechanism family,
  failure mode, what the rejection signal was, what the literature
  says). Future candidates from the same family see this in their
  synthesis prompt — explicit graveyard-aware proposing.

- **Hypothesis registry** (303 hypotheses, 206 specs, 517 predictions,
  299 verdicts, 101 autopsies). Each hypothesis can be traced back
  through its forward_vector_id → paper_id → claim_type, and forward
  to its dispatched spec_id → verdict_event_id → autopsy. Full
  lineage queryable through the typed API.

### Why this matters vs Kimi+WQ

`SKILL.md` is one markdown file. It accumulates string-level wisdom
("analyst field names are easy to confuse"). It does not let you
ask, "what was the historical pass rate for FACTOR_HYPOTHESIS
papers from journal X tagged as DECAY_STUDY by router v2?" Our event
store lets you. Their approach is fine for a single agent talking
to a single platform; ours is what you need when the agent has to
make routing decisions across multiple pipelines.

The per-family belief context is the harder load-bearing piece.
**Their system asks "is this expression Spectacular?" and learns
field-naming traps. Our system asks "given my historical 16-autopsy
batting average on CROSS_SEC_UNKNOWN family with 8 GREEN / 6
MARGINAL / 2 RED outcomes, what should I propose next?"** That's
the difference between adapting to a platform's interface and
adapting to the underlying research economics.

---

## L3 — Self-correction

> **Their answer to "how does the system get smarter?"**:
> `evolve_skill.py` appends new traps and patterns to `SKILL.md`.
>
> **Our answer**: Belief Layer 5 phases (calibration tracking +
> closed-loop prior) + 6-11 multi-layer epistemic review
> (α/β/γ persona + Devil's Advocate × 2 + 8 statistical tests +
> post-GREEN rigor + external LLM audit + PIT audit + Council
> critique + capability evidence).

### L3.1 — Belief Layer (5 phases)

| Phase | Module | What it does |
|---|---|---|
| 1 | `engine/research/belief.py` | **predict-commit air-gap**: every dispatch emits a predicted verdict distribution (GREEN/MARGINAL/RED) BEFORE the strict-gate logic runs. Predictions live in `predictions.jsonl`, NOT in `events.jsonl`. Lens / strict_gate / template / dispatcher code MUST NOT import from predictions. The air-gap is at the code level. |
| 2 | `engine/research/autopsy.py` | **autopsy join**: after each verdict, join (prediction, verdict) → `autopsies.jsonl`. One autopsy per dispatch. |
| 3 | `engine/research/belief_track_record_rigor.py` | **calibration surface**: 8 statistical tests run daily (Bootstrap CI on Brier, time-aware fair family-prior baseline, sign test, Per-family CI + Benjamini-Hochberg FDR q=0.10, Mann-Kendall trend test, Hosmer-Lemeshow goodness-of-fit, threshold×alpha sweep, LOOCV ensemble robustness). Output: `belief_track_record_rigor.md` + `.json`. |
| 4 | `engine/research/belief_prior_calibration.py` | **closed-loop prior**: per-family empirical posterior is computed from past autopsies, blended (after W7-v09 correction: w=1.0, pure family-empirical) into the LLM's prior for new dispatches in the same family. |
| 5 | `engine/research/belief_synthesis_context.py` | **track-record-aware synthesizer**: the synthesizer's prompt sees per-family belief depth + direction hints (EXPLORE / AVOID / MARGINAL-ONLY / MIXED). |

**Current published numbers** (n=94 autopsies):

- Predictor Brier: 0.374 (95% CI [0.334, 0.415])
- Random 3-class baseline: 0.444 (predictor beats this, p<0.001)
- **Time-aware fair family-prior baseline: 0.260**
- **Honest negative finding: predictor LOSES to family-prior by +0.114
  Brier (95% CI [+0.054, +0.173], strictly excludes zero)**
- Hosmer-Lemeshow: REJECTED at p=0.047 (predicted probabilities not
  well-calibrated even though aggregate Brier is)
- W7-v09 self-correction: in-sample ensemble Brier 0.246 → LOOCV
  revealed 0.278 (overfit +0.018) → reverted to pure family-empirical
  (w=1.0 globally) → published Brier 0.260. The W7-v09 commit message
  is literally a public self-correction.

### L3.2 — Multi-layer epistemic review (6-11 layers)

Per Pattern 5 ban (CLAUDE.md, Tetlock 2017 fake-diversity): we do
NOT run N personas debating the same hypothesis. We DO run N
specialist agents with distinct epistemic lenses sequentially. The
distinction matters: 5 LLMs all asked "will this work?" converge to
the same prior (fake diversity, fake n_trials inflation). 5 specialist
agents each asking a DIFFERENT question (real diversity) genuinely
add independent information.

| # | Layer | When | What it asks |
|---|---|---|---|
| L1 | α pre-mortem (`engine/agents/persona/alpha_skeptic.py`) | before dispatch | "How will this factor die?" (failure-mode enumeration) |
| L2 | β cross-domain (`engine/agents/persona/beta_cross_domain.py`) | before dispatch | "Does this mechanism work in other asset classes?" |
| L3 | γ replication (`engine/agents/persona/gamma_replication.py`) | before dispatch | "Can the original paper's result be replicated?" |
| L4 | Devil's Advocate (pre) (`engine/agents/devils_advocate.py`) | before strict gate | Veto demonstrably weak candidates (saves ~$0.005 + 60s per skipped candidate) |
| L5 | 8 statistical rigor tests (`engine/research/belief_track_record_rigor.py`) | during dispatch | Bootstrap CI / Hosmer-Lemeshow / BH-FDR / LOOCV / Politis-Romano / Jobson-Korkie / Mann-Kendall / Bailey-López de Prado DSR |
| L6 | Devil's Advocate (post) | after strict gate | Refute false-positive verdicts |
| L7 | Council critique | high-impact decisions | Multi-perspective architecture review; emitted as `council_critique` event |
| L8 | Post-GREEN rigor (`engine/research/post_green_rigor.py`) | after GREEN verdict | DEAD_POST_PUB (McLean-Pontiff 2016 post-publication decay) / SUBSUMED / SHORT_FEE_KILLS |
| L9 | External LLM audit (`engine/research/external_audits.py`) | periodic | Cross-vendor LLM (DeepSeek / Gemini) reviews the verdict — anti-monoculture |
| L10 | PIT audit (`engine/research/pit_audit.py`) | continuous | Look-ahead leakage detection on the entire deployed book |
| L11 | Capability evidence sign-off | every GREEN verdict | Mandatory `docs/capability_evidence/<spec_id>_<date>.md` file + `capability_evidence_filed` event before deployment |

A typical GREEN verdict passes through L1-L5 + L8-L11 (9 distinct
review functions). A typical RED verdict passes through L1-L5
(strict-gate vetoes it before the post-gate layers fire). The
multi-layer pass is **NOT** "5 LLMs voting" — it's 6-11 different
questions asked in sequence, with the next layer's prompt seeing
the prior layer's structured output.

### Why this matters vs Kimi+WQ

The article describes `evolve_skill.py` writing failures back to
`SKILL.md`. That's L3.0 — feedback exists. What it does NOT
describe:

- No calibration tracking (no Brier scoring of the LLM's predicted
  pass rate vs realized pass rate; the LLM's "long memory" is the
  string-level SKILL.md, not a quantitative track record)
- No multi-layer epistemic review (the LLM proposes, BRAIN scores,
  loop closes; nothing in between asks "how will this fail?" or
  "what's the published decay?" or "is the Sharpe robust to
  out-of-sample?")
- No honest negative finding (the article reports 24 passed / 20+
  Spectacular; it does not report "the Kimi predictor's pass-rate
  prediction was X, the actual pass rate was Y, here's the
  calibration gap")

These aren't oversights to fix later — they reflect that **the
target audience is different**. The Kimi+WQ article sells
"3-day autonomous factor mining" to a tech-press / retail-trading
audience. Our system is built for the audience that asks
"is the LLM calibrated?" — academic reviewers, AI safety teams,
institutional risk committees.

---

## What this enables that Kimi+WQ does not

1. **Publishable academic claims** with confidence intervals on
   every headline number. The arxiv preprint (`docs/arxiv_preprint_
   draft_2026-06-22.md`) leads with the honest negative finding —
   the kind of result most labs would not publish.

2. **Capital decisions stay human-gated**. BRAIN approval → ongoing
   alpha income means there is money flow inside the LLM loop. Our
   approval queue (`/approvals` page) requires explicit human
   review before any capital allocation change. This is doctrine.

3. **Substrate resilience**. WorldQuant BRAIN could deprecate the
   API; Gemini CLI shut down service was their own example. Our
   substrate (WRDS data + own Python pipeline) is reproducible
   from the code in this repo plus a WRDS subscription.

4. **Role-classified diversification**. We can deploy insurance
   sleeves (TLT/GLD crisis hedge) with negative-Sharpe expectation —
   a BRAIN submission of that factor would be rejected outright
   for low Sharpe, missing the entire purpose of the hedge.

5. **Audit trail for governance**. SEC 17a-4-style records require
   immutable, timestamped, queryable history. The typed event
   store satisfies that. `SKILL.md` does not.

---

## Reproducibility

| Claim | Reproducer command |
|---|---|
| Belief Layer Brier 0.374 + honest negative | `python scripts/reports/report_belief_track_record_rigor.py` |
| Per-family ensemble sweep + LOOCV | `python scripts/reports/report_belief_ensemble_sweep.py` |
| ClaimType router false-positive rate | `python -m pytest tests/test_claim_type_router.py` |
| Deployed book Sharpe 1.32 attribution | `python scripts/reports/report_deployed_book_attribution.py` |
| Workflow trace 8-stage counts | `curl http://localhost:8000/api/research/workflow/counts` |
| Belief calibration API | `curl http://localhost:8000/api/research/belief/calibration` |

---

## Acknowledgment

This document was prompted by the QuantML 2026-06-23 article on the
Kimi + WorldQuant BRAIN SKILL. That work is solid agentic-factor-mining
engineering and the three-layer framing it uses (API manual / experience
library / self-evolution) is a useful organizing principle we adopt
here for readability. The differentiating design decisions described
above were made independently as part of MacroAlphaPro between 2026-04
and 2026-06 and are documented in the per-commit message history of
this repository.
