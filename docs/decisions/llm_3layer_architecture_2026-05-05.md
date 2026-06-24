# LLM Three-Layer Architecture — Project Invariants (2026-05-05)

| Field | Value |
|---|---|
| Status | 🟢 ACTIVE — Project-level architecture invariants |
| Date | 2026-05-05 |
| Trigger | Supervisor critique session 2026-05-05: "如果 LLM 在项目里发挥不了太大作用，HITL 是 ceremony；但纯硬编码又和普通量化无差异" |
| Related sprint | B-pragmatic-v2 (A: HITL slim + S6: anomaly_screener) |
| Sibling docs | [hitl_architecture_audit_2026-05-05.md](hitl_architecture_audit_2026-05-05.md) (D2) |
| Pre-registration | SpecRegistry entry `arch.llm_3layer.v1` (pending D3) |
| Amendment ledger | First entry on initial commit |

---

## 1. Problem Statement

The project has accumulated 7 LLM-as-X falsifications:

1. narrative_risk_gate D1 (soft reject, regime artifact)
2. narrative_risk_gate D1.1 (same artifact, retry)
3. narrative_overlay Phase 0 (B-C ≈ 0 over 60-mo + 89-mo)
4. factor_mad LLM factor mining (0/24 candidates passed)
5. EFA three-piece quant uplift (Sharpe -0.174 vs baseline 0.236)
6. S1 multi-window self-falsification (mean Sharpe -0.06 over 6×5y windows)
7. B++ marginal verdict (QL01 +0.985 t=+2.31 fails BHY FDR over N=40)

After 7 rejections, the project's LLM contribution is unclear. Two failure modes loom:

- **Failure mode A — LLM as ornament**: LLM components remain (macro_research / memory_curator / paper_trading E LLM arm) but contribute no measurable value. HITL approval workflow becomes ceremony — supervisor stamps deterministic quant signals.
- **Failure mode B — Red line drift**: Without explicit architectural constraint, ad-hoc LLM additions creep into evaluation/audit layers, breaking the 0-LLM-in-evaluation invariant we have informally maintained.

This document fixes the architectural constraints so future LLM additions have a structural test, not a case-by-case judgment.

---

## 2. Three-Layer Architecture

All LLM usage in this project must be classified into exactly one of three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — GENERATION / EXPLORATION                              │
│  Divergent thinking; hypothesis / case / draft synthesis         │
│  ──────────────────────────────────────────────────              │
│  LLM: ALLOWED ✅                                                  │
│  Examples:                                                       │
│    - Anomaly case generation (S6 candidate component)            │
│    - Macro brief draft (macro_research_agent)                    │
│    - Reflection narrative draft (S2 reflection memory)           │
│    - Hypothesis brainstorm (rejected: S7 Hypothesis Generator)   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ (output flows to Layer 2)
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — EVALUATION / JUDGMENT                                 │
│  Convergent decisions; scoring, ranking, verdict, audit gate     │
│  ──────────────────────────────────────────────────              │
│  LLM: BANNED ❌  (per Zheng 2023 LLM-as-judge red line)          │
│  Examples (must be deterministic):                               │
│    - Brier scoring on macro forecasts                            │
│    - HARKing 4-rule detection                                    │
│    - EFFECTIVE_N_TRIALS counter                                  │
│    - Anomaly precision/recall verdict (M1 in S6)                 │
│    - Pass/fail verdict on falsification tests                    │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼ (verdict flows to Layer 3)
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — PERSISTENCE / AUDIT                                   │
│  Hash chains, spec_hash, amendment ledgers, decision logs        │
│  ──────────────────────────────────────────────────              │
│  LLM: BANNED ❌  100% deterministic                              │
│  Compliance basis: SEC 17a-4(b) / GIPS 2020 / López de Prado §10 │
│  Examples:                                                       │
│    - SHA-256 narrative hash chain                                │
│    - SpecRegistry spec_hash                                      │
│    - Amendment ledger immutable append                           │
│    - DecisionLog field freezing                                  │
└─────────────────────────────────────────────────────────────────┘
```

Layer assignment is mandatory at component design time. Any new LLM-using component must declare its layer and pass review.

---

## 3. Three Invariants

### Invariant 1 — Layer Separation Red Line

**Rule**: LLM is allowed only in Layer 1. Layer 2 and Layer 3 must be deterministic.

**Why**:
- Layer 2 with LLM = LLM-as-judge, which has documented reproducibility / verbosity bias (Zheng 2023 *Judging LLM-as-a-Judge*; Liu et al. 2023 *Geval*). Audit reproducibility (SEC 17a-4(b)) is impossible with stochastic judges.
- Layer 3 with LLM = audit trail can drift between reads. GIPS 2020 §III.A.18 requires deterministic record reconstruction.

**How to apply**:
- New LLM component design: declare layer first, before implementation.
- If component spans layers (e.g., LLM detects + LLM scores), split into deterministic boundaries.
- Existing components must be retroactively classified (see §5).

**Anti-pattern caught by this invariant**:
- LLM-as-judge for forecast accuracy (we banned this; Brier scoring is Layer 2 deterministic)
- LLM rewriting decision narratives between reads (Layer 3 must freeze on first write)

---

### Invariant 2 — Model Capability ≠ Task Value

**Rule**: When proposing to upgrade an LLM model (e.g., Flash → Pro, GPT-5 → GPT-6), the proposal must trace the capability improvement to a specific, project-adopted sub-task. Generic "X is stronger on benchmark Y" is insufficient.

**Why**:
- The project has rejected 7 LLM-as-X paths. "Bigger model is better at X" does not resurrect rejected X.
- Example caught 2026-05-05: claim "Pro is better at cross-market reasoning" — but cross-market reasoning is a rejected alpha path. Pro being stronger on it produces no marginal value to us.
- This invariant prevents budget creep on capability that does not map to project tasks.

**How to apply** — two-step gate:
1. **Task gate**: List the specific sub-task (anomaly pattern matching / macro brief / reflection draft). Verify it is in the project-adopted task list. If not in list (or in rejected list), upgrade is rejected.
2. **Capability gate**: For each task in (1), cite specific evidence that the larger model is meaningfully better on this sub-task (not generic benchmark). If no task-specific evidence, upgrade is rejected.

**Anti-pattern caught**:
- "GPT-7 has 20% higher MMLU" — irrelevant unless MMLU correlates with anomaly detection (it does not for our task profile)
- "Claude Opus is better at long-context" — irrelevant if our prompts are <50k tokens
- "Pro is better at reasoning" (the 2026-05-05 case) — irrelevant if reasoning task is rejected path

---

### Invariant 3 — LLM Evaluation Requires ≥1 Supervisor-Independent Metric

**Rule**: Any LLM component evaluation must include at least one objective metric that does not depend on supervisor labeling.

**Why**:
- Supervisor labels suffer well-documented bias (Liu et al. 2023; Zheng 2023). Using only supervisor judgment as ground truth = circular evaluation.
- Forward-only deterministic metrics (e.g., did the flagged ticker have a >2σ event in the next 5 days?) are immune to supervisor bias.
- Triangulation: supervisor metric (M2) and objective metric (M1) together produce 4 verdict quadrants:
  - M1 high + M2 high → real win, ship
  - M1 high + M2 low → LLM accurate, supervisor doesn't trust → UX/communication problem (fixable)
  - M1 low + M2 high → LLM is hallucinating persuasively → DANGEROUS, kill
  - M1 low + M2 low → not useful, kill

**How to apply**:
- Pre-register objective metric definition (threshold, time window, event class) BEFORE running evaluation.
- Threshold and window are part of spec_hash; changing them mid-eval = HARKing detected.
- Objective metric data source must be deterministic (price-based, SEC filing-based, etc.) — not another LLM.

**Anti-pattern caught**:
- Asking supervisor "did the LLM get this right?" as sole verdict → circular
- Using one LLM to score another LLM's output → circular (also violates Invariant 1)

---

## 4. S6 Anomaly Screener — Spec Preview

S6 implements anomaly detection in compliance with all three invariants:

| Component | Layer | Justification |
|---|---|---|
| LLM scans portfolio + news → generates flag candidates | Layer 1 | Generation; allowed |
| Rule-based detector generates flag candidates (parallel) | Layer 1 | Generation; deterministic but Layer 1 |
| Forward-window event verification (was there a >2σ move in K days) | Layer 2 | Evaluation; deterministic only |
| Composite verdict (LLM > rule by +5pp on M1 AND M2 > 30%) | Layer 2 | Evaluation; deterministic |
| Hash chain of flag + verdict + supervisor decision | Layer 3 | Persistence; deterministic only |

**Three invariants check**:
- Invariant 1: ✅ LLM only in Layer 1; verdict and persistence deterministic
- Invariant 2: ✅ Model = Gemini 2.5 Flash; sub-task = pattern matching (LLM strength); not upgraded to Pro because no project task requires Pro
- Invariant 3: ✅ M1 (objective hit rate) is supervisor-independent; M2 (acceptance) is the supervisor metric; both pre-registered

Detailed S6 spec lives in D2 + D4 implementation. This document fixes only the architectural constraints.

---

## 5. S7 Hypothesis Generator — Rejected

S7 was proposed (LLM autonomously generates new factor/signal hypotheses, runs through pre-registration, accumulates 8 cycles in 4 months). REJECTED on 2026-05-05 after self-audit revealed 10 risks, 2 of which are critical-unfixable:

### Critical risk 1 — LLM cutoff lookahead in hypothesis generation
LLM training cutoff = 2026-01. Asked to generate "novel" hypotheses, LLM is retrieving / recombining factor literature it has memorized (McLean-Pontiff 2016 catalogue, Harvey-Liu 2014 zoo, Asness corpus). "Novel" hypotheses contaminated with hindsight on which factors decayed. No clean fix in 4-month window — only post-cutoff forward data is uncontaminated.

### Critical risk 2 — Statistical power 25% on 8 cycles
For binomial test detecting LLM > random baseline at 50% vs 70% lift, 80% power requires ~30+ trials. 4-month cycle budget = 8 cycles. Power ≈ 25%. Even positive verdict would be statistically inconclusive.

### Additional high-severity risks (8)
- Sakana AI Scientist (Lu 2024) precedent: similar autonomous research claim received heavy academic critique for slop generation. Defense exposure.
- FunSearch (Romera-Paredes 2024) domain mismatch: math combinatorics ≠ noisy non-stationary finance.
- HARKing through hindsight in self-evolution loop (LLM avoids past failures, approaches past successes — meta-level hypothesis selection).
- Implementation cost realism: 60-80h estimated, honest revision 100-150h. Master's deadline incompatible.
- Brand risk amplification: "Autonomous quant research agent" reads AGI-adjacent.
- EFFECTIVE_N_TRIALS inflation: +8 spec entries → tighter BHY FDR thresholds for all other strategies.
- paper_trading E resource conflict: 24-month locked spec; new hypotheses may overlap.
- Random baseline definition unclean: factor zoo is itself cutoff-contaminated.

### Decision

S7 is rejected for the master's project deadline (2026-09). May be re-proposed post-defense as PhD-track work with 12-24 month horizon and strict post-cutoff-only forward data.

---

## 6. Retroactive Applicability — Existing LLM Components

This invariant set applies retroactively. Existing LLM components must be classified:

| Component | Current usage | Layer | Status |
|---|---|---|---|
| `macro_research_agent` | Generates regime forecasts + Brier-scored | Layer 1 (generation) + Layer 2 (Brier deterministic) | ✅ Compliant — Brier is deterministic |
| `memory_curator` (function-agent) | Monthly summary reports | Layer 1 only | ✅ Compliant |
| S2 reflection memory loop | LLM drafts CONTEXT/DECISION/OUTCOME/LESSON | Layer 1 (draft) + Layer 2 (deterministic structure validation) | ✅ Compliant — structure validation is non-LLM |
| `paper_trading E` LLM arm B | Sector_pipeline LLM ablation | Layer 1 only (decisions go to deterministic backtest) | ✅ Compliant |
| `decision_context.compose_supervisor_narrative` | Deterministic templating | Not LLM at all | ✅ Compliant by absence |
| Hash chain narrative snapshots | SHA-256 of frozen narrative | Layer 3 deterministic | ✅ Compliant |

**No retroactive violations identified at initial classification.** This will be re-audited during Tier 1 retroactive audit (Claim 2 in audit roster — see audit pending todo).

---

## 7. Forward Compatibility

Future LLM additions must:
1. Pre-declare layer assignment in spec doc
2. Pass three-invariant gate (layer separation / model upgrade gate / supervisor-independent metric)
3. Add SpecRegistry entry referencing this invariant doc
4. Append to amendment ledger if modifying existing component's layer

Failure to comply = component rejected, regardless of perceived value.

---

## 8. Lessons Encoded

### Lesson 1 — "好用 ≠ 任务有用"
Bigger / better / faster / cheaper LLM is irrelevant unless mapped to a project-adopted sub-task. The 2026-05-05 caught case (Pro for "cross-market reasoning") demonstrates how easily capability improvements get smuggled past task gates without explicit invariant enforcement.

### Lesson 2 — "评估闭环必须有 supervisor-independent 通道"
A 7-falsification project should not introduce an 8th risk through circular supervisor-only evaluation. Forward-deterministic metrics (event verification, price moves, SEC filings) are the project's ground truth.

### Lesson 3 — "层切割比红线更稳定"
"0 LLM in evaluation" stated as a flat red line invites case-by-case argument. Three-layer split provides a positive structure (Layer 1 LLM may, Layer 2-3 must not) that is easier to defend at design review.

---

## 9. References

**Academic**:
- Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Liu et al. 2023, *Geval: NLG Evaluation using GPT-4*
- Tian et al. 2023, *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores*
- Lu et al. 2024, *The AI Scientist* (Sakana AI; cited as cautionary precedent)
- Romera-Paredes et al. 2024, *Mathematical discoveries from program search with large language models* (FunSearch; domain mismatch case)
- McLean & Pontiff 2016, *Does Academic Research Destroy Stock Return Predictability?*
- Harvey & Liu 2014, *Backtesting*
- Lakatos 1970, *The Methodology of Scientific Research Programmes*

**Compliance**:
- SEC 17a-4(b) — Electronic record retention requirements
- GIPS 2020 §III.A.18 — Composite supervisor verification
- López de Prado 2018, *Advances in Financial Machine Learning* §10 hash chain methodology

**Project internal**:
- [s2_reflection_memory_evidence.md](s2_reflection_memory_evidence.md)
- [s3_pre_registration_enforcement_evidence.md](s3_pre_registration_enforcement_evidence.md)
- [paper_trading_e_v0_2_redesign.md](paper_trading_e_v0_2_redesign.md)
- All 7 rejected docs in this archive

---

## 10. Amendment Ledger

| Date | Change | Author | spec_hash before | spec_hash after |
|---|---|---|---|---|
| 2026-05-05 | Initial commit; 3 invariants codified | zhangxizhe (supervisor) | (n/a — first commit) | TBD on D3 SpecRegistry insert |

Future amendments to this document require:
- Reasoning logged here
- Old spec_hash recorded
- New spec_hash computed deterministically from this file content
- HARKing detector R3 check passed (no post-hoc threshold tweak)
