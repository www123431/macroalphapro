# Decision Archive — Macro Alpha Pro

Honest research log of LLM-as-alpha hypotheses tested + rejected + redesigned.
Each doc records: hypothesis, method, evidence, reject/redesign reasoning, lessons.

---

## Active

| Doc | Status | One-liner |
|---|---|---|
| [paper_trading_e_v0_2_redesign.md](paper_trading_e_v0_2_redesign.md) | 🟢 ACTIVE (demo) | Sector_pipeline LLM debate ablation; 3-arm forward paper trading w/ placebo control |
| [s2_reflection_memory_evidence.md](s2_reflection_memory_evidence.md) | 🟢 INFRA PASS / ACCUM PENDING | **S2 Self-reflecting agent memory loop** — 7-layer infra (ORM + LLM generator + RAG retriever + sector/macro hooks + backfill + UI) end-to-end verified; Reflexion + Generative Agents + Voyager 文献合成；32/32 verification facets PASS；spec ≥50 reflections by 2026-09 calendar-bound |
| [s3_pre_registration_enforcement_evidence.md](s3_pre_registration_enforcement_evidence.md) | 🟢 CAPABILITY PASS | **S3 Pre-Registration Enforcement** — spec_hash + amendment ledger + 4-rule HARKing detection (R1-R4) + EFFECTIVE_N_TRIALS dynamic integration + UI panel + cron hook；零 LLM；31/31 verification facets PASS；首日实战已用（SSRN AI 披露 amendment 走完整 workflow） |
| [p_fund_performance_reporting_evidence.md](p_fund_performance_reporting_evidence.md) | 🟢 CAPABILITY PASS | **P-FUND Investor-Grade Performance Reporting** — GIPS 2020-compliant TWR/MWR/HPR triple-method + supervisor deposit/withdraw + Modified Dietz Bacon Ch.2 KAT ±0.01% + live_dashboard 投资人视图集成；零 LLM；35/35 verification facets PASS；项目第一次 S3 forward registration |
| [p_audit_supervisor_panel_evidence.md](p_audit_supervisor_panel_evidence.md) | 🟢 CAPABILITY PASS | **P-AUDIT v1 Supervisor Approval Audit Panel** — Tier 2 deterministic context (8 modules) + 3a Replay viewer + 3b RAG-hybrid similar approvals + 3d cross-time analytics page；强制 review_rationale (≥10) + 6-enum category；零 LLM；12/12 facets PASS；项目第二次 S3 forward registration；3c Monte Carlo simulator 主动剔除（5 reasons documented） |
| [llm_3layer_architecture_2026-05-05.md](llm_3layer_architecture_2026-05-05.md) | 🟢 INVARIANT | **LLM Three-Layer Architecture** — Layer 1 generation (LLM allowed) / Layer 2 evaluation (LLM banned) / Layer 3 audit (deterministic). 3 invariants codified (layer separation / model capability ≠ task value / supervisor-independent metric requirement); S7 Hypothesis Generator REJECTED with 10 documented risks; retroactive applicability passed (existing macro_research / S2 reflection / paper E LLM all compliant) |
| [hitl_architecture_audit_2026-05-05.md](hitl_architecture_audit_2026-05-05.md) | 🟢 INVARIANT | **HITL Architecture Slim Refactor** — 9-issue audit + 5 governance categories (cash_flow / spec_amendment / risk_control / strategy_arm_toggle / universe_change) + 1 LLM-output (anomaly_screener post-S6) + routine_review (entry/rebalance auto-execute audit trail). Operations 3-tab UI (Governance / Routine Timeline 30d / Ergonomics). Trade-level approvals removed (70% ceremony cut); GIPS / Hansen / Knight Capital references for retained 4 cats |
| [s6_anomaly_screener_spec_2026-05-05.md](s6_anomaly_screener_spec_2026-05-05.md) | 🟢 ACTIVE — FORWARD TEST | **S6 Anomaly Screener** — Pre-registered 90d forward test (Slimmed-corrected ~48h impl). Three detectors: rule_baseline_a (5 rules) / rule_baseline_b (+ macro forecast) / llm (Gemini 2.5 Flash temp=0 thinking=5000). M1 precision + recall + F1 (no calibration/Brier/ROC AUC); M2 supervisor accept rate; M3 case study. Composite verdict: CLEAR_WIN / CLEAR_LOSS / INCONCLUSIVE / CATASTROPHIC. SpecRegistry id=23 hash cf305466d7a78289 +1 EFFECTIVE_N_TRIALS. Cost ceiling $250 / 90d. First LLM-positive evidence candidate; 4 verdict thesis chapters pre-committed |
| [tier1_retroactive_audit_2026-05-05.md](tier1_retroactive_audit_2026-05-05.md) | 🟢 ACTIVE — AUDIT | **Tier 1 Retroactive Audit** — Self-adversarial validation pass on 3 thesis-critical claims (7 falsifications power / 0-LLM-in-evaluation red line / spec_hash chain bypass). 47 PASS / 6 WARN / 0 FAIL after remediation. Reproducible via [scripts/tier1_retroactive_audit.py](../../scripts/tier1_retroactive_audit.py). Caught and remediated: (a) engine/lcs.py LLM-as-judge legacy → deprecated to no-op via B-revised after portfolio.py inspection showed C-refactor mathematically vacuous on `sign(raw_return)` rule; (b) spec_ui_redesign.md hash drift → fixed via amend_spec id=20 (+0 trials clarification). |
| [b_plus_prod_migration_2026-05-05.md](b_plus_prod_migration_2026-05-05.md) | 🟢 ACTIVE — PRODUCTION | **B-PLUS-PROD: TSMOM → QL01 BAB Migration** — Production strategy switched from TSMOM(12,1) (S1 self-falsified) to QL01 BAB (Frazzini-Pedersen 2014). B++ Mass FDR Tier 1 OOS Sharpe +0.985 NW t=+2.312 raw 5% sig (BHY FDR over N=40 fail = MARGINAL). Literature-conditional ship rule introduced (≥10y external lit support exempts strict BHY denominator). spec amendment id=6 (kind=hypothesis_amend). PaperTradingRun.signal_baseline column tags pre/post migration runs. Self-correction of prior "demonstrative baseline" recommendation per supervisor challenge. |
| [meta_audit_kill_simplify_2026-05-05.md](meta_audit_kill_simplify_2026-05-05.md) | 🟢 ACTIVE — CLEANUP | **Meta-Audit Kill & Simplify** — Post-Tier 1 + B-PLUS-PROD cleanup. 6 components reviewed: KILL macro_research weekly pipeline + paper trading E monthly hook (evaluation theater after REGIME_SCALE=1.0 + S6 supersedes); DOCUMENT SkillLibrary 0-row state + regime overlay dead-branch flag + regime limits binding under QL01 BAB; FALSE POSITIVE on FRED claim (genuinely used). Saves $50-100/yr LLM + 12h supervisor over 24 months. Second self-correction reversing prior "capability demo reframe" recommendation. |

## Rejected — LLM-as-Alpha (4) + Quant uplift (1) + Self-falsification (1)

| Doc | Status | Year/Quarter | Verdict |
|---|---|---|---|
| [narrative_risk_gate_d1_soft_rejected.md](narrative_risk_gate_d1_soft_rejected.md) | ❌ REJECTED | 2026-05 | LLM macro narrative → vol gate; soft reject (regime artifact inflation) |
| [narrative_risk_gate_d1_1_rejected.md](narrative_risk_gate_d1_1_rejected.md) | ❌ REJECTED | 2026-05 | D1.1 retry; same artifact |
| [narrative_overlay_phase0_rejected.md](narrative_overlay_phase0_rejected.md) | ❌ REJECTED | 2026-05 | LLM narrative → cross-sectional sector tilt; B-C ≈ 0 in 60-mo + 89-mo backtests |
| [factor_mad_reject.md](factor_mad_reject.md) | ❌ REJECTED | 2026-05 (Q1) | LLM factor mining; 0/24 candidates passed multiple-testing-aware critic |
| [three_piece_uplift_efa_reject.md](three_piece_uplift_efa_reject.md) | ❌ REJECTED | 2026-05 | Quant strategy uplift (universe expansion + asset-class short cap + TSMOM ensemble); FAIL on 5-y window (Sharpe -0.174 vs baseline 0.236) |
| [s1_multi_window_evidence.md](s1_multi_window_evidence.md) | ❌ REJECTED (own baseline) | 2026-05 | **Self-falsification**: TSMOM + composite signal across 6 × 5-yr windows 2010-2024; Mean Sharpe -0.06, 2/6 positive, 0/6 5%-sig, bootstrap CI crosses zero; regime-specific QE-era alpha decay |
| [b_plus_mass_search_evidence.md](b_plus_mass_search_evidence.md) | 🟡 MARGINAL | 2026-05 | **B++ Mass FDR**: 20 strategies × 2 universe tiers, weekly OOS 2018-2024; QL01 (Frazzini-Pedersen BAB) Sharpe +0.985, t=+2.31, β-neutral confirms pure factor alpha; fails BHY FDR over N=40; 38/40 negative — methodology + replication contribution |

## Pipeline-Ready (deferred)

| Doc | Status | Note |
|---|---|---|
| [narrative_overlay_phase0_pipeline_ready.md](narrative_overlay_phase0_pipeline_ready.md) | ⏸ DEFERRED | Phase 1 LLM stage gated on Phase 0 pass; Phase 0 rejected → never started |

## Rejected Specs (archived)

| Doc | Note |
|---|---|
| [rejected/spec_factor_mad_redesign.md](rejected/spec_factor_mad_redesign.md) | 2026-05-02 5-Sprint redesign spec; module deleted in cleanup |
| [rejected/spec_narrative_overlay_phase0.md](rejected/spec_narrative_overlay_phase0.md) | LP-IRF narrative overlay spec; module deleted in cleanup |
| [rejected/spec_narrative_risk_gate_d1.md](rejected/spec_narrative_risk_gate_d1.md) | D1 narrative-driven vol gate spec; module deleted |
| [rejected/spec_narrative_risk_gate_d1_1.md](rejected/spec_narrative_risk_gate_d1_1.md) | D1.1 retry spec; same fate |

---

## Honest Framing

### What this archive is

A **falsification history** — each rejected hypothesis was tested against pre-registered criteria, found wanting, and documented before deletion. The spec docs in `rejected/` preserve method-level transparency even though their corresponding code is deleted.

### What this archive is NOT

- **Not a backlog of future work.** Rejected = rejected. Don't restart unless a structurally different hypothesis emerges.
- **Not an apology.** Rejecting failed LLM alpha hypotheses is **the project's research rigor signal**, not a deficiency. 4 reject + 1 active is healthier than 5 active without verdict.

### Cross-cutting lessons

1. **LLM in macro alpha is structurally hard at master's-project scope** (Gemini-1.5 + public macro/news context + monthly rebalance). Lopez-Lira & Tang (2023) and Bybee (2023) literature priors hold up; our experiments are not anomalies.
2. **Placebo arm (Arm C) is sine qua non**. B-C ≈ 0 in narrative phase 0 was the strongest reject signal — stronger than ΔSharpe-vs-baseline.
3. **Power analysis precedes spec lock**. The paper_trading_e v0.1 → v0.2 redesign was triggered by realizing v0.1's 24-month verdict had power 22-41% on plausible LLM hit-rates.
4. **Pre-registration with hardcap matters**. Lakatos research programmes need falsifiability boundaries; without them, "just need more data" becomes degenerative.
5. **Pre-registration discipline applies to quant uplifts too, not just LLM**. The EFA three-piece uplift was rejected at FAIL verdict; defaults reverted without cherry-picking which component was "actually fine". Same standard as LLM hypothesis testing.
6. **Pre-registration discipline applies to your own baseline (deepest lesson)**. S1 multi-window analysis revealed earlier single-point Sharpe estimates (0.236 / 0.510 / 0.703) were measurement noise on a borderline-marginal window — the strategy's true generalization across 14 years is statistically indistinguishable from zero. Self-falsification of one's own work is the highest-grade research-integrity signal, consistent with McLean-Pontiff (2016) anomaly post-publication decay literature.

---

## Reading order for new readers

For interview / thesis context, read in this order:

1. [paper_trading_e_v0_2_redesign.md](paper_trading_e_v0_2_redesign.md) — current state
2. [narrative_overlay_phase0_rejected.md](narrative_overlay_phase0_rejected.md) — most thorough reject doc, full method
3. [factor_mad_reject.md](factor_mad_reject.md) — second reject, illustrates LLM factor mining failure mode
4. [narrative_risk_gate_d1_1_rejected.md](narrative_risk_gate_d1_1_rejected.md) — first reject, brief
