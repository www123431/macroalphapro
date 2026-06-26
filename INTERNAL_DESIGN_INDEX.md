# MacroAlphaPro — Internal Design Index

**Purpose**: This file is the **internal** consolidated design memory — meant for Claude + principal future-session use. PROJECT_OVERVIEW.md is the outside-facing version; this is the inside-facing version.

**Why this exists**: across long sessions (this one had 27+ commits across W6-rigor + W7-arxiv), key design decisions get committed but the rationale lives only in commit messages + scattered memory files. Future-me has limited bandwidth to reconstruct "why we built it this way". This file collects the load-bearing decisions in one place.

**Update protocol**: append new decisions as they happen. Mark deferred work explicitly with a "revisit-when" trigger. Never delete past decisions — strike-through if superseded.

**Last updated**: 2026-06-22 (after session ending in 27 commits W6→W7-arxiv-v09-OOS)

---

## 1. Active Standing Rules (MUST FOLLOW — pre-commit / behavior enforced)

### 1.1 Doctrine layer (CLAUDE.md + memory/feedback_*)

| Rule | Why | Where enforced |
|---|---|---|
| **Pattern 5 ban**: no multi-agent debate | Tetlock fake-diversity finding; n_trials inflation. Single-agent specialists OK (α/β/γ). | `memory/feedback_anti_n_persona_brainstorm_2026-06-14.md` |
| **Sizing-before-Signal**: exhaust HRP / drop-one / vol target / weight grid BEFORE signal-side enhance | 24 enhance tests in 2026-06-17 → 0 IMPROVEMENT. Sizing has 10-30% Sharpe potential vs signal 1-3%. | `memory/feedback_sizing_before_signal_2026-06-17.md` |
| **FORWARD vs ENHANCE statistical separation**: never mix Bailey-LdP DSR with paired bootstrap | Paired SE ~3x tighter at ρ≈0.95. Mixing kills 90%+ real improvements. | `memory/feedback_forward_vs_enhance_statistical_separation_2026-06-11.md` (project-root CLAUDE.md doctrine too) |
| **Strategy_family vs claim_family**: use canonical spec-derived `strategy_family_for_spec` for Bailey-LdP n_trials denominator | Same spec from different paper-tagged hypotheses must share one trial counter. | `memory/feedback_strategy_family_vs_claim_family_2026-06-12.md` |
| **Paper-driven research chain**: PAPER → HYPOTHESIS → TEST → VERDICT MUST trace explicitly, with verbatim chunk_id references | Prevents pretrain-grounded lessons polluting fresh research. | `memory/feedback_paper_driven_research_chain_locked_2026-06-04.md` |
| **LLM credit conservation**: design-audit BEFORE spend; default to cheapest acceptable provider; target ceiling $10/mo | Principal explicit 2026-06-22. Burned $0.15 earlier on misdirected backfill. | `memory/feedback_llm_credit_conservation_2026-06-22.md` |
| **Piece-by-piece commits**: each substantive piece = one commit | Principal explicit 2026-06-05. Stop after >3 substantive commits in a session. | `memory/feedback_piece_by_piece_not_batch_2026-06-05.md` |
| **Drop "agentic" overclaim**: frame as "doctrine-bounded agentic" or "LLM-augmented + measure-defined contribution" | LLM contribution to verdict prediction measured ~10%. Pure agentic narrative is marketing. | this session (paper Section 4.6 + agentic discussion) |
| **start.bat is default launch** | Principal explicit 2026-06-04. | `memory/feedback_default_launch_start_bat_2026-06-04.md` |
| **STREAMLIT DEPRECATED**: never propose | (standing across sessions) | (session memory) |
| **Bilingual UI**: all new `/research/*` strings t()-wrapped (zh + en) in `frontend/lib/i18n.tsx` | Principal explicit 2026-06-06. | `memory/feedback_research_area_must_be_i18n_2026-06-06.md` |
| **Self-audit blind spots**: LLM self-review structurally misses bugs; use external LLM adversarial audit + replication anchor + doctrine-as-test | Principal has caught major bugs Claude missed. | `memory/feedback_self_audit_blind_spots_2026-06-13.md` |
| **No hand-pattern guessing**: templates DECLARE column names in artifacts dict; lenses READ via `engine.research.lens_helpers.resolve_*` | Avoid scan-by-prefix scattered across modules. | `memory/feedback_explicit_artifacts_contract_no_string_pattern_guessing_2026-06-09.md` |
| **FWL sequential residual TRAP**: naive sequential `regress ε₁ on raw X₂` violates Frisch-Waugh-Lovell. α₂ becomes ~0 by OLS construction. Use joint model for α; sequential is REPORTING NARRATIVE only. | Caught 2026-06-09 during PIT SN audit. | `memory/feedback_fwl_sequential_residual_trap_2026-06-09.md` |
| **Anchor panel SEQUENTIAL residual doctrine**: when adding NEW anchor panel (industry/region/style) do SEQUENTIAL residual regression (Stage N peels prior layers) NOT joint multi-panel OLS. Order fixed by canonical precedence — never reorder. | Mirrors Fama-French historical build (CAPM→FF3→Carhart→FF5). | `memory/feedback_anchor_panel_sequential_residual_doctrine_2026-06-09.md` |
| **Random-data test tolerances from theory**: test bands MUST be derived from theoretical SE × critical-z, NOT picked by hand. Multi-comparison correction for K-anchor panels. | Caught after seed-101 false positive at hand-picked tight bound. | `memory/feedback_random_data_test_tolerances_from_theory_2026-06-09.md` |
| **Substrate vs Epistemics vs Taste vs Orchestration**: FIRST classify proposal axis, THEN propose | Caught 2x in 2026-06-11 proposing wrong-axis fix. | `memory/feedback_substrate_vs_epistemics_two_axes_2026-06-11.md` |
| **Don't fallback to "natural cron" when batch possible**: with human_override + sub-period dispatch, 13→30 autopsies in 30min @ ~$0.05 is doctrinally fine. Bailey-LdP governs threshold scaling ONLY, not weekly throughput. | Principal pushed 3× to accelerate verdicts; I lapsed to "natural cron" each time. | `memory/feedback_dont_fallback_to_natural_cron_when_batch_possible_2026-06-14.md` |
| **Anti-mental-rut doctrine**: solve INPUT-side corpus diversification (Semantic Scholar + adversarial author watchlist + SSRN/NBER + forward citation + RED-verdict surface), NOT by softening A/B's HXZ discipline | Empirical backing: McLean-Pontiff 32-58% Sharpe drop from relaxed multi-test. | `memory/project_anti_rut_doctrine_2026-06-07.md` |

### 1.2 Architecture invariants (code-enforced)

| Invariant | Where enforced |
|---|---|
| Belief Layer air-gap: NO `engine.research.belief` import from lens/template/gate | pre-commit grep + `tests/test_belief*` |
| Spec amendment governance: every `docs/spec_*.md` edit must call `engine.preregistration.amend_spec()` | pre-commit hook (active; bit me once 2026-06-22) |
| `factor_verdict_filed` events MUST be emitted via `engine.research_store.emit.factor_verdict` (not raw write) | `_validate_artifacts` requires real paths on disk |
| Subject_id MUST be in `engine.research_store.registry` before emit | `emit.*` checks; helpful "did you mean" suggestions |
| `signal_inputs` MUST prefix-match `PIT_CORRECT_SOURCES` whitelist | `pre_dispatch_check` gate 8 (extended 2026-06-22 with `move.` + `tlt.` for Bond-VRP) |
| `(signal_kind, universe)` MUST appear in some `TemplateContract` | gate 10 (extended 2026-06-22 with `vrp_treasury`) |
| Universe MUST have `_UNIVERSE_DATA_PROBES` entry | gate (extended 2026-06-22 with `us_treasury_options`) |
| LLM calls MUST go through `engine.llm.call` (or be grandfathered in `scripts/check_llm_client.py`) | pre-commit hook; grandfathered list maintained |

---

## 2. Architectural Decision Log (key choices + WHY)

### 2.1 Belief Layer

- **2026-06-11 initial design**: 5 phases (predict-commit / autopsy / calibration dashboard / closed-loop prior / track-record-aware ask). Decision: deferred at the time per memory `belief-layer-5-phases-deferred-2026-06-11`. **STATUS: PARTIALLY STALE.**
- **2026-06-11 to 2026-06-14**: Phases 1, 2, 4, 5 actually shipped (memory note was not updated).
- **2026-06-21 W3**: Phase 3 calibration surface shipped (the missing piece) — `belief_track_record.py` + `report_belief_track_record.py`. n=85 autopsies, mean Brier 0.373.
- **2026-06-22 W6-rigor**: 6 standard stat tests added (bootstrap CI / paired bootstrap baselines / sign test / FDR per-family / Mann-Kendall / Hosmer-Lemeshow). Headline negative finding: LLM-only loses to family-prior by 0.114 Brier.
- **2026-06-22 W6-rigor-A**: predict_verdict params tuned per sweep evidence (N=5→3, α=3→1).
- **2026-06-22 W7-arxiv-v05 sweep**: per-family ensemble blend predicted 0.246 in-sample.
- **2026-06-22 W7-arxiv-v06**: wired ensemble into predict_verdict, flag-gated OFF default.
- **2026-06-22 W7-arxiv-v07**: ACTIVATED (BELIEF_ENSEMBLE_BLEND_ENABLED=True) per principal evidence-based decision.
- **2026-06-22 W7-arxiv-v08 LOOCV**: revealed in-sample overfit (LOOCV 0.278 vs sweep 0.254). Activation still beats LLM by 26%.
- **2026-06-22 W7-arxiv-v09 HONEST CORRECTION**: per-family w_fam → all 1.0 (pure family-empirical when n≥3). Target Brier 0.260. **Current production state.**

### 2.2 Scripts + engine reorganization (W1-W2)

- **2026-06-18 Week 1 Phase 1**: scripts/ skeleton directories created (audits/, cron/, data_fetch/, smoke/, scouts/, reports/, runners/, oneoff/, __archive__/). `_*.py` (leading-underscore) moved to oneoff/.
- **2026-06-18 Week 1 Phase 2**: engine/ tier classification README (CORE / PRODUCTION / RESEARCH boundary marker, no physical moves).
- **2026-06-18 Week 2 Phase 1**: 3-pipeline architectural contract (FORWARD/ENHANCE/PROMOTE) shipped as `engine/research/__pipelines__.md` + thin re-export shims.
- **2026-06-18 Week 2 Phase 2**: 31/33 audit_*.py physically moved to scripts/audits/. 2 deferred (audit_agent_liveness + audit_narrative_chain — hot cron paths require coordinated `auto_audit_rules.py` refactor).

### 2.3 Risk Engine

- **2026-06-18 audit finding**: Risk Engine is **95% already built**. 12-gate Risk Manager (spec id=69) wired into daily paper trade cron. VaR-95, ES-95, killswitch, position caps (Basel-III 2-tier), HHI all live. **W3 Risk MVP plan from original 6-week-critical-path was redundant — pivoted to W3 Belief Phase 3 instead.**

### 2.4 ClaimType Stage 0 router (W4)

- **2026-06-21 W4-piece-1**: deterministic keyword router. Measured 83% UNKNOWN on 633 cached papers. Honest failure verdict.
- **2026-06-21 W4-piece-2**: Deepseek-v4-pro LLM fallback. UNKNOWN 83% → 3.5%. Cost ~$0.15 backfill, ~$0.001/paper ongoing.
- **2026-06-21 W4-piece-3**: wired classify_hybrid into summarize_paper. Every new paper auto-tags going forward.
- **2026-06-22 W6-rigor-A-router-v2**: killed 30% FP rate by removing ambiguous single-word triggers ("anomaly", "premia") + adding multi-word phrases. min_total_score kept at 1.

### 2.5 Cron architecture (W5)

- **2026-06-22 W5-a audit finding**: burndown cron WAS already registered (schtasks `MacroAlphaPro\burndown-daily`, actually Mon+Thu 09:00, not daily). Failures since 2026-06-18 traced to Anthropic balance depleted. **Audit revealed: NOT a missing-cron problem; an API-balance problem.**
- **2026-06-22 W5-a-B**: cron health precheck wrapper (single ~10-token Haiku ping). Fail-closed: precheck non-zero exit → skip burndown. Saves ~$0.30 per dead-API run.
- **2026-06-22 W6-rigor-A-cron-peel**: `daily_belief_refresh` cron registered (06:35 daily). Catches async-emitted verdict events, regenerates track record + rigor reports. $0/day.

### 2.6 Bond-VRP end-to-end demonstration (W4-E2E → W6-rigor-A loop closed)

- **2026-06-22 W4-E2E**: synthesizer autonomously produced 2 hypothesis candidates (Bond-VRP + Treasury duration-convexity). First end-to-end autonomous synthesis-to-hypothesis chain demonstrated.
- **2026-06-22 W6-rigor-A-validate-loop-closed**: built `vrp_treasury.py` template (MOVE + TLT, MVP). Verdict on Bond-VRP: RED (Sharpe +0.56, NW-t +1.69 < threshold 2.09). **Consistent with Carr-Wu 2009 equity-only VRP literature.**
- **2026-06-22 W6-rigor-A governance chain**: extended 3 governance layers to register the new (signal_kind=vrp, universe=us_treasury_options) combination. Each gate was a real defense, not theater.

---

## 3. Past Iterations / Pivots (LESSON-LEARNED)

| Pivot | What happened | Lesson |
|---|---|---|
| **Original W3 = Risk Engine MVP** → pivoted to **Belief Phase 3** | Read-before-build revealed Risk Engine already 95% done | "Read before build" is doctrine — saved 6+ hours and prevented redundant work this session |
| **W7-arxiv-v07 per-family activation** → **v09 pure family-empirical** | LOOCV revealed per-family w_fam optimization overfits at n=92 | The self-tuning loop IS the demonstration; honest iteration is the value, not perfect first-shot |
| **"agentic AI research OS" framing** → **"LLM-augmented bounded-autonomy workbench"** | Principal challenged the marketing frame; data showed LLM contribution ~10% | Frame to measurement, not aspiration |
| **2026-06-17 24 enhance tests, 0 IMPROVEMENT** | Burned commits on signal-side enhance without exhausting sizing | **Sizing-before-Signal doctrine** locked |
| **Sharpe 1.32 claim in v0.1 abstract** | Audit found I'd misremembered "6 months" — replay is 486 weeks (~9.4 years). Also found TWO parallel "deployed book" definitions (4-sleeve canonical + 5-sleeve research variant) | Cite the JSON source path in every numeric claim |
| **MOVE + TLT data fetcher initial path (FRED ticker)** | FRED doesn't have MOVE under known ticker; fell back to yfinance `^MOVE` | Always try multiple data sources for non-standard series |
| **scripts/burndown_run.py daily cron name** | Name "burndown-daily" misleading — actual schedule is Mon+Thu per Bailey-LdP doctrine (commit 2026-06-14 architecture reset) | Cron task names should reflect actual cadence |
| **Sonnet drift on spec interpretation** (per memory `project_sonnet_spec_interpretation_drift_2026-06-11`) | Sonnet extractor stretches spanning / METHODOLOGY / DECAY_STUDY claims into factor_combination / portfolio_overlay specs | DEFERRED architectural fix (claim-type router Stage 0 helps but not full fix) |

---

## 4. Deferred Work (with revisit-when triggers)

| Item | Estimate | Revisit when |
|---|---|---|
| **HRP-based sizing audit + drop-one recommender** | 2-3h, $0 | NEXT SESSION (highest-leverage open item per Sizing-before-Signal doctrine) |
| **Substrate gap detector** (regress deployed PnL on FF5+MOM+BAB+QMJ+CARRY+TSMOM+VRP, identify |t|<1.65 gaps) | 15-25h, $0 | Mid-term (1-3 months); high architectural leverage |
| **Cross-asset extension scanner** (β agent shipped 2026-06-14, not auto-cron'd) | 3-4h, $0.10-0.20 | After HRP sizing; build monthly cron for systematic Frazzini-Pedersen 70%-enhance discipline |
| **chief_of_staff weekly digest** | 4-5h, ~$0.05/week | After ensemble OOS validated (1-2 weeks); spec'd 2026-06-06 |
| **PROMOTE pipeline 9-check auto-runner** (`engine.research.promote/` stub) | 8-10h, $0 | When a GREEN forward verdict needs PROMOTE evaluation (currently scattered manually) |
| **Phase 2e: audit_agent_liveness + audit_narrative_chain hot-cron refactor** | 2-3h | Coordinated `auto_audit_rules.py` refactor session; cron-coupled risk |
| **Bond Curve Carry Depth Substrate** (Cochrane-Piazzesi 2005 LSC) | 10-12h | After commodity convenience yield substrate test; multi-month substrate build |
| **Commodity Convenience Yield Substrate** | multi-month | Sequenced before bond curve substrate per 2026-06-17 deferral |
| **External validation predictor** (test against verdicts from a different pipeline) | multi-hours | After n_new ≥ 200 on current ensemble |
| **N≥200 re-test of paper Section 4 findings** | passive | Naturally accumulates via daily cron; ~4-6 months at current cadence |
| **LaTeX → PDF actual compile** | minutes | When principal opens Overleaf and uploads .tex + figs |
| **Open-source release of doctrine + spec governance framework** | 8-10h | If publishing the arxiv paper generates external interest |
| **Sonnet spec interpretation drift fix** (claim-type router + spanning_test template) | 6-10h architectural | If/when next GP/A-class spec interpretation issue surfaces |
| **VaR-99 / ES-99 expansion in Risk Manager** | 2-3h | LOW PRIORITY — ES-95 hard-halt 3× threshold already covers 99% region |
| **Belief Layer Phase 3 calibration dashboard UI** (separate from markdown) | 4-6h | If markdown reports prove insufficient for daily ops |
| **W6-rigor sensitivity analysis** (alternative w_fam strategies beyond LOOCV) | 2-3h | If realized OOS Brier diverges from v0.9 0.260 target |

---

## 5. Open Design Questions (UNDECIDED)

1. **Should ensemble activation auto-deactivate if realized Brier > threshold?** Currently manual revert. Could add: "if 7-day rolling Brier > 0.32, set flag False + emit alert". Trade-off: automation vs human oversight.

2. **Should `synthesis` cron run more frequently?** Currently weekly. Memory `project_daily_burn_cron_wip_2026-06-11` says throughput hard-capped by Bailey-LdP at 9 verdicts/week, but synthesis is upstream of verdicts. More synthesis = more hypothesis queue. Unclear if queue depth is a bottleneck.

3. **Should chief_of_staff actually be built before HRP sizing?** chief_of_staff would orchestrate weekly review of belief track + ensemble + decay alerts + paper queue. HRP is a one-time analysis with ongoing maintenance. Sequencing depends on whether principal wants weekly auto-digest more or weekly Sharpe-lift more.

4. **Should we activate Phase 4 closed-loop prior calibration more aggressively?** Currently fires when n_obs ≥ 5 for a family. Could lower to n ≥ 3 (aligned with W6-rigor-A threshold). Risk: noisier prior on small-N families.

5. **What's the live-trading data target before we can publish a "live results" paper?** Current 1 month is too short. 6 months? 12 months? Need to set explicit threshold so we know when to stop saying "too early".

6. **Cross-asset extension via β agent — what's the right cadence?** Monthly seems right (Frazzini-Pedersen 70% enhance rate). But β cost is ~$0.30/sleeve × 13 deployed sleeves = ~$4/month if run on all. Could subset to top 4 family-GREEN sleeves only ($1.20/month).

7. **PROMOTE auto-runner: which 9 checks belong inside vs which stay as human review?** All 9 are deterministic, but the "human capital decision" check is by design human. Could auto-package the other 8 + queue for principal final approval.

---

## 6. Memory File Quick Reference

Path: `${REPO_ROOT}\.claude\projects\c--Users-${USER}-Desktop-intern\memory\`

| Filename | Type | Status |
|---|---|---|
| MEMORY.md | index | LIVE — refresh as new memories accumulate |
| arxiv-preprint-draft-v01-shipped-2026-06-22.md | project | LIVE |
| llm-credit-conservation-standing-2026-06-22.md | feedback STANDING | LIVE |
| belief-layer-phase-3-track-record-shipped-2026-06-21.md | project | LIVE |
| tsmom-5leg-universe-blend-rejection-2026-06-17.md | project | LIVE |
| six-week-critical-path-2026-06-18.md | project | LIVE — W1-W2-W3 done; W4-W5-W6 in flight |
| sizing-before-signal-standing-2026-06-17.md | feedback STANDING | LIVE |
| bond-curve-carry-depth-substrate-deferred-2026-06-17.md | project deferred | DEFERRED |
| factor-exposure-gap-detector-2026-06-17.md | project deferred | DEFERRED |
| brainstorm-senior-upgrade-2026-06-15.md | project | LIVE |
| brainstorm-architecture-2026-06-14.md | project | LIVE |
| alpha-pre-mortem-beta-cross-domain-transfer-2026-06-14.md | project | SHIPPED |
| anti-n-persona-brainstorm-standing-2026-06-14.md | feedback STANDING | LIVE |
| phase-b-belief-synthesis-closed-loop-2026-06-14.md | project | SHIPPED |
| dont-fallback-to-natural-cron-when-batch-possible-2026-06-14.md | feedback STANDING | LIVE |
| gemini-audit-billing-deferred-2026-06-14.md | project deferred | DEFERRED |
| self-audit-blind-spots-standing-2026-06-13.md | feedback STANDING | LIVE |
| architecture-review-2026-06-13.md | project | LIVE |
| strategy-family-vs-claim-family-standing-2026-06-12.md | feedback STANDING | LIVE |
| sonnet-spec-interpretation-drift-2026-06-11.md | project deferred | DEFERRED |
| forward-vs-enhance-statistical-separation-standing-2026-06-11.md | feedback STANDING | LIVE (also in CLAUDE.md) |
| quant-school-anchored-depth-standing-2026-06-11.md | project | LIVE |
| daily-burn-cron-wip-2026-06-11.md | project | LIVE |
| substrate-vs-epistemics-two-axes-standing-2026-06-11.md | feedback STANDING | LIVE |
| belief-layer-5-phases-deferred-2026-06-11.md | project | PARTIALLY STALE (superseded by belief-layer-phase-3-track-record-shipped-2026-06-21) |
| backtest-flex-layer-a-b-2026-06-11.md | project deferred | DEFERRED |
| dead-wall-monitoring-standing-2026-06-10.md | feedback STANDING | LIVE |
| tier-c-senior-construction-plan-2026-06-09.md | project | LIVE |
| explicit-artifacts-contract-no-string-pattern-guessing-2026-06-09.md | feedback STANDING | LIVE |
| fwl-sequential-residual-trap-2026-06-09.md | feedback STANDING | LIVE |
| anchor-panel-sequential-residual-doctrine-2026-06-09.md | feedback STANDING | LIVE |
| l3-elicitation-techniques-deferred-2026-06-09.md | feedback DEFERRED | DEFERRED |
| prompt-scaffolding-is-llm-criticality-ceiling-2026-06-09.md | feedback STANDING | LIVE |
| random-data-test-tolerances-from-theory-2026-06-09.md | feedback STANDING | LIVE |
| gpa-candidate-alpha-factor-2026-06-08.md | project | LIVE — first end-to-end GREEN |
| anti-rut-doctrine-2026-06-07.md | project | LIVE |
| synthesis-empty-is-genuine-2026-06-06.md | project | LIVE — DO NOT soften synthesizer discipline |
| a-plus-b-substrate-first-roadmap-2026-06-05.md | project | LIVE |
| four-employee-agentic-roadmap-2026-06-05.md | project | LIVE — A/B/C/D structure |
| papers-curator-full-architecture-2026-06-05.md | spec LOCKED | PARTIALLY BUILT (Phase 1.7 = 30% built; 7 funnels not all wired) |
| research-session-orchestrator-2026-06-06.md | spec LOCKED | DEFERRED — chief_of_staff |
| piece-by-piece-not-batch-standing-2026-06-05.md | feedback STANDING | LIVE |
| research-area-must-be-i18n-standing-2026-06-06.md | feedback STANDING | LIVE |
| r1-cost-route-audit-keep-claude-2026-06-05.md | project | LIVE — Deepseek fails 50%+ on tool_use; keep Sonnet for tool-heavy workloads |
| subsystem-rigor-audit-2026-06-05.md | project | LIVE |
| deferred-chat-accuracy-2026-06-04.md | project deferred | DEFERRED |
| deferred-interactive-multi-agent-2026-06-04.md | project deferred | DEFERRED |
| default-launch-start-bat-standing-2026-06-04.md | feedback STANDING | LIVE |
| paper-driven-research-chain-locked-standing-2026-06-04.md | feedback STANDING | LIVE |

---

## 7. The 6-Week Critical Path (locked 2026-06-18, status)

| Week | Original plan | Actual outcome |
|---|---|---|
| W1 | scripts/engine reorg (FIX 30%) | ✅ DONE 2026-06-18 |
| W2 | ENHANCE/DISCOVER/FORWARD pipeline split (FIX 30%) | ✅ DONE 2026-06-18 |
| W3 | Risk Engine MVP VaR+CVaR+killswitch (ADD 50%) | ⚙️ PIVOTED — Risk Engine already 95% done; built Belief Phase 3 calibration surface instead |
| W4 | ClaimType Stage 0 router + 2 funnels (ADD 50%) | ✅ DONE 2026-06-21 |
| W5 | FORWARD discover reactivation + paper ingest expansion (ADD 50%) | ⚙️ PARTIAL — W5-a cron health precheck done; FORWARD discover reactivation deferred (already auto-cron'd, blocker was API balance) |
| W6 | chief_of_staff deploy (ADD 50%) | ⚙️ PIVOTED — built W6-rigor + W7-arxiv arc instead; chief_of_staff deferred |
| W7 | (not planned) | ✅ arxiv preprint v0.1 → v0.9 + LOOCV + ACTIVATE + OOS evidence |

**Plan vs reality**: planned 30% FIX + 50% ADD + 20% OPTIMIZE. Actual delivery is closer to **30% FIX + 30% ADD + 40% MEASURE/PUBLISH**. The MEASURE/PUBLISH wasn't in the original plan but became the highest-leverage thing once we discovered the Risk Engine was already done.

---

## 8. Cost Tracking

This session (2026-06-22): ~$1.85 LLM spend across 27 commits.

Running total since project start (estimated from memory): ~$30-50 LLM spend over ~2 months.

Per-month operating cost (**verified from `data/llm_cost_ledger.jsonl`,
audit 2026-06-23 — corrects earlier $10/month estimate**):
- **Actual: ~$19/month run-rate** ($26.75 over 41 days 2026-05-13 →
  2026-06-22, all crons running)
- **Steady-state ~$16/month** (excluding initial backfill spike week
  of 2026-05-12: 15,094 LLM calls in one week from PDF backfill)
- By model: Gemini Flash 15,094 calls / $11.70 (89% of volume) ·
  Sonnet 4.6 523 calls / $12.27 (3% of volume, 46% of spend — by
  design, reserved for tool-use-heavy) · DeepSeek-Pro 1,281 / $2.78 ·
  DeepSeek-Flash 11 / $0.003 · Haiku 1 / $0.0002
- Exceeds principal's "$10/mo ceiling" target by ~2× — flag for
  cost-discipline doctrine review. Possible reasons: (a) Sonnet
  ratio higher than designed (3% volume → 46% spend), (b) backfill
  events not budgeted separately from steady-state.

---

## 9. How to Use This File

**Future-Claude on session start**:
1. Read PROJECT_OVERVIEW.md first (5 min) — understand current external story
2. Read this file (10-15 min) — understand internal design decisions + deferred work + open questions
3. Then read most-relevant memory files for the specific task
4. Then start work

**Principal on session start**:
1. Skim Section 2 (Architectural Decision Log) to remember what was decided
2. Skim Section 4 (Deferred Work) to remember what's pending
3. Skim Section 5 (Open Design Questions) to think about new direction

**When closing a session with substantive design decisions**:
- Append to Section 2 (Architectural Decision Log)
- Update Section 4 (move completed items to "DONE" status)
- Update Section 5 if new open questions surfaced
- Update Section 1.1 if new standing rules locked

---

*Generated 2026-06-22 in response to principal request to "整合起来" (integrate together) the design decisions accumulated over many sessions. Sibling document to PROJECT_OVERVIEW.md (external-facing).*

---

## 10. CORRECTIONS to v1 (2026-06-22 audit pass)

After v1 (commit `db2219d0`), principal flagged "I'm afraid you forget what we designed before". Comprehensive repo audit revealed several SHIPPED modules I had wrongly listed as DEFERRED or missed entirely. Corrections:

| Item | v1 status | Actual status (audit 2026-06-22) |
|---|---|---|
| `engine/research/factor_exposure_gap_detector.py` (671 LOC, Phase 1 MVP) | "DEFERRED" in Section 4 | **SHIPPED**. Regresses deployed sleeve PnL on FF5+MOM+BAB+XA_CARRY+XA_TSMOM+VRP; identifies |t|<1.65 gaps; emits deployment_demand rows. 4 driver scripts: `scripts/fegd_all_sleeves_scan.py` / `scripts/fegd_emit_demand_all_sleeves.py` / `scripts/fegd_equity_book_smoke.py` / `scripts/fegd_pre_enhance_demo_on_session_candidates.py`. Audit output at `data/research_store/audit/fegd_all_sleeves_2026_06_17/all_sleeves_exposure_scan.json`. |
| `engine/agents/strengthener/sleeve_strengthen_proposer.py` (359 LOC) | Not mentioned in v1 | **SHIPPED**. Single-LLM call per deployed sleeve → 0-3 improvement candidates with `improvement_kind` ∈ 6 enum values. The classifier principal referenced. |
| `engine/agents/strengthener/sleeve_strengthen_scan.py` (425 LOC) | Not mentioned | **SHIPPED**. Per-sleeve weekly scan orchestrator (max 5/week rotation, cost-disciplined, dry-run support, idempotency). |
| `engine/agents/strengthener/sleeve_fix_proposer.py` | Not mentioned | **SHIPPED**. Reactive variant of strengthen_proposer (P2): triggers from D signals, proposes fixes. |
| `engine/agents/direction_proposer.py` (405 LOC) | Section 2.4 mentioned router but missed direction_proposer | **SHIPPED + LIVE cron** (`MacroAlphaPro_DirectionProposer` daily; output in `data/agents/direction_snapshots/*.json`). 5-component multiplicative ROI score: paper-priority × data-avail × orthogonality × graveyard × family-saturation. |
| `engine/research/brainstorm/divergent_generator.py` + `lesson_distiller.py` + `promoter.py` | Not mentioned | **SHIPPED** (Phase 1-2 per memory `brainstorm-architecture-2026-06-14`). 4-layer experience-conditioned brainstorm. |
| `engine/research/discovery/` (full subdir, 10+ modules) | Not mentioned | **SHIPPED**. arxiv fetcher / auto_gate / binding_proposer / credibility_scorer / crossref fetcher / data_resolver / discovery_pipeline / family_thresholds / llm_feature_extractor / paper_extractor. |
| `engine/research/pfh/` (Pattern-5-compliant Forward Hypothesis generator) | Not mentioned | **SHIPPED**. axis_catalog / bayesian / catalog / constrained_generator / generator / proposer. |
| `engine/research/protocols/` (Phase X) | Not mentioned | **SHIPPED**. adaptive_diagnostics / protocol_designer / protocol_executor. |
| `engine/research/ablation/` (PBO / CPCV / portfolio stats) | Not mentioned | **SHIPPED**. cpcv / metrics / pbo / portfolio / runner / signals / weighting. |
| `engine/agents/persona/` (12 specialist personas) | Mentioned only α/β/γ | **SHIPPED** (12 personas): anomaly_sentinel / attribution_analyst / audit_recorder / chief_of_staff / decay_sentinel / **devils_advocate** / dq_inspector / risk_manager + memory_index / session_store / tools / turn_memory. |
| `engine/agents/chief_of_staff/` (memo / runner / substrate) | Listed as "specced not deployed" | **SHIPPED code**, deployment status unclear. Has runner.py + memo.py + substrate.py. Worth verifying cron status. |
| 100 specs in `docs/spec_*.md` | Not mentioned | **100 spec files exist** — many predate this session. Many are LOCKED. Many likely-built but build-status unverified. Need spot-check session. |

---

## 11. Comprehensive Module Inventory (audit-verified 2026-06-22)

### 11.1 engine/agents/ (25 module entries — see note below)

> **On the count.** The table below has 25 rows. "~19 agent types" is
> the conservative collapse-by-functional-role count: autopilot is one
> row covering 4 sub-files (`autopilot.py` + `_devils_advocate.py` +
> `_live.py` + `_pre_compute_da.py`); persona is one row containing a
> 12-persona container; `eval` and `governance` are cross-cutting
> infrastructure rather than domain agents. README cites "~20 specialist
> agent modules" which matches 25 minus the persona container counted as
> 1; some older session notes cite "19" which matches further collapsing
> `eval` + `governance` + `research_diagnostician` + `audit_verifier`
> into "audit/eval/governance" as one functional cluster. **None of the
> count framings is wrong — they're different aggregation choices over
> the same 25 module entries.**

| Agent | Subdir/file | Status | Purpose |
|---|---|---|---|
| **anomaly_sentinel** | `anomaly_sentinel/auto_halt.py` | SHIPPED | Per-ticker z-score anomaly detection → auto halt |
| **attribution** | `attribution/{helpers,lifecycle}.py` | SHIPPED | PnL attribution + sleeve decomposition |
| **audit_verifier** | `audit_verifier.py` | SHIPPED | Verifier of strict-gate verdicts |
| **autopilot** | `autopilot.py` + `_devils_advocate.py` + `_live.py` + `_pre_compute_da.py` | SHIPPED | Daily autopilot orchestration |
| **book_monitor** | `book_monitor/{runner,pattern_rules}.py` | SHIPPED | Pattern-based book health rules |
| **chief_of_staff** ⭐ | `chief_of_staff/{runner,memo,substrate}.py` | SHIPPED code, cron unverified | Weekly orchestrator |
| **cross_review** | `cross_review.py` | SHIPPED | Cross-agent verdict review |
| **daily_memo** | `daily_memo.py` | SHIPPED + cron (`MacroAlphaPro_DailyMemo`) | Daily principal memo |
| **decay_sentinel** | `decay_sentinel/{agent,narrator,reasoning}.py` | SHIPPED + cron | Deployed sleeve decay watch |
| **direction_proposer** ⭐ | `direction_proposer.py` | SHIPPED + cron (`MacroAlphaPro_DirectionProposer`) | New-direction ranker (5-factor multiplicative ROI) |
| **dq_inspector** | `dq_inspector/{agent,gates,narrator,...}.py` (8 files) | SHIPPED | Data-quality breach detector |
| **eval** | `eval/{cases,contract,manifest,runner}.py` | SHIPPED | Eval harness |
| **governance** | `governance/{authority,data_egress,perf_budget,tool_output_guard}.py` | SHIPPED | Cross-cutting governance |
| **graveyard_collision** | `graveyard_collision.py` | SHIPPED | Pre-test collision check vs RED history |
| **history_rag** | `history_rag/{bm25,index,retrieve,schema,eval,eval_v2,config}.py` | SHIPPED | RAG over project history |
| **hypothesis_extractor** | `hypothesis_extractor/{extractor,prompt,tool}.py` | SHIPPED | LLM-driven hypothesis extraction from papers |
| **ops_watchdog** | `ops_watchdog/{agent,auto_repair,notifications,prompt,tools,triage}.py` | SHIPPED + cron (`MacroAlphaPro_Watchdog`) | Daily ops health |
| **papers_curator** ⭐ | `papers_curator/` (20+ files) | SHIPPED + cron (daily 08:30) | Employee A; crawl→filter→summary→synthesis |
| **persona** ⭐ | `persona/` (12 persona files + base + tools) | SHIPPED | 12 LLM persona specialists |
| **research_diagnostician** | `research_diagnostician/{diagnostician,tools}.py` | SHIPPED | Research issue diagnostician |
| **risk_manager** | `risk_manager/{agent,advisory,cb_absorption,gates,narrator,orchestrator_hook,persist,thresholds}.py` | SHIPPED + wired into daily paper trade | 12-gate Risk Manager v1.0 (spec id=69) |
| **sector_pipeline** | `sector_pipeline/agent.py` | SHIPPED | Sector PnL pipeline |
| **spec_drafter** | `spec_drafter.py` | SHIPPED | Spec drafter from natural language |
| **strengthener** ⭐ | `strengthener/` (12+ files) | SHIPPED | Employee B; full dispatch chain |
| **workflow_executor** | `workflow_executor/{base,registry,runner}.py` | SHIPPED + cron (`MacroAlphaPro_WorkflowExecutor`) | Workflow orchestrator |

### 11.2 engine/research/ (117 module files + 8 subdirs)

**Major modules (selected, audit-prioritized)**:
- `belief.py` (predict_verdict, ensemble blend v0.9 ACTIVE)
- `belief_autopsy.py`, `belief_track_record.py`, `belief_track_record_rigor.py`, `belief_ensemble_sweep.py`, `belief_prior_calibration.py`, `belief_synthesis_context.py` (Belief Layer 5 phases LIVE)
- `factor_exposure_gap_detector.py` (671 LOC, SHIPPED — was wrongly listed deferred)
- `burndown_{caps,executor,planner,ranker}.py` (FORWARD pipeline orchestration)
- `enhance/{dispatcher,paired_bootstrap,verdict}.py` (ENHANCE pipeline)
- `promote_candidate.py` + `post_green_rigor.py` (PROMOTE pipeline pieces)
- `pre_mortem.py`, `replication_checker.py`, `cross_domain_transfer.py` (α/β/γ specialists)
- `signal_registry.py` (S-class declarative signal catalog)
- `verdict_thresholds.py` (BUG-3 multi-test corrected thresholds)
- `strategy_family_classifier.py` (n_trials denominator, per `strategy_family_for_spec`)
- `decay_watch_trigger.py`, `decay_retest.py`, `decay_history_log.py` (decay sentinel chain)
- `intuition_rules.py` (rules-based intuition catalog)
- `graveyard.py` (RED-verdict registry)
- `forward_decay_prediction.py`, `forward_vector_*` (forward vector pipeline)
- `industry_attribution.py`, `cross_asset_attribution.py`, `red_attribution.py` (anchor regression layer)
- `subsample_stability.py`, `specification_robustness.py`, `pnl_diagnostics.py` (L3 robustness gates)
- `discovery/` subdir (10+ modules — arxiv-q-fin fetcher / auto_gate / credibility_scorer / discovery_pipeline / family_thresholds)
- `brainstorm/` subdir (divergent_generator / lesson_distiller / promoter)
- `pfh/` subdir (axis_catalog / bayesian / constrained_generator / generator / proposer — Pattern-5 Forward Hypothesis)
- `protocols/` subdir (adaptive_diagnostics / protocol_designer / protocol_executor)
- `ablation/` subdir (cpcv / pbo / portfolio / runner / metrics / weighting)
- `options/` subdir (bs_pricer / skew_surface)
- `forward/` + `promote/` subdirs (re-export shims from W2 Phase 1)

### 11.3 engine/portfolio/ (41 module files)

**Deployed sleeve builders** (per `combined_book.py`):
- `build_equity_book_pit_sn` (PEAD-PIT-SN)
- `build_carry_book` (cross_asset_carry)
- `build_tsmom_book` (cross_asset_tsmom)
- `build_crisis_hedge_book` (TLT/GLD extended)
- `build_mom_hedge_book`
- `build_combined_book_regime_conditional` (the 5-sleeve research variant; NOT canonical deployment)

**Per-strategy modules**: `carry_sleeve.py`, `commodity_momentum.py`, `correlation_sentinel.py`, `credit_spread_momentum.py`, `crisis_hedge_tlt_gld{_extended}.py`, `cross_sectional_momentum{,_top1500}.py`, `dpead_*.py` (4 variants), `factor_anomalies.py`, `idiosyncratic_vol_top1500.py`, `issuance_anomaly.py`, `jp_pead.py`, `long_term_reversal.py`, `multisig_regime_overlay.py`, `vix_oas_regime_overlay.py`, `yield_curve_momentum.py`, plus K1_BAB / D_PEAD / PATH_N / CTA_PQTIX implementations.

**Cross-cutting**:
- `allocation_shrinkage.py` (Ledoit-Wolf)
- `attribution_logger.py` (trade-level logger)
- `capacity_simulator.py` (Almgren-Chriss)
- `deployed_registry.py` (canonical deployment registry)
- `execution_filter.py` (deployable filter)

### 11.4 engine/validation/ (24 sleeve-validation modules)

`_*_run.py` files (PIT validators per sleeve): book_config / carry_trend / cn_pead / combined_book / commodity_carry / crossasset_carry / etc. Plus `aqr_factors.py` (factor pull) and `factor_data.py` (data loader).

### 11.5 engine/data_sources/ + engine/data/

- `cftc_cot.py`, `cftc_etf_mapping.py` (CFTC COT data)
- `eia_stocks.py` (EIA petroleum stocks)
- `sp500_announcements/` (Wikipedia + reconciler)
- `engine/data/fetchers/scraper_wikipedia.py`

### 11.6 engine/research_store/

- `_index.py` (SQLite mirror)
- `emit.py` (event emission API)
- `registry.py` (subject vocabulary)
- `schema.py`, `exceptions.py`, `manifest.py`
- `mechanism_catalog.py` (mechanism family catalog)
- `shadow_emit.py` (shadow events for testing)
- `hypothesis/` (hypothesis schema)
- `papers/` (papers registry)
- `red_lessons/` (RED-verdict lessons catalog)
- `forward_vectors/` (forward vector registry)
- `da_briefing/` (devil's-advocate briefing structured output)

### 11.7 engine/llm/ + cost ledger

- `engine/llm/call.py` (canonical LLM call router; 40+ workloads)
- `engine/llm/providers/{anthropic,deepseek}_provider.py`
- `engine/llm_cost_ledger.py` (cost tracking + ALLOWED_AGENT_IDS gate)
- `engine/preregistration.py` (spec amendment gate)

### 11.8 scripts/ (247 top-level + 31 audits + subdirs)

**Crons (registered schtasks)**:
- `papers_curator_daily.py` + `papers_curator_daily_wrapper.bat` → `papers-curator-daily-ingest` (daily 08:30)
- `burndown_run.py` + `burndown_cron_wrapper.bat` → `burndown-daily` (Mon+Thu 09:00; misleadingly named)
- `cron/daily_belief_refresh.py` + `daily_belief_refresh_wrapper.bat` → `MacroAlphaPro\daily-belief-refresh` (daily 06:35)
- `cron/check_llm_provider_health.py` (W5-a-B precheck, used by wrappers)
- Plus dozens of unnamed daily-monitoring crons (paper_trade, watchdog, daily_memo, direction_proposer, etf_holdings, workflow_executor, etc.)

**Install scripts (cron registration helpers)**:
- `install_agentic_cron.py`
- `install_burndown_cron.py`
- `install_daily_ingest_cron.py`
- `install_research_cron.py`

**Cron entry points (LIVE)**:
- `cron_brainstorm.py`, `cron_cross_domain_transfer.py`, `cron_daily_memo.py`, `cron_decay_audit.py`, `cron_decay_retest.py`, `cron_direction_proposer.py`, `cron_pre_mortem.py`, `cron_replication_checker.py`, `cron_workflow_executor.py`

**Reports (recurring + on-demand)**:
- `reports/report_belief_track_record.py`
- `reports/report_belief_track_record_rigor.py`
- `reports/report_belief_track_record_figures.py`
- `reports/report_belief_ensemble_sweep.py`
- `reports/report_deployed_book_attribution.py`
- `reports/report_papers_curator_claim_type_backfill.py`
- `reports/convert_arxiv_md_to_tex.py`

**Audits (31 historical + new)**: `scripts/audits/audit_*.py` — see scripts/README.md migration status.

**Special purpose (selected)**:
- `fegd_*.py` (4 FEGD drivers — factor exposure gap detector callers)
- `synthesis_gold_test.py` (gold-test for synthesizer)
- `papers_curator_synthesis.py` + `run_papers_curator_synthesis.py`
- `burndown_run.py` (FORWARD verdict production)
- `run_paper_trade_daily.py` (daily paper trade NAV)
- `promote_candidate.py` (PROMOTE pipeline driver)

### 11.9 docs/spec_*.md (100 specs)

Too many to enumerate fully. Notable LOCKED specs:
- `spec_risk_manager_agent_v1.md` (spec id=69 — Risk Manager 12 gates)
- `spec_chief_of_staff_agent_v1.md`
- `spec_quant_co_pilot_*.md` (autonomous research loop / decision lineage / pattern recall / verdict reviewer)
- `spec_path_*.md` (~30 sleeve specs — many implemented as portfolio/* sleeves)
- `spec_factor_lab.md`, `spec_factor_library_v1.md`
- `spec_pre_registration_enforcement.md`
- `spec_factor_universe_optimizer_v0.1.md`
- `spec_supervisor_approval_panel_v1.md`
- `spec_etf_holdings_llm_risk_monitor.md`
- `spec_dq_inspector_agent_v1.md`
- `spec_simulated_execution.md`
- `spec_strategy_uplift_2026-05-03.md`
- `spec_book_allocation_carry_crisis_hedge_balance_v1.md`
- `spec_per_strategy_attribution_logger_v1.md`
- `spec_performance_reporting_v1.md`
- `spec_role_aware_test_routing.md`
- `spec_gate_framework_v2_2026-05-14.md` / `v3_2026-05-15.md`

**TODO future audit**: walk each spec, mark BUILT / PARTIALLY / NOT-BUILT. ~3-5h work. Defer to next session.

### 11.10 data/ artifacts

- `data/research/predictions.jsonl` (498 records)
- `data/research/autopsies.jsonl` (101 records)
- `data/research_store/events.jsonl` (290+ events; 8 event types)
- `data/research_store/hypotheses.jsonl` (303+ hypotheses)
- `data/research_store/papers_registry.jsonl`
- `data/research_store/red_lessons.jsonl`
- `data/research_store/forward_vector_reviews.jsonl`
- `data/papers_curator/cache.jsonl` (633 papers)
- `data/papers_curator/cache_with_claim_type.jsonl` (633 ClaimType-tagged)
- `data/papers_curator/summaries.jsonl` (108+ summaries; new ones tag claim_type)
- `data/papers_curator/judgments.jsonl` (filter judgments)
- `data/portfolio_replay/v1_combined_replay_verdict.json` (canonical 4-sleeve replay)
- `data/agents/direction_snapshots/*.json` (direction_proposer daily snapshots, June 13/16/18/19/21)
- `data/research_store/audit/fegd_*/all_sleeves_exposure_scan.json` (factor_exposure_gap_detector outputs)
- `data/cron_burndown/{plans,outcomes,logs}/*.json` (burndown audit trail)
- `data/research_store/tier_c_pnl/*.parquet` (per-verdict PnL series)
- `data/research_store/doctrine_chroma/` (ChromaDB for doctrine RAG)
- `data/research_store/papers_chroma/` (ChromaDB for papers RAG)

---

## 12. Controlled Enums (Classifiers — the "direction-deciding" layer)

These are the load-bearing enums that classify research/improvement decisions. Each is a single source of truth.

| Enum | Where | Values | Purpose |
|---|---|---|---|
| **ClaimType** | `engine/hypothesis_spec/enums.py` + W4 router | FACTOR_HYPOTHESIS, METHODOLOGY, DECAY_STUDY, CAPACITY, MICROSTRUCTURE, FACTOR_STRUCTURE, DOMAIN_FACT, OTHER, UNKNOWN (9 total) | Stage 0 routing of paper into 8 funnels |
| **FamilyV2** | `engine/hypothesis_spec/enums.py` | CARRY, MOMENTUM, REVERSAL, VALUE, QUALITY, LOW_VOL, SIZE, PROFITABILITY, INVESTMENT, VOL_RISK_PREMIUM, TERM_STRUCTURE, SHORT_INTEREST, ATTENTION, EARNINGS_DRIFT, SENTIMENT, SUPPLY_CHAIN, OPTIONS_IMPLIED, HOLDINGS_BASED, CROSS_ASSET_MOMENTUM, OTHER (20) | Top-level mechanism family (Bailey-LdP n_trials denominator) |
| **AssetClass** | `engine/hypothesis_spec/enums.py` | EQUITY, FX, RATES, COMMODITY, CREDIT, OPTIONS, DIGITAL, COMBINED, UNKNOWN (9) | Asset class |
| **SignalType** | `engine/hypothesis_spec/enums.py` | 50+ values (carry/momentum/reversal/value/low_vol/quality/profitability/VRP/skew/PEAD/revision/etc.) | Signal mechanism vocabulary |
| **IMPROVEMENT_KINDS** ⭐ | `engine/agents/strengthener/sleeve_strengthen_proposer.py` | **regime_filter, cost_aware_exec, position_weighting, replacement_seek, risk_overlay, data_quality_patch** (6) | **Per-deployed-sleeve improvement classifier** (the one principal asked about) |
| **claim_shape (11 shapes)** | `engine/agents/strengthener/claim_shape_router.py` | CROSS_SECTIONAL_ALPHA, SPANNING, VRP, FACTOR_COMBINATION, PORTFOLIO_OVERLAY, EVENT_DRIFT, TIME_SERIES_MOMENTUM, CARRY, DECAY_STUDY, CAPACITY, FACTOR_STRUCTURE (11) | Stage 0 hypothesis claim-shape pre-routing |
| **MechanismFamily** | `engine/research_store/hypothesis/schema.py` (mirror of FamilyV2 with case handling) | (same as FamilyV2) | Hypothesis store family enum |
| **SubjectType** | `engine/research_store/schema.py` | factor, sleeve, memory_doctrine, spec, data_quality | Event store subject classification |
| **DispatchKind** (procedural) | `engine/agents/strengthener/procedural_dispatcher.py` | controlled enum of procedural test kinds | When hypothesis is procedural (not factor), how to dispatch |
| **VerdictThresholds** (BUG-3) | `engine/research/verdict_thresholds.py` | thresholds per family n_trials | Multi-test corrected t-stat thresholds |

### 12.1 The "decide direction" chain

```
direction_proposer  →  rank NEW alpha directions (paper-corpus mining, 5-factor ROI)
                       → output: top-N directions for principal to consider
                       → cron: daily

sleeve_strengthen_proposer  →  classify per-sleeve improvement (6 IMPROVEMENT_KINDS)
                               → output: 0-3 candidates per deployed sleeve
                               → wrapped by sleeve_strengthen_scan (weekly orchestrator)
                               → cron: NOT YET REGISTERED (action item)

factor_exposure_gap_detector  →  identify deployed-book factor exposure gaps
                                  → output: ProposedDirection rows for gap factors
                                  → drivers: fegd_*.py scripts (4 scripts)
                                  → cron: NOT YET REGISTERED (action item)

decay_sentinel  →  detect deployed-sleeve decay
                   → output: decay_alert events
                   → routes to: sleeve_fix_proposer (reactive) → ENHANCE pipeline
                   → cron: daily

claim_shape_router  →  pre-route hypothesis to 11 claim shapes (Stage 0)
                       → output: shape + confidence + rationale
                       → reduces Sonnet drift (BUG-2 fix)
                       → fires: per-dispatch
```

---

## 13. Action Items From This Audit

1. **Verify chief_of_staff cron status** — code exists; is it actually scheduled?
2. **Register cron for `sleeve_strengthen_scan`** — weekly, low cost (~$0.50/wk × 13 sleeves capped at 5/wk rotation)
3. **Register cron for `factor_exposure_gap_detector`** — monthly run on all deployed sleeves
4. **Spot-check 100 spec files** — mark BUILT/PARTIAL/NOT-BUILT (~3-5h)
5. **Update PROJECT_OVERVIEW.md Section 8** — remove "designed but deferred" for factor_exposure_gap_detector (it's shipped); add the strengthen_proposer / direction_proposer to the "shipped" inventory
6. **Document the "classifier layer" explicitly** — Section 12 above is new; principals should know we have 6 specialized classifiers for routing decisions

---

*v2 update 2026-06-22: comprehensive audit pass after principal flagged forgotten designs. Sections 10-13 added; Sections 1-9 preserved unchanged.*

---

## 14. Session 2026-06-23 — UI Integration + Public Publish Pipeline

Two large work blocks shipped this session, both responding to
principal-flagged gaps.

### 14.1 Public GitHub publish pipeline (B1 dual-repo)

**Why**: principal needs to ship to GitHub for portfolio showcase
("因为我需要去展示我的作品集"). Direct push not viable — secrets.toml
in git history pre-cf06b833 contains 35 API keys; 14 high-risk top-
level binaries (memory.db 77MB, Backup codes.pdf, personal docs);
8.4 GB `data/` tree with sensitive PnL / autopsies.

**Architecture chosen**: B1 dual-repo (continuous private dev +
weekly sanitized snapshot to public mirror) — NOT B2 live-public.
Rationale: dev velocity here is 10+ commits/day; sanitize-on-every-
commit is permanent mental tax with high ongoing leak risk. Weekly
snapshot = one disciplined chore that costs nothing per dev commit.

**Files**:
- `.publishrc.yaml` — whitelist + blacklist + sanitize_patterns +
  post_check_forbidden. Whitelist is positive (include only known-
  safe paths); blacklist provides defense in depth.
- `scripts/publish/build_public_snapshot.py` (~290 LOC) — 4-stage
  pipeline: collect → copy+sanitize → post-check → report. Built
  custom glob matcher with `**` recursive support + `{a,b}` brace
  expansion (fnmatch doesn't natively).
- `scripts/publish/weekly_snapshot_wrapper.bat` — schtasks wrapper
  for weekly auto-publish. Skips push if snapshot post-check fails
  (defense against accidental leak via cron).
- `docs/PUBLISH_PIPELINE.md` — full operating guide (how to manual
  publish, how to set up first push, how to iterate when post-check
  fails, sanitize coverage audit).
- `LICENSE` — MIT.
- `README.md` — replaced stale v18 with portfolio-tailored hero
  (TL;DR table for 90-second recruiter scan, architecture ASCII,
  self-tuning loop story, 8 reproducer commands, 6 honest caveats,
  15 academic anchors).

**Verified**: live snapshot = 2,250 files / 32 MB (vs 8.4 GB raw =
0.4%); 85 files sanitized; 0 forbidden hits in post-check.

**USER ACTION pending before first push**: rotate AV_KEY +
GNEWS_KEY (in git history pre-cf06b833) + create
github.com/falsifiable-t/macroalphapro public repo.

### 14.2 UI integration response to 3-perspective senior audit

Principal asked for senior recommendation from systems architect /
quant AI engineer / academic perspectives on the UI:

> "从一个专业的系统架构工程师的角度来看我们这个软件以及专业的金融数据
>  的分析量化ai工程师学术大牛的角度给出最合适的工程方案和建议"

**Telemetry-baseline audit**: data/telemetry/events.jsonl was
already collecting (R4.2 prior commit) but never analyzed. 307
events over 11 days revealed:
- 23/41 routes visited (56%); **18 dead** (44%); top 5 = 50% of visits
- 48 visits to deprecated `/lab/today` (legacy mental model)
- 6 RESEARCH rail items at 0 visits (feature museum)

**4 phases shipped** in order A → B → C → D:

| Phase | Commit | Scope |
|---|---|---|
| A: telemetry | (already R4.2) | verified + analyzed 307 events |
| B: Brier story | `2a01dfe9` | `/api/research/belief/calibration` endpoint + Brier KPI in hero strip + `/research/calibration` HONEST FINDING page (~230 LOC) + 8 `/lab/*` muscle-memory redirects |
| C: workflow trace | `64d698a7` | `/api/research/workflow/counts` aggregator + `/research/workflow` single-picture system map (8-stage flow: 633 papers → 12 synth → 303 hyp → 206 spec → 517 pred → 299 verdict → 94 autopsy → Brier 0.374) |
| D: rail cull | `3bb53889` | RESEARCH rail 10→4 items + 6 demoted into "More (6)" collapsible disclosure. Routes preserved (URL still works, only rail prominence removed). Reversible. |

**Senior recommendation NOT taken**: 6th IA refactor. After 5 prior
IA reshuffles (4d.6 / U1 / 2026-06-02 / 2026-06-03 / 2026-06-04
R5), the 4-mode IA itself is sound — what was missing was telemetry-
backed cull. Phase D did that without touching nav structure.

**Senior recommendation initially TAKEN, then REVERSED**: "wait 1
week before cull" — principal pushed back ("为什么要等一周"),
correctly diagnosed as defensive hedge not rigor. 11 active days
of telemetry on a binary "did anyone visit this rail item" question
is sufficient.

**Visible changes**:
- Every page now shows Brier 0.374 (+0.11 vs fam) sticky in hero
  strip — the honest negative finding is no longer markdown-only.
- LEARN rail mode gains [Workflow] + [Calibration] entries.
- RESEARCH rail mode collapsed from 10 to 4 prominent + 6 in disclosure.
- /lab/today + 7 other deprecated paths now redirect instead of 404.
- /research/workflow is the answer to "what does this system do" —
  same goal as the arxiv abstract opening paragraph, but interactive.

**Verified clean**: tsc --noEmit 0 errors, next build 0 errors,
all routes pre-rendered as static, backend smoke-tests pass.

### 14.3 Cumulative scores (3 personas, before → after)

| Persona | Before today | After today | Δ |
|---|---|---|---|
| System architect (Linear/Vercel/Stripe) | 7/10 | 8.5/10 | +1.5 |
| Quant AI engineer (Citadel/Two Sigma) | 6.5/10 | 8.5/10 | +2.0 |
| Academic (LdP/Asness/Fama) | 7.5/10 | 9.0/10 | +1.5 |

### 14.4 Action items from this session

1. **USER**: rotate AV_KEY + GNEWS_KEY (alphavantage.co + gnews.io)
2. **USER**: create github.com/zhangxizhe/macroalphapro PUBLIC repo
3. **USER**: cd to public snapshot dir, init git, set remote, first push
4. **USER**: register `MacroAlphaPro\weekly-public-snapshot` schtasks
   (see docs/PUBLISH_PIPELINE.md for command)
5. **Future Claude**: when re-analyzing telemetry in 1-2 weeks,
   check if any DEMOTED item bumped to >0 visits → flip back to
   promoted. The demotion is fully reversible (1-line edit per item).
6. **Future Claude**: when telemetry shows /research/sessions
   visits remain at 0 despite Sessions being CLAUDE.md doctrine,
   investigate WHY — either UI is bad or the session protocol is
   being run CLI-only.

---

*v3 update 2026-06-23: Section 14 added documenting today's UI integration + public publish pipeline (4 phases A-D + B1 dual-repo). Sections 1-13 preserved unchanged.*

---

## 15. Session 2026-06-23 continuation — Operator Console (7 of 9 Pipeline Stations + bug fixes)

After Session 14 (UI integration), this session built the
Operator Console end-to-end: a UI-triggerable pipeline gated by
typed sessions, per-session cost cap, SSE progress streaming, and
audit trail. 7 of 9 design-doc stations shipped + Lab tab bug fix
+ stale-claim audit-fix + portfolio doc updates.

### 15.1 Foundation (engine/operator_console/)

  schema.py        DataTier / JobState / SessionType / 11 OperatorEventType enums
  pipeline_station.py  Abstract base class with 5-method contract
                        (preflight / estimate_cost / render_config_form
                        / execute / result_lineage)
  store.py         Typed event store + job store (data/operator_console/)
  emit.py          Typed emit helpers with pre-condition validation
  cost_ledger.py   Per-session cost cap (D4) + mid-execution halt threshold (R6)
  registry.py      Station auto-register hook
  worker.py        Async job execution + in-process SSE event queue
                   (D2 implementation; honors CancellationToken at
                   stage boundaries per R3)

### 15.2 Stations shipped (7 of 9)

| # | id | data_tier | session_types | cost | LOC | E2E tested |
|---|---|---|---|---|---|---|
| S1 | paper_ingest | user_data | exploration, research_new | $0.00 | ~340 | preflight |
| S3 | factorspec_extract | user_data | research_new | $0.05 | ~280 | preflight |
| S4 | forward_dispatch | wrds_required | audit, research_new | $0.10 | ~270 | preflight |
| S6 | verdict_view | snapshot_data | all 5 types | $0.00 | ~280 | **yes** (TestClient) |
| S7 | promote_9gate (MVP) | snapshot_data | research_new | $0.00 | ~340 | yes (TestClient, REFUSED_AT_GATE legacy event) |
| S8 | rollback | snapshot_data | audit, ops | $0.00 | ~330 | preflight |
| S8b | doctrine_lock | user_data | doctrine | $0.00 | ~290 | preflight |

  Deferred: S2 Synthesize (LLM-heavy papers → hypothesis), S5
  ENHANCE Dispatch (needs variant returns input wiring), S7
  Gates 2-8 full statistical implementations.

### 15.3 Doctrine adherence

- **Capital decisions HUMAN-only**: S7 and S8 both write
  pending-proposal rows to data/operator_console/ and route to
  /approvals. NEVER auto-execute capital state changes.
- **Pattern 5 ban**: S7's 9-gate design uses sequential statistical
  checks, not multi-agent debate. Gates 2-8 will each be a single
  deterministic computation.
- **Session protocol**: all stations declare
  requires_session_types; trigger refused without active session.
- **R6 lossy restart**: documented + mitigated via
  scan_orphaned_running_jobs on FastAPI startup
  (api/main.py mounts the recovery scan).
- **D1 anchoring**: S6 reads verdict→autopsy→prediction in that
  order so users see the prediction AFTER seeing the verdict (no
  contamination of the operator's read).

### 15.4 Lab tab bug + restoration (commits a6848e06 + 70744314)

Visual verification caught: top nav rendered TWO "Dashboard" tabs.
Root cause: i18n key built from href (not label), so two tabs
sharing href="/dashboard" both resolved to nav.dashboard → both
labeled "Dashboard". Fix: Lab tab → /research (unique destination)
+ i18n entry nav.research changed to "Lab"/"实验室". Lab workspace
concept intact (LabSideRail / mode IA / status bar all unchanged);
only the top-nav anchor was repaired.

### 15.5 Audit-fix (commit 00d483fe)

Verified resume metrics → caught 4 stale claims in portfolio docs:
  (1) "<$5/month" → actually ~$19/month run-rate (verified
      from data/llm_cost_ledger.jsonl: 16,910 calls / $26.75 / 41d)
  (2) "320 unit tests" → actually 5,805 pytest tests / 354 files
  (3) "3 halts on stale data" → actually 11 unique halt events
      (5 sleeve drift + 4 VIX + 2 ops_watchdog multi-mode)
  (4) "single-agent design" → corrected to "multi-agent LangGraph
      workbench + sequential specialist contract (NOT free debate
      per Tetlock 2017)"
Fixed in: PROJECT_OVERVIEW.md + INTERNAL_DESIGN_INDEX.md +
gen_resume_interview_guide.py (which regenerates the Word doc).

### 15.6 Public snapshot upload (this commit)

After 7-station + bug-fix work, the public snapshot was rebuilt
with a tightened `.publishrc.yaml`:
  - `docs/` switched from blanket glob to **curated whitelist**
    (150+ docs/ files → ~26 publication-quality items)
  - `docs/career/` excluded (resume Word doc is private)
  - `Dockerfile` excluded (Dash-era deploy file; Dash deprecated)

Public snapshot final state:
  ~2,165 files / ~31 MB / 73 sanitized / 0 forbidden hits

Pushed to github.com/falsifiable-t/macroalphapro (public repo) on
2026-06-24. Initial push went to a temporarily-named repo that was
then deleted + recreated with the canonical name to drop ~1 GB of
unreachable cruft from the force-push history.

### 15.7 Session commit chain (this session)

```
Phase 0a foundation backend     6376376d
Phase 0a foundation frontend    8992dd23
Phase 0b SessionLauncher fix    43584f59
Audit-fix stale claims          00d483fe
Lab tab bug fix                 a6848e06
Lab tab restore (/research)     70744314
Phase 1.1 S1 PaperIngest        5f8295ae
Phase 1.2 S4 FORWARDDispatch    18068561
Phase 1.3a S3 FactorSpecExtract 308c8d80
Phase 1.3b S6 VerdictView       37f7ad0c
Phase 3a S8b DoctrineLock       d2ed26aa
Phase 2a S7 PROMOTE 9-gate MVP  04f3f1aa
Phase 3b S8 Rollback            82d2d3e3
Publish prep                    94049623
README+publishrc docs cleanup   (this commit)
```

Session total: 54 commits (Session 14 + 15 cumulative).

---

*v4 update 2026-06-23 continuation: Section 15 added (Operator Console
+ Lab fix + audit + public snapshot upload). Sections 1-14 preserved.*

*v4.1 update 2026-06-24: 3-agent senior audit dispatched on v1 public
snapshot; 9 P0 fixes shipped in v3 (`31f6186`). Full audit findings +
deferred P1/P2 backlog + known nuances captured in
[docs/architecture/audit_2026-06-24_three_agents.md](docs/architecture/audit_2026-06-24_three_agents.md).
Read that doc first if continuing the Operator Console / public-repo
work line — it has the engineering debt with file:line refs and the
gotchas (4-vs-5 sleeves, n=94 vs n=101, $19/mo run-rate) that look
like bugs but aren't.*
