# Macro Research Capability Pipeline (2026-05-04)

**Status**: in progress
**Type**: capability extension (no new hypothesis, no spec amendment)
**Triggered by**: 2026-05-04 audit revealed macro_research had **5 AgentRuns total**, **3 AlphaMemory rows (all test data)**, **0 AgentReflections**, despite having `retrieve_relevant_reflections` already wired in [agent.py:115-153](../../engine/agents/macro_research/agent.py).

The reflection RETRIEVAL side was wired but the reflection WRITE side and the schedule that produces forecasts were missing — so the agent was a read-only no-op pipeline.

## Problem statement

> "Macro Intel 用 reflection 让它更智能、自主优化" (user, 2026-05-04)

User's intent maps to the **complete capability pipeline** (Reflexion / Generative Agents / Voyager 三件套学术框架), not to a single piece of strengthening:

1. **Cadence**: agent runs on schedule + on demand
2. **Verification**: deterministic outcome scoring (Brier) on forecasts as horizons expire
3. **Reflection**: post-outcome reflection memos (4-section CONTEXT/DECISION/OUTCOME/LESSON) written to `AgentReflection`
4. **Retrieval**: already wired (top-K cosine similarity injection into next prompt)
5. **Surface**: supervisor sees recent briefs at decision time (P-AUDIT panel integration)

## Sprint plan

| Sprint | Hours | Deliverable |
|---|---|---|
| **MACRO-0 Bootstrap** | 2.0 | (a) cleaned 3 test-data rows; (b) macro_research enrolled in weekly cycle scheduler; (c) supervisor manual trigger button on `pages/macro_brief.py`; (d) first real run executed end-to-end |
| **MACRO-V Verification infra** | 2.5 | `verify_macro_forecasts(today)` cron-style scorer: for each brief whose `horizon` has elapsed, fetch RegimeSnapshot at horizon date, compute multi-class Brier (Brier 1950), write `AlphaMemory.era_verdict` + `era_score`. Integrated into the same scheduler that runs the agent. |
| **MACRO-R Reflection write loop** | 2.0 | Listener on `macro.research_completed` event; on horizon expiration, builds 4-section reflection memo (Reflexion / Park 2023) and persists to `AgentReflection` with sentence-transformer embedding (already exists in `engine/agents/reflection.py`). |
| **MACRO-P P-AUDIT panel integration** | 1.5 | New section in [`pages/orchestrator.py`](../../pages/orchestrator.py) decision panel — "近 30 天 macro briefs 摘要 + each verdict" — surfaces macro context to supervisor at entry-approval time. |
| **Total** | **~8h** | Full bootstrap → verification → reflection → surface. |

## Non-goals (red lines)

- **Not LLM-as-alpha**: brief output does NOT influence trade signals (narrative_overlay was rejected 2026-05-02 with B-C ≈ 0 in 60-mo OOS; same red line maintained per [`narrative_overlay_phase0_rejected.md`](narrative_overlay_phase0_rejected.md))
- **Not LLM-as-judge**: Brier scoring is deterministic FRED-vs-actual computation. LLM does NOT score its own forecasts ([`feedback_no_llm_as_judge.md`](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/feedback_no_llm_as_judge.md))
- **No new agent**: this strengthens existing `macro_research`, does NOT add new agent ([`feedback_agent_addition_rule.md`](../../${REPO_ROOT}/.claude/projects/c--Users-${USER}-Desktop-intern/memory/feedback_agent_addition_rule.md))
- **No new ORM table**: reuses `AlphaMemory` (era_verdict + era_score columns already exist) and `AgentReflection`
- **No alpha claim**: this is capability evidence (Reflexion/Generative Agents/Voyager 学术合成), not alpha. Forward-only, no backtest claim.

## Academic anchors

- **Brier 1950** "Verification of Forecasts Expressed in Terms of Probability" — proper scoring rule foundation
- **Selten 1998** *Axiomatic Characterization of the Quadratic Scoring Rule* — multi-class Brier extension
- **Tetlock 2015** *Superforecasting* — calibration via verification + reflection accumulation
- **Shinn et al. 2023** *Reflexion* — verbal self-reflection + episodic memory loop
- **Park et al. 2023** *Generative Agents* — long-term memory + reflection synthesis
- **Wang et al. 2023** *Voyager* — skill library learning
- **Lopez-Lira & Tang 2023** *ChatGPT Stock Forecast* — LLM news-summary alpha decay literature
- **Bybee 2023** *News-driven Business Cycles* — LLM topic extraction for macro signal
- **Diebold-Lee-Weinbach 1994** *Regime Switching with Time-Varying Transition Probabilities* — ex-ante caveat for any forecast-vs-actual claim

## Verification

A new harness `scripts/verify_macro_pipeline.py` will assert:

1. test data deleted (no `logic IN ('test driver', 'x')` rows in AlphaMemory[macro_research])
2. cycle scheduler enrolment (string check)
3. manual-trigger button rendered on `pages/macro_brief.py` (AppTest)
4. `verify_macro_forecasts` callable returns expected dict shape
5. reflection-write-loop callable returns expected dict shape (will be n=0 until horizon hits)
6. `pages/orchestrator.py` AUDIT tab includes the macro-briefs section (AppTest seeded)
7. zero LLM call in any verifier path (asserted by mocking `engine.config.GENAI_MODEL`)

## Cross-references

- [`spec_agent_reflection_memory.md`](../spec_agent_reflection_memory.md) — S2 reflection memory spec (covers retrieval; this work extends to write side for macro_research)
- [`paper_trading_e_v0_2_redesign.md`](paper_trading_e_v0_2_redesign.md) — macro_research is part of Arm B (LLM path) of paper trading E
- [`p_audit_supervisor_panel_evidence.md`](p_audit_supervisor_panel_evidence.md) — MACRO-P integrates into this panel
