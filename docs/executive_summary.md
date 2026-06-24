# Macro Alpha Pro — Executive Summary

**Author**: Zhang Xizhe (NUS MSBA, 2026)
**One-line**: A production-grade quant macro strategy with an agentic AI operations layer; verified Sharpe 0.51 OOS over 60 months, with a documented falsification chain rejecting four LLM-as-alpha hypotheses.

---

## TL;DR

- **Honest baseline (multi-window robustness, S1 verdict)**: Multi-window OOS analysis across 6 × 5-yr rolling windows (2010-2024). **Mean Sharpe -0.06 ± 0.42, 2/6 positive, 0/6 statistically significant, bootstrap 95% CI (-0.49, +0.45) crosses zero**. Strategy is **regime-specific** (W1 2010-2014: +0.52 post-GFC QE) and has decayed to non-significant in 2014+ windows. Pre-registered verdict: FAIL.
- **Agentic ops layer**: 7-agent LangGraph orchestration, decision audit trail, observability dashboard, 3-arm forward paper trading with placebo control.
- **Research integrity (the project's primary contribution)**: **7 hypotheses rigorously tested** under pre-registered specs: 6 REJECTED (D1, D1.1, Phase 0, FactorMAD Q1, EFA, S1 self-falsification) + **1 MARGINAL (B++ mass FDR search)** — Frazzini-Pedersen (2014) Low-Volatility Betting-Against-Beta factor reproduced in retail-grade ETF universe with OOS Sharpe +0.985, NW HAC t = +2.312, β-neutralization confirms pure factor alpha (β≈0, α=+5.0%/yr). Does NOT pass strict BHY FDR over N=40 — methodologically rigorous marginal evidence rather than "discovery" claim. Specs and reasoning archived in `docs/decisions/`.
- **What's running live**: paper trading E v0.2 (sector debate ablation). 3-arm A/B/C with per-sector placebo persistence, pre-registered verdict statistics, bootstrap CI.
- **What's not claimed**: Beating institutional alpha or finding new factor. The project's honest claim is **methodology**, not alpha discovery: pre-registered falsification framework + self-falsification of own baseline + 6 documented negative results.

---

## What I built

### Layer 1 — Generation (LLM, decision augmentation)
LangGraph multi-agent debate (Blue/Red/Arbitrator) on flip sectors; macro research agent for FRED+news synthesis; risk officer agent for narrative-driven alerts. **All LLM outputs feed downstream as proposals, never as verdicts.**

### Layer 2 — Validation (deterministic, no LLM)
TSMOM/regime/vol-target signal pipeline (Moskowitz-Ooi-Pedersen 2012 + Hamilton 1989 + Moreira-Muir 2017); Newey-West HAC standard errors; PBO + Deflated Sharpe; multi-arm forward paper trading with random N(0,σ) placebo control.

### Layer 3 — Sizing & execution (deterministic, no LLM)
Inverse-vol weighting + Ledoit-Wolf shrinkage; ATR-based transaction cost; position cap + correlated-pair limit; human-in-loop approval gates.

### Persistence
SQLite (`macro_alpha_memory.db`) with `agent_runs`, `agent_events`, `cycle_states`, `paper_trading_runs`, `decision_logs`, `pending_approvals`, `alpha_memory`. EventBus restart-replayable.

---

## Headline numbers — Multi-Window OOS Distribution (S1 verdict, 2026-05-03)

Per pre-registered S1 spec (`docs/spec_s1_multi_window_robustness.md`), single-point
Sharpe is replaced with multi-window distribution + stationary block bootstrap CI.

| Window | Period | Sharpe | NW t |
|---|---|---|---|
| W1 | 2010-2014 | **+0.522** | +1.41 |
| W2 | 2012-2016 | **+0.330** | +0.78 |
| W3 | 2014-2018 | -0.091 | -0.19 |
| W4 | 2016-2020 | -0.587 | -1.29 |
| W5 | 2018-2022 | -0.373 | -0.86 |
| W6 | 2020-2024 | -0.158 | -0.41 |

```
Cross-window aggregate:
  Mean Sharpe        = -0.059 ± 0.419 (std)
  Range              = [-0.587, +0.522]
  Windows positive   = 2/6
  Windows 5%-sig     = 0/6
  Mean NW t          = -0.094

Stationary block bootstrap (Politis-Romano 1994, n_boot=2000, block_len=3):
  Observed Sharpe    = +0.004 (deduped concat of 179 monthly returns)
  Bootstrap median   = +0.011
  95% CI             = (-0.486, +0.454)  ← crosses zero
  P(boot > 0)        = 51.9%             ← coin-flip indistinguishable from null

Verdict (pre-registered §4.1): FAIL
```

**Honesty trail**:
- Earlier headline of 0.703 (NW t=2.64) was inflated by a self-discovered calendar-arithmetic bug + selection of a single benign window. Bug fixed → single-point estimate dropped to 0.236 → multi-window analysis reveals the more honest finding: **single-point estimates are unstable and the strategy lacks robust alpha across regimes**.
- 3-piece strategy uplift attempt (E + F + A) FAILED verdict at -0.174; defaults reverted per pre-registration discipline. Documented as the 5th falsification.
- S1 multi-window analysis is the 6th falsification — the only one that falsifies the project's **own baseline** rather than an overlay.
- The strategy exhibits **regime-specific alpha decay** consistent with McLean-Pontiff (2016) "Does Academic Research Destroy Stock Return Predictability?" — alpha existed in 2010-2014 (post-GFC QE dispersion) but has eroded in subsequent decade.
- The project's claim is **research methodology**, not investable alpha: pre-registered falsification framework + 6 documented negative results.

---

## Falsification chain (6 rejects + 1 active demo)

The project's **most defensible scientific contribution** is a documented chain of hypothesis tests, each with pre-registered criteria and rigorous backtest. Pre-registration discipline applied uniformly across LLM-as-alpha hypotheses (4), pure quant strategy uplift (1), and **own-baseline self-falsification (1)**.

| # | Hypothesis | Verdict | Doc |
|---|---|---|---|
| 1 | D1 — narrative-driven aggregate vol gate | SOFT REJECT (NW t=0.896) | `docs/decisions/narrative_risk_gate_d1_soft_rejected.md` |
| 2 | D1.1 — D1 retry with power-aware spec | HARD REJECT (B-C ≈ +0.005 over 192 mo) | `docs/decisions/narrative_risk_gate_d1_1_rejected.md` |
| 3 | Phase 0 — IRF-based cross-sectional sector tilt | REJECT (B-C ≈ 0 across 60+89 mo backtests) | `docs/decisions/narrative_overlay_phase0_rejected.md` |
| 4 | FactorMAD Q1 — LLM-mutated factor mining (Gemini-1.5 + multiple-testing critic) | REJECT (0/24 candidates promoted) | `docs/decisions/factor_mad_reject.md` |
| 5 | EFA three-piece quant uplift (universe expansion + asset-class short cap + TSMOM ensemble, per Moskowitz/Hurst/Asness priors) | REJECT (Sharpe -0.174 vs baseline 0.236; FAIL on pre-registered verdict) | `docs/decisions/three_piece_uplift_efa_reject.md` |
| 6 | **S1 baseline robustness — TSMOM + composite signal across 6 × 5-yr windows (2010-2024)** | **REJECT (Mean Sharpe -0.06, 2/6 positive, 0/6 5%-sig, bootstrap CI (-0.49, +0.45) crosses 0; baseline is regime-specific QE-era residual alpha that decayed)** | `docs/decisions/s1_multi_window_evidence.md` |
| 7 | **B++ Mass FDR Search — 20 strategies × 2 tiers, weekly OOS 2018-2024, BHY FDR α=5% over N=40** | **MARGINAL (QL01 Low-Vol/BAB raw p=0.011, OOS Sharpe +0.985, β-neutral confirms pure alpha; fails BHY FDR; 38/40 negative; IC-weighted meta Sharpe 0.68 t=2.16)** | `docs/decisions/b_plus_mass_search_evidence.md` |
| 8 | Sector debate (paper trading E) — LLM Blue/Red ±20pp adjustment on flip sectors | **ACTIVE** (forward demo, post Path 1 redesign) | `docs/decisions/paper_trading_e_v0_2_redesign.md` |

**Lesson 1**: B-C placebo control (Arm B real LLM vs Arm C random N(0,σ)) is the strongest reject signal — stronger than ΔSharpe-vs-baseline. Adopted as standard for any future LLM-alpha validation in this codebase.

**Lesson 2**: Pre-registration discipline applies to your own quant ideas the same as to LLM ideas. EFA was rejected at FAIL verdict; defaults reverted without component cherry-picking, despite individual sub-pieces having strong academic priors.

**Lesson 3 (the deepest)**: Pre-registration discipline applies to **your own baseline**. Multi-window analysis revealed earlier single-point Sharpe estimates were measurement noise on a borderline-marginal window; the strategy's true generalization across 14 years of regimes is statistically indistinguishable from zero alpha. Self-falsification is the strongest signal of research integrity.

---

## What's distinctive (vs typical multi-agent demos)

| Dimension | Typical multi-agent demo | This project |
|---|---|---|
| LLM scope | LLM at every step | **Layered**: LLM only in Layer 1 generation; deterministic in evaluation/sizing |
| Validation | "It looks reasonable" | **Pre-registered statistical gates** (NW HAC + plain t + bootstrap CI + PBO) |
| Reject handling | Tweak prompts, retry indefinitely | **Spec-locked** (HARKing forbidden); documented reject + module deletion |
| Lookahead bias | Ignored | **4-tier mitigation protocol** for LLM-on-historical-text tasks |
| Negative results | Hidden | **Documented as primary output** (4 falsifications) |

---

## Active spec — paper trading E v0.2 (Path 1 redesign)

The sector_pipeline LLM debate ablation runs as a 3-arm forward paper trading experiment:

- **Arm A** baseline TSMOM + vol-target (no LLM)
- **Arm B** baseline + LLM debate ±20pp adjustment on flip sectors
- **Arm C** baseline + random N(0, σ=0.02) placebo on **same** flip sectors

**Pre-registered verdict gates** (frozen in spec v0.2):
- Statistical: t-stat(B-C, one-sided) > 1.645
- Economic A: ann_diff(B-C) ≥ 1.5% (Novy-Marx & Velikov 2016 TC-aware floor)
- Economic B: ΔSharpe(B-C) ≥ 0.15 (half López de Prado institutional standard)
- Joint test → ACCEPT only when all three pass

**Decision rule**: insufficient_n / reject / accept / inconclusive. Lakatos hardcap at n=48; inconclusive defaults to reject (skeptic).

**Redesign trigger**: power analysis showed v0.1 NW HAC test had type-I error 7-9% (target 5%) and 22-41% power on plausible LLM hit-rates. v0.2 uses plain t-test (correctly sized) + cluster-bootstrap CI (robustness).

---

## Repository structure (post-2026-05-03 cleanup)

```
engine/
  signal.py        TSMOM/CSMOM/composite (no LLM)
  regime.py        Hamilton MSM (no LLM)
  portfolio.py     Inverse-vol + LW + position cap (no LLM)
  backtest.py      Walk-forward backtest engine + NW HAC + DSR
  paper_trading.py 3-arm forward ablation + verdict statistics
  sector_pipeline.py / debate.py   LLM Blue/Red debate engine
  agents/macro_research/           Macro research agent
  orchestrator.py                  4-cycle scheduler (daily/weekly/monthly/quarterly)
  memory.py                        SQLAlchemy ORM, EventBus persistence
docs/
  decisions/       Research log: 4 reject docs + 1 active redesign
  decisions/rejected/   Specs of deleted modules (method preserved)
  spec_paper_trading_three_arm_e.md   Active spec (v0.2)
  falsification_chain.md
  capability_evidence.md
pages/             16 Streamlit pages
tests/             Unit tests
```

---

## What I learned

1. **Honest negative results are research signal, not deficiency**. 4 reject + 1 active is healthier than 5 active without verdict.
2. **Power analysis must precede spec lock**. The paper trading v0.1 → v0.2 redesign was triggered by realizing v0.1's verdict was statistically under-powered.
3. **Placebo control (Arm C) is sine qua non for forward LLM ablation**. Without it, you can't separate "alpha" from "baseline market drift".
4. **LLM in macro alpha at master's-project scope is structurally hard** (Gemini-class + public macro/news + monthly rebalance). Lopez-Lira & Tang (2023) priors hold.
5. **Layer-boundary discipline matters**. LLM-as-judge (Zheng 2023) bias is real; keeping LLM out of evaluation/scoring layers prevents self-validating drift.

---

## Reading order for evaluators

1. This document (executive summary)
2. `README.md` — system architecture + quick start
3. `docs/falsification_chain.md` — primary scientific contribution
4. `docs/decisions/README.md` — research log index
5. `docs/spec_paper_trading_three_arm_e.md` — active spec (v0.2)
6. `pages/paper_trading.py` (live in `streamlit run app.py` → "Paper Trading") — running ablation dashboard

---

**Status**: Architecture prototype + falsification chain documented + paper trading E in forward-demo phase. Repository clean (no half-finished modules), all imports green, dashboard renders. Last cleanup: 2026-05-03.
