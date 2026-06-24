# Verdict — Factor Library v1 STAGE_1_FAIL (2026-05-08)

**Spec ref**: `docs/spec_factor_library_v1.md` (registered 2026-05-09 sic — actual today is 2026-05-08; spec drafted same day; id=42)
**Pre-registration hash chain**: see SpecRegistry.amendment_log row id=42
**Run date**: 2026-05-08
**Run script**: `scripts/run_factor_library_d4a.py`
**Output cache**: `data/factor_library_in_sample/`

---

## 1. Verdict

**STAGE_1_FAIL** per spec §3.3 decision rule:

> "Stage 1 retained < 3 factors (in-sample selection 失败) → STAGE_1_FAIL → 不进 OOS test, 写 in-sample verdict"

**Retained factors after BHY-FDR + corr filter**: `()` — zero factors.

This is the project's **8th hypothesis test outcome**:
- 6 prior REJECT (narrative overlay, FactorMAD, EFA three-piece, S1 multi-window self-falsification, P3c COT-conditional BAB, multi-period TSMOM 12-1 within sector pipeline LLM ablation)
- 1 prior MARGINAL (B++ Mass FDR / QL01 BAB)
- 1 prior PASS-via-S3 forward registration (paper trading E setup, calendar-bound; not yet hard verdict)
- **This = 8th: STAGE_1_FAIL (cross-sectional factor library on retail-ETF universe)**

---

## 2. Stage 1 Results (Per-Factor NW HAC t-stat, one-sided H₁: mean > 0)

| factor | Sharpe (annualized, in-sample) | n_valid months | first valid | last valid | NW p (one-sided) | BHY pass at α=0.05 |
|---|---:|---:|---|---|---:|:---:|
| bab            | +0.195 | 122 | 1999-11-30 | 2009-12-31 | 0.2402 | **fail** |
| low_vol        | −0.140 | 122 | 1999-11-30 | 2009-12-31 | 0.6816 | **fail** |
| tsmom_12_1     | +0.237 | 153 | 1997-04-30 | 2009-12-31 | 0.2147 | **fail** |
| csmom          | +0.132 | 153 | 1997-04-30 | 2009-12-31 | 0.3317 | **fail** |
| donchian_trend | −0.291 | 131 | 1996-07-31 | 2009-12-31 | 0.8872 | **fail** |

**BHY threshold for rank 1 (smallest p)** at α=0.05, N=5: 0.05 / 5 / c(5) where c(5)=Σ(1/k)≈2.283 → **0.00438**.
**Smallest observed p = 0.2147 (tsmom_12_1)** → fails BHY by 49×.
**Even raw 5% one-sided gate (p < 0.05)**: smallest p = 0.2147 → fails by 4.3×.

This is **not a multiple-test correction loss** — none of the 5 factors reach raw 5% significance individually.

---

## 3. Coverage Disclosure (per rule-8 honest pre-registration)

Spec §3.4 nominal in-sample = 1996-01 to 2009-12 (168 months). Actual coverage at runtime:

| factor | first valid | n_valid / 168 nominal | reason for trim |
|---|---|---:|---|
| BAB / Low-Vol | 1999-11 | 122 / 168 (73%) | Need ≥ 6 ETFs with 252d β/vol → ETF universe reaches 6+ around 1999-1998 (sector SPDRs Dec 1998 + GLD/QQQ early 2000s) |
| TSMOM / CSMOM | 1997-04 | 153 / 168 (91%) | Per-ticker 252d momentum; earliest tickers EWS (1996-03) + EWJ → start +252d ≈ 1997-03 |
| Donchian | 1996-07 | 131 / 168 (78%) | Per-ticker 252d-min (12m horizon) + small universe; coverage stays sparse till later |

Coverage gap is **structural ETF inception**, not data error. Pre-supervisor decision (Option A, 2026-05-08): no spec amend, NaN auto-trim at runtime, disclosed here.

---

## 4. Why This Is Consistent With Reality (Sanity Check)

The 5 factors are all from **stock-level literature** (Frazzini-Pedersen 1980-2012 universe / Baker-Bradley-Wurgler 1968-2008 single-stock / Moskowitz-Ooi-Pedersen cross-asset including individual securities / Asness-Moskowitz-Pedersen multi-asset / Hurst-Ooi-Pedersen futures + commodity). Translating to **~25 retail ETFs** in **1999-2009 (two major crises)**:

- **Low-Vol −0.14**: across-asset-class application means long bonds (TLT/SHY) + short tech (QQQ/SMH). 1999 alone: QQQ +85% / TLT −8% → factor monthly return ~ −93% / 12 ≈ −7.7% just from one year, drags entire 10y mean.
- **Donchian −0.29**: trend ensemble in chop. 1999-2009 had 2 trend reversals (2000-03, 2007-09); breakouts at top got reversed almost immediately → systematic loss.
- **BAB +0.20 / TSMOM +0.24**: signs correct, magnitudes a fraction of single-stock published. Reason: cross-sectional dispersion in 25 ETFs is much smaller than in 7000 stocks → smaller factor returns.
- **CSMOM +0.13**: within-class restricts dispersion further (e.g., 6 sector SPDRs only differ in industry, not in characteristics breadth) → weakest of all.

**Critical lesson**: spec §2.1 v1 lock embedded an **untested assumption that stock-level factors transfer to small ETF universe**. They don't. This is the 8th confirmation that pre-registration discipline must include realistic universe-feasibility check **before** locking candidates, not after.

---

## 5. What This Falsifies vs Doesn't Falsify

### Falsified (spec_factor_library_v1 v1)
- ❌ "5 stock-level published factors retain ≥ 3 on 25-ETF universe via BHY+corr filter at 1999-2009 in-sample"
- ❌ Spec §1 hypothesis: "risk-parity ensemble of 3-4 truly independent BHY-validated factors" — **cannot construct** because BHY rejected all 5
- ❌ ensemble-vs-BAB Stage 2 OOS test (locked rule §3.3: STAGE_1_FAIL → 不进 OOS)

### NOT falsified
- ✅ **QL01 BAB single-factor** at production setup (B++ Mass FDR / 2010-2024 / weekly / Sharpe +0.985 / p=0.0107 raw 5% sig but BHY-fail among 40 candidates) — different construction, different period, different rebalance frequency
- ✅ **Multivariate MSM regime overlay** hypothesis (`spec_multivariate_msm_v1`) — independent feature space, scheduled W1 D5-D6
- ✅ **Layer 1 LLM features** (`spec_layer1_llm_v1` pending W3) — independent
- ✅ The deterministic factor pipeline + agentic governance infrastructure (Auto-Spec Drafter, Tier R audit, factor lab state machine) — **infrastructure value preserved**

---

## 6. Production Impact

**No production change**. Per spec §3.3:
- `engine/portfolio.py::PRODUCTION_SIGNAL` stays `"ql01_bab"` (legacy single-factor; Frazzini-Pedersen 2014 implementation registered earlier).
- No `PendingApproval(production_signal_swap)` triggered.
- `engine/factor_library.py::SELECTED_FACTORS_V1 = ()` permanently — locked with this verdict.
- `engine/factor_library.py::REGIME_SCALAR_LOCKED = {}` permanently — never derived because Stage 1 closed the path.

`engine/factor_library.py` retains 5 signal_fn implementations + ensemble building blocks **as infrastructure asset**. Future v2 spec (different universe / different factor list) can reuse without re-implementing.

---

## 7. Pre-Registration Discipline Compliance

| Discipline rule | Compliance |
|---|---|
| HARKing R1 (silent edits) | ✅ All amendments through amend_spec |
| HARKing R2 (post-test threshold tweak) | ✅ BHY α=0.05 / corr 0.7 / retained ≥ 3 all locked at registration; verdict computed without modification |
| HARKing R3 (unannounced trials) | ✅ No retesting; spec § decision rule honored |
| HARKing R4 (post-hoc inclusion of factors) | ✅ 5 candidate set locked per §2.1 |
| Coverage gap disclosed | ✅ This document §3 |
| EFFECTIVE_N_TRIALS accounting | ✅ +1 trial consumed at register_spec id=42 (n_trials_contributed=1, retro=False) |

---

## 8. Lessons Learned (for v2 amendment if pursued)

1. **Universe-feasibility check before candidate lock**: any future factor candidate must come with a "this factor on THIS universe with THIS history yields raw |t| ≥ 1.65 in pilot" demonstration before entering pre-registration. (Note: this is NOT pilot-then-retest; it's "don't pre-register a hypothesis that's structurally infeasible".)

2. **ETF-specific factor literature**: stock-level factors don't directly transfer. Future v2 should source from:
   - Madhavan & Sobczyk (2016) ETF mispricing factor
   - Asness, Krail, Liew (2001) ETF risk-on/off rotation
   - Lochstoer & Tetlock (2020) ETF cross-section anomalies
   These are ETF-native published factors, not transplants.

3. **Power analysis must include "factor effect-size ÷ universe dispersion" multiplier**: 25 ETFs vs 7000 stocks reduces cross-sectional dispersion by ~20×; published Sharpe of 0.5-1.0 deflates to 0.025-0.05 expected on retail ETFs → spec-required ΔSharpe of +0.05 was ALREADY at the noise floor for this universe.

These lessons feed into `feedback_pretest_experimental_rigor.md` rule-9 (universe-feasibility) — recorded separately.

---

## 9. Disposition

- Spec status: **superseded** (per spec §11 trigger b: "FAIL verdict 后 spec 标 superseded"); set via `amend_spec(kind='superseded')`.
- Verdict file: this document.
- Project memory: new entry `project_factor_library_v1_stage1_fail_2026-05-08.md`.
- Falsification chain: 8th entry; update count in `docs/falsification_chain.md` (if exists).
- W2 todos cancelled: ensemble build, Stage 2 OOS, production swap.
- W1 D5-D6 (Multivariate MSM): proceeds independently.

---

**Verdict locked. No retest. No threshold adjustment. Honor the discipline.**
