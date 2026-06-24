# P3c — COT-Conditional BAB Pre-Reg Test Verdict

| Field | Value |
|---|---|
| Spec | `docs/spec_p3c_cot_conditional_bab_2026-05-07.md` |
| Spec hash (locked at registration) | `2ab64e667e30ba36` |
| SpecRegistry id | 39 |
| Pre-registered at | 2026-05-07 (BEFORE this code ran) |
| Test executed at | 2026-05-07 |
| Trial count contributed | 1 (locked at registration) |
| **Verdict** | **FAIL — directional signal but underpowered** |
| Decision document author | Auto-Spec Drafter (LLM proposer) + supervisor review |

> **One-line summary**: Effect direction strongly consistent with H1 (Sharpe lift +1.38) but n_extreme=18 conditional months produced bootstrap CI [-0.38, +3.58] crossing zero; raw p=0.166 → BHY-adjusted p=0.43 → fails the locked threshold of BHY-p < 0.05. Pre-registration prevents post-hoc spinning of the directional signal.

---

## 1. Result table

| Metric | Value |
|---|---|
| Period covered | 2020-01 to 2024-12 (5y) |
| BAB universe | SPY + 13 sector / broad equity ETFs (XLE/XLF/XLV/XLI/XLP/XLU/XLK/XLB/XLRE/XLY/XLC/QQQ/DIA) |
| Conditioning indicator | E-MINI S&P 500 Consolidated (CFTC `13874+`) leveraged-money net positioning, decile of trailing 252-week distribution |
| n_months total (unconditional sample) | 53 (after dropping NA / early-archive months) |
| n_months extreme (TOP10 + BOT10 deciles) | 18 |
| Regime distribution | NORMAL=35 · BOT10=12 · TOP10=6 · NA=6 |
| **Sharpe (unconditional)** | **−0.6222** |
| **Sharpe (conditional, extreme COT)** | **+0.7551** |
| **Sharpe lift** | **+1.3772** |
| 95% bootstrap CI on lift (5000 iter, stationary block) | [−0.3847, +3.5777] |
| Raw p-value (two-sided) | 0.1662 |
| BHY-adjusted p (n_eff = 45) | 0.4309 |
| Verdict per locked rule | FAIL |

---

## 2. Locked decision rule (from registered spec)

| Tier | Threshold | Met? |
|---|---|---|
| SHIP | Sharpe lift > +0.15 AND BHY-adjusted p < 0.05 | ❌ (lift OK; p=0.43 not OK) |
| MARGINAL | Sharpe lift > +0.15 AND 0.05 ≤ BHY-p < 0.10 | ❌ (p=0.43 above 0.10) |
| FAIL | all other outcomes | ✅ |

Strict BHY (no literature-conditional exemption) was specified at pre-registration since this is a novel extension test — no prior published support for this specific COT-conditional BAB variant.

---

## 3. Honest reading

### 3.1 What this is not

This is **not** a clean rejection in the "effect doesn't exist" sense. The point estimate (+1.38 Sharpe lift) is large + directionally consistent with H1. With more data the test could plausibly flip to MARGINAL or SHIP.

### 3.2 What this is

A **textbook underpowered test**. With only n=18 in the conditional bucket:
- Standard error of Sharpe lift ≈ 1.0
- Even an effect this large (+1.38) gives t-stat ≈ 1.4
- Under H0 (no lift), getting |lift| ≥ 1.38 by chance has ~17% probability
- BHY adjustment over 45 trials pushes that to 43%

**Pre-registration's value reveals itself here**: without it, one is tempted to spin "+1.38 Sharpe lift → BAB-COT works"; with it, the locked rule forces the honest "FAIL by sample-size insufficiency".

### 3.3 What changes the answer

To clear BHY-p < 0.05 (assuming the +1.38 effect is real):
- n_extreme ≥ 36 (≈ 5 more years of forward calendar OR full backfill of pre-2020 CFTC archive)
- The forward path is slow but clean
- The backfill path is fast but constitutes a **new pre-registration trial** (different period → new spec_hash → +1 to EFFECTIVE_N_TRIALS), not a continuation

---

## 4. Conditional re-test eligibility

This hypothesis is **not permanently rejected** in the directional-effect sense.

A future researcher (including future-self at any horizon) MAY re-evaluate the hypothesis IF AND ONLY IF:

1. n_extreme ≥ 36 conditional months are available, AND
2. The new test is registered as a **fresh pre-registration trial** with its own spec_hash (referencing this spec only as the prior-art motivation), AND
3. EFFECTIVE_N_TRIALS is incremented at the time of new spec registration, AND
4. The new test follows the same code (`scripts/run_p3c_cot_conditional_bab.py`) or publishes the diff.

This document does **not** schedule a re-test. Scheduling a future re-test would either be a fake commitment (high probability of not being executed → rotting promise → worse than no commitment) or would require a real automated trigger (out of scope for this verdict).

---

## 5. Implications for project narrative

This is the project's **8th pre-registered hypothesis test**:

| # | Test | Verdict |
|---|---|---|
| 1 | Narrative direction (LLM forecasts) | REJECT |
| 2 | FactorMAD Q1 | REJECT (0/24) |
| 3 | Sector LLM debate | REJECT |
| 4 | TSMOM monthly own-baseline | REJECT |
| 5 | EFA three-piece uplift | REJECT |
| 6 | S1 multi-window TSMOM | REJECT |
| 7 | B++ Mass FDR (40 strategies) | MARGINAL (QL01 BAB literature-conditional ship) |
| 8 | **P3c COT-conditional BAB** | **REJECT (underpowered)** |

The cumulative pattern (1 marginal ship + 7 rejections) reflects the **structural reality** of master's-scope quant research with free data, ETF universe, and a 5-year window. Most independent hypotheses don't clear strict statistical thresholds at this scale. The project's value is in the **falsification chain itself + the rigor infrastructure** that makes these verdicts honest, not in the absolute count of own-evidence ships.

P3c specifically adds value beyond the previous 6 clean rejections because:
1. Effect direction was correct (+1.38 lift in expected direction)
2. Failure mode is clearly identified (statistical power)
3. Future-research path is unambiguous (more conditional months OR full archive backfill)
4. Spec_hash chain remains INTACT — Tier R audit confirms no silent edits during the test

---

## 6. Cross-references

- Spec: `docs/spec_p3c_cot_conditional_bab_2026-05-07.md` (id=39, status=active)
- Test code: `scripts/run_p3c_cot_conditional_bab.py`
- Result JSON: `docs/decisions/p3c_cot_bab_verdict_2026-05-07.json`
- Auto-Spec Drafter capability used: `docs/agentic_auto_spec_capability.md`
- Falsification chain history: `docs/falsification_chain.md`
- Pre-registration system: `docs/spec_pre_registration_enforcement.md`
- BAB shipped strategy reference: `docs/decisions/b_plus_prod_migration_2026-05-05.md`
- CFTC data source: `engine/data_sources/cftc_cot.py`
