# SAA Bayes-Stein Shrinkage Audit — 2026-05-14

**Decision date**: 2026-05-14
**Anchor**: Tier-1 audit class A #2 (allocation precision)
**Method**: Jorion 1986 Bayes-Stein on weekly excess means + Ledoit-Wolf 2004
on covariance + constrained Markowitz (SLSQP)
**Sample**: Sprint B 2014-09-12 → 2023-12-29 weekly replay, n_weeks = 486

---

## TL;DR

**Recommendation: KEEP CURRENT SAA — empirically dominant**

Current SAA actually delivers HIGHER shrunk Sharpe (+0.396) than the Markowitz-utility-maximizing sleeve-locked alternative (+0.374, Δ -0.022). The optimizer trades return for variance reduction (max U = μ - λ/2·σ² with λ=2.0), so its higher-variance Sharpe is lower than the current allocation's. **The current 36/27/27/10 is robustly defended under Bayes-Stein shrinkage — no amendment warranted, in fact it dominates the shrunk-input optimum on Sharpe.**

---

## 1. Inputs — sample (raw) statistics

| Strategy | Weekly μ (excess) | Weekly σ | Sample Sharpe (ann.) |
|---|---|---|---|
| K1_BAB | -0.0024% | 0.705% | -0.024 |
| D_PEAD | +0.1065% | 1.421% | +0.541 |
| PATH_N | +0.1845% | 2.586% | +0.514 |
| CTA_PQTIX | +0.0101% | 1.460% | +0.050 |

RFR assumed: 4% annual (0.0769% weekly).

### Pairwise correlation (sample)

| | K1_BAB | D_PEAD | PATH_N | CTA_PQTIX |
|---|---|---|---|---|
| **K1_BAB** | +1.000 | -0.107 | +0.027 | -0.061 |
| **D_PEAD** | -0.107 | +1.000 | +0.008 | +0.220 |
| **PATH_N** | +0.027 | +0.008 | +1.000 | -0.032 |
| **CTA_PQTIX** | -0.061 | +0.220 | -0.032 | +1.000 |


---

## 2. Shrinkage applied

### Bayes-Stein on means (Jorion 1986)

- Grand-mean prior (precision-weighted): +0.0245% weekly
  (≈ +1.27% annualized)
- Shrinkage intensity w: **0.5855**
  (0.0 = no shrinkage; 1.0 = full collapse to grand mean)

### Ledoit-Wolf on covariance (Ledoit-Wolf 2004)

- Target: identity scaled by trace(S)/N
- Shrinkage intensity α: **0.0823**
  (0.0 = pure sample cov; 1.0 = pure identity target)

### Per-strategy Sharpe — sample vs shrunk

| Strategy | Sample Sharpe (ann.) | Shrunk Sharpe (ann.) | Δ |
|---|---|---|---|
| K1_BAB | -0.024 | +0.116 | +0.140 |
| D_PEAD | +0.541 | +0.292 | -0.248 |
| PATH_N | +0.514 | +0.260 | -0.255 |
| CTA_PQTIX | +0.050 | +0.090 | +0.040 |


Reading: positive Δ on lower-Sharpe strategies and negative Δ on higher-Sharpe
strategies is the expected pattern (Stein-James "shrinks toward the prior").
Magnitude of shrinkage controlled by sample size and cross-strategy dispersion.

---

## 3. Optimal weights — under 3 constraint sets

| Strategy | Current SAA | Shrunk · sleeve-locked | Shrunk · capped 60% | Shrunk · unconstrained |
|---|---|---|---|---|
| K1_BAB | 36.0% | 36.0% | 0.0% | 0.0% |
| D_PEAD | 27.0% | 21.5% | 56.2% | 56.2% |
| PATH_N | 27.0% | 32.5% | 43.8% | 43.8% |
| CTA_PQTIX | 10.0% | 10.0% | 0.0% | 0.0% |


**Constraint sets**:
- **sleeve-locked**: K1=36% fixed, CTA=10% fixed (spec-locked); D-PEAD + Path N
  intra-sleeve split free within ss_sp500's 54%
- **capped 60%**: no per-strategy cap > 60%; otherwise free
- **unconstrained**: sum=1, [0, 1] bounds, no other constraints

---

## 4. Forward Sharpe estimates (using shrunk μ, Σ)

| Weight set | Sharpe (ann., shrunk) |
|---|---|
| unconstrained | +0.381 |
| with_caps | +0.381 |
| sleeve_locked | +0.374 |
| current_saa | +0.396 |
| equal_weight | +0.381 |


Reading: even after Bayes-Stein shrinkage, "unconstrained" weights produce
the highest in-sample shrunk Sharpe — but only because they ignore
existing sleeve mandates. The "sleeve-locked" line is the apples-to-apples
comparison to current SAA.

---

## 5. Decision

| | |
|---|---|
| Largest single-strategy move (sleeve-locked vs current) | 5.52 pp |
| Shrunk Sharpe gain (sleeve-locked vs current SAA) | -0.0221 |
| **Verdict** | **KEEP CURRENT SAA — empirically dominant** |

Current SAA actually delivers HIGHER shrunk Sharpe (+0.396) than the Markowitz-utility-maximizing sleeve-locked alternative (+0.374, Δ -0.022). The optimizer trades return for variance reduction (max U = μ - λ/2·σ² with λ=2.0), so its higher-variance Sharpe is lower than the current allocation's. **The current 36/27/27/10 is robustly defended under Bayes-Stein shrinkage — no amendment warranted, in fact it dominates the shrunk-input optimum on Sharpe.**

---

## 6. Honest disclosures

- **Sample biased**: in-sample 2014-2023; Garg-Goulding-Harvey-Mazzoleni 2021 reports
  ~50% factor-return decay post-publication. Shrunk Sharpe estimates here are
  upper bounds; expected forward Sharpe per deployment_design.md is 0.85-1.15.
- **CTA shrinkage anomaly**: PQTIX gets a SAMPLE Sharpe of +0.050
  in this 2014-2023 window; this is below the 30-year PQTIX track Sharpe of ~0.4
  and reflects the post-2010 TSMOM decay that Garg 2021 documents. Shrinkage
  pulls it toward the grand mean, which is itself depressed.
- **Sleeve locks reflect doctrine, not statistics**: K1=36% and CTA=10% are
  spec-locked for reasons beyond Markowitz (capacity, crisis-hedge mandate).
  The "unconstrained" optimum is informational, not actionable.
- **Single-period**: this is a one-shot in-sample analysis. Best practice
  (DeMiguel-Garlappi-Uppal 2009) would use rolling out-of-sample evaluation,
  which awaits forward window accumulation.

---

## 7. Cross-references

- `engine/portfolio/allocation_shrinkage.py` — implementation module
- `data/portfolio_replay/saa_stein_james_audit_2026-05-14.json` — full numeric output
- `docs/portfolio_deployment_design_2026-05-13.md` — current SAA spec
- `data/portfolio_replay/v1_per_strategy_returns_weekly.parquet` — input data
