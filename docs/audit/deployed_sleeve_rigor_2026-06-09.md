# Deployed Sleeve Rigor Audit — 2026-06-09

**What this is**: each deployed sleeve's monthly PnL run through Tier C's L2-4 (Ken French FF5+MOM anchor regression) + L2-5 (4-split subsample stability + McLean-Pontiff decay) + L2-6 (12-Industry JOINT regression per post-FWL-fix mechanics) + cross-asset macro extension. **As of Phase 2 Commits 1-4 (2026-06-09): LRV 2011 HML_FX + DOL panel auto-included when cached** — replaces lite macro proxy for FX carry attribution.

**What this is NOT**: any kind of SLM action. Per A+B doctrine [[project-a-plus-b-substrate-first-roadmap-2026-06-05]] capital decisions stay HUMAN. Findings inform; do not auto-decommission.

**Caveats baked in**:
- Deployed sleeves' gross PnL + turnover are NOT persisted. We   use NET return as both gross AND net (Stage 1 gross-vs-net   delta is mechanical 0 here; ignore that column).
- Sleeves with `role=diversifier` or `role=insurance` are NOT   alpha-claim sleeves; spanning critique is N/A by design.   Phase 1 commit c31a81f6 routes them to Tier D separately.
- cross_asset_tsmom is SKIPPED (needs engine backtest run;   add to follow-on audit).
- LRV FX carry anchors window: 2002-04 → 2026-01 (binding   constraint = JPY 3M interbank rate series). For sleeves with   shorter history (e.g., equity_book 2014+), LRV adds nothing;   for cross_asset_carry (1999+) it adds the canonical academic   FX carry attribution.

---

## Summary table

| sleeve | role | n_mo | Sharpe | ann_ret | α₁ t (FF5+MOM) | α₂ t (+ Industry) | **α₃ t (+ Macro = full)** | Δα₁→α₃ | subsample stable? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **equity_book** | alpha | 123 | +1.89 | +15.7% | +8.069 | +8.680 | **+7.218** | +0.851 | ✓ |
| **cross_asset_carry** | alpha | 316 | +1.12 | +11.2% | +5.160 | — | **+3.966** | +1.194 | ✗ |
| **crisis_hedge_tlt_gld** | diversifier | 229 | +0.58 | +6.9% | +1.973 | — | **+2.431** | -0.457 | ✗ |
| **mom_hedge_overlay** | insurance | 157 | -1.10 | -17.8% | -1.363 | -1.660 | **-1.271** | -0.092 | ✗ |

---

## Per-sleeve detail

### equity_book — `role=alpha`

- **Label**: PIT SN (D_PEAD + IBES combo)
- **Paper / lineage**: Boehmer-Jones-Wu 2008-style short-interest signal
- **Window**: 2014-01-31 → 2024-03-31 (123 months)
- **Headline Sharpe**: +1.893  (ann_ret +15.70%, ann_vol 8.29%)
- **Expected spanning risk** (pre-audit): HIGH

**L2-4 Stage 1 (FF5+MOM anchor regression)**

- residual α NW-t = **+8.069**
- residual α annual = +14.652%
- R² = 0.3926
- joint F-test p-value = 1.171e-10
- Top 3 anchor loadings:
    - MOM: β=+0.356 t=+6.11 ***
    - CMA: β=-0.337 t=-2.62 ***
    - RMW: β=+0.206 t=+1.80 *

**L2-6 Joint Model (FF5+MOM + 12-Industry)**  *(post-FWL-fix 2026-06-09)*

- α_full NW-t = **+8.680**  (vs α₁ FF5+MOM-only = +8.069)
- α_full annual = +13.030%
- Δα (t-stat approx) = -0.610 (positive = industry ATE alpha)
- joint R² = 0.4249
- industry-subset F-test p-value = 0.4971  (H0: all 12 industry γ = 0; p < 0.01 → industries add explanation)
- Top 3 industry tilts (joint model):
    - BusEq: β=+0.241 t=+1.24 
    - Utils: β=+0.071 t=+1.04 
    - NoDur: β=-0.098 t=-0.97 

**Cross-Asset Joint Model** (`joint_ff5mom_plus_industry_plus_macro_plus_lrv_fx`)

- α_full NW-t = **+7.218**  (α₁ FF5+MOM = +8.069, Δ = +0.851)
- α_full annual = +13.069%
- joint R² (full) = 0.4538
- macro-subset F-test p = 0.2646
- **LRV FX carry-subset F-test p = 0.1316**  (HML_FX + DOL panel orthogonality test)
- Macro loadings (joint, sorted by |t|):
    - BAA_spread_change: β=+0.01914 t=+1.669 *
    - T10YIE_change: β=+0.02878 t=+1.659 *
    - VIX_change: β=-0.00064 t=-1.162 
    - T10Y3M_change: β=-0.00326 t=-0.470 
    - DXY_return: β=-0.07778 t=-0.264 
- **LRV FX carry loadings** (Lustig-Roussanov-Verdelhan 2011):
    - **HML_FX**: β=-0.00201 t=-1.563 
    - **DOL**: β=-0.00270 t=-1.131 

**L2-5 Subsample stability (4-split)**

- worst/best Sharpe ratio = 0.562  (institutional_stable = True)
- monotone_decay = False; monotone_growth = False
- decay slope = +0.0154%/yr  (NW t = +0.270)

Per-window:

| window | n | Sharpe | NW-t | ann_ret |
|---|---:|---:|---:|---:|
| 2014-01→2016-06 | 30 | +2.359 | +4.42 | +16.24% |
| 2016-07→2019-01 | 31 | +2.435 | +4.14 | +12.64% |
| 2019-02→2021-08 | 31 | +1.368 | +2.38 | +14.23% |
| 2021-09→2024-03 | 31 | +2.010 | +4.22 | +19.71% |

---

### cross_asset_carry — `role=alpha`

- **Label**: G10 carry 4-leg
- **Paper / lineage**: Koijen-Moskowitz-Pedersen-Vrugt 2013, Hassan-Mertens 2017
- **Window**: 2000-02-29 → 2026-05-31 (316 months)
- **Headline Sharpe**: +1.120  (ann_ret +11.20%, ann_vol 10.00%)
- **Expected spanning risk** (pre-audit): MED-HIGH

**L2-4 Stage 1 (FF5+MOM anchor regression)**

- residual α NW-t = **+5.160**
- residual α annual = +10.111%
- R² = 0.0456
- joint F-test p-value = 0.04119
- Top 3 anchor loadings:
    - MKT_RF: β=+0.138 t=+2.50 **
    - MOM: β=+0.052 t=+1.73 *
    - SMB: β=-0.095 t=-1.44 

**L2-6 Industry Joint Model**: SKIPPED for cross-asset sleeve (US-equity industry panel is mis-specified; see cross-asset macro section below)

**Cross-Asset Joint Model** (`joint_ff5mom_plus_macro_plus_lrv_fx`)

- α_full NW-t = **+3.966**  (α₁ FF5+MOM = +5.160, Δ = +1.194)
- α_full annual = +10.234%
- joint R² (full) = 0.2657
- macro-subset F-test p = 1.11e-05
- **LRV FX carry-subset F-test p = 2.457e-06**  (HML_FX + DOL panel orthogonality test)
- Macro loadings (joint, sorted by |t|):
    - T10Y3M_change: β=-0.03315 t=-4.111 ***
    - DXY_return: β=-0.68933 t=-1.737 *
    - BAA_spread_change: β=+0.01385 t=+1.020 
    - T10YIE_change: β=+0.01539 t=+0.826 
    - VIX_change: β=+0.00026 t=+0.421 
- **LRV FX carry loadings** (Lustig-Roussanov-Verdelhan 2011):
    - **HML_FX**: β=+0.00438 t=+4.679 ***
    - **DOL**: β=-0.00490 t=-1.910 *

**L2-5 Subsample stability (4-split)**

- worst/best Sharpe ratio = 0.390  (institutional_stable = False)
- monotone_decay = False; monotone_growth = False
- decay slope = -0.0145%/yr  (NW t = -0.842)

Per-window:

| window | n | Sharpe | NW-t | ann_ret |
|---|---:|---:|---:|---:|
| 2000-02→2006-08 | 79 | +1.284 | +3.62 | +11.89% |
| 2006-09→2013-03 | 79 | +1.648 | +4.22 | +17.02% |
| 2013-04→2019-10 | 79 | +0.644 | +1.89 | +6.30% |
| 2019-11→2026-05 | 79 | +0.913 | +3.02 | +9.59% |

---

### crisis_hedge_tlt_gld — `role=diversifier`

- **Label**: TLT + GLD overlay
- **Paper / lineage**: ad-hoc diversifier; not an alpha claim
- **Window**: 2005-01-31 → 2024-01-31 (229 months)
- **Headline Sharpe**: +0.581  (ann_ret +6.90%, ann_vol 11.88%)
- **Expected spanning risk** (pre-audit): N/A

**L2-4 Stage 1 (FF5+MOM anchor regression)**

- residual α NW-t = **+1.973**
- residual α annual = +5.109%
- R² = 0.0939
- joint F-test p-value = 0.005069
- Top 3 anchor loadings:
    - HML: β=-0.327 t=-3.19 ***
    - RMW: β=+0.213 t=+1.69 *
    - CMA: β=+0.233 t=+1.63 

**L2-6 Industry Joint Model**: SKIPPED for cross-asset sleeve (US-equity industry panel is mis-specified; see cross-asset macro section below)

**Cross-Asset Joint Model** (`joint_ff5mom_plus_macro_plus_lrv_fx`)

- α_full NW-t = **+2.431**  (α₁ FF5+MOM = +1.973, Δ = -0.457)
- α_full annual = +6.082%
- joint R² (full) = 0.5038
- macro-subset F-test p = 3.185e-14
- **LRV FX carry-subset F-test p = 0.002089**  (HML_FX + DOL panel orthogonality test)
- Macro loadings (joint, sorted by |t|):
    - T10Y3M_change: β=-0.04999 t=-4.918 ***
    - T10YIE_change: β=-0.02270 t=-1.806 *
    - BAA_spread_change: β=+0.01258 t=+1.201 
    - VIX_change: β=+0.00051 t=+0.844 
    - DXY_return: β=-0.22452 t=-0.795 
- **LRV FX carry loadings** (Lustig-Roussanov-Verdelhan 2011):
    - **HML_FX**: β=-0.00164 t=-1.650 
    - **DOL**: β=+0.00675 t=+3.458 ***

**L2-5 Subsample stability (4-split)**

- worst/best Sharpe ratio = 0.249  (institutional_stable = False)
- monotone_decay = False; monotone_growth = False
- decay slope = -0.0710%/yr  (NW t = -1.676)

Per-window:

| window | n | Sharpe | NW-t | ann_ret |
|---|---:|---:|---:|---:|
| 2005-01→2009-09 | 57 | +1.102 | +2.88 | +13.64% |
| 2009-10→2014-06 | 57 | +0.609 | +1.30 | +7.23% |
| 2014-07→2019-03 | 57 | +0.274 | +0.60 | +2.79% |
| 2019-04→2024-01 | 58 | +0.308 | +0.65 | +3.99% |

---

### mom_hedge_overlay — `role=insurance`

- **Label**: MTUM short β-overlay
- **Paper / lineage**: ad-hoc insurance; not an alpha claim
- **Window**: 2013-05-31 → 2026-05-31 (157 months)
- **Headline Sharpe**: -1.099  (ann_ret -17.80%, ann_vol 16.20%)
- **Expected spanning risk** (pre-audit): N/A

**L2-4 Stage 1 (FF5+MOM anchor regression)**

- residual α NW-t = **-1.363**
- residual α annual = -2.063%
- R² = 0.9049
- joint F-test p-value = 3.342e-70
- Top 3 anchor loadings:
    - MKT_RF: β=-1.077 t=-33.55 ***
    - MOM: β=-0.408 t=-8.41 ***
    - SMB: β=+0.103 t=+2.06 **

**L2-6 Joint Model (FF5+MOM + 12-Industry)**  *(post-FWL-fix 2026-06-09)*

- α_full NW-t = **-1.660**  (vs α₁ FF5+MOM-only = -1.363)
- α_full annual = -2.650%
- Δα (t-stat approx) = +0.296 (positive = industry ATE alpha)
- joint R² = 0.9117
- industry-subset F-test p-value = 0.05566  (H0: all 12 industry γ = 0; p < 0.01 → industries add explanation)
- Top 3 industry tilts (joint model):
    - Telcm: β=-0.074 t=-1.83 *
    - Utils: β=+0.044 t=+1.19 
    - Manuf: β=-0.070 t=-1.08 

**Cross-Asset Joint Model** (`joint_ff5mom_plus_industry_plus_macro_plus_lrv_fx`)

- α_full NW-t = **-1.271**  (α₁ FF5+MOM = -1.363, Δ = -0.092)
- α_full annual = -1.947%
- joint R² (full) = 0.9129
- macro-subset F-test p = 0.1803
- **LRV FX carry-subset F-test p = 0.6249**  (HML_FX + DOL panel orthogonality test)
- Macro loadings (joint, sorted by |t|):
    - BAA_spread_change: β=+0.01617 t=+1.843 *
    - T10Y3M_change: β=+0.00408 t=+0.897 
    - VIX_change: β=-0.00034 t=-0.882 
    - T10YIE_change: β=+0.00421 t=+0.343 
    - DXY_return: β=-0.00441 t=-0.017 
- **LRV FX carry loadings** (Lustig-Roussanov-Verdelhan 2011):
    - **HML_FX**: β=-0.00014 t=-0.157 
    - **DOL**: β=-0.00171 t=-0.947 

**L2-5 Subsample stability (4-split)**

- worst/best Sharpe ratio = N/A  (institutional_stable = False)
- monotone_decay = False; monotone_growth = False
- decay slope = -0.0601%/yr  (NW t = -0.710)

Per-window:

| window | n | Sharpe | NW-t | ann_ret |
|---|---:|---:|---:|---:|
| 2013-05→2016-07 | 39 | -1.441 | -4.02 | -16.05% |
| 2016-08→2019-10 | 39 | -1.387 | -2.65 | -16.65% |
| 2019-11→2023-01 | 39 | -0.504 | -1.02 | -10.67% |
| 2023-02→2026-05 | 40 | -1.499 | -3.08 | -27.59% |

---

## How to read this report

**For role=alpha sleeves** (`equity_book`, `cross_asset_carry`):
- Stage1 α NW-t < 1.96 → factor's apparent alpha is largely spanned   by FF5+MOM; allocation should be justified beyond 'it's a real factor'
- Stage2 α NW-t < 1.0 → factor is ALSO an industry tilt; the unique   alpha after both peels is statistically zero (GP/A pattern)
- worst/best Sharpe < 0.40 → REGIME-DEPENDENT, not a stable alpha
- post-pub decay > 32% → McLean-Pontiff signature; OOS expectation   should be discounted

**For role=diversifier / insurance**:
- α t-stats are not load-bearing — purpose is correlation structure,   not alpha
- Look at the windows table for crisis-period behavior instead

**Action protocol per A+B doctrine**: This report is a research artifact. SLM DECOMMISSION / RAMP_DOWN remain human decisions, informed by but not auto-triggered by these findings.
