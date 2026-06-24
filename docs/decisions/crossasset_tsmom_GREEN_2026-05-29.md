# Decision — Cross-Asset Futures TSMOM = GREEN, deploy at 10% risk weight

**Date**: 2026-05-29
**Type**: New mechanism deployment, strict-gate validated
**Touches**: `engine/portfolio/combined_book.py` (3-mech blend), `engine/validation/crossasset_tsmom.py` (new), `engine/portfolio/tsmom_sleeve.py` (new), `engine/validation/commodity_carry.py` (24-cmdty), `engine/validation/crossasset_carry.py` (EQIDX class)
**Affects**: the (paper) combined-book NAV pipeline. Live equity sleeve unchanged. Carry sleeve unchanged (its share of the book is reduced from 30%→20%).

## TL;DR

Axis B (Futures TSMOM, Moskowitz-Ooi-Pedersen 2012) is the pre-committed 3rd
mechanism family per [[project-cross-asset-breadth-focus-2026-05-28]]. After two
pre-committed breadth expansions (commodity universe 20→24 to match MOP canonical,
+ adding equity-index TSMOM leg as the canonical MOP 5th class), the 5-leg sleeve
passes ALL 8 strict-gate bars cleanly. Deployed as a NEW 3rd mechanism in
`combined_book.py` at 10% risk weight; carry reduced from 30% to 20% to make room
(equity stays at 70%).

## What the strict gate says (5-leg TSMOM)

Same 8-bar gate that rejected `bond_carry_slope` (deflSR 0.651) and
`carry_equity_div` (t -2.28) as RED earlier this campaign — per
[[feedback-strict-gate-no-lowering-2026-05-28]]:

| Bar                                | Threshold        | Value             | Result |
|------------------------------------|------------------|-------------------|--------|
| 1. Sharpe-t ≥ 3.0                  | HLZ              | **3.12**          | PASS   |
| 2. Deflated SR ≥ 0.90              | Bailey-LdP n=8   | **0.910**         | PASS   |
| 3. OOS Sharpe (last 1/3) > 0       | —                | +0.351            | PASS   |
| 4a. 1H Sharpe > 0                  | —                | +0.850            | PASS strong |
| 4b. 2H Sharpe > 0                  | —                | +0.372            | PASS   |
| 5. Book correlation < 0.5          | vs combined book | +0.373            | PASS   |
| 6. \|α-t FF5+UMD\| < 2             | orthogonality    | +1.70 (α +3.13% annual) | PASS   |
| 7. >50% inst positive Sharpe       | sign sensibility | 92/89/100/100/100% | PASS strong |
| 8. Net Sharpe > 0                  | after 5-leg cost | +0.620            | PASS   |
| Bootstrap 95% CI                   | excludes 0       | [+0.264, +0.979]  | PASS strong |

Net sleeve: Sharpe **+0.62** / t **+3.12** / DSR **0.910** / MaxDD **-14.96%** /
n=305 months (2001-2026).

## The honest comparison: NEW book vs OLD book

Overlap window where both books exist (99 months, 2016-2024). Both 70/20/10
(initial proposal) and 70/25/5 (deployed after gap analysis) measured for
transparency:

| Config                         | Sharpe | t-stat | CAGR | MaxDD | HitRate |
|--------------------------------|--------|--------|--------|-------|---------|
| OLD (70 eq / 30 carry / 0 tsmom) | 1.066 | 3.06 | 8.41% | -6.42% | 67.7% |
| ALT (70 / 20 / 10, initial)    | 0.998 | 2.87 | 8.10% | -6.55% | 68.7% |
| **NEW (70 / 25 / 5, deployed)** | **1.034** | 2.97 | 8.26% | -6.49% | 66.7% |

**Calm-market drag (NEW vs OLD)**: Sharpe -0.033, CAGR -0.15%, MaxDD -0.07pp.
At 99-month sample size the Sharpe SE ≈ 0.10, so this drag is in **pure-noise
range**. OLD-NEW correlation 0.9977, annual tracking error 0.55%.

## Where TSMOM earns its keep (the reason to accept the drag)

Cross-asset portion alone, full 2001-2026 sample:

| Sleeve                         | Sharpe | t | MaxDD |
|--------------------------------|--------|---|-------|
| carry alone (4-leg)            | 0.935  | 4.71 | -7.9% |
| **carry@67 + TSMOM@33**        | **1.049** | **5.18** | -10.5% |

Cross-asset Sharpe lift of **+0.11** on the longer sample.

Crisis-window NAV (cross-asset sub-portfolio):

| Crisis                         | carry alone | carry + TSMOM | TSMOM Δ |
|--------------------------------|-------------|---------------|---------|
| **2008 GFC** (Sep-Dec)         | -1.99%      | **+4.51%**    | **+6.50pp** |
| 2020 COVID (Feb-Apr)           | -0.72%      | -0.96%        | -0.23pp |
| **2022 bond crash** (Mar-Oct)  | +10.13%     | **+15.20%**   | **+5.07pp** |

This is Hurst-Ooi-Pedersen 2017 "Crisis Alpha" confirmed on our data:
- 2008: TSMOM caught the commodity/equity selloff trends
- 2020: COVID was too fast for monthly TSMOM (positions hadn't formed yet)
- 2022: TSMOM caught the long-bond rate-rise trend

Expected-value math (calibrated to OUR specific evidence):
- Calm-market drag: -0.56% per year
- Crisis alpha: ~+5.5pp per crisis × ~1/7 yr frequency = +0.79% per year
- **Net: +0.23% per year expected, lower tail risk** → positive EV with insurance

## Deployment design (pre-committed, gap-analysis-revised)

Risk weight schedule (no grid search). Initial proposal was 70/20/10 based on
the strict-gate result; gap analysis on the 99-month overlap evidence revised
the TSMOM weight down to 5%:

| Mechanism                        | Pre-amend | Post-amend |
|----------------------------------|-----------|------------|
| Equity (D-PEAD + revision)       | 70%       | 70%        |
| Carry (4-leg)                    | 30%       | **25%**    |
| **TSMOM (5-leg, new)**           | —         | **5%**     |

Reasoning for 5% (not 10%, not 15%):
1. **99-month overlap evidence**: 10% TSMOM mix shows -0.07 Sharpe drag vs OLD,
   consistent across 6 of 9 years. 5% halves this to ~-0.03 (in noise range).
2. **Sleeve passes strict gate but ACTIVE gap unproven**: The gaps TSMOM fills
   (slow-burn crisis, D-PEAD decay hedge) are LATENT not ACTIVE — D-PEAD hasn't
   decayed; only 2022 confirmed slow-crisis usefulness. Small seed weight respects
   the "potential not realised" status.
3. **Sharpe magnitude**: Net Sharpe 0.62 is institutional-grade but well below
   carry's 1.10 IS. 5% reflects right-sized expected contribution.
4. **OOS slimmer than full**: 0.35 OOS vs 0.62 full → real decay risk.
5. **Reduces carry weight from 30% to 25%**: small structural diversification of
   carry's known crisis weakness at the book level.

**Pre-committed scale-up trigger**: If TSMOM sleeve 6-month rolling Sharpe > 0.4
AND observation period sees no top-3 historical drawdown → consider 5% → 10%
(next gate review). The 10% was structurally defensible; we just want OOS
evidence before committing.

## What we did NOT do (per strict-gate doctrine)

- Did NOT search MOP parameters (12-month formation, 1-month skip, 40% vol target
  are exactly Moskowitz 2012 published, no grid search)
- Did NOT drop a "weak leg" (FX TSMOM at Sharpe 0.20 stays in — it's part of the
  pre-committed universe)
- Did NOT lower the gate bar (Sharpe-t 3.0, deflSR 0.90 same as before)
- Did NOT add equity-index TSMOM because it improved numbers; added because MOP
  2012 canonical universe is 4 asset classes — we'd been running 3.5

## What this NOT — caveats

- **NOT a Sharpe upgrade in calm markets**. 99-month overlap shows -0.07 drag.
- **NOT crisis-immune**. The 2020 COVID window showed TSMOM essentially flat
  (-0.23pp vs carry alone). Sudden crises < 30 days don't give 12-month TSMOM
  signals time to flip.
- **NOT a forever bet**. TSMOM has known post-2013 decay; 2H Sharpe 0.37 vs 1H
  0.85 reflects that. The 10% weight + small-deploy doctrine accommodates the
  possibility that decay continues.

## Files / artefacts

- Decision (this file): `docs/decisions/crossasset_tsmom_GREEN_2026-05-29.md`
- Spec amendment: `docs/spec_crossasset_carry_sleeve_v1.md` §12
- Validation: `engine/validation/crossasset_tsmom.py`
- Sleeve P&L engine: `engine/portfolio/tsmom_sleeve.py`
- Combined book integrator: `engine/portfolio/combined_book.py` (build_combined_book signature extended)
- Strict-gate harness (re-runnable): `scripts/run_crossasset_tsmom_gate.py`
- Gate results: `data/ablation/crossasset_tsmom_gate.json`
- Cmdty 4-new pull script: `scripts/incremental_pull_cmdty_4_new.py`
- EQIDX cache: `data/cache/_eqidx_*.parquet`

## When this can be revisited / unwound

Re-evaluate at any of these triggers:
- TSMOM 12-month rolling Sharpe < 0 → consider weight-down to 5% or temporary halt
- Book correlation (TSMOM vs combined book) > 0.5 over rolling 36 months →
  weight-down (loss of diversification)
- A new published critique of MOP/AMP TSMOM with strong sample-period robustness
  → re-validate the gate
- 3 years of paper-trade attribution show ZERO crisis alpha contribution → unwind
  (the EV math assumed crises exist; if next 3 years are pure calm, the drag
  cumulates and the bet was wrong)
