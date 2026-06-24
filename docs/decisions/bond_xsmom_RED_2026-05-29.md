# Decision — Bond Cross-Sectional Momentum POC = RED (AMP 2013)

**Date**: 2026-05-29
**Type**: Phase 2 §I.A.3 candidate evaluation
**Touches**: `engine/validation/bond_xsmom.py` (new POC); gate ledger entry #23
**Affects**: nothing deployed — RED verdict

## TL;DR

5th Phase 2 alpha-hunt candidate, FIRST in this session that's NOT
equity-cousin and IS on long sample (305 months 2001-2026). Tested whether
cross-sectional bond momentum (AMP 2013) can become a 4th mechanism family.

Result: **RED**. The strict gate cleanly rejects (Sharpe -0.337, α-t -2.10,
DSR 0.000, OOS -0.604). But this RED is INFORMATIONALLY DENSER than the prior
4 because the new #1 LLM Research Diagnostician (commit aa915eb) produced a
mechanism-specific causal narrative that deterministic rules cannot generate.

## Strict gate result (entry #23)

| Bar | Value | Status |
|---|---|---|
| standalone Sharpe | -0.337 | FAIL |
| Sharpe-t | -1.7 | FAIL |
| Deflated SR (n_trials=23) | 0.000 | FAIL (floor) |
| α-t vs FF5+UMD | -2.095 | SIGNIFICANT in WRONG direction |
| α annualized | -6.29% | NEGATIVE |
| α-t vs FF5+UMD+PEAD | -1.769 | not significant |
| OOS Sharpe | -0.604 | FAIL (worse than IS) |
| corr_with_book | +0.148 | LOW (would have been diversifying if it worked) |
| Sample | 2001-01 → 2026-05 (305 months) | LONG, multi-regime |

## #1 LLM Diagnostician output (first production use)

**Cost**: $0.13 | **Time**: 87s | **Tools called**: 6 (5 unique) | **Reflexion**: 3 rounds, converged

The LLM's refined diagnosis (verbatim ROOT CAUSE):

> Second-half signal collapse (Sharpe -0.604 vs. ≈ -0.07 first-half) reflects
> a combination of (1) post-publication alpha erosion in the AMP 2013
> cross-sectional bond momentum signal AND (2) mechanistic dispersion
> compression during the 2022 correlated rate shock, both of which
> independently extinguish the cross-sectional ranking edge across the
> 11-instrument G10 bond futures universe.

### What makes this diagnosis genuinely valuable

1. **Mechanism-specific 2022 hypothesis**: deterministic rules cannot generate
   "2022 rate crash creates correlated repricing that compresses
   cross-sectional dispersion, mechanistically destroying the ranking
   signal's information content". This is a CAUSAL CHAIN, not a metric flag.

2. **Co-dominant causes correctly named**: Reflexion 2nd round (verbatim):
   > "Two competing mechanisms explain this deterioration and were conflated
   > in the prior diagnosis: (1) post-publication crowding... (2) the 2022
   > rate crash... I did not disentangle regime-hostility from crowding as
   > competing explanations for the second-half decay."

3. **Self-critique caught measurement uncertainty**: the LLM noticed and
   surfaced that the first-half Sharpe is approximate (flagged by the
   subperiod tool) and adjusted confidence accordingly.

4. **Verified by cross-tool synthesis**:
   - sample_stress_coverage: confirmed 2022_rate_crash inside sample
   - subperiod_analysis: showed first-half vs second-half differential
   - check_deployed_overlap: confirmed carry_book direct overlap (carry
     mechanism shared with bond_xsmom's parent family `cross_asset_carry`)

## Why we DON'T pivot to inverse strategy

The alpha-t -2.10 against FF5+UMD is significant. If we reversed
(long-bottom-tercile / short-top-tercile), we'd get α-t +2.10 — tempting.

But per strict-gate doctrine ([[feedback-strict-gate-no-lowering-2026-05-28]]):
- Reversing a published factor based on observed losses = OVERFITTING
- The "bond mean reversion" interpretation needs INDEPENDENT pre-commitment
  + mechanism-first story, not sign-flip
- This is the same line we held for Quality (RED with α-t -5.39 reversed)

## Lessons for forward Phase 2 strategy

This is the **5th consecutive Phase 2 RED**:

| Candidate | Sample | Verdict | Failure mode |
|---|---|---|---|
| VIX carry | 2018-2026 | RED | publication bias; weak post-cost |
| Quality (NM) | 2013-2024 | RED | junk premium era (α-t -5.39) |
| Residual Momentum | 2018-2024 | RED | PEAD-cousin redundancy (book corr 0.66) |
| Sector lead-lag | 1999-2026 | RED | construction freq mismatch |
| **Bond XSMOM (this)** | 2001-2026 | RED | 2022 rate shock dispersion compression + publication crowding |

The pattern: **long sample (Bond XSMOM had 305 months including all canonical
stress periods) doesn't save you if the mechanism class has a specific
hostile regime in your sample**. The 2022 rate crash is structurally hostile
to cross-sectional bond momentum, just as 2013-2024 is structurally hostile
to Quality.

**What MIGHT work** (for future POCs, do NOT auto-attempt):
- Bond CARRY (not momentum) — different mechanism, same data, already partially
  covered by our deployed 4-leg carry sleeve
- Currency cross-sectional momentum (not bonds) — AMP 2013 also tested currency
  panels; 2022 was less universally hostile to FX
- Multi-asset XSMOM (currencies + commodities + bonds combined) — AMP 2013
  composite; dispersion compression in one asset class may be offset by
  others

## What this RED with diagnostician teaches us about our agentic AI

This is the **first end-to-end demonstration** of:
1. New candidate proposed → run_gate → non-GREEN verdict
2. #2 Knowledge Graph automatically used by #1 to find overlap with deployed
   carry_book (shared parent family `cross_asset_carry`)
3. #1 LLM with tools + Reflexion produces causal diagnosis
4. Diagnosis explicitly distinguishes co-dominant mechanisms
5. Output saved to `data/research/diagnostic_reports.jsonl`

The agentic infrastructure built this session WORKED on a real new case.
This is not the same as scaffolding demos — this is genuine production use.

## Files

- POC code: `engine/validation/bond_xsmom.py`
- Gate result: `data/research/gate_runs.jsonl` entry #23
- LLM diagnosis: `data/research/diagnostic_reports.jsonl` (ledger entry)
- This decision: `docs/decisions/bond_xsmom_RED_2026-05-29.md`
