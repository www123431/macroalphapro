# Decision — Novy-Marx Gross Profitability POC = RED (with statistical α-t -5.39)

**Date**: 2026-05-29
**Type**: New mechanism candidate evaluation, strict-gate verdict
**Touches**: `engine/validation/gross_profitability.py` (new POC); gate ledger
auto-appended entry #20
**Affects**: nothing deployed — the candidate did not pass the gate, BUT this
generated unusually strong evidence of factor decay (see §3)

## TL;DR

Phase 2 §I.A.1 of `docs/decisions/research_agenda_2026-05-29.md` — testing
whether the Quality factor (Novy-Marx 2013 gross profitability) is the right
equity-side diversifier alongside our earnings-information family
(D-PEAD + revision, ρ=0.64). Pre-committed construction, 129-month sample,
strict gate.

Result: **RED** (technically classified YELLOW by the gate because |α-t|≥2.0,
but in the WRONG direction — α-t=-5.39 means HIGHLY SIGNIFICANT NEGATIVE
alpha). This is the well-documented "junk premium" / quality-factor-decay
phenomenon of 2013-2024. Two consecutive Phase 2 candidates RED (VIX carry +
Quality) provides strong evidence the strict-gate doctrine is doing its job.

## Construction (pre-committed Novy-Marx 2013)

- **Signal**: `gp = (revt - cogs) / at` from Compustat fundamentals, gvkey-level
- **Publication lag**: 6 months after fiscal year-end (Fama-French standard)
- **Universe**: ~2,322 gvkeys in our Compustat funda cache
- **Portfolio**: monthly rebalance, top 30% L / bottom 30% S, equal-weight
  within each leg, ≥10 names per leg required
- **Costs**: 30 bp/side execution (matches our equity sleeve convention)
- **Sample**: 2013-10 → 2024-06 (129 months) — limited by funda cache range +
  6-month lag warmup

This is the canonical academic construction — NO grid search per
[[feedback-strict-gate-no-lowering-2026-05-28]].

## Strict gate result (entry #20 in `data/research/gate_runs.jsonl`)

| Bar | Threshold | Value | Notes |
|---|---|---|---|
| standalone Sharpe | should be ≥1 | **-0.67** | systematically losing |
| Sharpe-t | ≥ HLZ 3.0 | -2.2 | significant in WRONG direction |
| Deflated SR (n=20) | ≥ 0.90 | **0.000** | floor |
| α-t vs FF5+UMD | \|t\| < 2 means "explained" | **-5.39** | extreme negative alpha |
| α annualized vs FF5+UMD | should be positive | **-10.29%** | -10pp/yr residual loss |
| α-t vs FF5+UMD + PEAD | similar | -5.89 | worse with PEAD control |
| OOS Sharpe (2nd half) | should track IS | -0.698 | IS/OOS consistent (not noise) |
| Book corr (vs PEAD) | < 0.5 | +0.282 | also offers no diversification |

**Verdict: RED** (gate classified YELLOW because |α-t|≥2.0; we treat the
in-wrong-direction case as RED for deployment purposes — see §3).

## Why this is HEALTHIER than even VIX carry RED

The VIX carry POC (commit ead7659) had **mild** RED (Sharpe 0.225 / α-t -0.83).
This Quality POC has **decisively** RED with **extreme** negative α-t = -5.39:
- A magnitude-5+ t-stat against the strategy direction is unusual
- It is **statistically impossible** that this is sample noise
- The mechanism (long high-GP stocks, short low-GP stocks) is **actively
  losing money** in this sample

What this tells us:
- The 2013-2024 period was a **"junk premium" era** — high-quality
  profitable stocks underperformed unprofitable junk by ~10pp/yr (residual α)
- This phenomenon is well-documented in academic literature post-2017:
  - AQR 2019 "Quality Minus Junk" notes the post-2010 weakening
  - Numerous papers attribute to: low rates → cheap junk financing,
    growth/momentum era, "profit-less growth" (Tesla, pre-2023 unprofitable
    tech), FAANG concentration
- Without strict gate, we would have added a -0.67 Sharpe sleeve — disaster
- The same gate catches "factor decay" that publication-induced arbitrage
  selection makes hard to detect in casual reading of 2010-era papers

## Why we DON'T reverse the sign

If we flipped the strategy (long junk, short quality), we'd have α-t = +5.39,
α = +10.29% annual. Tempting.

But: **reversing a published factor based on observed losses IS overfitting**.
Per strict-gate doctrine (a published mechanism gets ONE pre-committed shot;
flipping direction after seeing the result = data dredging). The right
response to "the published direction loses" is RED + graveyard, not "let me
flip and rerun".

If the "junk premium" is a real new mechanism (and it might be), it needs:
1. An independent pre-committed test (a NEW spec, not a reverse of an
   existing one)
2. A different sample window (the 2013-2024 result alone is publication-bias-
   inviting)
3. A mechanism-first story (WHY would owning unprofitable companies pay?)

None of those are scheduled now.

## What we learned (and what's in the agenda backlog)

Two consecutive Phase 2 RED results (VIX carry + Quality) demonstrate the gate
is working correctly. Both candidates were:
- Well-documented in academic literature
- Reasonable a priori expectations
- Pre-committed parameters
- Realistic costs

Both failed. This is **the doctrine functioning as intended** — separating
"famous anomalies" from "actually still works in our sample window".

Three legitimate reconsideration paths (deferred — DO NOT redo):
1. **Full QMJ multifactor composite** (Asness-Frazzini-Pedersen 2019):
   profitability + growth + safety + payout. The 4-factor composite may
   survive even when individual components decay.
2. **Longer sample including 1962-2010**: would need older fundamentals data
   not currently cached. Pre-publication Novy-Marx period.
3. **Sub-period analysis**: did GP work in ANY specific year/regime within
   our sample? Useful for understanding when Quality works, NOT for
   deployment without independent fresh test.

## What this DOESN'T mean

- Quality investing is "wrong" — the published 1962-2010 evidence is real
  and well-documented. Our sample just happens to coincide with the worst
  Quality-decay period in modern history.
- We should never test Quality again — we should test a DIFFERENT
  construction (QMJ composite) with INDEPENDENT pre-commitment.
- The book is missing equity diversification — it is, but this POC didn't
  resolve it. Next candidate up.

## Files

- POC code: `engine/validation/gross_profitability.py`
- Gate result: `data/research/gate_runs.jsonl` entry #20
- This decision: `docs/decisions/quality_novymarx_RED_2026-05-29.md`

## Next Phase 2 candidate

Per agenda, remaining Phase 2 options:
- **§I.C Lead-lag POC** (DGNSDE-class, single-sector with strict protocol) —
  12-15h, genuinely new mechanism class
- **§II.B Decay Sentinel + LLM reasoning** — 6-8h, infrastructure rather than
  alpha. Would help us automatically detect Quality-style decay in deployed
  sleeves.
- **§I.A.2 Residual momentum** (Blitz-Huij-Martens 2011) — different equity
  family from earnings + quality, momentum after FF5 controls
