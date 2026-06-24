# Decision — VIX Term-Structure Carry POC = RED

**Date**: 2026-05-29
**Type**: New mechanism candidate evaluation, strict-gate RED verdict
**Touches**: `engine/validation/vix_carry.py` (new POC), gate ledger
`data/research/gate_runs.jsonl` (auto-appended)
**Affects**: nothing deployed — the candidate did not pass the gate

## TL;DR

The first Phase 2 task from the forward agenda
([docs/decisions/research_agenda_2026-05-29.md](research_agenda_2026-05-29.md)
§I.B) — testing whether the Variance Risk Premium (VRP), captured via VIX
term-structure carry, can become the book's 4th orthogonal mechanism.

Result: **RED**. The strategy as canonically constructed shows positive expected
return (+2.3% annual) but Sharpe 0.225 / Deflated SR 0.114 — nowhere near the
HLZ 3.0 / DSR 0.90 institutional bar. No deployment.

## What was built

`engine.validation.vix_carry.build_vix_carry_returns`:

- **Signal**: term-structure spread `VIX3M - VIX` (90-day implied vol minus
  30-day). Positive = contango = positive expected carry. This is the
  textbook short-vol-when-curve-upward filter.
- **Trade vehicle**: VXX (ProShares VIX Short-Term Futures ETN). Position
  = -1 when contango > 0 (else 0), vol-target sized at 10% annual using
  21-day realized vol, leverage capped at 2x.
- **Costs**: 5 bp/side execution + 0.85% annual ETN expense ratio (VXX),
  applied DAILY on |position|.
- **Output**: monthly compounded return series.

Sample: VXX availability 2018-01 → 2026-05 (101 monthly observations). The
sample covers Vol-mageddon (Feb 2018), COVID crash (Mar 2020), 2022 rate-hike
volatility spikes, and 2024 calm.

Pre-committed parameters (NO grid search per [[feedback-strict-gate-no-lowering-2026-05-28]]):
contango_threshold = 0, target_vol = 10%, vol_lookback = 21d, max_leverage = 2x.

## Strict gate result (commit-ready evidence)

Auto-appended to `data/research/gate_runs.jsonl` at 2026-05-29T13:26:44Z,
n_trials = 19 (gate honest accounting):

| Bar | Threshold | Value | Result |
|---|---|---|---|
| Sharpe-t (HLZ) | ≥ 3.0 | 0.65 | **FAIL** by wide margin |
| Deflated SR (Bailey-LdP n=19) | ≥ 0.90 | **0.114** | **FAIL** by wide margin |
| α-t vs FF5+UMD | \|t\| < 2 acceptable | -0.83 | technically PASS \|t\|<2 |
| α-t vs FF5+UMD + PEAD control | \|t\| < 2 | -0.42 | technically PASS |
| α annualized (FF5+UMD) | should be > 0 | **-2.66%** | NEGATIVE — bad sign |
| OOS Sharpe (2nd half, last 50m) | > 0 | +0.576 | PASS (notable, see below) |
| Book correlation vs PEAD leg | < 0.5 | **-0.182** | PASS (notable, see below) |
| Standalone Sharpe | should be ≥1 for confidence | 0.225 | far below |

Verdict: **RED** — both the headline Sharpe and the deflated SR fail badly.

## Two honest qualitative wrinkles

These are reported for completeness but DO NOT overturn the verdict:

### 1. OOS Sharpe (0.576) > IS Sharpe (0.225)

The strategy actually improves in the second half of the sample. This is
unusual (most failing strategies degrade OOS, not improve). Most likely
interpretation: the bad regimes (Feb 2018 Volmageddon, March 2020 COVID
crash) are in the first half; the second half is more stable.

But this is **NOT** a reason to reverse a RED verdict. The first-half
performance is real history — the strategy DID experience those losses
and cannot un-experience them. Future regimes may include similar shocks
(by construction VIX-short does, eventually).

### 2. Book correlation -0.182 (negative)

VIX carry returns are anti-correlated with our equity book. This is
genuinely valuable diversification — a rare property.

But: even a perfectly anti-correlated 0.225 Sharpe sleeve contributes very
little to the combined book Sharpe. Roughly:
- adding 5% weight (carry/TSMOM precedent) to a Sharpe 0.225 / -0.18 corr
  sleeve → estimated combined book Sharpe lift = ~+0.005 (negligible)
- adding 10% weight → ~+0.010
- meanwhile we'd be funding the position by reducing more-Sharpe sleeves

The math doesn't support deployment even with the diversification benefit.

### 3. α-t vs FF5+UMD is NEGATIVE

The headline 2.3% annual return is fully explained by passive equity-factor
exposures. There is no residual "new alpha" — the strategy IS a leveraged
short-equity-vol bet that happens to make a tiny dollar return after costs.

## Why this is also a positive result

Honest framing: this is a **healthy outcome for the strict-gate doctrine**:
- A famous, textbook anomaly (Karagozoglu-Lin 2010 / Eraker-Wu 2017) was
  tested. Pre-committed parameters, real costs, real sample.
- The gate caught a **publication-bias-style false positive**: the
  in-sample success of the 2006-2010 academic studies does NOT extend to
  the realizable post-cost 2018-2026 implementation.
- Without the gate, we would have added a Sharpe 0.225 sleeve based on
  reputation, diluting the book's Sharpe 1.03 → ~0.99.

This is exactly what gates are for. The campaign ledger now has one more
honest RED entry (#19), and the multiple-testing accounting tightens slightly
for the next candidate.

## What would change the verdict

The canonical academic Karagozoglu-Lin construction is a **calendar spread**
(short front-month + long back-month), NOT a directional short. Our POC
implemented the simpler directional version. A future session could:

1. Re-test with the proper calendar spread (short VXX + long VXZ,
   vol-neutral weighting) — still pre-committed, just a different formulation
2. Re-test with longer sample if VIX futures via WRDS (currently SSL-blocked)
   become accessible — gives pre-Volmageddon history
3. Test the delta-hedged short straddle variant of VRP (requires options
   data infrastructure)

None of these are scheduled now. They go in the agenda backlog with
"considered, RED on first formulation" annotation so a future session
doesn't redo the same first-pass.

## Files / artefacts

- POC code: `engine/validation/vix_carry.py`
- Cached data: `data/cache/_vix_carry_panel.parquet` (VXX + VIX + VIX3M)
- Gate result: `data/research/gate_runs.jsonl` (entry #19)
- This decision: `docs/decisions/vix_carry_RED_2026-05-29.md`

## Lineage / next

- Next item per agenda: `II.B` (Decay Sentinel + LLM reasoning layer) or
  `I.A.1` (Quality factor sleeve) or `I.C` (Lead-lag POC). User picks.
- This decision doc + gate ledger entry are evidence; if a future session
  proposes "let's try VIX carry", point them here and ask: "different
  construction or same?".
