# SAA Amendment Proposal — Add Carry Futures at 30% Book Risk Budget

**Proposal date**: 2026-05-28
**Author**: Quant Engineering (claude-opus-4-7 assist, 0-LLM-in-DECISION preserved — this memo is informational input, actual approval is Tier-3 user)
**Status**: 🟡 **PENDING TIER 3 APPROVAL** (paper-trade only, no real capital)
**Type**: SAA composition amendment — add 5th sleeve at 30%
**Cross-reference**: spec 77 §9 + §10 + §11 (carry sleeve + risk audit);
[[project-cross-asset-breadth-focus-2026-05-28]] (mechanism context)

---

## TL;DR

Proposal: **Update `PAPER_TRADE_SLEEVE_ALLOCATION` to add `carry_futures` at 30%
of book risk, reducing existing 4 sleeves proportionally by 30%.**

Concretely:

| Sleeve / Strategy | Current | Proposed | Δ |
|---|---|---|---|
| etf_l1 (K1_BAB) | 32.4% | **22.68%** | -9.72pp |
| ss_sp500 (D-PEAD + Path N) | 48.6% | **34.02%** | -14.58pp |
| cta_defensive (CTA PQTIX) | 9.0% | **6.30%** | -2.70pp |
| rms_crisis_hedge (AC TLT/GLD) | 10.0% | **7.00%** | -3.00pp |
| **carry_futures (NEW, 4-leg)** | 0.0% | **30.00%** | **+30pp** |
| **TOTAL** | 100% | 100% | — |

Carry sleeve is a model-marked NAV-only placeholder StrategyModule — no ticker
positions generated in paper-trade execution. Real fills will come from G1
(IB futures broker integration, separate spec). This L2 step exposes carry to
book NAV + UI artifact + sleeve allocation; L3 will route real orders.

---

## 1. Evidence base

### 1.1 Strategy verdict (spec 77 §9 + §10 + §11 amendment)

| Bar | Threshold | Value | Result |
|---|---|---|---|
| Standalone Sharpe-t (HLZ) | ≥ 3.0 | 5.63 | ✅ PASS |
| Deflated SR (n_trials=20) | ≥ 0.90 | 1.0000 (saturated even at n=30) | ✅ PASS |
| CPCV deploy Sharpe median | > 0.50 | **1.10** (5pct 0.80, 95pct 1.41) | ✅ PASS |
| 3rd-of-3 Sharpe | > 0 | 0.84 | ✅ PASS |
| FF5+UMD α-t orthogonality | ≈ 0 | 0.057 | ✅ PASS (correctly cross-asset orthogonal) |
| Corr w/ D_PEAD book | < 0.50 | 0.149 | ✅ PASS |
| Per-instrument sign sensible | 6/7 + structural AGB | verified | ✅ PASS |
| Subperiod robust | + in all halves/thirds | 1H 1.39 / 2H 0.83 / 1-3 thirds all + | ✅ PASS |

**8/8 strict-gate bars cleared with margin** (same bars that rejected
`bond_carry_slope deflSR 0.651` and `carry_equity_div t=-2.28`).

### 1.2 Combined-book performance evidence

99-month overlap with D_PEAD recon (2016-01..2024-03), equity@70% / carry@30%:

| Metric | Equity-only | Equity+Carry (4-leg) | Δ |
|---|---|---|---|
| Sharpe | 0.96 | **1.08** | +13% |
| MaxDD | -8.22% | **-6.61%** | -1.61pp (better) |

Cost calibration: `RT_CY = 12 bps` (Frazzini-Israel-Moskowitz 2015 upper-mid),
combined Sharpe impact -0.006 vs 10bps prior; full IB fill calibration awaits G1.

### 1.3 Weight derivation (per Phase A.2 risk audit, §11.2)

Walk-forward tangent (Sharpe-max) Markowitz median = 0.506, but OOS strict-gate
test of 30→40% lift FAILS (Sharpe lift +0.5% vs +3% bar; MaxDD reduction +0.12pp
vs +0.5pp bar; annual return drops 0.73pp). Walk-forward IS Markowitz 51% is
Michaud 1989 "error maximizer" overfit artifact.

**30% is empirically OOS-near-optimal** — directly carried over from spec 77 §4
(originally calibrated for 2-leg carry; reconfirmed for 4-leg via Phase A.2
walk-forward). Static-grid 38mo OOS:
- 30%: Sharpe 1.593
- 40%: Sharpe 1.601 (+0.5%, NS)
- 45%: Sharpe 1.595 (-0.4%)
- dynamic walk-forward: Sharpe 1.558 (worse than static 30%)

### 1.4 Senior synthesis

| Decision input | Output | Weight |
|---|---|---|
| Spec 77 §10 4-leg verdict (8/8 strict bars) | Strong support | **PRIMARY** |
| CPCV deploy Sharpe 1.10 (§11.1) | Strong support | HIGH |
| Walk-forward weight test (§11.2) | 30% is correct allocation | HIGH |
| FX-factor decomposition (§11.3) | Risk #4 resolved, no FX hedge needed | HIGH |
| Cost RT 12 bps conservative (§11.4) | Honest deploy estimate | INFORMATIONAL |
| Trend↑carry↑ coupling (Risk #3) | DEFERRED to spec 78 (separate amendment) | INFORMATIONAL |

**Net assessment**: Carry sleeve qualifies for SAA addition at 30% of book risk
per Asness-Israelov "Risk Mitigating Strategies" framework AND per cross-asset
diversification literature (Koijen-Moskowitz-Pedersen-Vrugt 2018 "Carry").
Equity-orthogonal alpha source on equity-correlation grounds; book Sharpe lift
+13% with MaxDD reduction.

---

## 2. Concrete deployment changes

### 2.1 Registry changes (Tier-3 governed)

`engine/strategies/registry.py`:
- `ALLOWED_SLEEVES`: add `"carry_futures"`
- `SleeveClass`: add `CARRY_FUTURES = "carry_futures"`

`engine/strategies/adapters.py`:
- Add `_META_CARRY_FUTURES` StrategyMeta (spec_id=77, hash 1726cf18)
- Add `CarryFuturesStrategy(StrategyModule)` placeholder class:
  - `generate_signal()`: returns empty positions dict (NAV-only, no orders)
  - `is_rebalance_day()`: returns False (no order generation)
- Register strategy + register sleeve at 0.30

`PAPER_TRADE_SLEEVE_ALLOCATION` updated via registry validation:
- etf_l1: 0.324 → 0.2268
- ss_sp500: 0.486 → 0.3402
- cta_defensive: 0.090 → 0.063
- rms_crisis_hedge: 0.100 → 0.070
- carry_futures: 0.000 → 0.300
- Sum = 1.000 ✓

### 2.2 What changes for users

**Effective immediately on next daily paper-trade run:**
- K1 BAB book weight: 0.324 × 1.5 = 0.486 → 0.2268 × 1.5 = **0.340** (-30%)
- D_PEAD: 0.486 × 0.5 × 1.5 = 0.3645 → 0.3402 × 0.5 × 1.5 = **0.2552** (-30%)
- PATH_N: same proportional reduction
- CTA_PQTIX: 0.135 → **0.0945** (-30%)
- AC TLT/GLD: 0.150 → **0.105** (-30%)
- carry_futures NAV (model-marked from combined_book): **+0.300** added to book

**Aggregate effect on paper-trade GROSS exposure**: 30% reduction in deployed
ticker positions (because carry placeholder generates 0 tickers). The "released"
30% of book exposure is held as model-marked carry NAV via
`engine.portfolio.combined_book.build_carry_book()`.

**Combined book NAV impact**: identical to current `build_combined_book` output
(which already uses 4-leg carry per §11; this L2 just exposes the structure to
the registry/UI layer).

### 2.3 What is NOT changed in L2

- **No real ticker orders generated for carry** — that's G1 (IB futures broker integration)
- **No trend coupling adjustment** — Risk #3 deferred to spec 78
- **No real capital deployment** — paper-trade only; Tier-3 governance applies; real-capital wait on G4 2028 OOS

---

## 3. Honest negative disclosure

### 3.1 Devil's Advocate concerns

1. **Major book-level intervention**: 30% reduction in all 5 deployed strategies'
   paper orders. K1 BAB and D-PEAD are currently in active forward paper-trade
   since 2026-05-13; this changes their realized paper-trade returns.

2. **Carry sleeve has NO real fills yet**: the 30% allocation goes to a
   placeholder. Until G1 IB integration, the "30% carry" is model-marked NAV
   only. Real-world tracking error vs model is unknown.

3. **Risk #3 (short-vol crisis) not addressed in L2**: spec 75 trend@10% sized
   for equity-only book. With carry@30% added, crisis hedge ratio reduced from
   10%/90% = 11.1% to 7%/93% = 7.5% on the trade side (trend itself reduced
   30%). This MAY warrant trend↑ coupling per L2.4/spec 78, but L2.4 is
   deferred to keep this amendment auditable.

4. **Carry weight not re-derived for new combined Sharpe** (§11.2): 30% was
   originally derived for 2-leg carry Sharpe 0.66. 4-leg now has Sharpe 1.10.
   Markowitz suggests 51% theoretically (Risk #2) but OOS analysis shows
   30→45% all within 0.5% Sharpe noise (Michaud 1989). 30% holds but is at
   the LOW end of acceptable range.

5. **Single-mechanism over-weighting**: With carry@30%, the book has 30% on
   ONE mechanism family (cross-asset carry). If carry experiences a regime
   shift like 2014-2018 commodity bear, the equity hedge (D_PEAD@~25%) may
   not absorb the drawdown.

### 3.2 Risk mitigants

- **Paper-trade only**: real capital separately Tier-3-governed; no real money
  at risk in L2.
- **Watchdog daily monitoring**: existing 06:10 SGT Watchdog will surface any
  paper-trade execution anomalies after rebalance.
- **Reversibility**: L2 can be reverted by reverting the registry change. Audit
  trail in `data/research/gate_runs.jsonl` + spec hash log.
- **Decay Sentinel monitoring**: spec 77 §8 carry decay rule (rolling-36m Sharpe
  < 0.15 SUSTAINED ≥ 18mo AND signal-IC faded) provides safety-net.
- **Devil's Advocate review by user**: this doc is the input.

### 3.3 Things NOT decided here

- L3 real-fill paper integration (waits on G1 IB)
- L4 real-capital deployment (waits on G4 2028 OOS)
- Spec 78 trend↑carry↑ coupling (Risk #3)
- AGB z-score normalize (Risk #5, marginal)

---

## 4. Approval gates

1. ✅ Spec 77 §9+§10+§11 hash-locked (1726cf18) + amend_spec audit trail
2. 🟡 **THIS DOC** — Devil's Advocate review input
3. 🟡 User Tier-3 sign-off (registers as commit acceptance)
4. ✅ Tests pass (carry_sleeve + combined_book regression)
5. ✅ build_ui_artifact carry_book_status section flows (committed `b435082`)
6. 🟡 Watchdog 06:10 SGT next-day surfaces no anomalies post-deploy

---

## 5. Sign-off block

**Tier 3 approval status**: PENDING

**Effective date if approved**: next `MacroAlphaPro_PaperTrade` daily run after
commit.

**Cross-reference**:
- [[project-cross-asset-breadth-focus-2026-05-28]] (project axis)
- [[feedback-strict-gate-no-lowering-2026-05-28]] (doctrine)
- spec 77 (current_hash 1726cf18, amendment_log 3 entries)
- saa_path_b_leverage_2026-05-15.md (precedent SAA amendment doc)
- saa_path_ac_addition_review_2026-05-15.md (precedent 5th-sleeve addition)
