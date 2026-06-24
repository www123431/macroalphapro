# Tier 3 SAA Amendment — Path B: 1.5x Leverage on 5-Sleeve Composition

**Decision date**: 2026-05-15
**Decision**: APPLY **1.5x constant leverage** to current 5-sleeve PAPER_TRADE_SLEEVE_ALLOCATION
**Status**: 🟡 PENDING TIER 3 APPROVAL
**Author**: Quant Engineering (claude-opus-4-7 assist · 0-LLM-in-DECISION preserved · this memo informational input · Tier 3 supervisor is the actual decision-maker)
**Type**: SAA mandate amendment — leverage application (NOT new sleeve)

---

## TL;DR

Proposal: **Apply LEVERAGE_FACTOR=1.5 uniformly to existing 5-sleeve composition. NO sleeve weight changes. NO new sleeves.**

Current PAPER_TRADE_SLEEVE_ALLOCATION (unchanged):
```
etf_l1:              0.324  (K1 BAB)
ss_sp500:            0.486  (D-PEAD + Path N)
cta_defensive:       0.090  (CTA-PQTIX)
rms_crisis_hedge:    0.100  (AC TLT/GLD)
Total notional:      1.000  (full investment)
```

After 1.5x leverage:
```
Each sleeve weight × 1.5, total notional 1.500 (50% borrowed at RFR)
Effective exposures:
  K1 BAB:           48.6%
  D-PEAD:           36.45%
  Path N:           36.45%
  CTA-PQTIX:        13.5%
  AC TLT/GLD:       15.0%
  Total:            150% gross / Net 100% equity (50% borrowed)
```

---

## 1. Justification

### 1.1 Strategic problem

Current 5-sleeve at 1x leverage has institutional-low vol (5.71%) and Sharpe 0.643 — yields expected total return ~7.67%/yr. For a "balanced multi-strategy fund" mandate targeting **institutional norm 8-10% vol**, this is insufficient. The user senior insight 2026-05-15: "raise portfolio vol toward institutional norm."

### 1.2 Why leverage (vs. alternatives evaluated 2026-05-15 session)

After 12 v3-native widening attempts (1 PASS / 1 MARGINAL / 7 FAIL / 3 overlay MARGINAL) covering:
- 5 alpha class attempts (top-1500 momentum / vol-managed / IVOL / yield curve / commodity / credit) — only 1 MARGINAL
- 5 overlay class attempts (vol-target / 3-signal regime / 2-signal regime / tail-only) — 0 PASS, structurally dead per Path AO finding that AC isn't universal tail hedge
- 1 insurance class (AC) — PASS, deployed at 10%

**Empirical conclusion**: free-data widening saturated. Remaining vol-raise paths:
1. ❌ Add more alpha sleeves — 9 FAILs prove low ROI
2. ❌ Add overlay — structurally inappropriate for our composition (5 tests confirm)
3. ❌ Multi-insurance composition redesign — marginal benefit (-0.07pp DD at -0.33pp CAGR cost vs leverage; user self-corrected earlier)
4. ✅ **Constant leverage on existing optimized 5-sleeve** — direct, predictable, simple

### 1.3 Academic + institutional anchors

- **Modigliani-Miller 1958**: leverage of efficient portfolio preserves risk-adjusted return
- **Sharpe 1966**: leverage doesn't change Sharpe (under unsecured-borrow-at-RFR assumption)
- **Asness 2012**: "Leverage Aversion and Risk Parity" — institutional rationale for moderate leverage on already-diversified portfolios
- **Bridgewater All Weather**: implicit 1.5-2x leverage to achieve target vol
- **CalPERS**: ~1.0-1.3x policy portfolio leverage typical

---

## 2. Empirical evaluation

### 2.1 Backtest stats (paper-trade, 2014-2023 sample)

| Metric | Static 5-sleeve 1x | **5-sleeve × 1.5x leverage** | Δ |
|---|---|---|---|
| Vol annualized | 5.71% | **8.56%** | **+50%** |
| Sharpe (paper) | 0.6431 | **0.6431** | **0 (preserved)** |
| Max DD | -5.45% | **-8.82%** | -3.37pp (proportional) |
| CAGR | 7.79% | **9.57%** | +1.78pp |
| Total return 9.5y | +109% | +146% | +37pp |

### 2.2 Crisis behavior

| Crisis | 1x DD | 1.5x DD | Manageable? |
|---|---|---|---|
| 2018-Q4 VolMageddon | -2.52% | **-3.96%** | ✅ |
| 2020-COVID | -1.41% | **-2.15%** | ✅ |
| 2022 stagflation | -4.13% | **-6.67%** | ✅ |
| **Max DD overall (9.5y)** | -5.45% | **-8.82%** | ✅ within -10% institutional acceptable |

### 2.3 vs institutional peer benchmarks

| Portfolio | Sharpe | Vol | Max DD | CAGR |
|---|---|---|---|---|
| 60/40 SPY/AGG | 0.33-0.50 | 10-12% | -34% (2008) | 6-7% |
| Bridgewater All Weather | 0.50-0.60 | 8-10% | -22% (2008) | 6-8% |
| Yale Endowment policy | ~0.55 | 8-10% | est private | ~10% (with illiquid premium) |
| AQR Multi-Strategy median | 0.40-0.65 | 10-15% | varies | 5-10% |
| **Our 5-sleeve × 1.5x (paper)** | **0.6431** | **8.56%** | **-8.82%** | **9.57%** |
| **Our 5-sleeve × 1.5x (production realistic)** | **~0.55** | **~8.5%** | est -10-12% | 6-8.5% |

**Positioning**: matches or exceeds institutional mid-tier peers.

---

## 3. Honest production-realistic adjustments

### 3.1 Borrow cost (Production only — paper-trade is RFR free assumption)

- IB / Tradier / Alpaca margin facility cost ≈ SOFR + 50-100bp
- For 0.5 portfolio borrowed: cost ≈ 0.5 × (4% + 0.75%) = 2.375%/yr
- Net of RFR offset (the borrowed 0.5 earns no RFR): net cost ≈ 0.5 × 0.75% = 0.375%/yr drag
- **Realistic production CAGR**: 9.57% - 0.4% = **9.17%/yr**
- **Realistic production Sharpe**: 0.643 - 0.025 = **~0.62**

### 3.2 Forward decay (academic priors)

- K1 BAB post-publication decay: -10-20% Sharpe
- D-PEAD post-decimalization decay: -20-30% Sharpe from in-sample 0.92 → forward 0.6-0.7
- Path N speculative (self-discovered, no prior decay)
- Estimated portfolio Sharpe degradation: -0.05-0.10
- **Realistic forward production Sharpe**: ~0.52-0.57

### 3.3 Risk factors

| Risk | Severity | Mitigation |
|---|---|---|
| Margin call in extreme tail (e.g., 2008-style -30% week) | 🟠 medium-high | 1.5x leverage well below 2-3x margin ceiling; circuit breaker existing infra; weekly_recon monitoring |
| Forward decay across all 5 sleeves | 🟠 medium-high | Sprint E E-1 audit 2026-07-15 forward window evidence trigger |
| Broker leverage access in market stress | 🟡 medium | Tier 3 mandates broker pre-approval before real-capital deployment |
| 2022-style stagflation amplified by leverage | 🟠 medium-high | Confirmed empirically -6.67% DD; manageable but acceptable upper bound |
| TC scaling | 🟡 low | 0.07% → ~0.10% annual TC drag; acceptable |

---

## 4. Doctrine compliance

- [x] Pre-registration: no new pre-registered spec required (this is SAA mandate amendment, not strategy test)
- [x] 0-LLM-in-DECISION: this memo informational; Tier 3 supervisor decides
- [x] Anti-HARK: leverage doesn't change strategy verdicts (V/W/X/Z/AA-AO all stand); only changes notional exposure of approved 5-sleeve
- [x] Falsification chain unaffected: not a new spec test
- [x] Memory updated: `project_overlay_structurally_inappropriate_2026-05-15.md` documents why we chose leverage over overlay

---

## 5. Implementation specifics

### 5.1 Code changes

**File**: `engine/portfolio/paper_trade_combined.py`

```python
# NEW constant
LEVERAGE_FACTOR: float = 1.5

# Update PAPER_TRADE_SLEEVE_ALLOCATION docstring to note leverage applied at combine step
# Sum of weights stays 1.0; leverage applied via LEVERAGE_FACTOR in run_paper_trade_day
```

**File**: `engine/portfolio_sleeves.py`

```python
# Update SleeveCapitalConfig to allow sum != 1.0 (allow up to 2.0 for leveraged composition)
# Or apply LEVERAGE_FACTOR at combine_sleeve_weights level
```

**Effective allocation after leverage**:
```
Each strategy notional × LEVERAGE_FACTOR
Daily NAV: simulated borrow cost = (LEVERAGE_FACTOR - 1) × WEEKLY_RFR per week
Real money production: actual broker margin facility cost
```

### 5.2 Testing strategy

- Smoke test 5-sleeve × 1.5x on backtest: verify vol/Sharpe/DD match this memo
- Forward paper-trade orchestrator: emit weighted positions with notional 1.5x
- Watchdog rule for leverage drift (if combined NAV diverges from intended 1.5x exposure)

### 5.3 Rollback condition

- If forward 1.5x leveraged Sharpe < 0.30 sustained 12 weeks → automatic alert via weekly_recon
- Tier 3 supervisor manual rollback authority retained at all times

---

## 6. What this does NOT do

- ❌ Does NOT change PAPER_TRADE_SLEEVE_ALLOCATION sleeve weights (K1/D-PEAD/Path N/CTA/AC unchanged)
- ❌ Does NOT deploy AC at increased weight
- ❌ Does NOT add new sleeves
- ❌ Does NOT modify DEFAULT_INITIAL_ALLOCATION (real-capital state)
- ❌ Does NOT touch any existing strategy spec hash-lock
- ❌ Does NOT auto-promote to real-capital (requires separate Tier 3 + broker integration)

This is **paper-trade mandate amendment ONLY**. Real-capital leverage requires separate governance event.

---

## 7. Tier 3 decision request

**Please respond with one of**:

(a) ✅ **APPROVE 1.5x leverage on existing 5-sleeve**: implement LEVERAGE_FACTOR=1.5, forward paper-trade begins with leveraged composition 2026-05-16 06:00 SGT.

(b) ⏸ **APPROVE WITH MODIFICATION**: deploy at different leverage (1.2x? 1.7x? 2.0x?). Re-run §2 evaluation at chosen leverage.

(c) 🟡 **HOLD**: defer leverage decision until forward window evidence (Sprint E E-1 audit 2026-07-15) shows 5-sleeve doesn't decay.

(d) ❌ **REJECT**: keep current 1x leverage; accept conservative vol 5.71% / return 7.67% profile.

---

## 8. Recommended approval reasoning (per (a))

- ✅ Achieves institutional vol target 8.56% (within 8-10% standard range)
- ✅ Sharpe preserved (0.6431 paper, ~0.55-0.62 production realistic)
- ✅ Max DD acceptable (-8.82% within institutional -10% range)
- ✅ Simpler than multi-insurance composition (math proves AC-up only +0.07pp DD benefit)
- ✅ Independent of widening saturation (free-data alpha exhausted but this works)
- ✅ Defers premium-data / broker / real-capital decisions to appropriate gating triggers
- ✅ Reversible at any time (Tier 3 retains rollback authority)

---

## 9. Cross-references

- `engine/portfolio/paper_trade_combined.py` (implementation target)
- `engine/portfolio_sleeves.py` (SleeveCapitalConfig potentially extended)
- `docs/decisions/saa_path_ac_addition_review_2026-05-15.md` (AC addition Tier 3 memo precedent)
- `docs/spec_path_ao_tail_only_overlay_v3_v1.md` (AO FAIL revealing AC structural limit)
- Memory `project_overlay_structurally_inappropriate_2026-05-15.md` (why no overlay)
- Memory `project_uncovered_sleeve_axes_2026-05-15.md` (uncovered axes inventory)
- Academic:
  - **Modigliani-Miller 1958**
  - **Sharpe 1966**
  - **Asness 2012** "Leverage Aversion and Risk Parity"
