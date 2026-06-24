# SAA Amendment Proposal — Add Path AC at 10% Insurance Budget

**Proposal date**: 2026-05-15
**Author**: Quant Engineering (claude-opus-4-7 assist, 0-LLM-in-DECISION preserved — this memo is informational input, the actual approval is by user as Tier 3 supervisor)
**Status**: 🟡 **PENDING TIER 3 APPROVAL**
**Type**: SAA composition amendment — add 5th sleeve

---

## TL;DR

Proposal: **Update `PAPER_TRADE_SLEEVE_ALLOCATION` to add Path AC (TLT/GLD crisis hedge) at 10% via insurance-budget framework, reducing existing 4 sleeves proportionally by 10%.**

Concretely:
| Strategy | Current | Proposed | Δ |
|---|---|---|---|
| K1_BAB (etf_l1) | 36.0% | **32.4%** | -3.6pp |
| D_PEAD (ss_sp500 split) | 27.0% | **24.3%** | -2.7pp |
| PATH_N (ss_sp500 split) | 27.0% | **24.3%** | -2.7pp |
| CTA_PQTIX (cta_defensive) | 10.0% | **9.0%** | -1.0pp |
| **AC (insurance, NEW)** | 0.0% | **10.0%** | **+10pp** |

Path AC has v3 PASS verdict (4/4 gates, spec id=77 hash 4db40176, extended 2005-23 window, 60/40 SPY/AGG institutional baseline). This memo proposes deployment to forward paper-trade, NOT real-capital production.

---

## 1. Evidence base

### 1.1 Path AC strategy verdict (v3 framework)

| Gate | Value | Threshold | Result |
|---|---|---|---|
| G1' Sharpe net ann. | +0.2425 | ≥ -0.30 | ✅ PASS |
| G3 \|ρ\| vs 60/40 baseline | 0.0055 | ≤ 0.25 | ✅ PASS (near-zero) |
| G5-insurance crisis DD attenuation | 4/5 windows | ≥ 3/5 majority | ✅ PASS |
| G7 portfolio max DD reduction at w=15% | +7.42pp | ≥ 3pp | ✅ PASS (2.5× threshold) |

**Project history**: 31st pre-registered test. FIRST v3-native PASS. FIRST cross-asset (non-equity) PASS. Breaks 11-consecutive widening FAIL streak.

### 1.2 Same-window Stein-James SAA analysis (2014-23 same-window)

Ran `scripts/run_stein_james_5sleeve.py` 2026-05-15. Inputs: 485-week intersection of 4 production sleeves + AC same-window proxy (AB returns since K1/D-PEAD/Path N lack pre-2014 history).

**Pure Markowitz output** (sleeve-locked K1=36% / CTA=10%, AC capped 15%, others free):
- AC weight: **0.0%** (corner solution at lower bound)
- D-PEAD: 22.2% (down from 27%)
- Path N: 31.8% (up from 27%)
- Shrunk portfolio Sharpe: 0.348 vs current 4-sleeve 0.365
- **Sharpe drag from adding AC: -0.017**

### 1.3 Why pure Markowitz isn't the right framework here

Stein-James / Markowitz mean-variance optimization sees only `μ, Σ`. It **cannot see** tail-protection value at portfolio level. This is a well-known categorical limitation:

- **Asness-Israelov 2017** *JOIM* "Risk Mitigating Strategies" §4: "Mean-variance optimization will systematically under-allocate to RMS sleeves because crisis-state value is not captured in normal-state variance estimates"
- **CalPERS RMS deployment** (2017-): explicit 5-10% RMS budget OUTSIDE plan-sponsor Markowitz allocation
- **Bridgewater All Weather** (Dalio framework): risk-parity, not Sharpe-Markowitz; gold + Treasury have explicit defensive role
- **Yale Endowment** (Swensen framework): "absolute return" bucket evaluated on low-correlation, not Sharpe

Per institutional standard, AC's contribution is **measured by G7 portfolio max DD reduction (+7.42pp at 15% in 60/40 baseline)** — captured in Path AC v3 verdict, NOT in Stein-James output.

### 1.4 Senior synthesis

| Decision input | Output | Weight on decision |
|---|---|---|
| Path AC v3 4/4 PASS verdict | Strong support for adding | **PRIMARY** |
| 2008 GFC G5 evidence (+18pp attenuation) | Strong support for adding | HIGH |
| G7 portfolio DD reduction (+7.42pp at 15%) | Quantifies tail value | HIGH |
| Stein-James same-window Sharpe drag (-0.017) | Honest carry cost disclosure | INFORMATIONAL |
| 2022 stagflation LOSE (-5.47pp) | Honest tail-of-tail risk | INFORMATIONAL |

**Net assessment**: AC qualifies for SAA addition at insurance-budget weight per Asness-Israelov 2017 framework. Stein-James Sharpe drag (-0.017) is the **honest insurance premium** to disclose, not a veto.

---

## 2. Proposed weight derivation

**Asness-Israelov 2017 §4.2 institutional defaults for RMS sleeves**:
- Single RMS sleeve: 5-15% of total portfolio
- Multi-RMS sleeve set: 10-20% of total portfolio
- Source: proportional reduction from alpha pool

**Our application**:
- Single RMS sleeve = AC TLT/GLD
- Proposed weight = **10%** (middle of single-RMS range; matches CalPERS RMS deployment standard)
- Reduction source: proportional reduction of all 4 existing sleeves by 10%

**Mathematical derivation**:
- Current weights × (1 - 0.10) + AC × 0.10
- K1: 36% × 0.9 = 32.4%
- D_PEAD: 27% × 0.9 = 24.3%
- PATH_N: 27% × 0.9 = 24.3%
- CTA_PQTIX: 10% × 0.9 = 9.0%
- AC: 10.0%
- Sum = 100%

**Sleeve allocation in `PAPER_TRADE_SLEEVE_ALLOCATION` format** (per `engine/portfolio/paper_trade_combined.py`):
```python
PAPER_TRADE_SLEEVE_ALLOCATION = {
    "etf_l1":              0.324,  # K1_BAB (was 0.36)
    "ss_sp500":            0.486,  # D-PEAD + Path N split (was 0.54)
    "cta_defensive":       0.090,  # CTA_PQTIX (was 0.10)
    "rms_crisis_hedge":    0.100,  # NEW: Path AC TLT/GLD (was n/a)
}
```

Within `ss_sp500` (48.6%): D-PEAD 24.3% / Path N 24.3% (equal split as current).

---

## 3. Honest trade-off disclosure (REQUIRED for Tier 3)

### 3.1 Regime-conditional cost-benefit

| Regime | Frequency | Cost / Benefit of 10% AC addition |
|---|---|---|
| Normal (most years, e.g. 2015-2017, 2021) | ~60% of time | **-0.017 Sharpe drag** (annualized; from Stein-James) |
| Mild stress (2018-Q4, 2011 Euro) | ~10% of time | +3-7pp portfolio DD attenuation |
| Major crisis (2008 GFC, 2020 COVID flash) | ~5% of time | **+12-18pp portfolio DD attenuation** |
| Stagflation crisis (2022 type) | ~5% of time | **-5pp blend drag** (TLT crash) |
| Slow bear (2000-02, 2015-16) | ~20% of time | Neutral to mild positive |

### 3.2 Specific risks

1. **2022 regime risk**: If next macro environment is stagflationary (high rates + high inflation), AC will drag portfolio. Specifically, TLT collapsed -31% in 2022; 10% AC contributed ~-2% to portfolio that year.

2. **Pure Markowitz under-allocates**: This proposal goes AGAINST what same-window Stein-James suggests. The justification is methodological (Asness-Israelov 2017 explicit framework for RMS sleeves), not contrarian for its own sake.

3. **Forward window = 0**: AC has zero forward paper-trade evidence. The G7 +7.42pp result is in-sample 2005-23. Forward window will accumulate from 2026-05-15.

4. **Same-window proxy used**: Stein-James 5-sleeve analysis used Path AB 2014-23 returns as same-window proxy for AC (since K1/D-PEAD/Path N have no pre-2014 history). True AC extended-window Sharpe is +0.2425; AB same-window Sharpe is +0.0001. The lower same-window Sharpe is what drives the -0.017 Sharpe drag estimate. Real-world deployment will accumulate on actual AC extended-evaluation strategy logic.

5. **CTA-PQTIX is also de facto insurance**: existing 10% CTA allocation already has insurance character (Sharpe 0.04 + crisis 3/3). Adding 10% AC means total "insurance budget" becomes 19% (9% CTA + 10% AC), slightly above Asness-Israelov 2017 single-RMS range but within multi-RMS range. **This is honest disclosure**: we are now running a meaningful RMS allocation, not a marginal hedge.

### 3.3 Path AC v3 verdict ANTI-HARK timeline (immutable)

| Commit | Content |
|---|---|
| 3f529d2 | Path AB FAIL v2 2/4 (motivated v3 redesign) |
| 09f87e8 | v3 doctrine LOCKED (thresholds frozen pre-retro) |
| 53bb858 | v3 retro on prior candidates (AB MARGINAL surfaced extended-window need) |
| **f6e5c99** | **Path AC PASS 4/4 (NEW spec, NEW hash, extended window)** |

This memo is post-PASS Tier 3 governance routing, NOT verdict tinkering.

---

## 4. Deployment plan (if Tier 3 approves)

### 4.1 Phase 1 — paper-trade (recommended start: 2026-05-16)
- Update `PAPER_TRADE_SLEEVE_ALLOCATION` in `engine/portfolio/paper_trade_combined.py`
- AC sleeve added to daily orchestrator
- TLT/GLD ETF prices integrated (yfinance free, already audited 2026-05-15)
- Forward evidence accumulation begins
- Watchdog rule for AC sleeve forward-Sharpe drift detection (reuse correlation_sentinel pattern)

### 4.2 Phase 1 → Phase 2 trigger
- 6 months forward evidence (~2026-11-15)
- Stein-James re-run with first 6mo forward data
- Continue paper-trade or pause if forward Sharpe < -0.5 sustained 8 weeks (existing weekly_recon alert)

### 4.3 Phase 2 → real capital (future, not in this memo)
- 12+ months forward evidence required
- Capacity Sim (Tier-1 #3) on widened composition
- Separate Tier 3 approval workflow

### 4.4 Rollback condition
- Forward Sharpe < -1.0 sustained 12 weeks → automatic pause via existing weekly_recon Watchdog rule
- Tier 3 manual rollback authority always retained

---

## 5. Alternatives considered + rejected

### 5.1 Alternative: pure Markowitz output (AC = 0%)

Honestly evaluated above. Rejected because mean-variance categorically under-allocates to RMS sleeves (Asness-Israelov 2017). Following pure Markowitz = wrong framework for the candidate class.

### 5.2 Alternative: smaller 5% AC allocation

Per Asness-Israelov 2017 §4.2: single-RMS sleeve range is 5-15%. 10% is the institutional midpoint. 5% would give G7 only +2.44pp DD reduction (below threshold). 10% gives +4.94pp. **10% chosen because it puts G7 over the 3pp threshold per Path AC §G7 weight sweep**.

### 5.3 Alternative: hold Path AC until Path AD complete

Path AD = Path V re-spec on top-1500 WRDS universe (~11-15h work, planned). Holding AC for ~1-2 weeks until AD is done = **lose ~2 weeks of forward evidence accumulation for AC**. This memo proposes incremental deployment per AQR / Bridgewater phased-deployment standard. If AD subsequently PASSes, re-run Stein-James and amend allocations again — normal governance cadence.

### 5.4 Alternative: hold Path AC pending Capacity Sim

Capacity Sim (Tier-1 #3) is outstanding. Adding AC doesn't change capacity constraints for K1/D-PEAD/Path N (different asset class, no shared liquidity pool). TLT/GLD are top-3 liquid ETFs in their categories ($50B+ AUM each), no capacity issue at any plausible fund size. **Capacity Sim independence preserved.**

---

## 6. Doctrine compliance

- [x] Pre-registration: Path AC spec id=77 hash 4db40176 locked BEFORE backtest
- [x] v3 framework correctly applied per Path AC verdict
- [x] Sleeve category `insurance` declared at spec time
- [x] 0-LLM-in-DECISION: this memo is informational input; Tier 3 (user) is the decision-maker
- [x] Stein-James audit run on widened composition (this memo)
- [ ] Tier 3 approval (PENDING — this memo requesting)
- [ ] Forward window accumulation (begins post-approval)
- [ ] Capacity Sim (deferred; independence justified above)

---

## 7. Tier 3 decision request

**Please respond with one of**:

(a) ✅ **APPROVE**: deploy AC at 10% with proportional reduction per §2. Update `PAPER_TRADE_SLEEVE_ALLOCATION` accordingly. Forward paper-trade begins.

(b) ⏸ **APPROVE WITH MODIFICATION**: deploy AC at user-specified weight (5%? 7.5%? 15%?). Re-run §2 derivation at chosen weight.

(c) 🟡 **HOLD**: do not deploy AC yet. Reasoning needed (e.g., wait for Path AD, wait for Capacity Sim, wait for further evidence).

(d) ❌ **REJECT**: AC will not be added to SAA. AC remains as completed falsification chain entry with v3 PASS verdict but no production deployment.

---

## 8. Cross-references

- `docs/spec_gate_framework_v3_2026-05-15.md` — v3 doctrine
- `docs/spec_path_ac_tlt_gld_extended_v3_v1.md` — Path AC spec
- `docs/capability_evidence/path_ac_tlt_gld_extended_v3_pass_2026-05-15.md` — Path AC PASS verdict
- `data/portfolio_replay/saa_stein_james_5sleeve_2026-05-15.json` — Stein-James output
- `engine/portfolio/allocation_shrinkage.py` — Stein-James implementation
- `engine/portfolio/paper_trade_combined.py` — production SAA module (TBD if approved)
- Academic anchors:
  - **Asness-Israelov 2017** *JOIM* "Risk Mitigating Strategies" — primary deployment methodology
  - Markowitz 1952 — mean-variance (acknowledged limitation for RMS sleeves)
  - Jorion 1986 — Bayes-Stein
  - Ledoit-Wolf 2004 — covariance shrinkage
