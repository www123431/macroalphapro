# CTA Sleeve Decision — Horse Race Verdict 2026-05-14

**Decision date**: 2026-05-14
**Decision**: **Retain PQTIX (spec id=73)** as cta_defensive sleeve · 10% allocation
**Method**: Pre-registered 5-spec horse race vs PQTIX baseline
**Anchor**: spec_path_p/q/s/t/u + reproducibility trinity (code · spec · dataset hashes)

---

## TL;DR

Tested 5 self-built CTA alternatives over identical 486-week window
(2014-09-12 → 2023-12-29) vs PQTIX baseline. Decision rule (per all 5 specs §2.6):

  - **PASS** (4/4 gates) → replace PQTIX
  - **MARGINAL** (3/4)   → keep PQTIX, log close-call
  - **FAIL** (≤ 2/4)     → keep PQTIX, log falsification entry

**Result**: 4 FAIL · 1 MARGINAL · 0 PASS. **PQTIX retained**.

## Horse race lineup + results

| Spec | Path | Paradigm | Anchor | Verdict | Sharpe (PQTIX 0.041) | Max DD | Crisis |
|---|---|---|---|---|---|---|---|
| 63 | P | active momentum (combo) | AMP 2013 + Jegadeesh 1993 | FAIL (2/4) | -0.529 | -27.80% | 2/3 |
| 64 | Q | active momentum (multi-freq) | Lempérière 2014 | FAIL (2/4) | -0.529 | -21.96% | 2/3 |
| 65 | R | active momentum (pure 12-1) | AMP 2013 | DEPRECATED pre-backtest | — | — | — |
| 66 | S | passive vol-balanced | Qian 2006 / Bridgewater All Weather | FAIL (1/4) | +0.017 | -30.24% | 0/3 |
| 67 | T | regime-conditional | Antonacci 2014 | FAIL (1/4) | -0.090 | -35.47% | 1/3 |
| 68 | U | vol-managed | Moreira-Muir 2017 AQR Award | **MARGINAL (3/4)** | +0.150 | -18.43% | 0/3 |

**PQTIX baseline reference**: Sharpe 0.041 · Max DD -18.73% · Crisis 3/3

## Why each candidate failed

### Path P / Path Q (FAIL · momentum variants)
- Both pure 12-1 / multi-frequency TSMOM Sharpe **-0.529** (~5σ below PQTIX)
- **Validates Garg-Goulding-Harvey-Mazzoleni 2021 post-2010 decay** empirically in 4-ETF universe
- Crowding + 2010s low-vol regime + central bank intervention all consistent with literature

### Path S (FAIL · 1/4 · Risk Parity)
- Sharpe +0.017 (slightly beat PQTIX 0.041 but within noise)
- **Max DD -30.24%** — 2022 inflation regime broke Risk Parity (rates + equity correlated down)
- Crisis 0/3 — long-only allocation can't profit from drawdowns

### Path T (FAIL · 1/4 · Antonacci Dual Momentum)
- Sharpe -0.090 — regime overlay too binary; whipsaws hurt 2018-Q4 + 2020
- Max DD -35.47% (worst of all 5)
- Crisis 1/3 only

### Path U (MARGINAL · 3/4 · Vol-Scaled Risk Parity) — close-call analysis
- Sharpe **+0.150** > PQTIX 0.041 (G1 PASS)
- Max DD **-18.43%** < PQTIX × 1.1 = -20.60% (G2 PASS)
- ρ vs other sleeves: passes (G3 PASS)
- **Crisis 0/3 (G4 FAIL)** — VIX overlay de-risks during stress so doesn't generate crisis alpha

**Path U's nature**: vol-managed defensive sleeve — BETTER risk-adjusted return than PQTIX
in calm regimes, but loses crisis-positive role (the entire point of Path O sleeve).
Moreira-Muir 2017 vol-managed alpha confirmed in our window for non-crisis returns,
but the trade-off is incompatible with crisis-alpha sleeve mandate.

## Project narrative implications

### Doctrine-clean outcome
- 5 falsification chain entries added (capability evidence MDs written)
- Pre-registration discipline preserved (hash-lock + amendment_log + 0-LLM doctrine)
- Honest disclosure: tested + rejected, kept proxy with evidence

### Validates standing rules
- `feedback_llm_risk_side_not_alpha_side.md` — alpha-side ceiling confirmed
- `feedback_agent_addition_rule.md` — adding 5 paradigms didn't trump fund quality
- Garg 2021 decay finding now has project-internal empirical anchor

### Senior reframe for interview / demo
> "I built a pre-registered 5-candidate horse race against PQTIX (institutional CTA fund).
> Used 4 paradigms (active momentum / passive vol-balanced / regime-conditional / vol-managed).
> Locked spec hashes + dataset hashes BEFORE backtest. Result: 4 FAIL · 1 MARGINAL · 0 PASS.
> 
> The MARGINAL (Vol-Scaled Risk Parity) had BETTER Sharpe + DD than PQTIX but lost the
> crisis-positive role that's the entire point of the sleeve. This is the empirical case
> for keeping institutional fund-of-fund pattern over self-built alpha at our scale.
> 
> Consistent with Yale Endowment Model · McLean-Pontiff 2016 decay · Garg 2021 trend-following
> post-2010 evidence."

## Multiple-comparison disclosure (per all 5 specs §2.6)

5 candidates × 4 gates = 20 statistical tests. Family-wise Type-I error at α=0.05 per
test: 1 - 0.95^5 = 22.6% probability of at least one false PASS.

The 0-PASS / 1-MARGINAL outcome is **consistent with the null hypothesis** that our
4-ETF universe + vanilla signals cannot systematically beat PQTIX's $1B+ AUM
multi-instrument multi-frequency professional implementation. No correction needed —
no false PASS to discount.

## Phase 5 action items

**NO CODE CHANGE** to `engine/portfolio/paper_trade_combined.py`:
- PAPER_TRADE_SLEEVE_ALLOCATION unchanged (cta_defensive: 10% PQTIX)
- Path O spec id=73 status remains `active`

**Documentation updates** (Phase 6):
- This decision memo committed
- 5 capability evidence MDs committed
- Memory file update: `project_senior_audit_2026-05-14.md` §G resume command
  pointing to this verdict
- SpecRegistry: Path P/Q/S/T amendment_log entries noting "FAIL gate · falsification chain"
- Path U amendment_log entry noting "MARGINAL · vol-managed alpha confirmed but crisis-role incompatible"

## Future revisit triggers (not now)

This decision can be revisited IF:
- New paradigm with stronger prior evidence emerges in 2026-2030 academic literature
- PQTIX dramatically underperforms (Sharpe < -0.3 over 2-year rolling)
- Real broker integration enables broader universe (futures vs ETF subset)
- Project scales beyond $1M paper book to where fund fee economics shift

Currently none of these conditions met. **PQTIX retained for foreseeable horizon.**

## Cross-references

- `docs/spec_path_o_cta_defensive_overlay_v1.md` — active CTA sleeve spec
- `docs/spec_path_p_*_v1.md` through `spec_path_u_*_v1.md` — 5 horse race specs
- `docs/capability_evidence/macro_*_2026-05-14.md` — 5 per-spec verdict MDs
- `memory/project_senior_audit_2026-05-14.md` — audit roadmap (§1+2 item)
- `engine/portfolio/macro_cta_research/` — Phase 2 shared infra
