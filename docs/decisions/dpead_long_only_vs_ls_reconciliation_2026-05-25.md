# Decision — D_PEAD long-only (deployed) vs L/S (spec/validated) reconciliation (2026-05-25)

**Context**: the "live delivers backtest" audit (docs/live_delivers_backtest_audit_2026-05-25.md)
found that the deployed D_PEAD is **long-only top-decile**, while spec id=62
(spec_path_d_dhs_behavioral_2factor_v1.md §2.3 + Amendment A.1) describes **long − 0.7·short**
(market-neutral-ish L/S), and the combined-book 1.04 was validated on a **full L/S** recon
(`_dpead_recon_base`, corr 0.895 w/ a clean from-source rebuild — see engine/portfolio/dpead_recon.py).
Three variants: live w=0 / recon w=1.0 / spec w=0.7.

## Verdict: this is NOT a bug. KEEP the live D_PEAD long-only.

The live paper book is a **deliberately LONG-BIASED multi-strategy book**, not a market-neutral
alpha book. Evidence:

1. **Directional sleeves by design**: rms_crisis_hedge (TLT/GLD) is long (crisis insurance);
   cta_defensive (PQTIX) is directional trend. These carry intentional net exposure.
2. **The MSM regime overlay presumes a directional book.** Production stack is
   `ql01_bab × multivariate_v3 × REGIME_SCALE=0.6` (engine/regime.py): in risk-off regimes it
   **scales the book down to 60%** (de-risk). A regime-de-risking overlay only makes sense on a
   book that carries directional (net) exposure to shed — a market-neutral book (net≈0) has
   almost nothing to de-risk. So the book's *architecture* is built around being long-biased.
3. **Quantified**: flipping D_PEAD to L/S (w=0.7) at the current allocation
   (ss_sp500 0.486 × intra 0.5 × LEV 1.5 = D_PEAD book wt 0.364) moves D_PEAD net 0.364→0.109,
   gross→0.620 → **book net +16.4% → −9.1% (flips NET SHORT), gross 1.08 → 1.34**. Flipping
   D_PEAD alone turns the long-biased book net-short — clearly not the intended posture.

Therefore long-only D_PEAD is **consistent with the book's deliberate long-biased,
regime-managed design**. The L/S D_PEAD + the 1.04 combined-book are a **separate
market-neutral RESEARCH / alpha-isolation construct** that validates the SIGNAL's edge
(Sharpe 1.43 at w=0.7 per spec A.1; 1.04 combined with carry) — they are NOT a mandate that the
deployed sleeve be market-neutral.

## Honest consequence (do not paper over)

The **live long-biased book's realized performance is NOT the 1.04.** The 1.04 is the
market-neutral alpha+carry research Sharpe; the live book's return is dominated by market beta +
the long-SUE / insurance / trend tilts, and is a different (un-isolated) number. Claims about the
book should not conflate the two.

## Actions

- **KEEP live D_PEAD long-only** (book posture unchanged; no live trading change). ✅ done (no change).
- **Step-1 flag retained**: `get_d_pead_signal(short_leg_weight=…)` implements the spec L/S,
  **default 0.0 = long-only**. This is the enabler IF a true market-neutral re-architecture is
  ever deliberately chosen — that would be a **separate project**: re-scale all sleeve allocs /
  LEVERAGE to a chosen net/gross target, re-check RM gross-net gates, re-validate the book.
- **RECOMMENDED governance follow-up (NOT done unilaterally — needs amend_spec)**: a clarifying
  amendment to spec id=62 distinguishing (a) the **L/S alpha-isolation construction** used for
  validation from (b) the **long-only deployed sleeve** within the long-biased book, so the spec
  no longer reads as mandating a market-neutral deployment it never intended.

## Residual (separate, lower priority)
The deployed signal is also missing the spec's **FIN factor** (2-factor DHS = PEAD + FIN); the
live is PEAD-TS long-only only. Same family of "deployed is a simplification of spec" — fold into
the spec-clarification amendment.
