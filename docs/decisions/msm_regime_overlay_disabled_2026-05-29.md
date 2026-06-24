# Decision — MSM regime overlay DISABLED at book level

**Date**: 2026-05-29
**Type**: Production deployment change, ablation-driven
**Touches**: `engine/portfolio_core.py` (new constant `ENABLE_REGIME_OVERLAY = False` + gate at top of `construct_portfolio`)
**Affects**: live paper-trade rebalance (`engine/portfolio_tracker.py:332`) and walk-forward backtest Portfolio B (`engine/backtest.py:666`); MSM module itself unchanged

## TL;DR

Walk-forward ablation 2018-01 → 2025-12 (95 months) proves the MSM regime
overlay HURTS the sector-ETF TSMOM book. We disable the overlay at the book
level and keep MSM running for daily-brief / dashboard reporting only.

## Evidence

`scripts/ablation_msm_on_vs_off.py` runs the existing `run_backtest` and
extracts both portfolios that the engine already computes on every rebalance:
- `tsmom`        = MSM-OFF (`regime=None` passed to `construct_portfolio`)
- `tsmom_regime` = MSM-ON  (`regime=regime_r` passed to `construct_portfolio`)

Full results: `data/ablation/msm_on_vs_off_2018_2025.json`

| Metric                                       | MSM-ON  | MSM-OFF | Δ (ON−OFF) |
|----------------------------------------------|---------|---------|------------|
| Sharpe (95 months, monthly)                  | +0.075  | +0.336  | **−0.262** |
| MaxDD                                        | −7.00%  | −4.94%  | −2.06pp    |
| Risk-off subperiod Sharpe (n=91)             | +0.071  | +0.340  | **−0.269** |
| Overlay-helped rate in risk-off              | 42.9%   | —       | < coin-flip |
| Δ Sharpe stationary bootstrap mean (n=2000)  | —       | —       | **−0.295** |
| Δ Sharpe 95% CI                              | —       | —       | [−0.762, +0.049] |

## Why this happened

- Over 95 months, MSM labelled **91 as risk-off** — effectively a permanent
  "risk-off" classifier given current MSM design (Hamilton MS on monthly
  yield_spread with switching variance + 0.65 prob threshold).
- A near-permanent risk-off label triggered the overlay's long-shrink
  (× 0.30) and concentrated position caps (`max_long=5`) for the entire
  2018-2025 bull market → systematic missed-bull cost.
- Conditional on the regime label, MSM-ON underperformed MSM-OFF inside
  the very subperiods where the overlay is supposed to add value
  (Δ = −0.27 inside the n=91 "risk-off" cluster).
- Statistical headline: bootstrap mean −0.30, 95% CI [−0.76, +0.05].
  Upper bound just barely crosses 0, so strict significance is not
  achieved — but the central effect and the consistent direction across
  every metric (Sharpe, MaxDD, subperiod Sharpe, hit rate) is overwhelming.

## What we changed

```python
# engine/portfolio_core.py
ENABLE_REGIME_OVERLAY = False
```

and at the top of `construct_portfolio`:

```python
if regime is not None and not ENABLE_REGIME_OVERLAY:
    regime = None
```

This short-circuits **all** downstream regime touches in one place:
- Step 3 `_regime_adjusted_cov` (LW cov +30% correlation shrink in risk-off)
- Step 5 long-weight × `regime_scale (0.3)` shrink
- `_get_position_limits` regime-conditional `(max_long, max_short)` caps

Call sites are intentionally not touched — `engine/backtest.py` still
computes both Portfolio A and B every walk-forward so the ablation harness
remains usable for re-verification. With the gate OFF, Portfolio A and B
now produce identical NAVs (smoke-verified: `data/ablation/_smoke_after_gate.json`).

## What we kept

- `get_regime_on()` keeps running and is still consumed at 9 sites in
  `engine/daily_batch.py` for daily-brief reporting + flip detection.
- MSM fit code (`_fit_and_filter`) is still wired and produces filtered
  `p_risk_on` for those reports.
- Reverse path (re-enabling) is a single-line constant flip.

## When to re-enable

Flip `ENABLE_REGIME_OVERLAY = True` only with **fresh** ablation evidence
on a different code path or a redesigned overlay where:
- Δ Sharpe ≥ +0.10 with the **same** Politis-Romano stationary bootstrap
- CI excludes 0
- MaxDD does not worsen
- Subperiod evidence consistent with bootstrap headline

Anything weaker = decoration with negative EV, do not redeploy.

## Notes on what this is NOT

- This does **not** invalidate the combined-book Sharpe 1.10 (`combined_book.py`):
  that book never used MSM. It's an equity (D-PEAD + revision) + 4-leg
  cross-asset carry book; carry sleeve build is regime-blind.
- This does **not** retire `engine/regime.py` — the module is methodologically
  fine (Hamilton 1989, filtered-only, walk-forward refit, BIC k-selection).
  What the ablation closes is the deployment question: "should this label
  touch portfolio weights at all?" Verdict: not in the current overlay form.
- This does **not** preclude a future redesigned overlay (e.g., shorter
  threshold band, asymmetric prob gate, different scaling). Any such
  redesign must re-pass the bar above.

## Files / artefacts

- Decision (this file): `docs/decisions/msm_regime_overlay_disabled_2026-05-29.md`
- Gate constant + comment block: `engine/portfolio_core.py:83-99`
- Gate short-circuit in `construct_portfolio`: `engine/portfolio_core.py:240-247`
- Ablation harness: `scripts/ablation_msm_on_vs_off.py`
- Evidence (8-year): `data/ablation/msm_on_vs_off_2018_2025.json`
- Evidence (post-gate smoke): `data/ablation/_smoke_after_gate.json`
