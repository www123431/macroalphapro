"""engine.research.ablation — Phase A v3 rigorous weighting-method ablation.

Doctrine (per [[project-position-weighting-precision-queued-2026-06-02]] +
[[project-l1-research-ops-inbox-doctrine-2026-06-02]]):

The deployed equity book uses 1/N equal weight within each leg of a long/
short decile portfolio. This package tests whether alternative weighting
schemes (signal-magnitude / inverse-vol / rank / combinations) provide
genuine OOS Sharpe improvement OVER the deployed baseline, controlling
for sector exposure, capacity constraints, multiple testing, autocorrelation,
and backtest overfitting.

15-axis rigor checklist (set 2026-06-02):
  ✓  1. PBO (Bailey-Borwein-Lopez de Prado-Zhu 2014 JCF)
  ✓  2. CPCV (Lopez de Prado 2018 §7)
  ✓  3. Newey-West HAC SE (overlapping 60d returns ⇒ autocorr)
  ✓  4. Family-aware n_trials accumulated across historical attempts
  ✓  5. Sector neutralization (GICS gsector, point-in-time via gvkey)
  ✓  6. Market-cap floor (drop micro-caps < $500M) + simulated ADV cap
  ✓  7. Multiple signal definitions: SUE_raw / SUE_z / abnormal_sue
  ✓  8. Vol-targeting normalization (each leg → 10% ex-ante vol)
  ✓  9. Metrics battery: Sharpe + Sortino + Calmar + maxDD + CVaR(5%) + skew + kurt
  ✓ 10. Tail attribution (long-only tail vs short-only tail)
  ✓ 11. PIT verification (rdq from Compustat, joined with point-in-time gvkey-sich)
  ✓ 12. Baseline = literal build_equity_book() output (or its decile-equivalent
        reconstruction matching the deployed pipeline exactly)
  ✓ 13. Politis-White 2004 adaptive block-length
  ✓ 14. Theoretical decomposition writeup per variant
  ✓ 15. PSR + DSR (probabilistic + deflated Sharpe ratios)

Modules:
  signals.py    — signal definitions (4 variants)
  weighting.py  — weighting methods (5 variants, decoupled from signal)
  portfolio.py  — L/S decile + sector neutralization + capacity + vol-targeting
  metrics.py    — Sharpe / Sortino / Calmar / maxDD / CVaR + Newey-West HAC
  cpcv.py       — Combinatorial Purged Cross-Validation
  pbo.py        — Probability of Backtest Overfitting
  runner.py     — Main orchestrator + grid search
  report.py     — Publication-quality report writer

CLI: scripts/run_phase_a_v3.py

Output:
  data/research/phase_a_v3_<date>/
    ablation_results.parquet
    cpcv_folds.parquet
    pbo_distribution.parquet
    metrics_battery.parquet
    report.md
"""
__version__ = "3.0.0"
