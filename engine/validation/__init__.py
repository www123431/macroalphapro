"""
engine/validation/ — alpha-side validation harness (Phase 1, 2026-05-19).

The honest "do I even have alpha" layer. Built AFTER the agent / workflow
infrastructure because the project's sequencing put the operational
scaffolding first; this module is the catch-up on the actual core.

Modules:
  deflated_sharpe.py   — Probabilistic + Deflated Sharpe Ratio
                         (Bailey & López de Prado 2014). Corrects an
                         observed Sharpe for sample length, non-normality,
                         AND the number of trials run during research.
  factor_data.py       — Ken French FF5 + Momentum loader (daily fetch
                         via pandas_datareader, resampled to weekly,
                         cached to parquet).
  factor_attribution.py— Regress each strategy's excess return on the
                         factor set; the regression intercept is the
                         residual alpha (the part that is NOT just
                         cheaply-buyable factor beta).
  report.py            — Combine into one per-strategy table:
                         naive Sharpe / deflated Sharpe / residual alpha
                         / t-stat / R².

Doctrine: this is diagnostic, read-only, deterministic. It does not
trade, does not mutate state. Its entire job is to tell the PM the
truth about which surviving strategies hold up under multiple-testing
correction and factor decomposition.
"""
from engine.validation.deflated_sharpe import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_ratio,
)

__all__ = [
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "sharpe_ratio",
]
