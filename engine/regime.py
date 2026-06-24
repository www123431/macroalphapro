"""engine/regime.py — DEPRECATED 2026-05-29 (soft delete).

The MSM regime detection machinery (Hamilton 1989 Markov Switching with
BIC k-selection on yield_spread, plus VIX co-input) was removed after
walk-forward ablation evidence (2018-01 → 2025-12, 95 months) showed the
overlay HURTS the book:

  Sharpe MSM-ON  +0.075 vs MSM-OFF +0.336  (Δ -0.262)
  MaxDD MSM-ON  -7.00% vs MSM-OFF -4.94%  (-2pp worse)
  Bootstrap mean Δ Sharpe -0.295, 95% CI [-0.762, +0.049]
  Inside MSM-flagged risk-off subperiods (n=91): ON +0.071 vs OFF +0.340

Full evidence: data/ablation/msm_on_vs_off_2018_2025.json
Decision doc:  docs/decisions/msm_regime_overlay_disabled_2026-05-29.md

Why the module is gone (not just gated)
---------------------------------------
The three things regime detection was supposed to do are already handled
by other mechanisms in the book — none of them depend on a regime label:

  1. Risk sizing       → vol-target (portfolio_core.TARGET_VOL = 10%
                          per sleeve, combined_book.DEFAULT_BOOK_VOL_TARGET
                          = 10% at book level, both rolling shift(1))
  2. Crisis defense    → permanent crisis-hedge sleeve
                          (Spec 80, 10.37% of book = 75% TLT-GLD + 25% trend)
  3. Signal regime     → mechanism diversification
                          (equity earnings ⊕ 4-leg cross-asset carry, ~0 corr)

The combined_book.py achieves Sharpe 1.10 without importing this module.
Keeping a half-functioning MSM around invited periodic re-enable attempts
(v1 → v2 → v3 → v4 specs all failed); deleting the machinery makes the
"we don't predict regimes; we hedge them" doctrine explicit in code.

What survives
-------------
A minimal stub that preserves the public surface so the ~10 existing call
sites in daily_batch.py / backtest.py / portfolio_tracker.py / signal.py
keep importing and calling without modification. The stub always returns
a neutral "unknown" regime — call sites that condition on "risk-off" /
"risk-on" simply fall through to their no-overlay code paths. This is
defense-in-depth on top of `ENABLE_REGIME_OVERLAY = False` in portfolio_core.

If you want raw macro context for daily brief, use the source numbers
directly (VIX, term spread, OAS) — they carry strictly more information
than a 2-state HMM label.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import pandas as pd


@dataclass
class RegimeResult:
    """Stub-shape result preserved for caller compatibility.

    All fields are deterministic and information-free:
      regime     = "unknown"
      p_risk_on  = 0.5
      p_risk_off = 0.5
      method     = "deprecated"
    """
    date:          datetime.date
    regime:        str
    p_risk_on:     float
    p_risk_off:    float
    method:        str
    n_obs:         int
    yield_spread:  float | None
    vix:           float | None
    warning:       str


_DEPRECATED_WARNING = (
    "regime detection deprecated 2026-05-29 — see "
    "docs/decisions/msm_regime_overlay_disabled_2026-05-29.md"
)

_REGIME_MEMO: dict[datetime.date, RegimeResult] = {}


def clear_regime_memo() -> None:
    """Clear the in-process memo. Used by tests / ablation scripts."""
    _REGIME_MEMO.clear()


def get_regime_on(
    as_of:          datetime.date,
    train_end:      datetime.date | None = None,
    n_train_months: int = 120,
) -> RegimeResult:
    """Deterministic neutral stub. No FRED / VIX / statsmodels involved.

    Returns regime='unknown' so call sites that gate on 'risk-off' or
    'risk-on' branches fall through to their default (no-overlay) code path.
    """
    del train_end, n_train_months  # signature-compat
    cached = _REGIME_MEMO.get(as_of)
    if cached is not None:
        return cached
    result = RegimeResult(
        date=as_of,
        regime="unknown",
        p_risk_on=0.5,
        p_risk_off=0.5,
        method="deprecated",
        n_obs=0,
        yield_spread=None,
        vix=None,
        warning=_DEPRECATED_WARNING,
    )
    _REGIME_MEMO[as_of] = result
    return result


def get_regime_series(
    dates:          list[datetime.date],
    n_train_months: int = 120,
) -> pd.DataFrame:
    """Stub series — every row reports the deprecated neutral result."""
    del n_train_months
    records = []
    for d in dates:
        r = get_regime_on(d)
        records.append({
            "date":         r.date,
            "regime":       r.regime,
            "p_risk_on":    r.p_risk_on,
            "p_risk_off":   r.p_risk_off,
            "method":       r.method,
            "n_obs":        r.n_obs,
            "yield_spread": r.yield_spread,
            "vix":          r.vix,
            "warning":      r.warning,
        })
    return pd.DataFrame(records).set_index("date")


def compare_with_human(model_regime: str, human_regime: str) -> dict:
    """Stub — regime comparison no longer carries information."""
    return {
        "model_regime":  model_regime,
        "human_regime":  human_regime,
        "human_mapped":  "unknown",
        "agreement":     True,
        "divergence":    _DEPRECATED_WARNING,
    }
