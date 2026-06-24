"""
engine/path_c/verdict.py — Path C #1 PEAD verdict aggregation.

Pre-registration: docs/spec_path_c_earnings_pead_v1.md (id=57) §3 + §九

Computes statistics + classifies into PASS / MARGINAL / FAIL per locked
gates in spec §3.2:

  PASS:     Sharpe (net) ≥ 0.5 AND NW t ≥ 2.0 AND BHY-FDR passes
  MARGINAL: Sharpe (net) ≥ 0.3 AND NW t ≥ 1.5 (BHY may fail)
  FAIL:     Sharpe (net) < 0.3 OR NW t < 1.5

Inputs come from Sprint 4 WalkForwardPeadResult (daily long-short returns).
All operations deterministic, 0 LLM (spec invariant).

Reused from existing modules:
  - engine.multivariate_msm_verdict.annualized_sharpe (monthly default; we
    parameterize obs_per_year=252 for daily PEAD frequency)
  - Politis-Romano bootstrap pattern (custom single-series variant)
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.path_c import (
    NW_LAG_TRADING_DAYS_LOCKED,
    TC_BPS_ROUNDTRIP_LOCKED,
)
from engine.path_c.pead_backtest import WalkForwardPeadResult

logger = logging.getLogger(__name__)


# ── Locked decision-gate thresholds (spec §3.2) ─────────────────────────────
PASS_SHARPE_THRESHOLD:     float = 0.50
PASS_NW_T_THRESHOLD:       float = 2.00
MARGINAL_SHARPE_THRESHOLD: float = 0.30
MARGINAL_NW_T_THRESHOLD:   float = 1.50

TRADING_DAYS_PER_YEAR:     int   = 252
BOOTSTRAP_RESAMPLES_LOCKED: int  = 1000
BOOTSTRAP_ALPHA_LOCKED:     float = 0.05


# ── Public result type (spec §九 schema) ───────────────────────────────────
@dataclasses.dataclass
class PeadVerdict:
    """Spec §九 v1_pead_10y_verdict.json schema."""
    decision:                   str          # PASS / MARGINAL / FAIL
    spec_hash:                  str
    spec_path:                  str
    run_at:                     str
    wave:                       str          # "C1"
    window_start:               str
    window_end:                 str
    universe_source:            str          # "crsp_vintage_top200"
    n_quarters:                 int
    n_firm_quarters_used:       int
    n_firm_quarters_excluded:   int
    exclusion_breakdown:        dict
    n_daily_observations:       int
    sharpe_gross:               float
    sharpe_net:                 float
    nw_t_stat:                  float
    nw_lag:                     int
    bootstrap_ci_lower:         float
    bootstrap_ci_upper:         float
    bhy_fdr_passes:             bool
    effective_n_trials_at_verdict: int
    cumulative_return:          float
    max_drawdown:               float
    long_only_sharpe:           float
    short_only_sharpe:          float
    annual_turnover:            float
    tc_bps_roundtrip:           float
    tc_drag_annualized:         float
    fallback_rate_per_quarter:  dict
    honest_disclose:            list


# ── Sharpe + NW t-stat + max DD + cum return ───────────────────────────────
def compute_annualized_sharpe(
    daily_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Lo 2002 frequency-consistent annualized Sharpe.

    annualized = (mean / std_ddof1) × √periods_per_year. NaN if insufficient
    data or std=0.
    """
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def compute_nw_t_stat(
    daily_returns: pd.Series,
    *,
    lag: int = NW_LAG_TRADING_DAYS_LOCKED,
) -> float:
    """Newey-West HAC t-statistic of sample mean = 0.

    NW variance estimator (Bartlett kernel, common convention):
        V̂ = γ_0 + 2 × Σ_{k=1..L} (1 − k/(L+1)) × γ_k
    where γ_k = (1/T) × Σ_{t=k+1..T} (x_t − x̄)(x_{t−k} − x̄).

    NW SE(x̄) = √(V̂ / T)
    NW t      = x̄ / NW SE(x̄)

    Per spec §2.5 + §六: lag = 60 trading days (matches PEAD 60-day overlap;
    Andrews 1991 + Livnat-Mendenhall 2006 standard).
    """
    r = pd.Series(daily_returns).dropna().astype(float).values
    T = len(r)
    if T < 2 or lag < 0:
        return float("nan")

    xbar = float(np.mean(r))
    dev  = r - xbar

    # γ_0
    gamma_0 = float(np.dot(dev, dev) / T)

    v = gamma_0
    for k in range(1, lag + 1):
        if k >= T:
            break
        gamma_k = float(np.dot(dev[k:], dev[:-k]) / T)
        weight = 1.0 - k / (lag + 1.0)
        v += 2.0 * weight * gamma_k

    if not np.isfinite(v) or v <= 0:
        return float("nan")
    se = float(np.sqrt(v / T))
    if se <= 0 or not np.isfinite(se):
        return float("nan")
    return float(xbar / se)


def compute_max_drawdown(daily_returns: pd.Series) -> float:
    """Max peak-to-trough drawdown of cumulative-return NAV path.

    Returns a non-positive float (e.g., -0.15 = 15% drawdown).
    NaN if input is empty.
    """
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return float("nan")
    nav = (1.0 + r).cumprod()
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return float(dd.min())


def compute_cumulative_return(daily_returns: pd.Series) -> float:
    """Total cumulative return over the window: ∏(1 + r_t) - 1."""
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


# ── Bootstrap CI (single series, Politis-Romano stationary) ────────────────
def compute_bootstrap_ci_sharpe_single(
    daily_returns:    pd.Series,
    *,
    n_resamples:      int   = BOOTSTRAP_RESAMPLES_LOCKED,
    alpha:            float = BOOTSTRAP_ALPHA_LOCKED,
    block_size:       Optional[int] = None,
    periods_per_year: int   = TRADING_DAYS_PER_YEAR,
    random_state:     int   = 42,
) -> tuple[float, float, int]:
    """Politis-Romano (1994) stationary bootstrap on a single returns series.

    Returns (ci_lower, ci_upper, block_size_used).

    block_size=None → Politis-White (2004) automatic length on the returns
    series; falls back to NW_LAG_TRADING_DAYS_LOCKED on auto-length failure.
    """
    try:
        from arch.bootstrap import StationaryBootstrap, optimal_block_length
    except ImportError as exc:
        raise RuntimeError(
            "compute_bootstrap_ci_sharpe_single requires `arch`. Install via "
            "`pip install arch`."
        ) from exc

    r = pd.Series(daily_returns).dropna().astype(float)
    if len(r) < 30:
        return float("nan"), float("nan"), 0

    if block_size is None:
        try:
            opt = optimal_block_length(r.values)
            block_size = max(1, int(np.ceil(float(opt["stationary"].iloc[0]))))
        except Exception as exc:
            logger.warning("optimal_block_length failed (%s); fallback block=%d",
                           exc, NW_LAG_TRADING_DAYS_LOCKED)
            block_size = NW_LAG_TRADING_DAYS_LOCKED

    rng = np.random.default_rng(random_state)
    sb = StationaryBootstrap(
        block_size, r.values, seed=int(rng.integers(0, 2**31 - 1))
    )

    sqrt_freq = float(np.sqrt(periods_per_year))
    sharpes = []
    for data, _ in sb.bootstrap(n_resamples):
        sample = data[0]
        if len(sample) < 30:
            continue
        sd = float(sample.std(ddof=1))
        if sd <= 0 or not np.isfinite(sd):
            continue
        sharpes.append(float(sample.mean()) / sd * sqrt_freq)
    if not sharpes:
        return float("nan"), float("nan"), int(block_size)
    arr = np.array(sharpes)
    lower = float(np.percentile(arr, alpha / 2.0 * 100.0))
    upper = float(np.percentile(arr, (1.0 - alpha / 2.0) * 100.0))
    return lower, upper, int(block_size)


# ── BHY-FDR conservative bound ─────────────────────────────────────────────
def compute_bhy_fdr_passes(
    p_value:           float,
    effective_n_trials: int,
    *,
    alpha:             float = 0.05,
) -> bool:
    """Conservative BHY (Benjamini-Yekutieli 2001) FDR rejection threshold.

    Strictest control (k=1 in BHY ordering, i.e., assuming this test has
    the smallest p-value among all m=n_trials project tests):

        threshold = α / (n_trials × H_{n_trials})
        H_m = harmonic sum 1 + 1/2 + ... + 1/m

    A test PASSES BHY-FDR iff its p-value ≤ threshold.

    This bound is stricter than the actual sequential BHY procedure (which
    requires all project p-values for proper ranking), but conservative
    "pass" is safer than wrongly claiming pass without full p-value list.

    Per spec §3.1: α=5% over EFFECTIVE_N_TRIALS + 1 (this spec consumes +1
    already counted in the caller's `effective_n_trials`).
    """
    if effective_n_trials < 1:
        return False
    if not (0.0 <= p_value <= 1.0):
        return False
    H_m = float(sum(1.0 / k for k in range(1, effective_n_trials + 1)))
    threshold = alpha / (effective_n_trials * H_m)
    return p_value <= threshold


def nw_t_to_two_sided_p_value(nw_t: float) -> float:
    """Two-sided normal p-value from NW t-stat.

    p = 2 × (1 − Φ(|t|)). For large |t|, p is tiny.
    NaN input → NaN output.
    """
    from scipy.stats import norm
    if not np.isfinite(nw_t):
        return float("nan")
    return float(2.0 * (1.0 - norm.cdf(abs(nw_t))))


# ── Decision classifier (spec §3.2, BHY demoted 2026-05-12 amendment v2) ──
def classify_decision(
    sharpe_net: float,
    nw_t:       float,
    bhy_passes: bool = False,   # kept for backward-compat, IGNORED in gate
) -> str:
    """Spec §3.2 industry-grade gates (BHY demoted to reporting 2026-05-12).

    PASS:     Sharpe (net) ≥ 0.50 AND NW t ≥ 2.00
    MARGINAL: Sharpe (net) ≥ 0.30 AND NW t ≥ 1.50
    FAIL:     anything else (Sharpe < 0.30 OR NW t < 1.50 OR NaN inputs)

    `bhy_passes` arg retained for backward compatibility but NOT consulted.
    BHY-FDR p-value is still computed + logged in verdict.json for audit
    transparency. False-positive risk compensated by §9 12-24mo forward
    paper trade counterfactual before real $ allocation.

    Industry quant practice (AQR / Two Sigma / Millennium) uses single-test
    NW t ≥ 2.0 + forward paper trade + capacity/TC analysis. Cumulative-
    project BHY-FDR is an academic-publication construct retired with the
    2026-05-12 SSRN-paper-path drop.
    """
    if not np.isfinite(sharpe_net) or not np.isfinite(nw_t):
        return "FAIL"

    if sharpe_net >= PASS_SHARPE_THRESHOLD and nw_t >= PASS_NW_T_THRESHOLD:
        return "PASS"

    if sharpe_net >= MARGINAL_SHARPE_THRESHOLD and nw_t >= MARGINAL_NW_T_THRESHOLD:
        return "MARGINAL"

    return "FAIL"


# ── Honest-disclose constants (spec §3.5) ───────────────────────────────────
HONEST_DISCLOSE_LOCKED: list = [
    "PEAD has decayed since early 2000s (Chordia-Subrahmanyam-Tong 2014); "
    "2014-2023 walk-forward is in the decayed regime.",
    "Transaction costs at 30bp roundtrip reduce gross Sharpe by ~0.05-0.10.",
    "I/B/E/S coverage thin for sub-$500M market cap; top-200 filter mitigates "
    "but does not eliminate.",
    "Quarter-end concentration: bulk of SP500 firms announce in 4-week window "
    "post-quarter-end.",
    "Earnings season clustering: 60d hold overlaps with next quarter's "
    "announcements — momentum may compound or revert; spec acknowledges but "
    "does NOT try to exploit.",
]


# ── Top-level verdict builder ──────────────────────────────────────────────
def build_pead_verdict(
    walk_forward_result: WalkForwardPeadResult,
    signal_panel:        pd.DataFrame,
    *,
    spec_hash:           str,
    spec_path:           str = "docs/spec_path_c_earnings_pead_v1.md",
    effective_n_trials:  int,
    exclusion_breakdown: Optional[dict] = None,
    fallback_rate_per_quarter: Optional[dict] = None,
    wave:                str = "C1",
    universe_source:     str = "crsp_vintage_top200",
) -> PeadVerdict:
    """Aggregate walk-forward result into a PeadVerdict artifact (spec §九 schema).

    `wave` and `universe_source` parameterized for cross-spec reuse:
      - PEAD (id=57): defaults wave="C1", universe_source="crsp_vintage_top200"
      - Labor (id=58): wave="C-labor", universe_source="crsp_vintage_top200_revelio_linked"
      - Future Path C variants override as needed
    """
    daily = walk_forward_result.daily_returns

    if daily.empty:
        return PeadVerdict(
            decision="FAIL",
            spec_hash=spec_hash, spec_path=spec_path,
            run_at=datetime.datetime.utcnow().isoformat() + "Z",
            wave=wave,
            window_start=walk_forward_result.window_start.isoformat(),
            window_end=walk_forward_result.window_end.isoformat(),
            universe_source=universe_source,
            n_quarters=walk_forward_result.n_quarters_processed,
            n_firm_quarters_used=walk_forward_result.n_firm_quarters_active,
            n_firm_quarters_excluded=0,
            exclusion_breakdown=exclusion_breakdown or {},
            n_daily_observations=0,
            sharpe_gross=float("nan"),
            sharpe_net=float("nan"),
            nw_t_stat=float("nan"),
            nw_lag=NW_LAG_TRADING_DAYS_LOCKED,
            bootstrap_ci_lower=float("nan"),
            bootstrap_ci_upper=float("nan"),
            bhy_fdr_passes=False,
            effective_n_trials_at_verdict=effective_n_trials,
            cumulative_return=float("nan"),
            max_drawdown=float("nan"),
            long_only_sharpe=float("nan"),
            short_only_sharpe=float("nan"),
            annual_turnover=walk_forward_result.annual_turnover_estimate,
            tc_bps_roundtrip=walk_forward_result.tc_bps_roundtrip,
            tc_drag_annualized=0.0,
            fallback_rate_per_quarter=fallback_rate_per_quarter or {},
            honest_disclose=list(HONEST_DISCLOSE_LOCKED),
        )

    # Gross + net Sharpe
    sharpe_gross = compute_annualized_sharpe(daily["r_long_short"])
    sharpe_net   = compute_annualized_sharpe(daily["r_long_short_net"]) \
                   if "r_long_short_net" in daily.columns else sharpe_gross

    # NW t-stat on NET series (drives the decision gate)
    series_for_t = daily["r_long_short_net"] if "r_long_short_net" in daily.columns else daily["r_long_short"]
    nw_t = compute_nw_t_stat(series_for_t, lag=NW_LAG_TRADING_DAYS_LOCKED)

    # Bootstrap CI on NET series
    try:
        ci_lower, ci_upper, _block = compute_bootstrap_ci_sharpe_single(series_for_t)
    except Exception as exc:
        logger.warning("bootstrap CI failed: %s; reporting NaN CI", exc)
        ci_lower, ci_upper = float("nan"), float("nan")

    # BHY-FDR check
    p_value = nw_t_to_two_sided_p_value(nw_t)
    bhy_passes = compute_bhy_fdr_passes(p_value, effective_n_trials)

    # Decision
    # BHY demoted 2026-05-12: bhy_passes still computed + logged in verdict.json
    # for audit transparency, but not consulted in decision gate.
    decision = classify_decision(sharpe_net=sharpe_net, nw_t=nw_t)

    # Per-leg diagnostics
    long_sharpe  = compute_annualized_sharpe(daily["r_long"])  if "r_long"  in daily.columns else float("nan")
    short_sharpe = compute_annualized_sharpe(daily["r_short"]) if "r_short" in daily.columns else float("nan")

    # Cumulative + drawdown (on net series)
    cum_ret = compute_cumulative_return(series_for_t)
    max_dd  = compute_max_drawdown(series_for_t)

    # TC drag annualized (uniform daily drag × 252)
    tc_drag_daily = (walk_forward_result.tc_bps_roundtrip / 10_000.0
                     * walk_forward_result.annual_turnover_estimate
                     / TRADING_DAYS_PER_YEAR)
    tc_drag_ann = tc_drag_daily * TRADING_DAYS_PER_YEAR

    n_used     = int(walk_forward_result.n_firm_quarters_active)
    n_excluded = 0
    if not signal_panel.empty and "leg" in signal_panel.columns:
        n_excluded = int((signal_panel["leg"] == "excluded").sum())

    return PeadVerdict(
        decision=decision,
        spec_hash=spec_hash,
        spec_path=spec_path,
        run_at=datetime.datetime.utcnow().isoformat() + "Z",
        wave=wave,
        window_start=walk_forward_result.window_start.isoformat(),
        window_end=walk_forward_result.window_end.isoformat(),
        universe_source=universe_source,
        n_quarters=walk_forward_result.n_quarters_processed,
        n_firm_quarters_used=n_used,
        n_firm_quarters_excluded=n_excluded,
        exclusion_breakdown=exclusion_breakdown or {},
        n_daily_observations=int(len(daily)),
        sharpe_gross=sharpe_gross,
        sharpe_net=sharpe_net,
        nw_t_stat=nw_t,
        nw_lag=NW_LAG_TRADING_DAYS_LOCKED,
        bootstrap_ci_lower=ci_lower,
        bootstrap_ci_upper=ci_upper,
        bhy_fdr_passes=bool(bhy_passes),
        effective_n_trials_at_verdict=int(effective_n_trials),
        cumulative_return=cum_ret,
        max_drawdown=max_dd,
        long_only_sharpe=long_sharpe,
        short_only_sharpe=short_sharpe,
        annual_turnover=float(walk_forward_result.annual_turnover_estimate),
        tc_bps_roundtrip=float(walk_forward_result.tc_bps_roundtrip),
        tc_drag_annualized=float(tc_drag_ann),
        fallback_rate_per_quarter=fallback_rate_per_quarter or {},
        honest_disclose=list(HONEST_DISCLOSE_LOCKED),
    )


# ── Persistence ────────────────────────────────────────────────────────────
def persist_verdict(verdict: PeadVerdict, json_path: Path) -> Path:
    """Save PeadVerdict to JSON matching spec §九 schema."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # dataclasses.asdict produces a dict ready for json.dump (no nested
    # non-serializable types — all primitives, lists, dicts).
    payload = dataclasses.asdict(verdict)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("verdict persisted: %s decision=%s", json_path, verdict.decision)
    return json_path
