"""engine.research.enhance.paired_bootstrap — Politis-Romano 1994
circular block bootstrap for paired Sharpe-ratio differences.

Why paired bootstrap (not naive resample)
=========================================
Two return series X (deployed sleeve) and X' (variant) usually have
high contemporaneous correlation (ρ ≈ 0.85-0.95 — they trade the same
universe with overlapping signals). The PAIRED difference series
d_t = x'_t - x_t has much lower variance than either series alone, so
its bootstrap distribution gives MUCH tighter confidence intervals
than naively resampling X and X' independently.

Why BLOCK bootstrap (not iid bootstrap)
=======================================
Monthly returns have autocorrelation (especially in trend-following /
mean-reverting families). Politis-Romano 1994 showed circular block
bootstrap preserves autocorrelation structure within blocks while
allowing inter-block independence — the right asymptotic regime for
serially-correlated series of moderate length (N=120-360 months).

Block-size rule (Politis-White 2004 / our adaptation)
=====================================================
We use the standard rule of thumb:
    block_size = ceil( N^(1/3) * (autocorr_strength)^(1/3) )
For solo-quant scale (N=120-360, monthly), this gives block sizes 4-9
which captures momentum / reversal autocorrelation up to ~6-9 months
without dropping below 4-month minimum for noise robustness. Default:
6 months when not specified; caller can override.

Output
======
PairedBootstrapResult holds the empirical bootstrap distribution of
Sharpe-difference (variant - baseline) with annualized scaling AND
the canonical statistics consumers need:

  - sharpe_diff_observed       : the actual point estimate
  - sharpe_diff_bootstrap_mean : centered around 0 if no improvement
  - sharpe_diff_bootstrap_std  : the SE for t-stat
  - sharpe_diff_t_stat         : observed / std
  - sharpe_diff_p_value        : one-sided (variant > baseline)
  - sharpe_diff_ci_lo/hi       : 95% CI bounds (BCa or percentile)
  - n_iterations               : how many bootstrap draws
  - block_size                 : the block size used
"""
from __future__ import annotations

import dataclasses as _dc
import math
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_N_BOOTSTRAP   = 2000
DEFAULT_BLOCK_SIZE    = 6        # months — captures ~half-year momentum/reversal
ANNUALIZATION_FACTOR  = 12.0     # monthly → annual
MIN_OBS_FOR_BOOTSTRAP = 24


@_dc.dataclass(frozen=True)
class PairedBootstrapResult:
    """Output of the paired block bootstrap on Sharpe-difference."""
    sharpe_diff_observed:       float
    sharpe_diff_bootstrap_mean: float
    sharpe_diff_bootstrap_std:  float
    sharpe_diff_t_stat:         float    # observed / bootstrap SE
    sharpe_diff_p_value:        float    # one-sided variant > baseline
    sharpe_diff_ci_lo:          float    # 2.5%ile (annualized)
    sharpe_diff_ci_hi:          float    # 97.5%ile (annualized)
    n_iterations:               int
    n_obs:                      int
    block_size:                 int
    correlation:                float

    def to_dict(self) -> dict:
        return _dc.asdict(self)


# ── Internal helpers ───────────────────────────────────────────────


def _annualized_sharpe(monthly_returns: np.ndarray) -> float:
    """Plain annualized Sharpe; nan if std degenerate."""
    if len(monthly_returns) < 2:
        return float("nan")
    std = float(monthly_returns.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        return float("nan")
    return float(monthly_returns.mean() / std * math.sqrt(ANNUALIZATION_FACTOR))


def _circular_block_sample(
    n_obs: int, block_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Generate one circular-block-bootstrap index sequence of length n_obs.

    Politis-Romano 1994: pick uniformly random start indices in [0, n_obs),
    take block_size consecutive observations (wrapping around modulo n_obs),
    concatenate until length n_obs.
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    starts = rng.integers(0, n_obs, size=n_blocks)
    indices = np.empty(n_blocks * block_size, dtype=np.int64)
    for i, s in enumerate(starts):
        indices[i * block_size:(i + 1) * block_size] = (
            (np.arange(block_size) + s) % n_obs
        )
    return indices[:n_obs]


# ── Public API ─────────────────────────────────────────────────────


def paired_block_bootstrap_sharpe_diff(
    baseline_returns: pd.Series,
    variant_returns:  pd.Series,
    *,
    n_iterations: int = DEFAULT_N_BOOTSTRAP,
    block_size:   int = DEFAULT_BLOCK_SIZE,
    seed:         int = 42,
) -> Optional[PairedBootstrapResult]:
    """Compute the paired bootstrap distribution of Sharpe difference
    (variant - baseline), annualized.

    Inputs MUST be:
      - same length OR overlap-trimmed (we align on intersection of indices)
      - monthly frequency (annualization uses √12)
      - actual returns (decimals, not bps or pct)

    Returns None when:
      - intersection < MIN_OBS_FOR_BOOTSTRAP
      - either series has degenerate variance
    """
    # Align on shared index
    df = pd.concat({"base": baseline_returns, "var": variant_returns}, axis=1).dropna()
    n_obs = len(df)
    if n_obs < MIN_OBS_FOR_BOOTSTRAP:
        return None

    base = df["base"].to_numpy()
    var  = df["var"].to_numpy()

    if base.std(ddof=1) <= 0 or var.std(ddof=1) <= 0:
        return None

    correlation = float(np.corrcoef(base, var)[0, 1])
    sharpe_diff_observed = _annualized_sharpe(var) - _annualized_sharpe(base)
    if not math.isfinite(sharpe_diff_observed):
        return None

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_iterations, dtype=np.float64)
    for i in range(n_iterations):
        idx = _circular_block_sample(n_obs, block_size, rng)
        bs_base = base[idx]
        bs_var  = var[idx]
        sh_b = _annualized_sharpe(bs_base)
        sh_v = _annualized_sharpe(bs_var)
        boot_diffs[i] = sh_v - sh_b

    # Filter nans (degenerate bootstrap samples)
    finite_mask = np.isfinite(boot_diffs)
    if finite_mask.sum() < n_iterations // 2:
        return None
    finite = boot_diffs[finite_mask]
    bs_mean = float(finite.mean())
    bs_std  = float(finite.std(ddof=1))
    if bs_std <= 0 or not math.isfinite(bs_std):
        return None

    # t-stat under the null that variant doesn't improve (sharpe_diff = 0)
    t_stat = float(sharpe_diff_observed / bs_std)
    # One-sided p: fraction of bootstrap draws ≤ 0 under shift-to-zero
    # (equivalent to fraction below observed under sharpe_diff_observed shift).
    # Standard percentile-based one-sided p for H1: variant > baseline:
    centered = finite - bs_mean
    n_below = int((centered + sharpe_diff_observed <= 0).sum())
    p_value = float(n_below / len(finite))

    ci_lo = float(np.percentile(finite, 2.5))
    ci_hi = float(np.percentile(finite, 97.5))

    return PairedBootstrapResult(
        sharpe_diff_observed       = sharpe_diff_observed,
        sharpe_diff_bootstrap_mean = bs_mean,
        sharpe_diff_bootstrap_std  = bs_std,
        sharpe_diff_t_stat         = t_stat,
        sharpe_diff_p_value        = p_value,
        sharpe_diff_ci_lo          = ci_lo,
        sharpe_diff_ci_hi          = ci_hi,
        n_iterations               = int(finite_mask.sum()),
        n_obs                      = n_obs,
        block_size                 = block_size,
        correlation                = correlation,
    )


def paired_block_bootstrap_summary(result: PairedBootstrapResult) -> str:
    """Human-readable one-liner for digest / verdict event metrics."""
    return (
        f"ΔSharpe={result.sharpe_diff_observed:+.3f} "
        f"(t={result.sharpe_diff_t_stat:.2f}, p={result.sharpe_diff_p_value:.3f}), "
        f"95% CI [{result.sharpe_diff_ci_lo:+.3f}, {result.sharpe_diff_ci_hi:+.3f}], "
        f"corr(base,var)={result.correlation:.3f}, n={result.n_obs}mo, "
        f"block={result.block_size}, B={result.n_iterations}"
    )
