"""Phase 2 enhance verdict framework tests.

Covers:
  - paired block bootstrap arithmetic + edge cases
  - verdict classifier with sign-correct thresholds
  - dispatcher end-to-end with synthetic baseline + variant
  - refusal paths (sleeve not resolved, low correlation, insufficient overlap)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.research.enhance.paired_bootstrap import (
    DEFAULT_BLOCK_SIZE,
    MIN_OBS_FOR_BOOTSTRAP,
    PairedBootstrapResult,
    paired_block_bootstrap_sharpe_diff,
    paired_block_bootstrap_summary,
)
from engine.research.enhance.verdict import (
    EnhanceVerdict,
    GREEN_THRESHOLD_CORRELATION_FLOOR,
    GREEN_THRESHOLD_SHARPE_DIFF,
    GREEN_THRESHOLD_T_STAT,
    RED_THRESHOLD_SHARPE_DIFF,
    classify_enhance_verdict,
)
from engine.research.enhance.dispatcher import (
    dispatch_enhance_hypothesis,
)


# ── Helpers ────────────────────────────────────────────────────────


def _idx(n: int, start: str = "2010-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _gen_correlated_pair(
    n: int,
    *,
    mean_base: float,
    mean_var:  float,
    std:       float,
    rho:       float,
    seed:      int = 42,
) -> tuple[pd.Series, pd.Series]:
    """Two monthly return series with correlation rho."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, std, size=(n, 2))
    # Apply correlation via Cholesky factor of [[1, rho], [rho, 1]]
    L = np.linalg.cholesky([[1.0, rho], [rho, 1.0]])
    corr_noise = noise @ L.T
    base = mean_base + corr_noise[:, 0]
    var  = mean_var  + corr_noise[:, 1]
    idx = _idx(n)
    return pd.Series(base, index=idx), pd.Series(var, index=idx)


# ── paired_block_bootstrap arithmetic ──────────────────────────────


def test_bootstrap_returns_none_below_min_obs():
    s = pd.Series([0.01] * 12, index=_idx(12))
    out = paired_block_bootstrap_sharpe_diff(s, s)
    assert out is None


def test_bootstrap_returns_none_for_degenerate_variance():
    s_const = pd.Series([0.01] * 48, index=_idx(48))
    s_real  = pd.Series(np.random.default_rng(0).normal(0, 0.04, 48), index=_idx(48))
    assert paired_block_bootstrap_sharpe_diff(s_const, s_real) is None
    assert paired_block_bootstrap_sharpe_diff(s_real, s_const) is None


def test_bootstrap_basic_shape():
    base, var = _gen_correlated_pair(n=120, mean_base=0.006, mean_var=0.008,
                                       std=0.04, rho=0.90, seed=7)
    out = paired_block_bootstrap_sharpe_diff(base, var, n_iterations=500)
    assert out is not None
    assert isinstance(out, PairedBootstrapResult)
    assert out.n_obs == 120
    assert out.block_size == DEFAULT_BLOCK_SIZE
    assert out.n_iterations <= 500
    assert math.isfinite(out.sharpe_diff_observed)
    assert math.isfinite(out.sharpe_diff_bootstrap_std)
    assert 0.5 < out.correlation < 1.0   # we asked for 0.90


def test_bootstrap_summary_string_contains_fields():
    base, var = _gen_correlated_pair(n=120, mean_base=0.006, mean_var=0.008,
                                       std=0.04, rho=0.90, seed=11)
    out = paired_block_bootstrap_sharpe_diff(base, var, n_iterations=300)
    text = paired_block_bootstrap_summary(out)
    assert "ΔSharpe" in text
    assert "t=" in text
    assert "p=" in text
    assert "CI" in text


# ── verdict classifier ────────────────────────────────────────────


def _synth_result(*, sharpe_diff=0.0, t_stat=0.0, p_value=0.5,
                   correlation=0.90) -> PairedBootstrapResult:
    return PairedBootstrapResult(
        sharpe_diff_observed       = sharpe_diff,
        sharpe_diff_bootstrap_mean = 0.0,
        sharpe_diff_bootstrap_std  = 0.1,
        sharpe_diff_t_stat         = t_stat,
        sharpe_diff_p_value        = p_value,
        sharpe_diff_ci_lo          = sharpe_diff - 0.2,
        sharpe_diff_ci_hi          = sharpe_diff + 0.2,
        n_iterations               = 2000,
        n_obs                      = 120,
        block_size                 = 6,
        correlation                = correlation,
    )


def test_verdict_improvement_when_all_conditions_met():
    r = _synth_result(sharpe_diff=+0.20, t_stat=+2.5, p_value=0.01)
    assert classify_enhance_verdict(r) == EnhanceVerdict.IMPROVEMENT


def test_verdict_noise_when_sharpe_below_threshold():
    r = _synth_result(sharpe_diff=+0.10, t_stat=+1.0, p_value=0.20)
    assert classify_enhance_verdict(r) == EnhanceVerdict.NOISE


def test_verdict_noise_when_t_stat_below_threshold():
    r = _synth_result(sharpe_diff=+0.20, t_stat=+1.5, p_value=0.07)
    assert classify_enhance_verdict(r) == EnhanceVerdict.NOISE


def test_verdict_degradation_when_mirror_of_improvement():
    r = _synth_result(sharpe_diff=-0.20, t_stat=-2.5, p_value=0.99)
    assert classify_enhance_verdict(r) == EnhanceVerdict.DEGRADATION


def test_verdict_low_correlation_routes_to_noise_not_improvement():
    # Even if sharpe_diff + t_stat would pass, low corr = new strategy
    r = _synth_result(sharpe_diff=+0.30, t_stat=+3.0, p_value=0.001,
                       correlation=0.20)
    assert classify_enhance_verdict(r) == EnhanceVerdict.NOISE


# ── dispatcher end-to-end ─────────────────────────────────────────


def test_dispatcher_refuses_when_sleeve_not_resolved(tmp_path):
    variant = pd.Series([0.01] * 48, index=_idx(48))
    log_path = tmp_path / "enhance_verdicts.jsonl"
    out = dispatch_enhance_hypothesis(
        hypothesis_id="h-test",
        sleeve_id="nonexistent_sleeve_xyz",
        variant_returns=variant,
        log_path=log_path,
    )
    assert out.verdict == "REFUSED"
    assert out.refusal_reason == "SLEEVE_NOT_RESOLVED"


def test_dispatcher_succeeds_with_baseline_override(tmp_path):
    base, var = _gen_correlated_pair(n=120, mean_base=0.006, mean_var=0.008,
                                       std=0.04, rho=0.90, seed=3)
    log_path = tmp_path / "enhance_verdicts.jsonl"
    out = dispatch_enhance_hypothesis(
        hypothesis_id="h-real-1",
        sleeve_id="test_sleeve",
        variant_returns=var,
        baseline_returns=base,
        n_iterations=200,
        cron_run_id="cr-t-1",
        cron_source="manual",
        log_path=log_path,
    )
    assert out.verdict in {"IMPROVEMENT", "NOISE", "DEGRADATION"}
    assert out.refusal_reason is None
    assert out.bootstrap_result is not None
    assert "ΔSharpe" in out.summary
    # Log row was written
    rows = log_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1


def test_dispatcher_refuses_low_correlation_route_to_forward(tmp_path):
    rng = np.random.default_rng(99)
    base = pd.Series(rng.normal(0.006, 0.04, 120), index=_idx(120))
    rng2 = np.random.default_rng(1234)
    var = pd.Series(rng2.normal(0.008, 0.04, 120), index=_idx(120))
    # base and var are independent → corr ≈ 0
    log_path = tmp_path / "enhance_verdicts.jsonl"
    out = dispatch_enhance_hypothesis(
        hypothesis_id="h-newfactor",
        sleeve_id="anything",
        variant_returns=var,
        baseline_returns=base,
        n_iterations=200,
        log_path=log_path,
    )
    assert out.verdict == "REFUSED"
    assert out.refusal_reason == "LOW_CORRELATION_NEW_FACTOR_ROUTE"
    assert "forward" in (out.refusal_detail or "").lower()


def test_dispatcher_refuses_insufficient_overlap(tmp_path):
    base = pd.Series([0.01] * 12, index=_idx(12))
    var  = pd.Series([0.012] * 12, index=_idx(12))
    log_path = tmp_path / "enhance_verdicts.jsonl"
    out = dispatch_enhance_hypothesis(
        hypothesis_id="h-short",
        sleeve_id="anything",
        variant_returns=var,
        baseline_returns=base,
        log_path=log_path,
    )
    assert out.verdict == "REFUSED"
    assert "INSUFFICIENT" in out.refusal_reason


def test_dispatcher_detects_real_improvement(tmp_path):
    """High-correlation pair where variant has materially higher mean
    should classify IMPROVEMENT."""
    rng = np.random.default_rng(2026)
    n = 240
    common = rng.normal(0.0, 0.04, n)
    base = pd.Series(0.004 + common, index=_idx(n))
    var  = pd.Series(0.011 + common, index=_idx(n))   # +0.7% higher mean per month
    log_path = tmp_path / "enhance_verdicts.jsonl"
    out = dispatch_enhance_hypothesis(
        hypothesis_id="h-improve",
        sleeve_id="anything",
        variant_returns=var,
        baseline_returns=base,
        n_iterations=500,
        log_path=log_path,
    )
    assert out.verdict == "IMPROVEMENT"
    br = out.bootstrap_result
    assert br["sharpe_diff_observed"] > 0.15
    assert br["sharpe_diff_t_stat"] > 1.96
    assert br["correlation"] > 0.95   # by construction (common noise dominates)
