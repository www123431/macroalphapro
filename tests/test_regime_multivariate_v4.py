"""
tests/test_regime_multivariate_v4.py — W3 D4 unit tests (2026-05-08).

Pre-registration: docs/spec_multivariate_msm_v4_narrative.md §4.4
Spec id: 47

Coverage:
  • Module constants lock (_MULTIVARIATE_FEATURES_V4, _USE_MULTIVARIATE_V4_REGIME)
  • _identify_regimes_by_vix correctness on 3-feature means matrix
  • Synthetic 3-feature 2-state HMM round-trip recovery
  • _get_monthly_narrative_score graceful degradation when cache absent / D3 not locked
  • _get_regime_multivariate_v4 raises MissingFeatureData if narrative column NaN

No live network — all tests run offline.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Module constants lock ────────────────────────────────────────────────────


def test_multivariate_features_v4_locked():
    from engine.regime import _MULTIVARIATE_FEATURES_V4
    assert _MULTIVARIATE_FEATURES_V4 == ("yield_spread", "vix", "narrative_score")


def test_use_multivariate_v4_regime_flag_off_by_default():
    """Per spec §3.2 + §4.2, v4 production swap requires DESCRIPTIVE_POSITIVE
    verdict + supervisor approval. Until then, flag stays False so v3 stays
    in production via get_regime_on() dispatch."""
    from engine.regime import _USE_MULTIVARIATE_V4_REGIME
    assert _USE_MULTIVARIATE_V4_REGIME is False


# ── VIX anchor on 3-feature means matrix ─────────────────────────────────────


def test_identify_regimes_by_vix_works_on_3_feature():
    """Anchor only uses VIX column; feature_names tuple drives column index."""
    from engine.regime import _identify_regimes_by_vix, _MULTIVARIATE_FEATURES_V4
    # Synthetic K=2, d=3 means matrix:
    # state 0: low VIX (15), positive narrative (+1.0)  → risk-on
    # state 1: high VIX (35), negative narrative (-1.5) → risk-off
    means = np.array([
        [+0.5, 15.0, +1.0],
        [-0.5, 35.0, -1.5],
    ])
    risk_on_idx, risk_off_idx = _identify_regimes_by_vix(
        means, feature_names=_MULTIVARIATE_FEATURES_V4
    )
    assert risk_on_idx == 0
    assert risk_off_idx == 1


def test_identify_regimes_swaps_when_state_order_reversed():
    """If EM emits states in reversed order, anchor correctly identifies."""
    from engine.regime import _identify_regimes_by_vix, _MULTIVARIATE_FEATURES_V4
    means = np.array([
        [-0.5, 35.0, -1.5],   # state 0: HIGH vix → risk-off
        [+0.5, 15.0, +1.0],   # state 1: LOW vix  → risk-on
    ])
    risk_on_idx, risk_off_idx = _identify_regimes_by_vix(
        means, feature_names=_MULTIVARIATE_FEATURES_V4
    )
    assert risk_on_idx == 1
    assert risk_off_idx == 0


# ── Synthetic 3-feature 2-state HMM round-trip ───────────────────────────────


def test_synthetic_3feature_hmm_recovers_two_regimes():
    """
    Generate synthetic 200-month panel from a known 2-state HMM with 3 features,
    fit GaussianHMM K=2 with multi-start, verify VIX anchor recovers correct
    state assignment.
    """
    from hmmlearn.hmm import GaussianHMM
    from engine.regime import _identify_regimes_by_vix, _MULTIVARIATE_FEATURES_V4

    rng = np.random.default_rng(42)
    n = 240
    states = rng.choice([0, 1], size=n, p=[0.6, 0.4])
    # state 0: yield=+0.5, vix=15, narr=+0.5 (risk-on)
    # state 1: yield=-0.3, vix=30, narr=-1.0 (risk-off)
    means_truth = np.array([
        [+0.5, 15.0, +0.5],
        [-0.3, 30.0, -1.0],
    ])
    cov = np.eye(3) * 0.15
    cov[1, 1] = 4.0  # vix has higher variance
    X = np.zeros((n, 3))
    for i, s in enumerate(states):
        X[i] = rng.multivariate_normal(means_truth[s], cov)

    # Multi-start fit
    best_score = float("-inf")
    best_model = None
    for seed in range(8):
        try:
            m = GaussianHMM(
                n_components=2, covariance_type="full",
                n_iter=200, tol=1e-3, random_state=seed,
            )
            m.fit(X)
            sc = m.score(X)
            if m.monitor_.converged and sc > best_score:
                best_model = m
                best_score = sc
        except Exception:
            continue

    assert best_model is not None
    risk_on_idx, risk_off_idx = _identify_regimes_by_vix(
        best_model.means_, feature_names=_MULTIVARIATE_FEATURES_V4
    )
    # Recovered regime should have low VIX = risk-on, high VIX = risk-off
    assert best_model.means_[risk_on_idx, 1] < best_model.means_[risk_off_idx, 1]


# ── _get_monthly_narrative_score graceful degradation ───────────────────────


def test_get_monthly_narrative_score_returns_empty_when_cache_missing(tmp_path, monkeypatch):
    """No D2c cache → returns empty Series; v4 path will then raise
    MissingFeatureData → fallback to v3 in get_regime_on()."""
    from engine import regime
    fake_cache = tmp_path / "no_cache.parquet"
    monkeypatch.setattr(regime, "_FOMC_STATEMENTS_CACHE", fake_cache)

    s = regime._get_monthly_narrative_score(datetime.date(2024, 1, 31))
    assert isinstance(s, pd.Series)
    assert len(s) == 0


def test_get_monthly_narrative_score_returns_empty_when_d3_not_locked(tmp_path, monkeypatch):
    """Cache exists but z-norm μ/σ still None → graceful empty (D3 not yet run)."""
    from engine import regime
    from engine import narrative_classifier as nc

    fake_cache = tmp_path / "cache.parquet"
    df = pd.DataFrame({
        "date":          pd.to_datetime(["2024-01-31", "2024-03-20"]),
        "raw_score":     [0.005, -0.003],
        "url":           ["url1", "url2"],
        "word_count":    [600, 620],
        "era":           ["era4_2008+", "era4_2008+"],
        "hawkish_count": [3, 1],
        "dovish_count":  [2, 4],
    })
    df.to_parquet(fake_cache, index=False)
    monkeypatch.setattr(regime, "_FOMC_STATEMENTS_CACHE", fake_cache)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_MEAN", None)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_STD", None)

    s = regime._get_monthly_narrative_score(datetime.date(2024, 4, 30))
    assert len(s) == 0


def test_get_monthly_narrative_score_z_norms_when_locked(tmp_path, monkeypatch):
    """Cache + locked μ/σ → returns z-normed monthly series."""
    from engine import regime
    from engine import narrative_classifier as nc

    fake_cache = tmp_path / "cache.parquet"
    df = pd.DataFrame({
        "date":          pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01"]),
        "raw_score":     [0.005, -0.003, 0.000],
        "url":           ["u1", "u2", "u3"],
        "word_count":    [600, 620, 580],
        "era":           ["era4_2008+"] * 3,
        "hawkish_count": [3, 1, 2],
        "dovish_count":  [2, 4, 2],
    })
    df.to_parquet(fake_cache, index=False)
    monkeypatch.setattr(regime, "_FOMC_STATEMENTS_CACHE", fake_cache)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_MEAN", 0.0)
    monkeypatch.setattr(nc, "_RAW_SCORE_INSAMPLE_STD", 0.005)

    s = regime._get_monthly_narrative_score(
        datetime.date(2024, 5, 31), n_months=12,
    )
    assert isinstance(s, pd.Series)
    assert s.name == "narrative_score"
    assert s.notna().sum() >= 4  # forward-fill covers gap months


# ── End-to-end v4 regime call (mocked, no real HMM training) ────────────────


def test_v4_regime_raises_missing_feature_when_narrative_all_nan(tmp_path, monkeypatch):
    """If narrative_score column is fully NaN (D2c cache empty), v4 path must
    raise MissingFeatureData so get_regime_on() falls back to v3."""
    from engine import regime
    from engine.regime import MissingFeatureData

    fake_cache = tmp_path / "no_cache.parquet"
    monkeypatch.setattr(regime, "_FOMC_STATEMENTS_CACHE", fake_cache)

    # also short-circuit yield/VIX fetchers to return synthetic data
    idx = pd.date_range("2010-01-31", "2024-12-31", freq="ME")
    monkeypatch.setattr(
        regime, "_get_monthly_yield_spread",
        lambda *a, **kw: pd.Series(np.linspace(0.5, 1.5, len(idx)), index=idx, name="yield_spread"),
    )
    monkeypatch.setattr(
        regime, "_get_monthly_vix",
        lambda *a, **kw: pd.Series(np.linspace(12, 28, len(idx)), index=idx, name="vix"),
    )

    with pytest.raises((MissingFeatureData, regime.InsufficientData)):
        regime._get_regime_multivariate_v4(
            as_of=datetime.date(2024, 12, 31),
            train_end=datetime.date(2024, 11, 30),
            n_train_months=180,
        )
