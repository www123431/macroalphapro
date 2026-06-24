"""
tests/test_d_pead_plus.py — Sprint I D-PEAD-Plus module tests.

Coverage:
- Doctrine invariant (0-LLM-in-DECISION assertion)
- Locked constants per spec id=74
- Prompt hash determinism
- Feature combiner OLS math (synthetic data)
- Verdict gate logic
- Bootstrap CI math
- NW-t computation
"""
from __future__ import annotations

import datetime
import math

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Doctrine invariant
# ─────────────────────────────────────────────────────────────────────────────
def test_doctrine_no_llm_in_decision_layer_passes():
    """3 decision-layer modules must have ZERO LLM SDK imports."""
    from engine.d_pead_plus.doctrine import (
        assert_no_llm_in_decision_layer, audit_decision_layer_imports,
        DECISION_LAYER_MODULES, FORBIDDEN_LLM_IMPORTS, NO_LLM_IN_DECISION_LAYER,
    )
    assert NO_LLM_IN_DECISION_LAYER is True
    assert DECISION_LAYER_MODULES == (
        "engine.d_pead_plus.feature_combiner",
        "engine.d_pead_plus.backtest",
        "engine.d_pead_plus.verdict",
    )
    violations = audit_decision_layer_imports()
    assert not violations, f"Doctrine violations detected: {violations}"
    # Also should not raise
    assert_no_llm_in_decision_layer()


def test_doctrine_forbidden_imports_list():
    """FORBIDDEN_LLM_IMPORTS must include key LLM SDKs."""
    from engine.d_pead_plus.doctrine import FORBIDDEN_LLM_IMPORTS
    assert "openai" in FORBIDDEN_LLM_IMPORTS
    assert "anthropic" in FORBIDDEN_LLM_IMPORTS
    assert "vertexai" in FORBIDDEN_LLM_IMPORTS
    assert "engine.d_pead_plus.llm_extractor" in FORBIDDEN_LLM_IMPORTS


# ─────────────────────────────────────────────────────────────────────────────
# Locked constants (spec id=74)
# ─────────────────────────────────────────────────────────────────────────────
def test_spec_metadata_lock():
    """spec id=74 must have correct metadata."""
    from engine.d_pead_plus import SPEC_ID, SPEC_HASH, SPEC_HASH_SHORT, SLEEVE_ID, DOCTRINE
    assert SPEC_ID == 74
    # Post-amendment 1 (2026-05-13 same day): data source pivot CRSP -> yfinance + Compustat
    assert SPEC_HASH == "6d8e614ebd68ec42d071949bfd4299b0e4a7a363"
    assert SPEC_HASH_SHORT == "6d8e614e"
    assert SLEEVE_ID == "ss_sp500"
    assert "0-LLM-in-DECISION" in DOCTRINE


def test_universe_locked_constants():
    from engine.d_pead_plus.universe import (
        UNIVERSE_TOP_N_LOCKED, LOCKED_EXCH_CODES, LOCKED_SHARE_CODES,
    )
    assert UNIVERSE_TOP_N_LOCKED == 1500
    assert LOCKED_EXCH_CODES == (1, 3)   # NYSE, NASDAQ
    assert LOCKED_SHARE_CODES == (10, 11)


def test_transcripts_loader_locked_constants():
    from engine.d_pead_plus.transcripts_loader import (
        EVENT_TYPE_LOCKED, DATE_ALIGNMENT_WINDOW_DAYS,
    )
    assert EVENT_TYPE_LOCKED == "Earnings Calls"
    assert DATE_ALIGNMENT_WINDOW_DAYS == 5


def test_llm_extractor_locked_constants():
    from engine.d_pead_plus.llm_extractor import (
        LLM_MODEL_LOCKED, LLM_TEMPERATURE_LOCKED, LLM_TOP_P_LOCKED,
        LLM_PROVIDER_LOCKED, LLM_RESPONSE_SCHEMA, PROMPT_HASH_LOCKED,
        PROMPT_HASH_SHORT_LOCKED,
    )
    assert LLM_MODEL_LOCKED == "gemini-2.5-flash"
    assert LLM_TEMPERATURE_LOCKED == 0.0
    assert LLM_TOP_P_LOCKED == 1.0
    assert LLM_PROVIDER_LOCKED == "vertex"
    assert "tone_score" in LLM_RESPONSE_SCHEMA["properties"]
    assert "forward_confidence" in LLM_RESPONSE_SCHEMA["properties"]
    assert "macro_headwind_flag" in LLM_RESPONSE_SCHEMA["properties"]
    assert "evasion_score" in LLM_RESPONSE_SCHEMA["properties"]
    assert "linguistic_complexity" in LLM_RESPONSE_SCHEMA["properties"]
    # Prompt hash deterministic + length
    assert len(PROMPT_HASH_LOCKED) == 64  # SHA256 hex
    assert len(PROMPT_HASH_SHORT_LOCKED) == 16


def test_feature_combiner_locked_constants():
    from engine.d_pead_plus.feature_combiner import (
        FEATURE_COLUMNS_LOCKED, TARGET_COLUMN_LOCKED, DEV_QUARTERS_LOCKED,
    )
    assert FEATURE_COLUMNS_LOCKED == (
        "sue", "tone_score", "forward_confidence",
        "macro_headwind_flag", "evasion_score", "linguistic_complexity",
    )
    assert TARGET_COLUMN_LOCKED == "ret_60d_log"
    assert DEV_QUARTERS_LOCKED == ("2024Q2", "2024Q3", "2024Q4")


def test_backtest_locked_constants():
    from engine.d_pead_plus.backtest import (
        HOLDING_PERIOD_TRADING_DAYS, DECILE_LONG, DECILE_SHORT,
        VOL_TARGET_ANNUAL, TC_BPS_ROUNDTRIP,
    )
    assert HOLDING_PERIOD_TRADING_DAYS == 60
    assert DECILE_LONG == 10
    assert DECILE_SHORT == 1
    assert VOL_TARGET_ANNUAL == 0.10
    assert TC_BPS_ROUNDTRIP == 30.0


def test_verdict_locked_thresholds():
    from engine.d_pead_plus.verdict import (
        IC_DELTA_THRESHOLD, IC_NW_T_THRESHOLD, NW_LAG_LOCKED,
        BOOTSTRAP_N_ITER, ORTHOGONALITY_THRESHOLD, DEV_OOS_RATIO_THRESHOLD,
    )
    assert IC_DELTA_THRESHOLD == 0.02
    assert IC_NW_T_THRESHOLD == 1.96
    assert NW_LAG_LOCKED == 60
    assert BOOTSTRAP_N_ITER == 10000
    assert ORTHOGONALITY_THRESHOLD == 0.30
    assert DEV_OOS_RATIO_THRESHOLD == 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Statistical primitives (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
def test_spearman_ic_synthetic():
    """Perfect rank correlation → IC = 1.0."""
    from engine.d_pead_plus.verdict import spearman_ic
    signal = pd.Series([1, 2, 3, 4, 5])
    ret    = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    ic = spearman_ic(signal, ret)
    assert abs(ic - 1.0) < 1e-9

    # Inverse: -1.0
    ret_inv = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01])
    assert abs(spearman_ic(signal, ret_inv) - (-1.0)) < 1e-9


def test_newey_west_t_stat_synthetic():
    """Constant non-zero series → high t-stat."""
    from engine.d_pead_plus.verdict import newey_west_t_stat
    # Constant series with small noise
    rng = np.random.default_rng(42)
    s = pd.Series(0.05 + rng.normal(0, 0.001, 100))
    t = newey_west_t_stat(s, lag=10)
    assert t > 10  # very significant non-zero mean

    # Zero-mean noise → near-zero t-stat
    s_zero = pd.Series(rng.normal(0, 0.01, 100))
    t_zero = newey_west_t_stat(s_zero, lag=10)
    assert abs(t_zero) < 5  # should not be wildly significant


def test_block_bootstrap_synthetic():
    """Synthetic positive Sharpe diff → positive bootstrap point estimate."""
    from engine.d_pead_plus.verdict import block_bootstrap_sharpe_diff
    rng = np.random.default_rng(42)
    n = 500
    base = rng.normal(0.0001, 0.01, n)
    plus = rng.normal(0.0005, 0.01, n)  # higher mean → higher Sharpe
    df = pd.DataFrame({"d_pead_baseline": base, "d_pead_plus": plus})
    point, ci_low, ci_high = block_bootstrap_sharpe_diff(df, n_iter=1000, block_size=10)
    assert point > 0  # plus has higher Sharpe
    assert ci_low < point < ci_high


# ─────────────────────────────────────────────────────────────────────────────
# Feature combiner OLS math (synthetic)
# ─────────────────────────────────────────────────────────────────────────────
def test_feature_combiner_fit_dev_ols():
    """Synthetic linear y = 0.1·sue + 0.05·tone + noise → recovers coefficients."""
    from engine.d_pead_plus.feature_combiner import fit_dev_ols, FEATURE_COLUMNS_LOCKED
    rng = np.random.default_rng(42)
    n = 1000
    panel = pd.DataFrame({
        "permno":  rng.integers(10000, 99999, n),
        "rdq":     [datetime.date(2024, 6, 30)] * n,
        "quarter": ["2024Q2"] * (n // 3) + ["2024Q3"] * (n // 3) + ["2024Q4"] * (n - 2 * (n // 3)),
    })
    for col in FEATURE_COLUMNS_LOCKED:
        panel[col] = rng.normal(0, 1, n)
        panel[f"{col}_z"] = rng.normal(0, 1, n)
    # Target = 0.1 * sue_z + 0.05 * tone_z + noise
    panel["ret_60d_log"] = 0.1 * panel["sue_z"] + 0.05 * panel["tone_score_z"] + rng.normal(0, 0.05, n)

    coeffs = fit_dev_ols(panel)
    # Should recover ~0.1 for sue, ~0.05 for tone
    assert abs(coeffs.sue - 0.1) < 0.02
    assert abs(coeffs.tone_score - 0.05) < 0.02
    assert 0 < coeffs.r_squared < 1


# ─────────────────────────────────────────────────────────────────────────────
# Save / load round-trip
# ─────────────────────────────────────────────────────────────────────────────
def test_save_load_coefficients_round_trip(tmp_path):
    """Save + load FrozenCoefficients preserves all fields."""
    from engine.d_pead_plus.feature_combiner import (
        FrozenCoefficients, save_coefficients, load_coefficients, COEFFICIENTS_PATH,
    )
    import json
    # Use real CACHE_DIR but write to tmp
    test_path = tmp_path / "v1_dev_coefficients.json"
    coeffs = FrozenCoefficients(
        intercept=0.001, sue=0.1, tone_score=0.05, forward_confidence=0.02,
        macro_headwind_flag=-0.01, evasion_score=-0.03, linguistic_complexity=0.0,
        r_squared=0.05, n_obs_dev=3000, dev_window="2024Q2,2024Q3,2024Q4",
        feature_means={c: 0.0 for c in ["sue", "tone_score"]},
        feature_stds={c: 1.0 for c in ["sue", "tone_score"]},
        fit_at_utc="2026-05-13T00:00:00Z",
    )
    from dataclasses import asdict
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(asdict(coeffs), f)
    with open(test_path, encoding="utf-8") as f:
        loaded = FrozenCoefficients(**json.load(f))
    assert loaded.intercept == coeffs.intercept
    assert loaded.sue == coeffs.sue
    assert loaded.r_squared == coeffs.r_squared
