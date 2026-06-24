"""tests/test_pit_audit_dpead.py — the PIT audit must CATCH look-ahead, not rubber-stamp.

Each critical check is tested on a CLEAN synthetic panel (PASS) and a panel with an
INJECTED look-ahead violation (FLAG). This is what makes the audit credible: it fails
loudly when the construction is dirty.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.validation.pit_audit_dpead import (
    check_seasonal_lag_integrity, check_sigma_no_lookahead, check_rdq_timing,
    check_delisting_returns, check_restatement_pit, run_pit_audit,
    SEASONAL_LAG_Q, SIGMA_WIN_Q, SIGMA_MIN_PERIODS,
)


def _synth_panel(n_q=32, lookahead_sigma=False, future_lag=False, scramble_rdq=False):
    """One firm, n_q quarters. eps_adj is a non-constant series so the current-included
    and current-excluded rolling stds differ (discriminating)."""
    rng = np.random.default_rng(0)
    eps = pd.Series(np.cumsum(rng.normal(0.05, 0.4, n_q)) + 2.0)
    fyq = [f"{2008 + i // 4}Q{i % 4 + 1}" for i in range(n_q)]
    rdq = pd.to_datetime("2008-04-25") + pd.to_timedelta(np.arange(n_q) * 91, unit="D")
    lag4 = eps.shift(SEASONAL_LAG_Q)
    delta = eps - lag4
    incl = delta.rolling(SIGMA_WIN_Q, min_periods=SIGMA_MIN_PERIODS).std()
    excl = incl.shift(1)
    sigma = incl if lookahead_sigma else excl          # inject look-ahead in sigma
    stored_lag4 = eps.shift(-SEASONAL_LAG_Q) if future_lag else lag4   # inject future lag
    if scramble_rdq:
        rdq = pd.Series(rdq[::-1].values)               # decreasing -> many negative gaps
    df = pd.DataFrame({
        "permno": 10001, "ticker": "AAA", "gvkey": 1, "fiscal_yearq": fyq, "rdq": rdq,
        "eps_adj": eps, "eps_adj_lag4": stored_lag4, "delta_eps": delta,
        "sigma_8q": sigma.where(sigma >= 0.01), "sue_raw": delta / sigma,
        "sue": (delta / sigma).clip(-10, 10), "market_cap_at_q": 10000.0,
    })
    return df


# ── sigma look-ahead (the core check) ────────────────────────────────────────
def test_sigma_clean_panel_passes():
    r = check_sigma_no_lookahead(_synth_panel(lookahead_sigma=False))
    assert r.status == "PASS"
    assert r.metric["match_excluded"] > r.metric["match_included"]


def test_sigma_injected_lookahead_is_flagged():
    r = check_sigma_no_lookahead(_synth_panel(lookahead_sigma=True))
    assert r.status == "FLAG"            # current-included sigma must be caught
    assert r.metric["match_included"] > r.metric["match_excluded"]


# ── seasonal lag integrity ────────────────────────────────────────────────────
def test_seasonal_lag_clean_passes():
    assert check_seasonal_lag_integrity(_synth_panel(future_lag=False)).status == "PASS"


def test_seasonal_lag_future_leak_is_flagged():
    # eps_adj_lag4 set to a FUTURE quarter (shift(-4)) must FLAG
    assert check_seasonal_lag_integrity(_synth_panel(future_lag=True)).status == "FLAG"


# ── rdq timing tolerance ──────────────────────────────────────────────────────
def test_rdq_monotonic_passes():
    assert check_rdq_timing(_synth_panel(scramble_rdq=False)).status == "PASS"


def test_rdq_scrambled_is_flagged():
    r = check_rdq_timing(_synth_panel(scramble_rdq=True))
    assert r.status == "FLAG" and r.metric["neg_gaps"] > 0


# ── documented limitations are always FLAG (honest negatives) ─────────────────
def test_documented_limitations_flag():
    assert check_restatement_pit().status == "FLAG"
    assert check_delisting_returns(_synth_panel()).status == "FLAG"


# ── integration on the real deployed panel (skip the heavy return pivot) ──────
def test_real_panel_is_look_ahead_clean():
    panel = Path("data/cache/_pead_ts_panel_2014_2023.parquet")
    if not panel.exists():
        pytest.skip("deployed panel cache not present")
    rep = run_pit_audit(panel_path=panel, ret_path=Path("__nonexistent__.parquet"))
    assert rep.critical_pass is True
    assert "LOOK-AHEAD CLEAN" in rep.overall
    crit = {c.name: c.status for c in rep.checks if c.name in
            {"seasonal_lag_integrity", "sigma_no_lookahead", "rdq_timing",
             "entry_skip_day", "consensus_window"}}
    assert all(s in ("PASS", "INFO") for s in crit.values()), crit
