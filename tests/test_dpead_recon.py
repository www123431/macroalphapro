"""tests/test_dpead_recon.py — reproducible D_PEAD recon (closes audit residual #3).

Pins that the combined-book 1.04 rests on a REPRODUCIBLE, look-ahead-free D_PEAD series:
a clean from-source reconstruction (engine.portfolio.dpead_recon) tracks the original
_dpead_recon_base artifact (corr > 0.85). Also pins the audit FINDING that the validated
recon is market-neutral L/S, NOT the live long-only leg.

Slow (rebuilds the daily L/S over 2014-2024); skips if the cached panel/artifact is absent.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.portfolio.dpead_recon import build_dpead_recon_returns

_PANEL = "data/cache/_pead_ts_panel_2014_2023.parquet"
_ORIG = "data/cache/_dpead_recon_base.parquet"
_skip = not (os.path.exists(_PANEL) and os.path.exists(_ORIG))


def _orig() -> pd.Series:
    s = pd.read_parquet(_ORIG).iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s


def _sharpe(x: pd.Series) -> float:
    a = (1 + x.clip(-0.2, 0.2)).resample("ME").prod() - 1
    return float(a.mean() * 12 / (a.std() * np.sqrt(12)))


@pytest.mark.skipif(_skip, reason="cached SUE panel / recon artifact absent")
def test_recon_reproduces_original_ls():
    rep = build_dpead_recon_returns(long_short=True)
    j = pd.concat([_orig().rename("o"), rep.rename("r")], axis=1).dropna()
    assert len(j) > 2000, len(j)
    assert j["o"].corr(j["r"]) > 0.85, j["o"].corr(j["r"])   # clean reproduction tracks the artifact
    assert _sharpe(rep) > 0.8                                # the signal is real + positive


@pytest.mark.skipif(_skip, reason="cached SUE panel / recon artifact absent")
def test_validated_recon_is_long_short_not_long_only():
    # Audit finding: the VALIDATED D_PEAD (recon) is market-neutral L/S — the live deployed
    # long-only leg is a DIFFERENT exposure (does not match the validated series).
    lo = build_dpead_recon_returns(long_short=False)
    j = pd.concat([_orig().rename("o"), lo.rename("r")], axis=1).dropna()
    assert j["o"].corr(j["r"]) < 0.5, j["o"].corr(j["r"])
