"""tests/test_options_skew_surface.py — SkewSurfaceLoader tests against
the freshly-fetched full SPY skew cache.

Validates:
  - Exact delta lookup at standard grid points (-90/-75/-50/-25/-10/10/25/50/75/90)
  - Maturity grid: 30/60/91/182/365
  - get_iv_interpolated() returns sensible value for non-grid delta
  - Errors are raised cleanly for unavailable data
"""
from __future__ import annotations

import datetime as _dt

import pytest

from engine.research.options.skew_surface import (
    DataNotAvailableError, SkewSurfaceLoader,
)

SPY_SECID = 108105    # SPX (S&P 500 index) — corrected from prior wrong secid=8957
TEST_DATE = _dt.date(2020, 6, 1)    # mid-window date with known coverage


@pytest.fixture(scope="module")
def loader():
    return SkewSurfaceLoader.from_cache()


class TestFullSkewLookup:
    def test_spy_atm_call_at_test_date(self, loader):
        iv = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=50, cp_flag="C", maturity_days=30,
        )
        # 2020-06-01 (post-Covid recovery) — SPY 30d ATM IV should be ~0.20-0.40
        assert 0.10 < iv < 0.60

    def test_spy_otm_put_minus_25_delta(self, loader):
        iv = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=-25, cp_flag="P", maturity_days=30,
        )
        assert 0.15 < iv < 0.70    # skew → OTM put IV higher than ATM

    def test_spy_otm_put_minus_10_delta_deeper_skew(self, loader):
        iv_25 = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=-25, cp_flag="P", maturity_days=30,
        )
        iv_10 = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=-10, cp_flag="P", maturity_days=30,
        )
        # Skew: more-OTM puts have higher IV
        assert iv_10 > iv_25

    def test_longer_maturity_available(self, loader):
        iv_30 = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=50, cp_flag="C", maturity_days=30,
        )
        iv_365 = loader.get_iv(
            secid=SPY_SECID, date=TEST_DATE,
            delta=50, cp_flag="C", maturity_days=365,
        )
        # Both should be valid IVs
        assert iv_30 > 0
        assert iv_365 > 0


class TestInterpolation:
    def test_interpolate_between_grid_deltas(self, loader):
        # delta=-30 is between -25 and -50
        iv_25 = loader.get_iv(secid=SPY_SECID, date=TEST_DATE, delta=-25,
                              cp_flag="P", maturity_days=30)
        iv_50 = loader.get_iv(secid=SPY_SECID, date=TEST_DATE, delta=-50,
                              cp_flag="P", maturity_days=30)
        iv_30 = loader.get_iv_interpolated(
            secid=SPY_SECID, date=TEST_DATE, target_delta=-30,
            cp_flag="P", maturity_days=30,
        )
        # Linear interpolation should give a value between the endpoints
        lo, hi = sorted([iv_25, iv_50])
        assert lo <= iv_30 <= hi

    def test_extrapolation_returns_nan(self, loader):
        # delta=-95 is outside grid (-90 is the deepest); should return NaN
        import math
        iv = loader.get_iv_interpolated(
            secid=SPY_SECID, date=TEST_DATE, target_delta=-95,
            cp_flag="P", maturity_days=30,
        )
        assert math.isnan(iv)


class TestErrors:
    def test_unknown_secid_raises(self, loader):
        with pytest.raises(DataNotAvailableError):
            loader.get_iv(secid=99999999, date=TEST_DATE,
                          delta=50, cp_flag="C", maturity_days=30)

    def test_unknown_maturity_raises(self, loader):
        with pytest.raises(DataNotAvailableError):
            loader.get_iv(secid=SPY_SECID, date=TEST_DATE,
                          delta=50, cp_flag="C", maturity_days=45)

    def test_date_before_data_window_raises(self, loader):
        with pytest.raises(DataNotAvailableError):
            loader.get_iv(secid=SPY_SECID, date=_dt.date(2000, 1, 1),
                          delta=50, cp_flag="C", maturity_days=30)
