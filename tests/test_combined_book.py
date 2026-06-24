"""tests/test_combined_book.py — 已部署的双机制合并 book。

快测:纯混合/波动目标化逻辑(合成数据)。
慢测:在缓存数据上重算，验证复现验证过的配置(carry@30% → Sharpe~1.04 / 回撤~-7.4%；
carry@0% → ~0.96)。缓存缺失则跳过。GREEN 结论本身 spec-locked，这里不重新论证。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from engine.portfolio.combined_book import (
    blend_at_risk_weight,
    book_stats,
    build_combined_book,
    scale_to_book_vol,
    voltarget,
)

_CACHE = "data/cache/_dpead_recon_base.parquet"


def test_blend_endpoints_and_mix():
    idx = pd.period_range("2010-01", periods=120, freq="M").to_timestamp()
    e = pd.Series(np.linspace(-0.01, 0.02, 120), index=idx)
    c = pd.Series(np.linspace(0.02, -0.01, 120), index=idx)
    assert np.allclose(blend_at_risk_weight(e, c, 0.0).values, e.values)   # 0% carry = 纯股票
    assert np.allclose(blend_at_risk_weight(e, c, 1.0).values, c.values)   # 100% = 纯 carry
    mix = blend_at_risk_weight(e, c, 0.30)
    assert np.allclose(mix.values, 0.7 * e.values + 0.3 * c.values)


def test_voltarget_no_lookahead_and_scales():
    idx = pd.period_range("2000-01", periods=200, freq="M").to_timestamp()
    r = pd.Series(np.random.default_rng(0).normal(0.004, 0.05, 200), index=idx)
    out = voltarget(r, target=0.10, lb=12)
    # shift(1) ⇒ 头 12 期(滚动窗口未满)为 NaN，无前视
    assert out.iloc[:12].isna().all()
    # 目标化后实现波动应大致靠近 10%(滚动估计，不会精确)
    assert 0.06 < out.dropna().std() * np.sqrt(12) < 0.16


def test_scale_to_book_vol_hits_target_preserves_sharpe():
    idx = pd.period_range("2000-01", periods=300, freq="M").to_timestamp()
    r = pd.Series(np.random.default_rng(3).normal(0.005, 0.024, 300), index=idx)  # ~8.3% ann vol
    out = scale_to_book_vol(r, 0.10)
    assert abs(out.std() * np.sqrt(12) - 0.10) < 1e-9               # hits 10% book vol
    assert abs((r.mean() / r.std()) - (out.mean() / out.std())) < 1e-9   # Sharpe unchanged


def test_book_stats_shape():
    idx = pd.period_range("2010-01", periods=60, freq="M").to_timestamp()
    r = pd.Series(np.random.default_rng(1).normal(0.006, 0.03, 60), index=idx)
    st = book_stats(r)
    assert {"n", "ann", "vol", "sharpe", "maxdd"} <= set(st)
    assert st["n"] == 60 and st["maxdd"] <= 0


@pytest.mark.skipif(not os.path.exists(_CACHE), reason="缓存缺失")
def test_reproduces_validated_config():
    # carry@30% 应复现验证区间:Sharpe ~1.0-1.1、回撤约 -6%~-9%。
    book30 = build_combined_book(carry_risk_weight=0.30)
    s30 = book_stats(book30)
    assert 0.95 <= s30["sharpe"] <= 1.15, s30
    assert -0.10 <= s30["maxdd"] <= -0.05, s30
    # carry@0% = 纯股票，Sharpe 略低、回撤更深(加 carry 是改善项)
    s0 = book_stats(build_combined_book(carry_risk_weight=0.0))
    assert s30["sharpe"] >= s0["sharpe"] - 0.02, (s0, s30)   # 加 carry 不应让 Sharpe 变差
