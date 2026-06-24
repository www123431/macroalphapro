"""tests/test_forensic_price_floor.py — the live Anomaly Sentinel price-spike flag needs economic
significance, not just a high z-score.

Regression for the 2026-05-24 fix in engine.agents.persona.tools.forensic_ticker_check: a sub-1%
daily move on a low-vol ETF can be many sigmas (statistically extreme, economically nothing) and
used to fire a price_spike. It now also requires |return| >= _FORENSIC_MIN_ABS_RETURN. yfinance is
mocked so the test is offline + deterministic.
"""
from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd
import pytest

import engine.agents.persona.tools as tools


def _frame(last_ret: float) -> pd.DataFrame:
    # 65 low-vol days (~0.05%/day) + an injected final move; both tiny & big moves are high-z.
    base = list(np.random.default_rng(0).normal(0, 0.0005, 64)) + [last_ret]
    closes = [100.0]
    for r in base:
        closes.append(closes[-1] * (1 + r))
    idx = pd.to_datetime([datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(len(closes))])
    return pd.DataFrame({"Close": closes, "Volume": [1_000_000] * len(closes)}, index=idx)


def _price_hits(ret: float, monkeypatch) -> list:
    monkeypatch.setattr("yfinance.download", lambda *a, **k: _frame(ret))
    out = json.loads(tools.forensic_ticker_check("TESTX", as_of="2026-03-10"))
    return [h for h in out.get("rule_hits", []) if h.get("rule") == "price_spike"]


def test_tiny_highz_move_does_not_flag(monkeypatch):
    # 0.3% move: ~6σ on this series, but below the 1% economic floor -> NO price_spike.
    assert _price_hits(0.003, monkeypatch) == []


def test_economically_significant_move_flags(monkeypatch):
    # 1.5% move: high z AND clears the floor -> price_spike fires.
    assert len(_price_hits(0.015, monkeypatch)) == 1
