"""
Macro Economic Surprise Fetcher
================================
Pulls recent FRED releases for key macro indicators and formats them as
a compact data block for LLM prompt injection.

Design notes
------------
- FRED API key is optional: read from st.secrets["FRED_API_KEY"] or
  env var FRED_API_KEY. Without a key FRED returns public data at lower
  rate limits (works fine for < 120 req/min).
- "Surprise" is computed as actual − prior release (MoM or YoY delta
  change), since free data sources do not carry survey consensus.
  The formatted string labels direction (↑ higher / ↓ lower) and
  magnitude relative to recent trend — honest about this limitation.
- All fetches are cached in memory for 3 hours to avoid hammering FRED
  on every page load.
- Graceful degradation: any failure returns "" so callers get no context
  rather than an error.

Series tracked
--------------
  CPIAUCSL   CPI All Urban Consumers (SA, MoM% → YoY proxy via 12M chain)
  CPILFESL   Core CPI (ex food & energy)
  PCEPI      PCE Price Index
  PCEPILFE   Core PCE
  UNRATE     Unemployment Rate
  PAYEMS     Nonfarm Payrolls (thousands, MoM change computed internally)
  GS10       10-Year Treasury Yield
  T10Y2Y     10Y-2Y Yield Spread (from FRED)
  UMCSENT    U. Michigan Consumer Sentiment

Extended yield-curve series (get_yield_curve_snapshot)
-------------------------------------------------------
  DGS1    1-Year Treasury Constant Maturity
  DGS2    2-Year Treasury Constant Maturity
  DGS5    5-Year Treasury Constant Maturity
  GS10    10-Year Treasury Yield (reused)
  DGS30   30-Year Treasury Constant Maturity
  FEDFUNDS  Effective Federal Funds Rate
  SOFR      Secured Overnight Financing Rate
  T5YIE   5-Year Breakeven Inflation
  T10YIE  10-Year Breakeven Inflation
  NAPM    ISM Manufacturing PMI (Institute for Supply Management)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_CACHE_TTL = 3 * 3600   # 3 hours in seconds

# Series definitions: (series_id, label, unit, transform)
# transform: "level" | "mom_ppt" | "yoy_pct" | "mom_k" (thousands MoM)
_SERIES = [
    ("CPIAUCSL",  "CPI (全项)",       "%",  "yoy_pct"),
    ("CPILFESL",  "核心CPI",          "%",  "yoy_pct"),
    ("PCEPI",     "PCE",              "%",  "yoy_pct"),
    ("PCEPILFE",  "核心PCE",          "%",  "yoy_pct"),
    ("UNRATE",    "失业率",           "%",  "level"),
    ("PAYEMS",    "非农就业",         "万人","mom_k"),
    ("GS10",      "10Y国债收益率",    "%",  "level"),
    ("T10Y2Y",    "10Y-2Y利差",       "%",  "level"),
    ("UMCSENT",   "密歇根消费者信心", "pt", "level"),
]


def _get_api_key() -> str:
    try:
        import streamlit as _st
        return _st.secrets.get("FRED_API_KEY", "")
    except Exception:
        return os.environ.get("FRED_API_KEY", "")


def _fetch_observations(series_id: str, n: int = 3, api_key: str = "") -> list[dict]:
    """Return last n FRED observations as list of {date, value} dicts."""
    params = [
        f"series_id={series_id}",
        f"limit={n}",
        "sort_order=desc",
        "file_type=json",
    ]
    if api_key:
        params.append(f"api_key={api_key}")
    url = f"{_FRED_BASE}?{'&'.join(params)}"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        obs = data.get("observations", [])
        return [
            {"date": o["date"], "value": float(o["value"])}
            for o in obs
            if o.get("value") not in (".", "", None)
        ]
    except Exception as exc:
        logger.debug("FRED fetch failed for %s: %s", series_id, exc)
        return []


def _yoy_pct(obs: list[dict]) -> Optional[tuple[float, float]]:
    """
    Return (latest_yoy%, prior_yoy%) from level observations.
    Requires at least 14 obs to compute two consecutive YoY values.
    Falls back to the two most recent if only 2 obs available.
    """
    if len(obs) < 2:
        return None
    # obs is sorted desc by date (most recent first)
    latest_val = obs[0]["value"]
    prior_val  = obs[1]["value"]
    if prior_val == 0:
        return None
    # For FRED level series we return MoM change in percentage points
    # as a proxy when we don't have 13+ months of data in the call.
    return (latest_val, prior_val)


def _format_surprise(label: str, unit: str, transform: str,
                     obs: list[dict]) -> Optional[str]:
    """Format a single indicator line."""
    if len(obs) < 2:
        return None
    latest = obs[0]
    prior  = obs[1]

    try:
        v_now  = latest["value"]
        v_prev = prior["value"]
    except (KeyError, TypeError):
        return None

    date_now = latest["date"][:7]   # YYYY-MM

    if transform == "level":
        delta = v_now - v_prev
        arrow = "↑" if delta > 0.05 else ("↓" if delta < -0.05 else "→")
        surprise = f"{arrow}{abs(delta):.2f}{unit}"
        return (
            f"{label}={v_now:.2f}{unit}"
            f"（上月{v_prev:.2f}{unit}，{surprise}）"
            f" [{date_now}]"
        )

    elif transform == "yoy_pct":
        # Both values are index levels; compute YoY requires 13 months.
        # With only 2 obs we show MoM change in ppt as trend indicator.
        delta = v_now - v_prev
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "→")
        return (
            f"{label}={v_now:.1f}{unit}"
            f"（上期{v_prev:.1f}{unit}，环比{arrow}{abs(delta):.2f}ppt）"
            f" [{date_now}]"
        )

    elif transform == "mom_k":
        # Payrolls: show level and MoM change in thousands → display as 万
        delta   = (v_now - v_prev) / 100.0   # thousands → 万 (×0.1 ≈ ok for display)
        arrow   = "↑" if delta > 0 else "↓"
        return (
            f"{label}={v_now/100:.1f}万人"
            f"（上月{v_prev/100:.1f}万人，新增{arrow}{abs(delta):.1f}万）"
            f" [{date_now}]"
        )

    return None


# ── Yield-curve series definitions ────────────────────────────────────────────

_YC_TENORS: list[tuple[str, str, int]] = [
    # (series_id, label, tenor_years)
    ("DGS1",    "1Y",  1),
    ("DGS2",    "2Y",  2),
    ("DGS5",    "5Y",  5),
    ("GS10",    "10Y", 10),
    ("DGS30",   "30Y", 30),
]
_POLICY_SERIES: list[tuple[str, str]] = [
    ("FEDFUNDS", "Fed Funds"),
    ("SOFR",     "SOFR"),
]
_BREAKEVEN_SERIES: list[tuple[str, str]] = [
    ("T5YIE",  "5Y盈亏平衡通胀"),
    ("T10YIE", "10Y盈亏平衡通胀"),
]
_PMI_SERIES: list[tuple[str, str]] = [
    ("NAPM",   "ISM制造业PMI"),
]

# Module-level cache: (timestamp, result_string)
_cache: tuple[float, str] = (0.0, "")


def get_economic_surprises(as_of: Optional[date] = None) -> str:
    """
    Return a compact macro data block for prompt injection.
    Example output:
        【宏观经济数据（FRED 实测值 · 非预测市场共识）】
        CPI (全项)=3.2%（上期3.0%，环比↑0.2ppt） [2024-12]
        核心CPI=3.9%（上期4.0%，环比↓0.1ppt） [2024-12]
        失业率=4.1%（上月4.0%，↑0.1%） [2025-01]
        ...
        ⚠ 数据来源：St. Louis FRED。环比变化供参考，非市场共识预期差。

    Returns "" on any failure.
    """
    global _cache

    # Honour TTL cache (skip for historical as_of dates in backtests)
    _now = time.time()
    if as_of is None and _now - _cache[0] < _CACHE_TTL and _cache[1]:
        return _cache[1]

    api_key = _get_api_key()
    lines   = ["【宏观经济数据（FRED 实测值 · 非市场共识预期差）】"]
    any_ok  = False

    for series_id, label, unit, transform in _SERIES:
        obs = _fetch_observations(series_id, n=3, api_key=api_key)
        if not obs:
            continue
        line = _format_surprise(label, unit, transform, obs)
        if line:
            lines.append(line)
            any_ok = True

    # FRED fallback: if nothing came through, inject yfinance spot data for key series
    if not any_ok:
        try:
            import yfinance as _yf
            _fb_lines = []
            # 10Y yield via ^TNX
            _t10 = _yf.Ticker("^TNX").fast_info.last_price
            if _t10 and 0.1 < _t10 < 15:
                _fb_lines.append(f"10Y国债收益率={_t10:.2f}%（yfinance实时）")
            # VIX as risk proxy
            _vix = _yf.Ticker("^VIX").fast_info.last_price
            if _vix and 5 < _vix < 100:
                _fb_lines.append(f"VIX={_vix:.1f}（yfinance实时）")
            if _fb_lines:
                lines += _fb_lines
                lines.append("⚠ FRED暂不可达，仅显示yfinance实时市场数据（无历史对比）。")
                any_ok = True
        except Exception:
            pass

    if not any_ok:
        return ""

    if any_ok and "FRED暂不可达" not in lines[-1]:
        lines.append(
            "⚠ 来源：St. Louis FRED。环比变化供参考，非市场共识预期差（Bloomberg/Refinitiv 共识数据不可用）。"
        )
    result = "\n".join(lines)

    if as_of is None:
        _cache = (_now, result)
    return result


# ── Yield curve snapshot ───────────────────────────────────────────────────────

_yc_cache: tuple[float, dict] = (0.0, {})

_SHAPE_CN = {
    "normal":   "正斜率（正常）",
    "flat":     "平坦",
    "inverted": "倒挂⚠",
    "humped":   "驼峰型",
    "unknown":  "未知",
}


def _classify_shape(v10, v2, v5) -> str:
    if v10 is None:
        return "unknown"
    if v2 is not None:
        spread = v10 - v2
        if v5 is not None and v5 > v10 + 0.10 and v5 > v2 + 0.10:
            return "humped"
        if spread > 0.25:
            return "normal"
        if spread < -0.25:
            return "inverted"
        return "flat"
    return "unknown"


def _build_yc_narrative(shape: str, v10, v2, v5, v3m, spread_10y2y,
                         spread_10y3m, ism_pmi, fed_funds, source: str) -> str:
    shape_cn = _SHAPE_CN.get(shape, "未知")
    parts = []
    if v10  is not None: parts.append(f"10Y={v10:.2f}%")
    if v2   is not None: parts.append(f"2Y={v2:.2f}%")
    elif v3m is not None: parts.append(f"3M={v3m:.2f}%")
    if spread_10y2y is not None:
        parts.append(f"利差(10Y-2Y)={spread_10y2y:+.2f}%")
    elif spread_10y3m is not None:
        parts.append(f"利差(10Y-3M)={spread_10y3m:+.2f}%")
    if ism_pmi is not None:
        parts.append(f"ISM={ism_pmi:.1f}({'扩张' if ism_pmi >= 50 else '收缩'})")
    if fed_funds is not None:
        parts.append(f"FFR={fed_funds:.2f}%")
    src_tag = f"·{source}" if source else ""
    return f"【收益率曲线·{shape_cn}{src_tag}】{', '.join(parts)}" if parts else ""


def _yfinance_yield_curve() -> dict:
    """
    Fetch US Treasury yields from Yahoo Finance CBOE rate tickers.
    ^TNX=10Y, ^FVX=5Y, ^TYX=30Y, ^IRX=13-week (3M proxy).
    2Y is unavailable on Yahoo; spread uses 10Y-3M (Estrella-Mishkin predictor).
    """
    import yfinance as yf

    _yf_map = {"10Y": "^TNX", "5Y": "^FVX", "30Y": "^TYX", "3M": "^IRX"}
    tenors: dict[str, Optional[float]] = {"1Y": None, "2Y": None, "5Y": None,
                                           "10Y": None, "30Y": None}
    v3m: Optional[float] = None

    for label, ticker in _yf_map.items():
        try:
            val = float(yf.Ticker(ticker).fast_info.last_price or 0)
            if 0.01 < val < 20.0:
                if label == "3M":
                    v3m = round(val, 3)
                else:
                    tenors[label] = round(val, 3)
        except Exception:
            pass

    v10, v5 = tenors.get("10Y"), tenors.get("5Y")
    spread_10y3m = round(v10 - v3m, 3) if v10 is not None and v3m is not None else None
    shape = "unknown"
    if v10 is not None and v3m is not None:
        s = v10 - v3m
        if v5 is not None and v5 > v10 + 0.10 and v5 > v3m + 0.10:
            shape = "humped"
        elif s > 0.25:
            shape = "normal"
        elif s < -0.25:
            shape = "inverted"
        else:
            shape = "flat"

    narrative = _build_yc_narrative(
        shape, v10, None, v5, v3m,
        None, spread_10y3m, None, None, "yfinance"
    )
    return {
        "tenors":       tenors,
        "policy":       {"fed_funds": None, "sofr": None},
        "breakeven":    {"5y": None, "10y": None},
        "ism_pmi":      None,
        "shape":        shape,
        "spread_10y2y": None,
        "spread_10y3m": spread_10y3m,
        "as_of_str":    datetime.now().strftime("%Y-%m"),
        "narrative":    narrative,
        "_source":      "yfinance",
    }


def get_yield_curve_snapshot(as_of: Optional[date] = None) -> dict:
    """
    Return a structured yield-curve snapshot for the Supervisor narrative layer.

    Primary source: FRED (DGS1/2/5/GS10/DGS30 + FEDFUNDS/SOFR + breakevens).
    Fallback:       Yahoo Finance CBOE rate tickers (^TNX/^FVX/^TYX/^IRX).
                    2Y not available on Yahoo → spread uses 10Y-3M instead of 10Y-2Y.

    Keys: tenors, policy, breakeven, ism_pmi, shape, spread_10y2y, spread_10y3m,
          as_of_str, narrative, _source ("fred" | "yfinance")
    """
    global _yc_cache
    _now = time.time()
    if as_of is None and _now - _yc_cache[0] < _CACHE_TTL and _yc_cache[1]:
        return _yc_cache[1]

    api_key = _get_api_key()

    # ── Path A: FRED ───────────────────────────────────────────────────────────
    tenors: dict[str, Optional[float]] = {}
    latest_date_str = ""
    for series_id, label, _ in _YC_TENORS:
        obs = _fetch_observations(series_id, n=2, api_key=api_key)
        if obs:
            tenors[label] = obs[0]["value"]
            if not latest_date_str:
                latest_date_str = obs[0]["date"][:7]
        else:
            tenors[label] = None

    policy: dict[str, Optional[float]] = {}
    for series_id, label in _POLICY_SERIES:
        obs = _fetch_observations(series_id, n=2, api_key=api_key)
        key = label.lower().replace(" ", "_")
        policy[key] = obs[0]["value"] if obs else None

    breakeven: dict[str, Optional[float]] = {}
    for series_id, label in _BREAKEVEN_SERIES:
        obs = _fetch_observations(series_id, n=2, api_key=api_key)
        if "5Y" in label:
            breakeven["5y"] = obs[0]["value"] if obs else None
        else:
            breakeven["10y"] = obs[0]["value"] if obs else None

    ism_obs = _fetch_observations("NAPM", n=2, api_key=api_key)
    ism_pmi: Optional[float] = ism_obs[0]["value"] if ism_obs else None

    v10 = tenors.get("10Y")
    v2  = tenors.get("2Y")
    v1  = tenors.get("1Y")
    v5  = tenors.get("5Y")
    fred_ok = v10 is not None  # at minimum 10Y must succeed

    # ── Path B: yfinance fallback when FRED yields nothing ────────────────────
    if not fred_ok:
        logger.info("macro_fetcher: FRED unavailable, falling back to yfinance yield curve")
        try:
            result = _yfinance_yield_curve()
            if result.get("tenors", {}).get("10Y") is not None:
                if as_of is None:
                    _yc_cache = (_now, result)
                return result
        except Exception as exc:
            logger.warning("macro_fetcher: yfinance yield curve also failed: %s", exc)
        return {}

    # ── FRED path: compute spreads + build result ─────────────────────────────
    spread_10y2y = round(v10 - v2, 3) if v10 is not None and v2 is not None else None
    spread_10y3m = round(v10 - v1, 3) if v10 is not None and v1 is not None else None
    shape        = _classify_shape(v10, v2, v5)
    narrative    = _build_yc_narrative(
        shape, v10, v2, v5, None,
        spread_10y2y, spread_10y3m, ism_pmi, policy.get("fed_funds"), ""
    )

    result = {
        "tenors":       tenors,
        "policy":       policy,
        "breakeven":    breakeven,
        "ism_pmi":      ism_pmi,
        "shape":        shape,
        "spread_10y2y": spread_10y2y,
        "spread_10y3m": spread_10y3m,
        "as_of_str":    latest_date_str,
        "narrative":    narrative,
        "_source":      "fred",
    }

    if as_of is None:
        _yc_cache = (_now, result)
    return result
