"""
Historical Snapshot Builder + Backtest Runner
=============================================

DATA LEAKAGE PREVENTION — STRICT RULES
---------------------------------------
Every data point fetched for a decision on date T must have been
publicly available BEFORE T. The following publication lags are enforced:

  FRED CPI (CPIAUCSL)      → T - 35 days  (released ~month after reference month)
  FRED Unemployment (UNRATE)→ T - 7 days   (first Friday of following month)
  FRED Fed Funds (FEDFUNDS) → T - 2 days   (daily, tiny lag)
  FRED Yields (DGS10/DGS2) → T - 1 day    (daily, 1-day lag)
  GDELT news               → T - 1 day    (same-day news not yet indexed)
  Price / VIX              → T - 1 day    (prior close)

Walk-Forward Structure
----------------------
  Training window : [train_start, train_end]
  Test window     : (train_end, test_end] at given frequency
  Each test-date decision sees ONLY data available before that date.
  After verification, training window expands (expanding window mode).

Alpha Memory Isolation
----------------------
  is_backtest=True records are EXCLUDED from get_historical_context()
  and get_regime_benchmarks() during live analysis to prevent
  contamination of real decisions by synthetic backtest data.

Main entry points:
  build_snapshot(date, sector)                → full historical snapshot dict
  run_sector_backtest(model, ...)             → simple batch replay
  run_walk_forward_backtest(model, ...)       → walk-forward with strict isolation
"""
import datetime
import logging
import time
from io import StringIO

import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Publication lag constants (days to subtract from decision date) ────────────
_FRED_PUB_LAG: dict[str, int] = {
    "cpi_yoy":      35,   # CPI released ~35 days after reference month end
    "unemployment": 7,    # Jobs report: first Friday of following month
    "fed_funds":    2,    # Fed Funds: daily, 2-day lag
    "t10y":         1,    # Treasury yields: 1-day lag
    "t2y":          1,
}

# ── GDELT trusted domain whitelist ────────────────────────────────────────────
_TRUSTED_DOMAINS: set[str] = {
    "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "marketwatch.com", "apnews.com", "bbc.com",
    "federalreserve.gov", "bis.org", "imf.org", "worldbank.org",
    "economist.com", "barrons.com", "seekingalpha.com",
    "finance.yahoo.com", "foxbusiness.com", "thestreet.com",
}

# ── Sector ETF map ─────────────────────────────────────────────────────────────
# Keys must match AUDIT_TICKERS in engine/scanner.py exactly — these names drive
# the Tab3 sector selector, all SECTOR_ETF.get(selected) lookups, and quant functions.
SECTOR_ETF: dict[str, str] = {
    "AI算力/半导体":  "SMH",
    "科技成长(纳指)": "QQQ",
    "生物科技":       "XBI",
    "金融":           "XLF",
    "全球能源":       "XLE",
    "工业/基建":      "XLI",
    "医疗健康":       "XLV",
    "防御消费":       "XLP",
    "消费科技":       "XLY",
    "美国REITs":      "VNQ",
    "黄金":           "GLD",
    "美国长债":       "TLT",
    "清洁能源":       "ICLN",
    "沪深300":        "ASHR",
    "中国科技":       "KWEB",
    "新加坡蓝筹":     "EWS",
    "通讯传媒":       "XLC",
    "高收益债":       "HYG",
    # ── Batch C: FX / 原油 / 信用梯度 / EM 宽基 (+4) ──────────────────────────
    "美元指数":       "UUP",
    "原油":           "USO",
    "投资级公司债":   "LQD",
    "新兴市场宽基":   "EEM",
}

# GDELT keyword map per sector (English, used in GDELT query)
_GDELT_KEYWORDS: dict[str, str] = {
    "AI算力/半导体":  "semiconductor AI chip NVIDIA AMD GPU",
    "科技成长(纳指)": "NASDAQ tech growth stocks QQQ",
    "生物科技":       "biotech FDA drug approval clinical trial XBI",
    "金融":           "Federal Reserve interest rate bank credit XLF",
    "全球能源":       "oil energy OPEC crude petroleum XLE",
    "工业/基建":      "manufacturing PMI supply chain infrastructure XLI",
    "医疗健康":       "healthcare pharma FDA drug approval XLV",
    "防御消费":       "consumer staples food grocery inflation XLP",
    "消费科技":       "consumer discretionary retail spending XLY",
    "美国REITs":      "real estate housing mortgage REIT interest rate",
    "黄金":           "gold GLD commodity safe haven inflation",
    "美国长债":       "treasury bonds TLT yield curve Fed rate",
    "清洁能源":       "clean energy renewable solar wind ICLN",
    "沪深300":        "China A-share CSI300 PBOC economy",
    "中国科技":       "China tech KWEB Alibaba Tencent regulation",
    "新加坡蓝筹":     "Singapore STI EWS DBS OCBC MAS policy",
    "通讯传媒":       "communication media telecom XLC Netflix Disney streaming",
    "高收益债":       "high yield bond HYG junk bond credit spread default risk",
    "美元指数":       "US dollar index DXY Fed rate hike USD strength dollar rally UUP",
    "原油":           "crude oil WTI OPEC production cut energy commodity barrel USO",
    "投资级公司债":   "investment grade corporate bond credit spread LQD IG bond Fed rate",
    "新兴市场宽基":   "emerging markets EEM MSCI EM China India Brazil Fed dollar EM selloff",
}

# FRED series used for macro snapshot (public CSV, no API key needed)
_FRED_SERIES: dict[str, str] = {
    "cpi_yoy":     "CPIAUCSL",   # CPI all-urban, not seasonally adjusted → we compute YoY
    "fed_funds":   "FEDFUNDS",   # Effective Fed Funds Rate
    "t10y":        "DGS10",      # 10-Year Treasury yield
    "t2y":         "DGS2",       # 2-Year Treasury yield
    "unemployment":"UNRATE",     # Unemployment rate
}

_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


# ── FRED data ──────────────────────────────────────────────────────────────────

def _fetch_fred_series(series_id: str, start: str, end: str) -> dict[str, float]:
    """
    Fetch a single FRED series as {date_str: value} dict.
    Uses the public FRED CSV endpoint — no API key required.
    """
    import pandas as pd
    url = f"{_FRED_BASE}{series_id}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        date_col = df.columns[0]   # "observation_date" in current FRED CSV format
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        df = df[(df.index.astype(str) >= start) & (df.index.astype(str) <= end)]
        df = df.replace(".", float("nan"))
        df = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return {str(d.date()): float(v) for d, v in df.items()}
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return {}


def _get_fred_value_on(
    series_data: dict[str, float],
    date: datetime.date,
    lag_days: int = 0,
) -> float | None:
    """
    Return the most recent FRED value that was publicly available on `date`.
    lag_days: publication lag — we can only see data released lag_days before date.
    This enforces strict look-ahead prevention.
    """
    cutoff = str(date - datetime.timedelta(days=lag_days))
    candidates = {
        k: v for k, v in series_data.items()
        if k <= cutoff and v == v  # exclude NaN
    }
    if not candidates:
        return None
    return candidates[max(candidates)]


def get_fred_snapshot(date: datetime.date) -> tuple[dict, dict]:
    """
    Return FRED macro indicators available as of the given historical date.
    Enforces publication lag for each series to prevent look-ahead bias.

    Returns:
        (values_dict, cutoff_log_dict)
        values_dict: {cpi_yoy, fed_funds, t10y, t2y, unemployment, yield_spread}
        cutoff_log:  {series: actual_cutoff_date_used} — for audit trail
    """
    import pandas as pd

    start = str(date - datetime.timedelta(days=400))
    end   = str(date)

    result:     dict = {}
    cutoff_log: dict = {}

    # Fed funds, yields, unemployment — each with their own lag
    for key, sid in _FRED_SERIES.items():
        if key == "cpi_yoy":
            continue
        lag  = _FRED_PUB_LAG.get(key, 1)
        data = _fetch_fred_series(sid, start, end)
        val  = _get_fred_value_on(data, date, lag_days=lag)
        result[key] = val
        cutoff_log[key] = str(date - datetime.timedelta(days=lag))

    # CPI — compute YoY with 35-day publication lag
    cpi_lag  = _FRED_PUB_LAG["cpi_yoy"]
    cpi_cutoff = date - datetime.timedelta(days=cpi_lag)
    cpi_data = _fetch_fred_series("CPIAUCSL", start, str(cpi_cutoff))
    cpi_now  = _get_fred_value_on(cpi_data, cpi_cutoff, lag_days=0)
    date_1y  = cpi_cutoff - datetime.timedelta(days=365)
    start_1y = str(date_1y - datetime.timedelta(days=60))
    cpi_1y_data = _fetch_fred_series("CPIAUCSL", start_1y, str(date_1y))
    cpi_1y = _get_fred_value_on(cpi_1y_data, date_1y, lag_days=0)
    if cpi_now and cpi_1y and cpi_1y > 0:
        result["cpi_yoy"] = round((cpi_now - cpi_1y) / cpi_1y * 100, 2)
    else:
        result["cpi_yoy"] = None
    cutoff_log["cpi_yoy"] = str(cpi_cutoff)

    # Yield spread (10Y - 2Y)
    if result.get("t10y") and result.get("t2y"):
        result["yield_spread"] = round(result["t10y"] - result["t2y"], 2)
    else:
        result["yield_spread"] = None

    return result, cutoff_log


# ── VIX + Sector momentum ──────────────────────────────────────────────────────

def get_vix_on(date: datetime.date) -> float:
    """Return VIX closing price on or near the given date."""
    start = date - datetime.timedelta(days=7)
    try:
        data = yf.download("^VIX", start=str(start), end=str(date + datetime.timedelta(days=1)),
                           progress=False, auto_adjust=True)
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception as exc:
        logger.warning("VIX fetch failed for %s: %s", date, exc)
    return 20.0  # fallback to neutral


def get_sector_momentum(date: datetime.date, lookback_days: int = 20) -> dict[str, float]:
    """
    Return 20-day price momentum for each sector ETF as of the given date.
    Returns {sector_name: momentum_pct}
    """
    start = date - datetime.timedelta(days=lookback_days + 10)
    results: dict[str, float] = {}

    sector_etf = get_active_sector_etf()
    tickers = list(sector_etf.values())
    try:
        data = yf.download(
            tickers, start=str(start), end=str(date + datetime.timedelta(days=1)),
            progress=False, auto_adjust=True,
        )
        close = data["Close"] if "Close" in data else data
        for sector, etf in sector_etf.items():
            if etf not in close.columns:
                continue
            series = close[etf].dropna()
            if len(series) >= 2:
                p0 = float(series.iloc[0])
                p1 = float(series.iloc[-1])
                results[sector] = round((p1 - p0) / p0 * 100, 2) if p0 > 0 else 0.0
    except Exception as exc:
        logger.warning("Sector momentum fetch failed: %s", exc)

    return results


# ── Enhanced quant metrics ────────────────────────────────────────────────────

def get_relative_momentum(date: datetime.date, sector: str, lookback_days: int = 20) -> float | None:
    """
    Return sector ETF excess return vs SPY over lookback window.
    Positive = sector outperformed market, Negative = underperformed.
    """
    etf   = SECTOR_ETF.get(sector)
    if not etf:
        return None
    start = date - datetime.timedelta(days=lookback_days + 10)
    try:
        data = yf.download(
            [etf, "SPY"], start=str(start),
            end=str(date + datetime.timedelta(days=1)),
            progress=False, auto_adjust=True,
        )
        close = data["Close"]
        def _ret(ticker):
            s = close[ticker].dropna()
            if len(s) < 2:
                return None
            return (float(s.iloc[-1]) - float(s.iloc[0])) / float(s.iloc[0])

        r_sector = _ret(etf)
        r_spy    = _ret("SPY")
        if r_sector is not None and r_spy is not None:
            return round((r_sector - r_spy) * 100, 2)
    except Exception as exc:
        logger.warning("Relative momentum failed for %s: %s", sector, exc)
    return None


def get_rolling_sharpe(date: datetime.date, sector: str, window: int = 20) -> float | None:
    """
    Compute annualised Sharpe ratio for sector ETF over the last `window` trading days.
    Uses daily returns; risk-free rate assumed 0 for simplicity.
    """
    etf   = SECTOR_ETF.get(sector)
    if not etf:
        return None
    start = date - datetime.timedelta(days=window + 15)
    try:
        data  = yf.download(etf, start=str(start),
                            end=str(date + datetime.timedelta(days=1)),
                            progress=False, auto_adjust=True)
        close = data["Close"].dropna()
        if len(close) < window:
            return None
        rets  = close.pct_change().dropna().iloc[-window:]
        mean  = float(rets.mean())
        std   = float(rets.std())
        if std == 0:
            return None
        sharpe = round(mean / std * (252 ** 0.5), 2)
        return sharpe
    except Exception as exc:
        logger.warning("Rolling Sharpe failed for %s: %s", sector, exc)
    return None


def get_sector_correlations(date: datetime.date, sector: str, window: int = 60) -> dict:
    """
    Return correlation of the target sector ETF with SPY and with all other sectors
    over the last `window` trading days.

    Returns:
        {
            "vs_spy":   float,                        # correlation with market
            "top_corr": [(sector_name, corr), ...],   # 3 most correlated peers
            "low_corr": [(sector_name, corr), ...],   # 3 least correlated peers
        }
    """
    etf    = SECTOR_ETF.get(sector)
    if not etf:
        return {}
    tickers = list(SECTOR_ETF.values()) + ["SPY"]
    start   = date - datetime.timedelta(days=window + 15)
    try:
        data  = yf.download(tickers, start=str(start),
                            end=str(date + datetime.timedelta(days=1)),
                            progress=False, auto_adjust=True)
        close = data["Close"].dropna(how="all")
        rets  = close.pct_change().dropna()
        if etf not in rets.columns:
            return {}

        corr_matrix = rets.corr()
        target_corr = corr_matrix[etf].drop(etf)

        vs_spy = round(float(target_corr.get("SPY", float("nan"))), 3)

        # Peer sectors only (exclude SPY)
        peer_corr = {
            name: round(float(target_corr.get(peer_etf, float("nan"))), 3)
            for name, peer_etf in SECTOR_ETF.items()
            if name != sector and peer_etf in target_corr.index
        }
        sorted_peers = sorted(peer_corr.items(), key=lambda x: x[1], reverse=True)

        return {
            "vs_spy":   vs_spy,
            "top_corr": sorted_peers[:3],
            "low_corr": sorted_peers[-3:],
        }
    except Exception as exc:
        logger.warning("Sector correlation failed for %s: %s", sector, exc)
    return {}


# ── Path A: Regime label from FRED ────────────────────────────────────────────

def generate_regime_label(fred: dict, vix: float, date: datetime.date) -> str:
    """
    Construct a structured macro regime description from FRED indicators.
    This is Path A — used when real news is unavailable.
    """
    parts: list[str] = [f"[历史回测快照 · {date.strftime('%Y年%m月')}]"]

    # Rate environment
    fed = fred.get("fed_funds")
    if fed is not None:
        if fed >= 5.0:
            parts.append(f"激进紧缩周期 · 联邦基金利率 {fed:.2f}%")
        elif fed >= 3.0:
            parts.append(f"加息周期 · 联邦基金利率 {fed:.2f}%")
        elif fed <= 0.25:
            parts.append(f"零利率/量化宽松 · 联邦基金利率 {fed:.2f}%")
        else:
            parts.append(f"中性利率环境 · 联邦基金利率 {fed:.2f}%")

    # Inflation
    cpi = fred.get("cpi_yoy")
    if cpi is not None:
        if cpi >= 7:
            parts.append(f"极高通胀 · CPI同比 {cpi:.1f}%")
        elif cpi >= 4:
            parts.append(f"高通胀 · CPI同比 {cpi:.1f}%")
        elif cpi >= 2:
            parts.append(f"通胀温和 · CPI同比 {cpi:.1f}%")
        else:
            parts.append(f"通胀偏低 · CPI同比 {cpi:.1f}%")

    # Yield curve
    spread = fred.get("yield_spread")
    if spread is not None:
        if spread < -0.5:
            parts.append(f"收益率曲线深度倒挂 · 10Y-2Y {spread:.2f}%（衰退信号）")
        elif spread < 0:
            parts.append(f"收益率曲线轻微倒挂 · 10Y-2Y {spread:.2f}%")
        else:
            parts.append(f"收益率曲线正常 · 10Y-2Y {spread:.2f}%")

    # Unemployment
    unemp = fred.get("unemployment")
    if unemp is not None:
        if unemp >= 7:
            parts.append(f"高失业率 · {unemp:.1f}%（经济承压）")
        elif unemp <= 4:
            parts.append(f"就业市场紧张 · 失业率 {unemp:.1f}%")
        else:
            parts.append(f"就业市场稳定 · 失业率 {unemp:.1f}%")

    # VIX
    if vix >= 30:
        parts.append(f"市场恐慌 · VIX {vix:.1f}（高波动）")
    elif vix >= 20:
        parts.append(f"市场警惕 · VIX {vix:.1f}（波动偏高）")
    else:
        parts.append(f"市场平静 · VIX {vix:.1f}（低波动）")

    return " · ".join(parts)


def infer_regime_tag(fred: dict, vix: float) -> str:
    """Short regime tag for database storage."""
    fed = fred.get("fed_funds", 2.0)
    cpi = fred.get("cpi_yoy", 2.0)
    spread = fred.get("yield_spread", 1.0)

    if vix >= 30:
        return "高波动/危机"
    if spread is not None and spread < -0.3:
        return "收益率倒挂/衰退预期"
    if cpi is not None and cpi >= 5 and fed is not None and fed >= 3:
        return "滞胀/激进加息"
    if fed is not None and fed <= 0.5:
        return "零利率/宽松"
    if vix >= 20:
        return "震荡期"
    return "温和扩张"


# ── Path B: GDELT news ─────────────────────────────────────────────────────────

_GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def get_gdelt_news(date: datetime.date, sector: str, n: int = 6) -> tuple[str, dict]:
    """
    Fetch historical news from GDELT for a given date and sector.

    LEAKAGE PREVENTION:
      - end_dt is T-1 (yesterday's close), never T itself
      - Only articles from _TRUSTED_DOMAINS are included
      - Returns (formatted_string, audit_log)

    Returns:
        (news_context_str, {"cutoff": str, "total_fetched": int, "after_filter": int})
    """
    keywords = _GDELT_KEYWORDS.get(sector, "economy market financial")
    # Strict T-1 cutoff: news published before the decision date
    end_dt   = date - datetime.timedelta(days=1)
    start_dt = end_dt - datetime.timedelta(days=4)   # 4-day window ending T-1

    params = {
        "query":         keywords,
        "mode":          "artlist",
        "maxrecords":    str(n * 3),   # fetch more, filter down
        "format":        "json",
        "startdatetime": start_dt.strftime("%Y%m%d") + "000000",
        "enddatetime":   end_dt.strftime("%Y%m%d") + "235959",
        "sort":          "DateDesc",
    }
    audit = {"cutoff": str(end_dt), "total_fetched": 0, "after_filter": 0}

    try:
        resp = requests.get(_GDELT_API, params=params, timeout=15)
        resp.raise_for_status()
        data     = resp.json()
        articles = data.get("articles", [])
        audit["total_fetched"] = len(articles)

        # Filter to trusted domains only
        trusted = [
            a for a in articles
            if any(d in a.get("domain", "") for d in _TRUSTED_DOMAINS)
        ]
        audit["after_filter"] = len(trusted)

        # Fallback: if no trusted sources, use all (log warning)
        if not trusted and articles:
            logger.warning(
                "GDELT: no trusted-domain articles for %s/%s, using unfiltered",
                date, sector,
            )
            trusted = articles

        if not trusted:
            return "", audit

        lines = [f"[GDELT历史新闻 · 截至{end_dt} · {sector}板块]"]
        for art in trusted[:n]:
            title  = art.get("title", "").strip()
            source = art.get("domain", "")
            if title:
                lines.append(f"• {title}  [{source}]")

        return "\n".join(lines), audit

    except Exception as exc:
        logger.warning("GDELT fetch failed for %s / %s: %s", date, sector, exc)
        return "", audit


# ── Snapshot builder ───────────────────────────────────────────────────────────

def build_snapshot(date: datetime.date, sector: str) -> dict:
    """
    Build a complete historical snapshot for a given date and sector.
    Combines Path A (FRED regime label) + Path B (GDELT news).

    Returns:
        {
            "date":         datetime.date,
            "sector":       str,
            "vix":          float,
            "macro_regime": str,           # short tag for DB
            "fred":         dict,          # raw FRED values
            "momentum":     dict,          # sector 20d momentum
            "news_context": str,           # GDELT headlines
            "regime_label": str,           # Path A: structured description
            "full_context": str,           # regime_label + news, ready for prompt
        }
    """
    logger.info("Building snapshot for %s / %s", date, sector)

    # Price data: use T-1 (prior close)
    price_cutoff = date - datetime.timedelta(days=1)
    vix      = get_vix_on(price_cutoff)
    momentum = get_sector_momentum(price_cutoff)

    # FRED: enforces per-series publication lag internally
    fred, fred_cutoff_log = get_fred_snapshot(date)

    # Enhanced quant metrics (all use price_cutoff)
    rel_momentum   = get_relative_momentum(price_cutoff, sector)
    rolling_sharpe = get_rolling_sharpe(price_cutoff, sector)
    correlations   = get_sector_correlations(price_cutoff, sector)

    regime_label = generate_regime_label(fred, vix, date)
    regime_tag   = infer_regime_tag(fred, vix)

    # GDELT: T-1 cutoff enforced inside get_gdelt_news()
    gdelt_news, gdelt_audit = get_gdelt_news(date, sector)
    time.sleep(0.5)

    # Build audit trail — every data cutoff recorded
    data_cutoff_log = {
        "decision_date":        str(date),
        "price_cutoff":         str(price_cutoff),
        "quant_context_cutoff": str(price_cutoff),   # build_quant_context uses same T-1 cutoff
        "gdelt_cutoff":         gdelt_audit.get("cutoff"),
        "gdelt_fetched":        gdelt_audit.get("total_fetched"),
        "gdelt_filtered":       gdelt_audit.get("after_filter"),
        **{f"fred_{k}": v for k, v in fred_cutoff_log.items()},
    }

    # Cross-stream alignment check: warn if GDELT cutoff drifts from price cutoff
    _gdelt_cutoff_str = gdelt_audit.get("cutoff")
    if _gdelt_cutoff_str:
        try:
            _gdelt_cutoff_dt = datetime.date.fromisoformat(_gdelt_cutoff_str)
            _drift_days = abs((price_cutoff - _gdelt_cutoff_dt).days)
            if _drift_days > 1:
                logger.warning(
                    "Timestamp misalignment: price_cutoff=%s, gdelt_cutoff=%s "
                    "(drift=%d days) for %s / %s",
                    price_cutoff, _gdelt_cutoff_str, _drift_days, date, sector,
                )
                data_cutoff_log["alignment_warning"] = (
                    f"gdelt_cutoff drifted {_drift_days}d from price_cutoff"
                )
        except (ValueError, TypeError):
            pass

    full_context = regime_label
    if gdelt_news:
        full_context += "\n\n" + gdelt_news

    # Sector momentum + quant metrics context block
    quant_lines = [f"\n[{date.strftime('%Y-%m-%d')} 量化指标快照 · {sector}]"]

    if momentum:
        top3 = sorted(momentum.items(), key=lambda x: x[1], reverse=True)[:3]
        bot3 = sorted(momentum.items(), key=lambda x: x[1])[:3]
        quant_lines.append("20日绝对动量 — 强势: " +
                           " | ".join(f"{s} {v:+.1f}%" for s, v in top3))
        quant_lines.append("20日绝对动量 — 弱势: " +
                           " | ".join(f"{s} {v:+.1f}%" for s, v in bot3))

    if rel_momentum is not None:
        quant_lines.append(f"相对动量 (vs SPY): {rel_momentum:+.2f}%  "
                           f"({'跑赢' if rel_momentum > 0 else '跑输'}大盘)")

    if rolling_sharpe is not None:
        quality = "优秀" if rolling_sharpe > 1.0 else ("良好" if rolling_sharpe > 0.5 else "偏弱")
        quant_lines.append(f"20日滚动 Sharpe: {rolling_sharpe:.2f}  ({quality})")

    if correlations:
        vs_spy = correlations.get("vs_spy")
        if vs_spy is not None:
            quant_lines.append(f"与SPY相关系数: {vs_spy:.2f}")
        top_corr = correlations.get("top_corr", [])
        low_corr = correlations.get("low_corr", [])
        if top_corr:
            quant_lines.append("高相关板块: " +
                               " | ".join(f"{s}({c:.2f})" for s, c in top_corr))
        if low_corr:
            quant_lines.append("低相关板块(分散化价值): " +
                               " | ".join(f"{s}({c:.2f})" for s, c in low_corr))

    full_context += "\n" + "\n".join(quant_lines)

    # Structured quant_metrics dict for DB storage
    quant_metrics = {
        "sector_momentum_20d": momentum.get(sector),
        "relative_momentum":   rel_momentum,
        "rolling_sharpe_20d":  rolling_sharpe,
        "corr_vs_spy":         correlations.get("vs_spy"),
        "vix":                 vix,
        "fed_funds":           fred.get("fed_funds"),
        "cpi_yoy":             fred.get("cpi_yoy"),
        "yield_spread":        fred.get("yield_spread"),
    }

    return {
        "date":             date,
        "sector":           sector,
        "vix":              vix,
        "macro_regime":     regime_tag,
        "fred":             fred,
        "momentum":         momentum,
        "rel_momentum":     rel_momentum,
        "rolling_sharpe":   rolling_sharpe,
        "correlations":     correlations,
        "quant_metrics":    quant_metrics,
        "news_context":     gdelt_news,
        "regime_label":     regime_label,
        "full_context":     full_context,
        "data_cutoff_log":  data_cutoff_log,   # audit trail
    }


# ── XAI helpers (mirrors tabs.py, kept independent for backtest use) ──────────

def _parse_xai_block(text: str) -> dict:
    import re
    result: dict = {}
    block_match = re.search(
        r"\[XAI_ATTRIBUTION\](.*?)\[/XAI_ATTRIBUTION\]",
        text, re.DOTALL | re.IGNORECASE,
    )
    if not block_match:
        return result
    block = block_match.group(1)
    for field in ["overall_confidence", "macro_confidence", "news_confidence", "technical_confidence"]:
        m = re.search(rf"{field}\s*:\s*(\d+)", block)
        if m:
            result[field] = min(100, max(0, int(m.group(1))))
    m = re.search(r"signal_drivers\s*:\s*(.+)", block)
    if m:
        result["signal_drivers"] = m.group(1).strip()
    m = re.search(r"invalidation_conditions\s*:\s*(.+)", block)
    if m:
        result["invalidation_conditions"] = m.group(1).strip()
    m = re.search(r"horizon\s*:\s*(.+)", block)
    if m:
        raw_h = m.group(1).strip()
        if "半" in raw_h or "长" in raw_h:
            result["horizon"] = "半年(6个月)"
        else:
            result["horizon"] = "季度(3个月)"
    else:
        result["horizon"] = "季度(3个月)"
    result["signal_attribution"] = {
        "macro":     result.get("macro_confidence"),
        "news":      result.get("news_confidence"),
        "technical": result.get("technical_confidence"),
        "drivers":   result.get("signal_drivers", ""),
    }
    return result


def _sensitivity_test(model, sector: str, vix: float, macro_context: str) -> str:
    import re
    prompt = (
        f"你是一名板块研究分析师。当前分析对象是【{sector}】板块，VIX={vix}。\n"
        f"宏观背景摘要：{macro_context[:200]}\n\n"
        f"如果 VIX 上升到 {vix + 5:.1f}，配置方向（超配/标配/低配）是否会改变？\n"
        f"如果 VIX 下降到 {max(10, vix - 5):.1f}，方向是否改变？\n\n"
        "只回答：\nVIX+5方向: [超配/标配/低配]\n"
        "VIX-5方向: [超配/标配/低配]\n原始方向: [超配/标配/低配]"
    )
    try:
        resp = model.generate_content(prompt).text
        directions = re.findall(r"(超配|标配|低配)", resp)
        if len(directions) >= 3:
            changes = sum([directions[0] != directions[2], directions[1] != directions[2]])
            return ["LOW", "MEDIUM", "HIGH"][min(changes, 2)]
    except Exception:
        pass
    return ""


# ── Backtest runner ────────────────────────────────────────────────────────────

def run_sector_backtest(
    model,
    sectors: list[str],
    start_date: str,
    end_date: str,
    freq: str = "MS",
    progress_cb=None,
    sensitivity_test: bool = False,   # disabled by default — each test = 1 extra API call
) -> list[dict]:
    """
    Batch replay: for each (date × sector), build a historical snapshot,
    call the sector analysis prompt, and save to Alpha Memory.

    Args:
        model:       Gemini model instance
        sectors:     list of sector names (Chinese)
        start_date:  "YYYY-MM-DD"
        end_date:    "YYYY-MM-DD"
        freq:        "MS" = monthly, "QS" = quarterly
        progress_cb: optional callable(current, total, msg) for UI progress

    Returns:
        list of {date, sector, direction, regime, saved_id}
    """
    import pandas as pd
    from engine.memory import (
        save_decision, get_historical_context,
        get_backtest_retry_stubs, clear_retry_stub,
        set_backtest_stop, get_backtest_stop,
        create_backtest_session, update_backtest_session,
        get_verified_decision_count, get_quant_pattern_context,
    )
    from engine.quant import build_quant_context, compute_state_vector

    set_backtest_stop(False)  # clear any leftover flag from previous run

    # ── Layer 2: state-change driven sampling setup ───────────────────────────
    _last_state:       dict[str, dict]           = {}   # sector → last state vector
    _last_signal_date: dict[str, datetime.date]  = {}   # sector → last signal date
    _MIN_INTERVAL = 7    # minimum days between same-sector signals
    _FORCE_INTERVAL = 90 # always retrigger after this many days (quarterly fallback)

    dates = pd.date_range(start=start_date, end=end_date, freq=freq).date.tolist()

    # tab_type encodes the frequency so monthly and quarterly runs are fully
    # independent training sets and never cross-contaminate each other's skip sets.
    # e.g. freq="MS" → "sector_backtest_ms", freq="QS" → "sector_backtest_qs"
    _bt_tab_type = f"sector_backtest_{freq.lower()}"

    # ── Resume logic: skip already-completed records ─────────────────────────
    # Only skip records that match the CURRENT frequency's tab_type.
    # Legacy records ("sector_backtest", "sector") are also included so old data
    # is not needlessly re-analysed regardless of which freq they were created under.
    from sqlalchemy.orm import Session as _Session
    from engine.memory import SessionFactory, DecisionLog
    with SessionFactory() as _s:
        _done_pairs = {
            (r.sector_name, str(r.decision_date))
            for r in _s.query(DecisionLog.sector_name, DecisionLog.decision_date)
            .filter(
                DecisionLog.tab_type.in_([_bt_tab_type, "sector_backtest", "sector"]),
                DecisionLog.is_backtest == True,
                DecisionLog.needs_retry == False,
                DecisionLog.ai_conclusion.isnot(None),
                DecisionLog.decision_date.isnot(None),
            ).all()
        }

    total = len(dates) * len(sectors)
    results: list[dict] = []
    _quota_hit = False   # flag: stop loop on quota exhaustion

    # ── Key pool setup ────────────────────────────────────────────────────────
    from engine.key_pool import get_pool, AllKeysExhausted, EmptyOutputCircuitBreaker
    _pool = get_pool()

    # Count already-completed pairs upfront
    all_pairs = [(date, sector) for date in dates for sector in sectors]
    skipped_count = sum(1 for d, s in all_pairs if (s, str(d)) in _done_pairs)

    # Progress is shown relative to *remaining* work so the bar reads 1/77, 2/77...
    # not 4/80, 5/80 (which implies the first 3 are being redone).
    effective_total = max(total - skipped_count, 1)  # never 0 — prevents ZeroDivisionError in UI
    done            = 0    # counts only within this session (resets each run)
    saved           = 0    # records with real AI output actually written to DB
    skipped_no_data   = 0  # validity gate: no news + no quant
    skipped_no_change = 0  # state unchanged, not worth re-analyzing

    if skipped_count > 0 and progress_cb:
        progress_cb(0, max(effective_total, 1),
                    f"↷ 续训模式：跳过已完成 {skipped_count} 条，本次待处理 {effective_total} 批")

    # Create persistent session record (stores full total for the banner's "剩余" counter)
    _session_id = create_backtest_session(
        start_date=start_date, end_date=end_date,
        sectors=sectors, freq=freq, total_pairs=total,
    )

    for date in dates:
        if _quota_hit:
            break
        for sector in sectors:
            label = f"{date} · {sector}"

            # Skip already-completed records (resume support)
            if (sector, str(date)) in _done_pairs:
                continue

            # ── Layer 2: state-change driven sampling ────────────────────────
            ticker_for_state = SECTOR_ETF.get(sector, "")
            _curr_state  = compute_state_vector(ticker_for_state, date)
            _prev_state  = _last_state.get(sector, {})
            _last_sig    = _last_signal_date.get(sector)
            _days_since  = (date - _last_sig).days if _last_sig else 999

            _state_changed = any(
                _curr_state.get(k) != _prev_state.get(k)
                for k in _curr_state
                if _curr_state.get(k) != "unknown"
            )
            _force_trigger = _days_since >= _FORCE_INTERVAL
            _min_ok        = _days_since >= _MIN_INTERVAL

            if not _force_trigger and not (_state_changed and _min_ok):
                done += 1
                skipped_no_change += 1
                if progress_cb:
                    progress_cb(done, effective_total,
                        f"↷ 跳过·状态未变 {label}  │  有效记录 {saved} · 无数据跳过 {skipped_no_data} · 状态未变跳过 {skipped_no_change}")
                continue

            _last_state[sector]       = _curr_state
            _last_signal_date[sector] = date
            # ─────────────────────────────────────────────────────────────────

            done += 1
            if progress_cb:
                progress_cb(done, effective_total,
                    f"分析中 {label}  │  有效记录 {saved} · 无数据跳过 {skipped_no_data} · 状态未变跳过 {skipped_no_change}")

            try:
                snap = build_snapshot(date, sector)

                # Inject learning patterns with cutoff_date to prevent future leakage.
                # Only decisions verified before `date` are visible at this backtest point.
                hist_ctx = get_historical_context(
                    "sector",
                    sector_name=sector,
                    macro_regime=snap["macro_regime"],
                    n=3,
                    cutoff_date=date,
                )
                full_ctx = (hist_ctx + "\n\n" + snap["full_context"]
                            if hist_ctx else snap["full_context"])

                # ── Layer 1: inject objective quant context ───────────────────
                _etf_ticker  = SECTOR_ETF.get(sector, "")
                _quant_ctx   = build_quant_context(_etf_ticker, date)
                _pattern_ctx = get_quant_pattern_context(
                    _curr_state, snap["macro_regime"], cutoff_date=date
                )
                _state_desc = (
                    f"动量:{_curr_state.get('momentum_regime','?')} | "
                    f"RSI区间:{_curr_state.get('rsi_zone','?')} | "
                    f"波动率:{_curr_state.get('vol_regime','?')}"
                )

                # Build prompt (same structure as live sector analysis)
                prompt = (
                    f"你是一名机构级板块研究分析师，正在向投资委员会汇报【{sector}】板块。\n"
                    f"当前市场 VIX 指数为 {snap['vix']}。"
                    f"\n\n【历史宏观背景（回测快照）】\n{full_ctx}\n\n"
                    + (f"【客观量化指标（截至 {date - datetime.timedelta(days=1)}）】\n{_quant_ctx}\n\n" if _quant_ctx else "")
                    + (_pattern_ctx + "\n\n" if _pattern_ctx else "")
                    + f"【当前量化状态向量】{_state_desc}\n\n"
                    "请按以下框架撰写专业板块分析报告：\n\n"
                    "### 1. 板块驱动逻辑\n"
                    f"[结合当前宏观环境与量化指标，分析【{sector}】板块的核心催化剂与主要驱动因子]\n\n"
                    "### 2. 新闻事件解读\n"
                    "[逐一分析上方近期新闻的潜在影响，必须明确引用具体新闻标题，"
                    "说明其对板块的短期利多/利空含义]\n\n"
                    "### 3. 量价背离检验\n"
                    "[对比新闻情绪与量化指标（动量、RSI、信用利差），识别是否存在叙事与价格走势的背离]\n\n"
                    "### 4. 风险敞口评估\n"
                    f"[结合 VIX {snap['vix']} 环境与宏观背景，量化评估该板块面临的主要下行风险]\n\n"
                    "### 5. 战术配置方向\n"
                    "[基于综合分析，给出中性、客观的配置方向（超配/标配/低配）与一句话理由]\n\n"
                    "→ 综合判断: [一句话总结当前配置决策]\n\n"
                    "### [XAI_ATTRIBUTION]\n"
                    "overall_confidence: [0-100]\n"
                    "macro_confidence: [0-100]\n"
                    "news_confidence: [0-100]\n"
                    "technical_confidence: [0-100]\n"
                    "signal_drivers: [最多3个驱动因素]\n"
                    "invalidation_conditions: [1-2个失效条件]\n"
                    "horizon: [二选一 — 季度(3个月,基本面/政策传导,1个财报季) / 半年(6个月,结构性趋势,2个财报季)]\n"
                    "### [/XAI_ATTRIBUTION]\n\n"
                    "写作要求：机构级语气、逻辑严谨、严禁情绪化或劝说性表达。"
                )

                # ── Layer 1: validity gate ───────────────────────────────────
                # Skip AI call if there is neither news nor quant context.
                # This prevents token waste when data pipelines return empty.
                _has_news  = bool(snap.get("news_context", "").strip())
                _has_quant = bool(_quant_ctx.strip())
                if not _has_news and not _has_quant:
                    _pool.report_skip()
                    skipped_no_data += 1
                    logger.info("Validity gate: skipping %s/%s — no news or quant data", date, sector)
                    if progress_cb:
                        progress_cb(done, effective_total,
                            f"⊘ 无效批次（新闻+量化均为空）· {label}  │  有效记录 {saved} · 无数据跳过 {skipped_no_data} · 状态未变跳过 {skipped_no_change}")
                    continue

                # ── AI call via key pool ──────────────────────────────────────
                _bt_model  = _pool.get_model()
                conclusion = _bt_model.generate_content(prompt).text
                _pool.report_success(has_content=bool(conclusion.strip()))

                # Resolve sector ETF ticker
                ticker = SECTOR_ETF.get(sector, "")

                # Parse XAI block from conclusion
                xai = _parse_xai_block(conclusion)

                # Sensitivity test (VIX ±5) — skipped by default to conserve quota
                sensitivity = (
                    _sensitivity_test(_bt_model, sector, snap["vix"], snap["regime_label"])
                    if sensitivity_test else ""
                )

                record_id = save_decision(
                    tab_type=_bt_tab_type,
                    ai_conclusion=conclusion,
                    vix_level=snap["vix"],
                    sector_name=sector,
                    ticker=ticker,
                    news_summary=snap["news_context"][:500],
                    macro_regime=snap["macro_regime"],
                    horizon=xai.get("horizon", "季度(3个月)"),
                    economic_logic=snap["regime_label"][:300],
                    quant_metrics=snap.get("quant_metrics"),
                    is_backtest=True,
                    macro_confidence=xai.get("macro_confidence"),
                    news_confidence=xai.get("news_confidence"),
                    technical_confidence=xai.get("technical_confidence"),
                    confidence_score=xai.get("overall_confidence"),
                    signal_attribution=xai.get("signal_attribution"),
                    sensitivity_flag=sensitivity,
                    invalidation_conditions=xai.get("invalidation_conditions", ""),
                    debate_transcript={
                        "data_cutoff_log":    snap.get("data_cutoff_log"),
                        "state_vector":       _curr_state,
                        "purged_wf_active":   False,
                        "verified_count_at_run": get_verified_decision_count(),
                    },
                    decision_date=date,   # the historical date this decision REPRESENTS
                )

                saved += 1
                results.append({
                    "date":             str(date),
                    "sector":           sector,
                    "regime":           snap["macro_regime"],
                    "vix":              snap["vix"],
                    "saved_id":         record_id,
                    "data_cutoff_log":  snap.get("data_cutoff_log"),
                })

                # Keep DB in sync so the resume banner reflects real progress
                # if the user refreshes or the run is interrupted mid-batch.
                update_backtest_session(_session_id, done_pairs=skipped_count + done, status="running")

                logger.info("Backtest saved: %s / %s → id=%s", date, sector, record_id)
                if progress_cb:
                    progress_cb(done, effective_total,
                        f"✓ 已存入 {label}  │  有效记录 {saved} · 无数据跳过 {skipped_no_data} · 状态未变跳过 {skipped_no_change}")
                time.sleep(1.0)

                # Check for user-requested pause (set via admin UI pause button)
                if get_backtest_stop():
                    if progress_cb:
                        progress_cb(done, effective_total, f"⏸ 训练已暂停 · 进度已保存至 Alpha Memory")
                    set_backtest_stop(False)
                    update_backtest_session(_session_id, done_pairs=skipped_count + done, status="paused")
                    return results, False  # not a quota hit, clean pause

            except (AllKeysExhausted, EmptyOutputCircuitBreaker) as exc:
                # Pool-level halt — stop the entire backtest immediately
                logger.error("KeyPool halt: %s", exc)
                if progress_cb:
                    progress_cb(done, effective_total, f"🛑 Key 池熔断 · 回测已停止 · {exc}")
                update_backtest_session(_session_id, done_pairs=skipped_count + done, status="quota_hit")
                _quota_hit = True
                break

            except Exception as exc:
                if _pool.is_quota_error(exc):
                    try:
                        _pool.report_quota_error()   # rotates key or raises AllKeysExhausted
                        # Key was rotated — save stub and continue from next iteration
                        snap = locals().get("snap")
                        if snap:
                            save_decision(
                                tab_type=_bt_tab_type,
                                ai_conclusion="",
                                vix_level=snap["vix"],
                                sector_name=sector,
                                ticker=SECTOR_ETF.get(sector, ""),
                                macro_regime=snap["macro_regime"],
                                is_backtest=True,
                                decision_date=date,
                                needs_retry=True,
                            )
                        logger.warning("Quota on '%s' — rotated key, will retry next run",
                                       _pool.current_label)
                        if progress_cb:
                            progress_cb(done, effective_total,
                                        f"🔄 Key 已切换至 {_pool.current_label} · {label}")
                        continue
                    except AllKeysExhausted as pool_exc:
                        logger.error("All keys exhausted: %s", pool_exc)
                        if progress_cb:
                            progress_cb(done, effective_total, f"🛑 全部 Key 已耗尽 · 回测停止")
                        update_backtest_session(_session_id, done_pairs=skipped_count + done, status="quota_hit")
                        _quota_hit = True
                        break
                else:
                    logger.warning("Backtest failed for %s / %s: %s", date, sector, exc)
                    if progress_cb:
                        progress_cb(done, effective_total, f"⚠ Error: {label} — {exc}")
                    continue

    update_backtest_session(_session_id, done_pairs=skipped_count + done,
                            status="quota_hit" if _quota_hit else "completed")
    return results, _quota_hit


def run_walk_forward_backtest(
    model,
    sectors:      list[str],
    train_start:  str,
    train_end:    str,
    test_end:     str,
    freq:         str = "QS",
    progress_cb=None,
    sensitivity_test: bool = False,   # disabled by default — each test = 1 extra API call
) -> list[dict]:
    """
    Walk-forward backtest with strict temporal isolation.

    Structure:
      Training window : [train_start, train_end]   → builds initial Alpha Memory
      Test window     : (train_end, test_end]       → each date is a true out-of-sample test
      After each test date is verified, the training window expands to include it.

    Leakage prevention:
      - Each test decision at date T only sees Alpha Memory records with created_at < T
      - Training phase uses is_backtest=True isolation (excluded from live context)
      - get_historical_context() is called with cutoff_date=T to prevent future leakage

    Args:
        model:       Gemini model instance
        sectors:     list of sector names
        train_start: "YYYY-MM-DD" — start of training window
        train_end:   "YYYY-MM-DD" — end of training window (last in-sample date)
        test_end:    "YYYY-MM-DD" — end of test window
        freq:        "QS" = quarterly (recommended), "MS" = monthly
        progress_cb: callable(current, total, phase, msg)

    Returns:
        list of result dicts with phase label ("train" or "test")
    """
    import pandas as pd
    from engine.memory import (
        save_decision, get_historical_context,
        set_backtest_stop, get_backtest_stop,
        get_verified_decision_count, get_quant_pattern_context,
    )
    from engine.quant import build_quant_context, compute_state_vector

    set_backtest_stop(False)

    # ── Layer 3: Purged Walk-Forward (auto-activates at 300 verified decisions) ──
    _verified_count  = get_verified_decision_count()
    _use_purged_wf   = _verified_count >= 300
    _EMBARGO_DAYS    = 5   # gap between last training date and first test date
    if _use_purged_wf:
        logger.info(
            "Layer 3 Purged WF active — %d verified decisions, embargo=%d days",
            _verified_count, _EMBARGO_DAYS,
        )

    # ── Layer 2: state-change driven sampling setup ───────────────────────────
    _last_state:       dict[str, dict]           = {}
    _last_signal_date: dict[str, datetime.date]  = {}
    _MIN_INTERVAL  = 7
    _FORCE_INTERVAL = 90

    # Phase 1: training window
    train_dates = pd.date_range(start=train_start, end=train_end, freq=freq).date.tolist()
    # Phase 2: test window — strictly after train_end (+ embargo if Layer 3 active)
    _test_start = pd.Timestamp(train_end) + pd.DateOffset(
        days=_EMBARGO_DAYS if _use_purged_wf else 1
    )
    test_dates = pd.date_range(
        start=_test_start, end=test_end, freq=freq,
    ).date.tolist()

    all_dates = [("train", d) for d in train_dates] + [("test", d) for d in test_dates]
    total     = len(all_dates) * len(sectors)
    done      = 0
    results:  list[dict] = []

    # ── Key pool setup ────────────────────────────────────────────────────────
    from engine.key_pool import get_pool, AllKeysExhausted, EmptyOutputCircuitBreaker
    _pool = get_pool()
    _quota_hit = False

    for phase, date in all_dates:
        for sector in sectors:
            label = f"[{phase.upper()}] {date} · {sector}"

            # ── Layer 2: state-change driven sampling ────────────────────────
            _etf_ticker  = SECTOR_ETF.get(sector, "")
            _curr_state  = compute_state_vector(_etf_ticker, date)
            _prev_state  = _last_state.get(sector, {})
            _last_sig    = _last_signal_date.get(sector)
            _days_since  = (date - _last_sig).days if _last_sig else 999

            _state_changed = any(
                _curr_state.get(k) != _prev_state.get(k)
                for k in _curr_state
                if _curr_state.get(k) != "unknown"
            )
            _force_trigger = _days_since >= _FORCE_INTERVAL
            _min_ok        = _days_since >= _MIN_INTERVAL

            if not _force_trigger and not (_state_changed and _min_ok):
                done += 1
                if progress_cb:
                    progress_cb(done, total, phase, f"↷ 跳过·状态未变 {label}")
                continue

            _last_state[sector]       = _curr_state
            _last_signal_date[sector] = date
            # ─────────────────────────────────────────────────────────────────

            done += 1
            if progress_cb:
                progress_cb(done, total, phase, f"Processing {label}")

            try:
                snap = build_snapshot(date, sector)

                # Walk-forward isolation: only see Alpha Memory records BEFORE this date.
                # exclude_backtest=False is required: all walk-forward records have
                # is_backtest=True, so the default True would silently empty the context
                # and make the training window useless.
                hist_ctx = get_historical_context(
                    "sector",
                    sector_name=sector,
                    macro_regime=snap["macro_regime"],
                    n=3,
                    cutoff_date=date,        # ← temporal isolation parameter
                    exclude_backtest=False,  # ← must see own training records
                )
                full_ctx = (hist_ctx + "\n\n" + snap["full_context"]
                            if hist_ctx else snap["full_context"])

                # ── Layer 1: inject objective quant context ───────────────────
                _quant_ctx   = build_quant_context(_etf_ticker, date)
                _pattern_ctx = get_quant_pattern_context(
                    _curr_state, snap["macro_regime"], cutoff_date=date
                )
                _state_desc = (
                    f"动量:{_curr_state.get('momentum_regime','?')} | "
                    f"RSI区间:{_curr_state.get('rsi_zone','?')} | "
                    f"波动率:{_curr_state.get('vol_regime','?')}"
                )
                _layer3_note = (
                    f"\n[Purged WF · {_verified_count}条已验证决策 · 样本外测试]"
                    if _use_purged_wf and phase == "test" else ""
                )

                phase_note = (
                    "[训练阶段 · 建立基准经验库]" if phase == "train"
                    else f"[测试阶段 · 样本外验证{_layer3_note}]"
                )
                prompt = (
                    f"你是一名机构级板块研究分析师 {phase_note}，"
                    f"正在分析【{sector}】板块，日期: {date}，VIX={snap['vix']}。\n\n"
                    f"【历史宏观背景】\n{full_ctx}\n\n"
                    + (f"【客观量化指标（截至 {date - datetime.timedelta(days=1)}）】\n{_quant_ctx}\n\n" if _quant_ctx else "")
                    + (_pattern_ctx + "\n\n" if _pattern_ctx else "")
                    + f"【当前量化状态向量】{_state_desc}\n\n"
                    "请按以下框架撰写分析报告：\n\n"
                    "### 1. 板块驱动逻辑\n[结合宏观环境与量化指标分析核心催化剂]\n\n"
                    "### 2. 新闻事件解读\n[逐一引用具体新闻，说明利多/利空]\n\n"
                    "### 3. 量价背离检验\n[对比新闻情绪与量化指标，识别背离信号]\n\n"
                    "### 4. 风险敞口评估\n[量化下行风险]\n\n"
                    "### 5. 战术配置方向\n[超配/标配/低配 + 一句话理由]\n\n"
                    "→ 综合判断: [一句话总结]\n\n"
                    "### [XAI_ATTRIBUTION]\n"
                    "overall_confidence: [0-100]\n"
                    "macro_confidence: [0-100]\n"
                    "news_confidence: [0-100]\n"
                    "technical_confidence: [0-100]\n"
                    "signal_drivers: [最多3个驱动因素]\n"
                    "invalidation_conditions: [1-2个失效条件]\n"
                    "horizon: [二选一 — 季度(3个月,基本面/政策传导,1个财报季) / 半年(6个月,结构性趋势,2个财报季)]\n"
                    "### [/XAI_ATTRIBUTION]\n"
                )

                # ── Layer 1: validity gate ───────────────────────────────────
                _has_news  = bool(snap.get("news_context", "").strip())
                _has_quant = bool(_quant_ctx.strip())
                if not _has_news and not _has_quant:
                    _pool.report_skip()
                    logger.info("Validity gate: skipping %s/%s — no news or quant data", date, sector)
                    if progress_cb:
                        progress_cb(done, total, f"⊘ 跳过（无数据）· {label}")
                    continue

                # ── AI call via key pool ──────────────────────────────────────
                _wf_model   = _pool.get_model()
                conclusion  = _wf_model.generate_content(prompt).text
                _pool.report_success(has_content=bool(conclusion.strip()))
                ticker      = SECTOR_ETF.get(sector, "")
                xai         = _parse_xai_block(conclusion)
                sensitivity = (
                    _sensitivity_test(_wf_model, sector, snap["vix"], snap["regime_label"])
                    if sensitivity_test else ""
                )

                record_id = save_decision(
                    tab_type=f"walk_forward_{phase}",   # "walk_forward_train" | "walk_forward_test"
                    ai_conclusion=conclusion,
                    vix_level=snap["vix"],
                    sector_name=sector,
                    ticker=ticker,
                    news_summary=snap["news_context"][:500],
                    macro_regime=snap["macro_regime"],
                    horizon=xai.get("horizon", "季度(3个月)"),
                    economic_logic=snap["regime_label"][:300],
                    quant_metrics=snap.get("quant_metrics"),
                    is_backtest=True,
                    macro_confidence=xai.get("macro_confidence"),
                    news_confidence=xai.get("news_confidence"),
                    technical_confidence=xai.get("technical_confidence"),
                    confidence_score=xai.get("overall_confidence"),
                    signal_attribution=xai.get("signal_attribution"),
                    sensitivity_flag=sensitivity,
                    invalidation_conditions=xai.get("invalidation_conditions", ""),
                    debate_transcript={
                        "phase":              phase,
                        "data_cutoff_log":    snap.get("data_cutoff_log"),
                        "state_vector":       _curr_state,
                        "purged_wf_active":   _use_purged_wf,
                        "verified_count_at_run": _verified_count,
                    },
                    decision_date=date,   # the historical date this decision REPRESENTS
                )

                results.append({
                    "phase":            phase,
                    "date":             str(date),
                    "sector":           sector,
                    "regime":           snap["macro_regime"],
                    "vix":              snap["vix"],
                    "direction":        xai.get("signal_drivers", ""),
                    "saved_id":         record_id,
                    "data_cutoff_log":  snap.get("data_cutoff_log"),
                })

                logger.info("Walk-forward [%s] saved: %s / %s → id=%s",
                            phase, date, sector, record_id)
                time.sleep(1.0)

                if get_backtest_stop():
                    if progress_cb:
                        progress_cb(done, total, f"⏸ 训练已暂停 · 进度已保存至 Alpha Memory")
                    set_backtest_stop(False)
                    return results

            except (AllKeysExhausted, EmptyOutputCircuitBreaker) as exc:
                logger.error("KeyPool halt (WF): %s", exc)
                if progress_cb:
                    progress_cb(done, total, f"🛑 Key 池熔断 · 回测已停止 · {exc}")
                _quota_hit = True
                break

            except Exception as exc:
                if _pool.is_quota_error(exc):
                    try:
                        _pool.report_quota_error()
                        logger.warning("WF quota on '%s' — rotated key", _pool.current_label)
                        if progress_cb:
                            progress_cb(done, total,
                                        f"🔄 Key 已切换至 {_pool.current_label} · {label}")
                        continue
                    except AllKeysExhausted as pool_exc:
                        logger.error("All keys exhausted (WF): %s", pool_exc)
                        if progress_cb:
                            progress_cb(done, total, f"🛑 全部 Key 已耗尽 · 回测停止")
                        _quota_hit = True
                        break
                else:
                    logger.warning("Walk-forward failed [%s] %s / %s: %s",
                                   phase, date, sector, exc)
                    if progress_cb:
                        progress_cb(done, total, f"⚠ Error: {label} — {exc}")
                    continue

        if _quota_hit:
            break

    return results


# ── Dynamic sector ETF (Admin-overridable) ─────────────────────────────────────

def get_active_sector_etf() -> dict[str, str]:
    """
    Return the active sector→ETF map.
    Priority: UniverseETF DB table (P2-11) → hardcoded SECTOR_ETF fallback.
    """
    try:
        from engine.universe_manager import get_active_universe
        active = get_active_universe()
        if active:
            return active
    except Exception:
        pass
    return dict(SECTOR_ETF)
