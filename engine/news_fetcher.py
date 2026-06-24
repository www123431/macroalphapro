"""
engine/news_fetcher.py — P2-17 三层新闻数据源
================================================
Layer 1: Finnhub — 结构化情绪分数（免费层 60次/分钟）
Layer 2: GNews API — 关键词精准检索（免费层 100次/天）
Layer 3: yfinance.news — 零 API Key 终极备用

P0-4 合规：sentiment_score 属于原始数值，可注入 prompt；
禁止在本模块内生成任何方向性结论文本。
"""
from __future__ import annotations

import datetime
import logging
import math
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SOURCE_TIER_WEIGHT = {1: 1.0, 2: 0.7, 3: 0.4}
_DECAY_HALFLIFE_DAYS = 1.5
_DECAY_LAMBDA = math.log(2) / _DECAY_HALFLIFE_DAYS


# ── 数据结构 ────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    title:           str
    summary:         str
    published_at:    datetime.datetime
    source:          str         # "finnhub" / "gnews" / "yfinance"
    source_tier:     int         # 1 / 2 / 3
    sentiment_score: float | None = None  # -1.0 到 +1.0（Finnhub VADER，非 LLM 级别）
    url:             str = ""
    relevance_score: float = 1.0


# ── API Key 获取 ────────────────────────────────────────────────────────────────

def _get_secret(name: str) -> str:
    """从 st.secrets 或环境变量获取 key，不存在则返回空字符串。"""
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ── Layer 1：Finnhub ─────────────────────────────────────────────────────────────

def fetch_finnhub_news(ticker: str, days: int = 3) -> list[NewsItem]:
    """
    Finnhub /company-news 端点，含内置情绪分数。
    免费层：60次/分钟。API Key：st.secrets["FINNHUB_KEY"]。
    注意：情绪基于 VADER/TextBlob，为辅助输入而非独立决策依据。
    """
    api_key = _get_secret("FINNHUB_KEY")
    if not api_key:
        return []

    import requests
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": str(start), "to": str(end), "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        items = []
        for art in (resp.json() or [])[:10]:
            pub_ts = art.get("datetime", 0)
            try:
                pub_dt = datetime.datetime.fromtimestamp(pub_ts) if pub_ts else datetime.datetime.utcnow()
            except Exception:
                pub_dt = datetime.datetime.utcnow()
            sentiment = art.get("sentiment", {})
            sent_score = sentiment.get("companyNewsScore") if isinstance(sentiment, dict) else None
            items.append(NewsItem(
                title=art.get("headline", ""),
                summary=art.get("summary", "")[:500],
                published_at=pub_dt,
                source="finnhub",
                source_tier=1,
                sentiment_score=float(sent_score) if sent_score is not None else None,
                url=art.get("url", ""),
            ))
        return items
    except Exception as exc:
        logger.warning("Finnhub news error for %s: %s", ticker, exc)
        return []


# ── Layer 2：GNews ───────────────────────────────────────────────────────────────

def fetch_gnews(query: str, days: int = 3, max_items: int = 5) -> list[NewsItem]:
    """
    GNews API 关键词检索。免费层：100次/天。API Key：st.secrets["GNEWS_KEY"]。
    """
    api_key = _get_secret("GNEWS_KEY")
    if not api_key:
        return []

    import requests
    from_dt = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    try:
        resp = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": max_items,
                    "from": from_dt, "apikey": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        items = []
        for art in resp.json().get("articles", []):
            pub_str = art.get("publishedAt", "")
            try:
                pub_dt = datetime.datetime.fromisoformat(
                    pub_str.replace("Z", "+00:00")
                ).replace(tzinfo=None)
            except Exception:
                pub_dt = datetime.datetime.utcnow()
            items.append(NewsItem(
                title=art.get("title", ""),
                summary=art.get("description", "")[:500],
                published_at=pub_dt,
                source="gnews",
                source_tier=2,
                url=art.get("url", ""),
            ))
        return items
    except Exception as exc:
        logger.warning("GNews error for %s: %s", query, exc)
        return []


# ── Layer 3：yfinance.news（终极备用）────────────────────────────────────────────

def fetch_yfinance_news(ticker: str, max_items: int = 5) -> list[NewsItem]:
    """零 API Key 备用。yfinance API 结构在版本间有变化，做容错处理。"""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).news or []
        items = []
        for art in raw[:max_items]:
            content = art.get("content", {})
            pub_str = content.get("pubDate") or art.get("providerPublishTime", "")
            try:
                pub_dt = (
                    datetime.datetime.fromisoformat(pub_str.replace("Z", ""))
                    if isinstance(pub_str, str) and pub_str
                    else datetime.datetime.fromtimestamp(pub_str)
                    if isinstance(pub_str, (int, float)) and pub_str
                    else datetime.datetime.utcnow()
                )
            except Exception:
                pub_dt = datetime.datetime.utcnow()
            items.append(NewsItem(
                title=content.get("title", art.get("title", "")),
                summary=content.get("summary", "")[:500],
                published_at=pub_dt,
                source="yfinance",
                source_tier=3,
            ))
        return items
    except Exception as exc:
        logger.warning("yfinance news error for %s: %s", ticker, exc)
        return []


# ── 三层编排 ────────────────────────────────────────────────────────────────────

def fetch_sector_news(
    sector: str,
    ticker: str,
    days: int = 3,
    max_total: int = 8,
) -> list[NewsItem]:
    """
    三层降级：Finnhub → GNews → yfinance。
    返回去重、按发布时间降序排列的新闻列表。
    """
    items: list[NewsItem] = []

    # Layer 1
    items.extend(fetch_finnhub_news(ticker, days=days))

    # Layer 2（Layer 1 不足时补充）
    if len(items) < 3:
        query = f"{sector} ETF {ticker} market"
        items.extend(fetch_gnews(query, days=days, max_items=5))

    # Layer 3（前两层都失败时）
    if not items:
        items.extend(fetch_yfinance_news(ticker, max_items=max_total))

    # 去重（title 前 30 字符）
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in items:
        key = it.title[:30].strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(it)

    unique.sort(key=lambda x: x.published_at, reverse=True)
    return unique[:max_total]


# ── 时效性加权摘要 ──────────────────────────────────────────────────────────────

def build_weighted_news_summary(
    items: list[NewsItem],
    max_chars: int = 1200,
    decay_halflife_days: float = 1.5,
) -> str:
    """
    将 NewsItem 列表合并为带时效权重的摘要字符串，供 prompt 注入。

    时效衰减：weight = exp(-ln(2) / halflife × days_old)
    来源权重：tier 1=1.0 / tier 2=0.7 / tier 3=0.4
    情绪标注：当 sentiment_score 不为 None 时附加 [情绪: ±x.xx]（原始数值，P0-4 合规）

    禁止：在此函数内添加"建议买入/超配/TSMOM 信号"等方向性文字。
    """
    decay_lambda = math.log(2) / max(decay_halflife_days, 0.1)
    now          = datetime.datetime.utcnow()

    scored: list[tuple[float, NewsItem]] = []
    for item in items:
        days_old = max(0.0, (now - item.published_at).total_seconds() / 86400)
        time_w   = math.exp(-decay_lambda * days_old)
        tier_w   = _SOURCE_TIER_WEIGHT.get(item.source_tier, 0.4)
        score    = time_w * tier_w * (item.relevance_score or 1.0)
        scored.append((score, item))

    scored.sort(reverse=True)

    lines: list[str] = []
    total = 0
    for score, item in scored:
        sent_str = (f" [情绪: {item.sentiment_score:+.2f}]"
                    if item.sentiment_score is not None else "")
        age_h = (now - item.published_at).total_seconds() / 3600
        line  = (
            f"[{item.source.upper()} {age_h:.0f}h前] "
            f"{item.title}{sent_str}\n{item.summary}"
        )
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n\n".join(lines) if lines else "（无可用新闻）"
