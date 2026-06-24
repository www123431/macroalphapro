"""
engine/news_fetchers.py — D4.3 of S6 anomaly_screener (2026-05-05)

Pre-registration: docs/decisions/s6_anomaly_screener_spec_2026-05-05.md §3 News Sources

Four news fetchers + a filter pipeline. Each fetcher returns a normalized
list of NewsItem dicts. Sources without API keys gracefully skip.

Tier 1 (required, no key needed OR key configured):
  • SEC EDGAR 8-K filings        — official material events; no key
  • yfinance news                — per-ticker headlines; no key
  • Alpha Vantage NEWS_SENTIMENT — company news + sentiment; AV_KEY in secrets

Tier 2 (optional, configured key):
  • GNews                        — general headlines fallback; GNEWS_KEY in secrets

Tier 3 (deferred future work):
  • Finnhub, FRED, NewsAPI       — not configured in this project

Filter pipeline:
  1. Per-source pull (rate-limited, lenient on failure)
  2. Keyword filter to portfolio tickers / sector keywords
  3. Dedup by URL hash
  4. Sort by publish_date desc, truncate to max_items=50
  5. Return combined NewsItem list
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

# ── User-agent for SEC EDGAR (required by SEC fair-access policy) ─────────────
SEC_USER_AGENT = "Macro Alpha Pro Research ${USER_EMAIL}"

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 2
DEFAULT_MAX_ITEMS     = 50


@dataclass
class NewsItem:
    """Normalized news item across all sources."""
    source:        str               # "sec_edgar_8k" | "yfinance" | "alpha_vantage" | "gnews"
    url:           str
    title:         str
    summary:       str               # may be empty if source doesn't provide
    publish_date:  datetime.date
    tickers:       list[str]         # which portfolio tickers this is about (post-filter)
    sentiment:     float | None = None  # -1 to +1 if source provides; else None
    raw:           dict | None = None  # original payload (for audit)

    def url_hash(self) -> str:
        return hashlib.sha256(self.url.encode("utf-8", errors="ignore")).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["publish_date"] = str(d["publish_date"])
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Secrets helper (graceful fallback when key missing)
# ─────────────────────────────────────────────────────────────────────────────

def _get_secret(name: str) -> str | None:
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        # outside Streamlit runtime; try env var
        import os
        return os.environ.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SEC EDGAR 8-K (no key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sec_edgar_8k(
    tickers: Iterable[str],
    days_back: int = DEFAULT_LOOKBACK_DAYS,
) -> list[NewsItem]:
    """
    Fetch recent 8-K filings for the given tickers from SEC EDGAR.

    8-K is the "current report" form filed within 4 business days of any
    material event (M&A, exec changes, financial restatements, defaults, etc).
    Reference: SEC Form 8-K (https://www.sec.gov/forms#sec_8k).
    """
    out: list[NewsItem] = []
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back + 1)
    for ticker in tickers:
        try:
            cik = _ticker_to_cik(ticker)
        except Exception as exc:
            logger.debug("sec_edgar: ticker→CIK failed for %s: %s", ticker, exc)
            continue
        if not cik:
            continue
        try:
            url = (
                f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
            )
            req = Request(url, headers={"User-Agent": SEC_USER_AGENT})
            with urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except (URLError, HTTPError) as exc:
            logger.debug("sec_edgar: fetch fail %s: %s", ticker, exc)
            continue
        recent = (payload.get("filings", {}).get("recent", {}) or {})
        forms = recent.get("form", []) or []
        dates = recent.get("filingDate", []) or []
        primary_doc = recent.get("primaryDocument", []) or []
        accession = recent.get("accessionNumber", []) or []
        for i, form in enumerate(forms):
            if form != "8-K":
                continue
            try:
                fdate = datetime.date.fromisoformat(dates[i])
            except Exception:
                continue
            if fdate < cutoff:
                break
            acc_clean = accession[i].replace("-", "")
            url_doc = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{acc_clean}/{primary_doc[i]}"
            )
            out.append(NewsItem(
                source="sec_edgar_8k",
                url=url_doc,
                title=f"{ticker} 8-K filing — {fdate}",
                summary=f"Material event disclosure. SEC EDGAR accession {accession[i]}",
                publish_date=fdate,
                tickers=[ticker],
                sentiment=None,
                raw={"form": form, "accession": accession[i]},
            ))
        time.sleep(0.12)   # SEC fair access ~ 10 req/s
    return out


# Tiny CIK lookup with on-disk cache
_CIK_CACHE: dict[str, str] = {}


def _ticker_to_cik(ticker: str) -> str | None:
    if not ticker:
        return None
    t = ticker.upper().strip()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]
    # SEC official ticker → CIK index
    try:
        req = Request("https://www.sec.gov/files/company_tickers.json",
                      headers={"User-Agent": SEC_USER_AGENT})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for v in data.values():
            if v.get("ticker", "").upper() == t:
                _CIK_CACHE[t] = str(v["cik_str"])
                return _CIK_CACHE[t]
    except Exception:
        pass
    _CIK_CACHE[t] = ""
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. yfinance news (no key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yfinance_news(
    tickers: Iterable[str],
    days_back: int = DEFAULT_LOOKBACK_DAYS,
) -> list[NewsItem]:
    """Headlines per ticker via yfinance.Ticker.news (best-effort)."""
    try:
        import yfinance as yf
    except Exception:
        return []
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back + 1)
    out: list[NewsItem] = []
    for t in tickers:
        try:
            news = yf.Ticker(t).news or []
        except Exception as exc:
            logger.debug("yfinance: news fail %s: %s", t, exc)
            continue
        for n in news:
            try:
                content = n.get("content") or n
                title = content.get("title") or ""
                provider = (content.get("provider") or {}).get("displayName") or ""
                pub_iso = (content.get("pubDate")
                           or content.get("displayTime")
                           or content.get("providerPublishTime") or "")
                if isinstance(pub_iso, (int, float)):
                    pdate = datetime.date.fromtimestamp(int(pub_iso))
                else:
                    pdate = datetime.date.fromisoformat(str(pub_iso)[:10])
                if pdate < cutoff:
                    continue
                clickurl = ((content.get("canonicalUrl") or {}).get("url")
                            or content.get("link") or "")
                if not clickurl or not title:
                    continue
                out.append(NewsItem(
                    source="yfinance",
                    url=clickurl,
                    title=title,
                    summary=content.get("summary", "") or "",
                    publish_date=pdate,
                    tickers=[t],
                    sentiment=None,
                    raw={"provider": provider},
                ))
            except Exception:
                continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Alpha Vantage NEWS_SENTIMENT (key)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_alpha_vantage_news(
    tickers: Iterable[str],
    days_back: int = DEFAULT_LOOKBACK_DAYS,
) -> list[NewsItem]:
    """
    Alpha Vantage NEWS_SENTIMENT — pre-computed sentiment + ticker tagging.

    Free tier: 5 requests/min, 25 requests/day. We batch tickers (≤50 per call).
    """
    key = _get_secret("AV_KEY")
    if not key:
        logger.info("alpha_vantage: AV_KEY missing, skipping")
        return []
    tickers_list = [t for t in tickers if t]
    if not tickers_list:
        return []

    cutoff = datetime.date.today() - datetime.timedelta(days=days_back + 1)
    cutoff_str = cutoff.strftime("%Y%m%dT%H%M")
    qs = urlencode({
        "function":  "NEWS_SENTIMENT",
        "tickers":   ",".join(tickers_list[:50]),
        "time_from": cutoff_str,
        "limit":     200,
        "sort":      "LATEST",
        "apikey":    key,
    })
    url = f"https://www.alphavantage.co/query?{qs}"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except (URLError, HTTPError) as exc:
        logger.debug("alpha_vantage: HTTP fail: %s", exc)
        return []
    feed = payload.get("feed", []) or []
    out: list[NewsItem] = []
    for item in feed:
        try:
            tp = item.get("time_published", "")
            pdate = datetime.date(int(tp[:4]), int(tp[4:6]), int(tp[6:8]))
            if pdate < cutoff:
                continue
            ticker_tags = [
                ts.get("ticker") for ts in item.get("ticker_sentiment", [])
                if ts.get("ticker") in tickers_list
            ]
            if not ticker_tags:
                continue
            sent = float(item.get("overall_sentiment_score", 0))
            out.append(NewsItem(
                source="alpha_vantage",
                url=item.get("url", ""),
                title=item.get("title", ""),
                summary=item.get("summary", "")[:500],
                publish_date=pdate,
                tickers=ticker_tags,
                sentiment=sent,
                raw={"label": item.get("overall_sentiment_label")},
            ))
        except Exception:
            continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. GNews (key, optional Tier 2)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gnews(
    keywords: Iterable[str],
    days_back: int = DEFAULT_LOOKBACK_DAYS,
    max_per_query: int = 10,
) -> list[NewsItem]:
    """
    GNews general headlines. Free tier 100/day. We query by sector/topic
    keywords (e.g. "energy oil", "technology AI") rather than ticker symbols.
    """
    key = _get_secret("GNEWS_KEY")
    if not key:
        logger.info("gnews: GNEWS_KEY missing, skipping")
        return []
    cutoff = datetime.date.today() - datetime.timedelta(days=days_back + 1)
    cutoff_iso = cutoff.isoformat() + "T00:00:00Z"
    out: list[NewsItem] = []
    for kw in list(keywords)[:6]:   # cap query count to preserve daily quota
        qs = urlencode({
            "q":         kw,
            "from":      cutoff_iso,
            "lang":      "en",
            "max":       max_per_query,
            "sortby":    "publishedAt",
            "apikey":    key,
        })
        url = f"https://gnews.io/api/v4/search?{qs}"
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        except (URLError, HTTPError) as exc:
            logger.debug("gnews: query %s failed: %s", kw, exc)
            continue
        for art in payload.get("articles", []):
            try:
                pdate = datetime.date.fromisoformat(art.get("publishedAt", "")[:10])
                if pdate < cutoff:
                    continue
                out.append(NewsItem(
                    source="gnews",
                    url=art.get("url", ""),
                    title=art.get("title", ""),
                    summary=(art.get("description", "") or "")[:500],
                    publish_date=pdate,
                    tickers=[],   # filled in filter step from keyword match
                    sentiment=None,
                    raw={"keyword": kw, "source_name": (art.get("source") or {}).get("name")},
                ))
            except Exception:
                continue
        time.sleep(0.4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Filter pipeline
# ─────────────────────────────────────────────────────────────────────────────

# Sector → keyword set for relevance filter (concise, anti-bloat)
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Technology":             ["tech", "AI", "semiconductor", "cloud", "software"],
    "Energy":                 ["oil", "gas", "OPEC", "sahdi", "energy", "crude", "refinery"],
    "Health Care":            ["health", "FDA", "pharma", "drug", "biotech", "medical"],
    "Financials":             ["bank", "Fed", "rate", "fintech", "yield", "loan"],
    "Consumer Discretionary": ["retail", "consumer", "auto", "EV", "Amazon"],
    "Consumer Staples":       ["staples", "food", "beverage"],
    "Industrials":            ["manufacturing", "industrial", "aerospace"],
    "Materials":              ["mining", "steel", "copper", "commodity"],
    "Utilities":              ["utility", "power grid", "electricity"],
    "Real Estate":            ["REIT", "real estate", "housing"],
    "Communication Services": ["telecom", "5G", "media", "streaming"],
}


def _portfolio_keywords(holdings_sectors: Iterable[str]) -> list[str]:
    """Combine sector keywords for active portfolio sectors."""
    kws: set[str] = set()
    for s in holdings_sectors:
        kws.update(SECTOR_KEYWORDS.get(s, []))
    return sorted(kws)


def _ticker_match(text: str, tickers: Iterable[str]) -> list[str]:
    """Return tickers mentioned in text (case-insensitive whole-word match)."""
    if not text:
        return []
    found = []
    text_u = text.upper()
    for t in tickers:
        if not t:
            continue
        # whole-word boundary match for ticker symbol
        pat = r"\b" + re.escape(t.upper()) + r"\b"
        if re.search(pat, text_u):
            found.append(t)
    return found


def filter_and_combine(
    news_lists: Iterable[Iterable[NewsItem]],
    *,
    portfolio_tickers: Iterable[str],
    portfolio_sectors: Iterable[str],
    max_items: int = DEFAULT_MAX_ITEMS,
) -> list[NewsItem]:
    """
    Combine news from multiple sources, filter to portfolio relevance, dedup
    by URL hash, sort by publish_date desc, truncate.

    Relevance rules:
      • Already-tagged ticker (sec_edgar / yfinance / alpha_vantage) — keep
      • GNews / generic headlines — match ticker mention OR sector keyword
    """
    pt = list(portfolio_tickers)
    ps = list(portfolio_sectors)
    sector_kw = set(k.lower() for k in _portfolio_keywords(ps))

    seen_urls: set[str] = set()
    relevant: list[NewsItem] = []
    for source_list in news_lists:
        for item in source_list:
            url_h = item.url_hash()
            if url_h in seen_urls:
                continue
            seen_urls.add(url_h)
            # Relevance check
            if item.tickers:
                relevant.append(item)
                continue
            text = (item.title or "") + " " + (item.summary or "")
            mentioned = _ticker_match(text, pt)
            if mentioned:
                item.tickers = mentioned
                relevant.append(item)
                continue
            text_low = text.lower()
            if any(kw in text_low for kw in sector_kw):
                # No ticker tag, but sector-relevant; keep with empty tickers
                relevant.append(item)

    relevant.sort(key=lambda n: (n.publish_date, n.source), reverse=True)
    return relevant[:max_items]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — pull everything for a portfolio
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_for_portfolio(
    tickers: Iterable[str],
    sectors: Iterable[str],
    *,
    days_back: int = DEFAULT_LOOKBACK_DAYS,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> list[NewsItem]:
    """
    Pull from all configured sources, filter to portfolio relevance, return
    deduped + sorted top-N items.

    Sources without API key gracefully skip (warning logged once). Fetch
    failures are non-fatal (per-source try/except).
    """
    tickers_list = [t for t in tickers if t]
    sectors_list = list(sectors)
    sector_kw    = _portfolio_keywords(sectors_list)

    sec_news = fetch_sec_edgar_8k(tickers_list, days_back=days_back)
    yf_news  = fetch_yfinance_news(tickers_list, days_back=days_back)
    av_news  = fetch_alpha_vantage_news(tickers_list, days_back=days_back)
    gn_news  = fetch_gnews(sector_kw, days_back=days_back)

    combined = filter_and_combine(
        [sec_news, yf_news, av_news, gn_news],
        portfolio_tickers=tickers_list,
        portfolio_sectors=sectors_list,
        max_items=max_items,
    )
    logger.info(
        "news_fetchers: %d sec_edgar + %d yfinance + %d alpha_vantage + %d gnews "
        "→ %d filtered (max_items=%d)",
        len(sec_news), len(yf_news), len(av_news), len(gn_news),
        len(combined), max_items,
    )
    return combined
