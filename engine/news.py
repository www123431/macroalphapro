"""
News Perceiver — three-source news aggregator.

Sources (priority order):
  1. Alpha Vantage News Sentiment API  (AV_KEY)   — ticker-level news + sentiment scores
  2. GNews API                          (GNEWS_KEY) — sector keyword search
  3. feedparser RSS fallback            (no key)    — Yahoo Finance / Google News RSS
"""
import logging
from datetime import datetime, timezone, timedelta

import feedparser
import requests

logger = logging.getLogger(__name__)

# ── News category → search keywords (for routing weight system) ───────────────
NEWS_CATEGORY_KEYWORDS: dict[str, str] = {
    "央行声明":   "Federal Reserve FOMC interest rate central bank monetary policy",
    "OPEC动态":  "OPEC oil supply production cut crude petroleum",
    "供应链":    "supply chain logistics semiconductor shortage inventory",
    "地缘政治":  "geopolitical conflict war sanctions trade war",
    "科技监管":  "tech regulation antitrust AI regulation data privacy",
    "PMI数据":   "PMI manufacturing ISM factory output industrial",
    "CPI数据":   "CPI inflation consumer price index PPI",
    "就业数据":  "jobs nonfarm payroll unemployment labor market hiring",
    "零售数据":  "retail sales consumer spending e-commerce",
    "信贷数据":  "credit lending bank loan mortgage default",
    "监管政策":  "regulation policy government legislation compliance",
    "FDA动态":   "FDA drug approval clinical trial biotech pharma",
    "政策医改":  "healthcare reform Medicare Medicaid insurance policy",
    "能源政策":  "energy policy renewable clean power grid climate",
}

# ── Sector → keyword query ────────────────────────────────────────────────────
SECTOR_QUERY_MAP: dict[str, str] = {
    "AI算力/半导体":   "semiconductor AI chip NVIDIA AMD",
    "科技成长(纳指)":  "NASDAQ tech QQQ growth stocks",
    "生物科技":        "biotech XBI FDA drug approval",
    "金融":            "financial sector XLF banks earnings",
    "全球能源":        "energy oil XLE crude OPEC",
    "工业/基建":       "industrial infrastructure XLI stocks",
    "医疗健康":        "healthcare XLV pharma stocks",
    "防御消费":        "consumer staples XLP stocks",
    "消费科技":        "consumer discretionary XLY retail",
    "美国REITs":       "REIT real estate VNQ interest rate Fed",
    "黄金":            "gold GLD commodity safe haven inflation",
    "美国长债":        "treasury bonds TLT Fed interest rate yield",
    "清洁能源":        "clean energy ICLN renewable solar wind",
    "沪深300":         "China A-share CSI300 economy PBOC",
    "中国科技":        "China tech KWEB Alibaba Tencent regulation",
    "新加坡蓝筹":      "Singapore STI EWS DBS OCBC MAS policy",
    "通讯传媒":        "communication media telecom XLC streaming Netflix Disney",
    "高收益债":        "high yield bond HYG junk bond credit spread default",
    # Macro-level query used by Tab 1
    "全球宏观":        "global economy Federal Reserve interest rate inflation GDP recession",
}

# Sentiment label → display emoji
_SENTIMENT_ICON = {
    "Bullish":          "🟢",
    "Somewhat-Bullish": "🟡",
    "Neutral":          "⚪",
    "Somewhat-Bearish": "🟠",
    "Bearish":          "🔴",
}

# ── Half-life decay rules ─────────────────────────────────────────────────────
# Each rule: (keyword_patterns, threshold_hours, label)
# First matching rule wins; fallback is the last entry.
_DECAY_RULES: list[tuple[list[str], int, str]] = [
    # Flash events — stale within 2 hours
    (["flash", "circuit breaker", "halt", "crash", "跳水", "熔断", "暴跌",
      "spike", "plunge", "급락"],                                               2,  "T-Flash"),
    # Hard data releases — stale within 12 hours
    (["CPI", "NFP", "nonfarm", "jobs report", "GDP", "PMI", "earnings",
      "财报", "就业", "通胀数据", "PPI"],                                        12, "T-Data"),
    # Policy decisions — stale within 24 hours
    (["FOMC", "rate decision", "rate hike", "rate cut", "interest rate",
      "加息", "降息", "利率决议", "央行决定"],                                   24, "S-Policy"),
    # Strategic guidance — stale within 72 hours
    (["outlook", "forecast", "annual report", "MAS", "five-year", "5-year",
      "roadmap", "展望", "年度", "中长期"],                                      72, "S-Strategic"),
    # Default: general financial news — 48 hours
    ([],                                                                        48, "T-General"),
]


def _classify_decay(title: str) -> tuple[int, str]:
    """Return (threshold_hours, label) for a headline based on keyword matching."""
    title_lower = title.lower()
    for keywords, hours, label in _DECAY_RULES:
        if not keywords:          # fallback rule
            return hours, label
        if any(kw.lower() in title_lower for kw in keywords):
            return hours, label
    return 48, "T-General"


def _freshness_badge(age_hours: float, threshold_hours: int) -> str:
    """Return a freshness badge string based on age vs threshold."""
    if age_hours <= threshold_hours * 0.25:
        return "🔴 LIVE"
    if age_hours <= threshold_hours * 0.6:
        return "🟡 ACTIVE"
    if age_hours <= threshold_hours:
        return "🟠 COOLING"
    return "📁 BACKGROUND"

_YAHOO_RSS  = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
_GNEWS_RSS  = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

# ── Cross-sector spillover transmission map ───────────────────────────────────
# For each target sector, list the sectors whose news historically transmits
# into it, along with a representative ticker and a one-line transmission description.
# Format: {target_sector: [(source_sector, source_ticker, transmission_description)]}
SPILLOVER_MAP: dict[str, list[tuple[str, str, str]]] = {
    "科技成长(纳指)": [
        ("美国长债",       "TLT",    "无风险利率上行 → 成长股折现率压缩估值"),
        ("AI算力/半导体",  "NVDA",   "算力供给景气 → 科技板块业绩预期"),
    ],
    "AI算力/半导体": [
        ("科技成长(纳指)", "QQQ",    "AI终端需求 → 算力订单能见度"),
        ("中国科技",       "KWEB",   "芯片出口管制 → 半导体供需格局"),
    ],
    "美国REITs": [
        ("美国长债",       "TLT",    "实际利率 → REITs折现率与融资成本"),
        ("金融",           "XLF",    "银行信贷环境 → 商业地产再融资"),
    ],
    "金融": [
        ("美国长债",       "TLT",    "收益率曲线斜率 → 银行净息差"),
        ("全球能源",       "XLE",    "大宗商品信贷风险 → 银行不良率预期"),
    ],
    "工业/基建": [
        ("全球能源",       "XLE",    "能源成本 → 工业生产利润率传导"),
        ("沪深300",        "000300.SS", "中国基建需求 → 全球工业订单"),
    ],
    "清洁能源": [
        ("全球能源",       "XLE",    "化石能源价格 → 清洁能源相对竞争力"),
        ("美国长债",       "TLT",    "利率环境 → 绿色项目融资成本"),
    ],
    "防御消费": [
        ("全球能源",       "XLE",    "能源价格 → 终端消费成本挤压"),
        ("美国长债",       "TLT",    "利率 → 消费信贷与储蓄率"),
    ],
    "消费科技": [
        ("科技成长(纳指)", "QQQ",    "科技周期景气度 → 消费电子需求"),
        ("防御消费",       "XLP",    "消费者信心 → 可选消费支出"),
    ],
    "生物科技": [
        ("医疗健康",       "XLV",    "医疗政策/医保覆盖 → 生物科技商业化预期"),
        ("美国长债",       "TLT",    "融资利率 → 早期生物科技研发资本"),
    ],
    "医疗健康": [
        ("生物科技",       "XBI",    "FDA审批节奏 → 医疗创新板块估值"),
        ("防御消费",       "XLP",    "防御性资金轮动 → 医疗板块资金面"),
    ],
    "黄金": [
        ("美国长债",       "TLT",    "实际利率 → 黄金持有机会成本"),
        ("全球能源",       "XLE",    "通胀预期 → 黄金保值需求"),
    ],
    "美国长债": [
        ("金融",           "XLF",    "银行资产负债表 → 债券需求与期限溢价"),
        ("全球能源",       "XLE",    "大宗商品通胀压力 → 长端利率预期"),
    ],
    "新加坡蓝筹": [
        ("沪深300",        "000300.SS", "中国经济 → 新加坡贸易与金融敞口"),
        ("金融",           "XLF",    "全球银行业 → 新加坡本地银行估值"),
    ],
    "中国科技": [
        ("AI算力/半导体",  "NVDA",   "芯片出口限制 → 中国AI发展能力"),
        ("沪深300",        "000300.SS", "A股政策情绪 → 中概股风险偏好"),
    ],
    "沪深300": [
        ("中国科技",       "KWEB",   "科技监管政策 → A股情绪"),
        ("全球能源",       "XLE",    "大宗商品价格 → 中国通胀与货币政策空间"),
    ],
    "全球能源": [
        ("工业/基建",      "XLI",    "工业需求 → 能源消耗预期"),
        ("沪深300",        "000300.SS", "中国经济增速 → 全球能源需求端"),
    ],
    "通讯传媒": [
        ("科技成长(纳指)", "QQQ",    "科技周期 → 流媒体/广告支出景气度"),
        ("消费科技",       "XLY",    "消费者支出 → 媒体订阅与广告收入"),
    ],
    "高收益债": [
        ("金融",           "XLF",    "信贷市场环境 → 高收益债违约率预期"),
        ("美国长债",       "TLT",    "无风险利率 → 信用利差压缩或扩大"),
    ],
}

# ── Trusted financial news sources for RSS fallback filtering ─────────────────
# When Google News RSS is used as fallback, only entries from these publishers
# are accepted. Yahoo Finance feed entries are always accepted (source-level trust).
_TRUSTED_RSS_SOURCES: frozenset[str] = frozenset({
    # Wire services
    "Reuters", "Bloomberg", "Associated Press", "AP News",
    # Major financial press
    "The Wall Street Journal", "WSJ", "Financial Times", "FT",
    "Barron's", "MarketWatch", "Investor's Business Daily", "IBD",
    # Business / macro news
    "CNBC", "Forbes", "Fortune", "Business Insider", "Axios",
    "The Economist", "Harvard Business Review",
    # Central-bank / official
    "Federal Reserve", "ECB", "MAS", "IMF", "World Bank",
    # Asian financial press
    "Nikkei Asia", "South China Morning Post", "Caixin", "The Straits Times",
    # Equity research aggregators
    "Seeking Alpha", "Motley Fool", "Morningstar",
    # Yahoo Finance RSS is always trusted (handled separately by URL check)
    "Yahoo Finance",
})


# ─────────────────────────────────────────────────────────────────────────────

class NewsPerceiver:
    def __init__(self, av_key: str = "", gnews_key: str = ""):
        self.av_key    = av_key
        self.gnews_key = gnews_key

    # ── Source 1: Alpha Vantage (ticker-level + sentiment) ───────────────────

    def _fetch_alpha_vantage(self, ticker: str, n: int = 5) -> tuple[list[dict], str]:
        """
        Returns (headlines, sentiment_summary).
        headline dict keys: title, source, published, sentiment_label, sentiment_score
        """
        if not self.av_key:
            return [], ""
        url = (
            "https://www.alphavantage.co/query"
            f"?function=NEWS_SENTIMENT&tickers={ticker}"
            f"&sort=LATEST&limit={n + 5}&apikey={self.av_key}"
        )
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json()
        except Exception as e:
            logger.warning("Alpha Vantage request failed: %s", e)
            return [], ""

        feed = data.get("feed", [])
        if not feed:
            return [], ""

        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        results = []
        sentiment_scores = []

        for item in feed:
            # AV time format: "20250331T142300"
            try:
                t = item.get("time_published", "")
                published = datetime.strptime(t, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
                if published < cutoff:
                    continue
                date_str = published.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                date_str = "recent"

            # Overall sentiment for this article
            label = item.get("overall_sentiment_label", "Neutral")
            score = float(item.get("overall_sentiment_score", 0))
            sentiment_scores.append(score)

            results.append({
                "title":           item.get("title", "").strip(),
                "source":          item.get("source", "Alpha Vantage"),
                "published":       date_str,
                "sentiment_label": label,
                "sentiment_score": score,
            })
            if len(results) >= n:
                break

        # Aggregate sentiment summary
        summary = ""
        if sentiment_scores:
            avg = sum(sentiment_scores) / len(sentiment_scores)
            if avg >= 0.35:
                overall = "Bullish"
            elif avg >= 0.15:
                overall = "Somewhat-Bullish"
            elif avg <= -0.35:
                overall = "Bearish"
            elif avg <= -0.15:
                overall = "Somewhat-Bearish"
            else:
                overall = "Neutral"
            icon = _SENTIMENT_ICON.get(overall, "⚪")
            summary = f"{icon} {overall}  (avg score: {avg:+.2f}  ·  based on {len(sentiment_scores)} articles)"

        return results, summary

    # ── Source 2: GNews API ───────────────────────────────────────────────────

    def _fetch_gnews(self, query: str, n: int = 5) -> list[dict]:
        if not self.gnews_key:
            return []
        url = (
            "https://gnews.io/api/v4/search"
            f"?q={requests.utils.quote(query)}"
            f"&token={self.gnews_key}&lang=en&max={n}&sortby=publishedAt"
        )
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json()
        except Exception as e:
            logger.warning("GNews request failed: %s", e)
            return []

        results = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        for article in data.get("articles", []):
            try:
                published = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")
                )
                if published < cutoff:
                    continue
                date_str = published.strftime("%Y-%m-%d %H:%M UTC")
            except Exception:
                date_str = "recent"

            results.append({
                "title":     article.get("title", "").strip(),
                "source":    article.get("source", {}).get("name", "GNews"),
                "published": date_str,
            })
            if len(results) >= n:
                break

        return results

    # ── Source 3: RSS fallback ────────────────────────────────────────────────

    def _fetch_rss(self, ticker: str, sector_name: str, n: int = 4) -> list[dict]:
        results = []
        urls = [
            (_YAHOO_RSS.format(ticker=ticker), True),   # Yahoo Finance: unconditionally trusted
            (_GNEWS_RSS.format(query=SECTOR_QUERY_MAP.get(sector_name, sector_name).replace(" ", "+")), False),
        ]
        for url, is_trusted_feed in urls:
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            feed_title = feed.feed.get("title", "RSS")
            cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            for entry in feed.entries:
                # For Google News RSS, filter by per-entry publisher whitelist
                if not is_trusted_feed:
                    entry_source = (
                        getattr(getattr(entry, "source", None), "title", None)
                        or entry.get("source", {}).get("title", "")
                        or ""
                    )
                    # Normalise: strip trailing punctuation / extra spaces
                    entry_source_clean = entry_source.strip().rstrip(".,")
                    if entry_source_clean not in _TRUSTED_RSS_SOURCES:
                        continue
                    source_label = entry_source_clean or feed_title
                else:
                    source_label = feed_title

                try:
                    published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if published < cutoff:
                        continue
                    date_str = published.strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    date_str = "recent"

                results.append({
                    "title":     entry.get("title", "").strip(),
                    "source":    source_label,
                    "published": date_str,
                })
                if len(results) >= n:
                    return results
        return results

    # ── Routing-weighted query builder ───────────────────────────────────────

    def _build_weighted_query(
        self, sector_name: str, macro_regime: str
    ) -> tuple[str, list[tuple[str, float]]]:
        """
        Build a search query weighted by learned NewsRoutingWeights.
        Returns (query_string, [(category, weight), ...] sorted desc).
        Falls back to SECTOR_QUERY_MAP if no learned weights exist.
        """
        try:
            from engine.memory import get_news_routing_weights
            weights = get_news_routing_weights(sector_name, macro_regime)
        except Exception:
            weights = {}

        if not weights:
            return SECTOR_QUERY_MAP.get(sector_name, sector_name), []

        # Sort categories by weight, take top 3
        sorted_cats = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_cats    = [(cat, w) for cat, w in sorted_cats if w >= 0.4][:3]

        if not top_cats:
            return SECTOR_QUERY_MAP.get(sector_name, sector_name), sorted_cats

        # Build query from top category keywords + base sector query
        keyword_parts = []
        for cat, _ in top_cats:
            kws = NEWS_CATEGORY_KEYWORDS.get(cat, "")
            if kws:
                # Take first 3 keywords from each category
                keyword_parts.extend(kws.split()[:3])

        # Append base sector keywords to ensure relevance
        base = SECTOR_QUERY_MAP.get(sector_name, "")
        base_words = base.split()[:3]
        keyword_parts.extend(w for w in base_words if w not in keyword_parts)

        query = " ".join(keyword_parts[:12])  # cap total length
        return query, sorted_cats

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch(
        self, sector_name: str, ticker: str,
        n: int = 6, macro_regime: str = "",
    ) -> tuple[list[dict], str, list[tuple[str, float]]]:
        """
        Aggregate from all three sources (AV → GNews → RSS fallback).
        Returns (headlines, sentiment_summary, routing_weights_used).
        """
        av_headlines, sentiment_summary = self._fetch_alpha_vantage(ticker, n=n)

        query, routing_used = self._build_weighted_query(sector_name, macro_regime)
        gn_headlines = self._fetch_gnews(query, n=n // 2 + 1)

        # Fallback to RSS if both APIs return nothing
        rss_headlines = []
        if not av_headlines and not gn_headlines:
            rss_headlines = self._fetch_rss(ticker, sector_name, n=n)

        # Merge and deduplicate
        seen, combined = set(), []
        for item in av_headlines + gn_headlines + rss_headlines:
            key = item["title"].lower()[:60]
            if key not in seen and item["title"]:
                seen.add(key)
                combined.append(item)
            if len(combined) >= n:
                break

        return combined, sentiment_summary, routing_used

    def build_context(
        self, sector_name: str, ticker: str,
        n: int = 6, macro_regime: str = "",
    ) -> str:
        """
        Return a structured news context string for agent prompt injection.

        Each headline is annotated with:
          - Decay label  (T-Flash / T-Data / S-Policy / S-Strategic / T-General)
          - Freshness badge (LIVE / ACTIVE / COOLING / BACKGROUND)
          - Sentiment score from Alpha Vantage (where available)

        BACKGROUND items are grouped separately so the agent treats them as
        historical context rather than actionable trading signals.
        """
        headlines, sentiment_summary, routing_used = self.fetch(
            sector_name, ticker, n=n, macro_regime=macro_regime
        )

        if not headlines:
            return "暂无近48小时相关新闻。"

        now = datetime.now(timezone.utc)
        active_lines: list[str] = []
        background_lines: list[str] = []

        for h in headlines:
            # Parse age
            try:
                pub = datetime.strptime(h["published"], "%Y-%m-%d %H:%M UTC").replace(
                    tzinfo=timezone.utc
                )
                age_h = (now - pub).total_seconds() / 3600
            except Exception:
                age_h = 0.0

            threshold, decay_label = _classify_decay(h["title"])
            badge = _freshness_badge(age_h, threshold)

            sentiment_tag = ""
            if "sentiment_label" in h:
                icon = _SENTIMENT_ICON.get(h["sentiment_label"], "⚪")
                sentiment_tag = f"  {icon} {h['sentiment_label']} ({h['sentiment_score']:+.2f})"

            entry = (
                f"[{badge}][{decay_label}] [{h['published']}]  {h['title']}"
                f"  —  {h['source']}{sentiment_tag}"
            )

            if badge == "📁 BACKGROUND":
                background_lines.append(entry)
            else:
                active_lines.append(entry)

        lines = [f"【新闻情报雷达 · {sector_name} ({ticker})】"]
        if sentiment_summary:
            lines.append(f"Alpha Vantage 情绪综合: {sentiment_summary}")

        # Show active routing weights so agent knows what was prioritised
        if routing_used:
            top3 = routing_used[:3]
            bars = " · ".join(
                f"{cat} {'█' * int(w * 8)}{' ' * (8 - int(w * 8))} {w:.2f}"
                for cat, w in top3
            )
            lines.append(f"新闻路由权重 [{macro_regime or '默认'}]: {bars}")

        lines.append("")

        if active_lines:
            lines.append("── 活跃信号区 (可触发交易逻辑) ──")
            for i, entry in enumerate(active_lines, 1):
                lines.append(f"{i}. {entry}")

        if background_lines:
            lines.append("")
            lines.append("── 历史背景区 (仅供上下文参考，不触发信号) ──")
            for i, entry in enumerate(background_lines, 1):
                lines.append(f"{i}. {entry}")

        return "\n".join(lines)

    def build_spillover_context(
        self, target_sector: str, macro_regime: str = "", n_per_source: int = 2,
    ) -> str:
        """
        Fetch headlines from sectors that transmit into target_sector.

        Source priority:
          1. Learned SpilloverWeight rows (activated when sample_count >= threshold)
             — sorted by |correlation|, conflict flag shown when sign reversed vs prior
          2. SPILLOVER_MAP priors (fallback when data insufficient)

        Returns empty string if no sources found or no headlines fetched.
        """
        # Build a lookup of prior transmission descriptions keyed by source_sector
        prior_sources: dict[str, tuple[str, str]] = {
            src: (ticker, desc)
            for src, ticker, desc in SPILLOVER_MAP.get(target_sector, [])
        }

        # Attempt to load learned weights (only returns rows above min_samples threshold)
        learned: list[dict] = []
        try:
            from engine.memory import get_spillover_weights
            learned = get_spillover_weights(target_sector, macro_regime)
        except Exception:
            pass

        # Decide which sources to use
        if learned:
            # Use learned layer: top sources by |correlation|
            learned_sorted = sorted(learned, key=lambda x: abs(x["correlation"]), reverse=True)
            active_sources: list[tuple[str, str, str, dict | None]] = []
            for row in learned_sorted[:4]:  # cap at 4 to limit prompt length
                src = row["source_sector"]
                ticker, desc = prior_sources.get(src, ("", f"学习传导系数 r={row['correlation']:+.2f}"))
                if not ticker:
                    # Not a known prior pair — use sector name as query key (fetch by name)
                    ticker = src
                active_sources.append((src, ticker, desc, row))
        else:
            # Fall back to static priors
            active_sources = [
                (src, ticker, desc, None)
                for src, ticker, desc in SPILLOVER_MAP.get(target_sector, [])
            ]

        if not active_sources:
            return ""

        sections: list[str] = [f"【跨板块溢出信号 · 传导至 {target_sector}】"]
        found_any = False

        for source_sector, source_ticker, transmission_desc, learned_row in active_sources:
            headlines, _, _ = self.fetch(
                source_sector, source_ticker,
                n=n_per_source, macro_regime=macro_regime,
            )
            if not headlines:
                continue
            found_any = True

            # Build header line with learned coefficient if available
            if learned_row:
                r = learned_row["correlation"]
                n = learned_row["sample_count"]
                coeff_tag = f"  学习系数 r={r:+.2f} (n={n})"
                if learned_row["conflicts_prior"]:
                    coeff_tag += "  ⚠ 方向与先验相反，传导关系存疑"
            else:
                coeff_tag = "  [先验]"

            sections.append(f"  ↳ {source_sector}  [{transmission_desc}]{coeff_tag}")
            for h in headlines[:n_per_source]:
                sentiment_tag = ""
                if "sentiment_label" in h:
                    icon = _SENTIMENT_ICON.get(h["sentiment_label"], "⚪")
                    sentiment_tag = f"  {icon}"
                sections.append(f"     • {h['title']}  —  {h['source']}{sentiment_tag}")

        if not found_any:
            return ""

        return "\n".join(sections)
