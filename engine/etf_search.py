"""
ETF Semantic Search
===================
Offline: build TF-IDF embeddings for each sector's description + keywords.
Online:  cosine-similarity query, returns top-k sectors with explanation.

No external API needed — uses sklearn TF-IDF + cosine similarity.
Supports Chinese + English mixed queries via character-level n-gram tokenisation.

Usage:
    from engine.etf_search import ETFSearchEngine
    engine = ETFSearchEngine.build()
    results = engine.search("新能源储能电池", top_k=3)
"""
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from engine.history import SECTOR_ETF

# Sector descriptions: Chinese keywords + English terms for each sector.
# Richer descriptions → better retrieval accuracy.
_SECTOR_DESCRIPTIONS: dict[str, str] = {
    "科技/AI":         "科技 人工智能 AI 芯片 半导体 算力 大模型 云计算 数据中心 nvidia 软件 technology semiconductor artificial intelligence chip cloud",
    "医疗健康":         "医疗 健康 生物技术 制药 医院 药品 疫苗 医疗器械 healthcare biotech pharma drug vaccine medical device",
    "消费":            "消费 零售 奢侈品 食品 饮料 可选消费 必需消费 consumer retail luxury food beverage discretionary staples",
    "金融":            "金融 银行 保险 券商 资产管理 利率 信用 finance bank insurance broker asset management interest rate credit",
    "能源/石油":        "能源 石油 天然气 原油 炼油 油气勘探 energy oil gas crude petroleum refinery exploration",
    "新能源/清洁能源":   "新能源 清洁能源 太阳能 风能 储能 锂电池 光伏 电动车 renewable clean energy solar wind battery storage EV electric vehicle",
    "材料/大宗商品":     "材料 大宗商品 铜 铁矿石 铝 黄金 化工 基础材料 materials commodities copper iron aluminum gold chemicals",
    "工业":            "工业 制造业 机械 航空 国防 基础设施 industrial manufacturing machinery aerospace defense infrastructure",
    "房地产/REITs":     "房地产 REIT 地产 写字楼 商业地产 住宅 物流仓储 real estate REIT office commercial residential logistics",
    "公用事业":         "公用事业 电力 水务 燃气 稳定分红 utilities electric water gas dividend stable",
    "通信服务":         "通信 电信 媒体 互联网 社交媒体 流媒体 communication telecom media internet social streaming",
    "国际市场":         "国际 新兴市场 欧洲 亚洲 全球 中国 emerging markets europe asia global china international",
    "加密/数字资产":     "加密货币 比特币 以太坊 区块链 数字资产 Web3 crypto bitcoin ethereum blockchain digital asset",
    "债券/固收":        "债券 固定收益 国债 信用债 高收益债 利率 久期 bonds fixed income treasury credit high yield duration",
    "黄金/贵金属":      "黄金 贵金属 白银 铂金 避险 通胀对冲 gold precious metals silver platinum safe haven inflation hedge",
    "杠杆/反向":        "杠杆 做空 反向 对冲 leveraged inverse short hedge",
    "多因子/Smart Beta": "多因子 因子 价值 成长 动量 质量 低波动 smart beta factor value growth momentum quality low volatility",
    "现金/短期债":       "现金 货币 短期 流动性 低风险 money market short term cash liquidity low risk",
}


@dataclass
class SearchResult:
    rank:        int
    sector:      str
    ticker:      str
    score:       float
    description: str


class ETFSearchEngine:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, sectors: list[str]):
        self._vectorizer = vectorizer
        self._matrix     = matrix
        self._sectors    = sectors

    @classmethod
    def build(cls, sector_descriptions: dict[str, str] | None = None) -> "ETFSearchEngine":
        """
        Build the search index from sector descriptions.
        Uses character-level 1-3-gram TF-IDF to handle Chinese without tokeniser.
        """
        desc_map = sector_descriptions or _SECTOR_DESCRIPTIONS

        # Fill in any sectors from SECTOR_ETF that lack a description
        all_sectors = list(SECTOR_ETF.keys())
        for s in all_sectors:
            if s not in desc_map:
                desc_map[s] = s  # sector name itself as fallback

        sectors = list(desc_map.keys())
        docs    = [desc_map[s] for s in sectors]

        # analyzer="char_wb": character n-grams within word boundaries
        # Works for Chinese (no tokeniser needed) and English simultaneously
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(docs)
        return cls(vectorizer, matrix, sectors)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """
        Find top-k sectors most similar to the query string.
        Returns list of SearchResult sorted by score descending.
        """
        q_vec = self._vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self._matrix).flatten()
        idx   = np.argsort(sims)[::-1][:top_k]
        sector_etf = SECTOR_ETF

        results = []
        for rank, i in enumerate(idx, 1):
            sector = self._sectors[i]
            results.append(SearchResult(
                rank        = rank,
                sector      = sector,
                ticker      = sector_etf.get(sector, "N/A"),
                score       = float(round(sims[i], 4)),
                description = _SECTOR_DESCRIPTIONS.get(sector, sector),
            ))
        return results

    def explain(self, query: str, result: SearchResult) -> str:
        """
        Return a brief explanation of why this sector matched the query.
        Extracts top overlapping terms between query and sector description.
        """
        q_lower   = query.lower()
        desc_lower = result.description.lower()

        # Find 2-char+ substrings in query that appear in description
        overlaps = []
        for length in range(4, 1, -1):
            for start in range(len(q_lower) - length + 1):
                chunk = q_lower[start:start + length]
                if chunk in desc_lower and chunk not in overlaps:
                    overlaps.append(chunk)
                    if len(overlaps) >= 4:
                        break
            if len(overlaps) >= 4:
                break

        if overlaps:
            return f"关键匹配：{' · '.join(overlaps)}"
        return f"语义相似度：{result.score:.2f}"


# Module-level singleton (lazy init)
_engine: ETFSearchEngine | None = None


def get_search_engine() -> ETFSearchEngine:
    global _engine
    if _engine is None:
        _engine = ETFSearchEngine.build()
    return _engine
