"""
engine/universe_manager.py — P2-11 动态 Universe 管理框架
============================================================
单一事实来源：所有可纳入 ETF 的注册表，替代 history.py 中的静态 SECTOR_ETF。

公开 API（供其他模块调用）：
  get_active_universe(asset_classes=None) -> dict[str, str]
  get_universe_by_class() -> dict[str, dict[str, str]]
  universe_health_check(as_of=None) -> UniverseHealthReport
  init_universe_db() -> None
  seed_batch_a() -> None
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field

import yfinance as yf
from sqlalchemy import Boolean, Column, Date, DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base

from engine.memory import SessionFactory, engine as _db_engine

logger = logging.getLogger(__name__)

_Base = declarative_base()


# ── ORM ────────────────────────────────────────────────────────────────────────

class UniverseETF(_Base):
    """
    每行代表一个可纳入 ETF 的注册记录。
    sector 字符串必须与 SECTOR_ETF 键完全一致，确保历史 DecisionLog 可关联。
    """
    __tablename__ = "universe_etfs"

    id             = Column(Integer,     primary_key=True, autoincrement=True)
    sector         = Column(String(100), nullable=False, unique=True)
    ticker         = Column(String(20),  nullable=False)
    asset_class    = Column(String(30),  nullable=False)
    # equity_sector / equity_factor / fixed_income / commodity / volatility
    batch          = Column(Integer,     nullable=False, default=0)
    # 0=初始18 / 1=批次A(权益因子+国际权益) / 2=批次B(债券/商品/波动率)
    inception_date = Column(Date,        nullable=True)
    active         = Column(Boolean,     default=True)
    added_at       = Column(Date,        nullable=True)
    removed_at     = Column(Date,        nullable=True)
    notes          = Column(String(200), nullable=True)


# ── 初始 18 个 ETF（与 history.py SECTOR_ETF 完全对应）──────────────────────────

_INITIAL_18 = [
    # (sector, ticker, asset_class, inception_date, notes)
    ("AI算力/半导体",  "SMH",   "equity_sector", "2000-05-05", "VanEck Semiconductor ETF"),
    ("科技成长(纳指)", "QQQ",   "equity_sector", "1999-03-10", "Invesco QQQ"),
    ("生物科技",       "XBI",   "equity_sector", "2006-01-31", "SPDR S&P Biotech"),
    ("金融",           "XLF",   "equity_sector", "1998-12-22", "Financial Select Sector SPDR"),
    ("全球能源",       "XLE",   "equity_sector", "1998-12-22", "Energy Select Sector SPDR"),
    ("工业/基建",      "XLI",   "equity_sector", "1998-12-22", "Industrial Select Sector SPDR"),
    ("医疗健康",       "XLV",   "equity_sector", "1998-12-22", "Health Care Select Sector SPDR"),
    ("防御消费",       "XLP",   "equity_sector", "1998-12-22", "Consumer Staples Select Sector SPDR"),
    ("消费科技",       "XLY",   "equity_sector", "1998-12-22", "Consumer Discretionary Select Sector SPDR"),
    ("美国REITs",      "VNQ",   "equity_sector", "2004-09-29", "Vanguard Real Estate ETF"),
    ("黄金",           "GLD",   "commodity",     "2004-11-18", "SPDR Gold Shares"),
    ("美国长债",       "TLT",   "fixed_income",  "2002-07-30", "iShares 20+ Year Treasury Bond"),
    ("清洁能源",       "ICLN",  "equity_sector", "2008-06-25", "iShares Global Clean Energy"),
    ("沪深300",        "ASHR",  "equity_sector", "2013-11-06", "Xtrackers Harvest CSI 300 China A-Shares"),
    ("中国科技",       "KWEB",  "equity_sector", "2013-01-31", "KraneShares CSI China Internet"),
    ("新加坡蓝筹",     "EWS",   "equity_sector", "1996-03-12", "iShares MSCI Singapore"),
    ("通讯传媒",       "XLC",   "equity_sector", "2018-06-18", "Communication Services Select Sector SPDR"),
    ("高收益债",       "HYG",   "fixed_income",  "2007-04-04", "iShares iBoxx $ High Yield Corporate Bond"),
]

# ── 批次 A：权益因子 + 国际权益（+7，共 25）──────────────────────────────────────

_BATCH_A = [
    ("小盘价值",   "IWN",  "equity_factor", "2000-07-24", "iShares Russell 2000 Value"),
    ("小盘成长",   "IWO",  "equity_factor", "2000-07-24", "iShares Russell 2000 Growth"),
    ("动量因子",   "MTUM", "equity_factor", "2013-04-16", "iShares MSCI USA Momentum Factor"),
    ("低波动因子", "USMV", "equity_factor", "2011-10-20", "iShares MSCI USA Min Vol Factor"),
    ("质量因子",   "QUAL", "equity_factor", "2013-07-18", "iShares MSCI USA Quality Factor"),
    ("日本",       "EWJ",  "equity_sector", "1996-03-12", "iShares MSCI Japan"),
    ("印度",       "INDA", "equity_sector", "2012-02-02", "iShares MSCI India"),
]

# ── 批次 B：跨资产（依赖 PRE-8 验证、within-class CSMOM 激活后才纳入）────────────
# 暂不自动 seed，等 P2-12 批次 B 实施时手动调用 seed_batch_b()
_BATCH_B = [
    ("美国综合债",  "AGG",  "fixed_income", "2003-09-29", "iShares Core U.S. Aggregate Bond"),
    ("美国中期国债","IEF",  "fixed_income", "2002-07-30", "iShares 7-10 Year Treasury Bond"),
    ("通胀保值债",  "TIP",  "fixed_income", "2003-12-05", "iShares TIPS Bond"),
    ("黄金矿业",    "GDX",  "commodity",    "2006-05-22", "VanEck Gold Miners"),
    ("农产品",      "DBA",  "commodity",    "2007-01-05", "Invesco DB Agriculture Fund"),
    ("波动率",      "VXX",  "volatility",   "2009-01-30", "iPath Series B S&P 500 VIX Short-Term Futures; 极性翻转"),
    ("抵押贷款信托","REM",  "equity_sector","2007-05-01", "iShares Mortgage Real Estate"),
]

_ADV_THRESHOLDS = {
    "equity_sector":  1_000_000,
    "equity_factor":  1_000_000,
    "fixed_income":     500_000,
    "commodity":        300_000,
    "volatility":       200_000,
    "fx":               200_000,
}

# ── 批次 C：外汇 / 原油 / 信用梯度 / EM 宽基（+4）───────────────────────────────
# PRE-8 备注：LQD 为债券 ETF，yfinance Adj Close 含票息再投资，使用前需验证偏差幅度。
# USO 2020-04 因 WTI 负油价事件进行了 1:8 反向拆股，yfinance 已调整，但 2020-04 前后
# 合约结构从单纯近月滚动变为跨月混合，momentum 特征存在结构性断点，使用时需注意。
_BATCH_C = [
    ("美元指数",   "UUP", "fx",           "2007-02-20", "Invesco DB US Dollar Index Bullish Fund"),
    ("原油",       "USO", "commodity",    "2006-04-10", "United States Oil Fund LP; 2020-04 反向拆股已调整"),
    ("投资级公司债","LQD", "fixed_income", "2002-07-22", "iShares iBoxx $ Investment Grade Corp Bond; Adj Close 含票息"),
    ("新兴市场宽基","EEM", "equity_sector","2003-04-07", "iShares MSCI Emerging Markets ETF"),
]

# ── 批次 D：补全期限/区域/EM债/商品宽基 缺口（+4）───────────────────────────────
# 学术依据：
#   SHY  — 短端国债，解锁 yield-curve slope 信号 (Estrella & Hardouvelis 1991)
#   VGK  — 欧洲 DM 权益缺口，~30% 全球 DM 市值
#   EMB  — EM 美元主权债，宏观 risk-on/off 最敏感固收类 (Du & Schreger 2016)
#   DBC  — 商品宽基（能源/金属/农），替代单品种 USO/DBA 的展期污染风险
_BATCH_D = [
    ("美国短期国债","SHY", "fixed_income", "2002-07-22", "iShares 1-3 Year Treasury Bond; 用于期限利差与 risk-off 现金流入"),
    ("欧洲发达",   "VGK", "equity_sector","2005-03-04", "Vanguard FTSE Europe ETF; 补 DM 欧洲缺口"),
    ("EM美元债",   "EMB", "fixed_income", "2007-12-17", "iShares JPMorgan USD Emerging Markets Bond"),
    ("商品宽基",   "DBC", "commodity",    "2006-02-03", "Invesco DB Commodity Index Tracking; 能源+金属+农产品"),
]

# ── 批次 E：strategy uplift cross-asset 扩展（+10 ETF, 2026-05-03）────────────
# 学术依据 spec: docs/spec_strategy_uplift_2026-05-03.md §1.1
# 依据 Moskowitz-Ooi-Pedersen (2012) — TSMOM 在 multi-asset universe 上 Sharpe 显著高于 sector-only。
#
# 新增填补 4 类 gap：
#   1. equity_sector US 缺 utilities (XLU) + 缺 DM Europe/Pacific (EWG/EWC/EWA)
#   2. commodity 缺 silver (SLV)
#   3. volatility 仅 1 polarity (VXX) → SVXY 为 inverse polarity
#   4. fx 仅 UUP → 加 EUR/JPY/CAD pure currency exposure
#   5. fixed_income 缺 non-USD govt (BWX)
_BATCH_E = [
    ("公用事业",      "XLU",  "equity_sector", "1998-12-22", "Utilities Select Sector SPDR; 补 US sector gap"),
    ("德国",         "EWG",  "equity_sector", "1996-03-12", "iShares MSCI Germany ETF; DM Europe diversifier"),
    ("加拿大",       "EWC",  "equity_sector", "1996-03-12", "iShares MSCI Canada ETF; commodity-linked DM"),
    ("澳大利亚",     "EWA",  "equity_sector", "1996-03-12", "iShares MSCI Australia ETF; commodity-linked DM"),
    ("白银",         "SLV",  "commodity",     "2006-04-21", "iShares Silver Trust; 补 precious metal gap"),
    ("短波动",       "SVXY", "volatility",    "2011-10-04", "ProShares Short VIX Futures; VXX 反向 polarity"),
    ("欧元",         "FXE",  "fx",            "2005-12-09", "Invesco CurrencyShares Euro Trust"),
    ("日元",         "FXY",  "fx",            "2007-02-12", "Invesco CurrencyShares Japanese Yen Trust"),
    ("加元",         "FXC",  "fx",            "2006-06-21", "Invesco CurrencyShares Canadian Dollar Trust"),
    ("国际政府债",   "BWX",  "fixed_income",  "2007-10-02", "SPDR Bloomberg International Treasury Bond"),
]

# ── 批次 F：Path K1 size-expanded production swap (+10 size/style, 2026-05-12) ─
# 学术依据 spec: docs/spec_path_k1_size_expanded_b_plus_v1.md (id=61, hash a0bbcbda3c28)
# Verdict: Path K1 PASS Sharpe 0.779 NW t 2.504 (2026-05-12); user-authorized
# production swap (project is paper-trade-only — no real $ at risk; per memory
# project_etf_strategy_reframe_t1_to_k1_todo_2026-05-12.md).
#
# 新增 10 size/style ETFs unlock size + value/growth/momentum rotation 维度;
# QL01 BAB production strategy unchanged (same signal_fn, larger universe).
_BATCH_F = [
    ("罗素2000小盘",     "IWM",  "equity_sector", "2000-05-22", "iShares Russell 2000 ETF; small-cap blend"),
    ("罗素2000小盘成长", "IWO",  "equity_factor", "2000-07-24", "iShares Russell 2000 Growth ETF"),
    ("罗素2000小盘价值", "IWN",  "equity_factor", "2000-07-24", "iShares Russell 2000 Value ETF"),
    ("罗素1000大盘",     "IWB",  "equity_sector", "2000-05-15", "iShares Russell 1000 ETF; large-cap blend broader than SPY"),
    ("罗素1000大盘价值", "IWD",  "equity_factor", "2000-05-22", "iShares Russell 1000 Value ETF"),
    ("罗素1000大盘成长", "IWF",  "equity_factor", "2000-05-22", "iShares Russell 1000 Growth ETF"),
    ("中盘500",          "IJH",  "equity_sector", "2000-05-22", "iShares Core S&P Mid-Cap ETF"),
    ("大盘动量",         "MTUM", "equity_factor", "2013-04-16", "iShares MSCI USA Momentum Factor ETF"),
    ("Vanguard小盘价值", "VBR",  "equity_factor", "2004-01-26", "Vanguard Small-Cap Value ETF; capacity diversifier (sector renamed from 小盘价值 to avoid legacy batch-1 inactive IWN row collision)"),
    ("大盘价值",         "VTV",  "equity_factor", "2004-01-26", "Vanguard Value ETF; capacity diversifier"),
]


# ── 初始化 ──────────────────────────────────────────────────────────────────────

def init_universe_db() -> None:
    """创建 universe_etfs 表，并 seed 初始 18 个 ETF（幂等）。"""
    _Base.metadata.create_all(_db_engine)
    _seed(rows=_INITIAL_18, batch=0)
    logger.info("UniverseManager: DB 初始化完成（batch=0, %d ETFs）", len(_INITIAL_18))


def seed_batch_a() -> None:
    """纳入批次 A（+7 权益 ETF），幂等。"""
    _seed(rows=_BATCH_A, batch=1)
    logger.info("UniverseManager: batch A seeded（+%d ETFs）", len(_BATCH_A))


def seed_batch_b() -> None:
    """
    纳入批次 B（+7 跨资产 ETF），幂等。
    ⚠️ 调用前必须：
      1. 完成 PRE-8（确认 AGG/IEF/TIP 的 yfinance Adj Close 含票息再投资）
      2. 激活 engine/signal.py 中的 within-class CSMOM（A-3）
    """
    _seed(rows=_BATCH_B, batch=2)
    logger.info("UniverseManager: batch B seeded（+%d ETFs）", len(_BATCH_B))


def seed_batch_c() -> None:
    """
    纳入批次 C（+4 FX/原油/信用/EM 宽基），幂等。
    ⚠️ 使用前注意：
      1. LQD 为债券 ETF，Adj Close 含票息再投资（与 AGG/IEF/TIP 同类偏差）
      2. USO 2020-04 前后合约结构有结构性断点，momentum 特征可能非平稳
      3. UUP 日均成交量较低（~200万股），回测滑点估算应适当放大
    """
    _seed(rows=_BATCH_C, batch=3)
    logger.info("UniverseManager: batch C seeded（+%d ETFs）", len(_BATCH_C))


def seed_batch_d() -> None:
    """
    纳入批次 D（+4 短端/欧洲/EM债/商品宽基），幂等。
    覆盖前几批的结构性缺口：期限利差、DM 欧洲、EM 主权债、商品宽基。
    """
    _seed(rows=_BATCH_D, batch=4)
    logger.info("UniverseManager: batch D seeded（+%d ETFs）", len(_BATCH_D))


def seed_batch_e() -> None:
    """
    纳入批次 E（+10 strategy uplift cross-asset），幂等。
    Spec: docs/spec_strategy_uplift_2026-05-03.md §1.1.
    填补 utilities/DM-Europe/precious-metal/inverse-vol/major-FX/intl-bond gap。
    """
    _seed(rows=_BATCH_E, batch=5)
    logger.info("UniverseManager: batch E seeded（+%d ETFs）", len(_BATCH_E))


def seed_batch_f() -> None:
    """
    纳入批次 F（+10 Path K1 size/style ETFs），幂等。
    Spec: docs/spec_path_k1_size_expanded_b_plus_v1.md (id=61) §2.1.
    Verdict: Path K1 PASS Sharpe 0.779 NW t 2.504 (2026-05-12); user-authorized
    production swap per memory project_etf_strategy_reframe_t1_to_k1_todo.
    QL01 BAB strategy unchanged; universe expands to include size + value/
    growth/momentum rotation dimension.
    """
    _seed(rows=_BATCH_F, batch=6)
    logger.info("UniverseManager: batch F seeded（+%d ETFs）", len(_BATCH_F))


def _seed(rows: list[tuple], batch: int) -> None:
    today = datetime.date.today()
    with SessionFactory() as session:
        for row in rows:
            sector, ticker, asset_class, inc_str, notes = row
            exists = session.query(UniverseETF).filter_by(sector=sector).first()
            if not exists:
                session.add(UniverseETF(
                    sector=sector,
                    ticker=ticker,
                    asset_class=asset_class,
                    batch=batch,
                    inception_date=datetime.date.fromisoformat(inc_str),
                    active=True,
                    added_at=today,
                    notes=notes,
                ))
        session.commit()


# ── 查询 API ────────────────────────────────────────────────────────────────────

def get_asset_class_map() -> dict[str, str]:
    """
    Return {ticker: asset_class} for all active ETFs.
    asset_class ∈ {equity_sector, equity_factor, fixed_income, commodity, fx, volatility}
    """
    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter_by(active=True).all()
        return {r.ticker: r.asset_class for r in rows}


def get_short_cap_group(ticker: str) -> str:
    """
    Strategy uplift F (2026-05-03): map ticker to short-cap group.
    Equity (sector + factor) → "equity" (squeeze risk; tighter cap)
    Bond / commodity / fx / volatility → "non_equity" (no squeeze; relaxed cap)

    Spec: docs/spec_strategy_uplift_2026-05-03.md §1.2
    Academic: Asness-Frazzini-Pedersen (2014); Moskowitz-Ooi-Pedersen (2012)
    """
    cls_map = get_asset_class_map()
    cls = cls_map.get(ticker, "equity_sector")  # unknown → safe default (equity)
    return "equity" if cls in ("equity_sector", "equity_factor") else "non_equity"


def get_active_universe(asset_classes: list[str] | None = None) -> dict[str, str]:
    """
    返回 {sector: ticker}，仅含 active=True 的行。
    asset_classes=None 返回全部；传 ["equity_sector"] 只取权益。
    """
    with SessionFactory() as session:
        q = session.query(UniverseETF).filter(UniverseETF.active == True)
        if asset_classes:
            q = q.filter(UniverseETF.asset_class.in_(asset_classes))
        rows = q.order_by(UniverseETF.batch, UniverseETF.id).all()
    return {r.sector: r.ticker for r in rows}


def get_universe_by_class() -> dict[str, dict[str, str]]:
    """
    返回 {asset_class: {sector: ticker}}。
    供 within-class CSMOM 使用（P2-12 批次 B 激活时）。
    """
    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter(UniverseETF.active == True).all()
    result: dict[str, dict[str, str]] = {}
    for r in rows:
        result.setdefault(r.asset_class, {})[r.sector] = r.ticker
    return result


def get_ticker_for_sector(sector: str) -> str | None:
    """单个 sector 查 ticker，用于替代 SECTOR_ETF.get(sector)。"""
    with SessionFactory() as session:
        row = session.query(UniverseETF).filter_by(sector=sector, active=True).first()
    return row.ticker if row else None


# ── 8-Dimensional ETF Tag Lookup (static, manually verified) ──────────────────
#
# Dimensions:
#   asset_class      : equity / fixed_income / commodity / volatility / real_estate / fx
#   equity_style     : large_cap / small_cap / mid_cap / multi_cap / N/A
#   factor           : momentum / value / quality / low_vol / growth / dividend / blend / none
#   sector_gics      : GICS Level-1 sector (or Broad Market / N/A for multi-sector / non-equity)
#   geography        : US / DM_ex_US / EM / global / Japan / India / China / Singapore
#   currency_exposure: USD / unhedged / hedged
#   thematic         : AI / clean_energy / infrastructure / none
#   liquidity_tier   : Tier1 (≥$50M ADV) / Tier2 (≥$10M ADV) / Tier3 (<$10M, needs review)
#
# Sources: ETF issuer factsheets, GICS classification, MSCI factor taxonomy.
# Last verified: 2026-04

ETF_TAGS: dict[str, dict[str, str]] = {
    # ── Batch 0: Initial 18 ──────────────────────────────────────────────────────
    "SMH":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "momentum",
             "sector_gics": "Information Technology", "geography": "US",        "currency_exposure": "USD",      "thematic": "AI",           "liquidity_tier": "Tier1"},
    "QQQ":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "growth",
             "sector_gics": "Information Technology", "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XBI":  {"asset_class": "equity",       "equity_style": "small_cap",  "factor": "none",
             "sector_gics": "Health Care",           "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XLF":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Financials",            "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XLE":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Energy",                "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XLI":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Industrials",           "geography": "US",        "currency_exposure": "USD",      "thematic": "infrastructure","liquidity_tier": "Tier1"},
    "XLV":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Health Care",           "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XLP":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "low_vol",
             "sector_gics": "Consumer Staples",      "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "XLY":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "growth",
             "sector_gics": "Consumer Discretionary","geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "VNQ":  {"asset_class": "real_estate",  "equity_style": "multi_cap",  "factor": "dividend",
             "sector_gics": "Real Estate",           "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "GLD":  {"asset_class": "commodity",    "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "global",    "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "TLT":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "ICLN": {"asset_class": "equity",       "equity_style": "multi_cap",  "factor": "none",
             "sector_gics": "Utilities",             "geography": "global",    "currency_exposure": "unhedged", "thematic": "clean_energy", "liquidity_tier": "Tier1"},
    "ASHR": {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Broad Market",          "geography": "China",     "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier2"},
    "KWEB": {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "growth",
             "sector_gics": "Communication Services","geography": "China",     "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    "EWS":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "dividend",
             "sector_gics": "Broad Market",          "geography": "Singapore", "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier2"},
    "XLC":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "growth",
             "sector_gics": "Communication Services","geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "HYG":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    # ── Batch A: Factor / Regional Equity ───────────────────────────────────────
    "IWN":  {"asset_class": "equity",       "equity_style": "small_cap",  "factor": "value",
             "sector_gics": "Broad Market",          "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "IWO":  {"asset_class": "equity",       "equity_style": "small_cap",  "factor": "growth",
             "sector_gics": "Broad Market",          "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "MTUM": {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "momentum",
             "sector_gics": "Broad Market",          "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "USMV": {"asset_class": "equity",       "equity_style": "multi_cap",  "factor": "low_vol",
             "sector_gics": "Broad Market",          "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "QUAL": {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "quality",
             "sector_gics": "Broad Market",          "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "EWJ":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Broad Market",          "geography": "Japan",     "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    "INDA": {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Broad Market",          "geography": "India",     "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    # ── Batch B: Cross-Asset ─────────────────────────────────────────────────────
    "AGG":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "IEF":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "TIP":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "GDX":  {"asset_class": "equity",       "equity_style": "multi_cap",  "factor": "none",
             "sector_gics": "Materials",             "geography": "global",    "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    "DBA":  {"asset_class": "commodity",    "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "global",    "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier2"},
    "VXX":  {"asset_class": "volatility",   "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "REM":  {"asset_class": "real_estate",  "equity_style": "multi_cap",  "factor": "dividend",
             "sector_gics": "Real Estate",           "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier2"},
    # ── Batch C: FX / Oil / IG Credit / EM ──────────────────────────────────────
    "UUP":  {"asset_class": "fx",           "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier2"},
    "USO":  {"asset_class": "commodity",    "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "global",    "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "LQD":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "EEM":  {"asset_class": "equity",       "equity_style": "multi_cap",  "factor": "none",
             "sector_gics": "Broad Market",          "geography": "EM",        "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    # ── Batch D: Short Treasury / Europe / EM Bond / Commodity Broad ──────────────
    "SHY":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "US",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "VGK":  {"asset_class": "equity",       "equity_style": "large_cap",  "factor": "none",
             "sector_gics": "Broad Market",          "geography": "DM_ex_US",  "currency_exposure": "unhedged", "thematic": "none",         "liquidity_tier": "Tier1"},
    "EMB":  {"asset_class": "fixed_income", "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "EM",        "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
    "DBC":  {"asset_class": "commodity",    "equity_style": "N/A",        "factor": "none",
             "sector_gics": "N/A",                  "geography": "global",    "currency_exposure": "USD",      "thematic": "none",         "liquidity_tier": "Tier1"},
}


# ── TICKER_METADATA alias (consumed by engine/signal.py compute_carry) ─────────
# Provides {ticker: {asset_class, ...}} lookup. Same shape as ETF_TAGS.
TICKER_METADATA: dict[str, dict[str, str]] = ETF_TAGS


# ── Tradability classification (REM/DBA/VXX → signal-only) ────────────────────
# These ETFs participate in signal computation (factor MAD, regime, spreads)
# but are excluded from portfolio sizing due to:
#   REM — ADV < 1M shares, institutional sizing not feasible
#   DBA — futures roll cost ~200 bp/yr contaminates momentum signal
#   VXX — daily roll decay; not a viable long-term position
_NON_TRADEABLE_TICKERS: set[str] = {"REM", "DBA", "VXX"}


def is_tradeable(ticker: str) -> bool:
    """True if ticker may receive portfolio weight; False = signal-only."""
    return ticker.upper() not in _NON_TRADEABLE_TICKERS


def get_non_tradeable_tickers() -> set[str]:
    """Return the signal-only ticker set (read-only copy)."""
    return set(_NON_TRADEABLE_TICKERS)


# ── Per-ticker max-weight overrides (Tier2 liquidity / theme concentration) ───
# When unset, portfolio.MAX_WEIGHT (default 25%) applies.
_PER_TICKER_MAX_WEIGHT: dict[str, float] = {
    "ICLN": 0.03,   # post-2021 squeeze liquidity decay; Tier2 cap
    "EWS":  0.05,   # Tier2 ADV
    "ASHR": 0.05,   # Tier2 ADV
    "UUP":  0.05,   # Tier2 ADV (~2M shares)
}


def get_max_weight_for_ticker(ticker: str, default: float) -> float:
    """Per-ticker max-weight override; falls back to default."""
    return _PER_TICKER_MAX_WEIGHT.get(ticker.upper(), default)


# ── Factor ETF set (used by signal.py for orthogonalization) ──────────────────
# Asness/Moskowitz/Pedersen 2013: factor-equity returns substantially overlap
# with broad-market and sector returns. Orthogonalizing target ETFs against
# these factor ETFs isolates the residual alpha for fairer cross-sectional
# ranking (avoids double-counting MTUM via SMH/QQQ etc.).
FACTOR_ETFS: set[str] = {"MTUM", "IWN", "IWO", "QUAL", "USMV"}


def get_universe_as_of(
    as_of_date: datetime.date,
    min_history_years: int = 3,
) -> dict[str, str]:
    """
    Return {sector: ticker} for ETFs that had at least min_history_years of
    history as of as_of_date.  Used by run_backtest() to prevent survivorship
    bias: a 2015 backtest date will NOT include XLC (IPO 2018) or MTUM (IPO 2013).

    Does NOT filter by active flag — an ETF removed from the universe today
    was still valid in historical periods.
    """
    try:
        cutoff = as_of_date.replace(year=as_of_date.year - min_history_years)
    except ValueError:
        # Feb 29 in non-leap target year
        cutoff = as_of_date.replace(year=as_of_date.year - min_history_years, day=28)

    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter(
            UniverseETF.inception_date != None,  # noqa: E711
            UniverseETF.inception_date <= cutoff,
        ).order_by(UniverseETF.batch, UniverseETF.id).all()
    return {r.sector: r.ticker for r in rows}


def load_all_etf_data() -> dict[str, tuple[str, datetime.date | None]]:
    """
    Return {sector: (ticker, inception_date)} for ALL ETFs in the registry.
    Pre-loads the full table once so run_backtest() can call get_universe_as_of()
    efficiently inside a tight loop without per-iteration DB queries.
    """
    with SessionFactory() as session:
        rows = session.query(UniverseETF).all()
    return {r.sector: (r.ticker, r.inception_date) for r in rows}


def get_universe_as_of_preloaded(
    as_of_date: datetime.date,
    all_etf_data: dict[str, tuple[str, datetime.date | None]],
    min_history_years: int = 3,
) -> dict[str, str]:
    """
    Fast variant of get_universe_as_of() for use inside walk-forward loops.
    Requires all_etf_data from load_all_etf_data() — no DB query per iteration.
    """
    try:
        cutoff = as_of_date.replace(year=as_of_date.year - min_history_years)
    except ValueError:
        cutoff = as_of_date.replace(year=as_of_date.year - min_history_years, day=28)

    return {
        sector: ticker
        for sector, (ticker, inc_date) in all_etf_data.items()
        if inc_date is not None and inc_date <= cutoff
    }


def get_etf_tags(ticker: str) -> dict[str, str] | None:
    """Return the 8-dimensional tag dict for a ticker, or None if not registered."""
    return ETF_TAGS.get(ticker.upper())


def query_universe_by_tag(**filters: str) -> dict[str, str]:
    """
    Filter the active universe by tag values.

    Example:
        query_universe_by_tag(factor="low_vol", geography="US")
        → {"防御消费": "XLP", "低波动因子": "USMV"}

    Only active ETFs are returned. Unknown tag keys are silently ignored.
    """
    active = get_active_universe()
    result: dict[str, str] = {}
    for sector, ticker in active.items():
        tags = ETF_TAGS.get(ticker, {})
        if all(tags.get(k) == v for k, v in filters.items()):
            result[sector] = ticker
    return result


# ── 健康检查 ────────────────────────────────────────────────────────────────────

@dataclass
class UniverseHealthReport:
    checked_at: datetime.date
    inactive_flagged: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def universe_health_check(as_of: datetime.date | None = None) -> UniverseHealthReport:
    """
    每月第一个交易日调用。检测 ADV 20日均量低于阈值的 ETF，自动标记 inactive。
    """
    if as_of is None:
        as_of = datetime.date.today()

    inactive_flagged: list[str] = []
    warnings: list[str] = []
    start = as_of - datetime.timedelta(days=35)

    with SessionFactory() as session:
        rows = session.query(UniverseETF).filter(UniverseETF.active == True).all()
        for row in rows:
            try:
                dl = yf.download(row.ticker, start=str(start), end=str(as_of),
                                 progress=False, auto_adjust=True)
                if dl.empty or "Volume" not in dl.columns:
                    warnings.append(f"{row.sector} ({row.ticker}): 无法获取成交量")
                    continue
                adv = float(dl["Volume"].tail(20).mean())
                threshold = _ADV_THRESHOLDS.get(row.asset_class, 500_000)
                if adv < threshold:
                    row.active = False
                    row.removed_at = as_of
                    inactive_flagged.append(row.sector)
                    logger.warning("Universe: %s (%s) ADV=%.0f < %.0f → inactive",
                                   row.sector, row.ticker, adv, threshold)
            except Exception as exc:
                warnings.append(f"{row.sector}: {exc}")
        session.commit()

    return UniverseHealthReport(
        checked_at=as_of,
        inactive_flagged=inactive_flagged,
        warnings=warnings,
    )
