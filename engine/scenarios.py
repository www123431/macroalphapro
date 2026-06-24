"""
Stress Test Scenario Engine
============================
Provides 13 pre-defined macro stress scenarios plus a free-form sandbox.

Scenario categories:
  What-if Sandbox : fully user-configurable
  Black Swan      : historically observed tail events
  Class A         : macro / monetary shocks
  Class B         : geopolitical shocks
  Class C         : sector-specific shocks

Each scenario is a dict that overrides the live market context before
passing it to the AI analysis prompt. The engine merges scenario params
with the current snapshot so the AI reasons about a plausible altered world.

Data leakage: none — scenarios only modify input parameters, not historical
training data. All analysis runs in a sandboxed session_state key so results
never contaminate the main Alpha Memory database.
"""
from __future__ import annotations
import datetime

# ── Sector ETF impact map ──────────────────────────────────────────────────────
# Each scenario lists expected directional impacts per sector.
# +1 = tailwind / beneficiary, -1 = headwind / hurt, 0 = neutral, None = unclear
_SECTOR_KEYS = [
    "AI算力/半导体", "科技成长(纳指)", "生物科技", "金融",
    "全球能源", "工业/基建", "医疗健康", "防御消费",
    "消费科技", "美国REITs", "黄金", "美国长债",
    "清洁能源", "沪深300", "中国科技", "新加坡蓝筹",
    "通讯传媒", "高收益债",
]

# ── Scenario definitions ───────────────────────────────────────────────────────

SCENARIOS: list[dict] = [

    # ── What-if Sandbox ────────────────────────────────────────────────────────
    {
        "id":       "sandbox",
        "category": "What-if 沙盒",
        "name":     "自定义情景",
        "description": "手动设置 VIX、利率、板块冲击，AI 基于你设定的参数生成分析。",
        "params": {
            "vix_override":         None,   # float or None (use current)
            "fed_funds_delta":      None,   # bps change, e.g. +100
            "oil_price_delta_pct":  None,   # % change in oil price
            "usd_delta_pct":        None,   # % change in DXY
            "custom_note":          "",     # free-text context
        },
        "sector_impacts": {k: 0 for k in _SECTOR_KEYS},
        "duration":    "用户定义",
        "probability": "自定义",
    },

    # ── Black Swan ─────────────────────────────────────────────────────────────
    {
        "id":       "gfc_2008",
        "category": "Black Swan",
        "name":     "全球金融危机重演（2008 GFC）",
        "description": (
            "雷曼兄弟式信用危机：银行间流动性冻结，信用利差急剧扩大，"
            "美联储启动紧急降息及量化宽松，全球股市腰斩。"
        ),
        "params": {
            "vix_override":         80.0,
            "fed_funds_delta":      -500,   # emergency 500bps cut over cycle
            "oil_price_delta_pct":  -70.0,
            "usd_delta_pct":        +12.0,  # dollar surges as safe haven
            "custom_note": (
                "信用市场完全冻结；HYG 信用利差扩至 2000bps；"
                "银行股市值蒸发 60%+；消费信贷断裂。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体": -1, "科技成长(纳指)": -1, "生物科技": -1,
            "金融": -1,          "全球能源": -1,        "工业/基建": -1,
            "医疗健康":  0,      "防御消费":  +1,       "消费科技": -1,
            "美国REITs": -1,     "黄金":      +1,       "美国长债": +1,
            "清洁能源":  -1,     "沪深300":   -1,       "中国科技": -1,
            "新加坡蓝筹": -1,    "通讯传媒":  -1,       "高收益债": -1,
        },
        "duration":    "12–18 个月",
        "probability": "极低（尾部事件）",
    },

    {
        "id":       "covid_2020",
        "category": "Black Swan",
        "name":     "全球疫情冲击（COVID-19 March 2020）",
        "description": (
            "黑天鹅式需求骤停：全球封锁导致经济活动急刹，"
            "股市 33 天内跌 34%，但随后史上最快 V 型反弹。"
        ),
        "params": {
            "vix_override":         85.0,
            "fed_funds_delta":      -150,
            "oil_price_delta_pct":  -60.0,
            "usd_delta_pct":        +8.0,
            "custom_note": (
                "实体经济停摆；航空/酒店/零售归零；居家经济爆发；"
                "美联储无限 QE；财政直升机撒钱。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体": +1,  "科技成长(纳指)": +1,  "生物科技": +1,
            "金融": -1,           "全球能源": -1,          "工业/基建": -1,
            "医疗健康":  +1,      "防御消费":  +1,         "消费科技": +1,
            "美国REITs": -1,      "黄金":      +1,         "美国长债": +1,
            "清洁能源":   0,      "沪深300":   -1,         "中国科技":  0,
            "新加坡蓝筹": -1,     "通讯传媒":  +1,         "高收益债": -1,
        },
        "duration":    "冲击 2 个月，反弹 12 个月",
        "probability": "极低（尾部事件）",
    },

    {
        "id":       "rate_hike_2022",
        "category": "Black Swan",
        "name":     "激进加息周期（2022 Fed Tightening）",
        "description": (
            "美联储 11 次加息共 525bps，40 年最快紧缩节奏。"
            "成长股估值大幅压缩，REITs 重挫，债券熊市创纪录。"
        ),
        "params": {
            "vix_override":         35.0,
            "fed_funds_delta":      +525,
            "oil_price_delta_pct":  +40.0,
            "usd_delta_pct":        +15.0,
            "custom_note": (
                "通胀 CPI 峰值 9.1%；美债 10Y 从 1.5% 升至 4.8%；"
                "纳指全年跌 33%；TLT 跌 30%（债券史上最差年份之一）。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技": -1,
            "金融":  +1,          "全球能源": +1,          "工业/基建":  0,
            "医疗健康":  0,       "防御消费": +1,          "消费科技": -1,
            "美国REITs": -1,      "黄金":      0,          "美国长债": -1,
            "清洁能源":  -1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹":  0,     "通讯传媒":  -1,         "高收益债": -1,
        },
        "duration":    "18 个月",
        "probability": "极低（尾部事件）",
    },

    # ── Class A: 宏观/货币冲击 ─────────────────────────────────────────────────
    {
        "id":       "emergency_rate_hike",
        "category": "Class A · 宏观冲击",
        "name":     "美联储紧急加息 +100bps",
        "description": (
            "通胀预期失控或就业超强迫使 FOMC 单次加息 100bps，"
            "超出市场定价，触发利率冲击和估值重定价。"
        ),
        "params": {
            "vix_override":         30.0,
            "fed_funds_delta":      +100,
            "oil_price_delta_pct":  0.0,
            "usd_delta_pct":        +5.0,
            "custom_note": "FOMC 单次加息 100bps，高于市场预期 75bps。",
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技": -1,
            "金融":  +1,          "全球能源":  0,          "工业/基建": -1,
            "医疗健康":  0,       "防御消费":  0,          "消费科技": -1,
            "美国REITs": -1,      "黄金":      0,          "美国长债": -1,
            "清洁能源":  -1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  -1,         "高收益债": -1,
        },
        "duration":    "1–3 个月",
        "probability": "低",
    },

    {
        "id":       "us_recession",
        "category": "Class A · 宏观冲击",
        "name":     "美国经济衰退（GDP -2%）",
        "description": (
            "连续两季度 GDP 负增长，失业率升至 6%+，"
            "企业盈利下修，信贷条件收紧。"
        ),
        "params": {
            "vix_override":         40.0,
            "fed_funds_delta":      -200,
            "oil_price_delta_pct":  -25.0,
            "usd_delta_pct":        -3.0,
            "custom_note": "GDP 连续两季负增长，失业率升至 6%+，企业盈利普遍下修 15–25%。",
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技":  0,
            "金融": -1,           "全球能源": -1,          "工业/基建": -1,
            "医疗健康":  +1,      "防御消费": +1,          "消费科技": -1,
            "美国REITs": -1,      "黄金":     +1,          "美国长债": +1,
            "清洁能源":  -1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  0,          "高收益债": -1,
        },
        "duration":    "6–18 个月",
        "probability": "中等",
    },

    {
        "id":       "usd_collapse",
        "category": "Class A · 宏观冲击",
        "name":     "美元大幅贬值（DXY -15%）",
        "description": (
            "美国财政赤字失控或去美元化加速，美元指数在 12 个月内跌 15%，"
            "新兴市场和大宗商品受益，美国本土资产相对承压。"
        ),
        "params": {
            "vix_override":         28.0,
            "fed_funds_delta":      -75,
            "oil_price_delta_pct":  +20.0,
            "usd_delta_pct":        -15.0,
            "custom_note": "DXY 跌 15%；黄金、大宗商品、新兴市场受益；进口型企业成本上升。",
        },
        "sector_impacts": {
            "AI算力/半导体":  0,  "科技成长(纳指)":  0,  "生物科技":  0,
            "金融":  -1,          "全球能源": +1,          "工业/基建": +1,
            "医疗健康":   0,      "防御消费":  -1,         "消费科技":  0,
            "美国REITs":  +1,     "黄金":     +1,          "美国长债":  -1,
            "清洁能源":   +1,     "沪深300":  +1,          "中国科技": +1,
            "新加坡蓝筹": +1,     "通讯传媒":  0,          "高收益债":  0,
        },
        "duration":    "12–24 个月",
        "probability": "低",
    },

    # ── Class B: 地缘政治冲击 ─────────────────────────────────────────────────
    {
        "id":       "us_china_trade_war",
        "category": "Class B · 地缘政治",
        "name":     "中美贸易战升级（关税 +60%）",
        "description": (
            "美国对中国商品加征 60% 关税，中国反制；"
            "供应链重构加速，科技脱钩深化，亚洲出口导向经济体承压。"
        ),
        "params": {
            "vix_override":         32.0,
            "fed_funds_delta":      0,
            "oil_price_delta_pct":  +5.0,
            "usd_delta_pct":        +6.0,
            "custom_note": "美对华关税 60%+；芯片出口管制全面升级；中概股资金外流加速。",
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技":  0,
            "金融": -1,           "全球能源":  0,          "工业/基建": -1,
            "医疗健康":   0,      "防御消费":  -1,         "消费科技": -1,
            "美国REITs":  0,      "黄金":      +1,         "美国长债": +1,
            "清洁能源":   -1,     "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  -1,         "高收益债": -1,
        },
        "duration":    "持续性冲击（12个月+）",
        "probability": "中等",
    },

    {
        "id":       "mideast_oil_shock",
        "category": "Class B · 地缘政治",
        "name":     "中东供油中断（油价 +50%）",
        "description": (
            "霍尔木兹海峡封锁或主要产油国冲突，"
            "全球原油供给减少 10–15%，油价短期飙升 50%。"
        ),
        "params": {
            "vix_override":         38.0,
            "fed_funds_delta":      0,
            "oil_price_delta_pct":  +50.0,
            "usd_delta_pct":        +4.0,
            "custom_note": "霍尔木兹海峡供应中断；Brent 原油冲击 $150；通胀预期二次上行。",
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技": -1,
            "金融": -1,           "全球能源": +1,          "工业/基建": -1,
            "医疗健康":   0,      "防御消费": -1,          "消费科技": -1,
            "美国REITs": -1,      "黄金":     +1,          "美国长债": -1,
            "清洁能源":  +1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  0,          "高收益债": -1,
        },
        "duration":    "3–6 个月（冲击期）",
        "probability": "低",
    },

    {
        "id":       "taiwan_strait",
        "category": "Class B · 地缘政治",
        "name":     "台海军事紧张升级",
        "description": (
            "台海军事演习升级为封锁态势，"
            "全球半导体供应链面临最高级别中断风险，亚洲金融市场剧烈波动。"
        ),
        "params": {
            "vix_override":         55.0,
            "fed_funds_delta":      0,
            "oil_price_delta_pct":  +15.0,
            "usd_delta_pct":        +10.0,
            "custom_note": (
                "台积电供应风险溢价飙升；全球芯片短缺预期重燃；"
                "亚洲股市资金外逃；美日韩联合军演。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技":  0,
            "金融": -1,           "全球能源":  +1,         "工业/基建": -1,
            "医疗健康":   0,      "防御消费":  +1,         "消费科技": -1,
            "美国REITs":  0,      "黄金":      +1,         "美国长债": +1,
            "清洁能源":   0,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  0,          "高收益债": -1,
        },
        "duration":    "持续性风险溢价（难预测）",
        "probability": "低",
    },

    # ── Class C: 板块特定冲击 ─────────────────────────────────────────────────
    {
        "id":       "tech_regulation",
        "category": "Class C · 板块冲击",
        "name":     "科技反垄断监管风暴",
        "description": (
            "美国/欧盟同步对大型科技平台发起反垄断拆分调查，"
            "科技股集体承压，AI 资本开支预期下修。"
        ),
        "params": {
            "vix_override":         28.0,
            "fed_funds_delta":      0,
            "oil_price_delta_pct":  0.0,
            "usd_delta_pct":        0.0,
            "custom_note": (
                "DOJ/FTC 发起拆分诉讼；GDPR 类法规扩展至 AI；"
                "Meta/Google/Amazon 同步受压；广告市场预期下修。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体": -1,  "科技成长(纳指)": -1,  "生物科技":  0,
            "金融":   0,          "全球能源":   0,         "工业/基建":  0,
            "医疗健康":   0,      "防御消费":   0,         "消费科技": -1,
            "美国REITs":  0,      "黄金":       0,         "美国长债":  0,
            "清洁能源":   0,      "沪深300":    0,         "中国科技": -1,
            "新加坡蓝筹":  0,     "通讯传媒":  -1,         "高收益债":  0,
        },
        "duration":    "6–24 个月（诉讼周期）",
        "probability": "中等",
    },

    {
        "id":       "china_property_collapse",
        "category": "Class C · 板块冲击",
        "name":     "中国房地产硬着陆",
        "description": (
            "中国主要房企大规模违约，银行系统不良率攀升，"
            "内需收缩、外资撤离，A 股承压，大宗商品需求下滑。"
        ),
        "params": {
            "vix_override":         35.0,
            "fed_funds_delta":      0,
            "oil_price_delta_pct":  -15.0,
            "usd_delta_pct":        +5.0,
            "custom_note": (
                "恒大/碧桂园式违约潮；中国银行系统 NPL 升至 5%+；"
                "铁矿石/铜价下跌 20%；A 股外资净流出创纪录。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体":  0,  "科技成长(纳指)":  0,  "生物科技":  0,
            "金融": -1,           "全球能源": -1,          "工业/基建": -1,
            "医疗健康":   0,      "防御消费":  0,          "消费科技":  0,
            "美国REITs":  0,      "黄金":     +1,          "美国长债": +1,
            "清洁能源":  -1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  0,          "高收益债": -1,
        },
        "duration":    "12–36 个月",
        "probability": "中等",
    },

    {
        "id":       "banking_stress",
        "category": "Class C · 板块冲击",
        "name":     "区域银行系统性压力（SVB 式）",
        "description": (
            "持续高利率环境下中小银行资产负债表恶化，"
            "存款挤兑引发系统性流动性担忧，信贷条件骤紧。"
        ),
        "params": {
            "vix_override":         42.0,
            "fed_funds_delta":      -50,
            "oil_price_delta_pct":  -10.0,
            "usd_delta_pct":        +3.0,
            "custom_note": (
                "3–5 家区域银行同步承压；FDIC 紧急接管；"
                "商业地产贷款风险暴露；货币市场基金流入加速。"
            ),
        },
        "sector_impacts": {
            "AI算力/半导体":  0,  "科技成长(纳指)": -1,  "生物科技": -1,
            "金融": -1,           "全球能源":  0,          "工业/基建": -1,
            "医疗健康":  +1,      "防御消费": +1,          "消费科技": -1,
            "美国REITs": -1,      "黄金":     +1,          "美国长债": +1,
            "清洁能源":  -1,      "沪深300":  -1,          "中国科技": -1,
            "新加坡蓝筹": -1,     "通讯传媒":  0,          "高收益债": -1,
        },
        "duration":    "1–3 个月（急性期）",
        "probability": "低",
    },
]

# ── Index for fast lookup ──────────────────────────────────────────────────────
SCENARIO_BY_ID: dict[str, dict] = {s["id"]: s for s in SCENARIOS}
SCENARIO_CATEGORIES: list[str] = list(dict.fromkeys(s["category"] for s in SCENARIOS))


def get_scenario(scenario_id: str) -> dict | None:
    return SCENARIO_BY_ID.get(scenario_id)


def build_stress_context(scenario: dict, base_vix: float) -> dict:
    """
    Merge scenario params with the current live context.
    Returns a dict that can be passed directly to the AI prompt builder.

    The base snapshot (prices, news) is NOT modified — only the macro
    parameters are overridden. This ensures the AI reasons about the
    scenario shock layered on top of current real-world conditions.
    """
    p = scenario["params"]
    effective_vix = p.get("vix_override") or base_vix

    lines = [
        f"【压力测试情景：{scenario['name']}】",
        f"类别：{scenario['category']}",
        "",
        scenario["description"],
        "",
        "── 情景参数覆盖 ──",
        f"VIX（压力值）：{effective_vix:.1f}",
    ]

    if p.get("fed_funds_delta") is not None:
        sign = "+" if p["fed_funds_delta"] >= 0 else ""
        lines.append(f"联邦基金利率变动：{sign}{p['fed_funds_delta']}bps")

    if p.get("oil_price_delta_pct") is not None:
        sign = "+" if p["oil_price_delta_pct"] >= 0 else ""
        lines.append(f"原油价格冲击：{sign}{p['oil_price_delta_pct']:.0f}%")

    if p.get("usd_delta_pct") is not None:
        sign = "+" if p["usd_delta_pct"] >= 0 else ""
        lines.append(f"美元指数（DXY）变动：{sign}{p['usd_delta_pct']:.0f}%")

    if p.get("custom_note"):
        lines.append(f"\n背景说明：{p['custom_note']}")

    lines += [
        "",
        "── 板块受益/受损先验 ──",
        "（+1 = 受益方向，-1 = 受损方向，0 = 中性）",
    ]
    for sector, impact in scenario.get("sector_impacts", {}).items():
        icon = "▲" if impact == 1 else ("▼" if impact == -1 else "·")
        lines.append(f"  {icon} {sector}")

    lines += [
        "",
        f"预期持续时间：{scenario.get('duration', '未知')}",
        f"历史发生概率估计：{scenario.get('probability', '未知')}",
        "",
        "⚠ 以上为假设情景，非实际市场状态。请基于此情景推演各板块的配置方向。",
    ]

    return {
        "stress_context":   "\n".join(lines),
        "effective_vix":    effective_vix,
        "scenario_id":      scenario["id"],
        "scenario_name":    scenario["name"],
        "sector_impacts":   scenario.get("sector_impacts", {}),
    }
