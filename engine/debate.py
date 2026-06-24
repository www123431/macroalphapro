"""
Sector Analysis Debate Engine
==============================
LangGraph-based multi-round debate for sector analysis.

Flow:
  blue_analysis → validate_format → red_challenge → blue_defend → ... → arbitrate → END
                        ↑ (retry if format invalid, max 3x)         ↑ (loop N rounds)

Design principles:
  - Format loop   : only checks structural completeness (XAI block fields present)
  - Debate rounds : test argument quality, not format — Red team MUST argue opposite direction
  - Arbitration   : third-party synthesis, weighs argument quality, produces final XAI
  - Asymmetry     : Red team given Blue's full output but forbidden from agreeing
"""
import json
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

import time
from engine.key_pool import (
    get_pool, AllKeysExhausted, EmptyOutputCircuitBreaker, BillingProtectionError,
    QUOTA_FAILS_BEFORE_SWITCH, RETRY_BASE_DELAY,
)

logger = logging.getLogger(__name__)


def _extract_blue_conclusion(blue_output: str) -> str:
    """
    Extract only the conclusion sections from blue_output to avoid
    anchoring the red team with Blue's full reasoning chain.

    Returns § 5 (配置建议) + 今日信号摘要 (→ 综合判断 line only),
    capped at ~200 chars. Red team sees Blue's WHAT, not Blue's WHY.
    """
    lines   = blue_output.splitlines()
    result  = []
    capture = False

    for line in lines:
        stripped = line.strip()
        # Start capturing at § 5 or 今日信号摘要
        if stripped.startswith("### 5.") or "今日信号摘要" in stripped:
            capture = True
        # Stop before XAI block
        if capture and stripped.startswith("### [XAI_ATTRIBUTION]"):
            break
        if capture:
            result.append(line)

    excerpt = "\n".join(result).strip()
    # Keep only lines with actual signal content (avoid blank padding)
    # Hard cap: 300 chars so red team can't absorb full reasoning
    return excerpt[:300] if excerpt else blue_output[:150]

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FORMAT_RETRIES = 3
DEFAULT_DEBATE_ROUNDS = 2      # Blue → Red → Blue-defend = 1 round; 2 rounds = 4 exchanges

# ── JSON Schema for structured sector analysis ─────────────────────────────────
# Used for Blue initial analysis and Arbitration nodes.
# All other nodes (Red challenge, Blue defense) remain free text.
# Key design goals:
#   1. Model-independent: any provider supporting JSON mode uses the same schema
#   2. Soft override: weight_adjustment_pct gives LLM ±20pp discretion
#   3. Fallback-safe: all fields have defaults, no silent data loss
_SECTOR_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "macro_transmission":       {"type": "string"},
        "news_catalysts":           {"type": "string"},
        "momentum_analysis":        {"type": "string"},
        "valuation_assessment":     {"type": "string"},
        "recommendation":           {"type": "string", "enum": ["超配", "标配", "低配"]},
        "recommendation_rationale": {"type": "string"},
        "immediate_signal":         {"type": "string"},
        "near_term_catalyst":       {"type": "string"},
        "structural_logic":         {"type": "string"},
        "synthesis":                {"type": "string"},
        # Soft override: LLM proposes a weight adjustment in ±20pp range.
        # Quant baseline weight + this delta = final weight.
        # LLM MUST justify any non-zero adjustment in adjustment_reason.
        "weight_adjustment_pct":    {"type": "number"},
        "adjustment_reason":        {"type": "string"},
        # Arbitration-only field; Blue analysis leaves this empty string.
        "arbitration_notes":        {"type": "string"},
        "overall_confidence":       {"type": "integer"},
        "macro_confidence":         {"type": "integer"},
        "news_confidence":          {"type": "integer"},
        "technical_confidence":     {"type": "integer"},
        "signal_drivers":           {"type": "string"},
        "invalidation_conditions":  {"type": "string"},
        "horizon":                  {"type": "string", "enum": ["季度(3个月)", "半年(6个月)"]},
        "quant_reconciliation":     {"type": "string"},
        # 0-100: LLM's estimate of the probability that the current TSMOM/CSMOM
        # signal will be invalidated within the next 60 days (regime flip, narrative
        # reversal, fundamental change). Low = signal likely to persist; high = fragile.
        "signal_invalidation_risk": {"type": "integer"},
        # P3-5: structured trade output fields
        "key_thesis":        {"type": "string"},
        "primary_risk":      {"type": "string"},
        "macro_regime_view": {"type": "string", "enum": ["risk-on", "neutral", "risk-off"]},
    },
    "required": [
        "recommendation", "recommendation_rationale",
        "weight_adjustment_pct", "adjustment_reason",
        "overall_confidence", "macro_confidence", "news_confidence", "technical_confidence",
        "signal_drivers", "invalidation_conditions", "horizon",
        "signal_invalidation_risk",
        "key_thesis", "primary_risk", "macro_regime_view",
    ],
}


# ── Key-pool call wrappers ─────────────────────────────────────────────────────

def _pool_call(prompt: str) -> str:
    """Free-text LLM call with key rotation. Used for Red/Blue-defense nodes."""
    _pool = get_pool()
    _max  = len(_pool._keys) * QUOTA_FAILS_BEFORE_SWITCH + 1

    for _ in range(_max):
        try:
            _pool.check_billing_limits()
            _m     = _pool.get_model()
            result = _m.generate_content(prompt).text
            _pool.report_success(has_content=bool(result.strip()))
            return result
        except BillingProtectionError as e:
            return f"[生成封锁: 计费保护 — {e}]"
        except AllKeysExhausted as e:
            return f"[生成失败: 所有 Key 已耗尽 — {e}]"
        except EmptyOutputCircuitBreaker as e:
            return f"[生成失败: 空输出熔断 — {e}]"
        except Exception as e:
            if _pool.is_quota_error(e):
                try:
                    _pool.report_quota_error()
                except AllKeysExhausted as ex:
                    return f"[生成失败: 所有 Key 已耗尽 — {ex}]"
                _wait = _pool._get_stats(_pool.current_label)["consecutive_quota"] * RETRY_BASE_DELAY
                if _wait > 0:
                    time.sleep(_wait)
                continue
            return f"[生成失败: {e}]"

    return "[生成失败: 超过最大重试次数]"


def _pool_call_json(prompt: str, schema: dict = None) -> dict | None:
    """
    JSON-mode LLM call. Returns parsed dict on success, None on failure.
    Uses response_mime_type='application/json' + response_schema — the model
    is forced to output valid JSON matching the schema.

    On any failure (quota, parse error, missing required fields):
        - logs the error
        - returns None (caller should invoke _quant_fallback_result)

    This is the P0 foundation: no regex parsing, no silent data loss.
    """
    _pool = get_pool()
    _max  = len(_pool._keys) * QUOTA_FAILS_BEFORE_SWITCH + 1

    for _ in range(_max):
        try:
            _pool.check_billing_limits()
            _m     = _pool.get_model(response_schema=schema or _SECTOR_ANALYSIS_SCHEMA)
            raw    = _m.generate_content(prompt).text
            _pool.report_success(has_content=bool(raw.strip()))

            data = json.loads(raw)
            failures = _assert_analysis_json(data)
            if failures:
                logger.warning("JSON assertion failures: %s", failures)
                # Auto-fixes applied in-place by _assert_analysis_json; continue with result
            return data

        except json.JSONDecodeError as e:
            logger.error("JSON mode parse failure (model returned non-JSON): %s", e)
            return None
        except BillingProtectionError:
            return None
        except AllKeysExhausted:
            return None
        except EmptyOutputCircuitBreaker:
            return None
        except Exception as e:
            if _pool.is_quota_error(e):
                try:
                    _pool.report_quota_error()
                except AllKeysExhausted:
                    return None
                _wait = _pool._get_stats(_pool.current_label)["consecutive_quota"] * RETRY_BASE_DELAY
                if _wait > 0:
                    time.sleep(_wait)
                continue
            logger.error("_pool_call_json unexpected error: %s", e)
            return None

    return None


# ── State ──────────────────────────────────────────────────────────────────────

class DebateState(TypedDict):
    # ── Input context ──────────────────────────────────────────────────────────
    sector_name:        str
    vix:                float
    macro_context:      str
    news_context:       str
    historical_context: str
    valuation_context:  str        # real-time valuation snapshot from yfinance
    quant_context:      dict       # pre-computed quant metrics; empty dict = not available
    quant_gate:         dict       # gate constraints from get_quant_gates(); {} = no gate
    max_rounds:         int        # configurable, default DEFAULT_DEBATE_ROUNDS

    # ── Blue team (initial analysis) ──────────────────────────────────────────
    blue_output:   str
    blue_xai:      dict
    blue_data:     dict   # raw JSON dict from blue_analysis (JSON mode)

    # ── Debate transcript ──────────────────────────────────────────────────────
    # Each entry: {"role": "red"|"blue_defense", "round": int, "content": str}
    debate_history: list

    # ── Control ───────────────────────────────────────────────────────────────
    debate_round:   int
    format_attempts: int
    format_valid:   bool

    # ── Final output ──────────────────────────────────────────────────────────
    final_output:       str    # the "winning" or synthesised analysis text
    final_xai:          dict   # arbitrator's final XAI block
    final_data:         dict   # raw JSON dict from arbitration (JSON mode)
    arbitration_notes:  str    # arbitrator's reasoning (stored in DecisionLog)


_HORIZON_VALID = ("季度(3个月)", "半年(6个月)")


# ── Structured output helpers ──────────────────────────────────────────────────

def _assert_analysis_json(data: dict) -> list[str]:
    """
    Validate and auto-fix a parsed analysis JSON dict.
    Returns list of assertion failures (empty = all passed).
    Auto-fixes are applied in-place (confidence clamping, horizon default, etc.)
    so the result is always usable even when failures are reported.
    """
    failures: list[str] = []

    if data.get("recommendation") not in ("超配", "标配", "低配"):
        failures.append(f"Invalid recommendation: {data.get('recommendation')!r}")
        data["recommendation"] = "标配"  # safe default

    for f in ("overall_confidence", "macro_confidence", "news_confidence", "technical_confidence"):
        v = data.get(f)
        if v is None:
            failures.append(f"Missing {f}")
            data[f] = 50
        else:
            data[f] = max(0, min(100, int(v)))

    if len(data.get("invalidation_conditions", "")) < 5:
        failures.append("invalidation_conditions too short")
        data["invalidation_conditions"] = data.get("recommendation_rationale", "未提供失效条件")

    if data.get("horizon") not in _HORIZON_VALID:
        data["horizon"] = "季度(3个月)"

    adj = data.get("weight_adjustment_pct")
    if adj is None or not isinstance(adj, (int, float)):
        data["weight_adjustment_pct"] = 0.0
    else:
        data["weight_adjustment_pct"] = max(-20.0, min(20.0, float(adj)))

    if not data.get("signal_drivers"):
        data["signal_drivers"] = data.get("recommendation_rationale", "—")

    inv_risk = data.get("signal_invalidation_risk")
    if inv_risk is None:
        failures.append("Missing signal_invalidation_risk")
        data["signal_invalidation_risk"] = 50
    else:
        data["signal_invalidation_risk"] = max(0, min(100, int(inv_risk)))

    return failures


def _render_analysis(data: dict, quant_gate: dict | None = None) -> str:
    """
    Render structured analysis JSON as markdown for UI display.
    Shows a soft gate warning (not hard block) when LLM direction diverges
    from quant signals — the divergence is recorded for accuracy attribution.
    """
    rec       = data.get("recommendation", "标配")
    rationale = data.get("recommendation_rationale", "—")
    adj       = float(data.get("weight_adjustment_pct", 0.0))
    adj_str   = (
        f"\n**权重调整建议：{adj:+.1f}pp** — {data.get('adjustment_reason', '')}"
        if abs(adj) >= 0.5 else ""
    )
    arb_notes   = data.get("arbitration_notes", "")
    arb_section = f"\n\n⚖️ 仲裁摘要：{arb_notes}" if arb_notes else ""

    gate_warn = ""
    if quant_gate and quant_gate.get("blocked") and rec in quant_gate.get("blocked", []):
        gate_warn = (
            f"\n\n> ⚠️ **量化信号分歧提示**：{quant_gate.get('reason', '')} — "
            f"LLM 建议「{rec}」与量化门控方向相反。"
            "分歧已记录，accuracy_score 后续将归因此次调整的贡献。"
        )

    is_fallback = data.get("_fallback", False)
    fallback_banner = (
        "\n\n> 🔴 **LLM 服务不可用 — 当前为纯量化信号结果**  \n"
        "> 以下分析不含基本面/新闻判断，仅反映 TSMOM/CSMOM/合成分。"
        "请在 LLM 恢复后重新生成。\n"
    ) if is_fallback else ""

    return "\n".join(filter(None, [
        fallback_banner,
        "### 1. 宏观与利率传导",
        data.get("macro_transmission", "—"),
        "\n### 2. 新闻与催化剂",
        data.get("news_catalysts", "—"),
        "\n### 3. 量价与动量",
        data.get("momentum_analysis", "—"),
        "\n### 4. 估值与市场定价",
        data.get("valuation_assessment", "—"),
        "\n### 5. 配置建议",
        f"**{rec}** — {rationale}{adj_str}",
        arb_section,
        "\n### 今日信号摘要",
        f"⚡ 即时扰动 (24小时内): {data.get('immediate_signal', '—')}",
        f"📅 近期催化剂 (1-3个月): {data.get('near_term_catalyst', '—')}",
        f"🏗 长期结构逻辑 (1年以上): {data.get('structural_logic', '—')}",
        f"→ 综合判断: {data.get('synthesis', '—')}",
        gate_warn,
    ]))


def _xai_from_json(data: dict) -> dict:
    """Build backward-compatible XAI dict from flat analysis JSON."""
    xai = {
        "overall_confidence":      data.get("overall_confidence", 50),
        "macro_confidence":        data.get("macro_confidence", 50),
        "news_confidence":         data.get("news_confidence", 50),
        "technical_confidence":    data.get("technical_confidence", 50),
        "signal_drivers":          data.get("signal_drivers", ""),
        "invalidation_conditions": data.get("invalidation_conditions", ""),
        "horizon":                 data.get("horizon", "季度(3个月)"),
    }
    xai["signal_invalidation_risk"] = data.get("signal_invalidation_risk", 50)
    qr = data.get("quant_reconciliation", "")
    if qr:
        xai["quant_reconciliation"] = {"raw": qr}
    return xai


def _quant_fallback_result(
    sector_name: str,
    quant_gate:  dict,
    quant_ctx:   dict,
) -> dict:
    """
    Generate a pure-quant analysis result when LLM is unavailable.
    Uses TSMOM + CSMOM + composite score to determine direction.
    All text fields are marked [LLM 不可用] so reviewers know the source.
    decision_source should be set to 'quant_fallback' by the caller.
    """
    gate   = quant_gate or {}
    tsmom  = gate.get("tsmom", 0)
    csmom  = gate.get("csmom", 0)
    comp   = float(gate.get("composite", 50.0))
    sig    = tsmom + csmom

    if sig >= 1:
        rec = "超配"
    elif sig <= -1:
        rec = "低配"
    else:
        rec = "标配"

    p_noise  = float((quant_ctx or {}).get("p_noise", 0.8))
    conf     = max(20, min(55, int((1.0 - p_noise) * 80)))
    rationale = (
        f"TSMOM={tsmom:+d}  CSMOM={csmom:+d}  "
        f"合成分={comp:.0f}/100  p_noise={p_noise:.0%}"
    )

    return {
        "macro_transmission":       "[LLM 不可用 — 纯量化模式]",
        "news_catalysts":           "[LLM 不可用 — 纯量化模式]",
        "momentum_analysis":        rationale,
        "valuation_assessment":     "[LLM 不可用 — 纯量化模式]",
        "recommendation":           rec,
        "recommendation_rationale": rationale,
        "immediate_signal":         "[LLM 不可用]",
        "near_term_catalyst":       "[LLM 不可用]",
        "structural_logic":         "[LLM 不可用]",
        "synthesis":                f"纯量化结论：{rec}（{rationale}）",
        "weight_adjustment_pct":    0.0,
        "adjustment_reason":        "LLM 不可用，权重调整为 0，沿用量化基准",
        "arbitration_notes":        "",
        "overall_confidence":       conf,
        "macro_confidence":         0,
        "news_confidence":          0,
        "technical_confidence":     conf,
        "signal_drivers":           rationale,
        "invalidation_conditions":  (
            f"TSMOM 或 CSMOM 信号翻转（当前 T={tsmom:+d} C={csmom:+d}）"
        ),
        "horizon":                  "季度(3个月)",
        "quant_reconciliation":     "纯量化模式，无 LLM 调整",
        "signal_invalidation_risk": 50,   # neutral default — no LLM insight available
        "_fallback":                True,
    }


_DIRECTION_PATTERNS = {
    "超配": ["超配", "overweight", "增持", "看多", "多头"],
    "标配": ["标配", "neutral", "中性", "持平"],
    "低配": ["低配", "underweight", "减持", "看空", "空头"],
}


def _extract_direction_from_text(text: str) -> str | None:
    """
    Extract the final direction (超配/标配/低配) from arbitration or Blue output.
    Looks for explicit 最终建议 / 配置建议 patterns first, then falls back
    to scanning the last 400 chars of text.
    Priority: explicit recommendation line > last occurrence in text.
    """
    # Try recommendation lines first
    for line in text.splitlines():
        stripped = line.strip()
        if any(kw in stripped for kw in ("最终建议", "配置建议", "建议：", "建议:")):
            for direction, patterns in _DIRECTION_PATTERNS.items():
                if any(p in stripped for p in patterns):
                    return direction

    # Fallback: last 400 chars
    tail = text[-400:]
    for direction in ["超配", "低配", "标配"]:  # priority order
        if direction in tail:
            return direction
    return None



# ── Node builders ──────────────────────────────────────────────────────────────

def build_debate_graph(model):
    """
    Factory: inject model via closure, return compiled LangGraph.
    """

    # ── Node 1: Blue team initial analysis (JSON mode) ───────────────────────
    def blue_analysis(state: DebateState) -> dict:
        sector     = state["sector_name"]
        vix        = state["vix"]
        macro_ctx  = state["macro_context"]
        news_ctx   = state["news_context"]
        hist_ctx   = state["historical_context"]
        val_ctx    = state.get("valuation_context", "")
        quant_ctx  = state.get("quant_context") or {}
        quant_gate = state.get("quant_gate") or {}

        val_section = (
            f"=== 估值与市场定价参考 ===\n{val_ctx}\n\n"
            if val_ctx and val_ctx != "估值数据暂不可用" else ""
        )

        # Quant metrics context — LLM must cite these in momentum_analysis
        quant_section = ""
        if quant_ctx:
            def _fmt(v): return f"{v:+.1%}" if v is not None else "数据不足"
            _pnoise_warn = (
                "超过30%警戒线，统计基础较弱"
                if quant_ctx.get("p_noise", 0) > 0.3 else "低于30%警戒线"
            )
            quant_section = (
                "=== 量化指标参考 ===\n"
                f"动量: 1M={_fmt(quant_ctx.get('mom_1m'))} "
                f"3M={_fmt(quant_ctx.get('mom_3m'))} "
                f"6M={_fmt(quant_ctx.get('mom_6m'))}\n"
                f"日VaR={quant_ctx.get('d_var',0):.2%}  "
                f"年化收益={quant_ctx.get('a_ret',0):+.1%}  "
                f"年化波动={quant_ctx.get('a_vol',0):.1%}\n"
                f"p_noise={quant_ctx.get('p_noise',1.0):.1%}（{_pnoise_warn}）\n"
                f"样本内R²={quant_ctx.get('val_r2','N/A')}  "
                f"样本外R²={quant_ctx.get('test_r2','N/A')}\n\n"
                "要求：momentum_analysis 字段必须显式引用以上动量数值。\n\n"
            )

        # Quant signal summary — always shown when gate data is available.
        # This ensures LLM sees quantitative signals regardless of gate status.
        gate_info = ""
        if quant_gate:
            tsmom_val  = quant_gate.get("tsmom", 0)
            csmom_val  = quant_gate.get("csmom", 0)
            comp_val   = quant_gate.get("composite", 50)
            gate_reason = quant_gate.get("reason", "无")
            severity   = quant_gate.get("severity", "clear")

            # Extra QuantAssessment metrics injected at call site (optional)
            ann_vol_str = (f"  年化波动: {quant_gate['ann_vol']:.1%}\n"
                           if "ann_vol" in quant_gate else "")
            sma_str     = (f"  vs SMA200: {quant_gate['price_vs_sma_200']:+.1%}  "
                           f"ATR(21): {quant_gate['atr_21']:.2f}\n"
                           if "atr_21" in quant_gate else "")
            vol_wt_str  = (f"  vol-parity建议权重: {quant_gate['vol_parity_weight']:.1%}  "
                           f"制度上限: {quant_gate['regime_weight_cap']:.1%}\n"
                           if "vol_parity_weight" in quant_gate else "")

            block_note = ""
            if quant_gate.get("blocked"):
                block_note = (
                    f"量化信号偏向：{gate_reason}\n"
                    "如需与量化信号相反方向，请在 adjustment_reason 中写明基本面理由，\n"
                    "并在 weight_adjustment_pct 填写调整幅度（-20 到 +20pp）。\n"
                )

            gate_info = (
                f"=== 量化信号参考（{severity}，软约束）===\n"
                f"TSMOM={tsmom_val:+d}  CSMOM={csmom_val:+d}  合成分={comp_val:.0f}/100\n"
                f"{ann_vol_str}{sma_str}{vol_wt_str}"
                f"{block_note}\n"
            )

        prompt = (
            f"你是高盛/摩根士丹利级别的板块策略分析师，专注于机构级跨资产配置研究。\n\n"
            f"分析对象：【{sector}】  VIX：{vix}\n\n"
            f"{gate_info}"
            f"=== 宏观背景 ===\n{macro_ctx}\n\n"
            f"=== 近期新闻 ===\n{news_ctx}\n\n"
            f"{val_section}"
            f"{quant_section}"
            f"=== 历史绩效参考 ===\n{hist_ctx}\n\n"
            "输出要求（JSON 字段说明）：\n"
            "- macro_transmission: 3-4句，货币政策传导机制（利率敏感性、信用利差、美元强弱）\n"
            "- news_catalysts: 3-4句，引用具体新闻标题，评估利多/利空方向\n"
            "- momentum_analysis: 3-4句，必须引用上方量化数值，评估动量持续性\n"
            "- valuation_assessment: 3-4句，估值合理性 + 是否已充分定价\n"
            "- recommendation: 超配/标配/低配 三选一\n"
            "- recommendation_rationale: 一句核心理由\n"
            "- immediate_signal: 24小时内即时扰动\n"
            "- near_term_catalyst: 1-3个月催化剂\n"
            "- structural_logic: 1年以上结构逻辑\n"
            "- synthesis: 综合判断一句话\n"
            "- weight_adjustment_pct: 建议权重调整（-20到+20pp，0=跟随量化基准）\n"
            "- adjustment_reason: 调整理由（若为0可填'跟随量化基准'）\n"
            "- arbitration_notes: 留空字符串（本节不填）\n"
            "- overall_confidence/macro_confidence/news_confidence/technical_confidence: 0-100\n"
            "- signal_drivers: 最多3个驱动因素，用·分隔\n"
            "- invalidation_conditions: 1-2个失效条件\n"
            "- horizon: 季度(3个月) 或 半年(6个月) 二选一\n"
            "- quant_reconciliation: 你的结论与量化信号的协调说明\n"
            "- signal_invalidation_risk: 0-100 整数，当前 TSMOM/CSMOM 信号在未来60天内被打断的概率估计。"
            "0=信号极稳固，100=几乎必然失效。请基于制度转换风险、叙事反转、基本面变化综合判断，"
            "而非对冲你的配置建议。低于30=趋势型环境；30-60=需要密切监控；60+=不建议建仓。\n\n"
            "机构语气，逻辑严谨，论据具体，禁止情绪化表达。"
        )

        data = _pool_call_json(prompt)
        if data is None:
            logger.warning("Blue analysis LLM failed for %s — using quant fallback", sector)
            data = _quant_fallback_result(sector, quant_gate, quant_ctx)

        rendered = _render_analysis(data, quant_gate)
        xai      = _xai_from_json(data)

        return {
            "blue_output":     rendered,
            "blue_xai":        xai,
            "blue_data":       data,
            "format_valid":    True,   # JSON mode: always structurally valid
            "format_attempts": 1,
        }

    # ── Node 2: Format validator ───────────────────────────────────────────────
    def validate_format(state: DebateState) -> dict:
        # Pure routing node — no LLM call, just return state unchanged
        # Routing logic is in the conditional edge below
        return {}

    # ── Node 3: Red team challenge ─────────────────────────────────────────────
    def red_challenge(state: DebateState) -> dict:
        sector      = state["sector_name"]
        vix         = state["vix"]
        blue_output = state["blue_output"]
        blue_dir    = state["blue_xai"].get("signal_drivers", "")
        round_num   = state.get("debate_round", 0) + 1
        history     = state.get("debate_history", [])

        # Give red team the latest defense if this is round 2+
        prev_defense = ""
        if history:
            last_defense = next(
                (h["content"] for h in reversed(history) if h["role"] == "blue_defense"),
                ""
            )
            if last_defense:
                prev_defense = f"\n\n蓝队最新辩护：\n{last_defense[:800]}"

        # Anchoring fix: red team sees only Blue's conclusion (§5 + 综合判断),
        # NOT the full 1200-char reasoning chain. The goal is adversarial
        # pressure on the position, not a critique of Blue's supporting logic.
        blue_conclusion = _extract_blue_conclusion(blue_output)

        prompt = (
            f"你是板块分析的红队（Devil's Advocate），第 {round_num} 轮挑战。\n\n"
            f"分析对象：【{sector}】  VIX：{vix}\n\n"
            f"=== 蓝队结论摘要 ===\n{blue_conclusion}"
            f"{prev_defense}\n\n"
            "你的任务：\n"
            "1. 你**必须**论证与蓝队完全相反的配置方向\n"
            "2. 逐条指出蓝队分析中最薄弱的3个论点，引用原文\n"
            "3. 提出蓝队忽视的关键风险或反向信号\n"
            "4. 明确说明在什么条件下蓝队的逻辑会崩溃\n\n"
            "禁止：部分同意、'蓝队有道理但...'、模糊立场\n"
            "要求：机构语气，论据具体，总长度150-200字\n\n"
            "格式：\n"
            "🔴 红队立场：[相反方向]\n"
            "⚔️ 核心攻击点：\n1. [引用蓝队原文] → [反驳理由]\n2. ...\n3. ...\n"
            "💣 致命风险：[蓝队完全未提及的最大下行风险]\n"
            "🚨 崩溃条件：[蓝队逻辑失效的具体触发条件]"
        )
        content = _pool_call(prompt)

        new_entry = {"role": "red", "round": round_num, "content": content}
        return {
            "debate_history": history + [new_entry],
            "debate_round":   round_num,
        }

    # ── Node 4: Blue team defense ──────────────────────────────────────────────
    def blue_defend(state: DebateState) -> dict:
        sector    = state["sector_name"]
        history   = state.get("debate_history", [])
        round_num = state.get("debate_round", 1)

        # Get the latest red challenge
        red_content = next(
            (h["content"] for h in reversed(history) if h["role"] == "red"),
            ""
        )

        prompt = (
            f"你是板块分析的蓝队，正在回应红队第 {round_num} 轮攻击。\n\n"
            f"分析对象：【{sector}】\n\n"
            f"=== 红队攻击 ===\n{red_content}\n\n"
            "你的任务：\n"
            "1. 逐条回应红队的每个攻击点，不能回避\n"
            "2. 如果红队某个论点有效，承认并调整置信度（不是改变方向）\n"
            "3. 强化你的核心论点，补充红队忽视的证据\n"
            "4. 明确说明你的方向是否维持，以及维持的底线条件\n\n"
            "格式：\n"
            "🔵 蓝队立场：[维持/微调，说明方向]\n"
            "🛡️ 逐点回应：\n1. [针对红队攻击点1]\n2. ...\n3. ...\n"
            "📌 补充证据：[红队未提及但支持蓝队的信号]\n"
            "⚖️ 置信度调整：[若有调整，说明幅度和原因；若无，说明理由]"
        )
        content = _pool_call(prompt)

        new_entry = {"role": "blue_defense", "round": round_num, "content": content}
        return {
            "debate_history": history + [new_entry],
        }

    # ── Node 5: Arbitration (JSON mode) ────────────────────────────────────────
    def arbitrate(state: DebateState) -> dict:
        sector     = state["sector_name"]
        vix        = state["vix"]
        history    = state.get("debate_history", [])
        quant_gate = state.get("quant_gate") or {}
        blue_data  = state.get("blue_data") or {}

        transcript_parts = [f"=== 蓝队初始分析 ===\n{state['blue_output'][:800]}"]
        for entry in history:
            role_label = "红队" if entry["role"] == "red" else "蓝队防御"
            transcript_parts.append(
                f"=== {role_label} · 第{entry['round']}轮 ===\n{entry['content']}"
            )
        transcript = "\n\n".join(transcript_parts)

        gate_info = ""
        if quant_gate and quant_gate.get("blocked"):
            gate_info = (
                f"=== 量化信号参考（软约束）===\n"
                f"TSMOM={quant_gate.get('tsmom',0):+d}  "
                f"CSMOM={quant_gate.get('csmom',0):+d}  "
                f"合成分={quant_gate.get('composite',50):.0f}/100\n"
                f"量化信号偏向方向：{quant_gate.get('reason','无')}\n"
                "如最终建议与量化信号相反，请在 adjustment_reason 中写明依据。\n\n"
            )

        blue_rec = blue_data.get("recommendation", "标配")
        blue_adj = float(blue_data.get("weight_adjustment_pct", 0.0))

        prompt = (
            f"你是独立仲裁者，负责裁定以下【{sector}】板块分析辩论。VIX={vix}\n\n"
            f"{gate_info}"
            f"{transcript}\n\n"
            f"蓝队初始建议：{blue_rec}（权重调整建议：{blue_adj:+.1f}pp）\n\n"
            "仲裁要求：\n"
            "1. 评估双方论据质量（具体性、逻辑性、证据充分度），不偏向任何一方\n"
            "2. 指出辩论中最有决定性的1-2个论据交锋\n"
            "3. 给出最终配置建议（可综合双方，不必完全采纳某一方）\n"
            "4. arbitration_notes 字段：2-3句仲裁摘要，说明谁的核心论点更有说服力\n"
            "5. weight_adjustment_pct：最终权重调整（-20到+20pp），可与蓝队不同\n"
            "6. confidence 分数应反映辩论后不确定性（双方争议大 → 整体置信度下降）\n\n"
            "输出要求（JSON 字段）：\n"
            "- macro_transmission/news_catalysts/momentum_analysis/valuation_assessment: 基于辩论综合的分析\n"
            "- recommendation: 超配/标配/低配 三选一\n"
            "- recommendation_rationale: 一句核心理由（反映辩论裁定结果）\n"
            "- immediate_signal/near_term_catalyst/structural_logic/synthesis: 综合判断\n"
            "- weight_adjustment_pct: 最终权重调整（-20到+20pp）\n"
            "- adjustment_reason: 调整理由\n"
            "- arbitration_notes: 仲裁摘要（谁赢了辩论，为什么，关键分歧在哪）\n"
            "- overall_confidence/macro_confidence/news_confidence/technical_confidence: 0-100\n"
            "- signal_drivers: 最多3个驱动因素，用·分隔\n"
            "- invalidation_conditions: 1-2个失效条件\n"
            "- horizon: 季度(3个月) 或 半年(6个月)\n"
            "- quant_reconciliation: 与量化信号的协调说明\n"
            "- signal_invalidation_risk: 0-100 整数，综合辩论双方观点后对信号失效概率的最终评估。"
            "辩论争议越大 → 失效风险越高。\n\n"
            "机构语气，逻辑严谨。"
        )

        data = _pool_call_json(prompt)
        if data is None:
            logger.warning("Arbitration LLM failed for %s — blue team wins by default", sector)
            data = dict(blue_data)
            data["arbitration_notes"] = "[仲裁 LLM 不可用，使用蓝队分析作为最终结果]"

        rendered  = _render_analysis(data, quant_gate)
        xai       = _xai_from_json(data)
        arb_notes = data.get("arbitration_notes", "")

        return {
            "final_output":      rendered,
            "final_xai":         xai,
            "final_data":        data,
            "arbitration_notes": arb_notes,
        }

    # ── Build graph ────────────────────────────────────────────────────────────
    builder = StateGraph(DebateState)
    builder.add_node("blue_analysis",  blue_analysis)
    builder.add_node("validate_format", validate_format)
    builder.add_node("red_challenge",  red_challenge)
    builder.add_node("blue_defend",    blue_defend)
    builder.add_node("arbitrate",      arbitrate)

    builder.set_entry_point("blue_analysis")
    builder.add_edge("blue_analysis", "validate_format")

    # Format loop: retry blue_analysis if XAI incomplete (max retries)
    builder.add_conditional_edges(
        "validate_format",
        lambda s: (
            "blue_analysis"
            if not s["format_valid"] and s["format_attempts"] < MAX_FORMAT_RETRIES
            else "red_challenge"
        ),
    )

    builder.add_edge("red_challenge", "blue_defend")

    # Debate loop: continue if rounds remaining, else arbitrate
    builder.add_conditional_edges(
        "blue_defend",
        lambda s: (
            "red_challenge"
            if s["debate_round"] < s.get("max_rounds", DEFAULT_DEBATE_ROUNDS)
            else "arbitrate"
        ),
    )

    builder.add_edge("arbitrate", END)

    return builder.compile()


# ── Public interface ───────────────────────────────────────────────────────────

def run_quant_coherence_check(xai: dict, quant: dict, direction: str = "") -> list[str]:
    """
    Method 3: Quant Coherence Test.

    Programmatically verify that the LLM's qualitative output is consistent
    with the pre-computed quantitative metrics. Returns a list of coherence flags;
    an empty list means no detected incoherence.

    Checks:
      QC-1  Final decision direction vs mom_3m sign (deterministic — no self-report).
            Uses the externally-extracted direction string, not quant_reconciliation.
            Goodhart fix: old QC-1 relied on quant_reconciliation["momentum_alignment"],
            which the LLM controls.  Replaced by direct comparison of direction (from
            debate output text) vs mom_3m (from yfinance data) — LLM influences neither.

      QC-2  Overconfidence relative to p_noise (deterministic — pure numeric comparison).
            Retained unchanged: both inputs are outside LLM control.

      QC-3/4  Informational hints only — NOT gating flags.
            p_noise_acknowledged / var_risk_addressed are self-reported booleans the LLM
            controls, so promoting them to gate conditions creates Goodhart dynamics.
            They are surfaced as display hints for the human reviewer but do NOT appear
            in the returned flags list that drives the QC warning UI.

    Parameters
    ----------
    xai       : parsed XAI dict from the arbitration node
    quant     : dict from compute_quant_metrics()
    direction : extracted direction string (e.g. "超配" / "低配" / "标配" / "long" / "short")
                obtained externally via extract_direction(ai_conclusion)
    """
    if not quant:
        return []   # no quant context in this run — skip all checks

    flags: list[str] = []

    # QC-1: deterministic direction vs momentum coherence
    # Compare the final decision direction (external) against mom_3m (market data).
    # No dependency on quant_reconciliation — Goodhart-safe.
    mom_3m = quant.get("mom_3m")
    if mom_3m is not None and direction:
        _dir = direction.lower()
        _bullish = any(k in _dir for k in ("超配", "多", "long", "看涨", "看多"))
        _bearish = any(k in _dir for k in ("低配", "空", "short", "看跌", "看空", "减仓"))
        if mom_3m > 0.05 and _bearish:
            flags.append(
                f"QC-1 方向与动量背离：结论方向偏空（{direction}）"
                f"但3M动量为正（{mom_3m:+.1%}）。"
                "若有基本面依据请在失效条件中明确说明。"
            )
        elif mom_3m < -0.05 and _bullish:
            flags.append(
                f"QC-1 方向与动量背离：结论方向偏多（{direction}）"
                f"但3M动量为负（{mom_3m:+.1%}）。"
                "请核查是否具备足够的基本面反转依据。"
            )

    # QC-2: overconfidence vs model noise (deterministic — unchanged)
    confidence = xai.get("overall_confidence", 50)
    p_noise    = quant.get("p_noise", 0.0)
    if confidence > 80 and p_noise > 0.30:
        flags.append(
            f"QC-2 置信度过高：overall_confidence={confidence} 但模型噪音估计"
            f"（p_noise={p_noise:.1%}）超过30%阈值，统计基础较弱，建议降至 70 以下。"
        )

    # QC-3/4: informational only — NOT added to flags (Goodhart downgrade)
    # These are still computed and available for callers that want display hints,
    # but they do not contribute to the gate logic.
    # Caller can inspect quant["p_noise"] and quant["d_var"] directly for display.

    return flags


def run_sector_debate(
    model,
    sector_name:        str,
    vix:                float,
    macro_context:      str  = "",
    news_context:       str  = "",
    historical_context: str  = "",
    valuation_context:  str  = "",
    quant_context:      dict | None = None,
    quant_gate:         dict | None = None,
    max_rounds:         int  = DEFAULT_DEBATE_ROUNDS,
) -> dict:
    """
    Run the full debate pipeline for a sector analysis.

    Returns:
        {
            "final_output":          str,   # arbitrated analysis (rendered markdown)
            "final_xai":             dict,  # arbitrated XAI fields
            "final_data":            dict,  # raw JSON from arbitration node
            "weight_adjustment_pct": float, # LLM soft override delta (-20 to +20pp)
            "blue_output":           str,   # original blue team analysis
            "blue_xai":              dict,  # blue team XAI fields
            "debate_history":        list,  # full transcript
            "arbitration_notes":     str,   # arbitrator's reasoning
        }
    """
    graph = build_debate_graph(model)

    initial_state: DebateState = {
        "sector_name":        sector_name,
        "vix":                vix,
        "macro_context":      macro_context,
        "news_context":       news_context,
        "historical_context": historical_context,
        "valuation_context":  valuation_context,
        "quant_context":      quant_context or {},
        "quant_gate":         quant_gate or {},
        "max_rounds":         max_rounds,
        "blue_output":        "",
        "blue_xai":           {},
        "blue_data":          {},
        "debate_history":     [],
        "debate_round":       0,
        "format_attempts":    0,
        "format_valid":       False,
        "final_output":       "",
        "final_xai":          {},
        "final_data":         {},
        "arbitration_notes":  "",
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        logger.error("Debate graph failed: %s", e)
        return {
            "final_output":      f"[辩论流程失败: {e}]",
            "final_xai":         {},
            "blue_output":       "",
            "blue_xai":          {},
            "debate_history":    [],
            "arbitration_notes": "",
        }

    final_data = result.get("final_data") or {}
    return {
        "final_output":        result.get("final_output", ""),
        "final_xai":           result.get("final_xai", {}),
        "final_data":          final_data,
        "weight_adjustment_pct": float(final_data.get("weight_adjustment_pct", 0.0)),
        "blue_output":         result.get("blue_output", ""),
        "blue_xai":            result.get("blue_xai", {}),
        "debate_history":      result.get("debate_history", []),
        "arbitration_notes":   result.get("arbitration_notes", ""),
    }
