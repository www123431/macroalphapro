"""
engine/narrative_builder.py — P3-11 Supervisor 叙述质量升级
============================================================
将机器码事件列表（"ATR_STOP: XLC"）转换为交易台备忘录风格的结构化叙述。

纯 Python 模板渲染，零 LLM 调用，延迟 < 5ms。

公开 API
--------
  NarrativeBuilder.build_batch_narrative(t_day, result, ...) -> str
      在 daily_batch.py 批次完成后调用，生成【市场状态】+【今日动作】两段。
      结果持久化到 DailyBriefSnapshot.narrative。

  NarrativeBuilder.build_section_b(snap, pending_gates, pending_approvals) -> str
      在 orchestrator.py 渲染时调用，追加实时【待审批事项】段。
      不持久化（每次渲染重新计算）。

  NarrativeBuilder.gate_summary_line(gate_label, dry_run_result) -> str
      生成单个 Gate 卡片的一行叙述摘要（Section A 用）。
"""
from __future__ import annotations

import datetime
import json
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from engine.daily_batch import BatchResult


# ── Regime display helpers ────────────────────────────────────────────��─────────

_REGIME_CN = {
    "risk-on":    "风险偏好（risk-on）",
    "risk-off":   "风险规避（risk-off）",
    "transition": "过渡制度（transition）",
    "unknown":    "制度识别中（HMM数据待更新）",
    "":           "制度待识别",
}
_REGIME_ARROW = {
    "risk-on":    "↑",
    "risk-off":   "↓",
    "transition": "→",
}

def _fmt_regime(regime: str, p: float) -> str:
    label = _REGIME_CN.get(regime, regime)
    return f"{label}，P(risk-on)={p:.0%}"


# ── Yield curve helper (optional, graceful degradation) ─────────────────────────

def _try_yield_curve_line() -> str:
    """Return one-line yield curve narrative, or "" when data unavailable.
    macro_fetcher.get_yield_curve_snapshot() handles FRED→yfinance fallback internally.
    """
    try:
        from engine.macro_fetcher import get_yield_curve_snapshot
        snap = get_yield_curve_snapshot()
        narr = (snap or {}).get("narrative", "")
        if narr and "数据不可用" not in narr:
            return narr
    except Exception as exc:
        logger.debug("narrative_builder: yield curve fetch skipped: %s", exc)
    return ""


# ── Event parser helpers ────────────────────────────────────────────────────────

def _parse_alert_sectors(alerts: list[str], keyword: str) -> list[str]:
    """Extract sector names from alert strings containing keyword."""
    return [a.split(":")[0].strip() for a in alerts if keyword in a]


def _sector_list(sectors: list[str], limit: int = 4) -> str:
    if not sectors:
        return ""
    shown = sectors[:limit]
    tail  = f"等 {len(sectors)} 个" if len(sectors) > limit else f"{len(sectors)} 个"
    return f"{tail}（{', '.join(shown)}{'…' if len(sectors) > limit else ''}）"


# ── Core builder ────────────────────────────────────────────────────────────────

class NarrativeBuilder:
    """
    Stateless helper — instantiate once, call as needed.
    All methods return plain Chinese text with 【section】 markers.
    """

    # ── Section 1: 市场状态 ──────────────────────────────────────────────────

    def _section_market(
        self,
        t_day: datetime.date,
        regime: str,
        p_risk_on: float,
        regime_prev: str,
        regime_changed: bool,
        include_yield_curve: bool = True,
    ) -> str:
        _regime_clean = regime if regime and regime not in ("unknown", "") else "transition"
        regime_str = _fmt_regime(_regime_clean, p_risk_on)

        if regime_changed and regime_prev and regime_prev not in ("unknown", ""):
            prev_cn = _REGIME_CN.get(regime_prev, regime_prev)
            transition = f"制度自 {prev_cn} 切换为 {regime_str}。"
        elif _regime_clean == "transition" and regime in ("unknown", ""):
            transition = f"宏观制度识别中（历史数据更新后自动确认）。"
        else:
            transition = f"宏观制度维持{regime_str}。"

        yc_line = _try_yield_curve_line() if include_yield_curve else ""

        parts = [transition]
        if yc_line:
            parts.append(yc_line)

        return "【市场状态】" + " ".join(parts)

    # ── Section 2: 今日动作 ──────────────────────────────────────────────────

    def _section_actions(self, result: "BatchResult") -> str:
        alerts = getattr(result, "risk_alerts", [])

        # Classify stops: auto (Layer 2) vs pending (Layer 3)
        auto_stops    = [a for a in alerts if "auto_executed" in a]
        hard_stops    = _parse_alert_sectors(alerts, "hard_stop")
        drawdn_stops  = _parse_alert_sectors(alerts, "drawdown_stop")
        all_auto_stop_sectors = _parse_alert_sectors(
            [a for a in alerts if "auto_executed" in a], ""
        ) or []
        # actually parse from the alert string directly
        auto_sectors = [
            a.split(":")[0].strip()
            for a in alerts
            if "auto_executed" in a
        ]
        pending_stop_sectors = [
            s for s in (hard_stops + drawdn_stops)
            if not any(s in a and "auto_executed" in a for a in alerts)
        ]

        tsmom_flips     = _parse_alert_sectors(alerts, "tsmom_flip")
        regime_compress = [a for a in alerts if "regime_compress" in a]
        entries         = getattr(result, "entries_triggered", [])
        invalidations   = getattr(result, "invalidations",      [])
        corr_blocked    = getattr(result, "corr_blocked",        [])
        rebalance       = getattr(result, "rebalance_orders",    [])

        events: list[str] = []

        if auto_sectors:
            events.append(
                f"止损触发 {len(auto_sectors)} 笔（Layer 2 自动执行：{', '.join(auto_sectors[:3])}{'…' if len(auto_sectors) > 3 else ''}）"
            )
        if pending_stop_sectors:
            events.append(
                f"止损待审批 {len(pending_stop_sectors)} 笔"
                f"（{', '.join(pending_stop_sectors[:3])}{'…' if len(pending_stop_sectors) > 3 else ''}）"
            )
        if tsmom_flips:
            events.append(
                f"TSMOM 方向翻转 {len(tsmom_flips)} 个资产"
                f"（{', '.join(tsmom_flips[:3])}{'…' if len(tsmom_flips) > 3 else ''}）"
            )
        if regime_compress:
            events.append(f"制度压缩触发 {len(regime_compress)} 笔")
        if entries:
            events.append(f"入场信号触发 {len(entries)} 个（{', '.join(entries[:3])}{'…' if len(entries) > 3 else ''}），进入审批队列")
        if invalidations:
            events.append(f"持仓无效化 {len(invalidations)} 个")
        if corr_blocked:
            events.append(f"相关性拦截 {len(corr_blocked)} 个")
        if rebalance:
            events.append(f"月末再平衡指令 {len(rebalance)} 笔，待 Gate 审批")

        if not events:
            body = "风险巡逻完成，今日无触发事件，组合状态平稳。"
        else:
            body = "；".join(events) + "。"

        if getattr(result, "skipped", False):
            body = "批次已在早前完成，当前为实时监控状态。"
        elif getattr(result, "errors", []):
            n = len(result.errors)
            body = f"批次完成，但出现 {n} 个步骤异常（详见 Engineering 区域）。" + body

        return "【今日动作】" + body

    # ── Section 3: 待审批事项（render-time only） ────────────────────────────

    def _section_pending(
        self,
        pending_gates: list,
        pending_approvals: list,
    ) -> str:
        n_gates     = len(pending_gates)
        n_approvals = len(pending_approvals)
        total       = n_gates + n_approvals

        if total == 0:
            return "【待审批事项】无，系统已处于静默状态。"

        parts: list[str] = []
        if n_gates:
            gate_labels = [pg.get("cycle_type", pg.get("gate", "")) for pg in pending_gates[:2]]
            parts.append(f"{n_gates} 个战略 Gate（{', '.join(gate_labels)}）")
        if n_approvals:
            # summarise by type
            type_counts: dict[str, int] = {}
            for pa in pending_approvals:
                t = pa.get("approval_type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            type_str = "、".join(f"{v} 个{k}" for k, v in type_counts.items())
            parts.append(f"{n_approvals} 个战术审批（{type_str}）")

        return "【待审批事项】" + "；".join(parts) + "，请在 Section A 处理。"

    # ── Public API ───────────────────────────────────────────────��──────────

    def build_batch_narrative(
        self,
        t_day: datetime.date,
        result: "BatchResult",
        regime: str = "unknown",
        p_risk_on: float = 0.5,
        regime_prev: str = "",
        regime_changed: bool = False,
    ) -> str:
        """
        Called by daily_batch.py after batch completion.
        Returns sections 1 + 2. Persisted to DailyBriefSnapshot.narrative.
        """
        s1 = self._section_market(
            t_day, regime, p_risk_on, regime_prev, regime_changed,
            include_yield_curve=True,
        )
        s2 = self._section_actions(result)
        return f"{s1}\n{s2}"

    def build_section_b(
        self,
        snap_dict: dict,
        pending_gates: list,
        pending_approvals: list,
    ) -> str:
        """
        Called at render time in orchestrator.py.
        Appends live Section 3 (pending actions) to the stored narrative.

        snap_dict: the dict returned by _load_brief_snapshot() in orchestrator.py
        """
        stored = snap_dict.get("narrative", "") or ""

        # Rebuild section 1 if stored narrative is stale / empty
        if not stored:
            regime      = snap_dict.get("regime", "unknown") or "unknown"
            p_risk_on   = float(snap_dict.get("p_risk_on", 0.5) or 0.5)
            regime_prev = snap_dict.get("regime_prev", "") or ""
            regime_changed = bool(snap_dict.get("regime_changed", False))
            # No BatchResult available here — produce minimal market-state section
            s1 = self._section_market(
                datetime.date.today(), regime, p_risk_on,
                regime_prev, regime_changed, include_yield_curve=True,
            )
            stored = s1

        s3 = self._section_pending(pending_gates, pending_approvals)
        return f"{stored}\n{s3}"

    def gate_summary_line(
        self,
        gate_label: str,
        dry_run_result: Optional[dict] = None,
    ) -> str:
        """
        One-line narrative for a Gate card in Section A.
        Example: "月度再平衡 — 换手率 34%，预估成本 14 bps，建议执行。"
        """
        if dry_run_result:
            turnover  = dry_run_result.get("turnover", 0)
            cost_bps  = dry_run_result.get("total_cost_bps", 0)
            n_trades  = len(dry_run_result.get("trades", []))
            return (
                f"{gate_label} — 换手率 {turnover:.0%}，"
                f"预估成本 {cost_bps:.1f} bps，{n_trades} 笔交易。"
            )
        return f"{gate_label} — 等待预览数据。"


# Module-level singleton
_builder = NarrativeBuilder()


def build_batch_narrative(
    t_day: datetime.date,
    result: "BatchResult",
    regime: str = "unknown",
    p_risk_on: float = 0.5,
    regime_prev: str = "",
    regime_changed: bool = False,
) -> str:
    """Convenience wrapper around NarrativeBuilder.build_batch_narrative()."""
    return _builder.build_batch_narrative(
        t_day, result, regime, p_risk_on, regime_prev, regime_changed
    )
