"""
engine/failure_attribution_agent.py — 失败归因自动标注 Agent
==============================================================
P1-7 任务的核心模块。每日批量扫描已验证但尚未归因的失败决策（accuracy_score < 0.5），
调用 LLM 输出结构化 failure_type + confidence + note，confidence ≥ 0.8 自动写入；
低于阈值的保持为"待人工审计"状态。

设计原则（量化金融视角）：
- 人工只判"为什么需要重新看"，"是什么类别"由模型在已审定 6 类 taxonomy 内分类
- confidence 阈值默认 0.80，对应 Bridgewater "machine-believable" 阈值
- failure_note 前缀 "[auto:conf=0.XX]" 永久保留可追溯证据，便于 governance review
- 任何解析失败 / model=None 路径 → 不写入，不引入 noise
- 与 set_failure_attribution() 同 6 类 taxonomy，避免多源不一致
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Optional

from engine.memory import (
    SessionFactory,
    DecisionLog,
    get_unattributed_failures,
    set_failure_attribution,
    _FAILURE_TYPES,
    _FAILURE_TYPE_LABELS,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIDENCE_THRESHOLD = 0.80
AUTO_NOTE_PREFIX = "[auto"


def _build_classification_prompt(record: dict) -> str:
    """构造单条记录的分类 prompt。仅注入决策时点信息 + 实际结果，无未来信息。"""
    type_lines = "\n".join(
        f"- {ft}: {_FAILURE_TYPE_LABELS.get(ft, ft)}"
        for ft in _FAILURE_TYPES
    )
    economic_logic     = (record.get("economic_logic") or "—")[:400]
    invalidation_conds = (record.get("invalidation_conditions") or "—")[:300]
    failure_mode_raw   = (record.get("failure_mode") or "")[:300]

    return f"""你是量化策略事后归因分析师。请将以下"已验证失败决策"归入 6 类 taxonomy 之一。

【6 类 taxonomy（互斥）】
{type_lines}

【决策上下文】
- 板块: {record.get("sector_name") or "—"}
- 方向: {record.get("direction") or "—"}
- 决策时置信度: {record.get("confidence_score") or "—"}
- 决策时宏观制度: {record.get("macro_regime") or "—"}
- 制度漂移: {"是" if record.get("regime_drifted") else "否"}
- 经济逻辑: {economic_logic}
- 失效条件（决策时声明）: {invalidation_conds}

【实际结果】
- accuracy_score: {record.get("accuracy_score") or 0:.2f}（< 0.5 视为失败）
- 既有 failure_mode 文本: {failure_mode_raw or "无"}

【要求】
1. 严格输出 JSON，无前缀无后缀文字
2. failure_type 必须是上述 6 类之一
3. confidence ∈ [0, 1]，反映你对此分类的把握
   - ≥ 0.80：证据指向单一类别，不存在合理替代解释
   - < 0.80：存在多类可能 / 上下文信息不足
4. note ≤ 60 字，给出 1 个具体证据要点
5. 严禁臆造数据；上下文未给出的事实不得作为依据

输出格式：
{{"failure_type": "...", "confidence": 0.85, "note": "..."}}
"""


def _parse_response(text: str) -> Optional[dict]:
    """从 LLM 文本中提取 JSON 对象，校验字段。失败返回 None。"""
    if not text:
        return None
    text = text.strip()
    candidates: list[str] = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group())
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        ft = obj.get("failure_type")
        if ft not in _FAILURE_TYPES:
            continue
        try:
            conf = float(obj.get("confidence", 0))
        except Exception:
            continue
        conf = max(0.0, min(1.0, conf))
        note = (obj.get("note") or "").strip()[:200]
        return {"failure_type": ft, "confidence": conf, "note": note}
    return None


def classify_failure(model, record: dict) -> Optional[dict]:
    """
    对单条 unattributed failure 调用 LLM，返回 dict 或 None。
    返回字段：failure_type / confidence / note。
    """
    if model is None:
        return None
    prompt = _build_classification_prompt(record)
    try:
        raw = model.generate_content(prompt).text
    except Exception as exc:
        logger.warning("classify_failure: model call failed for id=%s: %s",
                       record.get("id"), exc)
        return None
    parsed = _parse_response(raw)
    if parsed is None:
        logger.debug("classify_failure: parse failed for id=%s, raw=%r",
                     record.get("id"), (raw or "")[:200])
    return parsed


def auto_attribute_unattributed(
    model,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    min_age_days: int = 20,
    max_records: int = 50,
) -> dict:
    """
    每日批处理入口。扫描未归因失败 → 调用 LLM 分类 → 阈值以上自动写入。

    返回统计 dict：
    - scanned: 扫描条数
    - auto_attributed: 自动写入数（confidence ≥ threshold）
    - low_confidence: 低于阈值待人工审计数
    - parse_failures: 解析失败数（计入待人工）
    - skipped_model_none: model=None 时跳过数
    """
    stats = {
        "scanned": 0,
        "auto_attributed": 0,
        "low_confidence": 0,
        "parse_failures": 0,
        "skipped_model_none": 0,
    }
    if model is None:
        logger.info("auto_attribute_unattributed: model=None, skipping batch")
        return stats

    records = get_unattributed_failures(min_age_days=min_age_days)
    if not records:
        return stats
    records = records[:max_records]
    stats["scanned"] = len(records)

    for r in records:
        parsed = classify_failure(model, r)
        if parsed is None:
            stats["parse_failures"] += 1
            continue
        conf = parsed["confidence"]
        if conf < confidence_threshold:
            stats["low_confidence"] += 1
            continue
        note = f"{AUTO_NOTE_PREFIX}:conf={conf:.2f}] {parsed['note']}".strip()
        try:
            ok = set_failure_attribution(
                decision_id  = r["id"],
                failure_type = parsed["failure_type"],
                failure_note = note[:500],
            )
            if ok:
                stats["auto_attributed"] += 1
        except Exception as exc:
            logger.warning(
                "auto_attribute_unattributed: write failed id=%s: %s",
                r.get("id"), exc,
            )

    logger.info(
        "auto_attribute_unattributed: scanned=%d auto=%d low_conf=%d parse_fail=%d",
        stats["scanned"], stats["auto_attributed"],
        stats["low_confidence"], stats["parse_failures"],
    )
    return stats


def is_auto_attributed(failure_note: Optional[str]) -> bool:
    """判断 failure_note 是否为自动归因（用于 UI 展示与人工 audit 入口）。"""
    if not failure_note:
        return False
    return failure_note.lstrip().startswith(AUTO_NOTE_PREFIX)


def parse_auto_confidence(failure_note: Optional[str]) -> Optional[float]:
    """从 auto failure_note 中提取 confidence；非自动或解析失败返回 None。"""
    if not is_auto_attributed(failure_note):
        return None
    m = re.match(r"\[auto:conf=([\d.]+)\]", failure_note.lstrip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None
