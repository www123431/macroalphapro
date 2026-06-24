"""
engine/memory_curator.py — 记忆管理 Agent
==========================================
月末异步运行。扫描当月 DecisionLog，用二项检验识别行为模式，
对候选模式做 Benjamini-Hochberg FDR 校正，只将 confirmed 模式注入 SkillLibrary。

BH 校正门槛（P1-7 自动化标注层升级，2026-05-02）：
- n_supporting ≥ 100 AND BH 通过（FDR α=0.05）→ status="confirmed"，自动注入 SkillLibrary
- 其余                                          → status="tentative"，等待月度 review

阈值上调依据（project_llm_roadmap_critique 2026-05-01）：
- Niculescu-Mizil & Caruana 2005：校准 / 比例统计在 n<100 时分布尾部估计不稳
- BH α=0.05 是发表惯例；α=0.10 在自动化注入场景偏松，会污染 SkillLibrary

单元测试验证（在本文件末尾）：
  p_values = [0.001, 0.01, 0.05, 0.1, 0.5], alpha=0.05
  预期结果  = [True,  True, False, False, False]
"""
from __future__ import annotations

import datetime
import json
import logging
import math
from typing import TypedDict

from engine.memory import (
    SessionFactory,
    DecisionLog,
    MemoryCuratorReport,
    SkillLibrary,
    get_agent_confidence_weight,
)

logger = logging.getLogger(__name__)


class PatternCandidate(TypedDict):
    pattern_id:      str
    pattern_type:    str   # regime_bias / entry_timing / sector_preference / other
    description:     str
    n_supporting:    int
    n_contradicting: int
    p_value:         float
    bh_corrected:    bool
    status:          str   # tentative / confirmed


def _month_start(report_month: str) -> datetime.datetime:
    y, m = int(report_month[:4]), int(report_month[5:7])
    return datetime.datetime(y, m, 1)


def _month_end(report_month: str) -> datetime.datetime:
    y, m = int(report_month[:4]), int(report_month[5:7])
    if m == 12:
        return datetime.datetime(y + 1, 1, 1) - datetime.timedelta(seconds=1)
    return datetime.datetime(y, m + 1, 1) - datetime.timedelta(seconds=1)


def _binomial_p(n_supporting: int, n_total: int, p0: float = 0.5) -> float:
    """
    单尾二项检验：H0: p ≤ p0。
    返回 P(X ≥ n_supporting | n=n_total, p=p0)。
    使用正态近似（n≥10 时足够准确）。
    """
    if n_total < 10 or n_supporting == 0:
        return 1.0
    mu    = n_total * p0
    sigma = math.sqrt(n_total * p0 * (1 - p0))
    if sigma == 0:
        return 1.0
    z = (n_supporting - 0.5 - mu) / sigma  # continuity correction
    # 1 - Phi(z) via error function
    return 0.5 * (1 - math.erf(z / math.sqrt(2)))


def _bh_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR 校正。控制虚发现率在 alpha 以内。
    返回每个假设是否通过（bool 列表）。

    验证：
    >>> _bh_correction([0.001, 0.01, 0.05, 0.1, 0.5], alpha=0.05)
    [True, True, False, False, False]
    """
    if not p_values:
        return []
    k = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * k
    for rank, (orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / k) * alpha:
            results[orig_idx] = True
    return results


def _identify_pattern_candidates(decisions: list) -> list[PatternCandidate]:
    """
    从 DecisionLog 列表中识别候选行为模式。
    当前检测：
    1. regime_bias       — 特定制度下胜率显著高于 50%
    2. sector_preference — 特定板块胜率显著高于 50%
    3. entry_timing      — 高置信度决策（confidence_score≥80）胜率
    """
    candidates: list[PatternCandidate] = []
    verified = [d for d in decisions if d.verified and d.accuracy_score is not None]
    if len(verified) < 5:
        return candidates

    # ── 模式 1: regime_bias ───────────────────────────────────────────────────
    from collections import defaultdict
    regime_groups: dict[str, list[float]] = defaultdict(list)
    for d in verified:
        regime = d.macro_regime or "unknown"
        regime_groups[regime].append(d.accuracy_score)

    for regime, scores in regime_groups.items():
        n = len(scores)
        if n < 5:
            continue
        n_win = sum(1 for s in scores if s >= 0.5)
        p_val = _binomial_p(n_win, n)
        candidates.append(PatternCandidate(
            pattern_id      = f"regime_bias_{regime}",
            pattern_type    = "regime_bias",
            description     = f"制度 {regime} 下决策胜率 {n_win/n:.0%}（n={n}）",
            n_supporting    = n_win,
            n_contradicting = n - n_win,
            p_value         = p_val,
            bh_corrected    = False,
            status          = "tentative",
        ))

    # ── 模式 2: sector_preference ─────────────────────────────────────────────
    sector_groups: dict[str, list[float]] = defaultdict(list)
    for d in verified:
        sector = d.sector_name or "unknown"
        sector_groups[sector].append(d.accuracy_score)

    for sector, scores in sector_groups.items():
        n = len(scores)
        if n < 5:
            continue
        n_win = sum(1 for s in scores if s >= 0.5)
        p_val = _binomial_p(n_win, n)
        candidates.append(PatternCandidate(
            pattern_id      = f"sector_pref_{sector}",
            pattern_type    = "sector_preference",
            description     = f"板块 {sector} 决策胜率 {n_win/n:.0%}（n={n}）",
            n_supporting    = n_win,
            n_contradicting = n - n_win,
            p_value         = p_val,
            bh_corrected    = False,
            status          = "tentative",
        ))

    # ── 模式 3: entry_timing (高置信度决策) ────────────────────────────────────
    high_conf = [d for d in verified if (d.confidence_score or 0) >= 80]
    if len(high_conf) >= 5:
        n_hc     = len(high_conf)
        n_hc_win = sum(1 for d in high_conf if d.accuracy_score >= 0.5)
        p_val    = _binomial_p(n_hc_win, n_hc)
        candidates.append(PatternCandidate(
            pattern_id      = "entry_timing_high_conf",
            pattern_type    = "entry_timing",
            description     = f"高置信度（≥80）决策胜率 {n_hc_win/n_hc:.0%}（n={n_hc}）",
            n_supporting    = n_hc_win,
            n_contradicting = n_hc - n_hc_win,
            p_value         = p_val,
            bh_corrected    = False,
            status          = "tentative",
        ))

    return candidates


def _inject_confirmed_patterns_to_skill_library(
    confirmed: list[PatternCandidate],
    session,
    report_month: str,
) -> None:
    """将 confirmed 模式写入 SkillLibrary（upsert）。"""
    for c in confirmed:
        existing = session.query(SkillLibrary).filter(
            SkillLibrary.sector_name == c["pattern_id"],
        ).first()
        if existing:
            existing.description = c["description"]
            existing.updated_at  = datetime.datetime.utcnow()
        else:
            session.add(SkillLibrary(
                sector_name  = c["pattern_id"],
                macro_regime = "all",
                description  = c["description"],
                source       = "memory_curator",
                created_at   = datetime.datetime.utcnow(),
                updated_at   = datetime.datetime.utcnow(),
            ))
    logger.info(
        "memory_curator: injected %d confirmed patterns to SkillLibrary for %s",
        len(confirmed), report_month,
    )


def _generate_curator_summary(
    model,
    candidates: list[PatternCandidate],
    n_decisions: int,
    report_month: str,
) -> str:
    """LLM 生成月度摘要（失败时返回纯文字摘要）。"""
    confirmed  = [c for c in candidates if c["status"] == "confirmed"]
    tentative  = [c for c in candidates if c["status"] == "tentative"]

    fallback = (
        f"{report_month} 月度记忆摘要：扫描 {n_decisions} 条决策，"
        f"发现 {len(candidates)} 个候选模式，"
        f"其中 {len(confirmed)} 个通过 BH 校正（已注入 SkillLibrary），"
        f"{len(tentative)} 个暂列为 tentative（需更多样本）。"
    )
    if model is None or not candidates:
        return fallback

    try:
        conf_str = "\n".join(f"- [confirmed] {c['description']}" for c in confirmed) or "无"
        tent_str = "\n".join(f"- [tentative] {c['description']}" for c in tentative) or "无"
        prompt = f"""你是量化策略记忆管理员。用3-5句话总结以下月度分析结果，语言简洁专业，
不作超出数据的推断，对 tentative 模式保持谨慎态度。

月份：{report_month}
扫描决策数：{n_decisions}

已确认模式（已注入策略库）：
{conf_str}

候选模式（待积累更多样本）：
{tent_str}

请直接输出摘要文字，不需要任何前缀或标题。"""
        raw = model.generate_content(prompt).text
        return raw.strip() or fallback
    except Exception as e:
        logger.warning("memory_curator: LLM summary generation failed: %s", e)
        return fallback


def run_memory_curator(
    model,
    report_month: str,
    session,
) -> MemoryCuratorReport:
    """
    月末异步运行的主入口。

    步骤：
    1. 扫描当月 DecisionLog（已验证）
    2. 识别候选模式（二项检验）
    3. BH FDR 校正（α=0.10）
    4. n≥15 且 BH 通过 → confirmed，注入 SkillLibrary
    5. LLM 生成月度摘要
    6. 写 MemoryCuratorReport
    """
    # 检查是否已跑过
    existing = session.query(MemoryCuratorReport).filter_by(
        report_month=report_month
    ).first()
    if existing:
        logger.info("memory_curator: report for %s already exists, skipping", report_month)
        return existing

    decisions = session.query(DecisionLog).filter(
        DecisionLog.created_at >= _month_start(report_month),
        DecisionLog.created_at <= _month_end(report_month),
        DecisionLog.verified == True,  # noqa: E712
    ).all()

    n_decisions = len(decisions)
    candidates  = _identify_pattern_candidates(decisions)
    p_values    = [c["p_value"] for c in candidates]
    bh_results  = _bh_correction(p_values, alpha=0.05)

    confirmed_patterns: list[PatternCandidate] = []
    for i, candidate in enumerate(candidates):
        candidate["bh_corrected"] = bh_results[i] if i < len(bh_results) else False
        # P1-7: n≥100 + BH α=0.05 才自动注入；其余维持 tentative 等月度 review
        candidate["status"] = (
            "confirmed"
            if candidate["n_supporting"] >= 100 and candidate["bh_corrected"]
            else "tentative"
        )
        if candidate["status"] == "confirmed":
            confirmed_patterns.append(candidate)

    if confirmed_patterns:
        _inject_confirmed_patterns_to_skill_library(confirmed_patterns, session, report_month)

    summary = _generate_curator_summary(model, candidates, n_decisions, report_month)

    report = MemoryCuratorReport(
        report_month              = report_month,
        generated_at              = datetime.datetime.utcnow(),
        n_decisions_scanned       = n_decisions,
        patterns_found            = json.dumps(candidates,          ensure_ascii=False),
        bh_correction_passed      = json.dumps(
            [c["pattern_id"] for c in confirmed_patterns], ensure_ascii=False
        ),
        injected_to_skill_library = len(confirmed_patterns) > 0,
        report_summary            = summary,
    )
    session.add(report)
    try:
        session.commit()
        logger.info(
            "memory_curator: report written for %s (n=%d, confirmed=%d)",
            report_month, n_decisions, len(confirmed_patterns),
        )
    except Exception as e:
        session.rollback()
        logger.warning("memory_curator: DB commit failed: %s", e)

    return report
