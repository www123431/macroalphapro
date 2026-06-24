"""
Self-description docs for the Project History RAG (Tier 1 polish, 2026-05-07).

Hardcoded knowledge-base entries describing what each agentic capability /
page can answer. Indexed alongside live data (decision_log / spec /
amendments / approvals / audit / reflection) so meta-queries like:

    "你能帮我做什么？"
    "what can I ask?"
    "how do I find historical spec amendments?"
    "怎么看 reflexion memory？"

retrieve a real on-topic doc instead of forcing the LLM to synthesize a
nonsense answer from semantically-loose decision_log entries.

Why hardcoded
-------------
This file is the **canonical source-of-truth for project capability
descriptions**. It does NOT pull from runtime DB state because:
  1. Capability docs need to be stable / curated; LLM-generated descriptions
     of capabilities would drift each indexing.
  2. They need clear language for embedding-based retrieval; live spec
     text is too technical to match natural-language meta queries.

Maintenance
-----------
When a new agentic capability ships, add an entry here and re-run
`scripts/build_history_rag_index.py --reset` to embed it.
"""
from __future__ import annotations

from typing import Iterator

from engine.agents.history_rag.schema import IndexedDoc, SourceType


# Each entry covers one capability / surface. Keywords are deliberately
# rich + cross-language (中英) so embedding retrieval matches paraphrased
# meta queries.
_HELP_DOCS: list[dict] = [
    {
        "id":    "research_console",
        "title": "Research Console · 我能查什么 / 搜索 / 检索 / 查询能力",
        "deep_link": "pages/research_console.py",
        "text": (
            "本指引专门回答关于【搜索 / 检索 / 查询 / 问答】这一动作的问题——"
            "用户在 Research Console 里能查什么、能问什么、能搜什么。"
            "动词关键词：搜索 检索 查询 查找 找 查 问 问答 询问 检索 索引 找出 拉出 列出 "
            "search query find lookup ask answer search-engine retrieval. "
            "意图关键词：我能查什么 我能问什么 你能帮我查什么 我能搜什么 这里能查什么 "
            "这个搜索框能干嘛 怎么搜 怎么找 怎么查 chat 怎么用 "
            "what can I search · what can I ask · what queries are supported. \n\n"
            "本 Research Console（项目历史 RAG / 研究检索控制台）是一个 chat-style "
            "自然语言【检索】界面，专门用来在项目的完整内部历史里【搜】东西。"
            "支持以下 6 类检索话题：\n"
            "(1) 历史 hypothesis test 与 verdict —— 例如检索 '为什么 ship BAB'、"
            "'为什么 narrative overlay 被 reject'、'literature-conditional ship rule 是什么'。\n"
            "(2) Spec 哈希与 amendment ledger（修订记录） —— 例如检索 "
            "'列出 spec_b_plus_mass_fdr_search.md 的所有 amendment'、'HARKing detection 怎么工作'。\n"
            "(3) Pending approval（待审批）与 risk_control rationale —— 例如检索 "
            "'什么时候 regime overlay 被 disabled 为什么'、'最近的 risk control 决定有哪些'。\n"
            "(4) Reflexion-style 反思日志 —— 4-section CONTEXT/DECISION/OUTCOME/LESSON 结构，"
            "可以查 '某类决策的 lessons'、'agent 反思了什么'。\n"
            "(5) Tier R audit findings —— spec drift / hash chain break 等审计发现。\n"
            "(6) Decision log 决策日志 —— 各板块配置理由、宏观判断、信号解读。\n\n"
            "每个检索答案都会引用具体 source row（[C0]/[C1]），可以点回去验证原文。"
            "Chinese / English 均可。注意：本页面只【检索】历史，不能【创建】新记录"
            "（创建 spec 用 Spec Drafter 页；审批用 Operations 页）。"
        ),
    },
    {
        "id":    "spec_drafter",
        "title": "Spec Drafter · 起草 / 注册 / 创建新 spec / pre-registration",
        "deep_link": "pages/spec_drafter.py",
        "text": (
            "本指引专门回答关于【起草 / 注册 / 创建 / 写】新 spec 这一动作的问题——"
            "怎么把一个新假设变成 pre-registered spec 进 SpecRegistry。"
            "动词关键词：起草 注册 创建 写 新建 提案 锁定 freeze register draft create write. "
            "意图关键词：怎么注册新假设 怎么起草 spec 怎么写 spec 怎么开始一个新研究 "
            "新假设的流程 pre-registration 流程 spec_hash 怎么 lock auto-spec drafter "
            "how to register hypothesis · how to write a spec · pre-registration workflow. \n\n"
            "Auto-Spec Drafter（自动 spec 起草器）让 supervisor 用自然语言（中/英）描述假设，"
            "Gemini 2.5 Flash 自动起草完整 pre-registration spec，按项目现有 13-field 模板："
            "title、TL;DR、H0/H1、decision rule（含 SHIP/MARGINAL/FAIL 阈值）、"
            "multiple-testing impact（n_trials）、data requirements、predictions、"
            "implementation steps、success criteria、failure modes、out-of-scope、"
            "literature anchors、risks。\n\n"
            "Sakana AI Scientist 风格（Lu et al. 2024），但严格 PROPOSER 层——"
            "必须 supervisor review + 点 Stage+Register 才会 freeze spec_hash 到 SpecRegistry，"
            "EFFECTIVE_N_TRIALS 自动 +1。5 层 safety gate：forbidden-paths 检查、"
            "citation 必须引用 allow-list 内的真实学术论文、n_trials 诚实、"
            "no-overwrite、literature-conditional exemption 校验。"
            "注意：这里只【起草新 spec】；查询历史 spec 用 Research Console。"
        ),
    },
    {
        "id":    "reflection_journal",
        "title": "Reflection Journal 使用指引 · 反思日志 · agent 经验教训",
        "deep_link": "pages/reflection_journal.py",
        "text": (
            "用户常见触发短语：reflection 反思 agent 学到了什么 经验教训 "
            "lessons 4-section CONTEXT DECISION OUTCOME LESSON memory 记忆. \n\n"
            "Reflection Journal（反思日志）展示 Reflexion-style（Shinn 2023, NeurIPS）"
            "agent memory。每个 verified decision 后 agent 生成 4 段结构化反思："
            "[CONTEXT] 决策时的市场状态、regime；"
            "[DECISION] 预测了什么（方向、horizon、confidence）；"
            "[OUTCOME] 实际发生（active return、hit/miss）；"
            "[LESSON] 下次该怎么做。"
            "按 agent_id、hit/miss、决策日期过滤。当前正在累积；"
            "spec 目标 ≥50 reflections by 2026-09，calendar-bound。"
            "达到阈值后 RAG 会检索过去 lessons 反哺新决策上下文。"
        ),
    },
    {
        "id":    "auto_audit",
        "title": "Auto-Audit Loop 使用指引 · Tier R 三层治理 · 自动审计",
        "deep_link": "pages/auto_audit.py",
        "text": (
            "用户常见触发短语：审计 audit Tier R 治理 governance 哈希漂移 "
            "spec drift 红线 LLM-as-judge 0-LLM-in-evaluation 规则违规. \n\n"
            "Tier R Auto-Audit Loop（自动审计 / 三层治理）是 production-grade "
            "governance pipeline。Layer 0 = 18 条 deterministic 规则（11 critical 日审 + "
            "7 weekly slow-drift），检测 spec hash drift、hash chain break、cap violation、"
            "cash flow 不一致、sign convention drift 等。"
            "Layer 1 = LLM proposer（Gemini 2.5 Flash + response_schema），"
            "Layer 0 命中后起草修复方案。"
            "Layer 2 = 11 条 V-rules（V0-V10）safety gate，deterministic 校验提案"
            "（forbidden paths、diff size cap、risk × amendment kind 矩阵）。"
            "架构红线：0-LLM-in-evaluation——LLM 仅在 proposer 层，verdict deterministic。"
            "学术依据：Zheng 2023 LLM-as-judge bias 文献。"
            "本页面展示 audit run 历史、active findings、silenced rules、governance proposal 队列。"
        ),
    },
    {
        "id":    "falsification_chain",
        "title": "Falsification Chain · 证伪链 · 8 个 hypothesis test 历史",
        "deep_link": "docs/falsification_chain.md",
        "text": (
            "用户常见触发短语：证伪 falsification 假设测试 hypothesis test 历史 verdict "
            "ship reject marginal BAB COT FactorMAD narrative TSMOM EFA S1 B++ P3c "
            "做了哪些测试 测试结果 rejection. \n\n"
            "Falsification Chain（证伪链）是项目主要的科学产出：8 个 pre-registered hypothesis test，"
            "spec_hash + amendment_ledger 全留底。"
            "Stage 1: 横截面 narrative overlay（REJECT）。"
            "Stage 2: D1 aggregate risk gate（SOFT REJECT）。"
            "Stage 3: D1.1 power-aware re-evaluation（HARD REJECT）。"
            "Stage 4: FactorMAD Q1 LLM-mutated factor mining（0/24 promoted）。"
            "Stage 5: EFA three-piece uplift（REJECT）。"
            "Stage 6: S1 multi-window TSMOM（REJECT）。"
            "Stage 7: B++ Mass FDR 40-strategy weekly（MARGINAL——QL01 BAB 走 "
            "literature-conditional rule，依据 Frazzini-Pedersen 2014 + 5000 引用）。"
            "Stage 8: P3c COT-conditional BAB（REJECT——方向对 +1.38 Sharpe lift "
            "但 n_extreme=18 underpowered，BHY-adjusted p=0.43）。"
            "净 verdict：6 reject + 1 marginal + 1 underpowered-reject。"
            "每个 verdict 都有 spec_hash、amendment_log、bootstrap CI。"
        ),
    },
    {
        "id":    "ai_capabilities_overview",
        "title": "AI 助手总览 · agentic capabilities · 4 个 LLM proposer",
        "deep_link": "README.md",
        "text": (
            "用户常见触发短语：AI 助手 agentic capabilities LLM 能力 智能 自动化 "
            "项目用了什么 AI 概览 介绍下 总览. \n\n"
            "AI Assistants 区（AI 助手）汇集 4 个 LLM-at-proposer-layer 表面："
            "(1) Project History RAG（Research Console）——自然语言查询 "
            "decision_log / spec / amendments / approvals / audit。"
            "(2) Auto-Spec Drafter——自然语言假设 → 13-field pre-reg spec（Sakana AI Scientist 风格）。"
            "(3) Reflexion-style memory——每个 verified decision 后 agent 生成 4-section lesson。"
            "(4) Tier R Auto-Audit——deterministic rule 检测 drift 后 LLM 起草修复方案。"
            "全 4 项遵循 0-LLM-in-evaluation 红线："
            "LLM 仅 generate / propose / synthesize；deterministic gate 评分、校验、决策。"
            "总 LLM 成本约 $25/年；每个 feature 每天 $0.05 budget cap，"
            "用完自动 fallback 到 deterministic 路径。"
        ),
    },
    {
        "id":    "supervisor_pages",
        "title": "Supervisor 日常【操作】页面 · 看 NAV / 审批 / 业绩",
        "deep_link": "pages/executive_brief.py",
        "text": (
            "本指引专门回答关于【日常操作】的问题——supervisor 每天打开看 NAV / "
            "审批 / 看业绩走哪些页面。"
            "动词关键词：看 检查 审批 监控 操作 查看 view monitor approve. "
            "意图关键词：每天看什么 每天 routine 怎么审批 pending approval 在哪 "
            "看 NAV 看持仓 看业绩 portfolio journey 怎么看 supervisor 日常 daily check "
            "how do I monitor · how do I approve · NAV view. \n\n"
            "OPERATIONS 侧边栏组 5 个日常使用页面："
            "Brief（executive_brief）= 30 秒 supervisor 着陆页；展示 NAV、"
            "90 日 Sharpe、pending approvals、open audit findings、cron age、risk pulse、5 个 attention items。"
            "Positions（live_dashboard）= 当前组合持仓、target vs actual weight、按板块的 P&L attribution。"
            "Operations（orchestrator）= approval queue（entry / exit / risk_control / "
            "rebalance / spec_amendment），supervisor 带 rationale + category 审批。"
            "三层 narrative 模式（Tier 1 deterministic / Tier 2+3a+3b inline / Tier 3d analytics）。"
            "Performance = TWR / Sharpe / IR / β / TE；GIPS 2020 + Bacon Ch.2 KAT 合规。"
            "Portfolio Journey = realized NAV 轨迹（TWR / Simple HPR / MWR XIRR），"
            "strict no-counterfactual 契约。"
            "注意：这些是【操作页】不是【检索页】；查历史用 Research Console。"
        ),
    },
]


def iter_help_docs() -> Iterator[IndexedDoc]:
    """Yield self-description docs as IndexedDoc objects for build_index()."""
    for entry in _HELP_DOCS:
        yield IndexedDoc(
            doc_id      = f"system_help:{entry['id']}",
            source_type = SourceType.SYSTEM_HELP,
            source_id   = entry["id"],
            text        = entry["text"],
            title       = entry["title"],
            occurred_at = None,   # static knowledge; no date relevance
            metadata    = {
                "category":     "system_help",
                "deep_link_id": entry["id"],
            },
            deep_link   = entry.get("deep_link"),
        )
