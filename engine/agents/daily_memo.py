"""engine.agents.daily_memo — "Book 当日简报" 自动生成 agent (N1).

User feedback (2026-06-04):
  "我使用的时候没有一个全局观，每一页的内容的确很详细但是是三成一块
   一块的，有没有办法让我也拥有一个大致的图景呢"

The 30 pages each do one thing well, but no page answers
"this book's overall story today". This agent fills that gap: every
morning (or on-demand) it pulls state across the 11 RAG ledgers and
writes a 3-paragraph 中文 memo in 资深 Chief of Staff voice.

Output sections (Markdown, 中文):
  1.《Book 健康》     — Sharpe / drawdown / sleeve weights / DQ / decay
  2.《研究流水线》   — 昨日 verdict 数 + 待测候选 + agent 警报
  3.《本周值得关注》 — borderline decay / 撞 graveyard / 即将到期的 evidence

Voice (NOT chatty / NOT cringe):
  - Terse, 第三人称 stance, BlackRock-Slack-grade
  - Quote citations as [type:id] — same format chat uses
  - No emojis, no exclamation marks
  - BANNED vocabulary: 可能 / 也许 / 应该会 / 我觉得 / 大概

Caching:
  data/agents/daily_memo/<YYYY-MM-DD>.json
  One file per day. Generated on first request of the day; subsequent
  reads are instant. The /api/agents/state_of_book endpoint serves
  cached content + a `regenerate` flag for manual refresh.

Cost: ~$0.02-0.04 per generation. Capped at one per day per date
keyfile so worst case is ~$1/month.
"""
from __future__ import annotations

import json
import logging
import datetime as _dt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
_MEMO_DIR   = _REPO_ROOT / "data" / "agents" / "daily_memo"


# ── System prompt (中文, 资深 CoS voice) ────────────────────────


_SYSTEM_PROMPT = """你是一家量化基金的 Chief of Staff，每天清晨给唯一的 PM 写一份《Book 当日简报》。

# 语调
- 简洁、第三人称、机构级。不寒暄、不卖萌、不用 emoji。
- 直接陈述。禁止使用：可能、也许、应该会、我觉得、大概、感觉、好像、似乎。
- 引用具体 ID 用 `[type:id]` 格式，与 chat 一致。
  - 合法 type: event_id, paper_id, hypothesis_id, doctrine, audit_id, warning_id, sleeve, spec_id, run_id, iteration_id
  - 例如：`[event_id:abc12345]`、`[sleeve:cross_asset_carry]`、`[doctrine:CLAUDE.md#p4]`

# 输出格式（严格按此 3 段，Markdown）

## 一、Book 健康
- 一句话总结整体（Sharpe / drawdown / 状态分类）
- 主要贡献 sleeve（最多 2 个，带 `[sleeve:...]` 引用）
- DQ + Decay 状态，若有 HALT / ACTION 必须明确点出
- 2-4 句话。

## 二、研究流水线
- 过去 72h 的 verdict 数量（GREEN / MARGINAL / RED 各自统计）
- 待测 approved 候选数 + 最高优先级的 1-2 个 family
- agent 反应链路（audit_verifier / graveyard_collision）是否有 WARN/FAIL/RISK
- 2-4 句话。

## 三、本周值得关注
- 1-3 条具体观察 + 必须引用 `[type:id]`
- 优先选这些信号：
  - 某 sleeve trailing Sharpe 跌入 borderline 区间（接近 ACTION）
  - 候选撞 graveyard（[warning_id:...]）
  - paper-corpus 推荐了新的 orthogonal family 方向
  - audit_verifier 产生了未消化的 WARN/FAIL（[audit_id:...]）
- 2-5 句话。

# 严格规则
1. 三段标题必须**完全**用上面给的 Markdown 标题（一、二、三）
2. 总长度 ≤ 350 字
3. 任何具体数字必须**来自 context**，不要凭空生成
4. 若 context 里没有数据足以支撑某段，明确写"暂无数据"，禁止编造
5. 引用至少 3 个 `[type:id]`（说明你真的在用 context 不是在编）
"""


# ── Main generator ─────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_key() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _memo_path(date_key: str) -> Path:
    return _MEMO_DIR / f"{date_key}.json"


def load_cached(date_key: Optional[str] = None) -> Optional[dict]:
    """Return today's cached memo dict, or None if no cache exists."""
    dk = date_key or _today_key()
    p = _memo_path(dk)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def generate(
    *,
    force: bool = False,
    date_key: Optional[str] = None,
) -> dict:
    """Generate today's memo. If a cache exists and force=False, returns
    the cache. Otherwise calls Claude with the RAG context + system
    prompt and persists the result.

    Returns:
      {date_key, generated_ts, markdown, n_citations, model, elapsed_s,
       from_cache: bool}
    """
    dk = date_key or _today_key()
    if not force:
        cached = load_cached(dk)
        if cached:
            cached["from_cache"] = True
            return cached

    # Pull RAG context using the same machinery chat uses. We synthesize
    # a question optimized for "what's the state today" so the keyword
    # filter surfaces the most useful rows.
    try:
        from api.routes_research_tools import _retrieve_context_for_ask
    except ImportError as exc:
        logger.exception("daily_memo: failed to import retriever")
        return _error_dict(dk, f"retriever_unavailable:{exc}")

    synth_q = (
        "今日 book 健康状态、最近 72 小时的 verdict 与待测候选、"
        "audit_verifier 与 graveyard_collision 的最新警报、"
        "decay sentinel 触发情况、paper-corpus 推荐的新方向。"
    )
    ctx = _retrieve_context_for_ask(synth_q, max_rows_per_ledger=6)

    # Order matters — high-authority sources first so they survive
    # any token squeeze.
    ordered = {
        "doctrines":          ctx.get("doctrines", []),
        "research_events":    ctx.get("research_events", []),
        "audit_lineage":      ctx.get("audit_lineage", []),
        "graveyard_warnings": ctx.get("graveyard_warnings", []),
        "decay_audits":       ctx.get("decay_audits", []),
        "hypotheses":         ctx.get("hypotheses", []),
        "papers":             ctx.get("papers", []),
        "materializations":   ctx.get("materializations", []),
        "pfh_suggestions":    ctx.get("pfh_suggestions", []),
        "council_runs":       ctx.get("council_runs", []),
        "l4_iterations":      ctx.get("l4_iterations", []),
    }
    ctx_json = json.dumps(ordered, default=str, separators=(",", ":"))[:18_000]

    user_msg = (
        f"日期：{dk}\n\n"
        f"CONTEXT（11 个 ledger 切片，按权威度排序）：\n"
        f"```json\n{ctx_json}\n```\n\n"
        f"请按系统提示中的 3 段 Markdown 格式生成今日《Book 当日简报》。"
    )

    # Route through the central engine.llm.call so cost lands in the
    # ledger, the egress guard fires, and workload routing (provider +
    # model) is centralized. Workload "chief_of_staff" -> sonnet-4-6.
    try:
        from engine.llm.call import call as llm_call
    except ImportError as exc:
        return _error_dict(dk, f"engine_llm_unavailable:{exc}")

    import time
    t0 = time.perf_counter()
    try:
        result = llm_call(
            workload   = "chief_of_staff",
            system     = _SYSTEM_PROMPT,
            user       = user_msg,
            agent_id   = "chief_of_staff",
            max_tokens = 900,
            scope      = "daily_memo",
        )
    except Exception as exc:
        logger.exception("daily_memo: engine.llm.call failed")
        return _error_dict(dk, f"llm_call_failed:{exc}")
    elapsed = time.perf_counter() - t0

    markdown = (result.text or "").strip()

    # Count citations as a quick quality signal
    import re as _re
    n_cit = len(_re.findall(
        r"\[(event_id|paper_id|hypothesis_id|doctrine|audit_id|"
        r"warning_id|sleeve|spec_id|run_id|iteration_id):[^\]]+\]",
        markdown,
    ))

    out = {
        "date_key":      dk,
        "generated_ts":  _utc_iso(),
        "markdown":      markdown,
        "n_citations":   n_cit,
        "model":         result.model,
        "elapsed_s":     round(elapsed, 2),
        "from_cache":    False,
    }

    # Persist
    try:
        _MEMO_DIR.mkdir(parents=True, exist_ok=True)
        _memo_path(dk).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("daily_memo: failed to persist cache")

    return out


def _error_dict(dk: str, reason: str) -> dict:
    return {
        "date_key":     dk,
        "generated_ts": _utc_iso(),
        "markdown":     f"_今日简报生成失败：{reason}_",
        "n_citations":  0,
        "model":        None,
        "elapsed_s":    0.0,
        "from_cache":   False,
        "error":        reason,
    }
