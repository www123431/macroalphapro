"""engine.agents.papers_curator — Employee A (papers auto-discovery).

Phase 1 substrate (2026-06-05): the upstream input feeder per the
4-employee roadmap [[project-four-employee-agentic-roadmap-2026-06-05]].

Substrate stack (built bottom-up):
  1. crawler — fetch latest papers from each source per source-specific
               parser; return common-schema PaperCandidate list. NO LLM.
  2. store   — dedup against cache.jsonl, append new entries. NO LLM.

NEXT (not in this commit):
  3. filter  — Deepseek 1-line judgment "is this a tradable-factor
               candidate?" Cheap (~$0.001/call) — drops ~80% before UI.
  4. UI      — /research/papers/incoming daily digest with ingest/skip
               buttons.

Today only ships 1-2; user reviews cache.jsonl manually until 3+4 land.
"""
from engine.agents.papers_curator.crawler import (
    PaperCandidate,
    crawl_all,
    crawl_arxiv_qfin,
)
from engine.agents.papers_curator.store import (
    CACHE_PATH,
    load_cache,
    save_new_candidates,
)
from engine.agents.papers_curator.filter import (
    FilterJudgment,
    judge_paper,
)
from engine.agents.papers_curator.judgments_store import (
    JUDGMENTS_PATH,
    append_judgment,
    latest_by_paper,
    load_judgments,
)
from engine.agents.papers_curator.summarizer import (
    PaperSummary,
    summarize_paper,
)
from engine.agents.papers_curator.summaries_store import (
    SUMMARIES_PATH,
    append_summary,
    latest_by_paper as latest_summary_by_paper,
    load_summaries,
)

__all__ = [
    "PaperCandidate",
    "crawl_all",
    "crawl_arxiv_qfin",
    "CACHE_PATH",
    "load_cache",
    "save_new_candidates",
    "FilterJudgment",
    "judge_paper",
    "JUDGMENTS_PATH",
    "append_judgment",
    "latest_by_paper",
    "load_judgments",
    "PaperSummary",
    "summarize_paper",
    "SUMMARIES_PATH",
    "append_summary",
    "latest_summary_by_paper",
    "load_summaries",
]
