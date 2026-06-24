"""engine.agents.papers_curator.judgments_store — append-only jsonl
of FilterJudgment rows.

Kept separate from cache.jsonl (crawler-owned) so the two pipelines
can run independently — re-judging an existing cached paper does NOT
mutate the cache; reading "latest judgment per (source, source_id)"
is a deterministic latest-by-judged_ts pick.

File: data/papers_curator/judgments.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from engine.agents.papers_curator.filter import FilterJudgment

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
JUDGMENTS_PATH = _REPO_ROOT / "data" / "papers_curator" / "judgments.jsonl"


def load_judgments() -> list[FilterJudgment]:
    """Return ALL judgment rows. Order = chronological. For "latest
    per paper" use latest_by_paper()."""
    if not JUDGMENTS_PATH.is_file():
        return []
    out: list[FilterJudgment] = []
    with JUDGMENTS_PATH.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(FilterJudgment(**d))
            except Exception as exc:
                logger.warning("judgments.jsonl line %d malformed: %s",
                               ln_no, exc)
    return out


def latest_by_paper() -> dict[tuple[str, str], FilterJudgment]:
    """Return latest judgment per (source, source_id), picked by
    judged_ts. Used to determine which papers still need judging."""
    out: dict[tuple[str, str], FilterJudgment] = {}
    for j in load_judgments():
        key = (j.source, j.source_id)
        prev = out.get(key)
        if prev is None or j.judged_ts > prev.judged_ts:
            out[key] = j
    return out


def append_judgment(j: FilterJudgment) -> None:
    JUDGMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JUDGMENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(j.to_dict(), ensure_ascii=False) + "\n")
