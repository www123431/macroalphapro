"""engine.agents.papers_curator.summaries_store — append-only jsonl
of PaperSummary rows.

Schema-stable; latest-by-(source, source_id) wins on read. Re-summarizing
adds a new row (e.g. user requests re-check with updated context).

File: data/papers_curator/summaries.jsonl
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from engine.agents.papers_curator.summarizer import PaperSummary

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SUMMARIES_PATH = _REPO_ROOT / "data" / "papers_curator" / "summaries.jsonl"


def load_summaries() -> list[PaperSummary]:
    if not SUMMARIES_PATH.is_file():
        return []
    out: list[PaperSummary] = []
    with SUMMARIES_PATH.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # tuple field reconstruction
                d["risk_flags"] = tuple(d.get("risk_flags") or ())
                out.append(PaperSummary(**d))
            except Exception as exc:
                logger.warning("summaries.jsonl line %d malformed: %s",
                               ln_no, exc)
    return out


def latest_by_paper() -> dict[tuple[str, str], PaperSummary]:
    out: dict[tuple[str, str], PaperSummary] = {}
    for s in load_summaries():
        key = (s.source, s.source_id)
        prev = out.get(key)
        if prev is None or s.summarized_ts > prev.summarized_ts:
            out[key] = s
    return out


def append_summary(s: PaperSummary) -> None:
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARIES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
