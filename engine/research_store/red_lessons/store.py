"""engine.research_store.red_lessons.store — jsonl persistence for RED Lessons.

Single jsonl file at LESSONS_PATH. One lesson per line. Append-only for
new lessons; amendments are NEW lines with parent_lesson_id set.

DO NOT mutate prior lines. To "correct" a lesson, append a new version.
The latest version per candidate_name is the active one; prior versions
are kept for lineage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from engine.research_store.red_lessons.schema import REDLesson

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LESSONS_PATH = _REPO_ROOT / "data" / "research_store" / "red_lessons.jsonl"


def _ensure_parent_dir() -> None:
    LESSONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_lessons(path: Path | None = None) -> list[REDLesson]:
    """Load all RED Lessons from jsonl. Returns empty list if file absent.

    Does NOT dedupe by candidate_name — caller is responsible for picking
    the latest version if needed (use `latest_per_candidate` helper).
    """
    p = path or LESSONS_PATH
    if not p.is_file():
        return []
    out: list[REDLesson] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(REDLesson.from_dict(d))
            except Exception as e:
                logger.error("malformed lesson at %s:%d — %s", p, i, e)
                # Don't raise — corrupt single line shouldn't block the rest
    return out


def save_lesson(lesson: REDLesson, path: Path | None = None,
                *, validate_strict: bool = True) -> None:
    """Append a lesson to the jsonl store.

    Args:
        lesson:           REDLesson to persist.
        path:             override the default LESSONS_PATH (testing).
        validate_strict:  if True, raise on any validation error. If False,
                          log warnings and persist anyway (useful for
                          `proposed`-state lessons that aren't complete yet).
    """
    errs = lesson.validate()
    if errs and validate_strict:
        raise ValueError(f"REDLesson validation failed: {errs}")
    if errs:
        logger.warning("REDLesson %s saved with validation issues: %s",
                       lesson.lesson_id, errs)

    p = path or LESSONS_PATH
    _ensure_parent_dir() if path is None else p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(lesson.to_dict(), ensure_ascii=False) + "\n")


def latest_per_candidate(lessons: Iterable[REDLesson]) -> dict[str, REDLesson]:
    """Group lessons by candidate_name; return the highest-version one per group.

    Ties on version: break by created_ts (newest wins). Tie-break added
    2026-06-04 after end-to-end demo surfaced a real conflict — two
    independent v2 records for the same candidate (legacy paper_anchor
    pass + new paper_grounded demo). Without the tie-break, the FIRST
    v2 wins arbitrarily.
    """
    by_name: dict[str, REDLesson] = {}
    for L in lessons:
        prior = by_name.get(L.candidate_name)
        if prior is None:
            by_name[L.candidate_name] = L
            continue
        if L.version > prior.version:
            by_name[L.candidate_name] = L
            continue
        if L.version == prior.version and L.created_ts > prior.created_ts:
            by_name[L.candidate_name] = L
    return by_name
