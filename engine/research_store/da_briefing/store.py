"""engine.research_store.da_briefing.store — jsonl persistence for DAVerdicts.

Append-only jsonl at VERDICTS_PATH. Each save_verdict() call runs
schema self-validate + cross-store validate (optional via flags).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from engine.research_store.da_briefing.cross_validate import (
    validate_verdict_cross_store,
)
from engine.research_store.da_briefing.schema import DAVerdict

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VERDICTS_PATH = _REPO_ROOT / "data" / "research_store" / "da_verdicts.jsonl"


def _ensure_parent_dir() -> None:
    VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_verdicts(path: Path | None = None) -> list[DAVerdict]:
    p = path or VERDICTS_PATH
    if not p.is_file():
        return []
    out: list[DAVerdict] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(DAVerdict.from_dict(json.loads(line)))
            except Exception as e:
                logger.error("malformed verdict at %s:%d — %s", p, i, e)
    return out


def save_verdict(verdict: DAVerdict, path: Path | None = None,
                 *, validate_strict: bool = True,
                 skip_cross_checks: bool = False) -> None:
    """Persist a DAVerdict. Runs schema self-validate + cross-store.

    Args:
      validate_strict:   if True, ValueError on any error (default)
      skip_cross_checks: skip the papers_chroma / papers_registry /
                         hypotheses cross-resolution. Used in tests.
    """
    self_errs = verdict.validate()
    cross_errs: list[str] = []
    if not skip_cross_checks:
        cross_errs = validate_verdict_cross_store(verdict)

    all_errs = self_errs + cross_errs
    if all_errs and validate_strict:
        raise ValueError(
            f"DAVerdict validation failed for {verdict.verdict_id}: {all_errs}"
        )
    if all_errs:
        logger.warning("DAVerdict %s saved with validation issues: %s",
                       verdict.verdict_id, all_errs)

    p = path or VERDICTS_PATH
    if path is None:
        _ensure_parent_dir()
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(verdict.to_dict(), ensure_ascii=False) + "\n")
