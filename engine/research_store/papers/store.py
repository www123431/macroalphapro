"""engine.research_store.papers.store — jsonl persistence for papers registry.

Append-only jsonl. Amendments are NEW lines with parent_paper_id + version
bumped. Latest-per-doi (or per-paper_id when no doi) wins.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from engine.research_store.papers.schema import PaperRegistryEntry

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REGISTRY_PATH = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"


def _ensure_parent_dir() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_registry(path: Path | None = None) -> list[PaperRegistryEntry]:
    """Load all entries from jsonl. Returns empty if absent. Skips corrupt lines."""
    p = path or REGISTRY_PATH
    if not p.is_file():
        return []
    out: list[PaperRegistryEntry] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(PaperRegistryEntry.from_dict(d))
            except Exception as e:
                logger.error("malformed paper entry at %s:%d — %s", p, i, e)
    return out


def save_entry(entry: PaperRegistryEntry, path: Path | None = None,
               *, validate_strict: bool = True) -> None:
    """Append an entry to the registry jsonl.

    Raises ValueError on validation failure when validate_strict=True.
    """
    errs = entry.validate()
    if errs and validate_strict:
        raise ValueError(f"PaperRegistryEntry validation failed: {errs}")
    if errs:
        logger.warning("PaperRegistryEntry %s saved with issues: %s",
                       entry.paper_id, errs)

    p = path or REGISTRY_PATH
    _ensure_parent_dir() if path is None else p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")


def find_by_doi(doi: str, entries: Iterable[PaperRegistryEntry] | None = None
                ) -> PaperRegistryEntry | None:
    """Return the LATEST version of an entry with this DOI, or None."""
    if not doi:
        return None
    if entries is None:
        entries = load_registry()
    matches = [e for e in entries if e.doi.lower() == doi.lower()]
    if not matches:
        return None
    return max(matches, key=lambda e: e.version)


def latest_per_doi(entries: Iterable[PaperRegistryEntry]
                   ) -> dict[str, PaperRegistryEntry]:
    """Group entries by DOI; return the highest-version per group.

    Entries without DOI are grouped under their paper_id (each unique).
    """
    by_key: dict[str, PaperRegistryEntry] = {}
    for e in entries:
        key = e.doi if e.doi else f"_no_doi::{e.paper_id}"
        prior = by_key.get(key)
        if prior is None or e.version > prior.version:
            by_key[key] = e
    return by_key
