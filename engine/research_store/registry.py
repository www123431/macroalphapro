"""engine.research_store.registry — controlled subject vocabulary.

Senior data-engineering pattern: every event must reference a registered
subject. No fuzzy auto-mapping (decision 2026-06-02 — fuzzy was rejected
in favor of strict + helpful errors, matching Stripe / Datadog / OTel
semantic-conventions discipline).

Storage: data/research_store/subjects.yaml + data/research_store/aliases.yaml.
Both are version-controlled in git so the taxonomy itself has audit history.

Concurrency: this is a single-user system; we use simple flock-free read+
write. If we ever go multi-user, swap for atomic-rename + advisory locks.
"""
from __future__ import annotations

import datetime as _dt
import difflib
import os
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import yaml

from engine.research_store.exceptions import SubjectNotRegisteredError
from engine.research_store.schema import SubjectType


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE_DIR = _REPO_ROOT / "data" / "research_store"
_SUBJECTS_PATH = _STORE_DIR / "subjects.yaml"
_ALIASES_PATH  = _STORE_DIR / "aliases.yaml"

_LOCK = threading.Lock()


@dataclass
class Subject:
    subject_id:    str
    subject_type:  str    # SubjectType.value
    family:        Optional[str] = None
    description:   str = ""
    canonical_paper_id: Optional[str] = None
    created_ts:    str = ""
    created_by:    str = ""


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir() -> None:
    _STORE_DIR.mkdir(parents=True, exist_ok=True)


def _read_subjects() -> dict[str, Subject]:
    if not _SUBJECTS_PATH.is_file():
        return {}
    with _SUBJECTS_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    out: dict[str, Subject] = {}
    for sid, payload in (raw.get("subjects") or {}).items():
        out[sid] = Subject(subject_id=sid, **(payload or {}))
    return out


def _write_subjects(subjects: dict[str, Subject]) -> None:
    _ensure_dir()
    payload = {"subjects": {
        sid: {k: v for k, v in asdict(s).items() if k != "subject_id" and v is not None and v != ""}
        for sid, s in sorted(subjects.items())
    }}
    with _SUBJECTS_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, allow_unicode=True)


def _read_aliases() -> dict[str, str]:
    """alias_id -> canonical_id."""
    if not _ALIASES_PATH.is_file():
        return {}
    with _ALIASES_PATH.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return dict(raw.get("aliases") or {})


def _write_aliases(aliases: dict[str, str]) -> None:
    _ensure_dir()
    with _ALIASES_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"aliases": dict(sorted(aliases.items()))},
                       fh, sort_keys=True, allow_unicode=True)


# ── Public API ─────────────────────────────────────────────────────


def register_subject(
    subject_id: str,
    subject_type: SubjectType | str,
    family: Optional[str] = None,
    description: str = "",
    canonical_paper_id: Optional[str] = None,
    created_by: str = "claude",
) -> Subject:
    """Register a new subject. Idempotent: re-registering the same subject_id
    with matching subject_type is a no-op. Differing subject_type raises."""
    if isinstance(subject_type, SubjectType):
        subject_type = subject_type.value
    with _LOCK:
        subjects = _read_subjects()
        if subject_id in subjects:
            existing = subjects[subject_id]
            if existing.subject_type != subject_type:
                raise ValueError(
                    f"subject_id {subject_id!r} already registered with "
                    f"subject_type={existing.subject_type!r}; cannot re-register "
                    f"with {subject_type!r}."
                )
            return existing
        s = Subject(
            subject_id=subject_id,
            subject_type=subject_type,
            family=family,
            description=description,
            canonical_paper_id=canonical_paper_id,
            created_ts=_utc_iso(),
            created_by=created_by,
        )
        subjects[subject_id] = s
        _write_subjects(subjects)
        return s


def register_alias(canonical: str, alias: str) -> None:
    """Explicit alias: query/emit with `alias` will be treated as `canonical`.
    Both must already be... actually, canonical must exist; alias is the
    pointer being registered. Avoids the fuzzy-mapping non-determinism
    while still letting historical names point to current canonical."""
    with _LOCK:
        subjects = _read_subjects()
        if canonical not in subjects:
            raise SubjectNotRegisteredError(canonical, suggestions=_suggest(canonical, list(subjects)))
        if alias in subjects:
            raise ValueError(
                f"alias {alias!r} is itself a registered subject; cannot "
                f"alias it to {canonical!r}. Remove the subject first if "
                f"merging is intended."
            )
        aliases = _read_aliases()
        aliases[alias] = canonical
        _write_aliases(aliases)


def resolve(subject_id: str) -> Optional[Subject]:
    """Return the Subject this id resolves to, or None if unknown.
    Tries direct lookup first, then alias table."""
    subjects = _read_subjects()
    if subject_id in subjects:
        return subjects[subject_id]
    aliases = _read_aliases()
    canonical = aliases.get(subject_id)
    if canonical and canonical in subjects:
        return subjects[canonical]
    return None


def require(subject_id: str) -> Subject:
    """Like resolve() but raises SubjectNotRegisteredError with hints
    when not found. emit.* helpers use this."""
    subj = resolve(subject_id)
    if subj is not None:
        return subj
    subjects = _read_subjects()
    candidates = list(subjects) + list(_read_aliases())
    raise SubjectNotRegisteredError(subject_id, suggestions=_suggest(subject_id, candidates))


def list_subjects(family: Optional[str] = None) -> list[Subject]:
    """List registered subjects, optionally filtered by family. Sorted by id."""
    subjects = _read_subjects()
    out = list(subjects.values())
    if family is not None:
        out = [s for s in out if s.family == family]
    out.sort(key=lambda s: s.subject_id)
    return out


def list_aliases() -> dict[str, str]:
    """Return alias → canonical mapping."""
    return _read_aliases()


def _suggest(query: str, candidates: list[str], n: int = 3) -> list[str]:
    """Suggest the n closest candidates by string similarity. Pure
    library code (difflib) — no fuzzy auto-mapping happens here, only
    suggestion for the error message."""
    if not candidates:
        return []
    return difflib.get_close_matches(query, candidates, n=n, cutoff=0.6)
