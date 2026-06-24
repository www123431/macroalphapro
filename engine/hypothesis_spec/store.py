"""hypothesis_spec.store — persistent jsonl store for HypothesisSpec.

Append-only; one row per spec instance. Look up the LATEST spec for a
hypothesis via latest_for(source_hypothesis_id); look up by content via
by_spec_hash. Two specs with identical content yield identical
spec_hash, so a re-extraction of the same claim deduplicates naturally.

File: data/research_store/hypothesis_specs.jsonl

Storage rule (LdP §2): events are immutable; to "correct" a spec, emit
a new instance with bumped version + parent_spec_id pointing to the
prior. Consumers query LATEST per source_hypothesis_id by walking
ts-desc.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from engine.hypothesis_spec.schema import HypothesisSpec
from engine.hypothesis_spec.hash import spec_hash

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE_PATH = _REPO_ROOT / "data" / "research_store" / "hypothesis_specs.jsonl"
_WRITE_LOCK = threading.Lock()


def _read_all() -> list[dict]:
    if not _STORE_PATH.is_file():
        return []
    rows: list[dict] = []
    try:
        with _STORE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        logger.exception("hypothesis_spec.store: read failed")
    return rows


def append(spec: HypothesisSpec) -> str:
    """Persist a spec. Returns the computed spec_hash."""
    d = spec.to_dict()
    d["spec_hash"] = spec_hash(spec)
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with _STORE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
    return d["spec_hash"]


def all_specs() -> list[HypothesisSpec]:
    return [HypothesisSpec.from_dict(d) for d in _read_all()]


def latest_for(source_hypothesis_id: str) -> Optional[HypothesisSpec]:
    """Return the latest spec for a hypothesis id.

    F1 hotfix (2026-06-05): pre-fix used `s.version > best.version` to
    pick the latest. But every re-extract calls HypothesisSpec.new()
    which sets version=1 (a new spec_id, NOT a bump_version of the prior),
    so the version-comparison never fired and FIRST-wins silently became
    the rule. Same bug as direction_proposer._claim_type_map hotfix
    (commit 510e63a9). Use extraction.extracted_ts: same logical claim
    re-extracted with v2 prompt produces a later ts -> that wins.
    """
    best: Optional[HypothesisSpec] = None
    best_ts: str = ""
    for d in _read_all():
        if d.get("source_hypothesis_id") != source_hypothesis_id:
            continue
        s = HypothesisSpec.from_dict(d)
        ts = (s.extraction.extracted_ts or s.created_ts or "")
        if best is None or ts > best_ts:
            best = s
            best_ts = ts
    return best


def by_spec_hash(h: str) -> Optional[HypothesisSpec]:
    """Look up any spec whose spec_hash matches."""
    for d in _read_all():
        if d.get("spec_hash") == h:
            return HypothesisSpec.from_dict(d)
    return None


def specs_path() -> Path:
    return _STORE_PATH
