"""engine.research_store.forward_vectors.store — jsonl persistence."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from engine.research_store.forward_vectors.schema import ForwardVectorV2

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
FORWARD_VECTORS_PATH = _REPO_ROOT / "data" / "research_store" / "forward_vectors.jsonl"


def load_forward_vectors(path: Path | None = None) -> list[ForwardVectorV2]:
    p = path or FORWARD_VECTORS_PATH
    if not p.is_file():
        return []
    out: list[ForwardVectorV2] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ForwardVectorV2.from_dict(json.loads(line)))
            except Exception as e:
                logger.error("malformed forward_vector at %s:%d — %s", p, i, e)
    return out


def save_forward_vector(fv: ForwardVectorV2, path: Path | None = None,
                        *, validate_strict: bool = True) -> None:
    errs = fv.validate()
    if errs and validate_strict:
        raise ValueError(f"ForwardVectorV2 validation failed for {fv.forward_vector_id}: {errs}")
    if errs:
        logger.warning("ForwardVectorV2 %s saved with issues: %s",
                       fv.forward_vector_id, errs)

    p = path or FORWARD_VECTORS_PATH
    if path is None:
        FORWARD_VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fv.to_dict(), ensure_ascii=False) + "\n")
