"""engine.research_store.forward_vectors.review — PM review state.

Forward vectors are regenerated on-the-fly each API call (the generator
deterministically derives them from hypotheses + tested_lessons), so the
forward_vector_id is NOT stable across calls. Review state is therefore
keyed by source_hypothesis_id — which IS stable and uniquely identifies
"this specific claim from this specific paper".

State machine (matches user's 2026-06-04 "extracted → reviewed →
approved" mental model):

  extracted  — default (no human eyes yet)
  reviewed   — PM looked, marker for "I've seen this, still deciding"
  approved   — PM said yes — ready to test
  rejected   — PM said no — hide from default forward queue

Persistence: append-only jsonl. Latest entry per source_hypothesis_id
wins. The full history is preserved for audit (who toggled what when).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
REVIEWS_PATH = _REPO_ROOT / "data" / "research_store" / "forward_vector_reviews.jsonl"


class PMReviewStatus(str, Enum):
    EXTRACTED = "extracted"
    REVIEWED  = "reviewed"
    APPROVED  = "approved"
    REJECTED  = "rejected"


@_dc.dataclass(frozen=True)
class ForwardVectorReview:
    source_hypothesis_id: str
    status:               PMReviewStatus
    reviewed_ts:          str
    reviewed_by:          str
    note:                 str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_hypothesis_id": self.source_hypothesis_id,
            "status":               self.status.value,
            "reviewed_ts":          self.reviewed_ts,
            "reviewed_by":          self.reviewed_by,
            "note":                 self.note,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ForwardVectorReview":
        return cls(
            source_hypothesis_id = d["source_hypothesis_id"],
            status               = PMReviewStatus(d.get("status", "extracted")),
            reviewed_ts          = d["reviewed_ts"],
            reviewed_by          = d.get("reviewed_by", "user"),
            note                 = d.get("note", ""),
        )


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def append_review(review: ForwardVectorReview) -> None:
    """Append a review event (append-only — never overwrite)."""
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(review.to_dict(), ensure_ascii=False) + "\n")


def record_review(
    *,
    source_hypothesis_id: str,
    status:               str | PMReviewStatus,
    reviewed_by:          str = "user",
    note:                 str = "",
) -> ForwardVectorReview:
    """Convenience: build + persist a review event in one call."""
    s = PMReviewStatus(status) if isinstance(status, str) else status
    r = ForwardVectorReview(
        source_hypothesis_id = source_hypothesis_id,
        status               = s,
        reviewed_ts          = _utc_iso(),
        reviewed_by          = reviewed_by,
        note                 = note,
    )
    append_review(r)
    return r


def load_latest_reviews() -> dict[str, ForwardVectorReview]:
    """Latest review per source_hypothesis_id (newest wins).

    Tie-break: ISO timestamps have second resolution and rapid PM
    clicks (toggle approved → undo) can land in the same second, so
    file order is the secondary key — last line written wins. This
    matches the user's mental model ("the click I just made should
    take effect").

    Returns {} when the file does not exist or is empty.
    """
    if not REVIEWS_PATH.is_file():
        return {}
    latest: dict[str, ForwardVectorReview] = {}
    with REVIEWS_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s: continue
            try:
                r = ForwardVectorReview.from_dict(json.loads(s))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("malformed review at %s:%d — %s", REVIEWS_PATH, i, e)
                continue
            cur = latest.get(r.source_hypothesis_id)
            # >= so same-timestamp later-in-file wins.
            if cur is None or r.reviewed_ts >= cur.reviewed_ts:
                latest[r.source_hypothesis_id] = r
    return latest


def get_status(hypothesis_id: str,
               *, latest: dict[str, ForwardVectorReview] | None = None,
              ) -> PMReviewStatus:
    """Latest status for a hypothesis (defaults to EXTRACTED)."""
    cache = latest if latest is not None else load_latest_reviews()
    r = cache.get(hypothesis_id)
    return r.status if r else PMReviewStatus.EXTRACTED
