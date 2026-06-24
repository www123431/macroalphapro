"""api/routes_global_search.py — P1-E global data search across the
research_store + library YAML inventory.

User-walkthrough audit (2026-06-04): Cmd-K only searches ROUTES, not
DATA. A quant typing "momentum" should see every sleeve, lesson,
paper, and hypothesis that mentions momentum — not just the route
list. Without this, mid-research lookups require navigating to the
right surface manually and using that surface's filter.

Backed by:
  - engine.research_store._index   (R4.3 SQLite read indexes)
  - library_inventory() YAML scan  (mechanism YAMLs)

Response shape (uniform across kinds for cleaner UI rendering):
  [
    { kind: "paper"      | "hypothesis" | "lesson" | "sleeve",
      id:   <stable id>,
      label: <primary display string>,
      sub:   <secondary line — author / family / etc>,
      href:  <UI route to drill into>,
      score: <relevance: 1.0 best, 0.0 worst>
    }, ...
  ]

Performance: O(N) Python LIKE per source today. Each source is small
(papers ~35, lessons ~106, hypotheses ~210, sleeves ~13) so total
~360 rows — completes in well under 50ms for any query. SQL LIKE
indexing could come later if N grows.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel


router = APIRouter(prefix="/api/search", tags=["search"])


class GlobalSearchHit(BaseModel):
    kind:  str       # paper | hypothesis | lesson | sleeve
    id:    str
    label: str
    sub:   str = ""
    href:  str
    score: float = 0.0


def _score_for(needle: str, text: str) -> float:
    """Trivial relevance: lower for less-specific matches.
    - exact match anywhere:  1.0
    - case-insensitive in:   0.7
    - token in any word:     0.4
    - else:                  0.0 (caller filters out 0)
    """
    if not text:
        return 0.0
    needle_lc = needle.lower().strip()
    if not needle_lc:
        return 0.0
    text_lc = text.lower()
    if needle in text:        return 1.0
    if needle_lc in text_lc:  return 0.7
    # token boundary check
    parts = text_lc.split()
    for p in parts:
        if p.startswith(needle_lc):
            return 0.4
    return 0.0


@router.get("/global", response_model=list[GlobalSearchHit])
def global_search(
    q: str = Query(..., min_length=2, max_length=200),
    limit: int = Query(40, ge=1, le=200),
) -> list[GlobalSearchHit]:
    """Search across papers, hypotheses, lessons, sleeves."""
    needle = q.strip()
    if not needle:
        return []

    hits: list[GlobalSearchHit] = []

    # ── Papers ─────────────────────────────────────────────────
    try:
        from engine.research_store._index import latest_paper_per_doi
        for p in latest_paper_per_doi():
            title  = (p.get("title") or "")
            authors = ", ".join(p.get("authors") or [])
            score = max(
                _score_for(needle, title)         * 1.0,
                _score_for(needle, p.get("doi") or "") * 1.0,
                _score_for(needle, authors)       * 0.9,
                _score_for(needle, p.get("venue") or "") * 0.7,
            )
            if score > 0:
                year = p.get("year") or ""
                sub = f"{year}{' · ' if year else ''}{authors[:80]}"
                hits.append(GlobalSearchHit(
                    kind="paper", id=p["paper_id"], label=title[:160],
                    sub=sub.strip(), href=f"/research/papers/{p['paper_id']}",
                    score=score,
                ))
    except Exception:
        pass

    # ── Hypotheses ─────────────────────────────────────────────
    try:
        from engine.research_store._index import hypotheses_conn
        import json as _j
        con = hypotheses_conn()
        try:
            for r in con.execute("""
                SELECT h.hypothesis_id, h.claim, h.mechanism_family,
                       h.mechanism_subtype, h.source_paper_id
                FROM hypotheses h
                JOIN (
                    SELECT hypothesis_id, MAX(version) AS v
                    FROM hypotheses
                    GROUP BY hypothesis_id
                ) latest ON latest.hypothesis_id = h.hypothesis_id AND latest.v = h.version
            """).fetchall():
                hid, claim, fam, sub_, ppid = r
                score = max(
                    _score_for(needle, claim or "") * 1.0,
                    _score_for(needle, fam or "")   * 0.7,
                    _score_for(needle, sub_ or "")  * 0.7,
                )
                if score > 0:
                    hits.append(GlobalSearchHit(
                        kind="hypothesis", id=hid,
                        label=(claim or "")[:160],
                        sub=f"{fam} · {sub_}".strip(" ·"),
                        href=f"/research/papers/{ppid}" if ppid else "/research/forward",
                        score=score,
                    ))
        finally:
            con.close()
    except Exception:
        pass

    # ── Lessons ────────────────────────────────────────────────
    try:
        from engine.research_store._index import lessons_conn
        con = lessons_conn()
        try:
            for r in con.execute("""
                SELECT l.lesson_id, l.candidate_name, l.verdict,
                       l.mechanism_family, l.mechanism_subtype, l.summary
                FROM lessons l
                JOIN (
                    SELECT candidate_name, MAX(version) AS v
                    FROM lessons GROUP BY candidate_name
                ) latest ON latest.candidate_name = l.candidate_name
                       AND latest.v = l.version
            """).fetchall():
                lid, cname, verd, fam, sub_, summ = r
                score = max(
                    _score_for(needle, cname or "") * 1.0,
                    _score_for(needle, summ or "")  * 0.8,
                    _score_for(needle, fam or "")   * 0.7,
                    _score_for(needle, sub_ or "")  * 0.7,
                )
                if score > 0:
                    hits.append(GlobalSearchHit(
                        kind="lesson", id=lid,
                        label=f"{verd} · {cname}"[:160],
                        sub=f"{fam} · {sub_}".strip(" ·"),
                        href=f"/research/lessons/{lid}",
                        score=score,
                    ))
        finally:
            con.close()
    except Exception:
        pass

    # ── Sleeves (library YAMLs) ────────────────────────────────
    try:
        from api.routes_research_tools import library_inventory
        inv = library_inventory().get("entries", [])
        for e in inv:
            score = max(
                _score_for(needle, e.get("id") or "")         * 1.0,
                _score_for(needle, e.get("family") or "")     * 0.8,
                _score_for(needle, e.get("parent_family") or "") * 0.6,
                _score_for(needle, e.get("purpose") or "")    * 0.5,
            )
            if score > 0:
                hits.append(GlobalSearchHit(
                    kind="sleeve", id=e["id"],
                    label=e["id"],
                    sub=f"{e.get('family','')} · {e.get('purpose','')}".strip(" ·"),
                    href=f"/lab/library/detail?id={e['id']}",
                    score=score,
                ))
    except Exception:
        pass

    hits.sort(key=lambda x: (-x.score, x.kind, x.label.lower()))
    return hits[:limit]
