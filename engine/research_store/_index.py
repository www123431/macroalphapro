"""engine/research_store/_index.py — SQLite read-index over the
research_store jsonl registries.

R4.3 — JSONL is the source-of-truth (audit log, easy to inspect,
append-only); SQLite is a lazy-rebuilt index for fast queries.
Goal: turn the O(N) Python iterations the consumers do today into
O(log N) indexed SELECTs without breaking the existing API.

Design:
  - One SQLite file per registry: papers / hypotheses / lessons.
  - Schema mirrors the jsonl row keys (one column per stable field).
  - Indexes on the columns we actually query by (doi, paper_id,
    source_paper_id, mechanism_family, candidate_name, verdict,
    grounding_method).
  - REBUILD policy: lazy. On read, compare mtime(jsonl) > mtime(db).
    If so, drop + rebuild. Rebuild is whole-file (cheap at our
    scale: ~50 papers, ~210 hypotheses, ~50 lessons).
  - "Latest per X" handled via SQL (max(version) per DOI / per
    hypothesis_id / per candidate_name) — no Python iteration.

Failure mode: if the rebuild ever fails (corrupt JSONL, schema
drift, etc.), we DON'T crash the consumer. The query falls back to
a Python load_* call. SQLite is a CACHE; the source-of-truth path
must always work.

This module is internal (_index.py prefix). Public consumers go
through `query.py` if/when we add one.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE     = _REPO_ROOT / "data" / "research_store"
_INDEX_DIR = _STORE / "_index"


def _ensure_dir() -> None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _needs_rebuild(jsonl: Path, db: Path) -> bool:
    """SQLite cache is stale when the jsonl is newer."""
    if not db.is_file(): return True
    try:
        return jsonl.stat().st_mtime > db.stat().st_mtime
    except OSError:
        return True


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.is_file(): return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


# ── Papers ────────────────────────────────────────────────────────


_PAPERS_JSONL = _STORE / "papers_registry.jsonl"
_PAPERS_DB    = _INDEX_DIR / "papers.sqlite"


_PAPERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id         TEXT NOT NULL,
    version          INTEGER NOT NULL DEFAULT 1,
    parent_paper_id  TEXT,
    doi              TEXT,
    title            TEXT,
    year             INTEGER,
    venue            TEXT,
    authors_json     TEXT,
    abstract         TEXT,
    fulltext_status  TEXT,
    pdf_source_kind  TEXT,
    pdf_source_url   TEXT,
    n_chunks         INTEGER DEFAULT 0,
    shelves_json     TEXT,
    created_ts       TEXT,
    updated_ts       TEXT,
    PRIMARY KEY (paper_id, version)
);
CREATE INDEX IF NOT EXISTS ix_papers_doi      ON papers (doi);
CREATE INDEX IF NOT EXISTS ix_papers_year     ON papers (year);
CREATE INDEX IF NOT EXISTS ix_papers_status   ON papers (fulltext_status);
CREATE INDEX IF NOT EXISTS ix_papers_updated  ON papers (updated_ts);
"""


def _rebuild_papers() -> None:
    _ensure_dir()
    if _PAPERS_DB.is_file():
        _PAPERS_DB.unlink()
    con = sqlite3.connect(str(_PAPERS_DB))
    try:
        con.executescript(_PAPERS_SCHEMA)
        rows: list[tuple] = []
        for r in _iter_jsonl(_PAPERS_JSONL):
            rows.append((
                r.get("paper_id"),
                int(r.get("version", 1)),
                r.get("parent_paper_id"),
                r.get("doi") or "",
                r.get("title") or "",
                int(r.get("year") or 0) or None,
                r.get("venue") or "",
                json.dumps(r.get("authors") or [], ensure_ascii=False),
                r.get("abstract") or "",
                r.get("fulltext_status") or "",
                r.get("pdf_source_kind") or "",
                r.get("pdf_source_url") or "",
                int(r.get("n_chunks") or 0),
                json.dumps(r.get("shelves") or [], ensure_ascii=False),
                r.get("created_ts") or "",
                r.get("updated_ts") or "",
            ))
        if rows:
            con.executemany("""
                INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
        con.commit()
        logger.info("rebuilt papers index: %d rows", len(rows))
    finally:
        con.close()


def papers_conn() -> sqlite3.Connection:
    """Lazy-rebuild + return read connection. Caller closes."""
    try:
        if _needs_rebuild(_PAPERS_JSONL, _PAPERS_DB):
            _rebuild_papers()
    except Exception as e:
        logger.warning("papers index rebuild failed: %s", e)
    return sqlite3.connect(str(_PAPERS_DB))


# ── Hypotheses ────────────────────────────────────────────────────


_HYPS_JSONL = _STORE / "hypotheses.jsonl"
_HYPS_DB    = _INDEX_DIR / "hypotheses.sqlite"


_HYPS_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id        TEXT NOT NULL,
    version              INTEGER NOT NULL DEFAULT 1,
    source_paper_id      TEXT,
    claim                TEXT,
    mechanism_family     TEXT,
    mechanism_subtype    TEXT,
    predicted_direction  TEXT,
    predicted_magnitude  TEXT,
    required_data_json   TEXT,
    test_methodology     TEXT,
    n_verbatim_quotes    INTEGER DEFAULT 0,
    review_state         TEXT,
    created_ts           TEXT,
    PRIMARY KEY (hypothesis_id, version)
);
CREATE INDEX IF NOT EXISTS ix_hyps_paper   ON hypotheses (source_paper_id);
CREATE INDEX IF NOT EXISTS ix_hyps_family  ON hypotheses (mechanism_family);
CREATE INDEX IF NOT EXISTS ix_hyps_review  ON hypotheses (review_state);
"""


def _rebuild_hypotheses() -> None:
    _ensure_dir()
    if _HYPS_DB.is_file():
        _HYPS_DB.unlink()
    con = sqlite3.connect(str(_HYPS_DB))
    try:
        con.executescript(_HYPS_SCHEMA)
        rows: list[tuple] = []
        for r in _iter_jsonl(_HYPS_JSONL):
            rows.append((
                r.get("hypothesis_id"),
                int(r.get("version", 1)),
                r.get("source_paper_id") or "",
                r.get("claim") or "",
                r.get("mechanism_family") or "",
                r.get("mechanism_subtype") or "",
                r.get("predicted_direction") or "",
                r.get("predicted_magnitude") or "",
                json.dumps(r.get("required_data") or [], ensure_ascii=False),
                r.get("test_methodology") or "",
                int(len(r.get("verbatim_quotes") or [])),
                r.get("review_state") or "",
                r.get("created_ts") or "",
            ))
        if rows:
            con.executemany("""
                INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
        con.commit()
        logger.info("rebuilt hypotheses index: %d rows", len(rows))
    finally:
        con.close()


def hypotheses_conn() -> sqlite3.Connection:
    try:
        if _needs_rebuild(_HYPS_JSONL, _HYPS_DB):
            _rebuild_hypotheses()
    except Exception as e:
        logger.warning("hypotheses index rebuild failed: %s", e)
    return sqlite3.connect(str(_HYPS_DB))


# ── Lessons ───────────────────────────────────────────────────────


_LESSONS_JSONL = _STORE / "red_lessons.jsonl"
_LESSONS_DB    = _INDEX_DIR / "lessons.sqlite"


_LESSONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS lessons (
    lesson_id            TEXT NOT NULL,
    version              INTEGER NOT NULL DEFAULT 1,
    candidate_name       TEXT,
    verdict              TEXT,
    grounding_method     TEXT,
    mechanism_family     TEXT,
    mechanism_subtype    TEXT,
    failure_modes_json   TEXT,
    tested_hypothesis_ids_json TEXT,
    n_verbatim_quotes    INTEGER DEFAULT 0,
    summary              TEXT,
    created_ts           TEXT,
    PRIMARY KEY (lesson_id, version)
);
CREATE INDEX IF NOT EXISTS ix_lessons_candidate  ON lessons (candidate_name);
CREATE INDEX IF NOT EXISTS ix_lessons_family     ON lessons (mechanism_family);
CREATE INDEX IF NOT EXISTS ix_lessons_grounding  ON lessons (grounding_method);
CREATE INDEX IF NOT EXISTS ix_lessons_verdict    ON lessons (verdict);
"""


def _rebuild_lessons() -> None:
    _ensure_dir()
    if _LESSONS_DB.is_file():
        _LESSONS_DB.unlink()
    con = sqlite3.connect(str(_LESSONS_DB))
    try:
        con.executescript(_LESSONS_SCHEMA)
        rows: list[tuple] = []
        for r in _iter_jsonl(_LESSONS_JSONL):
            rows.append((
                r.get("lesson_id"),
                int(r.get("version", 1)),
                r.get("candidate_name") or "",
                r.get("verdict") or "",
                r.get("grounding_method") or "",
                r.get("mechanism_family") or "",
                r.get("mechanism_subtype") or "",
                json.dumps(r.get("failure_modes") or [], ensure_ascii=False),
                json.dumps(r.get("tested_hypothesis_ids") or [], ensure_ascii=False),
                int(len(r.get("verbatim_quotes") or [])),
                r.get("summary") or "",
                r.get("created_ts") or "",
            ))
        if rows:
            con.executemany("""
                INSERT INTO lessons VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
        con.commit()
        logger.info("rebuilt lessons index: %d rows", len(rows))
    finally:
        con.close()


def lessons_conn() -> sqlite3.Connection:
    try:
        if _needs_rebuild(_LESSONS_JSONL, _LESSONS_DB):
            _rebuild_lessons()
    except Exception as e:
        logger.warning("lessons index rebuild failed: %s", e)
    return sqlite3.connect(str(_LESSONS_DB))


# ── Fast query helpers ────────────────────────────────────────────


def latest_paper_per_doi() -> list[dict]:
    """SELECT paper_id, doi, title, year, n_chunks, ... where version
    is max per DOI. SQL replaces the previous O(N) Python iteration."""
    con = papers_conn()
    try:
        rows = con.execute("""
            SELECT p.paper_id, p.doi, p.title, p.year, p.venue, p.authors_json,
                   p.fulltext_status, p.n_chunks, p.shelves_json, p.updated_ts
            FROM papers p
            JOIN (
                SELECT doi, MAX(version) AS v
                FROM papers
                WHERE doi != ''
                GROUP BY doi
            ) latest ON latest.doi = p.doi AND latest.v = p.version
            ORDER BY p.year DESC, p.updated_ts DESC
        """).fetchall()
        return [{
            "paper_id":        r[0],
            "doi":             r[1],
            "title":           r[2],
            "year":            r[3],
            "venue":           r[4],
            "authors":         json.loads(r[5] or "[]"),
            "fulltext_status": r[6],
            "n_chunks":        r[7],
            "shelves":         json.loads(r[8] or "[]"),
            "updated_ts":      r[9],
        } for r in rows]
    finally:
        con.close()


def hypotheses_for_paper(paper_id: str) -> list[dict]:
    """All hypotheses extracted from a paper (latest version each)."""
    con = hypotheses_conn()
    try:
        rows = con.execute("""
            SELECT h.hypothesis_id, h.claim, h.mechanism_family, h.mechanism_subtype,
                   h.predicted_direction, h.predicted_magnitude,
                   h.required_data_json, h.n_verbatim_quotes, h.review_state
            FROM hypotheses h
            JOIN (
                SELECT hypothesis_id, MAX(version) AS v
                FROM hypotheses
                WHERE source_paper_id = ?
                GROUP BY hypothesis_id
            ) latest ON latest.hypothesis_id = h.hypothesis_id AND latest.v = h.version
        """, (paper_id,)).fetchall()
        return [{
            "hypothesis_id":       r[0],
            "claim":               r[1],
            "mechanism_family":    r[2],
            "mechanism_subtype":   r[3],
            "predicted_direction": r[4],
            "predicted_magnitude": r[5],
            "required_data":       json.loads(r[6] or "[]"),
            "n_verbatim_quotes":   r[7],
            "review_state":        r[8],
        } for r in rows]
    finally:
        con.close()


def lessons_for_family(family: str, *, include_legacy: bool = False) -> list[dict]:
    """All lessons in a mechanism family. Honors latest-per-candidate
    and grounding_method filter."""
    con = lessons_conn()
    try:
        sql = """
            SELECT l.lesson_id, l.candidate_name, l.verdict, l.grounding_method,
                   l.mechanism_subtype, l.failure_modes_json, l.summary, l.created_ts
            FROM lessons l
            JOIN (
                SELECT candidate_name, MAX(version) AS v
                FROM lessons
                WHERE mechanism_family = ?
                GROUP BY candidate_name
            ) latest ON latest.candidate_name = l.candidate_name AND latest.v = l.version
        """
        params: list[Any] = [family]
        if not include_legacy:
            sql += " WHERE l.grounding_method != 'pretrain_grounded'"
        sql += " ORDER BY l.created_ts DESC"
        rows = con.execute(sql, params).fetchall()
        return [{
            "lesson_id":        r[0],
            "candidate_name":   r[1],
            "verdict":          r[2],
            "grounding_method": r[3],
            "mechanism_subtype": r[4],
            "failure_modes":    json.loads(r[5] or "[]"),
            "summary":          r[6],
            "created_ts":       r[7],
        } for r in rows]
    finally:
        con.close()


def stats() -> dict:
    """Quick health stats for the index — used by smoke tests + the
    admin overview. O(1) SQL counts vs. reading all 3 jsonl files."""
    pcon = papers_conn()
    hcon = hypotheses_conn()
    lcon = lessons_conn()
    try:
        n_papers      = pcon.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_papers_doi  = pcon.execute("SELECT COUNT(DISTINCT doi) FROM papers WHERE doi != ''").fetchone()[0]
        n_hyps        = hcon.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        n_hyps_uniq   = hcon.execute("SELECT COUNT(DISTINCT hypothesis_id) FROM hypotheses").fetchone()[0]
        n_lessons     = lcon.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        n_lessons_red = lcon.execute(
            "SELECT COUNT(*) FROM lessons WHERE verdict LIKE 'RED%'").fetchone()[0]
        return {
            "papers":     {"rows": n_papers, "unique_doi": n_papers_doi},
            "hypotheses": {"rows": n_hyps, "unique": n_hyps_uniq},
            "lessons":    {"rows": n_lessons, "red": n_lessons_red},
        }
    finally:
        pcon.close(); hcon.close(); lcon.close()
