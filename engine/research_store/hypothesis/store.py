"""engine.research_store.hypothesis.store — jsonl persistence + cross-store check.

`save_hypothesis()` does the deep cross-store validation that
`Hypothesis.validate()` skips:

  1. source_paper_id resolves in papers_registry (latest version)
     AND that paper has fulltext_status == INGESTED

  2. Each chunk_id in source_chunk_ids resolves in papers_chroma
     (the actual collection, not just metadata)

  3. Each verbatim_quote.quote_text is a verbatim substring of the
     corresponding chunk's full text

If any check fails, save_hypothesis raises ValueError (or just logs
warnings if validate_strict=False).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from engine.research_store.hypothesis.schema import (
    Hypothesis, HypothesisReviewState,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
HYPOTHESES_PATH = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"


def _ensure_parent_dir() -> None:
    HYPOTHESES_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────── load / save ──────────────────────────────────


def load_hypotheses(path: Path | None = None) -> list[Hypothesis]:
    p = path or HYPOTHESES_PATH
    if not p.is_file():
        return []
    out: list[Hypothesis] = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Hypothesis.from_dict(json.loads(line)))
            except Exception as e:
                logger.error("malformed hypothesis at %s:%d — %s", p, i, e)
    return out


def save_hypothesis(hyp: Hypothesis, path: Path | None = None,
                    *, validate_strict: bool = True,
                    skip_cross_checks: bool = False) -> None:
    """Persist a Hypothesis. Runs schema self-validate + cross-store checks.

    Cross-store checks (skipped when skip_cross_checks=True, e.g. tests):
      - source_paper_id resolves + paper is fulltext_ingested
      - chunk_ids resolve in papers_chroma
      - verbatim_quote.quote_text is a substring of the chunk text

    Args:
      validate_strict:  if True, ValueError on any error (default)
      skip_cross_checks: if True, only run schema self-validate (testing)
    """
    self_errs = hyp.validate()
    cross_errs: list[str] = []
    if not skip_cross_checks:
        cross_errs = _cross_validate(hyp)

    all_errs = self_errs + cross_errs
    if all_errs and validate_strict:
        raise ValueError(
            f"Hypothesis validation failed for {hyp.hypothesis_id}: {all_errs}"
        )
    if all_errs:
        logger.warning("Hypothesis %s saved with validation issues: %s",
                       hyp.hypothesis_id, all_errs)

    p = path or HYPOTHESES_PATH
    if path is None:
        _ensure_parent_dir()
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(hyp.to_dict(), ensure_ascii=False) + "\n")


# ─────────────────────── cross-store validation ───────────────────────


def _cross_validate(hyp: Hypothesis) -> list[str]:
    """Deep checks that need other stores."""
    errs: list[str] = []

    # 1. source_paper_id resolves + INGESTED
    try:
        from engine.research_store.papers import (
            FulltextStatus, load_registry, latest_per_doi,
        )
        reg = load_registry()
        by_id = {e.paper_id: e for e in reg}
        paper = by_id.get(hyp.source_paper_id)
        if paper is None:
            # Try: maybe latest_per_doi resolves a version chain
            latest = latest_per_doi(reg)
            paper = next(
                (e for e in latest.values() if e.paper_id == hyp.source_paper_id),
                None,
            )
        if paper is None:
            errs.append(
                f"source_paper_id {hyp.source_paper_id} does not resolve in "
                f"papers_registry"
            )
        elif paper.fulltext_status != FulltextStatus.INGESTED:
            errs.append(
                f"source paper {hyp.source_paper_id} has fulltext_status="
                f"{paper.fulltext_status.value}, must be INGESTED"
            )
    except Exception as e:
        errs.append(f"papers_registry cross-check failed: {e}")

    # 2. chunk_ids resolve in papers_chroma + verbatim substring check
    try:
        from engine.research_store.red_lessons.papers_chroma import get_collection
        coll = get_collection()
        # Fetch chunks by chunk_id (chromadb get expects list[str])
        ids = list(hyp.source_chunk_ids) + [q.chunk_id for q in hyp.verbatim_quotes]
        ids = list(dict.fromkeys(ids))  # dedupe preserving order
        if ids:
            got = coll.get(ids=ids)
            got_ids = set(got.get("ids") or [])
            got_docs = dict(zip(got.get("ids") or [], got.get("documents") or []))
            missing = set(ids) - got_ids
            for m in missing:
                errs.append(f"chunk_id {m} not in papers_chroma")
            # Substring check on each verbatim quote
            for i, q in enumerate(hyp.verbatim_quotes):
                doc_text = got_docs.get(q.chunk_id, "")
                if doc_text and q.quote_text not in doc_text:
                    errs.append(
                        f"verbatim_quotes[{i}].quote_text is NOT a verbatim "
                        f"substring of chunk {q.chunk_id} text — "
                        f"possible LLM confabulation or whitespace mismatch"
                    )
    except Exception as e:
        # ChromaDB unreachable / collection missing — log but don't block
        # (allows hypothesis schema work to proceed without live ChromaDB)
        errs.append(f"papers_chroma cross-check could not run: {e}")

    return errs


# ─────────────────────── query helpers ────────────────────────────────


def find_by_id(hypothesis_id: str,
               hyps: Iterable[Hypothesis] | None = None
               ) -> Hypothesis | None:
    """Return the latest version of a hypothesis with given ID, or None."""
    if hyps is None:
        hyps = load_hypotheses()
    matches = [h for h in hyps if h.hypothesis_id == hypothesis_id]
    if not matches:
        return None
    return max(matches, key=lambda h: h.version)


def latest_per_paper(hyps: Iterable[Hypothesis]
                     ) -> dict[str, list[Hypothesis]]:
    """Group latest-version hypotheses by source_paper_id."""
    # First, collapse versions per hypothesis_id
    by_id: dict[str, Hypothesis] = {}
    for h in hyps:
        prior = by_id.get(h.hypothesis_id)
        if prior is None or h.version > prior.version:
            by_id[h.hypothesis_id] = h
    # Now group by paper
    by_paper: dict[str, list[Hypothesis]] = {}
    for h in by_id.values():
        by_paper.setdefault(h.source_paper_id, []).append(h)
    return by_paper


def hypotheses_resolvable(hypothesis_ids: Iterable[str]) -> bool:
    """Check that all given hypothesis_ids resolve in the store. Used
    by REDLesson save layer to enforce that tested_hypothesis_ids
    can't be dangling references."""
    hyps = load_hypotheses()
    known = {h.hypothesis_id for h in hyps}
    missing = set(hypothesis_ids) - known
    return not missing
