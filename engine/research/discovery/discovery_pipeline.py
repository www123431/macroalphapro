"""engine/research/discovery/discovery_pipeline.py — paper discovery pipeline.

Orchestrates: arXiv fetcher → LLM extractor → hygiene gates → review queue.

Pipeline stages:
  1. Fetch papers from arXiv q-fin for date range
  2. Pre-filter (deterministic): novelty heuristic, dedup vs existing library
  3. For each survivor: LLM extracts mechanism proposal
  4. Hygiene gates:
     - H3 equivalent: required_data tokens in inventory (deterministic)
     - H2 equivalent: family / parent_family cousin check vs library
     - Score threshold: confidence ≥ 0.5
  5. Survivors written to data/research/discovery_queue.jsonl for human review

Doctrine:
- NEVER auto-add to library
- LLM ADVISORY in extraction stage; deterministic gates AFTER
- Crossref verification on canonical paper happens at LIBRARY-ADD time (not here)
- Confidence threshold defaults to 0.5; can be tightened
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DISCOVERY_QUEUE = REPO_ROOT / "data" / "research" / "discovery_queue.jsonl"
DISCOVERY_BORDERLINE = REPO_ROOT / "data" / "research" / "discovery_borderline.jsonl"
DISCOVERY_LOG = REPO_ROOT / "data" / "research" / "discovery_log.jsonl"


_RECENT_DISCOVERY_CACHE: tuple[float, list[tuple[str, set[str]]]] = (0.0, [])


def _recent_discovery_titles(days: int = 90) -> list[tuple[str, set[str]]]:
    """Load (title, token-set) pairs from discovery_log.jsonl for the
    last N days. Cached for 5 min to avoid repeated file reads in a
    single batch run."""
    import datetime as _dt
    import time
    global _RECENT_DISCOVERY_CACHE
    now = time.time()
    if now - _RECENT_DISCOVERY_CACHE[0] < 300 and _RECENT_DISCOVERY_CACHE[1]:
        return _RECENT_DISCOVERY_CACHE[1]
    if not DISCOVERY_LOG.exists():
        _RECENT_DISCOVERY_CACHE = (now, [])
        return []
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
    out = []
    for line in DISCOVERY_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = rec.get("ts") or rec.get("timestamp_utc") or ""
        try:
            ts_dt = _dt.datetime.fromisoformat(ts.rstrip("Z"))
        except ValueError:
            continue
        if ts_dt < cutoff:
            continue
        title = (rec.get("title") or "").lower()
        if not title:
            continue
        tokens = set(title.split()) - {"a", "the", "of", "and", "in",
                                              "on", "for", "to"}
        if tokens:
            out.append((title, tokens))
    _RECENT_DISCOVERY_CACHE = (now, out)
    return out


def _is_dup_against_recent_discovery(title: str, *, days: int = 90) -> bool:
    """True if a paper with token-overlap >= 0.7 was processed within
    the last `days` days (cross-cron dedup)."""
    if not title:
        return False
    tokens = set(title.lower().split()) - {"a", "the", "of", "and",
                                                  "in", "on", "for", "to"}
    if not tokens:
        return False
    for _title, hist_tokens in _recent_discovery_titles(days=days):
        if not hist_tokens:
            continue
        overlap = len(tokens & hist_tokens) / max(len(tokens | hist_tokens), 1)
        if overlap >= 0.7:
            return True
    return False


def _is_dup_against_library(extraction, library_titles: set[str]) -> bool:
    """Crude dedup: arxiv title token-overlap >= 0.7 with any library mechanism's
    title in canonical_paper master index."""
    if not extraction.title:
        return False
    extraction_tokens = set(extraction.title.lower().split())
    extraction_tokens -= {"a", "the", "of", "and", "in", "on", "for", "to"}
    if not extraction_tokens:
        return False
    for title in library_titles:
        title_tokens = set(title.lower().split())
        title_tokens -= {"a", "the", "of", "and", "in", "on", "for", "to"}
        if not title_tokens:
            continue
        overlap = len(extraction_tokens & title_tokens) / max(
            len(extraction_tokens | title_tokens), 1
        )
        if overlap >= 0.7:
            return True
    return False


def _load_library_titles() -> set[str]:
    """Pull titles from _canonical_papers_tier1_2.yaml for dedup."""
    import yaml
    master_index_path = (REPO_ROOT / "data" / "research" / "mechanism_library"
                          / "_canonical_papers_tier1_2.yaml")
    if not master_index_path.exists():
        return set()
    try:
        master = yaml.safe_load(master_index_path.read_text(encoding="utf-8"))
        papers = master.get("papers", {})
        return {p.get("title", "") for p in papers.values() if p.get("title")}
    except Exception as exc:
        logger.warning("library title load failed: %s", exc)
        return set()


def _check_data_inventory(tokens: list[str]) -> tuple[bool, list[str]]:
    """Returns (all_present, missing_tokens). Deterministic check."""
    if not tokens:
        return False, []   # empty required_data is a red flag itself
    try:
        from engine.research.hygiene_tools import DATA_INVENTORY
        missing = [t for t in tokens if t not in DATA_INVENTORY]
        return not missing, missing
    except Exception:
        return False, list(tokens)


def _check_family_cousin(extraction, library_families: dict[str, list[str]]) -> dict:
    """Check if proposed family is already RED in our library (cousin reject).

    Returns: {is_cousin_of_red: bool, existing_examples: list[str]}.
    """
    fam = extraction.family_guess
    if fam == "unknown" or not fam:
        return {"is_cousin_of_red": False, "existing_examples": []}
    examples = library_families.get(fam, [])
    return {
        "is_cousin_of_red": len(examples) > 0,
        "existing_examples": examples,
    }


def _load_library_families() -> dict[str, list[str]]:
    """Load { family_id: [mechanism_id...] } where mechanism status is RED or
    DEPLOYED. Used to flag cousin proposals."""
    import yaml
    out: dict[str, list[str]] = {}
    lib_dir = REPO_ROOT / "data" / "research" / "mechanism_library"
    if not lib_dir.exists():
        return out
    for fp in sorted(lib_dir.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            entry = yaml.safe_load(fp.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        family = entry.get("family")
        status = entry.get("status_in_our_book")
        if family and status in ("RED", "DEPLOYED"):
            out.setdefault(family, []).append(entry.get("id", fp.stem))
    return out


def process_paper(
    paper_row: dict,
    *,
    use_llm: bool = True,
    confidence_threshold: float = 0.5,
    library_titles: set[str] | None = None,
    library_families: dict[str, list[str]] | None = None,
    use_llm_rescue: bool = False,
) -> dict:
    """Run one paper through the full pipeline.

    Returns: structured outcome dict; NEVER raises.
    """
    from engine.research.discovery.paper_extractor import extract_from_paper

    library_titles = library_titles if library_titles is not None else _load_library_titles()
    library_families = (library_families if library_families is not None
                          else _load_library_families())

    arxiv_id = paper_row.get("arxiv_id") or paper_row.get("source_id", "")
    title    = paper_row.get("title", "")
    abstract = paper_row.get("abstract", "")
    if not arxiv_id or not title:
        return {"arxiv_id": arxiv_id, "stage": "input_invalid",
                "verdict": "skip", "reason": "missing arxiv_id or title"}

    # Stage -1: GRAVEYARD AUTO-ROUTING for replication-source venues
    # (Critical Finance Review etc). These papers carry critical-replication
    # signal — route to library_negative_evidence side-channel BEFORE any
    # other processing, never waste LLM cost on them as "candidates".
    # Per [[project-senior-pipeline-roadmap-2026-05-30]] roadmap #3.
    if paper_row.get("graveyard_routing") == "auto_negative_evidence":
        return {
            "arxiv_id":     arxiv_id,
            "title":        title,
            "submitted_date": paper_row.get("submitted_date"),
            "stage":        "graveyard_auto_route",
            "verdict":      "route_to_negative_evidence",
            "reason":       (f"venue '{paper_row.get('venue', '')}' has "
                              f"graveyard_routing=auto_negative_evidence; "
                              f"paper queued for library/red review"),
            "graveyard_routing": "auto_negative_evidence",
            "ts":           datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    # Stage 0: EX-ANTE CREDIBILITY FILTER (senior roadmap #1)
    # Cheap deterministic features → skip LLM extraction cost on
    # papers that fail venue/sample/novelty/cite-rate baseline.
    # ADVISORY only — fully auditable score + per-feature explanations
    # logged to discovery_log for spot-checking.
    try:
        from engine.research.discovery.credibility_scorer import (
            PaperMetadata, score_paper,
        )
        cred = score_paper(PaperMetadata(
            title=title,
            abstract=abstract,
            authors=paper_row.get("authors", ""),
            venue=paper_row.get("venue", "") or paper_row.get("source", ""),
            source=paper_row.get("source", ""),
            submitted_date=paper_row.get("submitted_date"),
            doi=paper_row.get("doi"),
            arxiv_id=arxiv_id,
            affiliations=paper_row.get("affiliations", ""),
        ))
    except Exception as exc:
        logger.warning("credibility scorer failed (continuing): %s", exc)
        cred = None
    if cred is not None:
        cred_dict = cred.to_dict()
    else:
        cred_dict = {"error": "scorer failed; bypassed"}
    if cred is not None and not cred.passes_filter:
        return {
            "arxiv_id":     arxiv_id,
            "title":        title,
            "submitted_date": paper_row.get("submitted_date"),
            "stage":        "credibility_filter",
            "verdict":      "skip",
            "reason":       (f"credibility score {cred.score:.3f} < "
                              f"threshold {cred.threshold:.2f}"),
            "credibility":  cred_dict,
            "ts":           datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

    # Stage 1: extract via LLM
    extraction = extract_from_paper(arxiv_id, title, abstract, use_llm=use_llm)
    if extraction is None:
        return {"arxiv_id": arxiv_id, "stage": "llm_failed",
                "verdict": "skip", "reason": "LLM extraction failed"}

    out = {
        "arxiv_id":       arxiv_id,
        "title":          title,
        "submitted_date": paper_row.get("submitted_date"),
        "extraction":     extraction.to_dict(),
        "credibility":    cred_dict,
        "ts":             datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # Stage 2: dedup against library + discovery_log (last 90d).
    # Senior 漏洞 5: previously only checked library titles. Now also
    # checks recent discovery_log so the same paper appearing on
    # arxiv then later in JFE doesn't pay LLM cost twice.
    if _is_dup_against_library(extraction, library_titles):
        out.update({"stage": "dedup", "verdict": "skip",
                     "reason": "title token-overlap >= 0.7 with library"})
        return out
    if _is_dup_against_recent_discovery(title, days=90):
        out.update({"stage": "dedup_recent", "verdict": "skip",
                     "reason": ("title matches a paper processed in "
                                  "the last 90 days (cross-cron dedup)")})
        return out

    # Stage 3: TWO-TIER confidence routing (senior 2026-05-30 redesign).
    # Per [[project-e2e-smoke-v3-funnel-findings-2026-05-30]]:
    # apply family-aware threshold + LLM family bonus + tier classification.
    # Papers below borderline floor → skip. Borderline → keep going through
    # pipeline but mark for borderline_review verdict. Review-tier → normal.
    from engine.research.discovery.family_thresholds import (
        explain_routing,
    )
    effective_confidence = extraction.confidence

    # OPT-IN: LLM rescue for papers that would otherwise skip. Only fires
    # when use_llm_rescue=True AND the regex confidence is in the dead
    # zone (< borderline floor). The LLM extracts 7 boolean features and
    # credits the regex slot for each one it finds. ~$0.0008 per rescue.
    if use_llm_rescue and effective_confidence < 0.30:
        try:
            from engine.research.discovery.llm_feature_extractor import (
                compute_hybrid_confidence,
            )
            hybrid = compute_hybrid_confidence(
                title, abstract,
                required_data_tokens=extraction.required_data_tokens,
                family_guess=extraction.family_guess,
                enable_llm=True,
            )
            out["llm_rescue"] = {
                "base_confidence":   hybrid["base_confidence"],
                "hybrid_confidence": hybrid["hybrid_confidence"],
                "rescued_features":  hybrid["rescued_features"],
                "llm_cost_usd":      hybrid.get("llm_cost_usd", 0.0),
            }
            if hybrid["llm_extraction_ok"] and hybrid["hybrid_confidence"] > effective_confidence:
                effective_confidence = hybrid["hybrid_confidence"]
        except Exception as exc:
            logger.warning("llm rescue failed (continuing): %s", exc)
            out["llm_rescue"] = {"error": str(exc)}

    routing_info = explain_routing(
        effective_confidence, extraction.family_guess,
    )
    out["routing"] = routing_info
    if routing_info["routing"] == "skip":
        out.update({"stage": "low_confidence", "verdict": "skip",
                     "reason": (f"adjusted confidence "
                                  f"{routing_info['adjusted_confidence']:.2f} "
                                  f"< borderline floor "
                                  f"{routing_info['borderline_floor']:.2f}")})
        return out
    # Stage 3 doesn't skip borderline papers — they keep flowing through
    # the rest of the pipeline so graveyard / hygiene checks still apply.
    # The final verdict (queue_for_review vs borderline_review) is set
    # at the end of process_paper based on out['routing']['routing'].

    # Stage 4: data inventory check (deterministic)
    all_present, missing = _check_data_inventory(extraction.required_data_tokens)
    out["data_check"] = {"all_present": all_present, "missing": missing}
    if not all_present:
        out.update({"stage": "data_unavailable", "verdict": "review_with_caveat",
                     "reason": f"required data missing: {missing}"})
        return out

    # Stage 5: GRAVEYARD CHECK — comprehensive vs library RED + gate_runs RED
    # + negative_evidence + discovery rejected. Per user 2026-05-30:
    # "保证我们不会走上已经走过的死路".
    try:
        from engine.research.graveyard import (
            CandidateInfo, check_against_graveyard,
        )
        candidate = CandidateInfo(
            title=title,
            family=(extraction.family_guess
                     if extraction.family_guess != "unknown" else None),
            parent_family=(extraction.parent_family_guess
                             if extraction.parent_family_guess != "unknown" else None),
            required_data=extraction.required_data_tokens,
            economics_text=extraction.economic_intuition,
            arxiv_id=arxiv_id,
        )
        gv_match = check_against_graveyard(candidate)
        out["graveyard_check"] = gv_match.to_dict()
        if gv_match.recommendation == "block":
            out.update({"stage": "graveyard_block", "verdict": "skip",
                         "reason": f"graveyard match (block): {gv_match.explanation}"})
            return out
        if gv_match.recommendation == "warn":
            out.update({"stage": "graveyard_warn", "verdict": "review_with_caveat",
                         "reason": f"graveyard match (warn): {gv_match.explanation}"})
            return out
    except Exception as exc:
        logger.warning("graveyard check failed (continuing): %s", exc)
        out["graveyard_check"] = {"error": str(exc)}

    # Stage 6: legacy family cousin check (library-only; preserved for back-compat)
    cousin_info = _check_family_cousin(extraction, library_families)
    out["cousin_check"] = cousin_info
    if cousin_info["is_cousin_of_red"]:
        out.update({"stage": "family_cousin", "verdict": "review_with_caveat",
                     "reason": (f"family {extraction.family_guess!r} already has "
                                  f"RED/DEPLOYED entries: {cousin_info['existing_examples']}")})
        return out

    # Stage 7: META-LEARNER ADVISORY (Phase 6.5 Tier 1)
    # Attach per-family Beta-Binomial base rate as advisory annotation.
    # Per [[project-meta-learner-design-2026-05-30]] STRICT RED LINE:
    # this is INFORMATIONAL only, never alters routing or verdict —
    # the reviewer sees the prior odds when deciding promote vs skip.
    try:
        from engine.research.meta_learner import annotate_candidate
        family = extraction.family_guess or "unknown"
        out["meta_learner_advisory"] = annotate_candidate(family)
    except Exception as exc:
        logger.warning("meta-learner advisory failed (continuing): %s", exc)
        out["meta_learner_advisory"] = {"error": str(exc)[:200]}

    # Passed all gates → routing determines verdict
    # Senior 2026-05-30: borderline-tier papers go to a separate queue
    # so they're surfaced for human spot-check without crowding the
    # primary review queue.
    routing_tier = out.get("routing", {}).get("routing", "review")
    if routing_tier == "borderline":
        out.update({"stage": "queued_borderline",
                     "verdict": "borderline_review",
                     "reason": (f"borderline confidence "
                                  f"{out['routing']['adjusted_confidence']:.2f} "
                                  f"(family={out['routing']['family']}, "
                                  f"threshold={out['routing']['family_threshold']:.2f}); "
                                  f"queued for human spot-check")})
    else:
        out.update({"stage": "queued", "verdict": "queue_for_review",
                     "reason": "passed all hygiene gates"})
    return out


def run_discovery_batch(
    papers_df: pd.DataFrame,
    *,
    use_llm: bool = True,
    confidence_threshold: float = 0.5,
    log: bool = True,
    use_llm_rescue: bool = False,
) -> dict:
    """Process a batch of papers + write queue + log.

    Returns: summary dict with per-stage counts.
    """
    library_titles = _load_library_titles()
    library_families = _load_library_families()

    counts: dict[str, int] = {}
    queued = []
    review_with_caveat = []
    borderline = []
    all_outcomes = []
    for _, row in papers_df.iterrows():
        outcome = process_paper(
            row.to_dict(),
            use_llm=use_llm,
            confidence_threshold=confidence_threshold,
            library_titles=library_titles,
            library_families=library_families,
            use_llm_rescue=use_llm_rescue,
        )
        verdict = outcome.get("verdict", "skip")
        counts[verdict] = counts.get(verdict, 0) + 1
        all_outcomes.append(outcome)
        if verdict == "queue_for_review":
            queued.append(outcome)
        elif verdict == "review_with_caveat":
            review_with_caveat.append(outcome)
        elif verdict == "borderline_review":
            borderline.append(outcome)

    if log and queued:
        DISCOVERY_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with DISCOVERY_QUEUE.open("a", encoding="utf-8") as f:
            for entry in queued + review_with_caveat:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    if log and borderline:
        DISCOVERY_BORDERLINE.parent.mkdir(parents=True, exist_ok=True)
        with DISCOVERY_BORDERLINE.open("a", encoding="utf-8") as f:
            for entry in borderline:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    if log:
        DISCOVERY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DISCOVERY_LOG.open("a", encoding="utf-8") as f:
            for entry in all_outcomes:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    return {
        "total":         len(all_outcomes),
        "stage_counts":  counts,
        "queued":        len(queued),
        "review_with_caveat": len(review_with_caveat),
        "borderline":    len(borderline),
    }


def read_discovery_queue(limit: int = 50) -> list[dict]:
    if not DISCOVERY_QUEUE.exists():
        return []
    rows = [json.loads(l) for l in DISCOVERY_QUEUE.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[-limit:][::-1]
