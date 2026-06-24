"""engine.agents.papers_curator.synthesis_context — Phase 2.0 step 3b.

Reads existing stores and builds a SynthesisInput snapshot the
synthesis module consumes. Kept SEPARATE from synthesis.py so:

  - synthesis.py is unit-testable with synthetic SynthesisInput
  - synthesis_context.py is unit-testable with tmp_path fixtures
  - neither imports the other's testing concerns

Per [[spec-research-session-orchestrator-2026-06-06]] §"Employee A":
the gatherer reads SNAPSHOTS, never live state — the orchestrator
freezes state once and passes the frozen object to the LLM call.

Reused infrastructure (per the "we are one big component" principle —
no new stores, integrate into the existing graph):

  - data/papers_curator/cache.jsonl + judgments.jsonl + summaries.jsonl
    (Phase 1.5 / 1.5b)
  - data/research/mechanism_library/*.yaml  (existing library)
  - data/research_store/events.jsonl via engine.research_store.store
    (existing emit infrastructure)

Memory snippets (the 4th input source) are STUBBED for now —
query_doctrine is a Phase 1.7 L2 tool, not yet built. The synthesizer
prompt still works without doctrine snippets (just less context); when
query_doctrine ships, swap in real retrieval.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

from engine.agents.papers_curator.synthesis import (
    SynthesisInput,
    PaperSummaryRef,
    SleeveStateRef,
    RecentEventRef,
    DoctrineHit,
)

logger = logging.getLogger(__name__)


_REPO_ROOT      = Path(__file__).resolve().parent.parent.parent.parent
_PAPERS_DIR     = _REPO_ROOT / "data" / "papers_curator"
_LIBRARY_DIR    = _REPO_ROOT / "data" / "research" / "mechanism_library"
_EVENTS_PATH    = _REPO_ROOT / "data" / "research_store" / "events.jsonl"


# ────────────────────────────────────────────────────────────────────
# Tiny jsonl helper — read-only, tolerant of missing file
# ────────────────────────────────────────────────────────────────────
def _iter_jsonl(p: Path):
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s line %d malformed: %s", p.name, ln_no, exc)


# ────────────────────────────────────────────────────────────────────
# Recent paper summaries — join cache + judgments + summaries
# ────────────────────────────────────────────────────────────────────
def _load_recent_summaries(*, days: int = 14,
                            max_rows: int = 30) -> tuple[PaperSummaryRef, ...]:
    """Read papers_curator stores, join by (source, source_id), return
    candidates from the last N days. Prefers INGEST > READ_AND_DISCARD
    > SKIP; caps at `max_rows` to keep the LLM prompt bounded.

    Returns () if no summaries exist yet (fresh system or
    papers_curator not yet running)."""
    cache_path     = _PAPERS_DIR / "cache.jsonl"
    judgments_path = _PAPERS_DIR / "judgments.jsonl"
    summaries_path = _PAPERS_DIR / "summaries.jsonl"

    # Build lookup tables
    cache_by_key = {}
    for c in _iter_jsonl(cache_path):
        key = (c.get("source", ""), c.get("source_id", ""))
        cache_by_key[key] = c

    # latest judgment + summary per (source, source_id) (jsonl is append-only)
    judgments_by_key = {}
    for j in _iter_jsonl(judgments_path):
        key = (j.get("source", ""), j.get("source_id", ""))
        prev = judgments_by_key.get(key)
        if prev is None or j.get("judged_ts", "") > prev.get("judged_ts", ""):
            judgments_by_key[key] = j

    summaries_by_key = {}
    for s in _iter_jsonl(summaries_path):
        key = (s.get("source", ""), s.get("source_id", ""))
        prev = summaries_by_key.get(key)
        if prev is None or s.get("summarized_ts", "") > prev.get("summarized_ts", ""):
            summaries_by_key[key] = s

    # Filter by recency — use fetched_ts from cache as canonical "when
    # the paper arrived". Summaries+judgments TS slightly after but
    # this keeps the window definition stable.
    try:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        cutoff_iso = "1900-01-01T00:00:00Z"

    rows: list[PaperSummaryRef] = []
    for key, c in cache_by_key.items():
        if c.get("fetched_ts", "") < cutoff_iso:
            continue
        s = summaries_by_key.get(key)
        j = judgments_by_key.get(key)
        if s is None:
            # No summary yet — skip (filter-only rows aren't useful
            # for synthesis; we want at least the 5-field summary)
            continue
        # paper_id we use is the source-native id (e.g. arxiv 2401.12345)
        # — synthesizer will reference these in its output's
        # synthesizes_paper_ids field
        rows.append(PaperSummaryRef(
            paper_id            = f"{c.get('source', 'arxiv')}/{c.get('source_id', '')}",
            title               = str(c.get("title", ""))[:200],
            authors_short       = ", ".join(c.get("authors") or [])[:120],
            thesis              = str(s.get("thesis", ""))[:300],
            testable_hypothesis = str(s.get("testable_hypothesis", ""))[:200],
            why_matters_for_us  = str(s.get("why_matters_for_us", ""))[:200],
            risk_flags_short    = tuple((s.get("risk_flags") or [])[:5]),
            recommended_action  = str(s.get("recommended_action", "")),
        ))

    # Sort by recommended_action priority then by fetched_ts (newer first)
    action_priority = {"INGEST": 0, "READ_AND_DISCARD": 1, "SKIP": 2}
    rows.sort(key=lambda r: (action_priority.get(r.recommended_action, 3),
                              r.paper_id))
    return tuple(rows[:max_rows])


# ────────────────────────────────────────────────────────────────────
# Deployed sleeves — read library YAMLs
# ────────────────────────────────────────────────────────────────────
def _load_deployed_sleeves() -> tuple[SleeveStateRef, ...]:
    """Read library/*.yaml, filter to status_in_our_book == DEPLOYED.
    KPI fields come from the YAML when present; None when absent."""
    if not _LIBRARY_DIR.is_dir():
        return ()
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML missing — sleeve loading disabled")
        return ()

    out: list[SleeveStateRef] = []
    for p in sorted(_LIBRARY_DIR.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("library yaml %s parse failed: %s", p.name, exc)
            continue
        status = str(d.get("status_in_our_book", "")).upper()
        if status != "DEPLOYED":
            continue
        sleeve_id = str(d.get("id") or p.stem)
        family = str(d.get("family") or d.get("parent_family") or "OTHER")
        kpi = d.get("live_kpi") or d.get("kpi") or {}
        sharpe = None
        try:
            sharpe = float(kpi.get("ann_sharpe_live")) if kpi.get("ann_sharpe_live") is not None else None
        except (TypeError, ValueError):
            sharpe = None
        months = None
        try:
            months = int(kpi.get("months_since_deploy")) if kpi.get("months_since_deploy") is not None else None
        except (TypeError, ValueError):
            months = None
        decay_alert = kpi.get("last_decay_alert_ts") or None
        if decay_alert is not None:
            decay_alert = str(decay_alert)
        out.append(SleeveStateRef(
            sleeve_id           = sleeve_id,
            family              = family,
            status              = status,
            ann_sharpe_live     = sharpe,
            months_since_deploy = months,
            last_decay_alert    = decay_alert,
        ))
    return tuple(out)


# ────────────────────────────────────────────────────────────────────
# Recent events — read events.jsonl
# ────────────────────────────────────────────────────────────────────
_SYNTHESIS_RELEVANT_EVENT_TYPES = {
    "factor_verdict_filed",
    "capability_evidence_filed",
    "decay_alert",
    "doctrine_signal_detected",          # Phase 2.0 step 9, not yet emitted
    "council_critique",                  # Phase 4 DA refutes
}


def _load_recent_events(*, days: int = 30,
                          max_rows: int = 40) -> tuple[RecentEventRef, ...]:
    """Read events.jsonl tail, filter to synthesis-relevant types.

    Returns () if no events yet (fresh system)."""
    if not _EVENTS_PATH.is_file():
        return ()
    try:
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
        cutoff_iso = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        cutoff_iso = "1900-01-01T00:00:00Z"

    out: list[RecentEventRef] = []
    for ev in _iter_jsonl(_EVENTS_PATH):
        if str(ev.get("event_type", "")) not in _SYNTHESIS_RELEVANT_EVENT_TYPES:
            continue
        ts = str(ev.get("ts", ""))
        if ts < cutoff_iso:
            continue
        out.append(RecentEventRef(
            event_id   = str(ev.get("event_id", ""))[:36],
            event_type = str(ev.get("event_type", "")),
            subject_id = str(ev.get("subject_id", ""))[:30],
            family     = str(ev.get("family", "") or "OTHER"),
            verdict    = str(ev.get("verdict", "") or "—"),
            summary    = str(ev.get("summary", ""))[:200],
            ts         = ts,
        ))

    # Newest first
    out.sort(key=lambda r: r.ts, reverse=True)
    return tuple(out[:max_rows])


# ────────────────────────────────────────────────────────────────────
# Doctrine snippets — STUBBED until query_doctrine ships
# ────────────────────────────────────────────────────────────────────
def _load_doctrine_snippets(*, topic_hint: str = "",
                              top_k: int = 5) -> tuple[DoctrineHit, ...]:
    """Tier-2 (2026-06-07): real query_doctrine call.

    Returns up to top_k memory entries semantically relevant to the
    topic_hint, retrieved from doctrine_chroma (built over the
    principal's ~/.claude/projects/.../memory/*.md corpus).

    Pre-tier-2 this was a stub returning (). The change: A's prompt
    now sees the principal's locked doctrine on every synthesis call —
    closing the largest pre-tier-2 quality bottleneck.

    Empty topic_hint → () (cost discipline; no chroma fire if there's
    no anchor). Chroma infra failure → () (caller falls back to no-
    doctrine reasoning, same as pre-tier-2 stub behavior).
    """
    if not topic_hint or not topic_hint.strip():
        return ()
    try:
        from engine.agents.papers_curator.doctrine_index import query_doctrine
        raw_hits = query_doctrine(topic_hint, top_k=top_k)
    except Exception as exc:
        logger.warning("synthesis_context: query_doctrine raised: %s", exc)
        return ()

    # Adapt doctrine_index.DoctrineHit → synthesis.DoctrineHit (shape
    # difference: index hit has (name, description, entry_type, snippet,
    # distance, file_path); synthesis hit has
    # (memory_file_id, headline, snippet)).
    out: list[DoctrineHit] = []
    for h in raw_hits:
        out.append(DoctrineHit(
            memory_file_id = h.name,
            headline       = h.description[:160] if h.description
                              else h.name,
            snippet        = h.snippet[:400],
        ))
    return tuple(out)


def _build_topic_hint(
    recent_summaries: tuple,
    deployed_sleeves: tuple,
    recent_events:    tuple,
) -> str:
    """Compose a topic_hint string from the snapshot. The hint is what
    chroma's semantic search anchors against to find relevant doctrine.

    Strategy: include the highest-signal text from each source.
    Recent paper theses + family names of deployed sleeves + family
    names of recent doctrine_signal events. Cap total length so the
    embedding model isn't fed a novella."""
    parts: list[str] = []

    # Recent paper theses (top 5 INGEST-priority, already sorted by
    # _load_recent_summaries)
    paper_bits = []
    for s in (recent_summaries or ())[:5]:
        thesis = getattr(s, "thesis", "") or getattr(s, "title", "")
        if thesis:
            paper_bits.append(thesis[:160])
    if paper_bits:
        parts.append("Recent papers: " + " | ".join(paper_bits))

    # Deployed sleeves: just family names (the doctrine consulting we
    # want is "what's our doctrine for the families we already deploy")
    fams = sorted({getattr(s, "family", "") for s in (deployed_sleeves or ())
                    if getattr(s, "family", "")})
    if fams:
        parts.append("Deployed families: " + ", ".join(fams))

    # Recent doctrine_signal / decay events — surface the families that
    # are flagged for attention; that's where doctrine memory is most
    # load-bearing
    flagged_fams = set()
    for ev in (recent_events or ()):
        et = getattr(ev, "event_type", "")
        if et in ("doctrine_signal_detected", "decay_alert"):
            f = getattr(ev, "family", "") or ""
            if f:
                flagged_fams.add(f)
    if flagged_fams:
        parts.append("Flagged families: " + ", ".join(sorted(flagged_fams)))

    return ". ".join(parts)[:1000]


# ────────────────────────────────────────────────────────────────────
# Top-level
# ────────────────────────────────────────────────────────────────────
def _load_anchor_library() -> tuple:
    """Stage C Phase E (2026-06-07): load T1+T2 papers with anchor
    summaries from papers_registry. Each becomes an AnchorRef the
    LLM must verify orthogonality against.

    Returns () if loading fails or registry has no enriched anchors
    (synthesis prompt skips the anchor section gracefully)."""
    try:
        from engine.research_store.papers.store import load_registry
        from engine.research_store.papers.schema import PaperTier
        from engine.agents.papers_curator.synthesis import AnchorRef
    except Exception as exc:
        logger.warning("synthesis_context: anchor library import "
                        "failed: %s", exc)
        return ()
    try:
        raw = load_registry()
    except Exception as exc:
        logger.warning("synthesis_context: load_registry failed: %s",
                        exc)
        return ()

    # Latest-per-paper_id dedup (chain history artifact)
    by_pid: dict = {}
    for r in raw:
        prior = by_pid.get(r.paper_id)
        if prior is None or r.version > prior.version:
            by_pid[r.paper_id] = r
    latest = list(by_pid.values())

    # DOI dedup — prefer entry with enrichment
    by_doi: dict = {}
    no_doi: list = []
    for p in latest:
        d = (p.doi or "").strip().lower()
        if not d:
            no_doi.append(p)
            continue
        if d not in by_doi:
            by_doi[d] = p
        elif (p.tier_anchor_summary
              and not by_doi[d].tier_anchor_summary):
            by_doi[d] = p
    functional = list(by_doi.values()) + no_doi

    out = []
    for p in functional:
        if p.tier not in (PaperTier.T1_DOCTRINE, PaperTier.T2_ANCHOR):
            continue
        # T1 papers can ride without anchor_summary (their job is
        # methodology + we don't need the orthogonality framing for
        # them — the prompt sees them as "gates"). T2 papers MUST
        # have an anchor_summary or they're useless as citation
        # anchors — skip if empty.
        if (p.tier == PaperTier.T2_ANCHOR
            and not p.tier_anchor_summary):
            continue
        out.append(AnchorRef(
            paper_id       = p.paper_id[:8],
            tier           = p.tier.value,
            first_author   = (p.authors[0] if p.authors else "?"),
            year           = int(p.year or 0),
            anchor_summary = (p.tier_anchor_summary
                               or f"[methodology paper, no summary "
                                   f"needed]"),
        ))
    return tuple(out)


def build_synthesis_input(*, summaries_days: int = 14,
                             events_days: int = 30,
                             doctrine_top_k: int = 5) -> SynthesisInput:
    """Read all 5 sources and build the snapshot the LLM consumes.

    Tier-2 (2026-06-07): doctrine_snippets is no longer () — built
    from a topic_hint composed from the other 3 sources (papers +
    sleeves + flagged event families). chroma retrieves top-K
    semantically-relevant memory entries from the principal's
    200+ locked doctrine corpus.

    Stage C Phase E (2026-06-07): anchor_library added — T1+T2
    papers from papers_registry with their tier_anchor_summary, so A
    must explicitly state orthogonality vs canonical work.

    All loaders return () gracefully when their source is empty —
    the synthesizer prompt explicitly allows empty input paths."""
    snapshot_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Load the 3 source-only stores first so we can compose a
    # topic_hint for the doctrine query
    summaries = _load_recent_summaries(days=summaries_days)
    sleeves   = _load_deployed_sleeves()
    events    = _load_recent_events(days=events_days)

    topic_hint = _build_topic_hint(summaries, sleeves, events)
    doctrine   = _load_doctrine_snippets(topic_hint=topic_hint,
                                          top_k=doctrine_top_k)
    anchors    = _load_anchor_library()

    # Phase B (2026-06-14): belief layer summary from autopsy ledger.
    # Empty tuple if autopsy ledger absent — synthesis prompt
    # gracefully renders no belief section.
    try:
        from engine.research.belief_synthesis_context import (
            build_belief_summary,
        )
        belief = build_belief_summary(min_obs_per_family=3)
    except Exception:
        belief = ()

    return SynthesisInput(
        recent_summaries     = summaries,
        deployed_sleeves     = sleeves,
        recent_events        = events,
        doctrine_snippets    = doctrine,
        snapshot_ts          = snapshot_ts,
        anchor_library       = anchors,
        belief_layer_summary = belief,
    )
