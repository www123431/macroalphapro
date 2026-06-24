"""S1 — Paper Ingest.

First Pipeline Station. Accepts an arxiv URL (or arxiv ID) from the
user, fetches metadata from export.arxiv.org, classifies via Stage-0
ClaimType router, writes a minimal PaperRegistryEntry to
data/research_store/papers_registry.jsonl, emits a typed event.

This is the entry point of the paper → hypothesis → verdict pipeline.
Subsequent stations (S2 Synthesize, S3 SpecExtract, S4 FORWARD, etc.)
chain off the paper_id this station produces.

Design reference: docs/architecture/operator_console.md §5 (S1 spec).

MVP scope notes:
  - URL/ID input only; PDF upload deferred (multipart needs separate
    handling, adds complexity not justified for first station)
  - No LLM summary call (zero cost, demo_fixture-friendly); summary
    is the arxiv-provided abstract verbatim
  - Sync execute() that emits stage events sequentially via emitter;
    SSE wiring lives in the foundation per D2
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as _ET
from typing import Any

from engine.operator_console.pipeline_station import (
    PipelineStation,
    SSEEmitter,
    Session,
)
from engine.operator_console.schema import (
    CancellationToken,
    CostEstimate,
    DataTier,
    NextStationHint,
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
    SessionType,
    StationResult,
    StationSpec,
)
from engine.operator_console import emit as opcon_emit
from engine.operator_console import registry


logger = logging.getLogger(__name__)


# ── Arxiv fetch helpers ──────────────────────────────────────────


_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# Match a wide variety of arxiv URL shapes + bare IDs:
#   https://arxiv.org/abs/2401.12345
#   http://arxiv.org/abs/2401.12345v2
#   https://www.arxiv.org/pdf/2401.12345
#   arxiv:2401.12345
#   2401.12345
#   cs.LG/0312009 (old-style)
_ARXIV_ID_RE = re.compile(
    r"(?:arxiv[:/]|/abs/|/pdf/)?"
    r"(?P<id>(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}))"
    r"(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)


def _extract_arxiv_id(url_or_id: str) -> str | None:
    """Pull a canonical arxiv ID from a URL / bare-ID input. Returns
    None if the input doesn't match arxiv conventions."""
    if not url_or_id:
        return None
    m = _ARXIV_ID_RE.search(url_or_id.strip())
    return m.group("id") if m else None


def _fetch_arxiv_metadata(arxiv_id: str, *, timeout: float = 30.0) -> dict:
    """Fetch single-paper metadata from export.arxiv.org.

    Returns dict with keys: arxiv_id, title, abstract, authors (list),
    categories (list), abs_url, pdf_url, published_ts. Raises ValueError
    if the API returns no matching entry."""
    qs = urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1})
    req = urllib.request.Request(
        f"{_ARXIV_API_URL}?{qs}",
        headers={"User-Agent": "MacroAlphaPro/0.1 (research; mailto: see github)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    root = _ET.fromstring(body)
    entries = root.findall("atom:entry", _ARXIV_NS)
    if not entries:
        raise ValueError(f"arxiv returned no entry for id={arxiv_id}")

    entry = entries[0]
    title = (entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or "").strip()
    title = " ".join(title.split())

    abstract = (entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or "").strip()
    abstract = " ".join(abstract.split())

    published = entry.findtext("atom:published", default="", namespaces=_ARXIV_NS).strip()
    full_id = entry.findtext("atom:id", default="", namespaces=_ARXIV_NS).strip()
    abs_url = full_id
    pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

    authors = []
    for a in entry.findall("atom:author", _ARXIV_NS):
        name = (a.findtext("atom:name", default="", namespaces=_ARXIV_NS) or "").strip()
        if name:
            authors.append(name)

    cats = []
    for c in entry.findall("atom:category", _ARXIV_NS):
        term = c.attrib.get("term", "").strip()
        if term:
            cats.append(term)

    # Arxiv title "stuck" guard — if the title looks like an error stub,
    # treat as not-found (arxiv API returns a 1-entry feed even when
    # the id doesn't exist; the entry just contains an error message).
    if not title or title.lower().startswith("error"):
        raise ValueError(f"arxiv: no usable title for id={arxiv_id} (entry empty or error)")

    return {
        "arxiv_id":     arxiv_id,
        "title":        title,
        "abstract":     abstract,
        "authors":      authors,
        "categories":   cats,
        "abs_url":      abs_url,
        "pdf_url":      pdf_url,
        "published_ts": published,
    }


# ── Year extraction (papers_registry requires int year) ──────────


def _year_from_arxiv_id(arxiv_id: str) -> int:
    """Arxiv id YYMM.NNNNN → year. e.g. 2401.12345 → 2024."""
    m = re.match(r"^(\d{2})(\d{2})\.", arxiv_id)
    if m:
        yy = int(m.group(1))
        # arxiv switched to new ID scheme in 2007; 2007-2099 covered
        return 2000 + yy
    return 0   # unknown; old-style IDs use category/year shape


# ── The station ──────────────────────────────────────────────────


class PaperIngest(PipelineStation):
    """S1 — Paper Ingest. arxiv URL/ID → ClaimType-tagged paper_registry row."""

    STATION_SPEC = StationSpec(
        station_id              = "S1_paper_ingest",
        title                   = "Paper Ingest",
        description             = (
            "Pull a paper from arxiv by URL or ID. Stage-0 ClaimType router "
            "tags the abstract; result lands in the papers registry and seeds "
            "downstream synthesis (S2) or factor-spec extraction."
        ),
        data_tier               = DataTier.USER_DATA,
        requires_session_types  = {SessionType.RESEARCH_NEW, SessionType.EXPLORATION},
        estimated_minutes       = 1,
        estimated_cost_usd      = 0.0,   # no LLM call; deterministic
        icon                    = "FileText",
        title_key               = "console.station.s1.title",
        description_key         = "console.station.s1.description",
    )

    # ── Pre-flight ────────────────────────────────────────────────

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        # Session presence + type validity
        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck(
                name   = "session_active",
                status = PreflightStatus.RED,
                detail = "No active session passed to preflight.",
            ))
        else:
            checks.append(PreflightCheck(
                name   = "session_active",
                status = PreflightStatus.GREEN,
                detail = f"Session {session.session_id} ready.",
            ))

        # Config validity
        url_or_id = (config or {}).get("arxiv_url") or (config or {}).get("arxiv_id") or ""
        url_or_id = str(url_or_id).strip()
        if not url_or_id:
            checks.append(PreflightCheck(
                name   = "input_provided",
                status = PreflightStatus.RED,
                detail = "Provide an arxiv URL or ID.",
            ))
        else:
            extracted = _extract_arxiv_id(url_or_id)
            if not extracted:
                checks.append(PreflightCheck(
                    name   = "input_parseable",
                    status = PreflightStatus.RED,
                    detail = f"Could not extract arxiv ID from '{url_or_id[:60]}'.",
                ))
            else:
                checks.append(PreflightCheck(
                    name   = "input_parseable",
                    status = PreflightStatus.GREEN,
                    detail = f"Will fetch arxiv id={extracted}.",
                ))

        # arxiv API reachability — soft check; record as yellow so a
        # transient outage doesn't block trigger entirely.
        checks.append(PreflightCheck(
            name   = "arxiv_api_reachable",
            status = PreflightStatus.YELLOW,
            detail = "Arxiv API is queried at execute() time; offline networks will fail then.",
        ))

        return PreflightResult.from_checks(checks)

    # ── Cost estimate ─────────────────────────────────────────────

    def estimate_cost(self, config: dict) -> CostEstimate:
        # Deterministic: no LLM call in MVP. Always $0.
        return CostEstimate(llm_cost_usd_est=0.0, confidence="exact")

    # ── Config form (JSON Schema) ─────────────────────────────────

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "Paper Ingest input",
            "description": "Provide an arxiv URL or ID; metadata is fetched from export.arxiv.org.",
            "properties": {
                "arxiv_url": {
                    "type": "string",
                    "title": "Arxiv URL or ID",
                    "description": "Examples: https://arxiv.org/abs/2401.12345 · arxiv:2401.12345 · 2401.12345",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "https://arxiv.org/abs/2401.12345",
                },
                "user_note": {
                    "type": "string",
                    "title": "Why you're ingesting this paper (optional)",
                    "description": "Surfaces in the registry for downstream context.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 3,
                    "default": "",
                },
            },
            "required": ["arxiv_url"],
        }

    # ── Execute ───────────────────────────────────────────────────

    async def execute(
        self,
        session: Session,
        config: dict,
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        from engine.agents.papers_curator.claim_type_router import classify
        from engine.research_store.papers.schema import (
            PaperRegistryEntry, FulltextStatus, PaperTier, REGISTRY_SCHEMA_VERSION,
        )
        from engine.research_store.papers.store import save_entry, find_by_doi

        started_ts = _utc_iso()
        actor_id = getattr(session, "actor_id", "principal")
        session_id = getattr(session, "session_id", "")

        url_or_id = str((config or {}).get("arxiv_url") or (config or {}).get("arxiv_id") or "").strip()
        user_note = str((config or {}).get("user_note") or "").strip()

        # ── Stage 1: parse input ──────────────────────────────────
        if cancellation.cancelled:
            return self._cancelled_result(session, started_ts, "input_parse")
        emitter.stage_started("input_parse", expected_seconds=1)
        arxiv_id = _extract_arxiv_id(url_or_id)
        if not arxiv_id:
            emitter.stage_failed("input_parse", f"Could not extract arxiv id from '{url_or_id[:80]}'.")
            return self._failed_result(session, started_ts, "input_parse", f"Bad input: {url_or_id[:80]}")
        emitter.stage_completed("input_parse", {"arxiv_id": arxiv_id})

        # ── Stage 2: fetch arxiv metadata ─────────────────────────
        if cancellation.cancelled:
            return self._cancelled_result(session, started_ts, "fetch_metadata")
        emitter.stage_started("fetch_metadata", expected_seconds=10)
        try:
            meta = _fetch_arxiv_metadata(arxiv_id)
        except Exception as e:
            emitter.stage_failed("fetch_metadata", str(e)[:300])
            return self._failed_result(session, started_ts, "fetch_metadata", str(e)[:300])
        emitter.stage_completed("fetch_metadata", {
            "title":    meta["title"][:120],
            "authors":  meta["authors"][:5],
            "categories": meta["categories"],
        })

        # ── Stage 3: claim_type classification ────────────────────
        if cancellation.cancelled:
            return self._cancelled_result(session, started_ts, "claim_type")
        emitter.stage_started("claim_type", expected_seconds=3)
        verdict = classify(meta["title"], meta["abstract"])
        emitter.stage_completed("claim_type", {
            "claim_type": verdict.claim_type.value,
            "confidence": verdict.confidence,
            "top_hits":   [k for k, v in (verdict.hits or {}).items() if v][:3],
        })

        # ── Stage 4: persist to papers_registry ───────────────────
        if cancellation.cancelled:
            return self._cancelled_result(session, started_ts, "registry_write")
        emitter.stage_started("registry_write", expected_seconds=1)
        existing = find_by_doi("")    # doi="" → no dedup for arxiv-only papers
        # Dedup by arxiv_id in tags instead — search registry for any
        # entry with arxiv_id in tags. Skip dedup for MVP; downstream
        # consumer can merge on (created_by, tags) later.
        paper_id = str(uuid.uuid4())
        now = _utc_iso()
        try:
            entry = PaperRegistryEntry(
                paper_id              = paper_id,
                version               = 1,
                parent_paper_id       = None,
                doi                   = "",
                title                 = meta["title"],
                year                  = _year_from_arxiv_id(arxiv_id),
                authors               = tuple(meta["authors"]),
                venue                 = "arxiv",
                abstract              = meta["abstract"],
                fulltext_status       = FulltextStatus.METADATA_ONLY,
                pdf_source_kind       = "arxiv",
                pdf_source_url        = meta["pdf_url"],
                n_chunks              = 0,
                ingested_ts           = "",
                referenced_by_lessons   = (),
                referenced_by_factors   = (),
                referenced_by_sleeves   = (),
                referenced_by_doctrines = (),
                shelves               = (),
                shelf_notes           = {},
                created_ts            = now,
                updated_ts            = now,
                created_by            = "operator_console.s1_paper_ingest",
                tags                  = (f"arxiv:{arxiv_id}", f"claim_type:{verdict.claim_type.value}"),
                note                  = user_note,
                schema_version        = REGISTRY_SCHEMA_VERSION,
                tier                  = PaperTier.UNCLASSIFIED,
            )
            save_entry(entry, validate_strict=False)
        except Exception as e:
            emitter.stage_failed("registry_write", str(e)[:300])
            return self._failed_result(session, started_ts, "registry_write", str(e)[:300])

        emitter.stage_completed("registry_write", {
            "paper_id":     paper_id,
            "registry_url": f"/research/papers/{paper_id}",
        })

        # ── Emit station_completed + return result ────────────────
        completed_ts = _utc_iso()
        try:
            opcon_emit.station_completed(
                session_id        = session_id,
                actor_id          = actor_id,
                job_id            = "",   # filled by API layer if it tracks
                station_id        = self.STATION_SPEC.station_id,
                cost_actual_usd   = 0.0,
                artifacts         = {
                    "paper_registry_entry": f"data/research_store/papers_registry.jsonl#{paper_id}",
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")

        return StationResult(
            job_id           = "",
            station_id       = self.STATION_SPEC.station_id,
            session_id       = session_id,
            actor_id         = actor_id,
            started_ts       = started_ts,
            completed_ts     = completed_ts,
            success          = True,
            artifacts        = {
                "paper_id":     paper_id,
                "registry_url": f"/research/papers/{paper_id}",
                "abs_url":      meta["abs_url"],
                "pdf_url":      meta["pdf_url"],
                "title":        meta["title"],
                "claim_type":   verdict.claim_type.value,
                "confidence":   str(verdict.confidence),
            },
            events_emitted   = [],
            next_stations    = self._lineage_hints(paper_id, verdict.claim_type.value),
            cost_actual_usd  = 0.0,
        )

    # ── Lineage ───────────────────────────────────────────────────

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        paper_id = result.artifacts.get("paper_id", "")
        claim_type = result.artifacts.get("claim_type", "")
        return self._lineage_hints(paper_id, claim_type)

    @staticmethod
    def _lineage_hints(paper_id: str, claim_type: str) -> list[NextStationHint]:
        hints: list[NextStationHint] = []
        # Only FACTOR_HYPOTHESIS papers feed S2 synthesis directly
        if claim_type == "FACTOR_HYPOTHESIS":
            hints.append(NextStationHint(
                station_id        = "S2_hypothesis_synthesize",
                label             = "Synthesize hypothesis from this paper",
                suggested_config  = {"paper_ids": [paper_id]},
            ))
        # All papers can be viewed via existing /research/papers/<id>
        return hints

    # ── Helpers for failure / cancel returns ──────────────────────

    def _cancelled_result(self, session: Session, started_ts: str, stage: str) -> StationResult:
        return StationResult(
            job_id           = "",
            station_id       = self.STATION_SPEC.station_id,
            session_id       = getattr(session, "session_id", ""),
            actor_id         = getattr(session, "actor_id", "principal"),
            started_ts       = started_ts,
            completed_ts     = _utc_iso(),
            success          = False,
            error_message    = f"Cancelled at stage '{stage}'.",
        )

    def _failed_result(self, session: Session, started_ts: str, stage: str, err: str) -> StationResult:
        return StationResult(
            job_id           = "",
            station_id       = self.STATION_SPEC.station_id,
            session_id       = getattr(session, "session_id", ""),
            actor_id         = getattr(session, "actor_id", "principal"),
            started_ts       = started_ts,
            completed_ts     = _utc_iso(),
            success          = False,
            error_message    = f"Stage '{stage}' failed: {err}",
        )


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ── Register at import time ──────────────────────────────────────


registry.register(PaperIngest)
