"""engine.research_store.papers.schema — PaperRegistryEntry dataclass.

Identity:
  paper_id        — UUID4 (stable within this codebase)
  doi             — canonical external identifier (preferred for dedupe)

Lifecycle:
  - Append-only jsonl, latest-per-doi wins
  - To amend, write a new entry with parent_paper_id chain and version bump
  - Same doctrine as REDLesson: never mutate prior rows
"""
from __future__ import annotations

import dataclasses as _dc
import uuid as _uuid
from enum import Enum
from typing import Any

from engine.research_store.papers.shelves import Shelf


REGISTRY_SCHEMA_VERSION = 2  # bumped 2026-06-06 for ingestion_reason (Phase 1.7)


class IntentCategory(str, Enum):
    """Normalized intent for an ingestion. LLM extracts this from
    free_text post-ingest so we can aggregate calibration data later
    ("of the 12 expand_breadth picks, how many actually expanded the
    book?").

    Phase 1.7 step 2 (2026-06-06). See
    [[spec-papers-curator-full-architecture-2026-06-05]].
    """
    EXPAND_BREADTH           = "expand_breadth"
    IMPROVE_EXISTING_SLEEVE  = "improve_existing_sleeve"
    ADDRESS_DECAY            = "address_decay"
    METHODOLOGY_BORROW       = "methodology_borrow"
    CHALLENGE_DOCTRINE       = "challenge_doctrine"
    CURIOSITY                = "curiosity"
    FACT_CHECK               = "fact_check"
    AUTHOR_TRUST             = "author_trust"
    OTHER                    = "other"


class IngestionReasonSource(str, Enum):
    """Who authored the ingestion_reason.

    Symmetric model — whoever PICKED the paper writes the reason:
      USER   — user manually ingested OR opened /incoming and clicked
               "✏️ write my own" before submitting
      AGENT  — user clicked "Ingest now" on /incoming, accepting the
               agent's pre-filled reason verbatim (consent, not authorship)

    No third "edited" state — the UI forces an explicit choice between
    "trust agent" and "write my own" to keep authorship clean for the
    Phase 5 calibration log.
    """
    USER  = "user"
    AGENT = "agent"


@_dc.dataclass(frozen=True)
class IngestionReason:
    """The "why this paper" record. Whoever picked the paper authors
    the reason; the source field tags authorship.

    Fields:
      free_text         — the rationale (≤ 200 chars after trim)
      intent_category   — LLM-normalized category (None until extractor
                          runs; remains None for legacy entries pre-1.7)
      source            — USER or AGENT (see IngestionReasonSource)
      user_ts           — iso UTC when the reason was recorded

    Empty/whitespace free_text is treated as "no reason given" — the
    /papers/new form leaves the field None when textarea is blank, not
    an IngestionReason with empty text.
    """
    free_text:       str
    intent_category: "IntentCategory | None"
    source:          IngestionReasonSource
    user_ts:         str

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_text":       self.free_text,
            "intent_category": self.intent_category.value if self.intent_category else None,
            "source":          self.source.value,
            "user_ts":         self.user_ts,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IngestionReason":
        cat_raw = d.get("intent_category")
        cat: IntentCategory | None = None
        if cat_raw:
            try:
                cat = IntentCategory(cat_raw)
            except ValueError:
                cat = IntentCategory.OTHER     # forward-compat: unknown
                                                # category falls to OTHER
                                                # rather than failing
        src_raw = d.get("source", "user")
        try:
            src = IngestionReasonSource(src_raw)
        except ValueError:
            src = IngestionReasonSource.USER   # legacy / corrupt rows
                                                # default to USER (the
                                                # less-confident attribution)
        return cls(
            free_text       = str(d.get("free_text", "")).strip()[:200],
            intent_category = cat,
            source          = src,
            user_ts         = str(d.get("user_ts", "")),
        )


class FulltextStatus(str, Enum):
    """Lifecycle of full-text acquisition for a paper."""

    INGESTED        = "ingested"        # PDF acquired, chunked, embedded in ChromaDB
    METADATA_ONLY   = "metadata_only"   # OpenAlex metadata only; no PDF
    PAYWALLED       = "paywalled"       # PDF exists but only on paywalled hosts
    UNATTEMPTED     = "unattempted"     # not yet tried (e.g. doctrine seed before
                                         #   acquisition pass)


# Stage C Phase A (2026-06-07): tier classification per the "three
# libraries" doctrine. Each paper occupies ONE tier; tier determines
# how much depth (PDF / chunks / hypothesis extraction) the system
# spends on it.
class PaperTier(str, Enum):
    """Three-tier knowledge-library classification.

    T1 DOCTRINE: methodology-defining paper whose SPECIFIC phrasings
        matter for system behavior (DSR / Cochrane / multi-test
        framework). Must be fully ingested + chunked + embedded so
        A/B retrieval finds the exact passage. ~20-30 papers max.
        Examples: Bailey-Lopez de Prado 2014 DSR, McLean-Pontiff 2016,
        HXZ 2020, HLZ 2016.

    T2 ANCHOR: canonical mechanism paper that defines a class of
        factor/strategy. Functions as CITATION TARGET — A/B reference
        by title + 1-line summary; full chunks NOT needed.
        Examples: Fama-French 1992/1993/2015, Jegadeesh-Titman 1993,
        KMPV 2018 Carry, Carr-Wu 2009 VRP.

    T3 RECENT: anything else — new publications, narrow applications,
        unproven contributions. Default for substrate-crawl arrivals.
        Title + abstract only; upgrade to T2 on user action.

    UNCLASSIFIED: backward-compat default for pre-Phase-A entries +
        for entries where the classifier returned low confidence.
    """
    T1_DOCTRINE   = "T1_DOCTRINE"
    T2_ANCHOR     = "T2_ANCHOR"
    T3_RECENT     = "T3_RECENT"
    UNCLASSIFIED  = "UNCLASSIFIED"


@_dc.dataclass(frozen=True)
class PaperRegistryEntry:
    """One canonical paper record.

    Identity:
        paper_id              — UUID4 (auto-generated)
        doi                   — canonical external id (or "" if unavailable)
        version               — int, starts at 1
        parent_paper_id       — for amendments
        schema_version        — for compat checks

    Bibliographic metadata:
        title, year, authors, venue, abstract

    Acquisition:
        fulltext_status       — FulltextStatus enum
        pdf_source_kind       — "openalex_oa" / "ssrn" / "nber" / "arxiv" /
                                "manual" / ""
        pdf_source_url        — URL we downloaded from (if any)
        n_chunks              — chunks ingested into ChromaDB (0 if no PDF)
        ingested_ts           — ISO-8601 when ingest happened (or "")

    Reverse links (populated by Q-E cross-link pass):
        referenced_by_lessons   — REDLesson lesson_ids
        referenced_by_factors   — factor / candidate names
        referenced_by_sleeves   — deployed sleeve ids
        referenced_by_doctrines — doctrine memory_ids

    Partitioning (the load-bearing axis):
        shelves               — 1-N Shelf enum values (multi-label)
        shelf_notes           — {shelf_value: rationale string} for OTHER and
                                non-obvious assignments

    Metadata:
        created_ts, updated_ts, created_by, tags, note
    """

    # Identity
    paper_id:              str
    version:               int
    parent_paper_id:       str | None

    # Bibliographic
    doi:                   str
    title:                 str
    year:                  int
    authors:               tuple[str, ...]
    venue:                 str
    abstract:              str

    # Acquisition
    fulltext_status:       FulltextStatus
    pdf_source_kind:       str
    pdf_source_url:        str
    n_chunks:              int
    ingested_ts:           str

    # Reverse links
    referenced_by_lessons:    tuple[str, ...]
    referenced_by_factors:    tuple[str, ...]
    referenced_by_sleeves:    tuple[str, ...]
    referenced_by_doctrines:  tuple[str, ...]

    # Partitioning
    shelves:               tuple[Shelf, ...]
    shelf_notes:           dict[str, str]   # shelf_value → rationale

    # Metadata
    created_ts:            str
    updated_ts:            str
    created_by:            str
    tags:                  tuple[str, ...]
    note:                  str

    schema_version:        int = REGISTRY_SCHEMA_VERSION

    # Phase 1.7 step 2 (2026-06-06): user-supplied reason for ingesting
    # this paper. Drives Employee A's Layer 3 reasoning + Phase 5
    # calibration. None for entries created before 1.7 or when user
    # leaves the form blank. NOT auto-filled — surfacing a blank
    # ingestion_reason in the UI as "no reason given" is honest signal.
    ingestion_reason:      "IngestionReason | None" = None

    # Stage C Phase A (2026-06-07): tier classification per the "three
    # libraries" doctrine. Defaults to UNCLASSIFIED so pre-existing
    # rows load unchanged; classify_papers_into_tiers.py walks the
    # registry and proposes tiers via Sonnet for human review.
    tier:                  PaperTier = PaperTier.UNCLASSIFIED
    # ISO-8601 ts when tier was last set (audit). "" = never set.
    tier_classified_ts:    str = ""
    # 1-sentence rationale from the classifier (or human). "" = none.
    tier_rationale:        str = ""

    # NOTE on append-only: papers_registry is a CATALOG (state of one
    # object per paper_id), NOT an event log. Mutation via amend_entry
    # creates a new VERSION row for the SAME paper_id — this preserves
    # history without forcing event-log semantics. Dedup is handled
    # via in-place rewriting (scripts/compact_papers_registry.py); the
    # event-log stores (events.jsonl / hypotheses.jsonl / verdicts.jsonl)
    # remain strictly append-only.

    # Stage C Phase B (2026-06-07): T2 anchor enrichment. For T2_ANCHOR
    # papers, A doesn't need full PDF chunks — it needs the paper as a
    # CITATION TARGET with a 1-line meta-summary explaining WHAT makes
    # this paper canonical for its mechanism class. This field carries
    # that Sonnet-generated summary; populated by
    # scripts/enrich_t2_anchors.py. Empty for non-T2 or pre-enrichment.
    tier_anchor_summary:   str = ""

    # ─────────── factory ───────────

    @staticmethod
    def new_id() -> str:
        return str(_uuid.uuid4())

    # ─────────── (de)serialization ───────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id":              self.paper_id,
            "version":               self.version,
            "parent_paper_id":       self.parent_paper_id,
            "schema_version":        self.schema_version,

            "doi":      self.doi,
            "title":    self.title,
            "year":     self.year,
            "authors":  list(self.authors),
            "venue":    self.venue,
            "abstract": self.abstract,

            "fulltext_status":  self.fulltext_status.value,
            "pdf_source_kind":  self.pdf_source_kind,
            "pdf_source_url":   self.pdf_source_url,
            "n_chunks":         self.n_chunks,
            "ingested_ts":      self.ingested_ts,

            "referenced_by_lessons":    list(self.referenced_by_lessons),
            "referenced_by_factors":    list(self.referenced_by_factors),
            "referenced_by_sleeves":    list(self.referenced_by_sleeves),
            "referenced_by_doctrines":  list(self.referenced_by_doctrines),

            "shelves":      [s.value for s in self.shelves],
            "shelf_notes":  dict(self.shelf_notes),

            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "created_by": self.created_by,
            "tags":       list(self.tags),
            "note":       self.note,

            "ingestion_reason": (self.ingestion_reason.to_dict()
                                  if self.ingestion_reason is not None else None),

            "tier":               self.tier.value,
            "tier_classified_ts": self.tier_classified_ts,
            "tier_rationale":     self.tier_rationale,
            "tier_anchor_summary": self.tier_anchor_summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PaperRegistryEntry":
        return cls(
            paper_id              = d["paper_id"],
            version               = int(d.get("version", 1)),
            parent_paper_id       = d.get("parent_paper_id"),
            schema_version        = int(d.get("schema_version", REGISTRY_SCHEMA_VERSION)),

            doi      = d.get("doi", ""),
            title    = d.get("title", ""),
            year     = int(d.get("year", 0)),
            authors  = tuple(d.get("authors") or ()),
            venue    = d.get("venue", ""),
            abstract = d.get("abstract", ""),

            fulltext_status  = FulltextStatus(d.get("fulltext_status", "unattempted")),
            pdf_source_kind  = d.get("pdf_source_kind", ""),
            pdf_source_url   = d.get("pdf_source_url", ""),
            n_chunks         = int(d.get("n_chunks", 0)),
            ingested_ts      = d.get("ingested_ts", ""),

            referenced_by_lessons    = tuple(d.get("referenced_by_lessons") or ()),
            referenced_by_factors    = tuple(d.get("referenced_by_factors") or ()),
            referenced_by_sleeves    = tuple(d.get("referenced_by_sleeves") or ()),
            referenced_by_doctrines  = tuple(d.get("referenced_by_doctrines") or ()),

            shelves      = tuple(Shelf(s) for s in (d.get("shelves") or ())),
            shelf_notes  = dict(d.get("shelf_notes") or {}),

            created_ts = d.get("created_ts", ""),
            updated_ts = d.get("updated_ts", d.get("created_ts", "")),
            created_by = d.get("created_by", ""),
            tags       = tuple(d.get("tags") or ()),
            note       = d.get("note", ""),

            ingestion_reason = (IngestionReason.from_dict(d["ingestion_reason"])
                                if d.get("ingestion_reason") else None),

            tier                = PaperTier(d.get("tier", "UNCLASSIFIED")),
            tier_classified_ts  = d.get("tier_classified_ts", ""),
            tier_rationale      = d.get("tier_rationale", ""),
            tier_anchor_summary = d.get("tier_anchor_summary", ""),
        )

    # ─────────── validation ───────────

    def validate(self) -> list[str]:
        errs: list[str] = []

        if not self.title.strip():
            errs.append("title is empty")
        if not self.shelves:
            errs.append("at least 1 shelf is required (multi-label allowed)")
        if self.year and not (1900 <= self.year <= 2100):
            errs.append(f"year {self.year} outside plausible range")

        # Each OTHER shelf assignment must have an entry in shelf_notes
        for s in self.shelves:
            if s == Shelf.OTHER:
                if Shelf.OTHER.value not in self.shelf_notes:
                    errs.append(
                        "shelf=OTHER requires a rationale in shelf_notes['other']"
                    )

        # If status is INGESTED, n_chunks > 0 + ingested_ts must be set
        if self.fulltext_status == FulltextStatus.INGESTED:
            if self.n_chunks <= 0:
                errs.append("status=INGESTED requires n_chunks > 0")
            if not self.ingested_ts:
                errs.append("status=INGESTED requires ingested_ts")

        return errs
