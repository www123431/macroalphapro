"""engine.research_store.papers.amend — versioned amendments to registry entries.

Per the same doctrine as RED Lessons: registry is append-only. To "update"
an entry (add a shelf, add a reverse link, mark as ingested), write a NEW
version with `parent_paper_id` pointing to the prior + bumped `version`.

Helpers here build the new version cleanly.
"""
from __future__ import annotations

from typing import Iterable

from engine.research_store.papers.schema import PaperRegistryEntry, PaperTier
from engine.research_store.papers.shelves import Shelf


def amend_entry(prior: PaperRegistryEntry, *,
                add_shelves:    Iterable[Shelf] = (),
                add_shelf_notes: dict[str, str] | None = None,
                add_lessons:    Iterable[str] = (),
                add_factors:    Iterable[str] = (),
                add_sleeves:    Iterable[str] = (),
                add_doctrines:  Iterable[str] = (),
                add_tags:       Iterable[str] = (),
                # Stage C Phase A (2026-06-07): tier amendments
                set_tier:          "PaperTier | None" = None,
                set_tier_rationale: "str | None"      = None,
                set_tier_classified_ts: "str | None"   = None,
                # Stage C Phase B (2026-06-07): T2 anchor enrichment
                set_tier_anchor_summary: "str | None"  = None,
                set_abstract:      "str | None"        = None,
                updated_ts:     str = "",
                created_by:     str = "engine.amend",
                note_append:    str = "") -> PaperRegistryEntry:
    """Produce a new (v+1) entry merging the additions into the prior entry.

    Lists are unioned (preserving order; no duplicates).
    """
    def _union_tuple(prior_tup: tuple, additions: Iterable) -> tuple:
        seen = set(prior_tup)
        out = list(prior_tup)
        for x in additions:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return tuple(out)

    new_shelves       = _union_tuple(prior.shelves,                  add_shelves)
    new_lessons       = _union_tuple(prior.referenced_by_lessons,    add_lessons)
    new_factors       = _union_tuple(prior.referenced_by_factors,    add_factors)
    new_sleeves       = _union_tuple(prior.referenced_by_sleeves,    add_sleeves)
    new_doctrines     = _union_tuple(prior.referenced_by_doctrines,  add_doctrines)
    new_tags          = _union_tuple(prior.tags,                     add_tags)

    new_shelf_notes = dict(prior.shelf_notes)
    if add_shelf_notes:
        for k, v in add_shelf_notes.items():
            # If both prior and new have notes for the same shelf, keep the
            # newer (later writer); if prior has none, take new.
            new_shelf_notes[k] = v

    new_note = prior.note
    if note_append:
        if new_note:
            new_note = new_note + " | " + note_append
        else:
            new_note = note_append

    # IMPORTANT (2026-06-07): registry is a CATALOG (state of each
    # paper), not an event log. Amendments KEEP the same paper_id and
    # just bump version. Previously this method minted a NEW paper_id
    # per amendment + set parent_paper_id back — that pattern caused
    # the registry to bloat with orphan chains (226 raw rows → 57
    # distinct paper_ids → 35 actual papers) and is now fixed.
    # event-log stores (events/hypotheses/verdicts) remain strictly
    # append-only — they're different beasts. parent_paper_id is kept
    # in the schema for legacy compatibility but set to None on new
    # amendments.
    return PaperRegistryEntry(
        paper_id              = prior.paper_id,
        version               = prior.version + 1,
        parent_paper_id       = None,
        schema_version        = prior.schema_version,

        # Bibliographic — unchanged unless caller overrides abstract
        # (Phase B enrichment from CrossRef can backfill missing /
        # truncated abstracts).
        doi      = prior.doi,
        title    = prior.title,
        year     = prior.year,
        authors  = prior.authors,
        venue    = prior.venue,
        abstract = (set_abstract if set_abstract is not None
                     else prior.abstract),

        # Acquisition — unchanged unless caller explicitly overrides; not
        # supported here yet (acquisition amendments come from the
        # acquisition layer)
        fulltext_status  = prior.fulltext_status,
        pdf_source_kind  = prior.pdf_source_kind,
        pdf_source_url   = prior.pdf_source_url,
        n_chunks         = prior.n_chunks,
        ingested_ts      = prior.ingested_ts,

        referenced_by_lessons    = new_lessons,
        referenced_by_factors    = new_factors,
        referenced_by_sleeves    = new_sleeves,
        referenced_by_doctrines  = new_doctrines,

        shelves      = new_shelves,
        shelf_notes  = new_shelf_notes,

        created_ts = prior.created_ts,
        updated_ts = updated_ts or prior.updated_ts,
        created_by = created_by,
        tags       = new_tags,
        note       = new_note,

        # Tier — carry forward by default; explicit set_tier overrides
        tier               = (set_tier if set_tier is not None
                                else prior.tier),
        tier_classified_ts = (set_tier_classified_ts
                                if set_tier_classified_ts is not None
                                else prior.tier_classified_ts),
        tier_rationale     = (set_tier_rationale
                                if set_tier_rationale is not None
                                else prior.tier_rationale),
        tier_anchor_summary = (set_tier_anchor_summary
                                if set_tier_anchor_summary is not None
                                else prior.tier_anchor_summary),
    )
