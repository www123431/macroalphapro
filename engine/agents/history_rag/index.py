"""
Row → IndexedDoc transformers for the 5 history-RAG sources, plus the
top-level ``build_index()`` orchestrator that bulk-upserts into ChromaDB.

Sources
-------
1. DecisionLog            — 1 doc per row, narrative fields concatenated
2. SpecRegistry           — 1 doc per spec, status + path + amendment summary
3. spec_amendment         — 1 doc per amendment_log entry inside a SpecRegistry
                            row (parent spec referenced via metadata.parent_spec_id)
4. PendingApproval        — 1 doc per row when rationale is non-empty
5. AgentReflection        — 1 doc per row, 4-section CONTEXT/DECISION/OUTCOME/LESSON
6. AuditFinding           — 1 doc per row, rule_name + severity + notes

Each row > CHUNK_TARGET_CHARS gets chunked at sentence boundaries with
CHUNK_OVERLAP_CHARS overlap. Chunks share the same doc base_id with a
``#chunk_{i}`` suffix.

Idempotency
-----------
ChromaDB upsert uses doc_id as primary key. Re-running build_index() over
the same rows is safe — vectors get recomputed but no duplicates appear.
For incremental runs, pass ``modified_since=<timestamp>`` and only rows
with newer ``updated_at`` / ``created_at`` are touched.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from typing import Iterable, Iterator

from engine.agents.history_rag.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
)
from engine.agents.history_rag.schema import IndexedDoc, SourceType
from engine.agents.history_rag.store import get_store

logger = logging.getLogger(__name__)

# ── Chunker ──────────────────────────────────────────────────────────────────

# Match sentence boundaries in mixed-script text (Latin + CJK punctuation).
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


def _split_into_chunks(text: str,
                       target: int = CHUNK_TARGET_CHARS,
                       overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Sentence-aware chunker with character-based target + overlap.

    Behavior:
      - Empty / short input (≤ target chars) returns a single-chunk list.
      - Otherwise greedy-pack sentences into windows of ~target chars,
        starting each new window with the last ``overlap`` chars of the
        previous window so cross-chunk semantics survive.
      - Sentences longer than target chars (rare) get hard-split at
        target - overlap so we don't drop content.
    """
    text = (text or "").strip()
    if len(text) <= target:
        return [text] if text else []

    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for sent in sentences:
        if len(sent) > target:
            # Flush current chunk first
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            # Hard-split oversize sentence
            step = max(1, target - overlap)
            for i in range(0, len(sent), step):
                chunks.append(sent[i : i + target])
            continue
        if cur_len + len(sent) + 1 > target:
            chunks.append(" ".join(cur))
            # Seed next chunk with overlap tail of last chunk
            tail = chunks[-1][-overlap:] if overlap > 0 else ""
            cur = [tail, sent] if tail else [sent]
            cur_len = sum(len(x) for x in cur) + len(cur)
        else:
            cur.append(sent)
            cur_len += len(sent) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if c.strip()]


def _to_chunked_docs(base_id: str,
                     source_type: SourceType,
                     source_id: str,
                     text: str,
                     title: str,
                     occurred_at: datetime.datetime | None,
                     metadata: dict,
                     deep_link: str | None) -> list[IndexedDoc]:
    """Build ≥1 IndexedDoc objects from a single source row's text payload."""
    chunks = _split_into_chunks(text)
    if not chunks:
        return []
    if len(chunks) == 1:
        return [IndexedDoc(
            doc_id=base_id, source_type=source_type, source_id=source_id,
            text=chunks[0], title=title, occurred_at=occurred_at,
            metadata=metadata, deep_link=deep_link,
        )]
    out: list[IndexedDoc] = []
    for i, chunk in enumerate(chunks):
        out.append(IndexedDoc(
            doc_id=f"{base_id}#chunk_{i}",
            source_type=source_type,
            source_id=source_id,
            text=chunk,
            title=f"{title} (part {i + 1}/{len(chunks)})",
            occurred_at=occurred_at,
            metadata={**metadata, "chunk_index": i, "chunk_total": len(chunks)},
            deep_link=deep_link,
        ))
    return out


# ── Source-specific row → IndexedDoc transformers ────────────────────────────

def _index_decision_logs(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index narrative + reasoning fields from each DecisionLog row.

    Concatenates the most informative free-text columns into a single
    document per row, with structured metadata (regime, ticker,
    confidence, verified) for filtering.
    """
    from engine.memory import SessionFactory, DecisionLog

    with SessionFactory() as s:
        q = s.query(DecisionLog)
        if modified_since is not None:
            q = q.filter(DecisionLog.created_at >= modified_since)
        for row in q.yield_per(50):
            parts: list[str] = []
            # News summary intentionally excluded from indexed text (UX:
            # historical news context is decision-time input but visually
            # noisy in RAG evidence cards — drop at index layer, keep DL
            # row in DB intact so hash-chain integrity is preserved).
            for label, val in [
                ("Sector",            row.sector_name),
                ("Ticker",            row.ticker),
                ("Direction",         row.direction),
                ("Conclusion",        row.ai_conclusion),
                ("Economic logic",    row.economic_logic),
                ("Key thesis",        row.key_thesis),
                ("Primary risk",      row.primary_risk),
                ("Invalidation",      row.invalidation_conditions),
                ("Reflection",        row.reflection),
                ("Failure note",      row.failure_note),
                ("Adjustment reason", row.adjustment_reason),
                ("Revision reason",   row.revision_reason),
            ]:
                if val and str(val).strip():
                    parts.append(f"{label}: {str(val).strip()}")
            text = "\n".join(parts)
            if not text:
                continue

            occurred = row.created_at or (
                datetime.datetime.combine(row.decision_date, datetime.time.min)
                if row.decision_date else None
            )
            title = (
                f"DL #{row.id} {row.sector_name or '?'} "
                f"{row.direction or ''}".strip()
            )
            meta = {
                "ticker":       row.ticker,
                "sector":       row.sector_name,
                "regime":       row.macro_regime,
                "direction":    row.direction,
                "verified":     bool(row.verified),
                "confidence":   row.confidence_score,
                "is_backtest":  bool(row.is_backtest),
                "tab_type":     row.tab_type,
                "spec_hash":    row.spec_hash,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            yield from _to_chunked_docs(
                base_id=f"decision_log:{row.id}",
                source_type=SourceType.DECISION_LOG,
                source_id=str(row.id),
                text=text, title=title, occurred_at=occurred,
                metadata=meta,
                deep_link="pages/decisions.py",
            )


def _index_spec_registry(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index each SpecRegistry row + each amendment_log entry separately.

    Two doc types emerge from one row:
      - 1 spec_registry doc summarizing path + status + hash
      - N spec_amendment docs, one per parsed amendment_log entry

    Tracking amendments separately lets supervisors query "what amendments
    happened in 2026-05" and get per-amendment hits, not per-spec hits
    that bury the relevant amendment in a wall of text.
    """
    from engine.memory import SessionFactory, SpecRegistry

    with SessionFactory() as s:
        q = s.query(SpecRegistry)
        if modified_since is not None:
            q = q.filter(SpecRegistry.last_validated_at >= modified_since)
        for row in q.yield_per(50):
            spec_text = (
                f"Spec path: {row.spec_path}\n"
                f"Status: {row.status}\n"
                f"Current hash: {(row.current_hash or '')[:16]}\n"
                f"Registered: {row.registered_at}\n"
                f"N trials contributed: {row.n_trials_contributed}\n"
                f"Retro-registered: {row.retro_registered}"
            )
            spec_meta = {
                "spec_path":  row.spec_path,
                "status":     row.status,
                "spec_hash":  (row.current_hash or "")[:16],
                "n_trials":   row.n_trials_contributed,
            }
            spec_meta = {k: v for k, v in spec_meta.items() if v is not None}
            yield IndexedDoc(
                doc_id=f"spec_registry:{row.id}",
                source_type=SourceType.SPEC_REGISTRY,
                source_id=str(row.id),
                text=spec_text,
                title=f"Spec #{row.id} {row.spec_path}",
                occurred_at=row.registered_at,
                metadata=spec_meta,
                deep_link="pages/decisions.py",
            )

            # Amendments — parse JSON list if present
            amendments_raw = row.amendment_log
            if not amendments_raw:
                continue
            try:
                amendments = json.loads(amendments_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(amendments, list):
                continue
            for i, amd in enumerate(amendments):
                if not isinstance(amd, dict):
                    continue
                kind   = str(amd.get("kind", amd.get("type", "?")))
                rationale = str(amd.get("rationale", amd.get("reason", "")))
                applied_at_raw = amd.get("applied_at") or amd.get("date")
                applied_at = None
                if applied_at_raw:
                    try:
                        applied_at = datetime.datetime.fromisoformat(
                            str(applied_at_raw).replace("Z", "+00:00")
                        )
                    except Exception:
                        pass
                amd_text = (
                    f"Amendment to {row.spec_path}\n"
                    f"Kind: {kind}\n"
                    f"Applied: {applied_at_raw or '?'}\n"
                    f"Rationale: {rationale}"
                )
                if not rationale.strip():
                    # Skip empty-rationale amendments (rare but happens)
                    continue
                amd_meta = {
                    "spec_path":      row.spec_path,
                    "amendment_kind": kind,
                    "parent_spec_id": row.id,
                }
                yield IndexedDoc(
                    doc_id=f"spec_amendment:{row.id}:{i}",
                    source_type=SourceType.SPEC_AMENDMENT,
                    source_id=f"{row.id}:{i}",
                    text=amd_text,
                    title=f"Amend #{i} {row.spec_path}",
                    occurred_at=applied_at or row.registered_at,
                    metadata=amd_meta,
                    deep_link="pages/decisions.py",
                )


def _index_pending_approvals(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index governance decisions: PendingApproval rows with rationale text.

    Skips rows with no narrative content (auto-resolved approvals with
    only a triggered_condition and no review_rationale add noise without
    info gain).
    """
    from engine.memory import SessionFactory
    from engine.db_models import PendingApproval

    with SessionFactory() as s:
        q = s.query(PendingApproval)
        if modified_since is not None:
            q = q.filter(PendingApproval.created_at >= modified_since)
        for row in q.yield_per(50):
            parts: list[str] = []
            for label, val in [
                ("Approval type",        row.approval_type),
                ("Sector / Ticker",      f"{row.sector or '?'} / {row.ticker or '?'}"),
                ("Triggered condition",  row.triggered_condition),
                ("Suggested weight",     row.suggested_weight),
                ("Status",               row.status),
                ("Rationale",            row.review_rationale),
                ("Category",             row.review_category),
                ("Rejection reason",     row.rejection_reason),
                ("Risk override note",   row.risk_override_note),
                ("Post-hoc note",        row.post_hoc_note),
            ]:
                if val is not None and str(val).strip():
                    parts.append(f"{label}: {val}")
            text = "\n".join(parts)
            if len(text) < 80:
                # below this threshold the doc adds index bloat, not signal
                continue
            meta = {
                "approval_type":  row.approval_type,
                "ticker":         row.ticker,
                "sector":         row.sector,
                "status":         row.status,
                "approval_class": row.approval_class,
                "priority":       row.priority,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            yield from _to_chunked_docs(
                base_id=f"pending_approval:{row.id}",
                source_type=SourceType.PENDING_APPROVAL,
                source_id=str(row.id),
                text=text,
                title=f"PA #{row.id} {row.approval_type}",
                occurred_at=row.created_at,
                metadata=meta,
                deep_link="pages/orchestrator.py",
            )


def _index_agent_reflections(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index 4-section agent reflections (CONTEXT / DECISION / OUTCOME / LESSON).

    Reflexion-style memory accumulates over calendar time; this indexer
    stays empty initially (project_s2_reflection_complete_2026-05-04
    notes ≥50 reflections expected by 2026-09).
    """
    from engine.memory import SessionFactory
    from engine.db_models import AgentReflection

    with SessionFactory() as s:
        q = s.query(AgentReflection)
        if modified_since is not None:
            q = q.filter(AgentReflection.created_at >= modified_since)
        for row in q.yield_per(50):
            parts: list[str] = []
            for label, val in [
                ("Agent",            row.agent_id),
                ("Decision summary", row.decision_summary),
                ("Realized outcome", row.realized_outcome),
                ("Hit flag",         row.hit_flag),
                ("Factor context",   row.factor_context),
                ("Reflection",       row.reflection_text),
            ]:
                if val is not None and str(val).strip():
                    parts.append(f"{label}: {val}")
            text = "\n".join(parts)
            if not text:
                continue
            occurred = row.created_at or (
                datetime.datetime.combine(row.decision_date, datetime.time.min)
                if row.decision_date else None
            )
            meta = {
                "agent_id":  row.agent_id,
                "hit_flag":  row.hit_flag,
                "decision_ref_id": row.decision_ref_id,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            yield from _to_chunked_docs(
                base_id=f"agent_reflection:{row.id}",
                source_type=SourceType.AGENT_REFLECTION,
                source_id=str(row.id),
                text=text,
                title=f"Reflection #{row.id} {row.agent_id or '?'}",
                occurred_at=occurred,
                metadata=meta,
                deep_link="pages/decision_journal.py",
            )


def _index_audit_findings(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index Auto-Audit findings (rule + severity + notes + snapshot summary).

    Snapshot JSON is summarized to its top-level keys + values so the
    embedding picks up "what changed" terminology rather than raw JSON.
    """
    from engine.memory import SessionFactory
    from engine.auto_audit_models import AuditFinding

    with SessionFactory() as s:
        q = s.query(AuditFinding)
        if modified_since is not None:
            q = q.filter(AuditFinding.detected_at >= modified_since)
        for row in q.yield_per(50):
            snap_summary = ""
            if row.snapshot_json:
                try:
                    snap = json.loads(row.snapshot_json)
                    if isinstance(snap, dict):
                        snap_summary = "; ".join(
                            f"{k}={v}" for k, v in list(snap.items())[:8]
                            if isinstance(v, (str, int, float, bool))
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            text = (
                f"Audit finding {row.rule_name}\n"
                f"Severity: {row.severity}\n"
                f"Status: {row.status}\n"
                f"Notes: {row.notes or '(none)'}\n"
                f"Snapshot summary: {snap_summary or '(empty)'}"
            )
            meta = {
                "rule_name": row.rule_name,
                "severity":  row.severity,
                "status":    row.status,
                "run_id":    row.run_id,
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            yield IndexedDoc(
                doc_id=f"audit_finding:{row.id}",
                source_type=SourceType.AUDIT_FINDING,
                source_id=str(row.id),
                text=text,
                title=f"AF #{row.id} {row.rule_name}",
                occurred_at=row.detected_at,
                metadata=meta,
                deep_link="pages/auto_audit.py",
            )


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _index_system_help(modified_since: datetime.datetime | None) -> Iterator[IndexedDoc]:
    """Index hard-coded self-description docs (capability help).

    Source: engine/agents/history_rag/self_help_docs.py.
    Static knowledge — modified_since is ignored (these never change at
    runtime; only when the source file is edited + re-indexed).
    """
    from engine.agents.history_rag.self_help_docs import iter_help_docs
    yield from iter_help_docs()


_INDEXERS = {
    SourceType.DECISION_LOG:     _index_decision_logs,
    SourceType.SPEC_REGISTRY:    _index_spec_registry,    # also yields SPEC_AMENDMENT
    SourceType.PENDING_APPROVAL: _index_pending_approvals,
    SourceType.AGENT_REFLECTION: _index_agent_reflections,
    SourceType.AUDIT_FINDING:    _index_audit_findings,
    SourceType.SYSTEM_HELP:      _index_system_help,
}


def build_index(
    sources:         Iterable[SourceType] | None = None,
    modified_since:  datetime.datetime    | None = None,
    batch_size:      int                         = 64,
) -> dict[str, int]:
    """Build (or incrementally update) the history-RAG index.

    Args
    ----
    sources : which source types to (re)index. None = all.
    modified_since : when set, each indexer filters its rows by the
                     appropriate timestamp column (created_at /
                     last_validated_at / detected_at). None = full sweep.
    batch_size : upsert batch size (chromadb is memory-bound during embed).

    Returns
    -------
    {source_type: n_indexed} counters for telemetry / smoke verification.
    """
    if sources is None:
        sources = list(_INDEXERS.keys())
    coll = get_store()
    counters: dict[str, int] = {}

    for st in sources:
        fn = _INDEXERS.get(st)
        if fn is None:
            logger.warning("history_rag.build_index: unknown source %s", st)
            continue

        batch_ids: list[str]   = []
        batch_docs: list[str]  = []
        batch_metas: list[dict] = []
        n = 0

        for doc in fn(modified_since):
            batch_ids.append(doc.doc_id)
            batch_docs.append(doc.text)
            batch_metas.append(doc.chroma_metadata())
            n += 1
            if len(batch_ids) >= batch_size:
                coll.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
                batch_ids, batch_docs, batch_metas = [], [], []

        if batch_ids:
            coll.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)

        counters[st.value] = n
        logger.info("history_rag.build_index: %s indexed %d docs", st.value, n)

    counters["_total"] = sum(v for k, v in counters.items() if not k.startswith("_"))
    return counters
