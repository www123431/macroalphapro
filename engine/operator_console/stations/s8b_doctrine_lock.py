"""S8b — Doctrine Lock (form-only micro-station for `doctrine` session type).

Per docs/architecture/operator_console.md §5: the `doctrine` session
type was orphaned in Phase 1 because all 8 main stations mapped only
to research_new / audit / ops. S8b fills the gap so a user opening a
doctrine session has a station to use.

Intentionally simple — doctrine locking is a deterministic write
(memory markdown file + MEMORY.md index line + memory_doctrine_locked
event), not a multi-stage pipeline. No async work needed; runs
in-thread.

Design reference: docs/architecture/operator_console.md §5 (S8b spec).
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from pathlib import Path
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


# Memory dir per project convention. The principal's auto-memory lives
# under ~/.claude/projects/<sluggy-cwd>/memory/. Resolve from env so
# different installs (different home dirs) work.
def _memory_dir() -> Path:
    import os
    # Project-specific path used throughout this codebase:
    return Path(os.path.expanduser(
        r"~/.claude/projects/c--Users-${USER}-Desktop-intern/memory"
    ))


_VALID_TYPES = {"feedback", "project", "user", "reference"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _today_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _slug_to_filename(slug: str, doctrine_type: str) -> str:
    """Map slug → filename per project convention.

    Examples:
      slug="sizing-before-signal", type="feedback"
        → "feedback_sizing_before_signal_2026-06-23.md"
      slug="six-week-critical-path", type="project"
        → "project_six_week_critical_path_2026-06-23.md"
    """
    # kebab-case → snake_case
    snake = slug.replace("-", "_").lower()
    snake = re.sub(r"[^a-z0-9_]+", "_", snake)
    snake = re.sub(r"_+", "_", snake).strip("_")
    return f"{doctrine_type}_{snake}_{_today_iso()}.md"


class DoctrineLock(PipelineStation):
    """S8b — Write a doctrine memory + emit memory_doctrine_locked event."""

    STATION_SPEC = StationSpec(
        station_id              = "S8b_doctrine_lock",
        title                   = "Doctrine Lock",
        description             = (
            "Write a new doctrine memory: standing rule / project fact / "
            "user preference / external reference. Generates the markdown "
            "file, adds an index entry to MEMORY.md, and emits a "
            "memory_doctrine_locked event for the audit trail."
        ),
        data_tier               = DataTier.USER_DATA,
        requires_session_types  = {SessionType.DOCTRINE},
        estimated_minutes       = 5,
        estimated_cost_usd      = 0.0,
        icon                    = "BookOpen",
        title_key               = "console.station.s8b.title",
        description_key         = "console.station.s8b.description",
    )

    def preflight(self, session: Session, config: dict) -> PreflightResult:
        checks: list[PreflightCheck] = []

        if not session or not getattr(session, "session_id", ""):
            checks.append(PreflightCheck("session_active", PreflightStatus.RED,
                                         "No active session."))
        else:
            checks.append(PreflightCheck("session_active", PreflightStatus.GREEN,
                                         f"Session {session.session_id} ready."))

        c = config or {}
        slug = str(c.get("title_slug", "")).strip()
        if not slug:
            checks.append(PreflightCheck("title_slug_provided", PreflightStatus.RED,
                                         "Provide a kebab-case slug for the doctrine title."))
        elif not re.match(r"^[a-z0-9][a-z0-9\-]{1,80}$", slug):
            checks.append(PreflightCheck(
                "title_slug_valid", PreflightStatus.RED,
                "Slug must be kebab-case (a-z, 0-9, hyphen), 2-80 chars.",
            ))
        else:
            checks.append(PreflightCheck("title_slug_valid", PreflightStatus.GREEN,
                                         f"Filename will be {_slug_to_filename(slug, c.get('doctrine_type', 'feedback'))}"))

        doctrine_type = str(c.get("doctrine_type", "")).strip()
        if doctrine_type not in _VALID_TYPES:
            checks.append(PreflightCheck(
                "doctrine_type_valid", PreflightStatus.RED,
                f"doctrine_type must be one of: {sorted(_VALID_TYPES)}; got '{doctrine_type}'",
            ))
        else:
            checks.append(PreflightCheck("doctrine_type_valid", PreflightStatus.GREEN,
                                         f"Type '{doctrine_type}' valid."))

        body = str(c.get("body", "")).strip()
        if len(body) < 50:
            checks.append(PreflightCheck(
                "body_substantive", PreflightStatus.RED,
                f"Doctrine body must be ≥50 chars (got {len(body)}). A doctrine without substance isn't a doctrine.",
            ))
        else:
            checks.append(PreflightCheck("body_substantive", PreflightStatus.GREEN,
                                         f"Body length: {len(body)} chars."))

        why = str(c.get("why", "")).strip()
        if len(why) < 20:
            checks.append(PreflightCheck(
                "why_present", PreflightStatus.RED,
                f"`why` field is MANDATORY (got {len(why)} chars; needs ≥20). The reason this doctrine exists is load-bearing for future judgment.",
            ))
        else:
            checks.append(PreflightCheck("why_present", PreflightStatus.GREEN,
                                         "Why field present."))

        how_to_apply = str(c.get("how_to_apply", "")).strip()
        if len(how_to_apply) < 20:
            checks.append(PreflightCheck(
                "how_to_apply_present", PreflightStatus.RED,
                f"`how_to_apply` MANDATORY (got {len(how_to_apply)} chars; needs ≥20). Without 'when does this kick in', the doctrine is unusable.",
            ))
        else:
            checks.append(PreflightCheck("how_to_apply_present", PreflightStatus.GREEN,
                                         "How-to-apply field present."))

        # Memory dir writable
        memdir = _memory_dir()
        if not memdir.exists():
            checks.append(PreflightCheck(
                "memory_dir_writable", PreflightStatus.YELLOW,
                f"Memory dir doesn't exist; will create at {memdir}",
            ))
        else:
            checks.append(PreflightCheck("memory_dir_writable", PreflightStatus.GREEN,
                                         f"Memory dir found at {memdir}"))

        return PreflightResult.from_checks(checks)

    def estimate_cost(self, config: dict) -> CostEstimate:
        return CostEstimate(llm_cost_usd_est=0.0, confidence="exact")

    def render_config_form(self) -> dict:
        return {
            "type": "object",
            "title": "Doctrine Lock input",
            "description": (
                "Write a new doctrine to memory. Every field except "
                "`related_memories` is mandatory; `why` and `how_to_apply` "
                "are load-bearing for future judgment."
            ),
            "properties": {
                "title_slug": {
                    "type": "string",
                    "title": "Title slug (kebab-case)",
                    "description": "Used in filename. e.g. 'sizing-before-signal' → feedback_sizing_before_signal_2026-06-23.md",
                    "x-ui-widget": "text",
                    "x-ui-placeholder": "e.g. sizing-before-signal",
                },
                "doctrine_type": {
                    "type": "string",
                    "title": "Doctrine type",
                    "description": "feedback = how to work; project = current state; user = principal preference; reference = external pointer",
                    "enum": ["feedback", "project", "user", "reference"],
                    "default": "feedback",
                    "x-ui-widget": "select",
                },
                "headline": {
                    "type": "string",
                    "title": "One-line headline",
                    "description": "Shown in MEMORY.md index. ≤120 chars.",
                    "x-ui-widget": "text",
                },
                "body": {
                    "type": "string",
                    "title": "Body (markdown, ≥50 chars)",
                    "description": "The full doctrine text. Quote the precipitating incident if any.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 6,
                },
                "why": {
                    "type": "string",
                    "title": "Why (≥20 chars, MANDATORY)",
                    "description": "The reason this rule exists. Often a past incident or strong preference. Required for the doctrine to be useful when judging edge cases.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 2,
                },
                "how_to_apply": {
                    "type": "string",
                    "title": "How to apply (≥20 chars, MANDATORY)",
                    "description": "When this doctrine kicks in. Required so future readers know when to invoke.",
                    "x-ui-widget": "text-area",
                    "x-ui-rows": 2,
                },
                "related_memories": {
                    "type": "string",
                    "title": "Related memories (optional)",
                    "description": "Comma-separated [[slug]] references to existing doctrines (no .md extension).",
                    "x-ui-widget": "text",
                    "default": "",
                },
            },
            "required": ["title_slug", "doctrine_type", "headline", "body", "why", "how_to_apply"],
        }

    async def execute(
        self,
        session: Session,
        config: dict,
        emitter: SSEEmitter,
        cancellation: CancellationToken,
    ) -> StationResult:
        started_ts = _utc_iso()
        actor_id = getattr(session, "actor_id", "principal")
        session_id = getattr(session, "session_id", "")

        c = config or {}
        slug          = str(c.get("title_slug", "")).strip()
        doctrine_type = str(c.get("doctrine_type", "")).strip()
        headline      = str(c.get("headline", "")).strip()
        body          = str(c.get("body", "")).strip()
        why           = str(c.get("why", "")).strip()
        how_to_apply  = str(c.get("how_to_apply", "")).strip()
        related_raw   = str(c.get("related_memories", "")).strip()
        related_list = [r.strip().lstrip("[[").rstrip("]]")
                         for r in related_raw.split(",") if r.strip()]

        # ── Stage 1: write doctrine markdown file ────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "write_file")
        emitter.stage_started("write_file", expected_seconds=1)
        memdir = _memory_dir()
        memdir.mkdir(parents=True, exist_ok=True)
        filename = _slug_to_filename(slug, doctrine_type)
        filepath = memdir / filename

        if filepath.exists():
            emitter.stage_failed("write_file",
                                   f"file already exists: {filename}. Pick a different slug.")
            return self._failed(session, started_ts, "write_file",
                                f"file already exists: {filename}")

        related_md = "\n".join(f"- See also: [[{r}]]" for r in related_list)
        related_block = f"\n\n**Related memories**:\n{related_md}" if related_list else ""

        frontmatter = (
            f"---\n"
            f"name: {filename.removesuffix('.md')}\n"
            f"description: {headline[:200]}\n"
            f"metadata:\n"
            f"  type: {doctrine_type}\n"
            f"---\n\n"
        )

        markdown_body = (
            f"# {headline}\n\n"
            f"{body}\n\n"
            f"**Why**: {why}\n\n"
            f"**How to apply**: {how_to_apply}"
            f"{related_block}\n"
        )

        try:
            filepath.write_text(frontmatter + markdown_body, encoding='utf-8')
        except OSError as e:
            emitter.stage_failed("write_file", str(e)[:300])
            return self._failed(session, started_ts, "write_file", str(e)[:300])

        emitter.stage_completed("write_file", {
            "filename": filename,
            "path":     str(filepath),
            "bytes":    filepath.stat().st_size,
        })

        # ── Stage 2: append to MEMORY.md index ───────────────────
        if cancellation.cancelled:
            return self._cancelled(session, started_ts, "update_index")
        emitter.stage_started("update_index", expected_seconds=1)
        memory_index = memdir / "MEMORY.md"
        index_line = f"- [{headline}]({filename}) — {body[:80].replace(chr(10), ' ')}"
        try:
            existing = memory_index.read_text(encoding='utf-8') if memory_index.is_file() else "# Memory Index\n"
            if index_line not in existing:
                # Insert after the header (or at end if no header)
                lines = existing.split("\n")
                # Find first blank line after header → insert there
                insert_idx = 2  # default just after "# Memory Index" + blank
                for i, line in enumerate(lines):
                    if i > 0 and not line.strip():
                        insert_idx = i + 1
                        break
                lines.insert(insert_idx, index_line)
                memory_index.write_text("\n".join(lines), encoding='utf-8')
            emitter.stage_completed("update_index", {
                "index_line": index_line[:100],
                "index_size_kb": round(memory_index.stat().st_size / 1024, 1),
            })
        except OSError as e:
            emitter.stage_failed("update_index", str(e)[:300])
            # Don't fail the whole station — file was already written
            emitter.log_line("index update failed but doctrine file was written; can re-index later")

        # ── Stage 3: emit doctrine_proposed event ────────────────
        emitter.stage_started("emit_event", expected_seconds=1)
        try:
            opcon_emit.doctrine_proposed(
                session_id   = session_id,
                actor_id     = actor_id,
                title        = headline,
                body         = body,
                doctrine_type= doctrine_type,
                why          = why,
                how_to_apply = how_to_apply,
                related_memories = related_list,
            )
        except Exception as e:
            logger.exception("operator_console: failed to emit doctrine_proposed")
            emitter.log_line(f"emit warning: {e}")

        # Also emit station_completed for the worker pipeline
        try:
            opcon_emit.station_completed(
                session_id      = session_id,
                actor_id        = actor_id,
                job_id          = "",
                station_id      = self.STATION_SPEC.station_id,
                cost_actual_usd = 0.0,
                artifacts       = {
                    "doctrine_path": str(filepath),
                    "filename":      filename,
                    "doctrine_type": doctrine_type,
                    "slug":          slug,
                },
            )
        except Exception:
            logger.exception("operator_console: failed to emit station_completed")
        emitter.stage_completed("emit_event", {"filename": filename})

        return StationResult(
            job_id          = "",
            station_id      = self.STATION_SPEC.station_id,
            session_id      = session_id,
            actor_id        = actor_id,
            started_ts      = started_ts,
            completed_ts    = _utc_iso(),
            success         = True,
            artifacts       = {
                "doctrine_path": str(filepath),
                "filename":      filename,
                "doctrine_type": doctrine_type,
                "slug":          slug,
                "headline":      headline[:160],
            },
            events_emitted  = [],
            next_stations   = [],   # doctrine is terminal — session can close
            cost_actual_usd = 0.0,
        )

    def result_lineage(self, result: StationResult) -> list[NextStationHint]:
        return []   # terminal node

    def _cancelled(self, session: Session, started_ts: str, stage: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Cancelled at stage '{stage}'.",
        )

    def _failed(self, session: Session, started_ts: str, stage: str, err: str) -> StationResult:
        return StationResult(
            job_id        = "",
            station_id    = self.STATION_SPEC.station_id,
            session_id    = getattr(session, "session_id", ""),
            actor_id      = getattr(session, "actor_id", "principal"),
            started_ts    = started_ts,
            completed_ts  = _utc_iso(),
            success       = False,
            error_message = f"Stage '{stage}' failed: {err}",
        )


registry.register(DoctrineLock)
