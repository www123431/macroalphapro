"""engine.agents.strengthener.approval_view — Phase 2.0 step 12.

Read-side over B's verdicts.jsonl + a resolutions.jsonl side-file.
Composes the "pending approval" list the /approvals UI consumes.

Why a separate file-based store rather than the existing SQLite
`pending_approvals` table:
  - The legacy table is schema-fixed for ticker/sector/weight book
    decisions; B's verdicts carry different shape (hypothesis_id +
    confidence + reasoning + amendment summary).
  - Keeping B's flow on file-based jsonl matches the rest of the
    Phase 2.0 substrate (hypotheses.jsonl + events.jsonl +
    verdicts.jsonl) — single store paradigm.
  - When this surface proves valuable, we can migrate to the SQLite
    table without changing B's emit logic.

Resolution model:
  - Each pending verdict is keyed by hypothesis_id.
  - A `resolutions.jsonl` row records {hypothesis_id, decision, rationale,
    resolved_ts, resolved_by}. Decision ∈ {approved, rejected,
    deferred}.
  - This view returns verdicts that:
      (a) verdict_type ∈ {APPROVE_FOR_PIPELINE, DOCTRINE_AMENDMENT_NEEDED}
          (REJECT verdicts don't need human action — B already
          decided to drop them)
      (b) AND no resolution row exists yet
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_REPO_ROOT             = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_VERDICTS_PATH = _REPO_ROOT / "data" / "strengthener" / "verdicts.jsonl"
_DEFAULT_RESOLUTIONS_PATH = _REPO_ROOT / "data" / "strengthener" / "resolutions.jsonl"

_PENDING_VERDICT_TYPES = {"APPROVE_FOR_PIPELINE", "DOCTRINE_AMENDMENT_NEEDED"}
_DECISIONS              = {"approved", "rejected", "deferred"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ────────────────────────────────────────────────────────────────────
# Resolution storage
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class Resolution:
    hypothesis_id: str
    decision:      str          # approved / rejected / deferred
    rationale:     str
    resolved_ts:   str
    resolved_by:   str          # "user" or "agent:<name>"


def _iter_jsonl(p: Path):
    if not p.is_file():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_verdicts(path: Path) -> list[dict]:
    return list(_iter_jsonl(path))


def _load_resolutions(path: Path) -> dict[str, Resolution]:
    """Map hypothesis_id → latest Resolution (latest by resolved_ts)."""
    out: dict[str, Resolution] = {}
    for d in _iter_jsonl(path):
        hid = d.get("hypothesis_id")
        if not hid:
            continue
        r = Resolution(
            hypothesis_id = hid,
            decision      = str(d.get("decision", "")),
            rationale     = str(d.get("rationale", ""))[:1000],
            resolved_ts   = str(d.get("resolved_ts", "")),
            resolved_by   = str(d.get("resolved_by", "")),
        )
        prev = out.get(hid)
        if prev is None or r.resolved_ts > prev.resolved_ts:
            out[hid] = r
    return out


_DEFAULT_AMENDMENT_DRAFTS_DIR = _REPO_ROOT / "data" / "strengthener" / "amendment_drafts"


def write_amendment_draft(
    *,
    hypothesis_id:              str,
    blocking_doctrine_id:       str,
    proposed_amendment_summary: str,
    b_reasoning:                str,
    b_confidence:               float,
    drafts_dir:                 Optional[Path] = None,
) -> Path:
    """Phase 2.0 step 13: write a human-readable draft amendment file
    when the principal approves a DOCTRINE_AMENDMENT_NEEDED verdict.

    The file is NOT a unified-diff — it's a markdown template the
    principal opens in Claude Code (or any editor) to actually edit
    the targeted memory file. Memory file edits go through Claude's
    Write tool, not autonomous file mutation (per project doctrine).

    Returns the path to the written draft.
    """
    drafts_dir = drafts_dir or _DEFAULT_AMENDMENT_DRAFTS_DIR
    drafts_dir.mkdir(parents=True, exist_ok=True)
    out = drafts_dir / f"amendment_{hypothesis_id}.md"

    body = [
        f"# Doctrine Amendment Draft — {blocking_doctrine_id}",
        "",
        f"**Source hypothesis:** `{hypothesis_id}`",
        f"**B confidence:**      {b_confidence:.2f}",
        f"**Drafted at:**        {_utc_iso()}",
        "",
        "## Proposed amendment (B's summary)",
        "",
        proposed_amendment_summary.strip() or "(empty — see B reasoning below)",
        "",
        "## B's reasoning",
        "",
        b_reasoning.strip() or "(none)",
        "",
        "## Next step (manual)",
        "",
        f"Open `~/.claude/projects/c--Users-${USER}-Desktop-intern/memory/{blocking_doctrine_id}.md` "
        f"and apply this amendment. Then emit a `memory_doctrine_locked` event "
        f"with `parent_event_ids=` pointing at the `memory_amendment_proposed` "
        f"event id that produced this draft.",
        "",
        "---",
        "_Auto-generated by `engine.agents.strengthener.approval_view.write_amendment_draft`. "
        "Edit the memory file via Claude — do not autonomously rewrite it._",
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    return out


def find_hypothesis_family(hypothesis_id: str) -> Optional[str]:
    """Phase 2.1a helper: look up the mechanism_family for a
    hypothesis_id so the forward_vector_created event can be tagged
    with the right family. Returns None if hypothesis not found
    (caller should still emit, just without family tag)."""
    try:
        from engine.research_store.hypothesis.store import find_by_id
        h = find_by_id(hypothesis_id)
        if h is None:
            return None
        return h.mechanism_family.value
    except Exception as exc:
        logger.warning("find_hypothesis_family failed for %s: %s",
                        hypothesis_id, exc)
        return None


def find_verdict(
    hypothesis_id: str,
    *,
    verdicts_path: Optional[Path] = None,
) -> Optional[dict]:
    """Phase 2.1a helper: look up the latest B verdict for a given
    hypothesis_id. Used by /resolve to fetch verdict_type +
    b_confidence + extraction_method when emitting
    forward_vector_created on `approved`. Returns None if no verdict
    exists yet (caller decides what to do)."""
    p = verdicts_path or _DEFAULT_VERDICTS_PATH
    if not p.is_file():
        return None
    latest: Optional[dict] = None
    for v in _iter_jsonl(p):
        if v.get("hypothesis_id") != hypothesis_id:
            continue
        if latest is None or v.get("review_ts", "") > latest.get("review_ts", ""):
            latest = v
    return latest


def append_resolution(
    *,
    hypothesis_id: str,
    decision:      str,
    rationale:     str = "",
    resolved_by:   str = "user",
    path:          Optional[Path] = None,
) -> Resolution:
    """Append a Resolution row. Validates decision ∈ allowed set;
    the caller is trusted for hypothesis_id existence (the view ignores
    unmatched resolutions safely)."""
    if decision not in _DECISIONS:
        raise ValueError(
            f"unknown decision {decision!r}; choose from {sorted(_DECISIONS)}"
        )
    r = Resolution(
        hypothesis_id = hypothesis_id,
        decision      = decision,
        rationale     = rationale[:1000],
        resolved_ts   = _utc_iso(),
        resolved_by   = resolved_by,
    )
    p = path or _DEFAULT_RESOLUTIONS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_dc.asdict(r), ensure_ascii=False) + "\n")
    return r


# ────────────────────────────────────────────────────────────────────
# View — pending list
# ────────────────────────────────────────────────────────────────────
def list_pending_approvals(
    *,
    verdicts_path:    Optional[Path] = None,
    resolutions_path: Optional[Path] = None,
    include_resolved: bool = False,
) -> dict:
    """Return the structured payload the UI consumes.

    Returns:
      {
        "n_pending":   int,
        "n_resolved":  int,
        "rows": [
          {
            "hypothesis_id":            str,
            "verdict_type":             "APPROVE_FOR_PIPELINE" |
                                        "DOCTRINE_AMENDMENT_NEEDED",
            "one_line_summary":         str,
            "confidence":               float,
            "reasoning":                str,
            "similar_to_deployed":      str | None,
            "replaces_decaying":        str | None,
            "blocking_doctrine_id":     str | None,
            "proposed_amendment_summary": str | None,
            "recommended_pipeline_action": str | None,
            "risk_flags":               list[str],
            "review_ts":                str,
            "model":                    str,
            "resolved":                 bool,
            "resolution": {              # only if resolved or include_resolved
              "decision":    str,
              "rationale":   str,
              "resolved_ts": str,
              "resolved_by": str,
            } | None,
          },
          ...
        ],
      }

    Ordering: pending first (oldest review_ts first → FIFO for the
    principal's queue), then resolved (newest review_ts first).
    """
    verdicts_path    = verdicts_path    or _DEFAULT_VERDICTS_PATH
    resolutions_path = resolutions_path or _DEFAULT_RESOLUTIONS_PATH

    verdicts    = _load_verdicts(verdicts_path)
    resolutions = _load_resolutions(resolutions_path)

    pending: list[dict] = []
    resolved: list[dict] = []
    for v in verdicts:
        vt = v.get("verdict_type")
        if vt not in _PENDING_VERDICT_TYPES:
            continue
        hid = v.get("hypothesis_id")
        if not hid:
            continue
        res = resolutions.get(hid)
        is_resolved = res is not None

        row = {
            "hypothesis_id":               hid,
            "verdict_type":                vt,
            "one_line_summary":            v.get("one_line_summary", ""),
            "confidence":                  float(v.get("confidence", 0.5)),
            "reasoning":                   v.get("reasoning", ""),
            "similar_to_deployed":         v.get("similar_to_deployed"),
            "replaces_decaying":           v.get("replaces_decaying"),
            "blocking_doctrine_id":        v.get("blocking_doctrine_id"),
            "proposed_amendment_summary":  v.get("proposed_amendment_summary"),
            "recommended_pipeline_action": v.get("recommended_pipeline_action"),
            "risk_flags":                  list(v.get("risk_flags") or []),
            "review_ts":                   v.get("review_ts", ""),
            "model":                       v.get("model", ""),
            "resolved":                    is_resolved,
            "resolution":                  (
                _dc.asdict(res) if is_resolved else None
            ),
        }

        if is_resolved:
            resolved.append(row)
        else:
            pending.append(row)

    # Pending: oldest first (FIFO queue for principal)
    pending.sort(key=lambda r: r["review_ts"])
    # Resolved: newest first (recent decisions on top)
    resolved.sort(key=lambda r: r["review_ts"], reverse=True)

    rows = pending + (resolved if include_resolved else [])
    return {
        "n_pending":  len(pending),
        "n_resolved": len(resolved),
        "rows":       rows,
    }
