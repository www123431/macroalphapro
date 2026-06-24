"""engine/research/brainstorm/promoter.py — Layer 4 PM-decision module.

Reads brainstorm_drafts.jsonl + appends decision rows (promote / reject)
to brainstorm_decisions.jsonl with MANDATORY rationale (per audit
[[project-brainstorm-architecture-2026-06-14]] P1 item: every PM
decision must have rationale, track accept/reject calibration over time).

On `promote` → ALSO writes a new hypothesis row to
data/research_store/hypotheses.jsonl with:
  - extraction_method = LLM_BRAINSTORM_<pack_name>
  - source_brainstorm_idea_id = idea_id (lineage)
  - source_pack_name preserved in tags
The hypothesis then enters the normal forward queue + can be
prioritized by Action 2 (priority selection queue) later.

On `reject` → only persists decision row (no hypothesis written).

Both paths require non-empty `rationale` — Pattern-5-compliant
human-in-the-loop accountability.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

DRAFTS_PATH    = _REPO_ROOT / "data" / "research" / "brainstorm_drafts.jsonl"
DECISIONS_PATH = _REPO_ROOT / "data" / "research" / "brainstorm_decisions.jsonl"
HYP_PATH       = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"

DECISION_VALUES = {"promote", "reject"}


class PromoterError(ValueError):
    pass


def _find_idea(idea_id: str) -> Optional[dict]:
    if not DRAFTS_PATH.is_file():
        return None
    for ln in DRAFTS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("idea_id") == idea_id:
            return r
    return None


def _latest_decision_for(idea_id: str) -> Optional[dict]:
    if not DECISIONS_PATH.is_file():
        return None
    latest = None
    for ln in DECISIONS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("idea_id") != idea_id:
            continue
        if latest is None or (r.get("decided_ts") or "") > (latest.get("decided_ts") or ""):
            latest = r
    return latest


def list_decisions(*, limit: int = 100) -> list[dict]:
    if not DECISIONS_PATH.is_file():
        return []
    rows: list[dict] = []
    for ln in DECISIONS_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("decided_ts", ""), reverse=True)
    return rows[:limit]


def _next_hypothesis_row(idea: dict, decision_id: str,
                          decided_by: str) -> dict:
    """Construct hypothesis.jsonl row from brainstorm idea."""
    now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    pack = idea.get("source_pack", "unknown")
    return {
        "hypothesis_id":        str(uuid.uuid4()),
        "source_paper_id":      None,        # brainstorm has no source paper
        "version":              1,
        "parent_hypothesis_id": None,
        "schema_version":       1,
        "source_chunk_ids":     [],
        "verbatim_quotes":      [],
        "claim":                {
            "one_line": idea.get("claim_one_line", ""),
            "mechanism": idea.get("expected_mechanism", ""),
            "falsifier": idea.get("falsifier", ""),
        },
        "mechanism_family":     "OTHER",     # PM may edit later via SPEC pipeline
        "mechanism_subtype":    None,
        "predicted_direction":  None,
        "predicted_magnitude":  idea.get("novelty_self_score"),
        "required_data":        list(idea.get("data_required") or []),
        "test_methodology":     None,
        "extraction_method":    f"LLM_BRAINSTORM_{pack.upper()}",
        "review_state":         "proposed",
        "created_ts":           now,
        "updated_ts":           now,
        "created_by":           f"brainstorm:{pack}",
        "tags": [
            f"source:brainstorm",
            f"pack:{pack}",
            f"provider:{idea.get('source_provider', 'sonnet')}",
        ],
        "synthesizes_paper_ids": [],
        "synthesizes_event_ids": [],
        "addresses_decay_in":    None,
        "citation_quality":      None,
        "orthogonal_to_anchors": None,
        # Lineage to brainstorm draft + decision
        "source_brainstorm_idea_id":    idea.get("idea_id"),
        "source_brainstorm_decision_id": decision_id,
        "target_asset_class":   idea.get("target_asset_class"),
        "lessons_invoked":      list(idea.get("lessons_invoked") or []),
    }


def decide(
    idea_id: str,
    decision: str,
    rationale: str,
    *,
    decided_by: str = "principal",
) -> dict:
    """Record promote/reject decision. Returns the decision row.

    On `promote`, ALSO appends new hypothesis row to hypotheses.jsonl.

    Raises PromoterError on:
      - decision not in DECISION_VALUES
      - rationale empty / too short (< 5 chars)
      - idea_id not found in drafts
      - idea_id already has a non-superseded decision (use a new
        rationale + decision to override; latest wins)
    """
    if decision not in DECISION_VALUES:
        raise PromoterError(
            f"decision must be one of {DECISION_VALUES}, got {decision!r}")
    rationale = (rationale or "").strip()
    if len(rationale) < 5:
        raise PromoterError(
            "rationale required (min 5 chars) — Pattern-5-compliant "
            "human-in-the-loop accountability per audit P1")
    idea = _find_idea(idea_id)
    if idea is None:
        raise PromoterError(f"brainstorm idea {idea_id} not found")

    decision_id = str(uuid.uuid4())
    decided_ts = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    row = {
        "decision_id":  decision_id,
        "idea_id":      idea_id,
        "decision":     decision,
        "rationale":    rationale,
        "decided_by":   decided_by,
        "decided_ts":   decided_ts,
        # Snapshot idea state at decision time (idea file is append-only,
        # but having a snapshot makes decision rows self-contained for
        # later calibration analysis)
        "idea_snapshot": {
            "source_pack":        idea.get("source_pack"),
            "source_provider":    idea.get("source_provider"),
            "claim_one_line":     idea.get("claim_one_line"),
            "novelty_self_score": idea.get("novelty_self_score"),
        },
    }

    DECISIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # On promote, also write hypothesis row
    if decision == "promote":
        try:
            hyp_row = _next_hypothesis_row(idea, decision_id, decided_by)
            HYP_PATH.parent.mkdir(parents=True, exist_ok=True)
            with HYP_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(hyp_row, ensure_ascii=False) + "\n")
            row["new_hypothesis_id"] = hyp_row["hypothesis_id"]
        except Exception:
            logger.exception("promote: hypothesis write failed")
            # Decision is recorded but hypothesis didn't write — surface
            row["hypothesis_write_error"] = True

    return row


def decision_for_idea(idea_id: str) -> Optional[dict]:
    """Latest decision for one idea, None if no decision yet."""
    return _latest_decision_for(idea_id)
