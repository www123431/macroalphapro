"""engine/research/library_promoter.py — Phase 4e: promote an L4
iteration to a library DRAFT YAML.

Why DRAFT, not direct library entry: the canonical library YAMLs have
~20 mandatory fields (provenance, snooping audit, post-pub decay,
factor exposure, cost model, family classification, role) — most of
which need senior judgment that an L4 propose+critique+pipeline does
NOT cover. Writing direct entries would produce broken library state
on first use.

Draft path: data/research/mechanism_library/_drafts/<id>.yaml
  - Drafts directory is gitignored from library validators (they
    skip names starting with _)
  - Senior reviews + fills required fields + moves to canonical
    location to promote properly

Returns path + a checklist of fields needing senior input.
"""
from __future__ import annotations

import datetime as _dt
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DRAFTS_DIR = REPO_ROOT / "data" / "research" / "mechanism_library" / "_drafts"


_SAFE_ID_RE = re.compile(r"[^a-z0-9_]")


def _slugify(title: str) -> str:
    s = title.lower().strip()
    s = s.replace(" ", "_").replace("-", "_")
    s = _SAFE_ID_RE.sub("", s)
    return s[:48] or "unnamed"


def promote_iteration_to_draft(iteration: dict) -> dict:
    """Write a draft YAML for the given L4 iteration row.

    Caller is expected to have verified that:
      - effective_consensus == APPROVE
      - pipeline.ran is True
      - pipeline.final_decision indicates promotion-worthy outcome

    Returns:
      {
        "draft_path": str,
        "draft_id":   str,
        "checklist":  [str, ...]   # fields senior must fill
      }
    """
    proposal = iteration.get("proposal") or {}
    pipeline = iteration.get("pipeline") or {}
    council = iteration.get("council") or {}

    title = str(proposal.get("title") or "unnamed_promotion")
    draft_id = _slugify(title)
    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)

    # If a draft with the same id already exists, suffix with timestamp
    target = DRAFTS_DIR / f"{draft_id}.yaml"
    if target.exists():
        ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        draft_id = f"{draft_id}__{ts}"
        target = DRAFTS_DIR / f"{draft_id}.yaml"

    draft: dict = {
        "_schema_version": 2,
        "_draft": True,
        "_draft_created_at": _dt.datetime.utcnow().isoformat() + "Z",
        "_draft_source": {
            "iteration_id":  iteration.get("iteration_id"),
            "workflow_id":   iteration.get("workflow_id"),
            "council_run_id": council.get("run_id"),
        },

        # Identity (provided by architect)
        "id":            draft_id,
        "family":        proposal.get("family"),
        "parent_family": proposal.get("parent_family"),
        "purpose":       "candidate",  # drafts always start here
        "proposed_role": proposal.get("proposed_role"),

        # Provenance — TODO senior completes
        "canonical_paper_id": None,
        "key_followup_ids":   [],

        # Status — drafts NEVER auto-deploy
        "status_in_our_book": "PROPOSED_DRAFT",

        # Origin context from L4
        "_l4_origin": {
            "council_consensus":  council.get("consensus"),
            "council_rationale":  (council.get("rationale") or "")[:1500],
            "human_override":     iteration.get("human_override"),
            "pipeline_decision":  pipeline.get("final_decision"),
            "pipeline_rationale": (pipeline.get("rationale") or "")[:1500],
            "verdict_alignment":  iteration.get("verdict_alignment"),
        },

        # Mechanism description from architect
        "purpose_text":   (proposal.get("economics_text") or "")[:3000],
        "motivation":     (proposal.get("motivation") or "")[:1500],
        "required_data":  proposal.get("required_data") or [],

        # ── SENIOR-FILL FIELDS (validators require these for non-draft) ──
        "was_known_before_our_data_cutoff": None,
        "post_pub_decay":                   None,
        "factor_exposure":                  None,
        "cost_model":                       None,
        "expected_capacity_usd":            None,
        "expected_sharpe_range":            None,
    }

    target.write_text(
        yaml.safe_dump(draft, sort_keys=False, allow_unicode=True,
                       default_flow_style=False),
        encoding="utf-8",
    )

    checklist = [
        "canonical_paper_id   (verify via query_master_index first)",
        "key_followup_ids     (verified papers only)",
        "was_known_before_our_data_cutoff  (snooping risk audit)",
        "post_pub_decay       (McLean-Pontiff / Linnainmaa-Roberts deltas)",
        "factor_exposure      (BARRA tilt: MKT / SMB / HML / MOM / RMW / CMA)",
        "cost_model           (Almgren-Chriss params, NOT scalar bp)",
        "expected_capacity_usd  (deployable AUM ceiling)",
        "expected_sharpe_range  (honest IS estimate from pipeline)",
        "status_in_our_book = PROPOSED_DRAFT until canonical move",
    ]

    return {
        "draft_path": str(target.relative_to(REPO_ROOT)),
        "draft_id":   draft_id,
        "checklist":  checklist,
    }
