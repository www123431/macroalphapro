"""engine.agents.strengthener.sleeve_strengthen_scan — Stage B P3b.

Per-sleeve weekly strengthen scan orchestrator.

For each deployed sleeve in data/research/mechanism_library/*.yaml:
  1. Build SleeveContext (load YAML + query recent family REDs +
     query decay alerts + load doctrine snippets)
  2. Call sleeve_strengthen_proposer.run_strengthen_proposer (P3a)
  3. Adapt each StrengthenProposal → Hypothesis dataclass with
     extraction_method=LLM_SYNTHESIS, addresses_decay_in=sleeve_id,
     tags carrying improvement_kind + scan provenance
  4. Persist via existing hypothesis store
  5. Idempotency: track (sleeve_id, ISO_week) so re-runs in same
     week don't re-scan unless force=True

Cost discipline:
  - Default max_sleeves cap (5) so a weekly run doesn't burn through
    all 9 deployed sleeves' LLM budget in one go. Rotates through
    them across weeks (oldest-scanned-first).
  - Skip sleeves with status_in_our_book NOT in {DEPLOYED,
    cousin_anchor} — research-stage / decommissioned sleeves aren't
    worth strengthen scans.
  - dry_run mode: builds context + calls LLM but skips persist.

Per [[feedback-piece-by-piece-not-batch-2026-06-05]]: this is the
plumbing piece. P3c wires it into chief_of_staff weekly session.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import uuid
from pathlib import Path
from typing import Optional

import yaml

from engine.agents.strengthener.sleeve_strengthen_proposer import (
    SleeveContext, StrengthenProposal, run_strengthen_proposer,
)
from engine.research_store.hypothesis.schema import (
    ExtractionMethod, HypothesisDirection, HypothesisReviewState,
)
from engine.research_store.red_lessons.mechanism_families import (
    MechanismFamily,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LIBRARY_DIR    = (_REPO_ROOT / "data" / "research" / "mechanism_library")
SCAN_STATE_DIR = (_REPO_ROOT / "data" / "strengthener"
                    / "sleeve_scan_state")


# Statuses that warrant a strengthen scan. Research-only / decommissioned
# sleeves don't earn the LLM call.
_SCAN_WORTHY_STATUSES = {"DEPLOYED", "cousin_anchor"}


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_week_id(dt: Optional[_dt.datetime] = None) -> str:
    """e.g. '2026-W23'. Idempotency key."""
    d = dt or _dt.datetime.utcnow()
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


# ────────────────────────────────────────────────────────────────────
# Library YAML enumeration
# ────────────────────────────────────────────────────────────────────
def _load_sleeve_yamls(library_dir: Optional[Path] = None
                         ) -> list[dict]:
    """Return list of parsed sleeve YAMLs. Skip non-sleeve files
    (those starting with '_' like _canonical_papers_tier1_2.yaml)."""
    d = library_dir or LIBRARY_DIR
    if not d.is_dir():
        return []
    out: list[dict] = []
    for p in sorted(d.glob("*.yaml")):
        if p.stem.startswith("_"):
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                row = yaml.safe_load(f)
        except Exception as exc:
            logger.warning("sleeve_scan: failed to parse %s: %s",
                            p.name, exc)
            continue
        if isinstance(row, dict):
            row["_yaml_path"] = str(p)
            out.append(row)
    return out


def _is_scan_worthy(sleeve_dict: dict) -> bool:
    """Apply 'DEPLOYED or cousin_anchor only' filter. Status check
    falls back to 'purpose' field for older YAMLs that don't carry
    status_in_our_book."""
    status = (sleeve_dict.get("status_in_our_book") or "").upper()
    purpose = (sleeve_dict.get("purpose") or "").lower()
    return (status in {s.upper() for s in _SCAN_WORTHY_STATUSES}
              or purpose in {"deployed_sleeve", "cousin_anchor"})


# ────────────────────────────────────────────────────────────────────
# SleeveContext builder — JOIN with recent state
# ────────────────────────────────────────────────────────────────────
def _build_context(sleeve: dict,
                     *, lookback_days: int = 30
                    ) -> SleeveContext:
    """Build SleeveContext from a parsed YAML row + recent events.

    Pulls recent factor_verdict_filed RED in same family +
    doctrine_signal_detected events linked to this sleeve."""
    sleeve_id = str(sleeve.get("id") or "")
    family    = str(sleeve.get("family") or "")
    cutoff = (_dt.datetime.utcnow()
              - _dt.timedelta(days=lookback_days)
              ).strftime("%Y-%m-%dT%H:%M:%SZ")

    recent_red_ids: tuple[str, ...] = ()
    recent_decay_ids: tuple[str, ...] = ()
    try:
        from engine.research_store.store import filter_events
        red_events = filter_events(
            event_type="factor_verdict_filed",
            verdict="RED",
            family=family,
            since=cutoff,
            limit=20,
        )
        recent_red_ids = tuple(ev.event_id for ev in red_events
                                 if ev.event_id)[:10]

        decay_events = filter_events(
            event_type="doctrine_signal_detected",
            subject_id=sleeve_id,
            since=cutoff,
            limit=10,
        )
        recent_decay_ids = tuple(ev.event_id for ev in decay_events
                                   if ev.event_id)[:5]
    except Exception as exc:
        logger.warning("sleeve_scan: filter_events failed for %s: %s",
                        sleeve_id, exc)

    # Deployed summary — short one-liner from YAML state
    deployed_summary = _build_deployed_summary(sleeve)

    return SleeveContext(
        sleeve_id              = sleeve_id,
        family                 = family,
        canonical_paper_id     = str(sleeve.get("canonical_paper_id") or ""),
        mechanism_economics    = str(sleeve.get("mechanism_economics") or ""),
        canonical_universe     = str(sleeve.get("canonical_universe") or ""),
        typical_sample         = str(sleeve.get("typical_sample") or ""),
        deployed_summary       = deployed_summary,
        recent_family_red_ids  = recent_red_ids,
        recent_decay_alert_ids = recent_decay_ids,
        doctrine_snippet_ids   = (),    # P3c may wire doctrine ChromaDB
        snapshot_ts            = _utc_iso(),
    )


def _build_deployed_summary(sleeve: dict) -> str:
    """One-liner with the most-important deployment state. Pulls
    purpose / status / any KPI in post_pub_decay.our_observed."""
    parts = []
    parts.append(f"purpose={sleeve.get('purpose', '?')}")
    status = sleeve.get("status_in_our_book")
    if status:
        parts.append(f"status={status}")
    # Observed sharpe / decay if present
    decay = sleeve.get("post_pub_decay") or {}
    obs = decay.get("our_observed") or {}
    sharpe = obs.get("summary_sharpe_observed")
    if sharpe is not None:
        parts.append(f"observed_sharpe={sharpe}")
    last_upd = obs.get("last_updated")
    if last_upd:
        parts.append(f"last_kpi_refresh={last_upd}")
    return " · ".join(parts)


# ────────────────────────────────────────────────────────────────────
# Idempotency state — track (sleeve_id, iso_week) tuples
# ────────────────────────────────────────────────────────────────────
def _scan_state_path(*, state_dir: Optional[Path] = None) -> Path:
    return (state_dir or SCAN_STATE_DIR) / "scanned_weeks.json"


def _load_scan_state(*, state_dir: Optional[Path] = None
                       ) -> dict[str, str]:
    """{sleeve_id: last_scanned_iso_week}. Missing file → {}."""
    p = _scan_state_path(state_dir=state_dir)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("sleeve_scan_state load failed: %s", exc)
        return {}


def _save_scan_state(state: dict[str, str],
                      *, state_dir: Optional[Path] = None) -> None:
    p = _scan_state_path(state_dir=state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2, sort_keys=True),
                  encoding="utf-8")


# ────────────────────────────────────────────────────────────────────
# StrengthenProposal → Hypothesis adapter
# ────────────────────────────────────────────────────────────────────
def _resolve_family_enum(family_str: str) -> MechanismFamily:
    if not family_str:
        return MechanismFamily.OTHER
    try:
        return MechanismFamily(family_str.upper())
    except (ValueError, AttributeError):
        return MechanismFamily.OTHER


def _proposal_to_hypothesis(
    prop: StrengthenProposal,
    *, ctx: SleeveContext,
) -> "Hypothesis":
    """Adapt a StrengthenProposal → persistable Hypothesis."""
    from engine.research_store.hypothesis.schema import Hypothesis

    now = _utc_iso()
    # synthesizes_paper_ids: canonical paper of the sleeve plus any
    # extra papers the LLM cited
    synth_papers: list[str] = []
    if ctx.canonical_paper_id:
        synth_papers.append(ctx.canonical_paper_id)
    for pid in (prop.references_paper_ids or ()):
        if pid and pid not in synth_papers:
            synth_papers.append(pid)

    return Hypothesis(
        hypothesis_id        = str(uuid.uuid4()),
        source_paper_id      = "",   # synthesis path
        version              = 1,
        parent_hypothesis_id = None,
        source_chunk_ids     = (),
        verbatim_quotes      = (),
        claim                = prop.claim,
        mechanism_family     = _resolve_family_enum(ctx.family),
        mechanism_subtype    = prop.mechanism_subtype,
        predicted_direction  = HypothesisDirection.POSITIVE,
        predicted_magnitude  = prop.predicted_magnitude,
        required_data        = prop.required_data,
        test_methodology     = prop.test_methodology,
        extraction_method    = ExtractionMethod.LLM_SYNTHESIS,
        review_state         = HypothesisReviewState.PROPOSED,
        created_ts           = now,
        updated_ts           = now,
        created_by           = "engine.agents.strengthener.sleeve_strengthen_scan",
        tags                 = (
            "source:active_b_sleeve_scan",
            f"sleeve:{ctx.sleeve_id}",
            f"improvement_kind:{prop.improvement_kind}",
            f"scan_week:{_iso_week_id()}",
        ),
        synthesizes_paper_ids = tuple(synth_papers),
        synthesizes_event_ids = (
            ctx.recent_family_red_ids + ctx.recent_decay_alert_ids
        ),
        addresses_decay_in    = ctx.sleeve_id,
    )


# ────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────
def run_sleeve_strengthen_scan(
    *,
    max_sleeves:     int = 5,
    lookback_days:   int = 30,
    force:           bool = False,
    dry_run:         bool = False,
    library_dir:     Optional[Path] = None,
    state_dir:       Optional[Path] = None,
    hypotheses_path: Optional[Path] = None,
) -> dict:
    """Run the active-B per-sleeve scan over deployed sleeves.

    Picks the `max_sleeves` oldest-scanned (rotation) deployed
    sleeves, builds context for each, calls the LLM proposer, and
    persists 0-3 proposed Hypotheses per sleeve. Idempotent within
    ISO week (re-runs same week skip already-scanned sleeves unless
    force=True).

    Returns:
      {
        run_ts:                  iso,
        iso_week:                'YYYY-WNN',
        dry_run:                 bool,
        n_sleeves_eligible:      int,
        n_sleeves_scanned:       int,
        n_sleeves_skipped:       int,   # already-this-week
        n_proposals_total:       int,
        n_proposals_persisted:   int,
        proposed_ids:            list[str],  # hypothesis_ids
        per_sleeve:              [
          {sleeve_id, n_proposals, hypothesis_ids, errors},
          ...
        ],
        errors:                  list[str],
      }
    """
    from engine.research_store.hypothesis.store import save_hypothesis

    run_ts = _utc_iso()
    iso_week = _iso_week_id()
    result = {
        "run_ts":                run_ts,
        "iso_week":              iso_week,
        "dry_run":               dry_run,
        "n_sleeves_eligible":    0,
        "n_sleeves_scanned":     0,
        "n_sleeves_skipped":     0,
        "n_proposals_total":     0,
        "n_proposals_persisted": 0,
        "proposed_ids":          [],
        "per_sleeve":            [],
        "errors":                [],
    }

    sleeves = [s for s in _load_sleeve_yamls(library_dir=library_dir)
                if _is_scan_worthy(s)]
    result["n_sleeves_eligible"] = len(sleeves)
    if not sleeves:
        return result

    state = _load_scan_state(state_dir=state_dir)

    # Rotation: oldest-scanned-first (sleeves never scanned have
    # last_week='' which sorts first naturally)
    def _last_scanned(s):
        return state.get(s.get("id") or "", "")
    sleeves.sort(key=_last_scanned)

    scanned_count = 0
    for sleeve in sleeves:
        sleeve_id = sleeve.get("id") or ""
        if not sleeve_id:
            continue
        if scanned_count >= max_sleeves:
            break
        if not force and state.get(sleeve_id) == iso_week:
            result["n_sleeves_skipped"] += 1
            continue

        per_sleeve = {
            "sleeve_id":       sleeve_id,
            "n_proposals":     0,
            "hypothesis_ids":  [],
            "errors":          [],
        }
        try:
            ctx = _build_context(sleeve, lookback_days=lookback_days)
            proposals = run_strengthen_proposer(ctx)
        except Exception as exc:
            logger.exception("sleeve_scan: proposer raised for %s",
                              sleeve_id)
            per_sleeve["errors"].append(f"proposer: {exc}")
            result["per_sleeve"].append(per_sleeve)
            result["errors"].append(f"{sleeve_id}: proposer: {exc}")
            continue

        scanned_count += 1
        result["n_sleeves_scanned"] += 1
        per_sleeve["n_proposals"] = len(proposals)
        result["n_proposals_total"] += len(proposals)

        for prop in proposals:
            try:
                h = _proposal_to_hypothesis(prop, ctx=ctx)
            except Exception as exc:
                logger.exception("sleeve_scan: adapt failed for %s",
                                  sleeve_id)
                per_sleeve["errors"].append(f"adapt: {exc}")
                continue

            per_sleeve["hypothesis_ids"].append(h.hypothesis_id)
            result["proposed_ids"].append(h.hypothesis_id)

            if dry_run:
                continue
            try:
                save_hypothesis(h, path=hypotheses_path,
                                  skip_cross_checks=True)
                result["n_proposals_persisted"] += 1
            except Exception as exc:
                logger.exception("sleeve_scan: persist failed for %s",
                                  h.hypothesis_id)
                per_sleeve["errors"].append(
                    f"persist:{h.hypothesis_id}: {exc}")

        # Mark this sleeve scanned this week (even if proposer returned
        # 0 — that IS a result; don't re-burn LLM budget on it)
        if not dry_run:
            state[sleeve_id] = iso_week

        result["per_sleeve"].append(per_sleeve)

    # Persist state once at end (only if anything actually scanned)
    if not dry_run and result["n_sleeves_scanned"] > 0:
        try:
            _save_scan_state(state, state_dir=state_dir)
        except Exception as exc:
            logger.warning("sleeve_scan_state save failed: %s", exc)
            result["errors"].append(f"state_persist: {exc}")

    return result
