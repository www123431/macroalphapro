"""engine/research/discovery/queue_actions.py — close the loop on the
review queue with Promote and Skip actions.

Papers in discovery_queue.jsonl / discovery_borderline.jsonl need to
exit the queue once the user reviews them:

  Promote: writes a stub mechanism YAML to data/research/mechanism_library/
    so the strict-gate pipeline can pick it up. Removes the queue entry.
  Skip:    appends to data/research/discovery_rejected.jsonl (which
    the graveyard reader consumes). Removes the queue entry.

Both actions are LOSSLESS — original queue entry is moved, not deleted.
The audit trail (discovery_log.jsonl) is never touched.

Per [[feedback-iterate-and-solve-inflight-2026-05-29]]: ship v1 of the
loop closure now; cleanup actions (un-promote, restore from rejected)
can come later when needed.
"""
from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DISCOVERY_QUEUE = REPO_ROOT / "data" / "research" / "discovery_queue.jsonl"
DISCOVERY_BORDERLINE = REPO_ROOT / "data" / "research" / "discovery_borderline.jsonl"
DISCOVERY_REJECTED = REPO_ROOT / "data" / "research" / "discovery_rejected.jsonl"
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"


# ── Queue I/O ─────────────────────────────────────────────────────────────

def _read_queue(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_queue(path: Path, entries: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def _entry_id(entry: dict) -> str:
    """Best-available stable identifier for matching."""
    return (entry.get("source_id") or entry.get("arxiv_id")
              or entry.get("doi") or "")


def find_entry(source_id: str) -> tuple[dict | None, str | None]:
    """Return (entry, source_queue_name) or (None, None) if not found.

    Searches both primary review queue and borderline queue.
    """
    sid = (source_id or "").strip()
    if not sid:
        return None, None
    for path, name in ((DISCOVERY_QUEUE, "review"),
                          (DISCOVERY_BORDERLINE, "borderline")):
        for entry in _read_queue(path):
            if _entry_id(entry) == sid:
                return entry, name
    return None, None


def remove_entry(source_id: str) -> tuple[dict | None, str | None]:
    """Remove and return entry from whichever queue holds it."""
    sid = (source_id or "").strip()
    if not sid:
        return None, None
    for path, name in ((DISCOVERY_QUEUE, "review"),
                          (DISCOVERY_BORDERLINE, "borderline")):
        entries = _read_queue(path)
        kept = []
        removed = None
        for entry in entries:
            if _entry_id(entry) == sid and removed is None:
                removed = entry
            else:
                kept.append(entry)
        if removed is not None:
            _write_queue(path, kept)
            return removed, name
    return None, None


# ── Mechanism YAML stub generation (Promote target) ──────────────────────

_SLUG_RX = re.compile(r"[^a-z0-9_]+")


def _slug_from_title(title: str, fallback: str = "untitled") -> str:
    """Lowercase, underscored, [a-z0-9_] only — short enough to be a filename."""
    s = (title or fallback).lower()
    s = _SLUG_RX.sub("_", s)
    s = s.strip("_")
    return (s[:50] or fallback)


def _unique_mechanism_id(slug: str) -> str:
    """If slug.yaml exists, append _2, _3, ... until free."""
    base = LIBRARY_DIR / f"{slug}.yaml"
    if not base.exists():
        return slug
    i = 2
    while (LIBRARY_DIR / f"{slug}_{i}.yaml").exists():
        i += 1
    return f"{slug}_{i}"


# Family → default tunable_bindings whitelist.
# Per Huatai 自进化Skill paper (borrowed 2026-05-30): mechanism YAML
# should EXPLICITLY list which binding parameters Agent / auto-gate
# is allowed to vary. Anything else is locked logic. Prevents the
# "Agent silently rewrites the scoring function" anti-pattern.
_FAMILY_TUNABLES: dict[str, list[str]] = {
    "carry":               ["top_frac", "vol_target", "cost_bps_per_side", "rebal_freq"],
    "tsmom":               ["lookback_months", "vol_target", "cost_bps_per_side"],
    "momentum":            ["lookback_months", "skip_months", "top_frac", "vol_target"],
    "value":               ["top_frac", "vol_target", "rebal_freq", "cost_bps_per_side"],
    "quality":             ["top_frac", "vol_target", "rebal_freq", "cost_bps_per_side"],
    "low_vol":             ["lookback_months", "vol_target", "cost_bps_per_side"],
    "profitability":       ["top_frac", "vol_target", "cost_bps_per_side"],
    "investment":          ["top_frac", "vol_target", "cost_bps_per_side"],
    "residual_momentum":   ["lookback_months", "skip_months", "top_frac", "vol_target"],
    "pead":                ["hold_months", "skip_first_month", "cost_bps_per_side", "vol_target"],
    "post_earnings_drift": ["hold_months", "skip_first_month", "cost_bps_per_side", "vol_target"],
    "vol_carry":           ["lookback_months", "vol_target", "cost_bps_per_side"],
    "cross_asset_carry":   ["top_frac", "vol_target", "rebal_freq"],
    "cross_asset_tsmom":   ["lookback_months", "vol_target"],
    "lead_lag":            ["lookback_months", "top_frac", "vol_target"],
    "unknown":             [],     # no auto-gate tuning until human classifies
}


def build_mechanism_stub(entry: dict, *, target_status: str = "PENDING") -> dict:
    """Construct a mechanism YAML payload from a queue entry. The stub
    has status_in_our_book=PENDING by default; subsequent gate runs +
    library audit promote it to GREEN/RED.

    Per Huatai 自进化Skill paper (borrowed 2026-05-30): stub includes
    explicit tunable_bindings whitelist. Auto-gate may ONLY swap these
    parameters. Anything else is locked_logic_anchor (the family's
    canonical scoring rule). This prevents silent "Agent rewrites
    scoring function" and makes every iteration audit-able.
    """
    routing = entry.get("routing") or {}
    extraction = entry.get("extraction") or {}
    family = (routing.get("family")
                or extraction.get("family_guess")
                or "unknown").lower()

    return {
        "id":                    None,    # caller fills before write
        "title":                 entry.get("title") or "",
        "family":                family,
        "parent_family":         (extraction.get("parent_family_guess")
                                    or "unknown"),
        "status_in_our_book":    target_status,
        "mechanism_economics":   (extraction.get("economic_intuition")
                                    or extraction.get("mechanism_proposal")
                                    or ""),
        "required_data":         list(extraction.get("required_data_tokens") or []),
        # Huatai 借鉴 ①: explicit binding whitelist
        "tunable_bindings":      list(_FAMILY_TUNABLES.get(family, [])),
        "locked_logic_anchor":   (f"family={family} canonical scoring; "
                                    f"see engine.research.templates.<template_id>.run_*"),
        "source": {
            "kind":              entry.get("source") or "manual_nominate",
            "source_id":         entry.get("source_id"),
            "doi":               entry.get("doi"),
            "venue":             entry.get("venue"),
            "authors":           entry.get("authors"),
            "submitted_date":    entry.get("submitted_date"),
            "abs_url":           entry.get("abs_url"),
        },
        "promotion_metadata": {
            "promoted_at":       datetime.datetime.utcnow().isoformat() + "Z",
            "from_queue":        "review",
            "confidence_at_promotion": (
                routing.get("adjusted_confidence")
                or routing.get("base_confidence")
            ),
            "credibility_score": (entry.get("credibility") or {}).get("score"),
        },
    }


# ── Public actions ────────────────────────────────────────────────────────

def approve_binding(
    mechanism_id: str,
    *,
    template_id: str,
    binding: dict,
    required_data: list[str] | None = None,
) -> dict:
    """Approve a proposed (or user-edited) binding and write it to the
    library YAML's execution_template + required_data fields.

    After approval the YAML is in a state where forward_oos_runner can
    actually run it on real data — the 1-user throughput unlocker.

    Validates: template_id is registered; binding keys are whitelisted
    in tunable_bindings (Huatai 借鉴 ①); required_data tokens are
    implemented.
    """
    from engine.research.templates import TEMPLATES
    from engine.research.hygiene_tools import IMPLEMENTED_DATA

    yaml_path = LIBRARY_DIR / f"{mechanism_id}.yaml"
    if not yaml_path.exists():
        raise ValueError(f"mechanism_id {mechanism_id!r} not in library")
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"yaml parse failed: {exc}")

    if template_id not in TEMPLATES:
        raise ValueError(
            f"template_id {template_id!r} not registered; "
            f"valid: {sorted(TEMPLATES.keys())}"
        )

    # Whitelist enforcement (Huatai 借鉴 ①): only allow binding keys
    # that are in the YAML's tunable_bindings list
    tunable = set(data.get("tunable_bindings") or [])
    violations = sorted(set(binding.keys()) - tunable) if tunable else []
    if tunable and violations:
        logger.warning(
            "approve_binding: keys %s not in whitelist; filtering",
            violations,
        )
        binding = {k: v for k, v in binding.items() if k in tunable}

    # Data-token check
    unknown_data = [t for t in (required_data or [])
                      if t not in IMPLEMENTED_DATA]
    if unknown_data:
        raise ValueError(
            f"required_data tokens {unknown_data} not in IMPLEMENTED_DATA; "
            f"wire the fetcher first"
        )

    # Write back to YAML
    data["execution_template"] = {
        "template_id":      template_id,
        "template_version": 1,
        "binding":          binding,
    }
    if required_data:
        data["required_data"] = required_data
    data.setdefault("promotion_metadata", {})["binding_approved_at"] = (
        datetime.datetime.utcnow().isoformat() + "Z"
    )

    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    return {
        "ok":              True,
        "mechanism_id":    mechanism_id,
        "template_id":     template_id,
        "binding":         binding,
        "required_data":   required_data or [],
        "warnings":        (violations and
                                [f"keys {violations} filtered (not in whitelist)"]
                                or []),
    }


def validate_binding_changes_against_whitelist(
    mechanism_yaml: dict, proposed_bindings: dict,
) -> tuple[bool, list[str]]:
    """Verify that proposed_bindings only modifies keys in
    mechanism_yaml['tunable_bindings'] whitelist.

    Per Huatai 自进化Skill paper (借鉴 ①): Agent / auto-gate may only
    swap whitelisted params. Anything else is a spec violation.

    Returns: (is_valid, list_of_violations)
    """
    allowed = set(mechanism_yaml.get("tunable_bindings") or [])
    proposed_keys = set(proposed_bindings.keys())
    violations = sorted(proposed_keys - allowed)
    return (len(violations) == 0, violations)


# Failure attribution = REUSE graveyard.FailureMode enum (not duplicate).
# Per senior pushback 2026-05-30: don't create parallel taxonomies.
# Added a couple of skip-time-specific reasons that don't fit any
# graveyard FailureMode (off_topic / unclear are review-time reasons,
# not post-gate failure modes).
def _valid_failure_attributions() -> set[str]:
    """Resolve at call time so test monkeypatches of graveyard work."""
    try:
        from engine.research.graveyard import FailureMode
        gv_set = {m.value for m in FailureMode}
    except Exception:
        gv_set = set()
    # Review-time additions that don't have graveyard equivalents
    review_only = {"off_topic", "unclear"}
    return gv_set | review_only


# Module-level constant for backward-compat (callers can import this
# instead of calling the function). Computed once at import.
FAILURE_ATTRIBUTION_VALID = _valid_failure_attributions()


def promote(
    source_id: str, *,
    target_status: str = "PENDING",
    auto_gate: bool = True,
    hypothesis: str | None = None,
    propose_binding: bool = True,
) -> dict:
    """Move a queue entry into the mechanism library as a PENDING stub.

    auto_gate (default True): also trigger the strict gate via
      engine.research.discovery.auto_gate so the user gets an
      immediate GREEN/YELLOW/RED preview. Best-effort — gate failure
      does NOT undo the promote.

    hypothesis (Huatai 借鉴 ③, OPTIONAL on promote — REQUIRED on skip):
      One-sentence statement of WHAT this candidate is testing. Stored
      in promotion_metadata for audit. If None on promote, defaults to
      a stub note "[no hypothesis provided]".

    Also updates the first-author track ledger (+1 pass) per Tier 1 ②.

    Returns: {ok, mechanism_id, path, original_queue, auto_gate?}
    Raises:  ValueError if not found
    """
    removed, queue_name = remove_entry(source_id)
    if not removed:
        raise ValueError(f"source_id {source_id!r} not found in any queue")

    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    stub = build_mechanism_stub(removed, target_status=target_status)
    stub["promotion_metadata"]["from_queue"] = queue_name
    # Huatai 借鉴 ③: capture hypothesis at promotion time
    stub["promotion_metadata"]["hypothesis"] = (
        hypothesis or "[no hypothesis provided]"
    )

    # Senior 1-user throughput unlocker: LLM auto-proposes binding so
    # user can review+approve in 30s instead of writing it manually.
    # Stored under proposed_binding; the YAML's execution_template stays
    # empty until human approves (handled by /api/research/discovery/
    # approve_binding endpoint).
    proposal_dict = None
    if propose_binding:
        try:
            from engine.research.discovery.binding_proposer import (
                propose_binding as _propose,
            )
            extraction = removed.get("extraction") or {}
            prop = _propose(
                title=removed.get("title", ""),
                abstract=removed.get("abstract", ""),
                family_guess=extraction.get("family_guess") or "unknown",
                economic_intuition=extraction.get("economic_intuition") or "",
                existing_required_data=list(
                    extraction.get("required_data_tokens") or []
                ),
                use_llm=True,
            )
            proposal_dict = prop.to_dict()
            stub["proposed_binding"] = proposal_dict
        except Exception as exc:
            logger.warning("binding proposer failed for %s: %s",
                              removed.get("source_id"), exc)
            stub["proposed_binding"] = {
                "valid": False, "error": str(exc)[:300],
            }

    slug = _slug_from_title(removed.get("title", ""))
    mechanism_id = _unique_mechanism_id(slug)
    stub["id"] = mechanism_id

    out_path = LIBRARY_DIR / f"{mechanism_id}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stub, f, allow_unicode=True, sort_keys=False)

    # Tier 1 ②: author-track auto-feedback (best-effort)
    try:
        from engine.research.discovery.credibility_scorer import (
            update_author_track,
        )
        authors = (removed.get("authors") or "").strip()
        if authors:
            first_author = authors.split(";")[0].strip()
            if first_author:
                update_author_track(first_author, "pass")
    except Exception as exc:
        logger.warning("author-track update failed for promote: %s", exc)

    response: dict = {
        "ok":             True,
        "mechanism_id":   mechanism_id,
        "library_path":   str(out_path.relative_to(REPO_ROOT)),
        "original_queue": queue_name,
        "title":          removed.get("title", ""),
    }

    # Tier 1 ①: auto-gate (best-effort)
    auto_gate_dict = None
    if auto_gate:
        try:
            from engine.research.discovery.auto_gate import auto_gate as run_auto_gate
            gate_result = run_auto_gate(out_path)
            auto_gate_dict = gate_result.to_dict()
            response["auto_gate"] = auto_gate_dict
        except Exception as exc:
            logger.warning("auto_gate failed for %s: %s", mechanism_id, exc)
            response["auto_gate"] = {"ok": False, "error": str(exc)[:300]}

    # Senior B: register for Forward OOS observation (best-effort).
    # Closes the discovery → book bridge so we can later compare
    # synthetic auto-gate verdict against real forward-OOS performance.
    try:
        from engine.research.discovery.forward_oos_observer import (
            register_for_forward_oos,
        )
        watchlist_entry = register_for_forward_oos(
            mechanism_id,
            promoted_from=queue_name or "review",
            auto_gate_result=auto_gate_dict,
        )
        response["forward_oos_watchlist"] = {
            "registered":   True,
            "state":        watchlist_entry.state,
            "track_until":  watchlist_entry.track_until,
        }
    except Exception as exc:
        logger.warning("forward_oos register failed for %s: %s",
                          mechanism_id, exc)
        response["forward_oos_watchlist"] = {"registered": False,
                                                  "error": str(exc)[:200]}

    return response


def skip(
    source_id: str, *,
    reason: str = "user_skip",
    failure_attribution: str | None = None,
) -> dict:
    """Move a queue entry to the discovery_rejected log (which feeds
    the graveyard so future similar candidates get auto-flagged).

    Also updates the first-author track ledger (+1 fail) per Tier 1 ②
    — symmetric with promote (+1 pass) so author posteriors update
    in both directions.

    failure_attribution (Huatai 借鉴 ③): one of FAILURE_ATTRIBUTION_VALID
    enum values categorizing WHY this candidate is being skipped. When
    None, defaults to "unclear". Stored in rejection_record + propagated
    into graveyard reader so future similar candidates inherit the
    same failure_mode classification.

    Returns: {ok, rejected_path, original_queue}
    Raises:  ValueError if not found
    """
    # Normalize + validate failure_attribution against graveyard.FailureMode
    # (no parallel taxonomy — single source of truth).
    valid_set = _valid_failure_attributions()
    attribution = (failure_attribution or "unclear").lower()
    if attribution not in valid_set:
        logger.warning(
            "skip: failure_attribution %r not in valid set %s; "
            "defaulting to 'unclear'",
            attribution, sorted(valid_set),
        )
        attribution = "unclear"

    removed, queue_name = remove_entry(source_id)
    if not removed:
        raise ValueError(f"source_id {source_id!r} not found in any queue")

    rejection_record = {
        **removed,
        "skipped_at":          datetime.datetime.utcnow().isoformat() + "Z",
        "skip_reason":         reason,
        "failure_attribution": attribution,
        "from_queue":          queue_name,
    }
    DISCOVERY_REJECTED.parent.mkdir(parents=True, exist_ok=True)
    with DISCOVERY_REJECTED.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rejection_record, ensure_ascii=False, default=str) + "\n")

    # Tier 1 ②: author-track auto-feedback (best-effort, symmetric with promote)
    try:
        from engine.research.discovery.credibility_scorer import (
            update_author_track,
        )
        authors = (removed.get("authors") or "").strip()
        if authors:
            first_author = authors.split(";")[0].strip()
            if first_author:
                update_author_track(first_author, "fail")
    except Exception as exc:
        logger.warning("author-track update failed for skip: %s", exc)

    return {
        "ok":             True,
        "rejected_path":  str(DISCOVERY_REJECTED.relative_to(REPO_ROOT)),
        "original_queue": queue_name,
        "title":          removed.get("title", ""),
    }
