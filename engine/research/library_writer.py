"""engine/research/library_writer.py — gate → Mechanism Library feedback loop.

This is the NEW1 wire-up from the rigor audit. Without this module the
library is a frozen one-way explainer; with it the library evolves from
our own gate runs as additional post-publication evidence.

Contract (see docs/decisions/library_gate_feedback_loop_2026-05-29.md):

  After every gate run that writes to gate_runs.jsonl, this module:
  1. Matches candidate name → mechanism_id via candidate_to_mechanism_map.yaml
  2. Appends gate_run_id to library YAML post_pub_decay.our_observed.gate_run_ids
  3. Recomputes summary_sharpe_observed (mean across our runs)
  4. Recomputes delta_vs_published_lit (when MP 2016 range present)
  5. Updates last_updated
  6. Updates our_test_record (latest verdict + appends gate_run_id)
  7. Auto-flips UNTESTED → YELLOW (NOT DEPLOYED — that's a human decision)
  8. Writes the audit trail entry to library_updates.jsonl
  9. NEVER auto-flips audit_signature — that stays human-only

Doctrine:
- NEVER auto-create a new library YAML. Unmapped candidates go to
  orphan_candidates.jsonl for human review.
- NEVER touch published-literature decay fields (mclean_pontiff_2016 /
  post_2020_replications) — those are locked. Our evidence goes in
  our_observed only.
- Use ruamel.yaml to preserve comments and key order across round-trips.
  Falls back to PyYAML if ruamel unavailable (with warning).
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
MAP_PATH = REPO_ROOT / "data" / "research" / "candidate_to_mechanism_map.yaml"
ORPHAN_LOG = REPO_ROOT / "data" / "research" / "orphan_candidates.jsonl"
UPDATE_LOG = REPO_ROOT / "data" / "research" / "library_updates.jsonl"


# ── YAML round-trip helpers (ruamel preferred for comment preservation) ─

try:
    from ruamel.yaml import YAML

    _yaml_rt = YAML()
    _yaml_rt.preserve_quotes = True
    _yaml_rt.indent(mapping=2, sequence=4, offset=2)
    _yaml_rt.width = 4096
    _RUAMEL_AVAILABLE = True
except ImportError:
    import yaml as _pyyaml

    _yaml_rt = None
    _RUAMEL_AVAILABLE = False
    logger.warning(
        "ruamel.yaml not available; falling back to PyYAML. "
        "YAML comments will be lost on round-trip."
    )


def _load_yaml(path: Path):
    text = path.read_text(encoding="utf-8")
    if _RUAMEL_AVAILABLE:
        return _yaml_rt.load(text)
    return _pyyaml.safe_load(text)


def _save_yaml(path: Path, data) -> None:
    if _RUAMEL_AVAILABLE:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            _yaml_rt.dump(data, f)
    else:
        path.write_text(
            _pyyaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


# ── Map lookup ──────────────────────────────────────────────────────────

def _lookup_candidate_to_mechanism(name: str) -> str | None:
    """Return library mechanism_id for a candidate name, or None if unmapped.

    Returns None when name is not in the map OR is explicitly mapped to null
    (meaning "tracked but no library entry yet"). Both cases trigger orphan
    logging downstream."""
    if not MAP_PATH.exists():
        logger.warning("candidate_to_mechanism_map.yaml not found at %s", MAP_PATH)
        return None
    if _RUAMEL_AVAILABLE:
        with MAP_PATH.open("r", encoding="utf-8") as f:
            mapping_doc = _yaml_rt.load(f)
    else:
        mapping_doc = _pyyaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    mappings = (mapping_doc or {}).get("mappings", {})
    raw = mappings.get(name)
    if raw is None:
        return None
    return str(raw)


# ── Logging helpers ─────────────────────────────────────────────────────

def _log_orphan(gate_run_entry: dict, reason: str) -> None:
    ORPHAN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "candidate_name": gate_run_entry.get("name"),
        "verdict": gate_run_entry.get("verdict"),
        "reason": reason,
        "gate_run_ts": gate_run_entry.get("ts"),
    }
    with ORPHAN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _append_update_log(entry: dict) -> None:
    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with UPDATE_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Summary statistics (the actual evidence aggregation) ────────────────

def _recompute_summary_sharpe(our_observed_ids: list, all_gate_runs: list[dict]) -> float | None:
    """Mean standalone_sharpe across all our gate runs for this mechanism."""
    matching = [
        r for r in all_gate_runs
        if r.get("ts") in our_observed_ids or r.get("gate_run_id") in our_observed_ids
    ]
    sharpes = [r.get("standalone_sharpe") for r in matching
               if r.get("standalone_sharpe") is not None]
    if not sharpes:
        return None
    return round(sum(sharpes) / len(sharpes), 3)


def _compute_delta_vs_lit(entry, our_sharpe: float | None) -> float | None:
    """v1: returns None. The mclean_pontiff_2016.delta_range_estimate field
    semantics is "post-pub decay ratio" (e.g. -0.42 means 42% return
    reduction post-publication), NOT an absolute Sharpe level. Comparing
    our observed Sharpe to that midpoint is apples-to-oranges.

    Proper computation requires `lit_full_sample_sharpe_estimate` to be
    added to the library schema first (TODO v2). Until then this returns
    None deliberately rather than emit semantically-broken numbers."""
    return None


# ── Core update function ────────────────────────────────────────────────

def update_library_from_gate_run(
    gate_run_entry: dict,
    *,
    all_gate_runs: list[dict] | None = None,
    dry_run: bool = False,
) -> dict:
    """Update the matching mechanism YAML's our_observed block from a fresh
    gate-run verdict.

    Args:
      gate_run_entry: the dict that gets appended to gate_runs.jsonl
                       (output of engine.research.pipeline.run_gate)
      all_gate_runs:   optional pre-loaded ledger for summary stats; if
                       None, loaded on demand
      dry_run:         if True, compute the patched YAML but do NOT write.
                       Used by tests.

    Returns: {
      mechanism_id:   str | None
      orphan:         bool                # True if no library mapping
      updated_fields: list[str]
      promoted_to_candidate: bool         # UNTESTED → YELLOW only
      yaml_path:      str | None
    }
    """
    candidate_name = gate_run_entry.get("name")
    if not candidate_name:
        logger.warning("gate run entry missing name; skipping library update")
        return {"mechanism_id": None, "orphan": True, "updated_fields": [],
                "promoted_to_candidate": False, "yaml_path": None}

    mechanism_id = _lookup_candidate_to_mechanism(candidate_name)
    if not mechanism_id:
        _log_orphan(gate_run_entry, "no candidate_to_mechanism_map entry")
        return {"mechanism_id": None, "orphan": True, "updated_fields": [],
                "promoted_to_candidate": False, "yaml_path": None}

    yaml_path = LIBRARY_DIR / f"{mechanism_id}.yaml"
    if not yaml_path.exists():
        _log_orphan(gate_run_entry,
                    f"library YAML {mechanism_id}.yaml missing on disk")
        return {"mechanism_id": mechanism_id, "orphan": True,
                "updated_fields": [], "promoted_to_candidate": False,
                "yaml_path": None}

    entry = _load_yaml(yaml_path)
    updated_fields: list[str] = []

    # 1. Update post_pub_decay.our_observed
    ppd = entry.setdefault("post_pub_decay", {})
    observed = ppd.setdefault("our_observed", {})
    observed.setdefault("gate_run_ids", [])

    gate_run_id = gate_run_entry.get("ts") or gate_run_entry.get("gate_run_id")
    if gate_run_id and gate_run_id not in observed["gate_run_ids"]:
        observed["gate_run_ids"].append(gate_run_id)
        updated_fields.append("our_observed.gate_run_ids")

    if all_gate_runs is None:
        from engine.research.pipeline import read_ledger
        all_gate_runs = read_ledger(limit=10000)

    our_sharpe = _recompute_summary_sharpe(observed["gate_run_ids"], all_gate_runs)
    if our_sharpe is not None:
        observed["summary_sharpe_observed"] = our_sharpe
        updated_fields.append("our_observed.summary_sharpe_observed")

    delta_vs_lit = _compute_delta_vs_lit(entry, our_sharpe)
    if delta_vs_lit is not None:
        observed["delta_vs_published_lit"] = delta_vs_lit
        updated_fields.append("our_observed.delta_vs_published_lit")

    observed["last_updated"] = datetime.date.today().isoformat()
    updated_fields.append("our_observed.last_updated")

    # 2. Update our_test_record (always append; keep latest verdict)
    rec = entry.setdefault("our_test_record", {}) or {}
    if not isinstance(rec, dict):
        rec = {}
    entry["our_test_record"] = rec
    rec.setdefault("gate_run_ids", [])
    if gate_run_id and gate_run_id not in rec["gate_run_ids"]:
        rec["gate_run_ids"].append(gate_run_id)
    rec["verdict"] = gate_run_entry.get("verdict", rec.get("verdict"))
    rec["date"] = datetime.date.today().isoformat()
    updated_fields.append("our_test_record")

    # 3. Auto-promote UNTESTED → YELLOW only (never auto-DEPLOY)
    promoted = False
    if (
        entry.get("status_in_our_book") == "UNTESTED"
        and gate_run_entry.get("verdict") == "GREEN"
    ):
        entry["status_in_our_book"] = "YELLOW"
        entry["currently_unexplored_in_our_book"] = False
        updated_fields.append("status_in_our_book")
        promoted = True

    # 4. Persist + audit log
    if not dry_run:
        _save_yaml(yaml_path, entry)
        _append_update_log({
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "gate_run_id":           gate_run_id,
            "candidate_name":        candidate_name,
            "mechanism_id":          mechanism_id,
            "verdict":               gate_run_entry.get("verdict"),
            "updated_fields":        updated_fields,
            "promoted_to_candidate": promoted,
        })

    return {
        "mechanism_id":          mechanism_id,
        "orphan":                False,
        "updated_fields":        updated_fields,
        "promoted_to_candidate": promoted,
        "yaml_path":             str(yaml_path),
    }


# ── Retroactive sync (one-shot, for existing ledger) ────────────────────

def retro_sync_from_ledger() -> dict:
    """Walk gate_runs.jsonl in chronological order and call
    update_library_from_gate_run on each entry. Idempotent — gate_run_ids
    use set-style append so re-running doesn't duplicate."""
    from engine.research.pipeline import read_ledger
    all_runs = list(reversed(read_ledger(limit=100000)))  # oldest first
    summary = {"processed": 0, "matched": 0, "orphaned": 0, "promoted": 0}
    for run in all_runs:
        result = update_library_from_gate_run(run, all_gate_runs=all_runs)
        summary["processed"] += 1
        if result["orphan"]:
            summary["orphaned"] += 1
        else:
            summary["matched"] += 1
            if result["promoted_to_candidate"]:
                summary["promoted"] += 1
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--retro-sync", action="store_true",
                         help="Walk gate_runs.jsonl and update library YAMLs")
    parser.add_argument("--dry-run", action="store_true",
                         help="With --retro-sync: compute updates without writing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.retro_sync:
        s = retro_sync_from_ledger()
        print(f"retro_sync: processed={s['processed']} matched={s['matched']} "
              f"orphaned={s['orphaned']} promoted={s['promoted']}")
