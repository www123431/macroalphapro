# Decision — Gate → Library Feedback Loop (NEW1 from rigor audit)

**Date**: 2026-05-29 evening
**Type**: design contract (code wiring deferred to next session)
**Touches**: `engine/research/pipeline.py` (gate runner), `data/research/mechanism_library/*.yaml`
**Affects**: closes the symbiosis gap user identified earlier this session

## Why this exists

Issue NEW1 from the 44-issue rigor audit:

> No design exists for HOW the engine learns from gate verdicts back to
> library. Our gate verdicts ARE post-publication OOS data. Library was
> frozen — which means it's still a one-way explainer, not symbiosis.

The user's original goal — "agentic 和 quant 相辅相成" — requires this
loop. Otherwise we have a library that informs the generator but never
updates from our test results.

## The contract

After every gate run that completes (RED / GREEN / YELLOW verdict
written to `gate_runs.jsonl`):

1. Match candidate name to library mechanism (lookup table to be
   maintained in `data/research/candidate_to_mechanism_map.yaml`)
2. If a library YAML matches:
   - Append the new `gate_run_id` to `post_pub_decay.our_observed.gate_run_ids`
   - Recompute `summary_sharpe_observed` as mean across all our_observed runs
   - Recompute `delta_vs_published_lit` as our_sharpe vs the midpoint of
     `mclean_pontiff_2016.delta_range_estimate` (when present, else null)
   - Set `our_observed.last_updated` to today's date (YYYY-MM-DD)
   - Set `our_test_record.gate_run_ids` (append) and `verdict` (latest)
3. If status_in_our_book was UNTESTED and verdict is GREEN → flag
   `currently_unexplored_in_our_book: false` and `status_in_our_book: YELLOW`
   (DEPLOYED requires explicit user decision, not auto-promotion)
4. Write a single-line entry to `data/research/library_updates.jsonl`
   recording the (gate_run_id, mechanism_id, old_value, new_value) tuple
   for full audit trail
5. NEVER auto-flip `audit_signature` — that's still human-only

## What this is NOT

- NOT a generator update path. The library reflects evidence; whether the
  generator USES the updated entry is a separate decision (driven by
  `currently_unexplored_in_our_book` flag).
- NOT a way to auto-create library entries. Candidates not matching any
  library entry are LOGGED to `data/research/orphan_candidates.jsonl` for
  later human review. They do NOT silently create new library YAMLs.
- NOT a way to retroactively rewrite published-literature decay numbers.
  `mclean_pontiff_2016` and `post_2020_replications` are PUBLISHED data
  and stay locked. Our evidence goes in `our_observed` only.

## Implementation surface

Code to add in `engine/research/library_writer.py` (new file, ~150 lines):

```python
def update_library_from_gate_run(gate_run_entry: dict) -> dict:
    """Append our_observed evidence to the matching library mechanism.

    Returns: {mechanism_id: str | None, updated_fields: list[str],
              orphan: bool}
    """
    # 1. Match
    mechanism_id = _lookup_candidate_to_mechanism(gate_run_entry["name"])
    if mechanism_id is None:
        _log_orphan(gate_run_entry)
        return {"mechanism_id": None, "orphan": True, "updated_fields": []}

    # 2. Load YAML
    yaml_path = MECHANISM_LIBRARY_DIR / f"{mechanism_id}.yaml"
    entry = _load_yaml(yaml_path)

    # 3. Append our_observed
    observed = entry["post_pub_decay"]["our_observed"]
    observed["gate_run_ids"].append(gate_run_entry["gate_run_id"])
    observed["summary_sharpe_observed"] = _recompute_summary(
        observed["gate_run_ids"])
    observed["delta_vs_published_lit"] = _compute_delta_vs_lit(
        entry, observed["summary_sharpe_observed"])
    observed["last_updated"] = datetime.date.today().isoformat()

    # 4. Update our_test_record (latest verdict)
    rec = entry.setdefault("our_test_record", {})
    rec.setdefault("gate_run_ids", []).append(gate_run_entry["gate_run_id"])
    rec["verdict"] = gate_run_entry["verdict"]
    rec["date"] = gate_run_entry.get("date")

    # 5. Auto-flip currently_unexplored only on UNTESTED → GREEN
    promoted_to_candidate = False
    if (entry["status_in_our_book"] == "UNTESTED"
        and gate_run_entry["verdict"] == "GREEN"):
        entry["status_in_our_book"] = "YELLOW"  # not auto-DEPLOYED
        entry["currently_unexplored_in_our_book"] = False
        promoted_to_candidate = True

    # 6. Save + log
    _save_yaml(yaml_path, entry)
    _append_library_update_log({
        "ts": datetime.datetime.utcnow().isoformat(),
        "gate_run_id": gate_run_entry["gate_run_id"],
        "mechanism_id": mechanism_id,
        "promoted_to_candidate": promoted_to_candidate,
    })

    return {"mechanism_id": mechanism_id, "orphan": False,
            "updated_fields": ["our_observed", "our_test_record"] +
                              (["status_in_our_book"] if promoted_to_candidate else [])}
```

Wire point in `engine/research/pipeline.py` `run_gate()` (or its
post-processing step that writes to `gate_runs.jsonl`):

```python
# after gate_runs.jsonl append
from engine.research.library_writer import update_library_from_gate_run
update_library_from_gate_run(gate_run_entry)
```

The candidate→mechanism map (`candidate_to_mechanism_map.yaml`) is a
simple lookup:

```yaml
# data/research/candidate_to_mechanism_map.yaml
mappings:
  quality_novymarx_2013_v1: quality_qmj
  bond_xsmom_v1: bond_xsmom        # mechanism YAML to be added
  vix_carry_contango_filter_v1: vix_carry_vrp
  D_PEAD: post_earnings_drift
  # ... maintained as candidates run; orphan logger surfaces missing entries
```

## Status

- Schema field (`post_pub_decay.our_observed`) is in `_schema.md` v2 — DONE
- Map file & writer code: NOT YET BUILT (next session, after MUST-fix bundle commits)
- Pipeline wire-up: NOT YET BUILT (same next session)
- The 3 existing anchor YAMLs need the `our_observed` block added in their v2 migration

## Why deferring code to next session is acceptable

- The contract is fully specified above; implementation is mechanical
- Library scaling (Phase 0a) doesn't depend on the loop being live —
  fresh YAMLs come in with `our_observed: {gate_run_ids: [], ...}` empty
- When library_writer.py lands, it will retroactively populate from
  existing gate_runs.jsonl entries
- This avoids over-stuffing the current commit with cross-cutting changes

## Risks tracked

- Orphan candidates pile up (logged not silent) — review monthly
- Mapping drift (candidate renamed but map not updated) — covered by
  orphan log surfacing this
- Race condition if 2 gate runs land simultaneously — not at our scale
- YAML round-trip preserves comments/ordering — use ruamel.yaml not pyyaml
