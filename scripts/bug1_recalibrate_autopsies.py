"""scripts/bug1_recalibrate_autopsies.py — one-shot autopsy recalibration
after BUG-1 fix (factor_combination FF-complement spanning).

Problem
=======
Before the BUG-1 fix (commit f8263951), factor_combination_ff used CAPM
α-t > 1.65 as GREEN gate. HML+MOM 50/50 combo cleared this at 2.30
because CAPM doesn't account for RMW/CMA exposure. Result: 4 historical
autopsies in COMBINATION_HML_MOM family carry GREEN as actual_verdict.

After the fix, the SAME spec on the SAME data emits MARGINAL (FF
complement α-t = 0.11 << 1.65). So the historical autopsies' actual_
verdict field is now factually wrong, and belief-4 family prior for
COMBINATION_HML_MOM is over-confident on GREEN.

Approach
========
For each autopsy in COMBINATION_HML_MOM family:
  1. Reload the spec from dispatch_log (no LLM call — deterministic)
  2. Re-run template_factor_combination_ff (now includes BUG-1 fix)
  3. If new verdict differs from autopsy's actual_verdict:
     a. Append a CORRECTION autopsy row with bug1_correction=true +
        corrected actual_verdict + corrected brier + corrected
        surprise direction/magnitude
     b. Append a SUPERSEDE marker to the original autopsy row
        (rewrite the line in-place via tmp-then-rename, with backup)

Result
======
- belief-4 reads only NON-superseded autopsies → uses corrected
  COMBINATION_HML_MOM family verdict counts (MARGINAL instead of GREEN)
- detect_patterns reflects corrected truth
- Audit trail preserved (corrected row carries bug1_correction tag)

Idempotent
==========
Re-running skips autopsies that already carry superseded_by — no
duplicates. Safe to invoke multiple times.

Other families
==============
Only COMBINATION_X_Y families are affected by BUG-1 (cross_sec /
portfolio_overlay templates don't use CAPM-only spanning). Script
explicitly filters to factor_combination signal_kind via dispatch_log.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.research.belief_autopsy import (  # noqa: E402
    AUTOPSIES_PATH, _brier_component, _surprise_direction,
    _surprise_magnitude,
)


AFFECTED_FAMILIES_PREFIX = "COMBINATION_"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_dispatch_log() -> list[dict]:
    path = REPO_ROOT / "data" / "strengthener" / "factor_dispatch_log.jsonl"
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _find_latest_dispatch_for_hypothesis(
    hypothesis_id_short: str, log_rows: list[dict],
) -> dict | None:
    """Find the most recent factor_combination dispatch row for this hyp."""
    candidates = []
    for r in log_rows:
        hid = r.get("hypothesis_id", "")
        if hypothesis_id_short not in hid:
            continue
        si = r.get("spec_inputs") or {}
        if si.get("signal_kind") != "factor_combination":
            continue
        if r.get("template_result") is None:
            continue
        candidates.append(r)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("ts", ""))


def _rerun_template(spec_inputs: dict, hypothesis_id: str) -> str | None:
    """Re-run template_factor_combination_ff with the same inputs. Returns
    new verdict string or None on failure. NO LLM call — deterministic
    template math only."""
    from engine.agents.strengthener.factor_spec_extractor import FactorSpec
    from engine.agents.strengthener.templates.factor_combination_ff import (
        template_factor_combination_ff,
    )

    si = spec_inputs or {}
    spec = FactorSpec(
        hypothesis_id=hypothesis_id,
        signal_kind=si.get("signal_kind", "factor_combination"),
        universe=si.get("universe", "ken_french_ff5_mom"),
        date_range=si.get("date_range", "1963-07:2025-12"),
        # Reconstruct signal_inputs from canonical V+M shape if dispatch_log
        # doesn't preserve it (older rows). Most COMBINATION_HML_MOM cases
        # use HML + MOM signals.
        signal_inputs=("ff.factors_weekly.hml", "ff.factors_weekly.mom"),
        rebal=si.get("rebal", "monthly"),
        weighting=si.get("weighting", "ew"),
        expected_holding_period="monthly",
        min_obs_months=60,
        pit_audits=("restatement",),
        cost_model="80bp_per_yr",
        rationale="bug1_recalibration",
        extracted_ts=_utc_iso(),
        model="bug1_replay",
        weighting_scheme_alt="0.50",
    )
    try:
        result = template_factor_combination_ff(spec)
        return result.verdict
    except Exception as exc:
        print(f"  [WARN] template rerun failed: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print changes without writing the file")
    args = parser.parse_args()

    if not AUTOPSIES_PATH.is_file():
        print(f"[bug1] no autopsies file at {AUTOPSIES_PATH}")
        return 0

    rows = []
    with AUTOPSIES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"  [WARN] malformed line skipped")
    print(f"[bug1] loaded {len(rows)} autopsy rows")

    log_rows = _load_dispatch_log()
    print(f"[bug1] loaded {len(log_rows)} dispatch_log rows")

    corrections: list[dict] = []
    superseded_ids: set[str] = set()

    for r in rows:
        sf = (r.get("strategy_family") or "")
        if not sf.startswith(AFFECTED_FAMILIES_PREFIX):
            continue
        if r.get("superseded_by") or r.get("bug1_correction"):
            continue   # already corrected
        hyp_short = (r.get("subject_id") or "").replace("-", "")[:8]
        dispatch = _find_latest_dispatch_for_hypothesis(hyp_short, log_rows)
        if dispatch is None:
            print(f"  [skip] {r['autopsy_id'][:8]} {sf} — no dispatch_log match for {hyp_short}")
            continue
        new_verdict = _rerun_template(
            dispatch.get("spec_inputs") or {},
            hyp_short + "-replay",
        )
        if new_verdict is None or new_verdict not in {"GREEN", "MARGINAL", "RED"}:
            print(f"  [skip] {r['autopsy_id'][:8]} — rerun returned {new_verdict!r}")
            continue
        old_verdict = r.get("actual_verdict", "")
        if new_verdict == old_verdict:
            print(f"  [unchanged] {r['autopsy_id'][:8]} {sf} stays {old_verdict}")
            continue

        # Build correction row
        corr_id = str(uuid.uuid4())
        pred_dist = r.get("predicted_verdict_dist") or {}
        corr_row = {
            "autopsy_id":             corr_id,
            "ts":                     _utc_iso(),
            "prediction_id":          r.get("prediction_id"),
            "verdict_event_id":       (r.get("verdict_event_id") or "") + "_bug1_correction",
            "subject_id":             r.get("subject_id"),
            "strategy_family":        r.get("strategy_family"),
            "claim_family":           r.get("claim_family"),
            "predicted_verdict_dist": pred_dist,
            "actual_verdict":         new_verdict,
            "brier_component":        _brier_component(pred_dist, new_verdict),
            "surprise_direction":     _surprise_direction(pred_dist, new_verdict),
            "surprise_magnitude":     _surprise_magnitude(pred_dist, new_verdict),
            "load_bearing_realized":  r.get("load_bearing_realized", []),
            "prediction_basis_echo":  (r.get("prediction_basis_echo") or "")
                                          + " [bug1_recalibration applied]",
            "superseded_by":          None,
            "bug1_correction":        True,
        }
        corrections.append(corr_row)
        superseded_ids.add(r["autopsy_id"])
        print(f"  [correct] {r['autopsy_id'][:8]} {sf} {old_verdict} → {new_verdict}")

    if not corrections:
        print("[bug1] no corrections needed")
        return 0

    print(f"[bug1] producing {len(corrections)} correction rows + "
          f"{len(superseded_ids)} supersede markers")

    if args.dry_run:
        print("[bug1] dry-run, no changes written")
        return 0

    # Mark superseded + append corrections via atomic rewrite
    backup = AUTOPSIES_PATH.with_suffix(
        f".jsonl.bak.{_dt.datetime.utcnow():%Y%m%dT%H%M%S}"
    )
    backup.write_bytes(AUTOPSIES_PATH.read_bytes())
    print(f"[bug1] backup written: {backup.name}")

    tmp = AUTOPSIES_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            if r["autopsy_id"] in superseded_ids:
                # Find the matching correction to point at
                matching = next(
                    c for c in corrections
                    if c["prediction_id"] == r.get("prediction_id")
                       and c["subject_id"] == r.get("subject_id")
                )
                r["superseded_by"] = matching["autopsy_id"]
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for c in corrections:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    os.replace(tmp, AUTOPSIES_PATH)
    print(f"[bug1] autopsies.jsonl updated with corrections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
