"""engine/research/library_ca_filter_audit.py — Phase 5.7 SG5:
validate every DEPLOYED / PENDING_DEPLOY library YAML declares
cost_model.ca_filter_k + companion fields.

Per [[project-paper-borrow-ml-btc-costs-2026-06-01]] §5.7:
the BTC paper showed that without a Cost-Aware Execution Filter at
the signal-to-trade boundary, high-turnover strategies bleed -64% to
-98% ARC under realistic costs. Every sleeve that ships MUST declare
its ca_filter_k + signal_type + tcost_round_trip.

Schema (REQUIRED under cost_model for status DEPLOYED / PENDING_DEPLOY):

  cost_model:
    ca_filter_k:               float > 0      [REQUIRED]
    ca_filter_k_method:        one of {paper_default,
                                       pbb_sweep_calibrated,
                                       scalar_override}  [REQUIRED]
    ca_filter_k_audit_date:    YYYY-MM-DD                 [REQUIRED]
    ca_filter_k_audit_note:    str (>= 20 chars)           [REQUIRED]
    ca_signal_type:            one of SignalType enum     [REQUIRED]
    tcost_round_trip_bps:      int > 0                    [REQUIRED]

Three operating modes:
  --strict   exit non-zero if any DEPLOYED/PENDING_DEPLOY YAML missing
             the CA fields (CI use)
  default    summarize status, exit 0

CLI:
  python -m engine.research.library_ca_filter_audit
  python -m engine.research.library_ca_filter_audit --strict
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"

REQUIRED_FIELDS = {
    "ca_filter_k", "ca_filter_k_method", "ca_filter_k_audit_date",
    "ca_filter_k_audit_note", "ca_signal_type", "tcost_round_trip_bps",
}
VALID_METHODS = {
    "paper_default", "pbb_sweep_calibrated", "scalar_override",
    "not_applicable",
}
# When method=not_applicable, ca_filter_k_not_applicable_reason is REQUIRED
# (the WHY explanation) but ca_filter_k itself can be null/omitted
NOT_APPLICABLE_REQUIRED_FIELDS = {
    "ca_filter_k_method", "ca_filter_k_audit_date",
    "ca_filter_k_not_applicable_reason", "ca_signal_type",
    "tcost_round_trip_bps",
}
VALID_SIGNAL_TYPES = {
    "point_forecast", "cross_sect_rank", "regime_indicator",
    "vol_norm_zscore", "binary_trigger",
}
ENFORCED_STATUS = {"DEPLOYED", "PENDING_DEPLOY"}


def audit_yaml(path: Path) -> dict:
    """Audit one yaml. Returns:
      {status_in_book, ca_present: bool, violations: [str]}.
    """
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return {
            "status_in_book": None,
            "ca_present": False,
            "violations": [f"yaml parse failed: {exc}"],
            "enforced": False,
        }
    status = doc.get("status_in_our_book") or doc.get("status")
    enforced = status in ENFORCED_STATUS
    cm = doc.get("cost_model") or {}
    if not isinstance(cm, dict):
        return {
            "status_in_book": status, "ca_present": False,
            "violations": ["cost_model is not a dict"],
            "enforced": enforced,
        }
    method = cm.get("ca_filter_k_method")
    # NA path: a separate schema (k is null, _reason required)
    if method == "not_applicable":
        present_fields = NOT_APPLICABLE_REQUIRED_FIELDS & set(cm.keys())
        ca_present = bool(present_fields)
        violations: list[str] = []
        if enforced:
            missing = NOT_APPLICABLE_REQUIRED_FIELDS - present_fields
            if missing:
                violations.append(
                    f"missing NA fields: {sorted(missing)}"
                )
            reason = cm.get("ca_filter_k_not_applicable_reason", "")
            if reason and len(str(reason).strip()) < 40:
                violations.append(
                    "ca_filter_k_not_applicable_reason < 40 chars "
                    "(must give a sleeve-specific explanation, not "
                    "boilerplate)"
                )
            sig_type = cm.get("ca_signal_type")
            if sig_type and sig_type not in VALID_SIGNAL_TYPES:
                violations.append(
                    f"ca_signal_type {sig_type!r} not in "
                    f"{sorted(VALID_SIGNAL_TYPES)}"
                )
            tcost = cm.get("tcost_round_trip_bps")
            if tcost is not None and (not isinstance(tcost, (int, float))
                                        or tcost <= 0):
                violations.append(
                    f"tcost_round_trip_bps {tcost!r} must be positive"
                )
        return {
            "status_in_book": status,
            "ca_present": ca_present,
            "violations": violations,
            "enforced": enforced,
            "na": True,
        }
    # Standard path (active CA filter)
    present_fields = REQUIRED_FIELDS & set(cm.keys())
    ca_present = bool(present_fields)
    violations: list[str] = []
    if enforced:
        missing = REQUIRED_FIELDS - present_fields
        if missing:
            violations.append(
                f"missing CA fields: {sorted(missing)}"
            )
        if method and method not in VALID_METHODS:
            violations.append(
                f"ca_filter_k_method {method!r} not in {sorted(VALID_METHODS)}"
            )
        sig_type = cm.get("ca_signal_type")
        if sig_type and sig_type not in VALID_SIGNAL_TYPES:
            violations.append(
                f"ca_signal_type {sig_type!r} not in {sorted(VALID_SIGNAL_TYPES)}"
            )
        k = cm.get("ca_filter_k")
        if k is not None and (not isinstance(k, (int, float)) or k <= 0):
            violations.append(f"ca_filter_k {k!r} must be a positive number")
        tcost = cm.get("tcost_round_trip_bps")
        if tcost is not None and (not isinstance(tcost, (int, float))
                                    or tcost <= 0):
            violations.append(
                f"tcost_round_trip_bps {tcost!r} must be positive int"
            )
        note = cm.get("ca_filter_k_audit_note", "")
        if note and len(str(note).strip()) < 20:
            violations.append(
                "ca_filter_k_audit_note < 20 chars (must explain calibration "
                "method + future plan)"
            )
    return {
        "status_in_book": status,
        "ca_present": ca_present,
        "violations": violations,
        "enforced": enforced,
        "na": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                         help="exit non-zero if any DEPLOYED/PENDING_DEPLOY "
                              "yaml is missing or invalid")
    args = parser.parse_args()

    if not LIBRARY_DIR.is_dir():
        print(f"library dir missing: {LIBRARY_DIR}")
        return 0

    rows: list[tuple[str, dict]] = []
    for yp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if yp.name.startswith("_"):
            continue
        rows.append((yp.stem, audit_yaml(yp)))

    enforced_count = sum(1 for _, r in rows if r["enforced"])
    ca_present = sum(1 for _, r in rows if r["enforced"] and r["ca_present"]
                                            and not r["violations"])
    pending = sum(1 for _, r in rows if r["enforced"] and not r["ca_present"])
    bad = [(name, r) for name, r in rows
            if r["enforced"] and r["violations"]]

    print(f"[library_ca_filter_audit] total={len(rows)} enforced={enforced_count} "
          f"ca_present={ca_present} pending={pending} bad={len(bad)} "
          f"strict={args.strict}")

    if bad:
        print("Violations:")
        for name, r in bad:
            print(f"  {name} ({r['status_in_book']}): {'; '.join(r['violations'])}")

    if args.strict:
        if pending > 0:
            print(f"STRICT: {pending} enforced YAML(s) missing CA block")
        if bad:
            print(f"STRICT: {len(bad)} enforced YAML(s) have invalid CA fields")
        return 1 if (pending > 0 or bad) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
