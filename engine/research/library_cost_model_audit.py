"""engine/research/library_cost_model_audit.py — validate that every
Mechanism Library YAML carries a cost_model block, and report which
mechanisms still need a per-mechanism Almgren-Chriss audit.

Per [[feedback-cost-model-rigor-almgren-not-scalar-2026-05-30]]:
flat scalar-bps cost is wrong in both directions (over-conservative at
low AUM, under-conservative at high AUM / stress). Every mechanism
must be audited via the Almgren-Chriss model + capacity report before
its reported Sharpe is quoted in any deploy decision.

Schema (REQUIRED at the YAML top level):

  cost_model:
    audit_status: audited | pending     [REQUIRED]
    audit_priority: high | medium | low [REQUIRED if pending]
    # If audited, also REQUIRED:
    audit_date: YYYY-MM-DD
    audit_script: scripts/...
    audit_commit: <git-sha>
    type: almgren_chriss | scalar_bps_buffer
    half_spread_bps: float
    impact_coef: float
    daily_sigma_estimate: float
    universe_median_adv_usd: int
    monthly_turnover_estimate: float
    stress_multiplier: float
    rationale: str (>= 50 chars)
    multi_aum_sharpe_sleeve: {at_10M, at_100M, at_1B}
    capacity: {hard_capacity_usd, binding_constraint, safe_deploy_band_usd}
    # Also REQUIRED for both:
    current_default_in_gate.bps_per_side: float (what live gate uses pre-audit)

Three operating modes:
  --strict   exit non-zero if ANY YAML missing cost_model block (CI use)
  --pending  list YAMLs still in pending status (developer use)
  default    summarize per-YAML status + return 0

CLI:
  python -m engine.research.library_cost_model_audit
  python -m engine.research.library_cost_model_audit --strict
  python -m engine.research.library_cost_model_audit --pending
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

# Fields required when audit_status == "audited"
AUDITED_REQUIRED = {
    "audit_date", "audit_script", "audit_commit",
    "type", "half_spread_bps", "impact_coef",
    "daily_sigma_estimate", "universe_median_adv_usd",
    "monthly_turnover_estimate", "stress_multiplier",
    "rationale", "multi_aum_sharpe_sleeve", "capacity",
}
# Fields required when audit_status == "pending"
PENDING_REQUIRED = {"audit_priority", "current_default_in_gate"}


def _load_library_yamls() -> list[tuple[Path, dict]]:
    out = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = yaml.safe_load(fp.read_text(encoding="utf-8"))
            if d:
                out.append((fp, d))
        except Exception as e:
            logger.warning("parse failed %s: %s", fp, e)
    return out


def _check_one(fp: Path, entry: dict) -> dict:
    """Validate one YAML's cost_model block. Returns result dict."""
    cm = entry.get("cost_model")
    name = fp.stem
    if cm is None:
        return {
            "name":       name,
            "path":       str(fp),
            "status":     "MISSING_BLOCK",
            "audit_status": None,
            "missing":    ["cost_model:"],
            "pass":       False,
        }
    audit_status = cm.get("audit_status")
    if audit_status not in {"audited", "pending"}:
        return {
            "name":         name,
            "path":         str(fp),
            "status":       "INVALID_AUDIT_STATUS",
            "audit_status": audit_status,
            "missing":      [f"audit_status must be 'audited' or 'pending', got {audit_status!r}"],
            "pass":         False,
        }
    required = AUDITED_REQUIRED if audit_status == "audited" else PENDING_REQUIRED
    missing = [f for f in required if f not in cm]
    # Audited entries must have multi_aum_sharpe_sleeve as a dict with at_100M
    if audit_status == "audited" and "multi_aum_sharpe_sleeve" in cm:
        s = cm["multi_aum_sharpe_sleeve"]
        if not isinstance(s, dict) or "at_100M" not in s:
            missing.append("multi_aum_sharpe_sleeve.at_100M")
    # Audited entries must have rationale length >= 50
    if audit_status == "audited" and "rationale" in cm:
        if len(str(cm["rationale"]).strip()) < 50:
            missing.append("rationale (must be >= 50 chars)")
    return {
        "name":         name,
        "path":         str(fp),
        "status":       "OK" if not missing else "INCOMPLETE",
        "audit_status": audit_status,
        "audit_priority": cm.get("audit_priority"),
        "blocks_deploy": cm.get("audit_blocks_deploy_decision"),
        "missing":      missing,
        "pass":         len(missing) == 0,
    }


def audit_library() -> dict:
    entries = _load_library_yamls()
    results = [_check_one(fp, d) for fp, d in entries]
    n_audited = sum(1 for r in results if r["audit_status"] == "audited")
    n_pending = sum(1 for r in results if r["audit_status"] == "pending")
    n_missing = sum(1 for r in results if r["status"] == "MISSING_BLOCK")
    n_incomplete = sum(1 for r in results if r["status"] == "INCOMPLETE")
    n_pass = sum(1 for r in results if r["pass"])
    n_pending_high = sum(
        1 for r in results
        if r["audit_status"] == "pending" and r.get("audit_priority") == "high"
    )
    return {
        "total":                len(results),
        "audited":              n_audited,
        "pending":              n_pending,
        "pending_high_priority": n_pending_high,
        "missing_block":        n_missing,
        "incomplete":           n_incomplete,
        "pass":                 n_pass,
        "fail":                 len(results) - n_pass,
        "results":              results,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero if any YAML is missing or has incomplete cost_model",
    )
    parser.add_argument(
        "--pending", action="store_true",
        help="list only YAMLs in pending status",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = audit_library()
    print(
        f"[library_cost_model_audit] total={summary['total']} "
        f"audited={summary['audited']} pending={summary['pending']} "
        f"missing={summary['missing_block']} incomplete={summary['incomplete']} "
        f"strict={args.strict}"
    )
    if summary["pending_high_priority"] > 0:
        print(
            f"  WARN: {summary['pending_high_priority']} pending HIGH-priority "
            f"(DEPLOYED mechanisms that need audit before live deploy decisions)"
        )

    if args.pending:
        print("\nPending audits:")
        for r in summary["results"]:
            if r["audit_status"] == "pending":
                print(f"  - {r['name']:32s} priority={r['audit_priority']:<7s}"
                      f" blocks_deploy={r['blocks_deploy']}")

    if summary["fail"] > 0:
        print("\nFAILURES (missing/incomplete cost_model):")
        for r in summary["results"]:
            if not r["pass"]:
                print(f"  - {r['name']:32s} status={r['status']}")
                for m in r["missing"]:
                    print(f"      missing: {m}")

    if args.strict and summary["fail"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
