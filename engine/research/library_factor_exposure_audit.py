"""engine/research/library_factor_exposure_audit.py — validate that every
Mechanism Library YAML carries a factor_exposure block describing how its
alpha decomposes against the BARRA-equivalent factor model.

Per Phase 1 BARRA audit (commit f250a56 follow-on, 2026-05-30 evening): the
live carry sleeve revealed alpha_t collapses to 0.89 after MKT/SMB/MOM
control — 75% of reported return is factor-attributable. Every mechanism
must declare its factor exposure profile so reviewers can distinguish:
  - Genuine residual alpha (D_PEAD: t=3.93 survives)
  - Factor-tilted-by-design (smart beta, intentional)
  - Factor-explained (alpha collapse, weak diversification claim)

Schema (REQUIRED at the YAML top level):

  factor_exposure:
    audit_status: audited | pending     [REQUIRED]
    audit_priority: high | medium | low [REQUIRED if pending]
    # If audited, also REQUIRED:
    audit_date: YYYY-MM-DD
    audit_script: scripts/audits/audit_sleeve_factor_exposures.py
    audit_commit: <git-sha>
    phase: 1 | 2 | 3 | 4 | 5    # BARRA roadmap phase that produced these numbers
    n_months: int
    alpha_annualized: float     # alpha after factor control
    alpha_t_hac: float          # Newey-West HAC t-stat
    betas: {MKT, SMB, MOM, ...} # phase-1 minimum; +HML/QMJ phase-2; +sectors phase-3
    t_stats_hac: {alpha, MKT, SMB, MOM, ...}
    r_squared: float
    verdict: str (>= 50 chars)
    audit_blocks_deploy_decision: false   # SOFT gate per doctrine
    factor_tilted_by_design: bool         # smart-beta marker

Soft-gate doctrine (intentional): a sleeve with alpha_t < 2.0 after factor
control does NOT auto-reject. Many institutional strategies are factor-
tilted by design. The audit job is to SURFACE the truth (alpha-before vs
alpha-after) and let humans decide. Hard-block decisions on numbers
violates senior-quant practice.

CLI:
  python -m engine.research.library_factor_exposure_audit
  python -m engine.research.library_factor_exposure_audit --strict
  python -m engine.research.library_factor_exposure_audit --pending
  python -m engine.research.library_factor_exposure_audit --warn-weak-alpha
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

# Where build_factor_returns writes its phase caches (mirror of
# scripts/audits/audit_sleeve_factor_exposures.py constants).
PHASE_CACHE_PATHS = {
    1: REPO_ROOT / "data" / "cache" / "_barra_lite_factors.parquet",
    2: REPO_ROOT / "data" / "cache" / "_barra_lite_factors_phase2.parquet",
    3: REPO_ROOT / "data" / "cache" / "_barra_lite_factors_phase3.parquet",
    # Phase 4+ will register here when shipped.
}


def latest_available_phase() -> int:
    """Highest phase whose factor cache exists on disk. Used by the
    --upgrade-check flag to detect stale phase-N audited entries."""
    for p in sorted(PHASE_CACHE_PATHS.keys(), reverse=True):
        if PHASE_CACHE_PATHS[p].exists():
            return p
    return 0

# Fields required when audit_status == "audited"
# proposed_role added per [[feedback-loop-refinement-multi-role-candidates-2026-05-30]]
# (FLAW 3 fix from project_loop_design_flaws_discovered_2026-05-30).
AUDITED_REQUIRED = {
    "audit_date", "audit_script", "audit_commit", "phase",
    "n_months", "alpha_annualized", "alpha_t_hac",
    "betas", "t_stats_hac", "r_squared",
    "verdict", "audit_blocks_deploy_decision", "factor_tilted_by_design",
    "proposed_role",
}
# Fields required when audit_status == "pending"
PENDING_REQUIRED = {"audit_priority"}

# Phase-1 minimum factor set
PHASE_1_FACTORS = {"MKT", "SMB", "MOM"}
# Soft-warn threshold for alpha after factor control
SOFT_ALPHA_T_THRESHOLD = 2.0

# Valid candidate roles per multi-role doctrine. Each has different
# acceptance criteria — only alpha_seeker treats weak alpha as failure;
# the rest expect / are indifferent to alpha sign.
VALID_ROLES = {
    "alpha_seeker",            # D_PEAD — alpha t >= 2.0 required
    "risk_premium_harvester",  # carry / TSMOM — factor exposure expected
    "insurance",               # mom_hedge — negative drift OK, MOM β negative is the metric
    "regime_overlay",          # AN-1 / AM / AO — dynamic allocation, not static sleeve
    "diversifier",             # TLT/GLD — H9 cosine negative is the metric
}
# Roles where weak alpha (alpha_t < SOFT_ALPHA_T_THRESHOLD) is BY DESIGN
# and should not trigger weak_alpha_warn. alpha_seeker is the only role
# where weak alpha is a genuine concern.
ROLES_EXEMPT_FROM_WEAK_ALPHA_WARN = {
    "risk_premium_harvester",
    "insurance",
    "regime_overlay",
    "diversifier",
}


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
    """Validate one YAML's factor_exposure block."""
    fe = entry.get("factor_exposure")
    name = fp.stem
    if fe is None:
        return {
            "name":         name,
            "path":         str(fp),
            "status":       "MISSING_BLOCK",
            "audit_status": None,
            "missing":      ["factor_exposure:"],
            "pass":         False,
            "weak_alpha_warn": False,
        }
    audit_status = fe.get("audit_status")
    if audit_status not in {"audited", "pending"}:
        return {
            "name":         name,
            "path":         str(fp),
            "status":       "INVALID_AUDIT_STATUS",
            "audit_status": audit_status,
            "missing":      [f"audit_status must be 'audited' or 'pending', got {audit_status!r}"],
            "pass":         False,
            "weak_alpha_warn": False,
        }
    required = AUDITED_REQUIRED if audit_status == "audited" else PENDING_REQUIRED
    missing = [f for f in required if f not in fe]

    weak_alpha_warn = False
    proposed_role = fe.get("proposed_role")
    if audit_status == "audited":
        # Phase 1 minimum: must have all 3 factor betas
        betas = fe.get("betas") or {}
        for f in PHASE_1_FACTORS:
            if f not in betas:
                missing.append(f"betas.{f}")
        # Verdict length
        if "verdict" in fe and len(str(fe["verdict"]).strip()) < 50:
            missing.append("verdict (>= 50 chars)")
        # proposed_role must be a recognized role
        if proposed_role is not None and proposed_role not in VALID_ROLES:
            missing.append(
                f"proposed_role must be one of {sorted(VALID_ROLES)}, "
                f"got {proposed_role!r}"
            )
        # FLAW 2 fix: weak_alpha_warn ONLY for alpha_seeker role.
        # Insurance / diversifier / regime_overlay / risk_premium_harvester
        # all have weak-alpha by design and should not be flagged.
        alpha_t = fe.get("alpha_t_hac")
        tilted = fe.get("factor_tilted_by_design", False)
        role_exempts = proposed_role in ROLES_EXEMPT_FROM_WEAK_ALPHA_WARN
        if (isinstance(alpha_t, (int, float))
                and abs(alpha_t) < SOFT_ALPHA_T_THRESHOLD
                and not tilted
                and not role_exempts
                and entry.get("status_in_our_book") == "DEPLOYED"):
            weak_alpha_warn = True

    return {
        "name":         name,
        "path":         str(fp),
        "status":       "OK" if not missing else "INCOMPLETE",
        "audit_status": audit_status,
        "audit_priority": fe.get("audit_priority"),
        "phase":        fe.get("phase"),
        "alpha_t":      fe.get("alpha_t_hac"),
        "deployed":     entry.get("status_in_our_book") == "DEPLOYED",
        "tilted_by_design": fe.get("factor_tilted_by_design", False),
        "proposed_role": proposed_role,
        "missing":      missing,
        "pass":         len(missing) == 0,
        "weak_alpha_warn": weak_alpha_warn,
    }


def audit_library() -> dict:
    entries = _load_library_yamls()
    results = [_check_one(fp, d) for fp, d in entries]

    # Phase-upgrade detection: any audited entry whose phase < latest
    # available cache should be flagged for re-audit. This is the
    # mechanism that keeps the library current as we ship Phase 3, 4, 5.
    latest_phase = latest_available_phase()
    for r in results:
        r["stale_phase"] = False
        if r["audit_status"] == "audited" and r.get("phase") is not None:
            if int(r["phase"]) < latest_phase:
                r["stale_phase"] = True
                r["latest_available_phase"] = latest_phase

    n_audited = sum(1 for r in results if r["audit_status"] == "audited")
    n_pending = sum(1 for r in results if r["audit_status"] == "pending")
    n_missing = sum(1 for r in results if r["status"] == "MISSING_BLOCK")
    n_incomplete = sum(1 for r in results if r["status"] == "INCOMPLETE")
    n_pass = sum(1 for r in results if r["pass"])
    n_weak = sum(1 for r in results if r["weak_alpha_warn"])
    n_pending_high = sum(
        1 for r in results
        if r["audit_status"] == "pending" and r.get("audit_priority") == "high"
    )
    n_stale = sum(1 for r in results if r.get("stale_phase"))
    return {
        "total":                len(results),
        "audited":              n_audited,
        "pending":              n_pending,
        "pending_high_priority": n_pending_high,
        "missing_block":        n_missing,
        "incomplete":           n_incomplete,
        "pass":                 n_pass,
        "fail":                 len(results) - n_pass,
        "weak_alpha_warn":      n_weak,
        "stale_phase":          n_stale,
        "latest_available_phase": latest_phase,
        "results":              results,
    }


# -- CLI -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strict", action="store_true",
                          help="exit non-zero if any YAML missing/incomplete")
    parser.add_argument("--pending", action="store_true",
                          help="list pending entries with priority")
    parser.add_argument("--warn-weak-alpha", action="store_true",
                          help="list DEPLOYED sleeves with alpha_t_hac < 2.0 "
                                 "(not factor_tilted_by_design)")
    parser.add_argument("--upgrade-check", action="store_true",
                          help="list audited entries whose phase < latest "
                                 "available phase (i.e. should be re-audited)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summary = audit_library()
    print(
        f"[library_factor_exposure_audit] total={summary['total']} "
        f"audited={summary['audited']} pending={summary['pending']} "
        f"missing={summary['missing_block']} incomplete={summary['incomplete']} "
        f"weak_alpha={summary['weak_alpha_warn']} strict={args.strict}"
    )
    if summary["pending_high_priority"] > 0:
        print(
            f"  INFO: {summary['pending_high_priority']} pending HIGH-priority "
            f"(DEPLOYED mechanisms missing factor exposure audit)"
        )
    if summary["weak_alpha_warn"] > 0:
        print(
            f"  WARN: {summary['weak_alpha_warn']} DEPLOYED sleeves with "
            f"alpha_t_hac < {SOFT_ALPHA_T_THRESHOLD:.1f} (not flagged as "
            f"factor-tilted-by-design). SOFT gate per doctrine — not auto-rejected, "
            f"but should be reviewed."
        )
    if summary["stale_phase"] > 0:
        print(
            f"  UPGRADE: {summary['stale_phase']} audited entries on stale "
            f"phase (latest available: phase {summary['latest_available_phase']}). "
            f"Re-run scripts/audits/audit_sleeve_factor_exposures.py --phase "
            f"{summary['latest_available_phase']} and update YAMLs."
        )

    if args.pending:
        print("\nPending audits:")
        for r in summary["results"]:
            if r["audit_status"] == "pending":
                print(f"  - {r['name']:32s} priority={r['audit_priority']:<7s} "
                      f"deployed={r['deployed']}")

    if args.warn_weak_alpha:
        print("\nDEPLOYED sleeves with weak alpha after factor control:")
        for r in summary["results"]:
            if r["weak_alpha_warn"]:
                print(f"  - {r['name']:32s} alpha_t={r['alpha_t']:.2f} (phase {r['phase']})")
                print(f"      Likely either: (1) shared-macro driver explanation,")
                print(f"      (2) hidden beta, or (3) phase incomplete — wait for higher phase.")

    if args.upgrade_check:
        latest = summary["latest_available_phase"]
        print(f"\nAudited entries with stale phase (latest available: {latest}):")
        found = False
        for r in summary["results"]:
            if r.get("stale_phase"):
                found = True
                print(f"  - {r['name']:32s} on phase {r['phase']} -> "
                      f"upgrade to phase {latest}")
        if not found:
            print(f"  (none — all audited entries are on phase {latest})")

    if summary["fail"] > 0:
        print("\nFAILURES (missing/incomplete factor_exposure):")
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
