"""
scripts/bootstrap_auto_audit_baseline.py — One-shot baseline initialisation
for the Auto-Audit Loop (R-1.B.2 setup, 2026-05-06).

What this does (idempotent — safe to run multiple times):
  1. Registers the 7 PRODUCTION_CODE_FILES via register_spec(retro=True)
     so rule_spec_hash_vs_code_drift has a hash baseline to compare against.
     retro=True chosen per supervisor 2026-05-06: this is audit infra, not
     a new scientific hypothesis, and shouldn't consume an EFFECTIVE_N_TRIALS
     trial budget slot.
  2. Triggers rule_universe_drift_vs_registered once, which self-initialises
     the universe baseline hash in SystemConfig if not already present.

When to run:
  • Once after R-1.B.2 deployment.
  • Idempotent re-runs are safe (register_spec is idempotent, universe init
    skips if baseline already exists).
  • Re-run if a new file gets added to PRODUCTION_CODE_FILES.

What this does NOT do:
  • Does not amend any existing spec_registry rows.
  • Does not register cron jobs (see docs/auto_audit_cron_setup.md).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    from engine.auto_audit_rules import (
        PRODUCTION_CODE_FILES,
        rule_universe_drift_vs_registered,
    )
    from engine.memory import init_db, SessionFactory, SpecRegistry
    from engine.preregistration import register_spec

    init_db()

    # ── Phase 1: register PRODUCTION_CODE_FILES ─────────────────────────────
    print(f"Phase 1 — registering {len(PRODUCTION_CODE_FILES)} production code files (retro=True):")
    n_registered = 0
    n_already = 0
    for path in PRODUCTION_CODE_FILES:
        with SessionFactory() as s:
            existing = s.query(SpecRegistry).filter_by(spec_path=path).first()
        if existing is None:
            spec_id = register_spec(path, retro=True)
            print(f"  [REG]  {path}  → spec_id={spec_id}")
            n_registered += 1
        else:
            print(f"  [SKIP] {path}  (already registered, spec_id={existing.id})")
            n_already += 1

    print(f"\n  Phase 1 result: {n_registered} newly registered, {n_already} already present.")

    # ── Phase 2: trigger universe baseline self-init ────────────────────────
    print(f"\nPhase 2 — universe baseline self-init:")
    with SessionFactory() as s:
        from engine.db_models import SystemConfig
        existing = s.query(SystemConfig).filter_by(
            key="auto_audit.universe_baseline_hash"
        ).first()

    if existing is None:
        result = rule_universe_drift_vs_registered()
        # Self-init returns None (no finding); baseline now stored.
        print("  [INIT] universe baseline written to SystemConfig.")
    else:
        print(f"  [SKIP] universe baseline already initialised (hash={existing.value[:12]})")

    print("\nBootstrap complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
