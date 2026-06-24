"""scripts/check_spec_registry_drift.py — SpecRegistry / on-disk drift guard.

Runs against the LIVE production DB (whatever DATABASE_URL points to,
defaulting to macro_alpha_memory.db). For every SpecRegistry row whose
``spec_path`` resolves to a real file, recomputes the git-blob SHA-1 of
the file and compares it to ``current_hash``.

Two failure modes this catches:

  1. Spec file edited without a corresponding amend_spec call.
     e.g. someone tweaks a sentence in docs/spec_*.md but skips the
     workflow step that bumps the DB row. The DB grows stale.

  2. Literal hash strings pinned in a spec file's own frontmatter.
     Writing a hash into the file changes the file's content and
     therefore invalidates the pin on the next byte — a fixed-point
     bug. The 2026-05-19 spec_risk_manager_agent_v1.md header cleanup
     was triggered by exactly this.

Both manifest as ``DB.current_hash != git-blob(file_bytes)``.

Exit codes:
  0   no drift detected
  1   drift detected (printed per-row with remediation hint)
  2   environment error (DB unreachable / table missing / etc.)

Intended usage:
  python scripts/check_spec_registry_drift.py
  # add to a pre-commit hook or CI workflow alongside pytest
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is importable when invoked as `python scripts/check_*.py`.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    try:
        from engine.preregistration import _compute_git_blob_hash, list_specs
    except Exception as exc:
        print(f"[drift-check] cannot import engine.preregistration: {exc}",
              file=sys.stderr)
        return 2

    try:
        rows = list_specs()
    except Exception as exc:
        print(f"[drift-check] cannot read SpecRegistry: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print("[drift-check] SpecRegistry is empty — nothing to check.",
              file=sys.stderr)
        return 2

    checked = 0
    drifted: list[tuple[int, str, str, str]] = []   # (id, path, db_hash, file_hash)
    skipped: list[tuple[int, str, str]] = []        # (id, path, reason)

    # Statuses whose drift IS load-bearing. Other statuses (superseded,
    # deprecated, code_locked_legacy) are historical breadcrumbs — the
    # row exists to record a past lock event, not to enforce the file
    # stays byte-for-byte stable. Drift on those is expected and silent.
    DRIFT_RELEVANT_STATUSES = {"active"}

    for r in rows:
        path = r.get("spec_path")
        status = r.get("status") or ""
        if status not in DRIFT_RELEVANT_STATUSES:
            skipped.append((int(r["id"]), path or "(none)",
                            f"status={status} (not drift-relevant)"))
            continue
        if not path:
            skipped.append((int(r["id"]), "(none)", "no spec_path"))
            continue
        abs_path = path if os.path.isabs(path) else str(REPO_ROOT / path)
        if not os.path.exists(abs_path):
            skipped.append((int(r["id"]), path, "file missing on disk"))
            continue
        db_hash = r.get("current_hash") or ""
        file_hash = _compute_git_blob_hash(abs_path)
        checked += 1
        if db_hash != file_hash:
            drifted.append((int(r["id"]), path, db_hash, file_hash))

    print(f"[drift-check] checked={checked} drifted={len(drifted)} "
          f"skipped={len(skipped)}")

    for sid, path, reason in skipped:
        print(f"  SKIP id={sid} path={path}  reason={reason}")

    if drifted:
        print()
        print("DRIFT DETECTED — the following specs need amend_spec or revert:")
        for sid, path, db_hash, file_hash in drifted:
            print(f"  id={sid}  {path}")
            print(f"    DB current_hash : {db_hash}")
            print(f"    file git-blob   : {file_hash}")
        print()
        print("Fix: call engine.preregistration.amend_spec("
              "path, kind, reason) after any spec edit so the DB "
              "row stays canonical. If the edit was unintentional, "
              "revert it. Do NOT paste a literal hash into the spec "
              "frontmatter — that creates a fixed-point bug.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
