"""scripts/cron_workflow_executor.py — fire workflow_executor.run_all_due().

Daily cron entry. Iterates registered workflows, asks each is_due(),
runs the due ones (subject to all 10 rules: dry-run by default unless
explicitly in AUTORUN_WHITELIST).

Writes one health row per run. The user sees the result on /ops in
the WorkflowExecutorPanel.
"""
from __future__ import annotations

import json
import sys
import time
import datetime as _dt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    t0 = time.perf_counter()
    try:
        from engine.agents.workflow_executor import run_all_due
    except Exception as exc:
        print(f"[cron_workflow_executor] import failed: {exc}", file=sys.stderr)
        return 1

    try:
        results = run_all_due(trigger="cron")
    except Exception as exc:
        print(f"[cron_workflow_executor] run_all_due failed: {exc}", file=sys.stderr)
        return 1

    elapsed = round(time.perf_counter() - t0, 2)
    n_ok = sum(1 for r in results if r.status == "ok")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_err = sum(1 for r in results if r.status == "error")
    print(f"[cron_workflow_executor] ok · ran {len(results)} workflows · "
          f"{n_ok} ok · {n_skipped} skipped · {n_err} err · {elapsed}s")
    for r in results:
        flag = "[dry]" if r.dry_run else ""
        print(f"  - {r.workflow_id:35s}  {r.status:14s}  {flag}  {r.reason[:50]}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
