"""scripts/cron_daily_memo.py — daily 06:30 SGT cron entry for N1.

Purpose: turn the lazy-on-first-request StateOfBookTile into a TRULY
autonomous daily generation. Without this cron, the memo only writes
when the user opens /lab/today; that's not real autonomy. With this
cron, the memo is ALREADY THERE every morning when the user opens the
app — regardless of whether they opened the page yesterday.

Behavior:
  - Generates today's memo via engine.agents.daily_memo.generate(force=True)
  - Records start/end timestamps to data/agents/_health/daily_memo.jsonl
    so the AgentHealth tile can show "last ran X ago, OK".
  - Exits 0 on success, 1 on failure (so schtasks marks task as failed
    and you see it in Task Scheduler history).

Invocation: scheduled via Windows Task Scheduler (schtasks).
See scripts/install_agentic_cron.py for the registration entry.
"""
from __future__ import annotations

import json
import sys
import time
import datetime as _dt
from pathlib import Path


REPO_ROOT  = Path(__file__).resolve().parent.parent
HEALTH_DIR = REPO_ROOT / "data" / "agents" / "_health"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_run(agent_id: str, **fields) -> None:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    row = {"agent_id": agent_id, "ts": _utc_iso(), **fields}
    p = HEALTH_DIR / f"{agent_id}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    t0 = time.perf_counter()
    try:
        from engine.agents.daily_memo import generate
    except Exception as exc:
        _record_run("daily_memo", status="error", error=f"import_failed:{exc}",
                    elapsed_s=round(time.perf_counter() - t0, 2))
        print(f"[cron_daily_memo] import failed: {exc}", file=sys.stderr)
        return 1

    try:
        out = generate(force=True)
    except Exception as exc:
        _record_run("daily_memo", status="error", error=str(exc)[:300],
                    elapsed_s=round(time.perf_counter() - t0, 2))
        print(f"[cron_daily_memo] generate failed: {exc}", file=sys.stderr)
        return 1

    elapsed = round(time.perf_counter() - t0, 2)
    if out.get("error"):
        _record_run("daily_memo", status="error", error=str(out["error"])[:300],
                    elapsed_s=elapsed,
                    date_key=out.get("date_key"))
        print(f"[cron_daily_memo] generate returned error: {out['error']}",
              file=sys.stderr)
        return 1

    _record_run("daily_memo", status="ok",
                elapsed_s=elapsed,
                date_key=out.get("date_key"),
                n_citations=out.get("n_citations"),
                markdown_chars=len(out.get("markdown") or ""))
    print(f"[cron_daily_memo] ok · date={out.get('date_key')} · "
          f"{out.get('n_citations')} cites · {elapsed}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
