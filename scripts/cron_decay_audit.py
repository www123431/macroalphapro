"""scripts/cron_decay_audit.py — daily decay-history audit cron.

Wraps `engine.research.decay_history_log.run_history_audit()` so the
`data/research/decay_history.jsonl` ledger stays fresh, which is what
`engine.research.data_freshness.check_decay_sentinel` probes for the
LivenessBanner. Without this cron the ledger went stale after 14 days
and surfaced a permanent false "data source DEAD" banner (fixed
2026-06-14).

Records start/end timestamps to data/agents/_health/decay_audit.jsonl
so the AgentHealth tile can show "last ran X ago, OK" alongside
DailyMemo / DirectionProposer.

Invocation: Windows Task Scheduler via scripts/install_agentic_cron.py.
Cadence: daily 06:25 SGT (5 min before the daily memo so the memo can
read fresh decay state if needed).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTH_PATH = REPO_ROOT / "data" / "agents" / "_health" / "decay_audit.jsonl"


def _record(status: str, *, elapsed_s: float, n_mechanisms: int | None = None,
            error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "decay_audit",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "n_mechanisms": n_mechanisms,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    try:
        # 1) Append fresh row to decay_history.jsonl (consecutive-month
        #    HARD_ALERT escalation needs daily samples).
        from engine.research.decay_history_log import run_history_audit
        results = run_history_audit()
        # 2) Refresh the decay_sentinel artifact (data/decay_sentinel/
        #    decay_sentinel_<today>.json) consumed by /api/decay/report.
        #    Without this, the /dashboard "Book Health · as of …" pill
        #    drifts stale even when the history-log writer is alive.
        try:
            from engine.agents.decay_sentinel.agent import run_daily
            artifact = run_daily(save=True)
            artifact_as_of = artifact.get("as_of")
        except Exception as exc:
            logger.warning("decay_sentinel artifact refresh failed: %s", exc)
            artifact_as_of = None
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed, n_mechanisms=len(results))
        print(f"[cron_decay_audit] ok — {len(results)} mechanisms in {elapsed:.1f}s "
              f"(artifact as_of={artifact_as_of})")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed, error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_decay_audit failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
