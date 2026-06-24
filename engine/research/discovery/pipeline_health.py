"""engine/research/discovery/pipeline_health.py — SLO monitoring for
the research discovery pipeline.

Senior follow-up A per [[feedback-senior-review-as-we-build-2026-05-30]]:
when cron silently dies, no one notices for days. Add explicit health
checks that fail LOUD: the daily_summary top badge shows OK / DEGRADED
/ DOWN, the API exposes /api/research/discovery/health for the frontend
status bar, and alerts can fire from this single source of truth.

Health checks:
  - discovery_freshness: last discovery_runs.jsonl entry within
    DISCOVERY_SLO_HOURS (default 25, allows 1h slack on daily cron)
  - gate_freshness:      last gate_runs.jsonl within GATE_SLO_DAYS
    (default 7; gate runs are weekly-ish at typical cadence)
  - queue_drain:         queue + borderline counts not blowing up
    above QUEUE_BLOAT_THRESHOLD (default 100)
  - llm_budget:          weekly LLM cost not exceeding LLM_BUDGET_WEEKLY
    (default $50)

Each check returns OK / WARN / ALERT individually; overall status is
the worst of all checks.

NEVER blocks. Pure observability. If pipeline_health itself crashes,
caller reports DEGRADED with reason.
"""
from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Discovery side
DISCOVERY_RUNS = REPO_ROOT / "data" / "research" / "discovery_runs.jsonl"
DISCOVERY_QUEUE = REPO_ROOT / "data" / "research" / "discovery_queue.jsonl"
DISCOVERY_BORDERLINE = REPO_ROOT / "data" / "research" / "discovery_borderline.jsonl"
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
LLM_COST = REPO_ROOT / "data" / "llm_cost_ledger.jsonl"
# Book side — per-senior integration with existing systems
PAPER_TRADE_DIR = REPO_ROOT / "data" / "paper_trade"
DECAY_REPORT = REPO_ROOT / "data" / "decay" / "decay_report.json"
OPS_WIDGET_STATE = REPO_ROOT / "data" / "ops_watchdog" / "widget_state.json"
AGENT_SLO = REPO_ROOT / "data" / "agent_slo_metrics.jsonl"

# SLO thresholds — tunable
DISCOVERY_SLO_HOURS = 25      # daily cron + 1h slack
GATE_SLO_DAYS = 7              # weekly is reasonable for our cadence
QUEUE_BLOAT_THRESHOLD = 100   # if queue > this, alarm (we're not reviewing)
LLM_BUDGET_WEEKLY = 50.0       # USD
PAPER_TRADE_SLO_HOURS = 30    # daily run + slack
DECAY_REPORT_SLO_HOURS = 30   # decay sentinel runs daily


class HealthLevel(str, enum.Enum):
    OK = "OK"
    WARN = "WARN"
    ALERT = "ALERT"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass
class CheckResult:
    name:           str
    status:         HealthLevel
    detail:         str
    last_event_ts:  str | None = None
    expected:       str | None = None
    # Senior 2026-05-30 (borrow from PIT audit display): when FLAG/ALERT,
    # tell the operator what specifically to do. The detail says WHAT
    # went wrong; remedy says HOW TO FIX. Empty for OK checks.
    remedy:         str | None = None

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "status":        self.status.value,
            "detail":        self.detail,
            "last_event_ts": self.last_event_ts,
            "expected":      self.expected,
            "remedy":        self.remedy,
        }


def _read_jsonl_tail(path: Path, n: int = 200) -> list[dict]:
    """Last N parseable lines."""
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _last_ts_in_log(path: Path, *, ts_keys: tuple[str, ...] = ("ts", "timestamp_utc")) -> str | None:
    """Find most recent ts field in the JSONL (last N entries only)."""
    recent = _read_jsonl_tail(path, n=200)
    latest = None
    for rec in recent:
        for k in ts_keys:
            v = rec.get(k)
            if isinstance(v, str) and v:
                if latest is None or v > latest:
                    latest = v
                break
    return latest


def _hours_since(ts_str: str | None) -> float | None:
    """Hours elapsed since the ISO timestamp; None if invalid."""
    if not ts_str:
        return None
    try:
        ts = datetime.datetime.fromisoformat(ts_str.rstrip("Z"))
    except ValueError:
        return None
    delta = datetime.datetime.utcnow() - ts
    return delta.total_seconds() / 3600.0


# ── Individual checks ─────────────────────────────────────────────────────

def check_discovery_freshness() -> CheckResult:
    """Did discovery_runs see a successful run in last DISCOVERY_SLO_HOURS?"""
    latest = _last_ts_in_log(DISCOVERY_RUNS)
    hours = _hours_since(latest)
    expected = f"< {DISCOVERY_SLO_HOURS}h"
    if hours is None:
        return CheckResult(
            name="discovery_freshness",
            status=HealthLevel.ALERT,
            detail="no discovery runs found in log",
            last_event_ts=None, expected=expected,
            remedy=("First-time setup or log was wiped. Bootstrap with "
                    "python scripts/run_paper_discovery.py --new-flow"),
        )
    if hours <= DISCOVERY_SLO_HOURS:
        return CheckResult(
            name="discovery_freshness",
            status=HealthLevel.OK,
            detail=f"last run {hours:.1f}h ago",
            last_event_ts=latest, expected=expected,
        )
    if hours <= DISCOVERY_SLO_HOURS * 2:
        return CheckResult(
            name="discovery_freshness",
            status=HealthLevel.WARN,
            detail=f"last run {hours:.1f}h ago (SLO {DISCOVERY_SLO_HOURS}h)",
            last_event_ts=latest, expected=expected,
        )
    return CheckResult(
        name="discovery_freshness",
        status=HealthLevel.ALERT,
        detail=f"last run {hours:.1f}h ago — cron may be dead",
        last_event_ts=latest, expected=expected,
        remedy=("Windows: schtasks /Query /TN MacroAlphaPro\\research-discover-newflow "
                "to inspect. Run manually: python scripts/run_paper_discovery.py --new-flow"),
    )


def check_gate_freshness() -> CheckResult:
    """Did gate_runs see activity in last GATE_SLO_DAYS?"""
    latest = _last_ts_in_log(GATE_RUNS)
    hours = _hours_since(latest)
    expected = f"< {GATE_SLO_DAYS}d"
    if hours is None:
        return CheckResult(
            name="gate_freshness",
            status=HealthLevel.ALERT,
            detail="no gate runs found in log",
            last_event_ts=None, expected=expected,
            remedy=("First-time setup or log was wiped. Promote a paper from "
                    "the review queue to trigger first auto-gate."),
        )
    days = hours / 24.0
    if days <= GATE_SLO_DAYS:
        return CheckResult(
            name="gate_freshness",
            status=HealthLevel.OK,
            detail=f"last run {days:.1f}d ago",
            last_event_ts=latest, expected=expected,
        )
    if days <= GATE_SLO_DAYS * 2:
        return CheckResult(
            name="gate_freshness",
            status=HealthLevel.WARN,
            detail=f"last run {days:.1f}d ago (SLO {GATE_SLO_DAYS}d)",
            last_event_ts=latest, expected=expected,
        )
    return CheckResult(
        name="gate_freshness",
        status=HealthLevel.ALERT,
        detail=f"last run {days:.1f}d ago — no factor testing happening",
        last_event_ts=latest, expected=expected,
        remedy=("Check review_queue: if empty, no candidates to test (expected). "
                "If queue has entries, click Promote on UI to trigger auto-gate, "
                "or run a manual strict-gate script."),
    )


def check_queue_drain() -> CheckResult:
    """If queues bloat (reviewer not keeping up), warn."""
    review = _read_jsonl_tail(DISCOVERY_QUEUE, n=10_000)
    borderline = _read_jsonl_tail(DISCOVERY_BORDERLINE, n=10_000)
    total = len(review) + len(borderline)
    expected = f"< {QUEUE_BLOAT_THRESHOLD}"
    if total <= QUEUE_BLOAT_THRESHOLD:
        return CheckResult(
            name="queue_drain", status=HealthLevel.OK,
            detail=(f"review={len(review)} + borderline={len(borderline)} "
                      f"= {total}"),
            expected=expected,
        )
    if total <= QUEUE_BLOAT_THRESHOLD * 2:
        return CheckResult(
            name="queue_drain", status=HealthLevel.WARN,
            detail=(f"{total} entries — reviewer may be falling behind"),
            expected=expected,
        )
    return CheckResult(
        name="queue_drain", status=HealthLevel.ALERT,
        detail=(f"{total} entries — drainage urgent"),
        expected=expected,
        remedy=("Open the Research page review queue and Promote/Skip each entry. "
                "Borderline entries can be bulk-skipped if they're clearly non-factor."),
    )


def check_llm_budget() -> CheckResult:
    """Weekly LLM spend not exceeding budget."""
    recent = _read_jsonl_tail(LLM_COST, n=10_000)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    week_sum = 0.0
    for rec in recent:
        ts_str = rec.get("ts") or rec.get("timestamp_utc") or ""
        try:
            ts = datetime.datetime.fromisoformat(ts_str.rstrip("Z"))
        except ValueError:
            continue
        if ts < cutoff:
            continue
        week_sum += float(rec.get("cost_usd", 0) or 0)
    expected = f"< ${LLM_BUDGET_WEEKLY:.2f}"
    if week_sum <= LLM_BUDGET_WEEKLY:
        return CheckResult(
            name="llm_budget", status=HealthLevel.OK,
            detail=f"week spend ${week_sum:.4f}",
            expected=expected,
        )
    if week_sum <= LLM_BUDGET_WEEKLY * 1.5:
        return CheckResult(
            name="llm_budget", status=HealthLevel.WARN,
            detail=f"week spend ${week_sum:.4f} above budget",
            expected=expected,
        )
    return CheckResult(
        name="llm_budget", status=HealthLevel.ALERT,
        detail=f"week spend ${week_sum:.4f} — runaway?",
        expected=expected,
        remedy=("Check llm_cost_ledger.jsonl tail — look for caller spike. "
                "Tighten max_per_source on cadence runner or disable --use-llm-rescue."),
    )


# ── Aggregated report ─────────────────────────────────────────────────────

def check_paper_trade_freshness() -> CheckResult:
    """Did paper_trade run today? Check most recent log file mtime."""
    if not PAPER_TRADE_DIR.exists():
        return CheckResult(
            name="paper_trade_freshness",
            status=HealthLevel.UNKNOWN,
            detail="paper_trade dir missing",
            expected=f"< {PAPER_TRADE_SLO_HOURS}h",
        )
    # Use latest mtime among daily_run_*.log files
    logs = list(PAPER_TRADE_DIR.glob("daily_run_*.log"))
    if not logs:
        return CheckResult(
            name="paper_trade_freshness",
            status=HealthLevel.ALERT,
            detail="no daily_run_*.log files found",
            expected=f"< {PAPER_TRADE_SLO_HOURS}h",
            remedy=("Paper-trade has never run. Bootstrap: "
                    "python scripts/run_paper_trade_daily.py"),
        )
    latest_mtime = max(p.stat().st_mtime for p in logs)
    latest_ts = datetime.datetime.fromtimestamp(latest_mtime).isoformat() + "Z"
    hours = (datetime.datetime.utcnow().timestamp() - latest_mtime) / 3600.0
    expected = f"< {PAPER_TRADE_SLO_HOURS}h"
    if hours <= PAPER_TRADE_SLO_HOURS:
        return CheckResult(
            name="paper_trade_freshness", status=HealthLevel.OK,
            detail=f"last run {hours:.1f}h ago",
            last_event_ts=latest_ts, expected=expected,
        )
    if hours <= PAPER_TRADE_SLO_HOURS * 2:
        return CheckResult(
            name="paper_trade_freshness", status=HealthLevel.WARN,
            detail=f"last run {hours:.1f}h ago",
            last_event_ts=latest_ts, expected=expected,
        )
    return CheckResult(
        name="paper_trade_freshness", status=HealthLevel.ALERT,
        detail=f"last run {hours:.1f}h ago — paper-trade cron may be dead",
        last_event_ts=latest_ts, expected=expected,
        remedy=("Check Windows Task Scheduler for paper_trade_daily. "
                "Run manually: python scripts/run_paper_trade_daily.py"),
    )


def check_decay_sentinel() -> CheckResult:
    """Did decay sentinel produce a recent report?"""
    if not DECAY_REPORT.exists():
        return CheckResult(
            name="decay_sentinel",
            status=HealthLevel.UNKNOWN,
            detail="decay_report.json missing — sentinel not running?",
            expected=f"< {DECAY_REPORT_SLO_HOURS}h",
        )
    mtime = DECAY_REPORT.stat().st_mtime
    latest_ts = datetime.datetime.fromtimestamp(mtime).isoformat() + "Z"
    hours = (datetime.datetime.utcnow().timestamp() - mtime) / 3600.0
    expected = f"< {DECAY_REPORT_SLO_HOURS}h"
    if hours <= DECAY_REPORT_SLO_HOURS:
        return CheckResult(
            name="decay_sentinel", status=HealthLevel.OK,
            detail=f"report {hours:.1f}h old",
            last_event_ts=latest_ts, expected=expected,
        )
    if hours <= DECAY_REPORT_SLO_HOURS * 2:
        return CheckResult(
            name="decay_sentinel", status=HealthLevel.WARN,
            detail=f"report {hours:.1f}h old",
            last_event_ts=latest_ts, expected=expected,
        )
    return CheckResult(
        name="decay_sentinel", status=HealthLevel.ALERT,
        detail=f"report {hours:.1f}h old — book decay not being monitored",
        last_event_ts=latest_ts, expected=expected,
        remedy=("Run decay sentinel manually to refresh: "
                "see engine.persona.decay_sentinel for the runner."),
    )


def check_ops_watchdog() -> CheckResult:
    """Latest ops_watchdog widget state."""
    if not OPS_WIDGET_STATE.exists():
        return CheckResult(
            name="ops_watchdog",
            status=HealthLevel.UNKNOWN,
            detail="widget_state.json missing",
        )
    try:
        data = json.loads(OPS_WIDGET_STATE.read_text(encoding="utf-8"))
        severity = (data.get("severity") or data.get("status") or "").upper()
        ts = data.get("as_of") or data.get("timestamp") or ""
        if severity in ("OK", "GREEN", "NORMAL", "LOW"):
            return CheckResult(
                name="ops_watchdog", status=HealthLevel.OK,
                detail=f"watchdog {severity or 'OK'}",
                last_event_ts=ts,
            )
        if severity in ("WARN", "WARNING", "YELLOW", "MEDIUM"):
            return CheckResult(
                name="ops_watchdog", status=HealthLevel.WARN,
                detail=f"watchdog {severity}",
                last_event_ts=ts,
            )
        if severity in ("ALERT", "CRITICAL", "RED", "HALT", "HIGH", "SEVERE"):
            return CheckResult(
                name="ops_watchdog", status=HealthLevel.ALERT,
                detail=f"watchdog {severity}",
                last_event_ts=ts,
                remedy=("Check data/ops_watchdog/widget_state.json for which rule "
                        "triggered. Run engine.agents.ops_watchdog.runner to refresh."),
            )
        return CheckResult(
            name="ops_watchdog", status=HealthLevel.UNKNOWN,
            detail=f"unrecognized severity {severity!r}",
            last_event_ts=ts,
        )
    except Exception as exc:
        return CheckResult(
            name="ops_watchdog", status=HealthLevel.UNKNOWN,
            detail=f"widget_state parse failed: {exc}",
        )


ALL_CHECKS = [
    # Discovery side
    check_discovery_freshness,
    check_gate_freshness,
    check_queue_drain,
    check_llm_budget,
    # Book side — per-senior integration with existing systems
    check_paper_trade_freshness,
    check_decay_sentinel,
    check_ops_watchdog,
]


def _aggregate(checks: list[CheckResult]) -> HealthLevel:
    """Overall status = worst of all checks."""
    if any(c.status == HealthLevel.ALERT for c in checks):
        return HealthLevel.ALERT
    if any(c.status == HealthLevel.WARN for c in checks):
        return HealthLevel.WARN
    if all(c.status == HealthLevel.OK for c in checks):
        return HealthLevel.OK
    return HealthLevel.UNKNOWN


def report() -> dict:
    """Run all checks + emit a single report dict."""
    results: list[CheckResult] = []
    for fn in ALL_CHECKS:
        try:
            results.append(fn())
        except Exception as exc:
            logger.warning("health check %s crashed: %s", fn.__name__, exc)
            results.append(CheckResult(
                name=fn.__name__,
                status=HealthLevel.UNKNOWN,
                detail=f"check crashed: {exc}",
            ))
    overall = _aggregate(results)
    return {
        "status":    overall.value,
        "as_of":     datetime.datetime.utcnow().isoformat() + "Z",
        "checks":    [c.to_dict() for c in results],
        "tunables": {
            "discovery_slo_hours":     DISCOVERY_SLO_HOURS,
            "gate_slo_days":           GATE_SLO_DAYS,
            "queue_bloat_threshold":   QUEUE_BLOAT_THRESHOLD,
            "llm_budget_weekly_usd":   LLM_BUDGET_WEEKLY,
        },
    }


def _cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=["json", "human"], default="human")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = report()
    if args.format == "json":
        print(json.dumps(r, indent=2, default=str))
        return 0
    print(f"PIPELINE HEALTH: {r['status']}")
    print(f"As of: {r['as_of']}")
    print(f"{'─' * 56}")
    for c in r["checks"]:
        print(f"  [{c['status']:<5}] {c['name']:<24} {c['detail']}")
        if c["expected"]:
            print(f"            expected: {c['expected']}")
    return 0 if r["status"] in ("OK", "WARN") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
