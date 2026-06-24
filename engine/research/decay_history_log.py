"""engine/research/decay_history_log.py — persistence wrapper for the
existing engine.validation.decay_sentinel.

The validation/decay_sentinel module produces a rich per-run report
but does NOT persist results across runs. Phase 1 P0 robustness needs
HISTORY so we can detect "Sharpe below baseline for 6 consecutive
months" (vs single-run snapshot).

WHAT THIS WRAPPER DOES:
  1. Run engine.validation.decay_sentinel.sentinel_report()
  2. Persist each per-mechanism row to history JSONL (append)
  3. Compute consecutive-months-below-threshold from history
  4. Escalate to HARD_ALERT after N consecutive months
  5. Write current alert state to alerts JSONL (overwrite)

OUTPUT FILES:
  data/research/decay_history.jsonl   per-run per-mechanism rows (append)
  data/research/decay_alerts.jsonl    current alerts (overwrite)

CRON USAGE:
  python -m engine.research.decay_history_log    # daily / weekly

Per [[feedback-loop-is-robustness-doctrine-2026-05-31]] Phase 1 P0.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = REPO_ROOT / "data" / "research" / "decay_history.jsonl"
ALERTS_PATH = REPO_ROOT / "data" / "research" / "decay_alerts.jsonl"

DEFAULT_DECAY_RATIO_THRESHOLD = 0.5
DEFAULT_CONSECUTIVE_THRESHOLD = 6


def _load_history() -> dict[str, list[dict]]:
    """{mechanism_name: [audit, ...]} sorted by audit_date."""
    out: dict[str, list[dict]] = {}
    if not HISTORY_PATH.exists():
        return out
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            name = entry.get("mechanism")
            if name:
                out.setdefault(name, []).append(entry)
        except Exception:
            continue
    for name, audits in out.items():
        audits.sort(key=lambda e: e.get("audit_date", ""))
    return out


def _consecutive_below(history: list[dict], threshold: float) -> int:
    """Count consecutive most-recent audits with decay_ratio < threshold."""
    count = 0
    for entry in reversed(history):
        r = entry.get("decay_ratio")
        if r is None:
            break
        if r < threshold:
            count += 1
        else:
            break
    return count


def _escalate_alert(consecutive: int, threshold_consec: int) -> str:
    if consecutive >= threshold_consec:
        return "HARD_ALERT"
    if consecutive >= max(2, threshold_consec // 2):
        return "SOFT_ALERT"
    if consecutive >= 1:
        return "INFORMATIONAL"
    return "OK"


def _recommendation(alert_level: str, mech_name: str) -> str:
    if alert_level == "OK":
        return "no action; continue monitoring"
    if alert_level == "INFORMATIONAL":
        return f"{mech_name}: 1 month below threshold; informational"
    if alert_level == "SOFT_ALERT":
        return (f"{mech_name}: persistent underperformance; review "
                f"attribution + consider weight reduction")
    return (f"{mech_name}: HARD decay; recommend pause + full re-audit "
            f"via engine.research.candidate_pipeline; fallback to backup "
            f"sleeve if available")


def run_history_audit(
    ratio_threshold: float = DEFAULT_DECAY_RATIO_THRESHOLD,
    consecutive_threshold: int = DEFAULT_CONSECUTIVE_THRESHOLD,
) -> list[dict]:
    """Main entry: run sentinel_report + persist history + compute escalation.

    Returns list of per-mechanism dicts with:
      mechanism, audit_date, decay_ratio, rolling_sharpe, full_sharpe,
      crisis_payoff, role, structural_decay,
      consecutive_below_threshold, alert_level, recommendation
    """
    try:
        from engine.validation.decay_sentinel import (
            build_mechanisms, sentinel_report, _market_monthly,
        )
    except ImportError as exc:
        logger.error("decay_sentinel import failed: %s", exc)
        return []

    try:
        mechs = build_mechanisms()
        mkt = _market_monthly()
        report = sentinel_report(mechs, market=mkt)
    except Exception as exc:
        logger.error("sentinel_report failed: %s", exc)
        return []

    history = _load_history()
    today = datetime.date.today().isoformat()
    results = []
    for name, h in report.get("mechanisms", {}).items():
        decay_ratio = h.get("decay_ratio")
        # Some reports return decay_ratio as fraction, others as scalar
        if isinstance(decay_ratio, str):
            try:
                decay_ratio = float(decay_ratio.rstrip("%")) / 100
            except Exception:
                decay_ratio = None
        if decay_ratio is not None and (np.isnan(decay_ratio)
                                              or np.isinf(decay_ratio)):
            decay_ratio = None
        crisis = report.get("crisis", {}).get(name)
        if crisis is not None and (np.isnan(crisis) or np.isinf(crisis)):
            crisis = None

        audit_entry = {
            "mechanism":       name,
            "audit_date":      today,
            "decay_ratio":     decay_ratio,
            "rolling_sharpe":  h.get("rolling_sharpe"),
            "full_sharpe":     h.get("full_sharpe"),
            "rolling_t":       h.get("rolling_t"),
            "crisis_payoff":   crisis,
            "role":            report.get("roles", {}).get(name),
            "structural_decay": (
                report.get("decay", {}).get(name, {}).get("structural_decay")
            ),
        }
        # Compute consecutive
        prior = history.get(name, [])
        consec = 0
        if decay_ratio is not None and decay_ratio < ratio_threshold:
            consec = 1 + _consecutive_below(prior, ratio_threshold)
        audit_entry["consecutive_below_threshold"] = consec
        audit_entry["alert_level"] = _escalate_alert(consec, consecutive_threshold)
        audit_entry["recommendation"] = _recommendation(audit_entry["alert_level"], name)
        results.append(audit_entry)

    # Persist history (append) + alerts (overwrite)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            if r["alert_level"] != "OK":
                f.write(json.dumps(r) + "\n")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratio-threshold", type=float,
                          default=DEFAULT_DECAY_RATIO_THRESHOLD)
    parser.add_argument("--consecutive", type=int,
                          default=DEFAULT_CONSECUTIVE_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    results = run_history_audit(
        ratio_threshold=args.ratio_threshold,
        consecutive_threshold=args.consecutive,
    )
    print(f"[decay_history_log] {datetime.date.today().isoformat()} "
          f"audited {len(results)} mechanism(s)")
    print(f"  ratio_threshold={args.ratio_threshold}, "
          f"consecutive_for_HARD={args.consecutive}")
    print()
    print(f"  {'mechanism':<18}  {'role':<14}  {'roll_Sh':>8}  "
          f"{'decay':>8}  {'consec':>6}  {'alert':<15}")
    print(f"  {'-'*18}  {'-'*14}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*15}")
    for r in results:
        roll = r.get("rolling_sharpe")
        roll_str = f"{roll:>+8.3f}" if roll is not None else f"{'n/a':>8}"
        dec = r.get("decay_ratio")
        dec_str = f"{dec:>+8.2f}" if dec is not None else f"{'n/a':>8}"
        print(f"  {r['mechanism']:<18}  {(r.get('role') or 'n/a'):<14}  "
              f"{roll_str}  {dec_str}  "
              f"{r['consecutive_below_threshold']:>6d}  "
              f"{r['alert_level']:<15}")
        if r["alert_level"] != "OK":
            print(f"      -> {r['recommendation']}")
    print()
    n_hard = sum(1 for r in results if r["alert_level"] == "HARD_ALERT")
    n_soft = sum(1 for r in results if r["alert_level"] == "SOFT_ALERT")
    if n_hard + n_soft:
        print(f"  {n_hard} HARD + {n_soft} SOFT alert(s). See "
              f"{ALERTS_PATH.relative_to(REPO_ROOT)}")
    else:
        print(f"  all within tolerance; {len(results)} entries written to "
              f"{HISTORY_PATH.relative_to(REPO_ROOT)}")
    return 0 if n_hard == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
