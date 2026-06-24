"""scripts/research_daily_summary.py — one-pager status of the research
pipeline, generated daily by cron or on-demand.

Outputs a single Markdown file at data/research/daily_summary.md (and
prints to stdout). Surfaces:
  - Last 24h discovery run results (papers fetched, queued, borderline)
  - Current queue counts (review, borderline)
  - Promote/skip activity (last 7 days)
  - Funnel back-test agreement rate (if recently run)
  - Auto-gate verdicts in the last 24h
  - WRDS source health
  - LLM cost ledger (today, this week)

Designed to be readable in a terminal, in an email, or in a chat
notification. No emoji per project rule.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DISCOVERY_QUEUE = REPO_ROOT / "data" / "research" / "discovery_queue.jsonl"
DISCOVERY_BORDERLINE = REPO_ROOT / "data" / "research" / "discovery_borderline.jsonl"
DISCOVERY_LOG = REPO_ROOT / "data" / "research" / "discovery_log.jsonl"
DISCOVERY_RUNS = REPO_ROOT / "data" / "research" / "discovery_runs.jsonl"
DISCOVERY_REJECTED = REPO_ROOT / "data" / "research" / "discovery_rejected.jsonl"
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"
LLM_COST = REPO_ROOT / "data" / "llm_cost_ledger.jsonl"
SUMMARY_OUT = REPO_ROOT / "data" / "research" / "daily_summary.md"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return out


def _within(ts_str: str, days: int) -> bool:
    """True if ts_str is within the last `days` days."""
    if not ts_str:
        return False
    try:
        ts = datetime.datetime.fromisoformat(ts_str.rstrip("Z"))
    except ValueError:
        return False
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    return ts >= cutoff


def build_summary() -> str:
    """Build the markdown summary string. Does not write to disk."""
    now = datetime.datetime.utcnow().isoformat() + "Z"
    lines = [
        f"# Research Daily Summary",
        f"",
        f"Generated: {now}",
        f"",
    ]

    # Pipeline health badge — top of summary, borrowed from PIT audit
    # display style (PASS/FLAG inline with detail + remedy when needed)
    try:
        from engine.research.discovery.pipeline_health import report as health_report
        h = health_report()
        lines += [
            f"## PIPELINE HEALTH: **{h['status']}**",
            "",
        ]
        for c in h["checks"]:
            badge = c["status"]
            lines.append(f"  [{badge:<5}] **{c['name']}** — {c['detail']}")
            if c.get("remedy") and badge in ("WARN", "ALERT"):
                lines.append(f"           remedy: {c['remedy']}")
        lines.append("")
    except Exception as exc:
        lines += [f"## PIPELINE HEALTH: UNKNOWN (check crashed: {exc})", ""]

    # Queue counts
    review = _read_jsonl(DISCOVERY_QUEUE)
    borderline = _read_jsonl(DISCOVERY_BORDERLINE)
    lines += [
        "## Queue State",
        "",
        f"- Review queue (primary):       **{len(review)}** entries",
        f"- Borderline queue (spot-check): **{len(borderline)}** entries",
        "",
    ]

    # Last 24h discovery runs
    runs = _read_jsonl(DISCOVERY_RUNS)
    recent_runs = [r for r in runs if _within(r.get("timestamp_utc", ""), 1)]
    lines += [
        "## Discovery Runs (last 24h)",
        "",
        f"- Runs:    **{len(recent_runs)}**",
    ]
    total_fetched = sum(r.get("papers_fetched", 0) for r in recent_runs)
    lines += [f"- Papers fetched: **{total_fetched}**"]
    # Sum stage counts
    stage_totals: Counter = Counter()
    for r in recent_runs:
        s = (r.get("summary") or {}).get("stage_counts") or {}
        for k, v in s.items():
            stage_totals[k] += v
    if stage_totals:
        lines.append(f"- Stage outcomes:")
        for s, n in stage_totals.most_common():
            lines.append(f"  - {s}: {n}")
    lines.append("")

    # Promote / skip activity in last 7 days
    rejected = _read_jsonl(DISCOVERY_REJECTED)
    recent_skips = [r for r in rejected if _within(r.get("skipped_at", ""), 7)]
    lines += [
        "## Review Activity (last 7d)",
        "",
        f"- Skipped: **{len(recent_skips)}**",
    ]
    skip_reasons: Counter = Counter(r.get("skip_reason", "") for r in recent_skips)
    if skip_reasons:
        lines.append(f"- Top skip reasons:")
        for reason, n in skip_reasons.most_common(3):
            lines.append(f"  - {reason!r}: {n}")

    # Promotes: count YAMLs in library written today (best-effort)
    lib_dir = REPO_ROOT / "data" / "research" / "mechanism_library"
    today = datetime.date.today()
    cutoff_ts = (today - datetime.timedelta(days=7))
    recent_yamls = []
    if lib_dir.exists():
        for fp in lib_dir.glob("*.yaml"):
            try:
                mtime = datetime.date.fromtimestamp(fp.stat().st_mtime)
                if mtime >= cutoff_ts:
                    recent_yamls.append(fp.name)
            except OSError:
                continue
    lines.append(f"- Promoted (library YAMLs created in last 7d): **{len(recent_yamls)}**")
    for fn in recent_yamls[:10]:
        lines.append(f"  - {fn}")
    lines.append("")

    # Gate runs (last 24h)
    gates = _read_jsonl(GATE_RUNS)
    recent_gates = [g for g in gates if _within(g.get("ts", ""), 1)]
    lines += [
        "## Strict Gate Activity (last 24h)",
        "",
        f"- Runs: **{len(recent_gates)}**",
    ]
    if recent_gates:
        verdicts: Counter = Counter()
        for g in recent_gates:
            v = (g.get("verdict") or "").upper().split()[0] if g.get("verdict") else ""
            verdicts[v or "UNKNOWN"] += 1
        for v, n in verdicts.most_common():
            lines.append(f"  - {v}: {n}")
        # Provisional-synthetic count
        prov = sum(1 for g in recent_gates if g.get("provisional_synthetic"))
        if prov:
            lines.append(f"- Provisional-synthetic verdicts: **{prov}** "
                            f"(auto-gated, not real backtest)")
    lines.append("")

    # Forward OOS watchlist summary + calibration delta surface
    try:
        from engine.research.discovery.forward_oos_observer import (
            compute_calibration_delta, get_watchlist, watchlist_summary,
        )
        wl = watchlist_summary()
        lines += [
            "## Forward OOS Watchlist",
            "",
            f"- Total entries: **{wl['total']}**",
        ]
        if wl["by_state"]:
            lines.append("- By state:")
            for state, n in wl["by_state"].items():
                lines.append(f"  - {state}: {n}")
        if wl["total"] > 0:
            ready = wl["by_implementation"]["ready"]
            lines.append(
                f"- Implementation ready: **{ready}** "
                f"(others still need binding code + data wiring)"
            )
            if wl["overdue_for_review"]:
                lines.append(
                    f"- Overdue (track_until passed but not graduated): "
                    f"**{wl['overdue_for_review']}** — review needed"
                )

        # AUTO-GATE CALIBRATION DELTA — close the empirical feedback loop.
        # For each watchlist entry that has REAL gate runs (non-synthetic),
        # compute (real_sharpe - auto_gate_sharpe) so we can SEE whether
        # auto-gate is calibrated or just guessing.
        deltas = []
        mismatches = []
        for entry in get_watchlist():
            mid = entry.get("mechanism_id", "")
            if not mid:
                continue
            cd = compute_calibration_delta(mid)
            if not cd.get("has_real_runs"):
                continue
            if cd.get("calibration_delta") is not None:
                deltas.append((mid, cd["calibration_delta"]))
            if cd.get("verdict_mismatch"):
                mismatches.append(
                    (mid, cd.get("auto_gate_verdict"),
                       cd.get("real_latest_verdict"))
                )
        if deltas:
            lines.append("")
            lines.append("### Auto-gate Calibration (auto-gate vs real)")
            lines.append("")
            for mid, delta in deltas:
                direction = ("**underestimated**" if delta > 0
                              else "**overestimated**")
                lines.append(
                    f"  - `{mid}`: delta={delta:+.3f} ({direction} by Sharpe)"
                )
            if mismatches:
                lines.append("")
                lines.append("- Verdict mismatches:")
                for mid, ag, real in mismatches:
                    lines.append(f"  - `{mid}`: auto-gate={ag} -> real={real}")
        lines.append("")
    except Exception as exc:
        lines += [f"## Forward OOS Watchlist (check failed: {exc})", ""]

    # LLM cost (today + this week)
    costs = _read_jsonl(LLM_COST)
    today_costs = [c for c in costs if _within(c.get("ts", ""), 1)]
    week_costs = [c for c in costs if _within(c.get("ts", ""), 7)]
    today_usd = sum(c.get("cost_usd", 0) for c in today_costs)
    week_usd = sum(c.get("cost_usd", 0) for c in week_costs)
    lines += [
        "## LLM Cost",
        "",
        f"- Today:     **${today_usd:.4f}** ({len(today_costs)} calls)",
        f"- This week: **${week_usd:.4f}** ({len(week_costs)} calls)",
        "",
    ]

    return "\n".join(lines)


def write_summary(content: str, path: Path = SUMMARY_OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(SUMMARY_OUT),
                        help="Output path (default: data/research/daily_summary.md)")
    parser.add_argument("--no-write", action="store_true",
                        help="Print to stdout only; don't write file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    content = build_summary()
    print(content)
    if not args.no_write:
        out_path = write_summary(content, Path(args.out))
        print(f"\n[written to {out_path}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
