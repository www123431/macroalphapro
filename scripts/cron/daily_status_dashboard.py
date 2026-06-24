"""scripts/cron/daily_status_dashboard.py — first integration layer.

Built 2026-06-23 per principal: "现在的问题就是怎么把这些组件整合起来"
(the real problem is how to integrate these components).

This is the smallest valuable integration step: a single markdown
that aggregates the headline of every major output the system
produces. Read-only. $0 LLM. Future cross-classifier scoring +
weekly ranked queue can read this as input.

Aggregates from:
  - data/research/belief_track_record.md             (Brier headline)
  - data/research/belief_ensemble_sweep.md           (ensemble status)
  - data/agents/direction_snapshots/<latest>.json    (top-3 directions)
  - data/research_store/audit/fegd_*/all_sleeves_exposure_scan.json
                                                       (factor exposure gaps)
  - data/research/decay_alerts.jsonl                 (last 7d decay)
  - data/cron_burndown/outcomes/<recent>.json        (last 7d verdicts)
  - data/portfolio_replay/v1_combined_replay_verdict.json
                                                       (deployed book stats)
  - data/research_store/hypotheses.jsonl             (queue depth)

Output: data/research/STATUS.md (single dashboard, ~200 lines)

Cron: registered for daily 06:40 (after daily_belief_refresh at 06:35).
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "data" / "research" / "STATUS.md"


def _now() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Section builders ─────────────────────────────────────────────


def section_belief() -> list[str]:
    lines = ["## 1. Belief Layer (predictor calibration)", ""]
    track_record = REPO_ROOT / "data" / "research" / "belief_track_record.json"
    rigor = REPO_ROOT / "data" / "research" / "belief_track_record_rigor.json"
    ensemble = REPO_ROOT / "data" / "research" / "belief_ensemble_loocv.md"

    tr = _read_json(track_record)
    rg = _read_json(rigor)

    if tr:
        n = tr.get("n_autopsies", 0)
        mb = tr.get("mean_brier_overall", "n/a")
        lines.append(f"- **n_autopsies**: {n}")
        if isinstance(mb, (int, float)):
            lines.append(f"- **mean Brier**: {mb:.4f} (baseline random = 0.444; pure family-empirical CV ~= 0.260)")
        if rg and "T1_overall_brier_bootstrap" in rg:
            t1 = rg["T1_overall_brier_bootstrap"]
            lo, hi = t1.get("ci_95_lo"), t1.get("ci_95_hi")
            if lo is not None and hi is not None:
                lines.append(f"- **95% CI**: [{lo:.4f}, {hi:.4f}]")
        lines.append(f"- **ensemble**: ACTIVATED (BELIEF_ENSEMBLE_BLEND_ENABLED=True post-v0.9); pure family-empirical w=1.0 globally")
        if ensemble.is_file():
            lines.append(f"- LOOCV report: `{ensemble.relative_to(REPO_ROOT)}`")
    else:
        lines.append("_belief track record not available_")
    lines.append("")
    return lines


def section_directions() -> list[str]:
    lines = ["## 2. New-Direction Queue (direction_proposer top-3)", ""]
    snap_dir = REPO_ROOT / "data" / "agents" / "direction_snapshots"
    if not snap_dir.is_dir():
        lines.append("_no snapshots yet_")
        lines.append("")
        return lines
    snaps = sorted(snap_dir.glob("*.json"))
    if not snaps:
        lines.append("_no snapshots yet_")
        lines.append("")
        return lines
    latest = snaps[-1]
    data = _read_json(latest)
    if not data:
        lines.append(f"_failed to read {latest.name}_")
        lines.append("")
        return lines
    lines.append(f"_Latest snapshot: {latest.stem}_")
    lines.append("")
    top3 = data.get("top3", []) or []
    if not top3:
        lines.append("_no top-3 ranked directions_")
    else:
        lines.append("| # | family | subtype | paper | rationale |")
        lines.append("|---|---|---|---|---|")
        for i, t in enumerate(top3[:3], 1):
            fam = t.get("family", "?")
            subtype = (t.get("mechanism_subtype") or "")[:30]
            paper = (t.get("paper_title") or "")[:50].replace("|", "/")
            rat = (t.get("rationale") or "")[:60].replace("|", "/")
            lines.append(f"| {i} | {fam} | {subtype} | {paper} | {rat} |")
    lines.append("")
    return lines


def section_fegd() -> list[str]:
    lines = ["## 3. Factor Exposure Gaps (FEGD per-sleeve)", ""]
    audit_dir = REPO_ROOT / "data" / "research_store" / "audit"
    if not audit_dir.is_dir():
        lines.append("_no FEGD audit output yet_")
        lines.append("")
        return lines
    fegd_dirs = sorted(audit_dir.glob("fegd_*"))
    if not fegd_dirs:
        lines.append("_no FEGD audit output yet_")
        lines.append("")
        return lines
    latest = fegd_dirs[-1]
    scan = latest / "all_sleeves_exposure_scan.json"
    if not scan.is_file():
        scan = latest / "equity_book_exposure_report.json"
    data = _read_json(scan) or {}
    lines.append(f"_Latest FEGD scan: {latest.name}/{scan.name if scan.is_file() else 'N/A'}_")
    lines.append("")
    if data:
        gap_freq = data.get("gap_frequency", {}) or {}
        if gap_freq:
            lines.append("**Gap factor frequency across sleeves** (factor → sleeves missing it):")
            lines.append("")
            for fac, n in sorted(gap_freq.items(), key=lambda kv: -kv[1])[:8]:
                lines.append(f"- `{fac}`: missing in {n} sleeve(s)")
        proposals = data.get("proposals", []) or []
        if proposals:
            lines.append("")
            lines.append(f"**{len(proposals)} ProposedDirection rows emitted** for principal review")
    lines.append("")
    return lines


def section_decay() -> list[str]:
    lines = ["## 4. Decay Alerts (last 7 days)", ""]
    path = REPO_ROOT / "data" / "research" / "decay_alerts.jsonl"
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=7)
    recent = []
    for row in _iter_jsonl(path):
        ts = row.get("alert_ts") or row.get("ts", "")
        try:
            t = _dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue
        if t >= cutoff:
            recent.append(row)
    if not recent:
        lines.append("_no decay alerts in last 7 days_")
    else:
        lines.append("| sleeve | level | mechanism | ts |")
        lines.append("|---|---|---|---|")
        for r in recent[:8]:
            sl = r.get("sleeve_id", "?")
            lvl = r.get("alert_level") or r.get("severity", "?")
            mech = (r.get("mechanism") or "")[:40]
            ts = (r.get("alert_ts") or r.get("ts", ""))[:10]
            lines.append(f"| {sl} | {lvl} | {mech} | {ts} |")
    lines.append("")
    return lines


def section_recent_verdicts() -> list[str]:
    lines = ["## 5. Recent Verdicts (last 7 days, from burndown outcomes)", ""]
    out_dir = REPO_ROOT / "data" / "cron_burndown" / "outcomes"
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=7)
    if not out_dir.is_dir():
        lines.append("_no outcomes dir_")
        lines.append("")
        return lines
    recent_files = []
    for f in sorted(out_dir.glob("*.json")):
        try:
            t = _dt.datetime.strptime(f.stem.split("_")[0], "%Y-%m-%d")
        except Exception:
            continue
        if t >= cutoff:
            recent_files.append(f)
    if not recent_files:
        lines.append("_no burndown outcomes in last 7 days_")
        lines.append("")
        return lines
    summary: dict[str, int] = {}
    for f in recent_files:
        d = _read_json(f) or {}
        # Outcome JSON has "outcomes" key (list of per-candidate dicts)
        outcomes = d.get("outcomes") or d.get("per_candidate", []) or []
        for c in outcomes:
            v = c.get("verdict") or c.get("extract_err") or "UNKNOWN"
            summary[v] = summary.get(v, 0) + 1
    lines.append(f"_Aggregated across {len(recent_files)} burndown run(s)_")
    lines.append("")
    lines.append("| outcome | n |")
    lines.append("|---|---|")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    return lines


def section_deployed_book() -> list[str]:
    lines = ["## 6. Deployed Book (canonical 4-sleeve replay)", ""]
    path = REPO_ROOT / "data" / "portfolio_replay" / "v1_combined_replay_verdict.json"
    data = _read_json(path)
    if not data:
        lines.append("_replay verdict not found_")
        lines.append("")
        return lines
    cm = data.get("combined_metrics") or {}
    lines.append(f"- **Sharpe (replay)**: {cm.get('sharpe', '?')}")
    lines.append(f"- **ann return**: {cm.get('ann_ret', '?')}")
    lines.append(f"- **ann vol**: {cm.get('ann_vol', '?')}")
    lines.append(f"- **MaxDD**: {cm.get('max_dd', '?')}")
    lines.append(f"- **n_weeks**: {cm.get('n_weeks', '?')} (~9.4 years)")
    forward = data.get("expected_forward_band") or {}
    if forward:
        lines.append(f"- **forward expectation band**: Sharpe {forward.get('sharpe_low', '?')}-{forward.get('sharpe_high', '?')}")

    # Live paper trade
    nav_path = REPO_ROOT / "data" / "research" / "nav_history.jsonl"
    nav_rows = list(_iter_jsonl(nav_path))
    if nav_rows:
        first = nav_rows[0]
        last = nav_rows[-1]
        try:
            cum = (float(last.get("equity", 0)) / float(first.get("equity", 1)) - 1) * 100
        except Exception:
            cum = 0
        lines.append("")
        lines.append(f"_Live paper trade: {len(nav_rows)} NAV records from {first.get('as_of')} to {last.get('as_of')}, cum return {cum:+.2f}% (sample too short for Sharpe inference)_")
    lines.append("")
    return lines


def section_hypothesis_queue() -> list[str]:
    lines = ["## 7. Hypothesis Queue Depth", ""]
    hyps_path = REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
    rows = list(_iter_jsonl(hyps_path))
    if not rows:
        lines.append("_no hypotheses_")
        lines.append("")
        return lines
    by_review = {}
    for r in rows:
        rs = r.get("review_state", "unknown")
        by_review[rs] = by_review.get(rs, 0) + 1
    lines.append(f"- **total hypotheses**: {len(rows)}")
    for rs, n in sorted(by_review.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{rs}`: {n}")
    lines.append("")
    return lines


def section_cron_status() -> list[str]:
    lines = ["## 8. Cron Status (last-run timestamps)", ""]
    log_paths = {
        "burndown (Mon+Thu 09:00)":    REPO_ROOT / "data" / "cron_burndown" / "logs",
        "papers_curator (daily 08:30)": REPO_ROOT / "data" / "papers_curator" / "logs",
        "belief_refresh (daily 06:35)": REPO_ROOT / "data" / "belief" / "logs",
    }
    lines.append("| cron | last log file | last touched |")
    lines.append("|---|---|---|")
    for label, log_dir in log_paths.items():
        if not log_dir.is_dir():
            lines.append(f"| {label} | _no log dir_ | — |")
            continue
        logs = sorted(log_dir.glob("*.log"))
        if not logs:
            lines.append(f"| {label} | _no logs_ | — |")
            continue
        latest = logs[-1]
        mt = _dt.datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {label} | `{latest.name}` | {mt} |")
    lines.append("")
    return lines


def section_links() -> list[str]:
    return [
        "## 9. Related Artifacts (full reports)",
        "",
        "- `data/research/belief_track_record.md` — Phase-3 calibration aggregate",
        "- `data/research/belief_track_record_rigor.md` — W6 6-test statistical rigor",
        "- `data/research/belief_ensemble_sweep.md` — per-family ensemble analysis",
        "- `data/research/belief_ensemble_loocv.md` — LOOCV robustness check",
        "- `data/research/deployed_book_attribution.md` — full 4-sleeve attribution",
        "- `data/papers_curator/claim_type_coverage_report.md` — ClaimType router stats",
        "- `docs/arxiv_preprint_draft_2026-06-22.md` — arxiv preprint v0.9",
        "- `PROJECT_OVERVIEW.md` — external-facing project summary",
        "- `INTERNAL_DESIGN_INDEX.md` — internal design memory (v2)",
        "",
    ]


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:
    print(f"[status] generating {OUT_PATH.relative_to(REPO_ROOT)}...")
    sections = [
        f"# MacroAlphaPro — Daily Status Dashboard",
        "",
        f"_Generated {_now()}. Read-only aggregation of all major outputs. $0 LLM. Daily 06:40 refresh._",
        "",
        "**Read order**: scan Sections 1+6 (Belief + Deployed) first → if anomalies, drill into Section 9 detail reports.",
        "",
    ]
    builders = [
        section_belief,
        section_directions,
        section_fegd,
        section_decay,
        section_recent_verdicts,
        section_deployed_book,
        section_hypothesis_queue,
        section_cron_status,
        section_links,
    ]
    for fn in builders:
        try:
            sections.extend(fn())
        except Exception as exc:
            sections.append(f"_section `{fn.__name__}` failed: {type(exc).__name__}: {exc}_")
            sections.append("")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(sections), encoding="utf-8")
    print(f"[status] wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(sections)} lines)")


if __name__ == "__main__":
    main()
