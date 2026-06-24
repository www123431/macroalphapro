"""scripts/run_attribution_report.py — Layer 4 piece 3b CLI.

Produces a markdown report aggregating the attribution lifecycle over
a rolling window. Default window = 180 days.

Output is human-readable + machine-parseable (one section per
aggregate). Future piece 3c will read these aggregates directly via the
lifecycle.py API and use them to reweight watchlist / doctrine
retrieval — the markdown report is for the human reader.

Usage
-----
  python scripts/run_attribution_report.py
  python scripts/run_attribution_report.py --days 90
  python scripts/run_attribution_report.py --json    # machine-readable
  python scripts/run_attribution_report.py --lifecycle <hypothesis_id>
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import sys
from pathlib import Path

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.agents.attribution.lifecycle import (
    aggregate_by_author, aggregate_by_doctrine_snippet,
    aggregate_by_source, calibration_a_confidence,
    get_candidate_lifecycle,
)


def _pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def _render_author_section(rows) -> str:
    if not rows:
        return ("### By watchlist author\n\n"
                "_No watchlisted authors have been cited yet "
                "in the rolling window._\n")
    lines = [
        "### By watchlist author\n",
        "How often each adversarial author was cited + their "
        "downstream conversion rate to GREEN.\n",
        "| Author | Cited | B✓ | Princ✓ | Strict | GREEN | RED | →GREEN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.author_name} "
            f"| {r.n_candidates_cited} "
            f"| {r.n_b_approved} "
            f"| {r.n_principal_approved} "
            f"| {r.n_strict_gate_run} "
            f"| {r.n_green} "
            f"| {r.n_red} "
            f"| {_pct(r.conversion_rate_to_green)} |"
        )
    return "\n".join(lines) + "\n"


def _render_source_section(rows) -> str:
    if not rows:
        return ("### By source\n\n_No candidates in the rolling "
                "window._\n")
    lines = [
        "### By source\n",
        "arxiv (RSS crawl) vs semantic_scholar (watchlist-driven) "
        "vs unknown. Counts represent CANDIDATES that cited "
        "at least one paper from each source.\n",
        "| Source | Cited | B✓ | Princ✓ | Strict | GREEN | RED | →GREEN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.source} "
            f"| {r.n_candidates_cited} "
            f"| {r.n_b_approved} "
            f"| {r.n_principal_approved} "
            f"| {r.n_strict_gate_run} "
            f"| {r.n_green} "
            f"| {r.n_red} "
            f"| {_pct(r.conversion_rate_to_green)} |"
        )
    return "\n".join(lines) + "\n"


def _render_doctrine_section(rows) -> str:
    if not rows:
        return ("### By doctrine snippet\n\n_No synthesis runs "
                "produced candidates with doctrine_snippet_ids in "
                "the window._\n")
    lines = [
        "### By doctrine snippet\n",
        "Which memory entries (from ChromaDB doctrine collection) "
        "were retrieved before A produced GREEN candidates.\n",
        "| Memory entry | Runs | Cands | GREEN | RED | →GREEN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.memory_file_id}` "
            f"| {r.n_synthesis_runs_seen} "
            f"| {r.n_candidates_in_those_runs} "
            f"| {r.n_green} "
            f"| {r.n_red} "
            f"| {_pct(r.conversion_rate_to_green)} |"
        )
    return "\n".join(lines) + "\n"


def _render_calibration_section(rows) -> str:
    if not rows:
        return ("### A's confidence calibration\n\n"
                "_No candidates in the window._\n")
    lines = [
        "### A's confidence calibration\n",
        "For each tier A predicted in expected_outcome_prior — "
        "what % of those candidates actually reached GREEN?\n",
        "| Predicted tier | Cands | Reached strict | GREEN | actual→GREEN |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.a_predicted_tier or '(unknown)'} "
            f"| {r.n_candidates} "
            f"| {r.n_reached_strict_gate} "
            f"| {r.n_actual_green} "
            f"| {_pct(r.actual_green_rate)} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown_report(days: int) -> str:
    """Full attribution report as markdown."""
    auth   = aggregate_by_author(days)
    src    = aggregate_by_source(days)
    doct   = aggregate_by_doctrine_snippet(days)
    calib  = calibration_a_confidence(days)

    header = (
        f"# Attribution report — last {days} days\n\n"
        f"Generated {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
        "Tracks each candidate from generation (A) through B review, "
        "principal decision, and strict-gate outcome. Aggregates by "
        "watchlist author, source, doctrine snippet, and A's "
        "predicted confidence tier.\n\n"
        "**Interpretation**: rows with low →GREEN rate are signals "
        "for piece 3c reweighting (deprioritise low-conversion "
        "authors, demote rarely-useful doctrine snippets, re-balance "
        "between arxiv RSS and SS watchlist).\n\n"
    )

    body = (
        _render_author_section(auth)        + "\n" +
        _render_source_section(src)         + "\n" +
        _render_doctrine_section(doct)      + "\n" +
        _render_calibration_section(calib)
    )

    return header + body


def render_json_report(days: int) -> str:
    """Same payload as markdown report but JSON-serialised dataclasses."""
    def _ds(rows):
        return [dataclasses.asdict(r) for r in rows]

    payload = {
        "days":         days,
        "generated_ts": _dt.datetime.utcnow().strftime(
                            "%Y-%m-%dT%H:%M:%SZ"),
        "by_author":            _ds(aggregate_by_author(days)),
        "by_source":            _ds(aggregate_by_source(days)),
        "by_doctrine_snippet":  _ds(aggregate_by_doctrine_snippet(days)),
        "calibration":          _ds(calibration_a_confidence(days)),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_lifecycle(hypothesis_id: str) -> str:
    """Single-candidate lifecycle as readable text."""
    lc = get_candidate_lifecycle(hypothesis_id)
    if lc is None:
        return (f"No Hypothesis record found for "
                f"hypothesis_id={hypothesis_id!r}\n")
    d = dataclasses.asdict(lc)
    return json.dumps(d, indent=2, ensure_ascii=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, default=180,
                    help="Rolling window in days (default 180)")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of markdown")
    p.add_argument("--lifecycle", type=str, default=None,
                    help="Print full lifecycle for one hypothesis_id "
                          "and exit")
    args = p.parse_args()

    if args.lifecycle:
        sys.stdout.write(render_lifecycle(args.lifecycle))
        return 0

    if args.json:
        sys.stdout.write(render_json_report(args.days))
    else:
        sys.stdout.write(render_markdown_report(args.days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
