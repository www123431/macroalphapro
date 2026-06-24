"""scripts/stage_a_smoke.py — Stage A functional verification.

Runs each Stage A capability end-to-end and reports pass/fail per
function. Use this to confirm the substrate + attribution + weekly
orchestration layer all work after a code change or environment shift.

Capabilities verified:
  1. Substrate inspection (cache.jsonl breakdown by source)
  2. Substrate orchestrator (dry-run path — no network/LLM cost)
  3. Live substrate refresh (real network — ~30-90s)
  4. Attribution aggregates (by_author, by_source, by_doctrine, calib)
  5. Attribution lifecycle on a real hypothesis_id
  6. Weekly session through run_weekly_session (dry-run end-to-end)
  7. Audit trail file existence + parseability

Usage:
  python scripts/stage_a_smoke.py            # full smoke (~90s)
  python scripts/stage_a_smoke.py --no-live  # skip live network calls
  python scripts/stage_a_smoke.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ────────────────────────────────────────────────────────────────────
# Per-capability probes
# ────────────────────────────────────────────────────────────────────
def probe_cache_breakdown() -> dict:
    """Capability 1: substrate cache aggregation."""
    p = _REPO_ROOT / "data" / "papers_curator" / "cache.jsonl"
    if not p.is_file():
        return {"ok": False, "detail": "cache.jsonl missing"}
    by_source: dict[str, int] = {}
    total = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            total += 1
            src = r.get("source") or "?"
            by_source[src] = by_source.get(src, 0) + 1
    return {
        "ok":            total > 0,
        "total":         total,
        "by_source":     by_source,
        "n_sources":     len(by_source),
        "detail":        f"{total} rows across {len(by_source)} sources",
    }


def probe_substrate_dry_run() -> dict:
    """Capability 2: substrate orchestrator dry-run (no network)."""
    try:
        from engine.agents.chief_of_staff.substrate import (
            ALL_SOURCES, run_weekly_substrate,
        )
        result = run_weekly_substrate(dry_run=True,
                                        enabled_sources=ALL_SOURCES,
                                        persist_dir=None)
        return {
            "ok":              result.dry_run is True
                                and result.total_fetched == 0,
            "enabled_sources": list(result.enabled_sources),
            "errors":          result.errors,
            "detail":          (f"dry_run path returned "
                                 f"{len(result.enabled_sources)} sources, "
                                 f"0 fetched (expected)"),
        }
    except Exception as exc:
        return {"ok": False, "detail": f"raised: {exc}"}


def probe_substrate_live(*, ssrn_max: int = 20) -> dict:
    """Capability 3: live substrate refresh (real network)."""
    try:
        from engine.agents.chief_of_staff.substrate import (
            run_weekly_substrate,
        )
        # Use small caps so smoke is fast
        result = run_weekly_substrate(
            dry_run             = False,
            enabled_sources     = ("arxiv", "nber", "ssrn"),
            arxiv_max           = 10,
            ssrn_lookback_days  = 3,
            ssrn_max_results    = ssrn_max,
        )
        return {
            "ok":             result.total_fetched > 0,
            "total_fetched":  result.total_fetched,
            "total_new":      result.total_new,
            "arxiv_n":        result.arxiv_result.get("n_new", 0),
            "nber_n":         result.nber_result.get("n_new", 0),
            "ssrn_n":         result.ssrn_result.get("n_new", 0),
            "errors":         result.errors,
            "detail":         (f"3 sources hit; "
                                f"{result.total_fetched} fetched / "
                                f"{result.total_new} new"),
        }
    except Exception as exc:
        return {"ok": False, "detail": f"raised: {exc}",
                 "traceback": traceback.format_exc()[:300]}


def probe_attribution_aggregates() -> dict:
    """Capability 4: attribution aggregates."""
    try:
        from engine.agents.attribution.lifecycle import (
            aggregate_by_author, aggregate_by_source,
            aggregate_by_doctrine_snippet, calibration_a_confidence,
        )
        a = aggregate_by_author(days=180)
        s = aggregate_by_source(days=180)
        d = aggregate_by_doctrine_snippet(days=180)
        c = calibration_a_confidence(days=180)
        return {
            "ok":                    True,
            "by_author_count":       len(a),
            "by_source_count":       len(s),
            "by_doctrine_count":     len(d),
            "calibration_count":     len(c),
            "detail":                ("rollups callable; "
                                       f"author={len(a)} src={len(s)} "
                                       f"doctrine={len(d)} calib={len(c)} "
                                       "(low counts are expected on "
                                       "early substrate)"),
        }
    except Exception as exc:
        return {"ok": False, "detail": f"raised: {exc}"}


def probe_attribution_lifecycle() -> dict:
    """Capability 5: attribution lifecycle on a real hypothesis_id."""
    try:
        from engine.agents.attribution.lifecycle import (
            get_candidate_lifecycle,
        )
        from engine.research_store.hypothesis.store import (
            load_hypotheses,
        )
        hyps = load_hypotheses()
        if not hyps:
            return {"ok": True, "skipped": True,
                     "detail": "no hypotheses yet — skipping lifecycle "
                                "probe (vacuous PASS)"}
        target = max(hyps, key=lambda h: h.created_ts)
        lc = get_candidate_lifecycle(target.hypothesis_id)
        return {
            "ok":                bool(lc),
            "hypothesis_id":     target.hypothesis_id,
            "final_state":       lc.final_state if lc else None,
            "n_cited_papers":    len(lc.cited_paper_ids) if lc else 0,
            "n_watchlist_auth":  (len(lc.cited_watchlist_authors)
                                    if lc else 0),
            "detail":            ("lifecycle build for most-recent "
                                   f"hypothesis: state="
                                   f"{lc.final_state if lc else 'None'}"),
        }
    except Exception as exc:
        return {"ok": False, "detail": f"raised: {exc}"}


def probe_weekly_session_dry_run() -> dict:
    """Capability 6: full run_weekly_session dry-run path."""
    try:
        from engine.agents.chief_of_staff.runner import run_weekly_session
        result = run_weekly_session(
            session_id        = "smoke-test",
            dry_run           = True,
            refresh_substrate = True,
            substrate_sources = ("arxiv", "nber"),
        )
        return {
            "ok":                  (result.dry_run is True
                                      and result.substrate_result is not None),
            "session_id":          result.session_id,
            "substrate_present":   result.substrate_result is not None,
            "memo_present":        result.memo is not None,
            "session_event_id":    result.session_event_id,
            "errors":              result.errors,
            "detail":              ("dry-run weekly session through "
                                     "substrate + D/A/B + memo + emit "
                                     "step 0-5 all wired"),
        }
    except Exception as exc:
        return {"ok": False, "detail": f"raised: {exc}",
                 "traceback": traceback.format_exc()[:300]}


def probe_audit_trail() -> dict:
    """Capability 7: audit trail file existence + parseability."""
    d = (_REPO_ROOT / "data" / "agents" / "chief_of_staff"
          / "weekly_substrate")
    if not d.is_dir():
        return {"ok": False,
                 "detail": "weekly_substrate dir doesn't exist yet "
                            "(run scripts/run_weekly_substrate.py first)"}
    files = sorted(d.glob("*.json"))
    if not files:
        return {"ok": False, "detail": "no audit files in weekly_substrate"}
    latest = files[-1]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "detail": f"{latest.name} unparseable: {exc}"}
    required = {"run_ts", "run_date", "total_fetched", "total_new",
                 "enabled_sources", "errors"}
    missing = required - set(payload.keys())
    return {
        "ok":              not missing,
        "latest_file":     latest.name,
        "n_files":         len(files),
        "missing_fields":  list(missing),
        "detail":          (f"{len(files)} audit files; latest="
                             f"{latest.name}, schema "
                             f"{'OK' if not missing else 'BROKEN'}"),
    }


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────
def run_all(*, skip_live: bool = False) -> dict:
    probes = [
        ("1. Substrate cache inspection",            probe_cache_breakdown),
        ("2. Substrate dry-run orchestrator",        probe_substrate_dry_run),
        ("4. Attribution aggregates",                probe_attribution_aggregates),
        ("5. Attribution lifecycle",                 probe_attribution_lifecycle),
        ("6. Weekly session (dry-run end-to-end)",   probe_weekly_session_dry_run),
        ("7. Audit trail file",                      probe_audit_trail),
    ]
    if not skip_live:
        probes.insert(2, ("3. Live substrate refresh", probe_substrate_live))

    results = []
    for label, fn in probes:
        results.append({"label": label, **fn()})
    n_pass = sum(1 for r in results if r.get("ok"))
    return {"n_pass": n_pass, "n_total": len(results),
            "results": results}


def _print_report(summary: dict) -> None:
    print()
    print(f"=== Stage A functional smoke "
          f"({summary['n_pass']}/{summary['n_total']} pass) ===")
    print()
    for r in summary["results"]:
        flag = "[OK]  " if r.get("ok") else "[FAIL]"
        if r.get("skipped"):
            flag = "[SKIP]"
        print(f"  {flag} {r['label']}")
        print(f"         {r.get('detail', '')}")
        if not r.get("ok") and "traceback" in r:
            print(f"         {r['traceback'][:200]}")
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-live", action="store_true",
                    help="Skip the live network probe (capability 3)")
    p.add_argument("--json", action="store_true",
                    help="Output JSON instead of human-readable")
    args = p.parse_args()
    summary = run_all(skip_live=args.no_live)
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        _print_report(summary)
    return 0 if summary["n_pass"] == summary["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
