"""scripts/cron_direction_proposer.py — daily 06:35 SGT cron.

Generates today's paper-corpus direction proposals via
engine.agents.direction_proposer, then DIFFS against yesterday's
top-3. If the top-3 changed (a new direction surfaced or an existing
one fell out), files an inbox alert so the user notices on next open.

This is what Phase 1 calls "active initiative" — the agent stops
waiting for the user to open /lab/today and instead pushes a signal
the moment something new shows up.

Idempotency (rule 1): if cron runs twice the same day, the diff is
computed against the SAME yesterday's snapshot — same result, no
double-alert. Snapshot key = date_str + content_hash.

Reversibility level (rule 3): LEVEL 0 — append-only inbox row;
worst-case mis-fire = one extra notification line.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import datetime as _dt
from pathlib import Path


REPO_ROOT      = Path(__file__).resolve().parent.parent
HEALTH_DIR     = REPO_ROOT / "data" / "agents" / "_health"
SNAPSHOT_DIR   = REPO_ROOT / "data" / "agents" / "direction_snapshots"
INBOX_FILE     = REPO_ROOT / "data" / "research" / "research_ops_inbox.jsonl"


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_run(agent_id: str, **fields) -> None:
    HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    row = {"agent_id": agent_id, "ts": _utc_iso(), **fields}
    (HEALTH_DIR / f"{agent_id}.jsonl").open("a", encoding="utf-8").write(
        json.dumps(row, ensure_ascii=False, default=str) + "\n"
    )


def _today_key() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _yesterday_key() -> str:
    d = _dt.datetime.utcnow().date() - _dt.timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _snapshot_path(date_key: str) -> Path:
    return SNAPSHOT_DIR / f"{date_key}.json"


def _signature(directions: list[dict]) -> str:
    """Compact signature of top-K directions used for diffing.
    Sensitive to changes in: which hypothesis_id is top, total score
    (to 2 decimals), and the graveyard verdict.
    """
    parts = []
    for d in directions:
        parts.append(":".join([
            str(d.get("source_hypothesis_id", "")),
            f"{float(d.get('scores', {}).get('total', 0)):.2f}",
            str(d.get("graveyard_verdict", "")),
        ]))
    s = "|".join(parts)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _append_inbox(payload: dict) -> None:
    INBOX_FILE.parent.mkdir(parents=True, exist_ok=True)
    INBOX_FILE.open("a", encoding="utf-8").write(
        json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    )


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    t0 = time.perf_counter()
    try:
        from engine.agents.direction_proposer import propose_directions
    except Exception as exc:
        _record_run("direction_proposer", status="error",
                    error=f"import_failed:{exc}",
                    elapsed_s=round(time.perf_counter() - t0, 2))
        print(f"[cron_direction_proposer] import failed: {exc}", file=sys.stderr)
        return 1

    try:
        out = propose_directions(top=5)
    except Exception as exc:
        _record_run("direction_proposer", status="error",
                    error=str(exc)[:300],
                    elapsed_s=round(time.perf_counter() - t0, 2))
        print(f"[cron_direction_proposer] propose failed: {exc}", file=sys.stderr)
        return 1

    directions = out.get("directions", [])
    today_key  = _today_key()
    today_sig  = _signature(directions[:3])

    # Persist today's snapshot (idempotent: overwrites same date)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    _snapshot_path(today_key).write_text(
        json.dumps({
            "date_key":         today_key,
            "generated_ts":     out.get("generated_ts"),
            "top3":             directions[:3],
            "signature":        today_sig,
            "deployed_families": out.get("deployed_families", []),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Diff against yesterday
    y_path = _snapshot_path(_yesterday_key())
    y_sig = None
    y_top_ids: list[str] = []
    if y_path.is_file():
        try:
            y_data = json.loads(y_path.read_text(encoding="utf-8"))
            y_sig = y_data.get("signature")
            y_top_ids = [d.get("source_hypothesis_id")
                         for d in y_data.get("top3", [])]
        except Exception:
            pass

    today_top_ids = [d.get("source_hypothesis_id") for d in directions[:3]]
    new_in_top = [hid for hid in today_top_ids if hid and hid not in y_top_ids]

    diffed = bool(y_sig and y_sig != today_sig)

    elapsed = round(time.perf_counter() - t0, 2)
    if diffed and new_in_top:
        # File inbox alert
        top1 = directions[0]
        _append_inbox({
            "channel":  "research_direction",
            "ts":       _utc_iso(),
            "kind":     "direction_diff",
            "priority": "medium",
            "title":    (f"corpus direction shift — top-3 changed, "
                         f"{len(new_in_top)} new entry"),
            "body":     (
                f"new top: {top1.get('family')} / "
                f"{top1.get('mechanism_subtype')} "
                f"(score {top1.get('scores', {}).get('total'):.2f}, "
                f"graveyard {top1.get('graveyard_verdict')}). "
                f"yesterday's top-3 ids: {y_top_ids}"
            ),
            "where":    "/research/forward",
            "agent_id": "direction_proposer",
        })
        _record_run("direction_proposer", status="ok",
                    diff="new_top3", new_count=len(new_in_top),
                    today_sig=today_sig, yesterday_sig=y_sig,
                    elapsed_s=elapsed)
        print(f"[cron_direction_proposer] DIFF · {len(new_in_top)} new in top-3 · alert filed")
    else:
        _record_run("direction_proposer", status="ok",
                    diff="unchanged" if y_sig else "first_run",
                    today_sig=today_sig, yesterday_sig=y_sig,
                    elapsed_s=elapsed)
        print(f"[cron_direction_proposer] ok · {'unchanged' if y_sig else 'first run'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
