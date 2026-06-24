"""scripts/cron_replication_checker.py — γ Replication Checker
auto-trigger (2026-06-14).

Companion to scripts/cron_pre_mortem.py (α). For each eligible
hypothesis without a fresh replication check, run the γ specialist
(single Sonnet call scanning the hyp against Hou-Xue-Zhang 2020 /
McLean-Pontiff 2016 / Linnainmaa-Roberts 2018 / Fama-French 2018
catalogs).

Cap: REPLICATION_DAILY_CAP=5 → $0.25/day = $7.50/month.

Cron registration: scripts/install_agentic_cron.py — daily 07:10 SGT
(after α 07:00 so both run on the same eligible set without colliding).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ON_PATH))

REPO_ROOT        = _REPO_ROOT_ON_PATH
HEALTH_PATH      = REPO_ROOT / "data" / "agents" / "_health" / "replication_checker.jsonl"
HYP_PATH         = REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
REPL_PATH        = REPO_ROOT / "data" / "research" / "replication_checks.jsonl"

DAILY_CAP_DEFAULT       = 5
REPLICATION_REFRESH_DAYS = 90    # rarer than α since lit doesn't change weekly


def _record(status: str, *, elapsed_s: float, n_eligible: int = 0,
            n_run: int = 0, n_dead: int = 0, n_decayed: int = 0,
            error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "replication_checker",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "n_eligible":   n_eligible,
        "n_run":        n_run,
        "n_dead":       n_dead,
        "n_decayed":    n_decayed,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _eligible_hypotheses() -> list[dict]:
    if not HYP_PATH.is_file():
        return []
    eligible_states = {"proposed", "extracted", "brainstormed", "pm_review", ""}
    out: list[dict] = []
    for ln in HYP_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            h = json.loads(ln)
        except Exception:
            continue
        if (h.get("review_state") or "").lower() not in eligible_states:
            continue
        fam = (h.get("mechanism_family") or "").upper()
        if not fam or fam == "OTHER":
            continue
        out.append(h)
    return out


def _latest_repl_ts(hyp_id: str) -> str | None:
    if not REPL_PATH.is_file():
        return None
    latest = None
    for ln in REPL_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("hypothesis_id") != hyp_id:
            continue
        ts = r.get("assessed_ts") or ""
        if latest is None or ts > latest:
            latest = ts
    return latest


def _need_repl(candidates: list[dict]) -> list[dict]:
    cutoff = (_dt.datetime.utcnow() -
              _dt.timedelta(days=REPLICATION_REFRESH_DAYS)).strftime(
              "%Y-%m-%dT%H:%M:%SZ")
    out: list[dict] = []
    for h in candidates:
        latest = _latest_repl_ts(h["hypothesis_id"])
        if latest is None or latest < cutoff:
            out.append(h)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    cap = int(os.environ.get("REPLICATION_DAILY_CAP", DAILY_CAP_DEFAULT))
    try:
        from engine.research.replication_checker import check_replication
        eligible = _eligible_hypotheses()
        needs = _need_repl(eligible)
        needs.sort(key=lambda h: h.get("created_ts") or h.get("updated_ts") or "",
                   reverse=True)
        to_run = needs[:cap]

        n_run = n_dead = n_decayed = 0
        for h in to_run:
            try:
                c = check_replication(h["hypothesis_id"])
                if c is None:
                    continue
                n_run += 1
                if c.replication_status == "PROBABLY_DEAD":
                    n_dead += 1
                elif c.replication_status == "DECAYED_BUT_LIVE":
                    n_decayed += 1
            except Exception:
                logger.exception("replication_checker failed on %s",
                                  h["hypothesis_id"])

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed,
                n_eligible=len(eligible), n_run=n_run,
                n_dead=n_dead, n_decayed=n_decayed)
        print(f"[cron_replication_checker] ok — {len(eligible)} eligible / "
              f"{len(needs)} need fresh / ran {n_run} (cap={cap}) → "
              f"{n_dead} PROBABLY_DEAD, {n_decayed} DECAYED_BUT_LIVE, "
              f"in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_replication_checker failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
