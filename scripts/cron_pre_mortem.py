"""scripts/cron_pre_mortem.py — α Pre-Mortem auto-trigger (2026-06-14).

Step 1 of the α/β automation pair (the Step 2 sibling is
scripts/cron_cross_domain_transfer.py). Without this cron, the
Pre-Mortem button on /research/hypothesis stays manual — most
hypotheses entering /research/forward never get adversarial pre-
review, defeating the Stigler-1973 / Kahneman pre-mortem doctrine.

Selection logic:
  Eligible hypothesis =
      review_state ∈ {extracted, brainstormed, pm_review, ""} (NOT
        approved/rejected — those are post-decision)
      AND mechanism_family NOT IN {"", "OTHER"}  (skip catch-all)
      AND (no pre-mortem in data/research/pre_mortems.jsonl
            OR latest pre-mortem > PRE_MORTEM_REFRESH_DAYS old)

Daily cap (PRE_MORTEM_DAILY_CAP=5): cost control. At $0.05/hyp =
$0.25/day, $7.50/month. Configurable via env PRE_MORTEM_DAILY_CAP.

Cron registration: scripts/install_agentic_cron.py — daily 07:00 SGT
(15 min after workflow_executor so any new hypotheses from morning
synthesis are in scope).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/cron_pre_mortem.py` from any cwd (cron contexts
# don't always start in the repo root).
_REPO_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ON_PATH))

logger = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parents[1]
HEALTH_PATH        = REPO_ROOT / "data" / "agents" / "_health" / "pre_mortem.jsonl"
HYP_PATH           = REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
PRE_MORTEM_PATH    = REPO_ROOT / "data" / "research" / "pre_mortems.jsonl"

DAILY_CAP_DEFAULT          = 5
PRE_MORTEM_REFRESH_DAYS    = 30   # re-run if older than this


def _record(status: str, *, elapsed_s: float, n_eligible: int = 0,
            n_run: int = 0, n_kill: int = 0, n_caveats: int = 0,
            error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "pre_mortem",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "n_eligible":   n_eligible,
        "n_run":        n_run,
        "n_kill":       n_kill,
        "n_caveats":    n_caveats,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _load_eligible_hypotheses() -> list[dict]:
    """Hypotheses worth pre-mortem'ing: in research-eligible state,
    have a real family, and missing fresh pre-mortem."""
    if not HYP_PATH.is_file():
        return []
    eligible_states = {"proposed", "extracted", "brainstormed", "pm_review", ""}
    eligible: list[dict] = []
    for ln in HYP_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            h = json.loads(ln)
        except Exception:
            continue
        rs = h.get("review_state") or ""
        fam = (h.get("mechanism_family") or "").upper()
        if rs.lower() not in eligible_states:
            continue
        if not fam or fam == "OTHER":
            continue
        eligible.append(h)
    return eligible


def _latest_pre_mortem_ts(hyp_id: str) -> str | None:
    """Newest pre_mortem assessed_ts for this hypothesis_id, or None
    if no pre-mortem ever run for it."""
    if not PRE_MORTEM_PATH.is_file():
        return None
    latest = None
    for ln in PRE_MORTEM_PATH.read_text(encoding="utf-8").splitlines():
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


def _filter_need_pre_mortem(candidates: list[dict]) -> list[dict]:
    refresh_cutoff_iso = (
        _dt.datetime.utcnow() - _dt.timedelta(days=PRE_MORTEM_REFRESH_DAYS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[dict] = []
    for h in candidates:
        latest = _latest_pre_mortem_ts(h["hypothesis_id"])
        if latest is None or latest < refresh_cutoff_iso:
            out.append(h)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    daily_cap = int(os.environ.get("PRE_MORTEM_DAILY_CAP", DAILY_CAP_DEFAULT))
    try:
        from engine.research.pre_mortem import generate_pre_mortem
        eligible = _load_eligible_hypotheses()
        needs = _filter_need_pre_mortem(eligible)
        # Newest first — prefer recently-extracted hypotheses
        needs.sort(key=lambda h: h.get("created_ts") or h.get("updated_ts") or "", reverse=True)
        to_run = needs[:daily_cap]

        n_kill = 0
        n_caveats = 0
        n_run = 0
        for h in to_run:
            try:
                rep = generate_pre_mortem(h["hypothesis_id"])
                if rep is None:
                    continue
                n_run += 1
                if rep.overall_kill_recommendation == "KILL_BEFORE_TEST":
                    n_kill += 1
                elif rep.overall_kill_recommendation == "TEST_WITH_CAVEATS":
                    n_caveats += 1
            except Exception:
                logger.exception("pre_mortem failed on %s", h["hypothesis_id"])

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed,
                n_eligible=len(eligible), n_run=n_run,
                n_kill=n_kill, n_caveats=n_caveats)
        print(f"[cron_pre_mortem] ok — {len(eligible)} eligible / "
              f"{len(needs)} need fresh / ran {n_run} (cap={daily_cap}) → "
              f"{n_kill} KILL_BEFORE_TEST, {n_caveats} TEST_WITH_CAVEATS, "
              f"in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_pre_mortem failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
