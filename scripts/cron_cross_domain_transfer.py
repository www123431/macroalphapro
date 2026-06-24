"""scripts/cron_cross_domain_transfer.py — β monthly per-sleeve cron
(2026-06-14). Step 2 of the α/β automation pair.

For each currently-deployed sleeve, propose 1-2 cross-asset transfers
(Frazzini-Pedersen 2018 70% institutional alpha = enhance, not new
factor). Output rows in data/research/transfer_proposals.jsonl;
human reviews via /research/library/detail.

Selection logic:
  Eligible sleeve =
      mechanism_library YAML present + has `id` field
      AND no transfer proposal in last TRANSFER_REFRESH_DAYS days
      (default 30)

Cap per run: TRANSFER_MONTHLY_CAP (default 13 = current deployed sleeve
count). At ~$0.30/sleeve, monthly cost ~$4. Configurable via env
TRANSFER_MONTHLY_CAP.

Cron registration: scripts/install_agentic_cron.py — MONTHLY on day 1
at 07:05 SGT (immediately after pre_mortem). Monthly cadence matches
the timescale at which sleeve composition / asset-class catalog
changes — re-proposing weekly would just churn near-duplicate ideas.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ON_PATH))

logger = logging.getLogger(__name__)

REPO_ROOT          = Path(__file__).resolve().parents[1]
HEALTH_PATH        = REPO_ROOT / "data" / "agents" / "_health" / "cross_domain_transfer.jsonl"
LIBRARY_DIR        = REPO_ROOT / "data" / "research" / "mechanism_library"
TRANSFER_PATH      = REPO_ROOT / "data" / "research" / "transfer_proposals.jsonl"

MONTHLY_CAP_DEFAULT     = 13
TRANSFER_REFRESH_DAYS   = 30


def _record(status: str, *, elapsed_s: float, n_eligible: int = 0,
            n_run: int = 0, n_proposals: int = 0,
            error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":     "cross_domain_transfer",
        "ts":           _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":       status,
        "elapsed_s":    round(elapsed_s, 2),
        "date_key":     _dt.date.today().isoformat(),
        "n_eligible":   n_eligible,
        "n_run":        n_run,
        "n_proposals":  n_proposals,
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _load_deployed_sleeves() -> list[str]:
    """Read mechanism_library YAMLs, return list of sleeve ids."""
    try:
        import yaml as _pyyaml
    except Exception:
        logger.warning("yaml not installed")
        return []
    out: list[str] = []
    for fp in sorted(LIBRARY_DIR.glob("*.yaml")):
        if fp.name.startswith("_"):
            continue
        try:
            d = _pyyaml.safe_load(fp.read_text(encoding="utf-8"))
            if isinstance(d, dict) and d.get("id"):
                out.append(d["id"])
        except Exception:
            continue
    return out


def _latest_transfer_ts(sleeve_id: str) -> str | None:
    """Newest proposed_ts in transfer_proposals.jsonl for this sleeve,
    or None."""
    if not TRANSFER_PATH.is_file():
        return None
    latest = None
    for ln in TRANSFER_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("source_sleeve_id") != sleeve_id:
            continue
        ts = r.get("proposed_ts") or ""
        if latest is None or ts > latest:
            latest = ts
    return latest


def _filter_need_transfer(sleeves: list[str]) -> list[str]:
    refresh_cutoff_iso = (
        _dt.datetime.utcnow() - _dt.timedelta(days=TRANSFER_REFRESH_DAYS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: list[str] = []
    for sid in sleeves:
        latest = _latest_transfer_ts(sid)
        if latest is None or latest < refresh_cutoff_iso:
            out.append(sid)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    cap = int(os.environ.get("TRANSFER_MONTHLY_CAP", MONTHLY_CAP_DEFAULT))
    try:
        from engine.research.cross_domain_transfer import propose_transfers
        sleeves = _load_deployed_sleeves()
        needs = _filter_need_transfer(sleeves)
        to_run = needs[:cap]

        n_run = 0
        n_proposals = 0
        for sid in to_run:
            try:
                proposals = propose_transfers(sid)
                if proposals is None:
                    continue
                n_run += 1
                n_proposals += len(proposals)
            except Exception:
                logger.exception("transfer failed on %s", sid)

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed,
                n_eligible=len(sleeves), n_run=n_run, n_proposals=n_proposals)
        print(f"[cron_cross_domain_transfer] ok — {len(sleeves)} sleeves total / "
              f"{len(needs)} need fresh / ran {n_run} (cap={cap}) → "
              f"{n_proposals} proposals in {elapsed:.1f}s")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed,
                error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_cross_domain_transfer failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
