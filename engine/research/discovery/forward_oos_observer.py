"""engine/research/discovery/forward_oos_observer.py — bridge between
discovery (promote action) and book (paper-trade forward observation).

Senior B per [[feedback-senior-review-as-we-build-2026-05-30]] +
[[feedback-confirm-meaningful-before-borrowing-2026-05-30]]:
when promote() writes a library stub + auto-gate gives a PROVISIONAL
synthetic verdict, what happens next? Currently: nothing automatic.
The candidate sits as a YAML until a human writes real binding code
and runs strict-gate manually.

This module registers every promote into a watchlist + tracks
implementation status + (eventually) computes calibration delta
between auto-gate synthetic verdict and real forward-OOS Sharpe.

WHY THIS MATTERS (the 3 senior questions):
  1. Failure mode prevented: "promoted papers vanish without follow-up,
     so we can't tell if auto-gate is calibrated or just guessing."
  2. Equivalent in current code: NO — paper_trade exists for deployed
     strategies, but there's no watchlist for promoted-but-not-deployed
     candidates.
  3. Scale match: at 1-2 candidates/year, tracking each with a few
     watchlist entries + daily P&L sim is trivial overhead.

WATCHLIST STATES:
  registered      — just promoted, no binding code yet
  awaiting_data    — binding code written but waiting for fresh data
  tracking         — daily P&L being captured
  graduated        — track period (90d default) complete; verdict
                     compared against auto-gate
  retired          — manual user decision (no longer monitoring)

CALIBRATION DELTA (only computed when graduated + implementation exists):
  auto_gate_verdict  + auto_gate_sharpe   (synthetic prediction)
  forward_oos_sharpe (real-data observation)
  delta = forward_oos_sharpe - auto_gate_sharpe
  signed; positive = auto-gate underestimated; negative = overestimated
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
WATCHLIST_PATH = REPO_ROOT / "data" / "paper_trade" / "forward_oos_watchlist.jsonl"
LIBRARY_DIR = REPO_ROOT / "data" / "research" / "mechanism_library"
GATE_RUNS = REPO_ROOT / "data" / "research" / "gate_runs.jsonl"

DEFAULT_TRACK_DAYS = 90       # 30/60/90 milestones; final at 90
WATCHLIST_STATES = {
    "registered", "awaiting_data", "tracking", "graduated", "retired",
}


@dataclasses.dataclass
class WatchlistEntry:
    mechanism_id:     str
    registered_at:    str             # ISO timestamp
    promoted_from:    str             # which queue (review / borderline)
    state:            str             # see WATCHLIST_STATES
    track_until:      str             # ISO date (registered + DEFAULT_TRACK_DAYS)
    auto_gate_verdict: str | None = None
    auto_gate_sharpe:  float | None = None
    auto_gate_deflated_sr: float | None = None
    # Filled when graduated + implementation found:
    forward_oos_sharpe:     float | None = None
    forward_oos_n_months:   int | None = None
    calibration_delta:      float | None = None    # forward - auto-gate
    notes:            str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ── Watchlist I/O ─────────────────────────────────────────────────────────

def _read_watchlist() -> list[dict]:
    if not WATCHLIST_PATH.exists():
        return []
    out = []
    for line in WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write_watchlist(entries: list[dict]) -> None:
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with WATCHLIST_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")


def get_watchlist() -> list[dict]:
    """Return all current watchlist entries (most recent first)."""
    entries = _read_watchlist()
    return list(reversed(entries))


# ── Register on promote ───────────────────────────────────────────────────

def register_for_forward_oos(
    mechanism_id: str,
    *,
    promoted_from: str = "review",
    auto_gate_result: dict | None = None,
    track_days: int = DEFAULT_TRACK_DAYS,
) -> WatchlistEntry:
    """Add a freshly-promoted mechanism to the forward-OOS watchlist.

    If the mechanism_id is already in the watchlist (e.g. user
    re-promoted), the existing entry is preserved (no duplicate).

    auto_gate_result: pass the AutoGateResult.to_dict() output so we
    can later compute calibration delta. Synthetic data, but still
    a baseline prediction.
    """
    now = datetime.datetime.utcnow()
    track_until = (now + datetime.timedelta(days=int(track_days))).date().isoformat()

    existing = _read_watchlist()
    for entry in existing:
        if entry.get("mechanism_id") == mechanism_id:
            # Already registered — don't overwrite (preserves first
            # registration timestamp)
            return WatchlistEntry(**{
                k: v for k, v in entry.items()
                if k in WatchlistEntry.__dataclass_fields__
            })

    agr = auto_gate_result or {}
    entry = WatchlistEntry(
        mechanism_id=mechanism_id,
        registered_at=now.isoformat() + "Z",
        promoted_from=promoted_from,
        state="registered",
        track_until=track_until,
        auto_gate_verdict=agr.get("verdict"),
        auto_gate_sharpe=agr.get("sharpe"),
        auto_gate_deflated_sr=agr.get("deflated_sr"),
    )
    existing.append(entry.to_dict())
    _write_watchlist(existing)
    return entry


# ── Implementation status check ───────────────────────────────────────────

def check_implementation_status(mechanism_id: str) -> dict:
    """Examine library YAML to see if the mechanism has executable
    binding code yet.

    Returns: {
      yaml_exists,             # bool — was promote actually written
      has_bindings,            # bool — YAML has non-empty bindings dict
      tunable_bindings_count,  # int — how many params are tunable
      ready_for_paper_trade,   # bool — has bindings + required_data wired
    }
    """
    yaml_path = LIBRARY_DIR / f"{mechanism_id}.yaml"
    result = {
        "yaml_exists":            yaml_path.exists(),
        "has_bindings":           False,
        "tunable_bindings_count": 0,
        "ready_for_paper_trade":  False,
    }
    if not yaml_path.exists():
        return result
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return result
    bindings = data.get("bindings") or {}
    tunables = data.get("tunable_bindings") or []
    result["has_bindings"] = len(bindings) > 0
    result["tunable_bindings_count"] = len(tunables)
    # Ready = has SOMETHING in bindings + has required_data
    required = data.get("required_data") or []
    result["ready_for_paper_trade"] = bool(bindings) and bool(required)
    return result


# ── State transitions ─────────────────────────────────────────────────────

def update_state(mechanism_id: str, new_state: str) -> bool:
    """Move a watchlist entry to a new state. Returns True if updated."""
    if new_state not in WATCHLIST_STATES:
        raise ValueError(
            f"new_state must be one of {sorted(WATCHLIST_STATES)}, "
            f"got {new_state!r}"
        )
    entries = _read_watchlist()
    updated = False
    for entry in entries:
        if entry.get("mechanism_id") == mechanism_id:
            entry["state"] = new_state
            updated = True
            break
    if updated:
        _write_watchlist(entries)
    return updated


# ── Calibration delta (when graduated + implementation exists) ────────────

def compute_calibration_delta(mechanism_id: str) -> dict:
    """Compare auto-gate synthetic verdict against real forward-OOS
    observations from gate_runs.jsonl (after implementation).

    Looks for gate_runs entries with name matching mechanism_id (NOT
    starting with 'auto_gate__') and provisional_synthetic != True
    after the watchlist registration date.

    Returns: {
      has_real_runs, n_real_runs, real_sharpe_mean,
      auto_gate_sharpe, delta (real - auto-gate),
      verdict_mismatch (bool)
    }
    """
    entries = _read_watchlist()
    watch = next(
        (e for e in entries if e.get("mechanism_id") == mechanism_id),
        None,
    )
    if not watch:
        return {"error": f"{mechanism_id} not in watchlist"}

    registered_at = watch.get("registered_at", "")
    auto_gate_sharpe = watch.get("auto_gate_sharpe")
    auto_gate_verdict = watch.get("auto_gate_verdict")

    if not GATE_RUNS.exists():
        return {
            "has_real_runs":     False,
            "n_real_runs":       0,
            "auto_gate_sharpe":  auto_gate_sharpe,
            "auto_gate_verdict": auto_gate_verdict,
        }

    real_runs = []
    try:
        for line in GATE_RUNS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("name") or ""
            # Skip auto-gate entries; we want REAL backtest runs
            if name.startswith("auto_gate__"):
                continue
            if rec.get("provisional_synthetic"):
                continue
            if mechanism_id not in name and rec.get("mechanism") != mechanism_id:
                continue
            ts = rec.get("ts", "")
            if ts and registered_at and ts < registered_at:
                continue
            real_runs.append(rec)
    except Exception as exc:
        return {"error": str(exc)[:200]}

    n = len(real_runs)
    if n == 0:
        return {
            "has_real_runs":   False,
            "n_real_runs":     0,
            "auto_gate_sharpe": auto_gate_sharpe,
            "auto_gate_verdict": auto_gate_verdict,
        }

    real_sharpes = [r.get("standalone_sharpe") for r in real_runs
                       if r.get("standalone_sharpe") is not None]
    real_sharpe_mean = (sum(real_sharpes) / len(real_sharpes)
                          if real_sharpes else None)

    # Verdict mismatch check
    auto_g = (auto_gate_verdict or "").upper().split()[0] if auto_gate_verdict else ""
    real_verdicts = [(r.get("verdict") or "").upper().split()[0]
                        if r.get("verdict") else "" for r in real_runs]
    # Real verdict = latest non-empty
    real_v = next((v for v in reversed(real_verdicts) if v), "")
    verdict_mismatch = bool(auto_g and real_v and auto_g != real_v)

    delta = None
    if (real_sharpe_mean is not None
          and auto_gate_sharpe is not None):
        delta = real_sharpe_mean - auto_gate_sharpe

    return {
        "has_real_runs":     True,
        "n_real_runs":       n,
        "auto_gate_verdict": auto_gate_verdict,
        "auto_gate_sharpe":  auto_gate_sharpe,
        "real_sharpe_mean":  real_sharpe_mean,
        "real_latest_verdict": real_v,
        "calibration_delta": delta,
        "verdict_mismatch":  verdict_mismatch,
    }


# ── Watchlist summary for daily_summary integration ──────────────────────

def watchlist_summary() -> dict:
    """Aggregated stats for daily_summary."""
    entries = _read_watchlist()
    total = len(entries)
    by_state: dict[str, int] = {}
    by_implementation = {"ready": 0, "not_ready": 0}
    overdue = 0
    today_iso = datetime.date.today().isoformat()
    for e in entries:
        s = e.get("state", "")
        by_state[s] = by_state.get(s, 0) + 1
        impl = check_implementation_status(e.get("mechanism_id", ""))
        if impl.get("ready_for_paper_trade"):
            by_implementation["ready"] += 1
        else:
            by_implementation["not_ready"] += 1
        track_until = e.get("track_until", "")
        if (track_until and track_until < today_iso
              and e.get("state") not in ("graduated", "retired")):
            overdue += 1
    return {
        "total":            total,
        "by_state":         by_state,
        "by_implementation": by_implementation,
        "overdue_for_review": overdue,
    }
