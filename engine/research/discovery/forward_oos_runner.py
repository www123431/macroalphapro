"""engine/research/discovery/forward_oos_runner.py — daily engine that
actually runs paper-trade simulation for watchlist entries.

Senior loop ① per [[feedback-confirm-meaningful-before-borrowing-2026-05-30]]:
watchlist exists (B) + calibration_delta surface (②) exist, but no
daemon actually RUNS the watched mechanisms day-to-day. This module
is that daemon.

CADENCE:
  Daily cron → run_watchlist_pass():
    for entry in watchlist where state in {registered, awaiting_data, tracking}:
      status = check_implementation_status(entry)
      if not status.ready_for_paper_trade:
        keep state = registered or awaiting_data
        continue
      if state == registered:
        transition to tracking (first time we see implementation)
      run_mechanism_simulation(entry)  # write to data/paper_trade/forward_oos_runs/<id>.jsonl
      if today >= track_until:
        compute final calibration_delta + transition to graduated

The actual SIMULATION currently uses the same synthetic factor panel
auto_gate uses — this is HONEST because most watchlist entries don't
have real binding code that knows how to read CRSP. As soon as a
mechanism's YAML adds a real binding (e.g. via human edit), this
runner picks it up and forwards real data to the template instead
of synthetic.

DESIGN PRINCIPLE: honest about what's PROVISIONAL vs REAL. Each
forward_oos_runs entry carries `data_mode: synthetic|real` so the
calibration_delta computation can WEIGHT real-data runs more.

NOT auto-deployed. Cron / human still triggers this; the writer is
this module.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "data" / "paper_trade" / "forward_oos_runs"


@dataclasses.dataclass
class SimulationRun:
    mechanism_id: str
    ts:           str
    data_mode:    str           # "synthetic" | "real"
    state_before: str
    state_after:  str
    sharpe:       float | None = None
    alpha_t:      float | None = None
    deflated_sr:  float | None = None
    verdict:      str | None = None
    error:        str | None = None
    notes:        str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _runs_file_for(mechanism_id: str) -> Path:
    """One file per mechanism keeps things isolated."""
    safe = mechanism_id.replace("/", "_").replace("\\", "_")
    return RUNS_DIR / f"{safe}.jsonl"


def _append_run(run: SimulationRun) -> None:
    path = _runs_file_for(run.mechanism_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run.to_dict(), ensure_ascii=False,
                              default=str) + "\n")


def read_runs(mechanism_id: str) -> list[dict]:
    """All recorded daily runs for a mechanism. Empty if never run."""
    path = _runs_file_for(mechanism_id)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _simulate_mechanism(
    mechanism_id: str, *, write_ledger: bool = False,
) -> tuple[float | None, float | None, float | None, str, str | None]:
    """Run one simulation pass for a mechanism using its YAML.

    Tries REAL data path first (via data_resolver). Falls back to
    synthetic (via auto_gate) when real path can't be built — explicit
    data_mode tag tells caller which path was used so calibration_delta
    weights real-data runs more.

    Returns: (sharpe, alpha_t, deflated_sr, data_mode, error)
    data_mode = "real" (fetched real CRSP / etc.) | "synthetic" (fallback)
    """
    yaml_path = (REPO_ROOT / "data" / "research" / "mechanism_library"
                  / f"{mechanism_id}.yaml")
    if not yaml_path.exists():
        return None, None, None, "synthetic", "yaml_missing"

    # ── Path 1: real-data simulation ────────────────────────────────────
    try:
        import yaml
        stub = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        from engine.research.discovery.data_resolver import (
            can_resolve, resolve_panels_for_template,
        )
        ok, reason = can_resolve(stub)
        if ok:
            from engine.research.templates import TEMPLATES
            from engine.research.pipeline import run_gate
            import importlib

            template_id = stub["execution_template"]["template_id"]
            binding = stub["execution_template"].get("binding") or {}
            tunable = set(stub.get("tunable_bindings") or [])
            # Whitelist enforcement (Huatai 借鉴 ①): only pass binding keys
            # that are in the YAML's tunable_bindings whitelist
            effective_binding = {
                k: v for k, v in binding.items() if k in tunable
            } if tunable else dict(binding)

            panels = resolve_panels_for_template(stub)
            template_fn = TEMPLATES.get(template_id)
            if template_fn is None:
                raise RuntimeError(f"template {template_id!r} not registered")

            returns_series = template_fn(**panels, **effective_binding)

            # Use the template's GATE_PROFILE if it has one
            try:
                tmod = importlib.import_module(
                    f"engine.research.templates.{template_id}",
                )
                profile = getattr(tmod, "GATE_PROFILE", None)
            except Exception:
                profile = None

            verdict_record = run_gate(
                returns_series,
                name=f"forward_oos_real__{mechanism_id}",
                mechanism=f"forward-OOS REAL data run; "
                            f"template={template_id}",
                n_trials=1,
                pead_control=False,
                log=write_ledger,
                profile=profile,
            )
            return (
                verdict_record.get("standalone_sharpe"),
                verdict_record.get("alpha_t_ff5umd"),
                verdict_record.get("deflated_sr"),
                "real", None,
            )
        else:
            logger.info("data_resolver can't resolve %s: %s; falling back to synthetic",
                           mechanism_id, reason)
    except Exception as exc:
        logger.warning("real-data path failed for %s: %s; falling back to synthetic",
                          mechanism_id, exc)

    # ── Path 2: synthetic fallback (auto_gate) ──────────────────────────
    try:
        from engine.research.discovery.auto_gate import auto_gate
        result = auto_gate(yaml_path, write_ledger=write_ledger)
        if not result.ok:
            return None, None, None, "synthetic", result.error or "auto_gate_failed"
        return (
            result.sharpe, result.alpha_t, result.deflated_sr,
            "synthetic", None,
        )
    except Exception as exc:
        return None, None, None, "synthetic", str(exc)[:200]


def run_watchlist_pass(*, today: datetime.date | None = None) -> dict:
    """Scan watchlist + simulate active entries + transition states +
    compute graduation when track_until reached.

    Idempotent within a day — same mechanism only simulated once per
    calendar day even if cron fires twice (de-dup by ts date).

    Returns: aggregate stats for the daily summary.
    """
    from engine.research.discovery.forward_oos_observer import (
        check_implementation_status, compute_calibration_delta,
        get_watchlist, update_state,
    )

    today = today or datetime.date.today()
    today_iso = today.isoformat()

    summary = {
        "scanned":              0,
        "simulated":             0,
        "skipped_not_ready":    0,
        "skipped_already_today": 0,
        "transitioned":         0,
        "graduated":            0,
        "errors":               0,
    }

    for entry in get_watchlist():
        mechanism_id = entry.get("mechanism_id", "")
        if not mechanism_id:
            continue
        summary["scanned"] += 1
        state = entry.get("state", "")
        if state in ("graduated", "retired"):
            continue

        # Implementation gate
        impl = check_implementation_status(mechanism_id)
        if not impl.get("ready_for_paper_trade"):
            summary["skipped_not_ready"] += 1
            # If was "registered" but YAML now exists at least, keep at
            # awaiting_data so user sees the state has acknowledged the
            # paper exists (just no bindings yet).
            if state == "registered" and impl.get("yaml_exists"):
                if update_state(mechanism_id, "awaiting_data"):
                    summary["transitioned"] += 1
            continue

        # Already simulated today? Skip (idempotency).
        prior = read_runs(mechanism_id)
        ran_today = any(
            (r.get("ts", "")[:10] == today_iso) for r in prior
        )
        if ran_today:
            summary["skipped_already_today"] += 1
            continue

        # Transition to tracking on first ready run
        if state in ("registered", "awaiting_data"):
            if update_state(mechanism_id, "tracking"):
                summary["transitioned"] += 1
            state = "tracking"

        # Actual simulation
        sharpe, alpha_t, deflated_sr, data_mode, err = _simulate_mechanism(
            mechanism_id, write_ledger=False,
        )

        # Final state (only changes on graduation)
        track_until = entry.get("track_until", "")
        state_after = state
        if track_until and track_until <= today_iso:
            state_after = "graduated"
            if update_state(mechanism_id, "graduated"):
                summary["graduated"] += 1

        run = SimulationRun(
            mechanism_id=mechanism_id,
            ts=datetime.datetime.utcnow().isoformat() + "Z",
            data_mode=data_mode,
            state_before=state,
            state_after=state_after,
            sharpe=sharpe,
            alpha_t=alpha_t,
            deflated_sr=deflated_sr,
            verdict=None,    # computed downstream from sharpe accumulation
            error=err,
            notes=("first run since promotion" if state == "tracking"
                   and len(prior) == 0 else ""),
        )
        _append_run(run)
        if err:
            summary["errors"] += 1
        else:
            summary["simulated"] += 1

    return summary
