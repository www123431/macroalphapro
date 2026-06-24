"""engine.roadmap.store — YAML-persisted axis registry.

Roadmap is slow-changing intent (not high-frequency events), so YAML
serialization fits better than JSONL. Single file
`data/roadmap/axes.yaml` holds the full registry; git history is the
audit trail.

Mutations go through upsert_axis() — never write the YAML directly.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import yaml

from engine.roadmap.schema import (
    AxisOutcome, AxisState, AxisTier, ResearchAxis,
)

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AXES_PATH = _REPO_ROOT / "data" / "roadmap" / "axes.yaml"

_LOCK = threading.Lock()


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dir() -> None:
    _AXES_PATH.parent.mkdir(parents=True, exist_ok=True)


def _read_raw() -> dict:
    if not _AXES_PATH.is_file():
        return {"axes": {}}
    with _AXES_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {"axes": {}}


def _write_raw(raw: dict) -> None:
    _ensure_dir()
    with _AXES_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True,
                       default_flow_style=False, width=120)


def _decode_axis(axis_id: str, payload: dict) -> ResearchAxis:
    d = dict(payload or {})
    d.setdefault("axis_id", axis_id)
    return ResearchAxis.from_dict(d)


def _encode_axis(axis: ResearchAxis) -> dict:
    d = axis.to_dict()
    # YAML pretty: drop the axis_id key (it's the map key) and None metadata
    out: dict[str, Any] = {}
    for k, v in d.items():
        if k == "axis_id":
            continue
        if v is None or (isinstance(v, (list, dict)) and not v):
            # Skip empty optionals to keep YAML clean
            if k in ("decay_estimate", "capacity_estimate",
                     "related_subject_ids", "related_memory_files",
                     "next_actions"):
                continue
        out[k] = v
    return out


# ── Public API ────────────────────────────────────────────────────


def list_axes(
    state: Optional[AxisState | str] = None,
    tier: Optional[AxisTier | str] = None,
    family: Optional[str] = None,
) -> list[ResearchAxis]:
    """List axes. Optional filters. Returned order: state (active >
    queued > paused > closed), then updated_ts desc."""
    if isinstance(state, str):
        state = AxisState(state)
    if isinstance(tier, str):
        tier = AxisTier(tier)
    raw = _read_raw()
    out: list[ResearchAxis] = []
    for axis_id, payload in (raw.get("axes") or {}).items():
        try:
            axis = _decode_axis(axis_id, payload)
        except Exception:
            logger.exception("roadmap.list_axes failed to decode %s", axis_id)
            continue
        if state is not None and axis.state != state:
            continue
        if tier is not None and axis.tier != tier:
            continue
        if family is not None and axis.family != family:
            continue
        out.append(axis)
    state_order = {AxisState.active: 0, AxisState.queued: 1,
                   AxisState.paused: 2, AxisState.closed: 3}
    out.sort(key=lambda a: (state_order.get(a.state, 9), -float(a.updated_ts < a.updated_ts)))
    # secondary sort by updated_ts desc
    out.sort(key=lambda a: a.updated_ts, reverse=True)
    out.sort(key=lambda a: state_order.get(a.state, 9))
    return out


def get_axis(axis_id: str) -> Optional[ResearchAxis]:
    raw = _read_raw()
    payload = (raw.get("axes") or {}).get(axis_id)
    if payload is None:
        return None
    return _decode_axis(axis_id, payload)


def upsert_axis(
    axis_id: str,
    name: str,
    state: AxisState | str,
    tier: AxisTier | str,
    rationale: str,
    *,
    outcome: AxisOutcome | str = AxisOutcome.NONE,
    parent_axis_id: Optional[str] = None,
    family: Optional[str] = None,
    related_subject_ids: tuple[str, ...] = (),
    related_memory_files: tuple[str, ...] = (),
    next_actions: tuple[str, ...] = (),
    blocking_notes: str = "",
    decay_estimate: Optional[dict] = None,
    capacity_estimate: Optional[dict] = None,
    actor: str = "claude",
) -> ResearchAxis:
    """Insert or update an axis.

    On insert: created_ts + created_by set.
    On update: updated_ts + updated_by always refreshed; created_ts
    preserved.
    """
    if isinstance(state, str):
        state = AxisState(state)
    if isinstance(tier, str):
        tier = AxisTier(tier)
    if isinstance(outcome, str):
        outcome = AxisOutcome(outcome)

    with _LOCK:
        raw = _read_raw()
        axes = raw.setdefault("axes", {})
        existing = axes.get(axis_id)
        now = _utc_iso()
        created_ts = (existing or {}).get("created_ts") or now
        created_by = (existing or {}).get("created_by") or actor

        axis = ResearchAxis(
            axis_id=axis_id,
            name=name,
            state=state,
            tier=tier,
            outcome=outcome,
            parent_axis_id=parent_axis_id,
            family=family,
            related_subject_ids=tuple(related_subject_ids),
            related_memory_files=tuple(related_memory_files),
            rationale=rationale,
            next_actions=tuple(next_actions),
            blocking_notes=blocking_notes,
            decay_estimate=decay_estimate,
            capacity_estimate=capacity_estimate,
            created_ts=created_ts,
            updated_ts=now,
            created_by=created_by,
            updated_by=actor,
        )
        axes[axis_id] = _encode_axis(axis)
        _write_raw(raw)
        return axis


def delete_axis(axis_id: str) -> bool:
    """Hard-delete an axis from the registry. Returns True if removed.
    Prefer transitioning state to 'closed' over deletion — deletion
    loses audit history."""
    with _LOCK:
        raw = _read_raw()
        axes = raw.get("axes") or {}
        if axis_id not in axes:
            return False
        del axes[axis_id]
        raw["axes"] = axes
        _write_raw(raw)
        return True


def axes_path() -> Path:
    return _AXES_PATH
