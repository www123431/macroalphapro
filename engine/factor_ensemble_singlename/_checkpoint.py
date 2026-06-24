"""
engine/factor_ensemble_singlename/_checkpoint.py — walk-forward resilience layer.

Provides per-period checkpointing + resume capability so that:
  - Mid-run failures (WRDS rate-limit, connection drop, OS crash) don't lose
    hours of already-completed periods
  - Re-runs with same `run_id` skip completed periods automatically
  - Each completed period is durably persisted to disk as a JSONL line

Schema: JSONL (one line per completed period) at
  data/factor_ensemble_singlename/wave_b_checkpoints/<run_id>.jsonl

Each line:
  {
    "period_idx":         int,    # 0-indexed position in rebalance_dates
    "rebal_date":         "YYYY-MM-DD",
    "monthly_return_gross": float,
    "tc_drag":            float,
    "monthly_return_net": float,
    "turnover":           float,
    "n_active":           int,
    "weights":            {ticker: weight, ...}  # for prev_weights continuity
  }

Built 2026-05-11 for Wave B publishable verdict run resilience.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


_REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
_CHECKPOINT_DIR_DEFAULT = (
    _REPO_ROOT / "data" / "factor_ensemble_singlename" / "wave_b_checkpoints"
)


def _checkpoint_path(run_id: str, base_dir: Optional[Path] = None) -> Path:
    base = base_dir or _CHECKPOINT_DIR_DEFAULT
    base.mkdir(parents=True, exist_ok=True)
    safe_id = run_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    return base / f"{safe_id}.jsonl"


def write_period_checkpoint(
    run_id:               str,
    period_idx:           int,
    rebal_date:           datetime.date,
    monthly_return_gross: float,
    tc_drag:              float,
    monthly_return_net:   float,
    turnover:             float,
    n_active:             int,
    weights:              pd.Series,
    *,
    base_dir:             Optional[Path] = None,
) -> None:
    """Append one completed period to the checkpoint JSONL.

    Atomic per-line write (one JSON line per period). Crash-safe: partial
    writes only ever affect the LAST line, which is skipped on resume.
    """
    path = _checkpoint_path(run_id, base_dir)
    weights_dict = {str(t): float(w) for t, w in weights.items() if pd.notna(w)}
    record = {
        "period_idx":           int(period_idx),
        "rebal_date":           rebal_date.isoformat(),
        "monthly_return_gross": float(monthly_return_gross),
        "tc_drag":              float(tc_drag),
        "monthly_return_net":   float(monthly_return_net),
        "turnover":             float(turnover),
        "n_active":             int(n_active),
        "weights":              weights_dict,
        "written_at":           datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_checkpoints(
    run_id:    str,
    *,
    base_dir:  Optional[Path] = None,
) -> dict[int, dict[str, Any]]:
    """Load completed-period records from JSONL.

    Returns:
        {period_idx: record_dict}  — empty dict if checkpoint file missing
        or empty. Malformed lines are skipped (logged as warning) — JSONL
        is crash-safe so usually only the last line is partial.
    """
    path = _checkpoint_path(run_id, base_dir)
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                period_idx = int(rec["period_idx"])
                out[period_idx] = rec
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning(
                    "checkpoint %s line %d malformed (%s); skipping",
                    path.name, line_no, exc,
                )
    return out


def checkpoint_to_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Convert a loaded JSONL record into the in-memory `monthly_records`
    shape used by walk_forward.py (drops weights field; keeps return / tc /
    turnover / n_active fields)."""
    return {
        "rebal_date":           datetime.date.fromisoformat(rec["rebal_date"]),
        "monthly_return_gross": float(rec["monthly_return_gross"]),
        "tc_drag":              float(rec["tc_drag"]),
        "monthly_return_net":   float(rec["monthly_return_net"]),
        "turnover":             float(rec["turnover"]),
        "n_active":             int(rec["n_active"]),
    }


def checkpoint_to_weights(rec: dict[str, Any]) -> pd.Series:
    """Reconstruct prev_weights pd.Series from checkpoint record."""
    return pd.Series(rec.get("weights", {}), dtype=float)
