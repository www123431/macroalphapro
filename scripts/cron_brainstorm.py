"""scripts/cron_brainstorm.py — Phase 5 demand-driven brainstorm trigger
(2026-06-15).

Without this cron, the brainstorm system sits dormant — only fires when
a human clicks Run on /research/brainstorm. Senior-quant model is that
brainstorm fires when the system DEMANDS new direction:

  T1  CONFIRMED_DECAY in last 7d        → anomaly_inversion pack
  T2  empty family in expected list     → best-matching pack
  T3  RED cluster ≥3 in 30d same family → behavioral_inverse pack
  T4  weekly default (Mon)              → rotate 1 of 7 packs

Each fired trigger runs ONE brainstorm session (~$0.10 post-senior-
upgrade) and ideas land in brainstorm_drafts.jsonl for PM review.

Caps:
  - BRAINSTORM_WEEKLY_CAP=4 (env, default 4 sessions/week ≈ $0.40)
  - Per-run cap = 2 trigger types (so a noisy day doesn't fire all 4)
  - 24h dedup per (trigger_type, family/sleeve) — same condition
    doesn't fire twice in a day

Soft-warn on monthly cost overrun (per
[[project-brainstorm-architecture-2026-06-14]] design).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT_ON_PATH = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_ON_PATH) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_ON_PATH))

REPO_ROOT      = _REPO_ROOT_ON_PATH
HEALTH_PATH    = REPO_ROOT / "data" / "agents" / "_health" / "brainstorm.jsonl"
DRAFTS_PATH    = REPO_ROOT / "data" / "research" / "brainstorm_drafts.jsonl"
AUTOPSY_PATH   = REPO_ROOT / "data" / "research" / "autopsies.jsonl"
DECAY_PATH     = REPO_ROOT / "data" / "research" / "decay_retest_results.jsonl"
HYP_PATH       = REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
LIBRARY_DIR    = REPO_ROOT / "data" / "research" / "mechanism_library"
TRIGGERS_LOG   = REPO_ROOT / "data" / "research" / "brainstorm_triggers.jsonl"

WEEKLY_CAP_DEFAULT     = 4
PER_RUN_TRIGGER_CAP    = 2
DEDUP_WINDOW_HOURS     = 24
RED_CLUSTER_THRESHOLD  = 3
RED_CLUSTER_WINDOW_DAYS = 30
DECAY_TRIGGER_WINDOW_DAYS = 7

# Family → best-matching seed pack (T2 demand-driven mapping). When a
# family is empty in our queue, we pick the pack most likely to seed
# baseline ideas for that domain. Hardcoded judgment — review when
# seed pack roster changes.
FAMILY_TO_PACK = {
    # Carry across asset classes
    "CARRY":             "macro_regime_shifts",
    "CARRY_FX":          "macro_regime_shifts",
    "BOND_CARRY":        "macro_regime_shifts",
    "MUNI_CARRY":        "macro_regime_shifts",
    # Vol / VRP — physics-friendly (state transitions / tail dynamics)
    "VRP":               "physics_analogies",
    "VOL_RISK_PREMIUM":  "physics_analogies",
    # Momentum / TSMOM / cross-asset MOM — horizon-sensitive
    "MOMENTUM":          "time_horizon_arbitrage",
    "CROSS_ASSET_MOMENTUM": "time_horizon_arbitrage",
    "SPANNING_MOM":      "time_horizon_arbitrage",
    # Cross-section equity factor families
    "VALUE":             "cross_section_anomaly_inversion",
    "PROFITABILITY":     "cross_section_anomaly_inversion",
    "INVESTMENT":        "cross_section_anomaly_inversion",
    "SIZE":              "cross_section_anomaly_inversion",
    "LOW_VOL":           "behavioral_inverse",
    "REVERSAL":          "behavioral_inverse",
    "SHORT_INTEREST":    "behavioral_inverse",
    "ATTENTION":         "alternative_data",
    # Events
    "EVENT_DRIFT":       "alternative_data",
    "EARNINGS_DRIFT":    "alternative_data",
    # Fixed income / spreads / convertibles
    "CDX_BASIS":         "network_theory",
    "CONVERTIBLE_ARB":   "macro_regime_shifts",
    "MORTGAGE_PREPAY":   "macro_regime_shifts",
    "MERGER_ARB":        "network_theory",
    # Default for unknown family
    "_default":          "physics_analogies",
}

# T4 weekly rotation (Monday-only by default; controlled by env or
# fallback to weekday check)
WEEKLY_ROTATION_PACKS = [
    "physics_analogies",
    "network_theory",
    "behavioral_inverse",
    "alternative_data",
    "macro_regime_shifts",
    "cross_section_anomaly_inversion",
    "time_horizon_arbitrage",
]


def _record(status: str, *, elapsed_s: float, triggers_fired: list[dict],
            n_ideas: int, error: str | None = None) -> None:
    HEALTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "agent_id":       "brainstorm",
        "ts":             _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "status":         status,
        "elapsed_s":      round(elapsed_s, 2),
        "date_key":       _dt.date.today().isoformat(),
        "n_triggers":     len(triggers_fired),
        "n_ideas":        n_ideas,
        "trigger_kinds":  [t["trigger"] for t in triggers_fired],
    }
    if error:
        row["error"] = error[:500]
    with HEALTH_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _iter_jsonl(p: Path):
    if not p.is_file():
        return
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except Exception:
            continue


def _hours_since(ts_iso: str) -> float:
    try:
        ts = _dt.datetime.fromisoformat(ts_iso.rstrip("Z"))
        return (_dt.datetime.utcnow() - ts).total_seconds() / 3600.0
    except Exception:
        return 1e9


def _was_recently_triggered(trigger_kind: str, key: str) -> bool:
    """Dedup check: same (trigger, key) fired in last DEDUP_WINDOW_HOURS?"""
    for r in _iter_jsonl(TRIGGERS_LOG):
        if r.get("trigger") != trigger_kind:
            continue
        if r.get("key") != key:
            continue
        if _hours_since(r.get("fired_at") or "") < DEDUP_WINDOW_HOURS:
            return True
    return False


def _log_trigger_fired(trigger: dict) -> None:
    TRIGGERS_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "fired_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        **trigger,
    }
    with TRIGGERS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ─── Trigger detectors ──────────────────────────────────────────────


def _t1_confirmed_decay_triggers() -> list[dict]:
    """T1: any sleeve with CONFIRMED_DECAY in last DECAY_TRIGGER_WINDOW_DAYS."""
    cutoff = (_dt.datetime.utcnow() -
              _dt.timedelta(days=DECAY_TRIGGER_WINDOW_DAYS)).strftime(
              "%Y-%m-%dT%H:%M:%SZ")
    out: list[dict] = []
    seen: set[str] = set()
    for r in _iter_jsonl(DECAY_PATH):
        if r.get("verdict") != "CONFIRMED_DECAY":
            continue
        if (r.get("triggered_at") or "") < cutoff:
            continue
        sleeve = r.get("sleeve_id", "")
        if not sleeve or sleeve in seen:
            continue
        seen.add(sleeve)
        if _was_recently_triggered("decay_replacement", sleeve):
            continue
        out.append({
            "trigger":      "decay_replacement",
            "key":          sleeve,
            "pack":         "cross_section_anomaly_inversion",
            "context":      f"sleeve {sleeve} confirmed dead — propose replacement",
        })
    return out


def _t2_empty_family_triggers() -> list[dict]:
    """T2: families in EXPECTED_FAMILIES YAML (deployed_or_seeded) with
    0 hypothesis in queue."""
    out: list[dict] = []
    try:
        import yaml as _pyyaml
        yp = LIBRARY_DIR / "_expected_families.yaml"
        if not yp.is_file():
            return []
        d = _pyyaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        # Only fire T2 for DEPLOYED-or-SEEDED (these are families we
        # actively test in) — aspirational_gap is too speculative to
        # auto-brainstorm without confirmed data availability.
        expected = [str(x).upper() for x in (d.get("deployed_or_seeded") or [])]
    except Exception:
        return []
    fam_count: dict[str, int] = defaultdict(int)
    for h in _iter_jsonl(HYP_PATH):
        fam = (h.get("mechanism_family") or "").upper()
        if fam:
            fam_count[fam] += 1
    for fam in expected:
        if fam_count.get(fam, 0) > 0:
            continue
        if _was_recently_triggered("empty_family_seed", fam):
            continue
        pack = FAMILY_TO_PACK.get(fam, FAMILY_TO_PACK["_default"])
        out.append({
            "trigger": "empty_family_seed",
            "key":     fam,
            "pack":    pack,
            "context": f"family {fam} is expected-coverage but has 0 hypotheses — seed baseline ideas",
        })
    return out


def _t3_red_cluster_triggers() -> list[dict]:
    """T3: ≥ N RED autopsies in last 30d for same family → pivot."""
    cutoff = (_dt.datetime.utcnow() -
              _dt.timedelta(days=RED_CLUSTER_WINDOW_DAYS)).strftime(
              "%Y-%m-%dT%H:%M:%SZ")
    fam_red: dict[str, int] = defaultdict(int)
    for r in _iter_jsonl(AUTOPSY_PATH):
        if r.get("superseded_by") or r.get("actual_verdict") != "RED":
            continue
        if (r.get("ts") or "") < cutoff:
            continue
        fam = (r.get("strategy_family") or "").upper()
        if fam and fam != "OTHER":
            fam_red[fam] += 1
    out: list[dict] = []
    for fam, n in fam_red.items():
        if n < RED_CLUSTER_THRESHOLD:
            continue
        if _was_recently_triggered("red_cluster_pivot", fam):
            continue
        out.append({
            "trigger":  "red_cluster_pivot",
            "key":      fam,
            "pack":     "behavioral_inverse",
            "context":  f"{n} RED in {fam} last 30d — pivot mechanism via behavioral angle",
        })
    return out


def _t4_weekly_default_trigger() -> list[dict]:
    """T4: Monday default rotation through 7 packs (cycle by week-of-year)."""
    today = _dt.date.today()
    if today.weekday() != 0:    # Monday only by default
        return []
    week_of_year = today.isocalendar()[1]
    pack = WEEKLY_ROTATION_PACKS[week_of_year % len(WEEKLY_ROTATION_PACKS)]
    key = f"week_{today.isoformat()}_{pack}"
    if _was_recently_triggered("weekly_default", key):
        return []
    return [{
        "trigger": "weekly_default",
        "key":     key,
        "pack":    pack,
        "context": f"weekly default rotation (week {week_of_year}) → {pack}",
    }]


# ─── Main ───────────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = _dt.datetime.utcnow()
    weekly_cap = int(os.environ.get("BRAINSTORM_WEEKLY_CAP", WEEKLY_CAP_DEFAULT))
    try:
        # 1) Collect all triggers
        triggers: list[dict] = []
        triggers.extend(_t1_confirmed_decay_triggers())
        triggers.extend(_t2_empty_family_triggers())
        triggers.extend(_t3_red_cluster_triggers())
        triggers.extend(_t4_weekly_default_trigger())

        # 2) Rank by priority (decay > red-cluster > empty > weekly)
        priority = {"decay_replacement": 0, "red_cluster_pivot": 1,
                    "empty_family_seed": 2, "weekly_default": 3}
        triggers.sort(key=lambda t: priority.get(t["trigger"], 9))

        # 3) Apply weekly-cap soft warn
        n_this_week = _count_sessions_this_week()
        if n_this_week >= weekly_cap:
            logger.warning("brainstorm: weekly cap reached (%d/%d) — SOFT WARN, "
                           "continuing per soft-cap design decision",
                           n_this_week, weekly_cap)

        # 4) Cap per-run (avoid multi-trigger flood on one cron)
        to_run = triggers[:PER_RUN_TRIGGER_CAP]

        # 5) Fire each trigger
        from engine.research.brainstorm.divergent_generator import brainstorm_session
        n_ideas_total = 0
        fired: list[dict] = []
        for t in to_run:
            try:
                ideas = brainstorm_session(
                    t["pack"], trigger=t["trigger"],
                    trigger_context=t["context"],
                )
                if ideas is None:
                    logger.warning("brainstorm: trigger %s/%s LLM call failed",
                                    t["trigger"], t["pack"])
                    continue
                n_ideas_total += len(ideas)
                fired.append(t)
                _log_trigger_fired(t)
            except Exception:
                logger.exception("brainstorm: trigger %s failed", t["trigger"])

        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("ok", elapsed_s=elapsed, triggers_fired=fired,
                n_ideas=n_ideas_total)
        print(f"[cron_brainstorm] ok — {len(triggers)} triggers detected, "
              f"{len(fired)} fired (cap={PER_RUN_TRIGGER_CAP}), "
              f"{n_ideas_total} total ideas, {elapsed:.1f}s "
              f"(week count {n_this_week}/{weekly_cap})")
        return 0
    except Exception as exc:
        elapsed = (_dt.datetime.utcnow() - t0).total_seconds()
        _record("error", elapsed_s=elapsed, triggers_fired=[],
                n_ideas=0, error=f"{type(exc).__name__}: {exc}")
        logger.exception("cron_brainstorm failed")
        return 1


def _count_sessions_this_week() -> int:
    """Count brainstorm sessions in the current ISO week from drafts file
    (each session_id is one Sonnet call)."""
    today = _dt.date.today()
    monday = today - _dt.timedelta(days=today.weekday())
    monday_iso = monday.isoformat()
    sessions: set[str] = set()
    for r in _iter_jsonl(DRAFTS_PATH):
        ts = (r.get("generated_ts") or "")[:10]
        if ts >= monday_iso:
            sid = r.get("session_id")
            if sid:
                sessions.add(sid)
    return len(sessions)


if __name__ == "__main__":
    sys.exit(main())
