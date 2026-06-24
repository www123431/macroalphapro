"""engine.research.burndown_caps — cron burndown weekly throughput caps.

Tracks cron-burndown-tagged successful dispatches over the last 7 days
and enforces family-rotation + global limits.

Doctrine
========
- **Family cap (HARD)**: each watched family ≤ 3 dispatches/week — forces
  rotation across PROFITABILITY / MOMENTUM / VALUE / QUALITY / VOL / etc.
  Prevents accidental PROFITABILITY-only week from inflating Bailey-LdP
  n_trials family counter.
- **Global soft cap**: 15/week — average target throughput for principal
  review bandwidth. Cron will not select more than this many candidates
  per week.
- **Global hard cap**: 25/week — emergency stop. If somehow exceeded
  (manual sessions + cron), the next cron run no-ops with a CAP_EXCEEDED
  routing slip to capability_gaps.
- **Only SUCCESSFUL dispatches count** (refusal.is None). Substrate
  dead-walls (TIER_3 TEMPLATE_NOT_CERTIFIED / TIER_4 PIT data gap) skip
  the slot — they don't run a lens, so they shouldn't consume quota.
- **Cron usage is tracked SEPARATELY from manual session usage** — the
  dispatcher's WEEKLY_CAP gate ignores cron_run_id-tagged rows.

Inputs
======
Reads data/strengthener/factor_dispatch_log.jsonl. A row is "cron-burndown
attributable" iff its `cron_run_id` field is non-null.

NOT a cap in this module
========================
- N_TRIALS_HARD (Bailey-LdP family DSR penalty) — stays in
  factor_dispatcher.pre_dispatch_check; statistical gates always apply
  whether cron or manual.
- WEEKLY_CAP — stays for manual sessions; cron is exempt by design
  (cron has its own caps below).
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DISPATCH_LOG = _REPO_ROOT / "data" / "strengthener" / "factor_dispatch_log.jsonl"


# ── Cap configuration ──────────────────────────────────────────────


FAMILY_WEEKLY_CAP = 5            # 2026-06-14 raised 3→5: dedup gate
                                  # (#11) prevents same-spec dup, so 5
                                  # legitimately-different specs per
                                  # family is still safe under Bailey-LdP
                                  # n_trials threshold scaling
WEEKLY_GLOBAL_SOFT_CAP = 50      # 2026-06-14 raised 15→50: 15 was a
                                  # principal-review-bandwidth budget for
                                  # slow-cadence review mode. In active
                                  # build-out mode the principal can
                                  # absorb 8-12/day; weekly target 50.
                                  # Bailey-LdP family cap above is the
                                  # statistical bound, this is operational.
WEEKLY_GLOBAL_HARD_CAP = 100     # 2026-06-14 raised 25→100: emergency
                                  # stop only. Soft cap is what shapes
                                  # cron throughput; hard cap is
                                  # runaway-automation backstop.

# Families that get rotation enforcement. Any family OUTSIDE this set is
# allowed to fill any remaining global capacity (subject to the global
# cap). Add new families here when their accumulated DSR pressure makes
# rotation worth enforcing.
WATCHED_FAMILIES = frozenset({
    "PROFITABILITY",
    "MOMENTUM",
    "VALUE",
    "QUALITY",
    "VOL",
    "INVESTMENT",
    "ACCRUAL",
    "CARRY",
    "TSMOM",
    "ANALYST_REVISION",
    "CROSS_ASSET_MOMENTUM",
    "EARNINGS_UNDERREACTION",
    # 2026-06-14: synced to Phase 3.1 templates + DISPATCHABLE_FAMILIES
    # in burndown_ranker. Without these in WATCHED_FAMILIES, the families
    # bypass family-cap rotation entirely and concentrate quota — VRP
    # hit 6/wk this week before global soft cap blocked further.
    "VOL_RISK_PREMIUM",          # vrp_spx template (Carr-Wu 2009)
    "EARNINGS_DRIFT",            # event_drift_pead (Bernard-Thomas 1989)
})


@_dc.dataclass(frozen=True)
class WeeklyUsage:
    """Snapshot of cron-burndown throughput over the trailing 7 days."""
    global_count:  int
    by_family:     dict[str, int]
    window_start:  str       # ISO date — inclusive
    window_end:    str       # ISO date — inclusive

    def to_dict(self) -> dict:
        return _dc.asdict(self)


# ── Reading the dispatch log ───────────────────────────────────────


def _iter_log_rows(log_path: Path):
    if not log_path.is_file():
        return
    with log_path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("burndown_caps: dispatch_log line %d malformed", ln_no)


def _row_is_cron_burndown(row: dict) -> bool:
    """True iff the dispatch row was emitted by a cron burndown run.
    Identified by non-null cron_run_id field."""
    return bool(row.get("cron_run_id"))


_QUOTA_CONSUMING_VERDICTS = frozenset({"GREEN", "MARGINAL", "RED"})


def _row_is_successful_dispatch(row: dict) -> bool:
    """True iff the row represents a template that actually produced a
    strict-gate verdict. The following do NOT consume quota:

      - Refusals (TIER_3 TEMPLATE_NOT_CERTIFIED / TIER_4 PIT data gap /
        N_TRIALS_HARD / etc) — flex-3 demand ledger catches these.
      - CUSTOM_CODE_REQUIRED — template escape hatch fired; no lens
        stack actually ran. Burning a slot on these blocks cron
        throughput for nothing.
      - EXECUTION_ERROR — template raised; no verdict produced.
      - PENDING_BUILD — template stub.

    Only GREEN / MARGINAL / RED count as real research that consumed
    LLM + compute and produced a verdict event.
    """
    if row.get("refusal") is not None:
        return False
    tr = row.get("template_result") or {}
    return (tr.get("verdict") or "") in _QUOTA_CONSUMING_VERDICTS


def _parse_ts(ts_str: str) -> Optional[_dt.datetime]:
    if not ts_str:
        return None
    try:
        return _dt.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc,
        )
    except ValueError:
        return None


def usage_last_7d(
    *,
    log_path:  Optional[Path] = None,
    now:       Optional[_dt.datetime] = None,
) -> WeeklyUsage:
    """Aggregate cron-burndown successful dispatches in the trailing 7 days.

    Bailey-LdP n_trials dedup (2026-06-17, GP/A senior audit follow-up)
    ─────────────────────────────────────────────────────────────────
    The cap counts UNIQUE `spec_hash` values per family per window, not
    raw dispatch rows. Empirical 2026-06-13/14: A's extractor produced
    8 paraphrased hypotheses from Carr 2009 VRP; each routed to the
    SAME `spec_hash = afb46008e68c3625` (same template + same signal +
    same universe); each filed its own GREEN verdict event.

    Without dedup, the 8 same-spec dispatches inflate VOL_RISK_PREMIUM
    family trials by 7. The Bailey-LdP DSR threshold SR* = sqrt(2 ln
    n_trials / T) scales with log(n_trials) — log(8)/log(1) = +2.08
    bits of inflation → false RED downgrades on genuinely-fresh
    candidates in the same family.

    Semantically, a "trial" = a unique specification tested. Two
    hypotheses dispatching to the same spec_hash test the same hypothesis
    (the spec hash captures signal + universe + window + weighting +
    rebalance + bucket count). They're one trial, not n.

    The dedup is at `spec_hash` alone — not `(source_paper_id, spec_hash)`
    — because if two different papers happen to converge on identical
    specs, that's still one trial of one specification (just attributed
    to two source papers).

    Legacy rows without `spec_hash` (pre-2026-06-08 schema) fall back to
    row-level uniqueness via dispatch_event_id, preserving prior counts.
    """
    path = log_path or DEFAULT_DISPATCH_LOG
    if now is None:
        now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    window_start = now - _dt.timedelta(days=7)

    seen_specs_global: set[str]               = set()
    seen_specs_by_fam: dict[str, set[str]]    = {}
    duplicates_skipped: int                    = 0

    for row in _iter_log_rows(path):
        if not _row_is_cron_burndown(row):
            continue
        if not _row_is_successful_dispatch(row):
            continue
        row_ts = _parse_ts(row.get("ts", ""))
        if row_ts is None or row_ts < window_start:
            continue

        # Spec-hash dedup key. Legacy fallback: dispatch_event_id makes
        # each pre-schema row count as its own unique "spec" so the
        # change is non-regressive against historical data.
        spec_hash = row.get("spec_hash") or row.get("auto_test_spec_hash")
        if not spec_hash:
            spec_hash = f"__legacy:{row.get('dispatch_event_id') or id(row)}"

        if spec_hash in seen_specs_global:
            duplicates_skipped += 1
            continue
        seen_specs_global.add(spec_hash)

        fam = (row.get("family_hint") or "").upper()
        if fam:
            seen_specs_by_fam.setdefault(fam, set()).add(spec_hash)

    by_family = {fam: len(specs) for fam, specs in seen_specs_by_fam.items()}
    global_count = len(seen_specs_global)
    if duplicates_skipped:
        logger.info("burndown_caps: deduped %d same-spec_hash dispatches "
                       "in 7d window (Bailey-LdP n_trials hygiene)",
                       duplicates_skipped)

    return WeeklyUsage(
        global_count = global_count,
        by_family    = by_family,
        window_start = window_start.date().isoformat(),
        window_end   = now.date().isoformat(),
    )


# ── Capacity queries ───────────────────────────────────────────────


def family_capacity_left(family: str, usage: WeeklyUsage) -> int:
    """Slots remaining for a family this week. Returns FAMILY_WEEKLY_CAP if
    family is NOT in WATCHED_FAMILIES (only counts against global)."""
    fam = (family or "").upper()
    if fam not in WATCHED_FAMILIES:
        return FAMILY_WEEKLY_CAP  # informational; only global cap binds
    used = usage.by_family.get(fam, 0)
    return max(0, FAMILY_WEEKLY_CAP - used)


def global_capacity_left(usage: WeeklyUsage) -> int:
    """Slots remaining under the soft cap. Returns 0 once soft cap is hit
    even if hard cap has more room — soft cap is target throughput."""
    return max(0, WEEKLY_GLOBAL_SOFT_CAP - usage.global_count)


def global_hard_cap_breached(usage: WeeklyUsage) -> bool:
    """True iff combined cron throughput has exceeded the hard ceiling."""
    return usage.global_count >= WEEKLY_GLOBAL_HARD_CAP


def can_dispatch(family: str, usage: WeeklyUsage) -> tuple[bool, str]:
    """Composite check. Returns (ok, reason_if_not).

    The cron should call this BEFORE selecting a candidate of `family`.
    """
    if global_hard_cap_breached(usage):
        return False, f"GLOBAL_HARD_CAP_BREACHED ({usage.global_count}>={WEEKLY_GLOBAL_HARD_CAP})"
    if global_capacity_left(usage) <= 0:
        return False, f"GLOBAL_SOFT_CAP_HIT ({usage.global_count}/{WEEKLY_GLOBAL_SOFT_CAP})"
    if family_capacity_left(family, usage) <= 0:
        return False, f"FAMILY_CAP_HIT family={family} ({usage.by_family.get(family.upper(),0)}/{FAMILY_WEEKLY_CAP})"
    return True, ""


def usage_summary(usage: WeeklyUsage) -> str:
    """Human-readable one-liner for plan/digest output."""
    fam_str = ", ".join(
        f"{k}={v}" for k, v in sorted(usage.by_family.items()) if v > 0
    ) or "none"
    return (
        f"7d cron usage: global={usage.global_count}/{WEEKLY_GLOBAL_SOFT_CAP} "
        f"(hard {WEEKLY_GLOBAL_HARD_CAP}); families: {fam_str}"
    )
