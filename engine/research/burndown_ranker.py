"""engine.research.burndown_ranker — rank queued hypotheses for daily cron.

Scoring philosophy
==================
We have 234 hypotheses queued, more arriving via papers_curator. The cron
can only burn through ~15/week. Random selection is wrong (PROFITABILITY
floods); FIFO is wrong (oldest first lets stale cargo cult papers crowd
out fresh ideas); recency-only is wrong (newest gets all attention).

Three orthogonal factors are scored multiplicatively:

  1. **novelty_score** = 1 / (1 + family_trials)
     Bailey-LdP §3: each family already has accumulated trials. Lower-trial
     families get higher priority — pushes the cron toward less-tested
     ground. PROFITABILITY (n_trials=21) gets sigmoid-low; an unknown
     family gets ~1.0.

  2. **demand_score** = 1.0 + 0.5 * (1 if family is in capability_gaps demand ledger else 0)
     Cross-reference the demand ledger (flex-3). If principal has been
     asked for a signal in this family, that hypothesis jumps the queue.
     Capped at +50% to avoid stampede.

  3. **recency_score** = 1 / (1 + days_since_created / 30)
     Logistic decay over months. Today's hypothesis gets ~1.0; a hypothesis
     30 days old gets 0.5; one year old gets ~0.08. Forces churn —
     don't let the queue grow into permanent backlog.

Final = novelty × demand × recency. Tie-break by hypothesis_id for
determinism.

Filtering
=========
- review_state must be 'proposed' or 'ready_for_dispatch' (skip rejected /
  superseded / already-locked).
- hypothesis_id must NOT appear in factor_dispatch_log.jsonl (no duplicates).
- mechanism_family must be non-empty (caps need a family hint).
- Family capacity > 0 in `usage` snapshot.
- Global capacity > 0 in `usage` snapshot.

NOT done in this module (deferred)
==================================
- FactorSpec extraction (LLM cost). burn-1a only ranks hypothesis rows;
  burn-1b will extract specs at dispatch time.
- Dead-wall pre-filter. We don't know the template-cert status until we
  try to dispatch — TIER_3 dead-walls are caught at the dispatcher gate,
  logged to demand ledger, and skipped without consuming quota
  (per burndown_caps doctrine).
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
HYPOTHESES_PATH = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
DEFAULT_DISPATCH_LOG = _REPO_ROOT / "data" / "strengthener" / "factor_dispatch_log.jsonl"
DEFAULT_GAPS_PATH = _REPO_ROOT / "data" / "research" / "capability_gaps.jsonl"

# review states that are eligible for cron dispatch
ELIGIBLE_REVIEW_STATES = frozenset({
    "proposed",
    "ready_for_dispatch",
    # 'approved' is reserved for human-approved manual sessions
})

# Families whose hypotheses have a confirmed dispatch path through the
# existing strengthener template registry. Hypotheses tagged with any
# OTHER family stay in the queue for non-cron consumers (papers_curator
# meta-doctrine consumer, sleeve-improvement consumer, etc.).
#
# Why an allowlist (not a denylist):
#   The 2026-06-11 burndown_plan dry-run showed 71/234 hypotheses tagged
#   `OTHER` — Harvey-Liu-Zhu 2016 multiple-testing-threshold methodology
#   hypotheses, NOT factor hypotheses; ATTENTION / SENTIMENT / etc. are
#   thematic claim families with no template. Letting these into the cron
#   wastes quota AND clutters demand ledger with non-actionable TIER_3
#   refusals (no template will ever be built for "FDR threshold should
#   be 3.0σ" — it's a doctrine claim, not a backtest).
#
# To add a family: confirm a template handles it end-to-end (test a
# real dispatch first; if it errors with UNSUPPORTED_SIGNAL the family
# isn't ready). Then add it here.
# NOTE: family strings MUST match engine.research_store.red_lessons.
# mechanism_families.MechanismFamily enum (case-sensitive at the enum
# layer, upper-cased here for filter equality). Adding a family =
# (1) it exists in the canonical enum, AND (2) a template handles it.
DISPATCHABLE_FAMILIES = frozenset({
    # cross_sec_us_equities (9 grandfathered signals)
    "PROFITABILITY",
    "MOMENTUM",
    "VALUE",
    "LOW_VOL",
    "SIZE",
    "REVERSAL",
    "INVESTMENT",
    # multi-asset templates
    "CARRY",
    "CROSS_ASSET_MOMENTUM",
    # 2026-06-13: Phase 3.1 templates shipped today
    "VOL_RISK_PREMIUM",        # vrp_spx template (Carr-Wu 2009)
    "EARNINGS_DRIFT",          # event_drift_pead template (Bernard-Thomas 1989)
})

# Tag prefixes that identify meta-research / doctrine-signal hypotheses
# (D-employee derived, sleeve_fix_proposer authored). These wear a family
# tag (often the family they critique) but are NOT testable factor
# proposals — dispatching them to strict gate fails at signal_kind
# extraction. They have a different consumer (memory_amendment workflow
# in /approvals).
NON_FACTOR_TAG_PREFIXES = (
    "source:doctrine_signal",
    "source:active_b_sleeve_scan",   # Phase 1 (2026-06-11) enhance-class
)


@_dc.dataclass(frozen=True)
class RankedCandidate:
    """One hypothesis ranked for potential burndown."""
    hypothesis_id:      str
    family:             str
    claim_short:        str        # first 200 chars
    mechanism_subtype:  Optional[str]
    created_ts:         str
    age_days:           int
    source_paper_id:    Optional[str]
    novelty_score:      float
    demand_score:       float
    recency_score:      float
    rank_score:         float      # product of the three

    def to_dict(self) -> dict:
        return _dc.asdict(self)


# ── Loaders ────────────────────────────────────────────────────────


def _iter_jsonl(path: Path):
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("burndown_ranker: %s line %d malformed", path.name, ln_no)


def load_hypotheses(hyp_path: Optional[Path] = None) -> list[dict]:
    return list(_iter_jsonl(hyp_path or HYPOTHESES_PATH))


def load_dispatched_hypothesis_ids(log_path: Optional[Path] = None) -> set[str]:
    """All hypothesis_ids that have ever appeared in the dispatch log
    (regardless of refusal vs success). Cron should never re-dispatch."""
    path = log_path or DEFAULT_DISPATCH_LOG
    out: set[str] = set()
    for row in _iter_jsonl(path):
        hid = row.get("hypothesis_id")
        if hid:
            out.add(hid)
    return out


def load_demand_families(gaps_path: Optional[Path] = None) -> set[str]:
    """Families currently in the capability_gaps demand ledger.

    Reads the ledger flex-3 maintains; each row points at an unmet need
    that surfaced during refusal. We boost any hypothesis matching one
    of these families on the assumption that closing a demand-ledger gap
    has higher principal value than a random new test.
    """
    path = gaps_path or DEFAULT_GAPS_PATH
    fams: set[str] = set()
    for row in _iter_jsonl(path):
        fam = (row.get("family") or "").upper()
        if fam:
            fams.add(fam)
    return fams


# ── Scoring ────────────────────────────────────────────────────────


def _novelty_score(family: str) -> float:
    """1 / (1 + family_trials). PROFITABILITY ~ 0.045 if n=21;
    fresh family ~ 1.0."""
    try:
        from engine.research.family_trial_counter import count_trials_in_family
        n = count_trials_in_family(family.lower())
    except Exception:
        n = 0
    return 1.0 / (1.0 + float(n))


def _demand_score(family: str, demand_families: set[str]) -> float:
    return 1.5 if family.upper() in demand_families else 1.0


def _recency_score(created_ts: str, now: _dt.datetime) -> float:
    if not created_ts:
        return 0.5
    try:
        created = _dt.datetime.strptime(created_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc,
        )
    except ValueError:
        return 0.5
    age_days = (now - created).days
    if age_days < 0:
        age_days = 0
    return 1.0 / (1.0 + age_days / 30.0)


def _compute_age_days(created_ts: str, now: _dt.datetime) -> int:
    if not created_ts:
        return -1
    try:
        created = _dt.datetime.strptime(created_ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc,
        )
    except ValueError:
        return -1
    return max(0, (now - created).days)


# ── Public API ─────────────────────────────────────────────────────


def rank_candidates(
    *,
    top_k:              int = 20,
    hyp_path:           Optional[Path] = None,
    dispatch_log_path:  Optional[Path] = None,
    gaps_path:          Optional[Path] = None,
    now:                Optional[_dt.datetime] = None,
    usage:              Optional["object"] = None,    # WeeklyUsage if provided
) -> list[RankedCandidate]:
    """Rank the top_k eligible hypotheses.

    If `usage` (WeeklyUsage) is provided, families/global at capacity are
    filtered out — this is the cron's selection path. If `usage` is None,
    all eligible candidates are ranked regardless of cap (useful for
    inspection / debugging).
    """
    if now is None:
        now = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    hypotheses = load_hypotheses(hyp_path)
    dispatched = load_dispatched_hypothesis_ids(dispatch_log_path)
    demand_families = load_demand_families(gaps_path)

    # Cap accounting (optional). If usage is None we just rank without
    # capacity filter; if provided we filter and track per-family
    # remaining as we walk the sorted list.
    capacity_left_by_family: dict[str, int] = {}
    global_room: Optional[int] = None
    if usage is not None:
        from engine.research import burndown_caps
        if burndown_caps.global_hard_cap_breached(usage):
            return []
        global_room = burndown_caps.global_capacity_left(usage)
        capacity_left_by_family = {
            fam: burndown_caps.family_capacity_left(fam, usage)
            for fam in burndown_caps.WATCHED_FAMILIES
        }

    candidates: list[RankedCandidate] = []
    for h in hypotheses:
        hid = h.get("hypothesis_id")
        if not hid or hid in dispatched:
            continue
        if h.get("review_state") not in ELIGIBLE_REVIEW_STATES:
            continue
        family = (h.get("mechanism_family") or "").upper()
        if not family:
            continue
        if family not in DISPATCHABLE_FAMILIES:
            # Non-factor hypothesis (meta-research / sleeve-improvement /
            # thematic) — skip cron consumption but leave in the queue.
            continue
        # Skip doctrine-signal-derived meta claims even when their
        # family tag is in DISPATCHABLE_FAMILIES (these are critiques
        # of the family, not new factor proposals).
        hyp_tags = tuple(h.get("tags") or ())
        if any(any(t.startswith(p) for p in NON_FACTOR_TAG_PREFIXES) for t in hyp_tags):
            continue
        # Phase 1 (2026-06-11): addresses_decay_in non-null = enhance
        # class; forward cron must NOT touch these (different statistical
        # framework, see forward-vs-enhance-statistical-separation memo).
        if h.get("addresses_decay_in"):
            continue
        # burn-1b-followup (2026-06-11): hypothesis_type filter — only
        # FACTOR_PROPOSAL candidates pass. factor_analysis / methodology /
        # sleeve_improvement / unknown all skip (cf. 2026-06-11 first cron
        # run that burned $0.09 LLM on candidates the extractor refused).
        # If the field isn't on disk yet (pre-backfill), classify
        # lazily so the filter is correct from commit-time.
        h_type = h.get("hypothesis_type")
        if h_type is None or h_type == "unknown":
            from engine.research_store.hypothesis.classifier import classify_hypothesis_type
            h_type = classify_hypothesis_type(h)
        if h_type != "factor_proposal":
            continue

        novelty = _novelty_score(family)
        demand  = _demand_score(family, demand_families)
        recency = _recency_score(h.get("created_ts") or "", now)
        rank    = novelty * demand * recency

        candidates.append(RankedCandidate(
            hypothesis_id     = hid,
            family            = family,
            claim_short       = (h.get("claim") or "")[:200],
            mechanism_subtype = h.get("mechanism_subtype"),
            created_ts        = h.get("created_ts") or "",
            age_days          = _compute_age_days(h.get("created_ts") or "", now),
            source_paper_id   = h.get("source_paper_id"),
            novelty_score     = novelty,
            demand_score      = demand,
            recency_score     = recency,
            rank_score        = rank,
        ))

    # Sort descending by rank; deterministic tie-break by hypothesis_id
    candidates.sort(key=lambda c: (-c.rank_score, c.hypothesis_id))

    if usage is None:
        return candidates[:top_k]

    # Filter by family + global capacity, walking the sorted list
    selected: list[RankedCandidate] = []
    for c in candidates:
        if len(selected) >= top_k:
            break
        if global_room is not None and global_room <= 0:
            break
        fam_left = capacity_left_by_family.get(c.family, None)
        if fam_left is not None and fam_left <= 0:
            continue
        selected.append(c)
        if fam_left is not None:
            capacity_left_by_family[c.family] = fam_left - 1
        if global_room is not None:
            global_room -= 1
    return selected
