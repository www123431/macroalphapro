"""engine/research/brainstorm/lesson_distiller.py — Layer 1 of the
brainstorm architecture (Phase 1, 2026-06-14).

DETERMINISTIC Python. NEVER an LLM call here. Per the locked design
([[project-brainstorm-architecture-2026-06-14]]):

    "Using LLM to distill lessons that then prime LLM brainstorm =
    same-model-bias self-reinforcement loop. Rules in Python → LLM
    only APPLIES lessons in Layer 3 brainstorm, doesn't GENERATE
    them in Layer 1."

Reads structured evidence from existing ledgers + emits 5-15 Lesson
rows summarizing what our system has empirically learned. Lessons
become priors for Layer 3 (multi-provider divergent generator) so
the brainstorm is conditioned on our actual experience, not just
LLM training corpus.

Inputs scanned:
  data/research/autopsies.jsonl
  data/research/decay_retest_results.jsonl
  data/research/pre_mortems.jsonl
  data/research/replication_checks.jsonl
  data/research/transfer_proposals.jsonl
  data/research_store/events.jsonl (factor_verdict_filed only)
  data/research/mechanism_library/*.yaml

Output:
  data/research/lessons_distilled.jsonl
  (rewritten on each run — append-only would let stale lessons survive
  after the underlying evidence flipped)

Rule catalog (deterministic, edit here to add):

  R1  DEAD_FAMILY            — family has ≥ N_RED RED autopsies + 0 GREEN
  R2  ROBUST_FAMILY          — family has ≥ N_GREEN GREEN verdicts, decay-resistant
  R3  RECURRING_FAILURE_MODE — pre-mortem same category fired ≥ N_REC times
  R4  LIT_DEAD_CATEGORY      — replication γ flagged ≥ N PROBABLY_DEAD in same family
  R5  CONFIRMED_DECAY_SLEEVE — sleeve flipped CONFIRMED_DECAY by Phase 9
  R6  HIGH_POTENTIAL_TRANSFER — β cross-domain proposal with conf ≥ 0.7 + exp_corr ≥ 0.5
  R7  CAPABILITY_GAP_EMPTY_FAMILY — family in EXPECTED list but 0 hypothesis
  R8  CAPABILITY_GAP_ASSET_CLASS — asset class catalog entry but 0 deployed sleeve

Each Lesson:
  - claim (1 sentence empirical statement)
  - kind  (R1-R8 enum)
  - family or asset_class (when applicable)
  - evidence_count
  - confidence (heuristic, derived from evidence_count)
  - source_event_ids (lineage)
  - emitted_ts
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

LESSONS_OUT_PATH = _REPO_ROOT / "data" / "research" / "lessons_distilled.jsonl"

# Ledger inputs
AUTOPSIES_PATH      = _REPO_ROOT / "data" / "research" / "autopsies.jsonl"
DECAY_RETEST_PATH   = _REPO_ROOT / "data" / "research" / "decay_retest_results.jsonl"
PRE_MORTEM_PATH     = _REPO_ROOT / "data" / "research" / "pre_mortems.jsonl"
REPLICATION_PATH    = _REPO_ROOT / "data" / "research" / "replication_checks.jsonl"
TRANSFER_PATH       = _REPO_ROOT / "data" / "research" / "transfer_proposals.jsonl"
EVENTS_PATH         = _REPO_ROOT / "data" / "research_store" / "events.jsonl"
HYP_PATH            = _REPO_ROOT / "data" / "research_store" / "hypotheses.jsonl"
LIBRARY_DIR         = _REPO_ROOT / "data" / "research" / "mechanism_library"

# Rule thresholds
N_RED_FOR_DEAD_FAMILY    = 3
N_GREEN_FOR_ROBUST       = 3
N_REC_FOR_RECURRING      = 3
N_REPL_DEAD              = 2
TRANSFER_HIGH_CONF       = 0.70
TRANSFER_ENHANCE_CORR    = 0.50

# R9 TEMPORAL_DECAY thresholds (last-N-days vs older bucket)
TEMPORAL_RECENT_DAYS     = 90
TEMPORAL_MIN_OBS_RECENT  = 2     # need ≥ this in recent bucket
TEMPORAL_MIN_OBS_OLDER   = 2     # need ≥ this in older bucket

# R12 ANCHOR_SPANNING_RECURRENCE — anchor R² threshold meaning "spanned"
ANCHOR_R2_SPANNED        = 0.85
ANCHOR_SPANNING_MIN_N    = 3

# R13 PUB_YEAR_DECAY buckets
PUB_YEAR_OLD_CUTOFF      = 2005   # pre/post split for "old" vs "recent" anomalies

# R14 N_TRIALS_NEAR_CAP heuristic. Bailey-LdP DSR threshold grows
# ~sqrt(N); at N ≥ 5 the per-additional-trial penalty becomes
# material (DSR threshold rises ~10% per added trial). Flag at 5+.
N_TRIALS_NEAR_CAP        = 5


# ─── Expected-families YAML loader (R7 — moved from hardcode) ────────


def _load_expected_families() -> tuple[list[str], list[str]]:
    """Return (deployed_or_seeded, aspirational_gap) family lists from
    the YAML capability map. Falls back to empty lists if YAML missing
    or malformed (rule silently no-ops rather than crashing the run)."""
    yp = LIBRARY_DIR / "_expected_families.yaml"
    if not yp.is_file():
        logger.warning("expected_families YAML missing at %s", yp)
        return [], []
    try:
        import yaml as _pyyaml
        d = _pyyaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        return (
            [str(x).upper() for x in (d.get("deployed_or_seeded") or [])],
            [str(x).upper() for x in (d.get("aspirational_gap") or [])],
        )
    except Exception:
        logger.exception("expected_families YAML parse failed")
        return [], []


@_dc.dataclass(frozen=True)
class Lesson:
    lesson_id:        str
    kind:             str          # R1..R8
    claim:            str          # 1 sentence
    family:           Optional[str]
    asset_class:      Optional[str]
    evidence_count:   int
    confidence:       float        # 0.0-0.99
    source_event_ids: tuple[str, ...]
    emitted_ts:       str


# ─── helpers ────────────────────────────────────────────────────────


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


def _conf_from_n(n: int, *, k: float = 5.0) -> float:
    """Naive heuristic kept for back-compat (R6/R7 use it). For
    Bernoulli-rate lessons (R1/R2/R4/R12) prefer _beta_binom_conf
    below, which factors in source independence + base rate."""
    return min(0.95, n / (n + k))


def _beta_binom_conf(n_successes: int, n_trials: int, *,
                     alpha_prior: float = 2.0,
                     beta_prior: float = 2.0,
                     target_rate: float = 0.7) -> float:
    """Posterior probability that success rate exceeds `target_rate`,
    under Beta(alpha_prior, beta_prior) prior. Weakly-informative
    Beta(2,2) prior means n=3 successes / 3 trials → conf ~ 0.66,
    n=10/10 → 0.95. Much more honest than n/(n+5) for binomial
    rate lessons.

    For "this family kills 100% of attempts" lessons, we ask:
      P(p_red > 0.7 | observed) where p_red is the family's red rate.
    """
    try:
        # Posterior is Beta(alpha+s, beta+f). P(p > target) = 1 - CDF.
        from scipy import stats as _stats
        post = _stats.beta(alpha_prior + n_successes,
                            beta_prior + (n_trials - n_successes))
        return float(min(0.99, max(0.0, 1.0 - post.cdf(target_rate))))
    except Exception:
        # Scipy unavailable — fall back to naive
        return _conf_from_n(n_successes)


def _dedup_autopsies_by_hypothesis(autopsies: list[dict]) -> list[dict]:
    """Source-independence dedup: 3 autopsies of variants of the SAME
    hypothesis are 1 data point, not 3. Keep the most recent row per
    hypothesis_id (treats sub-period dispatches as 1 effective trial)."""
    by_hyp: dict[str, dict] = {}
    for r in autopsies:
        hid = r.get("hypothesis_id") or r.get("subject_id") or ""
        if not hid:
            # No hyp_id → treat as unique unconditionally
            by_hyp[r.get("autopsy_id", str(id(r)))] = r
            continue
        prev = by_hyp.get(hid)
        if prev is None or (r.get("ts") or "") > (prev.get("ts") or ""):
            by_hyp[hid] = r
    return list(by_hyp.values())


def _new_lesson(kind: str, claim: str, *,
                family: Optional[str] = None,
                asset_class: Optional[str] = None,
                evidence_count: int = 1,
                source_event_ids: tuple[str, ...] = (),
                confidence_override: Optional[float] = None) -> Lesson:
    import uuid as _uuid
    conf = (confidence_override if confidence_override is not None
            else _conf_from_n(evidence_count))
    return Lesson(
        lesson_id=str(_uuid.uuid4()),
        kind=kind,
        claim=claim,
        family=family,
        asset_class=asset_class,
        evidence_count=evidence_count,
        confidence=round(conf, 3),
        source_event_ids=source_event_ids,
        emitted_ts=_dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# ─── R1, R2 — family verdict mix ─────────────────────────────────────


def _r1_r2_family_lessons() -> list[Lesson]:
    """Read autopsies → per-family G/M/R distribution after source-
    independence dedup (variants of same hypothesis count as 1).
    R1 DEAD_FAMILY: ≥ N_RED unique-hyp RED + 0 GREEN.
    R2 ROBUST_FAMILY: ≥ N_GREEN unique-hyp GREEN, red ≤ green.
    Confidence uses Beta-Binomial posterior, not n/(n+5)."""
    autopsies = [r for r in _iter_jsonl(AUTOPSIES_PATH)
                 if not r.get("superseded_by")]
    deduped = _dedup_autopsies_by_hypothesis(autopsies)

    fam_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"GREEN": 0, "MARGINAL": 0, "RED": 0, "ids": []})
    for r in deduped:
        fam = r.get("strategy_family") or ""
        v   = r.get("actual_verdict") or ""
        if not fam or v not in ("GREEN", "MARGINAL", "RED"):
            continue
        d = fam_counts[fam]
        d[v] += 1
        if r.get("autopsy_id"):
            d["ids"].append(r["autopsy_id"])

    out: list[Lesson] = []
    for fam, d in fam_counts.items():
        n_red = d["RED"]; n_green = d["GREEN"]; n_marg = d["MARGINAL"]
        n_total = n_red + n_green + n_marg
        if n_red >= N_RED_FOR_DEAD_FAMILY and n_green == 0:
            # Beta-Binomial: posterior P(red_rate > 0.7 | obs)
            conf = _beta_binom_conf(n_red, n_total, target_rate=0.70)
            out.append(_new_lesson(
                kind="R1_DEAD_FAMILY",
                family=fam,
                claim=(f"Family {fam} has {n_red}R/{n_marg}M/0G across "
                       f"{n_total} INDEPENDENT hypotheses (post source-dedup) "
                       f"— every variant we tested died. DO NOT propose more "
                       f"variants in this family without a structurally "
                       f"different mechanism."),
                evidence_count=n_total,
                source_event_ids=tuple(d["ids"][:5]),
                confidence_override=conf,
            ))
        elif n_green >= N_GREEN_FOR_ROBUST and n_red <= n_green:
            conf = _beta_binom_conf(n_green, n_total, target_rate=0.50)
            out.append(_new_lesson(
                kind="R2_ROBUST_FAMILY",
                family=fam,
                claim=(f"Family {fam} has {n_green}G/{n_marg}M/{n_red}R across "
                       f"{n_total} INDEPENDENT hypotheses — mechanism is "
                       f"empirically robust. Explore adjacent sub-mechanisms "
                       f"/ cross-asset transfers."),
                evidence_count=n_total,
                source_event_ids=tuple(d["ids"][:5]),
                confidence_override=conf,
            ))
    return out


# ─── R3 — recurring pre-mortem categories ────────────────────────────


def _r3_recurring_failure_modes() -> list[Lesson]:
    cat_counts: Counter = Counter()
    cat_ids: dict[str, list[str]] = defaultdict(list)
    for r in _iter_jsonl(PRE_MORTEM_PATH):
        fms = r.get("failure_modes") or []
        pmid = r.get("pre_mortem_id", "")
        for fm in fms:
            cat = fm.get("category") if isinstance(fm, dict) else None
            sev = fm.get("severity") if isinstance(fm, dict) else None
            if cat and sev in ("HIGH", "MEDIUM"):  # exclude LOW noise
                cat_counts[cat] += 1
                if pmid:
                    cat_ids[cat].append(pmid)
    out: list[Lesson] = []
    for cat, n in cat_counts.items():
        if n < N_REC_FOR_RECURRING:
            continue
        out.append(_new_lesson(
            kind="R3_RECURRING_FAILURE_MODE",
            claim=(f"Pre-mortem flagged {cat} as HIGH/MEDIUM failure mode "
                   f"{n} times across hypotheses. Any new idea must "
                   f"explicitly address this category."),
            evidence_count=n,
            source_event_ids=tuple(cat_ids[cat][:5]),
        ))
    return out


# ─── R4 — replication-catalog dead categories ────────────────────────


def _r4_lit_dead_categories() -> list[Lesson]:
    fam_dead_counts: Counter = Counter()
    fam_ids: dict[str, list[str]] = defaultdict(list)
    # Map hyp_id → family from hypotheses.jsonl for join
    hyp_fam: dict[str, str] = {}
    for h in _iter_jsonl(HYP_PATH):
        hyp_fam[h.get("hypothesis_id", "")] = (h.get("mechanism_family") or "").upper()
    for r in _iter_jsonl(REPLICATION_PATH):
        if r.get("replication_status") != "PROBABLY_DEAD":
            continue
        hyp_id = r.get("hypothesis_id", "")
        fam = hyp_fam.get(hyp_id, "")
        if not fam or fam == "OTHER":
            continue
        fam_dead_counts[fam] += 1
        if r.get("check_id"):
            fam_ids[fam].append(r["check_id"])
    out: list[Lesson] = []
    for fam, n in fam_dead_counts.items():
        if n < N_REPL_DEAD:
            continue
        out.append(_new_lesson(
            kind="R4_LIT_DEAD_CATEGORY",
            family=fam,
            claim=(f"Replication checker flagged {n} hypotheses in family {fam} "
                   f"as PROBABLY_DEAD per Hou-Xue-Zhang 2020 / McLean-Pontiff "
                   f"2016 catalogs. Lit-mainstream variants of this family "
                   f"are already-discounted."),
            evidence_count=n,
            source_event_ids=tuple(fam_ids[fam][:5]),
        ))
    return out


# ─── R5 — confirmed decay sleeves ────────────────────────────────────


def _r5_confirmed_decay_sleeves() -> list[Lesson]:
    out: list[Lesson] = []
    for r in _iter_jsonl(DECAY_RETEST_PATH):
        if r.get("verdict") != "CONFIRMED_DECAY":
            continue
        sleeve = r.get("sleeve_id", "")
        if not sleeve:
            continue
        out.append(_new_lesson(
            kind="R5_CONFIRMED_DECAY_SLEEVE",
            family=sleeve,    # sleeve as family proxy for join
            claim=(f"Sleeve {sleeve} has CONFIRMED_DECAY per Chow + bootstrap "
                   f"retest. Mechanism is structurally broken in current "
                   f"regime. Replacement candidates should consider WHY this "
                   f"sleeve died (regime shift / capacity / arbitraged away)."),
            evidence_count=1,
            source_event_ids=(r.get("retest_id", ""),),
            confidence_override=0.85,    # single high-quality signal
        ))
    return out


# ─── R6 — high-potential transfer proposals ──────────────────────────


def _r6_high_potential_transfers() -> list[Lesson]:
    out: list[Lesson] = []
    for r in _iter_jsonl(TRANSFER_PATH):
        conf = r.get("confidence")
        exp_corr = r.get("expected_correlation_with_source")
        try:
            conf = float(conf); exp_corr = float(exp_corr)
        except (TypeError, ValueError):
            continue
        if conf < TRANSFER_HIGH_CONF:
            continue
        target = r.get("target_asset_class", "")
        source_sleeve = r.get("source_sleeve_id", "")
        is_enhance = exp_corr >= TRANSFER_ENHANCE_CORR
        out.append(_new_lesson(
            kind="R6_HIGH_POTENTIAL_TRANSFER",
            asset_class=target,
            claim=(f"β proposed transfer of {source_sleeve} → {target} with "
                   f"confidence {conf:.2f} and exp_corr {exp_corr:.2f} "
                   f"({'ENHANCE-class' if is_enhance else 'NEW-FACTOR-class'}). "
                   f"Mechanism: {r.get('mechanism_carry', '')[:150]}"),
            evidence_count=1,
            source_event_ids=(r.get("proposal_id", ""),),
            confidence_override=conf,
        ))
    return out


# ─── R7 — capability-gap empty families ──────────────────────────────


def _r7_capability_gap_empty_families() -> list[Lesson]:
    """Read capability map from _expected_families.yaml. Families in
    `deployed_or_seeded` get HIGH confidence gap; families in
    `aspirational_gap` get LOWER confidence (data may not exist)."""
    deployed, aspirational = _load_expected_families()
    fam_hyp_count: Counter = Counter()
    for h in _iter_jsonl(HYP_PATH):
        fam = (h.get("mechanism_family") or "").upper()
        if fam:
            fam_hyp_count[fam] += 1
    out: list[Lesson] = []
    for fam in deployed:
        if fam_hyp_count.get(fam, 0) == 0:
            out.append(_new_lesson(
                kind="R7_CAPABILITY_GAP_EMPTY_FAMILY",
                family=fam,
                claim=(f"Family {fam} is in the DEPLOYED-or-SEEDED expected "
                       f"list but has 0 hypotheses. We test in this family "
                       f"actively; brainstorm should seed baseline ideas."),
                evidence_count=0,
                confidence_override=0.70,
            ))
    for fam in aspirational:
        if fam_hyp_count.get(fam, 0) == 0:
            out.append(_new_lesson(
                kind="R7_CAPABILITY_GAP_EMPTY_FAMILY",
                family=fam,
                claim=(f"Family {fam} is ASPIRATIONAL gap (capability we want "
                       f"but data may not be accessible). Brainstorm only "
                       f"after confirming data path."),
                evidence_count=0,
                confidence_override=0.35,
            ))
    return out


# ─── R8 — capability-gap asset-class (library coverage) ──────────────


def _r8_capability_gap_asset_classes() -> list[Lesson]:
    """Asset classes with mechanism_library YAML but 0 deployed sleeve.
    For now this is a stub — our 13 library entries are all deployed.
    Reserved for future expansion when library splits into deployed vs
    ideation-only buckets."""
    return []


# ─── R9 — temporal decay (recent regime shift) ───────────────────────


def _r9_temporal_decay_lessons() -> list[Lesson]:
    """For each family with both recent (last 90d) and older autopsies,
    flag if recent verdict mix is meaningfully WORSE than older mix.
    Real signal of regime change. Source-deduped first."""
    autopsies = [r for r in _iter_jsonl(AUTOPSIES_PATH)
                 if not r.get("superseded_by")]
    deduped = _dedup_autopsies_by_hypothesis(autopsies)
    cutoff = (_dt.datetime.utcnow() -
              _dt.timedelta(days=TEMPORAL_RECENT_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fam_buckets: dict[str, dict] = defaultdict(
        lambda: {"recent": [], "older": [], "ids": []})
    for r in deduped:
        fam = r.get("strategy_family") or ""
        v   = r.get("actual_verdict") or ""
        ts  = r.get("ts") or ""
        if not fam or v not in ("GREEN", "MARGINAL", "RED"):
            continue
        bucket = "recent" if ts >= cutoff else "older"
        fam_buckets[fam][bucket].append(v)
        if r.get("autopsy_id"):
            fam_buckets[fam]["ids"].append(r["autopsy_id"])

    out: list[Lesson] = []
    for fam, b in fam_buckets.items():
        n_rec, n_old = len(b["recent"]), len(b["older"])
        if n_rec < TEMPORAL_MIN_OBS_RECENT or n_old < TEMPORAL_MIN_OBS_OLDER:
            continue
        rec_red_rate = sum(1 for v in b["recent"] if v == "RED") / n_rec
        old_red_rate = sum(1 for v in b["older"] if v == "RED") / n_old
        rec_green_rate = sum(1 for v in b["recent"] if v == "GREEN") / n_rec
        old_green_rate = sum(1 for v in b["older"] if v == "GREEN") / n_old
        # Decay: recent RED rate ≥ 0.6 AND meaningfully higher than older
        if rec_red_rate >= 0.6 and rec_red_rate - old_red_rate >= 0.30:
            out.append(_new_lesson(
                kind="R9_TEMPORAL_DECAY",
                family=fam,
                claim=(f"Family {fam} in last {TEMPORAL_RECENT_DAYS}d: "
                       f"{int(rec_red_rate*100)}% RED ({n_rec} obs) vs "
                       f"{int(old_red_rate*100)}% historically ({n_old} obs). "
                       f"Regime shift signal — caution propagating older "
                       f"GREEN priors to current period."),
                evidence_count=n_rec + n_old,
                source_event_ids=tuple(b["ids"][:5]),
                confidence_override=_beta_binom_conf(
                    sum(1 for v in b["recent"] if v == "RED"), n_rec,
                    target_rate=old_red_rate + 0.20),
            ))
        # Recovery: recent GREEN rate jumped vs older
        elif rec_green_rate >= 0.5 and rec_green_rate - old_green_rate >= 0.30:
            out.append(_new_lesson(
                kind="R9_TEMPORAL_DECAY",   # same kind, recovery direction
                family=fam,
                claim=(f"Family {fam} in last {TEMPORAL_RECENT_DAYS}d: "
                       f"{int(rec_green_rate*100)}% GREEN ({n_rec} obs) vs "
                       f"{int(old_green_rate*100)}% historically ({n_old} obs)."
                       f" Possible regime recovery — re-test deprecated ideas."),
                evidence_count=n_rec + n_old,
                source_event_ids=tuple(b["ids"][:5]),
                confidence_override=_beta_binom_conf(
                    sum(1 for v in b["recent"] if v == "GREEN"), n_rec,
                    target_rate=old_green_rate + 0.20),
            ))
    return out


# ─── R10 — sub-period stability (regime-conditional alpha) ───────────


def _r10_sub_period_stability_lessons() -> list[Lesson]:
    """Verdict events' metrics.subsample_stability often contains a
    sub_period decomposition. When ≥ 3 verdicts in a family show
    OOM-LARGE drop in sub_periods (alpha post-2010 << pre-2010),
    that's a real regime-conditional pattern."""
    fam_unstable: dict[str, list[str]] = defaultdict(list)
    for e in _iter_jsonl(EVENTS_PATH):
        if e.get("event_type") != "factor_verdict_filed":
            continue
        m = e.get("metrics") or {}
        fam = (m.get("strategy_family") or e.get("family") or "").upper()
        sub = m.get("subsample_stability") or {}
        if not fam or not isinstance(sub, dict):
            continue
        # Heuristic: subsample_stability may carry "post_pub_window_killed":
        # true OR a worst-sub-period verdict that's RED while overall is GREEN
        flag = False
        if sub.get("post_pub_window_killed") is True:
            flag = True
        # Or a "worst_subperiod_verdict" / "sub_period_results" indicating
        # at least one regime gave a flipped verdict
        worst = sub.get("worst_subperiod_verdict") or sub.get("worst_verdict")
        if isinstance(worst, str) and worst in ("RED", "MARGINAL") and \
           str(e.get("verdict") or "").upper().endswith("GREEN"):
            flag = True
        if flag and e.get("event_id"):
            fam_unstable[fam].append(e["event_id"])

    out: list[Lesson] = []
    for fam, ids in fam_unstable.items():
        if len(ids) < 3:
            continue
        out.append(_new_lesson(
            kind="R10_SUB_PERIOD_STABILITY",
            family=fam,
            claim=(f"Family {fam}: {len(ids)} verdicts flagged sub-period "
                   f"instability (full-sample GREEN but ≥1 sub-period RED, "
                   f"OR post-pub-window kill). Strict-gate should reject "
                   f"GREEN that fails in a specific regime — alpha is "
                   f"regime-conditional, not robust."),
            evidence_count=len(ids),
            source_event_ids=tuple(ids[:5]),
            confidence_override=_beta_binom_conf(len(ids), len(ids) + 3,
                                                  target_rate=0.50),
        ))
    return out


# ─── R12 — anchor-spanning recurrence (FF5 already explains) ─────────


def _r12_anchor_spanning_lessons() -> list[Lesson]:
    """Verdict events' metrics.anchor_orthogonality.r2 measures how
    much FF5 + MOM explains the strategy. If a family's hypotheses
    consistently have r2 ≥ 0.85, that family is structurally spanned
    by mainstream factors — alpha claim is mostly factor exposure."""
    fam_r2_hits: dict[str, list[str]] = defaultdict(list)
    for e in _iter_jsonl(EVENTS_PATH):
        if e.get("event_type") != "factor_verdict_filed":
            continue
        m = e.get("metrics") or {}
        fam = (m.get("strategy_family") or e.get("family") or "").upper()
        ao = m.get("anchor_orthogonality") or {}
        if not fam or not isinstance(ao, dict):
            continue
        r2 = ao.get("r2")
        try:
            r2v = float(r2)
        except (TypeError, ValueError):
            continue
        if r2v >= ANCHOR_R2_SPANNED and e.get("event_id"):
            fam_r2_hits[fam].append(e["event_id"])

    out: list[Lesson] = []
    for fam, ids in fam_r2_hits.items():
        if len(ids) < ANCHOR_SPANNING_MIN_N:
            continue
        out.append(_new_lesson(
            kind="R12_ANCHOR_SPANNING_RECURRENCE",
            family=fam,
            claim=(f"Family {fam}: {len(ids)} verdicts had FF5+MOM r² ≥ "
                   f"{ANCHOR_R2_SPANNED} (anchor spanning). Family's alpha "
                   f"is mostly mainstream-factor exposure; novel variants "
                   f"must show LOW residual r² to claim independent alpha."),
            evidence_count=len(ids),
            source_event_ids=tuple(ids[:5]),
            confidence_override=_beta_binom_conf(len(ids), len(ids) + 2,
                                                  target_rate=0.50),
        ))
    return out


# ─── R13 — publication-year decay (our own McLean-Pontiff catalog) ───


def _r13_pub_year_decay_lessons() -> list[Lesson]:
    """Join hypothesis.source_paper_id → papers_registry.year → verdict
    GREEN/RED. If post-2005 papers systematically RED while pre-2005
    GREEN, that's our own empirical post-publication decay signal."""
    # Build paper_id → year map
    paper_year: dict[str, int] = {}
    pr_path = _REPO_ROOT / "data" / "research_store" / "papers_registry.jsonl"
    for r in _iter_jsonl(pr_path):
        pid = r.get("paper_id")
        y = r.get("year") or r.get("publication_year")
        if pid and y:
            try:
                paper_year[pid] = int(y)
            except (TypeError, ValueError):
                continue
    # Build hyp_id → paper_id map
    hyp_paper: dict[str, str] = {}
    for h in _iter_jsonl(HYP_PATH):
        hid = h.get("hypothesis_id")
        pid = h.get("source_paper_id")
        if hid and pid:
            hyp_paper[hid] = pid

    # Walk verdicts → join → bucket by paper year
    by_bucket: dict[str, dict] = {
        "pre_cutoff":  {"GREEN": 0, "RED": 0, "MARGINAL": 0, "ids": []},
        "post_cutoff": {"GREEN": 0, "RED": 0, "MARGINAL": 0, "ids": []},
    }
    for e in _iter_jsonl(EVENTS_PATH):
        if e.get("event_type") != "factor_verdict_filed":
            continue
        m = e.get("metrics") or {}
        hyp_id = m.get("source_hypothesis_id") or m.get("hypothesis_id")
        if not hyp_id:
            continue
        pid = hyp_paper.get(hyp_id)
        if not pid:
            continue
        year = paper_year.get(pid)
        if year is None:
            continue
        v = str(e.get("verdict") or "").upper().replace("VERDICT.", "")
        if v not in ("GREEN", "MARGINAL", "RED"):
            continue
        bucket = "pre_cutoff" if year < PUB_YEAR_OLD_CUTOFF else "post_cutoff"
        by_bucket[bucket][v] += 1
        if e.get("event_id"):
            by_bucket[bucket]["ids"].append(e["event_id"])

    pre = by_bucket["pre_cutoff"]; post = by_bucket["post_cutoff"]
    pre_n  = pre["GREEN"] + pre["RED"] + pre["MARGINAL"]
    post_n = post["GREEN"] + post["RED"] + post["MARGINAL"]
    out: list[Lesson] = []
    if pre_n >= 5 and post_n >= 5:
        pre_green_rate  = pre["GREEN"] / pre_n
        post_green_rate = post["GREEN"] / post_n
        if pre_green_rate - post_green_rate >= 0.25:
            out.append(_new_lesson(
                kind="R13_PUB_YEAR_DECAY",
                claim=(f"Verdicts on papers pre-{PUB_YEAR_OLD_CUTOFF}: "
                       f"{int(pre_green_rate*100)}% GREEN ({pre_n} obs); "
                       f"post-{PUB_YEAR_OLD_CUTOFF}: "
                       f"{int(post_green_rate*100)}% GREEN ({post_n} obs). "
                       f"Our own empirical post-pub decay signal — McLean-"
                       f"Pontiff 2016 replicated in our data."),
                evidence_count=pre_n + post_n,
                source_event_ids=tuple((pre["ids"] + post["ids"])[:5]),
                confidence_override=_beta_binom_conf(
                    pre["GREEN"], pre_n, target_rate=post_green_rate + 0.15),
            ))
    return out


# ─── R14 — n_trials near Bailey-LdP material penalty ────────────────


def _r14_n_trials_near_cap_lessons() -> list[Lesson]:
    """Per family, if Bailey-LdP n_trials count ≥ N_TRIALS_NEAR_CAP,
    DSR threshold has materially risen — adding more variants in this
    family is costly per the multi-test penalty."""
    try:
        from engine.research.family_trial_counter import count_trials_in_family
    except Exception:
        return []
    # Get list of families to check from autopsies (where we have data)
    seen_families: set[str] = set()
    for r in _iter_jsonl(AUTOPSIES_PATH):
        fam = r.get("strategy_family") or ""
        if fam:
            seen_families.add(fam)
    out: list[Lesson] = []
    for fam in seen_families:
        try:
            n = count_trials_in_family(fam)
        except Exception:
            continue
        if n >= N_TRIALS_NEAR_CAP:
            out.append(_new_lesson(
                kind="R14_N_TRIALS_NEAR_CAP",
                family=fam,
                claim=(f"Family {fam} Bailey-LdP n_trials = {n} (≥ "
                       f"{N_TRIALS_NEAR_CAP}). Each additional trial raises "
                       f"DSR threshold ~10%. Adding more variants here is "
                       f"expensive in expected-alpha-net-of-multi-test — "
                       f"prefer cross-family novelty."),
                evidence_count=n,
                confidence_override=0.85,    # n_trials is deterministic
            ))
    return out


# ─── Public API ──────────────────────────────────────────────────────


def distill_lessons() -> list[Lesson]:
    """Run all rules, return combined lesson list."""
    lessons: list[Lesson] = []
    for fn in (_r1_r2_family_lessons,
               _r3_recurring_failure_modes,
               _r4_lit_dead_categories,
               _r5_confirmed_decay_sleeves,
               _r6_high_potential_transfers,
               _r7_capability_gap_empty_families,
               _r8_capability_gap_asset_classes,
               # P0 audit additions (2026-06-14)
               _r9_temporal_decay_lessons,
               _r10_sub_period_stability_lessons,
               _r12_anchor_spanning_lessons,
               _r13_pub_year_decay_lessons,
               _r14_n_trials_near_cap_lessons):
        try:
            lessons.extend(fn())
        except Exception:
            logger.exception("lesson rule %s failed", fn.__name__)
    return lessons


def persist_lessons(lessons: list[Lesson]) -> None:
    """OVERWRITE lessons_distilled.jsonl (not append). Stale lessons
    shouldn't survive if their underlying evidence changes."""
    LESSONS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LESSONS_OUT_PATH.open("w", encoding="utf-8") as f:
        for L in lessons:
            f.write(json.dumps(_dc.asdict(L), ensure_ascii=False) + "\n")


def load_lessons() -> list[dict]:
    """Read latest distilled lessons (for API + brainstorm Layer 3)."""
    if not LESSONS_OUT_PATH.is_file():
        return []
    out: list[dict] = []
    for ln in LESSONS_OUT_PATH.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def render_for_prompt(lessons: list[dict], *, limit: int = 12) -> str:
    """Render lessons as plain-text for injection into Layer 3 prompts."""
    if not lessons:
        return "(no distilled lessons yet — system is in early evidence phase)"
    # Rank priority (lower = surfaces earlier in prompt context):
    #   R5  confirmed decay (single highest-signal event)
    #   R9  temporal decay (recent regime shift — load-bearing)
    #   R1  dead family
    #   R13 pub-year decay (our own MP catalog)
    #   R10 sub-period instability
    #   R12 anchor spanning recurrence
    #   R3  recurring failure modes
    #   R4  lit-catalog dead categories
    #   R14 n_trials near cap (cost signal)
    #   R2  robust family
    #   R6  high-potential transfer (positive signal)
    #   R7  capability gap
    #   R8  asset-class gap
    rank = {"R5": 0, "R9": 1, "R1": 2, "R13": 3, "R10": 4, "R12": 5,
            "R3": 6, "R4": 7, "R14": 8, "R2": 9, "R6": 10, "R7": 11, "R8": 12}
    sorted_l = sorted(lessons, key=lambda L: (
        rank.get(L["kind"].split("_")[0], 9),
        -L.get("evidence_count", 0),
    ))[:limit]
    lines = ["LESSONS FROM OUR OWN EXPERIENCE (distilled, deterministic)",
             "=" * 56]
    for L in sorted_l:
        ev = L.get("evidence_count", 0)
        conf = L.get("confidence", 0.0)
        kind = L.get("kind", "?").split("_")[0]
        lines.append(f"  [{kind}] (n={ev}, conf={conf:.2f}) {L['claim']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    lessons = distill_lessons()
    persist_lessons(lessons)
    by_kind: Counter = Counter(L.kind.split("_")[0] for L in lessons)
    print(f"[lesson_distiller] {len(lessons)} lessons distilled → "
          f"{dict(by_kind)}")
    print()
    print(render_for_prompt([_dc.asdict(L) for L in lessons]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
