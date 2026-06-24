"""engine.agents.autopilot_live — F14b live runner (2026-06-05).

Picks up where F14a (dry-run preview) left off. Each invocation:

  1. Opens an auto-session (research_new) so emitted events carry the
     session tag and exit conditions are enforced.
  2. Resolves the top-1 candidate from compute_dry_run_plan(top_n=1).
  3. Runs compose() to materialize returns.
  4. Computes a strict-gate verdict (Sharpe IS/OOS, t-stat, deflated SR).
  5. Registers the subject if absent (subject_id = `auto_<spec_hash12>`).
  6. Writes a capability_evidence markdown to disk.
  7. Emits `capability_evidence_filed` (parent for the verdict).
  8. Emits `factor_verdict_filed` with parent = evidence event_id.
  9. Closes the session (exit conditions satisfied).

A+B hard line invariants — enforced by ABSENCE, not assertion:

  - This module does NOT import library.yaml writers / paper-trade
    allocators. There is literally no code path from here to a
    deployed sleeve.
  - PROMOTE_TO_PAPER_TRADE is a human decision, made manually after
    reviewing the verdict in `/research` UI.

Cost: $0 LLM per run (compose + verdict math are all offline). Wall
clock: ~60-90s per candidate.

Verdict rule (initial, conservative — will tune as we accumulate
ground-truth verdicts):

  Score = sum of 4 indicators:
    a) |IS t-stat|     >= 2.0
    b) IS Sharpe (ann) >= 0.5
    c) OOS Sharpe      >= 0.3      (last 30% holdout)
    d) Deflated SR     >= 0.85     (n_trials default = 20)

  Score == 4  → GREEN
  Score in 2-3 → MARGINAL
  Score in 0-1 → RED

n_trials is a placeholder constant; should later be derived per-family
from the catalog (specs tested + RED count). For now, 20 is a deliberate
mid-range value — too low (n=5) inflates DSR, too high (n=100) deflates
known winners. Per memory file
[[feedback-sharpe-se-for-strategy-comparison-2026-05-31]] we also expect
post-publication decay on factor library candidates — RED is the LIKELY
outcome and that's HONEST, not a bug.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_DIR  = _REPO_ROOT / "data" / "autopilot" / "_live"
_EVIDENCE_DIR = _REPO_ROOT / "docs" / "capability_evidence" / "autopilot"

_VERDICT_DSR_N_TRIALS = 20
_IS_OOS_SPLIT_FRAC    = 0.70


@dataclass(frozen=True)
class LiveRunResult:
    """Persisted artefact of one F14b live run."""
    ts:                  str
    session_id:          str
    subject_id:          str
    source_hypothesis_id: str
    spec_hash:           str
    family:              str
    signal_type:         str
    n_obs:               int
    is_sharpe:           float
    oos_sharpe:          float
    t_stat:              float
    deflated_sr:         float
    max_dd:              float
    verdict:             str           # GREEN | MARGINAL | RED (post-DA)
    score:               int           # post-DA score
    raw_verdict:         str = ""      # GREEN/MARGINAL/RED before DA downgrade
    raw_score:           int = 0       # score before DA downgrade
    da_fired:            bool = False
    da_tag:              str = "da_skipped"   # see apply_critique_to_verdict
    da_attack_vector:    str = ""
    da_severity:         str = ""
    da_confidence:       float = 0.0
    evidence_event_id:   str = ""
    verdict_event_id:    str = ""
    da_event_id:         str = ""      # council_critique event id, if DA fired
    capability_evidence_path: str = ""
    returns_parquet_path: str = ""
    elapsed_s:           float = 0.0
    # Phase 2.0 step 7 (2026-06-06): pre-compute DA gate fields.
    # pre_compute_skipped=True means the run short-circuited BEFORE
    # compose; verdict will be "SKIPPED" + all numeric metrics 0.
    pre_compute_da_fired:      bool   = False
    pre_compute_skipped:       bool   = False
    pre_compute_attack_vector: str    = ""
    pre_compute_confidence:    float  = 0.0
    pre_compute_event_id:      str    = ""


# ──────────────────────────────────────────────────────────────────────
# Verdict math
# ──────────────────────────────────────────────────────────────────────
def _is_oos_split(returns: pd.Series) -> tuple[pd.Series, pd.Series]:
    n = len(returns)
    split = int(n * _IS_OOS_SPLIT_FRAC)
    return returns.iloc[:split], returns.iloc[split:]


def _ann_factor_from_index(idx) -> int:
    """Heuristic: monthly index → 12, weekly → 52, daily → 252."""
    if len(idx) < 2:
        return 12
    deltas = pd.Series(idx[1:]) - pd.Series(idx[:-1])
    median_days = pd.Series(deltas).dt.days.median()
    if median_days < 5:
        return 252
    if median_days < 20:
        return 52
    return 12


def _sharpe(r: pd.Series, ann: int) -> float:
    r = r.dropna()
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * (ann ** 0.5))


def _t_stat(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.mean() / (r.std() / (len(r) ** 0.5)))


def _max_dd(r: pd.Series) -> float:
    cum = (1 + r.fillna(0)).cumprod()
    return float((cum / cum.cummax() - 1).min())


def compute_verdict_metrics(returns: pd.Series) -> dict:
    """Returns the full per-spec metrics dict that drives the verdict
    decision + capability evidence markdown. Pure math; no I/O."""
    from engine.validation.deflated_sharpe import deflated_sharpe_ratio
    ann = _ann_factor_from_index(returns.index)
    is_r, oos_r = _is_oos_split(returns)
    is_sharpe  = _sharpe(is_r, ann)
    oos_sharpe = _sharpe(oos_r, ann)
    t_stat     = _t_stat(returns)
    max_dd     = _max_dd(returns)
    try:
        dsr_res = deflated_sharpe_ratio(
            returns         = returns.dropna().tolist(),
            n_trials        = _VERDICT_DSR_N_TRIALS,
            periods_per_year= ann,
        )
        dsr = float(dsr_res.deflated_sr)
    except Exception as exc:
        logger.warning("DSR computation failed; defaulting to nan: %s", exc)
        dsr = float("nan")
    return {
        "n_obs":       int(len(returns)),
        "ann_factor":  ann,
        "is_sharpe":   is_sharpe,
        "oos_sharpe":  oos_sharpe,
        "t_stat":      t_stat,
        "deflated_sr": dsr,
        "max_dd":      max_dd,
    }


def decide_verdict(metrics: dict) -> tuple[str, int]:
    """Map metrics dict → (verdict_label, score). NaN counts as a fail
    on the corresponding indicator."""
    indicators = [
        (metrics["t_stat"],     lambda x: not np.isnan(x) and abs(x) >= 2.0),
        (metrics["is_sharpe"],  lambda x: not np.isnan(x) and x >= 0.5),
        (metrics["oos_sharpe"], lambda x: not np.isnan(x) and x >= 0.3),
        (metrics["deflated_sr"],lambda x: not np.isnan(x) and x >= 0.85),
    ]
    score = sum(1 for v, ok in indicators if ok(v))
    if score == 4:
        return "GREEN", score
    if score >= 2:
        return "MARGINAL", score
    return "RED", score


# ──────────────────────────────────────────────────────────────────────
# Evidence markdown
# ──────────────────────────────────────────────────────────────────────
def write_capability_evidence(
    *,
    subject_id:           str,
    spec,
    spec_hash_val:        str,
    metrics:              dict,
    verdict:              str,
    score:                int,
    returns_parquet_path: str,
    raw_verdict:          Optional[str] = None,
    raw_score:            Optional[int] = None,
    critique              = None,    # DACritique | None
    da_tag:               str = "da_skipped",
) -> Path:
    """Write the human-auditable evidence markdown for this verdict.
    Returns the file path (relative to repo root). MUST be on disk
    BEFORE emit.capability_evidence_filed (artifact-precondition)."""
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")
    out = _EVIDENCE_DIR / f"{today}_{subject_id}.md"
    primary = spec.legs[0] if spec.legs else None
    body = []
    body.append(f"# Autopilot capability evidence: `{subject_id}`")
    body.append("")
    body.append(f"- date: **{today}**  (F14b live run)")
    body.append(f"- verdict: **{verdict}**  (score {score}/4)")
    body.append(f"- spec_hash: `{spec_hash_val}`")
    body.append(f"- source hypothesis: `{spec.source_hypothesis_id}`")
    body.append(f"- family / signal: {spec.family.value} / "
                 f"{primary.signal_type.value if primary else '(none)'}")
    body.append(f"- universe: {spec.universe.asset_class.value}/"
                 f"{spec.universe.subset.value}")
    body.append(f"- weighting × rebalance: {spec.construction.weighting.value} × "
                 f"{spec.construction.rebalance.value}")
    body.append("")
    body.append("## Claim")
    body.append("")
    body.append(f"> {(spec.claim_text or '').strip()}")
    body.append("")
    body.append("## Returns metrics")
    body.append("")
    body.append("| metric | value | threshold | pass |")
    body.append("|---|---:|---:|:---:|")
    body.append(f"| n_obs            | {metrics['n_obs']}             | -       | -  |")
    body.append(f"| ann_factor       | {metrics['ann_factor']}        | -       | -  |")
    body.append(f"| IS t-stat        | {metrics['t_stat']:+.2f}      | ≥ 2.0 | "
                 f"{'Y' if abs(metrics['t_stat']) >= 2.0 else 'N'} |")
    body.append(f"| IS Sharpe (ann)  | {metrics['is_sharpe']:+.2f}   | ≥ 0.5 | "
                 f"{'Y' if metrics['is_sharpe'] >= 0.5 else 'N'} |")
    body.append(f"| OOS Sharpe       | {metrics['oos_sharpe']:+.2f}  | ≥ 0.3 | "
                 f"{'Y' if metrics['oos_sharpe'] >= 0.3 else 'N'} |")
    body.append(f"| Deflated SR      | {metrics['deflated_sr']:+.3f} | ≥ 0.85 | "
                 f"{'Y' if metrics['deflated_sr'] >= 0.85 else 'N'} |")
    body.append(f"| max DD           | {metrics['max_dd']*100:+.1f}% | -       | -  |")
    body.append("")
    body.append(f"DSR uses n_trials={_VERDICT_DSR_N_TRIALS} (autopilot placeholder; "
                 "tune per family later).")
    body.append("")
    body.append("## Decision")
    body.append("")
    if verdict == "GREEN":
        body.append("All 4 indicators pass. Candidate is a genuine alpha at this "
                     "evidence bar. Human-review required before PROMOTE_TO_PAPER_TRADE — "
                     "F14b does NOT auto-allocate.")
    elif verdict == "MARGINAL":
        body.append(f"{score}/4 indicators pass. Worth tracking but not strong enough "
                     "to promote. Consider re-test with longer window or stricter "
                     "universe before any deploy decision.")
    else:
        body.append(f"Only {score}/4 indicators pass. Most likely post-publication "
                     "decay (Sharpe collapses after factor enters textbook), or "
                     "implementation gap (e.g. construction differs from paper's). "
                     "No deploy. Add to gray-list; re-test if mechanism changes.")
    body.append("")
    # Devil's Advocate section (Phase 4)
    body.append("## Devil's Advocate review")
    body.append("")
    if critique is None and da_tag == "da_skipped":
        body.append("DA skipped (verdict was RED — no point challenging a kill, "
                     "or DA failed to return; treated as 'no critique').")
    elif critique is not None:
        body.append(f"- DA tag: `{da_tag}`")
        body.append(f"- refuted: **{critique.refuted}**  · severity: **{critique.severity}**  · confidence: {critique.confidence:.2f}")
        body.append(f"- attack: {critique.attack_vector}")
        body.append("")
        body.append(f"> {critique.reasoning}")
        if raw_verdict is not None and raw_verdict != verdict:
            body.append("")
            body.append(f"DA downgraded verdict: **{raw_verdict}** ({raw_score}/4) → "
                         f"**{verdict}** ({score}/4).")
    body.append("")

    body.append("## Returns artefact")
    body.append("")
    body.append(f"`{returns_parquet_path}`")
    body.append("")
    body.append("---")
    body.append("Auto-generated by `engine.agents.autopilot_live`. "
                 "DO NOT EDIT — re-run autopilot_live_run.py to refresh.")
    out.write_text("\n".join(body), encoding="utf-8")
    logger.info("capability evidence written: %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def _ensure_subject(spec) -> str:
    """Subject id convention: auto_<spec_hash[:12]>. Register lazily
    (idempotent) so emit.* preconditions pass."""
    from engine.research_store import registry
    from engine.research_store.schema import SubjectType
    from engine.hypothesis_spec.hash import spec_hash as _sh

    h = _sh(spec)
    sid = f"auto_{h[:12]}"
    existing = registry.resolve(sid)
    if existing is None:
        registry.register_subject(
            subject_id   = sid,
            subject_type = SubjectType.factor,
            family       = spec.family.value,
            description  = f"Autopilot F14b auto-test of hypothesis "
                            f"{spec.source_hypothesis_id[:8]} "
                            f"({spec.family.value}/{spec.legs[0].signal_type.value if spec.legs else 'NONE'})",
            created_by   = "engine.agents.autopilot_live",
        )
    return sid


def _count_recent_family_tests(family: str, *, days: int = 90) -> int:
    """Phase 2.0 step 7 helper: how many factor_verdict_filed events
    in this family within the recency window? Feeds pre-compute DA's
    n_trials budget signal. Failures return 0 (fail-quiet — the gate
    can still make a decent decision without this signal)."""
    try:
        from engine.research_store.store import filter_events
        import datetime as _dt
        cutoff = (_dt.datetime.utcnow()
                  - _dt.timedelta(days=days)
                  ).strftime("%Y-%m-%dT%H:%M:%SZ")
        evs = filter_events(
            event_type="factor_verdict_filed",
            family=family,
            since=cutoff,
        )
        return len(evs)
    except Exception as exc:
        logger.warning("recent_family_tests lookup failed for %s: %s", family, exc)
        return 0


def _lookup_paper_age_and_decay_target(spec) -> tuple[float | None, str | None]:
    """Phase 2.0 step 7 helper: look up the source paper's age (years
    since publication) and the hypothesis's addresses_decay_in field.
    Both feed pre-compute DA's post-pub-decay + targeted-decay signals.
    Failures return None — the gate decides without them."""
    paper_age = None
    addresses_decay = None
    try:
        from engine.research_store.hypothesis.store import find_by_id
        h = find_by_id(spec.source_hypothesis_id) if spec.source_hypothesis_id else None
        if h is not None:
            addresses_decay = h.addresses_decay_in
            if h.source_paper_id:
                try:
                    from engine.research_store.papers import load_registry, latest_per_doi
                    reg = latest_per_doi(load_registry())
                    paper = next((p for p in reg.values()
                                   if p.paper_id == h.source_paper_id), None)
                    if paper is not None and paper.published_ts:
                        import datetime as _dt
                        try:
                            pub = _dt.datetime.fromisoformat(
                                paper.published_ts.replace("Z", "")
                            )
                            paper_age = (_dt.datetime.utcnow() - pub).days / 365.25
                        except Exception:
                            pass
                except Exception as exc:
                    logger.debug("paper lookup failed: %s", exc)
    except Exception as exc:
        logger.warning("paper_age lookup failed: %s", exc)
    return paper_age, addresses_decay


def run_top1(*, force_compose: bool = False, force_da: bool = False,
              skip_pre_compute_da: bool = False) -> LiveRunResult:
    """Execute one F14b cycle: select top-1, compose, score, emit
    capability_evidence + factor_verdict, close session.

    force_compose:       bypasses compose cache (useful for re-runs).
    force_da:            fires Devil's Advocate even on RED verdicts
                          (smoke test only; production rule is DA only
                          on GREEN/MARGINAL).
    skip_pre_compute_da: bypasses the pre-compute DA gate (Phase 2.0
                          step 7). Use for force-running a candidate
                          the DA would normally veto. Production
                          default is False — pre-compute DA fires."""
    import time
    from engine.agents.autopilot import compute_dry_run_plan
    from engine.composer.composer import compose
    from engine.hypothesis_spec.store import all_specs
    from engine.hypothesis_spec.hash import spec_hash as _sh
    from engine.research_store import emit
    from engine.sessions.lifecycle import (
        open_session, record_preflight, close_session,
        SessionType, PreflightDigest,
    )

    t0 = time.perf_counter()
    today = _dt.datetime.utcnow().strftime("%Y-%m-%d")

    # 1. Select top-1 FIRST so preflight can name the exact family/signal
    #    that the F14a redundancy check ran against. PreflightDigest
    #    requires cockpit_reviewed=True + a non-empty graveyard query;
    #    for unattended cron runs the F14a substrate satisfies both:
    #      - cockpit equivalent = data/autopilot/<date>.md dry-run
    #      - graveyard search   = find_redundancy_for_spec on the cell
    #    Setting them True is honest because the substrate genuinely
    #    performed both checks programmatically.
    plan = compute_dry_run_plan(top_n=1)
    if not plan.decisions:
        # Open a session just to abandon it cleanly with the reason
        s = open_session(SessionType.research_new,
                          title=f"F14b abandoned {today}")
        from engine.sessions.lifecycle import abandon_session
        abandon_session(s.session_id, reason="no testable candidates in catalog")
        raise RuntimeError("F14b: no testable candidates available; abandoned session")
    top1 = plan.decisions[0]
    spec_candidates = [s for s in all_specs()
                        if s.source_hypothesis_id == top1.source_hypothesis_id]
    spec_candidates.sort(key=lambda s: s.extraction.extracted_ts or "", reverse=True)
    spec = spec_candidates[0]
    logger.info("F14b top-1: hyp=%s cell=%s/%s",
                 spec.source_hypothesis_id[:8], spec.family.value,
                 spec.legs[0].signal_type.value)

    # 2. Re-run redundancy on top-1 to populate the preflight digest
    #    with the actual query + hit count.
    from engine.research_store.mechanism_catalog import find_redundancy_for_spec
    redund_matches = find_redundancy_for_spec(spec.to_dict())
    gq = (f"family={spec.family.value} AND "
           f"signal_type={spec.legs[0].signal_type.value}")

    # 3. Open session + record preflight
    session = open_session(
        session_type = SessionType.research_new,
        title        = f"F14b autopilot live run {today}",
    )
    record_preflight(session.session_id, PreflightDigest(
        cockpit_reviewed       = True,
        decay_alerts_count     = 0,
        dq_breaches_count      = 0,
        graveyard_search_query = gq,
        graveyard_hits_count   = len(redund_matches),
        library_overlap_checked= True,
        goal                   = (f"F14b autopilot top-1 auto-test: "
                                   f"{spec.family.value}/"
                                   f"{spec.legs[0].signal_type.value} "
                                   f"(hyp {spec.source_hypothesis_id[:8]})"),
        notes                  = ("Unattended F14b cron. Cockpit equivalent = "
                                   f"F14a dry-run plan at data/autopilot/{today}.md. "
                                   f"Graveyard equivalent = mechanism_catalog."
                                   f"find_redundancy_for_spec returned "
                                   f"{len(redund_matches)} match(es)."),
    ))
    logger.info("F14b session opened: %s", session.session_id)

    # 3. Subject registration (must succeed before emit)
    subject_id = _ensure_subject(spec)

    # 3.5. Pre-compute DA gate (Phase 2.0 step 7, 2026-06-06)
    #      Cheap-veto opportunity BEFORE we spend ~$0.005 + 60-90s on
    #      compose. Fail-OPEN: if DA fails, proceed. If DA gates
    #      (worth_running=False): emit candidate_skipped_pre_compute,
    #      abandon session, return early with a "skipped" LiveRunResult.
    from engine.agents.autopilot_pre_compute_da import decide_pre_compute_gate
    paper_age, addresses_decay = _lookup_paper_age_and_decay_target(spec)
    family_count = _count_recent_family_tests(spec.family.value)
    proceed, pc_verdict = decide_pre_compute_gate(
        spec                      = spec,
        claim_text                = spec.claim_text or "",
        graveyard_matches         = list(redund_matches),
        family_recent_test_count  = family_count,
        paper_age_years           = paper_age,
        addresses_decay_in        = addresses_decay,
        skip                      = skip_pre_compute_da,
    )
    if not proceed:
        # pc_verdict is guaranteed non-None when proceed=False per
        # decide_pre_compute_gate's contract
        from engine.research_store import emit
        from engine.sessions.lifecycle import abandon_session
        skip_event_id = emit.candidate_skipped_pre_compute(
            subject_id           = subject_id,
            spec_hash            = _sh(spec),
            source_hypothesis_id = spec.source_hypothesis_id or "",
            attack_vector        = pc_verdict.attack_vector,
            reasoning            = pc_verdict.reasoning,
            confidence           = pc_verdict.confidence,
            family               = spec.family.value,
        )
        abandon_session(session.session_id,
                         reason=f"pre_compute_da: {pc_verdict.attack_vector[:120]}")
        elapsed = time.perf_counter() - t0
        logger.info("F14b skipped by pre_compute DA: %s (event %s, %.1fs)",
                     pc_verdict.attack_vector[:80], skip_event_id, elapsed)
        return LiveRunResult(
            ts                    = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            session_id            = session.session_id,
            subject_id            = subject_id,
            source_hypothesis_id  = spec.source_hypothesis_id or "",
            spec_hash             = _sh(spec),
            family                = spec.family.value,
            signal_type           = spec.legs[0].signal_type.value if spec.legs else "",
            n_obs                 = 0,
            is_sharpe             = 0.0, oos_sharpe = 0.0,
            t_stat                = 0.0, deflated_sr = 0.0,
            max_dd                = 0.0,
            verdict               = "SKIPPED",
            score                 = 0,
            raw_verdict           = "SKIPPED",
            raw_score             = 0,
            pre_compute_da_fired         = True,
            pre_compute_skipped          = True,
            pre_compute_attack_vector    = pc_verdict.attack_vector,
            pre_compute_confidence       = pc_verdict.confidence,
            pre_compute_event_id         = skip_event_id,
            elapsed_s             = elapsed,
        )

    # 4. Compose
    compose_result = compose(spec, force=force_compose)
    if not compose_result["ok"]:
        from engine.sessions.lifecycle import abandon_session
        abandon_session(session.session_id,
                         reason=f"compose failed: {compose_result.get('error')}")
        raise RuntimeError(f"F14b compose failed: {compose_result.get('error')}")
    returns = pd.read_parquet(compose_result["path"]).iloc[:, 0].dropna()
    if returns.empty:
        from engine.sessions.lifecycle import abandon_session
        abandon_session(session.session_id, reason="empty returns series")
        raise RuntimeError("F14b: returns series empty post-compose")

    # 5. Metrics + raw verdict
    metrics = compute_verdict_metrics(returns)
    raw_verdict, raw_score = decide_verdict(metrics)
    logger.info("F14b raw verdict: %s (score %d/4)", raw_verdict, raw_score)

    # 6. Devil's Advocate (Phase 4, 2026-06-05) — fires ONLY on positive
    # verdicts (GREEN / MARGINAL). RED skips DA (no point challenging a
    # kill). DA may downgrade the verdict per fixed rule; result also
    # surfaces in the evidence markdown for human review.
    from engine.agents.autopilot_devils_advocate import (
        run_autopilot_da, apply_critique_to_verdict,
    )
    critique = None
    if raw_verdict in ("GREEN", "MARGINAL") or force_da:
        critique = run_autopilot_da(
            spec       = spec,
            metrics    = metrics,
            verdict    = raw_verdict,
            score      = raw_score,
            claim_text = spec.claim_text or "",
        )
        if critique is not None:
            logger.info("F14b DA: refuted=%s severity=%s attack=%s",
                         critique.refuted, critique.severity,
                         critique.attack_vector[:80])
    verdict, score, da_tag = apply_critique_to_verdict(raw_verdict, raw_score, critique)
    if verdict != raw_verdict:
        logger.info("F14b DA downgrade: %s/%d -> %s/%d (%s)",
                     raw_verdict, raw_score, verdict, score, da_tag)

    # 7. Evidence markdown (artifact precondition for emit). Includes DA
    # section so the human-auditable file carries both sides.
    spec_hash_val = _sh(spec)
    evidence_path = write_capability_evidence(
        subject_id           = subject_id,
        spec                 = spec,
        spec_hash_val        = spec_hash_val,
        metrics              = metrics,
        verdict              = verdict,
        score                = score,
        returns_parquet_path = compose_result["path"],
        raw_verdict          = raw_verdict,
        raw_score            = raw_score,
        critique             = critique,
        da_tag               = da_tag,
    )

    # 7. Emit capability_evidence_filed
    rel_evidence = str(evidence_path.relative_to(_REPO_ROOT))
    rel_returns  = str(Path(compose_result["path"]).relative_to(_REPO_ROOT)) \
                    if Path(compose_result["path"]).is_absolute() \
                    else compose_result["path"]
    evidence_event_id = emit.capability_evidence_filed(
        subject_id       = subject_id,
        verdict          = verdict,
        artifacts        = {
            "capability_evidence": rel_evidence,
            "returns_parquet":     rel_returns,
        },
        summary          = f"Autopilot F14b live run on {today}; verdict={verdict} "
                            f"(score {score}/4, IS Sharpe={metrics['is_sharpe']:+.2f}, "
                            f"OOS Sharpe={metrics['oos_sharpe']:+.2f}, DSR={metrics['deflated_sr']:.2f})",
        metrics          = {k: (float(v) if not isinstance(v, int) else v)
                            for k, v in metrics.items()},
        family           = spec.family.value,
        # NB: emit.* auto-attaches session: and session_type: tags from
        # the active session — only add the autopilot tag here.
        tags             = ("autopilot_f14b",),
        actor            = "engine.agents.autopilot_live",
    )

    # 8. Emit factor_verdict_filed with parent (FINAL verdict post-DA)
    verdict_tags = ["autopilot_f14b", da_tag]
    if da_tag != "da_skipped" and verdict != raw_verdict:
        verdict_tags.append(f"raw_verdict:{raw_verdict}")
    verdict_event_id = emit.factor_verdict(
        subject_id       = subject_id,
        verdict          = verdict,
        artifacts        = {"capability_evidence": rel_evidence},
        summary          = f"Autopilot F14b verdict={verdict} (score {score}/4) "
                            f"for {spec.family.value}/{spec.legs[0].signal_type.value} "
                            f"hyp={spec.source_hypothesis_id[:8]} "
                            f"{'[DA-downgraded from ' + raw_verdict + ']' if verdict != raw_verdict else ''}",
        metrics          = {k: (float(v) if not isinstance(v, int) else v)
                            for k, v in metrics.items()},
        parent_event_ids = (evidence_event_id,),
        family           = spec.family.value,
        tags             = tuple(verdict_tags),
        actor            = "engine.agents.autopilot_live",
    )

    # 8b. Emit council_critique if DA fired (parent = verdict event)
    da_event_id = ""
    if critique is not None:
        da_event_id = emit.council_critique(
            subject_id       = subject_id,
            verdict          = "RED" if critique.refuted else "GREEN",
            metrics          = {
                "severity_label": critique.severity,
                "confidence":     critique.confidence,
                "score_after_da": score,
            },
            artifacts        = {"capability_evidence": rel_evidence},
            summary          = (f"DA {('refuted' if critique.refuted else 'confirmed')} "
                                 f"({critique.severity}): {critique.attack_vector[:200]}"),
            parent_event_ids = (verdict_event_id,),
            family           = spec.family.value,
            tags             = ("autopilot_f14b", da_tag),
            actor            = "engine.agents.autopilot_devils_advocate",
        )

    # 9. Close session (exit conditions: >= 1 capability + >= 1 verdict)
    close_session(session.session_id)

    elapsed = time.perf_counter() - t0
    result = LiveRunResult(
        ts                       = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        session_id               = session.session_id,
        subject_id               = subject_id,
        source_hypothesis_id     = spec.source_hypothesis_id,
        spec_hash                = spec_hash_val,
        family                   = spec.family.value,
        signal_type              = spec.legs[0].signal_type.value,
        n_obs                    = metrics["n_obs"],
        is_sharpe                = metrics["is_sharpe"],
        oos_sharpe               = metrics["oos_sharpe"],
        t_stat                   = metrics["t_stat"],
        deflated_sr              = metrics["deflated_sr"],
        max_dd                   = metrics["max_dd"],
        verdict                  = verdict,
        score                    = score,
        raw_verdict              = raw_verdict,
        raw_score                = raw_score,
        da_fired                 = critique is not None,
        da_tag                   = da_tag,
        da_attack_vector         = critique.attack_vector if critique else "",
        da_severity              = critique.severity if critique else "",
        da_confidence            = critique.confidence if critique else 0.0,
        evidence_event_id        = evidence_event_id,
        verdict_event_id         = verdict_event_id,
        da_event_id              = da_event_id,
        capability_evidence_path = rel_evidence,
        returns_parquet_path     = rel_returns,
        elapsed_s                = round(elapsed, 1),
    )

    # 10. Persist run log
    _LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LIVE_DIR / f"{today}.json"
    log_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    logger.info("F14b live run complete in %.1fs: verdict=%s log=%s",
                 elapsed, verdict, log_path)
    return result
