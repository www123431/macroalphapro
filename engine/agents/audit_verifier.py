"""engine.agents.audit_verifier — closed-loop lineage verifier.

First production-grade agentic loop in the codebase (2026-06-04 build):
EventBus subscriber that auto-verifies the lineage of every
`factor_verdict_filed` event the instant it lands, and emits a typed
verdict back onto the bus.

Why this exists
---------------
Pre-doctrine, lineage was checked by reviewer (the user or me, Claude)
manually. That's the same failure mode "graveyard discipline" was
designed to prevent: human-in-the-loop verification doesn't scale and
breaks when the reviewer is tired or rushed. This subscriber gives the
project its first piece of AGENTIC infrastructure that actually RUNS
(not just BUILT) — fires on every emit, gates on parent_event_ids +
artifacts + paired evidence, surfaces issues immediately.

Design notes
------------
* PURE DETERMINISTIC. No LLM call. Cheap, reproducible, no API spend.
  An LLM narration layer can come later if anomalies need explanation;
  the verification primitives themselves are checks any reviewer would
  run by hand.
* SUBSCRIBES at module import. Importing `engine.agents.audit_verifier`
  is the side-effect that wires the bus. API startup imports it.
* WRITES to data/audit_verifier/lineage_results.jsonl (append-only).
  Mirrors the research store pattern — events are immutable, downstream
  consumers query the file.
* SIDE-CHANNEL emit only. The verifier deliberately does NOT call back
  into engine.research_store.emit.* — that would publish to the bus,
  which would re-trigger the subscriber, which would loop. Lineage
  results are first-class but live in their own log.

Checks performed
----------------
For event_type == factor_verdict_filed:

  C1  parent_event_ids referenced by event MUST resolve in the store
  C2  artifacts['evidence_doc'] (or any *.md artifact) MUST exist on
      disk AT THE TIME OF VERIFICATION (re-check; emit already checks
      but a race could theoretically delete it)
  C3  if verdict in {GREEN, MARGINAL}, MUST have a paired
      capability_evidence_filed event within +/- 60s, family-matching,
      with this verdict event in its parent_event_ids
  C4  if metrics contains 'n_trials', MUST be reasonable for the
      family (within 1..1000)

Verdict shape:
  CLEAN — all checks PASS
  WARN  — at least one check soft-fails (e.g. C3 paired evidence
          missing but verdict was RED — RED doesn't always need evidence
          per current doctrine)
  FAIL  — at least one hard-fail (parent_event_ids invalid, artifact
          missing on re-check)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import datetime as _dt
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REPO_ROOT       = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR     = _REPO_ROOT / "data" / "audit_verifier"
_RESULTS_FILE    = _RESULTS_DIR / "lineage_results.jsonl"
_WRITE_LOCK      = threading.Lock()

_PAIRED_EVIDENCE_WINDOW_SEC = 60
_REASONABLE_N_TRIALS        = range(1, 1001)

# T2.1 (2026-06-05 audit V1 fix): minimum fields a ResearchEvent must
# carry for C1-C5 to even be meaningful. C0 short-circuits if any are
# missing so the per-check exceptions collapse into ONE clean FAIL row
# saying exactly what's missing, instead of 5 cryptic AttributeErrors.
_REQUIRED_EVENT_FIELDS = (
    "event_id", "event_type", "ts", "subject_type", "subject_id",
    "verdict", "metrics", "artifacts", "parent_event_ids",
)

# T2.1 (2026-06-05 audit V4 fix): legitimate parent event types per child
# type. A factor_verdict_filed event whose parent is e.g. a dq_breach
# event is a wiring bug — verdict shouldn't depend on a data-quality
# alert that way, the verdict should reference its capability_evidence
# and prior verdicts. Curate the allowed set per child type below.
_ALLOWED_PARENT_TYPES_BY_CHILD = {
    "factor_verdict_filed": frozenset({
        "capability_evidence_filed",   # the evidence doc + parquet this verdict cites
        "factor_verdict_filed",        # prior verdict on the same factor (re-test / amend)
        "memory_doctrine_locked",      # a doctrine that gated this verdict
        "spec_amended",                # spec amendment that triggered the re-test
        "council_critique",            # critique that triggered this verdict
    }),
}


# ── Utilities ──────────────────────────────────────────────────────


def _utc_iso() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(ts: str) -> Optional[_dt.datetime]:
    try:
        s = ts.rstrip("Z")
        return _dt.datetime.fromisoformat(s)
    except Exception:
        return None


def _artifact_exists(path: str) -> bool:
    if not path:
        return False
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p.exists()


def _append_result(row: dict) -> None:
    """Atomic-ish append. The lineage_results.jsonl is read-only for
    downstream consumers; only this function writes to it."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _WRITE_LOCK:
        with _RESULTS_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── Core verifier ──────────────────────────────────────────────────


def verify_factor_verdict_lineage(agent_event) -> dict:
    """EventBus handler for event_type=factor_verdict_filed.

    Pulls the underlying ResearchEvent via the store, runs C1-C4, writes
    a lineage_results.jsonl row, returns the result dict (for testing).
    Does NOT raise — handler exceptions would just be logged by the bus
    and dropped, but we want every emit to leave a verification trail.
    """
    payload = getattr(agent_event, "payload", {}) or {}
    research_event_id = payload.get("research_event_id")
    if not research_event_id:
        logger.warning("audit_verifier: agent_event lacks research_event_id; skip")
        return _record_skip(payload, "no_research_event_id")

    # Re-load the canonical event from the jsonl store. This is the
    # truth — payload on the bus is a denormalized copy.
    try:
        from engine.research_store import store
        event = store.by_event_id(research_event_id)
    except Exception as exc:
        logger.exception("audit_verifier: store reload failed")
        return _record_skip(payload, f"store_reload_error:{exc}")

    if event is None:
        return _record_skip(payload, "research_event_not_found_in_store")

    # T2.1 (2026-06-05 audit V1 fix): C0 schema pre-flight. If event
    # is mangled (missing required fields), C1-C5 would each raise
    # noisily inside _safe_check producing 5 cryptic AttributeError
    # rows. C0 collapses that into ONE clean FAIL ("schema_invalid:
    # missing 'verdict'") and short-circuits the downstream checks.
    c0 = _safe_check("C0_schema_valid", _check_schema, event)
    if c0["status"] == "FAIL":
        row = {
            "audit_id":          f"av_{research_event_id[:8]}_{int(_dt.datetime.utcnow().timestamp())}",
            "verified_ts":       _utc_iso(),
            "research_event_id": research_event_id,
            "subject_id":        getattr(event, "subject_id", None),
            "family":            getattr(event, "family", None),
            "verdict":           "FAIL",
            "checks":            [c0],
            "verifier":          "engine.agents.audit_verifier",
            "verifier_version":  3,
            "short_circuit":     "C0_schema_invalid",
        }
        _append_result(row)
        logger.warning("audit_verifier: %s C0 short-circuited: %s",
                       research_event_id[:8], c0.get("detail", ""))
        return row

    # T1.4 (2026-06-05 audit V2 fix): every check runs inside _safe_check
    # so one raising check cannot crash the whole verifier. Pre-T1.4,
    # an exception in any of C1-C5 propagated up — EventBus swallowed it
    # silently and NO audit row was written, making the verdict appear
    # "verified" when it actually wasn't checked at all. With this fix:
    #   - any check raising emits a FAIL row with detail='exception:...'
    #   - other checks still run
    #   - aggregate verdict + ledger row always emit
    checks = [
        c0,
        _safe_check("C1_parents_resolve",   _check_parents_resolve, store, event),
        _safe_check("C2_artifacts_exist",   _check_artifacts_exist, event),
        _safe_check("C3_paired_evidence",   _check_paired_evidence, store, event),
        _safe_check("C4_n_trials_reasonable", _check_n_trials,        event),
        _safe_check("C5_spec_grounded",     _check_spec_grounded,   event),
    ]

    # Aggregate verdict
    hard_fails = [c for c in checks if c["status"] == "FAIL"]
    warns      = [c for c in checks if c["status"] == "WARN"]
    if hard_fails:
        verdict = "FAIL"
    elif warns:
        verdict = "WARN"
    else:
        verdict = "CLEAN"

    row = {
        "audit_id":          f"av_{research_event_id[:8]}_{int(_dt.datetime.utcnow().timestamp())}",
        "verified_ts":       _utc_iso(),
        "research_event_id": research_event_id,
        "subject_id":        event.subject_id,
        "family":            event.family,
        "verdict":           verdict,
        "checks":            checks,
        "verifier":          "engine.agents.audit_verifier",
        "verifier_version":  3,    # T2.1 (2026-06-05): added C0 schema check + C1 parent-type validation
    }
    _append_result(row)
    logger.info("audit_verifier: %s -> %s (%d checks)",
                research_event_id[:8], verdict, len(checks))
    return row


def _safe_check(check_name: str, fn, *args, **kwargs) -> dict:
    """Run a single C1-C5 check with bounded blast radius.

    Per audit V2 (2026-06-05): individual checks previously could raise
    unhandled, crashing verify_factor_verdict_lineage; the EventBus
    swallowed the exception and NO audit row was written. This wrapper
    converts any raise into a structured FAIL row so the audit ledger
    always reflects what was attempted.

    The wrapper trusts the check function's own dict shape on success.
    On exception:
      status = "FAIL"
      detail = "exception in check: <short repr>"
    """
    try:
        result = fn(*args, **kwargs)
        if not isinstance(result, dict) or "status" not in result:
            return {"check": check_name, "status": "FAIL",
                    "detail": f"check returned malformed result: {type(result).__name__}"}
        # Ensure 'check' field is present (defensive — most check fns set it)
        result.setdefault("check", check_name)
        return result
    except Exception as exc:
        logger.exception("audit_verifier: %s raised", check_name)
        return {"check": check_name, "status": "FAIL",
                "detail": f"exception in check: {type(exc).__name__}: {str(exc)[:200]}"}


def _record_skip(payload: dict, reason: str) -> dict:
    row = {
        "audit_id":          f"av_skip_{int(_dt.datetime.utcnow().timestamp())}",
        "verified_ts":       _utc_iso(),
        "research_event_id": payload.get("research_event_id"),
        "verdict":           "SKIP",
        "reason":            reason,
        "verifier":          "engine.agents.audit_verifier",
        "verifier_version":  1,
    }
    try:
        _append_result(row)
    except Exception:
        logger.exception("audit_verifier: failed to write skip row")
    return row


# ── Individual checks ──────────────────────────────────────────────


def _check_schema(event) -> dict:
    """C0 (2026-06-05 audit V1 fix): pre-flight schema validation.

    Confirms the ResearchEvent loaded from the store has the minimum
    fields required for C1-C5 to make sense. If any are missing or
    None, we emit ONE clean FAIL ("schema_invalid: missing 'verdict'")
    and skip C1-C5 entirely (they'd just raise AttributeErrors).

    Why not catch at emit time? The store DOES validate at emit, but
    a corrupted jsonl row (manual edit, partial write during crash,
    schema-version drift) can still produce a malformed event when
    re-loaded. C0 is the verifier-side guard.
    """
    missing = []
    for f in _REQUIRED_EVENT_FIELDS:
        v = getattr(event, f, None)
        if v is None:
            missing.append(f)
            continue
        # Enum fields: also confirm .value is non-empty
        if f in ("event_type", "subject_type", "verdict"):
            try:
                _ = v.value
            except AttributeError:
                missing.append(f"{f}(not_enum)")
        # String fields: must be non-empty
        elif f in ("event_id", "subject_id", "ts"):
            if not isinstance(v, str) or not v.strip():
                missing.append(f"{f}(empty)")
    if missing:
        return {"check": "C0_schema_valid",
                "status": "FAIL",
                "detail": f"schema_invalid: missing/empty fields: {missing}"}
    return {"check": "C0_schema_valid",
            "status": "PASS",
            "detail": f"all {len(_REQUIRED_EVENT_FIELDS)} required fields present"}


def _check_parents_resolve(store_mod, event) -> dict:
    """C1: every parent_event_id must be findable in the store AND
    have an event_type that's a legitimate parent for this child type.

    T2.1 (2026-06-05 audit V4 fix): parent type validation added.
    Pre-T2.1 a factor_verdict_filed could legally cite a dq_breach as
    parent, which was a wiring bug — verdict shouldn't depend on a
    data-quality alert as upstream lineage. Now we validate parent
    types against the curated _ALLOWED_PARENT_TYPES_BY_CHILD map.
    """
    parent_ids = list(event.parent_event_ids or ())
    if not parent_ids:
        return {"check": "C1_parents_resolve",
                "status": "PASS",
                "detail": "no parents declared"}

    child_type = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
    allowed = _ALLOWED_PARENT_TYPES_BY_CHILD.get(child_type)

    missing = []
    wrong_type = []   # T2.1: parent resolves but is wrong type
    for pid in parent_ids:
        parent = store_mod.by_event_id(pid)
        if parent is None:
            missing.append(pid)
            continue
        if allowed is not None:
            ptype = parent.event_type.value if hasattr(parent.event_type, "value") else str(parent.event_type)
            if ptype not in allowed:
                wrong_type.append((pid[:8], ptype))

    if missing:
        return {"check": "C1_parents_resolve",
                "status": "FAIL",
                "detail": f"{len(missing)} of {len(parent_ids)} parent_event_ids "
                          f"not in store: {missing[:3]}{'...' if len(missing) > 3 else ''}"}
    if wrong_type:
        return {"check": "C1_parents_resolve",
                "status": "FAIL",
                "detail": (
                    f"{len(wrong_type)} of {len(parent_ids)} parents have "
                    f"disallowed type for child={child_type}; allowed="
                    f"{sorted(allowed)}; offenders={wrong_type[:3]}"
                )}
    return {"check": "C1_parents_resolve",
            "status": "PASS",
            "detail": f"all {len(parent_ids)} parents resolved"
                      f"{' + type-checked' if allowed is not None else ''}"}


def _check_artifacts_exist(event) -> dict:
    """C2: every artifact path exists on disk at the time of verification."""
    if not event.artifacts:
        return {"check": "C2_artifacts_exist",
                "status": "WARN",
                "detail": "no artifacts declared"}
    missing = {role: p for role, p in event.artifacts.items()
               if not _artifact_exists(p)}
    if missing:
        return {"check": "C2_artifacts_exist",
                "status": "FAIL",
                "detail": f"{len(missing)} artifact path(s) gone: "
                          f"{list(missing.keys())[:3]}"}
    return {"check": "C2_artifacts_exist",
            "status": "PASS",
            "detail": f"{len(event.artifacts)} artifact(s) on disk"}


def _check_paired_evidence(store_mod, event) -> dict:
    """C3: non-RED verdicts should have a paired capability_evidence_filed."""
    if event.verdict.value == "RED":
        return {"check": "C3_paired_evidence",
                "status": "PASS",
                "detail": "RED verdict; paired evidence optional"}
    if event.verdict.value == "NEUTRAL":
        return {"check": "C3_paired_evidence",
                "status": "PASS",
                "detail": "NEUTRAL verdict; not gated"}
    # GREEN / MARGINAL: look for a capability_evidence_filed within
    # +/- window seconds, same family, with this event in its parents.
    verdict_ts = _parse_ts(event.ts)
    if not verdict_ts:
        return {"check": "C3_paired_evidence",
                "status": "WARN",
                "detail": f"verdict ts unparseable: {event.ts}"}
    paired = []
    candidates = store_mod.filter_events(event_type="capability_evidence_filed", limit=50)
    for ev in candidates:
        if event.event_id not in (ev.parent_event_ids or ()):
            continue
        if event.family and ev.family and event.family != ev.family:
            continue
        ev_ts = _parse_ts(ev.ts)
        if not ev_ts:
            continue
        delta = abs((ev_ts - verdict_ts).total_seconds())
        if delta <= _PAIRED_EVIDENCE_WINDOW_SEC:
            paired.append(ev.event_id)
    if paired:
        return {"check": "C3_paired_evidence",
                "status": "PASS",
                "detail": f"paired capability_evidence: {paired[0][:8]}"}
    return {"check": "C3_paired_evidence",
            "status": "WARN",
            "detail": f"no paired capability_evidence_filed found within "
                      f"{_PAIRED_EVIDENCE_WINDOW_SEC}s; lineage incomplete"}


def _check_spec_grounded(event) -> dict:
    """C5 (2026-06-05): when a verdict cites a spec_hash, the
    Composer cache parquet for that hash must exist + the series in
    it must differ materially from the deployed sleeve. Closes the
    'replay_of_deployed = false test' failure mode the user called
    out as the project's soul gap.

    Tolerant: if no spec_hash referenced (legacy verdicts), passes
    with detail='no spec_hash referenced (pre-spec-layer)'.
    """
    artifacts = event.artifacts or {}
    metrics   = event.metrics or {}
    s_hash = (artifacts.get("spec_hash")
              or metrics.get("spec_hash")
              or None)
    if not s_hash:
        return {"check": "C5_spec_grounded",
                "status": "PASS",
                "detail": "no spec_hash referenced (pre-spec-layer verdict)"}
    # Check the Composer cache parquet exists
    cache_p = _REPO_ROOT / "data" / "composer_cache" / f"{s_hash}.parquet"
    if not cache_p.is_file():
        return {"check": "C5_spec_grounded",
                "status": "FAIL",
                "detail": f"spec_hash={s_hash[:8]}... cited but no Composer "
                          f"cache at data/composer_cache/{s_hash}.parquet"}
    return {"check": "C5_spec_grounded",
            "status": "PASS",
            "detail": f"spec_hash {s_hash[:8]} resolves to Composer cache"}


def _check_n_trials(event) -> dict:
    """C4: metrics['n_trials'] within 1..1000."""
    n = (event.metrics or {}).get("n_trials")
    if n is None:
        return {"check": "C4_n_trials_reasonable",
                "status": "PASS",
                "detail": "n_trials not reported"}
    try:
        n_int = int(n)
    except Exception:
        return {"check": "C4_n_trials_reasonable",
                "status": "WARN",
                "detail": f"n_trials non-numeric: {n!r}"}
    if n_int in _REASONABLE_N_TRIALS:
        return {"check": "C4_n_trials_reasonable",
                "status": "PASS",
                "detail": f"n_trials={n_int}"}
    return {"check": "C4_n_trials_reasonable",
            "status": "FAIL",
            "detail": f"n_trials={n_int} outside 1..1000; check family-aware count"}


# ── Subscription (import-time side effect) ────────────────────────


_SUBSCRIBED = False


def subscribe_to_bus() -> None:
    """Idempotent: attach `verify_factor_verdict_lineage` to the
    EventBus for the factor_verdict_filed event type. Safe to call
    multiple times; the second+ call is a no-op."""
    global _SUBSCRIBED
    if _SUBSCRIBED:
        return
    try:
        from engine.agents.event_bus import get_event_bus
        bus = get_event_bus()
        bus.subscribe("factor_verdict_filed", verify_factor_verdict_lineage)
        _SUBSCRIBED = True
        logger.info("audit_verifier: subscribed to factor_verdict_filed")
    except Exception as exc:
        logger.warning("audit_verifier: subscribe_to_bus failed: %s", exc, exc_info=True)


# Auto-subscribe on import unless explicitly disabled (tests).
if os.environ.get("AUDIT_VERIFIER_NO_AUTOSUBSCRIBE") != "1":
    subscribe_to_bus()
