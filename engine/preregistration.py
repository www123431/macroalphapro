"""
engine/preregistration.py — Pre-Registration Enforcement Layer (S3)

Spec: docs/spec_pre_registration_enforcement.md  (spec_hash 292fdd6039f90d05).

Three responsibilities:

  1. Spec immutability — register a spec on first lock, record its
     git-blob-style content hash, and require explicit `amend_spec` calls
     for any subsequent change.
  2. n_trials accounting — each amendment contributes to
     EFFECTIVE_N_TRIALS via a kind-graded multiplier (clarification = 0,
     threshold_tweak = 1, hypothesis_amend = 3, endpoint_swap = 5).
  3. HARKing detection — rules R1-R4 (silent edit, threshold drift,
     unannounced trial, predictions rewrite) flag suspicious patterns
     without LLM involvement (per feedback_no_llm_as_judge.md).

Public API:
  register_spec(path, retro=False, *, session=None)   -> SpecRegistry row id
  amend_spec(path, kind, reason, *, session=None)     -> SpecRegistry row id
  validate_reference(spec_path, *, session=None)      -> (ok: bool, reason: str)
  list_specs(*, session=None)                         -> list[dict]
  detect_harking(*, as_of=None, session=None)         -> list[dict]
  compute_pre_registration_n_trials(*, session=None)  -> int

CLI: see __main__ block at bottom.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Amendment kinds with their n_trials contribution.
#
# 2026-05-12 update: BHY-FDR doctrine retired (paper publication path dropped
# per project_wave_b_10y_verdict_regime_sensitivity_2026-05-12). All amendment
# kinds now contribute 0 trials. Kind LABELS retained as semantic audit-log
# tags (clarification vs threshold_tweak vs hypothesis_amend vs etc. still
# tells you what TYPE of change happened), but the +1/+3/+5 cost ladder is
# zeroed out — see feedback_pretest_experimental_rigor (rule #3 demoted) +
# feedback_pre_register_data_access_verify for context.
#
# Historical n_trials_contributed values in DB (e.g., id=57=1) PRESERVED.
# compute_pre_registration_n_trials() still returns sum for transparency.
# register_spec still contributes +1 per new spec (one hypothesis registered).
# Reversible: if SSRN publication path reopens, restore non-zero values.
AMENDMENT_KINDS = {
    "clarification":         0,
    "scope_narrow":          0,
    "threshold_tweak":       0,   # was 1 (pre-2026-05-12)
    "hypothesis_amend":      0,   # was 3 (pre-2026-05-12)
    "endpoint_swap":         0,   # was 5 (pre-2026-05-12)
    "superseded":            0,
    "lab_state_transition":  0,   # P-LAB workflow (2026-05-08)
}


# ─────────────────────────────────────────────────────────────────────────────
# Hash helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_git_blob_hash(path: str) -> str:
    """
    Compute the git-blob-style SHA-1 of a file (matches `git hash-object <path>`).
    git blob hash = sha1("blob " + size + "\\0" + content).

    On Windows with core.autocrlf=true (the default for Git for Windows), the
    working-tree file has CRLF but the blob stored by git is LF. We replicate
    git's behavior by stripping CR before LF when autocrlf is true, so the
    hook's hash matches the staged blob and the SpecRegistry DB row.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        content = f.read()
    if _git_autocrlf_active():
        content = content.replace(b"\r\n", b"\n")
    header = f"blob {len(content)}\0".encode("utf-8")
    return hashlib.sha1(header + content).hexdigest()


@functools.lru_cache(maxsize=1)
def _git_autocrlf_active() -> bool:
    try:
        out = subprocess.run(
            ["git", "config", "--get", "core.autocrlf"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip().lower() in ("true", "input")
    except Exception:
        return False


def _normalize_spec_path(path: str) -> str:
    """
    Repo-relative path with forward slashes; this is what's stored.
    Accepts absolute, relative, or with backslashes.
    """
    p = os.path.normpath(path).replace("\\", "/")
    repo_root = os.path.normpath(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))).replace("\\", "/")
    if p.startswith(repo_root + "/"):
        p = p[len(repo_root) + 1:]
    return p


def _resolve_to_abs(spec_path: str) -> str:
    """Inverse of _normalize_spec_path: turn repo-relative into absolute."""
    if os.path.isabs(spec_path):
        return spec_path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, spec_path)


# ─────────────────────────────────────────────────────────────────────────────
# Core API
# ─────────────────────────────────────────────────────────────────────────────

def register_spec(
    path:    str,
    retro:   bool = False,
    *,
    session: Any | None = None,
) -> int:
    """
    Register a spec for the first time. Idempotent: re-registering an
    already-registered spec returns the existing row id and updates the
    `current_hash` and `last_validated_at` fields.

    Args:
        path:    spec file path (absolute, relative, or repo-relative)
        retro:   True if registering an already-existing spec; sets
                 retro_registered=True and does NOT count toward forward
                 EFFECTIVE_N_TRIALS contribution.
    """
    from engine.memory import SpecRegistry, SessionFactory

    abs_path = _resolve_to_abs(path)
    rel_path = _normalize_spec_path(abs_path)
    blob_hash = _compute_git_blob_hash(abs_path)

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        existing = sess.query(SpecRegistry).filter(
            SpecRegistry.spec_path == rel_path
        ).first()
        if existing:
            existing.current_hash = blob_hash
            existing.last_validated_at = datetime.datetime.utcnow()
            sess.commit()
            return existing.id

        row = SpecRegistry(
            spec_path           = rel_path,
            git_blob_hash       = blob_hash,
            current_hash        = blob_hash,
            registered_at       = datetime.datetime.utcnow(),
            amendment_log       = "[]",
            status              = "active",
            retro_registered    = bool(retro),
            n_trials_contributed = 0 if retro else 1,
            last_validated_at   = datetime.datetime.utcnow(),
        )
        sess.add(row)
        sess.commit()
        logger.info(
            "register_spec: %s hash=%s retro=%s n_trials=%d",
            rel_path, blob_hash[:12], retro, row.n_trials_contributed,
        )
        return row.id
    finally:
        if own:
            sess.close()


def amend_spec(
    path:    str,
    kind:    str,
    reason:  str,
    *,
    session: Any | None = None,
) -> int:
    """
    Append an amendment to an existing spec. Re-hashes the file and adds
    a ledger entry recording (timestamp, kind, reason, new_hash, n_trials_added).

    Args:
        path:   spec file path
        kind:   one of AMENDMENT_KINDS keys
        reason: free-text justification (≥20 chars enforced)
    """
    from engine.memory import SpecRegistry, SessionFactory

    if kind not in AMENDMENT_KINDS:
        raise ValueError(
            f"unknown amendment kind {kind!r}; expected one of {list(AMENDMENT_KINDS)}"
        )
    if not reason or len(reason.strip()) < 20:
        raise ValueError("amend_spec: reason must be ≥20 characters")

    abs_path = _resolve_to_abs(path)
    rel_path = _normalize_spec_path(abs_path)
    new_hash = _compute_git_blob_hash(abs_path)

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row = sess.query(SpecRegistry).filter(
            SpecRegistry.spec_path == rel_path
        ).first()
        if row is None:
            raise RuntimeError(
                f"amend_spec: {rel_path} not registered; call register_spec first"
            )

        try:
            ledger = json.loads(row.amendment_log or "[]")
        except Exception:
            ledger = []

        n_trials_added = AMENDMENT_KINDS[kind]
        entry = {
            "at":             datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "kind":           kind,
            "reason":         reason.strip(),
            "prev_hash":      row.current_hash,
            "new_hash":       new_hash,
            "n_trials_added": n_trials_added,
        }
        ledger.append(entry)

        row.amendment_log = json.dumps(ledger, ensure_ascii=False)
        row.current_hash  = new_hash
        row.n_trials_contributed += n_trials_added
        if kind == "superseded":
            row.status = "superseded"
        row.last_validated_at = datetime.datetime.utcnow()

        sess.commit()
        logger.info(
            "amend_spec: %s kind=%s n_trials_added=%d cumulative=%d",
            rel_path, kind, n_trials_added, row.n_trials_contributed,
        )
        return row.id
    finally:
        if own:
            sess.close()


def validate_reference(
    spec_path: str,
    *,
    session:   Any | None = None,
) -> tuple[bool, str]:
    """
    Called by backtest / paper trading runs that reference a spec. Returns
    (ok, reason). Side effects: stamps first_referenced_at on first call.
    `ok = False` when:
      * spec not registered (HARKing R3-prone)
      * file content hash differs from current_hash (silent edit, R1)
    """
    from engine.memory import SpecRegistry, SessionFactory

    abs_path = _resolve_to_abs(spec_path)
    rel_path = _normalize_spec_path(abs_path)

    if not os.path.exists(abs_path):
        return (False, "spec_file_missing")

    actual_hash = _compute_git_blob_hash(abs_path)

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        row = sess.query(SpecRegistry).filter(
            SpecRegistry.spec_path == rel_path
        ).first()
        if row is None:
            return (False, "not_registered")

        # Stamp first reference (idempotent first-only)
        if row.first_referenced_at is None:
            row.first_referenced_at = datetime.datetime.utcnow()
        row.last_validated_at = datetime.datetime.utcnow()
        sess.commit()

        if actual_hash != row.current_hash:
            return (False, "silent_edit_detected")
        return (True, "ok")
    finally:
        if own:
            sess.close()


def list_specs(*, session: Any | None = None) -> list[dict]:
    """Return all registered specs as plain dicts (UI / CLI consumption)."""
    from engine.memory import SpecRegistry, SessionFactory

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        rows = sess.query(SpecRegistry).order_by(
            SpecRegistry.registered_at.asc()
        ).all()
        out = []
        for r in rows:
            try:
                ledger = json.loads(r.amendment_log or "[]")
            except Exception:
                ledger = []
            out.append({
                "id":                   r.id,
                "spec_path":            r.spec_path,
                "git_blob_hash":        r.git_blob_hash,
                "current_hash":         r.current_hash,
                "registered_at":        r.registered_at.isoformat() if r.registered_at else None,
                "status":               r.status,
                "retro_registered":     r.retro_registered,
                "n_amendments":         len(ledger),
                "n_trials_contributed": r.n_trials_contributed,
                "first_referenced_at":  r.first_referenced_at.isoformat() if r.first_referenced_at else None,
                "amendment_log":        ledger,
            })
        return out
    finally:
        if own:
            sess.close()


def compute_pre_registration_n_trials(
    *,
    session: Any | None = None,
) -> int:
    """
    Sum of n_trials_contributed across all NON-retro-registered specs.
    Per spec §三 Sprint 2: only forward (non-retro) registrations contribute
    to EFFECTIVE_N_TRIALS to avoid penalising the project for legacy specs.
    """
    from engine.memory import SpecRegistry, SessionFactory
    from sqlalchemy import func

    own = session is None
    sess = session if session is not None else SessionFactory()
    try:
        total = sess.query(func.coalesce(
            func.sum(SpecRegistry.n_trials_contributed), 0,
        )).filter(SpecRegistry.retro_registered.is_(False)).scalar()
        return int(total or 0)
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# HARKing detection (S3.3, called from S3.1 too for self-test)
# ─────────────────────────────────────────────────────────────────────────────

# R2 threshold drift: regex catalogue. Reusable across specs.
_THRESHOLD_PATTERNS = [
    re.compile(r"\bSharpe\s*[≥>=]+\s*[\d.]+", re.IGNORECASE),
    re.compile(r"\bNW\s*t\s*[≥>=]+\s*[\d.]+", re.IGNORECASE),
    re.compile(r"\bp\s*[<≤=]\s*0?\.\d+",       re.IGNORECASE),
    re.compile(r"\bα\s*=\s*\d+\s*%"),
    re.compile(r"\balpha\s*=\s*0?\.\d+",        re.IGNORECASE),
]


def _extract_thresholds(text: str) -> set[str]:
    matches: set[str] = set()
    for pat in _THRESHOLD_PATTERNS:
        for m in pat.findall(text):
            matches.add(m)
    return matches


def detect_harking(
    *,
    as_of:   datetime.datetime | None = None,
    session: Any | None = None,
) -> list[dict]:
    """
    Run R1-R4 across the registry. Returns a list of newly-detected flags
    (already persisted). Idempotent: existing unresolved flags for the same
    (rule, spec_path) tuple are not duplicated.

    Spec §2.4:
      R1 CRITICAL — current_hash != last amendment new_hash AND first_referenced_at IS NOT NULL
      R2 HIGH     — threshold drift detected via regex but no recent amendment
      R3 HIGH     — DecisionLog row references spec_hash not in registry
      R4 MEDIUM   — amendment_log accumulates ≥2 hypothesis_amend entries
    """
    from engine.memory import SpecRegistry, HARKingFlag, DecisionLog, SessionFactory

    if as_of is None:
        as_of = datetime.datetime.utcnow()

    own = session is None
    sess = session if session is not None else SessionFactory()
    flags_raised: list[dict] = []
    try:
        rows = sess.query(SpecRegistry).filter(
            SpecRegistry.status == "active"
        ).all()

        existing_unresolved = {
            (f.rule, f.spec_path): f
            for f in sess.query(HARKingFlag).filter(
                HARKingFlag.resolved_at.is_(None)
            ).all()
        }

        def raise_flag(rule: str, spec_path: str, severity: str, notes: str):
            key = (rule, spec_path)
            if key in existing_unresolved:
                return  # already open
            flag = HARKingFlag(
                rule=rule, spec_path=spec_path, severity=severity,
                detected_at=as_of, notes=notes,
            )
            sess.add(flag)
            flags_raised.append({
                "rule": rule, "spec_path": spec_path,
                "severity": severity, "notes": notes,
            })

        # R1 — silent edit after first reference
        for r in rows:
            try:
                ledger = json.loads(r.amendment_log or "[]")
            except Exception:
                ledger = []
            last_recorded_hash = (
                ledger[-1]["new_hash"] if ledger else r.git_blob_hash
            )
            if (r.first_referenced_at is not None
                    and r.current_hash != last_recorded_hash):
                raise_flag(
                    "R1", r.spec_path, "CRITICAL",
                    f"current_hash {r.current_hash[:12]} != last_recorded {last_recorded_hash[:12]}",
                )

        # R2 — threshold drift without recent amendment
        seven_days_ago = as_of - datetime.timedelta(days=7)
        for r in rows:
            abs_p = _resolve_to_abs(r.spec_path)
            if not os.path.exists(abs_p):
                continue
            with open(abs_p, "r", encoding="utf-8") as f:
                live_text = f.read()
            live_thresholds = _extract_thresholds(live_text)
            try:
                ledger = json.loads(r.amendment_log or "[]")
            except Exception:
                ledger = []
            recent_amend = any(
                datetime.datetime.fromisoformat(
                    e["at"].rstrip("Z")
                ) >= seven_days_ago
                for e in ledger
                if "at" in e
            )
            # Use git_blob_hash as the "registered" content for comparison.
            # If thresholds differ between current and original we raise.
            if r.current_hash != r.git_blob_hash and not recent_amend and not r.retro_registered:
                # Compare thresholds at original vs current. We can't read the
                # original content (only its hash), so this is conservative:
                # any divergent hash + threshold pattern in current = flag.
                if live_thresholds:
                    raise_flag(
                        "R2", r.spec_path, "HIGH",
                        f"current_hash drift, no amendment in 7d, "
                        f"thresholds present: {sorted(live_thresholds)[:3]}",
                    )

        # R3 — DecisionLog refers to spec_hash not in registry
        registered_hashes = {r.git_blob_hash for r in rows} | {r.current_hash for r in rows}
        # Plus any historical hashes from amendment ledgers
        for r in rows:
            try:
                ledger = json.loads(r.amendment_log or "[]")
            except Exception:
                ledger = []
            for e in ledger:
                if "new_hash" in e:
                    registered_hashes.add(e["new_hash"])
                if "prev_hash" in e:
                    registered_hashes.add(e["prev_hash"])

        try:
            unknown_hashes = (
                sess.query(DecisionLog.spec_hash)
                    .filter(DecisionLog.spec_hash.isnot(None))
                    .filter(~DecisionLog.spec_hash.in_(registered_hashes or [""]))
                    .distinct()
                    .all()
            )
            for (h,) in unknown_hashes:
                if h:
                    raise_flag(
                        "R3", "(decision_logs)", "HIGH",
                        f"DecisionLog references unknown spec_hash {h[:12]}",
                    )
        except Exception as exc:
            # Tolerate if DecisionLog has no spec_hash column yet (S3.2 not done)
            logger.debug("R3 skipped: %s", exc)

        # R4 — ≥2 hypothesis_amend entries
        for r in rows:
            try:
                ledger = json.loads(r.amendment_log or "[]")
            except Exception:
                ledger = []
            n_hyp = sum(1 for e in ledger if e.get("kind") == "hypothesis_amend")
            if n_hyp >= 2:
                raise_flag(
                    "R4", r.spec_path, "MEDIUM",
                    f"hypothesis_amend count = {n_hyp} (≥2)",
                )

        sess.commit()
        return flags_raised
    finally:
        if own:
            sess.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        prog="python -m engine.preregistration",
        description="Pre-registration enforcement layer (S3)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Register a spec")
    p_reg.add_argument("path")
    p_reg.add_argument("--retro", action="store_true",
                       help="Mark as retro-registered (no n_trials contribution)")

    p_amend = sub.add_parser("amend", help="Record an amendment")
    p_amend.add_argument("path")
    p_amend.add_argument("--kind", required=True, choices=list(AMENDMENT_KINDS))
    p_amend.add_argument("--reason", required=True)

    p_list = sub.add_parser("list", help="List all registered specs")
    p_list.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted table")

    p_val = sub.add_parser("validate", help="Validate one spec reference")
    p_val.add_argument("path")

    p_ntr = sub.add_parser("n_trials",
                           help="Show pre-registration contribution to EFFECTIVE_N_TRIALS")

    p_har = sub.add_parser("harking", help="Run HARKing R1-R4 detection")
    p_har.add_argument("--json", action="store_true")

    args = parser.parse_args()

    # Logging to stderr so JSON output stays clean on stdout
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stderr)

    from engine.memory import init_db
    init_db()

    if args.cmd == "register":
        rid = register_spec(args.path, retro=args.retro)
        print(f"OK: registered id={rid} path={args.path}")
    elif args.cmd == "amend":
        rid = amend_spec(args.path, kind=args.kind, reason=args.reason)
        print(f"OK: amendment recorded id={rid}")
    elif args.cmd == "list":
        rows = list_specs()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"{'id':>4} {'path':50s} {'hash':14s} {'retro':6s} "
                  f"{'amends':6s} {'n_trials':8s}")
            for r in rows:
                print(f"{r['id']:>4} {r['spec_path'][:50]:50s} "
                      f"{r['current_hash'][:12]:14s} "
                      f"{'yes' if r['retro_registered'] else 'no':6s} "
                      f"{r['n_amendments']:>6d} "
                      f"{r['n_trials_contributed']:>8d}")
    elif args.cmd == "validate":
        ok, reason = validate_reference(args.path)
        print(f"{'OK' if ok else 'FAIL'}: {reason}")
        sys.exit(0 if ok else 1)
    elif args.cmd == "n_trials":
        n = compute_pre_registration_n_trials()
        print(f"pre_registration_n_trials = {n}")
    elif args.cmd == "harking":
        flags = detect_harking()
        if args.json:
            print(json.dumps(flags, ensure_ascii=False, indent=2))
        else:
            if not flags:
                print("No new flags raised.")
            else:
                for f in flags:
                    print(f"[{f['severity']:>8}] {f['rule']} {f['spec_path']}: {f['notes']}")


if __name__ == "__main__":
    _cli()
