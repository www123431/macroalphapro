"""scripts/backfill_research_store.py — one-shot historical backfill.

Scans:
  1. docs/capability_evidence/*.md  → factor_verdict_filed + capability_evidence_filed
  2. memory/*.md                    → memory_doctrine_locked
  3. data/research/factory_ledger.jsonl → factor_verdict_filed (gap fill)

Idempotent: re-runs skip subjects+ts already in store. Verbose: reports
per-source N emitted / N skipped / N parse-failed.

Run:
    python scripts/backfill_research_store.py [--dry-run] [--limit N]

Doctrine: this is the one-time pass. After M2, ongoing emit happens via
the live producer (Claude / cron). Backfilled events carry
actor='backfill_2026-06-02' so they're distinguishable from live emits.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from engine.research_store import registry, store
from engine.research_store.exceptions import (
    SubjectNotRegisteredError, DuplicateEventError, ArtifactMissingError,
)
from engine.research_store.schema import (
    EventType, SubjectType, Verdict, ResearchEvent, SCHEMA_VERSION,
)
from engine.research_store.manifest import current_git_sha


_BACKFILL_ACTOR = "backfill_2026-06-02"
_MEMORY_DIR = Path(os.path.expanduser(
    r"~/.claude/projects/c--Users-${USER}-Desktop-intern/memory"
))


# ── Helpers ───────────────────────────────────────────────────────


def _derive_subject_id(filename_stem: str) -> str:
    """Strip trailing date and verdict tokens from filename to get subject id."""
    name = filename_stem
    # Strip trailing date YYYY-MM-DD
    name = re.sub(r"_\d{4}-\d{2}-\d{2}$", "", name)
    # Strip trailing verdict / status tokens (greedy across multiples)
    pat = r"_(verdict_)?(red|green|pass|fail|marginal|partial|positive|shipped|complete|baseline|preliminary|robust|descriptive)(?=_|$)"
    prev = None
    while prev != name:
        prev = name
        name = re.sub(pat, "", name)
    return name


def _detect_verdict(filename_stem: str, content: str) -> Optional[Verdict]:
    """Detect verdict from filename (most reliable) or content header."""
    fn = filename_stem.lower()
    # Filename signals
    if re.search(r"(?:_|^)(red|fail)(?:_|$)", fn):
        return Verdict.RED
    if re.search(r"(?:_|^)(green|pass|positive|shipped|complete|deployable)(?:_|$)", fn):
        return Verdict.GREEN
    if re.search(r"(?:_|^)(marginal|partial)(?:_|$)", fn):
        return Verdict.MARGINAL
    if re.search(r"(?:_|^)(baseline|preliminary|infrastructure|ready|unlocked|shipped)(?:_|$)", fn):
        return Verdict.NEUTRAL
    # Infrastructure / sprint milestones default to NEUTRAL
    if re.match(r"^(sprint_|forensic_)", fn):
        return Verdict.NEUTRAL
    # Content header signals
    m = re.search(r"\*\*Verdict\*\*:\s*\*?\*?([A-Z]+)", content)
    if m:
        v = m.group(1).upper()
        mapping = {
            "PASS": Verdict.GREEN, "GREEN": Verdict.GREEN,
            "FAIL": Verdict.RED,   "RED": Verdict.RED,
            "MARGINAL": Verdict.MARGINAL, "PARTIAL": Verdict.MARGINAL,
        }
        return mapping.get(v)
    m = re.search(r"##\s*Verdict[:\s]*\*?\*?([A-Z]+)", content)
    if m:
        v = m.group(1).upper()
        return {"PASS": Verdict.GREEN, "FAIL": Verdict.RED,
                "MARGINAL": Verdict.MARGINAL}.get(v)
    return None


def _detect_ts(filename_stem: str, path: Path) -> str:
    """Prefer date in filename; fall back to file mtime."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", filename_stem)
    if m:
        return f"{m.group(1)}T00:00:00Z"
    mt = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
    return mt


def _detect_git_sha(content: str) -> str:
    m = re.search(r"[Cc]ode commit[s:\s]+([a-f0-9]{7,40})", content)
    if m:
        return m.group(1)[:10]
    m = re.search(r"\bcommit\s+([a-f0-9]{7,40})", content)
    if m:
        return m.group(1)[:10]
    return "historical"


def _detect_family(subject_id: str) -> Optional[str]:
    """Heuristic family from subject_id prefix."""
    pm = re.match(r"path_([a-z]+\d*)", subject_id)
    if pm:
        return f"path_{pm.group(1)}"
    for prefix, fam in [
        ("macro_", "macro"), ("factor_ensemble", "factor_ensemble"),
        ("d_pead", "pead"), ("sprint_", "infra"), ("ops_", "ops"),
        ("forensic_", "forensic"), ("phase_a", "position_weighting"),
        ("phase_b", "position_weighting"), ("phase_c", "position_weighting"),
        ("fomc", "fomc"), ("factor_mining", "factor_mining"),
    ]:
        if subject_id.startswith(prefix):
            return fam
    return None


def _existing_event_keys() -> set[tuple[str, str, str]]:
    """Return {(subject_id, event_type, ts)} already in store — for idempotency."""
    return {(e.subject_id, e.event_type.value, e.ts) for e in store.all_events()}


def _direct_append(event: ResearchEvent) -> bool:
    """Bypass emit() artifact validation (historical artifacts may have moved
    or not exist as files). Still goes through store.append() so id idempotency
    is enforced. Returns True if appended, False if dup."""
    try:
        store.append(event)
        return True
    except DuplicateEventError:
        return False


# ── Scanners ──────────────────────────────────────────────────────


def scan_capability_evidence(dry_run: bool = False) -> dict:
    """Walk docs/capability_evidence/*.md."""
    src_dir = _REPO_ROOT / "docs" / "capability_evidence"
    if not src_dir.is_dir():
        return {"scanned": 0, "emitted": 0, "skipped": 0, "parse_fail": 0}

    existing = _existing_event_keys()
    n_scanned = n_emitted = n_skipped = n_parse_fail = 0
    parse_failures: list[str] = []

    for path in sorted(src_dir.glob("*.md")):
        n_scanned += 1
        content = path.read_text(encoding="utf-8")
        subject_id = _derive_subject_id(path.stem)
        verdict = _detect_verdict(path.stem, content)
        if verdict is None:
            n_parse_fail += 1
            parse_failures.append(path.name)
            continue
        ts = _detect_ts(path.stem, path)
        git_sha = _detect_git_sha(content)
        family = _detect_family(subject_id)

        # Two events per evidence doc: the verdict + the doc filing
        key_verdict  = (subject_id, EventType.factor_verdict_filed.value, ts)
        key_filing   = (subject_id, EventType.capability_evidence_filed.value, ts)
        if key_verdict in existing and key_filing in existing:
            n_skipped += 1
            continue

        # Register subject if absent (idempotent)
        if registry.resolve(subject_id) is None:
            if not dry_run:
                registry.register_subject(
                    subject_id=subject_id,
                    subject_type=SubjectType.factor,
                    family=family,
                    description=f"Backfilled from {path.name}",
                    created_by=_BACKFILL_ACTOR,
                )

        # First-line summary from doc TL;DR or content
        summary = _extract_summary(content) or f"Backfilled from {path.name}: verdict {verdict.value}."
        evidence_rel = str(path.relative_to(_REPO_ROOT)).replace("\\", "/")

        if not dry_run:
            # 1. Verdict event
            if key_verdict not in existing:
                ev_id = uuid.uuid4().hex
                vev = ResearchEvent(
                    event_id=ev_id,
                    event_type=EventType.factor_verdict_filed,
                    ts=ts,
                    session_id="backfill",
                    actor=_BACKFILL_ACTOR,
                    subject_type=SubjectType.factor,
                    subject_id=subject_id,
                    verdict=verdict,
                    metrics={},
                    artifacts={"evidence_doc": evidence_rel},
                    parent_event_ids=(),
                    family=family,
                    tags=("backfill", "capability_evidence"),
                    summary=summary[:380],
                    git_sha=git_sha,
                    schema_version=SCHEMA_VERSION,
                )
                if _direct_append(vev):
                    existing.add(key_verdict)
                    n_emitted += 1
                # 2. Capability evidence filing companion (lineage)
                if key_filing not in existing:
                    fev = ResearchEvent(
                        event_id=uuid.uuid4().hex,
                        event_type=EventType.capability_evidence_filed,
                        ts=ts,
                        session_id="backfill",
                        actor=_BACKFILL_ACTOR,
                        subject_type=SubjectType.factor,
                        subject_id=subject_id,
                        verdict=verdict,
                        metrics={},
                        artifacts={"evidence_doc": evidence_rel},
                        parent_event_ids=(ev_id,),
                        family=family,
                        tags=("backfill", "capability_evidence"),
                        summary=f"Evidence doc filed: {path.name}",
                        git_sha=git_sha,
                        schema_version=SCHEMA_VERSION,
                    )
                    if _direct_append(fev):
                        existing.add(key_filing)
                        n_emitted += 1
            elif key_filing not in existing:
                # Only filing missing — emit standalone (no parent)
                fev = ResearchEvent(
                    event_id=uuid.uuid4().hex,
                    event_type=EventType.capability_evidence_filed,
                    ts=ts, session_id="backfill", actor=_BACKFILL_ACTOR,
                    subject_type=SubjectType.factor, subject_id=subject_id,
                    verdict=verdict, metrics={},
                    artifacts={"evidence_doc": evidence_rel},
                    parent_event_ids=(), family=family,
                    tags=("backfill", "capability_evidence"),
                    summary=f"Evidence doc filed: {path.name}",
                    git_sha=git_sha, schema_version=SCHEMA_VERSION,
                )
                if _direct_append(fev):
                    existing.add(key_filing)
                    n_emitted += 1
        else:
            n_emitted += 2  # dry-run accounting

    return {
        "scanned": n_scanned, "emitted": n_emitted,
        "skipped": n_skipped, "parse_fail": n_parse_fail,
        "parse_fail_files": parse_failures[:10],
    }


def _extract_summary(content: str) -> Optional[str]:
    """Pull a 1-sentence summary from TL;DR / first paragraph of body."""
    # TL;DR section
    m = re.search(r"##\s*TL;DR\s*\n+(.{40,800}?)(?:\n\n|\n##)", content, re.DOTALL)
    if m:
        text = m.group(1).strip()
        # Pick first sentence
        text = re.sub(r"\*+", "", text)
        first_sentence = re.split(r"(?<=[.。])\s+", text)[0]
        return first_sentence.strip()
    # First body paragraph after headline
    m = re.search(r"##[^\n]+\n+(.{40,400}?)(?:\n\n|\n##)", content, re.DOTALL)
    if m:
        text = re.sub(r"\*+", "", m.group(1).strip())
        return re.split(r"(?<=[.。])\s+", text)[0].strip()
    return None


def scan_memory(dry_run: bool = False, limit: Optional[int] = None) -> dict:
    """Walk memory/*.md."""
    if not _MEMORY_DIR.is_dir():
        return {"scanned": 0, "emitted": 0, "skipped": 0, "parse_fail": 0}

    existing = _existing_event_keys()
    n_scanned = n_emitted = n_skipped = n_parse_fail = 0

    files = sorted(_MEMORY_DIR.glob("*.md"))
    if limit:
        files = files[:limit]

    for path in files:
        if path.name == "MEMORY.md":
            continue
        n_scanned += 1
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            n_parse_fail += 1
            continue

        # Parse YAML frontmatter
        m = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not m:
            n_parse_fail += 1
            continue
        fm = m.group(1)

        name_m = re.search(r"^name:\s*(.+)$", fm, re.MULTILINE)
        desc_m = re.search(r"^description:\s*(.+)$", fm, re.MULTILINE)
        type_m = re.search(r"^\s*type:\s*(.+)$", fm, re.MULTILINE)

        if not name_m:
            n_parse_fail += 1
            continue

        # subject_id namespaces memory entries to avoid collision with factor subjects
        subject_id = f"memory_{name_m.group(1).strip().replace('-', '_')}"
        description = (desc_m.group(1).strip() if desc_m else "")
        mem_type = (type_m.group(1).strip() if type_m else "memory")

        # ts from file mtime
        ts = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%SZ")
        key = (subject_id, EventType.memory_doctrine_locked.value, ts)
        if key in existing:
            n_skipped += 1
            continue

        # Family hint — extract from filename or memory type
        family = None
        if "feedback_" in path.name:
            family = "doctrine_feedback"
        elif "project_" in path.name:
            family = "doctrine_project"

        if registry.resolve(subject_id) is None and not dry_run:
            registry.register_subject(
                subject_id=subject_id,
                subject_type=SubjectType.memory_doctrine,
                family=family,
                description=description[:300] or f"Memory: {name_m.group(1)} ({mem_type})",
                created_by=_BACKFILL_ACTOR,
            )

        if not dry_run:
            mem_path_str = str(path).replace("\\", "/")
            ev = ResearchEvent(
                event_id=uuid.uuid4().hex,
                event_type=EventType.memory_doctrine_locked,
                ts=ts, session_id="backfill", actor=_BACKFILL_ACTOR,
                subject_type=SubjectType.memory_doctrine, subject_id=subject_id,
                verdict=Verdict.NEUTRAL, metrics={},
                artifacts={"memory_doc": mem_path_str},
                parent_event_ids=(), family=family,
                tags=("backfill", mem_type),
                summary=(description or f"Memory {name_m.group(1)}")[:380],
                git_sha="historical", schema_version=SCHEMA_VERSION,
            )
            if _direct_append(ev):
                existing.add(key)
                n_emitted += 1
        else:
            n_emitted += 1

    return {
        "scanned": n_scanned, "emitted": n_emitted,
        "skipped": n_skipped, "parse_fail": n_parse_fail,
    }


def scan_factory_ledger(dry_run: bool = False) -> dict:
    """Walk data/research/factory_ledger.jsonl — these are gate-runner results
    not captured by capability evidence docs."""
    path = _REPO_ROOT / "data" / "research" / "factory_ledger.jsonl"
    if not path.is_file():
        return {"scanned": 0, "emitted": 0, "skipped": 0, "parse_fail": 0}

    existing = _existing_event_keys()
    n_scanned = n_emitted = n_skipped = n_parse_fail = 0

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_parse_fail += 1
                continue
            candidate = row.get("candidate") or row.get("name")
            verdict_raw = (row.get("verdict") or "").upper()
            ts_raw = row.get("ts") or row.get("as_of")
            if not (candidate and verdict_raw and ts_raw):
                n_parse_fail += 1
                continue

            mapping = {
                "GREEN_WINNER": Verdict.GREEN, "GREEN": Verdict.GREEN,
                "RED": Verdict.RED, "FAIL": Verdict.RED,
                "MARGINAL": Verdict.MARGINAL,
                "TESTED_NEUTRAL": Verdict.NEUTRAL, "NEUTRAL": Verdict.NEUTRAL,
            }
            verdict = mapping.get(verdict_raw)
            if verdict is None:
                n_parse_fail += 1
                continue

            # subject_id from candidate (sanitize)
            subject_id = re.sub(r"[^a-zA-Z0-9_]+", "_", candidate.lower()).strip("_")
            if not subject_id:
                n_parse_fail += 1
                continue

            key = (subject_id, EventType.factor_verdict_filed.value, ts_raw)
            if key in existing:
                n_skipped += 1
                continue

            family = row.get("family")
            metrics = {k: row[k] for k in ("sharpe", "deflated_sr", "n_months", "t_stat")
                       if k in row and row[k] is not None}

            if registry.resolve(subject_id) is None and not dry_run:
                registry.register_subject(
                    subject_id=subject_id, subject_type=SubjectType.factor,
                    family=family,
                    description=f"Backfilled from factory_ledger: {candidate}",
                    created_by=_BACKFILL_ACTOR,
                )

            if not dry_run:
                ev = ResearchEvent(
                    event_id=uuid.uuid4().hex,
                    event_type=EventType.factor_verdict_filed,
                    ts=ts_raw, session_id="backfill", actor=_BACKFILL_ACTOR,
                    subject_type=SubjectType.factor, subject_id=subject_id,
                    verdict=verdict, metrics=metrics,
                    artifacts={"factory_ledger": "data/research/factory_ledger.jsonl"},
                    parent_event_ids=(), family=family,
                    tags=("backfill", "factory_ledger", row.get("source", "").split("/")[-1]),
                    summary=f"Factory ledger: {candidate} → {verdict_raw}",
                    git_sha="historical", schema_version=SCHEMA_VERSION,
                )
                if _direct_append(ev):
                    existing.add(key)
                    n_emitted += 1
            else:
                n_emitted += 1

    return {
        "scanned": n_scanned, "emitted": n_emitted,
        "skipped": n_skipped, "parse_fail": n_parse_fail,
    }


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and count but do not write to store.")
    ap.add_argument("--limit-memory", type=int, default=None,
                    help="Limit memory file scan (debug).")
    ap.add_argument("--skip-memory", action="store_true",
                    help="Skip memory backfill (large; ~300 files).")
    ap.add_argument("--skip-evidence", action="store_true")
    ap.add_argument("--skip-ledger", action="store_true")
    args = ap.parse_args()

    print(f"=== Research store backfill ({'DRY-RUN' if args.dry_run else 'LIVE'}) ===\n")

    if not args.skip_evidence:
        print("[1/3] Scanning docs/capability_evidence/*.md ...")
        r = scan_capability_evidence(dry_run=args.dry_run)
        print(f"   scanned={r['scanned']}  emitted={r['emitted']}  "
              f"skipped={r['skipped']}  parse_fail={r['parse_fail']}")
        if r["parse_fail"] and r.get("parse_fail_files"):
            for f in r["parse_fail_files"]:
                print(f"     parse-fail: {f}")
        print()

    if not args.skip_memory:
        print(f"[2/3] Scanning memory/*.md (limit={args.limit_memory or 'all'}) ...")
        r = scan_memory(dry_run=args.dry_run, limit=args.limit_memory)
        print(f"   scanned={r['scanned']}  emitted={r['emitted']}  "
              f"skipped={r['skipped']}  parse_fail={r['parse_fail']}")
        print()

    if not args.skip_ledger:
        print("[3/3] Scanning data/research/factory_ledger.jsonl ...")
        r = scan_factory_ledger(dry_run=args.dry_run)
        print(f"   scanned={r['scanned']}  emitted={r['emitted']}  "
              f"skipped={r['skipped']}  parse_fail={r['parse_fail']}")
        print()

    if not args.dry_run:
        total_events = len(store.all_events())
        total_subjects = len(registry.list_subjects())
        print(f"=== Store totals after backfill ===")
        print(f"   total events:   {total_events}")
        print(f"   total subjects: {total_subjects}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
