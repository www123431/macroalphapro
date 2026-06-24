"""Build the public GitHub snapshot from private dev repo.

Reads .publishrc.yaml at repo root and produces a clean copy at
the configured snapshot_root path. Whitelist approach: ONLY paths
matched by `include` are copied.

Pipeline (5 stages):
  1. Walk repo, collect candidate paths matched by `include`
  2. Apply `exclude` filter (defense-in-depth)
  3. Copy to snapshot_root with directory structure preserved
  4. Apply `sanitize_patterns` regex replace on text files
  5. Run `post_check_forbidden` grep — fail loudly if any hit

Usage:
  python scripts/publish/build_public_snapshot.py
  python scripts/publish/build_public_snapshot.py --dry-run
  python scripts/publish/build_public_snapshot.py --strict   # fail on warn

Written 2026-06-23 for B1 dual-repo arch. Re-run weekly via cron to
refresh public mirror.
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHRC = REPO_ROOT / ".publishrc.yaml"

TEXT_EXTENSIONS = {
    ".py", ".md", ".tex", ".bib", ".toml", ".yaml", ".yml",
    ".json", ".js", ".jsx", ".ts", ".tsx", ".css", ".html",
    ".sh", ".txt", ".cfg", ".ini",
}


@dataclass
class SnapshotReport:
    candidates_total: int = 0
    included: int = 0
    excluded: int = 0
    copied: int = 0
    sanitized_files: int = 0
    sanitize_hits: dict[str, int] = field(default_factory=dict)
    post_check_failures: list[tuple[str, str, str]] = field(default_factory=list)
    skipped_binary: int = 0
    bytes_copied: int = 0


def _load_config() -> dict:
    if not PUBLISHRC.exists():
        sys.exit(f"[publish] .publishrc.yaml not found at {PUBLISHRC}")
    with PUBLISHRC.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand_braces(pattern: str) -> list[str]:
    """Expand `{a,b,c}` brace groups into separate patterns. Single-level only."""
    m = re.search(r"\{([^{}]+)\}", pattern)
    if not m:
        return [pattern]
    options = m.group(1).split(",")
    prefix, suffix = pattern[: m.start()], pattern[m.end() :]
    out: list[str] = []
    for opt in options:
        out.extend(_expand_braces(prefix + opt + suffix))
    return out


def _glob_match(rel_path: str, pattern: str) -> bool:
    """Glob match with `**` recursive support + brace expansion.

    Tries each brace-expanded variant via fnmatch. For `**/*X` patterns,
    ALSO tries the no-intermediate-dir variant `*X` under the prefix.
    """
    for pat in _expand_braces(pattern):
        if fnmatch.fnmatch(rel_path, pat):
            return True
        # `dir/**/*.ext` should also match `dir/file.ext` (no intermediate)
        if "/**/" in pat:
            alt = pat.replace("/**/", "/")
            if fnmatch.fnmatch(rel_path, alt):
                return True
        if pat.startswith("**/"):
            alt = pat[3:]
            if fnmatch.fnmatch(rel_path, alt):
                return True
    return False


def _match_any(rel_path: str, patterns: list[str]) -> tuple[bool, bool]:
    """Return (matched, is_negation). Negation patterns start with `!`.

    Last-match-wins semantics for glob lists.
    """
    matched = False
    negated_match = False
    for pat in patterns:
        if pat.startswith("!"):
            if _glob_match(rel_path, pat[1:]):
                negated_match = True
                matched = False
        else:
            if _glob_match(rel_path, pat):
                matched = True
                negated_match = False
    return matched, negated_match


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def _collect_candidates(config: dict, report: SnapshotReport) -> list[Path]:
    include = config.get("include", [])
    exclude = config.get("exclude", [])
    candidates: list[Path] = []

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        report.candidates_total += 1

        inc_matched, _ = _match_any(rel, include)
        if not inc_matched:
            continue

        exc_matched, exc_negated = _match_any(rel, exclude)
        if exc_matched and not exc_negated:
            report.excluded += 1
            continue

        report.included += 1
        candidates.append(path)

    return candidates


def _apply_sanitize(text: str, patterns: list[dict], report: SnapshotReport) -> tuple[str, int]:
    """Apply regex sanitize patterns. Returns (new_text, hits_in_this_file)."""
    hits = 0
    for entry in patterns:
        pat = entry["pattern"]
        rep = entry["replacement"]
        compiled = re.compile(pat)
        new_text, n = compiled.subn(rep, text)
        if n > 0:
            hits += n
            report.sanitize_hits[pat] = report.sanitize_hits.get(pat, 0) + n
            text = new_text
    return text, hits


def _copy_and_sanitize(
    candidates: list[Path],
    snapshot_root: Path,
    sanitize_patterns: list[dict],
    report: SnapshotReport,
    dry_run: bool,
) -> None:
    for src in candidates:
        rel = src.relative_to(REPO_ROOT)
        dst = snapshot_root / rel
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)

        if _is_text_file(src):
            try:
                content = src.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                if not dry_run:
                    shutil.copy2(src, dst)
                report.skipped_binary += 1
                report.copied += 1
                report.bytes_copied += src.stat().st_size
                continue
            new_content, hits = _apply_sanitize(content, sanitize_patterns, report)
            if hits > 0:
                report.sanitized_files += 1
            if not dry_run:
                dst.write_text(new_content, encoding="utf-8")
            report.copied += 1
            report.bytes_copied += len(new_content.encode("utf-8"))
        else:
            if not dry_run:
                shutil.copy2(src, dst)
            report.skipped_binary += 1
            report.copied += 1
            report.bytes_copied += src.stat().st_size


def _post_check(snapshot_root: Path, forbidden: list[str], report: SnapshotReport) -> None:
    """Grep snapshot for forbidden patterns; populate report.post_check_failures."""
    if not snapshot_root.exists():
        return
    compiled = [(p, re.compile(p)) for p in forbidden]
    for path in snapshot_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pat_str, pat in compiled:
            m = pat.search(text)
            if m:
                rel = path.relative_to(snapshot_root).as_posix()
                snippet = m.group(0)[:60]
                report.post_check_failures.append((rel, pat_str, snippet))


def _write_report(report: SnapshotReport, config: dict, dry_run: bool) -> Path:
    report_rel = config.get("report_path", "data/publish/snapshot_report.md")
    report_path = REPO_ROOT / report_rel
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Public snapshot report")
    lines.append(f"\n_Mode_: {'DRY-RUN' if dry_run else 'LIVE'}")
    lines.append(f"_Snapshot root_: `{config['snapshot_root']}`\n")
    lines.append("## Counts\n")
    lines.append(f"- candidates scanned: {report.candidates_total}")
    lines.append(f"- included by whitelist: {report.included}")
    lines.append(f"- excluded by blacklist: {report.excluded}")
    lines.append(f"- copied to snapshot: {report.copied}")
    lines.append(f"- bytes copied: {report.bytes_copied / 1024 / 1024:.1f} MB")
    lines.append(f"- files where sanitize replaced something: {report.sanitized_files}")
    lines.append(f"- binary files copied without sanitize: {report.skipped_binary}")

    if report.sanitize_hits:
        lines.append("\n## Sanitize hits (pattern → total replacements)\n")
        for pat, n in sorted(report.sanitize_hits.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{pat}` → {n}")

    if report.post_check_failures:
        lines.append("\n## [FAIL] POST-CHECK FAILURES (forbidden patterns found in snapshot)\n")
        lines.append("These MUST be fixed before pushing to public GitHub.\n")
        for rel, pat, snip in report.post_check_failures[:50]:
            lines.append(f"- `{rel}` matched `{pat}`: `{snip}`")
        if len(report.post_check_failures) > 50:
            lines.append(f"\n_({len(report.post_check_failures) - 50} more...)_")
    else:
        lines.append("\n## [OK] Post-check clean -- no forbidden patterns in snapshot.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute report without writing files")
    ap.add_argument("--strict", action="store_true", help="Exit 1 on any post-check failure")
    args = ap.parse_args()

    config = _load_config()
    snapshot_root = Path(config["snapshot_root"]).resolve()
    report = SnapshotReport()

    print(f"[publish] mode: {'DRY-RUN' if args.dry_run else 'LIVE'}")
    print(f"[publish] snapshot root: {snapshot_root}")

    if not args.dry_run:
        if snapshot_root.exists():
            print(f"[publish] cleaning existing snapshot at {snapshot_root}")
            # Preserve a .git/ dir if user has initialized one for git mirror
            git_dir = snapshot_root / ".git"
            git_backup = None
            if git_dir.exists():
                git_backup = snapshot_root.parent / f".__git_backup_{snapshot_root.name}"
                if git_backup.exists():
                    shutil.rmtree(git_backup)
                shutil.move(str(git_dir), str(git_backup))
            shutil.rmtree(snapshot_root)
            snapshot_root.mkdir(parents=True)
            if git_backup is not None:
                shutil.move(str(git_backup), str(git_dir))
        else:
            snapshot_root.mkdir(parents=True)

    print("[publish] stage 1/4 — collecting candidates...")
    candidates = _collect_candidates(config, report)
    print(f"[publish]   {report.candidates_total} scanned / {report.included} included / {report.excluded} excluded")

    print("[publish] stage 2/4 — copy + sanitize...")
    _copy_and_sanitize(candidates, snapshot_root, config.get("sanitize_patterns", []), report, args.dry_run)
    print(f"[publish]   copied {report.copied} files ({report.bytes_copied / 1024 / 1024:.1f} MB)")
    print(f"[publish]   sanitized {report.sanitized_files} files")

    print("[publish] stage 3/4 — post-check forbidden patterns...")
    if not args.dry_run:
        _post_check(snapshot_root, config.get("post_check_forbidden", []), report)
    print(f"[publish]   {len(report.post_check_failures)} forbidden-pattern hits")

    print("[publish] stage 4/4 — writing report...")
    report_path = _write_report(report, config, args.dry_run)
    print(f"[publish]   report: {report_path}")

    if report.post_check_failures:
        print(f"\n[publish] [FAIL] POST-CHECK FAILED ({len(report.post_check_failures)} hits) -- see report above")
        if args.strict:
            return 1
        return 0

    print("\n[publish] [OK] snapshot built clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
