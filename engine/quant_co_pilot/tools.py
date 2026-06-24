"""
engine/quant_co_pilot/tools.py — 9 fixed-inventory tool implementations.

Pre-registration: docs/spec_quant_co_pilot_decision_lineage_v1.md (id=53) §2.2

Locked tool inventory (NO additions without amend_spec):
  1. read_spec_registry      — query SpecRegistry by spec_id
  2. search_amendments       — fuzzy search amendment_log reasons
  3. read_git_log            — git log for a file path
  4. read_git_blame          — git blame line-pattern
  5. query_p2_rag            — P2 Project History RAG fallback
  6. read_verdict_json       — read data/*/verdict.json
  7. read_memory_file        — full memory markdown file
  8. search_memory_index     — keyword scan MEMORY.md
  9. read_capability_evidence — full capability_evidence markdown

Each tool returns ToolResult(success, data, error_msg).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ToolResult:
    """Standard return shape for all tools."""
    success:   bool
    data:      Any
    error_msg: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Tool descriptions (for ReAct prompt)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
1. read_spec_registry(spec_id: int)
   → {spec_path, status, current_hash, n_trials_contributed, factor_kind, amendment_log[]}

2. search_amendments(reason_substring: str, limit: int = 10)
   → list[{spec_id, kind, reason, n_trials_added, at}]  matched against amendment reason text

3. read_git_log(file_path: str, max_commits: int = 20)
   → list[{commit_hash, author, date, message}]

4. read_git_blame(file_path: str, line_pattern: str)
   → list[{commit_hash, line_no, line_content, author, date}]  matching a regex

5. query_p2_rag(natural_language_query: str)
   → {answer, source_citations[]}  fallback to P2 docs RAG

6. read_verdict_json(verdict_path: str)
   → dict (verdict JSON content); path under data/* (e.g. "data/factor_ensemble_v1/v1_verdict.json")

7. read_memory_file(memory_filename: str)
   → str (full markdown);  filename only e.g. "project_b_plus_prod.md", no path prefix

8. search_memory_index(keyword: str)
   → list[{filename, hook_line}]  scans MEMORY.md index

9. read_capability_evidence(filename: str)
   → str (full markdown); filename only e.g. "factor_ensemble_v1_descriptive_positive_2026-05-09.md"
"""

TOOL_NAMES: tuple[str, ...] = (
    "read_spec_registry",
    "search_amendments",
    "read_git_log",
    "read_git_blame",
    "query_p2_rag",
    "read_verdict_json",
    "read_memory_file",
    "search_memory_index",
    "read_capability_evidence",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _memory_dir() -> Path:
    return Path.home() / ".claude" / "projects" / "c--Users-${USER}-Desktop-intern" / "memory"


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1: read_spec_registry
# ─────────────────────────────────────────────────────────────────────────────


def read_spec_registry(spec_id: int) -> ToolResult:
    try:
        from engine.memory import SessionFactory, SpecRegistry
        spec_id_int = int(spec_id)
        with SessionFactory() as s:
            row = s.query(SpecRegistry).filter(SpecRegistry.id == spec_id_int).first()
            if row is None:
                return ToolResult(success=False, data=None, error_msg=f"spec_id={spec_id_int} not found")
            try:
                ledger = json.loads(row.amendment_log or "[]")
            except Exception:
                ledger = []
            # Order: SUMMARY fields first (always visible even if truncated),
            # then full amendment_log last (may be truncated for long ledgers).
            return ToolResult(success=True, data={
                "spec_id":              row.id,
                "spec_path":            row.spec_path,
                "status":               row.status,
                "n_amendments":         len(ledger),
                "n_trials_contributed": row.n_trials_contributed,
                "factor_kind":          row.factor_kind,
                "retro_registered":     bool(row.retro_registered),
                "current_hash":         row.current_hash,
                "git_blob_hash":        row.git_blob_hash,
                "registered_at":        row.registered_at.isoformat() if row.registered_at else None,
                "last_validated_at":    row.last_validated_at.isoformat() if row.last_validated_at else None,
                "amendment_summary":    [   # short version always visible
                    {"kind": e.get("kind"), "n_trials_added": e.get("n_trials_added"),
                     "reason_excerpt": (e.get("reason") or "")[:80]}
                    for e in ledger
                ],
                "amendment_log_full":   ledger,  # full version, may be truncated by prompt cap
            })
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_spec_registry error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2: search_amendments
# ─────────────────────────────────────────────────────────────────────────────


def search_amendments(reason_substring: str, limit: int = 10) -> ToolResult:
    try:
        from engine.memory import SessionFactory, SpecRegistry
        substr = str(reason_substring).strip().lower()
        if not substr:
            return ToolResult(success=False, data=None, error_msg="reason_substring required")
        out: list[dict] = []
        with SessionFactory() as s:
            rows = s.query(SpecRegistry).all()
            for r in rows:
                try:
                    ledger = json.loads(r.amendment_log or "[]")
                except Exception:
                    ledger = []
                for entry in ledger:
                    reason = (entry.get("reason") or "").lower()
                    if substr in reason:
                        out.append({
                            "spec_id":        r.id,
                            "spec_path":      r.spec_path,
                            "kind":           entry.get("kind"),
                            "reason":         entry.get("reason"),
                            "n_trials_added": entry.get("n_trials_added"),
                            "at":             entry.get("at"),
                            "prev_hash":      entry.get("prev_hash"),
                            "new_hash":       entry.get("new_hash"),
                        })
                        if len(out) >= limit:
                            return ToolResult(success=True, data=out)
        return ToolResult(success=True, data=out)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"search_amendments error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3: read_git_log
# ─────────────────────────────────────────────────────────────────────────────


def read_git_log(file_path: str, max_commits: int = 20) -> ToolResult:
    try:
        max_n = max(1, min(int(max_commits), 100))
        # Windows GBK fix 2026-05-09: explicit utf-8 + errors=replace to handle
        # commit messages with non-ASCII chars (e.g. Chinese commit messages).
        result = subprocess.run(
            ["git", "log", f"-n{max_n}",
             "--pretty=format:%H|%an|%aI|%s",
             "--", str(file_path)],
            capture_output=True, text=True, timeout=10,
            cwd=str(_repo_root()),
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            return ToolResult(success=False, data=None, error_msg=f"git log error: {(result.stderr or '').strip()}")
        if result.stdout is None:
            return ToolResult(success=True, data=[])
        commits = []
        for line in result.stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                commits.append({
                    "commit_hash": parts[0],
                    "author":      parts[1],
                    "date":        parts[2],
                    "message":     parts[3],
                })
        return ToolResult(success=True, data=commits)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_git_log error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4: read_git_blame
# ─────────────────────────────────────────────────────────────────────────────


def read_git_blame(file_path: str, line_pattern: str) -> ToolResult:
    try:
        # Windows GBK fix 2026-05-09: explicit utf-8 + errors=replace.
        result = subprocess.run(
            ["git", "blame", "--line-porcelain", "--", str(file_path)],
            capture_output=True, text=True, timeout=15,
            cwd=str(_repo_root()),
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            return ToolResult(success=False, data=None, error_msg=f"git blame error: {(result.stderr or '').strip()}")
        if result.stdout is None:
            return ToolResult(success=True, data=[])

        try:
            pat = re.compile(str(line_pattern))
        except re.error as exc:
            return ToolResult(success=False, data=None, error_msg=f"invalid line_pattern regex: {exc!s}")

        out: list[dict] = []
        current: dict = {}
        line_no = 0
        for line in result.stdout.splitlines():
            if re.match(r"^[0-9a-f]{40} \d+ \d+", line) or re.match(r"^[0-9a-f]{40} \d+ \d+ \d+", line):
                parts = line.split()
                current = {
                    "commit_hash": parts[0],
                }
                line_no = int(parts[2])
            elif line.startswith("author "):
                current["author"] = line[len("author "):]
            elif line.startswith("author-time "):
                try:
                    import datetime
                    ts = int(line[len("author-time "):])
                    current["date"] = datetime.datetime.utcfromtimestamp(ts).isoformat()
                except Exception:
                    pass
            elif line.startswith("\t"):
                line_content = line[1:]
                if pat.search(line_content):
                    out.append({
                        "commit_hash":  current.get("commit_hash"),
                        "line_no":      line_no,
                        "line_content": line_content,
                        "author":       current.get("author"),
                        "date":         current.get("date"),
                    })
                if len(out) >= 50:  # cap
                    break
        return ToolResult(success=True, data=out)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_git_blame error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5: query_p2_rag
# ─────────────────────────────────────────────────────────────────────────────


def query_p2_rag(natural_language_query: str) -> ToolResult:
    """Fallback to P2 Project History RAG. Best-effort wrapper."""
    try:
        # P2 RAG entry point varies — try common module paths
        try:
            from engine.agents.history_rag.runner import answer_query as _rag_answer
        except ImportError:
            return ToolResult(success=False, data=None,
                              error_msg="P2 RAG module not importable (engine.agents.history_rag.runner)")
        result = _rag_answer(query=str(natural_language_query))
        # Normalize result shape
        if hasattr(result, "answer"):
            answer_text = result.answer
            citations = getattr(result, "source_citations", [])
        elif isinstance(result, dict):
            answer_text = result.get("answer", "")
            citations = result.get("source_citations", [])
        else:
            answer_text = str(result)
            citations = []
        return ToolResult(success=True, data={
            "answer":           answer_text,
            "source_citations": citations,
        })
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"query_p2_rag error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6: read_verdict_json
# ─────────────────────────────────────────────────────────────────────────────


def read_verdict_json(verdict_path: str) -> ToolResult:
    try:
        # Resolve to repo-relative
        path_str = str(verdict_path)
        if path_str.startswith("/") or ":" in path_str[:3]:
            target = Path(path_str)
        else:
            target = _repo_root() / path_str
        # Safety: must be under data/
        try:
            relative = target.resolve().relative_to(_repo_root() / "data")
        except ValueError:
            return ToolResult(success=False, data=None,
                              error_msg=f"verdict_path must be under data/, got {path_str}")
        if not target.exists():
            return ToolResult(success=False, data=None, error_msg=f"verdict file not found: {path_str}")
        if target.suffix.lower() != ".json":
            return ToolResult(success=False, data=None, error_msg=f"verdict_path must be .json, got {target.suffix}")
        try:
            return ToolResult(success=True, data=json.loads(target.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, data=None, error_msg=f"verdict JSON parse error: {exc!s}")
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_verdict_json error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7: read_memory_file
# ─────────────────────────────────────────────────────────────────────────────


def read_memory_file(memory_filename: str) -> ToolResult:
    try:
        fname = str(memory_filename).strip()
        if "/" in fname or "\\" in fname:
            return ToolResult(success=False, data=None,
                              error_msg=f"memory_filename must be filename only (no path), got {fname}")
        if not fname.endswith(".md"):
            return ToolResult(success=False, data=None,
                              error_msg=f"memory_filename must end with .md, got {fname}")
        target = _memory_dir() / fname
        if not target.exists():
            return ToolResult(success=False, data=None, error_msg=f"memory file not found: {fname}")
        content = target.read_text(encoding="utf-8")
        # Cap at ~8KB to keep prompt manageable
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... TRUNCATED at 8KB cap ...]"
        return ToolResult(success=True, data=content)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_memory_file error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8: search_memory_index
# ─────────────────────────────────────────────────────────────────────────────


def search_memory_index(keyword: str) -> ToolResult:
    try:
        kw = str(keyword).strip().lower()
        if not kw:
            return ToolResult(success=False, data=None, error_msg="keyword required")
        index_path = _memory_dir() / "MEMORY.md"
        if not index_path.exists():
            return ToolResult(success=False, data=None, error_msg="MEMORY.md index not found")
        out: list[dict] = []
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("- ") and kw in line.lower():
                # Format: "- [Title](filename.md) — hook"
                m = re.match(r"^-\s*\[([^\]]+)\]\(([^)]+)\)\s*[—\-]?\s*(.*)", line)
                if m:
                    title, filename, hook = m.group(1), m.group(2), m.group(3)
                    out.append({"title": title, "filename": filename, "hook_line": hook})
                else:
                    out.append({"title": line.strip(), "filename": "?", "hook_line": ""})
                if len(out) >= 30:
                    break
        return ToolResult(success=True, data=out)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"search_memory_index error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 9: read_capability_evidence
# ─────────────────────────────────────────────────────────────────────────────


def read_capability_evidence(filename: str) -> ToolResult:
    try:
        fname = str(filename).strip()
        if "/" in fname or "\\" in fname:
            return ToolResult(success=False, data=None,
                              error_msg=f"filename must be filename only (no path), got {fname}")
        if not fname.endswith(".md"):
            return ToolResult(success=False, data=None,
                              error_msg=f"filename must end with .md, got {fname}")
        target = _repo_root() / "docs" / "capability_evidence" / fname
        if not target.exists():
            return ToolResult(success=False, data=None, error_msg=f"capability_evidence not found: {fname}")
        content = target.read_text(encoding="utf-8")
        if len(content) > 12000:
            content = content[:12000] + "\n\n[... TRUNCATED at 12KB cap ...]"
        return ToolResult(success=True, data=content)
    except Exception as exc:
        return ToolResult(success=False, data=None, error_msg=f"read_capability_evidence error: {exc!s}")


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch + registry
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, Any] = {
    "read_spec_registry":       read_spec_registry,
    "search_amendments":        search_amendments,
    "read_git_log":             read_git_log,
    "read_git_blame":           read_git_blame,
    "query_p2_rag":             query_p2_rag,
    "read_verdict_json":        read_verdict_json,
    "read_memory_file":         read_memory_file,
    "search_memory_index":      search_memory_index,
    "read_capability_evidence": read_capability_evidence,
}


def dispatch_tool(action: str, action_input: dict) -> Any:
    """Dispatch a tool call. Per spec §2.2: unknown tool = fail loud (caller handles)."""
    if action not in TOOL_REGISTRY:
        return {"error": f"unknown tool '{action}'; valid: {sorted(TOOL_REGISTRY)}"}
    try:
        result: ToolResult = TOOL_REGISTRY[action](**(action_input or {}))
    except TypeError as exc:
        return {"error": f"tool '{action}' arg mismatch: {exc!s}"}
    except Exception as exc:
        return {"error": f"tool '{action}' raised: {exc!s}"}
    if not result.success:
        return {"error": result.error_msg or "tool returned failure"}
    return {"data": result.data}
