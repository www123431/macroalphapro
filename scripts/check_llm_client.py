"""scripts/check_llm_client.py — R4.4 LLM client centralization
ratchet.

R2 audit finding A8: 10 files call the Anthropic SDK directly
without going through engine.llm.call, bypassing the budget
ledger / retry doctrine / workload routing. Migrating all 10 is
multi-day careful refactor with subtle regression risk. Instead:

  1. Snapshot the grandfathered list of existing direct callers.
  2. Pre-commit hook fails if a NEW file imports `anthropic`
     directly without being on the grandfathered list.
  3. Migrations to engine.llm.call are encouraged but not forced.

The ratchet means the gap monotonically shrinks rather than
expands.

Exit codes:
  0  no NEW direct-anthropic callers (only grandfathered files)
  1  a new file imports anthropic without going through
     engine.llm.call — commit blocked; either migrate to the
     central client or update the grandfathered list with
     justification.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# Frozen list 2026-06-04 (R4.4). New entries require explicit edit
# here + a comment justifying why central client doesn't fit.
GRANDFATHERED = {
    # T7 chain extractor — bespoke tool_use force-invoke logic
    "engine/agents/hypothesis_extractor/extractor.py",
    # Decay sentinel narrator — pre-doctrine
    "engine/agents/decay_sentinel/reasoning.py",
    # Research diagnostician — pre-doctrine persona
    "engine/agents/research_diagnostician/diagnostician.py",
    # Wikipedia scraper — narrow tool call
    "engine/data/fetchers/scraper_wikipedia.py",
    # The central provider itself — exempt by definition
    "engine/llm/providers/anthropic_provider.py",
    # Council agent — multi-stage tool use
    "engine/research/agent_council.py",
    # Calibration feedback — periodic, narrow
    "engine/research/calibration_feedback.py",
    # Discovery sub-agents — three callers, batch ingestion
    "engine/research/discovery/binding_proposer.py",
    "engine/research/discovery/llm_feature_extractor.py",
    "engine/research/discovery/paper_extractor.py",
    # Inbox legacy paper flow (deprecated 2026-06-04 R2.x; kept for
    # archaeology — see engine/inbox/paper_fetcher.py header)
    "engine/inbox/paper_scorer.py",
    "engine/inbox/weekly_digest.py",
    # API helper — single LLM-cost-projection call
    "api/routes_paper_chain.py",
    # DA briefing structured-output adapter
    "engine/research_store/da_briefing/structured_output.py",
    # The orchestrator that routes through providers (delegates to
    # anthropic_provider) — exempt by definition
    "engine/llm/call.py",
    "engine/llm/__init__.py",
    # Research tools router — economic check + brief gen helpers
    "api/routes_research_tools.py",
    # Phase-3 PFH economic check (specialized prompt + JSON validation)
    "engine/research/economic_check.py",
    # Hypothesis-generator + mutation-proposer (PFH ideation)
    "engine/research/hypothesis_generator.py",
    "engine/research/mutation_proposer.py",
    # RBG brief generator (pre-doctrine)
    "engine/research/rbg/brief_generator.py",
    # W5-a-B (2026-06-22) — cron LLM health probe. Intentionally
    # bypasses engine.llm.call to avoid double-counting the ~$0.0001
    # health-check cost into the workload ledger. Single ~10-token
    # Haiku ping per cron fire (2/week). Justification:
    # ledger-tracking a per-fire health probe pollutes cost analytics
    # without adding value — the probe IS the cost-saving mechanism,
    # not a billed workload.
    "scripts/cron/check_llm_provider_health.py",
}


_ANTHROPIC_IMPORT = re.compile(r"^\s*(?:from\s+anthropic|import\s+anthropic)\b")


def _all_direct_callers() -> set[str]:
    """Find every .py file that imports anthropic directly. Uses
    git ls-files to ignore generated / vendor paths."""
    out: set[str] = set()
    try:
        listing = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=str(REPO_ROOT), check=True, capture_output=True, text=True,
        ).stdout.splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fall back to filesystem walk if git is unavailable.
        listing = [str(p.relative_to(REPO_ROOT)).replace("\\", "/")
                   for p in REPO_ROOT.rglob("*.py")
                   if "node_modules" not in p.parts
                      and "__pycache__" not in p.parts]

    for rel in listing:
        rel = rel.strip()
        if not rel: continue
        path = REPO_ROOT / rel
        if not path.is_file(): continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if _ANTHROPIC_IMPORT.search(line):
                out.add(rel.replace("\\", "/"))
                break
    return out


def main() -> int:
    callers = _all_direct_callers()
    new_callers = callers - GRANDFATHERED
    if not new_callers:
        print(f"llm-client: {len(callers)} direct anthropic callers, all grandfathered")
        return 0

    print("llm-client: NEW direct anthropic imports detected", file=sys.stderr)
    print("           (these should go through engine.llm.call instead)",
          file=sys.stderr)
    for rel in sorted(new_callers):
        print(f"  - {rel}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Fix options:", file=sys.stderr)
    print("  1. Refactor to use engine.llm.call() — preferred", file=sys.stderr)
    print("     (budget ledger + retry + workload routing for free)", file=sys.stderr)
    print("  2. Add the file to GRANDFATHERED in", file=sys.stderr)
    print("     scripts/check_llm_client.py with a 1-line comment", file=sys.stderr)
    print("     explaining why central client doesn't fit", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
