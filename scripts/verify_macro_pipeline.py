"""
MACRO capability pipeline verification (2026-05-04).

Coverage:
  Facet 1  test data cleaned (no logic IN ('test driver','x'))
  Facet 2  cycle scheduler enrolment in run_weekly source
  Facet 3  manual trigger button rendered on pages/macro_brief.py
  Facet 4  Brier scorer math (logic_correct / lucky_guess / logic_wrong)
  Facet 5  verify_macro_forecasts shape on empty DB
  Facet 6  generate_reflections_for_macro shape (no model = no LLM call)
  Facet 7  get_recent_macro_briefs returns []
  Facet 8  MACRO-P AUDIT tab renders without exception (AppTest)
  Facet 9  Agent liveness audit script callable
"""
from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _hr(t: str) -> None:
    print(f"\n{'─' * 70}\n{t}\n{'─' * 70}")


# Facet 1 — test data cleaned
_hr("Facet 1 — test data cleaned in AlphaMemory[macro_research]")
from engine.memory import SessionFactory, AlphaMemory
with SessionFactory() as s:
    bad = (
        s.query(AlphaMemory)
         .filter(AlphaMemory.source == "macro_research")
         .filter(AlphaMemory.logic_chain.in_(("test driver", "x")))
         .count()
    )
assert bad == 0, f"{bad} test rows still in DB"
print(f"  test rows in AlphaMemory[macro_research]: {bad}  OK")


# Facet 2 — run_weekly contains macro_research enrollment
_hr("Facet 2 — run_weekly enrolls macro_research")
with open("engine/orchestrator.py", encoding="utf-8") as f:
    src = f.read()
assert "MacroResearchAgent" in src
assert 'name="macro_research"' in src
assert 'name="macro_verification"' in src
assert 'name="macro_reflection"' in src
print("  orchestrator.run_weekly references all 3 macro stages  OK")


# Facet 3 — manual trigger button on macro_brief.py
_hr("Facet 3 — macro_brief.py renders manual trigger button")
with open("pages/macro_brief.py", encoding="utf-8") as f:
    src_brief = f.read()
assert "macro_research_manual_trigger" in src_brief
print("  manual trigger key present  OK")


# Facet 4 — Brier scorer math
_hr("Facet 4 — Brier scorer verdict mapping")
from engine.macro_verification import (
    _build_forecast_distribution, _brier_score, _brier_to_verdict,
)
fd = _build_forecast_distribution("risk-on", 0.85)
b_correct = _brier_score(fd, "risk-on")
b_wrong   = _brier_score(fd, "risk-off")
fd_uniform = _build_forecast_distribution(None, None)
b_uniform = _brier_score(fd_uniform, "risk-on")
print(f"  correct(0.85)={b_correct:.4f} → {_brier_to_verdict(b_correct)}")
print(f"  wrong(0.85)  ={b_wrong:.4f} → {_brier_to_verdict(b_wrong)}")
print(f"  uniform      ={b_uniform:.4f} → {_brier_to_verdict(b_uniform)}")
assert _brier_to_verdict(b_correct) == "logic_correct"
assert _brier_to_verdict(b_wrong)   in ("lucky_guess", "logic_wrong")
assert _brier_to_verdict(b_uniform) == "lucky_guess"


# Facet 5 — verify_macro_forecasts shape
_hr("Facet 5 — verify_macro_forecasts shape on empty DB")
from engine.macro_verification import verify_macro_forecasts
out = verify_macro_forecasts()
assert set(out.keys()) >= {"as_of","n_scanned","n_skipped","n_verified","n_failed","details"}
print(f"  shape OK: scanned={out['n_scanned']} skipped={out['n_skipped']} "
      f"verified={out['n_verified']} failed={out['n_failed']}")


# Facet 6 — generate_reflections_for_macro shape (no model)
_hr("Facet 6 — generate_reflections_for_macro shape (model=None, no LLM call)")
from engine.macro_verification import generate_reflections_for_macro
out_r = generate_reflections_for_macro(model=None)
assert set(out_r.keys()) >= {"as_of","n_eligible","n_written","n_skipped","n_failed","details"}
assert out_r["n_written"] == 0  # no model → cannot LLM-generate
print(f"  shape OK: eligible={out_r['n_eligible']} written={out_r['n_written']}")


# Facet 7 — get_recent_macro_briefs returns []
_hr("Facet 7 — get_recent_macro_briefs returns list")
from engine.macro_verification import get_recent_macro_briefs
briefs = get_recent_macro_briefs(lookback_days=30)
assert isinstance(briefs, list)
print(f"  briefs returned: {len(briefs)}  OK")


# Facet 8 — MACRO-P AUDIT tab renders
_hr("Facet 8 — pages/orchestrator.py AppTest cold (MACRO-P render path)")
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.abspath("pages/orchestrator.py"), default_timeout=240)
at.run()
print(f"  exc={len(at.exception)}")
for e in at.exception[:2]:
    print(f"    {str(e.value)[:200]}")
assert len(at.exception) == 0


# Facet 9 — Agent liveness audit callable
_hr("Facet 9 — Agent liveness audit script callable")
sys.path.insert(0, "scripts")
import audit_agent_liveness
results = audit_agent_liveness.run_audit()
assert isinstance(results, list) and len(results) >= 2
print(f"  audit returned {len(results)} agent records")
for r in results:
    print(f"    {r['agent_id']:18s}  verdict={r['verdict']}  flags={r['flags']}")


print("\n" + "=" * 70)
print("Macro Capability Pipeline verification: 9 / 9 facets PASS")
print("=" * 70)
