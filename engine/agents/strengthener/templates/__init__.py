"""engine.agents.strengthener.templates — Tier C-2 factor backtest templates.

Each template is a function (FactorSpec) -> TemplateResult that:
  - validates the spec fits its scope (signal_kind + universe)
  - fetches data via vetted PIT-clean sources (PIT_CORRECT_SOURCES)
  - runs a deterministic backtest (NO LLM)
  - returns metrics + GREEN/MARGINAL/RED verdict

Templates are intentionally INDEPENDENT — no shared base class, no
plugin framework. Each is ~150-300 lines, self-contained, and easy
to audit in isolation. Per [[feedback-no-fear-of-rework-only-unusable-2026-06-01]]:
shared abstraction across templates is premature until we have ≥3
real templates and see the actual overlap.

Per Tier C-2 piece-by-piece plan (2026-06-08):
  C-2b: tsmom_sector_etf — first end-to-end loop, smallest scope
  C-2e: cross_sec_us_equities — biggest (CRSP+Compustat WRDS)
  C-2f: carry_fx_g10
  later: vrp, event_drift
"""
from engine.agents.strengthener.templates.tsmom_sector_etf import (
    template_tsmom_sector_etf,
)
from engine.agents.strengthener.templates.cross_sec_us_equities import (
    template_cross_sec_us_equities,
)

__all__ = [
    "template_tsmom_sector_etf",
    "template_cross_sec_us_equities",
]
