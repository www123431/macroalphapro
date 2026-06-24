"""
engine/path_l/ — Path L Novy-Marx 2013 Gross Profitability Premium v1.

Pre-registration: docs/spec_path_l_profitability_v1.md (id=68 hash 5a2ab1cc)

Gross profitability factor (Novy-Marx 2013 JFE + Asness-Frazzini-Pedersen 2019 QMJ):
  GPA = (Revenue − COGS) / Total Assets
  TTM 4Q sum GP / lagged 4Q-avg TA per Novy-Marx canonical

Top decile LONG / bottom decile SHORT on top-1500 CRSP universe.
60-day post-rdq holding window (consistent with Path D D-PEAD).

Distinct hypothesis from D-PEAD: fundamental quality persistence
(not behavioral underreaction). Tested as ensemble complement to D-PEAD
for portfolio-level decay risk reduction.
"""
from __future__ import annotations
