"""
engine/path_m/ — Path M Thematic ETF Cross-Section Momentum v1.

Pre-registration: docs/spec_path_m_thematic_momentum_v1.md (id=69 hash a3f50c9f)

Cross-section momentum (Jegadeesh-Titman 1993 + Carhart 1997 canonical 12-1)
on 34 locked thematic ETF universe. Top-3 long / bottom-3 short, monthly
rebalance, Tier 3 thematic TC (5bp roundtrip).

Distinct ETF alpha from K1 BAB (capacity diversifier, leverage constraint)
and Path F VIX TS (vol carry). Tested as project's third ETF PASS candidate
after 11 prior ETF mechanism tests (1 PASS K1, 1 borderline Path F, rest FAIL).

Robustness scout (2026-05-13) confirmed signal:
- ARK-out Sharpe 0.67 ≈ baseline 0.68 (NOT ARK-driven)
- Leg size top-2..7 all PASS gate (Sharpe 0.61-0.75)
- Lookback 9-12mo stable
- 3/3 sub-periods Pre+0.59 / COVID+0.78 / Post+0.84 all > 0.5
"""
from __future__ import annotations
