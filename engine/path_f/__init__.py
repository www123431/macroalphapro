"""
engine/path_f/ — Path F VIX Term Structure Carry v1.

Pre-registration: docs/spec_path_f_vix_term_structure_v1.md (id=65)

Cheng 2019 RFS VIX premium via contango/backwardation signal on SVXY.
First ETF spec written under post-Path-E lesson framework:
- TC tier-specific (6bp/event for Tier 2 SVXY)
- Ex-ante stop-loss + cooling-off + winsorize
- Dual-method primary metric (Method A daily TS + Method B trade-time)
- Full 5-gate post-audit framework
"""
from __future__ import annotations
