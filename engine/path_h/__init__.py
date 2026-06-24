"""
engine/path_h/ — Path H 52-Week-High Momentum v1.

Pre-registration: docs/spec_path_h_52wh_v1.md (id=67 hash 7ecbaa3e)

George-Hwang 2004 *JF* anchoring-bias momentum signal:
position_d_long  = top decile of (price_d / max_252d_price)
position_d_short = bottom decile of same

Monthly cross-section rebalance; 21d skip + 126d hold (6 overlapping cohorts).
Single-stock sleeve gates 0.5 / 2.0 per feedback_sleeve_specific_pass_gates.

Distinct hypothesis from D-PEAD: anchoring bias (price-level reference)
vs underreaction-to-earnings (info diffusion). Same top-1500 universe to
isolate mechanism difference.
"""
from __future__ import annotations
