"""
engine/sector_scan_thresholds.py — Pre-trade check threshold constants.

Consumer: pages/sector_scan.py::_pretrade_check()

These are UI-status thresholds (PASS / REVIEW / FAIL pill on the TODAY
header) — operational defaults, not research hypotheses. Author-chosen
values; rationale in the inline comment per constant. Not registered with
SpecRegistry: pre-registration + amendment-ledger overhead is designed for
research methodology, not UI presentation knobs.

If a future need arises to make these supervisor-tunable at runtime, the
right move is to surface them as Settings → not to retroactively wrap them
in SpecRegistry governance.
"""
from __future__ import annotations

# β coverage = (rows with valid 252-day β) / (total active universe rows)
BETA_COVERAGE_REVIEW = 0.85   # below 85% target → REVIEW
BETA_COVERAGE_FAIL   = 0.60   # below 60% floor → FAIL (BAB tertile unstable)

# Largest |portfolio weight| — single-position concentration
MAX_ABS_WEIGHT_REVIEW = 0.25  # construct_portfolio soft cap
MAX_ABS_WEIGHT_FAIL   = 0.35  # hard-cap + 10pp buffer; above = rule break

# |Net exposure| — sum of weights, BAB nominal market-neutral
NET_EXPOSURE_REVIEW = 0.60    # symmetric band around 0; above = REVIEW only
                              # (not FAIL: high net is not a direct rule break)
