"""engine/agents/anomaly_sentinel — Anomaly Sentinel agent layer.

The persona definition is in `engine/agents/persona/anomaly_sentinel.py`
(read-only forensic agent for per-ticker questions). This package contains
the OPERATIONAL/ACTION extensions:

  - auto_halt: pre-committed deterministic trigger rules that halt new
    order submission when book health degrades; the missing TEETH layer
    for the persona's hitherto report-only behavior. See module docstring
    for trigger thresholds.
"""
