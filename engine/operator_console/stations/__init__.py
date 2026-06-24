"""Operator Console station implementations.

Each station auto-registers at import time. Import this package to
populate the registry before serving API requests.

Stations attached:
    S1 — Paper Ingest    (2026-06-23)
    S3 — FactorSpec Extract (2026-06-23)
    S4 — FORWARD Dispatch (2026-06-23)
    S6 — Verdict View (2026-06-23)
    S7 — PROMOTE 9-gate (MVP — Gates 1+9 wired, 2-8 deferred) (2026-06-23)
    S8 — Rollback (2026-06-23)
    S8b — Doctrine Lock (2026-06-23)

Future:
    S2 — Hypothesis Synthesize
    S5 — ENHANCE Dispatch
"""
from engine.operator_console.stations import s1_paper_ingest          # noqa: F401
from engine.operator_console.stations import s3_factorspec_extract    # noqa: F401
from engine.operator_console.stations import s4_forward_dispatch      # noqa: F401
from engine.operator_console.stations import s6_verdict_view          # noqa: F401
from engine.operator_console.stations import s7_promote_9gate         # noqa: F401
from engine.operator_console.stations import s8_rollback              # noqa: F401
from engine.operator_console.stations import s8b_doctrine_lock        # noqa: F401
