"""
engine/portfolio/ — Portfolio-level orchestration (multi-sleeve, multi-strategy).

Sprint A 2026-05-13: paper_trade_combined.py — stateless daily orchestrator
that integrates the project's 4 production-ready components:
  - K1 BAB ETF       (etf_l1 sleeve)
  - D-PEAD           (ss_sp500 sleeve)
  - Path N reconstitution drift (ss_sp500 sleeve)
  - PQTIX CTA SAA    (cta_defensive sleeve, Path O spec id=73)

Paper trade scope: 0 real capital. Daily signal log only. Per
docs/portfolio_deployment_design_2026-05-13.md Phase 1.

──────────────────────────────────────────────────────────────────────────────
PUBLIC-API RE-EXPORT (fix 2026-05-22).
The single-strategy portfolio engine `engine/portfolio.py` was renamed to
`engine/portfolio_core.py` when this package was created. But the canonical import
path across the codebase is `from engine.portfolio import construct_portfolio, ...`,
and a package shadows a same-named module — so without re-exporting, every such import
breaks (it surfaced via the lazily-imported stop-loss path in portfolio_tracker).
Re-export portfolio_core's full surface here to keep the public API stable; aggregate
the net-exposure caps from engine.config that callers also import via engine.portfolio.
"""
# Full faithful mirror of the renamed module (construct_portfolio, compute_tactical_overlay,
# MAX_WEIGHT/MAX_LEVERAGE/MAX_LONG/MAX_SHORT, CORR_PAIRS, PortfolioWeights, helpers, …).
from engine.portfolio_core import *  # noqa: F401,F403

# Explicit re-export of the names callers depend on AND that portfolio_core actually
# defines (verified). Survives even if portfolio_core later defines __all__; fails loudly
# here rather than at a random call site.
from engine.portfolio_core import (  # noqa: F401
    construct_portfolio,
    compute_tactical_overlay,
    MAX_WEIGHT,
    MAX_LEVERAGE,
    MAX_LONG,
    MAX_SHORT,
)
# NB CORR_PAIRS / MAX_SHORT_EQUITY / PRODUCTION_SIGNAL are imported by a few callers
# (operations_dashboard, a lazy paper_trading path, a _diag script) but are defined NOWHERE
# in engine/ — pre-existing dead references, not resurrected here (cannot re-export a name
# that does not exist). Those call sites were already broken independent of this fix.

# Net-exposure caps live in engine.config; aggregated here because callers import them
# via engine.portfolio (e.g. operations_dashboard).
try:
    from engine.config import MAX_NET, MIN_NET  # noqa: F401
except Exception:  # pragma: no cover - config import must not break the package
    pass
