"""
engine/portfolio/attribution_logger.py — Sprint H v1.0 trade-level forensic log.

Records the DECISION layer's state at each trade time:
  - Which strategy / spec / signal_value / event_trigger / expected_horizon

Enables post-hoc DD root cause analysis without re-building context from
raw data sources (Compustat / CRSP / S&P feed).

DOCTRINE: this module is WRITE-from-decision-layer / READ-by-forensic-layer.
It does NOT call LLM. LLM news summarization happens in a SEPARATE module
(engine.forensic.news_context) that READS this log and never feeds back into
strategy decisions. 0-LLM-in-DECISION preserved.

Spec: docs/spec_per_strategy_attribution_logger_v1.md
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from engine.portfolio.paper_trade_combined import PaperTradeRunResult

logger = logging.getLogger(__name__)


# Strategy → (spec_id, spec_hash_short, expected_horizon_days_default)
# Sourced from engine.strategies registry (truth lives in adapters.py META).
# Kept as a module-level back-compat shim — paper_trade_combined.py imports
# this symbol directly.
from engine.strategies import get_registry as _get_registry
STRATEGY_SPEC_MAP: dict[str, tuple[int, str, int]] = _get_registry().spec_map_dict()

# Deterministic UUID namespace for Sprint H trade IDs
_SPRINT_H_NS = uuid.UUID("1c3e7a4b-aaaa-4444-9999-deadbeefcafe")

ATTRIBUTION_JSONL_PATH = Path("data/paper_trade/attribution_log.jsonl")


@dataclasses.dataclass(frozen=True)
class TradeAttribution:
    """Per-ticker forensic context populated by each strategy at signal time.

    Populated inside each get_*_signal() function. Becomes part of StrategySignal.

    Fields:
      ticker:                  ticker symbol
      side:                    'long' or 'short'
      weight:                  signed weight (+long / -short) at intra-sleeve scale
      signal_value:            raw factor / signal score (None for non-signal strategies)
      event_trigger:           ISO date or descriptor (e.g., '2026-06-12' for rdq)
      expected_horizon_days:   days the position is expected to be held per spec
      notes_json:              per-strategy free-form JSON string
    """
    ticker:                str
    side:                  str
    weight:                float
    signal_value:          Optional[float]
    event_trigger:         Optional[str]
    expected_horizon_days: int
    notes_json:            str = "{}"


@dataclasses.dataclass(frozen=True)
class TradeLogRow:
    """Flat row ready for DB persist + JSONL append. Output of attributions_from_result()."""
    date:                  datetime.date
    trade_id:              str
    strategy_name:         str
    spec_id:               int
    spec_hash_short:       str
    sleeve_id:             str
    ticker:                str
    side:                  str
    weight:                float
    signal_value:          Optional[float]
    event_trigger:         Optional[str]
    expected_horizon_days: int
    is_rebalance_day:      bool
    notes_json:            str


def make_trade_id(date: datetime.date, strategy: str, ticker: str) -> str:
    """Deterministic UUID5 — same (date, strategy, ticker) → same UUID.

    Guarantees idempotency: re-running orchestrator for the same date produces
    the same trade_id for the same trade, enabling UPSERT semantics.
    """
    return str(uuid.uuid5(_SPRINT_H_NS, f"{date.isoformat()}|{strategy}|{ticker}"))


def attributions_from_result(
    result: "PaperTradeRunResult",
    is_rebalance_per_strategy: dict[str, bool],
) -> list[TradeLogRow]:
    """Flatten PaperTradeRunResult's per-strategy attributions into log rows.

    Args:
      result:                    PaperTradeRunResult from run_paper_trade_day()
      is_rebalance_per_strategy: e.g. {'K1_BAB': True, 'D_PEAD': False, ...}

    Returns:
      list[TradeLogRow] — flat rows ready to persist
    """
    rows: list[TradeLogRow] = []
    for sig in result.signals:
        spec_info = STRATEGY_SPEC_MAP.get(sig.strategy_name)
        if spec_info is None:
            logger.warning("Sprint H: no spec map entry for %s; skipping attributions",
                           sig.strategy_name)
            continue
        spec_id, spec_hash_short, _default_horizon = spec_info
        is_rebal = is_rebalance_per_strategy.get(sig.strategy_name, False)

        if not sig.trade_attributions:
            # Strategy didn't populate attributions (legacy or NO_SIGNAL); skip
            continue

        for attr in sig.trade_attributions:
            rows.append(TradeLogRow(
                date                  = result.as_of,
                trade_id              = make_trade_id(result.as_of, sig.strategy_name, attr.ticker),
                strategy_name         = sig.strategy_name,
                spec_id               = spec_id,
                spec_hash_short       = spec_hash_short,
                sleeve_id             = sig.sleeve_id,
                ticker                = attr.ticker,
                side                  = attr.side,
                weight                = attr.weight,
                signal_value          = attr.signal_value,
                event_trigger         = attr.event_trigger,
                expected_horizon_days = attr.expected_horizon_days,
                is_rebalance_day      = is_rebal,
                notes_json            = attr.notes_json,
            ))
    return rows


def persist_attribution_to_db(rows: list[TradeLogRow]) -> int:
    """SQLAlchemy UPSERT on composite PK (date, trade_id). Returns rows persisted.

    Uses session.merge() for portable upsert (SQLite + PostgreSQL).
    """
    if not rows:
        return 0
    from engine.db_models import PaperTradeTradeLog, SessionFactory

    session = SessionFactory()
    n_persisted = 0
    try:
        for row in rows:
            obj = PaperTradeTradeLog(
                date                  = row.date,
                trade_id              = row.trade_id,
                strategy_name         = row.strategy_name,
                spec_id               = row.spec_id,
                spec_hash_short       = row.spec_hash_short,
                sleeve_id             = row.sleeve_id,
                ticker                = row.ticker,
                side                  = row.side,
                weight                = row.weight,
                signal_value          = row.signal_value,
                event_trigger         = row.event_trigger,
                expected_horizon_days = row.expected_horizon_days,
                is_rebalance_day      = row.is_rebalance_day,
                notes_json            = row.notes_json,
            )
            session.merge(obj)
            n_persisted += 1
        session.commit()
        logger.info("Sprint H: persisted %d trade attribution rows to DB", n_persisted)
    except Exception:
        session.rollback()
        logger.exception("Sprint H: DB persist failed; transaction rolled back")
        raise
    finally:
        session.close()
    return n_persisted


def persist_attribution_to_jsonl(
    rows: list[TradeLogRow],
    path: Path = ATTRIBUTION_JSONL_PATH,
) -> int:
    """Append-only JSONL writer.

    Each line = one trade. Idempotency: re-runs append again (DB UPSERT is the
    authoritative store; JSONL is human-readable audit trail).
    """
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            d = dataclasses.asdict(row)
            d["date"] = row.date.isoformat()  # JSON-safe
            f.write(json.dumps(d) + "\n")
    logger.info("Sprint H: appended %d rows to %s", len(rows), path)
    return len(rows)


def query_trade_log(
    date_start:     Optional[datetime.date] = None,
    date_end:       Optional[datetime.date] = None,
    strategy_name:  Optional[str]           = None,
    ticker:         Optional[str]           = None,
    spec_id:        Optional[int]           = None,
) -> pd.DataFrame:
    """Forensic query API. Returns DataFrame for ad-hoc analysis.

    Example DD investigation:
        df = query_trade_log(date_start='2026-06-08', date_end='2026-06-15',
                             strategy_name='D_PEAD')
        # Join with yfinance returns, compute weight × realized → identify worst trade
    """
    from engine.db_models import PaperTradeTradeLog, SessionFactory

    session = SessionFactory()
    try:
        q = session.query(PaperTradeTradeLog)
        if date_start is not None:    q = q.filter(PaperTradeTradeLog.date >= date_start)
        if date_end is not None:      q = q.filter(PaperTradeTradeLog.date <= date_end)
        if strategy_name is not None: q = q.filter(PaperTradeTradeLog.strategy_name == strategy_name)
        if ticker is not None:        q = q.filter(PaperTradeTradeLog.ticker == ticker)
        if spec_id is not None:       q = q.filter(PaperTradeTradeLog.spec_id == spec_id)
        rows = q.all()
        if not rows:
            return pd.DataFrame()
        data = [{c.name: getattr(r, c.name) for c in PaperTradeTradeLog.__table__.columns}
                for r in rows]
        return pd.DataFrame(data)
    finally:
        session.close()
