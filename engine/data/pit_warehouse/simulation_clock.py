"""engine.data.pit_warehouse.simulation_clock — Tier C L2-1 Phase 2.

A simulation clock for backtests. Each dispatch instantiates ONE
clock; downstream data access is filtered to clock.now.

Per docs/spec_pit_data_accessor.md section 4.1, the clock has 3
responsibilities:

  1. Track current simulated time (now) — advances through the
     backtest window
  2. Boundary check: knows_about(as_of) returns True iff the
     timestamp is at or before clock.now — the only allowed PIT
     filter primitive
  3. Iterate rebal dates (the simulation time grid) given a freq
     string (pandas offset alias: ME, W-FRI, QE-DEC, etc.)

NOT a clock's job:
  - Reading data (Accessor does that)
  - Filtering data (Accessor uses the clock's knows_about)
  - Computing signals (Templates do that)

Design philosophy (4-layer architecture):
  Clock is L2 in the stack:
    L1 PIT data warehouse (parquets)
    L2 SimClock (this module)
    L3 PITDataAccessor (next module)
    L4 Template contracts + audit gates

This module has ZERO dependencies on pandas/numpy at the interface
level — uses only pd.Timestamp (a thin wrapper over numpy datetime).
Keeps the clock auditable + fast to import.
"""
from __future__ import annotations

import dataclasses as _dc
from typing import Iterator

import pandas as pd


@_dc.dataclass
class SimClock:
    """Backtest simulation clock.

    Each backtest instantiates ONE clock. Templates pass the clock
    to PITDataAccessor; accessor uses it to filter all reads. ANY
    data access where as_of > clock.now is REJECTED at the accessor
    level — this is the architectural guarantee that look-ahead
    bias is impossible inside the L1-L3 stack.

    Mutable on `now` ONLY through advance() (frozen `start`/`end`).
    """

    start:   pd.Timestamp
    end:     pd.Timestamp
    _now:    pd.Timestamp = _dc.field(init=False)

    def __post_init__(self):
        if not isinstance(self.start, pd.Timestamp):
            self.start = pd.Timestamp(self.start)
        if not isinstance(self.end, pd.Timestamp):
            self.end = pd.Timestamp(self.end)
        if self.start > self.end:
            raise ValueError(
                f"SimClock start ({self.start}) > end ({self.end})")
        self._now = self.start

    @property
    def now(self) -> pd.Timestamp:
        """Current simulated time. Read-only; mutate via advance()."""
        return self._now

    def advance(self, dt) -> "SimClock":
        """Advance clock by a pandas-compatible offset.

        Accepts:
          - pd.Timedelta / pd.DateOffset / str (e.g. "1ME", "5D")
          - pd.Timestamp (absolute set, must be > current now)

        Returns self for chaining.
        """
        if isinstance(dt, pd.Timestamp):
            if dt < self._now:
                raise ValueError(
                    f"Cannot rewind: requested {dt} < now {self._now}")
            new_now = dt
        else:
            offset = (pd.tseries.frequencies.to_offset(dt)
                      if isinstance(dt, str) else dt)
            new_now = self._now + offset
        if new_now > self.end:
            new_now = self.end
        self._now = new_now
        return self

    def knows_about(self, as_of) -> bool:
        """True iff `as_of` is at or before clock.now.

        This is THE PIT primitive. Any data point with publication
        timestamp > knows_about returns False = data REJECTED.

        Argument flexibility: accepts pd.Timestamp, datetime.date,
        ISO string. None is treated as "infinitely future" → always
        rejected (defensive).
        """
        if as_of is None:
            return False
        if not isinstance(as_of, pd.Timestamp):
            as_of = pd.Timestamp(as_of)
        return as_of <= self._now

    def reset(self) -> "SimClock":
        """Reset clock to start. Useful for re-running backtests
        without re-instantiating."""
        self._now = self.start
        return self

    def iter_rebal_dates(self, freq: str = "ME") -> Iterator[pd.Timestamp]:
        """Iterate rebal dates from start to end at given pandas
        offset alias. Standard values:
          ME    : month-end
          W-FRI : weekly Friday close
          QE    : quarter-end
          BME   : business month-end

        Does NOT advance the clock — caller is responsible for
        calling advance() at each iteration if they want state
        tracking. iter_rebal_dates is a pure iterator.
        """
        return iter(pd.date_range(self.start, self.end, freq=freq))

    def __repr__(self) -> str:
        return (f"SimClock(start={self.start.date()}, "
                f"end={self.end.date()}, now={self._now.date()})")
