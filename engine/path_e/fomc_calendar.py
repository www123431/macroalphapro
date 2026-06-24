"""
engine/path_e/fomc_calendar.py — Hardcoded FOMC meeting calendar 2014-2023.

Pre-registration: docs/spec_path_e_pre_fomc_drift_v1.md (id=64) §2.2

8 scheduled FOMC meetings per year × 10 years = 80 events.
2020 emergency meetings (Mar 3 + Mar 15) EXCLUDED per spec §六 to preserve
event-spacing regularity.

Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
Cross-verifiable via FRED rate-change events.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class FomcEvent:
    """One scheduled FOMC meeting.

    meeting_start_date: typically Day 1 of 2-day meeting
    statement_release_date: typically Day 2 of 2-day meeting (announcement at 14:00 ET)
    """
    meeting_start_date:     datetime.date
    statement_release_date: datetime.date


# 8 FOMC meetings per year × 10 years = 80 events.
# Each tuple: (meeting_start, statement_release)
_FOMC_2014_2023_RAW = [
    # 2014 — 8 meetings
    ("2014-01-28", "2014-01-29"),
    ("2014-03-18", "2014-03-19"),
    ("2014-04-29", "2014-04-30"),
    ("2014-06-17", "2014-06-18"),
    ("2014-07-29", "2014-07-30"),
    ("2014-09-16", "2014-09-17"),
    ("2014-10-28", "2014-10-29"),
    ("2014-12-16", "2014-12-17"),
    # 2015
    ("2015-01-27", "2015-01-28"),
    ("2015-03-17", "2015-03-18"),
    ("2015-04-28", "2015-04-29"),
    ("2015-06-16", "2015-06-17"),
    ("2015-07-28", "2015-07-29"),
    ("2015-09-16", "2015-09-17"),
    ("2015-10-27", "2015-10-28"),
    ("2015-12-15", "2015-12-16"),
    # 2016
    ("2016-01-26", "2016-01-27"),
    ("2016-03-15", "2016-03-16"),
    ("2016-04-26", "2016-04-27"),
    ("2016-06-14", "2016-06-15"),
    ("2016-07-26", "2016-07-27"),
    ("2016-09-20", "2016-09-21"),
    ("2016-11-01", "2016-11-02"),
    ("2016-12-13", "2016-12-14"),
    # 2017
    ("2017-01-31", "2017-02-01"),
    ("2017-03-14", "2017-03-15"),
    ("2017-05-02", "2017-05-03"),
    ("2017-06-13", "2017-06-14"),
    ("2017-07-25", "2017-07-26"),
    ("2017-09-19", "2017-09-20"),
    ("2017-10-31", "2017-11-01"),
    ("2017-12-12", "2017-12-13"),
    # 2018
    ("2018-01-30", "2018-01-31"),
    ("2018-03-20", "2018-03-21"),
    ("2018-05-01", "2018-05-02"),
    ("2018-06-12", "2018-06-13"),
    ("2018-07-31", "2018-08-01"),
    ("2018-09-25", "2018-09-26"),
    ("2018-11-07", "2018-11-08"),
    ("2018-12-18", "2018-12-19"),
    # 2019
    ("2019-01-29", "2019-01-30"),
    ("2019-03-19", "2019-03-20"),
    ("2019-04-30", "2019-05-01"),
    ("2019-06-18", "2019-06-19"),
    ("2019-07-30", "2019-07-31"),
    ("2019-09-17", "2019-09-18"),
    ("2019-10-29", "2019-10-30"),
    ("2019-12-10", "2019-12-11"),
    # 2020 — 8 scheduled (2020-03-03 + 2020-03-15 emergency excluded per spec §六)
    ("2020-01-28", "2020-01-29"),
    ("2020-03-17", "2020-03-18"),   # scheduled; originally Mar 17-18, postponed by emergency
    ("2020-04-28", "2020-04-29"),
    ("2020-06-09", "2020-06-10"),
    ("2020-07-28", "2020-07-29"),
    ("2020-09-15", "2020-09-16"),
    ("2020-11-04", "2020-11-05"),
    ("2020-12-15", "2020-12-16"),
    # 2021
    ("2021-01-26", "2021-01-27"),
    ("2021-03-16", "2021-03-17"),
    ("2021-04-27", "2021-04-28"),
    ("2021-06-15", "2021-06-16"),
    ("2021-07-27", "2021-07-28"),
    ("2021-09-21", "2021-09-22"),
    ("2021-11-02", "2021-11-03"),
    ("2021-12-14", "2021-12-15"),
    # 2022
    ("2022-01-25", "2022-01-26"),
    ("2022-03-15", "2022-03-16"),
    ("2022-05-03", "2022-05-04"),
    ("2022-06-14", "2022-06-15"),
    ("2022-07-26", "2022-07-27"),
    ("2022-09-20", "2022-09-21"),
    ("2022-11-01", "2022-11-02"),
    ("2022-12-13", "2022-12-14"),
    # 2023
    ("2023-01-31", "2023-02-01"),
    ("2023-03-21", "2023-03-22"),
    ("2023-05-02", "2023-05-03"),
    ("2023-06-13", "2023-06-14"),
    ("2023-07-25", "2023-07-26"),
    ("2023-09-19", "2023-09-20"),
    ("2023-10-31", "2023-11-01"),
    ("2023-12-12", "2023-12-13"),
]

FOMC_EVENTS_2014_2023: tuple[FomcEvent, ...] = tuple(
    FomcEvent(
        meeting_start_date=datetime.date.fromisoformat(start),
        statement_release_date=datetime.date.fromisoformat(end),
    )
    for start, end in _FOMC_2014_2023_RAW
)

assert len(FOMC_EVENTS_2014_2023) == 80, (
    f"Expected 80 FOMC events (8/year × 10y); got {len(FOMC_EVENTS_2014_2023)}"
)


def get_fomc_events_in_window(
    start_date: datetime.date,
    end_date:   datetime.date,
) -> list[FomcEvent]:
    """Return FOMC events whose statement_release_date is in [start_date, end_date]."""
    return [
        e for e in FOMC_EVENTS_2014_2023
        if start_date <= e.statement_release_date <= end_date
    ]


def get_events_per_year(year: int) -> int:
    """Count FOMC events in a given calendar year (verification helper)."""
    return sum(1 for e in FOMC_EVENTS_2014_2023
               if e.statement_release_date.year == year)
