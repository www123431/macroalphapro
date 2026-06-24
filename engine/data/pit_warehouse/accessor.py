"""engine.data.pit_warehouse.accessor — Tier C L2-1 Phase 2.2.

PITDataAccessor — the sole interface between templates and cached
data. Templates CANNOT read parquet directly (architectural rule
enforced by code review + future audit gate).

Per docs/spec_pit_data_accessor.md sections 3 + 4.2:

  L1 PIT Data Warehouse (parquet files)
                ↓
  L2 SimClock (knows_about predicate)
                ↓
  L3 PITDataAccessor (THIS MODULE) — wraps L1 reads + L2 filter
                ↓
  Template code (only via accessor methods)

ARCHITECTURAL GUARANTEE
=======================
ANY data point with publication timestamp > clock.now is REJECTED
at the accessor level. Templates therefore CANNOT accidentally
introduce look-ahead bias via the accessor API — the worst they can
do is fail to call advance().

Each accessor method documents:
  - Field-level lag (when is this data "known" relative to its
    nominal timestamp)
  - Whether it's auto-lagged (most data) or returned raw (callers
    who NEED current value, like forward returns)

DESIGN PRINCIPLE
================
"If you can read it from the accessor, it's PIT-safe to use right
now." — Senior critique 2026-06-08

Templates should never have to think about PIT lag. They should
think about backtest logic and trust the accessor to give them
data that was knowable as of clock.now.

LOADERS
=======
Module-level lru_cache(1) on each parquet load — parquet read cost
amortizes across dispatches within one process lifetime.

Re-running tests creates new process → fresh load → up-to-date
cache. Production cron also fresh per invocation.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.data.pit_warehouse.simulation_clock import SimClock

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CACHE_DIR = _REPO_ROOT / "data" / "cache"

# Compustat annual public-availability lag: ~120 days post fiscal year end
# (matches existing cross_sec_us_equities template assumption).
_FUNDA_PUBLIC_LAG_DAYS = 120


# ────────────────────────────────────────────────────────────────────
# L1 cache loaders (lru_cache for process-lifetime amortization)
# ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_crsp_msf() -> pd.DataFrame:
    path = _CACHE_DIR / "_crsp_msf_long_history.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"CRSP msf cache missing: {path}")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
    return df


@lru_cache(maxsize=1)
def _load_crsp_delisting() -> pd.DataFrame:
    path = _CACHE_DIR / "_crsp_dsedelist.parquet"
    if not path.is_file():
        return pd.DataFrame(columns=["permno", "dlstdt", "dlret"])
    df = pd.read_parquet(path)
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    df["dlst_month_end"] = df["dlstdt"] + pd.offsets.MonthEnd(0)
    return df


@lru_cache(maxsize=1)
def _load_ccm_link() -> pd.DataFrame:
    path = _CACHE_DIR / "_crsp_ccm_link.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"CCM link cache missing: {path}")
    df = pd.read_parquet(path)
    df["linkdt"]    = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
    df = df[df["linkprim"].isin(["P", "C"])]
    df["gvkey"]  = df["gvkey"].astype(str)
    df["permno"] = df["permno"].astype(int)
    return df


@lru_cache(maxsize=1)
def _load_compustat_funda_pit() -> pd.DataFrame:
    """PIT-correct Compustat funda from comp_pit.pithistdataus —
    MAX qtrsback per (gvkey, datadate) snapshot. Built by
    scripts/extend_compustat_funda_pit_history.py.

    L2-1 Phase 1.6 bitemporal: reads `knowable_at` column added by
    scripts/add_knowable_at_to_funda_pit.py (which JOINs with
    comp.fundq.rdq for the actual release date, with 120d fallback).
    NO application-layer lag arithmetic — accessor just filters
    rows by knowable_at <= clock.now."""
    path = _CACHE_DIR / "_compustat_funda_pit.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Compustat PIT cache missing: {path}. "
            f"Run scripts/extend_compustat_funda_pit_history.py to "
            f"backfill.")
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])
    if "knowable_at" in df.columns:
        # Phase 1.5/1.6 bitemporal-driven: knowable_at column in data
        df["knowable_at"] = pd.to_datetime(df["knowable_at"])
    else:
        # Pre-Phase-1.5 cache: fall back to application-layer +120d
        # (caller should run add_knowable_at_to_funda_pit.py to upgrade)
        logger.warning("Compustat PIT cache lacks knowable_at column; "
                          "using +120d fallback. Run "
                          "scripts/add_knowable_at_to_funda_pit.py.")
        df["knowable_at"] = df["datadate"] + pd.Timedelta(
            days=_FUNDA_PUBLIC_LAG_DAYS)
    return df


@lru_cache(maxsize=1)
def _load_compustat_funda_legacy() -> pd.DataFrame:
    """Legacy (latest-restated) Compustat funda. Kept for parity
    comparison only — templates should NEVER use this for backtests
    in production. B0 silent bug source (restatement look-ahead).

    Legacy cache has NO knowable_at column — always uses application-
    layer +120d approximation."""
    path = _CACHE_DIR / "_compustat_funda_long_history.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Legacy funda cache missing: {path}")
    df = pd.read_parquet(path)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["knowable_at"] = df["datadate"] + pd.Timedelta(
        days=_FUNDA_PUBLIC_LAG_DAYS)
    return df


@lru_cache(maxsize=1)
def _load_sp500_constituents() -> pd.DataFrame:
    """PIT S&P 500 constituents from crsp.dsp500list. Built by
    scripts/extend_sp500_constituents_pit.py."""
    path = _CACHE_DIR / "_sp500_constituents_pit.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"S&P 500 PIT cache missing: {path}. "
            f"Run scripts/extend_sp500_constituents_pit.py.")
    df = pd.read_parquet(path)
    df["permno"] = df["permno"].astype(int)
    df["start"]  = pd.to_datetime(df["start"])
    df["ending"] = pd.to_datetime(df["ending"])
    return df


# ────────────────────────────────────────────────────────────────────
# L3 PITDataAccessor — the unified template-facing API
# ────────────────────────────────────────────────────────────────────
class PITDataAccessor:
    """The only sanctioned way for templates to read data. Internally
    PIT-filters everything against the wrapped SimClock.

    Pattern:
      clock = SimClock(start, end)
      accessor = PITDataAccessor(clock)
      while clock.now <= end:
          mc = accessor.mktcap_panel(lagged=True)
          ...
          clock.advance("1ME")
    """

    def __init__(self, clock: SimClock,
                  *, funda_source: str = "pit",
                  contract=None):
        """
        funda_source: "pit" (default — uses _compustat_funda_pit.parquet,
                       PIT-clean) or "legacy" (uses latest-restated cache;
                       FOR PARITY COMPARISON ONLY, raises if used in
                       production templates)
        contract: optional TemplateContract whose required_data_shape
                   drives auto-coercion. Phase 6 piece 2 (2026-06-08):
                   when contract declares comp_pit.funda frequency=
                   "annual", accessor.funda_pit_panel filters PIT
                   quarterly rows down to fiscal year-end cohort.
                   None = no coercion (backward compat).
        """
        self._clock = clock
        if funda_source not in {"pit", "legacy"}:
            raise ValueError(
                f"funda_source must be 'pit' or 'legacy'; got "
                f"{funda_source!r}")
        self._funda_source = funda_source
        self._contract = contract

    @property
    def clock(self) -> SimClock:
        return self._clock

    # ── CRSP price layer ───────────────────────────────────────
    def mktcap_panel(
        self,
        *,
        lagged: bool = True,
        window: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Returns (month_end × permno) mktcap matrix.

        lagged=True (DEFAULT, recommended): returns mktcap shifted
          forward by 1 month so that lookup at month_end t returns
          mktcap @ t-1 (PIT-correct for universe selection at rebal
          time t — same-month mktcap is determined by close at t
          which we then trade on).
        lagged=False: returns current month mktcap. ONLY for ex-post
          analysis (e.g., return computation for portfolio formed at
          t-1, realized at t).
        window: optional (start, end) ts tuple to slice rows.
        Auto-filtered to clock.now.
        """
        msf = _load_crsp_msf()
        mc = msf.pivot(index="month_end", columns="permno",
                          values="mktcap")
        if lagged:
            mc = mc.shift(1)
        # Filter by clock.now
        mc = mc.loc[mc.index <= self._clock.now]
        if window is not None:
            ws, we = window
            mc = mc.loc[(mc.index >= pd.Timestamp(ws))
                          & (mc.index <= pd.Timestamp(we))]
        return mc

    def returns_panel(
        self,
        *,
        window: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """(month_end × permno) monthly returns. Auto-filtered to
        clock.now."""
        msf = _load_crsp_msf()
        rets = msf.pivot(index="month_end", columns="permno",
                            values="ret")
        rets = rets.loc[rets.index <= self._clock.now]
        if window is not None:
            ws, we = window
            rets = rets.loc[(rets.index >= pd.Timestamp(ws))
                              & (rets.index <= pd.Timestamp(we))]
        return rets

    def delisting_returns(self) -> pd.DataFrame:
        """(permno, dlst_month_end, dlret) — PIT-clean by
        construction. Auto-filtered to clock.now."""
        d = _load_crsp_delisting()
        d = d.loc[d["dlst_month_end"] <= self._clock.now]
        return d

    # ── Compustat fundamental layer ────────────────────────────
    # ── Phase 6 piece 2: data-shape coercion ───────────────────
    def coerce_funda_to_contract(self, funda: pd.DataFrame) -> pd.DataFrame:
        """Public API (Phase 6 piece 3): callers outside accessor can
        invoke the same coercion logic. Used by cross_sec template
        to replace its tactical B-fix legacy INNER JOIN with the
        architecturally-correct contract-driven path."""
        return self._coerce_funda_to_contract(funda)

    def _coerce_funda_to_contract(self, funda: pd.DataFrame) -> pd.DataFrame:
        """If self._contract has a comp_pit.funda DataShapeRequirement,
        apply the coercion. Currently supported:

          frequency="annual" + aggregation="fy_total":
            INNER JOIN with legacy comp.funda's (gvkey, datadate)
            key set → filter PIT to fiscal year-end cohort. (For
            aggregation, balance sheet fields are already correctly
            valued at FY-end; income statement fields are the comp_pit
            qh-quarterly values at FY-end which Compustat reports as
            fiscal-year aggregates for the annual datadate row.)

          frequency="quarterly" or no contract:
            pass through (PIT raw IS quarterly).

        Returns the (possibly filtered) DataFrame. Cohort filter is
        idempotent — re-applying produces same result.
        """
        if self._contract is None:
            return funda
        try:
            from engine.agents.strengthener.templates._template_contract import (
                DataShapeRequirement,
            )
        except ImportError:
            return funda
        shapes = getattr(self._contract, "required_data_shape", ())
        funda_shape = next(
            (s for s in shapes if s.source == "comp_pit.funda"), None,
        )
        if funda_shape is None:
            return funda
        if funda_shape.frequency != "annual":
            # Quarterly / other: pass through
            return funda

        # Annual frequency requested: filter to FY-end rows
        # via legacy comp.funda key set (canonical annual cohort)
        try:
            legacy = _load_compustat_funda_legacy()
        except FileNotFoundError:
            logger.warning(
                "Phase 6 coercion: legacy comp.funda missing — "
                "cannot apply FY-end filter. Returning quarterly "
                "PIT data (cohort mismatch risk per the contract).")
            return funda
        keys = legacy[["gvkey", "datadate"]].drop_duplicates()
        n_before = len(funda)
        coerced = funda.merge(keys, on=["gvkey", "datadate"],
                                 how="inner")
        logger.debug(
            "Phase 6 annual-coerce: %d quarterly → %d FY-end rows",
            n_before, len(coerced),
        )
        return coerced

    def funda_pit_panel(
        self,
        field: str,
        *,
        window: Optional[tuple] = None,
    ) -> pd.DataFrame:
        """Returns (month_end × permno) panel of a Compustat field,
        PIT-correct (first-report value), with CCM link applied +
        120d public-availability lag baked in.

        Implementation: for each (permno, month_end) in the window,
        find the most recent funda row whose knowable_at <= month_end
        — that is the latest data the market knew about.
        """
        if self._funda_source == "legacy":
            funda = _load_compustat_funda_legacy()
        else:
            funda = _load_compustat_funda_pit()
        if field not in funda.columns:
            raise ValueError(
                f"field {field!r} not in funda columns: "
                f"{list(funda.columns)}")

        # Phase 6 piece 2 (2026-06-08): auto-coerce to declared shape.
        # If contract has comp_pit.funda with frequency="annual",
        # filter PIT quarterly rows to FY-end cohort. Uses legacy
        # comp.funda's (gvkey, datadate) as canonical FY-end key set
        # — same authority the cross_sec template's B-fix uses.
        # After Phase 6 piece 3, cross_sec template's tactical patch
        # is removed and this coercion is the single source of truth.
        funda = self._coerce_funda_to_contract(funda)

        ccm = _load_ccm_link()

        # 1. funda × ccm merge with linkdt window
        fwp = funda[["gvkey", "knowable_at", field]].merge(
            ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
            on="gvkey",
        )
        mask = ((fwp["knowable_at"] >= fwp["linkdt"])
                & (fwp["knowable_at"] <= fwp["linkenddt"]))
        fwp = fwp.loc[mask, ["permno", "knowable_at", field]].copy()

        # 2. Get msf month_ends for the universe + apply clock filter
        msf = _load_crsp_msf()
        if window is not None:
            ws, we = window
            msf_keys = msf.loc[
                (msf["month_end"] >= pd.Timestamp(ws))
                & (msf["month_end"] <= pd.Timestamp(we))
                & (msf["month_end"] <= self._clock.now),
                ["month_end", "permno"],
            ].drop_duplicates()
        else:
            msf_keys = msf.loc[
                msf["month_end"] <= self._clock.now,
                ["month_end", "permno"],
            ].drop_duplicates()

        # 3. Sort by `on=` column GLOBALLY (pandas merge_asof
        # contract; sort by (by, on) is REJECTED). Verified bug
        # 2026-06-08 in cross_sec template.
        msf_keys = msf_keys.sort_values("month_end").reset_index(drop=True)
        fwp = fwp.sort_values("knowable_at").reset_index(drop=True)

        merged = pd.merge_asof(
            msf_keys.rename(columns={"month_end": "as_of"}),
            fwp.rename(columns={"knowable_at": "as_of"}),
            by="permno", on="as_of",
            direction="backward",
            allow_exact_matches=True,
        )
        # Pivot to wide
        return merged.pivot_table(
            index="as_of", columns="permno", values=field,
            aggfunc="last",
        ).rename_axis(index="month_end")

    # ── Universe construction ──────────────────────────────────
    def universe_top_n_by_mktcap(
        self,
        n: int,
        *,
        as_of: Optional[pd.Timestamp] = None,
    ) -> set:
        """Returns set of permnos in top N by LAGGED mktcap as of
        the given timestamp (default: clock.now). Uses lagged mktcap
        so same-month look-ahead is impossible (B1 architectural fix).
        """
        if as_of is None:
            as_of = self._clock.now
        else:
            as_of = pd.Timestamp(as_of)
        if not self._clock.knows_about(as_of):
            raise ValueError(
                f"PIT violation: as_of {as_of} > clock.now "
                f"{self._clock.now}")
        mc = self.mktcap_panel(lagged=True)
        if as_of not in mc.index:
            return set()
        row = mc.loc[as_of].dropna()
        return set(row.nlargest(n).index.astype(int))

    def universe_sp500_constituents(
        self,
        *,
        as_of: Optional[pd.Timestamp] = None,
    ) -> set:
        """Returns set of permnos in S&P 500 as of given timestamp
        (default: clock.now). PIT-correct survivor-bias-free from
        crsp.dsp500list (B2 architectural fix)."""
        if as_of is None:
            as_of = self._clock.now
        else:
            as_of = pd.Timestamp(as_of)
        if not self._clock.knows_about(as_of):
            raise ValueError(
                f"PIT violation: as_of {as_of} > clock.now "
                f"{self._clock.now}")
        sp = _load_sp500_constituents()
        return set(sp.loc[
            (sp["start"]  <= as_of)
            & (sp["ending"] >= as_of),
            "permno",
        ].astype(int))
