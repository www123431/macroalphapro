"""engine.agents.strengthener.templates.cross_sec_us_equities — Tier C-2e.1.

Cross-sectional rank template on the CRSP monthly universe. Second
real template after TSMOM-sector-ETF (C-2b). Covers the canonical
cross-sectional anomalies: size, low-vol, momentum, short-term
reversal.

Scope (C-2e.1 — CRSP-only signals):
  - signal_kind   : cross_sectional_rank
  - universe      : us_equities_top_3000 (top 3000 by mktcap each
                     month, from _crsp_msf_long_history.parquet)
  - signals       : 5 monthly-derivable, all CRSP-only:
                     mktcap        — size factor (long small)
                     vol_12m       — low-vol (long low-vol)
                     ret_12_1      — momentum (long winners,
                                       skip most recent month)
                     ret_6_1       — short-horizon momentum
                     reversal_1m   — short-term reversal (long
                                       prior-month losers)
  - rebal         : monthly (last trading day)
  - weighting     : quintile L/S, equal-weighted within bucket,
                     dollar-neutral (long quintile − short quintile)
  - delisting     : applied via _crsp_dsedelist.parquet (dlret)
  - cost          : 13 bp per round-trip * monthly turnover
  - history       : 1992-2024 effective (CRSP cache 1990-2024,
                     12mo warmup for vol/momentum signals)

NOT in C-2e.1 (deferred to C-2e.2 alongside Compustat backfill):
  - Compustat-derived signals (gross_profitability, book_to_market,
    investment, accruals) — need Compustat history extended back
    from current 2011-2024 cache to 1962+
  - us_equities_sp500 universe (constituents PIT table not cached)

Signal selection — heuristic on spec.signal_inputs:
  The extractor emits free-text signal name hints (e.g.
  "log_market_equity", "vol_12_months", "momentum_12_1"). The
  template introspects + maps to one of the 5 supported signals.
  No match → UNSUPPORTED_SIGNAL verdict (genuine research finding:
  "this signal_kind+inputs combo isn't templated yet").

Note on PIT whitelist + signal_inputs interaction:
  Dispatcher gate #8 PIT whitelist requires signal_inputs to start
  with PIT_CORRECT_SOURCES prefixes (crsp.msf. / compustat.funda. /
  etc.). For cross-sec specs from the current C-1 extractor, the
  LLM emits abstract feature names rather than cache paths. Until
  the extractor prompt is tightened (separate piece), callers can
  build FactorSpecs with prefixed signal_inputs like
  "crsp.msf.derived.vol_12m" to pass the gate. Template logic
  itself is gate-agnostic — it only reads signal_inputs as hints.

Verdict thresholds (mirror tsmom_sector_etf + factor_lab.runner):
  GREEN     |t| >= 1.96
  MARGINAL  1.65 <= |t| < 1.96
  RED       |t| < 1.65
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
_TEMPLATE_VERSION = "v1.0_2026-06-08"

_REPO_ROOT       = Path(__file__).resolve().parents[4]
_CRSP_MSF_PATH   = _REPO_ROOT / "data" / "cache" / "_crsp_msf_long_history.parquet"
_CRSP_DELIST_PATH = _REPO_ROOT / "data" / "cache" / "_crsp_dsedelist.parquet"
# C-2e.2: Compustat fundamentals 1962-2024 (added via
# scripts/extend_compustat_funda_history.py). Used by gp_at,
# book_to_market, at_growth, roe signals.
_COMPUSTAT_FUNDA_PATH = (_REPO_ROOT / "data" / "cache"
                            / "_compustat_funda_long_history.parquet")
_CCM_LINK_PATH        = _REPO_ROOT / "data" / "cache" / "_crsp_ccm_link.parquet"

# PIT lag: public availability of annual funda ≈ 120 days post fiscal
# year end. Applied to datadate when joining funda to CRSP month-end.
_FUNDA_PUBLIC_LAG_DAYS = 120

_UNIVERSE_TOP_N  = 3000
_N_QUINTILES     = 5
# L2-1 Phase 2.5: A-class safety constants from _safety_constants
from engine.agents.strengthener._safety_constants import (
    T_GREEN as _T_GREEN_SAFE,
    T_MARGINAL as _T_MARGINAL_SAFE,
    REPLICATION_T_TOLERANCE as _REPLICATION_T_TOL_SAFE,
    MIN_STOCKS_PER_BUCKET as _MIN_STOCKS_PER_BUCKET,
)
_VOL_LOOKBACK_M  = 12
_MOM_12_LOOKBACK = 12
_MOM_6_LOOKBACK  = 6
_TC_BP_PER_RT    = 13.0       # 13 bp per round-trip
_MIN_OBS_FLOOR   = 60         # never test < 5y regardless of spec.min_obs_months

# Local aliases — same values, semantics from _safety_constants
_T_GREEN    = _T_GREEN_SAFE
_T_MARGINAL = _T_MARGINAL_SAFE

# Commit 1 of the flexibility chain (2026-06-10): signal definitions
# moved to the S-class registry (engine.research.signal_registry).
# Aliases, required fields, formulas, direction, and guards all live
# THERE — adding a new signal is one SignalDefinition entry, zero
# template edits. The names below are thin re-exports kept for
# backward compat (tests + sibling modules import them).
from engine.research.signal_registry import (
    funda_signals as _registry_funda_signals,
    get_signal as _get_signal,
    match_signal_key as _registry_match_signal_key,
    required_columns as _registry_required_columns,
)

# Signals that need the Compustat-linked funda panel (derived).
_COMPUSTAT_SIGNALS = _registry_funda_signals()


def _pick_signal_key(signal_inputs: tuple[str, ...]) -> Optional[str]:
    """Resolve free-text LLM signal hints to a registry key.
    Behavior identical to the pre-registry regex table (the patterns
    moved verbatim into SignalDefinition.aliases; insertion order
    preserves matching priority)."""
    return _registry_match_signal_key(signal_inputs)


# ────────────────────────────────────────────────────────────────────
# Date range parsing — re-use shape from tsmom template
# ────────────────────────────────────────────────────────────────────
def _parse_date_range(s: str) -> tuple[_dt.date, _dt.date]:
    if ":" not in s:
        raise ValueError(f"date_range must contain ':': {s!r}")
    a, b = s.split(":", 1)
    start = _dt.date.fromisoformat(f"{a.strip()}-01")
    end_ts = pd.Timestamp(f"{b.strip()}-01") + pd.offsets.MonthEnd(0)
    return start, end_ts.date()


# ────────────────────────────────────────────────────────────────────
# Data loaders (cached after first call — single process lifetime)
# ────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_crsp_msf() -> pd.DataFrame:
    """Load + index CRSP monthly stock file. Cached for process
    lifetime to amortize parquet read cost across multiple
    dispatches."""
    if not _CRSP_MSF_PATH.is_file():
        raise FileNotFoundError(f"CRSP msf cache missing: {_CRSP_MSF_PATH}")
    df = pd.read_parquet(_CRSP_MSF_PATH)
    df["date"] = pd.to_datetime(df["date"])
    # Sort + ensure month-end index for clean resampling. Adjust date
    # to month-end so panel pivot aligns cleanly.
    df["month_end"] = df["date"] + pd.offsets.MonthEnd(0)
    return df


@lru_cache(maxsize=1)
def _load_crsp_delisting() -> pd.DataFrame:
    """Load CRSP delisting events. Each row: permno + dlstdt +
    dlret (delisting return — earned in the month of delist)."""
    if not _CRSP_DELIST_PATH.is_file():
        return pd.DataFrame(columns=["permno", "dlstdt", "dlret"])
    df = pd.read_parquet(_CRSP_DELIST_PATH)
    df["dlstdt"] = pd.to_datetime(df["dlstdt"])
    df["dlst_month_end"] = df["dlstdt"] + pd.offsets.MonthEnd(0)
    return df


_COMPUSTAT_FUNDA_PIT_PATH = (_REPO_ROOT / "data" / "cache"
                                / "_compustat_funda_pit.parquet")


@lru_cache(maxsize=1)
def _load_compustat_funda() -> pd.DataFrame:
    """L2-1 Phase 3.1 + Phase 6 piece 3 (2026-06-08): PIT-FIRST +
    accessor contract-driven cohort coercion.

    PREFERS PIT cache (_compustat_funda_pit.parquet); falls back
    to legacy latest-restated cache with WARNING (B0 silent bug
    surface).

    PHASE 6 PIECE 3: cohort coercion (filter to FY-end) now
    delegated to PITDataAccessor.coerce_funda_to_contract via
    cross_sec_us_equities's TemplateContract.required_data_shape
    declaration. Replaces the prior tactical B-fix (in-template
    legacy-keys INNER JOIN). Single source of truth for cohort
    coercion = accessor layer.

    Architectural guarantee: if cross_sec's contract declares
    frequency="annual", coercion happens. If we ever change to
    frequency="quarterly", coercion auto-skips. No code change
    needed — declarative.

    Lazy + cached for process lifetime."""
    # Phase 3.1 PIT-first
    if _COMPUSTAT_FUNDA_PIT_PATH.is_file():
        df = pd.read_parquet(_COMPUSTAT_FUNDA_PIT_PATH)
        df["datadate"] = pd.to_datetime(df["datadate"])
        if "knowable_at" in df.columns:
            df["public_date"] = pd.to_datetime(df["knowable_at"])
        else:
            df["public_date"] = df["datadate"] + pd.Timedelta(
                days=_FUNDA_PUBLIC_LAG_DAYS)

        # Phase 6 piece 3: contract-driven cohort coercion.
        # cross_sec_us_equities's contract declares comp_pit.funda
        # frequency="annual" → accessor filters PIT quarterly rows
        # to FY-end cohort. Replaces prior tactical B-fix INNER JOIN.
        try:
            from engine.data.pit_warehouse import (
                SimClock, PITDataAccessor,
            )
            from engine.agents.strengthener.templates._template_contract import (
                CONTRACT_REGISTRY,
            )
            # Dummy clock — coerce method does NOT use clock.now
            # (it filters by gvkey, datadate keys, time-independent)
            dummy_clock = SimClock(start="1970-01-01",
                                      end="2100-12-31")
            contract = CONTRACT_REGISTRY.get("cross_sec_us_equities")
            if contract is not None:
                accessor = PITDataAccessor(dummy_clock,
                                              contract=contract)
                n_before = len(df)
                df = accessor.coerce_funda_to_contract(df)
                logger.info(
                    "Phase 6 piece 3 cohort coerce: %d → %d rows "
                    "(quarterly → annual via accessor + contract)",
                    n_before, len(df),
                )
        except Exception:
            # Defensive: if accessor wiring fails, fall back to
            # legacy in-template B-fix. Logged to surface the issue.
            logger.exception(
                "Phase 6 contract coercion failed; falling back to "
                "tactical legacy INNER JOIN. Investigate.")
            if _COMPUSTAT_FUNDA_PATH.is_file():
                legacy = pd.read_parquet(_COMPUSTAT_FUNDA_PATH)
                legacy["datadate"] = pd.to_datetime(legacy["datadate"])
                fy_end_keys = (legacy[["gvkey", "datadate"]]
                                 .drop_duplicates())
                df = df.merge(fy_end_keys, on=["gvkey", "datadate"],
                                how="inner")
        return df

    # B0 fallback: legacy latest-restated (KNOWN silent bug)
    logger.warning("PIT cache missing — falling back to legacy "
                      "latest-restated comp.funda (B0 silent bug "
                      "surface). Run scripts/extend_compustat_funda_"
                      "pit_history.py + scripts/add_knowable_at_to_"
                      "funda_pit.py to upgrade.")
    if not _COMPUSTAT_FUNDA_PATH.is_file():
        raise FileNotFoundError(
            f"Compustat long-history cache missing: "
            f"{_COMPUSTAT_FUNDA_PATH}. Run "
            f"`python scripts/extend_compustat_funda_history.py` "
            f"to backfill."
        )
    df = pd.read_parquet(_COMPUSTAT_FUNDA_PATH)
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["public_date"] = df["datadate"] + pd.Timedelta(
        days=_FUNDA_PUBLIC_LAG_DAYS)
    return df


@lru_cache(maxsize=1)
def _load_ccm_link() -> pd.DataFrame:
    """CRSP-Compustat link table. Each row is a (gvkey, permno)
    link valid over [linkdt, linkenddt)."""
    if not _CCM_LINK_PATH.is_file():
        raise FileNotFoundError(
            f"CCM link cache missing: {_CCM_LINK_PATH}")
    df = pd.read_parquet(_CCM_LINK_PATH)
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    # Treat open-ended links (linkenddt NaT) as "valid through far
    # future" — clip with a sentinel
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2100-01-01"))
    # Filter to primary links only (LinkPrim P or C), as is standard
    df = df[df["linkprim"].isin(["P", "C"])]
    # permno comes as int from CRSP; ensure gvkey is str for joins
    df["gvkey"] = df["gvkey"].astype(str)
    df["permno"] = df["permno"].astype(int)
    return df


def _build_lagged_funda_panel(
    msf_window:    pd.DataFrame,
    funda:         pd.DataFrame,
    ccm:           pd.DataFrame,
    cols_wanted:   list[str],
) -> pd.DataFrame:
    """For each (month_end, permno) in msf_window's universe, look up
    the MOST RECENT Compustat funda row that was PUBLIC by month_end
    (datadate + 120d <= month_end), via CCM link. Returns a LONG-form
    DataFrame: month_end | permno | <cols_wanted columns>.

    Implementation:
      1. CCM-link merge funda with permno (asof-join on linkdt overlap)
      2. As-of merge funda + msf month_ends so each (permno, t) gets
         the funda row with the most recent public_date <= t

    This is O(N log N) on funda rows and runs in ~3-10s on the full
    universe (572K funda rows × ~20-30K active permnos per month).
    """
    # 1. Funda × CCM merge (gvkey → permno) restricted to link window
    funda_with_permno = funda.merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="gvkey", how="inner",
    )
    # Funda row's public_date must fall inside the link's valid window
    mask = ((funda_with_permno["public_date"] >= funda_with_permno["linkdt"])
            & (funda_with_permno["public_date"] <= funda_with_permno["linkenddt"]))
    funda_with_permno = funda_with_permno.loc[mask].copy()

    # Keep only requested signal-relevant columns. merge_asof
    # requires the `on` column to be sorted GLOBALLY on both sides
    # (sort by `by` first then `on` is rejected with "left keys must
    # be sorted"); `by` filters subsets AFTER the asof search.
    keep_cols = ["permno", "public_date"] + cols_wanted
    funda_with_permno = (funda_with_permno[keep_cols]
                            .sort_values("public_date")
                            .reset_index(drop=True))

    # 2. As-of merge with msf month_ends (left = msf, right = funda)
    msf_keys = (msf_window[["month_end", "permno"]]
                  .drop_duplicates()
                  .sort_values("month_end")
                  .reset_index(drop=True))

    merged = pd.merge_asof(
        msf_keys.rename(columns={"month_end": "as_of"}),
        funda_with_permno.rename(columns={"public_date": "as_of"}),
        by="permno", on="as_of",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.rename(columns={"as_of": "month_end"})
    return merged


# ────────────────────────────────────────────────────────────────────
# Signal computation
# ────────────────────────────────────────────────────────────────────
def _build_fundamental_signal(
    msf_window: pd.DataFrame,
    signal_key: str,
) -> pd.DataFrame:
    """Build a Compustat-derived cross-sectional signal panel from the
    registry definition. Returns (month_end × permno) DataFrame of
    RAW signal values (direction NOT yet applied — _build_signal
    orients centrally from SignalDefinition.direction).

    The PIT wall: every row in `merged` is already PIT-lagged by
    _build_lagged_funda_panel (knowable_at as-of merge). Registry
    formulas can only transform columns of that pre-lagged slice —
    they have no path to raw funda and physically cannot look ahead.

    Lazy-loads Compustat funda + CCM link via lru_cache. Raises
    FileNotFoundError if the long-history cache hasn't been
    backfilled (caller maps to DATA_ERROR verdict).
    """
    sdef = _get_signal(signal_key)
    if sdef is None or sdef.kind != "funda_long":
        raise ValueError(f"unsupported fundamental signal {signal_key!r}")

    funda = _load_compustat_funda()
    ccm   = _load_ccm_link()

    cols_needed = _registry_required_columns(signal_key)

    # Slice funda to a window that bounds the msf_window — keep one
    # extra year of pre-window funda so as-of merge can find lags
    msf_start = msf_window["month_end"].min()
    msf_end   = msf_window["month_end"].max()
    funda_slice = funda[
        (funda["public_date"] >= msf_start - pd.Timedelta(days=365 * 3)) &
        (funda["public_date"] <= msf_end + pd.Timedelta(days=365))
    ]
    merged = _build_lagged_funda_panel(
        msf_window=msf_window, funda=funda_slice, ccm=ccm,
        cols_wanted=cols_needed,
    )

    # Manifest-declared CRSP mktcap merge (e.g. book_to_market)
    if sdef.needs_mktcap_merge:
        mc = msf_window[["month_end", "permno", "mktcap"]]
        merged = merged.merge(mc, on=["month_end", "permno"], how="left")

    # Registry formula → RAW economic value
    merged["signal"] = sdef.formula(merged)

    # Drop inf; apply manifest-declared non-negativity guard (e.g.
    # B/M and ROE are meaningless for negative book equity)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    if sdef.nonneg_guard_field:
        merged.loc[merged[sdef.nonneg_guard_field] <= 0, "signal"] = np.nan

    # Pivot to wide: month_end × permno → signal (RAW)
    return merged.pivot_table(index="month_end", columns="permno",
                                  values="signal", aggfunc="last")


def _build_signal(
    panel:      pd.DataFrame,    # date × permno → ret / mktcap / log_prc
    signal_key: str,
) -> pd.DataFrame:
    """Build the per-date per-permno signal score from the registry
    definition. Returns a DataFrame indexed by month_end,
    columns=permnos, values=signal scores. Higher score = LONG SIDE.

    Commit 1 of the flexibility chain: the per-signal if-chain is
    gone. Formulas return the RAW economic value; orientation is
    applied HERE from SignalDefinition.direction — the manifest SAYS
    which side is long, instead of sign conventions hiding inside
    formula bodies (silent-sign-error protection).
    """
    from types import SimpleNamespace
    sdef = _get_signal(signal_key)
    if sdef is None:
        raise ValueError(f"unknown signal_key {signal_key!r}")

    if sdef.kind == "crsp_panel":
        rets = panel.pivot(index="month_end", columns="permno",
                              values="ret")
        mc   = panel.pivot(index="month_end", columns="permno",
                              values="mktcap")
        raw = sdef.formula(SimpleNamespace(rets=rets, mktcap=mc))
    elif sdef.kind == "funda_long":
        raw = _build_fundamental_signal(panel, signal_key)
    else:
        raise ValueError(f"unknown signal kind {sdef.kind!r}")

    # Central orientation: long_high keeps sign; long_low negates so
    # the top bucket is always the LONG side downstream.
    return raw if sdef.direction == "long_high" else -raw


# ────────────────────────────────────────────────────────────────────
# Quintile L/S backtest
# ────────────────────────────────────────────────────────────────────
def _quintile_long_short_pnl(
    signal_panel:    pd.DataFrame,   # month_end × permno
    return_panel:    pd.DataFrame,   # month_end × permno (returns FOR NEXT MONTH)
    mktcap_panel:    pd.DataFrame,   # for universe top-N gate
    delist_panel:    pd.DataFrame,   # delist returns by (permno, dlst_month_end)
    top_n:           int,
    tc_bp_per_rt:    float,
    n_buckets:       int = _N_QUINTILES,   # L2-1 Phase 3.0: B-class
) -> tuple[pd.Series, dict]:
    """For each rebal date t:
      1. Universe = top top_n stocks by LAGGED mktcap (excluding NaN
         signal). LAGGED is critical — same-month mktcap is
         determined by the close at t which we then trade on. Using
         lagged mktcap = PIT-correct universe selection.
      2. Form quintiles by signal_panel[t]
      3. Long Q5 (highest signal) − Short Q1 (lowest); equal-weight
      4. PnL[t→t+1] = mean(Q5 next-month returns) − mean(Q1 next-month returns)
         Applies delisting return for any stock that delists in (t, t+1].
      5. TC = round-trip turnover × tc_bp / 10000

    Bug 1 fix (2026-06-08, senior rigor pass):
      Pre-fix used `mktcap_panel.loc[t]` for universe selection.
      mktcap at month-end t is determined by close at t, which we
      trade on at rebal time t. Same-month look-ahead. Fix: shift
      mktcap by 1 month so universe selection at t uses mktcap @ t-1
      (the most recently known month-end close).

    Returns (pnl_series, diagnostics)
    """
    pnl: list[float] = []
    pnl_dates: list[pd.Timestamp] = []
    turnover: list[float] = []
    n_stocks: list[int] = []
    prev_long_set: set = set()
    prev_short_set: set = set()

    # Bug 1 fix: lagged mktcap for universe selection. Same-month
    # mktcap is the post-rebal-close value (look-ahead).
    #
    # SURVIVORSHIP / PIT AUDIT (2026-06-16, GP/A audit)
    # ─────────────────────────────────────────────────
    # The top-N universe selection in this template is LAGGED-PIT,
    # not "non-PIT top-3000" as self-doubt has occasionally claimed
    # (B2 false flag, see docs/capability_evidence/
    # gpa_pit_audit_b2_false_flag_2026-06-16.md).
    #
    # Two layers of defense against look-ahead / survivor bias:
    #   1. `mktcap_panel.shift(1)` below — universe at formation
    #      month t uses mktcap from month t-1, so even same-month
    #      look-ahead is impossible.
    #   2. _crsp_msf_long_history.parquet contains all historical
    #      PERMNOs (1992: 6,317 distinct; 2024: 4,139). Stocks that
    #      delisted between 1990 and today ARE in the panel for the
    #      months they existed. Delisting return applied via the
    #      delist_lookup (line ~ 547) at the right month.
    mktcap_lagged = mktcap_panel.shift(1)

    # Delisting return lookup: {(permno, month_end): dlret}
    delist_lookup = {
        (int(r.permno), r.dlst_month_end): float(r.dlret)
        for r in delist_panel.itertuples(index=False)
        if pd.notna(r.dlret)
    }

    sorted_dates = sorted(signal_panel.index)
    # Need at least t and t+1 in panel
    for i, t in enumerate(sorted_dates[:-1]):
        t_next = sorted_dates[i + 1]
        sig_t = signal_panel.loc[t]
        # Use LAGGED mktcap for universe — same-month would look ahead
        mc_t = mktcap_lagged.loc[t] if t in mktcap_lagged.index else None
        if mc_t is None or mc_t.notna().sum() == 0:
            continue
        # Universe: top N by mktcap, with non-null signal
        universe_mask = mc_t.notna() & sig_t.notna()
        if universe_mask.sum() < top_n // 4:
            # Sparse early-history months — skip
            continue
        # Rank candidates by mktcap, keep top N
        mc_in_universe = mc_t[universe_mask]
        top_perms = mc_in_universe.nlargest(top_n).index
        sig_in_top = sig_t.loc[top_perms]
        sig_in_top = sig_in_top.dropna()
        if len(sig_in_top) < n_buckets * _MIN_STOCKS_PER_BUCKET:
            continue

        # Form n_buckets. qcut handles ties via the rank() workaround.
        # L2-1 Phase 3.0: n_buckets now parameterizable via FactorSpec
        # B-class field (default _N_QUINTILES = 5 → quintile L/S).
        # Caller can pass n_buckets=10 for decile sorting.
        ranks = sig_in_top.rank(method="first")
        try:
            buckets = pd.qcut(ranks, n_buckets, labels=False,
                                duplicates="drop")
        except ValueError:
            continue

        q1_perms = set(sig_in_top.index[buckets == 0])
        q5_perms = set(sig_in_top.index[buckets == n_buckets - 1])
        if len(q1_perms) < _MIN_STOCKS_PER_BUCKET \
                or len(q5_perms) < _MIN_STOCKS_PER_BUCKET:
            continue

        # Realize next-month returns; substitute dlret where the permno
        # delists between (t, t_next]
        def _bucket_ret(perms: set) -> float:
            ret_t_next = return_panel.loc[t_next] if t_next in return_panel.index else None
            if ret_t_next is None:
                return float("nan")
            vals: list[float] = []
            for p in perms:
                # Did this permno delist this month? Substitute dlret
                dl = delist_lookup.get((int(p), t_next))
                if dl is not None:
                    vals.append(dl)
                else:
                    r = ret_t_next.get(p)
                    if pd.notna(r):
                        vals.append(float(r))
            return float(np.mean(vals)) if vals else float("nan")

        r_long  = _bucket_ret(q5_perms)
        r_short = _bucket_ret(q1_perms)
        if not (math.isfinite(r_long) and math.isfinite(r_short)):
            continue

        # Dollar-neutral L/S
        gross = r_long - r_short

        # Turnover: |Δlong| + |Δshort|, normalized to "fraction of book"
        if prev_long_set or prev_short_set:
            chg_long  = len(q5_perms.symmetric_difference(prev_long_set))  / max(len(q5_perms), 1)
            chg_short = len(q1_perms.symmetric_difference(prev_short_set)) / max(len(q1_perms), 1)
            to = chg_long + chg_short
        else:
            to = 2.0   # first month: full position build
        prev_long_set = q5_perms
        prev_short_set = q1_perms

        # Store gross + turnover; cost net computed downstream so
        # multi-cost stress (L2-3) only runs the heavy backtest ONCE
        # and re-stresses the cost subtraction.
        pnl.append(gross)
        pnl_dates.append(t_next)
        turnover.append(to)
        n_stocks.append(len(sig_in_top))

    if not pnl:
        return (pd.Series(dtype=float),
                {"n_months": 0, "avg_turnover": float("nan"),
                 "avg_universe_size": 0,
                 "pnl_gross_series": pd.Series(dtype=float),
                 "turnover_series":  pd.Series(dtype=float),
                 "tc_bp_per_rt":     tc_bp_per_rt})

    idx = pd.DatetimeIndex(pnl_dates)
    pnl_gross = pd.Series(pnl, index=idx)
    turnover_s = pd.Series(turnover, index=idx)
    # Headline net PnL = gross − default-TC. Multi-cost stress uses
    # the raw gross + turnover series to re-stress at other levels.
    pnl_net = pnl_gross - turnover_s * (tc_bp_per_rt / 10_000.0)
    return pnl_net, {
        "n_months":           len(pnl_net),
        "avg_turnover":       float(np.mean(turnover)),
        "avg_universe_size":  float(np.mean(n_stocks)),
        "pnl_gross_series":   pnl_gross,
        "turnover_series":    turnover_s,
        "tc_bp_per_rt":       tc_bp_per_rt,
    }


# ────────────────────────────────────────────────────────────────────
# L2-3 Multi-Cost Stress (2026-06-08)
# ────────────────────────────────────────────────────────────────────
# Default cost levels per Frazzini-Israel-Moskowitz 2018 empirical
# estimates on top-3000 cross-sec L/S:
#   0bp   — gross (no cost) — upper bound; only useful as anchor
#   30bp  — realistic cost for liquid name (top ADV decile)
#   60bp  — population-weighted cost for top-3000 universe
#   80bp  — bottom-decile (tail) realistic cost; survives = robust
COST_STRESS_LEVELS_BP: tuple[float, ...] = (0.0, 30.0, 60.0, 80.0)


def _cost_stress_verdict_from_t(t_stat: float) -> str:
    """Sign-aware verdict for cost stress.

    Differs from _verdict_from_t (abs-based, academic convention) by
    REQUIRING positive t — if cost erodes the alpha so much it flips
    sign-significant negative, that's RED not GREEN.

    Surfaced by L3-2 self_doubt 2026-06-08 on reversal_1m: 80bp Sharpe
    -1.501 (signed t ≈ -3.0) was reported as GREEN under abs-based
    verdict logic. Fix is local to cost stress because at the headline
    level abs(t) is defensible (the published direction may be wrong;
    flip the spec), but in cost stress the question is "does the SAME
    factor under the SAME convention survive?" — sign-flips fail by
    construction.
    """
    if not math.isfinite(t_stat):
        return "RED"
    if t_stat >= _T_GREEN:
        return "GREEN"
    if t_stat >= _T_MARGINAL:
        return "MARGINAL"
    return "RED"


def _compute_cost_stress(
    pnl_gross:        pd.Series,
    turnover_series:  pd.Series,
    cost_levels_bp:   tuple[float, ...] = COST_STRESS_LEVELS_BP,
) -> dict[str, dict]:
    """Re-compute Sharpe + NW t-stat + verdict at multiple cost
    levels using the SAME underlying gross PnL + turnover. Only the
    cost subtraction differs — the heavy backtest math runs once.

    Returns dict keyed by cost_level_bp_int → metrics dict containing
    sharpe, nw_t_stat, ann_return, ann_vol, verdict.
    """
    from engine.research.ablation.metrics import (
        annualized_sharpe, newey_west_sharpe_se,
    )
    out: dict[str, dict] = {}
    for bp in cost_levels_bp:
        net = pnl_gross - turnover_series * (bp / 10_000.0)
        if len(net.dropna()) < 12:
            out[f"{int(bp)}bp"] = {
                "sharpe":     None,
                "nw_t_stat":  None,
                "ann_return": None,
                "ann_vol":    None,
                "verdict":    "INSUFFICIENT_HISTORY",
            }
            continue
        sharpe = annualized_sharpe(net)
        se     = newey_west_sharpe_se(net)
        if (not math.isfinite(sharpe) or not math.isfinite(se)
                or se <= 0):
            t = float("nan")
        else:
            t = sharpe / se
        ann_ret = float(net.mean()) * 12.0
        ann_vol = float(net.std(ddof=1)) * math.sqrt(12.0)
        out[f"{int(bp)}bp"] = {
            "sharpe":     float(sharpe) if math.isfinite(sharpe) else None,
            "nw_t_stat":  float(t)     if math.isfinite(t)     else None,
            "ann_return": ann_ret      if math.isfinite(ann_ret) else None,
            "ann_vol":    ann_vol      if math.isfinite(ann_vol) else None,
            "verdict":    _cost_stress_verdict_from_t(t),
        }
    return out


# ────────────────────────────────────────────────────────────────────
# L2-2 Replication Mode (2026-06-08)
# ────────────────────────────────────────────────────────────────────
# Senior rigor: if hypothesis references a paper, run subsample on
# the paper-window overlap and compare our t to paper-reported t.
# Catches implementation bugs at the rigor level — if you can't
# match the paper in its OWN window, your implementation is wrong
# (not the factor).
#
# Why 0.5σ threshold:
#   Sharpe SE per Lo 2002 / Andrews 1991 ≈ sqrt(1+0.5·SR²)/√n_years
#   For SR=0.6, 30 years: SE ≈ 0.13. 0.5σ ≈ 0.07 t-stat range —
#   too tight, would mostly flag noise. Use 0.5σ as 0.5 t-stat
#   units (per Bailey-LdP guidance on replication tolerance).

_REPLICATION_T_TOLERANCE = _REPLICATION_T_TOL_SAFE


def _compute_replication_subsample(
    pnl_net:           pd.Series,
    paper_window:      str,
    paper_reported_t:  Optional[float],
) -> dict:
    """Compute subsample stats on the OVERLAP between our PnL and the
    paper's original window. If paper_reported_t is supplied, compare
    + flag REPLICATED / MISMATCH / NO_BENCHMARK.

    Args:
      pnl_net: net PnL series (already cost-adjusted at default level)
      paper_window: "YYYY-MM:YYYY-MM" from FactorSpec.paper_original_window
      paper_reported_t: paper's |t| (Optional). When None → can compute
                        our subsample t but cannot judge MISMATCH.

    Returns:
      {
        "window_intersection": "YYYY-MM:YYYY-MM" actually used (overlap),
        "n_months_overlap":    int,
        "our_sharpe":          float | None,
        "our_t":               float | None,
        "paper_reported_t":    float | None,
        "t_gap":               float | None,   # |our_t - paper_t|
        "status": "REPLICATED" | "MISMATCH" | "NO_BENCHMARK"
                  | "INSUFFICIENT_OVERLAP" | "NO_DATA",
      }
    """
    from engine.research.ablation.metrics import (
        annualized_sharpe, newey_west_sharpe_se,
    )
    try:
        p_start_yymm, p_end_yymm = paper_window.split(":")
        p_start = pd.Timestamp(f"{p_start_yymm.strip()}-01")
        p_end_ts = pd.Timestamp(f"{p_end_yymm.strip()}-01") + pd.offsets.MonthEnd(0)
    except Exception:
        return {
            "window_intersection": "",
            "n_months_overlap":    0,
            "our_sharpe":          None,
            "our_t":               None,
            "paper_reported_t":    paper_reported_t,
            "t_gap":               None,
            "status":              "NO_DATA",
        }

    # Intersect with our PnL's actual range
    pnl_idx = pnl_net.index
    overlap_mask = (pnl_idx >= p_start) & (pnl_idx <= p_end_ts)
    overlap = pnl_net.loc[overlap_mask].dropna()

    if len(overlap) < 24:   # less than 2 years overlap → insufficient
        actual_start = overlap.index.min().strftime("%Y-%m") if len(overlap) else "—"
        actual_end   = overlap.index.max().strftime("%Y-%m") if len(overlap) else "—"
        return {
            "window_intersection": f"{actual_start}:{actual_end}",
            "n_months_overlap":    len(overlap),
            "our_sharpe":          None,
            "our_t":               None,
            "paper_reported_t":    paper_reported_t,
            "t_gap":               None,
            "status":              "INSUFFICIENT_OVERLAP",
        }

    sharpe = annualized_sharpe(overlap)
    se     = newey_west_sharpe_se(overlap)
    if not (math.isfinite(sharpe) and math.isfinite(se) and se > 0):
        t_ours = None
    else:
        t_ours = sharpe / se

    actual_start = overlap.index.min().strftime("%Y-%m")
    actual_end   = overlap.index.max().strftime("%Y-%m")

    out: dict = {
        "window_intersection": f"{actual_start}:{actual_end}",
        "n_months_overlap":    len(overlap),
        "our_sharpe":          float(sharpe) if math.isfinite(sharpe) else None,
        "our_t":               float(t_ours) if t_ours is not None else None,
        "paper_reported_t":    paper_reported_t,
        "t_gap":               None,
        "status":              "NO_BENCHMARK",
    }

    if paper_reported_t is not None and t_ours is not None:
        # Compare absolute t-stats (sign convention may differ across papers)
        t_gap = abs(abs(t_ours) - abs(paper_reported_t))
        out["t_gap"]  = float(t_gap)
        out["status"] = ("REPLICATED" if t_gap <= _REPLICATION_T_TOLERANCE
                          else "MISMATCH")
    return out


# ────────────────────────────────────────────────────────────────────
# L2-8 Drawdown Metrics (2026-06-08)
# ────────────────────────────────────────────────────────────────────
# Senior quant standard: every backtest report shows max DD + Calmar +
# time-under-water. A "GREEN" with 60% max DD is not deployable.
# Verdict-level info is incomplete without these.

def _compute_drawdown_metrics(pnl: pd.Series) -> dict:
    """Compute max drawdown + Calmar ratio + time-under-water from
    a monthly PnL series. Returns dict of:
      max_drawdown_pct        : worst peak-to-trough decline (negative)
      max_underwater_months   : longest stretch below prior peak
      current_underwater_months : months currently below prior peak (0 if at new high)
      calmar_ratio            : ann_return / |max_drawdown|
      drawdown_at_end_pct     : drawdown at last observation
    NaN-safe; insufficient history → all None.
    """
    if len(pnl.dropna()) < 12:
        return {
            "max_drawdown_pct":           None,
            "max_underwater_months":      None,
            "current_underwater_months":  None,
            "calmar_ratio":               None,
            "drawdown_at_end_pct":        None,
        }
    nav = (1.0 + pnl.fillna(0.0)).cumprod()
    peak = nav.cummax()
    dd = (nav / peak) - 1.0   # negative or zero
    max_dd = float(dd.min())  # most negative

    # Time-under-water: longest consecutive run of dd < 0
    underwater = (dd < 0).astype(int)
    max_uw = 0
    current_uw = 0
    longest_uw = 0
    for v in underwater.values:
        if v == 1:
            current_uw += 1
            longest_uw = max(longest_uw, current_uw)
        else:
            current_uw = 0
    # Trailing-underwater (months below peak as of last obs)
    trailing_uw = 0
    for v in reversed(underwater.values):
        if v == 1:
            trailing_uw += 1
        else:
            break

    ann_ret = float(pnl.mean()) * 12.0
    calmar = (ann_ret / abs(max_dd)) if max_dd < 0 else float("inf")

    return {
        "max_drawdown_pct":           max_dd,
        "max_underwater_months":      int(longest_uw),
        "current_underwater_months":  int(trailing_uw),
        "calmar_ratio":               float(calmar) if math.isfinite(calmar) else None,
        "drawdown_at_end_pct":        float(dd.iloc[-1]),
    }


def _cost_robust_verdict(stress: dict[str, dict],
                          *,
                          robust_at_bp: float = 80.0) -> str:
    """Overall cost-robust verdict.

    L2-3 design: GREEN requires t > 1.96 at the highest stressed
    cost level. MARGINAL = passes mid (30/60bp) but fails high. RED
    if even mid fails.

    Per senior rigor doctrine: a factor that GREEN at 0bp but RED
    at 80bp is a 'paper tiger' — the alpha is spent on TC and
    won't survive live deploy. Reporting only the 13bp result
    (Tier C Layer 1 default) hid this — fix in L2-3.
    """
    key = f"{int(robust_at_bp)}bp"
    if key not in stress:
        return "UNKNOWN"
    return stress[key].get("verdict", "RED")


# ────────────────────────────────────────────────────────────────────
# Verdict mapping
# ────────────────────────────────────────────────────────────────────
def _verdict_from_t(t_stat: float) -> str:
    if not math.isfinite(t_stat):
        return "RED"
    a = abs(t_stat)
    if a >= _T_GREEN:
        return "GREEN"
    if a >= _T_MARGINAL:
        return "MARGINAL"
    return "RED"


# ────────────────────────────────────────────────────────────────────
# Template entry point
# ────────────────────────────────────────────────────────────────────
def template_cross_sec_us_equities(spec: FactorSpec):
    """Tier C-2e.1 template: cross-sectional rank L/S on CRSP top 3000."""
    from engine.agents.strengthener.factor_dispatcher import TemplateResult

    # ── 1. Scope guards ────────────────────────────────────────────
    if spec.signal_kind != "cross_sectional_rank":
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = (f"signal_kind={spec.signal_kind!r} "
                                  "misrouted to cross_sec template"),
            metrics          = {"misroute": True},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if spec.universe != "us_equities_top_3000":
        return TemplateResult(
            verdict          = "UNSUPPORTED_UNIVERSE",
            summary          = (f"universe={spec.universe!r} not "
                                  "supported by cross_sec template "
                                  "(only us_equities_top_3000 in C-2e.1)"),
            metrics          = {"unsupported_universe": spec.universe},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 2. Pick signal from spec.signal_inputs hints ──────────────
    signal_key = _pick_signal_key(spec.signal_inputs)
    if signal_key is None:
        from engine.research.signal_registry import dispatchable_signals
        # flex-3: routing slip + demand ledger (dead-wall doctrine)
        guidance: dict = {}
        try:
            from engine.research.capability_gaps import (
                guidance_unsupported_signal, log_gap,
            )
            guidance = guidance_unsupported_signal(spec.signal_inputs)
            log_gap(hypothesis_id=spec.hypothesis_id, guidance=guidance)
        except Exception:
            logger.exception("cross_sec: gap guidance failed")
        return TemplateResult(
            verdict          = "UNSUPPORTED_SIGNAL",
            summary          = ("no supported signal matched "
                                  f"signal_inputs={list(spec.signal_inputs)}; "
                                  "dispatchable keys: "
                                  + " / ".join(dispatchable_signals())
                                  + ". "
                                  + (guidance.get("next_action", "") or "")[:200]),
            metrics          = {"signal_inputs": list(spec.signal_inputs),
                                 "guidance": guidance},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 2b. Status gate (Commit 2 flexibility chain, 2026-06-10):
    # a PROPOSED signal matches aliases but cannot burn dispatch
    # quota until its verification card is approved. Per the
    # dead-wall doctrine this is a ROUTING SLIP, not a wall — the
    # summary says exactly how to unblock.
    from engine.research.signal_registry import dispatchable_signals
    if signal_key not in dispatchable_signals():
        return TemplateResult(
            verdict          = "SIGNAL_PENDING_APPROVAL",
            summary          = (
                f"signal {signal_key!r} is registered but status="
                "proposed. Next action: review its verification card "
                f"(docs/signal_cards/{signal_key}_card.md — generate "
                "via engine.research.signal_verification."
                "generate_verification_card) then approve via "
                "approve_signal(key, actor=..., reason=...). "
                "Effort: ~5 min review."
            ),
            metrics          = {"signal_key": signal_key,
                                 "gap_class": "TIER_2_PENDING_APPROVAL"},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 3. Parse date range ────────────────────────────────────────
    try:
        start_date, end_date = _parse_date_range(spec.date_range)
    except ValueError as exc:
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = f"date_range parse failed: {exc}",
            metrics          = {"error": str(exc)},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 4. Load CRSP data ──────────────────────────────────────────
    try:
        msf    = _load_crsp_msf()
        delist = _load_crsp_delisting()
    except Exception as exc:
        logger.exception("cross_sec: CRSP load failed")
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = (f"CRSP cache load failed: "
                                  f"{type(exc).__name__}: {exc}"),
            metrics          = {"error": str(exc)[:200]},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 5. Window slice (with warmup buffer for signal calculation) ─
    warmup_months = _VOL_LOOKBACK_M + 2
    fetch_start = pd.Timestamp(start_date) - pd.DateOffset(months=warmup_months)
    panel_window = msf[
        (msf["month_end"] >= fetch_start) &
        (msf["month_end"] <= pd.Timestamp(end_date))
    ]
    if panel_window.empty or panel_window["month_end"].nunique() < 12:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"CRSP window {fetch_start.date()}–"
                                  f"{end_date} has < 12 months of data"),
            metrics          = {
                "n_months_in_cache":
                    int(panel_window["month_end"].nunique()),
            },
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 6. Build signal + run backtest ─────────────────────────────
    try:
        signal_panel = _build_signal(panel_window, signal_key)
    except Exception as exc:
        logger.exception("cross_sec: signal build failed")
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = f"signal build failed: {exc}",
            metrics          = {"error": str(exc)[:200]},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    return_panel = panel_window.pivot(index="month_end", columns="permno",
                                          values="ret")
    mktcap_panel = panel_window.pivot(index="month_end", columns="permno",
                                          values="mktcap")
    # Trim to user date_range (drop warmup rows)
    keep_mask = signal_panel.index >= pd.Timestamp(start_date)
    signal_panel = signal_panel.loc[keep_mask]

    # L2-1 Phase 3.0: B-class params from FactorSpec v2 with
    # parity-preserving fallback to module defaults when None.
    eff_top_n     = spec.universe_size or _UNIVERSE_TOP_N
    eff_n_buckets = spec.n_buckets or _N_QUINTILES
    pnl, diag = _quintile_long_short_pnl(
        signal_panel  = signal_panel,
        return_panel  = return_panel,
        mktcap_panel  = mktcap_panel,
        delist_panel  = delist,
        top_n         = eff_top_n,
        tc_bp_per_rt  = _TC_BP_PER_RT,
        n_buckets     = eff_n_buckets,
    )

    # ── 7. Sample-size gate ────────────────────────────────────────
    n_months = len(pnl)
    if n_months < max(spec.min_obs_months, _MIN_OBS_FLOOR):
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"{n_months} months of PnL < "
                                  f"required {max(spec.min_obs_months, _MIN_OBS_FLOOR)}"),
            metrics          = {
                "n_months":     n_months,
                "min_required": max(spec.min_obs_months, _MIN_OBS_FLOOR),
                "signal":       signal_key,
            },
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 8. Stats ────────────────────────────────────────────────────
    from engine.research.ablation.metrics import (
        annualized_sharpe, newey_west_sharpe_se,
    )
    sharpe   = annualized_sharpe(pnl)
    se_sharpe = newey_west_sharpe_se(pnl)
    if (not math.isfinite(sharpe) or not math.isfinite(se_sharpe)
            or se_sharpe <= 0):
        t_stat = float("nan")
    else:
        t_stat = sharpe / se_sharpe

    ann_ret = float(pnl.mean()) * 12.0
    ann_vol = float(pnl.std(ddof=1)) * math.sqrt(12.0)
    naive_verdict = _verdict_from_t(t_stat)

    # L2-3 Multi-Cost Stress (2026-06-08): re-stress the same
    # gross PnL at 0/30/60/80bp. Overall cost_robust_verdict =
    # verdict at 80bp (Frazzini-Israel-Moskowitz 2018 tail-cost
    # empirical anchor). A factor that passes naive but fails
    # cost-stressed is a "paper tiger" — alpha spent on TC.
    cost_stress = _compute_cost_stress(
        pnl_gross       = diag["pnl_gross_series"],
        turnover_series = diag["turnover_series"],
    )
    cost_robust_verdict = _cost_robust_verdict(cost_stress)

    # L2-8 Drawdown Metrics (2026-06-08): compute on default-net PnL
    # AND on the 80bp-stress PnL. Senior quant standard — max DD +
    # Calmar + time-under-water. Verdict GREEN with 60% max DD is
    # not deployable; this surfaces that.
    drawdown_naive = _compute_drawdown_metrics(pnl)
    pnl_80bp_net = (diag["pnl_gross_series"]
                      - diag["turnover_series"] * (80.0 / 10_000.0))
    drawdown_80bp = _compute_drawdown_metrics(pnl_80bp_net)

    # L2-2 Replication Mode (2026-06-08): if spec has a paper window,
    # run subsample stats on the overlap and compare our t to paper's.
    # MISMATCH → headline verdict gets downgraded (even if full sample
    # is GREEN). Catches implementation bugs at the rigor level.
    replication: dict = {"status": "NOT_APPLICABLE"}
    if spec.paper_original_window:
        replication = _compute_replication_subsample(
            pnl_net          = pnl,
            paper_window     = spec.paper_original_window,
            paper_reported_t = spec.paper_reported_t,
        )

    # Headline verdict = the STRICTER of naive (13bp), 80bp cost-
    # stressed, AND replication. Senior rigor doctrine: report the
    # conservative value, force user to see when alpha hinges on
    # optimistic TC OR fails to replicate the paper benchmark.
    _verdict_severity = {"GREEN": 2, "MARGINAL": 1, "RED": 0,
                          "INSUFFICIENT_HISTORY": 0, "UNKNOWN": 0}
    if (_verdict_severity.get(cost_robust_verdict, 0)
            < _verdict_severity.get(naive_verdict, 0)):
        verdict = cost_robust_verdict
        cost_robust_note = (f" [cost-stress at 80bp dropped naive "
                              f"{naive_verdict} → {cost_robust_verdict}]")
    else:
        verdict = naive_verdict
        cost_robust_note = ""

    # L2-2: MISMATCH downgrades verdict to MARGINAL at most. The
    # implementation might be subtly wrong; user must investigate.
    replication_note = ""
    if replication.get("status") == "MISMATCH":
        if verdict == "GREEN":
            verdict = "MARGINAL"
        replication_note = (
            f" [REPLICATION_MISMATCH: paper t≈{replication.get('paper_reported_t'):.2f} "
            f"vs ours t={replication.get('our_t'):.2f} in overlap "
            f"{replication.get('window_intersection')}]"
        )

    summary = (f"cross_sec[{signal_key}] L/S quintile on "
                 f"top {eff_top_n} CRSP {spec.date_range}: "
                 f"Sharpe={sharpe:.2f}, t={t_stat:.2f}, "
                 f"n={n_months}mo → {verdict}{cost_robust_note}{replication_note}")

    # L2-4 prep (2026-06-08): pass monthly PnL series + turnover via
    # artifacts so factor_verdict_emit can persist as parquet under
    # data/research_store/tier_c_pnl/<spec_hash>_<verdict>.parquet.
    # Required substrate for L2-4 anchor orthogonality (residual
    # regression against anchor library), L2-5 subsample stability,
    # L2-6 attribution. DataFrame columns:
    #   pnl_gross / pnl_net_13bp / pnl_net_80bp / turnover
    # Index: monthly DatetimeIndex (month-end).
    _pnl_series_df = pd.DataFrame({
        "pnl_gross":    diag["pnl_gross_series"],
        "pnl_net_13bp": pnl,
        "pnl_net_80bp": pnl_80bp_net,
        "turnover":     diag["turnover_series"],
    }).dropna(how="all")
    # B.2 (2026-06-09) explicit artifacts contract — see
    # engine.research.lens_helpers. Lenses now read pnl_default_col
    # instead of guessing.
    _artifacts = {
        "pnl_series_df":   _pnl_series_df,
        "pnl_default_col": "pnl_net_13bp",
        "pnl_gross_col":   "pnl_gross",
    }

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "signal":              signal_key,
            "sharpe":              float(sharpe) if math.isfinite(sharpe) else None,
            "nw_t_stat":           float(t_stat) if math.isfinite(t_stat) else None,
            "nw_se_sharpe":        float(se_sharpe) if math.isfinite(se_sharpe) else None,
            "ann_return":          ann_ret if math.isfinite(ann_ret) else None,
            "ann_vol":             ann_vol if math.isfinite(ann_vol) else None,
            "n_months":            n_months,
            "avg_turnover":        diag["avg_turnover"],
            "avg_universe_size":   diag["avg_universe_size"],
            "n_quintiles":         eff_n_buckets,
            "top_n":               eff_top_n,
            "tc_bp_per_rt":        _TC_BP_PER_RT,
            "n_trials":            1,
            # L2-3 senior-rigor additions
            "naive_verdict":         naive_verdict,
            "cost_robust_verdict":   cost_robust_verdict,
            "cost_stress":           cost_stress,
            # L2-8 senior-rigor additions (max DD + Calmar)
            "drawdown_naive":        drawdown_naive,
            "drawdown_80bp":         drawdown_80bp,
            # L2-2 senior-rigor additions (paper replication)
            "replication":           replication,
        },
        artifacts        = _artifacts,
        template_version = _TEMPLATE_VERSION,
    )
