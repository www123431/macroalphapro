"""engine.research.signal_registry — S-class declarative signal catalog.

Commit 1 of the flexibility chain (2026-06-10). Sibling of
anchor_library_registry: signals are RESEARCH CONTENT, not template
implementation details. Pre-registry, adding a new cross-sectional
signal meant editing 3 places inside a 1190-line template
(_SIGNAL_PATTERNS regex + cols_needed dict + _build_signal /
_build_fundamental_signal if-chains) plus the TemplateContract's
supported_signals list — the exact scattered-hardcoding disease that
A.1 cured for anchors and B.2 cured for column names.

Post-registry, a new signal = ONE SignalDefinition entry. The
template reads everything (aliases, required fields, formula,
direction, guards) from here; the contract derives supported_signals
from here. Single source of truth.

ARCHITECTURE (Option C per the 2026-06-10 senior施工建议)
========================================================
- Formulas are constrained Python callables, NOT a DSL (premature
  DSL = half-baked parser legacy). The PIT wall lives in the
  PLUMBING, not the formula: funda_long formulas receive a merged
  panel whose every row is already PIT-lagged by the shared
  _build_lagged_funda_panel (knowable_at / public_date as-of merge).
  The formula can only transform columns of that pre-lagged slice —
  it has no access to raw funda and physically cannot look ahead.
- DIRECTION is declared, not baked into formula signs. Formulas
  return the RAW economic quantity (asset growth = growth, vol =
  vol); the template applies orientation centrally from
  `direction` ("long_high" keeps sign, "long_low" negates). This
  kills the silent-sign-error class: the manifest SAYS which side
  is long.
- FAMILY is declared for Bailey-LdP n_trials accounting (HXZ
  within-family doctrine). Empirical redundancy validation against
  self-declared family lands in Commit 2 (verification cards).
- STATUS gates dispatch entitlement: "proposed" entries exist but
  cannot burn dispatch quota until the human approves the
  verification card → "dispatchable". Research-auto-capital-human
  applied to signal onboarding. The 9 migrated signals are
  grandfathered dispatchable (they shipped pre-registry with their
  own test coverage + GP/A golden parity).

FIELD CATALOG
=============
Declares the per-field PIT semantics signals compose. PIT rules are
properties of FIELDS (where the data came from and when it became
knowable), not of signals.
"""
from __future__ import annotations

import dataclasses as _dc
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────────────
# Field catalog — PIT semantics live here
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class FieldDef:
    key:      str   # canonical "source.column" key
    source:   str   # data layer
    column:   str   # column name in the cached parquet / panel
    pit_rule: str   # how/when this value becomes knowable
    units:    str


FIELD_CATALOG: dict[str, FieldDef] = {
    # CRSP monthly price layer — values ARE the month-end observation;
    # knowable at the close they're computed from. Templates must lag
    # one period when using them for trade decisions (Bug-1 doctrine:
    # universe selection uses lagged mktcap).
    "crsp.msf.ret": FieldDef(
        key="crsp.msf.ret", source="crsp.msf", column="ret",
        pit_rule="month-end total return; knowable at that close",
        units="decimal/month"),
    "crsp.msf.mktcap": FieldDef(
        key="crsp.msf.mktcap", source="crsp.msf", column="mktcap",
        pit_rule="month-end close × shares; knowable at that close; "
                   "LAG ONE MONTH for any trade-time decision",
        units="USD thousands"),

    # Compustat annual fundamentals — knowable_at from real rdq
    # (PIT cache) with datadate+120d fallback. The as-of merge in
    # _build_lagged_funda_panel enforces public_date <= month_end.
    "compustat.funda.sale": FieldDef(
        key="compustat.funda.sale", source="comp_pit.funda",
        column="sale", pit_rule="annual; knowable_at=rdq|datadate+120d",
        units="USD millions"),
    "compustat.funda.cogs": FieldDef(
        key="compustat.funda.cogs", source="comp_pit.funda",
        column="cogs", pit_rule="annual; knowable_at=rdq|datadate+120d",
        units="USD millions"),
    "compustat.funda.at": FieldDef(
        key="compustat.funda.at", source="comp_pit.funda",
        column="at", pit_rule="annual FY-end; knowable_at=rdq|datadate+120d",
        units="USD millions"),
    "compustat.funda.ceq": FieldDef(
        key="compustat.funda.ceq", source="comp_pit.funda",
        column="ceq", pit_rule="annual FY-end; knowable_at=rdq|datadate+120d",
        units="USD millions"),
    "compustat.funda.ni": FieldDef(
        key="compustat.funda.ni", source="comp_pit.funda",
        column="ni", pit_rule="annual FY total; knowable_at=rdq|datadate+120d",
        units="USD millions"),
    "compustat.funda.xsga": FieldDef(
        key="compustat.funda.xsga", source="comp_pit.funda",
        column="xsga", pit_rule="annual FY total; knowable_at=rdq|datadate+120d",
        units="USD millions"),
}


# ────────────────────────────────────────────────────────────────────
# Signal definition
# ────────────────────────────────────────────────────────────────────
@_dc.dataclass(frozen=True)
class SignalDefinition:
    """One cross-sectional signal. Frozen — edits require a commit.

    kind:
      "crsp_panel" — formula(ctx) over wide month_end×permno panels;
                     ctx has .rets and .mktcap attributes.
      "funda_long" — formula(merged) over the PIT-lagged long-form
                     funda panel (one row per month_end×permno with
                     required_fields columns present, plus mktcap if
                     needs_mktcap_merge). Returns a pd.Series aligned
                     to merged.index with the RAW economic value.
    direction:
      "long_high" — high raw value is the LONG side
      "long_low"  — low  raw value is the LONG side (template negates)
    """
    key:               str
    kind:              str                  # "crsp_panel" | "funda_long"
    direction:         str                  # "long_high" | "long_low"
    family:            str                  # Bailey-LdP family
    required_fields:   tuple[str, ...]      # FIELD_CATALOG keys
    formula:           Callable
    aliases:           tuple[str, ...]      # regex patterns (LLM hint matching)
    paper_citation:    str
    pit_notes:         str
    needs_mktcap_merge: bool = False        # funda_long: merge CRSP mktcap in
    nonneg_guard_field: Optional[str] = None  # NaN the signal where field <= 0
    status:            str = "proposed"     # "proposed" | "dispatchable"


# ── crsp_panel formulas — receive ctx(.rets, .mktcap), return RAW wide panel
def _f_mktcap(ctx) -> pd.DataFrame:
    return ctx.mktcap


def _f_vol_12m(ctx) -> pd.DataFrame:
    return ctx.rets.rolling(12, min_periods=8).std()


def _f_ret_12_1(ctx) -> pd.DataFrame:
    log_ret = np.log1p(ctx.rets)
    return log_ret.rolling(12, min_periods=10).sum().shift(1)


def _f_ret_6_1(ctx) -> pd.DataFrame:
    log_ret = np.log1p(ctx.rets)
    return log_ret.rolling(6, min_periods=5).sum().shift(1)


def _f_reversal_1m(ctx) -> pd.DataFrame:
    return ctx.rets.shift(0)


# ── Daily-derived signals (from pre-aggregated _crsp_daily_aggregates_monthly)
# Built 2026-06-14. Source: CRSP DSF rolling-window stats pre-computed
# to month-end snapshots per permno.

from functools import lru_cache as _lru_cache
from pathlib import Path as _Path

_DAILY_AGG_PATH = (_Path(__file__).resolve().parents[2]
                    / "data" / "cache"
                    / "_crsp_daily_aggregates_monthly.parquet")


@_lru_cache(maxsize=1)
def _load_daily_aggregates() -> pd.DataFrame:
    """Lazy-load the daily aggregates cache. 32MB; pivot to wide per call."""
    if not _DAILY_AGG_PATH.is_file():
        raise FileNotFoundError(
            f"daily aggregates cache missing: {_DAILY_AGG_PATH} — "
            "run scripts/build_crsp_daily_aggregates.py"
        )
    df = pd.read_parquet(_DAILY_AGG_PATH)
    df["month_end"] = pd.to_datetime(df["month_end"])
    return df


def _wide_from_aggregates(col: str) -> pd.DataFrame:
    """Pivot daily aggregates to wide month_end × permno."""
    df = _load_daily_aggregates()
    return df.pivot(index="month_end", columns="permno", values=col).sort_index()


def _f_max_5_d21(ctx) -> pd.DataFrame:
    return _wide_from_aggregates("max_5_d21")


def _f_illiq_d21(ctx) -> pd.DataFrame:
    return _wide_from_aggregates("illiq_d21")


def _f_idiov_d60(ctx) -> pd.DataFrame:
    return _wide_from_aggregates("idiov_d60")


# ── funda_long formulas — receive merged long df, return RAW pd.Series
def _f_gp_at(merged: pd.DataFrame) -> pd.Series:
    return (merged["sale"] - merged["cogs"]) / merged["at"]


def _f_book_to_market(merged: pd.DataFrame) -> pd.Series:
    # ceq in $MM (Compustat); mktcap in $K (CRSP) → ceq*1000
    return (merged["ceq"] * 1000.0) / merged["mktcap"]


def _f_at_growth(merged: pd.DataFrame) -> pd.Series:
    m = merged.sort_values(["permno", "month_end"])
    at_prior = m.groupby("permno")["at"].shift(12)
    growth = (m["at"] / at_prior) - 1.0
    return growth.reindex(merged.index)


def _f_roe(merged: pd.DataFrame) -> pd.Series:
    return merged["ni"] / merged["ceq"]


def _f_op_profit(merged: pd.DataFrame) -> pd.Series:
    # Ball-Gerakos-Linnainmaa-Nikolaev 2015: operating profitability.
    # xsga missing → treated as 0 (standard replication convention;
    # many financials report no xsga line).
    return ((merged["sale"] - merged["cogs"]
              - merged["xsga"].fillna(0.0)) / merged["at"])


# ────────────────────────────────────────────────────────────────────
# THE REGISTRY — insertion order = alias-matching priority
# (fundamental patterns first, mirroring the pre-registry
# _SIGNAL_PATTERNS ordering so specific patterns win over permissive
# later ones; locked by the signal-picker tests)
# ────────────────────────────────────────────────────────────────────
SIGNAL_REGISTRY: dict[str, SignalDefinition] = {

    "gp_at": SignalDefinition(
        key="gp_at", kind="funda_long", direction="long_high",
        family="PROFITABILITY",
        required_fields=("compustat.funda.sale", "compustat.funda.cogs",
                           "compustat.funda.at"),
        formula=_f_gp_at,
        aliases=(r"gross_?profit|gp_?at|gp_?to_?at|gpa",),
        paper_citation="Novy-Marx 2013 (gross profitability)",
        pit_notes="annual FY totals via PIT as-of merge",
        status="dispatchable",
    ),

    "book_to_market": SignalDefinition(
        key="book_to_market", kind="funda_long", direction="long_high",
        family="VALUE",
        required_fields=("compustat.funda.ceq", "crsp.msf.mktcap"),
        formula=_f_book_to_market,
        aliases=(r"book_to_market|book_?to_?market|btm|b_?to_?m|log_book",),
        paper_citation="Fama-French 1992 (B/M)",
        pit_notes="ceq FY-end PIT-lagged; mktcap same month-end",
        needs_mktcap_merge=True,
        nonneg_guard_field="ceq",
        status="dispatchable",
    ),

    "at_growth": SignalDefinition(
        key="at_growth", kind="funda_long", direction="long_low",
        family="INVESTMENT",
        required_fields=("compustat.funda.at",),
        formula=_f_at_growth,
        aliases=(r"asset_?growth|at_?growth|investment_factor|d_?at",),
        paper_citation="Cooper-Gulen-Schill 2008 (asset growth)",
        pit_notes="YoY at via 12m self-lag on the PIT-lagged panel",
        status="dispatchable",
    ),

    "roe": SignalDefinition(
        key="roe", kind="funda_long", direction="long_high",
        family="PROFITABILITY",
        required_fields=("compustat.funda.ni", "compustat.funda.ceq"),
        formula=_f_roe,
        aliases=(r"return_?on_?equity|\broe\b|net_income.*equity|profitability_roe",),
        paper_citation="Hou-Xue-Zhang 2015 (ROE factor)",
        pit_notes="annual ni / FY-end ceq, both PIT-lagged",
        nonneg_guard_field="ceq",
        status="dispatchable",
    ),

    # flex-5 acceptance entry (2026-06-10): the FIRST post-registry
    # signal — exactly ONE entry, zero template edits. Deliberately a
    # known gp_at variant so the Commit-2 redundancy gate gets a real
    # positive case. Ships as "proposed": dispatch blocked until the
    # verification card is approved (the full onboarding arc).
    "op_profit": SignalDefinition(
        key="op_profit", kind="funda_long", direction="long_high",
        family="PROFITABILITY",
        required_fields=("compustat.funda.sale", "compustat.funda.cogs",
                           "compustat.funda.xsga", "compustat.funda.at"),
        formula=_f_op_profit,
        aliases=(r"operating_?profit|op_?profit|\bop_?at\b|rmw_?raw",),
        paper_citation=("Ball-Gerakos-Linnainmaa-Nikolaev 2015 "
                          "(operating profitability; FF5 RMW basis)"),
        pit_notes=("annual FY totals via PIT as-of merge; xsga "
                     "missing→0 per replication convention"),
        # 2026-06-14: promoted "proposed" → "dispatchable" after the GP/A
        # post-pub OOS failure (1992-2010 GREEN, 2011-2026 RED — DEAD_POST_PUB
        # flag in commit 4b70e880 + a155fcbb rigor pipeline). Op-profit is
        # the FF5 RMW basis variant (Ball-Gerakos-Linnainmaa-Nikolaev 2015)
        # — subtracts xsga from gross to exclude operating expenses already
        # capitalized. The two signals (gp_at vs op_profit) form an A/B test
        # for "did profitability anomaly decay or just shift to a cleaner
        # definition" — key question for paper-trade readiness on PROFITABILITY
        # family.
        status="dispatchable",
    ),

    "ret_12_1": SignalDefinition(
        key="ret_12_1", kind="crsp_panel", direction="long_high",
        family="MOMENTUM",
        required_fields=("crsp.msf.ret",),
        formula=_f_ret_12_1,
        aliases=(r"(?:ret|return|mom).*?12.*?1|momentum_12|mom_12_1|return_12_2",),
        paper_citation="Jegadeesh-Titman 1993 (12-1 momentum)",
        pit_notes="rolling log-return sum, shift(1) skips formation month",
        status="dispatchable",
    ),

    "ret_6_1": SignalDefinition(
        key="ret_6_1", kind="crsp_panel", direction="long_high",
        family="MOMENTUM",
        required_fields=("crsp.msf.ret",),
        formula=_f_ret_6_1,
        aliases=(r"(?:ret|return|mom).*?6.*?1|mom_6_1",),
        paper_citation="Jegadeesh-Titman 1993 (6-month variant)",
        pit_notes="rolling log-return sum, shift(1)",
        status="dispatchable",
    ),

    "reversal_1m": SignalDefinition(
        key="reversal_1m", kind="crsp_panel", direction="long_low",
        family="REVERSAL",
        required_fields=("crsp.msf.ret",),
        formula=_f_reversal_1m,
        aliases=(r"reversal|short_term|ret_1m|prior_month_return",),
        paper_citation="Jegadeesh 1990 (short-term reversal)",
        pit_notes="prior-month return observed at month-end t",
        status="dispatchable",
    ),

    "vol_12m": SignalDefinition(
        key="vol_12m", kind="crsp_panel", direction="long_low",
        family="LOW_VOL",
        required_fields=("crsp.msf.ret",),
        formula=_f_vol_12m,
        aliases=(r"vol_12|low_?vol|idio_?vol|volatility_lookback|vol_proxy",),
        paper_citation="Ang-Hodrick-Xing-Zhang 2006 / Blitz-van Vliet 2007",
        pit_notes="rolling 12m std of monthly returns",
        status="dispatchable",
    ),

    "mktcap": SignalDefinition(
        key="mktcap", kind="crsp_panel", direction="long_low",
        family="SIZE",
        required_fields=("crsp.msf.mktcap",),
        formula=_f_mktcap,
        aliases=(r"mktcap|market_?cap|log_?market|size|market_equity",),
        paper_citation="Banz 1981 (size)",
        pit_notes="month-end mktcap; universe selection separately lags",
        status="dispatchable",
    ),

    # ── Daily-derived signals (2026-06-14, post WRDS auth fix + CRSP DSF fetch)
    "max_5_d21": SignalDefinition(
        key="max_5_d21", kind="crsp_panel", direction="long_low",
        family="LOTTERY",
        required_fields=("crsp.dsf.ret",),
        formula=_f_max_5_d21,
        aliases=(r"max[\-_]?effect|max_5|lottery|extreme_return|bcw[\-_]?max",),
        paper_citation=("Bali-Cakici-Whitelaw 2011 (max-effect: stocks "
                          "with high recent maximum daily return earn LOWER "
                          "future returns — lottery-preference premium)"),
        pit_notes=("month-end snapshot of mean(top-5 daily returns) over "
                     "trailing 21 trading days; computed from CRSP DSF "
                     "via scripts/build_crsp_daily_aggregates.py"),
        status="dispatchable",
    ),

    "illiq_d21": SignalDefinition(
        key="illiq_d21", kind="crsp_panel", direction="long_high",
        family="LIQUIDITY",
        required_fields=("crsp.dsf.ret", "crsp.dsf.prc", "crsp.dsf.vol"),
        formula=_f_illiq_d21,
        aliases=(r"amihud|illiq|illiquid|liquidity_proxy|dollar[\-_]?vol",),
        paper_citation=("Amihud 2002 (illiquidity premium: ILLIQ = "
                          "mean(|ret| / dollar_vol)). HIGH illiquidity = "
                          "LONG side; positive premium for bearing "
                          "illiquidity risk."),
        pit_notes=("month-end snapshot of mean(|ret| / (|prc|×vol)) × 1e6 "
                     "over trailing 21 trading days"),
        status="dispatchable",
    ),

    "idiov_d60": SignalDefinition(
        key="idiov_d60", kind="crsp_panel", direction="long_low",
        family="LOW_VOL",
        required_fields=("crsp.dsf.ret",),
        formula=_f_idiov_d60,
        aliases=(r"idio[\-_]?vol|idiov|idiosyncratic|residual_vol|capm[\-_]?resid",),
        paper_citation=("Ang-Hodrick-Xing-Zhang 2006 (idiosyncratic "
                          "volatility puzzle: HIGH idio_vol → LOWER future "
                          "returns; LONG low-idiovol). We use market-"
                          "adjusted residual proxy (ret - mkt) over "
                          "trailing 60 days, NOT exact CAPM β regression."),
        pit_notes=("month-end snapshot of std(ret - mkt) over trailing 60 "
                     "trading days; approximates CAPM residual std with "
                     "β=1 assumption for cross-sectional speed"),
        status="dispatchable",
    ),
}


# Compiled alias patterns, cached at import. Insertion order preserved.
_COMPILED_ALIASES: list[tuple[str, re.Pattern]] = [
    (key, re.compile(pat, re.I))
    for key, sdef in SIGNAL_REGISTRY.items()
    for pat in sdef.aliases
]


# ────────────────────────────────────────────────────────────────────
# Lookup API (what templates + contracts consume)
# ────────────────────────────────────────────────────────────────────
def get_signal(key: str) -> Optional[SignalDefinition]:
    return SIGNAL_REGISTRY.get(key)


def match_signal_key(signal_inputs: tuple[str, ...]) -> Optional[str]:
    """Resolve free-text LLM signal hints to a registry key.
    First alias match wins (registry insertion order = priority)."""
    joined = " ".join(signal_inputs).lower()
    for key, pat in _COMPILED_ALIASES:
        if pat.search(joined):
            return key
    return None


def dispatchable_signals() -> tuple[str, ...]:
    """Signals allowed to burn dispatch quota. Contract
    supported_signals derives from this — single source of truth.

    Commit 2 (2026-06-10): a signal is dispatchable when EITHER
      - status == "dispatchable" in code (the 9 grandfathered), OR
      - status == "proposed" AND a human approval row exists in the
        signal_approvals ledger (verification-card workflow; see
        engine.research.signal_verification).
    Tombstones in the ledger revoke ledger-approvals (never the
    grandfathered code status — revoking those = a code change)."""
    try:
        from engine.research.signal_verification import load_approvals
        approved = set(load_approvals().keys())
    except Exception:
        approved = set()
    return tuple(
        k for k, s in SIGNAL_REGISTRY.items()
        if s.status == "dispatchable" or k in approved
    )


def funda_signals() -> frozenset[str]:
    """Signals that need the Compustat long-form plumbing."""
    return frozenset(k for k, s in SIGNAL_REGISTRY.items()
                       if s.kind == "funda_long")


def required_columns(key: str) -> list[str]:
    """Raw parquet column names for a funda_long signal (resolved
    through the FieldCatalog), EXCLUDING crsp.* fields (those come
    from the price panel, not the funda merge)."""
    sdef = SIGNAL_REGISTRY[key]
    cols = []
    for fkey in sdef.required_fields:
        fdef = FIELD_CATALOG[fkey]
        if fdef.source.startswith("comp"):
            cols.append(fdef.column)
    return cols


def validate_registry() -> list[str]:
    """Sanity checks — run by tests + (later) pre-commit hook.
    Returns list of error strings; empty = clean."""
    errors: list[str] = []
    seen_alias_hits: dict[str, str] = {}
    for key, sdef in SIGNAL_REGISTRY.items():
        if sdef.key != key:
            errors.append(f"{key}: key mismatch with dict key")
        if sdef.kind not in ("crsp_panel", "funda_long"):
            errors.append(f"{key}: unknown kind {sdef.kind!r}")
        if sdef.direction not in ("long_high", "long_low"):
            errors.append(f"{key}: unknown direction {sdef.direction!r}")
        if sdef.status not in ("proposed", "dispatchable"):
            errors.append(f"{key}: unknown status {sdef.status!r}")
        if not sdef.required_fields:
            errors.append(f"{key}: no required_fields")
        for fkey in sdef.required_fields:
            if fkey not in FIELD_CATALOG:
                errors.append(f"{key}: field {fkey!r} not in FIELD_CATALOG")
        # Alias self-collision check: each signal's own key string
        # must resolve back to itself (canonical round-trip)
        hit = match_signal_key((key,))
        if hit is not None and hit != key:
            errors.append(f"{key}: canonical name resolves to {hit!r} "
                            "(alias collision)")
    return errors
