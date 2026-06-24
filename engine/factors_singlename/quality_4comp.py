"""
engine/factors_singlename/quality_4comp.py — Wave B Quality 4-component factor.

Pre-registration: docs/spec_factor_ensemble_singlename_v1.md (id=52) §2.2 Wave B
Literature: Asness-Frazzini-Pedersen 2019 "Quality minus Junk", JFE.

Status (2026-05-10): SKELETON + MOCK MODE.
  - Real path queries Compustat `comp.fundq` for fundamentals + ratios per sub
  - Real path joins CRSP↔Compustat via `crsp.ccmxpf_lnkhist` (permno↔gvkey)
  - Activated when WRDS account approved (NUS BizFDB application 2026-05-09)
  - Until then, mock mode generates deterministic synthetic z-scores per sub
    so Wave B harness can run end-to-end pre-activation

AFP 2019 4 sub-components (each is itself a multi-ratio composite):
  1. PROFITABILITY  (PROF):    Gross-profit-to-assets + ROE + ROA + CFO/A + Gmar - Accruals
  2. GROWTH         (GROW):    5-year change in each profitability metric
  3. SAFETY         (SAFE):    Low β + Low leverage + Low EPS-vol + Low bankruptcy risk
  4. PAYOUT         (PAYO):    Low net-issuance (NIS) + Low equity-issuance + Low debt-issuance

For the skeleton + mock, each sub-component synthesizes its own seeded z-score.
On WRDS activation, each sub-component's stub gets replaced with the actual
multi-ratio Compustat query + within-universe z-score per ratio + sub-composite
average. The public composite function re-z-scores the 4-sub mean.

API mirror of value_pe.py / dividend_yield.py — drop-in replaceable in
factor ensemble (automated parity test in tests/).
"""
from __future__ import annotations

import datetime
import hashlib
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Locked per spec §2.2 Wave B + AFP 2019 standard composition
SUB_COMPONENTS_LOCKED: tuple[str, ...] = (
    "profitability",
    "growth",
    "safety",
    "payout",
)
SUB_WEIGHTS_LOCKED:    tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)  # equal-weight per AFP 2019

# Cross-section z-score min-universe gate (mirror value_pe / dividend_yield)
MIN_UNIVERSE_FOR_ZSCORE_LOCKED: int = 5


# ── Compustat field map (real-mode SQL stubs, for activation) ───────────────
# Each sub-component lists the Compustat (or computed-from-Compustat) fields
# it pulls. Mock mode does NOT read Compustat; real path will use these.
_COMPUSTAT_FIELDS_BY_SUB: dict[str, list[str]] = {
    "profitability": [
        "revt", "cogs", "at",     # → gross profit / assets
        "ib", "ceq",              # → ROE
        "oancf",                  # → CFO / assets
        "gp",                     # → gross margin
        "txt", "dp",              # → accruals proxy
    ],
    "growth": [
        # 5-year change in each profitability metric — pulled at as_of and as_of - 5y
        "revt", "cogs", "at", "ib", "ceq", "oancf", "gp",
    ],
    "safety": [
        "dlc", "dltt", "at",      # leverage = (dlc + dltt) / at
        "ib",                     # 5y rolling stdev of ib → earnings vol
        # β computed from CRSP returns, not Compustat
        # Altman Z = 1.2*(WC/A) + 1.4*(RE/A) + 3.3*(EBIT/A) + 0.6*(MV/L) + 1.0*(S/A)
        "wcap", "re", "oibdp", "lt", "csho", "prcc_f",
    ],
    "payout": [
        "csho",                   # share count → NIS = ΔlnCSHO
        "sstk",                   # equity issuance
        "prstkc",                 # equity buyback
        "dltis", "dltr",          # debt issuance / debt retirement
    ],
}


# ── Public API ──────────────────────────────────────────────────────────────
def compute_quality_singlestock_signal(
    as_of:         datetime.date,
    universe:      list[str],
    asset_classes: Optional[dict[str, str]] = None,
    panel:         Optional[pd.DataFrame] = None,
    *,
    mock_mode:     Optional[bool] = None,
) -> pd.Series:
    """
    Wave B Quality 4-component cross-section z-score signal (AFP 2019).

    Mechanism:
      1. Each of 4 sub-components computes its own within-universe z-score
      2. Composite = equal-weight (0.25 each) average of 4 sub z-scores
      3. Final = cross-section z-score of composite (mean=0, std=1 within universe)

    API mirror of `dividend_yield.compute_dividend_yield_singlestock_signal` and
    `value_pe.compute_value_pe_singlestock_signal` — drop-in replaceable in
    factor ensemble.

    Args:
        as_of:          decision date (no look-ahead)
        universe:       list of tickers
        asset_classes:  ignored (factor signature consistency)
        panel:          pre-fetched daily price panel (used for Safety β + price)
        mock_mode:      None (default) auto-detects via crsp_loader.is_wrds_available()

    Returns:
        pd.Series indexed by ticker, continuous z-score; NaN for missing data.
    """
    if not isinstance(as_of, datetime.date):
        raise TypeError(f"as_of must be datetime.date, got {type(as_of)}")
    if not universe:
        return pd.Series(dtype=float)
    if panel is None or panel.empty:
        logger.warning("compute_quality_singlestock_signal: panel required → all-NaN")
        return pd.Series(np.nan, index=universe, dtype=float)

    if mock_mode is None:
        try:
            from engine.universe_singlename.crsp_loader import is_wrds_available
            mock_mode = not is_wrds_available()
        except ImportError:
            mock_mode = True

    if mock_mode:
        return _mock_quality_signal(as_of, universe, panel)
    return _real_quality_signal(as_of, universe, panel)


# ── Sub-component computers (mock-mode synthesis) ──────────────────────────
def _mock_subscore(
    sub_name: str,
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Deterministic z-score per sub-component.

    Each (ticker, as_of, sub_name) triple seeds an independent N(0,1) draw.
    This makes each sub-component look statistically reasonable + reproducible
    + independent (4 sub-components are not collinear in mock).

    NaN for tickers not in panel (mirrors real-path data-availability gate).
    """
    raw: dict[str, float] = {}
    for ticker in universe:
        if ticker not in panel.columns:
            raw[ticker] = np.nan
            continue
        ts = panel[ticker].dropna()
        if ts.empty or ts[ts.index <= pd.Timestamp(as_of)].empty:
            raw[ticker] = np.nan
            continue

        seed_str = f"{sub_name}|{ticker}|{as_of.isoformat()}"
        seed = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        # Standard normal draw — already a z-score-shaped value
        raw[ticker] = float(rng.standard_normal())

    raw_series = pd.Series(raw, dtype=float)
    valid = raw_series.dropna()
    if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)

    # Re-z-score within universe so mean=0, std=1 exactly (sample-conditional)
    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = raw_series.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)


def _mock_quality_signal(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Mock composite: equal-weight 4 sub z-scores → re-z-score."""
    sub_zscores: dict[str, pd.Series] = {}
    for sub_name in SUB_COMPONENTS_LOCKED:
        sub_zscores[sub_name] = _mock_subscore(sub_name, as_of, universe, panel)

    return _composite_zscore(sub_zscores, universe)


# ── Real-mode stubs (per sub-component, activated on WRDS approval) ─────────
def _real_subscore_stub(sub_name: str) -> None:
    """Raise informative error — guides activation work."""
    fields = _COMPUSTAT_FIELDS_BY_SUB[sub_name]
    raise NotImplementedError(
        f"Real Compustat path for sub-component {sub_name!r} is stubbed — "
        f"activation tracked in spec id=52 amendment plan ('threshold_tweak +1 "
        f"trial' on WRDS approval). Compustat fields needed: {fields}. "
        f"See _COMPUSTAT_FIELDS_BY_SUB for full sub-component map."
    )


def _real_quality_signal(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Retry-wrapped entry point (delegate to _real_quality_signal_inner).

    Transient WRDS errors (connection drops / EOF prompts) auto-retry per
    `engine.universe_singlename.wrds_retry.with_wrds_retry`.
    """
    from engine.universe_singlename.wrds_retry import with_wrds_retry
    wrapped = with_wrds_retry(max_attempts=3, base_delay=5.0)(
        _real_quality_signal_inner
    )
    return wrapped(as_of, universe, panel)


def _real_quality_signal_inner(
    as_of:    datetime.date,
    universe: list[str],
    panel:    pd.DataFrame,
) -> pd.Series:
    """Real WRDS-Compustat Quality (4-component) signal.

    Activated 2026-05-11. Implements AFP 2019 "Quality Minus Junk" 4 sub-
    components on Wave B universe, using a simplified 1-ratio-per-sub
    composition (vs full multi-ratio AFP 2019 spec) for PRELIMINARY Wave B
    case study scope:

      profitability  = Return-on-Assets (ROA) = ib / at         (Compustat fundq)
      growth         = 5-year change in ROA                     (compares as_of vs as_of - 5y)
      safety         = -1 × leverage = -(dlc + dltt) / at       (low-leverage = high z)
      payout         = -1 × net-issuance = -ΔlnCSHO over 1y      (buybacks = positive)

    Honest scope (per memory + spec):
      - AFP 2019 strict uses 5-6 ratios per sub-component (GP/A + ROE + CFO/A +
        GMAR + ACC for profitability; multiple safety ratios incl. Altman Z).
      - PRELIMINARY Wave B uses 1 canonical ratio per sub (research-mining
        Tier 1 mining_runner already builds the framework; full multi-ratio
        composite is post-PRELIMINARY refinement).
      - This is explicitly disclosed in capability evidence + verdict markdown.
    """
    try:
        from engine.universe_singlename.crsp_loader import (
            _open_wrds_connection,
            is_wrds_available,
        )
    except ImportError as exc:
        raise RuntimeError(f"crsp_loader unavailable: {exc}")

    if not is_wrds_available():
        raise RuntimeError(
            "WRDS not configured. Pass mock_mode=True for skeleton testing, "
            "or install wrds + configure credentials for real-data path."
        )

    conn = _open_wrds_connection()
    try:
        # Resolve ticker → permno → gvkey
        names = conn.raw_sql(
            """
            SELECT permno, ticker
            FROM crsp.msenames
            WHERE ticker IN %(tickers)s
              AND namedt  <= %(as_of)s
              AND nameendt >= %(as_of)s
            """,
            params={
                "tickers": tuple(universe),
                "as_of":   as_of.isoformat(),
            },
            date_cols=[],
        )
        if names.empty:
            return pd.Series(np.nan, index=universe, dtype=float)
        permno_to_ticker: dict[int, str] = {}
        for _, row in names.iterrows():
            permno_to_ticker[int(row["permno"])] = str(row["ticker"]).strip()
        permnos = sorted(permno_to_ticker)

        link = conn.raw_sql(
            """
            SELECT lpermno, gvkey
            FROM crsp.ccmxpf_lnkhist
            WHERE lpermno IN %(permnos)s
              AND linktype IN ('LU', 'LC')
              AND linkprim IN ('P', 'C')
              AND linkdt <= %(as_of)s
              AND (linkenddt >= %(as_of)s OR linkenddt IS NULL)
            """,
            params={
                "permnos": tuple(permnos),
                "as_of":   as_of.isoformat(),
            },
            date_cols=[],
        )
        permno_to_gvkey: dict[int, str] = {}
        for _, row in link.iterrows():
            permno_to_gvkey[int(row["lpermno"])] = str(row["gvkey"])
        gvkeys = sorted(set(permno_to_gvkey.values()))
        if not gvkeys:
            return pd.Series(np.nan, index=universe, dtype=float)

        # Pull Compustat fundamentals (most-recent + 5y-ago snapshots)
        # Use comp.funda (annual) for stable ratio computation; fundq (quarterly)
        # has more noise + restatement issues for ratio bases like total assets.
        as_of_5y_ago = (as_of - datetime.timedelta(days=5*365)).isoformat()
        funda = conn.raw_sql(
            """
            SELECT gvkey, datadate, ib, at, dlc, dltt, csho
            FROM comp.funda
            WHERE gvkey IN %(gvkeys)s
              AND datadate <= %(as_of)s
              AND datadate >= %(as_of_minus_6y)s
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND consol = 'C'
              AND popsrc = 'D'
              AND at IS NOT NULL
            ORDER BY gvkey, datadate
            """,
            params={
                "gvkeys":         tuple(gvkeys),
                "as_of":          as_of.isoformat(),
                "as_of_minus_6y": (as_of - datetime.timedelta(days=6*365)).isoformat(),
            },
            date_cols=["datadate"],
        )

        # Build per-gvkey ratio dicts
        roa_latest:    dict[str, float] = {}
        roa_5y_prior:  dict[str, float] = {}
        leverage:      dict[str, float] = {}
        nis_1y:        dict[str, float] = {}   # 1-year ΔlnCSHO (net issuance proxy)

        cutoff_5y = pd.Timestamp(as_of) - pd.Timedelta(days=int(5.5 * 365))
        cutoff_1y = pd.Timestamp(as_of) - pd.Timedelta(days=int(1.5 * 365))

        for gvkey, group in funda.groupby("gvkey"):
            group = group.sort_values("datadate")
            latest = group.iloc[-1]
            at_val = float(latest["at"]) if pd.notna(latest["at"]) else 0.0
            if at_val <= 0:
                continue
            # Profitability: ROA = ib / at
            ib_val = float(latest["ib"]) if pd.notna(latest["ib"]) else np.nan
            if np.isfinite(ib_val):
                roa_latest[str(gvkey)] = ib_val / at_val
            # Safety: -1 × leverage = -(dlc + dltt) / at
            dlc_val = float(latest["dlc"])  if pd.notna(latest["dlc"])  else 0.0
            dltt_val = float(latest["dltt"]) if pd.notna(latest["dltt"]) else 0.0
            leverage[str(gvkey)] = -1.0 * (dlc_val + dltt_val) / at_val

            # Growth: 5-year ROA change (find datadate closest to 5y ago)
            old_rows = group[group["datadate"] <= cutoff_5y]
            if not old_rows.empty:
                old = old_rows.iloc[-1]
                at_old = float(old["at"]) if pd.notna(old["at"]) else 0.0
                ib_old = float(old["ib"]) if pd.notna(old["ib"]) else np.nan
                if at_old > 0 and np.isfinite(ib_old):
                    roa_5y_prior[str(gvkey)] = ib_old / at_old

            # Payout: ΔlnCSHO over ~1y (sign-flipped: low issuance = high z)
            old_1y = group[group["datadate"] <= cutoff_1y]
            csho_latest = float(latest["csho"]) if pd.notna(latest["csho"]) else 0.0
            if not old_1y.empty and csho_latest > 0:
                csho_old_raw = old_1y.iloc[-1]["csho"]
                if pd.notna(csho_old_raw):
                    csho_old = float(csho_old_raw)
                    if csho_old > 0:
                        nis_1y[str(gvkey)] = -1.0 * (np.log(csho_latest) - np.log(csho_old))

        # Build per-ticker raw signals
        def _build_subscore(gvkey_dict: dict[str, float]) -> pd.Series:
            raw: dict[str, float] = {}
            for ticker in universe:
                raw[ticker] = np.nan
                # Find first matching permno → gvkey for this ticker
                for p, t in permno_to_ticker.items():
                    if t == ticker and p in permno_to_gvkey:
                        gv = permno_to_gvkey[p]
                        if gv in gvkey_dict:
                            raw[ticker] = gvkey_dict[gv]
                        break
            # Cross-section z-score
            s = pd.Series(raw, dtype=float)
            valid = s.dropna()
            if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
                return pd.Series(np.nan, index=universe, dtype=float)
            mean = float(valid.mean())
            std = float(valid.std(ddof=1))
            if std <= 1e-9:
                return pd.Series(np.nan, index=universe, dtype=float)
            return pd.Series(
                {t: (s.get(t, np.nan) - mean) / std if np.isfinite(s.get(t, np.nan)) else np.nan
                 for t in universe},
                dtype=float,
            )

        # Growth = (ROA_latest - ROA_5y_prior); skip if either missing
        growth_dict: dict[str, float] = {}
        for gv, latest_val in roa_latest.items():
            if gv in roa_5y_prior:
                growth_dict[gv] = latest_val - roa_5y_prior[gv]

        sub_zscores = {
            "profitability": _build_subscore(roa_latest),
            "growth":        _build_subscore(growth_dict),
            "safety":        _build_subscore(leverage),
            "payout":        _build_subscore(nis_1y),
        }

        return _composite_zscore(sub_zscores, universe)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Composite helper (shared mock + real path) ──────────────────────────────
def _composite_zscore(
    sub_zscores: dict[str, pd.Series],
    universe:    list[str],
) -> pd.Series:
    """Equal-weight 4 sub z-scores → re-z-score within universe.

    Per AFP 2019: each sub is already z-scored, composite is mean of 4,
    final composite is re-z-scored within universe so output is ~ N(0, 1).

    NaN propagation: a ticker is included only if ≥ 3 of 4 sub-scores are
    valid (allows some sub-component data missing without dropping ticker).
    Mock mode always has all 4 valid; this matters mostly for real path.
    """
    sub_names = list(SUB_COMPONENTS_LOCKED)
    sub_df = pd.DataFrame({s: sub_zscores[s] for s in sub_names})

    # Tolerate up to 1 missing sub-component (≥ 3 of 4 valid)
    composite_raw = sub_df.mean(axis=1, skipna=True)
    n_valid_subs = sub_df.notna().sum(axis=1)
    composite_raw = composite_raw.where(n_valid_subs >= 3, np.nan)

    valid = composite_raw.dropna()
    if len(valid) < MIN_UNIVERSE_FOR_ZSCORE_LOCKED:
        return pd.Series(np.nan, index=universe, dtype=float)

    mean = float(valid.mean())
    std = float(valid.std(ddof=1))
    if std <= 1e-9:
        return pd.Series(np.nan, index=universe, dtype=float)

    out: dict[str, float] = {}
    for ticker in universe:
        v = composite_raw.get(ticker, np.nan)
        out[ticker] = (v - mean) / std if np.isfinite(v) else np.nan
    return pd.Series(out, dtype=float)
