"""engine.agents.strengthener.templates.carry_g10_fx — Tier C-2f.

Cross-sectional FX carry template on the G10 universe. Third real
template after TSMOM (C-2b) and cross-sec equities (C-2e). Closes
the loop with the LRV anchor library — the same HML_FX series we
already use as a cross-asset attribution anchor is now ALSO
constructible as a STANDALONE backtest spec (signal_kind="carry",
universe="fx_g10").

Scope (intentionally narrow per piece-by-piece doctrine):
  - signal_kind   : carry
  - universe      : fx_g10 (only — futures-carry-basket templates
                     come later as separate (carry, commodity_futures_27)
                     and (carry, us_treasury_curve) entries)
  - signal        : forward discount = lagged short-rate differential
                     vs USD (LRV 2011 §2.1 — interest-rate-parity
                     proxy for forward discount when forwards unavailable)
  - rebal         : monthly (last trading day)
  - weighting     : tercile L/S, equal-weighted within bucket,
                     dollar-neutral (top carry − bottom carry)
                     [B-class: spec.n_buckets overrides default]
  - cost          : 8 bp per round-trip (G10 spot bid-ask midpoint
                     2-3bp/side + slippage, per BIS Triennial 2022)
                     stressed at 0/8/16/24bp.
  - history       : 1999-01 onward (EUR-USD launch binding; earlier
                     observations limited by JPY rate series start
                     2002 + DKK rate gaps)

Why NOT reuse engine.b_plus_search / strategies.carry:
  Both pre-existing modules wrap the SAME LRV portfolio construction
  but live behind cron / strategy-registry abstractions designed for
  PRODUCTION SIGNALS not Tier C exploration. The dispatcher contract
  is "spec in, TemplateResult out, single function" — wrapping a
  cron-style strategy adds a flask of glue with zero gain. Re-using
  engine.research.fx_carry_anchors.build_carry_anchors is the right
  shared layer (it's a pure function, no side effects, no DB).

Verdict thresholds — mirror cross_sec + tsmom + factor_lab.runner:
  GREEN     |nw_t_stat| >= 1.96
  MARGINAL  1.65 <= |nw_t_stat| < 1.96
  RED       |nw_t_stat| < 1.65
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from engine.agents.strengthener.factor_spec_extractor import FactorSpec
from engine.agents.strengthener._safety_constants import (
    T_GREEN as _T_GREEN_SAFE,
    T_MARGINAL as _T_MARGINAL_SAFE,
    REPLICATION_T_TOLERANCE as _REPLICATION_T_TOL_SAFE,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Constants (D-class — implementation details)
# ────────────────────────────────────────────────────────────────────
_TEMPLATE_VERSION = "v1.0_2026-06-09"

_DEFAULT_N_BUCKETS = 3        # LRV / MSSS canonical tercile sort
_TC_BP_PER_RT     = 8.0       # G10 spot bid-ask ≈ 2-3bp/side + slip
_MIN_OBS_FLOOR    = 60        # never test < 5y regardless of spec.min_obs_months

# Cost stress levels — tighter than equity (FX TC is genuinely lower)
# Levels chosen to bracket the 13bp equity convention from below.
COST_STRESS_LEVELS_BP: tuple[float, ...] = (0.0, 8.0, 16.0, 24.0)
_COST_ROBUST_AT_BP   = 24.0   # cost-robust verdict reported at top stress

_T_GREEN    = _T_GREEN_SAFE
_T_MARGINAL = _T_MARGINAL_SAFE


# ────────────────────────────────────────────────────────────────────
# Date range parsing (mirrors cross_sec / tsmom templates)
# ────────────────────────────────────────────────────────────────────
def _parse_date_range(s: str) -> tuple[_dt.date, _dt.date]:
    if ":" not in s:
        raise ValueError(f"date_range must contain ':': {s!r}")
    a, b = s.split(":", 1)
    start = _dt.date.fromisoformat(f"{a.strip()}-01")
    end_ts = pd.Timestamp(f"{b.strip()}-01") + pd.offsets.MonthEnd(0)
    return start, end_ts.date()


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


def _cost_stress_verdict_from_t(t_stat: float) -> str:
    """Sign-aware cost-stress verdict (mirrors cross_sec). For
    cost stress the same factor under the same convention must
    remain positive — sign-flip ⇒ RED by construction."""
    if not math.isfinite(t_stat):
        return "RED"
    if t_stat >= _T_GREEN:
        return "GREEN"
    if t_stat >= _T_MARGINAL:
        return "MARGINAL"
    return "RED"


# ────────────────────────────────────────────────────────────────────
# Turnover for cost accounting
# ────────────────────────────────────────────────────────────────────
def _build_turnover_series(
    sort_keys:        pd.DataFrame,
    n_buckets:        int,
    g10_currencies:   tuple[str, ...],
) -> pd.Series:
    """Reconstruct the per-month L/S portfolio holdings (equal weight
    within bucket, +1 net in high, -1 net in low) and compute
    one-way turnover = 0.5 * Σ|w_t - w_{t-1}|.

    Cost accounting convention: turnover is one-way trade volume per
    dollar of gross notional. Cost = turnover * tc_bp_per_rt / 10_000
    treats `tc_bp_per_rt` as the FULL ROUND-TRIP cost. Same convention
    as cross_sec + tsmom.
    """
    # Holdings DataFrame indexed by date, columns = G10 currencies
    holdings = pd.DataFrame(0.0, index=sort_keys.index,
                                columns=list(g10_currencies))
    for date in sort_keys.index:
        keys = sort_keys.loc[date].dropna()
        if len(keys) < n_buckets * 2:
            continue
        sorted_ccys = keys.sort_values(ascending=False)
        n = len(sorted_ccys)
        bucket_size = n // n_buckets
        high_ccys = sorted_ccys.iloc[:bucket_size].index
        low_ccys  = sorted_ccys.iloc[-bucket_size:].index
        # Equal-weight long high, equal-weight short low; gross 2.0
        for c in high_ccys:
            holdings.at[date, c] = 1.0 / bucket_size
        for c in low_ccys:
            holdings.at[date, c] = -1.0 / bucket_size

    # One-way turnover from holdings diff
    diff = holdings.diff().abs().sum(axis=1) * 0.5
    diff.iloc[0] = holdings.iloc[0].abs().sum() * 0.5
    return diff


# ────────────────────────────────────────────────────────────────────
# Drawdown metrics (same shape as cross_sec for downstream parity)
# ────────────────────────────────────────────────────────────────────
def _compute_drawdown_metrics(pnl: pd.Series) -> dict:
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
    dd = (nav / peak) - 1.0
    max_dd = float(dd.min())

    underwater = (dd < 0).astype(int)
    current_uw = 0
    longest_uw = 0
    for v in underwater.values:
        if v == 1:
            current_uw += 1
            longest_uw = max(longest_uw, current_uw)
        else:
            current_uw = 0
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


# ────────────────────────────────────────────────────────────────────
# Cost stress + replication (mirror cross_sec contracts)
# ────────────────────────────────────────────────────────────────────
def _compute_cost_stress(
    pnl_gross:        pd.Series,
    turnover_series:  pd.Series,
    cost_levels_bp:   tuple[float, ...] = COST_STRESS_LEVELS_BP,
) -> dict[str, dict]:
    """Re-compute Sharpe + NW-t + verdict at each cost level using the
    SAME gross PnL and turnover. Returns dict keyed by "<int>bp"."""
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


def _cost_robust_verdict(stress: dict[str, dict],
                          *, robust_at_bp: float = _COST_ROBUST_AT_BP) -> str:
    key = f"{int(robust_at_bp)}bp"
    if key not in stress:
        return "UNKNOWN"
    return stress[key].get("verdict", "RED")


def _compute_replication_subsample(
    pnl_net:           pd.Series,
    paper_window:      str,
    paper_reported_t:  Optional[float],
) -> dict:
    """Sub-sample replication on the overlap with the paper's window.
    Same contract as cross_sec — see that module for full doctrine."""
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

    overlap_mask = (pnl_net.index >= p_start) & (pnl_net.index <= p_end_ts)
    overlap = pnl_net.loc[overlap_mask].dropna()
    if len(overlap) < 24:
        a_s = overlap.index.min().strftime("%Y-%m") if len(overlap) else "—"
        a_e = overlap.index.max().strftime("%Y-%m") if len(overlap) else "—"
        return {
            "window_intersection": f"{a_s}:{a_e}",
            "n_months_overlap":    len(overlap),
            "our_sharpe":          None,
            "our_t":               None,
            "paper_reported_t":    paper_reported_t,
            "t_gap":               None,
            "status":              "INSUFFICIENT_OVERLAP",
        }

    sharpe = annualized_sharpe(overlap)
    se     = newey_west_sharpe_se(overlap)
    t_ours = (sharpe / se) if (math.isfinite(sharpe)
                                 and math.isfinite(se) and se > 0) else None

    out: dict = {
        "window_intersection":
            f"{overlap.index.min().strftime('%Y-%m')}:"
            f"{overlap.index.max().strftime('%Y-%m')}",
        "n_months_overlap":    len(overlap),
        "our_sharpe":          float(sharpe) if math.isfinite(sharpe) else None,
        "our_t":               float(t_ours) if t_ours is not None else None,
        "paper_reported_t":    paper_reported_t,
        "t_gap":               None,
        "status":              "NO_BENCHMARK",
    }
    if paper_reported_t is not None and t_ours is not None:
        t_gap = abs(abs(t_ours) - abs(paper_reported_t))
        out["t_gap"]  = float(t_gap)
        out["status"] = ("REPLICATED" if t_gap <= _REPLICATION_T_TOL_SAFE
                          else "MISMATCH")
    return out


# ────────────────────────────────────────────────────────────────────
# Template entry point
# ────────────────────────────────────────────────────────────────────
def template_carry_g10_fx(spec: FactorSpec):
    """Tier C-2f template: LRV 2011 cross-sectional FX carry on G10.

    Returns a TemplateResult with the same metrics + artifact contract
    as cross_sec_us_equities (pnl_series_df ⇒ enables L2-4/L2-5/L2-6
    anchor lens execution downstream).
    """
    from engine.agents.strengthener.factor_dispatcher import TemplateResult
    from engine.research.fx_carry_anchors import (
        G10_CURRENCIES, build_carry_anchors,
        load_fx_spot_g10, load_g10_short_rates,
    )

    # ── 1. Scope guards ────────────────────────────────────────────
    if spec.signal_kind != "carry":
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = (f"signal_kind={spec.signal_kind!r} "
                                  "misrouted to carry_g10_fx template"),
            metrics          = {"misroute": True},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if spec.universe != "fx_g10":
        return TemplateResult(
            verdict          = "UNSUPPORTED_UNIVERSE",
            summary          = (f"universe={spec.universe!r} not "
                                  "supported by carry_g10_fx "
                                  "(only fx_g10 in C-2f)"),
            metrics          = {"unsupported_universe": spec.universe},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 2. Parse date range ────────────────────────────────────────
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

    # ── 3. Load FX spot + rates parquets (cached by LRV commits) ──
    spot_df = load_fx_spot_g10()
    if spot_df is None:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = ("G10 FX spot parquet missing — "
                                  "run scripts/fetch_fx_spot_g10.py"),
            metrics          = {"missing": "fx_spot_g10_monthly.parquet"},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    rates_df = load_g10_short_rates()
    if rates_df is None:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = ("G10 short rates parquet missing — "
                                  "run scripts/fetch_g10_short_rates.py"),
            metrics          = {"missing": "g10_short_rates_monthly.parquet"},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 4. Build carry portfolios (B-class n_buckets from spec) ───
    eff_n_buckets = spec.n_buckets or _DEFAULT_N_BUCKETS
    try:
        portfolios = build_carry_anchors(
            spot_df, rates_df, n_buckets=eff_n_buckets,
        )
    except Exception as exc:
        logger.exception("carry_g10_fx: build_carry_anchors raised")
        return TemplateResult(
            verdict          = "EXECUTION_ERROR",
            summary          = (f"build_carry_anchors failed: "
                                  f"{type(exc).__name__}: {exc}"),
            metrics          = {"error": str(exc)[:200]},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )
    if portfolios is None or portfolios.empty:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = ("build_carry_anchors returned empty — "
                                  "insufficient overlap between FX spot "
                                  "and rate panels"),
            metrics          = {"n_buckets": eff_n_buckets},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 5. Slice to user date range ────────────────────────────────
    portfolios = portfolios.loc[
        (portfolios.index >= pd.Timestamp(start_date))
        & (portfolios.index <= pd.Timestamp(end_date))
    ]
    if portfolios.empty:
        return TemplateResult(
            verdict          = "DATA_ERROR",
            summary          = (f"no carry observations in date_range "
                                  f"{spec.date_range} — first/last "
                                  "available is in FX/rates panel"),
            metrics          = {"date_range_requested": spec.date_range},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # HML_FX is already the L/S spread; expressed in % per month.
    # Convert to decimal returns for Sharpe / cost accounting consistency
    # (the equity templates work in decimal too).
    hml_pct       = portfolios["HML_FX"].dropna()
    pnl_gross_dec = hml_pct / 100.0

    # ── 6. Sample-size gate ────────────────────────────────────────
    n_months = int(len(pnl_gross_dec))
    required = max(spec.min_obs_months, _MIN_OBS_FLOOR)
    if n_months < required:
        return TemplateResult(
            verdict          = "INSUFFICIENT_HISTORY",
            summary          = (f"{n_months} months of carry PnL < "
                                  f"required {required} (FX panel "
                                  "starts ~1999 due to EUR launch + "
                                  "JPY rate start 2002)"),
            metrics          = {"n_months": n_months,
                                 "min_required": required,
                                 "n_buckets": eff_n_buckets},
            artifacts        = {},
            template_version = _TEMPLATE_VERSION,
        )

    # ── 7. Turnover series for cost accounting ─────────────────────
    # Reconstruct the per-month sort key (lagged rdiff) to derive
    # holdings + one-way turnover.
    rdiff_cols = [f"rdiff_{c}_pct" for c in G10_CURRENCIES
                    if f"rdiff_{c}_pct" in rates_df.columns]
    aligned_rdiff = (rates_df[rdiff_cols]
                     .reindex(portfolios.index, method=None)
                     .shift(1))
    aligned_rdiff.columns = [c.replace("rdiff_", "").replace("_pct", "")
                                for c in aligned_rdiff.columns]
    turnover = _build_turnover_series(
        sort_keys      = aligned_rdiff,
        n_buckets      = eff_n_buckets,
        g10_currencies = G10_CURRENCIES,
    )
    turnover = turnover.reindex(pnl_gross_dec.index).fillna(0.0)

    # Default net PnL at 13bp-equivalent (mid stress 8bp) so headline
    # verdict reflects realistic G10 spot TC.
    pnl_net_default = pnl_gross_dec - turnover * (_TC_BP_PER_RT / 10_000.0)

    # ── 8. Stats ────────────────────────────────────────────────────
    from engine.research.ablation.metrics import (
        annualized_sharpe, newey_west_sharpe_se,
    )
    sharpe   = annualized_sharpe(pnl_net_default)
    se_sharpe = newey_west_sharpe_se(pnl_net_default)
    if (not math.isfinite(sharpe) or not math.isfinite(se_sharpe)
            or se_sharpe <= 0):
        t_stat = float("nan")
    else:
        t_stat = sharpe / se_sharpe
    ann_ret = float(pnl_net_default.mean()) * 12.0
    ann_vol = float(pnl_net_default.std(ddof=1)) * math.sqrt(12.0)
    naive_verdict = _verdict_from_t(t_stat)

    # ── 9. Cost stress (0/8/16/24bp) ───────────────────────────────
    cost_stress = _compute_cost_stress(pnl_gross_dec, turnover)
    cost_robust_verdict = _cost_robust_verdict(cost_stress)

    # ── 10. Drawdown metrics ──────────────────────────────────────
    drawdown_naive = _compute_drawdown_metrics(pnl_net_default)
    pnl_24bp = pnl_gross_dec - turnover * (24.0 / 10_000.0)
    drawdown_24bp = _compute_drawdown_metrics(pnl_24bp)

    # ── 11. Replication mode (LRV 2011 paper window if supplied) ──
    replication: dict = {"status": "NOT_APPLICABLE"}
    if spec.paper_original_window:
        replication = _compute_replication_subsample(
            pnl_net          = pnl_net_default,
            paper_window     = spec.paper_original_window,
            paper_reported_t = spec.paper_reported_t,
        )

    # ── 12. Headline verdict = stricter of naive / cost / replication
    _verdict_severity = {"GREEN": 2, "MARGINAL": 1, "RED": 0,
                          "INSUFFICIENT_HISTORY": 0, "UNKNOWN": 0}
    if (_verdict_severity.get(cost_robust_verdict, 0)
            < _verdict_severity.get(naive_verdict, 0)):
        verdict = cost_robust_verdict
        cost_robust_note = (f" [cost-stress at {int(_COST_ROBUST_AT_BP)}bp "
                              f"dropped naive {naive_verdict} → {cost_robust_verdict}]")
    else:
        verdict = naive_verdict
        cost_robust_note = ""

    replication_note = ""
    if replication.get("status") == "MISMATCH":
        if verdict == "GREEN":
            verdict = "MARGINAL"
        replication_note = (
            f" [REPLICATION_MISMATCH: paper t≈{replication.get('paper_reported_t'):.2f} "
            f"vs ours t={replication.get('our_t'):.2f} in overlap "
            f"{replication.get('window_intersection')}]"
        )

    summary = (f"carry[HML_FX] L/S tercile on {len(G10_CURRENCIES)} "
                 f"G10 FCY {spec.date_range}: "
                 f"Sharpe={sharpe:.2f}, t={t_stat:.2f}, "
                 f"n={n_months}mo → {verdict}{cost_robust_note}{replication_note}")

    # ── 13. pnl_series_df for downstream lenses ───────────────────
    # Same column contract as cross_sec / TSMOM (where applicable) so
    # L2-4/L2-5/L2-6 anchor lenses can persist + regress without
    # template-specific branching.
    _pnl_series_df = pd.DataFrame({
        "pnl_gross":    pnl_gross_dec,
        "pnl_net_8bp":  pnl_net_default,
        "pnl_net_24bp": pnl_24bp,
        "turnover":     turnover,
    }).dropna(how="all")
    # B.2 (2026-06-09) explicit artifacts contract per
    # engine.research.lens_helpers: lenses read these declarations
    # instead of guessing column names from string patterns.
    _artifacts = {
        "pnl_series_df":   _pnl_series_df,
        "pnl_default_col": "pnl_net_8bp",
        "pnl_gross_col":   "pnl_gross",
    }

    return TemplateResult(
        verdict          = verdict,
        summary          = summary,
        metrics          = {
            "signal":              "carry_lrv_hml_fx",
            "sharpe":              float(sharpe) if math.isfinite(sharpe) else None,
            "nw_t_stat":           float(t_stat) if math.isfinite(t_stat) else None,
            "nw_se_sharpe":        float(se_sharpe) if math.isfinite(se_sharpe) else None,
            "ann_return":          ann_ret if math.isfinite(ann_ret) else None,
            "ann_vol":             ann_vol if math.isfinite(ann_vol) else None,
            "n_months":            n_months,
            "avg_turnover":        float(turnover.mean()),
            "n_currencies":        len(G10_CURRENCIES),
            "n_buckets":           eff_n_buckets,
            "tc_bp_per_rt":        _TC_BP_PER_RT,
            "n_trials":            1,
            # Senior-rigor additions (mirror cross_sec)
            "naive_verdict":         naive_verdict,
            "cost_robust_verdict":   cost_robust_verdict,
            "cost_stress":           cost_stress,
            "drawdown_naive":        drawdown_naive,
            "drawdown_24bp":         drawdown_24bp,
            "replication":           replication,
        },
        artifacts        = _artifacts,
        template_version = _TEMPLATE_VERSION,
    )
