"""
Portfolio Construction Layer
=============================
Converts raw TSMOM signals into final portfolio weights using
volatility targeting and regime-conditional position limits.

Method: Inverse-Volatility Weighting + Portfolio Vol Scaling
-------------------------------------------------------------
Step 1 — Inverse-vol raw weights
    For each asset i with signal s_i ∈ {-1, +1}:
        raw_w_i = s_i / σ_i
    Assets with signal = 0 receive zero weight.

Step 2 — Normalise to unit gross exposure
    w_i = raw_w_i / Σ|raw_w_j|

Step 3 — Estimate portfolio volatility (diagonal covariance)
    σ_port = sqrt(Σ (w_i × σ_i)²)
    Assumes zero cross-asset correlation — a known simplification
    for sector ETFs which do exhibit significant co-movement.
    Disclosed as such in report.

Step 4 — Scale to target volatility
    scalar = σ_target / σ_port
    scalar is capped at max_leverage to prevent excess concentration.
    final_w_i = w_i × scalar

Step 5 — Regime overlay
    In risk-off regime: long weights multiplied by regime_scale (default 0.3).
    Transition: partial scaling proportional to p_risk_off.
    Short weights unchanged (flight-to-safety effect).

Step 6 — Position limits
    Each weight capped at max_weight in absolute value.
    Rescaled after capping to maintain target volatility intent.

Academic references
-------------------
- Inverse-vol weighting: Leote de Carvalho et al. (2012)
  "Demystifying Equity Risk-Based Strategies"
- Vol targeting: Moreira & Muir (2017)
  "Volatility-Managed Portfolios", Journal of Finance
- Regime-conditional scaling: Ang & Bekaert (2004)
  "How Regimes Affect Asset Allocation"

Known limitations
-----------------
1. Zero-correlation assumption underestimates portfolio vol in stress periods
   (correlations spike toward 1.0 in crashes — this is precisely when
   the vol estimate is most misleading).
2. Vol estimate uses backward-looking realised vol (formation window),
   not forward-looking implied vol.
3. Monthly rebalancing frequency means position limits are checked once
   per month, not dynamically.

Integration
-----------
  Consumes: engine/signal.get_signal_dataframe()
            engine/regime.RegimeResult
  Consumed by: engine/backtest.run_backtest()  (replaces simple equal-weight)
               pages/backtest.py (display)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.regime import RegimeResult

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────────

TARGET_VOL   = 0.10   # 10% annualised portfolio volatility target
MAX_WEIGHT   = 0.25   # maximum absolute weight per position
MAX_LEVERAGE = 2.0    # cap on total gross exposure scalar
REGIME_SCALE = 0.30   # long weight multiplier in full risk-off

# ── 2026-05-29: MSM regime overlay DISABLED at the book level ─────────────────
# Walk-forward ablation 2018-2025 (95 months, scripts/ablation_msm_on_vs_off.py,
# evidence data/ablation/msm_on_vs_off_2018_2025.json):
#   Sharpe MSM-ON  +0.075 vs MSM-OFF +0.336  (Δ -0.262)
#   Bootstrap mean Δ Sharpe -0.295, 95% CI [-0.762, +0.049]
#   MaxDD MSM-ON -7.00% vs MSM-OFF -4.94% (-2pp worse)
#   In MSM-flagged risk-off subperiods (n=91): Sharpe ON +0.071 vs OFF +0.340
#   Overlay-helped rate in risk-off months: 42.9% (worse than coin flip)
# MSM flagged 91 of 95 months as risk-off — effectively a "permanent risk-off"
# bias that false-positive-shrank longs through the 2018-2025 bull market.
# We keep the MSM module running so daily brief / dashboards still get the
# p_risk_on signal, but at the book level we no longer multiply long weights
# by regime_scale or apply regime-conditional position caps. Flip this back
# to True only with fresh ablation evidence (Δ Sharpe ≥ +0.10, CI excludes 0).
ENABLE_REGIME_OVERLAY = False

# ── Tactical Overlay ───────────────────────────────────────────────────────────
#
# Implements a regime-adaptive execution filter on top of the strategic
# monthly rebalance.  Based on Moreira & Muir (2017, JF) volatility targeting
# and Ang & Bekaert (2004) regime-conditional asset allocation.
#
# Three parameters are regime-driven:
#   target_vol     — vol budget shrinks as regime deteriorates
#   entry_throttle — advisory flag: block NEW long entries in risk-off
#   stop_mult      — ATR stop-loss multiplier tightens in risk-off
#
# entry_throttle and stop_mult are advisory (surfaced in warnings / UI);
# target_vol is the only parameter with direct quantitative impact on weights.


@dataclass
class TacticalOverlay:
    regime:          str    # mirrors RegimeResult.regime
    target_vol:      float  # dynamic vol budget: 0.12 risk-on / 0.10 transition / 0.08 risk-off
    entry_throttle:  bool   # advisory: True = hold off new long entries
    stop_mult:       float  # advisory: ATR multiplier for stop-loss (2.0 → 1.5)
    note:            str    # human-readable rationale


def compute_tactical_overlay(regime: RegimeResult | None) -> TacticalOverlay:
    """
    Map RegimeResult → TacticalOverlay.

    Risk-on  (p_risk_on > 0.65): expand vol budget, open entries, loose stop
    Transition (0.35–0.65):       neutral vol budget, allow entries, tighter stop
    Risk-off (p_risk_on < 0.35): compress vol budget, throttle entries, tight stop

    The target_vol schedule (8% / 10% / 12%) is derived from the empirical
    observation in Moreira & Muir (2017) that scaling portfolio vol inversely
    with realised variance significantly improves Sharpe without drawdown costs.
    """
    if regime is None:
        return TacticalOverlay(
            regime="unknown", target_vol=TARGET_VOL,
            entry_throttle=False, stop_mult=2.0,
            note="制度未知，使用默认参数",
        )

    if regime.regime == "risk-on":
        return TacticalOverlay(
            regime="risk-on", target_vol=0.12,
            entry_throttle=False, stop_mult=2.0,
            note=f"risk-on (P={regime.p_risk_on:.2f})：扩大波动率预算至 12%，止损 2×ATR",
        )
    if regime.regime == "risk-off":
        return TacticalOverlay(
            regime="risk-off", target_vol=0.08,
            entry_throttle=True, stop_mult=1.5,
            note=f"risk-off (P={regime.p_risk_on:.2f})：压缩波动率预算至 8%，止损 1.5×ATR，建议暂停新多头",
        )
    # transition
    p = regime.p_risk_on
    blended_vol = 0.08 + (0.12 - 0.08) * p   # linear interpolation between 8% and 12%
    return TacticalOverlay(
        regime="transition", target_vol=round(blended_vol, 3),
        entry_throttle=False, stop_mult=1.75,
        note=f"transition (P={p:.2f})：过渡制度，波动率预算线性插值至 {blended_vol:.1%}，止损 1.75×ATR",
    )


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PortfolioWeights:
    """Output of portfolio construction for a single rebalancing date."""
    weights:          pd.Series    # index=sector, values=final weights
    gross_exposure:   float        # sum of absolute weights
    net_exposure:     float        # sum of signed weights (long - short)
    est_port_vol:     float        # estimated annualised portfolio vol (pre-scale)
    vol_scalar:       float        # scaling factor applied
    n_long:           int
    n_short:          int
    regime_applied:   str          # regime label used for overlay
    warnings:         list[str]


# ── Core construction ──────────────────────────────────────────────────────────

MAX_LONG  = 8   # maximum number of long positions (default / fallback)
MAX_SHORT = 6   # maximum number of short positions (default / fallback)

# P6-1a: Regime-conditional position limits
# Ang & Bekaert (2004): risk-off periods warrant more concentrated defensive positioning.
# Fewer longs (5) in risk-off preserves capital; more shorts (8) exploits wider opportunity set.
_REGIME_POSITION_LIMITS: dict[str, tuple[int, int]] = {
    "risk-on":    (10, 6),
    "transition": (7,  7),
    "risk-off":   (5,  8),
}


def _get_position_limits(regime_label: str) -> tuple[int, int]:
    """Return (max_long, max_short) for the given regime label."""
    return _REGIME_POSITION_LIMITS.get(regime_label, (MAX_LONG, MAX_SHORT))


def _regime_adjusted_cov(cov_matrix: np.ndarray, regime_label: str) -> np.ndarray:
    """
    P6-1a: In risk-off regime, shrink correlation matrix 30% toward perfect correlation.
    Longin & Solnik (2001), Forbes & Rigobon (2002): correlations spike toward 1.0 in crises.
    This makes portfolio vol estimates more conservative precisely when it matters most.
    """
    if regime_label != "risk-off":
        return cov_matrix
    std = np.sqrt(np.diag(cov_matrix))
    if np.any(std < 1e-12):
        return cov_matrix
    corr = cov_matrix / np.outer(std, std)
    shrink = 0.30
    corr_adj = (1.0 - shrink) * corr + shrink * np.ones_like(corr)
    np.fill_diagonal(corr_adj, 1.0)
    return corr_adj * np.outer(std, std)


def construct_portfolio(
    signal_df:              pd.DataFrame,
    regime:                 RegimeResult | None = None,
    target_vol:             float = TARGET_VOL,
    max_weight:             float = MAX_WEIGHT,
    max_leverage:           float = MAX_LEVERAGE,
    regime_scale:           float = REGIME_SCALE,
    returns_matrix:         pd.DataFrame | None = None,   # P2-3: T×N daily returns for LW shrinkage
    overlay:                "TacticalOverlay | None" = None,
    max_long:               int = MAX_LONG,
    max_short:              int = MAX_SHORT,
    max_net:                float = 0.40,
    min_net:                float = -0.10,
    prev_weights:           pd.Series | None = None,      # P6-1b: prior weights for turnover penalty
    turnover_penalty:       float = 0.0,                  # P6-1b: 0=strategic rebalance, 0.3=tactical
    apply_turnover_penalty: bool = False,                 # P6-1b: explicit gate (default off)
) -> PortfolioWeights:
    """
    Construct final portfolio weights from signal DataFrame and regime result.

    Args:
        signal_df      : output of signal.get_signal_dataframe() — must contain
                         columns: tsmom, ann_vol, inv_vol_wt
        regime         : output of regime.get_regime_on() — None = no overlay
        target_vol     : annualised portfolio volatility target (default 10%)
        max_weight     : absolute weight cap per position (default 25%)
        max_leverage   : cap on vol-scaling multiplier (default 2×)
        regime_scale   : long position multiplier in full risk-off (default 30%)
        returns_matrix : P2-3 — T×N DataFrame of daily returns (index=date,
                         columns=sector). When provided, uses Ledoit-Wolf shrinkage
                         covariance instead of diagonal approximation.

    Returns:
        PortfolioWeights dataclass with final weights and diagnostics.
    """
    warnings_log: list[str] = []

    # ── Book-level regime overlay gate (2026-05-29 ablation-driven) ───────────
    # See ENABLE_REGIME_OVERLAY in this file for evidence. Forcing regime=None
    # makes Step 3 (cov adjustment), Step 5 (long-weight shrink), and
    # _get_position_limits all behave as "no overlay" without touching call
    # sites — the MSM module keeps running for dashboards/daily brief.
    if regime is not None and not ENABLE_REGIME_OVERLAY:
        regime = None

    # ── Apply tactical overlay ─────────────────────────────────────────────────
    if overlay is not None:
        target_vol = overlay.target_vol
        if overlay.entry_throttle:
            warnings_log.append(
                f"⚠ 战术覆盖层：{overlay.note}  →  建议暂停新多头建仓"
            )
        else:
            warnings_log.append(f"战术覆盖层：{overlay.note}")

    # ── Validate inputs ────────────────────────────────────────────────────────
    required = {"tsmom", "ann_vol"}
    if not required.issubset(signal_df.columns):
        warnings_log.append(f"signal_df 缺少列: {required - set(signal_df.columns)}")
        return _empty_weights(warnings_log)

    df = signal_df.copy()
    df = df[df["tsmom"] != 0].copy()     # keep only non-neutral signals
    df = df.dropna(subset=["ann_vol"])
    df = df[df["ann_vol"] > 1e-6]        # drop zero-vol assets

    if df.empty:
        warnings_log.append("无有效信号（所有资产信号为中性或缺失波动率数据）")
        return _empty_weights(warnings_log)

    # P2-4 / P1-7: sizing vol priority chain — GARCH(1,1) > 21d realised > 12M formation.
    if "ann_vol_garch" in df.columns and df["ann_vol_garch"].notna().any():
        _sizing_vol = (
            df["ann_vol_garch"]
            .where(df["ann_vol_garch"] > 1e-6,
                   df.get("ann_vol_21d", df["ann_vol"]))
            .where(lambda v: v > 1e-6, df["ann_vol"])
        )
    elif "ann_vol_21d" in df.columns:
        _sizing_vol = df["ann_vol_21d"].where(df["ann_vol_21d"] > 1e-6, df["ann_vol"])
    else:
        _sizing_vol = df["ann_vol"]

    # ── Step 1: Inverse-vol raw weights ───────────────────────────────────────
    df["raw_w"] = df["tsmom"] / _sizing_vol

    # ── Step 2: Normalise to unit gross exposure ───────────────────────────────
    gross_raw = df["raw_w"].abs().sum()
    if gross_raw < 1e-9:
        return _empty_weights(warnings_log)
    df["w_norm"] = df["raw_w"] / gross_raw

    # ── Step 3: Estimate portfolio vol ────────────────────────────────────────
    _n_pos = int((df["w_norm"].abs() > 0.01).sum())
    lw_used = False

    if returns_matrix is not None and not returns_matrix.empty:
        # P2-3: Ledoit-Wolf shrinkage covariance
        # Align returns columns to current positions
        _common = [s for s in df.index if s in returns_matrix.columns]
        if len(_common) >= 3:
            try:
                from sklearn.covariance import LedoitWolf
                _ret = returns_matrix[_common].dropna()
                if len(_ret) >= 60:
                    _lw = LedoitWolf().fit(_ret)
                    _cov_ann = _lw.covariance_ * 252   # annualise daily cov
                    # P6-1a: apply regime-adjusted covariance (risk-off: +30% correlation shrinkage)
                    _regime_lbl = regime.regime if regime else "none"
                    _cov_ann = _regime_adjusted_cov(_cov_ann, _regime_lbl)
                    _w_vec = df.loc[_common, "w_norm"].values
                    _port_var = float(_w_vec @ _cov_ann @ _w_vec)
                    est_port_vol = float(np.sqrt(max(_port_var, 0.0)))
                    lw_used = True
            except Exception as _lw_err:
                logger.debug("LedoitWolf failed, falling back to diagonal: %s", _lw_err)

    if not lw_used:
        # Diagonal covariance fallback: σ_port = sqrt(Σ (w_i × σ_i)²)
        est_port_vol = float(np.sqrt((df["w_norm"] * df["ann_vol"]) ** 2).sum())

    # ── Step 3b: Diagonal-covariance upper bound warning (only when LW not used) ─
    if not lw_used and _n_pos > 1 and est_port_vol > 1e-4:
        _rho = 0.5
        vol_upper = est_port_vol * float(np.sqrt(1 + _rho * (_n_pos - 1)))
        if vol_upper > est_port_vol * 1.30:
            warnings_log.append(
                f"⚠ 零相关假设低估波动率：对角估计 {est_port_vol:.1%}，"
                f"ρ=0.5 上界约 {vol_upper:.1%}（{_n_pos} 个资产）"
            )

    if est_port_vol < 1e-6:
        warnings_log.append("组合波动率估计接近零，无法进行波动率目标化")
        df["w_scaled"] = df["w_norm"]
        vol_scalar = 1.0
    else:
        # ── Step 4: Scale to target vol ───────────────────────────────────────
        vol_scalar = min(target_vol / est_port_vol, max_leverage)
        df["w_scaled"] = df["w_norm"] * vol_scalar

        if vol_scalar >= max_leverage:
            warnings_log.append(
                f"杠杆上限触发（scalar={vol_scalar:.2f}x），"
                f"实际目标波动率将低于 {target_vol:.0%}"
            )

    # ── Step 5: Regime overlay ────────────────────────────────────────────────
    regime_label = "none"
    if regime is not None:
        regime_label = regime.regime
        # P6-1a: override max_long/max_short with regime-conditional limits
        _rl, _rs = _get_position_limits(regime_label)
        max_long  = _rl
        max_short = _rs
        if regime.regime == "risk-off":
            multiplier = regime_scale
        elif regime.regime == "transition":
            multiplier = regime_scale + (1.0 - regime_scale) * regime.p_risk_on
        else:
            multiplier = 1.0

        if multiplier < 1.0:
            df["w_scaled"] = df.apply(
                lambda r: r["w_scaled"] * multiplier if r["tsmom"] > 0 else r["w_scaled"],
                axis=1,
            )

    # ── Step 5b: Concentration filter — keep top-N long, top-N short ─────────
    # Rank by composite_score when available (higher = stronger signal quality).
    # Fallback: rank longs by w_scaled descending, shorts by w_scaled ascending.
    # Positions outside the top-N are zeroed out before position-limit clipping.
    _rank_col = "composite_score" if "composite_score" in df.columns else None

    _longs  = df[df["w_scaled"] > 0].copy()
    _shorts = df[df["w_scaled"] < 0].copy()

    if len(_longs) > max_long:
        if _rank_col:
            _keep_long = _longs[_rank_col].nlargest(max_long).index
        else:
            _keep_long = _longs["w_scaled"].nlargest(max_long).index
        _cut_long = _longs.index.difference(_keep_long)
        df.loc[_cut_long, "w_scaled"] = 0.0
        warnings_log.append(
            f"集中度约束：多头 {len(_longs)} → {max_long}，"
            f"剔除 {', '.join(str(s) for s in _cut_long.tolist()[:4])}"
            + ("…" if len(_cut_long) > 4 else "")
        )

    if len(_shorts) > max_short:
        if _rank_col:
            _keep_short = _shorts[_rank_col].nsmallest(max_short).index
        else:
            _keep_short = _shorts["w_scaled"].nsmallest(max_short).index
        _cut_short = _shorts.index.difference(_keep_short)
        df.loc[_cut_short, "w_scaled"] = 0.0
        warnings_log.append(
            f"集中度约束：空头 {len(_shorts)} → {max_short}，"
            f"剔除 {', '.join(str(s) for s in _cut_short.tolist()[:4])}"
            + ("…" if len(_cut_short) > 4 else "")
        )

    # Re-normalise after zeroing — maintain vol target intent
    df = df[df["w_scaled"] != 0.0].copy()
    if df.empty:
        return _empty_weights(warnings_log)
    gross_post = df["w_scaled"].abs().sum()
    if gross_post > 1e-9:
        df["w_scaled"] = df["w_scaled"] / gross_post * min(gross_post, max_leverage)

    # ── Step 5c: Net exposure clamp (long-biased: net ∈ [min_net, max_net]) ──
    net_now = float(df["w_scaled"].sum())
    if net_now > max_net:
        excess     = net_now - max_net
        long_gross = df.loc[df["w_scaled"] > 0, "w_scaled"].sum()
        if long_gross > 1e-9:
            df.loc[df["w_scaled"] > 0, "w_scaled"] *= (long_gross - excess) / long_gross
            warnings_log.append(
                f"净敞口约束：net {net_now:+.2%} → {max_net:+.2%}，等比例压缩多头"
            )
    elif net_now < min_net:
        deficit     = min_net - net_now           # positive
        short_gross = df.loc[df["w_scaled"] < 0, "w_scaled"].abs().sum()
        if short_gross > 1e-9:
            df.loc[df["w_scaled"] < 0, "w_scaled"] *= (short_gross - deficit) / short_gross
            warnings_log.append(
                f"净敞口约束：net {net_now:+.2%} → {min_net:+.2%}，等比例压缩空头"
            )

    # ── Step 5d: Score-weighted redistribution ───────────────────────────────
    # composite_score (0–100) redistributes weight among passed-gate positions.
    # High-conviction assets absorb weight from low-conviction ones; gross
    # exposure is preserved by re-normalising after modulation.
    # Gate blocking (score < 35) happens upstream — this step only adjusts sizing.
    if "composite_score" in df.columns and df["composite_score"].notna().any():
        _scores = df["composite_score"].clip(0, 100) / 100   # → [0, 1]
        _gross_pre = df["w_scaled"].abs().sum()
        df["w_scaled"] = df["w_scaled"] * _scores
        _gross_post = df["w_scaled"].abs().sum()
        if _gross_post > 1e-9:
            df["w_scaled"] = df["w_scaled"] * (_gross_pre / _gross_post)

    # ── Step 5e: Turnover penalty (tactical only — strategic rebalance uses penalty=0) ──
    # P6-1b: Prevents full convergence to vol-parity targets during tactical intraday
    # adjustments, keeping transaction costs proportionate.
    # Monthly strategic rebalance: apply_turnover_penalty=False → no friction.
    # Daily tactical overlay: apply_turnover_penalty=True, turnover_penalty=0.3.
    if apply_turnover_penalty and turnover_penalty > 0 and prev_weights is not None:
        _prev = prev_weights.reindex(df.index, fill_value=0.0)
        df["w_scaled"] = _prev + (df["w_scaled"] - _prev) * (1.0 - turnover_penalty)
        _gross_tp = df["w_scaled"].abs().sum()
        if _gross_tp > 1e-9:
            df["w_scaled"] = df["w_scaled"] / _gross_tp * min(_gross_tp, max_leverage)

    # ── Step 6: Position limits ───────────────────────────────────────────────
    df["w_final"] = df["w_scaled"].clip(lower=-max_weight, upper=max_weight)

    # Rescale after clipping to restore gross exposure intent
    gross_after = df["w_final"].abs().sum()
    gross_before = df["w_scaled"].abs().sum()
    if gross_after < gross_before * 0.95 and gross_after > 1e-9:
        # Clipping was binding — rescale proportionally
        clip_scalar = gross_before / gross_after
        df["w_final"] = df["w_final"] * min(clip_scalar, 1.2)  # small correction only
        warnings_log.append(
            f"仓位上限触发（{(df['w_scaled'].abs() > max_weight).sum()} 个资产被裁剪）"
        )

    # ── Step 6b: Correlated-pair weight cap ──────────────────────────────────
    # IWN/IWO and similar pairs share the same underlying risk factor.
    # Same-direction combined weight is capped at 1.5× the single-name cap.
    _CORR_PAIRS: list[tuple[str, str]] = [
        ("IWN", "IWO"),   # US small-cap value vs growth — ρ ≈ 0.85
        ("ASHR", "KWEB"), # China A-shares vs China internet
        ("XLV", "XBI"),   # Broad healthcare vs biotech
    ]
    _MAX_PAIR_COMBINED = max_weight * 1.5
    if "ticker" in df.columns:
        for tkr_a, tkr_b in _CORR_PAIRS:
            rows_a = df[df["ticker"] == tkr_a]
            rows_b = df[df["ticker"] == tkr_b]
            if rows_a.empty or rows_b.empty:
                continue
            idx_a, idx_b = rows_a.index[0], rows_b.index[0]
            w_a = df.loc[idx_a, "w_final"]
            w_b = df.loc[idx_b, "w_final"]
            if w_a * w_b <= 0:
                continue   # opposite directions — not a double-counting risk
            combined = abs(w_a) + abs(w_b)
            if combined > _MAX_PAIR_COMBINED:
                scale = _MAX_PAIR_COMBINED / combined
                df.loc[idx_a, "w_final"] *= scale
                df.loc[idx_b, "w_final"] *= scale
                warnings_log.append(
                    f"相关对约束触发（{tkr_a}/{tkr_b}）：合计{combined:.1%}→{_MAX_PAIR_COMBINED:.1%}"
                )

    # ── Assemble output ───────────────────────────────────────────────────────
    weights = df["w_final"].rename("weight")
    gross   = float(weights.abs().sum())
    net     = float(weights.sum())
    n_long  = int((weights > 0).sum())
    n_short = int((weights < 0).sum())

    if net > max_net or net < min_net:
        warnings_log.append(
            f"净敞口越界（net={net:+.2%}，允许范围 [{min_net:+.2%}, {max_net:+.2%}]）"
        )

    return PortfolioWeights(
        weights=weights,
        gross_exposure=round(gross, 4),
        net_exposure=round(net, 4),
        est_port_vol=round(est_port_vol, 4),
        vol_scalar=round(vol_scalar, 4),
        n_long=n_long,
        n_short=n_short,
        regime_applied=regime_label,
        warnings=warnings_log,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _empty_weights(warnings: list[str]) -> PortfolioWeights:
    return PortfolioWeights(
        weights=pd.Series(dtype=float),
        gross_exposure=0.0,
        net_exposure=0.0,
        est_port_vol=0.0,
        vol_scalar=0.0,
        n_long=0,
        n_short=0,
        regime_applied="none",
        warnings=warnings,
    )


def weights_to_dataframe(pw: PortfolioWeights, signal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge final weights with signal metadata for display.
    Returns DataFrame suitable for UI and backtest attribution.
    """
    if pw.weights.empty:
        return pd.DataFrame()

    df = signal_df[["ticker", "raw_return", "ann_vol", "tsmom", "csmom"]].copy()
    df["final_weight"] = pw.weights
    df["final_weight"] = df["final_weight"].fillna(0.0)
    df["direction"] = df["final_weight"].apply(
        lambda w: "多头" if w > 0.001 else ("空头" if w < -0.001 else "中性")
    )
    df["weight_%"] = (df["final_weight"] * 100).round(2)
    df["ann_vol_%"] = (df["ann_vol"] * 100).round(1)
    df["raw_ret_%"] = (df["raw_return"] * 100).round(2)

    return df.sort_values("final_weight", ascending=False)


def portfolio_diagnostics(pw: PortfolioWeights) -> dict:
    """
    Return key diagnostic metrics for display or logging.
    """
    return {
        "gross_exposure":  f"{pw.gross_exposure:.2f}x",
        "net_exposure":    f"{pw.net_exposure:+.2f}x",
        "est_port_vol":    f"{pw.est_port_vol:.1%}",
        "vol_scalar":      f"{pw.vol_scalar:.2f}x",
        "n_long":          pw.n_long,
        "n_short":         pw.n_short,
        "regime_applied":  pw.regime_applied,
        "warnings":        pw.warnings,
    }
