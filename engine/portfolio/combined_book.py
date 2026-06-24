"""engine/portfolio/combined_book.py — 已部署的双机制 book(股票 + carry)。

把验证过的合并配置(_book_config_run.py, commit db0897e)从一次性研究脚本，固化成
一个有测试、可被 book/接口调用的生产模块。合并方式是预先定死的风险预算规则(carry 占
30% 风险权重)——不是从一堆权重里挑最好看的(那是 p-hacking)。

构造完全沿用验证逻辑:
  股票 book = D_PEAD + 分析师修正(各自反向波动率加权，扣成本)
  carry    = 商品 + 外汇 carry(反向波动率合并，扣成本)
  合并      = 两边各自 vol-target 到 10%(滚动、无前视)，再按 carry 风险权重混合。
GREEN 结论本身已 spec-locked(id=77)，这里不重新论证，只把它做成可部署的算子。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RT_EQ = 30.0   # 股票腿单边成本 bps
RT_CY = 12.0   # carry(流动期货)单边成本 bps + roll/slippage buffer
               # Phase A.4 amendment 2026-05-28: 10→12 bps. Frazzini-Israel-Moskowitz
               # 2015 "Trading Costs of Asset Pricing Anomalies" estimates futures
               # RT ≈ 5-15bps for liquid contracts; we land at 12 (upper-mid) until
               # IB paper integration validates actual slippage. Combined book Sharpe
               # impact 1.085 → 1.079 (-0.006), negligible.
RT_TS = 12.0   # TSMOM(同 carry,同样的期货 pipeline,同样的 5-15bp 文献区间)
# Hedge-sleeve cost models per [[feedback-cost-model-rigor-almgren-not-scalar-
# 2026-05-30]]: TLT/GLD ETF half-spread 2bp + commission 2bp (IB 2024);
# MTUM short borrow already baked into mom_hedge_overlay.run_mom_hedge_backtest.
RT_CH = 8.0    # TLT/GLD crisis hedge: 2x (half_spread 2bp + commission 2bp)
# 2026-05-29 spec 77 §12 amendment: 30% carry → 25% carry + 5% TSMOM (3-mechanism,
# gap-analysis revision). TSMOM passed strict gate as standalone sleeve, but
# 99-month OLD-vs-NEW evidence (2016-2024 overlap) showed NEW @ 10% TSMOM has
# -0.07 Sharpe / -0.5pp/yr drag in calm markets. Initial deploy reduced to 5%:
# half the drag (negligible -0.035 Sharpe ≈ -0.28%/yr, basically noise), keeps
# mechanism diversification "seed", room to scale up if D-PEAD decay materializes.
# See docs/decisions/crossasset_tsmom_GREEN_2026-05-29.md.
DEFAULT_CARRY_RISK_WEIGHT = 0.25   # carry 让出 5% 给 TSMOM(was 30%)
DEFAULT_TSMOM_RISK_WEIGHT = 0.05   # 新机制 seed 仓位,observation period
# 2026-05-30 5-sleeve amendment (Post-BARRA Phase 3 risk-management add-on
# per [[project-barra-phase-chain-2026-05-30]] L1 factor budget).
# Added TLT/GLD crisis hedge (Path 1) + MTUM short MOM hedge (Path 2).
# Equity weight drops from 70% to 63% to fund the two hedge sleeves.
# Honest framing: hedges cost ~0.11 Sharpe ex-ante in normal regimes,
# bought as INSURANCE for MOM-crash + cross-asset shock protection.
DEFAULT_CRISIS_HEDGE_RISK_WEIGHT = 0.05   # TLT/GLD 50-50, cosine -0.36 with book
DEFAULT_MOM_HEDGE_RISK_WEIGHT    = 0.02   # MTUM short, direct MOM-β reduction
# Regime-conditional hedge grids per [[feedback-loop-refinement-multi-role-
# candidates-2026-05-30]] doctrine — pay insurance premium only when regime
# signal indicates need. Uses VIX 1y rolling z-score classifier (AN-1 simplified
# from VIX+OAS due to FRED reliability — full 2-signal upgrade pending).
REGIME_HEDGE_GRIDS = {
    "CALM":   {"crisis": 0.00, "mom_hedge": 0.00},   # no insurance in calm
    "NORMAL": {"crisis": 0.05, "mom_hedge": 0.02},   # standard insurance
    "STRESS": {"crisis": 0.10, "mom_hedge": 0.05},   # full insurance
}
VIX_Z_WINDOW_DAYS = 252       # 1-year rolling
REGIME_Z_THRESHOLD = 1.0      # ±1σ per AN-1 spec
DEFAULT_TARGET_VOL = 0.10          # 每条腿的 vol-target
# 已部署账本的整体目标波动(=杠杆/尺寸选择,不是预测)。10% 是机构级均衡账本的标准档,
# 且刻意留在回测舒适度之下(实盘比回测差、carry 有危机尾部 — 见部署决策)。
DEFAULT_BOOK_VOL_TARGET = 0.10


def voltarget(r: pd.Series, target: float = DEFAULT_TARGET_VOL, lb: int = 12) -> pd.Series:
    """滚动波动率目标化(用上一期的滚动波动定标，shift(1) 防前视)。"""
    rv = r.rolling(lb).std() * np.sqrt(12)
    return ((target / rv).clip(upper=2.0).shift(1) * r)


def blend_at_risk_weight(equity_vt: pd.Series, carry_vt: pd.Series,
                         carry_risk_weight: float = DEFAULT_CARRY_RISK_WEIGHT) -> pd.Series:
    """把两条已 vol-target 的收益序列按固定 carry 风险权重混合。预先定死，非网格搜索。"""
    J = pd.concat([equity_vt.rename("e"), carry_vt.rename("c")], axis=1).dropna()
    return ((1.0 - carry_risk_weight) * J["e"] + carry_risk_weight * J["c"]).rename("book")


def book_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return {"n": int(r.size), "ann": 0.0, "vol": 0.0, "sharpe": float("nan"), "maxdd": 0.0}
    vol = float(r.std() * np.sqrt(12))
    cum = (1 + r).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return {"n": int(r.size), "ann": round(float(r.mean() * 12), 4), "vol": round(vol, 4),
            "sharpe": round(float(r.mean() * 12 / vol), 3), "maxdd": round(dd, 4)}


def build_equity_book() -> pd.Series:
    """股票机制 book = D_PEAD + 分析师修正，反向波动率加权(扣成本)。沿用验证逻辑。"""
    from engine.validation.analyst_revision import build_revision_sleeve_buffered
    d = pd.read_parquet("data/cache/_dpead_recon_base.parquet").iloc[:, 0]
    d.index = pd.to_datetime(d.index)
    dp = ((1 + d.clip(-0.2, 0.2)).resample("ME").prod() - 1)
    dp_net = (dp - 5.0 * RT_EQ / 10000.0 / 12).rename("dp")
    rev, rev_turn = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    rev_net = (rev - rev_turn * RT_EQ / 10000.0 / 12).rename("rev")
    E = pd.concat([dp_net, rev_net], axis=1).dropna()
    vdp = E["dp"].rolling(12).std().shift(1)
    vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    return (w * E["dp"] + (1 - w) * E["rev"]).dropna().rename("equity_book")


def build_equity_book_pit_sn() -> pd.Series:
    """PIT SN deploy variant: PIT FF12 within-sector ranked D_PEAD + analyst
    revision, inverse-vol weighted. Replaces build_equity_book() once paper
    trade validates PIT SN improvement (Sharpe 1.06 → 2.10 gross, +13.71%/yr
    alpha t=9.65 after BARRA Phase 3 control).

    Library entry: data/research/mechanism_library/post_earnings_drift_pit_sn.yaml
    (audited 2026-05-31, cost_model + factor_exposure + capacity all green).
    """
    from engine.validation.analyst_revision import build_revision_sleeve_buffered
    sn = pd.read_parquet("data/cache/_dpead_sn_pit_monthly.parquet").iloc[:, 0]
    sn.index = pd.to_datetime(sn.index)
    # PIT SN parquet is already monthly returns — apply scalar bps cost only
    # (no per-month aggregation needed). Cost: RT_EQ × monthly_turnover (0.5)
    # = 30 × 0.5 / 10000 / 1 = 15bp per month per side, 2 sides = 30bp/mo gross
    # equivalent of 5.0 × RT_EQ / 10000 / 12 baseline (sn has higher turnover
    # so net cost roughly 1.25x parent).
    dp_net = (sn - 6.0 * RT_EQ / 10000.0 / 12).rename("dp_pit_sn")
    rev, rev_turn = build_revision_sleeve_buffered(q_in=0.2, q_out=0.4, weight="equal", disp_pctile=0.5)
    rev_net = (rev - rev_turn * RT_EQ / 10000.0 / 12).rename("rev")
    E = pd.concat([dp_net, rev_net], axis=1).dropna()
    vdp = E["dp_pit_sn"].rolling(12).std().shift(1)
    vre = E["rev"].rolling(12).std().shift(1)
    w = (1 / vdp) / (1 / vdp + 1 / vre)
    return (w * E["dp_pit_sn"] + (1 - w) * E["rev"]).dropna().rename("equity_book_pit_sn")


def build_tsmom_book() -> pd.Series:
    """TSMOM 机制 = 5-leg cross-asset futures TSMOM (commodity 24 + FX 9 + US rates 4
    + G10 rates XC 7 + equity indices 4), Moskowitz-Ooi-Pedersen 2012 标准 12-1 信号,
    per-instrument 40% vol-scaled, risk-parity 5-leg 合并, 扣 5×RT_TS 月度成本.
    Spec 77 §11+§12 amendment 2026-05-29. GREEN: Sharpe 0.62 net / t 3.12 / deflSR 0.91."""
    from engine.validation.crossasset_tsmom import build_tsmom_sleeve_returns
    tsmom_g = build_tsmom_sleeve_returns()
    return (tsmom_g - 5.0 * RT_TS / 10000.0 / 12).rename("tsmom")


def build_crisis_hedge_book() -> pd.Series:
    """TLT/GLD 50-50 monthly rebalance crisis-hedge sleeve (Path 1 risk
    management overlay, 2026-05-30 amendment). Source:
    engine.portfolio.crisis_hedge_tlt_gld_extended.run_ac_backtest.
    Returns monthly net return series (cost already inside RT_CH × turnover).
    """
    from engine.portfolio.crisis_hedge_tlt_gld_extended import run_ac_backtest
    r = run_ac_backtest()
    weekly_gross = r.weekly_returns_gross.copy()
    weekly_gross.index = pd.to_datetime(weekly_gross.index)
    monthly_gross = ((1 + weekly_gross).resample("ME").prod() - 1).rename("crisis")
    # Apply monthly cost: 12 rebals/yr × RT_CH bp/side / 10000 / 12 = RT_CH/10000 per yr
    # Per-month cost = RT_CH/10000/12 (matches the carry-book pattern)
    return (monthly_gross - 12.0 * RT_CH / 10000.0 / 12.0).rename("crisis_hedge")


def build_mom_hedge_book() -> pd.Series:
    """MTUM-short MOM-factor hedge (Path 2 risk management overlay,
    2026-05-30 amendment). Cost is INTERNAL to mom_hedge_overlay (borrow
    + spread + TC already netted in run_mom_hedge_backtest).
    """
    from engine.portfolio.mom_hedge_overlay import run_mom_hedge_backtest
    r = run_mom_hedge_backtest()
    return r.monthly_returns_net.rename("mom_hedge")


def build_vix_regime_monthly(z_window_days: int = VIX_Z_WINDOW_DAYS,
                                  threshold: float = REGIME_Z_THRESHOLD) -> pd.Series:
    """Monthly regime classification via VIX rolling z-score.

    Reuses cached VIX daily series (data/cache/_vix_spx_daily.parquet).
    Returns Series indexed by month-end with values in {'CALM','NORMAL',
    'STRESS'}.

    Simplified from AN-1's VIX+OAS composite due to FRED reliability
    issues in 2026-05-30 session; full 2-signal upgrade pending.
    """
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / "cache" / "_vix_spx_daily.parquet"
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    vix = df["VIX"].dropna()
    med = vix.rolling(z_window_days, min_periods=z_window_days).median()
    std = vix.rolling(z_window_days, min_periods=z_window_days).std()
    z = ((vix - med) / std).dropna()
    monthly_z = z.resample("ME").last()
    def _classify(zv):
        if pd.isna(zv): return "NORMAL"
        if zv > threshold: return "STRESS"
        if zv < -threshold: return "CALM"
        return "NORMAL"
    return monthly_z.apply(_classify).rename("regime")


def build_carry_book() -> pd.Series:
    """carry 机制 = 4-leg cross-asset carry(commodity + FX + US-rates + G10-rates-XC),
    反向波动率合并(扣成本)。Spec 77 §9+§10 amendments 2026-05-28 把这条从 2-leg
    升级到 4-leg(deduped data + US rates + G10 cross-country),Sharpe 0.66 → 1.10
    (IS), OOS 0.83 — 见 [[project-cross-asset-breadth-focus-2026-05-28]]."""
    from engine.portfolio.carry_sleeve import risk_parity_combine
    from engine.validation.crossasset_carry import (
        build_commodity_carry_ls, build_fx_carry, build_rates_carry, build_rates_xc_carry,
    )
    legs = {
        "cmdty":    build_commodity_carry_ls(),
        "fx":       build_fx_carry()[2],
        "rates_us": build_rates_carry()[2],
        "rates_xc": build_rates_xc_carry()[2],
    }
    carry_g = risk_parity_combine(legs)
    return (carry_g - 4.0 * RT_CY / 10000.0 / 12).rename("carry")


def scale_to_book_vol(book: pd.Series, book_vol_target: float) -> pd.Series:
    """把整本账本线性缩放到目标年化波动(尺寸/杠杆选择)。线性缩放 ⇒ Sharpe 不变，
    收益与回撤同步 1:1 放大。这是部署时的 sizing 旋钮。"""
    rv = book.dropna().std() * np.sqrt(12)
    if rv and rv > 0:
        return (book * (book_vol_target / rv)).rename(getattr(book, "name", "book"))
    return book


def blend_three(equity_vt: pd.Series, carry_vt: pd.Series, tsmom_vt: pd.Series,
                carry_risk_weight: float = DEFAULT_CARRY_RISK_WEIGHT,
                tsmom_risk_weight: float = DEFAULT_TSMOM_RISK_WEIGHT) -> pd.Series:
    """3-机制风险预算混合(equity / carry / tsmom)。所有腿都已 vol-target,所以
    risk weight = nominal weight。equity_weight = 1 - carry_weight - tsmom_weight。
    pre-committed weights,non-grid-search,符合 strict-gate doctrine。"""
    if carry_risk_weight + tsmom_risk_weight > 1.0:
        raise ValueError(
            f"carry+tsmom risk weight {carry_risk_weight + tsmom_risk_weight:.2f} > 1.0; "
            "equity weight cannot be negative")
    eq_w = 1.0 - carry_risk_weight - tsmom_risk_weight
    J = pd.concat([equity_vt.rename("e"), carry_vt.rename("c"), tsmom_vt.rename("t")], axis=1).dropna()
    return (eq_w * J["e"] + carry_risk_weight * J["c"] + tsmom_risk_weight * J["t"]).rename("book")


def blend_five(equity_vt: pd.Series, carry_vt: pd.Series, tsmom_vt: pd.Series,
                  crisis_vt: pd.Series, mom_hedge_vt: pd.Series,
                  carry_risk_weight: float = DEFAULT_CARRY_RISK_WEIGHT,
                  tsmom_risk_weight: float = DEFAULT_TSMOM_RISK_WEIGHT,
                  crisis_risk_weight: float = DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
                  mom_hedge_risk_weight: float = DEFAULT_MOM_HEDGE_RISK_WEIGHT,
                  ) -> pd.Series:
    """5-mechanism risk-budget blend including 2 hedge sleeves.
    equity_weight = 1 - carry - tsmom - crisis - mom_hedge.
    All legs are pre-vol-targeted so nominal weight = risk weight.
    """
    sum_non_eq = (carry_risk_weight + tsmom_risk_weight
                    + crisis_risk_weight + mom_hedge_risk_weight)
    if sum_non_eq >= 1.0:
        raise ValueError(
            f"sum of non-equity weights {sum_non_eq:.3f} >= 1.0; "
            f"equity weight would be non-positive"
        )
    eq_w = 1.0 - sum_non_eq
    J = pd.concat([equity_vt.rename("e"), carry_vt.rename("c"),
                       tsmom_vt.rename("t"), crisis_vt.rename("h"),
                       mom_hedge_vt.rename("m")], axis=1).dropna()
    return (eq_w * J["e"]
              + carry_risk_weight * J["c"]
              + tsmom_risk_weight * J["t"]
              + crisis_risk_weight * J["h"]
              + mom_hedge_risk_weight * J["m"]).rename("book")


def build_combined_book_regime_conditional(
    target_vol: float = DEFAULT_TARGET_VOL,
    book_vol_target: float = DEFAULT_BOOK_VOL_TARGET,
    grids: dict[str, dict[str, float]] | None = None,
) -> pd.Series:
    """Build the deployed 5-sleeve book with REGIME-CONDITIONAL hedge
    weights (the C config per 2026-05-30 deploy decision).

    Each month uses the regime-grid weights of the active VIX regime
    (CALM/NORMAL/STRESS per ±1σ threshold). equity weight floats to
    consume residual = 1 - carry - tsmom - crisis - mom_hedge.

    This is the INSTITUTIONAL-STANDARD risk management approach:
    pay insurance premium only in regimes that signal need.
    """
    grids = grids or REGIME_HEDGE_GRIDS
    eq_vt = voltarget(build_equity_book(), target_vol)
    cy_vt = voltarget(build_carry_book(), target_vol)
    ts_vt = voltarget(build_tsmom_book(), target_vol)
    crisis_vt = voltarget(build_crisis_hedge_book(), target_vol)
    momh_vt = voltarget(build_mom_hedge_book(), target_vol)
    J = pd.concat([eq_vt.rename("e"), cy_vt.rename("c"), ts_vt.rename("t"),
                       crisis_vt.rename("h"), momh_vt.rename("m")], axis=1).dropna()

    regime = build_vix_regime_monthly()
    book = pd.Series(index=J.index, dtype=float, name="book")
    for t in J.index:
        nearest = regime.index[regime.index <= t]
        r = "NORMAL" if len(nearest) == 0 else regime.loc[nearest[-1]]
        g = grids[r]
        crisis_w = g["crisis"]
        mom_w = g["mom_hedge"]
        eq_w = (1.0 - DEFAULT_CARRY_RISK_WEIGHT - DEFAULT_TSMOM_RISK_WEIGHT
                  - crisis_w - mom_w)
        book.loc[t] = (eq_w * J.loc[t, "e"]
                          + DEFAULT_CARRY_RISK_WEIGHT * J.loc[t, "c"]
                          + DEFAULT_TSMOM_RISK_WEIGHT * J.loc[t, "t"]
                          + crisis_w * J.loc[t, "h"]
                          + mom_w * J.loc[t, "m"])
    book = book.dropna()
    if book_vol_target:
        book = scale_to_book_vol(book, book_vol_target)
    return book


def build_combined_book(carry_risk_weight: float = DEFAULT_CARRY_RISK_WEIGHT,
                        target_vol: float = DEFAULT_TARGET_VOL,
                        book_vol_target: float | None = None,
                        tsmom_risk_weight: float = DEFAULT_TSMOM_RISK_WEIGHT,
                        crisis_risk_weight: float = 0.0,
                        mom_hedge_risk_weight: float = 0.0,
                        regime_conditional: bool = False,
                        ) -> pd.Series:
    """部署 book 月度收益序列。Per-sleeve vol-target → 按预定 risk weight 混合。
    若给 book_vol_target,再线性缩放整本 book 到该目标波动。

    History:
      2026-05-29 spec 77 §11+§12: 3-mechanism (equity 70 / carry 25 / tsmom 5).
      2026-05-30 5-sleeve risk-mgmt amendment: + crisis_hedge 5% + mom_hedge 2%
        funded by reducing equity to 63%. Pass crisis_risk_weight=0 +
        mom_hedge_risk_weight=0 to reproduce the pre-amendment 3-mechanism book.

    Calling convention (matches all preceding amendments):
      build_combined_book(book_vol_target=0.10)
        -> pre-amendment 3-mechanism book (DEFAULT behavior preserved).
      build_combined_book(book_vol_target=0.10,
                          crisis_risk_weight=DEFAULT_CRISIS_HEDGE_RISK_WEIGHT,
                          mom_hedge_risk_weight=DEFAULT_MOM_HEDGE_RISK_WEIGHT)
        -> NEW 5-sleeve hedged book.
    """
    if regime_conditional:
        return build_combined_book_regime_conditional(
            target_vol=target_vol,
            book_vol_target=book_vol_target or DEFAULT_BOOK_VOL_TARGET,
        )

    eq_vt = voltarget(build_equity_book(), target_vol)
    cy_vt = voltarget(build_carry_book(), target_vol)
    has_tsmom = tsmom_risk_weight > 0
    has_crisis = crisis_risk_weight > 0
    has_momh = mom_hedge_risk_weight > 0

    if has_crisis or has_momh:
        # 5-sleeve path
        if not has_tsmom:
            raise ValueError("5-sleeve config requires tsmom_risk_weight > 0")
        ts_vt = voltarget(build_tsmom_book(), target_vol)
        crisis_vt = voltarget(build_crisis_hedge_book(), target_vol)
        momh_vt = voltarget(build_mom_hedge_book(), target_vol)
        book = blend_five(
            eq_vt, cy_vt, ts_vt, crisis_vt, momh_vt,
            carry_risk_weight, tsmom_risk_weight,
            crisis_risk_weight, mom_hedge_risk_weight,
        )
    elif has_tsmom:
        ts_vt = voltarget(build_tsmom_book(), target_vol)
        book = blend_three(eq_vt, cy_vt, ts_vt,
                              carry_risk_weight, tsmom_risk_weight)
    else:
        book = blend_at_risk_weight(eq_vt, cy_vt, carry_risk_weight)

    if book_vol_target:
        book = scale_to_book_vol(book, book_vol_target)
    return book
