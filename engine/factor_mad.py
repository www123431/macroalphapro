"""
engine/factor_mad.py — P2-13 FactorMAD Alpha 因子自动挖掘引擎
==============================================================
Track A 纯量化模块，与 LLM 流程零交互。

四层防御架构：
  Layer 1: MI 污染扫描（前置，拦截统计前视）
  Layer 2: Proposer-Critic 辩论 + 验证集回测（核心，由 debate.py 驱动）
  Layer 3: 符号回归结构审计（后置，生成侦探报告）
  Layer 4: Supervisor 人工裁决（Admin UI）

本模块提供 Layer 1 + Layer 3 的统计工具，以及生产因子的
IC/ICIR 月度监控和复合评分计算。
"""
from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

from engine.memory import SessionFactory, FactorDefinition, FactorICIR, DiscoveredFactor
from engine.universe_manager import get_active_universe

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 已注册因子库（生产因子，直接计算 IC）
# ══════════════════════════════════════════════════════════════════════════════

FACTOR_REGISTRY: dict[str, callable] = {}


def register_factor(factor_id: str, description: str = ""):
    """装饰器：注册因子。签名：(prices: pd.DataFrame) -> pd.Series (index=sector名)"""
    def decorator(fn):
        FACTOR_REGISTRY[factor_id] = fn
        _ensure_factor_in_db(factor_id, description)
        return fn
    return decorator


def _ensure_factor_in_db(factor_id: str, description: str) -> None:
    """首次导入时将因子写入 factor_definitions 表（幂等）。"""
    try:
        with SessionFactory() as session:
            if not session.query(FactorDefinition).filter_by(factor_id=factor_id).first():
                session.add(FactorDefinition(factor_id=factor_id, description=description))
                session.commit()
    except Exception as exc:
        logger.debug("FactorMAD: DB write skipped (%s)", exc)


# ── 内置基准因子（无前视，作为 MI 校准白名单）──────────────────────────────────

@register_factor("mom_3m", "3个月动量（跳过最近1月，无前视）")
def factor_mom_3m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 65:
        return pd.Series(dtype=float)
    return prices.iloc[-65] / prices.iloc[-22] - 1


@register_factor("rev_1m", "1个月反转（负号）")
def factor_rev_1m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 22:
        return pd.Series(dtype=float)
    return -(prices.iloc[-1] / prices.iloc[-22] - 1)


@register_factor("vol_adj_mom_6m", "波动率调整6月动量")
def factor_vol_adj_mom_6m(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 130:
        return pd.Series(dtype=float)
    ret_6m  = prices.iloc[-130] / prices.iloc[-22] - 1
    vol_21d = prices.pct_change().iloc[-22:].std() * np.sqrt(252)
    return ret_6m / vol_21d.replace(0, np.nan)


@register_factor("trend_strength", "SMA200 偏离度（价格/200日均线 - 1）")
def factor_trend_strength(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 200:
        return pd.Series(dtype=float)
    sma200 = prices.iloc[-200:].mean()
    return prices.iloc[-1] / sma200 - 1


# ── MI 校准白名单 ──────────────────────────────────────────────────────────────
_BASELINE_FACTOR_IDS = ["mom_3m", "rev_1m", "vol_adj_mom_6m", "trend_strength"]
_MI_CONTAMINATION_MULTIPLIER = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1：MI 污染扫描
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_mi(
    factor_fn: callable,
    prices: pd.DataFrame,
    train_end: datetime.date,
    forward_return_days: int = 22,
    n_cross_sections: int = 24,
) -> float | None:
    """
    在训练集上估算因子值与未来收益之间的互信息（sklearn mutual_info_regression）。

    防前视机制：因子值用 t-forward_days 前的价格计算；
    未来收益用 t-forward_days → t 的窗口，两者不重叠。
    存在前视偏差的因子（使用了 t+1 数据）其 MI 会异常偏高。

    限制：MI 阈值为经验值，强因子 MI 本身也不低，建议积累 ≥10 个
    已知干净因子后重新校准 _MI_CONTAMINATION_MULTIPLIER。
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        logger.warning("sklearn 未安装，跳过 MI 扫描")
        return None

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    end_idx = prices.index[prices.index <= pd.Timestamp(train_end)]
    if len(end_idx) < forward_return_days * 2 + n_cross_sections:
        return None

    samples_factor, samples_fwd = [], []
    for i in range(n_cross_sections):
        offset   = i * forward_return_days + 1
        t_end_i  = end_idx[-(offset)]
        t_ref_i  = end_idx[-(offset + forward_return_days)]
        pre      = prices[prices.index <= t_ref_i]
        if len(pre) < 50:
            continue
        fvals = factor_fn(pre)
        if fvals.empty or fvals.isna().all():
            continue
        fwd = prices.loc[t_end_i] / prices.loc[t_ref_i] - 1
        df = pd.DataFrame({"f": fvals, "r": fwd}).dropna()
        if len(df) < 4:
            continue
        samples_factor.extend(df["f"].tolist())
        samples_fwd.extend(df["r"].tolist())

    if len(samples_factor) < 20:
        return None

    X = np.array(samples_factor).reshape(-1, 1)
    y = np.array(samples_fwd)
    mi = mutual_info_regression(X, y, n_neighbors=5, random_state=42)[0]
    return float(mi)


def scan_mi_contamination(
    candidate_fn: callable,
    prices: pd.DataFrame,
    train_end: datetime.date,
) -> dict:
    """
    Layer 1 入口。返回扫描结果 dict：
    {
        "candidate_mi": float | None,
        "baseline_mi_mean": float | None,
        "ratio": float | None,
        "flagged": bool,
        "reason": str,
    }
    flagged=True 时应终止该候选因子的辩论流程。
    """
    baseline_mis = []
    for fid in _BASELINE_FACTOR_IDS:
        fn = FACTOR_REGISTRY.get(fid)
        if fn is None:
            continue
        mi = compute_factor_mi(fn, prices, train_end)
        if mi is not None:
            baseline_mis.append(mi)

    if not baseline_mis:
        return {
            "candidate_mi": None, "baseline_mi_mean": None,
            "ratio": None, "flagged": False,
            "reason": "基准 MI 无法计算，跳过扫描",
        }

    baseline_mean = float(np.mean(baseline_mis))
    candidate_mi  = compute_factor_mi(candidate_fn, prices, train_end)

    if candidate_mi is None:
        return {
            "candidate_mi": None, "baseline_mi_mean": round(baseline_mean, 6),
            "ratio": None, "flagged": False,
            "reason": "候选因子 MI 无法计算，跳过扫描",
        }

    ratio   = candidate_mi / baseline_mean if baseline_mean > 1e-10 else 0.0
    flagged = ratio > _MI_CONTAMINATION_MULTIPLIER
    return {
        "candidate_mi":     round(candidate_mi, 6),
        "baseline_mi_mean": round(baseline_mean, 6),
        "ratio":            round(ratio, 3),
        "flagged":          flagged,
        "reason": (f"MI ratio={ratio:.2f}，超过阈值 {_MI_CONTAMINATION_MULTIPLIER}×，疑似统计前视"
                   if flagged else f"MI ratio={ratio:.2f}，未超阈值，通过"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3：符号回归结构审计
# ══════════════════════════════════════════════════════════════════════════════

def audit_factor_structure(
    candidate_fn: callable,
    candidate_description: str,
    prices: pd.DataFrame,
    train_end: datetime.date,
    n_cross_sections: int = 24,
    forward_return_days: int = 22,
) -> dict:
    """
    Layer 3 入口。使用 gplearn 符号回归尝试用基准因子拟合候选因子，
    输出三态报告给 Supervisor（非二值门控）。

    返回：
    {
        "signal_type": "positive" | "neutral" | "danger",
        "best_formula": str,
        "r2_train": float | None,
        "consistency_note": str,
        "raw_report": str,
    }

    限制：
    - 当前 18 ETF × 24 截面 ≈ 432 样本，拟合置信度有限；
      Phase 2 扩展到 25+ ETF 后效果显著改善。
    - R² 低不代表因子无效，Alpha 本身是噪音中的弱信号。
    - 一致性判断基于关键词匹配，存在误判，Supervisor 应自行解读公式。
    """
    try:
        from gplearn.genetic import SymbolicRegressor
    except ImportError:
        return {
            "signal_type": "neutral",
            "best_formula": "N/A",
            "r2_train": None,
            "consistency_note": "gplearn 未安装，跳过符号回归审计（pip install gplearn）",
            "raw_report": "gplearn not available",
        }

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()
    end_idx = prices.index[prices.index <= pd.Timestamp(train_end)]

    X_rows, y_vals = [], []
    baseline_fns = [FACTOR_REGISTRY[fid] for fid in _BASELINE_FACTOR_IDS
                    if fid in FACTOR_REGISTRY]

    for i in range(n_cross_sections):
        offset  = i * forward_return_days + 1
        if offset + forward_return_days >= len(end_idx):
            continue
        t_ref_i = end_idx[-(offset + forward_return_days)]
        pre     = prices[prices.index <= t_ref_i]
        if len(pre) < 50:
            continue
        cand_vals = candidate_fn(pre)
        if cand_vals.empty or cand_vals.isna().all():
            continue
        row_feats = []
        for bfn in baseline_fns:
            bvals = bfn(pre).reindex(cand_vals.index).fillna(0.0)
            row_feats.append(bvals.values)
        if not row_feats:
            continue
        X_rows.append(np.stack(row_feats, axis=1))
        y_vals.append(cand_vals.values)

    if not X_rows:
        return {
            "signal_type": "neutral", "best_formula": "无法收集训练数据",
            "r2_train": None, "consistency_note": "", "raw_report": "",
        }

    X = np.vstack(X_rows)
    y = np.concatenate(y_vals)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[valid], y[valid]

    if len(y) < 30:
        return {
            "signal_type": "neutral", "best_formula": "样本不足（<30）",
            "r2_train": None, "consistency_note": "无法进行符号回归", "raw_report": "",
        }

    try:
        from gplearn.genetic import SymbolicRegressor as _SymbolicRegressor
        sr = _SymbolicRegressor(
            population_size=500, generations=20, stopping_criteria=0.01,
            p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
            p_point_mutation=0.1, max_samples=0.9, verbose=0,
            parsimony_coefficient=0.01, random_state=42, n_jobs=1,
        )
        sr.fit(X, y)
    except Exception as exc:
        return {
            "signal_type": "neutral", "best_formula": f"拟合失败: {exc}",
            "r2_train": None, "consistency_note": "", "raw_report": str(exc),
        }

    formula_str = str(sr._program)
    r2          = float(sr.score(X, y))
    feat_names  = [fid for fid in _BASELINE_FACTOR_IDS if fid in FACTOR_REGISTRY]
    for i, fname in enumerate(feat_names):
        formula_str = formula_str.replace(f"X{i}", fname)

    signal_type, note = _classify_audit(r2, formula_str, candidate_description)
    raw_report = (
        f"=== 符号回归审计报告 ===\n"
        f"候选因子描述: {candidate_description}\n"
        f"最优公式: {formula_str}\n"
        f"训练集 R²: {r2:.4f}\n"
        f"信号类型: {signal_type}\n"
        f"解读: {note}\n"
    )
    return {
        "signal_type": signal_type,
        "best_formula": formula_str,
        "r2_train": round(r2, 4),
        "consistency_note": note,
        "raw_report": raw_report,
    }


_LOGIC_KEYWORDS = {
    "momentum", "reversal", "volatility", "vol", "trend", "sma", "moving average",
    "动量", "反转", "波动率", "趋势", "均线", "carry", "value", "quality",
}


def _classify_audit(r2: float, formula: str, description: str) -> tuple[str, str]:
    desc_kw    = {kw for kw in _LOGIC_KEYWORDS if kw in description.lower()}
    formula_kw = {kw for kw in _LOGIC_KEYWORDS if kw in formula.lower()}
    overlap    = desc_kw & formula_kw

    if r2 > 0.3:
        if overlap:
            return "positive", (
                f"R²={r2:.3f}，公式与宣称逻辑关键词重叠 {overlap}。"
                "可解释性获独立数学支持，可信度增强。"
            )
        return "danger", (
            f"R²={r2:.3f}，但公式关键词与宣称逻辑无重叠。"
            f"宣称: {desc_kw}，公式含: {formula_kw}。"
            "因子有效性来源可能与逻辑不符，建议 Supervisor 重点审查代码。"
        )
    return "neutral", (
        f"R²={r2:.3f}，符号回归无法用简单公式拟合。"
        "这可能说明该因子捕捉到了超出简单结构的规律（Alpha 典型特征），不构成否决理由。"
    )


# ══════════════════════════════════════════════════════════════════════════════
# IC 计算（生产因子月度监控）
# ══════════════════════════════════════════════════════════════════════════════

def compute_monthly_ic(
    factor_id: str,
    calc_date: datetime.date,
    lookback_prices_days: int = 280,
    forward_return_days: int = 22,
    asset_class: str = "equity_sector",
) -> float | None:
    """
    计算单因子在 calc_date 的月度 IC（Spearman）。
    防前视：因子值用 T-forward_days 的价格；实际收益 T-forward_days → T。
    """
    sector_etf = get_active_universe(asset_classes=[asset_class])
    if not sector_etf:
        return None

    tickers    = list(sector_etf.values())
    t_to_s     = {v: k for k, v in sector_etf.items()}
    start = calc_date - datetime.timedelta(days=lookback_prices_days + forward_return_days + 30)

    try:
        dl = yf.download(tickers, start=str(start),
                         end=str(calc_date + datetime.timedelta(days=5)),
                         progress=False, auto_adjust=True)
        if dl.empty:
            return None
        prices = dl["Close"] if "Close" in dl else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        prices = prices.reindex(columns=tickers).dropna(how="all")
    except Exception as exc:
        logger.warning("FactorMAD IC download error: %s", exc)
        return None

    prices.index = pd.to_datetime(prices.index).normalize()
    t_end   = prices.index[prices.index <= pd.Timestamp(calc_date)]
    if len(t_end) < forward_return_days + 20:
        return None

    t_ref  = t_end[-forward_return_days]
    t_last = t_end[-1]

    pre_prices = prices[prices.index <= t_ref]
    if len(pre_prices) < 50:
        return None

    factor_fn = FACTOR_REGISTRY.get(factor_id)
    if factor_fn is None:
        return None

    factor_vals = factor_fn(pre_prices).rename(index=t_to_s)
    fwd_rets    = (prices.loc[t_last] / prices.loc[t_ref] - 1).rename(index=t_to_s)

    combined = pd.DataFrame({"f": factor_vals, "r": fwd_rets}).dropna()
    if len(combined) < 4:
        return None

    ic, _ = spearmanr(combined["f"], combined["r"])
    return float(ic)


# ══════════════════════════════════════════════════════════════════════════════
# ICIR 月度更新（生命周期管理）
# ══════════════════════════════════════════════════════════════════════════════

_MIN_IC_MONTHS_FOR_DEACTIVATION = 24   # SE(IC)≈1/√24≈0.20; below this threshold has no power


def update_icir(calc_date: datetime.date, asset_class: str = "equity_sector") -> None:
    """
    每月调用一次。更新所有 active 因子的 IC，计算滚动 12 月 ICIR，
    连续 2 月 ICIR < 0.15 时自动标记 inactive。

    样本保护：历史 IC 记录 < 24 月时，无论 ICIR 多低均暂缓裁决。
    SE(IC) ≈ 1/√N；N=24 时 SE≈0.20，统计检验才有足够功效。
    """
    with SessionFactory() as session:
        active_factors = (
            session.query(FactorDefinition)
            .filter(FactorDefinition.active == True,
                    FactorDefinition.asset_class == asset_class)
            .all()
        )
        for fdef in active_factors:
            ic = compute_monthly_ic(fdef.factor_id, calc_date, asset_class=asset_class)
            if ic is None:
                continue

            row = FactorICIR(
                factor_id=fdef.factor_id,
                calc_date=calc_date,
                ic_value=ic,
                asset_class=asset_class,
            )
            session.merge(row)

            ic_history = (
                session.query(FactorICIR.ic_value)
                .filter(FactorICIR.factor_id == fdef.factor_id,
                        FactorICIR.asset_class == asset_class,
                        FactorICIR.calc_date <= calc_date)
                .order_by(FactorICIR.calc_date.desc())
                .limit(12)
                .all()
            )
            ic_vals = [r[0] for r in ic_history if r[0] is not None]
            row.icir_12m = (float(np.mean(ic_vals) / np.std(ic_vals))
                            if len(ic_vals) >= 3 and np.std(ic_vals) > 0 else None)

            n_active = len(get_active_universe(asset_classes=[asset_class]))
            row.n_assets = n_active

            if row.icir_12m is not None and row.icir_12m < 0.15:
                # Count total lifetime IC observations for this factor
                n_total_obs = (
                    session.query(FactorICIR)
                    .filter(FactorICIR.factor_id == fdef.factor_id,
                            FactorICIR.asset_class == asset_class,
                            FactorICIR.ic_value.isnot(None))
                    .count()
                )
                if n_total_obs < _MIN_IC_MONTHS_FOR_DEACTIVATION:
                    logger.info(
                        "FactorMAD: %s ICIR=%.3f<0.15 但历史IC仅%d月（<24）→ 样本不足，暂缓裁决",
                        fdef.factor_id, row.icir_12m, n_total_obs,
                    )
                else:
                    recent_bad = (
                        session.query(FactorICIR)
                        .filter(FactorICIR.factor_id == fdef.factor_id,
                                FactorICIR.icir_12m.isnot(None),
                                FactorICIR.icir_12m < 0.15)
                        .order_by(FactorICIR.calc_date.desc())
                        .limit(2)
                        .count()
                    )
                    if recent_bad >= 2:
                        fdef.active = False
                        logger.info("FactorMAD: %s 连续2月 ICIR<0.15 → inactive", fdef.factor_id)

        session.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 扩展：制度条件 ICIR（Regime-Conditional ICIR）
# ══════════════════════════════════════════════════════════════════════════════
#
# 动机：通用 ICIR（全样本）无法区分因子在 risk-on / risk-off 制度下的
# 表现差异。当 FactorMAD 维护制度专属因子池时（P2-X），制度 ICIR delta
# 是筛选"制度依赖型"因子的核心指标。
#
# 实现：用 VIX 阈值作为制度代理变量（无需 FRED 调用，避免跨截面 API 开销）。
#   VIX < 18  → risk-on
#   VIX > 25  → risk-off
#   其余       → transition（不纳入对比计算，保证两侧分布干净）
#
# 学术依据：Asness, Moskowitz & Pedersen (2013, JF) 实证了因子收益的制度依赖性；
# VIX 阈值分区参考 CBOE 官方宏观解读（complacency / normal / elevated）。
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimeICIR:
    factor_id:       str
    icir_risk_on:    float | None   # ICIR in VIX < 18 regime
    icir_risk_off:   float | None   # ICIR in VIX > 25 regime
    icir_full:       float | None   # ICIR over full sample (reference)
    delta:           float | None   # icir_risk_on - icir_risk_off
    n_risk_on:       int            # number of cross-sections in risk-on bucket
    n_risk_off:      int            # number of cross-sections in risk-off bucket
    interpretation:  str            # human-readable classification

    @property
    def is_regime_dependent(self) -> bool:
        """True if the factor shows meaningfully different ICIR across regimes."""
        if self.delta is None:
            return False
        return abs(self.delta) >= 0.20


def compute_regime_conditional_icir(
    factor_id:            str,
    as_of:                datetime.date,
    n_months:             int = 36,
    forward_return_days:  int = 22,
    asset_class:          str = "equity_sector",
    vix_risk_on_thresh:   float = 18.0,
    vix_risk_off_thresh:  float = 25.0,
) -> RegimeICIR | None:
    """
    Compute regime-split ICIR for a registered factor using historical IC records.

    Uses DB-stored IC values (populated by update_icir) and classifies each
    observation by VIX regime at the calculation date.

    Args:
        factor_id:           Registered factor ID (must exist in FACTOR_REGISTRY).
        as_of:               Evaluation date — uses IC history up to this date.
        n_months:            Look-back window for IC observations (default 36M).
        forward_return_days: Used to label IC observation horizon.
        asset_class:         Asset class filter for universe lookup.
        vix_risk_on_thresh:  VIX below this → risk-on (default 18).
        vix_risk_off_thresh: VIX above this → risk-off (default 25).

    Returns:
        RegimeICIR dataclass, or None if insufficient history.
    """
    try:
        from engine.history import get_vix_on
    except ImportError:
        logger.warning("RegimeICIR: cannot import get_vix_on from history")
        return None

    cutoff = as_of - datetime.timedelta(days=n_months * 31)

    with SessionFactory() as session:
        rows = (
            session.query(FactorICIR.calc_date, FactorICIR.ic_value)
            .filter(
                FactorICIR.factor_id   == factor_id,
                FactorICIR.asset_class == asset_class,
                FactorICIR.calc_date   >= cutoff,
                FactorICIR.calc_date   <= as_of,
                FactorICIR.ic_value.isnot(None),
            )
            .order_by(FactorICIR.calc_date)
            .all()
        )

    if len(rows) < 6:
        return None

    ic_on,  ic_off,  ic_all = [], [], []
    for row_date, ic_val in rows:
        if isinstance(row_date, str):
            row_date = datetime.date.fromisoformat(row_date)
        vix = get_vix_on(row_date)
        ic_all.append(ic_val)
        if vix is not None:
            if vix < vix_risk_on_thresh:
                ic_on.append(ic_val)
            elif vix > vix_risk_off_thresh:
                ic_off.append(ic_val)
        # VIX in [18, 25] → transition; excluded from regime comparison

    def _icir(vals: list[float]) -> float | None:
        if len(vals) < 3:
            return None
        arr = np.array(vals)
        std = float(np.std(arr))
        return float(np.mean(arr) / std) if std > 1e-9 else None

    icir_on   = _icir(ic_on)
    icir_off  = _icir(ic_off)
    icir_full = _icir(ic_all)
    delta     = (icir_on - icir_off) if (icir_on is not None and icir_off is not None) else None

    if delta is None:
        interp = "制度样本不足，无法计算 ICIR delta"
    elif delta >= 0.30:
        interp = f"强 risk-on 型因子（delta={delta:+.2f}）：在低 VIX 环境下显著更有效"
    elif delta <= -0.30:
        interp = f"强 risk-off 型因子（delta={delta:+.2f}）：在高 VIX 环境下显著更有效"
    elif abs(delta) >= 0.15:
        interp = f"弱制度依赖（delta={delta:+.2f}）：存在方向性偏好但不显著"
    else:
        interp = f"制度中性因子（delta={delta:+.2f}）：在两种制度下表现相近"

    return RegimeICIR(
        factor_id=factor_id,
        icir_risk_on=round(icir_on,  3) if icir_on  is not None else None,
        icir_risk_off=round(icir_off, 3) if icir_off is not None else None,
        icir_full=round(icir_full,    3) if icir_full is not None else None,
        delta=round(delta, 3) if delta is not None else None,
        n_risk_on=len(ic_on),
        n_risk_off=len(ic_off),
        interpretation=interp,
    )


def get_all_regime_icirs(
    as_of:       datetime.date,
    n_months:    int = 36,
    asset_class: str = "equity_sector",
) -> list[RegimeICIR]:
    """
    Compute regime-conditional ICIR for all registered active factors.
    Returns list sorted by abs(delta) descending — most regime-sensitive first.
    """
    with SessionFactory() as session:
        factor_ids = [
            r[0] for r in session.query(FactorICIR.factor_id)
            .filter(FactorICIR.asset_class == asset_class,
                    FactorICIR.calc_date   <= as_of)
            .distinct()
            .all()
        ]

    results = []
    for fid in factor_ids:
        r = compute_regime_conditional_icir(fid, as_of, n_months=n_months,
                                            asset_class=asset_class)
        if r is not None:
            results.append(r)

    results.sort(key=lambda r: abs(r.delta) if r.delta is not None else 0.0, reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 复合 FactorMAD 信号（集成到 signal.py）
# ══════════════════════════════════════════════════════════════════════════════

def get_factor_mad_scores(
    as_of: datetime.date,
    asset_class: str = "equity_sector",
    min_factors: int = 3,
) -> pd.Series | None:
    """
    返回 FactorMAD 复合截面得分（0-100），index=sector 名。
    若活跃因子数 < min_factors，返回 None（退化到当前权重方案）。
    """
    with SessionFactory() as session:
        active_factors = (
            session.query(FactorDefinition)
            .filter(FactorDefinition.active == True,
                    FactorDefinition.asset_class == asset_class)
            .all()
        )

    if len(active_factors) < min_factors:
        return None

    sector_etf = get_active_universe(asset_classes=[asset_class])
    tickers    = list(sector_etf.values())
    t_to_s     = {v: k for k, v in sector_etf.items()}
    start      = as_of - datetime.timedelta(days=280)

    try:
        dl = yf.download(tickers,
                         start=str(start),
                         end=str(as_of + datetime.timedelta(days=2)),
                         progress=False, auto_adjust=True)
        prices = dl["Close"] if "Close" in dl else dl
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = [c[0] for c in prices.columns]
        prices.index = pd.to_datetime(prices.index).normalize()
    except Exception:
        return None

    factor_scores: list[pd.Series] = []
    factor_weights: list[float]    = []

    with SessionFactory() as session:
        for fdef in active_factors:
            fn = FACTOR_REGISTRY.get(fdef.factor_id)
            if fn is None:
                continue
            vals = fn(prices).rename(index=t_to_s)
            if vals.empty or vals.isna().all():
                continue
            ranked = vals.rank(pct=True) * 100
            icir_row = (
                session.query(FactorICIR.icir_12m)
                .filter(FactorICIR.factor_id == fdef.factor_id,
                        FactorICIR.calc_date <= as_of,
                        FactorICIR.icir_12m.isnot(None))
                .order_by(FactorICIR.calc_date.desc())
                .first()
            )
            if icir_row is None:
                w = 0.1   # new factor — no ICIR history yet, small equal weight
            else:
                icir_val = float(icir_row[0])
                if icir_val < 0.05:
                    continue  # ICIR below noise floor — exclude from aggregation
                w = icir_val
            factor_scores.append(ranked)
            factor_weights.append(w)

    if not factor_scores:
        return None

    total_w   = sum(factor_weights) or 1.0
    composite = sum(s * w for s, w in zip(factor_scores, factor_weights)) / total_w
    return composite.round(1)


# ══════════════════════════════════════════════════════════════════════════════
# DiscoveredFactor 管理（候选因子审批流）
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FactorScanResult:
    passed: bool
    mi_report: dict
    audit_report: dict | None = None


def submit_candidate_factor(
    name: str,
    description: str,
    code_snippet: str,
    debate_log: str,
    ic_train: float,
    icir_train: float,
    ic_test: float,
    icir_test: float,
    correlation_with_existing: float,
    mi_report: dict,
    audit_report: dict | None = None,
) -> int:
    """
    将通过 Layer 2 测试集验证的候选因子写入 discovered_factors 表（status=pending）。
    返回新记录的 id。
    """
    with SessionFactory() as session:
        df = DiscoveredFactor(
            name=name,
            description=description,
            code_snippet=code_snippet,
            debate_log=debate_log,
            ic_train=ic_train,
            icir_train=icir_train,
            ic_test=ic_test,
            icir_test=icir_test,
            correlation_with_existing=correlation_with_existing,
            mi_ratio=mi_report.get("ratio"),
            audit_signal_type=audit_report.get("signal_type") if audit_report else None,
            audit_report=audit_report.get("raw_report") if audit_report else None,
            status="pending",
        )
        session.add(df)
        session.commit()
        return df.id


def approve_factor(factor_id_db: int, factor_id_code: str) -> None:
    """Supervisor 批准：写入 FactorDefinition 激活生产因子。"""
    with SessionFactory() as session:
        df = session.get(DiscoveredFactor, factor_id_db)
        if df is None:
            raise ValueError(f"DiscoveredFactor id={factor_id_db} not found")
        df.status      = "active"
        df.activated_at = datetime.datetime.utcnow()

        if not session.query(FactorDefinition).filter_by(factor_id=factor_id_code).first():
            session.add(FactorDefinition(
                factor_id=factor_id_code,
                description=df.description,
            ))
        session.commit()


def reject_factor(factor_id_db: int, reason: str) -> None:
    """Supervisor 驳回。"""
    with SessionFactory() as session:
        df = session.get(DiscoveredFactor, factor_id_db)
        if df:
            df.status           = "rejected"
            df.rejection_reason = reason
            session.commit()


def defer_factor(factor_id_db: int, reason: str) -> None:
    """Supervisor 要求补充验证（pending_further_review）。"""
    with SessionFactory() as session:
        df = session.get(DiscoveredFactor, factor_id_db)
        if df:
            df.status           = "pending_further_review"
            df.rejection_reason = reason
            session.commit()


# ══════════════════════════════════════════════════════════════════════════════
# P6: Harvey-Liu t-stat (display only) + BH multiple-testing correction
# ══════════════════════════════════════════════════════════════════════════════

def compute_harvey_liu_t(
    factor_id: str,
    as_of: datetime.date,
    asset_class: str = "equity_sector",
) -> float | None:
    """
    Harvey-Liu t-statistic for a registered factor — display only (M3 resolution).

    t = mean(IC) / (std(IC) / √n)  =  ICIR × √n

    Harvey, Liu & Zhu (2016) recommend t ≥ 3.0 for a factor to be considered
    genuine after accounting for selection bias (vs. t ≥ 2.0 for a priori hypotheses).
    The Layer 2 gate remains ICIR ≥ 0.3 — this stat is informational only.
    """
    with SessionFactory() as s:
        ic_records = (
            s.query(FactorICIR.ic_value)
             .filter(
                 FactorICIR.factor_id == factor_id,
                 FactorICIR.asset_class == asset_class,
                 FactorICIR.ic_value.isnot(None),
                 FactorICIR.calc_date <= as_of,
             )
             .all()
        )
    ic_vals = [float(r[0]) for r in ic_records if r[0] is not None]
    n = len(ic_vals)
    if n < 3:
        return None
    mean_ic = float(np.mean(ic_vals))
    std_ic  = float(np.std(ic_vals, ddof=1))
    if std_ic < 1e-9:
        return None
    return round(mean_ic / (std_ic / np.sqrt(n)), 3)


def run_quarterly_bh_correction(
    as_of: datetime.date,
    asset_class: str = "equity_sector",
    alpha: float = 0.10,
) -> list[str]:
    """
    Benjamini-Hochberg FDR correction (α=10%) on pending DiscoveredFactor candidates.

    Each candidate's p-value is derived from its icir_test via a t-approximation
    with n=24 test-window months (conservative — actual n may differ).
    Candidates that pass BH are promoted to PendingApproval(factor_candidate) for
    human review. Returns list of candidate names that pass.

    Benjamini & Hochberg (1995) "Controlling the false discovery rate."
    """
    from engine.memory import PendingApproval

    with SessionFactory() as s:
        candidates = (
            s.query(DiscoveredFactor)
             .filter(
                 DiscoveredFactor.status.in_(["pending", "pending_further_review"]),
                 DiscoveredFactor.icir_test.isnot(None),
             )
             .all()
        )
    if not candidates:
        return []

    # ── Compute one-sided p-values from t ≈ ICIR × √24 ─────────────────────
    N_TEST = 24   # conservative test-window assumption
    import math as _math

    def _t_to_p(t: float, df: int) -> float:
        """Approximate one-sided p-value from t using normal tail for large df."""
        try:
            from scipy.stats import t as _t
            return float(1.0 - _t.cdf(t, df=df))
        except ImportError:
            # Normal approximation fallback
            z = t
            return float(0.5 * _math.erfc(z / _math.sqrt(2)))

    pval_rows: list[tuple[DiscoveredFactor, float]] = []
    for cand in candidates:
        icir = float(cand.icir_test or 0.0)
        t_stat = icir * _math.sqrt(N_TEST)
        p = _t_to_p(t_stat, df=N_TEST - 2)
        pval_rows.append((cand, p))

    # ── BH procedure ──────────────────────────────────────────────────────────
    m = len(pval_rows)
    sorted_rows = sorted(pval_rows, key=lambda x: x[1])
    bh_cutoff = 0
    for k, (_, p) in enumerate(sorted_rows, 1):
        if p <= (k / m) * alpha:
            bh_cutoff = k

    if bh_cutoff == 0:
        logger.info("FactorMAD BH: 0/%d candidates pass FDR α=%.0f%% on %s", m, alpha * 100, as_of)
        return []

    passing = [sorted_rows[i][0] for i in range(bh_cutoff)]
    passing_names: list[str] = []

    # ── Write PendingApproval for each passing candidate ─────────────────────
    import datetime as _dt
    with SessionFactory() as s:
        for cand in passing:
            existing = (
                s.query(PendingApproval)
                 .filter(
                     PendingApproval.approval_type == "factor_candidate",
                     PendingApproval.status == "pending",
                     PendingApproval.sector == cand.name,
                 )
                 .first()
            )
            if not existing:
                s.add(PendingApproval(
                    approval_type="factor_candidate",
                    priority="normal",
                    sector=cand.name,
                    ticker="N/A",
                    triggered_condition=(
                        f"FactorMAD BH correction: ICIR_test={cand.icir_test:.3f} "
                        f"passes BH FDR α={alpha:.0%} (m={m} candidates, n_test≈{N_TEST}M)"
                    ),
                    triggered_date=as_of,
                    suggested_weight=None,
                ))
            passing_names.append(cand.name)
        s.commit()

    logger.info(
        "FactorMAD BH: %d/%d candidates pass FDR α=%.0f%% on %s: %s",
        len(passing), m, alpha * 100, as_of, passing_names,
    )
    return passing_names
