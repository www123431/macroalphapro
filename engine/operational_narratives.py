"""
Operational decision narratives — deterministic templating, 0 LLM.

Two composers covering the bottom-of-Operations decision panels:
  - compose_strategy_health_narrative(cadence, drift, events) -> str
        Decision A: backtest cadence + param drift + trigger events
  - compose_risk_posture_narrative(exposure, concentration, pairs, dd) -> str
        Decision C: gross/net exposure + per-ticker / pair concentration + dd

Output is markdown-formatted, structured per Tetlock 2015 *Superforecasting*
three-step (NATO STANAG 2014 brief format):
  1. WHAT  — current state (data points)
  2. SO WHAT — interpretation against pre-registered thresholds
  3. NOW WHAT — concrete next steps the supervisor should take

Same-input → byte-identical output. No LLM call. Any change to thresholds
or templates triggers a spec amendment per project pre-registration policy.
"""
from __future__ import annotations

from typing import Any


def _bullet(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items if x)


def compose_strategy_health_narrative(cadence: Any, drift: list, events: list) -> str:
    """Decision A — should we re-run the backtest now?"""
    paras: list[str] = []

    # ── WHAT ────────────────────────────────────────────────────────────
    days_since = getattr(cadence, "days_since", None)
    days_since_str = f"{days_since}d" if days_since is not None else "never"
    next_due_d = getattr(cadence, "next_quarterly_due", None)
    days_until = getattr(cadence, "days_until_quarterly", None)
    next_due_str = (
        f"{next_due_d} (in {days_until}d)" if days_until is not None and days_until >= 0
        else f"{next_due_d} ({abs(days_until or 0)}d overdue)" if days_until is not None
        else "—"
    )
    n_diverged = sum(1 for r in (drift or []) if getattr(r, "diverged", False))
    n_drift_total = len(drift or [])
    n_triggered = sum(1 for e in (events or []) if getattr(e, "triggered", False))
    n_struct = sum(1 for e in (events or [])
                    if getattr(e, "triggered", False) and getattr(e, "severity", "") == "structural")

    paras.append("**当前状态**")
    paras.append(_bullet([
        f"距上次 Tier-A 回测 **{days_since_str}**；下次 quarterly 截止日 {next_due_str}",
        f"参数漂移：{n_diverged} / {n_drift_total} 项与上次回测不一致",
        (f"触发事件：{n_triggered} 个文件改动（{n_struct} 项 structural）"
         if n_triggered else "触发事件：无关键文件改动"),
    ]))

    # ── SO WHAT ─────────────────────────────────────────────────────────
    paras.append("**这说明什么**")
    so_bullets: list[str] = []
    status = getattr(cadence, "status", "green")
    if status == "red":
        so_bullets.append(
            "Cadence 状态 **red** — 回测过期或参数漂移已触发硬阈值，再用旧 baseline 评判新策略风险高"
        )
    elif status == "yellow":
        so_bullets.append("Cadence 状态 yellow — 接近重测窗口，结果还可信但越接近 quarterly 越不稳")
    else:
        so_bullets.append("Cadence 状态 green — 当前回测仍在有效窗口内")

    if n_diverged > 0:
        so_bullets.append(
            f"参数 drift {n_diverged} 项触发 Hansen (2005) pre-registration 红线 — "
            f"未重测前任何 backtest 引用属于 retrospective tuning"
        )
    if n_struct > 0:
        so_bullets.append(
            f"{n_struct} 个 structural 文件 (signal / regime / portfolio 核心) 已改 — "
            f"语义可能漂移，旧 backtest 不再代表当前 logic"
        )
    paras.append(_bullet(so_bullets))

    # ── NOW WHAT ────────────────────────────────────────────────────────
    paras.append("**下一步建议**")
    nw_bullets: list[str] = []
    if status == "red" or n_diverged > 0 or n_struct > 0:
        nw_bullets.append(
            "进 **Backtest** 页面 → 选 quarterly window → 跑完整 Tier-A baseline"
        )
        nw_bullets.append(
            "关注数值：DSR > 0、NW t > 1.5、PBO < 30%（spec_power_analysis 阈值）"
        )
        nw_bullets.append(
            "任何一项不达标 → 通过 `engine.preregistration.amend_spec(kind=\"threshold_tweak\")` "
            "走 S3 amendment（消耗 +1 EFFECTIVE_N_TRIALS）；否则属 retrospective tuning"
        )
        nw_bullets.append(
            "数值达标 → 在 Operations 标记 backtest_run.id 完成 → cadence 自动 reset 为 green"
        )
    elif status == "yellow":
        nw_bullets.append("无紧急动作；下一次 quarterly 启动前 1-2 周准备重测")
    else:
        nw_bullets.append("无需操作；继续每日审批流；关注 trigger events 增量")

    if n_triggered > 0 and (status == "green"):
        nw_bullets.append(
            f"⚠ 文件已改但 status 仍 green → 检查改动是否仅 cosmetic（"
            "comment / formatting）；若是 structural 应人工把 cadence 标 red 并立即重测"
        )
    paras.append(_bullet(nw_bullets))

    return "\n\n".join(paras)


def compose_exposure_narrative(
    weights:         list[float],
    conc:            dict,
    n_long:          int,
    n_short:         int,
    gross_limit:     float = 1.0,
    net_limit_high:  float = 1.0,
    net_limit_low:   float = -1.0,
    hhi_warn:        float = 0.20,
) -> str:
    """
    Risk Console — Exposures tab analyst note.
    Markowitz 1952 / Sharpe 1964 / Pedersen 2015 §4 risk budgeting.
    """
    long_w  = sum(w for w in weights if w > 0)
    short_w = sum(w for w in weights if w < 0)
    net  = long_w + short_w
    gross = long_w + abs(short_w)
    hhi   = float(conc.get("hhi", 0.0) or 0.0)
    top1  = float(conc.get("top1_pct", 0.0) or 0.0)
    top5  = float(conc.get("top5_pct", 0.0) or 0.0)

    paras: list[str] = []
    paras.append("**当前状态**")
    paras.append(_bullet([
        f"Gross **{gross:.1%}** / cap {gross_limit:.0%}；Net **{net:+.1%}** / 区间 [{net_limit_low:+.0%}, {net_limit_high:+.0%}]",
        f"持仓数：**{n_long}L / {n_short}S**（合计 {n_long + n_short}）",
        f"集中度：HHI **{hhi:.3f}**, top1 {top1:.1%}, top5 {top5:.1%}",
    ]))

    paras.append("**这说明什么**")
    so: list[str] = []
    if gross > gross_limit:
        so.append(f"Gross 超 cap — 杠杆超限，必须减仓（Markowitz constraint）")
    elif gross > gross_limit * 0.85:
        so.append(f"Gross 接近 cap — 加仓空间有限")
    else:
        so.append("Gross 健康")
    if net > net_limit_high:
        so.append(f"Net 超上限 {net_limit_high:+.0%} — 单边过多头")
    elif net < net_limit_low:
        so.append(f"Net 低于下限 {net_limit_low:+.0%} — 单边过空头")
    if hhi > hhi_warn:
        so.append(f"HHI {hhi:.3f} > {hhi_warn:.2f} 警戒线 — 集中度过高（Hirschman 1964）")
    if top1 > 0.10:
        so.append(f"Top1 持仓 {top1:.1%} > 10% — 单仓风险敞口大")
    if not so:
        so.append("各维度均在正常区间")
    paras.append(_bullet(so))

    paras.append("**下一步建议**")
    nw: list[str] = []
    if gross > gross_limit or net > net_limit_high or net < net_limit_low:
        nw.append("**立即减仓** — 进 Positions 页选高 |w| ticker → 主动减仓审批")
    if hhi > hhi_warn:
        nw.append("减低集中度：下次 rebalance 优先 trim top1 / top5 持仓")
    if top1 > 0.10:
        nw.append(f"Top1 单仓 {top1:.1%} 已显著 — 加仓决策应避开此 ticker")
    if not nw:
        nw.append("无紧急动作；维持当前 exposure posture")
    paras.append(_bullet(nw))
    return "\n\n".join(paras)


def compose_stress_narrative(
    scenarios:    list[dict],     # [{name, scenario_dd, status}, ...]
    cb_status:    str | None = None,
) -> str:
    """
    Risk Console — Stress tab analyst note.
    Acharya-Pedersen 2005 conditional CAPM; Adrian-Brunnermeier 2016 CoVaR.
    """
    paras: list[str] = []

    paras.append("**当前状态**")
    if not scenarios:
        paras.append("- 暂无 stress scenario 计算结果。")
        paras.append("**下一步建议**\n- 等组合 NAV 时序累积后启动 stress engine。")
        return "\n\n".join(paras)

    worst = min(scenarios, key=lambda s: s.get("scenario_dd", 0) or 0)
    n_red = sum(1 for s in scenarios if s.get("status") == "red")
    n_yellow = sum(1 for s in scenarios if s.get("status") == "yellow")

    paras.append(_bullet([
        f"评估 {len(scenarios)} 个 stress scenario",
        f"最差 scenario: **{worst.get('name','—')}** → DD={worst.get('scenario_dd', 0)*100:+.1f}%",
        f"red 阈值触发: {n_red} 个；yellow: {n_yellow} 个",
        f"CB 状态: {cb_status or '—'}",
    ]))

    paras.append("**这说明什么**")
    so: list[str] = []
    if n_red > 0:
        so.append(
            f"{n_red} 个 scenario 触发 red threshold — 组合在该 stress 下亏损幅度超 RBAC "
            f"风险预算（Acharya-Pedersen 2005 conditional risk）"
        )
    elif n_yellow > 0:
        so.append(f"{n_yellow} 个 scenario 触发 yellow — 接近 stress 容忍阈但未越线")
    else:
        so.append("所有 scenario 在容忍区间内 — 当前 posture 对已建模冲击稳健")
    if worst.get("scenario_dd", 0) and worst["scenario_dd"] < -0.15:
        so.append(
            f"**最差 scenario DD < -15%** — Adrian-Brunnermeier 2016 CoVaR 类风险升高"
        )
    paras.append(_bullet(so))

    paras.append("**下一步建议**")
    nw: list[str] = []
    if n_red > 0:
        nw.append(
            f"针对 red 触发场景：识别 sub-portfolio 主要贡献者 → 减低对应 sector 暴露"
        )
        nw.append("考虑 hedge：买 put / 增配 risk-off 对冲资产")
    elif n_yellow > 0:
        nw.append("yellow 区间：监控；若多 scenario 同时进 yellow → 减仓警戒")
    else:
        nw.append("无紧急动作；周期性回看每日 stress 结果即可")
    if cb_status and cb_status.lower() in ("medium", "severe"):
        nw.append(f"CB 状态 {cb_status} — 严控新入场，配合自动减仓")
    paras.append(_bullet(nw))
    return "\n\n".join(paras)


def compose_correlation_narrative(
    n_pairs_high:    int,           # rolling 60d corr > 0.7 pair count
    avg_corr:        float | None,
    n_clusters:      int = 0,       # number of high-corr clusters
    regime:          str | None = None,
) -> str:
    """
    Risk Console — Correlations tab analyst note.
    Forbes-Rigobon 2002 contagion in stress; Engle-Sheppard 2001 DCC.
    """
    paras: list[str] = []
    paras.append("**当前状态**")
    avg_str = f"{avg_corr:.2f}" if avg_corr is not None else "—"
    paras.append(_bullet([
        f"高相关对（ρ > 0.7）: **{n_pairs_high}** 对",
        f"组合内平均相关: {avg_str}",
        f"高相关 cluster: {n_clusters} 个",
        f"当前 regime: {regime or '—'}",
    ]))

    paras.append("**这说明什么**")
    so: list[str] = []
    if n_pairs_high > 5:
        so.append(
            f"高相关对偏多（n={n_pairs_high}）— 同向相关敞口大，"
            "Forbes-Rigobon 2002 contagion 风险显著（stress 时 corr 进一步升高）"
        )
    elif n_pairs_high > 0:
        so.append(f"少量高相关对（n={n_pairs_high}）— 关注但未达警戒")
    else:
        so.append("无高相关对 — 组合已分散")
    if avg_corr is not None and avg_corr > 0.5:
        so.append(f"平均相关 {avg_corr:.2f} 偏高 — 分散收益低于理论值（Markowitz）")
    if regime and regime.lower() == "risk-off":
        so.append("Risk-off regime 内相关性历史上升 — Engle-Sheppard 2001 DCC 实证")
    paras.append(_bullet(so))

    paras.append("**下一步建议**")
    nw: list[str] = []
    if n_pairs_high > 5:
        nw.append("减弱 cluster 内的某腿；新入场避开已 cluster 的 ticker")
    if avg_corr is not None and avg_corr > 0.5:
        nw.append("下次 rebalance 优先加入低相关 / 负相关资产（如 long-short pair）")
    if not nw:
        nw.append("无紧急动作；Correlations 健康")
    paras.append(_bullet(nw))
    return "\n\n".join(paras)


def compose_limits_narrative(
    n_breaches:     int,          # red status count
    n_near:         int,          # yellow status count
    binding_specs:  list[str] | None = None,  # 触发 limit 的 dimension list
    tier_breakdown: dict | None = None,         # {tier1: n, tier2: n, ...}
) -> str:
    """
    Risk Console — Limits tab analyst note.
    Pedersen 2015 §4 / Bridgewater All-Weather risk parity / 4-tier 升级框架.
    """
    paras: list[str] = []
    binding_specs = binding_specs or []
    tier_breakdown = tier_breakdown or {}

    paras.append("**当前状态**")
    paras.append(_bullet([
        f"红色 (binding) limits: **{n_breaches}** 项",
        f"黄色 (near, ≥80%): **{n_near}** 项",
        (f"binding dimensions: {' · '.join(binding_specs[:5])}"
         if binding_specs else None),
    ]))

    paras.append("**这说明什么**")
    so: list[str] = []
    if n_breaches > 0:
        so.append(
            f"{n_breaches} 项 limit 已 binding — 必须减仓或暂停加仓；"
            "spec_power_analysis 红线维持"
        )
    elif n_near > 0:
        so.append(f"{n_near} 项 limit 接近 cap — 加仓空间快用完")
    else:
        so.append("所有 limit 在 cap 内 — 风险预算充足")
    if tier_breakdown.get("tier1", 0) > 0:
        so.append(f"**Tier 1** ({tier_breakdown['tier1']} 项) — 硬性 risk gate，必须立即处理")
    paras.append(_bullet(so))

    paras.append("**下一步建议**")
    nw: list[str] = []
    if n_breaches > 0:
        nw.append("**优先处理 Tier 1 binding**：减仓 / 平仓最高 |w| 仓位")
        nw.append("Tier 2 binding：下次 rebalance 强制平掉，期间不再加仓")
    elif n_near > 0:
        nw.append("near limit：加仓审批前确认未越 cap；超越则降仓")
    else:
        nw.append("无紧急动作；定期审视 limit 设置是否过宽 / 过严")
    paras.append(_bullet(nw))
    return "\n\n".join(paras)


def compose_risk_posture_narrative(
    exposure:      Any,
    concentration: list,
    pairs:         list,
    drawdown:      Any | None = None,
) -> str:
    """Decision C — are current holdings within risk limits?"""
    paras: list[str] = []

    if exposure is None:
        return (
            "**当前状态**\n暂无 SimulatedPosition 快照可评估。\n\n"
            "**下一步建议**\n等待下一次月度 rebalance 写入 SimulatedPosition 后此面板才有数据。"
        )

    # ── WHAT ────────────────────────────────────────────────────────────
    gross = getattr(exposure, "gross", 0.0)
    net = getattr(exposure, "net", 0.0)
    gross_cap = getattr(exposure, "gross_cap", 1.0)
    net_low = getattr(exposure, "net_cap_low", -1.0)
    net_high = getattr(exposure, "net_cap_high", 1.0)
    n_long = getattr(exposure, "n_long", 0)
    n_short = getattr(exposure, "n_short", 0)
    snapshot_date = getattr(exposure, "snapshot_date", None)
    stale = getattr(exposure, "stale", False)
    gross_status = getattr(exposure, "gross_status", "green")
    net_status = getattr(exposure, "net_status", "green")

    n_binding = sum(1 for r in (concentration or []) if getattr(r, "status", "") == "red")
    n_near = sum(1 for r in (concentration or []) if getattr(r, "status", "") == "yellow")
    n_pair_binding = sum(1 for p in (pairs or []) if getattr(p, "status", "") == "red")

    paras.append("**当前状态**")
    paras.append(_bullet([
        f"Gross **{gross:.1%}** / cap {gross_cap:.0%}；Net **{net:+.1%}** / 区间 [{net_low:+.0%}, {net_high:+.0%}]",
        f"持仓数：**{n_long}L / {n_short}S**（quick =  {n_long + n_short} 个）",
        f"Per-ticker 集中度：{n_binding} 个 binding (≥95% cap) / {n_near} 个 near (≥80%)",
        f"Pair 集中度：{n_pair_binding} 个相关对 binding (≥1.5×cap)",
        f"快照日 {snapshot_date}" + ("（**已过期 7d+**）" if stale else ""),
    ]))

    # Drawdown if present
    if drawdown and getattr(drawdown, "available", True):
        cur_dd = getattr(drawdown, "current_drawdown_pct", 0)
        u_days = getattr(drawdown, "underwater_days", 0)
        paras.append(_bullet([
            f"当前回撤 {cur_dd*100:+.2f}% · underwater {u_days} 天",
        ]))

    # ── SO WHAT ─────────────────────────────────────────────────────────
    paras.append("**这说明什么**")
    so_bullets: list[str] = []

    if gross_status == "red":
        so_bullets.append(f"Gross **{gross:.1%}** 超 cap {gross_cap:.0%} — 杠杆超标，必须减仓")
    elif gross_status == "yellow":
        so_bullets.append(f"Gross 接近 cap，加仓空间有限")
    else:
        so_bullets.append(f"Gross 在 cap 内，杠杆健康")

    if net_status == "red":
        if net > net_high:
            so_bullets.append(f"Net **{net:+.1%}** 超过上限 {net_high:+.0%} — 单边过多头")
        elif net < net_low:
            so_bullets.append(f"Net **{net:+.1%}** 低于下限 {net_low:+.0%} — 单边过空头")

    if n_binding > 0:
        so_bullets.append(
            f"{n_binding} 个 ticker binding（个仓 ≥95% cap）— Markowitz 集中度风险偏高"
        )
    if n_pair_binding > 0:
        so_bullets.append(
            f"{n_pair_binding} 个相关对 binding — 同方向相关敞口超 1.5×cap，"
            "Forbes-Rigobon 2002 contagion 风险升高"
        )
    if stale:
        so_bullets.append(
            "快照过期 ≥7 天 — 当前数字可能不反映 live state；月度 rebalance 应已 fire"
        )
    if drawdown and getattr(drawdown, "current_drawdown_pct", 0) <= -0.10:
        so_bullets.append(
            f"回撤 ≥10%（Pedersen 2015 §10 attention threshold）— 加仓决策应保守"
        )
    paras.append(_bullet(so_bullets))

    # ── NOW WHAT ────────────────────────────────────────────────────────
    paras.append("**下一步建议**")
    nw_bullets: list[str] = []

    if gross_status == "red" or net_status == "red":
        nw_bullets.append(
            "**立即减仓** — 进 Positions 页选 binding ticker → 主动平仓 / 减仓审批"
        )
    elif n_binding > 0:
        nw_bullets.append(
            f"对 {n_binding} 个 binding ticker：下次 rebalance **不要加仓**；"
            "若信号强烈需加，先批前置减仓 sub-cap"
        )

    if n_pair_binding > 0:
        nw_bullets.append(
            "对 binding 相关对：审批新入场时优先减弱原对手腿（避免拼命 long-only ETF 同向相关）"
        )

    if stale:
        nw_bullets.append(
            "进 Engineering → Force monthly rebalance（紧急 manual override）刷新快照"
        )

    if drawdown and getattr(drawdown, "current_drawdown_pct", 0) <= -0.10:
        nw_bullets.append(
            "Drawdown ≥10% 期间：日内审批保守批；优先批减仓 / 风控类，慎批新入场"
        )

    if not nw_bullets:
        nw_bullets.append("无紧急动作；维持当前 risk posture")

    paras.append(_bullet(nw_bullets))

    return "\n\n".join(paras)
