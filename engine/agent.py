from typing import TypedDict

from langgraph.graph import END, StateGraph

from engine.news import NewsPerceiver
from engine.quant import AnalyticsEngine, compute_quant_metrics
import time
from engine.key_pool import get_pool, AllKeysExhausted, EmptyOutputCircuitBreaker, BillingProtectionError, QUOTA_FAILS_BEFORE_SWITCH, RETRY_BASE_DELAY
from engine.memory import get_historical_context, SessionFactory
from engine.memory import SimulatedPosition


def build_position_context(sector_name: str) -> str:
    """Return a brief string describing the current simulated position for sector_name.
    Returns "" if no position exists (first-time analysis).
    """
    try:
        with SessionFactory() as session:
            pos = (
                session.query(SimulatedPosition)
                .filter_by(sector=sector_name, track="main")
                .order_by(SimulatedPosition.snapshot_date.desc())
                .first()
            )
            if pos is None:
                return ""
            parts = [
                f"当前持仓: {pos.direction or '未知'} | 权重: {pos.actual_weight:.1%}" if pos.actual_weight else "",
                f"入场日期: {pos.snapshot_date}",
                f"制度标签: {pos.regime_label}" if pos.regime_label else "",
            ]
            return "[持仓状态] " + " | ".join(p for p in parts if p)
    except Exception:
        return ""


class AgentState(TypedDict):
    # 输入
    target_assets: str
    vix_level: float
    macro_context: str
    sector_risks: str
    sector_rankings: list      # 全板块扫描排行榜，供反思节点使用
    news_context: str          # 近48小时相关新闻，由 NewsPerceiver 注入
    macro_regime: str          # 结构化制度标签，如 "宽松"/"收紧"/"高波动收缩"（供 Alpha Memory 查询）
    position_context: str      # P1-2: current simulated position for this sector
    # 中间态
    quant_results: dict
    quant_context_raw: str   # P0-4: raw numeric QuantAssessment context (no directional conclusions)
    red_team_critique: str
    technical_report: str
    # 输出
    is_robust: bool
    audit_memo: str              # translator 节点产出的决策备忘录
    alternative_suggestion: str  # 反思节点产出的替代建议
    reflection_chain: str        # 反思节点完整推理链（输入上下文+输出）


def build_agent_graph(model, preset_assets: dict, av_key: str = "", gnews_key: str = ""):
    """
    工厂函数：将 model 和 preset_assets 通过闭包注入各节点。
    图结构：
      researcher → red_team → (通过) auditor → translator → END
                            ↘ (拦截) reflection → END

    所有 AI 调用通过 _pool_call() 走 Gemini 号池，自动轮转并在额度耗尽时返回提示。
    """

    # ── Gemini 号池调用封装 ──────────────────────────────────────────────────
    def _pool_call(prompt: str) -> str:
        """
        通过 Gemini 号池发起 AI 调用，自动重试并轮转 Key。

        重试策略：
        - quota/429 错误 → report_quota_error()（达阈值自动轮转）→ 继续循环重试
        - 所有 Key 耗尽 → 返回错误字符串（唯一终止条件）
        - 空输出熔断    → 返回错误字符串
        - 非 quota 错误 → 立即返回错误字符串（不重试）

        Key 轮转对调用方完全透明，分析质量不受影响。
        """
        _pool = get_pool()
        # 最多尝试次数 = 每个 key 允许的 quota 失败次数 × key 总数 + 1
        _max = len(_pool._keys) * QUOTA_FAILS_BEFORE_SWITCH + 1

        for _attempt in range(_max):
            try:
                _pool.check_billing_limits()   # hard RPM/RPD gate — raises BillingProtectionError
                _m     = _pool.get_model()
                result = _m.generate_content(prompt).text
                _pool.report_success(has_content=bool(result.strip()))
                return result
            except BillingProtectionError as e:
                return f"🛡️ 计费保护封锁\n\n{e}"
            except AllKeysExhausted as e:
                return (
                    f"⛔ 所有 Gemini API Key 已耗尽每日配额\n\n"
                    f"原因：{e}\n\n"
                    "处理方式：\n"
                    "• 明天配额自动重置后可继续使用\n"
                    "• 前往 Key Pool Manager 页面添加更多 Key\n"
                    "• 量化指标数据不受影响，可直接参考"
                )
            except EmptyOutputCircuitBreaker as e:
                return (
                    f"⚠️ AI 输出异常熔断\n\n"
                    f"原因：连续多次返回空内容，已暂停调用\n"
                    f"详情：{e}\n\n"
                    "请检查 Key Pool Manager 中的异常日志"
                )
            except Exception as e:
                if _pool.is_quota_error(e):
                    try:
                        _pool.report_quota_error()  # 达阈值时轮转，可能抛 AllKeysExhausted
                    except AllKeysExhausted as ex:
                        return (
                            f"⛔ 所有 Gemini API Key 已耗尽每日配额\n\n"
                            f"详情：{ex}\n\n"
                            "前往 Key Pool Manager 页面查看状态或添加新 Key"
                        )
                    # 轮转前先等待：RPM 限制按分钟计，backoff 让同一个 key 有机会恢复
                    # consecutive_quota 在 report_quota_error 后已递增
                    _wait = _pool._get_stats(_pool.current_label)["consecutive_quota"] * RETRY_BASE_DELAY
                    if _wait > 0:
                        time.sleep(_wait)
                    continue
                # 非 quota 错误（网络、格式等），不重试
                return f"⚠️ AI 分析引擎暂时离线（错误：{str(e)[:120]}）"

        # 理论上不会到达（AllKeysExhausted 会在循环内提前返回）
        return "⛔ 超过最大重试次数，请检查 Key Pool Manager"

    def research_node(state: AgentState) -> dict:
        tickers = tuple(preset_assets.get(state["target_assets"], []))
        q = compute_quant_metrics(tickers, state["vix_level"])
        if not q:
            return {"is_robust": False, "quant_results": {}}

        p_noise = q["p_noise"]
        active  = q["active"]
        penalty_p = max(0, (p_noise - 0.05) * 200)
        penalty_s = 20 if active < 2 else 0
        confidence_score = max(0, min(100, 100 - penalty_p - penalty_s))

        return {
            "quant_results": {
                **q,                        # coefs and X already included
                "confidence_score": confidence_score,
            },
            "is_robust": p_noise < 0.3,
        }

    def red_team_node(state: AgentState) -> dict:
        q = state.get("quant_results") or {}
        if not q:
            return {"is_robust": False, "red_team_critique": "🔴 错误：未检测到量化研究数据。"}

        vix = state.get("vix_level", 20.0)
        p_noise = q.get("p_noise", 0)
        active_features = q.get("active", 0)

        math_critiques = []
        _mname_rt = q.get("model_name", "Lasso")
        if p_noise > 0.25:
            math_critiques.append(f"🔴 统计噪音过高 ({p_noise:.1%})，结果可能具有随机性。")
        if active_features < 2:
            # Should not happen after Ridge fallback; guard for unexpected edge cases
            math_critiques.append("🔴 有效特征数极少（<2），模型退化；建议检查输入数据质量。")
        elif "fallback" in _mname_rt:
            math_critiques.append(
                f"🟡 L1 正则化导致特征过度稀疏，已自动降级为 Ridge（L2）重拟合。"
                f"当前有效特征数 {active_features} 个，结果稳定性已改善。"
            )

        news_ctx = state.get("news_context", "暂无新闻数据")
        asset_name = state.get('target_assets', '未知')

        # Format momentum context for red team
        def _fmt_mom(v):
            return f"{v:+.1%}" if v is not None else "数据不足"
        mom_line = (
            f"动量信号 — 1M: {_fmt_mom(q.get('mom_1m'))} | "
            f"3M: {_fmt_mom(q.get('mom_3m'))} | "
            f"6M: {_fmt_mom(q.get('mom_6m'))}"
        )

        # ── Alpha Memory: historical performance for this sector × regime ────────
        # Returns "" when DB is empty — zero prompt impact until Phase-0 training runs.
        alpha_ctx = get_historical_context(
            tab_type="",                                   # cross-tab: all verified decisions
            sector_name=asset_name,
            macro_regime=state.get("macro_regime", ""),
        )
        alpha_block = f"\n{alpha_ctx}\n" if alpha_ctx else ""

        # ── Evidence stripping: blacklist quant system's primary support arguments ─
        # Force red team to find independent evidence outside already-known signals.
        blue_drivers = []
        if q.get("mom_3m") is not None:
            _dir = "正" if q["mom_3m"] > 0 else "负"
            blue_drivers.append(f"3M动量为{_dir}（{q['mom_3m']:+.1%}）")
        if q.get("a_ret", 0) > 0:
            blue_drivers.append(f"年化收益率为正（{q['a_ret']:+.1%}）")
        if q.get("p_noise", 1) < 0.3:
            blue_drivers.append(f"P-hacking风险低（{q['p_noise']:.1%}）")
        if q.get("a_vol", 0) < 0.2:
            blue_drivers.append(f"年化波动率在控（{q['a_vol']:.1%}）")
        blacklist_block = ""
        if blue_drivers:
            blacklist_block = (
                f"\n【证据链剥离指令】量化系统已确认的主要利好论据：{'；'.join(blue_drivers)}。"
                f"你的任务是从以上已知证据之外寻找潜在风险——"
                f"禁止将上述指标作为放行依据，必须提出量化系统尚未捕捉的独立反驳视角。\n"
            )

        quant_raw_block = state.get("quant_context_raw", "")
        quant_raw_section = (
            f"\n【量化原始数据（仅供参考，非方向性结论）】\n{quant_raw_block}\n"
            if quant_raw_block else ""
        )

        pos_ctx = state.get("position_context", "")
        pos_block = f"\n{pos_ctx}\n" if pos_ctx else ""

        audit_prompt = f"""
        你是一名资深量化风险审计师，代表机构风险委员会对以下资产包进行独立审计。
        {alpha_block}
        {pos_block}
        审计对象: {asset_name} | 当前 VIX: {vix} | 量化 VaR: {q.get('d_var', 0):.2%}
        年化收益: {q.get('a_ret', 0):.2%} | 年化波动率: {q.get('a_vol', 0):.2%}
        {mom_line}
        {quant_raw_section}
        宏观背景: {state.get('macro_context', '暂无')}
        行业风险: {state.get('sector_risks', '暂无')}
        {blacklist_block}
        {news_ctx}

        请严格按以下四个章节输出，每章节以 ### 章节名 作为唯一标题行，内容充实有据：

        ### 1. 量化信号与波动率环境匹配度
        VIX {vix} 处于历史什么分位区间（高/中/低波动）？VaR {q.get('d_var', 0):.2%} 在此环境下
        是否低估了尾部风险？对比历史相似波动率区间该类资产的实际回撤幅度，评估模型乐观程度。
        至少 3 句，引用具体数值。

        ### 2. 新闻情报深度审计
        逐一检查上方新闻：财报预警、监管收紧、地缘升级、流动性紧张等利空信号是否存在？
        每条关键新闻必须引用原标题，说明其对量化假设的冲击方向与力度，判断是否构成拦截依据。
        至少 3 句。

        ### 3. 宏观与行业双重压力传导
        结合宏观背景与行业风险，具体描述不利传导路径。
        点出模型最可能失效的 2 个情景（如利率超预期、板块轮动逆转、流动性冻结等），
        评估每个情景的触发概率与潜在冲击幅度。至少 3 句。

        ### 4. 综合审计结论
        基于以上三个维度给出明确最终结论：
        若存在严重矛盾，写"🚨【拦截】"并用 2-3 句说明核心拦截理由与建议观望条件。
        若无重大矛盾，写"✅【放行】"并用 2-3 句说明支撑逻辑与需持续监控的风险点。

        格式要求：章节标题只用 ### 前缀，正文禁止使用 ** 加粗符号，
        禁止在正文中嵌套 ### 标题，语气客观专业，立场中立。
        """
        ai_critique = _pool_call(audit_prompt)

        val_r2     = q.get("val_r2")
        test_r2    = q.get("test_r2")
        _mname     = q.get("model_name", "Lasso")   # Ridge / ElasticNet / Lasso
        if val_r2 is not None:
            r2_line = (
                f"📊 {_mname} 样本内 R²: {val_r2:.4f} | 样本外 R²: {test_r2:.4f} "
                f"（过拟合幅度: {val_r2 - test_r2:+.4f}）"
            )
            if test_r2 < 0:
                math_critiques.append(
                    f"🟡 样本外 R² 为负（{test_r2:.4f}），{_mname} 泛化能力弱于无条件均值，信号微弱。"
                )
        else:
            r2_line = f"⚪ 样本量不足（n<30），使用 {_mname}，R² 未计算（无法时间分割）。"

        if math_critiques:
            math_body = r2_line + "\n" + "\n".join(math_critiques)
        else:
            math_body = (
                f"{r2_line}\n"
                f"🟢 P-hacking 风险 {p_noise:.1%}，低于 30% 阈值，统计结论具备基础可信度。\n"
                f"🟢 {_mname} 有效特征数 {active_features} 个，模型未出现严重欠拟合或过度稀疏。\n"
                f"数学层面暂未触发硬性量化拦截条件。"
            )
        # Merge math audit as the first ### section, then AI's four sections follow
        full_critique = (
            f"### {asset_name} · 量化风险审计报告\n\n"
            f"### 统计硬性指标审计\n{math_body}\n\n"
            f"{ai_critique}"
        )

        veto_keywords = ["🚨", "拦截", "严重错配", "滞后性", "误导", "自欺欺人", "根本性矛盾"]
        # active_features < 2 only fires if Ridge fallback itself failed (edge case);
        # normal Ridge fallback restores active_features to full feature count.
        _ridge_fallback = "fallback" in q.get("model_name", "")
        is_robust = not (
            p_noise > 0.30
            or (active_features < 2 and not _ridge_fallback)
            or any(w in ai_critique for w in veto_keywords)
        )
        if not is_robust and "🚨【拦截】" not in full_critique:
            full_critique = "🚩 [AI 逻辑强制否决]\n" + full_critique

        return {
            "red_team_critique": full_critique,
            "is_robust": is_robust,
        }

    def reflection_node(state: AgentState) -> dict:
        """
        反思节点：当红队拦截后启动。
        从全板块排行榜中找到下一个候选，由 AI 判断能否规避拦截理由。
        """
        blocked = state.get("target_assets", "未知")
        critique = state.get("red_team_critique", "")
        rankings = state.get("sector_rankings", [])

        # 跳过被拦截的资产，取排名最高的替代
        alternatives = [r for r in rankings if r.get("name") != blocked]
        if not alternatives:
            return {"alternative_suggestion": "⚠️ 排行榜为空，无可用替代资产。请先运行扫描节点以获取数据。"}

        next_best = alternatives[0]

        # Fetch news for the alternative candidate to ground the reflection
        try:
            alt_ticker = next_best.get("ticker", next_best["name"])
            news_for_alt = NewsPerceiver(av_key=av_key, gnews_key=gnews_key).build_context(
                next_best["name"], alt_ticker, n=4
            )
        except Exception:
            news_for_alt = "暂无新闻数据"

        prompt = f"""
        你是一名量化策略反思师。当前首选资产【{blocked}】已被红队拦截，理由如下：
        {critique}

        基于全市场 {len(rankings)} 个板块扫描，下一候选资产为【{next_best['name']}】（排名 #{next_best.get('rank', '?')}）：
        - 夏普比率: {next_best['sharpe']:.2f}
        - 区间动能: {next_best['momentum']:.1%}
        - 市场契合度: {next_best['market_fit']}%
        - 资金活跃度: {next_best['fund_flow']:.2f}x

        {news_for_alt}

        任务（严格按顺序回答）：
        1. 【规避性分析】：【{next_best['name']}】是否能规避上述拦截理由？请逐点对照。
        2. 【替代逻辑】：若能规避，给出 2-3 句投资逻辑；若不能，说明原因并建议观望。
        3. 【风险提示】：该替代资产当前最主要的 1 个下行风险。

        格式要求：以"🔄 反思建议："开头，总字数控制在 200 字以内，专业简洁。
        """
        suggestion = _pool_call(prompt)

        alt_summary = (
            f"原始拦截目标：{blocked}\n"
            f"替代候选：{next_best['name']} "
            f"（夏普 {next_best['sharpe']:.2f} | 动能 {next_best['momentum']:.1%}）\n\n"
            f"{suggestion}"
        )
        # Full reasoning chain: context fed to reflection + output, for audit trail
        reflection_chain = (
            f"=== 反思节点输入上下文 ===\n"
            f"被拦截目标：{blocked}\n"
            f"替代候选：{next_best['name']} | 排名 #{next_best.get('rank','?')} | "
            f"夏普 {next_best['sharpe']:.2f} | 动能 {next_best['momentum']:.1%} | "
            f"契合度 {next_best['market_fit']}% | 资金活跃度 {next_best['fund_flow']:.2f}x\n"
            f"全市场候选总数：{len(rankings)}\n\n"
            f"=== 拦截理由摘要 ===\n{critique[:600]}\n\n"
            f"=== 反思节点输出 ===\n{suggestion}"
        )
        return {
            "alternative_suggestion": alt_summary,
            "reflection_chain":       reflection_chain,
        }

    def technical_audit_node(state: AgentState) -> dict:
        q = state["quant_results"]

        # Format val/test R² only when a real temporal split was feasible
        val_r2  = q.get("val_r2")
        test_r2 = q.get("test_r2")
        _mname  = q.get("model_name", "Lasso")
        if val_r2 is not None:
            r2_block = (
                f"样本内 R²（训练集）: {val_r2:.4f}\n"
                f"样本外 R²（时间分割留出集）: {test_r2:.4f}\n"
                f"过拟合幅度（样本内-样本外）: {val_r2 - test_r2:+.4f}\n"
                f"{_mname} 正则化搜索次数（alpha 候选数）: {q.get('n_alphas', 'N/A')}"
            )
        else:
            r2_block = f"样本量不足（n<30）：使用 {_mname}（无 CV/时间分割），R² 未计算。"

        prompt = f"""
        你是一名量化风险审计师。请针对以下指标撰写一份【硬核技术报告】：
        资产包: {state['target_assets']}
        VIX环境: {state['vix_level']}
        VaR: {q['d_var']:.2%}
        正则化模型: {_mname}（n<30 用 Ridge，30-100 用 ElasticNet，≥100 用 Lasso）
        有效特征数: {q['active']}
        过拟合评估（时间分割法）:
        {r2_block}
        P-hacking 综合风险估计: {q['p_noise']:.2%}

        注：样本外 R² 为负时，表示模型泛化能力弱于无条件均值预测器，
        这在金融数据中属于常见结果，应解读为"信号微弱而非模型崩溃"。

        要求：使用专业量化术语，分析模型的统计稳健性、过拟合风险及非对称风险暴露。
        输出格式：纯文字，禁止使用 ** 加粗符号，禁止使用 # 标题符号。
        """
        return {"technical_report": _pool_call(prompt)}

    def translator_node(state: AgentState) -> dict:
        tech_report = state.get("technical_report", "")
        prompt = f"""
        你是一名资深投研顾问，负责向基金经理（Fund Manager）汇报。
        请根据下方的【硬核技术报告】，将其重写为【CEO 级分层简报】。

        原始报告内容：{tech_report}

        请严格按以下三个章节输出，每章节以 ### 章节名 作为唯一标题行，内容要充实详尽：

        ### 首席执行决策建议
        用直白、果断的决策建议起头（买入 / 持有 / 减仓）。严禁统计术语，
        向基金经理说清楚"So What"——这个资产包当前是否值得配置，核心理由是什么，
        需要关注哪个最主要的风险信号。至少 3-4 句，结论明确不模糊。

        ### 关键洞察
        将复杂指标逐一翻译为业务语言：
        VaR 对应压力场景下的最大潜在亏损；Lasso 稀疏度对应决策信号的纯净度；
        P-hacking 风险对应分析结果的真实可信度。
        结合当前宏观环境，说明这些指标在实战配置中意味着什么。至少 4-5 句。

        ### 技术附录
        完整保留原始量化术语和所有具体数值，供量化同事复核。
        涵盖统计检验结论、风险分解、模型稳健性评估及非对称风险暴露分析。至少 3-4 句。

        格式要求：章节标题只用 ### 前缀，正文中禁止使用 ** 加粗符号，
        禁止在正文中嵌套 ### 标题，语气客观专业。
        """
        return {"audit_memo": _pool_call(prompt)}

    # --- 构建图 ---
    builder = StateGraph(AgentState)
    builder.add_node("researcher", research_node)
    builder.add_node("red_team", red_team_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("auditor", technical_audit_node)
    builder.add_node("translator", translator_node)

    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "red_team")
    builder.add_conditional_edges(
        "red_team",
        lambda x: "auditor" if x["is_robust"] else "reflection",
    )
    builder.add_edge("reflection", END)
    builder.add_edge("auditor", "translator")
    builder.add_edge("translator", END)

    return builder.compile()
