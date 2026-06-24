# PM Doctrine v0.5 DRAFT — Macro Alpha Pro

> **状态: v0.5 DRAFT,等 PM 逐条改后升 v1.0。**
> Claude Code 起草 2026-05-19。可推导的节按 spec 事实写实,标 `[事实·核对]`;
> 主观的节是基于项目已知信息的合理猜测,标 `[猜测·改掉不对的]` —— 每一行当成问题:你同意吗?不对就改。
> **所有权是你的,每个字你拍板。** 改完把 `v0.5 DRAFT` 改成 `v1.0`,告诉我,我接进 7 个 agent 的 system prompt。
> NO EMOJIS。

---

## 1. 我的 edge 是什么 / 不是什么  `[事实·核对]`

我的 alpha 来自 5 个有学术机制支撑的 sleeve,**每一个都是结构性 / 行为性的 risk premium 或 mispricing,不是预测**:

- **K1 BAB** (spec 61, etf_l1): Frazzini-Pedersen 2014 杠杆约束机制。杠杆受限的投资者为达到收益目标系统性超配高 β 资产,导致低 β 资产被低估。**edge 在杠杆约束最紧时最强**;若散户保证金便利化、杠杆约束放松,应预期衰减——这是 invalidation 信号之一。43 ETF 横截面 BAB,30 天再平衡。
- **D-PEAD** (spec 62, ss_sp500): DHS 2020 行为双因子,盈余公告后漂移 (PEAD)。价格对盈余 surprise 反应不足,漂移持续 1-3 月。**事件驱动统计套利,不是选股**。top-1500 point-in-time,60 天。
- **Path N** (spec 71, ss_sp500): Chen-Noronha-Singal 2004 指数重构漂移。S&P 500 增删前的可预测被动资金流。单名事件驱动,5 天 horizon。
- **CTA PQTIX** (spec 72, cta_defensive): 尾部对冲 / 危机正收益防御 sleeve。持续 10% 配置,**设计上平时拖累收益,危机时正贡献**。
- **AC TLT/GLD** (spec 73, rms_crisis_hedge): Asness-Israelov 2017 flight-to-quality + 黄金避险。TLT/GLD 50/50,危机保险。

我**不相信**自己有 edge 的地方 (避免越界):
- 日内 / 短期择时 — 没有
- 单股选股 — 没有 (D-PEAD 是统计漂移不是选股)
- 宏观方向预测 (利率/汇率/指数点位) — 没有
- 任何让我做以上的诱惑 = 过度自信,不是 alpha

---

## 2. 我什么时候加减仓 / 完全不动  `[猜测·改掉不对的]`

默认状态: **跟 signal 走,按各 sleeve 的 rebalance 周期 (5/30/60 天) 自动再平衡,不做日内干预。**

我手动干预的唯一合法理由:
- (a) RM HARD HALT
- (b) DQ HARD HALT
- (c) 我**事先写下**的 invalidation 条件触发 (见 §1 各 sleeve)

除此之外,看到回撤就想动手 = 我在交易自己的情绪,不是交易 signal。

> ← 这条你要确认:你真打算这么纪律化吗?还是你预期自己会想做些主观 overlay?诚实写,别写理想中的自己。

---

## 3. 我的 drawdown 红线  `[猜测·改掉不对的 —— 这里数字必须你定]`

纸面历史 MaxDD = -10.92%。我的**事先**规则 (不是回撤当下临时决定的):

| 触发 | 动作 |
|---|---|
| -8% | 开始每日复盘归因 — 是 edge 坏了还是正常波动? |
| -12% | 杠杆从 Path B 的 1.5x 降到 1.0x |
| -15% | 全部降到 insurance sleeve,观察一个月 |

> ← 这三个数字 (-8 / -12 / -15) 是我的猜测,锚在你纸面 -10.9% 上。你真实的心理红线是多少?
> 真钱时你能扛的痛感和纸面完全不同——很多人纸面 -15% 无感,真钱 -8% 就失眠。诚实填。
> 注意:这是关于你自己行为的**假设**,要等真回撤来检验。

---

## 4. 每个 sleeve 我的心理容忍区间  `[一半事实·一半猜测]`

当前配置: ss_sp500 48.6% / etf_l1 32.4% / rms_crisis_hedge 10% / cta_defensive 9%

- **insurance (AC TLT/GLD) `[事实]`**: 接受它长期零 Sharpe 甚至小亏。它的工作是 2008 (+18pp) / 2020 (+12pp) / 2018 (+7pp) 那种危机里正贡献,平时拖后腿是设计的一部分,不是 bug。**任何"因为 insurance 拖累收益就砍掉它"的念头都是危机前自毁保险。**
- **CTA overlay `[猜测]`**: 容忍 ±5pp 配置漂移。← 改成你的数
- **equity factor (K1 BAB) `[猜测]`**: 最警惕 crowding。BAB 是公开因子,拥挤时 edge 衰减最快。← 你怎么监控 crowding?写下来
- **single-stock (D-PEAD + Path N) `[猜测]`**: 1500 名分散,单名上限已由 RM Mode 1b 卡死 5%。我容忍它的高换手 (60d/5d)。← 确认

---

## 5. 什么情况我宁可空仓也不交易  `[一半事实·一半猜测]`

- (a) `[事实]` DQ 报告核心数据源 (FRED / bab_compat cache / PEAD panel) stale 超阈值 — 宁可不交易也不在烂数据上下注
- (b) `[猜测]` 我连续 3 天看不懂归因 (NAV 在动但 Attribution Analyst 拆不清来源) — 模型和现实脱节,停下查
- (c) `[猜测]` 出现我 spec 里完全没考虑过的 regime (如 2020-03 流动性枯竭)

> ← (b)(c) 是我的猜测。你还有别的"宁可空仓"的线吗?比如某个宏观事件、某个波动率水平?

---

## 6. 我对 agent 的信任边界  `[一半事实·一半猜测]`

**一定亲自复核 (不能只信 agent narrative):**
- (a) `[事实]` 任何 RM HARD HALT — 看原始 breach 数据,不只信 narrative
- (b) `[事实]` 任何 Anomaly Sentinel confidence >= 4 的 ticker — 亲自跑 forensic check
- (c) `[事实]` Devil's Advocate 的批判永远当"值得查的线索"不当"结论"
- (d) `[事实]` Attribution Analyst 说"我没有因子回归工具"时信它的诚实,绝不逼它编因子 beta

**不需要复核 (agent 查得比我准):**
- 常规 INFO 级 alert、spec 元数据查询、历史召回 (recall_past_turns)

> ← 这节我基本按 operating model 的设计写实。你有想加的信任/不信任边界吗?

---

## 7. 我现阶段的真实约束 (单人 MSBA,不是 BlackRock)  `[事实·核对]`

让 agent 知道我和机构 PM 的差异,别套不适用的 best practice:

- 没有 24/7 交易台 — 所有决策在我醒着时做
- 没有合规部 — Audit Recorder 就是我的合规
- capacity 上限 ~$5B (实测 ADV: K1 $1.8B / D-PEAD $2.3B / Path N $385M / AC $4.8B),不是无限
- 没有 prime broker 谈杠杆成本 — 按 retail 算
- **时间是最稀缺资源** — agent 帮我省时间的价值 > 帮我多赚 1bp。所以 advisory 要 terse,别给我机构那种 30 页 deck
- 当前是**纸面验证阶段**,不是真钱

---

## 8. 我的成功标准 (2028-05 真钱 gate 时)  `[事实·核对]`

2028-05 (24mo OOS) 时我要能诚实回答:
- (a) 24 个月 OOS 的 Sharpe 是否仍 > 0.4 (in-sample 0.54 合理衰减后)?
- (b) MaxDD 是否守住 < -15%?
- (c) 这套 operating model 是否让我一个人能稳定运营不崩溃?

**如果 (a)(b) 任一 fail,我接受不上真钱。** 纸面验证的全部意义就是允许我诚实地说 no——这是 doctrine 里最重要的一条。

---

## 元数据

- **版本**: v0.5 DRAFT (Claude Code 起草) → 待 PM 逐条改 → v1.0
- **最后更新**: 2026-05-19
- **复审周期**: 每季度 + 任何 drawdown 红线触发后
- **接入状态**: ☐ 待 PM 改定后接进 agent system prompt
- **关联**: docs/operating_model_v1.md §三 RACI "Accountable" 格
