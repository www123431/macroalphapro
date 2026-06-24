# Spec — Narrative Risk Gate D1.1 (Power-Aware Re-evaluation)

**版本**：v0.1（2026-05-02 ~23:50 起草）
**状态**：⏳ 等待用户审阅 §9 checklist；spec 通过后开工
**前置**：[narrative_risk_gate_d1_soft_rejected.md](decisions/narrative_risk_gate_d1_soft_rejected.md)
**相关 spec**：[spec_narrative_risk_gate_d1.md](spec_narrative_risk_gate_d1.md)（D1 原 spec，已 soft reject）
**Memory**：[project_narrative_overlay_decision.md](../memory/project_narrative_overlay_decision.md) +
[feedback_spec_power_analysis.md](../memory/feedback_spec_power_analysis.md)

---

## 0. TL;DR

D1 在 88 月 OOS + NW t ≥ 1.5 闸门下 soft reject（4/5 PASS, NW t=0.896 卡死）。
**问题诊断**：原 spec NW t ≥ 1.5 在 ΔSharpe=0.10 量级 marginal alpha 上 power=5% — spec 写作时未做事前
power analysis，是 confirmatory threshold 用在 marginal alpha 上的设计错误。

**D1.1 修正**（用户 2026-05-02 拍板）：
- OOS 期扩到 **192 月（2010-05-02 至 2026-05-02）**—— 增加 fresh data 解决 sample size
- NW t threshold 改为 **1.0**（exploratory standard）—— 基于事前 power analysis
- 其他 4 闸门**完全不变**（ΔSharpe ≥ 0.10 / B-C ≥ 0.05 / Subperiod ≥ -0.20 / PBO ≤ 50%）
- 明确标注 **exploratory test，不是 confirmatory**
- D1.1 PASS 后仍需 F (paper trading 6-12 月) 做 confirmatory
- D1.1 REJECT 整个 narrative 方向终止

---

## 1. 决策来源（用户授权）

用户 2026-05-02 ~23:50 明确接受：
1. ✅ NW t threshold = **1.0**
2. ✅ 接受 FPR = 16% one-sided（vs 原 7%）
3. ✅ 接受 PASS 是 exploratory 不是 confirmed
4. ✅ 其他 4 闸门保持
5. ✅ OOS 期 192 月 (2010-05-02 至 2026-05-02)
6. ✅ 不允许结果出来后调 threshold

**spec 锁定原则**：本 spec 通过后，任何 reject 决定不允许通过 retroactive 修改 spec 翻盘。

---

## 2. 严格事前 Power Analysis

按 [feedback_spec_power_analysis.md](../memory/feedback_spec_power_analysis.md) 强制要求。

### 2.1 Inputs（全部纯文献先验，独立于 D1 实际数字）

| 参数 | 值 | 来源 |
|---|---|---|
| n | 192 月 | spec § 3.2 |
| Expected ΔSharpe annual under H1 | **0.10** | Bybee-Kelly-Manela-Xiu (2023, RFS) 严格 OOS narrative alpha mid-low estimate |
| HAC SE inflation factor | **1.3** | Financial monthly data typical (mild positive autocorr) |
| α one-sided | **0.16** | Exploratory test, derived from threshold 1.0 |
| Target power | ≥ **20%** at ΔSharpe=0.10 | Exploratory standard, acknowledges fundamental sample-size limit |

### 2.2 Calculation

```
IID t under H1   = ΔSharpe × √(n/12) = 0.10 × √(192/12) = 0.10 × 4.00 = 0.40
NW t under H1    = IID t / HAC_inflation = 0.40 / 1.3       = 0.31
threshold = 1.0  → FPR = 1 - Φ(1.0) = 16% (one-sided)
                 → Power at H1 = 1 - Φ(1.0 - 0.31) = 1 - Φ(0.69) = 24%
```

### 2.3 Power 比较表

| Threshold | FPR | Power(ΔSharpe=0.10) | Power(ΔSharpe=0.15) | Power(ΔSharpe=0.20) |
|---|---|---|---|---|
| 1.5（D1 原）| 7% | 9% | 17% | 30% |
| 1.28（α=10%）| 10% | 17% | 26% | 40% |
| **1.0（D1.1 选定）** | **16%** | **24%** | **35%** | **48%** |

**残余风险**：即使 192 月 + threshold 1.0，对 ΔSharpe=0.10 量级 alpha 的 detection power 仍只 24%。
**这是 sample-size physics**，不是 spec 设计问题。无法在 88-200 月样本上同时满足 confirmatory 严格度
+ marginal alpha detection。D1.1 接受这个 trade-off，**用 exploratory standard + 后续 F 补强**。

### 2.4 Test 性质显式标注

- **Confirmatory test**（生产部署证据）：FPR ≤ 5% + Power ≥ 50%。**D1.1 不满足**。
- **Exploratory test**（pivot 验证 + 决定是否进 F）：FPR ≤ 20% + Power ≥ 20%。**D1.1 满足**。

---

## 3. Spec 参数（事前定，锁定）

### 3.1 OOS 期与数据

| 项 | 值 |
|---|---|
| OOS 起始 | 2010-05-02 |
| OOS 结束 | 2026-05-02（约 2026-04-30 月末） |
| 月数 | 192 月（实际可能 191-192 视 yfinance 边界） |
| Shocks 数据源 | `data/shocks.parquet`（已 cache，**真实**数据 synthetic=False） |
| ETF universe | 动态过滤（[backtest.py P3-10](../engine/backtest.py)）—— 每月只用当时已上市 ETF |
| Train 期 | **N/A** —— D1 mapping 是事前规则，无需 train |

### 3.2 D1.1 mapping（与 D1 完全一致）

```python
shock_state = {gpr_z, epu_z, nvix_z}  at month t  (来自 shocks.parquet)
max_z = max(|gpr_z|, |epu_z|, |nvix_z|)

if max_z >= 2.0:    vol_multiplier = 0.50
elif max_z >= 1.0:  vol_multiplier = 0.75
else:               vol_multiplier = 1.00

construct_portfolio(target_vol = base × vol_multiplier)
```

**阈值 {1.0, 2.0} 与 multipliers {0.50, 0.75, 1.00} 与 D1 完全一致。不调。**

### 3.3 三组对照（与 D1 完全一致）

| Arm | narrative_context_func |
|---|---|
| A baseline | None |
| B gate | `make_gate_factory(shocks_z, "B", ...)` |
| C placebo | `make_gate_factory(shocks_z, "C", ...)` (random N(0,1)) |

### 3.4 PBO sweep（与 D1 完全一致）

5 个 threshold 组合：
- (1.00, 2.00) — primary
- (0.75, 2.00)
- (1.25, 2.00)
- (1.00, 1.75)
- (1.00, 2.25)

PBO ≤ 50% 闸门保持。

---

## 4. Reject 闸门（事前定，锁定）

| 闸门 | 阈值 | 与 D1 比 |
|---|---|---|
| ΔSharpe(B - A) | ≥ **0.10** | **不变** |
| **NW HAC t-stat (B - A)** | ≥ **1.0** | **改：1.5 → 1.0**（power analysis 依据） |
| Sharpe(B) - Sharpe(C) | ≥ **0.05** | **不变** |
| 任一 subperiod ΔSharpe | ≥ **-0.20** | **不变** |
| PBO of arm B | ≤ **0.50** | **不变** |

**5 闸门全 PASS → D1.1 PASSED → 进 F 做 confirmatory。任意 1 fail → REJECTED → 关 narrative 方向。**

---

## 5. Subperiod 划分（适配 192 月新期间）

| Label | 起 | 止 | 月数 | 经济背景 |
|---|---|---|---|---|
| P0_post_gfc | 2010-05 | 2014-12 | 56 | 后 GFC 复苏 + EU 主权债危机 + Taper Tantrum (2013-05) |
| P1_pre_covid | 2015-01 | 2020-06 | 66 | Oil crash 2015-16 + 2018-Q4 sell-off + COVID 起 |
| P2_covid | 2020-07 | 2021-12 | 18 | COVID 复苏 + meme stock + 通胀起 |
| P3_tightening | 2022-01 | 2026-05 | 53 | Fed 加息 + 俄乌 + 银行危机 + AI revolution |

**每个 subperiod ΔSharpe(B-A) ≥ -0.20** 才 PASS（与 D1 原 spec subperiod 闸门一致）。
任一 subperiod 触发 -0.20 → fail。

---

## 6. 实施

**复用 D1 infrastructure**——0 新代码：

| 文件 | 状态 |
|---|---|
| `engine/narrative/risk_gate.py` | ✅ 已写好 |
| `engine/narrative/backtest_d1.py` | ✅ 已写好 |
| `engine/portfolio.py` duck-typed branch | ✅ 已写好 |
| `tests/test_narrative_overlay.py` | ✅ 17/17 通过 |
| `data/shocks.parquet` | ✅ 真实数据 cache |

**唯一改动**：执行命令时改 `--oos-start` 参数。

**执行命令**（spec 锁定后由 Claude 跑）：
```bash
D:/python/python.exe -m engine.narrative.backtest_d1 \
    --oos-start 2010-05-01 --oos-end 2026-05-31 \
    --seed 7
```

预计运行时间 **25-40 分钟**（含 PBO sweep 5× run_backtest）。

---

## 7. 学术诚实声明

按 [feedback_quant_perspective.md](../memory/feedback_quant_perspective.md) 主动指出 D1.1 局限：

1. **Power 仍不足**：24% at ΔSharpe=0.10 是 exploratory 标准，不是 confirmatory
2. **88 月 D1 旧数据 contamination**：D1.1 OOS 192 月**包含** D1 旧 88 月（2019-2026）。
   作为 mitigation，subperiod 报告会分层（P0/P1 是 fresh, P2/P3 与 D1 重叠）—— 但整体 metrics
   有 ~46% 数据 contamination
3. **ETF universe 早期不全**：2010-2014 期 KWEB/ASHR/MTUM/USMV/QUAL/INDA 等若干 factor/regional ETF
   尚未上市；动态过滤会自动跳过 —— 这是 conservative bias，可能 understate alpha
4. **NVIX-proxy 替代 NVIX**：仍是 VIX 月末（Manela 主页 404，未 fix），数据质量限制
5. **Sub-period 不平衡**：P0/P1/P3 各 50+ 月，但 P2_covid 只 18 月 —— 单一 subperiod fail
   (≥ -0.20) 触发 reject 在 P2 上对随机噪声敏感
6. **D1.1 PASS 不等于 confirmed alpha**：spec § 0 + § 2.4 强调 exploratory 性质。Production 决策必须
   等 F (paper trading) 6-12 月累积

---

## 8. PASS / REJECT 路径

### PASS 路径（5 闸门全过）

1. 写 `docs/decisions/narrative_risk_gate_d1_1_passed.md`
2. 启动 F (paper trading 6-12 月) spec
3. 这是项目第一次 narrative-driven alpha 跨 spec 闸门 —— 但仍需 confirmatory

### REJECT 路径（任 1 闸门 fail）

1. 写 `docs/decisions/narrative_risk_gate_d1_1_rejected.md`
2. **整个 narrative direction 终止**
3. FactorMAD 接管 P0 (Phase 0 期间已暂停的 sprint 完全恢复)
4. memory 标 narrative direction final fate

---

## 9. 用户审阅 checklist（≤ 3 分钟）

任意一项 NO → spec 修订，不开工：

- [ ] Power analysis (§ 2) 透明且可信
- [ ] Threshold 1.0 + FPR 16% 的 trade-off 你接受
- [ ] OOS 192 月 (2010-05 至 2026-05) 确认
- [ ] 4 闸门保持 + subperiod 划分（P0/P1/P2/P3）合理
- [ ] D1 旧 88 月 contamination 在 spec § 7.2 已诚实标注，可接受
- [ ] PASS = exploratory not confirmed → 后续 F 仍需 6-12 月
- [ ] REJECT = 整个 narrative 方向终止 + FactorMAD 重启
- [ ] 不允许结果出来后调闸门（spec 锁定硬条件）

---

**审阅人签字位**：（spec 通过后我立即跑 backtest，约 25-40 分钟）
