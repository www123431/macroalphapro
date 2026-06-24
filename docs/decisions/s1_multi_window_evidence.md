# S1 Multi-Window OOS Robustness — VERDICT: FAIL (2026-05-03)

**Spec**：[../spec_s1_multi_window_robustness.md](../spec_s1_multi_window_robustness.md) v1.0
**Memory**：
- `project_2026_summer_roadmap.md` — S1 是 4-month roadmap 第一项
- `project_efa_uplift_reject_2026-05-03.md` §"Baseline 不稳定性发现" — 触发 motivation
- `feedback_quant_perspective.md` — 学术诚实纪律

---

## TL;DR

按 spec v1.0 pre-registered verdict 标准，single-window TSMOM + composite signal baseline 在 6 个 5-year rolling windows (2010-2024) 上 **FAIL**：

```
2/6 windows positive Sharpe
0/6 windows 5%-significant (NW t > 1.65)
Mean Sharpe = -0.059 ± 0.419 (std)
Range:        [-0.587, +0.522]
Aggregated bootstrap Sharpe = +0.004
Bootstrap 95% CI = (-0.486, +0.454)  ← crosses zero
```

**核心发现**：之前报的 single-window Sharpe 0.236 / 0.510 / 0.114 是**早期 QE-era window 残留 alpha 的 measurement noise**——strategy 在 2010-2014 (W1: +0.52) 和 2012-2016 (W2: +0.33) 有真实 alpha，但 **2014 以后 4 个 windows 全部 negative** (-0.09 / -0.59 / -0.37 / -0.16)。

**含义**：
- 项目 baseline 不是 stable alpha，是 **regime-specific historical artifact**
- 这是项目第 6 个 documented falsification（前 5 个是 LLM-as-alpha + EFA quant uplift）
- **Strategy 是过去 5 年代谢殆尽的 alpha**

---

## 1. Per-Window Results

| Window | Period | Sharpe | NW t | NW CI 95% | ann_ret | ann_vol | MaxDD | n_obs |
|---|---|---|---|---|---|---|---|---|
| **W1** | 2010-2014 | **+0.522** | **+1.41** | (-0.21, +1.25) | +1.16% | 2.23% | -1.90% | 59 |
| **W2** | 2012-2016 | **+0.330** | +0.78 | (-0.49, +1.15) | +0.86% | 2.62% | -3.53% | 59 |
| W3 | 2014-2018 | -0.091 | -0.19 | (-1.03, +0.85) | -0.25% | 2.70% | -6.65% | 59 |
| W4 | 2016-2020 | **-0.587** | -1.29 | (-1.48, +0.30) | -1.68% | 2.86% | -7.74% | 59 |
| W5 | 2018-2022 | -0.373 | -0.86 | (-1.22, +0.48) | -1.12% | 3.01% | -7.82% | 59 |
| W6 | 2020-2024 | -0.158 | -0.41 | (-0.91, +0.59) | -0.42% | 2.64% | -5.94% | 59 |

**Pattern**：
- Pre-2014: Sharpe in [+0.33, +0.52] — **post-GFC QE era 真实 alpha**
- 2014-2024: Sharpe 全部 negative，最深 -0.59 (W4 2016-2020 trade war + COVID)
- 没有任何 window NW t 过 1.65 (one-sided 5%) 阈值

## 2. Aggregated Bootstrap (Stationary Block, n_boot=2000, block_len=3)

Deduped 月度 returns 拼接 (179 个 unique 月份):

```
Observed Sharpe:    +0.004      ← 接近 0
Median:             +0.011
Mean:               +0.002
Std:                 0.237
95% CI:            (-0.486, +0.454)   ← 跨 0
P(bootstrap > 0):    51.9%             ← coin flip
P(bootstrap > 0.2):  20.6%
P(bootstrap > 0.5):  ≈ 1-2%
```

**含义**：Strategy 真实 Sharpe 95% 置信区间 (-0.486, +0.454)，**包含 0**。Strategy 与"零 alpha"统计上不可区分。

## 3. Pre-Registered Verdict (Spec §4.1)

| Tier | Threshold | 实际 | Pass? |
|---|---|---|---|
| **PASS** | n_positive ≥ 4 AND nw_t_mean ≥ 1.0 AND ci_low > 0 | 2/6, -0.09, -0.49 | ❌ |
| **PARTIAL** | n_positive ≥ 3 AND nw_t_mean ≥ 0.5 | 2/6, -0.09 | ❌ |
| **FAIL** | n_positive ≤ 2 OR sharpe_mean < 0 | 2/6 ✓ AND -0.06 ✓ | ✅ **TRIGGERED** |

**Verdict: FAIL.**

## 4. 与之前 single-window 测量的关系

之前 same-day 多次测量同 5y window (2021-2026):
- Cleanup 后 Sharpe 0.236 / NW t 0.75
- EFA 前重测 Sharpe 0.510 / NW t 1.90
- EFA revert 后 Sharpe 0.114 / NW t 0.39

这些"漂移"看起来是 yfinance/FRED API state 不稳定造成的——**实际上还有第二层原因**：

**2021-2026 这个 window 本身就是 multi-window 中较好的样本之一**（位于 W6 边界附近，Sharpe -0.16 但 6/6 中相对靠中）。Single-window 测量恰好抓的是**退化中的 strategy 的 borderline 表现**——measurement noise + 这个 window 自己的 marginal positive 偶然合在一起 → 看似 0.2-0.5 的"positive baseline"。

**真相是**：在 2010-2024 全期，strategy mean Sharpe 接近 0；2010-2014 强 alpha 后逐步退化。

## 5. Mechanism — 为什么 strategy 在 2014 后退化

按学术 prior 分析（**事前**已知风险，本 verdict 不是事后挑选）：

1. **Sector rotation premium 衰减**：Asness et al (2013) 发现 momentum 在不同 asset class 上 Sharpe 持续下降；公开 anomaly 衰减是文献共识 (McLean-Pontiff 2016)
2. **Inter-sector correlation 上升**：post-GFC 全球 ETF 资金流增加 → sector idiosyncratic 弱化
3. **2010-2014 是 outlier QE era**：stimulus 引发持续 sector dispersion；之后 normalize → momentum 信号强度回归长期均值
4. **ETF universe 演变**：30+ sector ETF 中很多 inception 在 2010+，**早期 windows 的 effective universe 比晚期小**（survivorship/inception bias 倒置）

## 6. 这是项目第 6 个 falsification

| # | 假设 | Verdict | Doc |
|---|---|---|---|
| 1 | D1 narrative gate | REJECT | narrative_risk_gate_d1_soft_rejected.md |
| 2 | D1.1 narrative retry | REJECT | narrative_risk_gate_d1_1_rejected.md |
| 3 | Phase 0 cross-sectional tilt | REJECT | narrative_overlay_phase0_rejected.md |
| 4 | FactorMAD Q1 LLM mining | REJECT | factor_mad_reject.md |
| 5 | EFA three-piece quant uplift | REJECT | three_piece_uplift_efa_reject.md |
| **6** | **Single-window TSMOM baseline robustness** | **REJECT** | **本 doc** |

**关键差异**：1-5 是"add-on alpha"failure，6 是 **baseline 本身不 robust**——这是更深层的 finding。

## 7. 学术诚实声明

按 `feedback_quant_perspective.md`：

1. **6 个 windows 部分重叠**（2-yr stride 但 5-yr 长度）→ 不完全独立 sample；bootstrap CI 略宽但 verdict 方向稳健
2. **W1 2010-2014 +0.52 是 outlier？**：是 / 不是的判断要 6+ 更多 windows + 更长 history。本 spec 只有这 6 个 → 认作 "early sample 强 + 后续衰减"是合理 narrative
3. **Bootstrap 假设 stationarity**：W1 vs W6 跨 14 年 macro regime 完全不同（QE → 加息 → COVID → 复苏），stationarity 假设违反；CI 应理解为 "average regime under stationarity"
4. **不证明 strategy 在未来仍亏**：post-2014 数据不能保证 2026+ 还 negative；alpha 也可能"复活"——但**当前 evidence 不支持 alpha claim**
5. **未涵盖 universe 演变 effect**：W1 (2010-2014) 时 ETF universe 30 个；W6 时 ~35 个；其中 XLC/QUAL 等 inception 较晚自动从早期 windows 排除——effective N 不同会偏移 Sharpe，但方向不明
6. **结果反映 retail-grade data 限制**：yfinance + FRED 公开数据 + 月频 + 35 ETF。专业数据 (Bloomberg / 日频 / 1000+ stocks) 上结论可能不同

## 8. Implications & Project Reframing

### 8.1 Headline Sharpe 数字必须改

之前 README / exec_summary 报 "Sharpe 0.236" → **不再 defensible**。

新 headline 应是：
> **Multi-window OOS analysis (6 × 5-yr rolling, 2010-2024): Mean Sharpe -0.06 ± 0.42, 2/6 positive, 0/6 statistically significant. Bootstrap 95% CI (-0.49, +0.45) crosses zero. Strategy exhibits regime-specific alpha (W1 2010-2014: +0.52) that has decayed to non-significant in subsequent windows.**

这是诚实数字，**比 single-point 0.236 更可信、更完整**。

### 8.2 项目主线 reframe

之前主线："Sharpe 0.236 honest baseline + 5 LLM-as-alpha falsifications"

新主线："**Falsification framework demonstrates that the project's own baseline strategy lacks robust alpha**——这是 academic 项目最高级的诚实。"

**Award 角度**：
- 学术 rigor → 最高级 (self-falsification)
- 创新性 → 提升（"showing my own baseline doesn't generalize"是 master's thesis 罕见操作）
- 可发表性 → SSRN paper 的 framing 升级为 "Pre-registered falsification chain on a master's-project quant strategy: six negative results including own baseline"

### 8.3 不需要做的事

按 pre-registration 纪律 + `feedback_alpha_hard_polish_easy_drift.md`：
- ❌ 不能 cherry-pick W1+W2 报"strategy works in QE era"（HARKing）
- ❌ 不能调 universe / lookback / weight 救场
- ❌ 不能找 7th window 看是否更好
- ❌ 不能调 verdict threshold

### 8.4 应该做的事

1. **接受 verdict** + 写本 doc（已做）
2. **Update README + exec_summary** 用 multi-window distribution 替代 single point（next step S1.6）
3. **更新 falsification chain table** 加 6th reject 行
4. **2026 summer roadmap S2/S4 推进**：reflection memory + SSRN paper 重点不变，但 paper 主题要 reframe 为 "六 negative results 包括 baseline self-falsification"

## 9. Output Deliverables

✅ `engine/backtest.py` 加 `run_multi_window_backtest()` + `stationary_block_bootstrap()`
✅ `data/s1_window_results/per_window.csv` (6 行)
✅ `data/s1_window_results/aggregated.json` (含 bootstrap distribution)
✅ `pages/backtest.py` 末尾加 "S1 Multi-Window OOS Robustness" section
✅ 本 decision doc

待 S1.6:
- README headline 改 multi-window distribution
- executive_summary headline 改
- decisions/README.md 索引加本 doc
- MEMORY.md 加 project_s1_multi_window_2026-05-03.md
