// frontend/lib/glossary.ts — short EN/中文 definitions + reference frames for the terminal's
// metrics. A number alone is unreadable; every metric label can hover-define itself (def) and show
// what "good" is (ref). Plain data (no hooks) so any component can import it; the <Metric> /
// <GlossaryLabel> client components pick the language. `ref` mirrors the locked Risk Manager
// thresholds (spec id=69) — stable config, shown inline as the at-a-glance reference frame.
import type { Lang } from "@/lib/i18n";

export interface GlossaryEntry { en: string; zh: string; ref?: { en: string; zh: string } }

export const GLOSSARY: Record<string, GlossaryEntry> = {
  gross: {
    en: "Gross exposure = Σ|weights|. Total leverage deployed (longs + shorts, absolute).",
    zh: "总敞口 = Σ|权重|。投入的总杠杆(多 + 空,取绝对值)。",
    ref: { en: "target 1.50× · cap 2.0×", zh: "目标 1.50× · 上限 2.0×" },
  },
  net: {
    en: "Net exposure = Σ weights (longs − shorts). Directional market tilt.",
    zh: "净敞口 = Σ权重(多 − 空)。方向性的市场暴露。",
    ref: { en: "band [−100%, +100%]", zh: "区间 [−100%, +100%]" },
  },
  hhi: {
    en: "Herfindahl-Hirschman index = Σ wᵢ² of normalized weights. Higher = more concentrated (1 = single name).",
    zh: "赫芬达尔集中度 = 归一化权重的 Σ wᵢ²。越高越集中(1 = 全押单一标的)。",
    ref: { en: "cap 0.25", zh: "上限 0.25" },
  },
  max_weight: {
    en: "Largest single-name weight (absolute). Single-name concentration cap.",
    zh: "最大单一标的权重(绝对值)。单一标的集中度上限。",
    ref: { en: "cap 25%", zh: "上限 25%" },
  },
  short_ratio: {
    en: "Short-side weight as a fraction of gross. How much of the book is short.",
    zh: "空头权重占总敞口的比例。账本里有多少是做空。",
    ref: { en: "cap 50% of gross", zh: "上限 总敞口的 50%" },
  },
  var95: {
    en: "1-day Value-at-Risk at 95%: the loss the book should not exceed on 95% of days.",
    zh: "95% 单日在险价值:95% 的交易日里账本亏损不应超过该值。",
    ref: { en: "warn 2.5% · halt 4%", zh: "预警 2.5% · 熔断 4%" },
  },
  es95: {
    en: "1-day Expected Shortfall at 95%: the average loss on the worst 5% of days (tail risk).",
    zh: "95% 单日预期损失:最差 5% 交易日的平均亏损(尾部风险)。",
    ref: { en: "warn 3.5% · halt 5.5%", zh: "预警 3.5% · 熔断 5.5%" },
  },
  p_risk_on: {
    en: "Risk-on probability from the MSM regime model (0-100%). High = the market state favors risk.",
    zh: "MSM 状态模型给出的风险偏好概率(0-100%)。高 = 市场状态偏好风险。",
  },
  full_sharpe: {
    en: "Full-sample annualized Sharpe ratio (return per unit of volatility, since inception).",
    zh: "全样本年化夏普比率(自成立以来,单位波动的收益)。",
  },
  rolling_sharpe: {
    en: "Trailing 36-month rolling Sharpe. Compared to the full-sample Sharpe to detect decay.",
    zh: "滚动 36 个月夏普。与全样本夏普对比以检测衰减。",
  },
  decay_ratio: {
    en: "Rolling Sharpe ÷ full-sample Sharpe. Below ~0.5 flags a decaying alpha.",
    zh: "滚动夏普 ÷ 全样本夏普。低于约 0.5 视为 alpha 在衰减。",
  },
  crisis_payoff: {
    en: "Average monthly return during market-stress months. For a hedge this should be POSITIVE (that's its job).",
    zh: "市场承压月份的平均月收益。对冲腿这一项应为正(这正是它的职责)。",
  },
  signal_ic: {
    en: "Information coefficient: rank-correlation of the signal with forward returns. >0 = the signal still predicts.",
    zh: "信息系数:信号与未来收益的秩相关。>0 = 信号仍有预测力。",
  },
  crisis_payoff_short: { en: "crisis payoff", zh: "危机收益" },
};

export const glossaryText = (term: string, lang: Lang): string | undefined => {
  const e = GLOSSARY[term];
  return e ? e[lang] : undefined;
};
export const glossaryRef = (term: string, lang: Lang): string | undefined => {
  const e = GLOSSARY[term];
  return e?.ref ? e.ref[lang] : undefined;
};
