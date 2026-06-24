// frontend/lib/labels.ts — human-readable display names for snake_case identifiers.
// No raw underscored ids surface in the UI: agents/strategies/sleeves get proper names; anything
// unmapped falls back to a generic prettifier. One place owns the mapping.

const AGENT_NAMES: Record<string, string> = {
  chief_of_staff:            "Chief of Staff",
  risk_manager:              "Risk Manager",
  dq_inspector:              "DQ Inspector",
  decay_sentinel:            "Decay Sentinel",
  anomaly_sentinel:          "Anomaly Sentinel",
  attribution_analyst:       "Attribution Analyst",
  audit_recorder:            "Audit Recorder",
  devils_advocate:           "Devil's Advocate",
  ops_watchdog:              "Ops Watchdog",
  d_pead_plus_llm_extractor: "D-PEAD Extractor",
  forensic_news_context:     "Forensic News Context",
  deepseek:                  "DeepSeek",
};

const WORKLOAD_NAMES: Record<string, string> = {
  narrator:        "Narrator",
  rm_agent:        "Risk Manager",
  massive_context: "Massive Context",
};

const SLEEVE_NAMES: Record<string, string> = {
  etf_l1:           "ETF L1",
  ss_sp500:         "S&P 500 single-stock",
  cta_defensive:    "CTA defensive",
  rms_crisis_hedge: "Crisis hedge",
  alpha_equity_ls:  "Alpha equity L/S",
  alpha_single_stock: "Alpha single-stock",
  insurance:        "Insurance",
  cta_overlay:      "CTA overlay",
};

const STRATEGY_NAMES: Record<string, string> = {
  K1_BAB:            "K1 BAB",
  D_PEAD:            "D-PEAD",
  PATH_N:            "Path N",
  CTA_PQTIX:         "CTA PQTIX",
  AC_TLT_GLD:        "AC TLT/GLD",
  cross_asset_carry: "Cross-asset carry",
};

/** snake_case / kebab → Title Case words. */
export function prettify(s: string): string {
  if (!s) return s;
  return s.replace(/[_-]+/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).trim();
}

/** snake_case → Sentence case (nicer for tool names: "delegate_to_specialist" → "Delegate to specialist"). */
function sentence(s: string): string {
  if (!s) return s;
  const w = s.replace(/[_-]+/g, " ").trim();
  return w.charAt(0).toUpperCase() + w.slice(1);
}

export const agentName    = (id: string) => AGENT_NAMES[id] ?? prettify(id);
export const workloadName = (id: string) => AGENT_NAMES[id] ?? WORKLOAD_NAMES[id] ?? prettify(id);
export const sleeveName   = (id: string) => SLEEVE_NAMES[id] ?? prettify(id);
export const strategyName = (id: string) => STRATEGY_NAMES[id] ?? prettify(id);
export const toolName     = (id: string) => sentence(id);
export const roleName     = (id: string) => prettify(id);  // regime_premium → "Regime Premium"

// Strip snake_case tokens out of FREE TEXT (alert messages, evidence): replace known
// sleeve/strategy ids with proper names, and any other snake_case token with spaced words —
// so no underscore survives in a sentence.
export function humanizeText(s: string): string {
  if (!s) return s;
  return s.replace(/[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+/g, (tok) =>
    SLEEVE_NAMES[tok] ?? SLEEVE_NAMES[tok.toLowerCase()] ?? STRATEGY_NAMES[tok] ?? tok.replace(/_/g, " "),
  );
}
