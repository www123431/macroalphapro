// frontend/lib/api.ts — typed client for the FastAPI backend (api/main.py).
// BASE overridable via NEXT_PUBLIC_API_URL; defaults to the local uvicorn dev server.

// Same-origin by default (the production build is served BY FastAPI at one port, so
// /api/* is relative). In `next dev` we set NEXT_PUBLIC_API_URL=http://localhost:8000
// (.env.development) to reach the separate backend.
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(`${res.status} ${path}: ${detail?.detail ?? res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(`${res.status} ${path}: ${detail?.detail ?? res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ── /api/decay/report ────────────────────────────────────────────────────────
export interface MechanismHealth {
  role: string;
  weight: number | null;
  full_sharpe: number | null;
  rolling_sharpe: number | null;
  rolling_t?: number | null;
  decay_ratio: number | null;
  crisis_payoff: number | null;
  signal_ic: number | null;
  structural_decay: boolean;
  decay_reason?: string;     // the deterministic per-mechanism rule + its evaluation
}
export interface PairCorr {
  pair: string;
  rolling_corr: number | null;
  downside_corr: number | null;
  stress_corr: number | null;
}
export interface DecayReport {
  as_of: string;
  as_of_age_days?: number;
  overall: "HEALTHY" | "WATCH" | "ACTION";
  realloc_action: boolean;
  n_mechanisms: number;
  mechanisms: Record<string, MechanismHealth>;
  pairs: PairCorr[];
  base_weights: Record<string, number>;
  recommended_weights: Record<string, number>;
  alarms: { level: string; message: string }[];
  narrative: string;
  verdict_basis?: { rule: string; deciding_level: string | null; driving_alarms: string[]; n_driving: number };
  decided_by?: string;
  narrated_by?: string;
}

// ── /api/book/state ──────────────────────────────────────────────────────────
export interface BookStrategy {
  name: string; sleeve: string; status: string;
  n_positions: number | null; intra_w: number | null; notes: string;
}
export interface BookState {
  as_of: string;
  strategies: BookStrategy[];
  combined_gross: number | null;
  combined_target_gross?: number | null;
  combined_net: number | null;
  combined_n: number | null;
  sleeve_attribution?: Record<string, number> | null;
}

// ── /api/book/nav ──────────────────────────────────────────────────────────
export interface NavDay {
  date: string;
  nav_close: number | null;
  daily_dietz: number | null;
  external_flow: number;
  // 2026-06-02 SPY benchmark (for the NAV chart overlay; null when feed
  // failed on that date — the chart renders only present points).
  benchmark_close?: number | null;
}
export interface DailyBrief {
  as_of: string | null; regime?: string; regime_prev?: string; regime_changed?: boolean;
  p_risk_on?: number | null; n_long?: number; n_short?: number;
  n_entries?: number; n_invalidations?: number; n_rebalance?: number; icir_month?: number | null;
  signal_flips?: unknown[]; risk_alerts?: unknown[];
  book_as_of?: string;            // as-of of the live book the long/short counts come from
  long_short_source?: string;     // "live_book" when n_long/n_short are computed from the book
  // 2026-06-14: brief now derives as_of from live ui_artifact (was stuck
  // on dead snapshot table writer). regime is separate cron — surface
  // its own staleness explicitly so the user sees "book fresh, regime
  // 15d stale" honestly instead of one blended timestamp.
  regime_as_of?:      string | null;
  regime_days_stale?: number | null;
  extras_table_as_of?: string | null;  // last write to the legacy snapshot writer
}
export interface TradeRow {
  date: string; strategy: string; sleeve: string; ticker: string;
  side: string; weight: number | null; signal: number | null; event?: string | null;
}
export interface Blotter { as_of: string | null; n_total: number; trades: TradeRow[]; }

// ── /api/book/positions (combined per-ticker holdings) ───────────────────────
export interface PositionLeg {
  strategy: string; signal: number | null; event: string | null;
  is_rebalance: boolean; horizon_days: number | null;
  spec_id: number | null; spec_hash: string | null; date: string | null;
}
export interface BookPosition {
  ticker: string; weight: number; side: "long" | "short";
  strategies: string[]; sleeves: string[]; legs?: PositionLeg[];
}
export interface BookPositions {
  as_of: string | null; n: number; n_long: number; n_short: number;
  gross: number; net: number; biggest: BookPosition | null; positions: BookPosition[];
}

// ── /api/dq (DQ Inspector live data-quality verdict) ─────────────────────────
export interface DQCheck { mode_id: string; severity: string; rule: string; observed: number | null; threshold: number | null; affected: string[] }
export interface DQReport {
  as_of?: string; verdict: string; n_breaches?: number; checks?: DQCheck[];
  rationale?: string; scope?: string; decided_by?: string; narrated_by?: string;
  available?: boolean; reason?: string;
}

// ── /api/book/tracking (accumulating live-vs-backtest tracker) ───────────────
export interface LiveTracking {
  available: boolean; reason?: string;
  n_live_days?: number; live_window?: { start: string; end: string };
  live?: { ann_ret: number; ann_vol: number; sharpe: number | null; cum_return: number };
  backtest_expected?: { ann_ret: number; ann_vol: number; sharpe: number; max_dd: number };
  tracking?: { ann_ret_diff: number; live_cum: number; expected_cum: number; t_stat_vs_expected: number | null };
  significant?: boolean; min_days_for_significance?: number; note?: string;
}

// ── /api/research_ops/inbox — Research Ops composite inbox ──────────
export type ResearchOpsLane = "engine" | "direction" | "methodology" | "graveyard";
export type ResearchOpsTone = "ok" | "warn" | "alert" | "muted" | "info";

export interface ResearchOpsItem {
  id: string;
  ts: string;
  lane: ResearchOpsLane;
  source: string;
  title: string;
  summary: string;
  tone: ResearchOpsTone;
  href: string | null;
  metadata: Record<string, any>;
  unread?: boolean;
}

export interface ResearchOpsInbox {
  as_of: string;
  doctrine: string;
  n_total: number;
  n_unread: number;
  by_lane: Record<string, number>;
  items: ResearchOpsItem[];
  available?: boolean;
  reason?: string;
}

export interface ResearchOpsLiterature {
  as_of: string;
  doctrine: string;
  n_total: number;
  n_unread: number;
  by_family: Record<string, number>;
  items: ResearchOpsItem[];
  available?: boolean;
  reason?: string;
}


// ── /api/governance/approvals — v2 governance gateway ──────────────
export type V2ApprovalStatus = "pending" | "approved" | "rejected" | "expired";
export type V2ApprovalType =
  | "deploy_config_promote"
  | "weight_method_change"
  | "sleeve_weight_change"
  | "sleeve_add"
  | "sleeve_remove"
  | "manifest_edit";

export interface V2ApprovalRow {
  id: string;
  ts: string;
  request_type: V2ApprovalType;
  title: string;
  summary: string;
  proposed_payload: Record<string, any>;
  current_state: Record<string, any>;
  evidence_pack: Record<string, any>;
  cooling_off_seconds: number;
  created_at: string;
  expires_at: string;
  status: V2ApprovalStatus;
  decided_by: string | null;
  decision_reason: string | null;
  decided_ts?: string;
  fast_approve: boolean;
  execution_log: string | null;
}

export interface V2ApprovalList {
  available: boolean;
  reason?: string;
  n_total?: number;
  n_pending?: number;
  items?: V2ApprovalRow[];
}

// ── /api/ops/llm_budget — operational LLM budget guardrail ─────────
export interface LlmBudgetConfig {
  monthly_cap_usd:     number;
  agent_caps_usd:      Record<string, number>;
  alert_threshold_pct: number;
  _source?:            string;
  _last_updated?:      string;
}

export interface LlmBudgetAgentRow {
  agent_id:    string;
  spend_usd:   number;
  calls:       number;
  cap_usd:     number | null;
  pct_of_cap:  number | null;
  status:      "ok" | "alert" | "over";
  last_ts:     string | null;
}

export interface LlmBudgetUsage {
  month_start:         string;
  as_of:               string;
  monthly_cap_usd:     number;
  alert_threshold_pct: number;
  total_spend_usd:     number;
  total_pct_of_cap:    number | null;
  total_status:        "ok" | "alert" | "over";
  agents:              LlmBudgetAgentRow[];
  n_agents_alert:      number;
  n_agents_over:       number;
}

export interface LlmBudgetResponse {
  available: boolean;
  reason?:   string;
  budget?:   LlmBudgetConfig;
  usage?:    LlmBudgetUsage;
}


// ── /api/system/version + /api/system/cache/invalidate ─────────────────────
export interface SystemVersion {
  git_sha: string;
  git_dirty: boolean;
  uptime_s: number;
  uptime_human: string;
  process_started_iso: string;
  n_cached_keys: number;
  cached_keys: string[];
}

// ── /api/deploy/manifest (single source of truth for what's live) ─────────────
export interface DeployManifestSleeve {
  name: string;
  role: "alpha" | "diversifier" | "insurance" | string;
  base_weight: number;
  regime_modulated: boolean;
  builder: string;
  target_vol: number;
  signing_spec_ids: number[];
}
export interface DeployManifest {
  available: boolean;
  reason?: string;
  config_id?: string;
  label?: string;
  summary?: string;
  deploy_date?: string;
  days_since_deploy?: number;
  book_vol_target?: number;
  signing_spec_ids?: number[];
  expected_stats?: { sharpe?: number; ann_ret?: number; ann_vol?: number; max_dd?: number; backtest_window?: string };
  sleeves?: DeployManifestSleeve[];
  regime_grids?: Record<string, Record<string, number>>;
  regime_classifier?: { kind?: string; threshold_sigma?: number; lookback_days?: number };
  // Empty list = healthy. Non-empty = Python defaults disagree with the manifest.
  code_drift_issues?: string[];
}

// ── /api/book/combined (deployed book: 5-sleeve config C + 2-mech narrative) ─
export interface CombinedBookStats { sharpe: number; ann: number; vol: number; maxdd: number; n: number }
export interface DeployedSleeve {
  name: string;
  role: "alpha" | "diversifier" | "insurance" | string;
  base_weight: number;
  regime_modulated?: boolean;
}
export interface DeployedBookBlock {
  config_name: string;
  deploy_date: string;
  book_vol_target: number;
  stats: CombinedBookStats;
  sleeves: DeployedSleeve[];
  regime_grids: Record<string, Record<string, number>>;
  note?: string;
}
export interface NarrativeTwoMechBlock {
  title: string;
  carry_risk_weight: number;
  combined: CombinedBookStats;
  equity_only: CombinedBookStats;
  note?: string;
}
export interface PreInsuranceBlock {
  config_name: string;
  stats: CombinedBookStats;
  note?: string;
}
export interface CombinedBook {
  available: boolean; reason?: string; carry_risk_weight?: number; book_vol_target?: number;
  combined?: CombinedBookStats; equity_only?: CombinedBookStats;
  dates?: string[]; equity_curve?: number[]; note?: string;
  // 2026-06-02 amendment — truthful deployed config block + narrative block.
  deployed?: DeployedBookBlock;
  narrative_2_mechanism?: NarrativeTwoMechBlock;
  // 2026-06-02 — pre-insurance reference comparison for Tearsheet.
  pre_insurance_3_mech?: PreInsuranceBlock;
}

// ── /api/execution (paper-broker target-vs-actual reconciliation) ────────────
export interface ExecutionRow {
  ticker: string; target_weight: number; actual_weight: number; drift: number;
  held_qty: number; on_target: boolean;
}
export interface ExecutionReconcile {
  available: boolean; reason?: string;
  broker?: string; paper?: boolean; as_of?: string;
  equity?: number; cash?: number;
  n_targets?: number; n_positions?: number;
  gross_target?: number; gross_actual?: number; undeployed_weight?: number;
  tracking_error?: number; max_abs_drift?: number; n_on_target?: number;
  order_status?: Record<string, number>;
  breaks?: { targeted_not_held: string[]; held_not_targeted: string[] };
  rows?: ExecutionRow[];
  futures_sleeve?: {
    venue: string; equity: number; n_contracts: number; nav_points: number;
    last_nav?: { date: string; nav: number } | null;
  };
}

// ── /api/book/overlay (operator discretionary overlay sleeve) ────────────────
export interface OverlayPosition {
  ticker: string; weight: number; rationale?: string; set_at?: string; approval_id?: number | null;
}
export interface OverlayTrade {
  ts: string; date: string; ticker: string; action: string;
  weight_before: number; weight_after: number; weight_delta: number;
  approval_id?: number | null; resolved_by?: string; rationale?: string;
}
export interface OverlayData {
  as_of: string | null; positions: OverlayPosition[]; gross: number; net: number; n: number;
  single_name_cap: number; gross_cap: number; recent_trades: OverlayTrade[];
}

// ── /api/book/risk-contrib (position-level risk decomposition) ───────────────
export interface RiskContrib {
  ticker: string; weight: number; pct_risk: number; vol_annual: number;
  mctr_annual: number; diversifying: boolean;
}
export interface RiskContribData {
  available: boolean; reason?: string;
  port_vol_annual?: number; n_obs?: number; lookback_start?: string;
  coverage?: { n_covered: number; n_total: number; weight_covered: number };
  contributions?: RiskContrib[]; as_of?: string; panel_built?: string; note?: string;
}

// ── /api/book/factor-exposure (cross-asset 5-macro-β) ────────────────────────
export interface FactorBeta { factor: string; beta: number; risk_share: number }
export interface FactorExposure {
  available: boolean; reason?: string; as_of?: string;
  n_obs?: number; period?: [string, string]; r2?: number; idiosyncratic?: number; alpha_daily?: number;
  factors?: FactorBeta[]; proxies?: Record<string, string>; note?: string;
}

// ── /api/book/scenarios (stress) ─────────────────────────────────────────────
export interface ScenarioWindow { k: number; ret: number | null; end_date: string | null }
export interface ScenarioAttr { ticker: string; contrib: number; ret: number }
export interface ScenarioData {
  available: boolean; reason?: string; as_of?: string;
  n_obs?: number; period?: [string, string];
  worst?: Record<string, ScenarioWindow>; best?: Record<string, ScenarioWindow>;
  worst_day?: { date: string; book_ret: number; attribution: ScenarioAttr[] };
  market?: { proxy: string; book_beta: number; shocks: { mkt_move: number; book_pnl: number }[]; note: string };
}

export interface ReplayWindowMeta {
  // 2026-06-02 v2 doctrine: principled_* fields encode the "all-components-
  // honest" start date. principled_binding_* names the sleeve / data source
  // that pins it.
  principled_start_date?: string;
  principled_start_reason?: string;
  principled_binding_sleeve?: string;
  principled_binding_data_source?: string;

  replay_parquet?: Record<string, {
    start: string; end: string; n_weeks: number;
    sleeves?: string[];
    sleeves_legacy_columns?: string[];
    note?: string;
    column_name_remediation?: Record<string, {
      is_actually: string;
      honest_name: string;
      replaced_by_native_sleeve?: string;
      window_in_name_is_historical_baggage?: boolean;
      kept_for?: string;
    }>;
  }>;
  sleeve_data_availability?: Record<string, {
    // legacy v1 fields kept for back-compat with older meta files
    cached_earliest?: string;
    cached_source?: string;
    native_earliest_if_repulled?: string;
    binding_constraint?: string;
    // v2 fields
    deepest_raw_history?: string;
    cached_processed_panel_starts?: string;
    binding_constraint_for_full_book?: string;
  }>;
  extension_options?: Record<string, {
    summary: string;
    estimated_window?: string;
    effort_hours?: number;
    wrds_pull_estimate_hours?: number;
    downstream_rebuild_hours?: number;
    blockers?: string[];
    blocker?: string;
  }>;
  _last_audit_ts?: string;
  _supersedes?: string;
}
export interface BookPerf {
  n_weeks: number; start: string; end: string;
  stats: { ann_ret: number; ann_vol: number; sharpe: number | null; max_dd: number };
  dates: string[]; equity: number[]; drawdown: number[]; rolling_sharpe: (number | null)[];
  window_meta?: ReplayWindowMeta | null;
}
export interface NavHistory {
  n_rows: number;
  first_date?: string;
  last_date?: string;
  nav_first?: number;
  nav_last?: number;
  total_return?: number;
  days?: NavDay[];
  message?: string;
}

// ── /api/agents ──────────────────────────────────────────────────────────────
export interface AgentCard {
  agent_id: string;
  name: string;
  kind: "supervisor" | "specialist";
  role_id: string;
  workload: string;       // provider+model routing string
  spec_ref: string;
  max_iterations: number;
  tools: string[];        // real per-persona tool palette
  scope: string;
}
export interface AgentsDirectory {
  chief_of_staff: AgentCard;
  specialists: AgentCard[];
  delegation_rule: string;
}

// ── /api/chat (SSE) ────────────────────────────────────────────────────────
export interface ChatEvent {
  type: "start" | "iteration" | "assistant_text" | "tool_call" | "tool_result" | "done" | "error";
  // union of optional fields across event types
  agent_id?: string;
  name?: string;
  n?: number;
  iteration?: number;
  text?: string;
  input?: Record<string, unknown>;
  preview?: string;
  is_error?: boolean;
  final_text?: string;
  cost_usd?: number;
  latency_ms?: number;
  n_iterations?: number;
  stop_reason?: string;
  new_messages?: unknown[];
  detail?: string;   // error
  status?: number;   // error (HTTP status)
}

function openChatStream(message: string, history: unknown[], signal?: AbortSignal): Promise<Response> {
  return fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
    signal,
  });
}

// POST + parse the text/event-stream (EventSource is GET-only, so we read the body stream).
// onEvent fires per server event; pass an AbortSignal to support a Stop button. The INITIAL
// connection is retried once on a transient network failure (NOT on app errors like 503/429,
// and NOT after the user aborts) — a blip at send time shouldn't kill the turn.
export async function streamChat(
  message: string,
  history: unknown[],
  onEvent: (ev: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response | null = null;
  for (let attempt = 0; attempt < 2; attempt++) {
    try { res = await openChatStream(message, history, signal); break; }
    catch (e) {
      if ((e as Error)?.name === "AbortError") return;          // user stopped — silent
      if (attempt === 0) {
        onEvent({ type: "error", detail: "connection lost — reconnecting…" });
        await new Promise((r) => setTimeout(r, 800));
        continue;
      }
      onEvent({ type: "error", detail: `backend unreachable: ${(e as Error)?.message ?? e}` });
      return;
    }
  }
  if (!res) return;
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try { detail = (await res.json())?.detail ?? detail; } catch { /* non-JSON */ }
    onEvent({ type: "error", detail, status: res.status });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try { onEvent(JSON.parse(payload) as ChatEvent); } catch { /* skip malformed frame */ }
    }
  }
}

// ── /api/research/graveyard ──────────────────────────────────────────────────
export interface GraveyardEntry {
  name: string;
  family: string;
  date: string;
  verdict: string;
  why: string;
}
export interface Graveyard {
  as_of: string;
  note: string;
  entries: GraveyardEntry[];
}

// ── /api/research/discovery/* (paper nominate + queue) ──────────────────────
export interface DiscoveryRouting {
  base_confidence?: number;
  family?: string;
  family_threshold?: number;
  family_bonus_applied?: boolean;
  adjusted_confidence?: number;
  borderline_floor?: number;
  routing?: string;
}
export interface DiscoveryMetaAdvisory {
  family?: string;
  prior_pass_probability?: number;
  credible_interval_95?: [number, number];
  observations_in_family?: number;
  prior_source?: string;
  advisory_note?: string;
  error?: string;
}
export interface DiscoveryQueueEntry {
  source?: string;
  source_id?: string;
  ident_type?: string;
  title?: string;
  abstract?: string;
  authors?: string;
  venue?: string;
  doi?: string | null;
  submitted_date?: string | null;
  citation_count?: number | null;
  confidence?: number | Record<string, unknown>;
  scoring_method?: string;
  routing?: DiscoveryRouting;
  meta_learner_advisory?: DiscoveryMetaAdvisory;
  llm_rescue?: {
    base_confidence?: number;
    hybrid_confidence?: number;
    rescued_features?: Array<{ llm_feature: string; regex_feature: string; weight: number }>;
    llm_cost_usd?: number;
    error?: string;
  };
  nominated_via?: string;
  ts?: string;
  verdict?: string;
  reason?: string;
}
export interface DiscoveryQueues {
  review: DiscoveryQueueEntry[];
  borderline: DiscoveryQueueEntry[];
}
export interface NominateResult {
  ok?: boolean;
  error?: string;
  title?: string;
  venue?: string;
  confidence?: number;
  routing?: string;
  scoring_method?: string;
  queued_to?: string;
  ident_type?: string;
  ident_id?: string;
}
export interface DiscoveryBookmarklet {
  bookmarklet: string;
  endpoint: string;
  instructions: string;
}
export interface WatchlistEntry {
  mechanism_id: string;
  registered_at: string;
  promoted_from: string;
  state: string;
  track_until: string;
  auto_gate_verdict?: string | null;
  auto_gate_sharpe?: number | null;
  auto_gate_deflated_sr?: number | null;
  forward_oos_sharpe?: number | null;
  calibration_delta?: number | null;
  notes?: string;
}
export interface WatchlistSummary {
  total: number;
  by_state: Record<string, number>;
  by_implementation: { ready: number; not_ready: number };
  overdue_for_review: number;
}
export interface WatchlistPayload {
  summary: WatchlistSummary;
  entries: WatchlistEntry[];
}
export interface PromoteResult {
  ok: boolean;
  mechanism_id: string;
  library_path: string;
  original_queue: string;
  title: string;
}
export interface SkipResult {
  ok: boolean;
  rejected_path: string;
  original_queue: string;
  title: string;
}

// ── /api/research/pit-audit (point-in-time / look-ahead integrity) ───────────
export interface PitSurface { strategy: string; surface: string; control: string; anchor?: string }
export interface PitCheck { name: string; status: string; detail: string; anchor?: string }
export interface PitAudit {
  available: boolean;
  book?: { as_of: string; overall: string; book_clean: boolean; dpead_data_verified: boolean; undocumented: string[]; surfaces: PitSurface[] };
  dpead?: { as_of: string; target: string; n_rows: number; critical_pass: boolean; overall: string; checks: PitCheck[] };
}

// ── /api/ops/cost ──────────────────────────────────────────────────────────
export interface CostAgent {
  agent_id: string;
  total_usd: number;
  calls: number;
  last_ts: string | null;
  providers: Record<string, number>;
}
export interface OpsCost {
  as_of: string;
  today_usd: number;
  last7_usd: number;
  last30_usd: number;
  lifetime_usd: number;
  calls_total: number;
  by_agent: CostAgent[];
  by_provider: { provider: string; total_usd: number }[];
}

// ── /api/ops/health ────────────────────────────────────────────────────────
export interface SloAgent {
  agent_id: string; n: number; success_rate: number | null;
  p50_ms: number | null; p95_ms: number | null; last_ts: string | null;
}
export interface ProviderKey {
  label: string; status: string;
  today_calls: number; today_errors: number;
  total_calls: number; total_errors: number; last_used: string | null;
}
export interface ManifestAgent {
  agent_id: string; model: string[]; n_tools: number; prompt_sha: string; tools_sha: string;
}
export interface OpsHealth {
  slo?: { n: number; success_rate: number | null; p50_ms: number | null; p95_ms: number | null; by_agent: SloAgent[]; error?: string };
  providers?: { routing: { workload: string; provider: string; model: string }[]; keys: ProviderKey[]; error?: string };
  governance?: {
    clean: boolean; changed: string[]; added: string[]; removed: string[];
    agents: ManifestAgent[]; eval_cases: number;
    posture: { llm_in_decision: boolean; authority_enforced: boolean; manifest_pinned: boolean };
    error?: string;
  };
}

// ── /api/alerts ────────────────────────────────────────────────────────────
export interface AlertRow {
  source: string; date: string; mode_id?: number | string;
  severity: string; cb_severity?: string; rule_description: string;
  halt_decision?: boolean; affected?: unknown; source_id?: unknown;
  // 2026-06-02 ack
  alert_key?: string;
  is_acknowledged?: boolean;
  ack_ts?: string;
  ack_justification?: string;
}
export interface AnomalyRow {
  scan_date: string; ticker: string; sector?: string; detector: string;
  event_class?: string; confidence_likert: number; horizon_days?: number; evidence: string;
  alert_key?: string;
  is_acknowledged?: boolean;
  ack_ts?: string;
  ack_justification?: string;
}
export interface AlertsData {
  as_of: string; days_back: number;
  n_alerts: number; alerts: AlertRow[];
  n_anomalies: number; anomalies: AnomalyRow[];
}

// ── /api/risk ──────────────────────────────────────────────────────────────
export interface RiskMode {
  mode_id: string; name: string; observed: number | null;
  threshold: string; verdict: string; live: boolean; detail: string;
}
export interface RiskConsole {
  as_of: string; overall_severity: string; halt: boolean; n_breaches: number; rationale?: string;
  metrics: {
    gross: number; net: number; hhi: number; max_weight: number; short_ratio: number;
    n_positions: number; n_strategies: number; n_ok: number;
    var95: number | null; es95: number | null;
  };
  modes: RiskMode[]; note: string;
  decided_by?: string;
  narrated_by?: string;
}

// ── /api/provenance ──────────────────────────────────────────────────────────
export interface ProvenanceSource {
  source: string; kind: string; as_of: string | null;
  built?: string; note?: string; bdays_stale: number | null;
}
export interface Provenance {
  as_of: string;
  sources: ProvenanceSource[];
  point_in_time: string;
}

// ── /api/approvals (L2 human-in-the-loop) ────────────────────────────────────
export interface Approval {
  id: number; created_at?: string; approval_type?: string; approval_class?: string;
  priority?: string; ticker?: string; sector?: string; triggered_condition?: string;
  triggered_date?: string; suggested_weight?: number | null; position_rank?: number | null;
  llm_confidence?: number | null; contradicts_quant?: boolean | null;
  approval_deadline?: string | null; review_rationale?: string; review_category?: string;
  status?: string; resolved_at?: string; resolved_by?: string; rejection_reason?: string;
  effect_en?: string; effect_zh?: string; executes?: boolean;   // what Approve actually does
}
export interface ApprovalsData {
  n_pending: number;
  approvals: Approval[];
  charter?: { en: string; zh: string };   // governance-queue routing charter
}
export interface ResolveResult {
  submitted: number;
  resolved: { id: number; ok: boolean; message: string }[];
  skipped: { id: number; reason: string }[];
}

// ── /api/approvals/{id} (deterministic decision context, P-AUDIT v1; 0-LLM) ──
export interface ApprovalDetailBase {
  approval_id: number;
  approval_type?: string; priority?: string; status?: string;
  sector?: string; ticker?: string; amount_or_weight?: number | null;
  triggered_condition?: string; triggered_date?: string; triggered_price?: number | null;
  approval_deadline?: string | null; deadline_days_left?: number | null;
  governing_spec_path?: string | null; governing_spec_hash?: string | null;
  last_amend_days?: number | null; spec_excerpt_first_200_chars?: string | null;
  linked_decision_log_id?: number | null; linked_watchlist_id?: number | null;
  contradicts_quant?: boolean; llm_confidence?: number | null;
}
export interface SimilarPast {
  approval_id: number; decision_date?: string | null; approval_date?: string | null;
  sector?: string; ticker?: string; direction?: string | null; amount?: number | null;
  verdict?: string; review_category?: string | null; review_rationale?: string | null;
  active_return?: number | null; hit_flag?: string; retrieval_method?: string;
}
export interface ReplayStep {
  ts: string; type: string; actor: string; payload_summary: string;
  run_id_link?: string | null; reconstructed?: boolean;
}
// decision_context layers + cb_status/harking/quant_ctx/reject_preview are deep, schema-varying
// deterministic payloads — typed as flexible records and rendered defensively in the UI.
type Bag = Record<string, unknown>;
export interface ApprovalDetail {
  found: boolean;
  approval_id: number;
  base?: ApprovalDetailBase;
  cb_status?: Bag | null;
  harking?: Bag | null;
  quant_ctx?: Bag | null;
  reject_preview?: Bag | null;
  decision_context?: Record<string, Bag>;
  similar_past?: SimilarPast[];
  similar_past_status?: "ok" | "unavailable";
  decision_replay?: ReplayStep[];
  review_categories?: string[];
}

// ── /api/freshness (single as-of authority across pipelines) ─────────────────
export interface FreshnessSource {
  source: string; as_of: string | null; age_days: number | null;
  threshold_days: number; stale: boolean;
}
export interface Freshness {
  as_of: string; sources: FreshnessSource[];
  overall: "fresh" | "stale"; worst_age_days: number | null;
}

// ── /api/ops/refresh (the staleness banner's remediation action) ─────────────
export interface RefreshStatus {
  running: boolean; trigger?: string | null; started_at: string | null; finished_at: string | null;
  exit_code: number | null; ok: boolean | null; message: string | null;
  log_tail: string | null; already_running?: boolean;
}

// ── /api/ops/eval-* (agent behavioral-eval scores, made provable) ────────────
export interface EvalCaseScore {
  case_id: string; agent_id: string; pass: number; n: number;
  pass_rate: number | null; wilson_ci: [number | null, number | null];
}
export interface EvalLatest {
  found: boolean; generated_at?: string; static_all_pass?: boolean;
  live?: {
    pass_rate: number | null; wilson_ci: [number | null, number | null];
    runs: number; runs_passed: number; n_cases: number; n_samples: number;
    total_cost_usd: number; cases: EvalCaseScore[];
  } | null;
}
export interface EvalRunStatus {
  running: boolean; started_at: string | null; finished_at: string | null;
  exit_code: number | null; ok: boolean | null; message: string | null; already_running?: boolean;
}

// P0b liveness surface (2026-06-02). Mirrors api/routes_research_tools.liveness_status.
export interface LivenessHeartbeatRow {
  ts: string;
  as_of: string;
  exit_code: number;
  status: string;          // success | feed_partial | halt_cb | halt_risk | halt_dq | ...
  n_orders: number | null;
  n_fills: number | null;
  equity_before: number | null;
  n_strategies: number | null;
  gross_weight: number | null;
  halted_at_step: string | null;
  broker_ack: string | null;
  log_file: string | null;
  errors: string[];
  // P1a broker reconciliation (2026-06-02)
  broker_echo: {
    status: string;
    n_orders_intended: number | null;
    n_orders_submitted: number | null;
    n_fills: number | null;
    fill_rate: number | null;
    equity_before: number | null;
    broker_ack: string | null;
    n_warnings: number | null;
    live: {
      equity: number; cash: number; buying_power: number;
      n_positions: number; gross_exposure: number;
      position_tickers: string[];
    } | null;
    explanation: string | null;
  } | null;
  // P1b NAV anomaly (2026-06-02)
  nav_anomaly: {
    status: string;
    equity: number;
    log_return: number | null;
    z_score: number | null;
    explanation: string;
    baseline_n?: number;
    baseline_mu?: number;
    baseline_sd?: number;
  } | null;
  // P0c data freshness (2026-06-02)
  data_freshness: {
    worst_status: "fresh" | "aging" | "stale" | "dead" | "missing" | "unknown";
    worst_source: string | null;
    n_dead: number; n_stale: number; n_aging: number;
    n_fresh: number; n_missing: number; n_unknown: number;
    n_total: number;
    headline: string;
  } | null;
  data_sources: Array<{
    source: string;
    description: string;
    latest_date: string | null;
    age_days: number | null;
    status: string;
    n_rows: number | null;
    error: string | null;
  }> | null;
}
export interface LivenessVerdict {
  verdict: "OK" | "WARN_STATUS" | "ALERT_NO_SHOW" | "INFO_OFF_HOURS" | "INFO_WEEKEND";
  explanation: string;
  as_of: string;
  checked_at: string;
  latest?: LivenessHeartbeatRow | null;
  age_min?: number | null;
}
export interface LivenessSummary {
  verdict_code: string;
  tone: "ok" | "info" | "warn" | "danger" | "muted";
  headline: string;
}
export interface LivenessStatus {
  verdict: LivenessVerdict;
  recent: LivenessHeartbeatRow[];
  summary: LivenessSummary;
}

export const api = {
  decayReport: () => get<DecayReport>("/api/decay/report"),
  provenance: () => get<Provenance>("/api/provenance"),
  freshness: () => get<Freshness>("/api/freshness"),
  refreshStatus: () => get<RefreshStatus>("/api/ops/refresh"),
  startRefresh: () => post<RefreshStatus>("/api/ops/refresh", {}),
  evalLatest: () => get<EvalLatest>("/api/ops/eval-latest"),
  evalRunStatus: () => get<EvalRunStatus>("/api/ops/eval-run"),
  startEval: () => post<EvalRunStatus>("/api/ops/eval-run", {}),
  approvals: (includeResolved = false) => get<ApprovalsData>(`/api/approvals?include_resolved=${includeResolved}`),
  approvalDetail: (id: number) => get<ApprovalDetail>(`/api/approvals/${id}`),
  resolveApprovals: (body: { ids: number[]; approved: boolean; rationale: string; category?: string }) =>
    post<ResolveResult>("/api/approvals/resolve", body),
  risk: (asOf?: string | null) => get<RiskConsole>(`/api/risk${asOf ? `?as_of=${asOf}` : ""}`),
  alerts: (daysBack = 30) => get<AlertsData>(`/api/alerts?days_back=${daysBack}`),
  alertsAcknowledge: (alertKey: string, kind: "alert" | "anomaly", justification = "") =>
    post<{ ok: boolean; alert_key: string }>("/api/alerts/acknowledge",
      { alert_key: alertKey, kind, justification }),
  alertsUnacknowledge: (alertKey: string) =>
    post<{ ok: boolean }>("/api/alerts/unacknowledge", { alert_key: alertKey }),
  opsCost: () => get<OpsCost>("/api/ops/cost"),
  opsHealth: () => get<OpsHealth>("/api/ops/health"),
  bookState: (asOf?: string | null) => get<BookState>(`/api/book/state${asOf ? `?as_of=${asOf}` : ""}`),
  bookNav: (daysBack = 120) => get<NavHistory>(`/api/book/nav?days_back=${daysBack}`),
  bookPositions: (asOf?: string | null) => get<BookPositions>(`/api/book/positions${asOf ? `?as_of=${asOf}` : ""}`),
  bookDates: () => get<{ dates: string[]; latest: string | null }>("/api/book/dates"),
  overlay: () => get<OverlayData>("/api/book/overlay"),
  combined: () => get<CombinedBook>("/api/book/combined"),
  deployManifest: () => get<DeployManifest>("/api/deploy/manifest"),

  // ── Governance Gateway v2 ───────────────────────────────────────
  researchOpsInbox: (since?: string) =>
    get<ResearchOpsInbox>(
      `/api/research_ops/inbox${since ? `?since=${encodeURIComponent(since)}` : ""}`,
    ),
  researchOpsLiterature: (since?: string) =>
    get<ResearchOpsLiterature>(
      `/api/research_ops/literature${since ? `?since=${encodeURIComponent(since)}` : ""}`,
    ),
  researchOpsLastVisit: () =>
    get<{ visited_ts: string | null }>(`/api/research_ops/last_visit`),
  researchOpsRecordVisit: (visited_ts?: string) =>
    post<{ visited_ts: string }>(`/api/research_ops/last_visit`, { visited_ts }),

  v2ListApprovals: (status?: V2ApprovalStatus, limit = 100) =>
    get<V2ApprovalList>(
      `/api/governance/approvals${status ? `?status=${status}&limit=${limit}` : `?limit=${limit}`}`,
    ),
  v2GetApproval: (id: string) =>
    get<V2ApprovalRow>(`/api/governance/approvals/${encodeURIComponent(id)}`),
  v2ApproveApproval: (id: string, body: { decided_by: string; reason?: string; force_pre_cooling?: boolean }) =>
    post<V2ApprovalRow>(
      `/api/governance/approvals/${encodeURIComponent(id)}/approve`,
      body,
    ),
  v2RejectApproval: (id: string, body: { decided_by: string; reason: string }) =>
    post<V2ApprovalRow>(
      `/api/governance/approvals/${encodeURIComponent(id)}/reject`,
      body,
    ),
  systemVersion: () => get<SystemVersion>("/api/system/version"),
  llmBudget:       () => get<LlmBudgetResponse>("/api/ops/llm_budget"),
  llmBudgetUpdate: (body: { monthly_cap_usd?: number; agent_caps_usd?: Record<string, number>; alert_threshold_pct?: number }) =>
    post<LlmBudgetResponse>("/api/ops/llm_budget", body),
  systemCacheInvalidate: (key?: string) =>
    post<{ invalidated: string; n_dropped: number }>(
      `/api/system/cache/invalidate${key ? `?key=${encodeURIComponent(key)}` : ""}`,
      undefined,
    ),
  dq: () => get<DQReport>("/api/dq"),
  tracking: () => get<LiveTracking>("/api/book/tracking"),
  execution: () => get<ExecutionReconcile>("/api/execution"),
  riskContrib: () => get<RiskContribData>("/api/book/risk-contrib"),
  scenarios: () => get<ScenarioData>("/api/book/scenarios"),
  factorExposure: () => get<FactorExposure>("/api/book/factor-exposure"),
  bookPerf: () => get<BookPerf>("/api/book/perf"),
  bookTrades: (limit = 100) => get<Blotter>(`/api/book/trades?limit=${limit}`),
  brief: () => get<DailyBrief>("/api/brief"),
  graveyard: () => get<Graveyard>("/api/research/graveyard"),
  pitAudit: () => get<PitAudit>("/api/research/pit-audit"),

  // Phase 1.2 / 4.1 / B surfaces — added 2026-06-14
  postGreenRigorRecent: (days = 7, limit = 50) =>
    get<PostGreenRigorResponse>(
      `/api/research/post_green_rigor/recent?days=${days}&limit=${limit}`),
  externalAuditsRecent: (days = 7, limit = 50) =>
    get<ExternalAuditsResponse>(
      `/api/research/external_audits/recent?days=${days}&limit=${limit}`),
  beliefFamilies: (minObs = 3) =>
    get<BeliefFamiliesResponse>(
      `/api/research/belief/families?min_obs=${minObs}`),
  beliefCalibration: () =>
    get<BeliefCalibrationResponse>(`/api/research/belief/calibration`),
  workflowCounts: () =>
    get<WorkflowCountsResponse>(`/api/research/workflow/counts`),
  // Operator Console
  consoleStations: () =>
    get<ConsoleStationsResponse>(`/api/console/stations`),
  consoleStation: (stationId: string) =>
    get<{ spec: ConsoleStationSpec; config_form: Record<string, unknown> }>(
      `/api/console/stations/${encodeURIComponent(stationId)}`,
    ),
  consolePreflight: (req: { station_id: string; session_id: string; config: Record<string, unknown> }) =>
    post<ConsolePreflightResponse>(`/api/console/preflight`, req),
  consoleTrigger: (req: { station_id: string; session_id: string; config: Record<string, unknown> }) =>
    post<ConsoleTriggerResponse>(`/api/console/trigger`, req),
  consoleJobStatus: (jobId: string) =>
    get<ConsoleJobStatus>(`/api/console/status/${encodeURIComponent(jobId)}`),
  consoleCancelJob: (jobId: string) =>
    post<ConsoleTriggerResponse>(`/api/console/cancel/${encodeURIComponent(jobId)}`, {}),
  consoleCostStatus: (sessionId: string, capUsd = 1.0) =>
    get<ConsoleCostStatusResponse>(
      `/api/console/cost_status?session_id=${encodeURIComponent(sessionId)}&cap_usd=${capUsd}`,
    ),
  consoleJobsList: (params: { session_id?: string; state?: ConsoleJobState; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.session_id) q.set("session_id", params.session_id);
    if (params.state)      q.set("state",      params.state);
    if (params.limit)      q.set("limit",      String(params.limit));
    return get<{ jobs: ConsoleJobStatus[]; n: number }>(`/api/console/jobs?${q.toString()}`);
  },
  safetyRailsForHypothesis: (hypothesisId: string) =>
    get<SafetyRailsForHypothesis>(
      `/api/research/safety_rails_for_hypothesis/${encodeURIComponent(hypothesisId)}`),
  graveyardCollisions: (hypothesisId: string, topK = 5, minScore = 0.2) =>
    get<GraveyardCollisions>(
      `/api/research/graveyard_collisions/${encodeURIComponent(hypothesisId)}?top_k=${topK}&min_score=${minScore}`),
  discoveryQueues: (limit = 20) =>
    get<DiscoveryQueues>(`/api/research/discovery/queues?limit=${limit}`),
  discoveryNominate: (url: string) =>
    post<NominateResult>("/api/research/discovery/nominate", { url }),
  discoveryBookmarklet: () =>
    get<DiscoveryBookmarklet>("/api/research/discovery/bookmarklet"),
  discoveryPromote: (sourceId: string, targetStatus?: string) =>
    post<PromoteResult>("/api/research/discovery/promote",
      { source_id: sourceId, target_status: targetStatus ?? null }),
  discoverySkip: (sourceId: string, reason?: string) =>
    post<SkipResult>("/api/research/discovery/skip",
      { source_id: sourceId, reason: reason ?? null }),
  discoveryWatchlist: () =>
    get<WatchlistPayload>("/api/research/discovery/watchlist"),
  agents: () => get<AgentsDirectory>("/api/agents"),
  health: () => get<{ status: string }>("/health"),

  // Phase 4a.6 — shared research-tools REST shim (Cockpit + Assistant).
  // Same 9 tools the MCP server (engine.research.mcp_server) exposes
  // to Claude Code desktop; same audit ledger (ui_tool_calls.jsonl).
  researchTools: () =>
    get<{ n_tools: number; tools: { name: string; description: string; input_schema: any }[] }>(
      "/api/research/tools"),
  researchCall: <T = any>(
    toolName: string, args: Record<string, any>, caller?: string,
  ): Promise<{ tool: string; ok: boolean; result: T; result_hash: string; latency_ms: number }> => {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (caller) headers["X-Research-Caller"] = caller;
    return fetch(`${API_BASE}/api/research/call/${toolName}`, {
      method: "POST", headers, body: JSON.stringify({ args }),
    }).then(async (r) => {
      if (!r.ok) {
        const detail = await r.json().catch(() => ({}));
        throw new Error(`${r.status} researchCall(${toolName}): ${detail?.detail ?? r.statusText}`);
      }
      return r.json();
    });
  },
  researchAudit: (limit = 50, tool?: string, caller?: string) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (tool) qs.set("tool", tool);
    if (caller) qs.set("caller", caller);
    return get<{ n: number; entries: any[] }>(`/api/research/audit?${qs}`);
  },

  // Phase 4b.5 — 3-agent council runs ledger + trigger
  councilRuns: (limit = 50, consensus?: string) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (consensus) qs.set("consensus", consensus);
    return get<{ n: number; runs: any[] }>(`/api/research/council/runs?${qs}`);
  },
  // (councilRunDetail moved below — fully typed version)
  councilTrigger: (seedIdea: string, candidateReturnsPath?: string) =>
    post<any>("/api/research/council/run",
      { seed_idea: seedIdea, confirm_cost: true,
        candidate_returns_path: candidateReturnsPath ?? null }),

  // Phase 4e — human-in-loop signals on a running workflow
  councilPause: (workflowId: string) =>
    post<{ ok: boolean; signaled_at: string }>(
      `/api/research/council/workflow/${workflowId}/pause`, {}),
  councilResume: (workflowId: string) =>
    post<{ ok: boolean; signaled_at: string }>(
      `/api/research/council/workflow/${workflowId}/resume`, {}),
  councilOverride: (workflowId: string, verdict: string, justification: string) =>
    post<{ ok: boolean; verdict: string; signaled_at: string }>(
      `/api/research/council/workflow/${workflowId}/override`,
      { verdict, justification }),
  l4Promote: (iterationId: string, justification: string) =>
    post<{ draft_path: string; draft_id: string; checklist: string[] }>(
      "/api/research/l4/promote",
      { iteration_id: iterationId, justification }),

  // Step A — cached parquet inventory for Lab Series page
  parquets: (includeInternal = true, limit = 200) =>
    get<{
      n: number;
      cache_dir: string;
      parquets: Array<{
        filename:    string;
        relpath:     string;
        size_bytes:  number;
        mtime:       string;
        n_rows:      number | null;
        n_cols:      number | null;
        columns:     string[];
        date_start:  string | null;
        date_end:    string | null;
        is_internal: boolean;
        error:       string | null;
      }>;
    }>(`/api/research/parquets?include_internal=${includeInternal}&limit=${limit}`),

  // Phase 5.7 follow-up — sleeve CA filter calibration status
  sleevesCaCalibration: () =>
    get<{
      n: number;
      sleeves: Array<{
        id: string;
        status: "DEPLOYED" | "PENDING_DEPLOY";
        family: string | null;
        ca_filter_k: number | null;
        ca_filter_k_method: "paper_default" | "pbb_sweep_calibrated" | "scalar_override" | null;
        ca_filter_k_audit_date: string | null;
        ca_signal_type: string | null;
        tcost_round_trip_bps: number | null;
      }>;
    }>("/api/research/sleeves/ca_calibration"),

  // Phase 4f — trace timeline reader
  traces: (workflowId?: string, traceId?: string, limit = 500) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (workflowId) qs.set("workflow_id", workflowId);
    if (traceId)    qs.set("trace_id", traceId);
    return get<{
      n: number;
      spans: Array<{
        trace_id: string;
        span_id: string;
        parent_id: string | null;
        name?: string;
        kind?: string;  // "attr_update" markers
        start_ms?: number;
        end_ms?: number;
        ts_ms?: number;
        duration_ms?: number;
        ok?: boolean;
        error?: string | null;
        attrs?: Record<string, any>;
      }>;
    }>(`/api/research/traces?${qs}`);
  },

  // Phase 4d — L4 discovery loop iteration ledger
  l4Iterations: (limit = 50, consensus?: string, alignment?: string) => {
    const qs = new URLSearchParams({ limit: String(limit) });
    if (consensus) qs.set("consensus", consensus);
    if (alignment) qs.set("alignment", alignment);
    return get<{
      n: number;
      iterations: any[];
      calibration: {
        n_total: number;
        n_runnable: number;
        n_agree: number;
        agree_pct: number | null;
        by_alignment: Record<string, number>;
      };
    }>(`/api/research/l4/iterations?${qs}`);
  },
  // (l4IterationDetail moved below — fully typed version)

  // Phase 4d.5 — L1 candidate seed recommender
  councilSuggestions: (limit = 10) =>
    get<{
      n_total: number;
      by_source: { library: number; seed_pool: number };
      suggestions: Array<{
        title: string;
        family: string;
        seed: string;
        rationale: string;
        proposed_role: string;
        parent_family: string;
        risk_tag: "low" | "medium" | "high";
        source: "library" | "seed_pool";
        anchor_paper: string | null;
        score: number;
        score_components: Record<string, number>;
      }>;
    }>(`/api/research/council/suggestions?limit=${limit}`),

  // /research/library/[mechanism_id] detail
  libraryMechanismDetail: (mechanismId: string) =>
    get<{
      mechanism_id: string;
      yaml: Record<string, any>;
      filename: string;
      decay_history: Array<{
        sleeve: string;
        library_id: string;
        audit_date: string;
        trailing_sharpe: number | null;
        alert_level: string;
        recommendation: string;
      }>;
      graveyard_cousins: Array<{
        name: string;
        family: string;
        date: string;
        verdict: string;
        why: string;
      }>;
    }>(`/api/research/library/${encodeURIComponent(mechanismId)}`),

  // /research/decay/[sleeve] timeline
  decaySleeveTimeline: (sleeve: string) =>
    get<{
      sleeve: string;
      rows: Array<{
        sleeve: string;
        library_id: string;
        audit_date: string;
        trailing_sharpe: number | null;
        baseline_sharpe: number | null;
        ratio: number | null;
        consecutive_below_threshold: number;
        alert_level: string;
        recommendation: string;
      }>;
      n_audits: number;
      n_alerts: number;
      library_id: string | null;
      first_audit: string | null;
      last_audit: string | null;
      sharpe_min: number | null;
      sharpe_max: number | null;
      sharpe_last: number | null;
    }>(`/api/research/decay/sleeve/${encodeURIComponent(sleeve)}`),

  // G.4 (2026-06-09): canonical Tier C decay audit drill-down.
  // Returns decay_watch events from the research_store for one subject,
  // plus all factor_verdict_filed events carrying lens outputs
  // (subsample_stability / specification_robustness / anchor_orthogonality).
  // Used by /research/decay/detail to render the "Tier C decay watch" panel
  // alongside the legacy SLM timeline.
  decayAuditCanonical: (subject_id: string) =>
    get<{
      subject_id: string;
      n_decay_alerts: number;
      decay_alerts: Array<{
        event_id: string;
        ts: string;
        event_type: string;
        subject_id: string;
        verdict: string;
        summary: string;
        tags: string[];
        metrics: Record<string, any>;
        parent_event_ids: string[];
        actor: string;
        // G.5 + H (2026-06-09): server-computed ack/unack state
        is_ack_event?: boolean;
        is_unack_event?: boolean;
        is_admin_event?: boolean;
        ack_info?: {
          is_acknowledged: boolean;
          latest_event_id: string;
          latest_ts:       string;
          latest_action:   string | null;
          latest_reason:   string | null;
          latest_actor:    string | null;
          history: Array<{
            event_id: string;
            ts:       string;
            kind:     "acknowledged" | "unacknowledged";
            action:   string | null;
            reason:   string | null;
            actor:    string | null;
          }>;
        } | null;
      }>;
      n_factor_verdicts: number;
      factor_verdicts: Array<{
        event_id: string;
        ts: string;
        subject_id: string;
        verdict: string;
        summary: string;
        metrics: Record<string, any>;
        tags: string[];
      }>;
    }>(`/api/research_store/decay_audit/${encodeURIComponent(subject_id)}`),

  // G.5 (2026-06-09): acknowledge a canonical decay_alert event.
  // Emits a follow-up canonical event with parent_event_ids pointing
  // to the original — per the immutable-events doctrine. Body fields:
  //   action: one of "reviewed_no_action" | "reduced_allocation" |
  //                  "scheduled_review" | "false_positive"
  //   reason: required, ≥10 chars
  //   actor:  optional, defaults to "ui"
  ackDecayAlert: (event_id: string, body: {
    action: string;
    reason: string;
    actor?: string;
  }) =>
    post<{
      ok: boolean;
      ack_event_id: string;
      original_event_id: string;
    }>(`/api/research_store/decay_alert/${encodeURIComponent(event_id)}/acknowledge`, body),

  // L+M (2026-06-09): factor_verdict event drill-down. Used by
  // /research/verdict/[event_id] to render the full lens output
  // stack — surfaces the click target for G.2 (overfit) + G.3
  // (anchor-spanned) inbox rows.
  verdictDetail: (event_id: string) =>
    get<{
      event: {
        event_id: string;
        ts: string;
        event_type: string;
        subject_id: string;
        verdict: string;
        family: string | null;
        summary: string;
        actor: string;
        tags: string[];
        parent_event_ids: string[];
        metrics: Record<string, any>;
        artifacts: Record<string, any>;
      };
    }>(`/api/research_store/verdict/${encodeURIComponent(event_id)}`),

  // H (2026-06-09): re-open an acked decay_alert. event_id is the
  // ORIGINAL alert id; the endpoint walks the chain to find the
  // latest ack and unacks from there. Returns 422 if the alert
  // isn't currently acked (nothing to undo).
  unackDecayAlert: (event_id: string, body: {
    reason: string;
    actor?: string;
  }) =>
    post<{
      ok: boolean;
      unack_event_id: string;
      original_event_id: string;
      reverted_ack_event_id: string;
    }>(`/api/research_store/decay_alert/${encodeURIComponent(event_id)}/unacknowledge`, body),

  // Path C UI (2026-06-01): /research/library page
  libraryInventory: () =>
    get<{
      n: number;
      entries: Array<{
        id: string;
        family: string | null;
        parent_family: string | null;
        purpose: string | null;
        canonical_paper_id: string | null;
        ca_filter_k_method: string | null;
        audit_date: string | null;
        schema_version: number | null;
        filename: string;
        error?: string;
      }>;
    }>("/api/research/library/inventory"),

  // Path C UI: /research/decay page
  decayHistory: (limit = 200, sleeve?: string) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (sleeve) q.append("sleeve", sleeve);
    return get<{
      n: number;
      rows: Array<{
        sleeve: string;
        library_id: string;
        audit_date: string;
        trailing_sharpe: number | null;
        baseline_sharpe: number | null;
        ratio: number | null;
        consecutive_below_threshold: number;
        alert_level: string;
        recommendation: string;
      }>;
    }>(`/api/research/decay/history?${q.toString()}`);
  },

  // /lab/council/[run_id] — full council run detail
  councilRunDetail: (runId: string) =>
    get<{
      run_id: string;
      ts: string;
      stage?: string;
      consensus: string;
      rationale: string;
      proposal: {
        title: string;
        family: string;
        parent_family: string;
        proposed_role: string;
        economics_text: string;
        required_data: string[];
        motivation: string;
      };
      verdicts: Array<{
        agent_name: string;
        verdict: string;
        confidence: number;
        rationale: string;
        fatal_red_flags: string[];
        material_concerns: string[];
        tool_calls: Array<{
          tool_name: string;
          args: Record<string, any>;
          result_summary: string;
          elapsed_ms: number;
        }>;
        elapsed_s: number;
        round_1_verdict?: string | null;
        round_1_confidence?: number | null;
        reflection_action?: string | null;
      }>;
      elapsed_s: number;
      n_critics: number;
      n_tool_calls_total: number;
      reflection_enabled?: boolean;
      round_1_consensus?: string | null;
      round_1_rationale?: string | null;
      reflection_actions?: string[];
    }>(`/api/research/council/run/${runId}`),

  // Path C UI: /lab/council page
  councilRunsList: (limit = 50, consensus?: string) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (consensus) q.append("consensus", consensus);
    return get<{
      n: number;
      runs: Array<{
        run_id: string;
        ts: string;
        consensus?: string;
        proposal?: { title?: string; family?: string };
        elapsed_s?: number;
        n_critics?: number;
        n_tool_calls_total?: number;
        reflection_enabled?: boolean;
      }>;
    }>(`/api/research/council/runs?${q.toString()}`);
  },

  // /lab/l4 — Temporal workflow + cron status
  l4CronStatus: () =>
    get<{
      schedule: {
        exists: boolean;
        paused?: boolean;
        cron?: string[];
        next_run?: string | null;
        running?: boolean;
        n_recent_actions?: number;
        error?: string;
      };
      recent_runs: Array<{
        id: string;
        ts: string;
        seed?: string;
        title?: string;
        family?: string;
        source?: string;
        child_workflow_id?: string;
        skipped_reason?: string;
      }>;
    }>("/api/research/l4/cron/status"),

  // /lab/chains — chain catalogue + recent runs
  chainsCatalogue: () =>
    get<{
      chains: Array<{
        chain_id: string;
        description: string;
        n_steps: number;
        step_names: string[];
      }>;
    }>("/api/research/chains"),

  chainsRuns: (chainId?: string, limit = 50) => {
    const q = new URLSearchParams({ limit: String(limit) });
    if (chainId) q.append("chain_id", chainId);
    return get<{
      n: number;
      runs: Array<{
        chain_id: string;
        run_id: string;
        started_at: string;
        finished_at: string | null;
        status: string;
        elapsed_s: number;
        steps: Array<{
          name: string;
          status: string;
          tool: string;
          elapsed_ms: number;
          error?: string | null;
        }>;
        initial_context: Record<string, any>;
      }>;
    }>(`/api/research/chains/runs?${q.toString()}`);
  },

  // UI v2 (2026-06-01): /lab/factor-lab
  factorLabCatalog: () =>
    get<{
      universes:  string[];
      signals:    string[];
      weightings: string[];
      n_possible: number;
      n_untested: number;
      tested_tuples:   [string, string, string][];
      untested_tuples: [string, string, string][];
      labels_summary: {
        n_green: number;
        n_yellow: number;
        n_red: number;
        n_total: number;
        p_green: number | null;
      };
    }>("/api/research/factor_lab/catalog"),

  factorLabAxesDetails: () =>
    get<{
      universes:  Array<Record<string, any>>;
      signals:    Array<Record<string, any>>;
      weightings: Array<Record<string, any>>;
    }>("/api/research/factor_lab/axes/details"),

  factorLabPfhSuggest: (
    k = 6, mode: "open" | "constrained" = "constrained",
    maxPerFamily = 2, maxPerUniverse = 2,
  ) =>
    post<{
      run_id: string;
      ts: string;
      mode: string;
      base_rate_used: number;
      n_candidates_total: number;
      n_scored: number;
      k_requested: number;
      top: Array<{
        proposal: {
          candidate_id: string;
          proposal_kind: string;
          family_normalized: string;
          universe: string | null;
          signal_recipe: string | null;
          weighting: string | null;
          rebalance: string;
          cousin_warnings: string[];
          needs_new_axes: string[];
        };
        posterior: {
          posterior_mean: number;
          credible_05: number;
          credible_50: number;
          credible_95: number;
        };
        cousin_penalty: number;
        final_score: number;
        score_breakdown: Record<string, any>;
      }>;
      written_spec_paths: string[];
    }>("/api/research/factor_lab/pfh/suggest", {
      k, mode,
      max_per_family: maxPerFamily,
      max_per_universe: maxPerUniverse,
      write_specs: true,
      prior_strength: 4.0,
    }),

  factorLabMaterialize: (specId: string, force = false) =>
    post<{
      ok: boolean;
      result?: {
        spec_id: string;
        spec_kind: string;
        input_hash: string;
        cached: boolean;
        elapsed_s: number;
        validation: {
          ok: boolean;
          violations: string[];
          observed_n_rows: number | null;
          observed_start: string | null;
          observed_end: string | null;
          observed_ann_vol: number | null;
          observed_ann_sharpe: number | null;
        };
      };
      error?: string;
    }>("/api/research/factor_lab/materialize", { spec_id: specId, force }),

  // /lab/factor-lab/detail spec drill-down
  factorLabSpecDetail: (specId: string) =>
    get<{
      spec_id: string;
      spec_kind: "compose" | "function";
      yaml: Record<string, any>;
      axes_detail: {
        universe?:  Record<string, any>;
        signal?:    Record<string, any>;
        weighting?: Record<string, any>;
      };
      materializations: Array<{
        meta_filename: string;
        parquet_filename: string;
        input_hash: string;
        materialized_at: string;
        elapsed_s: number;
        spec_kind: string;
        validation: {
          ok: boolean;
          violations: string[];
          observed_n_rows: number | null;
          observed_start: string | null;
          observed_end: string | null;
          observed_ann_vol: number | null;
          observed_ann_sharpe: number | null;
        };
        compose_axes?: Record<string, string> | null;
      }>;
      posterior_context: {
        family: string;
        n_green: number;
        n_yellow: number;
        n_red: number;
        posterior_mean: number;
        credible_05: number;
        credible_95: number;
        base_rate_used: number;
      } | null;
      filename: string;
    }>(`/api/research/factor_lab/spec/${encodeURIComponent(specId)}`),

  // /lab/l4/detail iteration drill-down (existing backend endpoint)
  l4IterationDetail: (iterationId: string) =>
    get<{
      ts: string;
      iteration_id: string;
      workflow_id: string;
      stage: string;
      proposal: {
        title: string;
        family: string;
        proposed_role: string;
      };
      council: {
        consensus: string;
        rationale: string;
        n_critics: number;
        run_id: string;
      };
      human_override: { verdict: string } | null;
      effective_consensus: string;
      pipeline: {
        ran: boolean;
        final_decision?: string;
        rationale: string;
        n_steps: number;
        candidate_returns_path?: string;
      } | null;
      verdict_alignment: string;
      elapsed_s: number;
    }>(`/api/research/l4/iterations/${encodeURIComponent(iterationId)}`),

  // Chat Phase 2 ledger
  chatLogTurn: (commandId: string, command: string, kind: string,
                ok: boolean, summary?: string) =>
    post<{ ok: boolean; ts?: string; error?: string }>(
      "/api/research/chat/log_turn",
      { command_id: commandId, command, kind, ok, summary }),

  chatRecent: (limit = 50) =>
    get<{
      n: number;
      turns: Array<{
        ts: string; command_id: string; command: string; kind: string;
        ok: boolean; summary?: string; caller: string;
      }>;
    }>(`/api/research/chat/recent?limit=${limit}`),

  // P0b 2026-06-02 — liveness watcher surface
  livenessStatus: (limit = 14) =>
    get<LivenessStatus>(`/api/research/liveness/status?limit=${limit}`),

  // Chat Phase 3 — /ask scoped LLM with RAG (+ 2026-06-02 sessions)
  // Commit Z 2026-06-04 — citation.exists + verification counts added.
  // Commit Y 2026-06-04 — pageContext lifted into the system prompt.
  chatAsk: (question: string, sessionId?: string, pageContext?: string) =>
    post<{
      answer: string;
      citations: Array<{ type: string; id: string; exists?: boolean }>;
      verification?: {
        n_cited:           number;
        n_resolved:        number;
        n_unverified:      number;
        n_self_unverified: number;
      };
      n_context_rows: Record<string, number>;
      retrieval_mode?: string;
      model: string;
      elapsed_s: number;
      question: string;
      session_id: string;
      n_prior_turns: number;
    }>("/api/research/chat/ask", {
      question, confirm_cost: true,
      session_id:   sessionId    ?? null,
      page_context: pageContext  ?? null,
    }),

  chatSessionNew: () =>
    post<{ session_id: string }>("/api/research/chat/session/new", {}),

  chatSessionGet: (sessionId: string) =>
    get<{
      session_id: string;
      n_turns: number;
      title?: string | null;
      turns: Array<{
        ts: string;
        question: string;
        answer: string;
        citations: Array<{ type: string; id: string }>;
        retrieval_mode?: string;
        elapsed_s?: number;
      }>;
    }>(`/api/research/chat/session/${encodeURIComponent(sessionId)}`),

  chatSessionsList: (limit = 40) =>
    get<{
      n: number;
      sessions: Array<{
        session_id:     string;
        n_turns:        number;
        first_question: string | null;
        title:          string | null;
        last_ts:        string | null;
      }>;
    }>(`/api/research/chat/sessions?limit=${limit}`),

  chatSessionDelete: (sessionId: string) =>
    fetch(`${API_BASE}/api/research/chat/session/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }).then((r) => r.json() as Promise<{ ok: boolean; existed: boolean }>),

  factorLabPfhHistory: (limit = 20) =>
    get<{
      n: number;
      runs: Array<{
        run_id: string;
        ts: string;
        mode: string;
        n_candidates_total: number;
        base_rate_used: number;
      }>;
    }>(`/api/research/factor_lab/pfh/history?limit=${limit}`),

  // Path C UI: /lab/outcomes — critic calibration report
  criticCalibration: (sinceDays = 90) =>
    get<{
      since_days: number;
      n_total_rows: number;
      n_distinct_critics: number;
      per_critic: Record<string, {
        accuracy: {
          critic_name: string;
          n_total: number;
          n_decided: number;
          accuracy: number | null;
          by_alignment: Record<string, number>;
        };
        marginal_info: {
          critic_name: string;
          full_council_accuracy: number | null;
          without_critic_accuracy: number | null;
          marginal_information_gain: number | null;
          interpretation: string;
        };
      }>;
      pairwise_agreement: {
        pairs: Array<{
          pair: [string, string];
          n_iterations: number;
          verdict_agreement_pct: number | null;
          alignment_agreement_pct: number | null;
        }>;
      };
    }>(`/api/research/critic/calibration?since_days=${sinceDays}`),

  // ── Research event store (M3 2026-06-02) ────────────────────────
  //
  // Typed event store backing /research/library, /research graveyard,
  // /research/decay, Cockpit. Replaces ad-hoc scraping of capability_evidence/,
  // memory/, factory_ledger.jsonl, gate_runs.jsonl. See CLAUDE.md
  // "Research Event Emission Doctrine".

  researchStoreEvents: (params: {
    event_type?:   string;
    subject_type?: string;
    subject_id?:   string;
    verdict?:      string;
    family?:       string;
    since?:        string;
    limit?:        number;
  } = {}) => {
    const q = new URLSearchParams();
    if (params.event_type)   q.set("event_type",   params.event_type);
    if (params.subject_type) q.set("subject_type", params.subject_type);
    if (params.subject_id)   q.set("subject_id",   params.subject_id);
    if (params.verdict)      q.set("verdict",      params.verdict);
    if (params.family)       q.set("family",       params.family);
    if (params.since)        q.set("since",        params.since);
    if (params.limit != null) q.set("limit",       String(params.limit));
    const qs = q.toString();
    return get<{
      n: number;
      events: Array<ResearchStoreEvent>;
    }>(`/api/research_store/events${qs ? `?${qs}` : ""}`);
  },

  researchStoreSubjects: (family?: string) => {
    const qs = family ? `?family=${encodeURIComponent(family)}` : "";
    return get<{
      n: number;
      subjects: Array<{
        subject_id:          string;
        subject_type:        string;
        family:              string | null;
        description:         string;
        canonical_paper_id:  string | null;
        created_ts:          string;
        created_by:          string;
      }>;
    }>(`/api/research_store/subjects${qs}`);
  },

  researchStoreLineage: (event_id: string) =>
    get<{
      n: number;
      chain: Array<ResearchStoreEvent>;
    }>(`/api/research_store/lineage/${encodeURIComponent(event_id)}`),

  researchStoreSummary: () =>
    get<{
      n_total: number;
      by_event_type: Record<string, number>;
      by_verdict:    Record<string, number>;
      by_family:     Record<string, number>;
      first_ts:  string | null;
      latest_ts: string | null;
    }>("/api/research_store/summary"),

  // ── Sessions (typed user-initiated workflow protocol — P4 2026-06-03) ─

  sessionOpen: (req: { session_type: SessionType; title: string }) =>
    post<SessionRow>("/api/sessions/open", req),

  sessionPreflight: (sessionId: string, digest: PreflightDigestInput) =>
    post<SessionRow>(`/api/sessions/${encodeURIComponent(sessionId)}/preflight`, digest),

  sessionClose: (sessionId: string) =>
    post<SessionRow>(`/api/sessions/${encodeURIComponent(sessionId)}/close`, {}),

  sessionAbandon: (sessionId: string, reason: string = "") =>
    post<SessionRow>(`/api/sessions/${encodeURIComponent(sessionId)}/abandon`, { reason }),

  sessionActive: () =>
    get<{
      active:  { session_id: string; session_type: SessionType } | null;
      session: SessionRow | null;
      phase?:  SessionPhaseInfo | null;
    }>("/api/sessions/active"),

  sessionsList: (params: { limit?: number; state?: SessionState; session_type?: SessionType } = {}) => {
    const q = new URLSearchParams();
    if (params.limit != null)    q.set("limit",        String(params.limit));
    if (params.state)            q.set("state",        params.state);
    if (params.session_type)     q.set("session_type", params.session_type);
    const qs = q.toString();
    return get<{
      n: number;
      sessions: SessionRow[];
    }>(`/api/sessions${qs ? `?${qs}` : ""}`);
  },

  sessionGet: (sessionId: string) =>
    get<{
      session: SessionRow;
      events: ResearchStoreEvent[];
      n_events: number;
    }>(`/api/sessions/${encodeURIComponent(sessionId)}`),

  sessionTypes: () =>
    get<{
      types: Array<{
        session_type:      SessionType;
        description:       string;
        expected_duration: string;
      }>;
    }>("/api/sessions/types"),

  // ── Forward decay forecast (Gap B 2026-06-03) ────────────────────

  decayFamilies: () =>
    get<{
      families: Array<{
        family:           string;
        mp_2016_lambda:   number;
        lr_2018_lambda:   number;
        half_life_years:  number | null;
        notes:            string;
      }>;
    }>("/api/decay_forecast/families"),

  decayEstimate: (params: {
    family:            string;
    baseline_alpha?:   number;
    publication_year?: number;
  }) => {
    const q = new URLSearchParams();
    q.set("family", params.family);
    if (params.baseline_alpha   != null) q.set("baseline_alpha",   String(params.baseline_alpha));
    if (params.publication_year != null) q.set("publication_year", String(params.publication_year));
    return get<DecayEstimateResp>(`/api/decay_forecast/estimate?${q.toString()}`);
  },

  // ── Roadmap (Gap A 2026-06-03) ───────────────────────────────────

  roadmapList: (params: { state?: string; tier?: string; family?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.state)  q.set("state",  params.state);
    if (params.tier)   q.set("tier",   params.tier);
    if (params.family) q.set("family", params.family);
    const qs = q.toString();
    return get<{ n: number; axes: ResearchAxisRow[] }>(
      `/api/roadmap/axes${qs ? `?${qs}` : ""}`,
    );
  },

  roadmapGet: (axis_id: string) =>
    get<ResearchAxisRow>(`/api/roadmap/axes/${encodeURIComponent(axis_id)}`),

  roadmapUpsert: (axis: ResearchAxisUpsertInput) =>
    post<ResearchAxisRow>("/api/roadmap/axes", axis),

  // ── Capacity (Gap C 2026-06-03) ──────────────────────────────────

  capacityFamilies: () =>
    get<{
      families: Array<{
        family:                  string;
        capacity_class:          CapacityClass;
        estimated_capacity_usd:  number;
        comfortable_aum_usd:     number;
        notes:                   string;
      }>;
    }>("/api/capacity/families"),

  capacityEstimate: (family: string) =>
    get<CapacityEstimateResp>(
      `/api/capacity/estimate?family=${encodeURIComponent(family)}`,
    ),
};

// ── Capacity types ────────────────────────────────────────────────

export type CapacityClass =
  | "VERY_HIGH" | "HIGH" | "MEDIUM" | "LOW" | "VERY_LOW";

export interface CapacityEstimateResp {
  family:                   string;
  using_default:            boolean;
  capacity_class:           CapacityClass;
  estimated_capacity_usd:   number;
  comfortable_aum_usd:      number;
  minimum_aum_usd:          number;
  notes:                    string;
  methodology:              string;
  parent_family:            string | null;
}

// ── Roadmap types ─────────────────────────────────────────────────

export type AxisState = "active" | "queued" | "paused" | "closed";
export type AxisTier  = "committed" | "candidate" | "scratchpad";
export type AxisOutcome = "GREEN" | "RED" | "MARGINAL" | "ABANDONED" | "NONE";

export interface ResearchAxisRow {
  axis_id:              string;
  name:                 string;
  state:                AxisState;
  tier:                 AxisTier;
  outcome:              AxisOutcome;
  parent_axis_id:       string | null;
  family:               string | null;
  related_subject_ids:  string[];
  related_memory_files: string[];
  rationale:            string;
  next_actions:         string[];
  blocking_notes:       string;
  decay_estimate:       DecayEstimateResp | null;
  capacity_estimate:    CapacityEstimateResp | null;
  created_ts:           string;
  updated_ts:           string;
  created_by:           string;
  updated_by:           string;
  schema_version:       number;
}

export interface ResearchAxisUpsertInput {
  axis_id:               string;
  name:                  string;
  state:                 AxisState;
  tier:                  AxisTier;
  rationale:             string;
  outcome?:              AxisOutcome;
  parent_axis_id?:       string | null;
  family?:               string | null;
  related_subject_ids?:  string[];
  related_memory_files?: string[];
  next_actions?:         string[];
  blocking_notes?:       string;
}

// ── Decay forecast types ──────────────────────────────────────────

export type DecayRisk = "LOW" | "MEDIUM" | "HIGH" | "SEVERE";

export interface DecayEstimateResp {
  family:                   string;
  parent_family:            string | null;
  using_default:            boolean;
  baseline_alpha:           number;
  publication_year:         number | null;
  years_since_pub:          number;
  mp_2016_lambda:           number;
  lr_2018_lambda:           number;
  half_life_years:          number;
  expected_alpha_now:       number;
  expected_alpha_5y:        number;
  expected_alpha_10y:       number;
  expected_alpha_5y_lower:  number;
  expected_alpha_10y_lower: number;
  risk:                     DecayRisk;
  note:                     string;
}

// ── Sessions types ────────────────────────────────────────────────

export type SessionType =
  | "research_new" | "audit" | "ops" | "doctrine" | "exploration";

export type SessionState =
  | "pending_preflight" | "in_flight" | "closed" | "abandoned";

export interface PreflightDigestInput {
  cockpit_reviewed?:        boolean;
  decay_alerts_count?:      number;
  dq_breaches_count?:       number;
  graveyard_search_query?:  string;
  graveyard_hits_count?:    number;
  library_overlap_checked?: boolean;
  goal:                     string;  // always required
  notes?:                   string;
}

export interface SessionExitReport {
  exit_satisfied:       boolean;
  missing_requirements: string[];
  emitted_event_ids:    string[];
  git_commits:          string[];
  closed_ts:            string;
}

export type SessionPhase =
  | "pending_preflight" | "awaiting_claude" | "claude_working"
  | "awaiting_close" | "closed" | "abandoned";

export interface SessionPhaseInfo {
  phase:             SessionPhase;
  next_action_label: string;
  next_action_kind:  "copy_brief" | "wait" | "close" | "none";
  last_activity_ts:  string | null;
  n_events:          number;
  n_commits:         number;
}

export interface SessionRow {
  session_id:       string;
  session_type:     SessionType;
  state:            SessionState;
  opened_ts:        string;
  preflight_ts:     string | null;
  closed_ts:        string | null;
  preflight_digest: PreflightDigestInput | null;
  exit_report:      SessionExitReport | null;
  title:            string;
  actor:            string;
  schema_version:   number;
}

// ── Research event store types ───────────────────────────────────

export type ResearchStoreEventType =
  | "factor_verdict_filed"
  | "memory_doctrine_locked"
  | "spec_amended"
  | "deploy_changed"
  | "decay_alert"
  | "dq_breach"
  | "council_critique"
  | "capability_evidence_filed";

export type ResearchStoreVerdict = "GREEN" | "MARGINAL" | "RED" | "NEUTRAL";

export type ResearchStoreSubjectType =
  | "factor" | "sleeve" | "spec" | "memory_doctrine"
  | "data_quality" | "capacity" | "book";

export interface ResearchStoreEvent {
  event_id:          string;
  event_type:        ResearchStoreEventType;
  ts:                string;
  session_id:        string;
  actor:             string;
  subject_type:      ResearchStoreSubjectType;
  subject_id:        string;
  verdict:           ResearchStoreVerdict;
  metrics:           Record<string, unknown>;
  artifacts:         Record<string, string>;
  parent_event_ids:  string[];
  family:            string | null;
  tags:              string[];
  summary:           string;
  git_sha:           string;
  schema_version:    number;
}

// ── Phase 1.2 / 4.1 / B surfaces (rigor / audit / belief) ─────────
// Backend endpoints added 2026-06-14 per UI-architecture audit.
// Safety rails were invisible to /dashboard before this — every chip
// here represents a backend gate that fires but was unsurfaced.

export interface PostGreenRigorRow {
  rigor_id:         string;
  ts:               string;
  verdict_event_id: string;
  hypothesis_id:    string;
  family:           string;
  template_name:    string;
  original_verdict: string;
  oos_status:       string | null;
  oos_verdict:      string | null;
  spanning_status:  string | null;
  spanning_alpha_t: number | null;
  borrow_status:    string | null;
  flags:            string[];
}

export interface PostGreenRigorResponse {
  n:         number;
  n_flagged: number;
  rows:      PostGreenRigorRow[];
}


export type AuditSeverity = "skipped" | "ok" | "concern" | "critical";

export interface ExternalAuditRow {
  audit_id:           string;
  ts:                 string;
  audit_subject:      string;
  subject_ref:        string;
  provider:           string;
  severity:           AuditSeverity;
  flagged_categories: string[];
  cost_estimate_usd:  number | null;
}

export interface ExternalAuditsResponse {
  n:          number;
  n_concern:  number;
  n_critical: number;
  rows:       ExternalAuditRow[];
}


export interface BeliefFamily {
  family:         string;
  n_obs:          number;
  n_green:        number;
  n_marginal:     number;
  n_red:          number;
  direction_hint: string;
}

export interface BeliefFamiliesResponse {
  n_families:        number;
  n_total_obs:       number;
  n_green_total:     number;
  n_marginal_total:  number;
  n_red_total:       number;
  families:          BeliefFamily[];
}

// Workflow trace — single-aggregator view of the end-to-end pipeline
// from paper ingestion through verdict to belief layer. Drives the
// /research/workflow SVG diagram (2026-06-23). Each stage row maps
// to one node in the diagram.
export interface WorkflowStage {
  key:         string;     // "papers" | "synthesis" | "hypotheses" | ...
  label:       string;
  count:       number;
  is_float?:   boolean;    // formatting hint (Brier shown as decimal)
  sub:         string;     // secondary metric / context
  href:        string;     // drill destination
  description: string;
}
export interface WorkflowCountsResponse {
  stages:            WorkflowStage[];
  secondary_counts:  Record<string, number>;
}

// Operator Console (2026-06-23) — UI-triggered pipeline stations.
// Backend: api/routes_operator_console.py
// Design:  docs/architecture/operator_console.md
//
// Reuses existing SessionType / SessionRow / SessionState above. New
// types here only cover what stations + jobs + cost-cap add.

export type ConsoleDataTier =
  | "user_data"
  | "demo_fixture"
  | "snapshot_data"
  | "wrds_required";

export type ConsoleJobState =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "halted_cost_cap"
  | "recovered_unknown";

export type ConsolePreflightStatus = "green" | "yellow" | "red";

export interface ConsoleStationSpec {
  station_id:             string;
  title:                  string;
  description:            string;
  data_tier:              ConsoleDataTier;
  requires_session_types: string[];   // SessionType strings
  estimated_minutes:      number;
  estimated_cost_usd:     number;
  icon:                   string;       // lucide-react icon name
  title_key:              string | null;
  description_key:        string | null;
}

export interface ConsoleStationsResponse {
  stations:     ConsoleStationSpec[];
  n_registered: number;
}

export interface ConsolePreflightCheck {
  name:    string;
  status:  ConsolePreflightStatus;
  detail:  string;
}

export interface ConsolePreflightResponse {
  can_trigger: boolean;
  checks:      ConsolePreflightCheck[];
  estimate:    { total_usd: number };
}

export interface ConsoleTriggerResponse {
  job_id: string;
  state:  ConsoleJobState;
}

export interface ConsoleJobStatus {
  job_id:             string;
  station_id:         string;
  session_id:         string;
  actor_id:           string;
  state:              ConsoleJobState;
  config:             Record<string, unknown>;
  estimated_cost_usd: number;
  created_ts:         string;
  updated_ts:         string;
  started_ts:         string | null;
  completed_ts:       string | null;
  result:             Record<string, unknown> | null;
  error:              string | null;
}

export interface ConsoleCostStatusResponse {
  session_id:     string;
  cap_usd:        number;
  spent_usd:      number;
  remaining_usd:  number;
  over_tolerance: boolean;
}

// Belief layer headline calibration numbers — surfaces the system's
// most important honest finding (predictor LOSES to family prior) in
// the UI instead of burying it in markdown (2026-06-23).
export interface BeliefCalibrationResponse {
  available:                 boolean;
  n_autopsies?:              number;
  predictor_brier?:          number;
  predictor_ci_lo?:          number;
  predictor_ci_hi?:          number;
  random_baseline?:          number;   // 0.4444 (3-class random)
  predictor_beats_random?:   boolean;
  family_prior_brier?:       number;   // the FAIR time-aware baseline
  delta_predictor_minus_fp?: number;   // POSITIVE = predictor loses
  delta_ci_lo?:              number;
  delta_ci_hi?:              number;
  hl_chi2?:                  number;
  hl_p_value?:               number;
  hl_calibrated?:            boolean;  // False = HL REJECTED (honest)
  reason?:                   string;
}

// Per-hypothesis safety-rail aggregator — drives the inline banner on
// /approvals row decision UX. Added 2026-06-14 (Phase 5a).
export interface SafetyRailRigorRow {
  rigor_id:         string;
  ts:               string;
  verdict_event_id: string | null;
  family:           string | null;
  template_name:    string | null;
  oos_status:       string | null;
  spanning_status:  string | null;
  borrow_status:    string | null;
  flags:            string[];
}
export interface SafetyRailAuditRow {
  audit_id:           string;
  ts:                 string;
  provider:           string;
  severity:           AuditSeverity;
  flagged_categories: string[];
  subject_ref:        string;
}
export interface SafetyRailBeliefFamily {
  family:         string | null;
  hyp_family:     string;
  match_kind:     "exact" | "substring" | "no_belief_data_yet";
  n_obs:          number;
  n_green:        number;
  n_marginal:     number;
  n_red:          number;
  direction_hint: string;
}
export interface SafetyRailsForHypothesis {
  hypothesis_id:       string;
  rigor:               SafetyRailRigorRow[];
  audits:              SafetyRailAuditRow[];
  belief_family:       SafetyRailBeliefFamily | null;
  verdict_event_ids:   string[];
  n_critical:          number;
  n_concern:           number;
  n_flagged:           number;
}

export interface GraveyardCollision {
  kind:           "autopsy_red" | "verdict_red";
  ts:             string;
  hypothesis_id:  string | null;
  event_id?:      string;
  family:         string | null;
  claim_excerpt:  string;
  score:          number;
  family_match:   boolean;
  jaccard:        number;
}
export interface GraveyardCollisions {
  hypothesis_id:    string;
  src_family:       string | null;
  n_total_red:      number;
  top_collisions:   GraveyardCollision[];
  score_doctrine:   string;
}

