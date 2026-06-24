// frontend/lib/queries.ts — typed React Query hooks over the api client.
// One place owns cache/refresh policy per resource, so pages just call a hook (no useEffect /
// setInterval / manual retry). queryKeys are stable so cross-page navigation reuses the cache.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useAsOf } from "@/lib/asof";

export const qk = {
  health: ["health"] as const,
  decay: ["decay"] as const,
  bookState: ["book", "state"] as const,
  bookNav: (days: number) => ["book", "nav", days] as const,
  agents: ["agents"] as const,
  graveyard: ["research", "graveyard"] as const,
};

// Live-ish data: short stale, background poll.
export const useDecayReport = () =>
  useQuery({ queryKey: qk.decay, queryFn: () => api.decayReport(), refetchInterval: 60_000 });

// Book/risk/holdings views honor the global as-of (time-travel). When pinned to a past date the
// data is immutable, so we stop polling. Live (asOf=null) keeps the 60s background refresh.
export const useBookState = () => {
  const { asOf } = useAsOf();
  return useQuery({ queryKey: ["book", "state", asOf], queryFn: () => api.bookState(asOf), refetchInterval: asOf ? false : 60_000 });
};
export const useBookDates = () =>
  useQuery({ queryKey: ["book", "dates"], queryFn: () => api.bookDates(), staleTime: 5 * 60_000 });

export const useBookNav = (days = 120) =>
  useQuery({ queryKey: qk.bookNav(days), queryFn: () => api.bookNav(days), refetchInterval: 60_000 });

export const useBookPositions = () => {
  const { asOf } = useAsOf();
  return useQuery({ queryKey: ["book", "positions", asOf], queryFn: () => api.bookPositions(asOf), refetchInterval: asOf ? false : 60_000 });
};

export const useBookPerf = () =>
  useQuery({ queryKey: ["book", "perf"], queryFn: () => api.bookPerf(), staleTime: 5 * 60_000 });

export const useBookTrades = (limit = 100) =>
  useQuery({ queryKey: ["book", "trades", limit], queryFn: () => api.bookTrades(limit), refetchInterval: 60_000 });

// Slow-moving config: long stale, no poll.
export const useAgents = () =>
  useQuery({ queryKey: qk.agents, queryFn: () => api.agents(), staleTime: 5 * 60_000 });

export const useGraveyard = () =>
  useQuery({ queryKey: qk.graveyard, queryFn: () => api.graveyard(), staleTime: 10 * 60_000 });

export const usePitAudit = () =>
  useQuery({ queryKey: ["research", "pit"], queryFn: () => api.pitAudit(), staleTime: 10 * 60_000 });

export const useDiscoveryQueues = (limit = 20) =>
  useQuery({
    queryKey: ["research", "discovery", "queues", limit] as const,
    queryFn: () => api.discoveryQueues(limit),
    refetchInterval: 30_000,
  });

export const useDiscoveryBookmarklet = () =>
  useQuery({
    queryKey: ["research", "discovery", "bookmarklet"] as const,
    queryFn: () => api.discoveryBookmarklet(),
    staleTime: 60 * 60_000,
  });

export const useDiscoveryWatchlist = () =>
  useQuery({
    queryKey: ["research", "discovery", "watchlist"] as const,
    queryFn: () => api.discoveryWatchlist(),
    refetchInterval: 60_000,
  });

export const useDiscoveryNominate = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (url: string) => api.discoveryNominate(url),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["research", "discovery", "queues"] }),
  });
};

export const useDiscoveryPromote = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sourceId: string) => api.discoveryPromote(sourceId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["research", "discovery", "queues"] }),
  });
};

export const useDiscoverySkip = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ sourceId, reason }: { sourceId: string; reason?: string }) =>
      api.discoverySkip(sourceId, reason),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["research", "discovery", "queues"] }),
  });
};

export const useHealth = () =>
  useQuery({ queryKey: qk.health, queryFn: () => api.health(), refetchInterval: 30_000, retry: 1 });

// Data-freshness authority (distinct from server health). Polls with health so the nav can show
// "server live" AND "data fresh/stale" as separate signals.
export const useFreshness = () =>
  useQuery({ queryKey: ["freshness"], queryFn: () => api.freshness(), refetchInterval: 30_000, retry: 1 });

// Data-refresh job status (polls while the staleness banner is mounted, since that's exactly when
// a refresh is relevant). The mutation kicks off the production daily job in the background.
export const useRefreshStatus = (poll = true) =>
  useQuery({ queryKey: ["ops", "refresh"], queryFn: () => api.refreshStatus(),
    refetchInterval: poll ? 4000 : false, retry: 1 });
export const useStartRefresh = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.startRefresh(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ops", "refresh"] }),
  });
};

export const useRisk = () => {
  const { asOf } = useAsOf();
  return useQuery({ queryKey: ["risk", asOf], queryFn: () => api.risk(asOf), refetchInterval: asOf ? false : 60_000 });
};

// Position-level risk decomposition — heavier (covariance over the holdings panel), slow-moving
// (the panel rebuilds ~daily), so longer stale + no poll. LIVE book only.
export const useRiskContrib = () =>
  useQuery({ queryKey: ["risk", "contrib"], queryFn: () => api.riskContrib(), staleTime: 2 * 60_000 });

// Operator discretionary overlay sleeve — human-originated positions, refetched fairly often
// since the user just placed/changed them via the approve loop.
export const useOverlay = () =>
  useQuery({ queryKey: ["book", "overlay"], queryFn: () => api.overlay(), staleTime: 20_000 });

// Deployed two-mechanism book (equity + carry). Heavy backend (rebuilds from cached
// futures + equity), so long stale + no poll.
export const useDQ = () =>
  useQuery({ queryKey: ["dq", "report"], queryFn: () => api.dq(), staleTime: 2 * 60_000 });

export const useCombinedBook = () =>
  useQuery({ queryKey: ["book", "combined"], queryFn: () => api.combined(), staleTime: 10 * 60_000 });

// Deployment manifest — single source of truth for "what's live". Slow-moving
// (only changes via scripts/deploy_config.py promote), so cache aggressively.
export const useDeployManifest = () =>
  useQuery({ queryKey: ["deploy", "manifest"], queryFn: () => api.deployManifest(), staleTime: 10 * 60_000 });

// Backend version + uptime + cache stats. Refresh every 30s — uptime
// changes continuously, and the cached-key set turns over.
export const useSystemVersion = () =>
  useQuery({ queryKey: ["system", "version"], queryFn: () => api.systemVersion(), refetchInterval: 30_000 });

export const useLlmBudget = () =>
  useQuery({ queryKey: ["ops", "llm_budget"], queryFn: () => api.llmBudget(), refetchInterval: 60_000 });

// v2 governance approvals queue — refetch every 60s to catch new requests.
// Research Ops inbox — refetch every 60s. Pass `since` for unread tracking.
export const useResearchOpsInbox = (since?: string) =>
  useQuery({
    queryKey: ["research_ops_inbox", since ?? ""],
    queryFn: () => api.researchOpsInbox(since),
    refetchInterval: 60_000,
  });

export const useResearchOpsLiterature = (since?: string) =>
  useQuery({
    queryKey: ["research_ops_literature", since ?? ""],
    queryFn: () => api.researchOpsLiterature(since),
    refetchInterval: 60_000,
  });

export const useResearchOpsLastVisit = () =>
  useQuery({
    queryKey: ["research_ops_last_visit"],
    queryFn: () => api.researchOpsLastVisit(),
    staleTime: 60_000,
  });

export const useV2Approvals = (status?: "pending" | "approved" | "rejected" | "expired") =>
  useQuery({
    queryKey: ["v2_approvals", status ?? "all"],
    queryFn: () => api.v2ListApprovals(status, 100),
    refetchInterval: 60_000,
  });

// Accumulating live-vs-backtest tracker — grows with the daily NAV; cheap, refetch occasionally.
export const useTracking = () =>
  useQuery({ queryKey: ["book", "tracking"], queryFn: () => api.tracking(), staleTime: 60_000 });

// Paper-broker target-vs-actual reconciliation (Alpaca paper if configured, else sim). Live
// broker reads → modest stale; refetches as fills come in after market open.
export const useExecution = () =>
  useQuery({ queryKey: ["book", "execution"], queryFn: () => api.execution(), staleTime: 60_000 });

export const useScenarios = () =>
  useQuery({ queryKey: ["risk", "scenarios"], queryFn: () => api.scenarios(), staleTime: 2 * 60_000 });

export const useFactorExposure = () =>
  useQuery({ queryKey: ["risk", "factors"], queryFn: () => api.factorExposure(), staleTime: 2 * 60_000 });

export const useDailyBrief = () =>
  useQuery({ queryKey: ["brief"], queryFn: () => api.brief(), refetchInterval: 60_000 });

export const useProvenance = () =>
  useQuery({ queryKey: ["provenance"], queryFn: () => api.provenance(), staleTime: 5 * 60_000 });

// Phase 1.2 / 4.1 / B safety-rail surfaces (added 2026-06-14)
export const usePostGreenRigorRecent = (days = 7, limit = 50) =>
  useQuery({
    queryKey: ["research", "post_green_rigor", days, limit],
    queryFn:  () => api.postGreenRigorRecent(days, limit),
    refetchInterval: 60_000,
  });

export const useExternalAuditsRecent = (days = 7, limit = 50) =>
  useQuery({
    queryKey: ["research", "external_audits", days, limit],
    queryFn:  () => api.externalAuditsRecent(days, limit),
    refetchInterval: 60_000,
  });

export const useBeliefFamilies = (minObs = 3) =>
  useQuery({
    queryKey: ["research", "belief_families", minObs],
    queryFn:  () => api.beliefFamilies(minObs),
    refetchInterval: 120_000,
  });

// Belief headline calibration — refreshed daily by belief-refresh cron
// 06:35; UI re-polls every 5 min so the post-cron values land within a
// few minutes of being written.
export const useBeliefCalibration = () =>
  useQuery({
    queryKey: ["research", "belief_calibration"],
    queryFn:  () => api.beliefCalibration(),
    refetchInterval: 300_000,
    staleTime:       60_000,
  });

// Workflow trace counts — drives the /research/workflow SVG. Refresh
// every 30s so the user sees fresh counts when a cron fires during
// a session (cheap aggregator, no compute).
export const useWorkflowCounts = () =>
  useQuery({
    queryKey: ["research", "workflow_counts"],
    queryFn:  () => api.workflowCounts(),
    refetchInterval: 30_000,
    staleTime:       15_000,
  });

// ── Operator Console (2026-06-23) ────────────────────────────────
// Registry is small + rarely changes; cache long. Job status is
// polled while job is non-terminal.

export const useConsoleStations = () =>
  useQuery({
    queryKey: ["console", "stations"],
    queryFn:  () => api.consoleStations(),
    refetchInterval: 60_000,
    staleTime:       30_000,
  });

export const useConsoleStation = (stationId: string | null | undefined) =>
  useQuery({
    queryKey: ["console", "station", stationId],
    queryFn:  () => api.consoleStation(stationId!),
    enabled:  !!stationId,
    staleTime: 60_000,
  });

export const useConsoleJobStatus = (jobId: string | null | undefined) =>
  useQuery({
    queryKey: ["console", "job", jobId],
    queryFn:  () => api.consoleJobStatus(jobId!),
    enabled:  !!jobId,
    // Poll fast while terminal-or-not unknown; the component using
    // this hook can switch to no-refetch once the job is terminal.
    refetchInterval: (q) => {
      const state = (q.state.data as { state?: string } | undefined)?.state;
      const terminal = new Set(["completed", "failed", "cancelled", "halted_cost_cap", "recovered_unknown"]);
      return state && terminal.has(state) ? false : 2_000;
    },
  });

export const useConsoleCostStatus = (sessionId: string | null | undefined, capUsd = 1.0) =>
  useQuery({
    queryKey: ["console", "cost", sessionId, capUsd],
    queryFn:  () => api.consoleCostStatus(sessionId!, capUsd),
    enabled:  !!sessionId,
    refetchInterval: 5_000,
    staleTime:       2_000,
  });

// Per-hypothesis safety-rail lookup — used by approval rows for inline
// "Rigor / Audit / Belief" decision context. Stale-while-revalidate the
// dataset for 5 minutes (it doesn't change between resolves).
export const useSafetyRailsForHypothesis = (hypothesisId: string | null) =>
  useQuery({
    queryKey: ["research", "safety_rails", hypothesisId],
    queryFn:  () => api.safetyRailsForHypothesis(hypothesisId!),
    enabled:   !!hypothesisId,
    staleTime: 5 * 60_000,
  });

// Phase 8 (2026-06-14): graveyard collision detection — "have we
// already killed something like this?" — reactive subscriber.
export const useGraveyardCollisions = (hypothesisId: string | null) =>
  useQuery({
    queryKey: ["research", "graveyard_collisions", hypothesisId],
    queryFn:  () => api.graveyardCollisions(hypothesisId!),
    enabled:   !!hypothesisId,
    staleTime: 10 * 60_000,
  });

export const useApprovals = () =>
  useQuery({ queryKey: ["approvals"], queryFn: () => api.approvals(false), refetchInterval: 30_000 });

// Stage C bug-fix (2026-06-08): strengthener (B) approvals are a
// separate queue from the legacy ticker-level /api/approvals. Before
// this hook, the topbar ShieldCheck only counted legacy approvals
// (always 0) so the entry point was invisible even when B had
// pending APPROVE_FOR_PIPELINE / DOCTRINE_AMENDMENT_NEEDED waiting
// for the principal. UI now reads BOTH endpoints.
export const useStrengthenerApprovals = () =>
  useQuery({
    queryKey:        ["strengthener", "approvals"],
    queryFn:         async () => {
      const r = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? ""}/api/strengthener/approvals`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n_pending: number; n_resolved: number;
                                     rows: unknown[] }>;
    },
    refetchInterval: 30_000,
  });

// Tier C-2d.2 (2026-06-08): factor SPEC approval queue. Same shape
// + poll cadence as useStrengthenerApprovals so the topbar
// ShieldCheck badge can roll up BOTH pending counts.
export const useFactorSpecApprovals = () =>
  useQuery({
    queryKey:        ["strengthener", "factor_specs"],
    queryFn:         async () => {
      const r = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL ?? ""}/api/strengthener/factor_specs`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n_pending: number; n_resolved: number;
                                     rows: unknown[] }>;
    },
    refetchInterval: 30_000,
  });

// History/audit view: fetch incl. resolved (the page filters out pending). Slower-moving → no poll.
export const useApprovalsHistory = (enabled = true) =>
  useQuery({ queryKey: ["approvals", "history"], queryFn: () => api.approvals(true),
    enabled, staleTime: 30_000 });

// Per-approval decision context. Heavier (SQL + optional RAG), so longer stale + no poll;
// enabled only when an id is present (the review page reads ?id client-side).
export const useApprovalDetail = (id: number | null) =>
  useQuery({
    queryKey: ["approval", id],
    queryFn: () => api.approvalDetail(id as number),
    enabled: id != null && Number.isFinite(id),
    staleTime: 30_000,
    retry: 1,
  });

export const useResolveApprovals = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.resolveApprovals,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["approvals"] });
      qc.invalidateQueries({ queryKey: ["approval"] });
    },
  });
};

export const useAlerts = (daysBack = 30) =>
  useQuery({ queryKey: ["alerts", daysBack], queryFn: () => api.alerts(daysBack), refetchInterval: 60_000 });

export const useOpsCost = () =>
  useQuery({ queryKey: ["ops", "cost"], queryFn: () => api.opsCost(), refetchInterval: 60_000 });

export const useOpsHealth = () =>
  useQuery({ queryKey: ["ops", "health"], queryFn: () => api.opsHealth(), refetchInterval: 60_000 });

// Agent behavioral-eval: read the last scores (free); run is opt-in (costs LLM) + polled.
export const useEvalLatest = () =>
  useQuery({ queryKey: ["ops", "eval"], queryFn: () => api.evalLatest(), staleTime: 60_000 });
export const useEvalRunStatus = (poll = true) =>
  useQuery({ queryKey: ["ops", "eval", "run"], queryFn: () => api.evalRunStatus(),
    refetchInterval: poll ? 5000 : false, retry: 1 });
export const useStartEval = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.startEval(),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["ops", "eval", "run"] }),
  });
};

// ── Research event store (M3 2026-06-02) ─────────────────────────

export const useResearchStoreEvents = (params: {
  event_type?:   string;
  subject_type?: string;
  subject_id?:   string;
  verdict?:      string;
  family?:       string;
  since?:        string;
  limit?:        number;
} = {}) =>
  useQuery({
    queryKey: ["research_store_events", params],
    queryFn:  () => api.researchStoreEvents(params),
    staleTime: 30_000,
  });

export const useResearchStoreSubjects = (family?: string) =>
  useQuery({
    queryKey: ["research_store_subjects", family ?? ""],
    queryFn:  () => api.researchStoreSubjects(family),
    staleTime: 120_000,
  });

export const useResearchStoreLineage = (event_id?: string) =>
  useQuery({
    queryKey: ["research_store_lineage", event_id ?? ""],
    queryFn:  () => api.researchStoreLineage(event_id!),
    enabled:  Boolean(event_id),
    staleTime: 120_000,
  });

export const useResearchStoreSummary = () =>
  useQuery({
    queryKey: ["research_store_summary"],
    queryFn:  () => api.researchStoreSummary(),
    staleTime: 60_000,
  });

// ── Sessions (P5 2026-06-03) ─────────────────────────────────────

export const useActiveSession = () =>
  useQuery({
    queryKey: ["session_active"],
    queryFn:  () => api.sessionActive(),
    refetchInterval: 5_000,   // poll so banner reflects emit + close events
  });

export const useSessionsList = (params: { limit?: number; state?: string; session_type?: string } = {}) =>
  useQuery({
    queryKey: ["sessions_list", params],
    queryFn:  () => api.sessionsList(params as any),
    staleTime: 15_000,
  });

export const useSession = (sessionId?: string) =>
  useQuery({
    queryKey: ["session", sessionId ?? ""],
    queryFn:  () => api.sessionGet(sessionId!),
    enabled:  Boolean(sessionId),
    staleTime: 5_000,
  });

export const useSessionTypes = () =>
  useQuery({
    queryKey: ["session_types"],
    queryFn:  () => api.sessionTypes(),
    staleTime: 60 * 60 * 1000,    // static metadata, refresh hourly
  });

export const useOpenSession = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.sessionOpen,
    onSuccess:  () => {
      qc.invalidateQueries({ queryKey: ["session_active"] });
      qc.invalidateQueries({ queryKey: ["sessions_list"] });
    },
  });
};

export const useRecordPreflight = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, digest }: {
      sessionId: string;
      digest: import("@/lib/api").PreflightDigestInput;
    }) => api.sessionPreflight(sessionId, digest),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session_active"] });
      qc.invalidateQueries({ queryKey: ["sessions_list"] });
    },
  });
};

export const useCloseSession = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => api.sessionClose(sessionId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session_active"] });
      qc.invalidateQueries({ queryKey: ["sessions_list"] });
    },
  });
};

export const useAbandonSession = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ sessionId, reason }: { sessionId: string; reason: string }) =>
      api.sessionAbandon(sessionId, reason),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["session_active"] });
      qc.invalidateQueries({ queryKey: ["sessions_list"] });
    },
  });
};

// ── Roadmap (Gap A 2026-06-03) ───────────────────────────────────

export const useRoadmapAxes = (params: { state?: string; tier?: string; family?: string } = {}) =>
  useQuery({
    queryKey: ["roadmap_axes", params],
    queryFn:  () => api.roadmapList(params),
    staleTime: 30_000,
  });

export const useRoadmapAxis = (axis_id?: string) =>
  useQuery({
    queryKey: ["roadmap_axis", axis_id ?? ""],
    queryFn:  () => api.roadmapGet(axis_id!),
    enabled:  Boolean(axis_id),
    staleTime: 30_000,
  });

export const useUpsertAxis = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: api.roadmapUpsert,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["roadmap_axes"] });
      qc.invalidateQueries({ queryKey: ["roadmap_axis"] });
    },
  });
};
