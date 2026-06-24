// Page-context registry — what each major page IS, what the user can
// DO here, and what's most often the NEXT click. Used by:
//   - HelpOnThisPage (the "?" button on ModeHeader) to pre-fill chat
//     with a context block so chat answers are page-aware.
//   - NextClickHint (deterministic next-action chip in the header).
//
// One entry per top-level page. Sub-pages inherit from their nearest
// matching prefix unless they override. Pattern matching is "starts
// with" with a longest-prefix wins rule.
//
// Keep descriptions short: 1 sentence what + 1 sentence why + 3 common
// actions. The block ends up inside an LLM prompt, so brevity matters.

export type PageContext = {
  /** URL prefix that triggers this entry. Longest match wins. */
  pathPrefix:     string;
  /** Short page name shown in the chat preamble. */
  title:          string;
  /** What this page IS — 1-2 sentences. */
  description:    string;
  /** Most-frequent next actions, ordered by ROI. */
  commonActions:  string[];
};


const REGISTRY: PageContext[] = [
  // ─── OPERATE ────────────────────────────────────────────────
  {
    pathPrefix:    "/dashboard",
    title:         "Today (OPERATE landing)",
    description:   "Daily book status: pre-batch DQ + decay sentinel + active sessions + alerts. " +
                   "Top tile (Daily Directive) auto-aggregates state across the whole project and " +
                   "tells you the 3 ranked actions for today + the top paper-corpus directions.",
    commonActions: [
      "Click an item in TODAY to jump to the surface that resolves it",
      "Click a DIRECTIONS row to open the source paper",
      "Resolve any BLOCKER (DQ HALT / decay ACTION) before starting new research",
    ],
  },
  {
    pathPrefix:    "/research/decay",
    title:         "Decay sentinel",
    description:   "Tracks every DEPLOYED sleeve for alpha decay (rolling-OOS Sharpe drop, " +
                   "drawdown breach, factor-exposure drift). Verdicts: CLEAN / WATCH / ACTION.",
    commonActions: [
      "Open an ACTION sleeve to read the diagnostic",
      "Decide: re-test, ramp down, or de-deploy (governance flow, not a chat command)",
      "Verify the most recent decay run completed on schedule",
    ],
  },
  {
    pathPrefix:    "/research/sessions",
    title:         "Sessions (audit + history)",
    description:   "Every typed session (research_new / audit / ops / doctrine / exploration) " +
                   "with its lifecycle state and linked events. Use this to verify exit " +
                   "conditions before closing, or to find a past session by subject.",
    commonActions: [
      "Click a research_new session to verify its exit gate (≥1 verdict + ≥1 evidence)",
      "Use Abandon (with reason) if a session got stuck and you're moving on",
      "Replay a closed session's event stream to recover decisions",
    ],
  },
  {
    pathPrefix:    "/research/library",
    title:         "Mechanism library (deployed + ideation)",
    description:   "Every mechanism the project has notes for — deployed sleeves, paused, " +
                   "ideation-only, graveyard. Detail page shows YAML + canonical paper + " +
                   "audit trail + linked events.",
    commonActions: [
      "Click a DEPLOYED sleeve to verify its decay status",
      "Click an IDEATION-only entry to read why it isn't deployed yet",
      "Use the candidate-pipeline retest button to re-validate an existing sleeve",
    ],
  },
  // ─── RESEARCH ──────────────────────────────────────────────
  {
    pathPrefix:    "/research/forward",
    title:         "Forward research queue",
    description:   "Paper-grounded untested hypotheses extracted via T7 chain (PAPER → " +
                   "HYPOTHESIS → TEST → VERDICT). Each row carries a verbatim chunk_id citation " +
                   "to the source paper. Filter by family / priority / data availability / PM status.",
    commonActions: [
      "Approve a candidate (✓ chip) to mark it PM-cleared",
      "Click 'Open research session →' to start a research_new session against that hypothesis",
      "Multi-select 2-3 rows to read the Compare panel before deciding",
    ],
  },
  {
    pathPrefix:    "/research/enhance",
    title:         "Enhance workspace (Pick / Run / Decide)",
    description:   "Single-page workspace replacing the 11-step wizard. Three panels: PICK " +
                   "(approve + run carry candidates), RUN (graveyard check + strict-gate funnel " +
                   "+ pipeline SSE), DECIDE (verdict + adjacent untested).",
    commonActions: [
      "Multi-select candidates in PICK, then click 'Approve & Run' to fire all 4 steps",
      "In RUN, hand off to Claude if no parquet matches the family (carry has no auto-match)",
      "After verdict lands, use AdjacentActions to find untested same-subtype next",
    ],
  },
  {
    pathPrefix:    "/research/papers",
    title:         "Paper library",
    description:   "Every ingested paper with shelf classification (doctrine_method / " +
                   "yellow_motivation / green_critique / red_motivation / red_critique). Detail " +
                   "page shows extracted hypotheses + chunk_id-traced claims + ingestion " +
                   "lineage.",
    commonActions: [
      "Click a paper to read its extracted hypotheses",
      "Filter by shelf to find the highest-signal papers (doctrine_method)",
      "From paper detail, jump to /research/forward to test an untested hypothesis",
    ],
  },
  {
    pathPrefix:    "/research/papers/new",
    title:         "Paper ingestion",
    description:   "Self-service paper ingestion. Drop a PDF or paste a URL → preview classification " +
                   "→ approve → flows into the corpus. Decoupled from classification (you can " +
                   "re-classify later without re-ingesting).",
    commonActions: [
      "Drop a PDF or paste an OpenAlex / SSRN URL",
      "Verify the preview parses cleanly before approving",
      "If classification looks wrong, mark it and continue (re-classify is cheap)",
    ],
  },
  {
    pathPrefix:    "/research/candidate",
    title:         "Candidate pipeline (live SSE)",
    description:   "Stream the engine.research.candidate_pipeline_v2 LangGraph DAG against a " +
                   "chosen returns parquet. Graveyard check + strict-gate funnel are inline. " +
                   "Final decision (PROMOTE_TO_GATE / MARGINAL / HARD_REJECT) lands as a typed " +
                   "factor_verdict_filed event.",
    commonActions: [
      "Pick a parquet from the dropdown; click Run pipeline",
      "Watch each step's PASS/WARN/FAIL stream in",
      "On verdict, the lesson + AdjacentActions appear on /research/lessons",
    ],
  },
  {
    pathPrefix:    "/research/lessons",
    title:         "Lessons (verdicts + graveyard)",
    description:   "Every factor_verdict_filed event presented as a Lesson card with summary + " +
                   "failure modes + paper-chunk_id citation. RED lessons are the graveyard; " +
                   "GREEN/MARGINAL link to deployed sleeves.",
    commonActions: [
      "Filter by verdict=red to scan the graveyard before testing similar ideas",
      "Open a lesson to read AdjacentActions (same-subtype + same-family untested)",
      "From a lesson, file an audit_subject intent to re-investigate",
    ],
  },
  // ─── DECIDE / GOVERN ───────────────────────────────────────
  {
    pathPrefix:    "/inbox",
    title:         "Inbox (cross-channel triage)",
    description:   "Priority-sorted single column with engine self-reports + research-direction " +
                   "alerts + methodology nudges. Snooze / pin / archive per item.",
    commonActions: [
      "Scan top items; ack or snooze each",
      "Pin items you'll act on this week",
      "Items auto-archive after the configured retention",
    ],
  },
  {
    pathPrefix:    "/approvals",
    title:         "Approvals (governance gate)",
    description:   "PendingApproval queue: every action proposed by Chief of Staff, the user, " +
                   "or an agent that requires explicit human approval. Approving a position " +
                   "OVERLAY proposal moves real (paper) money; approving an advisory just " +
                   "records the decision.",
    commonActions: [
      "Read the rationale before approving",
      "Approve = decision recorded; for overlay proposals, the engine then sets the position",
      "Reject with a reason — that's the audit trail",
    ],
  },
];


/** Look up the page context for a path. Longest-prefix wins so
 *  /research/papers/new matches its specific entry, not /research/papers. */
export function getPageContext(pathname: string): PageContext | null {
  let best: PageContext | null = null;
  let bestLen = -1;
  for (const c of REGISTRY) {
    if (pathname.startsWith(c.pathPrefix) && c.pathPrefix.length > bestLen) {
      best = c;
      bestLen = c.pathPrefix.length;
    }
  }
  return best;
}


/** Format a PageContext into the prompt block that gets pre-pended
 *  to the user's chat question. */
export function formatContextForChat(ctx: PageContext, pathname: string): string {
  const lines = [
    `I'm on ${pathname} — ${ctx.title}.`,
    `What this page is: ${ctx.description}`,
    `Common next actions here: ${ctx.commonActions.map((a, i) => `(${i + 1}) ${a}`).join("; ")}.`,
    ``,
    `My question: `,
  ];
  return lines.join("\n");
}
