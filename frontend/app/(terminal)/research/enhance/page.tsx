"use client";

// /research/enhance — single-page workspace replacing the old 11-step
// step-rail wizard. Three panels:
//
//   ① PICK    inline CARRY forward-vector list, multi-select, one
//             "Approve & Run" button doing approve + open session +
//             file intent + start pipeline stream
//   ② RUN     inline graveyard check + SSE pipeline events, optional
//             Claude hand-off
//   ③ DECIDE  inline latest CARRY lesson + AdjacentActions, restart
//             loop button
//
// Lower panels auto-expand as upper panels complete. Polls every 10s
// for state changes (verdict landing, session being closed elsewhere).

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  CheckCircle2, Circle, ArrowRight, ChevronRight, ChevronDown,
  ChevronUp, Compass, Skull, AlertTriangle, ExternalLink,
  PlayCircle, Loader2, RotateCcw, Filter as FilterIcon,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { API_BASE } from "@/lib/api";
import { fileIntent } from "@/lib/intents";
import { Card, Badge, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { Tip } from "@/components/Tip";
import { StrictGateFunnel } from "@/components/StrictGateFunnel";
import { HandoffToClaude } from "@/components/HandoffToClaude";
import { SpecPreviewCard } from "@/components/SpecPreviewCard";
import {
  useWizardState, lessonForFamilyRecently,
} from "@/lib/wizardState";


// ── Types ───────────────────────────────────────────────────────


type ForwardVector = {
  forward_vector_id:    string;
  source_paper_id:      string;
  paper_title:          string;
  source_hypothesis_id: string;
  claim:                string;
  mechanism_family:     string;
  mechanism_subtype:    string;
  predicted_direction:  string;
  required_data:        string[];
  priority:             "high" | "medium" | "low";
  pm_status:            "extracted" | "reviewed" | "approved" | "rejected";
  data_status:          "have" | "partial" | "missing" | "unknown";
  data_have:            string[];
  data_missing:         string[];
};


type LessonMini = {
  lesson_id:           string;
  candidate_name:      string;
  verdict:             string;
  mechanism_family:    string;
  mechanism_subtype:   string;
  failure_modes:       string[];
  summary:             string;
};


type StepEvent = {
  node:      string;
  step_name: string;
  status:    "PASS" | "WARN" | "FAIL" | "SKIP" | "INFO";
  verdict:   string;
};


type FinalEvent = {
  final_decision:     string;
  rationale:          string;
  candidate_relation: string;
};


type PipelineCandidate = {
  series_path:    string;
  proposal_name:  string;
  proposed_role:  string;
  mechanism_id:   string;
  family:         string;
};


// Family this workspace handles. The PICK panel auto-filters to it.
const RECIPE_FAMILY = "CARRY";


const STATUS_TONE: Record<string, string> = {
  PASS: "bg-ok/15 text-ok",
  WARN: "bg-warn/15 text-warn",
  FAIL: "bg-danger/15 text-danger",
  SKIP: "bg-muted/15 text-muted",
  INFO: "bg-info/15 text-info",
};


// ── Page ────────────────────────────────────────────────────────


export default function EnhanceWorkspacePage() {
  const router = useRouter();
  const state = useWizardState({ family: "" });

  // Vectors — CARRY only, open + data=have. Polled separately so
  // optimistic approve-UI is independent of the wizard's 10s tick.
  const [vectors, setVectors] = useState<ForwardVector[]>([]);
  const [vectorsLoading, setVectorsLoading] = useState(true);
  const [vectorsError, setVectorsError] = useState<string | null>(null);
  // Pipeline candidates — the parquets we already have on disk.
  const [pipelineCandidates, setPipelineCandidates] = useState<PipelineCandidate[]>([]);
  // PICK state
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [approving, setApproving] = useState(false);
  const [pickError, setPickError] = useState<string | null>(null);
  // After "Approve & Run" — the hypothesis we're running on.
  const [activeHypId, setActiveHypId] = useState<string | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  // F6.2/F6.3 (2026-06-05): which build path won for the picked hypothesis.
  // "composer"          — F2 prepare-from-fv built a typed-spec parquet (best)
  // "series_factory"    — family/subtype builder produced a parquet (also good)
  // "legacy_string_match" — fell back to the closest pre-baked parquet (HONEST DEBT)
  // "claude_handoff"    — nothing built, user must use Claude
  const [buildPath, setBuildPath] = useState<
    "composer" | "series_factory" | "legacy_string_match" | "claude_handoff" | null
  >(null);
  // Human-readable explanation of why this path won (what the prior steps refused / failed)
  const [buildPathNote, setBuildPathNote] = useState<string | null>(null);
  // RUN state
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [finalDecision, setFinalDecision] = useState<FinalEvent | null>(null);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [pickedCandidate, setPickedCandidate] = useState<PipelineCandidate | null>(null);
  const esRef = useRef<EventSource | null>(null);
  // Funnel/disclosure
  const [funnelOpen, setFunnelOpen] = useState(false);
  // Panel expand state (auto-collapses after done)
  const [pickExpanded, setPickExpanded] = useState(true);
  const [runExpanded, setRunExpanded] = useState(false);
  const [decideExpanded, setDecideExpanded] = useState(false);

  // ── Initial loads ──

  const reloadVectors = () => {
    const params = new URLSearchParams({
      pm_status:        "open",
      data_status:      "have",
      mechanism_family: RECIPE_FAMILY,
      top:              "30",
    });
    setVectorsLoading(true);
    fetch(`${API_BASE}/api/paper_chain/forward-vectors?${params}`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data: ForwardVector[]) => { setVectors(data); setVectorsError(null); })
      .catch((e) => setVectorsError(String(e)))
      .finally(() => setVectorsLoading(false));
  };

  useEffect(() => { reloadVectors(); }, []);

  useEffect(() => {
    fetch(`${API_BASE}/api/pipeline/candidates`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((d) => setPipelineCandidates(d.candidates || []))
      .catch(() => setPipelineCandidates([]));
  }, []);

  // Clean up SSE on unmount.
  useEffect(() => () => esRef.current?.close(), []);

  // ── Panel-done predicates ──

  const pickDone = Boolean(activeHypId);
  const runDone  = Boolean(finalDecision) ||
                   Boolean(lessonForFamilyRecently(state, "carry", 60));
  const recentLesson = lessonForFamilyRecently(state, "carry", 60 * 24);
  const decideDone   = false;  // never auto-tick; user chooses to restart

  // Auto-expand the next panel as upper one completes.
  useEffect(() => {
    if (pickDone && !runExpanded && !runDone) {
      setRunExpanded(true);
      setPickExpanded(false);
    }
  }, [pickDone, runDone, runExpanded]);

  useEffect(() => {
    if (runDone && !decideExpanded) {
      setDecideExpanded(true);
      setRunExpanded(false);
    }
  }, [runDone, decideExpanded]);

  // ── Approve & Run action ──

  const pickedVectors = useMemo(
    () => vectors.filter((v) => picked.has(v.source_hypothesis_id)),
    [vectors, picked]
  );

  // Pick a sensible default pipeline candidate when the user clicks
  // Approve & Run. Match by family substring (case-insensitive).
  // If none matches, fall back to the first candidate so the SSE has
  // some series to run on — user can still re-select in the RUN panel.
  const autoCandidate = useMemo<PipelineCandidate | null>(() => {
    if (pipelineCandidates.length === 0) return null;
    const fam = RECIPE_FAMILY.toLowerCase();
    return (
      pipelineCandidates.find((c) => c.family.toLowerCase().includes(fam))
      ?? null  // do NOT silently substitute — show the picker instead
    );
  }, [pipelineCandidates]);

  const approveAndRun = async () => {
    if (pickedVectors.length === 0) return;
    setApproving(true);
    setPickError(null);
    try {
      // 1. Approve every picked hypothesis (PM status: approved)
      for (const v of pickedVectors) {
        const r = await fetch(`${API_BASE}/api/paper_chain/forward-vectors/review`, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({
            source_hypothesis_id: v.source_hypothesis_id,
            status:               "approved",
            reviewed_by:          "user",
            note:                 "approved via /research/enhance",
          }),
        });
        if (!r.ok) {
          const txt = await r.text();
          throw new Error(`approve failed for ${v.source_hypothesis_id}: ${txt.slice(0, 200)}`);
        }
      }
      // 2. Open a research_new session (first picked drives the title)
      const head = pickedVectors[0];
      const sesRes = await fetch(`${API_BASE}/api/sessions/open`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_type: "research_new",
          title:        `Enhance · ${head.mechanism_subtype || head.claim.slice(0, 60)}`,
        }),
      });
      let sessionId: string | null = null;
      if (sesRes.ok) sessionId = (await sesRes.json())?.session_id ?? null;
      else console.warn("[enhance] /sessions/open failed:", await sesRes.text());

      // 3. File research_test intent (typed handoff for the poll hook).
      // Not awaited — primary action is approve+open; intent is bookkeeping.
      // But surface failures to console so they aren't silently swallowed.
      fileIntent({
        kind:         "research_test",
        subject_type: "hypothesis",
        subject_id:   head.source_hypothesis_id,
        source_page:  "/research/enhance",
        payload:      {
          paper_id:          head.source_paper_id,
          paper_title:       head.paper_title,
          mechanism_family:  head.mechanism_family,
          mechanism_subtype: head.mechanism_subtype,
          priority:          head.priority,
          claim:             head.claim,
          session_id:        sessionId,
          batch_size:        pickedVectors.length,
        },
      })
        .then((r) => { if (!r.ok) console.warn("[enhance] research_test intent refused:", r); })
        .catch((e) => console.warn("[enhance] research_test intent errored:", e));

      // 4. Move to RUN
      setActiveHypId(head.source_hypothesis_id);
      setActiveSessionId(sessionId);
      reloadVectors();   // refresh approved status display

      // 5. F6.2/F6.3 (2026-06-05): typed build chain with graceful
      //    fallback. Pre-F6 this was a string-match into the 5
      //    pre-baked parquets — user picked a typed hypothesis but
      //    the pipeline tested a generic carry parquet, so the
      //    verdict didn't actually correspond to the picked spec.
      //
      //    New order (each step's failure mode surfaces in UI):
      //      A. POST /api/pipeline/prepare-from-fv  (Composer / typed spec)
      //      B. POST /api/series_factory/build      (family+subtype builder)
      //      C. string-match autoCandidate          (legacy debt — UI flags it)
      //      D. show Claude handoff                 (nothing built)
      let resolved: PipelineCandidate | null = null;
      let path: typeof buildPath = null;
      let note: string | null = null;

      // A: Composer / typed spec
      try {
        const r = await fetch(
          `${API_BASE}/api/pipeline/prepare-from-fv?source_hypothesis_id=${encodeURIComponent(head.source_hypothesis_id)}`,
          { method: "POST" },
        );
        if (r.ok) {
          const prep = await r.json();
          resolved = {
            series_path:   prep.parquet_path,
            proposal_name: prep.proposal_name,
            proposed_role: prep.proposed_role || "alpha_seeker",
            mechanism_id:  prep.signal_type || head.mechanism_subtype,
            family:        prep.family || head.mechanism_family,
          };
          path = "composer";
          note = `Composer built spec_hash=${(prep.spec_hash || "").slice(0, 8)} (typed spec)`;
        } else if (r.status === 412) {
          // missing_components — fall through to B
          const err = await r.json().catch(() => ({}));
          note = `Composer 412: ${err.detail?.n_gaps ?? "?"} component(s) missing → trying series_factory`;
        } else {
          const err = await r.text();
          note = `Composer ${r.status}: ${err.slice(0, 80)} → trying series_factory`;
        }
      } catch (e: any) {
        note = `Composer network error: ${e?.message ?? e} → trying series_factory`;
      }

      // B: series_factory
      if (!resolved) {
        try {
          const r = await fetch(`${API_BASE}/api/series_factory/build`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              family:        head.mechanism_family,
              subtype:       head.mechanism_subtype,
              hypothesis_id: head.source_hypothesis_id,
            }),
          });
          const data = await r.json();
          if (data.ok && data.path) {
            resolved = {
              series_path:   data.path,
              proposal_name: `fv_${head.source_hypothesis_id.slice(0, 8)}`,
              proposed_role: "alpha_seeker",
              mechanism_id:  head.mechanism_subtype || head.mechanism_family,
              family:        head.mechanism_family,
            };
            path = "series_factory";
            note = `series_factory built ${data.family}/${data.subtype} (${data.n_obs} obs)`;
          } else {
            note = `${note ? note + " · " : ""}series_factory: ${data.error || "ok=false"} → falling back to legacy`;
          }
        } catch (e: any) {
          note = `${note ? note + " · " : ""}series_factory network error → falling back to legacy`;
        }
      }

      // C: legacy string-match (transparent fallback — UI shows the warning)
      if (!resolved && autoCandidate) {
        resolved = autoCandidate;
        path = "legacy_string_match";
        note = `${note ? note + " · " : ""}Using pre-baked parquet ${autoCandidate.proposal_name} — NOT the typed spec`;
      }

      // D: nothing built
      if (!resolved) {
        path = "claude_handoff";
        note = note ?? "No build path available — use Claude handoff to build the series.";
      }

      setBuildPath(path);
      setBuildPathNote(note);
      setPickedCandidate(resolved);
      if (resolved) startStream(resolved);
    } catch (e: any) {
      setPickError(String(e?.message ?? e));
    } finally {
      setApproving(false);
    }
  };

  // ── SSE: start a pipeline stream against a chosen candidate ──

  const startStream = (cand: PipelineCandidate) => {
    if (esRef.current) esRef.current.close();
    setSteps([]); setFinalDecision(null); setStreamError(null);
    setRunning(true);
    const params = new URLSearchParams({
      series_path:   cand.series_path,
      proposal_name: cand.proposal_name,
      proposed_role: cand.proposed_role,
      mechanism_id:  cand.mechanism_id,
      family:        cand.family,
    });
    const es = new EventSource(`${API_BASE}/api/pipeline/stream?${params}`);
    esRef.current = es;
    es.addEventListener("step_complete", (e: MessageEvent) => {
      try { setSteps((p) => [...p, JSON.parse(e.data) as StepEvent]); } catch {}
    });
    es.addEventListener("pipeline_complete", (e: MessageEvent) => {
      try { setFinalDecision(JSON.parse(e.data) as FinalEvent); } catch {}
      es.close();
      setRunning(false);
    });
    es.addEventListener("pipeline_error", (e: MessageEvent) => {
      try { setStreamError(JSON.parse(e.data)?.error || "pipeline error"); }
      catch { setStreamError("pipeline error"); }
      es.close();
      setRunning(false);
    });
    es.onerror = () => {
      if (es.readyState === EventSource.CLOSED) setRunning(false);
    };
  };

  const restartLoop = () => {
    // Tear down and reset to PICK
    if (esRef.current) esRef.current.close();
    setActiveHypId(null);
    setActiveSessionId(null);
    setPicked(new Set());
    setSteps([]);
    setFinalDecision(null);
    setPickedCandidate(null);
    setBuildPath(null);
    setBuildPathNote(null);
    setPickExpanded(true);
    setRunExpanded(false);
    setDecideExpanded(false);
    reloadVectors();
  };

  // ── Render ──

  const n_done = [pickDone, runDone, decideDone].filter(Boolean).length;

  return (
    <div className="p-6 space-y-4">
      <ModeHeader
        mode="research"
        title="Enhance the book"
        subtitle={<>
          One-page workspace: <b>pick</b> a paper-grounded carry hypothesis,
          <b> run</b> it through the pipeline, <b>decide</b> the next loop.
          State polls every 10s; come back any time.
        </>}
      />

      <Card className="p-0 overflow-hidden">
        <div className="px-3 py-2 border-b border-border/30 flex items-center gap-3">
          <span className="text-[10px] uppercase tracking-[0.18em] text-muted/70">Recipe</span>
          <span className="text-[12px] font-semibold text-foreground">
            Carry timing overlay on cross_asset_carry
          </span>
          <Tip content={<>
            Why 3 panels instead of 11 steps? <b>Pick → Run → Decide</b> is
            the quant's mental model. The old wizard was a checklist; this
            is a workspace — each step's controls live inline, no page
            navigation.
          </>}>
            <span className="text-[10px] text-muted/60 cursor-help underline-offset-2">
              why this shape?
            </span>
          </Tip>
          <span className="ml-auto text-[11px] text-muted tnum">
            {n_done} / 3 panels done
          </span>
        </div>

        {/* ─── ① PICK ───────────────────────────────────── */}
        <PanelHeader
          n={1}
          title="Pick a Carry candidate"
          hint="Filter is locked to CARRY + data we have on disk. Multi-select OK."
          done={pickDone}
          expanded={pickExpanded}
          onToggle={() => setPickExpanded((v) => !v)}
        />
        <AnimatePresence initial={false}>
          {pickExpanded && (
            <motion.div key="pick" {...PANEL_ANIM} className="border-b border-border/30">
              <PickPanel
                vectors={vectors}
                loading={vectorsLoading}
                error={vectorsError}
                picked={picked}
                onTogglePicked={(hid) => setPicked((cur) => {
                  const n = new Set(cur);
                  if (n.has(hid)) n.delete(hid); else n.add(hid);
                  return n;
                })}
                onClear={() => setPicked(new Set())}
                approving={approving}
                approveError={pickError}
                onApproveAndRun={approveAndRun}
                autoCandidateAvailable={Boolean(autoCandidate)}
                onReload={reloadVectors}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── ② RUN ────────────────────────────────────── */}
        <PanelHeader
          n={2}
          title="Run the pipeline"
          hint="Graveyard check + strict-gate funnel inline. Stream is live."
          done={runDone}
          expanded={runExpanded}
          onToggle={() => setRunExpanded((v) => !v)}
          disabled={!pickDone}
        />
        <AnimatePresence initial={false}>
          {runExpanded && pickDone && (
            <motion.div key="run" {...PANEL_ANIM} className="border-b border-border/30">
              {/* F6.2 (2026-06-05): build-path provenance banner. The user
                  needs to know WHICH parquet is feeding the SSE — typed
                  Composer / family builder / legacy string-match — and
                  if the answer is legacy, it's load-bearing for honesty
                  that the verdict below corresponds to a generic
                  parquet, NOT the picked hypothesis's spec. */}
              {buildPath && (
                <div className={cn(
                  "px-3 py-1.5 border-b border-border/30 text-[10.5px] flex items-center gap-2",
                  buildPath === "composer"            && "bg-ok/[0.06] text-ok",
                  buildPath === "series_factory"     && "bg-info/[0.08] text-info",
                  buildPath === "legacy_string_match" && "bg-warn/[0.10] text-warn",
                  buildPath === "claude_handoff"     && "bg-danger/[0.10] text-danger",
                )}>
                  <span className="font-semibold uppercase tracking-wider text-[9px]">
                    {buildPath === "composer"            && "✓ Composer"}
                    {buildPath === "series_factory"     && "✓ series_factory"}
                    {buildPath === "legacy_string_match" && "⚠ Legacy fallback"}
                    {buildPath === "claude_handoff"     && "✗ Build failed"}
                  </span>
                  {buildPathNote && (
                    <span className="text-foreground/80 truncate">{buildPathNote}</span>
                  )}
                </div>
              )}
              <RunPanel
                family={RECIPE_FAMILY}
                pipelineCandidates={pipelineCandidates}
                pickedCandidate={pickedCandidate}
                onPickCandidate={(c) => setPickedCandidate(c)}
                onStart={() => pickedCandidate && startStream(pickedCandidate)}
                activeHypId={activeHypId}
                steps={steps}
                finalDecision={finalDecision}
                streamError={streamError}
                running={running}
                activeSessionId={activeSessionId}
                funnelOpen={funnelOpen}
                onToggleFunnel={() => setFunnelOpen((v) => !v)}
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* ─── ③ DECIDE ─────────────────────────────────── */}
        <PanelHeader
          n={3}
          title="Decide the next loop"
          hint="Latest verdict + adjacent untested ideas, ready in one click."
          done={decideDone}
          expanded={decideExpanded}
          onToggle={() => setDecideExpanded((v) => !v)}
          disabled={!runDone && !recentLesson}
        />
        <AnimatePresence initial={false}>
          {decideExpanded && (
            <motion.div key="decide" {...PANEL_ANIM}>
              <DecidePanel
                family={RECIPE_FAMILY}
                lesson={recentLesson}
                finalDecision={finalDecision}
                onRestart={restartLoop}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </Card>

      {state.loading && (
        <div className="text-[11px] text-muted/70">Polling state…</div>
      )}
      <p className="text-[10.5px] text-muted/60 leading-snug">
        Compressed from the prior 11-step rail. 3 panels = the quant mental
        model. Old wizard available via git history if you miss it.
      </p>
    </div>
  );
}


// ── Panel header (shared shell) ──────────────────────────────────


const PANEL_ANIM = {
  initial:    { height: 0, opacity: 0 },
  animate:    { height: "auto", opacity: 1 },
  exit:       { height: 0, opacity: 0 },
  transition: { duration: 0.18 },
  style:      { overflow: "hidden" as const },
};


function PanelHeader({
  n, title, hint, done, expanded, onToggle, disabled,
}: {
  n: number;
  title: string;
  hint: string;
  done: boolean;
  expanded: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onToggle}
      disabled={disabled}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-3 text-left transition-colors",
        disabled
          ? "opacity-50 cursor-not-allowed"
          : expanded
            ? "bg-panel2/30"
            : "hover:bg-panel2/30",
      )}>
      <span className={cn(
        "shrink-0 inline-flex items-center justify-center h-6 w-6 rounded-full text-[11px] font-semibold",
        done ? "bg-ok/20 text-ok" : "bg-panel2/60 text-muted",
      )}>
        {done ? <CheckCircle2 className="h-4 w-4" strokeWidth={2.2} /> : n}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[12.5px] font-semibold text-foreground/90">{title}</div>
        <div className="text-[10.5px] text-muted/70 leading-snug">{hint}</div>
      </div>
      {expanded
        ? <ChevronUp className="h-4 w-4 text-muted/60 shrink-0" />
        : <ChevronDown className="h-4 w-4 text-muted/60 shrink-0" />}
    </button>
  );
}


// ── ① PICK panel ─────────────────────────────────────────────────


function PickPanel({
  vectors, loading, error, picked, onTogglePicked, onClear,
  approving, approveError, onApproveAndRun,
  autoCandidateAvailable, onReload,
}: {
  vectors:                 ForwardVector[];
  loading:                 boolean;
  error:                   string | null;
  picked:                  Set<string>;
  onTogglePicked:          (hid: string) => void;
  onClear:                 () => void;
  approving:               boolean;
  approveError:            string | null;
  onApproveAndRun:         () => void;
  autoCandidateAvailable:  boolean;
  onReload:                () => void;
}) {
  const nPicked = picked.size;
  // The first picked hypothesis's id drives the SpecPreviewCard (B.4).
  // Multi-pick is OK for approve-and-run but the preview shows the
  // primary one to keep the panel scannable.
  const firstPickedId = nPicked > 0 ? Array.from(picked)[0] : null;
  return (
    <div className="p-4 space-y-3">
      {/* B.4 2026-06-05 — show the typed HypothesisSpec for the picked
          candidate so the user sees EXACTLY what will be tested before
          approving. spec_hash is the load-bearing reproducibility id. */}
      {firstPickedId && (
        <SpecPreviewCard hypothesisId={firstPickedId} />
      )}
      {error && (
        <div className="rounded border border-danger/40 bg-danger/5 px-3 py-2 text-[11.5px] text-danger">
          {error}
        </div>
      )}

      {loading ? (
        <div className="text-[11px] text-muted">Loading carry candidates…</div>
      ) : vectors.length === 0 ? (
        <div className="rounded border border-border/40 bg-panel2/30 px-3 py-2.5 text-[11.5px] text-muted">
          No open CARRY hypotheses with data on disk.{" "}
          <Link href="/research/papers/new" className="text-accent hover:underline">
            Ingest a paper
          </Link>{" "}
          or{" "}
          <Link href="/research/forward" className="text-accent hover:underline">
            broaden filter on /research/forward
          </Link>.
        </div>
      ) : (
        <ul className="space-y-1.5 max-h-[360px] overflow-y-auto pr-1">
          {vectors.map((v) => {
            const isPicked = picked.has(v.source_hypothesis_id);
            return (
              <li key={v.source_hypothesis_id}>
                <label className={cn(
                  "flex items-start gap-2.5 rounded-md px-2.5 py-2 border transition-colors cursor-pointer",
                  isPicked
                    ? "border-accent/40 bg-accent/[0.06]"
                    : "border-border/30 hover:bg-panel2/40",
                )}>
                  <input
                    type="checkbox"
                    className="mt-0.5 shrink-0"
                    checked={isPicked}
                    onChange={() => onTogglePicked(v.source_hypothesis_id)}
                    disabled={approving}
                  />
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-1.5 mb-0.5">
                      <Badge className={cn(
                        "text-[10px]",
                        v.priority === "high"
                          ? "bg-danger/15 text-danger"
                          : v.priority === "medium"
                            ? "bg-warn/15 text-warn"
                            : "bg-muted/15 text-muted",
                      )}>
                        {v.priority}
                      </Badge>
                      <code className="text-[10.5px] text-foreground/70">
                        {v.mechanism_subtype}
                      </code>
                      <Badge className="text-[10px] bg-ok/15 text-ok">data=have</Badge>
                    </div>
                    <div className="text-[12px] text-foreground/90 leading-snug line-clamp-2">
                      {v.claim}
                    </div>
                    <div className="text-[10.5px] text-muted/70 mt-0.5 truncate">
                      ↗ {v.paper_title}
                    </div>
                  </div>
                </label>
              </li>
            );
          })}
        </ul>
      )}

      {/* Action bar */}
      <div className="flex items-center gap-2 pt-2 border-t border-border/30">
        <span className="text-[11px] text-muted">
          {nPicked === 0
            ? "Pick 1 to test, or 2-3 to batch."
            : `${nPicked} selected.`}
        </span>
        {nPicked > 0 && (
          <button onClick={onClear}
            disabled={approving}
            className="text-[11px] text-muted hover:text-foreground">
            clear
          </button>
        )}
        <button onClick={onReload}
          disabled={approving}
          className="text-[11px] text-muted hover:text-foreground">
          ↻ reload
        </button>
        <button
          onClick={onApproveAndRun}
          disabled={nPicked === 0 || approving}
          className={cn(
            "ml-auto inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-[12px] font-semibold transition-colors",
            nPicked > 0 && !approving
              ? "bg-accent text-background hover:bg-accent/90"
              : "bg-muted/20 text-muted/60 cursor-not-allowed",
          )}>
          {approving
            ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Approving + opening session…</>
            : autoCandidateAvailable
              ? <>Approve &amp; Run <ArrowRight className="h-3 w-3" /></>
              : <>Approve &amp; open Run panel <ArrowRight className="h-3 w-3" /></>}
        </button>
      </div>

      {approveError && (
        <div className="rounded border border-danger/40 bg-danger/5 px-3 py-2 text-[11.5px] text-danger">
          {approveError}
        </div>
      )}

      <Tip content={<>
        <code>Approve &amp; Run</code> does 4 things in one click: marks
        each picked vector <b>approved</b>, opens a <b>research_new</b>{" "}
        session, files a <b>research_test</b> intent the Claude poll hook
        sees, and {autoCandidateAvailable
          ? <>starts the pipeline SSE stream against the matching CARRY parquet.</>
          : <>opens the RUN panel for series selection (no CARRY parquet auto-matched on disk).</>
        }
      </>}>
        <span className="text-[10.5px] text-muted/60 cursor-help underline-offset-2">
          what does "Approve &amp; Run" actually do?
        </span>
      </Tip>
    </div>
  );
}


// ── ② RUN panel ──────────────────────────────────────────────────


function RunPanel({
  family, pipelineCandidates, pickedCandidate, onPickCandidate,
  onStart, activeHypId, steps, finalDecision, streamError,
  running, activeSessionId, funnelOpen, onToggleFunnel,
}: {
  family:                string;
  pipelineCandidates:    PipelineCandidate[];
  pickedCandidate:       PipelineCandidate | null;
  onPickCandidate:       (c: PipelineCandidate) => void;
  onStart:               () => void;
  activeHypId:           string | null;
  steps:                 StepEvent[];
  finalDecision:         FinalEvent | null;
  streamError:           string | null;
  running:               boolean;
  activeSessionId:       string | null;
  funnelOpen:            boolean;
  onToggleFunnel:        () => void;
}) {
  // Brief Claude reads after the handoff fires — paste-ready, includes
  // the session_id so emits auto-correlate.
  const handoffPrompt =
    `Goal: run the candidate pipeline on the approved ${family} hypothesis ` +
    `and emit factor_verdict_filed when done.\n\n` +
    `Session: ${activeSessionId ?? "(none)"}\n` +
    `Hypothesis: ${activeHypId ?? "(none)"}\n` +
    `Family: ${family}\n\n` +
    `If no returns parquet exists for this family yet, build it first ` +
    `(engine.portfolio.<family>_sleeve or equivalent), then route through ` +
    `engine.research.candidate_pipeline_v2.`;
  return (
    <div className="p-4 space-y-3">
      {/* Active session badge */}
      {activeSessionId && (
        <div className="text-[11px] text-muted">
          Session{" "}
          <code className="text-ok">{activeSessionId.slice(0, 8)}</code>{" "}
          live · events emitted from here will auto-tag.
        </div>
      )}

      {/* Graveyard inline check */}
      <GraveyardCheckInline family={family} proposalName={pickedCandidate?.proposal_name ?? null} />

      {/* Strict-gate funnel — collapsible */}
      <Card className="p-0 overflow-hidden">
        <button onClick={onToggleFunnel}
          className="w-full flex items-center gap-2 px-3 py-2 hover:bg-panel2/40 transition-colors border-b border-border/30">
          <FilterIcon className="h-3.5 w-3.5 text-accent" strokeWidth={2} />
          <span className="text-[12px] font-semibold">Strict-gate funnel</span>
          <span className="text-[10px] text-muted/70">historical pass / fail per step</span>
          {funnelOpen
            ? <ChevronUp className="ml-auto h-4 w-4 text-muted" />
            : <ChevronDown className="ml-auto h-4 w-4 text-muted" />}
        </button>
        {funnelOpen && (
          <div className="p-3">
            <StrictGateFunnel />
          </div>
        )}
      </Card>

      {/* Series picker — filtered to the active hypothesis's family.
          Honest UX: never show parquets from unrelated families just
          because they exist on disk. If 0 family matches → don't show
          the dropdown at all; surface "no series built" + Claude
          handoff as the only legitimate path. Fixed 2026-06-05 after
          user pushback: "这个 pick a parquet 是什么东西需要我自己选吗" */}
      {!finalDecision && (() => {
        const familyMatches = pipelineCandidates.filter((c) =>
          c.family.toLowerCase().includes(family.toLowerCase())
          || family.toLowerCase().includes(c.family.toLowerCase()));
        const hasMatch = familyMatches.length > 0;
        return (
          <div className={cn(
            "rounded border px-3 py-2.5 space-y-2",
            hasMatch
              ? "border-border/30 bg-panel2/30"
              : "border-warn/30 bg-warn/[0.04]",
          )}>
            <div className="text-[11px] font-semibold text-foreground/85">
              {hasMatch
                ? `Returns series · ${familyMatches.length} matching '${family}'`
                : `No ${family} series built on disk yet`}
            </div>
            {hasMatch ? (
              familyMatches.length === 1 ? (
                <div className="text-[11px] text-foreground/85">
                  Auto-matched: <code className="text-accent">{familyMatches[0].proposal_name}</code>
                  <span className="text-muted/70"> · {familyMatches[0].family}</span>
                </div>
              ) : (
                <select
                  value={pickedCandidate?.proposal_name ?? ""}
                  onChange={(e) => {
                    const c = familyMatches.find((x) => x.proposal_name === e.target.value);
                    if (c) onPickCandidate(c);
                  }}
                  disabled={running}
                  className="w-full bg-panel border border-border/40 rounded px-2 py-1 text-[11.5px]">
                  <option value="">— choose one of {familyMatches.length} family matches —</option>
                  {familyMatches.map((c) => (
                    <option key={c.proposal_name} value={c.proposal_name}>
                      {c.proposal_name} · {c.family}
                    </option>
                  ))}
                </select>
              )
            ) : (
              <div className="space-y-2">
                <div className="text-[11px] text-muted leading-snug">
                  The {pipelineCandidates.length} cached parquets are all
                  <b> unrelated families</b> — running pipeline on them would test
                  those families, not the {family} hypothesis you picked.
                </div>
                <SeriesAutoBuildButton
                  family={family}
                  hypothesisId={activeHypId || ""}
                  onBuilt={(p) => {
                    // Synthesize a PipelineCandidate from the built parquet
                    const built: PipelineCandidate = {
                      series_path:    p,
                      proposal_name:  `built_${(activeHypId || "h").slice(0, 12)}`,
                      proposed_role:  "alpha_seeker",
                      mechanism_id:   activeHypId || "unknown",
                      family,
                    };
                    onPickCandidate(built);
                  }}
                />
              </div>
            )}
            <div className="flex items-center gap-2 pt-1">
              {hasMatch && (
                <button
                  onClick={() => {
                    // Auto-select if there's exactly 1 match
                    if (familyMatches.length === 1 && !pickedCandidate) {
                      onPickCandidate(familyMatches[0]);
                    }
                    onStart();
                  }}
                  disabled={running || (!pickedCandidate && familyMatches.length > 1)}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-[11.5px] font-semibold",
                    (pickedCandidate || familyMatches.length === 1) && !running
                      ? "bg-accent text-background hover:bg-accent/90"
                      : "bg-muted/20 text-muted/60 cursor-not-allowed",
                  )}>
                  {running
                    ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Streaming…</>
                    : <><PlayCircle className="h-3.5 w-3.5" /> Run pipeline</>}
                </button>
              )}
            <HandoffToClaude
              intent={{
                kind:         "explore_hypothesis",
                subject_type: "hypothesis",
                subject_id:   activeHypId ?? "(unknown)",
                source_page:  "/research/enhance",
                payload:      {
                  session_id: activeSessionId,
                  family,
                  ask:        `Run the candidate pipeline on the approved ${family} hypothesis; emit factor_verdict_filed when done.`,
                },
              }}
              prompt={handoffPrompt}
              label="Hand off to Claude"
            />
            <span className="text-[10.5px] text-muted/70">
              {hasMatch
                ? "Claude handles cases where you need a custom series or deeper review."
                : `Claude will build the ${family} returns series from this hypothesis spec.`}
            </span>
            </div>
          </div>
        );
      })()}

      {/* Stream events */}
      {(steps.length > 0 || running) && (
        <div className="rounded border border-border/30 bg-panel2/20 p-3 space-y-1.5">
          <div className="text-[10.5px] uppercase tracking-wider text-muted/80">
            Pipeline steps
          </div>
          {steps.map((s, i) => (
            <motion.div key={i}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.16 }}
              className="flex items-center gap-2 text-[11.5px]">
              <Badge className={cn("shrink-0", STATUS_TONE[s.status] || "bg-muted/15 text-muted")}>
                {s.status}
              </Badge>
              <code className="text-foreground/85">{s.step_name}</code>
              <span className="text-muted/80 truncate">{s.verdict}</span>
            </motion.div>
          ))}
          {running && (
            <div className="inline-flex items-center gap-1.5 text-[11px] text-muted">
              <Loader2 className="h-3 w-3 animate-spin" /> awaiting next step…
            </div>
          )}
        </div>
      )}

      {streamError && (
        <div className="rounded border border-danger/40 bg-danger/5 px-3 py-2 text-[11.5px] text-danger">
          {streamError}
        </div>
      )}

      {/* Final decision */}
      {finalDecision && (
        <div className={cn(
          "rounded border p-3 space-y-1",
          finalDecision.final_decision.includes("PROMOTE")
            ? "border-ok/40 bg-ok/5"
            : finalDecision.final_decision === "HARD_REJECT"
              ? "border-danger/40 bg-danger/5"
              : "border-warn/40 bg-warn/5",
        )}>
          <div className="flex items-center gap-2">
            <span className="text-[10.5px] uppercase tracking-wider text-muted/80">Final</span>
            <code className="text-[12.5px] font-semibold">
              {finalDecision.final_decision}
            </code>
          </div>
          <div className="text-[11.5px] text-foreground/85 leading-snug">
            {finalDecision.rationale}
          </div>
          {finalDecision.candidate_relation && (
            <div className="text-[10.5px] text-muted/70">
              relation: <code>{finalDecision.candidate_relation}</code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ── Graveyard inline check (compact variant for the workspace) ──


function GraveyardCheckInline({
  family, proposalName,
}: { family: string | null; proposalName: string | null }) {
  const [lessons, setLessons] = useState<LessonMini[]>([]);
  const [loading, setLoading] = useState(false);
  const [hidden, setHidden]   = useState(false);

  useEffect(() => {
    if (!family || hidden) { setLessons([]); return; }
    const params = new URLSearchParams({
      mechanism_family: family,
      verdict:          "red",
      include_legacy:   "true",
      limit:            "8",
    });
    setLoading(true);
    fetch(`${API_BASE}/api/paper_chain/lessons?${params}`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data: LessonMini[]) => setLessons(data || []))
      .catch(() => setLessons([]))
      .finally(() => setLoading(false));
  }, [family, hidden]);

  if (hidden || !family) return null;
  if (loading) return null;

  if (lessons.length === 0) {
    return (
      <div className="rounded border border-ok/30 bg-ok/[0.04] px-3 py-2 text-[11.5px] flex items-center gap-2">
        <Skull className="h-3.5 w-3.5 text-ok/70" />
        <span className="text-muted">
          No past RED verdicts on family <code className="text-ok">{family}</code>. Pipeline can proceed.
        </span>
      </div>
    );
  }

  const candidateTokens = new Set(
    (proposalName || "").toLowerCase().split(/[_\s-]+/).filter((t) => t.length >= 3)
  );
  const flaggedOverlap = lessons.filter((L) => {
    const lt = new Set(L.candidate_name.toLowerCase().split(/[_\s-]+/).filter((t) => t.length >= 3));
    let n = 0; candidateTokens.forEach((t) => { if (lt.has(t)) n++; });
    return n >= 2;
  });

  return (
    <div className="rounded border border-danger/40 bg-danger/[0.04] px-3 py-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <Skull className="h-4 w-4 text-danger shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-[10px] uppercase tracking-wider text-danger/80">
            Graveyard check
          </div>
          <div className="text-[12.5px] font-semibold text-danger">
            {lessons.length} past RED verdict{lessons.length === 1 ? "" : "s"} on family <code>{family}</code>
            {flaggedOverlap.length > 0 && " · name-overlap"}
          </div>
        </div>
        <button onClick={() => setHidden(true)}
          className="text-[10px] text-muted/70 hover:text-foreground">dismiss</button>
      </div>
      {flaggedOverlap.length > 0 && (
        <div className="flex items-start gap-2 rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-[11px] text-warn">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span>
            High-overlap kill candidate{flaggedOverlap.length === 1 ? "" : "s"}:{" "}
            {flaggedOverlap.slice(0, 3).map((L) => L.candidate_name).join(", ")}
            {flaggedOverlap.length > 3 && " …"}. Read before running.
          </span>
        </div>
      )}
      <ul className="space-y-1 text-[11px]">
        {lessons.slice(0, 3).map((L) => (
          <li key={L.lesson_id}
              className="flex items-start gap-2 rounded border border-border/30 bg-panel2/30 px-2 py-1">
            <Badge className="bg-danger/15 text-danger shrink-0 text-[10px]">
              {L.verdict.slice(0, 16)}
            </Badge>
            <div className="flex-1 min-w-0">
              <code className="text-[10.5px] text-foreground/90">{L.candidate_name}</code>
              <span className="text-muted/70"> · {L.summary.slice(0, 80)}</span>
            </div>
            <Link href={`/research/lessons/${L.lesson_id}`} target="_blank"
              className="text-muted/70 hover:text-accent shrink-0">
              <ExternalLink className="h-3 w-3" />
            </Link>
          </li>
        ))}
        {lessons.length > 3 && (
          <Link href={`/research/lessons?mechanism_family=${family}&verdict=red&include_legacy=true`}
            className="text-[10.5px] text-accent hover:underline">
            view all {lessons.length} →
          </Link>
        )}
      </ul>
    </div>
  );
}


// ── ③ DECIDE panel ────────────────────────────────────────────────


function DecidePanel({
  family, lesson, finalDecision, onRestart,
}: {
  family:        string;
  lesson:        { lesson_id: string; verdict: string } | null;
  finalDecision: FinalEvent | null;
  onRestart:     () => void;
}) {
  return (
    <div className="p-4 space-y-3">
      {finalDecision && (
        <div className="rounded border border-border/30 bg-panel2/20 p-3 text-[11.5px] space-y-1">
          <div className="flex items-center gap-2">
            <span className="text-[10.5px] uppercase tracking-wider text-muted/70">
              This loop's pipeline verdict
            </span>
            <code className="text-foreground/90">{finalDecision.final_decision}</code>
          </div>
          <div className="text-foreground/80 leading-snug line-clamp-3">
            {finalDecision.rationale}
          </div>
        </div>
      )}

      {lesson ? (
        <div className="rounded border border-ok/30 bg-ok/[0.04] p-3 text-[11.5px] space-y-2">
          <div className="flex items-center gap-2">
            <Compass className="h-4 w-4 text-ok" />
            <span className="text-[10.5px] uppercase tracking-wider text-ok/80">
              Latest carry lesson
            </span>
            <code className="text-ok">{lesson.verdict}</code>
          </div>
          <Link href={`/research/lessons/${lesson.lesson_id}`}
            className="inline-flex items-center gap-1.5 rounded bg-accent text-background hover:bg-accent/90 px-2.5 py-1 text-[11px] font-semibold">
            Open lesson detail (with AdjacentActions)
            <ArrowRight className="h-3 w-3" />
          </Link>
          <div className="text-[10.5px] text-muted/70">
            The lesson page surfaces same-subtype + same-family untested
            counts as one-click buttons. Pick one → it filters Forward.
          </div>
        </div>
      ) : (
        <div className="rounded border border-border/30 bg-panel2/30 px-3 py-2.5 text-[11.5px] text-muted">
          No carry lesson landed yet in the last 24h. Once Claude emits{" "}
          <code>factor_verdict_filed</code> this panel auto-fills.
        </div>
      )}

      <div className="flex items-center gap-2 pt-2 border-t border-border/30">
        <Link href={`/research/lessons?mechanism_family=${family}`}
          className="text-[11px] text-accent hover:underline inline-flex items-center gap-1">
          Browse all carry lessons <ChevronRight className="h-3 w-3" />
        </Link>
        <button onClick={onRestart}
          className="ml-auto inline-flex items-center gap-1.5 rounded border border-accent/40 bg-accent/5 text-accent hover:bg-accent/15 px-3 py-1.5 text-[11.5px]">
          <RotateCcw className="h-3.5 w-3.5" /> Start next loop
        </button>
      </div>
    </div>
  );
}


// ── SeriesAutoBuildButton ────────────────────────────────────────


function SeriesAutoBuildButton({
  family, hypothesisId, onBuilt,
}: {
  family:       string;
  hypothesisId: string;
  onBuilt:      (path: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr]   = useState<string | null>(null);
  const [hasBuilder, setHasBuilder] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API_BASE}/api/series_factory/families`, { cache: "no-store" })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d || cancelled) return;
        const fams = (d.families || []) as string[];
        setHasBuilder(fams.some(f => f.toLowerCase() === family.toLowerCase()));
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [family]);

  const handleBuild = async () => {
    if (!hypothesisId) return;
    setBusy(true); setErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/series_factory/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ family, hypothesis_id: hypothesisId }),
      });
      const data = await r.json();
      if (!r.ok || !data.ok) {
        throw new Error(data.error || `HTTP ${r.status}`);
      }
      onBuilt(data.path);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  if (hasBuilder === false) {
    return (
      <div className="text-[10.5px] text-muted/80 italic">
        No registered builder for family <code>{family}</code>.
        Use Claude handoff below to build the series manually.
      </div>
    );
  }
  if (hasBuilder === null) return null;

  return (
    <div className="space-y-1">
      <button
        onClick={handleBuild}
        disabled={busy || !hypothesisId}
        className={cn(
          "inline-flex items-center gap-1.5 rounded px-3 py-1.5 text-[11.5px] font-semibold",
          busy
            ? "bg-muted/20 text-muted/60 cursor-not-allowed"
            : "bg-accent text-background hover:bg-accent/90",
        )}>
        {busy
          ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Building {family} series…</>
          : <><PlayCircle className="h-3.5 w-3.5" /> Build {family} series for this hypothesis</>}
      </button>
      {err && (
        <div className="text-[10.5px] text-danger">
          build failed: {err}
        </div>
      )}
      <div className="text-[10px] text-muted/60 leading-snug">
        Calls engine.series_factory.build({family}, {hypothesisId.slice(0, 8)}…) →
        caches parquet → enables Run pipeline below.
      </div>
    </div>
  );
}
