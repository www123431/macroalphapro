"use client";

// Phase A.4 — live candidate_pipeline streaming view. Consumes SSE
// from /api/pipeline/stream and renders each step's verdict as it
// completes, plus final meta_decision card.
//
// 2026-06-04: prefixed with GraveyardCheck — when a candidate is
// selected, the same mechanism_family's past RED verdicts surface
// inline. Stops "test new mechanism without checking graveyard" being
// a separate manual step the user might skip.

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Skull, AlertTriangle, ExternalLink, ChevronDown, ChevronUp, Filter } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge } from "@/components/ui";
import { StrictGateFunnel } from "@/components/StrictGateFunnel";

type Candidate = {
  series_path: string;
  proposal_name: string;
  proposed_role: string;
  mechanism_id: string;
  family: string;
  // F3 (2026-06-05): paper-grounded candidates carry hypothesis_id
  // instead of a pre-built series_path. When present, the runner first
  // POSTs /api/pipeline/prepare-from-fv to compose the parquet, then
  // opens the returned stream_url. track="pre_baked_legacy" uses the
  // existing direct series_path flow.
  source_hypothesis_id?: string;
  track:                "pre_baked_legacy" | "paper_grounded";
  // Decoration shown in the dropdown + selection summary
  claim?:               string;
  signal_type?:         string;
  composer_status?:     "ready" | "missing_components" | "no_spec" | "not_factor";
  composer_gaps?:       Array<{ role: string; expected_key: string; reason: string }>;
  graveyard_verdict?:   "CLEAN" | "WARN" | "RISK";
  direction_score?:     number;
};

type RankedFV = {
  forward_vector_id:     string;
  source_hypothesis_id:  string;
  paper_title:           string;
  claim:                 string;
  family:                string;
  signal_type:           string;
  asset_class:           string;
  composer_status:       "ready" | "missing_components" | "no_spec" | "not_factor";
  composer_gaps:         Array<{ role: string; expected_key: string; reason: string }>;
  graveyard_verdict:     "CLEAN" | "WARN" | "RISK";
  graveyard_top_match:   string | null;
  direction_score:       number;
  direction_rank:        number | null;
};

type StepEvent = {
  node: string;
  step_name: string;
  status: "PASS" | "WARN" | "FAIL" | "SKIP" | "INFO";
  verdict: string;
};

type FinalEvent = {
  final_decision: string;
  rationale: string;
  candidate_relation: string;
  most_correlated_sleeve: string | null;
  most_correlated_value: number | null;
};

const STATUS_TONE: Record<string, string> = {
  PASS: "bg-ok/15 text-ok",
  WARN: "bg-warn/15 text-warn",
  FAIL: "bg-danger/15 text-danger",
  SKIP: "bg-muted/15 text-muted",
  INFO: "bg-info/15 text-info",
};


// ─── Graveyard inline check ──────────────────────────────────────
// Renders "similar past RED verdicts on the same mechanism family" so
// the user notices "we already killed this idea" BEFORE running the
// pipeline. Read-only; the user can dismiss but can't skip seeing.

type LessonMini = {
  lesson_id:           string;
  candidate_name:      string;
  verdict:             string;
  mechanism_family:    string;
  mechanism_subtype:   string;
  failure_modes:       string[];
  grounding_method:    string;
  summary:             string;
  created_ts:          string;
};

function GraveyardCheck({
  family, proposalName,
}: { family: string | null; proposalName: string | null }) {
  const [lessons, setLessons] = useState<LessonMini[]>([]);
  const [loading, setLoading] = useState(false);
  const [hidden, setHidden]   = useState(false);

  useEffect(() => {
    if (!family || family === "unknown" || hidden) { setLessons([]); return; }
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

  if (hidden || !family || family === "unknown") return null;
  if (loading) return null;
  if (lessons.length === 0) {
    return (
      <Card className="border-ok/30 bg-ok/[0.04] py-2.5 px-3">
        <div className="flex items-center gap-2 text-[12px]">
          <Skull className="h-3.5 w-3.5 text-ok/70 shrink-0" />
          <span className="text-muted">
            No past RED verdicts on family <code className="text-ok">{family}</code>.
            Pipeline can proceed.
          </span>
        </div>
      </Card>
    );
  }

  // Cheap candidate-name overlap heuristic — token bag intersection.
  // If the candidate name shares ≥2 tokens with a past RED candidate,
  // flag it as HIGH OVERLAP.
  const candidateTokens = new Set(
    (proposalName || "").toLowerCase().split(/[_\s-]+/).filter((t) => t.length >= 3)
  );
  const flaggedOverlap = lessons.filter((L) => {
    const lessonTokens = new Set(
      L.candidate_name.toLowerCase().split(/[_\s-]+/).filter((t) => t.length >= 3)
    );
    let n = 0;
    candidateTokens.forEach((t) => { if (lessonTokens.has(t)) n++; });
    return n >= 2;
  });

  return (
    <Card className="border-danger/40 bg-danger/[0.04] py-3 px-4 space-y-2">
      <div className="flex items-center gap-2">
        <Skull className="h-4 w-4 text-danger shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-[10px] uppercase tracking-wider text-danger/80">
            Graveyard check
          </div>
          <div className="text-[13px] font-semibold text-danger">
            {lessons.length} past RED verdict{lessons.length === 1 ? "" : "s"} on family{" "}
            <code>{family}</code>
            {flaggedOverlap.length > 0 && " · name-overlap detected"}
          </div>
        </div>
        <button onClick={() => setHidden(true)}
          className="text-[10px] text-muted/70 hover:text-foreground">
          dismiss
        </button>
      </div>

      {flaggedOverlap.length > 0 && (
        <div className="flex items-start gap-2 rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-[11px] text-warn">
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <span>
            High-overlap kill candidate{flaggedOverlap.length === 1 ? "" : "s"}:{" "}
            {flaggedOverlap.slice(0, 3).map((L) => L.candidate_name).join(", ")}
            {flaggedOverlap.length > 3 && " …"}. Read these BEFORE running.
          </span>
        </div>
      )}

      <ul className="space-y-1 text-[11.5px]">
        {lessons.slice(0, 5).map((L) => (
          <li key={L.lesson_id}
              className="flex items-start gap-2 rounded border border-border/30 bg-panel2/30 px-2 py-1.5">
            <Badge className="bg-danger/15 text-danger shrink-0">
              {L.verdict.slice(0, 16)}
            </Badge>
            <div className="flex-1 min-w-0">
              <div className="font-mono text-[11px] text-foreground/90">
                {L.candidate_name}
                {L.mechanism_subtype && (
                  <span className="text-muted/70"> · {L.mechanism_subtype}</span>
                )}
              </div>
              <div className="text-muted leading-snug line-clamp-2">
                {L.summary || L.failure_modes.slice(0, 2).join(", ")}
              </div>
            </div>
            <Link href={`/research/lessons/${L.lesson_id}`}
              className="text-muted/70 hover:text-accent shrink-0" target="_blank">
              <ExternalLink className="h-3 w-3" />
            </Link>
          </li>
        ))}
      </ul>

      {lessons.length > 5 && (
        <Link href={`/research/lessons?mechanism_family=${family}&verdict=red&include_legacy=true`}
              className="block text-[11px] text-accent hover:underline pt-1 border-t border-border/30">
          View all {lessons.length} RED verdicts on this family →
        </Link>
      )}
    </Card>
  );
}

// Next 16: useSearchParams() must live inside a Suspense boundary to
// avoid CSR-bailout-during-prerender. Default export wraps the inner
// component; everything below stays unchanged.
export default function CandidatePipelinePage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <CandidatePipelineInner />
    </Suspense>
  );
}

function CandidatePipelineInner() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState<StepEvent[]>([]);
  const [finalDecision, setFinalDecision] = useState<FinalEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  // V_new3 — historical strict-gate funnel. Collapsed by default;
  // expanding shows where past candidates died before the user
  // commits to a new run.
  const [funnelOpen, setFunnelOpen] = useState(false);
  const [pipelineMeta, setPipelineMeta] = useState<any>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const searchParams = useSearchParams();

  useEffect(() => {
    // F3 (2026-06-05): fetch BOTH tracks in parallel. Pre-baked 5 are
    // the legacy direct-parquet track; paper-grounded ones come from
    // the F1 ranked endpoint (joins fv + spec + composer + graveyard +
    // direction_score). Merging here keeps the dropdown a single
    // source of truth.
    Promise.all([
      fetch(`${API_BASE}/api/pipeline/candidates`).then((r) => r.json()),
      fetch(`${API_BASE}/api/paper_chain/forward-vectors/ranked?top=200`)
        .then((r) => r.ok ? r.json() : []),
    ])
      .then(([legacy, ranked]: [{ candidates: any[] }, RankedFV[]]) => {
        const legacyCands: Candidate[] = (legacy.candidates || []).map((c) => ({
          ...c,
          track: "pre_baked_legacy" as const,
        }));
        const paperCands: Candidate[] = (ranked || []).map((r) => ({
          // No series_path yet — prepare-from-fv builds it on demand
          series_path:          "",
          proposal_name:        `fv_${r.source_hypothesis_id.slice(0, 8)}`,
          proposed_role:        "alpha_seeker",
          mechanism_id:         r.signal_type,
          family:               r.family,
          source_hypothesis_id: r.source_hypothesis_id,
          track:                "paper_grounded" as const,
          claim:                r.claim,
          signal_type:          r.signal_type,
          composer_status:      r.composer_status,
          composer_gaps:        r.composer_gaps,
          graveyard_verdict:    r.graveyard_verdict,
          direction_score:      r.direction_score,
        }));
        setCandidates([...paperCands, ...legacyCands]);
      })
      .catch((e) => setError(`failed to load candidates: ${e}`));
  }, []);

  // Phase Lab-Step-A: prefill from URL params (link target from /lab/series).
  // If query carries series_path + proposal_name, synthesize a Candidate so
  // user can hit Run immediately — no manual dropdown selection.
  useEffect(() => {
    const sp = searchParams?.get("series_path");
    const pn = searchParams?.get("proposal_name");
    if (!sp || !pn) return;
    setSelected({
      series_path:   sp,
      proposal_name: pn,
      proposed_role: searchParams?.get("proposed_role") || "alpha_seeker",
      mechanism_id:  searchParams?.get("mechanism_id") || pn,
      family:        searchParams?.get("family") || "unknown",
      track:         "pre_baked_legacy",
    });
  }, [searchParams]);

  const startStream = async () => {
    if (!selected) return;
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setSteps([]);
    setFinalDecision(null);
    setError(null);
    setPipelineMeta(null);
    setRunning(true);

    // F3 (2026-06-05): paper-grounded path first POSTs /prepare-from-fv
    // which runs composer.compose to build the parquet, then returns the
    // ready-to-open stream_url. Pre-baked legacy path goes directly to
    // /stream with its hardcoded series_path.
    let streamUrl: string;
    if (selected.track === "paper_grounded" && selected.source_hypothesis_id) {
      try {
        const prepResp = await fetch(
          `${API_BASE}/api/pipeline/prepare-from-fv?source_hypothesis_id=${encodeURIComponent(selected.source_hypothesis_id)}`,
          { method: "POST" },
        );
        if (!prepResp.ok) {
          const err = await prepResp.json().catch(() => ({}));
          setError(`prepare-from-fv failed (${prepResp.status}): ${JSON.stringify(err.detail ?? err).slice(0, 280)}`);
          setRunning(false);
          return;
        }
        const prep = await prepResp.json();
        streamUrl = `${API_BASE}${prep.stream_url}`;
      } catch (e) {
        setError(`prepare-from-fv network error: ${e}`);
        setRunning(false);
        return;
      }
    } else {
      const params = new URLSearchParams({
        series_path:    selected.series_path,
        proposal_name:  selected.proposal_name,
        proposed_role:  selected.proposed_role,
        mechanism_id:   selected.mechanism_id,
        family:         selected.family,
      });
      streamUrl = `${API_BASE}/api/pipeline/stream?${params.toString()}`;
    }

    const es = new EventSource(streamUrl);
    eventSourceRef.current = es;

    es.addEventListener("pipeline_start", (e: MessageEvent) => {
      try {
        setPipelineMeta(JSON.parse(e.data));
      } catch {}
    });
    es.addEventListener("step_complete", (e: MessageEvent) => {
      try {
        const step = JSON.parse(e.data) as StepEvent;
        setSteps((prev) => [...prev, step]);
      } catch {}
    });
    es.addEventListener("pipeline_complete", (e: MessageEvent) => {
      try {
        const fin = JSON.parse(e.data) as FinalEvent;
        setFinalDecision(fin);
      } catch {}
      es.close();
      setRunning(false);
    });
    es.addEventListener("pipeline_error", (e: MessageEvent) => {
      try {
        const err = JSON.parse(e.data);
        setError(err.error || "unknown pipeline error");
      } catch {
        setError("pipeline error");
      }
      es.close();
      setRunning(false);
    });
    es.onerror = (ev) => {
      console.error("SSE error", ev);
      if (es.readyState === EventSource.CLOSED) {
        setRunning(false);
      }
    };
  };

  useEffect(() => () => eventSourceRef.current?.close(), []);

  const decisionTone =
    finalDecision?.final_decision?.includes("PROMOTE")
      ? "bg-ok/15 text-ok"
      : finalDecision?.final_decision === "HARD_REJECT"
        ? "bg-danger/15 text-danger"
        : "bg-warn/15 text-warn";

  return (
    <div className="space-y-4 p-6">
      <SectionTitle>Candidate Pipeline Live Stream</SectionTitle>

      {/* V_new3 — historical strict-gate funnel. Where the
          candidate has died in past audits. Helps the user judge
          which gates to expect to fail on a re-test. */}
      <Card className="p-0 overflow-hidden">
        <button onClick={() => setFunnelOpen((v) => !v)}
          className="w-full flex items-center gap-2 px-3 py-2 hover:bg-panel2/40 transition-colors border-b border-border/30">
          <Filter className="h-3.5 w-3.5 text-accent" strokeWidth={2} />
          <span className="text-[12px] font-semibold">Strict-gate funnel</span>
          <span className="text-[10px] text-muted/70">
            Historical pass / fail / skip per step · click to focus a step
          </span>
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

      {/* Graveyard inline check — surfaces past RED verdicts on the
          selected candidate's mechanism family BEFORE the user runs.
          Read-only; dismissable but unmissable on initial render. */}
      <GraveyardCheck
        family={selected?.family ?? null}
        proposalName={selected?.proposal_name ?? null} />

      <Card className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selected?.proposal_name ?? ""}
            onChange={(e) => {
              const c = candidates.find((x) => x.proposal_name === e.target.value);
              setSelected(c ?? null);
            }}
            disabled={running}
            className="rounded border border-muted/20 bg-panel2 text-foreground px-3 py-1.5 text-sm min-w-[420px] [&>option]:bg-panel2 [&>option]:text-foreground [&>optgroup]:bg-panel2 [&>optgroup]:text-muted"
          >
            <option value="">-- select candidate --</option>
            {/* F3: paper-grounded first (the 88 typed from B.2). Each option
                carries composer + graveyard state in the label so the user
                sees readiness without picking. */}
            {(() => {
              const paper = candidates.filter((c) => c.track === "paper_grounded");
              const ready = paper.filter((c) => c.composer_status === "ready").length;
              const legacy = candidates.filter((c) => c.track === "pre_baked_legacy");
              return (
                <>
                  {paper.length > 0 && (
                    <optgroup label={`Paper-grounded (${paper.length} · ${ready} ready)`}>
                      {paper.map((c) => {
                        const cs = c.composer_status === "ready"   ? "✓"
                                  : c.composer_status === "missing_components" ? "⚠"
                                  : "✗";
                        const gv = c.graveyard_verdict === "CLEAN"  ? "·"
                                  : c.graveyard_verdict === "WARN"  ? "△"
                                  : "✗";
                        return (
                          <option key={c.proposal_name} value={c.proposal_name}>
                            {cs}{gv} {c.family.slice(0, 14).padEnd(14)} {c.signal_type?.slice(0, 22)} — {(c.claim || "").slice(0, 60)}
                          </option>
                        );
                      })}
                    </optgroup>
                  )}
                  {legacy.length > 0 && (
                    <optgroup label={`Pre-baked legacy parquets (${legacy.length})`}>
                      {legacy.map((c) => (
                        <option key={c.proposal_name} value={c.proposal_name}>
                          ✓ {c.proposal_name} ({c.proposed_role}, {c.family})
                        </option>
                      ))}
                    </optgroup>
                  )}
                </>
              );
            })()}
          </select>
          <button
            onClick={startStream}
            disabled={!selected || running
                      || (selected.track === "paper_grounded"
                          && selected.composer_status !== "ready")}
            className="rounded bg-accent px-4 py-1.5 text-sm text-white disabled:opacity-50"
          >
            {running
              ? "Running..."
              : selected?.track === "paper_grounded"
                  && selected.composer_status !== "ready"
                ? `Blocked: ${selected.composer_status}`
                : "Run Pipeline"}
          </button>
        </div>
        {selected && selected.track === "paper_grounded" && (
          <div className="text-[11px] text-muted space-y-1">
            <div>
              <span className="text-foreground">composer:</span> {selected.composer_status}
              {" · "}
              <span className="text-foreground">graveyard:</span> {selected.graveyard_verdict}
              {" · "}
              <span className="text-foreground">direction_score:</span> {selected.direction_score?.toFixed(2)}
            </div>
            {selected.composer_status === "missing_components" && selected.composer_gaps && (
              <div className="text-warn">
                gaps: {selected.composer_gaps.slice(0, 4).map((g) => `${g.role}/${g.expected_key}`).join(", ")}
                {selected.composer_gaps.length > 4 ? ` +${selected.composer_gaps.length - 4} more` : ""}
              </div>
            )}
            {selected.claim && (
              <div className="text-muted/80">claim: {selected.claim.slice(0, 220)}{selected.claim.length > 220 ? "…" : ""}</div>
            )}
          </div>
        )}
        {selected && selected.track === "pre_baked_legacy" && (
          <div className="text-xs text-muted">
            series: {selected.series_path}
          </div>
        )}
      </Card>

      {pipelineMeta && (
        <Card>
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div>
              <span className="text-muted">n_months: </span>
              <span className="font-mono">{pipelineMeta.n_months}</span>
            </div>
            <div>
              <span className="text-muted">gross Sharpe: </span>
              <span className="font-mono">
                {pipelineMeta.gross_sharpe?.toFixed(3)}
              </span>
            </div>
            <div>
              <span className="text-muted">proposal: </span>
              <span className="font-mono">{pipelineMeta.proposal_name}</span>
            </div>
          </div>
        </Card>
      )}

      <Card>
        <SectionTitle>Pipeline Steps</SectionTitle>
        <div className="space-y-2">
          {steps.length === 0 && (
            <div className="text-sm text-muted">
              {running
                ? "Waiting for first step..."
                : "Click Run Pipeline to begin streaming."}
            </div>
          )}
          {steps.map((s, i) => (
            <div
              key={i}
              className="flex items-start gap-3 border-b border-muted/10 pb-2 last:border-0"
            >
              <Badge tone={STATUS_TONE[s.status] || "bg-muted/15 text-muted"}>
                {s.status}
              </Badge>
              <div className="flex-1">
                <div className="text-sm font-medium">{s.step_name}</div>
                <div className="text-xs text-muted">{s.verdict}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {finalDecision && (
        <Card className="space-y-3">
          <SectionTitle>Final Decision</SectionTitle>
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone={decisionTone}>{finalDecision.final_decision}</Badge>
            <Badge tone="bg-info/15 text-info">
              {finalDecision.candidate_relation}
            </Badge>
            {finalDecision.most_correlated_sleeve &&
              finalDecision.most_correlated_value !== null && (
                <span className="text-xs text-muted">
                  most_corr: {finalDecision.most_correlated_sleeve} ={" "}
                  {finalDecision.most_correlated_value.toFixed(3)}
                </span>
              )}
          </div>
          <div className="text-sm leading-relaxed">{finalDecision.rationale}</div>
        </Card>
      )}

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger">Error: {error}</div>
        </Card>
      )}
    </div>
  );
}
