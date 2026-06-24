"use client";

// /research/forward — paper-grounded forward research surface.
//
// Each row = one untested hypothesis from a real paper. User picks one,
// opens a research session against it. This is the canonical
// PAPER → HYPOTHESIS → TEST → VERDICT entry point.

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";
import { fileIntent } from "@/lib/intents";
import { Card, Badge } from "@/components/ui";
import PaperChainSearch from "@/components/PaperChainSearch";
import { ModeHeader } from "@/components/ModeHeader";
import { CompareBar, type CompareField } from "@/components/Compare";
import { Tip } from "@/components/Tip";
import { GraveyardCollisionChip } from "@/components/GraveyardCollisionChip";
import { NextStepHint } from "@/components/NextStepHint";
import { useI18n } from "@/lib/i18n";


type PMStatus = "extracted" | "reviewed" | "approved" | "rejected";

type DataStatus = "have" | "partial" | "missing" | "unknown";

type ForwardVector = {
  forward_vector_id:    string;
  source_paper_id:      string;
  paper_title:          string;
  source_hypothesis_id: string;
  claim:                string;
  mechanism_family:     string;
  mechanism_subtype:    string;
  predicted_direction:  string;
  predicted_magnitude:  string;
  required_data:        string[];
  priority:             "high" | "medium" | "low";
  priority_signals:     Record<string, unknown>;
  pm_status:            PMStatus;
  pm_reviewed_ts:       string | null;
  pm_reviewed_by:       string | null;
  pm_note:              string | null;
  data_status:          DataStatus;
  data_have:            string[];
  data_missing:         string[];
  // Stage C Tier B (2026-06-07): A's orthogonality statements vs the
  // anchor library. Empty for paper-rooted extractions (paper IS the
  // anchor) + pre-Phase-E synthesis rows.
  orthogonal_to_anchors: { anchor_paper_id: string; why_orthogonal: string }[];
};

const DATA_STATUS_TONE: Record<DataStatus, string> = {
  have:    "bg-ok/15 text-ok border-ok/40",
  partial: "bg-warn/15 text-warn border-warn/40",
  missing: "bg-danger/10 text-danger/80 border-danger/40",
  unknown: "bg-muted/15 text-muted border-muted/40",
};


const PM_STATUS_TONE: Record<PMStatus, string> = {
  approved:  "bg-ok/15 text-ok border-ok/40",
  reviewed:  "bg-info/15 text-info border-info/40",
  extracted: "bg-muted/15 text-muted border-muted/40",
  rejected:  "bg-danger/10 text-danger/80 border-danger/40",
};


const PRIORITY_TONE: Record<string, string> = {
  high:   "bg-danger/15 text-danger border-danger/40",
  medium: "bg-warn/15 text-warn border-warn/40",
  low:    "bg-muted/15 text-muted border-muted/40",
};

// Phase 2.1b track distinction — paper_stated (replicate/reinforce
// track) vs brainstorm (A's synthesis after B + principal approval)
// vs human_authored (manual escape hatch). Visual lane separation
// without splitting the queue page.
const TRACK_TONE: Record<string, string> = {
  paper_stated:   "bg-info/15 text-info border-info/40",
  brainstorm:     "bg-accent/15 text-accent border-accent/40",
  human_authored: "bg-muted/15 text-muted border-muted/40",
};
const TRACK_LABEL: Record<string, string> = {
  paper_stated:   "PAPER",
  brainstorm:     "BRAINSTORM",
  human_authored: "MANUAL",
};

const DIRECTION_TONE: Record<string, string> = {
  positive: "bg-ok/15 text-ok",
  negative: "bg-danger/15 text-danger",
  zero:     "bg-muted/15 text-muted",
};


type StatusFilter = "open" | "approved" | "reviewed" | "extracted" | "rejected" | "all";


export default function ForwardVectorsPage() {
  const router = useRouter();
  const { t } = useI18n();
  const [vectors, setVectors]   = useState<ForwardVector[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);
  const [familyFilter, setFamilyFilter] = useState<string>("");
  const [priorityFilter, setPriorityFilter] = useState<string>("");
  // P0-C — in-flight session-open spinner per hypothesis
  const [openingSession, setOpeningSession] = useState<string | null>(null);
  // Default = "open" — hides rejected, surfaces everything PM hasn't killed.
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("open");
  // P0-D — data-availability filter. Default = "all" so the user sees
  // everything; switch to "have"/"partial" to focus on actionable.
  const [dataFilter, setDataFilter] = useState<"all" | "have" | "partial" | "missing">("all");
  // In-flight review for optimistic UI
  const [pendingReview, setPendingReview] = useState<string | null>(null);
  // R2.9 — multi-select for compare. Keyed by source_hypothesis_id
  // (stable across re-renders + matches the PM-review schema).
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggleSelected = (hid: string) => {
    setSelected((cur) => {
      const next = new Set(cur);
      if (next.has(hid)) next.delete(hid);
      else next.add(hid);
      return next;
    });
  };
  const clearSelected = () => setSelected(new Set());

  // P0-C — "Open research session →" used to LIE: it just navigated to
  // /research/candidate with prefilled query params, never actually
  // opening a session. Fix: POST /api/sessions/open first, file the
  // intent with the real session_id, then navigate. Active session
  // shows up immediately on /dashboard.
  const openSessionForVector = async (v: ForwardVector) => {
    setOpeningSession(v.source_hypothesis_id);
    try {
      // 1. Open the typed session
      const briefTitle = `Test: ${v.claim.slice(0, 80)}`;
      const sesRes = await fetch(`${API_BASE}/api/sessions/open`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          session_type: "research_new",
          title:        briefTitle,
        }),
      });
      let sessionId: string | null = null;
      if (sesRes.ok) {
        const sessionRow = await sesRes.json();
        sessionId = sessionRow?.session_id ?? null;
      } else {
        // Don't block — the user can still test, just without the
        // session-tracking layer. Surface a subtle message.
        const txt = await sesRes.text();
        console.warn("[forward] /sessions/open failed:", txt);
      }

      // 2. File the research_test intent — payload now includes
      //    session_id so Claude can correlate. Bookkeeping; not awaited
      //    so navigation isn't blocked, but failures surface to console.
      fileIntent({
        kind:         "research_test",
        subject_type: "hypothesis",
        subject_id:   v.source_hypothesis_id,
        source_page:  "/research/forward",
        payload:      {
          paper_id:           v.source_paper_id,
          paper_title:        v.paper_title,
          mechanism_family:   v.mechanism_family,
          mechanism_subtype:  v.mechanism_subtype,
          priority:           v.priority,
          pm_status:          v.pm_status,
          claim:              v.claim,
          session_id:         sessionId,
        },
      })
        .then((r) => { if (!r.ok) console.warn("[forward] research_test intent refused:", r); })
        .catch((e) => console.warn("[forward] research_test intent errored:", e));

      // 3. Navigate to candidate pipeline with prefill
      const params = new URLSearchParams({
        from_hypothesis_id: v.source_hypothesis_id,
        proposal_name:      "test_" + v.mechanism_subtype.slice(0, 30),
        family:             v.mechanism_family,
      });
      if (sessionId) params.set("session_id", sessionId);
      router.push(`/research/candidate?${params.toString()}`);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setOpeningSession(null);
    }
  };

  const reload = () => {
    const params = new URLSearchParams();
    if (familyFilter)   params.set("mechanism_family", familyFilter);
    if (priorityFilter) params.set("priority", priorityFilter);
    if (statusFilter !== "all") params.set("pm_status", statusFilter);
    if (dataFilter !== "all")   params.set("data_status", dataFilter);
    params.set("top", "200");
    setLoading(true);
    fetch(`${API_BASE}/api/paper_chain/forward-vectors?${params}`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data: ForwardVector[]) => { setVectors(data); setError(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  useEffect(reload, [familyFilter, priorityFilter, statusFilter, dataFilter]);

  const setReview = async (hypId: string, status: PMStatus) => {
    setPendingReview(hypId);
    try {
      const res = await fetch(`${API_BASE}/api/paper_chain/forward-vectors/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_hypothesis_id: hypId, status }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Update locally to avoid full reload flicker
      setVectors((vs) => vs
        .map((v) => v.source_hypothesis_id === hypId
          ? { ...v, pm_status: status, pm_reviewed_ts: new Date().toISOString() }
          : v)
        // Filter out if the new state doesn't match current statusFilter
        .filter((v) => {
          if (statusFilter === "all")  return true;
          if (statusFilter === "open") return v.pm_status !== "rejected";
          return v.pm_status === statusFilter;
        })
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setPendingReview(null);
    }
  };

  const familyCounts = vectors.reduce<Record<string, number>>((acc, v) => {
    acc[v.mechanism_family] = (acc[v.mechanism_family] || 0) + 1;
    return acc;
  }, {});

  const statusCounts = vectors.reduce<Record<string, number>>((acc, v) => {
    acc[v.pm_status] = (acc[v.pm_status] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="p-6 space-y-4">
      <ModeHeader
        mode="research"
        title="Forward research vectors"
        subtitle="Paper-grounded untested hypotheses, ranked by priority. PM
                  approves before flowing into the candidate pipeline."
        right={<PaperChainSearch />}
      />

      <NextStepHint
        storageKey="hint:research_forward_v1"
        what={<>
          Each row is an <b>untested</b> hypothesis from a paper. Filter
          to <b>data=have</b> + <b>status=approved</b> to see what you
          can test today. Multi-select with the checkboxes to compare
          2-3 before approving.
        </>}
        next="Or go ingest a new paper"
        nextHref="/research/papers/new"
      />

      <Card>
        <div className="flex flex-wrap items-center gap-2 p-3 border-b border-line">
          <span className="text-xs uppercase text-muted">status:</span>
          {(["open", "approved", "reviewed", "extracted", "rejected", "all"] as StatusFilter[]).map((s) => (
            <button key={s}
                    className={`px-2 py-0.5 text-xs rounded border ${
                      s === statusFilter
                        ? "bg-accent/20 border-accent text-accent"
                        : "border-line text-muted hover:text-fg"
                    }`}
                    title={s === "open" ? "everything not rejected (default)" : `pm_status = ${s}`}
                    onClick={() => setStatusFilter(s)}>
              {s}
              {statusCounts[s] !== undefined && s !== "open" && s !== "all" && (
                <span className="ml-1 opacity-60 tnum">{statusCounts[s]}</span>
              )}
            </button>
          ))}
          <span className="text-xs uppercase text-muted ml-4">data:</span>
          {(["all", "have", "partial", "missing"] as const).map((d) => (
            <button key={d}
                    className={`px-2 py-0.5 text-xs rounded border ${
                      d === dataFilter
                        ? "bg-accent/20 border-accent text-accent"
                        : "border-line text-muted hover:text-fg"
                    }`}
                    title={
                      d === "have"    ? "required_data covered by data/cache+series" :
                      d === "partial" ? "some required_data covered, some not" :
                      d === "missing" ? "no required_data terms found in cache+series" :
                                        "all (default)"
                    }
                    onClick={() => setDataFilter(d)}>
              {d}
            </button>
          ))}

          <span className="text-xs uppercase text-muted ml-4">priority:</span>
          {(["", "high", "medium", "low"] as const).map((p) => (
            <button key={p || "all"}
                    className={`px-2 py-0.5 text-xs rounded border ${
                      p === priorityFilter
                        ? "bg-accent/20 border-accent text-accent"
                        : "border-line text-muted hover:text-fg"
                    }`}
                    onClick={() => setPriorityFilter(p)}>
              {p || "all"}
            </button>
          ))}
          <span className="text-xs uppercase text-muted ml-4">family:</span>
          <select className="bg-bg border border-line rounded text-xs px-2 py-0.5"
                  value={familyFilter}
                  onChange={(e) => setFamilyFilter(e.target.value)}>
            <option value="">all ({vectors.length})</option>
            {Object.entries(familyCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([fam, n]) => (
                <option key={fam} value={fam}>{fam} ({n})</option>
              ))}
          </select>
        </div>

        {loading && (
          <div className="p-6 text-sm text-muted">Loading forward vectors…</div>
        )}
        {error && (
          <div className="p-6 text-sm text-danger">Error: {error}</div>
        )}

        {!loading && !error && (
          <div className="divide-y divide-line">
            {vectors.length === 0 && (
              <div className="p-6 text-sm text-muted">No forward vectors match.</div>
            )}
            {vectors.map((v, i) => {
              const inFlight = pendingReview === v.source_hypothesis_id;
              const isApproved = v.pm_status === "approved";
              const isSel = selected.has(v.source_hypothesis_id);
              return (
              <div key={v.forward_vector_id}
                   className={`p-4 hover:bg-accent/5 ${isSel ? "bg-accent/[0.04]" : ""}`}>
                <div className="flex items-baseline gap-2 mb-2 flex-wrap">
                  <input type="checkbox"
                         checked={isSel}
                         onChange={() => toggleSelected(v.source_hypothesis_id)}
                         className="accent-accent cursor-pointer"
                         title="select for compare" />
                  <span className="text-xs text-muted w-8">#{i + 1}</span>
                  <Badge className={PRIORITY_TONE[v.priority]}>
                    {v.priority}
                  </Badge>
                  {/* Phase 2.1b track distinction badge — visible lane between
                      paper_stated (PAPER), brainstorm (BRAINSTORM, A's synthesis
                      after B + principal approval), and human_authored (MANUAL).
                      Read from priority_signals.track set by the dual-track
                      generator. */}
                  {typeof v.priority_signals?.track === "string" && (
                    <Badge className={TRACK_TONE[v.priority_signals.track] ||
                                      TRACK_TONE.human_authored}>
                      {TRACK_LABEL[v.priority_signals.track] ||
                       v.priority_signals.track}
                    </Badge>
                  )}
                  <Badge className={DIRECTION_TONE[v.predicted_direction]}>
                    {v.predicted_direction}
                  </Badge>
                  <Badge className={PM_STATUS_TONE[v.pm_status]}>
                    {v.pm_status}
                  </Badge>
                  <Tip
                    side="top"
                    content={
                      v.data_status === "have"
                        ? `Data covered — ${v.data_have.length}/${v.required_data.length} required terms match the local cache+series inventory.`
                      : v.data_status === "partial"
                        ? `Partial coverage. Have: ${v.data_have.length}; missing: ${v.data_missing.length}. Hover the row's "data:" line below for terms.`
                      : v.data_status === "missing"
                        ? "None of the required_data terms match the local inventory. You'd have to acquire data before testing."
                      : "required_data was empty; coverage unknown."
                    }>
                    <Badge className={DATA_STATUS_TONE[v.data_status]}>
                      data {v.data_status}
                    </Badge>
                  </Tip>
                  <span className="text-xs text-muted">
                    {v.mechanism_family} · {v.mechanism_subtype}
                  </span>

                  {/* Review actions */}
                  <div className="ml-auto flex items-center gap-1">
                    {v.pm_status !== "approved" && (
                      <button onClick={() => setReview(v.source_hypothesis_id, "approved")}
                              disabled={inFlight}
                              className="px-2 py-0.5 text-xs rounded border border-ok/40 bg-ok/10 text-ok hover:bg-ok/20 disabled:opacity-50"
                              title="Mark approved — ready to test">
                        ✓ approve
                      </button>
                    )}
                    {v.pm_status !== "reviewed" && v.pm_status !== "approved" && (
                      <button onClick={() => setReview(v.source_hypothesis_id, "reviewed")}
                              disabled={inFlight}
                              className="px-2 py-0.5 text-xs rounded border border-info/40 bg-info/10 text-info hover:bg-info/20 disabled:opacity-50"
                              title="Mark reviewed — seen but not yet decided">
                        ⊙ reviewed
                      </button>
                    )}
                    {v.pm_status !== "rejected" && (
                      <button onClick={() => setReview(v.source_hypothesis_id, "rejected")}
                              disabled={inFlight}
                              className="px-2 py-0.5 text-xs rounded border border-danger/40 text-danger/80 hover:bg-danger/10 disabled:opacity-50"
                              title="Reject — hide from default queue">
                        ✕
                      </button>
                    )}
                    {v.pm_status === "approved" || v.pm_status === "rejected" ? (
                      <button onClick={() => setReview(v.source_hypothesis_id, "extracted")}
                              disabled={inFlight}
                              className="px-1.5 py-0.5 text-xs text-muted/60 hover:text-fg"
                              title="Undo — back to extracted">
                        undo
                      </button>
                    ) : null}

                    <GraveyardCollisionChip hypothesisId={v.source_hypothesis_id} />
                    <a
                      href={`/research/hypothesis?id=${v.source_hypothesis_id}`}
                      className="ml-1 px-1.5 py-0.5 text-xs rounded border border-border/40 text-muted hover:text-foreground hover:border-accent/40"
                      title="Open this hypothesis drill-down (lineage + safety rails + verdicts spawned)">
                      drill
                    </a>
                    <button
                      onClick={() => openSessionForVector(v)}
                      disabled={openingSession === v.source_hypothesis_id}
                      className={`ml-1 px-2 py-0.5 text-xs rounded border disabled:opacity-50 ${
                        isApproved
                          ? "bg-accent text-background border-accent hover:bg-accent/90"
                          : "bg-accent/15 text-accent border-accent/40 hover:bg-accent/25"
                      }`}
                      title={isApproved
                        ? "PM-approved — opens REAL research_new session + files intent + jumps to candidate pipeline"
                        : "Opens REAL research_new session even though hypothesis isn't PM-approved yet"}>
                      {openingSession === v.source_hypothesis_id
                        ? "Opening…"
                        : "Open research session →"}
                    </button>
                  </div>
                </div>
                <div className="ml-11 space-y-1">
                  <p className="text-sm">{v.claim}</p>
                  <p className="text-xs text-muted">
                    <strong>magnitude:</strong> {v.predicted_magnitude}
                  </p>
                  {v.required_data?.length > 0 && (
                    <p className="text-xs text-muted">
                      <strong>data:</strong>{" "}
                      {v.required_data.slice(0, 3).join("; ")}
                      {v.required_data.length > 3 && "…"}
                    </p>
                  )}
                  <p className="text-xs text-muted">
                    <strong>source:</strong>{" "}
                    <Link href={`/research/papers/${v.source_paper_id}`}
                          className="underline hover:text-accent">
                      {v.paper_title}
                    </Link>
                    {" · "}
                    <code className="text-[10px]">
                      hyp:{v.source_hypothesis_id.slice(0, 8)}
                    </code>
                  </p>
                  {v.orthogonal_to_anchors?.length > 0 && (
                    <div className="text-xs text-muted pl-3 border-l-2 border-accent/30 mt-1.5">
                      <div className="text-[10px] uppercase tracking-wide text-accent/80 mb-0.5">
                        {t("forward.col.orthogonal")}
                      </div>
                      {v.orthogonal_to_anchors.slice(0, 3).map((o, i) => (
                        <div key={i} className="mb-0.5">
                          <Link
                            href={`/research/forward/anchors`}
                            className="font-mono text-[10px] text-accent hover:underline"
                          >
                            anchor:{o.anchor_paper_id?.slice(0, 8) ?? "?"}
                          </Link>
                          {" — "}
                          {o.why_orthogonal}
                        </div>
                      ))}
                      {v.orthogonal_to_anchors.length > 3 && (
                        <div className="text-[10px] text-muted/60">
                          +{v.orthogonal_to_anchors.length - 3} more
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
              );
            })}
          </div>
        )}
      </Card>

      {/* R2.9 + R2.12 — sticky compare bar using the shared Compare
          framework. Row definitions are forward-vector-specific;
          everything below is generic. */}
      {selected.size >= 1 && (
        <CompareBar<ForwardVector>
          items={vectors.filter((v) => selected.has(v.source_hypothesis_id))}
          getKey={(v) => v.source_hypothesis_id}
          onClear={clearSelected}
          onRemove={(v) => toggleSelected(v.source_hypothesis_id)}
          headerCell={(v) => (
            <div>
              <div className="flex items-center gap-1.5">
                <span className={`tnum text-[10px] px-1 rounded font-mono ${
                  v.priority === "high"   ? "bg-danger/15 text-danger" :
                  v.priority === "medium" ? "bg-warn/15 text-warn"   :
                                            "bg-muted/15 text-muted"
                }`}>
                  {v.priority}
                </span>
                <span className={`tnum text-[10px] px-1 rounded font-mono ${
                  v.pm_status === "approved" ? "bg-ok/15 text-ok" :
                  v.pm_status === "rejected" ? "bg-danger/10 text-danger/80" :
                                               "bg-muted/15 text-muted"
                }`}>
                  {v.pm_status}
                </span>
              </div>
              <code className="text-[9px] text-muted/60 mt-0.5 block">
                hyp:{v.source_hypothesis_id.slice(0, 8)}
              </code>
            </div>
          )}
          rows={[
            { label: "claim",         pick: (v) => v.claim },
            { label: "family",        pick: (v) => v.mechanism_family,  mono: true },
            { label: "subtype",       pick: (v) => v.mechanism_subtype, mono: true },
            { label: "direction",     pick: (v) => v.predicted_direction, mono: true },
            { label: "magnitude",     pick: (v) => v.predicted_magnitude },
            { label: "required data", pick: (v) => v.required_data.join(" · ") },
            { label: "paper",         pick: (v) => v.paper_title,
              href: (v) => `/research/papers/${v.source_paper_id}` },
          ]}
          actionCell={(v) => (
            <Link
              href={`/research/candidate?from_hypothesis_id=${v.source_hypothesis_id}` +
                    `&proposal_name=${encodeURIComponent("test_" + v.mechanism_subtype.slice(0, 30))}` +
                    `&family=${v.mechanism_family}`}
              className="inline-flex items-center gap-1 rounded bg-accent text-background hover:bg-accent/90 px-2 py-1 text-[10.5px] font-semibold">
              Open session →
            </Link>
          )} />
      )}
    </div>
  );
}
