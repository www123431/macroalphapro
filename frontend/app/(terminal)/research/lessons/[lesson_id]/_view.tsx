"use client";

// /research/lessons/[lesson_id] — single lesson detail with verbatim quotes
// + chain trace.

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { fileIntent } from "@/lib/intents";
import { Card, SectionTitle, Badge } from "@/components/ui";
import ChainTrace, { type ChainStage } from "@/components/ChainTrace";
import { PrintButton } from "@/components/PrintButton";


type VerbatimQuote = {
  chunk_id:       string;
  quote_text:     string;
  section_ref:    string;
  relevance_note: string;
};

type LessonDetail = {
  lesson_id:          string;
  candidate_name:     string;
  version:            number;
  verdict:            string;
  stat_evidence:      Record<string, unknown>;
  mechanism_family:   string;
  mechanism_subtype:  string;
  failure_modes:      string[];
  failure_evidence:   Record<string, string>;
  tested_hypothesis_ids: string[];
  verbatim_quotes:    VerbatimQuote[];
  grounding_method:   "paper_grounded" | "stat_only_grounded" | "pretrain_grounded";
  do_not_retry:       string[];
  summary:            string;
  created_ts:         string;
  tags:               string[];
};

const GROUNDING_TONE: Record<string, string> = {
  paper_grounded:     "bg-ok/15 text-ok border-ok/40",
  stat_only_grounded: "bg-warn/15 text-warn border-warn/40",
  pretrain_grounded:  "bg-muted/15 text-muted border-muted/40",
};


export default function LessonDetailPage(
  { params }: { params: Promise<{ lesson_id: string }> }
) {
  const { lesson_id } = use(params);
  const [lesson, setLesson] = useState<LessonDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/paper_chain/lessons/${lesson_id}`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((L: LessonDetail) => { setLesson(L); setError(null); })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [lesson_id]);

  if (loading) return <div className="p-6 text-sm text-muted">Loading…</div>;
  if (error)   return <div className="p-6 text-sm text-danger">Error: {error}</div>;
  if (!lesson) return <div className="p-6 text-sm text-muted">Not found.</div>;

  // Build ChainTrace stages (PAPER source not known from lesson alone — use
  // the first hypothesis_id's source paper via a follow-up state; for now,
  // show the chain only when we have paper_grounded data).
  const chainStages: ChainStage[] =
    lesson.grounding_method === "paper_grounded" && lesson.tested_hypothesis_ids.length > 0
      ? [
          {
            label:    "PAPER",
            kind:     "paper",
            title:    "Source paper",
            subtitle: `cited via ${lesson.verbatim_quotes.length} verbatim quote${
              lesson.verbatim_quotes.length > 1 ? "s" : ""
            }`,
            status:   "done",
          },
          {
            label:    "HYPOTHESIS",
            kind:     "hypothesis",
            title:    `${lesson.tested_hypothesis_ids.length} tested`,
            subtitle: lesson.tested_hypothesis_ids
              .map((h) => h.slice(0, 8))
              .join(", "),
            status:   "done",
          },
          {
            label:    "TEST",
            kind:     "test",
            title:    lesson.candidate_name,
            subtitle: lesson.mechanism_family,
            status:   "done",
          },
          {
            label:    "VERDICT",
            kind:     "verdict",
            title:    lesson.verdict,
            subtitle: `failure: ${lesson.failure_modes.join(", ")}`,
            status:   "done",
          },
        ]
      : [];

  return (
    <div className="p-6 space-y-4">
      <div className="space-y-1">
        <div className="flex items-baseline gap-3">
          <SectionTitle>{lesson.candidate_name}</SectionTitle>
          <Badge className={
            lesson.verdict === "RED" ? "bg-danger/15 text-danger"
                                     : "bg-warn/15 text-warn"
          }>{lesson.verdict}</Badge>
          <Badge className={`border ${GROUNDING_TONE[lesson.grounding_method]}`}>
            {lesson.grounding_method}
          </Badge>
          <span className="text-xs text-muted ml-auto">v{lesson.version}</span>
          <PrintButton title="Export this lesson as PDF (browser print dialog)"
                       label="Export PDF" />
        </div>
        <p className="text-xs text-muted">
          {lesson.mechanism_family} · {lesson.mechanism_subtype}
          {" · "}created {lesson.created_ts}
        </p>
      </div>

      {/* R2.10 — close the loop. Reading a RED Lesson without a "now
          what?" CTA is the dead-end the audit flagged. Adjacent CTAs
          push the user back into the workflow informed by this verdict. */}
      <AdjacentActions
        family={lesson.mechanism_family}
        subtype={lesson.mechanism_subtype}
        verdict={lesson.verdict}
        candidateName={lesson.candidate_name} />

      {chainStages.length > 0 && (
        <Card className="border-accent/30">
          <div className="p-3">
            <div className="text-xs uppercase text-muted mb-1">Chain trace</div>
            <ChainTrace stages={chainStages} />
          </div>
        </Card>
      )}

      <Card>
        <div className="p-4">
          <h3 className="text-sm font-semibold mb-1">Summary</h3>
          <p className="text-sm">{lesson.summary}</p>
        </div>
      </Card>

      {lesson.grounding_method === "paper_grounded" &&
        lesson.verbatim_quotes.length > 0 && (
        <Card className="border-ok/30">
          <div className="p-4">
            <h3 className="text-sm font-semibold mb-2">
              Verbatim paper quotes ({lesson.verbatim_quotes.length})
            </h3>
            <div className="space-y-3">
              {lesson.verbatim_quotes.map((q, i) => (
                <div key={i} className="border-l-2 border-ok/40 pl-3">
                  <p className="text-xs text-muted mb-0.5">
                    <code>{q.chunk_id}</code>
                    {q.section_ref && <> · {q.section_ref}</>}
                  </p>
                  <p className="text-sm italic">"{q.quote_text}"</p>
                  {q.relevance_note && (
                    <p className="text-xs text-muted mt-0.5">
                      → {q.relevance_note}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {lesson.tested_hypothesis_ids.length > 0 && (
        <Card>
          <div className="p-4">
            <h3 className="text-sm font-semibold mb-2">Hypotheses tested</h3>
            <ul className="text-xs space-y-1">
              {lesson.tested_hypothesis_ids.map((hid) => (
                <li key={hid}><code>{hid}</code></li>
              ))}
            </ul>
          </div>
        </Card>
      )}

      <div className="grid grid-cols-2 gap-3">
        <Card><div className="p-4">
          <h3 className="text-sm font-semibold mb-2">Failure modes</h3>
          <div className="space-y-2">
            {lesson.failure_modes.map((fm) => (
              <div key={fm}>
                <Badge className="text-[10px] bg-danger/15 text-danger">{fm}</Badge>
                {lesson.failure_evidence[fm] && (
                  <p className="text-xs text-muted mt-1 ml-1">
                    {lesson.failure_evidence[fm]}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div></Card>

        <Card><div className="p-4">
          <h3 className="text-sm font-semibold mb-2">Stat evidence</h3>
          <table className="text-xs">
            <tbody>
              {Object.entries(lesson.stat_evidence).map(([k, v]) => (
                <tr key={k}>
                  <td className="pr-3 text-muted">{k}:</td>
                  <td className="font-mono">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div></Card>
      </div>

      {lesson.do_not_retry.length > 0 && (
        <Card className="border-danger/30">
          <div className="p-4">
            <h3 className="text-sm font-semibold mb-2 text-danger">
              Do not retry
            </h3>
            <ul className="text-sm space-y-1">
              {lesson.do_not_retry.map((s, i) => <li key={i}>· {s}</li>)}
            </ul>
          </div>
        </Card>
      )}

      {lesson.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {lesson.tags.map((t) => (
            <Badge key={t} className="text-[10px] bg-muted/10 text-muted">
              {t}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}


// ── Adjacent actions (R2.10 — closes the loop) ────────────────────


function AdjacentActions({
  family, subtype, verdict, candidateName,
}: {
  family:        string;
  subtype?:      string;     // P1-B 2026-06-04 — use subtype for finer adjacency
  verdict:       string;
  candidateName: string;
}) {
  const [counts, setCounts] = useState<{
    untested:        number;
    tested:          number;
    untested_subtype: number;
    tested_subtype:   number;
  }>({ untested: 0, tested: 0, untested_subtype: 0, tested_subtype: 0 });

  useEffect(() => {
    if (!family) return;
    // P1-B: family-level is too coarse — RED on `news_attention_shock`
    // (family=OTHER) returns 700+ untested. Pull family-level counts AND
    // client-side filter by subtype for the precise sibling count.
    fetch(`${API_BASE}/api/paper_chain/forward-vectors?mechanism_family=${encodeURIComponent(family)}&pm_status=open&top=500`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((arr: any[]) => {
        const subt = (subtype || "").toLowerCase();
        const same = arr.filter((v) =>
          (v.mechanism_subtype || "").toLowerCase() === subt);
        setCounts((c) => ({
          ...c,
          untested:         arr.length,
          untested_subtype: subt ? same.length : 0,
        }));
      })
      .catch(() => {});
    fetch(`${API_BASE}/api/paper_chain/lessons?mechanism_family=${encodeURIComponent(family)}&include_legacy=true&limit=500`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((arr: any[]) => {
        const subt = (subtype || "").toLowerCase();
        const same = arr.filter((L) =>
          (L.mechanism_subtype || "").toLowerCase() === subt);
        setCounts((c) => ({
          ...c,
          tested:         arr.length,
          tested_subtype: subt ? same.length : 0,
        }));
      })
      .catch(() => {});
  }, [family, subtype]);

  return (
    <Card className="border-accent/30 bg-accent/[0.03]">
      <div className="p-3 flex flex-wrap items-center gap-2">
        <span className="text-[11px] text-muted">
          Now what? Adjacent moves
          {subtype ? <> for subtype <code className="text-foreground">{subtype}</code> (family <code className="text-muted">{family}</code>):</>
                   : <> for family <code className="text-foreground">{family}</code>:</>}
        </span>

        {/* Prefer subtype-level when we have a non-empty subtype */}
        {subtype && counts.untested_subtype > 0 && (
          <Link
            href={`/research/forward?mechanism_family=${encodeURIComponent(family)}&pm_status=open`}
            className="inline-flex items-center gap-1 rounded bg-accent text-background hover:bg-accent/90 px-2.5 py-1 text-[11px] font-semibold"
            title={`Untested hypotheses with subtype = ${subtype}`}>
            Find untested · same subtype
            <span className="text-[10px] opacity-80 tnum">({counts.untested_subtype})</span>
          </Link>
        )}

        <Link
          href={`/research/forward?mechanism_family=${encodeURIComponent(family)}&pm_status=open`}
          className={`inline-flex items-center gap-1 rounded ${
            subtype && counts.untested_subtype > 0
              ? "border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20"
              : "bg-accent text-background hover:bg-accent/90 font-semibold"
          } px-2.5 py-1 text-[11px]`}>
          Find untested · family
          <span className="text-[10px] opacity-80 tnum">({counts.untested})</span>
        </Link>

        <Link
          href={`/research/lessons?mechanism_family=${encodeURIComponent(family)}&include_legacy=true`}
          className="inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 px-2.5 py-1 text-[11px]">
          All verdicts on family
          <span className="text-[10px] opacity-80 tnum">({counts.tested})</span>
        </Link>

        {verdict === "RED" && (
          <Link
            href={`/research/sessions?type=audit&prefill_subject=${encodeURIComponent(candidateName)}`}
            onClick={() => {
              // Bookkeeping intent fires as the user navigates; surface
              // failures to console so they aren't silently swallowed.
              fileIntent({
                kind:         "audit_subject",
                subject_type: "lesson",
                subject_id:   candidateName,
                source_page:  "/research/lessons/[lesson_id]",
                payload:      { family, verdict, from: "lesson_detail" },
              })
                .then((r) => { if (!r.ok) console.warn("[lesson_detail] audit_subject intent refused:", r); })
                .catch((e) => console.warn("[lesson_detail] audit_subject intent errored:", e));
            }}
            className="inline-flex items-center gap-1 rounded border border-warn/40 bg-warn/10 text-warn hover:bg-warn/20 px-2.5 py-1 text-[11px]"
            title="Open a typed audit session — files audit_subject intent for Claude">
            Open audit session
          </Link>
        )}
      </div>
    </Card>
  );
}
