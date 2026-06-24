"use client";

// /research/hypothesis?id=<hyp_id> — first-class HYPOTHESIS object view.
//
// Phase 6 of UI restructure (2026-06-14). Pairs with /research/family
// to give research a real object-centric drill layer. Reached from:
//   - /research/forward row click (forward queue)
//   - /research/family page autopsies + pending tables
//   - /research/verdict drill (verdict knows hyp_id)
//   - /approvals row hyp_id citation
//
// One page shows the full lineage: source paper → claim + verbatim
// quote → B's verdict (APPROVE/AMENDMENT/REJECT) → human resolution →
// factor verdicts spawned → safety-rail context (rigor / audit / belief).

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { Quote, Activity, GitBranch, BookOpen } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { useGraveyardCollisions } from "@/lib/queries";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { Breadcrumb } from "@/components/Breadcrumb";
import { SafetyRailsBanner } from "@/components/SafetyRailsBanner";
import { PreMortemSection } from "@/components/PreMortemSection";
import { ReplicationCheckSection } from "@/components/ReplicationCheckSection";
import { Skull, AlertTriangle } from "lucide-react";


type HypothesisDetail = {
  hypothesis_id:   string;
  hypothesis:      any;
  source_paper:    any;
  resolution: {
    decision: string; rationale: string;
    resolved_ts: string; resolved_by: string;
  } | null;
  b_verdict?: {
    verdict_type: string; confidence: number;
    one_line_summary: string; reasoning: string;
  };
  verdicts: Array<{
    event_id: string; ts: string; verdict: string;
    subject_id: string; summary: string; family: string | null;
  }>;
  safety_rails: any;
};

const VERDICT_TONE: Record<string, string> = {
  GREEN:    "bg-ok/15 text-ok",
  MARGINAL: "bg-warn/15 text-warn",
  RED:      "bg-danger/15 text-danger",
};

export default function HypothesisPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <HypInner />
    </Suspense>
  );
}

function HypInner() {
  const sp = useSearchParams();
  const hypId = sp.get("id") || "";
  const [data, setData] = useState<HypothesisDetail | null>(null);
  const [err, setErr]   = useState<string | null>(null);

  useEffect(() => {
    if (!hypId) return;
    setErr(null);
    fetch(`${API_BASE}/api/research/hypothesis/${encodeURIComponent(hypId)}`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : r.json().then((j) => Promise.reject(j?.detail || `HTTP ${r.status}`)))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, [hypId]);

  if (!hypId) {
    return <div className="p-6 text-sm text-muted">No hypothesis id specified. Try ?id=&lt;uuid&gt;</div>;
  }

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <Breadcrumb crumbs={[
          { label: "Research", href: "/research" },
          { label: "Hypothesis", mono: false },
          { label: hypId.slice(0, 8) + "…", mono: true },
        ]} />
      </motion.div>

      {err && <Card className="border-danger/30 bg-danger/5 text-sm text-danger">{err}</Card>}
      {!data && !err && <Skeleton className="h-40 w-full" />}

      {data && data.hypothesis && (
        <>
          {/* Header */}
          <motion.div variants={fadeUp}>
            <Card>
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0">
                  <div className="text-[10px] uppercase tracking-wider text-muted">
                    Hypothesis · {data.hypothesis.mechanism_family || "—"}
                    {data.hypothesis.mechanism_subtype && (
                      <span className="text-muted/60"> · {data.hypothesis.mechanism_subtype}</span>
                    )}
                  </div>
                  <h1 className="text-base font-mono mt-0.5 break-all">
                    {data.hypothesis_id}
                  </h1>
                  {data.hypothesis.claim && (
                    <p className="text-[13px] text-foreground/90 mt-2 leading-relaxed">
                      {typeof data.hypothesis.claim === "string"
                        ? data.hypothesis.claim
                        : (data.hypothesis.claim.one_line || JSON.stringify(data.hypothesis.claim).slice(0, 300))}
                    </p>
                  )}
                </div>
                <div className="text-right shrink-0">
                  {data.hypothesis.mechanism_family && (
                    <Link href={`/research/family?id=${data.hypothesis.mechanism_family}`}
                      className="text-[11px] text-accent hover:underline inline-flex items-center gap-1">
                      family →
                    </Link>
                  )}
                  {data.hypothesis.source_paper_id && (
                    <div className="text-[10px] text-muted mt-1">
                      <Link href={`/research/papers/${data.hypothesis.source_paper_id}`}
                        className="inline-flex items-center gap-1 hover:text-accent">
                        <BookOpen className="h-3 w-3" /> source paper
                      </Link>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Verbatim quotes */}
          {Array.isArray(data.hypothesis.verbatim_quotes) && data.hypothesis.verbatim_quotes.length > 0 && (
            <motion.div variants={fadeUp}>
              <SectionTitle>
                <span className="inline-flex items-center gap-1.5">
                  <Quote className="h-3.5 w-3.5" />
                  Verbatim quotes ({data.hypothesis.verbatim_quotes.length})
                </span>
              </SectionTitle>
              <Card className="space-y-2">
                {data.hypothesis.verbatim_quotes.slice(0, 4).map((q: any, i: number) => (
                  <blockquote key={i} className="text-[12px] text-muted/90 border-l-2 border-accent/40 pl-3 leading-relaxed">
                    {typeof q === "string" ? q : (q.text || JSON.stringify(q))}
                    {q.chunk_id && (
                      <div className="mt-1 text-[10px] font-mono text-muted/50">
                        chunk_id={q.chunk_id}
                      </div>
                    )}
                  </blockquote>
                ))}
              </Card>
            </motion.div>
          )}

          {/* B verdict */}
          {data.b_verdict && (
            <motion.div variants={fadeUp}>
              <SectionTitle>Strengthener (B) verdict</SectionTitle>
              <Card className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge tone={
                    data.b_verdict.verdict_type === "APPROVE_FOR_PIPELINE" ? "bg-accent/15 text-accent" :
                    data.b_verdict.verdict_type === "DOCTRINE_AMENDMENT_NEEDED" ? "bg-warn/15 text-warn" :
                                                                                   "bg-muted/15 text-muted"
                  }>
                    {data.b_verdict.verdict_type}
                  </Badge>
                  <span className="text-[11px] text-muted tnum">
                    confidence {(data.b_verdict.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                {data.b_verdict.one_line_summary && (
                  <p className="text-[12.5px]">{data.b_verdict.one_line_summary}</p>
                )}
                {data.b_verdict.reasoning && (
                  <p className="text-[11px] text-muted leading-relaxed">{data.b_verdict.reasoning}</p>
                )}
              </Card>
            </motion.div>
          )}

          {/* Resolution */}
          {data.resolution && (
            <motion.div variants={fadeUp}>
              <SectionTitle>Resolution</SectionTitle>
              <Card>
                <div className="flex items-center gap-2 text-[12px]">
                  <Badge tone={
                    data.resolution.decision === "approved" ? "bg-ok/15 text-ok" :
                    data.resolution.decision === "rejected" ? "bg-danger/15 text-danger" :
                                                                "bg-muted/15 text-muted"
                  }>
                    {data.resolution.decision.toUpperCase()}
                  </Badge>
                  <span className="text-muted/70">{data.resolution.resolved_by}</span>
                  <span className="font-mono text-muted/60">
                    {data.resolution.resolved_ts?.slice(0, 16).replace("T", " ")}
                  </span>
                </div>
                {data.resolution.rationale && (
                  <p className="mt-2 text-[11px] text-muted">{data.resolution.rationale}</p>
                )}
              </Card>
            </motion.div>
          )}

          {/* Graveyard collision — Phase 8 reactive subscriber.
              Flags if this hyp looks like one we've already RED'd. */}
          <GraveyardCollisionsSection hypId={hypId} />

          {/* α Pre-Mortem Generator (2026-06-14). Skeptic persona
              enumerates concrete failure modes BEFORE strict-gate
              dispatch. Stigler 1973 adversarial review pattern. */}
          <motion.div variants={fadeUp}>
            <PreMortemSection hypothesisId={hypId} />
          </motion.div>

          {/* γ Replication Checker (2026-06-14). Lit-aware specialist
              against Hou-Xue-Zhang 2020 / McLean-Pontiff 2016 catalogs.
              Complements α (general adversarial). */}
          <motion.div variants={fadeUp}>
            <ReplicationCheckSection hypothesisId={hypId} />
          </motion.div>

          {/* Safety rails (full mode) */}
          <motion.div variants={fadeUp}>
            <SectionTitle>Backend safety rails</SectionTitle>
            <SafetyRailsBanner hypothesisId={hypId} compact={false} />
          </motion.div>

          {/* Verdicts spawned */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                Factor verdicts spawned ({data.verdicts.length})
              </span>
            </SectionTitle>
            <Card className="p-0 overflow-hidden">
              {data.verdicts.length === 0 ? (
                <p className="p-3 text-sm text-muted">No factor verdicts yet for this hypothesis.</p>
              ) : (
                <ul className="divide-y divide-border/30">
                  {data.verdicts.map((v) => (
                    <li key={v.event_id}>
                      <Link href={`/research/verdict?event_id=${v.event_id}`}
                        className="flex items-center gap-3 px-3 py-2 hover:bg-panel2/40 transition-colors">
                        <Badge tone={VERDICT_TONE[v.verdict] || "bg-muted/15 text-muted"}>
                          {v.verdict}
                        </Badge>
                        <span className="font-mono text-[11px] text-muted/70">
                          {v.ts.slice(0, 10)}
                        </span>
                        <span className="flex-1 truncate text-[12px]">
                          {v.summary || v.subject_id}
                        </span>
                        {v.family && (
                          <Link href={`/research/family?id=${v.family}`}
                            className="text-[10px] font-mono text-muted/60 hover:text-accent"
                            onClick={(e) => e.stopPropagation()}>
                            {v.family}
                          </Link>
                        )}
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </motion.div>
        </>
      )}
    </motion.div>
  );
}


function GraveyardCollisionsSection({ hypId }: { hypId: string }) {
  const { data } = useGraveyardCollisions(hypId);
  if (!data || data.top_collisions.length === 0) return null;
  const worst = Math.max(...data.top_collisions.map((c) => c.score));
  const tone =
    worst >= 0.7 ? "alert" :
    worst >= 0.4 ? "warn"  :
                    "muted" as const;
  const toneCls =
    tone === "alert" ? "border-alert/40 bg-alert/[0.04]" :
    tone === "warn"  ? "border-warn/40 bg-warn/[0.03]" :
                       "border-border/40 bg-panel2/30";
  const iconCls =
    tone === "alert" ? "text-alert" :
    tone === "warn"  ? "text-warn"  :
                       "text-muted";
  return (
    <motion.div variants={fadeUp}>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <Skull className="h-3.5 w-3.5" strokeWidth={1.75} />
          Graveyard collisions ({data.n_total_red})
        </span>
      </SectionTitle>
      <div className={cn("rounded-md border px-3 py-2 space-y-1.5", toneCls)}>
        <div className={cn("flex items-center gap-2 text-[11px]", iconCls)}>
          <AlertTriangle className="h-3.5 w-3.5" strokeWidth={2.2} />
          <span className="font-semibold">
            {data.n_total_red} RED outcome{data.n_total_red === 1 ? "" : "s"} found similar to this hypothesis
          </span>
          <span className="text-[10px] text-muted/70 ml-auto italic">
            top score {worst.toFixed(2)} · {data.score_doctrine}
          </span>
        </div>
        <div className="space-y-1">
          {data.top_collisions.map((c, i) => (
            <div key={i} className="flex items-center gap-2 text-[10.5px] pl-5">
              <span className={cn(
                "font-mono tabular-nums w-12",
                c.score >= 0.7 ? "text-alert" : c.score >= 0.4 ? "text-warn" : "text-muted",
              )}>{c.score.toFixed(2)}</span>
              <span className="font-mono text-muted/80">
                {c.family || "—"}
              </span>
              <span className="text-muted/60 tnum text-[9px]">
                fam={c.family_match ? "✓" : "·"} jac={c.jaccard.toFixed(2)}
              </span>
              <span className="flex-1 truncate text-muted/80">
                {c.claim_excerpt || "(no claim text)"}
              </span>
              {c.hypothesis_id && (
                <Link href={`/research/hypothesis?id=${c.hypothesis_id}`}
                  className="text-accent/70 hover:text-accent text-[9px]">view</Link>
              )}
              {c.event_id && !c.hypothesis_id && (
                <Link href={`/research/verdict?event_id=${c.event_id}`}
                  className="text-accent/70 hover:text-accent text-[9px]">verdict</Link>
              )}
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
