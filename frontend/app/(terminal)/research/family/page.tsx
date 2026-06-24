"use client";

// /research/family?id=<family_id> — first-class FAMILY object view.
//
// Phase 6 of UI restructure (2026-06-14): adds object-centric pages
// (family / hypothesis) so research has a real "single thing deep
// drill" surface in addition to the list views (/papers, /forward,
// /lessons). One family-level page surfaces the belief direction +
// every verdict / autopsy / pending hyp tagged with this family —
// the unit of work for "do I keep mining VRP variants or pivot off it".
//
// Family identifier = autopsy strategy_family (VRP / EVENT_DRIFT /
// CARRY_FX / etc). See [[feedback-strategy-family-vs-claim-family-
// 2026-06-12]] — this is the SPEC-derived label, the same denominator
// the Bailey-LdP n_trials counter uses.

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import { Brain, Activity, Skull, FileText, ChevronRight } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { Breadcrumb } from "@/components/Breadcrumb";


type FamilyRow = {
  family:                string;
  belief: {
    family: string; n_obs: number; n_green: number; n_marginal: number;
    n_red: number; direction_hint: string;
  } | null;
  recent_verdicts: Array<{
    event_id: string; ts: string; verdict: string;
    subject_id: string; summary: string;
    hypothesis_id: string | null;
  }>;
  autopsies: Array<{
    autopsy_id: string; ts: string; hypothesis_id: string;
    actual_verdict: string; surprise_direction: string | null;
    surprise_magnitude: number | null; brier_component: number | null;
  }>;
  pending_hypotheses: Array<{
    hypothesis_id: string; source_paper_id: string;
    mechanism_family: string; review_state: string;
    claim_one_line: string; created_ts: string;
  }>;
  n_verdicts_total: number;
  n_pending_total:  number;
  bailey_ldp_n_trials: {
    n_trials: number;
    library_entries: number;
    exploration_buffer: number;
    doctrine_ref: string;
  } | null;
};

const VERDICT_TONE: Record<string, string> = {
  GREEN:    "bg-ok/15 text-ok",
  MARGINAL: "bg-warn/15 text-warn",
  RED:      "bg-danger/15 text-danger",
};

function hintTone(hint: string): string {
  const f = hint.split(" ")[0];
  if (f === "EXPLORE")       return "bg-ok/15 text-ok";
  if (f === "AVOID")         return "bg-alert/15 text-alert";
  if (f === "MARGINAL-ONLY") return "bg-warn/15 text-warn";
  if (f === "MIXED")         return "bg-info/15 text-info";
  return "bg-muted/15 text-muted";
}

export default function FamilyPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <FamilyInner />
    </Suspense>
  );
}

function FamilyInner() {
  const sp = useSearchParams();
  const familyId = sp.get("id") || "";
  const [data, setData] = useState<FamilyRow | null>(null);
  const [err, setErr]   = useState<string | null>(null);

  useEffect(() => {
    if (!familyId) return;
    setErr(null);
    fetch(`${API_BASE}/api/research/family/${encodeURIComponent(familyId)}`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, [familyId]);

  if (!familyId) {
    return <div className="p-6 text-sm text-muted">No family id specified. Try ?id=VRP</div>;
  }

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <Breadcrumb crumbs={[
          { label: "Research", href: "/research" },
          { label: "Family", mono: false },
          { label: familyId, mono: true },
        ]} />
      </motion.div>

      {err && <Card className="border-danger/30 bg-danger/5 text-sm text-danger">{err}</Card>}
      {!data && !err && <Skeleton className="h-40 w-full" />}

      {data && (
        <>
          {/* Header — name + belief direction */}
          <motion.div variants={fadeUp}>
            <Card>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted">Family</div>
                  <h1 className="text-2xl font-semibold tracking-tight font-mono mt-0.5">
                    {data.family}
                  </h1>
                  {data.belief && (
                    <div className="mt-2 inline-flex items-center gap-2">
                      <Brain className="h-3.5 w-3.5 text-muted" />
                      <Badge tone={hintTone(data.belief.direction_hint)}>
                        {data.belief.direction_hint}
                      </Badge>
                    </div>
                  )}
                </div>
                {data.belief && (
                  <div className="text-right">
                    <div className="text-[10px] uppercase tracking-wider text-muted">distribution</div>
                    <div className="tnum text-lg font-semibold mt-1">
                      <span className="text-ok">{data.belief.n_green}G</span> ·{" "}
                      <span className="text-warn">{data.belief.n_marginal}M</span> ·{" "}
                      <span className="text-alert">{data.belief.n_red}R</span>
                    </div>
                    <div className="text-[10px] text-muted mt-0.5">n = {data.belief.n_obs}</div>
                  </div>
                )}
              </div>

              {/* Bailey-LdP n_trials counter — Phase 7 (2026-06-14).
                  Each new trial in this family pushes the DSR threshold
                  up; surfacing this lets the principal see the multi-
                  testing cost before approving yet another variant. */}
              {data.bailey_ldp_n_trials && (
                <div className="mt-3 pt-3 border-t border-border/30 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px]">
                  <span className="uppercase tracking-wider text-muted/70 text-[9px]">
                    Bailey-LdP DSR denominator
                  </span>
                  <span className="font-mono">
                    <span className="text-foreground/90 font-semibold">
                      N = {data.bailey_ldp_n_trials.n_trials}
                    </span>
                    <span className="text-muted/60"> = {data.bailey_ldp_n_trials.library_entries} library + {data.bailey_ldp_n_trials.exploration_buffer} exploration buffer</span>
                  </span>
                  <span className="text-[10px] text-muted/60 italic">
                    {data.bailey_ldp_n_trials.doctrine_ref}
                  </span>
                </div>
              )}
            </Card>
          </motion.div>

          {/* Recent verdicts */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" />
                Verdicts ({data.n_verdicts_total})
              </span>
            </SectionTitle>
            <Card className="p-0 overflow-hidden">
              {data.recent_verdicts.length === 0 ? (
                <p className="p-3 text-sm text-muted">No verdicts yet for this family.</p>
              ) : (
                <ul className="divide-y divide-border/30">
                  {data.recent_verdicts.map((v) => (
                    <li key={v.event_id}>
                      <Link href={`/research/verdict?event_id=${v.event_id}`}
                        className="flex items-center gap-3 px-3 py-2 hover:bg-panel2/40 transition-colors">
                        <Badge tone={VERDICT_TONE[v.verdict] || "bg-muted/15 text-muted"}>
                          {v.verdict}
                        </Badge>
                        <span className="font-mono text-[11px] text-muted/70">
                          {v.ts.slice(0, 10)}
                        </span>
                        <span className="flex-1 truncate text-[12px] text-foreground/90">
                          {v.summary || v.subject_id}
                        </span>
                        {v.hypothesis_id && (
                          <span className="text-[10px] font-mono text-muted/60">
                            hyp {v.hypothesis_id.slice(0, 8)}
                          </span>
                        )}
                        <ChevronRight className="h-3 w-3 text-muted/60" />
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </motion.div>

          {/* Autopsies */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <Skull className="h-3.5 w-3.5" />
                Autopsies ({data.autopsies.length})
              </span>
            </SectionTitle>
            <Card className="p-0 overflow-hidden">
              {data.autopsies.length === 0 ? (
                <p className="p-3 text-sm text-muted">No autopsies recorded.</p>
              ) : (
                <table className="min-w-full text-[11px]">
                  <thead className="text-left text-[9px] uppercase tracking-wider text-muted/80 border-b border-border/30">
                    <tr>
                      <th className="px-3 py-1.5">date</th>
                      <th className="px-3 py-1.5">hypothesis</th>
                      <th className="px-3 py-1.5">verdict</th>
                      <th className="px-3 py-1.5">surprise</th>
                      <th className="px-3 py-1.5 text-right">brier</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/20">
                    {data.autopsies.map((a) => (
                      <tr key={a.autopsy_id} className="hover:bg-panel2/40">
                        <td className="px-3 py-1.5 font-mono text-muted/70">{a.ts.slice(0, 10)}</td>
                        <td className="px-3 py-1.5">
                          <Link href={`/research/hypothesis?id=${a.hypothesis_id}`}
                            className="font-mono text-accent hover:underline">
                            {a.hypothesis_id.slice(0, 8)}
                          </Link>
                        </td>
                        <td className="px-3 py-1.5">
                          <Badge tone={VERDICT_TONE[a.actual_verdict] || "bg-muted/15 text-muted"}>
                            {a.actual_verdict}
                          </Badge>
                        </td>
                        <td className="px-3 py-1.5 text-muted/80">
                          {a.surprise_direction || "—"}
                          {a.surprise_magnitude != null && (
                            <span className="ml-1 tnum text-muted/60">
                              ({a.surprise_magnitude.toFixed(2)})
                            </span>
                          )}
                        </td>
                        <td className="px-3 py-1.5 text-right tnum text-muted/80">
                          {a.brier_component != null ? a.brier_component.toFixed(3) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Card>
          </motion.div>

          {/* Pending hypotheses */}
          {data.n_pending_total > 0 && (
            <motion.div variants={fadeUp}>
              <SectionTitle>
                <span className="inline-flex items-center gap-1.5">
                  <FileText className="h-3.5 w-3.5" />
                  Pending hypotheses ({data.n_pending_total})
                </span>
              </SectionTitle>
              <Card className="p-0 overflow-hidden">
                <ul className="divide-y divide-border/30">
                  {data.pending_hypotheses.map((h) => (
                    <li key={h.hypothesis_id}>
                      <Link href={`/research/hypothesis?id=${h.hypothesis_id}`}
                        className="block px-3 py-2 hover:bg-panel2/40 transition-colors space-y-1">
                        <div className="flex items-center gap-2 text-[10px] text-muted/70">
                          <span className="font-mono">{h.hypothesis_id.slice(0, 8)}</span>
                          <span>· {h.mechanism_family}</span>
                          <span>· {h.review_state || "pending"}</span>
                        </div>
                        <p className="text-[12px] text-foreground/90 leading-snug">
                          {h.claim_one_line || "(no one-line summary)"}
                        </p>
                      </Link>
                    </li>
                  ))}
                </ul>
              </Card>
            </motion.div>
          )}
        </>
      )}
    </motion.div>
  );
}
