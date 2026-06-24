"use client";

// SafetyRailsBanner — compact inline strip that surfaces backend gate
// state (Phase 1.2 external audit / Phase 4.1 post-GREEN rigor / Phase B
// belief layer) on /approvals decision rows. Added 2026-06-14 because
// before this, the principal had to navigate to /research/lessons to
// see whether the LLM-judge audit flagged the verdict as critical, OR
// open the rigor jsonl file by hand to see whether OOS / spanning /
// borrow-cost killed it. "Invisible safety rails = no safety rails"
// — surface them at the decision point.
//
// Compact mode renders one tight row; clicking expands details. Use the
// compact form on approval row cards; use the full form on the verdict
// detail page (/research/verdict).

import { Microscope, ShieldCheck, Brain, AlertTriangle, ChevronDown } from "lucide-react";
import { useState } from "react";
import Link from "next/link";
import { useSafetyRailsForHypothesis } from "@/lib/queries";
import { cn } from "@/components/ui";

const SEV_TONE: Record<string, string> = {
  critical: "text-alert",
  concern:  "text-warn",
  ok:       "text-ok",
  skipped:  "text-muted/60",
};

const HINT_TONE = (hint: string) => {
  const first = hint.split(" ")[0];
  if (first === "EXPLORE")       return "text-ok";
  if (first === "AVOID")         return "text-alert";
  if (first === "MARGINAL-ONLY") return "text-warn";
  if (first === "MIXED")         return "text-info";
  if (first === "WEAK")          return "text-warn";
  return "text-muted/70";
};

const PILL = "rounded px-1.5 py-0.5 text-[10px] font-medium tabular-nums";

export function SafetyRailsBanner({
  hypothesisId,
  compact = true,
}: {
  hypothesisId: string;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(!compact);
  const { data, isLoading, isError } = useSafetyRailsForHypothesis(hypothesisId);

  if (isLoading) return <div className="text-[10px] text-muted/50">loading rails…</div>;
  if (isError || !data) return null;

  const noSignal = data.rigor.length === 0 && data.audits.length === 0 && !data.belief_family;
  if (noSignal) return null;

  const worstSev = data.n_critical > 0 ? "critical" : data.n_concern > 0 ? "concern" : "ok";
  const beliefHint = data.belief_family?.direction_hint ?? "";
  const beliefAvoid = beliefHint.startsWith("AVOID");
  const hasAnyAlert = data.n_critical > 0 || data.n_flagged > 0 || beliefAvoid;

  const summary = (
    <div className={cn("inline-flex flex-wrap items-center gap-1.5 text-[10.5px]",
                       hasAnyAlert ? "text-foreground" : "text-muted/80")}>
      <Microscope className={cn("h-3 w-3", data.n_flagged > 0 ? "text-alert" : "text-muted/60")} strokeWidth={2.2} />
      <span className={PILL + " " + (data.n_flagged > 0 ? "bg-alert/15 text-alert" : "bg-panel2/40 text-muted")}>
        Rigor {data.rigor.length}{data.n_flagged > 0 ? `·${data.n_flagged}⚠` : ""}
      </span>
      <ShieldCheck className={cn("h-3 w-3", SEV_TONE[worstSev])} strokeWidth={2.2} />
      <span className={PILL + " " + (
        data.n_critical > 0 ? "bg-alert/15 text-alert" :
        data.n_concern  > 0 ? "bg-warn/15 text-warn" :
                              "bg-panel2/40 text-muted"
      )}>
        Audit {data.audits.length}{data.n_critical > 0 ? `·${data.n_critical}c` : ""}{data.n_concern > 0 ? `·${data.n_concern}?` : ""}
      </span>
      {data.belief_family && (
        <>
          <Brain className={cn("h-3 w-3", HINT_TONE(beliefHint))} strokeWidth={2.2} />
          <span className={PILL + " " + (
            beliefAvoid                ? "bg-alert/15 text-alert" :
            beliefHint.startsWith("EXPLORE") ? "bg-ok/15 text-ok" :
                                                "bg-panel2/40 text-muted"
          )} title={beliefHint}>
            {data.belief_family.family || data.belief_family.hyp_family}{" "}
            {data.belief_family.n_obs > 0 ? `${data.belief_family.n_green}G/${data.belief_family.n_marginal}M/${data.belief_family.n_red}R` : "thin"}
          </span>
        </>
      )}
      {hasAnyAlert && (
        <AlertTriangle className="h-3 w-3 text-alert" strokeWidth={2.5} />
      )}
    </div>
  );

  if (compact) {
    return (
      <div className="border border-border/40 rounded-md bg-panel2/30 px-2 py-1.5 space-y-1">
        <button onClick={() => setOpen((v) => !v)}
          className="w-full flex items-center justify-between gap-2 text-left">
          {summary}
          <ChevronDown className={cn("h-3 w-3 text-muted/50 transition-transform", open && "rotate-180")} />
        </button>
        {open && (
          <SafetyRailsDetail data={data} />
        )}
      </div>
    );
  }

  return (
    <div className="border border-border/40 rounded-md bg-panel2/30 px-3 py-2 space-y-2">
      {summary}
      <SafetyRailsDetail data={data} />
    </div>
  );
}


function SafetyRailsDetail({ data }: { data: NonNullable<ReturnType<typeof useSafetyRailsForHypothesis>["data"]> }) {
  return (
    <div className="border-t border-border/30 pt-1.5 space-y-1.5 text-[10.5px] text-muted">
      {/* Rigor rows */}
      {data.rigor.length > 0 && (
        <div className="space-y-0.5">
          <div className="text-[9px] uppercase tracking-wider text-muted/50">Post-GREEN rigor</div>
          {data.rigor.map((r, i) => (
            <div key={i} className="flex flex-wrap items-center gap-1.5 pl-1">
              <span className="font-mono text-foreground/80">{r.template_name || r.family || "—"}</span>
              <span className={r.oos_status === "FAILED" ? "text-alert" : r.oos_status === "SURVIVED" ? "text-ok" : "text-muted/60"}>
                OOS·{r.oos_status?.slice(0,5) || "—"}
              </span>
              <span className={r.spanning_status === "FAILED" ? "text-alert" : r.spanning_status === "PASSED" ? "text-ok" : "text-warn"}>
                Span·{r.spanning_status?.slice(0,5) || "—"}
              </span>
              {r.borrow_status && r.borrow_status !== "SKIPPED" && (
                <span className={r.borrow_status === "KILLED_BY_COST" ? "text-alert" : "text-muted/60"}>
                  Borrow·{r.borrow_status.slice(0,8)}
                </span>
              )}
              {r.flags.length > 0 && (
                <span className="text-alert">⚠ {r.flags.join(", ")}</span>
              )}
              {r.verdict_event_id && (
                <Link href={`/research/verdict?event_id=${r.verdict_event_id}`}
                  className="ml-auto text-accent/70 hover:text-accent text-[9px]">drill →</Link>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Audit rows */}
      {data.audits.length > 0 && (
        <div className="space-y-0.5">
          <div className="text-[9px] uppercase tracking-wider text-muted/50">External audit</div>
          {data.audits.map((a, i) => (
            <div key={i} className="flex flex-wrap items-center gap-1.5 pl-1">
              <span className="font-mono text-foreground/80">{a.provider}</span>
              <span className={SEV_TONE[a.severity]}>{a.severity}</span>
              {a.flagged_categories.length > 0 && (
                <span className="text-muted/70">· {a.flagged_categories.slice(0,3).join(", ")}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Belief */}
      {data.belief_family && (
        <div className="pl-1">
          <span className="text-[9px] uppercase tracking-wider text-muted/50 mr-1.5">Belief</span>
          <span className={HINT_TONE(data.belief_family.direction_hint)}>{data.belief_family.direction_hint}</span>
        </div>
      )}
    </div>
  );
}
