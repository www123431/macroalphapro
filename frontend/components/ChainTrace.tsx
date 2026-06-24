"use client";

// ChainTrace — visualizes the PAPER → HYPOTHESIS → TEST → VERDICT chain
// as a 4-stage horizontal swim-lane. Used on lesson detail (showing the
// chain that led to the verdict) and on forward vector detail (showing
// the chain that's PROPOSED but not yet tested).

import Link from "next/link";
import { Badge } from "@/components/ui";


export type ChainStage = {
  label:    string;         // e.g. "PAPER", "HYPOTHESIS"
  kind:     "paper" | "hypothesis" | "test" | "verdict";
  title:    string;         // main display string
  subtitle: string;         // secondary
  href?:    string;         // optional link
  status?:  "done" | "pending" | "skipped";
};


const KIND_TONE: Record<ChainStage["kind"], string> = {
  paper:      "bg-info/10 border-info/40 text-info",
  hypothesis: "bg-accent/10 border-accent/40 text-accent",
  test:       "bg-warn/10 border-warn/40 text-warn",
  verdict:    "bg-ok/10 border-ok/40 text-ok",
};

const STATUS_TONE: Record<NonNullable<ChainStage["status"]>, string> = {
  done:    "border-ok/60 ring-1 ring-ok/30",
  pending: "border-warn/60 border-dashed",
  skipped: "border-muted/40 opacity-60",
};


export default function ChainTrace({ stages }: { stages: ChainStage[] }) {
  return (
    <div className="flex items-stretch gap-2 overflow-x-auto py-3">
      {stages.map((stage, i) => (
        <div key={i} className="flex items-stretch flex-shrink-0">
          <div className={`flex flex-col rounded border ${
              KIND_TONE[stage.kind]
            } ${stage.status ? STATUS_TONE[stage.status] : ""} p-3 min-w-[200px] max-w-[260px]`}>
            <div className="text-[10px] uppercase tracking-wider text-muted mb-1">
              {stage.label}
            </div>
            {stage.href ? (
              <Link href={stage.href}
                    className="text-sm font-medium hover:underline line-clamp-2">
                {stage.title}
              </Link>
            ) : (
              <div className="text-sm font-medium line-clamp-2">{stage.title}</div>
            )}
            <div className="text-xs text-muted mt-auto pt-1 line-clamp-2">
              {stage.subtitle}
            </div>
            {stage.status && (
              <Badge className={`mt-1 self-start text-[9px] ${
                stage.status === "done"    ? "bg-ok/20 text-ok"     :
                stage.status === "pending" ? "bg-warn/20 text-warn" :
                                              "bg-muted/20 text-muted"
              }`}>
                {stage.status}
              </Badge>
            )}
          </div>

          {/* Arrow between stages */}
          {i < stages.length - 1 && (
            <div className="flex items-center px-1 text-muted text-xl">→</div>
          )}
        </div>
      ))}
    </div>
  );
}
