"use client";

// PreMortemSection — α Pre-Mortem Generator UI surface (2026-06-14).
//
// Lives on /research/hypothesis. Shows the most recent pre-mortem
// (Skeptic persona's enumerated failure modes) + a "Run pre-mortem"
// button when no report exists yet. Each failure mode has severity,
// category, description, and a concrete check_suggestion the strict
// gate engineer can wire.
//
// Academic anchors (per engine/research/pre_mortem.py): Stigler 1973
// adversarial review > deliberation; Kahneman pre-mortem technique;
// McLean-Pontiff 2016 post-pub decay; Bailey-LdP n_trials inflation.

import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Search, RefreshCw, AlertTriangle, ShieldX, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, cn } from "@/components/ui";


type FailureMode = {
  severity:         "HIGH" | "MEDIUM" | "LOW";
  category:         string;
  description:      string;
  check_suggestion: string;
};
type PreMortemRow = {
  pre_mortem_id:               string;
  hypothesis_id:               string;
  failure_modes:               FailureMode[];
  overall_kill_recommendation: "KILL_BEFORE_TEST" | "TEST_WITH_CAVEATS" | "PROCEED_NORMAL";
  rationale:                   string;
  assessed_ts:                 string;
  model:                       string;
  cost_usd:                    number;
  n_obs_inputs:                number;
};
type PreMortemResponse = {
  hypothesis_id: string;
  n: number;
  rows: PreMortemRow[];
  latest: PreMortemRow | null;
};

const SEV_TONE: Record<string, string> = {
  HIGH:   "bg-alert/15 text-alert border-alert/40",
  MEDIUM: "bg-warn/15 text-warn border-warn/40",
  LOW:    "bg-muted/15 text-muted border-border",
};

const KILL_TONE: Record<string, string> = {
  KILL_BEFORE_TEST:   "bg-alert/15 text-alert border-alert/40",
  TEST_WITH_CAVEATS:  "bg-warn/15 text-warn border-warn/40",
  PROCEED_NORMAL:     "bg-ok/15 text-ok border-ok/40",
};

export function PreMortemSection({ hypothesisId }: { hypothesisId: string }) {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["pre_mortem", hypothesisId],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/pre_mortem/${encodeURIComponent(hypothesisId)}`,
                              { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<PreMortemResponse>;
    },
    staleTime: 60_000,
  });
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/pre_mortem/${encodeURIComponent(hypothesisId)}/generate`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      qc.invalidateQueries({ queryKey: ["pre_mortem", hypothesisId] });
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  const latest = data?.latest ?? null;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <Search className="h-3.5 w-3.5" strokeWidth={1.75} />
            Pre-mortem (Skeptic persona) {data && data.n > 1 && (
              <span className="text-[10px] text-muted/60">· {data.n} runs</span>
            )}
          </span>
        </SectionTitle>
        <button onClick={run} disabled={running}
          className={cn(
            "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px]",
            "border-accent/40 bg-accent/10 text-accent hover:bg-accent/20",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          )}>
          {running
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Running…</>
            : latest
              ? <><RefreshCw className="h-3 w-3" /> Re-run (~$0.05)</>
              : <><Search className="h-3 w-3" /> Run pre-mortem (~$0.05)</>}
        </button>
      </div>

      {err && (
        <Card className="border-danger/30 bg-danger/5 mb-2 text-[11px] text-danger">
          {err}
        </Card>
      )}

      {isLoading && !data && (
        <Card className="text-[11px] text-muted/70">loading…</Card>
      )}

      {data && !latest && (
        <Card className="border-border/40 bg-panel2/20 text-[11px] text-muted/70">
          No pre-mortem yet. Click "Run pre-mortem" — Sonnet enumerates
          concrete failure modes the strict gate might miss
          (Stigler 1973 adversarial review). ~$0.05, ~15s.
        </Card>
      )}

      {latest && (
        <Card className="border-border/40 bg-panel2/20 space-y-3">
          {/* Header */}
          <div className="flex flex-wrap items-center gap-3 text-[11px]">
            <Badge tone={KILL_TONE[latest.overall_kill_recommendation]}>
              {latest.overall_kill_recommendation === "KILL_BEFORE_TEST" && <ShieldX className="h-3 w-3 inline mr-1" />}
              {latest.overall_kill_recommendation.replace(/_/g, " ")}
            </Badge>
            <span className="text-muted/60 font-mono">
              {latest.assessed_ts.slice(0, 16).replace("T", " ")}
            </span>
            <span className="text-muted/60">{latest.model}</span>
            <span className="ml-auto text-muted/60 tnum">
              ${latest.cost_usd.toFixed(3)} · {latest.failure_modes.length} modes · {latest.n_obs_inputs} context
            </span>
          </div>

          {/* Rationale */}
          <p className="text-[12px] text-foreground/90 leading-relaxed border-l-2 border-accent/40 pl-3 italic">
            {latest.rationale}
          </p>

          {/* Failure modes */}
          <div className="space-y-2">
            {latest.failure_modes.map((fm, i) => (
              <div key={i} className={cn(
                "rounded border px-2.5 py-1.5 space-y-1",
                SEV_TONE[fm.severity] || "border-border bg-panel2/40",
              )}>
                <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider">
                  <AlertTriangle className="h-3 w-3" strokeWidth={2.2} />
                  <span>{fm.severity}</span>
                  <span className="opacity-60">·</span>
                  <span className="font-mono">{fm.category}</span>
                </div>
                <div className="text-[11.5px] text-foreground/90 leading-snug">
                  {fm.description}
                </div>
                <div className="text-[10.5px] text-muted leading-snug border-t border-current/10 pt-1">
                  <span className="font-semibold opacity-80">CHECK: </span>
                  {fm.check_suggestion}
                </div>
              </div>
            ))}
          </div>

          <p className="text-[9px] italic text-muted/60 leading-snug pt-1 border-t border-border/30">
            Single Sonnet call, Stigler-1973 adversarial review (not multi-agent debate).
            Context fed: family belief layer, graveyard nearest-3 collisions, Bailey-LdP n_trials,
            strict-gate known silent bugs. Pre-mortem is generative for the gate (suggests checks),
            NOT a verdict vote (Pattern 5 ban).
          </p>
        </Card>
      )}
    </div>
  );
}
