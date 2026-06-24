"use client";

// γ Replication Checker UI surface (2026-06-14). Per-hypothesis panel
// on /research/hypothesis. Single Sonnet call with lit-aware persona:
// "does this hypothesis match a known replication-failure catalog
// entry (Hou-Xue-Zhang 2020 / McLean-Pontiff 2016 / Linnainmaa-
// Roberts 2018 / Fama-French 2018)?"
//
// Output: replication_status enum + matched papers + Sharpe-decay
// multiplier. Complements α (general adversarial) with literature-
// replication evidence.

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, RefreshCw, Loader2, AlertTriangle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, cn } from "@/components/ui";


type ReplicationFlag = {
  matched_paper:             string;
  replication_evidence:      string;
  estimated_alpha_decay_pct: number;
  confidence:                number;
};
type ReplicationRow = {
  check_id:                   string;
  hypothesis_id:              string;
  replication_status:         "PROBABLY_DEAD" | "DECAYED_BUT_LIVE" | "WORTH_TESTING" | "NOT_FOUND_IN_LIT";
  flags:                      ReplicationFlag[];
  rationale:                  string;
  est_post_pub_sharpe_factor: number;
  assessed_ts:                string;
  model:                      string;
  cost_usd:                   number;
};
type ReplicationResponse = {
  hypothesis_id: string;
  n: number;
  rows: ReplicationRow[];
  latest: ReplicationRow | null;
};

const STATUS_TONE: Record<string, string> = {
  PROBABLY_DEAD:    "bg-alert/15 text-alert border-alert/40",
  DECAYED_BUT_LIVE: "bg-warn/15 text-warn border-warn/40",
  WORTH_TESTING:    "bg-ok/15 text-ok border-ok/40",
  NOT_FOUND_IN_LIT: "bg-info/15 text-info border-info/40",
};

export function ReplicationCheckSection({ hypothesisId }: { hypothesisId: string }) {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["replication_check", hypothesisId],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/replication/${encodeURIComponent(hypothesisId)}`,
                              { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<ReplicationResponse>;
    },
    staleTime: 60_000,
  });
  const [running, setRunning] = useState(false);
  const [err, setErr]         = useState<string | null>(null);

  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/replication/${encodeURIComponent(hypothesisId)}/generate`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      qc.invalidateQueries({ queryKey: ["replication_check", hypothesisId] });
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
            <BookOpen className="h-3.5 w-3.5" strokeWidth={1.75} />
            Replication checker (lit catalog match)
            {data && data.n > 1 && (
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
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Checking…</>
            : latest
              ? <><RefreshCw className="h-3 w-3" /> Re-check (~$0.05)</>
              : <><BookOpen className="h-3 w-3" /> Check lit (~$0.05)</>}
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
          No lit check yet. Click "Check lit" — single Sonnet call against
          Hou-Xue-Zhang 2020 / McLean-Pontiff 2016 / Linnainmaa-Roberts
          2018 / Fama-French 2018 catalogs. ~$0.05, ~10s.
        </Card>
      )}

      {latest && (
        <Card className="border-border/40 bg-panel2/20 space-y-3">
          {/* Header */}
          <div className="flex flex-wrap items-center gap-3 text-[11px]">
            <Badge tone={STATUS_TONE[latest.replication_status]}>
              {latest.replication_status === "PROBABLY_DEAD" && <AlertTriangle className="h-3 w-3 inline mr-1" />}
              {latest.replication_status.replace(/_/g, " ")}
            </Badge>
            <span className="tnum">
              <span className="text-muted/60 text-[9px] uppercase mr-1">post-pub Sh ×</span>
              <span className={cn(
                "font-mono font-semibold",
                latest.est_post_pub_sharpe_factor < 0.4 ? "text-alert" :
                latest.est_post_pub_sharpe_factor < 0.7 ? "text-warn"  :
                                                          "text-ok",
              )}>
                {latest.est_post_pub_sharpe_factor.toFixed(2)}
              </span>
            </span>
            <span className="text-muted/60 font-mono">
              {latest.assessed_ts.slice(0, 16).replace("T", " ")}
            </span>
            <span className="ml-auto text-muted/60 tnum">
              ${latest.cost_usd.toFixed(3)} · {latest.flags.length} flags
            </span>
          </div>

          {/* Rationale */}
          <p className="text-[12px] text-foreground/90 leading-relaxed border-l-2 border-accent/40 pl-3 italic">
            {latest.rationale}
          </p>

          {/* Flags */}
          {latest.flags.length === 0 ? (
            <p className="text-[11px] text-muted/70 italic">
              No matched catalog papers (genuine novelty OR no strong lit prior).
            </p>
          ) : (
            <div className="space-y-2">
              {latest.flags.map((f, i) => (
                <div key={i} className="rounded border border-border/40 bg-panel/40 px-2.5 py-1.5 space-y-1">
                  <div className="flex items-center gap-2 text-[11px]">
                    <BookOpen className="h-3 w-3 text-accent shrink-0" />
                    <span className="font-mono text-foreground/90 truncate">{f.matched_paper}</span>
                    <span className="ml-auto tnum text-[10px]">
                      <span className="text-muted/70 mr-1">decay</span>
                      <span className={cn(
                        f.estimated_alpha_decay_pct >= 0.5 ? "text-alert" :
                        f.estimated_alpha_decay_pct >= 0.3 ? "text-warn"  :
                                                              "text-muted/80",
                      )}>
                        {(f.estimated_alpha_decay_pct * 100).toFixed(0)}%
                      </span>
                      <span className="text-muted/60 ml-2">conf {(f.confidence * 100).toFixed(0)}%</span>
                    </span>
                  </div>
                  <div className="text-[10.5px] text-muted leading-snug">
                    {f.replication_evidence}
                  </div>
                </div>
              ))}
            </div>
          )}

          <p className="text-[9px] italic text-muted/60 leading-snug pt-1 border-t border-border/30">
            Single Sonnet call against the published anomaly literature.
            Catalogs cited: Hou-Xue-Zhang 2020 (~50% lit replication
            failure under q-factor) / McLean-Pontiff 2016 (~58% mean
            post-pub Sharpe drop) / Linnainmaa-Roberts 2018 / Fama-French
            2018. Pattern-5-compliant (single agent + strict schema, no
            debate). γ complements α (general adversarial) and β
            (cross-asset transfer).
          </p>
        </Card>
      )}
    </div>
  );
}
