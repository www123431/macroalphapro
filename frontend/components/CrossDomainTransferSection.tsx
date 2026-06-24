"use client";

// β Cross-Domain Transfer UI surface (2026-06-14). Per-sleeve panel
// on /research/library/detail. Reads /api/research/transfers/by_sleeve
// + offers a "Propose transfers" button (~$0.30 per click).
//
// Each proposal: target asset class, mechanism carry, testable spec
// hint, precedent paper, confidence, expected correlation with source.
// Per Frazzini-Pedersen 2018 enhance-vs-new-factor doctrine:
//   exp_corr ≥ 0.5  → likely an ENHANCE candidate (paired bootstrap)
//   exp_corr < 0.5  → likely a NEW factor (forward gate + Bailey-LdP)

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRightLeft, RefreshCw, Loader2, BookOpen } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, cn } from "@/components/ui";


type TransferProposal = {
  proposal_id:                       string;
  source_sleeve_id:                  string;
  source_family:                     string;
  target_asset_class:                string;
  mechanism_carry:                   string;
  testable_spec_hint:                string;
  precedent_paper:                   string;
  confidence:                        number;
  expected_correlation_with_source:  number;
  rationale:                         string;
  proposed_ts:                       string;
  model:                             string;
  cost_usd:                          number;
};


export function CrossDomainTransferSection({ sleeveId }: { sleeveId: string }) {
  const qc = useQueryClient();
  const { data } = useQuery({
    queryKey: ["transfers", sleeveId],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/transfers/by_sleeve/${encodeURIComponent(sleeveId)}`,
                              { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ sleeve_id: string; n: number; rows: TransferProposal[] }>;
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
        `${API_BASE}/api/research/transfers/by_sleeve/${encodeURIComponent(sleeveId)}/generate`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      qc.invalidateQueries({ queryKey: ["transfers", sleeveId] });
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  const rows = data?.rows ?? [];

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <ArrowRightLeft className="h-3.5 w-3.5" strokeWidth={1.75} />
            Cross-asset transfers (Frazzini-Pedersen 70% rule)
            {rows.length > 0 && <span className="text-[10px] text-muted/60">· {rows.length}</span>}
          </span>
        </SectionTitle>
        <button onClick={run} disabled={running}
          className={cn(
            "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px]",
            "border-accent/40 bg-accent/10 text-accent hover:bg-accent/20",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          )}>
          {running
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Proposing…</>
            : <><RefreshCw className="h-3 w-3" /> {rows.length > 0 ? "Re-propose" : "Propose"} (~$0.30)</>}
        </button>
      </div>

      {err && (
        <Card className="border-danger/30 bg-danger/5 mb-2 text-[11px] text-danger">
          {err}
        </Card>
      )}

      {rows.length === 0 ? (
        <p className="text-[11px] text-muted/70 leading-snug">
          No transfer proposals yet. Click "Propose" — single Sonnet call,
          cross-asset thinker persona. Output is 1-2 testable transfers to
          OTHER asset classes (target = enhance pipeline if predicted
          correlation ≥ 0.5, forward gate if {"<"} 0.5).
          ~$0.30, ~15s.
        </p>
      ) : (
        <div className="space-y-2.5">
          {rows.map((p) => {
            const isEnhance = p.expected_correlation_with_source >= 0.5;
            return (
              <div key={p.proposal_id}
                className="rounded border border-border/40 bg-panel2/30 px-3 py-2 space-y-1.5">
                <div className="flex flex-wrap items-center gap-2 text-[11px]">
                  <Badge tone="bg-accent/15 text-accent">
                    → {p.target_asset_class}
                  </Badge>
                  <Badge tone={isEnhance ? "bg-info/15 text-info" : "bg-warn/15 text-warn"}>
                    {isEnhance ? "ENHANCE-class" : "NEW-FACTOR-class"} · corr~{p.expected_correlation_with_source.toFixed(2)}
                  </Badge>
                  <span className="text-muted/70 tnum text-[10px]">
                    conf {(p.confidence * 100).toFixed(0)}%
                  </span>
                  <span className="ml-auto text-muted/60 font-mono text-[9px]">
                    {p.proposed_ts.slice(0, 10)} · ${p.cost_usd.toFixed(3)}
                  </span>
                </div>

                <div className="text-[11.5px] text-foreground/90 leading-snug">
                  <span className="text-muted/70 font-semibold uppercase text-[9px] tracking-wider mr-1">mechanism</span>
                  {p.mechanism_carry}
                </div>

                <div className="text-[11px] text-muted leading-snug border-l-2 border-accent/30 pl-2">
                  <span className="text-muted/70 font-semibold uppercase text-[9px] tracking-wider mr-1">spec hint</span>
                  {p.testable_spec_hint}
                </div>

                {p.precedent_paper && (
                  <div className="text-[10px] text-muted/80 inline-flex items-center gap-1">
                    <BookOpen className="h-2.5 w-2.5" />
                    <span className="italic">{p.precedent_paper}</span>
                  </div>
                )}
              </div>
            );
          })}
          {rows.length > 0 && rows[0].rationale && (
            <p className="text-[10px] italic text-muted/70 leading-snug pt-2 border-t border-border/30">
              <span className="not-italic font-semibold opacity-80">Rationale: </span>
              {rows[0].rationale}
            </p>
          )}
        </div>
      )}

      <p className="text-[9px] italic text-muted/60 mt-2 leading-snug pt-2 border-t border-border/30">
        Frazzini-Pedersen 2018: 70% institutional alpha = enhance, not new factor.
        Single Sonnet call (cross-asset thinker persona). Pattern-5-compliant
        (single agent + schema, no debate). ENHANCE-class proposals route to
        paired-bootstrap pipeline; NEW-FACTOR-class to forward strict gate.
      </p>
    </Card>
  );
}
