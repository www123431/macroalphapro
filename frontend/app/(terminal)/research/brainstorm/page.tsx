"use client";

// /research/brainstorm — Phase 4 UI surface for the brainstorm system.
//
// Pattern: 4-panel page
//   Top    — 7 seed pack chips + Run Brainstorm button + cost cap badge
//   Middle — Drafts table, newest first, with inline α/γ vet expansion
//   Right  — PM Promote / Reject panel (when draft selected)
//
// The "promote" path writes a hypothesis row to hypotheses.jsonl with
// lineage back to the brainstorm draft. Rejected drafts stay visible
// for calibration tracking (later: PM decision quality vs verdict).

import { Suspense, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  Sparkles, Loader2, CheckCircle2, XCircle, RefreshCw,
  BookOpen, ChevronDown, ChevronUp, AlertTriangle,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { ModeHeader } from "@/components/ModeHeader";
import { SafetyRailsBanner } from "@/components/SafetyRailsBanner";
import { PreMortemSection } from "@/components/PreMortemSection";
import { ReplicationCheckSection } from "@/components/ReplicationCheckSection";


type Pack = {
  name: string;
  domain: string;
  short_description: string;
  n_principles: number;
  n_examples: number;
};

type Idea = {
  idea_id:               string;
  session_id:            string;
  source_pack:           string;
  source_provider:       string;
  claim_one_line:        string;
  target_asset_class:    string;
  expected_mechanism:    string;
  data_required:         string[];
  novelty_self_score:    number;
  falsifier:             string;
  precedent_paper:       string;
  lessons_invoked:       string[];
  generated_ts:          string;
  cost_usd:              number;
  decision: null | {
    decision_id: string;
    decision:    "promote" | "reject";
    rationale:   string;
    decided_by:  string;
    decided_ts:  string;
    new_hypothesis_id?: string;
  };
};

const PACK_TONE: Record<string, string> = {
  physics_analogies:              "bg-info/15 text-info",
  network_theory:                 "bg-accent/15 text-accent",
  behavioral_inverse:             "bg-warn/15 text-warn",
  alternative_data:               "bg-ok/15 text-ok",
  macro_regime_shifts:            "bg-info/15 text-info",
  cross_section_anomaly_inversion:"bg-warn/15 text-warn",
  time_horizon_arbitrage:         "bg-accent/15 text-accent",
};


export default function BrainstormPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <BrainstormInner />
    </Suspense>
  );
}

function BrainstormInner() {
  const qc = useQueryClient();
  const [selectedPack, setSelectedPack] = useState<string>("physics_analogies");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [expandedIdea, setExpandedIdea] = useState<string | null>(null);

  const packsQ = useQuery({
    queryKey: ["brainstorm", "packs"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/brainstorm/seed_packs`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n: number; packs: Pack[] }>;
    },
    staleTime: 5 * 60_000,
  });

  const draftsQ = useQuery({
    queryKey: ["brainstorm", "drafts"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/brainstorm/drafts?limit=50`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n: number; rows: Idea[] }>;
    },
    refetchInterval: 30_000,
  });

  const runBrainstorm = async () => {
    setRunning(true);
    setRunError(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/brainstorm/run?pack=${encodeURIComponent(selectedPack)}&trigger=manual`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      qc.invalidateQueries({ queryKey: ["brainstorm", "drafts"] });
    } catch (e: any) {
      setRunError(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  const drafts = draftsQ.data?.rows ?? [];
  const promoted = drafts.filter((d) => d.decision?.decision === "promote").length;
  const rejected = drafts.filter((d) => d.decision?.decision === "reject").length;
  const pending  = drafts.length - promoted - rejected;

  return (
    <motion.div variants={stagger(0.05)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="research"
          title="Brainstorm — experience-conditioned divergent generator"
          subtitle="Sonnet single-call against curated cross-domain seed packs + lessons distilled from our verdict history. Every idea has mandatory Popper-falsifier. PM reviews + promotes to hypothesis queue."
        />
      </motion.div>

      {/* KPI strip */}
      <motion.div variants={fadeUp}>
        <Card className="py-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[11px]">
            <KpiCell label="Drafts total"    value={drafts.length} tone="muted" />
            <KpiCell label="Pending review"  value={pending}  tone={pending > 0 ? "warn" : "muted"} />
            <KpiCell label="Promoted"        value={promoted} tone="ok" />
            <KpiCell label="Rejected"        value={rejected} tone="alert" />
          </div>
        </Card>
      </motion.div>

      {/* Pack selector + run */}
      <motion.div variants={fadeUp}>
        <Card>
          <SectionTitle>
            <span className="inline-flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5" strokeWidth={1.75} />
              Seed packs (cross-domain priors)
            </span>
          </SectionTitle>

          {packsQ.isLoading && <Skeleton className="h-24 w-full" />}
          {packsQ.data && (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 mt-2">
                {packsQ.data.packs.map((pk) => {
                  const sel = pk.name === selectedPack;
                  return (
                    <button key={pk.name}
                      onClick={() => setSelectedPack(pk.name)}
                      className={cn(
                        "text-left rounded-md border px-3 py-2 transition-colors",
                        sel
                          ? "border-accent/60 bg-accent/10 ring-1 ring-accent/40"
                          : "border-border/40 hover:border-accent/30 hover:bg-panel2/40",
                      )}>
                      <div className="flex items-center justify-between gap-2 mb-0.5">
                        <Badge tone={PACK_TONE[pk.name] || "bg-muted/15 text-muted"}>
                          {pk.name}
                        </Badge>
                        <span className="text-[9px] text-muted/60">
                          {pk.n_principles}p · {pk.n_examples}ex
                        </span>
                      </div>
                      <div className="text-[10.5px] text-muted leading-snug">
                        {pk.short_description}
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="mt-3 pt-3 border-t border-border/30 flex items-center gap-3">
                <button onClick={runBrainstorm} disabled={running}
                  className={cn(
                    "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-medium",
                    "bg-accent text-background hover:bg-accent/90",
                    "disabled:opacity-50 disabled:cursor-not-allowed",
                  )}>
                  {running
                    ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Brainstorming…</>
                    : <><Sparkles className="h-3.5 w-3.5" /> Run brainstorm on {selectedPack} (~$0.05)</>}
                </button>
                <span className="text-[10px] text-muted/70 leading-snug">
                  3-5 ideas, each w/ mandatory falsifier. ~15-30s.
                </span>
                {runError && (
                  <span className="text-[11px] text-danger ml-auto">{runError}</span>
                )}
              </div>
            </>
          )}
        </Card>
      </motion.div>

      {/* Drafts table */}
      <motion.div variants={fadeUp}>
        <SectionTitle>
          Drafts ({drafts.length}, newest first)
        </SectionTitle>
        {draftsQ.isLoading && !draftsQ.data && <Skeleton className="h-32 w-full" />}
        {drafts.length === 0 ? (
          <Card className="text-[12px] text-muted">
            No drafts yet. Pick a seed pack above + Run brainstorm.
          </Card>
        ) : (
          <div className="space-y-2.5">
            {drafts.map((idea) => (
              <IdeaRow key={idea.idea_id}
                idea={idea}
                expanded={expandedIdea === idea.idea_id}
                onToggle={() => setExpandedIdea(
                  expandedIdea === idea.idea_id ? null : idea.idea_id)}
                onDecision={() => qc.invalidateQueries({ queryKey: ["brainstorm", "drafts"] })} />
            ))}
          </div>
        )}
      </motion.div>

      <p className="text-[10px] italic text-muted/60 leading-snug pt-3 border-t border-border/30">
        Phase 2 single-Sonnet MVP. Phase 3 (multi-provider + dedup) +
        Phase 5 (demand-driven cron triggers) pending. Each promoted
        idea writes a hypothesis.jsonl row with extraction_method=
        LLM_BRAINSTORM_&lt;pack&gt; for lineage. PM decision rationale is
        mandatory (audit P1 accountability).
      </p>
    </motion.div>
  );
}


function KpiCell({ label, value, tone }: {
  label: string; value: number;
  tone: "ok" | "warn" | "alert" | "muted";
}) {
  const toneCls =
    tone === "ok"    ? "text-ok"    :
    tone === "warn"  ? "text-warn"  :
    tone === "alert" ? "text-alert" :
                       "text-muted";
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-muted/70">{label}</div>
      <div className={cn("tnum text-xl font-semibold", toneCls)}>{value}</div>
    </div>
  );
}


function IdeaRow({ idea, expanded, onToggle, onDecision }: {
  idea: Idea; expanded: boolean; onToggle: () => void; onDecision: () => void;
}) {
  const dec = idea.decision;
  const tone =
    dec?.decision === "promote" ? "border-ok/40 bg-ok/[0.03]" :
    dec?.decision === "reject"  ? "border-danger/30 bg-danger/[0.02] opacity-70" :
                                   "border-border/40 bg-panel2/20";
  return (
    <Card className={cn("space-y-2 p-3", tone)}>
      {/* Top row — pack badge + claim + decision badge + toggle */}
      <button onClick={onToggle}
        className="w-full text-left flex items-start gap-3">
        <div className="shrink-0 mt-0.5">
          <Badge tone={PACK_TONE[idea.source_pack] || "bg-muted/15 text-muted"}>
            {idea.source_pack}
          </Badge>
        </div>
        <div className="flex-1 min-w-0 space-y-0.5">
          <div className="text-[12.5px] text-foreground/90 leading-snug">
            {idea.claim_one_line}
          </div>
          <div className="text-[10px] text-muted/70 flex flex-wrap gap-x-3 gap-y-0.5">
            <span>→ {idea.target_asset_class}</span>
            <span>novelty {(idea.novelty_self_score * 100).toFixed(0)}%</span>
            <span className="font-mono">{idea.idea_id.slice(0, 8)}</span>
            <span>{idea.generated_ts.slice(0, 16).replace("T", " ")}</span>
            <span>${idea.cost_usd.toFixed(3)}</span>
          </div>
        </div>
        <div className="shrink-0 flex items-center gap-1.5">
          {dec && (
            <Badge tone={dec.decision === "promote" ? "bg-ok/15 text-ok" : "bg-danger/15 text-danger"}>
              {dec.decision.toUpperCase()}
            </Badge>
          )}
          {expanded ? <ChevronUp className="h-4 w-4 text-muted/60" />
                    : <ChevronDown className="h-4 w-4 text-muted/60" />}
        </div>
      </button>

      {expanded && (
        <div className="border-t border-border/30 pt-2 space-y-3">
          {/* Mechanism + falsifier + data */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 text-[11px]">
            <div className="space-y-1">
              <div className="text-[9px] uppercase tracking-wider text-muted/70">
                Expected mechanism
              </div>
              <div className="text-foreground/85">{idea.expected_mechanism}</div>
            </div>
            <div className="space-y-1">
              <div className="text-[9px] uppercase tracking-wider text-muted/70">
                <AlertTriangle className="h-2.5 w-2.5 inline mr-0.5 text-warn" />
                Falsifier (MANDATORY)
              </div>
              <div className="text-foreground/85 leading-snug">{idea.falsifier}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 text-[10.5px]">
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5">
                Data required
              </div>
              <ul className="space-y-0.5">
                {idea.data_required.map((d, i) => (
                  <li key={i} className="font-mono text-muted">· {d}</li>
                ))}
              </ul>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5">
                Lessons invoked (Layer 1)
              </div>
              <ul className="space-y-0.5">
                {idea.lessons_invoked.slice(0, 5).map((l, i) => (
                  <li key={i} className="text-muted/85">· {l}</li>
                ))}
              </ul>
              {idea.precedent_paper && (
                <div className="mt-1.5 inline-flex items-center gap-1 text-[10px] italic text-muted/80">
                  <BookOpen className="h-2.5 w-2.5" />
                  {idea.precedent_paper}
                </div>
              )}
            </div>
          </div>

          {/* Pre-vet (γ replication check) — before decision */}
          {!dec && <PreVetSection ideaId={idea.idea_id} />}

          {/* Decision row */}
          {dec ? (
            <DecisionDisplay decision={dec} />
          ) : (
            <DecisionForm ideaId={idea.idea_id} onDecided={onDecision} />
          )}
        </div>
      )}
    </Card>
  );
}


function DecisionDisplay({ decision }: { decision: NonNullable<Idea["decision"]> }) {
  return (
    <div className="border-t border-border/30 pt-2 text-[10.5px]">
      <span className="text-muted/70">Decision: </span>
      <Badge tone={decision.decision === "promote" ? "bg-ok/15 text-ok" : "bg-danger/15 text-danger"}>
        {decision.decision.toUpperCase()}
      </Badge>
      <span className="text-muted/70 ml-2">by {decision.decided_by} ·</span>
      <span className="font-mono text-muted/60 ml-1">
        {decision.decided_ts.slice(0, 16).replace("T", " ")}
      </span>
      {decision.rationale && (
        <p className="mt-1 text-muted">
          <span className="font-semibold opacity-80">Rationale: </span>
          {decision.rationale}
        </p>
      )}
      {decision.new_hypothesis_id && (
        <p className="mt-1 text-[10px]">
          <span className="text-muted/70">New hypothesis: </span>
          <Link href={`/research/hypothesis?id=${decision.new_hypothesis_id}`}
            className="font-mono text-accent hover:underline">
            {decision.new_hypothesis_id.slice(0, 8)}…
          </Link>
        </p>
      )}
    </div>
  );
}


function PreVetSection({ ideaId }: { ideaId: string }) {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setRunning(true);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/brainstorm/drafts/${encodeURIComponent(ideaId)}/prevet`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      setResult(await r.json());
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };
  const STATUS_TONE: Record<string, string> = {
    PROBABLY_DEAD:    "bg-alert/15 text-alert border-alert/40",
    DECAYED_BUT_LIVE: "bg-warn/15 text-warn border-warn/40",
    WORTH_TESTING:    "bg-ok/15 text-ok border-ok/40",
    NOT_FOUND_IN_LIT: "bg-info/15 text-info border-info/40",
  };
  return (
    <div className="border-t border-border/30 pt-2 space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-[9px] uppercase tracking-wider text-muted/70">
          Pre-vet (γ replication check vs HXZ/MP/LR/FF catalogs)
        </span>
        <button onClick={run} disabled={running}
          className="ml-auto inline-flex items-center gap-1 px-2 py-0.5 rounded border border-accent/40 bg-accent/10 text-accent text-[10.5px] hover:bg-accent/20 disabled:opacity-40">
          {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <BookOpen className="h-3 w-3" />}
          {result ? "Re-check" : "Pre-vet"} (~$0.05)
        </button>
      </div>
      {err && <p className="text-[10.5px] text-danger">{err}</p>}
      {result && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-[11px]">
            <Badge tone={STATUS_TONE[result.replication_status] || ""}>
              {result.replication_status.replace(/_/g, " ")}
            </Badge>
            <span className="tnum text-[10px]">
              <span className="text-muted/60">post-pub Sh ×</span>{" "}
              <span className={cn(
                "font-mono font-semibold",
                result.est_post_pub_sharpe_factor < 0.4 ? "text-alert" :
                result.est_post_pub_sharpe_factor < 0.7 ? "text-warn" : "text-ok",
              )}>{result.est_post_pub_sharpe_factor.toFixed(2)}</span>
            </span>
            <span className="ml-auto text-[9px] text-muted/60 tnum">
              ${result.cost_usd.toFixed(3)} · {result.flags.length} flags
            </span>
          </div>
          {result.rationale && (
            <p className="text-[10.5px] text-muted italic border-l-2 border-accent/30 pl-2">
              {result.rationale}
            </p>
          )}
          {result.flags.length > 0 && (
            <div className="space-y-1">
              {result.flags.slice(0, 3).map((f: any, i: number) => (
                <div key={i} className="text-[10.5px] flex items-start gap-1.5">
                  <BookOpen className="h-2.5 w-2.5 mt-0.5 shrink-0 text-accent" />
                  <div className="flex-1 min-w-0">
                    <div className="font-mono text-foreground/85 truncate">{f.matched_paper}</div>
                    <div className="text-muted/80 leading-snug">{f.replication_evidence}</div>
                    <div className="text-[9.5px] tnum text-muted/60">
                      decay {(f.estimated_alpha_decay_pct * 100).toFixed(0)}% · conf {(f.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


function DecisionForm({ ideaId, onDecided }: {
  ideaId: string; onDecided: () => void;
}) {
  const [rationale, setRationale] = useState("");
  const [submitting, setSubmitting] = useState<"promote"|"reject"|null>(null);
  const [err, setErr] = useState<string | null>(null);

  const submit = async (decision: "promote" | "reject") => {
    if (rationale.trim().length < 5) {
      setErr("rationale required (min 5 chars)");
      return;
    }
    setSubmitting(decision);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/brainstorm/drafts/${encodeURIComponent(ideaId)}/decide`
          + `?decision=${decision}&rationale=${encodeURIComponent(rationale)}`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      onDecided();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setSubmitting(null);
    }
  };

  return (
    <div className="border-t border-border/30 pt-2 space-y-2">
      <div className="text-[9px] uppercase tracking-wider text-muted/70">
        Decision (rationale required — Pattern-5 accountability)
      </div>
      <input type="text" value={rationale}
        onChange={(e) => setRationale(e.target.value)}
        placeholder="why promote OR reject — be specific"
        disabled={submitting !== null}
        className="w-full px-2 py-1 text-[11px] bg-panel2/40 border border-border/40 rounded focus:border-accent/50 focus:outline-none" />
      <div className="flex items-center gap-2">
        <button onClick={() => submit("promote")}
          disabled={submitting !== null || rationale.trim().length < 5}
          className={cn(
            "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px] font-medium",
            "bg-ok/10 text-ok border-ok/40 hover:bg-ok/20",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          )}>
          {submitting === "promote"
            ? <Loader2 className="h-3 w-3 animate-spin" />
            : <CheckCircle2 className="h-3 w-3" />}
          Promote → hypothesis
        </button>
        <button onClick={() => submit("reject")}
          disabled={submitting !== null || rationale.trim().length < 5}
          className={cn(
            "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px] font-medium",
            "bg-danger/10 text-danger border-danger/40 hover:bg-danger/20",
            "disabled:opacity-40 disabled:cursor-not-allowed",
          )}>
          {submitting === "reject"
            ? <Loader2 className="h-3 w-3 animate-spin" />
            : <XCircle className="h-3 w-3" />}
          Reject
        </button>
        {err && <span className="text-[10px] text-danger ml-1">{err}</span>}
      </div>
    </div>
  );
}
