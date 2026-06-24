"use client";

// /research/papers/incoming — Employee A daily digest.
//
// Phase 1.6 (2026-06-05) — closes the loop on the auto-discovery chain.
// Shows what arxiv q-fin papers came in over the last N days, ordered
// by the summarizer's recommended_action (INGEST > READ_AND_DISCARD >
// SKIP > unrated). Each row carries the 5-field summary the user can
// scan in 30 seconds + buttons for the 3 fates: ingest now / open arxiv
// to read / dismiss.
//
// Layout intentionally dense (one card per paper, no expansion needed)
// so the user can scroll through all of today's candidates without
// hunting for the "is this worth it" judgment.
//
// Source: GET /api/papers_curator/incoming?days=14

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  PlayCircle, BookOpen, MinusCircle, ExternalLink,
  RefreshCw, Loader2, AlertTriangle, FileText, Sparkles, X,
  GitMerge, Eye, Save,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { useI18n } from "@/lib/i18n";


// ── Phase 2.0 step 5c: cross-source synthesis trigger panel ─────────
// Lives ABOVE the daily digest because synthesis is the higher-order
// activity (the digest is per-paper triage; synthesis is across-papers
// + sleeves + events). Single-shot button — NOT auto-fired on mount;
// each click costs ≤ $0.10. Dry-run is the safe default.
type SynthCandidate = {
  claim: string;
  mechanism_family: string;
  mechanism_subtype: string;
  predicted_direction: string;
  predicted_magnitude: string;
  required_data: string[];
  test_methodology: string;
  synthesizes_paper_ids: string[];
  synthesizes_event_ids: string[];
  addresses_decay_in: string | null;
  cochrane_frame: string;
  novelty_vs_known: string;
  estimated_n_trials_in_family: number;
  graveyard_conflicts: string[];
  doctrine_conflicts: string[];
  expected_outcome_prior: string;
  generation_ts: string;
  model: string;
};

type SynthResult = {
  run_ts: string;
  dry_run: boolean;
  snapshot: {
    snapshot_ts: string;
    recent_summaries: number;
    deployed_sleeves: number;
    recent_events: number;
    doctrine_snippets: number;
  };
  candidates: SynthCandidate[];
  n_candidates: number;
  written_hypothesis_ids: string[];
  n_written: number;
  errors: string[];
};


function SynthesisPanel() {
  const { t } = useI18n();
  const [snapshot, setSnapshot] = useState<SynthResult["snapshot"] | null>(null);
  const [result, setResult]     = useState<SynthResult | null>(null);
  const [running, setRunning]   = useState(false);
  const [snapErr, setSnapErr]   = useState<string | null>(null);

  // Cheap snapshot read on mount — just shows "Snapshot: X papers,
  // Y sleeves, Z events" so the user knows what the LLM will see
  // BEFORE they click. Uses dry-run so it's safe even on the auto-load.
  // Wait — dry-run still calls the LLM (~$0.05). Don't auto-fire.
  // Instead, leave snapshot null until the user hits a button.
  // The button-bar shows generic "click to preview" until then.

  const runSynthesis = async (dryRun: boolean) => {
    setRunning(true);
    setSnapErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/papers_curator/synthesis/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dry_run: dryRun }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body: SynthResult = await r.json();
      setResult(body);
      setSnapshot(body.snapshot);
    } catch (e: any) {
      setSnapErr(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card className="p-0 overflow-hidden border border-accent/30">
      <div className="px-3 py-2 border-b border-border/30 bg-accent/5 flex items-center gap-2">
        <GitMerge className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
        <span className="text-[11.5px] font-semibold text-foreground">
          {t("synth.title")}
        </span>
        <span className="text-[10px] text-muted/60 ml-auto">{t("synth.cost_hint")}</span>
      </div>

      <div className="px-3 py-2 space-y-2">
        <p className="text-[10.5px] text-muted/80 leading-snug">{t("synth.subtitle")}</p>

        {snapshot && (
          <div className="text-[10.5px] flex flex-wrap items-center gap-x-3 gap-y-0.5">
            <span><span className="text-muted/70">{t("synth.snapshot.papers")}: </span>
              <span className="font-mono">{snapshot.recent_summaries}</span></span>
            <span><span className="text-muted/70">{t("synth.snapshot.sleeves")}: </span>
              <span className="font-mono">{snapshot.deployed_sleeves}</span></span>
            <span><span className="text-muted/70">{t("synth.snapshot.events")}: </span>
              <span className="font-mono">{snapshot.recent_events}</span></span>
            <span><span className="text-muted/70">{t("synth.snapshot.doctrine")}: </span>
              <span className="font-mono">{snapshot.doctrine_snippets}</span></span>
          </div>
        )}

        <div className="flex items-center gap-2 pt-1">
          <button
            onClick={() => runSynthesis(true)}
            disabled={running}
            className={cn(
              "px-2.5 py-1.5 rounded text-[10.5px] font-medium inline-flex items-center gap-1.5",
              "border border-border/40 hover:border-accent/50 hover:bg-accent/5",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}>
            {running ? <Loader2 className="h-3 w-3 animate-spin" />
                      : <Eye className="h-3 w-3" />}
            {running ? t("synth.btn.running") : t("synth.btn.preview")}
          </button>
          <button
            onClick={() => runSynthesis(false)}
            disabled={running}
            className={cn(
              "px-2.5 py-1.5 rounded text-[10.5px] font-medium inline-flex items-center gap-1.5",
              "bg-accent/10 text-accent border border-accent/40",
              "hover:bg-accent/20",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}>
            {running ? <Loader2 className="h-3 w-3 animate-spin" />
                      : <Save className="h-3 w-3" />}
            {running ? t("synth.btn.running") : t("synth.btn.persist")}
          </button>
        </div>

        {snapErr && (
          <div className="text-[10.5px] text-danger inline-flex items-center gap-1.5">
            <AlertTriangle className="h-3 w-3" /> {snapErr}
          </div>
        )}

        {result && <SynthesisResultBody result={result} />}
      </div>
    </Card>
  );
}


function SynthesisResultBody({ result }: { result: SynthResult }) {
  const { t } = useI18n();

  // Status banner: empty / preview-only / persisted
  let banner: { tone: "info" | "ok" | "warn"; text: string };
  if (result.n_candidates === 0) {
    banner = { tone: "info", text: t("synth.result.empty") };
  } else if (result.dry_run) {
    banner = { tone: "info", text: t("synth.result.preview_only") };
  } else {
    banner = { tone: "ok",
                text: t("synth.result.persisted").replace("{n}", String(result.n_written)) };
  }
  const bannerCls =
    banner.tone === "ok"   ? "bg-ok/10 text-ok border-ok/30"
    : banner.tone === "warn"? "bg-warn/10 text-warn border-warn/30"
    : "bg-panel2/30 text-muted/80 border-border/30";

  return (
    <div className="space-y-2 pt-1">
      <div className={cn("text-[10.5px] px-2 py-1.5 rounded border", bannerCls)}>
        {banner.text}
      </div>

      {result.errors.length > 0 && (
        <div className="text-[10.5px] text-danger px-2 py-1.5 bg-danger/5 border border-danger/30 rounded">
          <span className="font-semibold">{t("synth.errors.label")}</span>{" "}
          {result.errors.join("; ")}
        </div>
      )}

      {result.candidates.map((c, i) => (
        <CandidateCard key={i} c={c} />
      ))}
    </div>
  );
}


function CandidateCard({ c }: { c: SynthCandidate }) {
  const { t } = useI18n();
  return (
    <div className="border border-border/40 rounded p-2.5 bg-panel2/20 space-y-1.5">
      <div className="text-[11px] font-medium text-foreground leading-snug">
        {c.claim}
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
        <span>
          <span className="text-muted/60">{t("synth.field.family")}: </span>
          <span className="font-mono">{c.mechanism_family} / {c.mechanism_subtype}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.direction")}: </span>
          <span className="font-mono">{c.predicted_direction}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.magnitude")}: </span>
          <span className="font-mono">{c.predicted_magnitude}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.cochrane")}: </span>
          <span className="font-mono">{c.cochrane_frame}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.prior")}: </span>
          <span className="font-mono">{c.expected_outcome_prior}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.novelty")}: </span>
          <span className="font-mono">{c.novelty_vs_known}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.n_trials")}: </span>
          <span className="font-mono">{c.estimated_n_trials_in_family}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("synth.field.provenance")}: </span>
          <span className="font-mono">
            {c.synthesizes_paper_ids.length}p + {c.synthesizes_event_ids.length}e
          </span>
        </span>
      </div>

      {c.required_data.length > 0 && (
        <div className="text-[10px]">
          <span className="text-muted/60">{t("synth.field.required")}: </span>
          <span>{c.required_data.join(", ")}</span>
        </div>
      )}
      {c.test_methodology && (
        <div className="text-[10px]">
          <span className="text-muted/60">{t("synth.field.methodology")}: </span>
          <span>{c.test_methodology}</span>
        </div>
      )}
      {c.addresses_decay_in && (
        <div className="text-[10px]">
          <span className="text-muted/60">{t("synth.field.addresses_decay")}: </span>
          <span className="font-mono text-accent">{c.addresses_decay_in}</span>
        </div>
      )}
      {(c.graveyard_conflicts.length > 0 || c.doctrine_conflicts.length > 0) && (
        <div className="text-[10px] space-y-0.5 pt-0.5 border-t border-border/30">
          {c.graveyard_conflicts.length > 0 && (
            <div>
              <span className="text-warn/80">{t("synth.field.graveyard")}: </span>
              <span className="font-mono">{c.graveyard_conflicts.join(", ")}</span>
            </div>
          )}
          {c.doctrine_conflicts.length > 0 && (
            <div>
              <span className="text-warn/80">{t("synth.field.doctrine")}: </span>
              <span className="font-mono">{c.doctrine_conflicts.join(", ")}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


type IncomingRow = {
  source:        string;
  source_id:     string;
  title:         string;
  authors:       string[];
  abstract:      string;
  abs_url:       string;
  pdf_url:       string;
  published_ts:  string;
  categories:    string[];
  fetched_ts:    string;
  judged:               boolean;
  is_tradable_factor:   boolean;
  filter_confidence:    number;
  filter_reason:        string;
  filter_category:      string;
  summarized:           boolean;
  thesis:               string;
  mechanism:            string;
  testable_hypothesis:  string;
  why_matters_for_us:   string;
  risk_flags:           string[];
  recommended_action:   string;
  user_skipped:         boolean;
};

type Digest = {
  days_requested: number;
  n_total:        number;
  n_today:        number;
  counts:         { INGEST: number; READ_AND_DISCARD: number; SKIP: number;
                     unrated: number; user_skipped: number };
  rows:           IncomingRow[];
};


// Style + i18n KEY (not the rendered string) per action — caller resolves
// via t(). The "(unrated)" path uses the unrated key directly.
// LLM-output → shelf classifier (2026-06-06). Primary signal is the
// filter's category_guess; recommended_action is a secondary refinement
// only for new_factor papers. Mirrors the doctrine: theory/survey
// belong on the methodology shelf, refinements support an existing
// GREEN, new factors stay tentative OTHER until the verdict pipeline
// produces evidence (when the action says INGEST we give it the
// hopeful green_motivation tag; verdict can demote later).
function _classifyShelf(filter_category: string, recommended_action: string): string {
  const cat = (filter_category || "").toLowerCase();
  if (cat === "theory" || cat === "survey")  return "doctrine_method";
  if (cat === "refinement")                  return "green_critique";
  if (cat === "new_factor") {
    return recommended_action === "INGEST" ? "green_motivation" : "other";
  }
  // commentary / off_topic / unknown / unrated / empty → conservative OTHER
  return "other";
}


function _actionStyle(action: string): { bg: string; text: string; labelKey: string; Icon: typeof PlayCircle } {
  if (action === "INGEST")           return { bg: "bg-ok/15",     text: "text-ok",     labelKey: "incoming.verdict.ingest",           Icon: PlayCircle };
  if (action === "READ_AND_DISCARD") return { bg: "bg-warn/15",   text: "text-warn",   labelKey: "incoming.verdict.read",             Icon: BookOpen };
  if (action === "SKIP")             return { bg: "bg-muted/15",  text: "text-muted",  labelKey: "incoming.verdict.skip",             Icon: MinusCircle };
  return                                  { bg: "bg-panel2/30", text: "text-muted/70", labelKey: "incoming.unrated",                  Icon: FileText };
}


export default function PapersIncomingPage() {
  const router = useRouter();
  const { t } = useI18n();
  const [digest, setDigest]   = useState<Digest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [refreshTok, setRefreshTok] = useState(0);
  const [skippingId, setSkippingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/api/papers_curator/incoming?days=14`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((d) => { if (!cancelled) setDigest(d); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refreshTok]);

  const skip = async (row: IncomingRow) => {
    setSkippingId(`${row.source}/${row.source_id}`);
    try {
      await fetch(`${API_BASE}/api/papers_curator/skip`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ source: row.source, source_id: row.source_id }),
      });
      setRefreshTok((t) => t + 1);
    } catch {
      // surface in UI on next refresh
    } finally {
      setSkippingId(null);
    }
  };

  const ingestNow = (row: IncomingRow) => {
    // Prefill /papers/new via query params; paste-text path picks up
    // the abstract as initial text so the user just confirms.
    // Phase 1.7 step 3 (2026-06-06): also pass agent_reason from the
    // summarizer's why_matters_for_us so the textarea pre-populates
    // with the agent's authorship (source=agent). User can override
    // via the "Write my own reason" button.
    const agentReason = row.why_matters_for_us
      ? row.why_matters_for_us.slice(0, 200)
      : `Agent recommended ${row.recommended_action} based on filter + summary.`;
    const params = new URLSearchParams({
      title:    row.title,
      authors:  row.authors.join("|"),
      year:     row.published_ts.slice(0, 4),
      doi:      "",
      abstract: row.abstract,
      pdf_url:  row.pdf_url,
      // 2026-06-06: shelf comes from the LLM-output classifier — primary
      // by filter.category_guess (theory/survey/refinement/new_factor),
      // refined by summarizer.recommended_action only for new_factor.
      shelf: _classifyShelf(row.filter_category, row.recommended_action),
      agent_reason: agentReason,
    });
    router.push(`/research/papers/new?${params.toString()}`);
  };

  const counts = digest?.counts;

  return (
    <div className="p-6 space-y-4 max-w-5xl">
      <ModeHeader
        mode="research"
        title={t("incoming.title")}
        subtitle={t("incoming.subtitle")}
      />

      {error && (
        <Card className="border border-danger/30 bg-danger/5 p-3">
          <div className="text-[12px] text-danger inline-flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" /> {error}
          </div>
        </Card>
      )}

      {/* Cross-source synthesis trigger panel (Phase 2.0 step 5c). Sits
          ABOVE the daily digest — synthesis is the higher-order
          activity (across-papers + sleeves + events), digest is the
          per-paper triage feed it consumes. */}
      <SynthesisPanel />

      {/* Rollup strip */}
      <Card className="p-0 overflow-hidden">
        <div className="px-3 py-2 border-b border-border/30 bg-panel2/30 flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
          <span className="text-[11.5px] font-semibold text-foreground">{t("incoming.last_14d")}</span>
          <button
            onClick={() => setRefreshTok((n) => n + 1)}
            aria-label={t("incoming.refresh.title")}
            title={t("incoming.refresh.title")}
            className="ml-auto text-muted hover:text-foreground">
            <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")} strokeWidth={2.2} />
          </button>
        </div>
        <div className="px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px]">
          {digest && (
            <>
              <span>
                <span className="text-muted/70">{t("incoming.rollup.total")}: </span>
                <span className="font-mono">{digest.n_total}</span>
                <span className="text-muted/50"> ({digest.n_today} {t("incoming.rollup.today")})</span>
              </span>
              <span>
                <span className="text-ok">{t("incoming.verdict.ingest")}: </span>
                <span className="font-mono">{counts?.INGEST ?? 0}</span>
              </span>
              <span>
                <span className="text-warn">{t("incoming.verdict.read")}: </span>
                <span className="font-mono">{counts?.READ_AND_DISCARD ?? 0}</span>
              </span>
              <span>
                <span className="text-muted">{t("incoming.verdict.skip")}: </span>
                <span className="font-mono">{counts?.SKIP ?? 0}</span>
              </span>
              <span>
                <span className="text-muted/70">{t("incoming.rollup.unrated")}: </span>
                <span className="font-mono">{counts?.unrated ?? 0}</span>
              </span>
              <span>
                <span className="text-muted/50">{t("incoming.rollup.dismissed")}: </span>
                <span className="font-mono">{counts?.user_skipped ?? 0}</span>
              </span>
            </>
          )}
        </div>
      </Card>

      {/* Rows */}
      {loading && !digest && (
        <Card className="px-3 py-3 text-[10.5px] text-muted/70 inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> {t("incoming.loading")}
        </Card>
      )}

      {digest && digest.rows.length === 0 && (
        <Card className="px-3 py-3 text-[10.5px] text-muted/70">
          {t("incoming.empty")}
        </Card>
      )}

      {digest && digest.rows.map((row) => {
        const { bg, text, labelKey, Icon } = _actionStyle(row.recommended_action);
        const skipKey = `${row.source}/${row.source_id}`;
        const isDismissing = skippingId === skipKey;
        const cats = row.categories.slice(0, 3).join(", ");
        return (
          <Card key={skipKey}
                className={cn(
                  "p-0 overflow-hidden",
                  row.user_skipped && "opacity-50",
                )}>
            {/* Header row: action badge + title + meta */}
            <div className="px-3 py-2 border-b border-border/30 bg-panel2/15 flex items-start gap-2">
              <div className={cn("inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider shrink-0", bg, text)}>
                <Icon className="h-3 w-3" strokeWidth={2.2} />
                {t(labelKey)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-semibold text-foreground leading-tight">
                  {row.title}
                </div>
                <div className="text-[10px] text-muted/70 mt-0.5 truncate">
                  {row.authors.slice(0, 3).join(", ")}
                  {row.authors.length > 3 && " et al"} ·{" "}
                  {row.published_ts.slice(0, 10)} ·{" "}
                  <span className="font-mono">{cats || "(no cat)"}</span>
                </div>
              </div>
              {row.user_skipped && (
                <span className="text-[9.5px] uppercase tracking-wider text-muted/60 shrink-0">
                  {t("incoming.dismissed")}
                </span>
              )}
            </div>

            {/* Summary body (if summarized) */}
            {row.summarized ? (
              <div className="px-3 py-2 space-y-1.5 text-[11px] leading-snug">
                <div>
                  <span className="text-[9.5px] uppercase tracking-wider text-muted/60">{t("incoming.field.thesis")} </span>
                  <span className="text-foreground/90">{row.thesis}</span>
                </div>
                <div>
                  <span className="text-[9.5px] uppercase tracking-wider text-muted/60">{t("incoming.field.mechanism")} </span>
                  <span className="text-foreground/80">{row.mechanism}</span>
                </div>
                <div>
                  <span className="text-[9.5px] uppercase tracking-wider text-accent">{t("incoming.field.testable")} </span>
                  <span className="font-mono text-foreground/90">{row.testable_hypothesis}</span>
                </div>
                <div>
                  <span className="text-[9.5px] uppercase tracking-wider text-muted/60">{t("incoming.field.why_us")} </span>
                  <span className="text-foreground/80">{row.why_matters_for_us}</span>
                </div>
                {row.risk_flags.length > 0 && (
                  <div>
                    <span className="text-[9.5px] uppercase tracking-wider text-warn">{t("incoming.field.risks")} </span>
                    {row.risk_flags.map((r, i) => (
                      <span key={i} className="inline-block bg-warn/10 text-warn/90 text-[10px] px-1.5 py-0.5 rounded mr-1 mb-1">
                        {r}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ) : row.judged ? (
              <div className="px-3 py-2 text-[10.5px] text-muted/80">
                <span className="text-[9.5px] uppercase tracking-wider text-muted/60">{t("incoming.filter_only")} </span>
                {row.filter_reason}{" "}
                <span className="text-muted/50">
                  ({row.filter_category}, conf {row.filter_confidence.toFixed(2)})
                </span>
              </div>
            ) : (
              <div className="px-3 py-2 text-[10.5px] text-muted/60 italic">
                {t("incoming.not_judged")}
              </div>
            )}

            {/* Abstract preview (collapsible would be cleaner; for v1 truncate) */}
            <details className="px-3 py-1.5 text-[10.5px] text-muted/70 border-t border-border/20">
              <summary className="cursor-pointer hover:text-foreground/80">{t("incoming.abstract")}</summary>
              <p className="mt-1 leading-relaxed">{row.abstract}</p>
            </details>

            {/* Actions */}
            <div className="px-3 py-2 border-t border-border/30 bg-panel2/10 flex items-center gap-2">
              <button
                onClick={() => ingestNow(row)}
                disabled={row.user_skipped}
                className="rounded bg-accent text-background hover:bg-accent/90 disabled:opacity-40 px-2.5 py-1 text-[11px] font-semibold inline-flex items-center gap-1">
                <PlayCircle className="h-3.5 w-3.5" strokeWidth={2.2} /> {t("incoming.btn.ingest_now")}
              </button>
              <a
                href={row.abs_url}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded border border-border/40 text-muted hover:text-foreground hover:border-border/70 px-2.5 py-1 text-[11px] inline-flex items-center gap-1">
                <ExternalLink className="h-3.5 w-3.5" strokeWidth={2} /> {t("incoming.btn.open_arxiv")}
              </a>
              {!row.user_skipped ? (
                <button
                  onClick={() => skip(row)}
                  disabled={isDismissing}
                  className="ml-auto rounded text-muted hover:text-foreground px-2.5 py-1 text-[11px] inline-flex items-center gap-1 disabled:opacity-40">
                  {isDismissing ? <Loader2 className="h-3 w-3 animate-spin" /> : <X className="h-3 w-3" />}
                  {t("incoming.btn.dismiss")}
                </button>
              ) : (
                <span className="ml-auto text-[10px] text-muted/60">{t("incoming.btn.in_dismissed")}</span>
              )}
            </div>
          </Card>
        );
      })}
    </div>
  );
}
