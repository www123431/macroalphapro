"use client";

// /research/library/detail?id=... — Mechanism detail page.
//
// 2026-06-04 R2 REBUILD. Restructured from a stack of full-width
// cards to a master-detail-style tabbed layout. The same data,
// reorganized for faster scan + collaboration-aware CTAs that hand
// off to Claude (open audit / open research_new / open candidate
// pipeline) without the user typing URLs.
//
// Tabs:
//   Overview  — KPI strip + identity + family + canonical paper
//   Decay     — full decay-sentinel history with trailing Sharpe
//   Council   — past Council critiques on this family/proposal
//   Audit     — research event store timeline (verdicts, evidence,
//                spec amendments, decay alerts)
//   Lineage   — graveyard cousins (same family) + paper hypotheses
//   YAML      — raw payload (advanced)
//
// Each CTA is wired to the doctrine session protocol so the user
// can hand off to Claude in one click.

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  AlertCircle, TrendingDown, Skull, FileText, History,
  ExternalLink, Network, Atom, Bug, ArrowRight, BookOpen,
  Brain,
} from "lucide-react";
import { api, API_BASE } from "@/lib/api";
import { safeArtifactHref } from "@/lib/artifactLink";
import { fileIntent } from "@/lib/intents";
import { useResearchStoreEvents, useDQ } from "@/lib/queries";
import { Card, Badge, Skeleton, cn } from "@/components/ui";
import { Breadcrumb } from "@/components/Breadcrumb";
import { MetricWithContext } from "@/components/MetricWithContext";
import { PrintButton } from "@/components/PrintButton";
import { fadeUp, stagger } from "@/lib/motion";
import { CrossDomainTransferSection } from "@/components/CrossDomainTransferSection";

type Detail = Awaited<ReturnType<typeof api.libraryMechanismDetail>>;

type CouncilRun = {
  run_id:     string;
  ts:         string;
  consensus:  string;
  proposal:   { title?: string; family?: string };
  rationale?: string;
};

const PURPOSE_TONE: Record<string, string> = {
  deployed_sleeve:    "bg-ok/15 text-ok",
  deploy_replacement: "bg-ok/15 text-ok",
  hedge_replacement:  "bg-ok/15 text-ok",
  cousin_anchor:      "bg-info/15 text-info",
  candidate:          "bg-warn/15 text-warn",
};

const ALERT_TONE: Record<string, string> = {
  OK:    "bg-ok/15 text-ok",
  WARN:  "bg-warn/15 text-warn",
  SOFT:  "bg-warn/15 text-warn",
  HARD:  "bg-danger/15 text-danger",
  ALERT: "bg-danger/15 text-danger",
};

const CONSENSUS_TONE: Record<string, string> = {
  APPROVE:        "bg-ok/15 text-ok",
  NEEDS_REVISION: "bg-warn/15 text-warn",
  REJECT:         "bg-danger/15 text-danger",
};


// Council runs filtered by the candidate's family.
function useCouncilForFamily(family: string | undefined) {
  const [runs, setRuns] = useState<CouncilRun[]>([]);
  useEffect(() => {
    if (!family) { setRuns([]); return; }
    fetch(`${API_BASE}/api/research/council/runs?limit=200`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((d: { runs: CouncilRun[] }) => {
        const matched = (d.runs || []).filter((r) =>
          (r.proposal?.family || "").toLowerCase() === family.toLowerCase()
        );
        setRuns(matched);
      })
      .catch(() => setRuns([]));
  }, [family]);
  return runs;
}


type TabKey = "overview" | "decay" | "council" | "audit" | "lineage" | "doctrine" | "yaml";


type DoctrineHit = {
  file:       string;
  match_term: string;
  snippet:    string;
  line_no:    number;
};

type DoctrineResp = {
  mechanism_id: string;
  family:       string | null;
  search_terms?: string[];
  n_hits:       number;
  hits:         DoctrineHit[];
  warning?:     string;
};


function useDoctrineMemory(mechanism_id: string) {
  const [data, setData] = useState<DoctrineResp | null>(null);
  useEffect(() => {
    if (!mechanism_id) return;
    fetch(`${API_BASE}/api/research/library/${encodeURIComponent(mechanism_id)}/doctrine`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch(() => setData(null));
  }, [mechanism_id]);
  return data;
}


export default function MechanismDetailPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <MechanismDetailInner />
    </Suspense>
  );
}

function MechanismDetailInner() {
  const searchParams = useSearchParams();
  const mechanism_id = searchParams.get("id") || "";
  const tabParam = (searchParams.get("tab") as TabKey | null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<TabKey>(tabParam || "overview");
  // P1-D — when true, stack ALL tab bodies for a one-PDF archival
  // export. Restored to false after the print dialog closes.
  const [printAll, setPrintAll] = useState(false);

  // Wire `afterprint` so the user gets their tab back automatically.
  useEffect(() => {
    if (!printAll) return;
    const restore = () => setPrintAll(false);
    window.addEventListener("afterprint", restore);
    return () => window.removeEventListener("afterprint", restore);
  }, [printAll]);

  const triggerPrintAll = () => {
    setPrintAll(true);
    // Give React a tick to render the stacked layout before opening
    // the print dialog. setTimeout 0 isn't enough on all browsers.
    setTimeout(() => window.print(), 80);
  };

  useEffect(() => {
    if (!mechanism_id) return;
    api.libraryMechanismDetail(mechanism_id)
      .then(setDetail)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [mechanism_id]);

  const eventsQ = useResearchStoreEvents({
    subject_id: mechanism_id, limit: 50,
  });
  const storeEvents = eventsQ.data?.events ?? [];

  const councilRuns = useCouncilForFamily(detail?.yaml?.family);
  const doctrine = useDoctrineMemory(mechanism_id);

  const sortedDecay = useMemo(() => {
    if (!detail) return [];
    return [...detail.decay_history].sort(
      (a, b) => (a.audit_date || "").localeCompare(b.audit_date || "")
    );
  }, [detail]);

  if (!mechanism_id) {
    return <div className="p-6 text-sm text-muted">
      No mechanism id. <a href="/research/library" className="text-accent hover:underline">Back to library</a>.
    </div>;
  }

  if (error) {
    return (
      <div className="p-6">
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </div>
        </Card>
      </div>
    );
  }

  if (!detail) {
    return <div className="p-6"><Skeleton className="h-40 w-full" /></div>;
  }

  // ── KPI strip values ─────────────────────────────────────────
  const latest = sortedDecay[sortedDecay.length - 1];
  const baseline = sortedDecay[0];
  const decayRatio = (latest?.trailing_sharpe != null && baseline?.trailing_sharpe != null && baseline.trailing_sharpe !== 0)
    ? latest.trailing_sharpe / baseline.trailing_sharpe
    : null;

  const tabCount = (k: TabKey): string | undefined => {
    if (k === "decay")    return detail.decay_history.length ? String(detail.decay_history.length) : undefined;
    if (k === "council")  return councilRuns.length ? String(councilRuns.length) : undefined;
    if (k === "audit")    return storeEvents.length ? String(storeEvents.length) : undefined;
    if (k === "lineage")  return detail.graveyard_cousins.length ? String(detail.graveyard_cousins.length) : undefined;
    if (k === "doctrine") return doctrine?.n_hits ? String(doctrine.n_hits) : undefined;
    return undefined;
  };

  return (
    <motion.div variants={stagger(0.04)} initial="hidden" animate="show"
                className="space-y-3 p-6">
      <motion.div variants={fadeUp}>
        <Breadcrumb crumbs={[
          { label: "Lab",     href: "/dashboard" },
          { label: "Library", href: "/research/library" },
          { label: mechanism_id, mono: true },
        ]} />
      </motion.div>

      {/* Compact header — purpose + family + ID + paper. Dense, one line. */}
      <motion.div variants={fadeUp}>
        <Card className="p-3">
          <div className="flex items-start justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-3 flex-wrap">
              <h1 className="text-base font-semibold font-mono">{detail.mechanism_id}</h1>
              {detail.yaml.purpose && (
                <Badge tone={PURPOSE_TONE[detail.yaml.purpose] || "bg-muted/15 text-muted"}>
                  {detail.yaml.purpose}
                </Badge>
              )}
              {detail.yaml.family && (
                <span className="text-[11px] font-mono px-1.5 py-0.5 rounded bg-panel2/40 text-muted">
                  {detail.yaml.family}
                </span>
              )}
              {detail.yaml.parent_family && (
                <span className="text-[10px] text-muted/70">
                  parent: <span className="font-mono">{detail.yaml.parent_family}</span>
                </span>
              )}
              {detail.yaml.canonical_paper_id && (
                <span className="text-[10px] text-muted/70">
                  paper: <span className="font-mono text-accent">{detail.yaml.canonical_paper_id}</span>
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <CollaborationActions detail={detail} mechanism_id={mechanism_id} />
              <PrintButton title="Export only the currently-visible tab as PDF" />
              <button onClick={triggerPrintAll}
                data-no-print="true"
                title="Stack every tab body and export them as a single PDF (archival)"
                className="no-print inline-flex items-center gap-1 rounded border border-border/60 text-muted hover:text-foreground hover:border-border px-2 py-0.5 text-[11px]">
                Export all tabs
              </button>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* KPI strip — what's hot about THIS sleeve */}
      <motion.div variants={fadeUp}>
        <Card className="p-0 overflow-hidden">
          <div className="grid grid-cols-2 md:grid-cols-5 divide-x divide-border/30">
            <KPI label="Latest trailing Sharpe"
                 value={latest?.trailing_sharpe != null ? latest.trailing_sharpe.toFixed(3) : "—"}
                 tone={
                   latest?.trailing_sharpe == null ? "muted" :
                   latest.trailing_sharpe >= 1 ? "ok" :
                   latest.trailing_sharpe >= 0.5 ? "warn" : "danger"
                 }
                 sub={latest?.audit_date ? `as of ${latest.audit_date}` : undefined} />
            <KPI label="Decay vs baseline"
                 value={decayRatio != null ? `${Math.round(decayRatio * 100)}%` : "—"}
                 tone={
                   decayRatio == null ? "muted" :
                   decayRatio >= 0.8 ? "ok" :
                   decayRatio >= 0.5 ? "warn" : "danger"
                 } />
            <KPI label="Decay audits"
                 value={detail.decay_history.length}
                 sub={detail.decay_history.length === 0 ? "no audits yet" : undefined} />
            <KPI label="Council critiques"
                 value={councilRuns.length}
                 tone={
                   councilRuns.length === 0 ? "muted" :
                   councilRuns.some((r) => r.consensus === "REJECT") ? "danger" :
                   councilRuns.some((r) => r.consensus === "NEEDS_REVISION") ? "warn" :
                   "ok"
                 }
                 sub={councilRuns.length ? `family ${detail.yaml.family}` : undefined} />
            <KPI label="Graveyard cousins"
                 value={detail.graveyard_cousins.length}
                 tone={detail.graveyard_cousins.length > 0 ? "warn" : "muted"}
                 sub={detail.graveyard_cousins.length ? "same family — read before re-testing" : undefined} />
          </div>
        </Card>
      </motion.div>

      {/* Tab strip */}
      <motion.div variants={fadeUp} className="flex items-center gap-1 border-b border-border/40">
        <Tab k="overview" current={tab} onClick={setTab} label="Overview" />
        <Tab k="decay"    current={tab} onClick={setTab} label="Decay"    count={tabCount("decay")} />
        <Tab k="council"  current={tab} onClick={setTab} label="Council"  count={tabCount("council")} />
        <Tab k="audit"    current={tab} onClick={setTab} label="Audit trail" count={tabCount("audit")} />
        <Tab k="lineage"  current={tab} onClick={setTab} label="Lineage" count={tabCount("lineage")} />
        <Tab k="doctrine" current={tab} onClick={setTab} label="Doctrine" count={tabCount("doctrine")} />
        <Tab k="yaml"     current={tab} onClick={setTab} label="YAML" />
      </motion.div>

      {/* Tab panels. printAll mode stacks every tab body with
          page-break dividers so the user can export the whole sleeve
          dossier as one PDF (P1-D 2026-06-04). */}
      {printAll ? (
        <div className="space-y-4">
          <section><PrintSectionHeader title="Overview" /><OverviewTab detail={detail} /></section>
          <section className="page-break"><PrintSectionHeader title="Decay" /><DecayTab detail={detail} /></section>
          <section className="page-break"><PrintSectionHeader title="Council" /><CouncilTab runs={councilRuns} family={detail.yaml.family} /></section>
          <section className="page-break"><PrintSectionHeader title="Audit trail" /><AuditTab events={storeEvents} /></section>
          <section className="page-break"><PrintSectionHeader title="Lineage" /><LineageTab detail={detail} /></section>
          <section className="page-break"><PrintSectionHeader title="Doctrine memory" /><DoctrineTab data={doctrine} /></section>
          <section className="page-break"><PrintSectionHeader title="YAML" /><YamlTab detail={detail} /></section>
        </div>
      ) : (
        <>
          {tab === "overview" && <OverviewTab detail={detail} />}
          {tab === "decay"    && <DecayTab detail={detail} />}
          {tab === "council"  && <CouncilTab runs={councilRuns} family={detail.yaml.family} />}
          {tab === "audit"    && <AuditTab events={storeEvents} />}
          {tab === "lineage"  && <LineageTab detail={detail} />}
          {tab === "doctrine" && <DoctrineTab data={doctrine} />}
          {tab === "yaml"     && <YamlTab detail={detail} />}
        </>
      )}
    </motion.div>
  );
}


function PrintSectionHeader({ title }: { title: string }) {
  return (
    <div className="mb-2 mt-3 pb-1 border-b border-border/40">
      <h2 className="text-sm font-semibold uppercase tracking-[0.15em] text-muted">{title}</h2>
    </div>
  );
}


// ── Doctrine memory tab ────────────────────────────────────────────


function DoctrineTab({ data }: { data: DoctrineResp | null }) {
  if (!data) {
    return (
      <motion.div variants={fadeUp}>
        <Card className="p-4 text-[12px] text-muted/70">
          Loading doctrine memory…
        </Card>
      </motion.div>
    );
  }
  if (data.warning) {
    return (
      <motion.div variants={fadeUp}>
        <Card className="p-4 text-[12px] text-warn">
          {data.warning}
        </Card>
      </motion.div>
    );
  }
  if (data.n_hits === 0) {
    return (
      <motion.div variants={fadeUp}>
        <Card className="p-4 text-[12px] text-muted/70">
          No doctrine memory mentions <code className="text-foreground">{data.family || "this mechanism"}</code> yet.
          Lock a doctrine via Claude using <code>engine.research_store.emit.memory_doctrine_locked</code>{" "}
          or by writing to <code>~/.claude/projects/.../memory/</code> directly.
        </Card>
      </motion.div>
    );
  }
  return (
    <motion.div variants={fadeUp}>
      <Card className="p-0 overflow-hidden">
        <div className="px-3 py-2 border-b border-border/40 flex items-baseline gap-2">
          <Brain className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
          <span className="text-[12px] font-semibold">
            Doctrine memory · matches for{" "}
            <code className="text-accent">{data.family || data.mechanism_id}</code>
          </span>
          <span className="text-[10px] text-muted/60 ml-auto">
            ~/.claude/.../memory · {data.n_hits} snippet{data.n_hits === 1 ? "" : "s"}
          </span>
        </div>
        <div className="divide-y divide-border/30">
          {data.hits.map((h, i) => (
            <div key={`${h.file}-${i}`} className="px-3 py-2.5 hover:bg-panel2/30 transition-colors">
              <div className="flex items-center gap-2 text-[10.5px] mb-1.5">
                <FileText className="h-3 w-3 text-muted/60 shrink-0" />
                <code className="text-foreground/85 truncate">{h.file}</code>
                <span className="text-muted/60">· line {h.line_no}</span>
                <span className="ml-auto text-[9px] uppercase tracking-wider text-accent/70 font-mono px-1 rounded bg-accent/10">
                  {h.match_term}
                </span>
              </div>
              <p className="text-[11px] text-foreground/85 leading-relaxed whitespace-pre-line font-mono">
                {h.snippet}
              </p>
            </div>
          ))}
        </div>
      </Card>
    </motion.div>
  );
}


// ── Collaboration actions (Claude-handoff CTAs) ────────────────────


function CollaborationActions({ detail, mechanism_id }: { detail: Detail; mechanism_id: string }) {
  const family = detail.yaml.family || "unknown";
  // P1-C — when DQ is HALT the pipeline can't actually run, so the
  // execution-class CTAs should reflect that. Backend would refuse
  // anyway (409 dq_halt); making the button visibly disabled lets
  // the user avoid the click-and-bounce.
  const dqQ = useDQ();
  const dqHalt = (dqQ.data?.verdict || "") === "HALT";
  // R3.1 — each CTA files a typed intent BEFORE navigating so Claude
  // can pick the intent up via /api/intents/pending. Fire-and-forget;
  // the navigation isn't blocked on the file succeeding.
  const filePipelineIntent = () => {
    fileIntent({
      kind:         "pipeline_test",
      subject_type: "mechanism",
      subject_id:   mechanism_id,
      source_page:  "/research/library/detail",
      payload:      { family, proposal_name: `retest_${mechanism_id}` },
    })
      .then((r) => { if (!r.ok) console.warn("[library_detail] pipeline_test intent refused:", r); })
      .catch((e) => console.warn("[library_detail] pipeline_test intent errored:", e));
  };
  const fileAuditIntent = () => {
    fileIntent({
      kind:         "audit_subject",
      subject_type: "mechanism",
      subject_id:   mechanism_id,
      source_page:  "/research/library/detail",
      payload:      {
        family,
        purpose:           detail.yaml.purpose,
        canonical_paper_id: detail.yaml.canonical_paper_id,
      },
    })
      .then((r) => { if (!r.ok) console.warn("[library_detail] audit_subject intent refused:", r); })
      .catch((e) => console.warn("[library_detail] audit_subject intent errored:", e));
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {/* Run the candidate pipeline directly */}
      {dqHalt ? (
        <span
          className="inline-flex items-center gap-1 rounded border border-danger/30 bg-danger/[0.04] text-danger/70 px-2 py-0.5 text-[11px] cursor-not-allowed"
          title="DQ is HALT — pipeline_test intents are refused server-side. Resolve breaches in /lab/cockpit first.">
          <Atom className="h-3 w-3" /> Pipeline test (DQ HALT)
        </span>
      ) : (
        <Link
          href={`/research/candidate?proposal_name=${encodeURIComponent("retest_" + mechanism_id)}&family=${family}`}
          onClick={filePipelineIntent}
          className="inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 px-2 py-0.5 text-[11px]"
          title="Re-run the candidate pipeline against this mechanism (files a pipeline_test intent for Claude)">
          <Atom className="h-3 w-3" /> Pipeline test
          <ArrowRight className="h-2.5 w-2.5" />
        </Link>
      )}

      {/* Open an AUDIT session for this sleeve — typed Claude handoff */}
      <Link
        href={`/research/sessions?type=audit&prefill_subject=${encodeURIComponent(mechanism_id)}`}
        onClick={fileAuditIntent}
        className="inline-flex items-center gap-1 rounded border border-warn/40 bg-warn/10 text-warn hover:bg-warn/20 px-2 py-0.5 text-[11px]"
        title="Open a typed audit session — files an audit_subject intent for Claude">
        <Bug className="h-3 w-3" /> Audit session
        <ArrowRight className="h-2.5 w-2.5" />
      </Link>

      {/* Open the paper if attached */}
      {detail.yaml.canonical_paper_id && (
        <Link
          href={`/research/papers/${detail.yaml.canonical_paper_id}`}
          className="inline-flex items-center gap-1 rounded border border-border/60 text-muted hover:text-foreground hover:border-border px-2 py-0.5 text-[11px]"
          title="Open the canonical paper">
          <BookOpen className="h-3 w-3" /> Paper
        </Link>
      )}
    </div>
  );
}


// ── KPI cell + Tab button shared bits ──────────────────────────────


function KPI({
  label, value, tone, sub,
}: {
  label: string;
  value: React.ReactNode;
  tone?: "ok" | "warn" | "danger" | "muted" | "neutral";
  sub?:  React.ReactNode;
}) {
  const toneClass =
    tone === "ok"     ? "text-ok" :
    tone === "warn"   ? "text-warn" :
    tone === "danger" ? "text-danger" :
    tone === "muted"  ? "text-muted" :
                        "text-foreground";
  return (
    <div className="px-3 py-2 min-w-0">
      <div className="text-[9px] uppercase tracking-[0.15em] text-muted/60 leading-none">
        {label}
      </div>
      <div className={cn("tnum text-base font-semibold leading-tight mt-1", toneClass)}>
        {value}
      </div>
      {sub && (
        <div className="text-[10px] text-muted/60 leading-snug mt-0.5 truncate">
          {sub}
        </div>
      )}
    </div>
  );
}


function Tab({
  k, current, onClick, label, count,
}: {
  k:       TabKey;
  current: TabKey;
  onClick: (k: TabKey) => void;
  label:   string;
  count?:  string;
}) {
  const active = current === k;
  return (
    <button onClick={() => onClick(k)}
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-t border-b-2 transition-colors",
        active
          ? "text-accent border-accent bg-accent/[0.06]"
          : "text-muted border-transparent hover:text-foreground hover:bg-panel2/30",
      )}>
      <span>{label}</span>
      {count !== undefined && (
        <span className={cn("tnum text-[10px] px-1 rounded",
          active ? "bg-accent/15 text-accent" : "bg-muted/10 text-muted")}>
          {count}
        </span>
      )}
    </button>
  );
}


// ── Tab content components ─────────────────────────────────────────


function OverviewTab({ detail }: { detail: Detail }) {
  const sleeveId = (detail as any).yaml?.id || (detail as any).mechanism_id || "";
  return (
    <motion.div variants={fadeUp} className="space-y-3">
      <Card className="p-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[12px]">
          <Field label="Family"            value={detail.yaml.family} mono />
          <Field label="Parent family"     value={detail.yaml.parent_family} mono />
          <Field label="Cost-aware filter" value={detail.yaml.ca_filter_k_method} />
          <Field label="Canonical paper"   value={detail.yaml.canonical_paper_id || "—"} mono />
          <Field label="Schema version"    value={detail.yaml._schema_version} mono />
          <Field label="Filename"          value={detail.filename} mono />
        </div>
      </Card>

      {/* β Cross-Domain Transfer (Phase 10, 2026-06-14). Sonnet
          cross-asset thinker proposes 1-2 testable mechanism transfers
          to other asset classes (Frazzini-Pedersen 70% rule). */}
      {sleeveId && <CrossDomainTransferSection sleeveId={sleeveId} />}

      {/* Quick decay summary if available */}
      {detail.decay_history.length > 0 && (
        <Card className="p-3">
          <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-2">
            Mechanism health (decay sentinel)
          </div>
          {(() => {
            const sortedAsc = [...detail.decay_history].sort(
              (a, b) => (a.audit_date || "").localeCompare(b.audit_date || "")
            );
            const latest = sortedAsc[sortedAsc.length - 1];
            const baseline = sortedAsc[0];
            const ratio = (latest?.trailing_sharpe != null && baseline?.trailing_sharpe != null && baseline.trailing_sharpe !== 0)
              ? latest.trailing_sharpe / baseline.trailing_sharpe
              : null;
            return (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <MetricWithContext
                  label="Latest trailing Sharpe"
                  value={latest?.trailing_sharpe ?? null}
                  kind="ann_sharpe"
                  size="lg"
                />
                <MetricWithContext
                  label="Decay ratio vs first audit"
                  value={ratio}
                  kind="decay_ratio"
                  format={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <MetricWithContext
                  label="Cousins in graveyard"
                  value={detail.graveyard_cousins.length}
                  kind="cousin_warnings"
                  format={(v) => v.toFixed(0)}
                />
              </div>
            );
          })()}
        </Card>
      )}
    </motion.div>
  );
}


function DecayTab({ detail }: { detail: Detail }) {
  return (
    <motion.div variants={fadeUp}>
      <Card className="p-0 overflow-hidden">
        <div className="flex items-baseline justify-between px-3 py-2 border-b border-border/40">
          <span className="text-[12px] font-semibold inline-flex items-center gap-1.5">
            <TrendingDown className="h-3.5 w-3.5" strokeWidth={1.75} />
            Decay audit history
          </span>
          {detail.decay_history[0] && (
            <Link href={`/research/decay/detail?sleeve=${encodeURIComponent(detail.decay_history[0].sleeve)}`}
                  className="text-[10px] text-accent hover:underline inline-flex items-center gap-1">
              full timeline <ExternalLink className="h-3 w-3" />
            </Link>
          )}
        </div>
        {detail.decay_history.length === 0 ? (
          <p className="p-4 text-[12px] text-muted/70">No decay audits yet for this mechanism.</p>
        ) : (
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="min-w-full text-[11px]">
              <thead className="sticky top-0 bg-panel">
                <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                  <th className="px-3 py-1.5">date</th>
                  <th className="px-3 py-1.5">sleeve</th>
                  <th className="px-3 py-1.5 text-right">trailing Sharpe</th>
                  <th className="px-3 py-1.5">alert</th>
                </tr>
              </thead>
              <tbody>
                {detail.decay_history.slice(0, 50).map((r, i) => (
                  <tr key={i} className="border-b border-muted/10 last:border-0 hover:bg-panel2/30">
                    <td className="px-3 py-1.5 text-muted text-[10.5px]">{r.audit_date}</td>
                    <td className="px-3 py-1.5 font-mono">{r.sleeve}</td>
                    <td className="px-3 py-1.5 text-right tnum">
                      {r.trailing_sharpe != null ? r.trailing_sharpe.toFixed(3) : "—"}
                    </td>
                    <td className="px-3 py-1.5">
                      <Badge tone={ALERT_TONE[(r.alert_level || "").toUpperCase()] || "bg-muted/15 text-muted"}>
                        {r.alert_level || "—"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </motion.div>
  );
}


function CouncilTab({ runs, family }: { runs: CouncilRun[]; family?: string }) {
  return (
    <motion.div variants={fadeUp}>
      <Card className="p-0 overflow-hidden">
        <div className="flex items-baseline justify-between px-3 py-2 border-b border-border/40">
          <span className="text-[12px] font-semibold inline-flex items-center gap-1.5">
            <Network className="h-3.5 w-3.5" strokeWidth={1.75} />
            Council critiques · family {family || "?"}
          </span>
          <Link href="/lab/council" className="text-[10px] text-accent hover:underline inline-flex items-center gap-1">
            full council activity <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
        {runs.length === 0 ? (
          <p className="p-4 text-[12px] text-muted/70">
            No Council critiques yet for this family. Start one from{" "}
            <Link href="/lab/council" className="text-accent hover:underline">/lab/council</Link>.
          </p>
        ) : (
          <div className="divide-y divide-border/30">
            {runs.slice(0, 30).map((r) => (
              <Link key={r.run_id}
                href={`/lab/council/detail?run_id=${r.run_id}`}
                className="block p-3 hover:bg-panel2/30 transition-colors">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge tone={CONSENSUS_TONE[r.consensus] || "bg-muted/15 text-muted"}>
                    {r.consensus}
                  </Badge>
                  <span className="text-[11px] font-mono text-muted/70">
                    {r.run_id.slice(0, 12)}
                  </span>
                  <span className="text-[11px] text-foreground/85">
                    {r.proposal?.title || "(no title)"}
                  </span>
                  <span className="ml-auto text-[10px] text-muted/60">
                    {r.ts?.slice(0, 16).replace("T", " ")}
                  </span>
                </div>
                {r.rationale && (
                  <p className="text-[10.5px] text-muted/80 line-clamp-2 mt-1">
                    {r.rationale}
                  </p>
                )}
              </Link>
            ))}
          </div>
        )}
      </Card>
    </motion.div>
  );
}


function AuditTab({ events }: { events: any[] }) {
  if (events.length === 0) {
    return (
      <motion.div variants={fadeUp}>
        <Card className="p-4 text-[12px] text-muted/70">
          No events in research store for this subject_id yet.
        </Card>
      </motion.div>
    );
  }
  return (
    <motion.div variants={fadeUp}>
      <Card className="p-0 overflow-hidden">
        <div className="px-3 py-2 border-b border-border/40 flex items-baseline gap-2">
          <History className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
          <span className="text-[12px] font-semibold">Audit trail</span>
          <span className="text-[10px] text-muted/60 uppercase tracking-wider ml-1">
            research event store
          </span>
        </div>
        <div className="divide-y divide-border/30 max-h-[600px] overflow-y-auto">
          {events.map((ev) => {
            const verdictTone =
              ev.verdict === "RED"      ? "bg-danger/15 text-danger" :
              ev.verdict === "GREEN"    ? "bg-ok/15 text-ok"         :
              ev.verdict === "MARGINAL" ? "bg-warn/15 text-warn"     :
                                           "bg-muted/15 text-muted";
            const evidenceHref = safeArtifactHref(ev.artifacts?.evidence_doc);
            return (
              <div key={ev.event_id}
                   className="flex items-start gap-2 px-3 py-2 hover:bg-panel2/30 transition-colors">
                <span className="shrink-0 text-[10px] text-muted/60 tnum mt-0.5">
                  {ev.ts.slice(0, 10)}
                </span>
                <Badge tone={verdictTone} className="shrink-0">
                  {ev.verdict}
                </Badge>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-mono text-foreground/85 truncate">
                    {ev.event_type.replace(/_/g, " ")}
                  </div>
                  <div className="text-[10.5px] text-muted leading-snug line-clamp-2 mt-0.5">
                    {ev.summary}
                  </div>
                  {(ev.tags?.length ?? 0) > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {ev.tags.filter((t: string) => t !== "backfill").slice(0, 4).map((t: string) => (
                        <span key={t}
                          className="rounded bg-panel2/50 px-1 py-0 text-[9px] uppercase tracking-wider text-muted/70 font-mono">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                {evidenceHref && (
                  <a href={evidenceHref} target="_blank" rel="noopener noreferrer"
                    className="shrink-0 text-[10px] text-accent hover:underline inline-flex items-center gap-0.5 mt-0.5"
                    title="open evidence doc">
                    doc <ExternalLink className="h-2.5 w-2.5" />
                  </a>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    </motion.div>
  );
}


function LineageTab({ detail }: { detail: Detail }) {
  return (
    <motion.div variants={fadeUp} className="space-y-3">
      <Card className="p-0 overflow-hidden">
        <div className="px-3 py-2 border-b border-border/40 flex items-baseline gap-2">
          <Skull className="h-3.5 w-3.5 text-warn" strokeWidth={1.75} />
          <span className="text-[12px] font-semibold">Graveyard cousins · same family</span>
        </div>
        {detail.graveyard_cousins.length === 0 ? (
          <p className="p-4 text-[12px] text-muted/70">
            No RED verdicts on this family. Clean lineage.
          </p>
        ) : (
          <div className="divide-y divide-border/30">
            {detail.graveyard_cousins.map((c, i) => (
              <div key={i} className="p-3">
                <div className="flex items-baseline justify-between gap-2 mb-1">
                  <span className="font-mono text-[12px]">{c.name}</span>
                  <div className="flex items-center gap-2">
                    <Badge tone="bg-danger/15 text-danger">{c.verdict}</Badge>
                    <span className="text-[10px] text-muted">{c.date}</span>
                  </div>
                </div>
                <div className="text-[11px] text-muted leading-relaxed">{c.why}</div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </motion.div>
  );
}


function YamlTab({ detail }: { detail: Detail }) {
  return (
    <motion.div variants={fadeUp}>
      <Card className="p-3">
        <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-2 inline-flex items-center gap-1.5">
          <FileText className="h-3 w-3" /> Full YAML (audit-only)
        </div>
        <pre className="text-[11px] font-mono bg-bg/50 p-3 rounded border border-border/30 overflow-x-auto whitespace-pre-wrap leading-relaxed">
          {JSON.stringify(detail.yaml, null, 2)}
        </pre>
      </Card>
    </motion.div>
  );
}


function Field({ label, value, mono = false }: {
  label: string; value?: string | null; mono?: boolean;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("text-[12px] mt-0.5 text-foreground", mono && "font-mono")}>
        {value || "—"}
      </div>
    </div>
  );
}
