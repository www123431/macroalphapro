"use client";

// /research/library — Mechanism library YAML browser.
//
// 2026-06-04 R2.3 REBUILD. The previous version stacked 4
// accordion sections vertically (Lifecycle Gantt / SLM State
// Machine / Correlation Network / Inventory table) which forced
// the user to remember which sections were open. Now tabs:
//
//   Inventory    — read-only YAML browser (the page's bread-and-butter)
//   Lifecycle    — Gantt + state machine inside a single tab with toggle
//   Network      — correlation force graph
//
// Tab key syncs to ?tab=… so deep-links survive reload and back-
// button works.

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  Layers, FileText, Search, AlertCircle, ChevronRight,
  Activity, Network as NetworkIcon, FolderTree,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, Badge, Skeleton } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { Tabs, SegmentToggle } from "@/components/Tabs";
import { LifecycleGantt } from "@/components/LifecycleGantt";
import { SlmStateMachine } from "@/components/SlmStateMachine";
import { SleeveCorrelationNetwork } from "@/components/SleeveCorrelationNetwork";
import { fadeUp, stagger } from "@/lib/motion";

type LibEntry = Awaited<ReturnType<typeof api.libraryInventory>>["entries"][number];

const PURPOSE_TONE: Record<string, string> = {
  deployed_sleeve:      "bg-ok/15 text-ok",
  deploy_replacement:   "bg-ok/15 text-ok",
  hedge_replacement:    "bg-ok/15 text-ok",
  cousin_anchor:        "bg-info/15 text-info",
  candidate:            "bg-warn/15 text-warn",
};

export default function LabLibraryPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <LabLibraryPageInner />
    </Suspense>
  );
}

function LabLibraryPageInner() {
  const router = useRouter();
  const [entries, setEntries] = useState<LibEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  // Lifecycle tab inner toggle: Gantt = "where has each strategy been"
  //                            State = "what schema constrains them"
  const [lifecycleView, setLifecycleView] = useState<"gantt" | "state">("gantt");
  const [showUntracked, setShowUntracked] = useState(false);

  useEffect(() => {
    api.libraryInventory()
      .then((r) => setEntries(r.entries))
      .catch((e) => setError(String(e?.message ?? e)));
  }, []);

  const filtered = useMemo(() => {
    if (!entries) return [];
    const q = filter.trim().toLowerCase();
    if (!q) return entries;
    return entries.filter(
      (e) =>
        e.id.toLowerCase().includes(q)
        || (e.family || "").toLowerCase().includes(q)
        || (e.purpose || "").toLowerCase().includes(q),
    );
  }, [entries, filter]);

  // KPI strip: counts by purpose
  const kpis = useMemo(() => {
    if (!entries) return null;
    const c = {
      deployed: 0, cousin_anchor: 0, candidate: 0, total: entries.length,
    };
    for (const e of entries) {
      const p = e.purpose || "";
      if (p.startsWith("deployed") || p === "deploy_replacement"
          || p === "hedge_replacement") c.deployed++;
      else if (p === "cousin_anchor") c.cousin_anchor++;
      else if (p === "candidate") c.candidate++;
    }
    return c;
  }, [entries]);

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="govern"
          title="Mechanism Library"
          subtitle="Deployed mechanisms — identity, provenance (canonical paper), and lifecycle state."
        />
      </motion.div>

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </div>
        </Card>
      )}

      {/* KPI strip — one row, dense (matches OPERATE/library/lessons
          density convention). Replaces the 4-card grid. */}
      {kpis && (
        <motion.div variants={fadeUp}>
          <Card className="p-0 overflow-hidden">
            <div className="grid grid-cols-2 md:grid-cols-4 divide-x divide-border/30">
              <KpiCell label="Total"          value={kpis.total}          tone="muted" />
              <KpiCell label="Deployed"       value={kpis.deployed}       tone="ok" />
              <KpiCell label="Cousin anchors" value={kpis.cousin_anchor}  tone="accent" />
              <KpiCell label="Candidates"     value={kpis.candidate}      tone="warn" />
            </div>
          </Card>
        </motion.div>
      )}

      {/* Tab strip — replaces the 4-accordion stack. */}
      <motion.div variants={fadeUp}>
        <Card className="p-3">
          <Tabs
            urlParam="tab"
            defaultKey="inventory"
            tabs={[
              {
                key:    "inventory",
                label:  "Inventory",
                icon:   FolderTree,
                hint:   "Read-only YAML browser — search by id / family / purpose",
                count:  entries ? entries.length : undefined,
                body:   () => (
                  <InventoryPanel
                    entries={entries}
                    filter={filter}
                    setFilter={setFilter}
                    filtered={filtered} />
                ),
              },
              {
                key:    "lifecycle",
                label:  "Lifecycle",
                icon:   Activity,
                hint:   "Per-strategy journey + state-machine schema",
                body:   () => (
                  <LifecyclePanel
                    view={lifecycleView}
                    setView={setLifecycleView}
                    showUntracked={showUntracked}
                    setShowUntracked={setShowUntracked}
                    onStrategyClick={(sid) =>
                      router.push(`/research/library/detail?id=${encodeURIComponent(sid)}`)
                    } />
                ),
              },
              {
                key:    "network",
                label:  "Network",
                icon:   NetworkIcon,
                hint:   "Force-directed correlation graph — find unintended overlap",
                body:   () => <SleeveCorrelationNetwork height={480} />,
              },
            ]} />
        </Card>
      </motion.div>

      <motion.div variants={fadeUp} className="text-[10px] text-muted/60">
        <FileText className="h-3 w-3 inline mr-1" />
        Source files: <code>data/research/mechanism_library/*.yaml</code>.
        Editing happens via Claude Code / IDE; this page is read-only.
      </motion.div>
    </motion.div>
  );
}


// ── KPI cell ───────────────────────────────────────────────────────


function KpiCell({ label, value, tone }: {
  label: string;
  value: number;
  tone:  "ok" | "warn" | "danger" | "muted" | "accent";
}) {
  const cls =
    tone === "ok"     ? "text-ok" :
    tone === "warn"   ? "text-warn" :
    tone === "danger" ? "text-danger" :
    tone === "accent" ? "text-accent" :
                        "text-muted";
  return (
    <div className="px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.15em] text-muted/60 leading-none">
        {label}
      </div>
      <div className={`tnum text-lg font-semibold leading-tight mt-1 ${cls}`}>
        {value}
      </div>
    </div>
  );
}


// ── Inventory tab body ─────────────────────────────────────────────


type InventoryView = "live" | "reference" | "queue" | "all";


function InventoryPanel({
  entries, filter, setFilter, filtered,
}: {
  entries:   LibEntry[] | null;
  filter:    string;
  setFilter: (v: string) => void;
  filtered:  LibEntry[];
}) {
  // R2.8 — Live / Reference / Queue split. PMs think about these
  // three classes differently:
  //   live      = on the book today; daily question = decay/risk
  //   reference = historical cousin; question = "has this been tried"
  //   queue     = candidate in the pipeline; question = "is it next"
  const [view, setView] = useState<InventoryView>("live");

  const classified = useMemo(() => {
    const live: LibEntry[] = [];
    const ref:  LibEntry[] = [];
    const q:    LibEntry[] = [];
    for (const e of filtered) {
      const p = e.purpose || "";
      if (p.startsWith("deployed") || p === "deploy_replacement"
          || p === "hedge_replacement") live.push(e);
      else if (p === "cousin_anchor") ref.push(e);
      else if (p === "candidate")     q.push(e);
    }
    return { live, ref, q };
  }, [filtered]);

  const shown =
    view === "live"      ? classified.live :
    view === "reference" ? classified.ref  :
    view === "queue"     ? classified.q    :
                            filtered;

  return (
    <div className="space-y-3">
      {/* Filter + class sub-tabs */}
      <div className="flex flex-wrap items-center gap-3">
        <SegmentToggle
          value={view}
          onChange={setView}
          options={[
            { key: "live",      label: `Live · ${classified.live.length}`,
              hint: "Deployed sleeves + replacements — today's book" },
            { key: "reference", label: `Reference · ${classified.ref.length}`,
              hint: "Cousin anchors — historical reference points" },
            { key: "queue",     label: `Queue · ${classified.q.length}`,
              hint: "Candidates in the pipeline" },
            { key: "all",       label: `All · ${filtered.length}`,
              hint: "Combined view across all classes" },
          ]} />
        <div className="inline-flex items-center gap-2 text-xs ml-auto">
          <Search className="h-3.5 w-3.5 text-muted" />
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="filter by id / family / purpose"
            className="rounded border border-muted/20 bg-bg px-2 py-1 text-xs w-64" />
          <span className="text-muted tnum">
            {entries ? `${shown.length} / ${entries.length}` : "loading…"}
          </span>
        </div>
      </div>

      {!entries && <Skeleton className="h-24 w-full" />}

      {entries && shown.length === 0 && (
        <Card className="p-4 text-[12px] text-muted/70 text-center">
          {view === "live"      ? "No deployed sleeves match the current filter." :
           view === "reference" ? "No cousin anchors match the current filter." :
           view === "queue"     ? "Queue is empty — no candidates pending pipeline runs." :
                                  "No mechanisms match the current filter."}
        </Card>
      )}

      {entries && shown.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                <th className="px-2 py-1.5">mechanism</th>
                <th className="px-2 py-1.5">parent family</th>
                <th className="px-2 py-1.5">deployment status</th>
                <th className="px-2 py-1.5">canonical paper</th>
                <th className="px-2 py-1.5">cost-aware filter</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((e) => (
                <tr key={e.id} className="border-b border-muted/10 last:border-0 hover:bg-muted/5 group">
                  <td className="px-2 py-1.5 font-mono">
                    <Link href={`/research/library/detail?id=${encodeURIComponent(e.id)}`}
                          className="inline-flex flex-col gap-0.5 hover:text-accent transition-colors group/link">
                      <span className="inline-flex items-center gap-1.5">
                        <Layers className="h-3 w-3 text-accent/70 group-hover:text-accent" strokeWidth={1.75} />
                        <span className="font-medium">
                          {e.family
                            ? e.family.split(/[_-]/).map((w) => w[0]?.toUpperCase() + w.slice(1)).join(" ")
                            : e.id}
                        </span>
                        <ChevronRight className="h-3 w-3 opacity-0 group-hover:opacity-60 transition-opacity" />
                      </span>
                      <span className="text-[10px] text-muted/70 font-mono ml-4.5">{e.id}</span>
                    </Link>
                  </td>
                  <td className="px-2 py-1.5 text-muted text-[10px]">
                    {e.parent_family || "—"}
                  </td>
                  <td className="px-2 py-1.5">
                    {e.purpose ? (
                      <Badge tone={PURPOSE_TONE[e.purpose] || "bg-muted/15 text-muted"}>
                        {e.purpose}
                      </Badge>
                    ) : "—"}
                  </td>
                  <td className="px-2 py-1.5 font-mono text-[10px] text-muted">
                    {e.canonical_paper_id || "—"}
                  </td>
                  <td className="px-2 py-1.5 text-[10px] text-muted">
                    {e.ca_filter_k_method || "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}


// ── Lifecycle tab body (Gantt ⟷ State machine toggle) ──────────────


function LifecyclePanel({
  view, setView, showUntracked, setShowUntracked, onStrategyClick,
}: {
  view:           "gantt" | "state";
  setView:        (v: "gantt" | "state") => void;
  showUntracked:  boolean;
  setShowUntracked: (v: boolean) => void;
  onStrategyClick:  (sid: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <SegmentToggle
          value={view}
          onChange={setView}
          options={[
            { key: "gantt", label: "Journey view",
              hint: "Per-strategy timeline — where has THIS strategy been" },
            { key: "state", label: "Schema view",
              hint: "State machine — what states does the SYSTEM allow" },
          ]} />
        {view === "gantt" && (
          <label className="inline-flex items-center gap-1.5 text-[11px] text-muted cursor-pointer">
            <input type="checkbox"
                   checked={showUntracked}
                   onChange={(e) => setShowUntracked(e.target.checked)} />
            show untracked
          </label>
        )}
      </div>

      {view === "gantt" && (
        <LifecycleGantt
          showUntracked={showUntracked}
          onStrategyClick={onStrategyClick} />
      )}
      {view === "state" && <SlmStateMachine />}
    </div>
  );
}
