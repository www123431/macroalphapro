"use client";

// /research/papers — paper library index.
//
// 2026-06-04 R2.3 REBUILD. Removed accordion stack; converted to
// 3-tab layout (Library / Chain / Architecture) syncing to ?tab=…
// for deep links.

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Card, Badge } from "@/components/ui";
import PaperChainSearch from "@/components/PaperChainSearch";
import { ModeHeader } from "@/components/ModeHeader";
import { Tabs } from "@/components/Tabs";
import { PaperChainSankey } from "@/components/PaperChainSankey";
import { SystemFlowDiagram } from "@/components/SystemFlowDiagram";
import { BookOpen, BarChart3, Workflow } from "lucide-react";
import { useI18n } from "@/lib/i18n";


type PaperSummary = {
  paper_id:        string;
  title:           string;
  year:            number;
  authors:         string[];
  venue:           string;
  doi:             string;
  fulltext_status: "ingested" | "metadata_only" | "paywalled" | "unattempted";
  n_chunks:        number;
  shelves:         string[];
  n_hypotheses:    number;
  n_tested:        number;
};

const STATUS_TONE: Record<string, string> = {
  ingested:       "bg-ok/15 text-ok",
  metadata_only:  "bg-muted/15 text-muted",
  paywalled:      "bg-warn/15 text-warn",
  unattempted:    "bg-info/15 text-info",
};

const SHELF_TONE: Record<string, string> = {
  doctrine_method:    "bg-accent/15 text-accent",
  green_motivation:   "bg-ok/15 text-ok",
  green_critique:     "bg-ok/10 text-ok",
  yellow_motivation:  "bg-warn/15 text-warn",
  red_motivation:     "bg-danger/15 text-danger",
  red_critique:       "bg-danger/10 text-danger",
  dormant_revisit:    "bg-info/15 text-info",
  other:              "bg-muted/15 text-muted",
};

export default function PapersIndexPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <PapersIndexInner />
    </Suspense>
  );
}

function PapersIndexInner() {
  const { t } = useI18n();
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [shelfFilter, setShelfFilter] = useState<string>("");

  useEffect(() => {
    const params = new URLSearchParams();
    if (statusFilter) params.set("fulltext_status", statusFilter);
    if (shelfFilter)  params.set("shelf", shelfFilter);
    setLoading(true);
    fetch(`${API_BASE}/api/paper_chain/papers?${params}`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data: PaperSummary[]) => {
        // Sort: ingested first, then by hypothesis count desc, then by year desc
        const ordered = [...data].sort((a, b) => {
          if (a.fulltext_status !== b.fulltext_status) {
            return a.fulltext_status === "ingested" ? -1 : 1;
          }
          if (a.n_hypotheses !== b.n_hypotheses) return b.n_hypotheses - a.n_hypotheses;
          return b.year - a.year;
        });
        setPapers(ordered);
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [statusFilter, shelfFilter]);

  const totalChunks = papers.reduce((s, p) => s + p.n_chunks, 0);
  const totalHyps   = papers.reduce((s, p) => s + p.n_hypotheses, 0);
  const totalTested = papers.reduce((s, p) => s + p.n_tested, 0);

  return (
    <div className="p-6 space-y-4">
      <ModeHeader
        mode="research"
        title="Paper library"
        subtitle={<>
          <b className="text-foreground tabular-nums">{papers.length}</b> papers ·{" "}
          <b className="text-foreground tabular-nums">{totalChunks}</b> chunks ·{" "}
          <b className="text-foreground tabular-nums">{totalHyps}</b> hypotheses{" "}
          (<span className="text-ok">{totalTested}</span> tested)
        </>}
        right={
          <div className="flex items-center gap-2">
            <Link href="/research/papers/incoming"
              className="inline-flex items-center gap-1.5 rounded border border-accent/40 text-accent hover:bg-accent/10 px-2.5 py-1 text-[11px] font-semibold">
              {t("papers.todays_incoming")}
            </Link>
            <Link href="/research/papers/new"
              className="inline-flex items-center gap-1.5 rounded bg-accent text-background hover:bg-accent/90 px-2.5 py-1 text-[11px] font-semibold">
              {t("papers.ingest_button")}
            </Link>
            <PaperChainSearch />
          </div>
        }
      />

      <Card className="p-3">
        <Tabs
          urlParam="tab"
          defaultKey="library"
          tabs={[
            {
              key:    "library",
              label:  "Library",
              icon:   BookOpen,
              hint:   "Filterable paper table — drill into hypotheses",
              count:  papers.length,
              body:   () => (
                <LibraryTablePanel
                  papers={papers}
                  loading={loading}
                  error={error}
                  statusFilter={statusFilter}
                  setStatusFilter={setStatusFilter}
                  shelfFilter={shelfFilter}
                  setShelfFilter={setShelfFilter} />
              ),
            },
            {
              key:    "chain",
              label:  "Chain",
              icon:   BarChart3,
              hint:   "Sankey of PAPER → HYPOTHESIS → TEST → VERDICT flow",
              body:   () => <PaperChainSankey height={420} />,
            },
            {
              key:    "architecture",
              label:  "Architecture",
              icon:   Workflow,
              hint:   "System flow — INGEST → TRIAGE → TEST → VERDICT → DEPLOY",
              body:   () => (
                <div id="system-flow" className="bg-bg/30 p-2 rounded">
                  <SystemFlowDiagram />
                </div>
              ),
            },
          ]} />
      </Card>
    </div>
  );
}


// ── Library table tab body ─────────────────────────────────────────


function LibraryTablePanel({
  papers, loading, error,
  statusFilter, setStatusFilter,
  shelfFilter,  setShelfFilter,
}: {
  papers:        PaperSummary[];
  loading:       boolean;
  error:         string | null;
  statusFilter:  string;
  setStatusFilter: (v: string) => void;
  shelfFilter:   string;
  setShelfFilter:  (v: string) => void;
}) {
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 pb-3 border-b border-line mb-2">
        <span className="text-xs uppercase text-muted">status:</span>
        {(["", "ingested", "metadata_only", "paywalled"] as const).map((s) => (
          <button key={s || "all"}
                  className={`px-2 py-0.5 text-xs rounded border ${
                    s === statusFilter
                      ? "bg-accent/20 border-accent text-accent"
                      : "border-line text-muted hover:text-fg"
                  }`}
                  onClick={() => setStatusFilter(s)}>
            {s || "all"}
          </button>
        ))}
        <span className="text-xs uppercase text-muted ml-4">shelf:</span>
        <select className="bg-bg border border-line rounded text-xs px-2 py-0.5"
                value={shelfFilter}
                onChange={(e) => setShelfFilter(e.target.value)}>
          <option value="">all</option>
          {Object.keys(SHELF_TONE).map((sh) => (
            <option key={sh} value={sh}>{sh}</option>
          ))}
        </select>
      </div>

      {loading && <div className="p-6 text-sm text-muted">Loading…</div>}
      {error && <div className="p-6 text-sm text-danger">Error: {error}</div>}

      {!loading && !error && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-xs text-muted border-b border-line">
              <tr>
                <th className="text-left p-3">Title</th>
                <th className="text-left p-3">Authors</th>
                <th className="text-left p-3">Year</th>
                <th className="text-left p-3">Status</th>
                <th className="text-left p-3">Chunks</th>
                <th className="text-left p-3">Hypotheses</th>
                <th className="text-left p-3">Shelves</th>
              </tr>
            </thead>
            <tbody>
              {papers.map((p) => (
                <tr key={p.paper_id}
                    className="border-b border-line hover:bg-accent/5">
                  <td className="p-3">
                    <Link href={`/research/papers/${p.paper_id}`}
                          className="hover:text-accent">
                      {p.title}
                    </Link>
                  </td>
                  <td className="p-3 text-xs text-muted">
                    {p.authors.slice(0, 2).join(", ")}
                    {p.authors.length > 2 && " et al."}
                  </td>
                  <td className="p-3 text-xs">{p.year}</td>
                  <td className="p-3">
                    <Badge className={STATUS_TONE[p.fulltext_status]}>
                      {p.fulltext_status}
                    </Badge>
                  </td>
                  <td className="p-3 text-xs">{p.n_chunks}</td>
                  <td className="p-3 text-xs">
                    {p.n_hypotheses > 0 ? (
                      <>
                        {p.n_hypotheses}
                        {p.n_tested > 0 &&
                          <span className="text-ok ml-1">({p.n_tested} tested)</span>}
                      </>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td className="p-3">
                    <div className="flex flex-wrap gap-1">
                      {p.shelves.map((s) => (
                        <Badge key={s} className={`text-[10px] ${SHELF_TONE[s] || ""}`}>
                          {s}
                        </Badge>
                      ))}
                    </div>
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
