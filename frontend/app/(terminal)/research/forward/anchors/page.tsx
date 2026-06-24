"use client";

// /research/forward/anchors — Stage C Tier B-2.
//
// Browse the canonical anchor library A consults during synthesis:
// T1 doctrine (methodology gates) + T2 anchor (mechanism classes).
// Each row shows the tier_anchor_summary the principal can audit.
//
// Consumes GET /api/research/anchor_library (built Tier B-1).

import { useCallback, useEffect, useState } from "react";
import { RefreshCw, Loader2, AlertTriangle, BookMarked } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, Badge, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { useI18n } from "@/lib/i18n";


type AnchorItem = {
  paper_id:           string;
  title:              string;
  authors:            string[];
  year:               number;
  venue:              string;
  doi:                string;
  tier:               "T1_DOCTRINE" | "T2_ANCHOR";
  tier_rationale:     string;
  tier_anchor_summary:string;
  tier_classified_ts: string;
};

type AnchorResponse = {
  n_total: number;
  n_t1:    number;
  n_t2:    number;
  items:   AnchorItem[];
};


export default function AnchorLibraryPage() {
  const { t } = useI18n();
  const [data, setData]       = useState<AnchorResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]         = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/anchor_library`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const t1Items = data?.items.filter((i) => i.tier === "T1_DOCTRINE") ?? [];
  const t2Items = data?.items.filter((i) => i.tier === "T2_ANCHOR") ?? [];

  return (
    <div className="space-y-4">
      <ModeHeader
        mode="research"
        title={t("anchors.title")}
        subtitle={t("anchors.subtitle")}
      />

      {/* KPI strip */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-3 justify-between">
          <div className="flex items-center gap-3">
            <Badge className="bg-accent/15 text-accent border-accent/40">
              {data?.n_total ?? 0} total
            </Badge>
            <Badge className="bg-danger/15 text-danger border-danger/40">
              T1: {data?.n_t1 ?? 0}
            </Badge>
            <Badge className="bg-info/15 text-info border-info/40">
              T2: {data?.n_t2 ?? 0}
            </Badge>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className={cn(
              "flex items-center gap-1 px-2 py-1 text-xs rounded",
              "border border-muted/30 text-muted hover:border-accent/40",
              loading && "opacity-50 cursor-wait",
            )}
          >
            {loading
              ? <Loader2 className="w-3 h-3 animate-spin" />
              : <RefreshCw className="w-3 h-3" />}
            <span>{t("red.refresh")}</span>
          </button>
        </div>
      </Card>

      {err && (
        <Card className="p-4 border-danger/40">
          <div className="flex items-center gap-2 text-danger">
            <AlertTriangle className="w-4 h-4" />
            <span>{t("anchors.error")}: {err}</span>
          </div>
        </Card>
      )}

      {!err && loading && !data && (
        <Card className="p-6 flex items-center justify-center gap-2 text-muted">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>{t("anchors.loading")}</span>
        </Card>
      )}

      {!err && data && data.items.length === 0 && (
        <Card className="p-6 text-center text-muted">
          {t("anchors.empty")}
        </Card>
      )}

      {/* T1 doctrine section */}
      {t1Items.length > 0 && (
        <div>
          <div className="px-2 py-1 mb-2 flex items-center gap-3">
            <BookMarked className="w-4 h-4 text-danger" />
            <h3 className="text-sm font-medium text-danger">
              {t("anchors.t1_label")}
            </h3>
            <span className="text-xs text-muted">
              {t("anchors.t1_hint")}
            </span>
          </div>
          <Card className="p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/5 border-b border-muted/20">
                <tr className="text-left text-xs uppercase text-muted">
                  <th className="px-3 py-2 font-medium w-1/3">
                    {t("anchors.col.paper")}
                  </th>
                  <th className="px-3 py-2 font-medium">
                    {t("anchors.col.summary")}
                  </th>
                  <th className="px-3 py-2 font-medium w-1/4">
                    {t("anchors.col.rationale")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {t1Items.map((it) => (
                  <AnchorRow key={it.paper_id} item={it} t={t} />
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}

      {/* T2 anchor section */}
      {t2Items.length > 0 && (
        <div>
          <div className="px-2 py-1 mb-2 flex items-center gap-3">
            <BookMarked className="w-4 h-4 text-info" />
            <h3 className="text-sm font-medium text-info">
              {t("anchors.t2_label")}
            </h3>
            <span className="text-xs text-muted">
              {t("anchors.t2_hint")}
            </span>
          </div>
          <Card className="p-0 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/5 border-b border-muted/20">
                <tr className="text-left text-xs uppercase text-muted">
                  <th className="px-3 py-2 font-medium w-1/3">
                    {t("anchors.col.paper")}
                  </th>
                  <th className="px-3 py-2 font-medium">
                    {t("anchors.col.summary")}
                  </th>
                  <th className="px-3 py-2 font-medium w-1/4">
                    {t("anchors.col.rationale")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {t2Items.map((it) => (
                  <AnchorRow key={it.paper_id} item={it} t={t} />
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  );
}


function AnchorRow({ item, t }: {
  item: AnchorItem;
  t: (k: string) => string;
}) {
  const firstAuthor = item.authors[0] ?? "?";
  const moreAuthors = item.authors.length > 1
    ? ` +${item.authors.length - 1}`
    : "";
  return (
    <tr className="border-b border-muted/10 hover:bg-muted/5">
      <td className="px-3 py-2 align-top">
        <div className="text-xs font-mono text-muted">
          {item.paper_id.slice(0, 8)}
        </div>
        <div className="text-sm leading-snug">
          {item.title}
        </div>
        <div className="text-xs text-muted mt-0.5">
          {firstAuthor}{moreAuthors} · {item.year} · {item.venue || "—"}
        </div>
      </td>
      <td className="px-3 py-2 align-top">
        {item.tier_anchor_summary ? (
          <p className="text-xs leading-snug">
            {item.tier_anchor_summary}
          </p>
        ) : (
          <span className="text-xs text-muted/60 italic">
            {t("anchors.no_summary")}
          </span>
        )}
      </td>
      <td className="px-3 py-2 align-top">
        <p className="text-xs text-muted leading-snug">
          {item.tier_rationale || "—"}
        </p>
      </td>
    </tr>
  );
}
