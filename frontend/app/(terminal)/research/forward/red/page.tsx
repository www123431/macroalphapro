"use client";

// /research/forward/red — Stage A piece 6b.
//
// Surfaces recent strict-gate RED verdicts so the principal (and A's
// next synthesis run) can see WHICH directions have already been
// ruled out — preventing redundant proposals. Consumed from
// GET /api/research/red_outcomes (piece 6a).
//
// Layout: KPI strip → window selector → table. Per Cockpit doctrine
// — KPI scan first, then action, then detail.

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { RefreshCw, Loader2, AlertTriangle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, Badge, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { StalenessBadge } from "@/components/StalenessBadge";
import { useI18n } from "@/lib/i18n";


type RedItem = {
  event_id:             string;
  subject_id:           string;
  family:               string;
  verdict_ts:           string;
  score:                number;
  summary:              string;
  source_hypothesis_id: string | null;
  source_paper_id:      string | null;
  source_paper_title:   string | null;
};

type RedResponse = {
  since:      string;
  n_total:    number;
  n_returned: number;
  items:      RedItem[];
};


type WindowDays = 30 | 90 | 180 | 365;


export default function RedOutcomesPage() {
  const { t } = useI18n();
  const [days, setDays]       = useState<WindowDays>(30);
  const [data, setData]       = useState<RedResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr]         = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch(
        `${API_BASE}/api/research/red_outcomes?days=${days}&limit=200`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="space-y-4">
      <ModeHeader
        mode="research"
        title={t("red.title")}
        subtitle={t("red.subtitle")}
      />

      {/* KPI + window selector */}
      <Card className="p-4">
        <div className="flex flex-wrap items-center gap-3 justify-between">
          <div className="flex items-center gap-3">
            <Badge className="bg-danger/15 text-danger border-danger/40">
              {data?.n_total ?? 0} {t("red.total")}
            </Badge>
            <span className="text-xs text-muted">
              {t("red.window")}:
            </span>
            <div className="flex gap-1" role="radiogroup">
              {([30, 90, 180, 365] as WindowDays[]).map((d) => (
                <button
                  key={d}
                  role="radio"
                  aria-checked={days === d}
                  onClick={() => setDays(d)}
                  className={cn(
                    "px-2 py-1 text-xs rounded border transition-colors",
                    days === d
                      ? "bg-accent/15 text-accent border-accent/40"
                      : "border-muted/30 text-muted hover:border-accent/40",
                  )}
                >
                  {t(`red.window.${d}d`)}
                </button>
              ))}
            </div>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className={cn(
              "flex items-center gap-1 px-2 py-1 text-xs rounded",
              "border border-muted/30 text-muted hover:border-accent/40",
              loading && "opacity-50 cursor-wait",
            )}
            aria-label={t("red.refresh")}
          >
            {loading
              ? <Loader2 className="w-3 h-3 animate-spin" />
              : <RefreshCw className="w-3 h-3" />}
            <span>{t("red.refresh")}</span>
          </button>
        </div>
      </Card>

      {/* Body */}
      {err && (
        <Card className="p-4 border-danger/40">
          <div className="flex items-center gap-2 text-danger">
            <AlertTriangle className="w-4 h-4" />
            <span>{t("red.error")}: {err}</span>
          </div>
        </Card>
      )}

      {!err && loading && !data && (
        <Card className="p-6 flex items-center justify-center gap-2 text-muted">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>{t("red.loading")}</span>
        </Card>
      )}

      {!err && data && data.items.length === 0 && (
        <Card className="p-6 text-center text-muted">
          {t("red.empty")}
        </Card>
      )}

      {!err && data && data.items.length > 0 && (
        <Card className="p-0 overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-muted/5 border-b border-muted/20">
              <tr className="text-left text-xs uppercase text-muted">
                <th className="px-3 py-2 font-medium">
                  {t("red.col.family")}
                </th>
                <th className="px-3 py-2 font-medium">
                  {t("red.col.verdict_ts")}
                </th>
                <th className="px-3 py-2 font-medium">
                  {t("red.col.score")}
                </th>
                <th className="px-3 py-2 font-medium">
                  {t("red.col.summary")}
                </th>
                <th className="px-3 py-2 font-medium">
                  {t("red.col.paper")}
                </th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((it) => (
                <tr
                  key={it.event_id}
                  className="border-b border-muted/10 hover:bg-muted/5"
                >
                  <td className="px-3 py-2">
                    <Badge className="bg-muted/15 text-muted border-muted/30">
                      {it.family}
                    </Badge>
                  </td>
                  <td className="px-3 py-2 text-xs whitespace-nowrap">
                    <StalenessBadge
                      asOf={it.verdict_ts}
                      compact
                      warnAfterDays={14}
                      alertAfterDays={60}
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <span className={cn(
                      "px-1.5 py-0.5 text-xs rounded font-mono",
                      it.score <= 1
                        ? "bg-danger/15 text-danger"
                        : it.score <= 3
                          ? "bg-warn/15 text-warn"
                          : "bg-muted/15 text-muted",
                    )}>
                      {it.score}/7
                    </span>
                  </td>
                  <td className="px-3 py-2 max-w-xl">
                    <p className="text-xs leading-snug text-muted line-clamp-2">
                      {it.summary || "—"}
                    </p>
                  </td>
                  <td className="px-3 py-2 max-w-sm">
                    {it.source_paper_id && it.source_paper_title ? (
                      <Link
                        href={`/research/papers/${it.source_paper_id}`}
                        className="text-xs text-accent hover:underline line-clamp-2"
                      >
                        {it.source_paper_title}
                      </Link>
                    ) : (
                      <span className="text-xs text-muted/60 italic">
                        {t("red.no_paper")}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
