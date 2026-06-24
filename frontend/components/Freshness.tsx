"use client";

import { useEffect, useState } from "react";
import { RotateCw } from "lucide-react";
import { useI18n } from "@/lib/i18n";

function ago(ms: number): string {
  // Bare duration string — caller wraps in i18n "ago / 前" suffix.
  // (Previously this returned "12s ago" and the template added "ago"
  // again, producing "updated 12s ago ago". Fixed 2026-06-14.)
  if (!ms) return "—";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h`;
}

// "updated 12s ago" (re-renders each second), or "refreshing…" while a background fetch runs.
// `updatedAt` = React Query's dataUpdatedAt (when our COPY last refreshed) — distinct from the
// data's own `as_of` vintage, which pages show separately.
export function Freshness({ updatedAt, isFetching }: { updatedAt: number; isFetching?: boolean }) {
  const { t } = useI18n();
  const [, tick] = useState(0);
  useEffect(() => {
    const tmr = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(tmr);
  }, []);
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
      {isFetching
        ? <RotateCw className="h-3 w-3 animate-spin text-accent" />
        : <span className="h-1.5 w-1.5 rounded-full bg-ok/70" />}
      {isFetching ? t("fresh.refreshing") : `${t("fresh.updated")} ${ago(updatedAt)} ${t("fresh.ago")}`}
    </span>
  );
}
