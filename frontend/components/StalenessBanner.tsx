"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, X, RefreshCw } from "lucide-react";
import { useFreshness, useRefreshStatus, useStartRefresh } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/components/ui";

// The book SELF-HEALS: a data refresh is operational (not a decision), so it auto-runs when the
// live book goes stale (server-side _auto_refresh_loop) — no human click required. This banner is
// therefore informational ("auto-refreshing…") and only ASKS for the user when the auto-refresh
// FAILED (e.g. a Risk HARD HALT) — i.e. only when there is a real reason needing attention, with
// the reason shown + a manual retry. Routine upkeep never nags; decisions still go to /approvals.
export function StalenessBanner() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const { data } = useFreshness();
  const { data: rs } = useRefreshStatus(true);
  const start = useStartRefresh();
  const [hidden, setHidden] = useState(false);
  const prevFinished = useRef<string | null>(null);

  // On a successful refresh, invalidate everything so freshness re-evaluates and this banner clears.
  useEffect(() => {
    if (rs?.finished_at && rs.finished_at !== prevFinished.current) {
      prevFinished.current = rs.finished_at;
      if (rs.ok) qc.invalidateQueries();
    }
  }, [rs?.finished_at, rs?.ok, qc]);

  if (hidden || !data || data.overall !== "stale") return null;

  const running = start.isPending || !!rs?.running;
  const failed = !running && !!rs && rs.exit_code != null && rs.ok === false;
  const auto = rs?.trigger === "auto";
  const tone = failed ? "border-alert/40 bg-alert/10" : "border-warn/30 bg-warn/10";
  const accent = failed ? "text-alert" : "text-warn";

  return (
    <div className={cn("border-b", tone)}>
      <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-6 py-2 text-xs">
        {failed ? <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-alert" />
                : <RefreshCw className={cn("h-3.5 w-3.5 shrink-0", accent, running && "animate-spin")} strokeWidth={2.2} />}

        {running ? (
          <span className={cn("font-medium", accent)}>
            {t("fresh.refresh_running")}{auto && <span className="ml-1 font-normal text-muted">· {t("fresh.auto")}</span>}
          </span>
        ) : failed ? (
          <>
            <span className="font-medium text-alert">{t("fresh.refresh_failed")}</span>
            <span className="truncate text-muted" title={rs?.log_tail || undefined}>{rs?.message}</span>
            <button onClick={() => start.mutate()}
              className="ml-auto flex shrink-0 items-center gap-1.5 rounded-md border border-alert/40 bg-alert/10 px-2.5 py-1 text-alert transition-colors hover:bg-alert/20">
              <RefreshCw className="h-3 w-3" strokeWidth={2.2} /> {t("fresh.retry")}
            </button>
          </>
        ) : (
          <>
            <span className={cn("font-medium", accent)}>{t("fresh.auto_healing")}</span>
            <button onClick={() => start.mutate()}
              className="ml-auto flex shrink-0 items-center gap-1.5 rounded-md border border-warn/40 bg-warn/10 px-2.5 py-1 text-warn transition-colors hover:bg-warn/20">
              <RefreshCw className="h-3 w-3" strokeWidth={2.2} /> {t("fresh.refresh_now")}
            </button>
          </>
        )}

        <button onClick={() => setHidden(true)} aria-label={t("fresh.dismiss")}
          className={cn("shrink-0 rounded p-0.5 text-muted transition-colors hover:text-foreground", !failed && !running && "ml-0")}>
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}
