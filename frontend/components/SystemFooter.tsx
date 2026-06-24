"use client";

// SystemFooter — the operational "what code is actually running" footer.
//
// Why it exists (2026-06-02):
// During the dashboard-freshness PR series the user repeatedly hit the
// "I changed code but the page didn't update" footgun — uvicorn's
// --reload missed file changes on Windows twice, and there was no way
// to verify "is the backend on the commit I think it's on" without
// SSHing into the server console.
//
// This footer surfaces:
//   - backend git SHA (truncated) + dirty flag — eyeball check vs the
//     commit the frontend was built from (front-end SHA via env)
//   - backend process uptime — confirms "did I actually restart it"
//   - N cached compute keys + a button to invalidate them all
//
// Doctrine: ops surface = "what the system is doing". This is not a
// preference — it's a diagnostic. Lives in /ops, not /settings.

import { useQueryClient } from "@tanstack/react-query";
import { GitBranch, Clock, Database, RotateCw } from "lucide-react";
import { useState } from "react";
import { useSystemVersion } from "@/lib/queries";
import { api } from "@/lib/api";
import { Card, SectionTitle, cn } from "@/components/ui";


export function SystemFooter() {
  const { data, refetch } = useSystemVersion();
  const qc = useQueryClient();
  const [invalidating, setInvalidating] = useState(false);
  const [lastDrop, setLastDrop] = useState<number | null>(null);

  const onInvalidate = async () => {
    setInvalidating(true);
    try {
      const r = await api.systemCacheInvalidate();
      setLastDrop(r.n_dropped);
      // Also bust client-side React Query cache for good measure
      await qc.invalidateQueries();
      await refetch();
    } catch (e) {
      setLastDrop(-1);
    } finally {
      setInvalidating(false);
      setTimeout(() => setLastDrop(null), 4000);
    }
  };

  if (!data) {
    return (
      <div>
        <SectionTitle className="mb-0 flex items-center gap-1.5">
          <Database className="h-3.5 w-3.5" />
          <span>System</span>
        </SectionTitle>
        <Card className="mt-2 text-sm text-muted">loading runtime info…</Card>
      </div>
    );
  }

  // Compare with frontend's git SHA (Next.js exposes it via env at build time
  // if we configure it; for now we just display backend SHA prominently).
  const frontendSha = (process.env.NEXT_PUBLIC_GIT_SHA ?? "").slice(0, 7);
  const shaMatch = frontendSha && frontendSha === data.git_sha.slice(0, 7);

  return (
    <div>
      <SectionTitle className="mb-0 flex items-center gap-1.5">
        <Database className="h-3.5 w-3.5" />
        <span>System</span>
        <span className="text-[11px] text-muted font-normal">
          · backend runtime diagnostics
        </span>
      </SectionTitle>
      <Card className="mt-2 space-y-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-4 text-[12px]">
          {/* Git SHA */}
          <div>
            <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5 flex items-center gap-1">
              <GitBranch className="h-3 w-3" /> backend git SHA
            </div>
            <div className="font-mono text-foreground/95 flex items-center gap-1.5">
              <span className="text-base">{data.git_sha}</span>
              {data.git_dirty && (
                <span title="working tree has uncommitted source changes"
                      className="text-[9px] uppercase tracking-wider rounded bg-warn/15 text-warn px-1 py-0.5">
                  dirty
                </span>
              )}
            </div>
            {frontendSha && (
              <div className={cn("text-[10px] mt-0.5",
                shaMatch ? "text-ok/80" : "text-alert/80")}>
                frontend SHA {frontendSha} {shaMatch ? "(match)" : "(MISMATCH)"}
              </div>
            )}
          </div>

          {/* Uptime */}
          <div>
            <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5 flex items-center gap-1">
              <Clock className="h-3 w-3" /> uptime
            </div>
            <div className="font-mono text-foreground/95 text-base">{data.uptime_human}</div>
            <div className="text-[10px] text-muted/70 mt-0.5">
              since {data.process_started_iso.replace("T", " ").replace("Z", " UTC")}
            </div>
          </div>

          {/* Cache stats */}
          <div>
            <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5 flex items-center gap-1">
              <Database className="h-3 w-3" /> cached compute keys
            </div>
            <div className="font-mono text-foreground/95 text-base">{data.n_cached_keys}</div>
            <div className="text-[10px] text-muted/70 mt-0.5 truncate"
                 title={data.cached_keys.join(", ")}>
              {data.cached_keys.slice(0, 4).join(", ")}
              {data.cached_keys.length > 4 && ` +${data.cached_keys.length - 4}`}
            </div>
          </div>

          {/* Cache invalidate action */}
          <div className="flex flex-col items-start">
            <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5">
              actions
            </div>
            <button
              onClick={onInvalidate}
              disabled={invalidating}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-[12px] transition-colors",
                "border-accent/40 bg-accent/5 text-accent hover:bg-accent/15",
                invalidating && "opacity-60 cursor-not-allowed",
              )}>
              <RotateCw className={cn("h-3 w-3", invalidating && "animate-spin")} strokeWidth={2} />
              {invalidating ? "invalidating…" : "Invalidate all caches"}
            </button>
            {lastDrop != null && (
              <div className="text-[10px] text-muted mt-1">
                {lastDrop >= 0
                  ? `dropped ${lastDrop} keys + busted client cache`
                  : "invalidate failed — see network tab"}
              </div>
            )}
          </div>
        </div>

        {/* Footer note explaining when to use this */}
        <div className="border-t border-border/40 pt-2 text-[11px] text-muted/80 leading-relaxed">
          Use <span className="font-mono text-foreground/85">Invalidate all caches</span> when you've changed underlying data (re-pulled WRDS, edited active_deployment.yaml, etc.) and the dashboard still shows the old value despite a hard refresh. The first request after invalidation will be slow as caches re-warm.
          {data.git_dirty && (
            <> Backend SHA shows <span className="text-warn">dirty</span> — there are uncommitted source changes in the running process. If you intend production behaviour, commit + restart uvicorn.</>
          )}
        </div>
      </Card>
    </div>
  );
}
