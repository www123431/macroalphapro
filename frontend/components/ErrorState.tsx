"use client";

import { useI18n } from "@/lib/i18n";

// Errors arrive as strings from lib/api.ts: HTTP failures format as "<status> <path>: <detail>"
// (e.g. "404 /api/approvals: Not Found"), while a true backend-down fetch rejection is a bare
// message ("Failed to fetch") with NO status code. We classify on that:
//   404  → server is UP but the route is missing = a STALE build that predates a new endpoint.
//          The fix is a RESTART, not "start the backend". (This is the recurring-404 footgun.)
//   none → no HTTP status reached us = backend genuinely not reachable. Start it.
//   other (422/500/503…) → reachable, request errored. Show the detail; don't mislead.
// The status token is matched even when a page prefixes the message (e.g. "(same origin) · 404 /…").
function classify(message: string): "stale" | "unreachable" | "failed" {
  const m = /\b([1-5]\d{2})\s+\/\S/.exec(message);
  const status = m ? Number(m[1]) : null;
  if (status === 404) return "stale";
  if (status === null) return "unreachable";
  return "failed";
}

export function ErrorState({ message }: { message: string }) {
  const { t } = useI18n();
  const kind = classify(message);

  return (
    <div className="rounded-xl border border-alert/40 bg-panel/80 p-5 backdrop-blur-sm">
      {kind === "stale" && (
        <>
          <p className="font-medium text-alert">{t("err.stale")}</p>
          <p className="mt-1 text-sm text-muted">
            {t("err.stale_hint")}{" "}
            <code className="rounded bg-panel2 px-1.5 py-0.5 text-accent">python run_app.py</code>
          </p>
        </>
      )}
      {kind === "unreachable" && (
        <>
          <p className="font-medium text-alert">{t("err.unreachable")}</p>
          <p className="mt-1 text-sm text-muted">
            {t("err.start_it")}{" "}
            <code className="rounded bg-panel2 px-1.5 py-0.5 text-accent">python run_app.py</code>
          </p>
        </>
      )}
      {kind === "failed" && <p className="font-medium text-alert">{t("err.failed")}</p>}

      <p className="mt-2 break-words text-xs text-muted">{message}</p>
    </div>
  );
}
