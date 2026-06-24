"use client";

import { History, X } from "lucide-react";
import { useAsOf } from "@/lib/asof";
import { useI18n } from "@/lib/i18n";

// Shown across the terminal when the as-of picker is pinned to a past date — so historical book/
// risk/holdings figures are never mistaken for live. The decay monitor + data refresh stay live
// (forward-looking), which the note states. "Return to live" clears the pin.
export function HistoricalBanner() {
  const { t } = useI18n();
  const { asOf, setAsOf } = useAsOf();
  if (!asOf) return null;
  return (
    <div className="border-b border-warn/30 bg-warn/10">
      <div className="mx-auto flex w-full max-w-6xl items-center gap-3 px-6 py-2 text-xs">
        <History className="h-3.5 w-3.5 shrink-0 text-warn" />
        <span className="font-medium text-warn">{t("asof.viewing")} <span className="tnum">{asOf}</span></span>
        <span className="hidden text-muted sm:inline">· {t("asof.scope")}</span>
        <button onClick={() => setAsOf(null)}
          className="ml-auto flex shrink-0 items-center gap-1.5 rounded-md border border-warn/40 bg-warn/10 px-2.5 py-1 text-warn transition-colors hover:bg-warn/20">
          {t("asof.return_live")} <X className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
