"use client";

import { ShieldCheck } from "lucide-react";
import { useI18n } from "@/lib/i18n";

// The 0-LLM-in-DECISION provenance line: every verdict states the deterministic function that
// COMPUTED it and the agent that only NARRATES it. Shown under verdict evidence (decay, risk).
export function DecisionProvenance({ decidedBy, narratedBy }: { decidedBy?: string; narratedBy?: string }) {
  const { t } = useI18n();
  if (!decidedBy) return null;
  return (
    <p className="flex flex-wrap items-center gap-x-1.5 gap-y-1 border-t border-border pt-2 text-[11px] text-muted">
      <ShieldCheck className="h-3 w-3 text-ok" />
      {t("prov.computed_by")} <code className="tnum text-foreground/80">{decidedBy}</code>
      {narratedBy && <span>· {t("prov.narrated_by")} <span className="text-foreground/80">{narratedBy}</span></span>}
      <span>· <span className="text-ok">{t("prov.zero_llm")}</span></span>
    </p>
  );
}
