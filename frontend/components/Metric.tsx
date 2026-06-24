"use client";

import type { ReactNode } from "react";
import { useI18n } from "@/lib/i18n";
import { glossaryText, glossaryRef } from "@/lib/glossary";
import { cn } from "@/components/ui";

// A label that hover-defines itself when the term is in the glossary (dotted underline + native
// title). Cheap, reliable, no popover-clipping issues; upgrade to a styled tooltip later.
export function GlossaryLabel({ term, children, className = "" }:
  { term?: string; children: ReactNode; className?: string }) {
  const { lang } = useI18n();
  const def = term ? glossaryText(term, lang) : undefined;
  if (!def) return <span className={className}>{children}</span>;
  return (
    <span title={def}
      className={cn("cursor-help underline decoration-dotted decoration-muted/40 underline-offset-2", className)}>
      {children}
    </span>
  );
}

// A KPI with a REFERENCE FRAME: the value, a hover-definable label, and what "good" is (threshold/
// target/range) — so a lone number is never shown without context. `reference` overrides the
// glossary's default ref.
export function Metric({ label, term, value, reference, tone = "", className = "" }:
  { label: ReactNode; term?: string; value: ReactNode; reference?: string; tone?: string; className?: string }) {
  const { lang } = useI18n();
  const ref = reference ?? (term ? glossaryRef(term, lang) : undefined);
  return (
    <div className={className}>
      <div className="text-xs text-muted"><GlossaryLabel term={term}>{label}</GlossaryLabel></div>
      <div className={cn("tnum text-lg font-semibold", tone)}>{value}</div>
      {ref && <div className="tnum text-[10px] text-muted/70">{ref}</div>}
    </div>
  );
}
