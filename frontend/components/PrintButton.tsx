"use client";

// PrintButton — tiny "Save as PDF / print" trigger. Uses the browser's
// native print dialog (every modern browser exposes "Save as PDF" in
// it). No new dependencies.
//
// Print styling lives in app/globals.css under @media print:
//   * dark theme reverts to black-on-white
//   * .no-print elements are hidden
//   * .print-only elements appear
//   * .page-break forces a new page
//
// Usage:
//   <PrintButton title="Export this lesson" />
//
// The button itself carries .no-print so it doesn't show up in the
// printed output.

import { Printer } from "lucide-react";


export function PrintButton({
  title = "Save as PDF",
  label = "Export PDF",
  className = "",
}: {
  title?:     string;
  label?:     string;
  className?: string;
}) {
  const onClick = () => {
    if (typeof window !== "undefined") window.print();
  };
  return (
    <button onClick={onClick}
      data-no-print="true"
      title={title}
      className={`no-print inline-flex items-center gap-1 rounded border border-border/60 text-muted hover:text-foreground hover:border-border px-2 py-0.5 text-[11px] ${className}`}>
      <Printer className="h-3 w-3" />
      {label}
    </button>
  );
}
