"use client";

// NextStepHint — small one-line strip surfacing "what to do here +
// where to go after". Drop on any landing surface (forward / lessons
// / papers / etc) so first-time users have an inline orientation
// without consulting the Daily Flow card on Today.
//
// Render: arrow + plain-language one-liner + optional next CTA.
// Dismissible per-page via localStorage (persisted across sessions).

import { ReactNode, useEffect, useState } from "react";
import Link from "next/link";
import { Compass, ArrowRight, X } from "lucide-react";


export function NextStepHint({
  storageKey,
  what,
  next,
  nextHref,
}: {
  /** localStorage key — once dismissed, hint stays hidden across reloads */
  storageKey: string;
  /** Single sentence answering "what is this page" */
  what: ReactNode;
  /** Short label for the next-step CTA (e.g. "Open Forward vectors") */
  next?: string;
  nextHref?: string;
}) {
  const [dismissed, setDismissed] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      setDismissed(localStorage.getItem(storageKey) === "1");
    } catch {}
  }, [storageKey]);

  const dismiss = () => {
    setDismissed(true);
    if (typeof window === "undefined") return;
    try { localStorage.setItem(storageKey, "1"); } catch {}
  };

  if (dismissed) return null;

  return (
    <div className="rounded border border-accent/25 bg-accent/[0.03] px-3 py-1.5
                    flex items-center gap-2 text-[11px] text-foreground/85">
      <Compass className="h-3.5 w-3.5 text-accent shrink-0" strokeWidth={2} />
      <span className="flex-1">{what}</span>
      {next && nextHref && (
        <Link href={nextHref}
          className="text-accent hover:underline inline-flex items-center gap-0.5 shrink-0">
          {next} <ArrowRight className="h-2.5 w-2.5" />
        </Link>
      )}
      <button onClick={dismiss}
        title="Hide this hint for good"
        className="text-muted/60 hover:text-foreground shrink-0">
        <X className="h-3 w-3" />
      </button>
    </div>
  );
}
