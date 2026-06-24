"use client";

// Breadcrumb — Lab workspace navigation breadcrumb.
//
// Shows the user's path: Lab > Council > sim-42-0023
// Used on detail pages to provide back-navigation context.

import Link from "next/link";
import { ChevronRight } from "lucide-react";
import { cn } from "@/components/ui";

export type Crumb = {
  label: string;
  href?: string;   // omitted = leaf (current page)
  mono?: boolean;  // true for IDs / spec_ids
};

export function Breadcrumb({ crumbs }: { crumbs: Crumb[] }) {
  return (
    <nav className="flex items-center gap-1 text-[11px] text-muted">
      {crumbs.map((c, i) => (
        <span key={i} className="inline-flex items-center gap-1">
          {i > 0 && <ChevronRight className="h-3 w-3 text-muted/40" strokeWidth={2} />}
          {c.href ? (
            <Link href={c.href} className="hover:text-foreground transition-colors">
              <span className={cn(c.mono && "font-mono")}>{c.label}</span>
            </Link>
          ) : (
            <span className={cn(
              "text-foreground",
              c.mono && "font-mono",
            )}>
              {c.label}
            </span>
          )}
        </span>
      ))}
    </nav>
  );
}
