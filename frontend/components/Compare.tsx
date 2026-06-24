"use client";

// Compare — generic side-by-side comparison panel. Lifted from the
// /research/forward implementation (R2.9) and made polymorphic so
// any list page can support multi-select compare without re-writing
// the diff / highlight / sticky-bar pieces.
//
// Pattern: caller maintains a Set<string> of selected keys; passes
// the matching items to CompareBar. CompareBar renders a sticky
// bottom Card with:
//   - one COLUMN per selected item
//   - one ROW per field the caller declares (label, picker, optional
//     mono / href)
//   - the LEFT-most column is the "reference"; cells in other columns
//     whose value matches the reference get a soft green tint so the
//     eye picks up overlap.
//   - an optional action row at the bottom (e.g. "Open in pipeline →").
//
// Usage example (forward vectors):
//
//   <CompareBar
//     items={selectedForwardVectors}
//     getKey={(v) => v.source_hypothesis_id}
//     onClear={...} onRemove={(v) => toggleSelected(v.source_hypothesis_id)}
//     headerCell={(v) => <ForwardVectorHeaderChips v={v} />}
//     rows={[
//       { label: "claim",     pick: (v) => v.claim },
//       { label: "family",    pick: (v) => v.mechanism_family, mono: true },
//       { label: "paper",     pick: (v) => v.paper_title,
//         href: (v) => `/research/papers/${v.source_paper_id}` },
//     ]}
//     actionCell={(v) => <Link href={...}>Open session →</Link>}
//   />

import { ReactNode, useState } from "react";
import Link from "next/link";
import { Card } from "@/components/ui";


export type CompareField<T> = {
  label: string;
  pick:  (item: T) => string;
  mono?: boolean;
  href?: (item: T) => string;
};


export function CompareBar<T>({
  items, onClear, onRemove, getKey,
  headerCell, rows, actionCell,
  title = "Compare panel",
}: {
  items:       T[];
  onClear:     () => void;
  onRemove:    (item: T) => void;
  getKey:      (item: T) => string;
  headerCell:  (item: T) => ReactNode;
  rows:        CompareField<T>[];
  actionCell?: (item: T) => ReactNode;
  title?:      string;
}) {
  const [expanded, setExpanded] = useState(true);
  if (items.length === 0) return null;

  return (
    <div className="sticky bottom-3 z-20 mt-3">
      <Card className="border-accent/30 bg-panel/95 backdrop-blur shadow-lg shadow-bg/40">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border/30">
          <span className="text-[11px] font-semibold text-foreground">{title}</span>
          <span className="text-[10px] text-muted/70">
            {items.length} selected · click a column header to remove
          </span>
          <button onClick={() => setExpanded((v) => !v)}
            className="ml-auto text-[10px] text-muted hover:text-foreground">
            {expanded ? "collapse" : "expand"}
          </button>
          <button onClick={onClear}
            className="text-[10px] text-muted hover:text-danger">
            clear all
          </button>
        </div>

        {expanded && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-[11px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/70">
                  <th className="px-3 py-2 sticky left-0 bg-panel/95 z-10">field</th>
                  {items.map((it) => (
                    <th key={getKey(it)}
                        className="px-3 py-2 min-w-[200px] hover:text-danger cursor-pointer"
                        onClick={() => onRemove(it)}
                        title="click to remove">
                      {headerCell(it)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="text-[11px]">
                {rows.map((row, ri) => (
                  <CompareRow key={ri} row={row} items={items} getKey={getKey} />
                ))}
                {actionCell && (
                  <tr className="border-b border-border/30 bg-panel2/20">
                    <td className="px-3 py-2 text-[9px] uppercase tracking-wider text-muted/70 sticky left-0 bg-panel/95">
                      action
                    </td>
                    {items.map((it) => (
                      <td key={getKey(it)} className="px-3 py-2">
                        {actionCell(it)}
                      </td>
                    ))}
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}


function CompareRow<T>({
  row, items, getKey,
}: {
  row:    CompareField<T>;
  items:  T[];
  getKey: (item: T) => string;
}) {
  const ref = row.pick(items[0]);
  return (
    <tr className="border-b border-border/30">
      <td className="px-3 py-2 text-[9px] uppercase tracking-wider text-muted/70 sticky left-0 bg-panel/95">
        {row.label}
      </td>
      {items.map((it) => {
        const val = row.pick(it);
        const same = items.length >= 2 && val === ref;
        const cls = `${row.mono ? "font-mono" : ""} ${same ? "text-foreground/95" : "text-foreground/80"}`;
        const content = row.href ? (
          <Link href={row.href(it)} className={`${cls} underline-offset-2 hover:underline hover:text-accent`}>
            {val || "—"}
          </Link>
        ) : (
          <span className={cls}>{val || "—"}</span>
        );
        return (
          <td key={getKey(it)}
              className={`px-3 py-2 align-top ${same ? "bg-ok/[0.04]" : ""}`}>
            {content}
          </td>
        );
      })}
    </tr>
  );
}
