"use client";

// ChunkViewer — renders one paper chunk with hypothesis-quote highlighting.
//
// Highlights are exact substring matches against chunk.text. PDF
// whitespace artifacts can cause some legitimate quotes to NOT
// highlight; those are still shown in the side annotation list.

import { useMemo } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui";

export type QuoteAnnotation = {
  hypothesis_id: string;
  quote_text:    string;
  section_ref:   string;
  lesson_ids:    string[];
};

export type Chunk = {
  chunk_id:        string;
  text:            string;
  section:         string;
  paragraph_idx:   number;
  quoted_by:       QuoteAnnotation[];
};


type Span =
  | { kind: "text"; text: string }
  | { kind: "quote"; text: string; annotations: QuoteAnnotation[] };


/** Split chunk text into spans, marking quote regions. */
function buildSpans(text: string, annotations: QuoteAnnotation[]): Span[] {
  // Collect ranges by exact-substring search
  type Range = { start: number; end: number; ann: QuoteAnnotation };
  const ranges: Range[] = [];
  for (const ann of annotations) {
    const idx = text.indexOf(ann.quote_text);
    if (idx >= 0) {
      ranges.push({ start: idx, end: idx + ann.quote_text.length, ann });
    }
  }
  if (ranges.length === 0) return [{ kind: "text", text }];

  ranges.sort((a, b) => a.start - b.start);

  // Merge overlapping ranges (multiple annotations on same span)
  const merged: Array<{ start: number; end: number; anns: QuoteAnnotation[] }> = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r.start <= last.end) {
      last.end = Math.max(last.end, r.end);
      last.anns.push(r.ann);
    } else {
      merged.push({ start: r.start, end: r.end, anns: [r.ann] });
    }
  }

  const spans: Span[] = [];
  let cursor = 0;
  for (const m of merged) {
    if (m.start > cursor) {
      spans.push({ kind: "text", text: text.slice(cursor, m.start) });
    }
    spans.push({
      kind: "quote",
      text: text.slice(m.start, m.end),
      annotations: m.anns,
    });
    cursor = m.end;
  }
  if (cursor < text.length) {
    spans.push({ kind: "text", text: text.slice(cursor) });
  }
  return spans;
}


export default function ChunkViewer({ chunk }: { chunk: Chunk }) {
  const spans = useMemo(
    () => buildSpans(chunk.text, chunk.quoted_by),
    [chunk]
  );
  const unmatchedAnns = useMemo(
    () => chunk.quoted_by.filter(
      (a) => chunk.text.indexOf(a.quote_text) < 0
    ),
    [chunk]
  );

  const hasHighlights = chunk.quoted_by.length > 0;

  return (
    <div id={`chunk-${chunk.chunk_id}`}
         className={`border-l-2 pl-4 py-3 ${
           hasHighlights ? "border-accent/50 bg-accent/5" : "border-line"
         }`}>
      <div className="flex items-baseline gap-2 mb-1">
        <code className="text-[10px] text-muted">{chunk.chunk_id}</code>
        {chunk.section && (
          <span className="text-xs text-muted">· {chunk.section}</span>
        )}
        <span className="text-xs text-muted ml-auto">¶{chunk.paragraph_idx}</span>
        {hasHighlights && (
          <Badge className="text-[10px] bg-accent/15 text-accent">
            {chunk.quoted_by.length} quote{chunk.quoted_by.length > 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      <p className="text-sm leading-relaxed whitespace-pre-wrap">
        {spans.map((s, i) =>
          s.kind === "text" ? (
            <span key={i}>{s.text}</span>
          ) : (
            <mark key={i}
                  className="bg-accent/30 text-fg px-0.5 rounded"
                  title={`Cited by ${s.annotations.length} hypothesis ${
                    s.annotations.length > 1 ? "claims" : "claim"
                  }`}>
              {s.text}
            </mark>
          )
        )}
      </p>

      {/* Quote annotations (whether matched-and-highlighted or not) */}
      {chunk.quoted_by.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {chunk.quoted_by.map((ann, i) => {
            const matched = chunk.text.indexOf(ann.quote_text) >= 0;
            return (
              <div key={i} className="text-xs flex items-start gap-2">
                <span className={`mt-0.5 ${
                  matched ? "text-accent" : "text-warn"
                }`}>
                  {matched ? "✓" : "△"}
                </span>
                <div className="flex-1">
                  <div className="text-muted">
                    <code className="text-[10px]">{ann.hypothesis_id.slice(0, 8)}</code>
                    {ann.section_ref && <span> · {ann.section_ref}</span>}
                    {!matched && (
                      <span className="text-warn ml-1">
                        (paraphrase — quote not exact in chunk)
                      </span>
                    )}
                  </div>
                  {!matched && (
                    <p className="italic text-muted mt-0.5">
                      "{ann.quote_text.slice(0, 200)}{ann.quote_text.length > 200 && "…"}"
                    </p>
                  )}
                  {ann.lesson_ids.length > 0 && (
                    <p className="text-ok mt-0.5">
                      tested by:{" "}
                      {ann.lesson_ids.map((lid) => (
                        <Link key={lid} href={`/research/lessons/${lid}`}
                              className="underline hover:text-fg mr-1">
                          {lid.slice(0, 8)}
                        </Link>
                      ))}
                    </p>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
