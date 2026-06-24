"use client";

// renderAnswer — shared markdown renderer for Ask AI responses.
//
// The LLM returns markdown ("## 我能做什么", "**bold**", "`code`",
// `[type:id]` citations, bulleted lists, numbered lists). All three
// chat surfaces (ChatFloater side panel · Cmd-K Ask mode · /chat full
// page) need to render it as actual styled text, not raw markdown.
// Avoids adding a heavy react-markdown dep — the patterns the LLM
// uses are a small finite set.
//
// Citations remain clickable links to the relevant detail page.

import React from "react";
import Link from "next/link";


function citeUrl(type: string, id: string): string {
  switch (type) {
    case "run_id":       return `/lab/council/detail?run_id=${encodeURIComponent(id)}`;
    case "iteration_id": return `/lab/l4/detail?id=${encodeURIComponent(id)}`;
    case "spec_id":      return `/lab/factor-lab/detail?id=${encodeURIComponent(id)}`;
    case "sleeve":       return `/research/decay/detail?sleeve=${encodeURIComponent(id)}`;
    default:             return "#";
  }
}


// Inline-level parsing: citations · **bold** · `code`.
// Single-pass scan with combined alternation so nesting doesn't bite
// us. Whitespace tolerance for `**bold** ` and `** bold **` because
// the LLM sometimes emits the latter.
function parseInline(text: string, citationFontSize = "text-[11px]"): React.ReactNode[] {
  const pattern = /(\[(?:run_id|iteration_id|spec_id|sleeve):[a-zA-Z0-9_\-]+\])|(\*\*\s*[^*\n]+?\s*\*\*)|(`[^`\n]+?`)/g;
  const nodes: React.ReactNode[] = [];
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = pattern.exec(text)) !== null) {
    if (m.index > lastIdx) {
      nodes.push(text.slice(lastIdx, m.index));
    }
    const [matched, cite, bold, code] = m;
    if (cite) {
      const sub = /\[([^:]+):(.+)\]/.exec(cite);
      if (sub) {
        const type = sub[1], id = sub[2];
        nodes.push(
          <Link key={`c${key++}`} href={citeUrl(type, id)}
                className={`inline-flex items-baseline gap-0.5 mx-0.5 px-1 rounded bg-accent/10 text-accent hover:bg-accent/20 ${citationFontSize} font-mono`}>
            {type}:{id.slice(0, 14)}
          </Link>
        );
      }
    } else if (bold) {
      const inner = bold.replace(/^\*\*\s*/, "").replace(/\s*\*\*$/, "");
      nodes.push(<strong key={`b${key++}`} className="font-semibold text-foreground">{inner}</strong>);
    } else if (code) {
      const inner = code.slice(1, -1);
      nodes.push(
        <code key={`co${key++}`}
              className="rounded bg-panel2/60 px-1 py-0.5 font-mono text-[12px] text-foreground/95">
          {inner}
        </code>
      );
    }
    lastIdx = m.index + matched.length;
  }
  if (lastIdx < text.length) {
    nodes.push(text.slice(lastIdx));
  }
  return nodes;
}


// Block-level parsing: headings · paragraphs · bullets · numbered list.
// Consecutive bullet / numbered lines are kept as siblings (not grouped
// into a single <ul> / <ol>) which keeps the renderer simple while still
// looking right — institutional terminals tend to render flat lists.
export function renderAnswerMarkdown(
  text: string,
  options: { citationFontSize?: string } = {},
): React.ReactNode {
  const citationFontSize = options.citationFontSize || "text-[11px]";
  const lines = text.split(/\r?\n/);
  const blocks: React.ReactNode[] = [];
  let buffer: React.ReactNode[] = [];
  let key = 0;

  const flushBuffer = () => {
    if (buffer.length > 0) {
      blocks.push(
        <p key={`p${key++}`} className="leading-relaxed">
          {buffer}
        </p>
      );
      buffer = [];
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();

    if (line.trim() === "") {
      flushBuffer();
      continue;
    }

    // Heading: ##, ###, etc. (start of line; LLMs sometimes leave a
    // trailing colon or space)
    const heading = /^(#{1,6})\s+(.+?)\s*:?\s*$/.exec(line);
    if (heading) {
      flushBuffer();
      const level = heading[1].length;
      const Tag = (level <= 2 ? "h3" : level === 3 ? "h4" : "h5") as keyof React.JSX.IntrinsicElements;
      const size = level <= 2 ? "text-base" : level === 3 ? "text-sm" : "text-xs";
      blocks.push(
        React.createElement(
          Tag,
          {
            key: `h${key++}`,
            className: `${size} font-semibold text-foreground mt-3 mb-1.5 first:mt-0`,
          },
          parseInline(heading[2], citationFontSize)
        )
      );
      continue;
    }

    // Bulleted list: "- foo" or "* foo"
    const bullet = /^[-*]\s+(.+)$/.exec(line);
    if (bullet) {
      flushBuffer();
      blocks.push(
        <div key={`bl${key++}`} className="flex gap-1.5 ml-3 leading-relaxed">
          <span className="text-muted/70 shrink-0">·</span>
          <span className="flex-1">{parseInline(bullet[1], citationFontSize)}</span>
        </div>
      );
      continue;
    }

    // Numbered list: "1. foo"
    const numbered = /^(\d+)\.\s+(.+)$/.exec(line);
    if (numbered) {
      flushBuffer();
      blocks.push(
        <div key={`nu${key++}`} className="flex gap-1.5 ml-3 leading-relaxed">
          <span className="text-muted/70 shrink-0 tnum">{numbered[1]}.</span>
          <span className="flex-1">{parseInline(numbered[2], citationFontSize)}</span>
        </div>
      );
      continue;
    }

    // Regular text line — append to current paragraph buffer
    if (buffer.length > 0) buffer.push(" ");
    buffer.push(...parseInline(line, citationFontSize));
  }

  flushBuffer();

  return <div className="space-y-2">{blocks}</div>;
}
