"use client";

// HelpOnThisPage — the "?" button in ModeHeader.
//
// Solves the "项目太散" gap at the PAGE level (DailyDirective covers the
// global "what to do today" level). One click on the "?" opens the
// floating chat panel pre-filled with a context block describing what
// this page is + 3 common actions + "My question: <cursor>" suffix.
// User completes the question, hits send, chat answers with full
// awareness of where the user is.
//
// Two dispatches:
//   1. open-chat-panel        — opens the side panel (existing event)
//   2. prefill-chat-with      — NEW event picked up by ChatFloater
//
// If no PageContext is registered for the current path, falls back to
// a generic "I'm on <path>, what's this page for?" prompt.

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { HelpCircle } from "lucide-react";
import { getPageContext } from "@/lib/pageContexts";
import { cn } from "@/components/ui";


export const PREFILL_CHAT_EVENT = "prefill-chat-with";


export function HelpOnThisPage({ className }: { className?: string }) {
  const pathname = usePathname() || "/";
  const [hovering, setHovering] = useState(false);

  const ctx = getPageContext(pathname);
  const hasCtx = Boolean(ctx);
  const titleLine = hasCtx
    ? `Ask chat about "${ctx!.title}"`
    : `Ask chat about this page`;

  const handleClick = () => {
    // Commit Y 2026-06-04: page context is now sent as a separate
    // `page_context` field to /chat/ask (ChatFloater reads it from
    // usePathname every send). So we DON'T need to pre-fill the
    // user's textarea with context any more. The "?" button just
    // opens the panel + drops a starter question. If the user wants
    // to type their own, they just clear it.
    const starter = ctx
      ? `What's the next click on ${ctx.title.toLowerCase()} for me right now?`
      : `What's this page for and what should I do next?`;

    if (typeof document !== "undefined") {
      document.dispatchEvent(new CustomEvent("open-chat-panel"));
      document.dispatchEvent(new CustomEvent(PREFILL_CHAT_EVENT, {
        detail: { text: starter, source: "help_on_this_page", pathname },
      }));
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      onMouseEnter={() => setHovering(true)}
      onMouseLeave={() => setHovering(false)}
      aria-label={titleLine}
      title={titleLine}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-2 py-1 text-[10.5px] transition-colors",
        "border-border/40 text-muted hover:text-accent hover:border-accent/40 hover:bg-accent/[0.05]",
        className,
      )}>
      <HelpCircle className="h-3 w-3" strokeWidth={2} />
      <span className="hidden md:inline">
        {hovering && hasCtx ? "Ask about this page" : "?"}
      </span>
    </button>
  );
}
