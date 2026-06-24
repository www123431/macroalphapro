"use client";

// SlashMenu — autocomplete dropdown for the chat command input.
//
// Pattern reference: Slack /commands, Linear Cmd-K, Notion slash menu.
// Activated when the user types "/" as the first character; dismisses
// when no longer at start-of-line / arrow-keys-then-Enter selects.

import { useEffect, useRef, useState } from "react";
import { CommandDef, matchCommands } from "@/lib/chatCommands";
import { cn } from "@/components/ui";

interface SlashMenuProps {
  query: string;       // current input value (used to filter)
  open: boolean;       // controlled by parent (true when '/' typed)
  onSelect: (def: CommandDef) => void;
  onClose: () => void;
}

export function SlashMenu({ query, open, onSelect, onClose }: SlashMenuProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [highlight, setHighlight] = useState(0);
  const candidates = matchCommands(query);

  // Reset highlight when candidate list changes
  useEffect(() => {
    setHighlight(0);
  }, [query, open]);

  // Keyboard navigation
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHighlight((h) => Math.min(candidates.length - 1, h + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setHighlight((h) => Math.max(0, h - 1));
      } else if (e.key === "Enter" && candidates.length > 0) {
        // The host textarea also receives Enter; we intercept first
        e.preventDefault();
        onSelect(candidates[highlight]);
      } else if (e.key === "Escape") {
        onClose();
      } else if (e.key === "Tab" && candidates.length > 0) {
        e.preventDefault();
        onSelect(candidates[highlight]);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, candidates, highlight, onSelect, onClose]);

  if (!open || candidates.length === 0) return null;

  return (
    <div ref={containerRef}
         className="absolute bottom-full left-0 right-0 mb-1 z-30
                    rounded-lg border border-border bg-panel shadow-2xl
                    overflow-hidden max-h-[50vh] overflow-y-auto">
      <div className="px-3 py-1.5 border-b border-border/40 text-[10px]
                      uppercase tracking-wider text-muted bg-bg/30">
        commands · {candidates.length}
      </div>
      <div className="divide-y divide-border/20">
        {candidates.map((c, i) => (
          <button key={c.slug}
                  onClick={() => onSelect(c)}
                  onMouseEnter={() => setHighlight(i)}
                  className={cn(
                    "w-full text-left px-3 py-2 transition-colors flex items-baseline gap-3",
                    i === highlight ? "bg-accent/10" : "hover:bg-muted/5",
                  )}>
            <div className="flex-1 min-w-0">
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-sm text-accent">/{c.slug}</span>
                <span className="text-[10px] uppercase tracking-wider text-muted/60">
                  {c.category}
                </span>
              </div>
              <div className="text-xs text-muted mt-0.5">{c.description}</div>
              <div className="text-[10px] font-mono text-muted/60 mt-0.5">{c.usage}</div>
            </div>
            {i === highlight && (
              <div className="text-[10px] text-accent/60 whitespace-nowrap">
                Enter ↩
              </div>
            )}
          </button>
        ))}
      </div>
      <div className="px-3 py-1.5 border-t border-border/40 text-[10px] text-muted/60 bg-bg/30">
        ↑↓ navigate · Enter/Tab select · Esc dismiss
      </div>
    </div>
  );
}
