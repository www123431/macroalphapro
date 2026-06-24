"use client";

// Tip — minimal accessible tooltip primitive. Pure CSS positioning +
// React state for show/hide; no Radix dependency.
//
// Why not Radix Tooltip:
//   - Radix adds ~30kb + provider wiring in the layout
//   - Our use cases need: appear on hover/focus, styled to match
//     theme, optional shortcut key + caption rows. CSS-only covers
//     all of it in ~80 lines.
//
// Usage:
//   <Tip content="State definition: PROPOSED — candidate submitted, not yet audited">
//     <span className="px-1 rounded">PROPOSED</span>
//   </Tip>
//
//   <Tip side="right" content={<><b>g t</b>  Open Today</>}>
//     <kbd>g</kbd>
//   </Tip>
//
// Behavior:
//   - Shows on mouseenter (after a 200ms delay to suppress
//     jitter) AND on focus (keyboard a11y).
//   - Hides on mouseleave / blur / Esc.
//   - Renders absolutely above the trigger; positioning side
//     configurable.
//   - aria-describedby wired so screen readers announce the tip.
//
// Print-safe: the tooltip is a `display: none` element until
// hovered; never appears in PDFs.

import { ReactNode, useEffect, useId, useRef, useState } from "react";


type Side = "top" | "bottom" | "left" | "right";


export function Tip({
  content, children, side = "top", block = false, className = "",
}: {
  content:   ReactNode;
  children:  ReactNode;
  side?:     Side;
  /** Wrap children with display:block instead of inline-flex.
   *  Use when the child is a Link/div with full-width layout (e.g.
   *  KPI cells in a grid). */
  block?:    boolean;
  className?: string;
}) {
  const id = useId();
  const [visible, setVisible] = useState(false);
  const showT = useRef<ReturnType<typeof setTimeout> | null>(null);

  const show = () => {
    if (showT.current) clearTimeout(showT.current);
    showT.current = setTimeout(() => setVisible(true), 200);
  };
  const hide = () => {
    if (showT.current) { clearTimeout(showT.current); showT.current = null; }
    setVisible(false);
  };

  // Dismiss on Esc when visible.
  useEffect(() => {
    if (!visible) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setVisible(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [visible]);

  // Position classes per side. Anchored to the wrapper via absolute.
  const pos: Record<Side, string> = {
    top:    "bottom-full left-1/2 -translate-x-1/2 mb-1.5",
    bottom: "top-full left-1/2 -translate-x-1/2 mt-1.5",
    left:   "right-full top-1/2 -translate-y-1/2 mr-1.5",
    right:  "left-full top-1/2 -translate-y-1/2 ml-1.5",
  };

  const wrapperClass = block
    ? `relative block ${className}`
    : `relative inline-flex ${className}`;
  // Use a div when block to avoid invalid HTML (interactive Links
  // inside <span> is technically OK but block layout is cleaner with
  // a div wrapper).
  const Wrapper: any = block ? "div" : "span";

  return (
    <Wrapper className={wrapperClass}
             onMouseEnter={show}
             onMouseLeave={hide}
             onFocus={show}
             onBlur={hide}
             aria-describedby={visible ? id : undefined}>
      {children}
      {visible && (
        <span id={id} role="tooltip"
              className={`pointer-events-none absolute z-50 ${pos[side]} whitespace-pre rounded-md border border-border/50 bg-panel/95 backdrop-blur px-2 py-1 text-[10.5px] text-foreground/90 shadow-lg shadow-bg/40 max-w-xs no-print`}>
          {content}
        </span>
      )}
    </Wrapper>
  );
}
