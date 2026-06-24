"use client";

// KeyboardShortcuts — Linear / Bloomberg / Gmail-style `g + letter`
// two-key navigation. Quants reach for the keyboard far more often
// than the mouse; Cmd-K covers exploratory navigation, this covers
// MUSCLE-MEMORY navigation. Both stay.
//
// Mapping (each letter chosen for mnemonic, no duplicate prefixes):
//
//   g t  -> /dashboard                 OPERATE landing
//   g s  -> /research/sessions               session history + queue
//   g l  -> /research/library                deployed mechanisms
//   g r  -> /research/forward           "what to test next"
//   g p  -> /research/papers            paper library
//   g v  -> /research/lessons           verdicts
//   g i  -> /research/papers/new        ingest a paper (collab)
//   g c  -> /research/candidate         candidate pipeline
//   g d  -> /research/decay                  decay sentinel
//   g k  -> opens Cmd-K palette         (parity with Cmd-K via prefix)
//
// Plus singleton helpers:
//   ?    -> opens a hint cheat-sheet overlay
//   esc  -> dismiss the cheat sheet OR cancel a pending g-prefix
//
// Behaviour rules:
//   - Suppress entirely when focus is inside <input>, <textarea>,
//     [contenteditable], or [role=combobox]. Otherwise typing into
//     a search box would navigate away unexpectedly.
//   - g must be followed by the second key within 1.2s — after that
//     the prefix expires and the user is back at idle.
//   - Cmd-K / Ctrl-K is NOT handled here (CommandPalette owns it).

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";


type ShortcutMap = Record<string, string>;

const G_MAP: ShortcutMap = {
  t: "/dashboard",
  s: "/research/sessions",
  l: "/research/library",
  r: "/research/forward",
  p: "/research/papers",
  v: "/research/lessons",
  i: "/research/papers/new",
  c: "/research/candidate",
  d: "/research/decay",
  e: "/research/enhance",   // guided wizard
};


function isEditableContext(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  if (target.isContentEditable) return true;
  const role = target.getAttribute("role");
  if (role === "combobox" || role === "textbox") return true;
  return false;
}


export function KeyboardShortcuts() {
  const router = useRouter();
  const [gPending, setGPending]     = useState(false);
  const [showCheat, setShowCheat]   = useState(false);
  const gTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Modifier keys are owned by the palette / browser — skip.
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (isEditableContext(e.target)) return;

      const k = e.key.toLowerCase();

      // Esc — cancel pending g, dismiss cheat sheet
      if (e.key === "Escape") {
        if (gPending) setGPending(false);
        if (showCheat) setShowCheat(false);
        return;
      }

      // ? — open cheat sheet
      if (e.key === "?" && !gPending) {
        e.preventDefault();
        setShowCheat((v) => !v);
        return;
      }

      // g — start prefix
      if (k === "g" && !gPending) {
        e.preventDefault();
        setGPending(true);
        if (gTimer.current) clearTimeout(gTimer.current);
        gTimer.current = setTimeout(() => setGPending(false), 1200);
        return;
      }

      // g + <letter>
      if (gPending) {
        const route = G_MAP[k];
        if (gTimer.current) { clearTimeout(gTimer.current); gTimer.current = null; }
        setGPending(false);
        if (route) {
          e.preventDefault();
          router.push(route);
        }
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, gPending, showCheat]);

  return (
    <>
      {/* g pending indicator — minimal corner badge */}
      {gPending && (
        <div className="fixed bottom-4 left-4 z-50 inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-panel/95 backdrop-blur shadow-lg px-2.5 py-1.5">
          <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">g</kbd>
          <span className="text-[10.5px] text-muted">+ letter…</span>
        </div>
      )}

      {/* Cheat sheet overlay */}
      {showCheat && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-bg/70 backdrop-blur-sm"
             onClick={() => setShowCheat(false)}>
          <div className="rounded-md border border-border/40 bg-panel/95 shadow-xl p-5 max-w-md w-[92%]"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center mb-3">
              <span className="text-[13px] font-semibold">Keyboard shortcuts</span>
              <button onClick={() => setShowCheat(false)}
                className="ml-auto text-[11px] text-muted hover:text-foreground">esc</button>
            </div>
            <div className="space-y-1">
              {Object.entries(G_MAP).map(([key, path]) => (
                <div key={key} className="flex items-center justify-between text-[11.5px] text-muted hover:text-foreground transition-colors">
                  <span className="inline-flex items-center gap-1">
                    <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">g</kbd>
                    <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">{key}</kbd>
                  </span>
                  <code className="text-[11px] text-foreground/70">{path}</code>
                </div>
              ))}
              <div className="border-t border-border/30 my-2" />
              <div className="flex items-center justify-between text-[11.5px] text-muted">
                <span className="inline-flex items-center gap-1">
                  <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">⌘</kbd>
                  <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">K</kbd>
                </span>
                <span className="text-[11px] text-foreground/70">command palette · jump anywhere</span>
              </div>
              <div className="flex items-center justify-between text-[11.5px] text-muted">
                <span className="inline-flex items-center gap-1">
                  <kbd className="inline-flex items-center px-1 rounded border border-border/50 bg-panel2/40 font-mono text-[10px]">?</kbd>
                </span>
                <span className="text-[11px] text-foreground/70">toggle this cheat sheet</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
