"use client";

// HandoffToClaude — one-click bridge from a Web surface to Claude
// Code (the VS Code extension). Closes 断点 4 (R3.5 audit): all
// prior CTAs left the user with a copied-brief and a manual app
// switch. This component bundles intent file + clipboard + best-
// effort VS Code launch into a single click.
//
// What it actually does (in order):
//   1. fileIntent({...})                        — typed intent for Claude poll
//   2. navigator.clipboard.writeText(prompt)    — brief in the clipboard
//   3. window.location.href = vscode://...      — best-effort VS Code launch
//   4. inline toast confirming what fired
//
// The vscode:// URI scheme has 3 layers we can probe:
//   - vscode://file/<absolute-path>             opens / focuses workspace
//   - vscode://<extension-id>/<command>?args    invokes an extension command
//   - vscode://anthropic.claude-code/start      hypothetical (Claude Code's
//                                               actual scheme TBC — when
//                                               confirmed, swap CLAUDE_URI)
//
// Browsers vary on whether a non-installed scheme handler shows a
// prompt or silently fails. The clipboard + intent paths are the
// reliable surface; the URI is a nice-to-have. The toast tells the
// user what definitely happened.

import { useState } from "react";
import { Sparkles, CheckCircle2, Loader2 } from "lucide-react";
import { fileIntent, type FileIntentArgs } from "@/lib/intents";


// Repository workspace path — used as the vscode:// file argument.
// In a multi-machine setup this could come from a server endpoint;
// for now we read it from a constant + env var override.
const DEFAULT_WORKSPACE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_REPO_WORKSPACE)
  || "${REPO_ROOT}/Desktop/intern";


// Best-guess Claude Code extension URI. If the actual extension
// publishes a different scheme, change ONLY this constant — the rest
// of the component stays untouched.
const CLAUDE_URI = "vscode://anthropic.claude-code/open";


type Status = "idle" | "firing" | "fired" | "error";


export function HandoffToClaude({
  intent,
  prompt,
  workspacePath = DEFAULT_WORKSPACE,
  label = "Hand off to Claude",
  className = "",
}: {
  intent:        FileIntentArgs;
  prompt:        string;
  workspacePath?: string;
  label?:        string;
  className?:    string;
}) {
  const [status, setStatus] = useState<Status>("idle");
  const [note, setNote] = useState<string>("");

  const handleClick = async () => {
    setStatus("firing");
    setNote("");
    const parts: string[] = [];

    // 1. File the intent so Claude's hook picks it up regardless
    // of whether the URI handler / clipboard succeed.
    const r = await fileIntent(intent);
    if (r.ok) parts.push("intent filed");
    else if (r.status === 409) parts.push(`intent REFUSED — ${r.reason}`);

    // 2. Clipboard — fire-and-forget; some browsers gate this on
    // a user gesture (this IS one), so it'll usually succeed.
    if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(prompt);
        parts.push("brief copied");
      } catch { /* ignore */ }
    }

    // 3. Try the Claude Code URI; fall back to a plain workspace
    // open. Both forms run via window.location so the OS handler
    // can pick them up. Some browsers show a permission dialog;
    // that's fine — silent no-op otherwise.
    try {
      const url = `${CLAUDE_URI}?prompt=${encodeURIComponent(prompt.slice(0, 4000))}`;
      window.location.href = url;
      parts.push("Claude Code launching");
    } catch {
      try {
        window.location.href = `vscode://file/${workspacePath}`;
        parts.push("VS Code launching");
      } catch { /* ignore */ }
    }

    setStatus(parts.length ? "fired" : "error");
    setNote(parts.length ? parts.join(" · ") : "nothing fired — check console");
    setTimeout(() => { setStatus("idle"); setNote(""); }, 5000);
  };

  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <button onClick={handleClick}
        disabled={status === "firing"}
        className="inline-flex items-center gap-1.5 rounded-md bg-accent text-background hover:bg-accent/90 disabled:opacity-50 px-3 py-1.5 text-[12px] font-semibold">
        {status === "firing"
          ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
          : status === "fired"
          ? <CheckCircle2 className="h-3.5 w-3.5" />
          : <Sparkles className="h-3.5 w-3.5" />}
        {status === "fired" ? "Handed off" : label}
      </button>
      {note && (
        <span className="text-[10.5px] text-muted/80">{note}</span>
      )}
    </span>
  );
}
