// frontend/lib/intentRouter.ts — Natural-language → deterministic command router.
//
// Doctrine: the user should never NEED to type `/`. They type a question
// or instruction in plain English; this module detects intent via cheap
// regex / keyword matching and routes to the appropriate command. Only
// when no deterministic intent matches do we fall back to /ask (LLM RAG).
//
// Senior rationale: 90%+ of common quant queries hit a small set of
// patterns ("show pfh", "what sleeves are decaying", "open factor X").
// LLM-based intent classification is over-engineering for that recall
// rate. Start with regex; upgrade to LLM router if measured recall fails.
//
// Slash commands still work as explicit shortcuts for power users — see
// lib/chatCommands.ts. This module only fires when input does NOT start
// with `/`.

import { CommandResponse, ExecContext, executeCommandLine } from "@/lib/chatCommands";


export interface IntentMatch {
  rewrite: string;        // The command line we'll execute (e.g. "/pfh 6")
  intent: string;         // Human-readable intent name for debugging
  confidence: "high" | "medium";
}


/** Try to detect a deterministic command intent from natural language input.
 * Returns null if no pattern matches (caller should fall back to /ask). */
export function detectIntent(input: string): IntentMatch | null {
  const norm = input.trim().toLowerCase();
  if (!norm) return null;

  // ── /pfh — factor suggestions ───────────────────────────────────
  // "show 6 pfh suggestions" / "give me PFH 10" / "pfh constrained 6"
  // "generate factor candidates" / "suggest factors"
  {
    const m = norm.match(/(?:pfh|factor suggestion|suggest factor|generate.*candidate)/);
    if (m) {
      // Extract k if mentioned
      const kMatch = norm.match(/\b(\d{1,2})\b/);
      const k = kMatch ? kMatch[1] : "6";
      const mode = norm.includes(" open") ? "open" : "constrained";
      return {
        rewrite: `/pfh ${k} ${mode}`.trim(),
        intent: "pfh.suggest",
        confidence: "high",
      };
    }
  }

  // ── /decay — sleeve decay snapshot ──────────────────────────────
  // "decay" / "sleeve health" / "what sleeves are decaying"
  // "show decay" / "decay status"
  {
    const m = norm.match(/\b(decay|decaying|sleeve.*health|sleeve.*alert|health.*sleeve)\b/);
    if (m) {
      // Try to extract a sleeve name (anything after "for"/"of")
      const sleeveMatch = norm.match(/(?:for|of)\s+([a-z_]{3,30}\b)/);
      if (sleeveMatch) {
        return {
          rewrite: `/decay ${sleeveMatch[1]}`,
          intent: "decay.sleeve",
          confidence: "medium",
        };
      }
      return {
        rewrite: "/decay",
        intent: "decay.snapshot",
        confidence: "high",
      };
    }
  }

  // ── /council — recent critique runs ─────────────────────────────
  // "show council runs" / "council history" / "council last 5"
  // GUARD: any question word ("what" "who" "why" "how" "did") falls
  // through to /ask. Only explicit list/show/recent/last triggers.
  {
    const m = norm.match(/\bcouncil\b/);
    const isQuestion = /\b(what|who|why|how|did|when|where)\b/.test(norm)
                        || norm.includes("?");
    const isExplicitList = /\b(show|list|recent|last|all)\b/.test(norm)
                            || /^council(\s|$)/.test(norm);
    if (m && !isQuestion && isExplicitList) {
      const nMatch = norm.match(/(?:last|recent|show)\s+(\d{1,2})/);
      const n = nMatch ? nMatch[1] : "10";
      let consensus = "";
      if (norm.includes("approve")) consensus = " APPROVE";
      else if (norm.includes("reject")) consensus = " REJECT";
      else if (norm.includes("revision")) consensus = " NEEDS_REVISION";
      return {
        rewrite: `/council ${n}${consensus}`,
        intent: "council.list",
        confidence: "medium",
      };
    }
  }

  // ── /factor — open factor detail ────────────────────────────────
  // "open factor eq_mom_12_1_us_real" / "show me spec X"
  // "factor X" / "details for X"
  {
    const m = norm.match(/(?:open|show|details? for|factor)\s+([a-z][a-z0-9_]{6,})/);
    if (m) {
      const candidate = m[1];
      // Heuristic: must look like a spec_id (snake_case, ≥2 underscores
      // or known prefix) to avoid false positives like "show factor lab"
      const looksLikeSpecId = (candidate.match(/_/g) || []).length >= 2
                                || candidate.startsWith("pfh_")
                                || candidate.startsWith("eq_")
                                || candidate.includes("_carry_")
                                || candidate.includes("_momentum");
      if (looksLikeSpecId) {
        return {
          rewrite: `/factor ${candidate}`,
          intent: "factor.detail",
          confidence: "high",
        };
      }
    }
  }

  // ── /sleeve — sleeve timeline ───────────────────────────────────
  // "show sleeve equity_book" / "open sleeve carry_book"
  {
    const m = norm.match(/(?:sleeve|timeline)\s+([a-z][a-z0-9_]{3,30})/);
    if (m) {
      return {
        rewrite: `/sleeve ${m[1]}`,
        intent: "sleeve.timeline",
        confidence: "high",
      };
    }
  }

  // ── /library — mechanism browser ───────────────────────────────
  // "library" / "show library" / "library post_earnings_drift"
  {
    const m = norm.match(/^(?:show |open )?library(?:\s+([a-z][a-z0-9_]{3,30}))?$/);
    if (m) {
      return {
        rewrite: m[1] ? `/library ${m[1]}` : "/library",
        intent: "library.browse",
        confidence: "high",
      };
    }
  }

  // ── /chains — chain catalogue ──────────────────────────────────
  {
    const m = norm.match(/^(?:show |list )?chain(s|)\b/);
    if (m && !norm.includes("?")) {
      return {
        rewrite: "/chains",
        intent: "chains.list",
        confidence: "high",
      };
    }
  }

  // ── /help — help ───────────────────────────────────────────────
  if (norm === "help" || norm === "?" || norm === "commands"
      || norm === "what can you do" || norm === "what commands") {
    return { rewrite: "/help", intent: "help", confidence: "high" };
  }

  // ── No match → fall back to /ask ────────────────────────────────
  return null;
}


/** Execute natural language input via intent routing.
 *
 * Flow:
 *   1. If input starts with /, delegate to slash-command parser as-is.
 *   2. Otherwise: detectIntent → if hit, execute rewritten command line.
 *   3. Otherwise: fall back to /ask <input> for LLM RAG.
 *
 * The returned CommandResponse carries an `intent_route` field in its
 * payload so the UI can show "(routed from natural language)" hint.
 */
export async function executeUserInput(
  input: string, ctx: ExecContext,
): Promise<CommandResponse & { intent_route?: IntentMatch | null }> {
  const trimmed = input.trim();
  if (trimmed.startsWith("/")) {
    // Explicit command — no routing needed
    return executeCommandLine(trimmed, ctx);
  }

  const intent = detectIntent(trimmed);
  if (intent) {
    const resp = await executeCommandLine(intent.rewrite, ctx);
    return { ...resp, intent_route: intent };
  }

  // No matching intent → /ask
  const resp = await executeCommandLine(`/ask ${trimmed}`, ctx);
  return { ...resp, intent_route: null };
}
