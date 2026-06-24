// frontend/lib/intents.ts — small fire-and-forget helper for filing
// typed intents from page CTAs. The whole point is that the user
// CLICKS, the UI NAVIGATES, and in the background we file an
// intent record so Claude can pick it up via the hook poll.
//
// Failures are NOT surfaced to the user — a network blip during the
// intent file shouldn't block their navigation. Server-side audit
// can replay from access logs if needed.

import { API_BASE } from "@/lib/api";


export type IntentKind =
  | "audit_subject"
  | "pipeline_test"
  | "research_test"
  | "ingest_paper"
  | "re_audit_decay"
  | "review_lesson"
  | "explore_hypothesis"
  | "annotate_doctrine";


export type IntentSubjectType =
  | "mechanism" | "hypothesis" | "paper" | "lesson" | "sleeve" | "session" | "axis";


export type FileIntentArgs = {
  kind:         IntentKind;
  subject_type: IntentSubjectType;
  subject_id:   string;
  source_page?: string;
  filed_by?:    string;
  payload?:     Record<string, unknown>;
};


export type FileIntentResult =
  | { ok: true;  intent_id: string }
  | { ok: false; status: number; reason: string; hint?: string };


export async function fileIntent(args: FileIntentArgs): Promise<FileIntentResult> {
  try {
    const sourcePage = args.source_page
      ?? (typeof window !== "undefined" ? window.location.pathname : "");
    const res = await fetch(`${API_BASE}/api/intents/file`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        kind:         args.kind,
        subject_type: args.subject_type,
        subject_id:   args.subject_id,
        source_page:  sourcePage,
        filed_by:     args.filed_by ?? "user",
        payload:      args.payload ?? {},
      }),
    });
    if (res.ok) {
      const data = await res.json();
      return { ok: true, intent_id: data?.intent_id ?? "" };
    }
    // 409 surfaces a typed reason (P1-C: DQ HALT guard).
    if (res.status === 409) {
      try {
        const body = await res.json();
        const d = body?.detail || {};
        return {
          ok:     false,
          status: 409,
          reason: typeof d === "string" ? d : (d.rationale || d.error || "conflict"),
          hint:   typeof d === "object" ? d.hint : undefined,
        };
      } catch {
        return { ok: false, status: 409, reason: "conflict" };
      }
    }
    return { ok: false, status: res.status, reason: `HTTP ${res.status}` };
  } catch (e: any) {
    return { ok: false, status: 0, reason: String(e?.message ?? e) };
  }
}
