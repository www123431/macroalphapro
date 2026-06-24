// frontend/lib/wizardState.ts — shared state fetcher used by the
// /research/enhance wizard. Pulls every "did the user do X" signal
// the recipe's step engines need: DQ verdict, decay verdict, active
// session, approved forward vectors (optionally family-filtered),
// recent events, recent lessons.
//
// Polls every 10s so the wizard auto-advances as the user clicks
// through CTAs — no manual refresh required.

import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";


export type WizardState = {
  dq:               { verdict: string; n_breaches: number } | null;
  decay:            { overall: string; n_mechanisms: number } | null;
  activeSession:    { session_id: string; session_type: string } | null;
  approvedForward:  { hypothesis_id: string; mechanism_family: string; mechanism_subtype: string }[];
  recentIntents:    { intent_id: string; kind: string; status: string; filed_ts: string }[];
  recentEvents:     { event_id: string; event_type: string; verdict: string; ts: string }[];
  recentLessons:    { lesson_id: string; verdict: string; mechanism_family: string; created_ts: string }[];
  loading:          boolean;
};


const empty: WizardState = {
  dq:              null,
  decay:           null,
  activeSession:   null,
  approvedForward: [],
  recentIntents:   [],
  recentEvents:    [],
  recentLessons:   [],
  loading:         true,
};


export function useWizardState(opts?: { family?: string; pollMs?: number }): WizardState {
  const [state, setState] = useState<WizardState>(empty);
  const family = opts?.family ?? "";
  const pollMs = opts?.pollMs ?? 10_000;

  useEffect(() => {
    let cancelled = false;
    const fetchAll = async () => {
      const get = async <T = any,>(path: string): Promise<T | null> => {
        try {
          const r = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
          if (!r.ok) return null;
          return await r.json();
        } catch { return null; }
      };

      const [dq, decay, sess, fwd, intents, lessons] = await Promise.all([
        get("/api/dq"),
        get("/api/decay/report"),
        get("/api/sessions/active"),
        get(`/api/paper_chain/forward-vectors?pm_status=approved&top=50${family ? `&mechanism_family=${encodeURIComponent(family)}` : ""}`),
        get("/api/intents?status=pending&limit=20"),
        get("/api/paper_chain/lessons?include_legacy=false&limit=10"),
      ]);

      // Recent events for the active session — endpoint doesn't yet
      // filter by session_id, so pull recent verdict events and let
      // the step engines decide. Cheap (~ms) at our scale.
      let events: any[] = [];
      const evResp = await get(`/api/research_store/events?event_type=factor_verdict_filed&limit=20`);
      events = Array.isArray((evResp as any)?.events) ? (evResp as any).events : [];

      if (cancelled) return;
      setState({
        loading:          false,
        dq:               dq ? { verdict: (dq as any).verdict || "", n_breaches: (dq as any).n_breaches || 0 } : null,
        decay:            decay ? { overall: (decay as any).overall || "", n_mechanisms: (decay as any).n_mechanisms || 0 } : null,
        activeSession:    (sess as any)?.active || null,
        approvedForward:  Array.isArray(fwd) ? (fwd as any[]).map((v) => ({
          hypothesis_id:     v.source_hypothesis_id,
          mechanism_family:  v.mechanism_family,
          mechanism_subtype: v.mechanism_subtype,
        })) : [],
        recentIntents:    Array.isArray(intents) ? (intents as any[]).map((i) => ({
          intent_id: i.intent_id, kind: i.kind, status: i.status, filed_ts: i.filed_ts,
        })) : [],
        recentEvents:     events.map((e: any) => ({
          event_id: e.event_id, event_type: e.event_type, verdict: e.verdict, ts: e.ts,
        })),
        recentLessons:    Array.isArray(lessons) ? (lessons as any[]).map((L) => ({
          lesson_id: L.lesson_id, verdict: L.verdict,
          mechanism_family: L.mechanism_family, created_ts: L.created_ts,
        })) : [],
      });
    };

    fetchAll();
    const id = setInterval(fetchAll, pollMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [family, pollMs]);

  return state;
}


// Helpers used by step engines


export function isBookHealthy(s: WizardState): boolean {
  if (s.loading) return false;
  if (s.dq && s.dq.verdict === "HALT") return false;
  if (s.decay && s.decay.overall === "ACTION") return false;
  return true;
}


export function familyApproved(s: WizardState, family: string): boolean {
  if (!family) return s.approvedForward.length > 0;
  return s.approvedForward.some(
    (v) => v.mechanism_family.toLowerCase() === family.toLowerCase()
  );
}


export function familyApprovedCount(s: WizardState, family: string): number {
  if (!family) return s.approvedForward.length;
  return s.approvedForward.filter(
    (v) => v.mechanism_family.toLowerCase() === family.toLowerCase()
  ).length;
}


/** Returns true if any pending intent of one of these kinds was filed
 *  in the last `minutes` (default 60). Used to detect "user has handed
 *  off recently". */
export function recentIntentOf(
  s: WizardState,
  kinds: string[],
  minutes: number = 60,
): boolean {
  const cutoffMs = Date.now() - minutes * 60_000;
  return s.recentIntents.some((i) => {
    if (!kinds.includes(i.kind)) return false;
    const t = Date.parse(i.filed_ts.endsWith("Z") ? i.filed_ts : i.filed_ts + "Z");
    return Number.isFinite(t) && t >= cutoffMs;
  });
}


/** Verdict events emitted to the current session in the last `minutes`. */
export function verdictEmittedRecently(s: WizardState, minutes: number = 240): boolean {
  const cutoffMs = Date.now() - minutes * 60_000;
  return s.recentEvents.some((e) => {
    if (e.event_type !== "factor_verdict_filed") return false;
    const t = Date.parse(e.ts.endsWith("Z") ? e.ts : e.ts + "Z");
    return Number.isFinite(t) && t >= cutoffMs;
  });
}


export function lessonForFamilyRecently(
  s: WizardState,
  family: string,
  minutes: number = 240,
): { lesson_id: string; verdict: string } | null {
  const cutoffMs = Date.now() - minutes * 60_000;
  const famLc = family.toLowerCase();
  for (const L of s.recentLessons) {
    const t = Date.parse(L.created_ts.endsWith("Z") ? L.created_ts : L.created_ts + "Z");
    if (!Number.isFinite(t) || t < cutoffMs) continue;
    if (!famLc || L.mechanism_family.toLowerCase() === famLc) {
      return { lesson_id: L.lesson_id, verdict: L.verdict };
    }
  }
  return null;
}
