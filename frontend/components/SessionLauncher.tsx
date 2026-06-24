"use client";

// SessionLauncher — primary entry point for opening a typed user session.
//
// P7 2026-06-03 of the session protocol build (CLAUDE.md "Session Protocol
// Doctrine"). 5 typed buttons (research_new / audit / ops / doctrine /
// exploration), each opens a per-type pre-flight wizard.
//
// Design (audit-derived):
//   - typed buttons NOT a dropdown (Linear / Notion / Figma pattern)
//   - one-line hover hint per type so user can decide without clicking
//   - wizard fills the controlled PreflightDigest
//   - server-side preflight checker decides what's required; UI just
//     shows whatever fields the backend says are missing on 409
//   - after preflight passes, session transitions to in_flight + UI
//     navigates to /research/sessions?focus=<id> for ongoing tracking
//
// Mounted on /research/sessions page (P8) as the launch zone.

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Atom, Bug, Activity, BookOpen, Lightbulb,
  X, AlertTriangle, ChevronRight, Loader2,
} from "lucide-react";
import {
  useOpenSession, useRecordPreflight, useSessionTypes, useRoadmapAxis,
} from "@/lib/queries";
import type { SessionType, PreflightDigestInput } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { AlphaMortalityBadge } from "@/components/AlphaMortalityBadge";
import { CapacityBadge } from "@/components/CapacityBadge";


const TYPE_META: Record<SessionType, {
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  tone: string;
  label: string;
}> = {
  research_new: { icon: Atom,      tone: "border-accent/40 hover:bg-accent/10 text-accent",
                  label: "Research" },
  audit:        { icon: Bug,       tone: "border-warn/40 hover:bg-warn/10 text-warn",
                  label: "Audit" },
  ops:          { icon: Activity,  tone: "border-info/40 hover:bg-info/10 text-info",
                  label: "Ops" },
  doctrine:     { icon: BookOpen,  tone: "border-ok/40 hover:bg-ok/10 text-ok",
                  label: "Doctrine" },
  exploration:  { icon: Lightbulb, tone: "border-muted/40 hover:bg-muted/10 text-muted",
                  label: "Exploration" },
};


export function SessionLauncher({ prefillAxisId, prefillType }: {
  prefillAxisId?: string;
  prefillType?: SessionType;
} = {}) {
  const typesQ = useSessionTypes();
  const [picked, setPicked] = useState<SessionType | null>(null);

  // 2026-06-03: when ?axis_id=X (or ?type=X) is in the URL, auto-open
  // the wizard pre-filled from that axis. Lets /dashboard and
  // /research/roadmap link straight into the wizard with one click.
  const axisQ = useRoadmapAxis(prefillAxisId);
  useEffect(() => {
    if (picked) return;   // user already picked one; don't auto-redirect
    if (prefillType) {
      setPicked(prefillType);
      return;
    }
    if (prefillAxisId && axisQ.data) {
      // axis exists → default to research_new (axes are research targets)
      setPicked("research_new");
    }
  }, [prefillType, prefillAxisId, axisQ.data, picked]);

  const types = typesQ.data?.types ?? [];

  return (
    <>
      <Card>
        <div className="flex items-baseline justify-between mb-2">
          <div className="space-y-0.5">
            <h2 className="text-sm font-semibold">Start a session</h2>
            <p className="text-[11px] text-muted">
              Typed workflow per CLAUDE.md doctrine. Each type has a
              pre-flight wizard + exit conditions tracked at close.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-2 mt-3">
          {(["research_new", "audit", "ops", "doctrine", "exploration"] as SessionType[]).map((t) => {
            const meta = TYPE_META[t];
            const Icon = meta.icon;
            const typeInfo = types.find((x) => x.session_type === t);
            return (
              <button key={t}
                onClick={() => setPicked(t)}
                title={typeInfo?.description}
                className={cn(
                  "rounded-md border px-3 py-2 text-left transition-colors",
                  meta.tone,
                  "bg-panel/40",
                )}>
                <div className="flex items-center gap-1.5 mb-0.5">
                  <Icon className="h-3.5 w-3.5" strokeWidth={2} />
                  <span className="text-xs font-semibold">{meta.label}</span>
                </div>
                <div className="text-[10px] text-muted leading-snug line-clamp-2">
                  {typeInfo?.description ?? "—"}
                </div>
                <div className="text-[9px] text-muted/60 mt-1 font-mono">
                  ~{typeInfo?.expected_duration ?? "?"}
                </div>
              </button>
            );
          })}
        </div>
      </Card>

      {picked && (
        <PreflightWizard
          sessionType={picked}
          prefillAxis={axisQ.data}
          onClose={() => setPicked(null)}
        />
      )}
    </>
  );
}


function PreflightWizard({ sessionType, prefillAxis, onClose }: {
  sessionType: SessionType;
  prefillAxis?: any;
  onClose: () => void;
}) {
  const router = useRouter();
  const openMut = useOpenSession();
  const preflightMut = useRecordPreflight();

  // Build initial digest — empty if no prefill, else seeded from the axis.
  // Goal is concatenated rationale + first next_action for a sane wizard
  // starting point that the user can refine.
  const seedDigest = (): PreflightDigestInput => {
    if (prefillAxis) {
      const goalParts = [
        prefillAxis.rationale,
        prefillAxis.next_actions?.[0] ? `Starting with: ${prefillAxis.next_actions[0]}` : null,
      ].filter(Boolean);
      return {
        cockpit_reviewed:        false,
        decay_alerts_count:      0,
        dq_breaches_count:       0,
        graveyard_search_query:  prefillAxis.name || "",
        graveyard_hits_count:    0,
        library_overlap_checked: false,
        goal:                    goalParts.join(" — ").slice(0, 800),
        notes:                   `From roadmap axis: ${prefillAxis.axis_id}`,
      };
    }
    return {
      cockpit_reviewed:        false,
      decay_alerts_count:      0,
      dq_breaches_count:       0,
      graveyard_search_query:  "",
      graveyard_hits_count:    0,
      library_overlap_checked: false,
      goal:                    "",
      notes:                   "",
    };
  };

  const [title, setTitle] = useState(prefillAxis?.name ?? "");
  const [digest, setDigest] = useState<PreflightDigestInput>(seedDigest);
  const [error, setError] = useState<string | null>(null);
  const [missingFields, setMissingFields] = useState<string[]>([]);
  // Gap B 2026-06-03: research_new sessions need to surface alpha
  // mortality before committing. Pre-filled from axis.family when
  // launched via roadmap → session prefill.
  const [family, setFamily] = useState(prefillAxis?.family ?? "");
  const meta = TYPE_META[sessionType];
  const TypeIcon = meta.icon;

  const isBusy = openMut.isPending || preflightMut.isPending;

  const submit = async () => {
    setError(null);
    setMissingFields([]);
    try {
      const session = await openMut.mutateAsync({
        session_type: sessionType,
        title: title.trim() || meta.label,
      });
      try {
        await preflightMut.mutateAsync({ sessionId: session.session_id, digest });
        // Success → navigate to session detail
        router.push(`/research/sessions?focus=${encodeURIComponent(session.session_id)}`);
        onClose();
      } catch (e: any) {
        // 409 with missing list — re-render wizard highlighting missing
        const msg = String(e?.message ?? e);
        const detailMatch = msg.match(/preflight_incomplete/);
        if (detailMatch) {
          // Parse missing field names from server message
          const lines = msg.split("\n").filter((l) => l.includes("("));
          setMissingFields(lines.map((l) => l.trim()));
        } else {
          setError(msg);
        }
      }
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm"
         onClick={onClose}>
      <div className="w-full max-w-xl rounded-lg border border-border/70 bg-panel/95 backdrop-blur-md shadow-2xl"
           onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className={cn("flex items-center justify-between border-b border-border/40 px-4 py-2.5",
                            meta.tone.replace("hover:", ""))}>
          <span className="inline-flex items-center gap-2 font-semibold text-sm">
            <TypeIcon className="h-4 w-4" strokeWidth={2} />
            New {meta.label} session — pre-flight
          </span>
          <button onClick={onClose} aria-label="close">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4 max-h-[80vh] overflow-y-auto">
          {/* Title */}
          <div className="space-y-1">
            <label className="text-[10px] uppercase tracking-wider text-muted">Session title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Short label (e.g. 'EM FX 12m momentum')"
              className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-sm outline-none focus:border-accent/60"
            />
          </div>

          {/* Goal */}
          <div className="space-y-1">
            <label className="text-[10px] uppercase tracking-wider text-muted">
              Goal <span className="text-warn/80 normal-case">(required, ≥10 chars min — 30+ for research/audit/doctrine)</span>
            </label>
            <textarea
              value={digest.goal}
              onChange={(e) => setDigest({ ...digest, goal: e.target.value })}
              placeholder={
                sessionType === "research_new"
                  ? "What factor / mechanism are you testing? What's the hypothesis?"
                  : sessionType === "audit"
                  ? "What looks wrong? What do you suspect?"
                  : sessionType === "ops"
                  ? "What are you monitoring or responding to?"
                  : sessionType === "doctrine"
                  ? "What lesson / doctrine are you locking?"
                  : "What do you want to think about?"
              }
              rows={3}
              className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-sm outline-none focus:border-accent/60 resize-none"
            />
            <div className="text-[10px] text-muted/60 tnum">
              {digest.goal.length} chars
            </div>
          </div>

          {/* Alpha mortality badge — research_new only.
              Gap B 2026-06-03: surface forward-decay family-typical
              before committing time to strict gate. */}
          {sessionType === "research_new" && (
            <div className="space-y-2">
              <label className="text-[10px] uppercase tracking-wider text-muted">
                Factor family <span className="text-warn/80 normal-case">
                  (for forward-decay forecast — see /research/decay for the registry)
                </span>
              </label>
              <input
                value={family}
                onChange={(e) => setFamily(e.target.value)}
                placeholder="e.g. earnings_underreaction · momentum · carry · low_vol · tsmom"
                className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs font-mono outline-none focus:border-accent/60"
              />
              <AlphaMortalityBadge family={family} />
              <CapacityBadge family={family} />
            </div>
          )}

          {/* Pre-flight checkboxes — heavier types need these */}
          {(sessionType === "research_new" || sessionType === "audit") && (
            <div className="space-y-2 rounded-md border border-border/30 bg-panel2/30 p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted/80">
                UI pre-flight (required for research / audit)
              </div>
              <label className="flex items-center gap-2 text-xs cursor-pointer">
                <input
                  type="checkbox"
                  checked={digest.cockpit_reviewed}
                  onChange={(e) => setDigest({ ...digest, cockpit_reviewed: e.target.checked })}
                  className="h-3 w-3"
                />
                I reviewed Cockpit (deploy state / decay / DQ)
              </label>

              {sessionType === "research_new" && (
                <>
                  <label className="flex items-center gap-2 text-xs cursor-pointer">
                    <input
                      type="checkbox"
                      checked={digest.library_overlap_checked}
                      onChange={(e) => setDigest({ ...digest, library_overlap_checked: e.target.checked })}
                      className="h-3 w-3"
                    />
                    I checked /research/library for sleeve overlap
                  </label>
                  <div className="space-y-1">
                    <label className="text-[10px] uppercase tracking-wider text-muted">
                      Graveyard search query <span className="text-warn/80 normal-case">(what you searched in /research)</span>
                    </label>
                    <input
                      value={digest.graveyard_search_query}
                      onChange={(e) => setDigest({ ...digest, graveyard_search_query: e.target.value })}
                      placeholder="e.g. 'EM FX momentum'"
                      className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs"
                    />
                  </div>
                </>
              )}
            </div>
          )}

          {/* Missing fields (server 409) */}
          {missingFields.length > 0 && (
            <div className="rounded-md border border-warn/40 bg-warn/10 p-3 text-xs">
              <div className="font-semibold text-warn mb-1 inline-flex items-center gap-1">
                <AlertTriangle className="h-3.5 w-3.5" /> Pre-flight incomplete
              </div>
              <ul className="space-y-0.5 text-[11px] text-foreground/85 ml-4 list-disc">
                {missingFields.map((m) => <li key={m}>{m}</li>)}
              </ul>
            </div>
          )}

          {/* Generic error */}
          {error && (
            <div className="rounded-md border border-alert/40 bg-alert/10 p-3 text-xs">
              <div className="font-semibold text-alert mb-1 inline-flex items-center gap-1">
                <AlertTriangle className="h-3.5 w-3.5" /> Error
              </div>
              <pre className="whitespace-pre-wrap text-[11px]">{error}</pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border/40 px-4 py-2.5 flex items-center justify-between">
          <button onClick={onClose}
            disabled={isBusy}
            className="text-xs text-muted hover:text-foreground disabled:opacity-40">
            Cancel
          </button>
          <button onClick={submit}
            disabled={isBusy || !digest.goal.trim()}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-4 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40",
              meta.tone,
            )}>
            {isBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
            Open session
            <ChevronRight className="h-3 w-3" strokeWidth={2.5} />
          </button>
        </div>
      </div>
    </div>
  );
}
