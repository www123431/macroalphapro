"use client";

// PendingIntents — list of typed-intent records the user filed that
// Claude hasn't picked up yet. Mounted on /dashboard so the morning
// view answers "what handoffs am I still waiting for Claude on".
//
// Polls /api/intents/pending every 20s. Renders nothing if empty —
// silence when there's nothing waiting is more valuable than a
// permanent "0 pending" affordance.

import { useEffect, useState } from "react";
import Link from "next/link";
import { Inbox, Atom, Bug, BookOpen, FileSearch, ArrowRight } from "lucide-react";
import { API_BASE } from "@/lib/api";


type Intent = {
  intent_id:     string;
  kind:          string;
  subject_type:  string;
  subject_id:    string;
  filed_ts:      string;
  filed_by:      string;
  source_page:   string;
  payload:       Record<string, unknown>;
  status:        string;
};


const KIND_LABEL: Record<string, string> = {
  audit_subject:      "Audit",
  pipeline_test:      "Pipeline test",
  research_test:      "Test hypothesis",
  ingest_paper:       "Ingest paper",
  re_audit_decay:     "Re-audit decay",
  review_lesson:      "Review lesson",
  explore_hypothesis: "Explore",
  annotate_doctrine:  "Annotate doctrine",
};


const KIND_ICON: Record<string, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  audit_subject:      Bug,
  pipeline_test:      Atom,
  research_test:      Atom,
  ingest_paper:       BookOpen,
  re_audit_decay:     FileSearch,
};


function ageStr(iso: string): string {
  const ms = Date.now() - new Date(iso + (iso.endsWith("Z") ? "" : "Z")).getTime();
  if (!Number.isFinite(ms)) return "";
  const m = ms / 60_000;
  if (m < 1)  return "just now";
  if (m < 60) return `${Math.floor(m)}m ago`;
  if (m < 1440) return `${Math.floor(m / 60)}h ago`;
  return `${Math.floor(m / 1440)}d ago`;
}


export function PendingIntents() {
  const [items, setItems] = useState<Intent[]>([]);

  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/intents/pending?since_minutes=2880`,
                                { cache: "no-store" });
        if (!res.ok) return;
        const data = (await res.json()) as Intent[];
        if (!cancelled) setItems(data);
      } catch {}
    };
    fetchOnce();
    const id = setInterval(fetchOnce, 20_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (items.length === 0) {
    // Fresh users + steady state: keep a 1-line affordance so the
    // feature isn't invisible. Replaces the silent return that
    // confused first-time visitors per the user-walkthrough audit
    // (漏洞 1.2 2026-06-04).
    return (
      <div className="rounded border border-border/30 bg-panel2/20 px-3 py-1.5 text-[10.5px] text-muted/70 flex items-center gap-1.5">
        <Inbox className="h-3 w-3" strokeWidth={1.75} />
        <span>
          Pending intents queue is empty. Click any "Pipeline test" /
          "Audit session" / "Open research session →" CTA on
          /research/library/detail or /research/forward — Claude picks them up
          here.
        </span>
      </div>
    );
  }

  return (
    <div className="rounded border border-info/30 bg-info/[0.04] px-3 py-2 space-y-1.5">
      <div className="flex items-center gap-2 text-[11px] text-info">
        <Inbox className="h-3.5 w-3.5" strokeWidth={2} />
        <span className="font-semibold">
          {items.length} pending intent{items.length === 1 ? "" : "s"} — Claude can pick up
        </span>
        <span className="ml-auto text-[10px] text-muted/70 font-mono">
          GET /api/intents/pending
        </span>
      </div>

      <ul className="space-y-1">
        {items.slice(0, 6).map((it) => {
          const Icon = KIND_ICON[it.kind] ?? Inbox;
          return (
            <li key={it.intent_id}
                className="flex items-center gap-2 text-[11px] py-0.5">
              <Icon className="h-3 w-3 text-muted/70 shrink-0" strokeWidth={1.75} />
              <span className="text-foreground/85 font-mono">
                {KIND_LABEL[it.kind] ?? it.kind}
              </span>
              <span className="text-muted/80">·</span>
              <span className="font-mono text-muted">
                {it.subject_id}
              </span>
              <span className="ml-auto text-[10px] text-muted/60 tnum">
                {ageStr(it.filed_ts)}
              </span>
              {it.source_page && (
                <Link href={it.source_page}
                  className="text-[10px] text-muted/60 hover:text-accent inline-flex items-center gap-0.5"
                  title={`Open source page: ${it.source_page}`}>
                  <ArrowRight className="h-2.5 w-2.5" />
                </Link>
              )}
            </li>
          );
        })}
        {items.length > 6 && (
          <li className="text-[10px] text-muted/60 italic pl-5">
            + {items.length - 6} more
          </li>
        )}
      </ul>
    </div>
  );
}
