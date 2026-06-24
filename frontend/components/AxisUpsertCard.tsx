"use client";

// AxisUpsertCard — inline form for creating / updating a research axis.
//
// Gap A 2026-06-03. Used by /research/roadmap to add an axis without dropping
// to a doctrine session. (Doctrine session is recommended for stable axes
// — see CLAUDE.md "Session Protocol Doctrine" — but inline edit covers
// quick adjustments.)

import { useState } from "react";
import { AlertTriangle, ChevronRight, Loader2 } from "lucide-react";
import { useUpsertAxis } from "@/lib/queries";
import type {
  AxisState, AxisTier, AxisOutcome, ResearchAxisUpsertInput,
} from "@/lib/api";
import { Card, cn } from "@/components/ui";


const STATES: AxisState[] = ["active", "queued", "paused", "closed"];
const TIERS: AxisTier[]   = ["committed", "candidate", "scratchpad"];
const OUTCOMES: AxisOutcome[] = ["NONE", "GREEN", "RED", "MARGINAL", "ABANDONED"];


export function AxisUpsertCard({ onDone, initial }: {
  onDone: () => void;
  initial?: Partial<ResearchAxisUpsertInput>;
}) {
  const mut = useUpsertAxis();
  const [axisId, setAxisId]   = useState(initial?.axis_id   ?? "");
  const [name, setName]       = useState(initial?.name      ?? "");
  const [state, setState]     = useState<AxisState>(initial?.state ?? "queued");
  const [tier, setTier]       = useState<AxisTier>(initial?.tier ?? "candidate");
  const [outcome, setOutcome] = useState<AxisOutcome>((initial?.outcome ?? "NONE") as AxisOutcome);
  const [family, setFamily]   = useState(initial?.family ?? "");
  const [rationale, setRationale] = useState(initial?.rationale ?? "");
  const [nextActionsRaw, setNextActionsRaw] = useState(
    (initial?.next_actions ?? []).join("\n"),
  );
  const [blockingNotes, setBlockingNotes] = useState(initial?.blocking_notes ?? "");
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setError(null);
    if (!axisId.trim()) { setError("axis_id is required"); return; }
    if (!name.trim())   { setError("name is required"); return; }
    if (rationale.trim().length < 10) {
      setError("rationale must be ≥10 chars");
      return;
    }
    try {
      await mut.mutateAsync({
        axis_id: axisId.trim(),
        name: name.trim(),
        state, tier, outcome,
        family: family.trim() || null,
        rationale: rationale.trim(),
        next_actions: nextActionsRaw.split("\n").map((l) => l.trim()).filter(Boolean),
        blocking_notes: blockingNotes.trim(),
      });
      onDone();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  };

  return (
    <Card className="border-accent/40">
      <div className="space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <Field label="axis_id (short stable slug)" required>
            <input value={axisId} onChange={(e) => setAxisId(e.target.value)}
              placeholder="e.g. carry_gx_stir"
              className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs font-mono outline-none focus:border-accent/60" />
          </Field>
          <Field label="name (human label)" required>
            <input value={name} onChange={(e) => setName(e.target.value)}
              placeholder="e.g. G10 STIR carry expansion"
              className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs outline-none focus:border-accent/60" />
          </Field>
        </div>

        <div className="grid grid-cols-3 gap-3">
          <Field label="state">
            <Select value={state} onChange={(v) => setState(v as AxisState)} options={STATES} />
          </Field>
          <Field label="tier">
            <Select value={tier} onChange={(v) => setTier(v as AxisTier)} options={TIERS} />
          </Field>
          <Field label="outcome (if closed)">
            <Select value={outcome} onChange={(v) => setOutcome(v as AxisOutcome)} options={OUTCOMES} />
          </Field>
        </div>

        <Field label="factor family (auto-attaches decay forecast)">
          <input value={family} onChange={(e) => setFamily(e.target.value)}
            placeholder="e.g. carry / earnings_underreaction / momentum"
            className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs font-mono outline-none focus:border-accent/60" />
        </Field>

        <Field label="rationale (≥10 chars)" required>
          <textarea value={rationale} onChange={(e) => setRationale(e.target.value)}
            rows={3}
            placeholder="Why is this axis active / queued / closed?"
            className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs outline-none focus:border-accent/60 resize-none" />
        </Field>

        <Field label="next actions (one per line)">
          <textarea value={nextActionsRaw} onChange={(e) => setNextActionsRaw(e.target.value)}
            rows={3}
            placeholder={"Survey WRDS for STIR clscodes\nPre-register strict gate\nFetch + cache 3-month contracts"}
            className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs outline-none focus:border-accent/60 resize-none" />
        </Field>

        <Field label="blocking notes (optional)">
          <input value={blockingNotes} onChange={(e) => setBlockingNotes(e.target.value)}
            placeholder="What's preventing progress?"
            className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs outline-none focus:border-accent/60" />
        </Field>

        {error && (
          <div className="rounded-md border border-alert/40 bg-alert/10 p-2 text-xs inline-flex items-start gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 text-alert mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <div className="flex items-center justify-end gap-2 pt-1">
          <button onClick={onDone} disabled={mut.isPending}
            className="text-xs text-muted hover:text-foreground disabled:opacity-40">
            Cancel
          </button>
          <button onClick={submit} disabled={mut.isPending}
            className="inline-flex items-center gap-1 rounded-md border border-accent/40 bg-accent/15 px-3 py-1.5 text-xs text-accent hover:bg-accent/25 disabled:opacity-40">
            {mut.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
            Save axis
            <ChevronRight className="h-3 w-3" />
          </button>
        </div>
      </div>
    </Card>
  );
}


function Field({ label, required, children }: {
  label: string; required?: boolean; children: React.ReactNode;
}) {
  return (
    <div className="space-y-0.5">
      <label className="text-[10px] uppercase tracking-wider text-muted">
        {label} {required && <span className="text-warn/80 normal-case">(required)</span>}
      </label>
      {children}
    </div>
  );
}


function Select({ value, onChange, options }: {
  value: string; onChange: (v: string) => void; options: string[];
}) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-xs cursor-pointer outline-none focus:border-accent/60">
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  );
}
