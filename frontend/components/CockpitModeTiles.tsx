"use client";

// CockpitModeTiles — 4 large tiles, one per Lab mode, that turn the
// Cockpit from a "wall of stuff" into a workspace landing surface.
//
// Doctrine (PR-4 of 2026-06-02 IA refactor):
//   The previous Cockpit duplicated every Lab sub-page's content (council
//   activity, library list, graveyard summary, outcomes table). It was a
//   wall: lots of data, no hierarchy, no answer to "what should I do
//   right now". The new Cockpit answers ONE question per mode:
//
//     OPERATE  → Is the system running? Are deployed sleeves healthy?
//     RESEARCH → Do I have new ideas to test? What did PFH suggest?
//     DECIDE   → What's waiting for me to approve? What's the council saying?
//     LEARN    → How accurate is the council? What died recently?
//
//   Each tile shows ≤3 metrics + 1-line health verdict + drill link. The
//   user spends ~15 seconds reading the 4 tiles, then either clicks into
//   a mode to actually work, or closes the page knowing nothing's wrong.
//
// Data: each tile fetches its own minimal slice. Parallel fetches; tiles
// render independently so a slow one doesn't block the rest.

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  Zap, Atom, Network, Sprout, CheckCircle2, AlertCircle,
  AlertTriangle, Activity, ArrowRight,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, Skeleton, cn } from "@/components/ui";


// ── Per-tile data fetchers ─────────────────────────────────────────


type ModeState = "ok" | "info" | "warn" | "danger" | "loading" | "error";

const STATE_BG: Record<ModeState, string> = {
  ok:      "border-ok/30 bg-ok/5",
  info:    "border-info/30 bg-info/5",
  warn:    "border-warn/30 bg-warn/5",
  danger:  "border-danger/30 bg-danger/5",
  loading: "border-muted/20 bg-muted/5",
  error:   "border-muted/20 bg-muted/5",
};
const STATE_TEXT: Record<ModeState, string> = {
  ok:      "text-ok",
  info:    "text-info",
  warn:    "text-warn",
  danger:  "text-danger",
  loading: "text-muted",
  error:   "text-muted",
};


// ── OPERATE tile ───────────────────────────────────────────────────


function OperateTile() {
  const [liveness, setLiveness] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.livenessStatus(14);
        if (!cancelled) setLiveness(d);
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      }
    };
    tick();
    const id = setInterval(tick, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (error) {
    return <ModeTile mode="operate" state="error" verdict="probe failed" />;
  }
  if (!liveness) {
    return <ModeTile mode="operate" state="loading" verdict="loading…" />;
  }

  const code = liveness?.summary?.verdict_code as string;
  const state: ModeState =
    code === "OK"              ? "ok" :
    code === "WARN_STATUS"     ? "warn" :
    code === "ALERT_NO_SHOW"   ? "danger" :
    "info";
  const verdict =
    code === "OK"              ? "Cron live · book trading" :
    code === "WARN_STATUS"     ? "Last run had issues" :
    code === "ALERT_NO_SHOW"   ? "Heartbeat MISSING" :
    code === "INFO_WEEKEND"    ? "Weekend — no run expected" :
    "Off-hours";

  const latest = liveness?.verdict?.latest;
  const recent = liveness?.recent || [];
  const halts = recent.filter((r: any) => r?.status?.startsWith?.("halt")
                                          || r?.status?.includes?.("partial")).length;

  return (
    <ModeTile
      mode="operate"
      state={state}
      verdict={verdict}
      stats={[
        { label: "orders",  value: latest?.n_orders ?? "—",
          sub: latest?.n_fills != null && latest?.n_orders != null
                ? `${latest.n_fills} filled` : undefined },
        { label: "equity",  value: latest?.equity_before != null
                  ? `$${Math.round(latest.equity_before / 1000)}k` : "—" },
        { label: "halts 14d", value: halts, tone: halts > 0 ? "warn" : "ok" },
      ]}
      links={[
        { label: "Liveness", href: "/ops/liveness" },
        { label: "Decay",    href: "/research/decay" },
      ]}
    />
  );
}


// ── RESEARCH tile ──────────────────────────────────────────────────


function ResearchTile() {
  const [n_mech, setNMech] = useState<number | null>(null);
  const [pfhHistory, setPfhHistory] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      api.researchCall<{ entries: any[] }>("query_library", {}, "ui_lab_cockpit"),
      api.factorLabPfhHistory(5),
    ]).then(([libR, pfhR]) => {
      if (cancelled) return;
      if (libR.status === "fulfilled") {
        setNMech(libR.value?.result?.entries?.length ?? 0);
      }
      if (pfhR.status === "fulfilled") {
        setPfhHistory(pfhR.value);
      }
      if (libR.status === "rejected" && pfhR.status === "rejected") {
        setError("data unavailable");
      }
    });
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return <ModeTile mode="research" state="error" verdict="data unavailable" />;
  }
  if (n_mech == null && !pfhHistory) {
    return <ModeTile mode="research" state="loading" verdict="loading…" />;
  }

  const lastRun = pfhHistory?.runs?.[0];
  const lastRunAge = lastRun?.ts
    ? Math.floor((Date.now() - new Date(lastRun.ts).getTime()) / (1000 * 3600 * 24))
    : null;
  const stale = lastRunAge != null && lastRunAge > 7;
  const state: ModeState = stale ? "warn" : "info";
  const verdict = lastRun
    ? `Last PFH ${lastRunAge != null ? `${lastRunAge}d ago` : "—"}${stale ? " · catalog growing stale" : ""}`
    : "No PFH runs yet";

  return (
    <ModeTile
      mode="research"
      state={state}
      verdict={verdict}
      stats={[
        { label: "mechanisms", value: n_mech ?? "—" },
        { label: "PFH runs",   value: pfhHistory?.runs?.length ?? "—" },
        { label: "last batch", value: lastRun?.n_candidates_total ?? "—" },
      ]}
      links={[
        { label: "Factor Lab", href: "/lab/factor-lab" },
        { label: "Library",    href: "/research/library" },
      ]}
    />
  );
}


// ── DECIDE tile ────────────────────────────────────────────────────


function DecideTile() {
  const [councilRuns, setCouncilRuns] = useState<any>(null);
  const [l4Iters, setL4Iters] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      api.councilRuns(20),
      api.l4Iterations(50),
    ]).then(([cR, l4R]) => {
      if (cancelled) return;
      if (cR.status === "fulfilled") setCouncilRuns(cR.value);
      if (l4R.status === "fulfilled") setL4Iters(l4R.value);
      if (cR.status === "rejected" && l4R.status === "rejected") {
        setError("data unavailable");
      }
    });
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return <ModeTile mode="decide" state="error" verdict="data unavailable" />;
  }
  if (!councilRuns && !l4Iters) {
    return <ModeTile mode="decide" state="loading" verdict="loading…" />;
  }

  // Filter test runs
  const TEST_RUN_TITLES = new Set([
    "Test proposal", "ledger_test", "smoke test", "override_t",
  ]);
  const realRuns: any[] = (councilRuns?.runs || []).filter(
    (r: any) => !Array.from(TEST_RUN_TITLES).some(t => r.proposal?.title?.includes?.(t))
  );
  const consensus = { APPROVE: 0, NEEDS_REVISION: 0, REJECT: 0 };
  for (const r of realRuns) {
    if (r.consensus in consensus) (consensus as any)[r.consensus]++;
  }
  const pending = (l4Iters?.iterations || []).filter(
    (i: any) => !i.human_override && i.effective_consensus === "NEEDS_REVISION"
  ).length;

  const state: ModeState = pending > 0 ? "warn" : realRuns.length > 0 ? "info" : "loading";
  const verdict = pending > 0
    ? `${pending} candidate${pending === 1 ? "" : "s"} awaiting human verdict`
    : realRuns.length > 0
      ? `${realRuns.length} council run${realRuns.length === 1 ? "" : "s"} on record`
      : "No real council runs yet";

  return (
    <ModeTile
      mode="decide"
      state={state}
      verdict={verdict}
      stats={[
        { label: "approve",   value: consensus.APPROVE, tone: "ok" },
        { label: "revise",    value: consensus.NEEDS_REVISION, tone: "warn" },
        { label: "reject",    value: consensus.REJECT, tone: "danger" },
      ]}
      links={[
        { label: "Council", href: "/lab/council" },
        { label: "L4 Runs", href: "/lab/l4" },
      ]}
    />
  );
}


// ── LEARN tile ─────────────────────────────────────────────────────


function LearnTile() {
  const [calibration, setCalibration] = useState<any>(null);
  const [graveyard, setGraveyard] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      api.criticCalibration(90),
      api.researchCall<{ n_total: number; recent_deaths: any[] }>(
        "graveyard_summary", { top_n_families: 10 }, "ui_lab_cockpit"),
    ]).then(([cR, gR]) => {
      if (cancelled) return;
      if (cR.status === "fulfilled") setCalibration(cR.value);
      if (gR.status === "fulfilled") setGraveyard(gR.value);
      if (cR.status === "rejected" && gR.status === "rejected") {
        setError("data unavailable");
      }
    });
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return <ModeTile mode="learn" state="error" verdict="data unavailable" />;
  }
  if (!calibration && !graveyard) {
    return <ModeTile mode="learn" state="loading" verdict="loading…" />;
  }

  const nDecided = calibration?.n_total_rows ?? 0;
  const nCritics = calibration?.n_distinct_critics ?? 0;
  const deaths7d = (graveyard?.recent_deaths || []).filter((d: any) => {
    if (!d.date) return false;
    return (Date.now() - new Date(d.date).getTime()) / (1000 * 3600 * 24) < 7;
  }).length;

  // Health verdict: insufficient data if calibration N small
  const state: ModeState = nDecided < 5
    ? "info"
    : deaths7d > 0 ? "warn" : "ok";
  const verdict = nDecided < 5
    ? "Calibration data still accumulating"
    : `Council calibration tracked across ${nCritics} critic${nCritics === 1 ? "" : "s"}`;

  return (
    <ModeTile
      mode="learn"
      state={state}
      verdict={verdict}
      stats={[
        { label: "graveyard", value: graveyard?.n_total ?? "—" },
        { label: "deaths 7d", value: deaths7d, tone: deaths7d > 0 ? "warn" : "ok" },
        { label: "calib N",   value: nDecided },
      ]}
      links={[
        { label: "Outcomes",  href: "/lab/outcomes" },
        { label: "Graveyard", href: "/research" },
      ]}
    />
  );
}


// ── Shared tile shell ──────────────────────────────────────────────


type StatCell = {
  label: string;
  value: React.ReactNode;
  sub?: string;
  tone?: "ok" | "warn" | "danger" | "info";
};

const MODE_META: Record<string, { label: string; icon: any; description: string }> = {
  operate: {
    label: "Operate", icon: Zap,
    description: "Trading + monitoring",
  },
  research: {
    label: "Research", icon: Atom,
    description: "Discovery + exploration",
  },
  decide: {
    label: "Decide", icon: Network,
    description: "Council + promotion",
  },
  learn: {
    label: "Learn", icon: Sprout,
    description: "Retrospective + calibration",
  },
};

function ModeTile({
  mode, state, verdict, stats, links,
}: {
  mode: string;
  state: ModeState;
  verdict: string;
  stats?: StatCell[];
  links?: { label: string; href: string }[];
}) {
  const meta = MODE_META[mode];
  const ModeIcon = meta.icon;
  const HealthIcon =
    state === "ok"      ? CheckCircle2 :
    state === "warn"    ? AlertTriangle :
    state === "danger"  ? AlertCircle :
    Activity;

  const TONE_TEXT: Record<string, string> = {
    ok: "text-ok", warn: "text-warn", danger: "text-danger", info: "text-info",
  };

  return (
    <Card className={cn("border", STATE_BG[state], "flex flex-col")}>
      {/* Header: mode label + state icon */}
      <div className="flex items-start justify-between gap-2 pb-2 border-b border-border/30">
        <div className="space-y-0.5">
          <div className="inline-flex items-center gap-1.5">
            <ModeIcon className={cn("h-3.5 w-3.5", STATE_TEXT[state])} strokeWidth={2} />
            <span className="text-sm font-semibold uppercase tracking-wider">
              {meta.label}
            </span>
          </div>
          <div className="text-[10px] text-muted/70">{meta.description}</div>
        </div>
        <HealthIcon
          className={cn("h-4 w-4 shrink-0", STATE_TEXT[state])}
          strokeWidth={2}
        />
      </div>

      {/* Verdict (1-line state assessment) */}
      <div className={cn(
        "text-xs leading-snug pt-2.5 pb-3 min-h-[2.5rem]",
        STATE_TEXT[state] === "text-muted" ? "text-foreground" : STATE_TEXT[state],
      )}>
        {verdict}
      </div>

      {/* Stats grid (3 cells) */}
      {stats && stats.length > 0 && (
        <div className="grid grid-cols-3 gap-2 pb-3">
          {stats.map((s, i) => (
            <div key={i}>
              <div className={cn(
                "text-base font-mono tnum font-semibold tabular-nums",
                s.tone ? TONE_TEXT[s.tone] : "text-foreground",
              )}>
                {s.value}
              </div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70">
                {s.label}
              </div>
              {s.sub && (
                <div className="text-[9px] text-muted/60 tnum">{s.sub}</div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Drill links — always pinned to bottom */}
      {links && links.length > 0 && (
        <div className="mt-auto pt-2 border-t border-border/30 flex items-center gap-3">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="inline-flex items-center gap-1 text-[11px] text-accent hover:underline">
              {l.label}
              <ArrowRight className="h-3 w-3" />
            </Link>
          ))}
        </div>
      )}
    </Card>
  );
}


// ── Public component ───────────────────────────────────────────────


export function CockpitModeTiles() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
      <OperateTile />
      <ResearchTile />
      <DecideTile />
      <LearnTile />
    </div>
  );
}
