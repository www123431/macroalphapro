"use client";

// Lab workspace left rail — 4-mode IA (R3 of 2026-06-04 quant-lens
// rebuild).
//
// Doctrine (2026-06-02 amendment, fully implemented here):
//   rail grouping = USER MODE (OPERATE / RESEARCH / GOVERN / LEARN)
//   not lifecycle stage. Each mode maps to a question the quant
//   actually asks during the day, with 2-3 primary surfaces per mode
//   and tools demoted to Cmd-K + a separate "Tools" drawer.
//
//   OPERATE  — "Is my book OK right now?"
//   RESEARCH — "What should I test next?"
//   GOVERN   — "Is the strategy lifecycle clean?"
//   LEARN    — "What have we learned (verdicts + doctrine)?"
//
// 2026-06-04 rebuild:
//   - Promoted 4 modes from "one OPERATE group" to the full lattice.
//   - 17-surface Lab compressed to 12 primary items + 6 tools.
//   - Tools (factor-lab / axes / series / chains / cosine / outcomes)
//     are Cmd-K-only deep-dive builders, not landing surfaces.
//   - Each mode header lights up when the user is inside it; the
//     active label doubles as the page's mode header.

import { Suspense } from "react";
import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import {
  // Mode icons
  Activity, Sparkles, Shield, GraduationCap,
  // Item icons
  Compass, ListChecks, Layers, Heart, ScrollText, BookOpen, Book,
  Skull, FileBarChart, Command, Inbox, ShieldCheck,
} from "lucide-react";
// (useState dropped 2026-06-04 R5 with the Tools drawer)
import { cn } from "@/components/ui";
import { useI18n } from "@/lib/i18n";
import {
  useV2Approvals, useStrengthenerApprovals, useFactorSpecApprovals,
  useDecayReport, useActiveSession,
} from "@/lib/queries";


// ── Types ─────────────────────────────────────────────────────────


type RailItem = {
  label:    string;
  href:     string;
  icon:     React.ComponentType<{ className?: string; strokeWidth?: number }>;
  ready:    boolean;
  hint?:    string;
  badge?:   string;
  // Telemetry-evidenced demotion (2026-06-23 Phase D): items that
  // received 0 visits in the last 11 days of telemetry get demoted
  // to a "More" disclosure at the bottom of their mode. Route stays
  // alive (direct URL still works, Cmd-K still finds it), but rail
  // prominence is removed. Reversed automatically: bumping visits
  // for any demoted item is reason to flip this flag off.
  demoted?: boolean;
  // i18n keys (2026-06-06). When present, t(labelKey) / t(hintKey)
  // override the static label/hint. Set only on items we've translated;
  // unset items fall through to the English-only label/hint.
  labelKey?: string;
  hintKey?:  string;
};

type RailGroup = {
  key:      "operate" | "research" | "govern" | "learn";
  label:    string;
  modeIcon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  question: string;
  items:    RailItem[];
};


// ── 4-mode lattice ─────────────────────────────────────────────────


// OPERATE — "Is my book OK right now?" Daily morning landing.
// 2026-06-14 (Phase 2): /dashboard merged into /dashboard. The single
// morning landing is now /dashboard.
const OPERATE_ITEMS: RailItem[] = [
  { label: "Today",     href: "/dashboard",     icon: Activity,    ready: true,
    hint: "Daily book status — operator widgets + decay sentinel + safety rails" },
  { label: "Sessions",  href: "/research/sessions",  icon: ListChecks,  ready: true,
    hint: "Active + queued research sessions" },
  { label: "Liveness",  href: "/ops/liveness",  icon: Heart,       ready: true,
    hint: "Heartbeat — cron schedule + last-fire timestamps" },
];

// RESEARCH — "What should I test next?" The paper-driven chain.
const RESEARCH_ITEMS: RailItem[] = [
  { label: "Console", href: "/research/console", icon: Command, ready: true,
    hint: "Operator Console — UI-triggered Pipeline Stations gated by typed sessions. Foundation shipped; stations attach progressively (S1 paper ingest, S4 FORWARD dispatch, S6 verdict view first)." },
  { label: "Incoming papers", href: "/research/papers/incoming", icon: Inbox, ready: true,
    hint: "Employee A daily digest — auto-discovered arxiv q-fin candidates with Deepseek triage + summary",
    labelKey: "rail.research.incoming", hintKey: "rail.research.incoming.hint" },
  { label: "Enhance the book", href: "/research/enhance",  icon: Sparkles,   ready: true,
    hint: "Guided wizard — pick a recipe (carry timing, re-audit, replace decayed) and step through",
    labelKey: "rail.research.enhance", hintKey: "rail.research.enhance.hint" },
  { label: "Forward vectors", href: "/research/forward",   icon: Sparkles,   ready: true,
    hint: "Untested hypotheses extracted from papers — pick the next test",
    labelKey: "rail.research.forward", hintKey: "rail.research.forward.hint" },
  { label: "Brainstorm",      href: "/research/brainstorm", icon: Sparkles,  ready: true,
    demoted: true,    // 0 visits in 11d telemetry — moved to "More"
    hint: "Experience-conditioned divergent generator — cross-domain seed packs + lessons distilled from our verdict history" },
  { label: "Brainstorm metrics", href: "/research/brainstorm_metrics", icon: Sparkles, ready: true,
    demoted: true,    // 0 visits — moved to "More"
    hint: "Per-pack funnel + LLM calibration + RED failure modes — measurement substrate" },
  { label: "RED outcomes",    href: "/research/forward/red", icon: Skull, ready: true,
    demoted: true,    // 0 visits — moved to "More"; reachable via Forward
    hint: "Directions already ruled out by strict gate — what NOT to propose again",
    labelKey: "rail.research.red_outcomes", hintKey: "rail.research.red_outcomes.hint" },
  { label: "Anchor library",  href: "/research/forward/anchors", icon: Book, ready: true,
    demoted: true,    // 0 visits — moved to "More"; reachable via Forward
    hint: "Canonical T1+T2 papers A cites for orthogonality — methodology gates + mechanism anchors",
    labelKey: "rail.research.anchors", hintKey: "rail.research.anchors.hint" },
  { label: "Reading queue",   href: "/research/reading",     icon: ScrollText, ready: true,
    demoted: true,    // 0 visits — moved to "More"
    hint: "T7 paper registry projected as a relevance-scored reading list",
    labelKey: "rail.research.reading_queue", hintKey: "rail.research.reading_queue.hint" },
  { label: "Roadmap",         href: "/research/roadmap",        icon: Compass,    ready: true,
    demoted: true,    // 0 visits in /research/roadmap (but /lab/roadmap had 12 — should
                       // recover after redirect propagation)
    hint: "Long-term research axes — cross-asset / TSMOM / carry",
    labelKey: "rail.research.roadmap", hintKey: "rail.research.roadmap.hint" },
  { label: "Paper library",   href: "/research/papers",    icon: BookOpen,   ready: true,
    hint: "Full registry with hypothesis trace + verbatim quotes",
    labelKey: "rail.research.paper_library", hintKey: "rail.research.paper_library.hint" },
];

// GOVERN — "Is the strategy lifecycle clean?" 2 primary surfaces
// (R5 2026-06-04 second-trim): Council folded into Library detail as
// a per-sleeve tab; L4 demoted to a status pill on /dashboard. Both
// keep their URLs (no 404 on outside links) but neither is in the rail.
const GOVERN_ITEMS: RailItem[] = [
  { label: "Approvals",     href: "/approvals",         icon: ShieldCheck,  ready: true,
    hint: "Model Change Control — B's APPROVE/AMENDMENT + governance + tactical queues",
    labelKey: "rail.govern.approvals", hintKey: "rail.govern.approvals.hint" },
  { label: "Library",       href: "/research/library",       icon: Layers,       ready: true,
    hint: "Deployed mechanisms with KPIs + lifecycle state + Council tab" },
  { label: "Decay",         href: "/research/decay",         icon: FileBarChart, ready: true,
    hint: "Alpha decay sentinel — sleeve degradation watch" },
];

// LEARN — "What have we learned?" Retrospective verdicts + doctrine.
const LEARN_ITEMS: RailItem[] = [
  { label: "Workflow",    href: "/research/workflow", icon: Layers, ready: true,
    hint: "End-to-end pipeline trace: papers → synthesis → hypotheses → specs → predict → verdict → autopsy → belief. Single-picture system map." },
  { label: "Calibration", href: "/research/calibration", icon: FileBarChart, ready: true,
    hint: "Belief layer headline numbers + honest negative finding (predictor vs family prior). Refreshed daily." },
  { label: "RED Lessons", href: "/research/lessons", icon: Book,    ready: true,
    hint: "Verdicts — paper-grounded chain + 47 legacy" },
  { label: "Graveyard",   href: "/research/lessons?verdict=red&include_legacy=true",
    icon: Skull, ready: true,
    hint: "All RED verdicts — quick filter for the graveyard view" },
];

// (Tools drawer entirely removed 2026-06-04 R5. Factor Lab / Axes /
// Series / Chains / Cosine heatmap / Outcomes are Cmd-K-only — quants
// hit them monthly at most, sidebar real-estate is too expensive.)


const GROUPS: RailGroup[] = [
  { key: "operate",  label: "Operate",  modeIcon: Activity,      question: "Is my book OK?",          items: OPERATE_ITEMS },
  { key: "research", label: "Research", modeIcon: Sparkles,      question: "What to test next?",       items: RESEARCH_ITEMS },
  { key: "govern",   label: "Govern",   modeIcon: Shield,        question: "Lifecycle clean?",         items: GOVERN_ITEMS },
  { key: "learn",    label: "Learn",    modeIcon: GraduationCap, question: "What have we learned?",    items: LEARN_ITEMS },
];


// ── Active-mode resolution ─────────────────────────────────────────


function isActive(
  href: string,
  pathname: string,
  currentSearch: URLSearchParams,
  itemsWithQueryOnSamePath: Set<string>,
): boolean {
  const [cleanHref, hrefQuery = ""] = href.split("?");

  // Path-level match first.
  let pathMatches = false;
  if (cleanHref === pathname) pathMatches = true;
  else if (cleanHref === "/") pathMatches = pathname === "/";
  else if (pathname.startsWith(cleanHref + "/")) pathMatches = true;

  if (!pathMatches) return false;

  // If the rail item itself has a query string, demand the user's
  // current URL carries every key=value the link specifies. This is
  // what separates "Graveyard (verdict=red&include_legacy=true)" from
  // plain "RED Lessons" — both point at /research/lessons.
  if (hrefQuery) {
    const linkParams = new URLSearchParams(hrefQuery);
    for (const [k, v] of linkParams) {
      if (currentSearch.get(k) !== v) return false;
    }
    return true;
  }

  // Conversely, if THIS link has no query but another rail item shares
  // the same path with a query, suppress this one whenever the current
  // URL carries that other item's query — otherwise both light up.
  if (itemsWithQueryOnSamePath.has(cleanHref)) {
    // Light up plain link only when no relevant query param is set.
    // Heuristic: if any well-known filter key is present, prefer the
    // query-bearing sibling.
    const filterKeys = ["verdict", "include_legacy", "grounding_method", "mechanism_family"];
    if (filterKeys.some((k) => currentSearch.has(k))) return false;
  }

  return true;
}

// Precompute the set of paths that have AT LEAST ONE rail item with
// a query string — used by isActive to know when a plain link should
// defer to a query-bearing sibling.
const PATHS_WITH_QUERY_VARIANT: Set<string> = (() => {
  const s = new Set<string>();
  const allItems = [...OPERATE_ITEMS, ...RESEARCH_ITEMS, ...GOVERN_ITEMS,
                    ...LEARN_ITEMS];
  for (const it of allItems) {
    const [path, query] = it.href.split("?");
    if (query) s.add(path);
  }
  return s;
})();


function activeGroupKey(pathname: string): RailGroup["key"] | null {
  // Explicit overrides for routes that span modes ambiguously.
  if (pathname.startsWith("/research/candidate")) return "research";
  if (pathname.startsWith("/research/papers"))    return "research";
  if (pathname.startsWith("/research/forward"))   return "research";
  if (pathname.startsWith("/research/brainstorm")) return "research";
  if (pathname.startsWith("/research/brainstorm_metrics")) return "research";
  if (pathname.startsWith("/research/lessons"))   return "learn";
  if (pathname.startsWith("/research/legacy"))    return "learn";
  if (pathname.startsWith("/research/reading"))     return "research";
  if (pathname.startsWith("/dashboard"))          return "operate";
  if (pathname.startsWith("/research/sessions"))       return "operate";
  if (pathname.startsWith("/ops/liveness"))       return "operate";
  if (pathname.startsWith("/research/library"))        return "govern";
  if (pathname.startsWith("/research/decay"))          return "govern";
  if (pathname.startsWith("/research/roadmap"))        return "research";
  return null;
}


// ── Component ─────────────────────────────────────────────────────


// useSearchParams() forces CSR-bailout under `output: export`, so the
// inner body must live inside a Suspense boundary. The outer wrapper
// is a pure server-renderable shell that delegates to <LabSideRailBody />.
export function LabSideRail() {
  return (
    <Suspense fallback={<aside className="hidden lg:block w-[212px] shrink-0" />}>
      <LabSideRailBody />
    </Suspense>
  );
}

function LabSideRailBody() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const activeKey = activeGroupKey(pathname);
  const { t } = useI18n();
  // Per-item localized label/hint. Items WITHOUT labelKey fall back to
  // the static English string (gradually translated as keys are added).
  const lbl = (it: RailItem) => it.labelKey ? t(it.labelKey) : it.label;
  const hnt = (it: RailItem) => it.hintKey ? t(it.hintKey) : it.hint;

  // Phase 4 (2026-06-14): live count badges so the rail surfaces
  // queue depth at a glance (Linear / Notion pattern). Badges are
  // resolved per item href via a small map; if no badge, none renders.
  const v2     = useV2Approvals("pending");
  const strn   = useStrengthenerApprovals();
  const fspec  = useFactorSpecApprovals();
  const decayQ = useDecayReport();
  const sessQ  = useActiveSession();

  const approvalsTotal =
    (v2.data?.n_pending ?? 0) +
    (strn.data?.n_pending ?? 0) +
    (fspec.data?.n_pending ?? 0);

  const decayAlarms = (decayQ.data?.alarms ?? [])
    .filter((a: any) => a.level && a.level !== "INFO").length;

  const activeSessionCount = sessQ.data?.active ? 1 : 0;

  const badgeForHref = (href: string): { text: string; tone: "alert"|"warn"|"info" } | undefined => {
    if (href === "/approvals" && approvalsTotal > 0) {
      return { text: String(approvalsTotal),
               tone: approvalsTotal >= 5 ? "alert" : "warn" };
    }
    if (href === "/research/decay" && decayAlarms > 0) {
      const overall = decayQ.data?.overall;
      return { text: String(decayAlarms),
               tone: overall === "ACTION" ? "alert" : "warn" };
    }
    if (href === "/research/sessions" && activeSessionCount > 0) {
      return { text: String(activeSessionCount), tone: "info" };
    }
    return undefined;
  };

  // Stable URLSearchParams snapshot. useSearchParams returns a
  // ReadonlyURLSearchParams which behaves like one but we re-wrap for
  // type clarity downstream.
  const currentSearch = new URLSearchParams(searchParams?.toString() ?? "");

  const renderItem = (it: RailItem, indent: boolean = false) => {
    const Icon = it.icon;
    const active = isActive(it.href, pathname, currentSearch, PATHS_WITH_QUERY_VARIANT);
    if (!it.ready) {
      return (
        <span key={it.href} aria-disabled
              title={hnt(it) || "coming soon"}
              className={cn(
                "flex items-center gap-2 rounded-md px-2 py-1.5 text-[12px] text-muted/40 cursor-not-allowed",
                indent && "pl-3",
              )}>
          <Icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.5} />
          <span className="truncate">{lbl(it)}</span>
          <span className="ml-auto text-[9px] uppercase tracking-wider opacity-50">
            soon
          </span>
        </span>
      );
    }
    const dynBadge = badgeForHref(it.href);
    const badgeTone = dynBadge
      ? dynBadge.tone === "alert" ? "bg-alert/15 text-alert"
      : dynBadge.tone === "warn"  ? "bg-warn/15 text-warn"
                                  : "bg-info/15 text-info"
      : "bg-accent/15 text-accent";
    const badgeText = dynBadge?.text ?? it.badge;
    return (
      <Link key={it.href} href={it.href}
            title={hnt(it)}
            className={cn(
              "flex items-center gap-2 rounded-md px-2 py-1.5 text-[12px] transition-colors",
              indent && "pl-3",
              active
                ? "bg-accent/[0.12] text-accent font-medium"
                : "text-muted hover:bg-panel2 hover:text-foreground",
            )}>
        <Icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
        <span className="truncate">{lbl(it)}</span>
        {badgeText && (
          <span className={cn("ml-auto text-[9px] tabular-nums px-1 rounded font-semibold", badgeTone)}>
            {badgeText}
          </span>
        )}
      </Link>
    );
  };

  const renderGroup = (g: RailGroup) => {
    const isActiveGroup = g.key === activeKey;
    const ModeIcon = g.modeIcon;
    // Phase D telemetry-driven cull (2026-06-23): split items into
    // promoted (default-visible) and demoted (0-visit rail items
    // pushed under a "More" disclosure). Route still works, just
    // not rail-prominent.
    const promoted = g.items.filter((it) => !it.demoted);
    const demoted  = g.items.filter((it) =>  it.demoted);
    // If the user is currently ON a demoted route, auto-open the
    // disclosure so they can see it highlighted in context.
    const onDemotedPath = demoted.some((it) =>
      isActive(it.href, pathname, currentSearch, PATHS_WITH_QUERY_VARIANT));
    return (
      <div key={g.key} className="space-y-0.5">
        <div className={cn(
          "px-2 inline-flex items-center gap-1.5 text-[10px] uppercase tracking-wider mb-1",
          isActiveGroup ? "text-accent" : "text-muted/50",
        )}>
          <ModeIcon className="h-2.5 w-2.5" strokeWidth={2.2} />
          <span>{g.label}</span>
        </div>
        {promoted.map((it) => renderItem(it))}
        {demoted.length > 0 && (
          <details className="group" open={onDemotedPath}>
            <summary className="flex items-center gap-1.5 px-2 py-1.5 cursor-pointer text-[10.5px] text-muted/40 hover:text-muted transition-colors list-none [&::-webkit-details-marker]:hidden">
              <span className="transition-transform group-open:rotate-90 inline-block w-2 text-center">›</span>
              <span>More ({demoted.length})</span>
            </summary>
            <div className="space-y-0.5 opacity-70">
              {demoted.map((it) => renderItem(it))}
            </div>
          </details>
        )}
      </div>
    );
  };

  // The mode header card at the top — shows current zone + question.
  const activeGroup = GROUPS.find((g) => g.key === activeKey);
  const ActiveModeIcon = activeGroup?.modeIcon ?? Compass;
  const modeHeaderLabel    = activeGroup ? activeGroup.label.toUpperCase() : "WORKSPACE";
  const modeHeaderQuestion = activeGroup?.question;

  return (
    <aside className="hidden lg:block w-[212px] shrink-0">
      <nav style={{ top: "var(--chrome-h, 64px)" }}
           className="sticky space-y-3 text-sm pt-3">

        {/* Mode header — anchor the user's mental zone. */}
        <div className="px-2 pb-3 border-b border-border/40 space-y-1">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted/60">
            Lab studio
          </div>
          <div className="flex items-center gap-1.5">
            <ActiveModeIcon className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
            <span className="text-sm font-semibold tracking-wide text-foreground">
              {modeHeaderLabel}
            </span>
          </div>
          {modeHeaderQuestion && (
            <div className="text-[10px] text-muted/70 italic leading-snug pt-0.5">
              {modeHeaderQuestion}
            </div>
          )}
          <div className="inline-flex items-center gap-1 text-[9px] text-muted/60 pt-1">
            <kbd className="inline-flex items-center gap-0.5 rounded border border-border/40 px-1 py-0 text-[9px] bg-panel2/40">
              <Command className="h-2 w-2" />K
            </kbd>
            <span>jump anywhere</span>
          </div>
        </div>

        {/* The four modes, rendered uniformly. */}
        {GROUPS.map(renderGroup)}

        {/* Footer doctrine pin — keeps the 4-mode model visible. */}
        <div className="px-2 pt-3 text-[10px] text-muted/40 leading-relaxed border-t border-border/30 space-y-1">
          <div><strong className="text-muted/60">Operate</strong> · is my book OK</div>
          <div><strong className="text-muted/60">Research</strong> · what to test next</div>
          <div><strong className="text-muted/60">Govern</strong> · lifecycle clean</div>
          <div><strong className="text-muted/60">Learn</strong> · what we know</div>
        </div>
      </nav>
    </aside>
  );
}
