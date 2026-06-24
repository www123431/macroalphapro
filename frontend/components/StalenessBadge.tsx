"use client";

// StalenessBadge — universal "as of YYYY-MM-DD" + age renderer.
//
// Doctrine: every dashboard datum has a freshness budget. Past the budget
// the badge auto-recolours — amber > 3d, red > 14d — so the user never
// trusts a stale snapshot by accident. Tooltip surfaces *what to rerun*
// to refresh.
//
// 2026-06-02 — see project memory `feedback_dashboard_freshness_budget`
// for the standing rule that motivated this component.

import { cn } from "@/components/ui";

export interface StalenessBadgeProps {
  /** ISO date the data was generated (e.g. "2026-05-25"). */
  asOf?: string | null;
  /** Pre-computed age in days. If both are given, ageDays wins. */
  ageDays?: number | null;
  /** Optional refresh hint shown in tooltip / on hover. */
  refreshHint?: string;
  /** Days after which badge goes amber. Default 3. */
  warnAfterDays?: number;
  /** Days after which badge goes red. Default 14. */
  alertAfterDays?: number;
  /** Compact rendering — drop the "as of" prefix. */
  compact?: boolean;
  className?: string;
}


function _computeAge(asOf?: string | null, ageDays?: number | null): number | null {
  if (typeof ageDays === "number" && Number.isFinite(ageDays)) return ageDays;
  if (!asOf) return null;
  // Accept "2026-05-25" or "2026-05-25T12:00:00Z" — take first 10 chars.
  const ymd = asOf.slice(0, 10);
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(ymd);
  if (!m) return null;
  const d = new Date(Date.UTC(+m[1], +m[2] - 1, +m[3]));
  const now = new Date();
  const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  return Math.floor((today.getTime() - d.getTime()) / 86_400_000);
}


export function StalenessBadge({
  asOf,
  ageDays,
  refreshHint,
  warnAfterDays = 3,
  alertAfterDays = 14,
  compact = false,
  className,
}: StalenessBadgeProps) {
  const age = _computeAge(asOf, ageDays);
  const level: "fresh" | "warn" | "alert" | "unknown" =
    age == null ? "unknown" :
    age >= alertAfterDays ? "alert" :
    age >= warnAfterDays ? "warn" :
    "fresh";

  // Tooltip text: explain what fresh budget is + what to rerun
  const tip = (() => {
    if (level === "unknown") return "no as_of timestamp on this datum";
    const baseAge = `${age}d old (as of ${asOf?.slice(0, 10) ?? "?"})`;
    if (level === "fresh")  return `${baseAge} · within ${warnAfterDays}d budget`;
    if (level === "warn")   return `${baseAge} · past ${warnAfterDays}d budget${refreshHint ? ` — ${refreshHint}` : ""}`;
    return `${baseAge} · past ${alertAfterDays}d red budget${refreshHint ? ` — ${refreshHint}` : ""}`;
  })();

  const color =
    level === "alert"  ? "text-alert border-alert/40 bg-alert/10" :
    level === "warn"   ? "text-warn  border-warn/40  bg-warn/10"  :
    level === "fresh"  ? "text-muted border-border/40 bg-transparent" :
                          "text-muted/60 border-border/30 bg-transparent";

  const label = compact
    ? (age == null ? "—" : `${age}d`)
    : `${asOf?.slice(0, 10) ?? "—"}${age != null ? ` · ${age}d` : ""}`;

  return (
    <span
      title={tip}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] tnum",
        color,
        className,
      )}
      aria-label={tip}
    >
      {!compact && <span className="text-muted/70 mr-0.5">as of</span>}
      <span>{label}</span>
    </span>
  );
}
