"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Skull, CheckCircle2, ShieldCheck } from "lucide-react";
import { useDecayReport, useGraveyard, usePitAudit } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { strategyName, roleName } from "@/lib/labels";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton, ErrorState, ROLE_TONE, pct, num, cn } from "@/components/ui";
import PaperDiscoveryCard from "@/components/PaperDiscoveryCard";
import ForwardOOSWatchlistCard from "@/components/ForwardOOSWatchlistCard";
import { GraveyardFromStore } from "@/components/GraveyardFromStore";

// Point-in-time / look-ahead integrity — the #1 quant credibility lever, made visible. Surfaces the
// audit: per-strategy PIT controls (what could leak the future, how it's prevented, code anchor) +
// the D_PEAD panel's look-ahead checks (incl. honestly-FLAGged limitations). Read-only evidence.
function PitIntegrity() {
  const { t } = useI18n();
  const { data } = usePitAudit();
  if (!data || !data.available) return null;
  const b = data.book, dp = data.dpead;
  return (
    <motion.div variants={fadeUp}>
      <SectionTitle><span className="inline-flex items-center gap-1.5"><ShieldCheck className="h-3.5 w-3.5 text-ok" /> {t("pit.title")}</span></SectionTitle>
      <Card className="space-y-4">
        {b && (
          <div className="flex flex-wrap items-center gap-3">
            <Badge tone={b.book_clean ? "bg-ok/15 text-ok" : "bg-warn/15 text-warn"}>{b.book_clean ? t("pit.clean") : t("pit.not_clean")}</Badge>
            <span className="text-sm text-muted">{b.overall}</span>
          </div>
        )}
        {b?.surfaces && b.surfaces.length > 0 && (
          <div>
            <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">{t("pit.controls")}</div>
            <div className="space-y-2.5">
              {b.surfaces.map((s) => (
                <div key={s.strategy} className="border-l-2 border-border pl-3">
                  <div className="text-sm font-medium">{strategyName(s.strategy)}</div>
                  <div className="text-xs text-muted"><span className="text-warn/80">{t("pit.surface")}:</span> {s.surface}</div>
                  <div className="text-xs text-muted"><span className="text-ok/80">{t("pit.control")}:</span> {s.control}</div>
                  {s.anchor && <div className="tnum mt-0.5 text-[10px] text-muted/50">{s.anchor}</div>}
                </div>
              ))}
            </div>
          </div>
        )}
        {dp?.checks && dp.checks.length > 0 && (
          <div className="border-t border-border pt-3">
            <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">
              {t("pit.dpead")} <span className="normal-case text-muted/60">· {dp.n_rows?.toLocaleString()} {t("pit.rows")}</span>
            </div>
            <div className="space-y-1.5">
              {dp.checks.map((c) => (
                <div key={c.name} className="flex items-start gap-2 text-xs">
                  <Badge tone={c.status === "PASS" ? "bg-ok/15 text-ok" : "bg-warn/15 text-warn"} className="shrink-0">{c.status}</Badge>
                  <div className="min-w-0">
                    <span className="text-foreground">{c.name.replace(/_/g, " ")}</span>
                    <span className="text-muted"> — {c.detail}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
}

export default function ResearchPage() {
  const { t } = useI18n();
  const { data: decay } = useDecayReport();   // best-effort: live mechanisms (page still renders if it errors)
  // Graveyard is now fetched via GraveyardFromStore (research event store).
  // We still keep the page-level loading state best-effort on decay only;
  // errors don't block the page rendering.
  const mechanisms = decay ? Object.entries(decay.mechanisms) : [];

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight">{t("research.title")}</h1>
        <p className="text-sm text-muted">{t("research.subtitle")}</p>
      </motion.div>

      <motion.div variants={stagger(0.06)} initial="hidden" animate="show" className="space-y-8">
          {/* 4d.6 IA: Cockpit + Assistant promoted to top-level Lab nav.
              This page focuses on literature/PIT/discovery — the
              evidence-management side of research. */}

          {/* point-in-time integrity — the credibility headline (no look-ahead) */}
          <PitIntegrity />

          {/* paper discovery: nominate + review queue + borderline + bookmarklet */}
          <PaperDiscoveryCard />

          {/* forward OOS watchlist: tracks promoted mechanisms against auto-gate */}
          <ForwardOOSWatchlistCard />

          {/* Mechanism track record · live (demoted 2026-06-02 from
              6-card grid to compact table — richer composition views
              now live on /book Tearsheet + /ops Active Deploy. This
              row remains as the research-track-record narrative anchor
              that pairs visually with the graveyard below.) */}
          <div>
            <motion.div variants={fadeUp}>
              <SectionTitle className="mb-0 flex items-baseline gap-2">
                <span className="inline-flex items-center gap-1.5">
                  <CheckCircle2 className="h-3.5 w-3.5 text-ok" />
                  Mechanism track record · live
                </span>
                <span className="text-[11px] text-muted font-normal">
                  · per-mechanism full-period Sharpe (research view; deployment composition is on /book and /ops)
                </span>
              </SectionTitle>
            </motion.div>
            {mechanisms.length > 0 ? (
              <motion.div variants={fadeUp} className="mt-2">
                <Card className="overflow-hidden p-0">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                        <th className="px-3 py-2 font-medium">Mechanism</th>
                        <th className="px-3 py-2 font-medium">Role</th>
                        <th className="px-3 py-2 text-right font-medium">{t("mech.full_sharpe")}</th>
                        <th className="px-3 py-2 text-right font-medium">{t("mech.weight")}</th>
                        <th className="px-3 py-2 font-medium">Flag</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mechanisms.map(([name, m]) => (
                        <tr key={name} className="border-b border-border/40 last:border-0 hover:bg-panel2/30 transition-colors">
                          <td className="px-3 py-1.5 font-medium">{strategyName(name)}</td>
                          <td className="px-3 py-1.5">
                            <span className={cn("rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider", ROLE_TONE[m.role])}>
                              {roleName(m.role)}
                            </span>
                          </td>
                          <td className="tnum px-3 py-1.5 text-right font-semibold">{num(m.full_sharpe)}</td>
                          <td className="tnum px-3 py-1.5 text-right">{pct(m.weight, 0)}</td>
                          <td className="px-3 py-1.5">
                            {m.structural_decay ? (
                              <span className="rounded bg-alert/15 text-alert px-1.5 py-0.5 text-[10px] uppercase tracking-wider">
                                decay
                              </span>
                            ) : (
                              <span className="text-muted/40 text-xs">—</span>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </motion.div>
            ) : (
              <motion.div variants={fadeUp} className="mt-2"><Card><p className="text-sm text-muted">{t("research.live_on_dash")}</p></Card></motion.div>
            )}
          </div>

          {/* graveyard — REWIRED to research event store 2026-06-02 (M3).
              Old curated graveyard.json (~7 entries) replaced with the
              full store-backed view (50+ RED events with full search /
              filter / lineage). The curated 'why' fields are subsumed
              into event.summary; ancillary metrics + tags now visible
              on expand. Senior reviewers want to BE ABLE TO see
              everything, then narrow — that's the new pattern. */}
          <GraveyardFromStore />
      </motion.div>
    </>
  );
}
