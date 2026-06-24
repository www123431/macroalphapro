"use client";

// /research/calibration — Belief Layer headline numbers + HONEST NEGATIVE
// FINDING surfaced as a first-class page.
//
// Created 2026-06-23. Was buried in markdown (belief_track_record_rigor.md);
// now reachable in one click from the KpiHeroStrip "Brier" tile present
// on every page. The honest negative finding (predictor LOSES to a fair
// family-prior baseline by +0.114 Brier) is the project's most
// publishable claim — not a footnote.
//
// Data source: /api/research/belief/calibration (reads
// data/research/belief_track_record_rigor.json refreshed daily by
// daily-belief-refresh cron 06:35).
//
// Scope: read-only. For the full statistical rigor pass (all 8 tests
// with bin details) the user opens docs or runs
// `python scripts/reports/report_belief_track_record_rigor.py`.

import Link from "next/link";
import { motion } from "framer-motion";
import { Brain, ShieldCheck, AlertTriangle, ExternalLink, Microscope, FileText } from "lucide-react";
import { useBeliefCalibration, useBeliefFamilies } from "@/lib/queries";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton } from "@/components/ui";


function StatTile({ label, value, sub, tone }: {
  label: string;
  value: string;
  sub?:  string;
  tone?: "ok" | "warn" | "danger" | "info" | "muted";
}) {
  const toneCls =
    tone === "ok"     ? "text-ok"     :
    tone === "warn"   ? "text-warn"   :
    tone === "danger" ? "text-danger" :
    tone === "info"   ? "text-info"   :
                        "text-foreground";
  return (
    <Card className="space-y-1">
      <div className="text-[10px] uppercase tracking-wider text-muted/70">{label}</div>
      <div className={`text-2xl font-semibold tnum ${toneCls}`}>{value}</div>
      {sub && <div className="text-xs text-muted">{sub}</div>}
    </Card>
  );
}


export default function CalibrationPage() {
  const calibQ  = useBeliefCalibration();
  const famQ    = useBeliefFamilies(1);
  const c       = calibQ.data;

  if (calibQ.isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-12 w-2/3" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
        </div>
        <Skeleton className="h-40" />
      </div>
    );
  }

  if (!c?.available) {
    return (
      <Card className="text-sm text-muted">
        Belief calibration data not generated yet. Run{" "}
        <code className="rounded bg-panel2 px-1 py-0.5">scripts/reports/report_belief_track_record_rigor.py</code>{" "}
        or wait for the daily-belief-refresh cron (06:35) to populate
        <code className="ml-1 rounded bg-panel2 px-1 py-0.5">data/research/belief_track_record_rigor.json</code>.
      </Card>
    );
  }

  // Derived flags
  const predictorBeatsRandom  = !!c.predictor_beats_random;
  const deltaPositive         = (c.delta_predictor_minus_fp ?? 0) > 0;
  const deltaCIExcludesZero   = (c.delta_ci_lo ?? 0) > 0;
  const hlRejected            = c.hl_calibrated === false;

  return (
    <motion.div initial="hidden" animate="show" variants={stagger(0.08)} className="space-y-8">
      {/* Header */}
      <motion.div variants={fadeUp}>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
              <Brain className="h-5 w-5 text-accent" />
              Belief Layer — Calibration
            </h1>
            <p className="mt-1 text-sm text-muted">
              Predictor track record vs random + fair family-prior baselines.{" "}
              n = {c.n_autopsies} autopsies. Refreshed daily.
            </p>
          </div>
          <Badge tone={predictorBeatsRandom ? "bg-ok/15 text-ok" : "bg-warn/15 text-warn"}>
            {predictorBeatsRandom ? "Beats random" : "No edge vs random"}
          </Badge>
        </div>
      </motion.div>

      {/* Headline stat tiles */}
      <motion.div variants={fadeUp} className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile
          label="Predictor Brier"
          value={(c.predictor_brier ?? 0).toFixed(3)}
          sub={`95% CI [${(c.predictor_ci_lo ?? 0).toFixed(3)}, ${(c.predictor_ci_hi ?? 0).toFixed(3)}]`}
          tone="info"
        />
        <StatTile
          label="Random baseline"
          value={(c.random_baseline ?? 0.4444).toFixed(3)}
          sub="3-class uniform"
          tone="muted"
        />
        <StatTile
          label="Family-prior baseline"
          value={(c.family_prior_brier ?? 0).toFixed(3)}
          sub="time-aware (no future-info)"
          tone="ok"
        />
        <StatTile
          label="Δ Predictor − Fam"
          value={`${deltaPositive ? "+" : ""}${(c.delta_predictor_minus_fp ?? 0).toFixed(3)}`}
          sub={`CI [${(c.delta_ci_lo ?? 0).toFixed(3)}, ${(c.delta_ci_hi ?? 0).toFixed(3)}]${deltaCIExcludesZero ? " · excludes 0" : ""}`}
          tone={deltaPositive ? "warn" : "ok"}
        />
      </motion.div>

      {/* The honest negative finding — central, not hidden */}
      <motion.div variants={fadeUp}>
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 text-warn" /> The honest negative finding
          </span>
        </SectionTitle>
        <Card className="border-warn/25 space-y-3">
          <p className="text-sm leading-relaxed">
            The LLM predictor <span className="font-semibold text-warn">loses</span> to a
            deterministic family-prior baseline by{" "}
            <span className="tnum font-semibold">
              +{(c.delta_predictor_minus_fp ?? 0).toFixed(3)} Brier
            </span>
            {deltaCIExcludesZero && (
              <>
                {" "}(95% CI{" "}
                <span className="tnum">
                  [+{(c.delta_ci_lo ?? 0).toFixed(3)}, +{(c.delta_ci_hi ?? 0).toFixed(3)}]
                </span>{" "}
                strictly excludes zero).
              </>
            )}{" "}
            The family-prior baseline knows nothing — it predicts the historical
            verdict mix per family — yet beats the LLM. The LLM&apos;s
            value-add above family-empirical is{" "}
            <span className="font-semibold">not established</span> on this sample.
          </p>
          <p className="text-xs leading-relaxed text-muted">
            This is the kind of finding most labs would not publish.
            Surfacing it in the production UI — instead of burying it in
            an arxiv appendix — IS the differentiator. The W7-v09
            correction (forcing per-family <code>w_fam = 1.0</code>) is
            the system&apos;s direct response: pure family-empirical
            achieves Brier {(c.family_prior_brier ?? 0).toFixed(3)},
            beating the LLM-only path.
          </p>
        </Card>
      </motion.div>

      {/* Calibration goodness-of-fit */}
      <motion.div variants={fadeUp}>
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <Microscope className="h-3.5 w-3.5 text-accent" /> Hosmer-Lemeshow goodness-of-fit
          </span>
        </SectionTitle>
        <Card className="flex flex-wrap items-center justify-between gap-3 text-sm">
          <div>
            <div className="text-muted text-xs uppercase tracking-wider">verdict</div>
            <div className={`font-semibold ${hlRejected ? "text-warn" : "text-ok"}`}>
              {hlRejected ? "REJECTED — predicted probabilities not well-calibrated" : "Calibrated (fail to reject H0)"}
            </div>
          </div>
          <div className="flex items-center gap-4 text-xs text-muted tnum">
            <div>χ² = {(c.hl_chi2 ?? 0).toFixed(2)}</div>
            <div>p = {(c.hl_p_value ?? 1).toFixed(4)}</div>
            <div>α = 0.05</div>
          </div>
        </Card>
      </motion.div>

      {/* Per-family belief depth — gives context for the headline number */}
      {famQ.data && famQ.data.n_families > 0 && (
        <motion.div variants={fadeUp}>
          <SectionTitle>
            <span className="inline-flex items-center gap-1.5">
              <ShieldCheck className="h-3.5 w-3.5 text-info" /> Per-family belief depth
            </span>
          </SectionTitle>
          <Card>
            <div className="mb-3 text-xs text-muted">
              {famQ.data.n_total_obs} total observations across {famQ.data.n_families} families
              · <span className="text-ok">{famQ.data.n_green_total}G</span>
              · <span className="text-warn">{famQ.data.n_marginal_total}M</span>
              · <span className="text-danger">{famQ.data.n_red_total}R</span>
            </div>
            <div className="space-y-1.5">
              {famQ.data.families.map((f) => (
                <Link
                  key={f.family}
                  href={`/research/family?id=${encodeURIComponent(f.family)}`}
                  className="flex items-center justify-between text-xs hover:bg-panel2/40 rounded px-1 -mx-1 transition-colors"
                >
                  <div className="min-w-0 truncate">
                    <span className="font-mono text-foreground/80">{f.family}</span>
                    <span className="text-muted/60"> · n={f.n_obs}</span>
                  </div>
                  <div className="flex items-center gap-2 tnum">
                    <span className="text-ok">{f.n_green}G</span>
                    <span className="text-warn">{f.n_marginal}M</span>
                    <span className="text-danger">{f.n_red}R</span>
                    <span className="text-[10px] uppercase text-muted/60 ml-1">
                      {f.direction_hint?.split(" ")[0] || "—"}
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          </Card>
        </motion.div>
      )}

      {/* Pointers */}
      <motion.div variants={fadeUp}>
        <Card className="space-y-2 text-xs text-muted">
          <div className="flex items-center gap-2">
            <FileText className="h-3.5 w-3.5" />
            <span>Full rigor pass (8 tests with bin details):</span>
            <code className="rounded bg-panel2 px-1.5 py-0.5">data/research/belief_track_record_rigor.md</code>
          </div>
          <div className="flex items-center gap-2">
            <FileText className="h-3.5 w-3.5" />
            <span>Reproducer:</span>
            <code className="rounded bg-panel2 px-1.5 py-0.5">scripts/reports/report_belief_track_record_rigor.py</code>
          </div>
          <div className="flex items-center gap-2">
            <FileText className="h-3.5 w-3.5" />
            <span>Arxiv preprint draft:</span>
            <code className="rounded bg-panel2 px-1.5 py-0.5">docs/arxiv_preprint_draft_2026-06-22.md</code>
          </div>
          <div className="flex items-center gap-2">
            <ExternalLink className="h-3.5 w-3.5" />
            <Link href="/research/sessions" className="text-info hover:underline">
              See active research sessions →
            </Link>
          </div>
        </Card>
      </motion.div>
    </motion.div>
  );
}
