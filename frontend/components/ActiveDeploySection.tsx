"use client";

// ActiveDeploySection — the canonical "what's live RIGHT NOW" panel on /ops.
//
// Why it exists (2026-06-02):
// On 2026-06-02 the /book Tearsheet silently showed a 1.03 Sharpe from the
// old 2-mechanism narrative for 3 days after config C (5-sleeve, Sharpe
// 0.962, maxDD -7.10%) was actually deployed on 2026-05-30. The defect was
// structural — no single source of truth for "currently deployed", and no
// place in the UI that surfaced this state as an OPERATIONAL fact.
//
// This component is the operational answer. It reads /api/deploy/manifest
// (which reads data/portfolio/active_deployment.yaml, the SoT) and shows:
//   - which config is active + how long it's been live
//   - the per-sleeve composition (weights + roles + regime flags)
//   - regime grids (insurance modulation table)
//   - code_drift_issues — anything where Python constants disagree with
//     the YAML. Empty = healthy, non-empty = HARD red banner.
//
// Per the Settings vs Ops boundary doctrine: this is OPERATIONAL state, not
// a user preference. It lives in /ops, not /settings.

import { Network, AlertTriangle, CheckCircle2, FileCode } from "lucide-react";
import { DeployManifest } from "@/lib/api";
import { Card, SectionTitle, cn, num, pct } from "@/components/ui";


function DriftBanner({ issues }: { issues: string[] }) {
  if (!issues || issues.length === 0) return null;
  return (
    <div className="rounded-lg border border-alert/50 bg-alert/10 p-3 space-y-1.5">
      <div className="flex items-center gap-2 text-alert font-medium">
        <AlertTriangle className="h-4 w-4" />
        Code drift detected — {issues.length} mismatch{issues.length === 1 ? "" : "es"}
      </div>
      <div className="text-[12px] text-foreground/90 font-mono space-y-0.5">
        {issues.map((i, idx) => <div key={idx}>• {i}</div>)}
      </div>
      <div className="text-[11px] text-muted leading-relaxed mt-1.5">
        Python constants in <span className="font-mono">engine/portfolio/combined_book.py</span> disagree with the deployment manifest.
        Either update the YAML to match (if code changed legitimately) or roll back the code constant.
        Run <span className="font-mono">python scripts/deploy_config.py check</span> for the same report.
      </div>
    </div>
  );
}


function CleanBanner() {
  return (
    <div className="rounded-lg border border-ok/40 bg-ok/5 p-2.5 flex items-center gap-2 text-[12px] text-ok/90">
      <CheckCircle2 className="h-4 w-4 shrink-0" />
      <span>Manifest and Python constants agree. No drift.</span>
    </div>
  );
}


export function ActiveDeploySection({ manifest }: { manifest?: DeployManifest }) {
  if (!manifest || !manifest.available) {
    return (
      <div>
        <SectionTitle className="mb-0 flex items-center gap-1.5">
          <Network className="h-3.5 w-3.5" />
          <span>Active deployment</span>
        </SectionTitle>
        <Card className="mt-2">
          <div className="text-sm text-muted">
            {manifest?.reason ?? "Deploy manifest not loaded — check data/portfolio/active_deployment.yaml"}
          </div>
        </Card>
      </div>
    );
  }

  const drifty = (manifest.code_drift_issues ?? []).length > 0;

  return (
    <div>
      <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
        <span className="inline-flex items-center gap-1.5">
          <Network className="h-3.5 w-3.5" />
          Active deployment
        </span>
        <span className="text-[11px] text-muted font-normal">
          · {manifest.config_id}
          {manifest.deploy_date && (
            <> · live since {manifest.deploy_date} ({manifest.days_since_deploy}d)</>
          )}
          {manifest.book_vol_target != null && (
            <> · vol target {pct(manifest.book_vol_target)}</>
          )}
        </span>
      </SectionTitle>
      <Card className="mt-2 space-y-4">
        {/* Drift status banner — first thing the eye lands on */}
        {drifty ? <DriftBanner issues={manifest.code_drift_issues!} /> : <CleanBanner />}

        {/* Summary line */}
        {manifest.summary && (
          <p className="text-[12px] text-foreground/85 leading-relaxed">{manifest.summary}</p>
        )}

        {/* Per-sleeve composition table */}
        {manifest.sleeves && manifest.sleeves.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-1.5 font-medium">Sleeve</th>
                  <th className="px-3 py-1.5 font-medium">Role</th>
                  <th className="px-3 py-1.5 text-right font-medium">Weight</th>
                  <th className="px-3 py-1.5 text-center font-medium">Regime-modulated</th>
                  <th className="px-3 py-1.5 text-right font-medium">Vol target</th>
                  <th className="px-3 py-1.5 font-medium">Specs</th>
                  <th className="px-3 py-1.5 font-medium">Builder</th>
                </tr>
              </thead>
              <tbody>
                {manifest.sleeves.map((s) => (
                  <tr key={s.name} className="border-b border-border/40 last:border-0">
                    <td className="px-3 py-1.5 font-mono text-foreground/90">{s.name}</td>
                    <td className="px-3 py-1.5">
                      <span className={cn(
                        "rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider",
                        s.role === "alpha"       ? "bg-accent/10 text-accent" :
                        s.role === "insurance"   ? "bg-warn/10 text-warn"     :
                        s.role === "diversifier" ? "bg-ok/10 text-ok"         :
                                                    "bg-panel2/40 text-muted")}>
                        {s.role}
                      </span>
                    </td>
                    <td className="tnum px-3 py-1.5 text-right">{pct(s.base_weight)}</td>
                    <td className="px-3 py-1.5 text-center text-muted/80 text-[11px]">
                      {s.regime_modulated ? "✓" : "—"}
                    </td>
                    <td className="tnum px-3 py-1.5 text-right text-muted">{pct(s.target_vol)}</td>
                    <td className="px-3 py-1.5 font-mono text-[11px] text-muted">
                      {s.signing_spec_ids?.join(", ") || "—"}
                    </td>
                    <td className="px-3 py-1.5 font-mono text-[10px] text-muted/70 truncate" title={s.builder}>
                      {s.builder?.split(".").slice(-2).join(".") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Regime grids — only if at least one sleeve is regime-modulated */}
        {manifest.regime_grids && Object.keys(manifest.regime_grids).length > 0 &&
         manifest.sleeves?.some((s) => s.regime_modulated) && (
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-1.5">
              Regime-conditional insurance grid
              {manifest.regime_classifier?.kind && (
                <span className="normal-case text-muted/60 ml-1.5">
                  · {manifest.regime_classifier.kind} (±{manifest.regime_classifier.threshold_sigma}σ,
                  {" "}{manifest.regime_classifier.lookback_days}d lookback)
                </span>
              )}
            </div>
            <div className="grid grid-cols-3 gap-2 text-[11px]">
              {Object.entries(manifest.regime_grids).map(([rname, grid]) => (
                <div key={rname} className="rounded border border-border/40 bg-panel2/30 px-2 py-1.5">
                  <div className="font-mono text-foreground/90 mb-0.5">{rname}</div>
                  {Object.entries(grid).map(([sleeve, w]) => (
                    <div key={sleeve} className="tnum text-muted">{sleeve} {pct(w)}</div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Expected stats + spec references */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-[11px] border-t border-border/40 pt-2">
          {manifest.expected_stats?.sharpe != null && (
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70">Expected Sharpe</div>
              <div className="tnum text-base font-semibold">{num(manifest.expected_stats.sharpe)}</div>
            </div>
          )}
          {manifest.expected_stats?.max_dd != null && (
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70">Expected Max DD</div>
              <div className="tnum text-base font-semibold text-alert">{pct(manifest.expected_stats.max_dd)}</div>
            </div>
          )}
          {manifest.expected_stats?.backtest_window && (
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70">Backtest window</div>
              <div className="font-mono text-[12px] text-foreground/85">{manifest.expected_stats.backtest_window}</div>
            </div>
          )}
          {manifest.signing_spec_ids && manifest.signing_spec_ids.length > 0 && (
            <div>
              <div className="text-[9px] uppercase tracking-wider text-muted/70">Signing specs</div>
              <div className="font-mono text-[12px] text-foreground/85">
                {manifest.signing_spec_ids.join(", ")}
              </div>
            </div>
          )}
        </div>

        {/* Footer — how to make changes */}
        <div className="border-t border-border/40 pt-2 text-[11px] text-muted/80 leading-relaxed flex items-start gap-2">
          <FileCode className="h-3.5 w-3.5 shrink-0 mt-0.5" />
          <div>
            Source of truth: <span className="font-mono text-muted">data/portfolio/active_deployment.yaml</span>.
            Promote a new config via <span className="font-mono text-muted">python scripts/deploy_config.py promote --id … --reason …</span>.
            Check current drift with <span className="font-mono text-muted">deploy_config.py check</span>.
          </div>
        </div>
      </Card>
    </div>
  );
}
