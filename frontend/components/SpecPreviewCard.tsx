"use client";

// SpecPreviewCard — shows the structured HypothesisSpec for a picked
// hypothesis in /research/enhance PICK panel. Closes the gap the user
// called out as "项目灵魂": users now see EXACTLY what the system
// extracted before any test is run.
//
// Behavior
//   - Fetches /api/hypothesis_spec/{hypothesis_id}
//   - 404 → "no spec yet" + Extract Now button (calls POST .../extract)
//   - 200 → renders structured fields with confidence badge
//   - confidence < 0.5 → red warning + suggest editing
//   - spec_hash visible (the load-bearing reproducibility identifier)

import { useEffect, useState } from "react";
import {
  Sparkles, AlertTriangle, RefreshCw, Loader2, ChevronDown,
  ChevronUp, FileCheck2,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, Badge, cn } from "@/components/ui";


type SignalLeg = {
  signal_type: string;
  sign: string;
  lookback_periods: number[];
  quantile: number;
  role: string;
  note: string;
};


type Spec = {
  spec_id:              string;
  spec_version:         number;
  source_hypothesis_id: string;
  version:              number;
  spec_hash:            string;
  family:               string;
  claim_text:           string;
  legs:                 SignalLeg[];
  universe: {
    asset_class: string;
    subset: string;
    custom_tickers?: string[] | null;
    min_history_months: number;
  };
  construction: {
    weighting: string;
    rebalance: string;
    skip_first_day: boolean;
    holding_period_n: number;
  };
  risk: {
    vol_target_annual?: number | null;
    max_leverage?: number | null;
    turnover_cap_annual?: number | null;
    max_position?: number | null;
    drawdown_stop?: number | null;
  };
  outcome: {
    predicted_direction: string;
    predicted_sharpe_lo?: number | null;
    predicted_sharpe_hi?: number | null;
    rationale: string;
  };
  extraction: {
    method: string;
    confidence: number;
    extracted_ts: string;
    extractor_v: string;
  };
  created_ts: string;
  git_sha:    string;
};


type CoverageGap = {
  role: string;
  expected_key: string;
  reason: string;
};


type SpecCoverage = {
  hypothesis_id: string;
  spec_hash?: string | null;
  covered: boolean;
  gaps: CoverageGap[];
};


export function SpecPreviewCard({ hypothesisId }: { hypothesisId: string }) {
  const [spec, setSpec]       = useState<Spec | null>(null);
  const [coverage, setCov]    = useState<SpecCoverage | null>(null);
  const [building, setBuild]  = useState(false);
  const [buildErr, setBerr]   = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [extracting, setExtr] = useState(false);
  const [err, setErr]         = useState<string | null>(null);
  const [expanded, setExp]    = useState(false);

  const load = async () => {
    setLoading(true); setErr(null); setMissing(false);
    try {
      const r = await fetch(
        `${API_BASE}/api/hypothesis_spec/${encodeURIComponent(hypothesisId)}`,
        { cache: "no-store" },
      );
      if (r.status === 404) {
        setMissing(true); setSpec(null);
      } else if (r.ok) {
        setSpec(await r.json());
        // Also fetch composer coverage for this spec
        try {
          const cr = await fetch(
            `${API_BASE}/api/composer/coverage/spec/${encodeURIComponent(hypothesisId)}`,
            { cache: "no-store" },
          );
          if (cr.ok) setCov(await cr.json());
        } catch {}
      } else {
        throw new Error(`HTTP ${r.status}`);
      }
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setLoading(false); }
  };

  const buildSeries = async () => {
    setBuild(true); setBerr(null);
    try {
      const r = await fetch(`${API_BASE}/api/composer/build`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hypothesis_id: hypothesisId }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const out = await r.json();
      if (!out.ok) {
        throw new Error(out.error || "build failed");
      }
      setBerr(null);
      // Trigger a coverage reload to show success
      await load();
    } catch (e: any) {
      setBerr(String(e?.message ?? e));
    } finally { setBuild(false); }
  };

  const extract = async () => {
    setExtr(true); setErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/hypothesis_spec/extract`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hypothesis_id: hypothesisId }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setSpec(await r.json());
      setMissing(false);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setExtr(false); }
  };

  useEffect(() => {
    if (hypothesisId) load();
  }, [hypothesisId]);

  if (!hypothesisId) return null;

  if (loading && !spec && !missing) {
    return (
      <Card className="p-3">
        <div className="text-[10.5px] text-muted inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> loading hypothesis spec…
        </div>
      </Card>
    );
  }

  if (missing) {
    return (
      <Card className="p-3 border border-warn/30 bg-warn/[0.04]">
        <div className="text-[11px] text-foreground/90 mb-2 inline-flex items-center gap-1.5">
          <Sparkles className="h-3 w-3 text-warn" />
          No structured spec for this hypothesis yet
        </div>
        <p className="text-[10.5px] text-muted/80 mb-2">
          Without a typed HypothesisSpec the Composer can't build a
          spec-grounded returns series. Extract one now (one Claude call,
          ~$0.005, ~3s).
        </p>
        <button onClick={extract} disabled={extracting}
          className="inline-flex items-center gap-1.5 rounded bg-accent text-background hover:bg-accent/90 disabled:opacity-50 px-2.5 py-1 text-[10.5px] font-semibold">
          {extracting
            ? <><Loader2 className="h-3 w-3 animate-spin" /> Extracting…</>
            : <><Sparkles className="h-3 w-3" /> Extract spec via LLM</>}
        </button>
        {err && <div className="text-[10px] text-danger mt-2">{err}</div>}
      </Card>
    );
  }

  if (!spec) {
    return (
      <Card className="p-3 border border-danger/30 bg-danger/[0.04]">
        <div className="text-[11px] text-danger">
          spec load failed{err ? ` · ${err}` : ""}
        </div>
      </Card>
    );
  }

  const conf = spec.extraction.confidence;
  const confTone =
    conf >= 0.85 ? "ok" :
    conf >= 0.50 ? "warn" :
                    "danger";

  return (
    <Card className={cn(
      "p-0 overflow-hidden border",
      confTone === "ok"     && "border-ok/30 bg-ok/[0.03]",
      confTone === "warn"   && "border-warn/30 bg-warn/[0.04]",
      confTone === "danger" && "border-danger/30 bg-danger/[0.04]",
    )}>
      {/* Header */}
      <div className="px-3 py-2 border-b border-border/30 bg-panel2/30 flex items-center gap-2">
        <FileCheck2 className={cn(
          "h-3.5 w-3.5",
          confTone === "ok"     && "text-ok",
          confTone === "warn"   && "text-warn",
          confTone === "danger" && "text-danger",
        )} strokeWidth={2.2} />
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] font-semibold text-foreground">
            Hypothesis spec — what the Composer will test
          </div>
          <div className="text-[9.5px] text-muted/70 inline-flex items-center gap-1.5 flex-wrap">
            <span className="font-mono">spec_hash = {spec.spec_hash}</span>
            <span>·</span>
            <Badge className={cn(
              "text-[8.5px] uppercase tracking-wider",
              confTone === "ok"     && "bg-ok/15 text-ok",
              confTone === "warn"   && "bg-warn/15 text-warn",
              confTone === "danger" && "bg-danger/15 text-danger",
            )}>
              confidence {Math.round(conf * 100)}%
            </Badge>
            {coverage && (
              <>
                <span>·</span>
                <Badge className={cn(
                  "text-[8.5px] uppercase tracking-wider",
                  coverage.covered
                    ? "bg-ok/15 text-ok"
                    : "bg-warn/15 text-warn",
                )}>
                  composer {coverage.covered
                    ? "✓ 5 roles covered"
                    : `${coverage.gaps.length} gap${coverage.gaps.length === 1 ? "" : "s"}`}
                </Badge>
              </>
            )}
          </div>
        </div>
        <button onClick={() => extract()} disabled={extracting}
          title="re-extract spec (new LLM call)"
          className="text-muted hover:text-accent disabled:opacity-50">
          <RefreshCw className={cn("h-3 w-3", extracting && "animate-spin")} />
        </button>
        <button onClick={() => setExp((v) => !v)}
          aria-label={expanded ? "collapse" : "expand"}
          className="text-muted hover:text-foreground">
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </div>

      {confTone === "danger" && (
        <div className="px-3 py-1.5 bg-danger/10 border-b border-danger/30 text-[10.5px] text-danger inline-flex items-start gap-1.5">
          <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
          <span>
            Low extraction confidence — fields below may be guesses. Re-run
            extraction or hand-edit before testing.
          </span>
        </div>
      )}

      {/* Always-visible summary line */}
      <div className="px-3 py-2 text-[11px] text-foreground/85">
        <span className="font-mono text-accent">{spec.family}</span>
        {" on "}
        <span className="font-mono">{spec.universe.asset_class} / {spec.universe.subset}</span>
        {", "}
        <span>{spec.construction.rebalance.toLowerCase()}</span>
        {" rebalance, "}
        <span>{spec.construction.weighting.toLowerCase().replace(/_/g, " ")}</span>
        {" weighted, "}
        <span className={cn(
          spec.outcome.predicted_direction === "POSITIVE" && "text-ok",
          spec.outcome.predicted_direction === "NEGATIVE" && "text-danger",
        )}>{spec.outcome.predicted_direction.toLowerCase()}</span>
        {" direction"}
        {spec.legs.length > 1 && (
          <span className="text-muted/80"> · {spec.legs.length} legs</span>
        )}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 py-2 border-t border-border/30 text-[10.5px] space-y-2">
          {/* Legs */}
          <div>
            <div className="text-[9.5px] uppercase tracking-wider text-muted/70 mb-1">
              Signal legs
            </div>
            <ul className="space-y-1">
              {spec.legs.map((L, i) => (
                <li key={i} className="rounded border border-border/30 bg-panel2/20 px-2 py-1">
                  <div className="flex items-baseline gap-2">
                    <code className="text-[10.5px] text-foreground/95">{L.signal_type}</code>
                    <span className="text-muted/70">·</span>
                    <span>{L.sign.toLowerCase().replace(/_/g, " ")}</span>
                    <span className="text-muted/70">·</span>
                    <span>lookback {L.lookback_periods.join(",")}m</span>
                    <span className="text-muted/70">·</span>
                    <span>q={L.quantile}</span>
                    {L.role !== "primary" && (
                      <Badge className="bg-muted/15 text-muted/80 text-[9px] ml-auto">
                        {L.role}
                      </Badge>
                    )}
                  </div>
                  {L.note && (
                    <div className="text-[10px] text-muted/70 mt-0.5">{L.note}</div>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Risk */}
          <div className="grid grid-cols-2 gap-x-3 gap-y-1">
            <div>
              <span className="text-muted/70">vol target </span>
              <code>{spec.risk.vol_target_annual ?? "—"}</code>
            </div>
            <div>
              <span className="text-muted/70">max leverage </span>
              <code>{spec.risk.max_leverage ?? "—"}</code>
            </div>
            <div>
              <span className="text-muted/70">turnover cap </span>
              <code>{spec.risk.turnover_cap_annual ?? "—"}</code>
            </div>
            <div>
              <span className="text-muted/70">drawdown stop </span>
              <code>{spec.risk.drawdown_stop ?? "—"}</code>
            </div>
          </div>

          {/* Outcome */}
          {spec.outcome.rationale && (
            <div>
              <span className="text-muted/70">rationale: </span>
              <span className="text-foreground/85">{spec.outcome.rationale}</span>
            </div>
          )}
          {(spec.outcome.predicted_sharpe_lo != null || spec.outcome.predicted_sharpe_hi != null) && (
            <div>
              <span className="text-muted/70">predicted Sharpe range: </span>
              <code>
                {spec.outcome.predicted_sharpe_lo ?? "—"} … {spec.outcome.predicted_sharpe_hi ?? "—"}
              </code>
            </div>
          )}

          {/* Composer coverage + build action */}
          {coverage && (
            <div className="rounded border border-border/30 bg-panel2/20 p-2 space-y-1.5">
              <div className="text-[9.5px] uppercase tracking-wider text-muted/70">
                Composer coverage
              </div>
              {coverage.covered ? (
                <div className="text-[10.5px] text-ok/80">
                  All 5 component roles covered for this spec.
                </div>
              ) : (
                <ul className="space-y-0.5">
                  {coverage.gaps.map((g, i) => (
                    <li key={i} className="text-[10px] text-warn/80">
                      <code>{g.role}</code> / <code>{g.expected_key}</code>{" "}
                      <span className="text-muted/60">— {g.reason}</span>
                    </li>
                  ))}
                </ul>
              )}
              {coverage.covered && (
                <div>
                  <button
                    onClick={buildSeries}
                    disabled={building}
                    className={cn(
                      "inline-flex items-center gap-1 rounded px-2.5 py-1 text-[10.5px] font-semibold",
                      building
                        ? "bg-muted/20 text-muted/60 cursor-not-allowed"
                        : "bg-accent text-background hover:bg-accent/90",
                    )}>
                    {building
                      ? <><Loader2 className="h-3 w-3 animate-spin" /> Composing series…</>
                      : <>Build series via Composer</>}
                  </button>
                  {buildErr && (
                    <div className="text-[10px] text-danger mt-1">{buildErr}</div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Provenance footer */}
          <div className="text-[9.5px] text-muted/60 border-t border-border/30 pt-1">
            v{spec.version} · {spec.extraction.method} · extracted {spec.extraction.extracted_ts.slice(0, 19)}
            {spec.git_sha && <> · git {spec.git_sha.slice(0, 8)}</>}
          </div>
        </div>
      )}
    </Card>
  );
}
