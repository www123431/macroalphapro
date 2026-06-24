"use client";

// Phase 2.0 step 15 UI — chief_of_staff weekly session trigger.
// Lives on /dashboard right under DailyDirective.
//
// 2 buttons: Preview (dry-run, no events emitted / no hypotheses written)
//   and Run now (full session: D → A → B → memo → emit).
//
// After a run, render:
//   - memo headline + bullets + next focus (Sonnet's 5-bullet output)
//   - substep counts (D emitted / A candidates / B reviewed / B pending)
//   - audit event_id link
//   - errors if any
//
// Cost: <$0.10 dry-run, <$0.20 full run. User-initiated only — never
// polled.

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Loader2, AlertTriangle, Eye, Send, FileText, Activity,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { useI18n } from "@/lib/i18n";


type WeeklySessionResponse = {
  session_id: string;
  run_ts: string;
  dry_run: boolean;
  d_result:   { n_events_scanned: number; n_hits_total: number; n_hits_fresh: number; n_emitted: number; event_ids: string[]; errors: string[] };
  a_result:   { snapshot: { recent_summaries: number; deployed_sleeves: number; recent_events: number; doctrine_snippets: number }; n_candidates: number; n_written: number; written_hypothesis_ids: string[]; candidates: unknown[]; errors: string[]; event_id: string | null };
  b_result:   { n_candidates: number; n_reviewed: number; n_persisted: number; verdicts: unknown[]; errors: string[] };
  session_event_id: string | null;
  errors: string[];
  d_emitted: number;
  a_n_candidates: number;
  a_n_written: number;
  b_n_reviewed: number;
  b_n_pending_approval: number;
  memo: {
    session_id: string;
    headline: string;
    bullets: string[];
    whats_next: string;
    generated_ts: string;
    model: string;
  } | null;
};


export function ChiefOfStaffPanel() {
  const { t } = useI18n();
  const [running, setRunning] = useState(false);
  const [result, setResult]   = useState<WeeklySessionResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);

  const fire = async (dryRun: boolean) => {
    setRunning(true);
    setError(null);
    try {
      const r = await fetch(`${API_BASE}/api/chief_of_staff/run`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ dry_run: dryRun }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body: WeeklySessionResponse = await r.json();
      setResult(body);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <Card className="p-0 overflow-hidden border border-accent/25">
      <div className="px-3 py-2 border-b border-border/30 bg-accent/5 flex items-center gap-2">
        <Activity className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
        <span className="text-[12px] font-semibold text-foreground">
          {t("cos.title")}
        </span>
        <span className="text-[10px] text-muted/60 ml-auto">
          {t("cos.cost_hint")}
        </span>
      </div>

      <div className="px-3 py-3 space-y-2">
        <p className="text-[10.5px] text-muted/80 leading-snug">
          {t("cos.subtitle")}
        </p>

        <div className="flex items-center gap-2 pt-1">
          <button
            onClick={() => fire(true)}
            disabled={running}
            className={cn(
              "px-2.5 py-1.5 rounded text-[10.5px] font-medium inline-flex items-center gap-1.5",
              "border border-border/40 hover:border-accent/50 hover:bg-accent/5",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}>
            {running ? <Loader2 className="h-3 w-3 animate-spin" /> :
                        <Eye className="h-3 w-3" />}
            {running ? t("cos.btn.running") : t("cos.btn.preview")}
          </button>
          <button
            onClick={() => fire(false)}
            disabled={running}
            className={cn(
              "px-2.5 py-1.5 rounded text-[10.5px] font-medium inline-flex items-center gap-1.5",
              "bg-accent/10 text-accent border border-accent/40 hover:bg-accent/20",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}>
            {running ? <Loader2 className="h-3 w-3 animate-spin" /> :
                        <Send className="h-3 w-3" />}
            {running ? t("cos.btn.running") : t("cos.btn.run_now")}
          </button>
        </div>

        {error && (
          <div className="text-[10.5px] text-danger inline-flex items-center gap-1.5">
            <AlertTriangle className="h-3 w-3" /> {error}
          </div>
        )}

        {result && <ResultBody result={result} />}
      </div>
    </Card>
  );
}


function ResultBody({ result }: { result: WeeklySessionResponse }) {
  const { t } = useI18n();

  // Compact 4-cell counts strip
  const Strip = (
    <div className="grid grid-cols-4 gap-1 text-[10.5px]">
      <div className="px-2 py-1.5 bg-panel2/40 rounded">
        <div className="text-muted/60 text-[9px] uppercase tracking-wide">D</div>
        <div className="font-mono">{result.d_emitted} {t("cos.label.emitted")}</div>
      </div>
      <div className="px-2 py-1.5 bg-panel2/40 rounded">
        <div className="text-muted/60 text-[9px] uppercase tracking-wide">A</div>
        <div className="font-mono">{result.a_n_candidates} {t("cos.label.cands")}</div>
        <div className="text-[9px] text-muted/70 font-mono">{result.a_n_written} {t("cos.label.written")}</div>
      </div>
      <div className="px-2 py-1.5 bg-panel2/40 rounded">
        <div className="text-muted/60 text-[9px] uppercase tracking-wide">B</div>
        <div className="font-mono">{result.b_n_reviewed} {t("cos.label.reviewed")}</div>
      </div>
      <div className="px-2 py-1.5 bg-panel2/40 rounded">
        <div className="text-muted/60 text-[9px] uppercase tracking-wide">{t("cos.label.queue")}</div>
        <div className="font-mono">{result.b_n_pending_approval}</div>
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-2 pt-1">
      <div className="text-[10px] text-muted/70 inline-flex items-center gap-2">
        <span className="font-mono">{result.session_id}</span>
        <span>·</span>
        <span>{result.dry_run ? t("cos.label.dry_run") : t("cos.label.persisted")}</span>
        {result.session_event_id && (
          <>
            <span>·</span>
            <span className="font-mono text-muted/50">
              {result.session_event_id.slice(0, 8)}
            </span>
          </>
        )}
      </div>

      {Strip}

      {result.errors.length > 0 && (
        <div className="text-[10.5px] text-danger px-2 py-1.5 bg-danger/5 border border-danger/30 rounded">
          <span className="font-semibold">{t("cos.label.errors")}</span>{" "}
          {result.errors.join("; ")}
        </div>
      )}

      {result.memo && (
        <div className="border border-accent/20 bg-accent/[0.04] rounded p-2.5 space-y-1.5">
          <div className="inline-flex items-center gap-1.5 text-[10px] text-accent/80 uppercase tracking-wide">
            <FileText className="h-3 w-3" strokeWidth={2.2} />
            {t("cos.memo.title")}
          </div>
          <div className="text-[12px] font-medium leading-snug">
            {result.memo.headline}
          </div>
          <ol className="text-[10.5px] text-muted/85 leading-snug space-y-1 pl-4 list-decimal">
            {result.memo.bullets.map((b, i) => (
              <li key={i}>{b}</li>
            ))}
          </ol>
          {result.memo.whats_next && (
            <div className="text-[10.5px] text-foreground/85 pt-1 border-t border-border/30">
              <span className="text-muted/60">{t("cos.memo.next")}</span>{" "}
              {result.memo.whats_next}
            </div>
          )}
        </div>
      )}

      {/* honest-empty case when no memo (dry-run or memo failed) */}
      {!result.memo && result.errors.length === 0 && (
        <div className="text-[10.5px] text-muted/70 px-2 py-1.5 bg-panel2/30 rounded">
          {result.dry_run
            ? t("cos.empty.dry_run")
            : t("cos.empty.no_memo")}
        </div>
      )}
    </motion.div>
  );
}
