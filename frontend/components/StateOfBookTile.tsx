"use client";

// StateOfBookTile — N1 "Book 当日简报".
//
// Top of /dashboard, ABOVE DailyDirective. 3-paragraph Chinese memo
// auto-generated daily by engine.agents.daily_memo: Book health +
// research pipeline + watch items. Anchors the "global picture" that
// the rest of the per-page UI couldn't give the user.
//
// One LLM call per day (cached); ~$1/month. Manual "regenerate" button
// for testing or when the user wants a fresh take mid-day.

import { useEffect, useState } from "react";
import { Newspaper, RefreshCw, AlertCircle, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { renderAnswerMarkdown } from "@/lib/renderAnswer";
import { AgentMessage } from "@/components/AgentMessage";


type Memo = {
  date_key:     string;
  generated_ts: string;
  markdown:     string;
  n_citations:  number;
  model?:       string | null;
  elapsed_s:    number;
  from_cache:   boolean;
  error?:       string | null;
};


function _fmtTimeAgo(ts: string): string {
  const ms = Date.now() - Date.parse(ts.endsWith("Z") ? ts : ts + "Z");
  if (!Number.isFinite(ms) || ms < 0) return "刚刚";
  const s = Math.floor(ms / 1000);
  if (s < 60)   return `${s} 秒前`;
  const m = Math.floor(s / 60);
  if (m < 60)   return `${m} 分钟前`;
  const h = Math.floor(m / 60);
  if (h < 24)   return `${h} 小时前`;
  const d = Math.floor(h / 24);
  return `${d} 天前`;
}


export function StateOfBookTile() {
  const [memo, setMemo] = useState<Memo | null>(null);
  const [loading, setLoading] = useState(true);
  const [regenerating, setRegenerating] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = async (force = false) => {
    if (force) setRegenerating(true);
    else setLoading(true);
    setErr(null);
    try {
      const url = force
        ? `${API_BASE}/api/agents/state_of_book?force=true`
        : `${API_BASE}/api/agents/state_of_book`;
      const r = await fetch(url, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json() as Memo;
      setMemo(data);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setLoading(false);
      setRegenerating(false);
    }
  };

  useEffect(() => { load(false); }, []);

  if (loading && !memo) {
    return (
      <Card className="p-3">
        <div className="flex items-center gap-2 text-[11px] text-muted">
          <Loader2 className="h-3 w-3 animate-spin" />
          正在生成今日《Book 当日简报》…（首次约 15-20 秒）
        </div>
      </Card>
    );
  }

  if (!memo) {
    return (
      <Card className="p-3 border-danger/30 bg-danger/[0.04]">
        <div className="flex items-center gap-2 text-[11px] text-danger">
          <AlertCircle className="h-3 w-3" />
          简报加载失败 {err ? `· ${err}` : ""}
        </div>
      </Card>
    );
  }

  const hasError = Boolean(memo.error);

  return (
    <AgentMessage
      agentId    = "daily_memo"
      agentLabel = "Book 当日简报"
      kind       = "informational"
      icon       = {Newspaper}
      title      = {`${memo.date_key}`}
      subtitle   = {
        <>
          机构级 Chief of Staff 视角 · {memo.n_citations} 处引用
          {memo.elapsed_s > 0 && ` · 用时 ${memo.elapsed_s.toFixed(1)}s`}
          {memo.model && ` · ${memo.model}`}
        </>
      }
      generatedTs = {memo.generated_ts}
      rightSlot  = {
        <button
          onClick={() => load(true)}
          disabled={regenerating}
          title="重新生成（消耗一次 Claude 调用，约 $0.03）"
          className="text-muted hover:text-accent disabled:opacity-50 transition-colors">
          <RefreshCw className={cn("h-3.5 w-3.5", regenerating && "animate-spin")} />
        </button>
      }
      footer = {
        memo.from_cache
          ? "本日简报已生成；点右上角 ↻ 可重新生成（消耗一次 LLM 调用）。"
          : "本简报由 11 个 RAG 源生成 · 每日缓存一次 · 引用可点击跳转。"
      }>
      {hasError ? (
        <div className="text-[12px] text-danger">
          生成失败：{memo.error}
        </div>
      ) : (
        <div className="text-[12.5px] leading-relaxed text-foreground/90 [&_h2]:text-[12.5px] [&_h2]:font-semibold [&_h2]:mt-3 [&_h2]:mb-1 [&_h2]:text-accent/90 [&_p]:my-1 [&_ol]:my-1 [&_ol]:pl-5 [&_ol]:list-decimal [&_li]:my-0.5 [&_ul]:my-1 [&_ul]:pl-5 [&_ul]:list-disc [&_code]:bg-panel2/40 [&_code]:px-1 [&_code]:rounded [&_code]:text-[11px]">
          {renderAnswerMarkdown(memo.markdown, { citationFontSize: "text-[10.5px]" })}
        </div>
      )}
    </AgentMessage>
  );
}
