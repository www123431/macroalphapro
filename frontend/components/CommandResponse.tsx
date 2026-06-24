"use client";

// CommandResponse — Render a chat command response as a rich card.
//
// One renderer per response.kind. Each renderer reuses styling
// tokens from the list pages so the chat surface stays visually
// consistent with the rest of the Lab workspace.

import Link from "next/link";
import {
  Atom, Network, TrendingDown, Layers, Lightbulb, ExternalLink,
  AlertCircle, CheckCircle2, Sigma, Bot,
} from "lucide-react";
import { CommandResponse } from "@/lib/chatCommands";
import { Badge, cn } from "@/components/ui";

interface RendererProps {
  resp: CommandResponse;
}


// ── Tone palettes (reused) ──────────────────────────────────────


const CONSENSUS_TONE: Record<string, string> = {
  APPROVE:         "bg-ok/15 text-ok",
  NEEDS_REVISION:  "bg-warn/15 text-warn",
  REJECT:          "bg-danger/15 text-danger",
};
const PURPOSE_TONE: Record<string, string> = {
  deployed_sleeve:      "bg-ok/15 text-ok",
  deploy_replacement:   "bg-ok/15 text-ok",
  hedge_replacement:    "bg-ok/15 text-ok",
  cousin_anchor:        "bg-info/15 text-info",
  candidate:            "bg-warn/15 text-warn",
};


// ── Renderers ───────────────────────────────────────────────────


function PfhSuggestionsCard({ resp }: RendererProps) {
  const out = resp.payload as any;
  const top = (out?.top || []) as any[];
  return (
    <div className="space-y-2">
      <Header icon={Atom} label={`PFH · ${out?.mode || "?"} · k=${top.length}`}
              meta={`base rate ${out?.base_rate_used?.toFixed(3) || "—"} · ${out?.n_candidates_total} candidates enumerated`} />
      {top.length === 0 ? (
        <div className="text-sm text-muted">No candidates returned.</div>
      ) : (
        <div className="rounded border border-border/30 overflow-hidden">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="bg-bg/50 text-left text-[10px] uppercase tracking-wider text-muted">
                <th className="px-2 py-1.5">#</th>
                <th className="px-2 py-1.5">factor</th>
                <th className="px-2 py-1.5">family</th>
                <th className="px-2 py-1.5 text-right">post.</th>
                <th className="px-2 py-1.5 text-right">CI</th>
                <th className="px-2 py-1.5"></th>
              </tr>
            </thead>
            <tbody>
              {top.map((s: any, i: number) => {
                const cid = s.proposal.candidate_id;
                return (
                  <tr key={cid} className="border-t border-muted/10">
                    <td className="px-2 py-1.5 text-muted">{i + 1}</td>
                    <td className="px-2 py-1.5 font-mono text-[11px]">
                      <Link href={`/lab/factor-lab/detail?id=${encodeURIComponent(cid)}`}
                            className="hover:text-accent">
                        <div>{s.proposal.universe}</div>
                        <div className="text-muted/70">× {s.proposal.signal_recipe} × {s.proposal.weighting}</div>
                      </Link>
                    </td>
                    <td className="px-2 py-1.5">
                      <Badge tone="bg-info/15 text-info">{s.proposal.family_normalized}</Badge>
                    </td>
                    <td className="px-2 py-1.5 text-right tnum">
                      {s.posterior.posterior_mean.toFixed(3)}
                    </td>
                    <td className="px-2 py-1.5 text-right tnum text-muted">
                      [{s.posterior.credible_05.toFixed(2)}, {s.posterior.credible_95.toFixed(2)}]
                    </td>
                    <td className="px-2 py-1.5">
                      <Link href={`/lab/factor-lab/detail?id=${encodeURIComponent(cid)}`}
                            className="text-accent text-[10px] inline-flex items-center gap-1 hover:underline">
                        open ↗
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}


function CouncilRunsCard({ resp }: RendererProps) {
  const out = resp.payload as any;
  const runs = (out?.runs || []) as any[];
  return (
    <div className="space-y-2">
      <Header icon={Network} label={`Council runs · ${runs.length}`}
              meta={out?.n != null ? `${out.n} total` : ""} />
      {runs.length === 0 ? (
        <div className="text-sm text-muted">No council runs found.</div>
      ) : (
        <div className="rounded border border-border/30 overflow-hidden">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="bg-bg/50 text-left text-[10px] uppercase tracking-wider text-muted">
                <th className="px-2 py-1.5">ts</th>
                <th className="px-2 py-1.5">run_id</th>
                <th className="px-2 py-1.5">proposal</th>
                <th className="px-2 py-1.5">consensus</th>
              </tr>
            </thead>
            <tbody>
              {runs.slice(0, 12).map((r: any) => (
                <tr key={r.run_id} className="border-t border-muted/10">
                  <td className="px-2 py-1.5 text-[10px] text-muted">{r.ts?.slice(0, 19)}</td>
                  <td className="px-2 py-1.5 font-mono text-[10px]">
                    <Link href={`/lab/council/detail?run_id=${encodeURIComponent(r.run_id)}`}
                          className="hover:text-accent">
                      {r.run_id?.slice(0, 12)}
                    </Link>
                  </td>
                  <td className="px-2 py-1.5 text-muted">{r.proposal?.title || "—"}</td>
                  <td className="px-2 py-1.5">
                    {r.consensus && (
                      <Badge tone={CONSENSUS_TONE[r.consensus] || "bg-muted/15 text-muted"}>
                        {r.consensus}
                      </Badge>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {runs.length > 12 && (
            <div className="px-2 py-1 text-[10px] text-muted text-center bg-bg/30">
              … and {runs.length - 12} more · <Link href="/lab/council" className="text-accent">open full list</Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


function DecayHistoryCard({ resp }: RendererProps) {
  const out = resp.payload as any;
  const rows = (out?.rows || []) as any[];
  // Group by sleeve, take latest
  const latest = new Map<string, any>();
  for (const r of rows) {
    if (!latest.has(r.sleeve)) latest.set(r.sleeve, r);
  }
  const sleeves = Array.from(latest.values());
  return (
    <div className="space-y-2">
      <Header icon={TrendingDown} label={`Decay · ${sleeves.length} sleeves`} />
      <div className="rounded border border-border/30 overflow-hidden">
        <table className="min-w-full text-xs">
          <thead>
            <tr className="bg-bg/50 text-left text-[10px] uppercase tracking-wider text-muted">
              <th className="px-2 py-1.5">sleeve</th>
              <th className="px-2 py-1.5">audit date</th>
              <th className="px-2 py-1.5 text-right">Sharpe</th>
              <th className="px-2 py-1.5">alert</th>
            </tr>
          </thead>
          <tbody>
            {sleeves.map((r: any) => (
              <tr key={r.sleeve} className="border-t border-muted/10">
                <td className="px-2 py-1.5 font-mono">
                  <Link href={`/research/decay/detail?sleeve=${encodeURIComponent(r.sleeve)}`}
                        className="hover:text-accent">{r.sleeve}</Link>
                </td>
                <td className="px-2 py-1.5 text-muted">{r.audit_date}</td>
                <td className="px-2 py-1.5 text-right tnum">
                  {r.trailing_sharpe != null ? r.trailing_sharpe.toFixed(3) : "—"}
                </td>
                <td className="px-2 py-1.5">
                  <Badge tone={(r.alert_level || "OK") === "OK" ? "bg-ok/15 text-ok" : "bg-warn/15 text-warn"}>
                    {r.alert_level || "OK"}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


function FactorDetailCard({ resp }: RendererProps) {
  const out = resp.payload as any;
  const post = out?.posterior_context;
  return (
    <div className="space-y-2">
      <Header icon={Atom} label={`Factor · ${out?.spec_id}`}
              meta={`${out?.spec_kind} · ${out?.materializations?.length || 0} materializations`} />
      {post && (
        <div className="rounded border border-border/30 p-3 bg-bg/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1.5 inline-flex items-center gap-1">
            <Sigma className="h-3 w-3" /> Bayesian posterior · family {post.family}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <KV label="GREEN" value={`${post.n_green}`} accent="text-ok" />
            <KV label="RED"   value={`${post.n_red}`}   accent="text-danger" />
            <KV label="post mean" value={post.posterior_mean.toFixed(3)} />
            <KV label="90% CI" value={`[${post.credible_05.toFixed(2)}, ${post.credible_95.toFixed(2)}]`} />
          </div>
        </div>
      )}
      <Link href={`/lab/factor-lab/detail?id=${encodeURIComponent(out?.spec_id)}`}
            className="inline-flex items-center gap-1 text-xs text-accent hover:underline">
        open full factor detail <ExternalLink className="h-3 w-3" />
      </Link>
    </div>
  );
}


function NavigationCard({ resp }: RendererProps) {
  const p = resp.payload as any;
  return (
    <div className="rounded border border-info/20 bg-info/5 p-3">
      <div className="inline-flex items-center gap-1.5 text-sm">
        <ExternalLink className="h-3.5 w-3.5 text-info" />
        Navigated to <Link href={p.url} className="text-accent hover:underline">{p.label}</Link>
      </div>
    </div>
  );
}


function ChainCatalogueCard({ resp }: RendererProps) {
  const out = resp.payload as any;
  const chains = (out?.chains || []) as any[];
  return (
    <div className="space-y-2">
      <Header icon={Lightbulb} label={`Chains · ${chains.length}`} />
      <div className="space-y-1.5">
        {chains.map((c: any) => (
          <div key={c.chain_id} className="rounded border border-border/30 p-3 hover:bg-muted/5 transition-colors">
            <div className="flex items-baseline justify-between gap-2 mb-1">
              <span className="font-mono text-sm">{c.chain_id}</span>
              <Badge tone="bg-info/15 text-info">{c.n_steps} steps</Badge>
            </div>
            <div className="text-[11px] text-muted leading-relaxed">{c.description}</div>
          </div>
        ))}
      </div>
    </div>
  );
}


function HelpCard({ resp }: RendererProps) {
  const commands = (resp.payload?.commands || []) as any[];
  const groups: Record<string, any[]> = {};
  for (const c of commands) {
    (groups[c.category] ||= []).push(c);
  }
  return (
    <div className="space-y-3">
      <Header icon={Bot} label="Available commands" />
      {Object.entries(groups).map(([cat, cmds]) => (
        <div key={cat}>
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1.5">{cat}</div>
          <div className="space-y-1">
            {cmds.map((c) => (
              <div key={c.slug} className="text-xs font-mono">
                <span className="text-accent">/{c.slug}</span>
                <span className="text-muted/60 ml-2">{c.description}</span>
                <div className="text-[10px] text-muted/50 ml-4">{c.usage}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}


function ErrorCard({ resp }: RendererProps) {
  const message = resp.payload?.message || "An error occurred.";
  return (
    <div className="rounded border border-danger/30 bg-danger/5 p-3">
      <div className="inline-flex items-start gap-1.5 text-sm text-danger">
        <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
        <span>{message}</span>
      </div>
    </div>
  );
}


function AskAnswerCard({ resp }: RendererProps) {
  const p = resp.payload as any;
  const answer: string = p?.answer || "";
  const citations: Array<{type: string; id: string}> = p?.citations || [];

  // Citation type → URL builder
  const cite_url = (type: string, id: string): string => {
    switch (type) {
      case "run_id":       return `/lab/council/detail?run_id=${encodeURIComponent(id)}`;
      case "iteration_id": return `/lab/l4/detail?id=${encodeURIComponent(id)}`;
      case "spec_id":      return `/lab/factor-lab/detail?id=${encodeURIComponent(id)}`;
      case "sleeve":       return `/research/decay/detail?sleeve=${encodeURIComponent(id)}`;
      default:             return "#";
    }
  };

  // Replace [type:id] markers in answer with inline links
  const citation_re = /\[(run_id|iteration_id|spec_id|sleeve):([a-zA-Z0-9_\-]+)\]/g;
  const parts: Array<string | { type: string; id: string }> = [];
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  while ((m = citation_re.exec(answer)) !== null) {
    if (m.index > lastIdx) {
      parts.push(answer.slice(lastIdx, m.index));
    }
    parts.push({ type: m[1], id: m[2] });
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < answer.length) parts.push(answer.slice(lastIdx));

  const retrievalMode: string | undefined = p?.retrieval_mode;
  const semanticBadge = retrievalMode === "semantic+keyword"
    ? { label: "semantic + keyword", tone: "text-ok bg-ok/10" }
    : { label: "keyword only", tone: "text-muted bg-muted/10" };

  return (
    <div className="space-y-2">
      <Header icon={Bot} label="Scoped answer"
              meta={`model ${p?.model || "?"} · ${p?.elapsed_s}s`} />
      <div className="rounded border border-border/30 p-3 text-sm text-foreground leading-relaxed whitespace-pre-wrap">
        {parts.map((part, i) => {
          if (typeof part === "string") return <span key={i}>{part}</span>;
          return (
            <Link key={i} href={cite_url(part.type, part.id)}
                  className="inline-flex items-baseline gap-0.5 mx-0.5 px-1 py-0
                             rounded bg-accent/10 text-accent hover:bg-accent/20
                             text-[12px] font-mono">
              {part.type}:{part.id.slice(0, 14)}
              <ExternalLink className="h-2.5 w-2.5 opacity-60" />
            </Link>
          );
        })}
      </div>
      <div className="flex items-center gap-2 text-[10px] text-muted/70">
        {retrievalMode && (
          <span className={`inline-flex items-center px-1.5 py-0.5 rounded font-mono ${semanticBadge.tone}`}
                title="semantic = MiniLM vector recall over ledger snippets; keyword = recency + token overlap">
            retrieval: {semanticBadge.label}
          </span>
        )}
        {citations.length > 0 && (
          <span>
            {citations.length} citation{citations.length === 1 ? "" : "s"} ·{" "}
            {Object.entries(p?.n_context_rows || {})
              .map(([k, v]) => `${k}=${v}`).join(" · ")}
          </span>
        )}
      </div>
    </div>
  );
}


function TextCard({ resp }: RendererProps) {
  const p = resp.payload as any;
  return (
    <div className="rounded border border-border/30 p-3 text-sm text-foreground leading-relaxed">
      {p.message}
      {p.question && (
        <div className="text-[10px] text-muted/60 mt-2">
          (Question: <span className="italic">{p.question}</span>)
        </div>
      )}
    </div>
  );
}


// ── Shared subcomponents ────────────────────────────────────────


function Header({
  icon: Icon, label, meta,
}: {
  icon: any; label: string; meta?: string;
}) {
  return (
    <div className="flex items-baseline justify-between">
      <div className="inline-flex items-center gap-1.5 text-xs font-medium">
        <Icon className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
        {label}
      </div>
      {meta && <span className="text-[10px] text-muted/70">{meta}</span>}
    </div>
  );
}

function KV({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("text-sm tnum mt-0.5", accent || "text-foreground")}>{value}</div>
    </div>
  );
}


// ── Public dispatcher ──────────────────────────────────────────


export function CommandResponseRenderer({ resp }: { resp: CommandResponse }) {
  switch (resp.kind) {
    case "pfh_suggestions":  return <PfhSuggestionsCard resp={resp} />;
    case "council_run_result": return <CouncilRunsCard resp={resp} />;
    case "decay_history":    return <DecayHistoryCard resp={resp} />;
    case "factor_detail_card": return <FactorDetailCard resp={resp} />;
    case "chain_catalogue":  return <ChainCatalogueCard resp={resp} />;
    case "ask_answer":       return <AskAnswerCard resp={resp} />;
    case "navigation":       return <NavigationCard resp={resp} />;
    case "help":             return <HelpCard resp={resp} />;
    case "error":            return <ErrorCard resp={resp} />;
    case "text":             return <TextCard resp={resp} />;
    default:                 return <div className="text-sm text-muted">Unhandled response kind: {resp.kind}</div>;
  }
}
