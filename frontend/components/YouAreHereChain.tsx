"use client";

// YouAreHereChain — N2 "你在链上哪一步" 微型定位器.
//
// 出现在每个 page ModeHeader 右侧（NextClickHint 旁边）。显示这条研究
// 主链的 6 个节点：
//
//   PAPER → HYPOTHESIS → FORWARD → CANDIDATE → VERDICT → LIBRARY
//
// 当前 page 对应的节点高亮、其余节点淡灰。点节点 = 跳到那个 page。
// 比 NextClickHint 退一步：NextClickHint 告诉你"现在该做什么"，
// YouAreHereChain 告诉你"在更大的图里你在哪"。
//
// 纯静态（没有 LLM、没有 state poll）。每个 page 的链上位置写死在
// 一个 Map 里 — 路径前缀 → 节点 index。

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  FileText, Lightbulb, ListChecks, Atom, CheckCircle2, BookMarked,
} from "lucide-react";
import { cn } from "@/components/ui";


type ChainNode = {
  id:    string;
  short: string;        // 短名（中文）
  full:  string;        // 完整名（hover tooltip）
  href:  string;        // 主要跳转
  icon:  React.ComponentType<{ className?: string; strokeWidth?: number }>;
};


// 主链 6 节点。顺序就是 doctrine PAPER→HYPOTHESIS→TEST→VERDICT 的延展。
// Labels kept in English so the chain reads consistently with code-level
// terms (subject_type / event_type / library YAML), and so the chip
// stays narrow when stacked next to NextClickHint + HelpOnThisPage.
const CHAIN: ChainNode[] = [
  { id: "paper",      short: "Paper",     full: "Paper · ingested research papers",      href: "/research/papers",    icon: FileText },
  { id: "hypothesis", short: "Hypothesis", full: "Hypothesis · paper-grounded claims",   href: "/research/papers",    icon: Lightbulb },
  { id: "forward",    short: "Forward",   full: "Forward · untested queue, PM-reviewable", href: "/research/forward", icon: ListChecks },
  { id: "candidate",  short: "Candidate", full: "Candidate · strict-gate pipeline run",  href: "/research/candidate", icon: Atom },
  { id: "verdict",    short: "Verdict",   full: "Verdict · GREEN / MARGINAL / RED lessons", href: "/research/lessons", icon: CheckCircle2 },
  { id: "library",    short: "Library",   full: "Library · deployed sleeves + ideation",  href: "/research/library",      icon: BookMarked },
];


// 路径前缀 → 当前在链上第几个节点（0..5）。
// 不在链上的 page（/dashboard, /lab/cockpit, /inbox 等）返回 -1
// → 整条 chip 不渲染（避免误导）。
function _activeIndexFor(pathname: string): number {
  if (pathname.startsWith("/research/papers"))    return pathname.includes("/new") ? 0 : 0;
  if (pathname.startsWith("/research/forward"))   return 2;
  if (pathname.startsWith("/research/enhance"))   return 3;  // workspace = candidate stage
  if (pathname.startsWith("/research/candidate")) return 3;
  if (pathname.startsWith("/research/lessons"))   return 4;
  if (pathname.startsWith("/research/library"))        return 5;
  return -1;
}


export function YouAreHereChain() {
  const pathname = usePathname() || "/";
  const active = _activeIndexFor(pathname);
  if (active < 0) return null;

  return (
    <div className="inline-flex items-center gap-0.5 rounded border border-border/40 bg-panel2/30 px-1.5 py-1"
         title="研究主链 · 当前你在哪一步 · 点节点跳转">
      {CHAIN.map((n, i) => {
        const isActive  = i === active;
        const isVisited = i < active;
        const Icon = n.icon;
        return (
          <div key={n.id} className="inline-flex items-center">
            <Link
              href={n.href}
              title={n.full}
              className={cn(
                "inline-flex items-center gap-0.5 rounded px-1 py-0.5 text-[10px] transition-colors",
                isActive
                  ? "bg-accent/20 text-accent font-semibold"
                  : isVisited
                    ? "text-foreground/70 hover:text-accent"
                    : "text-muted/50 hover:text-foreground/70",
              )}>
              <Icon className="h-2.5 w-2.5 shrink-0" strokeWidth={2.2} />
              <span className="leading-none">{n.short}</span>
            </Link>
            {i < CHAIN.length - 1 && (
              <span className={cn(
                "px-0.5 leading-none",
                i < active ? "text-foreground/40" : "text-muted/30",
              )}>›</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
