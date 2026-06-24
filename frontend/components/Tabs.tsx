"use client";

// Tabs — reusable headless tab strip + body slot.
//
// Used to replace stacked collapsible accordions (R2.3 audit
// finding: accordion fatigue when ≥ 3 sections compete for the
// same vertical real-estate). A tab strip:
//
//   - keeps one section visible at a time → zero state to remember
//   - signals the workspace's mental zones at a glance
//   - matches institutional terminal convention (Bloomberg / Citadel
//     monitor + log + analytics tabs)
//
// Headless: each tab carries a `label`, optional `count` chip,
// optional `icon`, and a body render function. The hosting page
// supplies the tab descriptors; this component owns the active-key
// state. URL-syncing variant available via `urlParam` prop — when
// set, the active tab tracks `?{urlParam}=…` and pushes on change
// so deep-links survive reloads + the back button.

import { ReactNode, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { cn } from "@/components/ui";


export type TabDef = {
  key:    string;
  label:  string;
  hint?:  string;
  icon?:  React.ComponentType<{ className?: string; strokeWidth?: number }>;
  count?: number | string;
  body:   () => ReactNode;
};


export function Tabs({
  tabs,
  defaultKey,
  urlParam,
  className,
}: {
  tabs:       TabDef[];
  defaultKey?: string;
  urlParam?:  string;   // when set, sync active tab to ?{urlParam}=…
  className?: string;
}) {
  const router       = useRouter();
  const searchParams = useSearchParams();

  // Resolve initial active key: URL > default > first
  const urlKey = urlParam ? (searchParams?.get(urlParam) ?? null) : null;
  const initial = urlKey && tabs.some((t) => t.key === urlKey)
    ? urlKey
    : (defaultKey && tabs.some((t) => t.key === defaultKey) ? defaultKey : tabs[0]?.key);
  const [activeKey, setActiveKey] = useState<string>(initial);

  // Pick up URL changes (back/forward, deep-link) AFTER mount.
  useEffect(() => {
    if (!urlParam) return;
    const fromUrl = searchParams?.get(urlParam);
    if (fromUrl && fromUrl !== activeKey && tabs.some((t) => t.key === fromUrl)) {
      setActiveKey(fromUrl);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, urlParam]);

  const onChange = useCallback((next: string) => {
    setActiveKey(next);
    if (!urlParam) return;
    // Update URL without scroll-jumping. shallowly replace so back-button
    // still works between tabs (each click pushes one history entry).
    const params = new URLSearchParams(searchParams?.toString() ?? "");
    params.set(urlParam, next);
    router.push(`?${params.toString()}`, { scroll: false });
  }, [urlParam, searchParams, router]);

  const active = tabs.find((t) => t.key === activeKey) ?? tabs[0];

  return (
    <div className={className}>
      {/* Tab strip */}
      <div className="flex items-center gap-1 border-b border-border/40 overflow-x-auto no-scrollbar">
        {tabs.map((t) => {
          const isActive = t.key === active?.key;
          const Icon = t.icon;
          return (
            <button key={t.key}
              onClick={() => onChange(t.key)}
              title={t.hint}
              className={cn(
                "shrink-0 inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium",
                "rounded-t border-b-2 transition-colors",
                isActive
                  ? "text-accent border-accent bg-accent/[0.06]"
                  : "text-muted border-transparent hover:text-foreground hover:bg-panel2/30",
              )}>
              {Icon && <Icon className="h-3.5 w-3.5" strokeWidth={2} />}
              <span>{t.label}</span>
              {t.count !== undefined && t.count !== "" && (
                <span className={cn("tnum text-[10px] px-1 rounded",
                  isActive ? "bg-accent/15 text-accent" : "bg-muted/10 text-muted")}>
                  {t.count}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Body */}
      <div className="pt-3">
        {active?.body()}
      </div>
    </div>
  );
}


// SegmentToggle — sibling primitive for a 2-way (or N-way) switch
// inside a single tab body. Lighter than Tabs (no URL syncing, no
// keyboard nav scaffolding), used when the choice is a "view of the
// same data" rather than a separate workspace section.

export function SegmentToggle<K extends string>({
  options, value, onChange, className,
}: {
  options:  { key: K; label: string; hint?: string }[];
  value:    K;
  onChange: (k: K) => void;
  className?: string;
}) {
  return (
    <div className={cn(
      "inline-flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/30 p-0.5",
      className,
    )}>
      {options.map((o) => (
        <button key={o.key}
          onClick={() => onChange(o.key)}
          title={o.hint}
          className={cn(
            "rounded px-2 py-0.5 text-[11px] transition-colors",
            o.key === value
              ? "bg-accent/15 text-accent font-semibold"
              : "text-muted hover:text-foreground",
          )}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
