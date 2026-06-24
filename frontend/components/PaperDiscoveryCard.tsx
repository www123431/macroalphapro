// frontend/components/PaperDiscoveryCard.tsx — Paper Discovery section for
// the Research page. Owns: nominate input + bookmarklet + review/borderline
// queues. Wraps /api/research/discovery/* endpoints behind react-query hooks.
//
// Senior 2026-05-30: this lives inside the existing Research page, NOT a
// separate Discovery tab — the complete paper-→-gate pipeline lives in
// Research conceptually.
"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { BookOpen, Plus, AlertCircle, Bookmark, ArrowRight, X } from "lucide-react";
import type { DiscoveryQueueEntry, DiscoveryRouting } from "@/lib/api";
import {
  useDiscoveryQueues, useDiscoveryBookmarklet, useDiscoveryNominate,
  useDiscoveryPromote, useDiscoverySkip,
} from "@/lib/queries";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";

function confidenceValue(entry: DiscoveryQueueEntry): number | null {
  if (entry.routing?.adjusted_confidence != null) return entry.routing.adjusted_confidence;
  const c = entry.confidence;
  if (typeof c === "number") return c;
  if (c && typeof c === "object" && "confidence" in c) {
    const v = (c as Record<string, unknown>).confidence;
    if (typeof v === "number") return v;
  }
  return null;
}

function familyOf(routing: DiscoveryRouting | undefined): string {
  return routing?.family ?? "?";
}

function QueueRow({ entry, tier }: { entry: DiscoveryQueueEntry; tier: "review" | "borderline" }) {
  const conf = confidenceValue(entry);
  const family = familyOf(entry.routing);
  const venue = entry.venue || entry.source || "unknown";
  const ts = (entry.ts ?? "").slice(0, 16).replace("T", " ");
  const href = entry.doi
    ? `https://doi.org/${entry.doi}`
    : entry.ident_type === "arxiv" && entry.source_id
    ? `https://arxiv.org/abs/${entry.source_id}`
    : entry.ident_type === "openalex" && entry.source_id
    ? `https://openalex.org/${entry.source_id}`
    : null;

  const promote = useDiscoveryPromote();
  const skip = useDiscoverySkip();
  const busy = promote.isPending || skip.isPending;
  const sourceId = entry.source_id ?? "";

  return (
    <motion.div
      variants={fadeUp}
      className={cn(
        "rounded-md border-l-2 border bg-panel/40 px-3 py-2.5",
        tier === "review" ? "border-l-ok/70 border-border" : "border-l-warn/70 border-border",
        busy && "opacity-60",
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1 text-sm font-medium leading-snug">
          {href ? (
            <a href={href} target="_blank" rel="noopener noreferrer" className="hover:underline">
              {entry.title || "(untitled)"}
            </a>
          ) : (
            entry.title || "(untitled)"
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            disabled={busy || !sourceId}
            onClick={() => sourceId && promote.mutate(sourceId)}
            title="Promote to mechanism library as PENDING stub"
            className="inline-flex items-center gap-1 rounded border border-ok/40 bg-ok/10 px-2 py-0.5 text-[11px] font-medium text-ok transition-colors hover:bg-ok/20 disabled:opacity-40"
          >
            <ArrowRight className="h-3 w-3" /> Promote
          </button>
          <button
            type="button"
            disabled={busy || !sourceId}
            onClick={() => sourceId && skip.mutate({ sourceId })}
            title="Skip (write to discovery_rejected.jsonl)"
            className="inline-flex items-center gap-1 rounded border border-border bg-panel/60 px-2 py-0.5 text-[11px] text-muted transition-colors hover:bg-alert/10 hover:text-alert disabled:opacity-40"
          >
            <X className="h-3 w-3" /> Skip
          </button>
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <Badge tone="bg-slate-700/40 text-slate-300" className="font-mono">{venue}</Badge>
        <Badge tone="bg-slate-700/40 text-slate-300" className="font-mono">family={family}</Badge>
        <Badge
          tone={
            conf == null
              ? "bg-slate-700/40 text-slate-300"
              : conf >= 0.5
              ? "bg-ok/15 text-ok"
              : conf >= 0.3
              ? "bg-warn/15 text-warn"
              : "bg-alert/15 text-alert"
          }
          className="tnum"
        >
          conf={conf == null ? "n/a" : conf.toFixed(2)}
        </Badge>
        {entry.routing?.family_bonus_applied && (
          <Badge tone="bg-accent/15 text-accent">family bonus</Badge>
        )}
        {entry.scoring_method === "venue_tier_fallback" && (
          <Badge tone="bg-accent/15 text-accent">venue tier</Badge>
        )}
        {entry.llm_rescue?.hybrid_confidence != null && (
          <span title={`LLM rescued ${entry.llm_rescue.rescued_features?.length ?? 0} features`}>
            <Badge tone="bg-accent/15 text-accent">
              LLM rescue
            </Badge>
          </span>
        )}
        {entry.meta_learner_advisory?.prior_pass_probability != null && (
          <span title={entry.meta_learner_advisory.advisory_note}>
            <Badge tone="bg-slate-700/40 text-slate-300">
              prior={(entry.meta_learner_advisory.prior_pass_probability * 100).toFixed(0)}%
            </Badge>
          </span>
        )}
        {entry.nominated_via && (
          <Badge tone="bg-slate-700/40 text-slate-300">via {entry.nominated_via}</Badge>
        )}
        <span className="tnum ml-auto text-muted/60">{ts}</span>
      </div>
      {(promote.error || skip.error) && (
        <div className="mt-1 text-[10px] text-alert">
          {(promote.error instanceof Error ? promote.error.message : null) ||
            (skip.error instanceof Error ? skip.error.message : null)}
        </div>
      )}
    </motion.div>
  );
}

function NominateForm() {
  const [url, setUrl] = useState("");
  const nominate = useDiscoveryNominate();
  const last = nominate.data;
  const err = nominate.error instanceof Error ? nominate.error.message : null;

  return (
    <form
      className="flex flex-col gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        if (url.trim()) {
          nominate.mutate(url.trim(), {
            onSuccess: () => setUrl(""),
          });
        }
      }}
    >
      <div className="flex gap-2">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste DOI / arxiv URL / OpenAlex Work ID / SSRN URL"
          className="flex-1 rounded-md border border-border bg-panel/60 px-3 py-2 text-sm font-mono placeholder:text-muted/60 focus:border-accent/60 focus:outline-none"
          disabled={nominate.isPending}
        />
        <button
          type="submit"
          disabled={nominate.isPending || !url.trim()}
          className="inline-flex items-center gap-1.5 rounded-md border border-accent/50 bg-accent/10 px-3 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/20 disabled:opacity-50"
        >
          <Plus className="h-3.5 w-3.5" />
          {nominate.isPending ? "Adding..." : "Add"}
        </button>
      </div>
      {last?.ok && (
        <div className="rounded-md border border-ok/20 bg-ok/10 px-3 py-2 text-xs text-ok">
          Added <span className="font-medium">{last.title?.slice(0, 80)}</span>
          <span className="ml-2 text-muted">
            -- conf={last.confidence?.toFixed(2)}, routing={last.routing}
            {last.scoring_method === "venue_tier_fallback" && " (venue tier)"}
          </span>
        </div>
      )}
      {last?.error && (
        <div className="rounded-md border border-warn/20 bg-warn/10 px-3 py-2 text-xs text-warn">
          <AlertCircle className="mr-1 inline h-3 w-3" /> {last.error}
        </div>
      )}
      {err && (
        <div className="rounded-md border border-alert/20 bg-alert/10 px-3 py-2 text-xs text-alert">
          <AlertCircle className="mr-1 inline h-3 w-3" /> {err}
        </div>
      )}
    </form>
  );
}

function BookmarkletBlock() {
  const { data } = useDiscoveryBookmarklet();
  if (!data) return null;
  return (
    <div className="space-y-2 border-t border-border pt-3">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider text-muted">
        <Bookmark className="h-3 w-3" /> Bookmarklet
      </div>
      <p className="text-xs text-muted/80">{data.instructions}</p>
      <div className="flex flex-wrap items-center gap-3">
        <a
          href={data.bookmarklet}
          draggable
          onClick={(e) => e.preventDefault()}
          className="inline-block cursor-move rounded-md border border-dashed border-accent/50 bg-accent/10 px-3 py-1.5 text-xs font-medium text-accent"
        >
          [+] Add to Research Queue
        </a>
        <span className="text-[10px] text-muted/60">drag this to your bookmarks bar</span>
      </div>
    </div>
  );
}

export default function PaperDiscoveryCard() {
  const { data, isLoading } = useDiscoveryQueues(15);
  const review = data?.review ?? [];
  const borderline = data?.borderline ?? [];

  return (
    <div>
      <motion.div variants={fadeUp}>
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <BookOpen className="h-3.5 w-3.5 text-accent" /> Paper Discovery
          </span>
        </SectionTitle>
      </motion.div>
      <motion.div variants={fadeUp}>
        <Card className="space-y-4">
          <NominateForm />

          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-16" />
              <Skeleton className="h-16" />
            </div>
          ) : (
            <>
              <div>
                <div className="mb-2 flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-muted">
                  <span>Primary Review -- conf {">"}=0.5 or manual</span>
                  <span className="tnum text-muted/60">{review.length}</span>
                </div>
                {review.length === 0 ? (
                  <p className="text-xs italic text-muted/60">No reviews queued yet.</p>
                ) : (
                  <motion.div variants={stagger(0.04)} initial="hidden" animate="show" className="space-y-1.5">
                    {review.slice(0, 8).map((e, i) => (
                      // Suffix with index — same source_id can show up twice
                      // (e.g. duplicate Crossref nominations); React needs
                      // unique keys regardless.
                      <QueueRow key={`r-${e.source_id ?? "noid"}-${i}`}
                                 entry={e} tier="review" />
                    ))}
                  </motion.div>
                )}
              </div>

              <div className="border-t border-border pt-3">
                <div className="mb-2 flex items-center justify-between text-[11px] font-medium uppercase tracking-wider text-muted">
                  <span>Borderline -- spot-check (0.3-family threshold)</span>
                  <span className="tnum text-muted/60">{borderline.length}</span>
                </div>
                {borderline.length === 0 ? (
                  <p className="text-xs italic text-muted/60">No borderline papers yet.</p>
                ) : (
                  <motion.div variants={stagger(0.04)} initial="hidden" animate="show" className="space-y-1.5">
                    {borderline.slice(0, 6).map((e, i) => (
                      <QueueRow key={`b-${e.source_id ?? "noid"}-${i}`}
                                 entry={e} tier="borderline" />
                    ))}
                  </motion.div>
                )}
              </div>
            </>
          )}

          <BookmarkletBlock />
        </Card>
      </motion.div>
    </div>
  );
}
