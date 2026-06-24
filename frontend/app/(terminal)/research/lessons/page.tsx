"use client";

// /research/lessons — RED Lessons index.
//
// 2026-06-04 (PR-A+B): absorbed /research/legacy. The legacy graveyard
// is now reachable via the "include legacy" toggle + grounding filter.
// URL params honoured at mount:
//   ?verdict=red          → only RED lessons
//   ?include_legacy=true  → also surface pretrain_grounded (47 legacy)
//   ?mechanism_family=    → family filter (used by graveyard inline check)
//   ?grounding_method=    → grounding filter

import { Suspense, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { API_BASE } from "@/lib/api";
import { Card, Badge } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { CompareBar } from "@/components/Compare";


type LessonSummary = {
  lesson_id:            string;
  candidate_name:       string;
  version:              number;
  verdict:              string;
  mechanism_family:     string;
  mechanism_subtype:    string;
  failure_modes:        string[];
  grounding_method:     "paper_grounded" | "stat_only_grounded" | "pretrain_grounded";
  tested_hypothesis_ids: string[];
  n_verbatim_quotes:    number;
  created_ts:           string;
  summary:              string;
};

const VERDICT_TONE: Record<string, string> = {
  RED:    "bg-danger/15 text-danger",
  YELLOW: "bg-warn/15 text-warn",
};

const GROUNDING_TONE: Record<string, string> = {
  paper_grounded:     "bg-ok/15 text-ok border-ok/40",
  stat_only_grounded: "bg-warn/15 text-warn border-warn/40",
  pretrain_grounded:  "bg-muted/15 text-muted border-muted/40",
};


export default function LessonsPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <LessonsInner />
    </Suspense>
  );
}

function LessonsInner() {
  const searchParams = useSearchParams();
  const [lessons, setLessons] = useState<LessonSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [groundingFilter, setGroundingFilter] = useState<string>("");
  const [includeLegacy, setIncludeLegacy] = useState(false);
  const [verdictFilter, setVerdictFilter] = useState<string>("");
  const [familyFilter, setFamilyFilter]   = useState<string>("");
  // R2.12 — multi-select compare for lessons (same UX as forward).
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const toggleSelected = (lid: string) => {
    setSelected((cur) => {
      const next = new Set(cur);
      if (next.has(lid)) next.delete(lid); else next.add(lid);
      return next;
    });
  };

  // Honor URL params on mount + when the user navigates back/forward.
  // Used by the graveyard inline check (/research/candidate) which
  // routes here with ?mechanism_family=…&verdict=red&include_legacy=true.
  useEffect(() => {
    const v   = searchParams?.get("verdict") || "";
    const g   = searchParams?.get("grounding_method") || "";
    const fam = searchParams?.get("mechanism_family") || "";
    const il  = searchParams?.get("include_legacy") === "true";
    if (v)   setVerdictFilter(v);
    if (g)   setGroundingFilter(g);
    if (fam) setFamilyFilter(fam);
    if (il)  setIncludeLegacy(true);
  }, [searchParams]);

  useEffect(() => {
    const params = new URLSearchParams();
    params.set("include_legacy", includeLegacy ? "true" : "false");
    if (groundingFilter) params.set("grounding_method", groundingFilter);
    if (verdictFilter)   params.set("verdict",          verdictFilter);
    if (familyFilter)    params.set("mechanism_family", familyFilter);
    setLoading(true);
    fetch(`${API_BASE}/api/paper_chain/lessons?${params}`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data: LessonSummary[]) => {
        setLessons(data.sort((a, b) => b.created_ts.localeCompare(a.created_ts)));
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [groundingFilter, includeLegacy, verdictFilter, familyFilter]);

  const titleSuffix =
    verdictFilter === "red" && includeLegacy ? " · graveyard view" :
    verdictFilter === "red"                  ? " · RED only" :
    includeLegacy                            ? " + legacy" : "";

  return (
    <div className="p-6 space-y-4">
      <ModeHeader
        mode="learn"
        title={`RED Lessons${titleSuffix}`}
        subtitle={<>
          {lessons.length} entries
          {familyFilter && <> · family <code className="text-foreground">{familyFilter}</code></>}
          {familyFilter && (
            <>{" "}·{" "}
              <button onClick={() => setFamilyFilter("")}
                className="underline hover:text-foreground">
                clear family filter
              </button>
            </>
          )}
        </>}
      />

      <Card>
        <div className="flex flex-wrap items-center gap-2 p-3 border-b border-line">
          <span className="text-xs uppercase text-muted">verdict:</span>
          {(["", "red", "yellow"] as const).map((v) => (
            <button key={v || "all"}
                    className={`px-2 py-0.5 text-xs rounded border ${
                      v === verdictFilter
                        ? "bg-accent/20 border-accent text-accent"
                        : "border-line text-muted hover:text-fg"
                    }`}
                    onClick={() => setVerdictFilter(v)}>
              {v || "all"}
            </button>
          ))}

          <span className="text-xs uppercase text-muted ml-4">grounding:</span>
          {(["", "paper_grounded", "stat_only_grounded", "pretrain_grounded"] as const).map((g) => (
            <button key={g || "all"}
                    className={`px-2 py-0.5 text-xs rounded border ${
                      g === groundingFilter
                        ? "bg-accent/20 border-accent text-accent"
                        : "border-line text-muted hover:text-fg"
                    }`}
                    onClick={() => {
                      setGroundingFilter(g);
                      // Selecting pretrain_grounded implies include_legacy=true
                      if (g === "pretrain_grounded") setIncludeLegacy(true);
                    }}>
              {g || "all"}
            </button>
          ))}

          <label className="ml-4 inline-flex items-center gap-1.5 text-xs text-muted cursor-pointer">
            <input type="checkbox" checked={includeLegacy}
                   onChange={(e) => setIncludeLegacy(e.target.checked)} />
            include legacy (47 pre-2026-06-04)
          </label>
        </div>

        {loading && <div className="p-6 text-sm text-muted">Loading…</div>}
        {error && <div className="p-6 text-sm text-danger">Error: {error}</div>}

        {!loading && !error && (
          <div className="divide-y divide-line">
            {lessons.length === 0 && (
              <div className="p-6 text-sm text-muted">
                No active lessons yet. The chain locked 2026-06-04 — new
                lessons require paper-grounded or stat-only grounding.
              </div>
            )}
            {lessons.map((L) => {
              const isSel = selected.has(L.lesson_id);
              return (
              <div key={L.lesson_id}
                   className={`flex p-4 hover:bg-accent/5 ${isSel ? "bg-accent/[0.04]" : ""}`}>
                <input type="checkbox"
                       checked={isSel}
                       onChange={() => toggleSelected(L.lesson_id)}
                       className="accent-accent cursor-pointer mt-1 mr-3"
                       title="select for compare" />
                <Link href={`/research/lessons/${L.lesson_id}`}
                      className="flex-1 min-w-0">
                  <div className="flex items-baseline gap-3 mb-1">
                    <Badge className={VERDICT_TONE[L.verdict]}>{L.verdict}</Badge>
                    <Badge className={`border ${GROUNDING_TONE[L.grounding_method]}`}>
                      {L.grounding_method}
                    </Badge>
                    <span className="text-sm font-medium">{L.candidate_name}</span>
                    <span className="text-xs text-muted ml-auto">v{L.version}</span>
                  </div>
                  <p className="text-xs text-muted ml-2">
                    <span className="text-fg">{L.mechanism_family}</span>
                    {L.mechanism_subtype && <> · {L.mechanism_subtype}</>}
                    {" · "}
                    failure_modes: {L.failure_modes.join(", ")}
                  </p>
                  <p className="text-xs ml-2 mt-1">{L.summary}</p>
                  {L.grounding_method === "paper_grounded" && (
                    <p className="text-xs text-ok ml-2 mt-1">
                      cites {L.n_verbatim_quotes} verbatim quotes ·
                      tests {L.tested_hypothesis_ids.length} hypothesis_ids
                    </p>
                  )}
                </Link>
              </div>
              );
            })}
          </div>
        )}
      </Card>

      {/* R2.12 — Compare panel using the shared framework. Surfaces
          verdict / family / failure-mode overlap across selected
          lessons. Most useful for "did we kill these for the same
          reason?" cross-lesson audits. */}
      {selected.size >= 1 && (
        <CompareBar<LessonSummary>
          items={lessons.filter((L) => selected.has(L.lesson_id))}
          getKey={(L) => L.lesson_id}
          onClear={() => setSelected(new Set())}
          onRemove={(L) => toggleSelected(L.lesson_id)}
          title="Compare lessons"
          headerCell={(L) => (
            <div>
              <div className="flex items-center gap-1.5">
                <span className={`tnum text-[10px] px-1 rounded font-mono ${VERDICT_TONE[L.verdict]}`}>
                  {L.verdict}
                </span>
                <span className={`tnum text-[10px] px-1 rounded font-mono border ${GROUNDING_TONE[L.grounding_method]}`}>
                  {L.grounding_method.replace("_grounded", "")}
                </span>
              </div>
              <div className="text-[10px] font-mono text-foreground/80 mt-0.5 truncate max-w-[200px]">
                {L.candidate_name}
              </div>
            </div>
          )}
          rows={[
            { label: "verdict",          pick: (L) => L.verdict,           mono: true },
            { label: "grounding",        pick: (L) => L.grounding_method,  mono: true },
            { label: "family",           pick: (L) => L.mechanism_family,  mono: true },
            { label: "subtype",          pick: (L) => L.mechanism_subtype, mono: true },
            { label: "failure modes",    pick: (L) => L.failure_modes.join(" · ") },
            { label: "summary",          pick: (L) => L.summary },
            { label: "verbatim quotes",  pick: (L) => String(L.n_verbatim_quotes), mono: true },
            { label: "hypotheses tested",pick: (L) => String(L.tested_hypothesis_ids.length), mono: true },
            { label: "created",          pick: (L) => L.created_ts.slice(0, 10), mono: true },
          ]}
          actionCell={(L) => (
            <Link href={`/research/lessons/${L.lesson_id}`}
              className="inline-flex items-center gap-1 rounded bg-accent text-background hover:bg-accent/90 px-2 py-1 text-[10.5px] font-semibold">
              Open detail →
            </Link>
          )} />
      )}
    </div>
  );
}
