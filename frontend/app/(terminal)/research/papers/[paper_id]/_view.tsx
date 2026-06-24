"use client";

// /research/papers/[paper_id] — paper detail with hypotheses list.

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge } from "@/components/ui";


type PaperDetail = {
  paper_id:        string;
  title:           string;
  year:            number;
  authors:         string[];
  venue:           string;
  doi:             string;
  fulltext_status: string;
  n_chunks:        number;
  shelves:         string[];
  shelf_notes:     Record<string, string>;
  pdf_source_url:  string;
  referenced_by_lessons: string[];
  referenced_by_sleeves: string[];
  referenced_by_factors: string[];
};

type Hypothesis = {
  hypothesis_id:        string;
  source_paper_id:      string;
  claim:                string;
  mechanism_family:     string;
  mechanism_subtype:    string;
  predicted_direction:  string;
  predicted_magnitude:  string;
  required_data:        string[];
  n_verbatim_quotes:    number;
  review_state:         string;
  is_tested:            boolean;
  tested_by_lessons:    string[];
};


// PaperDetailBody — extracted 2026-06-06 so the same fetch + render
// can be reused from both the static dynamic route [paper_id]/page.tsx
// (for build-time-known papers) AND the query-param /view?id=... route
// (for newly-ingested papers whose ID didn't exist at build time, which
// would otherwise 404 + fall back to the home page).
export function PaperDetailBody({ paper_id }: { paper_id: string }) {
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [hyps, setHyps] = useState<Hypothesis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/paper_chain/papers/${paper_id}`,
            { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)),
      fetch(`${API_BASE}/api/paper_chain/papers/${paper_id}/hypotheses`,
            { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)),
    ])
      .then(([p, h]: [PaperDetail, Hypothesis[]]) => {
        setPaper(p);
        setHyps(h);
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [paper_id]);

  if (loading) return <div className="p-6 text-sm text-muted">Loading…</div>;
  if (error)   return <div className="p-6 text-sm text-danger">Error: {error}</div>;
  if (!paper)  return <div className="p-6 text-sm text-muted">Paper not found.</div>;

  return (
    <div className="p-6 space-y-4">
      <div className="space-y-1">
        <SectionTitle>{paper.title}</SectionTitle>
        <p className="text-xs text-muted">
          {paper.authors.join(", ")} · {paper.year} · {paper.venue}
          {paper.doi && <span> · doi:<code>{paper.doi}</code></span>}
        </p>
        <div className="flex gap-1 mt-1">
          {paper.shelves.map((s) => (
            <Badge key={s} className="text-[10px] bg-accent/15 text-accent">
              {s}
            </Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <Card><div className="p-3">
          <div className="text-xs uppercase text-muted">Status</div>
          <div className="text-sm">{paper.fulltext_status}</div>
          <div className="text-xs text-muted">{paper.n_chunks} chunks</div>
        </div></Card>
        <Card><div className="p-3">
          <div className="text-xs uppercase text-muted">Hypotheses</div>
          <div className="text-sm">
            {hyps.length} total
            {hyps.filter(h => h.is_tested).length > 0 && (
              <span className="text-ok ml-1">
                ({hyps.filter(h => h.is_tested).length} tested)
              </span>
            )}
          </div>
        </div></Card>
        <Card><div className="p-3">
          <div className="text-xs uppercase text-muted">Citations from book</div>
          <div className="text-xs text-muted space-y-0.5">
            {paper.referenced_by_lessons.length > 0 && (
              <div>{paper.referenced_by_lessons.length} lessons</div>
            )}
            {paper.referenced_by_sleeves.length > 0 && (
              <div>{paper.referenced_by_sleeves.length} sleeves</div>
            )}
            {paper.referenced_by_factors.length > 0 && (
              <div>{paper.referenced_by_factors.length} factors</div>
            )}
            {(paper.referenced_by_lessons.length === 0 &&
              paper.referenced_by_sleeves.length === 0 &&
              paper.referenced_by_factors.length === 0) && (
              <div>no inbound references yet</div>
            )}
          </div>
        </div></Card>
      </div>

      <div>
        <h3 className="text-sm font-semibold mb-2">
          Extracted hypotheses ({hyps.length})
        </h3>
        <Card>
          {hyps.length === 0 ? (
            <div className="p-4 text-sm text-muted">
              No hypotheses extracted yet (paper may be methodology-only or
              extraction is pending).
            </div>
          ) : (
            <div className="divide-y divide-line">
              {hyps.map((h, i) => (
                <div key={h.hypothesis_id} className="p-4 space-y-1.5">
                  <div className="flex items-baseline gap-2">
                    <span className="text-xs text-muted w-6">#{i + 1}</span>
                    {h.is_tested ? (
                      <Badge className="bg-ok/15 text-ok text-[10px]">tested</Badge>
                    ) : (
                      <Badge className="bg-warn/15 text-warn text-[10px]">untested</Badge>
                    )}
                    <span className="text-xs text-muted">
                      {h.mechanism_family} · {h.mechanism_subtype}
                    </span>
                    <span className="text-[10px] text-muted ml-auto">
                      direction:{h.predicted_direction} · {h.n_verbatim_quotes} quotes
                    </span>
                  </div>
                  <p className="text-sm ml-8">{h.claim}</p>
                  <p className="text-xs text-muted ml-8">
                    <strong>magnitude:</strong> {h.predicted_magnitude}
                  </p>
                  {h.is_tested && h.tested_by_lessons.length > 0 && (
                    <p className="text-xs ml-8 text-ok">
                      tested by:{" "}
                      {h.tested_by_lessons.map((lid) => (
                        <Link key={lid} href={`/research/lessons/${lid}`}
                              className="underline hover:text-accent mr-1">
                          {lid.slice(0, 8)}
                        </Link>
                      ))}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}


// Thin wrapper for the dynamic [paper_id] static route — resolves the
// Next.js params Promise then delegates to PaperDetailBody.
export default function PaperDetailPage(
  { params }: { params: Promise<{ paper_id: string }> }
) {
  const { paper_id } = use(params);
  return <PaperDetailBody paper_id={paper_id} />;
}
