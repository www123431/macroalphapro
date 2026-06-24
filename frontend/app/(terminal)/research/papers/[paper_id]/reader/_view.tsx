"use client";

// /research/papers/[paper_id]/reader — full-text reader for an ingested paper.
//
// Renders all chunks with hypothesis-quote highlighting + side TOC.
// User can jump to a specific chunk via the TOC.

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge } from "@/components/ui";
import ChunkViewer, { type Chunk } from "@/components/ChunkViewer";


type PaperDetail = {
  paper_id:        string;
  title:           string;
  year:            number;
  authors:         string[];
  venue:           string;
  doi:             string;
  fulltext_status: string;
  n_chunks:        number;
};


export default function PaperReaderPage(
  { params }: { params: Promise<{ paper_id: string }> }
) {
  const { paper_id } = use(params);
  const [paper, setPaper] = useState<PaperDetail | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [onlyAnnotated, setOnlyAnnotated] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/api/paper_chain/papers/${paper_id}`, { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)),
      fetch(`${API_BASE}/api/paper_chain/papers/${paper_id}/chunks`, { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`)),
    ])
      .then(([p, c]: [PaperDetail, Chunk[]]) => {
        setPaper(p);
        setChunks(c);
        setError(null);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [paper_id]);

  if (loading) return <div className="p-6 text-sm text-muted">Loading paper…</div>;
  if (error)   return <div className="p-6 text-sm text-danger">Error: {error}</div>;
  if (!paper)  return <div className="p-6 text-sm text-muted">Paper not found.</div>;

  // Group chunks by section for TOC
  const sections: { name: string; chunkIds: string[]; n_annotated: number }[] = [];
  let currentSection: typeof sections[0] | null = null;
  for (const c of chunks) {
    const sec = c.section || "(no section)";
    if (!currentSection || currentSection.name !== sec) {
      currentSection = { name: sec, chunkIds: [], n_annotated: 0 };
      sections.push(currentSection);
    }
    currentSection.chunkIds.push(c.chunk_id);
    if (c.quoted_by.length > 0) currentSection.n_annotated += 1;
  }

  const visibleChunks = onlyAnnotated
    ? chunks.filter((c) => c.quoted_by.length > 0)
    : chunks;

  const totalAnnotated = chunks.filter((c) => c.quoted_by.length > 0).length;
  const totalQuotes = chunks.reduce((s, c) => s + c.quoted_by.length, 0);

  return (
    <div className="grid grid-cols-[260px_1fr] gap-4 p-4 max-h-screen">
      {/* Side TOC */}
      <aside className="overflow-y-auto sticky top-0 max-h-screen pr-2">
        <div className="mb-3">
          <Link href={`/research/papers/${paper_id}`}
                className="text-xs text-muted hover:text-fg">
            ← back to paper detail
          </Link>
        </div>

        <div className="space-y-2 mb-4">
          <div className="text-xs uppercase text-muted">Reader stats</div>
          <div className="text-sm">{chunks.length} chunks</div>
          <div className="text-sm text-accent">{totalAnnotated} with quotes</div>
          <div className="text-xs text-muted">{totalQuotes} total verbatim citations</div>
        </div>

        <label className="flex items-center gap-2 text-xs text-muted mb-3">
          <input type="checkbox"
                 checked={onlyAnnotated}
                 onChange={(e) => setOnlyAnnotated(e.target.checked)}/>
          show only annotated chunks
        </label>

        <div className="text-xs uppercase text-muted mb-1">Sections</div>
        <nav className="space-y-0.5">
          {sections.map((s, i) => (
            <a key={i}
               href={`#chunk-${s.chunkIds[0]}`}
               className="block text-xs hover:text-accent py-0.5">
              <span className="text-muted mr-1">·</span>
              {s.name.slice(0, 32) || "(unlabeled)"}
              {s.n_annotated > 0 && (
                <Badge className="ml-1 text-[9px] bg-accent/15 text-accent">
                  {s.n_annotated}
                </Badge>
              )}
            </a>
          ))}
        </nav>
      </aside>

      {/* Reader body */}
      <main className="overflow-y-auto max-h-screen pr-2">
        <div className="mb-4">
          <SectionTitle>{paper.title}</SectionTitle>
          <p className="text-xs text-muted mt-1">
            {paper.authors.join(", ")} · {paper.year} · {paper.venue}
            {paper.doi && <> · doi:<code>{paper.doi}</code></>}
          </p>
        </div>

        <Card>
          <div className="p-4 space-y-3">
            {visibleChunks.length === 0 ? (
              <div className="text-sm text-muted">No chunks to show.</div>
            ) : (
              visibleChunks.map((c) => (
                <ChunkViewer key={c.chunk_id} chunk={c} />
              ))
            )}
          </div>
        </Card>
      </main>
    </div>
  );
}
