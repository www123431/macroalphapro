// frontend/lib/static-params.ts — read jsonl files at build time to
// generate dynamic-route IDs for Next.js static export.
//
// At build time (Node.js), generateStaticParams() needs to declare
// every URL slug Next will pre-render into frontend/out/. Our IDs
// live in jsonl files under data/research_store/. This helper reads
// them with process.cwd() = repo root (Next sets it correctly when
// `next build` runs from frontend/).

import fs from "node:fs";
import path from "node:path";

// `turbopackIgnore` comments below are required: Next.js' file-trace
// analyzer otherwise sees process.cwd()+".." as dynamic and bundles
// thousands of unrelated files. The fs reads only happen at build time
// in generateStaticParams (Node context), never at runtime, so trace
// inclusion is unnecessary.

function readJsonl<T = Record<string, unknown>>(relPath: string): T[] {
  const repoRoot = path.resolve(/*turbopackIgnore: true*/ process.cwd(), "..");
  const abs = path.join(/*turbopackIgnore: true*/ repoRoot, relPath);
  if (!fs.existsSync(/*turbopackIgnore: true*/ abs)) return [];
  const raw = fs.readFileSync(/*turbopackIgnore: true*/ abs, "utf-8");
  const out: T[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const s = line.trim();
    if (!s) continue;
    try { out.push(JSON.parse(s) as T); } catch { /* skip malformed */ }
  }
  return out;
}

/** All paper_ids in registry (latest per DOI; deduped by paper_id).
 *  Falls back to empty array if file missing — Next still builds the route
 *  shape but renders fallback at runtime via client-side fetch. */
export function getAllPaperIds(): string[] {
  const rows = readJsonl<{ paper_id?: string; doi?: string; version?: number }>(
    "data/research_store/papers_registry.jsonl"
  );
  // Group by DOI; pick highest version per DOI; collect paper_id of latest
  const latestPerDoi: Record<string, { paper_id: string; version: number }> = {};
  for (const r of rows) {
    if (!r.paper_id) continue;
    const key = r.doi || `_no_doi::${r.paper_id}`;
    const v = r.version ?? 1;
    const cur = latestPerDoi[key];
    if (!cur || v > cur.version) latestPerDoi[key] = { paper_id: r.paper_id, version: v };
  }
  return [...new Set(Object.values(latestPerDoi).map((x) => x.paper_id))];
}

/** All lesson_ids (every version — needed because we link to specific
 *  lesson_id, including the demo v2 paper_grounded). */
export function getAllLessonIds(): string[] {
  const rows = readJsonl<{ lesson_id?: string }>(
    "data/research_store/red_lessons.jsonl"
  );
  return [...new Set(rows.map((r) => r.lesson_id).filter((x): x is string => !!x))];
}
