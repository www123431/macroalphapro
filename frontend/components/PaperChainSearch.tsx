"use client";

// PaperChainSearch — global semantic search input + dropdown results.
// Hits /api/paper_chain/search; user clicks a result to navigate to the
// paper's reader page (and ideally jump to the chunk).

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";


type SearchHit = {
  chunk_id:    string;
  text:        string;
  paper_id:    string;
  paper_title: string;
  section:     string;
  distance:    number | null;
};


export default function PaperChainSearch() {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const ref = useRef<HTMLDivElement>(null);

  // Debounced search
  useEffect(() => {
    if (q.trim().length < 3) {
      setHits([]);
      setOpen(false);
      return;
    }
    const t = setTimeout(() => {
      setLoading(true);
      setError(null);
      fetch(`${API_BASE}/api/paper_chain/search?q=${encodeURIComponent(q)}&top=10`,
            { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
        .then((data: SearchHit[]) => { setHits(data); setOpen(true); })
        .catch((e) => setError(String(e)))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(t);
  }, [q]);

  // Close dropdown on outside click
  useEffect(() => {
    function onDocClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, []);

  const handleHitClick = (h: SearchHit) => {
    setOpen(false);
    setQ("");
    if (h.paper_id) {
      router.push(`/research/papers/${h.paper_id}/reader#chunk-${h.chunk_id}`);
    }
  };

  return (
    <div ref={ref} className="relative w-full max-w-xl">
      <input
        type="text"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        onFocus={() => hits.length > 0 && setOpen(true)}
        placeholder="Search paper text… (e.g. 'momentum decay')"
        className="w-full bg-bg border border-line rounded px-3 py-1.5 text-sm
                   focus:outline-none focus:border-accent placeholder-muted/60"
      />

      {(loading || open) && (
        <div className="absolute mt-1 w-full bg-bg border border-line rounded
                        shadow-xl max-h-[60vh] overflow-y-auto z-50">
          {loading && (
            <div className="p-3 text-xs text-muted">Searching…</div>
          )}
          {error && (
            <div className="p-3 text-xs text-danger">Error: {error}</div>
          )}
          {!loading && !error && hits.length === 0 && q.length >= 3 && (
            <div className="p-3 text-xs text-muted">No matches.</div>
          )}
          {!loading && !error && hits.map((h, i) => (
            <button
              key={i}
              onClick={() => handleHitClick(h)}
              className="block w-full text-left p-3 border-b border-line/50
                         hover:bg-accent/10">
              <div className="flex items-baseline gap-2 mb-1">
                <span className="text-xs font-medium truncate max-w-[40ch]">
                  {h.paper_title}
                </span>
                {h.section && (
                  <span className="text-[10px] text-muted">· {h.section}</span>
                )}
                {h.distance !== null && (
                  <span className="text-[10px] text-muted ml-auto">
                    sim {(1 - h.distance).toFixed(2)}
                  </span>
                )}
              </div>
              <p className="text-xs text-muted line-clamp-2">
                {h.text}
              </p>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
