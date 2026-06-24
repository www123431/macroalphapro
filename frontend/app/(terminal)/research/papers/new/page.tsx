"use client";

// /research/papers/new — self-service paper ingestion.
//
// Closes the project's largest collaboration gap (Collab-P0 in the
// R2.x audit): until today the only way to add a paper to the T7
// registry was to drop into Claude Code and run
// `scripts/extract_paper_hypotheses.py` by hand. Now the user can
// drop a PDF or paste an arXiv/NBER/JSTOR URL straight into the UI,
// review the heuristic-extracted metadata, assign shelves, and
// commit — Claude can pick up the new hypotheses immediately
// (forward-vectors regenerates on each call).
//
// Two-step UX (matches the backend's preview / ingest split):
//   Step 1 — drop or paste -> POST /papers/preview
//             - Heuristic title / authors / year / DOI / abstract guess
//             - Shows a text preview so the user can sanity-check
//   Step 2 — review / fix / assign shelves -> POST /papers/ingest
//             - Sync extraction (LLM call, 20-60s typical)
//             - On success, route to /research/papers/{paper_id}
//
// Shelf semantics (see vizTokens / source_papers_from_t7 in composer):
//   doctrine_method   = canonical methodology paper (HLZ, FF, etc.)
//   green_motivation  = mechanism we deploy (or cousin)
//   yellow_motivation = adjacent / partial-match
//   red_motivation    = mechanism we tested and killed
//   red_critique      = paper itself argues the mechanism doesn't work
//   dormant_revisit   = killed earlier; flag for re-look
//   other             = default — nothing more specific applies

import { useRef, useState, useEffect, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  Upload, Link2, FileText, Loader2, AlertCircle,
  CheckCircle2, ArrowRight, X, ArrowLeft,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { Tip } from "@/components/Tip";
import { useI18n } from "@/lib/i18n";


type PreviewResp = {
  preview_id:      string;
  title_guess:     string;
  authors_guess:   string[];
  year_guess:      number | null;
  doi_guess:       string;
  venue_guess:     string;
  abstract_guess:  string;
  n_pages:         number;
  text_chars:      number;
  text_preview:    string;
  extraction_note: string;
};


type IngestResp = {
  paper_id:      string;
  title:         string;
  n_chunks:      number;
  n_hypotheses:  number;
  registry_path: string;
  next_url:      string;
};


const SHELF_OPTIONS: { key: string; label: string; tone: string; hint: string }[] = [
  { key: "doctrine_method",   label: "doctrine method",   tone: "text-accent",
    hint: "Canonical methodology paper. The reference everyone cites — Harvey-Liu-Zhu |t|>3, Fama-French, Bailey-LdP deflated SR, etc." },
  { key: "green_motivation",  label: "green motivation",  tone: "text-ok",
    hint: "Mechanism we deploy (or want to). Reading reinforces / refines a sleeve already on the book." },
  { key: "green_critique",    label: "green critique",    tone: "text-ok",
    hint: "Paper sharpens / questions one of our GREEN mechanisms. Survives -> we keep deploying." },
  { key: "yellow_motivation", label: "yellow motivation", tone: "text-warn",
    hint: "Adjacent or partial match to a mechanism we run. Worth borrowing technique from, but not the headline result." },
  { key: "dormant_revisit",   label: "dormant revisit",   tone: "text-muted",
    hint: "Mechanism we killed years ago but data / regime may have changed. Flag for re-look." },
  { key: "red_motivation",    label: "red motivation",    tone: "text-danger",
    hint: "Mechanism we tested and the gate killed. Paper kept for the graveyard reasoning." },
  { key: "red_critique",      label: "red critique",      tone: "text-danger",
    hint: "Paper itself argues the mechanism doesn't survive. Strong anti-evidence; cite when justifying not-testing-it." },
  { key: "other",              label: "other",             tone: "text-muted",
    hint: "Default. Nothing more specific applies. Edit later via the registry tool if a better shelf emerges." },
];


export default function NewPaperPage() {
  // Wrap the inner page in Suspense so useSearchParams (CSR-only) doesn't
  // bail static export of /research/papers/new during build.
  return (
    <Suspense fallback={<div className="p-6 text-[12px] text-muted">Loading…</div>}>
      <NewPaperPageInner />
    </Suspense>
  );
}

function NewPaperPageInner() {
  const router  = useRouter();
  const fileRef = useRef<HTMLInputElement>(null);
  const searchParams = useSearchParams();
  const { t } = useI18n();

  // Step state
  const [phase, setPhase] = useState<"idle" | "previewing" | "review" | "ingesting" | "done">("idle");
  const [error, setError] = useState<string | null>(null);
  const [previewId, setPreviewId] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl]       = useState("");
  const [filename, setFilename]   = useState("");

  // Paste-text escape hatch — for scanned-image PDFs that pymupdf
  // chokes on (and image-only / DRM-locked / non-academic PDFs).
  // Hidden by default; user opens it from the inline tip after a 422.
  const [showPasteText, setShowPasteText] = useState(false);
  const [pasteText, setPasteText]         = useState("");
  const [pasteSrcLabel, setPasteSrcLabel] = useState("");

  // Phase 1.7 step 3 (2026-06-06): ingestion_reason. Whoever PICKED
  // the paper authors the reason. URL ?agent_reason=... means the
  // user clicked "Ingest now" on /incoming so agent's text is pre-
  // populated (source=AGENT). User can click "✏️ Write my own" to
  // clear and re-author (source=USER).
  const [ingestionReason, setIngestionReason] = useState("");
  const [ingestionSource, setIngestionSource] = useState<"user" | "agent">("user");

  // 2026-06-06 Step 2 simplification: replaced the 8-shelf chip grid
  // with a 4-option "initial intent" dropdown. Shelves default to
  // ["other"] silently and get auto-refined by the verdict pipeline
  // post-test. The intent maps to the IntentCategory enum that already
  // lived on IngestionReason. UI surfaces intent at ingest time
  // directly instead of waiting for LLM extraction from free_text.
  type IntentKey = "expand_breadth" | "improve_existing_sleeve"
                 | "methodology_borrow" | "other";
  const [intent, setIntent] = useState<IntentKey>("other");
  const [showMoreMetadata, setShowMoreMetadata] = useState(false);

  // Phase 1.6 (2026-06-05): Employee A "Ingest now" prefill. When the
  // user lands here from /research/papers/incoming, the title / authors
  // / year / abstract / pdf_url / shelf are already in the URL.
  // We don't auto-fire /preview — the user still has to drop the PDF or
  // paste text — but the metadata fields below are pre-filled so they
  // can hit Commit immediately if happy.
  useEffect(() => {
    if (!searchParams) return;
    const t = searchParams.get("title");
    const a = searchParams.get("authors");
    const y = searchParams.get("year");
    const d = searchParams.get("doi");
    const ab = searchParams.get("abstract");
    const pu = searchParams.get("pdf_url");
    const sh = searchParams.get("shelf");
    if (t)  setTitle(t);
    if (a)  setAuthors(a.split("|").filter(Boolean));
    if (y)  setYear(parseInt(y, 10) || "");
    if (d)  setDoi(d);
    if (ab) setAbstract(ab);
    if (pu) setPdfUrl(pu);
    if (sh) setShelves([sh]);

    // Phase 1.7 step 3 prefill — agent_reason from /incoming "Ingest now"
    const ar = searchParams.get("agent_reason");
    if (ar) {
      setIngestionReason(ar);
      setIngestionSource("agent");
    }
    // If we have an abstract from the prefill (the common case from
    // Incoming page), open the paste-text path automatically so the
    // user can drop in the full text without hunting for it.
    if (ab && ab.length > 200) {
      setShowPasteText(true);
      setPasteText(ab);
    }
    // run once on mount; ignore searchParams changes after
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Editable metadata after preview
  const [title,    setTitle]    = useState("");
  const [authors,  setAuthors]  = useState<string[]>([]);
  const [year,     setYear]     = useState<number | "">("");
  const [doi,      setDoi]      = useState("");
  const [venue,    setVenue]    = useState("");
  const [abstract, setAbstract] = useState("");
  const [shelves,  setShelves]  = useState<string[]>(["other"]);
  const [shelfNote, setShelfNote] = useState("");
  const [textPreview, setTextPreview] = useState("");
  const [extractStats, setExtractStats] = useState<{ pages: number; chars: number } | null>(null);

  const [result, setResult] = useState<IngestResp | null>(null);

  const resetAll = () => {
    setPhase("idle");
    setError(null);
    setPreviewId(null);
    setPdfUrl("");
    setFilename("");
    setTitle("");
    setAuthors([]);
    setYear("");
    setDoi("");
    setVenue("");
    setAbstract("");
    setShelves(["other"]);
    setShelfNote("");
    setIntent("other");
    setShowMoreMetadata(false);
    setTextPreview("");
    setExtractStats(null);
    setResult(null);
    setIngestionReason("");
    setIngestionSource("user");
  };

  const handlePreview = async (formData: FormData) => {
    setPhase("previewing");
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/paper_chain/papers/preview`, {
        method: "POST",
        body:   formData,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
      }
      const data = await res.json() as PreviewResp;
      setPreviewId(data.preview_id);
      setTitle(data.title_guess);
      setAuthors(data.authors_guess.length ? data.authors_guess : []);
      setYear(data.year_guess ?? "");
      setDoi(data.doi_guess);
      setVenue(data.venue_guess);
      setAbstract(data.abstract_guess);
      setTextPreview(data.text_preview);
      setExtractStats({ pages: data.n_pages, chars: data.text_chars });
      setPhase("review");
    } catch (e: any) {
      setError(String(e?.message ?? e));
      setPhase("idle");
    }
  };

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFilename(f.name);
    const fd = new FormData();
    fd.append("file", f);
    handlePreview(fd);
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (!f) return;
    setFilename(f.name);
    const fd = new FormData();
    fd.append("file", f);
    handlePreview(fd);
  };

  const onSubmitUrl = (e: React.FormEvent) => {
    e.preventDefault();
    if (!pdfUrl.trim()) return;
    setFilename("");
    const fd = new FormData();
    fd.append("pdf_source_url", pdfUrl.trim());
    handlePreview(fd);
  };

  const handlePasteText = async () => {
    if (pasteText.trim().length < 200) {
      setError("paste at least 200 chars");
      return;
    }
    setPhase("previewing");
    setError(null);
    setFilename(pasteSrcLabel || "pasted text");
    try {
      const res = await fetch(`${API_BASE}/api/paper_chain/papers/preview-from-text`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          text:      pasteText,
          src_label: pasteSrcLabel,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
      }
      const data = await res.json() as PreviewResp;
      setPreviewId(data.preview_id);
      setTitle(data.title_guess);
      setAuthors(data.authors_guess.length ? data.authors_guess : []);
      setYear(data.year_guess ?? "");
      setDoi(data.doi_guess);
      setVenue(data.venue_guess);
      setAbstract(data.abstract_guess);
      setTextPreview(data.text_preview);
      setExtractStats({ pages: data.n_pages, chars: data.text_chars });
      setPhase("review");
    } catch (e: any) {
      setError(String(e?.message ?? e));
      setPhase("idle");
    }
  };

  const handleIngest = async () => {
    if (!previewId) { setError("missing preview_id — re-upload the PDF"); return; }
    if (!title.trim()) { setError("title required"); return; }
    if (!abstract.trim()) { setError(t("new.err.abstract_required")); return; }
    setPhase("ingesting");
    setError(null);
    try {
      // 2026-06-06 simplification: shelves come from either URL ?shelf=
      // pre-fill (from /incoming) or the silent default ["other"].
      // shelf_notes auto-fills with intent label for the primary shelf —
      // covers the OTHER-shelf-requires-rationale validation without
      // bothering the user. Non-OTHER shelves accept the note harmlessly.
      const intentLabel = (
        intent === "expand_breadth"          ? "candidate / new mechanism" :
        intent === "improve_existing_sleeve" ? "strengthens existing sleeve" :
        intent === "methodology_borrow"      ? "methodology reference" :
                                                "(no specific intent)"
      );
      const primaryShelf = shelves[0] || "other";
      const shelf_notes: Record<string, string> = {
        [primaryShelf]: shelfNote.trim() || intentLabel,
      };
      // Phase 1.7 step 3: include ingestion_reason if user provided one
      // OR if intent != "other" (intent dropdown alone gives signal even
      // when user wrote no free-text).
      const hasReasonText = !!ingestionReason.trim();
      const hasNonOtherIntent = intent !== "other";
      const reasonPayload = (hasReasonText || hasNonOtherIntent)
        ? {
            free_text:        ingestionReason.trim().slice(0, 200),
            source:           ingestionSource,
            intent_category:  intent,       // 2026-06-06: user picks directly
          }
        : null;
      const res = await fetch(`${API_BASE}/api/paper_chain/papers/ingest`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          title, year: Number(year) || 0, authors, venue, doi, abstract,
          shelves, shelf_notes, pdf_source_url: pdfUrl, note: "",
          preview_id: previewId,
          ingestion_reason: reasonPayload,
        }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
      }
      const data = await res.json() as IngestResp;
      setResult(data);
      setPhase("done");
    } catch (e: any) {
      setError(String(e?.message ?? e));
      setPhase("review");
    }
  };

  return (
    <div className="p-6 space-y-4 max-w-5xl">
      {/* U6 2026-06-05: explicit back link reinforces the "I'm inside
          the Papers section" mental model. The route is intentionally
          not in the main nav — it's a contextual sub-action of
          /research/papers reachable via "+ Ingest paper" button. */}
      <Link href="/research/papers"
        className="inline-flex items-center gap-1 text-[10.5px] text-muted hover:text-accent transition-colors">
        <ArrowLeft className="h-3 w-3" />
        {t("new.back_to_library")}
      </Link>

      <ModeHeader
        mode="research"
        title={t("new.title")}
        subtitle={t("new.subtitle")}
      />

      {error && (
        <Card className="border border-danger/30 bg-danger/5 p-3">
          <div className="text-[12px] text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" />
            {error}
            <button onClick={() => setError(null)}
              className="ml-auto text-muted/70 hover:text-foreground">
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
          {/* When the PDF is a scanned image, surface the paste-text
              escape hatch right next to the error so the user doesn't
              have to hunt for it. */}
          {(error.includes("pymupdf") || error.includes("scanned")) && !showPasteText && (
            <button onClick={() => { setShowPasteText(true); setError(null); }}
              className="mt-2 text-[11px] text-accent hover:underline">
              {t("new.step1.paste_toggle")}
            </button>
          )}
        </Card>
      )}

      {/* Step 1 — drop / paste */}
      {(phase === "idle" || phase === "previewing") && (
        <>
          <Card className="p-0 overflow-hidden">
            <div className="p-3 border-b border-border/30 text-[10px] uppercase tracking-wider text-muted/70">
              {t("new.step1.title")}
            </div>
            <div className="p-4 space-y-3">
              {/* Drop zone */}
              <div onDragOver={(e) => e.preventDefault()}
                   onDrop={onDrop}
                   onClick={() => fileRef.current?.click()}
                   className={cn(
                     "rounded-md border-2 border-dashed cursor-pointer p-6 text-center transition-colors",
                     phase === "previewing"
                       ? "border-accent/60 bg-accent/5"
                       : "border-border/60 hover:border-accent/40 hover:bg-accent/[0.03]",
                   )}>
                <input ref={fileRef} type="file" accept="application/pdf"
                       className="hidden" onChange={onFileChange} />
                {phase === "previewing" ? (
                  <div className="inline-flex items-center gap-2 text-[12px] text-accent">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    {t("new.step1.extracting")} {filename && `(${filename})`}
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <Upload className="h-7 w-7 text-muted/60 mx-auto" strokeWidth={1.5} />
                    <p className="text-[13px] text-foreground/85">
                      {t("new.step1.drop_label")}
                    </p>
                    <p className="text-[11px] text-muted/60">
                      {t("new.step1.drop_hint")}
                    </p>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3 text-[10px] uppercase tracking-wider text-muted/50">
                <div className="flex-1 h-px bg-border/40" /> {t("new.step1.or")} <div className="flex-1 h-px bg-border/40" />
              </div>

              {/* URL form */}
              <form onSubmit={onSubmitUrl} className="flex items-center gap-2">
                <Link2 className="h-3.5 w-3.5 text-muted shrink-0" strokeWidth={2} />
                <input value={pdfUrl}
                       onChange={(e) => setPdfUrl(e.target.value)}
                       disabled={phase === "previewing"}
                       placeholder="https://arxiv.org/pdf/2401.12345.pdf  ·  https://www.nber.org/.../w12345.pdf"
                       className="flex-1 rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1.5 focus:outline-none focus:border-accent/60" />
                <button type="submit"
                        disabled={!pdfUrl.trim() || phase === "previewing"}
                        className="rounded bg-accent text-background hover:bg-accent/90 disabled:opacity-50 px-3 py-1.5 text-[12px] font-semibold">
                  {t("new.step1.fetch")}
                </button>
              </form>

              {/* Paste-text escape hatch — collapsed by default; auto-
                  expands when a pymupdf 422 error fires (see error
                  card above). Use for scanned-image PDFs, non-academic
                  sources, or any content where you have plain text but
                  no extractable PDF. */}
              {!showPasteText ? (
                <button onClick={() => setShowPasteText(true)}
                  className="block text-[10.5px] text-muted/70 hover:text-accent transition-colors mt-1">
                  {t("new.step1.paste_toggle")}
                </button>
              ) : (
                <div className="space-y-2 rounded-md border border-border/40 bg-panel2/20 p-3 mt-1">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] uppercase tracking-wider text-muted/70">
                      {t("new.paste.title")}
                    </span>
                    <button onClick={() => { setShowPasteText(false); setPasteText(""); setPasteSrcLabel(""); }}
                      className="text-muted/60 hover:text-foreground">
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                  <p className="text-[10.5px] text-muted/70 leading-relaxed">
                    {t("new.paste.hint")}
                  </p>
                  <input value={pasteSrcLabel}
                         onChange={(e) => setPasteSrcLabel(e.target.value)}
                         disabled={phase === "previewing"}
                         placeholder={t("new.paste.src_ph")}
                         className="w-full rounded border border-border/40 bg-panel2/30 text-[11px] px-2 py-1.5 focus:outline-none focus:border-accent/60" />
                  <textarea value={pasteText}
                            onChange={(e) => setPasteText(e.target.value)}
                            disabled={phase === "previewing"}
                            rows={10}
                            placeholder={t("new.paste.ph")}
                            className="w-full rounded border border-border/40 bg-panel2/30 text-[11px] px-2 py-1.5 font-mono focus:outline-none focus:border-accent/60 resize-y" />
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] text-muted/60 tnum">
                      {pasteText.length.toLocaleString()} {t("new.paste.chars")}
                      {pasteText.length < 200 && pasteText.length > 0 && (
                        <span className="text-warn ml-1">{t("new.paste.need_more")}</span>
                      )}
                    </span>
                    <button onClick={handlePasteText}
                            disabled={pasteText.trim().length < 200 || phase === "previewing"}
                            className="rounded bg-accent text-background hover:bg-accent/90 disabled:opacity-50 px-3 py-1.5 text-[12px] font-semibold inline-flex items-center gap-1">
                      {phase === "previewing" ? (
                        <><Loader2 className="h-3.5 w-3.5 animate-spin" /> {t("new.paste.processing")}</>
                      ) : (
                        <>{t("new.paste.use_text")} <ArrowRight className="h-3.5 w-3.5" /></>
                      )}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Quick reference panel */}
          <Card className="p-3 text-[11px] text-muted/80 leading-relaxed">
            <div className="font-semibold text-foreground mb-1">What happens:</div>
            <ol className="list-decimal list-inside space-y-0.5 ml-1">
              <li>We extract text from the PDF (pymupdf, no LLM).</li>
              <li>You confirm / fix title, authors, year, DOI, abstract — assign shelves.</li>
              <li>On commit: chunks indexed into ChromaDB + hypotheses extracted via Sonnet 4.6 (~20-60s).</li>
              <li>You land on <code>/research/papers/{"{id}"}</code> with the new paper + hypotheses visible.</li>
            </ol>
          </Card>
        </>
      )}

      {/* Step 2 — review & confirm */}
      {phase === "review" && (
        <>
          <Card className="p-0 overflow-hidden">
            <div className="p-3 border-b border-border/30 flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-muted/70">
                {t("new.step2.title")}
              </span>
              {extractStats && (
                <span className="text-[10px] text-muted/60 ml-2">
                  {extractStats.pages} · {extractStats.chars.toLocaleString()} {t("new.paste.chars")}
                </span>
              )}
              <button onClick={resetAll}
                className="ml-auto text-[10px] text-muted hover:text-foreground inline-flex items-center gap-1">
                <X className="h-3 w-3" /> {t("new.step2.start_over")}
              </button>
            </div>

            <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3">
              {/* Metadata form — title + abstract are load-bearing for
                  hypothesis extraction; the rest collapsed into an
                  optional accordion (2026-06-06 simplification). */}
              <div className="space-y-2">
                <Field label={t("new.field.title")} required>
                  <input value={title} onChange={(e) => setTitle(e.target.value)}
                         className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1" />
                </Field>
                <Field label={t("new.field.abstract")} required>
                  <textarea value={abstract} onChange={(e) => setAbstract(e.target.value)}
                            rows={6}
                            className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1 leading-relaxed" />
                  <p className="mt-1 text-[10px] text-muted/60 leading-snug">
                    {t("new.field.abstract.hint")}
                  </p>
                </Field>

                <details
                  open={showMoreMetadata}
                  onToggle={(e) => setShowMoreMetadata(
                    (e.target as HTMLDetailsElement).open)}
                  className="rounded border border-border/30 bg-panel2/15">
                  <summary className="cursor-pointer text-[10.5px] text-muted/80 hover:text-foreground px-2 py-1.5 select-none">
                    {t("new.section.more_metadata")}
                  </summary>
                  <div className="p-2 space-y-2 border-t border-border/30">
                    <Field label={t("new.field.authors")}>
                      <textarea value={authors.join("\n")}
                                onChange={(e) => setAuthors(e.target.value.split("\n").map((s) => s.trim()).filter(Boolean))}
                                rows={2}
                                className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1 font-mono" />
                    </Field>
                    <div className="grid grid-cols-3 gap-2">
                      <Field label={t("new.field.year")}>
                        <input type="number" value={year}
                               onChange={(e) => setYear(e.target.value ? Number(e.target.value) : "")}
                               className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1 tnum" />
                      </Field>
                      <Field label={t("new.field.venue")}>
                        <input value={venue} onChange={(e) => setVenue(e.target.value)}
                               className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1" />
                      </Field>
                      <Field label={t("new.field.doi")}>
                        <input value={doi} onChange={(e) => setDoi(e.target.value)}
                               className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1 font-mono" />
                      </Field>
                    </div>
                  </div>
                </details>
              </div>

              {/* Intent + ingestion_reason + text preview.
                  Shelves removed from this step — auto-defaulted to
                  ["other"], refined by verdict pipeline post-test. */}
              <div className="space-y-3">
                <Field label={t("new.field.intent")}>
                  <div className="space-y-1">
                    {[
                      { key: "expand_breadth",          labelKey: "new.intent.expand_breadth" },
                      { key: "improve_existing_sleeve", labelKey: "new.intent.improve_sleeve" },
                      { key: "methodology_borrow",      labelKey: "new.intent.methodology" },
                      { key: "other",                   labelKey: "new.intent.other" },
                    ].map((opt) => (
                      <label key={opt.key}
                             className={cn(
                               "flex items-center gap-2 rounded px-2 py-1 text-[12px] cursor-pointer transition-colors",
                               intent === opt.key
                                 ? "bg-accent/10 border border-accent/40 text-foreground"
                                 : "border border-border/30 text-muted hover:text-foreground",
                             )}>
                        <input type="radio" name="intent"
                               checked={intent === opt.key}
                               onChange={() => setIntent(opt.key as IntentKey)}
                               className="accent-accent" />
                        <span>{t(opt.labelKey)}</span>
                      </label>
                    ))}
                  </div>
                  <p className="mt-1.5 text-[10px] text-muted/60 leading-snug">
                    {t("new.shelves.auto_note")}
                  </p>
                </Field>

                {/* Phase 1.7 step 3 (2026-06-06): ingestion reason.
                    Whoever PICKED the paper authors the reason. */}
                <Field
                  label={ingestionSource === "agent"
                    ? t("new.field.reason.agent")
                    : t("new.field.reason.user")}>
                  {ingestionSource === "agent" ? (
                    <div className="space-y-1.5">
                      <div className="rounded border border-accent/30 bg-accent/[0.04] text-[12px] px-2 py-1.5 leading-relaxed">
                        <span className="text-[10px] uppercase tracking-wider text-accent/80 font-semibold mr-1.5">agent</span>
                        {ingestionReason || t("new.field.reason.empty")}
                      </div>
                      <button
                        type="button"
                        onClick={() => {
                          setIngestionReason("");
                          setIngestionSource("user");
                        }}
                        className="text-[10.5px] text-muted hover:text-accent transition-colors">
                        {t("new.field.reason.write_own")}
                      </button>
                    </div>
                  ) : (
                    <>
                      <textarea
                        value={ingestionReason}
                        onChange={(e) => {
                          setIngestionReason(e.target.value.slice(0, 200));
                          // any edit means USER authorship — already set
                        }}
                        rows={2}
                        placeholder={t("new.field.reason.ph")}
                        className="w-full rounded border border-border/40 bg-panel2/30 text-[12px] px-2 py-1.5 leading-relaxed resize-y" />
                      <div className="text-[10px] text-muted/60 tnum mt-0.5">
                        {t("new.field.reason.counter").replace("{n}", String(ingestionReason.length))}
                      </div>
                    </>
                  )}
                </Field>

                <Field label={t("new.field.text_preview")}>
                  <pre className="rounded border border-border/40 bg-bg/40 text-[10.5px] font-mono p-2 max-h-[180px] overflow-y-auto whitespace-pre-wrap leading-relaxed">
                    {textPreview || t("new.field.text_preview_empty")}
                  </pre>
                </Field>
              </div>
            </div>

            <div className="p-3 border-t border-border/30 flex items-center gap-2">
              <span className="text-[10.5px] text-muted/70">
                {t("new.commit.hint")}
              </span>
              <button onClick={handleIngest}
                disabled={!title.trim() || !abstract.trim()}
                className="ml-auto inline-flex items-center gap-1.5 rounded bg-accent text-background hover:bg-accent/90 disabled:opacity-50 px-3 py-1.5 text-[12px] font-semibold">
                {t("new.commit.button")} <ArrowRight className="h-3 w-3" />
              </button>
            </div>
          </Card>
        </>
      )}

      {/* Ingesting (spinner) */}
      {phase === "ingesting" && (
        <Card className="p-6 text-center space-y-2">
          <Loader2 className="h-7 w-7 text-accent animate-spin mx-auto" strokeWidth={2} />
          <p className="text-[13px] text-foreground/85">
            Indexing chunks + extracting hypotheses…
          </p>
          <p className="text-[11px] text-muted/70">
            Don't close this tab. Typical 20-60s. Long papers can take longer.
          </p>
        </Card>
      )}

      {/* Done */}
      {phase === "done" && result && (
        <Card className="p-4 border-ok/30 bg-ok/[0.04] space-y-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-ok" />
            <span className="text-[13px] font-semibold">
              Ingested · paper_id <code className="text-accent">{result.paper_id.slice(0, 8)}</code>
            </span>
          </div>
          <div className="text-[12px] text-muted leading-relaxed">
            <b className="text-foreground">{result.title}</b>{" "}
            · <span className="text-foreground">{result.n_chunks}</span> chunks
            · <span className="text-foreground">{result.n_hypotheses}</span> hypotheses extracted
          </div>
          <div className="flex flex-wrap gap-2">
            <Link href={result.next_url}
              className="inline-flex items-center gap-1 rounded bg-accent text-background hover:bg-accent/90 px-3 py-1.5 text-[12px] font-semibold">
              Open paper page <ArrowRight className="h-3 w-3" />
            </Link>
            <Link href="/research/forward"
              className="inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 px-3 py-1.5 text-[12px]">
              Forward vectors
            </Link>
            <button onClick={resetAll}
              className="ml-auto inline-flex items-center gap-1 rounded border border-border/40 text-muted hover:text-foreground px-3 py-1.5 text-[12px]">
              Ingest another
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}


function Field({ label, required, children }: {
  label:     string;
  required?: boolean;
  children:  React.ReactNode;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-[10px] uppercase tracking-wider text-muted/70">
        {label}{required && <span className="text-danger ml-0.5">*</span>}
      </span>
      {children}
    </label>
  );
}
