// Shared helper: turn a research_store event's artifact path into a
// browser-safe URL, or null if it's not safe to link to.
//
// Why this exists: artifact paths are stored as raw strings written by
// Python (often with Windows backslashes from older emit calls), and the
// browser resolves a path like "data\\foo\\bar.md" relative to the
// current URL — which on Windows-served static export can navigate to
// the desktop. Worse, paths to data/cache, data/audit_verifier, etc are
// not exposed by the static server at all; rendering them as <a href>
// just gives the user a 404 or a broken navigation.
//
// Rules (conservative — when in doubt, don't link):
//   1. Strip Windows-style backslashes (normalize to /).
//   2. Reject if absolute (starts with / or contains ://).
//   3. Reject if it points outside docs/ (the only directory we
//      actually publish from frontend/out as static files).
//   4. Reject unless the file extension is one of the known web-safe
//      doc types: md, html, txt, pdf.

const _SAFE_EXTENSIONS = new Set([".md", ".html", ".htm", ".txt", ".pdf"]);
// Only paths inside these top-level prefixes are exposed via the
// FastAPI static mount and worth linking. Everything else (data/,
// engine/, scripts/, etc) is server-side state, not browser-fetchable.
const _PUBLIC_PREFIXES = ["docs/"];


export function safeArtifactHref(path: string | null | undefined): string | null {
  if (!path || typeof path !== "string") return null;
  const norm = path.replace(/\\/g, "/").trim();
  if (!norm) return null;
  if (norm.startsWith("/")) return null;
  if (norm.includes("://")) return null;
  const lower = norm.toLowerCase();
  const dot = lower.lastIndexOf(".");
  if (dot < 0) return null;
  const ext = lower.slice(dot);
  if (!_SAFE_EXTENSIONS.has(ext)) return null;
  if (!_PUBLIC_PREFIXES.some((p) => norm.startsWith(p))) return null;
  return `/${norm}`;
}
