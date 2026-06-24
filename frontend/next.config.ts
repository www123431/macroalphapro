import type { NextConfig } from "next";

// Static export: `next build` emits a fully static site to frontend/out/, which the
// FastAPI backend serves at one origin (so http://localhost:8000 = the whole app, no
// second server, no CORS). `next dev` is unaffected (hot-reload dev still works on :3000,
// hitting the API on :8000 via .env.development).
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  // Next.js dev indicator: hidden because every corner is already
  // occupied by our own UI — bottom-left = LabSideRail footer, bottom-
  // right = LivenessPill, top corners = topbar. Static-vs-dynamic
  // route info is still available via `next build --debug` output.
  devIndicators: false,
};

export default nextConfig;
