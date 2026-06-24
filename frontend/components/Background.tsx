// frontend/components/Background.tsx — animated ambient background (fixed, behind content).
// Drifting grid + two slow floating color glows + a vignette. Pure CSS (perf-friendly).
//
// 2026-06-02: optional `variant="lab"` overlays a subtle warm radial
// gradient. The visual signal: Production / Ops feel "operational
// cool"; Lab feels "studio warm". Same accent palette, different
// ambient tint — like switching from Cursor's settings view to its
// editor view. Subtle: opacity ≤ 0.04 so it never fights with content.
export function Background({ variant }: { variant?: "default" | "lab" } = {}) {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-0 bg-background" />
      <div className="bg-grid absolute inset-0 opacity-[0.07]" />
      <div className="bg-glow bg-glow-1 absolute" />
      <div className="bg-glow bg-glow-2 absolute" />
      {/* Lab mode: warm focus tint — radial from top-center outward.
          Use accent base color so it stays brand-consistent. */}
      {variant === "lab" && (
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(ellipse 90% 60% at 50% -10%, rgb(255 180 120 / 0.04), transparent 60%)",
          }}
        />
      )}
      {/* top sheen + bottom fade for depth */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent/40 to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-b from-background/0 via-background/0 to-background" />
    </div>
  );
}
