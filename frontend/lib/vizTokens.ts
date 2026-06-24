// frontend/lib/vizTokens.ts — JS-side mirror of the --vz-* CSS
// custom properties declared in app/globals.css. Visualization
// components (SVG, ECharts) need raw hex strings at render time;
// running getComputedStyle per cell would be both slow and SSR-fragile.
//
// CONTRACT: every value in this file MUST equal its --vz-* twin in
// globals.css. When you change one, change both — a follow-up
// CI check is queued (Round-2.x polish).
//
// Why centralize: my earlier R2 viz commits hardcoded the same hex
// in 5+ places. That made any future theme switch (light mode, high
// contrast, color-blind-safe variants) a hand-grep nightmare. This
// module is the single source.

export const VZ = {
  verdict: {
    red:      "#ef4444",
    green:    "#34d399",
    marginal: "#f59e0b",
    pending:  "#94a3b8",
  },
  slm: {
    PROPOSED:        "#7aa2f7",
    BACKTESTED:      "#9aa2c0",
    AUDITED:         "#9aa2c0",
    APPROVED:        "#5fa8d3",
    PAPER_TRADE:     "#f5c518",
    SHADOW:          "#f59e0b",
    LIVE:            "#34d399",
    DECAY_WATCH:     "#fb923c",
    RAMP_DOWN:       "#ef4444",
    DECOMMISSIONED:  "#64748b",
    UNTRACKED:       "#475569",
  },
  role: {
    alpha:        "#7aa2f7",
    diversifier:  "#cbd5e1",
    insurance:    "#ef4444",
    hedge:        "#fb923c",
    carry:        "#34d399",
    mom_hedge:    "#fb923c",
    weak_alpha:   "#9aa5b1",
  },
  corr: {
    positive: "#ef4444",   // unintended overlap = red flag
    negative: "#34d399",   // designed diversifier = green
  },
  // System-flow diagram zones (5 phases of the data pipeline)
  zone: {
    ingest:  { text: "#7aa2f7", fill: "rgba(122,162,247,0.10)", stroke: "rgba(122,162,247,0.5)" },
    triage:  { text: "#f5c518", fill: "rgba(245,197,24,0.10)",  stroke: "rgba(245,197,24,0.5)"  },
    test:    { text: "#a77bf7", fill: "rgba(167,123,247,0.10)", stroke: "rgba(167,123,247,0.5)" },
    verdict: { text: "#cbd5e1", fill: "rgba(148,163,184,0.10)", stroke: "rgba(148,163,184,0.5)" },
    deploy:  { text: "#34d399", fill: "rgba(52,211,153,0.10)",  stroke: "rgba(52,211,153,0.5)"  },
  },
  // Visualization-neutral foreground/muted reused often.
  fg: {
    foreground: "#e2e8f0",
    muted:      "#94a3b8",
    mutedDim:   "#64748b",
    line:       "#475569",
    accent:     "#7aa2f7",
  },
} as const;
