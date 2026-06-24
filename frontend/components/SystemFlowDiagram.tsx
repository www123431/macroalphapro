"use client";

// SystemFlowDiagram — teaching-first architecture viz.
//
// Shows how a single hypothesis travels through the system:
//
//   INGEST → TRIAGE → TEST → VERDICT → DEPLOY (LIVE / GRAVEYARD)
//
// Each box is a real UI surface (clickable, routes the user there).
// Each label shows the LIVE counter from the corresponding API
// endpoint so an onboarding quant can see "where things are stuck"
// vs. just "what the system can do".
//
// Why custom SVG over ECharts: this diagram is a NARRATIVE — every
// box / arrow / label is hand-placed for didactic clarity. ECharts
// graph layout drifts; we want pixel-perfect for the explanation.
// Edges carry animated flow dots (CSS) so the eye picks up direction
// without needing arrowheads everywhere.

import { useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";


type Counts = {
  papers:        number | null;
  hypotheses:    number | null;
  hyps_untested: number | null;
  hyps_tested:   number | null;
  forward:       number | null;
  forward_approved: number | null;
  active_sessions: number | null;
  lessons_red:   number | null;
  lessons_green: number | null;
  deployed:      number | null;
};


function useSystemCounts() {
  const [c, setC] = useState<Counts>({
    papers: null, hypotheses: null, hyps_untested: null, hyps_tested: null,
    forward: null, forward_approved: null, active_sessions: null,
    lessons_red: null, lessons_green: null, deployed: null,
  });
  useEffect(() => {
    // Single Promise.all for cheap parallelism.
    Promise.all([
      fetch(`${API_BASE}/api/paper_chain/overview`).then((r) => r.json()),
      fetch(`${API_BASE}/api/paper_chain/forward-vectors?top=500`).then((r) => r.json()),
      fetch(`${API_BASE}/api/sessions/list?limit=200`).then((r) => r.json()),
      fetch(`${API_BASE}/api/paper_chain/lessons?include_legacy=true&limit=500`).then((r) => r.json()),
      fetch(`${API_BASE}/api/research/library/inventory`).then((r) => r.json()),
    ]).then(([ov, fwd, sess, lessons, lib]) => {
      const deployed = (lib.entries || []).filter((e: any) =>
        (e.purpose || "").startsWith("deployed") ||
        e.purpose === "deploy_replacement" ||
        e.purpose === "hedge_replacement"
      ).length;
      const fwdArr = Array.isArray(fwd) ? fwd : [];
      const sessArr = (sess?.sessions || []) as any[];
      const lessonsArr = Array.isArray(lessons) ? lessons : [];
      setC({
        papers:        ov?.papers?.total ?? null,
        hypotheses:    ov?.hypotheses?.total ?? null,
        hyps_untested: ov?.hypotheses?.untested ?? null,
        hyps_tested:   ov?.hypotheses?.tested ?? null,
        forward:       fwdArr.length,
        forward_approved: fwdArr.filter((v: any) => v.pm_status === "approved").length,
        active_sessions: sessArr.filter((s: any) =>
          s.state === "in_flight" || s.state === "pending_preflight"
        ).length,
        lessons_red:   lessonsArr.filter((L: any) => (L.verdict || "").toUpperCase().includes("RED")).length,
        lessons_green: lessonsArr.filter((L: any) => (L.verdict || "").toUpperCase().includes("GREEN")).length,
        deployed,
      });
    }).catch(() => { /* counts stay null */ });
  }, []);
  return c;
}


// ── Node definitions ─────────────────────────────────────────────


type NodeShape = {
  key:       string;
  label:     string;
  sublabel?: string;
  href:      string;
  x:         number;  // SVG coord (viewBox 0..1200)
  y:         number;  // SVG coord (viewBox 0..640)
  width:     number;
  height:    number;
  zone:      "ingest" | "triage" | "test" | "verdict" | "deploy";
  counter?:  (c: Counts) => { value: string; sub?: string };
};


// Zone tones lifted from VZ.zone — single source of truth, see
// @/lib/vizTokens.
const ZONE_TONE: Record<string, { fill: string; stroke: string; text: string }> = VZ.zone;


// Layout chosen for pixel-perfect didactic flow. Coords manual.
const NODES: NodeShape[] = [
  // ── INGEST ──
  { key: "paper_input", label: "PDF / URL",      sublabel: "ingest",
    href: "/research/papers", zone: "ingest",
    x:  40, y:  60, width: 140, height: 56 },

  { key: "chunker", label: "Chunker",            sublabel: "1000-tok windows",
    href: "/research/papers", zone: "ingest",
    x: 220, y:  60, width: 140, height: 56 },

  { key: "embeddings", label: "Embeddings",      sublabel: "ChromaDB",
    href: "/research/papers", zone: "ingest",
    x: 400, y:  60, width: 140, height: 56 },

  { key: "extractor", label: "Hypothesis Extractor", sublabel: "Sonnet 4.6, verbatim quotes",
    href: "/research/papers", zone: "ingest",
    x: 580, y:  60, width: 200, height: 56 },

  { key: "registry", label: "Paper Registry",    sublabel: "papers_registry.jsonl",
    href: "/research/papers", zone: "ingest",
    x: 220, y: 150, width: 200, height: 56,
    counter: (c) => ({ value: `${c.papers ?? "—"} papers` }) },

  { key: "hyp_store", label: "Hypothesis Store", sublabel: "hypotheses.jsonl",
    href: "/research/forward", zone: "ingest",
    x: 460, y: 150, width: 220, height: 56,
    counter: (c) => ({ value: `${c.hypotheses ?? "—"} hypotheses`, sub: c.hyps_untested != null ? `${c.hyps_untested} untested` : undefined }) },

  // ── TRIAGE ──
  { key: "forward", label: "Forward vectors",    sublabel: "untested · prioritized",
    href: "/research/forward", zone: "triage",
    x:  60, y: 270, width: 200, height: 60,
    counter: (c) => ({ value: `${c.forward ?? "—"} ready`, sub: c.forward_approved != null ? `${c.forward_approved} PM-approved` : undefined }) },

  { key: "approval", label: "PM Approval",       sublabel: "extracted → approved",
    href: "/research/forward", zone: "triage",
    x: 300, y: 270, width: 180, height: 60 },

  { key: "session", label: "Session",            sublabel: "typed protocol · Claude",
    href: "/research/sessions", zone: "triage",
    x: 520, y: 270, width: 180, height: 60,
    counter: (c) => ({ value: c.active_sessions != null ? `${c.active_sessions} active` : "—" }) },

  // ── TEST ──
  { key: "candidate", label: "Candidate pipeline", sublabel: "9-step strict gate",
    href: "/research/candidate", zone: "test",
    x: 740, y: 270, width: 220, height: 60 },

  { key: "gates", label: "Graveyard · Cousin · DeflSR · Bootstrap · FF5+UMD",
    sublabel: "decision math (0 LLM)",
    href: "/research/candidate", zone: "test",
    x: 740, y: 360, width: 380, height: 56 },

  // ── VERDICT ──
  { key: "verdict_red", label: "RED Lesson",     sublabel: "killed · paper-grounded",
    href: "/research/lessons?verdict=red&include_legacy=true", zone: "verdict",
    x: 320, y: 440, width: 200, height: 60,
    counter: (c) => ({ value: c.lessons_red != null ? `${c.lessons_red} RED` : "—" }) },

  { key: "verdict_green", label: "GREEN Lesson", sublabel: "kept · ready to deploy",
    href: "/research/lessons", zone: "verdict",
    x: 560, y: 440, width: 200, height: 60,
    counter: (c) => ({ value: c.lessons_green != null ? `${c.lessons_green} GREEN` : "—" }) },

  { key: "memory", label: "Doctrine memory",     sublabel: "MEMORY.md + lesson refs",
    href: "/research/lessons", zone: "verdict",
    x: 800, y: 440, width: 200, height: 60 },

  // ── DEPLOY ──
  { key: "slm", label: "SLM Lifecycle",          sublabel: "PROPOSED → LIVE → DECAY",
    href: "/research/library", zone: "deploy",
    x: 560, y: 540, width: 220, height: 56,
    counter: (c) => ({ value: c.deployed != null ? `${c.deployed} deployed` : "—" }) },

  { key: "book", label: "Book + Risk",           sublabel: "paper-trade · daily 06:30",
    href: "/book", zone: "deploy",
    x: 820, y: 540, width: 180, height: 56 },
];


type Edge = {
  from:   string;
  to:     string;
  curveY?: number;  // extra curvature
  label?: string;
  animated?: boolean;
};

const EDGES: Edge[] = [
  // Ingest pipeline
  { from: "paper_input", to: "chunker",     animated: true },
  { from: "chunker",     to: "embeddings",  animated: true },
  { from: "embeddings",  to: "extractor",   animated: true },
  { from: "chunker",     to: "registry" },
  { from: "extractor",   to: "hyp_store",   animated: true },

  // Ingest → Triage
  { from: "hyp_store",   to: "forward",     animated: true },

  // Triage pipeline
  { from: "forward",     to: "approval",    animated: true },
  { from: "approval",    to: "session",     animated: true, label: "approved" },

  // Triage → Test
  { from: "session",     to: "candidate",   animated: true },

  // Test pipeline
  { from: "candidate",   to: "gates",       animated: true },

  // Test → Verdict (branching)
  { from: "gates",       to: "verdict_red",   label: "fail" },
  { from: "gates",       to: "verdict_green", label: "pass" },

  // Verdict → Memory
  { from: "verdict_red", to: "memory" },
  { from: "verdict_green", to: "memory" },

  // GREEN → Deploy
  { from: "verdict_green", to: "slm",       animated: true },
  { from: "slm",         to: "book",        animated: true },
];


function nodeCenter(n: NodeShape): { x: number; y: number } {
  return { x: n.x + n.width / 2, y: n.y + n.height / 2 };
}

function edgePoints(from: NodeShape, to: NodeShape): { d: string; midX: number; midY: number } {
  // Right-side of "from" → left-side of "to" when horizontal; bottom→top when vertical.
  // Heuristic: pick the edge based on relative position.
  const fc = nodeCenter(from);
  const tc = nodeCenter(to);
  const dx = tc.x - fc.x;
  const dy = tc.y - fc.y;
  let sx: number, sy: number, ex: number, ey: number;

  if (Math.abs(dy) > Math.abs(dx)) {
    // Mostly vertical
    if (dy > 0) {  // going down
      sx = fc.x; sy = from.y + from.height;
      ex = tc.x; ey = to.y;
    } else {       // going up
      sx = fc.x; sy = from.y;
      ex = tc.x; ey = to.y + to.height;
    }
  } else {
    // Mostly horizontal
    if (dx > 0) {  // going right
      sx = from.x + from.width; sy = fc.y;
      ex = to.x;                ey = tc.y;
    } else {       // going left
      sx = from.x;              sy = fc.y;
      ex = to.x + to.width;     ey = tc.y;
    }
  }

  const midX = (sx + ex) / 2;
  const midY = (sy + ey) / 2;
  // Bezier curve: control points offset perpendicular to the line.
  const cx1 = midX, cy1 = sy;
  const cx2 = midX, cy2 = ey;
  const d = `M ${sx} ${sy} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${ex} ${ey}`;
  return { d, midX, midY };
}


export function SystemFlowDiagram() {
  const counts = useSystemCounts();
  const nodesByKey = Object.fromEntries(NODES.map((n) => [n.key, n]));

  return (
    <div className="w-full">
      <svg viewBox="0 0 1200 640" className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
          </marker>
          <marker id="arrow-accent" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#7aa2f7" />
          </marker>
        </defs>

        {/* Zone backdrops — soft tint per zone */}
        <rect x="20"  y="30"  width="780" height="200" rx="6"
              fill={ZONE_TONE.ingest.fill} stroke={ZONE_TONE.ingest.stroke} strokeDasharray="4 4" strokeWidth="1" />
        <text x="32" y="48" fontSize="10" fontWeight="700" letterSpacing="2" fill={ZONE_TONE.ingest.text}>
          INGEST  ·  PDF → registry + hypotheses
        </text>

        <rect x="20"  y="250" width="700" height="100" rx="6"
              fill={ZONE_TONE.triage.fill} stroke={ZONE_TONE.triage.stroke} strokeDasharray="4 4" strokeWidth="1" />
        <text x="32" y="246" fontSize="10" fontWeight="700" letterSpacing="2" fill={ZONE_TONE.triage.text}>
          TRIAGE  ·  what should I test next?
        </text>

        <rect x="730" y="250" width="450" height="180" rx="6"
              fill={ZONE_TONE.test.fill} stroke={ZONE_TONE.test.stroke} strokeDasharray="4 4" strokeWidth="1" />
        <text x="742" y="246" fontSize="10" fontWeight="700" letterSpacing="2" fill={ZONE_TONE.test.text}>
          TEST  ·  strict-gate pipeline (0 LLM in decision)
        </text>

        <rect x="300" y="420" width="720" height="100" rx="6"
              fill={ZONE_TONE.verdict.fill} stroke={ZONE_TONE.verdict.stroke} strokeDasharray="4 4" strokeWidth="1" />
        <text x="312" y="416" fontSize="10" fontWeight="700" letterSpacing="2" fill={ZONE_TONE.verdict.text}>
          VERDICT  ·  what we learned + doctrine memory
        </text>

        <rect x="540" y="520" width="480" height="100" rx="6"
              fill={ZONE_TONE.deploy.fill} stroke={ZONE_TONE.deploy.stroke} strokeDasharray="4 4" strokeWidth="1" />
        <text x="552" y="516" fontSize="10" fontWeight="700" letterSpacing="2" fill={ZONE_TONE.deploy.text}>
          DEPLOY  ·  SLM lifecycle → book
        </text>

        {/* Edges */}
        {EDGES.map((e, i) => {
          const from = nodesByKey[e.from];
          const to   = nodesByKey[e.to];
          if (!from || !to) return null;
          const { d, midX, midY } = edgePoints(from, to);
          const stroke = e.animated ? "#7aa2f7" : "#475569";
          return (
            <g key={i}>
              <path d={d} fill="none" stroke={stroke}
                    strokeWidth={e.animated ? 1.5 : 1}
                    strokeDasharray={e.animated ? "4 6" : "none"}
                    markerEnd={`url(#${e.animated ? "arrow-accent" : "arrow"})`}>
                {e.animated && (
                  <animate attributeName="stroke-dashoffset"
                           from="0" to="-30"
                           dur="1.6s" repeatCount="indefinite" />
                )}
              </path>
              {e.label && (
                <g>
                  <rect x={midX - 18} y={midY - 8} width="36" height="14" rx="3"
                        fill="rgba(15,23,42,0.92)" stroke="rgba(100,116,139,0.4)" strokeWidth="0.5" />
                  <text x={midX} y={midY + 2} fontSize="9" fontWeight="600" textAnchor="middle"
                        fill="#cbd5e1">
                    {e.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {NODES.map((n) => {
          const t = ZONE_TONE[n.zone];
          const counter = n.counter?.(counts);
          return (
            <Link key={n.key} href={n.href}>
              <g style={{ cursor: "pointer" }} className="group">
                <rect x={n.x} y={n.y} width={n.width} height={n.height} rx="8"
                      fill="#1e293b"
                      stroke={t.stroke} strokeWidth="1.5"
                      className="transition-colors group-hover:stroke-[#cbd5e1]" />
                <text x={n.x + n.width / 2} y={n.y + 22}
                      fontSize="12" fontWeight="600" textAnchor="middle"
                      fill="#e2e8f0">
                  {n.label}
                </text>
                {n.sublabel && (
                  <text x={n.x + n.width / 2} y={n.y + 36}
                        fontSize="9" textAnchor="middle"
                        fill="#94a3b8">
                    {n.sublabel}
                  </text>
                )}
                {counter && (
                  <>
                    <text x={n.x + n.width / 2} y={n.y + n.height - 10}
                          fontSize="10.5" fontWeight="600" textAnchor="middle"
                          fill={t.text}>
                      {counter.value}
                    </text>
                    {counter.sub && (
                      <text x={n.x + n.width / 2} y={n.y + n.height - 24}
                            fontSize="8.5" textAnchor="middle"
                            fill="#64748b">
                        {counter.sub}
                      </text>
                    )}
                  </>
                )}
              </g>
            </Link>
          );
        })}
      </svg>

      {/* Legend / how-to-read */}
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[10.5px] text-muted/70 px-1">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 border-t border-dashed" style={{ borderColor: "#7aa2f7" }} />
          animated edge — main data flow
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ background: "#475569" }} />
          static edge — secondary write or branch
        </span>
        <span className="inline-flex items-center gap-1.5">
          counters = live values from API · click any box to open it
        </span>
      </div>
    </div>
  );
}
