"use client";

import { useEffect, useRef } from "react";
import type { EChartsOption } from "echarts";

// Thin lazy-loaded ECharts wrapper: dynamic-imports echarts (kept out of the main bundle),
// inits once on the ref, resizes with the container, re-applies option on change, disposes on
// unmount. Dark-theme colors are set per-option by callers (no global theme needed).
export function EChart({ option, height = 280, className = "" }: {
  option: EChartsOption; height?: number; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const inst = useRef<any>(null);
  const optionRef = useRef(option);
  optionRef.current = option;

  useEffect(() => {
    let disposed = false;
    let ro: ResizeObserver | null = null;
    (async () => {
      const echarts = await import("echarts");
      if (disposed || !ref.current) return;
      inst.current = echarts.init(ref.current, null, { renderer: "canvas" });
      inst.current.setOption(optionRef.current);
      ro = new ResizeObserver(() => inst.current?.resize());
      ro.observe(ref.current);
    })();
    return () => { disposed = true; ro?.disconnect(); inst.current?.dispose(); inst.current = null; };
  }, []);

  useEffect(() => { inst.current?.setOption(option, true); }, [option]);

  return <div ref={ref} style={{ height }} className={className} />;
}
