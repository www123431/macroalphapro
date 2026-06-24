"use client";

import { createContext, useContext, useState } from "react";

// Global "as-of" time-travel state. null = LIVE (latest); a "YYYY-MM-DD" date pins the book/risk/
// holdings views to that historical artifact. Shared across pages + the nav picker so switching
// pages keeps the selected date. The data refresh / decay monitor are forward-looking and ignore it.
const Ctx = createContext<{ asOf: string | null; setAsOf: (d: string | null) => void }>({
  asOf: null,
  setAsOf: () => {},
});

export function AsOfProvider({ children }: { children: React.ReactNode }) {
  const [asOf, setAsOf] = useState<string | null>(null);
  return <Ctx.Provider value={{ asOf, setAsOf }}>{children}</Ctx.Provider>;
}

export const useAsOf = () => useContext(Ctx);
