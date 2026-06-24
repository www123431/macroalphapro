// frontend/lib/typography.ts — typographic scale convention.
//
// U5 2026-06-05 part 1 of 2. The codebase grew an organically-sized
// scale (text-[10px] / [10.5px] / [11px] / [11.5px] / [12px] / [12.5px]
// / [13px] / [13.5px]) that fights itself visually — every new component
// adds another half-step. This module fixes a 4-tier convention every
// NEW component must use. Existing components stay (avoid layout
// regression risk); future ones use these named constants.
//
// 4 tiers — matches Apple HIG / Material / Bloomberg conventions:
//
//   TY.meta    = text-[10px]      uppercase chips, timestamps,
//                                  citation source mono lines,
//                                  pagination, ALL <small>-class
//
//   TY.body    = text-[12px]      paragraph body, list items, normal
//                                  text in cards, tooltips
//
//   TY.emph    = text-[14px]      card titles, section labels,
//                                  data-value emphasis
//
//   TY.hero    = text-[16px]      page H1, hero numbers (NAV, Sharpe)
//
// Anything outside these 4 sizes is a code smell. Adopt one of them or
// document WHY (load-bearing chart label tick, etc).

export const TY = {
  meta: "text-[10px]",
  body: "text-[12px]",
  emph: "text-[14px]",
  hero: "text-[16px]",
} as const;


// Leading scale — paired with size tier.
export const TY_LEADING = {
  meta: "leading-tight",    // 1.1
  body: "leading-snug",     // 1.4
  emph: "leading-snug",     // 1.4
  hero: "leading-tight",    // 1.1
} as const;
