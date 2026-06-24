// frontend/lib/intentRouter.test.ts — Smoke tests for intent detection.
//
// Run manually with: npx tsx frontend/lib/intentRouter.test.ts
// (not part of next build; ts-node / tsx execution)
//
// Senior note: these are PURE unit tests on the detectIntent regex
// router. They don't hit the API or LLM. If recall drops in the wild,
// add the failing pattern here as a regression test.

import { detectIntent } from "./intentRouter";

const cases: Array<[string, string | null]> = [
  // ── /pfh ─────────────────────────────────────────────
  ["show me pfh suggestions",            "pfh.suggest"],
  ["pfh 10",                              "pfh.suggest"],
  ["give me 6 pfh candidates",            "pfh.suggest"],
  ["generate factor candidates",          "pfh.suggest"],
  ["suggest factor variants",             "pfh.suggest"],
  // ── /decay ──────────────────────────────────────────
  ["decay",                               "decay.snapshot"],
  ["what sleeves are decaying",           "decay.snapshot"],
  ["sleeve health overview",              "decay.snapshot"],
  ["decay for equity_book",               "decay.sleeve"],
  // ── /council ────────────────────────────────────────
  ["show council runs",                   "council.list"],
  ["recent 5 council runs",               "council.list"],
  ["council 10 APPROVE",                  "council.list"],
  // ── /factor ─────────────────────────────────────────
  ["open factor eq_mom_12_1_us_real",     "factor.detail"],
  ["details for cross_asset_carry_4leg",  "factor.detail"],
  // ── /sleeve ─────────────────────────────────────────
  ["sleeve equity_book",                  "sleeve.timeline"],
  ["timeline carry_book",                 "sleeve.timeline"],
  // ── /library ────────────────────────────────────────
  ["library",                             "library.browse"],
  ["library post_earnings_drift",         "library.browse"],
  // ── /chains ─────────────────────────────────────────
  ["chains",                              "chains.list"],
  ["show chains",                         "chains.list"],
  // ── /help ───────────────────────────────────────────
  ["help",                                "help"],
  ["?",                                   "help"],
  ["what can you do",                     "help"],
  // ── Fall-through to /ask (null) ─────────────────────
  ["what's the highest Sharpe factor?",   null],
  ["how did D_PEAD perform in 2020?",     null],
  ["why is crisis_hedge alert level OK",  null],
  ["who reviewed the latest council run", null],
];

let pass = 0, fail = 0;
for (const [input, expected] of cases) {
  const got = detectIntent(input);
  const actual = got?.intent ?? null;
  if (actual === expected) {
    pass++;
  } else {
    fail++;
    console.error(`FAIL "${input}"\n   expected: ${expected ?? "null (→ /ask)"}\n   got: ${actual ?? "null"}`);
  }
}
console.log(`\n${pass}/${cases.length} pass · ${fail} fail`);
if (fail > 0) process.exit(1);
