# Phase 8a arXiv Rate-Limit Finding — 2026-05-30

## Observation

Real-data smoke of arxiv_qfin_fetcher returned 0 papers after 3 attempts
with exponential backoff (30s, 60s, 90s = 180s total wait). arXiv API
returned HTTP 429 on each attempt.

Likely cause: prior dev iterations on the same IP triggered arXiv's
short-term throttle protection. arXiv's stated polite-use rate is
~1 req/3sec; we exceeded that during iterative testing before adding
the polite delay.

## Why this is NOT a blocker for Phase 8a

- Mocked-API tests (15 tests) all pass — architecture verified
- The pipeline (fetcher → extractor → hygiene → queue) is sound
- arXiv will release the throttle within hours

## Production hardening options (when real auto-discovery becomes routine)

1. **Use arXiv RSS endpoint as fallback** — RSS has different rate
   limits (more generous). Build a Layer 2 fallback in the fetcher.

2. **Distributed IP / cloud cron** — when Phase 7 cron lands, it runs in
   cloud (not dev workstation), naturally separates IP from dev
   iterations.

3. **24-hour cool-off after 429** — track 429 timestamp in a
   source_health.json; skip arxiv entirely until cool-off expires.

4. **Multi-source dispatch** — when arxiv unavailable, the pipeline can
   continue with NBER / SSRN / Tier-1 RSS (Phase 8b sources).

## Recommended fix order

When this issue actually bites in production:
- First do option 4 (multi-source dispatch is needed for Phase 8b anyway)
- Then option 3 (cool-off tracking integrates with health monitoring)
- Option 1 (RSS fallback) only if arxiv API stays problematic
- Option 2 (cloud cron IP) happens naturally in Phase 7

## Not committing a code change yet

This is a real-world finding to address WHEN it blocks something.
At current 9-mechanism library scale, supply-side discovery is not the
binding bottleneck.

## Linked

- `engine/research/discovery/arxiv_qfin_fetcher.py` (the fetcher with
  current 30/60/90s backoff)
- [[project-auto-paper-discovery-next-priority-2026-05-29]] (Phase 8 plan;
  multi-source is in scope)
- [[feedback-iterate-and-solve-inflight-2026-05-29]] (find issue,
  document, defer until actually blocks something)
