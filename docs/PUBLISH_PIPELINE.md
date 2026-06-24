# Publish pipeline — private dev → public GitHub mirror

This repo follows a **B1 dual-repo architecture**: continuous private
development; periodic sanitized snapshot to a separate public GitHub
mirror. The snapshot is fully reproducible — no manual file shuffling.

**Why dual-repo, not live-public?** Dev velocity here is ~10+
commits/day. Sanitize-on-every-commit is a permanent mental tax;
weekly snapshot is one disciplined chore that costs nothing per dev
commit. See INTERNAL_DESIGN_INDEX §10.

## Quick start — manual publish

```bash
# 1. dry-run (no files touched, just shows what would happen)
python scripts/publish/build_public_snapshot.py --dry-run

# 2. live snapshot (overwrites snapshot dir)
python scripts/publish/build_public_snapshot.py

# 3. inspect output
cat data/publish/snapshot_report.md
ls -la ${REPO_ROOT}/Desktop/macroalphapro-public/

# 4. (first time only) initialize the public git repo
cd ${REPO_ROOT}/Desktop/macroalphapro-public
git init
git add .
git commit -m "Initial public release"
git remote add origin git@github.com:www123431/macroalphapro.git
git push -u origin main

# 5. (subsequent runs) commit + push delta
cd ${REPO_ROOT}/Desktop/macroalphapro-public
git add -A
git commit -m "Weekly snapshot YYYY-MM-DD"
git push
```

## Architecture

```
                                                  ┌──────────────────┐
   private dev (C:/Users/.../intern)              │ rotate exposed   │
        │                                         │ keys BEFORE the  │
        │  build_public_snapshot.py               │ first push       │
        │  reads .publishrc.yaml                  │ (AV, GNEWS)      │
        ▼                                         └──────────────────┘
   ┌─────────────────────────┐
   │ 4-stage pipeline:       │
   │  1. collect (whitelist) │
   │  2. copy + sanitize     │
   │  3. post-check forbidden│
   │  4. write report        │
   └─────────────────────────┘
        │
        ▼
   C:/Users/.../macroalphapro-public/  ← clean, sanitized snapshot
        │  (manual git push, weekly)
        ▼
   github.com/www123431/macroalphapro   ← public mirror
```

## What gets included

Whitelist defined in `.publishrc.yaml` `include`:

- `engine/**/*.py` `engine/**/*.yaml` `engine/**/*.json` — core
  research + agent code (~31 MB)
- `scripts/**/*.py` — driver scripts (~3 MB)
- `tests/**/*.py` — test suite (~5 MB)
- `api/**/*.py` — FastAPI routes (~700 KB)
- `frontend/**/*.{ts,tsx,js,jsx,css,html,json}` excluding
  `node_modules/`, `.next/`, `build/`, `dist/` — Next.js app (~14 MB)
- `docs/**/*.md` `docs/**/*.tex` `docs/**/*.bib` `docs/figures/*.png`
  — papers + architecture
- Top-level: `README.md` `LICENSE` `PROJECT_OVERVIEW.md`
  `INTERNAL_DESIGN_INDEX.md` `CLAUDE.md` `.gitignore`
  `requirements.txt` `pyproject.toml` `package.json`
- `data/_samples/**` — hand-curated minimal sample data for tests

**Typical snapshot size: ~32 MB / ~2,250 files**
(vs ~8.4 GB / ~56,000 files in private dev — 0.4%).

## What's excluded

Blacklist in `.publishrc.yaml` `exclude` (applied AFTER include for
defense in depth):

- `data/` (except `_samples/`): per-session research artifacts,
  predictions, autopsies, paper cache, deployed sleeve attribution,
  PnL series — operational state that doesn't belong public.
- `.streamlit/`, secrets, `.env*` (except `.env.template`).
- All `*.db`, `*.sqlite*`, `*.log`, `__pycache__/`, `node_modules/`.
- Top-level personal binaries (`*.doc`, `*.docx`, `*.pdf`, `*.xlsx`).
- `app.py`, `app_dash.py` — Streamlit/Dash UIs deprecated by
  CLAUDE.md doctrine.
- `start.bat`, `dev.bat`, `*_wrapper.bat` — Windows-coupled
  automation; not useful to Linux/macOS users without adaptation.
- `Procfile`, `Dockerfile.streamlit` — legacy hosting.
- Cron registration scripts.

## Sanitize regex applied

Defined in `.publishrc.yaml` `sanitize_patterns`. Each match in any
included text file (`.py` `.md` `.toml` `.yaml` `.ts` etc.) is
replaced inline:

| Pattern | Replacement |
|---|---|
| `${REPO_ROOT}` / `${REPO_ROOT}` / `${REPO_ROOT}` | `${REPO_ROOT}` |
| `${USER}` (word boundary) | `${USER}` |
| `${WRDS_USER_1}` / `${WRDS_USER_2}` (WRDS users) | `${WRDS_USER_1}` / `${WRDS_USER_2}` |
| `${WRDS_PASS_1}` / `${WRDS_PASS_2}` (WRDS passwords) | `${WRDS_PASS_1}` / `${WRDS_PASS_2}` |
| `${WRDS_ALLOWED_IP}` (WRDS allowed IP) | `${WRDS_ALLOWED_IP}` |
| `${USER_EMAIL}` | `${USER_EMAIL}` |

Typical run replaces 80-100 substrings across 40-90 files.

## Post-check forbidden patterns

After the snapshot is built, the script greps the OUTPUT for
suspicious patterns; ANY hit fails the pipeline:

- `\b72360\b` / `\bzhang21\b` / `\bwang214\b` etc. — sanity check
  that sanitize_patterns did its job.
- `sk-ant-api[a-zA-Z0-9_-]{20,}` — real Anthropic API key shape
  (tightened so test fixtures like `sk-ant-fake` don't false-positive).
- `sk-proj-[a-zA-Z0-9]{20,}` — OpenAI project keys.
- `AIza[a-zA-Z0-9_-]{30,}` — Google API keys (~39 chars).
- `AV_KEY="..."` / `GNEWS_KEY="..."` with non-trivial value.

Add new patterns as new secret types appear. The list is the project's
trust boundary — be paranoid here, not parsimonious.

## Before the FIRST push (one-time)

```bash
# 1. ROTATE the API keys that were committed pre-cf06b833 historically.
#    Even though .streamlit/secrets.toml is gitignored NOW, the old
#    commits still contain the keys; rotation invalidates them so
#    the historical exposure doesn't matter.
#    - AV_KEY:    https://www.alphavantage.co/support/#api-key
#    - GNEWS_KEY: https://gnews.io/dashboard
#    Update .streamlit/secrets.toml with the new values.

# 2. Create the public GitHub repo.
#    github.com/new → name: macroalphapro → Public → no .gitignore (we
#    bring our own) → no LICENSE (we bring our own) → Create

# 3. Run the snapshot (see "Quick start" above) and push.
```

## Weekly refresh (after the first push)

Run the snapshot + push, ideally once a week. There's a wrapper bat
ready to be registered as a scheduled task:

```bash
# scripts/publish/weekly_snapshot_wrapper.bat
# Wraps build_public_snapshot.py + git push for cron use.
```

Register as a Windows scheduled task (run weekly Sunday 05:00):

```cmd
schtasks /Create /TN "MacroAlphaPro\weekly-public-snapshot" /SC WEEKLY ^
  /D SUN /ST 05:00 ^
  /TR "${REPO_ROOT}\Desktop\intern\scripts\publish\weekly_snapshot_wrapper.bat" /F
```

The wrapper logs to `data/publish/logs/`. Failed pushes emit a
non-zero exit code which schtasks records — check `schtasks /Query
/TN MacroAlphaPro\weekly-public-snapshot /V` for last status.

## Iteration loop when the snapshot fails

The post-check is loud and specific. If it fires:

1. Open `data/publish/snapshot_report.md`.
2. Each `[FAIL]` row names the file + pattern + snippet.
3. Two fixes possible:
   - **Real leak**: edit the source file in the private repo to
     remove / parameterize the secret, then re-run snapshot.
   - **False positive**: tighten the regex in `.publishrc.yaml`
     `post_check_forbidden` so the legitimate test fixture / mock
     stops matching.
4. Re-run snapshot. Repeat until `[OK] snapshot built clean`.

## Sanitize coverage audit (occasional)

Once a quarter, sanity-check that the include patterns still cover the
right surface area:

```bash
# How many files would be included today vs last snapshot:
python scripts/publish/build_public_snapshot.py --dry-run | tail -10

# What's the size delta:
du -sh ${REPO_ROOT}/Desktop/macroalphapro-public/
```

If sizes drift unexpectedly (e.g., a huge new file got pulled in by
glob expansion), update `.publishrc.yaml` `exclude` accordingly.
