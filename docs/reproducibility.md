# Backtest reproducibility (S-3)

> Anyone — examiner, second-reader, future me — can re-run a backtest
> and get the **same** numbers years later. yfinance / FRED upstream changes
> won't break this, because the data is frozen into a parquet bundle at
> snapshot time.

---

## The 3-step workflow

### 1) Freeze a snapshot (once, at result-publication time)

```bash
python scripts/freeze_backtest_data.py \
    --start=2009-01-01 \
    --end=2026-05-06 \
    --name=thesis_v1
```

Output: `data/snapshots/thesis_v1_2026-05-06/` containing
- `yf_monthly_etf.parquet` — 18 ETF monthly returns
- `yf_daily_etf.parquet`   — 18 ETF daily prices (for Ledoit-Wolf cov)
- `yf_vix.parquet`         — `^VIX` daily (regime input)
- `fred_macros.parquet`    — DGS10 / DGS2 / BAMLH0A0HYM2 / VIXCLS
- `manifest.json`          — per-file sha256 + tickers + dates + git commit

### 2) Run backtest against the snapshot

```bash
python scripts/run_backtest.py \
    --start=2010-01-01 --end=2024-12-31 \
    --use-snapshot=thesis_v1_2026-05-06 \
    > my_run.json
```

`run_backtest.py` outputs JSON with `metrics_a` (TSMOM-only),
`metrics_b` (TSMOM + regime overlay), `metrics_bm` (60/40 SPY/AGG).
Headline figure for the project = `metrics_a.sharpe` under
`regime_scale=1.0` (production baseline; regime overlay disabled per
2026-05-02 baseline switch).

### 3) Reproduce by comparison

To verify the same setup gives the same number:

```bash
diff <(python scripts/run_backtest.py --start=... --use-snapshot=thesis_v1_2026-05-06) \
     my_run.json
```

The two runs should be byte-identical because:
- yfinance / FRED reads are replaced by frozen parquet (deterministic)
- Random seeds are not used in run_backtest core
- Floating-point pandas/numpy ops are deterministic on identical input

---

## What's covered vs. not covered

### ✅ Covered by snapshot
- Monthly ETF returns (used by signal layer)
- Daily ETF prices (used by Ledoit-Wolf covariance)
- ^VIX series (potential regime input — wired through but only relevant
  if `regime_scale != 1.0`)
- FRED macro series

### ⚠️ NOT covered (intentional)
- **SPY / AGG benchmark prices**: external benchmark, swapped frequently;
  always live-fetched. If you need fully-frozen benchmark data too, add
  them to `--tickers` when calling freeze_backtest_data.
- **`engine.regime.get_regime_on()` internal data fetches**: regime
  affects only `metrics_b` (overlay portfolio). At production
  `REGIME_SCALE=1.0`, overlay is identity — `metrics_a == metrics_b`. So
  for thesis-baseline reproducibility, regime data fetch is moot.
  Documented limitation, not a hidden bug. R-1.F backlog can extend
  snapshot to regime if needed for non-baseline reproductions.
- **Code state**: the snapshot's `manifest.json` records the git commit
  at freeze time (`code_version`). Verify by `git log -1` matches.

### Tamper detection

Every parquet file's sha256 is recorded in `manifest.json`. `load_snapshot`
re-hashes on read; mismatch raises `ValueError("sha256 mismatch — TAMPER")`.
Disable with `verify_hashes=False` only if you know exactly why.

---

## Anchoring a thesis number

The headline number for thesis defense:

| Strategy | Window | Snapshot | Expected metric |
|---|---|---|---|
| QL01 BAB | TODO at thesis-write time | TODO `thesis_v1_<date>` | TODO Sharpe = X.XX |

To produce an anchored result, run:

```bash
python scripts/run_backtest.py \
    --start=<thesis-window-start> \
    --end=<thesis-window-end> \
    --use-snapshot=<your-snapshot-id> \
    --regime-scale=1.0
```

Compare the JSON `metrics_a.sharpe` against the table above.

> Note: don't add hard-coded golden-number tests in `tests/`. Methodology
> changes deliberately would break those tests on every iteration. The
> reproducibility check is **manual** (one-line diff above) and should
> be re-anchored each time methodology changes — the table above is the
> system of record.

---

## How a future examiner reproduces

1. `git clone` the repo
2. `git checkout <code_version>` (from `manifest.json`)
3. The snapshot directory is part of the repo (or downloaded separately
   if too large for git — see `data/snapshots/.gitignore` rules)
4. `pip install -r requirements.txt`
5. Run command from the table above
6. Diff JSON against the published thesis numbers

If diff is non-zero, candidate causes:
- Different pandas / numpy version (rare; ops are deterministic but
  edge cases exist)
- Manifest sha256 mismatch — somebody tampered with the parquet
- Different python version — float subtleties
- Code drift since snapshot's `code_version`

---

## When to re-snapshot

- **Before publishing thesis numbers**: freeze a `thesis_v1_<date>` snapshot
- **Before each major paper revision**: freeze a new dated snapshot
- **Before a defense**: ensure the snapshot used for the headline number
  is on disk + sha256-verified
- **Don't re-snapshot for normal development**: live yfinance is fine; the
  snapshot is for the audit trail at publication time

---

## Implementation pointers

- `engine/data_snapshot.py` — freeze + load + slice helpers
- `scripts/freeze_backtest_data.py` — CLI for step 1
- `scripts/run_backtest.py` — CLI for step 2
- `engine/backtest.py:566` `run_backtest(snapshot=...)` — the new arg
- `tests/test_data_snapshot.py` — round-trip + tamper detection tests

7 unit tests cover the round-trip + integrity + slice helpers + the
"refuse silent overwrite" invariant (FileExistsError on duplicate
snapshot_id).
