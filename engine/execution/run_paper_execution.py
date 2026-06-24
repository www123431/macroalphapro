"""engine/execution/run_paper_execution.py — bridge the systematic book → execution layer.

Reads the book's CURRENT target weights (from the daily UI artifact the live pipeline already
writes: data/ui_artifact/<date>.json) and rebalances a paper account toward them. Default adapter
is the offline SimAdapter (no keys); pass --alpaca to route to the Alpaca paper account once
ALPACA_KEY/ALPACA_SECRET are configured.

This is pure plumbing downstream of the deterministic engine — it executes the weights the math
already produced, it does not decide anything (0-LLM-in-DECISION preserved).

  python -m engine.execution.run_paper_execution            # sim, dry-run preview
  python -m engine.execution.run_paper_execution --submit   # sim, actually fill + mark NAV
  python -m engine.execution.run_paper_execution --alpaca --submit
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os

logger = logging.getLogger(__name__)


def build_target_weights_from_artifact(path: str | None = None) -> tuple[dict[str, float], str]:
    """Combined per-ticker target weight = Σ strategy.book_weight × position.intra_weight.
    Returns (weights, artifact_date)."""
    if path is None:
        files = sorted(glob.glob("data/ui_artifact/*.json"))
        if not files:
            raise FileNotFoundError("no data/ui_artifact/*.json — run the daily pipeline first")
        path = files[-1]
    d = json.load(open(path, encoding="utf-8"))
    bw = {s["strategy_name"]: float(s.get("book_weight", 0.0)) for s in d.get("strategy_states", [])}
    weights: dict[str, float] = {}
    for p in d.get("positions", []):
        tk = p.get("ticker")
        if not tk:
            continue
        w = bw.get(p.get("strategy_name"), 0.0) * float(p.get("intra_weight", 0.0))
        weights[tk] = weights.get(tk, 0.0) + w
    weights = {k: v for k, v in weights.items() if abs(v) > 1e-9}
    asof = d.get("_meta", {}).get("date") or os.path.basename(path)[:10]
    return weights, asof


def fetch_prices(tickers: list[str]) -> dict[str, float]:
    """Last close via yfinance (best-effort; tickers that fail are dropped)."""
    try:
        import yfinance as yf
    except Exception:
        logger.warning("yfinance unavailable — no prices")
        return {}
    out: dict[str, float] = {}
    try:
        data = yf.download(tickers, period="5d", progress=False, auto_adjust=True)
        close = data["Close"] if "Close" in data else data
        last = close.ffill().iloc[-1]
        for tk in tickers:
            try:
                px = float(last[tk]) if tk in last.index else float(last)
                if px > 0:
                    out[tk] = px
            except Exception:
                continue
    except Exception as exc:
        logger.warning("price fetch failed: %s", exc)
    return out


def run(use_alpaca: bool = False, submit: bool = False, starting_cash: float = 1_000_000.0,
        also_sim_fallback: bool = False, ignore_halt: bool = False) -> dict:
    # ── II.D auto-halt gate (Phase 1 Task II.D of research_agenda_2026-05-29) ──
    # If a pre-existing halt flag is active AND not human-acknowledged, refuse
    # to submit new orders. Dry-run is always allowed (so a human can preview
    # what the book would do without committing). ignore_halt=True is the
    # explicit override (caller takes responsibility).
    if submit and not ignore_halt:
        from engine.agents.anomaly_sentinel.auto_halt import is_halt_active
        halt_active, halt_payload = is_halt_active()
        if halt_active:
            return {
                "as_of":            None,
                "halted":           True,
                "halt_reason":      halt_payload.get("suggested_action") if halt_payload else "halt_flag.json present",
                "halt_payload":     halt_payload,
                "action_required":  ("delete data/paper_trade/halt_flag.json OR call "
                                      "engine.agents.anomaly_sentinel.auto_halt."
                                      "acknowledge_halt() after human review"),
            }

    weights, asof = build_target_weights_from_artifact()
    gross = sum(abs(w) for w in weights.values())
    tickers = sorted(weights)
    weights_original = dict(weights)              # save BEFORE Alpaca's tradable_filter

    from engine.execution.rebalancer import rebalance
    dropped: dict[str, str] = {}
    sim_fallback_report: dict | None = None
    if use_alpaca:
        from engine.execution.alpaca_adapter import AlpacaAdapter
        adapter = AlpacaAdapter()                 # raises if not paper-configured; fetches own prices
        weights, dropped = adapter.tradable_filter(weights)   # drop non-tradable + non-shortable shorts
        tickers = sorted(weights)
        prices = adapter.get_prices(tickers)      # for reporting count (rebalance refetches internally)
        allow_fractional = False                  # Alpaca: no fractional shorts / closed-mkt frac → whole shares
    else:
        from engine.execution.sim_adapter import SimAdapter
        adapter = SimAdapter(starting_cash=starting_cash)
        prices = fetch_prices(tickers)            # yfinance; sim needs an injected price source
        adapter.set_prices(prices)
        allow_fractional = True

    rep = rebalance(adapter, weights, dry_run=not submit, allow_fractional=allow_fractional)
    if submit and not use_alpaca:
        adapter.mark_nav(asof)                    # accumulate the paper NAV track

    # Multi-venue fallback: send Alpaca-rejected tickers to SimAdapter so the paper book is COMPLETE.
    # Alpaca can't borrow some ETFs (DBA/ICLN/USO/VXX) and structurally can't trade mutual funds
    # (e.g. PQTIX = 11% of book, used as CTA-sleeve proxy). The internal SimAdapter has no broker
    # constraints — it gives us the "as if there were no shorting limits" view, marked daily on
    # yfinance closes. multi_venue.py consolidates Alpaca + Sim into one combined NAV later.
    if use_alpaca and also_sim_fallback and dropped:
        from engine.execution.sim_adapter import SimAdapter
        fallback_weights = {t: weights_original[t] for t in dropped if t in weights_original}
        if fallback_weights:
            fb_tickers = sorted(fallback_weights)
            fb_prices = fetch_prices(fb_tickers)
            sim = SimAdapter(starting_cash=starting_cash,
                             state_path="data/execution/sim_fallback_state.json")
            sim.set_prices(fb_prices)
            sim_rep = rebalance(sim, fallback_weights, dry_run=not submit, allow_fractional=True)
            if submit:
                sim.mark_nav(asof)
            sim_fallback_report = {
                "venue": "sim_fallback",
                "reason": "Alpaca rejected (unborrowable / not_tradable / mutual_fund)",
                "weights_picked_up": {t: round(w, 4) for t, w in fallback_weights.items()},
                "gross": round(sum(abs(w) for w in fallback_weights.values()), 4),
                "n_priced": len(fb_prices),
                "n_unpriced": len(fb_tickers) - len(fb_prices),
                "report": sim_rep.to_dict(),
            }
            if submit:
                sim_fallback_report["sim_equity_after"] = round(sim.get_account().equity, 2)

    out = {"as_of": asof, "n_tickers": len(tickers), "gross_weight": round(gross, 4),
           "n_priced": len(prices), "dropped_untradable": dropped, "report": rep.to_dict()}
    if submit and not use_alpaca:
        out["sim_equity_after"] = round(adapter.get_account().equity, 2)
    if sim_fallback_report is not None:
        out["sim_fallback"] = sim_fallback_report
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca", action="store_true", help="route to Alpaca paper (needs keys)")
    ap.add_argument("--submit", action="store_true", help="actually submit (default = dry-run)")
    ap.add_argument("--also-sim-fallback", action="store_true",
                    help="when --alpaca, route Alpaca-rejected tickers to SimAdapter for 100% paper coverage")
    ap.add_argument("--ignore-halt", action="store_true",
                    help="OVERRIDE: submit despite active halt_flag (caller takes responsibility)")
    args = ap.parse_args()
    print(json.dumps(run(use_alpaca=args.alpaca, submit=args.submit,
                          also_sim_fallback=args.also_sim_fallback,
                          ignore_halt=args.ignore_halt),
                     indent=2, ensure_ascii=False))
