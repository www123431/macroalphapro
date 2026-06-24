"""engine/execution/alpaca_adapter.py — Alpaca PAPER trading adapter (free, no real capital).

Implements ExecutionAdapter against Alpaca's paper API (https://paper-api.alpaca.markets) using the
REST endpoints directly (urllib — no SDK dependency, matching engine/news_fetchers.py's pattern).

Credentials come from secrets.toml / env (graceful skip when absent, like AV_KEY/GNEWS_KEY):
  ALPACA_KEY     — paper API key id   (APCA-API-KEY-ID)
  ALPACA_SECRET  — paper API secret   (APCA-API-SECRET-KEY)
  ALPACA_BASE    — optional; defaults to the PAPER endpoint

SAFETY (no real money in this project): the constructor HARD-REFUSES any base URL that is not the
paper endpoint. is_paper is derived from the URL, and the rebalancer additionally refuses to submit
when is_paper is False — so a live endpoint can never be traded even by misconfiguration.

Setup for the user: sign up at alpaca.markets (free, no funding) → Paper Trading → generate API
keys → put ALPACA_KEY / ALPACA_SECRET in .streamlit/secrets.toml. Then construct AlpacaAdapter().
"""
from __future__ import annotations

import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from engine.execution.broker import Account, ExecutionAdapter, Fill, Order, Position

logger = logging.getLogger(__name__)

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


def _get_secret(name: str) -> str | None:
    try:
        import streamlit as st
        v = st.secrets.get(name)
        if v:
            return v
    except Exception:
        pass
    import os
    return os.environ.get(name)


class AlpacaConfigError(RuntimeError):
    pass


class AlpacaAdapter(ExecutionAdapter):
    name = "alpaca_paper"

    def __init__(self, base_url: str | None = None,
                 key: str | None = None, secret: str | None = None):
        self.base = (base_url or _get_secret("ALPACA_BASE") or PAPER_BASE).rstrip("/")
        # HARD paper-only guard: refuse anything that isn't the paper endpoint.
        if "paper-api.alpaca.markets" not in self.base:
            raise AlpacaConfigError(
                f"refusing non-paper Alpaca endpoint '{self.base}' — this project trades no real "
                "capital. Only the paper endpoint is allowed.")
        self.key = key or _get_secret("ALPACA_KEY")
        self.secret = secret or _get_secret("ALPACA_SECRET")
        if not (self.key and self.secret):
            raise AlpacaConfigError(
                "ALPACA_KEY / ALPACA_SECRET not configured (secrets.toml or env). "
                "Sign up at alpaca.markets → Paper Trading → generate keys.")

    @property
    def is_paper(self) -> bool:
        return "paper-api.alpaca.markets" in self.base

    # ---- HTTP helpers ----
    def _hdr(self) -> dict:
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret,
                "Content-Type": "application/json"}

    def _get(self, url: str) -> dict | list:
        req = Request(url, headers=self._hdr())
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, url: str, payload: dict) -> dict:
        req = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers=self._hdr(), method="POST")
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ---- adapter surface ----
    def get_account(self) -> Account:
        a = self._get(f"{self.base}/v2/account")
        return Account(equity=float(a.get("equity", 0.0)),
                       cash=float(a.get("cash", 0.0)),
                       buying_power=float(a.get("buying_power", 0.0)))

    def get_positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._get(f"{self.base}/v2/positions"):
            tk = p.get("symbol", "")
            out[tk] = Position(ticker=tk, qty=float(p.get("qty", 0.0)),
                               avg_price=float(p.get("avg_entry_price", 0.0)),
                               market_value=float(p.get("market_value", 0.0)))
        return out

    def get_prices(self, tickers: list[str]) -> dict[str, float]:
        """Latest trade price per ticker (IEX feed, free on paper)."""
        if not tickers:
            return {}
        out: dict[str, float] = {}
        # batch via the snapshots endpoint
        qs = ",".join(tickers)
        try:
            data = self._get(f"{DATA_BASE}/v2/stocks/snapshots?symbols={qs}")
            for tk, snap in (data.items() if isinstance(data, dict) else []):
                trade = (snap or {}).get("latestTrade") or {}
                px = trade.get("p")
                if px:
                    out[tk] = float(px)
        except (HTTPError, URLError) as exc:
            logger.warning("alpaca price fetch failed: %s", exc)
        return out

    def get_orders(self, status: str = "all", limit: int = 500) -> list[dict]:
        """Recent orders (raw Alpaca dicts) for honest status reconciliation."""
        try:
            return list(self._get(f"{self.base}/v2/orders?status={status}&limit={limit}"))
        except (HTTPError, URLError) as exc:
            logger.warning("alpaca get_orders failed: %s", exc)
            return []

    def assets_info(self, tickers: list[str]) -> dict[str, dict]:
        """Per-symbol tradability {symbol: {tradable, shortable, fractionable, status, class}}.
        Symbols Alpaca doesn't know are simply absent → treated as not tradable by the filter."""
        out: dict[str, dict] = {}
        for tk in tickers:
            try:
                a = self._get(f"{self.base}/v2/assets/{tk}")
                out[tk] = {"tradable": bool(a.get("tradable")), "shortable": bool(a.get("shortable")),
                           "fractionable": bool(a.get("fractionable")), "status": a.get("status"),
                           "class": a.get("class")}
            except (HTTPError, URLError) as exc:
                out[tk] = {"tradable": False, "status": f"lookup_failed:{getattr(exc,'code','?')}"}
        return out

    def tradable_filter(self, target_weights: dict[str, float]) -> tuple[dict[str, float], dict[str, str]]:
        """Drop symbols Alpaca can't execute. Returns (kept_weights, dropped{ticker: reason}).
        A SHORT target on a non-shortable name is also dropped (would 422)."""
        info = self.assets_info(sorted(target_weights))
        kept: dict[str, float] = {}
        dropped: dict[str, str] = {}
        for tk, w in target_weights.items():
            meta = info.get(tk, {"tradable": False, "status": "unknown"})
            if not meta.get("tradable"):
                dropped[tk] = f"not_tradable(status={meta.get('status')})"
            elif w < 0 and not meta.get("shortable"):
                dropped[tk] = "not_shortable"
            else:
                kept[tk] = w
        return kept, dropped

    def submit_order(self, order: Order) -> Fill:
        qty = abs(order.qty)
        payload = {"symbol": order.ticker, "qty": str(qty), "side": order.side,
                   "type": "market", "time_in_force": "day"}
        r = self._post(f"{self.base}/v2/orders", payload)
        # market order fills async; report submitted qty at last known price (filled_avg_price may
        # be null immediately). The caller reconciles fills via get_positions on the next pull.
        fpx = r.get("filled_avg_price")
        price = float(fpx) if fpx else float(self.get_prices([order.ticker]).get(order.ticker, 0.0))
        filled = r.get("filled_qty")
        fqty = float(filled) if filled else qty
        return Fill(ticker=order.ticker, qty=fqty if order.qty > 0 else -fqty,
                    price=price, order_id=str(r.get("id", "")))
