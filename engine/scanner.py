import yfinance as yf
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _get_universe() -> dict[str, str]:
    """Single source of truth: defer to get_active_sector_etf()."""
    try:
        from engine.history import get_active_sector_etf
        return get_active_sector_etf()
    except Exception:
        return {}


# AUDIT_TICKERS: {sector: [ticker]} — derived dynamically so it always
# matches the live universe. Callers that imported this directly will still
# work; the dict is rebuilt on each module import.
AUDIT_TICKERS: dict[str, list[str]] = {
    sec: [etf] for sec, etf in _get_universe().items()
}


class MarketScanner:
    def __init__(self):
        self.watchlist = _get_universe()

    def _fetch(self, ticker: str, period: str = "6mo") -> pd.DataFrame:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"无法获取 {ticker} 的数据")
        return df

    def _compute_metrics(self, name: str, ticker: str) -> dict:
        df = self._fetch(ticker)          # ~126 trading days of data
        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        returns = close.pct_change().dropna()

        ann_return = float(returns.mean() * 252)
        vol = float(returns.std() * np.sqrt(252))
        sharpe = ann_return / vol if vol > 0 else 0.0

        # Multi-period cross-sectional momentum (standard factor windows)
        def _mom(n_days: int) -> float:
            """Return over last n trading days; capped at available history."""
            n = min(n_days, len(close) - 1)
            p0, p1 = float(close.iloc[-n - 1]), float(close.iloc[-1])
            return (p1 - p0) / p0 if p0 > 0 else 0.0

        mom_1m = _mom(21)    # ~1 month
        mom_3m = _mom(63)    # ~3 months
        mom_6m = _mom(126)   # ~6 months (full download window)

        recent_vol = volume.iloc[-10:].mean()
        avg_vol = volume.mean()
        fund_flow = float(recent_vol / avg_vol) if avg_vol > 0 else 1.0

        return {
            "name":       name,
            "ticker":     ticker,
            "sharpe":     round(sharpe, 3),
            "volatility": round(vol, 4),
            "ann_return": round(ann_return, 4),
            "momentum":   round(mom_3m, 4),   # legacy key = 3m momentum
            "mom_1m":     round(mom_1m, 4),
            "mom_3m":     round(mom_3m, 4),
            "mom_6m":     round(mom_6m, 4),
            "fund_flow":  round(fund_flow, 3),
            "last_price": float(close.iloc[-1]),
        }

    def run_daily_scan(self) -> dict:
        """
        扫描全部板块。
        返回 {"best": {...}, "rankings": [...按夏普排序的完整列表...]}
        """
        results = []
        for name, ticker in self.watchlist.items():
            try:
                results.append(self._compute_metrics(name, ticker))
            except Exception as e:
                logger.warning("扫描 %s (%s) 失败: %s", name, ticker, e)

        if not results:
            raise RuntimeError("所有资产扫描均失败，请检查网络或数据源。")

        # market_fit：夏普比率在全部候选中的百分位（0-100）
        sharpes = np.array([r["sharpe"] for r in results])
        for r in results:
            rank = float(np.sum(sharpes <= r["sharpe"])) / len(sharpes)
            r["market_fit"] = round(rank * 100, 1)

        ranked = sorted(results, key=lambda x: x["sharpe"], reverse=True)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1

        return {"best": ranked[0], "rankings": ranked}
