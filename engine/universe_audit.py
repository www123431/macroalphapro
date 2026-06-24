"""
Universe Survivorship Bias Audit
=================================
Verifies that every ETF in SECTOR_ETF was already trading on the backtest
start date.  If an ETF was created AFTER the start date, including it from
the beginning introduces survivorship bias: we implicitly selected a winner
that didn't exist yet.

Findings are written to BacktestResult.warnings so they surface in the UI
without crashing the backtest.
"""

import datetime
from dataclasses import dataclass

# ── Verified inception dates (exchange listing date, not fund creation) ────────
# Sources: ETF issuer prospectuses / SEC EDGAR N-1A filings
ETF_INCEPTION: dict[str, datetime.date] = {
    # ── Batch 0: original 18 ─────────────────────────────────────────────────
    "EWS":  datetime.date(1996,  3, 12),   # iShares MSCI Singapore
    "QQQ":  datetime.date(1999,  3, 10),   # Invesco QQQ
    "XLF":  datetime.date(1998, 12, 16),   # Financial Select Sector SPDR
    "XLE":  datetime.date(1998, 12, 16),   # Energy Select Sector SPDR
    "XLI":  datetime.date(1998, 12, 16),   # Industrial Select Sector SPDR
    "XLV":  datetime.date(1998, 12, 16),   # Health Care Select Sector SPDR
    "XLP":  datetime.date(1998, 12, 16),   # Consumer Staples SPDR
    "XLY":  datetime.date(1998, 12, 16),   # Consumer Discretionary SPDR
    "SMH":  datetime.date(2000,  5,  5),   # VanEck Semiconductor
    "TLT":  datetime.date(2002,  7, 22),   # iShares 20+ Year Treasury
    "VNQ":  datetime.date(2004,  9, 23),   # Vanguard Real Estate
    "GLD":  datetime.date(2004, 11, 18),   # SPDR Gold Shares
    "XBI":  datetime.date(2006,  2,  6),   # SPDR S&P Biotech
    "HYG":  datetime.date(2007,  4,  4),   # iShares HY Corp Bond
    "ICLN": datetime.date(2008,  6, 24),   # iShares Global Clean Energy
    "KWEB": datetime.date(2013,  7, 31),   # KraneShares CSI China Internet
    "ASHR": datetime.date(2013, 11,  6),   # Xtrackers CSI 300 China A-Shares
    "XLC":  datetime.date(2018,  6, 18),   # Communication Services SPDR
    # ── Batch A: factor / regional equity (+7) ───────────────────────────────
    "IWN":  datetime.date(2000,  7, 24),   # iShares Russell 2000 Value
    "IWO":  datetime.date(2000,  7, 24),   # iShares Russell 2000 Growth
    "MTUM": datetime.date(2013,  4, 16),   # iShares MSCI USA Momentum Factor
    "USMV": datetime.date(2011, 10, 18),   # iShares MSCI USA Min Vol Factor
    "QUAL": datetime.date(2013,  7, 16),   # iShares MSCI USA Quality Factor
    "EWJ":  datetime.date(1996,  3, 12),   # iShares MSCI Japan
    "INDA": datetime.date(2012,  2,  2),   # iShares MSCI India
    # ── Batch B: cross-asset (+7) ────────────────────────────────────────────
    # PRE-8 confirmed: AGG/IEF/TIP Adj Close includes coupon reinvestment (-0.17%/-0.36%/-0.38% vs expected 0)
    "AGG":  datetime.date(2003,  9, 26),   # iShares Core US Aggregate Bond
    "IEF":  datetime.date(2002,  7, 22),   # iShares 7-10 Year Treasury
    "TIP":  datetime.date(2003, 12,  4),   # iShares TIPS Bond
    "GDX":  datetime.date(2006,  5, 22),   # VanEck Gold Miners
    "DBA":  datetime.date(2007,  1,  5),   # Invesco DB Agriculture
    "VXX":  datetime.date(2009,  1, 30),   # iPath Series B S&P 500 VIX (polarity-flipped)
    "REM":  datetime.date(2007,  5,  1),   # iShares Mortgage Real Estate
    # ── Batch C: FX / 原油 / 信用梯度 / EM 宽基 (+4) ─────────────────────────
    # LQD: Adj Close 含票息再投资（与 AGG/IEF/TIP 同类偏差，PRE-8 follow-up 需验证）
    # USO: 2020-04 反向拆股 1:8，合约结构同期从近月改为跨月混合，momentum 存在结构断点
    "UUP":  datetime.date(2007,  2, 20),   # Invesco DB US Dollar Index Bullish
    "USO":  datetime.date(2006,  4, 10),   # United States Oil Fund LP
    "LQD":  datetime.date(2002,  7, 22),   # iShares iBoxx $ IG Corporate Bond
    "EEM":  datetime.date(2003,  4,  7),   # iShares MSCI Emerging Markets
}


@dataclass
class AuditResult:
    ok: bool
    warnings: list[str]
    late_entries: dict[str, datetime.date]   # ticker → inception date (only if > start)


def audit_universe(
    tickers: list[str],
    backtest_start: str | datetime.date,
) -> AuditResult:
    """
    Check each ticker's inception date against the backtest start date.

    Args:
        tickers:         List of ETF tickers to audit (from SECTOR_ETF.values()).
        backtest_start:  "YYYY-MM-DD" or datetime.date — first signal date.

    Returns:
        AuditResult with ok=False and populated warnings if any ETF post-dates start.
    """
    if isinstance(backtest_start, str):
        backtest_start = datetime.date.fromisoformat(backtest_start)

    warnings: list[str] = []
    late: dict[str, datetime.date] = {}

    for ticker in tickers:
        inception = ETF_INCEPTION.get(ticker)
        if inception is None:
            warnings.append(
                f"[universe_audit] {ticker}: 成立日期未知，无法验证幸存者偏差。"
                " 请手动查证 SEC EDGAR N-1A 文件。"
            )
            continue
        if inception > backtest_start:
            late[ticker] = inception
            gap_days = (inception - backtest_start).days
            warnings.append(
                f"[universe_audit] ⚠️  幸存者偏差风险：{ticker} 成立于 {inception}，"
                f"晚于回测起点 {backtest_start} {gap_days} 天。"
                f" 建议将回测起点推迟至 {inception} 之后，或将 {ticker} 从该窗口排除。"
            )

    ok = len(late) == 0 and all("未知" not in w for w in warnings)
    return AuditResult(ok=ok, warnings=warnings, late_entries=late)
