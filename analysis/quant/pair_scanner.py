"""
Industry-group pair scanner.

Fetches prices for every ticker in a predefined group, then runs
cointegration tests + OLS Z-score for all combinations.
Results are sorted by p-value (strongest relationship first).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from data.collectors.price_collector import PriceCollector


# ── Industry group definitions ────────────────────────────────────────────────

INDUSTRY_GROUPS: dict[str, dict] = {
    "반도체 (KOSPI)": {
        "tickers": ["005930.KS", "000660.KS", "042700.KS"],
        "names": {
            "005930.KS": "삼성전자",
            "000660.KS": "SK하이닉스",
            "042700.KS": "한미반도체",
        },
    },
    "2차전지 (KOSPI)": {
        "tickers": ["373220.KS", "006400.KS", "005490.KS"],
        "names": {
            "373220.KS": "LG에너지솔루션",
            "006400.KS": "삼성SDI",
            "005490.KS": "POSCO홀딩스",
        },
    },
    "인터넷·플랫폼 (KOSPI)": {
        "tickers": ["035720.KS", "035420.KS", "251270.KS"],
        "names": {
            "035720.KS": "카카오",
            "035420.KS": "NAVER",
            "251270.KS": "넷마블",
        },
    },
    "빅테크 (NASDAQ)": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "META"],
        "names": {
            "AAPL":  "Apple",
            "MSFT":  "Microsoft",
            "GOOGL": "Alphabet",
            "META":  "Meta",
        },
    },
    "반도체 (NASDAQ)": {
        "tickers": ["NVDA", "AMD", "INTC", "QCOM"],
        "names": {
            "NVDA": "NVIDIA",
            "AMD":  "AMD",
            "INTC": "Intel",
            "QCOM": "Qualcomm",
        },
    },
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PairScanResult:
    ticker_a: str
    ticker_b: str
    name_a: str
    name_b: str
    pvalue: float
    is_cointegrated: bool
    hedge_ratio: float
    zscore_latest: float
    signal_a: str
    signal_b: str


# ── Scanner ───────────────────────────────────────────────────────────────────

class PairScanner:
    """
    Scans all n×(n-1)/2 combinations in an industry group for cointegration.

    Usage (split fetch + scan for progress-bar support):
        scanner = PairScanner(...)
        prices  = scanner.fetch_prices(group_name, progress_cb)
        results = scanner.scan(group_name, prices, progress_cb)
        df      = scanner.to_dataframe(results)
    """

    MIN_COMMON_DAYS = 60

    def __init__(
        self,
        period: str = "1y",
        zscore_window: int = 30,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self._collector = PriceCollector()
        self.period = period
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z

    # ── Public ────────────────────────────────────────────────────────────

    def fetch_prices(
        self,
        group_name: str,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> dict[str, pd.Series]:
        """
        Returns {ticker: Close price series} for all tickers in the group.
        Silently skips any ticker that fails or has fewer than MIN_COMMON_DAYS rows.
        """
        tickers = INDUSTRY_GROUPS[group_name]["tickers"]
        prices: dict[str, pd.Series] = {}

        for i, ticker in enumerate(tickers):
            if progress_cb:
                progress_cb(i / len(tickers), f"{ticker} 가격 수집 중…")
            df = self._collector.fetch(ticker, period=self.period)
            if not df.empty and len(df) >= self.MIN_COMMON_DAYS:
                prices[ticker] = df["Close"].dropna().rename(ticker)

        return prices

    def scan(
        self,
        group_name: str,
        prices: dict[str, pd.Series],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[PairScanResult]:
        """
        Runs cointegration + OLS Z-score for every pair in the group.
        Returns list sorted by p-value ascending (strongest relationship first).
        """
        group = INDUSTRY_GROUPS[group_name]
        names = group["names"]
        available = [t for t in group["tickers"] if t in prices]
        pairs = list(itertools.combinations(available, 2))

        results: list[PairScanResult] = []
        for i, (ta, tb) in enumerate(pairs):
            if progress_cb:
                label = f"{names.get(ta, ta)} / {names.get(tb, tb)} 검정 중…"
                progress_cb(i / len(pairs), label)
            result = self._analyze_pair(ta, tb, names, prices)
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.pvalue)
        return results

    def to_dataframe(self, results: list[PairScanResult]) -> pd.DataFrame:
        """Converts scan results to a display-ready DataFrame (hidden ticker columns prefixed _)."""
        rows = []
        for r in results:
            rows.append({
                "종목 A": f"{r.name_a} ({r.ticker_a})",
                "종목 B": f"{r.name_b} ({r.ticker_b})",
                "p-value": round(r.pvalue, 4),
                "공적분": "✅" if r.is_cointegrated else "❌",
                "헤지비율 β": round(r.hedge_ratio, 4),
                "Z-score": round(r.zscore_latest, 3),
                "A 신호": r.signal_a,
                "B 신호": r.signal_b,
                "_ticker_a": r.ticker_a,
                "_ticker_b": r.ticker_b,
            })
        return pd.DataFrame(rows)

    # ── Internal ──────────────────────────────────────────────────────────

    def _analyze_pair(
        self,
        ta: str,
        tb: str,
        names: dict[str, str],
        prices: dict[str, pd.Series],
    ) -> Optional[PairScanResult]:
        try:
            combined = pd.concat([prices[ta], prices[tb]], axis=1).dropna()
            if len(combined) < self.MIN_COMMON_DAYS:
                return None

            pa, pb = combined[ta], combined[tb]

            # Cointegration test
            _, pvalue, _ = coint(pa.values, pb.values)

            # OLS hedge ratio + spread
            model = OLS(pa.values, add_constant(pb.values)).fit()
            hedge_ratio = float(model.params[1])
            spread = pa - hedge_ratio * pb

            # Rolling Z-score
            w = self.zscore_window
            roll_mean = spread.rolling(w, min_periods=w).mean()
            roll_std  = spread.rolling(w, min_periods=w).std()
            zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

            valid = zscore.dropna()
            z_now = float(valid.iloc[-1]) if not valid.empty else float("nan")
            signal_a, signal_b = self._classify(z_now)

            return PairScanResult(
                ticker_a=ta,
                ticker_b=tb,
                name_a=names.get(ta, ta),
                name_b=names.get(tb, tb),
                pvalue=float(pvalue),
                is_cointegrated=bool(pvalue < 0.05),
                hedge_ratio=round(hedge_ratio, 4),
                zscore_latest=round(z_now, 4) if not np.isnan(z_now) else float("nan"),
                signal_a=signal_a,
                signal_b=signal_b,
            )
        except Exception:
            return None

    def _classify(self, z: float) -> tuple[str, str]:
        if np.isnan(z):
            return "WAIT", "WAIT"
        if z > self.entry_z:
            return "SELL", "BUY"
        if z < -self.entry_z:
            return "BUY", "SELL"
        if abs(z) < self.exit_z:
            return "CLOSE", "CLOSE"
        return "WAIT", "WAIT"
