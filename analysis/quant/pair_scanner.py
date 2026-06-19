"""
Industry-group pair scanner + dynamic peer discovery.

Static scan:   INDUSTRY_GROUPS → PairScanner.scan()
Dynamic scan:  PeerDiscovery.find(ticker) → PairScanner.scan_tickers()

Korean tickers (.KS/.KQ):
  - FinanceDataReader KRX listing  → market-cap ranked universe
  - yfinance sector/industry       → classification (KRX listing has no sector column)
  ⚠ UI must note "Yahoo Finance 업종 분류 기준"

US tickers:
  - FinanceDataReader S&P500 listing (GICS Sector column) → universe + classification
  - yfinance sector                 → seed ticker lookup
  ⚠ UI must note "Yahoo Finance 분류 기준"
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from typing import Callable, Optional

try:
    from yfinance.exceptions import YFRateLimitError as _YFRateLimitError
except ImportError:
    _YFRateLimitError = Exception  # older yfinance versions

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint

from data.collectors.price_collector import PriceCollector


# ── Static industry group definitions ────────────────────────────────────────

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


# ── Scan result type ──────────────────────────────────────────────────────────

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


# ── Static group scanner ──────────────────────────────────────────────────────

class PairScanner:
    """
    Scans all n×(n-1)/2 combinations in an industry group for cointegration.

    Usage:
        scanner = PairScanner(...)
        prices  = scanner.fetch_prices(group_name)
        results = scanner.scan(group_name, prices)          # predefined group
        results = scanner.scan_tickers(tickers, names, prices)  # ad-hoc list
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
        tickers = INDUSTRY_GROUPS[group_name]["tickers"]
        prices: dict[str, pd.Series] = {}
        for i, ticker in enumerate(tickers):
            if progress_cb:
                progress_cb(i / len(tickers), f"{ticker} 가격 수집 중…")
            df = self._collector.fetch(ticker, period=self.period)
            if not df.empty and len(df) >= self.MIN_COMMON_DAYS:
                prices[ticker] = df["Close"].dropna().rename(ticker)
        return prices

    def fetch_prices_for(
        self,
        tickers: list[str],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> dict[str, pd.Series]:
        """Fetches prices for an arbitrary ticker list (for dynamic peers)."""
        prices: dict[str, pd.Series] = {}
        for i, ticker in enumerate(tickers):
            if progress_cb:
                progress_cb(i / len(tickers), f"{ticker} 가격 수집 중…")
            try:
                df = self._collector.fetch(ticker, period=self.period)
                if not df.empty and len(df) >= self.MIN_COMMON_DAYS:
                    prices[ticker] = df["Close"].dropna().rename(ticker)
            except Exception:
                pass
        return prices

    def scan(
        self,
        group_name: str,
        prices: dict[str, pd.Series],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[PairScanResult]:
        group = INDUSTRY_GROUPS[group_name]
        names = group["names"]
        available = [t for t in group["tickers"] if t in prices]
        return self._scan_pairs(available, names, prices, progress_cb)

    def scan_tickers(
        self,
        tickers: list[str],
        names: dict[str, str],
        prices: dict[str, pd.Series],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> list[PairScanResult]:
        """Scans all combinations of given tickers (no INDUSTRY_GROUPS dependency)."""
        available = [t for t in tickers if t in prices]
        return self._scan_pairs(available, names, prices, progress_cb)

    def to_dataframe(self, results: list[PairScanResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "종목 A":   f"{r.name_a} ({r.ticker_a})",
                "종목 B":   f"{r.name_b} ({r.ticker_b})",
                "p-value":  round(r.pvalue, 4),
                "공적분":   "✅" if r.is_cointegrated else "❌",
                "헤지비율 β": round(r.hedge_ratio, 4),
                "Z-score":  round(r.zscore_latest, 3),
                "A 신호":   r.signal_a,
                "B 신호":   r.signal_b,
                "_ticker_a": r.ticker_a,
                "_ticker_b": r.ticker_b,
            })
        return pd.DataFrame(rows)

    # ── Internal ──────────────────────────────────────────────────────────

    def _scan_pairs(
        self,
        available: list[str],
        names: dict[str, str],
        prices: dict[str, pd.Series],
        progress_cb: Optional[Callable[[float, str], None]],
    ) -> list[PairScanResult]:
        pairs = list(itertools.combinations(available, 2))
        results: list[PairScanResult] = []
        for i, (ta, tb) in enumerate(pairs):
            if progress_cb:
                label = f"{names.get(ta, ta)} / {names.get(tb, tb)} 검정 중…"
                progress_cb(i / max(len(pairs), 1), label)
            result = self._analyze_pair(ta, tb, names, prices)
            if result is not None:
                results.append(result)
        results.sort(key=lambda r: r.pvalue)
        return results

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

            _, pvalue, _ = coint(pa.values, pb.values)

            model = OLS(pa.values, add_constant(pb.values)).fit()
            hedge_ratio = float(model.params[1])
            spread = pa - hedge_ratio * pb

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


# ══════════════════════════════════════════════════════════════════════════════
#  Dynamic peer discovery
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PeerGroup:
    seed_ticker: str
    sector: str
    industry: str
    source: str          # display string shown in UI
    tickers: list[str]
    names: dict[str, str]  # ticker → display name


# ── yfinance-to-GICS sector mapping (S&P500 listing uses GICS names) ─────────

_YF_TO_GICS: dict[str, str] = {
    "Technology":             "Information Technology",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Financial Services":     "Financials",
    "Healthcare":             "Health Care",
    "Communication Services": "Communication Services",
    "Consumer Defensive":     "Consumer Staples",
    "Industrials":            "Industrials",
    "Basic Materials":        "Materials",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
    "Energy":                 "Energy",
}


# ── Process-level cached listing loaders (re-fetched only on server restart) ──

@cache
def _load_krx_listing() -> pd.DataFrame:
    """Loads KRX full listing (Code, Name, Market, Marcap). No sector column."""
    import FinanceDataReader as fdr
    df = fdr.StockListing('KRX')
    df['Code']  = df['Code'].astype(str).str.zfill(6)
    df['Marcap'] = pd.to_numeric(df.get('Marcap', pd.Series(dtype=float)), errors='coerce').fillna(0)
    return df


@cache
def _load_sp500_listing() -> pd.DataFrame:
    """Loads S&P500 listing (Symbol, Name, Sector [GICS], Industry)."""
    import FinanceDataReader as fdr
    return fdr.StockListing('S&P500')


# ── Parallel yfinance info helper ─────────────────────────────────────────────

def _fetch_yf_info(ticker: str) -> tuple[str, str, str, str]:
    """Returns (ticker, sector, industry, shortName) via yfinance.
    Returns empty strings on rate-limit or any other error so callers can skip gracefully.
    """
    import yfinance as yf
    time.sleep(0.5)  # throttle each thread to stay under rate limits
    try:
        info = yf.Ticker(ticker).info
        return (
            ticker,
            info.get('sector',    ''),
            info.get('industry',  ''),
            info.get('shortName', ticker),
        )
    except _YFRateLimitError:
        return ticker, '', '', ticker
    except Exception:
        return ticker, '', '', ticker


def _parallel_yf_info(
    tickers: list[str],
    max_workers: int = 3,  # reduced from 10 — avoids rate-limit bursts
) -> dict[str, dict[str, str]]:
    """Fetches sector / industry / shortName for many tickers concurrently."""
    result: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ticker, sector, industry, name in ex.map(_fetch_yf_info, tickers):
            result[ticker] = {"sector": sector, "industry": industry, "name": name}
    return result


# ── Main class ────────────────────────────────────────────────────────────────

class PeerDiscovery:
    """
    Finds sector/industry peers for a single seed ticker.

    Korean (.KS/.KQ)
    ─────────────────
    FinanceDataReader KRX 리스팅은 업종(Sector) 컬럼을 제공하지 않습니다.
    따라서 업종 분류는 yfinance industry 필드를 사용하고,
    FDR KRX 시가총액(Marcap) 데이터로 상위 N개를 제한합니다.
    → 출처: "Yahoo Finance 업종 분류 기준 (FDR KRX 시가총액 순위 활용)"

    US tickers
    ──────────
    FDR S&P500 리스팅의 Sector(GICS) 컬럼으로 필터링합니다.
    → 출처: "Yahoo Finance 섹터 분류 기준 (S&P500 구성종목)"
    """

    def __init__(self, top_n: int = 10, scan_depth: int = 80) -> None:
        self.top_n = top_n
        # scan_depth: how many top-Marcap KR stocks to check yfinance for
        self.scan_depth = scan_depth

    @staticmethod
    def is_korean(ticker: str) -> bool:
        return ticker.upper().endswith(('.KS', '.KQ'))

    def find(self, ticker: str) -> PeerGroup:
        ticker = ticker.strip().upper()
        if self.is_korean(ticker):
            return self._find_kr_peers(ticker)
        return self._find_us_peers(ticker)

    # ── Korean peers ──────────────────────────────────────────────────────

    def _find_kr_peers(self, ticker: str) -> PeerGroup:
        import yfinance as yf

        suffix = '.KS' if ticker.endswith('.KS') else '.KQ'

        # Seed: get sector/industry — skip gracefully on rate-limit or network error
        seed_sector = ''
        seed_industry = ''
        try:
            info = yf.Ticker(ticker).info
            seed_sector   = info.get('sector',    '')
            seed_industry = info.get('industry',  '')
            time.sleep(0.5)
        except (_YFRateLimitError, Exception):
            pass

        if not seed_industry:
            raise ValueError(
                f"Yahoo Finance에서 '{ticker}' 업종 정보를 찾을 수 없습니다. "
                "잠시 후 다시 시도하거나 유효한 KOSPI/KOSDAQ 티커인지 확인해주세요."
            )

        # FDR KRX → top market-cap candidates
        krx = _load_krx_listing()
        market_prefix = 'KOSPI' if suffix == '.KS' else 'KOSDAQ'
        universe = (
            krx[krx['Market'].str.startswith(market_prefix, na=False)]
            .sort_values('Marcap', ascending=False)
            .head(self.scan_depth)
        )
        cand_tickers = [c + suffix for c in universe['Code']]

        # Parallel-fetch industry for candidates (~2-5s for 80 stocks)
        info_map = _parallel_yf_info(cand_tickers)

        # Filter: same industry → fallback to same sector if too few
        same_industry = [
            t for t in cand_tickers
            if info_map.get(t, {}).get('industry') == seed_industry
        ]
        if len(same_industry) < 2:
            same_industry = [
                t for t in cand_tickers
                if info_map.get(t, {}).get('sector') == seed_sector
            ]

        # Ensure seed is first
        if ticker not in same_industry:
            same_industry = [ticker] + same_industry
        peers = same_industry[: self.top_n]

        # Names: prefer Korean name from KRX listing
        krx_name = dict(zip(krx['Code'], krx['Name']))
        names: dict[str, str] = {}
        for t in peers:
            c = t[:-3]
            names[t] = krx_name.get(c) or info_map.get(t, {}).get('name', t) or t

        return PeerGroup(
            seed_ticker=ticker,
            sector=seed_sector,
            industry=seed_industry,
            source='Yahoo Finance 업종 분류 기준 (FDR KRX 시가총액 순위 활용)',
            tickers=peers,
            names=names,
        )

    # ── US peers ──────────────────────────────────────────────────────────

    def _find_us_peers(self, ticker: str) -> PeerGroup:
        import yfinance as yf

        seed_sector = ''
        seed_industry = ''
        seed_name = ticker
        try:
            info = yf.Ticker(ticker).info
            seed_sector   = info.get('sector',    '')
            seed_industry = info.get('industry',  '')
            seed_name     = info.get('shortName', ticker)
            time.sleep(0.5)
        except (_YFRateLimitError, Exception):
            pass

        if not seed_sector:
            raise ValueError(
                f"Yahoo Finance에서 '{ticker}' 섹터 정보를 찾을 수 없습니다. "
                "잠시 후 다시 시도하거나 유효한 NASDAQ/NYSE 티커인지 확인해주세요."
            )

        # S&P500 uses GICS sector names; map yfinance → GICS
        gics = _YF_TO_GICS.get(seed_sector, seed_sector)

        sp500 = _load_sp500_listing()

        def _match(s: str) -> bool:
            sl = str(s).lower()
            return gics.lower() in sl or seed_sector.lower() in sl

        peers_df = sp500[sp500['Sector'].apply(_match)].head(self.top_n)

        if peers_df.empty:
            raise ValueError(
                f"S&P500에서 '{seed_sector}' 섹터 종목을 찾을 수 없습니다."
            )

        tickers = peers_df['Symbol'].astype(str).tolist()
        names   = dict(zip(peers_df['Symbol'].astype(str), peers_df['Name'].astype(str)))

        if ticker not in tickers:
            tickers = [ticker] + tickers[: self.top_n - 1]
            names[ticker] = seed_name

        return PeerGroup(
            seed_ticker=ticker,
            sector=seed_sector,
            industry=seed_industry,
            source='Yahoo Finance 섹터 분류 기준 (S&P500 구성종목)',
            tickers=tickers,
            names=names,
        )
