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
        seed_ticker: Optional[str] = None,
    ) -> list[PairScanResult]:
        """Scans ticker pairs. When seed_ticker is given, generates only
        (seed, other) pairs with seed always as ticker_a.
        """
        available = [t for t in tickers if t in prices]
        if seed_ticker and seed_ticker in available:
            pairs: list[tuple[str, str]] = [
                (seed_ticker, t) for t in available if t != seed_ticker
            ]
        else:
            pairs = list(itertools.combinations(available, 2))
        return self._scan_pairs(available, names, prices, progress_cb, pairs=pairs)

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
        pairs: Optional[list[tuple[str, str]]] = None,
    ) -> list[PairScanResult]:
        if pairs is None:
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

def _github_krx_csv(subpath: str) -> pd.DataFrame:
    """Fetches KRX listing CSV directly from FDR's GitHub cache.

    Bypasses data.krx.co.kr (which returns 403 in cloud environments).
    Tries recent business days until a file is found (up to 14 calendar days back).
    subpath: 'krx' for marcap listing, 'desc' for descriptive (Industry) listing.
    """
    import io
    import requests
    from datetime import date, timedelta

    base = (
        'https://raw.githubusercontent.com/FinanceData/'
        'fdr_krx_data_cache/refs/heads/master/data/listing'
    )
    today = date.today()
    for delta in range(14):
        d = today - timedelta(days=delta)
        if d.weekday() >= 5:
            continue
        url = f'{base}/{subpath}/{d.strftime("%Y-%m-%d")}.csv'
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text), dtype={'Code': str})
        except Exception:
            continue
    raise ValueError(f"GitHub KRX cache ({subpath}) 데이터를 가져올 수 없습니다.")


@cache
def _load_krx_listing() -> pd.DataFrame:
    """Loads KRX full listing (Code, Name, Market, Marcap).
    Tries FDR first; falls back to GitHub direct fetch if data.krx.co.kr is blocked.
    """
    import FinanceDataReader as fdr
    try:
        df = fdr.StockListing('KRX')
    except Exception:
        df = _github_krx_csv('krx')
    df['Code']  = df['Code'].astype(str).str.zfill(6)
    df['Marcap'] = pd.to_numeric(df.get('Marcap', pd.Series(dtype=float)), errors='coerce').fillna(0)
    return df


@cache
def _load_sp500_listing() -> pd.DataFrame:
    """Loads S&P500 listing (Symbol, Name, Sector [GICS], Industry)."""
    import FinanceDataReader as fdr
    return fdr.StockListing('S&P500')


@cache
def _load_krx_desc() -> pd.DataFrame:
    """Loads KRX-DESC listing (Code, Name, Market, Sector, Industry).
    Tries FDR first; falls back to GitHub direct fetch if data.krx.co.kr is blocked.
    """
    import FinanceDataReader as fdr
    try:
        df = fdr.StockListing('KRX-DESC')
    except Exception:
        df = _github_krx_csv('desc')
    df['Code'] = df['Code'].astype(str).str.zfill(6)
    return df


# ── Korean peer discovery helpers ────────────────────────────────────────────

_NAVER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': 'https://finance.naver.com',
}


def _kr_peers_via_naver(ticker: str, top_n: int) -> "PeerGroup":
    """Primary KR peer source: Naver Finance upjong (KRX 업종) classification.

    Flow:
      1. coinfo.naver → extract upjong number
      2. sise_group_detail.naver → sector name + all peer codes
      3. FDR KRX listing → market-cap rank, Korean names, KOSPI/KOSDAQ suffix
    """
    import requests
    from bs4 import BeautifulSoup

    code = ticker[:-3]  # strip .KS / .KQ

    # ── Step 1: upjong number ────────────────────────────────────────────────
    r = requests.get(
        'https://finance.naver.com/item/coinfo.naver',
        params={'code': code},
        headers=_NAVER_HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')

    upjong_no: str | None = None
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'sise_group_detail' in href and 'upjong' in href and 'no=' in href:
            upjong_no = href.split('no=')[-1].split('&')[0]
            break

    if not upjong_no:
        raise ValueError(f"네이버 금융에서 '{ticker}' 업종 코드를 찾을 수 없습니다.")

    # ── Step 2: sector name + peer codes ────────────────────────────────────
    r2 = requests.get(
        'https://finance.naver.com/sise/sise_group_detail.naver',
        params={'type': 'upjong', 'no': upjong_no},
        headers=_NAVER_HEADERS,
        timeout=10,
    )
    r2.raise_for_status()
    soup2 = BeautifulSoup(r2.text, 'html.parser')

    sector_name = ''
    title_tag = soup2.find('title')
    if title_tag:
        sector_name = title_tag.get_text().split(':')[0].strip()

    peer_codes: set[str] = set()
    for a in soup2.find_all('a', href=lambda h: h and 'code=' in h and 'item/main' in h):
        c = a['href'].split('code=')[-1].split('&')[0]
        if c and len(c) == 6 and c.isdigit():
            peer_codes.add(c)

    if not peer_codes:
        raise ValueError(f"'{ticker}' 업종 종목 리스트를 가져올 수 없습니다.")

    # ── Step 3: rank by market cap, resolve suffix ───────────────────────────
    krx = _load_krx_listing()
    code_to_marcap = dict(zip(krx['Code'], krx['Marcap']))
    code_to_name   = dict(zip(krx['Code'], krx['Name']))
    code_to_market = dict(zip(krx['Code'], krx['Market']))

    ranked = sorted(peer_codes, key=lambda c: code_to_marcap.get(c, 0), reverse=True)
    if code in ranked:
        ranked = [code] + [c for c in ranked if c != code]
    ranked = ranked[:top_n]

    peers: list[str] = []
    names: dict[str, str] = {}
    for c in ranked:
        mkt = str(code_to_market.get(c, ''))
        suffix = '.KQ' if 'KOSDAQ' in mkt else '.KS'
        tkr = f'{c}{suffix}'
        peers.append(tkr)
        names[tkr] = code_to_name.get(c) or tkr

    if ticker not in peers:
        peers = [ticker] + peers[:top_n - 1]
        names.setdefault(ticker, code_to_name.get(code, ticker))

    return PeerGroup(
        seed_ticker=ticker,
        sector=sector_name,
        industry=sector_name,
        source=f'네이버 금융 업종 분류 (KRX 기준) + FDR 시가총액 순위',
        tickers=peers,
        names=names,
    )


def _kr_peers_via_pykrx(ticker: str, top_n: int) -> "PeerGroup":
    """Fallback KR peer source: pykrx get_market_sector_classifications."""
    from pykrx import stock
    from datetime import date, timedelta

    suffix = '.KS' if ticker.endswith('.KS') else '.KQ'
    code   = ticker[:-3]
    market = 'KOSPI' if suffix == '.KS' else 'KOSDAQ'

    # Try up to 5 recent business days to find a date with data
    df = None
    today = date.today()
    for delta in range(10):
        d = today - timedelta(days=delta)
        if d.weekday() >= 5:
            continue
        try:
            candidate = stock.get_market_sector_classifications(d.strftime('%Y%m%d'), market)
            if candidate is not None and not candidate.empty:
                df = candidate
                break
        except Exception:
            continue

    if df is None or df.empty:
        raise ValueError("pykrx에서 업종 분류 데이터를 가져올 수 없습니다.")

    # Identify sector column (pykrx returns '업종명' or similar)
    sector_col = next((c for c in df.columns if '업종' in c), None)
    if sector_col is None:
        raise ValueError("pykrx 데이터에서 업종 컬럼을 찾을 수 없습니다.")

    # Find seed's sector
    df.index = df.index.astype(str).str.zfill(6)
    if code not in df.index:
        raise ValueError(f"pykrx 데이터에서 '{ticker}' 종목을 찾을 수 없습니다.")

    seed_sector = str(df.loc[code, sector_col])
    peer_codes  = df[df[sector_col] == seed_sector].index.tolist()

    # Rank by market cap
    krx = _load_krx_listing()
    code_to_marcap = dict(zip(krx['Code'], krx['Marcap']))
    code_to_name   = dict(zip(krx['Code'], krx['Name']))

    ranked = sorted(peer_codes, key=lambda c: code_to_marcap.get(c, 0), reverse=True)
    if code in ranked:
        ranked = [code] + [c for c in ranked if c != code]
    ranked = ranked[:top_n]

    peers: list[str] = []
    names: dict[str, str] = {}
    for c in ranked:
        tkr = f'{c}{suffix}'
        peers.append(tkr)
        names[tkr] = code_to_name.get(c) or tkr

    if ticker not in peers:
        peers = [ticker] + peers[:top_n - 1]
        names.setdefault(ticker, code_to_name.get(code, ticker))

    return PeerGroup(
        seed_ticker=ticker,
        sector=seed_sector,
        industry=seed_sector,
        source='pykrx KRX 업종 분류 + FDR 시가총액 순위',
        tickers=peers,
        names=names,
    )


def _kr_peers_via_fdr(ticker: str, top_n: int) -> "PeerGroup":
    """Third-fallback KR peer source: FDR KRX-DESC Industry classification.

    KRX-DESC provides an Industry column (KIS 업종 분류) for all listed stocks.
    Combined with KRX Marcap for market-cap ranking.
    """
    code = ticker[:-3]

    desc = _load_krx_desc()
    seed_row = desc[desc['Code'] == code]
    if seed_row.empty:
        raise ValueError(f"KRX-DESC 데이터에서 '{ticker}' 종목을 찾을 수 없습니다.")

    industry = str(seed_row.iloc[0]['Industry'])
    if not industry or industry == 'nan':
        raise ValueError(f"KRX-DESC 데이터에서 '{ticker}' 업종 정보가 없습니다.")

    same_ind = desc[desc['Industry'] == industry][['Code', 'Name', 'Market']].copy()

    # Rank by Marcap from KRX listing
    krx = _load_krx_listing()
    code_to_marcap = dict(zip(krx['Code'], krx['Marcap']))
    same_ind['Marcap'] = same_ind['Code'].map(code_to_marcap).fillna(0)
    same_ind = same_ind.sort_values('Marcap', ascending=False)

    # Ensure seed is first
    ranked_df = pd.concat([
        same_ind[same_ind['Code'] == code],
        same_ind[same_ind['Code'] != code],
    ]).head(top_n)

    peers: list[str] = []
    names: dict[str, str] = {}
    for _, r in ranked_df.iterrows():
        c   = str(r['Code'])
        mkt = str(r['Market'])
        suffix = '.KQ' if 'KOSDAQ' in mkt else '.KS'
        tkr = f'{c}{suffix}'
        peers.append(tkr)
        names[tkr] = str(r['Name'])

    if ticker not in peers:
        peers = [ticker] + peers[:top_n - 1]
        seed_name = str(seed_row.iloc[0]['Name'])
        names.setdefault(ticker, seed_name)

    return PeerGroup(
        seed_ticker=ticker,
        sector=industry,
        industry=industry,
        source='FDR KRX-DESC 업종 분류 + FDR 시가총액 순위',
        tickers=peers,
        names=names,
    )


# ── Main class ────────────────────────────────────────────────────────────────

class PeerDiscovery:
    """
    Finds sector/industry peers for a single seed ticker.

    Korean (.KS/.KQ)
    ─────────────────
    Primary    : 네이버 금융 업종(upjong) 분류 → KRX 공식 업종 기준
    Fallback 1 : pykrx get_market_sector_classifications
    Fallback 2 : FDR KRX-DESC Industry 컬럼
    → 출처: 각 소스 표기

    US tickers
    ──────────
    FDR S&P500 리스팅의 Sector(GICS) 컬럼으로 필터링합니다.
    → 출처: "Yahoo Finance 섹터 분류 기준 (S&P500 구성종목)"
    """

    def __init__(self, top_n: int = 10, scan_depth: int = 80) -> None:
        self.top_n = top_n
        self.scan_depth = scan_depth  # kept for API compatibility; not used for KR peers

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
        errors: list[str] = []

        # Primary: Naver Finance upjong (KRX official sector classification)
        try:
            return _kr_peers_via_naver(ticker, self.top_n)
        except Exception as e:
            errors.append(f"네이버 금융: {e}")

        # Fallback 1: pykrx
        try:
            return _kr_peers_via_pykrx(ticker, self.top_n)
        except Exception as e:
            errors.append(f"pykrx: {e}")

        # Fallback 2: FDR KRX-DESC Industry column
        try:
            return _kr_peers_via_fdr(ticker, self.top_n)
        except Exception as e:
            errors.append(f"FDR KRX-DESC: {e}")

        raise ValueError(
            "업종 정보를 가져올 수 없습니다. 직접 분석 탭을 이용해주세요.\n"
            + " / ".join(errors)
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
