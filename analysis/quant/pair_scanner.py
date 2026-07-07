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

from analysis.quant.spread_diagnostics import copula_metrics, half_life, hurst_exponent
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
    "반도체 파운드리 (글로벌)": {
        "tickers": ["TSM", "UMC", "GFS", "0981.HK"],
        "names": {
            "TSM":     "TSMC",
            "UMC":     "UMC",
            "GFS":     "GlobalFoundries",
            "0981.HK": "SMIC",
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
    correlation: float = float("nan")   # daily-return Pearson correlation
    hurst: float = float("nan")         # spread Hurst exponent (<0.5 = 평균회귀)
    half_life_days: float = float("inf")  # spread OU half-life
    # Copula enrichment (top candidates only — see enrich_with_copula)
    kendall_tau: float = float("nan")
    tail_dep_lower: float = float("nan")
    tail_dep_upper: float = float("nan")
    copula_family: str = ""


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
        alpha: float = 0.05,
        stop_loss_mult: float = 1.75,
        min_corr: float = 0.0,
        max_hurst: float = 1.0,
        max_half_life: float = 0.0,
        distance_top_n: int = 0,
    ) -> None:
        self._collector = PriceCollector()
        self.period = period
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.alpha = alpha
        self.stop_loss_mult = stop_loss_mult
        # Minimum daily-return correlation between the two legs.
        # Pairs below this are skipped before the (more expensive)
        # cointegration test. 0.0 disables the filter.
        self.min_corr = min_corr
        # Spread quality gates (모두 스프레드 계산 후 적용):
        #   max_hurst      — 스프레드 Hurst가 이 값 초과면 제외 (≥1.0 = 해제).
        #                    0.5 미만 = 평균회귀 성향.
        #   max_half_life  — OU 반감기가 이 값(일) 초과 또는 ∞면 제외 (≤0 = 해제).
        self.max_hurst = max_hurst
        self.max_half_life = max_half_life
        # Distance method 사전 축소: 정규화 가격(시작=100) 유클리드 거리가
        # 가장 가까운 상위 N쌍만 공적분 검정 (0 = 해제).
        self.distance_top_n = distance_top_n

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
        prices, _ = self.fetch_market_data_for(tickers, progress_cb)
        return prices

    def fetch_market_data_for(
        self,
        tickers: list[str],
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> tuple[dict[str, pd.Series], dict[str, float]]:
        """Fetches Close prices + average daily dollar volume (USD) per ticker.

        Dollar volume = mean(Close × Volume) over the last 60 trading days.
        KRW-denominated tickers (.KS/.KQ) are converted to USD with the
        latest KRW=X rate so a single liquidity threshold works across
        markets.
        """
        prices: dict[str, pd.Series] = {}
        dollar_volumes: dict[str, float] = {}
        usdkrw: Optional[float] = None

        for i, ticker in enumerate(tickers):
            if progress_cb:
                progress_cb(i / len(tickers), f"{ticker} 가격 수집 중…")
            try:
                df = self._collector.fetch(ticker, period=self.period)
                if df.empty or len(df) < self.MIN_COMMON_DAYS:
                    continue
                prices[ticker] = df["Close"].dropna().rename(ticker)

                dv = float((df["Close"] * df["Volume"]).tail(60).mean())
                if ticker.upper().endswith((".KS", ".KQ")):
                    if usdkrw is None:
                        usdkrw = self._fetch_usdkrw()
                    dv /= usdkrw
                dollar_volumes[ticker] = dv
            except Exception:
                pass
        return prices, dollar_volumes

    def _fetch_usdkrw(self) -> float:
        """Latest USD/KRW rate; conservative fallback if the fetch fails."""
        try:
            fx = self._collector.fetch("KRW=X", period="5d")
            rate = float(fx["Close"].dropna().iloc[-1])
            if rate > 0:
                return rate
        except Exception:
            pass
        return 1350.0

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
                "상관계수": round(r.correlation, 3) if not np.isnan(r.correlation) else None,
                "Hurst":    round(r.hurst, 3) if not np.isnan(r.hurst) else None,
                "반감기(일)": round(r.half_life_days, 1) if np.isfinite(r.half_life_days) else None,
                "헤지비율 β": round(r.hedge_ratio, 4),
                "Z-score":  round(r.zscore_latest, 3),
                "A 신호":   r.signal_a,
                "B 신호":   r.signal_b,
                "_ticker_a": r.ticker_a,
                "_ticker_b": r.ticker_b,
            })
        return pd.DataFrame(rows)

    def enrich_with_copula(
        self,
        results: list[PairScanResult],
        prices: dict[str, pd.Series],
        top_k: int = 10,
    ) -> list[PairScanResult]:
        """상위 top_k 후보(이미 p-value 오름차순)에만 Copula 의존성 지표를
        채워 넣는다 (2차 필터용 — Kendall τ 역변환 닫힌형이라 저비용이지만
        전 쌍에 돌릴 이유가 없어 상위 후보로 제한).

        공적분·상관계수는 선형 관계만 보므로, 하방 꼬리 의존(Clayton)이
        강한 쌍은 동반 급락 위험 공유 → 페어로 더 신뢰할 수 있다.
        """
        for r in results[:top_k]:
            try:
                cm = copula_metrics(prices[r.ticker_a], prices[r.ticker_b])
                if cm is None:
                    continue
                r.kendall_tau = round(cm.kendall_tau, 3)
                r.tail_dep_lower = round(cm.tail_dep_lower, 3)
                r.tail_dep_upper = round(cm.tail_dep_upper, 3)
                r.copula_family = (
                    f"{cm.family} λ={cm.family_tail_dep:.2f}"
                    if cm.family != "독립적" else cm.family
                )
            except Exception:
                continue
        return results

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
        if self.distance_top_n > 0 and len(pairs) > self.distance_top_n:
            pairs = self._closest_pairs_by_distance(pairs, prices)
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

    def _closest_pairs_by_distance(
        self,
        pairs: list[tuple[str, str]],
        prices: dict[str, pd.Series],
    ) -> list[tuple[str, str]]:
        """Distance method (Gatev et al.): 정규화 가격 간 평균 제곱 거리가
        가장 작은 상위 distance_top_n쌍만 남긴다. 공적분 검정보다 훨씬 싸므로
        1차 후보 축소용."""
        scored: list[tuple[float, tuple[str, str]]] = []
        for ta, tb in pairs:
            try:
                combined = pd.concat([prices[ta], prices[tb]], axis=1).dropna()
                if len(combined) < self.MIN_COMMON_DAYS:
                    continue
                na = combined[ta] / float(combined[ta].iloc[0])
                nb = combined[tb] / float(combined[tb].iloc[0])
                # 평균 제곱 거리 (길이 차이에 영향받지 않도록 mean 사용)
                scored.append((float(((na - nb) ** 2).mean()), (ta, tb)))
            except Exception:
                continue
        scored.sort(key=lambda s: s[0])
        return [p for _, p in scored[: self.distance_top_n]]

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

            # Daily-return correlation gate: cheap pre-filter that removes
            # pairs that merely trended together (price-level correlation
            # alone is inflated by common trends).
            ret_corr = float(pa.pct_change().corr(pb.pct_change()))
            if self.min_corr > 0 and (np.isnan(ret_corr) or ret_corr < self.min_corr):
                return None

            _, pvalue, _ = coint(pa.values, pb.values)

            model = OLS(pa.values, add_constant(pb.values)).fit()
            hedge_ratio = float(model.params[1])
            spread = pa - hedge_ratio * pb

            # 스프레드 품질 게이트: 평균회귀 성향(Hurst)과 회귀 속도(반감기)
            h = hurst_exponent(spread)
            if self.max_hurst < 1.0 and (np.isnan(h) or h > self.max_hurst):
                return None
            hl = half_life(spread)
            if self.max_half_life > 0 and not (hl <= self.max_half_life):
                return None  # ∞(비회귀) 포함 제외

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
                is_cointegrated=bool(pvalue < self.alpha),
                hedge_ratio=round(hedge_ratio, 4),
                correlation=round(ret_corr, 4) if not np.isnan(ret_corr) else float("nan"),
                hurst=round(h, 3) if not np.isnan(h) else float("nan"),
                half_life_days=round(hl, 1) if np.isfinite(hl) else float("inf"),
                zscore_latest=round(z_now, 4) if not np.isnan(z_now) else float("nan"),
                signal_a=signal_a,
                signal_b=signal_b,
            )
        except Exception:
            return None

    def _classify(self, z: float) -> tuple[str, str]:
        if np.isnan(z):
            return "WAIT", "WAIT"
        if abs(z) >= self.entry_z * self.stop_loss_mult:
            return "STOP_LOSS", "STOP_LOSS"
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


# ── Non-KR/non-US ticker fallback (.HK, .SS, .SZ, .T, .TW, ...) ──────────────
#
# PeerDiscovery / KMeansPeerDiscovery only know how to classify KR (.KS/.KQ)
# and bare-US tickers (via S&P500 GICS sectors). Any other exchange suffix
# (e.g. SMIC = 0981.HK) has no automatic sector source, so it falls back to
# the curated INDUSTRY_GROUPS map (the same source 03_comprehensive.py uses).

def _is_other_market(ticker: str) -> bool:
    """True for tickers with a non-KR exchange suffix (e.g. .HK, .SS, .SZ, .T)."""
    if "." not in ticker:
        return False
    suffix = ticker.rsplit(".", 1)[-1]
    return suffix not in ("KS", "KQ")


def _static_group_peers(ticker: str, top_n: int, source_suffix: str = "") -> PeerGroup:
    """Looks up INDUSTRY_GROUPS for a static peer set.

    Used as a fallback for exchanges (HK/SS/SZ/T/...) that yfinance's
    S&P500-based US peer discovery and KRX-based KR peer discovery can't cover.
    """
    for gname, gdata in INDUSTRY_GROUPS.items():
        tickers = gdata.get("tickers", [])
        if ticker in tickers:
            names = gdata.get("names", {})
            others = [t for t in tickers if t != ticker]
            selected = [ticker] + others[: max(top_n - 1, 0)]
            return PeerGroup(
                seed_ticker=ticker,
                sector=gname,
                industry=gname,
                source=(
                    "정적 업종 그룹 (INDUSTRY_GROUPS) — Yahoo/S&P500·KRX 분류 미지원 해외 종목"
                    + source_suffix
                ),
                tickers=selected,
                names={t: names.get(t, t) for t in selected},
            )
    raise ValueError(
        f"'{ticker}'는 자동 업종 분류를 지원하지 않는 해외 거래소 종목입니다. "
        "현재 INDUSTRY_GROUPS에 등록된 종목군(예: 반도체 파운드리 - TSM/UMC/GFS/SMIC)만 지원합니다."
    )


def _augment_with_industry_groups(pg: "PeerGroup") -> "PeerGroup":
    """Supplements a KR/US classification-based PeerGroup with any
    INDUSTRY_GROUPS peers registered for the same seed ticker.

    PeerDiscovery's KR/US branches each search within a single country's
    listing (Naver/KRX-DESC for KR, S&P500 GICS for US), so they cannot
    surface a cross-border peer like SMIC (0981.HK) for a KR/US semiconductor
    seed. INDUSTRY_GROUPS is the only place in this module with curated
    cross-border groups (e.g. "반도체 파운드리 (글로벌)"), so when the seed
    happens to be registered there, its groupmates are appended as extra
    peers rather than being left out entirely.
    """
    for gname, gdata in INDUSTRY_GROUPS.items():
        group_tickers = gdata.get("tickers", [])
        if pg.seed_ticker not in group_tickers:
            continue
        group_names = gdata.get("names", {})
        extras = [t for t in group_tickers if t not in pg.tickers]
        if not extras:
            return pg
        pg.tickers = pg.tickers + extras
        for t in extras:
            pg.names[t] = group_names.get(t, t)
        pg.source = f"{pg.source} + INDUSTRY_GROUPS 보조 종목 ({gname})"
        return pg
    return pg


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


@cache
def _yf_sector_for(ticker: str) -> str:
    """Best-effort yfinance sector lookup, mapped to GICS-style naming via
    _YF_TO_GICS. Returns '' if yfinance has no sector for this ticker.

    Used by KMeansPeerDiscovery to pre-filter its global candidate pool to
    the seed's broad sector before clustering — otherwise price-only
    clustering can group tickers whose charts happen to move together despite
    being in unrelated industries (e.g. a pharma stock next to a foundry).
    Cached forever (process lifetime): this fires for every KOSPI/KOSDAQ/
    INDUSTRY_GROUPS candidate not already covered by the S&P500 GICS column,
    so without caching every new seed search would re-pay the full ~80-90
    call cost instead of reusing it.
    """
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        raw = info.get('sector') or ''
        time.sleep(0.3)
        return _YF_TO_GICS.get(raw, raw)
    except (_YFRateLimitError, Exception):
        return ''


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

    기타 해외 거래소 (.HK/.SS/.SZ/.T 등, 예: SMIC = 0981.HK)
    ─────────────────────────────────────────────────────────
    KR/US 분류 소스가 없으므로 INDUSTRY_GROUPS 정적 종목군으로 폴백합니다.
    → 출처: "정적 업종 그룹 (INDUSTRY_GROUPS)"

    국가 간 보조 통합
    ──────────────────
    KR/US 분류는 각각 자국 상장 종목 안에서만 검색되므로 (네이버/KRX-DESC는
    KR 상장만, S&P500 GICS는 US 상장만 커버) SMIC(0981.HK) 같은 해외 파운드리
    피어는 원천적으로 잡히지 않습니다. KMeansPeerDiscovery처럼 가격 기반
    완전 통합은 아니지만, 시드 종목이 INDUSTRY_GROUPS에도 등록되어 있으면
    그 그룹의 해외 종목을 결과에 보조로 추가합니다.
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
            return _augment_with_industry_groups(self._find_kr_peers(ticker))
        if _is_other_market(ticker):
            return _static_group_peers(ticker, self.top_n)
        return _augment_with_industry_groups(self._find_us_peers(ticker))

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

    @staticmethod
    def _rank_by_dollar_volume(symbols: list[str]) -> list[str]:
        """Ranks symbols by 1-month average dollar volume, descending.
        Returns [] if the batch download fails (caller falls back)."""
        import yfinance as yf

        try:
            raw = yf.download(symbols, period="1mo", auto_adjust=True, progress=False)
            if raw.empty:
                return []
            if isinstance(raw.columns, pd.MultiIndex):
                close, vol = raw['Close'], raw['Volume']
            else:  # single symbol
                close = raw[['Close']].set_axis(symbols[:1], axis=1)
                vol   = raw[['Volume']].set_axis(symbols[:1], axis=1)
            dollar = (close * vol).mean().dropna().sort_values(ascending=False)
            return [str(s) for s in dollar.index if s in symbols]
        except Exception:
            return []

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

        peers_df = sp500[sp500['Sector'].apply(_match)]

        if peers_df.empty:
            raise ValueError(
                f"S&P500에서 '{seed_sector}' 섹터 종목을 찾을 수 없습니다."
            )

        all_symbols = peers_df['Symbol'].astype(str).tolist()
        names       = dict(zip(peers_df['Symbol'].astype(str), peers_df['Name'].astype(str)))

        # The FDR S&P500 listing is roughly alphabetical, so head(top_n) would
        # pick "first N by name". Rank the sector's constituents by 1-month
        # average dollar volume (one batched download) and keep the most
        # liquid top_n instead. Falls back to listing order on failure.
        tickers = self._rank_by_dollar_volume(all_symbols)[: self.top_n]
        if not tickers:
            tickers = all_symbols[: self.top_n]

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


# ══════════════════════════════════════════════════════════════════════════════
#  K-means peer discovery
# ══════════════════════════════════════════════════════════════════════════════

class KMeansPeerDiscovery:
    """
    Finds peers via K-means clustering on normalized price movements,
    across a single global candidate pool — no country/exchange split.

    Sector pre-filter (safety gate)
    ─────────────────────────────────
    Pure price-movement clustering can group tickers that simply moved
    together over the window despite being in unrelated industries (e.g. a
    pharma stock and an insurer next to a semiconductor foundry). Before
    clustering, the candidate pool is filtered down to yfinance's broad
    sector (mapped to GICS-style naming) for the seed ticker — clustering
    only ever runs within that same-sector subset. If the seed has no
    yfinance sector at all, K-means is not attempted; the caller falls back
    to the curated INDUSTRY_GROUPS static group instead.

    Global pool (universe, pre-filter)
    ────────────────────────────────────
    KOSPI 상위 universe_size개 + KOSDAQ 상위 universe_size개
    + S&P500 동일 섹터 전체 (GICS 컬럼 기반, API 호출 불필요)
    + INDUSTRY_GROUPS 등록 해외 종목
      (홍콩/중국/대만 등, 예: SMIC = 0981.HK)
    을 하나의 후보 풀로 합친 뒤, 시드 종목의 yfinance 업종(대분류, GICS
    스타일)과 같은 종목만 남깁니다. 반도체처럼 국가 구분 없는 글로벌
    공급망 섹터라도, 업종이 다른 종목까지 섞이면 안 되므로 국가 구분은
    없애되 업종 구분은 유지합니다.

    통화 정규화
    ────────────
    원화(.KS/.KQ) 종목은 일별 KRW=X(USD/KRW) 환율로 USD 환산한 뒤
    시작가=100 정규화를 적용합니다. 국가 간 가격 단위 차이가 아니라
    순수한 "움직임 패턴"만으로 비교하기 위함입니다.

    Optimal K: elbow method (kneedle 방식 — 대각선까지 최대 거리)

    ⚠ 주가 움직임 기반 결과이므로, 실제로 신뢰할 종목 쌍인지는 반드시
    공적분 검정(Engle-Granger) 결과와 함께 확인해야 합니다.
    """

    MIN_COMMON_DAYS = 100
    MIN_SECTOR_CANDIDATES = 3  # 시드 제외, 같은 업종 후보가 이보다 적으면 폴백

    def __init__(
        self,
        top_n: int = 10,
        universe_size: int = 40,
        period: str = "2y",
        max_k: int = 12,
    ) -> None:
        self.top_n = top_n
        self.universe_size = universe_size
        self.period = period
        self.max_k = max_k

    # ── Public ────────────────────────────────────────────────────────────────

    def find(self, ticker: str) -> PeerGroup:
        ticker = ticker.strip().upper()
        seed_sector, seed_industry, _seed_name = self._seed_info(ticker)

        if not seed_sector:
            # No yfinance sector for the seed → nothing to safely filter by,
            # so don't expose an unfiltered (potentially unrelated) cluster.
            try:
                return _static_group_peers(
                    ticker, self.top_n,
                    source_suffix=" (K-means 미지원: 업종 정보 부족 → 정적 폴백)",
                )
            except Exception:
                raise ValueError(
                    f"'{ticker}'는 yfinance 업종(sector) 정보가 없어 K-means 클러스터링을 "
                    "지원하지 않습니다 (업종 필터 없이는 결과를 신뢰할 수 없어 노출하지 "
                    "않습니다). INDUSTRY_GROUPS 정적 종목군에도 등록되어 있지 않습니다."
                )

        try:
            return self._find_global(ticker, seed_sector, seed_industry)
        except Exception as primary_err:
            # Last-resort safety net: seed has no fetchable yfinance price
            # history, or too few same-sector candidates for a meaningful
            # cluster — fall back to the curated static group if the seed
            # happens to be registered there.
            try:
                return _static_group_peers(
                    ticker, self.top_n,
                    source_suffix=" (글로벌 K-means 풀 조회 실패 → 정적 폴백)",
                )
            except Exception:
                raise primary_err

    # ── Universe builder ─────────────────────────────────────────────────────

    def _build_universe_global(
        self, ticker: str, seed_gics_sector: str,
    ) -> tuple[list[str], dict[str, str], int]:
        """Combines KOSPI/KOSDAQ + S&P500 + INDUSTRY_GROUPS overseas tickers
        into one candidate pool, regardless of the seed ticker's own market,
        then keeps only candidates in the seed's broad sector.

        S&P500 candidates already carry a GICS 'Sector' column (free — no
        extra call). KOSPI/KOSDAQ/INDUSTRY_GROUPS candidates need a yfinance
        sector lookup (_yf_sector_for, cached per ticker for the process
        lifetime) since neither KRX listing has a GICS-comparable column.

        Returns (filtered_universe_incl_seed, names, raw_pool_size).
        """
        pool: list[str] = []
        names: dict[str, str] = {}
        sp500_gics: dict[str, str] = {}

        krx = _load_krx_listing()
        for market_key, suffix in (('KOSPI', '.KS'), ('KOSDAQ', '.KQ')):
            top_df = (
                krx[krx['Market'].str.upper() == market_key]
                .sort_values('Marcap', ascending=False)
                .head(self.universe_size)
            )
            for _, row in top_df.iterrows():
                t = f"{row['Code']}{suffix}"
                pool.append(t)
                names[t] = row['Name']

        # S&P500: the GICS Sector column is free (no per-ticker API call), so
        # instead of an arbitrary head() slice (the FDR listing is roughly
        # alphabetical, NOT market-cap ordered) we take every constituent in
        # the seed's sector.
        sp500 = _load_sp500_listing()
        for _, row in sp500.iterrows():
            sec = str(row['Sector'])
            if seed_gics_sector and sec != seed_gics_sector:
                continue
            t = str(row['Symbol'])
            pool.append(t)
            names[t] = str(row['Name'])
            sp500_gics[t] = sec

        for gdata in INDUSTRY_GROUPS.values():
            for t, n in gdata.get('names', {}).items():
                if t not in names:
                    pool.append(t)
                    names[t] = n

        raw_pool = list(dict.fromkeys(pool))  # de-dup, keep first occurrence order

        filtered = [ticker]
        for t in raw_pool:
            if t == ticker:
                continue
            gics = sp500_gics.get(t)
            if gics is None:
                gics = _yf_sector_for(t)
            if gics and gics == seed_gics_sector:
                filtered.append(t)

        names.setdefault(ticker, ticker)

        return filtered, names, len(raw_pool)

    # ── Seed classification (display only — clustering itself is pool-wide) ──

    def _seed_info(self, ticker: str) -> tuple[str, str, str]:
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', '') or ''
            industry = info.get('industry', '') or ''
            name = info.get('shortName') or info.get('longName') or ticker
            return sector, industry, name
        except Exception:
            return '', '', ticker

    # ── Price download ─────────────────────────────────────────────────────────

    def _fetch_close(self, tickers: list[str]) -> pd.DataFrame:
        """Batch-downloads closing prices via yfinance. KRW-denominated
        (.KS/.KQ) tickers are converted to USD using daily KRW=X history so
        that movement patterns are compared on a common currency basis
        before start=100 normalization.
        """
        import yfinance as yf

        kr_tickers = [t for t in tickers if t.endswith(('.KS', '.KQ'))]
        dl_tickers = list(tickers) + (['KRW=X'] if kr_tickers else [])

        raw = yf.download(
            dl_tickers,
            period=self.period,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError("가격 데이터를 가져올 수 없습니다.")

        if isinstance(raw.columns, pd.MultiIndex):
            close = raw['Close'].copy()
        elif 'Close' in raw.columns:
            close = raw[['Close']]
            close.columns = pd.Index(dl_tickers[:1])
        else:
            raise ValueError("yfinance 응답에서 Close 컬럼을 찾을 수 없습니다.")

        if kr_tickers and 'KRW=X' in close.columns:
            fx = close['KRW=X'].ffill()
            for t in kr_tickers:
                if t in close.columns:
                    close[t] = close[t] / fx
            close = close.drop(columns=['KRW=X'])

        # Drop columns with insufficient data, then align dates
        close = close.loc[:, close.notna().sum() >= self.MIN_COMMON_DAYS]
        close = close.dropna(how='any')

        if close.shape[0] < self.MIN_COMMON_DAYS or close.shape[1] < 2:
            raise ValueError(
                f"공통 거래일이 부족합니다 ({close.shape[0]}일). "
                "기간을 줄이거나 universe_size를 낮춰보세요."
            )

        return close

    # ── Clustering ────────────────────────────────────────────────────────────

    def _optimal_k(self, X: np.ndarray) -> int:
        """Elbow method (kneedle): K with maximum perpendicular distance to diagonal."""
        from sklearn.cluster import KMeans

        n = X.shape[0]
        max_k = min(self.max_k, n - 1)
        if max_k < 2:
            return 2

        k_range = list(range(2, max_k + 1))
        inertias = [
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_
            for k in k_range
        ]

        if len(inertias) < 2:
            return k_range[0]

        # Normalize inertia to [0, 1] (0 = first, 1 = last)
        span = inertias[0] - inertias[-1]
        if span <= 0:
            return k_range[0]

        x = np.linspace(0.0, 1.0, len(k_range))
        y_norm = (inertias[0] - np.array(inertias)) / span  # increasing, 0→1

        # Max perpendicular distance from y=x diagonal → elbow
        return k_range[int(np.argmax(np.abs(y_norm - x)))]

    def _cluster(
        self, close: pd.DataFrame, seed_ticker: str
    ) -> tuple[list[str], int, int]:
        """Normalizes prices, clusters, returns (peer_tickers, opt_k, cluster_size)."""
        from sklearn.cluster import KMeans

        if seed_ticker not in close.columns:
            raise ValueError(f"'{seed_ticker}' 가격 데이터를 가져올 수 없습니다.")

        # Normalize: each stock starts at 100
        X = ((close / close.iloc[0]) * 100.0).values.T  # (n_stocks, n_days)
        tickers_list = list(close.columns)

        opt_k = self._optimal_k(X)
        labels = KMeans(n_clusters=opt_k, random_state=42, n_init=10).fit_predict(X)

        seed_label = labels[tickers_list.index(seed_ticker)]
        peers = [tickers_list[i] for i, lbl in enumerate(labels) if lbl == seed_label]

        return peers, opt_k, len(peers)

    # ── Global finder ───────────────────────────────────────────────────────────

    def _find_global(
        self, ticker: str, seed_sector: str, seed_industry: str,
    ) -> PeerGroup:
        seed_gics = _YF_TO_GICS.get(seed_sector, seed_sector)

        universe, name_map, raw_pool_size = self._build_universe_global(ticker, seed_gics)

        n_candidates = len(universe) - 1  # excluding the seed itself
        if n_candidates < self.MIN_SECTOR_CANDIDATES:
            raise ValueError(
                f"'{seed_gics}' 섹터 내 같은 업종 후보가 부족합니다 "
                f"({n_candidates}개, 최소 {self.MIN_SECTOR_CANDIDATES}개 필요) — "
                "업종이 다른 종목을 억지로 묶는 대신 폴백합니다."
            )

        close = self._fetch_close(universe)

        peers, opt_k, cluster_size = self._cluster(close, ticker)

        if cluster_size <= 1:
            # Seed ended up alone in its own cluster — its price behavior
            # diverged from every other same-sector candidate over the
            # window. An empty peer list isn't useful; let find() fall back
            # to the curated static group instead of showing "no peers".
            raise ValueError(
                f"'{seed_gics}' 섹터 내에서도 '{ticker}'와 같은 가격 움직임 군집을 "
                f"찾지 못했습니다 (K={opt_k} 중 단독 클러스터)."
            )

        # Preserve pool order (KOSPI → KOSDAQ → S&P500 → 해외, each already
        # ranked within its own source); cross-market marcap isn't a
        # comparable unit, so no re-sort — just pin the seed first.
        if ticker in peers:
            peers = [ticker] + [t for t in peers if t != ticker]
        else:
            peers = [ticker] + peers

        peers = peers[: self.top_n]
        names = {t: name_map.get(t, t) for t in peers}

        return PeerGroup(
            seed_ticker=ticker,
            sector=seed_sector or "K-means 클러스터",
            industry=(
                seed_industry + " · " if seed_industry else ""
            ) + (
                f"클러스터 크기 {cluster_size}개 "
                f"(K={opt_k}, {seed_gics} 섹터 후보 {len(close.columns)}개 / "
                f"전체 풀 {raw_pool_size}개 중 필터링)"
            ),
            source=(
                f"K-means 클러스터링 · '{seed_gics}' 섹터 내 글로벌 통합 풀 "
                "(국가/거래소 구분 없음, 업종은 yfinance sector로 1차 필터링) · "
                f"KOSPI/KOSDAQ 상위 {self.universe_size}개 + "
                f"S&P500 동일 섹터 전체 + INDUSTRY_GROUPS 해외 종목 중 "
                "동일 섹터만 · 원화 종목은 일별 환율로 USD 환산 후 정규화 · "
                "⚠ 주가 움직임 기반 결과이므로 공적분 검정 결과를 함께 확인하세요"
            ),
            tickers=peers,
            names=names,
        )
