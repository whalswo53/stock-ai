"""
공통 티커 유틸리티.
- 정적 딕셔너리 (KOSPI/NASDAQ 주요 종목 + 한글명 → US 티커)
- FinanceDataReader 기반 동적 KRX 전체 + 미국 S&P500 로드 (6시간 캐시)
- KRX 로드 실패 시 1회 재시도(3초 대기) → 하드코딩 fallback 사용
- 주요 KRX ETF 하드코딩 (50여 종, pykrx 인증 불필요)
- search_tickers() : 한글/영문 부분 매칭 + 우선순위 정렬
"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from config.sources import (
    KOSPI_TICKER_MAP,
    NASDAQ_TICKER_MAP,
    HK_CN_TICKER_MAP,
    TICKER_KR_NAME as _TICKER_KR_NAME,
)

# ── Static base map ────────────────────────────────────────────────────────────
_NAME_TO_TICKER: dict[str, str] = {**KOSPI_TICKER_MAP, **NASDAQ_TICKER_MAP, **HK_CN_TICKER_MAP}
_SORTED_NAMES: list[str] = sorted(_NAME_TO_TICKER, key=len, reverse=True)

# ── 비상장 기업 (검색 시 안내만 표시, 매핑 티커 없음) ────────────────────────
_UNLISTED_NAMES: dict[str, str] = {
    "화웨이": "화웨이 — 비상장 기업으로 주가 데이터 없음",
    "Huawei": "Huawei — 비상장 기업으로 주가 데이터 없음",
}

# ── Cache (6시간 TTL) ─────────────────────────────────────────────────────────
_CACHE_TTL = 21_600  # 6 hours (reduced from 24h — retries faster after KRX outage)
_KRX_CACHE: dict[str, tuple[str, str]] | None = None  # {tkr: (name, market)}
_KRX_CACHE_TS: float = 0.0
_US_CACHE: dict[str, str] | None = None               # {symbol: name}
_US_CACHE_TS: float = 0.0

# ── KRX fallback (FDR 실패 시 사용) ──────────────────────────────────────────
# TICKER_KR_NAME의 KS/KQ 항목을 (name, market) 형태로 변환
_KRX_FALLBACK: dict[str, tuple[str, str]] = {
    tkr: (name, "KOSPI" if tkr.endswith(".KS") else "KOSDAQ")
    for tkr, name in _TICKER_KR_NAME.items()
    if tkr.endswith((".KS", ".KQ"))
}

# ── 주요 KRX ETF (yfinance 6자리 코드 → 한글 상품명) ─────────────────────────
# ETF는 FDR/pykrx로 가져올 수 없어 주요 50종 하드코딩. 모두 yfinance ".KS" 사용.
_KRX_ETF_NAMES: dict[str, str] = {
    # ── KODEX (삼성자산운용) ──────────────────────────────────────────────
    "069500": "KODEX 200",
    "122630": "KODEX 레버리지",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "229200": "KODEX 코스닥150",
    "233740": "KODEX 코스닥150레버리지",
    "251340": "KODEX 코스닥150인버스",
    "091160": "KODEX 반도체",
    "091170": "KODEX 은행",
    "305720": "KODEX 2차전지산업",
    "379800": "KODEX 미국S&P500TR",
    "219480": "KODEX 미국S&P500선물(H)",
    "304940": "KODEX 미국나스닥100선물(H)",
    "278530": "KODEX 200TR",
    "132030": "KODEX 골드선물(H)",
    "395160": "KODEX 시스템반도체",
    "102960": "KODEX 헬스케어",
    # ── TIGER (미래에셋자산운용) ──────────────────────────────────────────
    "102110": "TIGER 200",
    "091230": "TIGER 반도체",
    "360750": "TIGER 미국S&P500",
    "133690": "TIGER 미국나스닥100",
    "305540": "TIGER 2차전지테마",
    "381170": "TIGER 미국테크TOP10",
    "364980": "TIGER 2차전지TOP10",
    "157490": "TIGER 소프트웨어",
    "195930": "TIGER 유로스탁스50(합성H)",
    "329200": "TIGER 리츠부동산인프라",
    "305080": "TIGER 미국채10년선물",
    "143460": "TIGER 200선물레버리지",
    # ── KBSTAR / RISE (KB자산운용) ─────────────────────────────────────
    "368590": "KBSTAR 미국나스닥100",
    "091220": "KBSTAR 200",
    # ── ACE (한화자산운용, 구 ARIRANG) ────────────────────────────────────
    "152100": "ACE 200",
    "292050": "ACE 코스닥150",
    # ── HANARO (NH아문디자산운용) ─────────────────────────────────────────
    "292500": "HANARO 200",
    # ── 금/채권/기타 ──────────────────────────────────────────────────────
    "411060": "KINDEX KRX금현물",
    "148070": "KOSEF 국고채10년",
}

# ── 미국 주요 종목 한글명 → 티커 ──────────────────────────────────────────────
_US_KR_NAMES: dict[str, str] = {
    # 빅테크 / FAANG
    "애플": "AAPL", "아이폰": "AAPL",
    "마이크로소프트": "MSFT", "윈도우": "MSFT",
    "엔비디아": "NVDA", "에누비디아": "NVDA",
    "알파벳": "GOOGL", "구글": "GOOGL",
    "아마존": "AMZN", "아마존웹서비스": "AMZN",
    "메타": "META", "페이스북": "META",
    "테슬라": "TSLA",
    "넷플릭스": "NFLX",
    # 반도체
    "AMD": "AMD", "어드밴스드마이크로디바이시스": "AMD",
    "인텔": "INTC",
    "퀄컴": "QCOM",
    "브로드컴": "AVGO",
    "TSMC": "TSM", "대만반도체": "TSM",
    "마이크론": "MU",
    "ASML": "ASML",
    "텍사스인스트루먼트": "TXN",
    "어플라이드머티리얼즈": "AMAT",
    "램리서치": "LRCX",
    "KLA": "KLAC",
    "마벨": "MRVL",
    "온세미컨덕터": "ON", "온세미": "ON",
    "ARM": "ARM",
    "슈퍼마이크로": "SMCI", "슈퍼마이크로컴퓨터": "SMCI",
    # 핀테크 / 금융
    "페이팔": "PYPL",
    "코인베이스": "COIN",
    "팔란티어": "PLTR",
    "버크셔해서웨이": "BRK-B", "버크셔": "BRK-B",
    "JP모건": "JPM",
    "골드만삭스": "GS",
    "모건스탠리": "MS",
    "비자": "V",
    "마스터카드": "MA",
    "블랙록": "BLK",
    "찰스슈왑": "SCHW",
    "인터랙티브브로커": "IBKR",
    "뱅크오브아메리카": "BAC",
    "씨티그룹": "C", "씨티": "C",
    "웰스파고": "WFC",
    "아메리칸익스프레스": "AXP", "아멕스": "AXP",
    # 클라우드 / SaaS
    "세일즈포스": "CRM",
    "어도비": "ADBE",
    "오라클": "ORCL",
    "서비스나우": "NOW",
    "스노우플레이크": "SNOW",
    "데이터독": "DDOG",
    "몽고DB": "MDB",
    "크라우드스트라이크": "CRWD", "클라우드스트라이크": "CRWD",
    "팔로알토": "PANW",
    "포티넷": "FTNT",
    "줌": "ZM", "줌비디오": "ZM",
    "옥타": "OKTA",
    "지스케일러": "ZS",
    "워크데이": "WDAY",
    "인튜이트": "INTU",
    "시놉시스": "SNPS",
    "캐던스": "CDNS",
    "아틀라시안": "TEAM",
    # 모빌리티 / EV
    "리비안": "RIVN",
    "루시드": "LCID",
    "GM": "GM", "제너럴모터스": "GM",
    "포드": "F",
    "우버": "UBER",
    "리프트": "LYFT",
    # 바이오 / 헬스케어
    "모더나": "MRNA",
    "화이자": "PFE",
    "길리어드": "GILD",
    "애브비": "ABBV",
    "일라이릴리": "LLY", "릴리": "LLY",
    "존슨앤존슨": "JNJ",
    "머크": "MRK",
    "아스트라제네카": "AZN",
    "인튜이티브서지컬": "ISRG",
    "리제네론": "REGN",
    # 미디어 / 엔터
    "디즈니": "DIS",
    "컴캐스트": "CMCSA",
    "로블록스": "RBLX",
    "유니티": "U",
    "일렉트로닉아츠": "EA",
    "스포티파이": "SPOT",
    "에어비앤비": "ABNB",
    "부킹홀딩스": "BKNG", "부킹": "BKNG",
    "쇼피파이": "SHOP",
    # 소비재 / 리테일
    "코스트코": "COST",
    "월마트": "WMT",
    "타겟": "TGT",
    "홈디포": "HD",
    "나이키": "NKE",
    "스타벅스": "SBUX",
    "맥도날드": "MCD",
    "치폴레": "CMG",
    "코카콜라": "KO",
    "펩시코": "PEP", "펩시": "PEP",
    "로스스토어": "ROST",
    # 에너지 / 산업
    "엑슨모빌": "XOM",
    "쉐브런": "CVX",
    "보잉": "BA",
    "캐터필러": "CAT",
    "3M": "MMM",
    "허니웰": "HON",
    # 통신
    "AT&T": "T",
    "버라이즌": "VZ",
    "T모바일": "TMUS",
    # 중국 ADR
    "알리바바": "BABA",
    "바이두": "BIDU",
    "JD닷컴": "JD",
    "니오": "NIO",
    "리오토": "LI",
    "샤오펑": "XPEV",
    "핀둬둬": "PDD",
    # 버티브 / AI 인프라
    "버티브": "VRT",
    "마이크로스트래티지": "MSTR", "마이크로전략": "MSTR",
    "AST스페이스모바일": "ASTS", "AST": "ASTS",
    "로빈후드": "HOOD",
    "소파이": "SOFI",
    "어펌": "AFRM",
    # ETF
    "SPY": "SPY",
    "QQQ": "QQQ",
    "TQQQ": "TQQQ",
    "SQQQ": "SQQQ",
    "SOXL": "SOXL",
    "ARKK": "ARKK", "아크이노베이션": "ARKK",
    "VTI": "VTI",
    "IWM": "IWM",
    "IBM": "IBM",
    "시스코": "CSCO",
    "델": "DELL",
}

# ── Exact-match lookup (대소문자 무시) ────────────────────────────────────────
# "SMIC"/"TSMC"처럼 영문 1-5자 회사명이 실제 티커와 다른 경우,
# 원시 티커로 오인되기 전에 먼저 매핑되도록 대문자 키로 색인한다.
_EXACT_NAME_TO_TICKER: dict[str, str] = {
    **{k.upper(): v for k, v in _NAME_TO_TICKER.items()},
    **{k.upper(): v for k, v in _US_KR_NAMES.items()},
}

# ── NASDAQ100 영문명 (FDR 대신 하드코딩 — FDR NASDAQ 로드가 11초+ 소요) ─────
_NASDAQ100_EN: dict[str, str] = {
    "AAPL": "Apple Inc", "MSFT": "Microsoft Corp",
    "NVDA": "NVIDIA Corp", "AMZN": "Amazon.com Inc",
    "META": "Meta Platforms Inc", "TSLA": "Tesla Inc",
    "GOOGL": "Alphabet Inc Class A", "GOOG": "Alphabet Inc Class C",
    "AVGO": "Broadcom Inc", "NFLX": "Netflix Inc",
    "COST": "Costco Wholesale", "AMD": "Advanced Micro Devices",
    "ASML": "ASML Holding", "QCOM": "Qualcomm Inc",
    "INTC": "Intel Corp", "INTU": "Intuit Inc",
    "CSCO": "Cisco Systems", "AMAT": "Applied Materials",
    "TXN": "Texas Instruments", "ADP": "Automatic Data Processing",
    "BKNG": "Booking Holdings", "ISRG": "Intuitive Surgical",
    "MU": "Micron Technology", "SBUX": "Starbucks Corp",
    "KLAC": "KLA Corp", "ADI": "Analog Devices",
    "LRCX": "Lam Research", "PDD": "PDD Holdings",
    "PANW": "Palo Alto Networks", "SNPS": "Synopsys Inc",
    "CDNS": "Cadence Design Systems", "MELI": "MercadoLibre Inc",
    "CRWD": "CrowdStrike Holdings", "FTNT": "Fortinet Inc",
    "MRVL": "Marvell Technology", "WDAY": "Workday Inc",
    "TEAM": "Atlassian Corp", "ABNB": "Airbnb Inc",
    "ROST": "Ross Stores", "PCAR": "PACCAR Inc",
    "PAYX": "Paychex Inc", "ADSK": "Autodesk Inc",
    "IDXX": "IDEXX Laboratories", "EA": "Electronic Arts",
    "NXPI": "NXP Semiconductors", "DDOG": "Datadog Inc",
    "CDW": "CDW Corp", "ZS": "Zscaler Inc",
    "GILD": "Gilead Sciences", "MRNA": "Moderna Inc",
    "PLTR": "Palantir Technologies", "ZM": "Zoom Video Communications",
    "COIN": "Coinbase Global", "ARM": "ARM Holdings",
    "SMCI": "Super Micro Computer", "TSM": "Taiwan Semiconductor",
    "ON": "ON Semiconductor", "REGN": "Regeneron Pharmaceuticals",
    "MAR": "Marriott International", "ORLY": "O'Reilly Automotive",
    "CTAS": "Cintas Corp", "MNST": "Monster Beverage",
    "ROP": "Roper Technologies", "CPRT": "Copart Inc",
    "FAST": "Fastenal Co", "CHTR": "Charter Communications",
    "GEHC": "GE HealthCare", "BIIB": "Biogen Inc",
    "FANG": "Diamondback Energy", "TTWO": "Take-Two Interactive",
    "ALGN": "Align Technology", "AZN": "AstraZeneca",
    "SHOP": "Shopify Inc", "SNOW": "Snowflake Inc",
    "UBER": "Uber Technologies", "RIVN": "Rivian Automotive",
    "MDB": "MongoDB Inc", "OKTA": "Okta Inc",
    "CRM": "Salesforce Inc", "NOW": "ServiceNow Inc",
    "ADBE": "Adobe Inc", "ORCL": "Oracle Corp",
    "IBM": "IBM Corp", "PYPL": "PayPal Holdings",
    "BABA": "Alibaba Group", "JD": "JD.com Inc",
    "NIO": "NIO Inc", "XPEV": "XPeng Inc",
    "LI": "Li Auto Inc", "BIDU": "Baidu Inc",
    "VRT": "Vertiv Holdings", "MSTR": "MicroStrategy Inc",
    "ASTS": "AST SpaceMobile", "SMCI": "Super Micro Computer",
    "DELL": "Dell Technologies", "HPE": "HP Enterprise",
    "GE": "GE Aerospace", "BA": "Boeing Co",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "BAC": "Bank of America", "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase", "V": "Visa Inc",
    "MA": "Mastercard Inc", "AXP": "American Express",
    "BLK": "BlackRock Inc", "SCHW": "Charles Schwab",
    "LLY": "Eli Lilly", "ABBV": "AbbVie Inc",
    "PFE": "Pfizer Inc", "MRK": "Merck & Co",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth Group",
    "DIS": "Walt Disney Co", "CMCSA": "Comcast Corp",
    "NFLX": "Netflix Inc", "SPOT": "Spotify Technology",
    "RBLX": "Roblox Corp", "U": "Unity Software",
    "WMT": "Walmart Inc", "COST": "Costco Wholesale",
    "TGT": "Target Corp", "HD": "Home Depot",
    "NKE": "Nike Inc", "MCD": "McDonald's Corp",
    "SBUX": "Starbucks Corp", "CMG": "Chipotle Mexican Grill",
    "KO": "Coca-Cola Co", "PEP": "PepsiCo Inc",
    "XOM": "Exxon Mobil", "CVX": "Chevron Corp",
    "CAT": "Caterpillar Inc", "MMM": "3M Co",
    "HON": "Honeywell International", "T": "AT&T Inc",
    "VZ": "Verizon Communications", "TMUS": "T-Mobile US",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ ETF",
    "TQQQ": "ProShares UltraPro QQQ", "SOXL": "Direxion Daily Semicon Bull",
    "ARKK": "ARK Innovation ETF", "VTI": "Vanguard Total Stock Market",
    "IWM": "iShares Russell 2000 ETF", "HOOD": "Robinhood Markets",
    "SOFI": "SoFi Technologies", "AFRM": "Affirm Holdings",
    "LYFT": "Lyft Inc", "LCID": "Lucid Group",
    "F": "Ford Motor Co", "GM": "General Motors",
    "BRK-B": "Berkshire Hathaway B",
}


# ── Dynamic loading helpers ───────────────────────────────────────────────────

def _try_fdr_krx() -> dict[str, tuple[str, str]]:
    """
    FDR로 KOSPI + KOSDAQ 전체 종목을 로드 시도.
    실패 시 빈 dict 반환 (예외를 밖으로 전파하지 않음).
    """
    result: dict[str, tuple[str, str]] = {}
    try:
        import FinanceDataReader as fdr
        for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
            try:
                df = fdr.StockListing(market)
                code_col = next(
                    (c for c in df.columns if c in ("Code", "Symbol", "code")), None
                )
                name_col = next(
                    (c for c in df.columns if c in ("Name", "name", "종목명")), None
                )
                if code_col and name_col:
                    for _, row in df.iterrows():
                        code = str(row[code_col]).zfill(6)
                        name = str(row[name_col]).strip()
                        if code and name and code != "nan" and name != "nan":
                            result[f"{code}{suffix}"] = (name, market)
            except Exception:
                pass
    except ImportError:
        pass
    return result


def _load_krx_stocks() -> dict[str, tuple[str, str]]:
    """
    FDR로 KOSPI + KOSDAQ 전체 종목을 로드한다.
    Returns {yfinance_ticker: (name, market_display)}.
    6시간 캐시. 실패 시 3초 후 1회 재시도 → 하드코딩 fallback 사용.
    """
    global _KRX_CACHE, _KRX_CACHE_TS
    if _KRX_CACHE is not None and time.time() - _KRX_CACHE_TS < _CACHE_TTL:
        return _KRX_CACHE

    # 1차 시도
    result = _try_fdr_krx()

    # 실패 시 3초 대기 후 1회 재시도
    if not result:
        time.sleep(3)
        result = _try_fdr_krx()

    # 재시도도 실패 → 하드코딩 fallback 사용
    if not result:
        _KRX_CACHE = dict(_KRX_FALLBACK)
        _KRX_CACHE_TS = time.time()
        return _KRX_CACHE

    # pykrx 보조 (FDR 성공 시에만 실행, 인증 불필요한 경우)
    try:
        from pykrx import stock as krx
        from datetime import date
        today = date.today().strftime("%Y%m%d")
        for market, suffix in [("KOSPI", ".KS"), ("KOSDAQ", ".KQ")]:
            try:
                tickers = krx.get_market_ticker_list(date=today, market=market)
                for t in tickers:
                    key = f"{t}{suffix}"
                    if key not in result:
                        try:
                            name = krx.get_market_ticker_name(t)
                            if name:
                                result[key] = (name, market)
                        except Exception:
                            pass
            except Exception:
                pass
    except (ImportError, Exception):
        pass

    _KRX_CACHE = result
    _KRX_CACHE_TS = time.time()
    return result


def _load_us_stocks() -> dict[str, str]:
    """
    FDR S&P500 리스트를 로드한다. {symbol: company_name}. 24시간 캐시.
    """
    global _US_CACHE, _US_CACHE_TS
    if _US_CACHE is not None and time.time() - _US_CACHE_TS < _CACHE_TTL:
        return _US_CACHE

    result: dict[str, str] = {}
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("S&P500")
        sym_col = next(
            (c for c in df.columns if c in ("Symbol", "symbol")), None
        )
        name_col = next(
            (c for c in df.columns if c in ("Name", "name")), None
        )
        if sym_col and name_col:
            for _, row in df.iterrows():
                sym = str(row[sym_col]).strip()
                name = str(row[name_col]).strip()
                if sym and name and sym != "nan":
                    result[sym] = name
    except Exception:
        pass

    _US_CACHE = result
    _US_CACHE_TS = time.time()
    return result


def is_krx_cache_warm() -> bool:
    """KRX 캐시가 유효한지 확인 (UI 스피너 결정용)."""
    return _KRX_CACHE is not None and time.time() - _KRX_CACHE_TS < _CACHE_TTL


def is_us_cache_warm() -> bool:
    """US 캐시가 유효한지 확인."""
    return _US_CACHE is not None and time.time() - _US_CACHE_TS < _CACHE_TTL


# ── Core resolution functions ──────────────────────────────────────────────────

def resolve_ticker(raw: str) -> str:
    """
    한글/영문 회사명 또는 원시 입력 → yfinance 티커 변환.
    검색 우선순위:
      1. 정규 형식 (6자리 숫자+.KS/.KQ, 4-5자리 숫자+.HK/.SS/.SZ)
      2. 정적 맵 대소문자 무시 완전 일치 (_NAME_TO_TICKER + _US_KR_NAMES)
      3. 영문 1-5자 원시 티커 형식
      4. 정적 맵 부분 일치 (_NAME_TO_TICKER + _US_KR_NAMES)
      5. ETF 맵 (한글 ETF명)
      6. 입력값 대문자 반환 (fallback)
    """
    raw = raw.strip()
    if not raw:
        return raw
    if any(name in raw for name in _UNLISTED_NAMES):
        return ""
    if re.match(r"^\d{6}\.(KS|KQ)$", raw, re.IGNORECASE):
        return raw.upper()
    if re.match(r"^\d{4,5}\.(HK|SS|SZ)$", raw, re.IGNORECASE):
        return raw.upper()
    if raw.upper() in _EXACT_NAME_TO_TICKER:
        return _EXACT_NAME_TO_TICKER[raw.upper()]
    if re.match(r"^[A-Z]{1,5}(-[A-Z])?$", raw, re.IGNORECASE):
        return raw.upper()
    for name in _SORTED_NAMES:
        if name in raw:
            return _NAME_TO_TICKER[name]
    sorted_kr = sorted(_US_KR_NAMES, key=len, reverse=True)
    for name in sorted_kr:
        if name in raw:
            return _US_KR_NAMES[name]
    for code, etf_name in _KRX_ETF_NAMES.items():
        if etf_name.lower() in raw.lower() or raw.lower() in etf_name.lower():
            return f"{code}.KS"
    return raw.upper()


def detect_market(ticker: str) -> str:
    """티커 접미사로 시장을 자동 감지한다."""
    t = ticker.upper()
    if t.endswith(".KS"):
        return "KOSPI"
    if t.endswith(".KQ"):
        return "KOSDAQ"
    if t.endswith(".HK"):
        return "HKEX"
    if t.endswith(".SS"):
        return "SSE"
    if t.endswith(".SZ"):
        return "SZSE"
    return "NASDAQ"


def is_kr(ticker: str) -> bool:
    """한국 시장 티커(.KS / .KQ)이면 True."""
    return ticker.upper().endswith((".KS", ".KQ"))


def fmt_price(val: float, ticker: str) -> str:
    """티커 기반으로 통화 포맷을 자동 선택한다."""
    t = ticker.upper()
    if is_kr(t):
        return f"₩{val:,.0f}"
    if t.endswith(".HK"):
        return f"HK${val:,.2f}"
    if t.endswith((".SS", ".SZ")):
        return f"¥{val:,.2f}"
    return f"${val:,.2f}"


def get_display_name(ticker: str) -> str:
    """
    티커 → 표시용 회사명 반환 (정적 맵 기반, 네트워크 없음).
    TICKER_KR_NAME에 없으면 ticker 그대로 반환.
    """
    return _TICKER_KR_NAME.get(ticker, ticker)


# ── Search ────────────────────────────────────────────────────────────────────

def search_tickers(
    query: str,
    max_results: int = 8,
    *,
    include_us: bool = True,
    include_krx: bool = True,
) -> list[tuple[str, str, str]]:
    """
    부분 문자열 매칭으로 종목 후보를 반환한다.
    Returns list of (yfinance_ticker, display_name, market_label).

    정렬 키: (name_prio, source_rank, name_len)
      name_prio : 0=이름/티커 시작 일치, 1=포함 일치
      source_rank: 0=정적맵+ETF, 1=NASDAQ100, 2=동적 KRX/US
    ETF와 정적 맵은 동적으로 로드된 소형주보다 항상 앞에 배치.
    """
    q = query.strip()
    if not q:
        return []
    q_lo = q.lower()

    seen: set[str] = set()
    # (name_prio, source_rank, name_len, ticker, name, market)
    heap: list[tuple[int, int, int, str, str, str]] = []

    def _add(ticker: str, name: str, market: str, src_rank: int) -> None:
        if not ticker or ticker in seen:
            return
        name_lo = name.lower()
        tkr_lo = ticker.lower()
        if q_lo in name_lo or q_lo in tkr_lo:
            seen.add(ticker)
            name_prio = 0 if (name_lo.startswith(q_lo) or tkr_lo.startswith(q_lo)) else 1
            heap.append((name_prio, src_rank, len(name), ticker, name, market))

    # ① 정적 KOSPI/NASDAQ 맵 (src_rank=0)
    for name, tkr in _NAME_TO_TICKER.items():
        _add(tkr, name, detect_market(tkr), 0)

    # ② 한글 → US 추가 매핑 (src_rank=0)
    for kr_name, tkr in _US_KR_NAMES.items():
        _add(tkr, kr_name, "US", 0)

    # ③ KRX ETF 하드코딩 (src_rank=0 — ETF는 동적 소형주보다 앞)
    if include_krx:
        for code, etf_name in _KRX_ETF_NAMES.items():
            _add(f"{code}.KS", etf_name, "KRX ETF", 0)

    # ④ NASDAQ100 영문명 (src_rank=1)
    if include_us:
        for sym, en_name in _NASDAQ100_EN.items():
            _add(sym, en_name, "NASDAQ", 1)

    # ⑤ 동적 KRX 전체 (FDR, 캐시 있으면 사용, src_rank=2)
    if include_krx and is_krx_cache_warm():
        for tkr, (name, mkt) in _load_krx_stocks().items():
            _add(tkr, name, mkt, 2)

    # ⑥ 동적 S&P500 (캐시 있으면 사용, src_rank=2)
    if include_us and is_us_cache_warm():
        for sym, name in _load_us_stocks().items():
            _add(sym, name, "S&P500", 2)

    heap.sort(key=lambda x: (x[0], x[1], x[2]))
    results = [(t, n, m) for _, _, _, t, n, m in heap[:max_results]]

    # ⑦ 비상장 기업 안내 (매핑 티커 없음 — 항상 최상단에 표시)
    for name, notice in _UNLISTED_NAMES.items():
        if q_lo in name.lower():
            results.insert(0, ("", notice, "비상장"))
            break

    return results[:max_results]
