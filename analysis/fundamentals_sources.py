"""
밸류에이션 원시 소스 수집기 (yfinance / 네이버 금융 / Finnhub / pykrx).

각 함수는 4그룹 taxonomy(GROUP_FIELDS)와 동일한 중첩 dict를 반환한다.
값이 없으면 None(문자열 "N/A" 아님 — fundamentals.py가 병합 후 마지막에 N/A로
변환한다). 네트워크·파싱 실패는 절대 예외로 새지 않고 빈 결과로 조용히 폴백한다.
"""
from __future__ import annotations

from analysis.fundamentals_groups import GROUP_FIELDS
from utils.net_timeout import call_with_timeout

_PYKRX_CALL_TIMEOUT = 5.0
_PYKRX_MAX_CONSECUTIVE_FAILS = 2


def _empty() -> dict:
    return {group: {label: None for label in fields} for group, fields in GROUP_FIELDS.items()}


# ═══════════════════════════════════════════════════════════════════════════
# yfinance .info — 전 시장 공통 1차 소스
# ═══════════════════════════════════════════════════════════════════════════

_INFO_KEY: dict[str, dict[str, str]] = {
    "밸류에이션": {
        "PER": "trailingPE", "PBR": "priceToBook",
        "PSR": "priceToSalesTrailing12Months", "EV/EBITDA": "enterpriseToEbitda",
    },
    "수익성": {
        "ROE": "returnOnEquity", "ROA": "returnOnAssets",
        "영업이익률": "operatingMargins", "순이익률": "profitMargins",
    },
    "성장성": {
        "매출성장(YoY)": "revenueGrowth", "EPS성장": "earningsGrowth",
    },
    "안정성": {
        "부채비율": "debtToEquity", "유동비율": "currentRatio",
    },
}


def from_yfinance_info(info: dict | None) -> dict:
    info = info or {}
    out = _empty()
    for group, fields in _INFO_KEY.items():
        for label, key in fields.items():
            v = info.get(key)
            if v is not None:
                out[group][label] = float(v)

    # PCR(주가/주당현금흐름) = 시가총액 / 영업활동현금흐름 — 둘 다 총액 기준이라
    # 주당 환산 없이 나눠도 동일한 배수가 나온다.
    cap, ocf = info.get("marketCap"), info.get("operatingCashflow")
    if cap and ocf:
        out["밸류에이션"]["PCR"] = cap / ocf

    # 배당수익률: yfinance 버전에 따라 dividendYield가 "이미 %"(예: 0.52 = 0.52%)
    # 또는 trailingAnnualDividendYield가 "분수"(예: 0.0033 = 0.33%)로 온다.
    # 내부 표준은 분수이므로 dividendYield는 100으로 나눠 맞춘다.
    dy, tady = info.get("dividendYield"), info.get("trailingAnnualDividendYield")
    if dy is not None:
        out["밸류에이션"]["배당수익률"] = float(dy) / 100.0
    elif tady is not None:
        out["밸류에이션"]["배당수익률"] = float(tady)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 네이버 금융 — 한국 종목 PER/PBR/배당수익률/ROE/영업이익률/순이익률/부채비율 보완
# ═══════════════════════════════════════════════════════════════════════════

def from_naver(ticker: str) -> dict:
    out = _empty()
    code = ticker.split(".")[0]
    try:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(
            f"https://finance.naver.com/item/main.naver?code={code}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=8,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"  # 네이버 금융은 UTF-8 (예전 euc-kr 아님, 실측 확인)
        soup = BeautifulSoup(resp.text, "html.parser")

        def _num(text: str) -> float | None:
            text = text.replace(",", "").strip()
            try:
                return float(text)
            except ValueError:
                return None

        # 대문 상단 실시간 PER/PBR/배당수익률
        per_table = soup.select_one("table.per_table")
        if per_table:
            per = per_table.select_one("em#_per")
            pbr = per_table.select_one("em#_pbr")
            dvr = per_table.select_one("em#_dvr")
            if per:
                out["밸류에이션"]["PER"] = _num(per.get_text())
            if pbr:
                out["밸류에이션"]["PBR"] = _num(pbr.get_text())
            if dvr:
                v = _num(dvr.get_text())
                if v is not None:
                    out["밸류에이션"]["배당수익률"] = v / 100.0

        # "기업실적분석" 표 — 최근 연간 실적 중 가장 최근 확정치(오른쪽에서 첫 값)
        earn_table = soup.select_one("table.tb_type1_ifrs")
        if earn_table:
            row_targets = {
                "영업이익률": ("수익성", "영업이익률", True),
                "순이익률": ("수익성", "순이익률", True),
                "ROE": ("수익성", "ROE", True),
                "부채비율": ("안정성", "부채비율", False),
            }
            for tr in earn_table.select("tbody tr"):
                th = tr.select_one("th")
                if not th:
                    continue
                label = th.get_text(strip=True)
                key = next((k for k in row_targets if label.startswith(k)), None)
                if not key:
                    continue
                tds = [td.get_text(strip=True) for td in tr.select("td")]
                raw = next((v for v in reversed(tds) if v.strip()), "")
                val = _num(raw)
                if val is None:
                    continue
                group, field, is_pct = row_targets[key]
                out[group][field] = val / 100.0 if is_pct else val
    except Exception:
        pass
    return out


def from_pykrx(ticker: str) -> dict:
    """네이버도 실패했을 때 최후 보조 — PER/PBR만 (pykrx는 재무제표 API가 없음).

    pykrx는 자체 timeout이 없고 KRX 서비스 장애 중엔 호출당 수십 초씩
    블로킹될 수 있어, call_with_timeout으로 호출 하나하나에 상한을 걸고
    연속 실패 시 나머지 날짜는 시도하지 않고 조기 포기한다(무의미한
    반복 호출 방지 — 세그폴트 인시던트 원인 후보였던 무제한 재시도 패턴).
    """
    out = _empty()
    try:
        from datetime import date, timedelta
        from pykrx import stock as krx

        code = ticker.split(".")[0]
        d = date.today()
        consecutive_fails = 0
        for _ in range(10):
            if consecutive_fails >= _PYKRX_MAX_CONSECUTIVE_FAILS:
                break
            ds = d.strftime("%Y%m%d")
            try:
                fdf = call_with_timeout(
                    krx.get_market_fundamental_by_date, ds, ds, code,
                    timeout=_PYKRX_CALL_TIMEOUT,
                )
            except Exception:
                consecutive_fails += 1
                d -= timedelta(days=1)
                continue
            consecutive_fails = 0
            if fdf is not None and not fdf.empty:
                row = fdf.iloc[-1]
                per = float(row.get("PER", 0) or 0)
                pbr = float(row.get("PBR", 0) or 0)
                if per > 0:
                    out["밸류에이션"]["PER"] = per
                if pbr > 0:
                    out["밸류에이션"]["PBR"] = pbr
                break
            d -= timedelta(days=1)
    except Exception:
        pass
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Finnhub — 미국 종목 yfinance 공백분 보완 (무료 티어, 기존 FINNHUB_API_KEY 재사용)
# ═══════════════════════════════════════════════════════════════════════════

def from_finnhub(ticker: str) -> dict:
    """Finnhub /stock/metric?metric=all 기반 보완.

    필드명은 Finnhub 공개 문서 기준이며 이 환경에는 키가 없어 실호출 검증은
    못 했다 — 필드명이 틀려도 .get()이 None을 반환할 뿐이라 크래시는 없고,
    조용히 채워지지 않는 선에서 그친다.
    """
    out = _empty()
    try:
        from config.settings import FINNHUB_API_KEY
        if not FINNHUB_API_KEY:
            return out

        import requests
        resp = requests.get(
            "https://finnhub.io/api/v1/stock/metric",
            params={"symbol": ticker.upper(), "metric": "all", "token": FINNHUB_API_KEY},
            timeout=6,
        )
        resp.raise_for_status()
        m = (resp.json() or {}).get("metric") or {}

        def g(*keys):
            for k in keys:
                v = m.get(k)
                if v is not None:
                    return v
            return None

        val = g("peTTM", "peExclExtraTTM", "peBasicExclExtraTTM")
        if val is not None:
            out["밸류에이션"]["PER"] = float(val)
        val = g("pbAnnual", "pbQuarterly")
        if val is not None:
            out["밸류에이션"]["PBR"] = float(val)
        val = g("psTTM", "psAnnual")
        if val is not None:
            out["밸류에이션"]["PSR"] = float(val)
        val = g("currentDividendYieldTTM", "dividendYieldIndicatedAnnual")
        if val is not None:
            out["밸류에이션"]["배당수익률"] = float(val) / 100.0

        val = g("roeTTM", "roeRfy")
        if val is not None:
            out["수익성"]["ROE"] = float(val) / 100.0
        val = g("roaTTM", "roaRfy")
        if val is not None:
            out["수익성"]["ROA"] = float(val) / 100.0
        val = g("operatingMarginTTM", "operatingMarginAnnual")
        if val is not None:
            out["수익성"]["영업이익률"] = float(val) / 100.0
        val = g("netProfitMarginTTM", "netProfitMarginAnnual")
        if val is not None:
            out["수익성"]["순이익률"] = float(val) / 100.0

        val = g("revenueGrowthTTMYoy", "revenueGrowthQuarterlyYoy")
        if val is not None:
            out["성장성"]["매출성장(YoY)"] = float(val) / 100.0
        val = g("epsGrowthTTMYoy", "epsGrowthQuarterlyYoy")
        if val is not None:
            out["성장성"]["EPS성장"] = float(val) / 100.0

        val = g("totalDebt/totalEquityAnnual", "totalDebt/totalEquityQuarterly")
        if val is not None:
            out["안정성"]["부채비율"] = float(val)
        val = g("currentRatioAnnual", "currentRatioQuarterly")
        if val is not None:
            out["안정성"]["유동비율"] = float(val)
        val = g("netInterestCoverageTTM", "netInterestCoverageAnnual")
        if val is not None:
            out["안정성"]["이자보상배율"] = float(val)
    except Exception:
        pass
    return out
