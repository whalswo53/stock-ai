"""
공통 티커 유틸리티.
여러 페이지에서 동일하게 사용되는 함수들을 한 곳에서 관리한다.
"""
from __future__ import annotations

import re

from config.sources import KOSPI_TICKER_MAP, NASDAQ_TICKER_MAP

_NAME_TO_TICKER: dict[str, str] = {**KOSPI_TICKER_MAP, **NASDAQ_TICKER_MAP}
_SORTED_NAMES: list[str] = sorted(_NAME_TO_TICKER, key=len, reverse=True)


def resolve_ticker(raw: str) -> str:
    """
    한글 회사명 또는 원시 입력을 yfinance 티커 문자열로 변환한다.
    - 6자리 숫자+.KS/.KQ: 그대로 반환
    - 1-5 영문 대문자: 그대로 반환 (미국 티커)
    - 한글/영문 회사명: 가장 긴 일치 항목 우선 탐색
    - 찾지 못한 경우: 입력값 대문자 반환
    """
    raw = raw.strip()
    if re.match(r"^\d{6}\.(KS|KQ)$", raw, re.IGNORECASE):
        return raw.upper()
    if re.match(r"^[A-Z]{1,5}$", raw, re.IGNORECASE):
        return raw.upper()
    for name in _SORTED_NAMES:
        if name in raw:
            return _NAME_TO_TICKER[name]
    return raw.upper()


def detect_market(ticker: str) -> str:
    """티커 접미사로 시장을 자동 감지한다."""
    t = ticker.upper()
    if t.endswith(".KS"):
        return "KOSPI"
    if t.endswith(".KQ"):
        return "KOSDAQ"
    return "NASDAQ"


def is_kr(ticker: str) -> bool:
    """한국 시장 티커(.KS / .KQ)이면 True."""
    return ticker.upper().endswith((".KS", ".KQ"))


def fmt_price(val: float, ticker: str) -> str:
    """티커 기반으로 통화 포맷을 자동 선택한다."""
    return f"₩{val:,.0f}" if is_kr(ticker) else f"${val:,.2f}"
