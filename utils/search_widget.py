"""
종목 검색 자동완성 위젯.
사용법:
    from utils.search_widget import ticker_search_widget
    ticker = ticker_search_widget("page_key", "종목 검색")
    # → "005930.KS" / "NVDA" 형태의 yfinance 티커 반환
"""
from __future__ import annotations

import streamlit as st

from utils.ticker_utils import (
    is_krx_cache_warm,
    is_us_cache_warm,
    resolve_ticker,
    search_tickers,
    _load_krx_stocks,
    _load_us_stocks,
)

_PLACEHOLDER = "예: 삼성전자 · 005930.KS · NVDA · 엔비디아 · 반도체"


def _warm_cache(spinner_container=None) -> None:
    """최초 1회 FDR 로드 (KRX 전체 + S&P500). 이후 24시간 캐시."""
    need_krx = not is_krx_cache_warm()
    need_us = not is_us_cache_warm()
    if not need_krx and not need_us:
        return

    ctx = spinner_container or st
    with ctx.spinner("📦 종목 DB 초기화 중… (최초 1회, 약 3~5초)"):
        if need_krx:
            _load_krx_stocks()
        if need_us:
            _load_us_stocks()


def ticker_search_widget(
    key: str,
    label: str = "종목 검색",
    default: str = "",
    placeholder: str = _PLACEHOLDER,
    max_candidates: int = 8,
) -> str:
    """
    종목 자동완성 텍스트 입력 위젯.

    Parameters
    ----------
    key:            페이지마다 고유한 식별자 (Streamlit 위젯 키 접두사)
    label:          텍스트 입력창 레이블
    default:        초기 입력값 (예: 포트폴리오 jump 시)
    placeholder:    입력창 힌트 텍스트
    max_candidates: 자동완성 드롭다운 최대 후보 수

    Returns
    -------
    str: yfinance 호환 티커 문자열 (예: "005930.KS", "NVDA").
         아무 것도 입력하지 않으면 빈 문자열.
    """
    # 캐시 워밍 (최초 1회 스피너 표시)
    _warm_cache()

    q_key = f"_tsq_{key}"
    sel_key = f"_tss_{key}"

    # default 값 변경(포트폴리오 jump 등)이 있으면 세션 상태 업데이트
    if default and st.session_state.get(q_key, "") == "" :
        st.session_state[q_key] = default

    query = st.text_input(
        label,
        key=q_key,
        placeholder=placeholder,
    )

    if not query or not query.strip():
        return ""

    q = query.strip()
    candidates = search_tickers(q, max_results=max_candidates)

    if not candidates:
        # 후보 없음 → 직접 resolve
        return resolve_ticker(q)

    # 드롭다운 옵션 구성
    #   index 0 = "↩ 직접 입력"
    #   index 1+ = 후보들
    options = ["↩  직접 입력 (검색 건너뜀)"] + [
        f"{name}  ({tkr})  [{mkt}]"
        for tkr, name, mkt in candidates
    ]

    sel = st.selectbox(
        "검색 결과",
        options=options,
        key=sel_key,
        label_visibility="collapsed",
    )

    if sel.startswith("↩"):
        return resolve_ticker(q)

    # 선택한 항목에서 티커 추출
    idx = options.index(sel) - 1   # ↩ 항목 제외
    return candidates[idx][0]
