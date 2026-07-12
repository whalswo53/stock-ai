"""
앱 전체 공용 UI 컴포넌트 — 방향성 신호를 텍스트가 아니라 극성(polarity)으로
색칠하는 표준 컴포넌트를 여기 하나로 모은다.

버그 배경: st.metric(delta="매도주의")처럼 라벨 텍스트를 delta 인자에 넣으면
Streamlit은 부호 없는 문자열을 기본 초록·↑ 로 렌더한다 — 데드크로스·매도주의
같은 약세 신호가 상승처럼 보이는 원인. render_signal_card는 색상을 polarity
값 하나로만 결정하므로(텍스트 내용을 파싱하지 않음) 이 버그가 재발할 수 없다.
"""
from __future__ import annotations

import streamlit as st

POLARITY_COLOR = {"bullish": "#1b7a3d", "neutral": "#b8860b", "bearish": "#c0392b"}
POLARITY_BG = {"bullish": "#eaf6ee", "neutral": "#fdf6e3", "bearish": "#fbeaea"}
POLARITY_ICON = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}

_FALLBACK_COLOR = "#888888"
_FALLBACK_BG = "rgba(128,128,128,0.08)"

# 자주 등장하는 표준 신호 라벨 → 극성. 종목/모듈별 하드코딩이 아니라 라벨 자체의
# 범용 매핑이므로 페어트레이딩·단타 등 BUY/SELL/WAIT 신호를 쓰는 곳이면 어디서든
# 재사용 가능하다.
SIGNAL_POLARITY = {
    "BUY": "bullish", "SELL": "bearish",
    "HOLD": "neutral", "WAIT": "neutral", "CLOSE": "neutral",
    "STOP_LOSS": "bearish",
}


def polarity_from_signal(signal: str | None) -> str | None:
    if signal is None:
        return None
    return SIGNAL_POLARITY.get(signal.upper())


def render_signal_card(
    label: str,
    value: str,
    sub_label: str = "",
    polarity: str | None = None,
) -> None:
    """st.metric(delta=...) 대체. 색상은 polarity 값 하나로만 결정된다."""
    color = POLARITY_COLOR.get(polarity, _FALLBACK_COLOR)
    bg = POLARITY_BG.get(polarity, _FALLBACK_BG)
    icon = POLARITY_ICON.get(polarity, "")

    sub_html = ""
    if sub_label:
        sub_html = (
            f'<div style="font-size:12px;color:{color};margin-top:4px;font-weight:600">'
            f'{icon} {sub_label}</div>'
        )

    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {color};'
        f'border-radius:0 8px 8px 0;padding:10px 14px;margin-bottom:6px;min-height:84px">'
        f'  <div style="font-size:12px;color:#666">{label}</div>'
        f'  <div style="font-size:22px;font-weight:700;color:#1a1a1a">{value}</div>'
        f'  {sub_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_verdict_banner(
    label: str,
    confidence: float,
    polarity: str | None,
    sub_text: str = "",
) -> None:
    """종합 판정 배너 — 라벨(예: "매수 우위"/"관망") + 신뢰도 진행바."""
    color = POLARITY_COLOR.get(polarity, _FALLBACK_COLOR)
    bg = POLARITY_BG.get(polarity, _FALLBACK_BG)
    icon = POLARITY_ICON.get(polarity, "⏸")

    st.markdown(
        f'<div style="background:{bg};border:2px solid {color};border-radius:12px;'
        f'padding:18px 22px;margin-bottom:12px">'
        f'  <div style="font-size:28px;font-weight:800;color:{color}">{icon} {label}</div>'
        f'  <div style="font-size:13px;color:#555;margin-top:4px">{sub_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(confidence, 0.0), 1.0), text=f"신뢰도 {confidence:.0%}")
