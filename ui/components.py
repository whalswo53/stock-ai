"""
앱 전체 공용 UI 컴포넌트 — 방향성 신호를 텍스트가 아니라 극성(polarity)으로
색칠하는 표준 컴포넌트를 여기 하나로 모은다.

버그 배경: st.metric(delta="매도주의")처럼 라벨 텍스트를 delta 인자에 넣으면
Streamlit은 부호 없는 문자열을 기본 초록·↑ 로 렌더한다 — 데드크로스·매도주의
같은 약세 신호가 상승처럼 보이는 원인. render_signal_card는 색상을 polarity
값 하나로만 결정하므로(텍스트 내용을 파싱하지 않음) 이 버그가 재발할 수 없다.
"""
from __future__ import annotations

import html as _html
import re as _re

import streamlit as st

POLARITY_COLOR = {"bullish": "#1b7a3d", "neutral": "#b8860b", "bearish": "#c0392b"}
POLARITY_BG = {"bullish": "#eaf6ee", "neutral": "#fdf6e3", "bearish": "#fbeaea"}
POLARITY_ICON = {"bullish": "🟢", "neutral": "🟡", "bearish": "🔴"}
POLARITY_LABEL = {"bullish": "강세", "neutral": "중립", "bearish": "약세"}

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


_ICON_TO_POLARITY = {icon: polarity for polarity, icon in POLARITY_ICON.items()}


def _cell_polarity(value) -> str | None:
    """표 셀 값 하나에서 극성을 추론 — 페이지마다 따로 있던 판정 로직
    (🟢/🟡/🔴·✅/❌/➖/⚠️ 접두 판정문, BUY/SELL 등 시그널 문자열, 부호 있는
    수익률/손익 숫자) 네 패턴을 한 곳에서 통일 처리한다. 색상표는 항상 POLARITY_COLOR."""
    text = str(value).strip()
    sig = polarity_from_signal(text)
    if sig:
        return sig
    for icon, polarity in _ICON_TO_POLARITY.items():
        if text.startswith(icon):
            return polarity
    if text.startswith("✅"):
        return "bullish"
    if text.startswith("❌"):
        return "bearish"
    if text.startswith("⚠️"):
        return "neutral"
    if text.startswith("➖") or text in ("", "N/A", "—", "-"):
        return None
    # 통화기호(₩/$)·쉼표·단위(%,원) 등을 걷어내고 부호 있는 숫자만 추출 —
    # "₩+1,234,000"/"-3.20%" 같은 포맷된 금액·수익률 문자열의 부호로 판정.
    cleaned = _re.sub(r"[^0-9+\-.]", "", text)
    try:
        num = float(cleaned)
    except ValueError:
        return None
    if num > 0:
        return "bullish"
    if num < 0:
        return "bearish"
    return "neutral"


def render_clean_table(df, judgment_col: str | list[str] | None = None) -> None:
    """세로 격자선 없는 표 — 첫 컬럼(레이블)은 좌측, 나머지 값은 우측 정렬,
    헤더에만 얇은 하단 보더. judgment_col로 지정한 컬럼은 값을 _cell_polarity로
    판정해 POLARITY_COLOR로 색칠한다(색상표는 render_signal_card와 공유)."""
    judgment_cols = {judgment_col} if isinstance(judgment_col, str) else set(judgment_col or [])
    cols = list(df.columns)

    header_cells = "".join(
        f'<th style="text-align:{"left" if i == 0 else "right"};padding:6px 10px;'
        f'font-size:11px;color:#888;font-weight:600;white-space:nowrap;'
        f'border-bottom:1px solid rgba(128,128,128,0.35)">{_html.escape(str(c))}</th>'
        for i, c in enumerate(cols)
    )

    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for i, c in enumerate(cols):
            raw = row[c]
            text = _html.escape("" if raw is None else str(raw))
            align = "left" if i == 0 else "right"
            if c in judgment_cols:
                color = POLARITY_COLOR.get(_cell_polarity(raw), _FALLBACK_COLOR)
                text = f'<span style="color:{color};font-weight:700">{text}</span>'
            cells.append(f'<td style="text-align:{align};padding:6px 10px;font-size:13px">{text}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    st.markdown(
        f'<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )


def render_stat_grid(items: list[dict], columns: int = 3) -> None:
    """스냅샷 지표 그룹(밸류에이션·수익성 등) 카드 그리드.
    items: [{"label", "value", "eval"(선택), "polarity"(선택)}, ...]
    상단 3px 컬러바 = POLARITY_COLOR[polarity] — render_signal_card와 색상표 공유."""
    cols = st.columns(columns)
    for i, item in enumerate(items):
        color = POLARITY_COLOR.get(item.get("polarity"), _FALLBACK_COLOR)
        label = _html.escape(str(item.get("label", "")))
        value = _html.escape(str(item.get("value", "")))
        eval_text = _html.escape(str(item.get("eval") or ""))
        with cols[i % columns]:
            with st.container(border=True):
                eval_html = (
                    f'<div style="font-size:12px;color:{color};margin-top:2px;font-weight:600">{eval_text}</div>'
                    if eval_text else ""
                )
                st.markdown(
                    f'<div style="height:3px;margin:-1rem -1rem 10px -1rem;'
                    f'background:{color};border-radius:8px 8px 0 0"></div>'
                    f'<div style="font-size:12px;color:#888">{label}</div>'
                    f'<div style="font-size:20px;font-weight:700">{value}</div>'
                    f'{eval_html}',
                    unsafe_allow_html=True,
                )
