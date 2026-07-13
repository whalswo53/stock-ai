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

# CLOSE/청산 전용 — WAIT과 같은 neutral(앰버)이 아니라 별도 슬레이트로 구분한다
# (표 디자인 스펙 지정). POLARITY_COLOR/BG는 그대로 두고 이 케이스만 별도 처리.
SLATE_COLOR = "#64748b"
SLATE_BG = "rgba(100,116,139,0.14)"

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
    # "BUY (매수)"처럼 시그널 뒤에 한글 설명이 붙는 표시용 문자열도 커버 —
    # 앞쪽 영문 토큰만 뽑아 같은 SIGNAL_POLARITY로 재판정한다(새 매핑 아님).
    leading = _re.match(r"^[A-Za-z_]+", text)
    if leading:
        sig = polarity_from_signal(leading.group(0))
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


_DATE_TEXT_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC_LEADING_RE = _re.compile(r"^[+\-−±₩$]?\s*[\d,]*\.?\d")


def _is_numeric_text(text: str) -> bool:
    """이 셀이 '숫자 값'인지 — 전부 숫자로 파싱되는지가 아니라 부호/통화기호로
    시작해 숫자로 이어지는지만 본다("15.4일"/"±0.79σ"처럼 단위가 붙어도 숫자
    컬럼으로 인식). "삼성전자 (005930.KS)"처럼 숫자가 안쪽에 섞인 텍스트를
    전부 걷어내 파싱하면 오탐(false positive)이 나므로 그 방식은 쓰지 않는다.
    "2026-07-06" 같은 날짜는 숫자로 시작해도 예외로 텍스트 취급한다."""
    t = text.strip()
    if _DATE_TEXT_RE.match(t):
        return False
    return bool(_NUMERIC_LEADING_RE.match(t))


def _is_numeric_col(values: list[str]) -> bool:
    """열 하나가 '숫자 컬럼'인지 — 값 대다수(80%+)가 부호/통화기호/쉼표 등을
    걷어내고 숫자로 파싱되면 숫자 컬럼(우측 정렬 + tabular-nums)으로 본다."""
    non_empty = [v for v in values if v not in ("", "N/A", "—", "-")]
    if not non_empty:
        return False
    hits = sum(1 for v in non_empty if _is_numeric_text(v))
    return hits / len(non_empty) >= 0.8


def _badge_html(raw) -> str:
    """판정/신호 셀을 pill 배지로 렌더. CLOSE/청산만 neutral(앰버) 대신
    SLATE로 구분하고, 그 외 극성 색상은 POLARITY_COLOR/POLARITY_BG 그대로 재사용."""
    text = _html.escape(str(raw))
    leading = _re.match(r"^[A-Za-z_]+", str(raw).strip())
    token = leading.group(0).upper() if leading else ""
    if token == "CLOSE":
        color, bg = SLATE_COLOR, SLATE_BG
    else:
        polarity = _cell_polarity(raw)
        color = POLARITY_COLOR.get(polarity, _FALLBACK_COLOR)
        bg = POLARITY_BG.get(polarity, _FALLBACK_BG)
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'background:{bg};color:{color};font-weight:600;font-size:12px;'
        f'white-space:nowrap">{text}</span>'
    )


def render_clean_table(
    df,
    judgment_col: str | list[str] | None = None,
    label_col: str | list[str] | None = None,
    best_col: str | None = None,
    best_mode: str = "min",
) -> None:
    """세로 격자선 없는 공용 표 (Claude Design 확정 스펙 — 2단계a: best값강조만).

    - 세로 격자선 없음, border-collapse. thead 배경 #f7f8fa, 행 구분은
      border-bottom만.
    - 숫자로 인식되는 컬럼은 우측 정렬 + tabular-nums. 그 외 텍스트 컬럼은
      좌측 정렬(label_col로 지정한 컬럼만 font-weight:600 굵게 — 종목명·
      패턴명 등 핵심 식별값).
    - judgment_col: pill 배지로 렌더(_badge_html, POLARITY_COLOR/BG 재사용
      + CLOSE는 SLATE로 구분).
    - best_col + best_mode("min"|"max"): 후보 중 최적값 1개 셀을 굵은 초록으로 강조.

    주의: highlight_row_when(행 전체 하이라이트)은 세그폴트 인시던트
    (2026-07-13) 원인 격리를 위해 이번 단계에서는 제외 — best_col만 먼저
    배포 확인 후 별도 단계에서 재도입 예정.
    """
    judgment_cols = {judgment_col} if isinstance(judgment_col, str) else set(judgment_col or [])
    label_cols = {label_col} if isinstance(label_col, str) else set(label_col or [])
    cols = list(df.columns)

    numeric_cols = {
        c for c in cols
        if c not in judgment_cols and _is_numeric_col([str(v) for v in df[c]])
    }

    best_idx = None
    if best_col is not None and best_col in cols:
        parsed = []
        for i, v in enumerate(df[best_col]):
            cleaned = _re.sub(r"[^0-9+\-.]", "", str(v))
            try:
                parsed.append((float(cleaned), i))
            except ValueError:
                continue
        if parsed:
            best_idx = (min if best_mode == "min" else max)(parsed, key=lambda t: t[0])[1]

    def _col_align(c: str) -> str:
        return "right" if (c in numeric_cols or c in judgment_cols) else "left"

    header_cells = "".join(
        f'<th style="text-align:{_col_align(c)};padding:6px 10px;'
        f'font-size:11px;color:{SLATE_COLOR};font-weight:600;white-space:nowrap;'
        f'background:#f7f8fa;border-bottom:1px solid rgba(100,116,139,0.25)">'
        f'{_html.escape(str(c))}</th>'
        for c in cols
    )

    body_rows = []
    for row_i, (_, row) in enumerate(df.iterrows()):
        cells = []
        for c in cols:
            raw = row[c]
            align = _col_align(c)
            if c in judgment_cols:
                text = _badge_html(raw)
            else:
                text = _html.escape("" if raw is None else str(raw))
                if row_i == best_idx and c == best_col:
                    text = f'<span style="color:{POLARITY_COLOR["bullish"]};font-weight:700">{text}</span>'
                elif c in label_cols:
                    text = f'<span style="font-weight:600">{text}</span>'
            numeric_style = "font-variant-numeric:tabular-nums;" if c in numeric_cols else ""
            cells.append(
                f'<td style="text-align:{align};padding:6px 10px;font-size:13px;'
                f'{numeric_style}border-bottom:1px solid rgba(100,116,139,0.12)">{text}</td>'
            )
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
