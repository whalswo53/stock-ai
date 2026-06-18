"""
CNN Fear & Greed Index 전용 페이지.
현재 지수 + 히스토리(전일/1주/1개월/1년) + VIX 보조 지표
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
import plotly.graph_objects as go

from data.collectors.market_sentiment import MarketSentimentCollector, score_to_color

# ── Label helpers ─────────────────────────────────────────────────────────────
_RATING_TO_KR: dict[str, str] = {
    "extreme fear":  "극도의 공포",
    "fear":          "공포",
    "neutral":       "중립",
    "greed":         "탐욕",
    "extreme greed": "극도의 탐욕",
}

def _label_from_score(score: float) -> str:
    s = int(round(score))
    if s <= 20:  return "극도의 공포"
    if s <= 40:  return "공포"
    if s <= 60:  return "중립"
    if s <= 80:  return "탐욕"
    return "극도의 탐욕"

def _label_from_rating(rating: str) -> str:
    return _RATING_TO_KR.get(rating.lower(), _label_from_score(50))


# ── Data fetch (1h cache) ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_full() -> dict:
    """CNN API에서 현재 + 히스토리 데이터를 가져온다."""
    import requests
    import yfinance as yf

    out: dict = {
        "score": 50, "label": "중립", "source": "데이터 없음",
        "last_update": "", "vix": -1.0,
        "prev_close": None, "prev_1w": None, "prev_1m": None, "prev_1y": None,
        "vix_1w": None, "vix_1m": None,
    }

    # ── CNN F&G (직접 JSON 엔드포인트) ────────────────────────────────────
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        fg = resp.json().get("fear_and_greed", {})
        score = max(0, min(100, int(round(float(fg.get("score", 50))))))
        out["score"]       = score
        out["label"]       = _label_from_rating(fg.get("rating", "neutral"))
        out["source"]      = "CNN Fear & Greed Index"
        raw_ts             = fg.get("timestamp", "")
        out["last_update"] = raw_ts[:16].replace("T", " ") if raw_ts else ""
        out["prev_close"]  = fg.get("previous_close")
        out["prev_1w"]     = fg.get("previous_1_week")
        out["prev_1m"]     = fg.get("previous_1_month")
        out["prev_1y"]     = fg.get("previous_1_year")
    except Exception:
        # fallback to fear_and_greed package
        try:
            import fear_and_greed
            r = fear_and_greed.get()
            out["score"]  = max(0, min(100, int(round(r.value))))
            out["label"]  = _RATING_TO_KR.get(r.description, _label_from_score(out["score"]))
            out["source"] = "CNN Fear & Greed Index (패키지)"
        except Exception:
            pass

    # ── VIX 현재 + 히스토리 ────────────────────────────────────────────────
    try:
        vix_hist = yf.Ticker("^VIX").history(period="14mo")
        if not vix_hist.empty:
            out["vix"]    = round(float(vix_hist["Close"].iloc[-1]), 2)
            if len(vix_hist) >= 6:
                out["vix_1w"] = round(float(vix_hist["Close"].iloc[-6]), 2)
            if len(vix_hist) >= 22:
                out["vix_1m"] = round(float(vix_hist["Close"].iloc[-22]), 2)
    except Exception:
        pass

    return out


# ── Gauge chart ───────────────────────────────────────────────────────────────
def _build_gauge(score: int, label: str) -> go.Figure:
    color = score_to_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{label}</b>", "font": {"size": 22, "color": color}},
        number={"font": {"size": 90, "color": "white"}, "valueformat": "d"},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0", "20", "40", "60", "80", "100"],
                "tickcolor": "#666",
                "tickwidth": 1,
                "tickfont": {"color": "#888", "size": 11},
            },
            "bar": {"color": color, "thickness": 0.03},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  20], "color": "#c62828"},
                {"range": [20, 40], "color": "#e64a19"},
                {"range": [40, 60], "color": "#f9a825"},
                {"range": [60, 80], "color": "#558b2f"},
                {"range": [80, 100], "color": "#00695c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 6},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=60, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ── History card HTML ─────────────────────────────────────────────────────────
def _history_html(rows: list[tuple[str, float | None]]) -> str:
    parts = [
        '<div style="background:rgba(255,255,255,0.04);border-radius:10px;'
        'padding:14px 16px;font-family:sans-serif">'
        '<div style="font-size:12px;color:#888;font-weight:600;margin-bottom:10px;'
        'text-transform:uppercase;letter-spacing:0.05em">히스토리</div>'
    ]
    for label, score_val in rows:
        if score_val is None:
            parts.append(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.08)">'
                f'<span style="color:#888;font-size:13px">{label}</span>'
                f'<span style="color:#444;font-size:13px">—</span></div>'
            )
            continue
        sc = int(round(score_val))
        c  = score_to_color(sc)
        lbl = _label_from_score(sc)
        bar_pct = sc  # 0..100
        parts.append(
            f'<div style="padding:9px 0;border-bottom:1px solid rgba(255,255,255,0.08)">'
            f'  <div style="display:flex;justify-content:space-between;align-items:baseline">'
            f'    <span style="color:#aaa;font-size:13px">{label}</span>'
            f'    <span>'
            f'      <span style="font-size:20px;font-weight:700;color:{c}">{sc}</span>'
            f'      <span style="font-size:11px;color:{c};margin-left:5px">{lbl}</span>'
            f'    </span>'
            f'  </div>'
            f'  <div style="height:4px;border-radius:2px;margin-top:5px;'
            f'background:linear-gradient(to right,#c62828,#e64a19,#f9a825,#558b2f,#00695c)">'
            f'    <div style="position:relative;left:{bar_pct}%;width:3px;height:4px;'
            f'background:white;border-radius:2px;transform:translateX(-50%)"></div>'
            f'  </div>'
            f'</div>'
        )
    parts.append("</div>")
    return "".join(parts)


# ── Zone legend HTML ──────────────────────────────────────────────────────────
_ZONES_HTML = """
<div style="background:rgba(255,255,255,0.04);border-radius:10px;
            padding:14px 16px;margin-top:12px">
  <div style="font-size:12px;color:#888;font-weight:600;margin-bottom:10px;
              text-transform:uppercase;letter-spacing:0.05em">구간 설명</div>
  <div style="display:flex;flex-direction:column;gap:6px;font-size:13px">
    <div><span style="color:#c62828;font-weight:600">0–20</span>
         <span style="color:#aaa;margin-left:8px">극도의 공포 — 투자자들이 패닉 상태</span></div>
    <div><span style="color:#e64a19;font-weight:600">21–40</span>
         <span style="color:#aaa;margin-left:8px">공포 — 시장 불안감 우세</span></div>
    <div><span style="color:#f9a825;font-weight:600">41–60</span>
         <span style="color:#aaa;margin-left:8px">중립 — 방향성 불확실</span></div>
    <div><span style="color:#558b2f;font-weight:600">61–80</span>
         <span style="color:#aaa;margin-left:8px">탐욕 — 상승 기대감 우세</span></div>
    <div><span style="color:#00695c;font-weight:600">81–100</span>
         <span style="color:#aaa;margin-left:8px">극도의 탐욕 — 과열 주의</span></div>
  </div>
</div>
"""

# ── VIX section HTML ──────────────────────────────────────────────────────────
def _vix_html(vix: float, vix_1w: float | None, vix_1m: float | None) -> str:
    def _trend(curr: float, prev: float | None) -> str:
        if prev is None:
            return ""
        delta = curr - prev
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "—")
        col   = "#ef5350" if delta > 0 else ("#26a69a" if delta < 0 else "#888")
        return (
            f'<span style="color:{col};font-size:12px;margin-left:6px">'
            f'{arrow} {abs(delta):.1f}</span>'
        )

    rows = []
    rows.append(f'<span style="font-size:24px;font-weight:700;color:white">{vix:.1f}</span>'
                f'{_trend(vix, vix_1w)}')

    details = []
    if vix_1w is not None:
        details.append(f'<span style="color:#888">1주 전: <b style="color:#ccc">{vix_1w:.1f}</b></span>')
    if vix_1m is not None:
        details.append(f'<span style="color:#888">1달 전: <b style="color:#ccc">{vix_1m:.1f}</b></span>')

    interpretation = (
        "시장 변동성 낮음 (안정적)" if vix < 15 else
        "시장 변동성 보통"         if vix < 20 else
        "시장 변동성 높음 (주의)"   if vix < 30 else
        "⚠️ 시장 변동성 매우 높음 (경계)"
    )

    return (
        f'<div style="background:rgba(255,255,255,0.04);border-radius:10px;padding:14px 16px">'
        f'  <div style="font-size:12px;color:#888;font-weight:600;margin-bottom:8px;'
        f'text-transform:uppercase;letter-spacing:0.05em">VIX — 공포 지수 (변동성)</div>'
        f'  <div style="margin-bottom:6px">{"".join(rows)}</div>'
        f'  <div style="display:flex;gap:16px;margin-bottom:8px">{"  ".join(details)}</div>'
        f'  <div style="font-size:12px;color:#aaa;border-left:3px solid #444;padding-left:8px">'
        f'{interpretation}</div>'
        f'</div>'
    )


# ── Page ──────────────────────────────────────────────────────────────────────
st.title("😨  공포·탐욕 지수")
st.caption("CNN Fear & Greed Index — 시장 전반의 투자 심리를 0~100으로 수치화")

with st.spinner("CNN 데이터 불러오는 중…"):
    d = _fetch_full()

score  = d["score"]
label  = d["label"]
source = d["source"]
upd    = d.get("last_update", "")
vix    = d["vix"]

# ── 메인 레이아웃: 게이지(왼) + 히스토리(오른) ───────────────────────────────
col_gauge, col_history = st.columns([3, 2])

with col_gauge:
    st.plotly_chart(_build_gauge(score, label), width="stretch")
    if upd:
        st.caption(f"최종 업데이트: {upd} UTC  ·  출처: {source}")
    else:
        st.caption(f"출처: {source}")

with col_history:
    st.markdown("")
    history_rows = [
        ("전일 (Previous Close)", d.get("prev_close")),
        ("1주 전 (1 Week Ago)",   d.get("prev_1w")),
        ("1개월 전 (1 Month Ago)", d.get("prev_1m")),
        ("1년 전 (1 Year Ago)",    d.get("prev_1y")),
    ]
    st.markdown(_history_html(history_rows), unsafe_allow_html=True)

st.divider()

# ── 아래 행: 구간 설명 + VIX ─────────────────────────────────────────────────
col_legend, col_vix = st.columns([3, 2])

with col_legend:
    st.markdown(_ZONES_HTML, unsafe_allow_html=True)

with col_vix:
    if vix >= 0:
        st.markdown(
            _vix_html(vix, d.get("vix_1w"), d.get("vix_1m")),
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:11px;color:#555;margin-top:8px">'
            'VIX: CBOE 변동성 지수 (S&P500 옵션 시장 기반).<br>'
            '20 이하 = 안정, 20~30 = 주의, 30 이상 = 경계</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("VIX 데이터를 불러올 수 없습니다.")

st.divider()
st.info(
    "💡 **활용 팁**  \n"
    "- **극도의 공포(0~20)**: 역발상 매수 기회일 수 있음. 워런 버핏 전략  \n"
    "- **극도의 탐욕(80~100)**: 시장 과열 가능성. 분할 매도 고려  \n"
    "- **중립 구간(40~60)**: 섹터별 개별 종목 분석이 더 중요  \n"
    "- F&G는 단기 심리 지표이므로 단독 사용보다 기술적 분석과 함께 참고"
)
