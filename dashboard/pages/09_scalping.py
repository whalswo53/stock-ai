"""
단타(당일~1주일) 전용 분석 페이지.
장기 분석과 달리 빠른 지표 파라미터 사용:
  RSI-9 · MACD 5/13/5 · BB-10 · MA 5/10/20
  RSI 과매도/과매수 기준: 40/60 (장기 30/70 대신)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config.sources import TICKER_KR_NAME
from data.collectors.market_sentiment import MarketSentimentCollector, prompt_snippet
from data.collectors.price_collector import PriceCollector
from utils.ticker_utils import resolve_ticker as _resolve_base, detect_market, is_kr, fmt_price
from utils.clipboard import copy_button
from utils.search_widget import ticker_search_widget

# ── Scalping parameters ────────────────────────────────────────────────────────
ST_MA_WINDOWS  = [5, 10, 20]          # 단기 이동평균
ST_RSI_PERIOD  = 9                     # RSI 기간 (장기 14 대신)
ST_MACD_FAST   = 5                     # MACD 빠른선 (장기 12 대신)
ST_MACD_SLOW   = 13                    # MACD 느린선 (장기 26 대신)
ST_MACD_SIG    = 5                     # MACD 시그널 (장기 9 대신)
ST_BB_PERIOD   = 10                    # 볼린저밴드 기간 (장기 20 대신)
ST_BB_STD      = 2.0
VOL_AVG_WIN    = 20                    # 거래량 평균 윈도우
VOL_SPIKE_MULT = 2.0                   # 급증 기준 배수
RSI_OB         = 60                    # 과매수 기준 (장기 70 대신)
RSI_OS         = 40                    # 과매도 기준 (장기 30 대신)

PERIOD_OPTIONS = {"1개월": "1mo", "2개월": "2mo", "3개월": "3mo"}

UP   = "#26a69a"
DOWN = "#ef5350"
WAIT = "#9E9E9E"

MA_COLORS = {5: "#FF9800", 10: "#2196F3", 20: "#9C27B0"}


def _resolve(raw: str) -> str:
    return _resolve_base(raw)


# ── Scalping indicator computation ────────────────────────────────────────────

def compute_scalping(df: pd.DataFrame) -> pd.DataFrame:
    """단타 전용 지표 계산. 원본과 독립적인 파라미터 세트 사용."""
    out = df.copy()
    close  = out["Close"]
    volume = out["Volume"]

    # MA 5/10/20
    for w in ST_MA_WINDOWS:
        out[f"MA{w}"] = close.rolling(w, min_periods=w).mean()

    # RSI-9
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=ST_RSI_PERIOD - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=ST_RSI_PERIOD - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))

    # MACD 5/13/5
    ema_f = close.ewm(span=ST_MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=ST_MACD_SLOW, adjust=False).mean()
    out["MACD"]        = ema_f - ema_s
    out["MACD_Signal"] = out["MACD"].ewm(span=ST_MACD_SIG, adjust=False).mean()
    out["MACD_Hist"]   = out["MACD"] - out["MACD_Signal"]

    # BB-10
    bb_mid = close.rolling(ST_BB_PERIOD, min_periods=ST_BB_PERIOD).mean()
    bb_std = close.rolling(ST_BB_PERIOD, min_periods=ST_BB_PERIOD).std()
    out["BB_Upper"] = bb_mid + ST_BB_STD * bb_std
    out["BB_Mid"]   = bb_mid
    out["BB_Lower"] = bb_mid - ST_BB_STD * bb_std

    # 거래량 급증
    vol_avg         = volume.rolling(VOL_AVG_WIN, min_periods=1).mean()
    out["Vol_Avg"]   = vol_avg
    out["Vol_Ratio"] = volume / vol_avg.replace(0, np.nan)
    out["Vol_Spike"] = volume > VOL_SPIKE_MULT * vol_avg

    # OBV
    direction  = np.sign(close.diff()).fillna(0)
    out["OBV"] = (direction * volume).cumsum()

    return out


# ── Signal detection ──────────────────────────────────────────────────────────

def detect_signals(df: pd.DataFrame) -> dict:
    """
    단타 관점 신호 감지.
    기준: RSI 40/60 · BB 터치 · MACD 크로스 · 거래량 급증
    """
    if len(df) < 2:
        return {"signals": [], "overall": "WAIT", "rsi": 50.0,
                "bb_pct": 0.5, "vol_spike": False, "vol_ratio": 1.0}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    def _f(row, key: str, default: float = 0.0) -> float:
        v = row.get(key, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    rsi       = _f(last, "RSI", 50.0)
    close     = _f(last, "Close")
    bb_upper  = _f(last, "BB_Upper", close * 1.02)
    bb_lower  = _f(last, "BB_Lower", close * 0.98)
    macd      = _f(last, "MACD")
    macd_sig  = _f(last, "MACD_Signal")
    p_macd    = _f(prev, "MACD")
    p_sig     = _f(prev, "MACD_Signal")
    vol_spike = bool(last.get("Vol_Spike", False))
    vol_ratio = _f(last, "Vol_Ratio", 1.0)

    bb_range = bb_upper - bb_lower
    bb_pct   = (close - bb_lower) / bb_range if bb_range > 0 else 0.5

    signals: list[dict] = []

    # ── RSI ───────────────────────────────────────────────────────────────────
    if rsi <= RSI_OS:
        signals.append({
            "name": "RSI 과매도", "direction": "BUY",
            "detail": f"RSI {rsi:.1f} ≤ {RSI_OS} — 단기 반등 가능성",
        })
    elif rsi >= RSI_OB:
        signals.append({
            "name": "RSI 과매수", "direction": "SELL",
            "detail": f"RSI {rsi:.1f} ≥ {RSI_OB} — 단기 조정 가능성",
        })

    # ── Bollinger Band ────────────────────────────────────────────────────────
    if bb_pct <= 0.05:
        signals.append({
            "name": "BB 하단 터치", "direction": "BUY",
            "detail": f"현재가({close:,.0f})가 BB 하단({bb_lower:,.0f}) 도달 — 즉각 반등 주시",
        })
    elif bb_pct >= 0.95:
        signals.append({
            "name": "BB 상단 터치", "direction": "SELL",
            "detail": f"현재가({close:,.0f})가 BB 상단({bb_upper:,.0f}) 도달 — 즉각 조정 주시",
        })

    # ── MACD 크로스 ───────────────────────────────────────────────────────────
    macd_valid = not any(np.isnan(v) for v in [macd, macd_sig, p_macd, p_sig])
    if macd_valid:
        if macd > macd_sig and p_macd <= p_sig:
            signals.append({
                "name": "MACD 골든크로스", "direction": "BUY",
                "detail": f"MACD({ST_MACD_FAST}/{ST_MACD_SLOW}/{ST_MACD_SIG}) 상향 돌파 — 단기 상승 모멘텀",
            })
        elif macd < macd_sig and p_macd >= p_sig:
            signals.append({
                "name": "MACD 데드크로스", "direction": "SELL",
                "detail": f"MACD({ST_MACD_FAST}/{ST_MACD_SLOW}/{ST_MACD_SIG}) 하향 돌파 — 단기 하락 모멘텀",
            })

    # ── 거래량 급증 ────────────────────────────────────────────────────────────
    if vol_spike:
        bullish = float(last["Close"]) >= float(last["Open"])
        signals.append({
            "name": "거래량 급증", "direction": "BUY" if bullish else "SELL",
            "detail": (
                f"거래량 평균 {vol_ratio:.1f}배 — "
                f"{'상승 추진력 강화' if bullish else '매도 압력 증가'}"
            ),
        })

    # ── 종합 판정 ─────────────────────────────────────────────────────────────
    buy_n  = sum(1 for s in signals if s["direction"] == "BUY")
    sell_n = sum(1 for s in signals if s["direction"] == "SELL")

    if buy_n >= 2 and buy_n > sell_n:
        overall = "BUY"
    elif sell_n >= 2 and sell_n > buy_n:
        overall = "SELL"
    elif buy_n == 1 and sell_n == 0:
        overall = "BUY"
    elif sell_n == 1 and buy_n == 0:
        overall = "SELL"
    else:
        overall = "WAIT"

    return {
        "signals":   signals,
        "overall":   overall,
        "rsi":       rsi,
        "bb_pct":    bb_pct,
        "bb_upper":  bb_upper,
        "bb_lower":  bb_lower,
        "vol_spike": vol_spike,
        "vol_ratio": vol_ratio,
        "macd":      macd,
        "macd_sig":  macd_sig,
        "close":     close,
    }


_SCALPING_GRADIENT = "linear-gradient(135deg,#E65100,#BF360C)"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ 단타 설정")

    _jump = st.session_state.pop("portfolio_jump_ticker", None)
    if _jump:
        st.session_state["_tsq_scalping"] = _jump
    raw_input = ticker_search_widget(
        key="scalping",
        label="종목 코드 또는 한글명",
        default="005930.KS",
    ) or "005930.KS"

    period_label = st.selectbox("조회 기간", list(PERIOD_OPTIONS.keys()), index=0)
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("지표 설정")
    show_ma  = st.checkbox("이동평균 MA 5/10/20", value=True)
    show_bb  = st.checkbox("볼린저밴드 (BB-10)",  value=True)
    show_vol_spike = st.checkbox("거래량 급증 하이라이트", value=True)

    st.divider()
    st.caption(
        f"**단타 전용 파라미터**\n"
        f"- RSI {ST_RSI_PERIOD}  (기준 {RSI_OS}/{RSI_OB})\n"
        f"- MACD {ST_MACD_FAST}/{ST_MACD_SLOW}/{ST_MACD_SIG}\n"
        f"- BB {ST_BB_PERIOD}일\n"
        f"- MA {'/'.join(str(w) for w in ST_MA_WINDOWS)}\n"
        f"- 거래량 급증: {VOL_SPIKE_MULT:.0f}배 기준"
    )


# ── Resolve ticker ─────────────────────────────────────────────────────────────
ticker = raw_input  # search_widget이 이미 resolve한 티커
market = detect_market(ticker)
kr     = is_kr(ticker)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_scalping(ticker: str, period: str) -> pd.DataFrame:
    df = PriceCollector().fetch(ticker, period=period)
    if df.empty:
        return df
    return compute_scalping(df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    return PriceCollector().get_info(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fg() -> str:
    try:
        return prompt_snippet(MarketSentimentCollector().fetch())
    except Exception:
        return ""


with st.spinner(f"'{ticker}' 단기 데이터 수집 중…"):
    df = load_scalping(ticker, period)
    info = load_info(ticker)

if df.empty:
    st.error(f"'{ticker}' 데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
    st.stop()


# ── Detect signals ─────────────────────────────────────────────────────────────
sig = detect_signals(df)
last     = df.iloc[-1]
prev     = df.iloc[-2] if len(df) > 1 else last
close    = float(last["Close"])
vol      = float(last["Volume"])
vol_avg  = float(last.get("Vol_Avg", vol))
vol_ratio = sig["vol_ratio"]

company = TICKER_KR_NAME.get(ticker) or info.get("shortName") or info.get("longName") or ticker


# ── Page header ───────────────────────────────────────────────────────────────
st.title(f"⚡  {company} — 단타 분석")
st.caption(
    f"{ticker}  ·  {market}  ·  {period_label} 조회  ·  "
    f"RSI-{ST_RSI_PERIOD} / MACD {ST_MACD_FAST}/{ST_MACD_SLOW}/{ST_MACD_SIG} / BB-{ST_BB_PERIOD}"
)
st.warning(
    "⚠️ **이 페이지는 1주일 이내 단기 매매 전용입니다.**  \n"
    "RSI·MACD·볼린저밴드 파라미터가 단타 최적화 값으로 설정되어 있습니다.  \n"
    "중장기 투자 판단에는 **📈 장기/스윙 분석** 또는 **🎯 종합 분석** 페이지를 이용하세요.",
    icon=None,
)

# ── Top metrics ───────────────────────────────────────────────────────────────
change_abs = close - float(prev["Close"])
change_pct = (change_abs / float(prev["Close"])) * 100 if prev["Close"] else 0.0
period_high = float(df["High"].max())
period_low  = float(df["Low"].min())

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("현재가",    fmt_price(close, ticker), f"{change_pct:+.2f}%")
m2.metric("전일 대비", fmt_price(change_abs, ticker), delta_color="normal")
m3.metric("거래량",    f"{vol:,.0f}", f"{vol_ratio:.1f}x 평균")
m4.metric(f"{period_label} 고가", fmt_price(period_high, ticker))
m5.metric(f"{period_label} 저가", fmt_price(period_low,  ticker))


# ── Signal summary ─────────────────────────────────────────────────────────────
st.divider()

overall = sig["overall"]
overall_color = {"BUY": UP, "SELL": DOWN, "WAIT": WAIT}[overall]
overall_icon  = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏸"}[overall]
overall_label = {"BUY": "매수 신호", "SELL": "매도 신호", "WAIT": "관망"}[overall]

# Volume spike alert
if sig["vol_spike"] and show_vol_spike:
    st.warning(
        f"🔥 **거래량 급증 감지!** — 오늘 거래량이 20일 평균의 **{vol_ratio:.1f}배**입니다.  \n"
        f"{'가격 상승 중 → 단기 상승 모멘텀 강화' if close >= float(last['Open']) else '가격 하락 중 → 단기 매도 압력 증가'}"
    )

# Overall signal box
col_main, col_sigs = st.columns([1, 2])

with col_main:
    st.markdown(
        f"""<div style="
            background:rgba({','.join(str(int(overall_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.18);
            border:2px solid {overall_color};border-radius:12px;
            padding:20px;text-align:center">
          <div style="font-size:36px;margin-bottom:4px">{overall_icon}</div>
          <div style="font-size:22px;font-weight:800;color:{overall_color}">{overall_label}</div>
          <div style="font-size:12px;color:#aaa;margin-top:6px">종합 신호 (단타 기준)</div>
        </div>""",
        unsafe_allow_html=True,
    )

with col_sigs:
    if sig["signals"]:
        st.markdown("**감지된 신호**")
        for s in sig["signals"]:
            ic = "🟢" if s["direction"] == "BUY" else "🔴"
            c  = UP   if s["direction"] == "BUY" else DOWN
            st.markdown(
                f'<div style="margin:4px 0;padding:6px 10px;border-left:3px solid {c};'
                f'background:rgba(0,0,0,0.2);border-radius:0 6px 6px 0">'
                f'{ic} <strong>{s["name"]}</strong><br>'
                f'<span style="font-size:12px;color:#bbb">{s["detail"]}</span></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown("**감지된 신호 없음**")
        st.caption("현재 뚜렷한 단기 신호가 없습니다. 추가 확인 후 진입하세요.")

    # RSI + BB%B quick read
    rsi_val = sig["rsi"]
    bb_pct  = sig["bb_pct"]
    rsi_c   = DOWN if rsi_val >= RSI_OB else (UP if rsi_val <= RSI_OS else "#888")
    bb_c    = DOWN if bb_pct >= 0.9 else (UP if bb_pct <= 0.1 else "#888")

    r1, r2 = st.columns(2)
    r1.metric(f"RSI({ST_RSI_PERIOD})", f"{rsi_val:.1f}",
              "과매수" if rsi_val >= RSI_OB else ("과매도" if rsi_val <= RSI_OS else "중립"))
    r2.metric("BB 위치(%B)", f"{bb_pct:.2f}",
              "상단 근접" if bb_pct >= 0.85 else ("하단 근접" if bb_pct <= 0.15 else "중앙권"))


# ── Scalping chart ─────────────────────────────────────────────────────────────
st.divider()

def build_scalping_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # ── Row 1: Candlestick ────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name="가격",
            increasing=dict(line=dict(color=UP),   fillcolor=UP),
            decreasing=dict(line=dict(color=DOWN),  fillcolor=DOWN),
        ),
        row=1, col=1,
    )

    # MA overlays
    if show_ma:
        for w in ST_MA_WINDOWS:
            col = f"MA{w}"
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df[col], name=f"MA{w}",
                               line=dict(color=MA_COLORS[w], width=1.4)),
                    row=1, col=1,
                )

    # Bollinger Bands
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Upper"], name="BB 상단",
                       line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Lower"], name="BB 하단",
                       line=dict(color="rgba(255,193,7,0.7)", width=1, dash="dash"),
                       fill="tonexty", fillcolor="rgba(255,193,7,0.05)"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_Mid"], name="BB 중간",
                       line=dict(color="rgba(255,193,7,0.35)", width=1, dash="dot"),
                       showlegend=False),
            row=1, col=1,
        )

    # ── Row 2: Volume (spike highlighted) ─────────────────────────────────────
    if show_vol_spike and "Vol_Spike" in df.columns:
        vol_colors = [
            "#FF6F00" if spike else (UP if c >= o else DOWN)
            for spike, c, o in zip(df["Vol_Spike"], df["Close"], df["Open"])
        ]
        spike_legend_added = False
        for i, (idx, row) in enumerate(df.iterrows()):
            is_spike = bool(row.get("Vol_Spike", False))
            color = "#FF6F00" if is_spike else (UP if row["Close"] >= row["Open"] else DOWN)
            fig.add_trace(
                go.Bar(x=[idx], y=[row["Volume"]], name="거래량 급증" if is_spike else "거래량",
                       marker_color=color, opacity=0.7,
                       showlegend=(is_spike and not spike_legend_added),
                       legendgroup="vol"),
                row=2, col=1,
            )
            if is_spike:
                spike_legend_added = True
    else:
        vol_colors = [UP if c >= o else DOWN for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="거래량",
                   marker_color=vol_colors, opacity=0.65, showlegend=False),
            row=2, col=1,
        )

    # Vol average line
    if "Vol_Avg" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["Vol_Avg"], name="평균 거래량",
                       line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
                       showlegend=False),
            row=2, col=1,
        )

    # ── Row 3: MACD 5/13/5 ───────────────────────────────────────────────────
    if "MACD_Hist" in df.columns:
        hist       = df["MACD_Hist"].fillna(0)
        hist_colors = [UP if v >= 0 else DOWN for v in hist]
        fig.add_trace(
            go.Bar(x=df.index, y=hist, name="MACD Hist",
                   marker_color=hist_colors, opacity=0.7, showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], name=f"MACD({ST_MACD_FAST}/{ST_MACD_SLOW})",
                       line=dict(color="#2196F3", width=1.5)),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD_Signal"], name=f"Signal({ST_MACD_SIG})",
                       line=dict(color="#FF9800", width=1.5)),
            row=3, col=1,
        )

    # ── Row 4: RSI-9 (with 40/60 bands) ──────────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name=f"RSI({ST_RSI_PERIOD})",
                       line=dict(color="#9C27B0", width=1.5),
                       fill="tozeroy", fillcolor="rgba(156,39,176,0.05)"),
            row=4, col=1,
        )
        for level, color in [
            (RSI_OB, "rgba(239,83,80,0.5)"),
            (RSI_OS, "rgba(38,166,154,0.5)"),
        ]:
            fig.add_hline(y=level, line_dash="dash", line_color=color,
                          line_width=1, row=4, col=1)
        fig.add_hline(y=50, line_dash="dot",
                      line_color="rgba(128,128,128,0.25)", line_width=1, row=4, col=1)

    # ── Layout ────────────────────────────────────────────────────────────────
    tick_fmt = ",.0f" if kr else ".2f"
    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=12),
    )
    fig.update_yaxes(title_text="가격",   tickformat=tick_fmt, row=1, col=1)
    fig.update_yaxes(title_text="거래량", showgrid=False,      row=2, col=1)
    fig.update_yaxes(title_text="MACD",  row=3, col=1)
    fig.update_yaxes(title_text="RSI",   range=[0, 100],       row=4, col=1)

    return fig


st.plotly_chart(build_scalping_chart(df), width="stretch")


# ── Entry / Target / Stop-loss Calculator ─────────────────────────────────────
st.divider()
st.subheader("⚡ 진입가 · 목표가 · 손절가 계산기")

col_entry, col_table = st.columns([1, 2])

with col_entry:
    entry_price = st.number_input(
        "진입가 (현재가 기준)",
        value=float(round(close, 0) if kr else round(close, 2)),
        min_value=0.0,
        step=100.0 if kr else 0.01,
        format="%.0f" if kr else "%.2f",
    )
    st.caption(
        f"현재가: {fmt_price(close, ticker)}\n\n"
        "진입가를 수정하면 목표가·손절가가 자동 재계산됩니다."
    )

with col_table:
    if entry_price > 0:
        levels = [2, 3, 5]
        rows = []
        for pct in levels:
            tgt  = entry_price * (1 + pct / 100)
            stop = entry_price * (1 - pct / 100)
            rows.append({
                "비율":       f"±{pct}%",
                "목표가 (익절)": fmt_price(tgt,  ticker),
                "손절가":       fmt_price(stop, ticker),
            })
        # Also show asymmetric R:R scenarios (common scalping setups)
        rr_rows = []
        for stop_pct, tgt_pct in [(2, 3), (2, 5), (3, 5)]:
            tgt  = entry_price * (1 + tgt_pct  / 100)
            stop = entry_price * (1 - stop_pct / 100)
            rr   = tgt_pct / stop_pct
            rr_rows.append({
                "손절":       f"-{stop_pct}%",
                "익절":       f"+{tgt_pct}%",
                "R:R 비율":  f"{rr:.1f}:1",
                "손절가":    fmt_price(stop, ticker),
                "목표가":    fmt_price(tgt,  ticker),
            })

        st.caption("**대칭 계산**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        st.caption("**비대칭 R:R 시나리오 (권장)**")
        st.dataframe(pd.DataFrame(rr_rows), hide_index=True, width="stretch")

        # Quick guidance
        stop2 = entry_price * 0.98
        tgt3  = entry_price * 1.03
        st.info(
            f"💡 **단타 기본 설정 예시**  \n"
            f"손절: {fmt_price(stop2, ticker)} (-2%)  ·  "
            f"목표: {fmt_price(tgt3,  ticker)} (+3%)  →  **R:R = 1.5:1**\n\n"
            "1주일 이내 청산 기준. 손절선 이탈 시 즉시 매도."
        )


# ── AI Prompt (scalping-focused) ───────────────────────────────────────────────
st.divider()
st.subheader("🤖 단타 AI 분석 프롬프트")

def build_scalping_prompt(df: pd.DataFrame, sig: dict) -> str:
    last   = df.iloc[-1]

    def _f(key: str, fmt: str = ".2f") -> str:
        v = last.get(key)
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "N/A"

    rsi_val  = sig["rsi"]
    bb_pct   = sig["bb_pct"]
    vol_r    = sig["vol_ratio"]
    overall  = sig["overall"]

    rsi_label = (
        f"과매도 ({RSI_OS} 이하) — 단기 반등 주시"   if rsi_val <= RSI_OS else
        f"과매수 ({RSI_OB} 이상) — 단기 조정 주시"  if rsi_val >= RSI_OB else
        "중립 구간"
    )
    bb_label = (
        "BB 하단 터치 — 즉각 반등 가능"  if bb_pct <= 0.1 else
        "BB 상단 터치 — 즉각 조정 가능"  if bb_pct >= 0.9 else
        f"BB 중간 위치 (%B {bb_pct:.2f})"
    )
    cross_label = (
        "골든크로스 — 단기 상승 모멘텀" if sig["macd"] > sig["macd_sig"] else
        "데드크로스 — 단기 하락 모멘텀"
    )
    vol_label = f"{'🔥 급증 ' if sig['vol_spike'] else ''}{vol_r:.1f}배 (20일 평균 대비)"

    signal_list = "\n".join(
        f"- [{s['direction']}] {s['name']}: {s['detail']}"
        for s in sig["signals"]
    ) or "- 현재 뚜렷한 신호 없음"

    cols = [c for c in ["Close", "Volume", "RSI", "MACD"] if c in df.columns]
    recent_table = df.tail(5)[cols].to_string(float_format=lambda x: f"{x:,.2f}")

    fg_text = load_fg()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    entry_ref = round(close, 0) if kr else round(close, 2)
    stop_ref  = entry_ref * 0.98
    tgt_ref   = entry_ref * 1.03
    fmt = ",.0f" if kr else ",.2f"
    sym = "₩" if kr else "$"

    prompt = f"""# 단타 매매 분석 요청 — {ticker} ({company})

당신은 단타 매매 전문 트레이더입니다. 아래 단기 기술적 데이터를 바탕으로
**1주일 이내 청산** 관점으로만 분석해주세요.
펀더멘털(PER·PBR·기업가치 등) 분석은 제외합니다.

**분석 시각:** {now}
{f"**시장 분위기:** {fg_text}" if fg_text else ""}
**현재 종합 신호:** {overall}

---

## 단기 기술적 지표 (단타 전용 파라미터)

| 지표 | 값 | 해석 |
|------|-----|------|
| 현재가 | {_f('Close', fmt)} | — |
| RSI({ST_RSI_PERIOD}) | {rsi_val:.1f} | {rsi_label} |
| MACD({ST_MACD_FAST}/{ST_MACD_SLOW}/{ST_MACD_SIG}) | {_f('MACD', '.4f')} / Signal: {_f('MACD_Signal', '.4f')} | {cross_label} |
| MA5 / MA10 / MA20 | {_f('MA5', fmt)} / {_f('MA10', fmt)} / {_f('MA20', fmt)} | — |
| 볼린저밴드({ST_BB_PERIOD}) | 상단: {_f('BB_Upper', fmt)} / 하단: {_f('BB_Lower', fmt)} | {bb_label} |
| 거래량 | {_f('Volume', ',.0f')} | {vol_label} |

## 최근 5거래일 데이터

```
{recent_table}
```

## 현재 감지된 단기 신호

{signal_list}

## 참고 계산 (현재가 기준)

- 손절 -2%: {sym}{stop_ref:{fmt}}
- 목표 +3%: {sym}{tgt_ref:{fmt}}  (R:R ≈ 1.5:1)

---

## 분석 요청 (단타 관점 한정)

1. **1주일 내 방향성** — 상승·하락·횡보 중 어느 시나리오가 가장 유력한지
2. **진입 타이밍** — 지금 바로 진입 가능한지, 아니면 어떤 조건을 기다려야 하는지 (구체적 조건 명시)
3. **목표가 / 손절가** — 현재가 기준 구체적 숫자로 제시 (% + 절대가)
4. **예상 보유 기간** — 몇 시간 ~ 며칠 예상하는지
5. **최대 리스크** — 이 단기 매매에서 가장 주의해야 할 위험 한 가지

**주의:** 장기 투자 의견, 기업 실적, 밸류에이션 언급 금지.
**주의:** 모든 신호를 "1주일 이내 청산" 기준으로 해석.

**⚠️ 응답 마지막에 반드시 아래 JSON 블록을 포함해주세요:**

```json
{{
  "signal": "BUY 또는 SELL 또는 WAIT",
  "entry": 진입가 숫자,
  "target": 목표가 숫자,
  "stop_loss": 손절가 숫자,
  "hold_days": 예상 보유 일수 숫자,
  "confidence": 0.0~1.0,
  "reasons": ["이유1", "이유2", "이유3"]
}}
```"""

    return prompt


prompt = build_scalping_prompt(df, sig)

with st.expander("📄 프롬프트 내용 미리보기", expanded=False):
    st.code(prompt, language="markdown")

st.markdown("**단타 분석 프롬프트가 준비됐습니다. 복사 후 Claude.ai에 붙여넣으세요:**")
copy_button(prompt, "📋 단타 분석 프롬프트 복사", gradient=_SCALPING_GRADIENT)
st.caption("버튼 클릭 후 Claude.ai에 붙여넣으세요 (Ctrl+V)")
