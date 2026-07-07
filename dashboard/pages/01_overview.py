import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.collectors.price_collector import PriceCollector
from analysis.technical.indicators import TechnicalIndicators
from config.sources import TICKER_KR_NAME
from utils.ticker_utils import detect_market, is_kr
from utils.search_widget import ticker_search_widget

# ── Color palette ─────────────────────────────────────────────────────────────
UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"
MA_COLORS = {5: "#FF9800", 20: "#2196F3", 60: "#9C27B0", 120: "#F44336"}

PERIOD_OPTIONS = {"3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 설정")

    # Accept ticker pre-fill from portfolio page jump
    _jump = st.session_state.pop("portfolio_jump_ticker", None)
    if _jump:
        st.session_state["_tsq_overview"] = _jump
    ticker = ticker_search_widget(
        key="overview",
        label="종목 코드 또는 한글명",
        default="005930.KS",
    ) or "005930.KS"

    period_label = st.selectbox("기간", list(PERIOD_OPTIONS.keys()), index=2)
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("지표 설정")

    show_ma = st.checkbox("이동평균선 (MA)", value=True)
    ma_windows: list[int] = []
    if show_ma:
        ma_windows = st.multiselect(
            "표시할 MA 기간",
            options=[5, 20, 60, 120],
            default=[20, 60],
            format_func=lambda x: f"MA{x}",
        )

    show_bb = st.checkbox("볼린저밴드 (BB)", value=False)

    st.divider()
    st.caption("🔍 한글·영문 이름 또는 티커 직접 입력 후 목록에서 선택")


# ── Market auto-detected from ticker ─────────────────────────────────────────
market = detect_market(ticker)
kr = is_kr(ticker)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = PriceCollector().fetch(ticker, period=period)
    if df.empty:
        return df
    return TechnicalIndicators().compute(df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_info(ticker: str) -> dict:
    return PriceCollector().get_info(ticker)


@st.cache_data(ttl=60, show_spinner=False)
def load_quote(ticker: str):
    """미국 종목 실시간(Finnhub)/지연(yfinance) 시세. 60초 캐시."""
    from data.collectors.realtime_quote import get_quote
    return get_quote(ticker)


with st.spinner(f"'{ticker}' 데이터 불러오는 중…"):
    df = load_ohlcv(ticker, period)
    info = load_info(ticker)

if df.empty:
    st.error(f"'{ticker}' 데이터를 불러올 수 없습니다. 종목 코드를 확인해주세요.")
    st.stop()


# ── Header: company name + key metrics ───────────────────────────────────────
company_name = TICKER_KR_NAME.get(ticker) or info.get("shortName") or info.get("longName") or ticker
currency = "KRW" if kr else "USD"

last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

last_close = last["Close"]

# 미국 종목: 실시간(Finnhub) 또는 최신 체결가로 현재가 갱신 + 지연 배지
quote = None if kr else load_quote(ticker)
if quote and quote.price:
    last_close = float(quote.price)

change_abs = last_close - prev["Close"]
change_pct = (change_abs / prev["Close"]) * 100
volume_today = last["Volume"]
period_high = df["High"].max()
period_low = df["Low"].min()


def fmt_price(val: float) -> str:
    return f"₩{val:,.0f}" if kr else f"${val:,.2f}"


def fmt_volume(vol: float) -> str:
    if vol >= 1_000_000:
        return f"{vol / 1_000_000:.2f}M"
    if vol >= 1_000:
        return f"{vol / 1_000:.1f}K"
    return f"{vol:,.0f}"


st.title(f"📈  {company_name} — 장기/스윙 분석")
st.caption(f"{ticker}  ·  {market}  ·  {currency}  ·  {period_label} 기간  ·  중장기 매매 관점")

m1, m2, m3, m4 = st.columns(4)
m1.metric("현재가", fmt_price(last_close), f"{change_pct:+.2f}%")
m2.metric("전일 대비", fmt_price(change_abs), delta_color="normal")
m3.metric("거래량", fmt_volume(volume_today))
m4.metric(
    "기간 고가 / 저가",
    f"{fmt_price(period_high)} / {fmt_price(period_low)}",
)

if quote is not None:
    from data.collectors.realtime_quote import badge_text
    _badge = badge_text(quote)
    if _badge:
        st.caption(_badge)


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # ── Row 1: Candlestick ───────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="가격",
            increasing=dict(line=dict(color=UP_COLOR), fillcolor=UP_COLOR),
            decreasing=dict(line=dict(color=DOWN_COLOR), fillcolor=DOWN_COLOR),
        ),
        row=1, col=1,
    )

    for w in ma_windows:
        col = f"MA{w}"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=col,
                    line=dict(color=MA_COLORS[w], width=1.5),
                ),
                row=1, col=1,
            )

    if show_bb:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB 상단",
                line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB 하단",
                line=dict(color="rgba(180,180,180,0.7)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(180,180,180,0.06)",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Mid"],
                name="BB 중간",
                line=dict(color="rgba(180,180,180,0.4)", width=1, dash="dot"),
                showlegend=False,
            ),
            row=1, col=1,
        )

    # ── Row 2: Volume bars + OBV ─────────────────────────────────────────
    vol_colors = [
        UP_COLOR if c >= o else DOWN_COLOR
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="거래량",
            marker_color=vol_colors,
            opacity=0.55,
            showlegend=False,
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["OBV"],
            name="OBV",
            line=dict(color="#FF9800", width=1.3),
        ),
        row=2, col=1,
        secondary_y=True,
    )

    # ── Row 3: MACD ──────────────────────────────────────────────────────
    hist = df["MACD_Hist"].fillna(0)
    hist_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in hist]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=hist,
            name="MACD Hist",
            marker_color=hist_colors,
            opacity=0.65,
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            name="MACD",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_Signal"],
            name="Signal",
            line=dict(color="#FF9800", width=1.5),
        ),
        row=3, col=1,
    )

    # ── Row 4: RSI ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI",
            line=dict(color="#9C27B0", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.05)",
        ),
        row=4, col=1,
    )
    for level, color in [(70, "rgba(239,83,80,0.45)"), (30, "rgba(38,166,154,0.45)")]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color=color,
            line_width=1,
            row=4, col=1,
        )
    fig.add_hline(
        y=50,
        line_dash="dot",
        line_color="rgba(128,128,128,0.25)",
        line_width=1,
        row=4, col=1,
    )

    # ── Layout ───────────────────────────────────────────────────────────
    tick_fmt = ",.0f" if kr else ".2f"
    fig.update_layout(
        height=880,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(30,30,30,0.95)", font_size=12),
    )

    fig.update_yaxes(title_text="가격", tickformat=tick_fmt, row=1, col=1)
    fig.update_yaxes(title_text="거래량", showgrid=False, row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="OBV", showgrid=False, row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=4, col=1)

    return fig


st.plotly_chart(build_chart(df), width="stretch")


# ── Signal summary ────────────────────────────────────────────────────────────
st.divider()
st.subheader("기술적 시그널 요약")

latest = df.iloc[-1]
s1, s2, s3 = st.columns(3)

rsi_val = latest["RSI"]
if pd.isna(rsi_val):
    rsi_label, rsi_delta = "데이터 부족", None
elif rsi_val >= 70:
    rsi_label, rsi_delta = f"{rsi_val:.1f}", "과매수 구간"
elif rsi_val <= 30:
    rsi_label, rsi_delta = f"{rsi_val:.1f}", "과매도 구간"
else:
    rsi_label, rsi_delta = f"{rsi_val:.1f}", "중립"
s1.metric("RSI (14)", rsi_label, rsi_delta)

macd_val, macd_sig = latest["MACD"], latest["MACD_Signal"]
if pd.isna(macd_val) or pd.isna(macd_sig):
    macd_label, macd_delta = "데이터 부족", None
elif macd_val > macd_sig:
    macd_label, macd_delta = f"{macd_val:.4f}", "골든크로스 (상승)"
else:
    macd_label, macd_delta = f"{macd_val:.4f}", "데드크로스 (하락)"
s2.metric("MACD", macd_label, macd_delta)

ma5, ma20 = latest.get("MA5"), latest.get("MA20")
if ma5 is None or ma20 is None or pd.isna(ma5) or pd.isna(ma20):
    ma_label, ma_delta = "데이터 부족", None
elif ma5 > ma20:
    ma_label, ma_delta = f"{fmt_price(ma5)}", "단기 상승 추세"
else:
    ma_label, ma_delta = f"{fmt_price(ma5)}", "단기 하락 추세"
s3.metric("MA5 vs MA20", ma_label, ma_delta)

with st.expander("최근 가격 데이터"):
    display = df[["Open", "High", "Low", "Close", "Volume"]].tail(10).copy()
    display.index = display.index.strftime("%Y-%m-%d")
    price_fmt = (lambda x: f"₩{x:,.0f}") if kr else (lambda x: f"${x:,.2f}")
    for col in ["Open", "High", "Low", "Close"]:
        display[col] = display[col].map(price_fmt)
    display["Volume"] = display["Volume"].map(fmt_volume)
    st.dataframe(display, width="stretch")
