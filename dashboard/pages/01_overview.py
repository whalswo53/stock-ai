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
from analysis.technical import candle_patterns
from analysis.technical.indicators import (
    TechnicalIndicators, VOL_SPIKE_MULT, VWAP_WINDOW,
)
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

    # VWAP (롤링) — 가격 row 오버레이
    if "VWAP" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["VWAP"],
                name=f"VWAP{VWAP_WINDOW}",
                line=dict(color="#00BCD4", width=1.5, dash="dot"),
            ),
            row=1, col=1,
        )

    # ── Row 2: Volume bars + OBV ─────────────────────────────────────────
    # 급증일(평균 대비 VOL_SPIKE_MULT배 이상)은 노란색으로 강조
    spike = (
        df["Vol_Ratio"] >= VOL_SPIKE_MULT
        if "Vol_Ratio" in df.columns
        else pd.Series(False, index=df.index)
    )
    vol_colors = [
        "#FFD54F" if s else (UP_COLOR if c >= o else DOWN_COLOR)
        for c, o, s in zip(df["Close"], df["Open"], spike)
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

# 거래량 급증 안내 (최근 10거래일)
if "Vol_Ratio" in df.columns:
    recent_spikes = df[df["Vol_Ratio"] >= VOL_SPIKE_MULT].tail(10)
    latest_ratio = df["Vol_Ratio"].iloc[-1]
    if pd.notna(latest_ratio) and latest_ratio >= VOL_SPIKE_MULT:
        st.warning(
            f"🔊 **오늘 거래량 급증** — 20일 평균의 **{latest_ratio:.1f}배**. "
            "가격 움직임과 함께 해석하세요 (차트의 노란 거래량 바)."
        )
    elif not recent_spikes.empty:
        last_spike = recent_spikes.index[-1]
        st.caption(
            f"🔊 최근 거래량 급증일: {last_spike:%Y-%m-%d} "
            f"(평균 대비 {recent_spikes['Vol_Ratio'].iloc[-1]:.1f}배) — 차트의 노란 바"
        )


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

# ── Candle patterns ───────────────────────────────────────────────────────────
st.divider()
st.subheader("🕯️ 캔들 패턴 인식")

if not candle_patterns.is_available():
    st.caption(
        "TA-Lib이 설치되지 않아 캔들 패턴 분석을 사용할 수 없습니다 — "
        "`pip install ta-lib` 후 다시 실행하세요."
    )
else:
    with st.expander("❓ 신뢰도 산출 방식", expanded=False):
        st.markdown(
            "반전형(망치·장악·샛별·석별·도지)과 추세지속형(적삼병·흑삼병) 7종을 인식합니다.  \n"
            "**신뢰도**: 이 종목의 **최근 5년** 일봉에서 같은 패턴이 같은 방향으로 발생했던 "
            "모든 날의 **N일 후 수익률**로 승률·평균 수익률을 계산합니다.  \n"
            "⚠️ 표본 수가 적으면(특히 별형·삼병류) 통계적 의미가 약하니 참고용으로만 쓰세요. "
            "약세 패턴은 승률이 **낮을수록**(N일 후 하락) 패턴이 잘 맞은 것입니다."
        )

    horizon = st.radio(
        "수익률 측정 기간 (N일 후)", [3, 5, 10], index=1, horizontal=True,
        key="pattern_horizon",
    )

    @st.cache_data(ttl=86400, show_spinner=False)
    def load_pattern_history(ticker: str) -> pd.DataFrame:
        """패턴 통계용 5년 일봉 — 표시 기간과 무관하게 표본을 충분히 확보."""
        return PriceCollector().fetch(ticker, period="5y")

    with st.spinner("5년 히스토리에서 패턴 통계 계산 중…"):
        hist5y = load_pattern_history(ticker)

    if hist5y.empty or len(hist5y) < 120:
        st.caption("패턴 통계를 낼 만큼의 히스토리가 없습니다.")
    else:
        hits = candle_patterns.recent_hits(hist5y, lookback_days=5, horizon=int(horizon))
        if not hits:
            st.info("최근 5거래일 내 인식된 캔들 패턴이 없습니다.")
        else:
            rows = []
            for h_ in hits:
                stat = (
                    f"승률 {h_.win_rate * 100:.0f}% · 평균 {h_.avg_return:+.2f}% "
                    f"(표본 {h_.n_past}회)"
                    if h_.n_past > 0 else "과거 표본 없음"
                )
                rows.append({
                    "날짜":   f"{h_.date:%Y-%m-%d}",
                    "패턴":   h_.name,
                    "유형":   h_.kind,
                    "방향":   "🟢 강세" if h_.direction == "강세"
                              else ("🔴 약세" if h_.direction == "약세" else "⚪ 중립"),
                    f"과거 {horizon}일 후 성과": stat,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            low_sample = [h_ for h_ in hits if 0 < h_.n_past < 10]
            if low_sample:
                st.caption(
                    "⚠️ 표본 10회 미만인 패턴이 포함되어 있습니다 — 통계보다 캔들 맥락을 우선하세요."
                )

with st.expander("최근 가격 데이터"):
    display = df[["Open", "High", "Low", "Close", "Volume"]].tail(10).copy()
    display.index = display.index.strftime("%Y-%m-%d")
    price_fmt = (lambda x: f"₩{x:,.0f}") if kr else (lambda x: f"${x:,.2f}")
    for col in ["Open", "High", "Low", "Close"]:
        display[col] = display[col].map(price_fmt)
    display["Volume"] = display["Volume"].map(fmt_volume)
    st.dataframe(display, width="stretch")
