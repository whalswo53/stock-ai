import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import math

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.quant.aggregator import AggregatedPairResult, QuantAggregator
from analysis.quant.mean_reversion import MeanReversionResult

# ── Palette ───────────────────────────────────────────────────────────────────
SIGNAL_COLOR = {"BUY": "#26a69a", "SELL": "#ef5350", "CLOSE": "#FF9800", "WAIT": "#9E9E9E"}
OLS_COLOR    = "#2196F3"
KALMAN_COLOR = "#FF9800"

PRESET_PAIRS = {
    "삼성전자 / SK하이닉스 (반도체)": ("005930.KS", "000660.KS"),
    "AAPL / MSFT (미국 빅테크)":      ("AAPL",       "MSFT"),
    "NVDA / AMD (GPU)":               ("NVDA",       "AMD"),
    "카카오 / 네이버 (플랫폼)":        ("035720.KS",  "035420.KS"),
    "직접 입력":                       None,
}
PERIOD_OPTIONS = {"6개월": "6mo", "1년": "1y", "2년": "2y"}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 퀀트 분석 설정")

    mode = st.radio("분석 모드", ["페어 분석", "단일 종목 평균회귀"], horizontal=True)

    st.divider()

    if mode == "페어 분석":
        preset_label = st.selectbox("종목 쌍", list(PRESET_PAIRS.keys()))
        preset = PRESET_PAIRS[preset_label]
        if preset is None:
            c1, c2 = st.columns(2)
            ticker_a = c1.text_input("종목 A", "AAPL").strip().upper()
            ticker_b = c2.text_input("종목 B", "MSFT").strip().upper()
        else:
            ticker_a, ticker_b = preset
            st.caption(f"A: `{ticker_a}`  ·  B: `{ticker_b}`")
        single_ticker = ""
    else:
        single_ticker = st.text_input("종목 코드", "AAPL").strip().upper()
        st.caption("KOSPI 예시: 005930.KS  |  NASDAQ 예시: AAPL")
        ticker_a = ticker_b = ""

    period_label  = st.selectbox("분석 기간", list(PERIOD_OPTIONS.keys()), index=1)
    period        = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("신호 임계값")
    entry_z      = st.slider("진입 Z-score", 1.0, 3.5, 2.0, 0.1)
    exit_z       = st.slider("청산 Z-score", 0.1, 1.0, 0.5, 0.1)
    zscore_window = st.slider("Z-score 윈도우 (일)", 10, 60, 30, 5)

    if mode == "페어 분석":
        st.divider()
        st.subheader("칼만 필터")
        kalman_delta = st.select_slider(
            "적응 속도 (delta)",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4,
            format_func=lambda v: f"{v:.0e}",
            help="작을수록 느리고 안정적, 클수록 빠르게 적응",
        )


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _run_pair(ta, tb, period, window, ez, xz, delta) -> AggregatedPairResult:
    agg = QuantAggregator(
        period=period, zscore_window=window,
        entry_z=ez, exit_z=xz, kalman_delta=delta,
    )
    return agg.run_pair(ta, tb)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_single(ticker, period, window, ez, xz) -> MeanReversionResult:
    agg = QuantAggregator(period=period, zscore_window=window, entry_z=ez, exit_z=xz)
    return agg.run_single(ticker)


# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 통계적 차익거래 분석")
st.caption(
    "**페어 분석**: OLS + 칼만 필터 앙상블  |  "
    "**단일 종목**: 평균회귀 Z-score (ADF 검정 포함)"
)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE A: PAIR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "페어 분석":
    with st.spinner(f"'{ticker_a}' & '{ticker_b}' 앙상블 분석 중…"):
        try:
            result: AggregatedPairResult = _run_pair(
                ticker_a, ticker_b, period, zscore_window, entry_z, exit_z, kalman_delta
            )
            error_msg = None
        except ValueError as e:
            error_msg = str(e)

    if error_msg:
        st.error(error_msg)
        st.stop()

    coint  = result.coint_result
    ols    = result.ols_spread
    kalman = result.kalman_spread

    # ── 섹션 1: 공적분 검정 ──────────────────────────────────────────────────
    st.subheader("1. 공적분 검정 (Engle-Granger)")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    coint_ok = coint.is_cointegrated
    c1.metric("검정 결과", "공적분 있음 ✅" if coint_ok else "공적분 없음 ❌")
    c2.metric("p-value", f"{coint.pvalue:.4f}")
    c3.metric("검정 통계량", f"{coint.test_stat:.3f}")
    c4.metric("임계값 (5%)", f"{coint.critical_values['5%']:.3f}")

    if not coint_ok:
        st.warning(
            f"공적분 관계가 통계적으로 유의하지 않습니다 (p={coint.pvalue:.4f}). "
            "페어 전략 신뢰도가 낮을 수 있습니다."
        )

    # ── 섹션 2: 모델 기여도 테이블 ──────────────────────────────────────────
    st.subheader("2. 모델별 기여도")

    contrib_rows = []
    for c in result.contributions:
        contrib_rows.append({
            "모델": c.name,
            "Z-score": f"{c.zscore:+.3f}",
            "가중치": f"{c.weight * 100:.1f}%",
            "종목A 신호": c.signal_a,
            "종목B 신호": c.signal_b,
            "신뢰도 지표": c.confidence_label,
        })
    contrib_rows.append({
        "모델": "**종합 (앙상블)**",
        "Z-score": f"{result.composite_zscore:+.3f}",
        "가중치": "100%",
        "종목A 신호": result.signal_a,
        "종목B 신호": result.signal_b,
        "신뢰도 지표": "가중 평균",
    })

    st.dataframe(
        pd.DataFrame(contrib_rows),
        use_container_width=True,
        hide_index=True,
    )

    # Weight bar (visual)
    w_ols    = result.contributions[0].weight
    w_kalman = result.contributions[1].weight
    wbar_cols = st.columns([w_ols, w_kalman])
    wbar_cols[0].markdown(
        f'<div style="background:{OLS_COLOR};border-radius:4px;padding:4px 8px;'
        f'font-size:12px;text-align:center">OLS {w_ols*100:.0f}%</div>',
        unsafe_allow_html=True,
    )
    wbar_cols[1].markdown(
        f'<div style="background:{KALMAN_COLOR};border-radius:4px;padding:4px 8px;'
        f'font-size:12px;text-align:center;color:#000">Kalman {w_kalman*100:.0f}%</div>',
        unsafe_allow_html=True,
    )

    # ── 섹션 3: 차트 ─────────────────────────────────────────────────────────
    st.subheader("3. 가격·스프레드·Z-score 비교")

    def build_pair_chart(
        ols_sr, kalman_sr, entry_z: float, exit_z: float
    ) -> go.Figure:
        dates = ols_sr.dates

        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.30, 0.22, 0.24, 0.24],
            subplot_titles=[
                f"정규화 가격 ({ols_sr.price_a.name} vs {ols_sr.price_b.name})",
                "칼만 헤지비율 (동적) vs OLS (고정)",
                "OLS 스프레드 Z-score",
                "칼만 필터 Z-score",
            ],
        )

        # Row 1: Normalised prices
        norm_a = ols_sr.price_a / float(ols_sr.price_a.iloc[0]) * 100
        norm_b = ols_sr.price_b / float(ols_sr.price_b.iloc[0]) * 100
        for series, name, color in [
            (norm_a, str(ols_sr.price_a.name), OLS_COLOR),
            (norm_b, str(ols_sr.price_b.name), KALMAN_COLOR),
        ]:
            fig.add_trace(go.Scatter(x=dates, y=series, name=name,
                                     line=dict(color=color, width=1.8)), row=1, col=1)

        # Row 2: Hedge ratio comparison
        fig.add_trace(go.Scatter(
            x=dates, y=kalman_sr.hedge_ratio,
            name="칼만 β(t)", line=dict(color=KALMAN_COLOR, width=1.5),
        ), row=2, col=1)
        fig.add_hline(
            y=ols_sr.hedge_ratio, line_dash="dash",
            line_color=OLS_COLOR, line_width=1.5,
            annotation_text=f"OLS β={ols_sr.hedge_ratio:.3f}",
            annotation_position="right",
            row=2, col=1,
        )

        # Rows 3 & 4: Z-scores
        for row_idx, (zseries, name, color) in enumerate([
            (ols_sr.zscore,    "OLS Z",    OLS_COLOR),
            (kalman_sr.zscore, "Kalman Z", KALMAN_COLOR),
        ], start=3):
            fig.add_trace(go.Scatter(
                x=dates, y=zseries, name=name,
                line=dict(color=color, width=1.6), showlegend=False,
            ), row=row_idx, col=1)
            for level, lcolor, dash in [
                ( entry_z, "rgba(239,83,80,0.55)",  "dash"),
                (-entry_z, "rgba(38,166,154,0.55)", "dash"),
                ( exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                (-exit_z,  "rgba(255,152,0,0.40)",  "dot"),
                (0,        "rgba(128,128,128,0.25)", "solid"),
            ]:
                fig.add_hline(y=level, line_dash=dash, line_color=lcolor,
                              line_width=1, row=row_idx, col=1)

        fig.update_layout(
            height=820, template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="정규화 가격", row=1, col=1)
        fig.update_yaxes(title_text="헤지비율 β", row=2, col=1)
        fig.update_yaxes(title_text="OLS Z", row=3, col=1)
        fig.update_yaxes(title_text="Kalman Z", row=4, col=1)
        return fig

    st.plotly_chart(build_pair_chart(ols, kalman, entry_z, exit_z), width="stretch")

    # ── 섹션 4: 종합 신호 ────────────────────────────────────────────────────
    st.subheader("4. 종합 신호")

    s1, s2, s3 = st.columns(3)
    s1.metric(f"{ticker_a} 신호", result.signal_a)
    s2.metric(f"{ticker_b} 신호", result.signal_b)
    s3.metric("앙상블 Z-score", f"{result.composite_zscore:+.3f}")

    sig = result.signal_a
    msg = result.label
    if sig == "WAIT":
        st.info(f"**관망** — {msg}")
    elif sig == "CLOSE":
        st.success(f"**청산 신호** — {msg}")
    elif sig == "BUY":
        st.success(f"**매수/매도 진입** — {msg}")
    else:
        st.error(f"**매도/매수 진입** — {msg}")

    with st.expander("신호 해석 가이드"):
        st.markdown(
            f"| 앙상블 Z | {ticker_a} | {ticker_b} | 의미 |\n"
            f"|---|---|---|---|\n"
            f"| Z > **{entry_z}** | SELL | BUY | 스프레드 과대 → A 고평가 |\n"
            f"| Z < **−{entry_z}** | BUY | SELL | 스프레드 과소 → A 저평가 |\n"
            f"| |Z| < **{exit_z}** | CLOSE | CLOSE | 평균 회귀 완료 |\n"
            f"| 그 외 | WAIT | WAIT | 진입 조건 미충족 |\n\n"
            "**가중치 결정 원리**\n"
            "- OLS 가중치 = R² × max(0, 1−2·p_value) — 공적분이 강하고 적합도가 높을수록 증가\n"
            "- 칼만 가중치 = 헤지비율 안정성 (1−CV) — 동적 비율이 일관될수록 증가\n"
        )

    # ── 섹션 5: 최근 데이터 ──────────────────────────────────────────────────
    with st.expander("최근 20일 Z-score 상세"):
        ols_z    = ols.zscore.dropna().tail(20)
        kal_z    = kalman.zscore.reindex(ols_z.index)
        sp_ols   = ols.spread.reindex(ols_z.index)

        tbl = pd.DataFrame({
            "날짜":           ols_z.index.strftime("%Y-%m-%d"),
            "OLS 스프레드":   sp_ols.values,
            "OLS Z":          ols_z.values,
            "Kalman Z":       kal_z.values,
        }).reset_index(drop=True)

        def _color_z(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > entry_z:   return "color:#ef5350;font-weight:bold"
            if v < -entry_z:  return "color:#26a69a;font-weight:bold"
            if abs(v) < exit_z: return "color:#FF9800"
            return ""

        st.dataframe(
            tbl.style
            .format({"OLS 스프레드": "{:.4f}", "OLS Z": "{:.4f}", "Kalman Z": "{:.4f}"})
            .map(_color_z, subset=["OLS Z", "Kalman Z"]),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  MODE B: SINGLE STOCK MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════════════════
else:
    with st.spinner(f"'{single_ticker}' 평균회귀 분석 중…"):
        try:
            mr: MeanReversionResult = _run_single(
                single_ticker, period, zscore_window, entry_z, exit_z
            )
            error_msg = None
        except ValueError as e:
            error_msg = str(e)

    if error_msg:
        st.error(error_msg)
        st.stop()

    # ── 섹션 1: ADF 검정 결과 ────────────────────────────────────────────────
    st.subheader("1. ADF 단위근 검정 (평균회귀 가능성)")

    c1, c2, c3 = st.columns([2, 1, 1])
    mr_ok = mr.is_mean_reverting
    c1.metric("검정 결과", "평균회귀 가능 ✅" if mr_ok else "단위근 존재 ❌")
    c2.metric("ADF p-value", f"{mr.adf_pvalue:.4f}")
    hl = mr.half_life_days
    c3.metric(
        "OU 반감기",
        f"{hl:.1f}일" if math.isfinite(hl) else "∞ (비정상)",
        help="평균으로 절반쯤 되돌아오는 데 걸리는 추정 기간",
    )

    if not mr_ok:
        st.warning(
            f"ADF 검정에서 단위근을 기각하지 못했습니다 (p={mr.adf_pvalue:.4f}). "
            "가격이 추세를 갖고 있어 평균회귀 전략의 신뢰도가 낮습니다."
        )
    elif math.isfinite(hl) and hl > 60:
        st.info(f"반감기 {hl:.0f}일 — 평균회귀 속도가 느려 단기 전략 적용에 주의하세요.")

    # ── 섹션 2: 가격 + Z-score 차트 ─────────────────────────────────────────
    st.subheader("2. 가격 & Z-score")

    def build_single_chart(mr: MeanReversionResult, entry_z: float, exit_z: float) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.5, 0.5],
            subplot_titles=[f"{mr.ticker} 가격", f"Z-score (윈도우 {mr.zscore_window}일)"],
        )

        # Row 1: Price + rolling mean
        roll_mean = mr.price.rolling(mr.zscore_window).mean()
        fig.add_trace(go.Scatter(
            x=mr.dates, y=mr.price, name=mr.ticker,
            line=dict(color="#2196F3", width=1.8),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=mr.dates, y=roll_mean, name=f"MA{mr.zscore_window}",
            line=dict(color="#FF9800", width=1.3, dash="dash"), showlegend=True,
        ), row=1, col=1)

        # Row 2: Z-score
        fig.add_trace(go.Scatter(
            x=mr.dates, y=mr.zscore, name="Z-score",
            line=dict(color="#00BCD4", width=1.8),
            fill="tozeroy", fillcolor="rgba(0,188,212,0.07)",
            showlegend=False,
        ), row=2, col=1)
        for level, color, dash in [
            ( entry_z, "rgba(239,83,80,0.55)",  "dash"),
            (-entry_z, "rgba(38,166,154,0.55)", "dash"),
            ( exit_z,  "rgba(255,152,0,0.40)",  "dot"),
            (-exit_z,  "rgba(255,152,0,0.40)",  "dot"),
            (0,        "rgba(128,128,128,0.25)", "solid"),
        ]:
            fig.add_hline(y=level, line_dash=dash, line_color=color, line_width=1, row=2, col=1)

        fig.update_layout(
            height=600, template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="가격", row=1, col=1)
        fig.update_yaxes(title_text="Z-score", row=2, col=1)
        return fig

    st.plotly_chart(build_single_chart(mr, entry_z, exit_z), width="stretch")

    # ── 섹션 3: 현재 신호 ────────────────────────────────────────────────────
    st.subheader("3. 현재 신호")

    s1, s2 = st.columns(2)
    s1.metric(f"{single_ticker} 신호", mr.signal)
    z_disp = f"{mr.zscore_latest:+.3f}" if math.isfinite(mr.zscore_latest) else "N/A"
    s2.metric("현재 Z-score", z_disp)

    sig = mr.signal
    if sig == "WAIT":
        st.info(f"**관망** — {mr.label}")
    elif sig == "CLOSE":
        st.success(f"**청산 신호** — {mr.label}")
    elif sig == "BUY":
        st.success(f"**매수 신호** — {mr.label}")
    else:
        st.error(f"**매도 신호** — {mr.label}")

    # ── 최근 데이터 테이블 ────────────────────────────────────────────────────
    with st.expander("최근 20일 데이터"):
        z_tail = mr.zscore.dropna().tail(20)
        p_tail = mr.price.reindex(z_tail.index)
        tbl = pd.DataFrame({
            "날짜":   z_tail.index.strftime("%Y-%m-%d"),
            "가격":   p_tail.values,
            "Z-score": z_tail.values,
        }).reset_index(drop=True)

        def _color_z2(val):
            try:
                v = float(val)
            except (TypeError, ValueError):
                return ""
            if v > entry_z:    return "color:#ef5350;font-weight:bold"
            if v < -entry_z:   return "color:#26a69a;font-weight:bold"
            if abs(v) < exit_z: return "color:#FF9800"
            return ""

        st.dataframe(
            tbl.style
            .format({"가격": "{:,.2f}", "Z-score": "{:.4f}"})
            .map(_color_z2, subset=["Z-score"]),
            use_container_width=True,
        )
