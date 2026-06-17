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
from analysis.quant.pair_scanner import INDUSTRY_GROUPS, PairScanner, PairScanResult
from config.sources import TICKER_KR_NAME
from data.collectors.price_collector import PriceCollector

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
PRESET_DIRECT = "직접 입력"
PRESET_DIRECT_IDX = list(PRESET_PAIRS.keys()).index(PRESET_DIRECT)

PERIOD_OPTIONS = {"6개월": "6mo", "1년": "1y", "2년": "2y"}

# ── Ticker → display name lookup ──────────────────────────────────────────────
# Priority: INDUSTRY_GROUPS names > TICKER_KR_NAME > yfinance fallback
_NAME_LOOKUP: dict[str, str] = dict(TICKER_KR_NAME)
for _grp in INDUSTRY_GROUPS.values():
    _NAME_LOOKUP.update(_grp["names"])


@st.cache_data(ttl=3600, show_spinner=False)
def _get_name(ticker: str) -> str:
    """Returns a human-readable name for ticker, falling back to yfinance."""
    if ticker in _NAME_LOOKUP:
        return _NAME_LOOKUP[ticker]
    try:
        info = PriceCollector().get_info(ticker)
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def _label(ticker: str) -> str:
    """Returns 'Name (TICKER)' or just TICKER when no name is found."""
    name = _get_name(ticker)
    return f"{name} ({ticker})" if name != ticker else ticker


# ── Sidebar: shared settings ──────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 퀀트 분석 설정")

    period_label  = st.selectbox("분석 기간", list(PERIOD_OPTIONS.keys()), index=1)
    period        = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("신호 임계값")
    entry_z       = st.slider("진입 Z-score", 1.0, 3.5, 2.0, 0.1)
    exit_z        = st.slider("청산 Z-score", 0.1, 1.0, 0.5, 0.1)
    zscore_window = st.slider("Z-score 윈도우 (일)", 10, 60, 30, 5)

    st.divider()
    st.subheader("칼만 필터")
    kalman_delta = st.select_slider(
        "적응 속도 (delta)",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        value=1e-4,
        format_func=lambda v: f"{v:.0e}",
        help="직접 분석 탭에만 적용됩니다",
    )


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_group_prices(group_name: str, period: str) -> dict[str, list]:
    scanner = PairScanner(period=period)
    prices = scanner.fetch_prices(group_name)
    out: dict[str, list] = {}
    idx: list[str] = []
    for ticker, series in prices.items():
        out[ticker] = series.tolist()
        if not idx:
            idx = series.index.strftime("%Y-%m-%d").tolist()
    out["__index__"] = idx
    return out


def _restore_prices(raw: dict[str, list]) -> dict[str, pd.Series]:
    idx = pd.to_datetime(raw["__index__"])
    return {k: pd.Series(v, index=idx, name=k) for k, v in raw.items() if k != "__index__"}


@st.cache_data(ttl=1800, show_spinner=False)
def _run_scan(
    group_name: str, period: str, window: int, ez: float, xz: float
) -> list[PairScanResult]:
    scanner = PairScanner(period=period, zscore_window=window, entry_z=ez, exit_z=xz)
    raw = _fetch_group_prices(group_name, period)
    prices = _restore_prices(raw)
    return scanner.scan(group_name, prices)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_pair(ta, tb, period, window, ez, xz, delta) -> AggregatedPairResult:
    return QuantAggregator(
        period=period, zscore_window=window,
        entry_z=ez, exit_z=xz, kalman_delta=delta,
    ).run_pair(ta, tb)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_single(ticker, period, window, ez, xz) -> MeanReversionResult:
    return QuantAggregator(
        period=period, zscore_window=window, entry_z=ez, exit_z=xz
    ).run_single(ticker)


# ── Page header ───────────────────────────────────────────────────────────────
st.title("📊 통계적 차익거래 분석")
st.caption(
    "**자동 스캔**: 산업군 내 공적분 쌍 자동 발굴  |  "
    "**직접 분석**: OLS + 칼만 앙상블 또는 단일 종목 평균회귀"
)

tab_scan, tab_direct = st.tabs(["🔍 자동 스캔", "📈 직접 분석"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: AUTO SCAN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_scan:
    st.subheader("산업군 공적분 스캐너")
    st.caption("같은 산업군 내 모든 종목 조합에 대해 공적분 검정을 자동 실행합니다.")

    col_g, col_btn = st.columns([3, 1])
    with col_g:
        group_name = st.selectbox("산업군 선택", list(INDUSTRY_GROUPS.keys()),
                                  label_visibility="collapsed")
    with col_btn:
        run_scan = st.button("스캔 시작", type="primary", use_container_width=True)

    group_info  = INDUSTRY_GROUPS[group_name]
    ticker_list = group_info["tickers"]
    name_map    = group_info["names"]
    n_tickers   = len(ticker_list)
    n_pairs     = n_tickers * (n_tickers - 1) // 2

    names_str = "  ·  ".join(f"`{name_map.get(t, t)}`" for t in ticker_list)
    st.caption(f"종목 {n_tickers}개 ({n_pairs}쌍 검정 예정):  {names_str}")

    scan_key = (group_name, period, zscore_window, entry_z, exit_z)
    if run_scan or st.session_state.get("last_scan_key") == scan_key:
        st.session_state["last_scan_key"] = scan_key

        with st.spinner(f"'{group_name}' 그룹 {n_pairs}쌍 분석 중… (최초 실행 시 약 10~30초 소요)"):
            scan_results: list[PairScanResult] = _run_scan(
                group_name, period, zscore_window, entry_z, exit_z
            )

        if not scan_results:
            st.warning("분석 가능한 쌍이 없습니다. 기간을 늘리거나 다른 그룹을 선택해보세요.")
            st.stop()

        scanner   = PairScanner()
        df_all    = scanner.to_dataframe(scan_results)
        df_show   = df_all.drop(columns=["_ticker_a", "_ticker_b"])
        n_coint   = sum(1 for r in scan_results if r.is_cointegrated)

        m1, m2, m3 = st.columns(3)
        m1.metric("검정 쌍 수", n_pairs)
        m2.metric("공적분 확인", f"{n_coint}쌍")
        m3.metric("최저 p-value", f"{scan_results[0].pvalue:.4f}")

        st.markdown("##### 검정 결과 (p-value 낮은 순 · 상위 5쌍)")

        def _style_scan(df: pd.DataFrame) -> pd.DataFrame:
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            for i, row in df.iterrows():
                pv = row["p-value"]
                if pv < 0.05:
                    styles.loc[i, "p-value"] = "color:#26a69a;font-weight:bold"
                elif pv < 0.1:
                    styles.loc[i, "p-value"] = "color:#FF9800"
                else:
                    styles.loc[i, "p-value"] = "color:#9E9E9E"
                for col in ["A 신호", "B 신호"]:
                    sig = row[col]
                    if sig == "BUY":
                        styles.loc[i, col] = "color:#26a69a;font-weight:bold"
                    elif sig == "SELL":
                        styles.loc[i, col] = "color:#ef5350;font-weight:bold"
                    elif sig == "CLOSE":
                        styles.loc[i, col] = "color:#FF9800"
            return styles

        top5  = df_show.head(5)
        event = st.dataframe(
            top5.style.apply(_style_scan, axis=None),
            use_container_width=True, hide_index=True,
            selection_mode="single-row", on_select="rerun",
        )
        if len(df_show) > 5:
            with st.expander(f"전체 {len(df_show)}쌍 보기"):
                st.dataframe(df_show.style.apply(_style_scan, axis=None),
                             use_container_width=True, hide_index=True)

        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            picked = scan_results[selected_rows[0]]
            st.session_state["scan_prefill"] = {
                "ticker_a": picked.ticker_a,
                "ticker_b": picked.ticker_b,
                "name_a":   picked.name_a,
                "name_b":   picked.name_b,
            }
            st.success(
                f"**{picked.name_a} ({picked.ticker_a}) / "
                f"{picked.name_b} ({picked.ticker_b})** 선택됨 "
                f"(p={picked.pvalue:.4f}) — "
                "**직접 분석 탭**에서 상세 분석을 확인하세요."
            )
    else:
        st.info("산업군을 선택하고 **스캔 시작** 버튼을 누르세요.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: DIRECT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_direct:

    mode = st.radio("분석 모드", ["페어 분석", "단일 종목 평균회귀"], horizontal=True)

    # ── Ticker inputs ─────────────────────────────────────────────────────────
    if mode == "페어 분석":
        prefill = st.session_state.get("scan_prefill", {})

        if prefill:
            st.info(
                f"자동 스캔에서 선택된 쌍: "
                f"**{prefill['name_a']}** (`{prefill['ticker_a']}`) / "
                f"**{prefill['name_b']}** (`{prefill['ticker_b']}`)"
            )

        # Key encodes the active prefill pair.
        # When scan_prefill changes the key changes → widget resets to new default.
        prefill_sig   = f"{prefill.get('ticker_a','')}-{prefill.get('ticker_b','')}"
        preset_options = list(PRESET_PAIRS.keys())

        preset_label = st.selectbox(
            "프리셋",
            preset_options,
            # Force "직접 입력" whenever a scan prefill is active
            index=PRESET_DIRECT_IDX if prefill else 0,
            key=f"preset_sel_{prefill_sig}",
        )
        preset = PRESET_PAIRS[preset_label]

        if preset is None:
            default_a = prefill.get("ticker_a", "AAPL") if prefill else "AAPL"
            default_b = prefill.get("ticker_b", "MSFT") if prefill else "MSFT"
            c1, c2    = st.columns(2)
            ticker_a  = c1.text_input("종목 A", value=default_a,
                                      key=f"ta_{prefill_sig}").strip().upper()
            ticker_b  = c2.text_input("종목 B", value=default_b,
                                      key=f"tb_{prefill_sig}").strip().upper()
        else:
            ticker_a, ticker_b = preset
            st.caption(f"A: {_label(ticker_a)}  ·  B: {_label(ticker_b)}")

        single_ticker = ""

    else:
        single_ticker = st.text_input("종목 코드", "AAPL").strip().upper()
        st.caption("KOSPI 예시: 005930.KS  |  NASDAQ 예시: AAPL")
        ticker_a = ticker_b = ""

    st.divider()

    # ── Analysis ──────────────────────────────────────────────────────────────
    if mode == "페어 분석":
        # Pre-compute display labels once — used throughout this section
        label_a = _label(ticker_a)
        label_b = _label(ticker_b)

        with st.spinner(f"'{label_a}' & '{label_b}' 앙상블 분석 중…"):
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

        # ── 섹션 1: 공적분 검정 ──────────────────────────────────────────────
        st.subheader("1. 공적분 검정 (Engle-Granger)")

        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        coint_ok = coint.is_cointegrated
        c1.metric("검정 결과", "공적분 있음 ✅" if coint_ok else "공적분 없음 ❌")
        c2.metric("p-value", f"{coint.pvalue:.4f}")
        c3.metric("검정 통계량", f"{coint.test_stat:.3f}")
        c4.metric("임계값 (5%)", f"{coint.critical_values['5%']:.3f}")
        st.caption(f"종목쌍: **{label_a}** vs **{label_b}**")

        if not coint_ok:
            st.warning(
                f"공적분 관계가 통계적으로 유의하지 않습니다 (p={coint.pvalue:.4f}). "
                "페어 전략 신뢰도가 낮을 수 있습니다."
            )

        # ── 섹션 2: 모델 기여도 테이블 ──────────────────────────────────────
        st.subheader("2. 모델별 기여도")

        col_sig_a = f"{label_a} 신호"
        col_sig_b = f"{label_b} 신호"

        contrib_rows = []
        for c in result.contributions:
            contrib_rows.append({
                "모델":      c.name,
                "Z-score":   f"{c.zscore:+.3f}",
                "가중치":    f"{c.weight * 100:.1f}%",
                col_sig_a:   c.signal_a,
                col_sig_b:   c.signal_b,
                "신뢰도 지표": c.confidence_label,
            })
        contrib_rows.append({
            "모델":      "**종합 (앙상블)**",
            "Z-score":   f"{result.composite_zscore:+.3f}",
            "가중치":    "100%",
            col_sig_a:   result.signal_a,
            col_sig_b:   result.signal_b,
            "신뢰도 지표": "가중 평균",
        })

        st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

        # Weight bar
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

        # ── 섹션 3: 차트 ─────────────────────────────────────────────────────
        st.subheader("3. 가격·스프레드·Z-score 비교")

        def build_pair_chart(ols_sr, kalman_sr, la, lb, entry_z, exit_z) -> go.Figure:
            dates = ols_sr.dates
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                vertical_spacing=0.04, row_heights=[0.30, 0.22, 0.24, 0.24],
                subplot_titles=[
                    f"정규화 가격 ({la} vs {lb})",
                    f"칼만 헤지비율 (동적) vs OLS (고정)",
                    "OLS 스프레드 Z-score",
                    "칼만 필터 Z-score",
                ],
            )

            norm_a = ols_sr.price_a / float(ols_sr.price_a.iloc[0]) * 100
            norm_b = ols_sr.price_b / float(ols_sr.price_b.iloc[0]) * 100
            for series, name, color in [
                (norm_a, la, OLS_COLOR),
                (norm_b, lb, KALMAN_COLOR),
            ]:
                fig.add_trace(go.Scatter(x=dates, y=series, name=name,
                                         line=dict(color=color, width=1.8)), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=dates, y=kalman_sr.hedge_ratio,
                name="칼만 β(t)", line=dict(color=KALMAN_COLOR, width=1.5),
            ), row=2, col=1)
            fig.add_hline(
                y=ols_sr.hedge_ratio, line_dash="dash",
                line_color=OLS_COLOR, line_width=1.5,
                annotation_text=f"OLS β={ols_sr.hedge_ratio:.3f}",
                annotation_position="right", row=2, col=1,
            )

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
            fig.update_yaxes(title_text="헤지비율 β",  row=2, col=1)
            fig.update_yaxes(title_text="OLS Z",       row=3, col=1)
            fig.update_yaxes(title_text="Kalman Z",    row=4, col=1)
            return fig

        st.plotly_chart(
            build_pair_chart(ols, kalman, label_a, label_b, entry_z, exit_z),
            width="stretch",
        )

        # ── 섹션 4: 종합 신호 ────────────────────────────────────────────────
        st.subheader("4. 종합 신호")

        s1, s2, s3 = st.columns(3)
        s1.metric(f"{label_a} 신호", result.signal_a)
        s2.metric(f"{label_b} 신호", result.signal_b)
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
                f"| 앙상블 Z | {label_a} | {label_b} | 의미 |\n"
                f"|---|---|---|---|\n"
                f"| Z > **{entry_z}** | SELL | BUY | 스프레드 과대 → A 고평가 |\n"
                f"| Z < **−{entry_z}** | BUY | SELL | 스프레드 과소 → A 저평가 |\n"
                f"| |Z| < **{exit_z}** | CLOSE | CLOSE | 평균 회귀 완료 |\n"
                f"| 그 외 | WAIT | WAIT | 진입 조건 미충족 |\n\n"
                "**가중치 결정 원리**\n"
                "- OLS 가중치 = R² × max(0, 1−2·p_value) — 공적분이 강하고 적합도가 높을수록 증가\n"
                "- 칼만 가중치 = 헤지비율 안정성 (1−CV) — 동적 비율이 일관될수록 증가\n"
            )

        # ── 섹션 5: 최근 데이터 ──────────────────────────────────────────────
        with st.expander("최근 20일 Z-score 상세"):
            ols_z  = ols.zscore.dropna().tail(20)
            kal_z  = kalman.zscore.reindex(ols_z.index)
            sp_ols = ols.spread.reindex(ols_z.index)

            tbl = pd.DataFrame({
                "날짜":         ols_z.index.strftime("%Y-%m-%d"),
                "OLS 스프레드": sp_ols.values,
                "OLS Z":        ols_z.values,
                "Kalman Z":     kal_z.values,
            }).reset_index(drop=True)

            def _color_z(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v > entry_z:     return "color:#ef5350;font-weight:bold"
                if v < -entry_z:    return "color:#26a69a;font-weight:bold"
                if abs(v) < exit_z: return "color:#FF9800"
                return ""

            st.dataframe(
                tbl.style
                .format({"OLS 스프레드": "{:.4f}", "OLS Z": "{:.4f}", "Kalman Z": "{:.4f}"})
                .map(_color_z, subset=["OLS Z", "Kalman Z"]),
                use_container_width=True,
            )

    # ── MODE B: SINGLE STOCK ─────────────────────────────────────────────────
    else:
        single_label = _label(single_ticker)

        with st.spinner(f"'{single_label}' 평균회귀 분석 중…"):
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
        st.caption(f"종목: **{single_label}**")

        if not mr_ok:
            st.warning(
                f"ADF 검정에서 단위근을 기각하지 못했습니다 (p={mr.adf_pvalue:.4f}). "
                "가격이 추세를 갖고 있어 평균회귀 전략의 신뢰도가 낮습니다."
            )
        elif math.isfinite(hl) and hl > 60:
            st.info(f"반감기 {hl:.0f}일 — 평균회귀 속도가 느려 단기 전략 적용에 주의하세요.")

        st.subheader("2. 가격 & Z-score")

        def build_single_chart(mr: MeanReversionResult, label: str, entry_z, exit_z) -> go.Figure:
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                vertical_spacing=0.06, row_heights=[0.5, 0.5],
                subplot_titles=[
                    f"{label} 가격",
                    f"Z-score (윈도우 {mr.zscore_window}일)",
                ],
            )
            roll_mean = mr.price.rolling(mr.zscore_window).mean()
            fig.add_trace(go.Scatter(
                x=mr.dates, y=mr.price, name=label,
                line=dict(color="#2196F3", width=1.8),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=mr.dates, y=roll_mean, name=f"MA{mr.zscore_window}",
                line=dict(color="#FF9800", width=1.3, dash="dash"),
            ), row=1, col=1)
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
            fig.update_yaxes(title_text="가격",    row=1, col=1)
            fig.update_yaxes(title_text="Z-score", row=2, col=1)
            return fig

        st.plotly_chart(build_single_chart(mr, single_label, entry_z, exit_z), width="stretch")

        st.subheader("3. 현재 신호")

        s1, s2 = st.columns(2)
        s1.metric(f"{single_label} 신호", mr.signal)
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

        with st.expander("최근 20일 데이터"):
            z_tail = mr.zscore.dropna().tail(20)
            p_tail = mr.price.reindex(z_tail.index)
            tbl = pd.DataFrame({
                "날짜":    z_tail.index.strftime("%Y-%m-%d"),
                "가격":    p_tail.values,
                "Z-score": z_tail.values,
            }).reset_index(drop=True)

            def _color_z2(val):
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    return ""
                if v > entry_z:     return "color:#ef5350;font-weight:bold"
                if v < -entry_z:    return "color:#26a69a;font-weight:bold"
                if abs(v) < exit_z: return "color:#FF9800"
                return ""

            st.dataframe(
                tbl.style
                .format({"가격": "{:,.2f}", "Z-score": "{:.4f}"})
                .map(_color_z2, subset=["Z-score"]),
                use_container_width=True,
            )
