"""
백테스팅 페이지.
사전 정의된 5가지 기술적 전략을 KOSPI/NASDAQ 종목에 적용하고
성과 지표, 누적 수익률 차트, 월별 히트맵, 거래 기록을 비교한다.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analysis.backtest.engine import BacktestEngine, BacktestResult, benchmark_equity
from analysis.backtest.strategies import STRATEGIES, generate_signals
from data.collectors.price_collector import PriceCollector
from ui.components import render_clean_table
from utils.ticker_utils import resolve_ticker as _resolve, is_kr as _is_kr, get_display_name

# ── Constants ─────────────────────────────────────────────────────────────────
PERIOD_OPTIONS = {"1년": ("2y", 1), "3년": ("4y", 3), "5년": ("6y", 5)}

STRATEGY_COLORS = {
    "rsi_reversal": "#FF9800",
    "macd_cross":   "#2196F3",
    "bb_reversal":  "#9C27B0",
    "ma_cross":     "#26a69a",
    "scalping":     "#ef5350",
}
BENCHMARK_COLOR = "#aaaaaa"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _benchmark_ticker(ticker: str) -> str:
    return "^KS11" if _is_kr(ticker) else "QQQ"


def _benchmark_name(ticker: str) -> str:
    return "KOSPI (^KS11)" if _is_kr(ticker) else "NASDAQ (QQQ)"


# ── Metric rating ─────────────────────────────────────────────────────────────

def _rate(metric: str, value: float) -> tuple[str, str]:
    """Returns (emoji, short label) rating for a metric value."""
    rules: dict[str, list[tuple[float, float, str, str]]] = {
        "total_return": [
            (-1e9, 0,  "🔴", "손실"),
            (0,    20, "🟡", "보통"),
            (20,   1e9,"🟢", "좋음"),
        ],
        "cagr": [
            (-1e9, 5,  "🔴", "5% 미만"),
            (5,    15, "🟡", "5~15%"),
            (15,   1e9,"🟢", "15% 초과"),
        ],
        "mdd": [
            (-1e9, -30,"🔴", "-30% 이하"),
            (-30,  -15,"🟡", "-30~-15%"),
            (-15,  0,  "🟢", "-15% 이상"),
        ],
        "sharpe": [
            (-1e9, 0,  "🔴", "0 미만"),
            (0,    1,  "🟡", "0~1"),
            (1,    1e9,"🟢", "1 초과"),
        ],
        "win_rate": [
            (-1e9, 40, "🔴", "40% 미만"),
            (40,   55, "🟡", "40~55%"),
            (55,   1e9,"🟢", "55% 초과"),
        ],
    }
    for lo, hi, emoji, label in rules.get(metric, []):
        if lo <= value < hi:
            return emoji, label
    return "—", "—"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_prices(ticker: str, fetch_period: str) -> pd.DataFrame:
    return PriceCollector().fetch(ticker, period=fetch_period)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_benchmark(bm_ticker: str, fetch_period: str) -> pd.DataFrame:
    return PriceCollector().fetch(bm_ticker, period=fetch_period)


@st.cache_data(ttl=1800, show_spinner=False)
def _run_backtest(
    ticker: str,
    fetch_period: str,
    test_years: int,
    strategy_id: str,
    initial_capital: float,
    commission: float,
    slippage: float,
) -> BacktestResult | None:
    df = _load_prices(ticker, fetch_period)
    if df.empty or len(df) < 30:
        return None

    # Trim to the actual test period (extra data used for indicator warmup)
    cutoff = df.index[-1] - pd.DateOffset(years=test_years)
    df_test = df[df.index >= cutoff].copy()
    if len(df_test) < 20:
        return None

    signals = generate_signals(df_test, strategy_id)   # indicators computed on trimmed df

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
    )
    result = engine.run(df_test, signals, STRATEGIES[strategy_id].name, ticker)
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 백테스팅 설정")

    raw_input = st.text_input(
        "종목 코드 또는 한글명",
        value="005930.KS",
        placeholder="예: 005930.KS · 삼성전자 · AAPL",
    )

    period_label = st.selectbox("테스트 기간", list(PERIOD_OPTIONS.keys()), index=0)
    fetch_period, test_years = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("전략 선택")
    selected_ids = st.multiselect(
        "전략 (복수 선택 가능)",
        options=list(STRATEGIES.keys()),
        default=["rsi_reversal", "macd_cross"],
        format_func=lambda k: STRATEGIES[k].name,
    )

    st.divider()
    st.subheader("거래 파라미터")
    initial_capital = st.number_input(
        "초기 투자금 (원)",
        min_value=100_000,
        max_value=1_000_000_000,
        value=10_000_000,
        step=1_000_000,
        format="%d",
    )
    commission = st.number_input(
        "수수료 (%/회)",
        min_value=0.0, max_value=1.0, value=0.015, step=0.005, format="%.3f",
    )
    slippage = st.number_input(
        "슬리피지 (%/회)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05, format="%.2f",
    )

    st.divider()
    run_btn = st.button("🚀 백테스팅 실행", type="primary", use_container_width=True)

    with st.expander("전략 설명"):
        for cfg in STRATEGIES.values():
            st.markdown(f"**{cfg.name}**  \n{cfg.description}")


# ── Main ──────────────────────────────────────────────────────────────────────
ticker = _resolve(raw_input)
is_kr  = _is_kr(ticker)
company = get_display_name(ticker)

st.title("📊 백테스팅")
st.caption(
    f"{company} ({ticker})  ·  {period_label} 테스트  ·  "
    f"초기자금 ₩{initial_capital:,.0f}  ·  수수료 {commission:.3f}%  ·  슬리피지 {slippage:.2f}%"
)

if not run_btn and "bt_results" not in st.session_state:
    for cfg in STRATEGIES.values():
        st.info(
            f"**{cfg.name}**: {cfg.description}  \n"
            f"파라미터: {', '.join(f'{k}={v}' for k, v in cfg.params.items())}"
        )
    st.stop()

# ── Run (or restore from session state) ───────────────────────────────────────
if run_btn:
    if not selected_ids:
        st.warning("전략을 하나 이상 선택하세요.")
        st.stop()

    results: dict[str, BacktestResult] = {}
    errors:  list[str] = []

    prog = st.progress(0, text="백테스팅 실행 중…")
    for i, sid in enumerate(selected_ids):
        prog.progress((i + 1) / len(selected_ids), text=f"{STRATEGIES[sid].name} 계산 중…")
        r = _run_backtest(ticker, fetch_period, test_years, sid,
                          initial_capital, commission, slippage)
        if r is None:
            errors.append(f"{STRATEGIES[sid].name}: 데이터 부족")
        else:
            results[sid] = r
    prog.empty()

    # Benchmark
    bm_ticker = _benchmark_ticker(ticker)
    bm_df     = _load_benchmark(bm_ticker, fetch_period)

    if not bm_df.empty and results:
        first_eq = next(iter(results.values())).equity
        cutoff   = first_eq.index[0]
        bm_df_test = bm_df[bm_df.index >= cutoff].copy()
        bm_eq      = benchmark_equity(bm_df_test, initial_capital)
        bm_ret     = (float(bm_eq.iloc[-1]) / initial_capital - 1) * 100
        # Attach alpha
        for r in results.values():
            r.alpha = r.total_return - bm_ret
    else:
        bm_eq  = pd.Series(dtype=float)
        bm_ret = 0.0

    st.session_state["bt_results"]  = results
    st.session_state["bt_bm_eq"]    = bm_eq
    st.session_state["bt_bm_ret"]   = bm_ret
    st.session_state["bt_bm_name"]  = _benchmark_name(ticker)
    st.session_state["bt_errors"]   = errors

results  = st.session_state.get("bt_results", {})
bm_eq    = st.session_state.get("bt_bm_eq",   pd.Series(dtype=float))
bm_ret   = st.session_state.get("bt_bm_ret",  0.0)
bm_name  = st.session_state.get("bt_bm_name", "벤치마크")
errors   = st.session_state.get("bt_errors",  [])

for e in errors:
    st.warning(e)

if not results:
    st.error("결과를 계산할 수 없습니다. 종목 코드와 기간을 확인하세요.")
    st.stop()

tab_summary, tab_risk, tab_heatmap, tab_trades = st.tabs([
    "📊 성과 요약", "📉 리스크 분석", "📅 월별 수익률", "📋 거래 기록"
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: 성과 요약
# ═══════════════════════════════════════════════════════════════════════════════
with tab_summary:

    # ── 전략 비교 테이블 ──────────────────────────────────────────────────────
    st.subheader("전략별 성과 비교")

    rows = []
    for sid, r in results.items():
        r_emoji, _  = _rate("total_return", r.total_return)
        c_emoji, _  = _rate("cagr",         r.cagr)
        m_emoji, _  = _rate("mdd",          r.mdd)
        s_emoji, _  = _rate("sharpe",       r.sharpe)
        w_emoji, _  = _rate("win_rate",     r.win_rate * 100)

        rows.append({
            "전략":          r.strategy_name,
            "총수익률":      f"{r_emoji} {r.total_return:+.1f}%",
            "CAGR":         f"{c_emoji} {r.cagr:+.1f}%",
            "MDD":          f"{m_emoji} {r.mdd:.1f}%",
            "샤프":          f"{s_emoji} {r.sharpe:.2f}",
            "승률":          f"{w_emoji} {r.win_rate*100:.0f}%",
            "거래 수":       r.n_trades,
            "알파(벤치마크대비)": f"{r.alpha:+.1f}%",
        })

    # Benchmark row
    bm_emoji, _ = _rate("total_return", bm_ret)
    rows.append({
        "전략":          f"📌 {bm_name} (매수보유)",
        "총수익률":      f"{bm_emoji} {bm_ret:+.1f}%",
        "CAGR":         "—",
        "MDD":          "—",
        "샤프":          "—",
        "승률":          "—",
        "거래 수":       "—",
        "알파(벤치마크대비)": "0.0%",
    })

    render_clean_table(pd.DataFrame(rows), judgment_col=["총수익률", "CAGR", "MDD", "샤프", "승률"])

    # ── 평가 기준 범례 ────────────────────────────────────────────────────────
    with st.expander("📖 평가 기준"):
        criteria_rows = [
            {"지표": "총수익률", "🟢 좋음": "+20% 이상", "🟡 보통": "0~+20%", "🔴 나쁨": "마이너스"},
            {"지표": "CAGR",   "🟢 좋음": "15% 이상",  "🟡 보통": "5~15%",  "🔴 나쁨": "5% 미만"},
            {"지표": "MDD",    "🟢 좋음": "-15% 이상", "🟡 보통": "-15~-30%", "🔴 나쁨": "-30% 미만"},
            {"지표": "샤프",   "🟢 좋음": "1.0 이상",  "🟡 보통": "0~1.0",   "🔴 나쁨": "마이너스"},
            {"지표": "승률",   "🟢 좋음": "55% 이상",  "🟡 보통": "40~55%",  "🔴 나쁨": "40% 미만"},
        ]
        render_clean_table(pd.DataFrame(criteria_rows))
        st.markdown(
            "**CAGR**: 연평균 복리 수익률  ·  "
            "**MDD**: 최대 낙폭 (고점 대비 최저점)  ·  "
            "**샤프**: 위험 대비 수익 (높을수록 효율적)  ·  "
            "**알파**: 벤치마크 대비 초과 수익"
        )

    st.divider()

    # ── 누적 수익률 차트 ──────────────────────────────────────────────────────
    st.subheader("누적 수익률 추이")

    fig_eq = go.Figure()

    # Strategies
    for sid, r in results.items():
        norm = r.equity / float(r.equity.iloc[0]) * 100
        fig_eq.add_trace(go.Scatter(
            x=norm.index, y=norm,
            name=r.strategy_name,
            line=dict(color=STRATEGY_COLORS.get(sid, "#888"), width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
        ))

    # Benchmark
    if not bm_eq.empty:
        norm_bm = bm_eq / float(bm_eq.iloc[0]) * 100
        fig_eq.add_trace(go.Scatter(
            x=norm_bm.index, y=norm_bm,
            name=bm_name,
            line=dict(color=BENCHMARK_COLOR, width=1.5, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>벤치마크: %{y:.1f}<extra></extra>",
        ))

    fig_eq.add_hline(y=100, line_dash="dot", line_color="rgba(128,128,128,0.3)", line_width=1)
    fig_eq.update_layout(
        height=420,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        yaxis_title="누적 수익 (시작=100)",
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_eq, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: 리스크 분석
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:

    st.subheader("최대 낙폭(MDD) 분석")

    # MDD underwater chart
    fig_mdd = go.Figure()
    for sid, r in results.items():
        dd_pct = r.drawdown * 100
        fig_mdd.add_trace(go.Scatter(
            x=dd_pct.index, y=dd_pct,
            name=r.strategy_name,
            fill="tozeroy",
            fillcolor=STRATEGY_COLORS.get(sid, "#888").replace("#", "rgba(") + "40)" if False
                      else f"rgba({int(STRATEGY_COLORS.get(sid,'#888888')[1:3],16)},"
                           f"{int(STRATEGY_COLORS.get(sid,'#888888')[3:5],16)},"
                           f"{int(STRATEGY_COLORS.get(sid,'#888888')[5:7],16)},0.18)",
            line=dict(color=STRATEGY_COLORS.get(sid, "#888"), width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra>%{fullData.name}</extra>",
        ))

    fig_mdd.add_hline(y=0, line_color="rgba(128,128,128,0.3)", line_width=1)
    fig_mdd.update_layout(
        height=350,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        yaxis_title="낙폭 (%)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_mdd, width="stretch")

    # MDD 해석
    st.divider()
    st.subheader("리스크 지표 상세")

    risk_rows = []
    for sid, r in results.items():
        m_emoji, m_label = _rate("mdd",    r.mdd)
        s_emoji, s_label = _rate("sharpe", r.sharpe)

        avg_hold = (
            sum(t.holding_days for t in r.trades) / len(r.trades)
            if r.trades else 0
        )
        risk_rows.append({
            "전략":         r.strategy_name,
            "MDD":         f"{m_emoji} {r.mdd:.2f}%",
            "MDD 평가":    m_label,
            "샤프 비율":   f"{s_emoji} {r.sharpe:.3f}",
            "샤프 평가":   s_label,
            "최대 손실 거래": f"{min((t.return_pct for t in r.trades), default=0):+.2f}%",
            "최대 이익 거래": f"{max((t.return_pct for t in r.trades), default=0):+.2f}%",
            "평균 보유 기간": f"{avg_hold:.0f}일",
        })

    render_clean_table(pd.DataFrame(risk_rows), judgment_col=["MDD", "샤프 비율"])

    st.caption(
        "**MDD 해석**: 투자 기간 중 최고점 대비 최대 하락폭. "
        "-20% MDD는 1,000만원 투자 시 최악의 경우 200만원 평가손실 발생을 의미합니다.  \n"
        "**샤프 비율**: 변동성 1단위당 수익률. 1.0 이상이면 위험 대비 수익이 우수합니다."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: 월별 수익률 히트맵
# ═══════════════════════════════════════════════════════════════════════════════
with tab_heatmap:

    st.subheader("월별 수익률 히트맵")

    hm_sel = st.selectbox(
        "전략 선택",
        options=list(results.keys()),
        format_func=lambda k: results[k].strategy_name,
        key="hm_sel",
    )
    r = results[hm_sel]
    mo_ret = r.monthly_returns

    if mo_ret.empty or len(mo_ret) < 2:
        st.info("월별 수익률 데이터가 부족합니다 (최소 2개월 필요).")
    else:
        years  = sorted(mo_ret.index.year.unique(), reverse=True)
        months = list(range(1, 13))
        month_labels = ["1월","2월","3월","4월","5월","6월",
                        "7월","8월","9월","10월","11월","12월"]

        matrix = pd.DataFrame(np.nan, index=years, columns=months)
        for dt, val in mo_ret.items():
            if dt.year in matrix.index and dt.month in matrix.columns:
                matrix.loc[dt.year, dt.month] = val

        z    = matrix.values.tolist()
        text = [
            [f"{v:+.1f}%" if not np.isnan(v) else "" for v in row]
            for row in matrix.values
        ]

        fig_hm = go.Figure(go.Heatmap(
            z=z,
            x=month_labels,
            y=[str(y) for y in years],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "#c62828"],  [0.25, "#e57373"],
                [0.5, "#263238"],
                [0.75,"#66BB6A"],  [1.0,  "#1b5e20"],
            ],
            zmid=0,
            zmin=-15, zmax=15,
            showscale=True,
            colorbar=dict(title="수익률 (%)", tickformat="+.0f"),
            hoverongaps=False,
            hovertemplate="%{y}년 %{x}: %{text}<extra></extra>",
        ))
        fig_hm.update_layout(
            height=max(200, len(years) * 45 + 80),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_hm, width="stretch")

        # Annual summary
        annual = matrix.mean(axis=1).dropna()
        if not annual.empty:
            ann_df = pd.DataFrame({
                "연도":     annual.index.astype(str),
                "월평균 수익률": annual.map(lambda x: f"{x:+.2f}%"),
                "연간 누적":   matrix.apply(
                    lambda row: f"{(np.prod(1 + row.dropna() / 100) - 1) * 100:+.1f}%",
                    axis=1,
                ).values,
            })
            render_clean_table(ann_df, judgment_col=["월평균 수익률", "연간 누적"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: 거래 기록
# ═══════════════════════════════════════════════════════════════════════════════
with tab_trades:

    st.subheader("거래 기록")

    tr_sel = st.selectbox(
        "전략 선택",
        options=list(results.keys()),
        format_func=lambda k: results[k].strategy_name,
        key="tr_sel",
    )
    r = results[tr_sel]

    if not r.trades:
        st.info(
            "해당 전략에서 완결된 거래가 없습니다.  \n"
            "기간 내에 신호 조건이 충족되지 않았거나, 매수 후 매도 신호 없이 기간이 종료됐을 수 있습니다.  \n"
            "테스트 기간을 늘리거나 다른 전략을 시도해보세요."
        )
    else:
        win_n  = sum(1 for t in r.trades if t.return_pct > 0)
        loss_n = len(r.trades) - win_n

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("전체 거래",  f"{len(r.trades)}회")
            m2.metric("승리",       f"{win_n}회 ({r.win_rate*100:.0f}%)")
            m3.metric("패배",       f"{loss_n}회 ({(1-r.win_rate)*100:.0f}%)")
            m4.metric("총 손익",
                      f"₩{sum(t.profit for t in r.trades):+,.0f}" if is_kr
                      else f"${sum(t.profit for t in r.trades):+,.2f}")

        st.divider()

        trade_rows = []
        for i, t in enumerate(r.trades, 1):
            trade_rows.append({
                "#":       i,
                "매수일":  str(t.entry_date)[:10],
                "매수가":  f"₩{t.entry_price:,.0f}" if is_kr else f"${t.entry_price:,.2f}",
                "매도일":  str(t.exit_date)[:10],
                "매도가":  f"₩{t.exit_price:,.0f}" if is_kr else f"${t.exit_price:,.2f}",
                "보유(일)": t.holding_days,
                "수익률":  t.return_pct,
                "손익":    t.profit,
            })
        df_trades = pd.DataFrame(trade_rows)
        fmt_price = "₩{:,.0f}" if is_kr else "${:,.2f}"
        df_trades["수익률"] = df_trades["수익률"].map(lambda v: f"{v:+.2f}%")
        df_trades["손익"]   = df_trades["손익"].map(fmt_price.format)

        render_clean_table(df_trades, judgment_col=["수익률", "손익"])

        # 수익률 분포 bar chart
        st.divider()
        st.subheader("거래별 수익률 분포")

        rets = [t.return_pct for t in r.trades]
        bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in rets]
        fig_tr = go.Figure(go.Bar(
            x=list(range(1, len(rets) + 1)),
            y=rets,
            marker_color=bar_colors,
            text=[f"{v:+.2f}%" for v in rets],
            textposition="outside",
        ))
        fig_tr.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
        fig_tr.update_layout(
            height=280,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="거래 순서",
            yaxis_title="수익률 (%)",
        )
        st.plotly_chart(fig_tr, width="stretch")
