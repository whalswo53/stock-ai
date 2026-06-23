"""
포트폴리오 관리 페이지.
SQLite 저장 · 직접 입력 · CSV 업로드 · P&L 대시보드 · AI 분석 프롬프트

v3 변경:
  - 종목 추가 시 yfinance 유효성 검증 + 종목명 자동 설정
  - 실시간 USD/KRW 환율 표시, 미국 종목 USD 단위
  - 전체 수익률 환율 반영 원화 통합 계산
  - 보유 중 탭 카드별 수정 expander 추가 (그룹 변경 포함)
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from data.collectors.market_sentiment import MarketSentimentCollector, prompt_snippet
from data.collectors.price_collector import PriceCollector
from portfolio.manager import (
    GROUP_ACCUMULATING, GROUP_HOLDING, PortfolioManager, get_usd_krw_rate
)
from utils.ticker_utils import resolve_ticker as _resolve_base, is_kr, fmt_price, get_display_name
from utils.clipboard import copy_button

# ── Init ──────────────────────────────────────────────────────────────────────
pm = PortfolioManager()

GROUP_LABEL = {GROUP_HOLDING: "보유 중", GROUP_ACCUMULATING: "모으는 중"}
GROUP_ICON  = {GROUP_HOLDING: "💼",      GROUP_ACCUMULATING: "🌱"}

ACCUM_PERIOD_LABEL = {"daily": "매일", "weekly": "매주", "monthly": "매월"}
ACCUM_PERIOD_VAL   = {"매일": "daily", "매주": "weekly", "매월": "monthly"}
ACCUM_TYPE_LABEL   = {"amount": "금액 기준", "quantity": "수량 기준"}
ACCUM_TYPE_VAL     = {"금액 기준": "amount", "수량 기준": "quantity"}
CURRENCY_SYMBOL    = {"KRW": "₩", "USD": "$"}

UP   = "#26a69a"
DOWN = "#ef5350"

_PORTFOLIO_GRADIENT = "linear-gradient(135deg,#1565C0,#0D47A1)"


def _resolve(raw: str) -> str:
    return _resolve_base(raw)


def _display_name(h: dict) -> str:
    return h["name"] or get_display_name(h["ticker"])


def _plan_summary(h: dict) -> str:
    period = ACCUM_PERIOD_LABEL.get(h.get("accum_period", ""), "")
    val    = float(h.get("accum_value", 0) or 0)
    typ    = h.get("accum_type", "")
    if not period or not val:
        return ""
    if typ == "amount":
        currency = h.get("accum_currency") or "KRW"
        sym = CURRENCY_SYMBOL.get(currency, "₩")
        return f"{period} {sym}{val:,.0f}씩 적립"
    if typ == "quantity":
        return f"{period} {val:,.0f}주씩 적립"
    return ""


# ── Cached data helpers ───────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price(ticker: str) -> float:
    try:
        df = PriceCollector().fetch(ticker, period="5d")
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sector(ticker: str) -> str:
    try:
        info = PriceCollector().get_info(ticker)
        return info.get("sector") or info.get("industry") or "기타"
    except Exception:
        return "기타"


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_benchmark_return(bm_ticker: str) -> float | None:
    try:
        hist = yf.Ticker(bm_ticker).history(period="1y")
        if len(hist) < 2:
            return None
        return float((hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg() -> str:
    try:
        return prompt_snippet(MarketSentimentCollector().fetch())
    except Exception:
        return ""


@st.cache_data(ttl=3600, show_spinner=False)
def _validate_and_fetch_name(ticker: str) -> tuple[bool, str]:
    """yfinance 티커 유효성 확인 + 종목명 반환 (valid, longName).

    history() 기반으로 유효성을 먼저 판단하고, info는 이름 조회에만 사용.
    소형/중형주처럼 info가 불완전해도 거래 데이터가 있으면 유효로 처리.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="5d")
        if hist.empty:
            return False, ""
        try:
            info = t.info
            name = info.get("longName") or info.get("shortName") or ""
        except Exception:
            name = ""
        return True, name
    except Exception:
        return False, ""


# ── Portfolio P&L calculation ─────────────────────────────────────────────────

def build_pnl_df(holdings: list[dict], usd_krw: float = 1300.0) -> pd.DataFrame:
    rows = []
    for h in holdings:
        if h["group_type"] == GROUP_ACCUMULATING:
            qty, avg_cost = pm.calc_accumulated(h["id"])
        else:
            qty      = float(h["quantity"])
            avg_cost = float(h["avg_cost"])

        ticker      = h["ticker"]
        kr          = is_kr(ticker)
        fx          = 1.0 if kr else usd_krw
        cp          = _fetch_price(ticker)
        total_cost  = qty * avg_cost
        total_value = qty * cp
        pnl         = total_value - total_cost
        ret_pct     = ((cp - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0

        rows.append({
            "_id":          h["id"],
            "_ticker":      ticker,
            "_group":       h["group_type"],
            "_target_qty":  h.get("target_qty"),
            "_is_kr":       kr,
            "_fx":          fx,
            "종목":          _display_name(h),
            "티커":          ticker,
            "그룹":          GROUP_LABEL.get(h["group_type"], h["group_type"]),
            "수량":          qty,
            "목표수량":       h.get("target_qty"),
            "평균매입가":     avg_cost,
            "현재가":        cp,
            "매입총액":      total_cost,
            "매입총액_KRW":  total_cost * fx,
            "평가금액":      total_value,
            "평가금액_KRW":  total_value * fx,
            "평가손익":      pnl,
            "평가손익_KRW":  pnl * fx,
            "수익률(%)":    round(ret_pct, 2),
            "섹터":          h.get("sector") or "",
            "메모":          h.get("notes") or "",
        })
    return pd.DataFrame(rows)


# ── Sidebar: Add holding + CSV upload ─────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 포트폴리오")

    # ── Add holding form ──────────────────────────────────────────────────────
    with st.expander("➕ 종목 추가", expanded=pm.count() == 0):
        # 폼 밖에 배치 → 값 변경 시 즉시 리렌더링
        group_sel = st.radio(
            "구분",
            [GROUP_HOLDING, GROUP_ACCUMULATING],
            format_func=lambda g: f"{GROUP_ICON[g]} {GROUP_LABEL[g]}",
            horizontal=True,
            key="add_group_sel",
        )

        _period   = "매주"
        _type     = "금액 기준"
        _currency = "KRW"
        if group_sel == GROUP_ACCUMULATING:
            st.caption("**적립 계획 (선택)**")
            col_p, col_t = st.columns(2)
            _period = col_p.radio("적립 주기", ["매일", "매주", "매월"], horizontal=True, key="add_period")
            _type   = col_t.radio("적립 방식", ["금액 기준", "수량 기준"], horizontal=True, key="add_type")
            if _type == "금액 기준":
                _cur_raw  = st.radio(
                    "통화", ["KRW (원)", "USD (달러)"],
                    horizontal=True, key="add_currency",
                )
                _currency = "KRW" if "KRW" in _cur_raw else "USD"

        with st.form("add_holding", clear_on_submit=True):
            raw_input = st.text_input(
                "종목 코드 또는 한글명",
                placeholder="예: 005930.KS · 삼성전자 · NVDA",
            )

            col_q, col_c = st.columns(2)
            qty_input  = col_q.text_input("수량",       placeholder="예: 10")
            cost_input = col_c.text_input("평균매입가", placeholder="예: 50000")

            if group_sel == GROUP_ACCUMULATING:
                target_qty_input: float | None = st.number_input(
                    "목표 수량 (선택)", min_value=0.0, step=1.0, value=0.0
                ) or None
                if _type == "금액 기준":
                    _ph = "예: 100,000 (원)" if _currency == "KRW" else "예: 100 (달러)"
                else:
                    _ph = "예: 2 (주)"
                accum_value_str = st.text_input("적립 금액/수량", placeholder=_ph)
                try:
                    accum_value_input = float(accum_value_str.strip()) if accum_value_str.strip() else 0.0
                except ValueError:
                    accum_value_input = 0.0
                accum_period_input   = ACCUM_PERIOD_VAL[_period]
                accum_type_input     = ACCUM_TYPE_VAL[_type]
                accum_currency_input = _currency if _type == "금액 기준" else "KRW"
            else:
                target_qty_input     = None
                accum_period_input   = ""
                accum_type_input     = ""
                accum_value_input    = 0.0
                accum_currency_input = "KRW"

            notes_input = st.text_input("메모 (선택)", placeholder="예: 장기 보유")
            submitted   = st.form_submit_button("추가하기", type="primary", use_container_width=True)

            if submitted:
                try:
                    qty_f = float(qty_input.strip()) if qty_input.strip() else 0.0
                except ValueError:
                    qty_f = 0.0
                try:
                    cost_f = float(cost_input.strip()) if cost_input.strip() else 0.0
                except ValueError:
                    cost_f = 0.0

                if not raw_input.strip():
                    st.error("종목 코드 또는 이름을 입력하세요.")
                elif group_sel == GROUP_HOLDING and (qty_f <= 0 or cost_f <= 0):
                    st.error("'보유 중'은 수량과 평균매입가를 입력하세요.")
                else:
                    ticker = _resolve(raw_input)
                    with st.spinner(f"{ticker} 확인 중…"):
                        valid, yf_name = _validate_and_fetch_name(ticker)
                    if not valid:
                        st.error(
                            f"종목을 찾을 수 없습니다: **{ticker}**\n\n"
                            "티커 코드를 직접 입력하거나 `.KS` / `.KQ` 접미사를 확인하세요."
                        )
                    else:
                        mapped = get_display_name(ticker)
                        name   = mapped if mapped != ticker else (yf_name or raw_input.strip())
                        pm.add_holding(
                            ticker=ticker,
                            name=name,
                            quantity=qty_f,
                            avg_cost=cost_f,
                            group_type=group_sel,
                            target_qty=target_qty_input,
                            accum_period=accum_period_input,
                            accum_type=accum_type_input,
                            accum_value=accum_value_input,
                            accum_currency=accum_currency_input,
                            notes=notes_input,
                        )
                        st.cache_data.clear()
                        st.success(f"✅ {name} ({ticker}) 추가 완료")
                        st.rerun()

    # ── CSV upload ────────────────────────────────────────────────────────────
    with st.expander("📂 CSV 업로드"):
        st.caption("컬럼: 종목코드, 수량, 평균매입가 (+ 선택: 종목명, 그룹, 목표수량, 적립주기, 적립방식, 적립금액, 메모)")
        uploaded = st.file_uploader("CSV 파일 선택", type="csv", label_visibility="collapsed")
        if uploaded:
            content = uploaded.read().decode("utf-8-sig")
            n, errs = pm.import_csv(content)
            st.cache_data.clear()
            if n:
                st.success(f"{n}개 종목 가져오기 완료")
            for e in errs:
                st.warning(e)
            if n:
                st.rerun()
        st.download_button(
            "📄 CSV 템플릿 다운로드",
            data=PortfolioManager.csv_template(),
            file_name="portfolio_template.csv",
            mime="text/csv; charset=utf-8-sig",
            use_container_width=True,
        )

    # ── Delete ────────────────────────────────────────────────────────────────
    holdings_all = pm.get_all()
    if holdings_all:
        st.divider()
        with st.expander("🗑️ 종목 삭제"):
            del_options = {
                f"{_display_name(h)} ({h['ticker']})": h["id"]
                for h in holdings_all
            }
            del_sel = st.selectbox("삭제할 종목", list(del_options.keys()), label_visibility="collapsed")
            if st.button("삭제", type="secondary", use_container_width=True):
                pm.delete_holding(del_options[del_sel])
                st.cache_data.clear()
                st.success(f"삭제 완료: {del_sel}")
                st.rerun()

    # ── Quick stats ───────────────────────────────────────────────────────────
    st.divider()
    n_acc = len([h for h in holdings_all if h["group_type"] == GROUP_ACCUMULATING])
    n_hld = len([h for h in holdings_all if h["group_type"] == GROUP_HOLDING])
    c1, c2 = st.columns(2)
    c1.metric("모으는 중", n_acc)
    c2.metric("보유 중",   n_hld)


# ── Main content ──────────────────────────────────────────────────────────────
st.title("💼 포트폴리오 관리")

usd_krw = get_usd_krw_rate()
st.caption(f"💱 현재 환율: 1 USD = ₩{usd_krw:,.0f}")

if not holdings_all:
    st.info(
        "👈 사이드바에서 **종목 추가** 또는 **CSV 업로드**로 보유 종목을 등록하세요.\n\n"
        "예시: `삼성전자`, `005930.KS`, `AAPL`, `엔비디아`"
    )
    st.stop()

# ── 자동 적립 매수 내역 기록 (세션당 1회) ────────────────────────────────────
if "auto_record_done" not in st.session_state:
    with st.spinner("자동 적립 기록 확인 중…"):
        _auto_results = pm.auto_record_purchases(usd_krw=usd_krw)
    st.session_state["auto_record_done"] = True
    if _auto_results:
        st.cache_data.clear()
        for _ar_name, _ar_ticker, _ar_n in _auto_results:
            st.toast(f"✅ {_ar_name} ({_ar_ticker}): {_ar_n}건 자동 적립 기록됨")

with st.spinner("현재가 조회 중…"):
    df_all = build_pnl_df(holdings_all, usd_krw)

df_acc = df_all[df_all["_group"] == GROUP_ACCUMULATING].copy()
df_hld = df_all[df_all["_group"] == GROUP_HOLDING].copy()

tab_dash, tab_acc, tab_hld, tab_ai = st.tabs([
    "📊 대시보드",
    f"🌱 모으는 중 ({len(df_acc)})",
    f"💼 보유 중 ({len(df_hld)})",
    "🤖 AI 분석",
])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    # KRW 환산 기준으로 전체 합산 (미국 종목도 원화로 통합)
    total_cost  = df_all["매입총액_KRW"].sum()
    total_value = df_all["평가금액_KRW"].sum()
    total_pnl   = total_value - total_cost
    total_ret   = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("총 평가금액",  f"₩{total_value:,.0f}" if total_value < 1e8 else f"₩{total_value/1e8:.2f}억")
    m2.metric("총 매입금액",  f"₩{total_cost:,.0f}"  if total_cost  < 1e8 else f"₩{total_cost/1e8:.2f}억")
    m3.metric("평가손익",     f"₩{total_pnl:+,.0f}", delta_color="normal")
    m4.metric("전체 수익률",  f"{total_ret:+.2f}%",   delta=f"{total_pnl:+,.0f}")

    st.divider()
    st.subheader("종목별 현황")

    display_cols = ["종목", "티커", "그룹", "수량", "평균매입가", "현재가",
                    "평가금액", "평가손익", "수익률(%)"]
    df_show = df_all[display_cols].copy()

    def _color_ret(val: float) -> str:
        return (f"color:{UP};font-weight:bold" if val > 0
                else f"color:{DOWN};font-weight:bold" if val < 0 else "")

    st.dataframe(
        df_show.style
        .map(_color_ret, subset=["평가손익", "수익률(%)"])
        .format({
            "평균매입가": "{:,.4g}",
            "현재가":    "{:,.4g}",
            "평가금액":  "{:,.2f}",
            "평가손익":  "{:+,.2f}",
            "수익률(%)": "{:+.2f}",
        }),
        width="stretch",
        hide_index=True,
    )
    st.caption("※ 평균매입가/현재가는 종목 원래 통화 기준 (KRW: 원, USD: 달러)")

    st.divider()
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("섹터별 비중")
        sector_map: dict[str, str] = {}
        for _, row in df_all.iterrows():
            sec = row["섹터"] or _fetch_sector(row["_ticker"])
            sector_map[row["티커"]] = sec
        sector_series = df_all["_ticker"].map(sector_map).fillna("기타")
        sector_val    = df_all.groupby(sector_series)["평가금액_KRW"].sum().reset_index()
        sector_val.columns = ["섹터", "평가금액"]
        fig_pie = px.pie(
            sector_val, names="섹터", values="평가금액",
            color_discrete_sequence=px.colors.qualitative.Set3, hole=0.35,
        )
        fig_pie.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig_pie, width="stretch")

    with col_bar:
        st.subheader("종목별 수익률")
        df_ret  = df_all[["종목", "수익률(%)"]].sort_values("수익률(%)")
        colors  = [UP if v >= 0 else DOWN for v in df_ret["수익률(%)"]]
        fig_bar = go.Figure(go.Bar(
            x=df_ret["수익률(%)"], y=df_ret["종목"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}%" for v in df_ret["수익률(%)"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=320, margin=dict(l=0, r=60, t=20, b=10),
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="수익률 (%)", yaxis_title="",
        )
        st.plotly_chart(fig_bar, width="stretch")

    # ── Benchmark comparison ──────────────────────────────────────────────────
    st.divider()
    st.subheader("벤치마크 비교")

    with st.spinner("벤치마크 조회 중…"):
        bm_kospi  = _fetch_benchmark_return("^KS11")
        bm_nasdaq = _fetch_benchmark_return("^NDX")

    bm_rows = [
        ("내 포트폴리오", total_ret),
        ("KOSPI (^KS11)", bm_kospi),
        ("NASDAQ (^NDX)", bm_nasdaq),
    ]
    unavailable = [label for label, v in bm_rows[1:] if v is None]
    bm_rows_valid = [(label, v) for label, v in bm_rows if v is not None]

    if bm_rows_valid:
        bm_labels  = [r[0] for r in bm_rows_valid]
        bm_vals    = [r[1] for r in bm_rows_valid]
        bm_colors  = [UP if v >= 0 else DOWN for v in bm_vals]
        if bm_labels[0] == "내 포트폴리오":
            bm_colors[0] = "#7E57C2"
        fig_bm = go.Figure(go.Bar(
            x=bm_labels, y=bm_vals, marker_color=bm_colors,
            text=[f"{v:+.2f}%" for v in bm_vals],
            textposition="outside",
        ))
        fig_bm.add_hline(y=0, line_color="rgba(128,128,128,0.4)", line_width=1)
        fig_bm.update_layout(
            height=280, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="수익률 (%)", xaxis_title="",
        )
        st.plotly_chart(fig_bm, width="stretch")

    if unavailable:
        st.caption(f"⚠️ 데이터를 가져올 수 없습니다: {', '.join(unavailable)}")
    st.caption("기준일: 최근 1년 (1년치 첫 거래일 → 현재)")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: ACCUMULATING (모으는 중)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_acc:
    if df_acc.empty:
        st.info("'모으는 중' 종목이 없습니다. 종목 추가 시 **모으는 중** 그룹을 선택하세요.")
    else:
        acc_total_invested = df_acc["매입총액_KRW"].sum()
        acc_total_value    = df_acc["평가금액_KRW"].sum()
        acc_total_pnl      = acc_total_value - acc_total_invested
        acc_ret            = (acc_total_pnl / acc_total_invested * 100) if acc_total_invested > 0 else 0.0
        am1, am2, am3 = st.columns(3)
        am1.metric("총 투자금액",  f"₩{acc_total_invested:,.0f}")
        am2.metric("총 평가금액",  f"₩{acc_total_value:,.0f}")
        am3.metric("누적 수익률",  f"{acc_ret:+.2f}%")
        st.divider()

        for _, row in df_acc.iterrows():
            hid      = int(row["_id"])
            name     = row["종목"]
            ticker   = row["_ticker"]
            qty      = row["수량"]
            avg_cost = row["평균매입가"]
            cp       = row["현재가"]
            tgt      = row["_target_qty"]
            ret_pct  = row["수익률(%)"]
            pnl      = row["평가손익"]
            invested = row["매입총액"]
            kr       = bool(row["_is_kr"])

            h_data   = pm.get_by_id(hid) or {}
            plan_str = _plan_summary(h_data)

            with st.container(border=True):
                hdr_l, hdr_r = st.columns([4, 1])
                with hdr_l:
                    st.markdown(f"**{name}** `{ticker}`")
                with hdr_r:
                    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
                    if st.button("완료→보유", key=f"move_{hid}", use_container_width=True):
                        pm.move_group(hid, GROUP_HOLDING)
                        st.cache_data.clear()
                        st.success(f"{name} → 보유 중으로 이동")
                        st.rerun()
                    if st.button("분석", key=f"jump_acc_{hid}", use_container_width=True):
                        st.session_state["portfolio_jump_ticker"] = ticker
                        st.switch_page("pages/01_overview.py")

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("총 투자금액",  fmt_price(invested, ticker))
                mc2.metric("평균매입가",   fmt_price(avg_cost, ticker) if avg_cost > 0 else "—")
                mc3.metric("보유 수량",    f"{qty:,.0f}주")
                mc4.metric("현재가",       fmt_price(cp, ticker) if cp > 0 else "—")

                ret_c = UP if ret_pct >= 0 else DOWN
                pnl_str = fmt_price(abs(pnl), ticker)
                st.markdown(
                    f'<span style="color:{ret_c};font-weight:700">'
                    f'수익률 {ret_pct:+.2f}%</span>'
                    f'  <span style="color:{ret_c}">({("+" if pnl >= 0 else "-")}{pnl_str})</span>',
                    unsafe_allow_html=True,
                )

                if tgt and tgt > 0:
                    progress = min(qty / tgt, 1.0)
                    st.progress(progress, text=f"{qty:,.0f} / {tgt:,.0f} 주  ({progress*100:.1f}%)")

                if plan_str:
                    st.caption(f"📅 적립 계획: {plan_str}")

                # ── 매수 내역 ─────────────────────────────────────────────────
                purchases = pm.get_purchases(hid)
                with st.expander(
                    f"📋 매수 내역 ({len(purchases)}건)  —  매수 추가",
                    expanded=False,
                ):
                    with st.form(f"add_buy_{hid}", clear_on_submit=True):
                        fc1, fc2, fc3 = st.columns(3)
                        new_date  = fc1.date_input("매수일", value=date.today(), key=f"bd_{hid}")
                        new_qty   = fc2.number_input("수량 (주)", min_value=0.01, step=1.0, key=f"bq_{hid}")
                        new_price = fc3.number_input(
                            "매수가",
                            min_value=0.0001,
                            step=100.0 if kr else 0.01,
                            key=f"bp_{hid}",
                            help="KRW 종목: 원, USD 종목: 달러",
                        )
                        buy_submitted = st.form_submit_button(
                            "💾 매수 추가", type="primary", use_container_width=True
                        )
                        if buy_submitted:
                            if new_qty <= 0 or new_price <= 0:
                                st.error("수량과 매수가를 입력하세요.")
                            else:
                                pm.add_purchase(
                                    holding_id=hid,
                                    buy_date=str(new_date),
                                    quantity=new_qty,
                                    price=new_price,
                                )
                                st.cache_data.clear()
                                st.success(
                                    f"추가: {new_date}  {new_qty:,.0f}주  "
                                    f"@ {fmt_price(new_price, ticker)}"
                                )
                                st.rerun()

                    if purchases:
                        st.markdown("---")
                        ph1, ph2, ph3, ph4, ph5 = st.columns([2, 1, 2, 2, 1])
                        ph1.caption("매수일"); ph2.caption("수량")
                        ph3.caption("매수가"); ph4.caption("소계"); ph5.caption("")
                        for p in purchases:
                            sub = float(p["quantity"]) * float(p["price"])
                            r1, r2, r3, r4, r5 = st.columns([2, 1, 2, 2, 1])
                            r1.write(p["buy_date"])
                            r2.write(f"{float(p['quantity']):,.0f}주")
                            r3.write(fmt_price(float(p["price"]), ticker))
                            r4.write(fmt_price(sub, ticker))
                            if r5.button("🗑", key=f"del_ph_{p['id']}", help="삭제"):
                                pm.delete_purchase(p["id"])
                                st.cache_data.clear()
                                st.rerun()
                    else:
                        st.caption("아직 매수 내역이 없습니다. 위 폼으로 추가하세요.")

                # ── 정보 수정 expander ─────────────────────────────────────────
                with st.expander("✏️ 정보 수정", expanded=False):
                    with st.form(f"edit_acc_{hid}"):
                        se1, se2 = st.columns(2)
                        new_qty_s  = se1.number_input(
                            "초기 수량 (시드)", value=float(h_data.get("quantity", 0)),
                            min_value=0.0, key=f"seqa_{hid}"
                        )
                        new_cost_s = se2.number_input(
                            "초기 평균매입가 (시드)", value=float(h_data.get("avg_cost", 0)),
                            min_value=0.0, key=f"seca_{hid}"
                        )
                        new_tgt = st.number_input(
                            "목표 수량", value=float(h_data.get("target_qty") or 0),
                            min_value=0.0, key=f"seta_{hid}"
                        )

                        cur_grp  = h_data.get("group_type", GROUP_ACCUMULATING)
                        grp_opts = [GROUP_ACCUMULATING, GROUP_HOLDING]
                        new_grp  = st.radio(
                            "그룹", grp_opts,
                            index=grp_opts.index(cur_grp),
                            format_func=lambda g: f"{GROUP_ICON[g]} {GROUP_LABEL[g]}",
                            horizontal=True, key=f"seg_{hid}",
                        )

                        if new_grp == GROUP_ACCUMULATING:
                            st.caption("**적립 계획**")
                            _cur_p = ACCUM_PERIOD_LABEL.get(h_data.get("accum_period", ""), "매주")
                            _cur_t = ACCUM_TYPE_LABEL.get(h_data.get("accum_type", ""), "금액 기준")
                            ep1, ep2 = st.columns(2)
                            new_per = ep1.radio(
                                "주기", ["매일", "매주", "매월"],
                                index=["매일", "매주", "매월"].index(_cur_p),
                                horizontal=True, key=f"sep_{hid}"
                            )
                            new_typ = ep2.radio(
                                "방식", ["금액 기준", "수량 기준"],
                                index=["금액 기준", "수량 기준"].index(_cur_t),
                                horizontal=True, key=f"set_{hid}"
                            )
                            new_val = st.number_input(
                                "적립 금액/수량",
                                value=float(h_data.get("accum_value", 0) or 0),
                                min_value=0.0,
                                step=10000.0 if "금액" in new_typ else 1.0,
                                key=f"sev_{hid}"
                            )
                            if new_typ == "금액 기준":
                                _cur_c = h_data.get("accum_currency") or ("KRW" if is_kr(ticker) else "USD")
                                _new_c_raw = st.radio(
                                    "통화", ["KRW (원)", "USD (달러)"],
                                    index=0 if _cur_c == "KRW" else 1,
                                    horizontal=True, key=f"sec_{hid}",
                                )
                                _save_currency = "KRW" if "KRW" in _new_c_raw else "USD"
                            else:
                                _save_currency = "KRW"
                            _save_period = ACCUM_PERIOD_VAL[new_per]
                            _save_type   = ACCUM_TYPE_VAL[new_typ]
                            _save_val    = new_val
                        else:
                            _save_period   = ""
                            _save_type     = ""
                            _save_val      = 0.0
                            _save_currency = "KRW"

                        new_note = st.text_input("메모", value=h_data.get("notes", ""), key=f"sen_{hid}")
                        if st.form_submit_button("저장", type="primary"):
                            pm.update_holding(
                                hid,
                                quantity=new_qty_s,
                                avg_cost=new_cost_s,
                                target_qty=new_tgt or None,
                                group_type=new_grp,
                                accum_period=_save_period,
                                accum_type=_save_type,
                                accum_value=_save_val,
                                accum_currency=_save_currency,
                                notes=new_note,
                            )
                            st.cache_data.clear()
                            st.success("수정 완료")
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3: HOLDING (보유 중)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_hld:
    if df_hld.empty:
        st.info("'보유 중' 종목이 없습니다.")
    else:
        hld_cost  = df_hld["매입총액_KRW"].sum()
        hld_value = df_hld["평가금액_KRW"].sum()
        hld_pnl   = hld_value - hld_cost
        hld_ret   = (hld_pnl / hld_cost * 100) if hld_cost > 0 else 0.0
        h1, h2, h3 = st.columns(3)
        h1.metric("평가금액", f"₩{hld_value:,.0f}")
        h2.metric("평가손익", f"₩{hld_pnl:+,.0f}")
        h3.metric("수익률",   f"{hld_ret:+.2f}%")
        st.divider()

        for _, row in df_hld.iterrows():
            hid     = int(row["_id"])
            name    = row["종목"]
            ticker  = row["_ticker"]
            ret_pct = row["수익률(%)"]
            pnl     = row["평가손익"]
            kr      = bool(row["_is_kr"])
            ret_c   = UP if ret_pct >= 0 else DOWN
            h_data  = pm.get_by_id(hid) or {}

            with st.container(border=True):
                c_info, c_btns = st.columns([4, 1])

                with c_info:
                    pnl_str = fmt_price(abs(pnl), ticker)
                    st.markdown(
                        f"**{name}** `{ticker}` — "
                        f"<span style='color:{ret_c};font-weight:700'>{ret_pct:+.2f}%</span>"
                        f"  <span style='color:{ret_c}'>"
                        f"({('+' if pnl >= 0 else '-')}{pnl_str})</span>",
                        unsafe_allow_html=True,
                    )
                    cm1, cm2, cm3, cm4 = st.columns(4)
                    cm1.metric("수량",       f"{row['수량']:,.0f}주")
                    cm2.metric("평균매입가", fmt_price(row["평균매입가"], ticker))
                    cm3.metric("현재가",     fmt_price(row["현재가"], ticker))
                    cm4.metric("평가금액",   fmt_price(row["평가금액"], ticker))
                    reg = (h_data.get("created_at") or "")[:10]
                    if reg:
                        st.caption(f"등록일: {reg}")

                with c_btns:
                    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
                    if st.button("분석", key=f"jump_hld_{hid}", use_container_width=True):
                        st.session_state["portfolio_jump_ticker"] = ticker
                        st.switch_page("pages/01_overview.py")

                # ── 수정 expander (카드별) ────────────────────────────────────
                with st.expander("✏️ 수정", expanded=False):
                    with st.form(f"edit_hld_{hid}"):
                        ec1, ec2 = st.columns(2)
                        new_qty  = ec1.number_input(
                            "수량", value=float(h_data.get("quantity", 0)),
                            min_value=0.0, key=f"heq_{hid}"
                        )
                        new_cost = ec2.number_input(
                            "평균매입가", value=float(h_data.get("avg_cost", 0)),
                            min_value=0.0, key=f"hec_{hid}"
                        )

                        grp_opts = [GROUP_HOLDING, GROUP_ACCUMULATING]
                        new_grp  = st.radio(
                            "그룹", grp_opts,
                            index=0,
                            format_func=lambda g: f"{GROUP_ICON[g]} {GROUP_LABEL[g]}",
                            horizontal=True, key=f"heg_{hid}",
                        )

                        if new_grp == GROUP_ACCUMULATING:
                            st.caption("**적립 계획**")
                            col_p, col_t = st.columns(2)
                            new_per = col_p.radio(
                                "주기", ["매일", "매주", "매월"],
                                horizontal=True, key=f"hep_{hid}"
                            )
                            new_typ = col_t.radio(
                                "방식", ["금액 기준", "수량 기준"],
                                horizontal=True, key=f"het_{hid}"
                            )
                            new_val = st.number_input(
                                "적립 금액/수량", min_value=0.0,
                                step=10000.0 if "금액" in new_typ else 1.0,
                                key=f"hev_{hid}"
                            )
                            if new_typ == "금액 기준":
                                _def_c = "KRW" if is_kr(ticker) else "USD"
                                _new_hc_raw = st.radio(
                                    "통화", ["KRW (원)", "USD (달러)"],
                                    index=0 if _def_c == "KRW" else 1,
                                    horizontal=True, key=f"hec2_{hid}",
                                )
                                _sc = "KRW" if "KRW" in _new_hc_raw else "USD"
                            else:
                                _sc = "KRW"
                            new_tgt: float | None = st.number_input(
                                "목표 수량", min_value=0.0, key=f"hetq_{hid}"
                            ) or None
                            _sp = ACCUM_PERIOD_VAL[new_per]
                            _st_val = ACCUM_TYPE_VAL[new_typ]
                        else:
                            new_val = 0.0
                            new_tgt = None
                            _sp = ""
                            _st_val = ""
                            _sc = "KRW"

                        new_note = st.text_input(
                            "메모", value=h_data.get("notes", ""), key=f"hen_{hid}"
                        )
                        if st.form_submit_button("저장", type="primary"):
                            pm.update_holding(
                                hid,
                                quantity=new_qty,
                                avg_cost=new_cost,
                                group_type=new_grp,
                                accum_period=_sp,
                                accum_type=_st_val,
                                accum_value=new_val,
                                accum_currency=_sc,
                                target_qty=new_tgt,
                                notes=new_note,
                            )
                            st.cache_data.clear()
                            st.success("수정 완료")
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4: AI 분석
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.subheader("🤖 포트폴리오 AI 분석")
    st.caption("전체 보유 종목 현황을 프롬프트로 자동 생성합니다. 복사 후 Claude.ai에 붙여넣어 분석을 받으세요.")

    table_lines = [
        "| 종목 | 티커 | 그룹 | 수량 | 평단가 | 현재가 | 수익률 | 평가손익 |",
        "|------|------|------|------|--------|--------|--------|---------|",
    ]
    for _, row in df_all.iterrows():
        table_lines.append(
            f"| {row['종목']} | {row['_ticker']} | {row['그룹']} | "
            f"{row['수량']:,.0f} | {row['평균매입가']:,.4g} | {row['현재가']:,.4g} | "
            f"{row['수익률(%)']:+.2f}% | {row['평가손익']:+,.2f} |"
        )
    holdings_table = "\n".join(table_lines)

    ai_total_cost  = df_all["매입총액_KRW"].sum()
    ai_total_value = df_all["평가금액_KRW"].sum()
    ai_total_pnl   = ai_total_value - ai_total_cost
    ai_total_ret   = (ai_total_pnl / ai_total_cost * 100) if ai_total_cost > 0 else 0.0

    sectors: dict[str, float] = {}
    for _, row in df_all.iterrows():
        sec = row["섹터"] or _fetch_sector(row["_ticker"])
        sectors[sec] = sectors.get(sec, 0) + row["평가금액_KRW"]
    sector_lines = "\n".join(
        f"- {sec}: {val/ai_total_value*100:.1f}% (₩{val:,.0f})"
        for sec, val in sorted(sectors.items(), key=lambda x: -x[1])
    ) if ai_total_value > 0 else "(데이터 없음)"

    fg_text = _load_fg()
    now     = datetime.now().strftime("%Y-%m-%d %H:%M")

    prompt = f"""# 포트폴리오 전체 분석 요청

당신은 전문 포트폴리오 매니저입니다. 아래 보유 종목 현황을 바탕으로 종합적인 포트폴리오 분석과 구체적인 조언을 제시해주세요.

**분석 시각:** {now}
**적용 환율:** 1 USD = ₩{usd_krw:,.0f}
{f"**시장 전반 분위기:** {fg_text}" if fg_text else ""}

---

## 포트폴리오 종합 현황 (원화 환산 기준)

| 항목 | 금액 |
|------|------|
| 총 매입금액 | ₩{ai_total_cost:,.0f} |
| 총 평가금액 | ₩{ai_total_value:,.0f} |
| 총 평가손익 | ₩{ai_total_pnl:+,.0f} |
| 전체 수익률 | {ai_total_ret:+.2f}% |
| 보유 종목 수 | {len(df_all)}개 |

---

## 보유 종목별 현황

{holdings_table}

---

## 섹터별 비중

{sector_lines}

---

## 분석 요청 사항

위 포트폴리오를 종합하여 다음을 포함한 분석을 작성해주세요:

1. **전반적인 포트폴리오 평가** — 수익률 수준, 리스크 분산 정도
2. **섹터 집중도 분석** — 편향된 섹터가 있다면 리밸런싱 제안
3. **종목별 투자 의견** — 각 종목의 현 시점 BUY/HOLD/SELL 의견
4. **리스크 요인** — 포트폴리오 전체에서 가장 주의해야 할 위험 요소
5. **리밸런싱 제안** — 비중 조정이 필요한 종목과 대안

**⚠️ 응답 마지막에 반드시 아래 JSON 블록을 포함해주세요:**

```json
{{
  "overall_signal": "REBALANCE 또는 HOLD 또는 ADD",
  "confidence": 0.0~1.0,
  "top_risk": "가장 큰 리스크 한 줄",
  "rebalance": [{{"ticker": "종목", "action": "BUY/HOLD/SELL", "reason": "이유"}}],
  "report_md": "## 포트폴리오 분석 요약\\n상세 내용"
}}
```"""

    with st.expander("📄 프롬프트 미리보기", expanded=False):
        st.code(prompt, language="markdown")

    st.markdown("**프롬프트가 준비됐습니다. 복사 후 Claude.ai에 붙여넣으세요:**")
    copy_button(prompt, "📋 포트폴리오 분석 프롬프트 복사", gradient=_PORTFOLIO_GRADIENT)
    st.caption("버튼 클릭 후 Claude.ai에서 직접 붙여넣어주세요 (Ctrl+V)")
