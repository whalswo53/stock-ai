"""
기존 종목 분석 로직(기술적·캔들패턴·밸류에이션·뉴스·상대강도·시장분위기)을
core.analysis_registry 규격(ctx -> AnalysisResult)으로 감싸 등록한다.

ctx 규격 (종합분석 진입점에서 1번만 조립):
    ctx = {
        "ticker": str,
        "df": DataFrame,     # OHLCV + TechnicalIndicators (6mo)
        "info": dict | None, # yfinance .info
        "currency": str,     # 통화 코드 (resolve_currency)
        "symbol": str,       # 통화 기호
        "is_korean": bool,
        "market": str,       # 종합분석은 뉴스 신뢰 출처 필터에 필요
    }

이 파일을 import하는 순간 @register 데코레이터가 실행되어 ANALYSES에 모듈이
쌓인다 — 종합분석 페이지는 이 파일을 import하고 run_all(ctx)만 호출하면 된다.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from analysis.fundamentals import fmt_value, get_fundamentals, is_fully_missing, to_markdown_table
from analysis.technical import candle_patterns
from analysis.technical.indicators import TechnicalIndicators
from analysis.technical.signals import score as tech_score
from config.sources import TRUSTED_PUBLISHERS
from core.analysis_registry import AnalysisResult, register
from data.collectors.market_sentiment import MarketSentimentCollector, score_to_color
from data.collectors.news_collector import NewsCollector
from data.collectors.price_collector import PriceCollector
from ui.components import (
    POLARITY_LABEL,
    polarity_from_signal,
    render_clean_table,
    render_signal_card,
    render_stat_grid,
)
from utils.ticker_utils import fmt_price_currency


def _price(ctx: dict, val: float) -> str:
    return fmt_price_currency(val, ctx["currency"], ctx["symbol"])


# ═══════════════════════════════════════════════════════════════════════════
# 1. 기술적 분석
# ═══════════════════════════════════════════════════════════════════════════

@register("기술적 분석", order=10)
def technical_section(ctx: dict) -> AnalysisResult:
    df = ctx["df"]
    kr = ctx["is_korean"]
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    close = float(last["Close"])
    prev_close = float(prev["Close"])
    chg_pct = ((close - prev_close) / prev_close) * 100 if prev_close else 0.0

    rsi_val = float(last.get("RSI", 50) or 50)
    macd_v  = float(last.get("MACD", 0) or 0)
    macd_s  = float(last.get("MACD_Signal", 0) or 0)
    bb_up   = float(last.get("BB_Upper", close * 1.02) or close * 1.02)
    bb_lo   = float(last.get("BB_Lower", close * 0.98) or close * 0.98)
    ma5     = float(last.get("MA5", 0) or 0)
    ma20    = float(last.get("MA20", 0) or 0)
    t_score = float(tech_score(last))

    bb_range = bb_up - bb_lo
    bb_pct = (close - bb_lo) / bb_range if bb_range > 0 else 0.5

    rsi_interp   = "과매수 ⚠️" if rsi_val >= 70 else "과매도 💡" if rsi_val <= 30 else "중립"
    macd_interp  = "골든크로스 📈" if macd_v > macd_s else "데드크로스 📉"
    bb_interp    = ("상단 근접 ⚠️" if bb_pct >= 0.85 else "하단 근접 💡" if bb_pct <= 0.15 else f"중앙권 ({bb_pct:.0%})")
    trend_interp = "단기 상승" if ma5 > ma20 and ma20 > 0 else ("단기 하락" if ma5 < ma20 and ma20 > 0 else "—")
    sig_label    = "매수 관심 🟢" if t_score > 0.15 else ("매도 주의 🔴" if t_score < -0.15 else "중립 ⏸")

    # 극성(polarity) — 텍스트 라벨과 별개로 이미 계산된 수치 기준에서 직접 판정한다
    # (라벨 문자열을 파싱하지 않으므로 이모지·문구가 바뀌어도 색상 버그가 재발하지 않음).
    rsi_polarity = "bearish" if rsi_val >= 70 else "bullish" if rsi_val <= 30 else "neutral"
    macd_polarity = "bullish" if macd_v > macd_s else "bearish"
    bb_polarity = "bearish" if bb_pct >= 0.85 else "bullish" if bb_pct <= 0.15 else "neutral"
    score_polarity = "bullish" if t_score > 0.15 else "bearish" if t_score < -0.15 else "neutral"

    price_hist = df.tail(5)[["Open", "High", "Low", "Close", "Volume"]].to_string(
        float_format=lambda x: f"{x:,.2f}"
    )

    # ── 구간 추세 요약 (고가/저가 레인지 위치, MA50/MA200 이격률) ────────────
    # ctx["df"]는 종합분석 진입점 기준 6개월치. 그보다 짧으면(신규 상장 등)
    # 있는 만큼만 계산하고 실제 확보된 거래일수를 그대로 명시한다.
    n_days = len(df)
    span_start, span_end = df.index[0], df.index[-1]
    range_high = float(df["High"].max())
    range_low = float(df["Low"].min())
    range_span = range_high - range_low
    range_pos_pct = (close - range_low) / range_span * 100 if range_span > 0 else 50.0

    close_series = df["Close"]

    def _ma_value(window: int) -> float | None:
        ma = close_series.rolling(window, min_periods=window).mean().iloc[-1]
        return None if pd.isna(ma) else float(ma)

    def _ma_row(label: str, window: int, val: float | None) -> str:
        if val is None:
            return f"| {label} | 데이터 부족 ({window}거래일 필요, {n_days}거래일 확보) |"
        dev_pct = (close - val) / val * 100
        return f"| {label} 이격률 | {_price(ctx, val)} ({dev_pct:+.1f}%) |"

    ma50_val = _ma_value(50)
    ma200_val = _ma_value(200)

    trend_md = (
        f"**{n_days}거래일 구간 요약** ({span_start:%Y-%m-%d} ~ {span_end:%Y-%m-%d})\n\n"
        "| 지표 | 값 |\n"
        "|------|-----|\n"
        f"| 구간 고가 / 저가 | {_price(ctx, range_high)} / {_price(ctx, range_low)} |\n"
        f"| 현재가 레인지 내 위치 | {range_pos_pct:.0f}% (0%=저가, 100%=고가) |\n"
        f"{_ma_row('MA50', 50, ma50_val)}\n"
        f"{_ma_row('MA200', 200, ma200_val)}"
    )

    markdown = (
        f"**현재가:** {_price(ctx, close)} ({chg_pct:+.2f}%)\n"
        f"**기술적 점수:** {t_score:+.2f} → {sig_label}\n\n"
        f"{trend_md}\n\n"
        "| 지표 | 값 | 해석 |\n"
        "|------|-----|------|\n"
        f"| RSI(14) | {rsi_val:.1f} | {rsi_interp} |\n"
        f"| MACD | {macd_v:.4f} / Sig {macd_s:.4f} | {macd_interp} |\n"
        f"| BB 위치 | %B {bb_pct:.2f} | {bb_interp} |\n"
        f"| MA5 / MA20 | {_price(ctx, ma5)} / {_price(ctx, ma20)} | {trend_interp} |\n\n"
        f"**최근 5거래일 가격:**\n```\n{price_hist}\n```"
    )

    def _render() -> None:
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_signal_card("현재가", _price(ctx, close), f"{chg_pct:+.2f}%",
                                polarity="bullish" if chg_pct > 0 else "bearish" if chg_pct < 0 else "neutral")
        with c2:
            render_signal_card("RSI (14)", f"{rsi_val:.1f}", rsi_interp, polarity=rsi_polarity)
        with c3:
            render_signal_card("MACD", "골든" if macd_v > macd_s else "데드", macd_interp, polarity=macd_polarity)
        with c4:
            render_signal_card("BB 위치", f"{bb_pct:.0%}", bb_interp, polarity=bb_polarity)
        with c5:
            render_signal_card("기술 점수", f"{t_score:+.2f}", sig_label, polarity=score_polarity)

        st.caption(
            f"📐 {n_days}거래일 구간({span_start:%Y-%m-%d}~{span_end:%Y-%m-%d}) "
            f"레인지 {range_pos_pct:.0f}% 위치 · "
            f"구간 고가 {_price(ctx, range_high)} / 저가 {_price(ctx, range_low)}"
        )

        with st.expander("📊 차트 보기 (6개월)", expanded=False):
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="가격",
                increasing=dict(line=dict(color="#26a69a"), fillcolor="#26a69a"),
                decreasing=dict(line=dict(color="#ef5350"), fillcolor="#ef5350"),
            ))
            for w, col in [(20, "#2196F3"), (60, "#9C27B0")]:
                col_name = f"MA{w}"
                if col_name in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[col_name], name=f"MA{w}",
                        line=dict(color=col, width=1.3),
                    ))
            if "BB_Upper" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB 상단",
                                          line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dash")))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB 하단",
                                          line=dict(color="rgba(200,200,200,0.5)", width=1, dash="dash"),
                                          fill="tonexty", fillcolor="rgba(200,200,200,0.05)"))
            tick_fmt = ",.0f" if kr else ".2f"
            fig.update_layout(
                height=420, template="plotly_dark",
                xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            fig.update_yaxes(tickformat=tick_fmt)
            st.plotly_chart(fig, width="stretch")

    return AnalysisResult(
        title="기술적 분석",
        markdown=markdown,
        json={
            "기술_점수": t_score, "RSI": rsi_val, "MACD": macd_v, "MACD_Signal": macd_s,
            "현재가": close, "등락률": chg_pct,
            "구간_거래일수": n_days, "구간_고가": range_high, "구간_저가": range_low,
            "레인지_위치_pct": range_pos_pct, "MA50": ma50_val, "MA200": ma200_val,
        },
        render=_render,
        polarity=score_polarity,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. 캔들 패턴
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def _load_pattern_history(ticker: str) -> pd.DataFrame:
    df5 = PriceCollector().fetch(ticker, period="5y")
    if df5.empty:
        return df5
    return TechnicalIndicators().compute(df5)


@register("캔들 패턴", order=20)
def candle_section(ctx: dict) -> AnalysisResult:
    ticker = ctx["ticker"]

    if not candle_patterns.is_available():
        return AnalysisResult(
            title="캔들 패턴",
            markdown="TA-Lib이 설치되지 않아 캔들 패턴 분석을 사용할 수 없습니다.",
        )

    hist5y = _load_pattern_history(ticker)
    if hist5y.empty or len(hist5y) < 120:
        return AnalysisResult(title="캔들 패턴", markdown="패턴 통계를 낼 만큼의 히스토리가 없습니다.")

    hits = candle_patterns.recent_hits(hist5y, lookback_days=5)
    if not hits:
        return AnalysisResult(
            title="캔들 패턴",
            markdown="최근 5거래일 내 인식된 캔들 패턴이 없습니다.",
            json={"캔들_패턴": []},
        )

    horizon = 5
    seen_keys: set = set()
    rows: list[dict] = []
    json_hits: list[dict] = []
    for h in hits:
        key = (h.func, h.sign)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        stats = candle_patterns.full_stats([hist5y], h.func, h.sign, horizon)
        base = stats["base"]
        vd = candle_patterns.verdict(base, h.sign)
        rows.append({
            "날짜": f"{h.date:%Y-%m-%d}",
            "패턴": h.name,
            "방향": "강세" if h.sign > 0 else ("약세" if h.sign < 0 else "중립"),
            "표본": base.n,
            "평균수익": f"{base.avg_return:+.2f}%" if base.n else "—",
            "판정": vd,
        })
        json_hits.append({
            "pattern": h.name, "sign": h.sign, "n": base.n,
            "win_rate": base.win_rate, "avg_return": base.avg_return,
        })

    md_lines = [
        f"수익률 측정 {horizon}일 후 · 이 종목 5년 히스토리 기준 (표본 {candle_patterns.MIN_SAMPLES}회 미만은 판단 불가)\n",
        "| 날짜 | 패턴 | 방향 | 표본 | 평균수익 | 판정 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['날짜']} | {r['패턴']} | {r['방향']} | {r['표본']} | {r['평균수익']} | {r['판정']} |"
        )
    markdown = "\n".join(md_lines)

    def _render() -> None:
        st.caption(
            f"수익률 측정 {horizon}일 후 · 이 종목 5년 히스토리 기준 "
            f"(표본 {candle_patterns.MIN_SAMPLES}회 미만은 판단 불가)"
        )
        render_clean_table(pd.DataFrame(rows), judgment_col="판정", label_col="패턴")

    return AnalysisResult(
        title="캔들 패턴", markdown=markdown, json={"캔들_패턴": json_hits}, render=_render,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3. 밸류에이션
# ═══════════════════════════════════════════════════════════════════════════

def _per_polarity(fundamentals: dict) -> str | None:
    """PER 단순 임계값 휴리스틱 — 업종 평균 데이터가 없어 절대 수준으로만 판단.
    극단적으로 싸거나(딥밸류) 비싼(과열) 경우만 방향성을 주고, 그 사이는 중립."""
    per = fundamentals.get("밸류에이션", {}).get("PER")
    if per in (None, "N/A"):
        return None
    try:
        per_val = float(per)
    except (TypeError, ValueError):
        return None
    if per_val <= 0:
        return None  # 적자 등으로 PER이 의미 없는 구간
    if per_val < 8:
        return "bullish"
    if per_val > 40:
        return "bearish"
    return "neutral"


@register("밸류에이션", order=30)
def valuation_section(ctx: dict) -> AnalysisResult:
    fundamentals = get_fundamentals(ctx)
    missing = is_fully_missing(fundamentals)
    markdown = "펀더멘털 데이터 미제공" if missing else to_markdown_table(fundamentals)
    per_polarity = None if missing else _per_polarity(fundamentals)

    def _render() -> None:
        if missing:
            st.caption(markdown)
            return
        for group, fields in fundamentals.items():
            st.markdown(f"**{group}**")
            if all(v == "N/A" for v in fields.values()):
                st.caption("데이터 미제공")
                continue
            items = []
            for label, v in fields.items():
                is_per = group == "밸류에이션" and label == "PER"
                polarity = per_polarity if is_per else None
                items.append({
                    "label": label,
                    "value": fmt_value(label, v),
                    "eval": POLARITY_LABEL.get(polarity, ""),
                    "polarity": polarity,
                })
            render_stat_grid(items, columns=3)

    return AnalysisResult(
        title="밸류에이션", markdown=markdown, json=fundamentals,
        polarity=per_polarity, render=_render,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. 뉴스
# ═══════════════════════════════════════════════════════════════════════════

_POS_WORDS = {"급등", "상승", "호재", "사상최고", "기록", "흑자", "상향", "매수", "긍정", "성장", "기대"}
_NEG_WORDS = {"급락", "하락", "악재", "적자", "하향", "매도", "부정", "우려", "위기", "손실", "경고"}

# 7일 — SMIC처럼 발행량이 적은 종목은 48h로는 직접 언급 기사가 0~1건이라 사실상
# 안 보이는 경우가 많아 넓혔다. 정렬은 여전히 최신순이라 AAPL/삼성전자처럼
# 발행량이 많은 종목은 최신 기사가 그대로 위에 오므로 영향 없음.
_NEWS_WINDOW_HOURS = 168


def _relative_time(published_at: str) -> str:
    """published_at(ISO 문자열) → '3시간 전' / '5일 전'. 오래된 기사임을 명시하기 위함."""
    if not published_at:
        return ""
    try:
        dt = datetime.fromisoformat(published_at)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if hours < 1:
        return "방금"
    if hours < 24:
        return f"{hours:.0f}시간 전"
    return f"{hours / 24:.0f}일 전"


@st.cache_data(ttl=3600, show_spinner=False)
def _load_news(ticker: str, market: str, company_name: str = "") -> tuple[list[dict], int]:
    """Returns (관련 뉴스 목록, 수집 총 건수).

    company_name(yfinance longName/shortName)을 검색어·직접언급판정에 모두 써야
    HK/CN 등 정적 맵에 없는 종목에서도 실제 관련 기사가 잡힌다.
    """
    try:
        nc = NewsCollector()
        raw = nc.fetch_by_ticker(ticker, market, hours=_NEWS_WINDOW_HOURS, company_name=company_name)
        total = len(raw)
        filtered = nc.filter_relevant(raw, ticker, market, company_name=company_name)
        return nc.to_dicts(filtered), total
    except Exception:
        return [], 0


@register("뉴스", order=40)
def news_section(ctx: dict) -> AnalysisResult:
    ticker, market = ctx["ticker"], ctx["market"]
    info = ctx.get("info") or {}
    # shortName을 우선한다 — "SMIC"처럼 실제 언론이 쓰는 약칭이 longName(정식
    # 법인명)보다 검색 적중률이 훨씬 높다 (e.g. longName "Semiconductor
    # Manufacturing International Corporation"으로는 관련 한글 기사가 거의 안 잡힘).
    company_name = info.get("shortName") or info.get("longName") or ""

    articles, total_collected = _load_news(ticker, market, company_name)
    trusted_sources = TRUSTED_PUBLISHERS.get(market, [])

    n_direct = sum(1 for a in articles if a.get("relevance_tier") == "직접")
    n_sector = len(articles) - n_direct
    direct_articles = [a for a in articles if a.get("relevance_tier") == "직접"]

    # 감성은 "직접 언급" 기사에서만 낸다 — 섹터 키워드로만 걸린 기사로 감성을
    # 억지로 만들지 않는다 (정직 표기).
    if n_direct > 0:
        pos_cnt = sum(
            1 for a in direct_articles[:10]
            if any(w in (a.get("title", "") + a.get("summary", "")) for w in _POS_WORDS)
        )
        neg_cnt = sum(
            1 for a in direct_articles[:10]
            if any(w in (a.get("title", "") + a.get("summary", "")) for w in _NEG_WORDS)
        )
        sent_label = (
            "😀 긍정" if pos_cnt > neg_cnt + 1 else
            "😟 부정" if neg_cnt > pos_cnt + 1 else
            "😐 중립"
        )
        news_lines = "\n".join(
            f"- [{a.get('source', '')}] {a.get('title', '')} "
            f"({(a.get('published_at') or '')[:10]} · {_relative_time(a.get('published_at', ''))})"
            for a in direct_articles[:5]
        )
        markdown = (
            f"수집 {total_collected}건 → 직접 언급 {n_direct}건 · 섹터 키워드 매칭 {n_sector}건\n"
            f"전반 감성(직접 언급 기준): {sent_label} "
            f"(긍정 {pos_cnt} · 부정 {neg_cnt} / {min(len(direct_articles), 10)}건)\n\n"
            f"{news_lines}"
        )
    else:
        sent_label = "😐 판단 보류"
        pos_cnt = neg_cnt = 0
        if articles:
            markdown = f"관련 뉴스 없음 (수집 {total_collected}건 중 직접 언급 0건 · 섹터 키워드 매칭 {n_sector}건)"
        else:
            markdown = (
                "관련 뉴스를 찾을 수 없습니다. (종목명 직접 언급·섹터 키워드 매칭 기사 없음)"
                if total_collected > 0 else "최근 48시간 내 뉴스를 수집할 수 없습니다."
            )

    def _render() -> None:
        if total_collected > 0:
            st.caption(
                f"{total_collected}건 수집 → {len(articles)}건 채택 "
                f"(🎯 직접 언급 {n_direct} · 🏭 섹터 키워드 {n_sector}) — "
                "중복·재탕 및 관련성 낮은 기사 제외"
            )

        if not articles:
            st.caption(markdown)
            return

        sent_color = "#26a69a" if "긍정" in sent_label else ("#ef5350" if "부정" in sent_label else "#9E9E9E")
        sc1, sc2 = st.columns([1, 3])
        sc1.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;'
            f'padding:16px;text-align:center">'
            f'  <div style="font-size:11px;color:#888;margin-bottom:4px">전반 감성</div>'
            f'  <div style="font-size:26px;font-weight:700;color:{sent_color}">{sent_label}</div>'
            f'  <div style="font-size:11px;color:#666;margin-top:4px">'
            f'긍정 {pos_cnt} · 부정 {neg_cnt} / {min(len(direct_articles), 10)}건</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with sc2:
            for a in articles[:5]:
                pub    = (a.get("published_at") or "")[:10]
                pub_rel = _relative_time(a.get("published_at", ""))
                source = a.get("source", "")
                title  = a.get("title", "")
                url    = a.get("url", "")
                is_pos = any(w in title for w in _POS_WORDS)
                is_neg = any(w in title for w in _NEG_WORDS)
                dot_color = "#26a69a" if is_pos else ("#ef5350" if is_neg else "#888")
                trust_icon = "✅" if (trusted_sources and any(t in source for t in trusted_sources)) else "⚠️"
                tier = a.get("relevance_tier", "")
                rel  = a.get("relevance", 0.0)
                if tier == "직접":
                    rel_badge = (
                        f'<span style="background:rgba(38,166,154,0.18);color:#26a69a;'
                        f'border-radius:4px;padding:0 5px;font-size:10px">🎯 직접 {rel:.1f}</span> '
                    )
                elif tier == "섹터":
                    rel_badge = (
                        f'<span style="background:rgba(255,152,0,0.15);color:#FF9800;'
                        f'border-radius:4px;padding:0 5px;font-size:10px">🏭 섹터 {rel:.1f}</span> '
                    )
                else:
                    rel_badge = ""
                line = (
                    f'<div style="display:flex;gap:8px;padding:5px 0;'
                    f'border-bottom:1px solid rgba(255,255,255,0.07)">'
                    f'  <span style="color:{dot_color};margin-top:2px">●</span>'
                    f'  <div>'
                    f'    <span style="color:#888;font-size:11px">{rel_badge}{trust_icon} [{source}] {pub} · {pub_rel}</span><br>'
                )
                if url:
                    line += f'    <a href="{url}" target="_blank" style="color:#ccc;font-size:13px;text-decoration:none">{title}</a>'
                else:
                    line += f'    <span style="color:#ccc;font-size:13px">{title}</span>'
                line += "  </div></div>"
                st.markdown(line, unsafe_allow_html=True)

    return AnalysisResult(
        title="뉴스",
        markdown=markdown,
        json={
            "뉴스_감성": sent_label, "긍정_기사수": pos_cnt, "부정_기사수": neg_cnt,
            "수집_건수": total_collected, "직접언급_건수": n_direct, "섹터매칭_건수": n_sector,
        },
        render=_render,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. 상대강도 (페어 트레이딩)
# ═══════════════════════════════════════════════════════════════════════════

# 07_pairs_trading.py와 동일한 클래스를 재사용한다. PeerDiscovery(동종업종 탐색) →
# PairScanner(Engle-Granger 공적분 검정으로 후보 중 p-value 최저 피어 선정) →
# QuantAggregator(OLS + 칼만 앙상블). 같은 파라미터로 같은 쌍을 분석하면 페어
# 트레이딩 페이지의 수치와 일치한다.
_PAIR_ALPHA = 0.05
_PAIR_PERIOD = "1y"
_PAIR_ZSCORE_WINDOW = 30
_PAIR_ENTRY_Z = 2.0
_PAIR_EXIT_Z = 0.5
_PAIR_KALMAN_DELTA = 1e-4
_PAIR_STOP_MULT = 1.75  # |Z| ≥ entry_z × mult → STOP_LOSS (공적분 붕괴 의심)


@st.cache_data(ttl=86400, show_spinner=False)
def _discover_peer_candidates(ticker: str) -> tuple[list[str], dict[str, str], str, str]:
    from analysis.quant.pair_scanner import PeerDiscovery
    pg = PeerDiscovery(top_n=10).find(ticker)
    return pg.tickers, pg.names, pg.sector, pg.source


@st.cache_data(ttl=1800, show_spinner=False)
def _best_cointegrated_peer(
    ticker: str, tickers_tuple: tuple, names_json: str, alpha: float,
) -> tuple[str, float, bool] | None:
    import json
    from analysis.quant.pair_scanner import PairScanner

    names = json.loads(names_json)
    scanner = PairScanner(period=_PAIR_PERIOD, alpha=alpha)
    prices = scanner.fetch_prices_for(list(tickers_tuple))
    results = scanner.scan_tickers(list(tickers_tuple), names, prices, seed_ticker=ticker)
    if not results:
        return None
    best = results[0]
    return best.ticker_b, best.pvalue, best.is_cointegrated


@st.cache_data(ttl=1800, show_spinner=False)
def _run_pair_aggregate(ticker: str, peer: str, alpha: float):
    from analysis.quant.aggregator import QuantAggregator
    return QuantAggregator(
        period=_PAIR_PERIOD, zscore_window=_PAIR_ZSCORE_WINDOW,
        entry_z=_PAIR_ENTRY_Z, exit_z=_PAIR_EXIT_Z,
        kalman_delta=_PAIR_KALMAN_DELTA, alpha=alpha,
        stop_loss_mult=_PAIR_STOP_MULT,
    ).run_pair(ticker, peer)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_pair_analysis(ticker: str) -> dict | None:
    import json

    try:
        tickers, names, sector, source = _discover_peer_candidates(ticker)
    except Exception:
        return None

    if len([t for t in tickers if t != ticker]) < 1:
        return None

    try:
        best = _best_cointegrated_peer(
            ticker, tuple(tickers), json.dumps(names, ensure_ascii=False), _PAIR_ALPHA,
        )
    except Exception:
        best = None
    if not best:
        return None
    peer, _pvalue, _is_coint = best

    try:
        result = _run_pair_aggregate(ticker, peer, _PAIR_ALPHA)
    except Exception:
        return None

    return {
        "peer": peer,
        "peer_name": names.get(peer, peer),
        "sector": sector,
        "source": source,
        "result": result,  # AggregatedPairResult
    }


@register("상대강도(페어트레이딩)", order=50)
def pair_section(ctx: dict) -> AnalysisResult:
    ticker = ctx["ticker"]
    pair_info = _load_pair_analysis(ticker)

    if not pair_info:
        return AnalysisResult(
            title="상대강도(페어트레이딩)",
            markdown="이 종목에 대한 동종 비교 데이터가 없습니다.",
        )

    agg   = pair_info["result"]
    coint = agg.coint_result
    markdown = (
        f"비교 종목: {pair_info['peer_name']} ({pair_info['peer']}) [{pair_info['sector']}]\n"
        f"공적분(Engle-Granger) p-value: {coint.pvalue:.4f} "
        f"({'공적분 있음' if coint.is_cointegrated else '공적분 없음'}, 유의수준 {_PAIR_ALPHA:.0%})\n"
        f"종합(OLS+칼만) Z-score: {agg.composite_zscore:+.2f} σ → {agg.label}"
    )

    def _render() -> None:
        signal = agg.signal_a
        sig_color = {"BUY": "#26a69a", "SELL": "#ef5350", "CLOSE": "#FF9800",
                     "STOP_LOSS": "#ab47bc"}.get(signal, "#9E9E9E")
        sig_rgb = {"BUY": "38,166,154", "SELL": "239,83,80", "CLOSE": "255,152,0",
                   "STOP_LOSS": "171,71,188"}.get(signal, "100,100,100")

        pc1, pc2, pc3, pc4 = st.columns([1, 1, 1, 2])
        with pc1:
            render_signal_card("비교 종목", pair_info["peer_name"], pair_info["peer"], polarity=None)
        with pc2:
            render_signal_card(
                "공적분 (Engle-Granger)",
                "있음 ✅" if coint.is_cointegrated else "없음 ❌",
                f"p={coint.pvalue:.4f}",
                polarity="bullish" if coint.is_cointegrated else "bearish",
            )
        with pc3:
            render_signal_card(f"종합 Z-score ({_PAIR_ZSCORE_WINDOW}일)", f"{agg.composite_zscore:+.2f} σ",
                                signal, polarity=polarity_from_signal(signal))
        with pc4:
            st.markdown(
                f'<div style="background:rgba({sig_rgb},0.12);'
                f'border-left:3px solid {sig_color};border-radius:0 8px 8px 0;padding:10px 14px">'
                f'  {agg.label}'
                f'</div>',
                unsafe_allow_html=True,
            )

        w_ols, w_kalman = agg.contributions[0].weight, agg.contributions[1].weight
        st.caption(
            f"업종: {pair_info['sector']}  ·  {pair_info['source']}  ·  "
            f"OLS {w_ols*100:.0f}% / 칼만 {w_kalman*100:.0f}% 가중 앙상블  ·  "
            f"공적분 유의수준 {_PAIR_ALPHA:.0%}, 기간 {_PAIR_PERIOD}  \n"
            "📊 **페어 트레이딩** 페이지의 직접 분석 탭에서 이 쌍으로 상세 분석(차트 포함)을 확인할 수 있습니다."
        )

    return AnalysisResult(
        title="상대강도(페어트레이딩)",
        markdown=markdown,
        json={
            "페어_피어": pair_info["peer"], "페어_zscore": agg.composite_zscore,
            "페어_신호": agg.signal_a, "페어_공적분_pvalue": coint.pvalue,
        },
        render=_render,
        polarity=polarity_from_signal(agg.signal_a),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. 시장 분위기 (Fear & Greed)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg():
    return MarketSentimentCollector().fetch()


@register("시장 분위기(F&G)", order=60)
def fear_greed_section(ctx: dict) -> AnalysisResult:
    fg = _load_fg()
    fg_score, fg_label, fg_vix = fg.score, fg.label, fg.vix

    fg_interp = (
        "극도의 공포 상태 — 역발상 매수 기회일 수 있음" if fg_score <= 20 else
        "공포 우세 — 시장 불안감이 높음. 신중한 접근 권장" if fg_score <= 40 else
        "중립 — 시장 방향성 불확실. 종목 개별 분석 중심으로" if fg_score <= 60 else
        "탐욕 우세 — 상승 기대감 강함. 과열 여부 주의" if fg_score <= 80 else
        "극도의 탐욕 — 시장 과열 신호. 포트폴리오 리스크 점검 권장"
    )
    fg_polarity = "bearish" if fg_score < 40 else "bullish" if fg_score > 60 else "neutral"

    markdown = (
        f"CNN 공포·탐욕 지수: {fg_score}/100 ({fg_label})\n"
        f"해석: {fg_interp}\n"
        + (f"VIX: {fg_vix:.1f}\n" if fg_vix >= 0 else "")
        + f"업데이트: {fg.last_update}"
    )

    def _render() -> None:
        fg_color = score_to_color(fg_score)
        fg1, fg2 = st.columns([1, 3])
        with fg1:
            mini_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fg_score,
                number={"font": {"size": 36, "color": "white"}, "valueformat": "d"},
                gauge={
                    "axis": {"range": [0, 100], "tickvals": [0, 50, 100],
                             "tickcolor": "#666", "tickwidth": 1},
                    "bar": {"color": fg_color, "thickness": 0.04},
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": "#c62828"},
                        {"range": [20, 40], "color": "#e64a19"},
                        {"range": [40, 60], "color": "#f9a825"},
                        {"range": [60, 80], "color": "#558b2f"},
                        {"range": [80, 100], "color": "#00695c"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 4},
                                  "thickness": 0.85, "value": fg_score},
                },
            ))
            mini_fig.update_layout(
                height=160, margin=dict(l=20, r=20, t=20, b=5),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"},
            )
            st.plotly_chart(mini_fig, width="stretch")
        with fg2:
            _vix_row = (
                f'<div style="color:#888;font-size:12px">VIX: {fg_vix}</div>'
                if fg_vix >= 0 else ""
            )
            st.markdown(
                f'<div style="padding:10px 0">'
                f'  <div style="font-size:20px;font-weight:700;color:{fg_color};margin-bottom:4px">'
                f'{fg_label} ({fg_score}/100)</div>'
                f'  <div style="color:#bbb;font-size:14px;margin-bottom:10px">{fg_interp}</div>'
                f'  {_vix_row}'
                f'  <div style="color:#555;font-size:11px;margin-top:6px">출처: {fg.source}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    return AnalysisResult(
        title="시장 분위기(F&G)",
        markdown=markdown,
        json={"FG_점수": fg_score, "FG_라벨": fg_label, "VIX": fg_vix},
        render=_render,
        polarity=fg_polarity,
    )
