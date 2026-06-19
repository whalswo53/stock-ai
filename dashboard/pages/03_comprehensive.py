"""
종합 분석 페이지.
기술적 분석 · 뉴스/센티먼트 · 페어 상대강도 · 시장 분위기를 한 화면에서 확인하고
통합 AI 프롬프트를 생성한다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from analysis.technical.indicators import TechnicalIndicators
from analysis.technical.signals import score as tech_score
from config.sources import TICKER_KR_NAME, TRUSTED_PUBLISHERS
from data.collectors.market_sentiment import (
    MarketSentimentCollector,
    FearGreedResult,
    score_to_color,
)
from data.collectors.news_collector import NewsCollector
from data.collectors.price_collector import PriceCollector
from utils.clipboard import copy_button
from utils.ticker_utils import detect_market, is_kr, fmt_price
from utils.search_widget import ticker_search_widget

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 종합 분석")
    _jump = st.session_state.pop("portfolio_jump_ticker", None)
    if _jump:
        st.session_state["_tsq_comp"] = _jump
    ticker = ticker_search_widget(
        key="comp",
        label="종목 코드 또는 한글명",
        default="005930.KS",
    ) or "005930.KS"

    st.divider()
    st.caption(
        "⬆ 종목을 입력하면 아래 5개 섹션이 자동으로 채워집니다:\n\n"
        "1️⃣ 기술적 분석  2️⃣ 뉴스/센티먼트\n"
        "3️⃣ 상대 강도  4️⃣ 시장 분위기\n"
        "5️⃣ 통합 AI 프롬프트"
    )

market = detect_market(ticker)
kr     = is_kr(ticker)


# ── Data loaders (all cached) ─────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_price(ticker: str) -> tuple[pd.DataFrame, dict]:
    pc  = PriceCollector()
    df  = pc.fetch(ticker, period="6mo")
    if not df.empty:
        df = TechnicalIndicators().compute(df)
    info = pc.get_info(ticker)
    return df, info


@st.cache_data(ttl=3600, show_spinner=False)
def _load_news(ticker: str, market: str) -> tuple[list[dict], int]:
    """Returns (관련 뉴스 목록, 수집 총 건수)."""
    try:
        nc    = NewsCollector()
        raw   = nc.fetch_by_ticker(ticker, market, hours=48)
        total = len(raw)
        filtered = nc.filter_relevant(raw, ticker, market)
        return nc.to_dicts(filtered), total
    except Exception:
        return [], 0


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fg() -> FearGreedResult:
    return MarketSentimentCollector().fetch()


@st.cache_data(ttl=172800, show_spinner=False)  # 48h — reduces yfinance re-fetch frequency
def _load_peers(ticker: str) -> dict | None:
    """
    INDUSTRY_GROUPS에서 동종 종목을 찾아 ratio Z-score를 계산한다.
    cointegration 검정 없이 60일 rolling ratio 기반 단순 Z-score 사용.
    """
    try:
        from analysis.quant.pair_scanner import INDUSTRY_GROUPS
    except ImportError:
        return None

    peers: list[str] = []
    group_name = ""
    for gname, gdata in INDUSTRY_GROUPS.items():
        if ticker in gdata.get("tickers", []):
            peers = [t for t in gdata["tickers"] if t != ticker][:4]
            group_name = gname
            break

    if not peers:
        return None

    collector = PriceCollector()
    try:
        df_a = collector.fetch(ticker, period="6mo")[["Close"]].rename(columns={"Close": "A"})
    except Exception:
        return None

    best: dict | None = None
    best_abs_z = 0.0

    for peer in peers:
        try:
            df_b = (
                collector.fetch(peer, period="6mo")[["Close"]]
                .rename(columns={"Close": "B"})
            )
            df = df_a.join(df_b, how="inner").dropna()
            if len(df) < 30:
                continue
            ratio  = df["A"] / df["B"]
            mu     = ratio.rolling(60, min_periods=30).mean().iloc[-1]
            sigma  = ratio.rolling(60, min_periods=30).std().iloc[-1]
            if sigma == 0 or pd.isna(mu) or pd.isna(sigma):
                continue
            z = float((ratio.iloc[-1] - mu) / sigma)
            if abs(z) > best_abs_z:
                best_abs_z = abs(z)
                # peer name
                names = INDUSTRY_GROUPS.get(group_name, {}).get("names", {})
                peer_name = names.get(peer, peer)
                # direction interpretation
                if z > 0.5:
                    direction = f"{ticker}가 {peer_name} 대비 **상대적 고평가** (매수 신중)"
                elif z < -0.5:
                    direction = f"{ticker}가 {peer_name} 대비 **상대적 저평가** (매수 관심)"
                else:
                    direction = f"{ticker}와 {peer_name} 간 가격 비율이 **평균 수준** (중립)"
                best = {
                    "peer": peer,
                    "peer_name": peer_name,
                    "zscore": z,
                    "group": group_name,
                    "direction": direction,
                    "ratio": float(ratio.iloc[-1]),
                    "ratio_mean": float(mu),
                }
        except Exception:
            continue

    return best


# ── Page title ────────────────────────────────────────────────────────────────
st.title("🎯  종합 분석")

with st.spinner(f"'{ticker}' 데이터 수집 중… (가격 · 지표 · 뉴스 · 시장 분위기)"):
    df, info                 = _load_price(ticker)
    articles, total_collected = _load_news(ticker, market)
    fg                       = _load_fg()

if df.empty:
    st.error(f"'{ticker}' 가격 데이터를 불러올 수 없습니다. 티커 코드를 확인하세요.")
    st.stop()

company = TICKER_KR_NAME.get(ticker) or info.get("shortName") or info.get("longName") or ticker
last    = df.iloc[-1]
prev    = df.iloc[-2] if len(df) > 1 else last
close   = float(last["Close"])
chg_abs = close - float(prev["Close"])
chg_pct = (chg_abs / float(prev["Close"])) * 100

st.subheader(f"{company} ({ticker})  ·  {market}")

# ── Section 1: 기술적 분석 요약 ─────────────────────────────────────────────
st.markdown("### 1️⃣  기술적 분석 요약")

rsi_val = float(last.get("RSI", 50) or 50)
macd_v  = float(last.get("MACD", 0) or 0)
macd_s  = float(last.get("MACD_Signal", 0) or 0)
bb_up   = float(last.get("BB_Upper", close * 1.02) or close * 1.02)
bb_lo   = float(last.get("BB_Lower", close * 0.98) or close * 0.98)
ma5     = float(last.get("MA5", 0) or 0)
ma20    = float(last.get("MA20", 0) or 0)
t_score = float(tech_score(last))

bb_range = bb_up - bb_lo
bb_pct   = (close - bb_lo) / bb_range if bb_range > 0 else 0.5

# Interpretations
rsi_interp   = ("과매수 ⚠️" if rsi_val >= 70 else "과매도 💡" if rsi_val <= 30 else "중립")
macd_interp  = "골든크로스 📈" if macd_v > macd_s else "데드크로스 📉"
bb_interp    = ("상단 근접 ⚠️" if bb_pct >= 0.85 else "하단 근접 💡" if bb_pct <= 0.15 else f"중앙권 ({bb_pct:.0%})")
trend_interp = "단기 상승" if ma5 > ma20 and ma20 > 0 else ("단기 하락" if ma5 < ma20 and ma20 > 0 else "—")

# Overall card
sig_color = "#26a69a" if t_score > 0.15 else ("#ef5350" if t_score < -0.15 else "#9E9E9E")
sig_label = "매수 관심 🟢" if t_score > 0.15 else ("매도 주의 🔴" if t_score < -0.15 else "중립 ⏸")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가",          fmt_price(close, ticker), f"{chg_pct:+.2f}%")
c2.metric(f"RSI (14)",      f"{rsi_val:.1f}",  rsi_interp)
c3.metric("MACD",           "골든" if macd_v > macd_s else "데드", macd_interp)
c4.metric("BB 위치",         f"{bb_pct:.0%}",  bb_interp)
c5.metric("기술 점수",        f"{t_score:+.2f}", sig_label)

# Compact chart (expander, 기본 접힘)
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

st.divider()

# ── Section 2: 종목 관련 뉴스 ────────────────────────────────────────────────
st.markdown("### 2️⃣  종목 관련 뉴스")

_trusted_sources = TRUSTED_PUBLISHERS.get(market, [])
pos_words = {"급등", "상승", "호재", "사상최고", "기록", "흑자", "상향", "매수", "긍정", "성장", "기대"}
neg_words = {"급락", "하락", "악재", "적자", "하향", "매도", "부정", "우려", "위기", "손실", "경고"}

if total_collected > 0:
    st.caption(f"{total_collected}건 수집 → {len(articles)}건 관련 뉴스 필터링")

if articles:
    pos_cnt = sum(
        1 for a in articles[:10]
        if any(w in (a.get("title", "") + a.get("summary", "")) for w in pos_words)
    )
    neg_cnt = sum(
        1 for a in articles[:10]
        if any(w in (a.get("title", "") + a.get("summary", "")) for w in neg_words)
    )
    sent_label = (
        "😀 긍정"  if pos_cnt > neg_cnt + 1 else
        "😟 부정"  if neg_cnt > pos_cnt + 1 else
        "😐 중립"
    )
    sent_color = "#26a69a" if "긍정" in sent_label else ("#ef5350" if "부정" in sent_label else "#9E9E9E")

    sc1, sc2 = st.columns([1, 3])
    sc1.markdown(
        f'<div style="background:rgba(255,255,255,0.04);border-radius:8px;'
        f'padding:16px;text-align:center">'
        f'  <div style="font-size:11px;color:#888;margin-bottom:4px">전반 감성</div>'
        f'  <div style="font-size:26px;font-weight:700;color:{sent_color}">{sent_label}</div>'
        f'  <div style="font-size:11px;color:#666;margin-top:4px">'
        f'긍정 {pos_cnt} · 부정 {neg_cnt} / {min(len(articles), 10)}건</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    with sc2:
        for a in articles[:5]:
            pub    = (a.get("published_at") or "")[:10]
            source = a.get("source", "")
            title  = a.get("title", "")
            url    = a.get("url", "")
            is_pos = any(w in title for w in pos_words)
            is_neg = any(w in title for w in neg_words)
            dot_color = "#26a69a" if is_pos else ("#ef5350" if is_neg else "#888")
            trust_icon = "✅" if (_trusted_sources and any(t in source for t in _trusted_sources)) else "⚠️"
            line = (
                f'<div style="display:flex;gap:8px;padding:5px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.07)">'
                f'  <span style="color:{dot_color};margin-top:2px">●</span>'
                f'  <div>'
                f'    <span style="color:#888;font-size:11px">{trust_icon} [{source}] {pub}</span><br>'
            )
            if url:
                line += f'    <a href="{url}" target="_blank" style="color:#ccc;font-size:13px;text-decoration:none">{title}</a>'
            else:
                line += f'    <span style="color:#ccc;font-size:13px">{title}</span>'
            line += "  </div></div>"
            st.markdown(line, unsafe_allow_html=True)
else:
    if total_collected > 0:
        st.caption("관련 뉴스를 찾을 수 없습니다. (종목명·티커가 포함된 기사 없음)")
    else:
        st.caption("최근 48시간 내 뉴스를 수집할 수 없습니다.")
    sent_label = "😐 중립"
    pos_cnt = neg_cnt = 0

st.divider()

# ── Section 3: 페어 상대 강도 ───────────────────────────────────────────────
st.markdown("### 3️⃣  페어 상대 강도")

with st.spinner("동종 종목 비교 중…"):
    peer_result = _load_peers(ticker)

if peer_result:
    z       = peer_result["zscore"]
    z_color = "#ef5350" if z > 0.5 else ("#26a69a" if z < -0.5 else "#9E9E9E")
    z_icon  = "🔴" if z > 1 else ("🟠" if z > 0.5 else ("🟢" if z < -0.5 else "⚪"))

    pc1, pc2, pc3 = st.columns([1, 1, 2])
    pc1.metric(
        "비교 종목",
        f"{peer_result['peer_name']}",
        f"{peer_result['peer']}",
    )
    pc2.metric(
        "Z-score (60일)",
        f"{z:+.2f} σ",
        f"업종: {peer_result['group']}",
    )
    with pc3:
        st.markdown(
            f'<div style="background:rgba({("239,83,80" if z>0.5 else "38,166,154" if z<-0.5 else "100,100,100")},0.12);'
            f'border-left:3px solid {z_color};border-radius:0 8px 8px 0;padding:10px 14px">'
            f'  {z_icon} {peer_result["direction"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Z-score 해석: |Z| > 2 = 통계적 이상 편차, 0.5~2 = 주의 구간, |Z| < 0.5 = 정상 범위  "
        f"(60일 rolling ratio 기준, cointegration 검정 없음)"
    )
else:
    st.caption(
        "이 종목에 대한 동종 비교 데이터가 없습니다.  \n"
        "📊 **페어 트레이딩** 페이지에서 직접 종목 쌍을 입력해 상세 분석을 받을 수 있습니다."
    )
    peer_result = None

st.divider()

# ── Section 4: 시장 분위기 ──────────────────────────────────────────────────
st.markdown("### 4️⃣  시장 전반 분위기")

fg_score = fg.score
fg_label = fg.label
fg_color = score_to_color(fg_score)
fg_vix   = fg.vix

fg_interp = (
    "극도의 공포 상태 — 역발상 매수 기회일 수 있음" if fg_score <= 20 else
    "공포 우세 — 시장 불안감이 높음. 신중한 접근 권장" if fg_score <= 40 else
    "중립 — 시장 방향성 불확실. 종목 개별 분석 중심으로" if fg_score <= 60 else
    "탐욕 우세 — 상승 기대감 강함. 과열 여부 주의" if fg_score <= 80 else
    "극도의 탐욕 — 시장 과열 신호. 포트폴리오 리스크 점검 권장"
)

fg1, fg2 = st.columns([1, 3])

with fg1:
    # Mini gauge
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
                {"range": [0,  20], "color": "#c62828"},
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
        height=160,
        margin=dict(l=20, r=20, t=20, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
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

st.divider()

# ── Section 5: 통합 AI 프롬프트 ────────────────────────────────────────────
st.markdown("### 5️⃣  통합 AI 분석 프롬프트")
st.caption("위 4개 섹션 데이터가 모두 포함된 프롬프트입니다. 복사 후 Claude.ai에 붙여넣으세요.")

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_prompt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # tech
    rsi_txt  = f"RSI(14): {rsi_val:.1f} ({rsi_interp})"
    macd_txt = f"MACD: {macd_v:.4f} / Signal: {macd_s:.4f} → {macd_interp}"
    bb_txt   = (
        f"볼린저밴드: 상단 {fmt_price(bb_up, ticker)} / 하단 {fmt_price(bb_lo, ticker)} "
        f"(%B {bb_pct:.2f}) → {bb_interp}"
    )
    ma_txt   = (
        f"MA5: {fmt_price(ma5, ticker)} / MA20: {fmt_price(ma20, ticker)} → {trend_interp}"
        if ma5 > 0 else "MA: 데이터 부족"
    )
    price_hist = (
        df.tail(5)[["Open", "High", "Low", "Close", "Volume"]].to_string(
            float_format=lambda x: f"{x:,.2f}"
        )
    )

    # news
    if articles:
        news_lines = "\n".join(
            f"- [{a.get('source', '')}] {a.get('title', '')} ({(a.get('published_at') or '')[:10]})"
            for a in articles[:5]
        )
        news_txt = (
            f"수집 {total_collected}건 → 관련 뉴스 {len(articles)}건\n"
            f"전반 감성: {sent_label} (긍정 {pos_cnt} · 부정 {neg_cnt} / {min(len(articles),10)}건)\n\n"
            f"{news_lines}"
        )
    else:
        news_txt = f"수집 {total_collected}건 → 관련 뉴스 없음 (종목명·티커 포함 기사 없음)"

    # peer
    if peer_result:
        peer_txt = (
            f"비교 종목: {peer_result['peer_name']} ({peer_result['peer']}) "
            f"[{peer_result['group']}]\n"
            f"60일 Ratio Z-score: {peer_result['zscore']:+.2f} σ\n"
            f"해석: {peer_result['direction']}"
        )
    else:
        peer_txt = "동종 비교 데이터 없음"

    # fg
    fg_txt = (
        f"CNN 공포·탐욕 지수: {fg_score}/100 ({fg_label})\n"
        f"해석: {fg_interp}\n"
        + (f"VIX: {fg_vix:.1f}\n" if fg_vix >= 0 else "")
        + f"업데이트: {fg.last_update}"
    )

    return f"""# 종합 투자 분석 요청 — {ticker} ({company})

**분석 시각:** {now}
**시장:** {market}  |  **통화:** {"KRW" if kr else "USD"}

---

## 1️⃣ 기술적 분석 (6개월 기준)

**현재가:** {fmt_price(close, ticker)} ({chg_pct:+.2f}%)
**기술적 점수:** {t_score:+.2f} → {sig_label}

| 지표 | 값 | 해석 |
|------|-----|------|
| {rsi_txt.split(":")[0]} | {rsi_val:.1f} | {rsi_interp} |
| MACD | {macd_v:.4f} / Sig {macd_s:.4f} | {macd_interp} |
| BB 위치 | %B {bb_pct:.2f} | {bb_interp} |
| MA5 / MA20 | {fmt_price(ma5, ticker)} / {fmt_price(ma20, ticker)} | {trend_interp} |

**최근 5거래일 가격:**

```
{price_hist}
```

---

## 2️⃣ 종목 관련 뉴스

{news_txt}

---

## 3️⃣ 동종 종목 상대 강도

{peer_txt}

---

## 4️⃣ 시장 전반 분위기

{fg_txt}

---

## 분석 요청

위 4개 섹션 데이터를 종합하여 다음을 분석해주세요:

1. **현재 투자 매력도** — 매수·관망·매도 중 판단과 근거 3가지
2. **가장 주목할 긍정 요소와 리스크** — 각 2가지씩
3. **시장 분위기가 이 종목에 미치는 영향** — F&G + 뉴스 감성 연계
4. **시나리오별 전망:**
   - 단기 (1주일): 예상 방향성과 주의 가격대
   - 중기 (1개월): 추세 유지 조건
   - 장기 (3개월): 섹터·시장 관점 평가
5. **투자자 유형별 조언:**
   - 단타 트레이더 (1주 이내): 진입 조건 + 손절가 + 목표가
   - 중장기 투자자 (3개월+): 분할 매수 전략

---

응답 마지막에 반드시 아래 JSON 블록을 포함해주세요:

```json
{{
  "signal": "BUY 또는 SELL 또는 HOLD",
  "confidence": 0.0~1.0,
  "target_price": 숫자,
  "stop_loss": 숫자,
  "hold_period_days": 숫자,
  "sentiment": "positive 또는 neutral 또는 negative",
  "key_catalysts": ["촉매1", "촉매2"],
  "key_risks": ["리스크1", "리스크2"],
  "reasons": ["근거1", "근거2", "근거3"]
}}
```"""


prompt = _build_prompt()

with st.expander("📄 프롬프트 내용 미리보기", expanded=False):
    st.code(prompt, language="markdown")

st.markdown("**모든 분석 데이터가 포함된 통합 프롬프트입니다. 아래 버튼을 클릭하세요:**")
copy_button(prompt, "📋 종합 분석 프롬프트 복사",
            gradient="linear-gradient(135deg,#1565C0,#0D47A1)")
st.caption("복사 후 Claude.ai (claude.ai)에 붙여넣으면 전체 종합 분석 결과를 받을 수 있습니다.")
