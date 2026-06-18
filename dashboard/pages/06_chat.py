"""
Claude.ai 연동 종목 분석 페이지.
흐름: 종목 입력 → 데이터 수집 → 프롬프트 생성 → 클립보드 복사 + claude.ai 열기 → 답변 붙여넣기 → DB 저장
"""

import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from data.collectors.price_collector import PriceCollector
from data.collectors.news_collector import NewsCollector
from data.collectors.market_sentiment import MarketSentimentCollector, prompt_snippet
from analysis.technical.indicators import TechnicalIndicators
from analysis.technical.signals import score as tech_score
from analysis.ai.claude_analyst import ClaudeAnalyst
from memory.user_memory import UserMemory
from memory.pattern_analyzer import PatternAnalyzer
from utils.ticker_utils import detect_market, resolve_ticker as _resolve_base
from utils.clipboard import copy_button as _copy_button

# ── Session state ─────────────────────────────────────────────────────────────
if "prompt_data" not in st.session_state:
    st.session_state.prompt_data = None   # dict | None
if "save_count" not in st.session_state:
    st.session_state.save_count = 0
if "last_saved" not in st.session_state:
    st.session_state.last_saved = None    # success message | None

# ── Singletons ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_memory() -> UserMemory:
    return UserMemory()

@st.cache_resource
def get_pattern_analyzer() -> PatternAnalyzer:
    return PatternAnalyzer(get_memory())

memory = get_memory()
pattern_analyzer = get_pattern_analyzer()

# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_ticker(raw: str) -> tuple[str, str]:
    """
    한글 회사명 또는 원시 입력을 (ticker, market) 튜플로 변환한다.
    시장은 티커 접미사로 자동 감지: .KS→KOSPI, .KQ→KOSDAQ, else→NASDAQ.
    """
    ticker = _resolve_base(raw)
    return ticker, detect_market(ticker)


def copy_open_button(prompt: str) -> None:
    _copy_button(prompt, "📋 프롬프트 복사", height=60)


def collect_and_build(ticker: str, market: str) -> dict | None:
    """
    Fetches price data + news + Fear&Greed, builds prompt.
    Returns a dict with all context needed for the page, or None on failure.
    """
    collector = PriceCollector()
    df = collector.fetch(ticker, period="6mo")
    if df.empty:
        st.error(f"❌ '{ticker}' 가격 데이터를 불러올 수 없습니다. 티커 코드를 확인하세요.")
        return None

    df = TechnicalIndicators().compute(df)
    last_row = df.iloc[-1]
    t_score = float(tech_score(last_row))

    # Company info
    info = collector.get_info(ticker)
    company_name = info.get("shortName") or info.get("longName") or ticker

    # News (best-effort, never blocks) — 24h window
    articles: list[dict] = []
    try:
        news_col = NewsCollector()
        raw_articles = news_col.fetch_by_ticker(ticker, market, hours=24)
        articles = news_col.to_dicts(raw_articles)
    except Exception:
        pass

    fg_text = ""
    try:
        fg_text = prompt_snippet(MarketSentimentCollector().fetch())
    except Exception:
        pass

    analyst = ClaudeAnalyst(memory=memory)
    prompt = analyst.build_prompt(ticker, df, articles, market, fg_text)

    # MACD cross label for DB record
    try:
        macd_cross = (
            "골든크로스" if float(last_row.get("MACD", 0)) > float(last_row.get("MACD_Signal", 0))
            else "데드크로스"
        )
    except (TypeError, ValueError):
        macd_cross = ""

    return {
        "ticker": ticker,
        "market": market,
        "company_name": company_name,
        "prompt": prompt,
        "t_score": t_score,
        "last_row": last_row.to_dict(),
        "macd_cross": macd_cross,
        "articles": articles,
        "news_count": len(articles),
        "generated_at": datetime.now().strftime("%H:%M:%S"),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 종목 분석")

    ticker_input = st.text_input(
        "종목 코드 또는 한글 이름",
        placeholder="예: 삼성전자 / 005930.KS / NVDA / 엔비디아",
    )

    # Preview resolved ticker (market auto-detected from ticker format)
    if ticker_input.strip():
        resolved, resolved_market = resolve_ticker(ticker_input)
        if resolved != ticker_input.strip().upper():
            st.caption(f"→ 인식된 티커: **{resolved}** ({resolved_market})")

    analyze_clicked = st.button("📊 분석 시작", width="stretch", type="primary")

    st.divider()
    st.subheader("🧠 투자 메모리")
    stats = memory.stats()
    c1, c2 = st.columns(2)
    c1.metric("총 기록", stats["decisions"])
    c2.metric("결과 확인", stats["outcomes_recorded"])
    c3, c4 = st.columns(2)
    c3.metric("활성 규칙", stats["active_rules"])
    c4.metric("패턴", stats["patterns"])

    if st.button("🔄 패턴 재분석", width="stretch"):
        r = pattern_analyzer.analyze_patterns()
        st.success(f"패턴 {r['patterns']}개 / 규칙 {r['rules']}개 업데이트")

    # Recent decisions
    recent = memory.get_recent_decisions(5)
    if recent:
        st.divider()
        st.subheader("최근 투자 기록")
        for d in recent:
            outcome = ""
            if d.get("outcome_pct") is not None:
                outcome = f" → {d['outcome_pct']:+.1f}%"
            label_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(d["action"], "⚪")
            st.caption(
                f"{label_color} {d['ticker']} **{d['action']}** "
                f"{d['created_at'][:10]}{outcome}"
            )

# ── Trigger analysis ──────────────────────────────────────────────────────────
if analyze_clicked:
    if not ticker_input.strip():
        st.warning("종목 코드 또는 이름을 입력하세요.")
    else:
        resolved, resolved_market = resolve_ticker(ticker_input)
        with st.spinner(f"📡 {resolved} 데이터 수집 중… (가격 + 지표 + 뉴스)"):
            data = collect_and_build(resolved, resolved_market)
        if data:
            st.session_state.prompt_data = data
            st.session_state.last_saved = None  # clear any old save message

# ── Main content ──────────────────────────────────────────────────────────────
st.title("💬 Claude.ai로 종목 분석")

if st.session_state.prompt_data is None:
    st.info(
        "👈 사이드바에서 종목 코드(또는 한글 이름)를 입력하고 **분석 시작**을 클릭하세요.\n\n"
        "예시: `삼성전자`, `005930.KS`, `NVDA`, `엔비디아`"
    )
    st.stop()

# ── Step 1: Show prompt ───────────────────────────────────────────────────────
pdata    = st.session_state.prompt_data
ticker   = pdata["ticker"]
market   = pdata["market"]
company  = pdata["company_name"]
t_score  = pdata["t_score"]
last_row = pdata["last_row"]
prompt   = pdata["prompt"]

# Header row
col_title, col_time = st.columns([3, 1])
with col_title:
    st.subheader(f"1️⃣  {company} ({ticker}) — 분석 프롬프트 준비 완료")
with col_time:
    st.caption(f"생성: {pdata['generated_at']}  |  뉴스: {pdata['news_count']}건")

# Tech signal mini-summary
try:
    rsi_val = float(last_row.get("RSI", 50))
    macd_v  = float(last_row.get("MACD", 0))
    macd_s  = float(last_row.get("MACD_Signal", 0))
    close_v = float(last_row.get("Close", 0))
except (TypeError, ValueError):
    rsi_val, macd_v, macd_s, close_v = 50, 0, 0, 0

signal_emoji = "🟢" if t_score > 0.15 else ("🔴" if t_score < -0.15 else "🟡")
m1, m2, m3, m4 = st.columns(4)
m1.metric("현재가", f"{close_v:,.0f}")
m2.metric("기술 점수", f"{t_score:+.2f}", f"{signal_emoji}")
m3.metric("RSI", f"{rsi_val:.1f}", "과매도" if rsi_val < 30 else ("과매수" if rsi_val > 70 else "중립"))
m4.metric("MACD", "골든크로스" if macd_v > macd_s else "데드크로스")

st.divider()

# ── News headlines (참고용, 분석 흐름과 독립) ────────────────────────────────
articles = pdata.get("articles", [])
if articles:
    with st.expander(f"📰 최근 뉴스 헤드라인 ({len(articles)}건, 최근 24시간)", expanded=False):
        for i, a in enumerate(articles[:10], 1):
            pub = a.get("published_at", "")[:10] if a.get("published_at") else ""
            source = a.get("source", "")
            title = a.get("title", "")
            url = a.get("url", "")
            if url:
                st.markdown(f"{i}. **[{source}]** [{title}]({url}) <small>{pub}</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"{i}. **[{source}]** {title} <small>{pub}</small>", unsafe_allow_html=True)
else:
    with st.expander("📰 최근 뉴스 헤드라인 (0건)", expanded=False):
        st.caption("최근 24시간 내 관련 뉴스를 찾을 수 없습니다.")

st.divider()

# Prompt preview (collapsible) + copy button
with st.expander("📄 프롬프트 내용 미리보기", expanded=False):
    st.code(prompt, language="markdown")

st.markdown("**Claude.ai에 붙여넣을 프롬프트가 준비됐습니다. 아래 버튼을 클릭하세요:**")
copy_open_button(prompt)
st.caption("버튼 클릭 후 Claude.ai에서 직접 붙여넣어주세요 (Ctrl+V)")

st.divider()

# ── Step 2: Paste response ────────────────────────────────────────────────────
st.subheader("2️⃣  Claude 답변 붙여넣기")
st.caption("Claude.ai에서 받은 답변을 아래에 붙여넣으세요. JSON 블록이 있으면 자동으로 파싱합니다.")

paste_key = f"paste_{st.session_state.save_count}"
paste_text = st.text_area(
    "Claude 답변",
    height=300,
    placeholder="여기에 Claude.ai 답변을 붙여넣으세요 (Ctrl+V)…",
    key=paste_key,
    label_visibility="collapsed",
)

if paste_text.strip():
    analyst = ClaudeAnalyst(memory=memory)
    parsed = analyst.parse_response(paste_text, ticker)

    # Auto-parse result display
    st.markdown("---")
    sig_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(parsed.signal, "⚪")
    sent_emoji = {"positive": "😀 긍정", "negative": "😟 부정", "neutral": "😐 중립"}.get(
        parsed.sentiment or "neutral", "😐 중립"
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("감지된 시그널", f"{sig_color} {parsed.signal}")
    r2.metric("확신도", f"{parsed.confidence:.0%}")
    r3.metric("뉴스 감성", sent_emoji)
    r4.metric(
        "목표가",
        f"{parsed.price_target:,.0f}" if parsed.price_target else "—",
    )

    if parsed.reasons and parsed.reasons[0] != "JSON 블록을 찾지 못했습니다 — 아래에서 직접 시그널을 선택하세요.":
        st.markdown("**분석 근거:**")
        for r in parsed.reasons:
            st.markdown(f"- {r}")

    # ── Decision recording form ────────────────────────────────────────────
    st.divider()
    st.markdown("**투자 결정을 기록하세요** (패턴 학습에 사용됩니다)")

    with st.form("record_decision"):
        fc1, fc2 = st.columns([1, 2])
        with fc1:
            action = st.radio(
                "실제 투자 결정",
                ["BUY", "HOLD", "SELL"],
                index=["BUY", "HOLD", "SELL"].index(parsed.signal)
                if parsed.signal in ["BUY", "HOLD", "SELL"] else 1,
            )
        with fc2:
            default_price = close_v if close_v > 0 else 0.0
            price_input = st.number_input(
                "진입가 (선택)",
                value=default_price,
                min_value=0.0,
                format="%.2f",
            )

        default_reason = "\n".join(parsed.reasons) if parsed.reasons else ""
        reason_input = st.text_area(
            "결정 이유",
            value=default_reason,
            height=100,
            placeholder="예: RSI 과매도 + 실적 기대감",
        )

        save_btn = st.form_submit_button("✅ 저장하기", width="stretch", type="primary")

        if save_btn:
            try:
                rsi_db = float(last_row.get("RSI")) if last_row.get("RSI") is not None else None
            except (TypeError, ValueError):
                rsi_db = None

            decision_id = memory.record_decision(
                ticker=ticker,
                market=market,
                signal=parsed.signal,
                action=action,
                reason=reason_input,
                price=price_input if price_input > 0 else None,
                sector=pdata.get("sector", ""),
                rsi=rsi_db,
                macd_cross=pdata["macd_cross"],
                context={
                    "t_score": t_score,
                    "confidence": parsed.confidence,
                    "price_target": parsed.price_target,
                    "news_count": pdata["news_count"],
                    "news_sentiment": parsed.sentiment or "neutral",
                },
            )
            st.session_state.last_saved = (
                f"✅ **{ticker} {action}** 결정 저장 완료 (ID: {decision_id})  \n"
                f"나중에 결과를 업데이트하면 나만의 패턴 분석에 반영됩니다."
            )
            st.session_state.save_count += 1  # clears paste area on rerun
            st.rerun()

# Success banner (shown after save)
if st.session_state.last_saved:
    st.success(st.session_state.last_saved)
    if st.button("새 종목 분석하기"):
        st.session_state.prompt_data = None
        st.session_state.last_saved = None
        st.rerun()
