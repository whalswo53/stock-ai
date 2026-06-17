"""
Claude.ai 연동 종목 분석 페이지.
흐름: 종목 입력 → 데이터 수집 → 프롬프트 생성 → 클립보드 복사 + claude.ai 열기 → 답변 붙여넣기 → DB 저장
"""

import base64
import re
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st
import streamlit.components.v1 as components

from config.sources import KOSPI_TICKER_MAP, NASDAQ_TICKER_MAP
from data.collectors.price_collector import PriceCollector
from data.collectors.news_collector import NewsCollector
from analysis.technical.indicators import TechnicalIndicators
from analysis.technical.signals import score as tech_score
from analysis.ai.claude_analyst import ClaudeAnalyst
from memory.user_memory import UserMemory
from memory.pattern_analyzer import PatternAnalyzer

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

def resolve_ticker(raw: str, market: str) -> tuple[str, str]:
    """
    Resolves Korean company name or raw ticker code to (ticker, market).
    Returns (raw, market) unchanged if no mapping found.
    """
    raw = raw.strip()

    # Already a ticker code — pass through
    if re.match(r"^\d{6}\.(KS|KQ)$", raw, re.IGNORECASE):
        mkt = "KOSDAQ" if raw.upper().endswith(".KQ") else "KOSPI"
        return raw.upper(), mkt
    if re.match(r"^[A-Z]{1,5}$", raw):
        return raw.upper(), market

    # Korean name lookup
    sorted_kospi = sorted(KOSPI_TICKER_MAP.keys(), key=len, reverse=True)
    for name in sorted_kospi:
        if name in raw:
            t = KOSPI_TICKER_MAP[name]
            return t, "KOSDAQ" if t.endswith(".KQ") else "KOSPI"

    sorted_nasdaq = sorted(NASDAQ_TICKER_MAP.keys(), key=len, reverse=True)
    for name in sorted_nasdaq:
        if name in raw:
            return NASDAQ_TICKER_MAP[name], "NASDAQ"

    return raw, market


def copy_open_button(prompt: str) -> None:
    """
    Renders a single HTML button that:
    1. Copies the prompt to clipboard (execCommand fallback — works in iframes)
    2. Opens claude.ai/new in a new tab
    """
    b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    html = f"""
<style>
  .claude-btn {{
    background: linear-gradient(135deg, #7C3AED 0%, #5B21B6 100%);
    color: #fff;
    border: none;
    padding: 14px 0;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 700;
    cursor: pointer;
    width: 100%;
    letter-spacing: 0.3px;
    transition: opacity .15s;
  }}
  .claude-btn:hover {{ opacity: .88; }}
  .claude-btn:active {{ opacity: .75; transform: scale(.99); }}
</style>
<button class="claude-btn" onclick="(function(btn){{
  var text = atob('{b64}');
  var ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.focus(); ta.select();
  try {{ document.execCommand('copy'); }} catch(e) {{}}
  document.body.removeChild(ta);
  window.open('https://claude.ai/new', '_blank');
  btn.textContent = '✅ 복사 완료 — Claude.ai가 열렸습니다! Ctrl+V 후 전송하세요';
  setTimeout(function(){{
    btn.textContent = '📋 프롬프트 복사 + Claude.ai 열기';
  }}, 4000);
}})(this)">📋 프롬프트 복사 + Claude.ai 열기</button>
"""
    components.html(html, height=62)


def collect_and_build(ticker: str, market: str) -> dict | None:
    """
    Fetches price data + news, computes indicators, builds prompt.
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

    # News (best-effort, never blocks)
    articles: list[dict] = []
    try:
        news_col = NewsCollector()
        raw_articles = news_col.fetch_by_ticker(ticker, market, hours=48)
        articles = news_col.to_dicts(raw_articles)
    except Exception:
        pass

    analyst = ClaudeAnalyst(memory=memory)
    prompt = analyst.build_prompt(ticker, df, articles, market)

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
        "news_count": len(articles),
        "generated_at": datetime.now().strftime("%H:%M:%S"),
    }

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 종목 분석")

    market_sel = st.radio("시장", ["KOSPI", "NASDAQ", "KOSDAQ"], horizontal=True)

    ticker_input = st.text_input(
        "종목 코드 또는 한글 이름",
        placeholder="예: 삼성전자 / 005930.KS / NVDA",
    )

    # Preview resolved ticker
    if ticker_input.strip():
        resolved, resolved_market = resolve_ticker(ticker_input, market_sel)
        if resolved != ticker_input.strip():
            st.caption(f"→ 인식된 티커: **{resolved}** ({resolved_market})")
        else:
            resolved_market = market_sel

    analyze_clicked = st.button("📊 분석 시작", use_container_width=True, type="primary")

    st.divider()
    st.subheader("🧠 투자 메모리")
    stats = memory.stats()
    c1, c2 = st.columns(2)
    c1.metric("총 기록", stats["decisions"])
    c2.metric("결과 확인", stats["outcomes_recorded"])
    c3, c4 = st.columns(2)
    c3.metric("활성 규칙", stats["active_rules"])
    c4.metric("패턴", stats["patterns"])

    if st.button("🔄 패턴 재분석", use_container_width=True):
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
        resolved, resolved_market = resolve_ticker(ticker_input, market_sel)
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
pd = st.session_state.prompt_data
ticker   = pd["ticker"]
market   = pd["market"]
company  = pd["company_name"]
t_score  = pd["t_score"]
last_row = pd["last_row"]
prompt   = pd["prompt"]

# Header row
col_title, col_time = st.columns([3, 1])
with col_title:
    st.subheader(f"1️⃣  {company} ({ticker}) — 분석 프롬프트 준비 완료")
with col_time:
    st.caption(f"생성: {pd['generated_at']}  |  뉴스: {pd['news_count']}건")

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

# Prompt preview (collapsible) + copy button
with st.expander("📄 프롬프트 내용 미리보기", expanded=False):
    st.code(prompt, language="markdown")

st.markdown("**Claude.ai에 붙여넣을 프롬프트가 준비됐습니다. 아래 버튼을 클릭하세요:**")
copy_open_button(prompt)
st.caption("버튼 클릭 → 프롬프트 자동 복사 + claude.ai 새 탭 열림 → Ctrl+V → 전송")

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
    r1, r2, r3 = st.columns(3)
    r1.metric("감지된 시그널", f"{sig_color} {parsed.signal}")
    r2.metric("확신도", f"{parsed.confidence:.0%}")
    r3.metric(
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

        save_btn = st.form_submit_button("✅ 저장하기", use_container_width=True, type="primary")

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
                sector=pd.get("sector", ""),
                rsi=rsi_db,
                macd_cross=pd["macd_cross"],
                context={
                    "t_score": t_score,
                    "confidence": parsed.confidence,
                    "price_target": parsed.price_target,
                    "news_count": pd["news_count"],
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
