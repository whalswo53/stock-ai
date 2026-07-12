"""
종합 분석 페이지.

core/analysis_modules.py에 @register된 모든 분석 모듈을 순회(run_all)해서
그린다 — 개별 모듈을 이 페이지가 손으로 골라 호출하지 않으므로, 새 분석
모듈에 @register만 붙이면 이 페이지에 자동으로 나타난다.
(종합분석 ≡ 등록된 모든 모듈의 합집합)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

import core.analysis_modules  # noqa: F401 — import 시 @register 모듈들이 등록됨
from analysis.technical.indicators import TechnicalIndicators
from core.analysis_registry import aggregate_verdict, run_all
from data.collectors.price_collector import PriceCollector
from config.sources import TICKER_KR_NAME
from ui.components import render_signal_card, render_verdict_banner
from utils.clipboard import copy_button
from utils.ticker_utils import detect_market, is_kr, resolve_currency
from utils.search_widget import ticker_search_widget

_POLARITY_VALUE_LABEL = {"bullish": "강세", "neutral": "중립", "bearish": "약세"}

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
        "⬆ 종목을 입력하면 등록된 모든 분석 모듈이 자동으로 채워집니다.\n\n"
        "새 분석을 추가하려면 `core/analysis_modules.py`에 `@register`만 붙이면 "
        "이 페이지에 자동으로 나타납니다."
    )

market = detect_market(ticker)
kr     = is_kr(ticker)


# ── Price/info load (ctx의 기초 데이터) ────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_price(ticker: str):
    pc  = PriceCollector()
    df  = pc.fetch(ticker, period="6mo")
    if not df.empty:
        df = TechnicalIndicators().compute(df)
    info = pc.get_info(ticker)
    return df, info


# ── Page title ────────────────────────────────────────────────────────────────
st.title("🎯  종합 분석")

with st.spinner(f"'{ticker}' 데이터 수집 중…"):
    df, info = _load_price(ticker)

if df.empty:
    st.error(f"'{ticker}' 가격 데이터를 불러올 수 없습니다. 티커 코드를 확인하세요.")
    st.stop()

company = TICKER_KR_NAME.get(ticker) or info.get("shortName") or info.get("longName") or ticker
currency_code, currency_symbol = resolve_currency(info, kr)

st.subheader(f"{company} ({ticker})  ·  {market}  ·  {currency_code}")

# ── ctx 조립 (모든 등록 모듈이 이 규격을 공유) ─────────────────────────────────
ctx = {
    "ticker": ticker,
    "df": df,
    "info": info,
    "currency": currency_code,
    "symbol": currency_symbol,
    "is_korean": kr,
    "market": market,
}

with st.spinner("등록된 분석 모듈 실행 중… (기술적 · 캔들 · 밸류 · 뉴스 · 상대강도 · 시장분위기)"):
    results = run_all(ctx)

_NUM_EMOJI = [f"{i}️⃣" for i in range(1, 10)]  # 1️⃣ 2️⃣ … 9️⃣

# ── 종합 판정 배너 ────────────────────────────────────────────────────────────
verdict = aggregate_verdict(results)
render_verdict_banner(
    verdict["label"], verdict["confidence"], verdict["polarity"],
    sub_text=(
        f"강세 {verdict['n_bull']} · 중립 {verdict['n_neu']} · 약세 {verdict['n_bear']}"
        f"  (총 {verdict['total']}개 신호 집계, 방향성 없는 모듈 제외)"
    ),
)

# ── 신호 pill 한 줄 — results 길이만큼 자동 생성. 등록 모듈이 늘어나면 그대로 늘어난다 ──
cols = st.columns(len(results))
for col, r in zip(cols, results):
    with col:
        render_signal_card(
            r.title,
            _POLARITY_VALUE_LABEL.get(r.polarity, "정보"),
            "",
            polarity=r.polarity,
        )

# ── 세부 — 탭 목록도 results에서 자동 생성 ───────────────────────────────────────
tabs = st.tabs([r.title for r in results])
for tab, r in zip(tabs, results):
    with tab:
        if r.render:
            r.render()
        else:
            st.markdown(r.markdown)

# ── 대시보드용 통합 JSON ────────────────────────────────────────────────────────
merged_json: dict = {}
for r in results:
    merged_json.update(r.json)


# ── Section: 통합 AI 프롬프트 ───────────────────────────────────────────────────
_num_label = _NUM_EMOJI[len(results)] if len(results) < len(_NUM_EMOJI) else f"{len(results) + 1}."
st.markdown(f"### {_num_label}  통합 AI 분석 프롬프트")
st.caption(f"위 {len(results)}개 섹션 데이터가 모두 포함된 프롬프트입니다. 복사 후 Claude.ai에 붙여넣으세요.")


def _build_prompt() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sections_md = "\n\n---\n\n".join(
        f"## {i + 1}️⃣ {r.title}\n\n{r.markdown}" for i, r in enumerate(results)
    )

    return f"""# 종합 투자 분석 요청 — {ticker} ({company})

**분석 시각:** {now}
**시장:** {market}  |  **통화:** {currency_code}

---

{sections_md}

---

## 분석 요청

위 {len(results)}개 섹션 데이터를 종합하여 다음을 분석해주세요.
**이 종목의 섹터·상장지역 특성상 관련된 거시·지정학·규제 리스크를 반영해 분석하라.**

1. **현재 투자 매력도** — 매수·관망·매도 중 판단과 근거 3가지
2. **가장 주목할 긍정 요소와 리스크** — 각 2가지씩
3. **시장 분위기가 이 종목에 미치는 영향** — F&G + 뉴스 감성 연계
4. **밸류에이션 관점** — 현재 밸류가 기술적/뉴스 시그널과 부합하는지
5. **시나리오별 전망:**
   - 단기 (1주일): 예상 방향성과 주의 가격대
   - 중기 (1개월): 추세 유지 조건
   - 장기 (3개월): 섹터·시장 관점 평가
6. **투자자 유형별 조언:**
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
