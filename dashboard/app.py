import sys
import os
from pathlib import Path

# Make project root importable from all page scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from dotenv import load_dotenv
from data.collectors.market_sentiment import MarketSentimentCollector, score_to_color

load_dotenv()

st.set_page_config(
    page_title="Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_PASSWORD = os.getenv("APP_PASSWORD", "")

_HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] { display: none; }
[data-testid="collapsedControl"] { display: none; }
</style>
"""


def _show_login() -> None:
    st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)

    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        st.title("📈 Stock AI")
        st.subheader("로그인")

        if not _APP_PASSWORD:
            st.warning("APP_PASSWORD가 설정되지 않았습니다. .env 파일을 확인해주세요.")

        with st.form("login_form"):
            password = st.text_input("비밀번호", type="password")
            submitted = st.form_submit_button("로그인", width="stretch")

        if submitted:
            if _APP_PASSWORD and password == _APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_fear_greed() -> dict:
    fg = MarketSentimentCollector().fetch()
    return {"score": fg.score, "label": fg.label, "vix": fg.vix}


if not st.session_state.get("authenticated", False):
    _show_login()
    st.stop()

# Authenticated: sidebar Fear & Greed widget + logout
with st.sidebar:
    try:
        _fg = _load_fear_greed()
        _score = _fg["score"]
        _label = _fg["label"]
        _color = score_to_color(_score)
        _vix_txt = f"VIX {_fg['vix']:.1f}  ·  " if _fg["vix"] >= 0 else ""
        st.markdown(
            f"""<div style="padding:10px 4px 4px 4px">
            <div style="font-size:11px;color:#888;margin-bottom:6px">📊 공포·탐욕 지수</div>
            <div style="font-size:26px;font-weight:700;color:{_color};line-height:1.1">
              {_score}
              <span style="font-size:13px;font-weight:400;color:{_color}">{_label}</span>
            </div>
            <div style="background:linear-gradient(to right,#c62828,#e64a19,#f9a825,#558b2f,#00695c);
                        height:5px;border-radius:3px;margin:8px 0 4px 0;position:relative">
              <div style="position:absolute;left:{_score}%;transform:translateX(-50%);top:-4px;
                          width:3px;height:13px;background:white;border-radius:2px"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#555">
              <span>극도공포</span><span>중립</span><span>극도탐욕</span>
            </div>
            <div style="font-size:10px;color:#555;margin-top:4px">{_vix_txt}1시간 캐시</div>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    st.divider()
    if st.button("로그아웃", width="stretch"):
        st.session_state["authenticated"] = False
        st.rerun()

pages = {
    "분석": [
        st.Page("pages/01_overview.py",      title="종목 분석",    icon="📈"),
        st.Page("pages/09_scalping.py",      title="단타 분석",    icon="⚡"),
        st.Page("pages/07_pairs_trading.py", title="페어 트레이딩", icon="📊"),
        st.Page("pages/08_portfolio.py",     title="포트폴리오",   icon="💼"),
        st.Page("pages/10_backtest.py",      title="백테스팅",     icon="🔬"),
    ],
    "AI": [
        st.Page("pages/06_chat.py", title="AI 채팅", icon="💬"),
    ],
}

pg = st.navigation(pages)
pg.run()
