import sys
import os
import faulthandler
from pathlib import Path

# Make project root importable from all page scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# 세그폴트/SIGABRT 등 네이티브 크래시 시 C 레벨 스택을 파일로 남긴다.
# 로컬에서 재현이 안 되는 크래시라 다음 발생 시 정확한 크래시 지점을
# 잡기 위한 계측 — 평상시엔 오버헤드가 사실상 없다.
_faulthandler_log = Path(__file__).resolve().parent.parent / "logs" / "faulthandler.log"
_faulthandler_log.parent.mkdir(exist_ok=True)

# 터미널 접근이 안 되는 배포 환경(Streamlit Cloud)에서도 이전 실행의 크래시
# 로그를 볼 수 있도록, 재시작 시 파일 내용을 표준 로그 스트림에 그대로 찍는다.
if os.path.exists(_faulthandler_log):
    with open(_faulthandler_log) as f:
        content = f.read()
    if content.strip():
        print("=== PREVIOUS CRASH LOG ===")
        print(content)
        print("=== END CRASH LOG ===")

faulthandler.enable(file=open(_faulthandler_log, "a"), all_threads=True)

import streamlit as st
from dotenv import load_dotenv

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


if not st.session_state.get("authenticated", False):
    _show_login()
    st.stop()

# Authenticated: logout (F&G는 종합분석 페이지에 이미 섹션으로 있어 중복 제거됨)
with st.sidebar:
    if st.button("로그아웃", width="stretch"):
        st.session_state["authenticated"] = False
        st.rerun()

pages = {
    "분석": [
        st.Page("pages/03_comprehensive.py", title="종합 분석",     icon="🎯"),
        st.Page("pages/01_overview.py",      title="장기/스윙 분석", icon="📈"),
        st.Page("pages/09_scalping.py",      title="단타 분석",     icon="⚡"),
        st.Page("pages/07_pairs_trading.py", title="페어 트레이딩",  icon="📊"),
        st.Page("pages/08_portfolio.py",     title="포트폴리오",    icon="💼"),
        st.Page("pages/10_backtest.py",      title="백테스팅",      icon="🔬"),
    ],
    "AI": [
        st.Page("pages/06_chat.py", title="AI 채팅", icon="💬"),
    ],
}

pg = st.navigation(pages)
pg.run()
