import sys
import os
from pathlib import Path

# Make project root importable from all page scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

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

# Authenticated: render logout button before handing off to page runner
with st.sidebar:
    st.divider()
    if st.button("로그아웃", width="stretch"):
        st.session_state["authenticated"] = False
        st.rerun()

pages = {
    "분석": [
        st.Page("pages/01_overview.py", title="종목 분석", icon="📈"),
    ],
    "AI": [
        st.Page("pages/06_chat.py", title="AI 채팅", icon="💬"),
    ],
}

pg = st.navigation(pages)
pg.run()
