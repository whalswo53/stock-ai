import sys
from pathlib import Path

# Make project root importable from all page scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
