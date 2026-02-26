"""
台股分析儀表板 — 進入點
技術架構：Streamlit + fugle-marketdata + Plotly

頁面模組：
  Single_stock_page.py  →  📈 單股分析
  Screener_page.py      →  🔍 選股策略
  Score_page.py         →  🎯 綜合評分
共用工具：
  utils.py              →  資料層 + 技術指標計算
"""

import streamlit as st

from Single_stock_page import render_single_stock_page
from Screener_page import render_screener_page
from Score_page import render_score_page


def main() -> None:
    st.set_page_config(
        page_title="台股分析儀表板",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 台股分析儀表板")
    st.caption("資料來源：Fugle Market Data API")

    tab_single, tab_screener, tab_score = st.tabs(
        ["📈 單股分析", "🔍 選股策略", "🎯 綜合評分"]
    )

    with tab_single:
        render_single_stock_page()

    with tab_screener:
        render_screener_page()

    with tab_score:
        render_score_page()


if __name__ == "__main__":
    main()
