"""
台股分析儀表板 — 進入點
技術架構：Streamlit + fugle-marketdata + Plotly

頁面模組：
  Single_stock_page.py  →  📈 單股分析
  Screener_page.py      →  🔍 選股策略
  Score_page.py         →  🎯 綜合評分
  chips_analyzer.py     →  🏦 籌碼分析
  backtester.py         →  🔁 策略回測
  financial_report.py   →  📋 財務報告
  news_finder.py        →  📰 財經新聞
  valuation_analyzer.py →  💎 估值分析
共用工具：
  utils.py              →  資料層 + 技術指標計算
"""

import streamlit as st

from Single_stock_page import render_single_stock_page
from Screener_page import render_screener_page
from Score_page import render_score_page
from chips_analyzer import render_chips_page
from backtester import render_backtest_page
from financial_report import render_financial_page
from news_finder import render_news_page
from valuation_analyzer import render_valuation_page


def main() -> None:
    st.set_page_config(
        page_title="台股分析儀表板",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 台股分析儀表板")
    st.caption("資料來源：Fugle Market Data API")

    # ── 側邊欄開關：隱藏耗時的策略分析頁籤 ─────────────────────────
    with st.sidebar:
        hide_strategy = st.toggle(
            "隱藏策略分析頁籤",
            value=True,
            help="開啟時暫時隱藏「🔍 選股策略」與「🔁 策略回測」，加快頁面載入速度。",
        )

    # ── 依開關狀態動態組建 Tab 列表 ──────────────────────────────────
    if hide_strategy:
        (tab_single, tab_score, tab_chips,
         tab_fin, tab_news, tab_valuation) = st.tabs([
            "📈 單股分析", "🎯 綜合評分", "🏦 籌碼分析",
            "📋 財務報告", "📰 財經新聞", "💎 估值分析",
        ])
        tab_screener = None
        tab_backtest = None
    else:
        (tab_single, tab_screener, tab_score,
         tab_chips, tab_backtest, tab_fin, tab_news, tab_valuation) = st.tabs([
            "📈 單股分析", "🔍 選股策略", "🎯 綜合評分",
            "🏦 籌碼分析", "🔁 策略回測", "📋 財務報告", "📰 財經新聞", "💎 估值分析",
        ])

    with tab_single:
        render_single_stock_page()

    if tab_screener is not None:
        with tab_screener:
            render_screener_page()

    with tab_score:
        render_score_page()

    with tab_chips:
        render_chips_page()

    if tab_backtest is not None:
        with tab_backtest:
            render_backtest_page()

    with tab_fin:
        render_financial_page()

    with tab_news:
        render_news_page()

    with tab_valuation:
        render_valuation_page()


if __name__ == "__main__":
    main()
