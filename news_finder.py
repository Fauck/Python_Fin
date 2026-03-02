"""
關鍵字財經新聞頁面（Tab 7）。
使用 feedparser + requests 解析 Google News RSS，取得繁體中文財經新聞。

設計原則
--------
- 100% 使用者主動觸發（無任何排程、定時刷新、或背景通知邏輯）
- 僅解析公開 RSS 端點，不爬取任何新聞網站的 HTML DOM
- @st.cache_data(ttl=900) 避免短時間內重複呼叫 Google RSS
"""

import html
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List
from urllib.parse import quote

import feedparser
import requests
import streamlit as st


# ── Google News RSS 端點（繁體中文台灣版）
_GNEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q={kw}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
)

# ── requests 共用 Header（模擬瀏覽器，降低被封鎖機率）
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
}


# ═════════════════════════════════════════════
# 資料層：RSS 抓取與解析
# ═════════════════════════════════════════════

def _parse_published(raw: str) -> str:
    """
    將 RFC 2822 格式的日期字串轉為 YYYY-MM-DD HH:MM。
    轉換失敗時直接回傳原始字串前 16 字元。
    """
    try:
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw[:16] if len(raw) >= 16 else raw


@st.cache_data(ttl=900)   # 15 分鐘快取，避免頻繁呼叫 Google RSS
def get_keyword_news(keyword: str, limit: int = 15) -> List[Dict[str, str]]:
    """
    透過 Google News RSS 取得關鍵字相關新聞。

    流程
    ----
    1. 關鍵字 URL 編碼 → 組成 Google News RSS URL
    2. 使用 requests（含自訂 User-Agent）取得 RSS XML
    3. feedparser 解析 XML，提取 title / link / published / source
    4. HTML 實體解碼（如 &amp; → &）
    5. 截取前 limit 則回傳

    Parameters
    ----------
    keyword : 搜尋關鍵字（繁體中文或英文，空格亦可）
    limit   : 最多回傳幾則，預設 15

    Returns
    -------
    List of dicts:
        title     : 新聞標題（已解碼 HTML 實體）
        published : 格式化發布時間（YYYY-MM-DD HH:MM）
        link      : Google News 轉址連結（點擊後前往原始來源）
        source    : 新聞來源媒體名稱（若有）
    """
    url = _GNEWS_RSS.format(kw=quote(keyword.strip()))

    try:
        # ── 以 requests 取得 RSS XML（更穩定的 HTTP 客戶端）
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except requests.RequestException:
        # 網路異常或 HTTP 錯誤時直接讓 feedparser 自行嘗試
        try:
            feed = feedparser.parse(url)
        except Exception:
            return []
    except Exception:
        return []

    if not feed.entries:
        return []

    results: List[Dict[str, str]] = []
    for entry in feed.entries[:limit]:
        title     = html.unescape(getattr(entry, "title",     "（無標題）"))
        link      = getattr(entry, "link",      "#")
        published = _parse_published(getattr(entry, "published", ""))

        # Google News RSS 的 source 是巢狀結構
        source_obj = getattr(entry, "source", None)
        if source_obj:
            source = (
                source_obj.get("title", "")
                if isinstance(source_obj, dict)
                else getattr(source_obj, "title", "")
            )
        else:
            source = ""

        results.append({
            "title":     title,
            "published": published,
            "link":      link,
            "source":    str(source),
        })

    return results


# ═════════════════════════════════════════════
# UI 層：新聞頁面渲染
# ═════════════════════════════════════════════

def render_news_page() -> None:
    """關鍵字財經新聞頁面（Tab 7）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 搜尋條件")
        keyword = st.text_input(
            "搜尋關鍵字",
            value="台積電",
            key="news_keyword",
            help=(
                "支援中英文，多個關鍵字以空格分隔。\n"
                "例如：「台積電 法人」、「AI 伺服器」、「TSMC earnings」"
            ),
        ).strip()

        limit: int = st.slider(
            "顯示則數",
            min_value=5,
            max_value=15,
            value=10,
            step=1,
            key="news_limit",
        )

        search_btn = st.button(
            "搜尋新聞", type="primary", use_container_width=True,
            key="news_search",
        )

    with result_col:
        if not search_btn:
            st.info(
                "請在左側輸入關鍵字，點擊「搜尋新聞」。\n\n"
                "**資料來源：** Google News RSS（繁體中文 TW 版）\n\n"
                "**搜尋範例**\n"
                "- 台積電 法人\n"
                "- AI 伺服器 概念股\n"
                "- 升息 聯準會\n"
                "- TSMC earnings"
            )
            return

        if not keyword:
            st.error("關鍵字不得為空。")
            return

        with st.spinner(f"正在搜尋「{keyword}」相關新聞…"):
            news_list = get_keyword_news(keyword=keyword, limit=limit)

        st.markdown(f"##### 「{keyword}」最新財經新聞（共 {len(news_list)} 則）")

        if not news_list:
            st.warning(
                "查無相關新聞。\n\n"
                "可能原因：\n"
                "- 關鍵字過於冷門或拼寫有誤\n"
                "- Google News RSS 暫時無法存取\n"
                "- 請嘗試不同的關鍵字組合"
            )
            return

        for i, item in enumerate(news_list, 1):
            title     = item["title"]
            link      = item["link"]
            published = item["published"]
            source    = item["source"]

            source_tag = f" ｜ {source}" if source else ""
            st.markdown(
                f"**{i}.** [{title}]({link})  \n"
                f"<small>🕐 {published}{source_tag}</small>",
                unsafe_allow_html=True,
            )
            if i < len(news_list):
                st.divider()
