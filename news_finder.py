"""
關鍵字財經新聞頁面（Tab 7）。
使用 feedparser + requests 解析 Google News RSS，取得繁體中文財經新聞。
新增：公開資訊觀測站 (MOPS) 重大訊息、PTT 股版社群情緒。

設計原則
--------
- 100% 使用者主動觸發（無任何排程、定時刷新、或背景通知邏輯）
- Google News：解析公開 RSS 端點，不爬取 HTML DOM
- MOPS：POST 請求，BeautifulSoup + lxml 解析 HTML 表格
- PTT：GET 請求，BeautifulSoup + lxml 解析文章列表
- 所有 requests 呼叫均含隨機 User-Agent 與 timeout=5 秒
- @st.cache_data 快取避免重複呼叫；使用者不操作時不消耗任何網路資源
"""

import html
import random
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from urllib.parse import quote

import feedparser
import requests
import streamlit as st
from bs4 import BeautifulSoup
from utils import pull_shared_symbol


# ── Google News RSS 端點（繁體中文台灣版）
_GNEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q={kw}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
)

# ── MOPS 重大訊息 AJAX 端點
_MOPS_URL = "https://mops.twse.com.tw/mops/web/ajax_t05sr011_1"

# ── PTT 股版搜尋端點
_PTT_SEARCH_URL = "https://www.ptt.cc/bbs/Stock/search?q={kw}"

# ── 隨機 User-Agent 池（模擬真實瀏覽器，降低被封鎖機率）
_UA_POOL: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# ── 固定 Headers（get_keyword_news 維持向後相容）
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
}


def _rand_headers(referer: str = "") -> Dict[str, str]:
    """每次呼叫回傳隨機 User-Agent Headers，降低被封鎖機率。"""
    h: Dict[str, str] = {
        "User-Agent":      random.choice(_UA_POOL),
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    if referer:
        h["Referer"] = referer
    return h


# ═════════════════════════════════════════════
# 資料層：Google News RSS
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

    Parameters
    ----------
    keyword : 搜尋關鍵字（繁體中文或英文，空格亦可）
    limit   : 最多回傳幾則，預設 15

    Returns
    -------
    List of dicts { title, published, link, source }
    """
    url = _GNEWS_RSS.format(kw=quote(keyword.strip()))

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=10)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except requests.RequestException:
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
# 資料層：MOPS 公開資訊觀測站重大訊息
# ═════════════════════════════════════════════

def _parse_mops_html(html_text: str, limit: int) -> List[Dict[str, str]]:
    """
    解析 MOPS 重大訊息 HTML 回應，提取資料行。

    MOPS 表格欄位（ajax_t05sr011_1 回應）
    ──────────────────────────────────────
    td[0] 序號 | td[1] 發言日期 | td[2] 發言時間 | td[3] 主旨 | td[4] 附件

    搜尋策略：找到第一個內文含「發言」的 <table>（最可靠的 MOPS 表格識別方式）。
    """
    soup  = BeautifulSoup(html_text, "lxml")
    table = None
    for t in soup.find_all("table"):
        if "發言" in t.get_text():
            table = t
            break
    if table is None:
        return []

    results: List[Dict[str, str]] = []
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        date_str = cells[1].get_text(strip=True)
        time_str = cells[2].get_text(strip=True)
        subject  = cells[3].get_text(strip=True)

        if not date_str or not subject:
            continue

        link_tag = cells[3].find("a")
        href     = str(link_tag.get("href") or "") if link_tag else ""
        link     = (
            "https://mops.twse.com.tw" + href
            if href.startswith("/")
            else href
        )

        results.append({
            "date":    date_str,
            "time":    time_str,
            "subject": subject,
            "link":    link,
        })
        if len(results) >= limit:
            break

    return results


@st.cache_data(ttl=1800)   # 30 分鐘快取
def get_mops_material_info(
    symbol: str,
    limit: int = 10,
) -> Optional[List[Dict[str, str]]]:
    """
    從公開資訊觀測站抓取股票重大訊息（純手動觸發，無排程）。

    流程
    ----
    1. POST 請求至 MOPS AJAX 端點，帶入股票代號與當年民國年
    2. BeautifulSoup + lxml 解析回傳 HTML 表格
    3. 若當年查無資料，自動補查上一年度
    4. timeout=5 秒；連線失敗回傳 None（UI 層顯示 st.warning）

    Parameters
    ----------
    symbol : 台灣股票代號（4-6 位數字，例如 "2330"）
    limit  : 最多回傳筆數，預設 10

    Returns
    -------
    None       → 連線失敗（網路異常 / 逾時）
    []         → 查無資料
    List[Dict] → 含 {date, time, subject, link} 的最近 N 筆
    """
    symbol = symbol.strip()
    if not symbol:
        return []

    roc_year = date.today().year - 1911
    common_headers = {
        **_rand_headers("https://mops.twse.com.tw/mops/web/t05sr011_1"),
        "Origin":       "https://mops.twse.com.tw",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    def _post_year(year: int) -> Optional[str]:
        """對 MOPS POST 單一年度，回傳 HTML 字串；連線失敗回傳 None。"""
        form = {
            "encodeURIComponent": "1",
            "step":    "1",
            "firstin": "1",
            "off":     "1",
            "keyword4": "",
            "code1":    "",
            "TYPEK":   "all",
            "co_id":   symbol,
            "year":    str(year),
        }
        try:
            r = requests.post(
                _MOPS_URL, data=form, headers=common_headers, timeout=5
            )
            r.raise_for_status()
            r.encoding = "utf-8"
            return r.text
        except requests.RequestException:
            return None

    # 先查今年
    html_text = _post_year(roc_year)
    if html_text is None:
        return None   # 連線失敗

    results = _parse_mops_html(html_text, limit)

    # 今年若無資料，補查去年（公司可能今年尚未發佈重大訊息）
    if not results:
        html_prev = _post_year(roc_year - 1)
        if html_prev:
            results = _parse_mops_html(html_prev, limit)

    return results


# ═════════════════════════════════════════════
# 資料層：PTT 股版社群情緒
# ═════════════════════════════════════════════

@st.cache_data(ttl=300)   # 5 分鐘快取（社群內容更新較頻繁）
def get_ptt_stock_sentiment(
    keyword: str,
    limit: int = 10,
) -> Optional[List[Dict[str, str]]]:
    """
    爬取 PTT 股版搜尋結果，分析社群討論熱度（純手動觸發，無排程）。

    流程
    ----
    1. GET 請求至 PTT 網頁版股版搜尋頁（帶 over18 cookie 跳過年齡確認）
    2. BeautifulSoup + lxml 解析 div.r-ent 文章列表
    3. 推文數解析：「爆」→ 100；「Xn」→ 負評；數字 → 原始值；空 → 0
    4. timeout=5 秒；連線失敗回傳 None（UI 層顯示 st.warning）

    Parameters
    ----------
    keyword : 搜尋關鍵字（股票代號或名稱，例如 "2330" 或 "台積電"）
    limit   : 最多回傳篇數，預設 10

    Returns
    -------
    None       → 連線失敗（網路異常 / 逾時）
    []         → 查無符合關鍵字的貼文
    List[Dict] → 含 {date, pushes, title, url} 的最近 N 篇
    """
    keyword = keyword.strip()
    if not keyword:
        return []

    url = _PTT_SEARCH_URL.format(kw=quote(keyword))
    try:
        resp = requests.get(
            url,
            headers=_rand_headers("https://www.ptt.cc/"),
            cookies={"over18": "1"},   # PTT 年齡確認 cookie
            timeout=5,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"
    except requests.RequestException:
        return None

    soup     = BeautifulSoup(resp.text, "lxml")
    articles = soup.select("div.r-ent")
    if not articles:
        return []

    results: List[Dict[str, str]] = []
    for art in articles[:limit]:
        # ── 推文數
        nrec_el   = art.select_one("div.nrec span")
        nrec_text = str(nrec_el.get_text(strip=True)) if nrec_el else ""
        if nrec_text == "爆":
            pushes = 100
        elif nrec_text.startswith("X"):
            tail = nrec_text[1:]
            pushes = -int(tail) if tail.isdigit() else -1
        else:
            try:
                pushes = int(nrec_text)
            except ValueError:
                pushes = 0

        # ── 標題與連結（已刪文的 r-ent 沒有 <a>，略過）
        title_el = art.select_one("div.title a")
        if title_el is None:
            continue
        title = title_el.get_text(strip=True)
        post_url = "https://www.ptt.cc" + str(title_el.get("href") or "#")

        # ── 日期
        date_el  = art.select_one("div.meta div.date")
        date_str = date_el.get_text(strip=True) if date_el else ""

        results.append({
            "date":   date_str,
            "pushes": str(pushes),
            "title":  title,
            "url":    post_url,
        })

    return results


# ═════════════════════════════════════════════
# UI 層：各資料源渲染
# ═════════════════════════════════════════════

def _render_mops_section(
    mops_data: List[Dict[str, str]],
    symbol: str,
) -> None:
    """渲染公開資訊觀測站重大訊息區塊。"""
    st.markdown("---")
    st.subheader(f"🏢 {symbol} 官方重大訊息（公開資訊觀測站）")
    st.caption(
        "資料來源：公開資訊觀測站（mops.twse.com.tw）｜"
        "自動查詢當年度及上一年度，點擊主旨連結可查看完整公告"
    )

    if not mops_data:
        st.info(
            f"查無 **{symbol}** 的重大訊息。\n\n"
            "可能原因：代號輸入有誤，或近兩年內無重大公告。"
        )
        return

    for i, item in enumerate(mops_data, 1):
        link    = item.get("link", "")
        subject = item["subject"]
        title_md = f"[{subject}]({link})" if link else subject

        st.markdown(
            f"**{i}.** {title_md}  \n"
            f"<small>📅 {item['date']} &nbsp;{item['time']}</small>",
            unsafe_allow_html=True,
        )
        if i < len(mops_data):
            st.divider()


def _render_ptt_section(
    ptt_data: List[Dict[str, str]],
    keyword: str,
) -> None:
    """渲染 PTT 股版社群情緒區塊。"""
    st.markdown("---")
    st.subheader(f"💬 「{keyword}」鄉民討論熱度（PTT 股版）")
    st.caption(
        "資料來源：PTT 網頁版 Stock 版搜尋（ptt.cc/bbs/Stock）｜"
        "推文數 ≥ 50 顯示 🔥 熱門討論；爆文（100+）顯示 🔥 爆文"
    )

    if not ptt_data:
        st.info(
            "查無相關 PTT 貼文。\n\n"
            "可能原因：此關鍵字在 PTT 股版討論度較低，或搜尋服務暫時無回應。"
        )
        return

    for i, item in enumerate(ptt_data, 1):
        pushes   = int(item["pushes"])
        title    = item["title"]
        post_url = item["url"]
        date_str = item["date"]

        # 熱門標籤
        if pushes >= 100:
            hot_tag = (
                " &nbsp;<span style='color:#EF5350;font-weight:700;'>"
                "🔥 爆文</span>"
            )
            push_color = "#EF5350"
            push_label = "爆"
        elif pushes >= 50:
            hot_tag = (
                " &nbsp;<span style='color:#FF9800;font-weight:700;'>"
                "🔥 熱門討論</span>"
            )
            push_color = "#FF9800"
            push_label = str(pushes)
        elif pushes < 0:
            hot_tag    = ""
            push_color = "#9E9E9E"
            push_label = f"噓{abs(pushes)}"
        else:
            hot_tag    = ""
            push_color = "#26A69A"
            push_label = str(pushes) if pushes > 0 else "—"

        st.markdown(
            f"**{i}.** [{title}]({post_url}){hot_tag}  \n"
            f"<small>📅 {date_str} &nbsp;｜&nbsp; "
            f"💬 推文 <span style='color:{push_color};font-weight:600;'>"
            f"{push_label}</span></small>",
            unsafe_allow_html=True,
        )
        if i < len(ptt_data):
            st.divider()


# ═════════════════════════════════════════════
# UI 層：新聞頁面渲染（主進入點）
# ═════════════════════════════════════════════

def render_news_page() -> None:
    """關鍵字財經新聞頁面（Tab 7）。"""
    pull_shared_symbol("news_mops_symbol")
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 搜尋條件")
        keyword = st.text_input(
            "搜尋關鍵字",
            value="台積電",
            key="news_keyword",
            help=(
                "支援中英文，多個關鍵字以空格分隔。\n"
                "同時用於 Google News 與 PTT 股版搜尋。\n"
                "例如：「台積電 法人」、「AI 伺服器」、「TSMC earnings」"
            ),
        ).strip()

        limit: int = st.slider(
            "Google News 顯示則數",
            min_value=5,
            max_value=15,
            value=10,
            step=1,
            key="news_limit",
        )

        st.markdown("---")
        st.markdown("##### 延伸資訊查詢")

        show_mops = st.checkbox(
            "🏢 MOPS 官方重大訊息",
            value=True,
            key="news_show_mops",
            help="查詢公開資訊觀測站重大訊息，需輸入 4-6 位股票代號",
        )
        mops_symbol = st.text_input(
            "股票代號（MOPS 專用）",
            value="2330",
            max_chars=10,
            key="news_mops_symbol",
            disabled=not show_mops,
            help="僅供 MOPS 重大訊息查詢使用，例如 2330、0050",
        ).strip()

        show_ptt = st.checkbox(
            "💬 PTT 股版情緒",
            value=True,
            key="news_show_ptt",
            help="爬取 PTT 股版搜尋結果（使用上方搜尋關鍵字）",
        )

        search_btn = st.button(
            "搜尋", type="primary", use_container_width=True,
            key="news_search",
        )

    with result_col:
        if not search_btn:
            st.info(
                "請在左側輸入關鍵字，點擊「搜尋」。\n\n"
                "**資料來源**\n"
                "- 📰 Google News RSS（繁體中文 TW 版）\n"
                "- 🏢 公開資訊觀測站 MOPS 重大訊息\n"
                "- 💬 PTT 股版社群討論熱度\n\n"
                "**搜尋範例**\n"
                "- 關鍵字：台積電 / AI 伺服器 / TSMC earnings\n"
                "- MOPS 代號：2330 / 2317 / 0050"
            )
            return

        if not keyword:
            st.error("關鍵字不得為空。")
            return

        # ── Google News ───────────────────────────────
        with st.spinner(f"正在搜尋「{keyword}」Google 新聞…"):
            news_list = get_keyword_news(keyword=keyword, limit=limit)

        st.markdown(f"##### 📰 「{keyword}」最新財經新聞（共 {len(news_list)} 則）")

        if not news_list:
            st.warning(
                "查無 Google News 相關新聞。\n\n"
                "可能原因：關鍵字過於冷門、Google RSS 暫時無法存取，"
                "或請嘗試不同的關鍵字組合。"
            )
        else:
            for i, item in enumerate(news_list, 1):
                source_tag = f" ｜ {item['source']}" if item["source"] else ""
                st.markdown(
                    f"**{i}.** [{item['title']}]({item['link']})  \n"
                    f"<small>🕐 {item['published']}{source_tag}</small>",
                    unsafe_allow_html=True,
                )
                if i < len(news_list):
                    st.divider()

        # ── MOPS 重大訊息 ─────────────────────────────
        if show_mops:
            if not mops_symbol:
                st.warning("MOPS 查詢需要股票代號，請在左側輸入後重新搜尋。")
            else:
                with st.spinner(f"正在查詢 {mops_symbol} 重大訊息（MOPS）…"):
                    mops_data = get_mops_material_info(
                        symbol=mops_symbol, limit=10
                    )
                if mops_data is None:
                    st.markdown("---")
                    st.subheader("🏢 官方重大訊息（公開資訊觀測站）")
                    st.warning(
                        "目前無法連線至公開資訊觀測站（MOPS），請稍後再試。\n\n"
                        "網站：https://mops.twse.com.tw"
                    )
                else:
                    _render_mops_section(mops_data, mops_symbol)

        # ── PTT 股版情緒 ──────────────────────────────
        if show_ptt:
            with st.spinner(f"正在搜尋 PTT 股版「{keyword}」討論…"):
                ptt_data = get_ptt_stock_sentiment(
                    keyword=keyword, limit=10
                )
            if ptt_data is None:
                st.markdown("---")
                st.subheader("💬 PTT 股版社群情緒")
                st.warning(
                    "目前無法連線至 PTT，請稍後再試。\n\n"
                    "網站：https://www.ptt.cc/bbs/Stock"
                )
            else:
                _render_ptt_section(ptt_data, keyword)
