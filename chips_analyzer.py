"""
籌碼與基本面分析頁面（Tab 4）。
功能：三大法人買賣超趨勢 + 現金殖利率快覽
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import fetch_stock_candles, get_fugle_client, resolve_stock_input, push_shared_symbol, pull_shared_symbol


# ═════════════════════════════════════════════
# 資料層：三大法人 & 股利 API
# ═════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_institutional_trading(
    symbol: str,
    days: int = 60,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    取得三大法人每日買賣超（千張）。

    Parameters
    ----------
    symbol     : 股票代號
    days       : 回查日曆天數（start_date 未指定時使用），預設 60 天
    start_date : 明確的起始日期字串 "YYYY-MM-DD"；指定時優先於 days 計算

    Returns
    -------
    pd.DataFrame  欄位：date / foreign_net / trust_net / dealer_net
    查無資料或 API 端點不支援時回傳空 DataFrame。
    """
    try:
        client    = get_fugle_client()
        date_to   = datetime.today().strftime("%Y-%m-%d")
        date_from = (
            start_date
            if start_date is not None
            else (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        )

        raw = client.stock.historical.institutional(  # type: ignore[attr-defined]
            **{"symbol": symbol, "from": date_from, "to": date_to}
        )

        records: List[dict] = []
        if isinstance(raw, dict):
            records = list(raw.get("data", []))
        elif isinstance(raw, list):
            records = list(raw)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # ── 動態欄位對應（Fugle 可能使用駝峰或縮寫）──
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            c = col.lower()
            if "date" in c and "date" not in rename_map.values():
                rename_map[col] = "date"
            elif ("foreign" in c or "fini" in c) and "foreign_net" not in rename_map.values():
                rename_map[col] = "foreign_net"
            elif ("trust" in c or "siteq" in c) and "trust_net" not in rename_map.values():
                rename_map[col] = "trust_net"
            elif "dealer" in c and "dealer_net" not in rename_map.values():
                rename_map[col] = "dealer_net"

        if rename_map:
            df = df.rename(columns=rename_map)

        required = {"date", "foreign_net", "trust_net", "dealer_net"}
        if not required.issubset(df.columns):
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        for col in ["foreign_net", "trust_net", "dealer_net"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 1000  # 張 → 千張

        return (
            df[["date", "foreign_net", "trust_net", "dealer_net"]]
            .drop_duplicates(subset="date")
            .sort_values("date")
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def fetch_dividends(symbol: str) -> Optional[Dict[str, Any]]:
    """
    取得近三年現金股利，回傳平均每股現金股利（元）。

    Returns
    -------
    dict { avg_cash_3yr: float }  或  None（無資料或 API 不支援）
    """
    try:
        client = get_fugle_client()
        raw    = client.stock.historical.dividends(  # type: ignore[attr-defined]
            **{"symbol": symbol}
        )

        records: List[dict] = []
        if isinstance(raw, dict):
            records = list(raw.get("data", []))
        elif isinstance(raw, list):
            records = list(raw)

        if not records:
            return None

        df = pd.DataFrame(records)

        # 尋找現金股利欄位
        cash_col: Optional[str] = next(
            (c for c in df.columns
             if "cash" in c.lower() and ("div" in c.lower() or "dividend" in c.lower())),
            next((c for c in df.columns if "cash" in c.lower()), None),
        )
        year_col: Optional[str] = next(
            (c for c in df.columns if "year" in c.lower()),
            next((c for c in df.columns if "date" in c.lower()), None),
        )

        if cash_col is None:
            return None

        df[cash_col] = pd.to_numeric(df[cash_col], errors="coerce")
        df = df.dropna(subset=[cash_col])

        # 先按「年度」加總現金股利（修正季配 / 半年配重複計算問題）
        if year_col:
            df["_year"] = pd.DatetimeIndex(
                pd.to_datetime(df[year_col], errors="coerce")
            ).year
        else:
            # 無日期欄位時以列序替代（資料通常已按年排列）
            df["_year"] = range(len(df))

        annual = df.groupby("_year")[cash_col].sum()
        annual = annual.sort_index(ascending=False).head(3)  # 取最近 3 年

        if annual.empty:
            return None

        return {"avg_cash_3yr": round(float(annual.mean()), 2)}

    except Exception:
        return None


# ═════════════════════════════════════════════
# 演算法層：籌碼訊號分析
# ═════════════════════════════════════════════

def analyze_highlights(df_insti: pd.DataFrame) -> List[str]:
    """
    分析三大法人籌碼亮點，回傳訊號清單。

    Parameters
    ----------
    df_insti : fetch_institutional_trading 回傳的 DataFrame（日期升冪）

    Returns
    -------
    List[str]  亮點訊息（空清單表示無特殊訊號）
    """
    if df_insti.empty or len(df_insti) < 2:
        return []

    highlights: List[str] = []

    # ── 投信連買 ≥ 3 天 ────────────────────────
    if "trust_net" in df_insti.columns:
        consec = 0
        for val in df_insti["trust_net"].iloc[::-1]:
            if pd.isna(val):
                break
            if float(val) > 0:
                consec += 1
            else:
                break
        if consec >= 3:
            highlights.append(f"📈 投信連續買超 **{consec}** 個交易日")

    # ── 外資由賣轉買（第一天）────────────────────
    if "foreign_net" in df_insti.columns and len(df_insti) >= 2:
        v_today = df_insti["foreign_net"].iloc[-1]
        v_prev  = df_insti["foreign_net"].iloc[-2]
        if not pd.isna(v_today) and not pd.isna(v_prev):
            today_val = float(v_today)
            prev_val  = float(v_prev)
            if today_val > 0 and prev_val <= 0:
                highlights.append(
                    f"🔄 外資由賣轉買（今日淨買超 {today_val:.0f} 千張）"
                )

    # ── 近 5 日三大法人合計 ──────────────────────
    net_cols = [c for c in ["foreign_net", "trust_net", "dealer_net"]
                if c in df_insti.columns]
    if net_cols and len(df_insti) >= 5:
        total_net = float(df_insti.tail(5)[net_cols].sum().sum())
        if total_net > 5:
            highlights.append(
                f"✅ 近 5 日三大法人合計淨買超 {total_net:.0f} 千張"
            )
        elif total_net < -5:
            highlights.append(
                f"⚠️ 近 5 日三大法人合計淨賣超 {abs(total_net):.0f} 千張"
            )

    return highlights


# ═════════════════════════════════════════════
# UI 層：圖表與頁面渲染
# ═════════════════════════════════════════════

def _render_insti_chart(
    df_candle: pd.DataFrame,
    df_insti: pd.DataFrame,
    symbol: str,
) -> None:
    """繪製 K 線 + 三大法人買賣超疊加雙列子圖。"""
    # 將日期統一轉為字串（category 軸），消除週末假日造成的空白斷層
    candle_x = pd.DatetimeIndex(df_candle["date"]).strftime("%Y-%m-%d")
    insti_x  = pd.DatetimeIndex(df_insti["date"]).strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )

    # ── Row 1：K 線 ──────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=candle_x,
            open=df_candle["open"],
            high=df_candle["high"],
            low=df_candle["low"],
            close=df_candle["close"],
            name=symbol,
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # ── Row 2：三大法人買賣超（群組長條圖）──────────
    insti_meta = [
        ("foreign_net", "#1976D2", "外資"),
        ("trust_net",   "#E65100", "投信"),
        ("dealer_net",  "#6A1B9A", "自營商"),
    ]
    for col_name, color, label in insti_meta:
        if col_name not in df_insti.columns:
            continue
        vals       = df_insti[col_name].tolist()
        bar_colors = [
            color if (not pd.isna(v) and float(v) >= 0) else f"{color}60"
            for v in vals
        ]
        fig.add_trace(
            go.Bar(
                x=insti_x,
                y=df_insti[col_name],
                name=label,
                marker_color=bar_colors,
                opacity=0.88,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        height=560,
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    # category 軸：連續排列交易日，不留假日空白
    fig.update_xaxes(type="category", showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="收盤價", row=1, col=1,
                     gridcolor="#f0f0f0", showgrid=True)
    fig.update_yaxes(title_text="千張", row=2, col=1,
                     gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#bbb")

    st.plotly_chart(fig, use_container_width=True)


def render_chips_page() -> None:
    """籌碼與基本面分析頁面（Tab 4）。"""
    pull_shared_symbol("chips_page_symbol")
    with st.expander("🔍 查詢條件設定與操作", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            symbol = st.text_input(
                "股票代號/名稱", value="2330", max_chars=20,
                key="chips_page_symbol",
                help="支援台灣股票。可輸入數字代號 (如 2330) 或中文股名 (如 台積電)。",
            ).strip()
        with col_b:
            days: int = st.selectbox(  # type: ignore[assignment]
                "回查天數",
                options=[30, 60, 90],
                index=1,
                key="chips_page_days",
                help="查詢三大法人資料的回查日曆天數",
            )

        query_btn = st.button(
            "查詢籌碼", type="primary", use_container_width=True,
            key="chips_page_query",
        )

    if not query_btn:
        st.info(
            "請在上方輸入股票代號並選擇回查天數，點擊「查詢籌碼」。\n\n"
            "**功能說明**\n"
            "- 📊 K 線 ＋ 三大法人買賣超（外資 / 投信 / 自營商）疊加圖\n"
            "- 🔔 籌碼亮點自動偵測（投信連買、外資轉向）\n"
            "- 💰 近三年平均現金殖利率快覽"
        )
        return

    if not symbol:
        st.error("請輸入股票代號或名稱。")
        return

    resolved_code, display_name = resolve_stock_input(symbol)
    if not resolved_code:
        st.error(f"找不到符合「{symbol}」的標的，請重新輸入。")
        return
    push_shared_symbol(resolved_code)
    symbol = display_name

    with st.spinner(f"正在取得 {symbol} 資料…"):
        df_candle = fetch_stock_candles(
            symbol=resolved_code,
            limit=days,
            fields="open,high,low,close,volume",
        )
        # 以 K 線第一筆日期對齊籌碼資料起始點，消除時間軸錯位
        candle_start: Optional[str] = None
        if not df_candle.empty and "date" in df_candle.columns:
            candle_start = pd.Timestamp(str(df_candle.iloc[0]["date"])).strftime("%Y-%m-%d")
        df_insti = fetch_institutional_trading(
            symbol=resolved_code, days=days, start_date=candle_start
        )
        div_data = fetch_dividends(symbol=resolved_code)

    if df_candle.empty:
        st.warning(f"查無 **{symbol}** 的 K 線資料，請確認代號是否正確。")
        return

    # ── 籌碼亮點 Banner ─────────────────────────
    highlights = analyze_highlights(df_insti)
    if highlights:
        st.success("\n\n".join(highlights))
    elif not df_insti.empty:
        st.info("目前無特殊籌碼訊號。")

    # ── K 線 + 法人圖 ────────────────────────────
    st.markdown(f"##### {symbol} K 線 ＋ 三大法人買賣超（近 {days} 天）")
    if not df_insti.empty:
        _render_insti_chart(df_candle, df_insti, symbol)
    else:
        st.warning(
            "三大法人資料目前無法取得（API 端點可能不支援或此標的查無資料）。"
            "以下僅顯示 K 線。"
        )
        fig = go.Figure(go.Candlestick(
            x=df_candle["date"],
            open=df_candle["open"],
            high=df_candle["high"],
            low=df_candle["low"],
            close=df_candle["close"],
            name=symbol,
            increasing_line_color="#EF5350",
            decreasing_line_color="#26A69A",
        ))
        fig.update_layout(
            height=320,
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── 法人明細表 ──────────────────────────────
    if not df_insti.empty:
        st.markdown("---")
        st.markdown("##### 三大法人買賣超明細（千張）")
        disp = df_insti.copy()
        disp["date"] = pd.to_datetime(disp["date"]).dt.strftime("%Y-%m-%d")
        disp = disp.sort_values("date", ascending=False).reset_index(drop=True)
        disp.columns = ["日期", "外資", "投信", "自營商"]
        fmt: Dict[str, Any] = {"外資": "{:.2f}", "投信": "{:.2f}", "自營商": "{:.2f}"}
        st.dataframe(
            disp.style.format(fmt),
            use_container_width=True,
            hide_index=True,
        )

    # ── 股利快覽 ─────────────────────────────────
    st.markdown("---")
    st.markdown("##### 基本面：現金殖利率快覽")
    if div_data is not None and not df_candle.empty:
        avg_cash      = div_data["avg_cash_3yr"]
        current_close = float(df_candle["close"].iloc[-1])
        yield_pct     = avg_cash / current_close * 100 if current_close > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("近 3 年平均現金股利", f"{avg_cash:.2f} 元")
        c2.metric("現價", f"{current_close:.2f} 元")
        c3.metric(
            "估算現金殖利率",
            f"{yield_pct:.2f}%",
            delta="達標 ✅" if yield_pct >= 5.0 else None,
        )

        if yield_pct >= 5.0:
            st.success(
                f"**{symbol}** 估算現金殖利率 **{yield_pct:.2f}%**，"
                "已達長線配息門檻（5%），可列入長線佈局觀察。"
            )
        else:
            st.info(
                f"**{symbol}** 估算現金殖利率 **{yield_pct:.2f}%**"
                "（長線門檻：5%）。"
            )
    else:
        st.info(
            "股利資料目前無法取得（API 端點可能不支援或此標的查無資料）。"
        )
