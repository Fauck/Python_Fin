"""
股票分析 Web 應用程式
技術架構：Streamlit + fugle-marketdata + Plotly
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from fugle_marketdata import RestClient

# ─────────────────────────────────────────────
# 初始化：載入環境變數
# ─────────────────────────────────────────────
load_dotenv()


# ─────────────────────────────────────────────
# 資料層：API 呼叫邏輯（與 UI 完全解耦）
# ─────────────────────────────────────────────

def get_fugle_client() -> RestClient:
    """建立並回傳 Fugle RestClient 實例。"""
    api_key = os.getenv("FUGLE_API_KEY")
    if not api_key:
        raise ValueError("找不到 FUGLE_API_KEY，請確認 .env 檔案設定。")
    return RestClient(api_key=api_key)


def fetch_stock_candles(
    symbol: str,
    limit: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fields: str = "open,high,low,close,volume",
) -> pd.DataFrame:
    """
    透過 Fugle Historical API 取得股票 K 線資料。

    Parameters
    ----------
    symbol    : 股票代號（例如 "2330"）
    limit     : 最多回傳幾筆交易日資料（預設 10）
    date_from : 起始日期字串 "YYYY-MM-DD"；None 表示自動往前推算
    date_to   : 結束日期字串 "YYYY-MM-DD"；None 表示今日
    fields    : API 回傳欄位（逗號分隔）

    Returns
    -------
    pd.DataFrame  已排序（日期升冪）的最近 limit 筆資料
    """
    client = get_fugle_client()

    # 若未指定日期，自動向前推 60 個日曆天以確保涵蓋足夠的交易日
    if date_to is None:
        date_to = datetime.today().strftime("%Y-%m-%d")
    if date_from is None:
        date_from = (datetime.today() - timedelta(days=60)).strftime("%Y-%m-%d")

    raw = client.stock.historical.candles(
        **{
            "symbol": symbol,
            "from": date_from,
            "to": date_to,
            "fields": fields,
        }
    )

    # raw 可能是 dict（含 "data" key）或直接是 list
    if isinstance(raw, dict):
        records = raw.get("data", [])
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # 統一日期欄位名稱
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    # 取最近 limit 筆
    return df.tail(limit).reset_index(drop=True)


# ─────────────────────────────────────────────
# 展示層：畫面渲染函式（純 UI，不含業務邏輯）
# ─────────────────────────────────────────────

def render_data_table(df: pd.DataFrame, symbol: str) -> None:
    """以 DataFrame 表格形式展示股價資料。"""
    st.subheader(f"📋 {symbol} 近期歷史資料")

    display_df = df.copy()
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    # 欄位中文對照
    col_map = {
        "date": "日期",
        "open": "開盤價",
        "high": "最高價",
        "low": "最低價",
        "close": "收盤價",
        "volume": "成交量",
    }
    display_df = display_df.rename(columns={k: v for k, v in col_map.items() if k in display_df.columns})
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_close_chart(df: pd.DataFrame, symbol: str) -> None:
    """繪製收盤價折線走勢圖（Plotly）。"""
    if "close" not in df.columns or "date" not in df.columns:
        st.warning("資料缺少必要欄位，無法繪製走勢圖。")
        return

    st.subheader(f"📈 {symbol} 收盤價走勢")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines+markers",
            name="收盤價",
            line=dict(color="#2196F3", width=2),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        xaxis_title="日期",
        yaxis_title="收盤價（TWD）",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_candlestick_chart(df: pd.DataFrame, symbol: str) -> None:
    """繪製 K 線圖（需含 open/high/low/close 欄位）。"""
    required = {"open", "high", "low", "close", "date"}
    if not required.issubset(df.columns):
        return  # 欄位不足時靜默跳過

    st.subheader(f"🕯️ {symbol} K 線圖")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#EF5350",
                decreasing_line_color="#26A69A",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="日期",
        yaxis_title="價格（TWD）",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# 進入點：Streamlit 主程式
# ─────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="台股分析儀表板",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 台股分析儀表板")
    st.caption("資料來源：Fugle Market Data API")

    # ── Sidebar：查詢參數（保留未來擴充彈性）──────────────
    with st.sidebar:
        st.header("查詢條件")

        symbol = st.text_input(
            label="股票代號",
            value="2330",
            max_chars=10,
            help="輸入台灣股票代號，例如 2330（台積電）",
        ).strip()

        limit = st.number_input(
            label="顯示天數",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            help="最近幾個交易日的資料",
        )

        # 預留：日期區間（未來可取消註解啟用）
        # st.markdown("---")
        # st.subheader("自訂日期區間（選填）")
        # custom_from = st.date_input("起始日期", value=None)
        # custom_to   = st.date_input("結束日期",  value=None)

        # 預留：技術指標（未來可取消註解啟用）
        # st.markdown("---")
        # st.subheader("技術指標")
        # show_ma5  = st.checkbox("MA5",  value=False)
        # show_ma20 = st.checkbox("MA20", value=False)

        query_btn = st.button("查詢", type="primary", use_container_width=True)

    # ── 主畫面 ────────────────────────────────────────────
    if not query_btn:
        st.info("請在左側輸入股票代號後，點擊「查詢」按鈕。")
        return

    if not symbol:
        st.error("股票代號不得為空，請重新輸入。")
        return

    with st.spinner(f"正在取得 {symbol} 的歷史資料…"):
        try:
            df = fetch_stock_candles(symbol=symbol, limit=int(limit))
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"API 呼叫失敗：{e}\n\n請確認股票代號是否正確，或稍後再試。")
            return

    if df.empty:
        st.warning(f"查無 **{symbol}** 的資料，請確認代號是否正確。")
        return

    # 摘要指標
    latest = df.iloc[-1]
    prev   = df.iloc[-2] if len(df) >= 2 else latest
    price_delta = float(latest["close"]) - float(prev["close"]) if "close" in df.columns else 0

    col1, col2, col3, col4 = st.columns(4)
    if "close" in df.columns:
        col1.metric("收盤價", f"{latest['close']:,.2f}", f"{price_delta:+.2f}")
    if "open" in df.columns:
        col2.metric("開盤價", f"{latest['open']:,.2f}")
    if "high" in df.columns:
        col3.metric("最高價", f"{latest['high']:,.2f}")
    if "low" in df.columns:
        col4.metric("最低價", f"{latest['low']:,.2f}")

    st.markdown("---")

    # 圖表與表格
    render_candlestick_chart(df, symbol)
    render_close_chart(df, symbol)
    render_data_table(df, symbol)


if __name__ == "__main__":
    main()
