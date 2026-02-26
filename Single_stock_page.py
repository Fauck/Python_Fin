"""
單股分析頁面（Tab 1）。
渲染函式：render_data_table / render_close_chart / render_candlestick_chart
          render_ohlcv_chart / render_single_stock_page
"""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import fetch_stock_candles, compute_ma, compute_kd


# ═════════════════════════════════════════════
# 展示層：共用圖表 / 表格渲染函式
# ═════════════════════════════════════════════

def render_data_table(df: pd.DataFrame, symbol: str) -> None:
    """以 DataFrame 表格形式展示股價資料。"""
    st.subheader(f"📋 {symbol} 近期歷史資料")
    display_df = df.copy()
    if "date" in display_df.columns:
        display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    col_map = {
        "date": "日期", "open": "開盤價", "high": "最高價",
        "low": "最低價", "close": "收盤價", "volume": "成交量",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in col_map.items() if k in display_df.columns}
    )
    st.dataframe(display_df, width="stretch", hide_index=True)


def render_close_chart(df: pd.DataFrame, symbol: str) -> None:
    """繪製收盤價折線走勢圖（Plotly）。"""
    if "close" not in df.columns or "date" not in df.columns:
        st.warning("資料缺少必要欄位，無法繪製走勢圖。")
        return

    st.subheader(f"📈 {symbol} 收盤價走勢")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines+markers", name="收盤價",
        line=dict(color="#2196F3", width=2), marker=dict(size=6),
    ))
    fig.update_layout(
        xaxis_title="日期", yaxis_title="收盤價（TWD）",
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(type="category", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=30, b=0),
        autosize=True,
    )
    st.plotly_chart(fig)


def render_candlestick_chart(df: pd.DataFrame, symbol: str) -> None:
    """繪製 K 線圖（需含 open/high/low/close 欄位）。"""
    required = {"open", "high", "low", "close", "date"}
    if not required.issubset(df.columns):
        return

    st.subheader(f"🕯️ {symbol} K 線圖")
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#EF5350",
        decreasing_line_color="#26A69A",
    )])
    fig.update_layout(
        xaxis_title="日期", yaxis_title="價格（TWD）",
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(type="category", showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        margin=dict(l=0, r=0, t=30, b=0),
        autosize=True,
    )
    st.plotly_chart(fig)


def render_ohlcv_chart(
    df: pd.DataFrame,
    symbol: str,
    show_ma: Optional[List[int]] = None,
    show_kd: bool = False,
) -> None:
    """
    繪製 K 線 + 均線 + 成交量 + 成交值 + KD 子圖（Plotly subplots）。

    子圖結構（依資料與參數動態決定）：
      Row 1：K 線圖 + MA 均線覆蓋（Candlestick + Scatter）
      Row 2：成交量柱狀圖（依漲跌著色，若有資料）
      Row 3：成交值柱狀圖（若有資料）
      Row N：KD 值折線圖（若啟用）

    Parameters
    ----------
    df      : 含 OHLCV 欄位的 DataFrame；若已含 ma5/ma10/ma20/k_val/d_val 則直接使用
    symbol  : 股票代號
    show_ma : 要顯示的均線天數清單，例如 [5, 10, 20]；None 表示不顯示
    show_kd : 是否顯示 KD 子圖
    """
    required = {"open", "high", "low", "close", "date"}
    if not required.issubset(df.columns):
        return

    has_volume   = "volume"   in df.columns and df["volume"].notna().any()
    has_turnover = "turnover" in df.columns and df["turnover"].notna().any()
    ma_periods   = show_ma or []

    # 將日期轉為字串，確保 category 軸的 x 值與標註 x 值完全一致
    x_labels = df["date"].dt.strftime("%Y-%m-%d")

    # ── 動態建立子圖列表 ─────────────────────────
    # 每個 dict：title、base_height（歸一化前）
    rows_cfg = [{"title": f"{symbol} K 線", "h": 0.50}]
    if has_volume:
        rows_cfg.append({"title": "成交量（張）",  "h": 0.20})
    if has_turnover:
        rows_cfg.append({"title": "成交值（千元）", "h": 0.15})
    if show_kd:
        rows_cfg.append({"title": "KD 值",         "h": 0.20})

    total_h    = sum(r["h"] for r in rows_cfg)
    row_heights = [r["h"] / total_h for r in rows_cfg]
    n_rows      = len(rows_cfg)

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.025,
        subplot_titles=[r["title"] for r in rows_cfg],
    )

    # ── Row 1：K 線 ──────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x_labels,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#EF5350",
        decreasing_line_color="#26A69A",
        name="K線", showlegend=False,
    ), row=1, col=1)

    # ── Row 1：期間最高 / 最低標註 ──────────────
    if not df.empty:
        idx_high   = int(df["high"].idxmax())
        idx_low    = int(df["low"].idxmin())
        high_date  = x_labels.iloc[idx_high]
        high_price = float(df.loc[idx_high, "high"])
        low_date   = x_labels.iloc[idx_low]
        low_price  = float(df.loc[idx_low,  "low"])

        # 最高價：箭頭朝上，文字在 K 棒上方
        fig.add_annotation(
            x=high_date, y=high_price,
            text=f"最高<br><b>{high_price:,.2f}</b>",
            showarrow=True, arrowhead=2,
            arrowcolor="#EF5350", arrowwidth=1.5,
            ax=0, ay=-44,
            font=dict(color="#EF5350", size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#EF5350", borderwidth=1, borderpad=3,
            row=1, col=1,
        )
        # 最低價：箭頭朝下，文字在 K 棒下方
        fig.add_annotation(
            x=low_date,  y=low_price,
            text=f"最低<br><b>{low_price:,.2f}</b>",
            showarrow=True, arrowhead=2,
            arrowcolor="#26A69A", arrowwidth=1.5,
            ax=0, ay=44,
            font=dict(color="#26A69A", size=10),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#26A69A", borderwidth=1, borderpad=3,
            row=1, col=1,
        )

    # ── Row 1 覆蓋：均線 ─────────────────────────
    ma_styles = {
        5:  {"color": "#FF6B35", "dash": "solid"},   # 橙
        10: {"color": "#9C27B0", "dash": "solid"},   # 紫
        20: {"color": "#2196F3", "dash": "solid"},   # 藍
    }
    for p in ma_periods:
        col_name = f"ma{p}"
        if col_name not in df.columns:
            continue
        style = ma_styles.get(p, {"color": "#607D8B", "dash": "dot"})
        fig.add_trace(go.Scatter(
            x=x_labels, y=df[col_name],
            mode="lines", name=f"MA{p}",
            line=dict(color=style["color"], width=1.5, dash=style["dash"]),
        ), row=1, col=1)

    current_row = 2

    # ── Row 2：成交量 ────────────────────────────
    if has_volume:
        bar_colors = [
            "#EF5350" if float(c) >= float(o) else "#26A69A"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(go.Bar(
            x=x_labels, y=df["volume"],
            marker_color=bar_colors,
            name="成交量", showlegend=False,
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="張", row=current_row, col=1)
        current_row += 1

    # ── Row 3：成交值 ────────────────────────────
    if has_turnover:
        fig.add_trace(go.Bar(
            x=x_labels, y=df["turnover"],
            marker_color="#7E57C2",
            name="成交值", showlegend=False,
        ), row=current_row, col=1)
        fig.update_yaxes(title_text="千元", row=current_row, col=1)
        current_row += 1

    # ── Row N：KD 值 ─────────────────────────────
    if show_kd and "k_val" in df.columns and "d_val" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["k_val"],
            mode="lines", name="K",
            line=dict(color="#FF6B35", width=1.5),
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["d_val"],
            mode="lines", name="D",
            line=dict(color="#2196F3", width=1.5),
        ), row=current_row, col=1)
        # 超買 / 超賣參考線
        fig.add_hline(y=80, line=dict(color="#EF5350", dash="dash", width=1),
                      row=current_row, col=1)
        fig.add_hline(y=20, line=dict(color="#26A69A", dash="dash", width=1),
                      row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], title_text="KD", row=current_row, col=1)

    # ── 全域版面 ──────────────────────────────────
    chart_height = 380 + n_rows * 80
    fig.update_layout(
        height=chart_height,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        autosize=True,
    )
    for i in range(1, n_rows + 1):
        fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", row=i, col=1)
    fig.update_xaxes(type="category", showgrid=True, gridcolor="#f0f0f0")

    st.plotly_chart(fig)


# ═════════════════════════════════════════════
# 展示層：頁面渲染函式
# ═════════════════════════════════════════════

def render_single_stock_page() -> None:
    """單股分析頁面。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 查詢條件")
        symbol = st.text_input(
            "股票代號", value="2330", max_chars=10,
            key="single_stock_symbol",
            help="輸入台灣股票代號，例如 2330（台積電）",
        ).strip()
        limit = st.number_input(
            "顯示天數", min_value=1, max_value=60, value=10, step=1,
            help="最近幾個交易日的資料",
        )
        # 預留：日期區間（未來可取消註解啟用）
        # st.markdown("---")
        # st.markdown("##### 自訂日期區間（選填）")
        # custom_from = st.date_input("起始日期", value=None)
        # custom_to   = st.date_input("結束日期",  value=None)

        st.markdown("---")
        st.markdown("##### 技術指標")
        show_ma5  = st.checkbox("MA5",  value=True)
        show_ma10 = st.checkbox("MA10", value=True)
        show_ma20 = st.checkbox("MA20", value=True)
        show_kd   = st.checkbox("KD 值（9日）", value=True)

        query_btn = st.button("查詢", type="primary", width="stretch")

    with result_col:
        if not query_btn:
            st.info("請在左側輸入股票代號後，點擊「查詢」按鈕。")
            return

        if not symbol:
            st.error("股票代號不得為空，請重新輸入。")
            return

        # 決定需要哪些 MA 期數
        ma_periods = [p for p, flag in [(5, show_ma5), (10, show_ma10), (20, show_ma20)] if flag]

        # 計算指標需要額外的暖機資料
        # MA20 需 20 筆、KD(9) 需 9 筆，加 buffer 確保首幾筆也準確
        warmup = max([0] + ma_periods + ([9] if show_kd else [])) + 20
        fetch_limit = int(limit) + warmup

        with st.spinner(f"正在取得 {symbol} 的歷史資料…"):
            try:
                df_full = fetch_stock_candles(
                    symbol=symbol,
                    limit=fetch_limit,
                    fields="open,high,low,close,volume,turnover",
                )
            except ValueError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.error(f"API 呼叫失敗：{e}\n\n請確認股票代號是否正確，或稍後再試。")
                return

        if df_full.empty:
            st.warning(f"查無 **{symbol}** 的資料，請確認代號是否正確。")
            return

        # 在完整資料上計算指標（保留 warmup 確保準確性）
        if ma_periods:
            df_full = compute_ma(df_full, ma_periods)
        if show_kd:
            df_full = compute_kd(df_full)

        # 裁切至使用者指定的顯示天數
        df = df_full.tail(int(limit)).reset_index(drop=True)

        latest      = df.iloc[-1]
        prev        = df.iloc[-2] if len(df) >= 2 else latest
        price_delta = float(latest["close"]) - float(prev["close"]) if "close" in df.columns else 0

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        if "close"    in df.columns: m1.metric("收盤價",        f"{latest['close']:,.2f}",   f"{price_delta:+.2f}")
        if "open"     in df.columns: m2.metric("開盤價",        f"{latest['open']:,.2f}")
        if "high"     in df.columns: m3.metric("最高價",        f"{latest['high']:,.2f}")
        if "low"      in df.columns: m4.metric("最低價",        f"{latest['low']:,.2f}")
        if "volume"   in df.columns: m5.metric("成交量（張）",   f"{int(latest['volume']):,}")
        if "turnover" in df.columns: m6.metric("成交值（千元）", f"{int(latest['turnover']):,}")

        st.markdown("---")
        render_ohlcv_chart(df, symbol, show_ma=ma_periods, show_kd=show_kd)
        render_data_table(df, symbol)
