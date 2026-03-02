"""
企業財務報告頁面（Tab 6）。
使用 yfinance 取得損益表、資產負債表、現金流量表。
並提供關鍵財務指標分析、雙軸組合圖表及量化投資診斷。

台股財報注意事項
---------------
- 資料來源：Yahoo Finance（yfinance 爬取），非即時官方資料
- 財報延遲：最新一季財報可能延遲 2-8 週才更新至 Yahoo Finance
- 缺漏欄位：Yahoo Finance 對中小型台股的科目收錄不完整，顯示「—」
- 頻率限制：Yahoo Finance 有 HTTP 429 速率限制，@st.cache_data 可降低呼叫次數
- 台灣上市股票使用 .TW 後綴（例如 2330.TW）
- 台灣上櫃股票使用 .TWO 後綴（例如 6278.TWO）
  → 本模組自動嘗試 .TW，若無資料再改試 .TWO
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from financial_translations import METRICS_COL_ZH as _METRICS_COL_ZH
from financial_translations import STMT_INDEX_ZH as _STMT_INDEX_ZH


# ── 台股純數字代號識別（4~6 位）
_TW_CODE_RE = re.compile(r"^\d{4,6}$")


# ═════════════════════════════════════════════
# 資料層：財報抓取
# ═════════════════════════════════════════════

def _get_attr(ticker: yf.Ticker, *attr_names: str) -> Optional[pd.DataFrame]:
    """
    依序嘗試多個 yfinance 屬性名稱，回傳第一個非空的 DataFrame。

    背景：yfinance 0.2.x 將部分屬性重新命名（例如 financials → income_stmt），
    此函式同時相容新舊 API，避免版本差異導致資料取不到。
    """
    for name in attr_names:
        try:
            df = getattr(ticker, name, None)
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    return None


def _fetch_statements(
    symbol_full: str,
    quarterly: bool = False,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    使用 yfinance 取得三大財報（核心邏輯，不含 Streamlit 元素）。

    常見例外情境
    ----------
    - HTTP 429：Yahoo Finance 頻率限制，呼叫端以 @st.cache_data(ttl=3600) 降低次數
    - HTTP 404 / empty：代號錯誤或 Yahoo 未收錄
    - AttributeError：yfinance 版本差異，以 _get_attr 多屬性嘗試處理
    - timeout / ConnectionError：網路問題，try-except 捕獲後回傳 None
    """
    try:
        tk = yf.Ticker(symbol_full)
    except Exception:
        return {"income_stmt": None, "balance_sheet": None, "cash_flow": None}

    if quarterly:
        income = _get_attr(tk, "quarterly_income_stmt", "quarterly_financials")
        bs     = _get_attr(tk, "quarterly_balance_sheet")
        cf     = _get_attr(tk, "quarterly_cashflow")
    else:
        income = _get_attr(tk, "income_stmt", "financials")
        bs     = _get_attr(tk, "balance_sheet")
        cf     = _get_attr(tk, "cashflow")

    return {
        "income_stmt":   income,
        "balance_sheet": bs,
        "cash_flow":     cf,
    }


@st.cache_data(ttl=3600)
def get_financial_reports(
    symbol: str,
    quarterly: bool = False,
) -> Tuple[Dict[str, Optional[pd.DataFrame]], str]:
    """
    取得企業三大財務報表（含台股自動後綴識別）。

    台股自動識別流程
    ---------------
    1. 輸入為 4-6 位純數字（如 2330）→ 先嘗試 .TW（上市）
    2. .TW 三張報表全空 → 改嘗試 .TWO（上櫃）
    3. 已含後綴或英文代號（TSLA、AAPL）→ 直接使用

    Parameters
    ----------
    symbol    : 使用者輸入的股票代號（不分大小寫）
    quarterly : True → 季報；False（預設）→ 年報

    Returns
    -------
    (Dict of DataFrames, resolved_symbol)
    """
    symbol = symbol.strip().upper()

    if _TW_CODE_RE.match(symbol):
        resolved = f"{symbol}.TW"
        data     = _fetch_statements(resolved, quarterly)
        if all(v is None for v in data.values()):
            resolved = f"{symbol}.TWO"
            data     = _fetch_statements(resolved, quarterly)
    else:
        resolved = symbol
        data     = _fetch_statements(resolved, quarterly)

    return data, resolved


# ═════════════════════════════════════════════
# 格式化工具
# ═════════════════════════════════════════════

def _fmt_value(v: Any, is_tw: bool = True) -> str:
    """
    格式化財報數值。

    is_tw=True  → 億 / 萬（台股新台幣）
    is_tw=False → B / M / K（美股美元等）
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        fv = float(v)
        if is_tw:
            if abs(fv) >= 1e8:
                return f"{fv / 1e8:,.2f} 億"
            elif abs(fv) >= 1e4:
                return f"{fv / 1e4:,.2f} 萬"
            else:
                return f"{fv:,.2f}"
        else:
            if abs(fv) >= 1e9:
                return f"{fv / 1e9:,.2f} B"
            elif abs(fv) >= 1e6:
                return f"{fv / 1e6:,.2f} M"
            elif abs(fv) >= 1e3:
                return f"{fv / 1e3:,.2f} K"
            else:
                return f"{fv:,.2f}"
    except (ValueError, TypeError):
        return str(v)


def _prepare_display_df(df: pd.DataFrame, is_tw: bool = True) -> pd.DataFrame:
    """
    整理財報 DataFrame 為可展示格式：
    - 欄位名稱（Timestamp）→ "YYYY-MM-DD" 字串
    - 行索引（科目名稱）→ 套用 _STMT_INDEX_ZH 中英對照翻譯
    - 所有數值格式化（億/萬 或 B/M/K）
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # 欄位名稱（通常為 pd.Timestamp）→ 字串
    df.columns = pd.Index([
        col.strftime("%Y-%m-%d") if isinstance(col, pd.Timestamp) else str(col)
        for col in df.columns
    ])

    # 行索引科目名稱中文化（精確比對，未收錄者保留原文）
    df = df.rename(index=_STMT_INDEX_ZH)

    # 數值格式化
    df = df.apply(lambda col: col.apply(lambda v: _fmt_value(v, is_tw)))
    return df


# ═════════════════════════════════════════════
# 財務分析層：關鍵指標提取與診斷
# ═════════════════════════════════════════════

def _find_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """
    在財報 DataFrame 的 index 中搜尋目標科目。
    先精確比對（忽略大小寫），再模糊比對（包含關係），確保優先找到最正確欄位。
    """
    for candidate in candidates:
        for idx in df.index:
            if candidate.lower() == str(idx).lower():
                return df.loc[idx]
    for candidate in candidates:
        for idx in df.index:
            if candidate.lower() in str(idx).lower():
                return df.loc[idx]
    return None


def _extract_key_metrics(df_income: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    從損益表 DataFrame 提取核心財務指標，並計算衍生欄位。

    輸入：yfinance 損益表（index=科目, columns=日期，通常為降冪排列）
    輸出：升冪 DataFrame，內部欄位名稱（英文）供計算使用，
          顯示時再透過 _METRICS_COL_ZH 翻譯為繁體中文：
          period / revenue / gross_profit / net_income /
          gross_margin / net_margin / revenue_growth

    NaN 安全處理
    -----------
    - 若 revenue 為 0 或 NaN，毛利率 / 淨利率設為 NaN（避免 inf）
    - pct_change 首期為 NaN（正常，無前一期可比）
    """
    revenue_row = _find_row(df_income, [
        "Total Revenue", "TotalRevenue", "Revenue", "Net Revenue",
    ])
    gross_row = _find_row(df_income, [
        "Gross Profit", "GrossProfit",
    ])
    net_row = _find_row(df_income, [
        "Net Income", "NetIncome",
        "Net Income Common Stockholders",
        "Net Income From Continuing And Discontinued Operation",
    ])

    if revenue_row is None:
        return None

    # 欄位名稱（Timestamp → 字串）
    periods: List[str] = [
        col.strftime("%Y-%m-%d") if isinstance(col, pd.Timestamp) else str(col)
        for col in df_income.columns
    ]

    def _to_float_list(row: Optional[pd.Series]) -> List[float]:
        if row is None:
            return [float("nan")] * len(periods)
        return [float(pd.to_numeric(v, errors="coerce")) for v in row.values]

    df = pd.DataFrame({
        "period":       periods,
        "revenue":      _to_float_list(revenue_row),
        "gross_profit": _to_float_list(gross_row),
        "net_income":   _to_float_list(net_row),
    })

    # yfinance 通常降冪（最新在上）→ 反轉為升冪後再計算成長率
    df = df.iloc[::-1].reset_index(drop=True)

    # 衍生欄位：以 replace(0, nan) 避免除以零產生 inf
    safe_rev = df["revenue"].replace(0, float("nan"))
    df["gross_margin"] = (
        df["gross_profit"] / safe_rev * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)
    df["net_margin"] = (
        df["net_income"] / safe_rev * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)
    df["revenue_growth"] = df["revenue"].pct_change().mul(100).round(2)

    return df


def generate_financial_advice(df: pd.DataFrame) -> Dict[str, str]:
    """
    基於最近兩期財務數據給出量化投資診斷。

    診斷面向
    --------
    1. 獲利能力（雙率雙升）：毛利率 + 淨利率同步提升 → 🟢
    2. 營收動能：年增率 >10% 🟢 / -5%~10% 🟡 / <-5% 🔴
    3. 綜合評級：四種情境對應不同操作建議

    Parameters
    ----------
    df : _extract_key_metrics 回傳的升冪 DataFrame（至少 2 列）

    Returns
    -------
    dict { margin_signal, growth_signal, overall, detail }
    """
    _na: Dict[str, str] = {
        "margin_signal": "資料不足，無法判斷",
        "growth_signal": "資料不足，無法判斷",
        "overall":       "資料不足，無法給出評級",
        "detail":        "",
    }
    if df is None or len(df) < 2:
        return _na

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    # ── 1. 獲利能力（雙率雙升）────────────────────
    gm_l = float(latest["gross_margin"])
    nm_l = float(latest["net_margin"])
    gm_p = float(prev["gross_margin"])
    nm_p = float(prev["net_margin"])

    if any(pd.isna(v) for v in [gm_l, nm_l, gm_p, nm_p]):
        margin_signal = "🔘 毛利率或淨利率資料不完整，無法判斷"
        double_up = False
    elif gm_l > gm_p and nm_l > nm_p:
        margin_signal = (
            f"🟢 雙率雙升：毛利率 {gm_p:.1f}% → {gm_l:.1f}%，"
            f"淨利率 {nm_p:.1f}% → {nm_l:.1f}%，獲利能力轉強"
        )
        double_up = True
    elif gm_l > gm_p or nm_l > nm_p:
        which = "毛利率" if gm_l > gm_p else "淨利率"
        margin_signal = f"🟡 單率上升：{which}改善，另一項指標持平或下滑"
        double_up = False
    else:
        margin_signal = (
            f"🔴 雙率收斂：毛利率 {gm_p:.1f}% → {gm_l:.1f}%，"
            f"淨利率 {nm_p:.1f}% → {nm_l:.1f}%，獲利能力承壓"
        )
        double_up = False

    # ── 2. 營收動能 ───────────────────────────────
    growth = float(latest["revenue_growth"])
    if pd.isna(growth):
        growth_signal = "🔘 營收成長率資料不足"
        growth_strong = False
    elif growth > 10:
        growth_signal = f"🟢 營收強勁成長（{growth:.1f}%），具備成長動能"
        growth_strong = True
    elif growth >= -5:
        growth_signal = f"🟡 營收動能平穩（{growth:.1f}%），維持現狀"
        growth_strong = False
    else:
        growth_signal = f"🔴 營收年減（{growth:.1f}%），面臨衰退壓力"
        growth_strong = False

    # ── 3. 綜合評級 ───────────────────────────────
    if growth_strong and double_up:
        overall = "🌟 基本面強勢：營收與獲利雙重改善，適合長線偏多看待"
        detail  = "企業具備定價能力與規模效益，可列入長線觀察。"
    elif growth_strong and not double_up:
        overall = "⚠️ 留意做白工風險：營收成長但獲利率遭壓縮"
        detail  = "成長品質存疑，需關注毛利率變化趨勢；可短線操作，長線需謹慎。"
    elif not growth_strong and double_up:
        overall = "🟡 獲利效率改善，但營收動能不足"
        detail  = "企業正在改善成本結構，頂線成長待觀察，可追蹤下季動向。"
    else:
        overall = "🚨 基本面轉弱：營收與獲利同步下滑，建議觀望或保守操作"
        detail  = "需等待基本面反轉訊號再介入。"

    return {
        "margin_signal": margin_signal,
        "growth_signal": growth_signal,
        "overall":       overall,
        "detail":        detail,
    }


def build_combo_chart(
    df: pd.DataFrame,
    is_tw: bool = True,
    symbol: str = "",
) -> go.Figure:
    """
    繪製「總營收/淨利柱狀圖 ＋ 毛利率/淨利率折線圖」雙軸組合圖。

    圖表標籤全面中文化（標題、軸標籤、圖例、懸浮提示）。
    NaN 安全：柱狀圖資料以 fillna(0) 處理，折線圖保留 NaN（顯示斷線）。

    左 Y 軸：金額（億元 TWD 或 十億原始貨幣）
    右 Y 軸：比率（%）
    """
    divisor    = 1e8 if is_tw else 1e9
    amount_lbl = "億元（TWD）" if is_tw else "十億（原始貨幣）"
    unit_str   = "億" if is_tw else "B"
    periods    = df["period"].tolist()

    # 柱狀圖資料：NaN 填 0 避免 Plotly 渲染錯誤
    rev_vals = (df["revenue"].fillna(0)     / divisor).round(2).tolist()
    ni_vals  = (df["net_income"].fillna(0)  / divisor).round(2).tolist()

    # 折線圖資料：保留 NaN（Plotly 自動斷線，語意更正確）
    gm_vals  = df["gross_margin"].tolist()
    nm_vals  = df["net_margin"].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── 總營收（淺藍柱）
    fig.add_trace(
        go.Bar(
            name="總營收",
            x=periods,
            y=rev_vals,
            marker_color="#90CAF9",
            opacity=0.85,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"總營收：%{{y:.2f}} {unit_str}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # ── 淨利（深藍柱）
    fig.add_trace(
        go.Bar(
            name="淨利",
            x=periods,
            y=ni_vals,
            marker_color="#1565C0",
            opacity=0.88,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"淨利：%{{y:.2f}} {unit_str}"
                "<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    # ── 毛利率（%）（橘線 + 數值標籤）
    fig.add_trace(
        go.Scatter(
            name="毛利率 (%)",
            x=periods,
            y=gm_vals,
            mode="lines+markers+text",
            line=dict(color="#F57C00", width=2.5),
            marker=dict(size=7),
            text=[f"{v:.1f}%" if not pd.isna(v) else "" for v in gm_vals],
            textposition="top center",
            connectgaps=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "毛利率：%{y:.1f}%"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # ── 淨利率（%）（紅線 + 數值標籤）
    fig.add_trace(
        go.Scatter(
            name="淨利率 (%)",
            x=periods,
            y=nm_vals,
            mode="lines+markers+text",
            line=dict(color="#C62828", width=2.5),
            marker=dict(size=7),
            text=[f"{v:.1f}%" if not pd.isna(v) else "" for v in nm_vals],
            textposition="bottom center",
            connectgaps=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "淨利率：%{y:.1f}%"
                "<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    title_text = (
        f"{symbol}　企業近期財務表現與獲利能力"
        if symbol else "企業近期財務表現與獲利能力"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=15)),
        barmode="group",
        height=460,
        legend=dict(orientation="h", y=1.10, x=0, xanchor="left"),
        margin=dict(l=70, r=70, t=80, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
    )

    # X 軸
    fig.update_xaxes(title_text="財報期間 (季度/年度)", gridcolor="#f0f0f0")

    # 左 Y 軸（金額）
    fig.update_yaxes(
        title_text=f"金額（{amount_lbl}）",
        secondary_y=False,
        gridcolor="#f0f0f0",
    )

    # 右 Y 軸（百分比）
    fig.update_yaxes(
        title_text="百分比 (%)",
        secondary_y=True,
        gridcolor="#f0f0f0",
    )

    return fig


def analyze_financials(
    symbol: str,
    quarterly: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    主要外部函式：取得財務報表並執行關鍵指標分析。

    Parameters
    ----------
    symbol    : 股票代號（台股 4-6 位數字 / 美股英文）
    quarterly : True → 季報；False（預設）→ 年報

    Returns
    -------
    dict {
        "symbol": str,         # 解析後完整代號（含後綴）
        "df":     DataFrame,   # 關鍵指標升冪表（內部英文欄位名）
        "chart":  go.Figure,   # 雙軸組合圖
        "advice": dict,        # generate_financial_advice 回傳值
        "is_tw":  bool,
    }
    或 None（損益表資料不足無法分析）

    外部呼叫範例
    -----------
    from financial_report import analyze_financials

    result = analyze_financials("2330")
    if result:
        st.plotly_chart(result["chart"])
        st.write(result["advice"]["overall"])
    """
    data, resolved = get_financial_reports(symbol, quarterly=quarterly)
    is_tw = _TW_CODE_RE.match(symbol.strip().upper()) is not None

    df_income = data.get("income_stmt")
    if df_income is None or df_income.empty:
        return None

    metrics_df = _extract_key_metrics(df_income)
    if metrics_df is None or len(metrics_df) < 2:
        return None

    return {
        "symbol": resolved,
        "df":     metrics_df,
        "chart":  build_combo_chart(metrics_df, is_tw=is_tw, symbol=resolved),
        "advice": generate_financial_advice(metrics_df),
        "is_tw":  is_tw,
    }


# ═════════════════════════════════════════════
# UI 層：財報頁面渲染
# ═════════════════════════════════════════════

def render_financial_page() -> None:
    """企業財務報告頁面（Tab 6）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 查詢條件")
        raw_symbol = st.text_input(
            "股票代號",
            value="2330",
            max_chars=20,
            key="fin_symbol",
            help=(
                "台股輸入 4-6 位數字代號（如 2330），系統自動嘗試 .TW / .TWO。\n"
                "美股直接輸入英文代號（如 TSLA、AAPL）。"
            ),
        ).strip()

        period = st.radio(
            "財報期間",
            options=["年報", "季報"],
            horizontal=True,
            key="fin_period",
        )
        quarterly = period == "季報"

        query_btn = st.button(
            "查詢財報", type="primary", use_container_width=True,
            key="fin_query",
        )

    with result_col:
        if not query_btn:
            st.info(
                "請在左側輸入股票代號，點擊「查詢財報」。\n\n"
                "**支援範圍**\n"
                "- 台灣上市（.TW）/ 上櫃（.TWO）股票\n"
                "- 美股及其他 Yahoo Finance 收錄標的\n\n"
                "**資料說明**\n"
                "- 資料來源：Yahoo Finance（透過 yfinance）\n"
                "- 台股財報可能延遲 1-2 個季度\n"
                "- 「—」表示 Yahoo Finance 未收錄該科目\n"
                "- 台股單位：億元（新台幣）；美股單位：B / M（USD）"
            )
            return

        if not raw_symbol:
            st.error("股票代號不得為空。")
            return

        with st.spinner(f"正在查詢 {raw_symbol} 財報…"):
            try:
                data, resolved = get_financial_reports(raw_symbol, quarterly=quarterly)
            except Exception as e:
                st.error(
                    f"查詢失敗：{e}\n\n"
                    "可能原因：網路問題、Yahoo Finance 暫時封鎖（HTTP 429），請稍後再試。"
                )
                return

        is_tw     = _TW_CODE_RE.match(raw_symbol.strip().upper()) is not None
        unit_note = "億元（新台幣）" if is_tw else "B / M（原始報告貨幣）"

        st.markdown(f"##### {resolved}　財務報告（{period}）")

        if all(v is None for v in data.values()):
            st.warning(
                f"查無 **{resolved}** 的財報資料。\n\n"
                "可能原因：\n"
                "- 台股代號錯誤（上市用 .TW，上櫃用 .TWO）\n"
                "- Yahoo Finance 尚未收錄此標的\n"
                "- 目前為非交易時段，資料尚未更新\n"
                "- 請稍後重試或確認代號是否正確"
            )
            return

        # ── 三大財報原始科目表（科目行索引已透過 _STMT_INDEX_ZH 中文化）
        report_tabs = st.tabs(["📊 損益表", "🏛 資產負債表", "💰 現金流量表"])
        tab_map = [
            ("income_stmt",   "損益表"),
            ("balance_sheet", "資產負債表"),
            ("cash_flow",     "現金流量表"),
        ]

        for tab, (key, label) in zip(report_tabs, tab_map):
            with tab:
                raw_df = data[key]
                if raw_df is None:
                    st.info(f"**{label}** 資料目前無法取得（Yahoo Finance 未收錄或查詢逾時）。")
                    continue

                disp_df = _prepare_display_df(raw_df, is_tw=is_tw)
                if disp_df.empty:
                    st.info(f"**{label}** 回傳空資料。")
                    continue

                st.caption(
                    f"數值單位：{unit_note}　｜　"
                    f"共 {len(disp_df)} 個科目 × {len(disp_df.columns)} 期"
                )
                st.dataframe(disp_df, use_container_width=True)

        # ── 財務關鍵指標分析與投資診斷 ──────────────────
        st.markdown("---")
        st.markdown("##### 📈 財務關鍵指標分析與投資診斷")

        with st.spinner("正在計算財務指標…"):
            try:
                analysis = analyze_financials(raw_symbol, quarterly=quarterly)
            except Exception:
                analysis = None

        if analysis is None:
            st.info(
                "損益表資料不足（需至少兩期「總營收」資料），無法執行財務指標分析。"
            )
        else:
            # 雙軸組合圖（柱 + 折線）
            st.plotly_chart(analysis["chart"], use_container_width=True)

            # 投資診斷訊號
            advice = analysis["advice"]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**獲利能力（雙率雙升判斷）**")
                st.markdown(advice["margin_signal"])
            with col_b:
                st.markdown("**營收動能**")
                st.markdown(advice["growth_signal"])

            st.markdown(f"**綜合評級：** {advice['overall']}")
            if advice["detail"]:
                st.caption(advice["detail"])

            # 關鍵指標明細表（摺疊）
            # 使用 _METRICS_COL_ZH 的 rename(columns=...) 完成欄位中文化
            with st.expander("查看關鍵指標明細數字", expanded=False):
                disp = analysis["df"].copy()
                div  = 1e8 if analysis["is_tw"] else 1e9
                unit = "億" if analysis["is_tw"] else "B"

                for c in ["revenue", "gross_profit", "net_income"]:
                    disp[c] = (disp[c] / div).round(2).astype(str) + f" {unit}"
                for c in ["gross_margin", "net_margin", "revenue_growth"]:
                    disp[c] = disp[c].apply(
                        lambda v: f"{float(v):.2f}%" if not pd.isna(v) else "—"
                    )

                # 套用中英對照表重命名欄位
                disp = disp.rename(columns=_METRICS_COL_ZH)
                st.dataframe(disp, use_container_width=True, hide_index=True)
