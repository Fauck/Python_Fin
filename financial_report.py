"""
企業財務報告頁面（Tab 6）。
使用 FinMind API 取得損益表並提供關鍵財務指標分析、雙軸組合圖表及量化投資診斷。

台股財報注意事項
---------------
- 資料來源：FinMind API（https://api.finmindtrade.com/api/v4/data）
- Dataset：TaiwanStockFinancialStatements（含 Revenue / GrossProfit / OperatingIncome 等）
- 僅支援台灣上市上櫃股票，輸入 4-6 位數字代號（如 2330、6278）
- 財報延遲：最新一季財報可能延遲 1-4 週才更新
- 缺漏欄位：欄位顯示「—」表示 FinMind 未收錄該科目
- @st.cache_data(ttl=3600) 快取可降低重複呼叫次數
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from financial_translations import METRICS_COL_ZH as _METRICS_COL_ZH
from financial_translations import STMT_INDEX_ZH as _STMT_INDEX_ZH
from utils import resolve_stock_input


# ── 台股純數字代號識別（4~6 位）
_TW_CODE_RE = re.compile(r"^\d{4,6}$")

# ── 財務診斷預設門檻（UI 進階面板未調整時使用）
_DEFAULT_TARGET_REVENUE_GROWTH: float = 10.0   # 目標營收成長率 (%)
_DEFAULT_TARGET_GROSS_MARGIN:   float = 20.0   # 目標毛利率低標 (%)
_DEFAULT_TARGET_NET_MARGIN:     float = 10.0   # 目標淨利率低標 (%)
_DEFAULT_TARGET_DIV_YIELD:      float = 5.0    # 目標現金殖利率 (%)（保留供外部整合）


# ═════════════════════════════════════════════
# 資料層：財報抓取（FinMind API）
# ═════════════════════════════════════════════

_FINMIND_URL          = "https://api.finmindtrade.com/api/v4/data"
_FINMIND_DATASET      = "TaiwanStockFinancialStatements"
_FINMIND_FETCH_YEARS  = 3   # 抓取近 3 年財報資料


def _fetch_finmind_long(symbol: str) -> pd.DataFrame:
    """
    呼叫 FinMind TaiwanStockFinancialStatements，回傳 long-format DataFrame。

    回傳欄位：date（Timestamp）、type（科目名稱）、value（數值）
    失敗時回傳空 DataFrame。

    常見 type 值：
        Revenue / GrossProfit / OperatingIncome / NetIncome / EPS ...
    """
    start_date = (
        datetime.today() - timedelta(days=_FINMIND_FETCH_YEARS * 365)
    ).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            _FINMIND_URL,
            params={
                "dataset":    _FINMIND_DATASET,
                "data_id":    symbol,
                "start_date": start_date,
            },
            timeout=30,
        )
        resp.raise_for_status()
        records = resp.json().get("data", [])
        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["value"] = pd.to_numeric(df.get("value", pd.Series(dtype=float)), errors="coerce")
        df["date"]  = pd.to_datetime(df.get("date",  pd.Series(dtype=str)),  errors="coerce")
        return df.dropna(subset=["date", "value"])

    except Exception:
        return pd.DataFrame()


def _pivot_to_stmt_df(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    將 FinMind long-format 轉為財報分析格式：
    index=type（科目）、columns=Timestamp（降冪，最新在左）。

    此格式與 _find_row() / _extract_key_metrics() 相容。
    """
    if df_long.empty:
        return pd.DataFrame()
    try:
        piv = df_long.pivot_table(
            index="type", columns="date", values="value", aggfunc=lambda x: x.iloc[-1]
        )
        piv.columns = pd.DatetimeIndex(piv.columns)
        return piv.sort_index(axis=1, ascending=False)
    except Exception:
        return pd.DataFrame()


def _aggregate_to_annual(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    將 FinMind 季度 long-format 按自然年加總，轉為年報格式。

    分組邏輯：依 date 的年份分組，對同一科目的 value 加總。
    年度欄位以「該年 12-31」Timestamp 表示，降冪排列。

    注意：Revenue / GrossProfit / OperatingIncome / NetIncome / EPS
          加總後代表全年累計值，適合年報圖表使用。
    """
    if df_long.empty:
        return pd.DataFrame()
    try:
        tmp = df_long.copy()
        tmp["year"] = pd.DatetimeIndex(tmp["date"]).year
        annual = (
            tmp.groupby(["type", "year"])["value"]
            .sum()
            .reset_index()
        )
        annual["date"] = pd.to_datetime(
            annual["year"].astype(str) + "-12-31", errors="coerce"
        )
        piv = annual.pivot_table(
            index="type", columns="date", values="value", aggfunc=lambda x: x.iloc[-1]
        )
        piv.columns = pd.DatetimeIndex(piv.columns)
        return piv.sort_index(axis=1, ascending=False)
    except Exception:
        return pd.DataFrame()


def _fetch_statements_finmind(
    symbol: str,
    quarterly: bool = False,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    透過 FinMind API 取得台股損益表（季報或年報）。

    FinMind TaiwanStockFinancialStatements 涵蓋損益表科目（Revenue、GrossProfit 等），
    資產負債表與現金流量表不在此 dataset 內，回傳 None。

    Parameters
    ----------
    symbol    : 台股純數字代號（例如 "2330"），已驗證為 4-6 位
    quarterly : True → 直接使用季度資料；False → 按年加總
    """
    empty: Dict[str, Optional[pd.DataFrame]] = {
        "income_stmt":   None,
        "balance_sheet": None,
        "cash_flow":     None,
    }

    df_long = _fetch_finmind_long(symbol)
    if df_long.empty:
        return empty

    income = _pivot_to_stmt_df(df_long) if quarterly else _aggregate_to_annual(df_long)
    if not income.empty:
        empty["income_stmt"]   = income
        empty["balance_sheet"] = income
        empty["cash_flow"]     = income
    return empty


@st.cache_data(ttl=3600)
def get_financial_reports(
    symbol: str,
    quarterly: bool = False,
) -> Tuple[Dict[str, Optional[pd.DataFrame]], str]:
    """
    取得台股損益表財報（FinMind API，台股限定）。

    僅支援台灣上市上櫃股票（4-6 位純數字代號）。
    非台股代號直接回傳空結果，不發出 API 請求。

    Parameters
    ----------
    symbol    : 使用者輸入的股票代號
    quarterly : True → 季報；False（預設）→ 年報（按年加總）

    Returns
    -------
    (Dict of DataFrames, resolved_symbol)
    """
    symbol = symbol.strip().upper()

    if not _TW_CODE_RE.match(symbol):
        empty: Dict[str, Optional[pd.DataFrame]] = {
            "income_stmt": None, "balance_sheet": None, "cash_flow": None,
        }
        return empty, symbol

    data = _fetch_statements_finmind(symbol, quarterly)
    return data, symbol


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


def _extract_key_metrics(
    df_income: pd.DataFrame,
    quarterly: bool = False,
) -> Optional[pd.DataFrame]:
    """
    從損益表 DataFrame 提取核心財務指標，並計算衍生欄位。

    輸入：損益表（index=科目, columns=日期，通常為降冪排列）
    輸出：升冪 DataFrame，內部欄位名稱（英文）供計算使用，
          顯示時再透過 _METRICS_COL_ZH 翻譯為繁體中文：
          period / revenue / gross_profit / operating_income / net_income /
          gross_margin / operating_margin / net_margin / revenue_growth

    NaN 安全處理
    -----------
    - 若 revenue 為 0 或 NaN，各利潤率設為 NaN（避免 inf）
    - 季報 revenue_growth 使用 pct_change(periods=4) 計算 YoY 年增率
    - 年報 revenue_growth 使用 pct_change(periods=1)
    """
    revenue_row = _find_row(df_income, [
        "Total Revenue", "TotalRevenue", "Revenue", "Net Revenue",
        "totalRevenue", "revenue", "營業收入",
    ])
    gross_row = _find_row(df_income, [
        "Gross Profit", "GrossProfit",
        "grossProfit", "gross_profit", "毛利", "營業毛利",
    ])
    op_income_row = _find_row(df_income, [
        "Operating Income", "OperatingIncome", "Operating Profit", "EBIT",
        "operatingIncome", "operatingProfit", "營業利益",
    ])
    net_row = _find_row(df_income, [
        "NetIncomeAttributableToOwnersOfTheParent", # FinMind 首選：歸屬於母公司業主之淨利
        "ProfitLoss",                               # FinMind 常見：本期淨利
        "IncomeAfterTaxes",                         # FinMind 常見：稅後淨利
        "NetIncomeLoss",                            # FinMind 備用
        "歸屬於母公司業主之淨利",
        "本期淨利",
        "稅後淨利",
        "Net Income",
        "NetIncome",
        "netIncome",
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
        "period":            periods,
        "revenue":           _to_float_list(revenue_row),
        "gross_profit":      _to_float_list(gross_row),
        "operating_income":  _to_float_list(op_income_row),
        "net_income":        _to_float_list(net_row),
    })

    # 輸入資料為降冪（最新在上）→ 反轉為升冪後再計算成長率
    df = df.iloc[::-1].reset_index(drop=True)

    # 衍生欄位：以 replace(0, nan) 避免除以零產生 inf
    safe_rev = df["revenue"].replace(0, float("nan"))
    df["gross_margin"] = (
        df["gross_profit"] / safe_rev * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)
    df["operating_margin"] = (
        df["operating_income"] / safe_rev * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)
    df["net_margin"] = (
        df["net_income"] / safe_rev * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)

    # 季報使用 pct_change(4) 計算 YoY（去年同期比）；年報用 pct_change(1)
    growth_periods = 4 if quarterly else 1
    df["revenue_growth"] = (
        df["revenue"].pct_change(periods=growth_periods).mul(100).round(2)
    )

    return df


def generate_financial_advice(
    df: pd.DataFrame,
    target_revenue_growth: float = _DEFAULT_TARGET_REVENUE_GROWTH,
    target_gross_margin:   float = _DEFAULT_TARGET_GROSS_MARGIN,
    target_net_margin:     float = _DEFAULT_TARGET_NET_MARGIN,
) -> Dict[str, str]:
    """
    基於最新一期財務數據與使用者自訂門檻給出量化投資診斷。

    診斷面向
    --------
    1. 獲利能力達標判定：最新毛利率 / 淨利率 vs 使用者設定低標
    2. 營收動能判定：最新營收成長率 vs 使用者設定目標
    3. 綜合評級：依兩個面向的達標組合給出四種結論

    Parameters
    ----------
    df                    : _extract_key_metrics 回傳的升冪 DataFrame（至少 2 列）
    target_revenue_growth : 目標營收成長率下限（%），預設 10.0
    target_gross_margin   : 目標毛利率低標（%），預設 20.0
    target_net_margin     : 目標淨利率低標（%），預設 10.0

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

    # ── 1. 獲利能力達標判定 ───────────────────────
    gm_l = float(latest["gross_margin"])
    nm_l = float(latest["net_margin"])

    if pd.isna(gm_l) or pd.isna(nm_l):
        margin_signal = "🔘 毛利率或淨利率資料不完整，無法判斷"
        margin_ok = False
    elif gm_l >= target_gross_margin and nm_l >= target_net_margin:
        margin_signal = (
            f"🟢 獲利能力達標：毛利率 {gm_l:.1f}%（目標 ≥{target_gross_margin:.1f}%）、"
            f"淨利率 {nm_l:.1f}%（目標 ≥{target_net_margin:.1f}%），符合您設定的利潤標準"
        )
        margin_ok = True
    elif gm_l >= target_gross_margin:
        margin_signal = (
            f"🟡 部分達標：毛利率 {gm_l:.1f}% 達標，"
            f"但淨利率 {nm_l:.1f}%（目標 ≥{target_net_margin:.1f}%）未達標"
        )
        margin_ok = False
    elif nm_l >= target_net_margin:
        margin_signal = (
            f"🟡 部分達標：淨利率 {nm_l:.1f}% 達標，"
            f"但毛利率 {gm_l:.1f}%（目標 ≥{target_gross_margin:.1f}%）未達標"
        )
        margin_ok = False
    else:
        margin_signal = (
            f"🔴 獲利能力未達標：毛利率 {gm_l:.1f}%（目標 ≥{target_gross_margin:.1f}%）、"
            f"淨利率 {nm_l:.1f}%（目標 ≥{target_net_margin:.1f}%）均未達標"
        )
        margin_ok = False

    # ── 2. 營收動能判定 ───────────────────────────
    growth = float(latest["revenue_growth"])
    if pd.isna(growth):
        growth_signal = "🔘 營收成長率資料不足"
        growth_ok = False
    elif growth >= target_revenue_growth:
        growth_signal = (
            f"🟢 營收動能強勁：成長率 {growth:.1f}%，超越預期目標"
            f"（≥{target_revenue_growth:.1f}%）"
        )
        growth_ok = True
    elif growth > 0:
        growth_signal = (
            f"🟡 營收正成長但未達標：成長率 {growth:.1f}%"
            f"（目標 ≥{target_revenue_growth:.1f}%）"
        )
        growth_ok = False
    else:
        growth_signal = (
            f"🔴 營收衰退：成長率 {growth:.1f}%"
            f"（目標 ≥{target_revenue_growth:.1f}%）"
        )
        growth_ok = False

    # ── 3. 綜合評級 ───────────────────────────────
    if margin_ok and growth_ok:
        overall = "🌟 全面達標：獲利能力與成長動能均符合目標，基本面優質"
        detail  = "企業各項財務指標均符合您設定的標準，可列入重點追蹤清單。"
    elif growth_ok and not margin_ok:
        overall = "⚠️ 成長達標但獲利品質不足：留意做白工風險"
        detail  = "營收持續成長，但獲利率未達標準；需關注成本控制與定價能力。"
    elif margin_ok and not growth_ok:
        overall = "🟡 獲利穩健但成長趨緩：適合等待營收動能回升"
        detail  = "企業保有良好利潤水準，但頂線成長未達預期；可持續觀察下季動向。"
    else:
        overall = "🚨 全面未達標：獲利能力與成長動能均低於目標，基本面偏弱"
        detail  = "建議觀望，待財務指標回升至目標水準後再考慮介入。"

    # ── 4. 三率三升覆蓋判斷（最高優先權）────────────
    # 毛利率、營業利益率、淨利率三者皆較前一期提升 → 最高評價
    prev = df.iloc[-2]
    def _rate_rose(col: str) -> bool:
        try:
            v_now  = float(latest[col])
            v_prev = float(prev[col])
            return not pd.isna(v_now) and not pd.isna(v_prev) and v_now > v_prev
        except (KeyError, TypeError, ValueError):
            return False

    gm_rose = _rate_rose("gross_margin")
    om_rose = _rate_rose("operating_margin")   # 若無欄位，_rate_rose 回 False
    nm_rose = _rate_rose("net_margin")

    if gm_rose and om_rose and nm_rose:
        overall = "🔥 三率三升：本業與業外獲利全面爆發"
        detail  = (
            "毛利率、營業利益率、淨利率三者均較上期提升，獲利品質全面改善，基本面極強。"
        )

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
    om_vals  = df["operating_margin"].tolist() if "operating_margin" in df.columns else None

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

    # ── 營業利益率（%）（綠線 + 數值標籤，有資料才繪製）
    if om_vals is not None:
        fig.add_trace(
            go.Scatter(
                name="營業利益率 (%)",
                x=periods,
                y=om_vals,
                mode="lines+markers+text",
                line=dict(color="#2E7D32", width=2.5),
                marker=dict(size=7),
                text=[f"{v:.1f}%" if not pd.isna(v) else "" for v in om_vals],
                textposition="top right",
                connectgaps=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "營業利益率：%{y:.1f}%"
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
        margin=dict(l=10, r=10, t=80, b=10),
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
    target_revenue_growth: float = _DEFAULT_TARGET_REVENUE_GROWTH,
    target_gross_margin:   float = _DEFAULT_TARGET_GROSS_MARGIN,
    target_net_margin:     float = _DEFAULT_TARGET_NET_MARGIN,
) -> Optional[Dict[str, Any]]:
    """
    主要外部函式：取得財務報表並執行關鍵指標分析。

    Parameters
    ----------
    symbol                : 股票代號（台股 4-6 位數字 / 美股英文）
    quarterly             : True → 季報；False（預設）→ 年報
    target_revenue_growth : 目標營收成長率下限（%），預設 10.0
    target_gross_margin   : 目標毛利率低標（%），預設 20.0
    target_net_margin     : 目標淨利率低標（%），預設 10.0

    Returns
    -------
    dict {
        "symbol": str,         # 解析後完整代號（含後綴）
        "df":     DataFrame,   # 關鍵指標升冪表（內部英文欄位名）
        "chart":  go.Figure,   # 雙軸組合圖
        "advice": dict,        # generate_financial_advice 回傳值
        "is_tw":  bool,
        "thresholds": dict,    # 實際使用的門檻值（供 UI 顯示確認）
    }
    或 None（損益表資料不足無法分析）

    外部呼叫範例
    -----------
    from financial_report import analyze_financials

    # 使用預設門檻
    result = analyze_financials("2330")

    # 使用自訂門檻
    result = analyze_financials(
        "2330",
        target_revenue_growth=15.0,
        target_gross_margin=30.0,
        target_net_margin=15.0,
    )
    if result:
        st.plotly_chart(result["chart"])
        st.write(result["advice"]["overall"])
    """
    data, resolved = get_financial_reports(symbol, quarterly=quarterly)
    is_tw = _TW_CODE_RE.match(symbol.strip().upper()) is not None

    df_income = data.get("income_stmt")
    if df_income is None or df_income.empty:
        return None

    metrics_df = _extract_key_metrics(df_income, quarterly=quarterly)
    if metrics_df is None or len(metrics_df) < 2:
        return None

    return {
        "symbol": resolved,
        "df":     metrics_df,
        "chart":  build_combo_chart(metrics_df, is_tw=is_tw, symbol=resolved),
        "advice": generate_financial_advice(
            metrics_df,
            target_revenue_growth=target_revenue_growth,
            target_gross_margin=target_gross_margin,
            target_net_margin=target_net_margin,
        ),
        "is_tw":  is_tw,
        "thresholds": {
            "revenue_growth": target_revenue_growth,
            "gross_margin":   target_gross_margin,
            "net_margin":     target_net_margin,
        },
    }


# ═════════════════════════════════════════════
# UI 層：財報頁面渲染
# ═════════════════════════════════════════════

def render_financial_page() -> None:
    """企業財務報告頁面（Tab 6）。"""
    with st.expander("🔍 查詢條件設定與操作", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            raw_symbol = st.text_input(
                "股票代號/名稱",
                value="2330",
                max_chars=20,
                key="fin_symbol",
                help="支援台灣股票。可輸入數字代號 (如 2330) 或中文股名 (如 台積電)。",
            ).strip()
        with col_b:
            period = st.radio(
                "財報期間",
                options=["年報", "季報"],
                horizontal=True,
                key="fin_period",
            )
        quarterly = period == "季報"

        # ── 進階財務診斷參數設定（折疊，預設值可直接使用）
        with st.expander("⚙️ 進階財務診斷參數設定", expanded=False):
            st.caption("未調整時使用預設值，修改後點擊「查詢財報」生效。")
            target_revenue_growth: float = st.number_input(
                "目標營收成長率 (%)",
                min_value=-100.0,
                max_value=1000.0,
                value=_DEFAULT_TARGET_REVENUE_GROWTH,
                step=1.0,
                format="%.1f",
                key="fin_target_rev_growth",
                help="最新一期年增率 ≥ 此值視為「強勁成長」（預設 10%）",
            )
            target_gross_margin: float = st.number_input(
                "目標毛利率低標 (%)",
                min_value=0.0,
                max_value=100.0,
                value=_DEFAULT_TARGET_GROSS_MARGIN,
                step=1.0,
                format="%.1f",
                key="fin_target_gm",
                help="最新一期毛利率 ≥ 此值視為「獲利能力合格」（預設 20%）",
            )
            target_net_margin: float = st.number_input(
                "目標淨利率低標 (%)",
                min_value=-100.0,
                max_value=100.0,
                value=_DEFAULT_TARGET_NET_MARGIN,
                step=1.0,
                format="%.1f",
                key="fin_target_nm",
                help="最新一期淨利率 ≥ 此值視為「盈利能力合格」（預設 10%）",
            )
            st.number_input(
                "目標現金殖利率 (%)",
                min_value=0.0,
                max_value=50.0,
                value=_DEFAULT_TARGET_DIV_YIELD,
                step=0.5,
                format="%.1f",
                key="fin_target_div_yield",
                help="保留欄位，供未來串接殖利率資料時使用（預設 5%）",
                disabled=True,
            )

        query_btn = st.button(
            "查詢財報", type="primary", use_container_width=True,
            key="fin_query",
        )

    if not query_btn:
        st.info(
            "請在上方輸入股票代號，點擊「查詢財報」。\n\n"
            "**支援範圍**\n"
            "- 台灣上市 / 上櫃股票（4-6 位數字代號，如 2330、6278）\n\n"
            "**資料說明**\n"
            "- 資料來源：FinMind API（TaiwanStockFinancialStatements）\n"
            "- 財報可能延遲 1-4 週才更新\n"
            "- 「—」表示 FinMind 未收錄該科目（僅損益表科目有資料）\n"
            "- 台股單位：億元（新台幣）"
        )
        return

    if not raw_symbol:
        st.error("請輸入股票代號或名稱。")
        return

    resolved_code, display_name = resolve_stock_input(raw_symbol)
    if not resolved_code:
        st.error(f"找不到符合「{raw_symbol}」的標的，請重新輸入。")
        return

    with st.spinner(f"正在查詢 {display_name} 財報…"):
        try:
            data, resolved = get_financial_reports(resolved_code, quarterly=quarterly)
        except Exception as e:
            st.error(
                f"查詢失敗：{e}\n\n"
                "可能原因：網路問題、FinMind API 暫時異常，請稍後再試。"
            )
            return

    unit_note = "億元（新台幣）"

    st.markdown(f"##### {display_name}　財務報告（{period}）")

    if all(v is None for v in data.values()):
        st.warning(
            f"查無 **{display_name}** 的財報資料。\n\n"
            "可能原因：\n"
            "- 代號錯誤或 FinMind API 尚未收錄此標的\n"
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
                st.info(f"**{label}** 資料目前無法取得（FinMind API 未收錄或查詢逾時）。")
                continue

            disp_df = _prepare_display_df(raw_df, is_tw=True)
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
            analysis = analyze_financials(
                raw_symbol,
                quarterly=quarterly,
                target_revenue_growth=target_revenue_growth,
                target_gross_margin=target_gross_margin,
                target_net_margin=target_net_margin,
            )
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
        thr    = analysis["thresholds"]
        st.caption(
            f"診斷標準 ── 營收成長目標：{thr['revenue_growth']:.1f}%　｜　"
            f"毛利率低標：{thr['gross_margin']:.1f}%　｜　"
            f"淨利率低標：{thr['net_margin']:.1f}%"
        )

        st.markdown("**獲利能力達標判定**")
        st.markdown(advice["margin_signal"])
        st.markdown("**營收動能判定**")
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

            for c in ["revenue", "gross_profit", "operating_income", "net_income"]:
                if c in disp.columns:
                    disp[c] = (disp[c] / div).round(2).astype(str) + f" {unit}"
            for c in ["gross_margin", "operating_margin", "net_margin", "revenue_growth"]:
                if c in disp.columns:
                    disp[c] = disp[c].apply(
                        lambda v: f"{float(v):.2f}%" if not pd.isna(v) else "—"
                    )

            # 套用中英對照表重命名欄位
            disp = disp.rename(columns=_METRICS_COL_ZH)
            st.dataframe(disp, use_container_width=True, hide_index=True)
