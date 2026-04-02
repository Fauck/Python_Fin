"""
估值分析模組（Tab 8）。 
提供三種估值河流圖：本益比 (P/E)、股價淨值比 (P/B)、歷史殖利率通道。

設計原則
--------
- 100% 使用者主動觸發（無任何排程或背景任務）
- 資料來源（台股限定）：
    * 每日收盤價      → Fugle historical candles（fetch_stock_candles）
    * EPS / BVPS / 配息 → FinMind TaiwanStockPER（每日官方估值指標）反推
- TTM (Trailing Twelve Months)：近四季滾動 EPS，河流圖呈平滑波浪
- @st.cache_data(ttl=3600) 快取避免重複呼叫
- st.session_state 保存查詢結果，避免圖表因互動消失
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from utils import fetch_stock_candles, resolve_stock_input, push_shared_symbol, pull_shared_symbol

# ── FinMind API 設定
_FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# ── 台股代號識別（4~6 位純數字）
_TW_CODE_RE = re.compile(r"^\d{4,6}$")

# ── 估值帶標籤與顏色（由便宜至昂貴）
_BAND_LABELS = ["便宜", "合理偏低", "合理", "合理偏高", "昂貴"]
_LINE_COLORS = ["#1B5E20", "#388E3C", "#F9A825", "#E64A19", "#B71C1C"]
_FILL_COLORS = [
    "rgba(27,94,32,0.22)",
    "rgba(76,175,80,0.18)",
    "rgba(249,168,37,0.18)",
    "rgba(230,74,25,0.20)",
    "rgba(183,17,12,0.22)",
]

# ── 估值區間診斷（zone 0~4 對應便宜~昂貴）
_EVAL_ZONES: List[Tuple[str, str, str, str]] = [
    ("🟢", "便宜",    "#2E7D32", "股價具備安全邊際，為長線佈局良機"),
    ("🟢", "合理偏低", "#558B2F", "估值偏低，可搭配技術面突破訊號考慮進場"),
    ("🟡", "合理",    "#F9A825", "估值合理，建議搭配技術面突破訊號進場"),
    ("🟠", "合理偏高", "#E65100", "估值略高，建議等待回調或控制倉位"),
    ("🔴", "昂貴",    "#B71C1C", "估值偏高，潛在回檔風險大，不建議追高"),
]


# ═════════════════════════════════════════════
# 工具函式
# ═════════════════════════════════════════════

def _to_float(v: Any) -> Optional[float]:
    """安全轉型，NaN / Inf 回傳 None。"""
    try:
        f = float(v)
        return None if (pd.isna(f) or np.isinf(f)) else f
    except Exception:
        return None




def _five_equal_levels(
    ratio_series: pd.Series,
    clip_lo: float = 0.10,
    clip_hi: float = 0.90,
) -> Optional[List[float]]:
    """
    將比率序列裁切 outlier 後，等距切出 5 個水準值。
    """
    lo = ratio_series.quantile(clip_lo)
    hi = ratio_series.quantile(clip_hi)
    if pd.isna(lo) or pd.isna(hi) or lo <= 0 or hi <= lo:
        return None
    step = (hi - lo) / 4
    return [lo + i * step for i in range(5)]


# ═════════════════════════════════════════════
# FinMind 官方每日估值指標
# ═════════════════════════════════════════════

def _fetch_per_finmind(symbol: str, years: int) -> pd.DataFrame:
    """
    取得 FinMind TaiwanStockPER 每日官方估值指標（PE / PB / 殖利率）。

    回傳 DataFrame，index 為 date（DatetimeIndex），欄位：
        PE_ratio       — 本益比
        PB_ratio       — 股價淨值比
        dividend_yield — 現金殖利率（%）
    失敗 / 無資料時回傳空 DataFrame。
    """
    start_date = (datetime.today() - timedelta(days=years * 365 + 30)).strftime("%Y-%m-%d")
    try:
        resp = requests.get(
            _FINMIND_URL,
            params={"dataset": "TaiwanStockPER", "data_id": symbol, "start_date": start_date},
            timeout=30,
        )
        records = resp.json().get("data", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["PER", "PBR", "dividend_yield"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ═════════════════════════════════════════════
# 資料層：Fugle 全資料抓取
# ═════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_valuation_data(symbol: str, years: int = 5) -> Optional[Dict[str, Any]]:
    """
    取得估值分析所需歷史資料（台股限定）。

    資料來源策略
    -----------
    - 每日收盤價 : Fugle historical candles（fetch_stock_candles）
    - EPS / BVPS / 配息 : FinMind TaiwanStockPER（每日官方估值指標）反推

    反推公式
    --------
    EPS   = Price / PE_ratio
    BVPS  = Price / PB_ratio
    配息  = Price × (dividend_yield / 100)

    Returns
    -------
    None（非台股代號 / 資料抓取失敗）或 dict：
        symbol_full   : str
        price_df      : DataFrame (DatetimeIndex, col="close")
        eps_daily     : Series  — EPS（由 PE_ratio 反推），每日
        bvps_daily    : Series  — BVPS（由 PB_ratio 反推），每日
        div_annual    : Series  — 與 div_daily 相同（相容舊介面）
        div_daily     : Series  — 配息（由 dividend_yield 反推），每日
        current_price : float
        current_eps   : float | None
        current_bvps  : float | None
        current_div   : float | None
    """
    raw = symbol.strip().upper()
    if not _TW_CODE_RE.match(raw):
        return None

    # ── 1. 每日收盤價（Fugle candles）────────────────────────────
    limit     = years * 300
    date_to   = datetime.today().strftime("%Y-%m-%d")
    date_from = (datetime.today() - timedelta(days=years * 365 + 60)).strftime("%Y-%m-%d")

    df_candle = fetch_stock_candles(
        symbol=raw, limit=limit,
        date_from=date_from, date_to=date_to,
        fields="open,high,low,close,volume",
    )
    if df_candle.empty or "close" not in df_candle.columns:
        return None

    price_df = df_candle[["date", "close"]].copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.set_index("date")
    price_df.index = pd.DatetimeIndex(price_df.index).tz_localize(None)

    current_price = _to_float(price_df["close"].iloc[-1])
    if current_price is None:
        return None

    # ── 2. 透過 TaiwanStockPER 反推三大基本面數據 ─────────────────
    per_df = _fetch_per_finmind(raw, years)
    eps_daily:  pd.Series = pd.Series(dtype=float, index=price_df.index)
    bvps_daily: pd.Series = pd.Series(dtype=float, index=price_df.index)
    div_daily:  pd.Series = pd.Series(dtype=float, index=price_df.index)

    if not per_df.empty:
        # 對齊每日交易日索引，ffill 填補假日缺漏
        aligned_per = per_df.reindex(price_df.index).ffill()

        if "PER" in aligned_per.columns:
            pe_valid = aligned_per["PER"] > 0
            eps_daily[pe_valid] = (
                price_df["close"][pe_valid] / aligned_per["PER"][pe_valid]
            )

        if "PBR" in aligned_per.columns:
            pb_valid = aligned_per["PBR"] > 0
            bvps_daily[pb_valid] = (
                price_df["close"][pb_valid] / aligned_per["PBR"][pb_valid]
            )

        if "dividend_yield" in aligned_per.columns:
            dy_valid = aligned_per["dividend_yield"] > 0
            div_daily[dy_valid] = (
                price_df["close"][dy_valid] * (aligned_per["dividend_yield"][dy_valid] / 100.0)
            )

    # 平滑化：濾除 FinMind P/E 反推時產生的微小數學雜訊（5 日移動平均）
    eps_daily  = eps_daily.rolling(window=5, min_periods=1).mean()
    bvps_daily = bvps_daily.rolling(window=5, min_periods=1).mean()

    current_eps  = _to_float(eps_daily.dropna().iloc[-1])  if not eps_daily.dropna().empty  else None
    current_bvps = _to_float(bvps_daily.dropna().iloc[-1]) if not bvps_daily.dropna().empty else None
    current_div  = _to_float(div_daily.dropna().iloc[-1])  if not div_daily.dropna().empty  else None

    return {
        "symbol_full":   raw,
        "price_df":      price_df,
        "eps_daily":     eps_daily,
        "bvps_daily":    bvps_daily,
        "div_annual":    div_daily,   # 相容舊介面
        "div_daily":     div_daily,
        "current_price": current_price,
        "current_eps":   current_eps,
        "current_bvps":  current_bvps,
        "current_div":   current_div,
    }


# ═════════════════════════════════════════════
# 進階基本面指標（資料層）
# ═════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_advanced_metrics(
    symbol: str,
    current_price: float,
    current_eps: float,
    current_pe: float,
) -> Dict[str, Any]:
    """
    抓取並計算 8 個核心基本面與進階估值指標。

    Returns
    -------
    dict：
        eps_ttm   : float | None
        pe_ratio  : float | None
        roe       : float | None  （%）
        op_margin : float | None  （%）
        fcf_yi    : float | None  （億元）
        pb_ratio  : float | None  （倍）
        peg_ratio : float | None
        ps_ratio  : float | None  （倍）
    """
    _eps_ok = current_eps and not pd.isna(float(current_eps)) and float(current_eps) > 0
    _pe_ok  = current_pe  and not pd.isna(float(current_pe))  and float(current_pe)  > 0
    result: Dict[str, Any] = {
        "eps_ttm":    round(float(current_eps), 2) if _eps_ok else None,
        "pe_ratio":   round(float(current_pe),  2) if _pe_ok  else None,
        "roe":        None,
        "op_margin":  None,
        "fcf_yi":     None,
        "pb_ratio":   None,
        "peg_ratio":  None,
        "ps_ratio":   None,
    }

    start_2yr = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    # ── 1. TaiwanStockFinancialStatements ─────────────────────────
    stmt_df: pd.DataFrame = pd.DataFrame()
    try:
        resp = requests.get(
            _FINMIND_URL,
            params={
                "dataset":    "TaiwanStockFinancialStatements",
                "data_id":    symbol,
                "start_date": start_2yr,
            },
            timeout=30,
        )
        records = resp.json().get("data", [])
        if records:
            stmt_df = pd.DataFrame(records)
            stmt_df["date"]  = pd.to_datetime(stmt_df["date"])
            stmt_df["value"] = pd.to_numeric(stmt_df["value"], errors="coerce")
    except Exception:
        pass

    if not stmt_df.empty:
        def _ttm(candidates: List[str]) -> Optional[float]:
            """近四季合計（TTM）。"""
            for t in candidates:
                rows = stmt_df[stmt_df["type"] == t].sort_values("date")
                if len(rows) >= 4:
                    s = rows.tail(4)["value"].sum()
                    return None if pd.isna(s) else float(s)
            return None

        def _latest_val(candidates: List[str]) -> Optional[float]:
            """最新一季值。"""
            for t in candidates:
                rows = stmt_df[stmt_df["type"] == t].sort_values("date")
                if not rows.empty:
                    v = rows.iloc[-1]["value"]
                    return None if pd.isna(v) else float(v)
            return None

        rev_ttm    = _ttm(["Revenue", "RevenueFromContractsWithCustomers", "OperatingRevenue"])
        opinc_ttm  = _ttm(["OperatingIncome", "ProfitFromOperations", "OperatingProfit"])
        ni_ttm     = _ttm([
            "NetIncomeAttributableToOwnersOfTheParent",
            "ProfitLoss", "IncomeAfterTaxes", "NetIncomeLoss",
        ])
        equity_now = _latest_val([
            "TotalEquityAttributableToOwnersOfParent",
            "Equity", "StockholdersEquity", "TotalEquity",
        ])

        # EPS 序列（需 8 季以上才可算 YoY，用於 PEG）
        eps_series: Optional[pd.Series] = None
        for _et in ["EarningsPerShareBasic", "BasicEarningsLossPerShare", "EPS"]:
            _rows = stmt_df[stmt_df["type"] == _et].sort_values("date")
            if len(_rows) >= 4:
                eps_series = _rows["value"].reset_index(drop=True)
                break

        # Operating Margin
        if opinc_ttm is not None and rev_ttm and rev_ttm != 0:
            result["op_margin"] = round(opinc_ttm / rev_ttm * 100, 2)

        # ROE = TTM 淨利 / 最新股東權益
        if ni_ttm is not None and equity_now and equity_now > 0:
            result["roe"] = round(ni_ttm / equity_now * 100, 2)

        # P/S = P/E × 淨利率（無需股本資料）
        if (ni_ttm is not None and rev_ttm and rev_ttm != 0
                and _pe_ok and ni_ttm > 0):
            net_margin = ni_ttm / rev_ttm
            if net_margin > 0:
                result["ps_ratio"] = round(float(current_pe) * net_margin, 2)

        # PEG = P/E / EPS YoY 成長率
        if (eps_series is not None and len(eps_series) >= 8 and _pe_ok):
            ttm_now  = float(eps_series.tail(4).sum())
            ttm_prev = float(eps_series.tail(8).head(4).sum())
            if ttm_prev > 0 and ttm_now > 0:
                yoy_pct = (ttm_now - ttm_prev) / ttm_prev * 100
                if yoy_pct > 0:
                    result["peg_ratio"] = round(float(current_pe) / yoy_pct, 2)

    # ── 2. TaiwanStockCashFlowsStatement ──────────────────────────
    try:
        resp = requests.get(
            _FINMIND_URL,
            params={
                "dataset":    "TaiwanStockCashFlowsStatement",
                "data_id":    symbol,
                "start_date": start_2yr,
            },
            timeout=30,
        )
        records = resp.json().get("data", [])
        if records:
            cf_df = pd.DataFrame(records)
            cf_df["date"]  = pd.to_datetime(cf_df["date"])
            cf_df["value"] = pd.to_numeric(cf_df["value"], errors="coerce")

            def _cf_latest(candidates: List[str]) -> Optional[float]:
                for t in candidates:
                    rows = cf_df[cf_df["type"] == t].sort_values("date")
                    if not rows.empty:
                        v = rows.iloc[-1]["value"]
                        return None if pd.isna(v) else float(v)
                return None

            op_cf  = _cf_latest([
                "CashFlowsFromOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashFromOperatingActivities",
            ])
            inv_cf = _cf_latest([
                "CashFlowsFromInvestingActivities",
                "NetCashProvidedByUsedInInvestingActivities",
                "NetCashUsedInInvestingActivities",
                "NetCashFromInvestingActivities",
            ])
            if op_cf is not None and inv_cf is not None:
                # FinMind 台股現金流量表單位通常為千元（千 NTD）
                # 1 億 = 1e5 千元
                result["fcf_yi"] = round((op_cf + inv_cf) / 1e5, 2)
    except Exception:
        pass

    # ── 3. TaiwanStockPER → 最新 PBR ──────────────────────────────
    try:
        resp = requests.get(
            _FINMIND_URL,
            params={
                "dataset":    "TaiwanStockPER",
                "data_id":    symbol,
                "start_date": (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d"),
            },
            timeout=15,
        )
        records = resp.json().get("data", [])
        if records:
            per_df = pd.DataFrame(records)
            per_df["date"] = pd.to_datetime(per_df["date"])
            per_df = per_df.sort_values("date")
            if "PBR" in per_df.columns:
                pbr_s = pd.to_numeric(per_df["PBR"], errors="coerce").dropna()
                if not pbr_s.empty:
                    result["pb_ratio"] = round(float(pbr_s.iloc[-1]), 2)
    except Exception:
        pass

    return result


# ═════════════════════════════════════════════
# 計算層：P/E / P/B / 殖利率 河流資料
# ═════════════════════════════════════════════

def compute_pe_bands(
    data: Dict[str, Any],
    custom_levels: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    計算本益比（P/E）河流圖資料。

    帶線 = TTM EPS × 5 個 P/E 水準，隨 EPS 平滑漂移（波浪狀，非階梯狀）。
    custom_levels：使用者自訂的 5 個 P/E 倍數（任意順序，函式內部確保升冪）。
    """
    price_df    = data["price_df"]
    eps_daily   = data["eps_daily"]
    current_eps = data["current_eps"]

    if eps_daily.dropna().empty or current_eps is None or current_eps <= 0:
        return None

    valid        = eps_daily.notna() & (eps_daily > 0)
    ratio_series = (price_df["close"][valid] / eps_daily[valid]).dropna()

    bands: Dict[str, pd.Series] = {}

    if custom_levels is not None:
        pe_levels: List[float] = sorted(custom_levels)          # 強制升冪
        for i, lvl in enumerate(pe_levels):
            bands[_BAND_LABELS[i]] = (eps_daily * lvl).where(eps_daily > 0)
    else:
        # 滾動分位數（消除未來函數）：window=750 筆約 3 年，min_periods=250 約 1 年
        if len(ratio_series) < 250:
            return None
        _QUANTILES    = [0.10, 0.30, 0.50, 0.70, 0.90]
        rolling_obj   = ratio_series.rolling(window=750, min_periods=250)
        q_series_list = [rolling_obj.quantile(q) for q in _QUANTILES]
        # 取各分位數最新值作為 pe_levels（供 current_bands / 統計摘要使用）
        pe_levels = [float(q_s.dropna().iloc[-1]) for q_s in q_series_list]
        for i, q_s in enumerate(q_series_list):
            aligned_q = q_s.reindex(price_df.index).ffill()
            bands[_BAND_LABELS[i]] = (eps_daily * aligned_q).where(eps_daily > 0)

    bands_df = pd.DataFrame(bands, index=price_df.index)

    current_bands = sorted([lvl * current_eps for lvl in pe_levels])  # 升冪保證
    current_ratio = price_df["close"].iloc[-1] / current_eps

    return {
        "ratio_series":  ratio_series,
        "pe_levels":     pe_levels,
        "bands_df":      bands_df,
        "current_bands": current_bands,
        "current_ratio": round(current_ratio, 2),
        "current_fund":  current_eps,
        "ratio_name":    "P/E（本益比）",
        "fund_label":    "EPS（TTM）",
        "unit":          "倍",
    }


def compute_pb_bands(
    data: Dict[str, Any],
    custom_levels: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    計算股價淨值比（P/B）河流圖資料。使用季度 BVPS 前向填充（較年度更平滑）。
    custom_levels：使用者自訂的 5 個 P/B 倍數（任意順序，函式內部確保升冪）。
    """
    price_df     = data["price_df"]
    bvps_daily   = data["bvps_daily"]
    current_bvps = data["current_bvps"]

    if bvps_daily.dropna().empty or current_bvps is None or current_bvps <= 0:
        return None

    valid        = bvps_daily.notna() & (bvps_daily > 0)
    ratio_series = (price_df["close"][valid] / bvps_daily[valid]).dropna()

    bands: Dict[str, pd.Series] = {}

    if custom_levels is not None:
        pb_levels: List[float] = sorted(custom_levels)          # 強制升冪
        for i, lvl in enumerate(pb_levels):
            bands[_BAND_LABELS[i]] = (bvps_daily * lvl).where(bvps_daily > 0)
    else:
        # 滾動分位數（消除未來函數）：window=750 筆約 3 年，min_periods=250 約 1 年
        if len(ratio_series) < 250:
            return None
        _QUANTILES    = [0.10, 0.30, 0.50, 0.70, 0.90]
        rolling_obj   = ratio_series.rolling(window=750, min_periods=250)
        q_series_list = [rolling_obj.quantile(q) for q in _QUANTILES]
        pb_levels = [float(q_s.dropna().iloc[-1]) for q_s in q_series_list]
        for i, q_s in enumerate(q_series_list):
            aligned_q = q_s.reindex(price_df.index).ffill()
            bands[_BAND_LABELS[i]] = (bvps_daily * aligned_q).where(bvps_daily > 0)

    bands_df = pd.DataFrame(bands, index=price_df.index)

    current_bands = sorted([lvl * current_bvps for lvl in pb_levels])  # 升冪保證
    current_ratio = price_df["close"].iloc[-1] / current_bvps

    return {
        "ratio_series":  ratio_series,
        "pb_levels":     pb_levels,
        "bands_df":      bands_df,
        "current_bands": current_bands,
        "current_ratio": round(current_ratio, 2),
        "current_fund":  current_bvps,
        "ratio_name":    "P/B（股價淨值比）",
        "fund_label":    "每股淨值",
        "unit":          "倍",
    }


def compute_yield_bands(
    data: Dict[str, Any],
    custom_levels: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    計算歷史殖利率通道（Fugle 取得的年配息）。

    高殖利率 ↔ 低股價 → 便宜（帶線底部，綠色）
    低殖利率 ↔ 高股價 → 昂貴（帶線頂部，紅色）

    custom_levels：使用者自訂的 5 個殖利率（%），任意順序。
                   函式內部排成降冪（高殖利率在前）→ 對應帶線價格升冪。
    """
    price_df    = data["price_df"]
    div_daily   = data["div_daily"]
    current_div = data["current_div"]

    if div_daily.dropna().empty or current_div is None or current_div <= 0:
        return None

    valid = (
        price_df["close"].notna() & (price_df["close"] > 0)
        & div_daily.notna() & (div_daily > 0)
    )
    yield_series = (
        div_daily[valid] / price_df["close"][valid] * 100
    ).dropna()

    if custom_levels is not None:
        # 降冪排列殖利率（高殖利率=便宜在前）→ 帶線價格自然升冪
        raw_yields = sorted(custom_levels, reverse=True)
        # 確保所有殖利率 > 0（防呆）
        yield_levels: List[float] = [max(y, 0.01) for y in raw_yields]
    else:
        if len(yield_series) < 30:
            return None
        y_hi = yield_series.quantile(0.90)
        y_lo = yield_series.quantile(0.10)
        if pd.isna(y_lo) or pd.isna(y_hi) or y_lo <= 0 or y_hi <= y_lo:
            return None
        step = (y_hi - y_lo) / 4
        yield_levels = [y_hi - i * step for i in range(5)]

    bands: Dict[str, pd.Series] = {}
    for i, y_lvl in enumerate(yield_levels):
        bands[_BAND_LABELS[i]] = (div_daily / (y_lvl / 100)).where(div_daily > 0)
    bands_df = pd.DataFrame(bands, index=price_df.index)

    # current_bands：price = div / (yield/100)，yield 降冪 → price 升冪，再 sort 保證
    current_bands = sorted([current_div / (y / 100) for y in yield_levels])
    current_yield = current_div / price_df["close"].iloc[-1] * 100

    return {
        "ratio_series":  yield_series,
        "yield_levels":  yield_levels,
        "bands_df":      bands_df,
        "current_bands": current_bands,
        "current_ratio": round(current_yield, 2),
        "current_fund":  current_div,
        "ratio_name":    "殖利率通道",
        "fund_label":    "年配息",
        "unit":          "%",
    }


# ═════════════════════════════════════════════
# 診斷層
# ═════════════════════════════════════════════

def evaluate_current_price(
    current_price: float,
    current_bands: List[float],
    is_downtrend: bool = False,
) -> Dict[str, str]:
    """
    診斷目前股價所在的估值區間。
    current_bands 必須是 5 個升冪排列的帶線價格。
    is_downtrend：最新收盤 < 60MA 時為 True，觸發價值陷阱警告。
    """
    if len(current_bands) < 5:
        return {}

    b0, b1, b2, b3, b4 = [float(x) for x in current_bands]

    if current_price < b1:
        zone = 0
    elif current_price < b2:
        zone = 1
    elif current_price < b3:
        zone = 2
    elif current_price < b4:
        zone = 3
    else:
        zone = 4

    icon, label, color, desc = _EVAL_ZONES[zone]

    # 價值陷阱警告：估值便宜但季線走空，可能是基本面衰退而非真便宜
    if zone in {0, 1} and is_downtrend:
        desc  = (
            "⚠️ 價值陷阱警告：股價處於便宜區但季線下彎，"
            "可能反映基本面衰退，請勿盲目接刀，待技術面築底後再佈局。"
        )
        color = "#E65100"

    return {
        "zone":        str(zone),
        "label":       label,
        "icon":        icon,
        "color":       color,
        "description": desc,
    }


# ═════════════════════════════════════════════
# 圖表層：河流圖
# ═════════════════════════════════════════════

def build_river_chart(
    price_df: pd.DataFrame,
    bands_df: pd.DataFrame,
    title: str,
    current_price: float,
    eval_result: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """繪製估值河流圖（Filled Area Band Chart + 實際股價線）。"""
    fig = go.Figure()

    prev_y: Optional[pd.Series] = None

    for i, label in enumerate(_BAND_LABELS):
        if label not in bands_df.columns:
            continue

        y_vals  = bands_df[label].where(bands_df[label].notna())
        x_dates = pd.DatetimeIndex(bands_df.index).strftime("%Y-%m-%d")

        fill_mode = "tozeroy" if prev_y is None else "tonexty"

        fig.add_trace(go.Scatter(
            x=x_dates,
            y=y_vals,
            mode="lines",
            name=label,
            line=dict(color=_LINE_COLORS[i], width=1, dash="dot"),
            fill=fill_mode,
            fillcolor=_FILL_COLORS[i],
            connectgaps=False,
            hovertemplate=f"{label}<br>%{{x}}<br>參考價：%{{y:,.2f}}<extra></extra>",
        ))
        prev_y = y_vals

    price_x = pd.DatetimeIndex(price_df.index).strftime("%Y-%m-%d")
    fig.add_trace(go.Scatter(
        x=price_x,
        y=price_df["close"],
        mode="lines",
        name="收盤股價",
        line=dict(color="#212121", width=2.5),
        hovertemplate="<b>%{x}</b><br>收盤：%{y:,.2f}<extra></extra>",
    ))

    price_color = eval_result["color"] if eval_result else "#1565C0"
    fig.add_hline(
        y=current_price,
        line=dict(color=price_color, width=1.5, dash="dash"),
        annotation_text=f" 現價 {current_price:,.2f}",
        annotation_position="top right",
        annotation_font_color=price_color,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=90, b=10),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0", tickangle=-30, nticks=12),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


# ═════════════════════════════════════════════
# UI 層：頁面渲染
# ═════════════════════════════════════════════

def _render_eval_badge(
    eval_result: Dict[str, str],
    current_price: float,
    current_ratio: float,
    ratio_name: str,
    unit: str,
) -> None:
    """渲染估值診斷徽章。"""
    color = eval_result["color"]
    icon  = eval_result["icon"]
    label = eval_result["label"]
    desc  = eval_result["description"]

    st.markdown(f"""
<div style="
  border:2px solid {color};
  border-radius:14px;
  padding:20px 16px;
  text-align:center;
  background:{color}18;
  margin-bottom:12px;
">
  <div style="font-size:32px;line-height:1;">{icon}</div>
  <div style="font-size:22px;font-weight:800;color:{color};margin:8px 0 4px;">
    {label}
  </div>
  <div style="font-size:15px;font-weight:600;color:#333;">
    現價 {current_price:,.2f}
  </div>
  <div style="font-size:12px;color:#666;margin-top:4px;">
    當前 {ratio_name.split('（')[0]}：{current_ratio} {unit}
  </div>
</div>
""", unsafe_allow_html=True)

    st.caption(desc)


def _render_zone_cards(
    eval_result: Dict[str, str],
    current_bands: List[float],
) -> None:
    """渲染 5 個估值區間彩色卡片，顯示帶線價格與價格範圍。"""
    b = current_bands
    range_strs = [
        f"低於 {b[1]:,.0f}",
        f"{b[1]:,.0f} ～ {b[2]:,.0f}",
        f"{b[2]:,.0f} ～ {b[3]:,.0f}",
        f"{b[3]:,.0f} ～ {b[4]:,.0f}",
        f"高於 {b[4]:,.0f}",
    ]
    current_zone = int(eval_result.get("zone", -1))
    zone_cols = st.columns(5)
    for i, (zcol, band_p, rng) in enumerate(zip(zone_cols, b, range_strs)):
        icon, lbl, color, _ = _EVAL_ZONES[i]
        is_cur = (i == current_zone)
        border = f"2px solid {color}" if is_cur else f"1px solid {color}66"
        bg     = f"{color}22"         if is_cur else "#fafafa"
        marker = "▶ 現價所在區間"    if is_cur else ""
        zcol.markdown(f"""
<div style="border:{border};border-radius:10px;padding:14px 8px;
            text-align:center;background:{bg};min-height:140px;">
  <div style="font-size:20px;line-height:1.2;">{icon}</div>
  <div style="font-size:13px;font-weight:700;color:{color};
              margin:4px 0;">{lbl}</div>
  <div style="font-size:20px;font-weight:800;color:#212121;
              line-height:1.3;">{band_p:,.2f}</div>
  <div style="font-size:11px;color:#888;margin-top:4px;">{rng}</div>
  <div style="font-size:10px;font-weight:600;color:{color};
              min-height:14px;margin-top:4px;">{marker}</div>
</div>
""", unsafe_allow_html=True)


def _render_advanced_metrics(
    symbol: str,
    current_price: float,
    current_eps: float,
    current_pe: float,
) -> None:
    """渲染 8 個核心基本面與進階估值指標體檢卡片（兩排各 4 格）。"""
    st.markdown("---")
    st.markdown("##### 💎 核心基本面與進階估值體檢")

    with st.spinner("正在抓取進階基本面資料…"):
        m = fetch_advanced_metrics(
            symbol        = symbol,
            current_price = current_price,
            current_eps   = current_eps,
            current_pe    = current_pe,
        )

    def _fmt(v: Any, suffix: str = "", decimals: int = 2, na: str = "無資料") -> str:
        if v is None:
            return na
        try:
            f = float(v)
            if pd.isna(f) or np.isinf(f):
                return na
            return f"{f:.{decimals}f}{suffix}"
        except Exception:
            return na

    def _card(col: Any, title: str, value_str: str, caption_txt: str) -> None:
        is_na     = value_str in ("無資料", "不適用")
        txt_color = "#9E9E9E" if is_na else "#212121"
        col.markdown(
            f"""<div style="border:1px solid #E0E0E0;border-radius:12px;
                padding:18px 10px 12px;background:#FAFAFA;
                text-align:center;min-height:95px;">
  <div style="font-size:11px;color:#888;font-weight:600;
              letter-spacing:0.5px;margin-bottom:6px;">{title}</div>
  <div style="font-size:22px;font-weight:800;color:{txt_color};
              line-height:1.2;">{value_str}</div>
</div>""",
            unsafe_allow_html=True,
        )
        col.caption(caption_txt)

    # ── 第一排：盈利能力 ──────────────────────────────────────────
    r1 = st.columns(4)
    _card(r1[0], "EPS (TTM)",   _fmt(m["eps_ttm"],  " 元"),  "近四季累積盈餘")
    _card(r1[1], "P/E Ratio",   _fmt(m["pe_ratio"], " 倍"),  "投入回本年數")
    _card(r1[2], "ROE",         _fmt(m["roe"],       "%"),    "＞ 15% 屬優秀")
    _card(r1[3], "營益率",      _fmt(m["op_margin"], "%"),    "反映本業獲利能力")

    # ── 第二排：估值與現金流 ──────────────────────────────────────
    r2 = st.columns(4)
    _card(r2[0], "自由現金流 (FCF)", _fmt(m["fcf_yi"],   " 億"),           "落袋為安的真現金")
    _card(r2[1], "P/B Ratio",        _fmt(m["pb_ratio"], " 倍"),           "適合循環／金融股")
    _card(r2[2], "PEG Ratio",        _fmt(m["peg_ratio"], na="不適用"),     "< 1 代表成長股被低估")
    _card(r2[3], "P/S Ratio",        _fmt(m["ps_ratio"], " 倍"),           "適用高成長未獲利新創")


def _render_results(cache: Dict[str, Any]) -> None:
    """從 session_state cache 渲染所有估值結果（圖表、卡片、統計）。"""
    data          = cache["data"]
    band_data     = cache["band_data"]
    eval_result   = cache["eval_result"]
    resolved      = data["symbol_full"]
    current_price = data["current_price"]
    years         = cache["years"]
    is_custom     = cache.get("is_custom", False)

    st.markdown(f"##### {resolved} &nbsp; 估值分析（近 {years} 年）")
    st.caption(
        f"收盤價：Fugle API ｜ EPS／BVPS／配息：FinMind API ｜ 現價：{current_price:,.2f}"
    )

    if is_custom:
        st.warning("⚙️ 目前使用**自訂估值倍數**（非歷史統計分位數）", icon="⚠️")

    _render_eval_badge(
        eval_result   = eval_result,
        current_price = current_price,
        current_ratio = band_data["current_ratio"],
        ratio_name    = band_data["ratio_name"],
        unit          = band_data["unit"],
    )

    ratio_brief = band_data["ratio_name"].split("（")[0]
    suffix      = "自訂倍數" if is_custom else "TTM EPS｜滾動分位估值帶"
    title_str   = (
        f"{resolved}　{ratio_brief}河流圖"
        f"（近 {years} 年｜{suffix}）"
    )
    fig = build_river_chart(
        price_df      = data["price_df"],
        bands_df      = band_data["bands_df"],
        title         = title_str,
        current_price = current_price,
        eval_result   = eval_result,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 各估值區間價格卡片 ─────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 各估值區間帶線價格")
    _render_zone_cards(eval_result, band_data["current_bands"])

    # ── 歷史比率統計摘要 ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"##### {band_data['ratio_name']} 歷史統計摘要")
    rs = band_data["ratio_series"]
    stat_cols = st.columns(5)
    stats = [
        ("最低值（P10）", rs.quantile(0.10)),
        ("25 分位",       rs.quantile(0.25)),
        ("中位數",        rs.quantile(0.50)),
        ("75 分位",       rs.quantile(0.75)),
        ("最高值（P90）", rs.quantile(0.90)),
    ]
    unit = band_data["unit"]
    for col, (lbl, val) in zip(stat_cols, stats):
        col.metric(lbl, f"{val:.2f} {unit}")

    st.caption(
        f"當前 {band_data['ratio_name'].split('（')[0]}："
        f"**{band_data['current_ratio']} {band_data['unit']}**"
        f"　｜　統計區間：近 {years} 年（P10 ~ P90，去除首尾 10% 極端值）"
    )

    # ── 進階基本面與估值體檢 ──────────────────────────────────────
    _eps = _to_float(data.get("current_eps")) or 0.0
    _pe  = round(current_price / _eps, 2) if _eps > 0 else 0.0
    _render_advanced_metrics(
        symbol        = data["symbol_full"],
        current_price = current_price,
        current_eps   = _eps,
        current_pe    = _pe,
    )


def render_valuation_page() -> None:
    """估值分析頁面（Tab 8）。"""
    pull_shared_symbol("val_symbol")

    # session_state 保存查詢結果，避免圖表因互動消失
    if "val_cache" not in st.session_state:
        st.session_state["val_cache"] = None

    # ── 控制面板（expander）────────────────────────────────────
    with st.expander("🔍 查詢條件設定與操作", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            symbol = st.text_input(
                "股票代號/名稱",
                value="2330",
                max_chars=20,
                key="val_symbol",
                help="支援台灣股票。可輸入數字代號 (如 2330) 或中文股名 (如 台積電)。",
            ).strip()
            years: int = st.select_slider(
                "歷史年數",
                options=[3, 4, 5],
                value=5,
                key="val_years",
                help="往前取幾年的歷史資料進行估值區間計算",
            )
        with col_b:
            method = st.selectbox(
                "估值方法",
                options=[
                    "📊 本益比河流圖 (P/E)",
                    "📚 淨值比河流圖 (P/B)",
                    "💰 殖利率通道",
                ],
                key="val_method",
                help=(
                    "• P/E：適合有穩定獲利的成長股（使用 TTM 近四季 EPS）\n"
                    "• P/B：適合景氣循環股或金融股\n"
                    "• 殖利率：適合高股息 ETF 或穩定配息標的"
                ),
            )

        # ── 自訂倍數面板 ──────────────────────────────────────
        with st.expander("⚙️ 進階：自訂估值倍數", expanded=False):
            use_custom: bool = st.checkbox(
                "啟用自訂倍數（覆蓋歷史預設值）",
                key="use_custom_bands",
            )
            custom_levels: Optional[List[float]] = None

            if use_custom:
                mk = (method or "")[:1]
                if mk == "📊":
                    _labels   = ["便宜 P/E 倍", "合理偏低 P/E 倍", "合理 P/E 倍",
                                 "合理偏高 P/E 倍", "昂貴 P/E 倍"]
                    _defaults = [8.0, 12.0, 16.0, 20.0, 25.0]
                    _step, _fmt = 0.5, "%.1f"
                    st.caption("由小到大輸入；系統自動排序，確保帶線升冪。")
                elif mk == "📚":
                    _labels   = ["便宜 P/B 倍", "合理偏低 P/B 倍", "合理 P/B 倍",
                                 "合理偏高 P/B 倍", "昂貴 P/B 倍"]
                    _defaults = [0.8, 1.2, 1.6, 2.0, 2.5]
                    _step, _fmt = 0.1, "%.2f"
                    st.caption("由小到大輸入；系統自動排序，確保帶線升冪。")
                else:
                    _labels   = ["便宜 殖利率%", "合理偏低 殖利率%", "合理 殖利率%",
                                 "合理偏高 殖利率%", "昂貴 殖利率%"]
                    _defaults = [7.0, 5.5, 4.0, 3.0, 2.0]
                    _step, _fmt = 0.1, "%.2f"
                    st.caption("殖利率由高到低（高殖利率=便宜）；系統自動反推對應股價。")

                _vals: List[float] = []
                for _i, (_lbl, _dft) in enumerate(zip(_labels, _defaults)):
                    _v = st.number_input(
                        _lbl, value=_dft, step=_step, format=_fmt,
                        min_value=0.01, key=f"custom_level_{_i}",
                    )
                    _vals.append(float(_v))
                custom_levels = _vals

        query_btn = st.button(
            "查詢估值", type="primary", use_container_width=True,
            key="val_query",
        )

    # ── 按鈕按下 → 計算並存入 session_state ───────────────────
    if query_btn:
        if not symbol:
            st.error("請輸入股票代號或名稱。")
            st.session_state["val_cache"] = None

        else:
            resolved_code, display_name = resolve_stock_input(symbol)
            if not resolved_code:
                st.error(f"找不到符合「{symbol}」的標的，請重新輸入。")
                st.session_state["val_cache"] = None
            else:
                push_shared_symbol(resolved_code)
                with st.spinner(f"正在取得 {display_name} 歷史資料（{years} 年）…"):
                    try:
                        data = fetch_valuation_data(symbol=resolved_code, years=years)
                    except Exception as e:
                        st.error(f"資料抓取失敗：{e}")
                        st.session_state["val_cache"] = None
                        data = None

                if data is None:
                    st.warning(
                        f"查無 **{display_name}** 的歷史資料。\n\n"
                        "可能原因：\n"
                        "- 代號錯誤或 Fugle API 尚未收錄此標的\n"
                        "- 目前網路連線不穩定，請稍後再試"
                    )
                    st.session_state["val_cache"] = None

                else:
                    method_key = (method or "")[:1]
                    _HINTS = {
                        "📊": (
                            f"**{display_name}** 的本益比河流圖資料不足。\n\n"
                            "可能原因：\n"
                            "- FinMind API 尚未收錄此標的的 EPS 季報資料\n"
                            "- 近四季 EPS 有效資料不足（TTM 需至少 4 季）\n\n"
                            "建議改用「淨值比河流圖」或「殖利率通道」。"
                        ),
                        "📚": (
                            f"**{display_name}** 的淨值比河流圖資料不足。\n\n"
                            "可能原因：FinMind API 尚未收錄每股淨值季報資料。\n\n"
                            "建議改用「本益比」或確認代號是否正確。"
                        ),
                        "💰": (
                            f"**{display_name}** 近期無配息記錄，"
                            "無法建立殖利率通道。\n\n"
                            "建議改用「本益比」或「淨值比」河流圖。"
                        ),
                    }

                    with st.spinner("正在計算估值帶線…"):
                        if method_key == "📊":
                            band_data = compute_pe_bands(data, custom_levels=custom_levels)
                        elif method_key == "📚":
                            band_data = compute_pb_bands(data, custom_levels=custom_levels)
                        else:
                            band_data = compute_yield_bands(data, custom_levels=custom_levels)

                    if band_data is None:
                        st.warning(_HINTS.get(method_key, "資料不足，無法計算。"))
                        st.session_state["val_cache"] = None
                    else:
                        # 季線方向判斷（收盤 < 60MA → 空頭，防範價值陷阱）
                        _close  = data["price_df"]["close"]
                        _ma60   = _close.rolling(60).mean().dropna()
                        _is_downtrend = (
                            not _ma60.empty
                            and float(_close.iloc[-1]) < float(_ma60.iloc[-1])
                        )
                        eval_result = evaluate_current_price(
                            data["current_price"], band_data["current_bands"],
                            is_downtrend=_is_downtrend,
                        )
                        st.session_state["val_cache"] = {
                            "data":        data,
                            "band_data":   band_data,
                            "eval_result": eval_result,
                            "years":       years,
                            "is_custom":   use_custom,
                        }

    # ── 從 session_state 渲染（即使沒有按按鈕也能保留圖表）───────
    cache = st.session_state.get("val_cache")

    if cache is None and not query_btn:
        st.info(
            "請在上方輸入股票代號，選擇估值方法後點擊「查詢估值」。\n\n"
            "**支援範圍**：台灣上市上櫃股票（4-6 位數字代號，如 2330、6278）\n\n"
            "**三種估值方法說明**\n\n"
            "| 方法 | 適用標的 |\n"
            "|------|----------|\n"
            "| 本益比河流 (P/E) | 成長股、科技股、有穩定 EPS 者 |\n"
            "| 淨值比河流 (P/B) | 金融股、景氣循環股、資產股 |\n"
            "| 殖利率通道 | 高股息 ETF（0056、00878）、配息穩定存股標的 |\n\n"
            "**資料說明**\n\n"
            "- P/E 帶線採用 **TTM（近四季滾動 EPS）**，呈現平滑波浪而非階梯\n"
            "- 圖表縮放、互動後結果**不會消失**（session_state 保存）\n\n"
            "**資料來源：** 收盤價 → Fugle API ｜ EPS / BVPS / 配息 → FinMind API"
        )
    elif cache is not None:
        _render_results(cache)
