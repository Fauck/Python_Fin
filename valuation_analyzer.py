"""
估值分析模組（Tab 8）。
提供三種估值河流圖：本益比 (P/E)、股價淨值比 (P/B)、歷史殖利率通道。

設計原則
--------
- 100% 使用者主動觸發（無任何排程或背景任務）
- 資料來源：
    * 每日收盤價      → yfinance
    * 配息記錄        → Fugle API（台股）/ yfinance 回退
    * 季度 EPS/BVPS  → yfinance quarterly_income_stmt / quarterly_balance_sheet
- TTM (Trailing Twelve Months)：近四季滾動 EPS，河流圖呈平滑波浪
- @st.cache_data(ttl=3600) 快取避免重複呼叫
- st.session_state 保存查詢結果，避免圖表因互動消失
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils import get_fugle_client


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


def _find_row(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """在 DataFrame index 中尋找目標科目（先精確比對再模糊比對）。"""
    for c in candidates:
        for idx in df.index:
            if c.lower() == str(idx).lower():
                return df.loc[idx]
    for c in candidates:
        for idx in df.index:
            if c.lower() in str(idx).lower():
                return df.loc[idx]
    return None


def _row_to_series(row: pd.Series) -> pd.Series:
    """
    將 yfinance 財報 DataFrame 的一行（columns = Timestamps）
    轉為日期升冪索引的 pd.Series[float]。
    """
    data: Dict[pd.Timestamp, float] = {}
    for col, val in row.items():
        try:
            dt = pd.Timestamp(str(col)).tz_localize(None)
            v  = _to_float(val)
            if v is not None:
                data[dt] = v
        except Exception:
            pass
    return pd.Series(data).sort_index()


def _align_to_daily(
    annual_series: pd.Series,
    price_df: pd.DataFrame,
) -> pd.Series:
    """
    將基本面資料（年度或季度，索引為財報日期）前向填充至每日交易日索引。
    price_df 必須已有 DatetimeIndex。
    """
    combined = annual_series.reindex(
        annual_series.index.union(price_df.index)
    ).sort_index().ffill()
    return combined.reindex(price_df.index)


def _compute_ttm_quarterly(
    q_series: pd.Series,
    price_df: pd.DataFrame,
) -> pd.Series:
    """
    將季度 EPS Series（升冪日期索引）轉為 TTM（近四季滾動加總），
    再前向填充至每日。

    TTM EPS = 最近四個季度 EPS 之和，每當新季報發布即平滑更新。
    若有效季度 < 4，回傳空 Series（上層函式視為資料不足）。
    """
    q = q_series.sort_index().dropna()
    if len(q) < 4:
        return pd.Series(dtype=float, index=price_df.index)
    ttm = q.rolling(4, min_periods=4).sum().dropna()
    if ttm.empty:
        return pd.Series(dtype=float, index=price_df.index)
    return _align_to_daily(ttm, price_df)


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
# Fugle 配息資料（台股優先來源）
# ═════════════════════════════════════════════

def _fetch_dividends_fugle(symbol: str) -> Optional[pd.Series]:
    """
    透過 Fugle API 取得台股歷史配息，回傳以年度結尾日為索引的年度現金配息 Series。
    失敗時回傳 None（交由上層 fallback 至 yfinance）。

    Fugle dividends 資料格式（動態欄位名稱，以模式比對）：
    - 現金股利欄：含 "cash" + "div" 的欄位
    - 日期欄    ：含 "date" 或 "year" 的欄位
    """
    try:
        client = get_fugle_client()
        raw = client.stock.historical.dividends(  # type: ignore[attr-defined]
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

        # 找現金股利欄
        cash_col: Optional[str] = next(
            (c for c in df.columns if "cash" in c.lower() and "div" in c.lower()),
            next((c for c in df.columns if "cash" in c.lower()), None),
        )
        # 找日期欄
        date_col: Optional[str] = next(
            (c for c in df.columns if "date" in c.lower()),
            next((c for c in df.columns if "year" in c.lower()), None),
        )

        if cash_col is None or date_col is None:
            return None

        df[cash_col] = pd.to_numeric(df[cash_col], errors="coerce")
        df = df.dropna(subset=[cash_col])
        df = df[df[cash_col] > 0]

        if df.empty:
            return None

        # 日期欄轉型
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce")
        df = df.dropna(subset=[date_col])

        # 依年加總（同一年可能有多次配息）
        df["_year"] = pd.DatetimeIndex(df[date_col]).year
        annual = df.groupby("_year")[cash_col].sum()

        # 轉換為年底日期索引
        idx = pd.DatetimeIndex([f"{y}-12-31" for y in annual.index])
        return pd.Series(annual.values, index=idx, dtype=float).sort_index()

    except Exception:
        return None


# ═════════════════════════════════════════════
# 資料層：yfinance + Fugle 混合抓取
# ═════════════════════════════════════════════

@st.cache_data(ttl=3600)
def fetch_valuation_data(
    symbol: str,
    years: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    取得估值分析所需歷史資料。

    資料來源策略
    -----------
    - 每日收盤價    : yfinance history()（.TW / .TWO 自動識別）
    - 季度 EPS      : yfinance quarterly_income_stmt → TTM rolling 4Q
    - 季度 BVPS     : yfinance quarterly_balance_sheet（最新季前向填充）
    - 年度配息      : Fugle API dividends（台股）→ fallback yfinance dividends

    Returns
    -------
    None（抓取失敗）或 dict：
        symbol_full   : str
        price_df      : DataFrame (DatetimeIndex, col="close")
        eps_daily     : Series  — TTM EPS daily（或季度 fallback）
        bvps_daily    : Series  — 季度 BVPS daily
        div_annual    : Series  — 年度配息
        div_daily     : Series  — 年度配息前向填充至每日
        current_price : float
        current_eps   : float | None
        current_bvps  : float | None
        current_div   : float | None
        info          : dict
    """
    raw   = symbol.strip().upper()
    is_tw = bool(_TW_CODE_RE.match(raw))
    candidates = [f"{raw}.TW", f"{raw}.TWO"] if is_tw else [raw]

    # ── 1. 每日收盤價（yfinance）────────────────────────────────
    tk: Optional[yf.Ticker] = None
    price_df   = pd.DataFrame()
    info: Dict[str, Any] = {}
    resolved   = raw

    for sym in candidates:
        try:
            candidate = yf.Ticker(sym)
            hist = candidate.history(period=f"{years}y", auto_adjust=True)
            if hist.empty:
                continue
            tk       = candidate
            resolved = sym
            price_df = hist[["Close"]].rename(columns={"Close": "close"})
            price_df.index = pd.DatetimeIndex(price_df.index).tz_localize(None)
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            break
        except Exception:
            continue

    if price_df.empty:
        return None

    current_price = _to_float(price_df["close"].iloc[-1])
    if current_price is None:
        return None

    # ── 2. 季度 EPS → TTM（yfinance quarterly_income_stmt）─────
    eps_daily: pd.Series = pd.Series(dtype=float, index=price_df.index)

    if tk is not None:
        try:
            q_inc = getattr(tk, "quarterly_income_stmt", None) or \
                    getattr(tk, "quarterly_financials",  None)
            if isinstance(q_inc, pd.DataFrame) and not q_inc.empty:
                eps_row = _find_row(q_inc, ["Basic EPS", "Diluted EPS"])
                if eps_row is not None:
                    eps_daily = _compute_ttm_quarterly(_row_to_series(eps_row), price_df)
                else:
                    # Fallback：淨利 / 股數
                    ni_row = _find_row(q_inc, [
                        "Net Income", "Net Income Common Stockholders",
                    ])
                    shares = _to_float(info.get("sharesOutstanding"))
                    if ni_row is not None and shares and shares > 0:
                        q_eps = _row_to_series(ni_row) / shares
                        eps_daily = _compute_ttm_quarterly(q_eps, price_df)
        except Exception:
            pass

    # Fallback 2：年報 EPS（Basic/Diluted 或 Net Income / shares）
    if eps_daily.dropna().empty and tk is not None:
        try:
            a_inc = getattr(tk, "income_stmt", None) or \
                    getattr(tk, "financials",    None)
            if isinstance(a_inc, pd.DataFrame) and not a_inc.empty:
                eps_row = _find_row(a_inc, ["Basic EPS", "Diluted EPS"])
                if eps_row is not None:
                    eps_daily = _align_to_daily(_row_to_series(eps_row), price_df)
                else:
                    # 年報 Net Income / shares（台股常見格式）
                    ni_row = _find_row(a_inc, [
                        "Net Income", "Net Income Common Stockholders",
                    ])
                    shares_a = _to_float(info.get("sharesOutstanding"))
                    if ni_row is not None and shares_a and shares_a > 0:
                        eps_a = _row_to_series(ni_row) / shares_a
                        eps_daily = _align_to_daily(eps_a, price_df)
        except Exception:
            pass

    # Fallback 3：用 trailingEps 建立水平基準線（帶線為等距水平，仍可診斷當前估值）
    if eps_daily.dropna().empty:
        trailing = _to_float(info.get("trailingEps"))
        if trailing is not None and trailing > 0:
            eps_daily = pd.Series(trailing, index=price_df.index, dtype=float)

    current_eps = _to_float(info.get("trailingEps")) or (
        _to_float(eps_daily.dropna().iloc[-1]) if not eps_daily.dropna().empty else None
    )

    # ── 3. BVPS（季報 → 年報 → bookValue 水平線，三層 fallback）──
    bvps_daily: pd.Series = pd.Series(dtype=float, index=price_df.index)
    _EQ_ROWS = [
        "Stockholders Equity", "Total Stockholder Equity",
        "Common Stock Equity", "Total Equity Gross Minority Interest",
    ]

    if tk is not None:
        shares = _to_float(info.get("sharesOutstanding"))
        if shares and shares > 0:
            # Fallback 1：季報
            try:
                q_bs = getattr(tk, "quarterly_balance_sheet", None)
                if isinstance(q_bs, pd.DataFrame) and not q_bs.empty:
                    eq_row = _find_row(q_bs, _EQ_ROWS)
                    if eq_row is not None:
                        bvps_daily = _align_to_daily(
                            _row_to_series(eq_row) / shares, price_df
                        )
            except Exception:
                pass

            # Fallback 2：年報
            if bvps_daily.dropna().empty:
                try:
                    a_bs = getattr(tk, "balance_sheet", None)
                    if isinstance(a_bs, pd.DataFrame) and not a_bs.empty:
                        eq_row = _find_row(a_bs, _EQ_ROWS)
                        if eq_row is not None:
                            bvps_daily = _align_to_daily(
                                _row_to_series(eq_row) / shares, price_df
                            )
                except Exception:
                    pass

    # Fallback 3：bookValue 水平基準線
    if bvps_daily.dropna().empty:
        book_val = _to_float(info.get("bookValue"))
        if book_val is not None and book_val > 0:
            bvps_daily = pd.Series(book_val, index=price_df.index, dtype=float)

    current_bvps = _to_float(info.get("bookValue")) or (
        _to_float(bvps_daily.dropna().iloc[-1]) if not bvps_daily.dropna().empty else None
    )

    # ── 4. 年度配息（Fugle 優先，回退 yfinance）────────────────
    div_annual: pd.Series = pd.Series(dtype=float)

    if is_tw:
        fugle_divs = _fetch_dividends_fugle(raw)
        if fugle_divs is not None and not fugle_divs.empty:
            div_annual = fugle_divs

    if div_annual.empty and tk is not None:
        try:
            raw_div = tk.dividends
            if isinstance(raw_div, pd.Series) and not raw_div.empty:
                divs = raw_div.copy()
                divs.index = pd.DatetimeIndex(divs.index).tz_localize(None)
                yearly = divs.resample("YE").sum()
                div_annual = yearly[yearly > 0]
        except Exception:
            pass

    div_daily: pd.Series = pd.Series(dtype=float, index=price_df.index)
    if not div_annual.empty:
        div_daily = _align_to_daily(div_annual, price_df)

    current_div = (
        _to_float(div_annual.iloc[-1]) if not div_annual.empty else None
    )

    return {
        "symbol_full":  resolved,
        "price_df":     price_df,
        "eps_daily":    eps_daily,    # TTM EPS（或年度 fallback），已對齊每日
        "bvps_daily":   bvps_daily,   # 季度 BVPS，已對齊每日
        "div_annual":   div_annual,
        "div_daily":    div_daily,
        "current_price": current_price,
        "current_eps":  current_eps,
        "current_bvps": current_bvps,
        "current_div":  current_div,
        "info":         info,
    }


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

    if custom_levels is not None:
        pe_levels: List[float] = sorted(custom_levels)          # 強制升冪
    else:
        if len(ratio_series) < 30:
            return None
        hist_levels = _five_equal_levels(ratio_series)
        if hist_levels is None:
            return None
        pe_levels = hist_levels

    bands: Dict[str, pd.Series] = {}
    for i, lvl in enumerate(pe_levels):
        bands[_BAND_LABELS[i]] = (eps_daily * lvl).where(eps_daily > 0)
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

    if custom_levels is not None:
        pb_levels: List[float] = sorted(custom_levels)          # 強制升冪
    else:
        if len(ratio_series) < 30:
            return None
        hist_levels = _five_equal_levels(ratio_series)
        if hist_levels is None:
            return None
        pb_levels = hist_levels

    bands: Dict[str, pd.Series] = {}
    for i, lvl in enumerate(pb_levels):
        bands[_BAND_LABELS[i]] = (bvps_daily * lvl).where(bvps_daily > 0)
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
) -> Dict[str, str]:
    """
    診斷目前股價所在的估值區間。
    current_bands 必須是 5 個升冪排列的帶線價格。
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
        f"資料來源：Yahoo Finance (yfinance) ＋ Fugle API ｜ "
        f"現價：{current_price:,.2f}"
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
    suffix      = "自訂倍數" if is_custom else "TTM EPS｜5 等份估值帶"
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


def render_valuation_page() -> None:
    """估值分析頁面（Tab 8）。"""

    # session_state 保存查詢結果，避免圖表因互動消失
    if "val_cache" not in st.session_state:
        st.session_state["val_cache"] = None

    # ── 控制面板（expander）────────────────────────────────────
    with st.expander("🔍 查詢條件設定與操作", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            symbol = st.text_input(
                "股票代號",
                value="2330",
                max_chars=20,
                key="val_symbol",
                help=(
                    "台股輸入 4-6 位數字（如 2330），系統自動嘗試 .TW / .TWO。\n"
                    "美股輸入英文代號（如 AAPL）。"
                ),
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
            st.error("股票代號不得為空。")
            st.session_state["val_cache"] = None

        else:
            with st.spinner(f"正在取得 {symbol} 歷史資料（{years} 年）…"):
                try:
                    data = fetch_valuation_data(symbol=symbol, years=years)
                except Exception as e:
                    st.error(f"資料抓取失敗：{e}")
                    st.session_state["val_cache"] = None
                    data = None

            if data is None:
                st.warning(
                    f"查無 **{symbol}** 的歷史資料。\n\n"
                    "可能原因：代號錯誤、Yahoo Finance 尚未收錄此標的、"
                    "或目前網路連線不穩定。"
                )
                st.session_state["val_cache"] = None

            else:
                method_key = (method or "")[:1]
                _HINTS = {
                    "📊": (
                        f"**{data['symbol_full']}** 的本益比河流圖資料不足。\n\n"
                        "可能原因：\n"
                        "- Yahoo Finance 未提供此標的的 EPS 季報資料\n"
                        "- 近期出現虧損（EPS ≤ 0）\n\n"
                        "建議改用「淨值比河流圖」或「殖利率通道」。"
                    ),
                    "📚": (
                        f"**{data['symbol_full']}** 的淨值比河流圖資料不足。\n\n"
                        "可能原因：Yahoo Finance 未提供每股淨值季報資料。\n\n"
                        "建議改用「本益比」或確認代號是否正確。"
                    ),
                    "💰": (
                        f"**{data['symbol_full']}** 近期無配息記錄，"
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
                    eval_result = evaluate_current_price(
                        data["current_price"], band_data["current_bands"]
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
            "**三種估值方法說明**\n\n"
            "| 方法 | 適用標的 |\n"
            "|------|----------|\n"
            "| 本益比河流 (P/E) | 成長股、科技股、有穩定 EPS 者 |\n"
            "| 淨值比河流 (P/B) | 金融股、景氣循環股、資產股 |\n"
            "| 殖利率通道 | 高股息 ETF（0056、00878）、配息穩定存股標的 |\n\n"
            "**升級說明**\n\n"
            "- P/E 帶線採用 **TTM（近四季滾動 EPS）**，呈現平滑波浪而非階梯\n"
            "- 配息資料優先使用 **Fugle API**（台股），準確度更高\n"
            "- 圖表縮放、互動後結果**不會消失**（session_state 保存）\n\n"
            "**資料來源：** Yahoo Finance (yfinance) ＋ Fugle Market Data API"
        )
    elif cache is not None:
        _render_results(cache)
