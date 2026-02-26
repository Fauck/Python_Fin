"""
股票分析 Web 應用程式
技術架構：Streamlit + fugle-marketdata + Plotly
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import pandas_ta as ta  # noqa: F401
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
from fugle_marketdata import RestClient

# ─────────────────────────────────────────────
# 初始化：載入環境變數
# ─────────────────────────────────────────────
load_dotenv()


# ═════════════════════════════════════════════
# 資料層：API 呼叫邏輯（與 UI 完全解耦）
# ═════════════════════════════════════════════

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

    if date_to is None:
        date_to = datetime.today().strftime("%Y-%m-%d")
    if date_from is None:
        # 往前推 90 天，確保涵蓋足夠的交易日（含假期、休市）
        date_from = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

    raw = client.stock.historical.candles(
        **{
            "symbol": symbol,
            "from": date_from,
            "to": date_to,
            "fields": fields,
        }
    )

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

    return df.tail(limit).reset_index(drop=True)


# ═════════════════════════════════════════════
# 演算法層：各策略判斷函式（純邏輯，不含 Streamlit 元素）
#
# 所有策略函式共享相同簽名：
#   輸入：pd.DataFrame（含 open/high/low/close/volume/date，日期升冪）
#   輸出：dict（符合條件，含關鍵指標）或 None（不符合）
#
# 新增策略時，只需實作相同簽名的函式，並登記至 STRATEGY_REGISTRY 即可。
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# 策略一：盤整突破第一根
# ─────────────────────────────────────────────
# 參數調整說明：
#   consolidation_days (N)  預設 21 ↑增大→更長期盤整 ↓減小→短期盤整
#   amplitude_threshold (X) 預設 0.10 ↓減小→更嚴格（更緊密）
#   volume_ratio            預設 1.5 ↑增大→更強量能要求
#   check_volume            預設 True，False→僅判斷價格突破
# ─────────────────────────────────────────────

def check_consolidation_breakout(
    df: pd.DataFrame,
    consolidation_days: int = 21,
    amplitude_threshold: float = 0.10,
    volume_ratio: float = 1.5,
    check_volume: bool = True,
) -> Optional[Dict[str, Any]]:
    """判斷股票是否符合「盤整突破第一根」條件。"""
    required_cols = {"open", "high", "low", "close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < consolidation_days + 1:
        return None

    recent    = df.tail(consolidation_days).reset_index(drop=True)
    box       = recent.iloc[:-1]   # 前 N-1 天：定義盤整箱體
    today     = recent.iloc[-1]    # 最近交易日：突破候選日
    yesterday = recent.iloc[-2]    # 前一交易日：確認非第二根

    box_high = float(box["high"].max())
    box_low  = float(box["low"].min())

    # 盤整區間判定
    amplitude = (box_high - box_low) / box_low
    if amplitude >= amplitude_threshold:
        return None

    today_close     = float(today["close"])
    yesterday_close = float(yesterday["close"])
    today_volume    = float(today["volume"])
    avg_5d_volume   = float(box.tail(5)["volume"].mean())

    # 條件 A：今日收盤突破箱頂
    if today_close <= box_high:
        return None
    # 條件 B：昨日收盤未突破（確保是第一根）
    if yesterday_close > box_high:
        return None
    # 條件 C（可選）：帶量突破
    if check_volume and today_volume < avg_5d_volume * volume_ratio:
        return None

    return {
        "日期":       today["date"].strftime("%Y-%m-%d"),
        "收盤價":     round(today_close, 2),
        "箱頂":       round(box_high, 2),
        "箱底":       round(box_low, 2),
        "振幅(%)":    round(amplitude * 100, 2),
        "今日量":     int(today_volume),
        "5日均量":    int(avg_5d_volume),
        "量比":       round(today_volume / avg_5d_volume, 2) if avg_5d_volume > 0 else None,
    }


# ─────────────────────────────────────────────
# 策略二：均線多頭排列
# ─────────────────────────────────────────────
# 使用固定均線參數：5MA / 10MA / 20MA（約 1 個月）
# 條件：5MA > 10MA > 20MA，收盤 > 5MA，20MA 趨勢向上
# ─────────────────────────────────────────────

def check_bullish_ma_alignment(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """判斷股票是否符合「均線多頭排列」條件。"""
    required_cols = {"close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < 21:  # 計算 20MA 至少需要 20 筆，加 1 比較前後
        return None

    df = df.copy()
    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    if pd.isna(latest[["ma5", "ma10", "ma20"]]).any():
        return None

    ma5   = float(latest["ma5"])
    ma10  = float(latest["ma10"])
    ma20  = float(latest["ma20"])
    close = float(latest["close"])

    # 多頭排列：5MA > 10MA > 20MA
    if not (ma5 > ma10 > ma20):
        return None
    # 收盤在均線上方
    if close <= ma5:
        return None
    # 20MA 趨勢向上
    if ma20 <= float(prev["ma20"]):
        return None

    return {
        "日期":         latest["date"].strftime("%Y-%m-%d"),
        "收盤價":       round(close, 2),
        "5MA":          round(ma5, 2),
        "10MA":         round(ma10, 2),
        "20MA":         round(ma20, 2),
        "收vs5MA(%)":   round((close - ma5) / ma5 * 100, 2),
        "成交量":       int(latest["volume"]),
    }


# ─────────────────────────────────────────────
# 策略三：爆量長紅起漲
# ─────────────────────────────────────────────
# 參數調整說明：
#   volume_ratio 預設 2.0 ↑增大→要求更強爆量
#   body_pct     預設 0.03（3%）↑增大→要求更大紅K實體
# ─────────────────────────────────────────────

def check_volume_surge_bullish(
    df: pd.DataFrame,
    volume_ratio: float = 2.0,
    body_pct: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """判斷股票是否符合「爆量長紅起漲」條件。"""
    required_cols = {"open", "high", "low", "close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < 6:  # 需要前 5 日均量 + 今日
        return None

    today         = df.iloc[-1]
    past5         = df.iloc[-6:-1]  # 前 5 日（不含今日）

    today_close   = float(today["close"])
    today_open    = float(today["open"])
    today_volume  = float(today["volume"])
    avg_5d_volume = float(past5["volume"].mean())

    if avg_5d_volume <= 0:
        return None

    # 爆量：今日量 > 5日均量 × volume_ratio
    if today_volume < avg_5d_volume * volume_ratio:
        return None

    # 長紅：close > open 且實體漲幅 > body_pct
    body_ratio = (today_close - today_open) / today_open if today_open > 0 else 0
    if body_ratio <= body_pct:
        return None

    # 收高：收盤為近 5 日（含今日）最高收盤
    if today_close < float(df.tail(5)["close"].max()):
        return None

    return {
        "日期":        today["date"].strftime("%Y-%m-%d"),
        "收盤價":      round(today_close, 2),
        "K棒漲幅(%)":  round(body_ratio * 100, 2),
        "今日量":      int(today_volume),
        "5日均量":     int(avg_5d_volume),
        "量比":        round(today_volume / avg_5d_volume, 2),
    }


# ─────────────────────────────────────────────
# 策略四：乖離過大跌深反彈
# ─────────────────────────────────────────────
# 參數調整說明：
#   bias_threshold 預設 -0.10（-10%）↓減小→要求更深超跌
#   shadow_ratio   預設 0.30，下影線需 ≥ 實體 × 此比例
#                  ↑增大→要求更明顯下影線（更嚴格）
# ─────────────────────────────────────────────

def check_oversold_reversal(
    df: pd.DataFrame,
    bias_threshold: float = -0.10,
    shadow_ratio: float = 0.30,
) -> Optional[Dict[str, Any]]:
    """判斷股票是否符合「乖離過大跌深反彈」條件。"""
    required_cols = {"open", "high", "low", "close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < 21:  # 計算 20MA 需至少 20 筆
        return None

    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()

    today  = df.iloc[-1]
    ma20   = float(today["ma20"])
    close  = float(today["close"])
    open_  = float(today["open"])
    low    = float(today["low"])

    if pd.isna(ma20):
        return None

    # 負乖離過大：(close - MA20) / MA20 < bias_threshold
    bias = (close - ma20) / ma20
    if bias >= bias_threshold:
        return None

    # 紅 K（台灣：收盤 > 開盤 即為紅 K）
    if close <= open_:
        return None

    # 下影線判定：下影線 = min(open, close) - low
    # 條件：下影線 ≥ 紅 K 實體 × shadow_ratio
    body         = close - open_
    lower_shadow = min(close, open_) - low
    if body <= 0 or lower_shadow < body * shadow_ratio:
        return None

    return {
        "日期":         today["date"].strftime("%Y-%m-%d"),
        "收盤價":       round(close, 2),
        "月線(20MA)":   round(ma20, 2),
        "乖離率(%)":    round(bias * 100, 2),
        "下影線/實體":  round(lower_shadow / body, 2),
        "成交量":       int(today["volume"]),
    }


# ═════════════════════════════════════════════
# 通用批次掃描引擎
# ═════════════════════════════════════════════

def scan_watchlist(
    symbols: List[str],
    strategy_fn: Callable[[pd.DataFrame], Optional[Dict[str, Any]]],
    fetch_limit: int = 35,
    sleep_sec: float = 0.2,
    progress_callback: Optional[Callable[[float], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    通用批次掃描引擎。

    Parameters
    ----------
    symbols          : 股票代號清單
    strategy_fn      : 策略判斷函式（接受 DataFrame，回傳 dict 或 None）
    fetch_limit      : 每支股票拉取的最多 K 線筆數
    sleep_sec        : 每次 API 呼叫間隔（避免觸發 Fugle Rate Limit）
    progress_callback: 進度回呼（接受 0~1 的 float）
    status_callback  : 狀態文字回呼（接受字串）

    Returns
    -------
    (results, errors)  符合條件的清單 + 查詢異常清單
    """
    results: List[dict] = []
    errors:  List[dict] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if status_callback:
            status_callback(f"掃描中 [{i + 1}/{total}]：{symbol}")
        if progress_callback:
            progress_callback((i + 1) / total)

        try:
            df = fetch_stock_candles(symbol=symbol, limit=fetch_limit)
            if df.empty:
                errors.append({"代號": symbol, "原因": "查無資料"})
            else:
                hit = strategy_fn(df)
                if hit:
                    results.append({"代號": symbol, **hit})
        except Exception as e:
            errors.append({"代號": symbol, "原因": str(e)[:80]})

        time.sleep(sleep_sec)  # 控制 API 請求頻率

    return results, errors


# ═════════════════════════════════════════════
# 技術指標計算（演算法層，純邏輯）
# ═════════════════════════════════════════════

def compute_ma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """
    計算多期簡單移動平均線（SMA）。

    Parameters
    ----------
    df      : 含 close 欄位的 DataFrame
    periods : 要計算的天數清單，例如 [5, 10, 20]

    Returns
    -------
    含 ma5 / ma10 / ma20 等新欄位的 DataFrame 副本
    """
    df = df.copy()
    for p in periods:
        df[f"ma{p}"] = df["close"].rolling(p).mean()
    return df


def compute_kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """
    計算台灣市場標準 KD 指標（隨機指標）。

    公式：
      RSV = (Close - Lowest Low(N)) / (Highest High(N) - Lowest Low(N)) × 100
      K(t) = (2/3) × K(t-1) + (1/3) × RSV(t)   初始值 50
      D(t) = (2/3) × D(t-1) + (1/3) × K(t)      初始值 50

    Parameters
    ----------
    df     : 含 high / low / close 欄位的 DataFrame（日期升冪）
    period : RSV 計算週期，預設 9（台灣市場標準）

    Returns
    -------
    含 k_val / d_val 新欄位的 DataFrame 副本
    """
    df = df.copy()
    low_min  = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()

    denom = (high_max - low_min).replace(0, None)
    rsv   = ((df["close"] - low_min) / denom * 100).clip(0, 100).fillna(50)

    k_vals: List[float] = [50.0] * len(df)
    d_vals: List[float] = [50.0] * len(df)

    for i in range(1, len(df)):
        k_vals[i] = (2 / 3) * k_vals[i - 1] + (1 / 3) * float(rsv.iloc[i])
        d_vals[i] = (2 / 3) * d_vals[i - 1] + (1 / 3) * k_vals[i]

    df["k_val"] = [round(v, 2) for v in k_vals]
    df["d_val"] = [round(v, 2) for v in d_vals]
    return df


# ═════════════════════════════════════════════
# 評分模型：個股綜合買進評分（演算法層）
# ═════════════════════════════════════════════

# 資料抓取常數：往前 250 個日曆天（約 180 交易日），取最近 120 根 K 棒
_SCORE_FETCH_DAYS  = 250
_SCORE_FETCH_LIMIT = 120


def compute_score(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    計算個股綜合買進評分（100 分制）。

    評分維度
    --------
    趨勢  Trend       30 分  — 10MA / 20MA / 60MA 位置
    動能  Momentum    30 分  — RSI(14) + KD(9,3,3)
    震盪  Oscillator  20 分  — MACD(12,26,9) 柱狀圖 + 快慢線
    量能  Volume      20 分  — 今日量 vs 5 日均量

    Parameters
    ----------
    df : 含 open/high/low/close/volume/date 欄位的 DataFrame
         建議至少 65 個交易日（確保 60MA 有效）

    Returns
    -------
    dict  含 total / dimensions / details；資料不足回傳 None
    """
    if df.empty or len(df) < 65:
        return None

    df = df.copy().reset_index(drop=True)

    # ── MA ──────────────────────────────────────────
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # ── RSI(14)：pandas-ta 函式風格呼叫 ───────────
    # 回傳 Series，名稱為 "RSI_14"
    df["rsi14"] = ta.rsi(df["close"], length=14)

    # ── KD(9,3,3)：pandas-ta stoch ────────────────
    # 回傳 DataFrame，欄位 STOCHk_9_3_3 / STOCHd_9_3_3
    _stoch = ta.stoch(df["high"], df["low"], df["close"], k=9, d=3, smooth_k=3)
    if _stoch is not None and "STOCHk_9_3_3" in _stoch.columns:
        df["k_stoch"] = _stoch["STOCHk_9_3_3"].values
        df["d_stoch"] = _stoch["STOCHd_9_3_3"].values
    else:
        df["k_stoch"] = df["d_stoch"] = float("nan")

    # ── MACD(12,26,9)：pandas-ta 函式風格呼叫 ────
    # 回傳 DataFrame，欄位：
    #   MACD_12_26_9  → DIF（快線）
    #   MACDh_12_26_9 → 柱狀圖（DIF − DEA）
    #   MACDs_12_26_9 → DEA / 信號線（慢線）
    _macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if _macd is not None and "MACD_12_26_9" in _macd.columns:
        df["macd_dif"]  = _macd["MACD_12_26_9"].values
        df["macd_hist"] = _macd["MACDh_12_26_9"].values
        df["macd_dea"]  = _macd["MACDs_12_26_9"].values
    else:
        df["macd_dif"] = df["macd_hist"] = df["macd_dea"] = float("nan")

    # ── 取最後一根 K 棒的各指標值 ──────────────────
    last = df.iloc[-1]

    def _f(col: str) -> Optional[float]:
        v = last[col] if col in df.columns else None
        return None if (v is None or pd.isna(v)) else float(v)

    close     = _f("close")
    ma10      = _f("ma10")
    ma20      = _f("ma20")
    ma60      = _f("ma60")
    rsi       = _f("rsi14")
    k_        = _f("k_stoch")
    d_        = _f("d_stoch")
    macd_dif  = _f("macd_dif")
    macd_dea  = _f("macd_dea")
    macd_hist = _f("macd_hist")
    volume    = _f("volume") if "volume" in df.columns else None
    vol_5avg  = (
        float(df["volume"].iloc[-6:-1].mean())
        if "volume" in df.columns and len(df) >= 6 else None
    )

    # ── 維度一：趨勢 Trend（30 分）────────────────
    def _above(price: Optional[float], ma: Optional[float]) -> bool:
        return price is not None and ma is not None and price > ma

    t10 = 10 if _above(close, ma10) else 0
    t20 = 10 if _above(close, ma20) else 0
    t60 = 10 if _above(close, ma60) else 0
    trend_score = t10 + t20 + t60

    # ── 維度二：動能 Momentum（30 分）─────────────
    if rsi is not None:
        if 40 <= rsi <= 70:
            rsi_pts, rsi_st = 15, "健康多頭（40~70）"
        elif rsi < 30:
            rsi_pts, rsi_st = 15, "超賣反彈潛力（< 30）"
        elif rsi > 80:
            rsi_pts, rsi_st = 0,  "超買過熱（> 80）"
        else:
            rsi_pts, rsi_st = 5,  "中性偏弱（30~40 或 70~80）"
    else:
        rsi_pts, rsi_st = 0, "資料不足"

    if k_ is not None and d_ is not None:
        kd_pts, kd_st = (15, "K > D（黃金交叉）") if k_ > d_ else (0, "K ≤ D（死亡交叉）")
    else:
        kd_pts, kd_st = 0, "資料不足"

    momentum_score = rsi_pts + kd_pts

    # ── 維度三：震盪 Oscillator（20 分）───────────
    if macd_hist is not None:
        hist_pts, hist_st = (10, "柱狀 > 0（多頭動能）") if macd_hist > 0 else (0, "柱狀 ≤ 0（動能減弱）")
    else:
        hist_pts, hist_st = 0, "資料不足"

    if macd_dif is not None and macd_dea is not None:
        cross_pts, cross_st = (10, "DIF > DEA（多頭）") if macd_dif > macd_dea else (0, "DIF ≤ DEA（空頭）")
    else:
        cross_pts, cross_st = 0, "資料不足"

    oscillator_score = hist_pts + cross_pts

    # ── 維度四：量能 Volume（20 分）────────────────
    if volume is not None and vol_5avg is not None and vol_5avg > 0:
        vol_pts, vol_st = (20, "量能放大") if volume > vol_5avg else (0, "量能萎縮")
    else:
        vol_pts, vol_st = 0, "資料不足"

    volume_score = vol_pts
    total_score  = trend_score + momentum_score + oscillator_score + volume_score

    # ── 指標明細列表 ────────────────────────────────
    def _n(v: Optional[float], dec: int = 2) -> str:
        return f"{v:,.{dec}f}" if v is not None else "N/A"

    details: List[Dict[str, str]] = [
        # Trend
        {"維度": "趨勢 Trend",      "指標": "短線趨勢 (10MA)",
         "數值": f"收 {_n(close)} {'>' if t10 else '≤'} 10MA {_n(ma10)}",
         "判斷": "✅ 多頭" if t10 else "❌ 空頭",  "得分": f"{t10} / 10"},
        {"維度": "趨勢 Trend",      "指標": "中線趨勢 (20MA)",
         "數值": f"收 {_n(close)} {'>' if t20 else '≤'} 20MA {_n(ma20)}",
         "判斷": "✅ 多頭" if t20 else "❌ 空頭",  "得分": f"{t20} / 10"},
        {"維度": "趨勢 Trend",      "指標": "長線趨勢 (60MA)",
         "數值": f"收 {_n(close)} {'>' if t60 else '≤'} 60MA {_n(ma60)}",
         "判斷": "✅ 多頭" if t60 else "❌ 空頭",  "得分": f"{t60} / 10"},
        # Momentum
        {"維度": "動能 Momentum",   "指標": "RSI (14)",
         "數值": _n(rsi),           "判斷": rsi_st,   "得分": f"{rsi_pts} / 15"},
        {"維度": "動能 Momentum",   "指標": "KD (9,3,3)",
         "數值": f"K {_n(k_)}  D {_n(d_)}",  "判斷": kd_st,    "得分": f"{kd_pts} / 15"},
        # Oscillator
        {"維度": "震盪 Oscillator", "指標": "MACD 柱狀圖 (Hist)",
         "數值": _n(macd_hist),     "判斷": hist_st,  "得分": f"{hist_pts} / 10"},
        {"維度": "震盪 Oscillator", "指標": "MACD 快慢線 (DIF/DEA)",
         "數值": f"DIF {_n(macd_dif)}  DEA {_n(macd_dea)}",
         "判斷": cross_st,          "得分": f"{cross_pts} / 10"},
        # Volume
        {"維度": "量能 Volume",     "指標": "成交量 vs 5 日均量",
         "數值": f"今日 {_n(volume, 0)} 張  均 {_n(vol_5avg, 0)} 張",
         "判斷": vol_st,            "得分": f"{vol_pts} / 20"},
    ]

    return {
        "total": total_score,
        "dimensions": {
            "trend":      {"score": trend_score,      "max": 30, "label": "趨勢\nTrend"},
            "momentum":   {"score": momentum_score,   "max": 30, "label": "動能\nMomentum"},
            "oscillator": {"score": oscillator_score, "max": 20, "label": "震盪\nOscillator"},
            "volume":     {"score": volume_score,     "max": 20, "label": "量能\nVolume"},
        },
        "details": details,
    }


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


# ─────────────────────────────────────────────
# 選股頁面：各策略的 UI 設定區塊
# ─────────────────────────────────────────────

def _render_breakout_params() -> Tuple[Callable, int, str]:
    """盤整突破第一根：渲染參數控制項，回傳 (strategy_fn, fetch_limit, hint)。"""
    # ── 盤整天數 N ──  ↑增大→更長期盤整；↓減小→短期盤整
    consolidation_days = st.number_input(
        "盤整天數（N）", min_value=5, max_value=60, value=21, step=1,
        help="計算盤整區間使用的交易日天數，預設 21（約 1 個月）",
    )
    # ── 振幅門檻 X% ── ↓減小→更嚴格（更緊密）；↑增大→更寬鬆
    amplitude_pct = st.slider(
        "最大振幅（%）", min_value=1, max_value=30, value=10, step=1,
        help="盤整箱體的最大允許振幅，預設 10%",
    )
    st.markdown("---")
    check_volume = st.checkbox("啟用帶量突破（條件 C）", value=True)
    # ── 量比門檻 ── ↑增大→要求更強烈量能；↓減小→量能要求寬鬆
    volume_ratio = st.slider(
        "帶量倍數", min_value=1.0, max_value=5.0, value=1.5, step=0.1,
        disabled=not check_volume,
        help="今日成交量需大於近 5 日均量的幾倍，預設 1.5",
    )

    n      = int(consolidation_days)
    amp    = amplitude_pct / 100.0
    vr     = float(volume_ratio)
    chk    = check_volume

    vol_line = (
        f"- **條件 C**：今日量 > 近 5 日均量 × {vr:.1f} 倍（帶量突破）"
        if chk else "- 條件 C：已停用"
    )
    info = (
        f"- **盤整**：前 N-1 天振幅 (最高 − 最低) / 最低 < {amplitude_pct}%\n"
        "- **條件 A**：今日收盤 > 前 N-1 天最高價（突破箱頂）\n"
        "- **條件 B**：昨日收盤 ≤ 前 N-1 天最高價（確認是第一根）\n"
        + vol_line
    )

    return lambda df: check_consolidation_breakout(df, n, amp, vr, chk), n + 10, info


def _render_ma_alignment_params() -> Tuple[Callable, int, str]:
    """均線多頭排列：無額外參數，直接使用固定均線。"""
    st.caption("使用固定參數：5MA / 10MA / 20MA")
    info = (
        "- **5MA > 10MA > 20MA**（短中長多頭排列）\n"
        "- **收盤價 > 5MA**（維持強勢均線上方）\n"
        "- **20MA 趨勢向上**（今日 20MA > 昨日 20MA）"
    )
    return check_bullish_ma_alignment, 30, info


def _render_volume_surge_params() -> Tuple[Callable, int, str]:
    """爆量長紅起漲：渲染參數控制項。"""
    # ── 爆量倍數 ── ↑增大→要求更強爆量
    vol_ratio = st.slider(
        "爆量倍數", min_value=1.5, max_value=5.0, value=2.0, step=0.1,
        help="今日成交量需大於近 5 日均量的幾倍，預設 2.0",
    )
    # ── K 棒最小漲幅 ── ↑增大→要求更大紅K實體
    body_pct = st.slider(
        "K棒最小漲幅（%）", min_value=1, max_value=10, value=3, step=1,
        help="(收盤 - 開盤) / 開盤 的最小漲幅，預設 3%",
    )

    vr  = float(vol_ratio)
    bpct = body_pct / 100.0

    info = (
        f"- **爆量**：今日量 > 5 日均量 × {vr:.1f} 倍\n"
        f"- **長紅**：收盤 > 開盤，且 K 棒實體漲幅 > {body_pct}%\n"
        "- **收高**：今日收盤為近 5 日最高收盤價"
    )
    return lambda df: check_volume_surge_bullish(df, vr, bpct), 15, info


def _render_oversold_reversal_params() -> Tuple[Callable, int, str]:
    """乖離過大跌深反彈：渲染參數控制項。"""
    # ── 負乖離門檻 ── ↓減小→要求更深超跌
    bias_pct = st.slider(
        "最大負乖離（%）", min_value=-30, max_value=-5, value=-10, step=1,
        help="(收盤 - 20MA) / 20MA 低於此值才觸發，預設 -10%",
    )
    # ── 下影線比例 ── ↑增大→要求更明顯下影線
    shadow_ratio = st.slider(
        "下影線最小比例", min_value=0.1, max_value=1.5, value=0.3, step=0.05,
        help="下影線長度 ≥ K 棒實體 × 此比例，預設 0.30",
    )

    bpct = bias_pct / 100.0
    sr   = float(shadow_ratio)

    info = (
        f"- **超跌**：(收盤 - 20MA) / 20MA < {bias_pct}%\n"
        "- **紅 K**：今日收盤 > 開盤（止跌訊號）\n"
        f"- **下影線**：下影線長度 ≥ K棒實體 × {sr:.2f}（帶下影線的紅棒）"
    )
    return lambda df: check_oversold_reversal(df, bpct, sr), 30, info


# ─────────────────────────────────────────────
# 策略登記表（新增策略時擴充此處即可）
# ─────────────────────────────────────────────
STRATEGY_REGISTRY: Dict[str, Callable] = {
    "盤整突破第一根":    _render_breakout_params,
    "均線多頭排列":      _render_ma_alignment_params,
    "爆量長紅起漲":      _render_volume_surge_params,
    "乖離過大跌深反彈":  _render_oversold_reversal_params,
}

NO_RESULT_HINTS: Dict[str, str] = {
    "盤整突破第一根":    "可嘗試：放大振幅門檻、縮短盤整天數、或關閉帶量條件。",
    "均線多頭排列":      "可嘗試：確認觀察清單中有趨勢向上的股票，或待多頭排列形成後再掃描。",
    "爆量長紅起漲":      "可嘗試：降低爆量倍數或 K 棒漲幅門檻後重新掃描。",
    "乖離過大跌深反彈":  "可嘗試：將負乖離門檻放寬（例如 -8%）或降低下影線比例。",
}


def render_screener_page() -> None:
    """選股策略頁面（多策略版）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 選股策略")
        strategy = st.selectbox(
            "選擇策略",
            options=list(STRATEGY_REGISTRY.keys()),
            help="選擇要執行的選股策略",
        )

        st.markdown("---")
        st.markdown("#### 策略參數")

        # 依選擇的策略渲染對應參數，並取得策略函式
        render_params_fn = STRATEGY_REGISTRY[strategy]
        strategy_fn, fetch_limit, info_text = render_params_fn()

        st.markdown("---")
        scan_btn = st.button("開始掃描", type="primary", width="stretch")

    with result_col:
        st.markdown("#### 觀察清單")
        watchlist_input = st.text_area(
            "輸入股票代號（以逗號分隔）",
            value="2330, 1815, 2317, 2454, 3231",
            height=80,
            help="輸入欲掃描的股票代號，以逗號分隔。因 API 限制，建議清單勿超過 30 檔。",
        )

        st.info(f"**{strategy} 判定邏輯**\n\n{info_text}")

        if not scan_btn:
            return

        # 解析觀察清單
        symbols = [s.strip() for s in watchlist_input.split(",") if s.strip()]
        if not symbols:
            st.error("觀察清單為空，請至少輸入一個股票代號。")
            return

        # 批次掃描（含進度列）
        progress_bar = st.progress(0, text="準備掃描…")
        status_text  = st.empty()

        results, errors = scan_watchlist(
            symbols=symbols,
            strategy_fn=strategy_fn,
            fetch_limit=fetch_limit,
            sleep_sec=0.2,
            progress_callback=lambda p: progress_bar.progress(p),
            status_callback=lambda msg: status_text.text(msg),
        )

        progress_bar.empty()
        status_text.empty()

        # 結果展示
        st.markdown("---")
        st.subheader(f"掃描結果（共 {len(symbols)} 檔，符合 {len(results)} 檔）")

        if results:
            st.success(f"找到 **{len(results)}** 檔符合「{strategy}」的股票：")
            result_df = pd.DataFrame(results)
            # 對所有數值欄位格式化為小數點後兩位
            float_cols = result_df.select_dtypes(include="float").columns
            fmt = {col: "{:.2f}" for col in float_cols}
            st.dataframe(
                result_df.style.format(fmt, na_rep="—"),
                width="stretch",
                hide_index=True,
            )
        else:
            hint = NO_RESULT_HINTS.get(strategy, "請調整參數後重新掃描。")
            st.warning(f"本次掃描未找到符合「{strategy}」條件的股票。\n\n{hint}")

        if errors:
            with st.expander(f"查詢異常清單（{len(errors)} 檔）"):
                st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)


# ─────────────────────────────────────────────
# 評分頁面：雷達圖
# ─────────────────────────────────────────────

def render_radar_chart(score_result: Dict[str, Any]) -> None:
    """
    繪製四維度評分雷達圖（各維度正規化為 0~100%，方便視覺比較）。
    """
    dims       = score_result["dimensions"]
    dim_keys   = ["trend", "momentum", "oscillator", "volume"]
    labels     = [dims[k]["label"] for k in dim_keys]
    pcts       = [dims[k]["score"] / dims[k]["max"] * 100 for k in dim_keys]

    # 閉合多邊形
    r_vals     = pcts     + [pcts[0]]
    theta_vals = labels   + [labels[0]]

    total = score_result["total"]
    if total >= 80:
        fill_color, line_color = "rgba(76,175,80,0.20)", "#4CAF50"
    elif total >= 50:
        fill_color, line_color = "rgba(255,152,0,0.20)",  "#FF9800"
    else:
        fill_color, line_color = "rgba(244,67,54,0.20)",  "#F44336"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=theta_vals,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        marker=dict(size=7, color=line_color),
        hovertemplate="%{theta}<br>%{r:.0f}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=["25%", "50%", "75%", "100%"],
                tickfont=dict(size=9),
                gridcolor="#e0e0e0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                gridcolor="#e0e0e0",
            ),
        ),
        showlegend=False,
        height=340,
        margin=dict(l=60, r=60, t=20, b=20),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig)


# ─────────────────────────────────────────────
# 評分頁面主體
# ─────────────────────────────────────────────

def render_score_page() -> None:
    """個股綜合評分頁面（100 分制買進指標）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 查詢條件")
        symbol = st.text_input(
            "股票代號", value="2330", max_chars=10,
            key="score_page_symbol",
            help="輸入台灣股票代號，例如 2330（台積電）",
        ).strip()
        st.caption(
            f"抓取最近 {_SCORE_FETCH_LIMIT} 個交易日資料\n"
            "（確保季線 60MA 與 MACD 計算準確）"
        )
        query_btn = st.button("開始評分", type="primary", width="stretch")

    with result_col:
        if not query_btn:
            st.info("請在左側輸入股票代號後，點擊「開始評分」按鈕。")
            return

        if not symbol:
            st.error("股票代號不得為空，請重新輸入。")
            return

        date_from = (datetime.today() - timedelta(days=_SCORE_FETCH_DAYS)).strftime("%Y-%m-%d")

        with st.spinner(f"正在分析 {symbol}…"):
            try:
                df_full = fetch_stock_candles(
                    symbol=symbol,
                    limit=_SCORE_FETCH_LIMIT,
                    date_from=date_from,
                    fields="open,high,low,close,volume",
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

        score_result = compute_score(df_full)

        if score_result is None:
            st.warning(
                f"**{symbol}** 歷史資料不足（需至少 65 個交易日），無法進行評分。"
            )
            return

        total = score_result["total"]

        # ── 大字體總分（依分段著色）────────────────
        if total >= 80:
            score_color, score_label = "#4CAF50", "強烈建議關注"
        elif total >= 50:
            score_color, score_label = "#FF9800", "中性觀察"
        else:
            score_color, score_label = "#F44336", "偏弱勢"

        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {score_color}1A, {score_color}0A);
    border-left: 6px solid {score_color};
    border-radius: 8px;
    padding: 18px 28px;
    margin-bottom: 16px;
">
  <div style="color:{score_color}; font-size:12px; font-weight:600;
              text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;">
    {symbol} 綜合買進評分
  </div>
  <div style="color:{score_color}; font-size:54px; font-weight:700; line-height:1.1;">
    {total}
    <span style="font-size:22px; color:#888; font-weight:400;">/ 100</span>
  </div>
  <div style="color:{score_color}; font-size:16px; font-weight:500; margin-top:4px;">
    {score_label}
  </div>
</div>""", unsafe_allow_html=True)

        # ── 四維度分數卡片 ─────────────────────────
        dims = score_result["dimensions"]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("趨勢 Trend",      f"{dims['trend']['score']} / {dims['trend']['max']}")
        d2.metric("動能 Momentum",   f"{dims['momentum']['score']} / {dims['momentum']['max']}")
        d3.metric("震盪 Oscillator", f"{dims['oscillator']['score']} / {dims['oscillator']['max']}")
        d4.metric("量能 Volume",     f"{dims['volume']['score']} / {dims['volume']['max']}")

        st.markdown("---")

        # ── 雷達圖 + 指標明細並排 ──────────────────
        radar_col, table_col = st.columns([1, 1], gap="large")

        with radar_col:
            st.markdown("##### 四維度雷達圖")
            render_radar_chart(score_result)

        with table_col:
            st.markdown("##### 指標明細")
            detail_df = pd.DataFrame(score_result["details"])
            st.dataframe(detail_df, width="stretch", hide_index=True)


# ═════════════════════════════════════════════
# 進入點：Streamlit 主程式
# ═════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="台股分析儀表板",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 台股分析儀表板")
    st.caption("資料來源：Fugle Market Data API")

    tab_single, tab_screener, tab_score = st.tabs(
        ["📈 單股分析", "🔍 選股策略", "🎯 綜合評分"]
    )

    with tab_single:
        render_single_stock_page()

    with tab_screener:
        render_screener_page()

    with tab_score:
        render_score_page()


if __name__ == "__main__":
    main()
