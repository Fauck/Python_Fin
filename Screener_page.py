"""
選股策略頁面（Tab 2）。
包含：策略演算法層（check_*）+ 批次掃描引擎 + UI 渲染
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import compute_ma, fetch_stock_candles


# ─────────────────────────────────────────────
# 防禦模式：套用大盤濾網的突破型策略集合
# ─────────────────────────────────────────────
_DEFENSE_STRATEGIES = {"盤整突破第一根", "爆量長紅起漲", "52週高點突破", "布林擠壓突破"}

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
        "日期":       pd.Timestamp(today["date"]).strftime("%Y-%m-%d"),
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
        "日期":         pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
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
        "日期":        pd.Timestamp(today["date"]).strftime("%Y-%m-%d"),
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
        "日期":         pd.Timestamp(today["date"]).strftime("%Y-%m-%d"),
        "收盤價":       round(close, 2),
        "月線(20MA)":   round(ma20, 2),
        "乖離率(%)":    round(bias * 100, 2),
        "下影線/實體":  round(lower_shadow / body, 2),
        "成交量":       int(today["volume"]),
    }


# ─────────────────────────────────────────────
# 策略五：52 週高點突破
# ─────────────────────────────────────────────
# 邏輯：今日收盤 > 前 252 個交易日最高收盤，且帶量確認
# 意義：動能加速訊號——創近一年新高，代表多頭力道充沛
# 參數調整說明：
#   volume_ratio 預設 1.3 ↑增大→要求更強量能確認
# ─────────────────────────────────────────────

def check_52week_high_breakout(
    df: pd.DataFrame,
    volume_ratio: float = 1.3,
) -> Optional[Dict[str, Any]]:
    """判斷股票是否符合「52 週高點突破」條件。"""
    required_cols = {"open", "high", "low", "close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < 253:  # 需要 252 個交易日 + 今日
        return None

    today      = df.iloc[-1]
    prev_252   = df.iloc[-253:-1]           # 前 252 日（不含今日）

    close         = float(today["close"])
    high_252_cl   = float(prev_252["close"].max())  # 前 252 日最高收盤

    # 今日收盤需突破 52 週最高收盤
    if close <= high_252_cl:
        return None

    # 量能確認：今日量 > 近 5 日均量 × volume_ratio
    today_volume  = float(today["volume"])
    avg_5d_volume = float(df.iloc[-6:-1]["volume"].mean())
    if avg_5d_volume <= 0 or today_volume < avg_5d_volume * volume_ratio:
        return None

    return {
        "日期":        pd.Timestamp(today["date"]).strftime("%Y-%m-%d"),
        "收盤價":      round(close, 2),
        "52週高點":    round(high_252_cl, 2),
        "突破幅度(%)": round((close - high_252_cl) / high_252_cl * 100, 2),
        "今日量":      int(today_volume),
        "5日均量":     int(avg_5d_volume),
        "量比":        round(today_volume / avg_5d_volume, 2),
    }


# ─────────────────────────────────────────────
# 策略六：布林通道壓縮突破（升級版）
# ─────────────────────────────────────────────
# 邏輯：
#   壓縮判定 ── 過去 lookback_days 日平均帶寬 < bandwidth_threshold
#   突破條件 ── 今日收盤 > 布林上軌
#   量能確認 ── 今日量 > 近 5 日均量 × volume_ratio
# 參數調整說明：
#   period             預設 20    布林通道計算週期（月線）
#   std_dev            預設 2.0   標準差倍數（上下軌距）
#   bandwidth_threshold 預設 0.10  5日均帶寬低於此值視為壓縮（10%，參數可調）
#   lookback_days      預設 5     計算平均帶寬的回看天數
#   volume_ratio       預設 1.5   突破當日需放量倍數
# ─────────────────────────────────────────────

def check_bollinger_squeeze_breakout(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    bandwidth_threshold: float = 0.10,
    lookback_days: int = 5,
    volume_ratio: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """
    判斷股票是否符合「布林通道壓縮突破」條件（升級版）。

    壓縮定義
    --------
    過去 lookback_days 個交易日的平均帶寬（(上軌-下軌)/中軌）
    低於 bandwidth_threshold（預設 10%），代表籌碼極度壓縮。

    高勝率買點條件（三者同時成立）
    --------------------------------
    A. 壓縮：lookback_days 日平均帶寬 < bandwidth_threshold
    B. 突破：今日收盤 > 布林上軌（std_dev σ）
    C. 量能：今日成交量 > 近 5 日均量 × volume_ratio
    """
    required_cols = {"open", "high", "low", "close", "volume", "date"}
    if not required_cols.issubset(df.columns):
        return None
    if len(df) < period + lookback_days + 5:
        return None

    df = df.copy()
    rolling         = df["close"].rolling(period)
    df["_bb_mid"]   = rolling.mean()
    rolling_std     = rolling.std()
    df["_bb_upper"] = df["_bb_mid"] + std_dev * rolling_std
    df["_bb_lower"] = df["_bb_mid"] - std_dev * rolling_std
    df["_bb_width"] = (df["_bb_upper"] - df["_bb_lower"]) / df["_bb_mid"]

    today = df.iloc[-1]

    if pd.isna(today["_bb_upper"]) or pd.isna(today["_bb_width"]):
        return None

    close    = float(today["close"])
    bb_upper = float(today["_bb_upper"])
    bb_lower = float(today["_bb_lower"])
    bb_mid   = float(today["_bb_mid"])

    # ── 條件 B：突破布林上軌 ──
    if close <= bb_upper:
        return None

    # ── 條件 A：過去 lookback_days 日平均帶寬 < threshold ──
    valid_widths = df["_bb_width"].dropna()
    if len(valid_widths) < lookback_days:
        return None
    avg_bandwidth = float(valid_widths.iloc[-lookback_days:].mean())
    if avg_bandwidth >= bandwidth_threshold:
        return None

    # ── 條件 C：量能確認 ──
    today_volume  = float(today["volume"])
    avg_5d_volume = float(df.iloc[-6:-1]["volume"].mean())
    if avg_5d_volume <= 0 or today_volume < avg_5d_volume * volume_ratio:
        return None

    return {
        "日期":              pd.Timestamp(today["date"]).strftime("%Y-%m-%d"),
        "收盤價":            round(close, 2),
        "布林上軌":          round(bb_upper, 2),
        "布林中軌":          round(bb_mid, 2),
        "布林下軌":          round(bb_lower, 2),
        "突破幅度(%)":       round((close - bb_upper) / bb_upper * 100, 2),
        f"{lookback_days}日均帶寬": round(avg_bandwidth, 4),
        "壓縮閾值":          bandwidth_threshold,
        "今日量":            int(today_volume),
        "5日均量":           int(avg_5d_volume),
        "量比":              round(today_volume / avg_5d_volume, 2),
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
# 大盤環境濾網（Market Environment Filter）
# ═════════════════════════════════════════════

@st.cache_data(ttl=900)
def get_market_trend() -> dict:
    """
    抓取加權指數（IX0001）K 線，判斷目前大盤多空環境。

    Returns
    -------
    dict with keys:
        status : "bullish" | "bearish" | "unknown"
        close  : float | None   最新收盤點位
        ma20   : float | None   20 日均線
        ma60   : float | None   60 日均線
    """
    try:
        df = fetch_stock_candles(symbol="IX0001", limit=100)
        if df.empty or len(df) < 60:
            return {"status": "unknown", "close": None, "ma20": None, "ma60": None}

        df = compute_ma(df, [20, 60])
        latest = df.iloc[-1]

        close = float(latest["close"])
        ma20  = latest.get("ma20")
        ma60  = latest.get("ma60")

        if ma20 is None or ma60 is None or pd.isna(ma20) or pd.isna(ma60):
            return {"status": "unknown", "close": close, "ma20": None, "ma60": None}

        ma20 = float(ma20)
        ma60 = float(ma60)
        is_bullish = close > ma20 and close > ma60

        return {
            "status": "bullish" if is_bullish else "bearish",
            "close": round(close, 2),
            "ma20":  round(ma20, 2),
            "ma60":  round(ma60, 2),
        }
    except Exception:
        return {"status": "unknown", "close": None, "ma20": None, "ma60": None}


# ─────────────────────────────────────────────
# 選股頁面：各策略的 UI 設定區塊
# ─────────────────────────────────────────────

def _render_breakout_params(defense_boost: float = 0.0) -> Tuple[Callable, int, str, Optional[float]]:
    """盤整突破第一根：渲染參數控制項，回傳 (strategy_fn, fetch_limit, hint, effective_vol_ratio)。"""
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

    n   = int(consolidation_days)
    amp = amplitude_pct / 100.0
    vr  = float(volume_ratio)
    chk = check_volume

    effective_vr = round(max(vr + defense_boost, 2.5), 1) if (defense_boost > 0 and chk) else vr
    if defense_boost > 0 and chk:
        st.caption(f"防禦模式：量能門檻已自動提升至 **{effective_vr:.1f}** 倍（原設定 {vr:.1f}）")

    vol_line = (
        f"- **條件 C**：今日量 > 近 5 日均量 × {effective_vr:.1f} 倍（帶量突破）"
        if chk else "- 條件 C：已停用"
    )
    info = (
        f"- **盤整**：前 N-1 天振幅 (最高 − 最低) / 最低 < {amplitude_pct}%\n"
        "- **條件 A**：今日收盤 > 前 N-1 天最高價（突破箱頂）\n"
        "- **條件 B**：昨日收盤 ≤ 前 N-1 天最高價（確認是第一根）\n"
        + vol_line
    )

    return (
        lambda df: check_consolidation_breakout(df, n, amp, effective_vr, chk),
        n + 10,
        info,
        effective_vr if chk else None,
    )


def _render_ma_alignment_params() -> Tuple[Callable, int, str, Optional[float]]:
    """均線多頭排列：無額外參數，直接使用固定均線。"""
    st.caption("使用固定參數：5MA / 10MA / 20MA")
    info = (
        "- **5MA > 10MA > 20MA**（短中長多頭排列）\n"
        "- **收盤價 > 5MA**（維持強勢均線上方）\n"
        "- **20MA 趨勢向上**（今日 20MA > 昨日 20MA）"
    )
    return check_bullish_ma_alignment, 30, info, None


def _render_volume_surge_params(defense_boost: float = 0.0) -> Tuple[Callable, int, str, Optional[float]]:
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

    vr   = float(vol_ratio)
    bpct = body_pct / 100.0
    effective_vr = round(max(vr + defense_boost, 2.5), 1) if defense_boost > 0 else vr

    if defense_boost > 0:
        st.caption(f"防禦模式：量能門檻已自動提升至 **{effective_vr:.1f}** 倍（原設定 {vr:.1f}）")

    info = (
        f"- **爆量**：今日量 > 5 日均量 × {effective_vr:.1f} 倍\n"
        f"- **長紅**：收盤 > 開盤，且 K 棒實體漲幅 > {body_pct}%\n"
        "- **收高**：今日收盤為近 5 日最高收盤價"
    )
    return lambda df: check_volume_surge_bullish(df, effective_vr, bpct), 15, info, effective_vr


def _render_oversold_reversal_params() -> Tuple[Callable, int, str, Optional[float]]:
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
    return lambda df: check_oversold_reversal(df, bpct, sr), 30, info, None


def _render_52week_params(defense_boost: float = 0.0) -> Tuple[Callable, int, str, Optional[float]]:
    """52 週高點突破：渲染參數控制項。"""
    vol_ratio = st.slider(
        "量能確認倍數", min_value=1.0, max_value=3.0, value=1.3, step=0.1,
        help="今日成交量需大於近 5 日均量的幾倍，預設 1.3",
        key="sc_52w_vol",
    )
    vr = float(vol_ratio)
    effective_vr = round(max(vr + defense_boost, 2.5), 1) if defense_boost > 0 else vr

    if defense_boost > 0:
        st.caption(f"防禦模式：量能門檻已自動提升至 **{effective_vr:.1f}** 倍（原設定 {vr:.1f}）")

    info = (
        "- **52週高點**：今日收盤 > 前 252 個交易日最高收盤價\n"
        f"- **量能確認**：今日量 > 5 日均量 × {effective_vr:.1f} 倍\n"
        "- 動能加速訊號，配合趨勢順勢使用效果更佳"
    )
    return lambda df: check_52week_high_breakout(df, effective_vr), 270, info, effective_vr


def _render_bb_squeeze_params(defense_boost: float = 0.0) -> Tuple[Callable, int, str, Optional[float]]:
    """布林通道壓縮突破（升級版）：渲染參數控制項。"""
    period = st.number_input(
        "布林週期", min_value=10, max_value=60, value=20, step=1,
        help="計算布林通道使用的天數，預設 20（月線）",
        key="sc_bb_period",
    )
    std_dev = st.select_slider(
        "標準差倍數（σ）", options=[1.5, 2.0, 2.5, 3.0], value=2.0,
        help="布林通道寬度（標準差倍數），預設 2.0",
        key="sc_bb_std",
    )
    st.markdown("---")
    st.caption("壓縮濾網設定")
    bw_threshold = st.slider(
        "帶寬壓縮門檻（%）", min_value=5, max_value=25, value=10, step=1,
        help="過去 N 日平均帶寬低於此值視為壓縮，預設 10%（籌碼極度集中）",
        key="sc_bb_bw_thr",
    ) / 100.0
    lookback = st.number_input(
        "壓縮回看天數", min_value=3, max_value=10, value=5, step=1,
        help="計算平均帶寬的回看天數，預設 5 日",
        key="sc_bb_lookback",
    )
    vol_ratio = st.slider(
        "突破量能倍數", min_value=1.0, max_value=3.0, value=1.5, step=0.1,
        help="突破上軌當日成交量需 > 近 5 日均量 × 此倍數",
        key="sc_bb_vol",
    )

    p   = int(period)
    sd  = float(std_dev)
    bwt = float(bw_threshold)
    lb  = int(lookback)
    vr  = float(vol_ratio)
    effective_vr = round(max(vr + defense_boost, 2.5), 1) if defense_boost > 0 else vr

    if defense_boost > 0:
        st.caption(f"防禦模式：量能門檻已自動提升至 **{effective_vr:.1f}** 倍（原設定 {vr:.1f}）")

    info = (
        f"- **壓縮判定**：過去 {lb} 日平均帶寬 < {bwt*100:.0f}%（籌碼極度壓縮）\n"
        f"- **突破上軌**：今日收盤 > 布林上軌（{sd:.1f}σ，週期 {p}）\n"
        f"- **量能確認**：今日量 > 5 日均量 × {effective_vr:.1f} 倍\n"
        "- 三條件同時成立，為壓縮後爆發型高勝率買點"
    )
    return lambda df: check_bollinger_squeeze_breakout(df, p, sd, bwt, lb, effective_vr), p + 50, info, effective_vr


# ─────────────────────────────────────────────
# 策略登記表（新增策略時擴充此處即可）
# ─────────────────────────────────────────────
STRATEGY_REGISTRY: Dict[str, Callable] = {
    "盤整突破第一根":    _render_breakout_params,
    "均線多頭排列":      _render_ma_alignment_params,
    "爆量長紅起漲":      _render_volume_surge_params,
    "乖離過大跌深反彈":  _render_oversold_reversal_params,
    "52週高點突破":      _render_52week_params,
    "布林擠壓突破":      _render_bb_squeeze_params,
}

NO_RESULT_HINTS: Dict[str, str] = {
    "盤整突破第一根":    "可嘗試：放大振幅門檻、縮短盤整天數、或關閉帶量條件。",
    "均線多頭排列":      "可嘗試：確認觀察清單中有趨勢向上的股票，或待多頭排列形成後再掃描。",
    "爆量長紅起漲":      "可嘗試：降低爆量倍數或 K 棒漲幅門檻後重新掃描。",
    "乖離過大跌深反彈":  "可嘗試：將負乖離門檻放寬（例如 -8%）或降低下影線比例。",
    "52週高點突破":      "可嘗試：降低量能確認倍數，或確認觀察清單標的歷史資料足夠（需 253 個交易日以上）。",
    "布林擠壓突破":      "可嘗試：放寬帶寬壓縮門檻（例如 15%）、縮短回看天數或降低量能倍數後重新掃描。",
}


# ═════════════════════════════════════════════
# 展示層：頁面渲染函式
# ═════════════════════════════════════════════

def render_screener_page() -> None:
    """選股策略頁面（多策略版）。"""

    # ── 1. 大盤環境濾網 ─────────────────────────────────────────
    market = get_market_trend()
    if market["status"] == "bullish":
        close_str = f"{market['close']:,.2f}" if market["close"] is not None else "—"
        ma20_str  = f"{market['ma20']:,.2f}"  if market["ma20"]  is not None else "—"
        ma60_str  = f"{market['ma60']:,.2f}"  if market["ma60"]  is not None else "—"
        st.info(
            f"🟢 **大盤多頭格局**：加權指數站上月線與季線，突破策略維持標準參數。"
            f"  （指數 {close_str}｜20MA {ma20_str}｜60MA {ma60_str}）"
        )
        defense_boost = 0.0
    elif market["status"] == "bearish":
        close_str = f"{market['close']:,.2f}" if market["close"] is not None else "—"
        ma20_str  = f"{market['ma20']:,.2f}"  if market["ma20"]  is not None else "—"
        ma60_str  = f"{market['ma60']:,.2f}"  if market["ma60"]  is not None else "—"
        st.warning(
            f"🔴 **大盤空頭/震盪格局**：加權指數跌破重要均線，已自動啟動【防禦模式】，"
            f"提高突破策略的量能門檻防範假突破。"
            f"  （指數 {close_str}｜20MA {ma20_str}｜60MA {ma60_str}）"
        )
        defense_boost = 1.0
    else:
        st.info("⚪ 大盤狀態無法判斷（IX0001 資料抓取失敗），策略參數維持使用者設定。")
        defense_boost = 0.0

    # ── 2. 查詢條件設定（折疊控制區）────────────────────────────
    with st.expander("🔍 查詢條件設定", expanded=True):
        strategy = st.selectbox(
            "選擇策略",
            options=list(STRATEGY_REGISTRY.keys()),
            help="選擇要執行的選股策略",
        )

        st.markdown("---")
        st.markdown("##### 策略參數")

        render_params_fn = STRATEGY_REGISTRY[strategy]
        # 防禦型突破策略支援 defense_boost；反彈型策略無量能門檻，直接呼叫
        if strategy in _DEFENSE_STRATEGIES:
            strategy_fn, fetch_limit, info_text, effective_vr = render_params_fn(
                defense_boost=defense_boost
            )
        else:
            strategy_fn, fetch_limit, info_text, effective_vr = render_params_fn()

        st.markdown("---")
        st.markdown("##### 觀察清單")
        watchlist_input = st.text_area(
            "輸入股票代號（以逗號分隔）",
            value="2330, 1815, 2317, 2454, 3231",
            height=80,
            help="輸入欲掃描的股票代號，以逗號分隔。因 API 限制，建議清單勿超過 30 檔。",
        )

        scan_btn = st.button("開始掃描", type="primary", use_container_width=True)

    # ── 3. 策略說明 ──────────────────────────────────────────────
    st.info(f"**{strategy} 判定邏輯**\n\n{info_text}")

    if not scan_btn:
        return

    # ── 4. 防禦模式 Toast 提示 ───────────────────────────────────
    if defense_boost > 0 and strategy in _DEFENSE_STRATEGIES and effective_vr is not None:
        st.toast(
            f"防禦模式已啟動，量能門檻已提升至 {effective_vr:.1f} 倍",
            icon="⚠️",
        )

    # ── 5. 解析觀察清單 ──────────────────────────────────────────
    symbols = [s.strip() for s in watchlist_input.split(",") if s.strip()]
    if not symbols:
        st.error("觀察清單為空，請至少輸入一個股票代號。")
        return

    # ── 6. 批次掃描（含進度列）──────────────────────────────────
    progress_bar = st.progress(0, text="準備掃描…")
    status_text  = st.empty()

    def _on_progress(p: float) -> None:
        progress_bar.progress(p)

    def _on_status(msg: str) -> None:
        status_text.text(msg)

    results, errors = scan_watchlist(
        symbols=symbols,
        strategy_fn=strategy_fn,
        fetch_limit=fetch_limit,
        sleep_sec=0.2,
        progress_callback=_on_progress,
        status_callback=_on_status,
    )

    progress_bar.empty()
    status_text.empty()

    # ── 7. 結果展示 ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"掃描結果（共 {len(symbols)} 檔，符合 {len(results)} 檔）")

    if results:
        st.success(f"找到 **{len(results)}** 檔符合「{strategy}」的股票：")
        result_df = pd.DataFrame(results)
        float_cols = result_df.select_dtypes(include="float").columns
        fmt: Dict[str, Any] = {col: "{:.2f}" for col in float_cols}
        st.dataframe(
            result_df.style.format(fmt, na_rep="—"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        hint = NO_RESULT_HINTS.get(strategy, "請調整參數後重新掃描。")
        st.warning(f"本次掃描未找到符合「{strategy}」條件的股票。\n\n{hint}")

    if errors:
        with st.expander(f"查詢異常清單（{len(errors)} 檔）"):
            st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
