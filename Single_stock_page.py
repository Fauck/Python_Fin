"""
單股分析頁面（Tab 1）。66666test99999NEWWWW66666677777
渲染函式：render_data_table / render_close_chart / render_candlestick_chart
          render_ohlcv_chart / render_single_stock_page
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    fetch_stock_candles,
    compute_ma, compute_kd,
    compute_bollinger, compute_rsi, compute_macd,
    detect_all_candlestick_patterns,
)


# ═════════════════════════════════════════════
# 演算法層：均線扣抵值計算（純邏輯，不含 Streamlit 元素）
# ═════════════════════════════════════════════

def _deduction_trend(bias: float) -> Tuple[str, str]:
    """
    依乖離率（比例，非百分比）回傳趨勢標籤與顏色。

    Parameters
    ----------
    bias : (current_close - deduction_price) / deduction_price，例如 0.02 = 2%

    Returns
    -------
    (趨勢文字含 Emoji, 16 進位色碼)
    """
    if abs(bias) <= 0.01:          # |乖離| ≤ 1% → 盤整
        return "🟰 盤整轉折", "#FF9800"
    elif bias > 0.01:              # 現價 > 扣抵價 → 均線傾向上揚（台灣：紅漲）
        return "📈 易漲支撐", "#EF5350"
    else:                          # 現價 < 扣抵價 → 均線傾向下彎（台灣：綠跌）
        return "📉 易跌壓力", "#26A69A"


def calculate_deduction_values(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[List[Dict[str, Any]]]]:
    """
    計算 5MA / 10MA / 20MA / 60MA 的扣抵值與趨勢預判。

    扣抵值定義
    ----------
    N 日均線「明日扣抵價」= df.iloc[-N]['close']
    即明天計算均線時，最舊那一筆將被移出的收盤價。

    歷史欄位（使用 shift(N)，每個 row 記錄「N 個交易日前的收盤價」）
    ----------------------------------------------------------------
    deduction_N  : df['close'].shift(N)                 — 歷史扣抵價（供回測/畫圖）
    bias_pct_N   : (close - deduction_N) / deduction_N × 100  — 乖離率（百分比）
    trend_N      : 依乖離率判斷的趨勢標籤字串              — 供圖表顯示

    趨勢預判邏輯（嚴格遵守台股習慣：紅漲綠跌）
    ----------------------------------------
    |乖離| ≤ 1%  → 🟰 盤整轉折（橙）
    乖離 > +1%   → 📈 易漲支撐（紅）
    乖離 < -1%   → 📉 易跌壓力（綠）

    Parameters
    ----------
    df            : 含 close 欄位的 DataFrame（日期升冪），需至少 5 筆
    display_limit : 使用者選擇的顯示天數；只顯示 period ≤ display_limit 的均線

    Returns
    -------
    Tuple[pd.DataFrame, Optional[List[Dict[str, Any]]]]
        [0] 含 ma_N / deduction_N / bias_pct_N / trend_N 欄位的完整 DataFrame
            （供歷史回測與畫圖使用）
        [1] 最新交易日彙整 List[dict]（每條均線一筆），直接餵給 st.dataframe；
            無符合條件的均線時回傳 None
    """
    # 可計算的均線設定：(週期, 標籤, 副標題)
    ALL_CONFIGS: List[Tuple[int, str, str]] = [
        (5,  "5MA",  "周線"),
        (10, "10MA", "雙周線"),
        (20, "20MA", "月線"),
        (60, "60MA", "季線"),
    ]

    df = df.copy().reset_index(drop=True)

    # 篩選：只要資料筆數足夠計算即納入，不受前端顯示天數限制
    MA_CONFIGS: List[Tuple[int, str, str]] = [
        cfg for cfg in ALL_CONFIGS
        if len(df) >= cfg[0]
    ]

    if df.empty or not MA_CONFIGS:
        return df, None

    # ── 歷史欄位：整欄計算（供回測 / 畫圖）────────────────────
    for period, _, _ in MA_CONFIGS:
        # N 日均線值
        df[f"ma{period}"] = df["close"].rolling(period).mean()

        # deduction_N：第 i 行的扣抵價 = 第 i-N 行的收盤（shift(N)）
        df[f"deduction_{period}"] = df["close"].shift(period)

        # bias_pct_N：乖離率（百分比），NaN 發生於前 N 行資料不足處
        df[f"bias_pct_{period}"] = (
            (df["close"] - df[f"deduction_{period}"])
            / df[f"deduction_{period}"]
            * 100
        )

        # trend_N：趨勢標籤字串（NaN 行填 "—"）
        df[f"trend_{period}"] = df[f"bias_pct_{period}"].apply(
            lambda b: _deduction_trend(b / 100)[0] if pd.notna(b) else "—"
        )

    # ── 最新交易日彙整（供 Streamlit 卡片 / 表格顯示）──────────
    current_close = float(df["close"].iloc[-1])
    summary: List[Dict[str, Any]] = []

    for period, ma_name, subtitle in MA_CONFIGS:
        ma_val = df[f"ma{period}"].iloc[-1]
        if pd.isna(ma_val):
            continue

        # 最新扣抵價：df.iloc[-N]['close']（明日 MA 將移出的最舊一筆）
        deduction_price = float(df["close"].iloc[-period])
        bias            = (current_close - deduction_price) / deduction_price
        trend, color    = _deduction_trend(bias)

        summary.append({
            "period":          period,
            "ma_name":         ma_name,
            "subtitle":        subtitle,
            "ma_val":          round(float(ma_val), 2),
            "current_close":   round(current_close, 2),
            "deduction_price": round(deduction_price, 2),
            "diff_pct":        round(bias * 100, 2),   # 轉為百分比顯示
            "trend":           trend,
            "trend_color":     color,
        })

    return df, (summary if summary else None)


# ═════════════════════════════════════════════
# 演算法層：進場訊號評分
# ═════════════════════════════════════════════

def _sf(val) -> Optional[float]:
    """安全轉型：無法取得或 NaN 時回傳 None。"""
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def analyze_entry_signal(df_full: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    分析最新交易日進場適合度（技術面多維評分）。

    評分維度與權重
    ──────────────
    趨勢 (50 分)：站上季線 MA60 (10)、站上月線 MA20 (15)、均線多頭排列 (15)、
                  站上 MA5 (5)、MA20 方向向上 (5)
    動能 (35 分)：RSI(14) 狀態 (15)、MACD 柱狀正 (10)、KD K>D (10)
    波動 (20 分)：布林通道位置 (10)、BB 壓縮突破 (10)
    量能 (10 分)：成交量確認 (10)
    守門機制    ：收盤 < MA60 時，「強力進場」強制降為「偏多觀察」

    Returns
    -------
    None（資料不足）或 dict {
        signal:      str            —「強力進場」/「偏多觀察」/「偏空觀望」/「謹慎避開」
        color:       str            — CSS 色碼
        score_pct:   float          — 得分百分比 (0~100)
        total_pts:   int
        max_pts:     int
        checks:      List[Dict]     — 每項明細
        close:       float
        date:        str
        gate_reason: str | None     — 守門觸發說明（None 表示未觸發）
    }
    """
    if df_full is None or len(df_full) < 30:
        return None

    df = df_full.copy()

    # 確保所有指標已計算（不重複計算已存在的欄位）
    if "ma5" not in df.columns:
        df = compute_ma(df, [5, 10, 20, 60])
    if "k_val" not in df.columns:
        df = compute_kd(df)
    if "rsi_14" not in df.columns:
        df = compute_rsi(df)
    if "macd_hist" not in df.columns:
        df = compute_macd(df)
    if "bb_mid" not in df.columns:
        df = compute_bollinger(df)

    r     = df.iloc[-1]
    close = _sf(r.get("close"))
    if close is None:
        return None

    checks: List[Dict[str, Any]] = []
    total_pts = 0
    max_pts   = 0

    def add(cat: str, name: str, ok: bool, pts: int, max_p: int, detail: str) -> None:
        nonlocal total_pts, max_pts
        checks.append({"cat": cat, "name": name, "ok": ok,
                        "pts": pts, "max_pts": max_p, "detail": detail})
        total_pts += pts
        max_pts   += max_p

    # ── 趨勢 ─────────────────────────────────
    ma20 = _sf(r.get("ma20"))
    ma10 = _sf(r.get("ma10"))
    ma5  = _sf(r.get("ma5"))

    if ma20 is not None:
        ok = close > ma20
        add("📐 趨勢", "收盤站上月線 MA20",
            ok, 15 if ok else 0, 15,
            f"收盤 {close:.2f} {'> ↑ 多頭' if ok else '< ↓ 空頭'} MA20 {ma20:.2f}")

    if all(x is not None for x in [ma5, ma10, ma20]):
        ok = (ma5 > ma10 > ma20)  # type: ignore[operator]
        add("📐 趨勢", "均線多頭排列 (MA5>MA10>MA20)",
            ok, 15 if ok else 0, 15,
            f"MA5 {ma5:.1f} {'>' if ma5>ma10 else '≤'} MA10 {ma10:.1f} {'>' if ma10>ma20 else '≤'} MA20 {ma20:.1f}")  # type: ignore[operator]

    if ma5 is not None:
        ok = close > ma5
        add("📐 趨勢", "收盤站上短均線 MA5",
            ok, 5 if ok else 0, 5,
            f"收盤 {close:.2f} {'> MA5' if ok else '< MA5'} {ma5:.2f}")

    # 季線確認（MA60）——「大環境」最重要的趨勢過濾器
    ma60 = _sf(r.get("ma60"))
    if ma60 is not None:
        ok = close > ma60
        add("📐 趨勢", "收盤站上季線 MA60",
            ok, 10 if ok else 0, 10,
            f"收盤 {close:.2f} {'> ↑ 季線多頭' if ok else '< ↓ 季線空頭'} MA60 {ma60:.2f}")

    # MA20 斜率（月線是否仍在上揚？）
    if "ma20" in df.columns and len(df) >= 6:
        ma20_now  = _sf(df["ma20"].iloc[-1])
        ma20_prev = _sf(df["ma20"].iloc[-6])
        if ma20_now is not None and ma20_prev is not None and ma20_prev > 0:
            rising    = ma20_now > ma20_prev
            slope_pct = (ma20_now - ma20_prev) / ma20_prev * 100
            add("📐 趨勢", "月線方向向上 (MA20 斜率)",
                rising, 5 if rising else 0, 5,
                f"MA20 近5日{'上揚 ↑' if rising else '下彎 ↓'} {slope_pct:+.2f}%"
                f"（{ma20_prev:.2f} → {ma20_now:.2f}）")

    # ── 動能 ─────────────────────────────────
    rsi = _sf(r.get("rsi_14"))
    if rsi is not None:
        if 40 <= rsi <= 70:
            pts, ok, msg = 15, True,  f"RSI {rsi:.1f} — 健康動能區間 (40~70)"
        elif rsi < 30:
            pts, ok, msg = 8,  True,  f"RSI {rsi:.1f} — 超賣，可能反彈"
        elif 30 <= rsi < 40 or 70 < rsi <= 80:
            pts, ok, msg = 5,  False, f"RSI {rsi:.1f} — 邊界區間，待確認"
        else:
            pts, ok, msg = 0,  False, f"RSI {rsi:.1f} — 超買過熱 (>80)，謹慎"
        add("⚡ 動能", "RSI(14) 動能狀態", ok, pts, 15, msg)

    macd_hist      = _sf(r.get("macd_hist"))
    macd_line      = _sf(r.get("macd_line"))
    macd_signal    = _sf(r.get("macd_signal"))
    prev_macd_hist = _sf(df.iloc[-2].get("macd_hist")) if len(df) >= 2 else None
    if macd_hist is not None:
        # 動能擴張：今日柱 > 昨日柱（多頭擴張 或 空頭收斂，均視為有利）
        expanding = prev_macd_hist is not None and macd_hist > prev_macd_hist
        ok        = expanding if prev_macd_hist is not None else macd_hist > 0
        cross = ""
        if macd_line is not None and macd_signal is not None:
            cross = f"，DIF {macd_line:+.3f} / DEA {macd_signal:+.3f}"
        if prev_macd_hist is not None:
            delta     = macd_hist - prev_macd_hist
            direction = ("多頭擴張 ↑" if macd_hist > 0 and expanding else
                         "空頭收斂 ↑" if macd_hist <= 0 and expanding else
                         "多頭收斂 ↓" if macd_hist > 0 else
                         "空頭擴張 ↓")
            detail = (f"直方圖 {macd_hist:+.3f}（{direction}，Δ{delta:+.3f}）{cross}")
        else:
            detail = (f"直方圖 {macd_hist:+.3f}（{'正值' if macd_hist > 0 else '負值'}，無前日資料）{cross}")
        add("⚡ 動能", "MACD 柱狀圖動能", ok, 10 if ok else 0, 10, detail)

    k_val = _sf(r.get("k_val"))
    d_val = _sf(r.get("d_val"))
    if k_val is not None and d_val is not None:
        ok = k_val > d_val
        add("⚡ 動能", "KD 黃金 / 死亡交叉",
            ok, 10 if ok else 0, 10,
            f"K = {k_val:.1f}，D = {d_val:.1f}（{'K > D 多頭' if ok else 'K < D 空頭'}）")

    # ── 波動①：布林通道位置 (10 pts) ─────────────
    bb_mid   = _sf(r.get("bb_mid"))
    bb_upper = _sf(r.get("bb_upper"))
    bb_lower = _sf(r.get("bb_lower"))
    if bb_mid is not None and bb_upper is not None:
        if close >= bb_upper:
            pts, ok, msg = 3, False, f"突破上軌 {bb_upper:.2f}，留意超買延伸"
        elif close > bb_mid:
            pts, ok, msg = 10, True,  f"中軌 {bb_mid:.2f} 上方，布林多頭區"
        else:
            ll = f"，近下軌 {bb_lower:.2f}" if bb_lower is not None else ""
            pts, ok, msg = 0, False, f"中軌 {bb_mid:.2f} 下方，布林空頭區{ll}"
        add("📊 波動", "布林通道位置", ok, pts, 10, msg)

    # ── 波動②：BB 壓縮突破訊號 (10 pts) ─────────
    # 壓縮定義：近 5 日均帶寬 < 過去 60 根帶寬的第 20 分位數（動態相對標準）
    # 高勝率：壓縮 + 突破上軌 + 爆量（三者同時）
    if "bb_width" in df.columns and bb_mid is not None and bb_upper is not None:
        bw_series = df["bb_width"].dropna()
        if len(bw_series) >= 5:
            # 動態閾值：取過去至少 60 根（不足則全用）的 20% 分位
            bw_lookback = bw_series.iloc[-60:] if len(bw_series) >= 60 else bw_series
            bw_p20      = float(bw_lookback.quantile(0.2))

            avg_bw_5d   = float(bw_series.iloc[-5:].mean())
            is_squeeze  = avg_bw_5d < bw_p20
            is_breakout = close >= bb_upper

            vol_v    = _sf(r.get("volume"))
            avg5_vol = (float(df["volume"].iloc[-6:-1].mean())
                        if "volume" in df.columns and len(df) >= 6 else None)
            vol_surge = (vol_v is not None and avg5_vol is not None
                         and avg5_vol > 0 and vol_v > avg5_vol * 1.5
                         and vol_v >= 500)

            if is_squeeze and is_breakout and vol_surge:
                pts, ok = 10, True
                msg = (f"壓縮突破上軌且爆量"
                       f"（5日均帶寬 {avg_bw_5d:.3f} < P20 {bw_p20:.3f}）— 最高勝率訊號 ⭐")
            elif is_squeeze and is_breakout:
                pts, ok = 7, True
                msg = (f"壓縮後突破上軌（5日均帶寬 {avg_bw_5d:.3f} < P20 {bw_p20:.3f}），"
                       "放量確認後信號更強")
            elif is_squeeze:
                pts, ok = 5, True
                msg = (f"通道壓縮中（5日均帶寬 {avg_bw_5d:.3f} < P20 {bw_p20:.3f}），"
                       "等待突破上軌方向")
            else:
                pts, ok = 0, False
                msg = (f"通道未達壓縮（5日均帶寬 {avg_bw_5d:.3f}"
                       f" ≥ P20 {bw_p20:.3f}），無擠壓爆發條件")

            add("📊 波動", "BB壓縮突破訊號", ok, pts, 10, msg)

    # ── 量能 ─────────────────────────────────
    _VOL_MIN = 500   # 流動性絕對門檻：今日成交量須達 500 張
    vol_today = _sf(r.get("volume"))
    if vol_today is not None and len(df) >= 6:
        avg5 = float(df["volume"].iloc[-6:-1].mean())
        if avg5 > 0:
            ratio         = vol_today / avg5
            has_liquidity = vol_today >= _VOL_MIN
            if ratio >= 1.5 and has_liquidity:
                pts, ok, msg = 10, True,  (f"量比 {ratio:.2f}x，今日 {int(vol_today)} 張"
                                            " — 爆量，買盤強勁")
            elif ratio >= 1.5:
                pts, ok, msg =  3, False, (f"量比 {ratio:.2f}x 但今日僅 {int(vol_today)} 張"
                                            f"（< {_VOL_MIN} 張門檻）— 流動性不足，訊號無效")
            elif ratio >= 1.0 and has_liquidity:
                pts, ok, msg =  5, True,  (f"量比 {ratio:.2f}x，今日 {int(vol_today)} 張"
                                            " — 溫和放量")
            elif ratio >= 1.0:
                pts, ok, msg =  3, False, (f"量比 {ratio:.2f}x 但今日僅 {int(vol_today)} 張"
                                            f"（< {_VOL_MIN} 張門檻）— 流動性不足")
            else:
                pts, ok, msg =  0, False,  f"量比 {ratio:.2f}x — 縮量，注意觀望"
            add("📦 量能", "今日量 vs 5日均量", ok, pts, 10, msg)

    # ── 信號評級 ─────────────────────────────
    pct = total_pts / max_pts * 100 if max_pts > 0 else 0

    if pct >= 70:
        signal, color = "強力進場", "#2E7D32"
    elif pct >= 50:
        signal, color = "偏多觀察", "#F57F17"
    elif pct >= 30:
        signal, color = "偏空觀望", "#E65100"
    else:
        signal, color = "謹慎避開", "#B71C1C"

    # ── 多空環境守門（防呆機制）──────────────────────────────────
    # 若收盤站在季線（MA60）以下，最多給「偏多觀察」
    # 大環境空頭格局下，短線反彈訊號的歷史勝率顯著偏低
    gate_reason: Optional[str] = None
    ma60_gate = _sf(r.get("ma60"))
    if ma60_gate is not None and close < ma60_gate:
        if signal == "強力進場":
            signal = "偏多觀察"
            color  = "#F57F17"
            gate_reason = (
                f"季線守門：收盤 {close:.2f} 位於 MA60 {ma60_gate:.2f} 以下，"
                "大環境偏空，信號上限壓制為「偏多觀察」"
            )

    return {
        "signal":      signal,
        "color":       color,
        "score_pct":   round(pct, 1),
        "total_pts":   total_pts,
        "max_pts":     max_pts,
        "checks":      checks,
        "close":       close,
        "date":        str(df_full.iloc[-1]["date"])[:10],
        "gate_reason": gate_reason,
    }


# ═════════════════════════════════════════════
# 展示層：均線扣抵值儀表板
# ═════════════════════════════════════════════

def render_deduction_section(
    deduction_data: List[Dict[str, Any]],
    symbol: str,
) -> None:
    """
    渲染均線扣抵值儀表板：四欄卡片 + 明細表。
    """
    st.markdown("---")
    st.subheader(f"📊 {symbol} 均線扣抵值與趨勢預判")
    st.caption(
        "扣抵價 = 明日均線計算中將被移出的那筆收盤價（df.iloc[-N]['close']）｜"
        "乖離 ≤ ±1% 視為盤整轉折"
    )

    # ── 欄位數依實際均線數量動態決定（3 或 4 欄）──
    cols = st.columns(len(deduction_data))
    for col, d in zip(cols, deduction_data):
        color = d["trend_color"]
        with col:
            st.markdown(f"""
<div style="
    border: 1.5px solid {color};
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
    background: {color}12;
">
  <div style="font-size:13px; font-weight:700; color:#444;">
    {d['ma_name']}
    <span style="font-size:11px; color:#888; font-weight:400;">（{d['subtitle']}）</span>
  </div>
  <div style="font-size:18px; font-weight:700; color:{color}; margin:8px 0 6px; line-height:1.3;">
    {d['trend']}
  </div>
  <div style="font-size:12px; color:#555; line-height:2.0;">
    均線值&emsp;<b style="color:#333;">{d['ma_val']:,.2f}</b><br>
    扣抵價&emsp;<b style="color:{color};">{d['deduction_price']:,.2f}</b><br>
    乖離幅度&emsp;<b style="color:{color};">{d['diff_pct']:+.2f}%</b>
  </div>
</div>""", unsafe_allow_html=True)

    # ── 明細表 ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    table_rows = [
        {
            "均線":       f"{d['ma_name']}（{d['subtitle']}）",
            "目前收盤價": d["current_close"],
            "均線值":     d["ma_val"],
            "明日扣抵價": d["deduction_price"],
            "乖離幅度(%)": f"{d['diff_pct']:+.2f}%",
            "趨勢預判":   d["trend"],
        }
        for d in deduction_data
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════
# 展示層：進場訊號卡片
# ═════════════════════════════════════════════

def render_entry_signal(result: Dict[str, Any], symbol: str) -> None:
    """渲染進場訊號分析卡片（信號徽章 + 分項明細）。"""
    from collections import defaultdict

    st.markdown("---")
    st.subheader(f"🎯 {symbol} 進場訊號分析")

    gate_reason = result.get("gate_reason")
    if gate_reason:
        st.warning(f"⚠️ {gate_reason}", icon="🚧")

    col_badge, col_checks = st.columns([1, 2], gap="large")

    with col_badge:
        color  = result["color"]
        signal = result["signal"]
        pct    = result["score_pct"]
        total  = result["total_pts"]
        mx     = result["max_pts"]
        date_  = result["date"]
        close_ = result["close"]

        # 信號徽章
        st.markdown(f"""
<div style="
  border:2px solid {color};
  border-radius:14px;
  padding:22px 16px;
  text-align:center;
  background:{color}18;
">
  <div style="font-size:12px;color:#888;margin-bottom:4px;">{date_} 收盤</div>
  <div style="font-size:36px;font-weight:800;color:{color};line-height:1.2;">{signal}</div>
  <div style="font-size:22px;font-weight:600;color:#333;margin:10px 0 6px;">
    {close_:,.2f}
  </div>
  <div style="font-size:12px;color:#666;">
    技術評分 {total} / {mx} 分
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 進度條 + 百分比標籤
        st.caption(f"綜合達標率：{pct:.1f}%")
        st.progress(int(pct) / 100)

        # 信號說明
        tips = {
            "強力進場": "技術面多頭格局明確，量能配合，可考慮進場佈局。",
            "偏多觀察": "多頭趨勢初現，但部分指標尚未完全確認，可少量試單。",
            "偏空觀望": "趨勢偏弱或指標分歧，建議觀望待訊號更明朗。",
            "謹慎避開": "多項技術指標偏空，進場風險較高，建議暫時迴避。",
        }
        st.caption(tips.get(signal, ""))

    with col_checks:
        # 依類別分組顯示
        cat_groups: Dict[str, List] = defaultdict(list)
        for c in result["checks"]:
            cat_groups[c["cat"]].append(c)

        for cat, items in cat_groups.items():
            cat_pts = sum(i["pts"] for i in items)
            cat_max = sum(i["max_pts"] for i in items)
            pct_bar = cat_pts / cat_max if cat_max > 0 else 0

            # 類別標題 + 分數
            st.markdown(
                f"**{cat}** &nbsp;"
                f"<span style='color:#888;font-size:12px;'>{cat_pts} / {cat_max} 分</span>",
                unsafe_allow_html=True,
            )

            for item in items:
                icon = "✅" if item["ok"] else "❌"
                st.markdown(
                    f"{icon} **{item['name']}**  \n"
                    f"<span style='color:#666;font-size:12px;margin-left:20px;'>"
                    f"{item['detail']}</span>",
                    unsafe_allow_html=True,
                )

            # 類別進度條
            st.progress(pct_bar)
            st.markdown("")


# ═════════════════════════════════════════════
# 展示層：共用圖表 / 表格渲染函式
# ═════════════════════════════════════════════

def render_data_table(df: pd.DataFrame, symbol: str) -> None:
    """以 DataFrame 表格形式展示股價資料。"""
    st.subheader(f"📋 {symbol} 近期歷史資料")
    display_df = df.copy()
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
    col_map = {
        "date": "日期", "open": "開盤價", "high": "最高價",
        "low": "最低價", "close": "收盤價", "volume": "成交量",
    }
    display_df = display_df.rename(
        columns={k: v for k, v in col_map.items() if k in display_df.columns}
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)


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
    show_bb: bool = False,
    show_rsi: bool = False,
    show_macd: bool = False,
) -> None:
    """
    繪製 K 線 + 均線 + 布林 + 成交量 + 成交值 + KD / RSI / MACD 子圖。

    子圖結構（依資料與參數動態決定）：
      Row 1：K 線圖 + MA 均線覆蓋 + 布林通道（選用）
      Row 2：成交量柱狀圖（若有資料）
      Row 3：成交值柱狀圖（若有資料）
      Row N：KD / RSI / MACD（依啟用順序）

    Parameters
    ----------
    df        : 含 OHLCV 欄位的 DataFrame；若已含均線/指標欄位則直接使用
    symbol    : 股票代號
    show_ma   : 要顯示的均線天數清單，例如 [5, 10, 20]；None 表示不顯示
    show_kd   : 是否顯示 KD 子圖
    show_bb   : 是否在 K 線圖上疊加布林通道
    show_rsi  : 是否顯示 RSI(14) 子圖
    show_macd : 是否顯示 MACD 子圖
    """
    required = {"open", "high", "low", "close", "date"}
    if not required.issubset(df.columns):
        return

    has_volume   = "volume"   in df.columns and df["volume"].notna().any()
    has_turnover = "turnover" in df.columns and df["turnover"].notna().any()
    ma_periods   = show_ma or []

    # 將日期轉為字串，確保 category 軸的 x 值與標註 x 值完全一致
    x_labels = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # ── 動態建立子圖列表 ─────────────────────────
    # 每個 dict：title、base_height（歸一化前）
    rows_cfg = [{"title": f"{symbol} K 線", "h": 0.48}]
    if has_volume:
        rows_cfg.append({"title": "成交量（張）",     "h": 0.16})
    if has_turnover:
        rows_cfg.append({"title": "成交值（千元）",    "h": 0.12})
    if show_kd:
        rows_cfg.append({"title": "KD 值",            "h": 0.16})
    if show_rsi:
        rows_cfg.append({"title": "RSI (14)",          "h": 0.16})
    if show_macd:
        rows_cfg.append({"title": "MACD (12,26,9)",    "h": 0.20})

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
        high_price = float(df["high"].iloc[idx_high])
        low_date   = x_labels.iloc[idx_low]
        low_price  = float(df["low"].iloc[idx_low])

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

    # ── Row 1 覆蓋：布林通道 ─────────────────────
    if show_bb and {"bb_upper", "bb_mid", "bb_lower"}.issubset(df.columns):
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["bb_upper"],
            mode="lines", name="BB 上軌",
            line=dict(color="#EF5350", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["bb_mid"],
            mode="lines", name="BB 中軌",
            line=dict(color="#9E9E9E", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["bb_lower"],
            mode="lines", name="BB 下軌",
            line=dict(color="#26A69A", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(0,0,0,0.03)",
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

    # ── KD 值 ────────────────────────────────────
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
        fig.add_hline(y=80, line=dict(color="#EF5350", dash="dash", width=1),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.add_hline(y=20, line=dict(color="#26A69A", dash="dash", width=1),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(range=[0, 100], title_text="KD", row=current_row, col=1)
        current_row += 1

    # ── RSI(14) ───────────────────────────────────
    if show_rsi and "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=x_labels, y=df["rsi_14"],
            mode="lines", name="RSI",
            line=dict(color="#7B1FA2", width=1.5),
        ), row=current_row, col=1)
        fig.add_hline(y=70, line=dict(color="#EF5350", dash="dot", width=1),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.add_hline(y=50, line=dict(color="#9E9E9E", dash="dot", width=0.8),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.add_hline(y=30, line=dict(color="#26A69A", dash="dot", width=1),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=current_row, col=1)
        current_row += 1

    # ── MACD (12,26,9) ────────────────────────────
    if show_macd and "macd_hist" in df.columns:
        hist_colors = [
            "#EF5350" if (v is not None and not pd.isna(v) and float(v) >= 0) else "#26A69A"
            for v in df["macd_hist"]
        ]
        fig.add_trace(go.Bar(
            x=x_labels, y=df["macd_hist"],
            marker_color=hist_colors,
            name="MACD 柱", showlegend=False,
        ), row=current_row, col=1)
        if "macd_line" in df.columns:
            fig.add_trace(go.Scatter(
                x=x_labels, y=df["macd_line"],
                mode="lines", name="DIF",
                line=dict(color="#FF6B35", width=1.5),
            ), row=current_row, col=1)
        if "macd_signal" in df.columns:
            fig.add_trace(go.Scatter(
                x=x_labels, y=df["macd_signal"],
                mode="lines", name="DEA",
                line=dict(color="#2196F3", width=1.5),
            ), row=current_row, col=1)
        fig.add_hline(y=0, line=dict(color="#9E9E9E", dash="dot", width=0.8),
                      row=current_row, col=1)  # type: ignore[arg-type]
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1  # noqa: F841

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
            "股票代號", value="1815", max_chars=10,
            key="single_stock_symbol",
            help="輸入台灣股票代號，例如 1815、2345、0050",
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
        show_ma5  = st.checkbox("MA5",         value=True,  key="ss_ma5")
        show_ma10 = st.checkbox("MA10",        value=True,  key="ss_ma10")
        show_ma20 = st.checkbox("MA20",        value=True,  key="ss_ma20")
        show_kd   = st.checkbox("KD（9日）",   value=True,  key="ss_kd")
        show_bb   = st.checkbox("布林通道",     value=False, key="ss_bb",
                                help="Bollinger Bands (20,2)；灰底為通道帶寬區域")
        show_rsi  = st.checkbox("RSI（14）",   value=False, key="ss_rsi",
                                help="相對強弱指標；70 超買 / 30 超賣")
        show_macd = st.checkbox("MACD（12,26,9）", value=False, key="ss_macd",
                                help="DIF / DEA / 柱狀圖")

        query_btn = st.button("查詢", type="primary", use_container_width=True)

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
        # MACD slow=26 需 26 筆，Bollinger period=20，RSI period=14
        # 季線（60MA）扣抵值計算需至少 60 筆，故 fetch_limit 至少取 120
        warmup = max([0] + ma_periods + ([9] if show_kd else []) +
                     ([26] if show_macd else []) + ([20] if show_bb else []) +
                     ([14] if show_rsi else [])) + 30
        fetch_limit = max(int(limit) + warmup, 120)

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
        # 進場訊號分析固定需要所有指標，一次計算完畢
        df_full = compute_ma(df_full, [5, 10, 20, 60])   # 含季線，進場訊號守門需要
        df_full = compute_kd(df_full)
        df_full = compute_rsi(df_full)
        df_full = compute_macd(df_full)
        df_full = compute_bollinger(df_full)

        # 裁切至使用者指定的顯示天數
        df = df_full.tail(int(limit)).reset_index(drop=True)

        latest      = df.iloc[-1]
        prev        = df.iloc[-2] if len(df) >= 2 else latest
        price_delta = float(latest["close"]) - float(prev["close"]) if "close" in df.columns else 0

        # ── 最新報價指標卡 ─────────────────────────
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        if "close"    in df.columns: m1.metric("收盤價",        f"{latest['close']:,.2f}",   f"{price_delta:+.2f}")
        if "open"     in df.columns: m2.metric("開盤價",        f"{latest['open']:,.2f}")
        if "high"     in df.columns: m3.metric("最高價",        f"{latest['high']:,.2f}")
        if "low"      in df.columns: m4.metric("最低價",        f"{latest['low']:,.2f}")
        if "volume"   in df.columns: m5.metric("成交量（張）",   f"{int(latest['volume']):,}")
        if "turnover" in df.columns: m6.metric("成交值（千元）", f"{int(latest['turnover']):,}")

        # ── 技術指標快訊（RSI / MACD / KD 數值）─────
        has_rsi  = "rsi_14"    in df.columns
        has_macd = "macd_hist" in df.columns
        has_kd   = "k_val"     in df.columns
        has_bb   = "bb_upper"  in df.columns
        if any([has_rsi, has_macd, has_kd, has_bb]):
            n_info = sum([has_rsi, has_macd, has_kd, has_bb])
            info_cols = st.columns(n_info)
            ic = 0
            if has_rsi:
                rsi_v = float(latest["rsi_14"])
                rsi_label = "超買" if rsi_v > 70 else ("超賣" if rsi_v < 30 else "正常")
                info_cols[ic].metric("RSI(14)", f"{rsi_v:.1f}", rsi_label)
                ic += 1
            if has_macd:
                hist_v = float(latest["macd_hist"])
                info_cols[ic].metric("MACD 柱", f"{hist_v:+.3f}",
                                     "多頭" if hist_v > 0 else "空頭")
                ic += 1
            if has_kd:
                k_v = float(latest["k_val"])
                d_v = float(latest["d_val"])
                info_cols[ic].metric("KD",
                                     f"K {k_v:.1f} / D {d_v:.1f}",
                                     "黃金" if k_v > d_v else "死亡")
                ic += 1
            if has_bb:
                bb_w = float(latest["bb_width"]) if "bb_width" in df.columns else 0
                info_cols[ic].metric("BB 帶寬", f"{bb_w:.3f}",
                                     "擠壓" if bb_w < df["bb_width"].quantile(0.2) else "正常")

        # ── K 線型態快訊（酒田戰法，pandas-ta cdl_pattern）────────────
        cdl_patterns = detect_all_candlestick_patterns(df_full)
        if cdl_patterns:
            badges = []
            for p in cdl_patterns:
                is_bull = p.startswith("🟢")
                color   = "#2E7D32" if is_bull else "#C62828"
                bg      = "#E8F5E9" if is_bull else "#FFEBEE"
                badges.append(
                    f'<span style="display:inline-block;background:{bg};'
                    f'border:1.5px solid {color};border-radius:6px;'
                    f'padding:4px 12px;margin:3px 4px;font-size:14px;'
                    f'font-weight:600;color:{color};">{p}</span>'
                )
            st.markdown(
                '<div style="margin:6px 0;"><b>🕯️ 今日 K 線型態</b><br>'
                + "".join(badges) + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p style="font-size:13px;color:#999;margin:6px 0;">'
                "➖ 今日無特殊 K 線型態</p>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        render_ohlcv_chart(
            df, symbol,
            show_ma=ma_periods if ma_periods else [5, 10, 20],
            show_kd=show_kd,
            show_bb=show_bb,
            show_rsi=show_rsi,
            show_macd=show_macd,
        )
        render_data_table(df, symbol)

        # ── 均線扣抵值模組（使用完整資料集確保季線有效）──
        df_full, deduction_data = calculate_deduction_values(df_full)
        if deduction_data:
            render_deduction_section(deduction_data, symbol)
        else:
            st.info("顯示天數不足以計算任何均線扣抵值。")

        # ── 進場訊號分析 ─────────────────────────────
        entry_result = analyze_entry_signal(df_full)
        if entry_result:
            render_entry_signal(entry_result, symbol)
        else:
            st.info("資料筆數不足（需 30 筆以上）以進行進場訊號分析。")
