"""
個股綜合評分頁面（Tab 3）。
包含：評分演算法層（compute_score）+ 雷達圖 + UI 渲染
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401
import plotly.graph_objects as go
import streamlit as st

from utils import fetch_stock_candles


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
