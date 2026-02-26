"""
個股綜合評分頁面（Tab 3）。
雙模式動態權重評分：
  模式 A：短線動能與波段操作
  模式 B：長線資產累積（左側交易）
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import pandas_ta as ta  # noqa: F401
import plotly.graph_objects as go
import streamlit as st

from utils import fetch_stock_candles


# ═════════════════════════════════════════════
# 常數
# ═════════════════════════════════════════════

# 模式 B 需要 240MA，須確保至少 300 根 K 棒
# utils.fetch_stock_candles 會自動分段抓取（Fugle API 單次上限 < 365 天）
_SCORE_FETCH_LIMIT = 300

MODE_A = "A"
MODE_B = "B"


# ═════════════════════════════════════════════
# 評分模型：模式 A — 短線動能與波段操作
# ═════════════════════════════════════════════

def _has_deduction_pressure(df: pd.DataFrame, period: int) -> bool:
    """判斷 N-MA 扣抵值是否大於現價（有向下壓力）。"""
    if len(df) < period + 1:
        return False
    return float(df["close"].iloc[-period]) > float(df["close"].iloc[-1])


def compute_score_mode_a(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    模式 A：短線動能與波段操作評分（100 分制）。

    維度
    ----
    趨勢 Trend     40 分  — 10/20/60MA 多頭排列 + 均線扣抵壓力
    動能 Momentum  30 分  — RSI(14) > 50 + MACD 柱狀圖翻紅
    量能 Volume    30 分  — 今日量 vs 5 日均量（帶量突破）
    """
    if df.empty or len(df) < 65:
        return None

    df = df.copy().reset_index(drop=True)

    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    _rsi = ta.rsi(df["close"], length=14)
    df["rsi14"] = _rsi if _rsi is not None else float("nan")

    _macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if _macd is not None and "MACDh_12_26_9" in _macd.columns:
        df["macd_hist"] = _macd["MACDh_12_26_9"].values
    else:
        df["macd_hist"] = float("nan")

    last = df.iloc[-1]

    def _f(col: str) -> Optional[float]:
        v = last[col] if col in df.columns else None
        return None if (v is None or pd.isna(v)) else float(v)

    def _n(v: Optional[float], dec: int = 2) -> str:
        return f"{v:,.{dec}f}" if v is not None else "N/A"

    close     = _f("close")
    ma10      = _f("ma10")
    ma20      = _f("ma20")
    ma60      = _f("ma60")
    rsi       = _f("rsi14")
    macd_hist = _f("macd_hist")
    volume    = _f("volume") if "volume" in df.columns else None
    vol_5avg  = (
        float(df["volume"].iloc[-6:-1].mean())
        if "volume" in df.columns and len(df) >= 6 else None
    )

    def _above(price: Optional[float], ma: Optional[float]) -> bool:
        return price is not None and ma is not None and price > ma

    # ── 趨勢 Trend（40 分）──────────────────────
    t10 = 10 if _above(close, ma10) else 0
    t20 = 10 if _above(close, ma20) else 0
    t60 = 10 if _above(close, ma60) else 0

    pressure_count = sum([
        _has_deduction_pressure(df, 10),
        _has_deduction_pressure(df, 20),
        _has_deduction_pressure(df, 60),
    ])
    if pressure_count == 0:
        ded_pts, ded_st = 10, "✅ 三均線扣抵無壓（易漲）"
    elif pressure_count == 1:
        ded_pts, ded_st =  5, "⚠️ 1 條均線有扣抵壓力"
    else:
        ded_pts, ded_st =  0, f"❌ {pressure_count} 條均線有扣抵壓力（易跌）"

    trend_score = t10 + t20 + t60 + ded_pts

    # ── 動能 Momentum（30 分）───────────────────
    if rsi is not None:
        if 50 <= rsi <= 70:
            rsi_pts, rsi_st = 15, f"RSI {rsi:.1f}（50~70 健康多頭 ✅）"
        elif rsi > 70:
            rsi_pts, rsi_st = 10, f"RSI {rsi:.1f}（> 70 超買警示 ⚠️）"
        elif 40 <= rsi < 50:
            rsi_pts, rsi_st =  5, f"RSI {rsi:.1f}（40~50 中性偏弱）"
        else:
            rsi_pts, rsi_st =  0, f"RSI {rsi:.1f}（< 40 弱勢 ❌）"
    else:
        rsi_pts, rsi_st = 0, "資料不足"

    if macd_hist is not None:
        hist_pts = 15 if macd_hist > 0 else 0
        hist_st  = f"MACD 柱狀 {macd_hist:.4f}（{'翻紅 ✅' if macd_hist > 0 else '翻綠 ❌'}）"
    else:
        hist_pts, hist_st = 0, "資料不足"

    momentum_score = rsi_pts + hist_pts

    # ── 量能 Volume（30 分）─────────────────────
    if volume is not None and vol_5avg is not None and vol_5avg > 0:
        ratio = volume / vol_5avg
        if ratio >= 1.5:
            vol_pts, vol_st = 30, f"量能 {ratio:.1f}x 均量（帶量突破 ✅）"
        elif ratio >= 1.0:
            vol_pts, vol_st = 20, f"量能 {ratio:.1f}x 均量（略放量）"
        else:
            vol_pts, vol_st =  0, f"量能 {ratio:.1f}x 均量（量縮 ❌）"
    else:
        vol_pts, vol_st = 0, "資料不足"

    volume_score = vol_pts
    total_score  = trend_score + momentum_score + volume_score

    details: List[Dict[str, str]] = [
        {"維度": "趨勢 Trend",    "指標": "站上 10MA",
         "數值": f"收 {_n(close)} {'>' if t10 else '≤'} 10MA {_n(ma10)}",
         "判斷": "✅ 多頭" if t10 else "❌ 空頭", "得分": f"{t10} / 10"},
        {"維度": "趨勢 Trend",    "指標": "站上 20MA",
         "數值": f"收 {_n(close)} {'>' if t20 else '≤'} 20MA {_n(ma20)}",
         "判斷": "✅ 多頭" if t20 else "❌ 空頭", "得分": f"{t20} / 10"},
        {"維度": "趨勢 Trend",    "指標": "站上 60MA",
         "數值": f"收 {_n(close)} {'>' if t60 else '≤'} 60MA {_n(ma60)}",
         "判斷": "✅ 多頭" if t60 else "❌ 空頭", "得分": f"{t60} / 10"},
        {"維度": "趨勢 Trend",    "指標": "均線扣抵壓力",
         "數值": f"{pressure_count} 條均線有壓力",
         "判斷": ded_st,                          "得分": f"{ded_pts} / 10"},
        {"維度": "動能 Momentum", "指標": "RSI (14)",
         "數值": _n(rsi),         "判斷": rsi_st, "得分": f"{rsi_pts} / 15"},
        {"維度": "動能 Momentum", "指標": "MACD 柱狀圖",
         "數值": _n(macd_hist, 4), "判斷": hist_st, "得分": f"{hist_pts} / 15"},
        {"維度": "量能 Volume",   "指標": "量能 vs 5 日均量",
         "數值": f"今日 {_n(volume, 0)} 張  均 {_n(vol_5avg, 0)} 張",
         "判斷": vol_st,                          "得分": f"{vol_pts} / 30"},
    ]

    return {
        "total": total_score,
        "mode":  MODE_A,
        "dimensions": {
            "trend":    {"score": trend_score,    "max": 40, "label": "趨勢\nTrend"},
            "momentum": {"score": momentum_score, "max": 30, "label": "動能\nMomentum"},
            "volume":   {"score": volume_score,   "max": 30, "label": "量能\nVolume"},
        },
        "details": details,
    }


# ═════════════════════════════════════════════
# 評分模型：模式 B — 長線資產累積
# ═════════════════════════════════════════════

def compute_score_mode_b(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    模式 B：長線資產累積評分（100 分制）。

    核心：左側交易，越低越好，尋找長線佈局的便宜買點。

    維度
    ----
    價格位階 Price Level  40 分  — 相對 60MA / 240MA（年線）位置
    超賣指標 Oversold     40 分  — RSI(14) < 30 + 60MA 乖離率 < -10%
    長線基期 LT Baseline  20 分  — KD 低檔（< 20）黃金交叉
    """
    if df.empty or len(df) < 65:
        return None

    df = df.copy().reset_index(drop=True)

    df["ma60"]  = df["close"].rolling(60).mean()
    has_240ma   = len(df) >= 240
    if has_240ma:
        df["ma240"] = df["close"].rolling(240).mean()

    _rsi = ta.rsi(df["close"], length=14)
    df["rsi14"] = _rsi if _rsi is not None else float("nan")

    _stoch = ta.stoch(df["high"], df["low"], df["close"], k=9, d=3, smooth_k=3)
    if _stoch is not None and "STOCHk_9_3_3" in _stoch.columns:
        df["k_stoch"] = _stoch["STOCHk_9_3_3"].values
        df["d_stoch"] = _stoch["STOCHd_9_3_3"].values
    else:
        df["k_stoch"] = float("nan")
        df["d_stoch"] = float("nan")

    last = df.iloc[-1]

    def _f(col: str) -> Optional[float]:
        v = last[col] if col in df.columns else None
        return None if (v is None or pd.isna(v)) else float(v)

    def _n(v: Optional[float], dec: int = 2) -> str:
        return f"{v:,.{dec}f}" if v is not None else "N/A"

    close = _f("close")
    ma60  = _f("ma60")
    ma240 = _f("ma240") if has_240ma else None
    rsi   = _f("rsi14")
    k_    = _f("k_stoch")
    d_    = _f("d_stoch")

    # ── 價格位階 Price Level（40 分）────────────
    if close is not None and ma60 is not None:
        if close < ma60:
            price_pts = 40
            price_st  = f"收 {_n(close)} < 60MA {_n(ma60)}（深度折價區 ✅）"
        elif ma240 is not None and close < ma240:
            price_pts = 20
            price_st  = (f"60MA {_n(ma60)} ≤ 收 {_n(close)} "
                         f"< 240MA {_n(ma240)}（中間區）")
        else:
            price_pts = 10
            ref_str   = _n(ma240) if ma240 is not None else "（240MA 資料不足）"
            price_st  = f"收 {_n(close)} ≥ 240MA {ref_str}（偏貴區 ❌）"
    else:
        price_pts, price_st = 0, "資料不足"

    price_level_score = price_pts

    # ── 超賣指標 Oversold（40 分）───────────────
    if rsi is not None:
        rsi_pts = 20 if rsi < 30 else 0
        rsi_st  = (f"RSI {rsi:.1f}（< 30 嚴重超賣 ✅）" if rsi < 30
                   else f"RSI {rsi:.1f}（未超賣）")
    else:
        rsi_pts, rsi_st = 0, "資料不足"

    if close is not None and ma60 is not None:
        bias     = (close - ma60) / ma60 * 100
        bias_pts = 20 if bias < -10 else 0
        bias_st  = (f"乖離率 {bias:.1f}%（< -10% 深度超賣 ✅）" if bias < -10
                    else f"乖離率 {bias:.1f}%（未達 -10%）")
        bias_val = f"{bias:.1f}%"
    else:
        bias_pts, bias_st, bias_val = 0, "資料不足", "N/A"

    oversold_score = rsi_pts + bias_pts

    # ── 長線基期 LT Baseline（20 分）────────────
    if k_ is not None and d_ is not None:
        if k_ < 20 and d_ < 20 and k_ > d_:
            kd_pts = 20
            kd_st  = f"K={k_:.1f} D={d_:.1f}（低檔黃金交叉 ✅）"
        elif k_ < 20 and d_ < 20:
            kd_pts = 10
            kd_st  = f"K={k_:.1f} D={d_:.1f}（KD 低檔盤旋，尚未交叉）"
        elif k_ < 30 or d_ < 30:
            kd_pts =  5
            kd_st  = f"K={k_:.1f} D={d_:.1f}（接近超賣區）"
        else:
            kd_pts =  0
            kd_st  = f"K={k_:.1f} D={d_:.1f}（未在超賣區 ❌）"
    else:
        kd_pts, kd_st = 0, "資料不足"

    lt_baseline_score = kd_pts
    total_score = price_level_score + oversold_score + lt_baseline_score

    details: List[Dict[str, str]] = [
        {"維度": "價格位階 Price Level", "指標": "60 / 240MA 位置",
         "數值": f"收 {_n(close)}  60MA {_n(ma60)}  240MA {_n(ma240)}",
         "判斷": price_st, "得分": f"{price_pts} / 40"},
        {"維度": "超賣指標 Oversold", "指標": "RSI (14)",
         "數值": _n(rsi), "判斷": rsi_st, "得分": f"{rsi_pts} / 20"},
        {"維度": "超賣指標 Oversold", "指標": "60MA 乖離率",
         "數值": bias_val, "判斷": bias_st, "得分": f"{bias_pts} / 20"},
        {"維度": "長線基期 LT Baseline", "指標": "KD 低檔黃金交叉",
         "數值": f"K={_n(k_)} D={_n(d_)}", "判斷": kd_st, "得分": f"{kd_pts} / 20"},
    ]

    return {
        "total": total_score,
        "mode":  MODE_B,
        "dimensions": {
            "price_level": {"score": price_level_score, "max": 40,
                            "label": "價格位階\nPrice Level"},
            "oversold":    {"score": oversold_score,    "max": 40,
                            "label": "超賣指標\nOversold"},
            "lt_baseline": {"score": lt_baseline_score, "max": 20,
                            "label": "長線基期\nLT Baseline"},
        },
        "details": details,
    }


# ─────────────────────────────────────────────
# 評分頁面：雷達圖（動態維度，連動模式）
# ─────────────────────────────────────────────

def render_radar_chart(score_result: Dict[str, Any]) -> None:
    """繪製評分雷達圖，軸線依模式自動切換。"""
    dims     = score_result["dimensions"]
    dim_keys = list(dims.keys())
    labels   = [str(dims[k]["label"]) for k in dim_keys]
    pcts     = [int(dims[k]["score"]) / int(dims[k]["max"]) * 100 for k in dim_keys]

    # 閉合多邊形
    r_vals     = pcts   + [pcts[0]]
    theta_vals = labels + [labels[0]]

    total = int(score_result["total"])
    if total >= 80:
        fill_color, line_color = "rgba(76,175,80,0.20)",  "#4CAF50"
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
            angularaxis=dict(tickfont=dict(size=11), gridcolor="#e0e0e0"),
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
    """個股綜合評分頁面（雙模式 100 分制買進指標）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 查詢條件")
        symbol = st.text_input(
            "股票代號", value="2330", max_chars=10,
            key="score_page_symbol",
            help="輸入台灣股票代號，例如 2330（台積電）",
        ).strip()

        st.markdown("##### 投資策略模式")
        mode = st.radio(
            "投資策略模式",
            options=[MODE_A, MODE_B],
            format_func=lambda x: (
                "📈 短線動能與波段操作"
                if x == MODE_A else
                "🏦 長線資產累積"
            ),
            key="score_page_mode",
            label_visibility="collapsed",
        )

        if mode == MODE_A:
            st.caption(
                "追強勢策略\n"
                "趨勢 40% ＋ 動能 30% ＋ 量能 30%\n"
                "適合個股突破進場"
            )
        else:
            st.caption(
                "左側交易策略\n"
                "價格位階 40% ＋ 超賣指標 40% ＋ 長線基期 20%\n"
                "20 年期以上 / 大盤 ETF 定期定額"
            )

        query_btn = st.button("開始評分", type="primary", use_container_width=True)

    with result_col:
        if not query_btn:
            st.info("請在左側選擇投資策略模式並輸入股票代號，點擊「開始評分」。")
            return

        if not symbol:
            st.error("股票代號不得為空，請重新輸入。")
            return

        with st.spinner(f"正在分析 {symbol}…"):
            try:
                df_full = fetch_stock_candles(
                    symbol=symbol,
                    limit=_SCORE_FETCH_LIMIT,
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

        if mode == MODE_A:
            score_result = compute_score_mode_a(df_full)
        else:
            score_result = compute_score_mode_b(df_full)

        if score_result is None:
            st.warning(
                f"**{symbol}** 歷史資料不足（需至少 65 個交易日），無法進行評分。"
            )
            return

        total      = int(score_result["total"])
        mode_label = (
            "📈 短線動能與波段操作（適合個股突破）" if mode == MODE_A
            else "🏦 長線資產累積（20 年期以上 / 適合大盤 ETF 定期定額）"
        )

        # ── 提示語（依模式 + 分段）────────────────
        if total >= 80:
            score_color = "#4CAF50"
            score_hint  = (
                "技術面強勢，適合右側順勢進場。" if mode == MODE_A
                else "長線基期偏低，為優良的累積單位數時機。"
            )
        elif total >= 50:
            score_color = "#FF9800"
            score_hint  = (
                "技術面中性，等待更明確突破信號。" if mode == MODE_A
                else "長線價格尚在合理區間，可分批少量佈局。"
            )
        else:
            score_color = "#F44336"
            score_hint  = (
                "技術面偏弱，建議觀望。" if mode == MODE_A
                else "目前尚未進入超值買點，耐心等候回調。"
            )

        # ── 大字體總分卡 ──────────────────────────
        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {score_color}1A, {score_color}0A);
    border-left: 6px solid {score_color};
    border-radius: 8px;
    padding: 18px 28px;
    margin-bottom: 16px;
">
  <div style="color:#888; font-size:11px; font-weight:500; margin-bottom:2px;">
    {mode_label}
  </div>
  <div style="color:#555; font-size:12px; font-weight:600;
              text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;">
    {symbol} 綜合買進評分
  </div>
  <div style="color:{score_color}; font-size:54px; font-weight:700; line-height:1.1;">
    {total}
    <span style="font-size:22px; color:#888; font-weight:400;">/ 100</span>
  </div>
  <div style="color:{score_color}; font-size:15px; font-weight:500; margin-top:6px;">
    {score_hint}
  </div>
</div>""", unsafe_allow_html=True)

        # ── 各維度分數卡片（動態欄數）────────────
        dims     = score_result["dimensions"]
        dim_keys = list(dims.keys())
        metric_cols = st.columns(len(dim_keys))
        for col, k in zip(metric_cols, dim_keys):
            col.metric(
                str(dims[k]["label"]).replace("\n", " "),
                f"{int(dims[k]['score'])} / {int(dims[k]['max'])}",
            )

        st.markdown("---")

        # ── 雷達圖 + 指標明細並排 ──────────────────
        radar_col, table_col = st.columns([1, 1], gap="large")

        with radar_col:
            st.markdown("##### 評分雷達圖")
            render_radar_chart(score_result)

        with table_col:
            st.markdown("##### 指標明細")
            detail_df = pd.DataFrame(score_result["details"])
            st.dataframe(detail_df, use_container_width=True, hide_index=True)
