"""
策略歷史回測頁面（Tab 5）。
使用 Screener_page 策略函式對歷史 K 線進行逐筆模擬回測。
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import fetch_stock_candles, compute_atr
from Screener_page import (
    check_consolidation_breakout,
    check_bullish_ma_alignment,
    check_volume_surge_bullish,
    check_oversold_reversal,
    check_52week_high_breakout,
    check_bollinger_squeeze_breakout,
)


# ═════════════════════════════════════════════
# 常數
# ═════════════════════════════════════════════

_YEAR_LIMIT: Dict[str, int] = {
    "1 年（約 260 個交易日）":  260,
    "3 年（約 800 個交易日）":  800,
    "5 年（約 1300 個交易日）": 1300,
}


# ═════════════════════════════════════════════
# 策略登記表（預設參數版，供回測使用）
# ═════════════════════════════════════════════

def _build_strategy_registry() -> Dict[str, Callable[[pd.DataFrame], Optional[Dict[str, Any]]]]:
    """回傳策略名稱 → 函式的對照表（使用各策略預設參數）。"""
    return {
        "盤整突破第一根":   lambda df: check_consolidation_breakout(df),
        "均線多頭排列":     check_bullish_ma_alignment,
        "爆量長紅起漲":     lambda df: check_volume_surge_bullish(df),
        "乖離過大跌深反彈": lambda df: check_oversold_reversal(df),
        "52週高點突破":     lambda df: check_52week_high_breakout(df),
        "布林擠壓突破":     lambda df: check_bollinger_squeeze_breakout(df),
    }


# ═════════════════════════════════════════════
# 演算法層：回測核心
# ═════════════════════════════════════════════

def run_backtest(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], Optional[Dict[str, Any]]],
    take_profit_pct: float = 0.15,
    stop_loss_pct: float = 0.05,
    exit_on_ma20: bool = True,
    use_atr_stop: bool = False,
    atr_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """
    對給定 DataFrame 執行單一策略的模擬回測。

    規則
    ----
    - 訊號觸發日（i）→ 次日開盤（i+1）買入
    - 出場條件（從 i+2 起）：
        1. 收盤 ≥ 買入價 × (1 + take_profit_pct)       → 停利出場
        2. 固定停損：收盤 ≤ 買入價 × (1 − stop_loss_pct) → 停損出場
           ATR停損：收盤 ≤ 買入價 − ATR × atr_multiplier  → 停損出場
        3. 收盤 < 20MA（若 exit_on_ma20=True）           → 均線出場
    - 每筆交易結束後，從出場日+1繼續掃描（無重疊倉位）
    - 若達資料末尾仍未觸發出場條件，以最後一天收盤強制平倉

    Parameters
    ----------
    df              : 已排序（日期升冪）的歷史 DataFrame（需含 open/high/low/close）
    strategy_fn     : 策略判斷函式（接受 DataFrame，回傳 dict 或 None）
    take_profit_pct : 停利百分比（預設 15%）
    stop_loss_pct   : 固定停損百分比（use_atr_stop=False 時使用，預設 5%）
    exit_on_ma20    : 是否啟用跌破 20MA 出場條件
    use_atr_stop    : True → 改用 ATR 動態停損（入場價 − ATR × atr_multiplier）
    atr_multiplier  : ATR 停損倍數（預設 2.0）

    Returns
    -------
    dict {
        trades          : list of trade dicts,
        total_trades    : int,
        win_rate        : float（0~100 的百分比）,
        total_return    : float（累計報酬 %）,
        equity_curve    : list of float（起始 1.0，每筆交易後更新）,
        max_drawdown    : float（最大回撤 %）,
        sharpe          : float（每筆交易夏普比率，年化近似值）,
        profit_factor   : float（獲利因子，贏/虧；無虧損時為 inf）,
        max_consec_loss : int（最大連續虧損筆數）,
    }
    """
    _EMPTY: Dict[str, Any] = {
        "trades": [], "total_trades": 0,
        "win_rate": 0.0, "total_return": 0.0,
        "equity_curve": [1.0],
        "max_drawdown": 0.0, "sharpe": 0.0,
        "profit_factor": 0.0, "max_consec_loss": 0,
    }

    if df.empty or len(df) < 30:
        return _EMPTY

    df = df.copy().reset_index(drop=True)

    # 預先計算 20MA（向量化，避免逐行 rolling）
    df["_ma20"] = df["close"].rolling(20).mean()

    # ATR 動態停損：預先計算全段 ATR
    if use_atr_stop:
        df = compute_atr(df, period=14)

    trades: List[Dict[str, Any]] = []
    i = 0

    while i < len(df) - 1:
        # 策略偵測（傳入歷史前綴，模擬「當日收盤後看到訊號」）
        signal = strategy_fn(df.iloc[:i + 1])
        if signal is None:
            i += 1
            continue

        # 買入：次日開盤
        buy_idx = i + 1
        if buy_idx >= len(df):
            break

        entry_price = float(df.iloc[buy_idx]["open"])
        if entry_price <= 0:
            i = buy_idx + 1
            continue

        tp_price = entry_price * (1.0 + take_profit_pct)

        # 停損價：ATR 動態 或 固定百分比
        if use_atr_stop and "atr" in df.columns:
            atr_val = df.iloc[buy_idx]["atr"]
            if not pd.isna(atr_val) and float(atr_val) > 0:
                sl_price = entry_price - float(atr_val) * atr_multiplier
            else:
                sl_price = entry_price * (1.0 - stop_loss_pct)
        else:
            sl_price = entry_price * (1.0 - stop_loss_pct)

        # 出場掃描（預設強制平倉於最後一天）
        exit_idx    = len(df) - 1
        exit_price  = float(df.iloc[-1]["close"])
        exit_reason = "強制平倉"

        for j in range(buy_idx + 1, len(df)):
            close = float(df.iloc[j]["close"])
            ma20  = df.iloc[j]["_ma20"]

            if close >= tp_price:
                exit_idx, exit_price, exit_reason = j, close, "停利"
                break
            if close <= sl_price:
                exit_idx, exit_price, exit_reason = j, close, "停損"
                break
            if exit_on_ma20 and not pd.isna(ma20) and close < float(ma20):
                exit_idx, exit_price, exit_reason = j, close, "跌破MA20"
                break

        pnl_pct   = (exit_price - entry_price) / entry_price * 100
        hold_days = exit_idx - buy_idx
        trades.append({
            "買入日期": str(df.iloc[buy_idx]["date"])[:10],
            "賣出日期": str(df.iloc[exit_idx]["date"])[:10],
            "持倉(日)": hold_days,
            "買入價":   round(entry_price, 2),
            "賣出價":   round(exit_price,  2),
            "損益(%)":  round(pnl_pct, 2),
            "出場原因": exit_reason,
        })

        # 跳至出場日後，避免倉位重疊
        i = exit_idx + 1

    if not trades:
        return _EMPTY

    pnls = [t["損益(%)"] / 100 for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    n    = len(pnls)

    # 資金曲線 + 最大回撤（MDD）
    equity    = 1.0
    peak      = 1.0
    max_dd    = 0.0
    equity_curve: List[float] = [1.0]
    for p in pnls:
        equity *= 1.0 + p
        peak    = max(peak, equity)
        dd      = (peak - equity) / peak
        max_dd  = max(max_dd, dd)
        equity_curve.append(round(equity, 6))

    # 夏普比率（每筆交易近似，年化×√252）
    mean_r = sum(pnls) / n
    if n > 1:
        variance = sum((p - mean_r) ** 2 for p in pnls) / (n - 1)
        std_r    = variance ** 0.5
        sharpe   = round((mean_r / std_r * (252 ** 0.5)), 2) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # 獲利因子（Profit Factor）
    gains  = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    profit_factor = round(gains / losses, 2) if losses > 0 else float("inf")

    # 最大連續虧損
    max_consec = 0
    cur_consec  = 0
    for p in pnls:
        if p < 0:
            cur_consec += 1
            max_consec  = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    return {
        "trades":          trades,
        "total_trades":    n,
        "win_rate":        round(wins / n * 100, 1),
        "total_return":    round((equity - 1.0) * 100, 2),
        "equity_curve":    equity_curve,
        "max_drawdown":    round(max_dd * 100, 2),
        "sharpe":          sharpe,
        "profit_factor":   profit_factor,
        "max_consec_loss": max_consec,
    }


@st.cache_data(ttl=3600)
def _cached_backtest(
    symbol: str,
    fetch_limit: int,
    strategy_name: str,
    take_profit_pct: float,
    stop_loss_pct: float,
    exit_on_ma20: bool,
    use_atr_stop: bool,
    atr_multiplier: float,
) -> Dict[str, Any]:
    """快取版回測（依參數快取，避免重複 API 呼叫與重算）。"""
    df       = fetch_stock_candles(symbol=symbol, limit=fetch_limit)
    registry = _build_strategy_registry()

    if strategy_name not in registry:
        return {
            "trades": [], "total_trades": 0,
            "win_rate": 0.0, "total_return": 0.0,
            "equity_curve": [1.0],
            "max_drawdown": 0.0, "sharpe": 0.0,
            "profit_factor": 0.0, "max_consec_loss": 0,
        }

    strategy_fn = registry[strategy_name]
    return run_backtest(
        df, strategy_fn,
        take_profit_pct, stop_loss_pct,
        exit_on_ma20, use_atr_stop, atr_multiplier,
    )


# ═════════════════════════════════════════════
# UI 層：回測頁面渲染
# ═════════════════════════════════════════════

def render_backtest_page() -> None:
    """策略歷史回測頁面（Tab 5）。"""
    ctrl_col, result_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown("#### 回測設定")
        symbol = st.text_input(
            "股票代號", value="2330", max_chars=10,
            key="bt_symbol",
            help="輸入台灣股票代號，例如 2330（台積電）",
        ).strip()

        strategy_names = list(_build_strategy_registry().keys())
        strategy_name  = st.selectbox(
            "選擇策略",
            options=strategy_names,
            key="bt_strategy",
            help="回測使用各策略的預設參數",
        )

        year_label = st.selectbox(
            "回測期間",
            options=list(_YEAR_LIMIT.keys()),
            key="bt_year",
        )
        fetch_limit = _YEAR_LIMIT[year_label]

        st.markdown("---")
        st.markdown("##### 出場條件")

        take_profit_pct = st.slider(
            "停利（%）", min_value=5, max_value=50, value=15, step=1,
            key="bt_tp",
            help="收盤達買入價 × (1 + 停利%) 時出場",
        ) / 100.0

        use_atr_stop = st.checkbox(
            "ATR 動態停損", value=False,
            key="bt_atr_stop",
            help="勾選後改用 ATR(14) 動態停損，取代下方固定停損百分比",
        )

        atr_multiplier = st.slider(
            "ATR 倍數", min_value=1.0, max_value=5.0, value=2.0, step=0.5,
            key="bt_atr_mult",
            disabled=not use_atr_stop,
            help="停損價 = 買入價 − ATR(14) × 倍數",
        )

        stop_loss_pct = st.slider(
            "固定停損（%）", min_value=1, max_value=20, value=5, step=1,
            key="bt_sl",
            disabled=use_atr_stop,
            help="ATR 停損關閉時使用：收盤跌至買入價 × (1 - 停損%) 時出場",
        ) / 100.0

        exit_on_ma20 = st.checkbox(
            "跌破 20MA 出場", value=True,
            key="bt_ma20_exit",
            help="收盤跌破 20 日均線時強制出場",
        )

        run_btn = st.button(
            "開始回測", type="primary", use_container_width=True,
            key="bt_run",
        )

    with result_col:
        if not run_btn:
            st.info(
                "請在左側設定回測條件，點擊「開始回測」。\n\n"
                "**注意事項**\n"
                "- 回測為歷史模擬，不代表未來績效\n"
                "- 策略使用預設參數；進階參數調整請至「選股策略」頁面\n"
                "- 每筆交易次日開盤進場，無倉位重疊\n"
                "- 未觸發停利/停損/均線出場者，資料末尾強制平倉"
            )
            return

        if not symbol:
            st.error("股票代號不得為空。")
            return

        with st.spinner(f"正在執行 {symbol} × {strategy_name} 回測…"):
            try:
                result = _cached_backtest(
                    symbol=symbol,
                    fetch_limit=fetch_limit,
                    strategy_name=str(strategy_name),
                    take_profit_pct=take_profit_pct,
                    stop_loss_pct=stop_loss_pct,
                    exit_on_ma20=exit_on_ma20,
                    use_atr_stop=use_atr_stop,
                    atr_multiplier=float(atr_multiplier),
                )
            except Exception as e:
                st.error(f"回測失敗：{e}\n\n請確認股票代號是否正確，或稍後再試。")
                return

        total_trades = result["total_trades"]
        win_rate     = result["win_rate"]
        total_return = result["total_return"]
        max_dd       = result["max_drawdown"]
        sharpe       = result["sharpe"]
        pf           = result["profit_factor"]
        max_consec   = result["max_consec_loss"]

        # ── KPI 指標卡（第一行）────────────────────────
        st.markdown(f"##### {symbol} × {strategy_name}（{year_label}）")
        m1, m2, m3 = st.columns(3)
        m1.metric("總交易次數", f"{total_trades} 次")
        m2.metric(
            "勝率",
            f"{win_rate:.1f}%",
            delta=("高勝率 ✅" if win_rate >= 50 else "偏低 ⚠️") if total_trades > 0 else None,
        )
        m3.metric(
            "累計報酬",
            f"{total_return:.2f}%",
            delta=("正報酬 ✅" if total_return > 0 else "負報酬 ❌") if total_trades > 0 else None,
        )

        # ── KPI 指標卡（第二行）────────────────────────
        r1, r2, r3 = st.columns(3)
        r1.metric("最大回撤 (MDD)", f"{max_dd:.2f}%",
                  delta=("風險低 ✅" if max_dd < 15 else "風險高 ⚠️") if total_trades > 0 else None,
                  delta_color="inverse")
        r2.metric("夏普比率", f"{sharpe:.2f}",
                  delta=("優秀 ✅" if sharpe >= 1.0 else ("尚可 🟡" if sharpe >= 0 else "不佳 ❌")) if total_trades > 0 else None)
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
        r3.metric("獲利因子", pf_str,
                  delta=("良好 ✅" if (pf == float("inf") or pf >= 1.5) else ("偏低 ⚠️")) if total_trades > 0 else None)

        if total_trades == 0:
            st.warning(
                "此期間未偵測到任何交易訊號，請嘗試：\n"
                "- 延長回測期間\n"
                "- 換一個策略或不同標的"
            )
            return

        if max_consec > 0:
            st.caption(f"最大連續虧損：{max_consec} 筆")

        # ── 資金曲線 ──────────────────────────────────
        st.markdown("---")
        equity_curve = result["equity_curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode="lines+markers",
            line=dict(color="#1976D2", width=2),
            marker=dict(size=5, color="#1976D2"),
            fill="tozeroy",
            fillcolor="rgba(25,118,210,0.08)",
            hovertemplate="第 %{x} 筆交易後<br>資金倍率：%{y:.3f}x<extra></extra>",
        ))
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="#888",
            annotation_text="初始資金",
            annotation_position="bottom right",
        )
        fig.update_layout(
            title=dict(text="資金曲線（累積倍率）", font=dict(size=13)),
            xaxis_title="交易次序",
            yaxis_title="資金倍率（x）",
            height=320,
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )
        fig.update_yaxes(gridcolor="#f0f0f0")
        st.plotly_chart(fig, use_container_width=True)

        # ── 交易明細表 ────────────────────────────────
        st.markdown("---")
        with st.expander(f"交易明細（共 {total_trades} 筆）", expanded=True):
            from typing import Any as _Any  # noqa: F401 — re-used alias for fmt dict
            trades_df = pd.DataFrame(result["trades"])
            fmt: Dict[str, _Any] = {
                "買入價": "{:.2f}", "賣出價": "{:.2f}", "損益(%)": "{:.2f}",
            }
            st.dataframe(
                trades_df.style.format(fmt),
                use_container_width=True,
                hide_index=True,
            )
