"""
共用資料層與演算法層（技術指標計算）。
被 Single_stock_page.py、Screener_page.py、Score_page.py 共同引用。
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from fugle_marketdata import RestClient

load_dotenv()

# Fugle API 單次查詢上限（API 要求 < 365 天，保留 5 天緩衝）
_FUGLE_MAX_RANGE_DAYS = 360


# ═════════════════════════════════════════════
# 資料層：API 呼叫邏輯（與 UI 完全解耦）
# ═════════════════════════════════════════════

def get_fugle_client() -> RestClient:
    """建立並回傳 Fugle RestClient 實例。"""
    api_key = os.getenv("FUGLE_API_KEY")
    if not api_key:
        raise ValueError("找不到 FUGLE_API_KEY，請確認 .env 檔案設定。")
    return RestClient(api_key=api_key)


def _fetch_chunk(
    client: RestClient,
    symbol: str,
    date_from: str,
    date_to: str,
    fields: str,
) -> List[dict]:
    """單次 Fugle API 呼叫（日期範圍需 < 365 天），回傳原始 record list。"""
    raw = client.stock.historical.candles(
        **{"symbol": symbol, "from": date_from, "to": date_to, "fields": fields}
    )
    if isinstance(raw, dict):
        return list(raw.get("data", []))
    if isinstance(raw, list):
        return list(raw)
    return []


def fetch_stock_candles(
    symbol: str,
    limit: int = 10,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    fields: str = "open,high,low,close,volume",
) -> pd.DataFrame:
    """
    透過 Fugle Historical API 取得股票 K 線資料。
    若所需日期範圍超過 API 上限（365 天），自動分段抓取並合併。

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
        # 每個交易日約 1.5 個日曆天（含週末、假日），再加 30 天緩衝
        # 例如 limit=300 → 往前推 480 天（需分兩段抓取）
        days_back = max(90, int(limit * 1.5) + 30)
        date_from = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    dt_from = datetime.strptime(date_from, "%Y-%m-%d")
    dt_to   = datetime.strptime(date_to,   "%Y-%m-%d")

    # ── 分段抓取（Fugle API 限制：單次查詢 < 365 天）──
    all_records: List[dict] = []
    chunk_end = dt_to
    while chunk_end >= dt_from:
        chunk_start = max(dt_from, chunk_end - timedelta(days=_FUGLE_MAX_RANGE_DAYS - 1))
        all_records.extend(
            _fetch_chunk(
                client, symbol,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                fields,
            )
        )
        if chunk_start <= dt_from:
            break
        chunk_end = chunk_start - timedelta(days=1)

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # 統一日期欄位名稱
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col and date_col != "date":
        df = df.rename(columns={date_col: "date"})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = (df.drop_duplicates(subset="date")
                .sort_values("date")
                .reset_index(drop=True))

    return df.tail(limit).reset_index(drop=True)


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


def compute_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    計算布林通道（Bollinger Bands）。

    Parameters
    ----------
    df      : 含 close 欄位的 DataFrame
    period  : 計算週期（預設 20）
    std_dev : 標準差倍數（預設 2.0）

    Returns
    -------
    含 bb_mid / bb_upper / bb_lower / bb_width 新欄位的 DataFrame 副本
    bb_width = (bb_upper - bb_lower) / bb_mid（帶寬，越小表示越擠壓）
    """
    df = df.copy()
    rolling        = df["close"].rolling(period)
    df["bb_mid"]   = rolling.mean()
    df["bb_upper"] = df["bb_mid"] + std_dev * rolling.std()
    df["bb_lower"] = df["bb_mid"] - std_dev * rolling.std()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    計算平均真實範圍（ATR, Average True Range）。

    公式：
      TR  = max(High − Low,  |High − PrevClose|,  |Low − PrevClose|)
      ATR = EWM(TR, α = 1/period)  （Wilder 平滑法）

    Parameters
    ----------
    df     : 含 high / low / close 欄位的 DataFrame（日期升冪）
    period : ATR 計算週期，預設 14

    Returns
    -------
    含 atr 新欄位的 DataFrame 副本
    """
    df = df.copy()
    prev_close = df["close"].shift(1)
    high_low   = df["high"] - df["low"]
    high_prev  = (df["high"] - prev_close).abs()
    low_prev   = (df["low"]  - prev_close).abs()
    tr         = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    df["atr"]  = tr.ewm(alpha=1.0 / period, adjust=False).mean()
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


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    計算相對強弱指標（RSI, Relative Strength Index）。

    公式（Wilder 平滑，與 TradingView 一致）：
      Delta  = Close(t) − Close(t-1)
      Gain   = Delta.clip(0)
      Loss   = (−Delta).clip(0)
      AvgG   = EWM(Gain, α = 1/period, adjust=False)
      AvgL   = EWM(Loss, α = 1/period, adjust=False)
      RSI    = 100 − 100 / (1 + AvgG / AvgL)

    Parameters
    ----------
    df     : 含 close 欄位的 DataFrame（日期升冪）
    period : RSI 計算週期，預設 14

    Returns
    -------
    含 rsi_14（或 rsi_{period}）新欄位的 DataFrame 副本
    """
    df    = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    alpha    = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, float("inf"))
    rsi = (100.0 - 100.0 / (1.0 + rs)).clip(0, 100).round(2)

    col_name      = f"rsi_{period}" if period != 14 else "rsi_14"
    df[col_name]  = rsi
    return df


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    計算 MACD 指標（DIF / DEA / Histogram）。

    公式：
      EMA_fast   = EMA(Close, fast)
      EMA_slow   = EMA(Close, slow)
      MACD line  = EMA_fast − EMA_slow       （DIF）
      Signal     = EMA(MACD line, signal)    （DEA）
      Histogram  = MACD line − Signal

    Parameters
    ----------
    df     : 含 close 欄位的 DataFrame（日期升冪）
    fast   : 快線 EMA 週期（預設 12）
    slow   : 慢線 EMA 週期（預設 26）
    signal : 訊號線 EMA 週期（預設 9）

    Returns
    -------
    含 macd_line / macd_signal / macd_hist 新欄位的 DataFrame 副本
    """
    df               = df.copy()
    ema_fast         = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow         = df["close"].ewm(span=slow,   adjust=False).mean()
    df["macd_line"]   = (ema_fast - ema_slow).round(4)
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False).mean().round(4)
    df["macd_hist"]   = (df["macd_line"] - df["macd_signal"]).round(4)
    return df
