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
        # 每個交易日約 1.5 個日曆天（含週末、假日），再加 30 天緩衝
        # 例如 limit=100 → 往前推 180 天，確保拿得到足夠筆數
        days_back = max(90, int(limit * 1.5) + 30)
        date_from = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

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
