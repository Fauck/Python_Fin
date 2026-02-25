# 台股分析儀表板 — 專案架構說明

## 技術堆疊

| 項目 | 工具 |
|------|------|
| 語言 | Python 3.9 |
| 網頁框架 | Streamlit |
| 資料來源 | fugle-marketdata 2.4.1 |
| 圖表 | Plotly |
| 環境變數 | python-dotenv |
| 虛擬環境 | `Fauck_env/` |

## 專案結構

```
Python_Fin/
├── app.py            # Streamlit 主程式（核心）
├── First.py          # 原始 API 測試腳本（勿上傳至版本控制）
├── requirements.txt  # 相依套件清單
├── .env              # API Key（已加入 .gitignore，勿上傳）
├── .env.example      # .env 範本
├── CLAUDE.md         # 本檔：專案架構說明
└── Fauck_env/        # Python 虛擬環境
```

## app.py 架構

### 設計原則：資料層、演算法層、UI 層三層解耦

```
app.py
├── 資料層（不含任何 Streamlit 元素）
│   ├── get_fugle_client()
│   │     建立並回傳 Fugle RestClient 實例
│   │     讀取 FUGLE_API_KEY 環境變數
│   │
│   └── fetch_stock_candles(symbol, limit, date_from, date_to, fields)
│         透過 Historical API 取得 K 線資料
│         回傳已整理好的 pandas DataFrame
│
├── 演算法層（純邏輯，不含任何 Streamlit 元素）
│   ├── check_consolidation_breakout(df, consolidation_days, amplitude_threshold,
│   │     volume_ratio, check_volume)        → 盤整突破第一根
│   ├── check_bullish_ma_alignment(df)       → 均線多頭排列（5/10/20MA）
│   ├── check_volume_surge_bullish(df, volume_ratio, body_pct) → 爆量長紅起漲
│   ├── check_oversold_reversal(df, bias_threshold, shadow_ratio) → 乖離過大跌深反彈
│   │     ↑ 所有策略函式共享相同簽名：輸入 DataFrame，輸出 dict 或 None
│   │
│   └── scan_watchlist(symbols, strategy_fn, fetch_limit, sleep_sec, ...)
│         通用批次掃描引擎，接受任意策略函式
│         每次呼叫間加入 time.sleep 避免觸發 Rate Limit
│         回傳 (results, errors) tuple
│
└── UI 層（純渲染，不含業務邏輯）
    ├── render_data_table(df, symbol)        DataFrame 表格
    ├── render_close_chart(df, symbol)       收盤價折線圖（Plotly Scatter）
    ├── render_candlestick_chart(df, symbol) K 線圖（Plotly Candlestick）
    ├── render_single_stock_page()           單股分析頁面
    ├── render_screener_page()               盤整突破選股頁面
    └── main()                              st.tabs 導覽 + 頁面路由
```

### 頁面導覽

使用 `st.tabs` 分為兩個頁面，各頁使用 `st.columns([1, 3])` 模擬左欄控制面板：

| Tab | 頁面 | 功能 |
|-----|------|------|
| `📈 單股分析` | `render_single_stock_page()` | K線圖、走勢圖、歷史資料表 |
| `🔍 選股策略｜盤整突破` | `render_screener_page()` | 批次掃描觀察清單 |

### 策略函式統一簽名

```python
def check_xxx(df: pd.DataFrame, **params) -> Optional[Dict[str, Any]]:
    # 輸入：已排序（日期升冪）的 DataFrame
    # 輸出：符合條件 → dict（含關鍵指標）；不符合 → None
```

新增策略時：① 實作上述函式 → ② 建立對應的 `_render_xxx_params()` → ③ 登記至 `STRATEGY_REGISTRY`

### 策略參數對照表

| 策略 | 關鍵參數 | 預設值 | 調整說明 |
|------|---------|--------|---------|
| 盤整突破 | `consolidation_days` | 21 | ↑增大→更長期盤整 |
| 盤整突破 | `amplitude_threshold` | 10% | ↓減小→更嚴格（更緊密） |
| 盤整突破 | `volume_ratio` | 1.5x | ↑增大→更強量能 |
| 爆量長紅 | `volume_ratio` | 2.0x | ↑增大→要求更強爆量 |
| 爆量長紅 | `body_pct` | 3% | ↑增大→要求更大紅K實體 |
| 跌深反彈 | `bias_threshold` | -10% | ↓減小→要求更深超跌 |
| 跌深反彈 | `shadow_ratio` | 0.30 | ↑增大→要求更明顯下影線 |

### fetch_stock_candles 參數說明

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `symbol` | `str` | 必填 | 股票代號（例如 `"2330"`） |
| `limit` | `int` | `10` | 最多回傳幾筆交易日資料 |
| `date_from` | `str \| None` | `None` | 起始日期 `"YYYY-MM-DD"`；`None` 自動往前推 60 天 |
| `date_to` | `str \| None` | `None` | 結束日期 `"YYYY-MM-DD"`；`None` 為今日 |
| `fields` | `str` | `"open,high,low,close,volume"` | API 回傳欄位（逗號分隔） |

## Fugle API 用法

```python
from fugle_marketdata import RestClient

client = RestClient(api_key=os.getenv("FUGLE_API_KEY"))
raw = client.stock.historical.candles(**{
    "symbol": "2330",
    "from": "YYYY-MM-DD",
    "to": "YYYY-MM-DD",
    "fields": "open,high,low,close,volume",
})
# 回傳值為 dict（含 "data" key）或 list，需動態判斷型別
```

## 未來擴充指引

### 新增日期區間選擇器

在 [app.py](app.py) Sidebar 區塊取消以下註解（約第 167–172 行）：

```python
st.markdown("---")
st.subheader("自訂日期區間（選填）")
custom_from = st.date_input("起始日期", value=None)
custom_to   = st.date_input("結束日期",  value=None)
```

再將變數傳入 `fetch_stock_candles(date_from=..., date_to=...)` 即可，底層函式無需修改。

### 新增技術指標

1. 在 Sidebar 取消技術指標勾選框的註解（約第 175–178 行）
2. 在 `fetch_stock_candles` 回傳的 DataFrame 上計算指標（例如 `df["ma5"] = df["close"].rolling(5).mean()`）
3. 新增對應的 `render_*` 函式渲染至畫面

## 啟動方式

```bash
source Fauck_env/bin/activate
streamlit run app.py
```
