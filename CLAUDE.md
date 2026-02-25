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

### 設計原則：資料層與 UI 層完全解耦

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
└── UI 層（純渲染，不含業務邏輯）
    ├── render_data_table(df, symbol)
    │     以 DataFrame 表格顯示歷史價格資料
    │
    ├── render_close_chart(df, symbol)
    │     繪製收盤價折線走勢圖（Plotly Scatter）
    │
    ├── render_candlestick_chart(df, symbol)
    │     繪製 K 線圖（Plotly Candlestick）
    │
    └── main()
          Streamlit 進入點，負責 Sidebar 參數收集與流程控制
```

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
