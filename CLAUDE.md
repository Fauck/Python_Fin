# 台股分析儀表板 — 專案架構說明

## 技術堆疊

| 項目 | 工具 |
|------|------|
| 語言 | Python 3.12 |
| 網頁框架 | Streamlit |
| 資料來源 | fugle-marketdata 2.4.1 |
| 圖表 | Plotly |
| 技術指標 | pandas-ta 0.4.71b0 |
| 環境變數 | python-dotenv |
| 虛擬環境 | `Fauck_env/` |

## 專案結構

```
Python_Fin/
├── app.py                 # Streamlit 進入點（只含 main()，45 行）
├── utils.py               # 共用資料層 + 技術指標計算
├── Single_stock_page.py   # 單股分析頁面模組
├── Screener_page.py       # 選股策略頁面模組
├── Score_page.py          # 個股綜合評分頁面模組
├── First.py               # 原始 API 測試腳本（勿上傳至版本控制）
├── requirements.txt       # 相依套件清單
├── .env                   # API Key（已加入 .gitignore，勿上傳）
├── .env.example           # .env 範本
├── CLAUDE.md              # 本檔：專案架構說明
└── Fauck_env/             # Python 虛擬環境
```

## 模組架構

### Import 關係（無循環）

```
app.py
  ├── Single_stock_page  →  utils
  ├── Screener_page      →  utils
  └── Score_page         →  utils
```

### 設計原則：資料層、演算法層、UI 層三層解耦

```
utils.py（共用，不含任何 Streamlit 元素）
├── get_fugle_client()
│     建立並回傳 Fugle RestClient 實例
│     讀取 FUGLE_API_KEY 環境變數
│
├── fetch_stock_candles(symbol, limit, date_from, date_to, fields)
│     透過 Historical API 取得 K 線資料
│     回傳已整理好的 pandas DataFrame
│
├── compute_ma(df, periods)
│     計算多期 SMA；新增 ma5 / ma10 / ma20 等欄位
│
└── compute_kd(df, period=9)
      台灣市場標準 KD（RSV + 1/3 EMA 平滑，初始值 50）
      新增 k_val / d_val 欄位

Single_stock_page.py
├── render_data_table(df, symbol)        DataFrame 表格
├── render_close_chart(df, symbol)       收盤價折線圖（Plotly Scatter）
├── render_candlestick_chart(df, symbol) K 線圖（Plotly Candlestick）
├── render_ohlcv_chart(df, symbol, show_ma, show_kd)
│     K線 + 均線 + 成交量 + 成交值 + KD 子圖（動態 subplots）
│     x 軸使用 type="category"，所有 x 值統一為字串 "YYYY-MM-DD"
│     含期間最高 / 最低價標註（arrowhead annotation）
└── render_single_stock_page()           單股分析頁面

Screener_page.py
├── 演算法層（純邏輯）
│   ├── check_consolidation_breakout(df, ...)  → 盤整突破第一根
│   ├── check_bullish_ma_alignment(df)         → 均線多頭排列（5/10/20MA）
│   ├── check_volume_surge_bullish(df, ...)    → 爆量長紅起漲
│   └── check_oversold_reversal(df, ...)       → 乖離過大跌深反彈
│         ↑ 所有策略函式：輸入 DataFrame，輸出 dict 或 None
├── scan_watchlist(symbols, strategy_fn, ...)
│     通用批次掃描引擎，每次呼叫間加入 time.sleep 避免 Rate Limit
│     回傳 (results, errors) tuple
├── _render_*_params()  各策略的 UI 參數控制項，回傳 (fn, fetch_limit, info)
├── STRATEGY_REGISTRY   策略名稱 → _render_*_params 的映射 dict
├── NO_RESULT_HINTS     策略名稱 → 無結果時的提示文字
└── render_screener_page()  選股策略頁面

Score_page.py
├── _SCORE_FETCH_DAYS  = 250   往前推算的日曆天數
├── _SCORE_FETCH_LIMIT = 120   最多抓取的 K 棒筆數
├── compute_score(df)
│     100 分制買進評分（趨勢 30 + 動能 30 + 震盪 20 + 量能 20）
│     使用 pandas-ta：ta.rsi() / ta.stoch() / ta.macd()
│     回傳 dict { total, dimensions, details } 或 None
├── render_radar_chart(score_result)  四維度雷達圖（Plotly Scatterpolar）
└── render_score_page()              個股綜合評分頁面

app.py（進入點，僅 45 行）
└── main()  st.set_page_config + st.tabs 導覽 + 頁面路由
```

### 頁面導覽

使用 `st.tabs` 分為三個頁面，各頁使用 `st.columns([1, 3])` 模擬左欄控制面板：

| Tab | 頁面函式 | 功能 |
|-----|---------|------|
| `📈 單股分析` | `render_single_stock_page()` | K線圖（含均線/KD）、歷史資料表 |
| `🔍 選股策略` | `render_screener_page()` | 批次掃描觀察清單（4 種策略） |
| `🎯 綜合評分` | `render_score_page()` | 100 分制買進評分 + 雷達圖 |

### 策略函式統一簽名

```python
def check_xxx(df: pd.DataFrame, **params) -> Optional[Dict[str, Any]]:
    # 輸入：已排序（日期升冪）的 DataFrame
    # 輸出：符合條件 → dict（含關鍵指標）；不符合 → None
```

新增策略時：① 在 `Screener_page.py` 實作上述函式 → ② 建立對應的 `_render_xxx_params()` → ③ 登記至 `STRATEGY_REGISTRY`

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

### 評分維度對照表

| 維度 | 滿分 | 指標 | 得分條件 |
|------|------|------|---------|
| 趨勢 Trend | 30 | 10MA / 20MA / 60MA | 收盤 > 各均線各得 10 分 |
| 動能 Momentum | 30 | RSI(14) | 40~70 或 <30 得 15 分；>80 得 0 分 |
| 動能 Momentum | — | KD(9,3,3) | K > D 得 15 分 |
| 震盪 Oscillator | 20 | MACD 柱狀圖 | Hist > 0 得 10 分 |
| 震盪 Oscillator | — | DIF / DEA | DIF > DEA 得 10 分 |
| 量能 Volume | 20 | 今日量 vs 5日均量 | 今日量 > 均量得 20 分 |

### fetch_stock_candles 參數說明

| 參數 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `symbol` | `str` | 必填 | 股票代號（例如 `"2330"`） |
| `limit` | `int` | `10` | 最多回傳幾筆交易日資料 |
| `date_from` | `str \| None` | `None` | 起始日期 `"YYYY-MM-DD"`；`None` 自動往前推 90 天 |
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

在 [Single_stock_page.py](Single_stock_page.py) `render_single_stock_page()` 取消以下註解：

```python
st.markdown("---")
st.markdown("##### 自訂日期區間（選填）")
custom_from = st.date_input("起始日期", value=None)
custom_to   = st.date_input("結束日期",  value=None)
```

再將變數傳入 `fetch_stock_candles(date_from=..., date_to=...)` 即可，`utils.py` 無需修改。

### 新增技術指標

1. 在 [utils.py](utils.py) 新增計算函式（例如 `compute_bollinger(df)`）
2. 在 `Single_stock_page.py` 的 `render_ohlcv_chart` 加入對應 trace
3. 在 `render_single_stock_page` 控制欄新增勾選框並傳入參數

### 新增選股策略

1. 在 [Screener_page.py](Screener_page.py) 實作 `check_xxx(df)` 函式
2. 實作對應的 `_render_xxx_params()` → 回傳 `(strategy_fn, fetch_limit, info_text)`
3. 在 `STRATEGY_REGISTRY` 登記新策略名稱

## 啟動方式

```bash
source Fauck_env/bin/activate
streamlit run app.py
```

---

# 全域開發與 UI 指導原則 (Global UI & Development Guidelines)

這是一個基於 Streamlit 開發的台股量化分析 Web App。
未來的每一次修改與新增功能，請務必嚴格遵守以下「行動裝置優先 (Mobile-Friendly) 與響應式 (RWD)」的 UI 設計準則，不需要我每次額外提醒：

## 1. 版面配置原則 (Layout)

* **絕對禁用全域左右分欄**：嚴禁使用 `col1, col2 = st.columns([1, 3])` 將控制面板與圖表左右拆分。在手機上會導致嚴重的排版災難與無盡的垂直滾動。
* **頂部折疊控制區**：所有的輸入框 (text_input)、選項 (radio/selectbox) 與按鈕，必須統一放在頁面最上方的 `st.expander("🔍 查詢條件設定", expanded=True)` 中。
* **內部自適應分欄**：在 `st.expander` 內部可以使用 `st.columns` 進行水平排列，Streamlit 會在手機版自動將其轉換為垂直堆疊。

## 2. 圖表自適應最佳化 (Plotly Charts)

所有的 Plotly 圖表 (`go.Figure`) 在 `update_layout` 時，必須包含以下設定，以最大化手機螢幕利用率：

* **極小化邊距**：`margin=dict(l=10, r=10, t=50, b=10)`
* **圖例水平排列**：`legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0)`，絕對不要讓圖例擠壓到右側的 K 線圖空間。
* **適應寬度**：在 Streamlit 渲染時，務必加上 `st.plotly_chart(fig, use_container_width=True)`。

## 3. 數據與表格顯示 (Metrics & Dataframes)

* **指標卡片 (st.metric)**：若有超過 3 個指標，請使用多排的 `st.columns` (例如兩個 `st.columns(4)`)，避免在手機上文字重疊破版。
* **表格 (st.dataframe)**：所有表格必須加上 `use_container_width=True` 與 `hide_index=True`，方便在手機上橫向滑動查看。

## 4. 防呆與錯誤處理

* **資料擷取**：呼叫 FinMind 或 Fugle API 時，必須妥善處理空值 (`None` 或 `NaN`)。若查無資料，請使用 `st.warning` 或 `st.info` 顯示友善提示，絕不可拋出紅字 Exception。
* **禁止數學公式語法**：在任何 UI 顯示文字或卡片說明中，禁止使用 LaTeX 語法（嚴禁使用 $ 符號包覆文字）。數值請直接轉為字串格式化，例如 `15.2%`。
