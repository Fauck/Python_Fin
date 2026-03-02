"""
財報中英對照表。
供 financial_report.py 的 _prepare_display_df（rename index）
與關鍵指標明細表（rename columns）使用。

維護指引
--------
- _STMT_INDEX_ZH：新增 yfinance 科目時，直接在對應分類下追加一行
- _METRICS_COL_ZH：若 _extract_key_metrics 新增衍生欄位，同步新增此處對應項
- 兩個 dict 均以「英文 key → 繁體中文 value」格式維護
"""

from typing import Dict


# ═════════════════════════════════════════════
# 財報科目中英對照
# 供 _prepare_display_df 使用 rename(index=...)
# 涵蓋損益表、資產負債表、現金流量表的常見 yfinance 科目名稱
# 未收錄科目保留英文原文，不會報錯
# ═════════════════════════════════════════════

STMT_INDEX_ZH: Dict[str, str] = {
    # ── 損益表 ───────────────────────────────────────────────
    "Total Revenue":                                          "總營收",
    "Revenue":                                                "營收",
    "Net Revenue":                                            "淨營收",
    "Gross Profit":                                           "毛利",
    "Cost Of Revenue":                                        "營業成本",
    "Cost Of Goods Sold":                                     "銷貨成本",
    "Operating Income":                                       "營業利益",
    "Operating Expense":                                      "營業費用",
    "Selling General Administrative":                         "銷售及管理費用",
    "Research And Development":                               "研發費用",
    "Ebitda":                                                 "EBITDA",
    "Ebit":                                                   "EBIT",
    "Net Income":                                             "淨利",
    "Net Income Common Stockholders":                         "普通股東淨利",
    "Net Income From Continuing And Discontinued Operation":  "持續與停業淨利",
    "Net Income From Continuing Operation Net Minority Interest": "持續經營淨利（少數股權後）",
    "Normalized Income":                                      "標準化淨利",
    "Diluted NI Availto Com Stockholders":                    "稀釋後可分配淨利",
    "Pretax Income":                                          "稅前淨利",
    "Tax Provision":                                          "所得稅費用",
    "Total Expenses":                                         "總費用",
    "Interest Expense":                                       "利息費用",
    "Interest Income":                                        "利息收入",
    "Net Interest Income":                                    "淨利息收入",
    "Other Income Expense":                                   "其他收支",
    "Special Income Charges":                                 "特殊收支",
    "Other Non Operating Income Expenses":                    "其他非營業收支",
    "Reconciled Cost Of Revenue":                             "調整後營業成本",
    "Reconciled Depreciation":                                "調整後折舊",
    "Total Operating Income As Reported":                     "報告期營業利益",
    "Basic EPS":                                              "基本每股盈餘",
    "Diluted EPS":                                            "稀釋每股盈餘",
    "Basic Average Shares":                                   "基本加權平均股數",
    "Diluted Average Shares":                                 "稀釋加權平均股數",
    "Normalized EBITDA":                                      "標準化 EBITDA",
    "Tax Rate For Calcs":                                     "計算稅率",

    # ── 資產負債表 ───────────────────────────────────────────
    "Total Assets":                                           "總資產",
    "Total Liabilities Net Minority Interest":                "總負債",
    "Total Equity Gross Minority Interest":                   "總股東權益",
    "Stockholders Equity":                                    "股東權益",
    "Common Stock Equity":                                    "普通股股東權益",
    "Total Capitalization":                                   "總資本",
    "Preferred Stock":                                        "優先股",
    "Common Stock":                                           "普通股",
    "Retained Earnings":                                      "保留盈餘",
    "Additional Paid In Capital":                             "資本公積",
    "Treasury Stock":                                         "庫藏股",
    "Current Assets":                                         "流動資產",
    "Current Liabilities":                                    "流動負債",
    "Cash And Cash Equivalents":                              "現金及約當現金",
    "Cash Equivalents":                                       "約當現金",
    "Cash Cash Equivalents And Short Term Investments":       "現金及短期投資",
    "Accounts Receivable":                                    "應收帳款",
    "Gross Accounts Receivable":                              "應收帳款（毛額）",
    "Allowance For Doubtful Accounts Receivable":             "備抵呆帳",
    "Net PPE":                                                "廠房設備淨額",
    "Gross PPE":                                              "廠房設備（毛額）",
    "Accumulated Depreciation":                               "累計折舊",
    "Inventory":                                              "存貨",
    "Other Current Assets":                                   "其他流動資產",
    "Other Non Current Assets":                               "其他非流動資產",
    "Total Non Current Assets":                               "非流動資產合計",
    "Investments And Advances":                               "投資及預付款",
    "Goodwill And Other Intangible Assets":                   "商譽及無形資產",
    "Goodwill":                                               "商譽",
    "Intangible Assets":                                      "無形資產",
    "Short Long Term Debt":                                   "短期借款（含一年內到期長債）",
    "Long Term Debt":                                         "長期借款",
    "Current Debt":                                           "流動借款",
    "Long Term Debt And Capital Lease Obligation":            "長期負債及融資租賃",
    "Payables And Accrued Expenses":                          "應付帳款及應計費用",
    "Accounts Payable":                                       "應付帳款",
    "Total Debt":                                             "總借款",
    "Net Debt":                                               "淨負債",
    "Working Capital":                                        "營運資金",
    "Tangible Book Value":                                    "有形帳面價值",
    "Book Value":                                             "帳面價值",
    "Share Issued":                                           "已發行股數",
    "Ordinary Shares Number":                                 "普通股股數",

    # ── 現金流量表 ───────────────────────────────────────────
    "Free Cash Flow":                                         "自由現金流",
    "Operating Cash Flow":                                    "營業活動現金流",
    "Investing Cash Flow":                                    "投資活動現金流",
    "Financing Cash Flow":                                    "籌資活動現金流",
    "Capital Expenditure":                                    "資本支出",
    "End Cash Position":                                      "期末現金",
    "Beginning Cash Position":                                "期初現金",
    "Changes In Cash":                                        "現金淨增減",
    "Effect Of Exchange Rate Changes":                        "匯率影響",
    "Cash Dividends Paid":                                    "已支付現金股利",
    "Common Stock Dividend Paid":                             "普通股股利支出",
    "Repurchase Of Capital Stock":                            "庫藏股買回",
    "Repayment Of Debt":                                      "借款償還",
    "Issuance Of Debt":                                       "借款舉借",
    "Net Common Stock Issuance":                              "普通股淨發行",
    "Long Term Debt Issuance":                                "長期債務發行",
    "Long Term Debt Payments":                                "長期債務償還",
    "Short Term Debt Issuance":                               "短期債務發行",
    "Short Term Debt Payments":                               "短期債務償還",
    "Proceeds From Stock Option Exercised":                   "認股權行使所得",
    "Net Other Financing Charges":                            "其他籌資費用淨額",
    "Purchase Of Investment":                                 "購買投資",
    "Sale Of Investment":                                     "出售投資",
    "Net Investment Purchase Activity":                       "投資淨購買活動",
    "Purchase Of Business":                                   "企業購併支出",
    "Purchase Of PPE":                                        "購置廠房設備",
    "Net Other Investing Changes":                            "其他投資活動淨變動",
    "Change In Working Capital":                              "營運資金變動",
    "Change In Receivables":                                  "應收帳款變動",
    "Changes In Account Receivables":                         "應收帳款變動",
    "Change In Inventory":                                    "存貨變動",
    "Change In Payable":                                      "應付帳款變動",
    "Change In Payables And Accrued Expense":                 "應付帳款及費用變動",
    "Change In Other Current Liabilities":                    "其他流動負債變動",
    "Change In Other Current Assets":                         "其他流動資產變動",
    "Depreciation And Amortization":                          "折舊及攤銷",
    "Depreciation Amortization Depletion":                    "折舊、攤銷及耗竭",
    "Deferred Tax":                                           "遞延所得稅",
    "Deferred Income Tax":                                    "遞延所得稅費用",
    "Stock Based Compensation":                               "股份報酬費用",
    "Net Income From Continuing Operations":                  "持續經營淨利",
    "Gain Loss On Investment Securities":                     "投資證券損益",
    "Gain Loss On Sale Of Business":                          "出售事業損益",
    "Asset Impairment Charge":                                "資產減損費用",
    "Other Non Cash Items":                                   "其他非現金項目",
    "Other Operating Cashflow":                               "其他營業現金流",
    "Interest Paid Supplemental Data":                        "已支付利息",
    "Income Tax Paid Supplemental Data":                      "已支付所得稅",
    "Interest Received Cfo":                                  "已收利息",
    "Dividends Received Cfo":                                 "已收股利",
    "Taxes Refund Paid":                                      "退稅及已付稅",
    "Amortization Of Securities":                             "有價證券攤銷",
    "Provisionand Write Offof Assets":                        "資產提列及沖銷",
}


# ═════════════════════════════════════════════
# 關鍵指標 DataFrame 欄位名對照
# 供 render_financial_page 的明細表使用 rename(columns=...)
# ═════════════════════════════════════════════

METRICS_COL_ZH: Dict[str, str] = {
    "period":         "期別",
    "revenue":        "總營收",
    "gross_profit":   "毛利",
    "net_income":     "淨利",
    "gross_margin":   "毛利率 (%)",
    "net_margin":     "淨利率 (%)",
    "revenue_growth": "營收成長率 (%)",
}
