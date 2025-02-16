# K 線數據抓取與市場策略回測工具

本專案提供一個整合 K 線資料抓取、CSV 顯示以及多種交易策略回測與市場分析的工具。使用者可以透過 Gradio 圖形介面抓取指定交易對的 1 小時 K 線資料、儲存為 CSV 檔案，並利用內建的多種技術指標與策略進行回測分析。

## 功能概述

- **K 線資料抓取**
  - 從指定起始日期至現在抓取指定交易對（例如：BTC-USDT）的 1 小時 K 線資料
  - 自動將抓取到的資料轉換為 CSV 格式儲存

- **CSV 文件顯示**
  - 上傳 CSV 文件並以表格方式顯示內容

- **市場策略回測與分析**
  - 提供多種圖表展示方式（蠟燭圖、折線圖、成交量圖、移動平均、布林通道）
  - 支援多種交易策略回測，包括但不限於：
    - 移動平均交叉 (MA Crossover)
    - RSI 策略
    - 布林通道突破
    - SMC 策略
    - 多因子模型
    - 統計套利
    - 均值回歸
    - 動能交易
    - 配對交易
    - 市場微結構
    - 機器學習走勢預測 (Walk-Forward)
    - 綜合策略
    - MACD 策略
    - Ichimoku Cloud 策略
    - VWAP 策略
    - SuperTrend 策略
    - Donchian Channel 策略

- **優化與性能**
  - 使用 Pandas 向量化運算與 DataFrame 批次處理，提升資料處理效能
  - 使用 Numba 加速部分迴圈計算（如 SuperTrend 策略中的上下軌計算）

## 系統需求

- Python 3.7 以上
- 相關 Python 套件：
  - `pandas`
  - `numpy`
  - `plotly`
  - `gradio`
  - `statsmodels`
  - `scikit-learn`
  - (可選) `numba` – 用於加速部分迴圈計算
  - `bingX.perpetual.v2` – 請依照專案需求安裝或配置

## 安裝說明

1. **建立虛擬環境（推薦）**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    ```

2. **安裝所需套件**
    ```bash
    pip install pandas numpy plotly gradio statsmodels scikit-learn numba
    ```
    請根據實際需求安裝或配置 `bingX.perpetual.v2` 套件

3. **設定 API 金鑰**
   在程式碼中找到下列區塊，填入您的 API Key 與 Secret Key：
    ```python
    client = Perpetual("YOUR_API_KEY", "YOUR_SECRET_KEY")
    ```

## 使用方式

1. **啟動工具**
   執行主程式：
    ```bash
    python your_script.py
    ```
   程式將啟動 Gradio 的網頁介面，包含以下三個分頁：
   - **抓取 CSV**：輸入交易對與起始日期，抓取 K 線資料並儲存 CSV
   - **顯示 CSV**：上傳 CSV 文件並顯示內容
   - **回測分析**：上傳 CSV 文件，選擇圖表與策略，進行市場分析與回測

2. **抓取資料**
   - 在「抓取 CSV」分頁中，輸入交易對（例如：`BTC-USDT`）及起始日期（格式：`YYYY-MM-DD`），系統將自動抓取該期間的 K 線資料並儲存成 CSV 檔案

3. **查看 CSV 文件**
   - 在「顯示 CSV」分頁中，上傳剛剛生成的 CSV 文件，查看其內容

4. **進行回測分析**
   - 在「回測分析」分頁中，上傳 CSV 文件、選擇圖表類型（例如：Candlestick、Line Chart 等）與策略模式，並設定自訂本金及滑點參數，即可查看市場分析圖與回測交易明細

## 策略說明

本專案內建多種策略，每個策略使用不同的技術指標來生成買入與賣出訊號，以下是部分策略介紹：

- **移動平均交叉 (MA Crossover)**：根據短期與長期移動平均線交叉來產生交易信號
- **RSI 策略**：利用 RSI 指標判斷超買或超賣狀態
- **布林通道突破**：當價格突破布林通道上軌或下軌時產生交易訊號
- **SMC 策略**：結合多項技術指標與市場結構分析生成交易信號
- **MACD 策略**：利用 MACD 指標判斷趨勢反轉訊號
- **Ichimoku Cloud 策略**：透過一目均衡表判斷支撐與壓力位
- **VWAP 策略**：以成交量加權平均價格作為參考線
- **SuperTrend 策略**：根據 ATR 及價格動態調整趨勢線
- **Donchian Channel 策略**：根據一定時間區間內的最高與最低價格生成交易訊號

詳細的策略邏輯請參考程式碼中的各個策略函式。
