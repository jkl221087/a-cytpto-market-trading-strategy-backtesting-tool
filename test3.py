import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.decomposition import PCA

# 計算 RSI 指標
def compute_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 回測函式（考慮滑點）
def backtest_trades(df, buy_col, sell_col, initial_capital, slippage):
    """
    backtest_trades:
    - buy_col：DataFrame 中的布林欄位，True 表示該處產生買進訊號
    - sell_col：DataFrame 中的布林欄位，True 表示該處產生賣出訊號
    - initial_capital：初始本金
    - slippage：滑點（百分比），例如 0.001 代表 0.1% 滑點
    """
    capital = initial_capital
    in_position = False
    entry_time = None
    entry_price = None
    trades = []
    
    for time, row in df.iterrows():
        if not in_position and row.get(buy_col, False):
            in_position = True
            entry_time = time
            # 進場價格考慮買入滑點：收盤價 * (1 + slippage)
            entry_price = row['close'] * (1 + slippage)
        elif in_position and row.get(sell_col, False):
            exit_time = time
            # 出場價格考慮賣出滑點：收盤價 * (1 - slippage)
            exit_price = row['close'] * (1 - slippage)
            trade_return = exit_price / entry_price
            profit = capital * (trade_return - 1)
            capital = capital * trade_return
            trades.append({
                "進場時間": entry_time,
                "進場價格": round(entry_price, 4),
                "出場時間": exit_time,
                "出場價格": round(exit_price, 4),
                "獲利": round(profit, 2)
            })
            in_position = False
    return trades, capital

# 多因子模型策略：利用主成分分析(PCA)結合價值、動量與品質因子
def apply_multi_factor_model(df):
    for col in ['book_value', 'net_income', 'total_assets']:
        if col not in df.columns:
            df[col] = 1
    df['factor_value'] = df['close'] / df['book_value']
    df['factor_momentum'] = df['close'].pct_change(periods=252)
    df['factor_quality'] = df['net_income'] / df['total_assets']
    for col in ['factor_value', 'factor_momentum', 'factor_quality']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    df['composite_score'] = df['factor_value'] + df['factor_momentum'] + df['factor_quality']
    df['buy_MFM'] = df['composite_score'] > df['composite_score'].mean()
    df['sell_MFM'] = df['composite_score'] < df['composite_score'].mean()
    return df

# 統計套利策略（配對交易策略示例）
def apply_statistical_arbitrage(df):
    if not all(col in df.columns for col in ['close1', 'close2']):
        df['buy_SA'] = False
        df['sell_SA'] = False
        return df
    S1 = df['close1']
    S2 = df['close2']
    score, pvalue, _ = coint(S1, S2)
    if pvalue < 0.05:
        df['spread'] = S1 - S2
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()
        df['z_score'] = (df['spread'] - mean_spread) / std_spread
        df['buy_SA'] = df['z_score'] < -2
        df['sell_SA'] = df['z_score'] > 2
    else:
        df['buy_SA'] = False
        df['sell_SA'] = False
    return df

# 均值回歸策略：當收盤價偏離移動平均線一定標準差時產生訊號
def apply_mean_reversion(df, window=20, threshold=2):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std()
    df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    df['buy_MR'] = df['z_score'] < -threshold
    df['sell_MR'] = df['z_score'] > threshold
    return df

# 動量交易策略：根據一定期間內的價格變化判斷動量
def apply_momentum_trading(df, window=20):
    df['momentum'] = df['close'] - df['close'].shift(window)
    df['buy_MT'] = df['momentum'] > 0
    df['sell_MT'] = df['momentum'] < 0
    return df

# 配對交易策略：示例使用兩資產的收盤價（需 CSV 含 'close1' 與 'close2'）
def apply_pairs_trading(df):
    if not all(col in df.columns for col in ['close1', 'close2']):
        df['buy_PT'] = False
        df['sell_PT'] = False
        return df
    S1 = df['close1']
    S2 = df['close2']
    score, pvalue, _ = coint(S1, S2)
    if pvalue < 0.05:
        df['spread'] = S1 - S2
        mean_spread = df['spread'].mean()
        std_spread = df['spread'].std()
        df['z_score'] = (df['spread'] - mean_spread) / std_spread
        df['buy_PT'] = df['z_score'] < -2
        df['sell_PT'] = df['z_score'] > 2
    else:
        df['buy_PT'] = False
        df['sell_PT'] = False
    return df

# 市場微結構策略：利用成交量異常與當日漲跌判斷訂單流向
def apply_market_microstructure(df, window=20, vol_multiplier=1.5):
    if "volume" not in df.columns:
        df['buy_MM'] = False
        df['sell_MM'] = False
        return df
    df['vol_ma'] = df['volume'].rolling(window=window).mean()
    df['buy_MM'] = (df['volume'] > vol_multiplier * df['vol_ma']) & (df['close'] > df['open'])
    df['sell_MM'] = (df['volume'] > vol_multiplier * df['vol_ma']) & (df['close'] < df['open'])
    return df

# SMC 策略：包含訂單塊、失衡區、流動性與市場結構轉換
def apply_smc_strategy(df):
    df["prev_open"] = df["open"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["next_close"] = df["close"].shift(-1)
    df["bullish_OB"] = (df["prev_close"] < df["prev_open"]) & (df["next_close"] > df["prev_high"] * 1.01)
    df["bearish_OB"] = (df["prev_close"] > df["prev_open"]) & (df["next_close"] < df["prev_low"] * 0.99)
    fvg_list = [False] * len(df)
    for i in range(1, len(df)-1):
        if df.iloc[i-1]["high"] < df.iloc[i+1]["low"]:
            fvg_list[i] = True
    df["fvg"] = fvg_list
    df["swing_low"] = df["low"].rolling(window=5, center=True).min()
    df["swing_high"] = df["high"].rolling(window=5, center=True).max()
    df["liq_buy"] = (df["close"] <= df["swing_low"] * 1.01)
    df["liq_sell"] = (df["close"] >= df["swing_high"] * 0.99)
    df["choch_buy"] = (df["close"] > df["close"].shift(1)) & (df["prev_close"] < df["prev_open"])
    df["choch_sell"] = (df["close"] < df["close"].shift(1)) & (df["prev_close"] > df["prev_open"])
    df["buy_SMC"] = (df["bullish_OB"] | (df["fvg"] & df["liq_buy"])) & df["choch_buy"]
    df["sell_SMC"] = (df["bearish_OB"] | (df["fvg"] & df["liq_sell"])) & df["choch_sell"]
    return df

# 新增策略：資金費率套利策略
def apply_funding_rate_arbitrage(df, threshold=0.01):
    """
    假設 CSV 中包含 'funding_rate' 欄位，
    若資金費率 > threshold，則多頭成本較高，建議賣出；
    若資金費率 < -threshold，則空頭成本較高，建議買進；
    否則不產生訊號。
    """
    if "funding_rate" not in df.columns:
        df["buy_FR"] = False
        df["sell_FR"] = False
        return df
    df["buy_FR"] = df["funding_rate"] < -threshold
    df["sell_FR"] = df["funding_rate"] > threshold
    return df

# 新增策略：情緒分析策略
def apply_sentiment_strategy(df, threshold=0.5):
    """
    假設 CSV 中包含 'sentiment' 欄位，其數值介於0~1，
    當 sentiment > threshold 時，市場情緒樂觀，建議買進；
    當 sentiment < threshold 時，市場情緒悲觀，建議賣出；
    否則不產生訊號。
    """
    if "sentiment" not in df.columns:
        df["buy_sentiment"] = False
        df["sell_sentiment"] = False
        return df
    df["buy_sentiment"] = df["sentiment"] > threshold
    df["sell_sentiment"] = df["sentiment"] < threshold
    return df

# 主函式：根據使用者選擇的策略生成訊號並進行回測
def display_market_analysis(csv_file, chart_type, strategy_option, initial_capital, slippage):
    df = pd.read_csv(csv_file.name)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df.set_index("time", inplace=True)
    
    # 產生背景圖表
    if chart_type == "Candlestick":
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color='green',
            decreasing_line_color='red'
        )])
        fig.update_layout(title="蠟燭圖", yaxis_title="價格")
    elif chart_type == "Line Chart":
        fig = px.line(df, x=df.index, y="close", title="收盤價折線圖")
        fig.update_layout(xaxis_title="時間", yaxis_title="價格")
    elif chart_type == "Volume Chart":
        if "volume" in df.columns:
            fig = px.bar(df, x=df.index, y="volume", title="成交量圖")
            fig.update_layout(xaxis_title="時間", yaxis_title="成交量")
        else:
            return "CSV 文件中沒有 'volume' 欄位，無法生成成交量圖。", None
    elif chart_type == "Moving Average":
        df["MA20"] = df["close"].rolling(window=20).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode='lines', name='收盤價'))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='20日均線'))
        fig.update_layout(title="收盤價與20日移動平均線", xaxis_title="時間", yaxis_title="價格")
    elif chart_type == "Bollinger Bands":
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["STD20"] = df["close"].rolling(window=20).std()
        df["Upper"] = df["MA20"] + 2 * df["STD20"]
        df["Lower"] = df["MA20"] - 2 * df["STD20"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], mode='lines', name='收盤價'))
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], mode='lines', name='上軌', line=dict(dash='dot', color='grey')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='中軌', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], mode='lines', name='下軌', line=dict(dash='dot', color='grey')))
        fig.update_layout(title="布林通道", xaxis_title="時間", yaxis_title="價格")
    else:
        return "未知的圖表類型！", None

    # 初始化回測參數
    backtest_available = False
    trade_table = None
    buy_col = None
    sell_col = None

    # 根據使用者選擇的策略呼叫對應函式
    if strategy_option == "Moving Average Crossover":
        df["MA10"] = df["close"].rolling(window=10).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()
        df["signal"] = 0
        df.loc[df["MA10"] > df["MA50"], "signal"] = 1
        df.loc[df["MA10"] < df["MA50"], "signal"] = -1
        df["signal_shifted"] = df["signal"].shift(1)
        df["buy"] = ((df["signal"] == 1) & (df["signal_shifted"] == -1))
        df["sell"] = ((df["signal"] == -1) & (df["signal_shifted"] == 1))
        buy_col = "buy"
        sell_col = "sell"
        backtest_available = True

    elif strategy_option == "SMC Strategy":
        df = apply_smc_strategy(df)
        buy_col = "buy_SMC"
        sell_col = "sell_SMC"
        backtest_available = True

    elif strategy_option == "Multi-Factor Model":
        df = apply_multi_factor_model(df)
        buy_col = "buy_MFM"
        sell_col = "sell_MFM"
        backtest_available = True

    elif strategy_option == "Statistical Arbitrage":
        df = apply_statistical_arbitrage(df)
        buy_col = "buy_SA"
        sell_col = "sell_SA"
        backtest_available = True

    elif strategy_option == "Mean Reversion":
        df = apply_mean_reversion(df)
        buy_col = "buy_MR"
        sell_col = "sell_MR"
        backtest_available = True

    elif strategy_option == "Momentum Trading":
        df = apply_momentum_trading(df)
        buy_col = "buy_MT"
        sell_col = "sell_MT"
        backtest_available = True

    elif strategy_option == "Pairs Trading":
        df = apply_pairs_trading(df)
        buy_col = "buy_PT"
        sell_col = "sell_PT"
        backtest_available = True

    elif strategy_option == "Market Microstructure":
        df = apply_market_microstructure(df)
        buy_col = "buy_MM"
        sell_col = "sell_MM"
        backtest_available = True

    elif strategy_option == "Funding Rate Arbitrage":
        # 資金費率套利策略：假設 CSV 中包含 'funding_rate' 欄位
        df = apply_funding_rate_arbitrage(df)
        buy_col = "buy_FR"
        sell_col = "sell_FR"
        backtest_available = True

    elif strategy_option == "Sentiment Strategy":
        # 情緒分析策略：假設 CSV 中包含 'sentiment' 欄位，值介於 0~1
        df = apply_sentiment_strategy(df)
        buy_col = "buy_sentiment"
        sell_col = "sell_sentiment"
        backtest_available = True

    elif strategy_option == "Composite Strategy":
        # 組合策略：綜合 SMC、均值回歸與市場微結構
        df = apply_smc_strategy(df)
        df = apply_mean_reversion(df)
        df = apply_market_microstructure(df)
        for col in ["buy_SMC", "buy_MR", "buy_MM"]:
            if col not in df.columns:
                df[col] = False
        for col in ["sell_SMC", "sell_MR", "sell_MM"]:
            if col not in df.columns:
                df[col] = False
        df["buy_composite"] = ((df["buy_SMC"] & df["buy_MR"]) |
                               (df["buy_SMC"] & df["buy_MM"]) |
                               (df["buy_MR"] & df["buy_MM"]))
        df["sell_composite"] = ((df["sell_SMC"] & df["sell_MR"]) |
                                (df["sell_SMC"] & df["sell_MM"]) |
                                (df["sell_MR"] & df["sell_MM"]))
        buy_col = "buy_composite"
        sell_col = "sell_composite"
        backtest_available = True

    # 回測：若有可用的買賣訊號則計算交易明細與統計數據
    trade_table = None
    if backtest_available and buy_col and sell_col:
        trades, final_capital = backtest_trades(df, buy_col, sell_col, initial_capital, slippage)
        if trades:
            trade_returns = [t["出場價格"] / t["進場價格"] - 1 for t in trades]
            transaction_count = len(trade_returns)
            if transaction_count > 1 and np.std(trade_returns) != 0:
                sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
            else:
                sharpe_ratio = None
            trade_table = pd.DataFrame(trades)
            summary = pd.DataFrame([{
                "初始本金": initial_capital, 
                "最終本金": round(final_capital, 2),
                "總報酬率(%)": round((final_capital/initial_capital - 1)*100, 2),
                "交易筆數": transaction_count,
                "夏普指數": sharpe_ratio
            }])
            trade_table = pd.concat([summary, trade_table], ignore_index=True)
        else:
            trade_table = pd.DataFrame([{"訊息": "無完整交易訊號，無法進行回測計算"}])
    
    return fig, trade_table

# 新增：資金費率套利策略函式
def apply_funding_rate_arbitrage(df, threshold=0.01):
    """
    假設 CSV 中有 'funding_rate' 欄位
    若資金費率 > threshold，表示多頭需支付費用，建議賣出；
    若資金費率 < -threshold，表示空頭需支付費用，建議買入；
    否則無訊號。
    """
    if "funding_rate" not in df.columns:
        df["buy_FR"] = False
        df["sell_FR"] = False
        return df
    df["buy_FR"] = df["funding_rate"] < -threshold
    df["sell_FR"] = df["funding_rate"] > threshold
    return df

# 新增：情緒分析策略函式
def apply_sentiment_strategy(df, threshold=0.5):
    """
    假設 CSV 中有 'sentiment' 欄位，數值介於 0~1
    當 sentiment > threshold 時，市場情緒樂觀，建議買入；
    當 sentiment < threshold 時，市場情緒悲觀，建議賣出；
    否則無訊號。
    """
    if "sentiment" not in df.columns:
        df["buy_sentiment"] = False
        df["sell_sentiment"] = False
        return df
    df["buy_sentiment"] = df["sentiment"] > threshold
    df["sell_sentiment"] = df["sentiment"] < threshold
    return df

# 建立 Gradio 介面
iface = gr.Interface(
    fn=display_market_analysis,
    inputs=[
        gr.File(label="上傳 CSV 文件"),
        gr.Dropdown(choices=["Candlestick", "Line Chart", "Volume Chart", "Moving Average", "Bollinger Bands"], label="選擇圖表類型"),
        gr.Dropdown(choices=[
            "Moving Average Crossover", 
            "RSI Strategy", 
            "Bollinger Bands Breakout", 
            "MACD Strategy", 
            "Momentum Strategy", 
            "VWAP Strategy", 
            "SMC Strategy", 
            "Multi-Factor Model",
            "Statistical Arbitrage",
            "Mean Reversion",
            "Momentum Trading",
            "Pairs Trading",
            "Market Microstructure",
            "Funding Rate Arbitrage",
            "Sentiment Strategy",
            "Composite Strategy"
        ], label="選擇策略分析"),
        gr.Number(value=10000, label="自訂本金"),
        gr.Number(value=0.0001, label="滑點(百分比)")
    ],
    outputs=[
        gr.Plot(label="市場分析圖"),
        gr.DataFrame(label="回測交易明細")
    ],
    title="綜合市場策略分析與回測工具（加入滑點與多策略）",
    description="上傳包含市場數據的 CSV 文件，選擇圖表類型與策略模式（包含多因子模型、統計套利、均值回歸、動量交易、配對交易、市場微結構、資金費率套利、情緒分析及綜合策略），\n可自訂本金與滑點，展示技術指標、策略訊號及回測每筆交易獲利、交易筆數與夏普指數。"
)

if __name__ == "__main__":
    iface.launch()
