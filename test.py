import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def compute_rsi(df, period=14):
    # 計算 RSI 指標
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest_trades(df, buy_col, sell_col, initial_capital, slippage):
    """
    backtest_trades:
    - buy_col: DataFrame裡的布林欄位，為 True 表示此處產生買進訊號
    - sell_col: DataFrame裡的布林欄位，為 True 表示此處產生賣出訊號
    - initial_capital: 初始本金
    - slippage: 滑點(百分比)，例如 0.001 代表 0.1% 滑點
    """
    capital = initial_capital
    in_position = False
    entry_time = None
    entry_price = None
    trades = []
    
    for time, row in df.iterrows():
        # 判斷買進訊號
        if not in_position and row.get(buy_col, False):
            in_position = True
            entry_time = time
            # 進場價格考慮買入滑點 => close * (1 + slippage)
            entry_price = row['close'] * (1 + slippage)
        # 判斷賣出訊號
        elif in_position and row.get(sell_col, False):
            exit_time = time
            # 出場價格考慮賣出滑點 => close * (1 - slippage)
            exit_price = row['close'] * (1 - slippage)
            trade_return = exit_price / entry_price  # 報酬率
            profit = capital * (trade_return - 1)
            capital = capital * trade_return
            trades.append({
                "Entry Time": entry_time,
                "Entry Price": round(entry_price, 4),
                "Exit Time": exit_time,
                "Exit Price": round(exit_price, 4),
                "Profit": round(profit, 2)
            })
            in_position = False
    return trades, capital

def display_market_analysis(csv_file, chart_type, strategy_option, initial_capital, slippage):
    # 讀取 CSV 並轉換時間格式
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

    # Moving Average Crossover 範例
    if strategy_option == "Moving Average Crossover":
        df["MA10"] = df["close"].rolling(window=10).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()
        df["signal"] = 0
        df.loc[df["MA10"] > df["MA50"], "signal"] = 1
        df.loc[df["MA10"] < df["MA50"], "signal"] = -1
        df["signal_shifted"] = df["signal"].shift(1)
        df["buy"] = ((df["signal"] == 1) & (df["signal_shifted"] == -1))
        df["sell"] = ((df["signal"] == -1) & (df["signal_shifted"] == 1))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50', line=dict(color='orange')))
        fig.add_trace(go.Scatter(
            x=df.index[df["buy"]],
            y=df["close"][df["buy"]],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[df["sell"]],
            y=df["close"][df["sell"]],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="賣出訊號"
        ))
        buy_col = "buy"
        sell_col = "sell"
        backtest_available = True

    elif strategy_option == "SMC Strategy":
        # 1. 訂單塊判斷
        df["prev_open"] = df["open"].shift(1)
        df["prev_close"] = df["close"].shift(1)
        df["prev_high"] = df["high"].shift(1)
        df["prev_low"] = df["low"].shift(1)
        df["next_close"] = df["close"].shift(-1)
        # 看漲訂單塊
        df["bullish_OB"] = (df["prev_close"] < df["prev_open"]) & (df["next_close"] > df["prev_high"] * 1.01)
        # 看跌訂單塊
        df["bearish_OB"] = (df["prev_close"] > df["prev_open"]) & (df["next_close"] < df["prev_low"] * 0.99)
        
        # 2. 失衡區（FVG）檢測
        fvg_list = [False] * len(df)
        for i in range(1, len(df)-1):
            if df.iloc[i-1]["high"] < df.iloc[i+1]["low"]:
                fvg_list[i] = True
        df["fvg"] = fvg_list
        
        # 3. 流動性區域（利用滾動視窗計算當前區間內的最低/最高價）
        df["swing_low"] = df["low"].rolling(window=5, center=True).min()
        df["swing_high"] = df["high"].rolling(window=5, center=True).max()
        df["liq_buy"] = (df["close"] <= df["swing_low"] * 1.01)
        df["liq_sell"] = (df["close"] >= df["swing_high"] * 0.99)
        
        # 4. 市場結構轉換（CHoCH）
        df["choch_buy"] = (df["close"] > df["close"].shift(1)) & (df["prev_close"] < df["prev_open"])
        df["choch_sell"] = (df["close"] < df["close"].shift(1)) & (df["prev_close"] > df["prev_open"])
        
        # 5. 綜合SMC訊號
        df["buy_SMC"] = (df["bullish_OB"] | (df["fvg"] & df["liq_buy"])) & df["choch_buy"]
        df["sell_SMC"] = (df["bearish_OB"] | (df["fvg"] & df["liq_sell"])) & df["choch_sell"]
        
        fig.add_trace(go.Scatter(
            x=df.index[df["buy_SMC"]],
            y=df["close"][df["buy_SMC"]],
            mode="markers",
            marker=dict(symbol="triangle-up", color="darkgreen", size=10),
            name="SMC買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[df["sell_SMC"]],
            y=df["close"][df["sell_SMC"]],
            mode="markers",
            marker=dict(symbol="triangle-down", color="darkred", size=10),
            name="SMC賣出訊號"
        ))
        buy_col = "buy_SMC"
        sell_col = "sell_SMC"
        backtest_available = True

    elif strategy_option == "All Strategies":
        # 範例：以MA交叉作為「All Strategies」的基礎
        df["MA10"] = df["close"].rolling(window=10).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()
        df["signal"] = 0
        df.loc[df["MA10"] > df["MA50"], "signal"] = 1
        df.loc[df["MA10"] < df["MA50"], "signal"] = -1
        df["signal_shifted"] = df["signal"].shift(1)
        df["buy_MA"] = ((df["signal"] == 1) & (df["signal_shifted"] == -1))
        df["sell_MA"] = ((df["signal"] == -1) & (df["signal_shifted"] == 1))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50', line=dict(color='orange')))
        fig.add_trace(go.Scatter(
            x=df.index[df["buy_MA"]],
            y=df["close"][df["buy_MA"]],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="MA買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[df["sell_MA"]],
            y=df["close"][df["sell_MA"]],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="MA賣出訊號"
        ))
        buy_col = "buy_MA"
        sell_col = "sell_MA"
        backtest_available = True

    # 回測：若有可用的買賣訊號則計算交易明細與統計數據
    trade_table = None
    if backtest_available and buy_col and sell_col:
        trades, final_capital = backtest_trades(df, buy_col, sell_col, initial_capital, slippage)
        if trades:
            trade_returns = [t["Exit Price"] / t["Entry Price"] - 1 for t in trades]
            transaction_count = len(trade_returns)
            if transaction_count > 1 and np.std(trade_returns) != 0:
                sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
            else:
                sharpe_ratio = None
            trade_table = pd.DataFrame(trades)
            summary = pd.DataFrame([{
                "Initial Capital": initial_capital, 
                "Final Capital": round(final_capital, 2),
                "Total Return (%)": round((final_capital/initial_capital - 1)*100, 2),
                "Transaction Count": transaction_count,
                "Sharpe Ratio": sharpe_ratio
            }])
            trade_table = pd.concat([summary, trade_table], ignore_index=True)
        else:
            trade_table = pd.DataFrame([{"Message": "無完整交易訊號，無法進行回測計算"}])
    
    return fig, trade_table

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
            "All Strategies"
        ], label="選擇策略分析"),
        gr.Number(value=10000, label="自訂本金"),
        gr.Number(value=0.0001, label="滑點(百分比)")
    ],
    outputs=[
        gr.Plot(label="市場分析圖"),
        gr.DataFrame(label="回測交易明細")
    ],
    title="綜合市場策略分析與回測工具（加入滑點）",
    description="上傳包含市場數據的 CSV 文件，選擇圖表類型與策略模式（包含SMC策略），\n可自訂本金與滑點，展示技術指標、策略訊號及回測每筆交易獲利、交易筆數與夏普指數。"
)

if __name__ == "__main__":
    iface.launch()
