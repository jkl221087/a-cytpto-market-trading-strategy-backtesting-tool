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


def backtest_trades(df, buy_col, sell_col, initial_capital):
    capital = initial_capital
    in_position = False
    entry_time = None
    entry_price = None
    trades = []


    for time, row in df.iterrows():
        if not in_position and row.get(buy_col, False):
            in_position = True
            entry_time = time
            entry_price = row['close']
        elif in_position and row.get(sell_col, False):
            exit_time = time
            exit_price = row['close']
            trade_return = exit_price / entry_price
            profit = capital * (trade_return -1)
            capital = capital * trade_return
            trades.append({
                "Entry Time":entry_time,
                "Entry Price" :entry_price,
                "Exit Time":exit_time,
                "Exit Price": exit_price,
                "Profit": round(profit, 2)
            })
            in_position = False
    return trades, capital





def display_market_analysis(csv_file, chart_type, strategy_option, initial_capital):
    # 讀取 CSV 並轉換時間格式
    df = pd.read_csv(csv_file.name)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    df.set_index("time", inplace=True)
    
    # 根據 chart_type 決定圖表背景，預設以蠟燭圖作為背景
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
            return "CSV 文件中沒有 'volume' 欄位，無法生成成交量圖。"
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
        return "未知的圖表類型！"
    
    # 根據所選策略在圖表上加入買賣訊號
    if strategy_option == "Moving Average Crossover":
        # 移動平均交叉策略 (例：10日與50日)
        df["MA10"] = df["close"].rolling(window=10).mean()
        df["MA50"] = df["close"].rolling(window=50).mean()
        # 計算訊號：MA10 大於 MA50 為多頭，反之為空頭
        df["signal"] = 0
        df.loc[df["MA10"] > df["MA50"], "signal"] = 1
        df.loc[df["MA10"] < df["MA50"], "signal"] = -1
        df["signal_shifted"] = df["signal"].shift(1)
        # 當訊號由空轉多 => 買進；由多轉空 => 賣出
        df["buy"] = ((df["signal"] == 1) & (df["signal_shifted"] == -1))
        df["sell"] = ((df["signal"] == -1) & (df["signal_shifted"] == 1))
        # 將移動平均線加入圖中
        fig.add_trace(go.Scatter(x=df.index, y=df["MA10"], mode='lines', name='MA10', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='MA50', line=dict(color='orange')))
        # 標示買賣訊號
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
        
    elif strategy_option == "RSI Strategy":
        # RSI 策略：RSI < 30 為買進訊號，RSI > 70 為賣出訊號
        df["RSI"] = compute_rsi(df)
        buy_signals = df[df["RSI"] < 30]
        sell_signals = df[df["RSI"] > 70]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="RSI買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="RSI賣出訊號"
        ))
        
    elif strategy_option == "Bollinger Bands Breakout":
        # 布林通道突破策略：價格跌破下軌買進，價格突破上軌賣出
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["STD20"] = df["close"].rolling(window=20).std()
        df["Upper"] = df["MA20"] + 2 * df["STD20"]
        df["Lower"] = df["MA20"] - 2 * df["STD20"]
        # 加入布林通道線
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], mode='lines', name='上軌', line=dict(dash='dot', color='grey')))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode='lines', name='中軌', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], mode='lines', name='下軌', line=dict(dash='dot', color='grey')))
        # 標示訊號：價格低於下軌買進，高於上軌賣出
        buy_signals = df[df["close"] < df["Lower"]]
        sell_signals = df[df["close"] > df["Upper"]]
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="布林買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=sell_signals.index,
            y=sell_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="布林賣出訊號"
        ))
        
    elif strategy_option == "All Strategies":
        # 綜合策略：以移動平均交叉為主，並加入 RSI 與布林訊號輔助顯示
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
        # RSI 與布林訊號輔助顯示（不參與回測計算）
        df["RSI"] = compute_rsi(df)
        fig.add_trace(go.Scatter(
            x=df.index[df["RSI"] < 30],
            y=df["close"][df["RSI"] < 30],
            mode="markers",
            marker=dict(symbol="star", color="green", size=10),
            name="RSI買進訊號"
        ))
        fig.add_trace(go.Scatter(
            x=df.index[df["RSI"] > 70],
            y=df["close"][df["RSI"] > 70],
            mode="markers",
            marker=dict(symbol="star", color="red", size=10),
            name="RSI賣出訊號"
        ))
        df["STD20"] = df["close"].rolling(window=20).std()
        df["Upper"] = df["MA10"] + 2 * df["STD20"]
        df["Lower"] = df["MA10"] - 2 * df["STD20"]
        fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], mode='lines', name='上軌', line=dict(dash='dot', color='grey')))
        fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], mode='lines', name='下軌', line=dict(dash='dot', color='grey')))
        buy_col = "buy_MA"
        sell_col = "sell_MA"
        backtest_available = True


        trade_table = None

        if backtest_available and buy_col and sell_col:
            trades, final_capital = backtest_trades(df, buy_col, sell_col, initial_capital)
            if trades:
                trade_returns = [trade["Exit Price"] / trade["Entry Price"] - 1 for trade in trades]
                transaction_count = len(trade_returns)
                if transaction_count > 1 and np.std(trade_returns) != 0:
                    sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
                else:
                    sharpe_ratio = None
                trade_table = pd.DataFrame(trades)

                summary = pd.DataFrame([{"Initial Capital": initial_capital, 
                                    "Final Capital": round(final_capital, 2),
                                    "Total Return (%)": round((final_capital/initial_capital - 1)*100, 2),
                                    "Transaction Count": transaction_count,
                                    "Sharpe Ratio": sharpe_ratio
                                    }])
                
                
                trade_table = pd.concat([summary, trade_table], ignore_index=True)
            else:
                trade_table = pd.DataFrame([{"Message": "無完整交易訊號，無法進行回測計算"}])
        return fig, trade_table


# 建立 Gradio 介面，使用者可選擇圖表類型與策略分析
iface = gr.Interface(
    fn=display_market_analysis,
    inputs=[
        gr.File(label="上傳 CSV 文件"),
        gr.Dropdown(choices=["Candlestick", "Line Chart", "Volume Chart", "Moving Average", "Bollinger Bands"], label="選擇圖表類型"),
        gr.Dropdown(choices=["Moving Average Crossover", "RSI Strategy", "Bollinger Bands Breakout", "All Strategies"], label="選擇策略分析"),
        gr.Number(value=10000, label="自訂本金")
    ],
    outputs=[
        gr.Plot(label="市場分析圖"),
        gr.DataFrame(label="回測交易明細")
    ],
    title="綜合市場策略分析與回測工具",
    description="上傳包含市場數據的 CSV 文件，選擇圖表類型與策略模式，展示技術指標、策略訊號及回測每筆交易獲利情形。"
)

if __name__ == "__main__":
    iface.launch()
