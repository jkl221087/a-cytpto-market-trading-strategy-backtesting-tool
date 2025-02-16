import time
import datetime
import csv
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from statsmodels.tsa.stattools import coint
from sklearn.ensemble import RandomForestClassifier
# 若需要 PCA，可取消下一行註解
# from sklearn.decomposition import PCA
from bingX.perpetual.v2 import Perpetual

# 若有安裝 numba，可加速部分迭代運算
try:
    from numba import njit
except ImportError:
    def njit(func):
        return func

# ---------------------- 輔助函式：數值清洗 ----------------------
def clean_indicators(df):
    """
    將 Inf/-Inf 替換為 NaN，並用向後填補 (bfill) 清洗資料，
    以避免計算中出現錯誤。
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.bfill()
    return df

# ---------------------- 錯誤圖表函式 ----------------------
def error_plot(message):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(title="錯誤")
    return fig

# ---------------------- K 線數據抓取工具 ----------------------
client = Perpetual("YOUR_API_KEY", "YOUR_SECRET_KEY")

def fetch_kline_1h(symbol: str, start_ts: int, end_ts: int, limit: int = 1000):
    try:
        return client.kline(
            symbol=symbol,
            interval="1h",
            startTime=start_ts,
            endTime=end_ts,
            limit=limit
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_and_save(symbol: str, start_date_str: str):
    try:
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        return None, "起始日期格式錯誤，請使用 YYYY-MM-DD 格式"
    
    end_date = datetime.datetime.now()
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_kline_data = []
    current_ts = start_ts
    limit_per_call = 1000
    
    while current_ts < end_ts:
        batch_end_ts = min(current_ts + (1000 * 3600 * 1000), end_ts)
        kline_data = fetch_kline_1h(symbol, current_ts, batch_end_ts, limit=limit_per_call)
        if not kline_data or len(kline_data) == 0:
            print(f"No data returned for period: {datetime.datetime.fromtimestamp(current_ts/1000)} - {datetime.datetime.fromtimestamp(batch_end_ts/1000)}")
            break
        all_kline_data.extend(kline_data)
        print(f"Fetched {len(kline_data)} records. Total records: {len(all_kline_data)}")
        # 下一個批次從最後一筆時間加一小時開始
        current_ts = kline_data[-1]['time'] + 3600000  
        time.sleep(0.5)
    
    if not all_kline_data:
        return None, "未能獲取任何數據"
    
    # 利用 DataFrame 進行批次處理（更快且更簡潔）
    df = pd.DataFrame(all_kline_data)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    filename = f"{symbol.replace('-', '_')}_kline_data.csv"
    df.to_csv(filename, index=False, encoding="utf-8")
    
    status_message = f"共抓取 {len(all_kline_data)} 根 K 線資料，CSV 檔案已儲存：{filename}"
    return filename, status_message

# ---------------------- CSV 顯示工具 ----------------------
def show_csv(file):
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file.name)
        return df
    except Exception as e:
        return pd.DataFrame({'error': [f'無法讀取文件: {str(e)}']})

# ---------------------- 回測與策略函式 ----------------------
def compute_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = np.where(avg_loss == 0, np.nan, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = np.where(np.isnan(rsi), 100, rsi)
    return df['RSI']

def backtest_trades(df, buy_col, sell_col, initial_capital, slippage=0.0):
    capital = initial_capital
    in_position = False
    entry_time = None
    entry_price = None
    trades = []
    
    # 這裡使用 DataFrame.iterrows，如資料量大可考慮向量化或使用 itertuples()
    for time_idx, row in df.iterrows():
        try:
            if not in_position and row.get(buy_col, False):
                in_position = True
                entry_time = time_idx
                entry_price = row['close'] * (1 + slippage)
            elif in_position and row.get(sell_col, False):
                exit_time = time_idx
                exit_price = row['close'] * (1 - slippage)
                trade_return = exit_price / entry_price
                profit = capital * (trade_return - 1)
                capital *= trade_return
                trades.append({
                    "進場時間": entry_time,
                    "進場價格": round(entry_price, 4),
                    "出場時間": exit_time,
                    "出場價格": round(exit_price, 4),
                    "獲利": round(profit, 2)
                })
                in_position = False
        except Exception as e:
            print(f"交易回測錯誤 at {time_idx}: {e}")
    
    if in_position:
        final_time = df.index[-1]
        final_price = df.iloc[-1]['close'] * (1 - slippage)
        trade_return = final_price / entry_price
        profit = capital * (trade_return - 1)
        capital *= trade_return
        trades.append({
            "進場時間": entry_time,
            "進場價格": round(entry_price, 4),
            "出場時間": final_time,
            "出場價格": round(final_price, 4),
            "獲利": round(profit, 2)
        })
    
    return trades, capital

def apply_ml_strategy_walk_forward(df, train_window=252):
    df['return'] = df['close'].pct_change()
    df['lag1'] = df['return'].shift(1)
    df['lag2'] = df['return'].shift(2)
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['ma_diff'] = df['close'] - df['MA10']
    df['RSI'] = compute_rsi(df)
    df.dropna(inplace=True)
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    df['pred'] = np.nan
    df['buy_ML_WF'] = False
    df['sell_ML_WF'] = False
    features = ['lag1', 'lag2', 'ma_diff', 'RSI']
    dates = df.index.sort_values()
    for i in range(train_window, len(dates) - 1):
        train_start = dates[i - train_window]
        train_end = dates[i - 1]
        test_date = dates[i]
        train_mask = (df.index >= train_start) & (df.index < train_end)
        test_mask = (df.index == test_date)
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'target']
        if len(X_train) < 10:
            continue
        X_test = df.loc[test_mask, features]
        if len(X_test) == 0:
            continue
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        df.loc[test_mask, 'pred'] = pred[0]
    df.loc[df['pred'] == 1, 'buy_ML_WF'] = True
    df.loc[df['pred'] == 0, 'sell_ML_WF'] = True
    return clean_indicators(df)

def apply_ma_crossover(df):
    df["MA10"] = df["close"].rolling(window=10).mean()
    df["MA50"] = df["close"].rolling(window=50).mean()
    df["signal"] = 0
    df.loc[df["MA10"] > df["MA50"], "signal"] = 1
    df.loc[df["MA10"] < df["MA50"], "signal"] = -1
    df["signal_shifted"] = df["signal"].shift(1)
    df["buy_MA"] = ((df["signal"] == 1) & (df["signal_shifted"] == -1))
    df["sell_MA"] = ((df["signal"] == -1) & (df["signal_shifted"] == 1))
    return clean_indicators(df)

def apply_multi_factor_model(df):
    for col in ['book_value', 'net_income', 'total_assets']:
        if col not in df.columns:
            df[col] = 1
    df['factor_value'] = df['close'] / df['book_value']
    df['factor_momentum'] = df['close'].pct_change(periods=252)
    df['factor_quality'] = df['net_income'] / df['total_assets']
    for col in ['factor_value', 'factor_momentum', 'factor_quality']:
        std_val = df[col].std()
        if std_val == 0:
            df[col] = (df[col] - df[col].mean())
        else:
            df[col] = (df[col] - df[col].mean()) / std_val
    df['composite_score'] = df['factor_value'] + df['factor_momentum'] + df['factor_quality']
    mean_score = df['composite_score'].mean()
    df['buy_MFM'] = df['composite_score'] > mean_score
    df['sell_MFM'] = df['composite_score'] < mean_score
    return clean_indicators(df)

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
        df['z_score'] = np.where(std_spread==0, 0, (df['spread'] - mean_spread) / std_spread)
        df['buy_SA'] = df['z_score'] < -2
        df['sell_SA'] = df['z_score'] > 2
    else:
        df['buy_SA'] = False
        df['sell_SA'] = False
    return clean_indicators(df)

def apply_mean_reversion(df, window=20, threshold=2):
    df['rolling_mean'] = df['close'].rolling(window=window).mean()
    df['rolling_std'] = df['close'].rolling(window=window).std().replace(0, np.nan)
    df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    df['buy_MR'] = df['z_score'] < -threshold
    df['sell_MR'] = df['z_score'] > threshold
    return clean_indicators(df)

def apply_momentum_trading(df, window=20):
    df['momentum'] = df['close'] - df['close'].shift(window)
    df['buy_MT'] = df['momentum'] > 0
    df['sell_MT'] = df['momentum'] < 0
    return clean_indicators(df)

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
        df['z_score'] = np.where(std_spread==0, 0, (df['spread'] - mean_spread) / std_spread)
        df['buy_PT'] = df['z_score'] < -2
        df['sell_PT'] = df['z_score'] > 2
    else:
        df['buy_PT'] = False
        df['sell_PT'] = False
    return clean_indicators(df)

def apply_market_microstructure(df, window=20, vol_multiplier=1.5):
    if "volume" not in df.columns:
        df['buy_MM'] = False
        df['sell_MM'] = False
        return df
    df['vol_ma'] = df['volume'].rolling(window=window).mean()
    df['buy_MM'] = (df['volume'] > vol_multiplier * df['vol_ma']) & (df['close'] > df['open'])
    df['sell_MM'] = (df['volume'] > vol_multiplier * df['vol_ma']) & (df['close'] < df['open'])
    return clean_indicators(df)

def apply_smc_strategy(df):
    df["prev_open"] = df["open"].shift(1)
    df["prev_close"] = df["close"].shift(1)
    df["prev_high"] = df["high"].shift(1)
    df["prev_low"] = df["low"].shift(1)
    df["next_close"] = df["close"].shift(-1)
    df["bullish_OB"] = (df["prev_close"] < df["prev_open"]) & (df["next_close"] > df["prev_high"] * 1.01)
    df["bearish_OB"] = (df["prev_close"] > df["prev_open"]) & (df["next_close"] < df["prev_low"] * 0.99)
    # 優化前：使用迴圈計算 fvg
    # fvg_list = [False] * len(df)
    # for i in range(1, len(df)-1):
    #     if df.iloc[i-1]["high"] < df.iloc[i+1]["low"]:
    #         fvg_list[i] = True
    # df["fvg"] = fvg_list
    # 改為向量化計算：利用 shift()
    df["fvg"] = (df["high"].shift(1) < df["low"].shift(-1)).fillna(False)
    df["swing_low"] = df["low"].rolling(window=5, center=True).min()
    df["swing_high"] = df["high"].rolling(window=5, center=True).max()
    df["liq_buy"] = (df["close"] <= df["swing_low"] * 1.01)
    df["liq_sell"] = (df["close"] >= df["swing_high"] * 0.99)
    df["choch_buy"] = (df["close"] > df["close"].shift(1)) & (df["prev_close"] < df["prev_open"])
    df["choch_sell"] = (df["close"] < df["close"].shift(1)) & (df["prev_close"] > df["prev_open"])
    df["buy_SMC"] = (df["bullish_OB"] | (df["fvg"] & df["liq_buy"])) & df["choch_buy"]
    df["sell_SMC"] = (df["bearish_OB"] | (df["fvg"] & df["liq_sell"])) & df["choch_sell"]
    return clean_indicators(df)

def apply_composite_strategy(df):
    df = apply_smc_strategy(df)
    df = apply_mean_reversion(df)
    df = apply_market_microstructure(df)
    df = apply_ml_strategy_walk_forward(df, train_window=252)
    for col in ["buy_SMC", "buy_MR", "buy_MM", "buy_ML_WF"]:
        if col not in df.columns:
            df[col] = False
    for col in ["sell_SMC", "sell_MR", "sell_MM", "sell_ML_WF"]:
        if col not in df.columns:
            df[col] = False
    df["buy_composite"] = ((df["buy_SMC"] & df["buy_MR"]) |
                           (df["buy_SMC"] & df["buy_MM"]) |
                           (df["buy_SMC"] & df["buy_ML_WF"]) |
                           (df["buy_MR"] & df["buy_MM"]) |
                           (df["buy_MR"] & df["buy_ML_WF"]) |
                           (df["buy_MM"] & df["buy_ML_WF"]))
    df["sell_composite"] = ((df["sell_SMC"] & df["sell_MR"]) |
                            (df["sell_SMC"] & df["sell_MM"]) |
                            (df["sell_SMC"] & df["sell_ML_WF"]) |
                            (df["sell_MR"] & df["sell_MM"]) |
                            (df["sell_MR"] & df["sell_ML_WF"]) |
                            (df["sell_MM"] & df["sell_ML_WF"]))
    return clean_indicators(df)

# ===== 新增交易策略 =====

def apply_macd_strategy(df, short_period=12, long_period=26, signal_period=9):
    try:
        df['ema_short'] = df['close'].ewm(span=short_period, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=long_period, adjust=False).mean()
        df['macd'] = df['ema_short'] - df['ema_long']
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['buy_MACD'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        df['sell_MACD'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    except Exception as e:
        print(f"MACD 策略錯誤: {e}")
    return clean_indicators(df)

def apply_ichimoku_strategy(df):
    try:
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
        df['buy_ichimoku'] = (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b']) & (df['tenkan_sen'] > df['kijun_sen'])
        df['sell_ichimoku'] = (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b']) & (df['tenkan_sen'] < df['kijun_sen'])
    except Exception as e:
        print(f"Ichimoku 策略錯誤: {e}")
    return clean_indicators(df)

def apply_vwap_strategy(df):
    try:
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_vp'] = (df['typical_price'] * df['volume']).cumsum()
        df['vwap'] = df['cum_vp'] / df['cum_vol']
        df['buy_VWAP'] = df['close'] > df['vwap']
        df['sell_VWAP'] = df['close'] < df['vwap']
    except Exception as e:
        print(f"VWAP 策略錯誤: {e}")
    return clean_indicators(df)

# --- 使用 Numba 優化 SuperTrend 中的迭代計算 ---
@njit
def compute_final_bands(close, basic_upperband, basic_lowerband):
    n = len(close)
    final_upperband = np.empty(n)
    final_lowerband = np.empty(n)
    final_upperband[0] = basic_upperband[0]
    final_lowerband[0] = basic_lowerband[0]
    for i in range(1, n):
        if close[i-1] <= final_upperband[i-1]:
            final_upperband[i] = min(basic_upperband[i], final_upperband[i-1])
        else:
            final_upperband[i] = basic_upperband[i]
        if close[i-1] >= final_lowerband[i-1]:
            final_lowerband[i] = max(basic_lowerband[i], final_lowerband[i-1])
        else:
            final_lowerband[i] = basic_lowerband[i]
    return final_upperband, final_lowerband

def apply_supertrend_strategy(df, period=7, multiplier=3):
    try:
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift(1))
        df['low_close'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        df['basic_upperband'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
        df['basic_lowerband'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
        
        # 利用向量化方式計算 final bands，使用 numba 加速（若 numba 不可用則可保留原迴圈方式）
        basic_upperband = df['basic_upperband'].values
        basic_lowerband = df['basic_lowerband'].values
        close = df['close'].values
        final_upperband, final_lowerband = compute_final_bands(close, basic_upperband, basic_lowerband)
        df['final_upperband'] = final_upperband
        df['final_lowerband'] = final_lowerband
        
        # 計算 supertrend（這裡用向量化條件判斷）
        supertrend = np.empty(len(df))
        for i in range(period, len(df)):
            if close[i] <= final_upperband[i]:
                supertrend[i] = final_upperband[i]
            else:
                supertrend[i] = final_lowerband[i]
        df['supertrend'] = supertrend
        
        df['buy_supertrend'] = (df['close'] > df['supertrend']) & (df['close'].shift(1) <= df['supertrend'].shift(1))
        df['sell_supertrend'] = (df['close'] < df['supertrend']) & (df['close'].shift(1) >= df['supertrend'].shift(1))
    except Exception as e:
        print(f"SuperTrend 策略錯誤: {e}")
    return clean_indicators(df)

def apply_donchian_channel_strategy(df, window=20):
    try:
        df['donchian_upper'] = df['high'].rolling(window=window).max()
        df['donchian_lower'] = df['low'].rolling(window=window).min()
        df['buy_donchian'] = df['close'] > df['donchian_upper'].shift(1)
        df['sell_donchian'] = df['close'] < df['donchian_lower'].shift(1)
    except Exception as e:
        print(f"Donchian Channel 策略錯誤: {e}")
    return clean_indicators(df)

# ---------------------- 回測與策略分析 ----------------------
def display_market_analysis(csv_file, chart_type, strategy_option, initial_capital, slippage):
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return error_plot(f"讀取 CSV 檔案失敗: {e}"), pd.DataFrame()
    try:
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return error_plot(f"時間格式錯誤: {e}"), pd.DataFrame()
    df.set_index("time", inplace=True)
    
    # 繪製圖表背景
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
        if "volume" not in df.columns:
            return error_plot("CSV 文件中沒有 'volume' 欄位，無法生成成交量圖。"), pd.DataFrame()
        fig = px.bar(df, x=df.index, y="volume", title="成交量圖")
        fig.update_layout(xaxis_title="時間", yaxis_title="成交量")
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
        return error_plot("未知的圖表類型！"), pd.DataFrame()
    
    backtest_available = False
    buy_col = None
    sell_col = None

    # 根據策略選項呼叫對應策略函式
    if strategy_option == "Moving Average Crossover":
        df = apply_ma_crossover(df)
        buy_col = "buy_MA"
        sell_col = "sell_MA"
        backtest_available = True
    elif strategy_option == "RSI Strategy":
        df["RSI"] = compute_rsi(df)
        df["buy_RSI"] = df["RSI"] < 30
        df["sell_RSI"] = df["RSI"] > 70
        buy_col = "buy_RSI"
        sell_col = "sell_RSI"
        backtest_available = True
    elif strategy_option == "Bollinger Bands Breakout":
        df["MA20"] = df["close"].rolling(window=20).mean()
        df["STD20"] = df["close"].rolling(window=20).std()
        df["Upper"] = df["MA20"] + 2 * df["STD20"]
        df["Lower"] = df["MA20"] - 2 * df["STD20"]
        df["buy_BB"] = df["close"] < df["Lower"]
        df["sell_BB"] = df["close"] > df["Upper"]
        buy_col = "buy_BB"
        sell_col = "sell_BB"
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
    elif strategy_option == "Machine Learning (Walk-Forward)":
        df = apply_ml_strategy_walk_forward(df, train_window=252)
        buy_col = "buy_ML_WF"
        sell_col = "sell_ML_WF"
        backtest_available = True
    elif strategy_option == "Composite Strategy":
        df = apply_composite_strategy(df)
        buy_col = "buy_composite"
        sell_col = "sell_composite"
        backtest_available = True
    elif strategy_option == "MACD Strategy":
        df = apply_macd_strategy(df)
        buy_col = "buy_MACD"
        sell_col = "sell_MACD"
        backtest_available = True
    elif strategy_option == "Ichimoku Cloud Strategy":
        df = apply_ichimoku_strategy(df)
        buy_col = "buy_ichimoku"
        sell_col = "sell_ichimoku"
        backtest_available = True
    elif strategy_option == "VWAP Strategy":
        df = apply_vwap_strategy(df)
        buy_col = "buy_VWAP"
        sell_col = "sell_VWAP"
        backtest_available = True
    elif strategy_option == "SuperTrend Strategy":
        df = apply_supertrend_strategy(df)
        buy_col = "buy_supertrend"
        sell_col = "sell_supertrend"
        backtest_available = True
    elif strategy_option == "Donchian Channel Strategy":
        df = apply_donchian_channel_strategy(df)
        buy_col = "buy_donchian"
        sell_col = "sell_donchian"
        backtest_available = True
    else:
        return error_plot("未選擇正確的策略分析！"), pd.DataFrame()
    
    trade_table = None
    if backtest_available and buy_col and sell_col:
        trades, final_capital = backtest_trades(df, buy_col, sell_col, initial_capital, slippage)
        if trades:
            trade_returns = [t["出場價格"] / t["進場價格"] - 1 for t in trades]
            transaction_count = len(trade_returns)
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) if transaction_count > 1 and np.std(trade_returns) != 0 else None
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
            trade_table = pd.DataFrame([{"Message": "無完整交易訊號，無法進行回測計算"}])
    
    return fig, trade_table

# ---------------------- Gradio 分頁介面 ----------------------
iface_fetch = gr.Interface(
    fn=fetch_and_save,
    inputs=[
        gr.Textbox(label="交易對 (Symbol)", value="BTC-USDT"),
        gr.Textbox(label="起始日期 (YYYY-MM-DD)", value="2024-01-01")
    ],
    outputs=[
        gr.File(label="下載 CSV 檔案"),
        gr.Textbox(label="狀態訊息")
    ],
    title="K 線數據抓取工具",
    description="請輸入交易對與起始日期，從起始日期至現在抓取 K 線數據並儲存成 CSV 檔案。"
)

iface_show = gr.Interface(
    fn=show_csv,
    inputs=gr.File(label="上傳 CSV 文件"),
    outputs=gr.DataFrame(label="CSV 內容"),
    title="CSV 顯示工具",
    description="上傳 CSV 文件以顯示其內容。"
)

iface_analysis = gr.Interface(
    fn=display_market_analysis,
    inputs=[
        gr.File(label="上傳 CSV 文件"),
        gr.Dropdown(choices=["Candlestick", "Line Chart", "Volume Chart", "Moving Average", "Bollinger Bands"],
                    label="選擇圖表類型"),
        gr.Dropdown(choices=[
            "Moving Average Crossover", 
            "RSI Strategy", 
            "Bollinger Bands Breakout",
            "SMC Strategy",
            "Multi-Factor Model",
            "Statistical Arbitrage",
            "Mean Reversion",
            "Momentum Trading",
            "Pairs Trading",
            "Market Microstructure",
            "Machine Learning (Walk-Forward)",
            "Composite Strategy",
            "MACD Strategy",
            "Ichimoku Cloud Strategy",
            "VWAP Strategy",
            "SuperTrend Strategy",
            "Donchian Channel Strategy"
        ], label="選擇策略分析"),
        gr.Number(value=10000, label="自訂本金"),
        gr.Number(value=0.0001, label="滑點(百分比)")
    ],
    outputs=[gr.Plot(label="市場分析圖"), gr.DataFrame(label="回測交易明細")],
    title="回測與市場策略分析",
    description="上傳 CSV 文件，選擇圖表類型與策略模式，進行市場策略回測。"
)

tabbed = gr.TabbedInterface([iface_fetch, iface_show, iface_analysis], ["抓取 CSV", "顯示 CSV", "回測分析"])

if __name__ == "__main__":
    tabbed.launch()
