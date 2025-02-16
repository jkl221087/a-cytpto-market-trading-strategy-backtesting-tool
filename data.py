import time
import datetime
import csv
import gradio as gr
from bingX.perpetual.v2 import Perpetual
import pandas as pd

# 初始化客戶端
client = Perpetual(
    "oMUHyjXz7FiMyNWa3YToqkqWiFsKMSgJEZ5rBB83SpzmfIMN50PWLzz8UgYdY6jAbRMmiKwsDZlVLkYkL7c8A",
    "tHTkG1fb5wHcvZfx5TPy3CsUUHOMrXLdyDB9ru8WBSjCFhm0Y7xr8z7kaVwG0avXAhqMstF0VVxh4uWYMQ"
)

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
    
    # 轉換為毫秒級的 timestamp
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    all_kline_data = []
    current_ts = start_ts
    limit_per_call = 1000
    
    while current_ts < end_ts:
        # 計算當前批次的結束時間戳（1000小時後）
        batch_end_ts = min(current_ts + (1000 * 3600 * 1000), end_ts)
        
        # 獲取數據
        kline_data = fetch_kline_1h(symbol, current_ts, batch_end_ts, limit=limit_per_call)
        
        if not kline_data or len(kline_data) == 0:
            print(f"No data returned for period: {datetime.datetime.fromtimestamp(current_ts/1000)} - {datetime.datetime.fromtimestamp(batch_end_ts/1000)}")
            break
        
        all_kline_data.extend(kline_data)
        print(f"Fetched {len(kline_data)} records. Total records: {len(all_kline_data)}")
        
        # 更新下一次查詢的起始時間戳
        current_ts = kline_data[-1]['time'] + 3600000  # 加上一小時的毫秒數
        
        # 添加延遲以避免API限制
        time.sleep(0.5)
    
    if not all_kline_data:
        return None, "未能獲取任何數據"
    
    # 設定 CSV 檔名
    filename = f"{symbol.replace('-', '_')}_kline_data.csv"
    fieldnames = ["time", "open", "close", "high", "low", "volume"]
    
    # 將資料寫入 CSV 檔案
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_kline_data:
            row["time"] = datetime.datetime.fromtimestamp(row["time"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(row)
    
    status_message = f"共抓取 {len(all_kline_data)} 根 K 線資料，CSV 檔案已儲存：{filename}"
    return filename, status_message

def show_csv(file):
    """
    讀取並顯示 CSV 文件內容
    """
    if file is None:
        return pd.DataFrame()
    try:
        df = pd.read_csv(file.name)
        return df
    except Exception as e:
        return pd.DataFrame({'error': [f'無法讀取文件: {str(e)}']})

# Gradio 介面設置
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
    description="上傳 CSV 文件以顯示其內容"
)

tabbed = gr.TabbedInterface([iface_fetch, iface_show], ["抓取 CSV", "顯示 CSV", "計算數據"])

if __name__ == "__main__":
    tabbed.launch()