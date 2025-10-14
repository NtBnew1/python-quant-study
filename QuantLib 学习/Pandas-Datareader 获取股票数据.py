'''   现在需要获取股票数据.  现在用pandas-Datareader库来获取数据.'''


import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置时间
start = datetime.now() - timedelta(days=365*10) # 计算10年
end = datetime.now()

# 用户输入股票代码
ticker_input = input(f"请输入股票代码: ")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

for ticker in tickers:
    try:
        filename = f"./{ticker}_stock.xlsx"
        if os.path.exists(filename):
            print(f"{filename}已存在, 跳过下载")
            continue
        print(f"正在下载{ticker}数据.....")

        # 如果yahoo获取数据失败, 就改用stooq获取数据.
        try:
            df = web.DataReader(ticker, 'yahoo', start, end)
        except:
            print(f"Yahoo 失败, 尝试Stooq...")
            df = web.DataReader(f"{ticker}.US", "stooq", start, end)

        df.to_excel(filename)
        print(f"{ticker}已保存到{filename}")
    except Exception as e:
        print(f"获取{ticker}失败: {e}")


