'''
由于获取数据有limit, 我们就保存数据, 之后才读取数据来回测.
'''

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import time  # 加个延迟的

def save_stock_data(symbols, api_key):
    ts = TimeSeries(key=api_key, output_format='pandas')

    for symbol in symbols:
        try:
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            data = data.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })

            data.index = pd.to_datetime(data.index)
            data = data.sort_index()

            # 只要一年的数据
            one_year_ago = datetime.now() - pd.DateOffset(years=1)
            data = data[data.index >= one_year_ago]

            # 保存数据到CSV
            filename = f"./{symbol}_year_data.csv"
            data.to_csv(filename)
            print(f"已经保存{symbol}数据到{filename}")
        except Exception as e:
            print(f" 获取{symbol}失败: {e}")

        time.sleep(12)

if __name__ == "__main__":
    stock_input = input(f"请输入股票: ")
    stock_list = [s.strip().upper() for s in stock_input.split(',') if s.strip()]
    api_key = 'LNCEEYGQUGYZCRGO'
    save_stock_data(stock_list, api_key)


"""   没办法获取更多数据. 每个月最多25次. """
