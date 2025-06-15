'''
第4天：
	需要学习新的库用了获取股票数据.
	回测双均线策略，分析回测结果。
'''
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import backtrader as bt



def get_daily_data(symbol, API_Key):
    ts = TimeSeries(key=API_Key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')    #获取20年的数据
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'Low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data.index = pd.to_datetime(data.index)

    # 获取一年的数据.
    one_year_ago = datetime.now() - pd.DateOffset(years=1)
    data = data[data.index >= one_year_ago]
    data = data.sort_index()
    return data

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma5 = bt.indicators.SMA(period=5)
        self.sma20= bt.indicators.SMA(period=20)

    def next(self):
        if not self.position:
            if self.sma5[0] > self.sma20[0]:
                self.buy()
        elif self.sma5[0] < self.sma20[0]:
            self.sell()

def run_backtest():
    data = get_daily_data(symbol, API_Key)
    df = bt.feeds.PandasData(dataname=data)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.adddata(df)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print(f"初始资金: {cerebro.broker.getvalue(): .2f}")
    cerebro.run()
    print(f"回测结果资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    API_Key = 'LNCEEYGQUGYZCRGO'
    symbol = 'AAPL'
    df = get_daily_data(symbol, API_Key)
    run_backtest()