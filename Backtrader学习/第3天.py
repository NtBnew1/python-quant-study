'''
第3天:
	练习：实现双均线策略, 用backtrader库，设置短期和长期均线。
'''

import pandas as pd
import backtrader as bt
from yahooquery import Ticker

class DoubleMA_Strategy(bt.Strategy):
    params = (
        ('short_period', 10),
        ('long_period', 30)
              )

    def __init__(self):
        self.short_ma = bt.indicators.SMA(period=self.p.short_period)
        self.long_ma = bt.indicators.SMA(period=self.p.long_period)

    def next(self):
        if not self.position:
            if self.short_ma[0] > self.long_ma[0]:
                self.buy()
        elif self.short_ma[0] < self.long_ma[0]:
            self.sell()

def run_backtesting():
    ticker = input(f"请输入股票: ")
    stock = Ticker(ticker)
    df = stock.history(period='1y')
    df = df.reset_index()
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # 时间混乱, 需要改.
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
    df.set_index('datetime', inplace=True)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(DoubleMA_Strategy)

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f'初始资金: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'最终资金: {cerebro.broker.getvalue():.2f}')
    cerebro.plot()

if __name__ == "__main__":
    run_backtesting()


