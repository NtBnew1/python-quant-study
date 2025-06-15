'''
第1天：
	安装并配置Backtrader，了解其基本架构。
	练习：编写一个简单的Backtrader脚本，和获取股票数据。
'''

import pandas as pd
import backtrader as bt
from yahooquery import Ticker

class SMA(bt.Strategy):
    def __init__(self):
        self.sma5 = bt.indicators.SMA(period=5)
        self.sma10 = bt.indicators.SMA(period=10)
        self.sma40 = bt.indicators.SMA(period=40)

    def next(self):
        if not self.position:
            if self.sma5[0] > self.sma10[0] and self.sma10[0] > self.sma40[0]:
                self.buy()
        elif self.sma5[0] < self.sma10[0] and self.sma10[0] < self.sma40[0]:
            self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"买入: {order.executed.price: .2f}")
            elif order.issell():
                print(f"卖出: {order.executed.price: .2f}")

def run_backtesting():
    ticker = input(f"请输入股票代码: ")
    stock = Ticker(ticker)
    data = stock.history(start='2024-08-05', end='2025-05-05')

    data = data.reset_index()
    data = data[['date', 'open', 'high', 'low', 'close', 'volume']]
    data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SMA)
    df = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(df)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)


    print(f"初始资金: {cerebro.broker.getvalue(): .2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue(): .2f}")

    cerebro.plot()

if __name__ == "__main__":
    run_backtesting()
