'''
第9天:
	练习：实现止损和止盈策略，并重新回测。
'''

import backtrader as bt
import pandas as pd
from trio import sleep


def load_data():
    file_path = './AI_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['openinterest'] = 0
    df.set_index('date', inplace=True)
    # print(df.head())
    data = bt.feeds.PandasData(dataname=df)
    return data

class Stop_loss_take_profit(bt.Strategy):
    params = (
        ('ma_short', 5),
        ('ma_long', 30),
        ('stop_loss', 0.05),    # 止损5%
        ('take_profit', 0.1)    # 止盈10%
              )

    ''' 只有计算止盈和止损'''
    def __init__(self):
        self.order = None
        self.buy_price = None

        self.ma_short = bt.indicators.MovingAverageSimple(self.data.close, period=self.params.ma_short)
        self.ma_long = bt.indicators.MovingAverageSimple(self.data.close, period=self.params.ma_long)

    def next(self):
        if self.order:
            return

        if not self.position:
            ''' 在这里加入买的条件'''
            if self.ma_short[0] > self.ma_long[0]:
                self.order = self.buy()
                self.buy_price = self.data.close[0]
                print(f"买入时间: {self.datas[0].datetime.date(0)}, 买入价格: {self.buy_price:.2f}")
        else:
            current_price = self.data.close[0]
            # 计算止损和止盈
            take_profit_price = self.buy_price * (1 + self.params.take_profit)
            stop_lost_price = self.buy_price * (1 - self.params.stop_loss)

            if current_price >= take_profit_price:
                self.order = self.sell()
                print(f"止盈卖出时间: {self.datas[0].datetime.date(0)}, 当前价格: {current_price:.2f}")
            elif current_price <= stop_lost_price:
                self.order = self.sell()
                print(f"止损卖出时间: {self.datas[0].datetime.date(0)},  当前价格: {current_price:.2f}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

def run_testting():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Stop_loss_take_profit)

    data = load_data()
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终终极: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()


if __name__ == "__main__":
    run_testting()

