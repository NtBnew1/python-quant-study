'''
第10天：
	整合成交量指标，设计成交量均线策略。
'''

import pandas as pd
import backtrader as bt

# 获取数据
def load_data():
    file_path = './GME_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0
    # print(df)
    data = bt.feeds.PandasData(dataname=df)
    return data

class Volume_SMA_Strategy(bt.Strategy):
    params = (
        ('volume_ma_period', 10),
        ('price_ma_period', 20),
        ('stop_loss', 0.05),    # 止损
        ('take_profit', 0.1)    # 止盈
    )

    def __init__(self):
        self.volume_ma = bt.indicators.MovingAverageSimple(self.data.volume, period=self.params.volume_ma_period)
        self.price_ma = bt.indicators.MovingAverageSimple(self.data.close, period=self.params.price_ma_period)
        self.buy_price = None
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # 买入条件: 当前成交量 > 成交量均线 和 当前价格 > 价格均线
            if self.data.volume[0] > self.volume_ma[0] and self.data.close[0] > self.price_ma[0]:
                self.buy_price = self.data.close[0]
                self.order = self.buy()
                print(f"买入时间: {self.datas[0].datetime.date(0)}, 买入股价: {self.buy_price}")

        else:
            current_price = self.data.close[0]
            # 计算止损和止盈
            take_profit = self.buy_price * (1 + self.params.take_profit)
            stop_loss = self.buy_price * ( 1- self.params.stop_loss)

            if current_price >=take_profit:
                self.order = self.close()
                print(f"止盈卖出时间:{self.datas[0].datetime.date(0)}, 当前价格:{current_price:.2f}, 买入股价: {self.buy_price}")
            elif current_price <= stop_loss:
                self.order = self.close()
                print(f"止损卖出时间: {self.datas[0].datetime.date(0)}, 当前价格:{current_price:.2f}, 买入股价: {self.buy_price}")

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

def run_testing():
    cerebro = bt.Cerebro()
    data = load_data()
    cerebro.adddata(data)
    cerebro.addstrategy(Volume_SMA_Strategy)

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    run_testing()