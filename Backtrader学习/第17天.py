'''
第17天：
任务目标：
掌握多时间周期数据的加载和使用方法。

练习内容：
    学习cerebro.resampledata()和cerebro.adddata()的区别和用法；
    加载日线和小时线两种时间周期数据；
    编写策略同时使用不同周期的指标信号进行买卖决策；
    运行回测并分析多周期信号带来的策略变化。
'''

import backtrader as bt
import pandas as pd
from trio import sleep


# 加载数据
def load_data():
    file_path = './ALHC_year_data.csv'
    df = pd.read_csv(file_path)     # 读取数据
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

class MultiTimeFrameStrategy(bt.Strategy):
    params = (
        ('take_profit', 0.10),       # 止盈10%
        ('stop_loss', 0.40)         # 止损20%
    )

    def __init__(self):
        # self.datas[0] 是日线数据
        # self.datas[1] 是周线数据
        self.daily_sma = bt.indicators.SMA(self.datas[0].close, period=5)
        self.weekly_sma = bt.indicators.SMA(self.datas[1].close, period=5)
        self.buy_price = None   #初始化买入价格

    def next(self):
        if not self.position:
            if (self.datas[0].close[0] > self.daily_sma[0]) or (self.datas[1].close[0] > self.weekly_sma[0]):
                self.buy()
                self.buy_price = self.datas[0].close[0] # 记录买入价格
        else:
            # 当前价格
            current_price = self.datas[0].close[0]
            if self.buy_price:
                profit_pct = (current_price - self.buy_price) / self.buy_price

                # 止盈和止损
                if profit_pct >= self.params.take_profit:
                    self.log(f"止盈卖出: 盈利 {profit_pct*100:.2f}%")
                    self.close()
                    self.buy_price = None

                elif profit_pct <= -self.params.stop_loss:                  #忘记加 (-)
                    self.log(f"止损卖出: 亏损 {profit_pct*100:.2f}%")
                    self.close()
                    self.buy_price = None

            # 如果不满足止盈止损, 但价格跌破SMA, 可以自动平仓
            # elif current_price < self.daily_sma[0] and current_price <self.weekly_sma[0]:
            #     self.close()
            #     self.buy_price = None

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")

def run_testing():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MultiTimeFrameStrategy)

    data = load_data()
    cerebro.adddata(data)

    # 用resample 生成周线数据
    cerebro.resampledata(data, timeframe=bt.TimeFrame.Weeks)


    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.01)

    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    run_testing()




