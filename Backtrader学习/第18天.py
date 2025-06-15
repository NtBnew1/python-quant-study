'''
第18天：
任务目标：
学习多数据源策略，实现同时管理多只股票的交易逻辑。

练习内容：
    理解self.datas列表结构；
    在策略中分别访问和处理不同股票数据；
    实现两个股票不同买卖条件的示范策略；
    回测多数据策略，观察组合表现。
'''

import backtrader as bt
import pandas as pd



# 加载函数
def load_data(file_path, name):    # 这次要加载多个公司股票数据.
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0
    return bt.feeds.PandasData(dataname=df, name=name)

# 策略
class MultiStockStrategy(bt.Strategy):
    def __init__(self):
        # 按照名称获取数据源
        ''' 如果不用getdatabyname, 就需要用self.datas[0] 和self.datas[1] 代表UNH 和PLTR.
        用getdatabyname, 只需要填写股票名称, 就可以按照self.unh'''
        self.unh = self.getdatabyname('UNH')
        self.pltr = self.getdatabyname('PLTR')

        # 为每只股票独立创建的指标
        self.sma1 = bt.indicators.SMA(self.unh.close, period=5)
        self.sma2 = bt.indicators.SMA(self.pltr.close, period=10)

    def next(self):
        # 处理UNH
        unh1 = self.getposition(self.unh).size
        if not unh1 and self.unh.close[0] > self.sma1[0]:
            self.buy(data=self.unh)
        elif unh1 and self.unh.close[0] < self.sma1[0]:
            self.sell(data=self.unh)

        # 处理 PLTR
        pltr1 = self.getposition(self.pltr).size
        if not pltr1 and self.pltr.close[0] > self.sma2[0]:
            self.buy(data=self.pltr)
        elif pltr1 and self.pltr.close[0] < self.sma2[0]:
            self.sell(data=self.pltr)

def run_testing():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MultiStockStrategy)

    # 加载两个不同的数据
    data1 = load_data('./UNH_year_data.csv', 'UNH')
    data2 = load_data('./PLTR_year_data.csv', 'PLTR')

    cerebro.adddata(data1)
    cerebro.adddata(data2)


    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    print(f"初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    cerebro.plot()

if __name__ == "__main__":
    run_testing()






