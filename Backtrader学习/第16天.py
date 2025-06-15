'''
第16天：
任务目标：
编写一个自定义指标，深入理解Backtrader的Indicator类结构和用法。

练习内容：
    学习如何继承bt.Indicator类；
    理解lines、params和next()的作用；
    实现一个加权移动平均线（WMA）指标；
    将自定义指标加入现有策略，观察指标输出和信号。
'''


import backtrader as bt
import pandas as pd

# 加载数据
def load_data():
    file_path = './BABA_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

# 自定义指标
class WeightMovingAverage(bt.Indicator):
    lines = ('wma',)                # 指标输出: wma线
    params = (('period', 10),)      # 可调整参数: 10日线

    def __init__(self):
        # 构造权重:
        weights = list(range(1, self.p.period + 1))     # 权重从1 到period: 权重=[1,2,3,4,5,6,7,8,9,10]
        total_weight = sum(weights)         #权重总和

        # 一个个除以总和, 得到归一化的权重(加起来等于1)
        normalized_weights = []
        for w in weights:
            normalized_weights.append(w / total_weight)

        # 存储到self.weights
        self.weights = normalized_weights

    def next(self):
        if len(self.data) >= self.p.period:     # 如果数据的长度 还不够计算, 就不要算

            # 取出最近几天的价格
            recent_price = list(self.data.get(size=self.p.period))

            # 创建一个空列表. 用来放每个价格乘以权重的结果
            weighted_prices = []

            # 把每个价格和对应的权重相乘
            for i in range(self.p.period):
                price = recent_price[i]
                weight = self.weights[i]
                weighted_prices.append(price * weight)

            # 所有乘积加起来, 得到加权平均值
            wma_value = sum(weighted_prices)

            # 把结果保存到输出线上
            self.lines.wma[0] = wma_value

# 写一个策略来验证自定义的wma
class Test_WMA_Strategy(bt.Strategy):
    def __init__(self):
        self.wma = WeightMovingAverage(self.data, period=10)
        self.order = None   # 记录当前是否有订单

    def next(self):
        if self.order:
            return

        # 打印当前时间和wma的值
        dt = self.data.datetime.date(0)
        close = self.data.close[0]
        wma_val = self.wma[0]
        print(f"{dt} | 收盘价: {close:.2f} | WMA: {wma_val:.2f}")

        # 如果当前没有建仓库
        if not self.position:
            # 如果收盘价 > WMA =>买入
            if close > wma_val:
                print(f"{dt}买入信号 | 收盘价: {close:.2f} > WMA: {wma_val:.2f}")
                self.order = self.buy()     #把记录保存到self.order里
        else:
            # 如果收盘价 < WMA ==>卖出
            if close < wma_val:
                print(f"{dt}卖出信号 | {close:.2f} < WMA: {wma_val:.2f}")
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None   # 清空订单状态

# 运行回测构架
if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Test_WMA_Strategy)

    data = load_data()
    cerebro.adddata(data)

    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)

    print(f" 初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")

    cerebro.plot()









