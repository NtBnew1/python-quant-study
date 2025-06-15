'''
第19天：
任务目标：
实现复杂订单管理，掌握挂单和自动撤单的策略逻辑。

练习内容：
    理解订单状态和生命周期；
    在notify_order()中实现限价挂单，设定挂单有效期；
    挂单超过有效期自动撤单并重新挂单；
    运行回测测试挂单和撤单机制的效果。
'''

import pandas as pd
import backtrader as bt
from datetime import timedelta

def load_data():
    file_path = './UNH_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['openinterest'] = 0

    data = bt.feeds.PandasData(dataname=df)
    return data

class LimitOrderStrategy(bt.Strategy):
    params = dict(
        limit_price_buffer = 0.98,      # 限价买单 = 当前收盘价 * 0.98
        order_valid_days = 2            # 挂单有效期: 2天
    )

    def __init__(self):
        self.order = None
        self.order_time = None

    def next(self):
        dt = self.data.datetime.datetime(0)             # 当前K线的时间

        # 如果已有挂单, 检查是否已过期
        if self.order:
            expire_time = self.order_time + timedelta(days=self.p.order_valid_days)
            if dt >= expire_time:
                print(f"[{dt}] 挂单超过{self.p.order_valid_days}天未成交, 撤单")
                self.cancel(self.order)     # 主动撤单
            return      #已有订单挂出, 不再新下单

        # 如果当前没有仓位,没有挂单, 就挂一个限价买单
        if not self.position:
            limit_price = self.data.close[0] * self.p.limit_price_buffer
            self.order_time = dt  # 记录挂单时间

            # 创建限价买单, 并设置有效期2天
            self.order = self.buy(
                exectype=bt.Order.Limit,
                price=limit_price,
                valid=dt + timedelta(days=self.p.order_valid_days)
            )
            print(f" [{dt}] 下达限价买单, 价格:{limit_price:.2f}, 有效期: {self.p.order_valid_days}天")

    def notify_order(self, order):
        dt = self.data.datetime.datetime(0)

        # 如果订单被成交
        if order.status == order.Completed:
            if order.isbuy():
                print(f"[{dt}]限价单成交: 买入价格={order.executed.price:.2f}")
            self.order = None # 成交后清除订单记录

        # 如果订单被撤销或被拒绝
        elif order.status in [order.Canceled, order.Rejected]:
            print(f"[{dt}] 订单被撤销或拒绝, 将再下一根K线重新挂单")
            self.order = None   # 清除订单记录以便下一次 next()重新挂单

def run_testing():
    cerebro = bt.Cerebro()
    data = load_data()
    cerebro.adddata(data)
    cerebro.addstrategy(LimitOrderStrategy)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.01)

    print(f" 初始资金: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f" 最终资金: {cerebro.broker.getvalue():.2f}")

    cerebro.plot()

if __name__ == "__main__":
    run_testing()


