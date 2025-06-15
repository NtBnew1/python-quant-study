'''
第8天：
	添加风险管理模块，如止损和止盈机制。
'''

import pandas as pd
import backtrader as bt



class MACD_RSI_Strategy(bt.Strategy):
    params = (
        ('macd1', 12),      #macd 快线周期
        ('macd2', 26),      # macd 慢线周期
        ('macdsig', 9),     # macd 信号线周期
        ('rsi_period', 14),  # rsi 周期
        ('rsi_buy', 30),     #RSI 超买
        ('printlog', False),  #  是否打印日志
        ('trade_size', 0.95),    # 每次交易使用资金比列

        #风控参数
        ('stop_loss', 0.05),    # 止损5%
        ('take_profit', 0.1),   # 止盈10%
        ('trailing_stop', False),    # 是否跟踪止损
        ('trail_percent', 0.03),  #跟踪止损3%
        ('risk_reward_ratio', 2)    #风险收益比
    )

    # 日志打印函数
    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()}, {txt}")

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd1,
            period_me2=self.p.macd2,
            period_signal=self.p.macdsig)

        # 判断MACD金叉( macd上穿signal线)
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        # 初始化订单和状态
        self.order = None
        self.stop_order = None
        self.take_profit_order = None
        self.entry_price = 0

        # 统计交易次数
        self.trade_count = 0
        self.win_count = 0
        self.trade_history = []

    # 当订单状态改变时调用
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return # 订单还在处理中

        if order.status == order.Completed:
            if order.isbuy():
                # 买入成功
                self.log(f"买入执行, 价格: {order.executed.price:.2f}")
                self.entry_price = order.executed.price

                # 设置止损价和止盈价
                stop_price = self.entry_price * (1-self.p.stop_loss)
                tp_price = self.entry_price * (1+self.p.take_profit)

                # 生成止损订单
                self.stop_order = self.sell(
                    exectype=bt.Order.Stop,
                    price=stop_price,
                    size=order.executed.size,
                    transmit=False)

                # 生成止盈订单
                self.take_profit_order = self.sell(
                    exectype=bt.Order.Limit,
                    price=tp_price,
                    size=order.executed.size,
                    transmit=True)

                # 用跟踪止损
                if self.p.trailing_stop:
                    self.trailing_stop_order = self.sell(
                        exectype=bt.Order.StopTrail,
                        trailpercent=self.p.trail_percent,
                        size=order.executed.size)

            elif order.issell():
                #卖出
                self.log(f"执行卖出, 价格: {order.executed.price:.2f}")
                profit_pct = (order.executed.price / self.entry_price - 1)* 100
                if profit_pct > 0:
                    self.win_count += 1

                # 取消未触发的止盈和止损
                if self.stop_order:
                    self.cancel(self.stop_order)
                if self.take_profit_order:
                    self.cancel(self.take_profit_order)
                if hasattr(self, 'trailing_stop_order'):
                    self.cancel(self.trailing_stop_order)

            # 记录每笔交易
            self.trade_history.append({
                'date': self.data.datetime.date(0),
                'type': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm,
                'pnl': order.executed.pnl
            })

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单取消/保证金不足/被拒绝')
        self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            if self.macd_cross[0] > 0 and self.rsi[0] < self.p.rsi_buy:
                size = self.broker.getcash() * self.p.trade_size / self.data.close[0]
                self.order = self.buy(size=size)
                self.trade_count += 1
        else:
            # 持有仓位时考虑动态更新止损价
            current_price = self.data.close[0]
            if current_price > self.entry_price * (1+self.p.stop_loss * self.p.risk_reward_ratio):
                new_stop = current_price * (1-self.p.stop_loss)
                self.cancel(self.stop_order)
                self.stop_order = self.sell(
                    exectype=bt.Order.Stop,
                    price=new_stop,
                    size=self.position.size,
                    transmit=False)

    # 回测结果后运行
    def stop(self):
        self.log("期末资金: %.2f" % self.broker.getvalue(), doprint=True)
        self.log('总交易次数: %d' % self.trade_count, doprint=True)
        if self.trade_count > 0:
            self.log("胜率: %.2f%%" % (self.win_count / self.trade_count * 100), doprint=True)

# 先加载数据
def load_data():
    file_path = './AI_year_data.csv'
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    data = bt.feeds.PandasData(dataname=df)
    return data

# 回测主函数
def run_optimization():
    cerebro = bt.Cerebro(optreturn=False)

    data = load_data()      # 调用load_data
    cerebro.adddata(data)

    # 策略参数优化组合
    cerebro.optstrategy(
        MACD_RSI_Strategy,
        macd1 = [12],
        macd2 = [26],
        macdsig = [9],
        rsi_buy = [25, 30, 35],
        stop_loss = [0.03, 0.05],
        take_profit = [0.08, 0.1],
        trailing_stop = [False],
        printlog = [True]
    )

    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)

    opt_runs = cerebro.run()

    # 输出每组结果
    for run in opt_runs:
        strat = run[0]
        print('最终终极: %.2f, 参数: rsi_buy=%d, stop_loss=%.2f, take_profit=%.2f' % (
            strat.broker.getvalue(),
            strat.params.rsi_buy,
            strat.params.stop_loss,
            strat.params.take_profit
        ))

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    run_optimization()

