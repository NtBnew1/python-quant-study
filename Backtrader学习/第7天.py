'''
第7天:
    练习：编写优化脚本，测试不同参数组合的效果。
'''


import backtrader as bt
import pandas as pd

class RSI_EMA_Strategy(bt.Strategy):
    params = (
        ('ema_fast', 5),
        ('ema_slow', 20),
        ('rsi_period', 14),
        ('rsi_buy', 30),
        ('rsi_sell', 70)
    )

    def __init__(self):
        self.ema_fast = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.ema_slow)
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data.close, period=self.p.rsi_period)
        self.cross = bt. indicators.CrossOver(self.ema_fast, self.ema_slow)

        # 交易记录
        self.order = None
        self.trade_count = 0

    def next(self):
        if self.order:
            return

        # 打印调试信息
        print(f"日期: {self.data.datetime.date(0)}, 收盘价: {self.data.close[0]: .2f}, "
              f"EMA快: {self.ema_fast[0]:.2f}, EMA慢: {self.ema_slow[0]:.2f}, "
              f"RSI: {self.rsi[0]:.2f}, 交叉: {self.cross[0]}")

        if not self.position:
            # EMA金叉 和RSI 超卖, 买入
            if self.cross[0] > 0 or self.rsi[0] < self.p.rsi_buy:
                self.order = self.buy()
                print(f"买入信号: {self.data.close[0]:.2f}")
                self.trade_count += 1       # 每次都会加入到trade_count
        else:
            # 卖出条件: EMA死叉 和RSI超买, 卖出
            if self.cross[0] < 0 or self.rsi[0] > self.p.rsi_buy:
                self.order = self.sell()
                print(f"卖出信号: {self.data.close[0]:.2f}")
                self.trade_count += 1

    def stop(self):
        print(f"\n策略结果, 总交易次数: {self.trade_count}")

def run_strategy():
    # 加载数据
    try:
        file_path = './AAPL_year_data.csv'
        df = pd.read_csv(file_path, parse_dates=['date'])
        df.set_index('date', inplace=True)

        data = bt.feeds.PandasData(dataname=df)
    except Exception as e:
        print(f" 数据加载失败: {e}")
        return

    # 创建Cerebro
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addstrategy(RSI_EMA_Strategy)


    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # 运行策略
    print(f"\n开始回测.....")
    results = cerebro.run()             # 运行结果保存到results里.
    strat = results[0]
    print(f"初始资金: {strat.broker.getvalue():.2f}")


    # 输出结果:
    print(f"\n===回测结果===")
    print(f"最终资金: {strat.broker.getvalue():.2f}")

    # 分析结果:
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.2f}")
    print(f"最大回测: {drawdown.get('max', {}).get('min', 0):.2f}")
    print(f"总交易次数: {trades.get('total', {}).get('closed', 0)}")

    # 绘制结果;
    print(f"\n正在绘制结果图表......")
    cerebro.plot()

if __name__ == "__main__":
    run_strategy()

