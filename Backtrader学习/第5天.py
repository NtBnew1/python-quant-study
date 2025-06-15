'''
第5天:
	练习：运行回测脚本，获取策略的收益率曲线和统计指标。
'''

import pandas as pd
import backtrader as bt
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime
import matplotlib.pyplot as plt


def get_daily_data(symbol, API_Key):
    ts = TimeSeries(key=API_Key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data = data.rename(columns={
        '1. open': 'Open',
        '2. high': 'High',
        '3. low': 'low',
        '4. close': 'Close',
        '5. volume': 'Volume'
    })
    data.index = pd.to_datetime(data.index)

    one_year_ago = datetime.now() - pd.DateOffset(years=1)
    data = data[data.index >= one_year_ago]
    data = data.sort_index()
    return data

class SMA_Strategy(bt.Strategy):
    def __init__(self):
        self.sma5 = bt.indicators.SMA(period=5)
        self.sma20 = bt.indicators.SMA(period=20)
        self.portfolio_value = []       #   用于保存每日资金

    def next(self):
        self.portfolio_value.append(self.broker.getvalue()) #每日资金变化
        if not self.position:
            if self.sma5[0] > self.sma20[0]:
                self.buy()
        else:
            if self.sma5[0] < self.sma20[0]:
                self.sell()

    def stop(self):
        # 绘制收益率图
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio_value)
        plt.title('Portfolio Value Over Time')
        plt.xlabel('day')
        plt.ylabel('Portfolio Value')
        plt.grid()
        plt.tight_layout()
        plt.show()

def run_testing():
    data = get_daily_data(symbol, API_Key)
    df = bt.feeds.PandasData(dataname=data)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SMA_Strategy)
    cerebro.adddata(df)
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"初始资金: {cerebro.broker.getvalue(): .2f}")

    result = cerebro.run()
    strat = result[0]
    print(f"最终资金: {cerebro.broker.getvalue(): .2f}")
    print(f"\n_______回测统计指标_________")

    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"最大回测: {drawdown['max']['drawdown']: .2f}")

    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"夏普比率: {sharpe.get('sharperatio', '无法计算')}")

    trades = strat.analyzers.trades.get_analysis()
    print(f"总交易次数: {trades['total']['total']}")
    print(f"盈利次数: {trades['won']['total']}")
    print(f"亏损次数: {trades['lost']['total']}")
    print(f"胜率: {(trades['won']['total'] / trades['total']['total']): .2%}")

    cerebro.plot()


if __name__ == "__main__":
    API_Key = 'LNCEEYGQUGYZCRGO'
    symbol = input(f"请输入股票代码: ")
    run_testing()
