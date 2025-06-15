'''
第6天：
	优化策略参数，如短期和长期均线的周期。
'''

import pandas as pd
import backtrader as bt


class RsiStrategy(bt.Strategy):
    params = (
        ('rsi_short', 7),  # 短期RSI周期
        ('rsi_long', 28),  # 长期RSI周期
        ('oversold', 30),  # 超卖阈值
        ('overbought', 70),  # 超买阈值
    )

    def __init__(self):
        # 计算两个RSI指标
        self.rsi_short = bt.indicators.RSI(self.data.close, period=self.p.rsi_short)
        self.rsi_long = bt.indicators.RSI(self.data.close, period=self.p.rsi_long)

        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.rsi_short, self.rsi_long)

    def next(self):
        # 简单的交叉策略
        if not self.position:
            if self.rsi_short > self.rsi_long:  # 短期上穿长期
                self.buy()
        elif self.rsi_short < self.rsi_long:  # 短期下穿长期
            self.close()


def optimize_parameters():
    # 1. 加载本地AAL数据
    df = pd.read_csv('./AAL_year_data.csv', parse_dates=['date'], index_col='date')

    # 2. 确保列名正确
    df = df.rename(columns={
        'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close',
        'Volume': 'volume'
    })

    # 3. 创建回测引擎
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    # 4. 设置优化参数范围
    cerebro.optstrategy(
        RsiStrategy,
        rsi_short=range(5, 16, 2),  # 测试5-15的短期周期，步长2
        rsi_long=range(20, 41, 5)  # 测试20-40的长期周期，步长5
    )

    # 5. 回测设置
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)

    # 6. 添加分析指标
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    # 7. 运行优化
    print("开始参数优化...")
    results = cerebro.run()

    # 8. 分析优化结果
    best_sharpe = -float('inf')
    best_params = None
    results_list = []

    for run in results:
        for strat in run:
            ret = strat.analyzers.returns.get_analysis()['rnorm100']
            sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
            results_list.append({
                'rsi_short': strat.params.rsi_short,
                'rsi_long': strat.params.rsi_long,
                'return': ret,
                'sharpe': sharpe
            })

            # 选择夏普比率最高的参数
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {
                    'rsi_short': strat.params.rsi_short,
                    'rsi_long': strat.params.rsi_long,
                    'return': ret,
                    'sharpe': sharpe
                }

    # 9. 输出结果
    print("\n=== 最佳参数组合 ===")
    print(f"短期RSI周期: {best_params['rsi_short']}天")
    print(f"长期RSI周期: {best_params['rsi_long']}天")
    print(f"年化回报率: {best_params['return']:.2f}%")
    print(f"夏普比率: {best_params['sharpe']:.2f}")

    # 10. 保存所有结果到CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv('AAL_optimization_results.csv', index=False)
    print("\n所有参数组合结果已保存到 AAL_optimization_results.csv")

    # 用最佳参数绘制图
    print(f"\n使用最佳参数运行并绘制")
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(
        RsiStrategy,
        rsi_short = best_params['rsi_short'],
        rsi_long = best_params['rsi_long']
    )
    cerebro.broker.setcash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.run()
    cerebro.plot()

if __name__ == '__main__':
    optimize_parameters()
