'''
第4天 下：

在Backtrader中实现多因子策略（导入因子数据，信号触发买卖）
运行回测，生成收益曲线和策略绩效指标
练习：完成多因子策略回测代码，分析回测报告


'''

'''先用zscore选出3个股票最高得分'''

import pandas as pd
from scipy.stats import zscore
from datetime import datetime

# 1. 读取多因子数据文件
df = pd.read_excel('./Day4_factor_all_stocks.xlsx')

# 2. 设置股票代码为索引，方便按股票操作
df.set_index('Stock', inplace=True)

# 3. 选择需要用来打分的因子列
factor_col = [
    'PE', 'PB', 'EV_EBITDA', '12m_return', '6m_return', '3m_return',
    'ROE', 'ROA', 'NetMargin', 'Volatility', 'MaxDrawdown'
]

# 4. 标准化所有因子，因部分因子数值越小越好，需要取负数反向处理
df_z = df.copy()  # 复制一份，避免覆盖原始数据
reverse_factor = ['PE', 'PB', 'EV_EBITDA', 'Volatility', 'MaxDrawdown']  # 这些因子越小越好

for col in factor_col:
    # 如果是越小越好的因子，则乘以-1反向，否则保持正向
    df_z[col] = zscore(df[col]) * (-1 if col in reverse_factor else 1)

# 5. 计算综合得分，默认所有因子等权重，取均值
df_z['score'] = df_z[factor_col].mean(axis=1)


# 6. 按得分排序，取前3只股票作为回测标的
top_stocks = df_z.sort_values(by='score', ascending=False).head(3)
print('Top 3 stocks:', top_stocks.index.tolist())



''' HOOD, TME, HIMS 是最高得分. 现在用这3只股票导入到backtrader回测'''
# 导入backtrader库
import backtrader as bt

# 定义策略，买入得分最高的3只股票，20天轮换一次
# top3 = ['HOOD', 'TME', 'HIMS']  # 这里用上面得分最高的3只股票
top3 = top_stocks.index.tolist()  # 这里用上面得分最高的3只股票

class SimpleTop3Strategy(bt.Strategy):
    params = dict(top3=[])

    def __init__(self):
        self.rebalance_date = 0
        self.holding_period = 20   # 每20 天换仓一次

    def next(self):
        self.rebalance_date += 1

        if self.rebalance_date % self.holding_period != 0:
            return

        # 卖出不在当前top中
        for data in self.datas:                             # 遍历每只股票
            pos = self.getposition(data)                    # 查看当前持仓
            if pos.size > 0 and data._name not in self.params.top3:
                # 如果持仓中这只股票不在top3 中,
                self.close(data)        # 卖出

        # 买入新的top3 股票 (等权)
        weight = 1.0 / len(self.params.top3)                # 等权总是1.  这样分配有几只股票.
        for data in self.datas:                             # 遍历每只股票
            # 如果这只股票再top 3 中, 并且没有持仓
            if data._name in self.params.top3 and self.getposition(data).size == 0:
                # 按等权分配资金, 计算买入股数
                size = self.broker.getvalue() * weight / data.close[0]
                self.buy(data, size=size)


# 初始化Cerebro引擎
# symbols = ['HIMS', 'HOOD', 'TME']
symbols = top3

cerebro = bt.Cerebro()
initial_value = 1000000  # 初始资金100万美元
cerebro.broker.setcash(initial_value)
cerebro.broker.setcommission(commission=0.001)  # 交易手续费0.1%
cerebro.addstrategy(SimpleTop3Strategy, top3=top3)  # 添加策略和参数


# 加载数据
for symbol in symbols:
    df = pd.read_excel(f'{symbol}_all_data.xlsx')
    df['date'] = pd.to_datetime(df['Unnamed: 0'])  # 将日期列转换为datetime格式
    df.set_index('date', inplace=True)  # 设置日期为索引
    data = bt.feeds.PandasData(dataname=df, name=symbol)  # 转换成backtrader数据格式
    data._name = symbol  # 添加股票名称，方便识别
    cerebro.adddata(data)  # 加入Cerebro数据池

print(f"初始资金: {cerebro.broker.getvalue():.2f}")


# 添加分析器，计算夏普比率、最大回撤、交易统计和交易明细
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.Transactions, _name='transactions')


results = cerebro.run()

# 回测结束，获取最终资金、收益和收益率
final_value = cerebro.broker.getvalue()
profit = final_value - initial_value
roi = profit / initial_value * 100  # 百分比收益率

print(f'最终资金: {final_value:.2f}')
print(f'净收益: {profit:.2f}')
print(f'收益率: {roi:.2f}%')

strats = results[0]
# 获取策略分析结果
print(f" 夏普比率: {strats.analyzers.sharpe.get_analysis()}")
print(f'最大回测: {strats.analyzers.drawdown.get_analysis()}')
print(f'交易统计: {strats.analyzers.trades.get_analysis()}')

# 循环打印每笔交易记录
print(f'\n每笔交易记录:')
transactions = strats.analyzers.transactions.get_analysis()

for date, trans_list in transactions.items():
    print(f' \n 日期: {date.strftime("%Y-%m-%d")}')
    for trade in trans_list:
        size, price, action, symbol, value = trade
        action_str = '买入' if action == 0 else ('卖出' if action == 1 else '平仓')
        print(f'股票: {symbol:<5} | 操作: {action_str:<4} | 数量: {int(size):<8} | 单价: {price:<6.2} | 总金额: {value:<.2f}' )

'''
| 部分       | 含义                   |
| -------- | -------------------- |
| `:<5`    | 左对齐，占5位              |
| `:<6.2f` | 左对齐，占6位，小数点后2位       |
| `<.2f`   | 默认对齐（一般右对齐），保留小数点后两位 |
'''

cerebro.plot()



