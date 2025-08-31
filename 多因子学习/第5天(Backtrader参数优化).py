'''
第5天：

学习Backtrader参数优化工具，批量测试不同因子权重组合
了解参数过拟合与稳健性评估
练习：设计参数网格，运行参数优化，筛选最佳参数组合

'''

import os
import pandas as pd
import backtrader as bt
import datetime
import numpy as np
from scipy.stats import zscore

# =========== 第一步: 读取并处理多因子数据=================

# 读取所有的因子数据
file_path = './Day4_factor_all_stocks.xlsx'
df = pd.read_excel(file_path)

# 假设第一列是股票代码, 后面是因子
stock_col = df.columns[0]
factor_cols = df.columns[1:]

# 对所有因子列进行z-score标准化
df_z = df[factor_cols].apply(zscore)

# 反向因子处理, 这些因子越低越好,
reverse_factor = ['PE', 'PB', 'EV_EBITDA', 'Volatility', 'MaxDrawdown']
for col in reverse_factor:
    if col in df_z.columns:
        df_z[col] = -df_z[col]

# 把股票代码加回到标准化后的数据中
df_z[stock_col] = df[stock_col]

# 将每只股票的因子存入字典中,
factor_scores = {}
for _, row in df_z.iterrows():
    code = row[stock_col]
    factor_scores[code] = {col: row[col] for col in factor_cols}

# 打印检查结构
print(type(factor_scores))
print(factor_scores)

# ===========第二步: 因子组合打分, 选出top 5 股票===================

# 默认的打分方法是所有因子等权重相加
df_z['score'] = df_z[factor_cols].sum(axis=1)

# 根据综合得分排名, 取前top 5 股票
top5_df = df_z.sort_values(by='score', ascending=False).head(5)
top5_stock_list = top5_df[stock_col].tolist()
print(f"Top 5 股票: {top5_stock_list}")


# =============第三步: 定义策略类, 使用因子分数做多空判断==============
#全局变量,
GLOBAL_FACTOR_SCORE = factor_scores

class MultiFactorStrategy(bt.Strategy):
    # 定义可调的参数 (就是优化的因子权重)
    params = (
        ('pe_weight', 1.0),
        ('pb_weight', 1.0),
        ('momentum_weight', 1.0),
        ('volatility_weight', 1.0)
    )

    def __init__(self):
        # 引用全局的因子分数字典
        self.scores = GLOBAL_FACTOR_SCORE
        self.order = None   # 保存订单状态, 避免重复下单

    def next(self):
        # 如果当前有订单在处理中,
        if self.order:
            return

        for data in self.datas:
            stock = data._name
            if stock not in self.scores:
                continue

            f = self.scores[stock]

            # 用当前策略的参数计算综合打分
            score = (self.params.pe_weight * f.get('PE', 0) +
                     self.params.pb_weight * f.get('PB', 0) +
                     self.params.momentum_weight * f.get('Momentum', 0) +
                     self.params.volatility_weight * f.get('Volatility', 0)
                     )

            # 获取当前持仓量
            pos = self.getposition(data).size

            # 如果打分大于0, 买入, 小于0, 卖出
            if score > 0 and pos == 0:
                self.order = self.buy(data=data)
            elif score < 0 and pos > 0:
                self.order = self.sell(data=data)


#  =====================第四步: 运行Backtrader策略并进行参数优化==================
if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False)   # optreturn=False,保证返回完整策略

    # 为top 5 股票加载对应的价格数据
    for symbol in top5_stock_list:
        file_path = f"./{symbol}_all_data.xlsx"
        if not os.path.exists(file_path):
            print(f"找不到数据文件: {file_path}, 已跳过")
            continue

        # 读取 数据并确保包含基本行
        df_price = pd.read_excel(file_path, index_col=0, parse_dates=True)
        df_price.index.name = 'date'
        df_price = df_price[['open', 'high', 'low', 'close', 'volume']]

        # 将数据封装为Backtrader
        datafeed = bt.feeds.PandasData(dataname=df_price)
        cerebro.adddata(datafeed, name=symbol)

    # 设置参数网格, 进行多因子组合优化
    cerebro.optstrategy(
        MultiFactorStrategy,
        pe_weight=np.arange(-2, 2.1, 0.5),
        pb_weight=np.arange(-2, 2.1, 0.5),
        momentum_weight=np.arange(0, 2.1, 0.5),
        volatility_weight=np.arange(-2, 2.1, 0.5)
    )

    # 设置初始资金
    cerebro.broker.setcash(1000000)

    # 添加分析器: 年化收益率 与 夏普比率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    print(f"正在进行参数优化, 请等等.....\n")

    results = cerebro.run(maxcpus=1)    # 多线程不使用, 避免不稳定

    # 用于存储每组参数组合的回测结果
    results_list = []

    # 遍历每个优化结果
    for stratlist in results:
        strat = stratlist[0]
        params = strat.params
        rets = strat.analyzers.returns.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
        sharpe_val = sharpe if sharpe is not None else 0    # 防止为None

        # 打印结果
        print(f"参数组合: PE={params.pe_weight}, PB={params.pb_weight}, "
              f"Momentum={params.momentum_weight}, Volatility={params.volatility_weight}")
        print(f"年化收益率: {rets.get('rnorm100', 0):.2f}")
        print(f"夏普率: {sharpe_val:.2f}")
        print("_" * 40)

        # 保存结果到excel里.
        '''优化太多了大概有3000多,  不可能每次优化每次看. 只能保存起来'''
        result_dict = {
            "PE": params.pe_weight,
            'PB': params.pb_weight,
            'Momentum': params.momentum_weight,
            'Volatility': params.volatility_weight,
            'Annualized Return (%)': rets.get('rnorm100', 0),
            'Sharpe Ratio': sharpe_val
        }
        results_list.append(result_dict)

    # 转换为DataFrame保存到Excel
    df_results = pd.DataFrame(results_list)
    df_results.to_excel("Day5_parameter_optimization_results.xlsx", index=False)

    print(f"参数优化完成, 结果已保存到Day5_parameter_optimization_results.xlsx")

