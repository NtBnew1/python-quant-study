'''
第6天任务：

1. 加入止损、止盈规则，控制回撤和风险
2. 结合资金管理和仓位控制策略
3. 练习目标：完善多因子策略代码，实现风险管理功能，并回测验证
'''

# 导入相关库
import os  # 处理文件路径
import pandas as pd  # 用于读取Excel文件和数据处理
import numpy as np  # 数学计算库
import backtrader as bt  # 回测框架
from scipy.stats import zscore  # z-score标准化方法

# ========== Step 1: 读取并处理多因子数据 ==========
file_path = './Day4_factor_all_stocks.xlsx'  # 多因子Excel文件路径
df = pd.read_excel(file_path)  # 读取Excel

stock_col = df.columns[0]  # 获取股票代码所在列（第一列）
factor_cols = df.columns[1:]  # 获取除股票代码外的所有因子列

df_z = df[factor_cols].apply(zscore)  # 对所有因子进行z-score标准化

# 定义需要反向处理的因子（分数越低越好，因此取负）
reverse_factor = ['PE', 'PB', 'EV_EBITDA', 'Volatility', 'MaxDrawdown']
for col in reverse_factor:
    if col in df_z.columns:
        df_z[col] = -df_z[col]  # 将该因子取负，统一方向性（高分好）

df_z[stock_col] = df[stock_col]  # 恢复股票代码列

# 构建一个字典，存储每只股票的所有因子得分
factor_scores = {}
for _, row in df_z.iterrows():
    code = row[stock_col]  # 股票代码
    factor_scores[code] = {col: row[col] for col in factor_cols}  # 对应的因子分数组合

# 计算每只股票的综合得分：等权重相加
df_z['score'] = df_z[factor_cols].sum(axis=1)

# 选出得分前5名的股票
top5_df = df_z.sort_values(by='score', ascending=False).head(5)  # 按score降序排列
top5_stock_list = top5_df[stock_col].tolist()  # 获取Top 5股票代码
print(f"Top 5 stocks: {top5_stock_list}")  # 打印选中的股票代码列表


# ========== Step 2: 定义带风控的多因子策略 ==========

# 将因子得分设置为全局变量，供策略调用
GLOBAL_FACTOR_SCORE = factor_scores

# 定义策略类
class MultiFactorStrategy(bt.Strategy):
    params = (
        # 因子权重设置（可调）
        ('pe_weight', 1.0),
        ('pb_weight', 1.0),
        ('momentum_weight', 1.0),
        ('volatility_weight', 1.0),

        # 止损止盈控制参数
        ('stop_lost', 0.05),  # 单只股票亏损超过5%则止损
        ('take_profit', 0.10),  # 盈利超过10%则止盈

        # 仓位与资金管理参数
        ('max_position_pct', 0.1),  # 单只股票最大占比10%
        ('max_total_position', 0.7)  # 总仓位最大不能超过70%
    )

    def __init__(self):
        self.scores = GLOBAL_FACTOR_SCORE  # 获取因子分数字典
        self.order = None  # 当前订单状态
        self.buyprice = {}  # 记录每只股票的买入价（用于止盈止损判断）

    def next(self):
        if self.order:
            return  # 若当前有订单未完成，则跳过

        total_value = self.broker.getvalue()  # 当前账户总资产
        # 当前持仓市值总和
        current_position_value = sum([
            self.getposition(data).size * data.close[0] for data in self.datas
        ])
        current_position_pct = current_position_value / total_value if total_value > 0 else 0

        for data in self.datas:
            stock = data._name  # 股票代码

            if stock not in self.scores:
                continue  # 如果该股票没有得分，跳过

            pos = self.getposition(data)  # 当前持仓信息
            price = data.close[0]  # 当前收盘价

            # 计算该股票的综合得分（加权因子值）
            f = self.scores[stock]
            score = (
                self.params.pe_weight * f.get('PE', 0) +
                self.params.pb_weight * f.get('PB', 0) +
                self.params.momentum_weight * f.get('Momentum', 0) +
                self.params.volatility_weight * f.get('Volatility', 0)
            )

            # === 止损/止盈判断（已持仓） ===
            if pos.size > 0:
                buy_price = self.buyprice.get(stock, None)  # 获取买入价
                if buy_price is not None:
                    change_pct = (price - buy_price) / buy_price  # 计算涨跌幅
                    if change_pct < -self.params.stop_lost:  # 跌超5%
                        self.order = self.sell(data=data)  # 止损卖出
                        print(f"{stock}止损卖出, 跌幅{change_pct:.2%}")
                        self.buyprice.pop(stock, None)  # 删除记录
                        continue
                    if change_pct >= self.params.take_profit:  # 涨超10%
                        self.order = self.sell(data=data)  # 止盈卖出
                        print(f"{stock}止盈卖出, 涨幅{change_pct:.2%}")
                        self.buyprice.pop(stock, None)
                        continue

            # === 买入逻辑 ===
            if score > 0 and pos.size == 0:  # 得分正，未持仓
                if current_position_pct < self.params.max_total_position:  # 总仓位未超限
                    cash = self.broker.getcash()  # 当前现金
                    max_size = (total_value * self.params.max_position_pct) // price  # 可买股数
                    if max_size > 0 and cash > price:  # 有钱且能买
                        self.order = self.buy(data=data, size=max_size)
                        self.buyprice[stock] = price  # 记录买入价
                        print(f"{stock}买入, 仓位大小{max_size}股, 买入价{price}")
                else:
                    print('仓位达到上限, 暂不买入')

            # === 得分转负且已持仓，清仓 ===
            elif score < 0 and pos.size > 0:
                self.order = self.sell(data=data)
                self.buyprice.pop(stock, None)
                print(f"{stock} 多因子分数负, 卖出")


# ========== Step 3: 回测主程序入口 ==========

if __name__ == "__main__":
    cerebro = bt.Cerebro(optreturn=False)  # 创建回测引擎

    # === 加载Top5股票历史数据 ===
    for symbol in top5_stock_list:
        file_path = f"./{symbol}_all_data.xlsx"  # 每个股票的历史数据文件
        if not os.path.exists(file_path):
            print(f"找不到数据文件: {file_path}, 已跳过")
            continue

        df_price = pd.read_excel(file_path, index_col=0, parse_dates=True)  # 读取价格数据
        df_price.index.name = 'Unnamed: 0'  # 修正index名
        df_price = df_price[['open', 'high', 'low', 'close', 'volume']]  # 只保留五列

        datafeed = bt.feeds.PandasData(dataname=df_price)  # 转换为bt可用格式
        cerebro.adddata(datafeed, name=symbol)  # 添加到回测系统

    cerebro.addstrategy(MultiFactorStrategy)  # 添加策略
    cerebro.broker.setcash(1000000)  # 设置初始资金为100万

    # 添加分析器：夏普比率 + 年化收益率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    print("=======开始回测==========")
    results = cerebro.run()  # 执行回测
    strat = results[0]  # 获取策略对象

    # 提取分析器结果
    rets = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    # === 输出结果报告 ===
    init_cash = cerebro.broker.startingcash  # 初始资金
    final_cash = cerebro.broker.getvalue()  # 回测结束后的总资产

    print("=" * 50)
    print(f"初始资金: {init_cash:.2f}")
    print(f"结束资金: {final_cash:.2f}")
    print(f"总收益率: {(final_cash - init_cash) / init_cash * 100: .2f}%")
    print(f"年化收益率: {rets.get('rnorm100', 0): .2f}%")
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.2f}")
    print("=" * 50)

    cerebro.plot()  # 可视化回测结果


'''
====== 总结 ======

今天的任务是为多因子选股策略加入止损、止盈与仓位管理等风险控制模块。

我们首先对原始多因子数据进行了 z-score 标准化，并统一方向性（取负），用于后续打分。每只股票根据多个因子得分计算综合分数，并选出前5名股票进行回测。

策略中设定了：
- 止损阈值：下跌超过5%时止损
- 止盈阈值：上涨超过10%时止盈
- 仓位限制：单股最大仓位10%，总持仓不超过70%

回测过程中，我们使用 Backtrader 执行策略，对资金变化进行模拟，并输出收益率、夏普比率等指标。

通过本次实战，我们掌握了如何将风控逻辑与多因子策略结合，从而构建一个更稳健的量化交易系统。
'''