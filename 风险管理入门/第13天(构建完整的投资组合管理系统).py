'''
第13天：
整合投资组合优化、风险分析与回测模块。
练习：输出完整投资组合分析报告（包含图表和关键指标）。
'''

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
from scipy.optimize import minimize


warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CompletePortfolioAnalyzer:
    """
       完整的投资组合分析系统
       包含：优化 + 风险分析 + 回测 + 报告 + 图表
       """
    def __init__(self, stock_returns, risk_free_rate=0.03):
        """
                初始化分析器

                参数:
                stock_returns: 股票收益率DataFrame (日期为索引，股票为列)
                risk_free_rate: 无风险利率 (年化)
                """
        # 1. 存储输入数据
        self.stock_returns = stock_returns      # 存储股票收益率数据
        # 2. 转换无风险利率（年化→日利率，假设252个交易日）
        self.risk_free_rate = risk_free_rate / 252      # 转换为日利率
        # 3. 计算股票数量（列数）
        self.n_stocks = len(stock_returns.columns)
        # 4. 获取股票名称列表
        self.stock_names = stock_returns.columns.tolist()

        # 5. 初始化存储优化结果的变量
        self.max_sharpe_result = None       # 存储最大化夏普比率的结果
        self.min_vol_result = None          # 存储最小化波动率的结果
        self.efficient_frontier = None      # 存储有效前沿数据

        # 6. 计算基础统计指标（调用私有方法）
        self._calculate_basic_stats()

        # 7. 打印初始化完成信息
        print(f"🎯 完整投资组合分析器初始化完成")
        print(f"📊 包含股票: {self.n_stocks}只")  # 显示股票数量
        print(f"📅 数据期间: {stock_returns.index[0].date()} 到 {stock_returns.index[-1].date()}")  # 显示数据起止日期
        print(f"📈 交易日数: {len(stock_returns)}")  # 显示总交易日数

    def _calculate_basic_stats(self):
        """计算基础统计指标"""
        # 1. 计算每只股票的年化收益率：日收益率均值 × 252（年化因子）
        self.annual_returns = ( 1+ self.stock_returns.mean()) ** 252 - 1     # 正确计算年化收益率
        # 2. 计算每只股票的年化波动率：日收益率标准差 × √252
        self.annual_volatility = self.stock_returns.std() * np.sqrt(252)
        # 3. 计算每只股票的夏普比率：(年化收益率 - 无风险利率) / 年化波动率
        self.sharpe_ratios = (self.annual_returns - self.risk_free_rate * 252) / self.annual_volatility
        # 4. 计算协方差矩阵（年化）：日收益率协方差 × 252
        self.cov_matrix = self.stock_returns.cov() * 252
        # 5. 计算相关性矩阵：各股票收益率之间的相关系数（-1到1之间）
        self.corr_matrix = self.stock_returns.corr()
        # 6. 计算累计收益率：(1 + 日收益率) 的累积乘积
        self.cumulative_returns = (1 + self.stock_returns).cumprod()

    def optimize_portfolio(self, optimization_type='max_sharpe', constraints=None):
        """
                投资组合优化

                参数:
                optimization_type: 优化类型 ['max_sharpe', 'min_vol']
                constraints: 约束条件字典 {'max_weight': 0.3, 'min_weight': 0.01}
                """
        # 1. 打印优化过程开始信息，使用分隔线提高可读性
        print(f"\n{'=' * 50}")
        print(f"🔧 投资组合优化 ({optimization_type})")
        print('=' * 50)

        # 2. 设置初始权重：等权重分配（每只股票权重相同）
        initial_weights = np.ones(self.n_stocks) / self.n_stocks
        # 3. 定义权重边界：默认每只股票的权重在0到1之间（0-100%）
        bounds = tuple((0, 1) for _ in range(self.n_stocks))
        # 4. 定义基本约束条件：所有权重之和必须等于1（100%投资）
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # 5. 如果用户提供了额外约束，更新边界条件
        if constraints:
            if 'min_weight' in constraints:
                # 设置最小权重约束（避免持有极少量股票）
                bounds = tuple((constraints['min_weight'], 1) for _ in range(self.n_stocks))
            if 'max_weight' in constraints:
                # 设置最大权重约束（避免过度集中）
                bounds = tuple((0, constraints['max_weight']) for _ in range(self.n_stocks))

        # 6. 根据优化类型选择不同的目标函数
        if optimization_type == 'max_sharpe':
            # 6.1 最大化夏普比率的目标函数
            def objective(weights):
                # 计算投资组合的年化收益率
                port_return = np.sum(self.annual_returns * weights)
                # 计算投资组合的年化波动率：√(wᵀΣw)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                # 计算夏普比率
                sharpe = (port_return - self.risk_free_rate * 252) / port_vol
                # 返回负值，因为scipy.minimize是最小化函数
                return -sharpe

        elif optimization_type == 'min_vol':
            # 6.2 最小化波动率的目标函数
            def objective(weights):
                # 直接计算并返回投资组合的波动率
                port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
                return port_vol
        else:
            # 6.3 如果输入了不支持的优化类型，抛出错误
            raise ValueError(f"不支持的优化类型: {optimization_type}")

        # 7. 使用scipy的minimize函数执行优化计算
        result = minimize(
            objective,          # 目标函数（最大化夏普或最小化波动率）
            initial_weights,    # 优化的起始点（等权重）
            method='SLSQP',     # 优化算法：序列二次规划，适合有约束优化
            bounds=bounds,      # 权重边界条件（如0-1或用户定义的边界）
            constraints=constraints_list,  # 使用正确的约束列表
            options={'maxiter': 1000, 'ftol': 1e-9}  # 优化器设置
        )
        # 8. 检查优化是否成功
        if result.success:
            # 8.1 获取优化得到的最优权重
            optimal_weights = result.x
            # 8.2 对权重进行四舍五入，保留4位小数（提高可读性）
            optimal_weights = np.round(optimal_weights, 4)
            # 8.3 重新归一化权重（确保总和为100%，处理四舍五入误差）
            optimal_weights = optimal_weights / optimal_weights.sum()

            # 8.4 使用最优权重计算投资组合的各项指标
            # 计算投资组合年化收益率：权重与各股票收益率的加权和
            port_return = np.sum(self.annual_returns * optimal_weights)
            # 计算投资组合年化波动率：√(wᵀΣw)
            port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
            # 计算投资组合夏普比率：(收益-无风险利率)/波动率
            port_sharpe = (port_return - self.risk_free_rate * 252) / port_vol

            # 8.5 打印优化成功的结果
            print(f"✅ 优化成功！")
            print(f"📊 投资组合收益率: {port_return:.2%}")  # 格式化显示百分比
            print(f"📊 投资组合波动率: {port_vol:.2%}")
            print(f"📊 夏普比率: {port_sharpe:.3f}")

            # 8.6 创建权重字典：将股票名称与权重值对应
            weight_dict = dict(zip(self.stock_names, optimal_weights))

            # 8.7 将优化结果整理成字典格式，便于后续使用
            result_dict = {
                'weights': weight_dict,             # 股票权重字典
                'return': port_return,              # 预期收益率
                'volatility': port_vol,             # 预期波动率
                'sharpe': port_sharpe,              # 夏普比率
                'weights_array': optimal_weights    # 权重数组（原始格式）
            }

            # 8.8 根据优化类型将结果存储到对应的属性中
            if optimization_type == 'max_sharpe':
                self.max_sharpe_result = result_dict     # 存储最大夏普结果
            elif optimization_type == 'min_vol':
                self.min_vol_result = result_dict            # 存储最小波动率结果

            # 8.9 返回优化结果字典
            return result_dict
        else:
            # 9. 如果优化失败，打印错误信息并返回None
            print(f"❌ 优化失败: {result.message}")
            return None

    def calculate_efficient_frontier(self, n_points=20):
        """计算有效前沿"""
        # 1. 打印计算开始信息
        print(f"\n{'=' * 50}")
        print("📈 计算有效前沿")
        print('=' * 50)

        # 2. 确定有效前沿的收益率范围
        # 2.1 最小收益率：取所有股票最低收益率的80%（留有缓冲）
        min_return = self.annual_returns.min() * 0.8
        # 2.2 最大收益率：取所有股票最高收益率的120%（留有缓冲）
        max_return = self.annual_returns.max() * 1.2

        # 3. 生成目标收益率序列
        # 在最小和最大收益率之间生成n_points个均匀分布的点
        target_returns = np.linspace(min_return, max_return, n_points)
        # 4. 初始化存储有效前沿上各点的列表
        frontier_points = []

        # 5. 对每个目标收益率进行优化计算
        for target in target_returns:
            # 5.1 设置初始权重（等权重）
            initial_weights = np.ones(self.n_stocks) / self.n_stocks
            # 5.2 设置权重边界（0到1）
            bounds = tuple((0,1) for _ in range(self.n_stocks))
            # 5.3 设置约束条件：权重和为1 + 达到目标收益率
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},      # 权重和=1
                {'type': 'eq', 'fun': lambda x: np.sum(self.annual_returns * x) - target}    # 达到目标收益率
            ]

            # 5.4 定义目标函数：最小化波动率
            def objective(weights):
                # 计算投资组合波动率：√(wᵀΣw)
                return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

            # 5.5 执行优化
            result = minimize(
                objective,                  # 目标函数（最小化波动率）
                initial_weights,            # 初始权重
                method='SLSQP',             # 优化算法
                bounds=bounds,              # 权重边界
                constraints=constraints,    # 约束条件
                options={'maxiter': 1000, 'ftol': 1e-9}      # 优化器设置
            )

            # 5.6 如果优化成功，保存结果
            if result.success:
                # 归一化权重（处理计算误差）
                optimal_weights = result.x / result.x.sum()
                # 计算实际收益率（可能与目标略有差异）
                port_return = np.sum(self.annual_returns * optimal_weights)
                # 计算实际波动率
                port_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(self.cov_matrix, optimal_weights)))
                # 计算夏普比率
                port_sharpe = (port_return - self.risk_free_rate * 252) / port_vol

                # 5.7 将结果添加到列表中
                frontier_points.append({
                    'return': port_return,          # 收益率
                    'volatility': port_vol,         # 波动率
                    'sharpe': port_sharpe,          # 夏普比率
                    'weights': optimal_weights      # 权重分配
                })

        # 6. 将结果转换为DataFrame并存储到属性中
        self.efficient_frontier = pd.DataFrame(frontier_points)

        # 7. 打印计算结果摘要
        print(f"✅ 有效前沿计算完成")
        print(f"📊 点数: {len(self.efficient_frontier)}")
        print(
            f"📊 收益率范围: {self.efficient_frontier['return'].min():.2%} - "
            f"{self.efficient_frontier['return'].max():.2%}")
        print(
            f"📊 波动率范围: {self.efficient_frontier['volatility'].min():.2%} - "
            f"{self.efficient_frontier['volatility'].max():.2%}")

        # 8. 返回有效前沿数据
        return self.efficient_frontier

    def calculate_risk_metrics(self, weights):
        """
                计算风险指标

                参数:
                weights: 权重字典 {股票: 权重}
                """
        # 1. 打印风险指标计算开始信息
        print(f"\n{'=' * 50}")
        print("📊 风险指标计算")
        print('=' * 50)

        # 2. 将权重字典转换为权重数组（保持与股票顺序一致）
        weight_array = np.array([weights[stock] for stock in self.stock_names])

        # 3. 计算投资组合的基本指标
        # 3.1 年化收益率：各股票收益率的加权平均
        port_return = np.sum(self.annual_returns * weight_array)
        # 3.2 年化波动率：考虑协方差的加权组合风险
        port_vol = np.sqrt(np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array)))
        # 3.3 夏普比率：风险调整后的收益
        port_sharpe = (port_return - self.risk_free_rate * 252) / port_vol

        # 4. 计算投资组合的日收益率序列
        # 将每只股票的日收益率按权重加权求和
        port_returns_series = (self.stock_returns * weight_array).sum(axis=1)

        # 5. 计算风险价值（VaR） - 95%置信度
        # 5.1 计算日收益率的5%分位数（最坏的5%情况）
        var_daily = np.percentile(port_returns_series, 5)
        # 5.2 将日VaR年化：乘以√252
        var_95 = var_daily * np.sqrt(252)

        # 6. 计算最大回撤
        # 6.1 计算累计净值曲线：(1+收益率)的累积乘积
        cumulative = (1 + port_returns_series).cumprod()
        # 6.2 计算历史最高点序列（滚动最大值）
        running_max = cumulative.expanding().max()
        # 6.3 计算回撤：（当前净值-历史最高）/历史最高
        drawdown = (cumulative - running_max) / running_max
        # 6.4 找到最大回撤（最小值，因为回撤是负数）
        max_drawdown = drawdown.min()

        # 7. 计算Beta系数（系统性风险）
        # 7.1 假设投资组合本身作为市场基准（简化处理）
        market_returns = port_returns_series
        # 7.2 计算组合收益与市场收益的协方差
        cov_with_market = np.cov(port_returns_series, market_returns)[0, 1]
        # 7.3 计算市场收益的方差
        market_var = np.var(market_returns)
        # 7.4 计算Beta：协方差/方差
        beta = cov_with_market / market_var if market_var != 0 else 0

        # 8. 计算索提诺比率（只考虑下行风险）
        # 8.1 筛选出负收益率（下行风险）
        negative_returns = port_returns_series[port_returns_series < 0]
        # 8.2 计算下行波动率（负收益率的标准差）
        if len(negative_returns) > 0:
            downside_vol = negative_returns.std() * np.sqrt(252)
        else:
            downside_vol = 0
        # 8.3 计算索提诺比率：（收益-无风险利率）/下行波动率
        sortino_ratio = (port_return - self.risk_free_rate * 252) / downside_vol if downside_vol > 0 else 0

        # 9. 整理所有风险指标到字典中
        metrics = {
            '年化收益率': port_return,
            '年化波动率': port_vol,
            '夏普比率': port_sharpe,
            '索提诺比率': sortino_ratio,
            'VaR (95%)': var_95,
            '最大回撤': max_drawdown,
            'Beta系数': beta,
            '正收益天数比例': (port_returns_series >0).mean(),
            '平均日收益率': port_returns_series.mean(),
            '日收益率波动率': port_returns_series.std()
        }

        # 10. 打印关键风险指标
        print(f"📈 年化收益率: {port_return:.2%}")
        print(f"📉 年化波动率: {port_vol:.2%}")
        print(f"🎯 夏普比率: {port_sharpe:.3f}")
        print(f"📊 索提诺比率: {sortino_ratio:.3f}")
        print(f"⚠️  最大回撤: {max_drawdown:.2%}")
        print(f"💸 VaR (95%): {var_95:.2%}")
        print(f"📊 Beta系数: {beta:.3f}")
        print(f"📈 正收益天数比例: {(port_returns_series > 0).mean():.1%}")

        # 11. 返回风险指标字典
        return metrics

    def calculate_risk_contribution(self, weights):
        """
                计算风险贡献 (Brinson模型)

                参数:
                weights: 权重字典 {股票: 权重}
                """
        # 1. 打印风险贡献分析开始信息
        print(f"\n{'=' * 50}")
        print("📊 风险贡献分析 (Brinson模型)")
        print('=' * 50)

        # 2. 将权重字典转换为权重数组（保持股票顺序一致）
        weight_array = np.array([weights[stock] for stock in self.stock_names])

        # 3. 计算投资组合的总风险（波动率）
        # 3.1 计算投资组合方差：wᵀΣw
        portfolio_variance = np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array))
        # 3.2 计算投资组合波动率（标准差）：√方差
        portfolio_volatility = np.sqrt(portfolio_variance)

        # 4. 计算边际风险贡献
        # 边际风险贡献 = Σw / σₚ （协方差矩阵与权重的乘积除以总波动率）
        marginal_risk = self.cov_matrix @ weight_array / portfolio_volatility

        # 5. 计算绝对风险贡献
        # 绝对风险贡献 = 权重 × 边际风险贡献
        absolute_contributions = weight_array * marginal_risk

        # 6. 计算相对风险贡献（百分比）
        # 6.1 计算总风险贡献（所有绝对风险贡献之和）
        total_risk_contribution = np.sum(absolute_contributions)
        # 6.2 计算每只股票的相对风险贡献
        relative_contributions = absolute_contributions / total_risk_contribution

        # 7. 创建风险贡献分析DataFrame
        risk_df = pd.DataFrame({
            '股票': self.stock_names,                      # 股票名称
            '权重': weight_array,                          # 投资权重
            '年化波动率': self.annual_volatility.values,     # 个股波动率
            '绝对风险贡献': absolute_contributions,           # 对组合风险的绝对贡献
            '相对风险贡献': relative_contributions,           # 对组合风险的相对贡献（%）
            '风险倍数': relative_contributions / weight_array       # 风险贡献/权重 比值
        })

        # 8. 按相对风险贡献从高到低排序
        risk_df = risk_df.sort_values('相对风险贡献', ascending=False)

        # 9. 打印风险贡献分析结果
        print(f"📊 总投资组合风险: {portfolio_volatility:.2%}")
        print(f"📊 总风险贡献: {total_risk_contribution:.2%}")
        print(f"\n🎯 前5大风险贡献者:")

        # 10. 显示前5大风险来源
        for i, row in risk_df.head(5).iterrows():
            print(f"  {row['股票']}: 权重{row['权重']:.1%}, 风险贡献{row['相对风险贡献']:.1%}, 风险倍数{row['风险倍数']:.2f}x")

        # 11. 识别高风险和低风险股票
        # 11.1 高风险股票：风险倍数 > 1.5（风险贡献显著高于权重）
        high_risk = risk_df[risk_df['风险倍数'] > 1.5]
        # 11.2 低风险股票：风险倍数 < 0.7（风险贡献显著低于权重）
        low_risk = risk_df[risk_df['风险倍数'] < 0.7]

        # 12. 输出高风险股票警告
        if len(high_risk) > 0:
            print(f"\n⚠️  高风险股票 (风险贡献显著高于权重):")
            for _, row in high_risk.iterrows():
                print(f"  {row['股票']}: 风险倍数{row['风险倍数']:.2f}x")

        # 13. 输出低风险股票信息
        if len(low_risk) > 0:
            print(f"\n✅ 低风险股票 (风险贡献显著低于权重):")
            for _, row in low_risk.iterrows():
                print(f"  {row['股票']}: 风险倍数{row['风险倍数']:.2f}x")

        # 14. 计算风险集中度指数（赫芬达尔-赫希曼指数）
        # 风险贡献百分比的平方和，值越大表示风险越集中
        herfindahl_index = (risk_df['相对风险贡献'] ** 2).sum()
        print(f"\n📊 风险集中度指数: {herfindahl_index:.3f}")

        # 15. 根据风险集中度给出建议
        if herfindahl_index > 0.25:
            print("  🎯 风险集中度较高，建议进一步分散")
        elif herfindahl_index > 0.15:
            print("  🎯 风险集中度适中")
        else:
            print("  🎯 风险分散度良好")

        # 16. 返回风险贡献分析DataFrame
        return risk_df

    def backtest_portfolio(self, weights, initial_capital=10000, rebalance_freq='Q'):
        """
                回测投资组合表现

                参数:
                weights: 权重字典
                initial_capital: 初始资本
                rebalance_freq: 再平衡频率 ['M'=月, 'Q'=季, 'Y'=年]
                """
        # 1. 打印回测开始信息
        print(f"\n{'=' * 50}")
        print(f"📈 投资组合回测分析")
        print('=' * 50)

        # 2. 准备数据
        returns = self.stock_returns.copy()     # 复制收益率数据，避免修改原数据
        weight_array = np.array([weights[stock] for stock in self.stock_names]) # 权重数组

        # 3. 初始化变量
        current_weights = weight_array.copy()    # 当前权重（随时间变化）
        capital = initial_capital            # 当前资本
        capital_history = [capital]          # 资本历史记录
        date_history = [returns.index[0]]   # 日期历史记录
        weight_history = [current_weights.copy()]    # 权重历史记录

        # 4. 确定再平衡频率
        # 根据输入参数设置pandas的重采样频率
        if rebalance_freq == 'M':
            freq = 'MS'     # 每月开始
        elif rebalance_freq == 'Q':
            freq = 'QS'     # 每季开始
        elif rebalance_freq == 'Y':
            freq = 'YS'     # 每年开始
        else:
            freq = 'QS'     # 默认季度再平衡

        # 5. 生成再平衡日期序列
        # 从数据开始到结束，按指定频率生成再平衡日期
        rebalance_dates = pd.date_range(
            start=returns.index[0],
            end=returns.index[-1],
            freq=freq
        )

        # 6. 执行回测（逐日模拟）
        rebalance_count = 0 # 再平衡次数计数器
        for i in range(1, len(returns)):        # 从第2天开始
            current_date = returns.index[i]      # 当前日期
            # 6.1 计算当日投资组合收益率：各股票收益率按权重加权求和
            daily_return = np.sum(returns.iloc[i] * current_weights)
            # 6.2 更新资本：按收益率增长
            capital *= (1 + daily_return)
            # 6.3 记录历史数据
            capital_history.append(capital)
            date_history.append(current_date)
            weight_history.append(current_weights.copy())
            # 6.4 检查是否需要再平衡
            if current_date in rebalance_dates:
                # 恢复为目标权重（再平衡）
                current_weights = weight_array.copy()
                rebalance_count += 1    # 计数加1

        # 7. 创建回测结果DataFrame
        backtest_df = pd.DataFrame({
            'date': date_history,           # 日期序列
            'capital': capital_history      # 资本序列
        }).set_index('date')            # 设置日期为索引

        # 8. 计算回测绩效指标
        # 8.1 总收益率：（最终资本/初始资本）-1
        total_return = (capital_history[-1] / initial_capital) - 1
        # 8.2 年化收益率：(1+总收益率)^(252/天数) - 1
        annualized_return = ( 1+ total_return) ** (252/len(returns)) - 1
        # 8.3 计算日收益率序列（用于计算波动率）
        returns_series = backtest_df['capital'].pct_change().dropna()
        # 8.4 年化波动率：日收益率标准差×√252
        volatility = returns_series.std() * np.sqrt(252)
        # 8.5 夏普比率：（年化收益-无风险利率）/波动率
        sharpe = (annualized_return - self.risk_free_rate * 252) / volatility

        # 9. 计算最大回撤
        # 9.1 计算累计净值
        cumulative = (1 + returns_series).cumprod()
        # 9.2 计算历史最高点（滚动最大值）
        running_max = cumulative.expanding().max()
        # 9.3 计算回撤率
        drawdown = (cumulative - running_max) / running_max
        # 9.4 找到最大回撤（最小值）
        max_drawdown = drawdown.min()
        # 9.5 找到最大回撤发生日期
        max_dd_period = (drawdown == max_drawdown).idxmax() if len(drawdown) > 0 else None

        # 10. 整理回测指标到字典
        backtest_metrics = {
            '总收益率': total_return,                       # 整个期间的总收益
            '年化收益率': annualized_return,                 # 折算到每年的收益
            '年化波动率': volatility,                        # 风险水平
            '夏普比率': sharpe,                             # 风险调整后收益
            '最大回撤': max_drawdown,                       # 最大亏损幅度
            '最大回撤日期': max_dd_period,                    # 最大回撤发生时间
            '最终资本': capital_history[-1],                # 回测结束时的资本
            '再平衡次数': rebalance_count,                   # 再平衡操作次数
            '盈利天数比例': (returns_series > 0).mean(),      # 赚钱天数比例
            '最大单日涨幅': returns_series.max(),             # 最好的一天
            '最大单日跌幅': returns_series.min(),             # 最差的一天
            '平均日收益率': returns_series.mean()             # 日均收益
        }

        # 11. 打印回测结果
        print(f"💰 初始资本: ${initial_capital:,}")
        print(f"💰 最终资本: ${capital_history[-1]:,.2f}")
        print(f"📈 总收益率: {total_return:.2%}")
        print(f"📈 年化收益率: {annualized_return:.2%}")
        print(f"📉 年化波动率: {volatility:.2%}")
        print(f"🎯 夏普比率: {sharpe:.3f}")
        print(f"⚠️  最大回撤: {max_drawdown:.2%}")
        if max_dd_period:
            print(f"📅 最大回撤日期: {max_dd_period.date()}")
        print(f"🔄 再平衡次数: {rebalance_count}")
        print(f"📊 盈利天数比例: {(returns_series > 0).mean():.1%}")
        print(f"📈 最大单日涨幅: {returns_series.max():.2%}")
        print(f"📉 最大单日跌幅: {returns_series.min():.2%}")

        # 12. 返回回测数据和指标
        return backtest_df, backtest_metrics

    def generate_comprehensive_report(self, weights, benchmark_weights=None):
        """
                生成完整的投资组合分析报告

                参数:
                weights: 投资组合权重
                benchmark_weights: 基准组合权重 (可选)
                """
        # 1. 打印报告生成开始信息
        print(f"\n{'=' * 80}")
        print("📋 投资组合综合分析报告")
        print("=" * 80)

        # 2. 第一部分：基本统计信息
        print("\n1️⃣ 基本统计信息")
        print("-" * 40)

        # 2.1 创建基本统计DataFrame
        stats_df = pd.DataFrame({
            '股票': self.stock_names, # 股票名称
            '年化收益率': [self.annual_returns[s] for s in self.stock_names],    # 各股票年化收益
            '年化波动率': [self.annual_volatility[s] for s in self.stock_names],  # 各股票年化波动
            '夏普比率': [self.sharpe_ratios[s] for s in self.stock_names],      # 各股票夏普比率
            '权重': [weights.get(s, 0) for s in self.stock_names]     # 投资组合中的权重
        })

        # 2.2 如果有基准权重，添加基准相关列
        if benchmark_weights:
            stats_df['基准权重'] = [benchmark_weights.get(s, 0) for s in self.stock_names]
            stats_df['主动权重'] = stats_df['权重'] - stats_df['基准权重']     # 主动管理部分

        # 2.3 显示前10只股票（按权重降序）
        print(stats_df.sort_values('权重', ascending=False).head(10).round(4).to_string())

        # 3. 第二部分：投资组合指标
        print("\n2️⃣ 投资组合指标")
        print("-" * 40)
        # 调用之前定义的calculate_risk_metrics方法计算风险指标
        port_metrics = self.calculate_risk_metrics(weights)

        # 4. 第三部分：风险贡献分析
        print("\n3️⃣ 风险贡献分析")
        print("-" * 40)
        # 调用之前定义的calculate_risk_contribution方法分析风险贡献
        risk_df = self.calculate_risk_contribution(weights)

        # 5. 第四部分：回测结果
        print("\n4️⃣ 回测结果")
        print("-" * 40)
        # 调用之前定义的backtest_portfolio方法进行回测
        backtest_df, backtest_metrics = self.backtest_portfolio(weights)

        # 6. 第五部分：优化对比
        print("\n5️⃣ 优化对比")
        print("-" * 40)

        # 6.1 如果已经计算了最大夏普组合，显示对比
        if self.max_sharpe_result:
            print(f"🎯 最大夏普组合:")
            print(f"   收益率: {self.max_sharpe_result['return']:.2%}")
            print(f"   波动率: {self.max_sharpe_result['volatility']:.2%}")
            print(f"   夏普比率: {self.max_sharpe_result['sharpe']:.3f}")

            # 6.2 获取当前配置的指标
            current_return = port_metrics['年化收益率']
            current_vol = port_metrics['年化波动率']
            current_sharpe = port_metrics['夏普比率']

            # 6.3 对比当前配置与最优配置
            print(f"\n📊 当前配置 vs 最优配置:")
            print(f"   收益率差距: {current_return - self.max_sharpe_result['return']:+.2%}")
            print(f"   波动率差距: {current_vol - self.max_sharpe_result['volatility']:+.2%}")
            print(f"   夏普比率差距: {current_sharpe - self.max_sharpe_result['sharpe']:+.3f}")

        # 7. 第六部分：投资建议
        print("\n6️⃣ 投资建议")
        print("-" * 40)
        # 调用生成投资建议的方法
        self.generate_investment_advice(weights, risk_df, port_metrics)

        # 8. 打印报告完成信息
        print("\n" + "=" * 80)
        print("📋 报告生成完成！")
        print("=" * 80)

        # 9. 返回所有分析结果（便于进一步处理）
        return {
            'stats': stats_df,              # 基本统计信息
            'portfolio_metrics': port_metrics, # 投资组合指标
            'risk_analysis': risk_df,       # 风险贡献分析
            'backtest_metrics': backtest_metrics,    # 回测指标
            'backtest_data': backtest_df         # 回测数据
        }

    def generate_investment_advice(self, weights, risk_df, port_metrics):
        """生成投资建议"""
        # 1. 分析高风险股票（风险倍数 > 1.5）
        # 风险倍数 = 相对风险贡献 / 权重，>1.5表示风险贡献显著高于权重
        high_risk_stocks = risk_df[risk_df['风险倍数'] > 1.5]
        # 分析低风险股票（风险倍数 < 0.7）
        low_risk_stocks = risk_df[risk_df['风险倍数'] < 0.7]

        # 2. 高风险股票建议（建议减仓）
        if len(high_risk_stocks) > 0:
            print(f"⚠️  建议减仓的股票 (风险过高):")
            for _, row in high_risk_stocks.iterrows():
                current_weight = row['权重']  # 当前权重
                # 建议权重：降低到风险倍数=1的水平（风险贡献与权重匹配）
                suggested_weight = current_weight / row['风险倍数']
                # 计算需要减少的百分比
                reduction = (current_weight - suggested_weight) * 100
                print(f"  {row['股票']}: 当前{current_weight:.1%} → 建议{suggested_weight:.1%} (减少{reduction:.1f}%)")

        # 3. 低风险股票建议（建议加仓）
        if len(low_risk_stocks) > 0:
            print(f"\n✅ 建议加仓的股票 (风险利用不足):")
            for _, row in low_risk_stocks.iterrows():
                current_weight = row['权重']  # 当前权重
                # 建议权重：增加50%（充分利用低风险特性）
                suggested_weight = current_weight * 1.5
                # 计算需要增加的百分比
                increase = (suggested_weight - current_weight) * 100
                print(f"  {row['股票']}: 当前{current_weight:.1%} → 建议{suggested_weight:.1%} (增加{increase:.1f}%)")

        # 4. 基于夏普比率的建议
        current_sharpe = port_metrics['夏普比率']    # 当前投资组合的夏普比率

        if current_sharpe < 0.5:
            print(f"\n🎯 风险调整收益偏低 (夏普比率{current_sharpe:.3f})")
            print("  建议: 增加低波动资产，减少高风险股票")
        elif current_sharpe < 1.0:
            print(f"\n🎯 风险调整收益适中 (夏普比率{current_sharpe:.3f})")
            print("  建议: 保持当前配置，定期再平衡")
        else:
            print(f"\n🎯 风险调整收益优秀 (夏普比率{current_sharpe:.3f})")
            print("  建议: 当前配置良好，继续持有")

        # 5. 基于最大回撤的建议
        max_dd = port_metrics['最大回撤']       # 当前投资组合的最大回撤

        if max_dd < -0.20:  # 最大回撤超过-20%
            print(f"\n⚠️  风险控制需要加强 (最大回撤{max_dd:.2%})")
            print("  建议: 设置止损，增加防御性资产")
        elif max_dd < -0.10:    # 最大回撤在-10%到-20%之间
            print(f"\n📊 风险控制适中 (最大回撤{max_dd:.2%})")
            print("  建议: 监控高风险资产，保持流动性")
        else:   # 最大回撤小于-10%
            print(f"\n✅ 风险控制优秀 (最大回撤{max_dd:.2%})")
            print("  建议: 当前风险控制良好")

        # 6. 总体建议
        print(f"\n🔧 总体建议:")
        print("  1. 每季度重新平衡投资组合")
        print("  2. 定期监控风险贡献度")
        print("  3. 关注高风险股票的表现")
        print("  4. 根据市场环境调整风险预算")

    def plot_comprehensive_analysis(self, weights, backtest_df=None):
        """
                绘制完整的分析图表 (5张分图)

                参数:
                weights: 投资组合权重
                backtest_df: 回测数据
                """
        # 1. 打印图表生成开始信息
        print(f"\n{'=' * 50}")
        print("🎨 生成分析图表 (5张分图)")
        print('=' * 50)

        # ==================== 图1: 权重和收益波动 ====================
        # 创建第一个图形（1行2列）
        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig1.suptitle('图1: 投资组合权重和收益波动分析', fontsize=16, fontweight='bold')

        # 1.1 权重分布图（左图）
        # 将权重转换为Series并按权重降序排序
        weight_series = pd.Series(weights).sort_values(ascending=False)
        # 生成颜色：使用Set3颜色映射，为每只股票生成不同颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(weight_series)))
        # 绘制柱状图
        bars = ax1.bar(range(len(weight_series)), weight_series.values, color=colors)

        # 设置图表属性
        ax1.set_title('投资组合权重分布', fontsize=14, fontweight='bold')
        ax1.set_xlabel('股票')
        ax1.set_ylabel('权重 (%)')
        ax1.set_xticks(range(len(weight_series)))
        ax1.set_xticklabels(weight_series.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # 在柱状图上添加数值标签（权重大于1%才显示）
        for i, (bar, weight) in enumerate(zip(bars, weight_series.values)):
            if weight > 0.01:    # 只显示权重大于1%的标签
                ax1.text(i, weight + 0.01, f'{weight:.1%}',      # 在柱子上方显示百分比
                         ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 1.2 收益率vs波动率散点图（右图）
        # 绘制散点图：x=波动率，y=收益率，点大小=夏普比率×300
        scatter = ax2.scatter(self.annual_volatility, self.annual_returns,
                              s=self.sharpe_ratios * 300, alpha=0.6,
                              c=self.sharpe_ratios, cmap='RdYlGn', edgecolors='black')

        # 标记当前投资组合位置
        # 计算当前投资组合的收益率和波动率
        weight_array = np.array([weights[s] for s in self.stock_names])
        port_return = np.sum(self.annual_returns * weight_array)
        port_vol = np.sqrt(np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array)))

        # 用红色五角星标记当前组合
        ax2.scatter([port_vol], [port_return], s=300, marker='*',
                    color='red', edgecolors='black', linewidth=2, label='当前组合')

        # 设置图表属性
        ax2.set_title('收益率 vs 波动率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('年化波动率 (%)')
        ax2.set_ylabel('年化收益率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 添加颜色条（显示夏普比率颜色映射）
        cbar2 = plt.colorbar(scatter, ax=ax2)
        cbar2.set_label('夏普比率', fontsize=10)
        plt.tight_layout()
        plt.show()

        # ==================== 图2: 相关性和风险贡献 ====================
        # 创建第二个图形（1行2列）
        fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(16,6))
        fig2.suptitle('图2: 相关性和风险贡献分析', fontsize=16, fontweight='bold')

        # 2.1 相关性热图（左图）- 只显示前10只股票
        top_n = min(10, len(self.stock_names))  # 确定显示数量（最多10只）
        top_stocks = list(pd.Series(weights).nlargest(top_n).index)     # 选取权重最大的股票
        corr_top = self.corr_matrix.loc[top_stocks, top_stocks]      # 获取相关性矩阵子集

        # 绘制热图
        im = ax3.imshow(corr_top.values, cmap='coolwarm', vmin=1, vmax=1)

        # 设置图表属性
        ax3.set_title(f'前{top_n}只股票相关性热图', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(top_stocks)))
        ax3.set_yticks(range(len(top_stocks)))
        ax3.set_xticklabels(top_stocks, rotation=45, ha='right', fontsize=9)
        ax3.set_yticklabels(top_stocks, fontsize=9)

        # 在热图单元格中添加相关系数值
        for i in range(len(top_stocks)):
            for j in range(len(top_stocks)):
                corr_value = corr_top.iloc[i, j]
                ax3.text(j, i, f'{corr_value:.2f}',
                         ha='center', va='center',
                         color='white' if abs(corr_value) > 0.5 else 'black',    # 根据背景调整文字颜色
                         fontsize=8)
        # 添加颜色条
        cbar3 = plt.colorbar(im, ax=ax3)
        cbar3.set_label('相关系数', fontsize=10)

        # 2.2 风险贡献饼图（右图）
        # 计算风险贡献
        marginal_risk = self.cov_matrix @ weight_array
        total_risk = np.sqrt(np.dot(weight_array.T, np.dot(self.cov_matrix, weight_array)))
        risk_contributions = weight_array * marginal_risk / total_risk
        risk_share = risk_contributions / risk_contributions.sum()

        # 转换为Series并按风险贡献排序
        risk_series = pd.Series(risk_share, index=self.stock_names).sort_values(ascending=False)

        # 只显示主要风险贡献者（前8个），其余合并为"其他"
        top_risk = risk_series.head(min(8, len(risk_series)))
        if len(risk_series) > 8:
            other_risk = risk_series[8:].sum()
            # 创建Series，而不是列表
            other_series = pd.Series([other_risk], index=['其它'])
            top_risk=pd.concat([top_risk, other_series])        # 连接两个Series

        # 设置饼图突出显示（第一个扇区突出0.1）
        explode = [0.1 if i ==0 else 0 for i in range(len(top_risk))]
        # 生成颜色
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(top_risk)))

        # 绘制饼图
        wedges, texts, autotexts = ax4.pie(top_risk.values, labels=top_risk.index,
                                           autopct='%1.1f%%', startangle=90,
                                           explode=explode, shadow=True,
                                           colors=colors)
        ax4.set_title('风险贡献分布', fontsize=14, fontweight='bold')

        # 美化饼图文本（设置字体和颜色）
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        plt.tight_layout()
        plt.show()

        # ==================== 图3: 有效前沿和累计收益 ====================
        # 创建第三个图形（1行2列）
        fig3, (ax5, ax6) = plt.subplots(1,2, figsize=(16,6))
        fig3.suptitle('图3: 有效前沿和累计收益率', fontsize=16, fontweight='bold')

        # 3.1 有效前沿图（左图）
        # 如果还没有计算有效前沿，先计算
        if self.efficient_frontier is None:
            self.calculate_efficient_frontier()

        # 绘制有效前沿曲线
        if self.efficient_frontier is not None and not self.efficient_frontier.empty:
            ax5.plot(self.efficient_frontier['volatility'], self.efficient_frontier['return'],
                     'b-', linewidth=2, alpha=0.7, label='有效前沿')
            # 标记关键点：最小波动率组合
            min_vol_idx = self.efficient_frontier['volatility'].idxmin()
            ax5.scatter(self.efficient_frontier.loc[min_vol_idx, 'volatility'],
                        self.efficient_frontier.loc[min_vol_idx, 'return'],
                        s=200, color='green', marker='o',
                        label='最小波动率组合', edgecolors='black', linewidth=2)

            # 标记关键点：最大夏普比率组合
            max_sharpe_idx = self.efficient_frontier['sharpe'].idxmax()
            ax5.scatter(self.efficient_frontier.loc[max_sharpe_idx, 'volatility'],
                        self.efficient_frontier.loc[max_sharpe_idx, 'return'],
                        s=200, color='gold', marker='s',
                        label='最大夏普比率组合', edgecolors='black', linewidth=2)
            # 标记当前组合位置
            ax5.scatter([port_vol], [port_return], s=300, marker='*', color='red',
                        edgecolors='black', linewidth=2, label='当前组合')
            # 设置图表属性
            ax5.set_title('有效前沿', fontsize=14, fontweight='bold')
            ax5.set_xlabel('波动率 (%)')
            ax5.set_ylabel('收益率 (%)')
            ax5.legend(loc='best')
            ax5.grid(True, alpha=0.3)

        # 3.2 累计收益率比较图（右图）- 如果有回测数据
        if backtest_df is not None:
            # 投资组合累计收益率：归一化到起始点1
            port_cumulative = backtest_df['capital'] / backtest_df['capital'].iloc[0]
            ax6.plot(port_cumulative.index, port_cumulative.values, 'b-',
                     linewidth=2, label='投资组合', alpha=0.8)

            # 等权重基准：作为对比基准
            equal_weights = {s: 1 / len(self.stock_names) for s in self.stock_names}
            equal_array = np.array([equal_weights[s] for s in self.stock_names])
            equal_returns = (self.stock_returns * equal_array).sum(axis=1)
            equal_cumulative = (1+ equal_returns).cumprod()
            ax6.plot(equal_cumulative.index, equal_cumulative.values, 'r--',
                     linewidth=2, alpha=0.7, label='等权重基准')

            # 无风险基准：显示无风险投资的表现
            risk_free_cumulative = (1+ self.risk_free_rate) ** np.arange(len(equal_cumulative))
            ax6.plot(equal_cumulative.index, risk_free_cumulative, 'g', linewidth=2,
                     alpha=0.6, label='无风险利率')
            # 设置图表属性
            ax6.set_title('累计收益率对比', fontsize=14, fontweight='bold')
            ax6.set_xlabel('日期')
            ax6.set_ylabel('累计收益率')
            ax6.legend(loc='best')
            ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # ==================== 图4: 回撤和月度收益 ====================
        # 如果有回测数据，创建第四个图形（回撤分析和月度收益）
        if backtest_df is not None:
            fig4, (ax7, ax8) = plt.subplots(1,2, figsize=(16,6))
            fig4.suptitle('图4: 回撤分析和月度收益', fontsize=16, fontweight='bold')

            # 4.1 回撤图（左图）
            # 计算日收益率序列
            port_returns = backtest_df['capital'].pct_change().dropna()
            # 计算累计净值
            cumulative = ( 1+ port_returns).cumprod()
            # 计算历史最高点
            running_max = cumulative.expanding().max()
            # 计算回撤率
            drawdown = (cumulative - running_max) / running_max

            # 填充回撤区域（红色区域表示亏损）
            ax7.fill_between(drawdown.index, 0, drawdown.values,
                             color='red', alpha=0.3, label='回撤')
            # 绘制回撤曲线
            ax7.plot(drawdown.index, drawdown.values, 'r-', linewidth=1, alpha=0.7)
            # 添加零线
            ax7.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            # 标记最大回撤点
            max_dd = drawdown.min() # 最大回撤值
            max_dd_date = drawdown.idxmin() # 最大回撤发生日期
            ax7.scatter([max_dd_date], [max_dd], s=100, color='darkred', marker='x',
                        linewidth=2, label=f'最大回撤: {max_dd:.2%}') # 在图例中显示具体数值
            # 设置图表属性
            ax7.set_title('回撤分析', fontsize=14, fontweight='bold')
            ax7.set_xlabel('日期')
            ax7.set_ylabel('回撤 (%)')
            ax7.legend(loc='best')
            ax7.grid(True, alpha=0.3)

            # 4.2 月度收益率热图（右图）
            # 将日收益率重采样为月收益率
            monthly_returns = port_returns.resample('M').apply(lambda x: (1+x).prod() -1)

            # 创建月度收益矩阵（年份×月份）
            monthly_df = pd.DataFrame({
                'year': monthly_returns.index.year,
                'month': monthly_returns.index.month,
                'return': monthly_returns.values
            })

            # 将数据透视成年份×月份的矩阵形式
            monthly_matrix = monthly_df.pivot(index='year', columns='month', values='return')
            # 确保包含所有12个月份
            monthly_matrix = monthly_matrix.reindex(columns=range(1, 13))
            # 确保所有月份都有列（用NaN填充缺失的月份）
            for month in range(1, 13):
                if month not in monthly_matrix.columns:
                    monthly_matrix[month] = np.nan

            # 按月份排序
            monthly_matrix = monthly_matrix[sorted(monthly_matrix.columns)]
            # 绘制热图
            im8 = ax8.imshow(monthly_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=0.2)

            # 在热图单元格中添加月收益率数值
            for i in range(monthly_matrix.shape[0]):     # 遍历行（年份）
                for j in range(monthly_matrix.shape[1]): # 遍历列（月份）
                    if not pd.isna(monthly_matrix.iloc[i, j]):  # 如果不是NaN
                        return_value = monthly_matrix.iloc[i, j]
                        # 根据背景深浅调整文字颜色
                        color= 'white' if abs(return_value) > 0.1 else 'black'
                        ax8.text(j, i, f'{return_value:.1%}',
                                 ha='center', va='center', color=color, fontsize=8, fontweight='bold')

            # 设置图表属性
            ax8.set_title('月度收益率热图', fontsize=14, fontweight='bold')
            ax8.set_xlabel('月份')
            ax8.set_ylabel('年份')
            ax8.set_xticks(range(12))
            ax8.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8',
                                 '9', '10', '11', '12'], fontsize=9)
            ax8.set_yticks(range(len(monthly_matrix.index)))
            ax8.set_yticklabels(monthly_matrix.index, fontsize=9)

            # 添加颜色条
            cbar8 = plt.colorbar(im8, ax=ax8)
            cbar8.set_label('月收益率', fontsize=10)

            # 显示第四个图形
            plt.tight_layout()
            plt.show()

        # ==================== 图5: 风险收益比 ====================
        # 创建第五个图形（单独一图）
        fig5, ax9 = plt.subplots(1, 1, figsize=(16,6))
        fig5.suptitle('图5: 各股票风险收益比分析', fontsize=16, fontweight='bold')
        # 计算各股票的夏普比率（风险收益比）并排序
        risk_reward_ratios = self.sharpe_ratios.sort_values()
        # 创建水平条形图
        y_pos = np.arange(len(risk_reward_ratios))
        bars = ax9.barh(y_pos, risk_reward_ratios.values)

        # 根据夏普比率正负设置颜色
        for i, bar in enumerate(bars):
            value = risk_reward_ratios.iloc[i]
            if value >= 0:
                bar.set_color('green')  # 正夏普比率用绿色
                bar.set_alpha(0.7)
            else:
                bar.set_color('red')    # 负夏普比率用红色
                bar.set_alpha(0.7)

            # 在条形末端添加数值标签
            if value >= 0:
                ax9.text(value + 0.01 if value > 0 else 0.01, i,
                         f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
            else:
                ax9.text(value - 0.01, i, f'{value:.3f}', va='center', fontsize=9,
                         fontweight='bold')

        # 在当前投资组合中的股票旁边添加星号标记
        for i, stock in enumerate(risk_reward_ratios.index):
            if weights.get(stock, 0) > 0.01: # 权重大于1%的股票
                # 根据夏普比率正负决定星号位置
                position = -0.2 if risk_reward_ratios[stock] < 0 else -0.1
                ax9.text(position, i, '★', va='center', fontsize=12, color='gold', fontweight='bold')

        # 设置图表属性
        ax9.set_yticks(y_pos)
        ax9.set_yticklabels(risk_reward_ratios.index, fontsize=10)
        ax9.set_xlabel('夏普比率', fontsize=12)
        ax9.set_title('各股票风险收益比 (★ 表示在投资组合中的股票)',
                      fontsize=14, fontweight='bold')
        # 添加零线
        ax9.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax9.grid(True, alpha=0.3, axis='x')      # 仅x轴添加网格
        # 添加自定义图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='正夏普比率'),
            Patch(facecolor='red', alpha=0.7, label='负夏普比率'),
            Patch(facecolor='white', label='★ 表示在投资组合中')
        ]
        ax9.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.show()
        # 打印完成信息
        print("✅ 5张分析图表生成完成！")

    def run_complete_analysis(self, weights, benchmark_weights=None):
        """
                运行完整的分析流程

                参数:
                weights: 投资组合权重
                benchmark_weights: 基准组合权重
                """
        # 1. 打印完整分析开始信息
        print(f"\n{'=' * 80}")
        print("🚀 开始完整投资组合分析")
        print("=" * 80)

        # 2. 第一步：投资组合优化
        print("\n📋 第1步: 投资组合优化")
        # 2.1 执行最大化夏普比率优化（带约束：单只股票最多20%，最少1%）
        self.optimize_portfolio(optimization_type='max_sharpe',
                                constraints={'max_weight': 0.2, 'min_weight':0.01})

        # 3. 第二步：有效前沿计算
        print("\n📋 第2步: 有效前沿计算")
        # 3.1 计算有效前沿（默认20个点）
        self.calculate_efficient_frontier()

        # 4. 第三步：生成完整报告
        print("\n📋 第3步: 生成完整报告")
        # 4.1 调用之前定义的generate_comprehensive_report方法生成报告
        report = self.generate_comprehensive_report(weights, benchmark_weights)

        # 5. 第四步：获取回测数据
        print("\n📋 第4步: 获取回测数据")
        # 5.1 执行回测分析（默认初始资本10000，季度再平衡）
        backtest_df, backtest_metrics = self.backtest_portfolio(weights)

        # 6. 第五步：生成分析图表
        print("\n📋 第5步: 生成分析图表")
        # 6.1 调用之前定义的plot_comprehensive_analysis方法生成5张图表
        self.plot_comprehensive_analysis(weights, backtest_df)

        # 7. 打印分析总结
        print(f"\n{'=' * 80}")
        print("🎉 完整投资组合分析完成！")
        print("=" * 80)
        # 8. 返回所有分析结果
        return report

def load_real_stock_data(stock_list, start_date='2019-01-01', end_date='2025-12-02'):
    """
    从Excel文件加载真实的股票数据
    参数:
        stock_list -- 股票代码列表
        start_date -- 开始日期（默认为2019年，使用最近数据）
        end_date -- 结束日期（默认为2025年12月2日）
    功能说明:
        - 逐个加载每个股票的Excel数据文件
        - 自动识别日期列和价格列
        - 计算日收益率
        - 对齐所有股票的数据日期
        - 返回清理后的收益率DataFrame
    数据处理流程:
        1. 读取Excel文件
        2. 识别日期列（支持多种格式）
        3. 设置日期索引
        4. 识别价格列（支持多种格式）
        5. 计算收益率
        6. 数据质量检查
    为什么选择2019-2025年数据:
        1. 最近数据更能反映当前市场特征
        2. 避免过时的市场结构影响分析
        3. 足够的数据量进行可靠分析（6-7年）
        """
    print("📊 从Excel文件加载真实股票数据...")
    print(f"数据时间范围: {start_date} 到 {end_date}")

    all_returns = {}
    all_prices = {}
    loaded_stocks = []
    for stock in stock_list:
        try:
            file_path = f"./{stock}_stock_data.xlsx"
            if not os.path.exists(file_path):
                print(f"⚠️  文件不存在: {file_path}")
                continue
            df = pd.read_excel(file_path)
            print(f"\n📈 处理 {stock} 数据:")
            print(f"  文件列名: {list(df.columns)}")
            print(f"  数据行数: {len(df)}")

            # 检查数据时间范围
            # 尝试不同的日期列名
            date_columns = ['date', 'Date', 'DATE', 'datetime', 'Datetime', '日期', '时间']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    print(f"  找到日期列: {date_col}")
                    break

            if date_col is None:
                # 尝试第一列是否是日期类型
                first_col = df.columns[0]
                print(f"  尝试第一列作为日期: {first_col}")

                # 尝试多种日期格式
                try:
                    df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
                    if df[first_col].isnull().all():
                        raise ValueError("无法转换为日期")
                    date_col = first_col
                    print(f"  成功转换第一列为日期")
                except:
                    # 尝试其他可能的日期列
                    for col in df.columns:
                        if 'date' in str(col).lower() or 'time' in str(col).lower():
                            try:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                date_col = col
                                print(f"  找到日期列: {col}")
                                break
                            except:
                                continue
            if date_col is None:
                raise ValueError(f"未找到日期列，可用列: {list(df.columns)}")
            # 设置日期索引
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)

            df = df.sort_index()
            # 显示数据时间范围
            if len(df) > 0:
                print(f"  原始数据时间范围: {df.index[0].date()} 到 {df.index[-1].date()}")
            # 尝试不同的价格列名
            price_columns = ['close', 'Close', 'Adj Close', 'Price', 'price',
                             'Close_Price', 'close_price', 'Adj Close_Price',
                             '收盘价', '收盘', 'ClosePrice']
            price_col = None
            for col in price_columns:
                if col in df.columns:
                    price_col = col
                    print(f"  找到价格列: {price_col}")
                    break

            if price_col is None:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'price' in col_lower or 'close' in col_lower or 'adj' in col_lower:
                        price_col = col
                        print(f"  使用可能的价格列: {price_col}")
                        break
            if price_col is None and len(df.columns) >=2:
                # 如果还是没找到，尝试数值列
                for col in df.columns:
                    if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                        price_col = col
                        print(f"  使用数值列作为价格: {price_col}")
                        break
            if price_col is None:
                raise ValueError(f"未找到价格列，可用列: {list(df.columns)}")

            # 获取价格序列
            prices = df[price_col]

            # 检查价格数据
            print(f"  价格数据统计:")
            print(f"    非空值数量: {prices.count()}")
            print(f"    缺失值数量: {prices.isnull().sum()}")
            print(f"    价格范围: {prices.min():.2f} - {prices.max():.2f}")

            prices = prices.dropna()
            if len(prices) == 0:
                print(f"  ⚠️  {stock}: 价格数据为空，跳过")
                continue

            # 显示数据截止日期
            latest_date = prices.index[-1]
            print(f"  最新数据日期: {latest_date.date()}")

            # 确保有足够的历史数据
            min_required_date = pd.Timestamp('2019-01-01')
            if prices.index[0] > min_required_date:
                print(f"  ⚠️  {stock}: 历史数据不足，最早数据从 {prices.index[0].date()} 开始")

            # 使用完整数据范围（不进行额外过滤）
            print(f"  使用数据范围: {prices.index[0].date()} 到 {prices.index[-1].date()}")

            # 计算日收益率
            returns = prices.pct_change().dropna()

            # 检查收益率数据有效性
            print(f"  收益率数据统计:")
            print(f"    有效收益率数量: {len(returns)}")
            print(f"    平均日收益率: {returns.mean():.4%}")
            print(f"    日收益率波动率: {returns.std():.4%}")

            if len(returns) < 100:
                print(f"  ⚠️  {stock}: 有效收益率数据不足 ({len(returns)}天)，跳过")
                continue

            all_returns[stock] = returns
            all_prices[stock] = prices
            loaded_stocks.append(stock)

            print(f"  ✅ {stock}: 成功加载{len(prices)}天价格数据，{len(returns)}天收益率数据")
            print(f"     时间范围: {prices.index[0].date()} 到 {prices.index[-1].date()}")
        except Exception as e:
            print(f"  ❌ {stock}: 加载失败 - {str(e)[:100]}...")
            continue

    if not all_returns:
        raise ValueError("没有成功加载任何股票数据")

    # 创建收益率DataFrame
    returns_df = pd.DataFrame(all_returns)
    print(f"\n✅ 成功加载 {len(all_returns)} 只股票数据: {', '.join(loaded_stocks)}")
    print(f"   最终数据时间范围: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
    print(f"   交易日数: {len(returns_df)}")

    # 显示各股票数据量
    print(f"\n📊 各股票数据量统计:")
    for stock in loaded_stocks:
        if stock in returns_df.columns:
            data_count = returns_df[stock].count()
            if data_count > 0:
                date_range = f"{returns_df[stock].dropna().index[0].date()} 到{returns_df[stock].dropna().index[-1].date()}"
                print(f"  {stock}: {data_count}个交易日，{date_range}")
    return returns_df, all_prices, loaded_stocks

# ==================== 主函数 ====================
def main():
    """主函数：演示完整分析系统"""
    print("📊 基于您的真实持仓进行投资组合分析")
    print("=" * 60)

    # 1. 定义您的股票持仓（根据您提供的信息）
    stock_holdings = {
        'SCHD': 156,  # Schwab美国股息ETF
        'KO': 155,  # 可口可乐
        'VOO': 155,  # Vanguard标普500 ETF
        'GLD': 107,  # 黄金ETF
        'LLY': 103,  # 礼来公司
        'AAPL': 64,  # 苹果公司
        'TSLA': 49,  # 特斯拉
        'AA': 49,  # 美国铝业
        'AMZN': 48,  # 亚马逊
        'UPST': 43,  # Upstart Holdings
        'UNH': 42,  # 联合健康
        'GOOGL': 41,  # 谷歌
        'SBUX': 39,  # 星巴克
        'OMI': 32,  # Owens & Minor
        'RKLB': 22,  # Rocket Lab
        'ASTS': 22  # AST SpaceMobile
    }

    # 2. 计算总投资金额和权重
    total_value = sum(stock_holdings.values())
    portfolio_weights = {}
    print("\n📊 您的持仓详情:")
    print("-" * 50)
    print(f"{'股票':<8} {'持仓金额($)':<12} {'权重':<10}")
    print("-" * 50)

    for stock, value in stock_holdings.items():
        weight = value / total_value
        portfolio_weights[stock] = weight
        print(f"{stock:<8} ${value:<11} {weight:.2%}")
    print("-" * 50)
    print(f"{'总计':<8} ${total_value:<11} {sum(portfolio_weights.values()):.2%}")

    # 3. 加载真实股票数据
    print(f"\n🔄 正在加载股票数据...")

    try:
        # 调用您的数据加载函数
        stock_list = list(stock_holdings.keys())
        returns_df, all_prices, loaded_stocks = load_real_stock_data(
            stock_list=stock_list,
            start_date = '2020-01-01',
            end_date = '2025-12-12'
        )
        # 4. 检查哪些股票成功加载
        print(f"\n✅ 数据加载完成:")
        print(f"   成功加载股票: {len(loaded_stocks)}/{len(stock_list)}只")

        # 5. 检查是否有股票数据缺失
        missing_stocks = set(stock_list) - set(loaded_stocks)
        if missing_stocks:
            print(f"   ⚠️ 以下股票数据缺失: {', '.join(missing_stocks)}")
            print("   注意: 缺失股票将从分析中排除")
            # 更新权重，排除缺失的股票
            remaining_value = sum([stock_holdings[s] for s in loaded_stocks])
            for stock in missing_stocks:
                if stock in portfolio_weights:
                    del portfolio_weights[stock]

            # 重新计算权重
            for stock in loaded_stocks:
                portfolio_weights[stock] = stock_holdings[stock] / remaining_value

        print(f"\n📊 将分析的投资组合:")
        for stock in loaded_stocks:
            weight = portfolio_weights.get(stock, 0)
            value = stock_holdings[stock]
            print(f"   {stock}: ${value} ({weight:.2%})")

        # 6. 显示数据基本信息
        print(f"\n📈 数据基本信息:")
        print(f"   数据期间: {returns_df.index[0].date()} 到 {returns_df.index[-1].date()}")
        print(f"   有效交易日数: {len(returns_df)}")

        # 7. 创建分析器对象
        print(f"\n🔄 创建投资组合分析器...")
        analyzer = CompletePortfolioAnalyzer(returns_df, risk_free_rate=0.03)

        # 8. 运行完整分析
        print(f"\n{'=' * 80}")
        print("🚀 开始基于您持仓的完整投资组合分析")
        print("=" * 80)
        report = analyzer.run_complete_analysis(portfolio_weights)

        # 9. 打印分析总结
        print(f"\n📊 您的投资组合分析总结:")
        print("-" * 60)
        print(f"   总投资金额: ${total_value:,.2f}")
        print(f"   分析股票数量: {len(loaded_stocks)}只")
        print(f"   投资组合夏普比率: {report['portfolio_metrics']['夏普比率']:.3f}")
        print(f"   投资组合年化收益率: {report['portfolio_metrics']['年化收益率']:.2%}")
        print(f"   投资组合年化波动率: {report['portfolio_metrics']['年化波动率']:.2%}")
        print(f"   最大回撤: {report['backtest_metrics']['最大回撤']:.2%}")
        print(f"   最终资本模拟: ${report['backtest_metrics']['最终资本']:,.2f}")
        print(f"   再平衡次数: {report['backtest_metrics']['再平衡次数']}")
        print("-" * 60)

        # 10. 显示各股票表现
        print(f"\n🏆 各股票历史表现:")
        print("-" * 70)
        print(f"{'股票':<8} {'权重':<8} {'年化收益率':<12} {'年化波动率':<12} {'夏普比率':<10}")
        print("-" * 70)

        for stock in loaded_stocks:
            weight = portfolio_weights.get(stock, 0)
            return_val = analyzer.annual_returns[stock]
            vol_val = analyzer.annual_volatility[stock]
            sharpe_val = analyzer.sharpe_ratios[stock]
            print(f"{stock:<8} {weight:<8.2%} {return_val:<12.2%} {vol_val:<12.2%} {sharpe_val:<10.3f}")

        # 11. 提供针对性的投资建议
        print(f"\n💡 针对您持仓的投资建议:")

        # 获取风险分析结果
        risk_df = report['risk_analysis']

        # 识别高风险股票（风险倍数 > 1.5）
        high_risk = risk_df[risk_df['风险倍数']> 1.5]
        if len(high_risk) > 0:
            print(f"\n⚠️  高风险警报 - 建议考虑减仓:")
            for _, row in high_risk.iterrows():
                stock = row['股票']
                current_weight = row['权重']
                risk_multiplier = row['风险倍数']
                current_value = stock_holdings.get(stock, 0)

                # 建议减少到风险倍数=1
                suggested_weight = current_weight / risk_multiplier
                suggested_value = suggested_weight * total_value
                reduction = current_value - suggested_value
                print(f"   {stock}: 当前${current_value} ({current_weight:.1%}) → "
                      f"建议${suggested_value:,.0f} ({suggested_weight:.1%}) "
                      f"减少${reduction:,.0f}")

        # 识别低风险股票（风险倍数 < 0.7）
        low_risk = risk_df[risk_df['风险倍数'] < 0.7]
        if len(low_risk) > 0:
            print(f"\n✅ 低风险机会 - 可以考虑加仓:")
            for _, row in low_risk.iterrows():
                stock = row['股票']
                current_weight = row['权重']
                risk_multiplier = row['风险倍数']
                current_value = stock_holdings.get(stock, 0)
                # 建议增加50%
                suggested_weight = current_weight * 1.5
                suggested_value = suggested_weight * total_value
                increase = suggested_value - current_value
                print(f"   {stock}: 当前${current_value} ({current_weight:.1%}) → "
                      f"建议${suggested_value:,.0f} ({suggested_weight:.1%}) "
                      f"增加${increase:,.0f}")

        # 12. 返回分析结果
        print(f"\n✅ 分析完成！")
        print("💡 请查看生成的5张图表获取可视化分析")
        return report
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {str(e)}")
        print("请检查:")
        print("1. Excel文件是否存在且格式正确")
        print("2. 股票代码是否正确（注意大小写）")
        print("3. 数据文件命名格式: {股票代码}_stock_data.xlsx")
        print("4. Excel文件中包含正确的日期和价格数据")
        import traceback
        traceback.print_exc()
        return None

# ==================== 程序入口点 ====================
if __name__ == "__main__":
    # 运行分析
    report = main()

    if report:
        print("\n🙏 感谢使用投资组合分析系统！")
        print("📊 提示: 请查看生成的5张图表获取可视化分析")
    else:
        print("\n❌ 分析失败，请检查以上错误信息")










