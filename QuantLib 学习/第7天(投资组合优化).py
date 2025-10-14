'''
Day 7：投资组合优化
目标：
-结合QuantLib和优化工具（cvxpy/ PyPortfolioOpt）。
-构建最优投资组合。
任务：
-导入历史收益率数据。
-计算期望收益、协方差矩阵。
-求解最优权重组合（最小方差或最大夏普比率）。
输出：组合优化脚本。
'''

# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
import QuantLib as ql
import os
from datetime import datetime, timedelta



# 设置中文字体,
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SmartPortfolioOptimizer:
    '''
    智能投资组合优化器类
    主要: 自动选股票, 计算收益率, 优化权重, 风险分析, 可视化结果
    '''
    def __init__(self, max_stocks=15):
        '''
        初始化优化器
        参数: max_stocks: 最大选择股票的数量, 避免过多股票导致优化复杂
        '''
        self.risk_free_rate = 0.02  # 无风险利率, 默认2%
        self.max_stocks = max_stocks

    def load_all_stock_data(self):
        '''
        加载所有可用的股票数据
        功能: 扫描当前目录, 读取所有股票数据Excel文件
        返回: 包含所有股票数据的字典
        '''
        print(f"...扫描并加载所有股票数据...")
        # 使用列表推到式找到所有的以'_stock.xlsx' 结尾的文件
        stock_files = [f for f in os.listdir('.') if f.endswith('_stock.xlsx')]

        # 检查是否找到股票文件
        if not stock_files:
            print(f" 未找到任何股票数据文件 (*_stock.xlsx)")
            return None

        stock_data = {} # 用于存储所有股票数据

        # 遍历每个股票文件并加载数据
        for file in sorted(stock_files):    # sorted 确保按字母顺序处理
            try:
                # 从文件名提取股票代码: AAPL_stock.xlsx -->AAPL
                stock_code = file.replace('_stock.xlsx', '')
                # 读取excel文件, index_col=0表示第一列作为索引 (通常是日期)
                df = pd.read_excel(file, index_col=0)
                # 验证数据格式, 必须有Close列且数据不为空
                if 'Close' in df.columns and not df.empty:
                    stock_data[stock_code] = df # 存储到字典中
                    print(f"加载{stock_code}: {len(df)}个交易日数据")
            except Exception as e:
                print(f"加载{file} 失败:{e}")
        print(f"\n 成功加载 {len(stock_data)}只股票")
        return stock_data

    def filter_stocks_by_performance(self, stock_data, min_trading_days = 1000):
        '''
        根据股票表现筛选优质股票
        功能: 计算每只股票的夏普比率, 选择表现最好的前N只
        参数: stock_data: 所有股票数据
            min_trading_days: 最小交易天数要求, 确保数据充足
        返回: 筛选后的股票数据
        '''
        print(f"\n 筛选股票 (最少{min_trading_days}个交易日)")

        filtered_stocks = {}    # 存储筛选后的股票数据
        performance_stats = {}  # 存储每只股票的性能指标
        # 遍历每只股票, 计算关键性能指标
        for ticker, df in stock_data.items():
            # 首先检查交易天数是否满足要求
            if len(df) < min_trading_days:
                continue    # 跳过不满足天数要求的数据
            try:
                # 计算日收益率: 今日收盘价/昨日收盘价 - 1
                returns = df['Close'].pct_change().dropna()
                # 再次检查收益率数据的长度
                if len(returns) < min_trading_days:
                    continue

                # 计算关键金融指标
                daily_return = returns.mean()           # 日均收益率
                daily_vol = returns.std()               # 日波动率 (标准差)
                annual_return = daily_return * 252      # 年化收益率 (252个交易日)
                annual_vol = daily_vol * np.sqrt(252)   # 年化波动率

                # 计算夏普比率: (年化收益 - 无风险利率) / 年化波动率
                # 夏普比率衡量每单位风险获得的超额收益
                sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol if (
                        annual_vol > 0) else -10
                # 存储性能指标
                performance_stats[ticker] = {
                    'annual_return': annual_return,
                    'annual_vol': annual_vol,
                    'sharpe_ratio': sharpe_ratio,
                    'trading_days': len(returns)
                }
                filtered_stocks[ticker] = df # 存储通过初步筛选的股票
            except Exception as e:
                print(f" 分析{ticker}失败: {e}")

        # 按夏普比率从高到底排序, 选择表现最好的股票
        # sorted函数: 对performance_stats字典按夏普比率降序排列
        sorted_stocks = sorted(performance_stats.items(),
                               key=lambda x: x[1]['sharpe_ratio'],
                               reverse=True)
        selected_stocks = {}    # 最终选择的股票
        selected_count = min(self.max_stocks, len(sorted_stocks))   # 实际选择数量
        print(f" \n 选择前{selected_count}只表现最好的股票:")

        # 遍历排序后的股票, 选择前selected_count只
        for i, (ticker, stats) in enumerate(sorted_stocks[:selected_count]):
            selected_stocks[ticker] = stock_data[ticker]    # 从原始数据获取完整数据
            # 格式化输出股票信息
            print(f"{i+1:2d}. {ticker}: 夏普{stats['sharpe_ratio']:+.2f},"
                  f"年化{stats['annual_return']:.2%}, 波动{stats['annual_vol']:.2%}")
        return selected_stocks

    def calculate_returns(self, stock_data):
        '''
        计算所有股票的收益率数据
        功能: 将价格数据转为收益率数据, 为优化做准备
        参数: 筛选后的股票数据
        返回: 包含所有股票收益率的DataFrame
        '''
        print(f"\n 计算股票收益率........")
        returns_data = {}   # 存储每只股票的收益率序列

        for ticker, df in stock_data.items():
            try:
                # 计算日收益率: (今日收盘价 - 昨日收盘价) / 昨日收盘价
                returns = df['Close'].pct_change().dropna()
                returns_data[ticker] = returns    # 存储到字典
                # 计算显示关键指标
                daily_return = returns.mean()
                daily_vol = returns.std()
                annual_return = daily_return * 252
                annual_vol = daily_vol * np.sqrt(252)
                sharpe_ratio = (annual_return - self.risk_free_rate) / annual_vol

                # 输出每只股票的计算数据
                print(f" {ticker}: 夏普{sharpe_ratio:+.2%}, 年化{annual_return:+.2%}, 波动{annual_vol:.2%}")
            except Exception as e:
                print(f" 计算{ticker} 收益率失败: {e}")

        # 将收益率字典转为DataFrame, 并删除包含NaN的行
        returns_df = pd.DataFrame(returns_data).dropna()
        print(f" \n 最终收益率数据框形状: {returns_df.shape}")
        return returns_df

    def portfolio_optimization(self, returns_df, method='sharpe'):
        '''
        投资组合优化核心函数
        功能: 使用数学优化方法找到最优的权重分配
        参数: returns_df: 收益率数据
            method: 优化方法 'sharpe' = 最大夏普比率, 'min_variance'=最小方差
        返回: 最优权重字典和组合表现
        '''
        print(f"\n 进行投资组合优化 - {method}...")
        # 计算年化统计量
        expected_returns = returns_df.mean() * 252  # 年化期望收益
        cov_matrix = returns_df.cov() * 252         # 年化协方差矩阵

        n_assets = len(expected_returns)    # 资产数量
        print(f" 优化资产数量: {n_assets}")

        # 设置优化约束条件: 权重之和必须等于1 (100%)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # 设置边界条件: 每个权重在 0 到 1 之间 ( 不允许卖空)
        bounds = tuple((0,1) for _ in range(n_assets))

        # 初始猜测: 等权重分配
        initial_weights = n_assets * [1.0 / n_assets]
        # 根据优化方法选择目标函数
        if method == 'sharpe':
            # 最大化夏普比率 = 最小化负夏普比率
            objective = lambda w: -self._calculate_sharpe(w, expected_returns, cov_matrix)
        else:
            # 最小化波动率 ( 方差)
            objective = lambda w: self._calculate_sharpe(w, cov_matrix)

        # 使用SLSQP算法进行序列最小二乘规划优化
        result = sco.minimize(objective, initial_weights,
                              method='SLSQP', bounds=bounds, constraints=constraints)

        # 检查优化是否成功
        if result.success:
            optimal_weights = result.x  # 最优权重向量
            portfolio_return = np.sum(optimal_weights * expected_returns)   # 组合期望收益
            portfolio_vol = self._calculate_volatility(optimal_weights, cov_matrix) # 组合波动率
            sharpe_ratio = self._calculate_sharpe(optimal_weights, expected_returns, cov_matrix)    # 夏普比率
            # 将权重向量转为字典格式 ( 股票代码: 权重)
            weights_dict = dict(zip(returns_df.columns, optimal_weights))
            print(f" 优化成功")
            return weights_dict, (portfolio_return, portfolio_vol, sharpe_ratio)
        else:
            print(f" 优化失败: {result.message}")
            # 优化失败时使用等权重作为备选方案
            return self._equal_weight_fallback(returns_df, expected_returns, cov_matrix)

    def _calculate_sharpe(self, weights, expected_returns, cov_matrix):
        '''
        计算夏普比率的辅助函数
        公式: 夏普比率 = (组合收益 - 无风险利率) / 组合波动率
        '''
        port_return = np.sum(weights * expected_returns)    # 组合期望收益
        port_vol = self._calculate_volatility(weights, cov_matrix)  # 组合波动率
        if port_vol == 0:
            return 0        # 避免除零错误
        return (port_return - self.risk_free_rate) / port_vol

    def _calculate_volatility(self, weights, cov_matrix):
        '''
        计算组合波动率的辅助函数
        公式: 波动率 = sqrt(权重^T * 协方差矩阵 * 权重)
        '''
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def _equal_weight_fallback(self, returns_df, expected_returns, cov_matrix):
        '''
        等权重备选方案
        当优化失败时使用简单的等权重分配
        '''
        print(f" 使用等权重组合")
        n_assets = len(returns_df.columns)
        equal_weights = np.array([1/n_assets] * n_assets)   # 等权重向量

        #计算等权重组合的表现
        port_return = np.sum(equal_weights * expected_returns)
        port_vol = self._calculate_volatility(equal_weights, cov_matrix)
        sharpe_ratio = self._calculate_sharpe(equal_weights, expected_returns, cov_matrix)

        weights_dict = dict(zip(returns_df.columns, equal_weights))
        return weights_dict, (port_return, port_vol, sharpe_ratio)

    def filter_significant_weights(self, weights, min_weight=0.01):
        '''
        过滤权重太小的股票
        功能: 简化优化显示, 只保留重要的权重分配
        参数: weights: 原始权重字典
            min_weight: 最小权重值 (1%)
        返回: 过滤并重新归一化后的权重
        '''
        # 只保留权重大于等于只的股票
        significant_weights = {k: v for k, v in weights.items() if v >= min_weight}

        if significant_weights:
            # 重新归一化权重, 使其总和为1
            total = sum(significant_weights.values())
            normalized_weights = {k: v/total for k, v in significant_weights.items()}
            print(f"\n筛选后保留{len(normalized_weights)} 只重要股票(权重>={min_weight:.1%})")
            return normalized_weights
        else:
            print(f"所有股票权重都太小, 返回原始权重")
            return weights

    def efficient_frontier_analysis(self, returns_df):
        """
        有效前沿分析
        功能：通过蒙特卡洛模拟生成有效前沿，展示风险收益权衡
        有效前沿：在给定风险水平下能获得的最大收益边界
        """
        print("\n 生成有效前沿...")

        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        n_assets = len(expected_returns)

        n_portfolios = 5000 # 模拟的投资组合数量
        results = np.zeros((3, n_portfolios))   # 存储结果: 收益, 风险, 夏普比率

        # 蒙特卡洛模拟：随机生成权重，计算组合表现
        for i in range(n_portfolios):
            # 生成随机权重
            weights = np.random.random(n_assets)
            weights /= weights.sum()    # 归一化权重
            # 计算组合表现
            port_return = np.sum(weights * expected_returns)
            port_vol = self._calculate_volatility(weights, cov_matrix)
            sharpe_ratio = self._calculate_sharpe(weights, expected_returns, cov_matrix)
            # 存储结果
            results[0, i] = port_return # 收益
            results[1, i] = port_vol    # 风险
            results[2, i] = sharpe_ratio    # 夏普比率
        return results

    def plot_optimization_results(self, returns_df, weights, performance, efficient_frontier=None):
        """
        绘制优化结果图表
        功能：通过4个子图全面展示优化结果
        参数：
            returns_df: 收益率数据
            weights: 最优权重
            performance: 组合表现（收益、风险、夏普）
            efficient_frontier: 有效前沿数据
        """
        expected_return, volatility, sharpe_ratio = performance

        # 只显示权重 > 0 的股票, 简化图表
        non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
        # 创建 2 X 2 的子图布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('投资组合优化分析结果', fontsize=12, fontweight='bold')

        # 子图1, 权重分配柱状图
        if non_zero_weights:
            stocks = list(non_zero_weights.keys())
            weight_values = [w * 100 for w in non_zero_weights.values()]    # 转换为百分比

            # 使用Set3 色彩, 为每只股票分配不同颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(stocks)))
            bars = ax1.bar(stocks, weight_values, color=colors, alpha=0.8)
            ax1.set_title('投资组合权重分配 (权重>0)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('权重 (%)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # 在柱状图上添加数值标签
            for bar, value, in zip(bars, weight_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5,0.5, '无显权重分配', ha='center', va='venter', transform=ax1.transAxes)
            ax1.set_title('投资组合权重分配', fontsize=12, fontweight='bold')

        # 子图2: 有效前沿
        if efficient_frontier is not None:
            scatter = ax2.scatter(efficient_frontier[1]*100, efficient_frontier[0]*100,
                                  c=efficient_frontier[2], cmap='viridis', alpha=0.6, s=1)
            # 标记最优组合位置
            ax2.scatter(volatility*100, expected_return*100, color='red', s=200,
                        marker='*', edgecolors='black', label='最优组合')
            ax2.set_xlabel('波动率 (%)')
            ax2.set_ylabel('期望收益 (%)')
            ax2.set_title('有效前沿', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='夏普比率')

        # 子图3: 累积收益率对比图
        # 只显示权重较大的股票, 避免图表过于拥挤
        significant_stocks = [k for k, v in weights.items() if v >= 0.01]
        if not significant_stocks:
            significant_stocks = list(weights.keys())[:6]   # 如果没用, 显示前6只
        # 绘制每只重要股票的累积收益率曲线
        for ticker in significant_stocks:
            cumulative = (1 + returns_df[ticker]).cumprod() # 计算累积收益
            ax3.plot(cumulative.index, cumulative, label=ticker, alpha=0.8, linewidth=2)
        # 计算并绘制投资组合的累积收益率
        portfolio_returns = (returns_df * list(weights.values())).sum(axis=1)
        portfolio_cumulative = ( 1 + portfolio_returns).cumprod()
        ax3.plot(portfolio_cumulative.index, portfolio_cumulative,
                 label='投资组合', linewidth=3, color='black', linestyle='--')
        ax3.set_title('主要股票累积收益率对比', fontsize=12, fontweight='bold')
        ax3.set_ylabel('累积收益')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 组合表现指标柱状图
        metrics = ['期望收益率', '波动率', '夏普比率']
        values = [expected_return*100, volatility*100, sharpe_ratio]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
        ax4.set_title('组合表现指标', fontsize=12, fontweight='bold')
        ax4.set_ylabel('百分比/比率')
        ax4.grid(True, alpha=0.3)

        # 在柱状图上添加数值标签
        for bar, value, metric in zip(bars, values, metrics):
            unit = '%' if metric != '夏普比率' else ''  # 夏普比率没用单位
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() +0.1,
                     f"{value:.2f}{unit}", ha='center', va='bottom', fontweight='bold')

        # 调整子图距离
        plt.tight_layout()
        plt.show()

    def risk_analysis(self, returns_df, weights):
        print(f"\n 风险分析")
        # 计算投资组合的日收益率
        portfolio_returns = (returns_df * list(weights.values())).sum(axis=1)

        # VaR计算 (Value at Risk)
        # 95% VaR: 有95% 的把握损失不会超过这个值
        var_95 = -np.percentile(portfolio_returns, 5) * 100
        # 99% VaR: 有99% 的把握损失不会超过这个值
        var_99 = -np.percentile(portfolio_returns, 1) * 100

        # 最大回撤计算
        cumulative = (1+ portfolio_returns).cumprod()   # 累积收益
        running_max = cumulative.expanding().max()      # 运行最大值
        drawdown = (cumulative - running_max) / running_max # 回撤比列
        max_drawdown = drawdown.min() * 100             # 最大回撤

        print(f"日VaR (95%): {var_95:.2f}%")
        print(f"日VaR (99%): {var_99:.2f}%")
        print(f"最大回撤: {max_drawdown:.2f}%")

    def run_complete_analysis(self):
        """
        运行完整的投资组合分析流程
        这是主要的执行函数，按步骤调用各个功能模块
        """
        print('=' * 70)
        print('智能投资组合优化分析')
        print('=' * 70)

        #1. 加载所有数据
        stock_data = self.load_all_stock_data()
        if not stock_data:
            return

        #2. 筛选优质股票
        selected_stocks = self.filter_stocks_by_performance(stock_data)
        if not selected_stocks:
            print(f"没用符合条件的股票")
            return

        # 3. 计算收益率
        returns_df = self.calculate_returns(selected_stocks)
        if returns_df.empty:
            return

        #4. 执行投资组合优化
        weights, performance = self.portfolio_optimization(returns_df, 'sharpe')
        if weights and performance:
            expected_return, volatility, sharpe_ratio=performance
            print(f"\n 优化结果: ")
            print(f"预期年化收益率: {expected_return:+.2%}")
            print(f"预期年化波动率: {volatility:.2%}")
            print(f"夏普比率: {sharpe_ratio:.2f}")

            # 过滤小权重图片
            significant_weights = self.filter_significant_weights(weights)
            print(f"\n 重要权重分配")
            for stock, weight in significant_weights.items():
                print(f"{stock}: {weight:.2%}")

            # 5. 风险分析
            self.risk_analysis(returns_df, weights)
            #6.
            efficient_frontier = self.efficient_frontier_analysis(returns_df)
            # 7. 可视化
            self.plot_optimization_results(returns_df, weights, performance, efficient_frontier)
        print(f"\n 分析完成")

# 程序入口点
if __name__ == '__main__':
    optimizer = SmartPortfolioOptimizer(max_stocks=15)
    optimizer.run_complete_analysis()


'''
投资组合优化项目总结
🎯 项目目标完成情况
成功构建了一个完整的智能投资组合优化系统，实现了从数据加载、股票筛选、组合优化到风险分析和可视化的全流程自动化。

📊 核心成果
1. 数据管理
✅ 自动扫描加载多股票历史数据
✅ 数据质量验证和完整性检查
✅ 支持15只股票同时分析

2. 智能筛选
✅ 基于夏普比率的股票排名
✅ 年化收益率和波动率计算
✅ 自动选择表现最佳的股票组合

3. 组合优化
✅ 最大夏普比率优化
✅ 最小方差组合优化
✅ 权重约束处理（和为1，禁止卖空）

4. 风险分析
✅ VaR风险价值计算（95%/99%）
✅ 最大回撤分析
✅ 波动率估计

5. 可视化展示
✅ 四合一专业图表
✅ 权重分配柱状图
✅ 有效前沿散点图
✅ 累积收益曲线对比
✅ 性能指标展示

🔧 技术实现亮点
算法应用
SLSQP优化算法 - 处理复杂约束条件
蒙特卡洛模拟 - 生成有效前沿
协方差矩阵 - 资产相关性建模

代码质量
模块化设计 - 功能独立，易于维护
异常处理 - 完善的错误处理机制
参数可配置 - 灵活调整优化参数
'''