'''
第2天：投资组合优化系统
使用PyPortfolioOpt实现投资组合优化，最大化夏普比率
功能：资产配置优化、绩效分析、可视化展示
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持，确保图表能正确显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class PortfolioOptimizer:
    """
    投资组合优化器类
    主要功能：数据加载、收益率计算、组合优化、绩效分析、可视化
    """

    def __init__(self, risk_free_rate=0.02):
        """
        初始化优化器

        Parameters:
        risk_free_rate: 无风险利率，默认2%（年化）
        """
        self.risk_free_rate = risk_free_rate
        self.weights = None          # 存储优化后的资产权重
        self.performance = None      # 存储组合绩效指标
        self.data = None             # 存储股票价格数据
        self.returns = None          # 存储股票收益率数据

    def load_stock_data_from_current_dir(self):
        """
        从当前目录加载股票数据
        要求：数据文件格式为 {股票代码}_stock_data.xlsx，包含Close列

        Returns:
        bool: 数据加载是否成功
        """
        print("正在从当前目录加载股票数据...")
        all_data = {}  # 存储所有股票数据
        valid_tickers = []  # 存储有效的股票代码

        # 查找所有符合命名规则的股票数据文件
        stock_files = glob.glob('./*_stock_data.xlsx')

        if not stock_files:
            print("错误: 当前目录下未找到股票数据文件")
            print("请确保文件命名格式为: ./AAPL_stock_data.xlsx")
            return False

        print(f"找到 {len(stock_files)} 个股票数据文件")

        # 逐个加载股票数据文件
        for file_path in stock_files:
            filename = os.path.basename(file_path)
            ticker = filename.replace('_stock_data.xlsx', '')

            try:
                # 读取Excel文件
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)

                # 数据验证：必须有Close列且数据量足够
                if 'Close' in df.columns and len(df) > 500:
                    df = df.sort_index()  # 按日期排序

                    # 计算数据时间范围
                    date_range = df.index[-1] - df.index[0]
                    years = date_range.days / 365.25

                    # 要求至少2年历史数据
                    if years >= 2:
                        all_data[ticker] = df['Close']
                        valid_tickers.append(ticker)
                        print(f"✓ 加载 {ticker} 数据成功 ({len(df)} 天, {years:.1f} 年)")
                    else:
                        print(f"✗ {ticker}: 数据时间范围不足 ({years:.1f} 年)")
                else:
                    print(f"✗ {ticker}: 数据无效或数据点不足 ({len(df)} 天)")

            except Exception as e:
                print(f"✗ 加载 {ticker} 失败: {e}")

        # 检查是否有足够的股票进行组合优化
        if len(valid_tickers) < 2:
            print(f"错误: 需要至少2只股票进行组合优化，当前只有 {len(valid_tickers)} 只")
            return False

        # 合并所有股票数据
        self.data = pd.DataFrame(all_data)
        self.data = self.data.sort_index()  # 确保数据按日期排序

        # 数据清洗：前向填充缺失值，删除仍有缺失的行
        self.data = self.data.ffill().dropna()

        # 检查合并后的数据量是否足够
        if len(self.data) < 500:
            print(f"错误: 合并后数据量不足，至少需要500个交易日，当前只有 {len(self.data)} 天")
            return False

        # 计算日收益率
        self.returns = self.data.pct_change().dropna()

        # 计算并显示数据统计信息
        total_days = len(self.data)
        date_range = self.data.index[-1] - self.data.index[0]
        years = date_range.days / 365.25

        print(f"\n✅ 数据加载完成!")
        print(f"   有效股票数量: {len(self.data.columns)}")
        print(f"   交易日数: {len(self.data)}")
        print(f"   时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 到 {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   数据覆盖: {years:.1f} 年")

        return True

    def calculate_annual_returns(self):
        """
        计算各年度收益率分析
        显示每只股票每年的收益率、总收益率和年化收益率
        """
        if self.data is None:
            print("请先加载数据!")
            return

        print(f"\n📊 各年度收益率分析:")
        print("=" * 80)

        # 添加年份列用于分组
        data_with_year = self.data.copy()
        data_with_year['Year'] = data_with_year.index.year
        years = sorted(data_with_year['Year'].unique())

        annual_returns_df = pd.DataFrame()

        # 计算每年的收益率
        for year in years:
            year_data = data_with_year[data_with_year['Year'] == year]
            if len(year_data) > 50:  # 至少50个交易日才算完整的一年
                start_prices = year_data.iloc[0].drop('Year')
                end_prices = year_data.iloc[-1].drop('Year')
                year_returns = (end_prices / start_prices - 1)
                annual_returns_df[year] = year_returns

        # 计算总收益率
        total_returns = (self.data.iloc[-1] / self.data.iloc[0] - 1)
        annual_returns_df['总收益率'] = total_returns

        # 计算年化收益率
        total_days = len(self.data)
        annualized_returns = (1 + total_returns) ** (252 / total_days) - 1
        annual_returns_df['年化收益率'] = annualized_returns

        # 显示结果
        print(annual_returns_df.round(4))

        return annual_returns_df

    def calculate_individual_performance(self):
        """
        计算各股票的单独表现指标
        包括：累计收益率、年化收益率、年化波动率、夏普比率
        """
        if self.data is None or self.returns is None:
            print("请先加载数据!")
            return

        # 计算累计收益率和年化收益率
        total_days = len(self.data)
        total_returns = (self.data.iloc[-1] / self.data.iloc[0] - 1)
        annual_returns = (1 + total_returns) ** (252 / total_days) - 1

        # 计算年化波动率
        annual_volatility = self.returns.std() * np.sqrt(252)

        # 创建绩效数据框
        performance_df = pd.DataFrame({
            '数据天数': [len(self.data[col].dropna()) for col in self.data.columns],
            '累计收益率': total_returns,
            '年化收益率': annual_returns,
            '年化波动率': annual_volatility,
            '夏普比率': (annual_returns - self.risk_free_rate) / annual_volatility
        })

        print(f"\n📈 各股票历史表现:")
        print("=" * 80)
        print(performance_df.round(4))

        return performance_df

    def optimize_portfolio(self, weight_bounds=(0.01, 0.4)):
        """
        执行投资组合优化 - 最大化夏普比率

        Parameters:
        weight_bounds: 单个资产权重限制，默认1%-40%

        Returns:
        tuple: (优化权重, 绩效指标)
        """
        if self.data is None or self.returns is None:
            print("请先加载数据!")
            return None, None

        print("\n" + "=" * 60)
        print("开始投资组合优化 - 最大化夏普比率")
        print("=" * 60)

        # 1. 计算预期收益率（使用对数收益率更稳定）
        print("\n1. 计算预期收益率...")
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        mu = log_returns.mean() * 252  # 年化对数收益率

        print("各资产预期年化收益率 (基于对数收益率):")
        for asset in mu.index:
            annual_ret = mu[asset]
            print(f"  {asset:<10}: {annual_ret:>8.2%}")

        # 2. 计算风险模型（协方差矩阵）
        print("\n2. 计算风险模型...")
        S = risk_models.sample_cov(self.data)
        print(f"协方差矩阵维度: {S.shape}")

        # 3. 创建有效前沿对象
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        # 4. 最大化夏普比率
        print("\n3. 执行最大化夏普比率优化...")
        try:
            ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        except Exception as e:
            print(f"优化失败: {e}")
            # 备选方案：最小方差组合
            print("尝试最小方差组合作为备选...")
            ef.min_volatility()

        # 5. 获取优化权重
        self.weights = ef.clean_weights()

        # 6. 计算组合绩效
        expected_return, volatility, sharpe_ratio = ef.portfolio_performance(
            risk_free_rate=self.risk_free_rate, verbose=False
        )

        # 存储绩效指标
        self.performance = {
            'annual_return': expected_return,
            'annual_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'expected_daily_return': expected_return / 252,
            'daily_volatility': volatility / np.sqrt(252)
        }

        return self.weights, self.performance

    def print_optimization_results(self):
        """打印优化结果，显示资产权重分配和绩效指标"""
        if self.weights is None or self.performance is None:
            print("请先执行优化!")
            return

        print('\n' + '=' * 60)
        print("🎯 投资组合优化结果")
        print('=' * 60)

        # 显示完整的资产权重分配
        print(f"\n📊 完整资产权重分配:")
        print('-' * 50)
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)

        total_weight = 0
        selected_count = 0

        # 打印每个资产的权重
        for asset, weight in sorted_weights:
            if weight > 0.001:  # 只显示权重大于0.1%的资产
                print(f"  {asset:<10}: {weight:>8.2%} ✓")
                selected_count += 1
                total_weight += weight
            else:
                print(f"  {asset:<10}: {weight:>8.2%} ✗")

        print(f"  {'总计':<10}: {total_weight:>8.2%}")
        print(f'\n  选中资产: {selected_count} 只')
        print(f"  未选资产: {len(self.weights) - selected_count} 只")

        # 计算集中度指标
        top3_weight = sum([w for _, w in sorted_weights[:3]])
        top5_weight = sum([w for _, w in sorted_weights[:5]])
        print(f"\n  前3大资产集中度: {top3_weight:.2%}")
        print(f"  前5大资产集中度: {top5_weight:.2%}")

        # 显示绩效指标
        print(f"\n📈 绩效指标:")
        print('-' * 40)
        perf = self.performance
        print(f"  年化收益率:    {perf['annual_return']:>8.2%}")
        print(f"  年化波动率:    {perf['annual_volatility']:>8.2%}")
        print(f"  夏普比率:      {perf['sharpe_ratio']:>8.2f}")
        print(f"  无风险利率:    {self.risk_free_rate:>8.2%}")

        # 风险调整后收益
        excess_return = perf['annual_return'] - self.risk_free_rate
        print(f"  超额收益率:    {excess_return:>8.2%}")

    def _get_top_assets(self, n=10):
        """
        获取权重最高的前n个资产

        Parameters:
        n: 返回的资产数量

        Returns:
        list: 前n大权重资产的代码列表
        """
        if self.weights is None:
            return []

        sorted_assets = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        top_assets = [asset for asset, weight in sorted_assets if weight > 0.001][:n]
        return top_assets

    def plot_asset_allocation(self):
        """绘制资产配置图表 - 只显示前10大权重资产"""
        if self.weights is None or self.data is None:
            print("请先执行优化!")
            return

        top_assets = self._get_top_assets(10)
        if not top_assets:
            print("没有足够的资产数据来绘制图表")
            return

        # 创建包含两个子图的图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 饼图显示资产配置
        weights_values = [self.weights[asset] for asset in top_assets]
        ax1.pie(weights_values, labels=top_assets, autopct='%1.1f%%', startangle=90)
        ax1.set_title('前10大资产配置权重', fontsize=14, fontweight='bold')

        # 2. 柱状图显示权重分布
        bars = ax2.bar(top_assets, weights_values, color='skyblue', alpha=0.7)
        ax2.set_title('前10大资产权重分布', fontsize=14, fontweight='bold')
        ax2.set_ylabel('权重')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 在柱状图上添加数值标签
        for i, v in enumerate(weights_values):
            ax2.text(i, v + 0.005, f"{v:.1%}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_performance_comparison(self):
        """绘制绩效对比图表 - 显示前10大资产与组合的对比"""
        if self.weights is None or self.data is None:
            print("请先执行优化!")
            return

        top_assets = self._get_top_assets(10)
        if not top_assets:
            print("没有足够的资产数据来绘制图表")
            return

        # 创建绩效对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 价格走势图（归一化）
        normalized_prices = self.data[top_assets] / self.data[top_assets].iloc[0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_assets)))

        for i, asset in enumerate(top_assets):
            ax1.plot(normalized_prices.index, normalized_prices[asset],
                     label=f"{asset} ({self.weights[asset]:.1%})",
                     linewidth=2, alpha=0.8, color=colors[i])

        ax1.set_title('前10大资产价格走势（归一化）', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格倍数 (起始=1.0)')
        ax1.set_xlabel('日期')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 处理日期标签重叠
        if len(normalized_prices) > 60:
            ax1.tick_params(axis='x', rotation=45)

        # 2. 收益率对比图
        total_days = len(self.data)
        individual_annual_returns = {}

        # 计算各资产的年化收益率
        for asset in top_assets:
            if asset in self.data.columns:
                total_return = (self.data[asset].iloc[-1] / self.data[asset].iloc[0] - 1)
                annual_return = (1 + total_return) ** (252 / total_days) - 1
                individual_annual_returns[asset] = annual_return

        assets_display = top_assets
        individual_returns = [individual_annual_returns[asset] for asset in assets_display]

        x_pos = np.arange(len(assets_display))
        bars = ax2.bar(x_pos, individual_returns, color='lightcoral', alpha=0.7,
                       label='个股年化收益率')

        # 添加组合收益率参考线
        ax2.axhline(y=self.performance['annual_return'], color='red', linestyle='--',
                    linewidth=2, label=f'组合年化收益率: {self.performance["annual_return"]:.2%}')

        ax2.set_title('前10大资产 vs 组合年化收益率', fontsize=14, fontweight='bold')
        ax2.set_ylabel('年化收益率')
        ax2.set_xlabel('资产')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(assets_display, rotation=45)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 在柱状图上添加数值标签
        for i, v in enumerate(individual_returns):
            ax2.text(i, v + 0.005, f"{v:.1%}", ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    def efficient_frontier_analysis(self, points=100):
        """
        分析有效前沿
        显示投资组合的有效边界和最大夏普比率组合
        """
        if self.data is None or self.weights is None:
            print("请先加载数据并执行优化!")
            return

        from pypfopt import plotting

        # 计算预期收益率和协方差矩阵
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        mu = log_returns.mean() * 252
        S = risk_models.sample_cov(self.data)

        # 创建有效前沿
        ef = EfficientFrontier(mu, S)
        fig, ax = plt.subplots(figsize=(12, 8))

        # 计算最大夏普比率组合
        ef_max_sharpe = ef.deepcopy()
        ef_max_sharpe.max_sharpe(risk_free_rate=self.risk_free_rate)
        ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()

        # 绘制有效前沿
        ef_efficient = ef.deepcopy()
        plotting.plot_efficient_frontier(ef_efficient, ax=ax, show_assets=False)

        # 获取最大夏普比率组合的前5大资产
        max_sharpe_weights = ef_max_sharpe.clean_weights()
        top_5_assets = sorted(max_sharpe_weights.items(), key=lambda x: x[1], reverse=True)[:5]

        # 创建资产信息文本
        top_assets_text = '最大夏普比率组合前5大资产:\n'
        for asset, weight in top_5_assets:
            if weight > 0.01:  # 只显示权重大于1%的资产
                top_assets_text += f"{asset}: {weight:.1%}\n"

        # 标记最大夏普比率点
        ax.scatter(std_tangent, ret_tangent, marker='*', s=200, c='red',
                   label=f"最大夏普比率组合\n年化收益: {ret_tangent:.1%}\n年化波动: {std_tangent:.1%}")

        # 添加资产信息文本框
        ax.text(0.02, 0.98, top_assets_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        ax.set_title('有效前沿与最大夏普比率组合', fontsize=14, fontweight='bold')
        ax.set_xlabel('年化波动率')
        ax.set_ylabel('年化收益率')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 在控制台打印详细信息
        print(f"\n🎯 最大夏普比率组合详细信息:")
        print(f"   年化收益率: {ret_tangent:.2%}")
        print(f"   年化波动率: {std_tangent:.2%}")
        print(f"   前5大权重资产:")
        for asset, weight in top_5_assets:
            if weight > 0.01:
                print(f"     {asset}: {weight:.2%}")

    def discrete_allocation(self, total_portfolio_value=100000):
        """
        离散资产分配 - 计算实际可购买的股票数量

        Parameters:
        total_portfolio_value: 总投资金额，默认10万美元
        """
        if self.weights is None or self.data is None:
            print("请先执行优化!")
            return

        try:
            # 获取最新价格
            latest_prices = get_latest_prices(self.data)
            da = DiscreteAllocation(self.weights, latest_prices,
                                    total_portfolio_value=total_portfolio_value)
            allocation, leftover = da.lp_portfolio()

            print(f"\n💵 离散资产分配 (总投资: ${total_portfolio_value:,}):")
            print('-' * 50)

            total_invested = 0
            # 显示每个资产的购买详情
            for asset, shares in allocation.items():
                price = latest_prices[asset]
                value = shares * price
                total_invested += value
                weight = self.weights[asset]
                print(f"  {asset:<8}: {shares:>6} 股 × ${price:>7.2f} = ${value:>9.2f} ({weight:>6.2%})")

            print('-' * 50)
            print(f"  股票总投资:   ${total_invested:>9.2f}")
            print(f"  剩余现金:     ${leftover:>9.2f}")
            print(f"  现金比例:     {leftover / total_portfolio_value:>9.2%}")

            return allocation, leftover

        except Exception as e:
            print(f"离散资产分配计算失败: {e}")
            return None, None


def main():
    """
    主函数 - 投资组合优化系统的入口点
    """
    print('=' * 70)
    print("PyPortfolioOpt 投资组合优化系统")
    print('=' * 70)

    # 创建优化器实例
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)

    # 加载数据
    if optimizer.load_stock_data_from_current_dir():
        # 显示数据详情
        print(f"\n📅 数据详情:")
        print(f"    总交易日数: {len(optimizer.data)}")
        print(f"    数据开始日期: {optimizer.data.index[0].strftime('%Y-%m-%d')}")
        print(f"    数据结束日期: {optimizer.data.index[-1].strftime('%Y-%m-%d')}")

        # 计算大约年数
        days = len(optimizer.data)
        years = days / 252  # 假设252个交易日一年
        print(f"    大约年数: {years:.1f} 年")

        # 执行分析流程
        optimizer.calculate_annual_returns()          # 年度收益率分析
        optimizer.calculate_individual_performance()  # 个股表现分析
        weights, performance = optimizer.optimize_portfolio()  # 组合优化

        if weights and performance:
            # 显示优化结果
            optimizer.print_optimization_results()

            # 生成可视化图表
            print(f"\n📊 正在生成资产配置图表...")
            optimizer.plot_asset_allocation()

            print(f"\n📈 正在生成绩效对比图表...")
            optimizer.plot_performance_comparison()

            print("\n📈 正在生成有效前沿...")
            optimizer.efficient_frontier_analysis()

            # 离散资产分配
            optimizer.discrete_allocation(total_portfolio_value=100000)

            print('\n' + '=' * 60)
            print(f"✅ 投资组合优化完成!")
            print('=' * 60)
        else:
            print(f"❌ 投资组合优化失败!")


if __name__ == "__main__":
    main()


'''
📊 系统功能总结：

1. 数据管理功能：
   - 自动加载当前目录的股票数据文件
   - 数据验证和清洗（时间范围、数据完整性）
   - 要求至少2年历史数据，确保分析可靠性

2. 分析计算功能：
   - 年度收益率分析（逐年显示收益情况）
   - 个股绩效分析（收益率、波动率、夏普比率）
   - 投资组合优化（最大化夏普比率）
   - 有效前沿分析（风险收益权衡）

3. 可视化功能：
   - 资产配置图表（饼图+柱状图）
   - 绩效对比图表（价格走势+收益率对比）
   - 有效前沿图表（标记最优组合）

4. 实用工具：
   - 离散资产分配（实际购买方案）
   - 集中度分析（前3/5大资产权重）
   - 风险调整收益计算（超额收益率）

🎯 核心算法：
   - 使用Markowitz现代投资组合理论
   - 基于均值-方差优化框架
   - 最大化夏普比率作为优化目标
   - 考虑协方差矩阵降低组合风险

💡 使用价值：
   - 为投资者提供科学的资产配置方案
   - 帮助理解风险与收益的平衡关系
   - 提供可视化的投资决策支持
   - 生成可执行的实际投资方案
'''