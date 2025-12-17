'''
第3天：
实现最小方差投资组合，比较其与最大夏普比率组合的区别。
练习：绘制有效前沿曲线，直观展示不同组合风险收益关系。
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

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class AdvancedPortfolioOptimizer:
    """
    高级投资组合优化器
    新增功能：最小方差组合、组合比较、有效前沿分析
    """

    def __init__(self, risk_free_rate=0.02):
        """
        初始化优化器

        Parameters:
        risk_free_rate: 无风险利率，默认2%
        """
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.returns = None
        self.weights_max_sharpe = None  # 最大夏普比率组合权重
        self.weights_min_vol = None     # 最小方差组合权重
        self.performance_max_sharpe = None
        self.performance_min_vol = None

    def load_stock_data_from_current_dir(self):
        """
        从当前目录加载股票数据
        """
        print("正在从当前目录加载股票数据...")
        all_data = {}
        valid_tickers = []

        stock_files = glob.glob('./*_stock_data.xlsx')

        if not stock_files:
            print("错误: 当前目录下未找到股票数据文件")
            print("请确保文件命名格式为: ./AAPL_stock_data.xlsx")
            return False

        print(f"找到 {len(stock_files)} 个股票数据文件")

        for file_path in stock_files:
            filename = os.path.basename(file_path)
            ticker = filename.replace('_stock_data.xlsx', '')

            try:
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)

                if 'Close' in df.columns and len(df) > 500:
                    df = df.sort_index()
                    date_range = df.index[-1] - df.index[0]
                    years = date_range.days / 365.25

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

        if len(valid_tickers) < 2:
            print(f"错误: 需要至少2只股票进行组合优化，当前只有 {len(valid_tickers)} 只")
            return False

        self.data = pd.DataFrame(all_data)
        self.data = self.data.sort_index()
        self.data = self.data.ffill().dropna()

        if len(self.data) < 500:
            print(f"错误: 合并后数据量不足，至少需要500个交易日，当前只有 {len(self.data)} 天")
            return False

        self.returns = self.data.pct_change().dropna()

        total_days = len(self.data)
        date_range = self.data.index[-1] - self.data.index[0]
        years = date_range.days / 365.25

        print(f"\n✅ 数据加载完成!")
        print(f"   有效股票数量: {len(self.data.columns)}")
        print(f"   交易日数: {len(self.data)}")
        print(f"   时间范围: {self.data.index[0].strftime('%Y-%m-%d')} 到 {self.data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   数据覆盖: {years:.1f} 年")

        return True

    def optimize_both_strategies(self, weight_bounds=(0.01, 0.4)):
        """
        同时优化两种策略：最大夏普比率和最小方差

        Parameters:
        weight_bounds: 权重限制范围
        """
        if self.data is None or self.returns is None:
            print("请先加载数据!")
            return None, None

        print("\n" + "=" * 70)
        print("开始双策略投资组合优化")
        print("=" * 70)

        # 计算预期收益率和协方差矩阵
        print("\n1. 计算预期收益率和风险模型...")
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        mu = log_returns.mean() * 252
        S = risk_models.sample_cov(self.data)

        # 创建有效前沿对象
        ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

        print("\n2. 优化最大夏普比率组合...")
        try:
            ef_max_sharpe = ef.deepcopy()
            ef_max_sharpe.max_sharpe(risk_free_rate=self.risk_free_rate)
            self.weights_max_sharpe = ef_max_sharpe.clean_weights()

            # 计算绩效指标
            ret_sharpe, vol_sharpe, sharpe_ratio = ef_max_sharpe.portfolio_performance(
                risk_free_rate=self.risk_free_rate, verbose=False
            )

            self.performance_max_sharpe = {
                'annual_return': ret_sharpe,
                'annual_volatility': vol_sharpe,
                'sharpe_ratio': sharpe_ratio,
                'strategy': '最大夏普比率'
            }
            print("✓ 最大夏普比率组合优化成功")

        except Exception as e:
            print(f"✗ 最大夏普比率优化失败: {e}")
            return None, None

        print("\n3. 优化最小方差组合...")
        try:
            ef_min_vol = ef.deepcopy()
            ef_min_vol.min_volatility()
            self.weights_min_vol = ef_min_vol.clean_weights()

            # 计算绩效指标
            ret_min_vol, vol_min_vol, sharpe_min_vol = ef_min_vol.portfolio_performance(
                risk_free_rate=self.risk_free_rate, verbose=False
            )

            self.performance_min_vol = {
                'annual_return': ret_min_vol,
                'annual_volatility': vol_min_vol,
                'sharpe_ratio': sharpe_min_vol,
                'strategy': '最小方差'
            }
            print("✓ 最小方差组合优化成功")

        except Exception as e:
            print(f"✗ 最小方差优化失败: {e}")
            return None, None

        return (self.weights_max_sharpe, self.performance_max_sharpe), (self.weights_min_vol, self.performance_min_vol)

    def print_comparison_results(self):
        """
        打印两种策略的对比结果
        """
        if (self.performance_max_sharpe is None or
            self.performance_min_vol is None):
            print("请先执行双策略优化!")
            return

        print("\n" + "=" * 80)
        print("🎯 双策略投资组合对比分析")
        print("=" * 80)

        # 创建对比表格
        comparison_data = []

        # 最大夏普比率组合数据
        sharpe_perf = self.performance_max_sharpe
        comparison_data.append({
            '策略': '最大夏普比率',
            '年化收益率': f"{sharpe_perf['annual_return']:.2%}",
            '年化波动率': f"{sharpe_perf['annual_volatility']:.2%}",
            '夏普比率': f"{sharpe_perf['sharpe_ratio']:.2f}",
            '风险调整收益': '最优'
        })

        # 最小方差组合数据
        min_vol_perf = self.performance_min_vol
        comparison_data.append({
            '策略': '最小方差',
            '年化收益率': f"{min_vol_perf['annual_return']:.2%}",
            '年化波动率': f"{min_vol_perf['annual_volatility']:.2%}",
            '夏普比率': f"{min_vol_perf['sharpe_ratio']:.2f}",
            '风险调整收益': '稳健'
        })

        # 计算差异
        return_diff = sharpe_perf['annual_return'] - min_vol_perf['annual_return']
        vol_diff = sharpe_perf['annual_volatility'] - min_vol_perf['annual_volatility']

        comparison_data.append({
            '策略': '差异',
            '年化收益率': f"{return_diff:+.2%}",
            '年化波动率': f"{vol_diff:+.2%}",
            '夏普比率': f"{sharpe_perf['sharpe_ratio'] - min_vol_perf['sharpe_ratio']:+.2f}",
            '风险调整收益': '风险收益权衡'
        })

        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 策略绩效对比:")
        print(comparison_df.to_string(index=False))

        # 打印权重对比
        self._print_weights_comparison()

    def _print_weights_comparison(self):
        """
        打印两种策略的权重对比
        """
        print(f"\n📈 资产权重分配对比:")
        print("-" * 70)
        print(f"{'资产':<12} {'最大夏普权重':<15} {'最小方差权重':<15} {'权重差异':<15}")
        print("-" * 70)

        all_assets = set(self.weights_max_sharpe.keys()) | set(self.weights_min_vol.keys())

        for asset in sorted(all_assets):
            weight_sharpe = self.weights_max_sharpe.get(asset, 0)
            weight_min_vol = self.weights_min_vol.get(asset, 0)
            weight_diff = weight_sharpe - weight_min_vol

            if weight_sharpe > 0.001 or weight_min_vol > 0.001:
                sharpe_str = f"{weight_sharpe:.2%}" if weight_sharpe > 0.001 else "0.00%"
                min_vol_str = f"{weight_min_vol:.2%}" if weight_min_vol > 0.001 else "0.00%"
                diff_str = f"{weight_diff:+.2%}"

                print(f"{asset:<12} {sharpe_str:<15} {min_vol_str:<15} {diff_str:<15}")

    def plot_efficient_frontier_with_both_strategies(self, points=100):
        """
        绘制包含两种策略的有效前沿曲线

        Parameters:
        points: 有效前沿上的点数
        """
        if self.data is None:
            print("请先加载数据!")
            return

        from pypfopt import plotting

        print("\n正在生成有效前沿曲线...")

        # 计算预期收益率和协方差矩阵
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        mu = log_returns.mean() * 252
        S = risk_models.sample_cov(self.data)

        # 创建有效前沿
        ef = EfficientFrontier(mu, S)

        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))

        # 绘制有效前沿
        plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

        # 标记最大夏普比率点
        if self.performance_max_sharpe:
            ret_sharpe = self.performance_max_sharpe['annual_return']
            vol_sharpe = self.performance_max_sharpe['annual_volatility']
            ax.scatter(vol_sharpe, ret_sharpe, marker="*", s=300, c="red",
                      label=f"最大夏普比率组合\n收益: {ret_sharpe:.1%}\n波动: {vol_sharpe:.1%}\n夏普: {self.performance_max_sharpe['sharpe_ratio']:.2f}")

        # 标记最小方差点
        if self.performance_min_vol:
            ret_min_vol = self.performance_min_vol['annual_return']
            vol_min_vol = self.performance_min_vol['annual_volatility']
            ax.scatter(vol_min_vol, ret_min_vol, marker="D", s=300, c="green",
                      label=f"最小方差组合\n收益: {ret_min_vol:.1%}\n波动: {vol_min_vol:.1%}\n夏普: {self.performance_min_vol['sharpe_ratio']:.2f}")

        # 添加理论说明
        ax.text(0.02, 0.98,
                "投资组合理论说明:\n"
                "• 有效前沿: 最优风险收益边界\n"
                "• 最大夏普: 最优风险调整收益\n" 
                "• 最小方差: 最低波动率组合",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=10)

        ax.set_title("有效前沿与投资组合策略对比", fontsize=16, fontweight='bold')
        ax.set_xlabel("年化波动率 (风险)", fontsize=12)
        ax.set_ylabel("年化收益率", fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_weights_comparison_chart(self):
        """
        绘制两种策略的权重对比图表
        """
        if (self.weights_max_sharpe is None or
            self.weights_min_vol is None):
            print("请先执行双策略优化!")
            return

        # 获取前10大权重资产（基于最大夏普组合）
        top_assets = self._get_top_assets(self.weights_max_sharpe, 10)

        if not top_assets:
            print("没有足够的资产数据来绘制图表")
            return

        # 创建对比图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. 权重对比柱状图
        sharpe_weights = [self.weights_max_sharpe.get(asset, 0) for asset in top_assets]
        min_vol_weights = [self.weights_min_vol.get(asset, 0) for asset in top_assets]

        x = np.arange(len(top_assets))
        width = 0.35

        bars1 = ax1.bar(x - width/2, sharpe_weights, width, label='最大夏普比率',
                       color='red', alpha=0.7)
        bars2 = ax1.bar(x + width/2, min_vol_weights, width, label='最小方差',
                       color='green', alpha=0.7)

        ax1.set_xlabel('资产')
        ax1.set_ylabel('权重')
        ax1.set_title('两种策略的资产权重对比', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(top_assets, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.01:  # 只显示大于1%的权重
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{height:.1%}', ha='center', va='bottom', fontsize=8)

        # 2. 绩效对比雷达图
        categories = ['收益率', '波动率', '夏普比率', '风险调整']

        # 标准化数据用于雷达图
        sharpe_values = [
            self.performance_max_sharpe['annual_return'] * 10,  # 放大显示
            1 - self.performance_max_sharpe['annual_volatility'],  # 波动率取反
            self.performance_max_sharpe['sharpe_ratio'] * 2,    # 放大显示
            self.performance_max_sharpe['sharpe_ratio'] * 2     # 风险调整能力
        ]

        min_vol_values = [
            self.performance_min_vol['annual_return'] * 10,
            1 - self.performance_min_vol['annual_volatility'],
            self.performance_min_vol['sharpe_ratio'] * 2,
            self.performance_min_vol['sharpe_ratio'] * 2
        ]

        # 闭合数据
        sharpe_values += sharpe_values[:1]
        min_vol_values += min_vol_values[:1]
        categories_radar = categories + [categories[0]]

        angles = np.linspace(0, 2*np.pi, len(categories_radar), endpoint=True)

        ax2 = fig.add_subplot(122, polar=True)
        ax2.plot(angles, sharpe_values, 'o-', linewidth=2, label='最大夏普比率', color='red')
        ax2.fill(angles, sharpe_values, alpha=0.25, color='red')
        ax2.plot(angles, min_vol_values, 'o-', linewidth=2, label='最小方差', color='green')
        ax2.fill(angles, min_vol_values, alpha=0.25, color='green')

        ax2.set_thetagrids(angles[:-1] * 180/np.pi, categories)
        ax2.set_title('策略绩效雷达图对比', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def _get_top_assets(self, weights, n=10):
        """获取权重最高的前n个资产"""
        sorted_assets = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_assets = [asset for asset, weight in sorted_assets if weight > 0.001][:n]
        return top_assets

    def discrete_allocation_comparison(self, total_portfolio_value=100000):
        """
        比较两种策略的离散资产分配
        """
        if (self.weights_max_sharpe is None or
            self.weights_min_vol is None or
            self.data is None):
            print("请先执行优化!")
            return

        try:
            latest_prices = get_latest_prices(self.data)

            print(f"\n💵 离散资产分配对比 (总投资: ${total_portfolio_value:,}):")
            print("=" * 70)

            # 最大夏普比率组合分配
            print(f"\n📈 最大夏普比率组合分配:")
            print("-" * 50)
            da_sharpe = DiscreteAllocation(self.weights_max_sharpe, latest_prices,
                                         total_portfolio_value=total_portfolio_value)
            allocation_sharpe, leftover_sharpe = da_sharpe.lp_portfolio()

            self._print_allocation_details(allocation_sharpe, latest_prices, self.weights_max_sharpe)
            print(f"剩余现金: ${leftover_sharpe:>9.2f}")

            # 最小方差组合分配
            print(f"\n🛡️  最小方差组合分配:")
            print("-" * 50)
            da_min_vol = DiscreteAllocation(self.weights_min_vol, latest_prices,
                                          total_portfolio_value=total_portfolio_value)
            allocation_min_vol, leftover_min_vol = da_min_vol.lp_portfolio()

            self._print_allocation_details(allocation_min_vol, latest_prices, self.weights_min_vol)
            print(f"剩余现金: ${leftover_min_vol:>9.2f}")

        except Exception as e:
            print(f"离散资产分配计算失败: {e}")

    def _print_allocation_details(self, allocation, latest_prices, weights):
        """打印分配详情"""
        total_invested = 0
        for asset, shares in allocation.items():
            price = latest_prices[asset]
            value = shares * price
            total_invested += value
            weight = weights[asset]
            print(f"  {asset:<8}: {shares:>6} 股 × ${price:>7.2f} = ${value:>9.2f} ({weight:>6.2%})")

        print(f"{'总投资':<8}: ${total_invested:>9.2f}")


def main():
    """
    主函数 - 第3天任务执行
    """
    print('=' * 70)
    print("第3天：最小方差组合 vs 最大夏普比率组合")
    print('=' * 70)

    # 创建高级优化器实例
    optimizer = AdvancedPortfolioOptimizer(risk_free_rate=0.02)

    # 加载数据
    if optimizer.load_stock_data_from_current_dir():
        print(f"\n📅 数据详情:")
        print(f"    总交易日数: {len(optimizer.data)}")
        print(f"    数据开始日期: {optimizer.data.index[0].strftime('%Y-%m-%d')}")
        print(f"    数据结束日期: {optimizer.data.index[-1].strftime('%Y-%m-%d')}")

        days = len(optimizer.data)
        years = days / 252
        print(f"    大约年数: {years:.1f} 年")

        # 执行双策略优化
        print(f"\n🚀 开始执行双策略投资组合优化...")
        result_sharpe, result_min_vol = optimizer.optimize_both_strategies()

        if result_sharpe and result_min_vol:
            # 显示对比结果
            optimizer.print_comparison_results()

            # 生成可视化图表
            print(f"\n📊 正在生成有效前沿曲线...")
            optimizer.plot_efficient_frontier_with_both_strategies()

            print(f"\n📈 正在生成权重对比图表...")
            optimizer.plot_weights_comparison_chart()

            # 离散资产分配对比
            optimizer.discrete_allocation_comparison(total_portfolio_value=100000)

            print('\n' + '=' * 70)
            print("✅ 第3天任务完成！")
            print("   成功实现最小方差组合并与最大夏普比率组合进行对比分析")
            print('=' * 70)
        else:
            print(f"❌ 双策略优化失败!")


if __name__ == "__main__":
    main()


'''
📊 第3天任务总结：

🎯 核心成果：
1. ✅ 实现最小方差投资组合优化
2. ✅ 完成与最大夏普比率组合的全面对比
3. ✅ 绘制包含两种策略的有效前沿曲线
4. ✅ 直观展示不同组合的风险收益关系

📈 理论价值：
• 理解Markowitz投资组合理论的两个重要特例
• 掌握风险收益权衡的量化分析方法
• 学会在不同市场环境下选择合适的投资策略

💡 实践应用：
• 激进投资者：选择最大夏普比率组合，追求最优风险调整收益
• 保守投资者：选择最小方差组合，注重资本保值和风险控制
• 机构投资者：根据客户风险偏好灵活配置两种策略

🔍 关键发现：
1. 最大夏普组合通常有更高的收益率但伴随较高波动
2. 最小方差组合提供最低风险但可能牺牲部分收益
3. 有效前沿展示了理论上的最优风险收益边界
4. 两种策略的资产配置差异反映了不同的风险分散逻辑
'''