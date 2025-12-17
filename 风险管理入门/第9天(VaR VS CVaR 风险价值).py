'''
第9天：
学习并实现VaR（风险价值）和CVaR（条件风险价值）指标。
练习：使用历史模拟法和正态分布法计算VaR和CVaR。
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体 - 确保图表能正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class VARCVARCalculator:
    """
        VaR和CVaR风险指标计算器 - 完全基于真实数据

        设计理念:
        - 使用真实历史数据计算所有统计指标，避免主观假设
        - 提供两种VaR计算方法，便于比较和验证
        - 完整的可视化分析，直观展示风险特征
        """
    def __init__(self, portfolio):
        """
                初始化风险计算器

                参数:
                    portfolio: 投资组合字典 {股票代码: 投资金额}

                数据成员说明:
                - total_value: 总投资金额，预计算提高效率
                - stock_data: 存储各股票的原始价格数据
                - returns_data: 存储各股票的收益率数据（基于价格计算）
                - stock_stats: 存储各股票的详细统计信息
                """
        self.portfolio = portfolio
        # 计算总投资金额 - 预计算避免重复计算
        self.total_value = sum(portfolio.values())
        self.stock_data = {}        # 存储股票价格数据 {股票代码: 价格序列}
        self.returns_data = {}      # 存储股票收益率数据 {股票代码: 收益率序列}
        self.stock_stats = {}       # 存储各股票的统计信息
        print("💰 VaR和CVaR风险分析初始化...")
        print("=" * 50)

        # 详细展示投资组合构成，便于验证数据准确性
        for stock, value in portfolio.items():
            print(f"{stock}: ${value}({value/self.total_value:.1%})")
        print(f"总投资: ${self.total_value}")
        print("=" * 50)

    def load_stock_data(self):
        """
               加载股票数据并基于真实数据计算统计指标

               执行流程:
               1. 尝试从Excel文件加载股票价格数据
               2. 计算日收益率、波动率等统计指标
               3. 如果数据加载失败，跳过该股票

               设计考虑:
               - 支持多种价格列名，提高代码兼容性
               - 要求至少30个数据点，确保统计显著性
               - 详细输出每只股票的统计信息，便于验证
               """
        print("\n📊 加载股票数据并计算真实统计指标...")
        for stock in self.portfolio.keys():
            try:
                # 尝试从Excel文件加载数据 - 假设数据文件名为 {股票代码}_stock_data.xlsx
                file_path = f'./{stock}_stock_data.xlsx'
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                # 寻找价格列 - 支持多种常见的列名格式
                price_columns = ['Close', 'close', 'Adj Close', 'Price', 'price']
                price_col = next((col for col in price_columns if col in df.columns), None)
                if price_col:
                    prices = df[price_col].dropna()
                    # 确保有足够的数据进行统计分析
                    if len(prices) < 30:
                        raise ValueError(f"数据量不足，只有{len(prices)}天数据")
                    # 存储原始价格数据
                    self.stock_data[stock] = prices
                    #   计算收益率并存储
                    returns = self.calculate_returns(prices)
                    self.returns_data[stock] = returns

                    # ==================== 基于真实数据计算统计指标 ====================
                    daily_return = returns.mean()       # 日均收益率
                    volatility = returns.std()          # 日波动率（标准差）
                    annual_return = daily_return * 252  # 年化收益率 = 日收益率 × 252个交易日
                    annual_volatility = volatility * np.sqrt(252)   # 年化波动率 = 日波动率 × √252

                    # 计算最大回撤 - 衡量历史最差表现
                    max_drawdown = self.calculate_max_drawdown(prices)

                    # 计算夏普比率 - 风险调整后收益（假设无风险利率为2%）
                    risk_free_rate = 0.02 / 252     # 日无风险利率
                    sharpe_ratio = (daily_return - risk_free_rate) / volatility if volatility >0 else 0
                    annual_sharpe = sharpe_ratio * np.sqrt(252) # 年化夏普比率

                    # 存储股票的详细统计信息
                    self.stock_stats[stock] = {
                        'daily_return': daily_return,
                        'volatility': volatility,
                        'annual_return': annual_return,
                        'annual_volatility': annual_volatility,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'data_points': len(prices),
                        'data_period': f"{prices.index[0].strftime('%Y-%m-%d')} 至 {prices.index[-1].strftime('%Y-%m-%d')}",
                        'is_real_data': True         # 标记为真实数据
                    }
                    # 详细输出每只股票的统计信息
                    print(f"✅ {stock}: {len(prices)}天数据 ({self.stock_stats[stock]['data_period']})")
                    print(f"   📈 日收益: {daily_return * 100:+.4f}% | 年化收益: {annual_return * 100:+.2f}%")
                    print(f"   📊 日波动: {volatility * 100:.4f}% | 年化波动: {annual_volatility * 100:.2f}%")
                    print(f"   ⚠️  最大回撤: {max_drawdown * 100:.2f}% | 夏普比率: {annual_sharpe:.2f}")
                    print(f"   ──────────────────────────────────────────")

                else:
                    raise ValueError("未找到价格列")
            except Exception as e:
                # 如果数据加载失败，跳过该股票并继续处理其他股票
                print(f"❌ {stock}: 数据加载失败 - {e}")
                print(f"   💡 无法获取真实数据，跳过该股票")
                continue

    def calculate_returns(self, prices):
        """
                计算日收益率 - 基于真实价格数据

                公式: r_t = (P_t - P_{t-1}) / P_{t-1}

                参数:
                    prices: 价格序列 (pd.Series)

                返回:
                    returns: 收益率序列 (pd.Series)

                为什么使用百分比收益率而不是对数收益率:
                - 百分比收益率更直观，易于理解
                - 金融行业标准，便于与其他工具对接
                - 对于日收益率，两种方法差异很小 """
        # 计算百分比变化: (今日价格 - 昨日价格) / 昨日价格
        returns = prices.pct_change().dropna()
        return returns

    def calculate_max_drawdown(self, prices):
        """
        计算最大回撤 - 基于真实价格数据

        最大回撤定义: 从前期高点到后期低点的最大跌幅
        计算公式: Max Drawdown = (波谷值 - 峰值) / 峰值

        参数:
            prices: 价格序列

        返回:
            max_drawdown: 最大回撤 (负数表示损失)

        为什么计算最大回撤:
        - 衡量历史最差表现
        - 反映投资组合的下跌风险
        - 是风险管理的重要指标
        """
        # 计算累积收益率
        cumulative_returns = (1 + self.calculate_returns(prices)).cumprod()
        # 计算历史峰值
        peak = cumulative_returns.expanding().max()
        # 计算回撤: (当前值 - 峰值) / 峰值
        drawdown = (cumulative_returns - peak) / peak
        # 找到最大回撤（最小值）
        max_drawdown = drawdown.min()
        return max_drawdown

    def calculate_portfolio_returns(self):
        """
               计算投资组合的日收益率 - 基于真实数据

               投资组合收益率公式: R_p = Σ(w_i × r_i)
               其中: w_i = 第i只股票的权重, r_i = 第i只股票的收益率

               返回:
                   portfolio_returns: 投资组合收益率序列

               设计考虑:
               - 只使用成功加载的股票数据
               - 找到所有股票的共有日期范围，确保数据一致性
               - 重新计算权重，反映实际可用的投资组合
               """
        if not self.returns_data:
            raise ValueError("没有可用的股票数据，请先成功加载至少一只股票的数据")
        if len(self.returns_data) == 0:
            raise ValueError('没有成功加载任何股票数据')

        # 找到所有股票共有的日期范围 - 确保数据时间对齐
        common_datas = None
        for returns in self.returns_data.values():
            if common_datas is None:
                common_datas = returns.index
            else:
                common_datas = common_datas.intersection(returns.index)
        if len(common_datas) == 0:
            raise ValueError('股票数据没有共同的日期范围')

        # 重新计算权重（只包括成功加载的股票）
        available_stocks = list(self.returns_data.keys())
        available_value = sum(self.portfolio[stock] for stock in available_stocks)

        print(f"\n📋 使用 {len(available_stocks)} 只股票计算投资组合:")
        for stock in available_stocks:
            weight = self.portfolio[stock] / available_value
            print(f"   {stock}: {weight:.1%}")
        # 初始化投资组合收益率序列
        portfolio_returns = pd.Series(0.0, index=common_datas)
        # 按权重加权计算投资组合收益率
        for stock, returns in self.returns_data.items():
            weight = self.portfolio[stock] / available_value
            aligned_returns = returns.loc[common_datas]     # 对齐日期
            portfolio_returns += aligned_returns * weight
        return portfolio_returns

    def historical_var_cvar(self, portfolio_returns, confidence_level=0.95):
        '''
        使用历史模拟法计算VaR和CVaR - 基于真实收益率数据

        历史模拟法原理:
        - 不假设收益率分布，直接使用历史数据的分位数
        - VaR = 历史收益率的分位数
        - CVaR = 超过VaR的所有损失的平均值

        公式:
        VaR_historical = Percentile(returns, 1 - confidence_level)
        CVaR_historical = Mean(returns < VaR_historical)

        参数:
            portfolio_returns: 投资组合收益率序列
            confidence_level: 置信水平 (0.95 或 0.99)
        返回:
            historical_var: 历史模拟法VaR
            historical_cvar: 历史模拟法CVaR

        优点:
        - 不需要分布假设，更符合实际市场
        - 能够捕捉市场的肥尾现象
        - 计算简单直观
        '''
        # 对收益率进行排序（从小到大） - 便于计算分位数
        sorted_returns = np.sort(portfolio_returns)
        # 计算VaR (历史分位数)
        # 例如95%置信水平：取5%分位点的收益率
        var_index = int((1 - confidence_level) * len(sorted_returns))
        historical_var = sorted_returns[var_index]

        # 计算CVaR (超过VaR的所有损失的平均值)
        # 反映在极端情况下的平均损失程度
        tail_returns = sorted_returns[:var_index]   # 所有小于VaR的收益率
        historical_cvar = np.mean(tail_returns) if len(tail_returns) > 0 else historical_var
        return historical_var, historical_cvar

    def parametric_var_cvar(self, portfolio_returns, confidence_level=0.95):
        """
        使用参数法（正态分布法）计算VaR和CVaR - 基于真实统计参数

        参数法原理:
        - 假设投资组合收益率服从正态分布
        - 基于均值和标准差计算风险指标

        公式:
        VaR_param = μ + Z_{1-α} × σ
        CVaR_param = μ - (σ × φ(Z_{1-α}) / (1 - α))

        其中:
        μ: 收益率均值, σ: 收益率标准差
        Z_{1-α}: 标准正态分布的(1-α)分位数
        φ(): 标准正态分布的概率密度函数
        参数:
            portfolio_returns: 投资组合收益率序列
            confidence_level: 置信水平

        返回:
            parametric_var: 参数法VaR
            parametric_cvar: 参数法CVaR

        优点:
        - 计算速度快
        - 只需要均值和标准差两个参数
        - 理论基础完善
        """
        # 计算均值和标准差 - 正态分布的两个关键参数
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)

        # 计算VaR (基于正态分布分位数)
        # Z_score: 标准正态分布的分位数
        # 例如95%置信水平对应Z=-1.645, 99%对应Z=-2.326
        z_score = stats.norm.ppf(1 - confidence_level)
        parametric_var = mean_return + z_score * std_return

        # 计算CVaR (正态分布下的期望短缺)
        # 公式推导基于条件期望理论
        parametric_cvar = mean_return - (std_return * stats.norm.pdf(z_score)/ (1 - confidence_level))
        return parametric_var, parametric_cvar

    def calculate_risk_metrics(self, confidence_levels=[0.95, 0.99]):
        """
               计算所有风险指标 - 完全基于真实数据

               置信水平选择原理:
               - 95%: 常用水平，对应20个交易日发生1次超过VaR的损失
               - 99%: 更保守的水平，对应100个交易日发生1次超过VaR的损失

               返回:
                   results: 包含所有风险指标的字典
                   portfolio_returns: 投资组合收益率序列
               """
        print("\n🚀 开始基于真实数据计算风险指标...")
        # 计算投资组合收益率 - 风险分析的基础
        portfolio_returns = self.calculate_portfolio_returns()

        # ==================== 计算投资组合整体统计指标 ====================
        portfolio_daily_return = portfolio_returns.mean()
        portfolio_volatility = portfolio_returns.std()
        portfolio_annual_return = portfolio_daily_return * 252
        portfolio_annual_volatility = portfolio_volatility * np.sqrt(252)

        # 计算投资组合最大回撤
        portfolio_max_drawdown = self.calculate_max_drawdown_from_returns(portfolio_returns)

        print(f"\n📊 投资组合整体统计 (基于{len(portfolio_returns)}个交易日):")
        print(f"   📈 日收益率: {portfolio_daily_return * 100:+.4f}%")
        print(f"   📊 日波动率: {portfolio_volatility * 100:.4f}%")
        print(f"   💰 年化收益率: {portfolio_annual_return * 100:+.2f}%")
        print(f"   ⚡ 年化波动率: {portfolio_annual_volatility * 100:.2f}%")
        print(f"   ⚠️  最大回撤: {portfolio_max_drawdown * 100:.2f}%")
        print(f"   📅 数据期间: {portfolio_returns.index[0].strftime('%Y-%m-%d')} 至"
            f" {portfolio_returns.index[-1].strftime('%Y-%m-%d')}")

        results = {}

        # 对每个置信水平计算风险指标
        for confidence in confidence_levels:
            print(f"\n📈 计算 {confidence * 100}% 置信水平下的风险指标...")
            # 历史模拟法 - 基于实际数据
            hist_var, hist_cvar = self.historical_var_cvar(portfolio_returns, confidence)
            # 参数法 - 基于分布假设
            param_var, param_cvar = self.parametric_var_cvar(portfolio_returns, confidence)
            # 转换为金额形式 - 便于业务理解
            hist_var_amount = abs(hist_var) * self.total_value
            hist_cvar_amount = abs(hist_cvar) * self.total_value
            param_var_amount = abs(param_var) * self.total_value
            param_cvar_amount = abs(param_cvar) * self.total_value

            # 存储结果 - 结构化数据便于后续分析
            results[confidence] = {
                'historical': {
                    'var': hist_var,
                    'cvar': hist_cvar,
                    'var_pct': abs(hist_var) * 100,      # 百分比形式
                    'cvar_pct': abs(hist_cvar) * 100,    # 百分比形式
                    'var_amount': hist_var_amount,       # 金额形式
                    'cvar_amount': hist_cvar_amount      # 金额形式
                },
                'parametric': {
                    'var': param_var,
                    'cvar': param_cvar,
                    'var_pct': abs(param_var) * 100,
                    'cvar_pct': abs(param_cvar) * 100,
                    'var_amount': param_var_amount,
                    'cvar_amount': param_cvar_amount
                }
            }
            # 打印结果 - 即时反馈
            print(f"   历史模拟法: VaR = {abs(hist_var) * 100:.2f}% (${hist_var_amount:.2f}), "
                  f"CVaR = {abs(hist_cvar) * 100:.2f}% (${hist_cvar_amount:.2f})")
            print(f"   参数法: VaR = {abs(param_var) * 100:.2f}% (${param_var_amount:.2f}), "
                  f"CVaR = {abs(param_cvar) * 100:.2f}% (${param_cvar_amount:.2f})")
        return results, portfolio_returns

    def calculate_max_drawdown_from_returns(self, returns):
        """
                从收益率序列计算最大回撤

                参数:
                    returns: 收益率序列

                返回:
                    max_drawdown: 最大回撤
                """
        # 计算累积收益率
        cumulative_returns = (1 + returns).cumprod()
        # 计算历史峰值
        peak = cumulative_returns.expanding().max()
        # 计算回撤
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    # ==================== 可视化分析方法 ====================
    def plot_comprehensive_analysis(self, results, portfolio_returns):
        """
               绘制综合分析图表 - 分成两个图表显示

               设计理念:
               - 避免信息过载，分两个图表显示
               - 每个图表聚焦特定的分析维度
               - 提供完整的风险视角
               """
        print("\n🎨 生成风险分析图表...")
        # 第一个图表：基础分布和比较
        self._plot_chart1_returns_and_comparison(results, portfolio_returns)
        # 第二个图表：风险关系和贡献分析
        self._plot_chart2_relationship_and_contribution(results)

    def _plot_chart1_returns_and_comparison(self, results, portfolio_returns):
        """
                第一个图表：收益率分布和VaR比较

                包含:
                - 收益率分布直方图：了解收益率的统计特性
                - VaR方法比较：对比不同计算方法的差异
                """
        # 创建1行2列的子图 - 并排比较
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig.suptitle('投资组合风险分析 - 收益率分布与VaR比较', fontsize=16, fontweight='bold')

        # 图表1: 收益率分布与VaR/CVaR标记
        self._plot_returns_distribution(ax1, portfolio_returns, results)

        # 图表2: 不同方法VaR比较
        self._plot_var_comparison(ax2, results)
        plt.tight_layout()
        plt.show()

    def _plot_chart2_relationship_and_contribution(self, results):
        """
                第二个图表：VaR-CVaR关系和风险贡献

                包含:
                - VaR vs CVaR关系：理解尾部风险特征
                - 风险贡献分析：识别主要风险来源
                """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
        fig.suptitle('投资组合风险分析 - 风险关系与贡献度', fontsize=16, fontweight='bold')

        # 图表3: VaR和CVaR对比
        self._plot_var_cvar_comparison(ax1, results)

        # 图表4: 风险贡献分析
        self._plot_risk_contribution(ax2)

        plt.tight_layout()
        plt.show()

    def _plot_returns_distribution(self, ax, portfolio_returns, results):
        """
                绘制收益率分布直方图 - 增强版：同时显示VaR和CVaR

                分析目的:
                - 检验收益率是否接近正态分布
                - 识别分布的偏度和峰度
                - 可视化VaR和CVaR在分布中的位置
                """
        # 将收益率转换为百分比 - 便于理解和比较
        returns_pct = portfolio_returns * 100
        # 绘制收益率分布直方图 - 直观展示数据分布
        ax.hist(returns_pct, bins=50, alpha=0.7, color='lightblue',
                edgecolor='black', density=True)
        # 添加正态分布曲线 - 对比实际分布与理论分布
        x = np.linspace(returns_pct.min(), returns_pct.max(), 100)
        mean, std = returns_pct.mean(), returns_pct.std()
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'r-', linewidth=2, label='正态分布')

        # 标记VaR和CVaR水平
        confidence_levels = [0.95, 0.99]
        var_colors = ['red', 'darkred']
        cvar_colors = ['orange', 'darkorange']

        for i, confidence in enumerate(confidence_levels):
            # VaR标记 - 虚线
            var_pct = results[confidence]['historical']['var_pct']
            ax.axvline(x=-var_pct, color=var_colors[i], linestyle='--',
                       linewidth=2, label=f'{confidence*100}% VaR: {var_pct:.2f}%')
            # CVaR标记 - 点划线
            cvar_pct = results[confidence]['historical']['cvar_pct']
            ax.axvline(x=-cvar_pct, color=cvar_colors[i], linestyle='-',
                       linewidth=2, label=f'{confidence*100}% CVaR: {cvar_pct:.2f}%')
            # 添加VaR和CVaR之间的填充区域 - 显示尾部风险区域
            x_fill = np.linspace(-cvar_pct, -var_pct, 50)
            y_fill = stats.norm.pdf(x_fill, mean, std)
            ax.fill_between(x_fill, y_fill, alpha=0.3, color=cvar_colors[i])

        ax.set_title('投资组合收益率分布 - VaR和CVaR风险标记', fontweight='bold', fontsize=14)
        ax.set_xlabel('日收益率 (%)', fontsize=12)
        ax.set_ylabel('概率密度', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 添加统计信息文本框
        stats_text = f"""分布统计:
        均值: {returns_pct.mean():.3f}%
        标准差: {returns_pct.std():.3f}%
        偏度: {stats.skew(returns_pct):.3f}
        峰度: {stats.kurtosis(returns_pct):.3f}

        风险解释:
        • VaR: 最大可能损失
        • CVaR: 极端损失平均值
        • 差值: 尾部风险程度"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_var_comparison(self, ax, results):
        """
                绘制不同方法的VaR比较

                分析目的:
                - 比较历史法和参数法的差异
                - 评估参数法对极端风险的估计偏差
                """
        confidence_levels = [0.95, 0.99]
        methods = ['historical', 'parametric']
        method_names = ['历史模拟法', '参数法']
        colors = ['#ff6b6b', '#4ecdc4']

        bar_width = 0.35
        x_pos = np.arange(len(confidence_levels))

        for i, method in enumerate(methods):
            var_values = [results[conf][method]['var_pct'] for conf in confidence_levels]
            bars = ax.bar(x_pos + i * bar_width, var_values, bar_width,
                          label=method_names[i], color=colors[i], alpha=0.7)
            # 在柱子上添加数值
            for bar, value in zip(bars, var_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.2f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_title('不同计算方法的VaR比较', fontweight='bold', fontsize=14)
        ax.set_xlabel('置信水平', fontsize=12)
        ax.set_ylabel('VaR (%)', fontsize=12)
        ax.set_xticks(x_pos + bar_width/2)
        ax.set_xticklabels([f'{conf*100}%' for conf in confidence_levels], fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加图表说明
        ax.text(0.02, 0.98, 'VaR: 在给定置信水平下的最大可能损失',
                transform = ax.transAxes, verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    def _plot_var_cvar_comparison(self, ax, results):
        """
                绘制VaR和CVaR的对比

                分析目的:
                - 理解VaR和CVaR的关系
                - 评估尾部风险的严重程度
                """
        confidence_levels = [0.95, 0.99]
        methods = ['historical', 'parametric']
        method_name = ['历史法', '参数法']
        markers = ['o', 's']
        colors= ['#ff6b6b', '#4ecdc4']

        for i, method in enumerate(methods):
            var_values = []
            cvar_values = []
            for conf in confidence_levels:
                var_values.append(results[conf][method]['var_pct'])
                cvar_values.append(results[conf][method]['cvar_pct'])

            ax.plot(var_values, cvar_values, marker=markers[i],
                    markersize=10, linewidth=2, label=method_name[i],
                    color=colors[i])
            # 添加数据点标注
            for j, (var, cvar) in enumerate(zip(var_values, cvar_values)):
                ax.annotate(f'{confidence_levels[j]*100}%', (var, cvar),
                            xytext=(8,8), textcoords='offset points',
                            fontsize=10, fontweight='bold')
        ax.set_title('VaR vs CVaR 关系对比', fontweight='bold', fontsize=14)
        ax.set_xlabel('VaR (%)', fontsize=12)
        ax.set_ylabel('CVaR (%)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        # 添加参考线 - VaR=CVaR的理想情况
        min_val = min(min(var_values), min(cvar_values))
        max_val = max(max(var_values), max(cvar_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--',
                alpha=0.5, label='参考线')
        # 添加图表说明
        explanation_text = """CVaR (条件风险价值):
        • 衡量超过VaR的平均损失
        • 反映尾部风险
        • 通常 > VaR"""
        ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    def _plot_risk_contribution(self, ax):
        """
               绘制各股票的风险贡献饼图

               风险贡献计算原理:
               风险贡献 = 权重 × 波动率
               这反映了各股票对组合总体风险的贡献程度
               """
        if not self.stock_stats:
            # 如果没有数据，显示提示信息
            ax.text(0.5,0.5, '无可用数据', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.set_title('风险贡献分析', fontweight='bold', fontsize=14)
            return

        stocks = list(self.stock_stats.keys())
        weights = [self.portfolio[stock] / self.total_value for stock in stocks]
        volatilities= [self.stock_stats[stock]['volatility']*100 for stock in stocks]

        # 计算风险贡献（权重 × 波动率）
        risk_contributions = [w * v for w, v in zip(weights, volatilities)]
        total_risk = sum(risk_contributions)
        risk_percentages = [r/total_risk * 100 for r in risk_contributions]

        # 创建数据框便于排序
        risk_df = pd.DataFrame({
            'Stock': stocks,
            'Weight': weights,
            'Volatility': volatilities,
            'Risk_Contribution': risk_contributions,
            'Risk_Percentage': risk_percentages
        })

        # 按风险贡献排序 - 识别主要风险来源
        risk_df = risk_df.sort_values('Risk_Contribution', ascending=False)

        # 只显示前8个主要贡献者，其余合并为"其他"
        # 避免饼图过于碎片化，提高可读性
        if len(risk_df) > 8:
            top_8 = risk_df.head(8)
            other_risk = risk_df.iloc[8:]['Risk_Contribution'].sum()
            other_percentage = risk_df.iloc[8:]['Risk_Percentage'].sum()

            display_df = pd.concat([
                top_8,
                pd.DataFrame({
                    'Stock': ['其它'],
                    'Weight': [risk_df.iloc[8:]['Weight'].sum()],
                    'Volatility': [0],
                    'Risk_Contribution': [other_risk],
                    'Risk_Percentage': [other_percentage]
                })
            ])
        else:
            display_df = risk_df

        # 设置颜色 - 使用Set3色系，区分度好
        colors = plt.cm.Set3(np.linspace(0, 1, len(display_df)))

        # 绘制饼图
        wedges, texts, autotexts = ax.pie(display_df['Risk_Contribution'],
                                          labels=display_df['Stock'],
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          textprops={'fontsize': 9})

        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('各股票风险贡献度分析', fontweight='bold', fontsize=14)

        # 添加图例说明
        legend_text = f"""风险贡献度计算:
权重 × 个体波动率
总风险: {total_risk:.2f}%"""

        if len(risk_df) >= 3:
            legend_text += f"""
前3大风险来源:
1. {risk_df.iloc[0]['Stock']}: {risk_df.iloc[0]['Risk_Percentage']:.1f}%
2. {risk_df.iloc[1]['Stock']}: {risk_df.iloc[1]['Risk_Percentage']:.1f}%
3. {risk_df.iloc[2]['Stock']}: {risk_df.iloc[2]['Risk_Percentage']:.1f}%"""

        ax.text(-1.5, -1.2, legend_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    def generate_detailed_report(self, results, portfolio_returns):
        """
                生成详细风险报告

                报告设计理念:
                - 结构化展示关键信息
                - 提供业务解释和建议
                - 便于决策者理解和使用
                """
        print("\n" + "=" * 70)
        print("📋 VaR和CVaR风险分析详细报告 - 基于真实历史数据")
        print("=" * 70)

        # 基本统计信息
        print(f"\n📊 投资组合基本信息:")
        print(f"   组合总价值: ${self.total_value:,.2f}")
        print(f"   成功加载股票: {len(self.stock_stats)}/{len(self.portfolio)}只")
        print(f"   数据周期: {len(portfolio_returns)} 个交易日")
        print(f"   数据期间: {portfolio_returns.index[0].strftime('%Y-%m-%d')} 至 "
              f"{portfolio_returns.index[-1].strftime('%Y-%m-%d')}")
        print(f"   平均日收益率: {portfolio_returns.mean() * 100:.4f}%")
        print(f"   收益率波动率: {portfolio_returns.std() * 100:.4f}%")

        # 风险指标结果
        print(f"\n⚠️  VaR和CVaR风险指标:")
        for confidence in results.keys():
            print(f"\n   {confidence * 100}% 置信水平下的风险:")

            for method_name, method in [('历史模拟法', 'historical'),
                                      ('参数法', 'parametric')]:
                data = results[confidence][method]
                print(f"\n   {method_name}:")
                print(f"     VaR: {data['var_pct']:.2f}% (${data['var_amount']:,.2f})")
                print(f"     CVaR: {data['cvar_pct']:.2f}% (${data['cvar_amount']:,.2f})")
                print(f"     风险差额: {data['cvar_pct'] - data['var_pct']:.2f}%")

        # 风险解释和建议
        print(f"\n💡 风险解释与建议:")
        hist_var_95 = results[0.95]['historical']['var_amount']
        hist_cvar_95 = results[0.95]['historical']['cvar_amount']

        print(f"   1. 在95%置信水平下:")
        print(f"      • 明天最大可能损失不超过: ${hist_var_95:,.2f}")
        print(f"      • 如果发生极端损失，平均损失约为: ${hist_cvar_95:,.2f}")
        print(f"      • 建议保持 ${hist_cvar_95 * 1.5:,.2f} 的流动性缓冲")

        print(f"\n   2. 风险管理建议:")
        print(f"      • 定期监控VaR和CVaR指标")
        print(f"      • 建立风险预警机制")
        print(f"      • 考虑使用止损策略")
        print(f"      • 分散投资以降低极端风险")

        print("=" * 70)

def main():
    """
      主函数：运行VaR和CVaR分析

      执行流程:
      1. 定义投资组合
      2. 创建计算器实例
      3. 加载数据
      4. 计算风险指标
      5. 生成可视化
      6. 输出报告
      """
    # 定义投资组合
    portfolio = {
        'KO': 150,  # 可口可乐 - 消费股
        'SCHD': 150,  # 红利ETF
        'VOO': 150,  # S&P500 ETF
        'LLY': 120,  # 礼来制药 - 医药股
        'GLD': 100,  # 黄金ETF
        'AAPL': 61,  # 苹果 - 科技股
        'NBIS': 50,  # 其他股票
        'AA': 40,  # 美国铝业 - 工业股
        'UNH': 40,  # 联合健康 - 医药股
        'SBUX': 40,  # 星巴克 - 消费股
        'GOOGL': 30,  # 谷歌 - 科技股
        'LCID': 30,  # Lucid汽车 - 汽车股
        'META': 23,  # Meta - 科技股
        'AZTA': 10,  # 其他股票
        'ALMS': 10  # 其他股票
    }

    # 创建风险计算器实例
    calculator = VARCVARCalculator(portfolio)

    # 加载股票数据
    calculator.load_stock_data()

    # 计算风险指标
    results, portfolio_returns = calculator.calculate_risk_metrics()

    # 生成图表
    calculator.plot_comprehensive_analysis(results, portfolio_returns)

    # 生成详细报告
    calculator.generate_detailed_report(results, portfolio_returns)

# 程序入口点
if __name__ == "__main__":
    main()



'''
VaR和CVaR风险分析学习总结
🎯 核心概念理解
1. VaR (风险价值)
定义: 在给定置信水平和时间范围内，投资组合的最大可能损失
计算公式: P(损失 > VaR) = 1 - 置信水平
业务意义: "在95%的情况下，明天我的损失不会超过X元"

2. CVaR (条件风险价值)
定义: 当损失超过VaR时，这些极端损失的平均值
业务意义: "在最坏的5%情况下，平均会损失Y元"
优势: 比VaR更能反映尾部风险

两个重要概念
VaR（风险价值）：95%的情况下，最大亏损不会超过这个数
CVaR（条件风险价值）：在那些最坏的5%情况里，平均会亏多少钱



'''




