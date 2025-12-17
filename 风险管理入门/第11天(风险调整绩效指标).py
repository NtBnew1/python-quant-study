'''
第11天：
学习风险调整后的绩效指标，如Sortino比率、信息比率等。
练习：计算各指标，丰富投资组合绩效分析报告。
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

## 设置中文字体 - 确保图表能正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedPortfolioAnalyzer:
    def __init__(self, portfolio, window_size=100):
        """
               初始化高级投资组合分析器
               参数:
               portfolio -- 股票字典 {代码: 金额}
               window_size -- 滚动窗口大小，默认100个交易日
               功能说明:
               - 存储投资组合基本信息
               - 初始化数据存储字典
               - 设置分析参数
               """
        # 投资组合基本信息
        self.portfolio = portfolio                  # 股票字典 {代码: 金额}
        self.total_value = sum(portfolio.values())  # 总投资金额
        self.window_size = window_size              # 滚动窗口大小

        # 数据存储字典
        self.stock_data = {}        # 存储原始价格数据 {股票: 价格序列}
        self.returns_data = {}      # 存储收益率数据 {股票: 收益率序列}
        self.stock_stats = {}        # 存储统计指标 {股票: 统计字典}

        print("🔄 高级投资组合分析器初始化...")
        print(f"窗口大小: {window_size}个交易日")
        print(f"投资组合总价值: ${self.total_value:,.2f}")

    def load_stock_data(self):
        """
                加载股票数据并计算真实统计指标

                功能说明:
                - 从本地Excel文件读取股票数据
                - 计算每只股票的收益率和基本统计
                - 过滤数据量不足的股票
                - 返回成功加载的股票列表

                文件格式要求:
                - 文件路径: ./{股票代码}_stock_data.xlsx
                - 第一列为日期索引
                - 包含价格列（Close, close, Adj Close, Price, price等）
                """
        print("\n📊 加载股票数据...")
        available_stocks = []   # 成功加载的股票列表
        min_required_days = self.window_size + 50       # 需要比窗口多50天数据
        for stock in self.portfolio.keys():
            try:
                # 读取Excel文件，第一列为日期索引
                file_path = f"./{stock}_stock_data.xlsx"
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                # 寻找价格列（支持多种列名格式
                price_columns = ['close', 'Close', 'Adj Close', 'Price', 'price']
                price_col = next((col for col in price_columns if col in df.columns), None)

                if price_col:
                    prices = df[price_col].dropna()  # 清理缺失值
                    # 检查数据量是否足够进行滚动计算
                    if len(prices) >= min_required_days:
                        self.stock_data[stock] = prices

                        # 计算日收益率：(今日价格-昨日价格)/昨日价格
                        returns = prices.pct_change().dropna()
                        self.returns_data[stock] = returns
                        # 基于真实数据计算统计指标
                        daily_return = returns.mean()       # 真实日均收益率
                        volatility = returns.std()           # 真实日波动率
                        annual_return = daily_return * 252   # 年化收益率
                        annual_volatility = volatility * np.sqrt(252)     # 年化波动率

                        # 存储股票的详细统计信息
                        self.stock_stats[stock] = {
                            'daily_return': daily_return,
                            'volatility': volatility,
                            'annual_return': annual_return,
                            'annual_volatility': annual_volatility,
                            'data_points': len(prices)
                        }
                        available_stocks.append(stock)
                        print(f"✅ {stock}: {len(prices)}天数据")
                    else:
                        print(f"⚠️  {stock}: 数据不足 ({len(prices)}天)，跳过")
                else:
                    raise ValueError("未找到价格列")

            except Exception as e:
                print(f"❌ {stock}: 数据加载失败 - {e}")
                continue        # 跳过这只股票，继续处理其他
        print(f"\n📋 成功加载 {len(available_stocks)} 只股票数据")
        return available_stocks

    def calculate_portfolio_returns(self):
        """
                计算投资组合的日收益率

                功能说明:
                - 找到所有股票共同的交易日期（确保时间对齐）
                - 按投资金额计算每只股票的权重
                - 计算加权平均的组合日收益率
                - 输出组合构成和基本统计

                数学公式:
                组合收益率 = Σ(单股票收益率 × 该股票权重)
                权重 = 单股票金额 / 组合总金额

                返回:
                portfolio_returns -- 投资组合日收益率的时间序列
        """
        if not self.returns_data:
            raise ValueError("没有可用的股票数据")
        print("\n💰 计算投资组合收益率...")
        # 找到所有股票共同的交易日期（确保数据时间对齐）
        common_dates = None
        for returns in self.returns_data.values():
            if common_dates is None:
                common_dates = returns.index        # 第一个股票的日期
            else:
                common_dates = common_dates.intersection(returns.index)  # 取交集
        print(f"   共同日期范围: {len(common_dates)}天")

        # 重新计算权重（只包括成功加载的股票）
        available_stocks = list(self.returns_data.keys())
        available_value = sum(self.portfolio[stock] for stock in available_stocks)

        # 初始化投资组合收益率序列（全零序列）
        portfolio_returns = pd.Series(0.0, index=common_dates)
        print(f"\n   投资组合构成:")

        # 按权重加权计算组合收益率
        for stock in available_stocks:
            weight = self.portfolio[stock] / available_value      # 计算股票权重
            aligned_returns = self.returns_data[stock].loc[common_dates]    # 对齐日期
            portfolio_returns += aligned_returns * weight        # 加权累加
            print(f"   {stock}: {weight:.1%}")  # 输出每只股票的权重

        # 计算投资组合的真实统计
        portfolio_daily_return = portfolio_returns.mean()
        portfolio_volatility = portfolio_returns.std()

        print(f"\n📊 投资组合真实统计:")
        print(f"   日收益率: {portfolio_daily_return * 100:+.4f}%")
        print(f"   日波动率: {portfolio_volatility * 100:.4f}%")
        print(f"   年化收益率: {portfolio_daily_return * 252 * 100:.2f}%")
        print(f"   年化波动率: {portfolio_volatility * np.sqrt(252) * 100:.2f}%")
        return portfolio_returns

    def calculate_risk_adjusted_metrics(self, portfolio_returns, benchmark_returns = None, risk_free_rate=0.02):
        """
        计算风险调整后的绩效指标
            参数:
                portfolio_returns -- 投资组合日收益率序列
                benchmark_returns -- 基准收益率序列（可选）
                risk_free_rate -- 年化无风险利率，默认2%
            功能说明:
                - 计算夏普比率、索提诺比率等核心指标
                - 分析收益分布特征（偏度、峰度）
                - 计算风险价值（VaR、CVaR）
                - 返回包含所有指标的字典
            重要概念:
                - 夏普比率：总风险调整后收益
                - 索提诺比率：只考虑下行风险
                - 卡玛比率：基于最大回撤的风险调整
                - 信息比率：主动管理能力评估
        """
        print("\n📈 计算风险调整后绩效指标...")
        # 年化无风险利率转换
        annual_rf = risk_free_rate   # 年化无风险利率，如2%
        daily_rf = annual_rf / 252   # 转换为日无风险利率

        # 投资组合年化收益率和波动率
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)

        # 超额收益率（组合收益 - 无风险收益）
        excess_returns = portfolio_returns - daily_rf

        # 下行收益率（只考虑负收益部分）
        downside_returns = portfolio_returns.copy()
        downside_returns[downside_returns > 0] = 0      # 正收益设为0，只保留负收益

        # 初始化指标字典
        metrics = {}
        # ==================== 1. 夏普比率 ====================
        """
              夏普比率公式:
              夏普比率 = (年化收益率 - 无风险利率) / 年化波动率

              解释:
              - 衡量每单位总风险获得的超额收益
              - 数值越大，风险调整后收益越好
              - 行业标准：>1优秀，>0.5良好，>0合格
              """
        sharpe_ratio = (annual_return - annual_rf) / annual_volatility
        metrics['夏普比率'] = sharpe_ratio

        # ==================== 2. 索提诺比率 ====================
        """
        索提诺比率公式:
        索提诺比率 = (年化收益率 - 无风险利率) / 年化下行波动率

        解释:
        - 只考虑下行风险（损失风险），忽略上行波动
        - 对于厌恶损失的投资者更有意义
        - 通常比夏普比率更能反映真实风险调整收益
        """
        downside_volatility = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - annual_rf) / downside_volatility if downside_volatility > 0 else 0

        metrics['索提诺比率'] = sortino_ratio
        # ==================== 3. 卡玛比率 ====================
        """
        卡玛比率公式:
        卡玛比率 = (年化收益率 - 无风险利率) / 最大回撤

        解释:
        - 基于最大回撤的风险调整指标
        - 关注投资者可能承受的最大损失
        - 适合评估趋势跟踪策略
        """

        # 计算累计收益和最大回撤
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()    # 滚动最高点
        drawdown = (cumulative_returns - rolling_max) / rolling_max     # 回撤计算
        max_drawdown = drawdown.min()   # 最大回撤（最小值为最大损失）

        calmar_ratio = (annual_return - annual_rf) / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics['卡玛比率'] = calmar_ratio

        # ==================== 4. 特雷诺比率 ====================
        """
        特雷诺比率公式:
        特雷诺比率 = (年化收益率 - 无风险利率) / Beta

        解释:
        - 基于系统性风险（Beta）的调整
        - 这里简化假设Beta=1，实际应用中需要计算真实Beta
        """
        treynor_ratio = (annual_return - annual_rf) / 1.0   # Beta假设为1
        metrics['特雷诺比率'] = treynor_ratio

        # ==================== 5. 信息比率 ====================
        """
        信息比率公式:
        信息比率 = (组合年化收益 - 基准年化收益) / 跟踪误差

        解释:
        - 衡量主动管理的能力
        - 跟踪误差：组合与基准收益差的标准差
        - >0表示有超额收益，数值越大能力越强
        """
        if benchmark_returns is not None:
            # 对齐基准数据
            common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_aligned = portfolio_returns.loc[common_idx]
            benchmark_aligned = benchmark_returns.loc[common_idx]

            # 计算主动收益和跟踪误差
            active_returns = portfolio_aligned = benchmark_aligned
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = ((portfolio_aligned.mean() * 252 - benchmark_aligned.mean() * 252) /
                                 tracking_error)
            metrics['信息比率'] = information_ratio

        # ==================== 6. 欧米伽比率 ====================
        """
                欧米伽比率公式:
                欧米伽比率 = 超过阈值的收益总和 / 低于阈值的损失总和

                解释:
                - 考虑整个收益分布，不依赖正态分布假设
                - >1表示收益大于损失，数值越大越好
                """
        threshold = daily_rf     # 以无风险利率为阈值
        gains = portfolio_returns[portfolio_returns > threshold].sum()       # 超过阈值的收益
        losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())    # 低于阈值的损失
        omega_ratio = gains / losses if losses != 0 else float('inf')   # 避免除零
        metrics['欧米伽比率'] = omega_ratio

        # ==================== 7. 分布特征 ====================
        """
        偏度和峰度解释:
        - 偏度 > 0: 右偏，大涨概率高
        - 偏度 < 0: 左偏，大跌概率高  
        - 峰度 > 0: 尖峰厚尾，极端事件更多
        - 峰度 < 0: 低峰薄尾，分布更平缓
        """
        metrics['收益偏度'] = stats.skew(portfolio_returns)     # 分布对称性
        metrics['收益峰度'] = stats.kurtosis(portfolio_returns)  # 分布尖峭程度

        # ==================== 8. 基础统计和风险价值 ====================
        metrics['年化收益率'] = annual_return
        metrics['年化波动率'] = annual_volatility
        metrics['最大回撤'] = max_drawdown
        metrics['下行波动率'] = downside_volatility

        # VaR和CVaR计算（95%置信水平）
        metrics['VaR_95%'] = np.percentile(portfolio_returns, 5)     # 5%分位数
        metrics['CVaR_95%'] = portfolio_returns[portfolio_returns <= metrics['VaR_95%']].mean()  # 尾部平均损失
        print("✅ 风险指标计算完成")
        return metrics

    def plot_risk_metrics_comparison(self, metrics, benchmark_metrics=None):
        """
                绘制风险指标对比图 - 分成两个图表，每个图表2个子图

                参数:
                metrics -- 包含所有风险指标的字典
                benchmark_metrics -- 基准指标（可选）

                功能说明:
                - 第一个图表：核心比率 + 收益风险特征
                - 第二个图表：分布特征 + 风险价值
                - 每个子图都有详细的数值标签和说明
                - 使用协调的颜色方案提高可读性
                """
        print("\n🎨 绘制风险指标对比图...")
        # ==================== 第一个图表：核心绩效指标 ====================
        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig1.suptitle('投资组合核心绩效指标分析', fontsize=16, fontweight='bold')

        # 1. 主要比率对比 - 左子图
        """
        展示三大核心风险调整比率：
        - 夏普比率：总风险调整
        - 索提诺比率：下行风险调整  
        - 卡玛比率：回撤风险调整
        """
        ratio_metrics = ['夏普比率', '索提诺比率', '卡玛比率']
        ratio_values = [metrics.get(m, 0) for m in ratio_metrics]

        # 设置颜色方案
        colors = ['#2E86AB', '#A23B72', '#F18F01']  # 蓝色、紫色、橙色
        bars = ax1.bar(ratio_metrics, ratio_values, color=colors, alpha=0.8)
        ax1.set_title('风险调整收益比率', fontweight='bold', fontsize=14)
        ax1.set_ylabel('比率值', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 在柱子上添加数值标签
        for bar, value in zip(bars, ratio_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f"{value:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=11)
        # 添加比率说明文本框
        ax1.text(0.02, 0.98, '指标说明:\n• 夏普: 总风险调整\n• 索提诺: 下行风险调整\n• 卡玛: 回撤风险调整',
                 transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 2. 收益风险特征 - 右子图
        """
        展示四个关键风险收益指标（百分比形式）：
        - 年化收益率：投资回报
        - 年化波动率：总风险
        - 下行波动率：损失风险  
        - 最大回撤：最坏情况
        """
        risk_metrics = ['年化收益率', '年化波动率', '下行波动率', '最大回撤']
        risk_values = [metrics.get(m, 0) * 100 for m in risk_metrics]   # 转换为百分比

        colors_risk = ['#2E8B57', '#DC143C', '#FF8C00', '#8B008B']      # 绿色、红色、橙色、紫色
        bars2 = ax2.bar(risk_metrics, risk_values, color=colors_risk, alpha=0.8)
        ax2.set_title('收益与风险特征 (%)', fontweight='bold', fontsize=14)
        ax2.set_ylabel('百分比 (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)    # x轴标签旋转45度避免重叠
        ax2.grid(True, alpha=0.3)
        # 在柱子上添加百分比数值
        for bar, value in zip(bars2, risk_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.show()

        # ==================== 第二个图表：风险特征分析 ====================
        fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(16,6))
        fig2.suptitle('投资组合风险特征分析', fontsize=16, fontweight='bold')

        # 3. 分布特征 - 左子图
        """
        展示收益分布的统计特征：
        - 收益偏度：分布对称性
        - 收益峰度：尾部厚度
        """
        dist_metrics = ['收益偏度', '收益峰度']
        dist_values = [metrics.get(m, 0) for m in dist_metrics]
        colors_dist = ['#1E90FF', '#00CED1']    # 蓝色、青色
        bars3 = ax3.bar(dist_metrics, dist_values, color=colors_dist, alpha=0.8)
        ax3.set_title('收益分布特征', fontweight='bold', fontsize=14)
        ax3.set_ylabel('统计量', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 在柱子上添加数值，根据正负调整位置
        for bar, value in zip(bars3, dist_values):
            height = bar.get_height()
            va_position = 'bottom' if value >= 0 else 'top' # 正数在顶部，负数在底部
            offset = 0.01 if value >= 0 else -0.01
            ax3.text(bar.get_x() + bar.get_width()/2., height+offset,
                     f'{value:.3f}', ha='center', va=va_position,
                     fontweight='bold', fontsize=11)

        # 添加分布特征的专业解读
        skewness = metrics['收益偏度']
        kurtosis = metrics['收益峰度']
        skew_text = "右偏" if skewness > 0 else "左偏" if skewness < 0 else "对称"
        kurt_text = "尖峰厚尾" if kurtosis > 0 else "低峰薄尾" if kurtosis < 0 else "正态分布"
        ax3.text(0.02, 0.98, f'分布分析:\n偏度: {skew_text}\n峰度: {kurt_text}',
                 transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        # 4. 风险价值指标 - 右子图
        """
        展示尾部风险指标：
        - VaR (95%)：95%置信水平下的最大可能损失
        - CVaR (95%)：超过VaR的平均损失（预期短缺）
        """
        var_metrics = ['VaR_95%', 'CVaR_95%']
        var_value = [metrics.get(m, 0) * 100 for m in var_metrics]  # 转换为百分比
        colors_var = ['#B22222', '#FF4500']     # 深红色、橙红色
        bars4 = ax4.bar(var_metrics, var_value, color=colors_var, alpha=0.8)

        ax4.set_title('风险价值指标 (%)', fontweight='bold', fontsize=14)
        ax4.set_ylabel('损失百分比 (%)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        # 在柱子上添加百分比数值
        for bar, value in zip(bars4, var_value):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{value:.3f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=11)

        # 添加VaR的专业说明
        ax4.text(0.02, 0.98, '指标说明:\n• VaR: 95%置信水平下\n  最大可能损失\n• CVaR: 超过VaR的\n  平均损失',
                 transform=ax4.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        plt.tight_layout()
        plt.show()

    def calculate_rolling_metrics(self, portfolio_returns, window=126):
        """
        计算滚动风险指标 - 观察指标随时间的变化

        参数:
            portfolio_returns -- 投资组合日收益率序列
            window -- 滚动窗口大小，默认126天（约半年）

        功能说明:
            - 在滚动窗口上计算风险指标
            - 观察指标的稳定性和趋势
            - 检测市场环境变化对指标的影响
        滚动计算逻辑:
        对于每个时间点t，使用[t-window, t-1]的数据计算指标
        这样可以观察指标如何随时间演变
        """
        print(f"\n🔄 计算滚动风险指标 (窗口: {window}天)...")
        # 初始化存储字典
        rolling_data = {}

        # 准备存储列表
        dates = []
        sharpe_rolling = []
        sortino_rolling = []
        volatility_rolling = []
        max_dd_rolling = []

        # 检查数据是否足够
        if len(portfolio_returns) <= window:
            print(f"⚠️  数据不足，需要至少{window + 1}个数据点")
            return pd.DataFrame()
        print(f"   开始滚动计算，共{len(portfolio_returns) - window}个数据点...")
        # 滚动计算：从第window天开始到最后一天
        for i in range(window, len(portfolio_returns)):
            # 获取当前窗口数据（过去window天的收益率）
            window_returns = portfolio_returns.iloc[i-window:i]
            current_date = portfolio_returns.index[i]

            # ==================== 计算滚动夏普比率 ====================
            """
            滚动夏普计算步骤:
            1. 计算窗口内年化收益率
            2. 计算窗口内年化波动率  
            3. 应用夏普比率公式
            """
            window_annual_return = window_returns.mean() * 252
            window_annual_vol = window_returns.std() * np.sqrt(252)
            sharpe = (window_annual_return - 0.02) / window_annual_vol if window_annual_vol > 0 else 0

            # ==================== 计算滚动索提诺比率 ====================
            """
            滚动索提诺计算步骤:
            1. 识别下行收益（负收益）
            2. 计算下行波动率
            3. 应用索提诺比率公式
            """
            downside_returns = window_returns.copy()
            downside_returns[downside_returns > 0] = 0  # 只保留负收益
            downside_vol = downside_returns.std() * np.sqrt(252)
            sortino = (window_annual_return - 0.02) / downside_vol if downside_vol > 0 else 0

            # ==================== 计算滚动最大回撤 ====================
            """
            滚动最大回撤计算步骤:
            1. 计算窗口内累计收益
            2. 计算滚动最高点
            3. 计算回撤序列
            4. 找到最大回撤值
            """
            window_cumulative = ( 1+ window_returns).cumprod()
            window_rolling_max = window_cumulative.expanding().max()
            window_drawdown = (window_cumulative - window_rolling_max) / window_rolling_max
            max_dd = window_drawdown.min()

            # 存储计算结果
            dates.append(current_date)
            sharpe_rolling.append(sharpe)
            sortino_rolling.append(sortino)
            volatility_rolling.append(window_annual_vol)
            max_dd_rolling.append(max_dd)

            # 进度显示（每100个点显示一次）
            if (i - window) % 100 == 0:
                print(f"   已完成 {i - window}/{len(portfolio_returns) - window} 个点")

        # 创建DataFrame存储所有滚动指标
        rolling_data = pd.DataFrame({
            '夏普比率': sharpe_rolling,
            '索提诺比率': sortino_rolling,
            '年化波动率': volatility_rolling,
            '最大回撤': max_dd_rolling
        })

        print(f"✅ 滚动计算完成: {len(rolling_data)}个数据点")

        # 输出滚动指标统计
        print(f"\n📊 滚动指标统计:")
        print(f"   夏普比率 - 均值: {rolling_data['夏普比率'].mean():.3f}, 标准差: {rolling_data['夏普比率'].std():.3f}")
        print(f"   索提诺比率 - 均值: {rolling_data['索提诺比率'].mean():.3f}, 标准差: {rolling_data['索提诺比率'].std():.3f}")
        print(f"   年化波动率 - 均值: {rolling_data['年化波动率'].mean() * 100:.2f}%")
        print(f"   最大回撤 - 均值: {rolling_data['最大回撤'].mean() * 100:.2f}%")

        return rolling_data

    def plot_rolling_metrics(self, rolling_metrics):
        """
        绘制滚动指标时间序列 - 分成两个图表，每个图表2个子图

        参数:
            rolling_metrics -- 包含滚动指标的DataFrame

        功能说明:
            - 第一个图表：风险调整比率趋势分析
            - 第二个图表：波动性和回撤分析
            - 每个图表都包含统计摘要和专业解读
        """
        if rolling_metrics.empty:
            print("❌ 没有滚动数据可绘制")
            return
        # ==================== 第一个图表：风险调整比率分析 ====================
        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig1.suptitle('滚动风险调整比率分析', fontsize=16, fontweight='bold')
        # 1. 夏普比率和索提诺比率趋势 - 左子图
        """
        展示两个核心比率的时间序列：
        - 夏普比率：蓝色线条，圆形标记
        - 索提诺比率：红色线条，方形标记
        - 观察两者的相对表现和趋势
        """
        ax1.plot(rolling_metrics.index, rolling_metrics['夏普比率'],
                 label='夏普比率', linewidth=2.5, color='blue',
                 marker='o', markersize=3, alpha=0.8)
        ax1.plot(rolling_metrics.index, rolling_metrics['索提诺比率'],
                 label='索提诺比率', linewidth=2.5, color='red',
                 marker='s', markersize=3, alpha=0.8)

        ax1.set_title('风险调整比率趋势', fontweight='bold', fontsize=14)
        ax1.set_ylabel('比率值', fontsize=12)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)

        # 添加统计信息文本框
        sharpe_mean = rolling_metrics['夏普比率'].mean()
        sharpe_std = rolling_metrics['夏普比率'].std()
        sortino_mean = rolling_metrics['索提诺比率'].mean()
        sortino_std = rolling_metrics['索提诺比率'].std()

        stats_text = (f'平均值:\n'
                     f'夏普: {sharpe_mean:.3f}\n'
                     f'索提诺: {sortino_mean:.3f}\n\n'
                     f'稳定性:\n'
                     f'夏普标准差: {sharpe_std:.3f}\n'
                     f'索提诺标准差: {sortino_std:.3f}')

        ax1.text(0.02,0.98, stats_text,
                 transform=ax1.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 2. 比率差异分析 - 右子图
        """
        展示索提诺比率与夏普比率的差异：
        - 正差异：索提诺 > 夏普，说明下行风险控制好
        - 负差异：索提诺 < 夏普，说明上行波动被惩罚
        - 零线：参考线，帮助判断差异方向
        """
        ratio_diff = rolling_metrics['索提诺比率'] - rolling_metrics['夏普比率']
        ax2.plot(rolling_metrics.index, ratio_diff, label='索提诺 - 夏普',
                 linewidth=2.5, color='green',
                 marker='^', markersize=3, alpha=0.8)
        # 添加零参考线
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.set_title('索提诺与夏普比率差异', fontweight='bold', fontsize=14)
        ax2.set_ylabel('差异值', fontsize=12)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)

        # 添加差异分析统计
        pos_diff_days = len(ratio_diff[ratio_diff > 0])  # 正差异天数
        total_days = len(ratio_diff)
        pos_ratio = pos_diff_days / total_days  # 正差异比例

        diff_stats = (f'差异分析:\n'
                     f'正差异天数: {pos_diff_days}/{total_days}\n'
                     f'占比: {pos_ratio:.1%}\n\n'
                     f'平均差异: {ratio_diff.mean():.3f}\n'
                     f'最大差异: {ratio_diff.max():.3f}')

        ax2.text(0.02, 0.98, diff_stats,
                 transform=ax2.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        plt.tight_layout()
        plt.show()

        # ==================== 第二个图表：波动性和回撤分析 ====================
        fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(16,6))
        fig2.suptitle('滚动风险指标分析', fontsize=16, fontweight='bold')

        # 3. 波动率分析 - 左子图
        """
        展示年化波动率的时间序列：
        - 观察市场波动性的变化
        - 识别高波动和低波动时期
        - 评估风险管理的有效性
        """
        ax3.plot(rolling_metrics.index, rolling_metrics['年化波动率'] * 100,
                 label='年化波动率', linewidth=2.5, color='purple',
                 marker='d', markersize=3, alpha=0.8)

        ax3.set_title('滚动年化波动率', fontweight='bold', fontsize=14)
        ax3.set_ylabel('波动率 (%)', fontsize=12)
        ax3.legend(fontsize=11, loc='best')
        ax3.grid(True, alpha=0.3)
        # 添加波动率统计信息
        vol_mean = rolling_metrics['年化波动率'].mean() * 100
        vol_std = rolling_metrics['年化波动率'].std() * 100
        vol_max = rolling_metrics['年化波动率'].max() * 100
        vol_min = rolling_metrics['年化波动率'].min() * 100

        vol_stats = (f'波动率统计:\n'
                    f'均值: {vol_mean:.1f}%\n'
                    f'标准差: {vol_std:.1f}%\n'
                    f'范围: {vol_min:.1f}% - {vol_max:.1f}%')

        ax3.text(0.02, 0.98, vol_stats,
                 transform=ax3.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

        # 4. 最大回撤分析 - 右子图
        """
        展示最大回撤的时间序列：
        - 观察投资组合的下跌风险
        - 识别压力测试时期
        - 评估风险承受能力
        """
        ax4.plot(rolling_metrics.index, rolling_metrics['最大回撤'] * 100,
                 label='最大回撤',  linewidth=2.5, color='orange',
                marker='*', markersize=4, alpha=0.8)

        ax4.set_title('滚动最大回撤', fontweight='bold', fontsize=14)
        ax4.set_ylabel('回撤 (%)', fontsize=12)
        ax4.legend(fontsize=11, loc='best')
        ax4.grid(True, alpha=0.3)

        # 添加回撤统计信息
        max_dd_mean = rolling_metrics['最大回撤'].mean() * 100
        max_dd_min = rolling_metrics['最大回撤'].min() * 100
        max_dd_std = rolling_metrics['最大回撤'].std() * 100

        dd_stats = (f'回撤统计:\n'
                   f'平均回撤: {max_dd_mean:.1f}%\n'
                   f'最差回撤: {max_dd_min:.1f}%\n'
                   f'回撤波动: {max_dd_std:.1f}%')

        ax4.text(0.02, 0.98, dd_stats,
                 transform=ax4.transAxes, verticalalignment='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def generate_performance_report(self, metrics, rolling_metrics):
        """
        生成详细的绩效分析报告
        参数:
            metrics -- 静态风险指标字典
            rolling_metrics -- 滚动指标DataFrame
        功能说明:
            - 分类展示所有绩效指标
            - 提供专业评估和建议
            - 包含滚动指标的统计分析
            - 输出易于理解的投资建议
        """
        print("\n" + "=" * 80)
        print("📊 投资组合绩效分析报告")
        print("=" * 80)

        # ==================== 基础绩效指标 ====================
        print(f"\n📈 基础绩效指标:")
        print(f"   年化收益率: {metrics['年化收益率'] * 100:+.2f}%")
        print(f"   年化波动率: {metrics['年化波动率'] * 100:.2f}%")
        print(f"   最大回撤: {metrics['最大回撤'] * 100:.2f}%")
        print(f"   下行波动率: {metrics['下行波动率'] * 100:.2f}%")

        # 计算收益风险比
        return_to_risk = abs(metrics['年化收益率']/metrics['年化波动率']) if metrics['年化波动率'] != 0 else 0
        print(f"   收益风险比: {return_to_risk:.2f}")

        # ==================== 风险调整后指标 ====================
        print(f"\n🎯 风险调整后指标:")
        print(f"   夏普比率: {metrics['夏普比率']:.3f}")
        print(f"   索提诺比率: {metrics['索提诺比率']:.3f}")
        print(f"   卡玛比率: {metrics['卡玛比率']:.3f}")
        print(f"   特雷诺比率: {metrics['特雷诺比率']:.3f}")
        print(f"   欧米伽比率: {metrics['欧米伽比率']:.3f}")

        # 如果有信息比率，也显示
        if '信息比率' in metrics:
            print(f"   信息比率: {metrics['信息比率']:.3f}")

        # ==================== 分布特征 ====================
        print(f"\n📊 分布特征:")
        print(f"   收益偏度: {metrics['收益偏度']:.3f}")
        print(f"   收益峰度: {metrics['收益峰度']:.3f}")
        print(f"   VaR (95%): {metrics['VaR_95%'] * 100:.2f}%")
        print(f"   CVaR (95%): {metrics['CVaR_95%'] * 100:.2f}%")

        # 偏度和峰度的专业解读
        skewness = metrics['收益偏度']
        kurtosis = metrics['收益峰度']
        if skewness > 0.5:
            skew_interpret = "显著右偏 - 大涨概率较高"
        elif skewness > 0.1:
            skew_interpret =  "轻微右偏"
        elif skewness < -0.5:
            skew_interpret = "显著左偏 - 大跌风险较高"
        elif skewness < -0.1:
            skew_interpret = "轻微左偏"
        else:
            skew_interpret = "基本对称"

        if kurtosis > 3:
            kurt_interpret = "尖峰厚尾 - 极端事件较多"
        elif kurtosis < 1:
            kurt_interpret = "低峰薄尾 - 分布较平缓"
        else:
            kurt_interpret = "接近正态分布"

        print(f"   分布解读: {skew_interpret}, {kurt_interpret}")

        # ==================== 绩效评估 ====================
        print(f"\n💡 绩效评估:")

        # 夏普比率评估
        sharpe = metrics['夏普比率']
        if sharpe > 1.0:
            sharpe_rating = "优秀"
            sharpe_color = "🟢"
        elif sharpe > 0.5:
            sharpe_rating = "良好"
            sharpe_color = "🟡"
        elif sharpe > 0:
            sharpe_rating = "一般"
            sharpe_color = "🟠"
        else:
            sharpe_rating = "较差"
            sharpe_color = "🔴"
        print(f"   夏普比率: {sharpe_color} {sharpe_rating} (当前: {sharpe:.3f})")

        # 索提诺比率评估
        sortino = metrics['索提诺比率']
        if sortino > sharpe * 1.2:
            sortino_comment = "下行风险控制优秀"
        elif sortino > sharpe:
            sortino_comment = "下行风险控制良好"
        elif sortino == sharpe:
            sortino_comment = "上下行风险相当"
        else:
            sortino_comment = "需关注下行风险控制"
        print(f"   索提诺比率: {sortino_comment} (当前: {sortino:.3f})")

        # 最大回撤评估
        max_dd = metrics['最大回撤']
        if max_dd > -0.10:  # 小于10%回撤
            dd_rating = "风险控制优秀"
            dd_color = "🟢"
        elif max_dd > -0.20:  # 10%-20%回撤
            dd_rating = "风险控制良好"
            dd_color = "🟡"
        elif max_dd > -0.35:  # 20%-35%回撤
            dd_rating = "风险控制一般"
            dd_color = "🟠"
        else:  # 大于35%回撤
            dd_rating = "风险控制需加强"
            dd_color = "🔴"
        print(f"   最大回撤: {dd_color} {dd_rating} (当前: {max_dd * 100:.1f}%)")

        # 卡玛比率评估
        calmar = metrics['卡玛比率']
        if calmar > 1.0:
            calmar_rating = "回撤补偿优秀"
        elif calmar > 0.5:
            calmar_rating = "回撤补偿良好"
        elif calmar > 0:
            calmar_rating = "回撤补偿一般"
        else:
            calmar_rating = "回撤补偿不足"
        print(f"   卡玛比率: {calmar_rating} (当前: {calmar:.3f})")

        # ==================== 滚动指标统计 ====================
        if not rolling_metrics.empty:
            print(f"\n🔄 滚动指标统计 (最近{len(rolling_metrics)}期):")
            print(f"   平均夏普比率: {rolling_metrics['夏普比率'].mean():.3f}")
            print(f"   平均索提诺比率: {rolling_metrics['索提诺比率'].mean():.3f}")
            print(f"   夏普比率稳定性: {rolling_metrics['夏普比率'].std():.3f}")
            print(f"   平均年化波动率: {rolling_metrics['年化波动率'].mean() * 100:.1f}%")
            print(f"   平均最大回撤: {rolling_metrics['最大回撤'].mean() * 100:.1f}%")

            # 趋势分析
            if len(rolling_metrics) >= 20:
                recent_sharpe = rolling_metrics['夏普比率'].iloc[-10:].mean()
                earlier_sharpe = rolling_metrics['夏普比率'].iloc[:10].mean()
                if recent_sharpe > earlier_sharpe * 1.1:
                    trend = "显著改善"
                    trend_color = "🟢"
                elif recent_sharpe > earlier_sharpe:
                    trend = "轻微改善"
                    trend_color = "🟡"
                elif recent_sharpe < earlier_sharpe * 0.9:
                    trend = "显著恶化"
                    trend_color = "🔴"
                else:
                    trend = "基本稳定"
                    trend_color = "⚪"

                print(f"   近期表现趋势: {trend_color} {trend}")
                print(f"   (前期: {earlier_sharpe:.3f}, 近期: {recent_sharpe:.3f})")

        # ==================== 投资建议 ====================
        print(f"\n🎯 投资建议:")

         # 基于夏普比率的建议
        if sharpe > 1.0:
            print("   • 当前策略表现优秀，可考虑维持或适度增加投资")
        elif sharpe > 0.5:
            print("   • 策略表现良好，建议持续监控并优化")
        elif sharpe > 0:
            print("   • 策略表现一般，建议分析改进空间")
        else:
            print("   • 策略需要重大调整，建议重新评估投资方法")

        # 基于索提诺比率的建议
        if sortino > sharpe * 1.2:
            print("   • 下行风险控制优秀，适合风险厌恶型投资者")
        elif sortino < sharpe:
            print("   • 需加强下行风险管理，考虑增加防御性资产")

        # 基于最大回撤的建议
        if max_dd < -0.35:
            print("   • 回撤过大，建议降低仓位或增加对冲策略")
        elif max_dd > -0.15:
            print("   • 回撤控制良好，风险承受能力适当")

        # 基于滚动稳定性的建议
        if not rolling_metrics.empty:
            sharpe_std = rolling_metrics['夏普比率'].std()
            if sharpe_std > 0.5:
                print("   • 策略表现不稳定，建议分析原因并调整")
            elif sharpe_std < 0.2:
                print("   • 策略表现稳定，可预测性较高")

        # 通用建议
        print("   • 建议定期（每月）回顾这些指标")
        print("   • 结合市场环境理解指标变化")
        print("   • 不同投资目标应关注不同指标")

        print("=" * 80)

def main():
    """
        主函数：执行完整的投资组合绩效分析流程

        功能说明:
        - 定义投资组合配置
        - 按顺序执行所有分析步骤
        - 处理可能的异常情况
        - 提供用户友好的输出
    分析流程:
        1. 初始化分析器
        2. 加载股票数据
        3. 计算组合收益率
        4. 计算风险指标
        5. 计算滚动指标
        6. 生成可视化图表
        7. 输出详细报告
    """

    # ==================== 1. 定义投资组合 ====================
    """
    投资组合配置说明:
    - 键: 股票代码
    - 值: 投资金额（美元）

    组合设计原则:
    - 分散化：不同行业、不同市值
    - 平衡性：成长股与价值股搭配
    - 流动性：选择交易活跃的股票
    """
    portfolio = {
        'KO': 150,  # 可口可乐 - 消费必需品，稳定收益
        'SCHD': 150,  # 红利ETF - 稳定股息收入
        'VOO': 150,  # S&P500 ETF - 市场基准
        'LLY': 120,  # 礼来制药 - 医药股，成长性
        'GLD': 100,  # 黄金ETF - 避险资产
        'AAPL': 61,  # 苹果 - 科技巨头
        'AA': 40,  # 美国铝业 - 工业周期股
        'UNH': 40,  # 联合健康 - 医疗保健
        'SBUX': 40,  # 星巴克 - 消费周期性
        'GOOGL': 30,  # 谷歌 - 科技成长股
        'META': 23,  # Meta - 科技社交媒体
    }
    print("🚀 开始执行投资组合绩效分析...")
    print("=" * 50)

    # ==================== 2. 创建分析器实例 ====================
    """
    分析器参数说明:
    portfolio: 投资组合配置
    window_size: 滚动窗口大小（100个交易日 ≈ 5个月）

    窗口大小选择:
    - 太短：噪声过多，不够稳定
    - 太长：反应迟钝，难以捕捉变化
    - 100天是经验上的平衡点
    """

    analyzer = AdvancedPortfolioAnalyzer(portfolio, window_size=100)
    try:
        # ==================== 3. 加载股票数据 ====================
        """
        数据加载检查:
        - 检查本地Excel文件是否存在
        - 验证数据量是否足够
        - 处理加载失败的股票
        """
        available_stocks = analyzer.load_stock_data()
        # 检查是否有足够股票数据进行分析
        if len(available_stocks) < 3:
            print("❌ 可用股票数量不足，无法进行可靠的分析")
            print("   建议检查数据文件或调整投资组合")
            return

        print(f"✅ 数据加载完成，共{len(available_stocks)}只股票可用于分析")

        # ==================== 4. 计算组合收益率 ====================
        """
        组合收益率计算:
        - 时间对齐：确保所有股票日期一致
        - 权重计算：按投资金额比例
        - 加权平均：计算每日组合收益
        """
        portfolio_returns = analyzer.calculate_portfolio_returns()

        # 检查收益率数据质量
        if len(portfolio_returns) < 200:
            print("⚠️  收益率数据量较少，分析结果可能不够稳定")
        else:
            print(f"✅ 收益率计算完成，共{len(portfolio_returns)}个交易日数据")

            # ==================== 5. 计算风险调整指标 ====================
        """
        风险指标计算:
        - 使用95%置信水平计算VaR/CVaR
        - 无风险利率默认2%（可调整）
        - 包含分布特征分析
        """
        metrics = analyzer.calculate_risk_adjusted_metrics(portfolio_returns)

        # ==================== 6. 计算滚动指标 ====================
        """
        滚动指标参数:
        window=126: 约半年交易日的滚动窗口
        为什么选择126天？
        - 足够长以平滑噪声
        - 足够短以捕捉市场变化
        - 行业标准实践
        """
        rolling_metrics = analyzer.calculate_rolling_metrics(portfolio_returns, window=252)

        # ==================== 7. 生成可视化图表 ====================
        """
        图表生成流程:
        - 静态指标对比图（2个图表）
        - 滚动指标趋势图（2个图表）
        - 所有图表自动保存显示
        """
        print("\n📊 开始生成分析图表...")
        analyzer.plot_risk_metrics_comparison(metrics)
        analyzer.plot_rolling_metrics(rolling_metrics)
        print("✅ 图表生成完成")

        # ==================== 8. 生成详细报告 ====================
        """
        报告内容:
        - 基础绩效指标
        - 风险调整指标
        - 分布特征分析
        - 投资建议
        """
        analyzer.generate_performance_report(metrics, rolling_metrics)

        # ==================== 9. 分析完成总结 ====================
        print("\n🎉 投资组合绩效分析完成！")
        print("=" * 50)
        print("\n📋 分析成果总结:")
        print("   ✅ 风险调整指标计算")
        print("   ✅ 动态滚动分析")
        print("   ✅ 专业可视化图表")
        print("   ✅ 详细投资建议")

        print("\n💡 后续行动建议:")
        print("   1. 重点关注夏普比率和最大回撤")
        print("   2. 定期（每月）重新运行分析")
        print("   3. 比较不同时期的指标变化")
        print("   4. 根据建议调整投资策略")

        print("\n🔍 深入学习方向:")
        print("   • 理解每个指标的业务含义")
        print("   • 分析指标间的相互关系")
        print("   • 跟踪指标随市场环境的变化")
        print("   • 优化投资组合配置")

        print("\n" + "=" * 50)

    except Exception as e:
        # ==================== 异常处理 ====================
        """
        异常处理策略:
        - 捕获所有可能的错误
        - 提供友好的错误信息
        - 给出具体的解决建议
        - 打印详细错误信息便于调试
        """
        print(f"\n❌ 程序执行过程中出现错误: {e}")
        print("\n🔧 可能的原因和解决方案:")
        print("   1. 数据文件不存在")
        print("      → 检查Excel文件路径和命名")
        print("   2. 数据格式不正确")
        print("      → 确保文件包含日期索引和价格列")
        print("   3. 数据量不足")
        print("      → 需要至少150个交易日数据")
        print("   4. 内存不足")
        print("      → 尝试减少股票数量或数据范围")

        print("\n📋 调试信息:")
        import traceback
        traceback.print_exc()    # 打印详细错误堆栈

# ==================== 程序入口点 ====================
if __name__ == "__main__":
    """
        程序入口点说明:
        - 当直接运行此文件时执行main函数
        - 如果被其他文件导入则不执行
        - 这是Python的标准做法

        使用方法:
        1. 确保所有股票数据文件在正确路径
        2. 直接运行此Python文件
        3. 查看输出结果和图表
        """
    print("🏦 高级投资组合绩效分析系统")
    print("版本: 1.0")
    print("功能: 风险调整指标计算 + 动态滚动分析")
    print("=" * 60)

    # 执行主函数
    main()

    print("\n🙏 感谢使用投资组合分析系统！")
    print("如有问题或建议，请随时反馈。")


'''
==========================总结=====================
1. 类设计结构
class AdvancedPortfolioAnalyzer:
    def __init__(self)          # 初始化数据存储结构
    def load_stock_data()       # 数据加载和预处理
    def calculate_portfolio_returns()  # 核心计算逻辑
    def calculate_risk_adjusted_metrics()  # 指标算法实现
    def plot_risk_metrics_comparison()    # 可视化绘制
    def calculate_rolling_metrics()       # 时间序列分析
    def generate_performance_report()     # 结果输出
    
2. 数据处理技术
    文件读取：使用pandas读取Excel，自动识别日期索引
    数据对齐：通过索引交集找到共同交易日期
    缺失值处理：dropna()清理无效数据
    数据验证：检查最小数据量要求

3. 核心算法实现
    # 收益率计算
        returns = prices.pct_change().dropna()

    # 年化转换  
        annual_return = daily_return * 252
        annual_volatility = volatility * np.sqrt(252)

    # 下行风险计算
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0

4. 指标计算技术
    夏普比率：超额收益/总波动率
    索提诺比率：超额收益/下行波动率
    最大回撤：(当前净值-历史最高)/历史最高
    滚动计算：使用expanding().max()和窗口切片

5. 可视化技术
# 多子图布局
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)
# 柱状图定制
bars = ax.bar(metrics, values, color=colors, alpha=0.8)
# 文本标注
ax.text(x, y, text, bbox=dict(boxstyle='round', facecolor='lightblue'))

6. 工程化特性
    异常处理：try-catch包装文件操作
    进度显示：循环中定期输出进度
    参数化配置：窗口大小、无风险利率可调
    内存管理：使用生成器和适当的数据结构
    
7. 代码质量特点
    模块化设计：每个方法职责单一
    详细注释：数学公式和业务逻辑说明
    错误处理：友好的用户提示
    可扩展性：易于添加新指标

8. 关键技术点
    pandas时间序列操作：索引对齐、滚动计算
    numpy数值计算：统计量、百分位数
    matplotlib高级绘图：多子图、自定义样式
    scipy统计函数：偏度、峰度计算

9. 性能优化方面
    向量化计算：避免循环，使用pandas操作
    数据复用：缓存中间结果
    批量处理：一次性计算所有指标
'''

'''
======================风险调整指标计算方法分析========================
1. 夏普比率 (Sharpe Ratio)
# 代码实现
annual_return = portfolio_returns.mean() * 252
annual_volatility = portfolio_returns.std() * np.sqrt(252)
sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
计算方法分析：
    分子：年化超额收益 = 年化收益率 - 年化无风险利率(2%)
    分母：年化波动率 = 日波动率 × √252
    时间转换：252个交易日年化
    假设：收益率服从正态分布

技术要点：
    使用.mean()和.std()计算均值和标准差
    年化因子：收益率用252，波动率用√252

2. 索提诺比率 (Sortino Ratio)
# 代码实现
downside_returns = portfolio_returns.copy()
downside_returns[downside_returns > 0] = 0  # 只保留负收益
downside_volatility = downside_returns.std() * np.sqrt(252)
sortino_ratio = (annual_return - risk_free_rate) / downside_volatility

计算方法分析：
    下行风险定义：只考虑负收益的波动率
    数据处理：将正收益设为0，保留负收益计算标准差
    优势：不过度惩罚上涨波动

3. 卡玛比率 (Calmar Ratio)
# 代码实现
cumulative_returns = (1 + portfolio_returns).cumprod()
rolling_max = cumulative_returns.expanding().max()
drawdown = (cumulative_returns - rolling_max) / rolling_max
max_drawdown = drawdown.min()
calmar_ratio = (annual_return - risk_free_rate) / abs(max_drawdown)

计算方法分析：
    最大回撤计算：
    计算累计收益：(1 + returns).cumprod()
    计算滚动最高点：expanding().max()
    计算回撤：(当前值-最高点)/最高点
    分母：取最大回撤的绝对值

技术难点：
    使用expanding().max()计算历史最高点
    回撤计算涉及时间序列操作

4. 特雷诺比率 (Treynor Ratio)
# 代码实现（简化版）
treynor_ratio = (annual_return - risk_free_rate) / 1.0  # Beta假设为1

当前实现问题：
❌ Beta硬编码为1，这是不准确的
✅ 正确方法应该：

# 需要基准数据计算Beta
covariance = portfolio_returns.cov(benchmark_returns)
benchmark_variance = benchmark_returns.var()
beta = covariance / benchmark_variance
treynor_ratio = (annual_return - risk_free_rate) / beta


5. 信息比率 (Information Ratio)
# 代码实现
common_idx = portfolio_returns.index.intersection(benchmark_returns.index)
portfolio_aligned = portfolio_returns.loc[common_idx]
benchmark_aligned = benchmark_returns.loc[common_idx]

active_returns = portfolio_aligned - benchmark_aligned
tracking_error = active_returns.std() * np.sqrt(252)
information_ratio = ((portfolio_aligned.mean() * 252 - benchmark_aligned.mean() * 252) / tracking_error)

计算方法分析：
    主动收益：组合收益 - 基准收益
    跟踪误差：主动收益的年化标准差
    数据对齐：确保时间索引一致

技术细节：
    使用索引交集.intersection()对齐数据
    跟踪误差计算需要年化

6. 欧米伽比率 (Omega Ratio)
# 代码实现
threshold = daily_risk_free_rate  # 无风险利率
gains = portfolio_returns[portfolio_returns > threshold].sum()
losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())
omega_ratio = gains / losses if losses != 0 else float('inf')

计算方法分析：
    收益部分：超过阈值的收益总和
    损失部分：低于阈值的损失总和（取绝对值）
    阈值：通常使用无风险利率

技术特点：
    不依赖正态分布假设
    考虑整个收益分布
    使用布尔索引进行条件筛选



'''













