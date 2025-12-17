'''
第10天：
实现滚动窗口VaR监控，跟踪风险指标的时间序列变化。
练习：绘制滚动VaR曲线，分析VaR突破情况，验证模型稳定性。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings

from empyrical import annual_volatility
from mpl_toolkits.mplot3d.proj3d import transform

warnings.filterwarnings('ignore')

# 设置中文字体 - 确保图表能正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class RollingVARMonitor:
    def __init__(self, portfolio, window_size=100):
        """
            初始化风险监控器
            """
        # 投资组合基本信息
        self.portfolio = portfolio                      # 股票字典 {代码: 金额}
        self.total_value =sum(portfolio.values())       # 总投资金额
        self.window_size = window_size                  # 滚动窗口大小（过去N天

        # 数据存储字典
        self.stock_data = {}        # 存储原始价格数据 {股票: 价格序列}
        self.returns_data = {}      # 存储收益率数据 {股票: 收益率序列}
        self.stock_stats = {}       # 存储统计指标 {股票: 统计字典}

        print("🔄 滚动窗口VaR监控器初始化...")
        print(f"窗口大小: {window_size}个交易日")  # 每天用过去N天数据计算VaR
        print(f"投资组合总价值: ${self.total_value:,.2f}")

        ''' window_size 100天是 用前100天来计算当天的VaR'''

    def load_stock_data(self):
        """
               加载股票数据并计算真实统计指标
               返回成功加载的股票列表
               """
        print("\n📊 加载股票数据...")
        available_stocks = []   # 成功加载的股票列表
        min_required_days = self.window_size + 50       # 需要比窗口多50天数据，确保计算可靠

        # 遍历投资组合中的每只股票
        for stock in self.portfolio.keys():
            try:
                # 读取Excel文件，第一列为日期索引
                file_path = f'./{stock}_stock_data.xlsx'
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                # 寻找价格列（支持多种列名格式
                price_columns = ['Close', 'close', 'Adj Close', 'Price', 'price']
                price_col = next((col for col in price_columns if col in df.columns), None)

                if price_col:
                    prices = df[price_col].dropna()     # 清理缺失值，确保数据质量
                    # 检查数据量是否足够进行滚动计算
                    if len(prices) >= min_required_days:
                        self.stock_data[stock] = prices # 存储原始价格数据

                        # 计算日收益率：(今日价格-昨日价格)/昨日价格
                        returns = prices.pct_change().dropna()
                        self.returns_data[stock] = returns

                        # 基于真实数据计算统计指标（无任何假设）
                        daily_return = returns.mean()           # 真实日均收益率
                        volatility = returns.std()              # 真实日波动率
                        annual_return = daily_return * 252      # 年化收益率 = 日收益 × 252个交易日
                        annual_volatility = volatility * np.sqrt(252)    # 年化波动率

                        # 存储股票的详细统计信息
                        self.stock_stats[stock] = {
                            'daily_return': daily_return,
                            'volatility': volatility,
                            'annual_return': annual_return,
                            'annual_volatility': annual_volatility,
                            'data_points': len(prices)      # 数据点数量
                        }

                        available_stocks.append(stock)  # 添加到成功列表
                        print(f"✅ {stock}: {len(prices)}天数据")
                    else:
                        print(f"⚠️  {stock}: 数据不足 ({len(prices)}天)，跳过")
                else:
                    raise ValueError("未找到价格列")

            except Exception as e:
                # 如果加载失败，跳过该股票继续处理其他
                print(f"❌ {stock}: 数据加载失败 - {e}")
                break

        print(f"\n📋 成功加载 {len(available_stocks)} 只股票数据")
        return available_stocks     # 返回成功加载的股票列表

    def calculate_returns(self, prices):
        """
                计算日收益率
                参数: prices - 价格序列
                返回: returns - 收益率序列
                """
        returns = prices.pct_change().dropna()  # 计算百分比变化，删除NaN值
        return returns

    def calculate_portfolio_returns(self):
        """
                计算投资组合的日收益率
                按权重加权计算整体组合收益
                """
        if not self.returns_data:
            raise ValueError("没有可用的股票数据")
        print("\n💰 计算投资组合收益率...")

        # 找到所有股票共同的交易日期（确保数据时间对齐
        common_dates = None
        for returns in self.returns_data.values():
            if common_dates is None:
                common_dates = returns.index         # 第一个股票的日期
            else:
                common_dates = common_dates.intersection(returns.index) # 取交集
        print(f"   共同日期范围: {len(common_dates)}天")

        # 重新计算权重（只包括成功加载的股票）
        available_stocks = list(self.returns_data.keys())
        available_value = sum(self.portfolio[stock] for stock in available_stocks)

        # 初始化投资组合收益率序列
        portfolio_returns = pd.Series(0.0, index=common_dates)

        print(f"\n   投资组合构成:")
        # 按权重加权计算组合收益率
        for stock in available_stocks:
            weight = self.portfolio[stock] / available_value          # 计算股票权重
            aligned_returns = self.returns_data[stock].loc[common_dates]    # 对齐日期
            portfolio_returns += aligned_returns * weight   # 加权累加
            print(f"   {stock}: {weight:.1%}")

        # 计算投资组合的真实统计
        portfolio_daily_return = portfolio_returns.mean()
        portfolio_volatility = portfolio_returns.std()
        print(f"\n📊 投资组合真实统计:")
        print(f"   日收益率: {portfolio_daily_return * 100:+.4f}%")
        print(f"   日波动率: {portfolio_volatility * 100:.4f}%")

        return portfolio_returns        # 返回组合收益率时间序列

    def calculate_rolling_var_cvar(self, portfolio_returns, confidence_level=0.95):
        """
                滚动计算每天的VaR和CVaR
                用过去window_size天的数据预测当天的风险
                """
        print(f"\n📈 计算滚动{confidence_level * 100}% VaR和CVaR...")
        returns_array = portfolio_returns.values    # 转换为数组便于计算
        dates = portfolio_returns.index              # 日期索引

        # 检查数据是否足够进行滚动计算
        if len(returns_array) <= self.window_size:
            raise ValueError(f"数据不足，需要至少{self.window_size + 1}个数据点")

        # 初始化结果存储列表
        historical_vars = []        # 存储历史模拟法VaR
        historical_cvars = []       # 存储历史模拟法CVaR
        parametric_vars = []        # 存储参数法VaR
        parametric_cvars = []       # 存储参数法CVaR
        actual_returns = []         # 存储当天实际收益率

        total_points = len(returns_array) - self.window_size    # 总计算点数
        print(f"   开始滚动计算，共{total_points}个数据点...")

        # 滚动计算：从第window_size+1天开始到最后一天
        for i in range(self.window_size, len(returns_array)):
            # 获取窗口数据：过去window_size天的收益率
            window_returns = returns_array[i-self.window_size:i]
            # 当天实际收益率（我们要预测的风险对应的实际值）
            current_return = returns_array[i]

            # ==================== 历史模拟法计算 ====================
            sorted_returns = np.sort(window_returns)        # 对窗口内收益率排序
            var_index = int((1-confidence_level) * len(sorted_returns))  # 计算分位数位置
            hist_var = sorted_returns[var_index]    # VaR = 排序后的分位数对应值

            # CVaR = 超过VaR的所有损失的平均值
            tail_returns = sorted_returns[:var_index]   # 所有小于VaR的收益率
            hist_cvar = np.mean(tail_returns) if len(tail_returns) > 0 else hist_var

            # ==================== 参数法计算（正态分布假设） ====================
            mean_return = np.mean(window_returns)   # 窗口内平均收益率
            std_return = np.std(window_returns)     # 窗口内收益率标准差
            z_score = stats.norm.ppf(1-confidence_level)     # 标准正态分布分位数
            param_var = mean_return + z_score * std_return  # VaR = 均值 + Z分数×标准差
            param_cvar = mean_return - (std_return * stats.norm.pdf(z_score) / (1-confidence_level))

            # 存储计算结果
            historical_vars.append(hist_var)
            historical_cvars.append(hist_cvar)
            parametric_vars.append(param_var)
            parametric_cvars.append(param_cvar)
            actual_returns.append(current_return)

        # 创建结果DataFrame，便于后续分析和绘图
        results_df = pd.DataFrame({
            'date': dates[self.window_size:],           # 日期（从第一个可计算日期开始）
            'actual_return': actual_returns,            # 当天实际收益率
            'historical_var': historical_vars,          # 历史法VaR预测
            'historical_cvar': historical_cvars,        # 历史法CVaR预测
            'parametric_var': parametric_vars,          # 参数法VaR预测
            'parametric_cvar': parametric_cvars          # 参数法CVaR预测
        })

        results_df.set_index('date', inplace=True)  # 设置日期为索引
        print(f"✅ 完成滚动计算: {len(results_df)}个数据点")
        return results_df

    def analyze_var_breaks(self, results_df, confidence_level=0.95):
        """
                分析VaR突破情况
                检查实际损失是否超过VaR预测
                """
        print(f"\n🔍 分析VaR突破情况...")
        break_analysis = {}     # 存储突破分析结果
        # 对两种计算方法分别分析
        for method in ['historical', 'parametric']:
            var_col = f'{method}_var'        # VaR列名
            actual_returns = results_df['actual_return']     # 实际收益率
            var_values = results_df[var_col]                 # VaR预测值

            # 识别突破点：实际损失超过VaR预测的情况
            breaks = actual_returns < var_values        # 布尔序列，True表示突破
            break_dates = results_df.index[breaks]      # 突破发生的日期
            break_returns = actual_returns[breaks]      # 突破时的实际收益
            break_var_values = var_values[breaks]       # 突破时的VaR预测值

            # 计算突破统计
            total_days = len(results_df)                            # 总观察天数
            break_days = len(break_dates)                           # 突破天数
            expected_breaks = (1-confidence_level) * total_days     # 理论预期突破次数

            # 突破严重程度 = VaR预测值 - 实际收益（正数表示突破程度）
            break_severity = break_var_values = break_returns

            # 存储该方法的突破分析结果
            break_analysis[method] = {
                'break_dates': break_dates,             # 突破日期
                'break_returns': break_returns,         # 突破时的实际收益
                'break_var_values': break_var_values,   # 突破时的VaR值
                'break_severity': break_severity,       # 突破严重程度
                'total_breaks': break_days,              # 总突破次数
                'break_ratio': break_days / total_days,      # 突破比例
                'expected_breaks': expected_breaks,     # 预期突破次数
                'avg_severity': break_severity.mean() if len(break_severity) > 0 else 0,    # 平均突破程度
                'max_severity': break_severity.max() if len(break_severity) > 0 else 0       # 最大突破程度
            }

            # 输出突破统计
            print(f"   {method}方法:")
            print(f"     实际突破: {break_days}次 (预期: {expected_breaks:.1f}次)")
            print(f"     突破比例: {break_days / total_days:.2%} (预期: {(1 - confidence_level):.2%})")
            if break_days > 0:
                print(f"     平均突破程度: {break_severity.mean() * 100:.3f}%")
                print(f"     最大突破程度: {break_severity.max() * 100:.3f}%")
        return break_analysis

    def plot_rolling_analysis(self, results_df, break_analysis, confidence_level=0.95):
        """
                绘制滚动分析图表
                分成两个图表，每个图表2个子图，避免过于拥挤
                """
        print("\n🎨 生成监控图表...")
        if len(results_df) == 0:
            print("❌ 没有数据可绘制")
            return

        # ==================== 第一个图表：收益率与突破分析 ====================
        fig1, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig1.suptitle(f'滚动VaR监控 - 收益率与突破分析 (置信水平: {confidence_level*100}%)',
                     fontsize=16, fontweight='bold')

        # 子图1：实际收益率与VaR对比
        self._plot_returns_vs_var(ax1, results_df, break_analysis, confidence_level)
        # 子图2：突破次数分析
        self._plot_break_analysis(ax2, break_analysis, confidence_level)

        plt.tight_layout()
        plt.show()

        # ==================== 第二个图表：风险序列与稳定性 ====================
        fig2, (ax3, ax4) = plt.subplots(1,2, figsize=(16,6))
        fig2.suptitle(f'滚动VaR监控 - 风险序列与稳定性 (置信水平: {confidence_level*100}%)',
                     fontsize=16, fontweight='bold')

        # 子图3：VaR时间序列
        self._plot_var_series(ax3, results_df, confidence_level)
        # 子图4：模型稳定性检验
        self._plot_stability(ax4, results_df, break_analysis, confidence_level)

        plt.tight_layout()
        plt.show()

    def _plot_returns_vs_var(self, ax, results_df, break_analysis, confidence_level):
        """
                绘制实际收益率与VaR风险边界的对比图
                显示每天的实际收益和两种VaR预测
                """
        dates = results_df.index
        actual_returns = results_df['actual_return'] * 100  # 转换为百分比便于阅读

        # 绘制实际收益率曲线（蓝色细线）
        ax.plot(dates, actual_returns, 'blue', alpha=0.7, linewidth=1, label='实际日收益率')

        # 两种VaR计算方法
        methods=['historical', 'parametric']
        colors=['red', 'orange']
        labels=['历史法VaR', '参数法VaR']

        # 绘制两种VaR方法的预测边界
        for i, method in enumerate(methods):
            var_values = results_df[f'{method}_var'] * 100      # VaR转换为百分比
            ax.plot(dates, var_values, color=colors[i], linewidth=2, label=labels[i], alpha=0.8)

            # 标记突破点：实际损失超过VaR预测的位置
            breaks = break_analysis[method]
            if len(breaks['break_dates']) > 0:
                ax.scatter(breaks['break_dates'],
                           breaks['break_returns'] * 100,       # 突破时的实际收益
                           color=colors[i], s=30, alpha=0.7,
                           label='')

        #手动添加突破点图例项，避免重复
        if any(len(break_analysis[method]['break_dates']) > 0 for method in methods):
            ax.scatter([], [], color='gray', s=30, alpha=0.7, label='突破点')

        ax.set_title('实际收益率 vs VaR风险边界', fontweight='bold', fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('收益率 (%)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # 添加统计信息文本框
        total_days = len(results_df)
        stats_text = f"""统计信息:
观察期: {total_days}天
平均收益: {actual_returns.mean():.3f}%
收益波动: {actual_returns.std():.3f}%
偏度: {stats.skew(actual_returns):.3f}"""  # 分布偏度

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _plot_break_analysis(self, ax, break_analysis, confidence_level):
        """
               绘制VaR突破次数对比柱状图
               比较实际突破次数与理论预期
               """
        methods = ['historical', 'parametric']
        method_names = ['历史模拟法', '参数法']
        colors = ['#ff6b6b', '#4ecdc4']

        # 准备柱状图数据
        actual_breaks = [break_analysis[method]['total_breaks'] for method in methods]
        expected_breaks = [break_analysis[method]['expected_breaks'] for method in methods]
        x_pos = np.arange(len(methods))      # 柱子位置
        bar_width = 0.35    # 柱子宽度

        # 绘制实际突破次数（左侧柱子）
        bars1 = ax.bar(x_pos - bar_width/2, actual_breaks, bar_width,
                       label='实际突破次数', color=colors[0], alpha=0.7)

        # 绘制预期突破次数（右侧柱子）
        bars2 = ax.bar(x_pos + bar_width/2, expected_breaks, bar_width,
                       label='预期突破次数', color=colors[1], alpha=0.7)

        # 在柱子上方添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height,
                        f"{height:.0f}",        # 使用:.0f格式化为整数
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_title('VaR突破次数对比', fontweight='bold', fontsize=14)
        ax.set_xlabel('计算方法', fontsize=12)
        ax.set_ylabel('突破次数', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')       # 只显示y轴网格

        # 在柱子顶部添加突破比例信息
        for i, method in enumerate(methods):
            ratio = break_analysis[method]['break_ratio']
            expected_ratio = 1 - confidence_level
            ax.text(x_pos[i], max(actual_breaks[i], expected_breaks[i]) * 1.1,
                     f'实际: {ratio:.2%}\n预期: {expected_ratio:.2%}',  # 修复字符串格式
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    def _plot_var_series(self, ax, results_df, confidence_level):
        """
                绘制VaR时间序列图
                显示两种方法VaR值随时间的变化趋势
                """
        dates = results_df.index

        # 两种VaR计算方法
        methods = ['historical', 'parametric']
        colors = ['red', 'blue']
        labels = ['历史法VaR', '参数法VaR']
        linestyles = ['-', '--']     # 使用不同线型区分方法

        # 绘制两种VaR方法的时间序列
        for i, method in enumerate(methods):
            var_values = results_df[f'{method}_var'] * 100      # VaR值
            cvar_values = results_df[f'{method}_cvar'] * 100     # CVaR值

            # 绘制VaR主线（较粗）
            ax.plot(dates, var_values, color=colors[i], linewidth=2,
                    label=labels[i], linestyle=linestyles[i])

            # 绘制CVaR辅助线（较细，半透明）
            ax.plot(dates, cvar_values, color=colors[i], linewidth=2,
                    label=labels[i], linestyle=linestyles[i])

        ax.set_title('滚动VaR和CVaR时间序列', fontweight='bold', fontsize=14)
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('风险值 (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 添加VaR统计信息
        var_stats = []
        for i, method in enumerate(methods):
            var_series = results_df[f'{method}_var'] * 100
            # 计算均值和标准差
            var_stats.append(f"{labels[i]}: {var_series.mean():.3f}% ± {var_series.std():.3f}%")

        stats_text = "VaR统计:\n" + "\n".join(var_stats)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment = 'top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def _plot_stability(self, ax, results_df, break_analysis, confidence_level):
        """
                绘制模型稳定性检验图
                评估VaR预测的稳定性和可靠性
                """
        methods = ['historical', 'parametric']
        method_names = ['历史法', '参数法']

        # 计算VaR变异性：标准差越小说明模型越稳定
        var_variability = []
        for method in methods:
            var_series = results_df[f'{method}_var'] * 100
            # 计算30日滚动标准差，再取平均作为稳定性指标
            rolling_std = var_series.rolling(window=30).std()
            var_variability.append(rolling_std.mean())

        x_pos = np.arange(len(methods))

        # 绘制VaR变异性柱状图
        bars = ax.bar(x_pos, var_variability, 0.6,
                      color=['lightcoral', 'lightgreen'], alpha=0.7)

        # 添加数值标签
        for bar, value, in zip(bars, var_variability):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f"{value:.3f}%", ha='center', va='bottom',
                    fontweight='bold', fontsize=10)

        ax.set_title('VaR变异性分析', fontweight='bold', fontsize=14)
        ax.set_xlabel('计算方法', fontsize=12)
        ax.set_ylabel('VaR标准差 (%)', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_names, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        # 添加稳定性说明
        ax.text(0.02, 0.98, "指标说明:\n• VaR变异性越小\n  模型越稳定\n• 稳定模型便于\n  风险管理",
                transform=ax.transAxes, verticalalignment='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    def generate_monitoring_report(self, results_df, break_analysis, confidence_level=0.95):
        """
                生成详细的VaR监控报告
                总结分析结果并提供风险管理建议
                """
        print("\n" + "=" * 70)
        print("📊 滚动VaR监控详细报告")
        print("=" * 70)

        total_days = len(results_df)    # 总观察天数
        expected_break_ratio =  1 - confidence_level        # 理论突破比例

        # ==================== 监控概况 ====================
        print(f"\n📈 监控概况:")

        # 确保日期按正确顺序显示（从早到晚）
        start_date = results_df.index.min()     # 最早日期
        end_date = results_df.index.max()       # 最晚日期

        print(f"   观察期间: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        print(f"   总观察天数: {total_days}天")
        print(f"   滚动窗口: {self.window_size}个交易日")
        print(f"   置信水平: {confidence_level * 100}%")

        # 计算投资组合整体统计
        portfolio_stats = self._calculate_portfolio_stats(results_df)
        print(f"   投资组合平均日收益: {portfolio_stats['mean_return'] * 100:+.4f}%")
        print(f"   投资组合日波动率: {portfolio_stats['volatility'] * 100:.4f}%")

        # ==================== VaR突破分析 ====================
        print(f"\n⚠️  VaR突破分析:")
        for method, method_name in [('historical', '历史模拟法'), ('parametric', '参数法')]:
            analysis = break_analysis[method]
            print(f"\n   {method_name}:")
            print(f"     实际突破次数: {analysis['total_breaks']}次")
            print(f"     实际突破比例: {analysis['break_ratio']:.2%}")
            print(f"     预期突破次数: {analysis['expected_breaks']:.1f}次 ({expected_break_ratio:.2%})")

            # 如果有突破，显示突破程度
            if analysis['total_breaks'] > 0:
                print(f"     平均突破程度: {analysis['avg_severity'] * 100:.3f}%")
                print(f"     最大突破程度: {analysis['max_severity'] * 100:.3f}%")

            # 模型评估：比较实际与预期突破比例
            deviation = abs(analysis['break_ratio'] - expected_break_ratio) / expected_break_ratio
            if deviation < 0.2:
                assessment =  "优秀"  # 偏差小于20%
                color = "🟢"
            elif deviation < 0.5:
                assessment = "良好"  # 偏差小于50%
                color = "🟡"     # yellow
            else:
                assessment = "需要改进"  # 偏差大于50%
                color = "🔴"
            print(f"     模型评估: {color} {assessment} (偏差: {deviation:.1%})")

        # ==================== 风险管理建议 ====================
        print(f"\n💡 风险管理建议:")

        # 基于突破分析给出具体建议
        hist_analysis = break_analysis['historical']
        param_analysis = break_analysis['parametric']

        # 历史法建议
        if hist_analysis['break_ratio'] > expected_break_ratio * 1.5:
            print("   • 历史法VaR可能低估风险，建议增加20-30%安全边际")
        elif hist_analysis['break_ratio'] < expected_break_ratio * 0.5:
            print("   • 历史法VaR可能过于保守，可考虑适当提高风险承受")
        else:
            print("   • 历史法VaR表现良好，可继续使用")

        # 参数法建议
        if param_analysis['break_ratio'] > expected_break_ratio * 1.5:
            print("   • 参数法受正态分布假设影响，可能低估尾部风险")
            print("   • 建议结合历史法或其他方法综合评估")
        elif param_analysis['break_ratio'] < expected_break_ratio * 0.5:
            print("   • 参数法可能过于保守，在市场平稳时可以使用")
        else:
            print("   • 参数法在当前市场环境下表现合理")

        # 极端风险建议
        if any(analysis['max_severity'] > 0.5 for analysis in break_analysis.values()):
            print("   • 存在严重突破事件（>5%），建议加强尾部风险管理")
            print("   • 考虑使用CVaR作为主要风险指标")

        # 通用建议
        print("   • 建议结合两种方法的结果进行综合判断")
        print("   • 定期（如每季度）重新评估和调整窗口大小")
        print("   • 关注VaR变异性的变化，及时调整风险管理策略")

        print("=" * 70)

    def _calculate_portfolio_stats(self, results_df):
        """
                计算投资组合的统计指标
                辅助函数，用于生成报告
                """
        actual_returns = results_df['actual_return']
        return {
            'mean_return': actual_returns.mean(),       # 平均收益率
            'volatility': actual_returns.std(),         # 收益波动率
            'total_return': actual_returns.sum(),       # 累计收益
            'min_return': actual_returns.min(),         # 最小收益（最大损失）
            'max_return': actual_returns.max()          # 最大收益
        }


def main():
    """
       主函数：程序入口点
       按顺序执行整个VaR监控流程
       """
    # ==================== 1. 定义投资组合 ====================
    portfolio = {
        'KO': 150,  # 可口可乐 - 消费股
        'SCHD': 150,  # 红利ETF - 稳定收益
        'VOO': 150,  # S&P500 ETF - 市场基准
        'LLY': 120,  # 礼来制药 - 医药股
        'GLD': 100,  # 黄金ETF - 避险资产
        'AAPL': 61,  # 苹果 - 科技股
        'AA': 40,  # 美国铝业 - 工业股
        'UNH': 40,  # 联合健康 - 医药股
        'SBUX': 40,  # 星巴克 - 消费股
        'GOOGL': 30,  # 谷歌 - 科技股
        'META': 23,  # Meta - 科技股
    }

    # ==================== 2. 创建监控器 ====================
    print("开始执行滚动VaR监控分析...")
    monitor = RollingVARMonitor(portfolio, window_size=100)  # 使用100天窗口

    try:
        # ==================== 3. 加载股票数据 ====================
        available_stocks = monitor.load_stock_data()

        # 检查是否有足够股票数据
        if len(available_stocks) < 3:
            print("❌ 可用股票数量不足，无法进行可靠的计算")
            return

        # ==================== 4. 计算组合收益率 ====================
        portfolio_returns = monitor.calculate_portfolio_returns()

        # ==================== 5. 设置置信水平 ====================
        confidence_level = 0.95 # 95%置信水平

        # ==================== 6. 滚动计算VaR和CVaR ====================
        results_df = monitor.calculate_rolling_var_cvar(portfolio_returns, confidence_level)

        # ==================== 7. 分析VaR突破情况 ====================
        break_analysis = monitor.analyze_var_breaks(results_df, confidence_level)

        # ==================== 8. 绘制分析图表 ====================
        monitor.plot_rolling_analysis(results_df, break_analysis, confidence_level)

        # ==================== 9. 生成详细报告 ====================
        monitor.generate_monitoring_report(results_df, break_analysis, confidence_level)

        print("\n🎉 VaR监控分析完成！")
        print("您现在可以：")
        print("• 查看风险随时间的变化趋势")
        print("• 评估VaR模型的准确性")
        print("• 根据报告建议调整风险管理策略")

    except Exception as e:
        # 异常处理：显示错误信息
        print(f"\n❌ 程序执行过程中出现错误: {e}")
        print("可能的原因：")
        print("• 股票数据文件不存在或格式不正确")
        print("• 数据量不足进行滚动计算")
        print("• 内存不足或文件权限问题")
        import traceback
        traceback.print_exc()   # 打印详细错误信息

# 程序入口
if __name__ == "__main__":
    """
        程序启动点
        当直接运行此文件时执行main函数
        """
    main()







