'''
第8天：
开展压力测试，模拟极端市场事件对投资组合的冲击。
练习：设计“黑天鹅”事件，输出风险报告并分析回撤情况。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')   #  # 忽略警告信息，让输出更整洁

# 设置中文字体，确保图表能正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class OptimizedBlackSwanTester:
    """
    优化版黑天鹅压力测试器
    功能：模拟极端市场事件对投资组合的影响，并生成详细的可视化分析
    """
    def __init__(self):
        """
               初始化压力测试器
               设置投资组合数据和基本参数
               """
        # 定义投资组合：股票代码 -> 投资金额
        self.portfolio = {
            'AA': 40,
            'LLY': 120,
            'NVO': 50,
            'GLD': 100,
            'MU': 30,
            'VOO': 150,
            'SCHD': 150,
            'KO': 150,
            'AAPL': 61,
            'META': 23
        }
        # 计算投资组合总价值
        self.total_value = sum(self.portfolio.values())
        self.results = {}           # 存储测试结果的字典
        self.stock_data = {}        # 存储股票数据的字典

        # 打印投资组合分析信息
        print("💰 投资组合分析:")
        print("=" * 50)  # 打印分隔线
        # 遍历投资组合，打印每只股票的信息
        for stock, value in self.portfolio.items():
            # 打印股票代码、金额和占比
            print(f"{stock}: ${value}({value/self.total_value:.1%})")
        print(f"总投资: ${self.total_value}")  # 打印总投资金额
        print("=" * 50)  # 打印分隔线
        # 加载所有股票数据
        self.load_all_stock_data()

    def load_all_stock_data(self):
        """加载所有股票数据到内存中"""
        print("\n📊 加载股票数据...")
        # 遍历投资组合中的每只股票
        for stock in self.portfolio.keys():
            file_path = f'./{stock}_stock_data.xlsx'    # 构建文件路径
            # 检查文件是否存在
            if os.path.exists(file_path):
                try:
                    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index) # 确保索引是日期时间格式
                    df = df.sort_index()    # 按日期排序
                    # 定义可能的价格列名称
                    price_columns = ['Close', 'close', 'Adj Close', 'Price', 'price']
                    # 寻找实际存在的价格列
                    price_col = next((col for col in price_columns if col in df.columns), None)
                    # 如果没找到标准列名但只有一列，假设该列就是价格
                    if price_col is None and len(df.columns) == 1:
                        price_col = df.columns[0]

                    # 如果找到价格列，存储数据
                    if price_col:
                        self.stock_data[stock] = {
                            'data': df,      # 完整数据框
                            'price_col': price_col,     # 价格列名称
                            'prices': df[price_col]     # 价格数据序列
                        }
                        # 打印成功信息
                        print(f"✅ {stock}: {len(df)}天数据")
                except Exception as e:
                    # 打印错误信息
                    print(f"❌ {stock}: 加载失败 - {e}")
            else:
                # 打印文件不存在信息
                print(f"❌ {stock}: 文件不存在")

    def plot_black_swan_analysis_4charts(self, crisis_name, crisis_data):
        """
                绘制黑天鹅事件分析图表
                参数:
                    crisis_name: 危机名称
                    crisis_data: 危机相关数据
                """
        print(f"\n🎨 绘制 {crisis_name} 分析图表 (4张图表)...")

        # 依次创建4张分析图表
        self._create_chart1_value_and_drawdown(crisis_name, crisis_data)     # 价值变化和回撤分析
        self._create_chart2_contribution_and_risk(crisis_name, crisis_data)     # 贡献度和风险分析
        self._create_chart3_recovery_and_correlation(crisis_name, crisis_data)  # 恢复时间和相关性分析
        self._create_chart4_detailed_analysis(crisis_name, crisis_data)     # 详细分析和总结

    def _create_chart1_value_and_drawdown(self, crisis_name, crisis_data):
        """创建图表1：投资组合价值变化和各股票回撤分析"""
        # 创建1行2列的子图，设置图表大小
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
        # 设置图表总标题
        fig.suptitle(f"{crisis_name} - 价值变化与回撤分析", fontsize=16, fontweight='bold')

        # ==================== 子图1：投资组合价值变化 ====================
        portfolio_values = crisis_data['portfolio_values']   # 获取投资组合价值序列
        # 生成日期范围
        dates = pd.date_range(start=crisis_data['crisis_start'],
                              periods=len(portfolio_values), freq='D')
        # 绘制投资组合价值变化曲线
        ax1.plot(dates, portfolio_values, linewidth=3, color='#1f77b4', label='投资组合价值')
        # 添加初始价值参考线
        ax1.axhline(y=self.total_value, color='red', linestyle='--', linewidth=2, label='初始价值')

        # 标记最低价值点
        min_idx = np.argmin(portfolio_values)   # 找到最低点的索引
        ax1.scatter(dates[min_idx], portfolio_values[min_idx], color='red', s=100, zorder=5)
        # 添加最低点标注
        ax1.annotate(f'最低: ${portfolio_values[min_idx]:,.0f}',
                     xy=(dates[min_idx], portfolio_values[min_idx]),    # 标注点坐标
                     xytext=(10,10), textcoords='offset points',        # 文本偏移量
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))    # 文本框样式
        # 设置子图1的标题和标签
        ax1.set_title('投资组合价值变化')
        ax1.set_ylabel('投资组合价值 ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # 设置y轴格式为美元
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.tick_params(axis='x', rotation=45)     # x轴标签旋转45度
        # 添加统计信息文本框
        total_return = (portfolio_values[-1] - self.total_value) / self.total_value # 计算总收益率
        ax1.text(0.02, 0.98, f'总收益: {total_return:+.1%}',
                 transform = ax1.transAxes, verticalalignment='top',  # 使用相对坐标
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))  # 文本框样式

        # ==================== 子图2：各股票回撤分析 ====================
        # 计算每只股票的最大回撤
        drawdowns = [abs(crisis_data['stock_impacts'][stock]) for stock in self.portfolio.keys()]
        # 根据回撤幅度设置颜色：红色(>40%)、黄色(20-40%)、绿色(<20%)
        colors = ['#ff4444' if dd > 0.4 else '#ffaa00' if dd > 0.2 else '#aadd00' for dd in drawdowns]

        # 绘制柱状图
        bars = ax2.bar(self.portfolio.keys(), drawdowns, color=colors, alpha=0.7)
        ax2.set_title('各股票最大回撤')
        ax2.set_ylabel('回撤幅度')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        # 在柱子上添加回撤百分比
        for bar, dd in zip(bars, drawdowns):
            height = bar.get_height()    # 获取柱子高度
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{dd:.1%}',
                     ha='center', va='bottom', fontweight='bold')
        # 调整布局并显示图表
        plt.tight_layout()
        plt.show()

    def _create_chart2_contribution_and_risk(self, crisis_name, crisis_data):
        """创建图表2：股票贡献度和风险贡献分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
        fig.suptitle(f'{crisis_name} - 贡献度与风险分析', fontsize=16, fontweight='bold')

        # ==================== 子图1：各股票对组合收益的贡献 ====================
        stock_contributions = []     # 存储贡献度数据
        # 计算每只股票对组合收益的贡献
        for stock, impact in crisis_data['stock_impacts'].items():
            # 贡献度 = 股票收益率 × 股票权重
            contribution = impact * self.portfolio[stock] / self.total_value
            stock_contributions.append((stock, contribution))
        # 按贡献度排序（从小到大）
        stock_contributions.sort(key=lambda x: x[1])
        stocks = [x[0] for x in stock_contributions]    # 提取股票代码
        contributions = [x[1] for x in stock_contributions]  # 提取贡献度值

        # 设置颜色：红色(负贡献大)、黄色(小幅负贡献)、绿色(正贡献)
        colors = ['#ff4444' if c < -0.05 else '#ffaa00' if c < 0 else '#aadd00' for c in contributions]
        # 绘制水平柱状图
        bars = ax1.barh(stocks, contributions, color=colors, alpha=0.7)
        # 在柱子上添加贡献度百分比
        for i, (bar, contrib) in enumerate(zip(bars, contributions)):
            # 正数右对齐，负数左对齐
            ax1.text(contrib, i, f'{contrib:+.1%}',
                     ha='left' if contrib >= 0 else 'right',
                     va='center', fontweight = 'bold')
        ax1.set_title('各股票对组合收益的贡献')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)     # 零线
        ax1.grid(True, alpha=0.3, axis='x')  # x轴网格
        # ==================== 子图2：风险贡献分析 ====================
        risk_contributions = []     # 存储风险贡献数据
        for stock in self.portfolio.keys():
            weight = self.portfolio[stock] / self.total_value   # 计算权重
            impact = abs(crisis_data['stock_impacts'][stock])     # 取绝对值的冲击
            risk_contribution = weight * impact # 风险贡献 = 权重 × 冲击幅度
            risk_contributions.append((stock, risk_contribution))

        # 按风险贡献从大到小排序
        risk_contributions.sort(key=lambda x: x[1], reverse=True)
        stocks_risk = [x[0] for x in risk_contributions]    # 提取股票代码
        risks = [x[1] for x in risk_contributions]  # 提取风险贡献值

        # 使用循环颜色绘制柱状图
        bars = ax2.bar(stocks_risk, risks, color=['#ff6b6b', '#ffa726', '#ffee58', '#4ecdc4', '#45b7d1'] * 2)
        ax2.set_title('各股票对组合风险的贡献')
        ax2.set_ylabel('风险贡献度')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        # 计算总风险
        total_risk = sum(risks)
        # 在柱子上添加风险占比
        for bar, risk in zip(bars, risks):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height, f'{risk/total_risk:.1%}',
                     ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def _create_chart3_recovery_and_correlation(self, crisis_name, crisis_data):
        """创建图表3：恢复时间和相关性分析"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
        fig.suptitle(f'{crisis_name}- 恢复时间与相关性分析', fontsize=16, fontweight='bold')
        # ==================== 子图1：恢复时间分析 ====================
        recovery_data = []   # 存储恢复时间数据
        for stock in self.portfolio.keys():
            impact = crisis_data['stock_impacts'][stock]    # 获取股票冲击
            # 根据冲击幅度估算恢复时间
            if impact < -0.5:       # 下跌超过50%，恢复4年
                recovery = 4
            elif impact < -0.3:     # 下跌30-50%，恢复2年
                recovery = 2
            elif impact < -0.2:     # 下跌20-30%，恢复1年
                recovery = 1
            else:                   # 下跌小于20%，恢复0.5年
                recovery = 0.5
            recovery_data.append((stock, recovery))
        stocks_rec = [x[0] for x in recovery_data]    # 提取股票代码
        recoveries = [x[1] for x in recovery_data]      # 提取恢复时间
        # 绘制恢复时间柱状图
        bars = ax1.bar(stocks_rec, recoveries, color='#ff9ff3', alpha=0.7)
        ax1.set_title('各股票预计恢复时间')
        ax1.set_ylabel('恢复时间 (年)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        # 在柱子上添加恢复时间
        for bar, rec in zip(bars, recoveries):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{rec}年',
                     ha='center', va='bottom', fontweight='bold')

        # ==================== 子图2：相关性热力图 ====================
        # 计算危机期间的相关性矩阵
        correlation_matrix = self.calculate_crisis_correlation(crisis_name, crisis_data)
        # 创建热力图，使用红蓝配色
        im = ax2.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        # 设置坐标轴标签
        ax2.set_xticks(range(len(self.portfolio)))
        ax2.set_xticklabels(list(self.portfolio.keys()))
        ax2.set_yticks(range(len(self.portfolio)))
        ax2.set_yticklabels(list(self.portfolio.keys()))

        # 在热力图上添加相关系数值
        for i in range(len(self.portfolio)):
            for j in range(len(self.portfolio)):
                # 根据背景色深浅选择文字颜色
                ax2.text(j, i, f'{correlation_matrix[i,j]:.2f}',
                         ha='center', va='center',
                         color='black' if abs(correlation_matrix[i,j]) < 0.7 else 'white')

        ax2.set_title('危机期间股票相关性热力图')
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.show()

    def _create_chart4_detailed_analysis(self, crisis_name, crisis_data):
        """创建图表4：详细分析和总结"""
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,6))
        fig.suptitle(f'{crisis_name} - 详细分析与总结', fontsize=10, fontweight='bold')
        # ==================== 子图1：各股票表现对比 ====================
        # 获取每只股票的冲击数据
        impacts = [crisis_data['stock_impacts'][stock] for stock in self.portfolio.keys()]
        # 设置颜色：红色(大跌)、黄色(中跌)、绿色(小跌或上涨)
        colors=['#ff4444' if imp < -0.3 else '#ffaa00' if imp < -0.1 else '#aadd00' for imp in impacts]
        # 绘制各股票表现柱状图
        bars = ax1.bar(self.portfolio.keys(), impacts, color=colors, alpha=0.7)
        ax1.set_title('各股票在危机中的表现')
        ax1.set_ylabel('收益率')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        # 在柱子上添加收益率
        for bar, imp in zip(bars, impacts):
            height = bar.get_height()
            # 负数在下方显示，正数在上方显示
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{imp:+.1%}',
                     ha='center', va='bottom' if imp < 0 else 'top', fontweight='bold')
        # ==================== 子图2：总结统计信息 ====================
        ax2.axis('off') # 关闭坐标轴，创建纯文本区域
        # 计算关键统计指标
            #    总收益率
        total_return = (crisis_data['portfolio_values'][-1] - self.total_value) / self.total_value
        max_drawdown = crisis_data['max_drawdown']   # 最大回撤
        # 找出受影响最大的3只股票（按收益率排序）
        worst_stocks = sorted(crisis_data['stock_impacts'].items(), key=lambda x: x[1])[:3]
        # 找出表现最好的3只股票
        best_stocks = sorted(crisis_data['stock_impacts'].items(), key=lambda x: x[1], reverse=True)[:3]

        # 创建总结文本
        summary_text = f"""
危机分析总结

投资组合表现:
    初始价值: ${self.total_value:,.0f}
    最终价值: ${crisis_data['portfolio_values'][-1]:,.0f}
    总收益率: {total_return:+.1%}
    最大回撤: {max_drawdown:.1%}

受影响最大的股票:
    {worst_stocks[0][0]}: {worst_stocks[0][1]:+.1%}
    {worst_stocks[1][0]}: {worst_stocks[1][1]:+.1%}
    {worst_stocks[2][0]}: {worst_stocks[2][1]:+.1%}

表现最好的股票:
    {best_stocks[0][0]}: {best_stocks[0][1]:+.1%}
    {best_stocks[1][0]}: {best_stocks[1][1]:+.1%}
    {best_stocks[2][0]}: {best_stocks[2][1]:+.1%}

风险提示:
    组合脆弱点: {worst_stocks[0][0]} (下跌{abs(worst_stocks[0][1]):.1%})
    防御资产: {best_stocks[0][0]} (相对稳定)
"""
        # 在子图2中显示总结文本
        ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                 fontfamily='SimHei')
        plt.tight_layout()
        plt.show()

    def calculate_crisis_correlation(self, crisis_name, crisis_data):
        """
                计算危机期间的相关性矩阵
                注意：这是简化版本，实际应用中应该基于真实数据计算
                """
        n_stocks = len(self.portfolio)
        crisis_corr = 0.7        # 危机期间相关性系数
        # 创建相关性矩阵：对角线为1，其他位置为危机相关性
        corr_matrix = np.eye(n_stocks) * (1 - crisis_corr) + crisis_corr
        np.fill_diagonal(corr_matrix, 1.0)
        return corr_matrix

    def simulate_black_swan_event(self, crisis_name, start_date, end_date, severity=0.3):
        """
               模拟黑天鹅事件对投资组合的影响
               参数:
                   crisis_name: 危机名称
                   start_date: 开始日期
                   end_date: 结束日期
                   severity: 危机严重程度 (0-1)
               """
        print(f"\n🔴 模拟 {crisis_name}...")
        # 计算危机持续天数
        crisis_days = (datetime.strptime(end_date, '%Y-%m-%d') -
                       datetime.strptime(start_date, '%Y-%m-%d')).days
        # 初始化投资组合价值序列
        portfolio_values = [self.total_value]
        stock_impacts = {}  # 存储每只股票的冲击程度
        # ==================== 为每只股票生成冲击 ====================\
        for stock in self.portfolio.keys():
            # 根据不同股票类型设置不同的冲击程度
            if stock in ['LLY', 'NVO']:   # 医药股 - 相对防御
                impact = -severity * 0.8 + np.random.normal(0, 0.05)      # 下跌幅度较小
            elif stock in ['AAPL', 'META', 'MU']:   # 科技股 - 受影响较大
                impact = -severity * 1.2 + np.random.normal(0, 0.08)     # 下跌幅度较大
            elif stock in ['KO', 'SCHD']:  # 防御性股票 - 相对稳定
                impact = -severity * 0.5 + np.random.normal(0, 0.03) # 下跌幅度小
            elif stock in 'GLD':        # 黄金 - 避险资产，可能上涨
                impact = -severity * 0.3 + np.random.normal(0, 0.02)    # 可能小幅上涨
            else:   # 其他股票 - 中等影响
                impact = -severity + np.random.normal(0, 0.06)   # 中等下跌

            stock_impacts[stock] = impact       # 存储冲击数据

        # ==================== 生成每日价值变化 ====================
        for day in range(crisis_days):
            daily_return = 0     # 初始化日收益率
            for stock, value in self.portfolio.items():
                weight = value / self.total_value       # 计算股票权重
                # 计算每日冲击因子，危机初期冲击较小，逐渐增大
                day_factor = min(1.0, (day+1) / (crisis_days * 0.03))
                # 计算股票日收益率
                stock_daily_return = stock_impacts[stock] * day_factor / crisis_days
                daily_return += stock_daily_return * weight  # 累加到组合日收益率

            # 计算新的投资组合价值
            new_value = portfolio_values[-1] * (1 + daily_return)
            portfolio_values.append(new_value)  # 添加到价值序列

        # 整理危机数据
        crisis_data = {
            'crisis_start': start_date,  # 危机开始日期
            'crisis_end': end_date,     # 危机结束日期
            'portfolio_values': portfolio_values,       # 投资组合价值序列
            'stock_impacts': stock_impacts,     # 各股票冲击数据
            'total_return': (portfolio_values[-1] - self.total_value) / self.total_value,   # 总收益率
            'max_drawdown': min(stock_impacts.values()) # 最大回撤
        }

        # 存储结果
        self.results[crisis_name] = crisis_data
        return crisis_data

    def run_optimized_analysis(self):
        """运行完整的优化版分析流程"""
        print("🚀 开始优化版黑天鹅压力测试...")

        # 定义要测试的黑天鹅事件.    这些数据都是从deepseek找的.
        black_swan_events = {
            '2008年金融危机': ('2007-10-01', '2009-03-31', 0.5),  # 严重危机
            '2020年新冠疫情': ('2020-02-01', '2020-04-30', 0.3),  # 中等危机
            '医药监管危机': ('2022-01-01', '2022-03-31', 0.4),  # 针对医药股
            '科技股崩盘': ('2021-11-01', '2022-01-31', 0.35)  # 针对科技股
        }

        # 对每个黑天鹅事件进行模拟和分析
        for crisis_name, (start, end, severity) in black_swan_events.items():
            # 模拟危机影响
            crisis_data = self.simulate_black_swan_event(crisis_name, start, end, severity)
            # 绘制分析图表
            self.plot_black_swan_analysis_4charts(crisis_name, crisis_data)

# 程序入口点
def main():
    """主函数：创建测试器并运行分析"""
    tester = OptimizedBlackSwanTester()     # 创建压力测试器实例
    tester.run_optimized_analysis()          # 运行分析

# 如果直接运行此文件，执行main函数
if __name__ == "__main__":
    main()


'''
黑天鹅压力测试项目 - 完整总结报告
🎯 项目核心价值
成功构建了一个专业的投资组合压力测试系统，能够模拟极端市场事件对投资组合的冲击，为风险管理提供数据支持和决策依据。

🏗️ 系统架构与功能
1. 投资组合管理
资产配置: 10只股票，总投资$874
行业分布: 医药、科技、ETF、消费品、贵金属
权重分析: 自动计算各资产占比和风险暴露

2. 四维度分析框架
📈 图表1: 价值变化与回撤分析
投资组合价值曲线
最大回撤标记
个股回撤对比

⚖️ 图表2: 贡献度与风险分析
收益贡献度(水平条形图)
风险贡献度(垂直条形图)
颜色编码风险等级

⏰ 图表3: 恢复时间与相关性
预估恢复时间
相关性热力图
危机联动分析

📋 图表4: 详细总结报告
关键统计数据
最佳/最差表现股票
风险提示与建议

 风险指标体系
核心监控指标:
    最大回撤 (Max Drawdown) - 组合最大损失幅度
    总收益率 (Total Return) - 危机期间整体表现
    风险贡献度 (Risk Contribution) - 各资产风险暴露
    恢复时间 (Recovery Time) - 资金回本预估
    相关性矩阵 (Correlation Matrix) - 资产联动性

测试场景覆盖
模拟的4种黑天鹅事件:
    2008年金融危机 (严重程度: 50%)
    2020年新冠疫情 (严重程度: 30%)
    医药监管危机 (针对性冲击: 40%)
    科技股崩盘 (针对性冲击: 35%)
'''





