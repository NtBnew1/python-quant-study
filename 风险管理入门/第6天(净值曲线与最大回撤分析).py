'''
第6天：
构建投资组合净值曲线，实现回测基础。
练习：计算累计收益、最大回撤和夏普比率，绘制净值曲线。
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import glob
import os

# 设置中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DajaVu Sans']
plt.rcParams['axes.unicode_minus'] =  False
plt.rcParams['font.size'] = 10

class DetailPortfolioBacktest:
    """详细标记关键点的回测系统"""
    def __init__(self, inital_capital = 10000):
        """
                初始化回测系统

                参数:
                    initial_capital: 初始资金，默认1万
                """
        self.initial_capital = inital_capital

    def load_historical_data(self):
        """
                加载历史股票数据

                功能:
                    1. 读取当前目录下所有股票数据文件
                    2. 合并数据并清理
                    3. 计算收益率

                返回:
                    bool: 数据加载是否成功
                """
        print("正在加载历史数据...")
        all_data = {}
        # 使用glob查找所有股票数据文件
        stock_files = glob.glob('./*_stock_data.xlsx')

        for file_path in stock_files:
            try:
                # 从文件名提取股票代码
                ticker = os.path.basename(file_path).replace('_stock_data.xlsx', ' ')
                # 读取Excel文件，第一列为索引（日期）
                df = pd.read_excel(file_path, index_col=0, parse_dates=True)
                # 检查数据是否完整：必须有Close列且数据量足够
                if 'Close' in df.columns and len(df) > 500:
                    all_data[ticker] = df['Close']
                    print(f"{ticker}")
                else:
                    print(f"{ticker}: 数据不足")
            except Exception as e:
                print(f"{ticker}: 加载失败: {e}")
        # 将字典转换为DataFrame，每列是一只股票的收盘价
        self.data = pd.DataFrame(all_data)
        # 按日期排序并向前填充缺失值，然后删除仍有缺失的行
        self.data = self.data.sort_index().ffill().dropna()

        # 计算日收益率：(今日收盘-昨日收盘)/昨日收盘
        self.returns = self.data.pct_change().dropna()
        # 保存日期索引，方便后续使用
        self.dates = self.returns.index

        print(f"\n📊 加载完成: {len(self.data.columns)}只股票")
        print(f"📅 时间范围: {self.dates[0].strftime('%Y-%m-%d')} 至 {self.dates[-1].strftime('%Y-%m-%d')}")
        return True

    def calculate_performance(self):
        """
                计算投资组合表现

                方法:
                    1. 等权重分配资金
                    2. 计算组合日收益率
                    3. 计算累计净值

                解释:
                    - 等权重：每只股票分配相同比例的资金
                    - 组合收益率 = 各股票收益率 × 权重 的和
                    - 累计净值 = 初始资金 × (1 + 累计收益率)
                """
        # 股票数量
        n_stocks = len(self.returns.columns)
        # 创建等权重数组：每只股票权重为 1/n
        weights = np.array([1 / n_stocks] * n_stocks)
        # 计算组合日收益率：各股票收益率乘以权重后求和
        portfolio_returns = (self.returns * weights).sum(axis=1)

        # 计算累计净值：(1 + 收益率) 的累积乘积 × 初始资金
        self.portfolio_values = self.initial_capital * (1 + portfolio_returns).cumprod()
        return True

    def find_max_drawdown_details(self):
        """
                详细分析最大回撤
                回撤定义:
                    从前期高点到后期低点的跌幅
                返回:
                    dict: 包含回撤详细信息
                """
        # 初始化变量
        peak = self.portfolio_values.iloc[0]    # 初始峰值
        max_drawdown = 0                    # 最大回撤点
        peak_date = self.dates[0]           # 峰值日期
        trough_date = self.dates[0]         # 谷底日期
        recovery_date = None                # 恢复日期

        peak_values = []        # 记录所有局部峰值点
        trough_values = []      # 记录所有局部谷底点

        # 遍历每个时间点，计算回撤
        for i, (date, value) in enumerate(zip(self.dates, self.portfolio_values)):
            # 如果当前值创出新高，更新峰值
            if value > peak:
                peak = value
                peak_date = date

            # 计算当前回撤：(峰值-当前值)/峰值
            drawdown = (peak - value) / peak

            # 如果当前回撤大于历史最大回撤，更新最大回撤信息
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_peak_date = peak_date       # 最大回撤开始日期
                max_trough_date = date          # 最大回撤最低点日期

            # 识别局部极值点（用于买卖点分析）
            if i > 1 and i < len(self.portfolio_values) - 1:
                prev_val = self.portfolio_values.iloc[i - 1]     # 前一日值
                next_val = self.portfolio_values.iloc[i + 1]     # 后一日值
                # 如果是局部峰值（比前后都高）
                if value > prev_val and value > next_val:
                    peak_values.append((date, value))
                # 如果是局部谷底（比前后都低）
                elif value < prev_val and value < next_val:
                    trough_values.append((date, value))

        # 寻找回撤恢复日期（净值回到前高）
        for i, (date, value) in enumerate(zip(self.dates, self.portfolio_values)):
            # 在最大回撤低点之后，且净值恢复到了回撤前的高点
            if date > max_trough_date and value >= self.portfolio_values.loc[max_peak_date]:
                recovery_date = date
                break
        return {
            'max_drawdown': max_drawdown,      # 最大回撤比例
            'peak_date': max_peak_date,        # 回撤开始日期
            'trough_date': max_trough_date,    # 回撤最低点日期
            'recovery_date': recovery_date,    # 回撤恢复日期
            'peak_values': peak_values[-10:],  # 最近10个局部峰值（卖点候选）
            'trough_values': trough_values[-10:]  # 最近10个局部谷底（买点候选）
        }

    def plot_detailed_analysis(self):
        """
                绘制详细分析图表 - 分为两张独立图表

                图表1: 净值曲线与关键点位
                图表2: 回撤分析
                """
        # 获取详细的最大回撤信息
        drawdown_info = self.find_max_drawdown_details()

        # ===== 第一张图：净值曲线与关键点 =====
        plt.figure(figsize=(15,8))
        # 绘制净值曲线
        plt.plot(self.dates, self.portfolio_values,
                 linewidth=2, color='blue', label='投资组合净值', alpha=0.8)
        # 标记最大回撤的关键点
        peak_val = self.portfolio_values.loc[drawdown_info['peak_date']]
        trough_val = self.portfolio_values.loc[drawdown_info['trough_date']]

        # 回撤开始点（红色三角）
        plt.scatter(drawdown_info['peak_date'], peak_val,
                    color='red', s=150, zorder=5, label='回撤开始点', marker='^')
        # 最大回撤点（橙色倒三角）
        plt.scatter(drawdown_info['trough_date'], trough_val,
                    color='orange', s=150, zorder=5, label='最大回撤点', marker='v')
        # 标记回撤恢复点（如果有）- 绿色方块
        if drawdown_info['recovery_date']:
            recovery_val = self.portfolio_values.loc[drawdown_info['recovery_date']]
            plt.scatter(drawdown_info['recovery_date'], recovery_val,
                        color='green', s=150, zorder=5, label='回撤恢复点', marker='s')
        # 标记买卖点（最近5个峰值和谷底）
        # 紫色三角：卖点候选（局部峰值）
        for date, value in drawdown_info['peak_values'][-5:]:
            plt.scatter(date, value, color='purple', s=80, alpha=0.6, marker='^')
        # 棕色倒三角：买点候选（局部谷底）
        for date, value in drawdown_info['trough_values'][-5:]:
            plt.scatter(date, value, color='brown', s=80, alpha=0.6, marker='v')

        # 绘制回撤区间阴影
        end_date = drawdown_info['trough_date'] if not drawdown_info['recovery_date'] else drawdown_info['recovery_date']
        plt.axvspan(drawdown_info['peak_date'], end_date, alpha=0.2, color='red',
                    label='最大回撤区间')

        # 设置图表属性
        plt.title('投资组合净值曲线 - 关键点位分析\n(红三角:卖点, 棕三角:买点, 橙点:最大回撤)',
                  fontsize=10, fontweight='bold', pad=20)
        plt.ylabel('净值', fontsize=10)
        plt.xlabel('日期', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        # Y轴格式化为万元显示
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x / 10000:.1f}万"))
        # X轴日期格式设置
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.tight_layout()
        plt.show()

        # ===== 第二张图：回撤分析 =====
        plt.figure(figsize=(15,6))
        # 计算每日回撤
        drawdowns = []
        peak = self.portfolio_values.iloc[0]        # 初始峰值

        for value in self.portfolio_values:
            # 更新运行峰值
            if value > peak:
                peak = value
            # 计算当前回撤
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        # 绘制回撤区域（填充）
        plt.fill_between(self.dates, drawdowns, 0, color='red', alpha=0.3, label='回撤区域')
        # 绘制回撤曲线
        plt.plot(self.dates, drawdowns, color='red', linewidth=1, alpha=0.8)
        # 标记最大回撤点
        max_dd_idx = np.argmax(drawdowns)       # 找到最大回撤的索引
        plt.scatter(self.dates[max_dd_idx], drawdowns[max_dd_idx],
                    color='orange', s=100, zorder=5, label='最大回撤点')
        # 设置图表属性
        plt.title('回撤分析', fontsize=10, fontweight='bold', pad=20)
        plt.ylabel('回撤比列', fontsize=10)
        plt.xlabel('日期', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        # Y轴显示为百分比
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        # X轴日期格式设置
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.tight_layout()
        plt.show()

        # ===== 计算并输出关键指标 =====
        print("\n🔍 关键点位分析:")
        print(f"最大回撤: {drawdown_info['max_drawdown']:.2%}")
        print(f"回撤开始: {drawdown_info['peak_date'].strftime('%Y-%m-%d')}")
        print(f"回撤最低: {drawdown_info['trough_date'].strftime('%Y-%m-%d')}")

        # 计算夏普比率和其他风险调整收益指标
        portfolio_returns = self.portfolio_values.pct_change().dropna()
        # 年化收益率 = 日均收益率 × 252个交易日
        annual_return = portfolio_returns.mean() * 252
        # 年化波动率 = 日收益率标准差 × √252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        # 夏普比率 = 年化收益率 / 年化波动率（假设无风险利率为0）
        sharpe_ratio = annual_return / annual_volatility

        print(f"年化夏普比率: {sharpe_ratio:.2f}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {annual_volatility:.2%}")

        # 回撤恢复信息
        if drawdown_info['recovery_date']:
            recovery_days = (drawdown_info['recovery_date'] - drawdown_info['peak_date']).days
            print(f"回撤恢复: {drawdown_info['recovery_date'].strftime('%Y-%m-%d')}"
                  f"(历时{recovery_days}天)")
        else:
            print(f"当前仍处于回撤中，尚未恢复前高")

        print(f"\n💡 买卖点提示:")
        print(f"最近买点(谷底): {drawdown_info['trough_values'][-1][0].strftime('%Y-%m-%d') if drawdown_info['trough_values'] else '无'}")
        print(f"最近卖点(峰值): {drawdown_info['peak_values'][-1][0].strftime('%Y-%m-%d') if drawdown_info['peak_values'] else '无'}")

def main():
    """
        主函数 - 程序入口点

        执行流程:
            1. 初始化回测系统
            2. 加载历史数据
            3. 计算组合表现
            4. 绘制分析图表
        """
    print("🎯 第6天任务：详细点位分析")
    print("=" * 50)

    # 初始化回测系统，设置初始资金10万
    backtest = DetailPortfolioBacktest(inital_capital=10000)
    # 加载数据并执行回测
    if backtest.load_historical_data():
        backtest.calculate_performance()
        print("\n📈 绘制详细分析图表...")
        backtest.plot_detailed_analysis()
        print("\n" + "=" * 50)
        print("✅ 详细分析完成！")
        print("=" * 50)

if __name__ == "__main__":
    main()














