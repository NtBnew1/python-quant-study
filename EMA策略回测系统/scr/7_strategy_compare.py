"""
文件：src/7_strategy_compare.py

功能：

加载三种结果：
-results/backtest_results.csv（动量_日线）
-results/backtest_results_ema_日线.csv
-results/backtest_results_ema_周线.csv

绘制 4 张对比图：
-平均收益率对比（柱状图）
-胜率对比（柱状图）
-夏普比率对比（柱状图）
-热力图（每只股票 × 三种策略）
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
from matplotlib.font_manager import FontProperties


# 使用系统字体
font_path = r'C:\Windows\Fonts\msyh.ttc'
chinese_font = FontProperties(fname=font_path)

sns.set_style('whitegrid')

class StrategyCompare:
    """ 日线 VS 周线策略对比类"""
    def __init__(self, project_dir):
        """
        初始化
        Args:
            project_dir:  项目根目录
        """
        self.project_dir = project_dir
        self.results_dir = project_dir / "results"
        self.charts_dir = project_dir / "charts" / "日线VS周线对比图"
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        # 汇总对比图目录
        self.summary_charts_dir = self.charts_dir / "日线VS周线_汇总对比图"
        self.summary_charts_dir.mkdir(parents=True, exist_ok=True)

        # 个股对比图目录
        self.stock_charts_dir = self.charts_dir / "日线VS周线_个股对比图"
        self.stock_charts_dir.mkdir(parents=True, exist_ok=True)

        print(f'=' * 60)
        print(f" 日线 VS 周线 策略对比图")
        print(f'=' * 60)
        print(f" 图表保存目录: {self.charts_dir}")
        print(f" 汇总图目录: {self.summary_charts_dir}")
        print(f" 个股图目录: {self.stock_charts_dir}")
        print()

    def load_results(self):
        """加载日线和周线回测结果"""
        files = {
            '日线': self.results_dir / "汇总" /"backtest_results_ema_日线.csv",
            '周线': self.results_dir / "汇总" /"backtest_results_ema_周线.csv"
        }

        self.dfs = {}

        for name, file_path in files.items():
            if file_path.exists():
                df = pd.read_csv(file_path)
                self.dfs[name] = df
                print(f" 加载{name}: {len(df)}条记录")
                print(f' 列名: {df.columns.tolist()}') # 添加这行显示列名
            else:
                print(f" 文件不存在: {file_path}")
                self.dfs[name] = None

        return self.dfs

    def load_trade_details(self):
        """加载日线和周线的交易明细"""
        daily_detail_dir = self.results_dir / "明细_日线"
        weekly_detail_dir = self.results_dir / "明细_周线"

        self.daily_trades = {}
        self.weekly_trades = {}

        if daily_detail_dir.exists():
            for stock_dir in daily_detail_dir.iterdir():
                if stock_dir.is_dir():
                    stock_code = stock_dir.name
                    self.daily_trades[stock_code] = {}
                    for csv_file in stock_dir.glob('*.csv'):
                        signal_type = csv_file.stem
                        df = pd.read_csv(csv_file)
                        self.daily_trades[stock_code][signal_type] = df
            print(f" 加载日线明细; {len(self.daily_trades)} 只股票")
        else:
            print(f" 日线明细目录不存在: {daily_detail_dir}")

        if weekly_detail_dir.exists():
            for stock_dir in weekly_detail_dir.iterdir():
                if stock_dir.is_dir():
                    stock_code = stock_dir.name
                    self.weekly_trades[stock_code] = {}
                    for csv_file in stock_dir.glob('*.csv'):
                        signal_type = csv_file.stem
                        df = pd.read_csv(csv_file)
                        self.weekly_trades[stock_code][signal_type] = df
            print(f" 加载周线明细: {len(self.weekly_trades)} 只股票")
        else:
            print(f" 周线明细目录不存在: {weekly_detail_dir}")

        return self.daily_trades, self.weekly_trades


    def plot_avg_return(self):
        """ 图1. 平均收益率对比 (柱状图)"""
        print(f" \n 绘制图1: 平均收益率对比.......")

        # 计算每种策略的平均收益率
        data = {}
        for name, df in self.dfs.items():
            if df is not None and 'profit_pct' in df.columns:
                data[name] = df['profit_pct'].mean()

        if not data:
            print(f" 没有数据!   ")
            return

        # =======转为列表 ============
        strategy_names = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(8,6))
        bars = ax.bar(strategy_names, values, color=['#1f77b4', '#ff7f0e'])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontproperties=chinese_font)

        ax.set_title('日线 vs 周线 平均收益率对比', fontproperties=chinese_font)
        ax.set_ylabel('平均收益率 (%)', fontproperties=chinese_font)
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels(strategy_names, fontproperties=chinese_font)


        ax.set_ylim(0, max(data.values()) * 1.2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ====== 保存 =========
        output_file = self.summary_charts_dir / "平均收益率对比.png"
        plt.savefig(output_file, dpi=300)
        print(f" 已保存: {output_file}")
        plt.show()
        plt.close()

    def plot_win_rate(self):
        """图2: 胜率对比 (柱状图)"""
        print(f'\n' + '=' * 60)

        # 计算每种策略的平均胜率
        data = {}
        for name, df in self.dfs.items():
            if df is not None and 'profit' in df.columns:
                win_count = (df['profit'] > 0).sum()
                total_count = len(df)
                data[name] = (win_count / total_count) * 100
                print(f' {name}:')
                print(f' 盈利: {win_count}/{total_count} = {data[name]:.1f}%')
                print(f" 平均收益: ${df['profit'].mean():,.2f}")

        if not data:
            print(f" 没有数据!  ")
            return

        strategy_names = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(8,6))
        bars = ax.bar(strategy_names, values, color=['#1f77b4', '#ff7f0e'])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}%", ha='center', va='bottom', fontproperties=chinese_font)

        ax.set_title(' 日线 vs 周线 平均胜率对比', fontproperties=chinese_font)
        ax.set_ylabel('平均胜率 (%)', fontproperties=chinese_font)
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels(strategy_names, fontproperties=chinese_font)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ========== 保存 ========
        output_file = self.summary_charts_dir / "胜率对比.png"
        plt.savefig(output_file, dpi=300)
        print(f" 已保存: {output_file}")
        plt.show()
        plt.close()

    def plot_sharpe_ratio(self):
        """ 图3: 夏普比率对比 """
        print(f' \n 绘制图3: 夏普比率对比........')

        # 计算每种策略的平均夏普比率
        data = {}
        for name, df in self.dfs.items():
            if df is not None and 'sharpe_ratio' in df.columns:
                data[name] = df['sharpe_ratio'].mean()
                print(f' {name}: 平均夏普比率 = {data[name]:.3f}')

        if not data:
            print(f' 没有数据! ')
            return

        strategy_names = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(figsize=(8,6))
        bars = ax.bar(strategy_names, values, color=['#1f77b4', '#ff7f0e'])

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontproperties=chinese_font)


        ax.set_title('日线 vs 周线 平均夏普比率对比', fontproperties=chinese_font)
        ax.set_ylabel('平均夏普比率', fontproperties=chinese_font)
        ax.set_xticks(range(len(strategy_names)))
        ax.set_xticklabels(strategy_names, fontproperties=chinese_font)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ========= 保存 =============
        output_file = self.summary_charts_dir / "夏普比率对比.png"
        plt.savefig(output_file, dpi=300)
        print(f" 已保存: {output_file}")
        plt.show()
        plt.close()

    def plot_heatmap(self):
        """图4: 热力图 (每只股票 x 日线/周线)"""
        print(f'\n 绘制图4: 热力图')

        # 构建数据
        stock_data = {}
        all_stocks = set()

        for name, df in self.dfs.items():
            if df is not None and 'profit_pct' in df.columns and 'stock_code' in df.columns:
                # 对每只股票, 计算三种信号的平均收益率
                stock_avg = df.groupby('stock_code')['profit_pct'].mean()
                stock_data[name] = stock_avg
                all_stocks.update(stock_avg.index)

        if not stock_data:
            print(f' 没有数据! ')
            return

        # 创建热力图数据
        stocks = sorted(list(all_stocks))
        periods = ['日线', '周线']

        heatmap_data = []
        for stock in stocks:
            row = []
            for period in periods:
                if period in stock_data:
                    row.append(stock_data[period].get(stock, 0))
                else:
                    row.append(0)
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(10,8))
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

        # 设置刻度
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontproperties=chinese_font)
        ax.set_yticks(range(len(stocks)))
        ax.set_yticklabels(stocks, fontproperties=chinese_font)

        # 添加数值标签
        for i in range(len(stocks)):
            for j in range(len(periods)):
                value = heatmap_data[i][j]
                color = 'white' if abs(value) > 30 else 'black'
                ax.text(j,i, f'{value:.1f}%', ha='center', va='center', color=color, fontsize=9)

        ax.set_title('每只股票 x 日线/周线 收益率 (%)', fontproperties=chinese_font, fontsize=14)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('收益率(%)', fontproperties=chinese_font)

        plt.tight_layout()

        # ======= 保存 =======
        output_file = self.summary_charts_dir / "热力图.png"
        plt.savefig(output_file, dpi=300)
        print(f' 已保存: {output_file}')
        plt.show()
        plt.close()

    def aggregate_monthly(self, trade_dict):
        """把交易明细按月份聚合
        Args:
            trade_dict: 交易明细字典 {股票代码: {信号类型: DataFrame}}

        Returns:
            dict: {股票代码: {信号类型: {月份: 收益}}}
        """

        monthly_data = {}

        for stock_code, signals in trade_dict.items():
            monthly_data[stock_code] = {}

            for signal_type, df in signals.items():
                if df is None or len(df) == 0:
                    continue

                # 检查是否有 exit_date 和 profit 列
                if 'exit_date' not in df.columns or 'profit' not in df.columns:
                    continue

                # 把exit_date 转成 Datetime
                df = df.copy()
                df['exit_date'] = pd.to_datetime(df['exit_date'], utc=True, errors='coerce')
                df = df.dropna(subset=['exit_date'])

                # 提取月份
                df['year_month'] = df['exit_date'].dt.strftime('%Y-%m')

                # 按月份聚合收益
                monthly = df.groupby('year_month')['profit'].sum().to_dict()

                monthly_data[stock_code][signal_type] = monthly

        return monthly_data

    def plot_stock_compare(self, stock_code, daily_monthly, weekly_monthly):
        """绘制单只股票的日线 vs 周线月度收益对比图
        Args:
            stock_code: 股票代码,
            daily_monthly: 日线月度收益,
            weekly_monthly: 周线月度收益
        """
        print(f' \n 绘制: {stock_code}')

        # 三种信号
        signal_types = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']

        for signal_type in signal_types:
            print(f" \n 绘制: {stock_code} - {signal_type}")

            # 获取日线月度数据
            daily_data = daily_monthly.get(stock_code, {}).get(signal_type, {})

            # 获取周线月度数据
            weekly_data = weekly_monthly.get(stock_code, {}).get(signal_type, {})

            if not daily_data and not weekly_data:
                print(f" {stock_code} 没有数据")
                return

            # 合并所有月份
            all_months = sorted(set(list(daily_data.keys()) + list(weekly_data.keys())))

            # 计算累计收益
            daily_cumulative = []
            weekly_cumulative = []
            daily_total = 0
            weekly_total = 0

            for month in all_months:
                daily_total += daily_data.get(month, 0)
                weekly_total += weekly_data.get(month, 0)
                daily_cumulative.append(daily_total)
                weekly_cumulative.append(weekly_total)


            # 绘制
            fig, ax = plt.subplots(figsize=(14, 6))

            x = range(len(all_months))
            width = 0.35

            # 画折线
            ax.plot(x, daily_cumulative, marker='o', label='日线', color='#1f77b4', linewidth=2, markersize=6)
            ax.plot(x, weekly_cumulative, marker='s', label='周线', color='#ff7f0e', linewidth=2, markersize=6)

            # 添加零线
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

            # 设置标签
            ax.set_title(f'{stock_code} - {signal_type} - 日线 vs 周线 月度收益对比', fontproperties=chinese_font, fontsize=14)
            ax.set_xlabel('月份', fontproperties=chinese_font)
            ax.set_ylabel('收益金额 ($)', fontproperties=chinese_font)
            ax.set_xticks(x)
            ax.set_xticklabels(all_months, rotation=45, ha='right')
            ax.legend(prop=chinese_font)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # ======= 保存 ===========
            stock_dir = self.stock_charts_dir / stock_code
            stock_dir.mkdir(parents=True, exist_ok=True)
            output_file = stock_dir / f"{signal_type}_累计收益对比.png"
            plt.savefig(output_file, dpi=300)
            print(f" 已保存: {output_file}")
            plt.show()
            plt.close()

            print(f' 已绘制: {stock_code}')

    def plot_all_stocks(self, daily_monthly, weekly_monthly):
        """ 循环所有股票, 绘制日线 vs 周线累计收益对比图
        Args:
            daily_monthly: 日线月度收益
            weekly_monthly: 周线月度收益
            """
        print(f'\n' + '=' * 60)
        print(f" 绘制所有股票的对比图")
        print(f"=" * 60)

        # 获取所有股票
        all_stocks = set(daily_monthly.keys()) | set(weekly_monthly.keys())
        all_stocks = sorted(list(all_stocks))

        print(f" 共 {len(all_stocks)} 只股票")
        print(f'-' * 40)

        # 循环每只股票
        for stock_code in all_stocks:
            self.plot_stock_compare(stock_code, daily_monthly, weekly_monthly)

        print(f' \n' +"=" * 60)
        print(f" 所有股票对比图绘制完成! ")
        print(f" =" * 60)

# ===================================
# 测试
# ===================================
if __name__ == "__main__":
    print(f'\n' + '=' * 60)
    print(f" 测试 __init__")
    print('=' * 60)

    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    compare = StrategyCompare(project_dir)

    # 加载数据
    dfs = compare.load_results()
    print(f" \n load_results 测试完成! ")
    if dfs.get('日线') is not None:
        print(f" 日线数据: {len(dfs['日线'])} 条")
    else:
        print(f" 日线数据: 不存在")

    if dfs.get('周线') is not None:
        print(f" 周线数据: {len(dfs['周线'])} 条")
    else:
        print(f' 周线数据: 不存在')

    # 绘制平均收益率对比图
    compare.plot_avg_return()

    # 绘制平均胜率对比图
    compare.plot_win_rate()

    # 绘制平均夏普比率对比图
    compare.plot_sharpe_ratio()

    # 绘制热力图
    compare.plot_heatmap()

    # 加载交易明细
    print(f" \n 测试 load_trades_details")
    daily_trades, weekly_trades = compare.load_trade_details()
    print(f" load_trade_details 测试完成! ")
    if daily_trades:
        print(f" 日线股票: {list(daily_trades.keys())}")
    if weekly_trades:
        print(f" 周线股票: {list(weekly_trades.keys())}")


    # 按月聚合
    print(f" \n 测试 aggregate_monthly")
    daily_monthly = compare.aggregate_monthly(daily_trades)
    weekly_monthly = compare.aggregate_monthly(weekly_trades)

    print(f" aggregate_monthly 测试完成")

    # 打印 AAPL 的月度收益示列
    if 'AAPL' in daily_monthly:
        print(f' \n AAPL 日线月度收益: ')
        for signal_type, months in daily_monthly['AAPL'].items():
            print(f" {signal_type}: ")
            for month, profit in sorted(months.items()):
                print(f" {month}: ${profit:,.2f}")

    # ======= 添加周线的打印 ==============
    if 'AAPL' in weekly_monthly:
        print(f" \n AAPL 周线月度收益: ")
        for signal_type, months in weekly_monthly['AAPL'].items():
            print(f" {signal_type}: ")
            for month, profit in sorted(months.items()):
                print(f" {month}: ${profit:,.2f}")


    # 加载交易明细
    daily_trades, weekly_trades = compare.load_trade_details()

    # 按月聚合
    daily_monthly = compare.aggregate_monthly(daily_trades)
    weekly_monthly = compare.aggregate_monthly(weekly_trades)

    # ============== 循环绘制所有股票 ================
    compare.plot_all_stocks(daily_monthly, weekly_monthly)



"""
## 策略对比模块 (7_strategy_compare.py)

**任务目标**：
- 加载日线和周线回测结果
- 加载日线和周线交易明细
- 绘制四张汇总对比图（平均收益率、胜率、夏普比率、热力图）
- 绘制每只股票的日线 vs 周线月度累计收益对比图

**实现方案**：
1. **数据加载**：
   - 加载日线回测结果（results/汇总/backtest_results_ema_日线.csv）
   - 加载周线回测结果（results/汇总/backtest_results_ema_周线.csv）
   - 加载日线交易明细（results/明细_日线/{股票代码}/*.csv）
   - 加载周线交易明细（results/明细_周线/{股票代码}/*.csv）

2. **汇总对比图**（4张图）：
   - **图1：平均收益率对比**：日线 vs 周线平均收益率柱状图，标注数值
   - **图2：胜率对比**：日线 vs 周线盈利次数占比柱状图，显示盈利/总数
   - **图3：夏普比率对比**：日线 vs 周线平均夏普比率柱状图
   - **图4：热力图**：每只股票 × 日线/周线收益率热力图，红绿渐变

3. **交易明细按月聚合**：
   - 将每笔交易的`exit_date`转换为日期格式
   - 按月份（YYYY-MM）聚合收益
   - 计算累计收益曲线

4. **个股对比图**：
   - 每只股票、每种信号分别绘制
   - 日线 vs 周线月度累计收益折线图
   - 添加零线参考
   - 保存到charts/日线VS周线对比图/日线VS周线_个股对比图/

5. **中文字体支持**：
   - 使用微软雅黑字体（C:\Windows\Fonts\msyh.ttc）
   - 解决中文显示问题

**核心代码结构**：
```python
class StrategyCompare:
    ├── __init__()               # 初始化目录
    ├── load_results()            # 加载日线和周线回测结果
    ├── load_trade_details()      # 加载日线和周线交易明细
    ├── plot_avg_return()         # 图1：平均收益率对比
    ├── plot_win_rate()           # 图2：胜率对比
    ├── plot_sharpe_ratio()       # 图3：夏普比率对比
    ├── plot_heatmap()            # 图4：热力图
    ├── aggregate_monthly()       # 按月份聚合交易明细
    ├── plot_stock_compare()      # 绘制单只股票对比图
    └── plot_all_stocks()         # 绘制所有股票对比图
```
"""



























