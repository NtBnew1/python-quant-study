"""
第5天：绘制结果
1. 文件名：5. 绘制结果.py
2. 输入：results/backtest_results.csv
3. 输出：charts/*.png

4. 步骤：
1. 创建 charts/ 目录
2. 加载回测结果
3. 绘制净值曲线
4. 绘制收益对比柱状图
5. 绘制动量20 vs 60对比图
6. 保存图表

5. 图表清单：
-- 净值曲线.png
-- 收益对比.png
-- 动量20_vs_60.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================================
# 回测结果可视化
# ================================
class ResultVisualizer:
    """回测结果可视化类"""
    def __init__(self, project_dir):
        """
        初始化
        Args:
            project_dir: 项目根目录
        """
        self.project_dir = project_dir
        self.charts_dir = None
        self.df = None

        print(f'=' * 60)
        print(f" 回测结果可视化")
        print(f'=' * 60)
        print(f' 项目目录: {self.project_dir}')
        print()

    def setup_directories(self):
        """创建 Charts / 目录"""
        self.charts_dir = self.project_dir / "charts"
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        print(f' 图表目录已创建: {self.charts_dir}')
        return self.charts_dir

    def load_data(self):
        """
        加载回测结果数据
        从 results/backtest_results.csv 读取
        """
        results_file = self.project_dir / "results" / "backtest_results.csv"
        if not results_file.exists():
            print(f' 回测结果文件不存在: {results_file}')
            print(f" 请先运行 4. Backtrader 回测.py 生成回测结果")
            return None

        self.df = pd.read_csv(results_file)
        print(f" 数据加载成功: {len(self.df)} 条记录")
        print(f' 列名: {self.df.columns.tolist()}')
        print(self.df.head())
        print()
        return self.df

    def plot_equity_curve(self):
        """
        绘制净值曲线
        显示每只股票在三种信号下的最终资金对比
        """
        if self.df is None:
            print(f" 请先加载数据: Load_data()")
            return

        # 准备数据: 按股票和信号类型组织
        pivot_df = self.df.pivot(index='stock_code', columns='signal_type', values='final_cash')

        # 绘制
        fig, ax = plt.subplots(figsize=(12, 6))

        # 设置柱状图位置
        x = range(len(pivot_df.index))
        width = 0.25

        # 三种信号分别绘制柱状图
        ax.bar([i-width for i in x], pivot_df['signal_20_60'], width, label='signal_20_60', color='#1f77b4')
        ax.bar(x, pivot_df['signal_10_60'], width, label='signal_10_60', color='#ff7f0e')
        ax.bar([i+width for i in x], pivot_df['signal_10_20_60'], width, label='signal_10_20_60', color='#2ca02c')

        # 添加初始资金参考线
        ax.axhline(y=30000, color='red', linestyle='--', linewidth=1, label='初始资金 $30,000')

        # 添加标签
        ax.set_xlabel('股票代码')
        ax.set_ylabel('最终资金 ($)')
        ax.set_title('各股票在不同信号下的净值对比')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 保存图表
        output_file = self.charts_dir / "净值曲线.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        plt.close()
        print(f" 净值曲线已保存: {output_file}")
        return output_file

    def plot_return_comparison(self):
        """
        绘制收益对比柱状图
        显示每只股票在三种信号下的收益率对比
        """
        if self.df is None:
            print(f" 请先加载数据: load_data()")
            return

        # 准备数据: 按股票和信号类型组织
        pivot_df = self.df.pivot(index='stock_code', columns='signal_type', values='profit_pct')

        # 绘制
        fig, ax = plt.subplots(figsize=(12,6))

        # 设置柱状图位置
        x = range(len(pivot_df.index))
        width = 0.25

        # 三种信号分别绘制柱状图
        bars1 = ax.bar([i-width for i in x], pivot_df['signal_20_60'], width, label='signal_20_60',
                       color='#1f77b4')
        bars2 = ax.bar(x, pivot_df['signal_10_60'], width, label='signal_10_60', color='#ff7f0e')
        bars3 = ax.bar([i+width for i in x], pivot_df['signal_10_20_60'], width, label='signal_10_20_60',
                       color='#2ca02c')

        # 添加零线
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

        # 添加标签
        ax.set_xlabel('股票代码')
        ax.set_ylabel(' 收益率 (%)')
        ax.set_title('各股票在不同信号下的收益率对比')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 在柱状图上添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x()+bar.get_width()/2., height+0.5,
                            f"{height:.1f}%", ha='center', va='bottom', fontsize=8)
                else:
                    ax.text(bar.get_x()+bar.get_width()/2., height - 0.5,
                            f"{height:.1f}%", ha='center', va='top', fontsize=8)

        # 保存图表
        output_file = self.charts_dir / "收益对比.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()          # 在close 前面加入show就会显示图表出来.
        plt.close()

        print(f" 收益对比图已保存: {output_file}")

        return output_file

    def plot_momentum_comparison(self):
        """
        绘制所有动量对比图
        显示动量10, 动量20, 动量60 的对比
        使用三种信号的平均收益率来代表不同动量的表现
        """
        if self.df is None:
            print(f" 请先加载数据: Load_data()")
            return

        # 计算每种信号的平均收益率
        signal_avg = self.df.groupby('signal_type')['profit_pct'].mean()

        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # 绘制
        fig, ax = plt.subplots(figsize=(12, 6))

        # 绘制柱状图
        bars = ax.bar(signal_avg.index, signal_avg.values, color=colors, edgecolor='black', linewidth=1.2)

        # 添加数值标题
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height+0.3,
                    f"{height:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 添加零线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        # 添加标签
        ax.set_xlabel('信号类型', fontsize=12)
        ax.set_ylabel(' 平均收益率 (%)', fontsize=12)
        ax.set_title('三种动量信号的平均收益率对比', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 保存图表
        output_file = self.charts_dir / "所有动量对比.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        plt.close()
        print(f" 所有动量对比图已保存: {output_file}")

        return output_file

# ===============================
# 测试
# ===============================
if __name__ == "__main__":
    # 设置目录
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent

    # 创建可视化对象
    visualizer = ResultVisualizer(project_dir)

    # 1. 测试创建目录
    visualizer.setup_directories()
    print(f" \n 第一部分: 测试完成")

    # 2 .加载数据
    df = visualizer.load_data()
    print(f' 第二部分: 测试完成')

    # 3. 绘制净值曲线
    if df is not None:
        visualizer.plot_equity_curve()
        visualizer.plot_return_comparison()
        visualizer.plot_momentum_comparison()
    print(f" \n 第五部分: 测试完成")
    print('=' * 60)
    print(f" 所有图表已生成完毕!")
    print(f" 保存位置: {visualizer.charts_dir}")
    print(f'=' * 60)




"""
## 第5课：绘制结果

**任务目标**：
- 加载回测结果数据
- 绘制净值曲线对比图
- 绘制收益对比柱状图
- 绘制三种动量信号的平均收益率对比图
- 保存所有图表到 charts/ 目录

**实现方案**：
1. **目录管理**：
   - 自动创建 charts/ 目录
   - 检查 results/backtest_results.csv 是否存在

2. **数据加载**：
   - 读取回测结果 CSV 文件
   - 显示数据基本信息（列名、前5行）

3. **净值曲线绘制**：
   - 按股票和信号类型组织数据（pivot 透视表）
   - 三种信号分别用不同颜色柱状图展示
   - 添加初始资金参考线（$30,000）
   - 保存为 净值曲线.png

4. **收益对比柱状图**：
   - 显示每只股票在三种信号下的收益率
   - 在柱状图上添加数值标签
   - 添加零线（区分正负收益）
   - 保存为 收益对比.png

5. **动量对比图**：
   - 计算三种信号的平均收益率
   - 绘制柱状图并添加数值标签
   - 保存为 所有动量对比.png

**核心代码结构**：
```python
class ResultVisualizer:
    ├── __init__()              # 初始化项目目录
    ├── setup_directories()     # 创建 charts/ 目录
    ├── load_data()             # 加载回测结果 CSV
    ├── plot_equity_curve()     # 绘制净值曲线（柱状图）
    ├── plot_return_comparison() # 绘制收益对比柱状图
    └── plot_momentum_comparison() # 绘制动量对比图
```
"""




