'''
第9天
					均线策略回测分析
-计算策略累计收益
-与买入并持有策略对比
-绘制对比图表

练习：
-总结均线策略优缺点
'''

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

class StrategyAnalyzer():
    def __init__(self):
        '''初始化策略分析器'''
        print('\n' + '=' * 80)
        print(f' 第9天 - 方法1: 初始化策略分析器.................')
        print('=' * 80)

        # 1. 获取当前文件所有的目录
        current_dir = Path(__file__).parent
        print(f" 当前文件目录: {current_dir}")

        # 2. 找到项目根目录
        self.project_root = current_dir.parent
        print(f" 项目根目录: {self.project_root}")

        # 3. 设置基础目录
        self.base_dir = self.project_root / "data" / "策略结果"
        print(f" 基础目录: {self.base_dir}")

        # 4. 自动查找最新的回测结果文件夹
        if self.base_dir.exists():
            # 获取所有以"回测结果_" 开头的文件夹
            result_folders = list(self.base_dir.glob('回测结果_*'))

            if result_folders:
                # 按修改时间排序, 取最新
                latest_folder = max(result_folders, key=lambda x: x.stat().st_mtime)
                self.data_dir = latest_folder
                print(f" 找到最新结果: {latest_folder.name}")
                # 统计Excel 文件
                files = list(self.data_dir.glob("*.xlsx"))
                print(f" 目录中有 {len(files)} 个Excel文件")
            else:
                print(f" 没有找到回测结果文件夹")
                self.data_dir = None

        else:
            print(f' 基础目录不存在')
            self.data_dir = None

        # 5. 设置输出目录
        self.output_dir = self.project_root / "data" / "策略分析"
        print(f'输出目录: {self.output_dir}')

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f" 输出目录已创建")
        else:
            print(f" 输出目录已存在")

        # 6. 设置图表目录
        self.charts_dir = self.project_root / "charts" / "策略分析"
        print(f" 图表目录: {self.charts_dir}")

        if not self.charts_dir.exists():
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            print(f" 图表目录已创建")
        else:
            print(f" 图表目录已存在")

        # 7. 配置中文字体
        self._setup_chinese_font()

        # 8. 初始化结果存储
        self.analysis_results = []

        print('\n' + '=' * 70)
        print(f" 初始化完成")
        print(f" 数据目录: {self.data_dir}")
        print(f" 输出目录: {self.output_dir}")
        print(f" 图表目录: {self.charts_dir}")

    def _setup_chinese_font(self):
        """配置中文字体"""
        import matplotlib.font_manager as fm
        import os

        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf'
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                    font_name = fm.FontProperties(fname=path).get_name()
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f' 中文配置字体成功: {font_name}')
                    return
                except:
                    continue
        print(f" 使用默认字体")
        plt.rcParams['axes.unicode_minus'] = False

    def load_detail_data(self, file_path):
        """加载单个股票的详细回测结果数据"""
        print("\n" + "=" * 70)
        print("📹 第9天 - 方法3：加载单个股票详细数据")
        print("=" * 70)

        # 添加设置均线参数
        short_window = 5
        long_window = 20
        print(f" 均线参数: MA{short_window} × MA{long_window}")


        # 1. 显示要加载的文件
        print(f' \n1. 加载文件')
        print(f" 文件路径: {file_path.name}")

        # 2. 检查文件是否存在
        print(f" \n2. 检查文件是否存在")
        if not file_path.exists():
            print(f" 文件不存在: {file_path}")
            return None
        print(f" 文件存在")

        # 3. 读取Excel文件
        print(f" \n3. 读取Excel文件")
        try:
            df = pd.read_excel(file_path)
            print(f" 读取成功")
            print(f" 数据形状: {df.shape[0]} 行 x {df.shape[1]}列")
        except Exception as e:
            print(f" 读取失败: {e}")
            return None

        # 4. 清理列名(去除空格)
        print(f" \n4. 清理列名")
        df.columns = df.columns.str.strip()
        print(f" 列名已清理")

        # 5. 显示列名 ( 前10列)
        print(f" \n5. 显示列名 (前10列)")
        print('-' * 40)
        for i, col in enumerate(df.columns[:10], 1):
            print(f' {i:2d}.{col}')
        if len(df.columns) > 10:
            print(f" 还有{len(df.columns) - 10}")
        print('-' * 40)

        # 6. 检查必要的列是否存在
        print(f" \n6. 检查必要的列")
        required_cols = ['strategy_cumulative', 'benchmark_cumulative']
        missing_cols = []

        for col in required_cols:
            if col in df.columns:
                print(f" {col}: 存在")
            else:
                print(f' {col}: 不存在')
                missing_cols.append(col)

        # ========== 新增：如果缺少必要列，从原始数据重新计算 ==========
        if missing_cols:
            print(f"\n⚠️ 缺少必要列: {missing_cols}")
            print(f" 尝试从原始数据重新计算...")

            # 提取股票代码
            symbol = file_path.stem.split('_')[0].strip()    # 添加 .strip() 去除首尾空格
            print(f" 提取的股票代码: '{symbol}'")  # 看看有没有多余空格
            print(f" 股票代码长度: {len(symbol)}")



            # 查找原始数据文件
            original_dir = self.project_root / "data" / "returns"
            print(f" 原始数据目录: {original_dir}")

            all_files = list(original_dir.glob("*.xlsx"))
            print(f" 目录中所有文件: {[f.name for f in all_files[:5]]}")

            original_files = list(original_dir.glob(f"*{symbol}*_收益率_*.xlsx"))
            print(f" 匹配 {symbol} 的文件: {[f.name for f in original_files]}")
            if not original_files:
                original_files = list(original_dir.glob(f"{symbol}_*.xlsx"))
                print(f" 尝试第二种匹配: {[f.name for f in original_files]}")

            original_file = original_files[0]
            print(f" ✅ 找到原始数据: {original_file.name}")

            # 读取原始数据
            df_original = pd.read_excel(original_file)
            df_original.columns = df_original.columns.str.strip()
            print(f" 原始数据形状: {df_original.shape[0]} 行 x {df_original.shape[1]} 列")

            # 计算均线
            df_original['MA_short'] = df_original['close'].rolling(window=short_window).mean()
            df_original['MA_long'] = df_original['close'].rolling(window=long_window).mean()

            # 生成信号
            print(f" 生成交易信号...")
            df_original['ma_diff'] = df_original['MA_short'] - df_original['MA_long']
            df_original['signal'] = 0
            df_original.loc[(df_original['ma_diff'].shift(1) <= 0) & (df_original['ma_diff']>0), 'signal'] = 1
            df_original.loc[(df_original['ma_diff'].shift(1) >= 0) & (df_original['ma_diff']<0), 'signal'] = -1

            # 计算持仓
            print(f" 计算持仓...")
            position = 0
            positions = []
            for sig in df_original['signal']:
                if sig == 1:
                    position = 1
                elif sig == -1:
                    position = 0
                positions.append(position)
            df_original['position'] = positions

            # 计算策略收益
            print(f" 计算策略收益...")
            if '日收益率' in df_original.columns:
                df_original['strategy_return'] = df_original['position'] * df_original['日收益率']
                df_original['strategy_cumulative'] = (1+df_original['strategy_return']).cumprod()
                df_original['benchmark_cumulative'] = (1+df_original['日收益率']).cumprod()
            else:
                print(f" 没有日收益率列")
                return None

            #  # 替换原df
            df = df_original
            print(f" ✅ 重新计算完成！")
            print(f" 新数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")



        # 7. 检查日期列
        print(f' \n7. 检查日期列')
        date_col = None
        for col in ['Date', 'date', '交易日期', '日期']:
            if col in df.columns:
                date_col = col
                print(f" 找到日期列: {col}")
                break

        if date_col is None:
            print(f" 没有找到日期列 ")

        # 8. 提取股票代码
        print(f" \n8. 提取股票代码")
        symbol = file_path.stem.split('_')[0]
        print(f" 股票代码: {symbol}")


        # 9. 显示数据基本信息
        print(f"\n9. 数据基本信息")
        final_strategy = df['strategy_cumulative'].iloc[-1]
        final_benchmark = df['benchmark_cumulative'].iloc[-1]
        print(f" 最终策略净值: {final_strategy:.4f}")
        print(f" 最终基准净值: {final_benchmark:.4f}")

        # 10. 计算超额收益 (策略 - 基准)
        print(f" \n10. 计算超额收益 (策略 - 基准)")
        df['excess_return'] = df['strategy_cumulative'] - df['benchmark_cumulative']
        final_excess = df['excess_return'].iloc[-1]
        print(f" 最终超额收益: {final_excess:.4f}")

        # 11. 计算相对强弱 (策略 / 基准)
        print(f" \n11. 计算相对强弱 (策略 / 基准)")
        df['relative_strength']= df['strategy_cumulative'] / df['benchmark_cumulative']
        final_rs = df['relative_strength'].iloc[-1]
        print(f" 最终相对强弱: {final_rs:.4f}")

        # 12. 计算跑赢基准的比例
        print(f" \n12. 计算跑赢基准的比例")
        df['outperform'] = df['strategy_cumulative'] > df['benchmark_cumulative']
        outperform_ratio = df['outperform'].mean()
        print(f" 跑赢基准天数: {df['outperform'].sum()} / {len(df)}")
        print(f" 跑赢比例: {outperform_ratio:.2%}")

        # 13. 保存数据到对象属性
        print(f" \n13. 保存数据到对象属性")
        self.current_df = df
        self.current_symbol = symbol
        self.date_col = date_col
        self.current_metrics = {
            'symbol': symbol,
            'final_strategy': final_strategy,
            'final_benchmark': final_benchmark,
            'final_excess': final_excess,
            'final_rs': final_rs,
            'outperform_ratio': outperform_ratio,
            'date_days': len(df)
        }

        print(f' self.current_df 已保存')
        print(f" self.current_symbol = {symbol}")
        print(f' self.date_col = {date_col}')
        print(f" self.current_metrics 已保存")

        # 14. 返回数据
        result = {
            'df': df,
            'symbol': symbol,
            'date_col': date_col,
            'final_strategy': final_strategy,
            'final_benchmark': final_benchmark,
            'final_excess': final_excess,
            'final_rs': final_rs,
            'outperform_ratio': outperform_ratio,
        }

        print("\n" + "-" * 70)
        print("✅ 方法3完成：数据加载成功")
        print("-" * 70)
        print(f"股票代码: {symbol}")
        print(f"数据天数: {len(df)}")
        print(f"策略净值: {final_strategy:.4f}")
        print(f"基准净值: {final_benchmark:.4f}")
        print(f"超额收益: {final_excess:.4f}")
        print(f"跑赢比例: {outperform_ratio:.2%}")

        return result

    def plot_comparison_charts(self, save=True):
        """绘制策略对比图表（净值曲线、超额收益、回撤、相对强弱）"""
        print("\n" + "=" * 70)
        print("📹 第9天 - 方法4：绘制策略对比图表")
        print("=" * 70)

        # 1. 检查是否有数据
        if not hasattr(self, 'current_df'):
            print(f" 没有数据, 请先运行 load_detailed_data()")
            return None

        df = self.current_df
        symbol = self.current_symbol
        date_col = self.date_col

        print(f" 股票代码: {symbol}")
        print(f" 数据天数: {len(df)}")

        # 2. 准备数据 (取最近500天, 避免图表太拥挤)
        if len(df) > 500:
            df_plot = df.tail(500).copy()
            print(f' 取最近500天数据')
        else:
            df_plot = df.copy()

        # 3. 获取日期列
        if date_col and date_col in df_plot.columns:
            x = pd.to_datetime(df_plot[date_col])
            x_label = '日期'
            print(f" 使用日期列: {date_col}")
        else:
            x = range(len(df_plot))
            x_label = '日期'
            print(f' 使用索引作为x轴')

        # 4. 创建2x2 子图
        fig, axes = plt.subplots(2,2, figsize=(15,12))
        fig.suptitle(f"{symbol} - 均线策略回测分析", fontsize=16, fontweight='bold')
        print(f" 创建2x2子图, 大小15x12英寸")

        # 5. 子图1. 净值曲线对比
        print(f" 绘制净值曲线对比图")
        ax1 = axes[0,0]
        ax1.plot(x, df_plot['strategy_cumulative'], linewidth=2, color='green', label='策略净值')
        ax1.plot(x, df_plot['benchmark_cumulative'], linewidth=2, color='blue', label='基准净值')

        final_strategy = df_plot['strategy_cumulative'].iloc[-1]
        final_benchmark = df_plot['benchmark_cumulative'].iloc[-1]
        ax1.text(0.02, 0.95, f"策略净值: {final_strategy:.2f}", transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax1.text(0.02, 0.80, f"基准净值: {final_benchmark:.2f}", transform=ax1.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        ax1.set_title('净值曲线对比', fontsize=12)
        ax1.set_ylabel('净值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        print(f" 净值曲线图完成")

        # 6. 子图2. 超额收益曲线
        print(f" 绘制超额收益曲线图")
        ax2 = axes[0,1]
        df_plot['excess'] = df_plot['strategy_cumulative'] - df_plot['benchmark_cumulative']

        # 正收益绿色，负收益红色
        ax2.fill_between(x, 0, df_plot['excess'], where=(df_plot['excess']>=0), color='green',
                         alpha=0.3)
        ax2.fill_between(x, 0, df_plot['excess'], where=(df_plot['excess']<=0), color='red',
                         alpha=0.3)
        ax2.plot(x, df_plot['excess'], linewidth=1, color='black', alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        final_excess = df_plot['excess'].iloc[-1]
        ax2.text(0.02, 0.95, f"最终超额: {final_excess:.4f}", transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax2.set_title("超额收益曲线", fontsize=12)
        ax2.set_ylabel('超额收益')
        ax2.grid(True, alpha=0.3)
        print(f' 超额收益曲线图完成')

        # 子图3. 回撤对比
        print(f' 绘制回撤对比图')
        ax3 = axes[1,0]

        # 计算策略回撤
        strategy_peak = df_plot['strategy_cumulative'].expanding().max()
        strategy_dd = (df_plot['strategy_cumulative'] - strategy_peak) / strategy_peak
        # 计算基准回撤
        benchmark_peak = df_plot['benchmark_cumulative'].expanding().max()
        benchmark_dd = (df_plot['benchmark_cumulative'] - benchmark_peak) / benchmark_peak

        ax3.fill_between(x, strategy_dd, 0, color='red', alpha=0.3, label='策略回撤')
        ax3.fill_between(x, benchmark_dd, 0, color='blue', alpha=0.3, label='基准回撤')
        ax3.plot(x, strategy_dd, linewidth=1, color='darkred', alpha=0.7)
        ax3.plot(x, benchmark_dd, linewidth=1, color='darkblue', alpha=0.7)

        ax3.set_title('回撤对比', fontsize=12)
        ax3.set_ylabel('回撤幅度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        print(f" 回撤对比图完成")

        # 8. 子图4. 相对强弱曲线
        print("\n   📊 绘制相对强弱曲线图")
        ax4 = axes[1,1]
        df_plot['relative_strength'] = df_plot['strategy_cumulative']/df_plot['benchmark_cumulative']
        ax4.plot(x,df_plot['relative_strength'], linewidth=2, color='purple')
        ax4.axhline(y=1, color='red', linestyle='--', linewidth=1, label='基准线(1.0)')

        final_rs = df_plot['relative_strength'].iloc[-1]
        color = 'green' if final_rs >= 1 else 'red'
        ax4.text(0.02, 0.95, f"最终相对强弱: {final_rs:.4f}", transform=ax4.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
        ax4.set_title('相对强弱曲线（策略/基准）', fontsize=12)
        ax4.set_xlabel(x_label)
        ax4.set_ylabel('相对强弱')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        print(f' 相对强弱曲线完成')

        # 9. 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

        print("\n" + "-" * 70)
        print("✅ 方法4完成：策略对比图表绘制成功")
        print("-" * 70)

        return fig

    def generate_summary_report(self):
        """生成策略优缺点总结报告"""
        print('\n' + '=' * 80)
        print(f" 方法5: 生成策略优缺点总结报告")
        print('=' * 80)

        # 1. 检查是否有数据
        if not hasattr(self, 'current_metrics'):
            print(f' 没有策略指标, ')
            return None

        metrics = self.current_metrics
        symbol = self.current_symbol
        print(f" \n1. 分析股票: {symbol}")

        # 2.提取关键指标
        strategy_return = metrics['final_strategy'] -1
        benchmark_return = metrics['final_benchmark'] - 1
        excess_return = metrics['final_excess']
        outperform_ratio = metrics['outperform_ratio']
        data_days = metrics['date_days']

        print(f" \n2. 策略表现数据: ")
        print(f" 策略总收益: {strategy_return:.2%}")
        print(f" 基准总收益: {benchmark_return:.2%}")
        print(f" 超额收益: {excess_return:.2%}")
        print(f" 跑赢基准比例: {outperform_ratio:.2%}")
        print(f" 数据天数: {data_days}")

        # 3. 策略评估
        print(f" \n3. 策略评估")
        if excess_return > 0:
            print(f" 策略跑赢基准: {excess_return:.2%}")
        else:
            print(f" 策略跑输基准: {abs(excess_return):.2%}")

        if outperform_ratio > 0.5:
            print(f" {outperform_ratio:.1%} 的时间跑赢基准")
        else:
            print(f" 只有{outperform_ratio:.1%} 的时间跑赢基准")

        # 4. 优点
        print(f" \n4. 策略优点")
        if excess_return > 0:
            print(f" 1. 能够获得超额收益, 跑赢买入持有策略 {excess_return:.2%}")
        else:
            print(f" 1. 策略收益为 {strategy_return:.2%}, 与基准差距不大")

        if outperform_ratio > 0.6:
            print(f" 2. 超过{outperform_ratio:.0%} 的时间跑赢基准, 稳定性比较好")
        else:
            print(f" 2. 跑赢基准比例为 {outperform_ratio:.0%}, 有待提升.")

        # 5. 缺点
        print(f" \n5. 策略缺点")
        print(f"   1. 滞后性：均线是历史数据的平均值，对价格变化反应滞后")
        print(f"   2. 震荡市表现差：在横盘震荡行情中容易产生频繁的错误信号")
        print(f"   3. 参数敏感：不同股票需要不同的参数优化")
        print(f"   4. 无法预测极端行情：在突发性大涨大跌时反应不及时")

        if outperform_ratio < 0.5:
            print(f" 5. 跑赢基准天数不足一半 ({outperform_ratio:.0%}), 策略稳定性差")

        return {
            'symbol': symbol,
            'strategy_return': strategy_return,
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'outperform_ratio': outperform_ratio
        }

    def save_all_analysis(self, save_charts=True, save_report=True):
        """保存所有分析结果"""
        print("\n" + "=" * 80)
        print(f' 方法6: 保存所有分析结果')
        print('=' * 80)

        if not hasattr(self, 'current_df'):
            print(f" 没有数据")
            return None

        symbol = self.current_symbol
        saved_files= []

        # 1. 保存图表
        if save_charts:
            print(f" \n1. 保存策略对比图表")
            # 确保目录存在
            self.charts_dir.mkdir(parents=True, exist_ok=True)

            fig = self.plot_comparison_charts(save=False)
            if fig:
                chart_path = self.charts_dir / f"{symbol}_策略对比分析.png"
                fig.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(str(chart_path))
                print(f" 图表已保存: {chart_path}")

        # 2. 保存报告
        if save_report:
            print(f"\n2. 保存策略分析报告")

            # 提取指标
            metrics = self.current_metrics
            strategy_return = metrics['final_strategy'] - 1
            benchmark_return = metrics['final_benchmark'] - 1
            excess_return = metrics['final_excess']
            outperform_ratio = metrics['outperform_ratio']
            data_days = metrics['date_days']

            # 判断表现
            if excess_return > 0:
                performance_text = f"策略跑赢基准 {excess_return:.2%}"
            else:
                performance_text = f"策略跑输基准 {abs(excess_return):.2%}"

            # 优点
            advantages = []
            if excess_return > 0:
                advantages.append(f"能够获得超额收益, 跑赢买入持有策略 {excess_return:.2%}")
            else:
                advantages.append(f"策略收益为 {strategy_return:.2%}, 与基准差距不大")

            if outperform_ratio > 0.6:
                advantages.append(f"超过{outperform_ratio:.0%}的时间跑赢基准, 稳定性较好")

            # 缺点
            disadvantages = [
                "  • 滞后性：均线是历史数据的平均值，对价格变化反应滞后",
                "  • 震荡市表现差：在横盘震荡行情中容易产生频繁的错误信号",
                "  • 参数敏感：不同股票需要不同的参数优化",
                "  • 无法预测极端行情：在突发性大涨大跌时反应不及时"
            ]
            if outperform_ratio < 0.5:
                disadvantages.insert(0, f"跑赢基准天数不足一半({outperform_ratio:.0%}), 策略稳定性差")


            # 保存到文件
            report_path = self.output_dir / f"{symbol}_策略分析报告.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write(f"均线策略分析报告 - {symbol}\n")
                f.write("=" * 70 + "\n\n")

                f.write("1. 策略表现数据\n")
                f.write("-" * 50 + "\n")
                f.write(f"策略总收益: {strategy_return:.2%}\n")
                f.write(f"基准总收益: {benchmark_return:.2%}\n")
                f.write(f"超额收益: {excess_return:.2%}\n")
                f.write(f"跑赢基准比例: {outperform_ratio:.2%}\n")
                f.write(f"数据天数: {data_days} 天\n\n")

                f.write("2. 策略评估\n")
                f.write("-"* 50 + "\n")
                f.write(f"{performance_text}\n")
                f.write(f"{outperform_ratio:.1%} 的时间跑赢基准\n\n")

                f.write("3. 策略优点\n")
                f.write("-" * 50 + "\n")
                for adv in advantages:
                    f.write(adv + "\n")
                f.write("\n")

                f.write("4. 策略优点\n")
                f.write("-" * 50 + "\n")
                for dis in disadvantages:
                    f.write(dis + "\n")
                f.write("\n")

            saved_files.append(str(report_path))
            print(f" 报告已保存: {report_path}")

        # 3. 显示保存结果
        print("\n" + "-" * 80)
        print(f" 方法6 完成: 所有分析结果已保存")
        print("-" * 80)
        print(f" 保存目录:")
        print(f" 图表: {self.charts_dir}")
        print(f" 报告: {self.output_dir}")
        print(f" 共保存: {len(saved_files)} 个文件")

        return saved_files



# 添加调用
if __name__ == "__main__":
    analyer = StrategyAnalyzer()

    detail_folder = analyer.data_dir / "各股票详细结果"
    all_files = list(detail_folder.glob("*_详细结果.xlsx"))
    print(f" \n找打{len(all_files)} 个股票文件")


    # 循环
    for file_path in all_files:
        data = analyer.load_detail_data(file_path)

        if data:
            analyer.plot_comparison_charts(save=True)
            report = analyer.generate_summary_report()
            if report:
                print(f"\n📋 报告摘要:")
                print(f"   股票: {report['symbol']}")
                print(f"   策略收益: {report['strategy_return']:.2%}")
                print(f"   超额收益: {report['excess_return']:.2%}")

            # 保存所有分析结果
            saved_files = analyer.save_all_analysis(save_charts=True, save_report=True)
            if saved_files:
                print(f" \n 所有结果已保存, 共{len(saved_files)}个文件")


"""
## 第9天：均线策略回测分析

**任务目标**：
- 计算策略累计收益并与买入并持有策略对比
- 绘制净值曲线、超额收益、回撤、相对强弱对比图表
- 生成策略优缺点总结报告
- 总结均线策略的适用场景和局限性

**实现方案**：
1. **数据源管理**：
   - 自动定位第8天生成的最新回测结果文件夹
   - 扫描“各股票详细结果”子目录下的所有Excel文件
   - 智能加载策略累计收益和基准累计收益数据
   - 若缺少必要列，自动从原始数据重新计算均线策略

2. **核心对比指标计算**：
   - **超额收益**：策略净值 - 基准净值（买入持有策略）
   - **相对强弱**：策略净值 / 基准净值（>1表示跑赢）
   - **跑赢比例**：策略跑赢基准的交易天数占比
   - **回撤对比**：分别计算策略和基准的最大回撤

3. **可视化分析系统**（2×2子图布局）：
   - **左上**：净值曲线对比图（策略净值 vs 基准净值）
   - **右上**：超额收益曲线图（正收益绿色填充，负收益红色填充）
   - **左下**：回撤对比图（策略回撤 vs 基准回撤）
   - **右下**：相对强弱曲线图（标注基准线1.0）

4. **策略评估报告**：
   - **表现数据**：策略总收益、基准总收益、超额收益、跑赢比例
   - **策略优点**：超额收益能力、稳定性评估
   - **策略缺点**：滞后性、震荡市表现差、参数敏感、极端行情反应慢
   - 自动保存为TXT格式报告文件

5. **批量分析处理**：
   - 循环处理所有股票的详细结果文件
   - 为每个股票生成对比图表和分析报告
   - 自动保存图表到charts/策略分析/目录
   - 自动保存报告到data/策略分析/目录

**核心代码结构**：
```python
class StrategyAnalyzer:
    ├── __init__()                       # 初始化，定位最新回测结果
    ├── _setup_chinese_font()             # 配置中文字体
    ├── load_detail_data()                # 加载单个股票详细数据
    ├── plot_comparison_charts()          # 绘制策略对比图表（2×2）
    ├── generate_summary_report()         # 生成策略优缺点总结报告
    └── save_all_analysis()               # 保存所有分析结果
```
"""
