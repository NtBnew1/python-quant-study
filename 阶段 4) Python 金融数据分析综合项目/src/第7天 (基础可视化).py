'''
第7天
					基础可视化
-绘制价格曲线
-绘制收益率分布
-绘制回撤曲线

练习：
-整理一组基础分析图表
'''
import os.path


# 导入库
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import bbox_artist


class DataVisualizer:
    '''这是第7天的: 基础可视化'''
    def __init__(self):
        """第一步: 初始化可视化"""
        print("=" * 70)
        print(f" 可视化初始化")
        print("=" * 70)

        # 1. 获取当前文件目录
        print(f" \n1. 获取当前文件目录........")
        current_dir = Path(__file__).parent
        print(f" 当前文件目录: {current_dir}")

        # 2. 找到项目根目录
        print(f" \n2. 找到项目根目录......")
        self.project_root = current_dir.parent
        print(f" 项目根目录: {self.project_root}")

        # 3. 设置数据目录
        print(f" \n3. 设置数据目录.......")
        self.data_dir = self.project_root / "data" / "returns"
        print(f" 数据目录: {self.data_dir}")

        # 检查数据目录是否存在
        if self.data_dir.exists():
            print(f" 数据目录存在!")
            # 统计文件数量
            files = list(self.data_dir.glob("*.xlsx"))
            print(f" 目录中有{len(files)}个Excel文件")
        else:
            print(f" 数据目录不存在")

        # 4. 设置图表输出目录
        print(f" \n4. 设置图表输出目录.......")
        self.charts_dir = self.project_root / "charts"
        print(f" 图表输出目录: {self.charts_dir}")

        # 5. 创建图表目录
        print(f' \n5. 创建图表目录........')
        if not self.charts_dir.exists():    #这些代码是如果charts不存在,
            print(f" 目录不存在, 正在创建")
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            print(f" 目录已创建: {self.charts_dir}")
        else:
            print(f" 目录已存在: {self.charts_dir}")

        self.font_prop = self._setup_chinese_font()

        plt.style.use('seaborn-v0_8-darkgrid')
        print('初始化完成\n')


        # 8. 初始化完成
        print("\n" + "-" * 70)
        print(f" 初始化成功")
        print("-" * 70)
        print(f"当前对象属性")
        print(f" self.project_root = {self.project_root}")
        print(f" self.data_dir = {self.data_dir}")
        print(f" self.charts_dir = {self.charts_dir}")
        print(f"=" * 70)

    def _setup_chinese_font(self):
        '''配置中文字体, 返回FontProperties对象'''
        font_paths = [
            'C:/Windows/Fonts/Msyh.ttc',        # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',      # 黑体
            'C:/Windows/Fonts/simsun.ttf',      # 宋体
            'C:/Windows/Fonts/simkai.ttf',      # 楷体
        ]

        selected_font = None
        for path in font_paths:
            if os.path.exists(path):
                selected_font = path
                break

        if selected_font is None:
            print(f" 未找到中文字体文件, 中文可能显示为方块")
            plt.rcParams['axes.unicode_minus'] = False
            return None

        try:
            # 添加字体文件到管理器
            fm.fontManager.addfont(selected_font)
            # 获取真实字体名称
            prop = fm.FontProperties(fname=selected_font)
            font_name = prop.get_name()
            # 设置全局默认字体(后备)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f" 中文字体设置成功: {font_name}")
            return prop
        except Exception as e:
            print(f" 字体设置失败: {e}")
            plt.rcParams['axes.unicode_minus'] = False
            return None

    def find_data_files(self):
        """第二步: 查找数据文件"""
        # 1. 检查数据目录
        print(f" 数据目录: {self.data_dir}")

        if not self.data_dir.exists():
            print(f" 数据目录不存在")
            return None

        print(f"数据目录存在")

        # 2.查找所有excel文件
        print(f" \n2. 查找所有excel文件")
        print(f" 执行: list(self.data_dir.glob('*.xlsx'))")

        excel_files = list(self.data_dir.glob("*.xlsx"))
        print(f" 找到{len(excel_files)}个excel文件")

        # 3. 如果没有找到文件
        if len(excel_files) == 0:
            print(f" \n3. 没有找到excel文件")
            print(f" 可能的原因:")
            print(f" -文件格式不是.xlsx")
            print(f" -文件在错误的目录")
            return None

        # 4. 保存文件列表
        print(f" \n4. 保存文件列表到对象属性.......")
        self.all_files = excel_files
        print(f" self.all_files = 包含 {len(self.all_files)}个文件")

        # 5. 显示文件列表
        print(f" \n5. 显示文件列表 (前10个):")
        print('=' * 70)
        for i, file in enumerate(self.all_files[:10], 1):
            # 获取数据大小:    这个没什么用的代码只是显示数据大小
            file_size = file.stat().st_size / 1024  # 转为KB
            # 从文件名提取股票代码
            filename = file.stem
            parts = filename.split('_')
            symbol = parts[0] if len(parts) > 0 else filename
            print(f" {i:2d}. {file.name}")
            print(f" 股票代码: {symbol}")
            print(f" 文件大小: {file_size:.1f}KB")
            print(f" 修改时间: {pd.Timestamp(file.stat().st_mtime, unit='s').strftime('%Y-%m-%d')}")

        if len(self.all_files) > 10:
            print(f" ......还有{len(self.all_files) - 10}个文件未显示")

        # 6. 选择第一个文件作为测试
        print(f" \n5. 选择第一个文件作为测试数据.........")
        self.test_file = self.all_files[0]          # 刚刚换成3,
        print(f" 选择的文件: {self.test_file.name}")

        # 显示选择的文件信息
        file_size = self.test_file.stat().st_size / 1024
        filename = self.test_file.stem
        parts = filename.split('_')
        self.symbol= parts[0] if len(parts) > 0 else filename

        print(f" \n 测试文件详情")
        print(f" 文件名: {self.test_file.name}")
        print(f" 股票代码: {symbol}")
        print(f" 文件大小: {file_size}")
        print(f" 完整路径: {self.test_file}")

        # 7. 返回结果
        print('\n' + '-' * 70)
        print(f" 视频2完成: 数据文件查找成功")
        print('-' * 70)
        print(f"总共找到: {len(self.all_files)} 个文件")
        print(f"测试文件: {self.test_file.name}")
        print(f"股票代码: {self.symbol}")

        return self.all_files

    def load_test_data(self):
        """加载测试数据"""
        print("\n" + "=" * 70)
        print(f" 视频3. 加载测试数据")
        print("=" * 70)

        # 1. 检查测试文件
        print("\n🔍 第1步：检查测试文件")
        if not hasattr(self, 'test_file'):
            print(f" 没有测试文件")
            print(f" 请先运行 find_data_files()")
            return None

        print(f" 测试文件: {self.test_file.name}")
        print(f" 股票代码: {self.symbol}")          # 刚刚代码和前面的不一致所以没办法用.

        # 2. 读取Excel 文件
        print(f" \n 第二步: 读取Excel 文件")
        print(f" 执行: pd.read_excel('{self.test_file.name}')")

        try:
            df = pd.read_excel(self.test_file)
            print(f" 读取成功")
            print(f" 数据形状: {df.shape[0]}行 x {df.shape[1]}列")
        except Exception as e:
            print(f" 读取失败: {e}")
            return None

        # 3. 查看列名
        print(f" \n 第3步: 查看数据列")
        print('-' * 40)
        for i, col in enumerate(df.columns[:8], 1):
            print(f" {i:2d}. {col}")
        if len(df.columns) > 8:
            print(f" ...还有 {len(df.columns) - 8}列")
        print('-' * 40)

        # 4. 检查必要列
        print(f" \n 第4步: 检查必要列")
        need_cols = ['close', '日收益率', '回撤']
        for col in need_cols:
            if col in df.columns:
                # 检查数据是否有空值
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    print(f" {col}: 存在(有{null_count}个空值)")
                else:
                    print(f" {col}: 存在(无空值)")
            else:
                print(f" {col}: 不存在")

        # 5. 检查日期列
        print(f" \n 第5步: 检查日期列")
        date_col = None
        for col in ['Date', 'date', '交易日期', '日期']:
            if col in df.columns:
                date_col = col
                print(f" 找到日期列: {col}")

                # 显示日期范围
                try:
                    dates = pd.to_datetime(df[col])
                    print(f" 日期范围: {dates.min().strftime('%Y-%m-%d')} 到{dates.max().strftime('%Y-%m-%d')}")
                    print(f" 总天数: {(dates.max() - dates.min()).days}天")
                except:
                    print(f" 无法转换为日期格式")
                break

        if date_col is None:                    # 把in 改成 is
            print(f" 没有找到日期列")

        # 6. 查看数据预览
        print(f" \n 第6步: 查看数据预览 (前3行)")
        print('-' * 60)
        print(df.head(3).to_string())
        print('-' * 60)

        # 7. 保存数据到对象
        print(f" \n 第7步: 保存数据")
        self.current_df = df
        self.current_symbol = self.symbol       # 这里用 self.symbol 赋值给 self.current_symbol
        self.date_col = date_col
        print(f"   ✅ self.current_df = 数据已保存")
        print(f"   ✅ self.current_symbol = {self.current_symbol}")
        print(f"   ✅ self.date_col = {self.date_col}")

        print("\n" + "=" * 70)
        print("✅ 视频3完成：数据加载成功")
        print("=" * 70)
        print(f"股票代码: {self.current_symbol}")
        print(f"数据形状: {df.shape[0]}行 × {df.shape[1]}列")
        print(f"日期列: {self.date_col if self.date_col else '无'}")

        return {
            'df': df,
            'symbol': self.current_symbol,
            'date_col': date_col
        }

    def process_all_files(self):
        """循环处理所有文件"""
        print("\n"+"="*70)
        print(f" 视频4. 循环处理所有文件")
        print("="*70)

        # 1. 检查文件列表
        print(f" \n1. 检查文件列表")
        if not hasattr(self, 'all_files'):
            print(f" 没有文件列表")
            return None

        total = len(self.all_files)
        print(f" 共有{total}个文件")

        # 2. 初始化计数器
        print(f" \n2. 初始化计数器")
        success = 0
        failed = 0
        drawdown_success = 0
        drawdown_failed = 0
        return_success = 0
        return_failed = 0
        print(f" success = {success}")
        print(f" failed = {failed}")
        print(f" drawdown_success = {drawdown_success}")
        print(f" drawdown_failed = {drawdown_failed}")
        print(f" return_success = {return_success}")
        print(f" return_failed = {return_failed}")

        # 3. 开始循环
        print(f" \n3. 开始循环处理")
        print('-' * 50)
        for i, file in enumerate(self.all_files, 1):
            print(f" \n[{i}/{total}] 处理: {file.name}")

            '''就在这里修改代码. 把plot从单图换成循环图'''
            # 调用价格曲线绘图方法
            if self.plot_price_curve(file):
                success += 1
                print(f" 价格曲线绘制成功")
            else:
                failed += 1
                print(f" 价格曲线绘制失败")

            # 调用回撤曲线绘制图方法
            if self.plot_drawdown_curve(file):
                drawdown_success += 1
                print(f" 回撤曲线绘制成功")
            else:
                drawdown_failed += 1
                print(f" 回撤曲线绘制失败")

            # 调用绘制收益率分布图
            if self.plot_return_distribution(file):
                return_success += 1
                print(f" 收益率分布绘制成功")
            else:
                return_failed += 1
                print(f" 收益率分布绘制失败")

        # 4. 显示结果
        print("\n" + "-" * 50)
        print(f" 4. 处理结果")
        print(f" 总文件: {total}")
        print(f" 成功: {success}")
        print(f" 失败: {failed}")
        print(f" 成功率: {success/total*100:.1f}%")
        print(f" 视频4完成")


    def plot_price_curve(self, file_path, save=True, show=False):
        """绘制单个股票的价格曲线"""
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip() # 清理列名空格
        except Exception as e:
            print(f" 读取失败: {e}")
            return False

        symbol = file_path.stem.split('_')[0]

        # 查找收盘价列
        close_col = next((c for c in ['close', 'Close', '收盘价', '收盘', 'CLOSE'] if c in df.columns), None)
        if close_col is None:
            print(f" {symbol}没有找到收盘价列")
            return False

        # 查找日期
        date_col = next((c for c in ['Date', 'date', '交易日期', '日期'] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            x = df[date_col]
            x_label = '日期'
        else:
            x = range(len(df))
            x_label = '交易日'

        # 查找成交量列
        volume_col = next((c for c in ['volume', 'Volume', '成交量', 'VOLUME'] if c in df.columns), None)

        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12,8))

        # 使用fontproperties 显示指定中文字体.    之前的字体都不能显示, 现在只能指定添加
        title_font = self.font_prop if self.font_prop else None
        fig.suptitle(f"{symbol} - 价格曲线", fontproperties=title_font, fontsize=14, fontweight='bold')

        # 上图: 收盘价走势
        ax1 = axes[0]
        ax1.plot(x, df[close_col], linewidth=1.5, color='blue')
        ax1.set_ylabel('收盘价', fontproperties=title_font)
        ax1.set_title('收盘价走势', fontproperties=title_font)
        ax1.set_xlabel(x_label, fontproperties=title_font)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)

        # 下图: 成交量
        ax2 = axes[1]
        if volume_col:
            ax2.bar(x, df[volume_col], alpha=0.6, color='green')
            ax2.set_ylabel('成交量', fontproperties=title_font)
            ax2.set_title('成交量', fontproperties=title_font)
        else:
            ax2.text(0.5, 0.5, '无成交量数据', ha='center', va='center', fontproperties=title_font)
            ax2.set_title('成交量 (无数据)', fontproperties=title_font)

        ax2.set_xlabel(x_label, fontproperties=title_font)
        ax2.grid(True)

        plt.tight_layout()

        # 根据数据决定保存还是显示
        if save:
            self.save_chart(fig, symbol, "价格曲线")
        elif show:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)  # 不存在也不显示, 就直接关闭

        return True

    def plot_drawdown_curve(self, file_path, save=True, show=False):
        """绘制回撤曲线及相关风险指标"""
        print(f" \n 绘制回撤曲线: {file_path.name}")

        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f' 读取失败: {e}')
            return False

        symbol = file_path.stem.split('_')[0]

        # 检查必要列
        if '回撤' not in df.columns:
            print(f" 未找到'回撤'列, 无法绘制回测曲线")
            return False

        # 查找净值列(优先使用'净值', 否则用累计收益率+1)
        nav_col = '净值' if '净值' in df.columns else None
        if nav_col is None and '累计收益率' in df.columns:
            # 从累计收益率计算净值
            df['净值'] = 1 + df['累计收益率']
            nav_col = '净值'
        elif nav_col is None:
            print(f" 未找到净值数据, 将只显示回测曲线")

        # 查找日期列
        date_col = next((c for c in ['Date', 'date', '交易日期', '日期'] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            x = df[date_col]
            x_label = '日期'
        else:
            x = range(len(df))
            x_label = '日期'

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(14,12))
        title_font = self.font_prop if self.font_prop else None
        fig.suptitle(f" {symbol} - 风险分析", fontproperties = title_font, fontsize=14, fontweight='bold')

        # 1. 回测曲线
        ax1 = axes[0,0]
        drawdown = df['回撤']
        ax1.fill_between(x, drawdown, 0, where=(drawdown < 0), color='red', alpha=0.3, interpolate=True)
        ax1.plot(x, drawdown, linewidth=1, color='darkred')
        ax1.set_title('回撤曲线', fontproperties=title_font)
        ax1.set_ylabel('回撤幅度', fontproperties=title_font)
        ax1.set_xlabel(x_label, fontproperties=title_font)
        if date_col:
            ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True)

        # 2. 最大回测
        ax2 = axes[0, 1]
        if '最大回撤' in df.columns:
            max_dd = df['最大回撤']
            ax2.plot(x, max_dd, linewidth=2, color='darkorange', label='最大回撤')
            min_dd = max_dd.min()
            ax2.axhline(y=min_dd, color='red', linestyle='--', linewidth=1, label=f'最大回撤:{min_dd:.2%}')
        else:
            # 如果没有最大回测列, 用回测的滚动最小值
            running_max_drawdown = drawdown.expanding().min()
            ax2.plot(x, running_max_drawdown, linewidth=2, color='darkorange', label='历史最大回撤')
            min_dd = running_max_drawdown.min()
            ax2.axhline(y=min_dd, color='red', linestyle='--', linewidth=1, label=f'最大回撤:{min_dd:.2%}')

        ax2.set_title('最大回撤曲线', fontproperties=title_font)
        ax2.set_ylabel('最大回撤', fontproperties=title_font)
        ax2.set_xlabel('x_label', fontproperties=title_font)
        if date_col:
            ax2.tick_params(axis='x', rotation=45)
        ax2.legend(prop=title_font)
        ax2.grid(True)

        # 3. 净值曲线
        ax3 = axes[1,0]
        if nav_col:
            ax3.plot(x, df[nav_col], linewidth=2, color='green')
            #标记净值最高点
            max_nav_idx = df[nav_col].idxmax()
            ax3.scatter(x[max_nav_idx], df[nav_col].max(), color='red', s=50, zorder=5, label='最高点')
            ax3.set_title('净值曲线', fontproperties=title_font)
            ax3.set_ylabel('净值', fontproperties=title_font)
        else:
            ax3.text(0.5, 0.5, '无净值数据', ha='center', va='center', transform=ax3.transAxes,
                     fontproperties=title_font)
            ax3.set_title('净值曲线(无数据)', fontproperties=title_font)
        ax3.set_xlabel(x_label, fontproperties=title_font)
        if date_col:
            ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True)

        # 4. 风险统计
        ax4 = axes[1,1]
        ax4.axis('off')
        # 计算统计
        total_days = len(df)
        drawdown_days = (drawdown < 0).sum()
        max_dd_value = drawdown.min()
        avg_dd = drawdown.mean()

        # 最大回测的结束位置
        end_idx = drawdown.idxmin()
        # 从开始到结束，寻找净值最高点作为开始
        if nav_col and pd.notna(end_idx):
            # 取净值数据
            nav_series = df[nav_col]
            # 再end_idx之前找最大值位置
            pre_nav = nav_series.iloc[:end_idx+1]
            start_idx = pre_nav.idxmax()
            start_date = df[date_col].iloc[start_idx] if date_col else f"第{start_idx+1}天"
            end_date = df[date_col].iloc[end_idx] if date_col else f"第{end_idx+1}天"
            recovery_info = f" 回测期间: {start_date} 至{end_date}\n"
        else:
            recovery_info = ""

        stats_text = f" 风险统计"
        stats_text += f" 最大回测: {max_dd_value:.2%}\n"
        stats_text += f" 平均回测: {avg_dd:.2%}\n"
        stats_text += f" 回撤天数: {drawdown_days} / {total_days}\n"
        stats_text += f" 回撤占比: {drawdown_days/total_days:.1%}\n"
        stats_text += recovery_info
        if nav_col:
            final_nav = df[nav_col].iloc[-1]
            stats_text += f" 最终净值: {final_nav:.4f}"
            total_return = (final_nav - 1) * 100
            stats_text += f" 总收益率: {total_return:.2f}%"

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontproperties=title_font, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1, # 左边距
            bottom=0.1, # 下边距（为x_label留出空间）
            right=0.95,  # 右边距
            top=0.88, # 上边距（为总标题留出空间）
            wspace=0.3, # 子图之间的水平间距
            hspace=0.4 # 子图之间的垂直间距（增加这个值可以防止标题重叠）
        )

        # 根据参数决定保存还是显示
        if save:
            self.save_chart(fig, symbol, "回撤曲线")
        elif show:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
        return True

    def plot_return_distribution(self, file_path, save=True, show=False):
        """绘制收益率分布图"""
        print(f'\n 绘制收益率分布图: {file_path.name}')
        try:
            df = pd.read_excel(file_path)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f" 读取失败: {e}")
            return False

        symbol = file_path.stem.split('_')[0]

        # 查找收益率列
        return_col = next((c for c in ['日收益率', 'daily_return', 'return', '收益率'] if c in df.columns), None)
        if return_col is None:
            print(f" 未找到收益率")
            return False

        returns = df[return_col].dropna()
        if len(returns) < 2:
            print(f' 收益率数据不足')
            return False

        # 查找日期列
        date_col = next((c for c in ['Date', 'date', '交易日期', '日期'] if c in df.columns), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        # 创建2x2子图
        fig, axes = plt.subplots(2,2, figsize=(14, 12))
        title_font = self.font_prop if self.font_prop else None
        fig.suptitle(f"{symbol}-收益率分布", fontproperties=title_font, fontsize=14, fontweight='bold')

        # 1. 直方图
        ax1 = axes[0,0]
        ax1.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(x=returns.mean(), color='red', linestyle='--', label=f"均值:{returns.mean():.4f}")
        ax1.axvline(x=0, color='green', linestyle='-')
        ax1.set_title('收益率直方图', fontproperties=title_font)
        ax1.set_xlabel('日收益率', fontproperties=title_font)
        ax1.set_ylabel('频数', fontproperties=title_font)
        ax1.legend(prop=title_font)
        ax1.grid(True)

        # 2. 收益率时间序列
        ax2 = axes[0, 1]
        if date_col:
            ax2.plot(df[date_col], df[return_col], linewidth=1, color='green', alpha=0.7)
            ax2.set_xlabel('日期', fontproperties=title_font)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.plot(returns, linewidth=1, color='green', alpha=0.7)
            ax2.set_xlabel('交易日', fontproperties=title_font)
        ax2.axhline(y=0, color='red', linestyle='-')
        ax2.set_title('收益率时间序列', fontproperties=title_font)
        ax2.set_ylabel('日收益率', fontproperties=title_font)
        ax2.grid(True)

        # 3. 累计收益率
        ax3 = axes[1,0]
        if '累计收益率' in df.columns:
            cumulative = df['累计收益率']
        else:
            cumulative= (1 + returns).cumprod()
        ax3.plot(cumulative, linewidth=2, color='red')
        ax3.set_title('累计收益率曲线', fontproperties=title_font)
        ax3.set_xlabel('交易日', fontproperties=title_font)
        ax3.set_ylabel('累计收益率', fontproperties=title_font)
        ax3.grid(True)

        # 4. 统计信息
        ax4 = axes[1,1]
        ax4.axis('off')
        stats_text = f" 收益率统计\n\n"
        stats_text += f"数量: {len(returns)}\n"
        stats_text += f"均值: {returns.mean():.4f}\n"
        stats_text += f"标准差: {returns.std():.4f}\n"
        stats_text += f"最大值: {returns.max():.4f}\n"
        stats_text += f"最小值: {returns.min():.4f}\n"
        stats_text += f"正收益: {(returns > 0).sum()}\n"
        stats_text += f"负收益: {(returns < 0).sum()}\n"
        stats_text += f"胜率: {(returns > 0).sum()/len(returns):.1%}\n"

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                 fontproperties=title_font, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.1,  # 左边距
            bottom=0.1,  # 下边距（为x_label留出空间）
            right=0.95,  # 右边距
            top=0.8,  # 上边距（为总标题留出空间）
            wspace=0.3,  # 子图之间的水平间距
            hspace=0.4  # 子图之间的垂直间距（增加这个值可以防止标题重叠）
        )

        if save:
            self.save_chart(fig, symbol, '收益率分布')
        elif show:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)


        return True

    '''现在就是单独写一个def 用来保存这些绘制图'''
    def save_chart(self, fig, symbol, chart_type):
        # 生成文件名
        filename = f"{symbol}_{chart_type}.png"
        save_path = self.charts_dir / filename

        # 保存图表
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图表释放内存

        print(f" 图表已保存: {save_path}")
        return save_path

    '''还没完.  需要去每一个plot 修改'''

# ----------------------主程序----------------------
if __name__ == "__main__":
    # 1. 创建可视化器对象 (调用视频1.的初始化)
    visualizer = DataVisualizer()

    # 2. 调用find_data_files方法 (视频2.)
    files = visualizer.find_data_files()


    if files:
        # 3. 运行Load_test_data
        visualizer.load_test_data()
        # 4. 循环处理所有文件
        visualizer.process_all_files()

        # 单独测试显示一个文件(设置save=False, show=True)
        # visualizer.plot_price_curve(visualizer.test_file, save=False, show=True)
        # visualizer.plot_drawdown_curve(visualizer.test_file, save=False, show=True)
        # visualizer.plot_return_distribution(visualizer.test_file, save=False, show=True)
        '''单独测试不需要.  除非你只想绘制其中一种. '''
