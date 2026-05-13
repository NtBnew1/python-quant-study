'''
第8天
					策略一 —— 均线策略
-设计均线交易逻辑
-计算交易信号
-生成策略收益

练习：
-测试不同均线参数组合
'''

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from jupyter_core.version import parts
from matplotlib.patches import bbox_artist


class MovingAverageStrategy:
    """均线策略"""
    def __init__(self, short_window=5, long_window=20):
        """初始化均线策略

                Args:
                    short_window: 短期均线窗口（默认5天）
                    long_window: 长期均线窗口（默认20天）
        """
        print("=" * 70)
        print(f" 初始化均线策略")
        print("=" * 70)

        # 1. 设置策略参数
        print(f" \n1. 设置策略参数")
        self.short_window = short_window
        self.long_window = long_window
        print(f" 短均线窗口: {self.short_window}天")
        print(f" 长均线窗口: {self.long_window}天")

        # 检查参数是否合理
        if self.short_window >= self.long_window:
            print(f" 警告: 短期窗口应该小于长期窗口")

        # 2. 获取当前文件目录
        print(f" \n2. 获取当前文件目录")
        current_dir = Path(__file__).parent
        print(f" 当前文件目录: {current_dir}")

        # 3. 找到项目根目录
        print(f" \n3. 找到项目根目录")
        self.project_root = current_dir.parent
        print(f" 项目根目录: {self.project_root}")

        # 4. 设置数据目录 ( 使用第5天的收益率数据)
        print(f" \n4. 设置数据目录")
        self.data_dir = self.project_root / "data" / 'returns'
        print(f' 数据目录: {self.data_dir}')

        # 5. 设置结果输出目录
        print(f" \n5. 设置结果输出目录")
        self.output_dir = self.project_root / 'data' / '策略结果'
        print(f" 输出目录: {self.output_dir}")

        # 创建输出目录
        if not self.output_dir.exists():
            print(f' 目录不存在, 正在创建.........')
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f" 目录已创建")
        else:
            print(f' 目录已存在')

        # 6. 设置图表输出目录
        print(f" \n6. 设置图表输出目录")
        self.charts_dir = self.project_root / "charts" / "策略图表"
        print(f" 图表目录: {self.charts_dir}")

        # 创建图表目录
        if not self.charts_dir.exists():
            print(f" 目录不存在, 正在创建.......")
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            print(f" 目录已创建")
        else:
            print(f" 目录已存在")

        # 7. 初始化结果存储
        print(f" \n7. 初始化结果存储")
        self.results = []           # 存储所有股票的策略结果
        self.trade_records = []     # 存储交易记录
        print(f"   ✅ self.results = [] (用于存储策略结果)")
        print(f"   ✅ self.trade_records = [] (用于存储交易记录)")

        # 8. 配置中文字体 (可以从第7天里copy)
        print(f"\n8. 配置中文字体")
        self._setup_chinese_font()

        print("\n" + "-" * 70)
        print("✅ 视频1完成：初始化成功")
        print("-" * 70)
        print(f"策略参数: MA{self.short_window} × MA{self.long_window}")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"图表目录: {self.charts_dir}")
        print("=" * 70)

    def _setup_chinese_font(self):
        """配置中文字体（复用第7天的代码）"""
        import matplotlib.font_manager as fm
        import os

        font_paths = [
            'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                    font_name = fm.FontProperties(fname=path).get_name()
                    plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False
                    print(f" 中文字体配置成功: {font_name}")
                    return
                except:
                    continue

        print(f" 未找到中文字体, 使用默认字体")
        plt.rcParams['axes.unicode_minus'] =False

    def find_data_files(self):
        """第二步, 查找收益率数据文件"""
        print('\n' + '=' * 70)
        print(f" 视频2, 查找数据文件")
        print('=' * 70)

        # 1. 检查数据目录是否存在
        print(f" \n1. 检查数据目录........")
        print(f" 数据目录: {self.data_dir}")

        if not self.data_dir.exists():
            print(f" 数据目录不存在")
            return None

        print(f" 数据目录存在")

        # 2. 查找所有Excel文件
        print(f" \n2. 查找所有Excel文件")
        print(f" 请执行: list(self.data_dir.glob('*.xlsx'))")

        excel_files = list(self.data_dir.glob('*.xlsx'))
        print(f" 找到{len(excel_files)}个Excel文件")

        # 3. 如果没有找到文件
        if len(excel_files) == 0:
            print(f" \n3. 没有找到Excel文件")
            print(f" 可能的原因")
            print(f" -文件格式不是.xlsx")
            print(f" -文件在错误的目录")
            return None

        # 4. 保存文件列表到对象属性
        print(f" \n4. 保存文件列表")
        self.all_files = excel_files
        print(f" self.all_files = 包含{len(self.all_files)}个文件")

        # 5. 显示文件列表
        print(f' \n5. 显示文件列表(前10个)')
        print('-' * 70)

        for i, file in enumerate(self.all_files[:10], 1):
            # 获取文件大小(转为KB)
            file_size = file.stat().st_size / 1024

            # 从文件名提取股票代码
            filename = file.stem    # 去掉扩展名
            parts = filename.split('_')
            symbol = parts[0] if len(parts) > 0 else filename

            print(f' {i:2d}: {file.name}')
            print(f" 股票代码: {symbol}")
            print(f" 文件大小: {file_size:.1f} KB")
            print(f" 修改时间: {pd.Timestamp(file.stat().st_mtime, unit='s').strftime('%Y-%m-%d')}")

        if len(self.all_files) > 10:
            print(f" 还有{len(self.all_files) - 10}个文件")

        # 6. 选择第一个文件作为测试
        print(f" \n 6. 选择测试文件")
        self.test_file = self.all_files[0]

        # 从测试文件提取股票代码
        filename = self.test_file.stem
        parts = filename.split('_')
        self.test_symbol = parts[0] if len(parts) > 0 else filename

        print(f' 测试文件: {self.test_file.name}')
        print(f' 股票代码: {self.test_symbol}')

        # 7. 返回结果
        print("\n" + "=" * 70)
        print(f" 视频2完成. 找到{len(self.all_files)}个文件")
        print(f' 测试文件; {self.test_file.name}')
        print(f" 股票代码: {self.test_symbol}")
        print("=" * 70)
        return self.all_files

    def load_data(self, file_path):
        """第3步, 加载股票数据"""
        print('\n'+'='*70)
        print(f" 视频3. 加载股票数据")
        print('='*70)

        # 1. 显示要加载的文件
        print(f" \n1. 加载文件")
        print(f' 文件路径: {file_path.name}')

        # 2. 读取Excel 文件
        print(f' \n2. 读取Excel文件')
        try:
            df = pd.read_excel(file_path)
            print(f' 读取成功')
            print(f' 数据形状: {df.shape[0]}行 x {df.shape[1]}列')
        except Exception as e:
            print(f" 读取失败: {e}")
            return None

        # 3. 清理列名 (去除可能存在的空格)
        print(f" \n3. 清理列名")
        df.columns = df.columns.str.strip()
        print(f" 列名已清理")

        # 4. 显示列名
        print(f" \n4. 数据列名 (前10列):")
        print('-' * 40)
        for i, col in enumerate(df.columns[:10], 1):
            print(f" {i:2d}.{col}")
        if len(df.columns) > 10:
            print(f" 还有{len(df.columns) - 10}列")

        # 5. 检查必要列是否存在
        print(f" \5. 检查必要列")
        required_cols = ['close', '日收益率']
        missing_cols = []

        for col in required_cols:
            if col in df.columns:
                # 显示该列的基本信息
                non_null = df[col].count()
                null_count = df[col].isnull().sum()
                print(f" {col}: 存在(有效数据: {non_null}, 空值:{null_count}")
            else:
                print(f" {col}: 不存在")
                missing_cols.append(col)

        if missing_cols:
            print(f" \n 缺少必要列: {missing_cols}")
            print(f" 无法进行策略回测")
            return None

        # 6. 检查日期列
        print(f" \n6. 检查日期列")
        date_col = None
        possible_date_cols = ["Date", "date", "交易日期", "日期"]

        for col in possible_date_cols:
            if col in df.columns:
                date_col = col
                print(f" 找到日期列: {col}")

                #  转换日期格式
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    print(f" 日期范围: {df[date_col].min().strftime('%Y-%m-%d')}到{df[date_col].max().strftime('%Y-%m_%d')}")
                    print(f" 总天数: {(df[date_col].max() - df[date_col].min()).days}天")
                except:
                    print(f" 无法转换日期格式")
                break

        if date_col is None:
            print(f" 没有找到日期列, 将使用索引作为x轴")
            # 创建一个虚拟的日期列
            df['虚拟日期'] = range(len(df))
            date_col = '虚拟日期'

        # 7. 查看数据预览
        print(f' \n7. 数据预览: ')
        print('-' * 60)

        # 选择要显示的列
        display_cols = ['close', '日收益率', '累计收益率', '回撤'] if '累计收益率' in df.columns else ['close', '日收益率']
        display_cols = [col for col in display_cols if col in df.columns]

        print(df[display_cols].head(3).to_string())
        print('-' * 60)

        # 8. 检查数据完整性
        print("\n8. 检查数据完整性")
        print(f"   总数据量: {len(df)} 行")
        print(f"   有效收盘价: {df['close'].count()} 行")
        print(f"   有效收益率: {df['日收益率'].count()} 行")

        # 9. 保存数据到对象属性
        print(f" \n9. 保存数据")
        self.current_df = df
        self.current_symbol = file_path.stem.split('_')[0]
        self.date_col = date_col

        print(f" self.current_df 已保存")
        print(f" self.current_symbol = {self.current_symbol}")
        print(f" self.date_col = {date_col}")

        print('\n'+'='*70)
        print(f" 视频3完成: 数据加载成功")
        print('=' * 70)
        print(f" 股票代码: {self.current_symbol}")
        print(f" 数据范围: {len(df)}天")
        print(f" 日期列: {date_col}")
        print('='*70)

        return df

    def calculate_ma(self, df):
        """第4步, 计算移动平均线"""
        print("\n" + "=" * 70)
        print(f" 视频4, 计算移动平均线")
        print("=" * 70)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据, ")
            return None

        print(f" 数据形状; {df.shape[0]}行 x {df.shape[1]}列")

        # 2. 检查收盘价列
        print(f' \n2. 检查收盘价列')
        if 'close' not in df.columns:
            print(f" 没有找到收盘价列 'close'")
            return None

        close_data = df['close'].dropna()
        print(f" 收盘价数据点数: {len(close_data)}")
        print(f" 收盘价范围: {close_data.min():.2f} - {close_data.max():.2f}")

        # 3. 计算短期均线 (SMA - simple Moving Average)
        print(f" \n3. 计算短期均线 MA{self.short_window}")
        print(f" 公式: MA{self.short_window} = 过去{self.short_window}天的收盘价平均值")
        print(f" 执行: df['MA_short'] = df['close'].rolling(window={self.short_window}).mean()")

        df['MA_short'] = df['close'].rolling(window=self.short_window).mean()

        # 统计短期均线
        ma_short_count = df['MA_short'].count()
        ma_short_first = df['MA_short'].iloc[self.short_window - 1] if len(df) >= self.short_window else None

        print(f" 短期均线计算完成")
        print(f" 有效数据点: {ma_short_count}个")
        if ma_short_first:
            print(f" 第一个有效值: {ma_short_first:.2f}")

        #  4. 计算长期均线
        print(f" \n4. 计算长期均线 MA{self.long_window}")
        print(f" 公式: MA{self.long_window} = 过去{self.long_window}天的收盘价平均值")
        print(f" 执行: df['MA_long'] = df['close'].rolling(window={self.long_window}).mean()")

        df['MA_long'] = df['close'].rolling(window=self.long_window).mean()

        # 统计长期均线
        ma_long_count = df['MA_long'].count()
        ma_long_first = df['MA_long'].iloc[self.long_window - 1] if len(df) >= self.long_window else None

        print(f" 长期均线计算完成")
        print(f" 有效数据点: {ma_long_count}个")
        if ma_long_first:
            print(f" 第一个有效值: {ma_long_first:.2f}")

        # 5. 显示均线对比
        print(f" \n5. 均线对比 (最近5天) :")
        print('-'*70)

        # 获取最近5天的数据
        recent_data = df.tail(5)[['close', 'MA_short', 'MA_long']].copy()

        for idx, row in recent_data.iterrows():
            close_val = row['close']
            ma_short_val = row['MA_short']
            ma_long_val = row['MA_long']

            # 判断均线关系
            if pd.notna(ma_short_val) and pd.notna(ma_long_val):
                if ma_short_val > ma_long_val:
                    relation =  "短期↑ > 长期↑ (金叉可能)"
                elif ma_short_val < ma_long_val:
                    relation = "短期↓ < 长期↓ (死叉可能)"
                else:
                    relation = "短期 = 长期"
            else:
                relation = "数据不足"

            print(f' 收盘价: {close_val:.2f} | MA{self.short_window}: {ma_short_val:.2f} | '
                  f'MA{self.long_window}: {ma_long_val:.2f} | {relation}')

        # 6. 检查数据是否足够
        print(f" \n6. 检查数据是否足够")
        min_required = self.long_window
        actual_days = len(df)

        print(f" 需要最少天数: {min_required}天 ")
        print(f" 实际实际天数: {actual_days}天 ")

        if actual_days < min_required:
            print(f" 数据不足! 需要至少{min_required}天数据")
            print(f" 建议: 选择数据更长的股票, 或降低长期均线窗口")
        else:
            print(f" 数据充足, 可以进行策略回测")

        # 7. 保存计算结果
        print(f" \n7. 保存计算结果")
        self.current_df = df
        print(f" 均线已添加到 DataFrame")
        print(f" 新添加列: 'MA_short'(MA{self.short_window})")
        print(f" 新添加列: 'MA_long'(MA{self.long_window})")

        print('\n' + '-' * 70)
        print(f' 视频4完成: 移动平均线计算成功')
        print('-' * 70)
        print(f" MA{self.short_window} 有效数据点: {ma_short_count}")
        print(f" MA{self.long_window} 有效数据点: {ma_long_count}")

        print('=' * 70)
        return df

    def generate_signals(self, df):
        """第5步, 生成交易信号 (金叉买入, 死叉卖出)"""
        print('=' * 70)
        print(f" 视频5: 生成交易信号")
        print('=' * 70)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据, ")
            return None

        if "MA_short" not in df.columns or "MA_long" not in df.columns:
            print(f" 没有找到均线数据, ")
            return None

        print(f" 数据形状: {df.shape[0]}行")
        print(f" MA_short 有效数据: {df['MA_short'].count()}")
        print(f" MA_long 有效数据: {df['MA_long'].count()}")

        # 2. 初始化信号列
        print(f" \n2. 初始化信号列")
        df['signal'] = 0
        df['position'] = 0
        print(f" signal 列已创建 (0=无信号)")
        print(f" position 列已创建 (0=空仓")

        # 3. 计算均线关系 (短期 - 长期)
        print(f" \n3. 计算均线关系")
        print(f" 公式: 均线差 = MA_short - MA_long")

        df['ma_diff'] = df['MA_short'] - df['MA_long']
        print(f" 均线差已计算")

        # 显示均线差的统计
        diff_stats = df['ma_diff'].describe()
        print(f" 均线差统计: ")
        print(f" 平均值: {diff_stats['mean']:.4f}")
        print(f" 最小值: {diff_stats['min']:.4f}")
        print(f" 最大值: {diff_stats['max']:.4f}")

        # 4. 生成交易信号
        print(f" \n4. 生成交易信号")
        print(f" 规则: ")
        print(f" - 金叉 (买入信号): 短期均线上穿长期均线 -> signal = 1")
        print(f" - 死叉 (卖出信号): 长期均线下次短期均线 -> signal = -1")

        # 找出金叉和死叉的位置
        # 金叉: 前一天均线差 <= 0, 今天均线差 > 0
        # 死叉: 前一天均线差 >= 0, 今天均线差 < 0

        # 创建金叉信号
        golden_cross = (df['ma_diff'].shift(1) <= 0) & (df['ma_diff'] > 0)
        # 创建死叉信号
        death_cross = (df['ma_diff'].shift(1) >= 0) & (df['ma_diff'] < 0)

        # 设置信号
        df.loc[golden_cross, 'signal'] = 1  # 买入
        df.loc[death_cross, 'signal'] = -1  #卖出

        # 统计信号数量
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()

        print(f"\n 信号生成完成")
        print(f" 买入信号 (金叉): {buy_signals} 次")
        print(f" 卖出信号 (死叉): {sell_signals} 次")

        # 5. 计算持仓 (position)
        print(f" \n5. 计算持仓")
        print(f" 规则: 遇到买入信号 -> 持仓=1, 遇到卖出信号 -> 持仓=0")

        #  初始持仓为0
        position = 0
        position_changes = []

        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:       # 买入信号
                position = 1
            elif df['signal'].iloc[i] == -1:    # 卖出信号
                position = 0
            position_changes.append(position)

        df['position'] = position_changes

        # 统计持仓天数
        position_days = (df['position'] == 1).sum()
        print(f" 持仓计算完成")
        print(f" 持仓天数: {position_days} 天")
        print(f" 持仓比例: {position_days/len(df)*100:.1f}%")

        # 6. 显示最近10天的信号
        print(f" \n6. 最近10天的交易信号: ")
        print('-' * 80)
        print(f"{'日期':<12} {'收盘价':<8} {'MA5':<8} {'MA20':<8} {'均线差':<8} {'信号':<6} {'持仓':<4}")
        print('-' * 80)

        # 获取最近10天行数据
        recent = df.tail(10)

        for idx, row in recent.iterrows():
            date_str = row[self.date_col].strftime('%Y-%m-%d') if self.date_col in df.columns else str(idx)
            close_val = row['close']
            ma5 = row['MA_short'] if pd.notna(row['MA_short']) else 0
            ma20 = row['MA_long'] if pd.notna(row['MA_long']) else 0
            diff = row['ma_diff'] if pd.notna(row['ma_diff']) else 0
            signal = row['signal']
            position = row['position']

            # 信号显示
            signal_str = "买入" if signal == 1 else "卖出" if signal == -1 else "无"
            position_str = "持仓" if position == 1 else "空仓"

            print(f"{date_str:<12} {close_val:<8.2f} {ma5:<8.2f} {ma20:<8.2f} {diff:<8.4f} {signal_str:<6}"
                  f"{position_str:<4}")

        # 7. 显示所有交易信号
        print(f" \n7. 完整交易记录:")
        print('-' * 70)

        # 找出所有有信号的交易点
        trades = df[df['signal'] != 0].copy()

        if len(trades) > 0:
            print(f" 共有 {len(trades)} 次交易信号")
            print()

            # 显示前10次交易
            for i, (idx, row) in enumerate(trades.head(10).iterrows(), 1):
                date_str = row[self.date_col].strftime('%Y-%m-%d') if self.date_col in df.columns else str(idx)

                signal_type = "买入" if row['signal'] == 1 else "卖出"
                price = row['close']
                ma5 = row['MA_short']
                ma20 = row['MA_long']

                print(f" 交易{i:2d}: {date_str} - {signal_type}信号")
                print(f" 价格: {price:.2f}, MA{self.short_window}: {ma5:.2f}, MA{self.long_window}: {ma20:.2f}")

            if len(trades) > 10:
                print(f" 还有{len(trades) - 10} 次交易未显示")
        else:
            print(f" 没有交易信号生成")
            print(f" 可能原因: 均线没有交叉")

        # 8. 保存结果
        print(f" \n8. 保存结果")
        self.current_df = df
        print(f" 信号已保存到 self.current_dir")
        print(f" 新添加列: signal (交易信号)")
        print(f" 新添加列: position (持仓状态)")
        print(f" 新添加列: ma_diff (均线差)")

        print('\n' + '=' * 80)
        print(f" 视频5完成: 交易信号生成成功")
        print('=' * 80)
        print(f" 总买入信号: {buy_signals} 次")
        print(f" 总卖出信号: {sell_signals} 次")
        print(f" 持仓天数: {position_days} / {len(df)} = {position_days/len(df)*100:.1f}%")

        return df

    def calculate_strategy_returns(self, df):
        """第6步: 计算策略收益"""
        print('\n' + '=' *70)
        print(f" 视频6: 计算策略收益")
        print('=' * 70)

        # 1. 检查数据
        print(f' \n1. 检查数据')
        if df is None:
            print(f" 没有数据")
            return None

        if 'position' not in df.columns:
            print(f" 没有持仓数据, ")
            return None

        if '日收益率' not in df.columns:
            print(f" 没有日收益率数据")
            return None

        print(f' 数据形状: {df.shape[0]} 行')
        print(f" 持仓数据: {(df['position'] == 1).sum()} 天持仓")

        # 2. 计算策略收益率
        print(f" \n2. 计算策略收益率")
        print(f" 公式: 策略日收益率 = 持仓状态 x 股票日收益率")
        print(f" 说明: 持仓时获得股票收益, 空仓时收益为0")

        df['strategy_return'] = df['position'] * df['日收益率']

        # 统计策略收益率
        strategy_returns = df['strategy_return'].dropna()
        print(f" 策略收益率计算完成")
        print(f" 有效数据点: {len(strategy_returns)}")
        print(f" 平均日收益率: {strategy_returns.mean():.6f}")
        print(f" 日收益率标准差: {strategy_returns.std():.6f}")

        # 3. 计算累计收益
        print(f" \n3. 计算累计收益")
        print(f" 公式: 累计收益 = (1 + 日收益率)的累积乘积")

        # 策略累计收益
        df['strategy_cumulative'] = (1 + df['strategy_return']).cumprod()
        # 基准累计收益(买入持有)
        df['benchmark_cumulative'] = (1 + df['日收益率']).cumprod()

        print(f" 累计收益计算完成")
        print(f" 最终策略净值: {df['strategy_cumulative'].iloc[-1]:.4f}")
        print(f" 最终基准净值: {df['benchmark_cumulative'].iloc[-1]:.4f}")

        # 4. 计算总收益率
        print(f" \n4. 计算总收益率")
        print(f" 公式: 总收益率 = 最终净值 - 1")

        strategy_total_return = df['strategy_cumulative'].iloc[-1] - 1
        benchmark_total_return = df['benchmark_cumulative'].iloc[-1] - 1

        print(f" 策略总收益率; {strategy_total_return:.2%}")
        print(f" 基准总收益率: {benchmark_total_return:.2%}")

        # 5. 计算超额收益
        print(f" \n5. 计算超额收益")
        print(f" 公式: 超额收益 = 策略收益 - 基准收益")

        excess_return = strategy_total_return - benchmark_total_return
        print(f" 超额收益: {excess_return:.2%}")

        if excess_return > 0:
            print(f" 策略跑赢基准: {excess_return:.2%}")
        elif excess_return < 0:
            print(f" 策略跑输基准: {abs(excess_return):.2%}")
        else:
            print(f" 策略与基准持平")

        # 6. 计算年化收益率
        print(f" \n6. 计算年化收益率")
        print(f" 公式: 年化收益率: (1 + 总收益率)^(252/天数) - 1")

        trading_days = len(df)
        annual_factor = 252 / trading_days

        strategy_annual = ( 1 + strategy_total_return) ** annual_factor - 1
        benchmark_annual = (1 + benchmark_total_return) ** annual_factor - 1

        print(f" 策略年化收益率: {strategy_annual:.2%}")
        print(f" 基准年化收益率: {benchmark_annual:.2%}")

        # 7. 计算策略胜率
        print(f" \n7. 计算策略胜率")
        print(f" 公式: 胜率 = 盈利交易日 / 总交易日")

        #找出有交易的交易日 ( 持仓且收益率不为0)
        trading_days = df[df['position'] == 1].copy()
        if len(trading_days) > 0:
            winning_days = (trading_days['日收益率'] > 0).sum()
            win_rate = winning_days / len(trading_days)

            print(f" 持仓交易日: {len(trading_days)} 天")
            print(f" 盈利天数: {winning_days} 天")
            print(f" 亏损天数: {len(trading_days) - winning_days} 天")
            print(f" 胜率: {win_rate:.2%}")
        else:
            print(f" 没有持仓交易日")
            win_rate = 0

        # 8. 计算盈亏比
        print(f" \n8. 计算盈亏比")
        print(f" 公式: 盈亏比 = 平均盈利 / 平均亏损")

        if len(trading_days) > 0:
            avg_profit = trading_days[trading_days['日收益率'] > 0]['日收益率'].mean() if winning_days > 0 else 0
            avg_loss = abs(trading_days[trading_days['日收益率'] < 0]['日收益率'].mean()) if (len(trading_days) - winning_days) > 0 else 0

            if avg_loss > 0:
                profit_loss_ratio = avg_profit / avg_loss
                print(f" 平均盈利: {avg_profit:.4f}")
                print(f" 平均亏损: {avg_loss:.4f}")
                print(f" 盈亏比: {profit_loss_ratio:.2f}")
            else:
                profit_loss_ratio = 0
                print(f" 无亏损记录")
        else:
            profit_loss_ratio = 0

        # 9. 计算最大回撤
        print(f" \n9. 计算最大回撤")
        print(f" 公式: 最大回撤 = (当前净值 - 历史最高净值) / 历史最高净值")

        # 计算策略回撤
        strategy_peak = df['strategy_cumulative'].expanding().max()
        strategy_drawdown = (df['strategy_cumulative'] - strategy_peak) / strategy_peak
        strategy_max_dd = strategy_drawdown.min()

        # 计算基准回撤
        benchmark_peak = df['benchmark_cumulative'].expanding().max()
        benchmark_drawdown = (df['benchmark_cumulative'] - benchmark_peak) / benchmark_peak
        benchmark_max_dd = benchmark_drawdown.min()

        print(f" 策略最大回撤: {strategy_max_dd:.2%}")
        print(f" 基准最大回撤: {benchmark_max_dd:.2%}")

        # 10. 计算夏普比率
        print(f" \n10. 计算夏普比率")
        print(f" 公式: 夏普比率 = (策略年化收益率 - 无风险利率) / 策略年化波动率")

        # 计算策略日收益率的标准差 (年化)
        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        risk_free_rate = 0.02   # 假设无风险利率为2%

        if strategy_volatility > 0:
            sharpe_ratio = (strategy_annual - risk_free_rate) / strategy_volatility
            print(f" 策略年化波动率: {strategy_volatility:.2%}")
            print(f" 夏普比率: {sharpe_ratio:.4f}")
        else:
            sharpe_ratio = 0
            print(f" 无法计算夏普比率 (波动率为0)")

        # 11. 汇总策略指标
        print(f" \n11. 策略指标汇总")
        print('-' * 80)
        print(f" 策略总收益率: {strategy_total_return:>12.2%}")
        print(f" 基准总收益率: {benchmark_total_return:>12.2%}")
        print(f" 超额收益: {excess_return:>15.2%}")
        print(f" 策略年化收益: {strategy_annual:>12.2%}")
        print(f" 策略年化波动: {strategy_volatility:>12.2%}")
        print(f" 最大回撤: {strategy_max_dd:>15.2%}")
        print(f" 胜率: {win_rate:>19.2%}")
        print(f" 盈亏比: {profit_loss_ratio:>17.2f}")
        print(f" 夏普比率: {sharpe_ratio:>14.4f}")
        print("-" * 80)

        # 12. 保存策略指标
        print(f" \n12. 保存策略指标")
        self.strategy_metrics = {
            '股票代码': self.current_symbol,
            '短期均线': self.short_window,
            '长期均线': self.long_window,
            '策略总收益率': strategy_total_return,
            '基准总收益率': benchmark_total_return,
            '超额收益': excess_return,
            '策略年化收益率': strategy_annual,
            '策略年化波动率': strategy_volatility,
            '最大回撤': strategy_max_dd,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '夏普比率': sharpe_ratio,
            '买入信号次数': (df['signal'] == 1).sum(),
            '卖出信号次数': (df['signal'] == -1).sum(),
            '持仓天数': (df['position'] == 1).sum(),
            '总交易天数': len(df)
        }

        print(f" 策略指标已保存到 self.strategy_metrics")
        print(f" 共 {len(self.strategy_metrics)} 个指标")

        # 13. 保存结果到对象属性
        self.current_df = df
        print('\n' + '=' * 80)
        print(f" 视频6完成: 策略收益计算成功")
        print('=' * 80)
        print(f" 策略总收益: {strategy_total_return:.2%}")
        print(f" 基准总收益: {benchmark_total_return:.2%}")
        print(f" 超额收益: {excess_return:.2%}")
        print(f" 夏普比率: {sharpe_ratio:.4f}")

        return df

    def plot_strategy(self, df, symbol, save=True, show=True):
        """第7步: 绘制策略图表"""
        print('=' * 70)
        print(f" 视频7: 绘制策略图表")
        print('=' * 70)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据, ")
            return None

        print(f" 数据形状: {df.shape[0]}行")
        print(f" 股票代码: {symbol}")

        # 2. 获取字体 (从__init__ 中已经配置)
        title_font = self.font_prop if hasattr(self, 'font_prop') else None
        print(f" 中文字体: {'已配置' if title_font else '使用默认'}")

        # 3. 准备数据 (取最近500天, 避免图表太拥挤)
        print(f" \n3. 准备数据")
        if len(df) > 500:
            df_plot = df.tail(500).copy()
            print(f" 数据超过500天, 取最近500天绘制")
        else:
            df_plot = df.copy()
            print(f" 使用全部{len(df_plot)} 天数据")

        # 4. 获取日期列
        print(f" \n4. 获取日期列")
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            x = df_plot[self.date_col]
            x_label = '日期'
            print(f" 使用日期列: {self.date_col}")
        else:
            x = range(len(df_plot))
            x_label = '日期'
            print(f" 没有日期列, 使用索引")

        # 5. 创建图表
        print(f" \n5. 创建图表")
        fig, axes = plt.subplots(2,1,figsize=(14,10))
        fig.suptitle(f"{symbol} - 均线策略回测", fontproperties=title_font, fontsize=14, fontweight='bold')
        print(f' 图表大小: 14x10英寸')

        # 6. 第一张图, 价格和均线
        print(f" \n6. 绘制价格和均线图")
        ax1 = axes[0]

        # 绘制收盘价
        ax1.plot(x, df_plot['close'], linewidth=1.5, color='black', label='收盘价', alpha=0.7)
        print(f" 收盘价曲线已添加")

        # 绘制短期均线
        if 'MA_short' in df_plot.columns:
            ax1.plot(x, df_plot['MA_short'], linewidth=1.2, color='blue',
                     label=f"MA{self.short_window}", alpha=0.8)
            print(f" 短期均线: MA{self.short_window}")

        # 绘制长期均线
        if "MA_long" in df_plot.columns:
            ax1.plot(x, df_plot['MA_long'], linewidth=1.2, color='red',
                     label=f"MA{self.long_window}", alpha=0.8)
            print(f" 长期均线: MA{self.long_window}")

        # 标记买入点 (金叉)
        buy_signals = df_plot[df_plot['signal'] == 1]
        if len(buy_signals) > 0:
            # 获取买入点的 x 坐标
            if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
                buy_x = buy_signals[self.date_col]
            else:
                buy_x = buy_signals.index
            ax1.scatter(buy_x, buy_signals['close'],
                        color='green', s=100, marker='^', zorder=5, label='买入信号')
            print(f' 买入信号已标记: {len(buy_signals)} 个')

        # 标记卖出点 (死叉)
        sell_signals = df_plot[df_plot['signal'] == -1]
        if len(sell_signals) > 0:
            if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
                sell_x = sell_signals[self.date_col]
            else:
                sell_x = sell_signals.index
            ax1.scatter(sell_x, sell_signals['close'],
                        color='red', s=100, marker='v', zorder=5, label='卖出信号')
            print(f" 卖出信号已标记: {len(sell_signals)} 个")

        ax1.set_ylabel('价格', fontproperties=title_font, fontsize=12)
        ax1.set_xlabel(f"{symbol}-价格走势与交易信号", fontproperties=title_font, fontsize=12)
        ax1.legend(prop=title_font)
        ax1.grid(True, alpha=0.3)
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            ax1.tick_params(axis='x', rotation=45)
        print(f" 价格图表完成")

        # 7. 第2张图: 策略净值 VS 基准净值
        print(f' \n7. 绘制净值曲线图')
        ax2 = axes[1]

        # 绘制策略净值
        if 'strategy_cumulative' in df_plot.columns:
            ax2.plot(x, df_plot['strategy_cumulative'], linewidth=2, color='green',
                     label='策略净值', alpha=0.9)
            print(f" 策略净值曲线已添加")

        # 绘制基准净值
        if 'benchmark_cumulative' in df_plot.columns:
            ax2.plot(x, df_plot['benchmark_cumulative'], linewidth=2, color='blue',
                     label='基准净值', alpha=0.9)
            print(f" 基准净值曲线已添加")

        #标注最终净值
        final_strategy = df_plot['strategy_cumulative'].iloc[-1] if 'strategy_cumulative' in df_plot.columns else 1
        final_benchmark = df_plot['benchmark_cumulative'].iloc[-1] if 'benchmark_cumulative' in df_plot.columns else 1

        ax2.text(0.02, 0.95, f"策略净值: {final_strategy:.4f}", transform=ax2.transAxes,
                 fontproperties=title_font, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax2.text(0.02, 0.88, f"基准净值: {final_benchmark:.4f}", transform=ax2.transAxes,
                 fontproperties=title_font, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # 标注超额收益
        excess = final_strategy - final_benchmark
        if excess > 0:
            color = 'lightgreen'
            label = f" 超额收益: +{excess:.2%}"
        else:
            color = 'lightblue'
            label = f"超额收益: {excess:.2%}"

        ax2.text(0.02, 0.81, label, transform=ax2.transAxes,
                 fontproperties=title_font, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))

        ax2.set_ylabel('净值', fontproperties=title_font, fontsize=12)
        ax2.set_xlabel(x_label, fontproperties=title_font, fontsize=12)
        ax2.set_title('策略净值 VS 基准净值', fontproperties=title_font, fontsize=12)
        ax2.legend(prop=title_font)
        ax2.grid(True, alpha=0.3)
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            ax2.tick_params(axis='x', rotation=45)
        print(f" 净值曲线图完成")

        # 8. 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()  # 显示绘制图

        # ========== 添加保存图表的代码 ==========
        if save:
            # 确保图表目录存在
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            chart_path = self.charts_dir / f"{symbol}_均线策略.png"
            fig.savefig(chart_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ 图表已保存: {chart_path}")

        if show:
            plt.show()  # 显示绘制图
        else:
            plt.close()



        print('\n' + '-' * 70)
        print(f" 视频7完成: 策略图表绘制成功")
        print('-' * 70)
        print(f' 买入信号: {len(buy_signals)} 个')
        print(f' 卖出信号: {len(sell_signals)} 个')
        print(f' 最终策略净值: {final_strategy:.4f}')
        print(f' 最终基准净值: {final_benchmark:.4f}')

        return fig

    def optimize_parameters(self, file_path, short_range, long_range, save=True):
        """第8步: 优化均线参数组合
        Args:
            file_path: 数据文件路径
            short_range: 短期均线测试范围, 如 [5, 10, 20]
            long_range: 长期均线测试范围, 如 [20, 30, 50, 60]
            save: 是否保存结果
        """
        print('\n' + '=' * 80)
        print(f" 视频8: 参数优化 - 寻求最佳均线组合")
        print('=' * 80)

        # 1. 加载数据
        print(f' \n1. 加载数据')
        df = self.load_data(file_path)
        if df is None:
            print(f" 加载失败")
            return None

        symbol = file_path.stem.split('_')[0]
        print(f" 股票代码: {symbol}")

        # 2. 初始化结果存储
        print(f' \n2. 初始化结果存储')
        results = []
        print(f" 创建空列表存储优化结果")

        # 3. 计算参数组合总数
        total_combinations = len(short_range) * len(long_range)
        print(f" \n3. 开始参数优化")
        print(f" 短期均线范围: {short_range}")
        print(f" 长期均线范围: {long_range}")
        print(f" 总组合数: {total_combinations}")
        print('-' * 70)

        # 4. 循环测试每种参数组合
        current = 0
        for short in short_range:
            for long in long_range:
                current += 1

                # 跳过无效组合 (短期 >= 长期)
                if short >= long:
                    print(f" \n[{current}/{total_combinations}] 跳过: MA{short} x MA{long} (短期必须小于长期)")
                    continue
                print(f"\n[{current}/{total_combinations}] 测试: MA{short} x MA{long}")

                try:
                    # 临时保存当前参数
                    original_short = self.short_window
                    original_long = self.long_window
                    # 设置新参数
                    self.short_window = short
                    self.long_window = long
                    # 计算均线
                    df_temp = self.calculate_ma(df.copy())
                    if df_temp is None:
                        print(f" 均线计算失败")
                        continue

                    # 生成信号
                    df_temp = self.generate_signals(df_temp)
                    if df_temp is None:
                        print(f" 信号生成失败")
                        continue

                    # 计算收益
                    df_temp = self.calculate_strategy_returns(df_temp)
                    if df_temp is None:
                        print(f" 收益计算失败")
                        continue

                    # 记录结果:
                    result = {
                        '短期均线': short,
                        '长期均线': long,
                        '策略总收益': self.strategy_metrics['策略总收益率'],
                        '基准总收益': self.strategy_metrics['基准总收益率'],
                        '超额收益': self.strategy_metrics['超额收益'],
                        '策略年化收益': self.strategy_metrics['策略年化收益率'],
                        '最大回撤': self.strategy_metrics['最大回撤'],
                        '胜率': self.strategy_metrics['胜率'],
                        '盈亏比': self.strategy_metrics['盈亏比'],
                        '夏普比率': self.strategy_metrics['夏普比率'],
                        '买入次数': self.strategy_metrics['买入信号次数'],
                        '卖出次数': self.strategy_metrics['卖出信号次数'],
                        '持仓天数': self.strategy_metrics['持仓天数']
                    }
                    results.append(result)
                    print(f' 策略总收益: {results["策略总收益"]:.2%}')
                    print(f' 夏普比率: {results["夏普比率"]:.4f}')

                    # 恢复原参数
                    self.short_window = original_short
                    self.long_window = original_long

                except Exception as e:
                    print(f" 测试失败 {e}")
                    # 恢复原参数
                    self.short_window = original_short
                    self.long_window = original_long
                    continue

        # 5. 转换为DataFrame
        print(f"\n5. 整理结果")
        if not results:
            print(f" 没有有效的测试结果")
            return None

        results_df = pd.DataFrame(results)
        print(f" 有效结果数: {len(results_df)}")

        # 6. 按策略总收益排序
        print(f'\n6. 按策略总收益排序')
        results_df = results_df.sort_values('策略总收益', ascending=False)
        print(f" 排序完成")

        # 7. 显示前10名最佳组合
        print(f" \n7. 最佳参数组合 (前10名)")
        print( '=' * 70)
        print(f"{'排名':<4}{'短期':<6}{'长期':<6}{'策略收益':<10}{'超额收益':<10}{'夏普比率':<10}{'胜率':<8}")
        print( '=' * 70)
        for i, row in results_df.head(10).iterrows():
            rank = list(results_df.index).index(i) + 1
            print(f"{rank:<4}{row['短期均线']:<6}{row['长期均线']:<6}"
                  f"{row['策略总收益']:>9.2%}{row['超额收益']:>9.2%}"
                  f"{row['夏普比率']:>9.4f}{row['胜率']:>7.2%}")

        # 8. 找出最佳组合
        print(f" \n8. 找出最佳组合")
        best = results_df.iloc[0]
        print(f" 最佳组合: MA{best['短期均线']} x MA{best['长期均线']}")
        print(f" 策略总收益: {best['策略总收益']:.2%}")
        print(f" 超额收益: {best['超额收益']:.2%}")
        print(f" 夏普比率: {best['夏普比率']:.4f}")
        print(f" 胜率: {best['胜率']:.2%}")
        print(f" 最大回撤: {best['最大回撤']:.2%}")



        # 9. 找出最稳健组合 ( 最高夏普比率)
        print(f' \n9. 最稳健组合 (最高夏普比率)')
        best_sharpe = results_df.sort_values('夏普比率', ascending=False).iloc[0]
        print(f" 最稳健: MA{best_sharpe['短期均线']} x MA{best_sharpe['长期均线']}")
        print(f" 夏普比率: {best_sharpe['夏普比率']:.4f}")
        print(f" 策略收益: {best_sharpe['策略总收益']:.2%}")
        print(f" 最大回撤: {best_sharpe['最大回撤']:.2%}")


        # 10. 找出最低回撤组合
        print(f" \n10. 最低回撤组合 (最小回撤)")
        best_dd = results_df.sort_values('最大回撤', ascending=False).iloc[0]
        print(f" 最小回撤: MA{best_dd['短期均线']} x {best_dd['长期均线']}")
        print(f" 最大回撤: {best_dd['最大回撤']:.2%}")
        print(f" 策略收益: {best_dd['策略总收益']:.2%}")
        print(f" 夏普比率: {best_dd['夏普比率']:.4f}")

        # 11. 保存优化结果:   等下次一起存储

        # 12. 返回结果
        print('\n' + '=' * 80)
        print(f" 视频8完成: 参数优化成功")
        print('=' * 80)
        print(f" 总组合数: {total_combinations}")
        print(f" 有效组合: {len(results_df)}")
        print(f" 最佳组合: MA{best['短期均线']} x MA{best['长期均线']}")
        print(f" 最佳收益: {best['策略总收益']:.2%}")

        return results_df

    def batch_backtest(self, short_window, long_window, plot_all=False, plot_top=3, show_plots=True):
        """批量回测所有股票
        Args:
            short_window: 短期均线窗口
            long_window: 长期均线窗口
            plot_all: 是否绘制所有股票的图表（默认False，只计算不绘图）
            plot_top: 如果plot_all=False，绘制表现最好的前N个股票（默认3个）
            show_plots: 是否显示图表窗口（默认True，False时只保存）
        """
        print('\n' + '=' *80)
        print(f' 视频9: 批量回测所有股票')
        print('=' * 80)

        # 1. 检查文件列表
        print(f" \n1. 检查文件列表")
        if not hasattr(self, 'all_files') or len(self.all_files) == 0:
            print(f' 没有找到文件列表, ')
            return None

        total = len(self.all_files)
        print(f" 共有 {total} 个股票需要回测")
        print(f' 策略参数: MA{short_window} x MA{long_window}')

        # 2. 初始化结果存储
        print(f'\n2. 初始化结果存储')
        all_results = []
        success_count = 0
        failed_count = 0
        print(f" 创建空列表存储结果")

        # 3. 开始循环处理
        print(f" \3. 开始批量回测")
        print('=' * 80)

        for i, file_path in enumerate(self.all_files, 1):
            # 提取股票代码
            symbol = file_path.stem.split('_')[0]
            print(f' \n[{i}/{total}] 回测: {symbol}')

            try:
                # 加载数据
                df = self.load_data(file_path)
                if df is None:
                    print(f" 数据加载失败")
                    failed_count += 1
                    continue

                # 计算均线
                df = self.calculate_ma(df)
                if df is None:
                    print(f" 均线计算失败")
                    failed_count += 1
                    continue

                # 生成信号
                df = self.generate_signals(df)
                if df is None:
                    print(f" 信号生成失败")
                    failed_count += 1
                    continue

                # 计算收益
                df = self.calculate_strategy_returns(df)
                if df is None:
                    print(f" 收益计算失败")
                    failed_count += 1
                    continue

                # 记录结果
                result = {
                    '股票代码': symbol,
                    '短期均线': short_window,
                    '长期均线': long_window,
                    '策略总收益': self.strategy_metrics['策略总收益率'],
                    '基准总收益': self.strategy_metrics['基准总收益率'],
                    '超额收益': self.strategy_metrics['超额收益'],
                    '策略年化收益': self.strategy_metrics['策略年化收益率'],
                    '最大回撤': self.strategy_metrics['最大回撤'],
                    '胜率': self.strategy_metrics['胜率'],
                    '盈亏比': self.strategy_metrics['盈亏比'],
                    '夏普比率': self.strategy_metrics['夏普比率'],
                    '买入次数': self.strategy_metrics.get('买入信号次数', 0),
                    '卖出次数': self.strategy_metrics.get('卖出信号次数', 0),
                    '持仓天数': self.strategy_metrics.get('持仓天数', 0),
                    '数据天数': len(df)
                }
                all_results.append(result)
                success_count += 1

                print(f" 策略总收益: {result['策略总收益率']:.2%}")
                print(f" 夏普比率: {result['夏普比率']:.4f}")

            except Exception as e:
                print(f" 回测失败: {e}")
                failed_count += 1
                continue

        # 4. 显示汇总结果
        print('\n' + '=' * 80)
        print(f" \n4. 批量回测完成")
        print('=' * 80)
        print(f" 总股票数: {total}")
        print(f" 成功回测: {success_count}")
        print(f" 失败: {failed_count}")

        if not all_results:
            print(f" 没有成功的回测结果")
            return None

        # 5. 转换为DataFrame并排序
        print(f" \n5. 整理结果")
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('策略总收益', ascending=False)
        print(f" 共{len(results_df)} 个有效的结果")

        # 6. 显示前10名
        print(f"\n6. 表现最好的前10名股票: ")
        print('-' * 80)
        print(f"{'排名':<4}{'股票代码':<8}{'策略收益':<10}{'超额收益':<10}{'夏普比率': <10}{'胜率':<8}")
        print('-' * 80)

        for i, row in results_df.head(10).iterrows():
            rank = list(results_df.index).index(i) + 1
            print(f" {rank:<4} {row['股票代码']:<8}"
                  f"{row['策略总收益']:>9.2%} {row['超额收益']:>9.2%}"
                  f"{row['夏普比率']:>9.4f} {row['胜率']:>7.2%}")

        # 7. 显示后5名
        print(f" \n7. 表现最差的5名股票: ")
        print('-' * 80)
        for i, row in results_df.tail(5).iterrows():
            rank = len(results_df) - list(results_df.index).index(i)
            print(f" {rank:<4} {row['股票代码']:<8}"
                  f"{row['策略总收益']:>9.2%} {row['超额收益']:>9.2%}"
                  f"{row['夏普比率']:>9.4f} {row['胜率']:>7.2%}")

        # 8. 统计汇总
        print(f" \n8. 统计汇总")
        print('=' * 80)
        print(f" 平均策略收益: {results_df['策略总收益'].mean():.2%}")
        print(f" 平均超额收益: {results_df['超额收益'].mean():.2%}")
        print(f" 平均夏普比率: {results_df['夏普比率'].mean():.4f}")
        print(f" 平均胜率: {results_df['胜率'].mean():.2%}")
        print(f" 平均最大回撤: {results_df['最大回撤'].mean():.2%}")

        # 9. 绘制图表
        print(f" \n9. 绘制策略图表")
        print('=' * 70)
        if plot_all:
            symbols_to_plot = results_df['股票代码'].tolist()
            print(f" 绘制所有{len(symbols_to_plot)} 个股票图表")
        else:
            symbols_to_plot = results_df.head(plot_top)['股票代码'].tolist()
            print(f" 绘制表现最好的前{plot_top}名股票图表")

        # 绘制图表
        plotted_count = 0
        for symbol in symbols_to_plot:
            # 找到对应的文件路径
            file_path= None
            for f in self.all_files:
                if f.stem.split('_')[0] == symbol:
                    file_path = f
                    break

            if file_path is None:
                print(f" 未找到{symbol}的文件")
                continue
            print(f' 文件: {file_path.name}')

            try:
                df_temp = self.load_data(file_path)
                if df_temp is None:
                    print(f" 数据加载失败")
                    continue

                df_temp = self.calculate_ma(df_temp)
                if df_temp is None:
                    print(f" 均线计算失败")
                    continue

                df_temp = self.generate_signals(df_temp)
                if df_temp is None:
                    print(f" 信号生成失败")
                    continue

                df_temp = self.calculate_strategy_returns(df_temp)
                if df_temp is None:
                    print(f" 收益计算失败")
                    continue

                self.plot_strategy(df_temp, symbol, save=True, show=show_plots)
                plotted_count += 1
                print(f" {symbol} 图表绘制成功")

            except Exception as e:
                print(f" 绘制失败: {e}")
                import traceback
                traceback.print_exc()
        print(f' 成功绘制 {plotted_count}/{len(symbols_to_plot)}个股票图表')







        # 10.返回结果
        print('\n'+ '=' * 80)
        print(f" 视频9完成: 批量回测成功")
        print('=' * 80)
        print(f" 成功回测: {success_count} 个股票")
        print(f" 结果已保存在 all_results 中")

        return results_df

    def save_all_results(self, results_df, single_result_df=None):
        """保存所有回测结果到Excel文件"""
        print("\n" + "=" * 70)
        print("📹 视频10：保存所有结果")
        print("=" * 70)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # 1. 创建保存目录
        print(f" \n1. 创建保存目录")
        save_dir = self.output_dir / f"回测结果_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f" 创建目录: {save_dir}")

        saved_files = []
        saved_detail_count = 0

        # 2. 保存批量回测结果
        print(f" \n2. 保存批量回测结果 ")
        if results_df is not None and len(results_df) > 0:
            batch_file = save_dir / "批量回测结果.xlsx"
            with pd.ExcelWriter(batch_file, engine='openpyxl') as writer:
                # Sheet 1: 按收益排序的结果
                results_df_sorted = results_df.sort_values("策略总收益", ascending=False)
                results_df_sorted.to_excel(writer, sheet_name='按收益排序', index=False)

                # Sheet 2. 统计汇总
                summary_data = {
                    '指标': ['总股票数', '平均策略收益', '平均超额收益', '平均夏普比率',
                        '平均胜率', '平均最大回撤', '最佳股票', '最佳收益'],
                    '数值': [
                        len(results_df),
                        results_df['策略总收益'].mean(),
                        results_df['超额收益'].mean(),
                        results_df['夏普比率'].mean(),
                        results_df['胜率'].mean(),
                        results_df['最大回撤'].mean(),
                        results_df.iloc[0]['股票代码'],
                        results_df.iloc[0]['策略总收益']
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='统计汇总', index=False)

            files_size = batch_file.stat().st_size / 1024
            print(f" 已保存: {batch_file.name}({files_size:.1f}KB)")
            saved_files.append(str(batch_file))

        # 3.  保存单次回测结果
        print("\n3. 保存单次回测结果")
        if hasattr(self, 'strategy_metrics') and self.strategy_metrics:
            single_file = save_dir / f"{self.current_symbol}_单次回测结果.xlsx"
            with pd.ExcelWriter(single_file, engine='openpyxl') as writer:
                # Sheet 1: 策略指标
                metrics_df = pd.DataFrame([self.strategy_metrics])
                metrics_df.to_excel(writer, sheet_name='策略指标', index=False)
                # Sheet 2: 交易信号
                if hasattr(self, 'current_df'):
                    signals_df = self.current_df[self.current_df['signal'] != 0].copy()
                    if len(signals_df) > 0:
                        cols = ['Date', 'close', 'MA_short', 'MA_long', 'signal']
                        cols = [c for c in cols if c in signals_df.columns]
                        signals_df[cols].to_excel(writer, sheet_name='交易信号', index=False)
                        print(f" 交易信号: {len(signals_df)}")
            file_size = single_file.stat().st_size / 1024
            print(f" 已保存: {single_file.name} ({file_size:.1f} KB)")
            saved_files.append(str(single_file))

        # 4. 保存各股票详细结果
        print("\n4. 保存各股票详细结果")
        if results_df is not None and len(results_df) > 0:
            detailed_dir = save_dir / "各股票详细结果"
            detailed_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in results_df.head(20).iterrows(): # 只保存前10名
                symbol = row['股票代码']
                print(f"   处理: {symbol}...", end=" ")
                # 找到对应的文件路径
                file_path = None
                for f in self.all_files:
                    if f.stem.split('_')[0] == symbol:
                        file_path = f
                        break
                if file_path:
                    try:
                        df_detail = self.load_data(file_path)
                        if df_detail is not None:
                            df_detail = self.calculate_ma(df_detail)
                            df_detail = self.generate_signals(df_detail)
                            df_detail = self.calculate_strategy_returns(df_detail)
                            detail_file = detailed_dir / f" {symbol}_详细结果.xlsx"

                            with pd.ExcelWriter(detail_file, engine='openpyxl') as writer:
                                if hasattr(self, 'strategy_metrics'):
                                    pd.DataFrame([self.strategy_metrics]).to_excel(writer, sheet_name='策略指标', index=False)
                                signals_df = df_detail[df_detail['signal'] != 0].copy()
                                if len(signals_df) > 0:
                                    cols = ['Date', 'close', 'MA_short', 'MA_long', 'signal']
                                    ools = [c for c in cols if c in signals_df.columns]
                                    signals_df[cols].to_excel(writer, sheet_name='交易信号', index=False)
                            saved_detail_count += 1
                            print(f" ✅")
                    except Exception as e:
                        print(f" ❌ {e}")
                else:
                    print(f" 文件不存在")
            print(f"  ✅ 已保存 {saved_detail_count} 个股票的详细结果")

        # 5. 复制图表到保存目录
        print("\n5. 保存图表")
        chart_dir = save_dir / "图表"
        chart_dir.mkdir(parents=True, exist_ok=True)
        if self.charts_dir.exists():
            import shutil
            chart_files = list(self.charts_dir.glob("*.png"))
            for chart_file in chart_files:
                dest_file = chart_dir / chart_file.name
                shutil.copy2(chart_file, dest_file)
            print(f" ✅ 已复制 {len(chart_files)} 个图表")

        # 6. 生成汇总报告
        print("\n6. 生成汇总报告")
        report_file = save_dir / "回测汇总报告.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('=' * 70 + '\n')
            f.write("量化回测汇总报告\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"策略参数: MA{self.short_window} × MA{self.long_window}\n\n")

            if results_df is not None and len(results_df) > 0:
                f.write("-" * 50 + "\n")
                f.write("批量回测统计\n")
                f.write("-" * 50 + "\n")
                f.write(f"总股票数: {len(results_df)}\n")
                f.write(f"平均策略收益: {results_df['策略总收益'].mean():.2%}\n")
                f.write(f"平均超额收益: {results_df['超额收益'].mean():.2%}\n")
                f.write(f"平均夏普比率: {results_df['夏普比率'].mean():.4f}\n")
                f.write(f"平均胜率: {results_df['胜率'].mean():.2%}\n")
                f.write(f"平均最大回撤: {results_df['最大回撤'].mean():.2%}\n\n")

                f.write("-" * 50 + "\n")
                f.write("表现最好的前10名\n")
                f.write("-" * 50 + "\n")
                for idx, row in results_df.head(10).iterrows():
                    f.write(f"{row['股票代码']}: 收益 {row['策略总收益']:.2%}, "
                            f"夏普 {row['夏普比率']:.4f}, 胜率 {row['胜率']:.2%}\n")

        print(f" 已保存: {report_file.name}")
        saved_files.append(str(report_file))

        # 7. 显示保存结果
        print("\n" + "-" * 70)
        print("✅ 视频10完成：所有结果已保存")
        print("-" * 70)
        print(f"保存目录: {save_dir}")
        print(f" - 批量回测结果.xlsx")
        print(f" - 单次回测结果.xlsx")
        print(f" - 各股票详细结果/ ({saved_detail_count} 个文件)")
        print(f" - 图表/ ({len(chart_files) if self.charts_dir.exists() else 0} 个)")
        print(f" - 回测汇总报告.txt")

        return save_dir




# 调用:
if __name__ == "__main__":
    print(f" 初始化策略")
    print('=' * 70)

    # 使用默认参数创建策略
    strategy = MovingAverageStrategy(short_window=5, long_window=20)
    print("\n📋 策略对象属性:")
    print(f"   short_window = {strategy.short_window}")
    print(f"   long_window = {strategy.long_window}")
    print(f"   data_dir = {strategy.data_dir}")
    print(f"   output_dir = {strategy.output_dir}")
    print(f"   charts_dir = {strategy.charts_dir}")
    print(f" 测试完成.......")

    # 调用Find_data_file
    files = strategy.find_data_files()

    if files:
        # ==================方式1. 向做单次回撤, 再做参数优化===============
        print("\n" + "=" * 70)
        print("第一部分：单次回测（使用默认参数 MA5 × MA20）")
        print("=" * 70)

        #加载测试文件的数据
        df = strategy.load_data(strategy.test_file)

        if df is not None:
            print(f"\n 数据加载成功")
            print(f" 数据形状: {df.shape}")
            print(f" 股票代码: {strategy.current_symbol}")
            # 调用计算均线
            df = strategy.calculate_ma(df)

            if df is not None:
                print(f' \n 均线计算完成')
                print(f' 数据形状: {df.shape}')
                print(f" 新添加列: MA_short, MA_long")

            if df is not None:
                # 生成交易信号
                df = strategy.generate_signals(df)
                if df is not None:
                    print(f" \n 信号生成完成")
                    print(f' 买入信号: {(df["signal"] == 1).sum()}')
                    print(f" 卖出信号: {(df['signal'] == -1).sum()}")

                if df is not None:
                    # 技术收益
                    df = strategy.calculate_strategy_returns(df)
                    if df is not None:
                        print(f" \n 策略回测完成")
                        print(f" 策略总收益: {strategy.strategy_metrics['策略总收益率']:.2%}")
                        print(f" 夏普比率: {strategy.strategy_metrics['夏普比率']:.4f}")

                        strategy.plot_strategy(df, strategy.test_symbol, save=True)

        # ====================方式2. 参数优化(找出最佳参数组合) =====================
        print('\n' + '=' * 80)
        print(f" 第2部分: 参数优化 (寻找最佳均线组合) ")
        print('=' * 80)

        # 定义要测试参数的范围
        short_range = [5, 10, 15, 20]       # 短线均线测试范围
        long_range = [20, 30, 40, 50, 60]   # 长期均线测试范围

        # 运行参数优化
        results = strategy.optimize_parameters(
            file_path=strategy.test_file,   # 数据文件
            short_range=short_range,        # 短期均线范围
            long_range=long_range,          # 长期均线范围
            save=True                       # 保存结果到文件:  现在想不保存
        )

        # 初始化最佳参数变量
        best_short = 5
        best_long = 20
        optimization_success = False

        if results is not None and len(results) > 0:
            optimization_success = True
            best_short = results.iloc[0]['短期均线']
            best_long = results.iloc[0]['长期均线']
            print(f" \n参数优化完成!")
            print(f" 最佳组合: MA{best_short} x MA{best_long}")
            print(f" 最佳收益: {results.iloc[0]['策略总收益']:.2%}")
            print(f" 夏普比率: {results.iloc[0]['夏普比率']:.4f}")
        else:
            print(f' \n 参数优化失败, 将使用默认参数 MA5 x MA20')




        # ===========================方式3. 批量回测所有股票===========================
        print('\n' + '=' * 80)
        print(f" 第3部分: 批量回测所有股票")
        print('=' * 80)

        # 方式3-1. 使用默认参数批量回测
        print(f" \n3-1. 使用默认参数 MA5 x MA20 批量回测所有股票")
        all_results_default = strategy.batch_backtest(short_window=5, long_window=20)
        if all_results_default is not None:
            print(f" \n默认参数回测完成")
            print(f" 共分析 {len(all_results_default)} 个股票")
            print(f" 最佳股票: {all_results_default.iloc[0]['股票代码']} (收益; {all_results_default.iloc[0]['策略总收益']:.2%})")
            print(f" 平均收益: {all_results_default['策略总收益'].mean():.2%}")

        # 方式3-2. 使用优化的最佳参数批量回测
        if results is not None:
            print(f" \n3-2. 使用优化的最佳参数批量回测所有股票")
            print(f" 最佳参数: MA{best_short} x MA{best_long}")
            all_results_best = strategy.batch_backtest(short_window=best_short, long_window=best_long)

            if all_results_best is not None:
                print(f'\n 最佳参数回测完成!')
                print(f" 共分析: {len(all_results_best)}个股票")
                print(f" 最佳股票: {all_results_best.iloc[0]['股票代码']} (收益: {all_results_best.iloc[0]['策略总收益']:.2%})")
                print(f" 平均收益: {all_results_best['策略总收益'].mean():.2%}")

                #对比两种参数的平均收益
                if all_results_default is not None:
                    avg_default = all_results_default['策略总收益'].mean()
                    avg_best = all_results_best['策略总收益'].mean()
                    print(f" \n 参数对比")
                    print(f" 默认参数 (MA5 x MA20) 平均收益: {avg_default:.2%}")
                    print(f" 优化参数 (MA{best_short}xMA{best_long}) 平均收益: {avg_best:.2%}")
                    if avg_best > avg_default:
                        print(f" 优化参数提升了{avg_best - avg_default:.2%}")
                    else:
                        print(f" 优化参数没有提升")

        # ====================保存所有结果====================
        print('\n' + '=' * 80)
        print(f' 第4部分: 保存所有结果到Excel')
        print('=' * 80)

        # 保存默认参数的回测结果
        if all_results_default is not None:
            save_dir = strategy.save_all_results(all_results_default)
            print(f"\n✅ 默认参数回测结果已保存到: {save_dir}")

        # 如果最佳参数回测结果存在，也保存
        if results is not None and all_results_best is not None:
            save_dir_best = strategy.save_all_results(all_results_best)
            print(f"\n✅ 最佳参数回测结果已保存到: {save_dir_best}")

        print('\n' + '=' * 80)
        print(f' 第8天所有任务完成！')
        print('=' * 80)


'''
## 第8天：策略一 —— 均线策略

**任务目标**：
- 设计均线交易逻辑（金叉买入、死叉卖出）
- 计算交易信号并生成策略持仓
- 计算策略收益并与基准对比
- 测试不同均线参数组合，寻找最佳参数
- 批量回测所有股票，评估策略有效性

**实现方案**：
1. **策略设计**：
   - **金叉买入信号**：短期均线上穿长期均线（MA短期 > MA长期）
   - **死叉卖出信号**：短期均线下穿长期均线（MA短期 < MA长期）
   - **持仓规则**：买入后持仓直至卖出信号出现

2. **核心计算流程**：
   - **移动平均线计算**：使用`rolling(window).mean()`计算短期和长期SMA
   - **交易信号生成**：基于均线差（MA短期 - MA长期）的符号变化识别金叉/死叉
   - **策略收益计算**：策略日收益率 = 持仓状态 × 股票日收益率
   - **绩效指标**：总收益率、年化收益率、夏普比率、最大回撤、胜率、盈亏比

3. **参数优化系统**：
   - 遍历短期均线范围（如5-20天）和长期均线范围（如20-60天）
   - 自动跳过无效组合（短期 ≥ 长期）
   - 按策略总收益率排序，找出最佳组合
   - 同时识别最高夏普比率（最稳健）和最低回撤组合

4. **批量回测引擎**：
   - 循环处理所有股票数据文件
   - 统一使用指定参数进行回测
   - 生成股票表现排名（按策略收益排序）
   - 自动绘制表现最佳股票的图表

5. **可视化分析**：
   - **图表一**：价格走势 + 均线 + 买卖信号标记
   - **图表二**：策略净值 vs 基准净值对比曲线
   - 标注最终净值、超额收益等关键指标

6. **结果保存系统**：
   - 保存批量回测结果Excel（按收益排序）
   - 保存各股票详细结果（策略指标、交易信号）
   - 自动复制图表到结果目录
   - 生成汇总报告文本文件

**核心代码结构**：
```python
class MovingAverageStrategy:
    ├── __init__()                       # 初始化策略参数和目录
    ├── _setup_chinese_font()             # 配置中文字体
    ├── find_data_files()                 # 查找收益率数据文件
    ├── load_data()                       # 加载股票数据
    ├── calculate_ma()                    # 计算移动平均线
    ├── generate_signals()                # 生成交易信号（金叉/死叉）
    ├── calculate_strategy_returns()      # 计算策略收益和绩效指标
    ├── plot_strategy()                   # 绘制策略图表
    ├── optimize_parameters()             # 参数优化（寻找最佳均线组合）
    ├── batch_backtest()                  # 批量回测所有股票
    └── save_all_results()                # 保存所有结果到Excel
```   
    '''




