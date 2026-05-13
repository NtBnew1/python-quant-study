'''
第10天
					策略二 —— 动量策略
-设计动量策略逻辑
-计算动量信号
-生成策略收益

练习：
-调整动量窗口参数
'''


import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



class MomentumStrategy:
    """动量策略"""
    def __init__(self, momentum_window=20):
        """初始化动量策略
        Args:
           momentum_window: 动量计算窗口（默认20天）
        """

        print('=' * 70)
        print('第10天 - 方法1: 初始化动量策略')
        print('=' * 70)

        # 1. 设置策略参数
        print(f" \n1. 设置策略参数")
        self.momentum_window = momentum_window
        print(f' 动量窗口: {self.momentum_window}天')
        print(f" 说明: 计算过去{momentum_window}天的累计收益率")

        # 2. 获取当前文件目录
        print(f" \n2. 获取当前文件目录")
        current_dir = Path(__file__).parent
        print(f' 当前文件目录: {current_dir}')

        # 3. 找到项目根目录
        print(f" \n3. 找到项目根目录")
        self.project_root = current_dir.parent
        print(f" 项目根目录: {self.project_root}")

        # 4. 设置数据目录
        print(f" \n4. 设置数据目录")
        self.data_dir = self.project_root / "data" / "returns"
        print(f" 数据目录: {self.data_dir}")

        # 5. 设置结果输出目录
        print(f" \n5. 设置结果输出目录")
        self.output_dir = self.project_root / "data" / "动量策略结果"
        print(f' 输出目录: {self.output_dir}')

        # 创建输出目录
        if not self.output_dir.exists():
            print(f" 目录不存在, 正在创建.........")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f" 目录已创建")
        else:
            print(f' 目录已创建')

        # 6. 设置图表输出目录
        print(f" \n6. 设置图表输出目录")
        self.charts_dir = self.project_root / "charts" / "动量策略图表"
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
        self.all_results = []
        self.trade_records = []
        print(f" self.all_results = [] (存储策略结果)")
        print(f" self.trade_records = [] (存储交易记录)")

        # 8. 配置中文字体
        print(f" \n8. 配置中文字体")
        self._setup_chinese_font()

        # 9. 初始化完成
        print("\n" + "-" * 70)
        print(f" 方法1完成: 初始化成功")
        print('-' * 70)
        print(f" 动量窗口: {self.momentum_window}天")
        print(f" 数据目录: {self.data_dir}")
        print(f" 输出目录: {self.output_dir}")
        print(f" 图表目录: {self.charts_dir}")
        print('=' * 70)

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
                    print(f" 中文字体配置成功: {font_name}")
                    return
                except:
                    continue

        print(f" 未找到中文字体, 使用默认字体")
        plt.rcParams['axes.unicode_minus'] = False

    def find_data_files(self):
        """查找收益率数据"""
        print(f" \n" + "=" * 70)
        print(f" 方法2: 查找数据文件")
        print("=" * 70)

        # 1. 检查数据目录是否存在
        print(f" \n1. 检查数据目录")
        print(f" 数据目录: {self.data_dir}")
        if not self.data_dir.exists():
            print(f" 数据目录不存在")
            return None
        print(f" 数据目录存在")

        # 2. 查找所有Excel文件
        print(f" \n2. 查找所有Excel文件")
        print(f" 执行: list(self.data_dir.glob('*.xlsx')")
        excel_files = list(self.data_dir.glob('*.xlsx'))
        print(f" 找到{len(excel_files)} 个Excel文件")

        # 3. 如果没有找到文件
        if len(excel_files) == 0:
            print(f" \n3. 没有找到Excel文件")
            return None     # 如果没有找到文件就会返回 None

        # 4. 保存文件列表到对象属性
        print(f" \n4. 保存文件列表")
        self.all_files = excel_files
        print(f" self.all_files = 包含 {len(self.all_files)} 个文件")

        # 5. 显示文件列表
        print(f" \n4. 显示文件列表")
        print(f"-" * 70)
        for i, file in enumerate(self.all_files,1):
            # 转换文件大小 (转为KB)
            file_size = file.stat().st_size / 1024

            # 从文件名提取股票代码:
            """这里是循环提取股票代码. 循环之后就会结束"""
            filename = file.stem
            parts = filename.split('_')
            symbol = parts[0] if len(parts) > 0 else filename
            print(f" {i:2d}. {file.name}")
            print(f" 股票代码: {symbol}")
            print(f" 文件大小: {file_size:.1f} KB")

        # 6. 选择第一个文件作为测试
        print(f" \n6. 选择测试文件")
        self.test_file = self.all_files[0]

        # 提取股票代码
        """这里是提取单个股票代码"""
        filename = self.test_file.stem
        parts = filename.split('_')
        self.test_symbol = parts[0] if len(parts) > 0 else filename
        print(f" 测试文件: {self.test_file.name}")
        print(f" 股票代码: {self.test_symbol}")

        # 7. 返回结果
        print(f" \n" + "=" * 80)
        print(f" 方法2完成: 数据文件查找成功")
        print('=' * 80)
        print(f" 总文件数: {len(self.all_files)}")
        print(f" 测试文件: {self.test_file.name}")
        print(f" 测试股票: {self.test_symbol}")

        return self.all_files

    def load_data(self, file_path):
        """加载股票数据"""
        print("\n" + '=' * 80)
        print(f" 方法3: 加载股票数据")
        print('=' * 80)

        # 1. 显示要加载的文件
        print(f" \n1. 加载文件")
        print(f" 文件路径: {file_path.name}")

        # 2. 读取Excel文件
        print(f" \n2. 读取Excel文件")
        try:
            df = pd.read_excel(file_path)
            print(f" 读取成功")
            print(f" 数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        except Exception as e:
            print(f" 读取失败: {e}")
            return None

        # 3. 清理列名 (去除空格)
        print(' \n3. 清理列名')
        df.columns = df.columns.str.strip()
        print(f'列名已清理')

        # 4.显示列名
        print(f" \n4. 数据列名:")
        print(f'-' * 30)
        for i, col in enumerate(df.columns, 1):
            print(f" {i:2d}. {col}")
        print(f" -" * 40)

        # 5. 检查必要列
        print(f" \n5. 检查必要列")
        required_cols = ['close', '日收益率']
        missing_cols = []

        for col in required_cols:
            if col in df.columns:
                non_null = df[col].count()
                null_count = df[col].isnull().sum()
                print(f" {col}: 存在(有效数据: {non_null}, 空值:{null_count})")
            else:
                print(f" {col}: 不存在")
                missing_cols.append(col)

        if missing_cols:
            print(f" \n 缺少必要列: {missing_cols}")
            return None

        # 6. 检查日期列
        print(f" \n6. 检查日期列")
        date_col = None
        for col in ['Date', 'date', '交易日期', '日期']:
            if col in df.columns:
                date_col = col
                print(f" 找到日期列: {col}")
                break

        # 7. 查看数据预览
        print(f" \n7. 数据预览:")
        print('-' * 30)
        display_cols = ['close', '日收益率'] if '日收益率'in df.columns else ['close']
        print(df[display_cols].head(10).to_string())  # 数据全部打印. 没必要. 只显示打印前10个数据

        print(f" -" * 60)

        # 8. 提取股票代码
        print(f" \n8. 提取股票代码")
        symbol = file_path.stem.split('_')[0]
        print(f" 股票代码: {symbol}")

        # 9. 保存数据到对象属性
        print(f" \n9. 保存数据")
        self.current_df = df
        self.current_symbol = symbol
        self.date_col = date_col

        print(f' self.current_df 已保存')
        print(f' self.current_symbol = {symbol}')
        print(f" self.date_col = {date_col}")

        print('\n' + '=' * 80)
        print(f" 方法3完成: 数据加载成功")
        print('=' * 80)
        print(f" 股票代码: {symbol}")
        print(f" 数据范围: {len(df)} 天")
        print(f" 价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

        return df

    def calculate_momentum(self, df):
        """计算动量指标 (过去N天的累计收益率)"""
        print('\n' + '=' * 80)
        print(f" 方法4: 计算动量指标")
        print('=' * 80)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据")
            return None
        print(f" 数据形状: {df.shape[0]}行 x {df.shape[1]} 列")

        # 2. 检查收盘价列
        print(f" \n2. 检查收盘价列")
        if 'close' not in df.columns:
            print(f" 没有找到收盘价列")
            return None

        close_data = df['close'].dropna()
        print(f" 收盘价数据点数: {len(close_data)}")
        print(f" 收盘价范围: {close_data.min()} - {close_data.max()}")

        # 3. 计算动量 (方法1. 使用累计收益率)
        print(f" \n3. 计算动量指标 (过去{self.momentum_window}天)")
        print(f" 公式: 动量 = (当前价格 - N天前价格) / N天前价格")
        print(f" 执行: df['momentum'] = df['close'].pct_change(periods={self.momentum_window})")
        df['momentum'] = df['close'].pct_change(periods=self.momentum_window)

        # 统计动量
        momentum_count = df['momentum'].count()
        momentum_mean = df['momentum'].mean()
        momentum_std = df['momentum'].std()
        print(f" \n 动量计算完成")
        print(f" 有效数据点: {momentum_count} 个")
        print(f" 动量均值: {momentum_mean}")
        print(f" 动量标准差: {momentum_std}")

        # 4. 显示动量统计:
        print(f'\n4. 动量统计')
        print('=' * 50)
        print(f" 最小值: {df['momentum'].min():.4f}")
        print(f" 25%分位: {df['momentum'].quantile(0.25):.4f}")
        print(f" 中位数: {df['momentum'].median():.4f}")
        print(f" 75%分位: {df['momentum'].quantile(0.75):.4f}")
        print(f" 最大值: {df['momentum'].max():.4f}")

        # 5. 显示最近几天的动量
        print(f" \n5. 最近5天的动量")
        print('-' * 60)
        recent_momentum = df[['close', 'momentum']].tail(5)
        for idx, row in recent_momentum.iterrows():
            print(f" 收盘价: {row['close']} | 动量:{row['momentum']:.4f}({row['momentum']:.2%})")

        # 6. 判断动量方向
        print(f" \n6. 动量方向统计")
        positive_count = (df['momentum'] > 0).sum()
        negative_count = (df['momentum']< 0).sum()
        zero_count = (df['momentum'] == 0).sum()

        print(f" 动量 > 0: {positive_count}天({positive_count/momentum_count*100:.1f}%)")
        print(f" 动量 < 0: {negative_count}天({negative_count/momentum_count*100:.1f}%)")
        print(f" 动量 = 0: {zero_count}天")

        # 7. 保存计算结果
        print(f" \n7. 保存计算结果")
        self.current_df = df
        print(f" 动量已添加到 DataFrame")
        print(f" 新增列: 'momentum' (MA{self.momentum_window}动量)")

        print(f" \n" + "-" * 70)
        print(f" 方法4完成: 动量指标计算成功")
        print('-' * 70)
        print(f" 动量窗口:{self.momentum_window}天")
        print(f" 有效数据点: {momentum_count}个")
        print(f" 平均动量: {momentum_mean:.4f}")
        return df

    def generate_signals(self, df):
        """生成交易信号 (动量>0买入, 动量<0卖出)"""
        print('=' * 80)
        print(f" 方法5: 生成交易信号")
        print('=' * 80)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据")
            return None

        if 'momentum' not in df.columns:
            print(f" 没有找到动量列")
            return None
        print(f" 数据形状: {df.shape[0]} 行")
        print(f" 动量有效数据: {df['momentum'].count()}个")

        # 2. 初始化信号列
        print(f" \n2. 信号列")
        df['signal'] = 0
        df['position'] = 0
        print(f" signal 列已创建 (0=无信号)")
        print(f" position 列已创建 (0 = 空仓)")

        # 3. 生成交易信号
        print(f" \n3. 生成交易信号")
        print(f" 规则:")
        print(f" 动量 > 0 (上涨趋势) -> 买入信号 signal = 1")
        print(f" 动量 < 0 (下跌趋势) -> 卖出信号 signal = -1")
        print(f" 动量 = 0 -> 无信号 signal = 0")

        # 生成信号
        df.loc[df['momentum'] > 0, 'signal'] = 1    # 买入
        df.loc[df['momentum'] < 0, 'signal'] = -1   # 卖出

        # 统计信号数量
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        print(f" \n 信号生成完成")
        print(f" 买入信号 (动量>0): {buy_signals}次")
        print(f" 卖出信号 (动量<0): {sell_signals}次")

        # 4.计算持仓
        print(f" \n4. 计算持仓")
        print(f" 规则: 遇到买入信号 -> 持仓=1, 遇到卖出信号 -> 持仓=0")
        position = 0
        positions = []

        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:   #买入信号
                position = 1
            elif df['signal'].iloc[i] == -1: # 卖出信号
                position = 0
            positions.append(position)

        df['position'] = positions

        # 统计持仓天数
        position_days = (df['position'] == 1).sum()
        print(f" 持仓计算完成")
        print(f" 持仓天数: {position_days} 天")
        print(f" 持仓比例: {position_days/len(df)*100:.1f}%")

        # 5. 显示最近10天的信号
        print(f" \n5. 最近10天的信号")
        print('=' * 80)
        print(f" {'日期':<12}{'收盘价':<10}{'动量':<12}{'信号':<8}{'持仓':<6}")
        print(f" =" * 80)

        # 获取最近10天数据
        recent = df.tail(10)
        for idx, row in recent.iterrows():
            # 获取日期
            if self.date_col and self.date_col in df.columns:
                date_str = row[self.date_col].strftime('%Y-%m-%d')
            else:
                date_str = str(idx)
            close_val = row['close']
            momentum = row['momentum'] if pd.notna(row['momentum']) else 0
            signal = row['signal']
            position = row['position']

            # 信号显示
            signal_str = "买入" if signal == 1 else "卖出" if signal == -1 else "无"
            position_str = "持仓" if position == 1 else "空仓"

            print(f" {date_str:<12}{close_val:<10.2f}{momentum:<12.4f}{signal_str:<8}{position_str:<6}")

        # 6. 显示交易记录
        print(f" \n6. 显示交易记录")
        print(f' -' * 60)

        # 找出信号变化的点
        trades = df[df['signal'] != 0].copy()

        if len(trades) >0:
            print(f" 共有 {len(trades)} 次交易信号")
            print(f" 买入信号: {buy_signals} 次")
            print(f" 卖出信号: {sell_signals} 次")

            # 显示前5次交易
            print(f" \n 前5次交易: ")
            for i, (idx, row) in enumerate(trades.head(5).iterrows(), 1):
                if self.date_col and self.date_col in df.columns:
                    date_str = row[self.date_col].strftime("%Y-%m-%d")
                else:
                    date_str = str(idx)
                signal_type = '买入' if row['signal'] == 1 else "卖出"
                price = row['close']
                momentum = row['momentum']
                print(f'{i}.{date_str} - {signal_type} (价格:{price:.2f}, 动量:{momentum:.4f})')
        else:
            print(f" 没有交易信号生成")

        # 7. 保存结果
        print(f" \n7. 保存结果")
        self.current_df = df
        print(f" 信号已保存到 self.current_df")
        print(f' 新添加: signal (交易信号)')
        print(f" 新添加: position (持仓状态)")

        print(f" \n" + "=" * 80)
        print(f" 方法5完成: 交易信号生成完成")
        print('=' * 80)
        print(f" 总买入信号: {buy_signals} 次")
        print(f" 总卖出信号: {sell_signals} 次")
        print(f" 持仓天数: {position_days}/ {len(df)} = {position_days/len(df)*100:.1f}%")
        return df

    def calculate_strategy_returns(self, df):
        """计算策略收益"""
        print(f" \n" + "=" * 80)
        print(f" 方法6: 计算策略收益 ")
        print('=' * 80)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据")
            return None

        if 'position' not in df.columns:
            print(f" 没有持仓数据, ")
            return None

        if '日收益率' not in df.columns:
            print(f" 没有日收益率数据")
            return None

        print(f" 数据形状: {df.shape[0]} 行")
        print(f" 持仓数据: {(df['position'] ==1).sum()} 天持仓")

        # 2. 计算策略收益率
        print(f" \n2. 计算策略收益率")
        print(f" 公式: 策略日收益率 = 持仓状态 x 股票日收益率")
        print(f" 说明: 持仓时获得股票收益, 空仓时收益为 0")
        df['strategy_return'] = df['position'] * df['日收益率']

        # 统计策略收益率
        strategy_returns = df['strategy_return'].dropna()
        print(f" 策略收益率计算完成")
        print(f" 有效数据点: {len(strategy_returns)}")
        print(f" 平均日收益率: {strategy_returns.mean():.6f}")
        print(f" 日收益率标准差: {strategy_returns.std():.6f}")

        # 3. 计算累计收益
        print(f' \n3. 计算累计收益')
        print(f" 公式: 累计收益=(1+日收益率)的累计乘积")

        # 策略累计收益
        df['strategy_cumulative'] = (1+df['strategy_return']).cumprod()
        # 基准累计收益 (买入持有)
        df['benchmark_cumulative'] = (1+ df['日收益率']).cumprod()

        final_strategy = df['strategy_cumulative'].iloc[-1]
        final_benchmark = df['benchmark_cumulative'].iloc[-1]

        print(f" 累计收益计算完成")
        print(f" 最终策略净值: {final_strategy:.4f}")
        print(f" 最终基准净值: {final_benchmark:.4f}")

        # 4. 计算总收益率
        print(f" \n4. 计算总收益率")
        print(f" 公式: 总收益率 = 最终净值 - 1")

        strategy_total_return = final_strategy - 1
        benchmark_total_return = final_benchmark - 1
        excess_return = strategy_total_return - benchmark_total_return
        print(f" 策略总收益率: {strategy_total_return:.2%}")
        print(f" 基准总收益率: {benchmark_total_return:.2%}")
        print(f" 超额收益: {excess_return:.2%}")

        if excess_return > 0:
            print(f" 策略跑赢基准: {excess_return:.2%}")
        elif excess_return < 0:
            print(f" 策略跑输基准: {abs(excess_return):.2%}")
        else:
            print(f" 策略与基准持平")

        # 5. 计算年化收益率
        print(f" \n5. 计算年化收益率")
        print(f' 公式: 年化收益率 = (1 + 总收益率)^(252/天数) - 1')

        trading_days = len(df)
        annual_factor = 252 / trading_days

        strategy_annual = (1+ strategy_total_return) ** annual_factor -1
        benchmark_annual = (1+benchmark_total_return) ** annual_factor -1

        print(f" 策略年化收益率: {strategy_annual}")
        print(f" 基准年化收益率: {benchmark_annual}")

        # 6. 计算策略胜率
        print(f" \n6. 计算策略胜率")
        print(f" 公式: 胜率 = 盈利交易日 / 总交易日")

        trading_days_df = df[df['position'] ==1].copy()
        if len(trading_days_df) > 0:
            winning_days = (trading_days_df['日收益率'] > 0).sum()
            win_rate = winning_days/ len(trading_days_df)

            print(f" 持仓交易日: {len(trading_days_df)} 天")
            print(f" 盈利天数: {winning_days} 天")
            print(f" 亏损天数: {len(trading_days_df) - winning_days} 天")
            print(f" 胜率: {win_rate:.2%}")
        else:
            win_rate = 0
            print(f" 没有持仓交易日")

        # 7. 计算盈亏比
        print(f' \n7. 计算盈亏比')
        print(f" 公式: 盈亏比 = 平均盈利 / 平均亏损")

        if len(trading_days_df) > 0:
            avg_profit = trading_days_df[trading_days_df['日收益率']>0]['日收益率'].mean() if winning_days >0 else 0
            avg_loss = abs(trading_days_df[trading_days_df['日收益率']<0]['日收益率'].mean() if (len(trading_days_df)-winning_days) > 0 else 0)

            if avg_loss >0:
                profit_loss_ratio = avg_profit / avg_loss
                print(f" 平均盈利: {avg_profit}")
                print(f" 平均亏损: {avg_loss}")
                print(f" 盈亏比: {profit_loss_ratio}")
            else:
                profit_loss_ratio = 0
                print(f" 无亏损记录")
        else:
            profit_loss_ratio = 0

        # 8. 计算最大回撤
        print(f" \n8. 计算最大回撤")
        print(f" 公式: 最大回撤 = (当前净值 - 历史最高净值) / 历史最高净值")

        # 策略最大回撤
        strategy_peak = df['strategy_cumulative'].expanding().max()
        strategy_drawdown = (df['strategy_cumulative'] - strategy_peak) / strategy_peak
        strategy_max_dd = strategy_drawdown.min()

        # 基准最大回撤
        benchmark_peak = df['benchmark_cumulative'].expanding().max()
        benchmark_drawdown = (df['benchmark_cumulative'] - benchmark_peak) / benchmark_peak
        benchmark_max_dd = benchmark_drawdown.min()

        print(f" 策略最大回撤: {strategy_max_dd}")
        print(f" 基准最大回撤: {benchmark_max_dd}")

        # 9. 计算夏普比率
        print(f' \n9. 计算夏普比率')
        print(f" 公式: 夏普比率 = (策略年化收益率 - 无风险利率) / 策略年化波动率")

        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        risk_free_rate = 0.02  # 假设无风险利率为2%

        if strategy_volatility > 0:
            sharpe_ratio = (strategy_annual - risk_free_rate) / strategy_volatility
            print(f" 策略年化波动率: {strategy_volatility}")
            print(f" 夏普比率: {sharpe_ratio}")
        else:
            sharpe_ratio = 0
            print(f" 无法计算夏普比率 (波动率为0)")

        # 10. 汇总策略指标
        print(f' \n10. 策略指标汇总')
        print(f' -' * 80)
        print(f" 策略总收益率: {strategy_total_return}")
        print(f" 策略总收益率: {strategy_total_return:>12.2%}")
        print(f" 基准总收益率: {benchmark_total_return:}")
        print(f" 基准总收益率: {benchmark_total_return:>12.2%}")
        print(f" 超额收益: {excess_return}")
        print(f" 超额收益: {excess_return:>15.2%}")
        print(f" 策略年化收益率: {strategy_annual}")
        print(f" 策略年化收益率: {strategy_annual:>12.2%}")
        print(f" 基准年化收益率: {benchmark_annual}")
        print(f" 基准年化收益率: {benchmark_annual:.2%}")
        print(f" 最大回撤: {strategy_max_dd}")
        print(f" 最大回撤: {strategy_max_dd:.2%}")
        print(f" 胜率: {win_rate}")
        print(f" 胜率: {win_rate:.2%}")
        print(f" 盈亏比: {profit_loss_ratio}")
        print(f" 盈亏比: {profit_loss_ratio:.2f}")
        print(f" 夏普比率: {sharpe_ratio}")
        print(f" 夏普比率: {sharpe_ratio:.4f}")
        print(f"-" * 80)

        # 11. 保存策略指标
        print(f" \n11. 保存策略指标")
        self.strategy_metrics = {
            '股票代码': self.current_symbol,
            '动量窗口': self.momentum_window,
            '策略总收益率': strategy_total_return,
            '基准总收益率': benchmark_total_return,
            '超额收益': excess_return,
            '策略年化收益率': strategy_annual,
            '基准年化收益率': benchmark_annual,
            '策略年化波动率': strategy_volatility,
            '最大回撤': strategy_max_dd,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '夏普比率': sharpe_ratio,
            '买入信号次数': (df['signal']==1).sum(),
            '卖出信号次数': (df['signal']==-1).sum(),
            '持仓天数': (df['position']==1).sum(),
            '总交易天数': len(df)
        }

        print(f" 策略指标已保存到self.strategy_metrics")
        print(f" 共 {len(self.strategy_metrics)} 个指标")

        # 12. 保存结果到对象属性
        self.current_df = df
        print(f" \n" +'-' * 80)
        print(f" 方法6完成: 策略收益计算成功")
        print('-' * 80)
        print(f" 策略总收益: {strategy_total_return:.2%}")
        print(f" 基准总收益: {benchmark_total_return:.2%}")
        print(f" 超额收益: {excess_return:.2%}")
        print(f" 夏普比率: {sharpe_ratio:.4f}")
        return df

    def plot_strategy(self, df, symbol, save=True, show=True):
        """绘制动量策略图表"""
        print("\n" + "=" * 70)
        print("📹  方法7：绘制策略图表")
        print("=" * 70)

        # 1. 检查数据
        print(f" \n1. 检查数据")
        if df is None:
            print(f" 没有数据")
            return None

        print(f' 数据形状: {df.shape[0]}行')
        print(f" 股票代码: {symbol}")

        # 2. 获取字体
        title_font = self.font_prop if hasattr(self, 'font_prop') else None
        print(f" 中文字体: {'已配置' if title_font else '使用默认'}")

        # 3. 准备数据
        print(f" \n2. 准备数据")
        if len(df) > 500:
            df_plot = df.tail(500).copy()
            print(f" 数据超过500天, 取最近500天绘制")
        else:
            df_plot = df.copy()
            print(f' 使用全部{len(df_plot)} 天数据')

        # 4. 获取日期列
        print(f" \n4. 获取日期列")
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            x = df_plot[self.date_col]
            x_label = '日期'
            print(f" 使用日期列: {self.date_col}")
        else:
            x = range(len(df_plot))
            x_label = '交易日'
            print(f' 没有日期列, 使用索引')

        # 5. 创建图表
        print(f' \n5. 创建图表')
        fig, axes = plt.subplots(2,1,figsize=(14,10))
        fig.suptitle(f"{symbol}-动量策略回测", fontsize=14, fontweight='bold')
        print(f" 图表大小: 14 x 10英寸")

        # 6. 第一张图: 价格和动量信号
        print(f" \n6. 绘制价格和动量信号图")
        ax1 = axes[0]

        # 绘制收盘价
        ax1.plot(x, df_plot['close'], linewidth=1.5, color='black', label='收盘价', alpha=0.7)
        print(f" 收盘价曲线已添加")

        # 绘制动量线 (归一化到价格区间)
        price_range = df_plot['close'].max() - df_plot['close'].min()
        momentum_scaled = df_plot['momentum'] * price_range + df_plot['close'].mean()
        ax1.plot(x, momentum_scaled, linewidth=1.2, color='orange', label='动量', alpha=0.7)
        print(f"   ✅ 动量曲线已添加")

        # 标记买入点 (动量 >0)
        buy_signals = df_plot[df_plot['signal'] == 1]
        if len(buy_signals) > 0:
            if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
                buy_x = buy_signals[self.date_col]
            else:
                buy_x = buy_signals.index
            ax1.scatter(buy_x, buy_signals['close'], color='green', s=100, marker='^', zorder=5, label='买入信号')
            print(f" 买入信号已标记: {len(buy_signals)} 个")

        # 标记卖出点 (动量 <0)
        sell_signals = df_plot[df_plot['signal'] == -1]
        if len(sell_signals) > 0:
            if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
                sell_x = sell_signals[self.date_col]
            else:
                sell_x = sell_signals.index
            ax1.scatter(sell_x, sell_signals['close'], color='red', s=100, marker='v', zorder=5, label='卖出信号')
            print(f" 买入信号已标记: {len(sell_signals)} 个")

        ax1.set_ylabel('价格', fontsize=12)
        ax1.set_title(f"{symbol} - 价格走势与动量信号", fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            ax1.tick_params(axis='x', rotation=45)
        print(f" 价格图表完成")

        # 7. 第二张图: 策略净值 VS 基准净值
        print(f" \n7. 绘制净值曲线图")
        ax2 = axes[1]

        # 绘制策略净值
        if 'strategy_cumulative' in df_plot.columns:
            ax2.plot(x, df_plot['strategy_cumulative'], linewidth=2, color='green', label='策略净值', alpha=0.9)
            print(f' 策略净值曲线已添加')

        # 绘制基准净值
        if 'benchmark_cumulative' in df_plot.columns:
            ax2.plot(x, df_plot['benchmark_cumulative'], linewidth=2, color='blue', label='基准净值', alpha=0.9)
            print(f" 基准净值曲线已添加")

        # 标注最终净值
        final_strategy = df_plot['strategy_cumulative'].iloc[-1] if 'strategy_cumulative' in df_plot.columns else 1
        final_benchmark = df_plot['benchmark_cumulative'].iloc[-1] if 'benchmark_cumulative' in df_plot.columns else 1

        # 计算超额收益
        excess = final_strategy - final_benchmark
        ax2.text(0.02, 0.95, f"策略净值: {final_strategy:.4f}", transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='green', alpha=0.5))
        ax2.text(0.02, 0.88, f"基准净值: {final_benchmark:.4f}", transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))

        # 标注超额收益
        if excess > 0:
            color = 'green'
            label = f" 超额收益: +{excess:.2%}"
        else:
            color = 'red'
            label = f" 超额收益: {excess:.2%}"

        ax2.text(0.02, 0.81, label, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.5))

        ax2.set_ylabel('净值', fontsize=12)
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_title('策略净值 VS 基准净值', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.2)
        if hasattr(self, 'date_col') and self.date_col in df_plot.columns:
            ax2.tick_params(axis='x', rotation=45)
        print(f" 净值曲线图完成")

        # 8. 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)

        #===============添加保存图表的代码======================
        if save:
            # 确保图表目录存在
            self.charts_dir.mkdir(parents=True, exist_ok=True)
            # 保存图表
            chart_path = self.charts_dir / f"{symbol}_动量策略.png"
            fig.savefig(chart_path, dpi=150, bbox_inches='tight')
            print(f" 图表已保存: {chart_path}")

        # 9. 显示图表
        if show:
            plt.show()
        else:
            plt.close(fig)

        print('\n' + '=' * 80)
        print(f" 方法7完成: 策略图表绘制成功")
        print('=' * 80)
        print(f" 买入信号: {len(buy_signals)} 个")
        print(f" 卖出信号: {len(sell_signals)} 个")
        print(f" 最终策略净值: {final_strategy}")
        print(f" 最终基准净值: {final_benchmark}")


        return fig

    def batch_backtest(self, momentum_window=20, plot_all=False, plot_top=10, show_plot=True):
        """批量回测所有股票
        Args:
                momentum_window: 动量窗口
                plot_all: 是否绘制所有股票的图表
                plot_top: 绘制表现最好的前N个股票
                show_plot: 是否显示图表窗口"""

        print('\n' + '=' * 80)
        print(f' 方法8: 批量回测所有股票')
        print('=' * 80)

        # 1. 检查文件列表
        print(f' \n1. 检查文件列表')
        if not hasattr(self, 'all_files') or len(self.all_files) == 0:
            print(f" 没有找到文件列表")
            return None

        total = len(self.all_files)
        print(f' 共有{total}个股票想要回测')
        print(f" 动量窗口: {momentum_window}天")

        # 2. 设置当前动量窗口
        self.momentum_window = momentum_window

        # 3. 初始化结果存储
        print(f" \n2. 初始化结果存储")
        all_results = []
        success_count = 0
        failed_count = 0
        print(f" 创建空列表存储结果")

        # 4. 开始循环处理
        print(f" \n4. 开始批量回测")
        print(f" =" * 50)

        for i, file_path in enumerate(self.all_files, 1):
            # 提取股票代码
            symbol = file_path.stem.split('_')[0]
            print(f" \n [{i}/{total}] 回测: {symbol}")

            try:
                """这里把其它的def 调用到这里显示循环"""
                # 加载数据
                df = self.load_data(file_path)
                if df is None:
                    print(f" 数据加载失败")
                    failed_count += 1
                    continue

                # 计算动量
                df = self.calculate_momentum(df)
                if df is None:
                    print(f" 信号生成失败")
                    failed_count += 1
                    continue

                # 添加: 生成交易信号
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
                    '动量窗口': momentum_window,
                    '策略总收益': self.strategy_metrics['策略总收益率'],
                    '基准总收益': self.strategy_metrics['基准总收益率'],
                    '超额收益': self.strategy_metrics['超额收益'],
                    '策略年货收益': self.strategy_metrics['策略年化收益率'],
                    '最大回撤': self.strategy_metrics['最大回撤'],
                    '胜率': self.strategy_metrics['胜率'],
                    '盈亏比': self.strategy_metrics['盈亏比'],
                    '夏普比率': self.strategy_metrics['夏普比率'],
                    '买入次数': self.strategy_metrics['买入信号次数'],
                    '卖出次数': self.strategy_metrics['卖出信号次数'],
                    '持仓天数': self.strategy_metrics['持仓天数'],
                    '数据天数': len(df)
                }
                all_results.append(result)
                success_count += 1
                print(f' 策略总收益: {result["策略总收益"]:.2%}')
                print(f" 夏普比率: {result['夏普比率']:.4f}")
            except Exception as e:
                print(f" 回撤失败: {e}")
                failed_count += 1
                continue

        # 5. 显示汇总结果
        print('\n' + "-" * 70)
        print(f' \n5. 批量回撤完成')
        print(f"-" * 70)
        print(f" 总股票数: {total}")
        print(f" 成功回测: {success_count}")
        print(f" 失败: {failed_count}")

        if not all_results:
            print(f" 没有成功的回测结果")
            return None

        # 6. 转换为DataFrame并排序
        print(f" \n6. 整理结果")
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('策略总收益', ascending=False)
        print(f" 共{len(results_df)} 个有效结果")

        # 7. 显示前10名
        print(f" \n7. 表现最好的前10名股票")
        print(f" =" * 70)
        print(f" {'排名':<4} {'股票代码':<8}{'策略收益':<10}{'超额收益':<10}{'夏普比率':<10}{'胜率':<8}")
        print(f" =" * 70)
        for idx, row in results_df.head(10).iterrows():
            rank = list(results_df.index).index(idx) + 1
            print(f" {rank:<4}{row['股票代码']:<8}"
                  f"{row['策略总收益']:>9.2%}{row['超额收益']:>9.2%}"
                  f"{row['夏普比率']:>9.4f}{row['胜率']:>7.2%}")

        # 8. 统计汇总
        print(f" \n8. 统计汇总")
        print(f" -" * 70)
        print(f" 平均策略收益: {results_df['策略总收益'].mean():.2%}")
        print(f" 平均超额收益: {results_df['超额收益'].mean():.2%}")
        print(f" 平均夏普比率: {results_df['夏普比率'].mean():.4f}")
        print(f" 平均胜率: {results_df['胜率'].mean():.2%}")
        print(f" 平均最大回撤: {results_df['最大回撤'].mean():.2%}")

        # 9. 绘制图表 (这里就是循环绘制图表)
        print(f" \n9. 绘制策略图表")
        print('=' * 70)
        if plot_all:
            symbols_to_plot = results_df['股票代码'].tolist()
            print(f" 绘制所有 {len(symbols_to_plot)} 个股票图表..............")
        else:
            symbols_to_plot = results_df.head(plot_top)['股票代码'].tolist()
            print(f' 绘制表现最好的前{plot_top}名股票图表.......')

        plotted_count = 0
        for symbol in symbols_to_plot:
            # 找到对应的文件路径
            file_path = None
            for f in self.all_files:
                if f.stem.split('_')[0] == symbol:
                    file_path = f
                    break
            if file_path:
                print(f' 绘制: {symbol}')
                try:
                    # 重新加载数据并计算
                    df_temp = self.load_data(file_path)
                    if df_temp is not None:
                        df_temp = self.calculate_momentum(df_temp)
                        df_temp = self.generate_signals(df_temp)
                        df_temp = self.calculate_strategy_returns(df_temp)
                        self.plot_strategy(df_temp, symbol, save=True, show=show_plot)
                        plotted_count += 1
                except Exception as e:
                    print(f" 绘制失败: {e}")
            else:
                print(f" 未找到{symbol}的文件")
        print(f" 已绘制{plotted_count} 个股票图表")

        # 10. 返回结果
        print('\n' + '=' * 80)
        print(f' 方法8完成: 批量回撤成功')
        print('=' * 80)
        print(f" 成功回撤: {success_count} 个股票")

        return results_df

    def save_all_results(self, results_df):
        """保存所有回测结果到Excel文件"""
        print('\n' + '=' * 80)
        print(f' 方法9: 保存所有回测结果')
        print('=' * 80)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d")

        # 1. 创建保存目录
        print(f" \n1. 创建保存目录")
        save_dir = self.output_dir / f"动量回测结果_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f" 创建目录: {save_dir}")

        saved_files = []

        # 2. 保存批量回测结果
        print(f" \n2. 保存批量回测结果")
        if results_df is not None and len(results_df) > 0:
            batch_file = save_dir / "批量回测结果.xlsx"

            with pd.ExcelWriter(batch_file, engine='openpyxl') as writer:
                # sheet 1. 按收益排序
                results_df_sorted = results_df.sort_values('策略总收益', ascending=False)
                results_df_sorted.to_excel(writer, sheet_name='按收益排序', index=False)

                # sheet 2. 按夏普比率排序
                if '夏普比率' in results_df.columns:
                    results_df_sharpe = results_df.sort_values('夏普比率', ascending=False)
                    results_df_sharpe.to_excel(writer, sheet_name='按夏普比率排序', index=False)

                # sheet 3. 统计汇总
                summary_data = {
                    '指标': ['总股票数', '平均策略收益', '平均超额收益', '平均夏普比率',
                             '平均胜率', '平均最大回撤', '最佳股票', '最佳收益'],
                    '数值': [
                        len(results_df),
                        results_df['策略总收益'].mean(),
                        results_df['超额收益'].mean(),
                        results_df['夏普比率'].mean() if "夏普比率" in results_df.columns else 0,
                        results_df['胜率'].mean() if '胜率' in results_df.columns else 0,
                        results_df['最大回撤'].mean() if '最大回撤' in results_df.columns else 0,
                        results_df.iloc[0]['股票代码'],
                        results_df.iloc[0]['策略总收益']
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='统计汇总', index=False)

            file_size = batch_file.stat().st_size / 1024
            print(f" 已保存; {batch_file.name} ({file_size:.1f} KB)")
            saved_files.append(str(batch_file))

        # 3. 保存各股票详细结果(前10名)
        print(f' \n3. 保存各股票详细结果 (前10名)')
        if results_df is not None and len(results_df) > 0:
            detail_dir = save_dir / "各股票详细结果"
            detail_dir.mkdir(parents=True, exist_ok=True)

            saved_count = 0
            for idx, row in results_df.head(10).iterrows():
                symbol = row['股票代码']
                print(f" 处理: {symbol}........", end=" ")

                # 找到对应的原始数据文件
                file_path = None
                for f in self.all_files:
                    if f.stem.split('_')[0] == symbol:
                        file_path = f
                        break

                if file_path:
                    try:
                        # 重新加载并计算该股票的数据
                        df_detail = self.load_data(file_path)
                        if df_detail is not None:
                            df_detail = self.calculate_momentum(df_detail)
                            df_detail = self.generate_signals(df_detail)
                            df_detail = self.calculate_strategy_returns(df_detail)

                            detail_file = detail_dir / f"{symbol}_详细结果.xlsx"
                            with pd.ExcelWriter(detail_file, engine='openpyxl') as writer:
                                # sheet 1. 策略指标
                                metrics_df = pd.DataFrame([self.strategy_metrics])
                                metrics_df.to_excel(writer, sheet_name='策略指标', index=False)

                                # sheet 2. 交易信号
                                signals_df = df_detail[df_detail['signal'] != 0].copy()
                                if len(signals_df) > 0:
                                    cols = ['Date', 'close', 'momentum', 'signal']
                                    cols = [c for c in cols if c in signals_df.columns]
                                    signals_df[cols].to_excel(writer, sheet_name='交易信号', index=False)

                                # sheet 3. 每日数据 (最近100天)
                                daily_cols = ['Date', 'close', 'momentum', 'strategy_cumulative',
                                              'benchmark_cumulative', 'positive']
                                daily_cols = [c for c in daily_cols if c in df_detail.columns]
                                df_detail[daily_cols].tail(100).to_excel(writer, sheet_name='每日数据', index=False)
                            saved_count += 1
                            print('完成')
                    except Exception as e:
                        print(f" 失败 {e}")
                else:
                    print(f" 文件不存在")
            print(f'已保存{saved_count}个 股票的详细结果')

        # 4. 复制股票到保存目录
        print(f" \n4. 保存图表")
        chart_dir = save_dir / "图表"
        chart_dir.mkdir(parents=True, exist_ok=True)

        if self.charts_dir.exists():
            import shutil
            chart_files = list(self.charts_dir.glob("*.png"))
            print(f" 找到 {len(chart_files)} 个图表文件")

            for chart_file in chart_files:
                dest_file = chart_dir / chart_file.name
                shutil.copy2(chart_file, dest_file)
                print(f" 复制; {chart_file.name}")
            print(f" 已复制 {len(chart_files)} 个图表到{chart_dir}")

        # 5. 生成汇总报告
        print(f" \n5. 生成汇总报告")
        report_file = save_dir / "动量策略汇总报告.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('=' * 70 + "\n")
            f.write("动量策略回测汇总报告\n")
            f.write('=' * 70 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y%m%d')}\n")
            f.write(f"动量窗口: {self.momentum_window}天\n\n")

            if results_df is not None and len(results_df) > 0:
                f.write("-" * 50 + "\n")
                f.write('批量回测统计\n')
                f.write("-" * 50 + "\n")
                f.write(f" 总股票数: {len(results_df)}\n")
                f.write(f" 平均策略收益: {results_df['策略总收益'].mean():.2%}\n")
                f.write(f" 平均超额收益: {results_df['超额收益'].mean():.2%}\n")
                if "夏普比率" in results_df.columns:
                    f.write(f" 平均夏普比率: {results_df['夏普比率'].mean():.4f}\n")
                if '胜率' in results_df.columns:
                    f.write(f" 平均胜率: {results_df['胜率'].mean():.2%}\n")
                if "最大回撤" in results_df.columns:
                    f.write(f" 平均最大回撤: {results_df['最大回撤'].mean():.2%}\n\n")

                f.write("-" * 50 + "\n")
                f.write('表现最好的前10名\n')
                f.write("-" * 50 + "\n")
                for idx, row in results_df.head(10).iterrows():
                    f.write(f"{row['股票代码']}: 收益{row['策略总收益']:.2%}, ")
                    if "夏普比率" in row:
                        f.write(f" 夏普 {row['夏普比率']:.4f}, ")
                    if "胜率" in row:
                        f.write(f" 胜率 {row['胜率']:.2%}\n")
        print(f" 已保存: {report_file.name}")
        saved_files.append(str(report_file))

        # 6. 显示保存结果
        print("\n" + "-" * 80)
        print(f" 方法9完成: 所有结果已保存")
        print('-' * 80)
        print(f" 保存目录: {save_dir}")
        print(f" - 批量回测结果.xlsx")
        print(f" - 各股票详细结果/({saved_count} 个文件)")
        print(f" - 图表/ ({len(chart_files) if self.charts_dir.exists() else 0} 个)")
        print(f" - 动量策略汇总报告.txt")

        return save_dir

# 测试代码
if __name__ == "__main__":
    strategy = MomentumStrategy(momentum_window=20)

    files = strategy.find_data_files()

    if files:
        # # 3. 加载测试股票数据 (测试单个, 不是循环)
        # df = strategy.load_data(strategy.test_file)
        # # 4. 计算动量
        # df = strategy.calculate_momentum(df)
        # # 5. 生成交易信号
        # df = strategy.generate_signals(df)
        # df = strategy.calculate_strategy_returns(df)
        # # 绘制图表
        # strategy.plot_strategy(df, strategy.test_symbol, save=True, show=True)

        # 批量回撤所有股票 (这里会循环处理所有股票并绘制图表)
        all_results = strategy.batch_backtest(
            momentum_window=20,
            plot_all = False,   # 不绘制所有 (避免太多窗口)
            plot_top = 10,      # 只绘制表现最好的前10名
            show_plot=True     # 不显示窗口, 只保存   (True就会显示图表, false就不会显示图表, 只保存)
        )

        if all_results is not None:
            print(f" \n 批量回测完成, 共 {len(all_results)} 个股票")
            print(f" 最佳股票; {all_results.iloc[0]['股票代码']} (收益: {all_results.iloc[0]['策略总收益']:.2%})")
            save_dir = strategy.save_all_results(all_results)
            print(f" \n 所有结果已保存到: {save_dir}")


"""
## 第10天：策略二 —— 动量策略

**任务目标**：
- 设计动量策略逻辑（动量>0买入，动量<0卖出）
- 计算动量信号并生成策略持仓
- 计算策略收益并与基准对比
- 调整动量窗口参数，观察策略效果变化

**实现方案**：
1. **策略设计**：
   - **动量指标计算**：使用`pct_change(periods=N)`计算过去N天的累计收益率
   - **买入信号**：动量 > 0（上涨趋势）
   - **卖出信号**：动量 < 0（下跌趋势）
   - **持仓规则**：动量>0时持仓，动量<0时空仓

2. **核心计算流程**：
   - **动量计算**：`动量 = (当前价格 - N天前价格) / N天前价格`
   - **交易信号生成**：基于动量正负直接产生买卖信号
   - **策略收益计算**：策略日收益率 = 持仓状态 × 股票日收益率
   - **绩效指标**：总收益率、年化收益率、夏普比率、最大回撤、胜率、盈亏比

3. **参数优化**：
   - 动量窗口可调整（默认20天）
   - 不同窗口长度影响信号频率和策略表现
   - 短窗口：信号更灵敏，交易更频繁
   - 长窗口：信号更平滑，交易次数减少

4. **批量回测引擎**：
   - 循环处理所有股票数据文件
   - 统一使用指定动量窗口进行回测
   - 生成股票表现排名（按策略收益排序）
   - 自动绘制表现最佳股票的图表

5. **可视化分析**：
   - **图表一**：价格走势 + 动量曲线（归一化显示）+ 买卖信号标记
   - **图表二**：策略净值 vs 基准净值对比曲线
   - 标注最终净值、超额收益等关键指标

6. **结果保存系统**：
   - 保存批量回测结果Excel（按收益排序、按夏普比率排序）
   - 保存各股票详细结果（策略指标、交易信号、每日数据）
   - 自动复制图表到结果目录
   - 生成动量策略汇总报告文本文件

**核心代码结构**：
```python
class MomentumStrategy:
    ├── __init__()                       # 初始化策略参数和目录
    ├── _setup_chinese_font()             # 配置中文字体
    ├── find_data_files()                 # 查找收益率数据文件
    ├── load_data()                       # 加载股票数据
    ├── calculate_momentum()              # 计算动量指标
    ├── generate_signals()                # 生成交易信号（动量正负）
    ├── calculate_strategy_returns()      # 计算策略收益和绩效指标
    ├── plot_strategy()                   # 绘制策略图表
    ├── batch_backtest()                  # 批量回测所有股票
    └── save_all_results()                # 保存所有结果到Excel
    ```
"""






