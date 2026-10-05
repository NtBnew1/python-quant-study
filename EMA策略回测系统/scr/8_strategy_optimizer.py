"""
文件: src/8_strategy_optimizer.py                                                                                                      一、优化目标
项目	                                     说明
目标	                                     从 750 种参数组合中，找出收益率最高、风险最低的参数组合
优化对象	                                     EMA策略（日线）
优化方法	                                     网格搜索（Grid Search）
评估指标	                                     平均收益率 + 夏普比率 + 胜率


二、优化参数范围
参数	                            取值范围	                                                                              组合数
EMA周期组合	            (3,10,30), (5,10,60), (5,15,30), (8,21,55), (10,20,60)	      5 种
止损比例	                            5%, 10%, 15%, 20%, 25%, 30%	                                      6 种
仓位比例	                            20%, 30%, 40%, 50%, 100%	                                      5 种
信号阈值	                            0%, 1%, 2%, 3%, 5%	                                                      5 种


总组合数: 5 × 6 × 5 × 5 = 750 种
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
import warnings

# ============= 循环所有股票 + 所有的参数, 时间大概要2-5小时 ==================
# ============= 现在要用multiprocessing 来加快进度 ========================
from multiprocessing import Pool
import os

warnings.filterwarnings('ignore')

from config import (
DATA_PROCESSED_DIR,
RESULTS_DIR,
INITIAL_CASH,
COMMISSION
)



class StrategyOptimizer:
    """ 策略优化类 """
    def __init__(self, project_dir):
        """初始化"""
        self.project_dir = project_dir
        self.data_dir = DATA_PROCESSED_DIR

        # 优化结果目录
        base_dir = RESULTS_DIR / "优化"
        base_dir.mkdir(parents=True, exist_ok=True)

        # 数据目录
        self.data_output_dir = base_dir / "数据"
        self.data_output_dir.mkdir(parents=True, exist_ok=True)

        # 图表目录
        self.charts_output_dir = base_dir / "图表"
        self.charts_output_dir.mkdir(parents=True, exist_ok=True)

        print(f'=' * 60)
        print(f" 策略参数优化")
        print(f" 数据目录: {self.data_output_dir}")
        print(f" 图表目录: {self.charts_output_dir}")
        print()

    # 现在是2250种
    def define_param_grid(self):
        """ 定义测试用的参数组合 (完整版: 2250 种组合)
        Returns:
            list: 参数组合列表
        """
        # 完整参数范围
        ema_combinations = [(3, 10, 30),(5,10,60),(5,15,30), (8,21,55),(10,20,60)]        # 5种 EMA
        signal_types = ['short_long', 'mid_long', 'short_mid_long']                       # 3种信号 (短 > 长)
        thresholds = [0.00, 0.01, 0.02, 0.03, 0.05]                                       # 5种阈值
        stop_losses = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]                                # 6种止损
        position_ratios = [0.20, 0.30, 0.40, 0.50, 1.00]                                  # 5种仓位

        # 生成所有组合
        all_params = []
        for ema, signal, threshold, stop_loss, position in product(
            ema_combinations, signal_types, thresholds, stop_losses, position_ratios
        ):
            all_params.append({
                'ema': ema,
                'signal': signal,
                'threshold': threshold,
                'stop_loss': stop_loss,
                'position_ratio': position,
            })

        print(f" 定义了 {len(all_params)} 种参数组合")
        for i, p in enumerate(all_params, 1):
            print(f" 组合{i}: EMA{p['ema']}, 信号{p['signal']}, 阈值{p['threshold']}, "
                  f"止损{p['stop_loss']*100:.0f}%, 仓位{p['position_ratio']*100:.0f}%")

        return all_params

    def calculate_ema_signals(self, df, ema_periods, signal_type, threshold):
        """计算EMA 和信号
        Args:
            df: 价格数据
            ema_periods: ema周期组合
            signal_type: 信号类型
            threshold: 信号阈值

        Returns:
            DataFrame: 添加了, EMA 和信号列的数据
        """

        df = df.copy()
        short_period, mid_period, long_period = ema_periods

        # 计算三种 EMA
        df[f'ema_{short_period}'] = df['Close'].ewm(span=short_period, adjust=False).mean()
        df[f'ema_{mid_period}'] = df['Close'].ewm(span=mid_period, adjust=False).mean()
        df[f'ema_{long_period}'] = df['Close'].ewm(span=long_period, adjust=False).mean()

        short_ema = df[f'ema_{short_period}']
        mid_ema = df[f'ema_{mid_period}']
        long_ema = df[f'ema_{long_period}']

        # 根据信号类型生成信号
        signal_col = 'signal'
        df[signal_col] = 0

        if signal_type == 'short_long':
            # 短期 > 长期 -> 买入
            buy_cond = (short_ema > long_ema) & ((short_ema - long_ema) / long_ema > threshold)
            sell_cond = (short_ema < long_ema) & ((short_ema - long_ema) / long_ema < -threshold)
        elif signal_type == 'mid_long':
            # 中期 > 长期 -> 买入
            buy_cond = (mid_ema > long_ema) & ((mid_ema - long_ema) / long_ema > threshold)
            sell_cond = (mid_ema < long_ema) & ((mid_ema - long_ema) / long_ema < -threshold)
        elif signal_type == 'short_mid_long':
            # 短期 > 中期 > 长期 -> 买入
            buy_cond = (short_ema > mid_ema) & (mid_ema > long_ema)
            sell_cond = (short_ema < mid_ema) & (mid_ema < long_ema)
        else:
            print(f" 未知信号类型: {signal_type}")
            return df

        df.loc[buy_cond, signal_col] = 1
        df.loc[sell_cond, signal_col] = -1

        return df

    def run_signal_backtest(self, df, ema_periods, signal_type, threshold,
                            stop_loss, position_ratio):
        """用一组参数跑回测
        Args:
            df: 价格数据
            ema_periods: EMA 周期组合
            signal_type: 信号类型
            threshold: 阈值
            stop_loss: 止损比例
            position_ratio: 仓位比例
        Returns:
            dict: 回测结果
        """

        # 1. 计算 EMA 和信号
        df_signals = self.calculate_ema_signals(df, ema_periods, signal_type, threshold)

        # 2. 自定义数据源
        class PandasDataWithSignal(bt.feeds.PandasData):
            lines = ('signal',)
            params = (
                ('datetime', None),
                ('open', 'Open'),
                ('high', 'High'),
                ('low', 'Low'),
                ('close', 'Close'),
                ('volume', 'Volume'),
                ('openinterest', -1),
                ('signal', 'signal'),
            )

        # 3. 定义策略
        class OptStrategy(bt.Strategy):
            params = (
                ('stop_loss', stop_loss),
                ('position_ratio', position_ratio),
            )

            def __init__(self):
                self.order = None
                self.entry_price = 0
                self.signal = self.datas[0].signal

            def next(self):
                if self.order:
                    return

                price = self.datas[0].close[0]
                signal = self.signal[0]

                # 买入
                if signal == 1 and not self.position:
                    size = int(self.broker.getcash() * self.params.position_ratio / price)
                    if size > 0:
                        self.order = self.buy(size=size)

                # 卖出
                elif signal == -1 and self.position:
                    self.order = self.sell(size=self.position.size)

                # 止损
                elif self.position and self.entry_price > 0:
                    loss_pct = (self.entry_price - price) / self.entry_price
                    if loss_pct >= self.params.stop_loss:
                        self.order = self.sell(size=self.position.size)

            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.entry_price = order.executed.price
                    else:
                        self.entry_price = 0
                self.order = None

        # 4. 创建回测引擎
        cerebro = bt.Cerebro()
        data = PandasDataWithSignal(dataname=df_signals)
        cerebro.adddata(data)
        cerebro.addstrategy(OptStrategy)

        cerebro.broker.setcash(INITIAL_CASH)
        cerebro.broker.setcommission(commission=COMMISSION)

        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

        # 5. 运行回测
        resutls = cerebro.run()
        strat = resutls[0]

        # 6. 提取结果
        returns = strat.analyzers.returns.get_analysis()
        sharpe = strat.analyzers.sharpe.get_analysis()

        final_cash = cerebro.broker.getvalue()
        total_return = returns.get('rtot', 0) * 100
        sharpe_ratio = sharpe.get('sharperatio', 0)

        result = {
            'final_cash': final_cash,
            'profit': final_cash - INITIAL_CASH,
            'profit_pct': (final_cash - INITIAL_CASH) / INITIAL_CASH * 100,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
        }

        return result

    def run_optimization(self, stock_code, params_list):
        """ 循环所有参数组合, 对一只股票跑回测
        Args:
            stock_code: 股票代码,
            params_list: 参数组合列表
        Returns:
            list: 所有组合的回测结果
            """
        print(f"\n{'='*60}")
        print(f' 优化股票: {stock_code}')
        print(f" 参数组合数: {len(params_list)}")
        print(f" =" +'80')

        # 读取股票数据
        stock_file = self.data_dir / f"{stock_code}.csv"
        if not stock_file.exists():
            print(f" 文件不存在: {stock_file}")
            return []

        df = pd.read_csv(stock_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)

        print(f" 数据: {len(df)} 行")

        all_results = []

        # 循环每种参数组合
        for i, p in enumerate(params_list, 1):
            print(f" \n[{i}/{len(params_list)}] EMA{p['ema']}, 信号{p['signal']}, 阈值{p['threshold']},"
                  f"止损{p['stop_loss']*100:.0f}%, 仓位{p['position_ratio']*100:.0f}%")

            result = self.run_signal_backtest(
                df,
                ema_periods=p['ema'],
                signal_type=p['signal'],
                threshold=p['threshold'],
                stop_loss=p['stop_loss'],
                position_ratio=p['position_ratio']
            )

            # 合并参数和结果
            record = {**p, **result}
            record['stock_code'] = stock_code
            all_results.append(record)

            print(f" 收益: ${result['profit']:,.2f} ({result['profit_pct']:.2f}%)")
            print(f" 夏普: {result['sharpe_ratio']:.3f}")



        return all_results

    def run_all_stocks_parallel(self, params_list, stock_codes=None, n_jobs=9):
        """ 多进程批量优化所有股票"""
        print(f" \n" + "=" * 60)
        print(f" 批量优化所有股票 (多进程加速)")
        print(f" 参数组合数: {len(params_list)}")
        print(f" 并行进程数: {n_jobs}")
        print(f" \n"+'=' * 60)

        if stock_codes is None:
            stock_files = list(self.data_dir.glob('*.csv'))
            stock_codes = [f.stem for f in stock_files]

        print(f" 股票数量: {len(stock_codes)}")
        print(f" 总回测次数: {len(stock_codes) * len(params_list)}")
        print(f'-' * 40)

        tasks = [(stock_code, params_list) for stock_code in stock_codes]

        all_results = []

        with Pool(processes=n_jobs) as pool:
            for i, stock_results in enumerate(pool.imap(run_stock_task, tasks), 1):
                all_results.extend(stock_results)
                print(f" [{i}/{len(stock_codes)}] 完成: {stock_codes[i-1]}")

                # 边跑边保存
                temp_df = pd.DataFrame(all_results)
                temp_file = self.data_output_dir / "optimization_results_temp.csv"
                temp_df.to_csv(temp_file, index=False)
                print(f" 已保存: {len(all_results)} 条")

        print(f" \n" + '=' * 60)
        print(f" 所有股票优化完成! ")
        print(f" 总结果数: {len(all_results)}")
        print(f' \n' + '=' * 60)

        return all_results

    # 专门写保存的代码.
    def save_results(self, results_df):
        """ 保存优化结果
        Args:
            results_df: 结果 DataFrame
            """

        print(f' \n' + '=' * 70)
        print(f" 保存优化结果")
        print(f" =" * 60)

        # 1. 保存完整结果
        results_file = self.data_output_dir / "optimization_results.csv"
        results_df.to_csv(results_file, index=False)
        print(f' 完整结果已保存: {results_file}')
        print(f" 共 {len(results_df)}条记录")

        # 2. 每只股票单独保存
        stock_dir = self.data_output_dir / "分股票"
        stock_dir.mkdir(parents=True, exist_ok=True)

        for stock_code in results_df['stock_code'].unique():
            stock_df = results_df[results_df['stock_code'] == stock_code]
            stock_file = stock_dir / f"{stock_code}.csv"
            stock_df.to_csv(stock_file, index=False)
            print(f" {stock_code}: {len(stock_df)} 条")

        # 3. 保存每只股票的最优参数
        idx = results_df.groupby('stock_code')['profit_pct'].idxmax()
        best_per_stock = results_df.loc[idx].copy()
        best_per_stock = best_per_stock.sort_values('profit_pct', ascending=False)

        best_stock_file = self.data_output_dir / "best_per_stock.csv"
        best_per_stock.to_csv(best_stock_file, index=False)
        print(f" 每只股票最优参数已保存: {best_stock_file}")

        # 4. 保存全局最优参数
        best_global = results_df.loc[results_df['profit_pct'].idxmax()]

        best_file = self.data_output_dir / "best_params.txt"
        with open(best_file, 'w', encoding='utf-8') as f:
            f.write('=' * 60 + '\n')
            f.write("全局最优参数\n")
            f.write("=" * 60 + "\n")
            f.write(f" 股票代码: {best_global['stock_code']}\n")
            f.write(f" EMA 周期: {best_global['ema']}\n")
            f.write(f" 信号类型: {best_global['signal']}\n")
            f.write(f" 信号阈值: {best_global['threshold']}\n")
            f.write(f" 止损比例: {best_global['stop_loss']*100:.0f}%\n")
            f.write(f" 仓位比例: {best_global['position_ratio']*100:.0f}%")
            f.write(f"-" * 60 + "\n")
            f.write(f" 最终资金: ${best_global['final_cash']:,.2f}\n")
            f.write(f" 收益金额: ${best_global['profit']:,.2f}\n")
            f.write(f" 收益率: {best_global['profit_pct']:.2f}%\n")
            f.write(f" 夏普比率: {best_global['sharpe_ratio']:.3f}\n")
            f.write("=" * 60 + "\n")

        print(f' 全局最优参数已保存: {best_file}')



# 类外面添加全局函数.   添加在类下面.
def run_stock_task(args):
    """ 多进程任务函数 (全局函数, 必须在类外面)
    """
    stock_code, params_list = args
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    optimizer = StrategyOptimizer(project_dir)

    return optimizer.run_optimization(stock_code, params_list)

# ============================
# 测试
# ============================
if __name__ == "__main__":
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent

    optimizer = StrategyOptimizer(project_dir)


    # 定义参数网格
    params = optimizer.define_param_grid()

    # ============== 多进程跑 (9个进程) =======================
    all_results = optimizer.run_all_stocks_parallel(params, n_jobs=9)

    # 汇总
    results_df = pd.DataFrame(all_results)

    print(f" \n 每只股票的最优参数: ")
    print(f'-' * 40)

    # 用Groupby 找每只股票最大收益的索引
    idx = results_df.groupby('stock_code')['profit_pct'].idxmax()
    best_per_stock = results_df.loc[idx]

    for _, row in best_per_stock.iterrows():
        print(f"{row['stock_code']}: 收益{row['profit_pct']:.2f}%, 夏普{row['sharpe_ratio']:.3f}")

    # 全局最优
    best_global = results_df.loc[results_df['profit_pct'].idxmax()]
    print(f" \n 全局最优参数:")
    print(f" 股票: {best_global['stock_code']}")
    print(f" EMA: {best_global['ema']}")
    print(f" 信号: {best_global['signal']}")
    print(f" 阈值: {best_global['threshold']}")
    print(f" 止损: {best_global['stop_loss']*100:.0f}%")
    print(f" 仓位: {best_global['position_ratio']*100:.0f}%")
    print(f" 收益率: {best_global['profit_pct']:.2f}%")
    print(f" 夏普: {best_global['sharpe_ratio']:.3f}")

    print(f" 所有股票优化完成! ")


"""

## 策略参数优化模块 (8_strategy_optimizer.py)

**任务目标**：
- 从 2250 种参数组合中找出收益率最高、风险最低的参数组合
- 优化对象：EMA策略（日线）
- 优化方法：网格搜索（Grid Search）
- 评估指标：平均收益率 + 夏普比率 + 胜率
- 使用多进程加速回测

**实现方案**：
1. **参数网格定义**（共 2250 种组合）：
   - **EMA周期组合**：5种：(3,10,30), (5,10,60), (5,15,30), (8,21,55), (10,20,60)
   - **信号类型**：3种：short_long、mid_long、short_mid_long
   - **信号阈值**：5种：0%, 1%, 2%, 3%, 5%
   - **止损比例**：6种：5%, 10%, 15%, 20%, 25%, 30%
   - **仓位比例**：5种：20%, 30%, 40%, 50%, 100%
   - **总组合数**：5 × 3 × 5 × 6 × 5 = 2250 种

2. **信号计算**：
   - 根据EMA周期组合计算三条EMA线（短期、中期、长期）
   - 根据信号类型生成买入/卖出信号：
     - `short_long`：短期EMA > 长期EMA 且差值 > 阈值
     - `mid_long`：中期EMA > 长期EMA 且差值 > 阈值
     - `short_mid_long`：短期EMA > 中期EMA > 长期EMA
   - 使用相对阈值（差值/长期EMA）而非固定阈值

3. **回测执行**：
   - 每个参数组合独立运行回测
   - 初始资金 30000，手续费 0.1%
   - 止损机制：亏损达到设定比例强制平仓
   - 仓位控制：每次使用可用资金的指定比例

4. **多进程加速**：
   - 使用 `multiprocessing.Pool` 并行处理多只股票
   - 默认 9 个进程同时运行
   - 边跑边保存中间结果（防止意外中断丢失数据）

5. **结果保存**：
   - **完整结果**：optimization_results.csv（所有组合的详细结果）
   - **分股票保存**：分股票/{stock_code}.csv（每只股票的所有组合）
   - **每只股票最优参数**：best_per_stock.csv
   - **全局最优参数**：best_params.txt（含股票代码、EMA周期、信号类型、阈值、止损、仓位、收益、夏普）

**核心代码结构**：
```python
class StrategyOptimizer:
    ├── __init__()                    # 初始化目录
    ├── define_param_grid()            # 定义参数网格（2250种组合）
    ├── calculate_ema_signals()        # 计算EMA和信号
    ├── run_signal_backtest()          # 单组参数回测
    ├── run_optimization()             # 循环所有参数组合
    ├── run_all_stocks_parallel()      # 多进程批量优化
    └── save_results()                 # 保存所有优化结果
```

# 多进程任务函数（必须在类外面）
def run_stock_task(args):
    ...
    
"""








