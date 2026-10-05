"""
文件：src/6_ema_backtest_weekly.py

功能：
-读取 data/raw/*.csv（价格数据）
-读取 data/signals_ema_周线/*.csv（信号数据）
-运行 Backtrader 回测
-保存结果
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import datetime

from IPython.core.pylabtools import activate_matplotlib

from EMA策略回测系统.scr.config import DATA_SIGNALS_EMA_WEEKLY_DIR

# 设置中文字体 -> 图表用的
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from config import (
DATA_RAW_DIR,
RESULTS_DIR,
INITIAL_CASH,
STOP_LOSS,
POSITION_RATIO,
COMMISSION,
DATA_SIGNALS_EMA_WEEKLY_DIR
)

# 周线信号目录
DATA_SIGNALS_EMA_WEEKLY_DIR = Path(__file__).parent.parent / 'data' / "signals_ema_周线"

# =====================================
# 类1: 数据加载器 (周线版本)
# =====================================
class DataLoader:
    """数据加载器 (周线版本)"""
    @staticmethod
    def load_weekly_data(stock_code):
        """
        加载价格数据和周线信号数据, 合并成一个DataFrame
        Args:
            stock_code: 股票代码
        Returns:
            DataFrame: 合并后的数据 (OHLCV + 三种信号 + EMA 列)
        """
        signal_file = DATA_SIGNALS_EMA_WEEKLY_DIR / f"{stock_code}.csv"
        if not signal_file.exists():
            print(f" 周线文件不存在: {signal_file}")
            return None
        df = pd.read_csv(signal_file)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        print(f" 周线数据: {len(df)} 行, {df.index[0]} 至 {df.index[-1]}")
        return df


# ==========================================
# 类2: 策略类 (周线版本)
# ==========================================
class EMASignalStrategy(bt.Strategy):
    """EMA 信号策略 (周线版本) """
    params = (
        ('stop_loss', 0.20),
        ('position_ratio', 0.30),
        ('active_signal_name', 'signal_ema5_60'),
    )

    def __init__(self, active_signal_name='signal_ema5_60'):
        self.order = None
        self.entry_price = 0
        self.active_signal_name = active_signal_name

        # ==================新添加 ===================
        self.last_buy_price = 0
        self.last_buy_size = 0
        self.trade_count = 0    # 交易计算器
        self.cash_before_buy = 0
        self.trade_records = [] # 存储每笔交易的明细
        self.entry_date = None

        # 获取三种信号
        self.signal_ema5_60 = self.datas[0].signal_ema5_60
        self.signal_ema10_60 = self.datas[0].signal_ema10_60
        self.signal_ema5_10_60 = self.datas[0].signal_ema5_10_60

        # ===========新添加: 获取 EMA 列 (用于显示原因)============
        self.ema_5 = self.datas[0].ema_5
        self.ema_10 = self.datas[0].ema_10
        self.ema_60 = self.datas[0].ema_60

        # 根据参数选择信号
        if active_signal_name == 'signal_ema10_60':
            self.active_signal = self.signal_ema10_60
        elif active_signal_name == 'signal_ema5_10_60':
            self.active_signal = self.signal_ema5_10_60
        else:
            self.active_signal = self.signal_ema5_60

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                # 计算买入金额
                buy_amount = order.executed.price * order.executed.size

                # 每笔新交易开始时打印分隔线
                self.trade_count += 1
                print(f" \n{'='*50}")
                print(f" 第{self.trade_count} 笔交易")
                print(f"{'='*50}")

                # 获取EMA 值用于显示原因
                ema5 = self.ema_5[0]
                ema10 = self.ema_10[0]
                ema60 = self.ema_60[0]

                # 显示买入原因
                if self.active_signal_name == 'signal_ema5_60':
                    reason = f"EMA5({ema5:.2f}) > EMA60({ema60:.2f})"
                elif self.active_signal_name == 'signal_ema10_60':
                    reason = f"EMA10({ema10:.2f}) > EMA60({ema60:.2f})"
                else:  # signal_ema5_10_60
                    reason = f"EMA5({ema5:.2f}) > EMA10({ema10:.2f}) > EMA60({ema60:.2f})"

                # 计算 初始金额百分比
                cash_before = self.cash_before_buy
                pct_of_cash = (buy_amount / cash_before) * 100 if cash_before > 0 else 0


                self.log(f"买入信号 | 价格:{order.executed.price:.2f} | 原因: {reason}")
                self.log(f"买入 | 价格: {order.executed.price:.2f} | 数量: {order.executed.size} 股"
                         f"| 金额: ${buy_amount:,.2f} ({pct_of_cash:.1f}% of ${cash_before:,.2f})")

                # =====记录进场日期==========
                self.entry_date = self.datas[0].datetime.date(0)

                self.entry_price = order.executed.price
                self.last_buy_price = order.executed.price  # 记录买入价
                self.last_buy_size = order.executed.size # 记录买入数量
            else:
                # 计算收益率
                buy_price = self.last_buy_price
                sell_price = order.executed.price
                profit_pct = (sell_price - buy_price) / buy_price * 100 if buy_price > 0 else 0
                sell_amount = sell_price * abs(order.executed.size)

                # 显示卖出原因
                ema5 = self.ema_5[0]
                ema10 = self.ema_10[0]
                ema60 = self.ema_60[0]
                if self.active_signal_name == 'signal_ema5_60':
                    reason = f"EMA5({ema5:.2f}) < EMA60({ema60:.2f})"
                elif self.active_signal_name == 'signal_ema10_60':
                    reason = f"EMA10({ema10:.2f}) < EMA60({ema60:.2f})"
                else:  # signal_ema5_10_60
                    reason = f"EMA5({ema5:.2f}) < EMA10({ema10:.2f}) < EMA60({ema60:.2f})"


                # 检查是否是止损单 (卖出价 < 买入价 * (1 - stopp_loss))
                is_stop_loss = profit_pct <= -self.params.stop_loss * 100

                if is_stop_loss:
                    self.log(f"卖出信号 | 价格: {order.executed.price:.2f} | 原因: {reason}")
                    self.log(f"卖出 (止损) | 价格: {sell_price:.2f} | 数量: {order.executed.size} 股"
                         f"| 亏损: {profit_pct:.2f}% | 金额: ${sell_amount:,.2f}")
                else:
                    self.log(f"卖出信号 | 原因: {reason}")
                    self.log(f"卖出 | 价格: ${sell_price:.2f} | 数量:{order.executed.size}股 "
                             f"| 收益率:{profit_pct:.2f}% | 金额: ${sell_amount:,.2f}")
                self.entry_price = 0
        self.order = None

    # ===============新添加==============
    def notify_trade(self, trade):
        """每笔交易完成时自动调用"""
        if trade.isclosed:
            profit = trade.pnl
            profit_comm = trade.pnlcomm
            dt = self.datas[0].datetime.date(0)

            # 计算当前总金额
            final_cash = self.broker.getvalue()

            # 记录交易明细
            self.trade_records.append({
                'entry_date': self.entry_date if hasattr(self, 'entry_date') else '',
                'exit_date': self.datas[0].datetime.date(0),
                'entry_price': self.last_buy_price,
                'exit_price': trade.data.close[0],
                'size': abs(self.last_buy_size),
                'profit': profit,
                'profit_comm': profit_comm,
                'profit_pct': (profit/(self.last_buy_price * abs(self.last_buy_size))) * 100 if self.last_buy_price > 0 else 0,
                'final_cash': final_cash
            })

            self.log(f"交易结束 | 毛利润; ${profit:.2f} | 净利润: ${profit_comm:.2f} | 总资金: ${final_cash:,.2f}")

    def next(self):
        if self.order:
            return
        price = self.datas[0].close[0]
        signal = self.active_signal[0]



        # 买入信号
        if signal == 1 and not self.position:
            cash_before_buy = self.broker.getcash() #买入前的现金
            self.cash_before_buy = cash_before_buy # 保存到 self里
            size = int(cash_before_buy * self.params.position_ratio / price)
            if size > 0:
                self.order = self.buy(size=size)


        # 卖出信号
        elif signal == -1 and self.position:
            size = self.position.size
            self.order = self.sell(size=size)


        # 止损
        elif self.position and self.entry_price > 0:
            loss_pct = (self.entry_price - price) / self.entry_price
            if loss_pct >= self.params.stop_loss:
                size = self.position.size
                sell_amount = price * size
                self.order = self.sell(size=size)
                self.log(f" 🛑 触发止损 | 买入价: ${self.entry_price:.2f} | 当前价: ${price:.2f}"
                         f"| 亏损:{loss_pct*100:.2f} | 卖出: {size}股 | 金额: ${sell_amount:,.2f}")


# =========================================
# 函数: 运行单只股票回测 (周线版本)
# =========================================
def run_backtest(stock_code, signal_type='signal_ema5_60'):
    """运行单只股票的周线回测
    Args:
        stock_code: 股票代码
        signal_type: 信号类型
    Returns:
        dict: 回测结果
    """
    print(f" \n 回测: {stock_code} - {signal_type}")
    print(f'=' * 60)

    # 1. 加载数据
    df = DataLoader.load_weekly_data(stock_code)
    if df is None:
        return None

    # 2. 自定义数据源
    class PandasDataWithSignals(bt.feeds.PandasData):
        lines = ('signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60', 'ema_5', 'ema_10', 'ema_60')
        params = (
            ('datetime', None),
            ('open', 'Open'),
            ('high', 'High'),
            ('low', 'Low'),
            ('close', 'Close'),
            ('volume', 'Volume'),
            ('openinterest', -1),
            ('signal_ema5_60', 'signal_ema5_60'),
            ('signal_ema10_60', 'signal_ema10_60'),
            ('signal_ema5_10_60', 'signal_ema5_10_60'),
            ('ema_5', 'ema_5'),
            ('ema_10', 'ema_10'),
            ('ema_60', 'ema_60')
        )

    # 3. 创建回测
    cerebro = bt.Cerebro()
    data = PandasDataWithSignals(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(EMASignalStrategy, active_signal_name=signal_type)

    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    # 4 运行回测
    results = cerebro.run()
    strat = results[0]

    # 5. 提取结果
    returns = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    final_cash = cerebro.broker.getvalue()
    total_return = returns.get('rtot', 0) * 100
    sharpe_ratio = sharpe.get('sharperatio', 0)

    result = {
        'stock_code': stock_code,
        'signal_type': signal_type,
        'initial_cash': INITIAL_CASH,
        'final_cash': final_cash,
        'profit': final_cash - INITIAL_CASH,
        'profit_pct': (final_cash - INITIAL_CASH) / INITIAL_CASH * 100,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio if sharpe_ratio else 0,
        'trade_records': strat.trade_records,
    }

    print(f" 收益: ${result['profit']:,.2f} ({result['profit_pct']:.2f}%)")
    print(f" 夏普: {result['sharpe_ratio']:.3f}")
    return result


# ======================================
# 函数: 循环所有股票的回测 (周线版本)
# ======================================
def run_all_backtests():
    """ 循环所有股票, 三种信号分别回测
    只打印结果, 不保存 CSV
    """
    # 1. 获取所有股票
    raw_files = list(DATA_RAW_DIR.glob('*.csv'))
    if len(raw_files) == 0:
        print(f" 没有找到数据文件")
        return None

    stock_codes = [f.stem for f in raw_files]
    signal_types = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']

    print(f" 找到{len(stock_codes)} 只股票")
    print(f'-' * 40)

    all_results = []

    # 2. 循环每只股票
    for stock_code in stock_codes:
        print(f" \n 股票: {stock_code}")

        # 3. 循环每种信号
        for signal_type in signal_types:
            result = run_backtest(stock_code, signal_type)
            if result:
                all_results.append(result)

    # 4. 转换为 DataFrame
    results_df = pd.DataFrame(all_results)

    # 5. 打印汇总
    if len(results_df) > 0:
        print(f"\n" + "=" * 60)
        print(f" 周线回测汇总")
        print(f" =" * 60)
        print(f" 总回测次数: {len(results_df)}")
        print(f" 平均收益率: {results_df['profit_pct'].mean():.2f}%")
        print(f" 平均夏普比率: {results_df['sharpe_ratio'].mean():.3f}")
        print(f" 盈利次数: {(results_df['profit'] > 0).sum()} / {len(results_df)}")

        for signal_type in signal_types:
            subset = results_df[results_df['signal_type'] == signal_type]
            if len(subset) > 0:
                print(f" \n {signal_type}: ")
                print(f" 平均收益率: {subset['profit_pct'].mean():.2f}%")
                print(f" 平均夏普: {subset['sharpe_ratio'].mean():.3f}")
                print(f" 盈利/总数: {(subset['profit'] > 0).sum()} / {len(subset)}")

        if len(results_df) > 0:
            best_idx = results_df['profit_pct'].idxmax()
            best = results_df.loc[best_idx]
            print(f" \n  最佳表现: {best['stock_code']} - {best['signal_type']}")
            print(f" 收益率: {best['profit_pct']:.2f}%")
    return results_df


# =======================================
# 类: 图表绘制器 (周线版本)
# =======================================
class WeeklyChartPlotter:
    """周线图表绘制类"""
    def __init__(self):
        # 周线图表目录: charts/周线
        self.charts_dir = Path(__file__).parent.parent / "charts" / '周线'
        self.charts_dir.mkdir(parents=True, exist_ok=True)
        print(f" 周线图表目录: {self.charts_dir}")

    def plot_signal(self, stock_code, signal_type='signal_ema5_60',
                    start_date='2020-01-01', end_date='2026-08-04'):
        """ 绘制单只股票的周线EMA均线图和买卖点"""
        print(f" 绘制: {stock_code} - {signal_type}")

        # 加载周线数据
        df = DataLoader.load_weekly_data(stock_code)
        if df is None:
            return
        df = df.loc[start_date:end_date]
        signal_col = signal_type

        signal_names = {
            'signal_ema5_60': 'EMA5 vs EMA60',
            'signal_ema10_60': 'EMA10 vs EMA60',
            'signal_ema5_10_60': 'EMA5 > EMA10 > EMA60'
        }

        buy_dates = df[df[signal_col] == 1].index
        sell_dates = df[df[signal_col] == -1].index
        buy_prices = df.loc[buy_dates, 'Close']
        sell_prices = df.loc[sell_dates, 'Close']

        fig, ax = plt.subplots(figsize=(16,8))

        ax.plot(df.index, df['Close'], label='Close', color='black', linewidth=1)
        ax.plot(df.index, df['ema_5'], label='EMA5', color='blue', linewidth=1.5)
        ax.plot(df.index, df['ema_10'], label='EMA10', color='orange', linewidth=1.5)
        ax.plot(df.index, df['ema_60'], label='EMA60', color='red', linewidth=2)

        ax.scatter(buy_dates, buy_prices, color='green', marker='^', s=80, label='买入', zorder=5)
        ax.scatter(sell_dates, sell_prices, color='red', marker='v', s=80, label='卖出', zorder=5)

        print(f" 买入: {len(buy_dates)} 次, 卖出: {len(sell_dates)} 次")

        signal_display = signal_names.get(signal_type, signal_type)
        ax.set_title(f'{stock_code} - 周线{signal_display}', fontsize=14)
        ax.set_xlabel('日期')
        ax.set_ylabel('价格 ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)


        # ======保存图表 ==============
        stock_charts_dir = self.charts_dir / stock_code
        stock_charts_dir.mkdir(parents=True, exist_ok=True)

        output_file = stock_charts_dir / f'{stock_code}_{signal_type}.png'
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f" 已保存: {output_file}")

        return fig

    def plot_all(self):
        """ 绘制所有股票的周线EMA均线图和买卖点 (三种信号) """
        print(f"\n" +'=' * 50)
        print(f" 绘制所有股票 周线 EMA 均线图 + 买卖点 (三种信号) ")
        print(f'=' * 60)

        # 1. 获取所有股票
        raw_files = list(DATA_RAW_DIR.glob('*.csv'))
        if len(raw_files) == 0:
            print(f" 没有找到数据文件")
            return

        stock_codes = [f.stem for f in raw_files]
        signal_types = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']

        print(f" 找到 {len(stock_codes)} 只股票, {len(signal_types)} 种信号")
        print(f'-' * 40)

        # 2. 信号每只股票
        for stock_code in stock_codes:
            print(f" \n {stock_code}: ")

            # 3. 循环每种信号
            for signal_type in signal_types:
                self.plot_signal(stock_code, signal_type)

        print(f' \n' + '=' * 60)
        print(f" 所有周线图表绘制完成! ")
        print(f'=' * 60)


# =======================================
# 函数: 保存回测结果到CSV
# =======================================
def save_all_results(results_df, charts_dir_name='周线'):
    """统一保存所有回测结果和图表
    """
    saved_files = {}
    # 1. 保存汇总 CSV
    if results_df is not None and len(results_df) > 0:
        # 汇总目录
        summary_dir = RESULTS_DIR / "汇总"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / 'backtest_results_ema_周线.csv'

        # 从DataFrame 中去掉 trade_records 列 (因为不能保存列表)
        summary_df = results_df.drop(columns=['trade_records'], errors='ignore')
        summary_df.to_csv(summary_file, index=False)

        saved_files['csv'] = summary_file
        print(f" 汇总已保存: {summary_file}")
        print(f" 共保存: {len(summary_df)} 条记录")

        # =========== 保存每只股票的明细 ===============
        detailed_dir = RESULTS_DIR / '明细_周线'
        detailed_dir.mkdir(parents=True, exist_ok=True)

        for _, row in results_df.iterrows():
            stock_code = row['stock_code']
            signal_type = row['signal_type']
            trade_records = row.get('trade_records', [])

            if trade_records and len(trade_records) > 0:
                # 创建股票文件夹
                stock_dir = detailed_dir / stock_code
                stock_dir.mkdir(parents=True, exist_ok=True)

                # 保存明细
                trade_df = pd.DataFrame(trade_records)
                trade_file = stock_dir / f'{signal_type}.csv'
                trade_df.to_csv(trade_file, index=False)
                print(f" 明细已保存: {trade_file}")
    else:
        print(f" 没有数据可保存")
        return None

    # 2. 保存图表
    plotter = WeeklyChartPlotter()
    plotter.plot_all()
    saved_files['charts'] = plotter.charts_dir

    return saved_files



# ===============================
# 测试第1个函数
# ===============================
if __name__ == "__main__":
    # 1. 运行回测
    results_df = run_all_backtests()

    # 2. 统一保存所有结果
    if results_df is not None:
        save_all_results(results_df)

    print(f' \n 全部完成!  ')


"""
## EMA策略回测模块（周线） (6_ema_backtest_weekly.py)

**任务目标**：
- 从 data/raw/ 读取价格数据（OHLCV）
- 从 data/signals_ema_周线/ 读取周线信号数据
- 使用 Backtrader 进行周线级别回测
- 保存回测结果到 results/
- 绘制周线EMA均线图 + 买卖点

**实现方案**：
1. **数据加载**：
   - 从 data/signals_ema_周线/ 读取信号数据（已包含OHLCV + EMA列 + 信号列）
   - 无需单独合并价格数据

2. **策略类设计**：
   - 根据信号值执行买卖（1买入、-1卖出）
   - 止损机制：亏损 20% 强制平仓
   - 每次使用可用资金的 30% 买入
   - 支持选择三种信号之一
   - **新增功能**：每笔交易记录详细明细（买入原因、卖出原因、收益率）

3. **交易明细记录**：
   - 买入时显示：价格、数量、金额、占资金比例、买入原因（EMA交叉情况）
   - 卖出时显示：价格、数量、收益率、卖出原因
   - 每笔交易结束时打印毛利润、净利润、总资金
   - 存储交易记录到列表

4. **回测运行**：
   - 循环所有股票和三种信号
   - 初始资金 30000
   - 手续费 0.1%
   - 添加 Returns 和 SharpeRatio 分析器

5. **结果汇总**：
   - 保存为 CSV 文件到 results/
   - 按信号类型分组统计平均收益和夏普比率
   - 识别最佳表现股票

6. **图表绘制**：
   - 绘制周线价格曲线 + EMA5、EMA10、EMA60
   - 标记买入点（绿色上三角）和卖出点（红色下三角）
   - 每只股票创建一个文件夹，存放三种信号的图表
   - 保存到 charts/周线/ 目录

**核心代码结构**：
```python
# 数据加载器
class DataLoader:
    └── load_weekly_data(stock_code)

# 策略类
class EMASignalStrategy(bt.Strategy):
    ├── __init__()              # 初始化信号和参数
    ├── log()                   # 日志打印
    ├── notify_order()          # 订单状态回调（含买入/卖出原因）
    ├── notify_trade()          # 交易完成回调（记录交易明细）
    └── next()                  # 每个交易日主逻辑

# 单只股票回测
run_backtest(stock_code, signal_type)

# 批量回测
run_all_backtests()

# 图表绘制器
class WeeklyChartPlotter:
    ├── plot_signal()           # 绘制单只股票
    └── plot_all()              # 绘制所有股票

# 保存结果
save_all_results(results_df)
"""




