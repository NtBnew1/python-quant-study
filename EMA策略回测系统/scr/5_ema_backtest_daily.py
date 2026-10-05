"""
文件：src/5_ema_backtest_daily.py

功能：
-读取 data/raw/*.csv（价格数据）
-读取 data/signals_ema/*.csv（信号数据）
-运行 Backtrader 回测
-保存结果
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.utils.io import raw_print

# ========== 中文字体设置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Ariel Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from EMA策略回测系统.scr.config import CHARTS_DIR
from config import (
DATA_RAW_DIR,
DATA_SIGNALS_EMA_DIR,
RESULTS_DIR,
INITIAL_CASH,
STOP_LOSS,
POSITION_RATIO,
COMMISSION
)

# ==================================
# 函数1: 加载并合并数据
# ==================================
def load_and_merge_data(stock_code):
    """加载价格数据和信号数据, 合并成一个DataFrame
    Args:
        stock_code: 股票代码
    Returns:
        DATa Frame: 合并后的数据 (OHLCV + 三种信号)
    """
    # 1. 读取价格数据 (从data/raw)
    price_file = DATA_RAW_DIR / f"{stock_code}.csv"
    if not price_file.exists():
        print(f" 价格文件不存在: {price_file}")
        return None

    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date'], utc=True)
    price_df.set_index('Date', inplace=True)


    # 2. 读取信号数据 (从data/signals_ema/)
    signal_file = DATA_SIGNALS_EMA_DIR / f"{stock_code}.csv"
    if not signal_file.exists():
        print(f" 信号文件不存在: {signal_file}")
        return None

    signal_df = pd.read_csv(signal_file)
    signal_df['Date'] = pd.to_datetime(signal_df['Date'], utc=True)
    signal_df.set_index('Date', inplace=True)

    # 3. 提取需要列: 信号列 + EMA列
    # signal_cols = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']
    columns_needed =[
        'signal_ema5_60',
        'signal_ema10_60',
        'signal_ema5_10_60',
        'ema_5',
        'ema_10',
        'ema_60'
    ]

    # 只取存在的列
    existing_cols = [col for col in columns_needed if col in signal_df.columns]

    # 4. 合并: 以价格数据为主, 只保留两者都有的日期
    merged_df = price_df.join(signal_df[existing_cols], how='inner')

    # 5. 过滤时间范围
    merged_df = merged_df.loc[pd.to_datetime('2020-01-01', utc=True):pd.to_datetime('2026-08-04', utc=True)]

    if len(merged_df) == 0:
        print(f' 合并后没有数据')
        return None

    return merged_df



# =================================
# 函数2: 运行单只股票回测
# =================================
class EMASignalStrategy(bt.Strategy):
    """EMA 信号策略
    根据信号值执行买卖
    """

    params = (
        ('stop_loss', 0.20),        # 止损比例 20%
        ('position_ratio', 0.30),    # 每次买入 30% 资金
    )

    def __init__(self, active_signal_name='signal_ema5_60'):
        """策略初始化 - 只在回测开始时执行一次"""
        self.order = None
        self.entry_price = 0
        self.trade_records = []
        self.entry_date = None
        self.last_buy_price = 0
        self.last_buy_size = 0

        # 获取三种信号
        self.signal_ema5_60 = self.datas[0].signal_ema5_60
        self.signal_ema10_60 = self.datas[0].signal_ema10_60
        self.signal_ema5_10_60 = self.datas[0].signal_ema5_10_60

        # 根据参数选择信号
        if active_signal_name == "signal_ema10_60":
            self.active_signal = self.signal_ema10_60
        elif active_signal_name == "signal_ema5_10_60":
            self.active_signal = self.signal_ema5_10_60
        else:
            self.active_signal = self.signal_ema5_60


        self.log(f"策略初始化完成 - 使用{active_signal_name}")
        self.log(f" 止损比例: {self.params.stop_loss*100}%")
        self.log(f" 买入比例: {self.params.position_ratio*100}%")


    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f" {dt.isoformat()} {txt}")

    def notify_order(self, order):
        """订单完成时自动调用"""
        # 只处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 订单完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入| 价格: {order.executed.price:.2f} | 数量: {order.executed.size}")
                self.entry_price = order.executed.price
                self.last_buy_price = order.executed.price
                self.last_buy_size = order.executed.size
                self.entry_date = self.datas[0].datetime.date(0)
            else:
                self.log(f"卖出| 价格: {order.executed.price:.2f} | 数量: {order.executed.size}")
                self.entry_price= 0
        self.order = None

    def notify_trade(self, trade):
        """每笔交易完成时自动调用"""
        if trade.isclosed:
            profit = trade.pnl
            profit_comm = trade.pnlcomm
            final_cash = self.broker.getvalue()

            self.trade_records.append({
                'entry_date': self.entry_date if self.entry_date else '',
                'exit_date': self.datas[0].datetime.date(0),
                'entry_price': self.last_buy_price,
                'exit_price': trade.data.close[0],
                'size': abs(self.last_buy_size),
                'profit': profit,
                'profit_comm': profit_comm,
                'profit_pct': (profit/(self.last_buy_price*abs(self.last_buy_size))) * 100 if self.last_buy_price > 0 else 0,
                'final_cash': final_cash
            })
            self.log(f"交易日期 | 毛利润: ${profit:.2f} | 净利润: ${profit_comm:.2f} | 总资金: ${final_cash:,.2f}")

    def next(self):
        """每个交易日执行"""
        # 如果有挂单, 不执行新操作
        if self.order:
            return

        price = self.datas[0].close[0]
        signal = self.active_signal[0]

        # 买入信号
        if signal == 1 and not self.position:
            size = int(self.broker.getcash() * self.params.position_ratio / price)
            if size > 0:
                self.order = self.buy(size=size)
                self.log(f"买入信号 | 价格: {price:.2f} | 数量: {size}")

        # 卖出信号
        elif signal == -1 and self.position:
            size = self.position.size
            self.order = self.sell(size=self.position.size)
            self.log(f"卖出信号 | 价格: {price:.2f} | 数量: {size}")

        # 止损检查
        elif self.position and self.entry_price > 0:
            loss_pct = (self.entry_price - price) / self.entry_price
            if loss_pct >= self.params.stop_loss:
                self.order = self.sell(size=self.position.size)
                self.log(f" 止损 | 价格: {price:.2f} | 亏损: {loss_pct*100:.2f}%")


# =================================
# 函数3: 运行单只股票回测
# =================================
def run_backtest(stock_code, signal_type='signal_ema5_60'):
    """运行单只股票回测
    Args:
        stock_code: 股票代码

    Returns:
        dict: 回测结果
    """
    print(f" \n 回测股票: {stock_code} - {signal_type}")
    print(f'-' * 50)

    # 1. 加载数据
    df = load_and_merge_data(stock_code)
    if df is None:
        return None

    # 2. 自定义数据源
    class PandasDataWithSignals(bt.feeds.PandasData):
        lines = ('signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60')
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
        )

    # 3. 创建回测
    cerebro = bt.Cerebro()
    data = PandasDataWithSignals(dataname=df)
    cerebro.adddata(data)

    # 4. 传入信号类型到策略
    cerebro.addstrategy(
        EMASignalStrategy,
        active_signal_name = signal_type
    )

    # 5. 设置资金和手续费
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # 6. 添加分析器
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    # 7. 运行回测
    results = cerebro.run()
    strat = results[0]

    # 8. 提取结果
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
        'trade_records': strat.trade_records
    }

    print(f" \n 回测结果: ")
    print(f" 初始资金: ${result['initial_cash']:,.2f}")
    print(f" 最终资金: ${result['final_cash']:,.2f}")
    print(f" 收益: ${result['profit']:,.2f}")
    print(f" 收益率: {result['profit_pct']:.2f}%")
    print(f" 夏普比率: {result['sharpe_ratio']:.3f}")

    print(f" 收益; ${result['profit']:,.2f} ({result['profit_pct']:.2f}%)")
    print(f" 夏普: {result['sharpe_ratio']:.3f}")

    return result



# ================================
# 函数4: 循环所有股票, 三种信号分别回测
# ================================
def run_all_backtests():
    """循环所有股票, 对每只股票分别回测三种信号
    保存结果到csv
    """
    print(f'\n' + '=' * 60)
    print(f" 运行所有股票回测 (日线)")
    print(f'=' * 60)

    # 1. 获取所有股票
    raw_files = list(DATA_RAW_DIR.glob('*.csv'))

    if len(raw_files) == 0:
        print(f" 没有找到数据文件")
        return None

    stock_codes = [f.stem for f in raw_files]
    print(f" 找到{len(stock_codes)}只股票: {stock_codes}")

    # 2. 三种信号
    signal_types = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']

    # 3. 存储所有结果
    all_results = []

    # 4. 循环每只股票
    for stock_code in stock_codes:
        print(f"\n{'='*60}")
        print(f"股票: {stock_code}")
        print(f"=" * 60)

        # 对每种信号分别回测
        for signal_type in signal_types:
            print(f'\n 信号: {signal_type}')
            print(f'-' * 50)

            result = run_backtest(stock_code, signal_type)

            if result:
                all_results.append(result)

    # 5. 转换为 DataFrame
    results_df = pd.DataFrame(all_results)

    # 6. 保存结果
    if len(results_df) > 0:
        summary_dir = RESULTS_DIR / "汇总"
        summary_dir.mkdir(parents=True, exist_ok=True)
        output_file = summary_dir / 'backtest_results_ema_日线.csv'
        results_df.to_csv(output_file, index=False)
        print(f" \n 结果已保存到: {output_file}")

        # ===========保存每只股票的明细 ==============
        detailed_dir = RESULTS_DIR / '明细_日线'
        detailed_dir.mkdir(parents=True, exist_ok=True)

        # 去掉 trade_records 列再保存汇总
        summary_df = results_df.drop(columns=['trade_records'], errors='ignore')
        summary_df.to_csv(output_file, index=False)

        for _, row in results_df.iterrows():
            stock_code = row['stock_code']
            signal_type = row['signal_type']
            trade_records = row.get('trade_records', [])

            if trade_records and len(trade_records) > 0:
                stock_dir = detailed_dir / stock_code
                stock_dir.mkdir(parents=True, exist_ok=True)
                trade_df = pd.DataFrame(trade_records)
                trade_file = stock_dir / f"{signal_type}.csv"
                trade_df.to_csv(trade_file, index=False)
                print(f" 明细已保存: {trade_file}")

        # 7. 打印汇总
        print(f"\n" +'=' * 80)
        print(f" 回测汇总")
        print(f'=' * 60)
        print(f" 总回测次数: {len(results_df)}")
        print(f" 平均收益率: {results_df['profit_pct'].mean():.2f}%")
        print(f" 平均夏普比率: {results_df['sharpe_ratio'].mean():.3f}")
        print(f" 盈利次数: {(results_df['profit'] > 0).sum()} / {len(results_df)}")

        # 按信号类型汇总
        print(f'\n 按信号类型汇总: ')
        for signal_type in signal_types:
            subset = results_df[results_df['signal_type'] == signal_type]
            if len(subset) > 0:
                print(f" \n {signal_type}: ")
                print(f" 平均收益率: {subset['profit_pct'].mean():.2f}%")
                print(f" 平均夏普: {subset['sharpe_ratio'].mean():.3f}")
                print(f" 盈利/总数: {(subset['profit'] > 0).sum()} / {len(subset)}")

        # 最佳表现
        if len(results_df) > 0:
            best_idx = results_df['profit_pct'].idxmax()
            best = results_df.loc[best_idx]
            print(f" \n 最佳表现: {best['stock_code']} - {best['signal_type']}")
            print(f" 收益率: {best['profit_pct']:.2f}%")
            print(f" 夏普比率: {best['sharpe_ratio']:.3f}")

    return results_df


# ====================================
# 函数5: 绘制EMA均线图 + 买卖点
# ====================================
def plot_ema_with_signals(stock_code, signal_type='signal_ema5_60', start_date='2020-01-01', end_date='2026-08-04'):
    """绘制EMA均线图 和 买卖点
    Args:
        stock_code: 股票代码
        signal_type: 信号类型
        start_date: 开始日期
        end_date: 结束日期
    """
    print(f" \n 绘制: {stock_code} - {signal_type}")

    # 直接调用load_and_merge_data
    merged_df = load_and_merge_data(stock_code)
    if merged_df is None:
        return

    # 过滤时间范围
    merged_df = merged_df.loc[start_date:end_date]

    # 根据信号类型提取买卖点
    signal_col = signal_type

    # 信号名称
    signal_names = {
        'signal_ema5_60': 'EMA5 VS EMA60',
        'signal_ema10_60': 'EMA10 VS EMA60',
        'signal_ema5_10_60': 'EMA5 > EMA10 > EMA60'
    }
    buy_dates = merged_df[merged_df[signal_col] == 1].index
    sell_dates = merged_df[merged_df[signal_col] == -1].index

    buy_prices = merged_df.loc[buy_dates, 'Close']
    sell_prices = merged_df.loc[sell_dates, 'Close']

    # 创建图表
    fig, ax = plt.subplots(figsize=(16,8))

    # 绘制价格和均线
    ax.plot(merged_df.index, merged_df['Close'], label='Close', color='black', linewidth=1)
    ax.plot(merged_df.index, merged_df['ema_5'], label='EMA5', color='blue', linewidth=1.5)
    ax.plot(merged_df.index, merged_df['ema_10'], label='EMA10', color='orange', linewidth=1.5)
    ax.plot(merged_df.index, merged_df['ema_60'], label='EMA60', color='red', linewidth=2)

    # 标记买卖点
    ax.scatter(buy_dates, buy_prices, color='green', marker='^', s=80, label='买入', zorder=5)
    ax.scatter(sell_dates, sell_prices, color='red', marker='v', s=80, label='卖出', zorder=5)
    print(f" 买入: {len(buy_dates)}次, 卖出: {len(sell_dates)}次")

    # 设置标题
    signal_display = signal_names.get(signal_type, signal_type)
    ax.set_title(f"{stock_code} - {signal_display} 均线 + 买卖点", fontsize=14)
    ax.set_xlabel('日期')
    ax.set_ylabel('价格 ($)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 创建股票专属文件夹
    stock_charts_dir = CHARTS_DIR / '日线' / stock_code
    stock_charts_dir.mkdir(parents=True, exist_ok=True)

    # 保存图表
    output_file = stock_charts_dir / f"{stock_code}_{signal_type}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    # plt.show()  # 显示图表
    plt.close()

    print(f" 已保存: {output_file}")
    return output_file


# =====================================
# 函数6: 绘制所有股票的EMA均线图 + 买卖点
# =====================================
def plot_all_stocks():
    """ 绘制所有股票的EMA均线图 和 买卖点 (三种信号分别绘制)
    每只股票创建一个文件夹, 里面放3张图
    """
    print(f" \n" + '=' * 80)
    print(f" 绘制所有股票 EMA 均线图 + 买卖点 (三种信号)")
    print(f" =" * 80)

    # 1. 获取所有股票
    raw_files = list(DATA_RAW_DIR.glob("*.csv"))
    if len(raw_files) == 0:
        print(f" 没有找到数据文件")
        return

    stock_codes = [f.stem for f in raw_files]
    signal_types = ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']

    print(f" 找到{len(stock_codes)} 只股票, {len(signal_types)} 种信号")
    print(f" 共生成 {len(stock_codes) * len(signal_types)} 张图")
    print(f'-' * 40)

    # 2. 循环每只股票
    for stock_code in stock_codes:
        print(f" \n {stock_code}: ")

        # 对每种信号分别绘制图
        for signal_type in signal_types:
            plot_ema_with_signals(stock_code, signal_type)

    # 3. 完成提示
    print(f" \n" + '=' * 60)
    print(f" 所有图表绘制完成")
    print(f" 图表保存在: {CHARTS_DIR}")
    print(f" 共生成 {len(stock_codes) * len(signal_types)} 张图")
    print(f" =" * 60)



# =================================
# 测试第1个函数
# =================================
if __name__ == "__main__":
    results_df = run_all_backtests()
    if results_df is not None:
        print(f" \n 所有回测完成! ")
        # 绘制所有股票的EMA均线图 + 买卖点
        plot_all_stocks()
        print(f" 全部完成! ")


"""
## EMA策略回测模块（日线） (5_ema_backtest_daily.py)

**任务目标**：
- 从 data/raw/ 读取价格数据（OHLCV）
- 从 data/signals_ema/ 读取EMA信号数据
- 使用 Backtrader 进行回测
- 保存回测结果到 results/
- 绘制EMA均线图 + 买卖点

**实现方案**：
1. **数据加载与合并**：
   - 从 data/raw/ 读取价格数据（OHLCV）
   - 从 data/signals_ema/ 读取信号数据
   - 合并价格数据和信号数据（内连接）
   - 过滤时间范围（2020-01-01 至 2026-08-04）

2. **策略类设计**：
   - 根据信号值执行买卖（1买入、-1卖出）
   - 止损机制：亏损 20% 强制平仓
   - 每次使用可用资金的 30% 买入
   - 支持选择三种信号之一（默认 signal_ema5_60）

3. **回测运行**：
   - 循环所有股票和三种信号
   - 初始资金 30000
   - 手续费 0.1%
   - 添加 Returns 和 SharpeRatio 分析器

4. **结果汇总**：
   - 保存为 CSV 文件到 results/
   - 按信号类型分组统计平均收益和夏普比率
   - 识别最佳表现股票

5. **图表绘制**：
   - 绘制价格曲线 + EMA5、EMA10、EMA60
   - 标记买入点（绿色上三角）和卖出点（红色下三角）
   - 每只股票创建一个文件夹，存放三种信号的图表
   - 保存为 PNG 格式

**核心代码结构**：
```python
# 数据加载与合并
load_and_merge_data(stock_code)

# 策略类
class EMASignalStrategy(bt.Strategy):
    ├── __init__()              # 初始化信号和参数
    ├── log()                   # 日志打印
    ├── notify_order()          # 订单状态回调
    └── next()                  # 每个交易日主逻辑

# 单只股票回测
run_backtest(stock_code, signal_type)

# 批量回测
run_all_backtests()

# 绘制单只股票
plot_ema_with_signals(stock_code, signal_type)

# 绘制所有股票
plot_all_stocks()
"""
