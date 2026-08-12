"""
第4天：Backtrader回测
1. 文件名：4. backtrader回测.py
2. 输入：data/signals/*.csv
3. 输出：results/backtest_results.csv

4. 步骤：
1. 创建 results/ 目录
2. 创建策略类
3. 设置初始资金 10000
4. 设置手续费 0.1%
5. 回测每只股票（20天和60天）
6. 汇总结果

5. 测试：
--检查收益是否合理
"""

import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

from keyring.backends.libsecret import available

warnings.filterwarnings('ignore')

#=======================================================
# 第一部分: 数据加载和合并
#=======================================================

def load_and_merge_data(stock_code, project_dir):
    """加载并合并价格数据和信号数据
    Args:
        stock_code: 股票代码 (文件名)
        project_dir: 项目根目录
    Returns:
        DataFrame: 合并后的数据 (包含OHLCV + 三种信号)

        止损的时候需要OHLCV 数据.  就是在raw文件夹里(原始文件).
    """

    # 1. 从raw 读取价格数据 (OHLCV)
    price_file = project_dir / "data" / "raw" / f"{stock_code}.csv"
    if not price_file.exists():
        print(f" 价格文件不存在: {price_file}")
        return None

    price_df = pd.read_csv(price_file)
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df.set_index('Date', inplace=True)

    # =======新增:转换价格列为数字类型 ==================
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in price_cols:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

    # 2. 从 signals 读取信号数据
    signal_file = project_dir / "data" / "signals" / f"{stock_code}.csv"
    if not signal_file.exists():
        print(f" 信号文件不存在: {signal_file}")
        return None

    signal_df = pd.read_csv(signal_file)
    signal_df['Date'] = pd.to_datetime(signal_df['Date'])
    signal_df.set_index('Date', inplace=True)

    # 3. 检查信号列是否存在
    required_signals = ['signal_20_60', 'signal_10_60', 'signal_10_20_60']
    missing = [s for s in required_signals if s not in signal_df.columns]
    if missing:
        print(f" 信号文件缺少列: {missing}")
        return None

    # 4. 合并: 以价格数据为主, 内连接 (只保留两者都有的日期)
    merged_df = price_df.join(
        signal_df[required_signals],
        how='inner'
    )

    # 5. 过滤时间范围; 2020-01-01 到 2026_08_04
    merged_df = merged_df.loc['2020-01-01':'2026-08-04']

    if len(merged_df) == 0:
        print(f" 合并后没有数据")
        return None

    return merged_df


# =================================================
# 第二部分: 自定义数据源类
# =================================================

class PandasDataWithSignals(bt.feeds.PandasData):
    """
    自定义数据源 - 包含价格数据和三种信号
    这个类告诉Backtrader 如何读取我们的数据:
    -OHLCV 价格数据
    -三种动量信号 (signal_20_60, signal_10_60, signal_10_20_60)
    """
    # 定义三条信号线 (backtrader 的line 概念)
    lines = ('signal_20_60', 'signal_10_60', 'signal_10_20_60')

    # 参数映射: DataFrame列名 -> Backtrader线名
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', -1),
        ('signal_20_60', 'signal_20_60'),
        ('signal_10_60', 'signal_10_60'),
        ('signal_10_20_60', 'signal_10_20_60')
    )

# ==================================================
# 第三部分: 正式策略类
# ==================================================
class MultiSignalStrategy(bt.Strategy):
    """
    多信号策略 - 同时回测三种动量信号
    功能:
        1. 同时监控三种信号 (signal_20_60, signal_10_60, signal_10_20_60)
        2. 每个信号独立使用 10000 资金
        3. 使用 ATR 控制仓位 (波动大少买, 波动小多买)
        4. 亏损 20% 强制止损
    """

    params = (
        ('stop_loss', 0.20),        # 止损比例 20%
        ('atr_period', 14),         # ATR 计算周期
        ('position_ratio', 0.30),   # 每次使用可用资金的 30%
        ('commission', 0.001),      # 手续费 0.1%
        ('active_signal', 'signal_10_60'),  # 当前激活的信号
    )

    def __init__(self):
        """策略初始化 - 设置记录变量"""

        # 记录每个信号的订单 (防止重复下单)
        self.orders = {
            'signal_20_60': None,
            'signal_10_60': None,
            'signal_10_20_60': None
        }

        # ======修改这里: positions -> signal_posistions =======
        self.signal_positions = {
            'signal_20_60': 0,
            'signal_10_60': 0,
            'signal_10_20_60': 0
        }

        # 记录每个信号的买入价格 (用于止损计算)
        self.entry_prices = {
            'signal_20_60': 0,
            'signal_10_60': 0,
            'signal_10_20_60': 0
        }

        # 获取三种信号数据 (从数据源中读取)
        self.signal_20_60 = self.datas[0].signal_20_60
        self.signal_10_60 = self.datas[0].signal_10_60
        self.signal_10_20_60 = self.datas[0].signal_10_20_60


        # 打印策略参数
        print(f" \n {'='*60}")
        print(f" 策略初始化完成")
        print(f" {'='*60}")
        print(f" 止损比例: {self.params.stop_loss*100}%")
        print(f" ATR周期: {self.params.atr_period}")
        print(f" 每笔风险: {self.params.position_ratio*100}%")
        print(f" 手续费: {self.params.commission*100}%")
        print(f" {'='*60}")

    def log(self, txt, dt=None):
        """打印带时间的日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f" {dt.isoformat()}{txt}")

    def notify_order(self, order):
        """订单状态变化时的回调 - Backtrader自动调用"""
        # 只处理已提交或已接受的订单
        if order.status in [order.Submitted, order.Accepted]:
            return

        # 获取订单对应的信号类型
        signal_type = getattr(order, 'signal_type', 'unknown')

        # 订单已完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"[{signal_type}] 买入 | 价格: {order.executed.price:.2f} | 数量: {order.executed.size}")
                self.signal_positions[signal_type] += order.executed.size
            else:
                self.log(f"[{signal_type}] 卖出 | 价格: {order.executed.price:.2f} | 数量: {order.executed.size}")

                # 卖出后直接清零
                self.signal_positions[signal_type] = 0

        # 订单被取消或拒绝
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"[{signal_type}] 订单被取消/拒绝")

        # 清空订单记录, 允许下次交易
        self.orders[signal_type] = None

    def calculate_position_size(self, signal_type):
        """
        计算仓位大小 - 固定比例法
        每次用该信号可用资金的 30% 买入
        """
        price = self.datas[0].close[0]

        # 防止除零错误
        if price <= 0:
            return 0

        # 该信号可用资金 = 总资金 / 3
        cash_per_signal = self.broker.getcash() / 3

        # 每次使用可用资金的 30%
        available_cash = cash_per_signal * 0.30

        # 计算股数
        size = int(available_cash / price)

        # 至少买1股
        size = max(size, 1)

        return size

    def next(self):
        """每个交易日执行得主逻辑 - Backtrader 自动调用"""
        # # 等待 ATR 有足够数据
        # if len(self.datas[0]) < self.params.atr_period:
        #     return

        # 获取当前价格
        price = self.datas[0].close[0]

        # 只处理当前激活的信号
        active_signal = self.params.active_signal

        if active_signal == 'signal_10_60':
            signal_value = self.signal_10_60[0]
        elif active_signal == 'signal_20_60':
            signal_value = self.signal_20_60[0]
        elif active_signal == 'signal_10_20_60':
            signal_value = self.signal_10_20_60[0]
        else:
            return

        # 处理该信号
        self._process_signal(active_signal, signal_value, price)

        # 检查止损
        self._check_stop_loss(price)

    def _process_signal(self, signal_type, signal_value, price):
        """处理单个信号的买卖逻辑"""
        # 如果有挂单, 不执行新操作
        if self.orders[signal_type] is not None:
            return

        # 修改这里=========================
        current_position = self.signal_positions[signal_type]

        # =========买入逻辑:信号 = 1, 且没有持仓 ==============
        if signal_value == 1 and current_position == 0:
            size = self.calculate_position_size(signal_type)
            if size > 0:
                order = self.buy(size=size)
                order.signal_type = signal_type
                self.orders[signal_type] = order
                self.entry_prices[signal_type] = price
                self.log(f"[{signal_type}] 买入信号 | 价格; {price:.2f} | 数量: {size}")

        # =========卖出逻辑: 信号=-1, 且有持仓 ================
        elif signal_value == -1 and current_position > 0:
            order = self.sell(size=current_position)
            order.signal_type = signal_type
            self.orders[signal_type] = order

            # 这里直接设置为0, 确保卖出后持仓清零
            self.signal_positions[signal_type] = 0
            self.log(f"[{signal_type}] 卖出信号 | 价格; {price:.2f} | 数量: {current_position}")

    def _check_stop_loss(self, current_price):
        """检查止损条件 - 亏损 20% 强制平仓"""
        for signal_type in self.signal_positions:
            # 只检查有持仓的信号
            if self.signal_positions[signal_type] > 0:
                entry_price = self.entry_prices[signal_type]
                if entry_price > 0:
                    # 计算亏损百分比
                    loss_pct = (entry_price - current_price) / entry_price

                    # 亏损达到20%, 触发止损
                    if loss_pct >= self.params.stop_loss:
                        self.log(f"[{signal_type}] 触发止损 | 买入价: {entry_price:.2f} | 当前价: {current_price:.2f} | 亏损:{loss_pct*100:.2f}%")
                        # 卖出全部持仓
                        if self.orders[signal_type] is None:
                            order = self.sell(size=self.signal_positions[signal_type])
                            order.signal_type = signal_type
                            self.orders[signal_type] = order


# ================================================
# 第四部分: 回测运行函数
# ================================================
def run_backtrader(project_dir, stock_code=None, signal_type=None):
    """
    运行回测主函数: 循环所有股票和所有信号
    Args:
        project_dir: 项目根目录
        stock_code: 股票代码
        signal_type: 信号类型
    Return:
        DataFrame: 所有回测结果
    """

    # 1. 检查目录
    raw_dir = project_dir / "data" / "raw"
    signal_dir = project_dir / "data" / "signals"

    if not raw_dir.exists():
        print(f" raw目录不存在: {raw_dir}")
        return None
    if not signal_dir.exists():
        print(f" signal目录不存在: {signal_dir}")
        return None

    # 2. 获取所有股票
    raw_files = list(raw_dir.glob("*.csv"))
    if len(raw_files) == 0:
        print(f" 没有找到任何CSV文件")
        return None

    # 如果指定了股票代码, 只回测该股票
    if stock_code:
        raw_files = [f for f in raw_files if f.stem == stock_code]
        if len(raw_files) == 0:
            print(f" 未找到股票: {stock_code}")
            return None

    # 3. 三种信号列表
    signal_list = ['signal_20_60', 'signal_10_60', 'signal_10_20_60']

    # 如果指定了信号类型, 只回测该信号
    if signal_type:
        if signal_type not in signal_list:
            print(f' 不支持的信号类型: {signal_type}')
            print(f' 支持的信号: {signal_list}')
            return None
        signal_list = [signal_type]

    print(f' -' * 60)
    print(f" 开始回测: {len(raw_files)} 只股票, {len(signal_list)} 种信号")

    # 4. 存储所有结果
    all_results = []

    # 5. 循环每只股票
    for stock_idx, stock_file in enumerate(raw_files, 1):
        stock_code = stock_file.stem

        print(f" [{stock_idx}/{len(raw_files)}] 股票: {stock_code}")
        print(f'=' * 60)

        # 加载数据
        df = load_and_merge_data(stock_code, project_dir)
        if df is None:
            print(f" 跳过 {stock_code}: 数据加载失败")
            continue
        print(f" 数据加载成功: 共 {len(df)} 行")

        # 打印信号分布
        print(f'\n 信号分布: ')
        for signal in signal_list:
            counts = df[signal].value_counts()
            print(f' {signal}:')
            print(f" 买入(1): {counts.get(1,0):6d} 天")
            print(f" 卖出(-1): {counts.get(-1,0):6d} 天")
            print(f" 观望(0): {counts.get(0,0):6d} 天")

        # 6. 循环每种信号
        for sig_type in signal_list:
            print(f"\n{'='*60}")
            print(f" 开始回测: {sig_type}")

            # 创建数据源
            data = PandasDataWithSignals(dataname=df)

            # 创建回测
            cerebro = bt.Cerebro()
            cerebro.adddata(data)

            # 添加策略
            cerebro.addstrategy(
                MultiSignalStrategy,
                stop_loss=0.20,
                atr_period=14,
                position_ratio=0.30,
                commission=0.001,
                active_signal=sig_type
            )
            total_cash = 30000
            cerebro.broker.setcash(total_cash)
            cerebro.broker.setcommission(commission=0.001)

            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

            print(f" \n 初始总资金: ${total_cash:,.2f} (3个信号 x $10,000)")
            print('-' * 40)

            # 运行回测
            results = cerebro.run()
            strat = results[0]

            # 提取结果
            returns = strat.analyzers.returns.get_analysis()
            sharpe = strat.analyzers.sharpe.get_analysis()

            final_cash = cerebro.broker.getvalue()
            total_return = returns.get('rtot', 0) * 100
            sharpe_ratio = sharpe.get('sharperatio', 0)
            if sharpe_ratio is None:
                sharpe_ratio = 0

            result = {
                'stock_code': stock_code,
                'signal_type': sig_type,
                'initial_cash': total_cash,
                'final_cash': final_cash,
                'profit': final_cash - total_cash,
                'profit_pct': (final_cash - total_cash) / total_cash * 100,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio
            }
            all_results.append(result)

            print(f' \n 回测结果: {stock_code} - {sig_type}')
            print('-' * 40)
            print(f" 初始资金: ${total_cash:,.2f}")
            print(f" 最终资金: ${final_cash:,.2f}")
            print(f" 总收益: ${final_cash - total_cash:,.2f}")
            print(f" 总收益率: {total_return:.2f}%")
            print(f" 夏普比率: {sharpe_ratio:.3f}")

    # =========== 下面缩进到 for 循环外面 ==========
    # 7. 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 8. 保存结果
    if len(results_df) > 0:
        results_dir = project_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_file = results_dir / "backtest_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f" \n 结果已保存到: {output_file}")

    # 9. 打印汇总
    if len(results_df) > 0:
        print(f"\n" + '=' * 60)
        print(f" 所有回测汇总")
        print(f" 总回测次数：{len(results_df)}")
        print(f" 平均收益率: {results_df['profit_pct'].mean():.2f}%")
        print(f" 平均夏普比率: {results_df['sharpe_ratio'].mean():.3f}")
        print(f" 盈利次数: {(results_df['profit'] > 0).sum()} / {len(results_df)}")

        # 按信号类型汇总
        print(f" \n 按信号类型汇总:")
        for sig_type in ['signal_20_60', 'signal_10_60', 'signal_10_20_60']:
            subset = results_df[results_df['signal_type'] == sig_type]
            if len(subset) > 0:
                print(f" {sig_type}:")
                print(f" 平均收益率: {subset['profit_pct'].mean():.2f}%")
                print(f" 平均夏普: {subset['sharpe_ratio'].mean():.3f}")
                print(f" 盈利/总数: {(subset['profit'] > 0).sum()} / {len(subset)}")

    return results_df


if __name__ == "__main__":
    # 设置目录
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent

    # 回测所有股票的所有信号
    results_df = run_backtrader(project_dir)

    if results_df is not None:
        print(f' \n 全部回测完成!')


"""
## 第4课：Backtrader回测

**任务目标**：
- 使用 Backtrader 框架进行量化策略回测
- 同时回测三种动量信号（20/60、10/60、10/20/60）
- 设置初始资金、手续费、止损等参数
- 汇总所有回测结果并保存

**实现方案**：
1. **数据加载与合并**：
   - 从 data/raw/ 读取价格数据（OHLCV）
   - 从 data/signals/ 读取信号数据
   - 按日期合并，过滤时间范围（2020-01-01 至 2026-08-04）
   - 检查信号列是否存在

2. **自定义数据源类**：
   - 继承 `bt.feeds.PandasData`
   - 定义三条信号线（signal_20_60、signal_10_60、signal_10_20_60）
   - 映射 DataFrame 列名到 Backtrader 线名

3. **策略类设计**：
   - 同时监控三种信号，每个信号独立使用 10000 资金
   - **ATR控制仓位**：波动大少买，波动小多买
   - **止损机制**：亏损 20% 强制平仓
   - **手续费**：0.1%
   - **每笔风险**：使用可用资金的 30%

4. **回测运行**：
   - 循环所有股票和所有信号
   - 初始总资金 30000（3个信号 × 10000）
   - 添加 Returns 和 SharpeRatio 分析器
   - 提取总收益、夏普比率等指标

5. **结果汇总**：
   - 保存为 CSV 文件到 results/ 目录
   - 按信号类型分组统计平均收益和夏普比率
   - 统计盈利次数

**核心代码结构**：
```python
# 数据加载
load_and_merge_data(stock_code, project_dir)

# 自定义数据源
class PandasDataWithSignals(bt.feeds.PandasData):
    lines = ('signal_20_60', 'signal_10_60', 'signal_10_20_60')

# 策略类
class MultiSignalStrategy(bt.Strategy):
    ├── __init__()              # 初始化信号和记录变量
    ├── log()                   # 日志打印
    ├── notify_order()          # 订单状态回调
    ├── calculate_position_size() # 计算仓位大小
    ├── next()                  # 每个交易日主逻辑
    ├── _process_signal()       # 处理单个信号买卖
    └── _check_stop_loss()      # 止损检查

# 回测运行
run_backtrader(project_dir, stock_code=None, signal_type=None)
"""