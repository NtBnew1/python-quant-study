"""
文件：src/4_ema_signals_weekly.py

功能：
-从 data/processed/ 读取日线数据
-重采样为周线（每周五收盘价）
-计算周线 EMA5, EMA10, EMA60
-生成 3 种信号（同第3步）
"""

import pandas as pd
import numpy as np
from pathlib import Path

from EMA策略回测系统.scr.config import DATA_SIGNALS_EMA_WEEKLY_DIR
from config import DATA_PROCESSED_DIR


# =========================
# 函数1. 加载数据
# =========================
def load_processed_data():
    """加载所有清洗后的数据文件
    Returns:
        list: 所有文件的路径列表
    """
    # 1. 找到 data/processed / 下所有csv文件
    processed_files = list(DATA_PROCESSED_DIR.glob("*.csv"))

    # 2. 如果没有文件, 打印提示并返回空列表
    if len(processed_files) == 0:
        print(f" 没有找到清洗后的数据文件, 请先运行 2_data_cleaner.py")
        return []

    # 3. 打印找到的文件数量
    print(f" 找到{len(processed_files)} 个数据文件")

    # 4. 返回文件列表
    return processed_files


# ==========================
# 函数2. 日线 转 周线
# ==========================
def resample_to_weekly(df):
    """将日线数据重采样为周线
    Args:
        df: 日线数据 (包含Date列)
    Returns:
        DataFrame: 周线数据
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # 重采样为周线 (每周五)
    weekly = df.resample('W-FRI').agg({
        'Open': 'first',        # 周一开盘
        'High': 'max',          # 本周最高价
        'Low': 'min',           # 本周最低价
        'Close': 'last',        # 周五收盘价
        'Volume': 'sum'         # 本周总成交量
    })

    weekly = weekly.dropna()
    weekly = weekly.reset_index()
    return weekly



# ==============================
# 函数3. 计算EMA
# ==============================
def calculate_ema(df, stock_code, periods):
    """计算EMA指标
    Args:
        df: 数据
        stock_code: 股票动漫
        periods: EMA 周期元组, 如(5, 10, 60)
    Returns:
        DATa Frame: 添加了EMA列的数据
    """
    df = df.copy()

    for period in periods:
        df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    print(f' {stock_code}: 已计算 周线EMA {periods[0]}, EMA{periods[1]}, EMA{periods[2]}')
    return df

# ==============================
# 函数4: 计算ATR
# ==============================
def calculate_atr(df, stock_code, period=14):
    """计算ATR (平均真实波幅) - 周线版本
    Args:
        df: 包含OHLC的数据
        stock_code: 股票代码
        period: ATR周线, 默认14
    Returns:
        DATa Frame: 添加了ATR列的数据
    """
    df = df.copy()

    # 计算True Range
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = (df['High'] - df['Close'].shift(1)).abs()
    df['low_close'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(period).mean()

    df = df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1)

    return df


# ===============================
# 函数5: 生成周线信号
# ===============================
def generate_weekly_signals(df, stock_code, atr_multiplier=0.5):
    """生成三种周线交易信号
    信号1: ema5 vs ema60
    信号2: ema10 vs ema60
    信号3: 三线多头 ema5 > ema10 > ema60
    Args:
        df: 包含EMA 和 ATR 列的数据
        stock_code: 股票代码
        atr_multiplier: ATR倍数, 默认0.5
    Returns:
        DataFrame: 添加了信号列的数据
    """
    df = df.copy()
    ema_5 = df['ema_5']
    ema_10 = df['ema_10']
    ema_60 = df['ema_60']
    atr = df['atr']

    # 信号1: ema5 vs ema60
    df['signal_ema5_60'] = 0
    buy_cond1 = (ema_5 > ema_60) & ((ema_5 - ema_60) > atr_multiplier * atr)
    df.loc[buy_cond1, 'signal_ema5_60'] = 1
    sell_cond1 = (ema_5 < ema_60) & ((ema_5 - ema_60) < -atr_multiplier * atr)
    df.loc[sell_cond1, 'signal_ema5_60'] = -1

    # 信号2: ema10 vs ema60
    df['signal_ema10_60'] = 0
    buy_cond2 = (ema_10 > ema_60) & ((ema_10 - ema_60) > atr_multiplier * atr)
    df.loc[buy_cond2, 'signal_ema10_60'] = 1
    sell_cond2 = (ema_10 < ema_60) & ((ema_10 - ema_60) < -atr_multiplier * atr)
    df.loc[sell_cond2, 'signal_ema10_60'] = -1

    # 信号3: 三线多头
    df['signal_ema5_10_60'] = 0
    df.loc[(ema_5 > ema_10) & (ema_10 > ema_60), 'signal_ema5_10_60'] = 1
    df.loc[(ema_5 < ema_10) & (ema_10 < ema_60), 'signal_ema5_10_60'] = -1

    # 统计信号分布
    print(f" {stock_code} 周线信号分布: ")
    for signal in ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']:
        counts = df[signal].value_counts()
        print(f" {signal}: 买入={counts.get(1, 0):5d}, 卖出={counts.get(-1,0):5d}, 观望={counts.get(0,0):5d}")

    return df

# ============================
# 函数6: 循环所有股票
# ============================
def process_all_stocks():
    """为所有股票生成周线EMA信号"""
    print(f' \n' +'=' * 60)
    print(f" 第4步: 生成EMA信号 (周线)")
    print(f'=' * 80)
    print(f" EMA周线: (5, 10, 60)")
    print(f" ATR周期: 14")
    print(f" ATR倍数: 0.5")
    print(f" =" * 60)

    # 创建输出目录
    DATA_SIGNALS_EMA_WEEKLY_DIR = Path(__file__).parent.parent / "data" / "signals_ema_周线"
    DATA_SIGNALS_EMA_WEEKLY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载所有文件
    files = load_processed_data()
    if not files:
        print(f" 没有找到数据文件")
        return

    success_count = 0

    # 2. 循环每只股票
    for file_path in files:
        stock_code = file_path.stem
        print(f" \n 处理中: {stock_code}..............")

        try:
            # 3. 读取日线数据
            df = pd.read_csv(file_path)

            # 4. 日线 -> 周线
            df_weekly = resample_to_weekly(df)

            # 5. 计算EMA
            df_weekly = calculate_ema(df_weekly, stock_code, (5, 10, 60))

            # 6. 计算ATR
            df_weekly = calculate_atr(df_weekly, stock_code, 14)

            # 7. 生成信号
            df_weekly = generate_weekly_signals(df_weekly, stock_code, atr_multiplier=0.5)

            # 8. 保存
            output_file = DATA_SIGNALS_EMA_WEEKLY_DIR / f"{stock_code}.csv"
            df_weekly.to_csv(output_file, index=False)

            print(f" {stock_code}: 已保存 -> {output_file}")
            success_count += 1

        except Exception as e:
            print(f" {stock_code}: 处理失败 -> {str(e)}")

    print(f" \n" + '=' * 60)
    print(f' 第4步完成')
    print(f'=' * 60)
    print(f" 成功生成周线信号: {success_count} 只")
    print(f" 数据保存在: {DATA_SIGNALS_EMA_WEEKLY_DIR}")
    print(f'=' * 60)

if __name__ == "__main__":
    # 为所有股票生成周线信号
    process_all_stocks()


"""
## EMA信号生成模块（周线） (4_ema_signals_weekly.py)

**任务目标**：
- 从 data/processed/ 读取日线数据
- 重采样为周线（每周五收盘价）
- 计算周线 EMA5、EMA10、EMA60
- 计算周线 ATR（平均真实波幅）
- 生成三种交易信号（使用 ATR 动态阈值）
- 保存信号数据到 data/signals_ema_周线/

**实现方案**：
1. **日线转周线**：
   - 重采样周期：每周五（W-FRI）
   - Open：周一开盘价（first）
   - High：本周最高价（max）
   - Low：本周最低价（min）
   - Close：周五收盘价（last）
   - Volume：本周总成交量（sum）

2. **EMA计算**：
   - 使用 `ewm(span=period, adjust=False).mean()` 计算指数移动平均
   - 周期：(5, 10, 60) 周线

3. **ATR计算**：
   - **True Range** = max(High - Low, |High - 前一周Close|, |Low - 前一周Close|)
   - **ATR** = TR 的 14 周移动平均
   - 用于动态阈值，适应周线级别波动

4. **信号生成规则**（同第3步）：
   - **信号1 (ema5 vs ema60)**：ATR动态阈值 ±0.5 × ATR
   - **信号2 (ema10 vs ema60)**：ATR动态阈值 ±0.5 × ATR
   - **信号3 (三线多头)**：ema5 > ema10 > ema60（无阈值）

5. **批量处理**：
   - 循环处理所有股票
   - 保存信号数据到 data/signals_ema_周线/

**核心代码结构**：
```python
# 日线转周线
resample_to_weekly(df)

# EMA计算
calculate_ema(df, stock_code, periods)

# ATR计算
calculate_atr(df, stock_code, period=14)

# 周线信号生成
generate_weekly_signals(df, stock_code, atr_multiplier=0.5)

# 批量处理
process_all_stocks()
"""
