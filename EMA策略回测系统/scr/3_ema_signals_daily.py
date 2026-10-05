"""
文件：src/3_ema_signals_daily.py

功能：
-从 data/processed/ 读取数据
-计算 EMA5, EMA10, EMA60
-生成 3 种信号（阈值 0.02）：

"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_PROCESSED_DIR, DATA_SIGNALS_EMA_DIR

# ========================================
# 函数1. 计算EMA
# ========================================
def calculate_ema(df, stock_code, periods):
    """ 计算EMA指标
    Args:
        df: 清洗后的数据
        stock_code: 股票代码
        periods: EMA 周期元组, 如(5, 10, 60)
    Returns:
        DataFrame: 添加了EMA列的数据
    """
    df = df.copy()

    for period in periods:
        df[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    print(f" {stock_code}: 已计算 EMA{periods[0]}, EMA{periods[1]}, EMA{periods[2]}")
    return df


# =============================================
# 函数2. 计算ATR
# =============================================
def calculate_atr(df, stock_code, period=14):
    """计算ATR (平均真实波幅)
    Args:
        df: 包含OHLC的数据
        stock_code: 股票代码
        period: ATR周期, 默认14
    Returns:
        DataFrame: 添加了ATR列的数据
    """
    df = df.copy()

    # 计算True Range(真实波幅)
    # TR = max(High - Low, |High - 前一日Close|, |Low - 前一日Close|)
    df['high_low'] = df['High'] - df['Low']
    df['high_close'] = (df['High'] - df['Close'].shift(1)).abs()
    df['low_close'] = (df['Low'] - df['Close'].shift(1)).abs()

    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # 计算ATR (TR 的14日移动平均)
    df['atr'] = df['tr'].rolling(period).mean()

    # 删除中间列
    df = df.drop(['high_low', 'high_close', 'low_close', 'tr'], axis=1)

    print(f" {stock_code}: 已计算 ATR{period}")
    return df


# ==============================================
# 函数3. 生成信号
# ==============================================
def generate_signals(df, stock_code, atr_multiplier=0.5):
    """ 生成三种交易信号 (使用 ATR动态阈值)
    信号1: ema5 vs ema60 (ATR阈值)
    信号2: ema10 vs ema60 (ATR阈值)
    信号3: 三线多头 ema5 > ema10 > ema60 (无阈值)

    Args:
        df: 包含EMA 和 ATR 列的数据
        stock_code: 股票代码
        atr_multiplier: ATR倍数, 默认0.5

    Returns:
        DataFrame: 添加了信号列的数据
    """
    df = df.copy()

    # 获取列
    ema_5 = df['ema_5']
    ema_10 = df['ema_10']
    ema_60 = df['ema_60']
    atr = df['atr']

    # ===========================
    # 信号1: ema5 vs ema60
    # ===========================
    df['signal_ema5_60'] = 0

    # 买入: ema5 > ema60 且 差值 > atr_multiplier x ATR
    buy_cond1 = (ema_5 > ema_60) & ((ema_5 - ema_60) > atr_multiplier * atr)
    df.loc[buy_cond1, 'signal_ema5_60'] = 1

    # 卖出: ema5 < ema60 且 差值 < -atr_multiplier x ATR
    sell_cond1 = (ema_5 < ema_60) & ((ema_5 - ema_60) < -atr_multiplier * atr)
    df.loc[sell_cond1, 'signal_ema5_60'] = -1

    # ===========================
    # 信号2: ema10 vs ema60
    # ===========================
    df['signal_ema10_60'] = 0

    # 买入: ema10 > ema60 且 差值 > atr_multiplier x ATR
    buy_cond2 = (ema_10 > ema_60) & ((ema_10 - ema_60) > atr_multiplier * atr)
    df.loc[buy_cond2, 'signal_ema10_60'] = 1

    # 卖出: ema10 < ema60 且 差值 < -atr_multiplier x ATR
    sell_cond2 = (ema_10 < ema_60) & ((ema_10 - ema_60) < -atr_multiplier * atr)
    df.loc[sell_cond2, 'signal_ema10_60'] = -1


    # ================================
    # 信号3: 三线多头 ema5 > ema10 > ema60
    # ================================
    df['signal_ema5_10_60'] = 0
    df.loc[(ema_5 > ema_10) & (ema_10 > ema_60), 'signal_ema5_10_60'] = 1
    df.loc[(ema_5 < ema_10) & (ema_10 < ema_60), 'signal_ema5_10_60'] = -1


    # 统计信号分布
    print(f" {stock_code}信号分布: ")
    for signal in ['signal_ema5_60', 'signal_ema10_60', 'signal_ema5_10_60']:
        counts = df[signal].value_counts()
        print(f' {signal}: 买入={counts.get(1,0):5d}, 卖出={counts.get(-1,0):5d}, 观望={counts.get(0,0):5d}')

    return df

# =================================
# 函数4. 循环所有股票
# =================================
def generate_all_signals():
    """ 为所有股票生成EMA 信号"""
    print(f'\n' + '=' * 60)
    print(f' 第3步: 生成EMA信号 (日线)')
    print(f' =' * 60)
    print(f" EMA周期: (5, 10, 60) ")
    print(f" ATR周期: 14 ")
    print(f" ATR倍数: 0.5")
    print(f" =" * 60)

    # 创建输出目录
    DATA_SIGNALS_EMA_DIR.mkdir(parents=True, exist_ok=True)

    # 获取所有清洗后的数据文件
    processed_files = list(DATA_PROCESSED_DIR.glob("*.csv"))

    if len(processed_files) == 0:
        print(f" 没有找到清洗后的数据文件, 请先运行 2_data_cleaner.py")
        return

    print(f'\n 找到 {len(processed_files)} 个数据文件')
    print()

    success_count = 0

    for processed_file in processed_files:
        stock_code = processed_file.stem
        print(f' 处理中: {stock_code}........')
        try:
            # 获取清洗后的数据
            df = pd.read_csv(processed_file)

            # 计算EMA
            df = calculate_ema(df, stock_code, (5, 10, 60))

            # 计算ATR
            df = calculate_atr(df, stock_code, 14)

            # 生成信号
            df = generate_signals(df, stock_code, atr_multiplier=0.5)

            # 保存信号数据
            output_file = DATA_SIGNALS_EMA_DIR / f"{stock_code}.csv"
            df.to_csv(output_file, index=False)

            print(f' {stock_code}: 已保存 -> {output_file}')
            success_count += 1
            print()

        except Exception as e:
            print(f" {stock_code}: 处理失败 - {str(e)}")
            print()

    print(f" =" * 60)
    print(f' 第3步完成')
    print(f'=' * 60)
    print(f" 成功生成信号: {success_count} 只")
    print(f" 数据保存在: {DATA_SIGNALS_EMA_DIR}")
    print(f'=' * 60)

# ==========================================
# 测试第1个函数
# ==========================================
if __name__ == "__main__":
    """为所有股票生成信号"""
    generate_all_signals()



"""
## EMA信号生成模块 (3_ema_signals_daily.py)

**任务目标**：
- 从 data/processed/ 读取清洗后的数据
- 计算 EMA5、EMA10、EMA60
- 计算 ATR（平均真实波幅）
- 生成三种交易信号（使用 ATR 动态阈值）
- 保存信号数据到 data/signals_ema/

**实现方案**：
1. **EMA计算**：
   - 使用 `ewm(span=period, adjust=False).mean()` 计算指数移动平均
   - 周期：(5, 10, 60)

2. **ATR计算**：
   - **True Range** = max(High - Low, |High - 前一日Close|, |Low - 前一日Close|)
   - **ATR** = TR 的 14 日移动平均
   - 用于动态阈值，适应市场波动

3. **信号生成规则**（使用 ATR 动态阈值）：
   - **信号1 (ema5 vs ema60)**：
     - 买入：ema5 > ema60 且差值 > 0.5 × ATR
     - 卖出：ema5 < ema60 且差值 < -0.5 × ATR
   - **信号2 (ema10 vs ema60)**：
     - 买入：ema10 > ema60 且差值 > 0.5 × ATR
     - 卖出：ema10 < ema60 且差值 < -0.5 × ATR
   - **信号3 (三线多头)**：
     - 买入：ema5 > ema10 > ema60（无阈值）
     - 卖出：ema5 < ema10 < ema60（无阈值）

4. **信号统计**：
   - 统计每个信号的买入天数、卖出天数、观望天数
   - 显示信号分布

5. **批量处理**：
   - 循环处理所有股票
   - 保存信号数据到 data/signals_ema/

**核心代码结构**：
```python
# EMA计算
calculate_ema(df, stock_code, periods)

# ATR计算
calculate_atr(df, stock_code, period=14)

# 信号生成
generate_signals(df, stock_code, atr_multiplier=0.5)

# 批量处理
generate_all_signals()
"""

