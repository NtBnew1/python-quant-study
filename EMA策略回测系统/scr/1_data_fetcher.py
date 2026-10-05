"""
文件：src/data_fetcher.py

功能：
-从 config.py 读取 STOCKS
-用 yfinance 下载数据
-保存到 data/raw/
"""

import yfinance as yf
import pandas as pd
import time
from pathlib import Path

from config import START_DATE, END_DATE, DATA_RAW_DIR

def fetch_stock_data(stock_code, start_date, end_date):
    """
    下载单只股票数据
    Args:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期 (当天)
    Returns:
        DataFrame: 包含OHLCV数据, 失败返回None
    """
    print(f" 下载中: {stock_code}..........")

    try:
        ticker = yf.Ticker(stock_code)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            print(f' {stock_code}: 没有数据')
            return None

        df = df.reset_index()
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date'])

        print(f" {stock_code}: {len(df)} 行数据")
        return df

    except Exception as e:
        print(f" {stock_code}: 下载失败 - {str(e)}")
        return None

def fetch_stocks(stocks):
    """
    下载多只股票数据
    Args:
        stocks: 股票代码列表, 如 ['AAPL', 'MSFT', 'GOOGL']
    Returns:
        tuple: (成功列表, 失败列表)
    """

    print(f'\n' + '+' * 80)
    print(f" 第1步: 获取数据")
    print(f'=' * 60)
    print(f" 股票数量: {len(stocks)} 只")
    print(f" 股票列表: {stocks}")
    print(f" 时间范围: {START_DATE} 至 {END_DATE}")
    print('=' * 60)
    print()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    success_list = []
    fail_list = []

    for i, stock_code in enumerate(stocks, 1):
        print(f" \n[{i}/{len(stocks)}] {stock_code}")
        df = fetch_stock_data(stock_code, START_DATE, END_DATE)

        if df is not None:
            output_file = DATA_RAW_DIR / f"{stock_code}.csv"
            df.to_csv(output_file, index=False)
            print(f" \n 已保存: {output_file}")
            success_list.append(stock_code)
        else:
            fail_list.append(stock_code)

        # 避免请求过快 (yfinance 限制)
        time.sleep(0.5)
    print(f' \n' + '=' * 60)
    print(f' 下载完成')
    print('=' * 60)
    print(f" 成功: {len(success_list)} 只 - {success_list}")
    print(f" 失败: {len(fail_list)} 只 - {fail_list}")
    print(f" 数据保存在: {DATA_RAW_DIR}")
    print(f'=' * 60)

    return success_list, fail_list

def fetch_single_stock(stock_code):
    """
    下载单只股票 (用于测试)
    Args:
        stock_code: 股票代码
    Returns:
        DataFrame 或 None
    """

    df = fetch_stock_data(stock_code, START_DATE, END_DATE)
    if df is not None:
        output_file = DATA_RAW_DIR / f"{stock_code}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n 已保存: {output_file}")
        print(f" \n 前5行数据: ")
        print(df.head())
        print(f' \n 数据信息: ')
        print(df.info())

        return df


if __name__ == "__main__":
    user_input = input(f" \n 股票代码: ").strip()
    if user_input == "":
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'VTI', 'SPY', 'QQQ', 'SNDK', 'CRWV', 'CBRS', 'CRCL']
        print(f' \n 使用默认列表: {stocks}')
    else:
        stocks = user_input.upper().split()
        print(f' \n 使用自定义列表: {stocks}')

    fetch_stocks(stocks)



"""
## 数据获取模块 (data_fetcher.py)

**任务目标**：
- 从 config.py 读取配置参数
- 使用 yfinance 下载股票数据
- 保存数据到 data/raw/ 目录
- 支持自定义股票列表或使用默认列表

**实现方案**：
1. **数据下载**：
   - 使用 yfinance 库（免费、无需 API 密钥）
   - 支持美股、港股等多市场
   - 自动处理 OHLCV 数据格式

2. **主要功能**：
   - `fetch_stock_data()`：下载单只股票数据
   - `fetch_stocks()`：批量下载多只股票数据
   - `fetch_single_stock()`：下载单只股票（用于测试）

3. **参数配置**：
   - 从 config.py 读取开始日期、结束日期、数据目录
   - 时间范围自动适配（START_DATE 至 END_DATE）

4. **数据保存**：
   - 重置索引，将日期列作为普通列
   - 只保留需要的列（Date、Open、High、Low、Close、Volume）
   - 保存为 CSV 文件到 data/raw/ 目录

5. **用户交互**：
   - 支持用户输入自定义股票代码（空格分隔）
   - 默认股票列表：AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, VTI, SPY, QQQ

**核心代码结构**：
```python
# 单只股票下载
fetch_stock_data(stock_code, start_date, end_date)

# 批量下载
fetch_stocks(stocks)

# 单只测试
fetch_single_stock(stock_code)
```
"""










