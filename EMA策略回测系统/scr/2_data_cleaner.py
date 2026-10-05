"""
文件：src/data_cleaner.py

功能：
-从 data/raw/ 读取数据
-检查缺失值、处理异常值
-统一列名（Date, Open, High, Low, Close, Volume）
-保存到 data/processed/

"""


import pandas as pd
from pathlib import Path

from IPython.utils.io import raw_print
from fontTools.misc.cython import returns

from config import DATA_RAW_DIR, DATA_PROCESSED_DIR

def clean_stock_data(df, stock_code):
    """
    清洗单只股票数据
    Args:
        df: 原始DataFrame
        stock_code: 股票代码

    return:
        DataFrame: 清洗后的数据
    """

    # 1. 复制数据, 避免修改原始数据
    df = df.copy()

    # 2. 确保日期格式正确
    df['Date'] = pd.to_datetime(df['Date'], utc=True)

    # 3. 检查缺失值
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f' {stock_code}: 发现缺失值')
        print(missing[missing > 0])
        df = df.dropna()

    # 4. 检查价格是否为0或负数
    for col in ['Open', 'High', 'Low', 'Close']:
        if (df[col] <= 0).any():
            print(f" {stock_code}: {col} 列有零或负值, 已删除")
            df = df[df[col]] > 0

    # 5. 检查 Volume 是否为负数
    if (df['Volume'] < 0).any():
        print(f" {stock_code}: Volume 有负值, 已取绝对值")
        df['Volume'] = df['Volume'].abs()

    # 6. 确保 High >= Low
    invalid_hl = df[df['High'] < df['Low']]
    if len(invalid_hl) > 0:
        print(f" {stock_code}: 发现{len(invalid_hl)} 行 High < Low, 已删除")
        df = df[df['High'] >= df['Low']]

    # 7. 确保 Close 在High 和 Low 之间
    invalid_close = df[(df['Close'] > df['High']) | (df['Close'] < df['Low'])]
    if len(invalid_close) > 0:
        print(f" {stock_code}: 发现{len(invalid_close)} 行 Close 超出High/Low范围, 已修正")
        df.loc[df['Close'] > df['High'], 'Close'] = df.loc[df['Close'] > df['High'], 'High']
        df.loc[df['Close'] < df['Low'], 'Close'] = df.loc[df['Close'] < df['Low'], 'Low']

    # 8. 按日期排序
    df = df.sort_values('Date').reset_index(drop=True)

    return df

def clean_all_stocks():
    """ 清洗所有股票数据"""
    print(f'=' * 60)
    print(f" 第2步: 清洗数据")
    print(f"=" * 60)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = list(DATA_RAW_DIR.glob("*.csv"))

    if len(raw_files) == 0:
        print(f" 没有找到原始数据文件, 请先运行 1_data_fetcher")
        return

    print(f" 找到 {len(raw_files)} 个数据文件")
    print()

    success_count = 0

    for raw_file in raw_files:
        stock_code = raw_file.stem
        print(f" 清洗中: {stock_code}...")

        try:
            df = pd.read_csv(raw_file)
            df_clean = clean_stock_data(df, stock_code)
            output_file = DATA_PROCESSED_DIR / f"{stock_code}.csv"
            df_clean.to_csv(output_file, index=False)

            print(f" {stock_code}: {len(df_clean)} 行 -> {output_file}")
            success_count += 1

        except Exception as e:
            print(f" {stock_code}: 清洗失败 - {str(e)}")

    print(f'\n' +"=" * 60)
    print(f' 第2步: 完成!')
    print(f'=' * 60)
    print(f" 成功清洗: {success_count} 只")
    print(f" 数据保存在: {DATA_PROCESSED_DIR}")
    print(f'=' * 60)


if __name__ == "__main__":
    # 清洗所有股票
    clean_all_stocks()


"""
## 数据清洗模块 (data_cleaner.py)

**任务目标**：
- 从 data/raw/ 读取原始股票数据
- 检查并处理缺失值
- 检查并处理异常值（零值、负值、逻辑错误）
- 统一列名格式
- 保存清洗后的数据到 data/processed/

**实现方案**：
1. **数据清洗流程**：
   - 复制数据，避免修改原始数据
   - 确保日期格式正确（转换为 datetime）
   - 检查缺失值并删除
   - 检查价格列（Open、High、Low、Close）是否有零值或负值，删除异常行
   - 检查 Volume 是否有负值，取绝对值
   - 确保 High >= Low，删除异常行
   - 确保 Close 在 High 和 Low 之间，超出时修正为边界值
   - 按日期排序并重置索引

2. **主要功能**：
   - `clean_stock_data()`：清洗单只股票数据
   - `clean_all_stocks()`：批量清洗所有股票数据

3. **参数配置**：
   - 从 config.py 读取数据目录（DATA_RAW_DIR、DATA_PROCESSED_DIR）

4. **异常处理**：
   - 价格列为零或负数 → 删除该行
   - Volume 为负数 → 取绝对值
   - High < Low → 删除该行
   - Close 超出 High/Low 范围 → 修正为 High 或 Low

**核心代码结构**：
```python
# 单只股票清洗
clean_stock_data(df, stock_code)

# 批量清洗
clean_all_stocks()
"""













