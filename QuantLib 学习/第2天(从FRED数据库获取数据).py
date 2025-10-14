'''
Day 2：获取美国国债数据
目标：学会获取真实的美国国债数据，用于债券定价和利率曲线构建。
任务：
选择数据源：
可用 pandas_datareader 获取国债 ETF（如 TLT, IEF）数据；
或使用 FRED 数据库（可用 pandas-datareader）。
下载并查看：
收益率、到期时间、票息等信息。
将数据整理成QuantLib可以使用的格式（利率曲线或现金流表）。
输出：可用国债数据的CSV或DataFrame。
'''

import pandas as pd
from pandas_datareader import data as pdr
import datetime
import QuantLib as ql

# 设置时间范围
start = datetime.datetime(2020,1,1)       # 数据开始时间：2020年1月1日
end = datetime.datetime.today()           # 数据结束时间：今天


#  FRED 国债收益率代码
# DGS1 -> 1年期
# DGS2 -> 2年期
# DGS5 -> 5年期
# DGS10 -> 10年期

fred_codes = ['DGS1', 'DGS2', 'DGS5', 'DGS10', 'DGS30']

# 用pandas_datareader获取数据
df = pdr.DataReader(fred_codes, 'fred', start, end)     #从 FRED 数据库 获取指定时间范围内的多列数据。

# 删除缺失值
df.dropna(inplace=True)     # 有些日期（比如节假日）没有数据，dropna 删除这些行。

# 重设索引
df.reset_index(inplace=True)        # reset_index 把日期索引变成普通列。
df.rename(columns={'index': 'Date'}, inplace=True)      # 把 index 改名为 Date。

print(df.head())

df.to_excel('./US_Treasury_Yields.xlsx', index=False)

# 整理为QuantLib 可用的表格
# 选择10年期的收益率列
ten_year_df = df[['DATE', 'DGS10']].copy()

# QuantLib 不能直接用 pandas 的日期，必须转换为 ql.Date。
dates = [ql.Date(d.day, d.month, d.year) for d in ten_year_df['DATE']]
rates = list(ten_year_df['DGS10'] / 100)    # 百分比转小数

# 构建零息利率曲线
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)  # 美国政府债券日历（考虑节假日）
day_count = ql.Actual360()   # 日计数方式，360天为一年
zero_curve = ql.ZeroCurve(dates, rates, day_count, calendar)


# 尝试: 获取最新利率
'''.zeroRate(..., ql.Continuous) 表示按连续复利方式计算。
.rate() 获取数值。'''
latest_rates = zero_curve.zeroRate(dates[-1], day_count, ql.Continuous).rate()

print(f'最新 10 年期零息利率: {latest_rates:.4%}')

# 绘制图
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.plot(df['DATE'], df['DGS1'], label='1y')
plt.plot(df['DATE'], df['DGS2'], label='2y')
plt.plot(df['DATE'], df['DGS5'], label='5y')
plt.plot(df['DATE'], df['DGS10'], label='10y')
plt.plot(df['DATE'], df['DGS30'], label='30y')

plt.title('US Treasury Yields (FRED)')
plt.xlabel('Date')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid(True)
plt.show()


'''
====================总结================
1. 获取数据

使用 pandas_datareader 从 FRED（美国联邦储备经济数据库） 获取国债收益率数据；
-选择了 1年、2年、5年、10年、30年期（DGS1, DGS2, DGS5, DGS10, DGS30）；
-时间范围：2020 年至今；
-删除缺失值，整理成 DataFrame。
👉 用处：拿到真实的 美国国债官方利率数据。

2. 保存数据
-将整理后的国债数据表保存为 US_Treasury_Yields.xlsx。
👉 用处：以后可以直接用 Excel 文件里的数据，不用每次都联网获取。

3. 转换为 QuantLib 可用格式
-提取 10年期国债收益率；
-转换成 QuantLib 的日期对象 (ql.Date) 和利率（小数形式）；
-构建 零息利率曲线（ZeroCurve）。
👉 用处：QuantLib 需要利率曲线来做 债券定价、利率建模。

4. 计算最新利率
-用 QuantLib 获取最近一天的 10年期零息利率；
-打印结果。
👉 用处：展示 最新的市场利率水平。

5. 绘图展示
-使用 Matplotlib 绘制了 1年、2年、5年、10年、30年期国债收益率的走势曲线；
-横轴：日期，纵轴：收益率（%）。
👉 用处：直观展示 利率随时间的变化趋势，方便观察市场走势。


'''












