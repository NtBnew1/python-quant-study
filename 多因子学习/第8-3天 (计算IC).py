"""
第五部分: 使用滑动窗口IC计算因子的稳定性 + 绘图
输入: factors_and_standardized.xlsx -> Standardized_Factors
输出: 绘图 + IC汇总在控制台显示
说明:
- IC（Information Coefficient，信息系数）通常用因子值与“未来收益”的横截面相关系数度量。
- 本脚本按“每个交易日”在所有股票的截面上计算IC，再对IC做滚动均值平滑，并统计整体表现指标。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt



# =====================参数====================
INPUT_FILE = 'factors_and_standardized.xlsx'    # 数据excel文件名
SHEET_NAME = 'Standardized_Factors'             # 数据excel文件sheet名
RETURN_COL = 'future_return_20'                # 收益率列名
ROLLING_WINDOW = 60                             # 计算滚动IC的窗口长度 (单位: 交易日)
CORR_METHOD = 'spearman'                        # 相关系数计算方法
ANNUALIZE_FREQ = 252                            # 年化收益率计算频率 (单位: 交易日)

# =========读取数据========
# 1.) 从excele中读取数据
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
# 2.) 将字符串日期转为Datetime, 方便排序, 索引, 绘图
df['Date'] = pd.to_datetime(df['Date'])
# 3.) 先按日期, 再按公司代码排序, 保证同一天的多只股票放在一起
df = df.sort_values(['Date', 'company']).reset_index(drop=True)

# =============自动识别因子列====================
# 1.) 取所有数值型列 ( 因子和值, 收益等通常都是数值型)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# 2.) 从数值列中排除 "未来收益率" 和 "日收益列", 剩下的列默认视为候选因子列
#       - RETURN_COL: 未来收益列, 用来与因子做相关得到IC
#       - "daily_return": 当日收益 ( 一般不用来做IC)
factor_cols = [c for c in numeric_cols if c not in [RETURN_COL, "dail_return"]]

# =====================每日截面IC 计算========================
def compute_daily_ic(df, factor, ret_col, method=CORR_METHOD):
    '''
    计算 "单个因子" 的每日截面IC (信息系数)
    思路:
        -对每一天t, 取所有股票在t的 "因子值" 和 "未来收益率", 计算横截面相关系数 -> 得到当天的IC.
    参数:
        df: 包含日期, 公司, 因子, 收益的完整数据表
        factor: 因子列名
        ret_col: 未来收益列名 (字符串)
        method: 相关系数方法, 'pearson' 或'spearman'
    返回:
        DataFrame, 列为['Date', 'IC'], 每一行代表某一天的IC值
    '''
    # 1.) 保留当前因子, 未来收益, 日期, 公司 4列; 并在因子或收益缺失的行上做dropna
    sub = df[['Date', 'company', factor, ret_col]].dropna(subset=[factor, ret_col])

    # 2.) 按日期分组, 对每个日期的截面数据计算相关系数
    #     x[factor].corr(x[ret_col], method=method)
    #     在这里使用 .groupby().apply() 生成一个包含IC的sheets, 然后转为DataFrame
    #     注意: as_index=False, 可以避免未来pandas版本对groupby.apply返回结构的弃用警告
    ic_by_date = sub.groupby('Date', as_index=False).apply(
        lambda x: pd.Series({'IC': x[factor].corr(x[ret_col], method=method)})
    )

    # 3.) 只保留日期和IC两列作为返回结果
    return ic_by_date[['Date', 'IC']]

# ==================生成每日IC矩阵 (列=因子, 行=日期)===========================
# 思路:
#    - 对每个因子调用 compute_daily_ic() 计算每日IC
#    - 以日期为索引拼接成一个DateFrame: daily_ic_df
daily_ic_dict = {}
for f in factor_cols:
    ic_df = compute_daily_ic(df, f, RETURN_COL)         # 计算某个因子的每日IC
    daily_ic_dict[f] = ic_df.set_index("Date")['IC']    # 以日期为索引, 值为IC (Series)

# 将字典拼成表: 行=日期, 列=因子, 值=IC
daily_ic_df = pd.DataFrame(daily_ic_dict)

# =====================滚动IC计算======================
# 对每日IC 做滚动均值平滑(窗口=ROLLING_WINDOW), 可以过滤短期噪声, 观察稳定性趋势
# min_periods=10, 表示至少有10个有效值才计算滚动均值, 避免开头阶段大量NaN
rolling_ic_df = daily_ic_df.rolling(ROLLING_WINDOW, min_periods=10).mean()

# =====================IC 汇总指标 (整体表现)============
# 对每个因子统计:
#    - IC_mean: 平均IC (越高越好,  >0 代表正相关,  <0 代表负相关)
#    - IC_std: 标准差 (越小越好, 代表IC的稳定性)
#    - IC_IR_annual: 年化ICIR = (mean/std) * sqrt(年化收益率频率), 衡量 "稳定的超额相关性"
#    - Hit_Ratio: IC > 0 的比例 (越高代表更多时候反向预测正确)
#    - N: IC样本数量
summary_rows = []
for f in factor_cols:
    s = daily_ic_df[f].dropna()     # 去掉NaN值
    mean_ic = s.mean()             # 平均IC
    std_ic = s.std(ddof=1)               # 标准差
    icir = mean_ic / std_ic * sqrt(ANNUALIZE_FREQ) if std_ic else np.nan  # 防止被零除
    hit_ratio = ( s > 0 ).mean()       # 大于0的占比
    summary_rows.append([f, mean_ic, std_ic, icir, hit_ratio, len(s)])

#汇总表转为DataFrame并打印
summary_df = pd.DataFrame(
    summary_rows,
    columns = ['Factor', 'IC_mean', 'IC_std', 'ICIR_annual', 'Hit_Ratio', 'N']
)
print("\n=====IC 汇总指标=====")
print(summary_df)

# ================绘制图 "滚动IC曲线"========================
# 每一条代表一个因子的滚动IC均值, 便于比较不同因子的时间维度上的稳定性
plt.figure(figsize=(14,8))
for f in factor_cols:
    plt.plot(rolling_ic_df.index, rolling_ic_df[f], label=f)    # x= 日期索引, y=滚动IC均值
plt.axhline(0, color='black', linestyle='--', linewidth=1)       # y=0 的参数线
plt.title(f"Rolling IC ({RETURN_COL}) - Window={ROLLING_WINDOW}", fontsize=14)
plt.xlabel('Date')
plt.ylabel("IC Value")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# ================绘制图 "IC汇总对比图"============================
# 该图同时展示: 各因子的IC_mean, 年化ICIR

#取出要画图的列
factors = summary_df["Factor"]    # 因子列
ic_means = summary_df["IC_mean"]    # 平均IC列
ic_std = summary_df["IC_std"]    # 标准差列
icir = summary_df['ICIR_annual']    # 年化ICIR列
hit_ratio = summary_df['Hit_Ratio']    # 命中率列


# x 轴: 因子
x = np.arange(len(factors))
width = 0.2


fig, ax1 = plt.subplots(figsize=(14,8))

rects1 = ax1.bar(x - width, ic_means, width, label='IC_mean', color='skyblue')
rects2 = ax1.bar(x, icir/10, width, label='IC_IR_annual/10', color='lightblue')
ax1.set_ylabel('IC_mean & ICIR/10')
ax1.set_xticks(x)
ax1.set_xticklabels(factors, rotation=45, ha='right')
ax1.axhline(0, color='black', linestyle='--', linewidth=1)
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.5, linestyle='--')

ax2 = ax1.twinx()
rects3 = ax2.bar(x + width, hit_ratio, width, label='Hit_Ratio', color='red')
ax2.set_ylabel('Hit_Ratio')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper right')

plt.title(f"IC Summary Visualization", fontsize=14)
plt.tight_layout()
plt.show()




