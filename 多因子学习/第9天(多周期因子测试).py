'''
第9天：

多周期因子测试（如月度、季度因子表现比较）
学习因子轮动与动态因子调整方法
练习：实现多周期因子计算与简单动态调整策略
'''

import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# ====================1. 获取标准化多因子数据======================
# 读取之前生成的标准化因子数据
file_path = "Day8-2_factors_and_standardized.xlsx"
df = pd.read_excel(file_path, sheet_name='Standardized_Factors')

# 确保日期列为datetime类型，方便后续按月、季度分组
df['Date'] = pd.to_datetime(df['Date'])

# 定义所有因子列
factor_cols=[
    '12m_return', '6m_return', '3m_return',    # 历史收益类因子
    'volatility_12m', 'MaxDrawdown',          # 风险因子
    'PE', 'PB', 'EV_EBITDA',                  # 估值因子
    'ROE', 'ROA', 'NetMargin'                 # 财务盈利能力因子
]

# 目标列：未来20日收益率，用于计算IC和策略收益
target = 'future_return_20'

# 添加月份和季度列，用pandas Period类型，方便时间运算
df['Month'] = df['Date'].dt.to_period('M')
df['Quarter'] = df['Date'].dt.to_period('Q')

# ==================2. 工具函数=================
def calc_ic(group, factor, target):
    '''
    计算一个时间段内（例如某个月、某个季度）
    单个因子与未来收益的Spearman相关系数(IC)
    group: 分组后的DataFrame
    factor: 因子列名
    target: 未来收益列名
    返回IC值或None
    '''
    if group[factor].isna().all() or group[target].isna().all():
        return None
    return spearmanr(group[factor], group[target], nan_policy='omit')[0]

def group_apply_dict(df, by_col, factor_cols, target):
    '''
    对DataFrame按指定周期（by_col）分组
    并计算每组下所有因子的IC
    返回：Series，index为Period，values为字典{因子: IC}
    '''
    func = lambda g: {f: calc_ic(g, f, target) for f in factor_cols}
    try:
        # pandas >= 2.1 版本可使用 include_groups=False
        s = df.groupby(by_col).apply(func, include_groups=False)
    except TypeError:
        # 老版本直接apply
        s = df.groupby(by_col).apply(func)
    return s

# =====================3. 多周期因子表现====================
# 月度 IC
monthly_series = group_apply_dict(df, 'Month', factor_cols, target)
monthly_ic = pd.DataFrame.from_records(
    monthly_series.values,
    index=monthly_series.index.astype(str)  # 转成字符串方便查看
).sort_index()
monthly_ic.index.name = 'Month'

# 季度 IC
quarterly_series = group_apply_dict(df, 'Quarter', factor_cols, target)
quarterly_ic = pd.DataFrame.from_records(
    quarterly_series.values,
    index=quarterly_series.index.astype(str)
).sort_index()
quarterly_ic.index.name = 'Quarter'

# 打印前5行，快速查看
print("月度IC: \n", monthly_ic.head())
print('季度IC: \n', quarterly_ic.head())

# ========================4. 动态因子轮动策略=====================
# 思路：
# 1. 每个月计算所有因子的IC，选出当月IC最高的因子
# 2. 下个月使用该因子选股：买入前20%（long），卖出后20%（short）
# 3. 记录每个月的long/short收益和Spread

results = []
for month, group in df.groupby('Month'):
    # 计算每个因子在该月的IC
    ic_dict = {factor: calc_ic(group, factor, target) for factor in factor_cols}
    if all(v is None for v in ic_dict.values()):  # 全None跳过
        continue
    # 选出IC最高的因子
    best_factor = max(ic_dict, key=lambda x: ic_dict[x] if ic_dict[x] is not None else float('-inf'))

    # 下个月使用该因子构建组合
    next_month = month + 1
    next_group = df[df['Month'] == next_month]
    if next_group.empty:
        continue

    next_group = next_group.copy()
    if next_group[best_factor].isna().all():
        continue
    next_group = next_group.dropna(subset=[best_factor, target])
    if next_group.empty:
        continue

    # 按因子值排名
    next_group['rank'] = next_group[best_factor].rank(ascending=False, method='first')

    # 前20%做多，后20%做空
    n = len(next_group)
    if n < 5:           # 数据太少跳过
        continue
    top = next_group[next_group['rank'] <= n * 0.2]
    bottom = next_group[next_group['rank'] > n * 0.8]
    if top.empty or bottom.empty:
        continue

    # 计算组合收益
    long_ret = top[target].mean()
    short_ret = bottom[target].mean()
    spread = (long_ret - short_ret) if (long_ret is not None and short_ret is not None) else None

    # 记录结果
    results.append({
        'Month': str(month),
        'BestFactor': best_factor,
        'LongReturn': long_ret,
        'ShortReturn': short_ret,
        'Spread': spread,
        'LongCompanies': ', '.join(top['company'].drop_duplicates().astype(str)),
        'ShortCompanies': ', '.join(bottom['company'].drop_duplicates().astype(str))
    })

# 转为DataFrame
rotation_df = pd.DataFrame(results)
print("\n动态因子轮动策略结果 ( 前5行): \n", rotation_df.head())

# =====================5. 绘制策略收益曲线======================
# Spread累积收益曲线，>0表示赚钱，<0表示亏钱
rotation_df['CumulativeReturn'] = (1 + rotation_df['Spread']).cumprod()

plt.figure(figsize=(12,6))
plt.plot(rotation_df['Month'], rotation_df['CumulativeReturn'], label='Strategy Cumulative Return', color='blue')
plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)  # 基准线
plt.title('Dynamic Factor Rotation Strategy ----- Cumulative Return')
plt.xlabel('Month')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # 日期倾斜45度，防止重叠
plt.tight_layout()
plt.show()

# =====================6. 保存结果到Excel======================
output_file = './Day9_factor_rotation_results.xlsx'

with pd.ExcelWriter(output_file) as writer:
    monthly_ic.to_excel(writer, sheet_name='Monthly_IC')
    quarterly_ic.to_excel(writer, sheet_name='Quarterly_IC')
    rotation_df.to_excel(writer, sheet_name='Rotation_Strategy', index=False)

print(f'\n所有结果已保存到{output_file}')

'''
=================总结:===========================


1. 月度IC（Monthly IC）：
   - IC（Information Coefficient）表示因子与未来收益的相关性（Spearman相关系数）。
   - 每个月IC>0：说明该因子在当月有效，因子值高的股票未来表现相对好，可以作为买入信号。
   - IC<0：说明因子在当月可能失效或反向，因子值高的股票未来表现可能差，作为做空信号参考。
   - IC接近0：因子当月对未来收益预测能力弱，不可靠。
   - 月度IC可反映因子短期有效性和波动性，适合观察因子在不同月份的表现。

2. 季度IC（Quarterly IC）：
   - 类似月度IC，但按季度统计，更平滑、更稳健。
   - IC>0：因子整体在该季度有效。
   - IC<0：因子在该季度可能失效或反向。
   - 季度IC可用来观察因子长期趋势和稳定性，避免短期噪声干扰。

3. 动态因子轮动策略（Dynamic Factor Rotation）：
   - 每个月计算所有因子IC，选出IC最高的因子作为下个月的选股依据。
   - 构建组合：
       - Long：因子值排名前20%的股票
       - Short：因子值排名后20%的股票
       - Spread = LongReturn - ShortReturn
   - Spread>0：组合赚钱，策略有效。
   - Spread<0：组合亏钱，策略失效或市场反向。
   - 将Spread进行累积乘积（CumulativeReturn）可以直观看到策略长期收益趋势：
       - 累积收益曲线持续上升：策略长期盈利
       - 曲线波动或下降：策略风险较高或存在周期性失效
   - 通过Spread和累积收益曲线可以评估策略是否值得在实际投资中应用。

4. 输出结果：
   - Excel包含三张表：
       - Monthly_IC：每个月各因子的IC
       - Quarterly_IC：每个季度各因子的IC
       - Rotation_Strategy：动态轮动策略结果，包括BestFactor、Long/Short收益、Spread和CumulativeReturn



新添加代码到results.append(): 
        'LongCompanies': ', '.join(top['company'].drop_duplicates().astype(str)),
        'ShortCompanies': ', '.join(bottom['company'].drop_duplicates().astype(str))
        这些会显示这个月有哪些公司做多, 和哪些公司做空. 
'''
