'''
Day 3：债券定价与现金流分析

目标：
- 使用Day 2获取的数据进行债券建模与定价。
- 理解现金流、票息、到期收益率（YTM）。

任务：
- 使用QuantLib创建零息债券和固定利率债券对象。
- 计算债券净现值（NPV）、到期收益率。
- 绘制债券现金流表。
输出：债券定价脚本 + 现金流图表。
'''

import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# 1. 读取数据
# -------------------------------
# 国债收益率数据
df_yields = pd.read_excel('./US_Treasury_Yields.xlsx')

# 读取国债拍卖信息
'''https://www.treasurydirect.gov/auctions/announcements-data-results/''' # 这是国债信息
df_sec = pd.read_csv('./Securities.csv')

print(df_yields.head())
print(df_sec.head())

# 选择5年期国债(DGS5) 对应的CUSIP
bond_info = df_sec[df_sec['CUSIP'] == '91282CNX5'].iloc[0]

# -----------------------
# 日期处理
# ----------------------
# 将字符串日期 (如 8/30/2030) 转为QuantLib.Date
def to_ql_date(date_str):
    dt = datetime.strptime(date_str, "%m/%d/%Y")
    return ql.Date(dt.day, dt.month, dt.year)

# QuantLib 格式日期
issue_date = to_ql_date(bond_info['Issue Date'])
maturity_date = to_ql_date(bond_info['Maturity Date'])

# Python datetime 用于绘制图
maturity_dt_py = datetime.strptime(bond_info['Maturity Date'], "%m/%d/%Y")

# ---------------------------
# 债卷基本参数
# ---------------------------
face_value = 100    # 国债面值
coupon_rate = 0.0167    # DGS5 年利率1.67%
frequency = ql.Semiannual   # 半年付息

# QuantLib 日计数方法, 需要指定Convention
day_count = ql.ActualActual(ql.ActualActual.ISDA)

calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)  # 官方债卷日历
settlement_days = 1 # 结算日期


# ------------------------------
# 2. 创建固定利率债卷对象
# ------------------------------
# 构建付息时间表
schedule = ql.Schedule(issue_date,
                       maturity_date,
                       ql.Period(frequency),    # 半年一次付息
                       calendar,
                       ql.Following,            # 结算日规则
                       ql.Following,
                       ql.DateGeneration.Backward,   # 生成日期顺序
                       False)

# 创建固定利率债卷对象
fixed_bond = ql.FixedRateBond(settlement_days, face_value, schedule, [coupon_rate], day_count)

#----------------------
# 3. 构建贴现曲线(FlatForward)
#----------------------
# 选择特定日期的收益率 ( 这里用2020-01-02 的DGS5)
ytm_val = df_yields.loc[df_yields['DATE'] == '2020-01-02', 'DGS5'].values[0]
ytm = ytm_val / 100     # 转换为小数形式

# 创建恒定利率贴现曲线
discount_curve = ql.FlatForward(issue_date, ytm, day_count, ql.Compounded, frequency)
discount_curve_handle = ql.YieldTermStructureHandle(discount_curve)

# 设置债卷定价
engine = ql.DiscountingBondEngine(discount_curve_handle)
fixed_bond.setPricingEngine(engine)

# 计算债卷净现值(NPV)
npv = fixed_bond.NPV()
print(f"固定利率债卷净现值 (NPV) : {npv: .4f}")

# -------------------------
# 4. 绘制现金流图
# --------------------------
# 获取现金流金额
cf_amounts = [cf.amount() for cf in fixed_bond.cashflows()]

# 将QuantLib.Date 转换Python datetime, 用于绘图
cf_dates_py = [datetime(cf.date().year(), cf.date().month(), cf.date().dayOfMonth())
               for cf in fixed_bond.cashflows()]

# 设置中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.dates as mdates
plt.figure(figsize=(10,5))

cf_dates_mpl = mdates.date2num(cf_dates_py)

plt.bar(cf_dates_mpl, cf_amounts, width=15, color='skyblue', label='现金流')

plt.plot(cf_dates_mpl, cf_amounts, color='red', marker='o', label='现金流曲线')

plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

plt.title('DGS5 真实5年期国债现金流图')
plt.xlabel('日期')
plt.ylabel('现金流 ($)')
plt.legend()
plt.tight_layout()
plt.show()



'''
======================总结==========================
今天主要做了三件事：  
1. 把国债的收益率和债券信息导入进来，找到目标债券（DGS5）。  
2. 用 QuantLib 建模，创建了一个固定利率债券，设定了面值、票息率、付息频率和到期日。  
3. 根据市场收益率搭建折现曲线，算出了债券的净现值（NPV），然后把未来的现金流绘制成图表。  

最大的收获：  
- 明白了债券价格其实就是“未来现金流折现的总和”。  
- 会用 QuantLib 快速生成现金流表，还能画图直观看每期的付款情况。

'''









