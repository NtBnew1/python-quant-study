"""
Day 4：利率曲线与贴现因子

目标：
- 构建利率曲线 (Yield Curve)
- 计算贴现因子、远期利率
- 简单利率互换 (Swap) 定价
- 中文图表可视化
"""

import QuantLib as ql
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================读取数据=================
df_yields = pd.read_excel('./US_Treasury_Yields.xlsx')   # 读取国债收益率数据

# 选择某一天的收益率 (例如 2020-01-02)
date_str = "2020-01-02"
row = df_yields.loc[df_yields['DATE'] == date_str].iloc[0]  # 获取指定日期

# 提取各期限国债收益率 (% -> 小数)
rates_dict = {
    1: row['DGS1'] / 100,
    2: row['DGS2'] /100,
    5: row['DGS5'] /100,
    10: row['DGS10'] /100,
    30: row['DGS30'] /100,
}
print("使用的收益率点: \n", rates_dict)

# =============================设置QuantLib 日期和日历===========================
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)  # 美国政府债卷日历
today = ql.Date(2,1,2020)   # 评估日
ql.Settings.instance().evaluationDate = today   # 设置全局评估日
day_count = ql.Actual365Fixed() # 日计数方法

# 构造到期日和利率列表
dates = [today + ql.Period(n, ql.Years) for n in rates_dict.keys()] # 各期限到期日
rates = list(rates_dict.values())   # 对应收益率

# ==========================构建ZeroCurve (零息利率曲线)==========================
zero_curve = ql.ZeroCurve(dates, rates, day_count, calendar, ql.Linear())   #零息曲线插值
yield_curve = ql.YieldTermStructureHandle(zero_curve)   # 封装为Handle , 用于折现或Swap

# =======================计算贴现因子=============================
maturities = [1, 2, 5, 10, 30]  # 年期
discount_factors = []

for m in maturities:
    d = today + ql.Period(m, ql.Years)  # 到期日
    df = yield_curve.discount(d)    # 贴现因子 df = e^(-r*t)
    discount_factors.append((m, df))

print("\n贴现因子:")
for m, df in discount_factors:
    print(f"{m}年期: {df: .6f}")

# ==========================计算远期利率===================
# 示列: 1年后的1年远期利率
start = today + ql.Period(1, ql.Years)
end = today + ql.Period(2, ql.Years)
forward_rate = yield_curve.forwardRate(start, end, day_count, ql.Compounded).rate()
print(f"\n1年后的1年远期利率: {forward_rate*100: .2f}%")

# ============================中文绘制设置===========================
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 利率曲线
plt.figure(figsize=(10,5))
plt.plot(maturities, [r*100 for r in rates], "ro-", label='零息利率')
plt.title('利率曲线 (Zero Curve) -' + date_str)
plt.xlabel("到期期限 (年) ")
plt.ylabel("利率 (%)")
plt.grid(True)
plt.legend()
plt.show()

# 贴现因子曲线
plt.figure(figsize=(10,5))
plt.plot([m for m,_ in discount_factors], [df for _,df in discount_factors], 'bo-', label='贴现因子')
plt.title('贴现因子曲线 - ' + date_str)
plt.xlabel('到期期限 (年)')
plt.ylabel('贴现因子 (DF)')
plt.grid(True)
plt.legend()
plt.show()

# ======================利率互换Swap定价=====================

fixed_rate = 0.02   # 固定利率2%
tenor = ql.Period(5, ql.Years)  # Swap 期限 5年
notional = 1e6  # 名义本金100万

# Swap 起息日设置在today + 1 年 ( 为避免负时间)
start_date = calendar.advance(today, ql.Period(1, ql.Years))
maturity = calendar.advance(start_date, tenor)  # Swap 到期日

# 固定leg 付息时间表
fixed_schedule = ql.Schedule(
    start_date, maturity,
    ql.Period(ql.Annual),    # 每年付息一次
    calendar,
    ql.ModifiedFollowing, ql.ModifiedFollowing,
    ql.DateGeneration.Backward,  # backward 避免负时间
    False
)

# 浮动 leg 付息时间表
float_schedule = ql.Schedule(
    start_date, maturity,
    ql.Period(ql.Semiannual),       # 每半年付息
    calendar,
    ql.ModifiedFollowing, ql.ModifiedFollowing,
    ql.DateGeneration.Backward,
    False
)

# 浮动 利率 Index (6M USD Libor)
ibro_index = ql.USDLibor(ql.Period(6, ql.Months), yield_curve)

# 创建VanillaSwap 对象
swap = ql.VanillaSwap(
    ql.VanillaSwap.Payer,   # 付款方: 支付固定利率, 收浮动利率
    notional,       # 名义本金
    fixed_schedule,     # 固定leg schedule
    fixed_rate,     # 固定利率
    day_count,      # 固定 leg 日计数
    float_schedule,     # 浮动 leg schedule
    ibro_index,     # 浮动利率 index
    0.0,            # spread
    day_count       # 浮动 leg 日计数
)

# 定价引擎
engine = ql.DiscountingSwapEngine(yield_curve)  # 使用零息曲线折现
swap.setPricingEngine(engine)

# 输出swap.NPV
print(f"\n5年期利率互换 NPV: {swap.NPV():,.2f}")


'''
==================================总结===========================
1️⃣ 零息利率曲线 (Zero Curve)
-根据不同期限国债收益率构建
-插值方法保证任意期限利率可用
-用于折现现金流和计算远期利率

2️⃣ 贴现因子 (Discount Factor)
-折现未来现金流至评估日
-公式：DF(t) = e^(-r * t)
-不同期限的贴现因子不同，体现利率期限结构

3️⃣ 远期利率 (Forward Rate)
-表示未来某段时间的利率预期
-计算方式：通过零息曲线推导
-可用于利率互换、利率衍生品定价

4️⃣ 利率互换 (Interest Rate Swap)
-固定 leg：支付固定利率
-浮动 leg：收取市场浮动利率（如6M USD Libor）
-NPV（净现值）：
    -NPV > 0：对固定利率收款方有利
    -NPV < 0：对固定利率付款方有利
-Swap 定价依赖零息曲线折现未来现金流

5️⃣ 可视化
-利率曲线和贴现因子曲线直观显示时间价值
-中文字体与负号设置便于阅读
'''


