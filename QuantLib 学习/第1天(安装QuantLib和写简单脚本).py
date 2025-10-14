'''
第1天：
安装并配置QuantLib和相关Python库，了解QuantLib的基本功能。
练习：编写一个简单脚本，使用QuantLib计算零息债券和固定利息债券的定价。
'''

'''
QuantLib 简单介绍
什么是 QuantLib？
QuantLib 是一个开源的金融计算库，主要用于 金融工具定价、风险管理和利率建模。
它原本用 C++ 写成，现在也有 Python 版本，叫 QuantLib-Python。

QuantLib 的用处：
- 可以计算 债券价格（零息债、固定利息债等）
- 可以构建 利率曲线，做贴现因子计算
- 可以对 期权、互换 等金融衍生品定价
- 可以进行 风险管理 和 蒙特卡洛模拟

QuantLib 的基础功能：
1. 日期与日历：处理交易日、节假日
2. 利率曲线：贴现、远期、收益率
3. 债券定价：零息债券、固定利息债券
4. 期权定价：欧式、美式期权
5. 风险建模：随机过程、模拟
'''

import QuantLib as ql

# 1. 设置评估日期
today = ql.Date(10,9,2025)  # 2025年9月10日
ql.Settings.instance().evalutionDate = today

# 利率曲线  ( 假设无风险利率为 3%)
rate = ql.SimpleQuote(0.03)
day_count = ql.Actual365Fixed()
calendar = ql.TARGET()
curve = ql.FlatForward(today, ql.QuoteHandle(rate), day_count)
curve_handle = ql.YieldTermStructureHandle(curve)


# 2. 零息债卷定价
maturity_date = calendar.advance(today, ql.Period(5, ql.Years)) # 5年期
face_value = 100.0

zero_coupon_bond = ql.ZeroCouponBond(2, calendar, face_value, maturity_date)
engine = ql.DiscountingBondEngine(curve_handle)
zero_coupon_bond.setPricingEngine(engine)

print("零息债卷价格: ", zero_coupon_bond.NPV())


# 3. 固定利息债卷定价
issue_date = today
schedule = ql.Schedule(issue_date,
                       maturity_date,
                       ql.Period(ql.Annual),    # 每年付息
                       calendar,
                       ql.Unadjusted, ql.Unadjusted,
                       ql.DateGeneration.Forward, False
                       )

fixed_rate = [0.05] # 固定票息率 5%
fixed_bond = ql.FixedRateBond(settlementDays = 2,
                              faceAmount=face_value,
                              schedule=schedule,
                              paymentDayCounter=day_count,
                              coupons=fixed_rate)

fixed_bond.setPricingEngine(engine)
print('固定利息债卷价格: ', fixed_bond.NPV())

