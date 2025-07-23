'''
第1天：

理解多因子策略概念和常见因子类别（价值、动量、质量、波动率等）
学习如何从历史数据中选择和构建因子
练习：用pandas计算几个基础因子（如市盈率PE、过去12个月收益率等）

'''

# ========导入库========
import pandas as pd
import numpy as np
from datetime import datetime



# ===========读取多sheet==========
file_path = './HIMS_all_data.xlsx'          # 路径
sheets = pd.read_excel(file_path, sheet_name=None)      # 一次性读取多个sheets

# 分别提取股价数据和3个报表
price_df = sheets['Two_Year_Stock']             # 这几个sheets名称是你保存数据的名称. 需要和excel sheet对称.
income_df = sheets['Income_Statement']
balance_df = sheets['Balance_Sheet']
cashflow_df = sheets['Cash_Flow']

# =============清除股价数据============
# print(price_df.columns)            #  这是打印股价数据的columns. 需要索引

# 现在把Unnamed 改成date. 用date来索引
price_df['date'] = pd.to_datetime(price_df['Unnamed: 0'])            # excel里的date是Unnamed.

# 设置date为索引
price_df.set_index('date', inplace=True)

# 按照日期排序
price_df = price_df.sort_index()
# print(price_df)               # 确认打印有没有问题

# =====================动量因子:(12个月收益率/6个月收益率/3个月收益率)=========================
'''动量多因子就是收益率: 现在要计算收益率'''
price_df['12m_return'] = price_df['close'] / price_df['close'].shift(252) - 1   #计算一年
price_df['6m_return'] = price_df['close'] / price_df['close'].shift(126) - 1    #计算半年
price_df['3m_return'] = price_df['close'] / price_df['close'].shift(63) - 1     #计算3个月

# '''这个股价数据500行.  大概两个数据.  现在要选择只要一年数据'''
one_year_ago = datetime.now() - pd.DateOffset(years=1)      # 时间只要1年
price_df_1yr = price_df[price_df.index >= one_year_ago]     # 把所有数据减1年

# ''' 打印收益率'''
print(f"一年的收益率: \n{price_df_1yr['12m_return']}")
print(f"半年的收益率: \n{price_df_1yr['6m_return']}")
print(f"3个月的收益率: \n{price_df_1yr['3m_return']}")
''' length 249 意思就是去年只有249天股价数据. 249天就是一年.  有的时候是252天一年. '''

# =====================价值因子:(PE/PB/EV/EBITDA) =======================================
# 提取最新一期利润表 和资产负债表
latest_income = income_df.iloc[0]
latest_balance = balance_df.iloc[0]

#  计算EPS:  需要提取net_income 和 share_outstanding 数据
net_income = float(latest_income['netIncome'])    # 净利润, 需要去income statement 查看. 有些名称不对
shares_outstanding = float(latest_balance.get('commonStockSharesOutstanding', np.nan))   # 总资本
eps = net_income / shares_outstanding if shares_outstanding != 0 else np.nan
print(f"EPS: \n{eps}")

# 最新收盘价
latest_price = price_df['close'].iloc[-1]

# 计算PE (市盈率)
pe_ratio = latest_price / eps if eps else np.nan

# 计算PB (市净率)
total_equity = float(latest_balance.get('totalShareholderEquity', np.nan)) #股东权益
book_value_per_share = total_equity / shares_outstanding if shares_outstanding != 0 else np.nan
pb_ratio = latest_price / book_value_per_share if book_value_per_share else np.nan
print(f"PB_Ratio: \n{pb_ratio}")

# 计算EV / EBITDA
ebitda = float(latest_income.get('ebitda', np.nan))
# EV = 市值 + 债务 - 现金
total_debt = float(latest_balance.get('totalLiabilites', 0))     # 总负债
cash = float(latest_balance.get('cashAndCashEquivalentAtCarryingValue', 0))     # 现金
market_cap = latest_price * shares_outstanding         # 市值
ev = market_cap + total_debt - cash

# EV/EBITDA 指标
ev_ebitda = ev / ebitda if ebitda else np.nan

#  ---------打印多因子-----------
print(f"\n [价值因子计算结果]")
print(f"EPS: {eps: .2f}")                    # 每一股盈利是0.20
print(f"股价: {latest_price: .2f}")           # 最新股价
print(f"PE (市盈率): {pe_ratio: .2f}")         # eps: 0.20, 股价是49.85. 投资这家股票需要付出248倍来购买: 是高估严重
print(f"PB (市净率): {pb_ratio: .2f}")          # 当前 PB = 22.38，说明公司账面上每股资产较低，但市场出价高得多，代表市场对公司未来成长极度看好或被高估。
print(f"EV/EBITDA: {ev_ebitda: .2f}")        # 整体估值远远高于盈利能力;


# =====================质量因子:(ROE/净利润率/ROA)=========================================
# ROE = 净利润 / 股东权益
roe = net_income / total_equity if total_equity != 0 else np.nan

# ROA = 净利润 / 总资产
total_assets = float(latest_balance.get('totalAssets', np.nan))
roa = net_income / total_assets if total_assets != 0 else np.nan

# 净利润率 = 净利润 / 营收
revenue = float(latest_income.get('totalRevenue', np.nan))
net_margin = net_income / revenue if revenue != 0 else np.nan

# ====打印质量因子结果: =====
print(f"\n[质量因子计算结果]")
print(f"ROE (净资产市盈率): {roe: .2%}")
print(f"ROA (资产收益率) : {roa: .2%}")
print(f"净利润率: {net_margin: .2%}")


# =====================波动率因子:(收益标准差/最大回撤/Beta)=================================
# 计算每日收益率
price_df['daily_return'] = price_df['close'].pct_change()

# 选择最近一年数据
vol_df = price_df[price_df.index >= one_year_ago]

# 收益率标值 (年化波动率)
volatility = vol_df['daily_return'].std() * np.sqrt(252)

# 最大回测 (使用累计收益最大跌幅)
cumulative = ( 1+ vol_df['daily_return']).cumprod()
peak = cumulative.cummax()
drawdown = (cumulative - peak) / peak
max_drawdown = drawdown.min()

# =====打印波动率因子结果======
print(f"\n[波动率因子计算结果]")
print(f"收益年化标准差 (波动率): {volatility: .2%}")
print(f"最大回测: {max_drawdown: .2%}")


