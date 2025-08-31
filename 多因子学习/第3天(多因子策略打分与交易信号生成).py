'''
第3天：
设计策略框架，定义因子权重和组合打分方法
制定买入卖出信号的规则（如加权得分阈值）
练习：用Python代码实现简单的多因子买卖信号计算逻辑
'''

# 1. 导入库
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import zscore      # 用于标准化
import seaborn as sns
import matplotlib.pyplot as plt
import os


# 2. 读取数据
folder_path = './'
stock_files = ['HIMS_all_data.xlsx', 'HOOD_all_data.xlsx', 'IBRX_all_data.xlsx', 'JD_all_data.xlsx', 'NVDA_all_data.xlsx']

# 3. 计算所有多因子
# =======存储所有股票的因子数据===========
factor_list = []

for file in stock_files:
    try:
        sheets = pd.read_excel(os.path.join(folder_path, file), sheet_name=None)
        # 分别提取所需表格
        price_df = sheets['Two_Year_Stock']
        income_df = sheets['Income_Statement']
        balance_df = sheets['Balance_Sheet']
        cashflow_df = sheets['Cash_Flow']

        # 清除股价数据
        price_df['date'] = pd.to_datetime(price_df['Unnamed: 0'])   #把Unnamed 列当成日期
        price_df.set_index('date', inplace=True)            #设置日期为索引
        price_df = price_df.sort_index()

        #  =======================动量因子=========================
        price_df['12m_return'] = price_df['close'] / price_df['close'].shift(252) - 1
        price_df['6m_return'] = price_df['close'] / price_df['close'].shift(126) - 1
        price_df['3m_return'] = price_df['close'] / price_df['close'].shift(63) - 1

        # 取一年的数据
        one_year_ago = datetime.now() - pd.DateOffset(years=1)
        price_df_one_year = price_df[price_df.index >= one_year_ago]

        # ==============================价值因子===================
        latest_income = income_df.iloc[0]       # 最近一季度利润表
        latest_balance = balance_df.iloc[0]     # 最近一季度资产负债表

        # =====计算EPS======
        net_income = float(latest_income['netIncome'])
        share_outstanding = float(latest_balance.get('commonStockSharesOutstanding', np.nan))   # 总股本
        eps = net_income / share_outstanding if share_outstanding != 0 else np.nan

        # ========PE========
        latest_price = price_df['close'].iloc[-1]       # 最近收盘价
        pe_ratio = latest_price / eps if eps else np.nan    # PE= 股价 / 每股收益

        # ========PB=========
        total_equity = float(latest_balance.get('totalShareholderEquity', np.nan))  # 股东权益
        book_value_per_share = total_equity / share_outstanding if share_outstanding != 0 else np.nan
        pb_ratio = latest_price / book_value_per_share if book_value_per_share != 0 else np.nan  # PB = 股价 / 每股净资产

        # ========计算 EV/EBITDA==================
        ebitda = float(latest_income.get('ebitda', np.nan))             # EBITDA
        total_debt = float(latest_balance.get('totalLiabilites', 0))    # 总负债
        cash = float(latest_balance.get('cashAndCashEquivalAtCarryingValue', 0))    # 现金
        market_cap = latest_price * share_outstanding           # 市值
        ev = market_cap + total_debt - cash                     # 企业价值 EV
        ev_ebitda = ev / ebitda if ebitda else np.nan           # EV/EBITDA

        # ======================质量因子=================================
        roe = net_income / total_equity if total_equity != 0 else np.nan       # ROE
        total_assets = float(latest_balance.get('totalAssets', np.nan))        # 总资产
        roa = net_income / total_assets if total_assets != 0 else np.nan       # ROA
        revenue = float(latest_income.get('totalRevenue', np.nan))             # 营收
        net_margin = net_income / revenue if revenue != 0 else np.nan          # 净利润率

        # ==================波动率因子=======================
        price_df['daily_return'] = price_df['close'].pct_change()              # 每日收益率
        vol_df = price_df[price_df.index >= one_year_ago]                      # 最近一年的数据

        volatility = vol_df['daily_return'].std() * np.sqrt(252)               # 收益标准差 (年化波动率)

        # 最大回测计算
        cumulative = (1 + vol_df['daily_return']).cumprod()                    # 累积收益
        peak = cumulative.cummax()                                             # 历史最高点
        drawdown = (cumulative - peak) / peak                                  # 回测
        max_drawdown = drawdown.min()                                          # 最大回测


        # ====================整合所有因子=================
        factors = {
            'Stock': file.replace('_all_data.xlsx', ''),
            'PE': pe_ratio,
            'PB': pb_ratio,
            'EV_EBITDA': ev_ebitda,
            '12m_return': price_df_one_year['12m_return'].iloc[-1],
            '6m_return': price_df_one_year['6m_return'].iloc[-1],
            '3m_return': price_df_one_year['3m_return'].iloc[-1],
            'ROE': roe,
            'ROA': roa,
            'NetMargin': net_margin,
            'Volatility': volatility,
            'MaxDrawdown': max_drawdown
        }
        factor_list.append(factors)         # 一行数据, 每列一个因子
    except Exception as e:
        print(f'出错: {file}, 错误信息: {e}')

# ==========整合所有股票因子到DataFrame================
df = pd.DataFrame(factor_list)
df.set_index('Stock', inplace=True)
print(f"\n所有股票的因子数据")
print(df)


# 4.标准化每一个因子 (z-score)
# ================因子标准化 (z-score)=====================
standardized_factors = df.apply(zscore)
print(f" \n ===================标准化后的因子得分 (z-score)================= ")
print(standardized_factors.T)

# ================因子相关性分析============================
correlation_matrix = df.corr()
print(f" \n =================因子相关性矩阵===============")
print(correlation_matrix)

#=================可视化===========
plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置中文字体黑色
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('因子相关性热力图')
plt.tight_layout()
plt.show()

''' 前面4个就是昨天得代码.   后面3个才是今天学习的.  我们把昨天的代码copy.'''

''' 现在学习下面的代码. '''


# =================第3天任务学习============================
# 5. 分配因子权重
'''设定每个因子的权重 ( 可以自定义修改, 但必须加强=1)'''
weights = {
    'PE': 0.1,             # 价值因子：PE 越小越好 → 标准化后 PE 得分越高越好
    'PB': 0.1,             # PB 越小越好 → 标准化后 PB 得分越高越好
    'EV_EBITDA': 0.1,      # 企业价值比：越小越好
    '12m_return': 0.1,     # 动量因子：收益率越高越好
    '6m_return': 0.1,
    '3m_return': 0.1,
    'ROE': 0.1,            # 质量因子：越高越好
    'ROA': 0.1,
    'NetMargin': 0.1,
    'Volatility': -0.05,   # 波动率越小越好 → 所以设置为负权重
    'MaxDrawdown': -0.05   # 最大回撤越小越好 → 负权重
}

# 6. 计算加权得分
'''方法: 每个股票的标准化得分 * 因子权重, 然后求和'''
score = pd.Series(0, index=standardized_factors.index)  # 初始化得分Series

# 遍历每个因子, 按权重相乘后加到总得分:
for factor, weight in weights.items():
    score += standardized_factors[factor] * weight

# 把score 添加到DateFrame
df['Score'] = score

# 打印各股票的加权得分
print(f'\n===============综合得分==============')
print(df[['Score']].sort_values(by='Score', ascending=False))

# 7. 根据得分生成交易信号
# 设定规则: 得分排名前40% 的为'buy', 后40%的为'sell', 中间为'hold'
n = len(df)
buy_n = int(n * 0.4)    # 买入的股票数量
sell_n = int(n * 0.4)   # 卖出的股票数量

# 按照得分从高到低排序
df_sorted = df.sort_values(by='Score', ascending=False)

# 初始化默认信号为 Hold ( 持有)
df['Signal'] = 'Hold'

# 设置买入信号
df.loc[df_sorted.head(buy_n).index, 'Signal'] = 'Buy'

# 设置卖出信号
df.loc[df_sorted.tail(sell_n).index, 'Signal'] = 'Sell'

# 结果: 包含得分和信号
print('\n================交易信号==================')
print(df[['Score', 'Signal']].sort_values(by='Score', ascending=False))



''' 我们可以分开写.  应该说写成库或者保存到excel里, 
比如说计算因子,  很多代码, 我们不用每次都要计算, 我们从昨天代码copy. 我们可以把好的因子保存到excel里, 
改天需要这些因子直接加载excel.  如我们加载股票数据.    
其实都可以保存的,   我们也可以打包成库. 我们自己调用. 用class, def, 比较容易读. 
明天我们重新写和计算用class, def. 这样比较容易读.'''

'''还有权重我们是自己定义的.   我们可以计算权重的. 只是比较麻烦. 权重不管多少数据都要=1.'''

''' 5个股票, 就两个因为因子得分选择买, 一个hold, 其它sell. '''
