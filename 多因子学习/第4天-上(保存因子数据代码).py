'''
Day 4:

    计算多因子数据保存

'''


# 1. 导入库
import pandas as pd
import numpy as np
from datetime import datetime
import os


# 2. 读取数据
folder_path = './'
stock_files = ['HIMS_all_data.xlsx', 'HOOD_all_data.xlsx', 'IBRX_all_data.xlsx', 'JD_all_data.xlsx', 'NVDA_all_data.xlsx',
               'PLTR_all_data.xlsx', 'TEM_all_data.xlsx', 'TME_all_data.xlsx', 'TSLA_all_data.xlsx','ZETA_all_data.xlsx',
               'CRWD_all_data.xlsx', 'DDD_all_data.xlsx', 'LI_all_data.xlsx', 'RBLX_all_data.xlsx', 'XNET_all_data.xlsx',
               'XPEV_all_data.xlsx', 'KC_all_data.xlsx', 'DUOL_all_data.xlsx', 'RBLX_all_data.xlsx', 'ADSK_all_data.xlsx',
               'AVGO_all_data.xlsx', 'NIO_all_data.xlsx', 'BABA_all_data.xlsx', 'BIDU_all_data.xlsx', 'PDD_all_data.xlsx',
               'TAL_all_data.xlsx', 'EDU_all_data.xlsx', 'SOHU_all_data.xlsx']

# 3. 计算所有多因子
# =======存储所有股票的因子数据===========
factor_list = []

for file in stock_files:
    try:
        sheets = pd.read_excel(os.path.join(folder_path, file), sheet_name=None)
        # 分别提取所需表格
        price_df = sheets['price']
        income_df = sheets['Income_Statement']
        balance_df = sheets['Balance_Sheet']
        # cashflow_df = sheets['Cash_Flow']

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

# 保存到DataFrame
factor_df = pd.DataFrame(factor_list)

# 创建保存路径
output_path = './Day4_factor_all_stocks.xlsx'
factor_df.to_excel(output_path, index=False)
print(f'所有股票的因子数据已保存到: {output_path}')



