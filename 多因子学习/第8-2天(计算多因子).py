'''
重写计算多因子准备计算IC.
今天是把获取的公司时间计算成多因子, 并且保存.
然后计算标准化因子, 并且保存.
'''


# ==============导入库============
import os
import pandas as pd
import numpy as np
from scipy.stats import zscore      # 这是用标准化因子的库

# ==============读取数据============
def load_all_data(path='.'):
    '''
    从指定目录读取所有以 "_all_data.xlsx" 结尾的文件,
    每个excel文件对应一个股票, 包含多个sheet(如价格, 财务表等)
    返回格式:
        {
        '股票代码': {
            'sheet名称1: DataFrame,
            'sheet名称2: DataFrame,
           ...
        },
       ...
        }
    '''

    #找到所有文件名以 "_all_data.xlsx" 结尾的文件
    file_list = [f for f in os.listdir(path) if f.endswith('_all_data.xlsx')]
    all_data = {}
    for file in file_list:
        # 提取股票代码
        stock_code = file.replace('_all_data.xlsx', '')
        xls = pd.ExcelFile(file)
        # 读取改excel 所有sheet为DataFrame, 并传入字典
        sheet_dict = {sheet: xls.parse(sheet) for sheet in xls.sheet_names}
        all_data[stock_code] = sheet_dict
    return all_data

# ==============时间筛选-==================
def get_recent_data(df, date_col, years=5):
    """
    从DataFrame中筛选最近 years 年的数据
    参数:
        df: 原始数据 DataFrame
        date_col: 日期列名
        years: 筛选最近 years 年的数据
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])  # 转换日期列为datetime格式
    df = df.sort_values(by=date_col)

    # 获取数据中最早和最晚日期
    last_date = df[date_col].max()
    earliest_date = df[date_col].min()
    avaliable_years = (last_date - earliest_date).days / 365  # 可用年数

    # 如果数据不足 "years" 年, 就用数据实际可用的年数
    years_to_use = min(years, avaliable_years)
    start_date = last_date - pd.Timedelta(days=int(years_to_use * 365))

    # 筛选出指定时间范围的数据
    return df[df[date_col] >= start_date]


# ==================计算未来收益率======================
def calc_future_return(price_df, close_col='close', periods=[20, 60]):
    """
    根据收盘价计算未来 N 天的收益率
    参数:
        price_df: 包含收盘价的 DataFrame
        close_col: 收盘价列名
        period: 计算未来 N 天的收益率, 如 [20, 60] 表示计算 20, 60 天的收益率
    """
    df = price_df.copy()
    for p in periods:
        # shift(-p) 向上移动 p 行, 表示未来 p 天的价格
        df[f"future_return_{p}"] = df[close_col].shift(-p) / df[close_col] - 1
    return df

# ==============计算多因子=========================
def calculate_factors(price_df, income_df, balance_df):
    """
    计算价格类因子 + 财务类因子
    返回:
        包含所有因子, 未来收益率的DataFrame
    """
    price_df = price_df.copy()
    price_df['Date'] = pd.to_datetime(price_df['Unnamed: 0'])     # 用alpha vantage获取数据, 是没有日期的.
    price_df.set_index('Date', inplace=True)
    price_df.sort_index(inplace=True)

    # 计算未来20 天 和 60 天的收益率
    price_df = calc_future_return(price_df, close_col='close', periods=[20, 60])    # 这是调用计算未来收益率的函数

    # ===价格类因子====
    price_df['12m_return'] = price_df['close'] / price_df['close'].shift(252) - 1  # 过去12月收益率
    price_df['6m_return'] = price_df['close'] / price_df['close'].shift(126) - 1   # 过去6月收益率
    price_df['3m_return'] = price_df['close'] / price_df['close'].shift(63) -1     # 过去3月收益率

    # 波动率 ( 过去12个月的日收益率标准差)
    price_df['daily_return'] = price_df['close'].pct_change()
    price_df['volatility_12m'] = price_df['daily_return'].rolling(window=252).std()

    # 最大回测 (过去12个月)
    def max_drawdown(array):
        roll_max = np.maximum.accumulate(array)    # 历史最高值
        drawdown = (array - roll_max) / roll_max    # 回测率
        return np.min(drawdown)    # 最大回测率
    price_df['MaxDrawdown'] = price_df['close'].rolling(window=252).apply(max_drawdown, raw=True)

    # =======财务类因子 (取最近一期财报) ====
    latest_income = income_df.copy().iloc[-1]
    latest_balance = balance_df.copy().iloc[-1]

    # EPS, PE
    net_income = float(latest_income['netIncome'])
    share_outstanding = float(latest_balance.get('commonStockSharesOutstanding', np.nan))
    eps = net_income / share_outstanding if share_outstanding != 0 else np.nan
    latest_price = price_df['close'].iloc[-1]
    pe_ratio = latest_price / eps if eps else np.nan

    # PB
    total_equity = float(latest_balance.get('totalShareholderEquity', np.nan))
    book_value_per_share = total_equity / share_outstanding if share_outstanding != 0 else np.nan
    pb_ratio = latest_price / book_value_per_share if book_value_per_share != 0 else np.nan

    # EV/EBITDA
    ebitda = float(latest_income.get('ebitda', np.nan))
    total_debt = float(latest_balance.get('totalLiabilities', 0))
    cash = float(latest_balance.get('cashAndCashEquivalAtCarryingValue', 0))
    market_cap = latest_price * share_outstanding
    ev = market_cap + total_debt - cash
    ev_ebitda = ev / ebitda if ebitda else np.nan

    # ROE, ROA, 净利率
    roe = net_income / total_equity if total_equity != 0 else np.nan
    total_assets = float(latest_balance.get('totalAssets', np.nan))
    roa = net_income / total_assets if total_assets != 0 else np.nan
    revenue = float(latest_income.get('totalRevenue', np.nan))
    net_margin = net_income / revenue if revenue != 0 else np.nan

    # 将财务因子添加到价格数据中
    price_df['PE'] = pe_ratio
    price_df['PB'] = pb_ratio
    price_df['EV_EBITDA'] = ev_ebitda
    price_df['ROE'] = roe
    price_df['ROA'] = roa
    price_df['NetMargin'] = net_margin

    # 输出列
    factor_cols = ['12m_return', '6m_return', '3m_return', 'volatility_12m',
                   'MaxDrawdown', 'PE', 'PB', 'EV_EBITDA', 'ROE', 'ROA', 'NetMargin']
    output_cols = ['Date'] + factor_cols + ['future_return_20', 'future_return_60']

    # 还原index 并输出
    output_df = price_df.reset_index()[output_cols]
    return output_df

# ===============主程序====================
if __name__ == '__main__':
    # 1. 读取所有股票数据
    all_data = load_all_data()

    # 2. 计算多因子(原始值)
    all_factors_list = []
    for company, sheets, in all_data.items():
        price_df = sheets.get('price')
        income_df = sheets.get('Income_Statement')
        balance_df = sheets.get('Balance_Sheet')

        # 如果数据缺失则跳过改股票
        if price_df is None or income_df is None or balance_df is None:
            print(f'{company} 数据不完整, 跳过')
            continue

        # 筛选最近 5 年的数据
        price_df = get_recent_data(price_df, 'Unnamed: 0', years=5)
        income_df = get_recent_data(income_df, 'fiscalDateEnding', years=5)
        balance_df = get_recent_data(balance_df, 'fiscalDateEnding', years=5)

        if price_df.empty or income_df.empty or balance_df.empty:
            print(f'{company} 数据不足 5 年, 跳过')
            continue

        # 计算因子
        factors_df = calculate_factors(price_df, income_df, balance_df)
        factors_df['company'] = company
        all_factors_list.append(factors_df)

    # 合并所有股票的因子数据
    all_factors_df = pd.concat(all_factors_list, ignore_index=True)

    # 3. 标准化因子 ( 每个日期对所有股票做 Z-score 标准化 )
    factor_cols = ['12m_return', '6m_return', '3m_return', 'volatility_12m',
                   'MaxDrawdown', 'PE', 'PB', 'EV_EBITDA', 'ROE', 'ROA', 'NetMargin']

    standardized_df = all_factors_df.copy()
    for col in factor_cols:
        standardized_df[col] = standardized_df.groupby('Date')[col].transform(
            lambda x: zscore(x, nan_policy='omit')
        )

    # 4. 保存结果到Excel ( 一个文件, 两个sheet)
    with pd.ExcelWriter('Day8-2_factors_and_standardized.xlsx') as writer:
        all_factors_df.to_excel(writer, sheet_name='Raw_Factors', index=False)   # 原始因子
        standardized_df.to_excel(writer, sheet_name='Standardized_Factors', index=False)    # 标准化因子

    print("\n 多因子原始数据和标准化数据已保存到 dAY8-2_factors_and_standardized.xlsx 文件中.")


''' 可以下载Fitten Code Chat. 这是Pycharm 扩展.  是AI的. 可以知道你下面要写什么代码.  挺好用. '''










