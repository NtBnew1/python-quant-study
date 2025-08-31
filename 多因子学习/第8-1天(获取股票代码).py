'''
第8天：

学习因子稳定性检测（滑动窗口IC）
分析因子时序表现，剔除不稳定因子
练习：实现滑动窗口IC计算代码

'''

'''
现在我们来学习一下因子的稳定性检测。  新的一天, 有新的麻烦. 
IC 是可以计算预期下一期收益.   需要用多因子和股价. 时间需要3-5年.  我们之前只有两年.  现在需要改代码
然后重新获取公司的数据.   有可能需要3-天时间来获取20-30家公司数据.   alpha_vantage每天有上限. 


重新学习获取数据.  这次是获取全部的数据.
'''

import pandas as pd
import requests
import time
from datetime import datetime
import os


# 获取数据
def fetch_daily_price(symbol, api_key):
    url = (f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
           f"&symbol={symbol}&outputsize=full&apikey={api_key}")
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        print(f"[错误] 无法获取{symbol}, 无数据")
        print(data)
        return pd.DataFrame()

    ts = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df

def fetch_financial_report(symbol, report_type, api_key):
    url = f"https://www.alphavantage.co/query?function={report_type}&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if 'quarterlyReports' in data:
        reports = data['quarterlyReports']

    elif 'annualReports' in data:
        reports = data['annualReports']

    else:
        print(f"[错误]获取{symbol}的{report_type}报表失败")
        print(data)
        return pd.DataFrame()

    df = pd.DataFrame(reports)
    if 'fiscalDateEnding' in df.columns:
        cols = ['fiscalDateEnding'] + [c for c in df.columns if c != 'fiscalDateEnding']
        df = df[cols]
    return df

# 保存数据到excel
def save_to_excel(symbol, api_key, filename):
    print(f'开始获取{symbol}的数据...')

    price_df = fetch_daily_price(symbol, api_key)
    time.sleep(12)

    income_df = fetch_financial_report(symbol, 'INCOME_STATEMENT', api_key)
    time.sleep(12)

    balance_df = fetch_financial_report(symbol, 'BALANCE_SHEET', api_key)
    time.sleep(12)

    cashflow_df = fetch_financial_report(symbol, 'CASH_FLOW', api_key)
    time.sleep(12)

    with pd.ExcelWriter(filename) as writer:
        if not price_df.empty:
            price_df.to_excel(writer, sheet_name='price')

        if not income_df.empty:
            income_df.to_excel(writer, sheet_name='Income_Statement')

        if not balance_df.empty:
            balance_df.to_excel(writer, sheet_name='Balance_Sheet')

        if not cashflow_df.empty:
            cashflow_df.to_excel(writer, sheet_name='Cash_Flow')
    print(f"{symbol}所有数据已保存到{filename}")

# 调用主程序
if __name__ == "__main__":
    api_key = 'LNCEEYGQUGYZCRGO'     # 自己申请的api_key
    symbols = input("请输入股票代码 (如aapl): ").strip().upper()
    filename = f"./{symbols}_all_data.xlsx"     # 还是保存这样的格式.  这样就可以替换之前的数据
    save_to_excel(symbols, api_key, filename)

'''

第一部分: 获取数据, 

第二部分: 计算多因子

第三部分: 合并所有公司的因子数据

第四部分: 生成未来收益率, 尝试因子预测目标

第五部分: 使用滑动窗口IC计算因子的稳定性

'''



