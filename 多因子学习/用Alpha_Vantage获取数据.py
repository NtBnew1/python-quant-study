'''
用于批量获取多个美股公司的财务报表数据（利润表 / 资产负债表 / 现金流量表），并自动保存为CSV文件，方便后续量化分析或因子构建使用。

✔ 使用 Alpha Vantage 接口
✔ 支持多个股票批量处理
✔ 支持选择报表类型（季度）
✔ 输出标准CSV格式，直接可用于pandas分析

 api_key = 'LNCEEYGQUGYZCRGO'
 api_key = 'Q9K528229K528229'
 api_key = 'YSLMR9OGQ6SBC9UM'


💡 适合初学者理解财报因子抓取流程，作为多因子选股策略的数据准备部分。
'''
from fileinput import filename

'''  要用多因子需要几个报表, 现在就是要用alpha vantage获取股价数据和报表. '''

# 导入库
import requests # 这是用于爬取网址数据, 或者说发送HTTP请求
import pandas as pd
import time     # 用于延迟时间, 防止API限流
from datetime import datetime   # 获取当前时间

def fetch_daily_price(symbol, api_key):
    # 构造ALpha Vantage 的API请求URL (TIME_SERIES_DAILY函数)
    url = (f'http://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
           f'&symbol={symbol}&outputsize=full&apikey={api_key}')

    # 发送HTTP请求
    response = requests.get(url)
    data = response.json()  # 将后回的数据转换为json格式

    # 检查数据是否包含每日价格数据
    if "Time Series (Daily)" not in data:
        print(f"[错误]无法获取{symbol}的价格数据, ")
        print(data)
        return pd.DataFrame()   # 后回空DataFrame

    # 提取每日数据字典
    ts = data['Time Series (Daily)']

    # 将字典转换为DataFrame, 日期作为行索引
    df = pd.DataFrame.from_dict(ts, orient='index')
    df.index = pd.to_datetime(df.index) # 转换日期格式
    df.sort_index(inplace=True) # 按时间排序

    # 重命名列名,
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    # ALpha vantage获取数据都是全部的.  现在只要1年的股价数据
    two_year_ago = datetime.now() - pd.DateOffset(years=2)
    df = df[df.index >= two_year_ago]
    return df

# 获取财务报表 (收入, 资产负债, 现金流) 数据
def fetch_financial_report(symbol, report_type, api_key):
    # 再构造alpha vantage 请求URL, report_type 可以是INCOME_STATEMENT等
    url = f'https://www.alphavantage.co/query?function={report_type}&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()

    # 尝试获取季度或者年度报表
    if 'quarterlyReports' in data:
        reports = data['quarterlyReports']
    elif 'annualReports' in data:
        reports = data['annualReports']
    else:
        print(f"[错误]{symbol} 获取{report_type}失败, ")
        print(data)
        return pd.DataFrame()

    # 将报表数据转换DataFrame
    df = pd.DataFrame(reports)

    # 将"fiscalDateEnding"列放到最前面
    if 'fiscalDateEnding' in df.columns:
        cols = ['fiscalDateEnding'] + [c for c in df.columns if c != 'fiscalDateEnding']
        df = df[cols]
    return df

# 现在要全部保存到excel里
def save_all_to_excel(symbol, api_key, filename):
    print(f'开始获取{symbol}的数据.....')

    # 获取股价数据
    price_df = fetch_daily_price(symbol, api_key)
    time.sleep(12)  # 加延迟, 防止API限流

    # 获取收入报表
    income_df = fetch_financial_report(symbol, 'INCOME_STATEMENT', api_key)
    time.sleep(12)

    # 获取资产负债报表
    balance_df = fetch_financial_report(symbol, 'BALANCE_SHEET', api_key)
    time.sleep(12)

    # 获取现金流报表
    cashflow_df = fetch_financial_report(symbol, 'CASH_FLOW', api_key)
    time.sleep(12)

    # 使用pandas 的ExcelWriter将多个DataFrame写入一个Excel文件里, 不同sheet
    with pd.ExcelWriter(filename) as writer:
        if not price_df.empty:
            price_df.to_excel(writer, sheet_name='Two_Year_Stock')
        if not income_df.empty:
            income_df.to_excel(writer, sheet_name='Income_Statement')
        if not balance_df.empty:
            balance_df.to_excel(writer, sheet_name='Balance_Sheet')
        if not cashflow_df.empty:
            cashflow_df.to_excel(writer, sheet_name='Cash_Flow')

    print(f" 所有数据已保存到{filename}")

# 主函数入口
if __name__ == "__main__":
    api_key = 'LNCEEYGQUGYZCRGO'        # 直接去alpha vantage官网可以得到免费api. 这个是个人的.
    # api_key = 'YSLMR9OGQ6SBC9UM'
    symbol = input(f"请输入股票代码: ").strip().upper()    # 用户输入股票代码并转大写
    filename = (f"./{symbol}_all_data.xlsx")    # 输出文件名格式
    save_all_to_excel(symbol, api_key, filename)        # 调用这个函数



